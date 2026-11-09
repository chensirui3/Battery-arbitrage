"""ERCOT real-time grid conditions, each value stored with the time it was published.

A decision sees settlement prices 25 minutes late: an interval's price posts 5 minutes after it
ends. ERCOT publishes faster signals, kept here:

- ``sced``: every run of real-time dispatch (SCED, every 5 minutes, posted within seconds),
  from "Real-Time ORDC and Reliability Deployment Price Adders and Reserves by SCED Interval"
  (NP6-323-CD). Kept: the system-wide energy price (system lambda plus the real-time price
  adders) and Physical Responsive Capability (PRC, MW), ERCOT's measure of real-time reserves.
- ``rtd``: every run of ERCOT's real-time look-ahead (RTD, every 5 minutes), from NP6-325-CD:
  the indicative system-wide energy price for each of the next eleven 5-minute intervals.
- ``outages``: Hourly Resource Outage Capacity (NP3-233-CD), published every hour: MW of
  generation out of service in each hour of the coming week, wind and solar reported apart.

ERCOT's real-time market redesign (RTC+B, December 5, 2025) changed the two 5-minute reports.
The reserve adders and PRC are gone from them, and the price adder is the reliability
deployment adder alone. PRC is missing (NaN) from then on.

The archive allows 50 files per download request for the 5-minute reports. So only the runs a
decision would use are kept: for each decision time (every 15 minutes, 5 minutes before an
interval starts), the latest run posted by then. That is a third of all runs. A run the archive
lists but refuses to serve (HTTP 400; one RTD run, posted 2026-05-05 23:35) is replaced by the
latest run posted before it, which was public sooner.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from batt.data.fundamentals import ForecastVintages, _client, _periods, _within_lead
from batt.data.store import ts_ns
from batt.forecast.history import MINUTE_NS

TIMEZONE = "US/Central"  # ERCOT's archive and report timestamps are local time
REPORT_IDS = {"sced": "np6-323-cd", "rtd": "np6-325-cd", "outages": "np3-233-cd"}
MAX_DOCS = {"sced": 50, "rtd": 50, "outages": 1000}  # files per download request
SECONDS_BETWEEN_CALLS = 2.0  # stay under the archive's rate limit
TOKEN_MARGIN_SECONDS = 300.0  # sign in again this long before the token's hour is up
ATTEMPTS = 3  # tries per report and month; what still fails is left for the next run
RETRY_SECONDS = 30.0  # wait before the second try, twice that before the third
MAX_REFUSED = 5  # files per report and month the archive may refuse before the month fails
RTD_STEPS = 12  # look-ahead intervals kept per RTD run (ERCOT publishes 11)
OUTAGE_SERIES = {"outage_resource": "Total Resource MW", "outage_irr": "Total IRR MW"}
SCED_COLUMNS = ["publish_time_utc", "time_utc", "price", "prc"]
RTD_COLUMNS = ["publish_time_utc", "run_time_utc", "interval_end_utc", "price"]
OUTAGE_COLUMNS = ["series", "publish_time_utc", "interval_start_utc", "value"]


# --- Point-in-time storage -----------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class Snapshots:
    """SCED runs sorted by publish time: when each ran, its price ($/MWh), and PRC (MW)."""

    publish_ns: np.ndarray
    time_ns: np.ndarray
    price: np.ndarray
    prc: np.ndarray

    @classmethod
    def from_frame(cls, frame: pd.DataFrame) -> Snapshots:
        df = frame.sort_values(["publish_time_utc", "time_utc"])
        return cls(
            _ns(df["publish_time_utc"]),
            _ns(df["time_utc"]),
            df["price"].to_numpy(dtype=float),
            df["prc"].to_numpy(dtype=float),
        )

    def latest(self, issue_ns) -> np.ndarray:
        """Index of the latest run published at or before each issue time (-1 if none)."""
        return np.searchsorted(self.publish_ns, np.asarray(issue_ns, dtype=np.int64), "right") - 1


@dataclass(frozen=True, eq=False)
class LookAhead:
    """RTD runs sorted by publish time. ``end_ns[r, j]`` is the end of run r's j-th 5-minute
    look-ahead interval and ``price[r, j]`` its indicative price (padded with NaN)."""

    publish_ns: np.ndarray
    end_ns: np.ndarray
    price: np.ndarray

    @classmethod
    def from_frame(cls, frame: pd.DataFrame, steps: int = RTD_STEPS) -> LookAhead:
        df = frame.sort_values(["run_time_utc", "interval_end_utc"])
        runs, first, row = np.unique(
            _ns(df["run_time_utc"]), return_index=True, return_inverse=True
        )
        publish = np.full(runs.size, np.iinfo(np.int64).min, dtype=np.int64)
        np.maximum.at(publish, row, _ns(df["publish_time_utc"]))
        position = np.arange(len(df)) - first[row]  # look-ahead step within the run
        keep = position < steps
        end = np.full((runs.size, steps), np.iinfo(np.int64).min, dtype=np.int64)
        price = np.full((runs.size, steps), np.nan)
        end[row[keep], position[keep]] = _ns(df["interval_end_utc"])[keep]
        price[row[keep], position[keep]] = df["price"].to_numpy(dtype=float)[keep]
        order = np.argsort(publish, kind="stable")
        return cls(publish[order], end[order], price[order])

    def mean_price(self, issue_ns, start_ns, end_ns, step_ns: int = 5 * MINUTE_NS) -> np.ndarray:
        """Average indicative price over [start, end) from the latest run published at or before
        each issue time, or NaN unless that run covers every 5-minute interval in it."""
        issue = np.asarray(issue_ns, dtype=np.int64)
        start = np.asarray(start_ns, dtype=np.int64)
        end = np.asarray(end_ns, dtype=np.int64)
        issue, start, end = np.broadcast_arrays(issue, start, end)
        out = np.full(issue.shape, np.nan)
        if self.publish_ns.size == 0:
            return out
        run = np.searchsorted(self.publish_ns, issue, side="right") - 1
        ok = run >= 0
        r = np.maximum(run, 0)
        ends = self.end_ns[r]  # (..., steps)
        inside = (ends > start[..., None]) & (ends <= end[..., None]) & ok[..., None]
        prices = np.where(inside, self.price[r], 0.0)
        count = inside.sum(axis=-1)
        needed = (end - start) // step_ns
        full = ok & (count == needed) & (needed > 0)
        out[full] = prices.sum(axis=-1)[full] / count[full]
        return out


@dataclass(frozen=True, eq=False)
class GridConditions:
    sced: Snapshots
    rtd: LookAhead
    outages: ForecastVintages  # MW of generation out (wind and solar excluded)
    frames: dict[str, pd.DataFrame] | None = None  # the rows everything was built from, if kept

    @classmethod
    def from_frames(
        cls, sced: pd.DataFrame, rtd: pd.DataFrame, outages: pd.DataFrame, keep: bool = True
    ) -> GridConditions:
        """Build the lookup arrays. ``keep=False`` drops the rows afterwards, as the backtest
        copies the data into every process."""
        resource = outages[outages["series"] == "outage_resource"]
        frames = {"sced": sced, "rtd": rtd, "outages": outages} if keep else None
        return cls(
            Snapshots.from_frame(sced),
            LookAhead.from_frame(rtd),
            ForecastVintages.from_frame(resource),
            frames,
        )

    def view(self, now) -> ConditionsView:
        return ConditionsView(self, None if now is None else ts_ns(now))

    def coverage(self) -> tuple[pd.Timestamp, pd.Timestamp] | None:
        """From the latest first publication to the earliest last one, across the reports."""
        times = [self.sced.publish_ns, self.rtd.publish_ns, self.outages.publish_ns]
        if any(t.size == 0 for t in times):
            return None
        first, last = max(t[0] for t in times), min(t[-1] for t in times)
        return pd.Timestamp(first, tz="UTC"), pd.Timestamp(last, tz="UTC")

    def empty_months(self, start, end) -> list[str]:
        """Each local calendar month overlapping [start, end) in which a report has no
        publication at all, as "report YYYY-MM": a download that failed or was never made."""
        first, last = _local(start), _local(end)
        if last <= first:
            return []
        months = pd.date_range(
            first.normalize().replace(day=1), last - pd.Timedelta(1, "ns"), freq="MS"
        )
        bounds = _ns(months.append(pd.DatetimeIndex([months[-1] + pd.offsets.MonthBegin()])))
        found = {"sced": self.sced.publish_ns, "rtd": self.rtd.publish_ns}
        found["outages"] = self.outages.publish_ns
        gaps = []
        for report, times in found.items():
            counts = np.diff(np.searchsorted(times, bounds))
            gaps += [f"{report} {m:%Y-%m}" for m, n in zip(months, counts, strict=True) if n == 0]
        return gaps


class ConditionsView:
    """Grid conditions as public at ``now`` (UTC ns; None for the oracle view).

    Every query takes explicit issue times, which may not be later than ``now``.
    """

    def __init__(self, data: GridConditions, now_ns: int | None):
        self._data = data
        self.now_ns = now_ns

    def _check(self, issue_ns) -> np.ndarray:
        issue = np.asarray(issue_ns, dtype=np.int64)
        if self.now_ns is not None and issue.size and issue.max() > self.now_ns:
            raise ValueError("grid conditions queried for a time after the view's moment")
        return issue

    def sced(self, issue_ns) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Time (ns, -1 if none), price, and PRC of the latest SCED run public at each issue."""
        s = self._data.sced
        i = s.latest(self._check(issue_ns))
        ok = i >= 0
        j = np.maximum(i, 0)
        if s.publish_ns.size == 0:
            nan = np.full(i.shape, np.nan)
            return np.full(i.shape, -1, dtype=np.int64), nan, nan
        return (
            np.where(ok, s.time_ns[j], -1),
            np.where(ok, s.price[j], np.nan),
            np.where(ok, s.prc[j], np.nan),
        )

    def rtd_price(self, issue_ns, start_ns, end_ns) -> np.ndarray:
        return self._data.rtd.mean_price(self._check(issue_ns), start_ns, end_ns)

    def outage(self, issue_ns, target_ns) -> np.ndarray:
        return self._data.outages.asof(self._check(issue_ns), target_ns)


def _ns(values) -> np.ndarray:
    return pd.DatetimeIndex(values).tz_convert("UTC").as_unit("ns").asi8


# --- From ERCOT reports to the stored schema ----------------------------------------------


def _event_utc(stamps: pd.Series, repeated: pd.Series) -> pd.Series:
    """Local report timestamps to UTC. ``repeated`` is ERCOT's flag for the second pass through
    the hour repeated when daylight saving time ends."""
    local = pd.to_datetime(stamps, format="mixed")
    first_pass = repeated.astype(str).str.upper().ne("Y").to_numpy()
    utc = pd.DatetimeIndex(local).tz_localize(
        TIMEZONE, ambiguous=first_pass, nonexistent="shift_forward"
    )
    return pd.Series(utc.tz_convert("UTC").as_unit("ns"), index=stamps.index)


def _publish_utc(posted: pd.Series, event: pd.Series) -> pd.Series:
    """Archive post times (local, no flag) to UTC: in the repeated hour, the earliest reading
    that is not before the event the file reports on."""
    local = pd.DatetimeIndex(pd.to_datetime(posted, format="mixed"))
    early = local.tz_localize(
        TIMEZONE, ambiguous=np.ones(len(local), bool), nonexistent="shift_forward"
    )
    late = local.tz_localize(
        TIMEZONE, ambiguous=np.zeros(len(local), bool), nonexistent="shift_forward"
    )
    early = pd.Series(early.tz_convert("UTC").as_unit("ns"), index=posted.index)
    late = pd.Series(late.tz_convert("UTC").as_unit("ns"), index=posted.index)
    return early.where(early >= event, late)


def _price(df: pd.DataFrame) -> pd.Series:
    """System lambda plus the real-time price adders that apply to energy: the online reserve
    and reliability deployment adders before RTC+B, the reliability deployment adder after."""
    adders = [c for c in ("RTORPA", "RTORDPA", "RTRDPA") if c in df]
    total = df["SystemLambda"].astype(float)
    for column in adders:
        total = total + df[column].astype(float).fillna(0.0)
    return total


def reduce_sced(df: pd.DataFrame) -> pd.DataFrame:
    """NP6-323-CD rows (one per SCED run, with ``postDatetime``) -> ``SCED_COLUMNS``."""
    stamp = df["SCEDTimestamp"] if "SCEDTimestamp" in df else df["SCEDTimeStamp"]
    event = _event_utc(stamp, df["RepeatedHourFlag"])
    prc = df["PRC"].astype(float) if "PRC" in df else pd.Series(np.nan, index=df.index)
    out = pd.DataFrame(
        {
            "publish_time_utc": _publish_utc(df["postDatetime"], event),
            "time_utc": event,
            "price": _price(df),
            "prc": prc,
        }
    )
    return out.drop_duplicates("time_utc").reset_index(drop=True)


def reduce_rtd(df: pd.DataFrame) -> pd.DataFrame:
    """NP6-325-CD rows (one per RTD run and look-ahead interval) -> ``RTD_COLUMNS``."""
    run = _event_utc(df["RTDTimestamp"], df["RepeatedHourFlag"])
    flag = df["IERepeatedHourFlag"] if "IERepeatedHourFlag" in df else df["RepeatedHourFlag"]
    out = pd.DataFrame(
        {
            "publish_time_utc": _publish_utc(df["postDatetime"], run),
            "run_time_utc": run,
            "interval_end_utc": _event_utc(df["IntervalEnding"], flag),
            "price": _price(df),
        }
    )
    return out.drop_duplicates(["run_time_utc", "interval_end_utc"]).reset_index(drop=True)


def reduce_outages(df: pd.DataFrame) -> pd.DataFrame:
    """Hourly Resource Outage Capacity (gridstatus columns) -> ``OUTAGE_COLUMNS`` rows.

    ERCOT added zonal columns to the report on 2021-05-27. In a download spanning that day,
    gridstatus sums the zones into the total for every row, so rows in the old format have no
    total there; theirs is in the report's own column (``TotalResourceMW``), used instead.
    """
    frames = []
    for series, column in OUTAGE_SERIES.items():
        value = pd.to_numeric(df[column], errors="coerce")
        old = column.replace(" ", "")
        if old in df:
            value = value.fillna(pd.to_numeric(df[old], errors="coerce"))
        frames.append(
            pd.DataFrame(
                {
                    "series": series,
                    "publish_time_utc": pd.DatetimeIndex(df["Publish Time"]).tz_convert("UTC"),
                    "interval_start_utc": pd.DatetimeIndex(df["Interval Start"]).tz_convert("UTC"),
                    "value": value.to_numpy(dtype=float),
                }
            )
        )
    rows = pd.concat(frames, ignore_index=True)
    rows["publish_time_utc"] = rows["publish_time_utc"].dt.as_unit("ns")
    rows["interval_start_utc"] = rows["interval_start_utc"].dt.as_unit("ns")
    rows = rows.drop_duplicates(["series", "publish_time_utc", "interval_start_utc"])
    return _within_lead(rows).reset_index(drop=True)


# --- Download, store, load ----------------------------------------------------------------


def conditions_dir(data_dir) -> Path:
    return Path(data_dir) / "conditions"


def decision_times(start, end, step_minutes: int = 15, lead_minutes: int = 5) -> np.ndarray:
    """UTC ns of the decisions in [start, end): ``lead`` minutes before each interval start."""
    step, lead = int(step_minutes * MINUTE_NS), int(round(lead_minutes * MINUTE_NS))
    lo, hi = ts_ns(start), ts_ns(end)
    first = -(-(lo + lead) // step) * step - lead
    return np.arange(first, hi, step, dtype=np.int64)


def select_runs(post_ns: np.ndarray, decisions_ns: np.ndarray) -> np.ndarray:
    """Indices of the latest post at or before each decision (each index once)."""
    order = np.argsort(post_ns, kind="stable")
    i = np.searchsorted(post_ns[order], decisions_ns, side="right") - 1
    return np.unique(order[i[i >= 0]])


def _listing_utc(posted: list[str]) -> np.ndarray:
    """Archive post times, in the order ERCOT posted them, to UTC ns. If the repeated hour
    cannot be resolved, it is read as the later hour, which never makes a file look earlier."""
    local = pd.DatetimeIndex(pd.to_datetime(pd.Series(posted), format="mixed"))
    try:
        utc = local.tz_localize(TIMEZONE, ambiguous="infer", nonexistent="shift_forward")
    except Exception:
        utc = local.tz_localize(
            TIMEZONE, ambiguous=np.zeros(len(local), bool), nonexistent="shift_forward"
        )
    return utc.tz_convert("UTC").as_unit("ns").asi8


def _throttle(client, seconds: float) -> None:
    """Space out the client's requests by at least ``seconds``."""
    call = client.make_api_call
    last = [0.0]

    def throttled(*args, **kwargs):
        wait = last[0] + seconds - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        try:
            return call(*args, **kwargs)
        finally:
            last[0] = time.monotonic()

    client.make_api_call = throttled


def _refresh_early(client, margin: float = TOKEN_MARGIN_SECONDS) -> None:
    """Sign in again ``margin`` seconds before the token's hour is up. gridstatus starts the
    hour when the token arrives, after ERCOT has started it, so a request in the last moment
    went out with an expired token (HTTP 401) and cost a whole month's download."""
    get_token = client.get_token

    def early(*args, **kwargs):
        result = get_token(*args, **kwargs)
        if client.token_expiry is not None:
            client.token_expiry -= margin
        return result

    client.get_token = early


def _retrying(
    fetch: Callable,
    client,
    *,
    attempts: int = ATTEMPTS,
    wait: float = RETRY_SECONDS,
    log: Callable[[str], None] = print,
    label: str = "",
):
    """``fetch()``, tried up to ``attempts`` times, signing in again and waiting longer before
    each new try. A LookupError (nothing was posted) is final."""
    for attempt in range(1, attempts + 1):
        try:
            return fetch()
        except LookupError:
            raise
        except Exception as exc:
            if attempt == attempts:
                raise
            log(f"  {label}: try {attempt} failed ({type(exc).__name__}: {exc}); retrying")
            client.token = None  # sign in again before the next request
            time.sleep(wait * attempt)


def _with_allowed_batch(client, batch: int, request: Callable):
    """Run ``request`` with ``batch`` files per download. If the archive answers that the batch
    is too big, retry with the size it allows."""
    import requests

    while True:
        client.batch_size = batch
        try:
            return request()
        except requests.HTTPError as exc:
            text = exc.response.text if exc.response is not None else ""
            allowed = re.search(r"maximum number allowed .*\((\d+)\)", text)
            if not allowed or int(allowed.group(1)) >= batch:
                raise
            batch = int(allowed.group(1))


def _refused(exc: BaseException) -> bool:
    """Whether the archive answered HTTP 400 for a reason other than the batch size."""
    response = getattr(exc, "response", None)
    if response is None or response.status_code != 400:
        return False
    return "maximum number allowed" not in response.text


def _download(client, ids: list[str], emil: str, limit: int = MAX_REFUSED):
    """Download the documents ``ids``, ``client.batch_size`` per request. A batch the archive
    answers with HTTP 400 is fetched again one file at a time, to find the files it refuses.
    Returns the files served, by id, and the ids refused. More than ``limit`` refused raises:
    then the requests are at fault, not a few files."""
    served, refused = {}, []
    size = client.batch_size
    for first in range(0, len(ids), size):
        batch = ids[first : first + size]
        try:
            docs = client._bulk_download_documents(batch, emil)
        except Exception as exc:
            if not _refused(exc):
                raise
        else:
            served.update(zip(batch, (data for data, _ in docs), strict=True))
            continue
        for doc in batch:
            try:
                [(data, _)] = client._bulk_download_documents([doc], emil)
            except Exception as exc:
                if not _refused(exc):
                    raise
                refused.append(doc)
                if len(refused) > limit:
                    raise RuntimeError(
                        f"the archive refuses more than {limit} {emil} files"
                    ) from exc
            else:
                served[doc] = data
    return served, refused


def _fetch_runs(
    client, report: str, lo, hi, step_minutes: int, lead_minutes: int, log=print
) -> tuple[pd.DataFrame, int]:
    """The runs the decisions in [lo, hi) would use: for each, the latest run posted by then
    that the archive serves."""
    emil = REPORT_IDS[report]
    links = client._get_historical_data_links(emil, lo, hi)
    if not links:
        raise LookupError(f"no {emil} files posted from {lo} to {hi}")
    ids = np.array([int(link.split("=")[-1]) for link, _ in links])
    order = np.argsort(ids, kind="stable")  # posting order
    ids = ids[order]
    posted = [links[i][1] for i in order]
    post_ns = _listing_utc(posted)
    decisions = decision_times(lo, hi, step_minutes, lead_minutes)
    usable = np.ones(ids.size, dtype=bool)
    files = {}
    while True:
        candidates = np.flatnonzero(usable)
        chosen = candidates[select_runs(post_ns[candidates], decisions)]
        wanted = [str(i) for i in ids[chosen] if str(i) not in files]
        left = MAX_REFUSED - int((~usable).sum())
        served, refused = _with_allowed_batch(
            client,
            MAX_DOCS[report],
            lambda wanted=wanted, left=left: _download(client, wanted, emil, left),
        )
        files.update(served)
        if not refused:
            break
        for doc in refused:
            i = int(np.flatnonzero(ids == int(doc))[0])
            usable[i] = False
            log(
                f"  {report} {lo:%Y-%m-%d}: the archive refuses file {doc}, posted {posted[i]}; "
                "its decision uses the run posted before it"
            )
    frames = []
    for i in chosen:
        frame = pd.read_csv(files[str(ids[i])], compression="zip")
        frame["postDatetime"] = posted[i]
        frames.append(frame)
    raw = pd.concat(frames, ignore_index=True)
    rows = reduce_sced(raw) if report == "sced" else reduce_rtd(raw)
    return rows, chosen.size


def _fetch_outages(client, lo, hi):
    raw = _with_allowed_batch(
        client,
        MAX_DOCS["outages"],
        lambda: client.get_hourly_resource_outage_capacity(date=lo, end=hi),
    )
    rows = reduce_outages(raw)
    return rows, rows["publish_time_utc"].nunique()


REPORTS = tuple(REPORT_IDS)


def ingest(
    data_dir,
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    reports: Iterable[str] | None = None,
    force: bool = False,
    step_minutes: int = 15,
    lead_minutes: int = 5,
    log: Callable[[str], None] = print,
) -> list[Path]:
    """Download what was posted in [start, end) from the ERCOT Public API archive (needs the
    same free account as ``batt.data.fundamentals``), one calendar month at a time.

    Months already stored are skipped unless ``force``; the month in progress is refreshed.
    Reports are fetched month by month together, so history fills in from the start for all of
    them. A month that still fails after ``ATTEMPTS`` tries is skipped and named at the end;
    running again fetches it. Returns the files written.
    """
    chosen = list(REPORTS) if reports is None else list(reports)
    unknown = sorted(set(chosen) - set(REPORTS))
    if unknown:
        raise ValueError(f"unknown reports {unknown}; choose from {list(REPORTS)}")
    client = _client("api")
    _throttle(client, SECONDS_BETWEEN_CALLS)
    _refresh_early(client)
    out = conditions_dir(data_dir)
    out.mkdir(parents=True, exist_ok=True)
    start, end = _local(start), _local(end)
    now = pd.Timestamp.now(tz=TIMEZONE)
    written, failed = [], []
    for period in _periods(start, end, monthly=True):
        lo, hi = max(period[0], start), min(period[1], end, now)
        if hi <= lo:
            continue
        in_progress = period[1] > now
        label_end = period[1] if in_progress else hi
        for report in chosen:
            path = out / f"{report}_{lo:%Y-%m-%d}_{label_end:%Y-%m-%d}.parquet"
            if path.exists() and not force and not in_progress:
                continue
            began = time.perf_counter()
            label = f"{report} {lo:%Y-%m-%d}"

            def fetch(report=report, lo=lo, hi=hi):
                if report == "outages":
                    return _fetch_outages(client, lo, hi)
                return _fetch_runs(client, report, lo, hi, step_minutes, lead_minutes, log=log)

            try:
                rows, runs = _retrying(fetch, client, log=log, label=label)
            except LookupError as exc:
                log(f"  {label}: nothing posted ({exc})")
                failed.append(label)
                continue
            except Exception as exc:
                log(f"  {label}: FAILED after {ATTEMPTS} tries ({type(exc).__name__}: {exc})")
                failed.append(label)
                continue
            partial = path.with_name(path.name + ".part")
            rows.to_parquet(partial, index=False)
            partial.replace(path)
            written.append(path)
            log(
                f"  {report} {lo:%Y-%m-%d} -> {hi:%Y-%m-%d}: {runs} publications, "
                f"{len(rows):,} rows ({time.perf_counter() - began:.0f}s)"
            )
    if failed:
        log(f"not stored: {', '.join(failed)}. Run the same command again to fetch them.")
    return written


def _local(ts) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    return ts.tz_localize(TIMEZONE) if ts.tzinfo is None else ts.tz_convert(TIMEZONE)


def load_conditions(data_dir) -> GridConditions | None:
    """Everything stored under ``data/conditions`` (None unless all three reports are there)."""
    folder = conditions_dir(data_dir)
    parts = {}
    for report in REPORTS:
        files = sorted(folder.glob(f"{report}_*.parquet"))
        if not files:
            return None
        parts[report] = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    sced = parts["sced"].drop_duplicates("time_utc")
    rtd = parts["rtd"].drop_duplicates(["run_time_utc", "interval_end_utc"])
    outages = parts["outages"].drop_duplicates(["series", "publish_time_utc", "interval_start_utc"])
    return GridConditions.from_frames(
        sced[SCED_COLUMNS], rtd[RTD_COLUMNS], outages[OUTAGE_COLUMNS], keep=False
    )
