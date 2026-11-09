"""ERCOT fundamentals: load, wind, and solar forecasts as they stood when each was published.

Prices only say where the market already thinks prices will be. What moves real-time prices
away from day-ahead prices is new information after the DA market closes: the load turning out
higher than forecast, the wind dropping off, clouds over West Texas. ERCOT publishes its own
hourly-updated forecasts of all three, and each publication is kept here as a separate
*vintage*, so a decision can be given exactly the forecast that was public at its moment.

Series (all MW, system-wide, hourly):

- ``load_forecast``: Seven-Day Load Forecast (NP3-565-CD), system total of the model ERCOT
  had in use for that publication. Published every hour at about :30.
- ``wind_forecast`` / ``solar_forecast``: STWPF / STPPF from the hourly wind and solar
  production reports (NP4-732-CD, NP4-737-CD), published every hour at about :55.
- ``wind_actual`` / ``solar_actual``: actual output from the same reports. An hour's actual
  first appears about 55 minutes after the hour ends; only that first publication is kept.

History comes from the ERCOT Public API archive, which needs a free account (see the README).
The last few days are also on ERCOT's public website without an account (``source="mis"``).
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from batt.data.store import ts_ns
from batt.forecast.history import HOUR_NS

SERIES = ("load_forecast", "wind_forecast", "solar_forecast", "wind_actual", "solar_actual")
FORECASTS = ("load_forecast", "wind_forecast", "solar_forecast")
ACTUALS = ("wind_actual", "solar_actual")
COLUMNS = ["series", "publish_time_utc", "interval_start_utc", "value"]
MAX_LEAD_HOURS = 48  # hours after a publication's hour kept from each forecast
READ_TIMEOUT_SECONDS = 120  # for ERCOT API downloads (see _client)
CREDENTIALS = ("ERCOT_API_USERNAME", "ERCOT_API_PASSWORD", "ERCOT_PUBLIC_API_SUBSCRIPTION_KEY")


# --- Point-in-time storage -----------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class ForecastVintages:
    """Hourly forecasts published repeatedly.

    ``values[p, h]`` is publication ``p``'s forecast for the hour starting ``h`` hours after
    the hour it was published in (lead 0 is the hour in progress). Missing values are NaN.
    """

    publish_ns: np.ndarray  # (publications,) sorted
    base_ns: np.ndarray  # (publications,) start of the hour each publication was made in
    values: np.ndarray  # (publications, MAX_LEAD_HOURS)

    @classmethod
    def from_frame(cls, frame: pd.DataFrame, max_lead: int = MAX_LEAD_HOURS) -> ForecastVintages:
        """From rows of (publish_time_utc, interval_start_utc, value)."""
        publish = _ns(frame["publish_time_utc"])
        target = _ns(frame["interval_start_utc"]) // HOUR_NS * HOUR_NS
        times, row = np.unique(publish, return_inverse=True)
        base = times // HOUR_NS * HOUR_NS
        lead = (target - base[row]) // HOUR_NS
        keep = (lead >= 0) & (lead < max_lead)
        values = np.full((times.size, max_lead), np.nan)
        values[row[keep], lead[keep]] = frame["value"].to_numpy(dtype=float)[keep]
        return cls(times, base, values)

    def asof(self, issue_ns, target_ns, max_back: int = 3) -> np.ndarray:
        """Latest forecast for each target's hour published at or before each issue time.

        Falls back up to ``max_back`` earlier publications where the latest one lacks a value.
        """
        issue = np.asarray(issue_ns, dtype=np.int64)
        hour = np.asarray(target_ns, dtype=np.int64) // HOUR_NS * HOUR_NS
        issue, hour = np.broadcast_arrays(issue, hour)
        out = np.full(hour.shape, np.nan)
        if self.publish_ns.size == 0:
            return out
        latest = np.searchsorted(self.publish_ns, issue, side="right") - 1
        n_leads = self.values.shape[1]
        for back in range(max_back + 1):
            p = latest - back
            lead = (hour - self.base_ns[np.maximum(p, 0)]) // HOUR_NS
            ok = np.isnan(out) & (p >= 0) & (lead >= 0) & (lead < n_leads)
            out[ok] = self.values[p[ok], lead[ok]]
        return out


@dataclass(frozen=True, eq=False)
class ActualSeries:
    """Hourly actuals with the time each first became public, sorted by hour."""

    start_ns: np.ndarray
    publish_ns: np.ndarray
    values: np.ndarray

    @classmethod
    def from_frame(cls, frame: pd.DataFrame) -> ActualSeries:
        df = frame.sort_values("publish_time_utc").drop_duplicates("interval_start_utc")
        df = df.sort_values("interval_start_utc")
        start, publish = _ns(df["interval_start_utc"]), _ns(df["publish_time_utc"])
        # Point-in-time lookups need actuals to become public in hour order.
        publish = np.maximum.accumulate(publish) if publish.size else publish
        return cls(start, publish, df["value"].to_numpy(dtype=float))

    def latest(self, issue_ns) -> tuple[np.ndarray, np.ndarray]:
        """Hour start and value of the latest actual public at each issue time (NaN if none)."""
        issue = np.asarray(issue_ns, dtype=np.int64)
        i = np.searchsorted(self.publish_ns, issue, side="right") - 1
        ok = i >= 0
        j = np.maximum(i, 0)
        start = np.where(ok, self.start_ns[j] if self.start_ns.size else 0, -1)
        value = np.where(ok, self.values[j] if self.values.size else np.nan, np.nan)
        return start, value


@dataclass(frozen=True, eq=False)
class Fundamentals:
    forecasts: dict[str, ForecastVintages]
    actuals: dict[str, ActualSeries]
    frame: pd.DataFrame | None  # the rows everything was built from (COLUMNS), if kept

    @classmethod
    def from_frame(cls, frame: pd.DataFrame, keep_frame: bool = True) -> Fundamentals:
        """Build the lookup arrays. ``keep_frame=False`` drops the rows afterwards: years of
        history are millions of rows, and the backtest copies the data into every process."""
        missing = set(SERIES) - set(frame["series"].unique())
        if missing:
            raise ValueError(f"fundamentals are missing series: {sorted(missing)}")
        forecasts = {s: ForecastVintages.from_frame(frame[frame.series == s]) for s in FORECASTS}
        actuals = {s: ActualSeries.from_frame(frame[frame.series == s]) for s in ACTUALS}
        return cls(forecasts, actuals, frame if keep_frame else None)

    def view(self, now) -> FundamentalsView:
        return FundamentalsView(self, None if now is None else ts_ns(now))

    def coverage(self) -> tuple[pd.Timestamp, pd.Timestamp] | None:
        """First and last publication time across forecasts (None if empty)."""
        times = [v.publish_ns for v in self.forecasts.values() if v.publish_ns.size]
        if not times:
            return None
        first = max(t[0] for t in times)
        last = min(t[-1] for t in times)
        return pd.Timestamp(first, tz="UTC"), pd.Timestamp(last, tz="UTC")


class FundamentalsView:
    """Fundamentals as public at ``now`` (UTC ns; None for the oracle view).

    Every query takes explicit issue times, which may not be later than ``now``.
    """

    def __init__(self, data: Fundamentals, now_ns: int | None):
        self._data = data
        self.now_ns = now_ns

    def _check(self, issue_ns) -> np.ndarray:
        issue = np.asarray(issue_ns, dtype=np.int64)
        if self.now_ns is not None and issue.size and issue.max() > self.now_ns:
            raise ValueError("fundamentals queried for a time after the view's moment")
        return issue

    def forecast(self, name: str, issue_ns, target_ns) -> np.ndarray:
        return self._data.forecasts[name].asof(self._check(issue_ns), target_ns)

    def latest_actual(self, name: str, issue_ns) -> tuple[np.ndarray, np.ndarray]:
        return self._data.actuals[name].latest(self._check(issue_ns))


def _ns(values) -> np.ndarray:
    return pd.DatetimeIndex(values).tz_convert("UTC").as_unit("ns").asi8


# --- From ERCOT reports to the stored schema ----------------------------------------------


def _rows(series: str, publish, start, value) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "series": series,
            "publish_time_utc": pd.DatetimeIndex(publish).tz_convert("UTC").as_unit("ns"),
            "interval_start_utc": pd.DatetimeIndex(start).tz_convert("UTC").as_unit("ns"),
            "value": np.asarray(value, dtype=float),
        }
    )


def _within_lead(frame: pd.DataFrame) -> pd.DataFrame:
    base = frame["publish_time_utc"].dt.floor("h")
    lead = frame["interval_start_utc"] - base
    keep = (lead >= pd.Timedelta(0)) & (lead < pd.Timedelta(MAX_LEAD_HOURS, unit="h"))
    return frame[keep & frame["value"].notna()]


def reduce_load_forecast(df: pd.DataFrame) -> pd.DataFrame:
    """Seven-Day Load Forecast by Model (gridstatus columns) -> ``load_forecast`` rows."""
    used = df[df["In Use Flag"].astype(bool)].sort_values(["Publish Time", "Model"])
    rows = _rows(
        "load_forecast", used["Publish Time"], used["Interval Start"], used["System Total"]
    )
    rows = rows.drop_duplicates(["publish_time_utc", "interval_start_utc"])
    return _within_lead(rows).reset_index(drop=True)


def _column(df: pd.DataFrame, name: str) -> pd.Series:
    """One column by label, merging duplicates into the first non-missing value per row.

    ERCOT renamed the wind and solar report columns on 2024-09-26, and gridstatus maps the old
    and new names to the same label, so a download spanning that day holds the label twice.
    """
    same = df.loc[:, df.columns == name]
    if same.shape[1] == 0:
        raise KeyError(name)
    return pd.to_numeric(same.bfill(axis=1).iloc[:, 0])


def reduce_renewable(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    """Hourly wind or solar production report (gridstatus columns) -> forecast + actual rows."""
    forecast_column = {"wind": "STWPF SYSTEM WIDE", "solar": "STPPF SYSTEM WIDE"}[kind]
    forecast = _within_lead(
        _rows(
            f"{kind}_forecast",
            df["Publish Time"],
            df["Interval Start"],
            _column(df, forecast_column),
        )
    )
    generation = _column(df, "GEN SYSTEM WIDE")
    measured = generation.notna().to_numpy()
    actual = _rows(
        f"{kind}_actual",
        df["Publish Time"][measured],
        df["Interval Start"][measured],
        generation[measured],
    )
    actual = actual.sort_values("publish_time_utc").drop_duplicates("interval_start_utc")
    return pd.concat([forecast, actual], ignore_index=True)


REPORTS: dict[str, tuple[str, Callable[[pd.DataFrame], pd.DataFrame]]] = {
    # report: (gridstatus method, reducer)
    "load": ("get_load_forecast_by_model", reduce_load_forecast),
    "wind": ("get_wind_actual_and_forecast_hourly", lambda df: reduce_renewable(df, "wind")),
    "solar": ("get_solar_actual_and_forecast_hourly", lambda df: reduce_renewable(df, "solar")),
}


# --- Download, store, load ----------------------------------------------------------------


def fundamentals_dir(data_dir) -> Path:
    return Path(data_dir) / "fundamentals"


def load_dotenv(path: Path = Path(".env")) -> None:
    """Set variables from a ``KEY=VALUE`` file without overriding ones already set."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def missing_credentials() -> list[str]:
    return [name for name in CREDENTIALS if not os.environ.get(name)]


def _client(source: str):
    try:
        import gridstatus
        from gridstatus.ercot_api.ercot_api import ErcotAPI
    except ImportError as exc:  # pragma: no cover - depends on the optional extra
        raise ImportError(
            'fundamentals downloads need gridstatus: pip install -e ".[data]"'
        ) from exc
    if source == "mis":
        return gridstatus.Ercot()
    if missing_credentials():
        raise RuntimeError("ERCOT API credentials are not set: " + ", ".join(missing_credentials()))
    # gridstatus waits 15 s for a response, but ERCOT often takes longer to build a month's zip
    # of ~740 files, and three timeouts in a row lose the month. Its requests read this
    # module-level setting at call time.
    from gridstatus.ercot_api import ercot_api

    ercot_api.REQUEST_TIMEOUT = (ercot_api.CONNECT_TIMEOUT_SECONDS, READ_TIMEOUT_SECONDS)
    return ErcotAPI(sleep_seconds=2.0)


def ingest(
    data_dir,
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    source: str = "api",
    force: bool = False,
    reports: Iterable[str] | None = None,
    log: Callable[[str], None] = print,
) -> list[Path]:
    """Download publications made in [start, end) and store them by report and month.

    ``source="api"`` reads the ERCOT Public API archive (years of history; needs credentials).
    ``source="mis"`` reads ERCOT's public website (only the last few days; no account).
    ``reports`` limits the run to some of ``REPORTS`` (default: all), so separate runs can
    download reports side by side. Months already stored are skipped unless ``force``; the
    month in progress is always refreshed. Returns the files written.
    """
    chosen = list(REPORTS) if reports is None else list(reports)
    unknown = sorted(set(chosen) - set(REPORTS))
    if unknown:
        raise ValueError(f"unknown reports {unknown}; choose from {list(REPORTS)}")
    client = _client(source)
    out = fundamentals_dir(data_dir) / ("archive" if source == "api" else "recent")
    out.mkdir(parents=True, exist_ok=True)
    tz = "US/Central"
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    start = start.tz_localize(tz) if start.tzinfo is None else start.tz_convert(tz)
    end = end.tz_localize(tz) if end.tzinfo is None else end.tz_convert(tz)
    now = pd.Timestamp.now(tz=tz)
    written = []
    for report in chosen:
        method, reduce = REPORTS[report]
        for period in _periods(start, end, monthly=source == "api"):
            lo, hi = max(period[0], start), min(period[1], end, now)
            if hi <= lo:
                continue
            in_progress = period[1] > now
            # Name files by what they hold, so a partial period (a --start or --end inside it)
            # never stands in for the whole one. The period in progress keeps one name and is
            # always refreshed.
            label_end = period[1] if in_progress else hi
            path = out / f"{report}_{lo:%Y-%m-%d}_{label_end:%Y-%m-%d}.parquet"
            if path.exists() and not force and not in_progress:
                continue
            began = time.perf_counter()
            try:
                raw = getattr(client, method)(date=lo, end=hi)
            except Exception as exc:  # no archive for this period, or a transient failure
                log(f"  {report} {lo:%Y-%m-%d}: no data ({type(exc).__name__}: {exc})")
                continue
            rows = reduce(raw)
            # Write, then rename: an interrupted run or a parallel reader never sees half a file.
            partial = path.with_name(path.name + ".part")
            rows.to_parquet(partial, index=False)
            partial.replace(path)
            written.append(path)
            pubs = rows["publish_time_utc"].nunique()
            log(
                f"  {report} {lo:%Y-%m-%d} -> {hi:%Y-%m-%d}: {pubs} publications, "
                f"{len(rows):,} rows ({time.perf_counter() - began:.0f}s)"
            )
    return written


ARCHIVE_IDS = {"load": "np3-565-cd", "wind": "np4-732-cd", "solar": "np4-737-cd"}


def check_access(log: Callable[[str], None] = print, years: Iterable[int] = ()) -> bool:
    """Sign in to the ERCOT API, list yesterday's archived publications for each report, and
    count what the archive holds on Jan 15 of each of ``years``. Never prints credentials."""
    tz = "US/Central"
    try:
        client = _client("api")
        client.get_token()
    except Exception as exc:
        log(f"sign-in failed ({exc}): check ERCOT_API_USERNAME and ERCOT_API_PASSWORD in .env")
        return False
    log("signed in to the ERCOT API")
    yesterday = pd.Timestamp.now(tz=tz).normalize() - pd.Timedelta(1, unit="D")
    for report, archive in ARCHIVE_IDS.items():
        try:
            found = client._get_historical_data_links(
                archive, yesterday, yesterday + pd.Timedelta(1, unit="D")
            )
        except Exception as exc:
            log(
                f"listing {report} publications failed ({type(exc).__name__}): check "
                "ERCOT_PUBLIC_API_SUBSCRIPTION_KEY in .env (the Public API primary key)"
            )
            return False
        log(f"  {report}: {len(found)} publications on {yesterday:%Y-%m-%d}")
    for year in years:
        day = pd.Timestamp(f"{year}-01-15", tz=tz)
        counts = []
        for report, archive in ARCHIVE_IDS.items():
            try:
                links = client._get_historical_data_links(
                    archive, day, day + pd.Timedelta(1, unit="D")
                )
                counts.append(f"{report} {len(links)}")
            except Exception:
                counts.append(f"{report} 0")
        log(f"  archive on {day:%Y-%m-%d}: " + ", ".join(counts))
    return True


def _periods(start, end, monthly: bool) -> Iterable[tuple[pd.Timestamp, pd.Timestamp]]:
    """Calendar months (or days) overlapping [start, end), in US Central time."""
    freq = "MS" if monthly else "D"
    first = start.normalize().replace(day=1) if monthly else start.normalize()
    edges = pd.date_range(first.tz_localize(None), end.tz_localize(None), freq=freq)
    edges = edges.tz_localize(start.tz, nonexistent="shift_forward", ambiguous=False)
    step = pd.DateOffset(months=1) if monthly else pd.DateOffset(days=1)
    for edge in edges:
        stop = (edge.tz_localize(None) + step).tz_localize(start.tz)
        yield edge, stop


def load_fundamentals(data_dir) -> Fundamentals | None:
    """Everything stored under ``data/fundamentals`` (None if nothing is)."""
    files = sorted(fundamentals_dir(data_dir).glob("*/*.parquet"))
    if not files:
        return None
    frame = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    frame["series"] = frame["series"].astype("category")
    frame = frame.drop_duplicates(["series", "publish_time_utc", "interval_start_utc"])
    if not set(SERIES) <= set(frame["series"].unique()):
        return None
    return Fundamentals.from_frame(frame[COLUMNS], keep_frame=False)
