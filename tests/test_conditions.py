"""Grid conditions (``batt.data.conditions``): reading ERCOT's reports, point-in-time lookups,
the six features, and ERCOT's look-ahead as a forecast (``batt.forecast.lookahead``)."""

import io
import time
import zipfile

import numpy as np
import pandas as pd
import pytest
import requests

from batt.data import conditions
from batt.data.conditions import (
    GridConditions,
    LookAhead,
    _refresh_early,
    _retrying,
    decision_times,
    reduce_outages,
    reduce_rtd,
    reduce_sced,
    select_runs,
)
from batt.data.synthetic import synthetic_conditions
from batt.forecast.features import CONDITION_FEATURES, condition_features
from batt.forecast.history import PriceHistory
from batt.forecast.lookahead import LookAheadForecaster, hub_basis
from batt.policies import make_policy
from batt.sim import simulate
from helpers import corrupt_after

UTC = "UTC"
NOW = pd.Timestamp("2024-03-08 17:55", tz=UTC)  # a decision time, mid-backtest
STEP = pd.Timedelta(15, unit="min")


def utc(text: str) -> pd.Timestamp:
    return pd.Timestamp(text, tz=UTC)


def test_sced_rows_resolve_local_times_and_both_market_designs():
    # Both passes through the hour repeated when daylight saving time ends, then a run after
    # the RTC+B redesign, which has no reserve adders and no PRC.
    raw = pd.DataFrame(
        {
            "SCEDTimestamp": ["11/03/2024 01:20:00", "11/03/2024 01:20:00", "12/06/2025 15:20:14"],
            "RepeatedHourFlag": ["N", "Y", "N"],
            "SystemLambda": [30.0, 40.0, 50.0],
            "RTORPA": [5.0, 0.0, np.nan],
            "RTORDPA": [1.0, 0.0, np.nan],
            "RTRDPA": [np.nan, np.nan, 2.0],
            "PRC": [4000.0, 4100.0, np.nan],
            "postDatetime": ["2024-11-03T01:20:03", "2024-11-03T01:20:03", "2025-12-06T15:20:17"],
        }
    )
    rows = reduce_sced(raw)
    expected = [utc("2024-11-03 06:20"), utc("2024-11-03 07:20"), utc("2025-12-06 21:20:14")]
    assert list(rows["time_utc"]) == expected
    assert list(rows["publish_time_utc"] - rows["time_utc"]) == [pd.Timedelta(3, "s")] * 3
    np.testing.assert_allclose(rows["price"], [36.0, 40.0, 52.0])
    assert rows["prc"].iloc[:2].tolist() == [4000.0, 4100.0] and np.isnan(rows["prc"].iat[2])


def test_post_times_in_the_repeated_hour_are_never_before_the_run():
    # ERCOT's post times carry no flag. A file posted at 01:00:01 reports on a run either just
    # before (the first 01:00 is daylight time) or an hour later (the second, standard time).
    raw = pd.DataFrame(
        {
            "SCEDTimestamp": ["11/03/2024 00:59:58", "11/03/2024 01:59:58"],
            "RepeatedHourFlag": ["N", "N"],
            "SystemLambda": [30.0, 30.0],
            "postDatetime": ["2024-11-03T01:00:01", "2024-11-03T01:00:01"],
        }
    )
    rows = reduce_sced(raw)
    assert list(rows["publish_time_utc"]) == [
        utc("2024-11-03 06:00:01"),
        utc("2024-11-03 07:00:01"),
    ]
    assert (rows["publish_time_utc"] >= rows["time_utc"]).all()


def test_rtd_rows():
    raw = pd.DataFrame(
        {
            "RTDTimestamp": ["07/15/2023 15:15:03"] * 2,
            "RepeatedHourFlag": ["N", "N"],
            "IntervalEnding": ["07/15/2023 15:20:00", "07/15/2023 15:25:00"],
            "IERepeatedHourFlag": ["N", "N"],
            "SystemLambda": [51.0, 60.0],
            "RTORPA": [0.0, 10.0],
            "RTORDPA": [0.0, 0.0],
            "postDatetime": ["2023-07-15T15:15:22.722"] * 2,
        }
    )
    rows = reduce_rtd(raw)
    assert list(rows["interval_end_utc"]) == [utc("2023-07-15 20:20"), utc("2023-07-15 20:25")]
    assert (rows["run_time_utc"] == utc("2023-07-15 20:15:03")).all()
    np.testing.assert_allclose(rows["price"], [51.0, 70.0])


def test_outage_totals_across_the_format_change():
    # ERCOT added zonal columns on 2021-05-27. Over a download spanning that day, gridstatus
    # sums the zones into the total, leaving it missing for rows in the old format.
    first = pd.Timestamp("2021-05-27 15:01", tz="US/Central")  # old format
    second = pd.Timestamp("2021-05-27 17:01", tz="US/Central")  # new format
    raw = pd.DataFrame(
        {
            "Publish Time": [first, first, second, second],
            "Interval Start": pd.date_range(
                "2021-05-27 15:00", periods=4, freq="h", tz="US/Central"
            ),
            "Total Resource MW": [np.nan, np.nan, 900.0, 950.0],
            "Total IRR MW": [np.nan, np.nan, 30.0, 35.0],
            "TotalResourceMW": [1000.0, 1100.0, np.nan, np.nan],
            "TotalIRRMW": [40.0, 45.0, np.nan, np.nan],
        }
    )
    rows = reduce_outages(raw)
    value = rows.groupby("series")["value"].apply(list)
    assert value["outage_resource"] == [1000.0, 1100.0, 900.0, 950.0]
    assert value["outage_irr"] == [40.0, 45.0, 30.0, 35.0]
    assert rows["publish_time_utc"].nunique() == 2


def test_one_run_kept_per_decision():
    start = pd.Timestamp("2024-07-01 00:00", tz="US/Central")
    decisions = decision_times(start, start + pd.Timedelta(1, "h"), 15, 5.0)
    local = [pd.Timestamp(d, tz=UTC).tz_convert("US/Central").strftime("%H:%M") for d in decisions]
    assert local == ["00:10", "00:25", "00:40", "00:55"]
    runs = start + pd.to_timedelta(np.arange(0, 60, 5), "min") + pd.Timedelta(3, "s")
    # The latest run posted by each decision: those at :05, :20, :35, and :50.
    assert select_runs(runs.as_unit("ns").asi8, decisions).tolist() == [1, 4, 7, 10]


def test_look_ahead_price_needs_the_whole_interval():
    run = utc("2024-07-01 12:00:03")
    frame = pd.DataFrame(
        {
            "publish_time_utc": run + pd.Timedelta(12, "s"),
            "run_time_utc": run,
            "interval_end_utc": pd.date_range("2024-07-01 12:05", periods=11, freq="5min", tz=UTC),
            "price": np.arange(11.0),
        }
    )
    ahead = LookAhead.from_frame(frame)

    def price(issue, start):
        a = pd.Timestamp(start, tz=UTC)
        return ahead.mean_price(issue.value, a.value, (a + STEP).value)

    later = run + pd.Timedelta(1, "min")
    assert price(later, "2024-07-01 12:15") == pytest.approx(4.0)  # ends 12:20, 12:25, 12:30
    assert np.isnan(price(later, "2024-07-01 12:45"))  # 13:00 is beyond the look-ahead
    assert np.isnan(price(run, "2024-07-01 12:15"))  # not posted yet


def test_empty_months_name_months_without_publications(frames):
    sced, rtd, outages = synthetic_conditions(frames[0], seed=7)  # Feb 1 - Apr 10, 2024
    local = sced["publish_time_utc"].dt.tz_convert("US/Central")
    march = (local >= pd.Timestamp("2024-03-01", tz="US/Central")) & (
        local < pd.Timestamp("2024-04-01", tz="US/Central")
    )
    gappy = GridConditions.from_frames(sced[~march], rtd, outages)
    assert gappy.empty_months("2024-02-01", "2024-04-01") == ["sced 2024-03"]
    assert gappy.empty_months("2024-02-15", "2024-03-02") == ["sced 2024-03"]
    assert gappy.empty_months("2024-02-01", "2024-03-01") == []  # the end is excluded
    whole = GridConditions.from_frames(sced, rtd, outages)
    assert whole.empty_months("2024-02-01", "2024-04-01") == []


class FakeClient:
    """Just enough of gridstatus's ErcotAPI for the download loop."""

    def __init__(self):
        self.token, self.token_expiry, self.sign_ins = None, None, 0

    def get_token(self):
        self.sign_ins += 1
        self.token, self.token_expiry = "token", time.time() + 3600

    def refresh_token_if_needed(self):
        if not self.token or time.time() >= self.token_expiry:
            self.get_token()

    def make_api_call(self, *args, **kwargs):
        return None


def test_tokens_are_renewed_before_their_hour_is_up():
    client = FakeClient()
    _refresh_early(client, margin=300)
    client.refresh_token_if_needed()
    assert client.sign_ins == 1
    assert client.token_expiry == pytest.approx(time.time() + 3300, abs=5)


def test_failed_downloads_are_retried_after_signing_in_again(monkeypatch):
    monkeypatch.setattr(conditions.time, "sleep", lambda seconds: None)
    client, seen = FakeClient(), []
    client.token = "stale"

    def flaky():
        seen.append(client.token)
        if len(seen) < 3:
            raise RuntimeError("401 Client Error: Unauthorized")
        return "rows"

    assert _retrying(flaky, client, log=lambda message: None) == "rows"
    assert seen == ["stale", None, None]  # signed out after a failure, so the next try signs in

    def nothing_posted():
        seen.append("lookup")
        raise LookupError("no files posted")

    seen.clear()
    with pytest.raises(LookupError):
        _retrying(nothing_posted, client, log=lambda message: None)
    assert seen == ["lookup"]  # not retried


def test_months_that_keep_failing_are_skipped_and_named(monkeypatch, tmp_path):
    monkeypatch.setattr(conditions.time, "sleep", lambda seconds: None)

    def fetch_runs(client, report, lo, hi, step_minutes, lead_minutes, log=None):
        if report == "sced" and lo.month == 2:
            raise RuntimeError("archive unavailable")
        return pd.DataFrame({"value": [1.0]}), 1

    monkeypatch.setattr(conditions, "_client", lambda source: FakeClient())
    monkeypatch.setattr(conditions, "_fetch_runs", fetch_runs)
    monkeypatch.setattr(
        conditions, "_fetch_outages", lambda client, lo, hi: fetch_runs(client, "", lo, hi, 0, 0)
    )
    messages = []
    written = conditions.ingest(tmp_path, "2024-01-01", "2024-03-01", log=messages.append)
    assert sorted(p.name for p in written) == [
        "outages_2024-01-01_2024-02-01.parquet",
        "outages_2024-02-01_2024-03-01.parquet",
        "rtd_2024-01-01_2024-02-01.parquet",
        "rtd_2024-02-01_2024-03-01.parquet",
        "sced_2024-01-01_2024-02-01.parquet",
    ]
    assert sum("try" in m and "failed" in m for m in messages) == conditions.ATTEMPTS - 1
    assert messages[-1].startswith("not stored: sced 2024-02-01.")


class Archive:
    """The archive's listing and downloads of a few runs, some of which it refuses to serve."""

    batch_size = 50

    def __init__(self, posted, refused=(), status=400):
        self.posted, self.refused, self.status, self.requests = posted, set(refused), status, []

    def _get_historical_data_links(self, emil, lo, hi):
        return [
            (f"https://archive/download?docId={1000 + i}", p) for i, p in enumerate(self.posted)
        ]

    def _bulk_download_documents(self, ids, emil):
        self.requests.append(list(ids))
        if self.refused & set(ids):
            response = requests.Response()
            response.status_code, response.encoding = self.status, "utf-8"
            response._content = b'{"message":"Your request could not be processed."}'
            raise requests.HTTPError(f"{self.status} Client Error", response=response)
        files = []
        for doc in ids:
            buffer = io.BytesIO()
            with zipfile.ZipFile(buffer, "w") as archive:
                archive.writestr(f"{doc}.csv", f"doc\n{doc}\n")
            buffer.seek(0)
            files.append((buffer, f"{doc}.zip"))
        return files


def test_a_run_the_archive_refuses_is_replaced_by_the_one_before(monkeypatch):
    monkeypatch.setattr(conditions, "reduce_rtd", lambda raw: raw)
    start = pd.Timestamp("2024-07-01 00:00", tz="US/Central")
    end = start + pd.Timedelta(1, "h")
    posted = [f"2024-07-01T00:{m:02d}:03" for m in range(0, 60, 5)]  # decisions at :10, :25, ...
    client, messages = Archive(posted, refused={"1004"}), []
    rows, runs = conditions._fetch_runs(client, "rtd", start, end, 15, 5, log=messages.append)
    # The decision at 00:25 would use the run posted at 00:20 (1004); it gets the one at 00:15.
    assert rows["doc"].tolist() == [1001, 1003, 1007, 1010] and runs == 4
    assert len(messages) == 1 and "1004, posted 2024-07-01T00:20:03" in messages[0]
    assert client.requests[-1] == ["1003"]

    other = Archive(posted, refused={"1004"}, status=401)  # not a refusal: left to the retries
    with pytest.raises(requests.HTTPError):
        conditions._fetch_runs(other, "rtd", start, end, 15, 5, log=messages.append)
    monkeypatch.setattr(conditions, "MAX_REFUSED", 0)  # more refused than that: the month fails
    with pytest.raises(RuntimeError, match="refuses more than 0"):
        conditions._fetch_runs(Archive(posted, {"1004"}), "rtd", start, end, 15, 5, log=print)


def test_views_refuse_later_queries(market):
    conditions = market.view(NOW).conditions
    with pytest.raises(ValueError):
        conditions.sced(np.array([NOW.value + 1]))
    with pytest.raises(ValueError):
        conditions.outage(np.array([NOW.value + 1]), np.array([NOW.value]))


def _features(market, now, timing, n=8):
    view = market.view(now)
    hist = PriceHistory.from_view(
        view, now - pd.Timedelta(3, "D"), now + pd.Timedelta(2, "D"), timing.timezone
    )
    k = hist.index(now + pd.Timedelta(5, "min")) + np.arange(n)
    return condition_features(view.conditions, hist, now.value, k)


def test_features_use_only_published_data(market, timing):
    clean = _features(market, NOW, timing)
    noisy_market = corrupt_after(market, NOW)
    np.testing.assert_array_equal(clean, _features(noisy_market, NOW, timing))
    assert clean.shape[1] == len(CONDITION_FEATURES)
    rtd = clean[:, CONDITION_FEATURES.index("rtd_spread")]
    assert np.isfinite(rtd[:3]).all() and np.isnan(rtd[3:]).all()  # the look-ahead covers 3
    assert np.isfinite(clean[:, CONDITION_FEATURES.index("outage_rev")]).all()
    # The corruption itself is visible once its data are public.
    later = NOW + pd.Timedelta(2, "h")
    assert not np.allclose(_features(market, later, timing), _features(noisy_market, later, timing))


def _forecast(market, timing, n=8, **settings):
    forecaster = LookAheadForecaster(**settings)
    forecaster.reset(timing)
    view = market.view(NOW)
    first = NOW + pd.Timedelta(5, "min")
    hist = PriceHistory.from_view(
        view, NOW - pd.Timedelta(1, "D"), first + n * STEP, timing.timezone
    )
    k = hist.index(first) + np.arange(n)
    starts = hist.start_ns(k)
    rtd = view.conditions.rtd_price(np.full(n, NOW.value), starts, starts + STEP.value)
    out = forecaster.predict(view, NOW, first, n)
    return out, rtd, hist.visible_da(k, NOW.value), hub_basis(view.conditions, hist, NOW.value)


def test_look_ahead_forecast_settings(market, timing):
    out, rtd, da, _ = _forecast(market, timing, mode="all")
    np.testing.assert_allclose(out[:3], rtd[:3])
    np.testing.assert_allclose(out[3:], da[3:])

    out, rtd, da, basis = _forecast(market, timing, mode="all", basis="latest")
    assert basis != 0
    np.testing.assert_allclose(out[:3], rtd[:3] + basis)

    out, rtd, da, _ = _forecast(market, timing, mode="spikes", threshold=1e6)
    np.testing.assert_allclose(out, da)  # no deviation is that big
    out, rtd, da, _ = _forecast(market, timing, mode="spikes", threshold=1e-6)
    np.testing.assert_allclose(out[:3], rtd[:3])


def test_mpc_rtd_plans_on_the_look_ahead(frames, market, grid, spec, timing):
    # With an exact look-ahead used for every interval it covers, mpc_rtd knows the next 45
    # minutes' prices and should earn more than planning on DA prices. (With the default
    # synthetic noise it trades on noise and loses, which is why the plan also tests a
    # spikes-only mode.)
    rt, _ = frames
    exact = market.with_conditions(
        GridConditions.from_frames(*synthetic_conditions(rt, seed=7, rtd_noise=0.0))
    )
    policies = {
        "mpc_da": make_policy("mpc_da"),
        "mpc_rtd": make_policy("mpc_rtd", {"forecaster_params": {"mode": "all"}}),
    }
    revenue = {
        name: simulate(policy, exact, grid, spec, timing).ledger.revenue_net.sum()
        for name, policy in policies.items()
    }
    assert revenue["mpc_rtd"] > 1.05 * revenue["mpc_da"]


def test_fits_log_the_conditions_share_of_gain(market, timing):
    from batt.forecast import GBMForecaster, SpikeForecaster

    gbm, spike = GBMForecaster(conditions=True), SpikeForecaster(conditions=True, min_positives=20)
    for model in (gbm, spike):
        model.reset(timing)
        model.prepare(market.view(NOW), NOW, NOW + pd.Timedelta(5, "min"))
    assert 0 < gbm.fit_log[-1]["condition_gain"] < 1
    shares = [v for k, v in spike.fit_log[-1].items() if k.endswith("_condition_gain")]
    assert shares and all(0 < share < 1 for share in shares)  # each side fitted


def test_days_after_rtc_b_are_their_own_period():
    from batt.timeutils import period_labels

    local = ["2024-06-01 12:00", "2025-12-04 23:45", "2025-12-05 00:00", "2026-01-02 08:00"]
    labels = period_labels(pd.DatetimeIndex(local).tz_localize("America/Chicago"))
    assert labels.tolist() == ["2024", "2025", "2025 RTC+B", "2026 RTC+B"]
