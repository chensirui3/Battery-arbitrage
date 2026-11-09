"""ERCOT fundamentals: point-in-time storage, report parsing, features, and their value."""

from datetime import date

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from batt.cli import app
from batt.data import fundamentals as fund
from batt.data.fundamentals import ActualSeries, ForecastVintages, Fundamentals
from batt.data.store import MarketData, ts_ns
from batt.data.synthetic import synthetic_fundamentals, synthetic_prices
from batt.forecast import GBMForecaster, PriceHistory, published_horizon
from batt.forecast.features import (
    FEATURES,
    FUNDAMENTAL_FEATURES,
    bid_deadline_ns,
    build_features,
)
from batt.forecast.history import HOUR_NS, MINUTE_NS
from batt.forecast.registry import make_forecaster
from batt.policies import make_policy
from batt.sim import SimulationGrid, simulate
from batt.timeutils import minutes
from helpers import corrupt_after

TZ = "America/Chicago"
STEP = 15 * MINUTE_NS


def local(text: str) -> pd.Timestamp:
    return pd.Timestamp(text).tz_localize(TZ).tz_convert("UTC")


def vintage_rows(publish, starts, values) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "publish_time_utc": pd.to_datetime(publish, utc=True),
            "interval_start_utc": pd.to_datetime(starts, utc=True),
            "value": values,
        }
    )


def test_forecast_vintages_return_the_latest_published_value():
    rows = vintage_rows(
        ["2024-03-01 10:30", "2024-03-01 10:30", "2024-03-01 11:30", "2024-03-01 11:30"],
        ["2024-03-01 12:00", "2024-03-01 13:00", "2024-03-01 12:00", "2024-03-01 14:00"],
        [100.0, 200.0, 110.0, 300.0],
    )
    vintages = ForecastVintages.from_frame(rows)
    target = pd.to_datetime(["2024-03-01 12:15", "2024-03-01 13:45", "2024-03-01 14:00"], utc=True)
    before = vintages.asof(ts_ns(pd.Timestamp("2024-03-01 11:00", tz="UTC")), target.asi8)
    after = vintages.asof(ts_ns(pd.Timestamp("2024-03-01 11:30", tz="UTC")), target.asi8)
    np.testing.assert_array_equal(before, [100.0, 200.0, np.nan])
    # 13:00 is missing from the 11:30 publication, so the 10:30 value stands.
    np.testing.assert_array_equal(after, [110.0, 200.0, 300.0])
    assert np.isnan(
        vintages.asof(ts_ns(pd.Timestamp("2024-03-01 10:00", tz="UTC")), target.asi8)
    ).all()


def test_actuals_are_visible_only_once_published():
    actuals = ActualSeries.from_frame(
        vintage_rows(
            ["2024-03-01 11:55", "2024-03-01 12:55", "2024-03-01 13:20"],  # last one a re-post
            ["2024-03-01 10:00", "2024-03-01 11:00", "2024-03-01 10:00"],
            [5.0, 6.0, 99.0],
        )
    )
    issue = pd.to_datetime(["2024-03-01 11:54", "2024-03-01 12:00", "2024-03-01 14:00"], utc=True)
    hour, value = actuals.latest(issue.asi8)
    np.testing.assert_array_equal(value, [np.nan, 5.0, 6.0])  # first publication wins
    assert hour[2] == pd.Timestamp("2024-03-01 11:00", tz="UTC").value


def test_reports_reduce_to_point_in_time_rows():
    publish = pd.Timestamp("2024-07-01 10:30", tz=TZ)
    starts = pd.date_range("2024-07-01 00:00", periods=72, freq="h", tz=TZ)
    load = pd.concat(
        [
            pd.DataFrame(
                {
                    "Interval Start": starts,
                    "Publish Time": publish,
                    "Model": model,
                    "System Total": total,
                    "In Use Flag": in_use,
                }
            )
            for model, total, in_use in (("E", 50_000.0, False), ("X", 60_000.0, True))
        ]
    )
    rows = fund.reduce_load_forecast(load)
    assert (rows["value"] == 60_000.0).all()  # the model in use
    lead = (rows.interval_start_utc - rows.publish_time_utc.dt.floor("h")) / pd.Timedelta(1, "h")
    assert lead.min() == 0 and lead.max() == fund.MAX_LEAD_HOURS - 1

    wind = pd.DataFrame(
        {
            "Interval Start": list(starts[:12]) * 2,
            "Publish Time": [pd.Timestamp("2024-07-01 10:55", tz=TZ)] * 12
            + [pd.Timestamp("2024-07-01 11:55", tz=TZ)] * 12,
            "GEN SYSTEM WIDE": [1000.0 + i if i < 9 else np.nan for i in range(12)]
            + [2000.0 + i if i < 10 else np.nan for i in range(12)],
            "STWPF SYSTEM WIDE": np.arange(24.0),
        }
    )
    rows = fund.reduce_renewable(wind, "wind")
    actual = rows[rows.series == "wind_actual"].set_index("interval_start_utc")["value"]
    assert actual.iloc[0] == 1000.0 and actual.iloc[9] == 2009.0  # each hour's first posting
    forecast = rows[rows.series == "wind_forecast"]
    assert (forecast.interval_start_utc >= forecast.publish_time_utc.dt.floor("h")).all()


def test_columns_renamed_mid_download_are_merged():
    # ERCOT renamed the report columns on 2024-09-26 and gridstatus maps the old and new names
    # to one label, so a month spanning that day holds each label twice.
    hour = pd.date_range("2024-09-26 00:00", periods=4, freq="h", tz=TZ)
    old, new = pd.Timestamp("2024-09-26 01:55", tz=TZ), pd.Timestamp("2024-09-26 02:55", tz=TZ)
    wind = pd.DataFrame(
        [  # start, publish, generation (old, new name), forecast (old, new name)
            [hour[0], old, 10.0, np.nan, np.nan, np.nan],
            [hour[2], old, np.nan, np.nan, 5.0, np.nan],
            [hour[1], new, np.nan, 11.0, np.nan, np.nan],
            [hour[3], new, np.nan, np.nan, np.nan, 6.0],
        ]
    )
    wind.columns = (
        ["Interval Start", "Publish Time"] + ["GEN SYSTEM WIDE"] * 2 + ["STWPF SYSTEM WIDE"] * 2
    )
    rows = fund.reduce_renewable(wind, "wind")
    actual = rows[rows.series == "wind_actual"].sort_values("interval_start_utc")
    assert actual["value"].tolist() == [10.0, 11.0]
    forecast = rows[rows.series == "wind_forecast"].sort_values("publish_time_utc")
    assert forecast["value"].tolist() == [5.0, 6.0]
    with pytest.raises(ValueError, match="unknown reports"):
        fund.ingest("unused", hour[0], hour[1], reports=["gas"])


def test_bid_deadline_is_10am_the_day_before_even_across_dst():
    targets = [local("2024-03-10 18:00"), local("2024-03-11 00:30"), local("2024-11-03 23:00")]
    deadlines = bid_deadline_ns(np.array([t.value for t in targets]), TZ)
    assert [
        pd.Timestamp(d, tz="UTC").tz_convert(TZ).strftime("%m-%d %H:%M") for d in deadlines
    ] == [
        "03-09 10:00",
        "03-10 10:00",
        "11-02 10:00",
    ]


@pytest.fixture(scope="module")
def world(timing):
    """Synthetic prices whose RT-DA spread is driven by net-load surprises."""
    rows, truth = synthetic_fundamentals(date(2024, 1, 1), 80, seed=2)
    rt, da = synthetic_prices(date(2024, 1, 1), 80, seed=9)
    hour = rt["interval_start_utc"].dt.floor("h")
    da_hourly = da.set_index("interval_start_utc")["price"]
    noise = np.random.default_rng(3).normal(0, 3, len(rt))
    rt = rt.assign(
        price=da_hourly.reindex(hour).to_numpy()
        + 0.03 * truth["net_surprise"].reindex(hour).to_numpy()
        + noise
    )
    market = MarketData.from_price_frames("SYN", rt, da, timing)
    return market.with_fundamentals(Fundamentals.from_frame(rows))


def test_fundamental_features_never_use_unpublished_data(world):
    full = PriceHistory.from_view(world.view(None), local("2024-01-01"), local("2024-03-20"), TZ)
    noisy_market = corrupt_after(world, local("2024-03-10 15:55"))
    for when in ("2024-03-05 08:00", "2024-03-10 16:00"):  # the second just after the cutoff
        first = local(when)
        issue = ts_ns(first - minutes(5))
        view = world.view(pd.Timestamp(issue, tz="UTC"))
        n = published_horizon(view, ts_ns(first), STEP)
        targets = ts_ns(first) + np.arange(n) * STEP
        k_full = (targets - full.origin_ns) // STEP
        oracle = build_features(full, issue, k_full, world.view(None).fundamentals)
        live = build_features(full, issue, k_full, view.fundamentals)
        noisy = build_features(
            full, issue, k_full, noisy_market.view(pd.Timestamp(issue, tz="UTC")).fundamentals
        )
        extra = slice(len(FEATURES), None)
        np.testing.assert_array_equal(live[:, extra], oracle[:, extra])
        np.testing.assert_array_equal(live[:, extra], noisy[:, extra])
        assert np.isfinite(live[:, extra]).all()
        assert live.shape[1] == len(FEATURES) + len(FUNDAMENTAL_FEATURES)


def test_fundamentals_improve_forecasts_when_they_drive_prices(world, timing):
    plain = GBMForecaster(train_days=60, num_boost_round=80, params={"num_threads": 1})
    informed = GBMForecaster(
        train_days=60, num_boost_round=80, params={"num_threads": 1}, fundamentals=True
    )
    errors = {"plain": [], "informed": []}
    for forecaster in (plain, informed):
        forecaster.reset(timing)
    for day in pd.date_range("2024-03-05", "2024-03-15", freq="2D"):
        first = local(f"{day.date()} 14:00")
        now = first - minutes(5)
        view = world.view(now)
        n = published_horizon(view, ts_ns(first), STEP)
        actual = world.aligned("rt", pd.date_range(first, periods=n, freq="15min"))
        for name, forecaster in (("plain", plain), ("informed", informed)):
            forecaster.prepare(view, now, first)
            errors[name].append(np.abs(forecaster.predict(view, now, first, n) - actual))
    plain_mae, informed_mae = (np.concatenate(errors[k]).mean() for k in ("plain", "informed"))
    assert informed_mae < 0.6 * plain_mae


def test_policies_with_fundamentals_never_see_unpublished_data(world, spec, timing):
    grid = SimulationGrid.for_days(date(2024, 3, 4), date(2024, 3, 11), TZ)
    cutoff = local("2024-03-07 12:00")
    params = {"policy": "mpc_gbm", "forecaster_params": {"fundamentals": True, "refit": "day"}}
    settings = {"gbm": {"num_boost_round": 25, "params": {"num_threads": 1}}}
    runs = [
        simulate(make_policy("mpc_gbm_f", params, forecasters=settings), m, grid, spec, timing)
        for m in (world, corrupt_after(world, cutoff))
    ]
    decided = np.asarray(grid.starts - minutes(5) <= cutoff)
    clean, noisy = (r.ledger.requested_mw.to_numpy() for r in runs)
    np.testing.assert_allclose(clean[decided], noisy[decided])
    assert not np.allclose(clean, noisy)


def test_forecasters_without_fundamentals_data_say_what_to_do(market, timing):
    forecaster = GBMForecaster(fundamentals=True)
    forecaster.reset(timing)
    first = local("2024-03-12 15:00")
    with pytest.raises(ValueError, match="ingest-fundamentals"):
        forecaster.prepare(market.view(first - minutes(5)), first - minutes(5), first)


def test_forecaster_variants_start_from_the_base_settings():
    catalog = {"gbm": {"train_days": 365, "num_boost_round": 50}}
    variant = make_forecaster("gbm_f", {"forecaster": "gbm", "fundamentals": True}, catalog)
    assert isinstance(variant, GBMForecaster) and variant.name == "gbm_f"
    assert variant.fundamentals and variant.train_days == 365 and variant.num_boost_round == 50
    with pytest.raises(KeyError):
        make_forecaster("x", {"forecaster": "nope"})


def test_dotenv_fills_only_missing_variables(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text(
        "# comment\nERCOT_API_USERNAME = 'me@example.com'\nERCOT_API_PASSWORD=from_file\n"
    )
    monkeypatch.delenv("ERCOT_API_USERNAME", raising=False)
    monkeypatch.setenv("ERCOT_API_PASSWORD", "already_set")
    fund.load_dotenv(env)
    import os

    assert os.environ["ERCOT_API_USERNAME"] == "me@example.com"
    assert os.environ["ERCOT_API_PASSWORD"] == "already_set"


def test_ingest_without_credentials_explains_the_setup(tmp_path, monkeypatch):
    for name in fund.CREDENTIALS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)  # no .env here
    result = CliRunner().invoke(app, ["ingest-fundamentals"])
    assert result.exit_code == 1
    assert "apiexplorer.ercot.com" in result.output and "--recent" in result.output


def test_vintage_lookup_is_fast_enough_for_training(world):
    vintages = world.fundamentals.forecasts["load_forecast"]
    issue = np.repeat(ts_ns(local("2024-03-05 08:00")), 200_000)
    target = issue + np.random.default_rng(0).integers(0, 30, issue.size) * HOUR_NS
    import time

    started = time.perf_counter()
    values = vintages.asof(issue, target)
    assert time.perf_counter() - started < 0.5
    assert np.isfinite(values).all()
