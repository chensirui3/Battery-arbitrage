"""Price history, point-in-time features, and forecasters."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from batt.data.store import MarketData, ts_ns
from batt.data.synthetic import synthetic_prices
from batt.forecast import (
    LOOKBACK_DAYS,
    DayAheadForecaster,
    GBMForecaster,
    PriceHistory,
    RefitSchedule,
    SpreadAR1Forecaster,
    build_features,
    published_horizon,
)
from batt.forecast.features import COLUMN
from batt.forecast.history import DAY_NS, MINUTE_NS, local_naive_ns
from batt.timeutils import days, minutes

TZ = "America/Chicago"
STEP = 15 * MINUTE_NS
LEAD = 5 * MINUTE_NS


def local(text: str) -> pd.Timestamp:
    return pd.Timestamp(text).tz_localize(TZ).tz_convert("UTC")


def test_local_time_conversion_matches_pandas():
    utc = pd.date_range("2023-03-11", "2023-11-07", freq="5min", tz="UTC")  # both DST changes
    for tz in ("America/Chicago", "Australia/Lord_Howe", "Asia/Kolkata", "UTC"):
        expected = utc.tz_convert(tz).tz_localize(None).asi8
        np.testing.assert_array_equal(local_naive_ns(utc.asi8, tz), expected)


def test_history_holds_only_published_values(market):
    now = local("2024-03-06 10:00")
    hist = PriceHistory.from_view(market.view(now), now - days(3), now + days(2), TZ)
    starts = pd.to_datetime(hist.start_ns(np.arange(len(hist))), utc=True)

    rt_visible = ~np.isnan(hist.rt)
    np.testing.assert_array_equal(hist.rt[rt_visible], market.aligned("rt", starts)[rt_visible])
    assert (hist.rt_publish[rt_visible] <= ts_ns(now)).all()
    latest = starts[rt_visible][-1]  # ended, plus the 5-minute publish delay, before now
    assert latest + minutes(20) <= now < latest + minutes(35)

    # At 10:00 the DA prices run to the end of today; tomorrow's post at 13:30.
    da_visible = ~np.isnan(hist.da)
    np.testing.assert_array_equal(hist.da[da_visible], market.aligned("da", starts)[da_visible])
    assert starts[da_visible][-1] == local("2024-03-06 23:45")
    hourly = hist.da[da_visible].reshape(-1, 4)  # history starts on the hour
    assert (hourly == hourly[:, :1]).all()


def test_horizon_reaches_tomorrow_once_its_da_prices_post(market):
    def horizon(first: pd.Timestamp) -> int:
        return published_horizon(market.view(first - minutes(5)), ts_ns(first), STEP)

    assert horizon(local("2024-03-06 13:15")) == 43  # to midnight tonight
    assert horizon(local("2024-03-06 13:45")) == 41 + 96  # to midnight tomorrow


def test_features_never_use_unpublished_data(market):
    """Features built from all data must equal features built from only what was public."""
    full = PriceHistory.from_view(
        market.view(None), local("2024-02-01 00:00"), local("2024-04-10 00:00"), TZ
    )
    rng = np.random.default_rng(0)
    firsts = [full.start_ns(k) for k in rng.integers(36 * 96, len(full) - 3 * 96, size=20)]
    firsts += [ts_ns(local("2024-03-12 13:30")), ts_ns(local("2024-03-12 13:45"))]
    for first in firsts:
        issue = first - LEAD
        view = market.view(pd.Timestamp(issue, tz="UTC"))
        n = published_horizon(view, first, STEP)
        live_hist = PriceHistory.from_view(
            view, issue - LOOKBACK_DAYS * DAY_NS, first + n * STEP, TZ
        )
        targets = first + np.arange(n) * STEP
        live = build_features(live_hist, issue, (targets - live_hist.origin_ns) // STEP)
        batch = build_features(full, issue, (targets - full.origin_ns) // STEP)
        np.testing.assert_allclose(live, batch, rtol=1e-9, atol=1e-6, equal_nan=True)
        assert np.isfinite(live[:, COLUMN["spread_30d"]]).all()
        assert np.isfinite(live[:, COLUMN["da"]]).all()


def test_refit_schedule():
    times = [local(t) for t in ("2024-01-31 23:45", "2024-02-01", "2024-02-15", "2024-03-01")]
    monthly = RefitSchedule("month")
    assert [monthly.due(t, TZ) for t in times] == [True, True, False, True]
    once = RefitSchedule("once")
    assert [once.due(t, TZ) for t in times] == [True, False, False, False]
    weekly = RefitSchedule("week")  # 2024-02-05 is a Monday
    stamps = [local(t) for t in ("2024-02-04 23:45", "2024-02-05", "2024-02-11 12:00")]
    assert [weekly.due(t, TZ) for t in stamps] == [True, True, False]
    with pytest.raises(ValueError):
        RefitSchedule("hourly")


def test_day_ahead_forecaster_returns_published_da(market, timing):
    first = local("2024-03-12 15:00")
    now = first - minutes(5)
    view = market.view(now)
    forecaster = DayAheadForecaster()
    forecaster.reset(timing)
    n = published_horizon(view, ts_ns(first), STEP)
    starts = pd.date_range(first, periods=n, freq="15min")
    np.testing.assert_array_equal(
        forecaster.predict(view, now, first, n), market.aligned("da", starts)
    )
    # Past the published horizon the DA price is unknown.
    assert np.isnan(forecaster.predict(view, now, first, n + 4)[-4:]).all()


def test_ar1_forecast_decays_from_latest_spread_toward_da(market, timing):
    first = local("2024-03-12 18:00")
    now = first - minutes(5)
    view = market.view(now)
    forecaster = SpreadAR1Forecaster()
    forecaster.reset(timing)
    forecaster.prepare(view, now, first)
    assert 0 < forecaster.rho < 1

    n = published_horizon(view, ts_ns(first), STEP)
    base = DayAheadForecaster()
    base.reset(timing)
    deviation = (
        forecaster.predict(view, now, first, n)
        - base.predict(view, now, first, n)
        - forecaster.mean
    )
    hist = PriceHistory.from_view(view, now - days(1), now, TZ)
    latest = np.clip(hist.spread[hist.latest_rt(ts_ns(now))], -1000, 1000)
    # The latest RT price is two intervals before the one being decided.
    assert deviation[0] == pytest.approx(forecaster.rho**2 * (latest - forecaster.mean))
    assert np.all(np.abs(deviation[1:]) <= np.abs(deviation[:-1]))


def test_gbm_learns_a_systematic_spread(timing):
    """RT runs $60 above DA in the evening and $10 below otherwise; DA alone can't know."""
    rt, da = synthetic_prices(date(2024, 1, 1), 80, seed=11)
    hour_price = da.set_index("interval_start_utc")["price"]
    starts = rt["interval_start_utc"]
    evening = starts.dt.tz_convert(TZ).dt.hour.between(17, 20).to_numpy()
    noise = np.random.default_rng(1).normal(0, 3, len(rt))
    rt = rt.assign(
        price=hour_price.reindex(starts.dt.floor("h")).to_numpy()
        + np.where(evening, 60.0, -10.0)
        + noise
    )
    market = MarketData.from_price_frames("SYN", rt, da, timing)
    gbm = GBMForecaster(train_days=60, num_boost_round=100)
    base = DayAheadForecaster()
    for forecaster in (gbm, base):
        forecaster.reset(timing)

    errors = {"gbm": [], "da": []}
    for day in pd.date_range("2024-03-05", "2024-03-15", freq="2D"):
        first = local(f"{day.date()} 06:00")
        now = first - minutes(5)
        view = market.view(now)
        gbm.prepare(view, now, first)
        n = published_horizon(view, ts_ns(first), STEP)
        actual = market.aligned("rt", pd.date_range(first, periods=n, freq="15min"))
        for name, forecaster in (("gbm", gbm), ("da", base)):
            errors[name].append(np.abs(forecaster.predict(view, now, first, n) - actual))
    assert len(gbm.fit_times) == 1  # one monthly refit (March)
    gbm_mae, da_mae = (np.concatenate(errors[k]).mean() for k in ("gbm", "da"))
    assert gbm_mae < 0.3 * da_mae
