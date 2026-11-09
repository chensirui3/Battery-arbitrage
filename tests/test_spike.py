"""Next-hour spike nowcasting: features, training sets, forecasts, benchmarks, and scores."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from batt.data.store import MarketData, ts_ns
from batt.data.synthetic import synthetic_prices
from batt.forecast import (
    DayAheadForecaster,
    PriceHistory,
    SpikeForecaster,
    evaluate_nowcasts,
    nowcast_scores,
    published_horizon,
)
from batt.forecast.history import DAY_NS, MINUTE_NS, NEVER
from batt.forecast.spike import (
    PRIOR_FEATURES,
    SPIKE_FEATURES,
    auc,
    average_precision,
    nowcast_targets,
    spike_history,
)
from batt.policies import make_policy
from batt.sim import SimulationGrid, simulate
from batt.timeutils import minutes
from helpers import corrupt_after

TZ = "America/Chicago"
STEP = 15 * MINUTE_NS
LEAD = 5 * MINUTE_NS
SMALL = {"train_days": 60, "num_boost_round": 40, "min_positives": 50}


def local(text: str) -> pd.Timestamp:
    return pd.Timestamp(text).tz_localize(TZ).tz_convert("UTC")


@pytest.fixture(scope="module")
def spiky(timing) -> MarketData:
    """Synthetic prices plus two-hour, $800 plateaus on a quarter of the afternoons, which the
    DA prices do not anticipate: spikes that persist, so recent RT prices predict them."""
    rt, da = synthetic_prices(date(2024, 1, 1), 120, seed=11)
    rng = np.random.default_rng(11)
    start = rt["interval_start_utc"].dt.tz_convert(TZ)
    day = (start.dt.tz_localize(None).dt.normalize() - pd.Timestamp("2024-01-01")).dt.days
    hour = start.dt.hour + start.dt.minute / 60
    price = rt["price"].to_numpy().copy()
    for d in np.flatnonzero(rng.random(120) < 0.25):
        begin = rng.uniform(13, 18)
        price[((day == d) & (hour >= begin) & (hour < begin + 2)).to_numpy()] += 800
    return MarketData.from_price_frames("SYN_HUB", rt.assign(price=price), da, timing)


def test_spike_history_counts_published_spikes():
    n = 3 * 96
    origin = ts_ns(local("2024-03-04 00:00"))
    spread = np.zeros(n)
    spread[[10, 100, 150, 200]] = [150.0, 120.0, -130.0, 99.0]  # 99 is not a spike
    da = np.full(n, 30.0)
    starts = origin + np.arange(n) * STEP
    hist = PriceHistory(
        origin, STEP, TZ, da + spread, starts + 20 * MINUTE_NS, da, np.full(n, origin - DAY_NS)
    )
    issue = starts[250] + 20 * MINUTE_NS  # interval 250 just published
    out = spike_history(hist, [issue], 100.0)[0]
    window = slice(250 - 95, 251)  # the last 24 hours: intervals 155..250
    assert out[SPIKE_FEATURES.index("up_24h")] == pytest.approx(0 / 96)
    assert out[SPIKE_FEATURES.index("down_24h")] == pytest.approx(0 / 96)
    assert np.sum(spread[window] > 100) == 0
    assert out[SPIKE_FEATURES.index("up_30d")] == pytest.approx(2 / 251)
    assert out[SPIKE_FEATURES.index("down_30d")] == pytest.approx(1 / 251)
    # Nothing published yet: unknown, not zero.
    assert np.isnan(spike_history(hist, [origin], 100.0)).all()
    # Unpublished values never count.
    late = hist.rt_publish.copy()
    late[150] = NEVER
    hidden = PriceHistory(origin, STEP, TZ, hist.rt, late, hist.da, hist.da_publish)
    hidden.rt[150] = np.nan
    assert spike_history(hidden, [issue], 100.0)[0][3] == 0.0


def test_nowcast_targets_pair_each_decision_with_the_hour_it_starts(market, timing):
    hist = PriceHistory.from_view(market.view(None), local("2024-03-04"), local("2024-03-06"), TZ)
    first, end = ts_ns(local("2024-03-04 12:00")), ts_ns(local("2024-03-04 14:00"))
    issue, target = nowcast_targets(hist, LEAD, first, end, 4)
    assert issue.size == 8 * 4
    decided = (issue + LEAD - hist.origin_ns) // STEP
    np.testing.assert_array_equal(target - decided, np.tile(np.arange(4), 8))
    assert (hist.start_ns(target) >= issue + LEAD).all()


def test_nowcast_features_never_use_unpublished_data(market, timing):
    """Features from all data must equal features from only what was public at issue time."""
    forecaster = SpikeForecaster()
    forecaster.reset(timing)
    full = PriceHistory.from_view(market.view(None), local("2024-02-01"), local("2024-04-10"), TZ)
    rng = np.random.default_rng(1)
    firsts = [full.start_ns(k) for k in rng.integers(36 * 96, len(full) - 3 * 96, size=20)]
    for first in firsts:
        issue = first - LEAD
        view = market.view(pd.Timestamp(issue, tz="UTC"))
        live_hist, k = forecaster._history(view, issue, first, 4)  # what predict() uses
        live = forecaster.features(live_hist, issue, k)
        batch = forecaster.features(full, issue, (first - full.origin_ns) // STEP + np.arange(4))
        np.testing.assert_allclose(live, batch, rtol=1e-9, atol=1e-6, equal_nan=True)
        names = forecaster.feature_names
        assert np.isfinite(live[:, [names.index(n) for n in SPIKE_FEATURES]]).all()
        assert np.isfinite(live[:, forecaster.prior_columns]).all()


def test_training_set_uses_only_spreads_published_by_the_fit(market, timing):
    forecaster = SpikeForecaster()
    forecaster.reset(timing)
    cutoff = local("2024-03-10 12:00")
    x, y = forecaster.training_set(market.view(cutoff), cutoff)
    x_noisy, y_noisy = forecaster.training_set(corrupt_after(market, cutoff).view(cutoff), cutoff)
    assert y.size > 30 * 96 * 4 - 10
    np.testing.assert_array_equal(x, x_noisy)
    np.testing.assert_array_equal(y, y_noisy)
    assert set(PRIOR_FEATURES) <= set(forecaster.feature_names)


def test_without_a_fitted_model_mpc_spike_is_mpc_da(market, grid, spec, timing):
    params = {"forecaster_params": {"min_positives": 10**9}}
    spike = simulate(make_policy("mpc_spike", params), market, grid, spec, timing)
    da = simulate(make_policy("mpc_da"), market, grid, spec, timing)
    np.testing.assert_array_equal(spike.ledger.requested_mw, da.ledger.requested_mw)


def test_nowcast_learns_that_spikes_persist(spiky, timing):
    frame = evaluate_nowcasts(
        spiky, SpikeForecaster(**SMALL), date(2024, 3, 4), date(2024, 3, 25), timing
    )
    spread = pd.Series(
        spiky.aligned("rt", pd.DatetimeIndex(frame["target_utc"].unique()))
        - spiky.aligned("da", pd.DatetimeIndex(frame["target_utc"].unique())),
        index=frame["target_utc"].unique(),
    )
    # The latest RT price public at each issue time is two intervals before the decided one.
    latest = frame["target_utc"] - pd.to_timedelta((frame["step"] + 2) * 15, unit="min")
    after_spike = latest.map(spread).to_numpy() > 100
    assert after_spike.sum() > 50
    # A plateau seen 2-5 intervals earlier is still on about half the time.
    news = frame["p_up"] - frame["prior_up"]
    assert frame.loc[after_spike, "p_up"].mean() > 0.35
    assert frame.loc[~after_spike, "p_up"].mean() < 0.05
    assert news[after_spike].mean() > 0.3
    scores = nowcast_scores(frame, 100.0)["up"]
    assert scores.at["all", "auc"] > scores.at["all", "auc_prior"]
    assert scores.at["all", "brier_skill"] > 0.2


def test_bulk_nowcasts_match_live_ones(spiky, timing):
    frame = evaluate_nowcasts(
        spiky, SpikeForecaster(**SMALL), date(2024, 3, 4), date(2024, 3, 7), timing
    )
    live = SpikeForecaster(**SMALL)
    live.fit_prior = True  # as evaluate_nowcasts does, for the reference scores
    live.reset(timing)
    first = local("2024-03-04 00:00")
    live.prepare(spiky.view(first - minutes(5)), first - minutes(5), first)
    assert live.active["up"]
    issues = frame["issue_utc"].unique()
    for issue in issues[np.random.default_rng(2).choice(len(issues), 15, replace=False)]:
        issue = pd.Timestamp(issue)
        probs = live.nowcast(spiky.view(issue), issue, issue + minutes(5))
        rows = frame[frame["issue_utc"] == issue].sort_values("step")
        for key, values in probs.items():
            np.testing.assert_allclose(rows[key].to_numpy(), values, rtol=1e-9, equal_nan=True)


@pytest.mark.parametrize("adjust", ["raw", "news", "regime"])
def test_forecast_moves_da_for_the_next_hour_only(adjust, spiky, timing):
    forecaster = SpikeForecaster(**SMALL, adjust=adjust)
    forecaster.reset(timing)
    assert (forecaster.fit_prior, forecaster.needs_oracle) == (adjust == "news", False)
    da_model = DayAheadForecaster()
    da_model.reset(timing)
    # Calm times, and the first decision of March whose latest published price is a plateau.
    starts = pd.date_range(local("2024-03-05"), local("2024-03-31"), freq="15min")
    latest = spiky.aligned("rt", starts - minutes(30)) - spiky.aligned("da", starts - minutes(30))
    during = starts[np.argmax(latest > 300)]
    moved = 0
    for when in (during, local("2024-03-06 03:00"), local("2024-03-12 15:00")):
        first = when
        now = first - minutes(5)
        view = spiky.view(now)
        forecaster.prepare(view, now, first)  # fits once, at the first call
        n = published_horizon(view, ts_ns(first), STEP)
        forecast = forecaster.predict(view, now, first, n)
        da = da_model.predict(view, now, first, n)
        np.testing.assert_array_equal(forecast[4:], da[4:])
        probs = forecaster.nowcast(view, now, first)
        latest = spiky.aligned("rt", [first - minutes(30)]) - spiky.aligned(
            "da", [first - minutes(30)]
        )
        expected = da[:4].copy()
        for side, sign in (("up", 1), ("down", -1)):
            if not forecaster.active[side]:
                continue
            p = probs[f"p_{side}"]
            if adjust == "news":
                p = p - probs[f"prior_{side}"]
            if adjust == "regime" and not sign * latest[0] > 100:
                p = 0 * p
            expected += sign * p * forecaster.size[side]
        np.testing.assert_allclose(forecast[:4], expected)
        moved += not np.allclose(forecast[:4], da[:4])
    assert moved >= 1
    assert forecaster.size["up"] == pytest.approx(800, rel=0.15)


def test_spike_oracle_knows_the_next_hours_spikes_and_nothing_else(market, timing):
    forecaster = SpikeForecaster(threshold=15, perfect="spikes")
    forecaster.reset(timing)
    assert forecaster.needs_oracle
    for first in pd.date_range(local("2024-03-06 06:00"), local("2024-03-06 23:00"), freq="15min"):
        now = first - minutes(5)
        n = published_horizon(market.view(now), ts_ns(first), STEP)
        starts = pd.date_range(first, periods=n, freq="15min")
        rt, da = market.aligned("rt", starts), market.aligned("da", starts)
        spike = np.abs(rt - da) > 15
        if spike[:4].any() and not spike[:4].all():
            break
    else:
        pytest.fail("no hour with both spikes and calm intervals")
    forecast = forecaster.predict(market.view(None), now, first, n)
    expected = np.where(np.arange(n) < 4, np.where(spike, rt, da), da)
    np.testing.assert_allclose(forecast, expected)


def test_timing_oracle_knows_when_spikes_come_but_not_how_big(spiky, timing):
    forecaster = SpikeForecaster(**SMALL, perfect="timing")
    forecaster.reset(timing)
    # The first decision of March whose next hour holds a plateau's start.
    starts = pd.date_range(local("2024-03-05"), local("2024-03-31"), freq="15min")
    spikes = spiky.aligned("rt", starts) - spiky.aligned("da", starts) > 100
    first = starts[np.argmax(~spikes[:-3] & spikes[3:])]
    now = first - minutes(5)
    forecaster.prepare(spiky.view(now), now, first)
    assert not forecaster.models and not forecaster.priors  # no classifier to fit
    assert forecaster.active["up"]
    n = published_horizon(spiky.view(now), ts_ns(first), STEP)
    forecast = forecaster.predict(spiky.view(None), now, first, n)
    starts = pd.date_range(first, periods=n, freq="15min")
    spread = spiky.aligned("rt", starts) - spiky.aligned("da", starts)
    expected = spiky.aligned("da", starts)
    for side, sign in (("up", 1), ("down", -1)):
        if forecaster.active[side]:
            expected[:4] += sign * (sign * spread[:4] > 100) * forecaster.size[side]
    assert (spread[:4] > 100).any() and not (spread[:4] > 100).all()
    np.testing.assert_allclose(forecast, expected)


def test_mpc_spike_earns_more_than_mpc_da_when_spikes_persist(spiky, spec, timing):
    grid = SimulationGrid.for_days(date(2024, 3, 4), date(2024, 4, 1), TZ)
    params = {"forecaster_params": SMALL}
    spike = simulate(make_policy("mpc_spike", params), spiky, grid, spec, timing)
    da = simulate(make_policy("mpc_da"), spiky, grid, spec, timing)
    gain = spike.ledger.revenue_net.sum() - da.ledger.revenue_net.sum()
    assert gain > 0.05 * da.ledger.revenue_net.sum()


def test_policies_refuse_the_wrong_kind_of_forecaster():
    with pytest.raises(ValueError):
        make_policy("mpc_spike", {"forecaster_params": {"perfect": "spikes"}})
    assert make_policy("mpc_oracle_spike").forecaster.perfect == "spikes"
    timing_oracle = {"policy": "mpc_oracle_spike", "forecaster_params": {"perfect": "timing"}}
    assert make_policy("mpc_oracle_timing", timing_oracle).forecaster.perfect == "timing"
    with pytest.raises(ValueError):
        SpikeForecaster(perfect="sometimes")


def test_ranking_scores():
    label = np.array([0, 0, 1, 0, 1, 1], dtype=bool)
    assert auc(np.array([0.1, 0.2, 0.9, 0.3, 0.8, 0.7]), label) == 1.0
    assert auc(np.array([0.9, 0.8, 0.1, 0.7, 0.2, 0.3]), label) == 0.0
    assert auc(np.full(6, 0.5), label) == 0.5
    assert average_precision(np.array([0.1, 0.2, 0.9, 0.3, 0.8, 0.7]), label) == 1.0
    # Ranked: hit, miss, hit -> precision 1 and 2/3 at the two hits.
    assert average_precision(np.array([0.9, 0.8, 0.7]), np.array([1, 0, 1], bool)) == pytest.approx(
        (1 + 2 / 3) / 2
    )
    assert np.isnan(auc(np.ones(3), np.ones(3, dtype=bool)))
