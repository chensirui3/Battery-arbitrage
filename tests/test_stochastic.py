"""Scenario forecasts, probabilistic scoring, and dispatch under uncertainty."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from batt.config import BatterySpec
from batt.data.store import MarketData, ts_ns
from batt.data.synthetic import synthetic_prices
from batt.forecast import (
    AnalogForecaster,
    AnchoredAnalogForecaster,
    DayAheadForecaster,
    OracleForecaster,
    PriceHistory,
    QuantileGBMForecaster,
    SpreadMarkovChain,
    published_horizon,
)
from batt.forecast.evaluate import crps
from batt.forecast.history import MINUTE_NS
from batt.forecast.quantile import conformal_offset
from batt.optimize.lp import optimize_dispatch
from batt.optimize.scenario import ScenarioDispatchSolver
from batt.policies import AnalogScenarioMPC, StochasticDP, make_policy
from batt.sim import SimulationGrid, simulate
from batt.timeutils import days, minutes
from helpers import corrupt_after

TZ = "America/Chicago"
DT = 0.25
STEP = 15 * MINUTE_NS


def local(text: str) -> pd.Timestamp:
    return pd.Timestamp(text).tz_localize(TZ).tz_convert("UTC")


def scenario_prices(rng, n_scenarios: int, n: int) -> np.ndarray:
    base = 40 + 30 * np.sin(np.arange(n) / 12 + rng.uniform(0, 6))
    spikes = (rng.random((n_scenarios, n)) < 0.03) * rng.uniform(200, 2000, (n_scenarios, n))
    return base + rng.normal(0, 20, (n_scenarios, n)) + spikes


# --- Scenario LP --------------------------------------------------------------------------


def test_one_scenario_without_hourly_steps_matches_the_lp(spec):
    solver = ScenarioDispatchSolver(spec, DT, 1, fine_steps=500, capacity=150)
    rng = np.random.default_rng(0)
    for n in (1, 2, 17, 96, 140):
        prices = 40 + 30 * np.sin(np.arange(n) / 9) + rng.normal(0, 10, n)
        soc0 = rng.uniform(0, spec.energy_mwh)
        mine = solver.solve(prices, soc0, spec.soc_initial_mwh)
        lp = optimize_dispatch(prices, spec, DT, soc0, spec.soc_initial_mwh, exclusive="lp")
        assert mine.expected_profit == pytest.approx(lp.objective, rel=1e-9, abs=1e-6)


def test_stochastic_programming_bounds(spec):
    """Committing to the average-scenario plan <= the stochastic plan <= perfect information."""
    rng = np.random.default_rng(1)
    n_scen = 8
    stochastic = ScenarioDispatchSolver(spec, DT, n_scen)
    single = ScenarioDispatchSolver(spec, DT, 1)
    for _ in range(6):
        n = int(rng.integers(20, 140))
        prices = scenario_prices(rng, n_scen, n)
        soc0 = rng.uniform(0, spec.energy_mwh)
        rp = stochastic.solve(prices, soc0, spec.soc_initial_mwh).expected_profit
        ws = np.mean([single.solve(p, soc0, spec.soc_initial_mwh).expected_profit for p in prices])
        ev = single.solve(prices.mean(axis=0), soc0, spec.soc_initial_mwh)
        eev = stochastic.solve(prices, soc0, spec.soc_initial_mwh, first_mw=ev.first_net_mw)
        assert eev.expected_profit <= rp + 1e-6
        assert rp <= ws + 1e-6


def test_one_shared_plan_is_planning_on_the_average(spec):
    prices = scenario_prices(np.random.default_rng(2), 5, 60)
    shared = ScenarioDispatchSolver(spec, DT, 5, shared_steps=10_000)
    average = ScenarioDispatchSolver(spec, DT, 1)
    assert shared.solve(prices, 80.0, 100.0).expected_profit == pytest.approx(
        average.solve(prices.mean(axis=0), 80.0, 100.0).expected_profit
    )


def test_hourly_steps_lose_nothing_when_prices_are_hourly(spec):
    hourly = np.repeat(np.random.default_rng(3).normal(50, 30, 30), 4)  # 120 intervals
    coarse = ScenarioDispatchSolver(spec, DT, 1, fine_steps=16)
    fine = ScenarioDispatchSolver(spec, DT, 1, fine_steps=500)
    assert coarse.solve(hourly, 100.0, 100.0).expected_profit == pytest.approx(
        fine.solve(hourly, 100.0, 100.0).expected_profit, rel=1e-6
    )


# --- Analog scenarios ---------------------------------------------------------------------


def _analog_setup(market, timing, cls=AnalogForecaster, when="2024-03-20 15:00"):
    forecaster = cls(n_scenarios=5, pool_days=40)
    forecaster.reset(timing)
    first = local(when)
    now = first - minutes(5)
    view = market.view(now)
    forecaster.prepare(view, now, first)
    n = published_horizon(view, ts_ns(first), STEP)
    return forecaster, first, now, view, n


def test_analog_scenarios_replay_past_spreads_at_the_same_local_time(market, timing):
    forecaster, first, now, view, n = _analog_setup(market, timing)
    paths = forecaster.scenarios(view, now, first, n)
    base = DayAheadForecaster()
    base.reset(timing)
    spreads = paths - base.predict(view, now, first, n)

    starts = pd.date_range(first, periods=n, freq="15min")
    local_starts = starts.tz_convert(TZ).tz_localize(None)
    today = local_starts[0].normalize()
    pool = forecaster._pool
    for s, row in enumerate(forecaster._choice):
        analog_day = pd.Timestamp("1970-01-01") + days(pool.first_day + int(row))
        # Same wall-clock times, shifted back to the analog day (across the DST change).
        then = (local_starts - (today - analog_day)).tz_localize(TZ).tz_convert("UTC")
        expected = market.aligned("rt", then) - market.aligned("da", then)
        np.testing.assert_allclose(spreads[s], expected, atol=1e-9)
        assert analog_day + days(2) <= today  # its next day was complete before today


def test_analog_scenarios_use_only_published_data(market, timing):
    paths = []
    for data in (market, corrupt_after(market, local("2024-03-20 14:55"))):
        forecaster, first, now, view, n = _analog_setup(data, timing)
        paths.append(forecaster.scenarios(view, now, first, n))
    np.testing.assert_array_equal(*paths)


def test_anchored_scenarios_fade_from_the_latest_spread(market, timing):
    plain, first, now, view, n = _analog_setup(market, timing, when="2024-03-20 18:00")
    anchored, *_ = _analog_setup(market, timing, AnchoredAnalogForecaster, "2024-03-20 18:00")
    assert 0 < anchored.rho < 1
    np.testing.assert_array_equal(plain._choice, anchored._choice)
    shift = anchored.scenarios(view, now, first, n) - plain.scenarios(view, now, first, n)
    big = np.abs(shift[:, :-1]) > 1e-6
    np.testing.assert_allclose((shift[:, 1:] / np.where(big, shift[:, :-1], 1))[big], anchored.rho)


# --- Probabilistic scores and quantile forecasts --------------------------------------------


def test_crps_matches_its_definition():
    rng = np.random.default_rng(4)
    samples, actual = rng.normal(size=(7, 5)), rng.normal(size=5)
    expected = np.abs(samples - actual).mean(axis=0) - 0.5 * np.abs(
        samples[:, None, :] - samples[None, :, :]
    ).mean(axis=(0, 1))
    np.testing.assert_allclose(crps(samples, actual), expected)
    np.testing.assert_allclose(crps(actual[None, :], actual + 3), 3.0)  # point forecast: |error|


def test_conformal_offset_hits_the_level():
    residuals = np.random.default_rng(5).normal(size=2000)
    for level in (0.05, 0.5, 0.95):
        assert np.mean(residuals <= conformal_offset(residuals, level)) == pytest.approx(
            level, abs=0.002
        )


def test_quantile_forecasts_are_calibrated(timing):
    """RT = DA + noise whose size depends on the hour; 5%-95% ranges should cover ~90%."""
    rt, da = synthetic_prices(date(2024, 1, 1), 80, seed=4)
    starts = rt["interval_start_utc"]
    evening = starts.dt.tz_convert(TZ).dt.hour.between(16, 21).to_numpy()
    noise = np.random.default_rng(6).normal(0, 1, len(rt)) * np.where(evening, 40.0, 8.0)
    hourly = da.set_index("interval_start_utc")["price"]
    rt = rt.assign(price=hourly.reindex(starts.dt.floor("h")).to_numpy() + noise)
    market = MarketData.from_price_frames("SYN", rt, da, timing)
    forecaster = QuantileGBMForecaster(
        train_days=50, calib_days=14, num_boost_round=60, params={"num_threads": 1}
    )
    forecaster.reset(timing)

    inside, widths = [], {True: [], False: []}
    for day in pd.date_range("2024-03-05", "2024-03-18"):
        first = local(f"{day.date()} 06:00")
        now = first - minutes(5)
        view = market.view(now)
        forecaster.prepare(view, now, first)
        n = published_horizon(view, ts_ns(first), STEP)
        q = forecaster.quantiles(view, now, first, n)
        assert np.all(np.diff(q, axis=0) >= 0)  # quantiles never cross
        starts_ = pd.date_range(first, periods=n, freq="15min")
        actual = market.aligned("rt", starts_)
        inside.append((actual >= q[0]) & (actual <= q[-1]))
        late = starts_.tz_convert(TZ).hour.isin(range(16, 22))
        widths[True].append((q[-1] - q[0])[late])
        widths[False].append((q[-1] - q[0])[~late])
    assert len(forecaster.fit_times) == 1
    assert 0.83 < np.concatenate(inside).mean() < 0.97
    # It learned where prices are uncertain: evening ranges are much wider.
    assert np.concatenate(widths[True]).mean() > 3 * np.concatenate(widths[False]).mean()


def test_partial_oracle_knows_only_the_next_hours(market, timing):
    forecaster = OracleForecaster(hours=1)
    forecaster.reset(timing)
    first = local("2024-03-12 15:00")
    now = first - minutes(5)
    n = published_horizon(market.view(now), ts_ns(first), STEP)
    prices = forecaster.predict(market.view(None), now, first, n)
    starts = pd.date_range(first, periods=n, freq="15min")
    np.testing.assert_array_equal(prices[:4], market.aligned("rt", starts[:4]))
    np.testing.assert_array_equal(prices[4:], market.aligned("da", starts[4:]))


# --- Stochastic dynamic programming -------------------------------------------------------


def test_markov_chain_is_a_proper_model(market):
    hist = PriceHistory.from_view(market.view(None), local("2024-02-01"), local("2024-04-01"), TZ)
    chain = SpreadMarkovChain.fit(hist)
    assert chain.transition.shape == (24, 7, 7)
    np.testing.assert_allclose(chain.transition.sum(axis=2), 1.0)
    assert (chain.transition >= 0).all()
    assert list(chain.state([-100, 0, 30, 5000])) == [0, 2, 3, 6]


class CertainDP(StochasticDP):
    """SDP whose spread model says RT = DA exactly: no uncertainty left."""

    name = "sdp_certain"

    def fit_chain(self, hist):
        return SpreadMarkovChain(np.zeros(0), np.zeros((24, 1)), np.ones((24, 1, 1)))


def test_dynamic_programming_without_uncertainty_matches_the_lp(spec, timing):
    policy = CertainDP(soc_levels=81, action_levels=33)
    grid = SimulationGrid.for_days(date(2024, 3, 4), date(2024, 3, 5), TZ)
    policy.reset(grid, spec, timing)
    policy.chain = policy.fit_chain(None)
    da = 40 + 35 * np.sin(np.arange(96) / 96 * 2 * np.pi * 2) ** 3
    policy._solve(da, np.zeros(98, dtype=int))
    start = int(np.searchsorted(policy._grid_e, spec.soc_initial_mwh))
    lp = optimize_dispatch(da, spec, DT, spec.soc_initial_mwh, spec.soc_initial_mwh)
    assert policy.value[start, 0] == pytest.approx(lp.objective, rel=0.02)
    assert policy.value[start, 0] <= lp.objective + 1e-6


def test_dynamic_programming_without_uncertainty_trades_like_mpc(market, grid, spec, timing):
    certain = simulate(CertainDP(), market, grid, spec, timing)
    mpc = simulate(make_policy("mpc_da"), market, grid, spec, timing)
    a, b = certain.ledger.revenue_net.sum(), mpc.ledger.revenue_net.sum()
    assert a == pytest.approx(b, rel=0.03)
    assert not certain.ledger.clipped.any()


# --- Policy variants -----------------------------------------------------------------------


def test_policy_variants_are_built_by_name():
    policy = make_policy("smpc_1h", {"policy": "smpc_analog", "shared_hours": 1})
    assert isinstance(policy, AnalogScenarioMPC)
    assert policy.name == "smpc_1h" and policy.shared_hours == 1
    oracle = make_policy(
        "oracle_1h",
        {"policy": "mpc_oracle", "forecaster_params": {"hours": 1}},
        forecasters={"oracle": {}},
    )
    assert oracle.is_oracle and oracle.forecaster.hours == 1
    tuned = make_policy("smpc_analog", forecasters={"analog": {"n_scenarios": 7}})
    assert tuned.forecaster.n_scenarios == 7
    with pytest.raises(KeyError):
        make_policy("no_such_policy")
    with pytest.raises(KeyError):
        make_policy("variant", {"policy": "no_such_policy"})


def test_stochastic_mpc_respects_battery_limits(market, grid, spec, timing):
    result = simulate(make_policy("smpc_analog"), market, grid, spec, timing)
    assert not result.ledger.clipped.any()
    assert result.ledger.soc_end_mwh.between(-1e-6, spec.energy_mwh + 1e-6).all()
    assert result.ledger.requested_mw.abs().max() <= spec.power_mw + 1e-6


def test_uncertainty_planners_refuse_cycle_caps_they_cannot_model(grid, timing):
    capped = BatterySpec(max_cycles_per_day=1.0)
    with pytest.raises(ValueError, match="cycle"):
        ScenarioDispatchSolver(capped, DT, 3)
    with pytest.raises(ValueError, match="cycle"):
        StochasticDP().reset(grid, capped, timing)
