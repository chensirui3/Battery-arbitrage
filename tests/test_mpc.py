"""Warm-started dispatch solver and model-predictive control policies."""

import numpy as np
import pandas as pd
import pytest

from batt.config import BatterySpec
from batt.data.store import MarketData
from batt.evaluate import compare
from batt.optimize.lp import optimize_dispatch
from batt.optimize.rolling import RollingDispatchSolver
from batt.policies import make_policy
from batt.sim import simulate

DT = 0.25


def test_rolling_solver_matches_a_fresh_solve(spec):
    solver = RollingDispatchSolver(spec, DT, capacity=48)
    rng = np.random.default_rng(3)
    # Shrinking and jumping horizons as in MPC, including one longer than the model.
    for n in (40, 39, 38, 120, 119, 30, 2, 1, 96):
        prices = 40 + 30 * np.sin(np.arange(n) / 10 + rng.uniform(0, 6)) + rng.normal(0, 8, n)
        soc0 = rng.uniform(0, spec.energy_mwh)
        warm = solver.solve(prices, soc0, spec.soc_initial_mwh)
        fresh = optimize_dispatch(prices, spec, DT, soc0, spec.soc_initial_mwh, exclusive="lp")
        assert warm.objective == pytest.approx(fresh.objective, rel=1e-9, abs=1e-6)
        assert warm.soc_mwh.shape == (n,)
        # The schedule is feasible: SoC follows from the actions and stays in bounds.
        replay = soc0 + np.cumsum(
            (spec.eta_charge * warm.charge_mw - warm.discharge_mw / spec.eta_discharge) * DT
        )
        np.testing.assert_allclose(warm.soc_mwh, replay, atol=1e-4)
        assert warm.soc_mwh.min() >= -1e-6 and warm.soc_mwh.max() <= spec.energy_mwh + 1e-6
    assert solver.fallbacks == 0


def test_rolling_solver_never_charges_and_discharges_at_once(spec):
    # Deeply negative prices make the LP burn energy through losses; the MILP may not.
    prices = np.r_[np.full(8, -300.0), np.full(8, 50.0)]
    lp = optimize_dispatch(prices, spec, DT, spec.energy_mwh, spec.soc_initial_mwh, exclusive="lp")
    assert lp.has_simultaneous
    solver = RollingDispatchSolver(spec, DT)
    solution = solver.solve(prices, spec.energy_mwh, spec.soc_initial_mwh)
    assert solution.used_milp and not solution.has_simultaneous
    assert solver.fallbacks == 1


def test_rolling_solver_uses_the_full_model_for_cycle_caps():
    capped = BatterySpec(max_cycles_per_day=1.0, degradation_cost_per_mwh=0.0)
    prices = np.tile([10.0, 10.0, 90.0, 90.0], 24)
    solution = RollingDispatchSolver(capped, DT).solve(prices, 100.0, 100.0)
    expected = optimize_dispatch(prices, capped, DT, 100.0, 100.0)
    assert solution.objective == pytest.approx(expected.objective)
    assert solution.discharge_mw.sum() * DT <= capped.eta_discharge * capped.usable_mwh + 1e-6


def test_mpc_with_perfect_forecasts_nearly_matches_perfect_foresight(market, grid, spec, timing):
    results = [
        simulate(make_policy(name), market, grid, spec, timing)
        for name in ("perfect_rt", "mpc_oracle", "mpc_da", "da_informed")
    ]
    table = compare(results, benchmark="perfect_rt", n_boot=50)
    assert table.at["mpc_oracle", "capture_rate"] > 0.9
    assert table.at["mpc_oracle", "capture_rate"] > table.at["mpc_da", "capture_rate"]
    for result in results:
        ledger = result.ledger
        assert not ledger["clipped"].any()
        assert ledger["soc_end_mwh"].between(-1e-6, spec.energy_mwh + 1e-6).all()


def test_mpc_replans_less_often_when_asked(market, grid, spec, timing):
    every_interval = simulate(make_policy("mpc_ar1"), market, grid, spec, timing)
    hourly = simulate(make_policy("mpc_ar1", {"replan_every": 4}), market, grid, spec, timing)
    assert not hourly.ledger["clipped"].any()
    assert not np.allclose(every_interval.ledger["requested_mw"], hourly.ledger["requested_mw"])


def test_mpc_idles_on_days_without_day_ahead_prices(frames, grid, spec, timing):
    rt, da = frames
    gap = pd.Timestamp("2024-03-12").date()
    da_day = da["interval_start_utc"].dt.tz_convert(timing.timezone).dt.date
    market = MarketData.from_price_frames("SYN_HUB", rt, da[da_day != gap], timing)
    result = simulate(make_policy("mpc_da"), market, grid, spec, timing)
    on_gap = (result.ledger["local_date"] == pd.Timestamp(gap)).to_numpy()
    assert (result.ledger.loc[on_gap, "requested_mw"] == 0).all()
    assert (result.ledger.loc[~on_gap, "requested_mw"] != 0).any()
    assert result.notes["days_without_forecast"] == [gap]
