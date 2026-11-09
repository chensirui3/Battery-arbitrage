import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from batt.config import BatterySpec
from batt.optimize import optimize_dispatch

DT = 0.25  # hours


def make_spec(**overrides) -> BatterySpec:
    params = dict(
        power_mw=100,
        energy_mwh=200,
        round_trip_efficiency=0.81,  # 0.9 each way
        soc_min_frac=0.0,
        soc_max_frac=1.0,
        soc_initial_frac=0.0,
        degradation_cost_per_mwh=0.0,
    )
    params.update(overrides)
    return BatterySpec(**params)


def test_two_price_day_matches_hand_solution():
    spec = make_spec()
    prices = np.r_[np.full(48, 10.0), np.full(48, 100.0)]
    sol = optimize_dispatch(prices, spec, DT, soc_initial_mwh=0, soc_terminal_mwh=0)
    # Fill 200 MWh of storage while cheap (drawing 200/0.9 MWh), then deliver 200*0.9 MWh.
    expected = 200 * 0.9 * 100 - 200 / 0.9 * 10
    assert sol.objective == pytest.approx(expected, rel=1e-6)
    assert sol.charge_mw[48:].sum() == pytest.approx(0, abs=1e-6)
    assert sol.discharge_mw[:48].sum() == pytest.approx(0, abs=1e-6)


def test_flat_prices_do_nothing():
    spec = make_spec(round_trip_efficiency=0.85, degradation_cost_per_mwh=5, soc_initial_frac=0.5)
    sol = optimize_dispatch(np.full(96, 40.0), spec, DT, 100, 100)
    assert np.allclose(sol.net_mw, 0)
    assert sol.objective == pytest.approx(0, abs=1e-6)


def test_spread_below_efficiency_breakeven_does_nothing():
    spec = make_spec(round_trip_efficiency=0.85)
    prices = np.r_[np.full(48, 50.0), np.full(48, 55.0)]  # 55 * 0.85 < 50
    sol = optimize_dispatch(prices, spec, DT, 0, 0)
    assert np.allclose(sol.net_mw, 0)


def test_degradation_cost_can_make_a_spread_unprofitable():
    prices = np.r_[np.full(48, 20.0), np.full(48, 40.0)]
    cheap = optimize_dispatch(prices, make_spec(), DT, 0, 0)
    costly = optimize_dispatch(prices, make_spec(degradation_cost_per_mwh=20), DT, 0, 0)
    assert cheap.objective > 0
    assert np.allclose(costly.net_mw, 0)


def test_terminal_soc_is_respected():
    spec = make_spec(soc_initial_frac=0.5)
    prices = np.random.default_rng(1).normal(40, 30, 96)
    sol = optimize_dispatch(prices, spec, DT, 100, 100)
    assert sol.soc_mwh[-1] == pytest.approx(100, abs=1e-6)


def test_unreachable_terminal_soc_moves_to_nearest_reachable():
    spec = make_spec()
    # One hour at 100 MW can store at most 100 * 0.9 = 90 MWh.
    sol = optimize_dispatch(np.full(4, 30.0), spec, DT, 0, 200)
    assert sol.soc_mwh[-1] == pytest.approx(90, abs=1e-6)


def test_negative_prices_never_charge_and_discharge_together():
    spec = make_spec(soc_initial_frac=1.0)
    prices = np.full(16, -200.0)
    lp = optimize_dispatch(prices, spec, DT, 200, 200, exclusive="lp")
    assert lp.has_simultaneous  # the artifact the MILP fallback exists to remove
    auto = optimize_dispatch(prices, spec, DT, 200, 200, exclusive="auto")
    assert auto.used_milp
    assert not auto.has_simultaneous
    assert auto.objective <= lp.objective + 1e-6


def test_cycle_cap_limits_daily_discharge():
    block = np.r_[np.full(12, 5.0), np.full(12, 150.0)]
    prices = np.tile(block, 4)  # room for several cycles in one day
    capped_spec = make_spec(max_cycles_per_day=1.0)
    capped = optimize_dispatch(prices, capped_spec, DT, 0, 0, cycle_groups=np.zeros(96))
    uncapped = optimize_dispatch(prices, make_spec(), DT, 0, 0)
    delivered = capped.discharge_mw.sum() * DT
    assert delivered <= 0.9 * 200 + 1e-6  # one cycle delivers eta_d * usable energy
    assert uncapped.discharge_mw.sum() * DT > delivered + 1


@settings(max_examples=40, deadline=None)
@given(
    prices=st.lists(st.floats(-250, 3_000, allow_nan=False), min_size=4, max_size=96),
    start_frac=st.floats(0, 1),
)
def test_random_prices_give_feasible_nonnegative_schedules(prices, start_frac):
    spec = make_spec(round_trip_efficiency=0.85, degradation_cost_per_mwh=8)
    start = start_frac * 200
    sol = optimize_dispatch(np.array(prices), spec, DT, start, start)
    assert sol.objective >= -1e-6  # doing nothing is feasible and earns zero
    assert not sol.has_simultaneous
    assert np.all(sol.charge_mw <= 100 + 1e-9)
    assert np.all(sol.discharge_mw <= 100 + 1e-9)
    replay = start + np.cumsum(
        (spec.eta_charge * sol.charge_mw - sol.discharge_mw / spec.eta_discharge) * DT
    )
    assert np.all(replay >= -1e-4)
    assert np.all(replay <= 200 + 1e-4)
    assert np.allclose(replay, sol.soc_mwh, atol=1e-4)
