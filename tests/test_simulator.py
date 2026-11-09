import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from batt.config import BatterySpec
from batt.data.store import align
from batt.evaluate import compare
from batt.optimize import optimize_dispatch
from batt.policies import PerfectForesightDA, PerfectForesightRT, Policy, make_policy
from batt.sim import apply_physics, simulate


class Scripted(Policy):
    """Replays a fixed list of requested powers."""

    name = "scripted"

    def __init__(self, requests):
        self.requests = list(requests)

    def decide(self, ctx):
        return self.requests[ctx.index]


def test_physics_blocks_discharging_empty_and_charging_full(spec):
    c, d, _ = apply_physics(1e9, spec.soc_min_mwh, spec, 0.25)
    assert (c, d) == (0.0, 0.0)
    c, d, _ = apply_physics(-1e9, spec.soc_max_mwh, spec, 0.25)
    assert (c, d) == (0.0, 0.0)


def test_physics_applies_efficiency(spec):
    c, d, soc = apply_physics(50, 100, spec, 0.25)
    assert d == 50
    assert soc == pytest.approx(100 - 50 * 0.25 / spec.eta_discharge)


@settings(max_examples=60, deadline=None)
@given(st.lists(st.floats(-300, 300, allow_nan=False), min_size=1, max_size=200))
def test_soc_bounds_and_energy_conservation_hold_for_any_requests(requests):
    spec = BatterySpec()
    soc, stored_in, stored_out = spec.soc_initial_mwh, 0.0, 0.0
    for r in requests:
        c, d, soc_next = apply_physics(r, soc, spec, 0.25)
        assert c == 0 or d == 0
        assert 0 <= c <= spec.power_mw and 0 <= d <= spec.power_mw
        assert spec.soc_min_mwh - 1e-9 <= soc_next <= spec.soc_max_mwh + 1e-9
        stored_in += spec.eta_charge * c * 0.25
        stored_out += d / spec.eta_discharge * 0.25
        soc = soc_next
    assert soc - spec.soc_initial_mwh == pytest.approx(stored_in - stored_out, abs=1e-6)


def test_ledger_accounting_identities(market, grid, spec, timing):
    result = simulate(make_policy("da_informed"), market, grid, spec, timing)
    ledger = result.ledger
    assert np.allclose(ledger.revenue_energy, ledger.revenue_da + ledger.revenue_rt)
    assert np.allclose(ledger.revenue_net, ledger.revenue_energy - ledger.degradation_cost)
    assert np.allclose(ledger.soc_end_mwh.to_numpy()[:-1], ledger.soc_start_mwh.to_numpy()[1:])
    stored = (spec.eta_charge * ledger.charge_mwh - ledger.discharge_mwh / spec.eta_discharge).sum()
    assert ledger.soc_end_mwh.iat[-1] - spec.soc_initial_mwh == pytest.approx(stored, abs=1e-6)


def test_clipping_is_recorded(market, grid, spec, timing):
    result = simulate(Scripted([spec.power_mw] * len(grid)), market, grid, spec, timing)
    ledger = result.ledger
    assert ledger.clipped.any()  # the battery runs empty and can't keep discharging
    assert ledger.soc_end_mwh.min() >= spec.soc_min_mwh - 1e-9


def test_perfect_foresight_matches_its_lp(market, grid, spec, timing):
    result = simulate(PerfectForesightRT(window="week"), market, grid, spec, timing)
    prices = align(market.view(None).series("rt"), grid.starts, 15)
    expected = 0.0
    soc = spec.soc_initial_mwh
    for window in grid.windows("week"):
        sol = optimize_dispatch(
            prices[window],
            spec,
            grid.dt_hours,
            soc,
            spec.soc_initial_mwh,
            cycle_groups=grid.day_id[window],
        )
        expected += sol.objective
        soc = sol.soc_mwh[-1]
    assert result.ledger.revenue_net.sum() == pytest.approx(expected, rel=1e-6)
    assert not result.ledger.clipped.any()


def test_perfect_foresight_bounds_realistic_policies(market, grid, spec, timing):
    best = simulate(PerfectForesightRT(window="month"), market, grid, spec, timing)
    for name in ("naive", "da_informed"):
        other = simulate(make_policy(name), market, grid, spec, timing)
        assert other.ledger.revenue_net.sum() <= best.ledger.revenue_net.sum() + 1e-6


def test_perfect_da_settles_entirely_at_da_prices(market, grid, spec, timing):
    result = simulate(PerfectForesightDA(), market, grid, spec, timing)
    ledger = result.ledger
    assert np.allclose(ledger.revenue_rt, 0, atol=1e-6)
    assert ledger.revenue_da.sum() > 0
    assert np.allclose(ledger.net_mw, ledger.da_position_mw, atol=1e-6)


def test_compare_reports_capture_rates_and_baseline_cis(market, grid, spec, timing):
    results = [
        simulate(make_policy(n), market, grid, spec, timing)
        for n in ("perfect_rt", "naive", "da_informed")
    ]
    table = compare(results, benchmark="perfect_rt", baseline="naive", n_boot=200)
    assert table.at["perfect_rt", "capture_rate"] == pytest.approx(1.0)
    assert 0 < table.at["da_informed", "capture_rate"] < 1
    low = table.at["da_informed", "vs_naive_ci_low"]
    high = table.at["da_informed", "vs_naive_ci_high"]
    assert low <= table.at["da_informed", "vs_naive_usd_kw_yr"] <= high
