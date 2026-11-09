"""Leakage tests: no decision may change when data published after it is corrupted.

For every realistic policy, replace all values published after a cutoff with random noise and
re-run. Real-time requests decided before the cutoff, and day-ahead positions committed
before it, must be identical. The same check must FAIL for the perfect-foresight
benchmarks, which proves the test can detect leakage. Forecasters that learn are refit inside
the simulation, so the same check covers their training data.
"""

import numpy as np
import pandas as pd
import pytest

from batt.forecast import GBMForecaster, SpikeForecaster
from batt.policies import POLICIES, make_policy
from batt.sim import simulate
from batt.sim.simulator import da_decision_times
from batt.timeutils import minutes
from helpers import corrupt_after

REALISTIC = sorted(name for name, cls in POLICIES.items() if not cls.is_oracle)
ORACLES = sorted(name for name, cls in POLICIES.items() if cls.is_oracle)


def decisions_before(result, cutoff, grid, timing):
    lead = minutes(timing.decision_lead_minutes)
    decided = np.asarray((grid.starts - lead) <= cutoff)
    committed = np.zeros(len(grid), dtype=bool)
    for day, t in enumerate(da_decision_times(grid, timing)):
        if t <= cutoff:
            committed[grid.day_slice(day)] = True
    ledger = result.ledger
    return (
        ledger.requested_mw.to_numpy()[decided],
        ledger.da_position_mw.to_numpy()[committed],
    )


@pytest.fixture(scope="module")
def cutoff(grid, timing):
    # Local noon mid-backtest: after that day's DA bid deadline, before its DA results post.
    day = grid.days[len(grid.days) // 2]
    return pd.Timestamp(f"{day} 12:00").tz_localize(timing.timezone).tz_convert("UTC")


@pytest.mark.parametrize("name", REALISTIC)
def test_decisions_never_use_unpublished_data(name, market, grid, spec, timing, cutoff):
    clean = simulate(make_policy(name), market, grid, spec, timing)
    noisy = simulate(make_policy(name), corrupt_after(market, cutoff), grid, spec, timing)
    for a, b in zip(
        decisions_before(clean, cutoff, grid, timing),
        decisions_before(noisy, cutoff, grid, timing),
        strict=True,
    ):
        np.testing.assert_allclose(a, b)


GBM_DAILY = {"refit": "day", "num_boost_round": 25}
# A low threshold and minimum so the spike classifiers really are fit on the test data.
SPIKE_DAILY = {"refit": "day", "num_boost_round": 25, "threshold": 30, "min_positives": 20}
REFIT_DAILY = {
    "mpc_gbm": ("mpc_gbm", GBM_DAILY),
    "mpc_spike": ("mpc_spike", SPIKE_DAILY),
    "mpc_gbm_c": ("mpc_gbm", {**GBM_DAILY, "conditions": True}),
    "mpc_spike_c": ("mpc_spike", {**SPIKE_DAILY, "conditions": True}),
}


@pytest.mark.parametrize("variant", sorted(REFIT_DAILY))
def test_models_refit_daily_never_see_unpublished_data(variant, market, grid, spec, timing, cutoff):
    # Refit every day, so models are also trained on the cutoff day and after it.
    name, settings = REFIT_DAILY[variant]
    params = {"forecaster_params": settings}
    clean_policy = make_policy(name, params)
    clean = simulate(clean_policy, market, grid, spec, timing)
    noisy_market = corrupt_after(market, cutoff)
    noisy = simulate(make_policy(name, params), noisy_market, grid, spec, timing)
    before = decisions_before(clean, cutoff, grid, timing)[0]
    np.testing.assert_allclose(before, decisions_before(noisy, cutoff, grid, timing)[0])
    assert not np.allclose(clean.ledger.requested_mw, noisy.ledger.requested_mw)
    assert len(clean_policy.forecaster.fit_times) == grid.n_days


@pytest.mark.parametrize(
    "forecaster",
    [
        GBMForecaster(),
        SpikeForecaster(),
        GBMForecaster(conditions=True),
        SpikeForecaster(conditions=True),
    ],
)
def test_training_examples_use_only_data_published_by_the_fit(forecaster, market, timing, cutoff):
    forecaster.reset(timing)
    x, y = forecaster.training_set(market.view(cutoff), cutoff)
    x_noisy, y_noisy = forecaster.training_set(corrupt_after(market, cutoff).view(cutoff), cutoff)
    assert y.size > 1000
    np.testing.assert_array_equal(x, x_noisy)
    np.testing.assert_array_equal(y, y_noisy)


@pytest.mark.parametrize("name", ORACLES)
def test_leakage_check_detects_oracles(name, market, grid, spec, timing, cutoff):
    clean = simulate(make_policy(name), market, grid, spec, timing)
    noisy = simulate(make_policy(name), corrupt_after(market, cutoff), grid, spec, timing)
    pairs = zip(
        decisions_before(clean, cutoff, grid, timing),
        decisions_before(noisy, cutoff, grid, timing),
        strict=True,
    )
    assert any(not np.allclose(a, b) for a, b in pairs)
