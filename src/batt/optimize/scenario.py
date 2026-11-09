r"""Stochastic dispatch: one plan for the near term, a separate plan per price scenario after.

Deterministic MPC plans against one forecast as if it were certain. When that forecast is an
average over very different futures (a price spike tonight in some, a calm evening in others),
its plan ignores that stored energy is an option on future prices, worth more when they are
uncertain. Here the first ``shared_steps`` steps get one plan shared by all scenarios, because
those actions must be chosen before anyone knows which future unfolds, and each scenario gets
its own plan for the rest of the horizon. The objective is the average revenue over the
(equally likely) scenarios.

Sharing only the very first step is the textbook two-stage model, but it is optimistic: each
scenario's plan then acts as if it knew that scenario's whole future from 15 minutes on, so it
"waits" for spikes it could never time in practice (prices are seen 25 minutes late).
Sharing a longer prefix removes that foresight for the hours that matter most.

Variables are energies per step (MWh), so a step's length only enters its bounds. The first
``fine_steps`` steps are single 15-minute intervals; later intervals are grouped into hourly
blocks at their average price (DA prices are hourly anyway), which keeps the problem small
enough to re-solve every 15 minutes. With steps t, scenarios s, and probabilities pi_s:

    maximize    sum_s pi_s sum_t [ p_ts (d_ts - c_ts) - k d_ts ]
    subject to  e_ts = e_(t-1)s + eta_c c_ts - d_ts / eta_d,        e_(-1)s = e_init
                0 <= c_ts, d_ts <= P dt_t,   E_min <= e_ts <= E_max,   e_(T-1)s = e_end
                c_ts, d_ts, e_ts equal across scenarios for t < shared_steps

Unlike ``optimize_dispatch`` there is no MILP fallback: at deeply negative scenario prices a
plan may charge and discharge in the same step. Only the first step is executed, as a net
power, so this can slightly overstate a plan's value but never changes the physics.
"""

from __future__ import annotations

from dataclasses import dataclass

import highspy
import numpy as np
import scipy.sparse as sp

from batt.config import BatterySpec
from batt.optimize.lp import OptimizationError, soc_endpoints


def step_sizes(n_intervals: int, fine_steps: int, coarse_intervals: int) -> np.ndarray:
    """Intervals per step: ``fine_steps`` single intervals, then blocks of ``coarse_intervals``."""
    fine = min(n_intervals, fine_steps)
    blocks, remainder = divmod(n_intervals - fine, coarse_intervals)
    sizes = [np.ones(fine), np.full(blocks, coarse_intervals)]
    if remainder:
        sizes.append(np.array([remainder]))
    return np.concatenate(sizes).astype(np.int64)


@dataclass(frozen=True)
class ScenarioSolution:
    first_net_mw: float  # action for the interval being decided: + discharge, - charge
    expected_profit: float  # average over scenarios of revenue minus degradation, $
    steps: int  # number of steps in this horizon


class ScenarioDispatchSolver:
    """Re-solvable stochastic dispatch LP for a fixed number of equally likely scenarios.

    One HiGHS model is kept with room for ``capacity`` intervals and warm-started between
    solves, like ``RollingDispatchSolver``. Shorter horizons are padded with idle steps.
    """

    def __init__(
        self,
        spec: BatterySpec,
        interval_hours: float,
        n_scenarios: int,
        fine_steps: int = 16,
        coarse_intervals: int = 4,
        shared_steps: int = 1,
        capacity: int = 144,
    ):
        if min(n_scenarios, fine_steps, coarse_intervals, shared_steps) < 1:
            raise ValueError("n_scenarios, fine_steps, coarse_intervals, shared_steps must be >= 1")
        if spec.max_cycles_per_day is not None:
            raise ValueError("scenario plans do not model a daily cycle cap (max_cycles_per_day)")
        self.spec = spec
        self.dt = interval_hours
        self.n_scenarios = n_scenarios
        self.fine_steps = fine_steps
        self.coarse_intervals = coarse_intervals
        self.shared_steps = shared_steps
        self._build(capacity)

    def step_sizes(self, n_intervals: int) -> np.ndarray:
        return step_sizes(n_intervals, self.fine_steps, self.coarse_intervals)

    def _build(self, capacity: int) -> None:
        spec, n_scen = self.spec, self.n_scenarios
        steps = self.step_sizes(capacity).size
        shared = min(self.shared_steps, steps)
        m = max(steps - shared, 1)  # scenario-specific steps (at least one, possibly padding)
        # Shared steps: c_t = t, d_t = shared + t, e_t = 2 * shared + t.
        t_sh = np.arange(shared)
        self._sc, self._sd, self._se = t_sh, shared + t_sh, 2 * shared + t_sh
        # Scenario steps: shape (scenarios, m).
        u = np.arange(m)
        base = 3 * shared + np.arange(n_scen)[:, None] * 3 * m
        self._c, self._d, self._e = base + u, base + m + u, base + 2 * m + u
        self._n_cols = 3 * shared + n_scen * 3 * m
        n_rows = shared + n_scen * m

        # Dynamics: e_t - e_(t-1) - eta_c c_t + d_t / eta_d = (e_init if t == 0 else 0).
        sh_rows = t_sh
        sc_rows = shared + np.arange(n_scen)[:, None] * m + u
        prev_sh = self._se[:-1]  # e_(t-1) for shared t >= 1
        prev_sc = np.where(u == 0, self._se[-1], self._e - 1)
        rows = np.concatenate([sh_rows, sh_rows[1:], sh_rows, sh_rows] + [sc_rows.ravel()] * 4)
        cols = np.concatenate(
            [self._se, prev_sh, self._sc, self._sd]
            + [self._e.ravel(), prev_sc.ravel(), self._c.ravel(), self._d.ravel()]
        )
        k_sh, k_sc = np.ones(shared), np.ones(n_scen * m)
        values = np.concatenate(
            [
                k_sh,
                -np.ones(shared - 1),
                -spec.eta_charge * k_sh,
                k_sh / spec.eta_discharge,
                k_sc,
                -k_sc,
                -spec.eta_charge * k_sc,
                k_sc / spec.eta_discharge,
            ]
        )
        a = sp.csc_matrix((values, (rows, cols)), shape=(n_rows, self._n_cols))

        lp = highspy.HighsLp()
        lp.num_col_ = self._n_cols
        lp.num_row_ = n_rows
        lp.col_cost_ = np.zeros(self._n_cols)
        lp.col_lower_ = np.zeros(self._n_cols)
        lp.col_upper_ = np.zeros(self._n_cols)
        lp.row_lower_ = np.zeros(n_rows)
        lp.row_upper_ = np.zeros(n_rows)
        lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
        lp.a_matrix_.start_ = a.indptr
        lp.a_matrix_.index_ = a.indices
        lp.a_matrix_.value_ = a.data
        highs = highspy.Highs()
        highs.setOptionValue("output_flag", False)
        highs.passModel(lp)
        self._highs = highs
        self._capacity = capacity
        self._columns = np.arange(self._n_cols, dtype=np.int32)

    def solve(
        self,
        prices,
        soc_initial_mwh: float,
        soc_terminal_mwh: float | None = None,
        first_mw: float | None = None,
    ) -> ScenarioSolution:
        """Best first action given price scenarios (rows), equally likely.

        ``first_mw`` fixes the first action instead (it must be feasible), which gives the
        value of committing to it, e.g. to price the deterministic plan's choice.
        """
        p = np.asarray(prices, dtype=float)
        if p.ndim == 1:
            p = p[None, :]
        if p.ndim != 2 or p.shape[0] != self.n_scenarios or p.shape[1] == 0:
            raise ValueError(f"prices must have shape ({self.n_scenarios}, n_intervals)")
        if not np.all(np.isfinite(p)):
            raise ValueError("prices contain NaN or infinite values")
        n = p.shape[1]
        if n > self._capacity:
            self._build(max(n, 2 * self._capacity))

        spec = self.spec
        sizes = self.step_sizes(n)
        dt = sizes * self.dt
        starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
        step_price = np.add.reduceat(p, starts, axis=1) / sizes  # (scenarios, steps)
        soc0, soc_end = soc_endpoints(spec, n, self.dt, soc_initial_mwh, soc_terminal_mwh)
        k = spec.degradation_cost_per_mwh
        weight = 1.0 / self.n_scenarios

        cost = np.zeros(self._n_cols)
        lower = np.zeros(self._n_cols)
        upper = np.zeros(self._n_cols)
        lower[self._se] = lower[self._e] = spec.soc_min_mwh
        upper[self._se] = upper[self._e] = spec.soc_max_mwh

        n_steps = sizes.size
        n_shared = min(n_steps, self._sc.size)
        mean_price = step_price[:, :n_shared].mean(axis=0)
        sc, sd = self._sc[:n_shared], self._sd[:n_shared]
        cost[sc], cost[sd] = mean_price, -(mean_price - k)
        upper[sc] = upper[sd] = spec.power_mw * dt[:n_shared]
        if first_mw is not None:
            energy = float(first_mw) * dt[0]
            lower[self._sc[0]] = upper[self._sc[0]] = max(-energy, 0.0)
            lower[self._sd[0]] = upper[self._sd[0]] = max(energy, 0.0)

        n_own = n_steps - n_shared  # scenario-specific steps in this horizon
        if n_own:
            c, d = self._c[:, :n_own], self._d[:, :n_own]
            own_price = step_price[:, n_shared:]
            cost[c] = weight * own_price
            cost[d] = -weight * (own_price - k)
            upper[c] = upper[d] = spec.power_mw * dt[n_shared:]
        if soc_end is not None:
            last = self._se[n_shared - 1] if n_own == 0 else self._e[:, n_own - 1]
            lower[last] = upper[last] = soc_end

        x = self._run(cost, lower, upper, soc0)
        return ScenarioSolution(
            first_net_mw=float((x[self._sd[0]] - x[self._sc[0]]) / dt[0]),
            expected_profit=float(-(cost @ x)),
            steps=int(n_steps),
        )

    def _run(self, cost, lower, upper, soc0) -> np.ndarray:
        highs = self._highs
        for attempt in range(2):
            highs = self._highs
            highs.changeColsCost(self._n_cols, self._columns, cost)
            highs.changeColsBounds(self._n_cols, self._columns, lower, upper)
            highs.changeRowBounds(0, soc0, soc0)
            highs.run()
            if highs.getModelStatus() == highspy.HighsModelStatus.kOptimal:
                return np.asarray(highs.getSolution().col_value)
            if attempt == 0:
                self._build(self._capacity)  # retry once from a fresh model
        status = highs.modelStatusToString(highs.getModelStatus())
        raise OptimizationError(f"scenario LP failed: {status}")
