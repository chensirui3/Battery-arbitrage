"""Warm-started re-solves of the dispatch LP, for model-predictive control.

MPC solves the same LP before every 15-minute interval with a slightly different forecast,
starting SoC, and horizon. Rebuilding it through SciPy each time costs about 2.7 ms; keeping
one HiGHS model and changing only costs and bounds lets the solver restart from the previous
optimal basis, about 0.45 ms (100 MW / 200 MWh battery, 36-hour horizon). Over a five-year
backtest that is the difference between minutes and most of an hour per policy.
"""

from __future__ import annotations

import highspy
import numpy as np
import scipy.sparse as sp

from batt.config import BatterySpec
from batt.optimize.lp import (
    DispatchSolution,
    Exclusive,
    check_prices,
    make_solution,
    optimize_dispatch,
    soc_endpoints,
)


class RollingDispatchSolver:
    """Same problem and answers as ``optimize_dispatch``, built for repeated solves.

    The HiGHS model has room for ``capacity`` intervals (it grows when a longer horizon
    arrives). Shorter horizons are padded with idle intervals: no charging or discharging,
    so the SoC just carries over to the end. Cases outside this fast path go to
    ``optimize_dispatch``: a daily cycle cap, a solution that charges and discharges in the
    same interval (re-solved as a MILP when ``exclusive="auto"``), or a failed solve.
    """

    def __init__(
        self,
        spec: BatterySpec,
        dt_hours: float,
        capacity: int = 192,
        exclusive: Exclusive = "auto",
    ):
        self.spec = spec
        self.dt = dt_hours
        self.exclusive = exclusive
        self.fallbacks = 0  # solves handed to optimize_dispatch
        self._build(capacity)

    def _build(self, capacity: int) -> None:
        n, spec, dt = capacity, self.spec, self.dt
        eye = sp.identity(n, format="csc")
        previous = sp.eye(n, k=-1, format="csc")
        a = sp.hstack(
            [-spec.eta_charge * dt * eye, (dt / spec.eta_discharge) * eye, eye - previous],
            format="csc",
        )
        lp = highspy.HighsLp()
        lp.num_col_ = 3 * n
        lp.num_row_ = n
        lp.col_cost_ = np.zeros(3 * n)
        lp.col_lower_ = np.zeros(3 * n)
        lp.col_upper_ = np.zeros(3 * n)
        lp.row_lower_ = np.zeros(n)
        lp.row_upper_ = np.zeros(n)
        lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
        lp.a_matrix_.start_ = a.indptr
        lp.a_matrix_.index_ = a.indices
        lp.a_matrix_.value_ = a.data
        highs = highspy.Highs()
        highs.setOptionValue("output_flag", False)
        highs.passModel(lp)
        self._highs = highs
        self._capacity = n
        self._columns = np.arange(3 * n, dtype=np.int32)

    def solve(
        self, prices, soc_initial_mwh: float, soc_terminal_mwh: float | None = None
    ) -> DispatchSolution:
        p = check_prices(prices)
        spec, dt = self.spec, self.dt
        if self.exclusive == "milp" or spec.max_cycles_per_day is not None:
            return self._fallback(p, soc_initial_mwh, soc_terminal_mwh, self.exclusive)
        n = p.size
        if n > self._capacity:
            self._build(max(n, 2 * self._capacity))
        soc0, soc_end = soc_endpoints(spec, n, dt, soc_initial_mwh, soc_terminal_mwh)

        cap = self._capacity
        cost = np.zeros(3 * cap)
        cost[:n] = p * dt  # charging pays the price
        cost[cap : cap + n] = -(p - spec.degradation_cost_per_mwh) * dt  # discharging earns it
        lower = np.zeros(3 * cap)
        upper = np.zeros(3 * cap)
        upper[:n] = spec.power_mw
        upper[cap : cap + n] = spec.power_mw
        lower[2 * cap :] = spec.soc_min_mwh
        upper[2 * cap :] = spec.soc_max_mwh
        if soc_end is not None:
            lower[2 * cap + n - 1] = upper[2 * cap + n - 1] = soc_end

        highs = self._highs
        highs.changeColsCost(3 * cap, self._columns, cost)
        highs.changeColsBounds(3 * cap, self._columns, lower, upper)
        highs.changeRowBounds(0, soc0, soc0)
        highs.run()
        if highs.getModelStatus() != highspy.HighsModelStatus.kOptimal:
            self._build(cap)  # start the next solve from a clean model
            return self._fallback(p, soc0, soc_end, self.exclusive)

        x = np.asarray(highs.getSolution().col_value)
        solution = make_solution(
            p, dt, spec, x[:n], x[cap : cap + n], x[2 * cap : 2 * cap + n], used_milp=False
        )
        if self.exclusive == "auto" and solution.has_simultaneous:
            return self._fallback(p, soc0, soc_end, "milp")
        return solution

    def _fallback(self, p, soc0, soc_end, exclusive: Exclusive) -> DispatchSolution:
        self.fallbacks += 1
        return optimize_dispatch(p, self.spec, self.dt, soc0, soc_end, exclusive=exclusive)
