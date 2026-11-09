"""Perfect-foresight benchmarks. They see the future and exist only as upper bounds."""

from __future__ import annotations

import numpy as np

from batt.config import BatterySpec, MarketTiming
from batt.data.store import align
from batt.optimize.lp import Exclusive, optimize_dispatch
from batt.policies.base import Policy
from batt.sim.context import DayAheadContext, DecisionContext
from batt.sim.grid import SimulationGrid, Window
from batt.timeutils import minutes


class PerfectForesightRT(Policy):
    """Optimal real-time arbitrage given the realized RT prices.

    Solved one window (default: calendar month) at a time, each ending at the initial SoC. It
    is the denominator of the capture rate.
    """

    name = "perfect_rt"
    is_oracle = True

    def __init__(self, window: Window = "month", exclusive: Exclusive = "auto"):
        self.window = window
        self.exclusive = exclusive
        self._schedule: np.ndarray | None = None

    def reset(self, grid: SimulationGrid, spec: BatterySpec, timing: MarketTiming) -> None:
        self._schedule = None

    def decide(self, ctx: DecisionContext) -> float:
        if self._schedule is None:
            self._schedule = self._solve(ctx)
        return float(self._schedule[ctx.index])

    def _solve(self, ctx: DecisionContext) -> np.ndarray:
        grid, spec = ctx.grid, ctx.spec
        prices = align(ctx.oracle.series("rt"), grid.starts, grid.interval_minutes)
        if np.isnan(prices).any():
            raise ValueError("perfect foresight needs an RT price for every interval")
        schedule = np.zeros(len(grid))
        soc = ctx.soc_mwh
        for window in grid.windows(self.window):
            solution = optimize_dispatch(
                prices[window],
                spec,
                grid.dt_hours,
                soc,
                spec.soc_initial_mwh,
                exclusive=self.exclusive,
                cycle_groups=grid.day_id[window],
            )
            schedule[window] = solution.net_mw
            soc = solution.soc_mwh[-1]
        return schedule


class PerfectForesightDA(Policy):
    """Optimal day-ahead-only trading given the realized DA prices.

    Bids the optimal schedule for each day's actual DA prices, then follows it physically, so
    everything settles at DA prices.
    """

    name = "perfect_da"
    is_oracle = True

    def __init__(self, exclusive: Exclusive = "auto"):
        self.exclusive = exclusive
        self._schedule = np.zeros(0)
        self.days_without_prices: list = []

    def reset(self, grid: SimulationGrid, spec: BatterySpec, timing: MarketTiming) -> None:
        self._schedule = np.zeros(len(grid))
        self.days_without_prices = []

    def plan_day_ahead(self, ctx: DayAheadContext) -> np.ndarray | None:
        intervals = ctx.intervals
        starts = ctx.grid.starts[intervals]
        end = starts[-1] + minutes(ctx.grid.interval_minutes)
        prices = align(ctx.oracle.series("da", start=starts[0], end=end), starts, 60)
        if np.isnan(prices).any():
            self.days_without_prices.append(ctx.grid.days[ctx.day])
            return None
        solution = optimize_dispatch(
            prices,
            ctx.spec,
            ctx.grid.dt_hours,
            ctx.spec.soc_initial_mwh,
            ctx.spec.soc_initial_mwh,
            exclusive=self.exclusive,
        )
        self._schedule[intervals] = solution.net_mw
        return solution.net_mw

    def decide(self, ctx: DecisionContext) -> float:
        return float(self._schedule[ctx.index])
