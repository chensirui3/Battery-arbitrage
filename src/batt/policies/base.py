"""The policy interface and a reusable plan-once-per-day policy."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date
from typing import ClassVar

import numpy as np

from batt.config import BatterySpec, MarketTiming
from batt.optimize.lp import Exclusive, optimize_dispatch
from batt.sim.context import DayAheadContext, DecisionContext
from batt.sim.grid import SimulationGrid


class Policy(ABC):
    """A dispatch strategy.

    Subclasses implement ``decide``. Day-ahead trading (``plan_day_ahead``) is optional.
    Oracle policies see future data through ``ctx.oracle``; they exist only as benchmarks.
    """

    name: ClassVar[str] = "policy"
    is_oracle: ClassVar[bool] = False

    def reset(self, grid: SimulationGrid, spec: BatterySpec, timing: MarketTiming) -> None:  # noqa: B027
        """Clear cached state before a new simulation."""

    def plan_day_ahead(self, ctx: DayAheadContext) -> np.ndarray | None:
        """Day-ahead positions in MW for every interval of ``ctx.day``, or None for no DA trade.

        Called at the DA bid deadline on the day before each operating day.
        """
        return None

    @abstractmethod
    def decide(self, ctx: DecisionContext) -> float:
        """Requested net power for interval ``ctx.index``: + discharge, - charge (MW)."""


class DailyPlanPolicy(Policy):
    """Plans one schedule per operating day from a price forecast, then follows it.

    The plan is made when the day's first interval is decided (shortly before midnight), from
    data visible at that moment, and returns the battery to its initial SoC by the day's end.
    """

    def __init__(self, exclusive: Exclusive = "auto"):
        self.exclusive = exclusive
        self._day = -1
        self._offset = 0
        self._plan = np.zeros(0)
        self.days_without_forecast: list[date] = []

    def reset(self, grid: SimulationGrid, spec: BatterySpec, timing: MarketTiming) -> None:
        self._day = -1
        self._offset = 0
        self._plan = np.zeros(0)
        self.days_without_forecast = []

    @abstractmethod
    def forecast(self, ctx: DecisionContext, intervals: slice) -> np.ndarray | None:
        """Expected price ($/MWh) for each interval in ``intervals``, using only ``ctx.view``."""

    def decide(self, ctx: DecisionContext) -> float:
        if ctx.day != self._day:
            self._day = ctx.day
            remaining = slice(ctx.index, ctx.grid.day_slice(ctx.day).stop)
            self._offset = ctx.index
            prices = self.forecast(ctx, remaining)
            if prices is None:
                self.days_without_forecast.append(ctx.grid.days[ctx.day])
                self._plan = np.zeros(remaining.stop - remaining.start)
            else:
                solution = optimize_dispatch(
                    prices,
                    ctx.spec,
                    ctx.grid.dt_hours,
                    ctx.soc_mwh,
                    ctx.spec.soc_initial_mwh,
                    exclusive=self.exclusive,
                )
                self._plan = solution.net_mw
        return float(self._plan[ctx.index - self._offset])
