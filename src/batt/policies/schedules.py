"""Realistic baselines that plan each day from information available before it starts."""

from __future__ import annotations

import numpy as np

from batt.data.store import align
from batt.optimize.lp import Exclusive
from batt.policies.base import DailyPlanPolicy
from batt.sim.context import DecisionContext
from batt.timeutils import days, minutes


class TrailingAverageSchedule(DailyPlanPolicy):
    """Naive baseline: assume each day looks like the average day of the last N days.

    The average daily shape of recent RT prices (by local time of day) is the forecast, so the
    battery ends up following a nearly fixed daily schedule that drifts with the seasons.
    """

    name = "naive"

    def __init__(self, lookback_days: int = 30, exclusive: Exclusive = "auto"):
        super().__init__(exclusive)
        self.lookback_days = lookback_days

    def forecast(self, ctx: DecisionContext, intervals: slice) -> np.ndarray | None:
        history = ctx.view.series("rt", start=ctx.now - days(self.lookback_days))
        if len(history) < 24 * 60 // ctx.grid.interval_minutes:
            return None  # less than a day of history
        local = history.index.tz_convert(ctx.grid.timezone)
        profile = history.groupby(local.hour * 60 + local.minute).mean()
        target = ctx.grid.local_starts[intervals]
        values = profile.reindex(target.hour * 60 + target.minute).to_numpy(dtype=float)
        if np.isnan(values).any():
            values = np.where(np.isnan(values), float(profile.mean()), values)
        return values


class DayAheadPriceSchedule(DailyPlanPolicy):
    """Plan each day against its published day-ahead prices; execute and settle in real time.

    DA prices for day D are public by mid-afternoon on D-1, so they are a legitimate (and
    strong) forecast of D's real-time prices.
    """

    name = "da_informed"

    def forecast(self, ctx: DecisionContext, intervals: slice) -> np.ndarray | None:
        starts = ctx.grid.starts[intervals]
        end = starts[-1] + minutes(ctx.grid.interval_minutes)
        da = ctx.view.series("da", start=starts[0].floor("h"), end=end)
        values = align(da, starts, 60)
        return None if np.isnan(values).any() else values
