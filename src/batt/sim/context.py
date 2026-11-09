"""What a policy is allowed to know when it makes a decision."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from batt.config import BatterySpec, MarketTiming
from batt.data.store import DataView
from batt.sim.grid import SimulationGrid


@dataclass(frozen=True, slots=True)
class DecisionContext:
    """Inputs to a real-time decision for one interval."""

    now: pd.Timestamp  # wall-clock time of the decision (UTC)
    index: int  # position of the interval being decided
    soc_mwh: float  # stored energy at the start of that interval
    grid: SimulationGrid
    spec: BatterySpec
    timing: MarketTiming
    view: DataView  # data published at or before `now`
    oracle: DataView | None  # all data; only given to oracle (benchmark) policies

    @property
    def interval_start(self) -> pd.Timestamp:
        return self.grid.starts[self.index]

    @property
    def day(self) -> int:
        return int(self.grid.day_id[self.index])


@dataclass(frozen=True, slots=True)
class DayAheadContext:
    """Inputs to day-ahead bidding for one operating day, at the bid deadline the day before."""

    now: pd.Timestamp
    day: int
    grid: SimulationGrid
    spec: BatterySpec
    timing: MarketTiming
    view: DataView
    oracle: DataView | None

    @property
    def intervals(self) -> slice:
        return self.grid.day_slice(self.day)
