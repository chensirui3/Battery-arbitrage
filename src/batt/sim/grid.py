"""The calendar of a backtest: regular UTC intervals grouped into local operating days."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from functools import cached_property
from typing import Literal

import numpy as np
import pandas as pd

Window = Literal["day", "week", "month"]


@dataclass(frozen=True)
class SimulationGrid:
    """Intervals covering whole local operating days from ``first_day`` up to ``end_day``.

    Operating days are local (Central) days, so they have 92, 96, or 100 fifteen-minute
    intervals depending on daylight-saving changes.
    """

    starts: pd.DatetimeIndex  # UTC interval starts
    interval_minutes: int
    timezone: str
    days: tuple[date, ...]
    day_id: np.ndarray  # operating-day index of each interval
    day_bounds: np.ndarray  # first interval of each day, plus the total count

    @classmethod
    def for_days(
        cls, first_day: date, end_day: date, timezone: str, interval_minutes: int = 15
    ) -> SimulationGrid:
        if not first_day < end_day:
            raise ValueError("first_day must be before end_day")
        start = pd.Timestamp(first_day).tz_localize(timezone).tz_convert("UTC")
        end = pd.Timestamp(end_day).tz_localize(timezone).tz_convert("UTC")
        starts = pd.date_range(start, end, freq=f"{interval_minutes}min", inclusive="left")
        local_day = starts.tz_convert(timezone).tz_localize(None).normalize()
        codes, uniques = pd.factorize(local_day)
        day_id = codes.astype(np.int64)
        first_of_day = np.flatnonzero(np.r_[True, np.diff(day_id) != 0])
        bounds = np.r_[first_of_day, len(starts)]
        days = tuple(d.date() for d in uniques)
        return cls(starts, interval_minutes, timezone, days, day_id, bounds)

    def __len__(self) -> int:
        return len(self.starts)

    @property
    def dt_hours(self) -> float:
        return self.interval_minutes / 60.0

    @property
    def n_days(self) -> int:
        return len(self.days)

    @cached_property
    def local_starts(self) -> pd.DatetimeIndex:
        return self.starts.tz_convert(self.timezone)

    @cached_property
    def local_dates(self) -> pd.DatetimeIndex:
        """Operating day of each interval, as naive midnight timestamps."""
        return pd.DatetimeIndex(pd.to_datetime(list(self.days)))[self.day_id]

    def day_slice(self, day: int) -> slice:
        return slice(int(self.day_bounds[day]), int(self.day_bounds[day + 1]))

    def windows(self, window: Window) -> list[slice]:
        """Contiguous interval ranges of whole days: one per day, 7-day block, or calendar month."""
        n = self.n_days
        if window == "day":
            keys = np.arange(n)
        elif window == "week":
            keys = np.arange(n) // 7
        elif window == "month":
            idx = pd.DatetimeIndex(pd.to_datetime(list(self.days)))
            keys = (idx.year * 12 + idx.month).to_numpy()
        else:
            raise ValueError(f"unknown window {window!r}")
        out, first = [], 0
        for d in range(1, n + 1):
            if d == n or keys[d] != keys[d - 1]:
                out.append(slice(int(self.day_bounds[first]), int(self.day_bounds[d])))
                first = d
        return out
