"""Analog-ensemble price scenarios: today's DA prices plus how RT deviated on similar days.

The deviation of RT from DA prices is heavy-tailed and bursty: most of the time a few dollars,
occasionally thousands for an hour or two, usually in the afternoon or evening. Point models
either average spikes away or chase them. An analog ensemble sidesteps that by replaying
history: find the past days whose DA price profile looks most like today's (and tomorrow's,
once published), and use each one's actual RT-DA spread path, at the same local times of day,
as one scenario. Spike sizes, timing, and persistence are real ones, and they show up about
as often as they did on days like this.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from batt.data.store import DataView, ts_ns
from batt.forecast.history import DAY_NS, MINUTE_NS, PriceHistory, local_naive_ns
from batt.forecast.models import Forecaster, Refit, SpreadAR1Forecaster


@dataclass(frozen=True)
class AnalogPool:
    """Past operating days' RT-DA spreads and DA prices, by local time-of-day slot.

    Row ``r`` is local day ``first_day + r``. Slots are RT intervals of the day (96 for
    15 minutes); the repeated hour of the autumn DST change is averaged, and the skipped
    spring hour is missing, which makes that day incomplete.
    """

    first_day: int
    spread: np.ndarray  # (days, slots)
    da: np.ndarray  # (days, slots)
    candidates: np.ndarray  # rows whose day, previous day, and next day are all complete

    @classmethod
    def from_history(cls, hist: PriceHistory) -> AnalogPool:
        slots = int(DAY_NS // hist.step_ns)
        if len(hist) == 0:
            empty = np.zeros((0, slots))
            return cls(0, empty, empty, np.zeros(0, dtype=np.int64))
        day, slot = day_and_slot(hist, np.arange(len(hist)))
        rows = day - day[0]
        n_days = int(rows[-1]) + 1
        spread = _by_slot(hist.spread, rows, slot, n_days, slots)
        da = _by_slot(hist.da, rows, slot, n_days, slots)
        complete = ~(np.isnan(spread).any(axis=1) | np.isnan(da).any(axis=1))
        usable = complete[:-2] & complete[1:-1] & complete[2:]
        return cls(int(day[0]), spread, da, np.flatnonzero(usable) + 1)


def day_and_slot(hist: PriceHistory, k) -> tuple[np.ndarray, np.ndarray]:
    """Local day number and time-of-day slot of grid intervals ``k``."""
    local = local_naive_ns(hist.start_ns(k), hist.timezone)
    day = local // DAY_NS
    return day, (local - day * DAY_NS) // hist.step_ns


def _by_slot(values, rows, slot, n_days: int, n_slots: int) -> np.ndarray:
    ok = ~np.isnan(values)
    total = np.zeros((n_days, n_slots))
    count = np.zeros((n_days, n_slots))
    np.add.at(total, (rows[ok], slot[ok]), values[ok])
    np.add.at(count, (rows[ok], slot[ok]), 1.0)
    with np.errstate(invalid="ignore"):
        return np.where(count > 0, total / count, np.nan)


class AnalogForecaster(Forecaster):
    """Scenarios = published DA prices + RT-DA spread paths of the most similar past days.

    Similarity is the mean squared difference between DA price profiles: today's, plus
    tomorrow's once it is published. Candidates come from the last ``pool_days`` days whose
    prices were complete by the start of today. The forecast (``predict``) is the scenario
    average.
    """

    name = "analog"
    anchored: ClassVar[bool] = False

    def __init__(
        self,
        n_scenarios: int = 20,
        pool_days: int = 365,
        fit_days: int = 90,
        refit: Refit = "month",
    ):
        super().__init__()
        if n_scenarios < 1:
            raise ValueError("n_scenarios must be positive")
        self._n_scenarios = n_scenarios
        self.pool_days = pool_days
        self._ar1 = SpreadAR1Forecaster(fit_days=fit_days, refit=refit) if self.anchored else None
        self._pool: AnalogPool | None = None
        self._pool_day: int | None = None
        self._choice_key: tuple | None = None
        self._choice = np.zeros(0, dtype=np.int64)

    @property
    def n_scenarios(self) -> int:
        return self._n_scenarios

    @property
    def rho(self) -> float:
        return self._ar1.rho if self._ar1 is not None else 0.0

    def reset(self, timing):
        super().reset(timing)
        self._pool = None
        self._pool_day = None
        self._choice_key = None
        if self._ar1 is not None:
            self._ar1.reset(timing)

    def prepare(self, view, now, first):
        day = int(local_naive_ns([ts_ns(first)], self.timezone)[0] // DAY_NS)
        if day != self._pool_day:
            issue = ts_ns(now)
            start = issue - (self.pool_days + 3) * DAY_NS
            hist = PriceHistory.from_view(view, start, issue, self.timezone)
            self._pool = AnalogPool.from_history(hist)
            self._pool_day = day
            self._choice_key = None
        if self._ar1 is not None:
            self._ar1.prepare(view, now, first)

    def predict(self, view, now, first, n):
        return self.scenarios(view, now, first, n).mean(axis=0)

    def scenarios(self, view: DataView, now, first, n: int) -> np.ndarray:
        issue = ts_ns(now)
        first_ns = ts_ns(first)
        step = view.resolution("rt") * MINUTE_NS
        hist = PriceHistory.from_view(view, issue - 2 * DAY_NS, first_ns + n * step, self.timezone)
        k = hist.index(first_ns) + np.arange(n)
        da = hist.visible_da(k, issue)
        pool = self._pool
        if pool is None or pool.candidates.size == 0:
            return np.repeat(da[None, :], self._n_scenarios, axis=0)

        day, slot = day_and_slot(hist, k)
        today = int(day[0])
        offset = day - today  # 0 today, 1 tomorrow
        has_tomorrow = bool(np.isfinite(da[offset == 1]).any())
        analogs = self._analogs(hist, issue, today, has_tomorrow)
        spread = pool.spread[analogs[:, None] + offset[None, :], slot[None, :]]

        last = int(hist.latest_rt(issue))
        if self.anchored and last >= 0 and np.isfinite(hist.spread[last]):
            # Shift each path to start from the spread just observed, fading at the AR(1) rate.
            last_day, last_slot = day_and_slot(hist, [last])
            then = pool.spread[analogs + int(last_day[0]) - today, int(last_slot[0])]
            decay = self.rho ** (k - last).astype(float)
            spread = spread + decay[None, :] * (hist.spread[last] - then)[:, None]
        return da[None, :] + spread

    def _analogs(self, hist: PriceHistory, issue: int, today: int, has_tomorrow: bool):
        key = (today, has_tomorrow)
        if key == self._choice_key:
            return self._choice
        pool = self._pool
        days = [today, today + 1] if has_tomorrow else [today]
        target = np.concatenate([self._day_profile(hist, issue, d) for d in days])
        rows = pool.candidates
        history = np.concatenate([pool.da[rows + i] for i in range(len(days))], axis=1)
        diff = history - target[None, :]
        ok = ~np.isnan(diff)
        distance = np.where(ok, diff**2, 0.0).sum(axis=1) / np.maximum(ok.sum(axis=1), 1)
        order = np.lexsort((rows, distance))  # nearest first; ties broken by date
        chosen = rows[order[: self._n_scenarios]]
        if chosen.size < self._n_scenarios:  # short history: reuse the nearest days
            chosen = np.resize(chosen, self._n_scenarios)
        self._choice_key, self._choice = key, chosen
        return chosen

    @staticmethod
    def _day_profile(hist: PriceHistory, issue: int, day: int) -> np.ndarray:
        """DA prices of one local day by slot, as visible at the issue time (NaN if not)."""
        slots = int(DAY_NS // hist.step_ns)
        k = np.flatnonzero(hist.day_num == day)
        profile = np.full(slots, np.nan)
        if k.size:
            _, slot = day_and_slot(hist, k)
            values = hist.visible_da(k, issue)
            profile = _by_slot(values, np.zeros_like(slot), slot, 1, slots)[0]
        return profile


class AnchoredAnalogForecaster(AnalogForecaster):
    """Analog scenarios that start from the latest observed RT-DA spread.

    Each historical path is shifted by the gap between the spread just published and that
    path's value at the same time, and the shift fades at the fitted AR(1) rate, so the near
    term reflects what is happening now and later hours follow the analogs.
    """

    name = "analog_rt"
    anchored = True
