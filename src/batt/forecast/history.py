"""Prices on a regular time grid, with the moment each value became public."""

from __future__ import annotations

import functools
from dataclasses import dataclass
from functools import cached_property

import numpy as np
import pandas as pd

from batt.data.store import DataView, ts_ns

MINUTE_NS = 60_000_000_000
HOUR_NS = 60 * MINUTE_NS
DAY_NS = 24 * HOUR_NS
NEVER = np.iinfo(np.int64).max  # publish time of values that are missing or not public yet


def local_naive_ns(utc_ns, timezone: str) -> np.ndarray:
    """Wall-clock times in ``timezone`` as naive int64 nanoseconds.

    Uses a cached table of the zone's UTC-offset changes (a binary search instead of a pandas
    conversion, which is ~40x slower on the small arrays a live forecast needs).
    """
    utc = np.asarray(utc_ns, dtype=np.int64)
    changes, offsets = _offset_table(timezone)
    if utc.size and (utc.min() < changes[0] or utc.max() >= _TABLE_END_NS):
        idx = pd.to_datetime(utc, unit="ns", utc=True)
        return idx.tz_convert(timezone).tz_localize(None).asi8
    return utc + offsets[np.searchsorted(changes, utc, side="right") - 1]


def local_to_utc_ns(local_ns, timezone: str) -> np.ndarray:
    """UTC nanoseconds for naive wall-clock times in ``timezone`` (inverse of the above).

    Exact away from the hour around a DST change, where wall-clock times are skipped or
    repeated; callers use it for fixed times of day such as the 10:00 DA bid deadline.
    """
    local = np.asarray(local_ns, dtype=np.int64)
    guess = local - (local_naive_ns(local, timezone) - local)  # offset near that instant
    return local - (local_naive_ns(guess, timezone) - guess)


_TABLE_START = pd.Timestamp("1990-01-01", tz="UTC")
_TABLE_END = pd.Timestamp("2060-01-01", tz="UTC")
_TABLE_END_NS = _TABLE_END.value


@functools.lru_cache(maxsize=8)
def _offset_table(timezone: str) -> tuple[np.ndarray, np.ndarray]:
    """UTC instants (1990-2060) at which ``timezone``'s offset changes, and the new offsets.

    Offsets are sampled hourly, then each change is located to the minute.
    """

    def offset(utc: np.ndarray) -> np.ndarray:
        local = pd.to_datetime(utc, unit="ns", utc=True).tz_convert(timezone)
        return local.tz_localize(None).asi8 - utc

    hourly = pd.date_range(_TABLE_START, _TABLE_END, freq="h", inclusive="left").asi8
    hourly_offset = offset(hourly)
    change = np.flatnonzero(np.diff(hourly_offset)) + 1
    # Each change happened in (hourly[c - 1], hourly[c]]; find the first minute with the new offset.
    candidates = hourly[change - 1][:, None] + np.arange(1, 61) * MINUTE_NS
    new = offset(candidates.ravel()).reshape(candidates.shape) != hourly_offset[change - 1][:, None]
    instants = candidates[np.arange(change.size), np.argmax(new, axis=1)]
    changes = np.concatenate([[hourly[0]], instants])
    offsets = np.concatenate([[hourly_offset[0]], hourly_offset[change]])
    return changes, offsets


def published_horizon(view: DataView, first_ns: int, step_ns: int) -> int:
    """Intervals from ``first_ns`` to the end of the last day with DA prices in ``view``.

    That is 10.5 to 34.5 hours, depending on whether tomorrow's DA prices are out yet. ``view``
    must be a point-in-time view: an oracle view would see DA prices for every future day.
    """
    starts, _, _ = view.rows("da", start=first_ns - 2 * DAY_NS)
    if starts.size == 0:
        return 0
    end_ns = int(starts[-1]) + view.resolution("da") * MINUTE_NS
    return max(0, (end_ns - first_ns) // step_ns)


@dataclass(frozen=True, eq=False)
class PriceHistory:
    """RT and DA prices on one grid of RT intervals; index ``k`` starts at origin + k * step.

    Each DA price is repeated on the RT intervals of its hour. Values that are missing, or not
    yet public in the view the history was built from, are NaN with publish time ``NEVER``.
    """

    origin_ns: int
    step_ns: int
    timezone: str
    rt: np.ndarray
    rt_publish: np.ndarray
    da: np.ndarray
    da_publish: np.ndarray

    @classmethod
    def from_view(cls, view: DataView, start, end, timezone: str) -> PriceHistory:
        """Grid covering [start, end), holding what ``view`` shows."""
        step_ns = view.resolution("rt") * MINUTE_NS
        origin = ts_ns(start) // step_ns * step_ns
        stop = -(-ts_ns(end) // step_ns) * step_ns
        n = max((stop - origin) // step_ns, 0)
        rt, rt_publish = _place(view, "rt", origin, stop, step_ns, n)
        da, da_publish = _place(view, "da", origin, stop, step_ns, n)
        return cls(origin, step_ns, timezone, rt, rt_publish, da, da_publish)

    def __len__(self) -> int:
        return self.rt.size

    @property
    def per_hour(self) -> int:
        return HOUR_NS // self.step_ns

    @property
    def per_day(self) -> int:
        return DAY_NS // self.step_ns

    def start_ns(self, k) -> np.ndarray:
        return self.origin_ns + np.asarray(k, dtype=np.int64) * self.step_ns

    def index(self, ts) -> int:
        return (ts_ns(ts) - self.origin_ns) // self.step_ns

    @cached_property
    def spread(self) -> np.ndarray:
        """RT minus DA price (NaN unless both exist).

        Checks that no DA price was published after the RT price of its interval, so a
        spread is visible exactly when its RT price is.
        """
        both = (self.rt_publish != NEVER) & (self.da_publish != NEVER)
        if np.any(self.da_publish[both] > self.rt_publish[both]):
            raise ValueError("a DA price was published after the RT price of its interval")
        return self.rt - self.da

    @cached_property
    def day_num(self) -> np.ndarray:
        """Local operating day of each interval, as days since 1970-01-01."""
        return local_naive_ns(self.start_ns(np.arange(len(self))), self.timezone) // DAY_NS

    def latest_rt(self, issue_ns) -> np.ndarray:
        """Index of the latest RT price published by each issue time (-1 if none)."""
        return _latest(self._rt_order, issue_ns)

    def latest_da(self, issue_ns) -> np.ndarray:
        """Index of the last interval with a DA price published by each issue time (-1 if none)."""
        return _latest(self._da_order, issue_ns)

    @cached_property
    def _rt_order(self):
        return _publish_order(self.rt_publish, "RT")

    @cached_property
    def _da_order(self):
        return _publish_order(self.da_publish, "DA")

    def visible_da(self, k, issue_ns) -> np.ndarray:
        """DA price of interval ``k`` if published by the issue time, else NaN."""
        k = np.asarray(k, dtype=np.int64)
        if len(self) == 0:
            return np.full(k.shape, np.nan)
        inside = (k >= 0) & (k < len(self))
        kk = np.where(inside, k, 0)
        ok = inside & (self.da_publish[kk] <= issue_ns)
        return np.where(ok, self.da[kk], np.nan)

    def trailing_spread(self, last, window: int) -> tuple[np.ndarray, np.ndarray]:
        """Mean and standard deviation of the spread over ``window`` intervals up to ``last``."""
        s1, s2, count = self._spread_sums
        hi = np.asarray(last, dtype=np.int64) + 1
        lo = np.maximum(hi - window, 0)
        n = count[hi] - count[lo]
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = (s1[hi] - s1[lo]) / n
            var = (s2[hi] - s2[lo]) / n - mean**2
        mean = np.where(n > 0, mean, np.nan)
        std = np.where(n > 1, np.sqrt(np.maximum(var, 0.0)), np.nan)
        return mean, std

    @cached_property
    def _spread_sums(self):
        s = self.spread
        ok = ~np.isnan(s)
        x = np.where(ok, s, 0.0)
        return (
            np.concatenate([[0.0], np.cumsum(x)]),
            np.concatenate([[0.0], np.cumsum(x * x)]),
            np.concatenate([[0], np.cumsum(ok)]),
        )

    def da_day_stats(self, k) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Mean, max, and min DA price of the operating day containing each interval ``k``.

        Uses all of the day's prices, so callers use it only where the day's DA prices are
        public. A day's DA prices are published together; this checks that they were.
        """
        k = np.asarray(k, dtype=np.int64)
        out = np.full((3, k.size), np.nan)
        inside = (k >= 0) & (k < len(self))
        if inside.any():
            days, inverse = np.unique(self.day_num[k[inside]], return_inverse=True)
            lo = np.searchsorted(self.day_num, days, side="left")
            hi = np.searchsorted(self.day_num, days, side="right")
            stats = np.full((3, days.size), np.nan)
            for j in range(days.size):
                values = self.da[lo[j] : hi[j]]
                ok = ~np.isnan(values)
                if not ok.any():
                    continue
                publish = self.da_publish[lo[j] : hi[j]][ok]
                if publish.min() != publish.max():
                    raise ValueError("a day's DA prices must be published together")
                v = values[ok]
                stats[:, j] = v.mean(), v.max(), v.min()
            out[:, inside] = stats[:, inverse]
        return out[0], out[1], out[2]


def _place(view: DataView, name: str, origin: int, stop: int, step_ns: int, n: int):
    resolution_ns = view.resolution(name) * MINUTE_NS
    if resolution_ns % step_ns:
        raise ValueError(f"{name} intervals do not divide into {step_ns // MINUTE_NS}-minute steps")
    starts, publish, values = view.rows(name, origin // resolution_ns * resolution_ns, stop)
    out = np.full(n, np.nan)
    out_publish = np.full(n, NEVER, dtype=np.int64)
    offset = starts - origin
    if np.any(offset % step_ns):
        raise ValueError(f"{name} intervals are not aligned to the grid")
    first = offset // step_ns
    for j in range(resolution_ns // step_ns):
        k = first + j
        ok = (k >= 0) & (k < n)
        out[k[ok]] = values[ok]
        out_publish[k[ok]] = publish[ok]
    return out, out_publish


def _publish_order(publish: np.ndarray, name: str):
    idx = np.flatnonzero(publish != NEVER)
    times = publish[idx]
    if np.any(np.diff(times) < 0):
        raise ValueError(f"{name} prices must be published in interval order")
    return idx, times


def _latest(order, issue_ns) -> np.ndarray:
    idx, times = order
    issue = np.asarray(issue_ns, dtype=np.int64)
    if idx.size == 0:
        return np.full(issue.shape, -1, dtype=np.int64)
    pos = np.searchsorted(times, issue, side="right") - 1
    return np.where(pos >= 0, idx[np.maximum(pos, 0)], -1)
