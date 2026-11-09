"""Point-in-time market data.

Every value carries the time it became public (``publish_time_utc``). A ``DataView`` taken
"as of" a moment only exposes values published at or before that moment, which is how the
backtest guarantees that no decision can see the future. Benchmark (oracle) policies get an
unrestricted view and are labeled as such.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from batt.config import MarketTiming
from batt.timeutils import days, minutes

UTC = "UTC"


def to_ns(values) -> np.ndarray:
    """int64 nanoseconds since the epoch for timezone-aware timestamps."""
    idx = pd.DatetimeIndex(values)
    if idx.tz is None:
        raise ValueError("timestamps must be timezone-aware")
    return idx.tz_convert(UTC).as_unit("ns").asi8


def ts_ns(ts) -> int:
    """int64 nanoseconds since the epoch. Integers are taken to already be UTC nanoseconds."""
    if isinstance(ts, int | np.integer):
        return int(ts)
    ts = pd.Timestamp(ts)
    if ts.tzinfo is None:
        raise ValueError("timestamps must be timezone-aware")
    return int(ts.tz_convert(UTC).as_unit("ns").value)


class PointInTimeTable:
    """Values for time intervals, each stamped with the moment it was published.

    Rows are kept sorted by publish time, so the rows visible at any moment form a prefix of
    the table that a binary search finds. A table may hold several published versions
    ("vintages") of the same interval, as forecasts do; queries return the latest visible one.
    """

    def __init__(self, frame: pd.DataFrame, resolution_minutes: int):
        required = {"interval_start_utc", "value", "publish_time_utc"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"missing columns: {sorted(missing)}")
        df = frame.sort_values(["publish_time_utc", "interval_start_utc"], kind="stable")
        self.frame = df.reset_index(drop=True)
        self.resolution_minutes = resolution_minutes
        self._publish = _read_only(to_ns(self.frame["publish_time_utc"]))
        self._start = _read_only(to_ns(self.frame["interval_start_utc"]))
        self._value = _read_only(self.frame["value"].to_numpy(dtype=float))
        self._has_vintages = bool(pd.Index(self._start).has_duplicates)
        # When intervals appear in publish order with no repeats (true for prices), the visible
        # rows of any time window are one contiguous slice, found by binary search.
        self._ordered = bool(np.all(np.diff(self._start) > 0))

    def __len__(self) -> int:
        return len(self._value)

    def visible_count(self, now) -> int:
        """Number of rows published at or before ``now`` (all rows if ``now`` is None)."""
        if now is None:
            return len(self._publish)
        return int(np.searchsorted(self._publish, ts_ns(now), side="right"))

    def rows(self, now, start=None, end=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Visible ``(start_ns, publish_ns, value)`` for intervals starting in [start, end).

        Sorted by interval start, one row per interval: its latest vintage visible at ``now``.
        The arrays are read-only and may share memory with the table.
        """
        k = self.visible_count(now)
        if self._ordered:
            lo = 0 if start is None else int(np.searchsorted(self._start, ts_ns(start)))
            hi = k if end is None else int(np.searchsorted(self._start, ts_ns(end)))
            window = slice(min(lo, k), min(hi, k))
            return self._start[window], self._publish[window], self._value[window]
        starts, publish, values = self._start[:k], self._publish[:k], self._value[:k]
        mask = np.ones(k, dtype=bool)
        if start is not None:
            mask &= starts >= ts_ns(start)
        if end is not None:
            mask &= starts < ts_ns(end)
        starts, publish, values = starts[mask], publish[mask], values[mask]
        if self._has_vintages:
            # Rows are in publish order, so an interval's last occurrence is its latest vintage.
            latest = ~pd.Index(starts).duplicated(keep="last")
            starts, publish, values = starts[latest], publish[latest], values[latest]
        order = np.argsort(starts, kind="stable")
        return starts[order], publish[order], values[order]

    def series(self, now, start=None, end=None) -> pd.Series:
        """Visible values for intervals starting in [start, end), indexed by UTC start time."""
        starts, _, values = self.rows(now, start, end)
        index = pd.to_datetime(starts, unit="ns", utc=True)
        return pd.Series(values.copy(), index=index, name="value")


def _read_only(values: np.ndarray) -> np.ndarray:
    out = np.array(values, copy=True)
    out.flags.writeable = False
    return out


class DataView:
    """Market data as visible at ``now`` (UTC). ``now=None`` sees everything (oracle only)."""

    __slots__ = ("_conditions", "_fundamentals", "_tables", "now")

    def __init__(
        self, tables: Mapping[str, PointInTimeTable], now, fundamentals=None, conditions=None
    ):
        self._tables = tables
        self.now = None if now is None else pd.Timestamp(now)
        self._fundamentals = fundamentals
        self._conditions = conditions

    @property
    def fundamentals(self):
        """Load, wind, and solar forecasts and actuals as public at ``now`` (None if not loaded).

        See ``batt.data.fundamentals``.
        """
        if self._fundamentals is None:
            return None
        return self._fundamentals.view(self.now)

    @property
    def conditions(self):
        """Real-time grid conditions (SCED prices and reserves, ERCOT's look-ahead, outages) as
        public at ``now`` (None if not loaded). See ``batt.data.conditions``.
        """
        if self._conditions is None:
            return None
        return self._conditions.view(self.now)

    @property
    def is_oracle(self) -> bool:
        return self.now is None

    def names(self) -> list[str]:
        return list(self._tables)

    def series(self, name: str, start=None, end=None) -> pd.Series:
        return self._tables[name].series(self.now, start, end)

    def rows(self, name: str, start=None, end=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Visible ``(start_ns, publish_ns, value)`` arrays; see ``PointInTimeTable.rows``."""
        return self._tables[name].rows(self.now, start, end)

    def resolution(self, name: str) -> int:
        """Interval length of a table, in minutes."""
        return self._tables[name].resolution_minutes

    def at(self, earlier) -> DataView:
        """The same data as of an earlier moment. Views can be narrowed, never widened."""
        earlier = pd.Timestamp(earlier)
        if self.now is not None and earlier > self.now:
            raise ValueError("cannot move a point-in-time view forward")
        return DataView(self._tables, earlier, self._fundamentals, self._conditions)


def align(series: pd.Series, starts: pd.DatetimeIndex, resolution_minutes: int) -> np.ndarray:
    """Value in effect at each interval start.

    Coarser series (hourly day-ahead prices) are broadcast to the finer intervals they
    contain. Missing values stay NaN so gaps are never silently filled.
    """
    keys = pd.DatetimeIndex(starts).floor(f"{resolution_minutes}min")
    return series.reindex(keys).to_numpy(dtype=float)


def rt_publish_times(interval_end_utc: pd.Series, timing: MarketTiming) -> pd.Series:
    """A real-time price is published shortly after its interval ends."""
    return interval_end_utc + minutes(timing.rt_publish_delay_minutes)


def da_publish_times(interval_start_utc: pd.Series, timing: MarketTiming) -> pd.Series:
    """Day-ahead prices for operating day D are published on D-1 at the configured local time."""
    local = interval_start_utc.dt.tz_convert(timing.timezone)
    operating_day = local.dt.tz_localize(None).dt.normalize()
    t = timing.da_publish_local_time
    publish_local = operating_day - days(1) + minutes(60 * t.hour + t.minute)
    return publish_local.dt.tz_localize(timing.timezone).dt.tz_convert(UTC)


def _price_table(df: pd.DataFrame, publish: pd.Series, resolution_minutes: int) -> PointInTimeTable:
    frame = pd.DataFrame(
        {
            "interval_start_utc": df["interval_start_utc"].to_numpy(),
            "value": df["price"].to_numpy(dtype=float),
            "publish_time_utc": publish.to_numpy(),
        }
    )
    return PointInTimeTable(frame, resolution_minutes=resolution_minutes)


class MarketData:
    """Point-in-time prices for one settlement point.

    Tables: ``"rt"`` (real-time, 15-minute) and ``"da"`` (day-ahead, hourly). Publish times
    come from the timing assumptions, so changing an assumption never requires re-downloading.
    """

    def __init__(
        self,
        location: str,
        tables: Mapping[str, PointInTimeTable],
        fundamentals=None,
        conditions=None,
    ):
        self.location = location
        self.tables = dict(tables)
        self.fundamentals = fundamentals  # batt.data.fundamentals.Fundamentals, or None
        self.conditions = conditions  # batt.data.conditions.GridConditions, or None

    @classmethod
    def from_price_frames(
        cls, location: str, rt: pd.DataFrame, da: pd.DataFrame, timing: MarketTiming
    ) -> MarketData:
        """Build from clean price frames for one location (``batt.data.ercot.CLEAN_COLUMNS``)."""
        for name, df in (("rt", rt), ("da", da)):
            if "location" in df and df["location"].nunique() > 1:
                raise ValueError(f"{name} prices contain several locations; filter to one first")
            if df["interval_start_utc"].duplicated().any():
                raise ValueError(f"{name} prices have duplicate intervals for {location}")
        tables = {
            "rt": _price_table(
                rt, rt_publish_times(rt["interval_end_utc"], timing), timing.interval_minutes
            ),
            "da": _price_table(da, da_publish_times(da["interval_start_utc"], timing), 60),
        }
        return cls(location, tables)

    def view(self, now) -> DataView:
        """Everything published at or before ``now``; ``None`` gives the oracle view."""
        return DataView(self.tables, now, self.fundamentals, self.conditions)

    def aligned(self, name: str, starts: pd.DatetimeIndex) -> np.ndarray:
        """Final values for each interval start, ignoring publish times (used for settlement)."""
        table = self.tables[name]
        return align(table.series(None), starts, table.resolution_minutes)

    def with_tables(self, **tables: PointInTimeTable) -> MarketData:
        """A copy with some tables replaced or added (e.g. forecasts, or corrupted test data)."""
        return MarketData(
            self.location, {**self.tables, **tables}, self.fundamentals, self.conditions
        )

    def with_fundamentals(self, fundamentals) -> MarketData:
        """A copy with fundamentals data attached (or removed, with None)."""
        return MarketData(self.location, self.tables, fundamentals, self.conditions)

    def with_conditions(self, conditions) -> MarketData:
        """A copy with grid conditions attached (or removed, with None)."""
        return MarketData(self.location, self.tables, self.fundamentals, conditions)
