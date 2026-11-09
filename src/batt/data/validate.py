"""Data-quality checks for stored price data.

Checks per market and location: every local operating day has the expected number of intervals
(96 for 15-minute data, or 92 and 100 on daylight-saving days), no duplicate or conflicting
intervals, no gaps, and no missing or implausible prices.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import pandas as pd

from batt.timeutils import days as _days
from batt.timeutils import hours as _hours

one_day, one_hour = _days(1), _hours(1)

RESOLUTION_MINUTES = {"rt": 15, "da": 60}
PLAUSIBLE_RANGE = (-1_000.0, 10_000.0)  # $/MWh; ERCOT's offer cap is $5,000


@dataclass
class PriceValidation:
    market: str
    location: str
    rows: int
    first_day: date | None
    last_day: date | None
    incomplete_days: list[tuple[date, int, int]] = field(default_factory=list)
    duplicate_rows: int = 0
    conflicting_intervals: int = 0
    missing_intervals: int = 0
    bad_prices: int = 0
    min_price: float = float("nan")
    max_price: float = float("nan")

    @property
    def ok(self) -> bool:
        return not (
            self.incomplete_days or self.duplicate_rows or self.missing_intervals or self.bad_prices
        )

    def summary(self) -> str:
        status = "OK  " if self.ok else "WARN"
        line = (
            f"{status} {self.market:<2} {self.location:<14} {self.rows:>8} rows  "
            f"{self.first_day} -> {self.last_day}  price {self.min_price:,.2f} .. "
            f"{self.max_price:,.2f}"
        )
        problems = []
        if self.incomplete_days:
            days = ", ".join(f"{d} ({n}/{e})" for d, n, e in self.incomplete_days[:3])
            more = "" if len(self.incomplete_days) <= 3 else f" +{len(self.incomplete_days) - 3}"
            problems.append(f"incomplete days: {days}{more}")
        if self.duplicate_rows:
            problems.append(
                f"{self.duplicate_rows} duplicate rows ({self.conflicting_intervals} conflicting)"
            )
        if self.missing_intervals:
            problems.append(f"{self.missing_intervals} missing intervals")
        if self.bad_prices:
            problems.append(f"{self.bad_prices} missing or implausible prices")
        return line + "".join(f"\n       - {p}" for p in problems)


def expected_intervals_per_day(days: pd.DatetimeIndex, timezone: str, minutes: int) -> pd.Series:
    """Intervals in each local day (23, 24, or 25 hours long around DST changes)."""
    starts = days.tz_localize(timezone)
    ends = (days + one_day).tz_localize(timezone)
    hours = (ends.tz_convert("UTC") - starts.tz_convert("UTC")) / one_hour
    return pd.Series((hours * 60 / minutes).astype(int), index=days.date)


def validate_prices(
    df: pd.DataFrame,
    market: str,
    timezone: str = "America/Chicago",
    plausible: tuple[float, float] = PLAUSIBLE_RANGE,
) -> list[PriceValidation]:
    """Validate clean price data (one or many locations) for a market."""
    minutes = RESOLUTION_MINUTES[market]
    reports = []
    for location, g in df.groupby("location", sort=True):
        local = g["interval_start_utc"].dt.tz_convert(timezone)
        local_day = local.dt.tz_localize(None).dt.normalize()
        first, last = local_day.min(), local_day.max()
        days = pd.date_range(first, last, freq="D")
        expected = expected_intervals_per_day(days, timezone, minutes)
        unique_starts = g["interval_start_utc"].drop_duplicates()
        per_day = (
            unique_starts.dt.tz_convert(timezone).dt.tz_localize(None).dt.normalize().dt.date
        ).value_counts()
        actual = per_day.reindex(expected.index, fill_value=0)
        incomplete = [
            (d, int(actual[d]), int(expected[d]))
            for d in expected.index
            if actual[d] != expected[d]
        ]
        dup_mask = g.duplicated(["interval_start_utc"], keep=False)
        conflicting = 0
        if dup_mask.any():
            conflicting = int(
                (g.loc[dup_mask].groupby("interval_start_utc")["price"].nunique() > 1).sum()
            )
        price = g["price"]
        bad = int(price.isna().sum() + ((price < plausible[0]) | (price > plausible[1])).sum())
        reports.append(
            PriceValidation(
                market=market,
                location=str(location),
                rows=len(g),
                first_day=first.date(),
                last_day=last.date(),
                incomplete_days=incomplete,
                duplicate_rows=int(g.duplicated(["interval_start_utc"]).sum()),
                conflicting_intervals=conflicting,
                missing_intervals=int(max(expected.sum() - len(unique_starts), 0)),
                bad_prices=bad,
                min_price=float(price.min()),
                max_price=float(price.max()),
            )
        )
    return reports
