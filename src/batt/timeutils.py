"""Timedelta helpers, and the market-design periods results are reported by.

``pd.Timedelta(minutes=5)`` emits a NumPy deprecation warning with pandas 2.3 on NumPy 2.5
(and NumPy plans to make it an error); the positional ``unit=`` form does not.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

RTC_B = date(2025, 12, 5)  # ERCOT's real-time market redesign (RTC+B) took effect


def minutes(n: float) -> pd.Timedelta:
    return pd.Timedelta(n, unit="min")


def hours(n: float) -> pd.Timedelta:
    return pd.Timedelta(n, unit="h")


def days(n: float) -> pd.Timedelta:
    return pd.Timedelta(n, unit="D")


def period_labels(local) -> np.ndarray:
    """The calendar year of each local time, as text, marked " RTC+B" from the day ERCOT's
    real-time market redesign took effect, so that no period mixes the two designs."""
    stamps = pd.DatetimeIndex(local)
    naive = stamps.tz_localize(None) if stamps.tz is not None else stamps
    years = np.asarray(naive.year).astype(str)
    return np.where(naive >= pd.Timestamp(RTC_B), np.char.add(years, " RTC+B"), years)
