"""Synthetic ERCOT-like prices for tests and offline demos.

Real-time prices follow a daily shape (midday solar dip, evening peak), a random daily level,
autocorrelated noise, occasional negative midday prices, and rare evening spikes. Day-ahead
prices are an hourly, smoothed view of the same day that only partly anticipates spikes, the
way real DA prices relate to RT.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd


def _local_day_range(first_day: date, n_days: int, timezone: str, minutes: int) -> pd.DatetimeIndex:
    start = pd.Timestamp(first_day).tz_localize(timezone).tz_convert("UTC")
    end = pd.Timestamp(first_day + timedelta(days=n_days)).tz_localize(timezone).tz_convert("UTC")
    return pd.date_range(start, end, freq=f"{minutes}min", inclusive="left")


def _frame(starts: pd.DatetimeIndex, minutes: int, prices: np.ndarray, market: str, location: str):
    return pd.DataFrame(
        {
            "interval_start_utc": starts,
            "interval_end_utc": starts + pd.Timedelta(minutes, unit="min"),
            "location": location,
            "location_type": "Synthetic",
            "market": market,
            "price": prices,
        }
    )


def synthetic_prices(
    first_day: date,
    n_days: int,
    *,
    timezone: str = "America/Chicago",
    location: str = "SYN_HUB",
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (rt, da) frames in the clean schema used by ``batt.data.ercot``."""
    rng = np.random.default_rng(seed)
    starts = _local_day_range(first_day, n_days, timezone, 15)
    local = starts.tz_convert(timezone)
    hour = local.hour.to_numpy() + local.minute.to_numpy() / 60
    day = (local.tz_localize(None).normalize() - pd.Timestamp(first_day)).days.to_numpy()

    shape = 32 - 14 * np.exp(-(((hour - 13) / 3) ** 2)) + 38 * np.exp(-(((hour - 19.5) / 1.6) ** 2))
    level = rng.normal(0, 6, n_days)[day]
    noise = np.zeros(len(starts))
    eps = rng.normal(0, 5, len(starts))
    for i in range(1, len(starts)):
        noise[i] = 0.8 * noise[i - 1] + eps[i]
    rt = shape + level + noise

    midday = (hour >= 11) & (hour < 15)
    negative_days = rng.random(n_days) < 0.10
    rt[midday & negative_days[day]] -= rng.uniform(40, 70)

    spike_days = np.flatnonzero(rng.random(n_days) < 0.06)
    spike_size = np.zeros(n_days)
    for d in spike_days:
        height = rng.uniform(300, 2_500)
        spike_size[d] = height
        center = rng.uniform(18, 21)
        rt += (day == d) * height * np.exp(-(((hour - center) / 0.4) ** 2))
    rt = np.clip(rt, -250, 5_000)

    # Day-ahead: hourly average of the "expected" RT (no noise) plus a partial view of spikes.
    hourly = pd.Series(shape + level, index=starts).resample("1h").mean()
    da_starts = hourly.index
    da_local = da_starts.tz_convert(timezone)
    da_day = (da_local.tz_localize(None).normalize() - pd.Timestamp(first_day)).days.to_numpy()
    da_hour = da_local.hour.to_numpy() + 0.5
    anticipated = 0.25 * spike_size[da_day] * np.exp(-(((da_hour - 19.5) / 1.5) ** 2))
    da = hourly.to_numpy() + anticipated + rng.normal(0, 3, len(da_starts))

    return (
        _frame(starts, 15, np.round(rt, 2), "rt", location),
        _frame(da_starts, 60, np.round(da, 2), "da", location),
    )


# How fast each fundamentals forecast's error shrinks: full size a day or more ahead, zero at
# the hour itself. Errors are per target hour, so every forecast of that hour is consistently
# off in the same direction, as real weather-driven misses are.
_ERROR_MW = {"load": 1500.0, "wind": 1500.0, "solar": 1000.0}


def _error_weight(lead_hours) -> np.ndarray:
    return np.clip(np.asarray(lead_hours, dtype=float) / 24.0, 0.0, 1.0)


def synthetic_fundamentals(
    first_day: date,
    n_days: int,
    *,
    timezone: str = "America/Chicago",
    seed: int = 0,
    deadline_hour: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """ERCOT-like load, wind, and solar actuals and hourly forecast vintages.

    Returns ``(rows, truth)``: rows in the ``batt.data.fundamentals`` schema, and an hourly
    frame (indexed by UTC hour start) with the actuals and ``net_surprise``, the actual net
    load minus the net-load forecast that was current at the DA bid deadline for that hour.
    Load forecasts are published at :30 and wind/solar at :55 every hour, each covering the
    next 48 hours; wind and solar actuals are published 55 minutes after their hour ends.
    """
    from batt.data.fundamentals import MAX_LEAD_HOURS

    rng = np.random.default_rng(seed)
    hours = _local_day_range(first_day, n_days + 2, timezone, 60)
    n = hours.size
    local_hour = hours.tz_convert(timezone).hour.to_numpy()
    daily = np.repeat(rng.normal(0, 3000, n // 24 + 2), 24)[:n]
    load = 45000 + 8000 * np.exp(-(((local_hour - 17) / 4.0) ** 2)) + daily + rng.normal(0, 300, n)
    wind = np.empty(n)
    wind[0] = 8000
    for t in range(1, n):
        wind[t] = 8000 + 0.95 * (wind[t - 1] - 8000) + rng.normal(0, 600)
    wind = np.clip(wind, 300, None)
    sun = np.clip(np.sin(np.pi * (local_hour - 7) / 12), 0, None)
    solar = 15000 * sun * rng.uniform(0.5, 1.0, n // 24 + 2).repeat(24)[:n]
    actual = {"load": load, "wind": wind, "solar": solar}
    error = {k: rng.normal(0, _ERROR_MW[k], n) for k in actual}

    rows = []
    for kind, minute in (("load", 30), ("wind", 55), ("solar", 55)):
        publish = hours + pd.Timedelta(minute, unit="min")
        p, lead = np.meshgrid(np.arange(n), np.arange(MAX_LEAD_HOURS), indexing="ij")
        target = p + lead
        ok = target < n
        p, lead, target = p[ok], lead[ok], target[ok]
        noise = rng.normal(0, 50, target.size)
        value = actual[kind][target] + error[kind][target] * _error_weight(lead) + noise
        rows.append(
            pd.DataFrame(
                {
                    "series": f"{kind}_forecast",
                    "publish_time_utc": publish[p],
                    "interval_start_utc": hours[target],
                    "value": np.maximum(value, 0.0) if kind != "load" else value,
                }
            )
        )
        if kind != "load":
            rows.append(
                pd.DataFrame(
                    {
                        "series": f"{kind}_actual",
                        "publish_time_utc": hours + pd.Timedelta(115, unit="min"),
                        "interval_start_utc": hours,
                        "value": actual[kind],
                    }
                )
            )

    # The forecast in use at the DA bid deadline for each hour's operating day.
    local_day = hours.tz_convert(timezone).normalize().tz_localize(None)
    deadline = local_day - pd.Timedelta(1, unit="D") + pd.Timedelta(deadline_hour, unit="h")
    deadline = deadline.tz_localize(timezone).tz_convert("UTC")
    lead_at_bids = ((hours - deadline.floor("h")) / pd.Timedelta(1, unit="h")).to_numpy()
    weight = _error_weight(lead_at_bids)
    surprise = -(error["load"] - error["wind"] - error["solar"]) * weight
    truth = pd.DataFrame({**actual, "net_surprise": surprise}, index=hours)
    return pd.concat(rows, ignore_index=True), truth


def synthetic_conditions(
    rt: pd.DataFrame, *, seed: int = 0, steps: int = 11, rtd_noise: float = 4.0
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """ERCOT-like grid conditions consistent with synthetic RT prices (``synthetic_prices``).

    Returns ``(sced, rtd, outages)`` in the ``batt.data.conditions`` schema:

    - a SCED run every 5 minutes, posted 3 seconds later: the RT price of its interval plus
      noise, and PRC that falls as prices rise;
    - an RTD run every 5 minutes, posted 15 seconds later, projecting the next ``steps``
      5-minute intervals: their RT prices plus noise, ``rtd_noise`` $/MWh times the lead in
      5-minute steps;
    - an outage publication every hour at :02, covering the next 48 hours.
    """
    rng = np.random.default_rng(seed)
    starts = pd.DatetimeIndex(rt["interval_start_utc"])
    price = rt["price"].to_numpy(dtype=float)
    five = pd.Timedelta(5, unit="min")
    runs = pd.date_range(starts[0], starts[-1] + 2 * five, freq="5min") + pd.Timedelta(10, "s")
    interval = np.clip(starts.searchsorted(runs, side="right") - 1, 0, len(starts) - 1)
    sced = pd.DataFrame(
        {
            "publish_time_utc": runs + pd.Timedelta(3, "s"),
            "time_utc": runs,
            "price": price[interval] + rng.normal(0, 5, runs.size),
            "prc": 6_000 - 1.5 * price[interval] + rng.normal(0, 200, runs.size),
        }
    )

    rows = []
    for j in range(1, steps + 1):
        ends = runs.floor("5min") + j * five
        target = np.clip(starts.searchsorted(ends - five, side="right") - 1, 0, len(starts) - 1)
        rows.append(
            pd.DataFrame(
                {
                    "publish_time_utc": runs + pd.Timedelta(15, "s"),
                    "run_time_utc": runs,
                    "interval_end_utc": ends,
                    "price": price[target] + rng.normal(0, rtd_noise * j, runs.size),
                }
            )
        )
    rtd = pd.concat(rows, ignore_index=True)

    hours = pd.date_range(starts[0].floor("h"), starts[-1].ceil("h"), freq="1h")
    level = 8_000 + np.cumsum(rng.normal(0, 150, hours.size + 48))
    publish, target = np.meshgrid(np.arange(hours.size), np.arange(48), indexing="ij")
    outages = pd.DataFrame(
        {
            "series": "outage_resource",
            "publish_time_utc": hours[publish.ravel()] + pd.Timedelta(2, unit="min"),
            "interval_start_utc": hours[publish.ravel()] + pd.to_timedelta(target.ravel(), "h"),
            "value": level[(publish + target).ravel()] + rng.normal(0, 100, publish.size),
        }
    )
    for frame in (sced, rtd, outages):
        for column in [c for c in frame if c.endswith("_utc")]:
            frame[column] = pd.DatetimeIndex(frame[column]).tz_convert("UTC").as_unit("ns")
    return sced, rtd, outages
