"""Shared test utilities."""

import numpy as np
import pandas as pd

from batt.data.conditions import GridConditions
from batt.data.fundamentals import Fundamentals
from batt.data.store import MarketData, PointInTimeTable

# What each grid-conditions table holds, and the range of noise that replaces it.
CONDITION_NOISE = {
    "sced": {"price": (-250, 5_000), "prc": (0, 10_000)},
    "rtd": {"price": (-250, 5_000)},
    "outages": {"value": (0, 50_000)},
}


def corrupt_after(market: MarketData, cutoff: pd.Timestamp, seed: int = 0) -> MarketData:
    """A copy where every value published after ``cutoff`` is replaced by random noise:
    prices, fundamentals, and grid conditions alike."""
    rng = np.random.default_rng(seed)
    tables = {}
    for name, table in market.tables.items():
        frame = table.frame.copy()
        late = frame["publish_time_utc"] > cutoff
        frame.loc[late, "value"] = rng.uniform(-250, 5_000, int(late.sum()))
        tables[name] = PointInTimeTable(frame, table.resolution_minutes)
    corrupted = market.with_tables(**tables)
    if market.fundamentals is not None:
        frame = market.fundamentals.frame.copy()
        late = frame["publish_time_utc"] > cutoff
        frame.loc[late, "value"] = rng.uniform(0, 80_000, int(late.sum()))
        corrupted = corrupted.with_fundamentals(Fundamentals.from_frame(frame))
    if market.conditions is not None:
        frames = {k: v.copy() for k, v in market.conditions.frames.items()}
        for name, columns in CONDITION_NOISE.items():
            late = frames[name]["publish_time_utc"] > cutoff
            for column, (low, high) in columns.items():
                frames[name].loc[late, column] = rng.uniform(low, high, int(late.sum()))
        corrupted = corrupted.with_conditions(
            GridConditions.from_frames(frames["sced"], frames["rtd"], frames["outages"])
        )
    return corrupted
