"""Pool saved backtests across locations: daily revenue averaged over locations, day by day.

Used for comparisons fixed in advance across ERCOT's trading hubs (see docs/). The hubs share
weather and scarcity events, so they are not independent samples. The bootstrap therefore
resamples the same weeks at every hub: a heat wave stays one event, not five.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date
from pathlib import Path

import pandas as pd

from batt.config import load_config
from batt.evaluate.metrics import (
    DAYS_PER_YEAR,
    ledger_daily_net_revenue,
    moving_block_bootstrap_mean,
)
from batt.timeutils import period_labels

POOLED = "pooled"


def latest_runs(
    results_dir: Path, locations: Sequence[str], start: date, end: date
) -> dict[str, Path]:
    """The most recently written backtest folder of each location for one period."""
    runs = {}
    for location in locations:
        found = sorted(
            results_dir.glob(f"{location}_{start}_{end}_*"), key=lambda p: p.stat().st_mtime
        )
        if not found:
            raise FileNotFoundError(
                f"no backtest of {location} from {start} to {end} in {results_dir}"
            )
        runs[location] = found[-1]
    return runs


def run_location(run: Path) -> str:
    """The location a saved backtest folder was run for."""
    return load_config(run / "config.yaml").backtest.location


def pool_runs(
    runs: Mapping[str, Path],
    baseline: str,
    policies: Sequence[str],
    *,
    since: date | None = None,
    until: date | None = None,
    n_boot: int = 2000,
    level: float = 0.95,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Revenue per location and pooled, and each policy minus ``baseline`` with bootstrap CIs.

    ``runs`` maps each location to a saved backtest folder; only days in [since, until) count.

    Returns ``(levels, differences)``, both in $/kW-year:

    - ``levels``: one row per location plus "pooled", one column per policy. A location that
      did not run a policy has NaN there, and so does the pooled row.
    - ``differences``: indexed by (policy, scope), with columns mean, low, high. The scope is a
      location, "pooled" (daily differences averaged over locations), or "pooled <period>": a
      calendar year, with the days from ERCOT's RTC+B redesign on apart (``period_labels``).
      Pooled rows need every location to have run the policy.
    """
    names = [baseline, *(p for p in policies if p != baseline)]
    daily: dict[str, dict[str, pd.Series]] = {}
    powers = set()
    for location, run in runs.items():
        spec = load_config(run / "config.yaml").battery
        powers.add(spec.power_mw)
        daily[location] = {}
        for name in names:
            path = run / "ledgers" / f"{name}.parquet"
            if path.exists():
                revenue = ledger_daily_net_revenue(pd.read_parquet(path), spec)
                revenue.index = pd.to_datetime(revenue.index)
                daily[location][name] = _window(revenue, since, until)
        if baseline not in daily[location]:
            raise ValueError(f"{run} has no ledger for the baseline {baseline}")
    if len(powers) != 1:
        raise ValueError(f"the runs use different battery sizes: {sorted(powers)} MW")
    scale = DAYS_PER_YEAR / (powers.pop() * 1000)

    levels = pd.DataFrame(
        {
            name: {
                location: d[name].mean() * scale if name in d else float("nan")
                for location, d in daily.items()
            }
            for name in names
        }
    )
    levels.loc[POOLED] = levels.mean(skipna=False)

    def row(policy: str, scope: str, values: pd.Series) -> dict:
        ci = moving_block_bootstrap_mean(values.to_numpy(), n_boot=n_boot, level=level)
        return {
            "policy": policy,
            "scope": scope,
            "mean": ci.mean * scale,
            "low": ci.low * scale,
            "high": ci.high * scale,
        }

    rows = []
    for name in names[1:]:
        diffs = {loc: d[name] - d[baseline] for loc, d in daily.items() if name in d}
        frame = pd.DataFrame(diffs)
        if frame.isna().any().any():
            raise ValueError("the runs cover different days; pool runs of the same period")
        rows += [row(name, loc, diff) for loc, diff in diffs.items()]
        if len(diffs) < len(runs):
            continue
        pooled = frame.mean(axis=1)
        rows.append(row(name, POOLED, pooled))
        rows += [
            row(name, f"{POOLED} {period}", part)
            for period, part in pooled.groupby(period_labels(pooled.index))
        ]
    differences = pd.DataFrame(rows, columns=["policy", "scope", "mean", "low", "high"])
    return levels, differences.set_index(["policy", "scope"])


def _window(daily: pd.Series, since: date | None, until: date | None) -> pd.Series:
    keep = pd.Series(True, index=daily.index)
    if since is not None:
        keep &= daily.index >= pd.Timestamp(since)
    if until is not None:
        keep &= daily.index < pd.Timestamp(until)
    if not keep.any():
        raise ValueError(f"no days between {since} and {until}")
    return daily[keep]
