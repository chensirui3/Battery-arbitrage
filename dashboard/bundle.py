"""Data and statistics behind the dashboard: plain pandas and NumPy, no Streamlit.

The data come from scripts/export_dashboard_data.py. The statistics repeat the project's own
(`batt pool`, `batt.timeutils.period_labels`), and tests/test_dashboard.py checks that they
agree, so the dashboard reproduces the README's numbers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent / "data"
DAYS_PER_YEAR = 365.25
RTC_B = date(2025, 12, 5)  # ERCOT's real-time market redesign
TIMEZONE = "US/Central"
BASELINE, ORACLE = "mpc_da", "perfect_rt"
INTERVAL_HOURS = 0.25

# Named periods for comparisons: (first day, day after the last), None for open ends.
PERIODS = {
    "January 2022 to October 2026": (None, None),
    "From 2024": (date(2024, 1, 1), None),
    "Before RTC+B (to December 4, 2025)": (None, RTC_B),
    "Since RTC+B (December 5, 2025)": (RTC_B, None),
    **{str(year): (date(year, 1, 1), date(year + 1, 1, 1)) for year in range(2022, 2026)},
}


@dataclass(frozen=True)
class Bundle:
    daily: pd.DataFrame  # hub, policy, date, revenue: net revenue per operating day, $
    intervals: pd.DataFrame  # one hub, every 15 minutes: prices, look-ahead, stored energy
    policies: pd.DataFrame  # indexed by policy: group, name, description
    forecasts: pd.DataFrame  # forecaster, lead, crps, mae
    nowcasts: pd.DataFrame  # forecaster, side, period, and skill against the prior
    manifest: dict

    @property
    def per_kw_year(self) -> float:
        """Converts a mean daily revenue in $ to $/kW-year."""
        return DAYS_PER_YEAR / (self.manifest["battery"]["power_mw"] * 1000)

    @property
    def per_kw(self) -> float:
        """Converts a revenue in $ to $/kW."""
        return 1 / (self.manifest["battery"]["power_mw"] * 1000)

    @property
    def hubs(self) -> list[str]:
        return list(self.manifest["hubs"])

    def run_at(self, hubs) -> list[str]:
        """Policies that ran at every hub in ``hubs``, in the order of the policy table."""
        ran = self.daily.groupby("policy")["hub"].unique()
        return [p for p in self.policies.index if p in ran and set(hubs) <= set(ran[p])]

    def label(self, policy: str) -> str:
        return f"{self.policies.at[policy, 'name']} ({policy})"


def load(folder: Path = DATA) -> Bundle:
    daily = pd.read_parquet(folder / "daily.parquet")
    for column in ("hub", "policy"):
        daily[column] = daily[column].astype(str)
    intervals = pd.read_parquet(folder / "intervals.parquet")
    for name in ("rt", "da", "lookahead"):
        intervals[name] = intervals.pop(f"{name}_cents").astype("float64") / 100
    local = intervals["time"].dt.tz_convert(TIMEZONE)
    midnight = local.dt.normalize()
    intervals["date"] = midnight.dt.tz_localize(None)
    # Hours since local midnight: unique within a day, 23 or 25 of them when the clocks change.
    intervals["hour"] = (local - midnight).dt.total_seconds() / 3600
    intervals["clock"] = local.dt.strftime("%H:%M")
    return Bundle(
        daily=daily,
        intervals=intervals,
        policies=pd.read_csv(folder / "policies.csv", index_col="policy"),
        forecasts=pd.read_csv(folder / "forecasts.csv"),
        nowcasts=pd.read_csv(folder / "nowcasts.csv"),
        manifest=json.loads((folder / "manifest.json").read_text(encoding="utf-8")),
    )


# --- Statistics, as in batt.evaluate and batt.timeutils ---------------------------------------


def period_labels(days) -> np.ndarray:
    """The calendar year of each day, marked " RTC+B" from the redesign on."""
    stamps = pd.DatetimeIndex(days)
    naive = stamps.tz_localize(None) if stamps.tz is not None else stamps
    years = np.asarray(naive.year).astype(str)
    return np.where(naive >= pd.Timestamp(RTC_B), np.char.add(years, " RTC+B"), years)


def bootstrap_mean(
    values, block: int = 7, n_boot: int = 2000, level: float = 0.95, seed: int = 0
) -> tuple[float, float, float]:
    """Mean and confidence interval of serially correlated daily values: a moving-block
    bootstrap over weeks, drawing the same blocks as the project's backtests do."""
    x = np.asarray(values, dtype=float)
    n = x.size
    if n == 0:
        raise ValueError("no values to bootstrap")
    block = max(1, min(block, n))
    n_blocks = -(-n // block)
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n - block + 1, size=(n_boot, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)).reshape(n_boot, -1)[:, :n]
    means = x[idx].mean(axis=1)
    tail = (1 - level) / 2
    return float(x.mean()), float(np.quantile(means, tail)), float(np.quantile(means, 1 - tail))


def window(frame, since: date | None = None, until: date | None = None):
    """The days in [since, until) of a frame or series indexed by day."""
    keep = np.ones(len(frame), dtype=bool)
    if since is not None:
        keep &= frame.index >= pd.Timestamp(since)
    if until is not None:
        keep &= frame.index < pd.Timestamp(until)
    return frame[keep]


def by_day(bundle: Bundle, hub: str) -> pd.DataFrame:
    """Net revenue per operating day at ``hub``, $: one column per policy."""
    part = bundle.daily[bundle.daily["hub"] == hub]
    return part.pivot(index="date", columns="policy", values="revenue")


def compare(
    bundle: Bundle,
    hubs,
    baseline: str,
    policies,
    since: date | None = None,
    until: date | None = None,
) -> pd.DataFrame:
    """Each policy minus ``baseline``, $/kW-year, with 95% CIs. Daily differences are averaged
    over ``hubs`` day by day, and the bootstrap resamples the same weeks at every hub, as
    `batt pool` does. Policies that did not run at every hub are left out."""
    tables = {hub: window(by_day(bundle, hub), since, until) for hub in hubs}
    rows = []
    for policy in policies:
        if policy == baseline or any(policy not in t for t in tables.values()):
            continue
        diff = pd.concat([t[policy] - t[baseline] for t in tables.values()], axis=1).mean(axis=1)
        mean, low, high = bootstrap_mean(diff.to_numpy())
        rows.append(
            {
                "policy": policy,
                "mean": mean * bundle.per_kw_year,
                "low": low * bundle.per_kw_year,
                "high": high * bundle.per_kw_year,
                "days": len(diff),
            }
        )
    return pd.DataFrame(rows, columns=["policy", "mean", "low", "high", "days"])


def levels(bundle: Bundle, hub: str, since=None, until=None) -> pd.DataFrame:
    """Each policy's $/kW-year and capture (its share of perfect-foresight revenue)."""
    table = window(by_day(bundle, hub), since, until)
    return pd.DataFrame(
        {
            "usd_kw_year": table.mean() * bundle.per_kw_year,
            "capture": table.sum() / table[ORACLE].sum(),
        }
    )


def by_period(bundle: Bundle, hub: str, policies) -> tuple[pd.DataFrame, pd.DataFrame]:
    """$/kW-year and capture by calendar year, the days from RTC+B on apart."""
    table = by_day(bundle, hub)
    labels = period_labels(table.index)
    usd = table[list(policies)].groupby(labels).mean() * bundle.per_kw_year
    totals = table.groupby(labels).sum()
    capture = totals[list(policies)].div(totals[ORACLE], axis=0)
    return usd.rename_axis("period"), capture.rename_axis("period")


def cumulative(bundle: Bundle, hub: str, policies) -> pd.DataFrame:
    """Cumulative net revenue, $/kW, day by day."""
    return by_day(bundle, hub)[list(policies)].cumsum() * bundle.per_kw


def day(bundle: Bundle, when: date, policies) -> pd.DataFrame:
    """One operating day at the interval hub: prices by interval start, and each policy's
    stored energy at the end of each interval, plus the level it started the day with."""
    frame = bundle.intervals
    rows = np.flatnonzero(frame["date"] == pd.Timestamp(when))
    if rows.size == 0:
        raise ValueError(f"no intervals on {when}")
    part = frame.iloc[rows][["hour", "clock", "rt", "da", "lookahead"]].reset_index(drop=True)
    for policy in policies:
        end = frame[f"soc_{policy}"].to_numpy()[rows].astype(float)
        before = frame[f"soc_{policy}"].iat[rows[0] - 1] if rows[0] > 0 else end[0]
        part[f"start_{policy}"] = np.concatenate([[before], end[:-1]])
        part[f"end_{policy}"] = end
    return part


def best_days(bundle: Bundle, hub: str, n: int = 10) -> pd.Series:
    """The days on which perfect foresight earned the most, $/kW."""
    return by_day(bundle, hub)[ORACLE].nlargest(n) * bundle.per_kw


def gaps(bundle: Bundle, hub: str, policy: str, baseline: str = BASELINE, n: int = 5):
    """A policy's worst and best days against ``baseline``, $/kW."""
    table = by_day(bundle, hub)
    diff = (table[policy] - table[baseline]) * bundle.per_kw
    return diff.nsmallest(n), diff.nlargest(n)
