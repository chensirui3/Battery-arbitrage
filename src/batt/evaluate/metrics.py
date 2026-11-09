"""Backtest metrics: revenue, capture rate, cycling, and bootstrap confidence intervals.

Revenue is reported as $/kW-year, the industry's standard unit: under the price-taker
assumption revenue scales linearly with battery size, so this number doesn't depend on MW.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from batt.config import BatterySpec
from batt.sim.simulator import SimulationResult
from batt.timeutils import period_labels

DAYS_PER_YEAR = 365.25


@dataclass(frozen=True)
class BootstrapCI:
    mean: float
    low: float
    high: float
    level: float


def moving_block_bootstrap_mean(
    values, block: int = 7, n_boot: int = 2000, level: float = 0.95, seed: int = 0
) -> BootstrapCI:
    """Confidence interval for a mean of serially correlated daily values.

    Resamples contiguous blocks (default: weeks) so that runs of related days, like a heat
    wave, stay together. That matters here because a few days earn most of the revenue.
    """
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
    return BootstrapCI(
        float(x.mean()), float(np.quantile(means, tail)), float(np.quantile(means, 1 - tail)), level
    )


def daily_net_revenue(result: SimulationResult) -> pd.Series:
    """Net revenue (energy revenue minus degradation) per operating day, $.

    The last day also carries the SoC adjustment, so these days add up to the net revenue in
    ``summarize`` and every table (totals, periods, bootstrap CIs) agrees.
    """
    return ledger_daily_net_revenue(result.ledger, result.spec)


def ledger_daily_net_revenue(ledger: pd.DataFrame, spec: BatterySpec) -> pd.Series:
    """``daily_net_revenue`` from a ledger and its battery, e.g. a backtest read from disk."""
    daily = ledger.groupby("local_date")["revenue_net"].sum()
    daily.iloc[-1] += _soc_adjustment(ledger, spec)
    return daily


def soc_adjustment(result: SimulationResult, reference_price: float | None = None) -> float:
    """Value of the change in stored energy over the backtest, at a reference price.

    Energy left in the battery could still be sold; energy missing would have to be bought
    back. Valuing the difference keeps policies that end at different SoCs comparable. The
    default reference price is the average RT price of the backtest.
    """
    return _soc_adjustment(result.ledger, result.spec, reference_price)


def _soc_adjustment(
    ledger: pd.DataFrame, spec: BatterySpec, reference_price: float | None = None
) -> float:
    if reference_price is None:
        reference_price = float(ledger["rt_price"].mean())
    delta = float(ledger["soc_end_mwh"].iat[-1]) - spec.soc_initial_mwh
    if delta >= 0:
        return delta * spec.eta_discharge * reference_price
    return delta / spec.eta_charge * reference_price


def summarize(result: SimulationResult) -> dict:
    ledger, spec, grid = result.ledger, result.spec, result.grid
    years = grid.n_days / DAYS_PER_YEAR
    energy = float(ledger["revenue_energy"].sum())
    degradation = float(ledger["degradation_cost"].sum())
    adjustment = soc_adjustment(result)
    net = energy - degradation + adjustment
    discharged = float(ledger["discharge_mwh"].sum())
    charged = float(ledger["charge_mwh"].sum())
    selling = ledger["discharge_mwh"] > 0
    buying = ledger["charge_mwh"] > 0
    daily = daily_net_revenue(result)
    # Equivalent full cycles: energy taken out of storage divided by usable capacity.
    cycles = discharged / spec.eta_discharge / spec.usable_mwh
    return {
        "policy": result.policy,
        "oracle": result.is_oracle,
        "days": grid.n_days,
        "net_revenue": net,
        "usd_per_kw_year": net / (spec.power_mw * 1000) / years,
        "energy_revenue": energy,
        "degradation_cost": degradation,
        "soc_adjustment": adjustment,
        "cycles_per_day": cycles / grid.n_days,
        "avg_sell_price": float(ledger.loc[selling, "revenue_energy"].sum() / discharged)
        if discharged
        else float("nan"),
        "avg_buy_price": float(-ledger.loc[buying, "revenue_energy"].sum() / charged)
        if charged
        else float("nan"),
        "losing_day_share": float((daily < 0).mean()),
        "worst_day": daily.idxmin().date(),
        "worst_day_revenue": float(daily.min()),
        "clipped_intervals": int(ledger["clipped"].sum()),
    }


def _per_kw_year(daily: pd.Series, power_mw: float) -> float:
    return float(daily.mean() * DAYS_PER_YEAR / (power_mw * 1000))


def compare(
    results: Sequence[SimulationResult],
    benchmark: str | None = "perfect_rt",
    baseline: str | None = None,
    *,
    n_boot: int = 2000,
    level: float = 0.95,
) -> pd.DataFrame:
    """One row per policy; adds capture rate and, optionally, bootstrap CIs vs a baseline."""
    table = pd.DataFrame([summarize(r) for r in results]).set_index("policy")
    if benchmark in table.index:
        table["capture_rate"] = table["net_revenue"] / table.at[benchmark, "net_revenue"]
    by_name = {r.policy: r for r in results}
    if baseline in by_name:
        base_daily = daily_net_revenue(by_name[baseline])
        power = by_name[baseline].spec.power_mw
        prefix = f"vs_{baseline}"
        for name, result in by_name.items():
            if name == baseline:
                continue
            diff = daily_net_revenue(result).sub(base_daily, fill_value=0.0)
            ci = moving_block_bootstrap_mean(diff.to_numpy(), n_boot=n_boot, level=level)
            scale = DAYS_PER_YEAR / (power * 1000)
            table.at[name, f"{prefix}_usd_kw_yr"] = ci.mean * scale
            table.at[name, f"{prefix}_ci_low"] = ci.low * scale
            table.at[name, f"{prefix}_ci_high"] = ci.high * scale
    return table


def by_period(
    results: Sequence[SimulationResult], freq: str = "Y", benchmark: str | None = "perfect_rt"
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """$/kW-year per period for each policy, plus capture rates. ``freq`` "M" gives months;
    "Y" gives calendar years, with the days from ERCOT's RTC+B redesign on apart
    (``period_labels``), so that no year mixes the two market designs."""
    revenue, usd_kw_yr = {}, {}
    for r in results:
        daily = daily_net_revenue(r)
        periods = period_labels(daily.index) if freq == "Y" else daily.index.to_period(freq)
        grouped = daily.groupby(periods)
        revenue[r.policy] = grouped.sum()
        usd_kw_yr[r.policy] = grouped.apply(lambda d, p=r.spec.power_mw: _per_kw_year(d, p))
    usd = pd.DataFrame(usd_kw_yr).rename_axis("period")
    capture = None
    if benchmark in revenue:
        totals = pd.DataFrame(revenue).rename_axis("period")
        capture = totals.div(totals[benchmark], axis=0)
    return usd, capture
