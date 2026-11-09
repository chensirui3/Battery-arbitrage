"""Leakage-safe backtest engine.

One simulator runs every policy. A clock walks through the intervals. Before each one it asks
the policy for an action, showing it only data published by then, then applies battery physics
and settles at realized prices. Policies never see settlement prices or change the battery
state directly, so every strategy gets the same physics and the same accounting.

Settlement follows ERCOT's two-settlement design: a day-ahead position is paid the DA price,
and any difference between physical output and that position is settled at the RT price.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from batt.config import BatterySpec, MarketTiming
from batt.data.store import MarketData
from batt.sim.context import DayAheadContext, DecisionContext
from batt.sim.grid import SimulationGrid
from batt.timeutils import minutes

if TYPE_CHECKING:
    from batt.policies.base import Policy

CLIP_TOL_MW = 1e-6


@dataclass
class SimulationResult:
    policy: str
    is_oracle: bool
    location: str
    ledger: pd.DataFrame
    spec: BatterySpec
    timing: MarketTiming
    grid: SimulationGrid
    notes: dict = field(default_factory=dict)


def apply_physics(requested_mw: float, soc_mwh: float, spec: BatterySpec, dt: float):
    """Clip a requested net power to what the battery can do this interval.

    Returns ``(charge_mw, discharge_mw, soc_after_mwh)``.
    """
    p = min(max(requested_mw, -spec.power_mw), spec.power_mw)
    charge = discharge = 0.0
    if p > 0:
        available = max(soc_mwh - spec.soc_min_mwh, 0.0) * spec.eta_discharge / dt
        discharge = min(p, available)
    elif p < 0:
        headroom = max(spec.soc_max_mwh - soc_mwh, 0.0) / (spec.eta_charge * dt)
        charge = min(-p, headroom)
    soc_after = soc_mwh + (spec.eta_charge * charge - discharge / spec.eta_discharge) * dt
    soc_after = min(max(soc_after, spec.soc_min_mwh), spec.soc_max_mwh)
    return charge, discharge, soc_after


def da_decision_times(grid: SimulationGrid, timing: MarketTiming) -> list[pd.Timestamp]:
    """When day-ahead bids for each operating day are committed (UTC)."""
    lead = minutes(timing.decision_lead_minutes)
    out = []
    for day in grid.days:
        deadline = datetime.combine(day - timedelta(days=1), timing.da_bid_deadline_local_time)
        out.append(pd.Timestamp(deadline).tz_localize(timing.timezone).tz_convert("UTC") - lead)
    return out


def simulate(
    policy: Policy,
    market: MarketData,
    grid: SimulationGrid,
    spec: BatterySpec,
    timing: MarketTiming,
) -> SimulationResult:
    """Run one policy over the grid and return a per-interval ledger."""
    if grid.interval_minutes != timing.interval_minutes:
        raise ValueError("grid and timing disagree on the interval length")
    n, dt = len(grid), grid.dt_hours

    rt_price = market.aligned("rt", grid.starts)
    missing = np.isnan(rt_price)
    if missing.any():
        first = grid.starts[np.argmax(missing)]
        raise ValueError(f"{int(missing.sum())} intervals have no RT price (first: {first})")
    da_price = market.aligned("da", grid.starts)

    oracle = market.view(None) if policy.is_oracle else None
    policy.reset(grid=grid, spec=spec, timing=timing)

    decision_times = grid.starts - minutes(timing.decision_lead_minutes)
    da_times = da_decision_times(grid, timing)
    next_da_day = 0

    requested = np.zeros(n)
    charge = np.zeros(n)
    discharge = np.zeros(n)
    soc_start = np.zeros(n)
    da_position = np.zeros(n)
    soc = spec.soc_initial_mwh

    for i in range(n):
        now = decision_times[i]
        while next_da_day < grid.n_days and da_times[next_da_day] <= now:
            t = da_times[next_da_day]
            day_ctx = DayAheadContext(t, next_da_day, grid, spec, timing, market.view(t), oracle)
            positions = policy.plan_day_ahead(day_ctx)
            if positions is not None:
                da_position[day_ctx.intervals] = _check_positions(positions, day_ctx, spec)
            next_da_day += 1

        ctx = DecisionContext(now, i, soc, grid, spec, timing, market.view(now), oracle)
        req = float(policy.decide(ctx))
        if not math.isfinite(req):
            raise ValueError(f"{policy.name} returned {req} at {grid.starts[i]}")
        c, d, soc_after = apply_physics(req, soc, spec, dt)
        requested[i], charge[i], discharge[i], soc_start[i] = req, c, d, soc
        soc = soc_after

    if np.any((da_position != 0) & np.isnan(da_price)):
        raise ValueError("day-ahead positions were taken in intervals without a DA price")
    net = discharge - charge
    revenue_da = np.where(da_position != 0, np.nan_to_num(da_price) * da_position * dt, 0.0)
    revenue_rt = rt_price * (net - da_position) * dt
    revenue_energy = revenue_da + revenue_rt
    degradation = spec.degradation_cost_per_mwh * discharge * dt
    soc_end = np.r_[soc_start[1:], soc]

    ledger = pd.DataFrame(
        {
            "interval_start_utc": grid.starts,
            "local_date": grid.local_dates,
            "rt_price": rt_price,
            "da_price": da_price,
            "requested_mw": requested,
            "charge_mw": charge,
            "discharge_mw": discharge,
            "net_mw": net,
            "da_position_mw": da_position,
            "soc_start_mwh": soc_start,
            "soc_end_mwh": soc_end,
            "charge_mwh": charge * dt,
            "discharge_mwh": discharge * dt,
            "revenue_da": revenue_da,
            "revenue_rt": revenue_rt,
            "revenue_energy": revenue_energy,
            "degradation_cost": degradation,
            "revenue_net": revenue_energy - degradation,
            "clipped": np.abs(net - requested) > CLIP_TOL_MW,
        }
    )
    notes = {k: v for k, v in vars(policy).items() if k.startswith("days_without")}
    return SimulationResult(
        policy=policy.name,
        is_oracle=policy.is_oracle,
        location=market.location,
        ledger=ledger,
        spec=spec,
        timing=timing,
        grid=grid,
        notes=notes,
    )


def _check_positions(positions, ctx: DayAheadContext, spec: BatterySpec) -> np.ndarray:
    sl = ctx.intervals
    arr = np.asarray(positions, dtype=float)
    if arr.shape != (sl.stop - sl.start,):
        raise ValueError("day-ahead positions must have one value per interval of the day")
    if not np.all(np.isfinite(arr)) or np.any(np.abs(arr) > spec.power_mw + CLIP_TOL_MW):
        raise ValueError("day-ahead positions must be finite and within the power rating")
    return arr
