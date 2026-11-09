"""Walk-forward accuracy of the price forecasters, issued the way the MPC policies use them.

Point accuracy (MAE, RMSE, bias) is scored on each forecaster's point forecast, which for a
scenario forecaster is the scenario average. The whole distribution is scored with the CRPS
(continuous ranked probability score): the expected absolute error of a draw from the
forecast, minus half the expected distance between two draws. It is a proper score, lower is
better, and for a single-valued forecast it equals the absolute error, so scenario and point
forecasters are directly comparable. Coverage is how often the realized price falls inside
the central 50% and 80% ranges of the scenarios (ideally 50% and 80%).
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date

import numpy as np
import pandas as pd

from batt.config import MarketTiming
from batt.data.store import MarketData, ts_ns
from batt.forecast.history import MINUTE_NS, published_horizon
from batt.forecast.models import Forecaster
from batt.sim.grid import SimulationGrid
from batt.timeutils import days, minutes, period_labels

MAX_STEPS = 200  # longest horizon tracked (intervals); published horizons stop near 140
HORIZON_BUCKETS_H = (0, 1, 4, 12, 24, 36)  # lead-time bucket edges in hours
SUMS = (
    "count",
    "sum_error",
    "sum_abs_error",
    "sum_sq_error",
    "sum_crps",
    "dist_count",
    "in_50",
    "in_80",
)


def crps(samples: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """CRPS of equally weighted samples (rows) for each column's actual value."""
    s = np.sort(samples, axis=0)
    n = s.shape[0]
    spread_weights = (2 * np.arange(1, n + 1) - n - 1)[:, None]
    return np.abs(s - actual).mean(axis=0) - (spread_weights * s).sum(axis=0) / n**2


def evaluate_forecasters(
    market: MarketData,
    forecasters: Mapping[str, Forecaster],
    start: date,
    end: date,
    timing: MarketTiming,
    issue_every: int = 4,
) -> pd.DataFrame:
    """Out-of-sample forecast errors, aggregated by forecaster, year (the days after RTC+B
    apart, see ``period_labels``), and steps ahead.

    Every ``issue_every`` intervals in [start, end), each forecaster predicts the RT price from
    the next interval to the end of the last day with published DA prices, seeing only data
    published by then and refitting on its own schedule, exactly as inside the backtest.
    """
    grid = SimulationGrid.for_days(start, end, timing.timezone, timing.interval_minutes)
    step_ns = grid.interval_minutes * MINUTE_NS
    lead = minutes(timing.decision_lead_minutes)
    targets = pd.date_range(
        grid.starts[0], grid.starts[-1] + days(2), freq=f"{grid.interval_minutes}min"
    )
    actual = market.aligned("rt", targets)  # same regular UTC grid as grid.starts
    oracle = market.view(None)
    for forecaster in forecasters.values():
        forecaster.reset(timing)

    sums: dict[tuple[str, str], np.ndarray] = {}
    years = period_labels(grid.local_starts)
    for i in range(0, len(grid), issue_every):
        first = grid.starts[i]
        now = first - lead
        view = market.view(now)
        n = min(published_horizon(view, ts_ns(first), step_ns), MAX_STEPS)
        if n == 0:
            continue
        truth = actual[i : i + n]
        for name, forecaster in forecasters.items():
            forecaster.prepare(view, now, first)
            source = oracle if forecaster.needs_oracle else view
            paths = forecaster.scenarios(source, now, first, n)
            error = paths.mean(axis=0) - truth
            ok = np.isfinite(error)
            acc = sums.setdefault((name, str(years[i])), np.zeros((len(SUMS), MAX_STEPS)))
            acc[0, :n] += ok
            acc[1, :n] += np.where(ok, error, 0.0)
            acc[2, :n] += np.where(ok, np.abs(error), 0.0)
            acc[3, :n] += np.where(ok, error**2, 0.0)
            acc[4, :n] += np.where(ok, crps(np.nan_to_num(paths), np.nan_to_num(truth)), 0.0)
            if paths.shape[0] > 1:
                q10, q25, q75, q90 = np.quantile(np.nan_to_num(paths), [0.1, 0.25, 0.75, 0.9], 0)
                acc[5, :n] += ok
                acc[6, :n] += ok & (truth >= q25) & (truth <= q75)
                acc[7, :n] += ok & (truth >= q10) & (truth <= q90)

    rows = []
    for (name, year), acc in sums.items():
        used = np.flatnonzero(acc[0])
        frame = pd.DataFrame(
            {
                "forecaster": name,
                "year": year,
                "step": used,
                "lead_h": (used * grid.interval_minutes + timing.decision_lead_minutes) / 60,
            }
        )
        for j, column in enumerate(SUMS):
            frame[column] = acc[j, used]
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def summarize_forecasts(errors: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Accuracy tables by lead-time bucket and by year, from ``evaluate_forecasters``."""
    edges = HORIZON_BUCKETS_H
    labels = [f"{a}-{b}h" for a, b in zip(edges[:-1], edges[1:], strict=False)]
    bucket = pd.cut(errors["lead_h"], edges, labels=labels, right=False)

    def table(keys) -> pd.DataFrame:
        sums = errors.groupby(keys, observed=True)[list(SUMS)].sum()
        with np.errstate(invalid="ignore", divide="ignore"):
            return pd.DataFrame(
                {
                    "mae": sums["sum_abs_error"] / sums["count"],
                    "rmse": np.sqrt(sums["sum_sq_error"] / sums["count"]),
                    "bias": sums["sum_error"] / sums["count"],
                    "crps": sums["sum_crps"] / sums["count"],
                    "coverage_50": sums["in_50"] / sums["dist_count"].replace(0, np.nan),
                    "coverage_80": sums["in_80"] / sums["dist_count"].replace(0, np.nan),
                    "count": sums["count"],
                }
            )

    by_lead = table([errors["forecaster"], bucket.rename("lead")])
    by_year = table([errors["forecaster"], errors["year"]])
    out = {f"{metric}_by_lead": by_lead[metric].unstack("lead") for metric in by_lead}
    out.update({f"{metric}_by_year": by_year[metric].unstack("year") for metric in by_year})
    out["overall"] = table([errors["forecaster"]])
    out.pop("count_by_lead")
    out.pop("count_by_year")
    return out
