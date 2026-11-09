"""Quantile LightGBM with split-conformal calibration: a predictive distribution per interval.

One LightGBM model per quantile level is trained on the RT-DA spread, with the same
point-in-time features as the ``gbm`` forecaster and the pinball (quantile) loss, which is
robust to spikes without discarding them. Quantile models are often miscalibrated out of
sample, so the most recent ``calib_days`` of each training window are held out and used to
shift each level until it is a calibrated quantile on that data, separately by lead time:
conformalized quantile regression (Romano, Patterson & Candes, 2019), one level at a time.
"""

from __future__ import annotations

from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from batt.data.store import ts_ns
from batt.forecast.features import (
    COLUMN,
    LOOKBACK_DAYS,
    build_features,
    conditions_of,
    fundamentals_of,
    training_set,
)
from batt.forecast.history import DAY_NS, MINUTE_NS, PriceHistory
from batt.forecast.models import (
    DEFAULT_GBM_PARAMS,
    Forecaster,
    Refit,
    RefitSchedule,
    feature_names,
)

DEFAULT_LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)
LEAD_EDGES_H = (1.0, 4.0, 12.0, 24.0)  # calibration buckets by lead time
MIN_BUCKET = 200  # smaller buckets use the offsets of all lead times together


def conformal_offset(residuals: np.ndarray, level: float) -> float:
    """Shift that makes a ``level`` quantile calibrated on held-out residuals (y - q)."""
    n = residuals.size
    adjusted = min(1.0, np.ceil((n + 1) * level) / n)  # finite-sample correction
    return float(np.quantile(residuals, adjusted, method="higher"))


class QuantileGBMForecaster(Forecaster):
    """Calibrated quantiles of the RT price; scenarios are evenly spaced quantiles.

    ``scenarios`` evaluates the calibrated quantile function at levels (i + 0.5) / n, with
    linear extrapolation beyond the outermost modeled levels. These paths move together
    (each is one quantile throughout), which is right for scoring each interval's
    distribution but is not a model of how errors evolve over time. The forecast
    (``predict``) is their average.
    """

    name = "qgbm"

    def __init__(
        self,
        levels=DEFAULT_LEVELS,
        n_scenarios: int = 20,
        refit: Refit = "month",
        train_days: int = 730,
        calib_days: int = 60,
        num_boost_round: int = 200,
        seed: int = 0,
        params: dict[str, Any] | None = None,
        fundamentals: bool = False,
        conditions: bool = False,
    ):
        super().__init__()
        self.fundamentals = fundamentals
        self.conditions = conditions
        self.levels = np.asarray(sorted(levels), dtype=float)
        if self.levels.size < 2 or self.levels[0] <= 0 or self.levels[-1] >= 1:
            raise ValueError("need at least two quantile levels strictly between 0 and 1")
        self._n_scenarios = n_scenarios
        self.train_days = train_days
        self.calib_days = calib_days
        self.num_boost_round = num_boost_round
        self.seed = seed
        self.params = {**DEFAULT_GBM_PARAMS, **(params or {}), "seed": seed}
        self.schedule = RefitSchedule(refit)
        self.boosters: list[lgb.Booster] = []
        self.offsets = np.zeros((len(LEAD_EDGES_H) + 1, self.levels.size))
        self.fit_times: list[pd.Timestamp] = []

    @property
    def n_scenarios(self) -> int:
        return self._n_scenarios

    def reset(self, timing):
        super().reset(timing)
        self.schedule.reset()
        self.boosters = []
        self.offsets[:] = 0.0
        self.fit_times = []

    def prepare(self, view, now, first):
        if not self.schedule.due(first, self.timezone):
            return
        x, y, issue = training_set(
            view,
            now,
            self.train_days,
            self.lead_ns,
            self.timezone,
            seed=self.seed,
            fundamentals=self.fundamentals,
            deadline_minutes=self.deadline_minutes,
            conditions=self.conditions,
        )
        held_out = issue >= ts_ns(now) - self.calib_days * DAY_NS
        if (~held_out).sum() < 1000 or held_out.sum() < MIN_BUCKET:
            return  # too little history; keep the previous models (or DA)
        names = feature_names(self.fundamentals, self.conditions)
        boosters = []
        for level in self.levels:
            params = {**self.params, "objective": "quantile", "alpha": float(level)}
            data = lgb.Dataset(x[~held_out], y[~held_out], feature_name=names)
            boosters.append(lgb.train(params, data, num_boost_round=self.num_boost_round))
        raw = np.column_stack([b.predict(x[held_out]) for b in boosters])
        residuals = y[held_out][:, None] - raw
        bucket = np.digitize(x[held_out][:, COLUMN["lead_h"]], LEAD_EDGES_H)
        everything = [conformal_offset(residuals[:, j], lv) for j, lv in enumerate(self.levels)]
        offsets = np.tile(everything, (len(LEAD_EDGES_H) + 1, 1))
        for b in range(offsets.shape[0]):
            mine = bucket == b
            if mine.sum() >= MIN_BUCKET:
                offsets[b] = [
                    conformal_offset(residuals[mine, j], lv) for j, lv in enumerate(self.levels)
                ]
        self.boosters, self.offsets = boosters, offsets
        self.fit_times.append(pd.Timestamp(now))

    def quantiles(self, view, now, first, n) -> np.ndarray:
        """Calibrated quantiles of the RT price, shape ``(levels, n)``; NaN where unknown."""
        issue, first_ns = ts_ns(now), ts_ns(first)
        step = view.resolution("rt") * MINUTE_NS
        start = issue - LOOKBACK_DAYS * DAY_NS
        hist = PriceHistory.from_view(view, start, first_ns + n * step, self.timezone)
        fund = fundamentals_of(view) if self.fundamentals else None
        cond = conditions_of(view) if self.conditions else None
        k = hist.index(first_ns) + np.arange(n)
        x = build_features(hist, issue, k, fund, self.deadline_minutes, cond)
        da = x[:, COLUMN["da"]]
        out = np.repeat(da[None, :], self.levels.size, axis=0)
        known = ~np.isnan(da)
        if self.boosters and known.any():
            spread = np.vstack([b.predict(x[known]) for b in self.boosters])
            bucket = np.digitize(x[known, COLUMN["lead_h"]], LEAD_EDGES_H)
            spread = np.sort(spread + self.offsets[bucket].T, axis=0)  # no crossing
            out[:, known] = da[known] + spread
        return out

    def scenarios(self, view, now, first, n):
        q = self.quantiles(view, now, first, n)
        u = (np.arange(self._n_scenarios) + 0.5) / self._n_scenarios
        j = np.clip(np.searchsorted(self.levels, u) - 1, 0, self.levels.size - 2)
        lo, hi = self.levels[j], self.levels[j + 1]
        w = ((u - lo) / (hi - lo))[:, None]  # outside [lo, hi] at the ends: extrapolation
        return q[j] + w * (q[j + 1] - q[j])

    def predict(self, view, now, first, n):
        return self.scenarios(view, now, first, n).mean(axis=0)
