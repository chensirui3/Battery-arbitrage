"""Real-time price forecasters.

A forecaster answers: "it is ``now``; what will the RT price be in each of the next ``n``
intervals?" It sees market data only through a point-in-time view. Forecasters that learn
refit on a schedule (e.g. monthly) from the same view, so training only ever uses data that
was public at the time, in the backtest and in ``batt forecast`` alike.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar, Literal

import lightgbm as lgb
import numpy as np
import pandas as pd

from batt.config import MarketTiming
from batt.data.store import DataView, ts_ns
from batt.forecast.features import (
    COLUMN,
    CONDITION_FEATURES,
    FEATURES,
    FUNDAMENTAL_FEATURES,
    LOOKBACK_DAYS,
    build_features,
    conditions_of,
    fundamentals_of,
    gain_share,
    training_set,
)
from batt.forecast.history import DAY_NS, MINUTE_NS, PriceHistory

Refit = Literal["day", "week", "month", "once"]


class RefitSchedule:
    """Says when a model is due for refitting: on the first decision of each period."""

    def __init__(self, every: Refit = "month"):
        if every not in ("day", "week", "month", "once"):
            raise ValueError(f"unknown refit period {every!r}")
        self.every = every
        self._key: object = None

    def reset(self) -> None:
        self._key = None

    def due(self, first: pd.Timestamp, timezone: str) -> bool:
        local = pd.Timestamp(first).tz_convert(timezone)
        if self.every == "day":
            key: object = local.date()
        elif self.every == "week":
            key = local.isocalendar()[:2]
        elif self.every == "month":
            key = (local.year, local.month)
        else:
            key = "once"
        if key == self._key:
            return False
        self._key = key
        return True


class Forecaster(ABC):
    """Forecasts RT prices for the next intervals from data published by the issue time."""

    name: ClassVar[str] = "forecaster"
    needs_oracle: ClassVar[bool] = False  # benchmarks only: sees realized prices

    def __init__(self) -> None:
        self.timezone = "UTC"
        self.lead_ns = 0
        self.deadline_minutes = 600

    def reset(self, timing: MarketTiming) -> None:
        """Forget fitted state before a new run."""
        self.timezone = timing.timezone
        self.lead_ns = int(timing.decision_lead_minutes * MINUTE_NS)
        deadline = timing.da_bid_deadline_local_time
        self.deadline_minutes = 60 * deadline.hour + deadline.minute

    def prepare(self, view: DataView, now: pd.Timestamp, first: pd.Timestamp) -> None:  # noqa: B027
        """Refit if due. ``view`` shows data published by ``now``; ``first`` is being decided."""

    @abstractmethod
    def predict(self, view: DataView, now: pd.Timestamp, first: pd.Timestamp, n: int) -> np.ndarray:
        """RT price ($/MWh) for the ``n`` intervals starting at ``first``; NaN where unknown."""

    @property
    def n_scenarios(self) -> int:
        return 1

    def scenarios(
        self, view: DataView, now: pd.Timestamp, first: pd.Timestamp, n: int
    ) -> np.ndarray:
        """Equally likely RT price paths, shape ``(n_scenarios, n)``.

        A point forecaster gives one path, its forecast; scenario forecasters override this.
        """
        return self.predict(view, now, first, n)[None, :]


class DayAheadForecaster(Forecaster):
    """RT price = the published DA price of the same hour."""

    name = "da"

    def predict(self, view, now, first, n):
        first_ns = ts_ns(first)
        end_ns = first_ns + n * view.resolution("rt") * MINUTE_NS
        hist = PriceHistory.from_view(view, first_ns, end_ns, self.timezone)
        return hist.visible_da(np.arange(n), ts_ns(now))


class OracleForecaster(Forecaster):
    """The realized RT prices: perfect forecasts, for benchmarks only.

    With ``hours``, only the next ``hours`` are perfect and the published DA prices stand in
    beyond: what perfect short-term forecasts alone would be worth.
    """

    name = "oracle"
    needs_oracle = True

    def __init__(self, hours: float | None = None):
        super().__init__()
        self.hours = hours

    def predict(self, view, now, first, n):
        first_ns = ts_ns(first)
        step = view.resolution("rt") * MINUTE_NS
        hist = PriceHistory.from_view(view, first_ns, first_ns + n * step, self.timezone)
        if self.hours is None:
            return hist.rt.copy()
        known = np.arange(n) * step < self.hours * 60 * MINUTE_NS
        return np.where(known, hist.rt, hist.visible_da(np.arange(n), ts_ns(now)))


class SpreadAR1Forecaster(Forecaster):
    """DA price plus the latest RT-DA spread, decaying toward its mean at a fitted AR(1) rate.

    The spread (RT minus DA) is persistent: scarcity, congestion, or a forecast miss that
    pushes RT above DA tends to last for a while. This is the classic time-series baseline.
    """

    name = "ar1"

    def __init__(self, fit_days: int = 90, refit: Refit = "month", clip: float = 1000.0):
        super().__init__()
        self.fit_days = fit_days
        self.clip = clip
        self.schedule = RefitSchedule(refit)
        self.mean = 0.0
        self.rho = 0.0

    def reset(self, timing):
        super().reset(timing)
        self.schedule.reset()
        self.mean = 0.0
        self.rho = 0.0

    def prepare(self, view, now, first):
        if not self.schedule.due(first, self.timezone):
            return
        issue = ts_ns(now)
        hist = PriceHistory.from_view(view, issue - self.fit_days * DAY_NS, issue, self.timezone)
        s = np.clip(hist.spread, -self.clip, self.clip)
        prev, cur = s[:-1], s[1:]
        ok = ~np.isnan(prev) & ~np.isnan(cur)
        if ok.sum() < hist.per_day:
            return  # not enough history yet; keep the previous fit
        mean = float(np.mean(cur[ok]))
        x, y = prev[ok] - mean, cur[ok] - mean
        denominator = float(np.dot(x, x))
        self.mean = mean
        self.rho = float(np.clip(np.dot(x, y) / denominator, 0.0, 0.999)) if denominator else 0.0

    def predict(self, view, now, first, n):
        issue = ts_ns(now)
        step = view.resolution("rt") * MINUTE_NS
        first_ns = ts_ns(first)
        hist = PriceHistory.from_view(view, issue - DAY_NS, first_ns + n * step, self.timezone)
        k = hist.index(first_ns) + np.arange(n)
        da = hist.visible_da(k, issue)
        last = int(hist.latest_rt(issue))
        spread = hist.spread[last] if last >= 0 else np.nan
        if np.isnan(spread):
            return da + self.mean
        spread = float(np.clip(spread, -self.clip, self.clip))
        decay = self.rho ** (k - last).astype(float)
        return da + self.mean + decay * (spread - self.mean)


DEFAULT_GBM_PARAMS: dict[str, Any] = {
    "objective": "regression",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_data_in_leaf": 200,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 10.0,
    "deterministic": True,
    "force_row_wise": True,
    "verbose": -1,
}


def feature_names(fundamentals: bool = False, conditions: bool = False) -> list[str]:
    """Names of ``build_features`` columns: prices, then fundamentals, then conditions."""
    names = list(FEATURES)
    if fundamentals:
        names += FUNDAMENTAL_FEATURES
    if conditions:
        names += CONDITION_FEATURES
    return names


class GBMForecaster(Forecaster):
    """LightGBM model of the RT-DA spread, from the features in ``batt.forecast.features``.

    Refit on a schedule from the last ``train_days`` of data visible at the time, with a
    squared-error loss (the battery's revenue is linear in price, so it needs the conditional
    mean). Labels are winsorized at +/- ``label_clip`` $/MWh so a few extreme intervals can't
    dominate the fit. That has a cost: it biases forecasts for spike-prone afternoon and
    evening hours low. Without it (``label_clip=None``) the fit is noisier, and in the
    2022-2026 backtest both its forecasts and its dispatch were worse (see the README).
    Until the first fit, it forecasts DA prices. With ``fundamentals``, ERCOT's load, wind,
    and solar forecasts are added to the features (``FUNDAMENTAL_FEATURES``); with
    ``conditions``, ERCOT's real-time grid conditions (``CONDITION_FEATURES``). With ``hours``,
    the model is used only for the next ``hours`` and the DA prices stand in beyond, the way
    ``mpc_oracle_1h`` uses realized prices.
    """

    name = "gbm"

    def __init__(
        self,
        refit: Refit = "month",
        train_days: int = 730,
        num_boost_round: int = 300,
        label_clip: float | None = 1000.0,
        seed: int = 0,
        params: dict[str, Any] | None = None,
        fundamentals: bool = False,
        hours: float | None = None,
        conditions: bool = False,
    ):
        super().__init__()
        if hours is not None and hours <= 0:
            raise ValueError("hours must be positive")
        self.hours = hours
        self.fundamentals = fundamentals
        self.conditions = conditions
        self.train_days = train_days
        self.num_boost_round = num_boost_round
        self.label_clip = label_clip
        self.seed = seed
        self.params = {**DEFAULT_GBM_PARAMS, **(params or {}), "seed": seed}
        self.schedule = RefitSchedule(refit)
        self.booster: lgb.Booster | None = None
        self.fit_times: list[pd.Timestamp] = []
        self.fit_log: list[dict[str, Any]] = []  # one row per fit

    def reset(self, timing):
        super().reset(timing)
        self.schedule.reset()
        self.booster = None
        self.fit_times = []
        self.fit_log = []

    def training_set(self, view: DataView, now) -> tuple[np.ndarray, np.ndarray]:
        """Examples issued in the last ``train_days`` before ``now``, labeled by then."""
        x, y, _ = training_set(
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
        return x, y

    @property
    def feature_names(self) -> list[str]:
        return feature_names(self.fundamentals, self.conditions)

    def prepare(self, view, now, first):
        if not self.schedule.due(first, self.timezone):
            return
        x, y = self.training_set(view, now)
        if y.size < 1000:
            return  # too little history; keep the previous model (or DA)
        if self.label_clip is not None:
            y = np.clip(y, -self.label_clip, self.label_clip)
        data = lgb.Dataset(x, y, feature_name=self.feature_names)
        self.booster = lgb.train(self.params, data, num_boost_round=self.num_boost_round)
        self.fit_times.append(pd.Timestamp(now))
        log: dict[str, Any] = {"time": pd.Timestamp(now), "examples": int(y.size)}
        if self.conditions:
            log["condition_gain"] = gain_share(self.booster, CONDITION_FEATURES)
        self.fit_log.append(log)

    def predict(self, view, now, first, n):
        issue = ts_ns(now)
        step = view.resolution("rt") * MINUTE_NS
        first_ns = ts_ns(first)
        start = issue - LOOKBACK_DAYS * DAY_NS
        hist = PriceHistory.from_view(view, start, first_ns + n * step, self.timezone)
        fund = fundamentals_of(view) if self.fundamentals else None
        cond = conditions_of(view) if self.conditions else None
        k = hist.index(first_ns) + np.arange(n)
        x = build_features(hist, issue, k, fund, self.deadline_minutes, cond)
        da = x[:, COLUMN["da"]]
        if self.booster is None:
            return da
        out = np.full(n, np.nan)
        known = ~np.isnan(da)
        if known.any():
            # (Passing num_threads here would make each call ~5x slower.)
            out[known] = da[known] + self.booster.predict(x[known])
        if self.hours is not None:
            beyond = np.arange(n) * step >= self.hours * 60 * MINUTE_NS
            out[beyond] = da[beyond]
        return out
