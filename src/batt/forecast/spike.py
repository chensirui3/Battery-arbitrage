"""Next-hour spike nowcasting: will the RT price break far away from the DA price soon?

The value-of-information benchmarks say most of the revenue a realistic policy misses lies in
the next hour. This forecaster estimates, for each interval of the next hour, the probability
of an up-spike (RT more than ``threshold`` above DA) and of a down-spike (RT more than
``threshold`` below it). LightGBM classifiers are refit monthly on the two years of data public
at the time, as ``gbm`` is. ``size`` is the average spike size in the training window,
winsorized like ``gbm``'s labels. After the next hour the forecast is the DA price, as for
``mpc_oracle_1h``.

``adjust`` says how the probabilities move the DA price in the next hour:

- ``"raw"``: by the expected spike premium, ``p_up * size_up - p_down * size_down``.
- ``"news"``: by the same, with each probability minus a "prior": a second classifier that sees
  roughly what was known when the DA prices were set (``PRIOR_FEATURES``: the calendar, the DA
  prices, slow-moving RT history). The DA price is the market's own expectation, so this moves
  it only by what is new since.
- ``"regime"``: by the raw premium, but only on a side whose latest published RT price is
  itself a spike, i.e. during an ongoing spike. Otherwise the forecast is the DA price.

With ``perfect``, the forecaster is a benchmark that sees realized prices (oracle policies only):

- ``"spikes"``: the realized RT price where it is a spike, the DA price elsewhere.
- ``"timing"``: DA + ``size_up`` where an up-spike comes, DA - ``size_down`` where a down-spike
  comes, the DA price elsewhere: a perfect classifier with typical spike sizes.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Literal

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import rankdata

from batt.config import MarketTiming
from batt.data.store import DataView, MarketData, ts_ns
from batt.forecast.features import (
    CONDITION_FEATURES,
    FEATURES,
    FUNDAMENTAL_FEATURES,
    LOOKBACK_DAYS,
    build_features,
    conditions_of,
    fundamentals_of,
    gain_share,
)
from batt.forecast.history import DAY_NS, MINUTE_NS, PriceHistory
from batt.forecast.models import DEFAULT_GBM_PARAMS, Forecaster, Refit, RefitSchedule
from batt.sim.grid import SimulationGrid
from batt.timeutils import days, minutes, period_labels

Perfect = Literal["spikes", "timing"]
Adjust = Literal["raw", "news", "regime"]
SIDES = ("up", "down")

# Share of the published intervals in the last 24 hours and 30 days that were spikes.
SPIKE_FEATURES = ("up_24h", "down_24h", "up_30d", "down_30d")

# Roughly what was known when the DA prices were set: the calendar, the DA prices, and RT
# history that moves slowly (30-day averages, the same time of day over the past week).
PRIOR_FEATURES = (
    "tod_h",
    "dow",
    "doy_sin",
    "doy_cos",
    "da",
    "da_prev_hour",
    "da_next_hour",
    "da_day_mean",
    "da_day_max",
    "da_day_min",
    "da_pos",
    "spread_30d",
    "spread_tod_7d",
    "up_30d",
    "down_30d",
)


def spike_history(hist: PriceHistory, issue_ns, threshold: float) -> np.ndarray:
    """``SPIKE_FEATURES`` as of each issue time, from RT prices published by then."""
    issue = np.asarray(issue_ns, dtype=np.int64)
    hi = hist.latest_rt(issue) + 1  # prefix-sum index just past the latest published interval
    spread = hist.spread
    published = ~np.isnan(spread)

    def prefix(mask: np.ndarray) -> np.ndarray:
        return np.concatenate([[0], np.cumsum(mask)])

    count = prefix(published)
    spikes = {"up": prefix(published & (spread > threshold))}
    spikes["down"] = prefix(published & (spread < -threshold))
    out = np.full((issue.size, len(SPIKE_FEATURES)), np.nan)
    for w, window in enumerate((hist.per_day, 30 * hist.per_day)):
        lo = np.maximum(hi - window, 0)
        n = count[hi] - count[lo]
        for s, side in enumerate(SIDES):
            with np.errstate(invalid="ignore", divide="ignore"):
                share = (spikes[side][hi] - spikes[side][lo]) / n
            out[:, 2 * w + s] = np.where(n > 0, share, np.nan)
    return out


def nowcast_targets(
    hist: PriceHistory, lead_ns: int, first_issue_ns: int, end_issue_ns: int, steps: int
) -> tuple[np.ndarray, np.ndarray]:
    """(issue time, target index) pairs: every decision issued in [first, end), as in the
    backtest (interval start minus the decision lead), with each of the ``steps`` intervals
    from the one being decided."""
    decided = np.arange(len(hist), dtype=np.int64)
    issue = hist.start_ns(decided) - lead_ns
    keep = (issue >= first_issue_ns) & (issue < end_issue_ns)
    decided, issue = decided[keep], issue[keep]
    target = (decided[:, None] + np.arange(steps)[None, :]).ravel()
    issue = np.repeat(issue, steps)
    inside = target < len(hist)
    return issue[inside], target[inside]


class SpikeForecaster(Forecaster):
    """The DA price, moved for the next hour by a spike nowcast (see the module docstring)."""

    name = "spike"

    def __init__(
        self,
        threshold: float = 100.0,
        hours: float = 1.0,
        adjust: Adjust = "regime",
        refit: Refit = "month",
        train_days: int = 730,
        num_boost_round: int = 300,
        label_clip: float | None = 1000.0,
        min_positives: int = 200,
        seed: int = 0,
        params: dict[str, Any] | None = None,
        fundamentals: bool = False,
        perfect: Perfect | None = None,
        conditions: bool = False,
    ):
        super().__init__()
        if threshold <= 0 or hours <= 0:
            raise ValueError("threshold and hours must be positive")
        if perfect not in (None, "spikes", "timing"):
            raise ValueError(f"perfect must be 'spikes', 'timing', or None, not {perfect!r}")
        if adjust not in ("raw", "news", "regime"):
            raise ValueError(f"adjust must be 'raw', 'news', or 'regime', not {adjust!r}")
        self.threshold = float(threshold)
        self.hours = float(hours)
        self.adjust = adjust
        self.fit_prior = adjust == "news"  # evaluate_nowcasts also fits it, as a reference
        self.train_days = train_days
        self.num_boost_round = num_boost_round
        self.label_clip = label_clip
        self.min_positives = min_positives
        self.seed = seed
        self.params = {**DEFAULT_GBM_PARAMS, **(params or {}), "objective": "binary", "seed": seed}
        self.fundamentals = fundamentals
        self.conditions = conditions
        self.perfect = perfect
        self.schedule = RefitSchedule(refit)
        self._clear()

    @property
    def needs_oracle(self) -> bool:  # type: ignore[override]
        """Only the benchmark modes see realized prices."""
        return self.perfect is not None

    def _clear(self) -> None:
        self.models: dict[str, lgb.Booster | None] = {}
        self.priors: dict[str, lgb.Booster | None] = {}
        self.size = {side: 0.0 for side in SIDES}
        self.active = {side: False for side in SIDES}  # fitted at least once
        self.fit_times: list[pd.Timestamp] = []
        self.fit_log: list[dict[str, Any]] = []

    def reset(self, timing: MarketTiming) -> None:
        super().reset(timing)
        self.schedule.reset()
        self._clear()

    def steps(self, step_ns: int) -> int:
        """Intervals covered by the nowcast."""
        return max(1, round(self.hours * 60 * MINUTE_NS / step_ns))

    @property
    def feature_names(self) -> list[str]:
        extra = FUNDAMENTAL_FEATURES if self.fundamentals else ()
        grid = CONDITION_FEATURES if self.conditions else ()
        return [*FEATURES, *SPIKE_FEATURES, *extra, *grid]

    @property
    def prior_columns(self) -> list[int]:
        names = self.feature_names
        return [names.index(name) for name in PRIOR_FEATURES]

    def side_data(self, view: DataView) -> tuple:
        """The point-in-time fundamentals and grid conditions used (None where off)."""
        fund = fundamentals_of(view) if self.fundamentals else None
        cond = conditions_of(view) if self.conditions else None
        return fund, cond

    def features(self, hist: PriceHistory, issue_ns, target, fund=None, cond=None) -> np.ndarray:
        """Feature rows (``feature_names``) for (issue time, target index) pairs."""
        k = np.asarray(target, dtype=np.int64)
        issue = np.broadcast_to(np.asarray(issue_ns, dtype=np.int64), k.shape)
        base = build_features(hist, issue, k, fund, self.deadline_minutes, cond)
        n = len(FEATURES)
        spikes = spike_history(hist, issue, self.threshold)
        return np.hstack([base[:, :n], spikes, base[:, n:]])

    def training_set(self, view: DataView, now) -> tuple[np.ndarray, np.ndarray]:
        """Features and RT-DA spreads for every decision issued in the last ``train_days`` before
        ``now``, each paired with every interval of the hour it starts. Labels come from the
        view, so a view as of ``now`` only yields spreads published by then."""
        side = self.side_data(view)
        issue_now = ts_ns(now)
        start = issue_now - (self.train_days + LOOKBACK_DAYS) * DAY_NS
        hist = PriceHistory.from_view(view, start, issue_now + 2 * DAY_NS, self.timezone)
        first = start + LOOKBACK_DAYS * DAY_NS
        issue, target = nowcast_targets(
            hist, self.lead_ns, first, issue_now, self.steps(hist.step_ns)
        )
        spread = hist.spread[target]
        labeled = ~np.isnan(spread)
        x = self.features(hist, issue[labeled], target[labeled], *side)
        known = ~np.isnan(x[:, FEATURES.index("da")])  # DA price public at the issue time
        return x[known], spread[labeled][known]

    def prepare(self, view, now, first):
        if self.perfect == "spikes" or not self.schedule.due(first, self.timezone):
            return
        x, spread = self.training_set(view, now)
        clip = np.inf if self.label_clip is None else self.label_clip
        log: dict[str, Any] = {"time": pd.Timestamp(now), "examples": int(spread.size)}
        fitted = False
        for side in SIDES:
            y = spread > self.threshold if side == "up" else spread < -self.threshold
            log[f"{side}_spikes"] = int(y.sum())
            if y.sum() < self.min_positives:
                continue  # too few spikes to learn from; keep the previous fit (or none)
            self.size[side] = float(np.minimum(np.abs(spread[y]), clip).mean())
            if self.perfect is None:
                self.models[side] = self._fit(x, y, self.feature_names)
                if self.conditions:
                    share = gain_share(self.models[side], CONDITION_FEATURES)
                    log[f"{side}_condition_gain"] = share
                if self.fit_prior:
                    columns = self.prior_columns
                    self.priors[side] = self._fit(x[:, columns], y, list(PRIOR_FEATURES))
            self.active[side] = fitted = True
        log.update({f"size_{side}": self.size[side] for side in SIDES})
        self.fit_log.append(log)
        if fitted:
            self.fit_times.append(pd.Timestamp(now))

    def _fit(self, x: np.ndarray, y: np.ndarray, names: list[str]) -> lgb.Booster:
        data = lgb.Dataset(x, y.astype(float), feature_name=names)
        return lgb.train(self.params, data, num_boost_round=self.num_boost_round)

    def probabilities(self, x: np.ndarray) -> dict[str, np.ndarray]:
        """Spike probabilities for feature rows: ``p_up``, ``p_down``, and the priors
        ``prior_up``, ``prior_down`` (NaN where there is no fitted model)."""
        out = {}
        for side in SIDES:
            for key, models, columns in (
                (f"p_{side}", self.models, slice(None)),
                (f"prior_{side}", self.priors, self.prior_columns),
            ):
                model = models.get(side)
                if model is None or len(x) == 0:
                    out[key] = np.full(len(x), np.nan)
                else:
                    # (Passing num_threads here would make each call much slower.)
                    out[key] = model.predict(x[:, columns])
        return out

    def adjustment(self, x: np.ndarray, probs: dict[str, np.ndarray]) -> np.ndarray:
        """$/MWh added to the DA price for feature rows ``x`` with ``probabilities`` ``probs``."""
        out = np.zeros(len(x))
        latest = x[:, FEATURES.index("spread_last")]  # latest published RT-DA spread
        for side, sign in (("up", 1.0), ("down", -1.0)):
            if not self.active[side]:
                continue
            p = probs[f"p_{side}"]
            if self.adjust == "news":
                p = p - probs[f"prior_{side}"]
            elif self.adjust == "regime":
                p = np.where(np.nan_to_num(sign * latest) > self.threshold, p, 0.0)
            out += sign * p * self.size[side]
        return out

    def timing_adjustment(self, spread: np.ndarray) -> np.ndarray:
        """``perfect="timing"``: typical spike sizes where realized spreads are spikes."""
        out = np.zeros(len(spread))
        for side, sign in (("up", 1.0), ("down", -1.0)):
            if self.active[side]:
                hit = np.nan_to_num(sign * np.asarray(spread, dtype=float)) > self.threshold
                out += sign * hit * self.size[side]
        return out

    def _history(self, view: DataView, now, first, n: int) -> tuple[PriceHistory, np.ndarray]:
        """Live history for a forecast of ``n`` intervals from ``first``, and their indices.

        It reaches back ``LOOKBACK_DAYS`` and forward past the end of the operating day of every
        interval forecast, since the DA features summarize each target's whole day.
        """
        issue, first_ns = ts_ns(now), ts_ns(first)
        step = view.resolution("rt") * MINUTE_NS
        end = first_ns + max(n, self.steps(step)) * step + 2 * DAY_NS
        hist = PriceHistory.from_view(view, issue - LOOKBACK_DAYS * DAY_NS, end, self.timezone)
        return hist, hist.index(first_ns) + np.arange(n)

    def nowcast(self, view: DataView, now, first) -> dict[str, np.ndarray]:
        """Probabilities (``probabilities``) for each interval of the nowcast horizon from
        ``first``, using only data in ``view``."""
        m = self.steps(view.resolution("rt") * MINUTE_NS)
        hist, k = self._history(view, now, first, m)
        return self.probabilities(self.features(hist, ts_ns(now), k, *self.side_data(view)))

    def predict(self, view, now, first, n):
        issue = ts_ns(now)
        hist, k = self._history(view, now, first, n)
        out = hist.visible_da(k, issue)
        m = min(self.steps(hist.step_ns), n)
        near = k[:m]
        if self.perfect == "spikes":
            spike = np.abs(np.nan_to_num(hist.rt[near] - hist.da[near])) > self.threshold
            out[:m] = np.where(spike, hist.rt[near], out[:m])
            return out
        if m == 0 or not any(self.active.values()):
            return out
        if self.perfect == "timing":
            out[:m] = out[:m] + self.timing_adjustment(hist.rt[near] - hist.da[near])
            return out
        x = self.features(hist, issue, near, *self.side_data(view))
        out[:m] = out[:m] + self.adjustment(x, self.probabilities(x))
        return out


def evaluate_nowcasts(
    market: MarketData,
    forecaster: SpikeForecaster,
    start: date,
    end: date,
    timing: MarketTiming,
) -> pd.DataFrame:
    """Walk-forward spike probabilities for every decision in [start, end) and each interval of
    the hour it starts, with the realized RT-DA spread.

    The forecaster is refit on its own schedule from a view as of each refit's decision time,
    exactly as in the backtest. Between refits, features are computed in bulk from the full
    history. That is safe because every feature uses only values published by its issue time;
    ``tests/test_spike.py`` checks that bulk and live probabilities agree.
    """
    if forecaster.perfect is not None:
        raise ValueError("evaluate a realistic nowcaster, not a benchmark")
    forecaster.fit_prior = True  # the reference: what the DA prices already implied
    grid = SimulationGrid.for_days(start, end, timing.timezone, timing.interval_minutes)
    step_ns = grid.interval_minutes * MINUTE_NS
    lead = minutes(timing.decision_lead_minutes)
    forecaster.reset(timing)
    steps = forecaster.steps(step_ns)
    oracle = market.view(None)
    side = forecaster.side_data(oracle)
    hist = PriceHistory.from_view(
        oracle,
        grid.starts[0] - days(LOOKBACK_DAYS + 1),
        grid.starts[-1] + days(2),
        timing.timezone,
    )

    # Decisions where a refit is due start a new block (a fresh schedule, same rule).
    schedule = RefitSchedule(forecaster.schedule.every)
    firsts = [i for i, s in enumerate(grid.starts) if schedule.due(s, timing.timezone)]
    frames = []
    for a, b in zip(firsts, [*firsts[1:], len(grid)], strict=True):
        now = grid.starts[a] - lead
        forecaster.prepare(market.view(now), now, grid.starts[a])
        decided = hist.index(grid.starts[a]) + np.arange(b - a)
        issue = hist.start_ns(decided) - lead.value
        target = (decided[:, None] + np.arange(steps)[None, :]).ravel()
        issue = np.repeat(issue, steps)
        probs = forecaster.probabilities(forecaster.features(hist, issue, target, *side))
        frames.append(
            pd.DataFrame(
                {
                    "issue_utc": pd.to_datetime(issue, utc=True),
                    "target_utc": pd.to_datetime(hist.start_ns(target), utc=True),
                    "step": np.tile(np.arange(steps), b - a),
                    **probs,
                    "spread": hist.spread[target],
                }
            )
        )
    frame = pd.concat(frames, ignore_index=True)
    frame["year"] = period_labels(frame["target_utc"].dt.tz_convert(timing.timezone))
    return frame


def auc(score: np.ndarray, label: np.ndarray) -> float:
    """Area under the ROC curve: the chance a random spike scores above a random non-spike."""
    label = np.asarray(label, dtype=bool)
    pos, neg = int(label.sum()), int((~label).sum())
    if pos == 0 or neg == 0:
        return float("nan")
    ranks = rankdata(score)
    return float((ranks[label].sum() - pos * (pos + 1) / 2) / (pos * neg))


def average_precision(score: np.ndarray, label: np.ndarray) -> float:
    """Average precision (area under the precision-recall curve), ties broken by order."""
    label = np.asarray(label, dtype=bool)
    if not label.any():
        return float("nan")
    hits = label[np.argsort(-np.asarray(score), kind="stable")]
    precision = np.cumsum(hits) / np.arange(1, hits.size + 1)
    return float(precision[hits].mean())


def _log_loss(p: np.ndarray, label: np.ndarray) -> float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(np.where(label, np.log(p), np.log(1 - p))))


def nowcast_scores(frame: pd.DataFrame, threshold: float) -> dict[str, pd.DataFrame]:
    """Skill of the nowcast against the prior (what the DA prices already implied), for up- and
    down-spikes: overall, by step ahead, by year, plus a reliability table."""

    def scores(part: pd.DataFrame, side: str) -> dict[str, float]:
        p, prior = part[f"p_{side}"].to_numpy(), part[f"prior_{side}"].to_numpy()
        spread = part["spread"].to_numpy()
        label = spread > threshold if side == "up" else spread < -threshold
        brier, brier_prior = np.mean((p - label) ** 2), np.mean((prior - label) ** 2)
        return {
            "examples": len(part),
            "spike_rate": float(label.mean()),
            "mean_p": float(p.mean()),
            "auc": auc(p, label),
            "auc_prior": auc(prior, label),
            "avg_precision": average_precision(p, label),
            "avg_precision_prior": average_precision(prior, label),
            "brier": float(brier),
            "brier_prior": float(brier_prior),
            "brier_skill": float(1 - brier / brier_prior) if brier_prior > 0 else float("nan"),
            "log_loss": _log_loss(p, label),
            "log_loss_prior": _log_loss(prior, label),
        }

    out = {}
    for side in SIDES:
        scored = frame.dropna(subset=[f"p_{side}", f"prior_{side}", "spread"])
        if scored.empty:
            continue
        rows = {"all": scores(scored, side)}
        rows.update({f"step {s}": scores(g, side) for s, g in scored.groupby("step")})
        rows.update({str(y): scores(g, side) for y, g in scored.groupby("year")})
        out[side] = pd.DataFrame(rows).T
        p = scored[f"p_{side}"]
        spread = scored["spread"]
        label = spread > threshold if side == "up" else spread < -threshold
        edges = [0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0]
        bins = pd.cut(p, edges, include_lowest=True)
        out[f"{side}_reliability"] = pd.DataFrame(
            {
                "examples": p.groupby(bins, observed=True).size(),
                "mean_p": p.groupby(bins, observed=True).mean(),
                "spike_rate": label.groupby(bins, observed=True).mean(),
            }
        )
    return out
