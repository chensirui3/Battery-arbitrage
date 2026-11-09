"""Point-in-time features for forecasting real-time prices.

An example is a pair (issue time, target interval): "at this moment, what will the RT price of
that interval be?". Every feature uses only values published at or before the issue time, so
one function builds both training sets (many issue times at once, from a long history) and
live forecasts (one issue time, from a recent window). ``tests/test_forecast.py`` checks this
by recomputing features from data truncated at each issue time.
"""

from __future__ import annotations

import numpy as np

from batt.data.store import DataView, ts_ns
from batt.forecast.history import (
    DAY_NS,
    HOUR_NS,
    MINUTE_NS,
    PriceHistory,
    local_naive_ns,
    local_to_utc_ns,
)

# The longest feature window is 30 days; live histories reach back this far.
LOOKBACK_DAYS = 32

FEATURES = (
    # When the target is, and how far ahead
    "lead_h",
    "tod_h",
    "dow",
    "doy_sin",
    "doy_cos",
    # Day-ahead prices for the target's hour and its operating day (published the day before)
    "da",
    "da_prev_hour",
    "da_next_hour",
    "da_day_mean",
    "da_day_max",
    "da_day_min",
    "da_pos",
    # Real-time prices published by the issue time (the latest is ~25 minutes old)
    "rt_last",
    "spread_last",
    "spread_1h",
    "spread_4h",
    "spread_24h",
    "spread_std_24h",
    "rt_max_1h",
    "spread_30d",
    "spread_tod_7d",
)
COLUMN = {name: j for j, name in enumerate(FEATURES)}

# From ERCOT's load, wind, and solar forecasts (batt.data.fundamentals), all system-wide MW
# for the target's hour. "_rev" is how much the latest forecast moved since the DA bid
# deadline (10:00 the day before), i.e. information the DA prices could not reflect; "_err"
# is how far the latest published actual came in from its own DA-deadline forecast.
FUNDAMENTAL_FEATURES = (
    "load_fc",
    "wind_fc",
    "solar_fc",
    "netload_fc",
    "load_rev",
    "wind_rev",
    "solar_rev",
    "netload_rev",
    "wind_err",
    "solar_err",
)


# From ERCOT's real-time grid conditions (batt.data.conditions; docs/grid-conditions-plan.md).
# "sced_spread" is the latest 5-minute system price, about 5 minutes old at a decision, minus
# the DA price of its interval. "prc" is real-time reserves (MW; missing after RTC+B) and
# "prc_change" its change over the last 15 minutes. "rtd_spread" is ERCOT's look-ahead price
# for the target interval minus its DA price (missing beyond the look-ahead). "outage_mw" is
# the MW of generation out in the target's hour, and "outage_rev" its change since the DA bid
# deadline.
CONDITION_FEATURES = (
    "sced_spread",
    "prc",
    "prc_change",
    "rtd_spread",
    "outage_mw",
    "outage_rev",
)


def gain_share(booster, names) -> float:
    """Share of a fitted LightGBM model's total split gain that goes to the features ``names``."""
    gain = dict(zip(booster.feature_name(), booster.feature_importance("gain"), strict=True))
    total = float(sum(gain.values()))
    return sum(float(gain.get(name, 0.0)) for name in names) / total if total > 0 else np.nan


def bid_deadline_ns(target_ns, timezone: str, minutes_after_midnight: int = 600) -> np.ndarray:
    """DA bid deadline (local time, the day before) for each target's operating day."""
    day = local_naive_ns(target_ns, timezone) // DAY_NS
    local = (day - 1) * DAY_NS + minutes_after_midnight * (HOUR_NS // 60)
    return local_to_utc_ns(local, timezone)


def fundamental_features(
    fund, issue_ns, target_ns, timezone: str, deadline_minutes: int = 600
) -> np.ndarray:
    """``FUNDAMENTAL_FEATURES`` for (issue time, target start) pairs, from a point-in-time
    ``FundamentalsView``: only forecasts and actuals published by each issue time are used."""
    target = np.asarray(target_ns, dtype=np.int64)
    issue = np.broadcast_to(np.asarray(issue_ns, dtype=np.int64), target.shape)
    deadline = np.minimum(bid_deadline_ns(target, timezone, deadline_minutes), issue)
    latest = {s: fund.forecast(f"{s}_forecast", issue, target) for s in ("load", "wind", "solar")}
    at_bids = {s: fund.forecast(f"{s}_forecast", deadline, target) for s in latest}
    net_latest = latest["load"] - latest["wind"] - latest["solar"]
    net_at_bids = at_bids["load"] - at_bids["wind"] - at_bids["solar"]
    columns = [
        latest["load"],
        latest["wind"],
        latest["solar"],
        net_latest,
        *(latest[s] - at_bids[s] for s in ("load", "wind", "solar")),
        net_latest - net_at_bids,
    ]
    for kind in ("wind", "solar"):
        hour, actual = fund.latest_actual(f"{kind}_actual", issue)
        when = np.minimum(bid_deadline_ns(hour, timezone, deadline_minutes), issue)
        columns.append(actual - fund.forecast(f"{kind}_forecast", when, hour))
    return np.column_stack(columns)


def condition_features(
    cond, hist: PriceHistory, issue_ns, target, deadline_minutes: int = 600
) -> np.ndarray:
    """``CONDITION_FEATURES`` for (issue time, target index) pairs, from a point-in-time
    ``ConditionsView``: only runs and publications public by each issue time are used."""
    k = np.asarray(target, dtype=np.int64)
    issue = np.broadcast_to(np.asarray(issue_ns, dtype=np.int64), k.shape)
    target_ns = hist.start_ns(k)
    when, price, prc = cond.sced(issue)
    ran_in = np.where(when >= 0, (when - hist.origin_ns) // hist.step_ns, -1)
    inside = (ran_in >= 0) & (ran_in < len(hist))
    da_then = np.full(k.shape, np.nan)
    if inside.any():
        da_then[inside] = hist.visible_da(ran_in[inside], issue[inside])
    _, _, prc_before = cond.sced(issue - 15 * MINUTE_NS)
    rtd = cond.rtd_price(issue, target_ns, target_ns + hist.step_ns)
    deadline = np.minimum(bid_deadline_ns(target_ns, hist.timezone, deadline_minutes), issue)
    outage = cond.outage(issue, target_ns)
    return np.column_stack(
        [
            price - da_then,
            prc,
            prc - prc_before,
            rtd - hist.visible_da(k, issue),
            outage,
            outage - cond.outage(deadline, target_ns),
        ]
    )


def build_features(
    hist: PriceHistory, issue_ns, target, fund=None, deadline_minutes: int = 600, cond=None
) -> np.ndarray:
    """Feature matrix (one row per example) for issue times and target grid indices.

    "spread" is the RT price minus the DA price of the same interval. Missing values (nothing
    published yet, or gaps) are NaN, which gradient-boosted trees handle natively. With a
    point-in-time ``fund`` (``DataView.fundamentals``), ``FUNDAMENTAL_FEATURES`` follow; with
    ``cond`` (``DataView.conditions``), ``CONDITION_FEATURES`` come last.
    """
    base = _price_features(hist, issue_ns, target)
    if fund is None and cond is None:
        return base
    k = np.asarray(target, dtype=np.int64)
    issue = np.broadcast_to(np.asarray(issue_ns, dtype=np.int64), k.shape)
    parts = [base]
    if fund is not None:
        parts.append(
            fundamental_features(fund, issue, hist.start_ns(k), hist.timezone, deadline_minutes)
        )
    if cond is not None:
        parts.append(condition_features(cond, hist, issue, k, deadline_minutes))
    return np.hstack(parts)


def _price_features(hist: PriceHistory, issue_ns, target) -> np.ndarray:
    k = np.asarray(target, dtype=np.int64)
    issue = np.broadcast_to(np.asarray(issue_ns, dtype=np.int64), k.shape)
    out = np.full((k.size, len(FEATURES)), np.nan)

    def put(name: str, values) -> None:
        out[:, COLUMN[name]] = values

    target_ns = hist.start_ns(k)
    local = local_naive_ns(target_ns, hist.timezone)
    day = local // DAY_NS
    put("lead_h", (target_ns - issue) / HOUR_NS)
    put("tod_h", (local - day * DAY_NS) / HOUR_NS)
    put("dow", (day + 3) % 7)  # 1970-01-01 was a Thursday; Monday = 0
    angle = 2 * np.pi * _day_of_year(local) / 365.25
    put("doy_sin", np.sin(angle))
    put("doy_cos", np.cos(angle))

    da = hist.visible_da(k, issue)
    known = ~np.isnan(da)
    put("da", da)
    put("da_prev_hour", da - hist.visible_da(k - hist.per_hour, issue))
    put("da_next_hour", hist.visible_da(k + hist.per_hour, issue) - da)
    mean, high, low = hist.da_day_stats(k)
    put("da_day_mean", np.where(known, mean, np.nan))
    put("da_day_max", np.where(known, high, np.nan))
    put("da_day_min", np.where(known, low, np.nan))
    width = high - low
    with np.errstate(invalid="ignore", divide="ignore"):
        position = np.where(width > 0, (da - low) / width, 0.5)
    put("da_pos", np.where(known, position, np.nan))

    last = hist.latest_rt(issue)
    put("rt_last", _take(hist.rt, last))
    put("spread_last", _take(hist.spread, last))
    put("spread_1h", hist.trailing_spread(last, hist.per_hour)[0])
    put("spread_4h", hist.trailing_spread(last, 4 * hist.per_hour)[0])
    mean_24h, std_24h = hist.trailing_spread(last, hist.per_day)
    put("spread_24h", mean_24h)
    put("spread_std_24h", std_24h)
    put("rt_max_1h", _trailing_max(hist.rt, last, hist.per_hour))
    put("spread_30d", hist.trailing_spread(last, 30 * hist.per_day)[0])
    put("spread_tod_7d", _same_time_mean(hist.spread, k, last, hist.per_day, 7))
    return out


def training_set(
    view: DataView,
    now,
    train_days: int,
    lead_ns: int,
    timezone: str,
    seed: int = 0,
    fundamentals: bool = False,
    deadline_minutes: int = 600,
    conditions: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Examples issued in the last ``train_days`` before ``now``, labeled with what ``view``
    shows (so a view as of ``now`` only yields labels published by then).

    Returns features, labels, and each example's issue time (ns).
    """
    fund = fundamentals_of(view) if fundamentals else None
    cond = conditions_of(view) if conditions else None
    issue = ts_ns(now)
    start = issue - (train_days + LOOKBACK_DAYS) * DAY_NS
    hist = PriceHistory.from_view(view, start, issue + 2 * DAY_NS, timezone)
    first = start + LOOKBACK_DAYS * DAY_NS
    return training_examples(
        hist,
        lead_ns,
        first,
        issue,
        seed,
        True,
        fund=fund,
        deadline_minutes=deadline_minutes,
        cond=cond,
    )


def fundamentals_of(view: DataView):
    """The view's fundamentals, or a clear error when none were loaded."""
    fund = view.fundamentals
    if fund is None:
        raise ValueError(
            "this forecaster uses ERCOT fundamentals, but none are loaded: run "
            "`batt ingest-fundamentals` (see the README) or turn `fundamentals` off"
        )
    return fund


def conditions_of(view: DataView):
    """The view's grid conditions, or a clear error when none were loaded."""
    cond = view.conditions
    if cond is None:
        raise ValueError(
            "this forecaster uses ERCOT grid conditions, but none are loaded: run "
            "`batt ingest-conditions` (see the README) or turn `conditions` off"
        )
    return cond


def training_examples(
    hist: PriceHistory,
    lead_ns: int,
    first_issue_ns: int,
    end_issue_ns: int,
    seed: int = 0,
    return_issue: bool = False,
    fund=None,
    deadline_minutes: int = 600,
    cond=None,
):
    """Features and labels (RT minus DA price) for decisions issued in [first, end).

    Issue times are the decision times of the grid's intervals (interval start minus the
    decision lead), as in the backtest. Each one gets two targets among the intervals whose
    DA prices are already out: one drawn uniformly and one log-uniformly, so short horizons,
    where recent RT prices matter most, are well covered. Labels come from ``hist`` itself,
    so a history built as of the training time only yields labels published by then.
    """
    decided = np.arange(len(hist), dtype=np.int64)
    issue = hist.start_ns(decided) - lead_ns
    keep = (issue >= first_issue_ns) & (issue < end_issue_ns)
    decided, issue = decided[keep], issue[keep]
    span = hist.latest_da(issue) + 1 - decided  # targets with published DA prices
    keep = span > 0
    decided, issue, span = decided[keep], issue[keep], span[keep]

    rng = np.random.default_rng(seed)
    u = rng.random((2, decided.size))
    uniform = np.floor(u[0] * span).astype(np.int64)
    log_uniform = np.floor(np.exp(u[1] * np.log(span + 1.0))).astype(np.int64) - 1
    horizon = np.concatenate([uniform, np.clip(log_uniform, 0, span - 1)])
    target = np.concatenate([decided, decided]) + horizon
    issue = np.concatenate([issue, issue])

    label = hist.rt[target] - hist.da[target]
    keep = ~np.isnan(label)
    x = build_features(hist, issue[keep], target[keep], fund, deadline_minutes, cond)
    if return_issue:
        return x, label[keep], issue[keep]
    return x, label[keep]


def _day_of_year(local_ns: np.ndarray) -> np.ndarray:
    d = local_ns.astype("datetime64[ns]").astype("datetime64[D]")
    year_start = d.astype("datetime64[Y]").astype("datetime64[D]")
    return (d - year_start).astype(np.int64) + 1


def _take(values: np.ndarray, idx: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return np.full(idx.shape, np.nan)
    return np.where(idx >= 0, values[np.maximum(idx, 0)], np.nan)


def _trailing_max(values: np.ndarray, last: np.ndarray, window: int) -> np.ndarray:
    if values.size == 0:
        return np.full(last.shape, np.nan)
    idx = last[:, None] - np.arange(window)[None, :]
    v = np.where(idx >= 0, values[np.maximum(idx, 0)], np.nan)
    m = np.where(np.isnan(v), -np.inf, v).max(axis=1)
    return np.where(np.isfinite(m), m, np.nan)


def _same_time_mean(spread, k, last, per_day: int, n_days: int) -> np.ndarray:
    """Mean spread at the target's time of day over the previous days, where already public."""
    if spread.size == 0:
        return np.full(k.shape, np.nan)
    idx = k[:, None] - per_day * np.arange(1, n_days + 1)[None, :]
    ok = (idx >= 0) & (idx <= last[:, None])
    v = np.where(ok, spread[np.clip(idx, 0, spread.size - 1)], np.nan)
    ok &= ~np.isnan(v)
    count = ok.sum(axis=1)
    total = np.where(ok, v, 0.0).sum(axis=1)
    return np.where(count > 0, total / np.maximum(count, 1), np.nan)
