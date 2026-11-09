"""ERCOT's own real-time look-ahead (RTD) used directly as the next hour's price forecast.

Every 5 minutes, ERCOT's real-time dispatch projects the system-wide energy price for the next
eleven 5-minute intervals (``batt.data.conditions``). It knows the current load, wind, and
solar ramps, the units online, and the reserves, which no model trained on prices sees. At a
decision, the latest public run covers the next three 15-minute intervals.

The comparison with ``mpc_da`` is fixed in ``docs/grid-conditions-plan.md``.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from batt.data.store import DataView, ts_ns
from batt.forecast.features import conditions_of
from batt.forecast.history import DAY_NS, MINUTE_NS, PriceHistory
from batt.forecast.models import Forecaster

Mode = Literal["all", "spikes"]
Basis = Literal["none", "latest"]


class LookAheadForecaster(Forecaster):
    """RT prices from ERCOT's look-ahead where its latest run covers an interval, DA elsewhere.

    For each of the next ``hours``' intervals fully covered by the latest RTD run public at the
    issue time, the forecast is that run's price averaged over the interval. Two settings, both
    chosen before the backtest period (see the plan):

    - ``mode="all"`` uses RTD's price wherever it covers; ``"spikes"`` only where it is more
      than ``threshold`` $/MWh from the DA price.
    - RTD's price is system-wide. ``basis="latest"`` adds the latest published hub price minus
      the SCED price from that same interval; ``"none"`` adds nothing.

    The defaults (``spikes``, ``none``) earned the most of the four in July-December 2021,
    though all four earned less than planning on DA prices. There is nothing to fit.
    """

    name = "rtd"

    def __init__(
        self,
        mode: Mode = "spikes",
        basis: Basis = "none",
        threshold: float = 100.0,
        hours: float = 1.0,
    ):
        super().__init__()
        if mode not in ("all", "spikes"):
            raise ValueError(f"mode must be 'all' or 'spikes', not {mode!r}")
        if basis not in ("none", "latest"):
            raise ValueError(f"basis must be 'none' or 'latest', not {basis!r}")
        if threshold <= 0 or hours <= 0:
            raise ValueError("threshold and hours must be positive")
        self.mode = mode
        self.basis = basis
        self.threshold = float(threshold)
        self.hours = float(hours)
        self.conditions = True  # always uses grid conditions (see registry.uses_conditions)

    def predict(self, view: DataView, now, first, n):
        issue, first_ns = ts_ns(now), ts_ns(first)
        step = view.resolution("rt") * MINUTE_NS
        hist = PriceHistory.from_view(view, issue - DAY_NS, first_ns + n * step, self.timezone)
        k = hist.index(first_ns) + np.arange(n)
        out = hist.visible_da(k, issue)
        m = min(n, max(1, round(self.hours * 60 * MINUTE_NS / step)))
        cond = conditions_of(view)
        start = hist.start_ns(k[:m])
        rtd = cond.rtd_price(np.full(m, issue), start, start + step)
        if self.basis == "latest":
            rtd = rtd + hub_basis(cond, hist, issue)
        use = ~np.isnan(rtd) & ~np.isnan(out[:m])
        if self.mode == "spikes":
            use &= np.abs(rtd - out[:m]) > self.threshold
        out[:m] = np.where(use, rtd, out[:m])
        return out


def hub_basis(cond, hist: PriceHistory, issue_ns: int) -> float:
    """The latest published hub price minus the price of a SCED run inside that same 15-minute
    interval: how far the hub sat from the system-wide price most recently (0 if unknown)."""
    last = int(hist.latest_rt(np.array([issue_ns]))[0])
    if last < 0 or np.isnan(hist.rt[last]):
        return 0.0
    start = int(hist.start_ns(last))
    end = start + hist.step_ns
    when, price, _ = cond.sced(np.array([min(issue_ns, end)]))
    if not start <= when[0] < end or np.isnan(price[0]):
        return 0.0
    return float(hist.rt[last] - price[0])
