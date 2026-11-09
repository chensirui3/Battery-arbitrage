r"""Stochastic dynamic programming: a dispatch *policy* that plans for what it will learn.

Scenario-based MPC either plans as if the future were known (each scenario's plan "times" its
own spikes) or averages scenarios away. Dynamic programming gets the information structure
right. The state is the stored energy and the spread state of the latest published RT price;
RT prices follow a Markov chain (``batt.forecast.markov``) on top of the published DA prices.
Backward induction over the horizon gives, for every interval, stored energy, and observed
spread state, the value of each action, so the battery only reacts to a spike after it would
actually see it (two intervals later) and only for as long as spikes tend to last.

With decisions for interval t made knowing the spread state z of interval t - 2:

    V_t(e, z) = max_a  (DA_t + E[x_t | z]) g_a - k max(g_a, 0) + E[ V_(t+1)(e'(e, a), z') | z ]
    V_T(e, .) = -penalty |e - e_target|

where g_a is the energy sold (negative: bought) and e' the stored energy after the action.
Stored energy is a grid (``soc_levels``) with linear interpolation; actions are evenly spaced
power levels clipped at full and empty. The tables are recomputed once a day, when the next
day's DA prices are published (and when the model is refit); each decision is a lookup.
"""

from __future__ import annotations

from datetime import date

import numpy as np

from batt.config import BatterySpec, MarketTiming
from batt.data.store import ts_ns
from batt.forecast.history import (
    DAY_NS,
    HOUR_NS,
    MINUTE_NS,
    PriceHistory,
    local_naive_ns,
    published_horizon,
)
from batt.forecast.markov import SpreadMarkovChain
from batt.forecast.models import Refit, RefitSchedule
from batt.optimize.lp import Exclusive
from batt.policies.base import Policy
from batt.sim.context import DecisionContext
from batt.sim.grid import SimulationGrid

TERMINAL_PENALTY = 1000.0  # $/MWh away from the target SoC at the end of the horizon


class StochasticDP(Policy):
    """Stochastic dynamic programming over stored energy and the observed spread state."""

    name = "sdp"

    def __init__(
        self,
        soc_levels: int = 41,
        action_levels: int = 17,
        fit_days: int = 90,
        refit: Refit = "month",
        exclusive: Exclusive = "auto",  # actions are single-direction by construction
    ):
        if soc_levels < 3 or action_levels < 3:
            raise ValueError("soc_levels and action_levels must be at least 3")
        self.soc_levels = soc_levels
        self.action_levels = action_levels
        self.fit_days = fit_days
        self.schedule = RefitSchedule(refit)
        self.chain: SpreadMarkovChain | None = None
        self.days_without_forecast: list[date] = []
        self.solves = 0

    def reset(self, grid: SimulationGrid, spec: BatterySpec, timing: MarketTiming) -> None:
        if spec.max_cycles_per_day is not None:
            raise ValueError("stochastic DP does not model a daily cycle cap (max_cycles_per_day)")
        self.spec = spec
        self.timezone = timing.timezone
        self.dt = grid.dt_hours
        self.step_ns = grid.interval_minutes * MINUTE_NS
        self.schedule.reset()
        self.chain = None
        self.days_without_forecast = []
        self.solves = 0
        self._tables_key: tuple | None = None
        self._start_ns = 0
        # Power levels from full charge to full discharge, idle first so ties stay idle.
        levels = np.linspace(-1.0, 1.0, self.action_levels)
        self._power = spec.power_mw * levels[np.argsort(np.abs(levels), kind="stable")]
        self._grid_e = np.linspace(spec.soc_min_mwh, spec.soc_max_mwh, self.soc_levels)
        sold, after = self._outcomes(self._grid_e[:, None])
        self._sold, self._wear = sold, spec.degradation_cost_per_mwh * np.maximum(sold, 0.0)
        self._idx, self._w = self._locate(after)

    def _outcomes(self, energy):
        """Energy sold (MWh, negative when buying) and stored energy after each action."""
        spec = self.spec
        grid_energy = self._power * self.dt
        stored = energy - np.where(
            grid_energy > 0, grid_energy / spec.eta_discharge, grid_energy * spec.eta_charge
        )
        after = np.clip(stored, spec.soc_min_mwh, spec.soc_max_mwh)
        change = energy - after  # stored energy removed (negative when added)
        sold = np.where(change > 0, change * spec.eta_discharge, change / spec.eta_charge)
        return sold, after

    def _locate(self, energy):
        e = self._grid_e
        pos = (energy - e[0]) / (e[-1] - e[0]) * (e.size - 1)
        idx = np.clip(np.floor(pos).astype(np.int64), 0, e.size - 2)
        return idx, np.clip(pos - idx, 0.0, 1.0)

    def decide(self, ctx: DecisionContext) -> float:
        first = ctx.interval_start
        first_ns, now_ns = ts_ns(first), ts_ns(ctx.now)
        if self.schedule.due(first, self.timezone):
            hist = PriceHistory.from_view(
                ctx.view, now_ns - self.fit_days * DAY_NS, now_ns, self.timezone
            )
            if np.isfinite(hist.spread).sum() >= 4 * hist.per_day:
                self.chain = self.fit_chain(hist)
                self._tables_key = None
        n = published_horizon(ctx.view, first_ns, self.step_ns)
        end_ns = first_ns + n * self.step_ns
        if self.chain is None or n == 0 or not self._tables_ready(ctx, first_ns, end_ns):
            self._note_idle(ctx)
            return 0.0

        recent = PriceHistory.from_view(ctx.view, now_ns - 2 * HOUR_NS, now_ns, self.timezone)
        last = int(recent.latest_rt(now_ns))
        observed = recent.spread[last] if last >= 0 else np.nan
        z = int(self.chain.state(observed)) if np.isfinite(observed) else self._typical_state
        t = (first_ns - self._start_ns) // self.step_ns
        return self._best_action(ctx.soc_mwh, z, t) / self.dt

    def fit_chain(self, hist: PriceHistory) -> SpreadMarkovChain:
        """The spread model, refit on schedule from data published by then."""
        return SpreadMarkovChain.fit(hist)

    def _note_idle(self, ctx: DecisionContext) -> None:
        day = ctx.grid.days[ctx.day]
        if day not in self.days_without_forecast:
            self.days_without_forecast.append(day)

    def _tables_ready(self, ctx, first_ns: int, end_ns: int) -> bool:
        """(Re)build value tables when the horizon's end moves (once a day) or after a refit."""
        if self._tables_key == end_ns:
            return True
        n = (end_ns - first_ns) // self.step_ns
        hist = PriceHistory.from_view(ctx.view, first_ns, end_ns, self.timezone)
        da = hist.visible_da(np.arange(n), ts_ns(ctx.now))
        if np.isnan(da).any():
            return False
        hours = (
            local_naive_ns(first_ns + np.arange(-2, n) * self.step_ns, self.timezone) % DAY_NS
        ) // HOUR_NS
        self._solve(da, hours)
        self._start_ns = first_ns
        self._tables_key = end_ns
        return True

    def _solve(self, da: np.ndarray, hours: np.ndarray) -> None:
        """Backward induction; ``hours[t + 2]`` is the local hour of interval t."""
        chain, spec = self.chain, self.spec
        n, k = da.size, chain.n_states
        target = spec.soc_initial_mwh
        value = np.repeat((-TERMINAL_PENALTY * np.abs(self._grid_e - target))[:, None], k, axis=1)
        self._continuation = np.empty((n, self._grid_e.size, k))
        self._expected = np.empty((n, k))
        for t in range(n - 1, -1, -1):
            step1 = chain.transition[hours[t]]  # spread state of t-2 -> t-1
            expected = step1 @ chain.transition[hours[t + 1]] @ chain.values[hours[t + 2]]
            cont = value @ step1.T  # (energy, state now): expected value after this step
            self._continuation[t], self._expected[t] = cont, expected
            ahead = (
                cont[self._idx] * (1 - self._w)[..., None]
                + cont[self._idx + 1] * self._w[..., None]
            )
            price = da[t] + expected
            q = self._sold[..., None] * price - self._wear[..., None] + ahead
            value = q.max(axis=1)
        self.value = value  # V_0(stored energy grid, spread state): expected profit from now
        self._da = da
        self._typical_state = int(np.digitize(0.0, chain.edges))
        self.solves += 1

    def _best_action(self, energy: float, z: int, t: int) -> float:
        """Energy sold this interval (MWh) by the best action at the actual stored energy."""
        sold, after = self._outcomes(np.array(energy))
        idx, w = self._locate(after)
        cont = self._continuation[t][:, z]
        price = self._da[t] + self._expected[t, z]
        q = sold * price - self.spec.degradation_cost_per_mwh * np.maximum(sold, 0.0)
        q = q + cont[idx] * (1 - w) + cont[idx + 1] * w
        return float(sold[int(np.argmax(q))])
