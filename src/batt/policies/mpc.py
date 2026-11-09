"""Model-predictive control: re-plan before every interval from a fresh price forecast."""

from __future__ import annotations

from datetime import date
from typing import Any, ClassVar

import numpy as np

from batt.config import BatterySpec, MarketTiming
from batt.data.store import ts_ns
from batt.forecast.analog import AnalogForecaster, AnchoredAnalogForecaster
from batt.forecast.history import MINUTE_NS, published_horizon
from batt.forecast.lookahead import LookAheadForecaster
from batt.forecast.models import (
    DayAheadForecaster,
    Forecaster,
    GBMForecaster,
    OracleForecaster,
    SpreadAR1Forecaster,
)
from batt.forecast.spike import SpikeForecaster
from batt.optimize.lp import Exclusive
from batt.optimize.rolling import RollingDispatchSolver
from batt.optimize.scenario import ScenarioDispatchSolver, step_sizes
from batt.policies.base import Policy
from batt.sim.context import DecisionContext
from batt.sim.grid import SimulationGrid


def _note_idle(days: list[date], ctx: DecisionContext) -> None:
    day = ctx.grid.days[ctx.day]
    if day not in days:
        days.append(day)


class ForecastMPC(Policy):
    """Rolling-horizon dispatch from a price forecast.

    Before each interval (5 minutes ahead), the forecaster predicts RT prices from that
    interval to the end of the last operating day whose DA prices are public: 10.5 hours
    ahead just before 13:30, up to 34.5 hours after. The battery LP is solved over that
    horizon from the current SoC, ending at the initial SoC, and only the first interval of
    the plan is executed. Because tomorrow's prices are in view from mid-afternoon, the
    battery no longer has to return to 50% every midnight as the daily-plan baselines do.

    Forecasters that learn are refit inside the simulation from the same point-in-time view,
    so their training data is also limited to what was public at the time.
    """

    forecaster_cls: ClassVar[type[Forecaster]]

    def __init__(
        self,
        replan_every: int = 1,
        forecaster_params: dict[str, Any] | None = None,
        exclusive: Exclusive = "auto",
    ):
        if replan_every < 1:
            raise ValueError("replan_every must be at least 1")
        self.replan_every = replan_every
        self.exclusive = exclusive
        self.forecaster = self.forecaster_cls(**(forecaster_params or {}))
        self._solver: RollingDispatchSolver | None = None
        self._step_ns = 0
        self._plan = np.zeros(0)
        self._plan_start = 0
        self._next_replan = 0
        self.days_without_forecast: list[date] = []

    def reset(self, grid: SimulationGrid, spec: BatterySpec, timing: MarketTiming) -> None:
        self.forecaster.reset(timing)
        self._solver = RollingDispatchSolver(spec, grid.dt_hours, exclusive=self.exclusive)
        self._step_ns = grid.interval_minutes * MINUTE_NS
        self._plan = np.zeros(0)
        self._plan_start = 0
        self._next_replan = 0
        self.days_without_forecast = []

    def decide(self, ctx: DecisionContext) -> float:
        if ctx.index >= self._next_replan or ctx.index - self._plan_start >= self._plan.size:
            self._plan = self._replan(ctx)
            self._plan_start = ctx.index
            self._next_replan = ctx.index + self.replan_every
        offset = ctx.index - self._plan_start
        return float(self._plan[offset]) if offset < self._plan.size else 0.0

    def _replan(self, ctx: DecisionContext) -> np.ndarray:
        first = ctx.interval_start
        n = published_horizon(ctx.view, ts_ns(first), self._step_ns)
        prices = np.full(0, np.nan)
        if n > 0:
            self.forecaster.prepare(ctx.view, ctx.now, first)
            source = ctx.oracle if self.forecaster.needs_oracle else ctx.view
            prices = self.forecaster.predict(source, ctx.now, first, n)
        unknown = np.isnan(prices)
        if unknown.any():
            prices = prices[: int(np.argmax(unknown))]  # plan over the known prefix only
        if prices.size == 0:
            _note_idle(self.days_without_forecast, ctx)
            return np.zeros(0)
        solution = self._solver.solve(prices, ctx.soc_mwh, ctx.spec.soc_initial_mwh)
        return solution.net_mw


class ScenarioMPC(Policy):
    """Two-stage stochastic MPC: re-plan before every interval against price scenarios.

    Same horizon and end-of-horizon SoC target as ``ForecastMPC``, but instead of trusting
    one forecast it chooses the action that does best on average across the forecaster's
    scenarios, given that each scenario's later steps can still adapt
    (``batt.optimize.scenario``). Steps are 15 minutes for the first ``fine_hours``, then
    hourly. With ``use_mean`` it plans against the scenarios' average instead: the same
    information and model, minus the uncertainty, which isolates what planning for
    uncertainty is worth.
    """

    forecaster_cls: ClassVar[type[Forecaster]]
    use_mean: ClassVar[bool] = False

    def __init__(
        self,
        fine_hours: float = 4.0,
        shared_hours: float = 0.25,
        forecaster_params: dict[str, Any] | None = None,
        exclusive: Exclusive = "auto",  # scenario plans don't enforce it; see the solver
    ):
        if fine_hours <= 0 or shared_hours <= 0:
            raise ValueError("fine_hours and shared_hours must be positive")
        self.fine_hours = fine_hours
        self.shared_hours = shared_hours
        self.exclusive = exclusive
        self.forecaster = self.forecaster_cls(**(forecaster_params or {}))
        self._solver: ScenarioDispatchSolver | None = None
        self._step_ns = 0
        self.days_without_forecast: list[date] = []

    def reset(self, grid: SimulationGrid, spec: BatterySpec, timing: MarketTiming) -> None:
        self.forecaster.reset(timing)
        n_scenarios = 1 if self.use_mean else self.forecaster.n_scenarios
        fine_steps = max(1, round(self.fine_hours * 60 / grid.interval_minutes))
        coarse = max(1, 60 // grid.interval_minutes)
        # Steps whose combined length first reaches shared_hours.
        hours = np.cumsum(step_sizes(10_000, fine_steps, coarse)) * grid.dt_hours
        shared_steps = int(np.searchsorted(hours, self.shared_hours - 1e-9)) + 1
        self._solver = ScenarioDispatchSolver(
            spec, grid.dt_hours, n_scenarios, fine_steps, coarse, shared_steps=shared_steps
        )
        self._step_ns = grid.interval_minutes * MINUTE_NS
        self.days_without_forecast = []

    def decide(self, ctx: DecisionContext) -> float:
        first = ctx.interval_start
        n = published_horizon(ctx.view, ts_ns(first), self._step_ns)
        known = 0
        if n > 0:
            self.forecaster.prepare(ctx.view, ctx.now, first)
            source = ctx.oracle if self.forecaster.needs_oracle else ctx.view
            paths = self.forecaster.scenarios(source, ctx.now, first, n)
            if self.use_mean:
                paths = paths.mean(axis=0, keepdims=True)
            finite = np.isfinite(paths).all(axis=0)
            known = n if finite.all() else int(np.argmin(finite))  # known prefix
        if known == 0:
            _note_idle(self.days_without_forecast, ctx)
            return 0.0
        solution = self._solver.solve(paths[:, :known], ctx.soc_mwh, ctx.spec.soc_initial_mwh)
        return solution.first_net_mw


class DayAheadMPC(ForecastMPC):
    """MPC that forecasts RT = DA: isolates the value of the rolling horizon."""

    name = "mpc_da"
    forecaster_cls = DayAheadForecaster


class SpreadAR1MPC(ForecastMPC):
    """MPC with DA prices plus a decaying latest RT-DA spread (AR(1))."""

    name = "mpc_ar1"
    forecaster_cls = SpreadAR1Forecaster


class GradientBoostingMPC(ForecastMPC):
    """MPC with the LightGBM forecaster, refit monthly inside the backtest."""

    name = "mpc_gbm"
    forecaster_cls = GBMForecaster


class OracleMPC(ForecastMPC):
    """MPC with perfect forecasts over the same horizon: the ceiling for any forecaster."""

    name = "mpc_oracle"
    is_oracle = True
    forecaster_cls = OracleForecaster


class SpikeMPC(ForecastMPC):
    """MPC on DA prices, moved for the next hour by the spike nowcast (``batt.forecast.spike``).

    When the nowcast holds no news (no fitted model yet, or probabilities equal to the prior),
    its forecast is the DA price and the policy acts exactly like ``mpc_da``.
    """

    name = "mpc_spike"
    forecaster_cls = SpikeForecaster

    def __init__(
        self,
        replan_every: int = 1,
        forecaster_params: dict[str, Any] | None = None,
        exclusive: Exclusive = "auto",
    ):
        if (forecaster_params or {}).get("perfect") is not None:
            raise ValueError("mpc_spike is realistic; perfect spike knowledge is mpc_oracle_spike")
        super().__init__(replan_every, forecaster_params, exclusive)


class LookAheadMPC(ForecastMPC):
    """MPC on ERCOT's own real-time look-ahead for the next hour, DA prices after it
    (``batt.forecast.lookahead``; plan in ``docs/grid-conditions-plan.md``)."""

    name = "mpc_rtd"
    forecaster_cls = LookAheadForecaster


class SpikeOracleMPC(ForecastMPC):
    """Benchmark: the next hour's spikes known in advance, DA prices otherwise.

    ``perfect: spikes`` (default) knows when spikes come and how big they are; ``perfect:
    timing`` knows only when, which is the ceiling of ``mpc_spike`` with a perfect classifier.
    """

    name = "mpc_oracle_spike"
    is_oracle = True
    forecaster_cls = SpikeForecaster

    def __init__(
        self,
        replan_every: int = 1,
        forecaster_params: dict[str, Any] | None = None,
        exclusive: Exclusive = "auto",
    ):
        params = {**(forecaster_params or {})}
        if params.get("perfect") is None:
            params["perfect"] = "spikes"
        super().__init__(replan_every, params, exclusive)


class AnalogMeanMPC(ScenarioMPC):
    """Plans against the average of the analog scenarios (the control for ``smpc_analog``)."""

    name = "mpc_analog"
    forecaster_cls = AnalogForecaster
    use_mean = True


class AnalogScenarioMPC(ScenarioMPC):
    """Stochastic MPC on analog-ensemble scenarios (DA prices + similar days' RT spreads)."""

    name = "smpc_analog"
    forecaster_cls = AnalogForecaster


class AnchoredAnalogScenarioMPC(ScenarioMPC):
    """Stochastic MPC on analog scenarios that start from the latest observed RT spread."""

    name = "smpc_analog_rt"
    forecaster_cls = AnchoredAnalogForecaster
