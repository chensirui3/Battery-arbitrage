r"""Battery dispatch as a linear program, solved with HiGHS through SciPy.

For intervals k = 0..T-1 of length dt hours, with prices p_k ($/MWh):

    c_k  charging power drawn from the grid (MW)        0 <= c_k <= P
    d_k  discharging power delivered to the grid (MW)   0 <= d_k <= P
    s_k  stored energy at the END of interval k (MWh)   S_min <= s_k <= S_max

    maximize    sum_k p_k (d_k - c_k) dt  -  k_deg sum_k d_k dt
    subject to  s_k = s_{k-1} + (eta_c c_k - d_k / eta_d) dt,     s_{-1} = s_init
                s_{T-1} = s_terminal                               (optional)
                sum_{k in g} d_k dt <= N_cycles * eta_d * (S_max - S_min)  per group g (optional)

eta_c = eta_d = sqrt(round-trip efficiency). Ending at a fixed SoC matters: without it the
optimizer sells all stored energy at the end of every horizon and overstates revenue.

At negative prices an LP can profit by charging and discharging in the same interval, burning
energy through conversion losses. Real batteries can't do that, so ``exclusive="auto"``
re-solves as a MILP, with a binary charge/discharge mode per interval, whenever the LP
solution does it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import scipy.sparse as sp
from scipy.optimize import Bounds, LinearConstraint, linprog, milp

from batt.config import BatterySpec

Exclusive = Literal["auto", "lp", "milp"]
TOL = 1e-6


class OptimizationError(RuntimeError):
    """The solver failed to find an optimal schedule."""


@dataclass(frozen=True)
class DispatchSolution:
    charge_mw: np.ndarray
    discharge_mw: np.ndarray
    soc_mwh: np.ndarray  # stored energy at the end of each interval
    objective: float  # revenue minus degradation cost at the input prices, $
    used_milp: bool

    @property
    def net_mw(self) -> np.ndarray:
        """Net injection: positive = discharging (selling), negative = charging (buying)."""
        return self.discharge_mw - self.charge_mw

    @property
    def has_simultaneous(self) -> bool:
        return bool(np.any(np.minimum(self.charge_mw, self.discharge_mw) > TOL))


@dataclass(frozen=True)
class _Problem:
    prices: np.ndarray
    dt: float
    spec: BatterySpec
    cost: np.ndarray
    a_eq: sp.csr_matrix
    b_eq: np.ndarray
    a_ub: sp.csr_matrix | None
    b_ub: np.ndarray | None
    lb: np.ndarray
    ub: np.ndarray

    @property
    def n_intervals(self) -> int:
        return len(self.prices)


def reachable_soc(spec: BatterySpec, soc_initial: float, n_intervals: int, dt: float):
    """Lowest and highest SoC reachable from ``soc_initial`` within ``n_intervals``."""
    span = n_intervals * dt * spec.power_mw
    low = max(spec.soc_min_mwh, soc_initial - span / spec.eta_discharge)
    high = min(spec.soc_max_mwh, soc_initial + span * spec.eta_charge)
    return low, high


def check_prices(prices) -> np.ndarray:
    p = np.asarray(prices, dtype=float)
    if p.ndim != 1 or p.size == 0:
        raise ValueError("prices must be a non-empty 1-D array")
    if not np.all(np.isfinite(p)):
        raise ValueError("prices contain NaN or infinite values")
    return p


def soc_endpoints(
    spec: BatterySpec, n_intervals: int, dt: float, soc_initial: float, soc_terminal: float | None
) -> tuple[float, float | None]:
    """Initial SoC clipped to the SoC window; terminal SoC moved into the reachable range."""
    soc0 = float(np.clip(soc_initial, spec.soc_min_mwh, spec.soc_max_mwh))
    if soc_terminal is None:
        return soc0, None
    low, high = reachable_soc(spec, soc0, n_intervals, dt)
    return soc0, float(np.clip(soc_terminal, low, high))


def make_solution(prices, dt, spec, charge, discharge, soc, used_milp) -> DispatchSolution:
    """Clean a solver's raw output (tiny negatives, round-off) and price it."""
    power = spec.power_mw
    charge = np.clip(charge, 0.0, power)
    discharge = np.clip(discharge, 0.0, power)
    charge[charge < TOL] = 0.0
    discharge[discharge < TOL] = 0.0
    profit = np.sum(prices * (discharge - charge)) * dt
    degradation = spec.degradation_cost_per_mwh * np.sum(discharge) * dt
    soc = np.asarray(soc, dtype=float)
    return DispatchSolution(charge, discharge, soc, float(profit - degradation), used_milp)


def optimize_dispatch(
    prices,
    spec: BatterySpec,
    dt_hours: float,
    soc_initial_mwh: float,
    soc_terminal_mwh: float | None = None,
    *,
    exclusive: Exclusive = "auto",
    cycle_groups=None,
) -> DispatchSolution:
    """Revenue-maximizing charge/discharge schedule for a known (or forecast) price path.

    ``cycle_groups`` labels each interval with its operating day; it is only used when
    ``spec.max_cycles_per_day`` is set (without labels the whole horizon is one group with a
    prorated budget). A terminal SoC that cannot be reached in time is moved to the nearest
    reachable value.
    """
    p = check_prices(prices)
    soc0, soc_end = soc_endpoints(spec, p.size, dt_hours, soc_initial_mwh, soc_terminal_mwh)
    problem = _build(p, spec, dt_hours, soc0, soc_end, cycle_groups)
    if exclusive == "milp":
        return _solve_milp(problem)
    solution = _solve_lp(problem)
    if exclusive == "auto" and solution.has_simultaneous:
        return _solve_milp(problem)
    return solution


def _build(p, spec, dt, soc0, soc_end, cycle_groups) -> _Problem:
    n = p.size
    k_deg = spec.degradation_cost_per_mwh
    # linprog minimizes, so the objective is the negative of profit.
    cost = np.concatenate([p * dt, -(p - k_deg) * dt, np.zeros(n)])

    eye = sp.identity(n, format="csr")
    previous = sp.eye(n, k=-1, format="csr")  # (previous @ s)_k = s_{k-1}
    a_eq = sp.hstack(
        [-spec.eta_charge * dt * eye, (dt / spec.eta_discharge) * eye, eye - previous],
        format="csr",
    )
    b_eq = np.zeros(n)
    b_eq[0] = soc0

    lb = np.concatenate([np.zeros(2 * n), np.full(n, spec.soc_min_mwh)])
    ub = np.concatenate([np.full(2 * n, spec.power_mw), np.full(n, spec.soc_max_mwh)])
    if soc_end is not None:
        lb[-1] = ub[-1] = soc_end

    a_ub = b_ub = None
    if spec.max_cycles_per_day is not None:
        # One equivalent full cycle delivers eta_d * usable energy to the grid.
        per_cycle = spec.eta_discharge * spec.usable_mwh
        if cycle_groups is None:
            groups = np.zeros(n, dtype=int)
            budget = np.array([spec.max_cycles_per_day * per_cycle * n * dt / 24])
        else:
            _, groups = np.unique(np.asarray(cycle_groups), return_inverse=True)
            budget = np.full(groups.max() + 1, spec.max_cycles_per_day * per_cycle)
        rows = groups
        cols = n + np.arange(n)
        a_ub = sp.csr_matrix((np.full(n, dt), (rows, cols)), shape=(budget.size, 3 * n))
        b_ub = budget
    return _Problem(p, dt, spec, cost, a_eq, b_eq, a_ub, b_ub, lb, ub)


def _unpack(problem: _Problem, x: np.ndarray, used_milp: bool) -> DispatchSolution:
    n = problem.n_intervals
    return make_solution(
        problem.prices,
        problem.dt,
        problem.spec,
        x[:n],
        x[n : 2 * n],
        x[2 * n : 3 * n],
        used_milp,
    )


def _solve_lp(problem: _Problem) -> DispatchSolution:
    result = linprog(
        problem.cost,
        A_ub=problem.a_ub,
        b_ub=problem.b_ub,
        A_eq=problem.a_eq,
        b_eq=problem.b_eq,
        bounds=np.column_stack([problem.lb, problem.ub]),
        method="highs",
    )
    if result.status != 0:
        raise OptimizationError(f"LP failed: {result.message}")
    return _unpack(problem, result.x, used_milp=False)


def _solve_milp(problem: _Problem) -> DispatchSolution:
    n = problem.n_intervals
    power = problem.spec.power_mw
    eye = sp.identity(n, format="csr")
    zero = sp.csr_matrix((n, n))
    # Binary u_k = 1 allows charging only, u_k = 0 discharging only:
    #   c_k - P u_k <= 0   and   d_k + P u_k <= P
    mode = sp.vstack(
        [sp.hstack([eye, zero, zero, -power * eye]), sp.hstack([zero, eye, zero, power * eye])],
        format="csr",
    )
    constraints = [
        LinearConstraint(sp.hstack([problem.a_eq, zero], format="csr"), problem.b_eq, problem.b_eq),
        LinearConstraint(
            mode, np.full(2 * n, -np.inf), np.concatenate([np.zeros(n), np.full(n, power)])
        ),
    ]
    if problem.a_ub is not None:
        extra = sp.csr_matrix((problem.a_ub.shape[0], n))
        constraints.append(
            LinearConstraint(sp.hstack([problem.a_ub, extra], format="csr"), -np.inf, problem.b_ub)
        )
    result = milp(
        np.concatenate([problem.cost, np.zeros(n)]),
        integrality=np.concatenate([np.zeros(3 * n), np.ones(n)]),
        bounds=Bounds(
            np.concatenate([problem.lb, np.zeros(n)]), np.concatenate([problem.ub, np.ones(n)])
        ),
        constraints=constraints,
        options={"mip_rel_gap": 1e-7},
    )
    if result.status != 0 or result.x is None:
        raise OptimizationError(f"MILP failed: {result.message}")
    return _unpack(problem, result.x[: 3 * n], used_milp=True)
