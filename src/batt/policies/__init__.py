"""Dispatch policies and a registry to build them by name."""

from __future__ import annotations

from typing import Any

from batt.optimize.lp import Exclusive
from batt.policies.base import DailyPlanPolicy, Policy
from batt.policies.benchmarks import PerfectForesightDA, PerfectForesightRT
from batt.policies.mpc import (
    AnalogMeanMPC,
    AnalogScenarioMPC,
    AnchoredAnalogScenarioMPC,
    DayAheadMPC,
    ForecastMPC,
    GradientBoostingMPC,
    LookAheadMPC,
    OracleMPC,
    ScenarioMPC,
    SpikeMPC,
    SpikeOracleMPC,
    SpreadAR1MPC,
)
from batt.policies.schedules import DayAheadPriceSchedule, TrailingAverageSchedule
from batt.policies.sdp import StochasticDP

POLICIES: dict[str, type[Policy]] = {
    cls.name: cls
    for cls in (
        PerfectForesightRT,
        OracleMPC,
        SpikeOracleMPC,
        PerfectForesightDA,
        StochasticDP,
        AnchoredAnalogScenarioMPC,
        AnalogScenarioMPC,
        AnalogMeanMPC,
        LookAheadMPC,
        SpikeMPC,
        GradientBoostingMPC,
        SpreadAR1MPC,
        DayAheadMPC,
        DayAheadPriceSchedule,
        TrailingAverageSchedule,
    )
}


def policy_class(name: str, params: dict[str, Any] | None = None) -> type[Policy]:
    """The class behind a policy name. ``params`` may name a registered policy under
    ``policy:``, which makes ``name`` a variant of it (e.g. other settings for a study)."""
    base = (params or {}).get("policy", name)
    if base not in POLICIES:
        raise KeyError(f"unknown policy {base!r}; available: {sorted(POLICIES)}")
    return POLICIES[base]


def make_policy(
    name: str,
    params: dict[str, Any] | None = None,
    *,
    exclusive: Exclusive = "auto",
    forecasters: dict[str, dict[str, Any]] | None = None,
) -> Policy:
    """Build a policy by name (or a named variant, see ``policy_class``).

    MPC policies get their forecaster's settings from ``forecasters`` (keyed by forecaster
    name), updated with any ``forecaster_params`` in ``params``.
    """
    cls = policy_class(name, params)
    kwargs = {k: v for k, v in (params or {}).items() if k != "policy"}
    if issubclass(cls, ForecastMPC | ScenarioMPC):
        shared = (forecasters or {}).get(cls.forecaster_cls.name) or {}
        kwargs["forecaster_params"] = {**shared, **(kwargs.get("forecaster_params") or {})}
    policy = cls(**kwargs, exclusive=exclusive)
    if name != cls.name:
        policy.name = name  # results are labeled with the variant's name
    return policy


__all__ = [
    "POLICIES",
    "AnalogMeanMPC",
    "AnalogScenarioMPC",
    "AnchoredAnalogScenarioMPC",
    "DailyPlanPolicy",
    "DayAheadMPC",
    "DayAheadPriceSchedule",
    "ForecastMPC",
    "GradientBoostingMPC",
    "OracleMPC",
    "PerfectForesightDA",
    "PerfectForesightRT",
    "Policy",
    "ScenarioMPC",
    "LookAheadMPC",
    "SpikeMPC",
    "SpikeOracleMPC",
    "SpreadAR1MPC",
    "StochasticDP",
    "TrailingAverageSchedule",
    "make_policy",
    "policy_class",
]
