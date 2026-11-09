"""Typed configuration: battery specs, market timing assumptions, and backtest settings.

Every assumption lives here (and in ``configs/*.yaml``) rather than in code, so a result can
always be traced back to the exact assumptions that produced it.
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import date, time
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class BatterySpec(_Frozen):
    """Physical and economic parameters of the battery, measured on the AC (grid) side."""

    power_mw: float = Field(100.0, gt=0, description="Maximum charge and discharge power")
    energy_mwh: float = Field(200.0, gt=0, description="Nameplate energy capacity")
    round_trip_efficiency: float = Field(0.85, gt=0, le=1, description="AC-to-AC efficiency")
    soc_min_frac: float = Field(0.0, ge=0, le=1)
    soc_max_frac: float = Field(1.0, ge=0, le=1)
    soc_initial_frac: float = Field(
        0.5, ge=0, le=1, description="Starting state of charge, also the end-of-day target"
    )
    degradation_cost_per_mwh: float = Field(10.0, ge=0, description="$ per MWh discharged")
    max_cycles_per_day: float | None = Field(
        None, gt=0, description="Cap on equivalent full cycles per operating day"
    )

    @model_validator(mode="after")
    def _check_soc_window(self) -> BatterySpec:
        if not self.soc_min_frac < self.soc_max_frac:
            raise ValueError("soc_min_frac must be below soc_max_frac")
        if not self.soc_min_frac <= self.soc_initial_frac <= self.soc_max_frac:
            raise ValueError("soc_initial_frac must lie inside the SoC window")
        return self

    @property
    def eta_charge(self) -> float:
        """One-way charging efficiency (the round-trip efficiency split evenly)."""
        return math.sqrt(self.round_trip_efficiency)

    @property
    def eta_discharge(self) -> float:
        """One-way discharging efficiency."""
        return math.sqrt(self.round_trip_efficiency)

    @property
    def soc_min_mwh(self) -> float:
        return self.soc_min_frac * self.energy_mwh

    @property
    def soc_max_mwh(self) -> float:
        return self.soc_max_frac * self.energy_mwh

    @property
    def soc_initial_mwh(self) -> float:
        return self.soc_initial_frac * self.energy_mwh

    @property
    def usable_mwh(self) -> float:
        return self.soc_max_mwh - self.soc_min_mwh

    @property
    def duration_hours(self) -> float:
        return self.energy_mwh / self.power_mw


class MarketTiming(_Frozen):
    """When information becomes public and when decisions must be committed.

    Defaults follow ERCOT practice (Central Prevailing Time): day-ahead bids are due at 10:00
    the day before and results post around 13:30; real-time prices settle in 15-minute
    intervals and are published shortly after each interval ends.
    """

    timezone: str = "America/Chicago"
    interval_minutes: int = Field(15, gt=0)
    rt_publish_delay_minutes: float = Field(5.0, ge=0)
    decision_lead_minutes: float = Field(5.0, ge=0)
    da_publish_local_time: time = time(13, 30)
    da_bid_deadline_local_time: time = time(10, 0)

    @property
    def dt_hours(self) -> float:
        return self.interval_minutes / 60.0


class DataConfig(_Frozen):
    dir: Path = Path("data")
    years: list[int] = Field(default_factory=lambda: [2022, 2023, 2024, 2025, 2026])


class OptimizerConfig(_Frozen):
    # "auto" solves an LP and falls back to a MILP only if the LP charges and discharges
    # in the same interval (which it can profitably do at negative prices).
    exclusive: Literal["auto", "lp", "milp"] = "auto"


class BacktestConfig(_Frozen):
    location: str = "HB_NORTH"
    start: date = date(2022, 1, 1)
    end: date = date(2026, 11, 1)
    policies: list[str] = Field(
        default_factory=lambda: [
            "perfect_rt",
            "mpc_oracle",
            "perfect_da",
            "mpc_gbm",
            "mpc_ar1",
            "mpc_da",
            "da_informed",
            "naive",
        ]
    )
    benchmark: str = "perfect_rt"
    baseline: str | None = "da_informed"
    results_dir: Path = Path("results")

    @model_validator(mode="after")
    def _check_dates(self) -> BacktestConfig:
        if not self.start < self.end:
            raise ValueError("backtest.start must be before backtest.end")
        return self


class AppConfig(_Frozen):
    data: DataConfig = Field(default_factory=DataConfig)
    battery: BatterySpec = Field(default_factory=BatterySpec)
    timing: MarketTiming = Field(default_factory=MarketTiming)
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    policy_params: dict[str, dict[str, Any]] = Field(default_factory=dict)
    # Forecaster settings by forecaster name, shared by the MPC policies and `batt forecast`.
    forecasters: dict[str, dict[str, Any]] = Field(default_factory=dict)

    def fingerprint(self) -> str:
        """Short, stable hash of the full configuration, used to label result folders."""
        payload = json.dumps(self.model_dump(mode="json"), sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:10]

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.model_dump(mode="json"), sort_keys=False)

    def with_overrides(self, section: str, **values: Any) -> AppConfig:
        """Return a validated copy with some fields of one section replaced."""
        raw = self.model_dump(mode="json")
        raw[section].update({k: v for k, v in values.items() if v is not None})
        return AppConfig.model_validate(raw)


def load_config(path: str | Path | None = None) -> AppConfig:
    """Load a YAML config; missing sections and fields fall back to the defaults above."""
    raw: dict[str, Any] = {}
    if path is not None:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return AppConfig.model_validate(raw)
