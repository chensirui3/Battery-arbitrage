"""Forecasters by name, as used in configs and on the command line."""

from __future__ import annotations

from typing import Any

from batt.forecast.analog import AnalogForecaster, AnchoredAnalogForecaster
from batt.forecast.lookahead import LookAheadForecaster
from batt.forecast.models import (
    DayAheadForecaster,
    Forecaster,
    GBMForecaster,
    OracleForecaster,
    SpreadAR1Forecaster,
)
from batt.forecast.quantile import QuantileGBMForecaster
from batt.forecast.spike import SpikeForecaster

FORECASTERS: dict[str, type[Forecaster]] = {
    cls.name: cls
    for cls in (
        DayAheadForecaster,
        SpreadAR1Forecaster,
        GBMForecaster,
        QuantileGBMForecaster,
        AnalogForecaster,
        AnchoredAnalogForecaster,
        SpikeForecaster,
        LookAheadForecaster,
        OracleForecaster,
    )
}


def make_forecaster(
    name: str,
    params: dict[str, Any] | None = None,
    catalog: dict[str, dict[str, Any]] | None = None,
) -> Forecaster:
    """Build a forecaster by name.

    ``params`` may name a registered forecaster under ``forecaster:``, which makes ``name`` a
    variant of it; the variant starts from that forecaster's settings in ``catalog`` (the
    config's ``forecasters`` section) and overrides some.
    """
    settings = dict(params or {})
    base = settings.pop("forecaster", name)
    if base not in FORECASTERS:
        raise KeyError(f"unknown forecaster {base!r}; available: {sorted(FORECASTERS)}")
    if base != name:
        settings = {**((catalog or {}).get(base) or {}), **settings}
    forecaster = FORECASTERS[base](**settings)
    if base != name:
        forecaster.name = name
    return forecaster


def uses_fundamentals(forecaster: Forecaster) -> bool:
    return bool(getattr(forecaster, "fundamentals", False))


def uses_conditions(forecaster: Forecaster) -> bool:
    return bool(getattr(forecaster, "conditions", False))
