from datetime import date

import pytest

from batt.config import BatterySpec, MarketTiming
from batt.data.conditions import GridConditions
from batt.data.store import MarketData
from batt.data.synthetic import synthetic_conditions, synthetic_prices
from batt.sim.grid import SimulationGrid

TZ = "America/Chicago"


@pytest.fixture(scope="session")
def timing() -> MarketTiming:
    return MarketTiming()


@pytest.fixture(scope="session")
def spec() -> BatterySpec:
    return BatterySpec()


@pytest.fixture(scope="session")
def frames():
    # 70 days of synthetic prices spanning the March 2024 daylight-saving change.
    return synthetic_prices(date(2024, 2, 1), 70, seed=7)


@pytest.fixture(scope="session")
def market(frames, timing) -> MarketData:
    """Synthetic prices with synthetic grid conditions (``batt.data.conditions``) attached."""
    rt, da = frames
    market = MarketData.from_price_frames("SYN_HUB", rt, da, timing)
    return market.with_conditions(GridConditions.from_frames(*synthetic_conditions(rt, seed=7)))


@pytest.fixture(scope="session")
def grid(timing) -> SimulationGrid:
    # Two weeks, including the 92-interval DST day (2024-03-10), with a month of history before.
    return SimulationGrid.for_days(date(2024, 3, 4), date(2024, 3, 18), timing.timezone)
