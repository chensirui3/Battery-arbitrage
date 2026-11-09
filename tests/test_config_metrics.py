from pathlib import Path

import numpy as np
import pytest

from batt.config import AppConfig, BatterySpec, load_config
from batt.evaluate import moving_block_bootstrap_mean

ROOT = Path(__file__).resolve().parents[1]


def test_default_config_loads_and_fingerprint_is_stable():
    cfg = load_config(ROOT / "configs" / "default.yaml")
    assert cfg.battery.duration_hours == 2
    assert cfg.timing.da_publish_local_time.hour == 13
    assert cfg.fingerprint() == load_config(ROOT / "configs" / "default.yaml").fingerprint()
    assert cfg.fingerprint() != cfg.with_overrides("battery", energy_mwh=400).fingerprint()


def test_overrides_are_validated():
    with pytest.raises(ValueError):
        AppConfig().with_overrides("battery", round_trip_efficiency=1.5)


def test_soc_window_is_validated():
    with pytest.raises(ValueError):
        BatterySpec(soc_min_frac=0.6, soc_max_frac=0.5)


def test_bootstrap_interval_contains_mean_and_narrows_with_more_data():
    rng = np.random.default_rng(0)
    short = moving_block_bootstrap_mean(rng.normal(5, 10, 60), n_boot=500)
    long = moving_block_bootstrap_mean(rng.normal(5, 10, 2000), n_boot=500)
    assert short.low <= short.mean <= short.high
    assert (long.high - long.low) < (short.high - short.low)


def test_yearly_tables_keep_the_days_after_rtc_b_apart():
    from types import SimpleNamespace

    import pandas as pd

    from batt.evaluate import by_period

    days = pd.date_range("2025-11-20", periods=30, freq="D")  # RTC+B starts on day 16
    spec = BatterySpec()
    results = [
        SimpleNamespace(
            policy=name,
            spec=spec,
            ledger=pd.DataFrame(
                {"local_date": days, "revenue_net": value, "rt_price": 40.0, "soc_end_mwh": 100.0}
            ),
        )
        for name, value in (("perfect_rt", 2e5), ("mpc", 1e5))
    ]
    usd, capture = by_period(results, "Y", "perfect_rt")
    assert list(usd.index) == ["2025", "2025 RTC+B"]
    assert capture.loc["2025 RTC+B", "mpc"] == pytest.approx(0.5)
    monthly, _ = by_period(results, "M", "perfect_rt")
    assert [str(p) for p in monthly.index] == ["2025-11", "2025-12"]
