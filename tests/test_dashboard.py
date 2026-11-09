"""The dashboard (dashboard/): its statistics agree with the project's, its committed data
reproduce the README's test result, and the app runs."""

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from batt.evaluate import moving_block_bootstrap_mean, pool_runs
from batt.timeutils import period_labels

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dashboard"))
import bundle  # noqa: E402


def test_bootstrap_draws_the_same_blocks_as_the_backtests():
    rng = np.random.default_rng(3)
    for n in (5, 30, 400):
        values = rng.normal(0, 1, n)
        ci = moving_block_bootstrap_mean(values, n_boot=300)
        assert bundle.bootstrap_mean(values, n_boot=300) == pytest.approx(
            (ci.mean, ci.low, ci.high)
        )


def test_period_labels_agree():
    days = pd.date_range("2025-11-28", periods=20, freq="D")
    assert list(bundle.period_labels(days)) == list(period_labels(days))


def save_run(root: Path, hub: str, revenue: dict, days) -> Path:
    """A backtest folder as `batt backtest` writes it, one ledger row per day."""
    run = root / f"{hub}_run"
    (run / "ledgers").mkdir(parents=True)
    config = {"backtest": {"location": hub}, "battery": {"power_mw": 100.0}}
    (run / "config.yaml").write_text(yaml.safe_dump(config))
    for policy, values in revenue.items():
        ledger = pd.DataFrame(
            {"local_date": days, "revenue_net": values, "rt_price": 40.0, "soc_end_mwh": 100.0}
        )
        ledger.to_parquet(run / "ledgers" / f"{policy}.parquet", index=False)
    return run


def test_comparisons_match_batt_pool(tmp_path):
    days = pd.date_range("2025-11-01", periods=60, freq="D")  # RTC+B starts on day 35
    rng = np.random.default_rng(0)
    runs, rows = {}, []
    for hub in ("A", "B"):
        revenue = {"base": rng.normal(1e5, 2e4, 60), "new": rng.normal(1.05e5, 2e4, 60)}
        runs[hub] = save_run(tmp_path, hub, revenue, days)
        rows += [
            pd.DataFrame({"hub": hub, "policy": p, "date": days, "revenue": v})
            for p, v in revenue.items()
        ]
    data = bundle.Bundle(
        daily=pd.concat(rows, ignore_index=True),
        intervals=pd.DataFrame(),
        policies=pd.DataFrame(index=pd.Index(["base", "new"], name="policy")),
        forecasts=pd.DataFrame(),
        nowcasts=pd.DataFrame(),
        manifest={"battery": {"power_mw": 100.0}, "hubs": ["A", "B"]},
    )
    for since, until in ((None, None), (bundle.RTC_B, None), (None, bundle.RTC_B)):
        _, expected = pool_runs(runs, "base", ["new"], since=since, until=until)
        ours = bundle.compare(data, ["A", "B"], "base", ["new"], since, until)
        assert ours.set_index("policy").loc["new", ["mean", "low", "high"]].to_numpy() == (
            pytest.approx(expected.loc[("new", "pooled"), ["mean", "low", "high"]].to_numpy())
        )
    for hub in ("A", "B"):
        _, expected = pool_runs({hub: runs[hub]}, "base", ["new"])
        ours = bundle.compare(data, [hub], "base", ["new"]).set_index("policy")
        assert ours.at["new", "mean"] == pytest.approx(expected.loc[("new", hub), "mean"])


def test_the_committed_data_reproduce_the_readme():
    data = bundle.load()
    assert data.hubs == ["HB_NORTH", "HB_WEST", "HB_SOUTH", "HB_HOUSTON", "HB_PAN"]
    steps = np.diff(pd.DatetimeIndex(data.intervals["time"]).asi8)
    assert (steps == 15 * 60 * 10**9).all()  # a regular 15-minute grid in UTC
    # The test fixed before running: mpc_rtd minus mpc_da over the five hubs, 2022 to Oct 2026.
    result = bundle.compare(data, data.hubs, "mpc_da", ["mpc_rtd"]).set_index("policy")
    assert result.loc["mpc_rtd", ["mean", "low", "high"]].round(1).tolist() == [-2.6, -6.5, 0.5]
    levels = bundle.levels(data, "HB_NORTH")
    assert round(levels.at["mpc_da", "usd_kw_year"], 1) == 52.3
    assert round(levels.at["mpc_da", "capture"], 2) == 0.59
    # The clocks went back on 2025-11-02: 25 hours, each interval once.
    fall_back = bundle.day(data, date(2025, 11, 2), ["mpc_da"])
    assert len(fall_back) == 100 and fall_back["hour"].is_unique


def test_the_app_runs():
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(ROOT / "dashboard" / "app.py"), default_timeout=120).run()
    assert not app.exception
    app.selectbox(key="compare_period").set_value("Since RTC+B (December 5, 2025)").run()
    app.multiselect(key="compare_hubs").set_value(["HB_NORTH"]).run()
    app.selectbox(key="time_focus").set_value("mpc_spike").run()
    app.date_input(key="day").set_value(date(2025, 11, 2)).run()
    assert not app.exception
