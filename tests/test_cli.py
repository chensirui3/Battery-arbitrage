"""End-to-end CLI runs on synthetic data (no network)."""

from datetime import date

import pandas as pd
import pytest
import yaml
from typer.testing import CliRunner

from batt.cli import app
from batt.data.ercot import clean_path
from batt.data.synthetic import synthetic_prices


@pytest.fixture(scope="module")
def config(tmp_path_factory):
    root = tmp_path_factory.mktemp("cli")
    rt, da = synthetic_prices(date(2024, 2, 1), 50, seed=3)
    for market, frame in (("rt", rt), ("da", da)):
        path = clean_path(root / "data", market, 2024)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)
    path = root / "config.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "data": {"dir": str(root / "data")},
                "backtest": {
                    "location": "SYN_HUB",
                    "start": "2024-03-04",
                    "end": "2024-03-18",
                    "results_dir": str(root / "results"),
                },
                "policy_params": {
                    "oracle_1h": {"policy": "mpc_oracle", "forecaster_params": {"hours": 1}},
                    "gbm_fund": {"policy": "mpc_gbm", "forecaster_params": {"fundamentals": True}},
                },
                "forecasters": {
                    "gbm": {"num_boost_round": 30},
                    "qgbm": {"num_boost_round": 20, "train_days": 30, "calib_days": 7},
                    "analog": {"n_scenarios": 5, "pool_days": 30},
                    "analog_rt": {"n_scenarios": 5, "pool_days": 30},
                    "spike": {"num_boost_round": 20, "threshold": 50, "min_positives": 5},
                },
            }
        )
    )
    return path


def test_validate(config):
    result = CliRunner().invoke(app, ["validate", "-c", str(config), "--location", "SYN_HUB"])
    assert result.exit_code == 0, result.output


@pytest.mark.parametrize("jobs", ["1", "2"])
def test_backtest(config, jobs):
    policies = "perfect_rt,oracle_1h,perfect_da,naive,da_informed,mpc_da,mpc_gbm,smpc_analog,sdp"
    args = ["backtest", "-c", str(config), "--n-boot", "100", "--jobs", jobs]
    result = CliRunner().invoke(app, [*args, "--policies", policies])
    assert result.exit_code == 0, result.output
    assert "capture" in result.output
    run_dirs = [p for p in (config.parent / "results").iterdir() if p.name.startswith("SYN")]
    summary = pd.read_csv(run_dirs[0] / "summary.csv", index_col=0)
    assert list(summary.index) == policies.split(",")
    assert summary.at["perfect_rt", "capture_rate"] == 1.0
    assert (run_dirs[0] / "ledgers" / "mpc_gbm.parquet").exists()


def test_backtest_skips_policies_that_need_missing_fundamentals(config):
    args = ["backtest", "-c", str(config), "--n-boot", "50", "--jobs", "1"]
    result = CliRunner().invoke(app, [*args, "--policies", "perfect_rt,mpc_da,gbm_fund"])
    assert result.exit_code == 0, result.output
    assert "skipping gbm_fund" in result.output and "ingest-fundamentals" in result.output
    assert "gbm_fund" not in result.output.split("skipping gbm_fund")[1]


def test_backtest_skips_policies_that_need_missing_grid_conditions(config):
    args = ["backtest", "-c", str(config), "--n-boot", "50", "--jobs", "1"]
    result = CliRunner().invoke(app, [*args, "--policies", "perfect_rt,mpc_da,mpc_rtd"])
    assert result.exit_code == 0, result.output
    assert "skipping mpc_rtd" in result.output and "ingest-conditions" in result.output


def test_backtest_rejects_unknown_policies(config):
    result = CliRunner().invoke(app, ["backtest", "-c", str(config), "--policies", "nope"])
    assert result.exit_code != 0
    assert "unknown policy" in result.output


def test_forecast(config):
    result = CliRunner().invoke(app, ["forecast", "-c", str(config), "--issue-every", "240"])
    assert result.exit_code == 0, result.output
    (run_dir,) = (config.parent / "results").glob("forecast_*")
    names = {"da", "ar1", "gbm", "qgbm", "analog", "analog_rt", "spike"}
    mae = pd.read_csv(run_dir / "mae_by_lead.csv", index_col=0)
    crps = pd.read_csv(run_dir / "crps_by_lead.csv", index_col=0)
    coverage = pd.read_csv(run_dir / "coverage_80_by_lead.csv", index_col=0)
    assert set(mae.index) == set(crps.index) == names
    assert (mae.to_numpy() > 0).all()
    # A point forecast's CRPS is its MAE; scenario forecasters have a coverage rate.
    pd.testing.assert_series_equal(crps.loc["da"], mae.loc["da"], check_names=False)
    assert coverage.loc[["qgbm", "analog", "analog_rt"]].notna().all().all()
    assert coverage.loc[["da", "ar1", "gbm"]].isna().all().all()


def test_nowcast(config):
    result = CliRunner().invoke(app, ["nowcast", "-c", str(config)])
    assert result.exit_code == 0, result.output
    assert "up-spikes (skill against the prior)" in result.output
    (run_dir,) = (config.parent / "results").glob("nowcast_spike_*")
    up = pd.read_csv(run_dir / "up.csv", index_col=0)
    assert {"all", "step 0", "step 3", "2024"} <= set(up.index)
    assert 0 <= up.at["all", "auc"] <= 1
    assert (run_dir / "nowcasts.parquet").exists() and (run_dir / "fits.csv").exists()


def test_nowcast_rejects_other_forecasters(config):
    result = CliRunner().invoke(app, ["nowcast", "-c", str(config), "--forecaster", "gbm"])
    assert result.exit_code != 0
    assert "not a spike nowcaster" in result.output
