"""Pooling saved backtests across locations (batt.evaluate.pool, ``batt pool``)."""

import os

import numpy as np
import pandas as pd
import pytest
import yaml
from typer.testing import CliRunner

from batt.cli import app
from batt.evaluate import latest_runs, moving_block_bootstrap_mean, pool_runs
from batt.evaluate.metrics import DAYS_PER_YEAR

DAYS = pd.date_range("2024-12-20", periods=30, freq="D")  # spans two years
SCALE = DAYS_PER_YEAR / (100 * 1000)  # $/day -> $/kW-year, 100 MW battery


def save_run(root, location, revenue: dict[str, np.ndarray], power_mw=100.0, days=DAYS):
    """A backtest folder as `batt backtest` writes it, with one ledger row per day."""
    run = root / f"{location}_2024-12-20_2025-01-19_0123456789"
    (run / "ledgers").mkdir(parents=True)
    config = {"backtest": {"location": location}, "battery": {"power_mw": power_mw}}
    (run / "config.yaml").write_text(yaml.safe_dump(config))
    for policy, values in revenue.items():
        ledger = pd.DataFrame(
            # The SoC ends where it started (50% of 200 MWh), so there is no end adjustment.
            {"local_date": days, "revenue_net": values, "rt_price": 40.0, "soc_end_mwh": 100.0}
        )
        ledger.to_parquet(run / "ledgers" / f"{policy}.parquet", index=False)
    return run


@pytest.fixture
def runs(tmp_path):
    rng = np.random.default_rng(0)
    base = {loc: rng.normal(1e5, 2e4, len(DAYS)) for loc in ("A", "B")}
    gain = {loc: rng.normal(mu, 1e4, len(DAYS)) for loc, mu in (("A", 5e3), ("B", 2e4))}
    return (
        {
            "A": save_run(tmp_path, "A", {"base": base["A"], "new": base["A"] + gain["A"]}),
            "B": save_run(
                tmp_path,
                "B",
                {"base": base["B"], "new": base["B"] + gain["B"], "only_b": base["B"]},
            ),
        },
        base,
        gain,
    )


def test_pool_averages_hubs_day_by_day(runs):
    folders, base, gain = runs
    levels, diffs = pool_runs(folders, "base", ["new", "only_b"], n_boot=200)

    assert levels.at["A", "base"] == pytest.approx(base["A"].mean() * SCALE)
    assert levels.at["pooled", "new"] == pytest.approx(
        (base["A"] + gain["A"] + base["B"] + gain["B"]).mean() / 2 * SCALE
    )
    assert np.isnan(levels.at["A", "only_b"]) and np.isnan(levels.at["pooled", "only_b"])

    # The pooled CI is the bootstrap of the hub-average daily difference: the same days
    # (and so the same weeks) at every hub, not separate draws per hub.
    pooled = (gain["A"] + gain["B"]) / 2
    ci = moving_block_bootstrap_mean(pooled, n_boot=200)
    row = diffs.loc[("new", "pooled")]
    assert row["mean"] == pytest.approx(pooled.mean() * SCALE)
    assert (row["low"], row["high"]) == pytest.approx((ci.low * SCALE, ci.high * SCALE))
    assert diffs.loc[("new", "A"), "mean"] == pytest.approx(gain["A"].mean() * SCALE)
    # by year: 12 days of 2024, 18 of 2025
    assert diffs.loc[("new", "pooled 2024"), "mean"] == pytest.approx(pooled[:12].mean() * SCALE)
    # a policy that only one hub ran gets that hub's row but no pooled rows
    assert ("only_b", "B") in diffs.index and ("only_b", "pooled") not in diffs.index


def test_days_after_rtc_b_pool_into_their_own_period(tmp_path):
    days = pd.date_range("2025-11-20", periods=30, freq="D")  # RTC+B starts on day 16
    gain = np.arange(30.0) * 1e3
    folders = {
        loc: save_run(tmp_path, loc, {"base": np.full(30, 1e5), "new": 1e5 + gain}, days=days)
        for loc in ("A", "B")
    }
    _, diffs = pool_runs(folders, "base", ["new"], n_boot=50)
    assert diffs.loc[("new", "pooled 2025"), "mean"] == pytest.approx(gain[:15].mean() * SCALE)
    assert diffs.loc[("new", "pooled 2025 RTC+B"), "mean"] == pytest.approx(
        gain[15:].mean() * SCALE
    )
    assert diffs.loc[("new", "pooled"), "mean"] == pytest.approx(gain.mean() * SCALE)


def test_pool_window_and_checks(runs, tmp_path):
    folders, base, gain = runs
    since, until = DAYS[5].date(), DAYS[25].date()
    _, diffs = pool_runs(folders, "base", ["new"], since=since, until=until, n_boot=50)
    pooled = (gain["A"] + gain["B"])[5:25] / 2
    assert diffs.loc[("new", "pooled"), "mean"] == pytest.approx(pooled.mean() * SCALE)

    other = tmp_path / "other"
    short = save_run(other, "C", {"base": base["A"][:20], "new": base["A"][:20]}, days=DAYS[:20])
    with pytest.raises(ValueError, match="different days"):
        pool_runs({**folders, "C": short}, "base", ["new"], n_boot=50)
    big = save_run(other, "D", {"base": base["A"], "new": base["A"]}, power_mw=50)
    with pytest.raises(ValueError, match="battery sizes"):
        pool_runs({**folders, "D": big}, "base", ["new"], n_boot=50)
    with pytest.raises(ValueError, match="baseline"):
        pool_runs(folders, "missing", ["new"], n_boot=50)


def test_latest_runs_picks_the_newest_folder(tmp_path):
    new = tmp_path / "HB_X_2024-01-01_2025-01-01_aaaa"  # sorts first by name
    old = tmp_path / "HB_X_2024-01-01_2025-01-01_bbbb"
    for folder, mtime in ((new, 2e9), (old, 1e9)):
        folder.mkdir()
        os.utime(folder, (mtime, mtime))
    found = latest_runs(tmp_path, ["HB_X"], "2024-01-01", "2025-01-01")
    assert found == {"HB_X": new}
    with pytest.raises(FileNotFoundError):
        latest_runs(tmp_path, ["HB_Y"], "2024-01-01", "2025-01-01")


def test_pool_cli(runs):
    folders, _, _ = runs
    args = ["pool", "--runs", ",".join(str(p) for p in folders.values())]
    result = CliRunner().invoke(app, [*args, "--baseline", "base", "--policies", "new"])
    assert result.exit_code == 0, result.output
    assert "pooled 2025" in result.output and "minus base" in result.output
    result = CliRunner().invoke(app, [*args, "--baseline", "nope", "--policies", "new"])
    assert result.exit_code == 1
