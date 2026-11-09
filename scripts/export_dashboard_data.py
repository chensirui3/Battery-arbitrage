"""Export the compact data the dashboard reads (dashboard/data/).

    python scripts/export_dashboard_data.py

Reads the saved backtests in results/ (January 2022 to October 2026, as `batt backtest` writes
them), the forecast and nowcast scores, and ERCOT's look-ahead in data/conditions/. The dashboard
reads only dashboard/data/, so it runs, and deploys, without those folders.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from batt.config import load_config
from batt.data.conditions import LookAhead
from batt.evaluate.metrics import ledger_daily_net_revenue

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
CONDITIONS = ROOT / "data" / "conditions"
OUT = ROOT / "dashboard" / "data"
START, END = "2022-01-01", "2026-11-01"
HUBS = ["HB_NORTH", "HB_WEST", "HB_SOUTH", "HB_HOUSTON", "HB_PAN"]
DAY_HUB = "HB_NORTH"  # the hub whose intervals the bundle keeps
MINUTE_NS = 60 * 10**9
LOOKAHEAD_LEAD = 35  # minutes: a decision sees the third interval ahead this long before it

# Policy -> (group, short name, what it knows when it decides), as in the README's results.
POLICIES = {
    "perfect_rt": ("oracle", "Perfect foresight", "Realized RT prices, a month at a time"),
    "mpc_oracle": (
        "oracle",
        "Knows RT prices to the DA horizon",
        "Realized RT prices to the end of the published DA prices",
    ),
    "mpc_oracle_4h": (
        "oracle",
        "Knows the next 4 hours",
        "Realized RT prices for the next 4 hours, DA prices after",
    ),
    "mpc_oracle_1h": (
        "oracle",
        "Knows the next hour",
        "Realized RT prices for the next hour, DA prices after",
    ),
    "mpc_oracle_spike": (
        "oracle",
        "Knows the next hour's spikes",
        "Realized RT prices for the next hour where they are spikes (more than $100/MWh from DA), "
        "DA prices elsewhere",
    ),
    "mpc_oracle_timing": (
        "oracle",
        "Knows when the next hour spikes",
        "Which of the next hour's intervals spike, assuming a typical spike size",
    ),
    "perfect_da": ("oracle", "Perfect day-ahead foresight", "Realized DA prices, settled at DA"),
    "mpc_da": (
        "baseline",
        "Re-plans on day-ahead prices",
        "DA prices, re-planned every 15 minutes",
    ),
    "mpc_analog": ("realistic", "Analog scenarios, averaged", "The average of 20 analog scenarios"),
    "mpc_spike": (
        "realistic",
        "Spike nowcast",
        "DA prices, raised or lowered for the next hour by a spike nowcast while a spike is "
        "under way",
    ),
    "mpc_spike_f": (
        "realistic",
        "Spike nowcast + ERCOT's forecasts",
        "The spike nowcast, with ERCOT's load, wind, and solar forecasts as features",
    ),
    "mpc_spike_c": (
        "realistic",
        "Spike nowcast + grid conditions",
        "The spike nowcast, with real-time grid conditions as features",
    ),
    "mpc_ar1": (
        "realistic",
        "Latest spread, decaying",
        "DA prices plus the latest RT-DA spread, decaying",
    ),
    "mpc_rtd": (
        "realistic",
        "ERCOT's one-hour look-ahead",
        "ERCOT's own one-hour price look-ahead where it is more than $100/MWh from DA, DA prices "
        "elsewhere",
    ),
    "mpc_gbm": ("realistic", "LightGBM forecast", "A LightGBM price forecast"),
    "mpc_gbm_f": (
        "realistic",
        "LightGBM + ERCOT's forecasts",
        "The LightGBM forecast, with ERCOT's load, wind, and solar forecasts as features",
    ),
    "mpc_gbm_c": (
        "realistic",
        "LightGBM + grid conditions",
        "The LightGBM forecast, with real-time grid conditions as features",
    ),
    "mpc_gbm_1h": (
        "realistic",
        "LightGBM, next hour only",
        "The LightGBM forecast for the next hour, DA prices after",
    ),
    "smpc_analog_4h": (
        "realistic",
        "Stochastic MPC, 4-hour shared plan",
        "Analog scenarios: one plan for the first 4 hours, then one per scenario",
    ),
    "smpc_analog_1h": (
        "realistic",
        "Stochastic MPC, 1-hour shared plan",
        "Analog scenarios: one plan for the first hour, then one per scenario",
    ),
    "smpc_analog": (
        "realistic",
        "Two-stage stochastic MPC",
        "Analog scenarios: one action, then one plan per scenario",
    ),
    "smpc_analog_rt": (
        "realistic",
        "Stochastic MPC from the latest spread",
        "Two-stage stochastic MPC, with scenarios that start from the latest RT spread",
    ),
    "sdp": (
        "realistic",
        "Stochastic dynamic programming",
        "Stochastic dynamic programming on a Markov model of the spread",
    ),
    "da_informed": ("realistic", "Plans once a day", "DA prices, in one plan a day"),
    "naive": ("realistic", "Last 30 days' average shape", "The last 30 days' average daily shape"),
}


def runs(hub: str) -> list[Path]:
    """Saved backtests of ``hub`` over the period, newest first."""
    found = RESULTS.glob(f"{hub}_{START}_{END}_*")
    return sorted(found, key=lambda p: p.stat().st_mtime, reverse=True)


def ledgers(hub: str) -> dict[str, Path]:
    """Each policy's ledger from the newest backtest of ``hub`` that ran it."""
    chosen: dict[str, Path] = {}
    for run in runs(hub):
        for path in sorted((run / "ledgers").glob("*.parquet")):
            if path.stem in POLICIES:
                chosen.setdefault(path.stem, path)
    return chosen


def daily_revenue() -> tuple[pd.DataFrame, dict, dict]:
    frames, sources, specs = [], {}, set()
    for hub in HUBS:
        sources[hub] = {}
        for policy, path in ledgers(hub).items():
            spec = load_config(path.parents[1] / "config.yaml").battery
            specs.add(spec)
            day = ledger_daily_net_revenue(pd.read_parquet(path), spec)
            frames.append(
                pd.DataFrame(
                    {
                        "hub": hub,
                        "policy": policy,
                        "date": pd.to_datetime(day.index),
                        "revenue": day.to_numpy(),
                    }
                )
            )
            sources[hub][policy] = path.parents[1].name
    if len(specs) != 1:
        raise ValueError(f"the backtests use different batteries: {specs}")
    spec = specs.pop()
    battery = {
        "power_mw": spec.power_mw,
        "energy_mwh": spec.energy_mwh,
        "round_trip_efficiency": spec.round_trip_efficiency,
        "degradation_cost_per_mwh": spec.degradation_cost_per_mwh,
    }
    daily = pd.concat(frames, ignore_index=True)
    for column in ("hub", "policy"):
        daily[column] = daily[column].astype("category")
    return daily, sources, battery


def lookahead(starts: pd.DatetimeIndex) -> np.ndarray:
    """ERCOT's look-ahead price for each interval, as published LOOKAHEAD_LEAD minutes before."""
    files = sorted(CONDITIONS.glob("rtd_*.parquet"))
    if not files:
        return np.full(len(starts), np.nan)
    ahead = LookAhead.from_frame(pd.concat([pd.read_parquet(f) for f in files]))
    begin = starts.asi8
    return ahead.mean_price(begin - LOOKAHEAD_LEAD * MINUTE_NS, begin, begin + 15 * MINUTE_NS)


def cents(values) -> pd.arrays.IntegerArray:
    return pd.array(np.round(np.asarray(values, dtype=float) * 100), dtype="Int32")


def intervals() -> pd.DataFrame:
    """Prices in cents, the look-ahead, and every policy's stored energy in whole MWh, interval
    by interval (UTC starts, every 15 minutes without gaps)."""
    paths = ledgers(DAY_HUB)
    base = pd.read_parquet(paths["mpc_da"]).set_index("interval_start_utc")
    steps = np.diff(base.index.asi8)
    if not (steps == 15 * MINUTE_NS).all():
        raise ValueError("the intervals are not a regular 15-minute grid")
    frame = pd.DataFrame(
        {
            "rt_cents": cents(base["rt_price"]),
            "da_cents": cents(base["da_price"]),
            "lookahead_cents": cents(lookahead(base.index)),
        },
        index=base.index,
    )
    for policy, path in paths.items():
        soc = pd.read_parquet(path).set_index("interval_start_utc")["soc_end_mwh"]
        if not soc.index.equals(frame.index):
            raise ValueError(f"{policy}'s intervals differ from mpc_da's")
        frame[f"soc_{policy}"] = soc.round().clip(0, 255).astype("uint8")
    return frame.rename_axis("time").reset_index()


def write_intervals(frame: pd.DataFrame, path: Path) -> None:
    """Delta-encode the regular timestamps and the prices, which shrinks them several-fold."""
    delta = ["time", "rt_cents", "da_cents", "lookahead_cents"]
    pq.write_table(
        pa.Table.from_pandas(frame, preserve_index=False),
        path,
        compression="zstd",
        use_dictionary=[c for c in frame.columns if c not in delta],
        column_encoding=dict.fromkeys(delta, "DELTA_BINARY_PACKED"),
    )


def forecast_scores() -> pd.DataFrame:
    """CRPS and MAE by lead time, each forecaster from its newest `batt forecast` run."""
    found = sorted(
        RESULTS.glob(f"forecast_{DAY_HUB}_{START}_{END}_*"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    rows, seen = [], set()
    for run in found:
        if not (run / "crps_by_lead.csv").exists():  # runs from before CRPS was scored
            continue
        crps = pd.read_csv(run / "crps_by_lead.csv", index_col=0)
        mae = pd.read_csv(run / "mae_by_lead.csv", index_col=0)
        for name in crps.index.difference(seen):
            for lead in crps.columns:
                rows.append(
                    {
                        "forecaster": name,
                        "lead": lead,
                        "crps": crps.at[name, lead],
                        "mae": mae.at[name, lead],
                    }
                )
            seen.add(name)
    return pd.DataFrame(rows)


def nowcast_scores() -> pd.DataFrame:
    """Spike classifiers' skill against the prior, overall and by period."""
    rows = []
    for forecaster in ("spike", "spike_c"):
        found = sorted(
            RESULTS.glob(f"nowcast_{forecaster}_{DAY_HUB}_{START}_{END}_*"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not found:
            continue
        for side in ("up", "down"):
            table = pd.read_csv(found[0] / f"{side}.csv", index_col=0)
            table = table[~table.index.str.startswith("step")]
            for period, row in table.iterrows():
                rows.append({"forecaster": forecaster, "side": side, "period": period, **row})
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    daily, sources, battery = daily_revenue()
    daily.to_parquet(OUT / "daily.parquet", index=False, compression="zstd")
    write_intervals(intervals(), OUT / "intervals.parquet")
    forecast_scores().to_csv(OUT / "forecasts.csv", index=False)
    nowcast_scores().to_csv(OUT / "nowcasts.csv", index=False)
    policies = pd.DataFrame(
        [(p, *values) for p, values in POLICIES.items()],
        columns=["policy", "group", "name", "description"],
    )
    policies.to_csv(OUT / "policies.csv", index=False)
    manifest = {
        "exported": date.today().isoformat(),
        "period": {"start": START, "end": END},
        "hubs": HUBS,
        "interval_hub": DAY_HUB,
        "battery": battery,
        "lookahead_lead_minutes": LOOKAHEAD_LEAD,
        "sources": sources,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    for path in sorted(OUT.iterdir()):
        print(f"{path.name:20} {path.stat().st_size / 1e6:6.2f} MB")


if __name__ == "__main__":
    main()
