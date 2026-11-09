"""Command-line interface: ``batt ingest``, ``ingest-fundamentals``, ``validate``, ``backtest``,
``forecast``, ``nowcast``, and ``pool``."""

from __future__ import annotations

import json
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer

from batt.config import AppConfig, load_config

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="ERCOT battery arbitrage: data, backtests, and results.",
)

DEFAULT_CONFIG = Path("configs/default.yaml")
ConfigOption = Annotated[Path, typer.Option("--config", "-c", help="YAML config file")]


def _load(config: Path) -> AppConfig:
    if not config.exists():
        typer.echo(f"config {config} not found; using built-in defaults", err=True)
        return load_config(None)
    return load_config(config)


@app.command()
def ingest(
    config: ConfigOption = DEFAULT_CONFIG,
    year: Annotated[
        list[int] | None,
        typer.Option("--year", "-y", help="Year to ingest (repeatable); default: data.years"),
    ] = None,
    force: Annotated[
        bool, typer.Option(help="Re-download and re-parse even if stored (use for this year)")
    ] = False,
) -> None:
    """Download ERCOT hub and load-zone prices (RT 15-min and DA hourly) and store them."""
    from batt.data import ercot
    from batt.data.validate import validate_prices

    cfg = _load(config)
    for y in year or cfg.data.years:
        for market in ("da", "rt"):
            target = ercot.clean_path(cfg.data.dir, market, y)
            if target.exists() and not force:
                typer.echo(f"{market} {y}: already stored ({target}); --force to refresh")
                continue
            started = time.perf_counter()
            df = ercot.ingest_year(market, y, cfg.data.dir, force=force)
            elapsed = time.perf_counter() - started
            typer.echo(f"{market} {y}: {len(df):,} rows ({elapsed:.0f}s)")
            hubs = df[df["location_type"] == "Trading Hub"]
            for report in validate_prices(hubs, market, cfg.timing.timezone):
                if not report.ok:
                    typer.echo("  " + report.summary())


SETUP_HELP = """\
ERCOT API credentials are missing: {missing}.

They are free. To set them up:
  1. Sign up at https://apiexplorer.ercot.com (Sign In, then "Sign up now").
  2. Open Products > Public API > Subscribe, then copy the primary key from your Profile.
  3. Open .env in this folder (copy .env.example to .env if it is missing) and type your
     ERCOT username (your email), password, and that key after the = signs. Git ignores .env.
Then check them with: batt ingest-fundamentals --check

Or add --recent to fetch the last few days from ERCOT's website, which needs no account."""


@app.command("ingest-fundamentals")
def ingest_fundamentals(
    config: ConfigOption = DEFAULT_CONFIG,
    start: Annotated[
        str | None,
        typer.Option(help="First day to fetch, YYYY-MM-DD (default: Jan 1 of the first data year)"),
    ] = None,
    end: Annotated[str | None, typer.Option(help="Day after the last one (default: now)")] = None,
    recent: Annotated[
        bool, typer.Option(help="Last few days only, from ERCOT's website (no account needed)")
    ] = False,
    force: Annotated[bool, typer.Option(help="Re-download periods already stored")] = False,
    reports: Annotated[
        str | None,
        typer.Option(
            help="Comma-separated reports to fetch: load, wind, solar (default: all). "
            "Separate runs can fetch different reports at the same time."
        ),
    ] = None,
    check: Annotated[
        bool,
        typer.Option(
            help="Only test the credentials and show how far back the archive goes "
            "(downloads nothing)"
        ),
    ] = False,
) -> None:
    """Download ERCOT load, wind, and solar forecasts and actuals, with publish times.

    History comes from the ERCOT Public API archive (free account; README, "Fundamentals data"),
    one report and month at a time; the run can be stopped and resumed. Expect hours for
    several years: about 720 publications per report and month.
    """
    from batt.data import fundamentals

    cfg = _load(config)
    fundamentals.load_dotenv(Path(".env"))
    now = pd.Timestamp.now(tz="US/Central")
    if recent:
        source, first = "mis", now.normalize() - pd.Timedelta(3, unit="D")
    else:
        source, first = "api", pd.Timestamp(f"{cfg.data.years[0]}-01-01")
        missing = fundamentals.missing_credentials()
        if missing:
            typer.echo(SETUP_HELP.format(missing=", ".join(missing)), err=True)
            raise typer.Exit(code=1)
    if check:
        years = range(min(cfg.data.years) - 2, now.year + 1)
        if not fundamentals.check_access(log=typer.echo, years=years):
            raise typer.Exit(code=1)
        return
    first = pd.Timestamp(start) if start else first
    last = pd.Timestamp(end) if end else now
    typer.echo(f"ERCOT fundamentals from {source.upper()}: {first:%Y-%m-%d} -> {last:%Y-%m-%d}")
    chosen = [r.strip() for r in reports.split(",") if r.strip()] if reports else None
    fundamentals.ingest(
        cfg.data.dir, first, last, source=source, force=force, reports=chosen, log=typer.echo
    )
    data = fundamentals.load_fundamentals(cfg.data.dir)
    coverage = data.coverage() if data is not None else None
    if coverage is None:
        typer.echo("nothing stored yet")
    else:
        typer.echo(
            f"stored: publications {coverage[0]:%Y-%m-%d %H:%M} -> {coverage[1]:%Y-%m-%d %H:%M} UTC"
        )


@app.command("ingest-conditions")
def ingest_conditions(
    config: ConfigOption = DEFAULT_CONFIG,
    start: Annotated[
        str | None,
        typer.Option(help="First day to fetch, YYYY-MM-DD (default: Jan 1 of the first data year)"),
    ] = None,
    end: Annotated[str | None, typer.Option(help="Day after the last one (default: now)")] = None,
    force: Annotated[bool, typer.Option(help="Re-download months already stored")] = False,
    reports: Annotated[
        str | None,
        typer.Option(help="Comma-separated reports: sced, rtd, outages (default: all)"),
    ] = None,
) -> None:
    """Download ERCOT real-time grid conditions with publish times: SCED prices and reserves
    (PRC), ERCOT's one-hour look-ahead (RTD), and hourly outage capacity.

    Uses the ERCOT Public API archive and the same free account as `ingest-fundamentals`. The
    5-minute reports keep the run each 15-minute decision would see. All of 2021-2026 takes
    about 10 hours; the run can be stopped and resumed, and running it again fetches any
    month that failed.
    """
    from batt.data import conditions, fundamentals

    cfg = _load(config)
    fundamentals.load_dotenv(Path(".env"))
    missing = fundamentals.missing_credentials()
    if missing:
        typer.echo(SETUP_HELP.format(missing=", ".join(missing)), err=True)
        raise typer.Exit(code=1)
    now = pd.Timestamp.now(tz="US/Central")
    first = pd.Timestamp(start) if start else pd.Timestamp(f"{cfg.data.years[0]}-01-01")
    last = pd.Timestamp(end) if end else now
    typer.echo(f"ERCOT grid conditions: {first:%Y-%m-%d} -> {last:%Y-%m-%d}")
    conditions.ingest(
        cfg.data.dir,
        first,
        last,
        reports=_parse_list(reports),
        force=force,
        step_minutes=cfg.timing.interval_minutes,
        lead_minutes=cfg.timing.decision_lead_minutes,
        log=typer.echo,
    )
    data = conditions.load_conditions(cfg.data.dir)
    coverage = data.coverage() if data is not None else None
    if coverage is None:
        typer.echo("not all three reports are stored yet")
        return
    typer.echo(
        f"stored: publications {coverage[0]:%Y-%m-%d %H:%M} -> {coverage[1]:%Y-%m-%d %H:%M} UTC"
    )
    # Finished months with nothing stored for a report; backtests over them are refused.
    upto = last if last.tzinfo is not None else last.tz_localize("US/Central")
    gaps = data.empty_months(first, min(upto, now.normalize().replace(day=1)))
    if gaps:
        typer.echo(f"missing: {', '.join(gaps)}. Run the same command again to fetch them.")
        raise typer.Exit(code=1)


@app.command()
def validate(
    config: ConfigOption = DEFAULT_CONFIG,
    location: Annotated[str | None, typer.Option(help="Only this settlement point")] = None,
    all_locations: Annotated[
        bool, typer.Option("--all", help="Include load zones, not just trading hubs")
    ] = False,
) -> None:
    """Check stored prices: complete days (incl. DST), no gaps, duplicates, or bad prices."""
    from batt.data import ercot
    from batt.data.validate import validate_prices

    cfg = _load(config)
    problems = 0
    for market in ("rt", "da"):
        df = ercot.load_clean(cfg.data.dir, market, location)
        if location is None and not all_locations:
            df = df[df["location_type"] == "Trading Hub"]
        for report in validate_prices(df, market, cfg.timing.timezone):
            typer.echo(report.summary())
            problems += not report.ok
    if problems:
        raise typer.Exit(code=1)


def _run_policy(name: str, cfg: AppConfig, market, grid):
    """Simulate one policy (module level so worker processes can run it)."""
    from batt.policies import make_policy
    from batt.sim import simulate

    policy = make_policy(
        name,
        cfg.policy_params.get(name),
        exclusive=cfg.optimizer.exclusive,
        forecasters=cfg.forecasters,
    )
    started = time.perf_counter()
    result = simulate(policy, market, grid, cfg.battery, cfg.timing)
    return result, time.perf_counter() - started


def _parse_list(value: str | None) -> list[str] | None:
    return [p.strip() for p in value.split(",") if p.strip()] if value else None


def _echo_fundamentals(market) -> None:
    coverage = market.fundamentals.coverage() if market.fundamentals is not None else None
    if coverage is not None:
        typer.echo(f"  fundamentals published {coverage[0]:%Y-%m-%d} -> {coverage[1]:%Y-%m-%d}")
    coverage = market.conditions.coverage() if market.conditions is not None else None
    if coverage is not None:
        typer.echo(f"  grid conditions published {coverage[0]:%Y-%m-%d} -> {coverage[1]:%Y-%m-%d}")


def _fundamentals_gap(market, start) -> str | None:
    """Why fundamentals can't support a run from ``start`` (None if they can)."""
    coverage = market.fundamentals.coverage() if market.fundamentals is not None else None
    if coverage is None:
        return "needs ERCOT fundamentals, which are not ingested (see `batt ingest-fundamentals`)"
    first = pd.Timestamp(start).tz_localize("UTC")
    if coverage[0] > first:
        return (
            f"needs ERCOT fundamentals from {first:%Y-%m-%d}, but they start "
            f"{coverage[0]:%Y-%m-%d} (ingest more, or use a later --start)"
        )
    return None


def _conditions_gap(market, start, end) -> str | None:
    """Why grid conditions can't support a run over [start, end) (None if they can)."""
    coverage = market.conditions.coverage() if market.conditions is not None else None
    if coverage is None:
        return "needs ERCOT grid conditions, which are not ingested (see `batt ingest-conditions`)"
    first, last = pd.Timestamp(start).tz_localize("UTC"), pd.Timestamp(end).tz_localize("UTC")
    if coverage[0] > first or coverage[1] < last - pd.Timedelta(1, unit="D"):
        return (
            f"needs ERCOT grid conditions from {first:%Y-%m-%d} to {last:%Y-%m-%d}, but they "
            f"cover {coverage[0]:%Y-%m-%d} to {coverage[1]:%Y-%m-%d} (see `batt ingest-conditions`)"
        )
    gaps = market.conditions.empty_months(first, last)
    if gaps:
        return (
            f"needs ERCOT grid conditions for every month, but nothing is stored for "
            f"{', '.join(gaps)} (run `batt ingest-conditions` again to fetch them)"
        )
    return None


def _data_gap(forecaster, market, start, end) -> str | None:
    """Why ``market`` lacks data ``forecaster`` needs over [start, end) (None if it has it)."""
    from batt.forecast.registry import uses_conditions, uses_fundamentals

    if uses_fundamentals(forecaster):
        gap = _fundamentals_gap(market, start)
        if gap:
            return gap
    if uses_conditions(forecaster):
        return _conditions_gap(market, start, end)
    return None


def _can_run_policy(name: str, cfg: AppConfig, market) -> bool:
    """False (with a note) for policies whose forecaster needs data we don't have."""
    from batt.policies import make_policy

    policy = make_policy(name, cfg.policy_params.get(name), forecasters=cfg.forecasters)
    forecaster = getattr(policy, "forecaster", None)
    if forecaster is not None:
        gap = _data_gap(forecaster, market, cfg.backtest.start, cfg.backtest.end)
        if gap:
            typer.echo(f"  skipping {name}: {gap}")
            return False
    return True


@app.command()
def backtest(
    config: ConfigOption = DEFAULT_CONFIG,
    location: Annotated[str | None, typer.Option(help="Settlement point, e.g. HB_NORTH")] = None,
    start: Annotated[str | None, typer.Option(help="First operating day, YYYY-MM-DD")] = None,
    end: Annotated[str | None, typer.Option(help="Day after the last operating day")] = None,
    policies: Annotated[str | None, typer.Option(help="Comma-separated policy names")] = None,
    n_boot: Annotated[int, typer.Option(help="Bootstrap resamples for CIs")] = 2000,
    jobs: Annotated[
        int, typer.Option(help="Policies simulated in parallel (0: one process per policy)")
    ] = 0,
) -> None:
    """Run every configured policy over the same period and compare them."""
    from batt.data.ercot import load_market
    from batt.evaluate import by_period, compare
    from batt.policies import policy_class
    from batt.sim import SimulationGrid

    cfg = _load(config)
    cfg = cfg.with_overrides(
        "backtest", location=location, start=start, end=end, policies=_parse_list(policies)
    )
    bt, timing = cfg.backtest, cfg.timing
    for name in bt.policies:
        try:
            policy_class(name, cfg.policy_params.get(name))
        except KeyError as exc:
            raise typer.BadParameter(str(exc), param_hint="--policies") from exc

    market = load_market(cfg.data.dir, bt.location, timing)
    grid = SimulationGrid.for_days(bt.start, bt.end, timing.timezone, timing.interval_minutes)
    spec = cfg.battery
    typer.echo(
        f"{bt.location}  {bt.start} -> {bt.end} ({grid.n_days} days)  "
        f"{spec.power_mw:g} MW / {spec.energy_mwh:g} MWh, RTE {spec.round_trip_efficiency:.0%}, "
        f"degradation ${spec.degradation_cost_per_mwh:g}/MWh"
    )
    _echo_fundamentals(market)
    names = [name for name in bt.policies if _can_run_policy(name, cfg, market)]

    workers = min(jobs or len(names), len(names), os.cpu_count() or 1)
    started = time.perf_counter()
    if workers <= 1:
        runs = []
        for name in names:
            runs.append(_run_policy(name, cfg, market, grid))
            typer.echo(f"  ran {name:<12} {runs[-1][1]:6.1f}s")
    else:
        # Start workers fresh ("spawn", the default on Windows and macOS), not by fork (Linux's
        # default before Python 3.14): a forked worker hangs if this process has already trained
        # LightGBM, whose OpenMP threads don't survive a fork. The test suite does exactly that.
        spawn = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=spawn) as pool:
            futures = [pool.submit(_run_policy, name, cfg, market, grid) for name in names]
            for name, future in zip(names, futures, strict=True):
                typer.echo(f"  ran {name:<12} {future.result()[1]:6.1f}s")
            runs = [future.result() for future in futures]
        typer.echo(f"  ({workers} processes, {time.perf_counter() - started:.0f}s in total)")
    results = [result for result, _ in runs]

    summary = compare(results, benchmark=bt.benchmark, baseline=bt.baseline, n_boot=n_boot)
    yearly, yearly_capture = by_period(results, "Y", bt.benchmark)
    monthly, monthly_capture = by_period(results, "M", bt.benchmark)

    out = bt.results_dir / f"{bt.location}_{bt.start}_{bt.end}_{cfg.fingerprint()}"
    (out / "ledgers").mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(cfg.to_yaml(), encoding="utf-8")
    summary.to_csv(out / "summary.csv")
    yearly.to_csv(out / "usd_per_kw_year_by_year.csv")
    monthly.to_csv(out / "usd_per_kw_year_by_month.csv")
    if yearly_capture is not None:
        yearly_capture.to_csv(out / "capture_by_year.csv")
        monthly_capture.to_csv(out / "capture_by_month.csv")
    for r in results:
        r.ledger.to_parquet(out / "ledgers" / f"{r.policy}.parquet", index=False)
    notes = {r.policy: {k: [str(d) for d in v] for k, v in r.notes.items() if v} for r in results}
    (out / "notes.json").write_text(json.dumps(notes, indent=2), encoding="utf-8")

    typer.echo("\n" + format_summary(summary, bt.baseline))
    typer.echo("\n$/kW-year by year:\n" + yearly.round(1).to_string())
    if yearly_capture is not None:
        typer.echo(
            "\nCapture rate by year:\n" + yearly_capture.map(lambda v: f"{v:.0%}").to_string()
        )
    for policy, items in notes.items():
        for key, days in items.items():
            typer.echo(
                f"\nnote: {policy} {key.replace('_', ' ')}: {len(days)} ({', '.join(days[:5])})"
            )
    typer.echo(f"\nsaved to {out}")


@app.command()
def forecast(
    config: ConfigOption = DEFAULT_CONFIG,
    location: Annotated[str | None, typer.Option(help="Settlement point, e.g. HB_NORTH")] = None,
    start: Annotated[str | None, typer.Option(help="First operating day, YYYY-MM-DD")] = None,
    end: Annotated[str | None, typer.Option(help="Day after the last operating day")] = None,
    forecasters: Annotated[
        str | None,
        typer.Option(
            help="Comma-separated forecaster names (default: all registered ones except the "
            "oracle, plus the config's named variants)"
        ),
    ] = None,
    issue_every: Annotated[int, typer.Option(help="Minutes between forecast issue times")] = 60,
) -> None:
    """Walk-forward accuracy of the RT price forecasters that the MPC policies use."""
    from batt.data.ercot import load_market
    from batt.forecast import evaluate_forecasters, make_forecaster, summarize_forecasts

    cfg = _load(config)
    cfg = cfg.with_overrides("backtest", location=location, start=start, end=end)
    bt, timing = cfg.backtest, cfg.timing
    if issue_every <= 0 or issue_every % timing.interval_minutes:
        raise typer.BadParameter("--issue-every must be a multiple of the interval length")
    from batt.forecast import FORECASTERS

    names = _parse_list(forecasters) or [
        *(name for name in FORECASTERS if name != "oracle"),
        *(name for name, settings in cfg.forecasters.items() if "forecaster" in (settings or {})),
    ]
    market = load_market(cfg.data.dir, bt.location, timing)
    _echo_fundamentals(market)
    models = {}
    for name in names:
        model = make_forecaster(name, cfg.forecasters.get(name), cfg.forecasters)
        gap = _data_gap(model, market, bt.start, bt.end)
        if gap:
            typer.echo(f"  skipping {name}: {gap}")
            continue
        models[name] = model

    started = time.perf_counter()
    errors = evaluate_forecasters(
        market, models, bt.start, bt.end, timing, issue_every // timing.interval_minutes
    )
    tables = summarize_forecasts(errors)
    out = bt.results_dir / f"forecast_{bt.location}_{bt.start}_{bt.end}_{cfg.fingerprint()}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(cfg.to_yaml(), encoding="utf-8")
    errors.to_csv(out / "errors_by_step.csv", index=False)
    for key, table in tables.items():
        table.to_csv(out / f"{key}.csv")
    for name, model in models.items():
        if getattr(model, "fit_log", None):  # one row per refit, e.g. the conditions' gain
            pd.DataFrame(model.fit_log).to_csv(out / f"fits_{name}.csv", index=False)

    typer.echo(
        f"{bt.location}  {bt.start} -> {bt.end}: forecasts issued every {issue_every} min "
        f"({time.perf_counter() - started:.0f}s)"
    )
    for key, title in (
        ("mae_by_lead", "MAE ($/MWh) by lead time"),
        ("rmse_by_lead", "RMSE ($/MWh) by lead time"),
        ("bias_by_lead", "Bias, forecast minus actual ($/MWh), by lead time"),
        ("crps_by_lead", "CRPS ($/MWh; equals MAE for point forecasts) by lead time"),
        ("mae_by_year", "MAE ($/MWh) by year, all lead times"),
        ("crps_by_year", "CRPS ($/MWh) by year, all lead times"),
    ):
        typer.echo(f"\n{title}:\n" + tables[key].round(1).to_string())
    for level in (50, 80):
        coverage = tables[f"coverage_{level}_by_lead"].dropna(how="all")
        if not coverage.empty:
            typer.echo(
                f"\nShare of prices inside the central {level}% of scenarios:\n"
                + coverage.map(lambda v: f"{v:.0%}").to_string()
            )
    typer.echo(f"\nsaved to {out}")


@app.command()
def nowcast(
    config: ConfigOption = DEFAULT_CONFIG,
    location: Annotated[str | None, typer.Option(help="Settlement point, e.g. HB_NORTH")] = None,
    start: Annotated[str | None, typer.Option(help="First operating day, YYYY-MM-DD")] = None,
    end: Annotated[str | None, typer.Option(help="Day after the last operating day")] = None,
    forecaster: Annotated[
        str, typer.Option(help="Spike nowcaster from the config, e.g. spike or spike_f")
    ] = "spike",
) -> None:
    """Walk-forward skill of the next-hour spike nowcaster, against the prior (what the DA
    prices already implied): AUC, average precision, Brier score, log loss, reliability."""
    from batt.data.ercot import load_market
    from batt.forecast import SpikeForecaster, evaluate_nowcasts, make_forecaster, nowcast_scores

    cfg = _load(config)
    cfg = cfg.with_overrides("backtest", location=location, start=start, end=end)
    bt, timing = cfg.backtest, cfg.timing
    try:
        model = make_forecaster(forecaster, cfg.forecasters.get(forecaster), cfg.forecasters)
    except KeyError as exc:
        raise typer.BadParameter(str(exc), param_hint="--forecaster") from exc
    if not isinstance(model, SpikeForecaster) or model.perfect is not None:
        raise typer.BadParameter(
            f"{forecaster} is not a spike nowcaster", param_hint="--forecaster"
        )
    market = load_market(cfg.data.dir, bt.location, timing)
    _echo_fundamentals(market)
    gap = _data_gap(model, market, bt.start, bt.end)
    if gap:
        typer.echo(f"{forecaster} {gap}", err=True)
        raise typer.Exit(code=1)

    started = time.perf_counter()
    frame = evaluate_nowcasts(market, model, bt.start, bt.end, timing)
    tables = nowcast_scores(frame, model.threshold)
    out = (
        bt.results_dir
        / f"nowcast_{forecaster}_{bt.location}_{bt.start}_{bt.end}_{cfg.fingerprint()}"
    )
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(cfg.to_yaml(), encoding="utf-8")
    frame.to_parquet(out / "nowcasts.parquet", index=False)
    pd.DataFrame(model.fit_log).to_csv(out / "fits.csv", index=False)
    for key, table in tables.items():
        table.to_csv(out / f"{key}.csv")

    typer.echo(
        f"{bt.location}  {bt.start} -> {bt.end}: {forecaster} nowcasts for every decision, "
        f"spikes beyond ${model.threshold:g}/MWh ({time.perf_counter() - started:.0f}s)"
    )
    columns = ["spike_rate", "mean_p", "auc", "auc_prior", "avg_precision"]
    columns += ["avg_precision_prior", "brier_skill"]
    for side in ("up", "down"):
        if side not in tables:
            typer.echo(f"\n{side}-spikes: no fitted model in this period")
            continue
        typer.echo(f"\n{side}-spikes (skill against the prior):\n" + _fmt(tables[side][columns]))
        typer.echo(f"\n{side}-spike reliability:\n" + _fmt(tables[f"{side}_reliability"]))
    typer.echo(f"\nsaved to {out}")


TRADING_HUBS = "HB_NORTH,HB_WEST,HB_SOUTH,HB_HOUSTON,HB_PAN"


@app.command()
def pool(
    policies: Annotated[str, typer.Option(help="Comma-separated policies to compare")],
    config: ConfigOption = DEFAULT_CONFIG,
    baseline: Annotated[
        str | None, typer.Option(help="Policy to compare with (default: backtest.baseline)")
    ] = None,
    locations: Annotated[str, typer.Option(help="Comma-separated settlement points")] = (
        TRADING_HUBS
    ),
    start: Annotated[str | None, typer.Option(help="First operating day of the runs")] = None,
    end: Annotated[str | None, typer.Option(help="Day after the runs' last day")] = None,
    since: Annotated[str | None, typer.Option(help="Count only days from this one on")] = None,
    until: Annotated[str | None, typer.Option(help="Count only days before this one")] = None,
    runs: Annotated[
        str | None,
        typer.Option(
            help="Comma-separated backtest folders, one per location, instead of each "
            "location's latest run for --start/--end"
        ),
    ] = None,
    n_boot: Annotated[int, typer.Option(help="Bootstrap resamples for CIs")] = 2000,
) -> None:
    """Compare saved backtests across locations: daily revenue averaged over the locations day
    by day, with CIs that resample the same weeks at every location."""
    from batt.evaluate.pool import POOLED, latest_runs, pool_runs, run_location

    cfg = _load(config)
    bt = cfg.with_overrides("backtest", start=start, end=end).backtest
    baseline = baseline or bt.baseline
    if not baseline:
        raise typer.BadParameter("no baseline: pass --baseline", param_hint="--baseline")
    try:
        if runs:
            folders = [Path(p) for p in _parse_list(runs)]
            chosen = {run_location(folder): folder for folder in folders}
            if len(chosen) < len(folders):
                raise typer.BadParameter("two runs share a location", param_hint="--runs")
        else:
            chosen = latest_runs(bt.results_dir, _parse_list(locations), bt.start, bt.end)
        cut = {k: v and pd.Timestamp(v).date() for k, v in (("since", since), ("until", until))}
        levels, differences = pool_runs(
            chosen, baseline, _parse_list(policies), n_boot=n_boot, **cut
        )
    except (FileNotFoundError, ValueError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    for location, folder in chosen.items():
        typer.echo(f"  {location:<12} {folder}")
    if since or until:
        typer.echo(f"  days from {since or 'the start'} to {until or 'the end'}")
    typer.echo("\n$/kW-year:\n" + levels.round(1).to_string())
    if differences.empty:
        return
    cells = differences.apply(
        lambda r: f"{r['mean']:+.1f} [{r['low']:+.1f}, {r['high']:+.1f}]", axis=1
    ).unstack("policy")
    scopes = [*chosen, POOLED, *sorted(s for s in cells.index if s.startswith(f"{POOLED} "))]
    cells = cells.reindex(
        index=[s for s in scopes if s in cells.index],
        columns=[p for p in levels.columns if p in cells.columns],
    )
    typer.echo(f"\nminus {baseline}, $/kW-year (95% CI):\n" + cells.fillna("").to_string())


def _fmt(table: pd.DataFrame) -> str:
    return table.to_string(float_format=lambda v: f"{v:.3f}")


def format_summary(table: pd.DataFrame, baseline: str | None) -> str:
    """Human-readable version of ``batt.evaluate.compare`` output."""
    view = pd.DataFrame(index=table.index)
    view["$/kW-yr"] = table["usd_per_kw_year"].map(lambda v: f"{v:,.1f}")
    if "capture_rate" in table:
        view["capture"] = table["capture_rate"].map(lambda v: f"{v:.0%}")
    view["cycles/day"] = table["cycles_per_day"].map(lambda v: f"{v:.2f}")
    view["sell $/MWh"] = table["avg_sell_price"].map(lambda v: f"{v:,.1f}")
    view["buy $/MWh"] = table["avg_buy_price"].map(lambda v: f"{v:,.1f}")
    view["losing days"] = table["losing_day_share"].map(lambda v: f"{v:.0%}")
    prefix = f"vs_{baseline}"
    if baseline and f"{prefix}_usd_kw_yr" in table:

        def fmt(row) -> str:
            if pd.isna(row[f"{prefix}_usd_kw_yr"]):
                return "-"
            return (
                f"{row[f'{prefix}_usd_kw_yr']:+.1f} "
                f"[{row[f'{prefix}_ci_low']:+.1f}, {row[f'{prefix}_ci_high']:+.1f}]"
            )

        view[f"vs {baseline} $/kW-yr (95% CI)"] = table.apply(fmt, axis=1)
    return view.to_string()


if __name__ == "__main__":  # pragma: no cover
    app()
