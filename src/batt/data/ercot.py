"""Download and parse ERCOT's yearly hub and load-zone settlement point price files.

ERCOT publishes one zip per year for real-time prices (report 13061, 15-minute intervals) and
day-ahead prices (report 13060, hourly). Each zip is kept unchanged under ``data/raw`` with
ERCOT's own file name, which includes its publish timestamp. Parsing reuses gridstatus's ERCOT
helpers, which turn hour-ending labels and daylight-saving flags into proper timestamps.

gridstatus's own ``get_rtm_spp`` merges each load zone's plain price (type LZ) with its
energy-weighted price (type LZEW) under one name: it looks for a column called
``SettlementPointType`` while the yearly files call it ``Settlement Point Type``. Renaming the
column first keeps them apart, so energy-weighted zones come out as e.g. ``LZ_NORTH_EW``.
"""

from __future__ import annotations

import zipfile
from collections.abc import Iterable
from pathlib import Path

import pandas as pd
import requests

from batt.config import MarketTiming
from batt.data.store import MarketData

REPORT_TYPE_IDS = {"rt": 13061, "da": 13060}
FILE_PREFIXES = {"rt": "RTMLZHBSPP", "da": "DAMLZHBSPP"}
MARKETS = tuple(REPORT_TYPE_IDS)
CLEAN_COLUMNS = [
    "interval_start_utc",
    "interval_end_utc",
    "location",
    "location_type",
    "market",
    "price",
]


def _check_market(market: str) -> None:
    if market not in REPORT_TYPE_IDS:
        raise ValueError(f"unknown market {market!r}; expected one of {MARKETS}")


def _gridstatus():
    try:
        import gridstatus
    except ImportError as exc:  # pragma: no cover - depends on the optional extra
        raise RuntimeError("ERCOT downloads need the data extra: pip install -e '.[data]'") from exc
    return gridstatus


def raw_dir(data_dir: Path) -> Path:
    return Path(data_dir) / "raw"


def clean_path(data_dir: Path, market: str, year: int) -> Path:
    return Path(data_dir) / "clean" / f"ercot_{market}_spp_{year}.parquet"


def download_year(
    market: str, year: int, data_dir: Path, *, force: bool = False, timeout: float = 300
) -> Path:
    """Download ERCOT's price zip for one market and year; return the local path.

    A file already on disk with the same ERCOT name (same publish timestamp) is reused unless
    ``force``. When ERCOT re-publishes the current year's file it gets a new name, so the
    newer version is downloaded next to the old one.
    """
    _check_market(market)
    gs = _gridstatus()
    doc = gs.Ercot()._get_document(
        report_type_id=REPORT_TYPE_IDS[market], constructed_name_contains=f"_{year}.zip"
    )
    target_dir = raw_dir(data_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / doc.constructed_name
    if path.exists() and not force:
        return path
    response = requests.get(doc.url, timeout=timeout)
    response.raise_for_status()
    partial = path.with_name(path.name + ".part")
    partial.write_bytes(response.content)
    with zipfile.ZipFile(partial) as zf:
        bad = zf.testzip()
    if bad is not None:
        raise OSError(f"corrupt download for {path.name}: {bad}")
    partial.replace(path)
    return path


def latest_raw_file(data_dir: Path, market: str, year: int) -> Path | None:
    """Newest downloaded zip for a market and year (ERCOT file names sort by publish time)."""
    _check_market(market)
    files = sorted(raw_dir(data_dir).glob(f"*{FILE_PREFIXES[market]}_{year}.zip"))
    return files[-1] if files else None


def parse_year_file(path: Path, market: str) -> pd.DataFrame:
    """Parse one yearly zip into gridstatus's SPP layout (local-time ``Interval Start`` etc.)."""
    _check_market(market)
    gs = _gridstatus()
    with zipfile.ZipFile(path) as zf, zf.open(zf.namelist()[0]) as fh:
        sheets = pd.read_excel(fh, sheet_name=None)
    # The current year's file has empty sheets for months that haven't happened yet.
    df = pd.concat([s for s in sheets.values() if not s.empty], ignore_index=True)
    df = df.rename(columns={"Settlement Point Type": "SettlementPointType"})
    if market == "rt":
        # Some yearly RT files contain one all-empty row; more than one means a layout change.
        empty = df[["Delivery Hour", "Delivery Interval"]].isna().all(axis=1)
        if empty.sum() > 1:
            raise ValueError(f"{path.name}: {int(empty.sum())} empty rows; layout may have changed")
        df = df.loc[~empty].copy()
        df["Delivery Interval"] = df["Delivery Interval"].astype("Int64")
    iso = gs.Ercot()
    df = iso.parse_doc(df)
    market_enum = gs.Markets.REAL_TIME_15_MIN if market == "rt" else gs.Markets.DAY_AHEAD_HOURLY
    return iso._finalize_spp_df(df, market=market_enum)


def normalize(parsed: pd.DataFrame, market: str) -> pd.DataFrame:
    """gridstatus SPP frame -> this project's clean schema (UTC timestamps, one price column)."""
    _check_market(market)
    df = pd.DataFrame(
        {
            "interval_start_utc": pd.to_datetime(parsed["Interval Start"], utc=True),
            "interval_end_utc": pd.to_datetime(parsed["Interval End"], utc=True),
            "location": parsed["Location"].astype(str),
            "location_type": parsed["Location Type"].astype(str),
            "market": market,
            "price": parsed["SPP"].astype(float),
        }
    )
    for col in ("interval_start_utc", "interval_end_utc"):
        df[col] = df[col].dt.as_unit("ns")
    df = df.drop_duplicates()
    return df.sort_values(["location", "interval_start_utc"], kind="stable").reset_index(drop=True)


def ingest_year(market: str, year: int, data_dir: Path, *, force: bool = False) -> pd.DataFrame:
    """Download (if needed), parse, normalize, and store one market-year; return the frame."""
    path = download_year(market, year, data_dir, force=force)
    clean = normalize(parse_year_file(path, market), market)
    out = clean_path(data_dir, market, year)
    out.parent.mkdir(parents=True, exist_ok=True)
    clean.to_parquet(out, index=False)
    return clean


def load_clean(
    data_dir: Path,
    market: str,
    location: str | None = None,
    years: Iterable[int] | None = None,
) -> pd.DataFrame:
    """Read stored clean prices for a market, optionally for one location and some years."""
    _check_market(market)
    folder = Path(data_dir) / "clean"
    if years is None:
        paths = sorted(folder.glob(f"ercot_{market}_spp_*.parquet"))
    else:
        paths = [clean_path(data_dir, market, y) for y in years]
        paths = [p for p in paths if p.exists()]
    if not paths:
        raise FileNotFoundError(f"no clean {market} prices in {folder}; run `batt ingest` first")
    filters = [("location", "==", location)] if location else None
    frames = [pd.read_parquet(p, filters=filters) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    if location and df.empty:
        raise KeyError(f"location {location!r} not found in stored {market} prices")
    return df.sort_values(["location", "interval_start_utc"], kind="stable").reset_index(drop=True)


def load_market(data_dir: Path, location: str, timing: MarketTiming) -> MarketData:
    """Point-in-time RT and DA prices for one settlement point, from the clean store.

    Fundamentals (``batt.data.fundamentals``) and grid conditions (``batt.data.conditions``)
    are attached when they have been ingested.
    """
    from batt.data.conditions import load_conditions
    from batt.data.fundamentals import load_fundamentals

    rt = load_clean(data_dir, "rt", location)
    da = load_clean(data_dir, "da", location)
    market = MarketData.from_price_frames(location, rt, da, timing)
    market = market.with_fundamentals(load_fundamentals(data_dir))
    return market.with_conditions(load_conditions(data_dir))
