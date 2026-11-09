"""Draw the README figures from saved backtests and grid conditions.

    python scripts/make_figures.py

Reads the backtests in results/ (January 2022 to October 2026 at the five trading hubs, as
`batt backtest` writes them; the policy lists are in the plans in docs/) and ERCOT's look-ahead
in data/conditions/. Writes each figure twice to docs/figures/, for light and dark page themes.
Needs matplotlib: pip install -e ".[figures]".
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Patch, Rectangle

from batt.data.conditions import LookAhead
from batt.evaluate.pool import pool_runs

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
CONDITIONS = ROOT / "data" / "conditions"
OUT = ROOT / "docs" / "figures"
START, END = "2022-01-01", "2026-11-01"
HUBS = ["HB_NORTH", "HB_WEST", "HB_SOUTH", "HB_HOUSTON", "HB_PAN"]
TIMEZONE = "US/Central"
MINUTE_NS = 60 * 10**9

DPI = 200
SCALE = 2  # image pixels per screen pixel: the PNG is shown at half its pixel size
PX = SCALE * 72 / DPI  # points per screen pixel
WIDTH = 8.4  # inches: 1,680 pixels, shown at up to 840
MARGIN = 0.02  # left margin of the title, legend, and labels, as a share of the width
FONT, SMALL = 13 * PX, 12 * PX
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"]

# Chart tokens: the first three slots of a validated categorical palette, stepped per theme,
# by role as in the dashboard: oracles aqua, the policy studied blue, the baseline orange.
THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "ink": "#0b0b0b",
        "ink2": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "blue": "#2a78d6",
        "orange": "#eb6834",
        "aqua": "#1baf7a",
        "gray": "#898781",
    },
    "dark": {
        "surface": "#1a1a19",
        "ink": "#f0efec",
        "ink2": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "blue": "#3987e5",
        "orange": "#d95926",
        "aqua": "#199e70",
        "gray": "#898781",
    },
}

# Policies in plain words. The code names follow in parentheses on the figures.
CAPTURE = {
    "perfect_rt": "Perfect foresight",
    "mpc_oracle_4h": "Knows the next 4 hours",
    "mpc_oracle_1h": "Knows the next hour",
    "mpc_oracle_spike": "Knows the next hour's spikes",
    "mpc_oracle_timing": "Knows when they come",
    "mpc_analog": "Analog scenarios, averaged",
    "mpc_da": "Re-plans on day-ahead prices",
    "mpc_spike": "Spike nowcast",
    "mpc_ar1": "Latest spread, decaying",
    "mpc_rtd": "ERCOT's one-hour look-ahead",
    "da_informed": "Plans once a day",
    "smpc_analog": "Two-stage stochastic MPC",
    "mpc_gbm": "LightGBM price forecast",
    "sdp": "Stochastic dynamic programming",
    "naive": "Last 30 days' average shape",
}
ORACLES = {"perfect_rt", "mpc_oracle_4h", "mpc_oracle_1h", "mpc_oracle_spike", "mpc_oracle_timing"}
BASELINE = "mpc_da"
POOLED = {
    "mpc_oracle_1h": "Knows the next hour",
    "mpc_oracle_spike": "Knows the next hour's spikes",
    "mpc_oracle_timing": "Knows when they come",
    "mpc_spike": "Spike nowcast: tested",
    "mpc_spike_c": "... plus grid conditions",
    "mpc_rtd": "ERCOT's look-ahead: tested",
    "mpc_gbm": "LightGBM price forecast",
    "mpc_gbm_c": "... plus grid conditions",
}
DAY, DAY_HUB, DAY_POLICY = "2022-07-13", "HB_NORTH", "mpc_rtd"
DAY_HOURS = (10.0, 20.0)
LOOKAHEAD_LEAD = 35 * MINUTE_NS  # a decision sees the third interval ahead this long before


# --- Data -----------------------------------------------------------------------------------


def newest_run(hub: str, policy: str) -> Path:
    """The most recently written backtest of ``hub`` over the period that ran ``policy``."""
    found = sorted(
        RESULTS.glob(f"{hub}_{START}_{END}_*"), key=lambda p: p.stat().st_mtime, reverse=True
    )
    for run in found:
        if all((run / "ledgers" / f"{name}.parquet").exists() for name in (policy, BASELINE)):
            return run
    raise FileNotFoundError(f"no backtest of {policy} at {hub} from {START} to {END}")


def capture_table() -> pd.DataFrame:
    """Each policy's share of perfect-foresight revenue at HB_NORTH, lowest first."""
    rows = {}
    for policy in CAPTURE:
        summary = pd.read_csv(newest_run("HB_NORTH", policy) / "summary.csv", index_col="policy")
        rows[policy] = summary.loc[policy, ["capture_rate", "usd_per_kw_year"]]
    return pd.DataFrame(rows).T.astype(float).sort_values("capture_rate")


def pooled_table() -> pd.DataFrame:
    """Each policy minus mpc_da, $/kW-year, averaged over the five hubs, with 95% CIs."""
    rows = {}
    for policy in POOLED:
        runs = {hub: newest_run(hub, policy) for hub in HUBS}
        _, differences = pool_runs(runs, BASELINE, [policy])
        rows[policy] = differences.loc[(policy, "pooled")]
    return pd.DataFrame(rows).T.astype(float)


def day_table() -> tuple[pd.DataFrame, dict[str, float]]:
    """July 13, 2022 at HB_NORTH, interval by interval, and each policy's revenue that day."""
    run = newest_run(DAY_HUB, DAY_POLICY)
    day = {}
    for policy in (BASELINE, DAY_POLICY):
        ledger = pd.read_parquet(run / "ledgers" / f"{policy}.parquet")
        day[policy] = ledger[ledger["local_date"] == pd.Timestamp(DAY)].set_index(
            "interval_start_utc"
        )
    starts = day[BASELINE].index
    month = pd.Timestamp(DAY).to_period("M")
    rtd = pd.read_parquet(
        CONDITIONS / f"rtd_{month.start_time:%Y-%m-%d}_{(month + 1).start_time:%Y-%m-%d}.parquet"
    )
    begin = starts.asi8
    seen = LookAhead.from_frame(rtd).mean_price(
        begin - LOOKAHEAD_LEAD, begin, begin + 15 * MINUTE_NS
    )
    local = starts.tz_convert(TIMEZONE)
    frame = pd.DataFrame(
        {
            "hour": local.hour + local.minute / 60,
            "rt": day[BASELINE]["rt_price"].to_numpy(),
            "da": day[BASELINE]["da_price"].to_numpy(),
            "lookahead": seen,
            **{f"soc_start_{p}": day[p]["soc_start_mwh"].to_numpy() for p in day},
            **{f"soc_end_{p}": day[p]["soc_end_mwh"].to_numpy() for p in day},
        }
    )
    return frame, {p: float(day[p]["revenue_net"].sum()) for p in day}


# --- Drawing --------------------------------------------------------------------------------


def signed(value: float, digits: int = 1) -> str:
    return f"{value:+.{digits}f}".replace("-", "\N{MINUS SIGN}")


def new_figure(t: dict, height: float) -> Figure:
    fig = Figure(figsize=(WIDTH, height), dpi=DPI, facecolor=t["surface"])
    FigureCanvasAgg(fig)
    return fig


def inches(fig: Figure, value: float) -> float:
    """A vertical distance in inches, as a share of the figure's height."""
    return value / fig.get_figheight()


def heading(fig: Figure, t: dict, title: str, subtitle: str) -> float:
    """Title and subtitle at the top left. Returns the height below them."""
    y = 1 - inches(fig, 0.16)
    fig.text(MARGIN, y, title, color=t["ink"], fontsize=16 * PX, weight="semibold", va="top")
    y -= inches(fig, 0.34)
    fig.text(MARGIN, y, subtitle, color=t["ink2"], fontsize=FONT, va="top", linespacing=1.5)
    return y - inches(fig, 0.24 * (subtitle.count("\n") + 1) + 0.14)


def legend(owner, t: dict, handles, anchor: tuple[float, float]) -> None:
    """One row of legend entries; ``anchor`` is its top-left corner in the owner's frame."""
    owner.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=anchor,
        ncol=len(handles),
        frameon=False,
        fontsize=FONT,
        labelcolor=t["ink2"],
        handlelength=1.2,
        handleheight=0.8,
        columnspacing=1.6,
        borderaxespad=0,
        borderpad=0,
    )


def style(ax, t: dict, grid: str) -> None:
    ax.set_facecolor(t["surface"])
    for side in ax.spines.values():
        side.set_visible(False)
    ax.tick_params(length=0, labelsize=FONT, labelcolor=t["ink2"], pad=6 * PX)
    ax.tick_params(axis="x" if grid == "x" else "y", labelsize=SMALL, labelcolor=t["muted"])
    ax.grid(axis=grid, color=t["grid"], linewidth=PX)
    ax.set_axisbelow(True)


def fit_left(ax) -> None:
    """Move the axes' left edge so its widest y tick label starts at the left margin."""
    fig = ax.figure
    renderer = fig.canvas.get_renderer()
    widest = max(label.get_window_extent(renderer).width for label in ax.get_yticklabels())
    x0, y0, width, height = ax.get_position().bounds
    left = MARGIN + (widest + 6 * SCALE) / (fig.get_figwidth() * DPI)
    ax.set_position((left, y0, x0 + width - left, height))


def rounded_barh(ax, y: float, width: float, height: float, color: str) -> None:
    """A horizontal bar, square at the baseline with a 4-pixel rounded end."""
    (x0, y0), (x1, y1) = ax.transData.inverted().transform([(0, 0), (1, 1)])
    per_px_x, per_px_y = abs(x1 - x0), abs(y1 - y0)  # data units per image pixel
    radius = min(4 * SCALE * per_px_x, width / 2)
    ax.add_patch(Rectangle((0, y - height / 2), width - radius, height, color=color, lw=0))
    ax.add_patch(
        FancyBboxPatch(
            (width - 2 * radius, y - height / 2),
            2 * radius,
            height,
            boxstyle=f"round,pad=0,rounding_size={radius}",
            mutation_aspect=per_px_y / per_px_x,
            color=color,
            lw=0,
        )
    )


def draw_capture(table: pd.DataFrame, t: dict, path: Path) -> None:
    n = len(table)
    fig = new_figure(t, 0.34 * n + 1.75)
    below = heading(
        fig,
        t,
        "Share of perfect-foresight revenue captured",
        "100 MW / 200 MWh battery at ERCOT's North Hub, January 2022 to October 2026",
    )
    legend(
        fig,
        t,
        [
            Patch(color=t["aqua"], label="Knows future prices"),
            Patch(color=t["orange"], label="The baseline to beat"),
            Patch(color=t["gray"], label="Uses only published data"),
        ],
        (MARGIN, below),
    )
    bottom, top = inches(fig, 0.42), below - inches(fig, 0.42)
    ax = fig.add_axes((0.3, bottom, 0.62, top - bottom))
    style(ax, t, "x")
    ax.set_xlim(0, 106)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_yticks(range(n), [f"{CAPTURE[p]} ({p})" for p in table.index])
    ax.set_xticks(range(0, 101, 20), [f"{x}%" for x in range(0, 101, 20)])
    fit_left(ax)
    for i, (policy, row) in enumerate(table.iterrows()):
        color = t["aqua"] if policy in ORACLES else t["gray"]
        color = t["orange"] if policy == BASELINE else color
        share = 100 * row["capture_rate"]
        rounded_barh(ax, i, share, 0.56, color)
        ax.text(share + 1.2, i, f"{share:.0f}%", va="center", color=t["ink2"], fontsize=FONT)
    fig.savefig(path, facecolor=t["surface"])


def draw_pooled(table: pd.DataFrame, t: dict, path: Path) -> None:
    rows, y = {}, 0.0
    for group in (ORACLES, set(table.index) - ORACLES):
        for policy in (p for p in table.index if p in group):
            rows[policy] = y
            y += 1
        y += 0.6
    fig = new_figure(t, 0.4 * len(table) + 2.45)
    below = heading(
        fig,
        t,
        "What knowing the next hour is worth, and what the attempts earned",
        "Revenue minus re-planning on day-ahead prices (mpc_da), $/kW-year, with 95% CIs.\n"
        "Averaged over ERCOT's five trading hubs, January 2022 to October 2026.",
    )
    legend(
        fig,
        t,
        [
            Line2D(
                [],
                [],
                color=t["aqua"],
                marker="o",
                lw=2 * PX,
                ms=8 * PX,
                label="Knows future prices",
            ),
            Line2D(
                [],
                [],
                color=t["blue"],
                marker="o",
                lw=2 * PX,
                ms=8 * PX,
                label="Uses only published data",
            ),
            Line2D([], [], color=t["orange"], lw=2 * PX, label="The baseline: zero"),
        ],
        (MARGIN, below),
    )
    bottom, top = inches(fig, 0.42), below - inches(fig, 0.42)
    ax = fig.add_axes((0.3, bottom, 0.45, top - bottom))
    style(ax, t, "x")
    lo, hi = table["low"].min(), table["high"].max()
    ax.set_xlim(np.floor(lo / 5) * 5 - 2, np.ceil(hi / 5) * 5 + 2)
    ax.set_ylim(y - 1.0, -0.6)  # the last row sits at y - 1.6
    ax.set_yticks(list(rows.values()), [f"{POOLED[p]} ({p})" for p in rows])
    ax.xaxis.set_major_formatter(lambda x, _: "0" if x == 0 else signed(x, 0))
    fit_left(ax)
    ax.axvline(0, color=t["orange"], linewidth=1.5 * PX, zorder=1)
    for policy, yy in rows.items():
        row = table.loc[policy]
        color = t["aqua"] if policy in ORACLES else t["blue"]
        ax.hlines(
            yy, row["low"], row["high"], color=color, linewidth=2 * PX, capstyle="round", zorder=2
        )
        ax.plot(row["mean"], yy, "o", ms=10 * PX, mfc=color, mec=t["surface"], mew=2 * PX, zorder=3)
        ax.text(
            1.04,
            yy,
            f"{signed(row['mean'])}  [{signed(row['low'])}, {signed(row['high'])}]",
            transform=ax.get_yaxis_transform(),
            va="center",
            color=t["ink2"],
            fontsize=FONT,
        )
    fig.savefig(path, facecolor=t["surface"])


def draw_day(frame: pd.DataFrame, revenue: dict[str, float], t: dict, path: Path) -> None:
    lo, hi = DAY_HOURS
    part = frame[(frame["hour"] >= lo) & (frame["hour"] < hi)]
    edges = np.append(part["hour"].to_numpy(), part["hour"].iat[-1] + 0.25)
    fig = new_figure(t, 6.6)
    below = heading(
        fig,
        t,
        "July 13, 2022 at ERCOT's North Hub",
        "ERCOT's look-ahead saw the spike, but only 45 minutes ahead. Beyond that the plan used\n"
        "day-ahead prices, so mpc_rtd sold early, while mpc_da kept its energy for their peak.",
    )
    left, width = 0.11, 0.86
    split, gap = inches(fig, 1.95), inches(fig, 0.75)
    prices = fig.add_axes((left, split + gap, width, below - inches(fig, 0.42) - split - gap))
    energy = fig.add_axes(
        (left, inches(fig, 0.42), width, split - inches(fig, 0.42)), sharex=prices
    )
    for ax in (prices, energy):
        style(ax, t, "y")
        ax.set_xlim(lo, hi)

    def steps(values: pd.Series) -> np.ndarray:
        return np.append(values.to_numpy(), values.iat[-1])

    prices.step(edges, steps(part["da"]), where="post", color=t["gray"], lw=2 * PX)
    prices.step(
        edges, steps(part["lookahead"]), where="post", color=t["gray"], lw=2 * PX, ls=(0, (4, 2))
    )
    prices.step(edges, steps(part["rt"]), where="post", color=t["ink2"], lw=2 * PX)
    prices.set_ylim(0, 6000)
    prices.set_yticks(range(0, 6001, 2000), [f"${y:,}" for y in range(0, 6001, 2000)])
    prices.tick_params(axis="x", labelbottom=False)
    legend(
        prices,
        t,
        [
            Line2D([], [], color=t["ink2"], lw=2 * PX, label="Real-time price, $/MWh"),
            Line2D(
                [],
                [],
                color=t["gray"],
                lw=2 * PX,
                ls=(0, (4, 2)),
                label="ERCOT's look-ahead, 35 min before",
            ),
            Line2D([], [], color=t["gray"], lw=2 * PX, label="Day-ahead price"),
        ],
        (0, 1 + inches(fig, 0.36) / prices.get_position().height),
    )

    for policy, color in ((BASELINE, t["orange"]), (DAY_POLICY, t["blue"])):
        level = np.append(part[f"soc_start_{policy}"].to_numpy(), part[f"soc_end_{policy}"].iat[-1])
        energy.plot(edges, level, color=color, lw=2 * PX, solid_joinstyle="round")
    energy.set_ylim(0, 215)
    energy.set_yticks([0, 100, 200], ["0", "100", "200 MWh"])
    hours = np.arange(lo, hi + 0.1, 2)
    energy.set_xticks(hours, [f"{int(h)}:00" for h in hours])
    energy.tick_params(axis="x", labelsize=SMALL, labelcolor=t["muted"])
    legend(
        energy,
        t,
        [
            Line2D(
                [],
                [],
                color=t["orange"],
                lw=2 * PX,
                label=f"Energy stored by {BASELINE}: ${revenue[BASELINE] / 1e6:.2f}M that day",
            ),
            Line2D(
                [],
                [],
                color=t["blue"],
                lw=2 * PX,
                label=f"{DAY_POLICY}: ${revenue[DAY_POLICY] / 1e6:.2f}M",
            ),
        ],
        (0, 1 + inches(fig, 0.36) / energy.get_position().height),
    )
    fig.savefig(path, facecolor=t["surface"])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    capture, pooled = capture_table(), pooled_table()
    frame, revenue = day_table()
    for name, t in THEMES.items():
        draw_capture(capture, t, OUT / f"capture-{name}.png")
        draw_pooled(pooled, t, OUT / f"next-hour-{name}.png")
        draw_day(frame, revenue, t, OUT / f"july-13-2022-{name}.png")
    print(capture.round(3).to_string())
    print(pooled.round(1).to_string())
    print({p: round(v) for p, v in revenue.items()})
    print(f"wrote {len(THEMES) * 3} figures to {OUT}")


if __name__ == "__main__":
    main()
