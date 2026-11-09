"""Explore the battery arbitrage backtests.

    streamlit run dashboard/app.py

Reads only dashboard/data/, which scripts/export_dashboard_data.py builds from the saved results.
"""

from __future__ import annotations

from datetime import date

import altair as alt
import bundle as data
import pandas as pd
import streamlit as st

REPO = "https://github.com/chensirui3/Battery-arbitrage"
DOCS = f"{REPO}/blob/main/docs"
DEFAULT_DAY = date(2022, 7, 13)
DEFAULT_FOCUS = "mpc_rtd"
README_COMPARISONS = [
    "mpc_oracle_1h",
    "mpc_oracle_spike",
    "mpc_oracle_timing",
    "mpc_spike",
    "mpc_spike_c",
    "mpc_rtd",
    "mpc_gbm",
    "mpc_gbm_c",
]
FORECASTERS = {
    "da": "Day-ahead price",
    "ar1": "DA + latest spread, decaying",
    "gbm": "LightGBM",
    "gbm_f": "LightGBM + ERCOT's forecasts",
    "gbm_c": "LightGBM + grid conditions",
    "qgbm": "Quantile LightGBM",
    "qgbm_f": "Quantile LightGBM + ERCOT's forecasts",
    "qgbm_c": "Quantile LightGBM + grid conditions",
    "analog": "Analog scenarios",
    "analog_rt": "Analog scenarios from the latest spread",
    "spike": "Spike nowcast",
    "spike_f": "Spike nowcast + ERCOT's forecasts",
    "rtd": "ERCOT's look-ahead where $100 from DA",
    "rtd_all": "ERCOT's look-ahead as published",
}
LEADS = ["0-1h", "1-4h", "4-12h", "12-24h", "24-36h"]

# Colors by role, never by rank: the first three slots of a validated categorical palette, a
# diverging pair for gains and losses, and neutral inks. Dark mode has its own steps.
LIGHT = {
    "focus": "#2a78d6",
    "baseline": "#eb6834",
    "oracle": "#1baf7a",
    "gain": "#2a78d6",
    "loss": "#e34948",
    "other": "#898781",
    "ink": "#52514e",
    "muted": "#898781",
    "surface": "#ffffff",
}
DARK = {
    "focus": "#3987e5",
    "baseline": "#d95926",
    "oracle": "#199e70",
    "gain": "#3987e5",
    "loss": "#e66767",
    "other": "#898781",
    "ink": "#c3c2b7",
    "muted": "#898781",
    "surface": "#0e1117",
}
ROLES = {
    "oracle": "Knows future prices",
    "baseline": "The baseline to beat",
    "realistic": "Uses only published data",
}

st.set_page_config(page_title="Battery arbitrage on the Texas grid", layout="wide")


@st.cache_resource
def get_bundle() -> data.Bundle:
    return data.load()


@st.cache_data
def compare(hubs: tuple, baseline: str, policies: tuple, period: str) -> pd.DataFrame:
    since, until = data.PERIODS[period]
    return data.compare(get_bundle(), hubs, baseline, policies, since, until)


def palette() -> dict:
    theme = getattr(st.context, "theme", None)
    return DARK if getattr(theme, "type", None) == "dark" else LIGHT


def money(value: float, digits: int = 1) -> str:
    return f"{value:+,.{digits}f}".replace("-", "\N{MINUS SIGN}")


def hour_axis(hours: float = 24) -> alt.Axis:
    """Clock times on an ordinary day; plain hour counts on a day the clocks change."""
    if hours == 24:
        return alt.Axis(values=list(range(0, 25, 3)), labelExpr="datum.value + ':00'", grid=False)
    return alt.Axis(values=list(range(0, int(hours) + 1, 3)), grid=False)


# --- Charts -----------------------------------------------------------------------------------

# Ranked charts put each row's name above its mark rather than on an axis to the left, so the
# marks keep the full width on a phone as on a desktop.
ROW = 38  # pixels per row: the name, then the mark below it
BELOW = 8  # how far below a row's center its mark sits


def legend(stacked: bool = False) -> alt.Legend:
    """A legend above the chart; ``stacked`` puts one entry per line, for long names."""
    return alt.Legend(title=None, orient="top", labelLimit=420, columns=1 if stacked else 0)


def rows(order: list[str]) -> alt.Y:
    return alt.Y("label:N", sort=order, axis=None)


def row_names(base: alt.Chart, c: dict) -> alt.Chart:
    """Each row's name at the left edge, just above its mark."""
    return base.mark_text(align="left", baseline="bottom", x=0, dy=-1, color=c["ink"]).encode(
        text="label:N"
    )


def capture_chart(table: pd.DataFrame, c: dict) -> alt.LayerChart:
    """Each policy's share of perfect-foresight revenue, ranked, colored by what it knows."""
    domain = list(ROLES.values())
    ranked = list(table.sort_values("capture", ascending=False)["label"])  # layers ignore "-x"
    base = alt.Chart(table).encode(
        y=rows(ranked),
        tooltip=[
            alt.Tooltip("name:N", title="Policy"),
            alt.Tooltip("policy:N", title="Code"),
            alt.Tooltip("capture:Q", title="Capture", format=".0%"),
            alt.Tooltip("usd_kw_year:Q", title="$/kW-year", format=",.1f"),
            alt.Tooltip("description:N", title="Knows"),
        ],
    )
    x = alt.X(
        "capture:Q",
        title="Share of perfect-foresight revenue",
        axis=alt.Axis(format="%"),
        scale=alt.Scale(domain=[0, 1.1]),
    )
    bars = base.mark_bar(size=12, cornerRadiusEnd=4, yOffset=BELOW).encode(
        x=x,
        color=alt.Color(
            "role:N",
            scale=alt.Scale(domain=domain, range=[c["oracle"], c["baseline"], c["other"]]),
            legend=legend(stacked=True),
        ),
    )
    values = base.mark_text(align="left", dx=5, yOffset=BELOW, color=c["ink"]).encode(
        x=x, text=alt.Text("capture:Q", format=".0%")
    )
    return (bars + values + row_names(base, c)).properties(height=alt.Step(ROW))


def forest_chart(table: pd.DataFrame, baseline: str, c: dict) -> alt.LayerChart:
    """Differences against the baseline with 95% CIs; the baseline is the zero line."""
    color = alt.Color(
        "role:N",
        scale=alt.Scale(
            domain=[ROLES["oracle"], ROLES["realistic"]], range=[c["oracle"], c["focus"]]
        ),
        legend=legend(stacked=True),
    )
    tooltip = [
        alt.Tooltip("label:N", title="Policy"),
        alt.Tooltip("mean:Q", title=f"Minus {baseline}, $/kW-year", format="+.1f"),
        alt.Tooltip("low:Q", title="95% CI from", format="+.1f"),
        alt.Tooltip("high:Q", title="95% CI to", format="+.1f"),
        alt.Tooltip("days:Q", title="Days"),
    ]
    base = alt.Chart(table).encode(y=rows(list(table["label"])), tooltip=tooltip)
    x = alt.X("low:Q", title=f"$/kW-year minus {baseline}")
    ranges = base.mark_rule(strokeWidth=2, strokeCap="round", yOffset=BELOW)
    ranges = ranges.encode(x=x, x2="high:Q", color=color)
    points = base.mark_circle(
        size=110, opacity=1, stroke=c["surface"], strokeWidth=2, yOffset=BELOW
    ).encode(x="mean:Q", color=color)
    zero = alt.Chart(pd.DataFrame({"x": [0.0]})).mark_rule(color=c["baseline"], strokeWidth=1.5)
    layers = zero.encode(x="x:Q") + ranges + points + row_names(base, c)
    return layers.properties(height=alt.Step(ROW))


def crosshair_lines(
    long: pd.DataFrame,
    x: str,
    x_title: str | None,
    y_title: str,
    colors: dict,
    step: bool = False,
    clock: dict | None = None,
) -> alt.LayerChart:
    """Lines with a vertical hover rule whose tooltip lists every series at that x.
    ``long`` has columns x, series, value; ``colors`` maps each series name to its color. An x
    in hours since midnight is labeled in the tooltip by ``clock``, its time of day."""
    x_type = "T" if pd.api.types.is_datetime64_any_dtype(long[x]) else "Q"
    x_enc = alt.X(f"{x}:{x_type}", title=x_title)
    if x_type == "Q":
        domain = [0, max(24.0, float(long[x].max()))]
        x_enc = alt.X(
            f"{x}:Q", title=x_title, axis=hour_axis(domain[1]), scale=alt.Scale(domain=domain)
        )
    names = list(colors)
    color = alt.Color(
        "series:N",
        scale=alt.Scale(domain=names, range=[v["color"] for v in colors.values()]),
        legend=legend(stacked=True),
    )
    dash = alt.StrokeDash(
        "series:N",
        scale=alt.Scale(domain=names, range=[v.get("dash", [1, 0]) for v in colors.values()]),
        legend=None,
    )
    lines = alt.Chart(long).mark_line(strokeWidth=2, interpolate="step-after" if step else "linear")
    lines = lines.encode(x=x_enc, y=alt.Y("value:Q", title=y_title), color=color, strokeDash=dash)
    order = list(long["series"].drop_duplicates())
    wide = long.pivot(index=x, columns="series", values="value")[order].reset_index()
    wide.columns = [x, *[f"s{i}" for i in range(len(order))]]  # plain names for Vega-Lite
    nearest = alt.selection_point(nearest=True, on="pointerover", fields=[x], empty=False)
    first = (
        alt.Tooltip(f"{x}:T", title="Day", format="%b %d, %Y")
        if x_type == "T"
        else alt.Tooltip("time:N", title="Time")
    )
    if x_type == "Q":
        wide["time"] = wide[x].map(clock or {})
    rule = (
        alt.Chart(wide)
        .mark_rule(color=palette()["muted"], strokeWidth=1)
        .encode(
            x=f"{x}:{x_type}",
            opacity=alt.condition(nearest, alt.value(0.8), alt.value(0)),
            tooltip=[first]
            + [alt.Tooltip(f"s{i}:Q", title=name, format=",.1f") for i, name in enumerate(order)],
        )
    )
    return alt.layer(lines, rule.add_params(nearest))


def monthly_chart(diff: pd.Series, c: dict) -> alt.Chart:
    """Monthly difference, $/kW: gains and losses in a diverging pair."""
    frame = diff.rename("value").rename_axis("month").reset_index()
    frame["sign"] = frame["value"].map(lambda v: "Gain" if v >= 0 else "Loss")
    return (
        alt.Chart(frame)
        .mark_bar(width={"band": 0.7}, cornerRadiusEnd=2)
        .encode(
            x=alt.X("yearmonth(month):T", title=None),
            y=alt.Y("value:Q", title="$/kW in the month"),
            color=alt.Color(
                "sign:N",
                scale=alt.Scale(domain=["Gain", "Loss"], range=[c["gain"], c["loss"]]),
                legend=legend(),
            ),
            tooltip=[
                alt.Tooltip("yearmonth(month):T", title="Month"),
                alt.Tooltip("value:Q", title="$/kW", format="+.2f"),
            ],
        )
        .properties(height=220)
    )


def forecast_chart(table: pd.DataFrame, c: dict) -> alt.LayerChart:
    """One lead time's CRPS by forecaster, best first; day-ahead prices are the baseline."""
    ranked = list(table.sort_values("crps")["label"])
    base = alt.Chart(table).encode(
        y=rows(ranked),
        tooltip=[
            alt.Tooltip("label:N", title="Forecaster"),
            alt.Tooltip("crps:Q", title="CRPS", format=".1f"),
            alt.Tooltip("mae:Q", title="MAE", format=".1f"),
        ],
    )
    x = alt.X("crps:Q", title="CRPS, $/MWh (lower is better)")
    bars = base.mark_bar(size=12, cornerRadiusEnd=4, yOffset=BELOW).encode(
        x=x,
        color=alt.Color(
            "role:N",
            scale=alt.Scale(
                domain=["Day-ahead prices", "Forecast models"], range=[c["baseline"], c["focus"]]
            ),
            legend=legend(),
        ),
    )
    values = base.mark_text(align="left", dx=5, yOffset=BELOW, color=c["ink"]).encode(
        x=x, text=alt.Text("crps:Q", format=".1f")
    )
    return (bars + values + row_names(base, c)).properties(height=alt.Step(ROW))


# --- Page ---------------------------------------------------------------------------------------

b = get_bundle()
c = palette()
battery = b.manifest["battery"]
day_hub = b.manifest["interval_hub"]
spec = (
    f"{battery['power_mw']:g} MW / {battery['energy_mwh']:g} MWh battery, "
    f"{battery['round_trip_efficiency']:.0%} round-trip efficiency, "
    f"${battery['degradation_cost_per_mwh']:g}/MWh degradation"
)

st.title("Battery arbitrage on the Texas grid")
st.markdown(
    f"How much of a grid battery's arbitrage revenue can a realistic operator capture in ERCOT, "
    f"and where does the rest go? A {spec}, January 2022 to October 2026. Every decision sees "
    f"only what was public at the time. [Code, write-up, and plans]({REPO})."
)
overview, comparisons, over_time, one_day, forecasts, about = st.tabs(
    [
        "Overview",
        "Compare policies",
        "Revenue over time",
        "A day in the market",
        "Forecasts",
        "About",
    ]
)

with overview:
    hub = st.selectbox("Hub", b.hubs, key="overview_hub")
    table = data.levels(b, hub).join(b.policies, how="inner")
    table["role"] = table["group"].map(ROLES)
    table["label"] = [b.label(p) for p in table.index]
    table = table.rename_axis("policy").reset_index()
    by_policy = table.set_index("policy")
    left, middle, right = st.columns(3)
    left.metric("Perfect foresight", f"${by_policy.at[data.ORACLE, 'usd_kw_year']:.1f}/kW-year")
    middle.metric(
        "Re-plans on day-ahead prices",
        f"{by_policy.at[data.BASELINE, 'capture']:.0%}",
        help=f"${by_policy.at[data.BASELINE, 'usd_kw_year']:.1f}/kW-year",
    )
    if "mpc_oracle_1h" in by_policy.index:
        right.metric("Knows only the next hour", f"{by_policy.at['mpc_oracle_1h', 'capture']:.0%}")
    st.altair_chart(capture_chart(table, c), width="stretch")
    tests = compare(
        tuple(b.hubs), data.BASELINE, tuple(README_COMPARISONS), "January 2022 to October 2026"
    )
    tests = tests.set_index("policy")
    st.markdown(
        "**What stands out**\n\n"
        "- Re-planning every 15 minutes on day-ahead prices is the policy to beat. No policy that "
        "uses only published data beat it detectably.\n"
        f"- The missing revenue is in the next few hours, mostly in spikes. Averaged over the "
        f"five hubs, knowing the next hour is worth {money(tests.at['mpc_oracle_1h', 'mean'])}"
        f"/kW-year more than the baseline, and knowing only its spikes "
        f"{money(tests.at['mpc_oracle_spike', 'mean'])}.\n"
        f"- Two attempts were tested in comparisons fixed before running, and failed: a spike "
        f"nowcast ({money(tests.at['mpc_spike', 'mean'])} [{money(tests.at['mpc_spike', 'low'])}, "
        f"{money(tests.at['mpc_spike', 'high'])}]) and ERCOT's own one-hour look-ahead "
        f"({money(tests.at['mpc_rtd', 'mean'])} [{money(tests.at['mpc_rtd', 'low'])}, "
        f"{money(tests.at['mpc_rtd', 'high'])}]). See *Compare policies*.\n"
        "- Both failed the same way: seeing an hour ahead, with day-ahead prices beyond it, the "
        "battery sells into the start of a long spike. See *A day in the market*."
    )

with comparisons:
    st.markdown(
        "Each policy's revenue minus a baseline's, averaged over the chosen hubs day by day. "
        "The 95% confidence intervals come from a moving-block bootstrap over weeks "
        "(2,000 resamples) that draws the same weeks at every hub, since the hubs share weather "
        "and scarcity events. With all five hubs and the whole period, these are the README's "
        "numbers."
    )
    left, right = st.columns(2)
    hubs = left.multiselect("Hubs", b.hubs, default=b.hubs, key="compare_hubs")
    period = right.selectbox("Period", list(data.PERIODS), key="compare_period")
    if not hubs:
        st.info("Choose at least one hub.")
    else:
        options = b.run_at(hubs)
        baseline = left.selectbox(
            "Baseline",
            options,
            index=options.index(data.BASELINE),
            format_func=b.label,
            key="compare_baseline",
        )
        defaults = [p for p in README_COMPARISONS if p in options and p != baseline]
        chosen = right.multiselect(
            "Policies",
            [p for p in options if p != baseline],
            default=defaults,
            format_func=b.label,
            key="compare_policies",
        )
        if len(hubs) < len(b.hubs):
            st.caption(f"{len(options)} policies ran at every chosen hub.")
        if chosen:
            result = compare(tuple(hubs), baseline, tuple(chosen), period)
            result["label"] = [b.label(p) for p in result["policy"]]
            groups = b.policies.loc[result["policy"], "group"].to_numpy()
            result["role"] = [
                ROLES["oracle"] if g == "oracle" else ROLES["realistic"] for g in groups
            ]
            st.altair_chart(forest_chart(result, baseline, c), width="stretch")
            st.dataframe(
                result[["label", "mean", "low", "high", "days"]],
                hide_index=True,
                column_config={
                    "label": "Policy",
                    "mean": st.column_config.NumberColumn(f"Minus {baseline}", format="%+.1f"),
                    "low": st.column_config.NumberColumn("95% CI from", format="%+.1f"),
                    "high": st.column_config.NumberColumn("95% CI to", format="%+.1f"),
                    "days": st.column_config.NumberColumn("Days"),
                },
            )

with over_time:
    left, middle, right = st.columns(3)
    hub = left.selectbox("Hub", b.hubs, key="time_hub")
    options = [p for p in b.run_at([hub]) if p not in (data.BASELINE, data.ORACLE)]
    focus = middle.selectbox(
        "Policy",
        options,
        index=options.index(DEFAULT_FOCUS),
        format_func=b.label,
        key="time_focus",
    )
    with_oracle = right.toggle("Show perfect foresight", key="time_oracle")
    series = [focus, data.BASELINE, *([data.ORACLE] if with_oracle else [])]
    colors = {
        b.label(focus): {"color": c["focus"]},
        b.label(data.BASELINE): {"color": c["baseline"]},
    }
    if with_oracle:
        colors[b.label(data.ORACLE)] = {"color": c["oracle"]}
    cum = data.cumulative(b, hub, series).rename(columns=b.label).rename_axis("date")
    long = cum.reset_index().melt(id_vars="date", var_name="series", value_name="value")
    st.markdown(f"**Cumulative net revenue at {hub}, $/kW**")
    st.altair_chart(
        crosshair_lines(long, "date", None, "$/kW", colors).properties(height=320), width="stretch"
    )
    table = data.by_day(b, hub)
    diff = (table[focus] - table[data.BASELINE]) * b.per_kw
    st.markdown(f"**{b.label(focus)} minus {data.BASELINE}, by month**")
    st.altair_chart(monthly_chart(diff.resample("MS").sum(), c), width="stretch")
    usd, capture = data.by_period(b, hub, series)
    st.markdown(
        "**By year** (the days from RTC+B on apart: ERCOT redesigned its real-time market "
        "on December 5, 2025)"
    )
    left, right = st.columns(2)
    left.caption("$/kW-year")
    left.dataframe(usd.style.format("{:.1f}"), width="stretch")
    right.caption("Capture: share of perfect-foresight revenue")
    right.dataframe(capture.style.format("{:.0%}"), width="stretch")
    worst, best = data.gaps(b, hub, focus)
    note = (
        f" Open one in *A day in the market* (interval data cover {day_hub})."
        if hub == day_hub
        else ""
    )
    st.markdown(
        f"**Days that decide it** ($/kW, {focus} minus {data.BASELINE}).{note}\n\n"
        + "  \n".join(f"{d:%Y-%m-%d}: {money(v, 2)}" for d, v in worst.items())
        + "\n\n"
        + "  \n".join(f"{d:%Y-%m-%d}: {money(v, 2)}" for d, v in best.items())
    )

with one_day:
    st.markdown(
        f"Prices and the battery's stored energy at {day_hub}, interval by interval, in Central "
        "time. Each policy decides 5 minutes before an interval starts and sees settlement "
        "prices only 5 minutes after one ends. ERCOT's look-ahead is shown as it was published "
        "35 minutes before each interval: the furthest interval `mpc_rtd` uses."
    )
    stored = [p for p in b.run_at([day_hub]) if f"soc_{p}" in b.intervals.columns]
    options = [p for p in stored if p not in (data.BASELINE, data.ORACLE)]
    worst, _ = data.gaps(b, day_hub, DEFAULT_FOCUS)
    notable = {f"{d:%Y-%m-%d}: one of {DEFAULT_FOCUS}'s worst days": d.date() for d in worst.index}
    for d, v in data.best_days(b, day_hub, 8).items():
        notable.setdefault(f"{d:%Y-%m-%d}: perfect foresight earned ${v:.2f}/kW", d.date())
    first, last = b.intervals["date"].iloc[0].date(), b.intervals["date"].iloc[-1].date()
    st.session_state.setdefault("day", DEFAULT_DAY)

    def jump() -> None:
        if st.session_state.get("day_jump"):
            st.session_state["day"] = notable[st.session_state["day_jump"]]

    left, middle, right = st.columns(3)
    left.selectbox(
        "Notable days",
        list(notable),
        index=None,
        placeholder="Jump to a day",
        key="day_jump",
        on_change=jump,
    )
    when = middle.date_input("Day", min_value=first, max_value=last, key="day")
    focus = right.selectbox(
        "Policy",
        options,
        index=options.index(DEFAULT_FOCUS),
        format_func=b.label,
        key="day_focus",
    )
    with_oracle = st.toggle("Show perfect foresight", key="day_oracle")
    series = [focus, data.BASELINE, *([data.ORACLE] if with_oracle else [])]
    part = data.day(b, when, series)
    end = part["hour"].iat[-1] + data.INTERVAL_HOURS
    clock = {**dict(zip(part["hour"], part["clock"], strict=True)), end: "24:00"}
    x_title = "Hour (Central time)"
    if end != 24:
        x_title = f"Hours since midnight: the clocks changed this day, which had {end:g} hours"
    closing = part.iloc[[-1]].assign(hour=end)
    prices = pd.concat([part, closing])[["hour", "rt", "da", "lookahead"]]
    names = {
        "rt": "Real-time price",
        "da": "Day-ahead price",
        "lookahead": "ERCOT's look-ahead, 35 min before",
    }
    long = prices.melt(id_vars="hour", var_name="series", value_name="value")
    long["series"] = long["series"].map(names)
    price_colors = {
        names["rt"]: {"color": c["ink"]},
        names["da"]: {"color": c["muted"]},
        names["lookahead"]: {"color": c["muted"], "dash": [5, 3]},
    }
    st.markdown("**Prices, $/MWh**")
    st.altair_chart(
        crosshair_lines(
            long, "hour", x_title, "$/MWh", price_colors, step=True, clock=clock
        ).properties(height=280),
        width="stretch",
    )
    energy = []
    for policy in series:
        hours = [*part["hour"], end]
        level = [*part[f"start_{policy}"], part[f"end_{policy}"].iat[-1]]
        energy.append(pd.DataFrame({"hour": hours, "series": b.label(policy), "value": level}))
    roles = {focus: c["focus"], data.BASELINE: c["baseline"], data.ORACLE: c["oracle"]}
    energy_colors = {b.label(p): {"color": roles[p]} for p in series}
    st.markdown("**Energy stored, MWh**")
    st.altair_chart(
        crosshair_lines(
            pd.concat(energy), "hour", x_title, "MWh", energy_colors, clock=clock
        ).properties(height=220),
        width="stretch",
    )
    revenue = data.by_day(b, day_hub).loc[pd.Timestamp(when), series]
    columns = st.columns(len(series))
    for column, policy in zip(columns, series, strict=True):
        delta = None
        if policy != data.BASELINE:  # an ASCII sign, which st.metric reads for the arrow
            delta = f"{(revenue[policy] - revenue[data.BASELINE]) / 1e3:+,.0f}k vs {data.BASELINE}"
        column.metric(f"{b.label(policy)}, that day", f"${revenue[policy] / 1e3:,.0f}k", delta)

with forecasts:
    st.markdown(
        f"Out-of-sample forecasts of the real-time price at {day_hub}, issued every hour and "
        "refit monthly exactly as inside the backtest. CRPS scores a whole predicted "
        "distribution in $/MWh, and equals the mean absolute error for a single-value forecast, "
        "so every forecaster sits on one scale."
    )
    scores = b.forecasts[b.forecasts["forecaster"].isin(FORECASTERS)].copy()
    scores["label"] = scores["forecaster"].map(FORECASTERS)
    scores["role"] = [
        "Day-ahead prices" if f == "da" else "Forecast models" for f in scores["forecaster"]
    ]
    lead = st.segmented_control("Lead time", LEADS, default=LEADS[0], key="forecast_lead")
    if lead:
        st.altair_chart(forecast_chart(scores[scores["lead"] == lead], c), width="stretch")
    grid = scores.pivot(index="label", columns="lead", values="crps")[LEADS].rename_axis(
        "Forecaster"
    )
    st.dataframe(grid.style.format("{:.1f}"), width="stretch")
    st.markdown(
        "**Spike nowcast skill** against a prior that sees only what day-ahead prices knew. A "
        "spike is a real-time price more than $100/MWh from day-ahead; `spike_c` also sees "
        "real-time grid conditions."
    )
    side = st.segmented_control("Side", ["up", "down"], default="up", key="nowcast_side")
    if side:
        skill = b.nowcasts[b.nowcasts["side"] == side]
        shown = skill.pivot(
            index="period", columns="forecaster", values=["auc", "avg_precision", "brier_skill"]
        )
        shown.columns = [f"{m} ({f})" for m, f in shown.columns]
        order = ["all", *sorted(p for p in shown.index if p != "all")]
        st.dataframe(shown.loc[order].style.format("{:.3f}", na_rep="none"), width="stretch")

with about:
    st.markdown(
        f"""
**What the backtest assumes.** A price-taking {spec}, trading real-time energy only. Every
value is stored with the time it was published, and each decision, and each model fit, sees only
what was public by then: RT prices 5 minutes after their interval ends, and day-ahead prices at
13:30 the day before. Leakage tests replace everything published after a cutoff with noise and
check that no earlier decision changes.

**Tests fixed before running.** The spike nowcast and the look-ahead each had their open choices
made on July to December 2021, before the test period. Their tests were then written down and
run once on 2022–2026: [spike nowcasting]({DOCS}/spike-nowcasting-plan.md),
[grid conditions]({DOCS}/grid-conditions-plan.md). The [write-up]({DOCS}/writeup.md) tells the
story in one read.

**This app** reads a compact export of the saved backtests (`dashboard/data/`, from
`scripts/export_dashboard_data.py`): daily net revenue at the five hubs, and prices and stored
energy at {day_hub} every 15 minutes. Its confidence intervals repeat the project's own
computation, and the tests check that they agree. Exported {b.manifest["exported"]}.
"""
    )
    with st.expander("Backtests behind each hub's numbers"):
        sources = b.manifest["sources"]
        st.dataframe(
            pd.DataFrame(
                [
                    (hub, policy, run)
                    for hub, runs in sources.items()
                    for policy, run in runs.items()
                ],
                columns=["Hub", "Policy", "Backtest folder"],
            ),
            hide_index=True,
            width="stretch",
        )
