# Spike nowcasting

Part of [Battery arbitrage on the Texas grid](../README.md).

Perfect knowledge of the next hour's RT prices is worth $20/kW-year at `HB_NORTH`. Can a
model trained to predict spikes in the next hour, from data public at decision time, earn
part of that? The design and the test were written down before any spike model ran on
2022–2026 ([plan and dated outcomes](spike-nowcasting-plan.md)).

**Design.**

- A spike is an interval whose RT price is more than $100/MWh above its DA price (an up-spike)
  or below it (a down-spike), a threshold fixed in advance. At `HB_NORTH`, 1.0% of intervals
  from 2022 to October 2026 were up-spikes and 1.5% down-spikes.
- One LightGBM classifier per side gives the probability for each interval of the next hour.
  It uses `gbm`'s 21 features plus the share of spikes among the intervals published in the
  last 24 hours and 30 days. It is refit monthly on two years, like `gbm`. A spike's typical
  size is its average in the training window, winsorized at $1,000/MWh.
- The one open choice was how to turn probabilities into the next hour's prices. It was made
  on July to December 2021, before the backtest period, at all five hubs, by revenue, among
  three candidates. Averaged over the hubs, against `mpc_da`:
  - `raw`, DA plus each side's probability times its typical size: −$5.0/kW-year.
  - `news`, the same with each probability minus that of a model that sees only what the DA
    prices knew: −$4.4.
  - `regime`, `raw` but only on a side whose latest published price is itself a spike: −$0.0.
    Selected.

  The other two moved prices on every interval, also when nothing was happening.
- `mpc_spike` re-plans every 15 minutes with that forecast for the next hour and DA prices after
  it.
- Two benchmarks show what spikes are worth. `mpc_oracle_spike` knows the next hour's RT
  prices where they are spikes. `mpc_oracle_timing` knows which of the next hour's intervals
  spike but assumes a typical size, like a perfect classifier.

**Test.** `mpc_spike` minus `mpc_da` from January 2022 to October 2026. Daily revenue is
averaged over the five trading hubs, and the CIs resample the same weeks at every hub. It counts
as confirmed only if the 95% CI is above zero. It was run once.

**Result: not confirmed.** −$1.3/kW-year [−4.2, +1.3].

| $/kW-year, 2022 to Oct 2026 | `HB_NORTH` | `HB_WEST` | `HB_SOUTH` | `HB_HOUSTON` | `HB_PAN` | Average |
|---|---:|---:|---:|---:|---:|---:|
| `perfect_rt` | 88.0 | 95.9 | 81.5 | 91.4 | 93.1 | 90.0 |
| `mpc_oracle_1h` | 72.3 | 78.9 | 67.0 | 75.5 | 76.1 | 73.9 |
| `mpc_oracle_spike` | 67.8 | 73.0 | 62.2 | 70.5 | 68.3 | 68.4 |
| `mpc_oracle_timing` | 57.8 | 63.6 | 54.6 | 60.2 | 58.0 | 58.8 |
| `mpc_da` | 52.3 | 56.9 | 47.0 | 52.8 | 53.3 | 52.5 |
| `mpc_spike` | 50.5 | 56.0 | 46.2 | 50.4 | 52.9 | 51.2 |
| `mpc_spike` − `mpc_da` | −1.8 | −0.9 | −0.8 | −2.4 | −0.4 | −1.3 |

No single hub's 95% CI excludes zero either.

| Minus `mpc_da`, averaged over the hubs (95% CI) | 2022 to Oct 2026 | From 2024 |
|---|---|---|
| `mpc_oracle_1h` | +21.5 [+14.5, +29.9] | +8.0 [+6.4, +9.6] |
| `mpc_oracle_spike` | +15.9 [+9.0, +24.2] | +3.8 [+2.3, +5.5] |
| `mpc_oracle_timing` | +6.4 [+1.6, +11.6] | +1.4 [0.0, +2.7] |
| `mpc_spike` | −1.3 [−4.2, +1.3] | +0.0 [−1.1, +1.2] |

By year, `mpc_spike` minus `mpc_da` was −9.3 [−21.6, −0.4] in 2022. After that it was +3.1,
+1.1, −0.9, and −0.2, none distinguishable from zero.

**The classifier itself works.** It was scored on every decision at `HB_NORTH` against a prior
that sees calendar features, DA prices, and 7-to-30-day spread and spike rates, but no recent
RT prices (`batt nowcast`):

| | Up-spikes | Down-spikes |
|---|---:|---:|
| ROC AUC (prior) | 0.961 (0.863) | 0.998 (0.994) |
| Average precision (prior) | 0.34 (0.06) | 0.90 (0.65) |
| Brier skill score against the prior | 0.21 | 0.45 |

Down-spikes are easy even for the prior. 99.7% of them came when the DA price was above
$100/MWh, and 60% of the intervals priced above $150 day-ahead were down-spikes. As a price
forecast (`batt forecast`), the nowcast's first-hour MAE is $19.6/MWh, against $21.6 for DA
prices and $19.3 for `gbm`.

**Why the skill did not become revenue.** These diagnostics came after the test, and nothing
was changed because of them.

- **Even a perfect classifier earns little with typical sizes, except in 2023.** The timing
  oracle gained +26.1 in 2023, but +0.7 in 2022 and +1.4 from 2024 on. Knowing each spike's
  real size as well is worth far more: +15.9 against +6.4 over the period.
- **A few extreme days decide the result.** `mpc_spike` acts on a spike already under way and
  keeps DA prices beyond the next hour, so it sells into the spike now. `mpc_da` keeps its
  energy for the DA peak. Averaged over the hubs, `mpc_spike`'s 10 best days gained $1.09M
  and its 10 worst lost $1.25M. Over the whole period it lost $0.62M.
  - On December 23, 2022 (Winter Storm Elliott), RT at `HB_NORTH` reached $2,400–3,900/MWh at
    06:00–07:00. DA's morning peak came after, and by its end RT had fallen to $77. Selling
    early paid.
  - On July 13, 2022, `mpc_spike` sold most of the battery at $1,190–2,530/MWh between 13:00
    and 14:15. RT then stayed above $4,800 from 14:30 to 18:15, through DA's peak, when
    `mpc_da` sold everything. That day `mpc_spike` earned $0.41M and `mpc_da` $0.94M.
- **During a spike, the forecast sits far below the price that follows.** At `HB_NORTH` the
  up-spike side was active at 1.0% of decisions. Averaged by year, it added $50–171/MWh to DA,
  while the realized spread averaged $155–497/MWh. On the worst days it even charged during
  spikes, for example at about $3,000/MWh on February 24, 2022, when its forecast was about
  $300.

Further variants at `HB_NORTH`, minus `mpc_da`, $/kW-year (95% CI):

| Variant | 2022 to Oct 2026 |
|---|---|
| `mpc_spike_raw`, `mpc_spike_news`: the two mappings not selected | −2.8 [−7.0, +0.4], −4.6 [−9.3, −1.1] |
| `mpc_spike_f`: classifiers also see ERCOT's load, wind, and solar forecasts | −1.1 [−4.2, +1.9] |
| `mpc_spike_50`, `mpc_spike_250`: spikes beyond $50 or $250 | −2.2 [−4.9, +0.4], +0.6 [−2.5, +3.4] |
| `mpc_gbm_1h`: the LightGBM price forecast for the next hour, DA prices after | −4.1 [−6.6, −1.7] |

`mpc_gbm_1h` separates the spike target from the one-hour horizon. LightGBM's ordinary price
forecast, used for the next hour only, still loses $4.1.

The spike oracle by threshold, minus `mpc_da` at `HB_NORTH`, is +17.6 at $25, +16.4 at $50,
+15.5 at $100, +14.2 at $250, +12.0 at $500, and +9.6 at $1,000. The one-hour oracle is +19.9,
so spreads beyond $1,000/MWh alone carry about half its value.

To reproduce, run `batt backtest --location <hub> --policies <list>` at each hub, then
`batt nowcast`. `batt pool --policies mpc_spike` averages the hubs day by day and prints the
CIs (`--since 2024-01-01` for the later cut). The exact policy lists are in the
[plan](spike-nowcasting-plan.md).
