# Spike nowcasting: analysis plan

Written on 2026-11-07, before any spike model was run on the backtest period (January 2022 to
October 2026). The sections "Fixed design", "Development stage", and "Test" are not edited
after this date. Outcomes are added below them, dated.

## Question

Perfect knowledge of the next hour's RT prices is worth $20/kW-year over re-planning on DA
prices (`mpc_oracle_1h` vs `mpc_da` at `HB_NORTH`, 2022-2026). Can a model trained to predict
price spikes in the next hour, from data public at decision time, earn part of that?

## Fixed design

- **Spike.** An interval's RT price is more than $100/MWh above its DA price (up-spike) or more
  than $100/MWh below it (down-spike). The threshold was chosen as a round number before
  looking at any backtest-period data. The only data examined was 2021, which is training
  history: without Winter Storm Uri (February 10-20), 1.0% of 2021's intervals were up-spikes
  and 0.6% were down-spikes. Of the up-spikes, 37% were still spikes two intervals later, the
  delay at which the policy first sees a price.
- **Nowcast.** For each of the four intervals of the next hour, one LightGBM binary classifier
  per side gives the spike probability. Settings are the same as the `gbm` forecaster
  (`objective: binary`, 300 rounds, learning rate 0.05, 31 leaves, at least 200 examples per
  leaf). Refit monthly on the 730 days of data public at the time. Every decision in the
  window is an example, paired with each interval of the hour it starts.
- **Features.** The 21 price features of `gbm`, plus the share of spikes, up and down, among
  the intervals published in the last 24 hours and 30 days. `mpc_spike_f` adds `gbm_f`'s ten
  ERCOT fundamentals features.
- **Spike size.** The average up- and down-spike size in the training window, winsorized at
  $1,000/MWh like `gbm`'s labels.
- **Policy.** `mpc_spike` re-plans every 15 minutes, as `mpc_da` does. It uses the nowcast's
  price for the next hour and DA prices after that, as `mpc_oracle_1h` does with realized
  prices.
- **Benchmarks.**
  - `mpc_oracle_spike` knows the next hour's realized RT price wherever it is a spike, and
    uses DA prices elsewhere.
  - `mpc_oracle_timing` knows which of the next hour's intervals spike, but assumes the
    typical spike size: a perfect classifier.

## Development stage

**Why it exists.** A smoke run on December 2021, before this plan was written, was meant to
check runtime. It showed two problems.

- The first design moved DA prices only by "news": each probability minus that of a prior
  classifier that sees roughly what the DA prices knew. It lost to `mpc_da` ($12.1 vs
  $16.1/kW-year for that month). Two separately fitted models never quite agree, and their
  difference times a ~$600 spike size moved the next hour's prices by several dollars even
  when no spike was coming.
- The timing benchmark, as first defined, subtracted the same prior on every interval. So
  even with perfect timing it lost ($10.7). It is now defined without the prior (above).

The other numbers from that month, all in $/kW-year:

| Policy | Dec 2021 |
|---|---:|
| `mpc_oracle_1h` | 24.8 |
| `mpc_oracle_spike` | 18.0 |
| `mpc_gbm_1h` (LightGBM for the next hour only, DA after) | 16.7 |
| raw mapping (below) | 16.4 |
| `mpc_da` | 16.1 |

December 2021 is before the backtest period and is part of the development period.

**Candidates.** Three ways to turn probabilities into the next hour's price (`adjust`):

| `adjust` | Next-hour price |
|---|---|
| `raw` | DA + p_up × size_up − p_down × size_down |
| `news` | the same with p minus the prior's probability |
| `regime` | `raw`, but only on a side whose latest published RT price is itself a spike (an ongoing spike); DA otherwise |

**Period.** July 1 to December 31, 2021 (184 days) at all five trading hubs: `HB_NORTH`,
`HB_WEST`, `HB_SOUTH`, `HB_HOUSTON`, `HB_PAN`.

- Models are refit monthly on the data available from January 2021: 6 to 11 months, including
  Uri.
- Policies: `perfect_rt`, `mpc_oracle_1h`, `mpc_oracle_spike`, `mpc_oracle_timing`, `mpc_da`,
  and the three candidates.
- At `HB_NORTH` only, the spike oracle is also run at thresholds $25, $50, $250, $500, and
  $1,000. This only describes where the one-hour oracle's value lies; the $100 threshold stays.

**Selection rule.**

- The candidate with the highest revenue, averaged over the five hubs, becomes `mpc_spike`
  for the test.
- The test then runs once, whatever the development results show.
- Nothing else changes after the development stage. Any bug fix would be reported with its
  effect.

## Test

**Primary.**

- Measure: `mpc_spike` minus `mpc_da`, $/kW-year, from January 2022 to October 2026.
- Pooling: daily revenue is averaged over the five hubs day by day.
- Confidence interval: weekly moving-block bootstrap that resamples the same weeks at every
  hub, 2,000 resamples.
- Confirmed only if the 95% CI lies above zero.

**Secondary (reported, not judged):**

1. `mpc_spike` minus `mpc_da` per hub, per year, and from 2024 on, pooled. From 2024 on, every
   model trains on two full years without Uri. The 2024 split is set here, in advance.
2. All five hubs: `mpc_oracle_1h`, `mpc_oracle_spike`, `mpc_oracle_timing`. These show how
   much of the next hour's value lies in spikes, and what a perfect classifier with typical
   sizes would earn.
3. `HB_NORTH` only:
   - the spike-oracle threshold sweep
   - the two candidates not selected
   - `mpc_spike_f` (with fundamentals)
   - `mpc_gbm_1h`: the existing LightGBM forecast for the next hour only, DA after. It
     separates the spike target from the one-hour horizon.
   - `mpc_spike` with thresholds of $50 and $250
4. Classifier skill at `HB_NORTH` (`batt nowcast`) against the prior: ROC AUC, average
   precision, Brier skill score, and reliability, by step ahead and by year.

## Outcome of the development stage (2026-11-07, before the test)

July 1 to December 31, 2021, $/kW-year. The middle columns are the policy minus `mpc_da`,
daily revenue averaged over the five hubs, with 95% CIs from resampling the same weeks at every
hub:

| Policy | Pooled | minus `mpc_da` (95% CI) | `HB_NORTH` |
|---|---:|---|---:|
| `perfect_rt` | 44.2 | +17.3 [+13.3, +22.8] | 42.1 |
| `mpc_oracle_1h` | 36.0 | +9.1 [+6.2, +12.8] | 34.7 |
| `mpc_oracle_spike` | 32.9 | +6.0 [+3.1, +10.5] | 32.5 |
| `mpc_oracle_timing` | 31.7 | +4.8 [+1.8, +9.4] | 31.5 |
| `mpc_da` | 26.9 | | 26.2 |
| `regime` | 26.9 | −0.0 [−1.9, +2.2] | 26.7 |
| `news` | 22.5 | −4.4 [−8.3, −1.1] | 21.8 |
| `raw` | 21.9 | −5.0 [−10.1, −0.9] | 21.9 |

**Selected: `regime`** (highest pooled revenue). It is now the default (`forecasters.spike.adjust`
in `configs/default.yaml`), so `mpc_spike`, `mpc_spike_f`, and the threshold variants use it.
`raw` and `news` run in the test at `HB_NORTH` as the candidates not selected.

Spike-oracle threshold sweep at `HB_NORTH` (minus `mpc_da`): $25: +7.0, $50: +6.8, $100:
+6.4, $250: +5.0, $500: +1.5, $1,000: 0.0 (no spread that large in this period). The one-hour
oracle is +8.5 there, so in this period most of its value lies in spreads above $100.

Observations, not used to change anything: perfect timing with typical sizes earns about half
of the one-hour oracle's gain. Yet the realistic `raw` and `news` mappings lose about $5, so
their probabilities move prices when nothing happens. `regime` acts only during an ongoing
spike and ties `mpc_da`.

## Outcome of the test (2026-11-07)

Run once, as fixed above, January 2022 to October 2026 at the five hubs. Every run finished.
`HB_NORTH`'s `perfect_rt` (88.0), `mpc_oracle_1h` (72.3), and `mpc_da` (52.3) match the main
backtest exactly.

**Primary: not confirmed.** `mpc_spike` minus `mpc_da`, daily revenue averaged over the five
hubs: **−$1.3/kW-year [−4.2, +1.3]**.

**Secondary 1.** `mpc_spike` minus `mpc_da`, $/kW-year (95% CI):

| | 2022 to Oct 2026 | From 2024 |
|---|---|---|
| `HB_NORTH` | −1.8 [−5.5, +1.5] | −0.3 [−1.9, +1.3] |
| `HB_WEST` | −0.9 [−4.2, +2.0] | −0.0 [−1.8, +1.7] |
| `HB_SOUTH` | −0.8 [−3.3, +1.4] | +0.3 [−0.8, +1.4] |
| `HB_HOUSTON` | −2.4 [−6.1, +0.5] | −0.0 [−0.8, +0.8] |
| `HB_PAN` | −0.4 [−3.4, +2.6] | +0.1 [−1.1, +1.7] |
| Pooled | −1.3 [−4.2, +1.3] | +0.0 [−1.1, +1.2] |

Pooled by year: 2022 −9.3 [−21.6, −0.4], 2023 +3.1 [−3.2, +10.6], 2024 +1.1 [−0.8, +3.7],
2025 −0.9 [−2.3, +0.2], 2026 (January to October) −0.2 [−2.3, +1.4].

**Secondary 2.** Oracles, pooled, minus `mpc_da`:

| | 2022 to Oct 2026 | From 2024 |
|---|---|---|
| `perfect_rt` | +37.5 [+28.1, +48.6] | +18.4 [+15.6, +21.3] |
| `mpc_oracle_1h` | +21.5 [+14.5, +29.9] | +8.0 [+6.4, +9.6] |
| `mpc_oracle_spike` | +15.9 [+9.0, +24.2] | +3.8 [+2.3, +5.5] |
| `mpc_oracle_timing` | +6.4 [+1.6, +11.6] | +1.4 [0.0, +2.7] |

The timing oracle by year: 2022 +0.7 [−13.1, +11.5], 2023 +26.1 [+9.9, +47.1], 2024 +3.2
[+0.8, +6.4], 2025 +0.2 [−2.0, +2.2], 2026 +0.7 [−1.5, +2.5].

**Secondary 3.** `HB_NORTH`, minus `mpc_da`, $/kW-year (95% CI):

- Spike-oracle sweep: $25: +17.6, $50: +16.4, $100: +15.5, $250: +14.2, $500: +12.0, $1,000:
  +9.6 [+3.5, +16.7]. The one-hour oracle: +19.9.
- The candidates not selected: `raw` −2.8 [−7.0, +0.4], `news` −4.6 [−9.3, −1.1].
- `mpc_spike_f` (with fundamentals): −1.1 [−4.2, +1.9].
- `mpc_gbm_1h`: −4.1 [−6.6, −1.7].
- Thresholds: `mpc_spike_50` −2.2 [−4.9, +0.4], `mpc_spike_250` +0.6 [−2.5, +3.4].

**Secondary 4.** Classifier skill at `HB_NORTH`: every decision from 2022 to October 2026
(169,436), each paired with the four intervals of its hour. The prior sees calendar, DA-price,
and 7-to-30-day spread and spike-rate features, but no recent RT prices.

| | Up-spikes | Down-spikes |
|---|---:|---:|
| Share of intervals | 1.0% | 1.5% |
| ROC AUC (prior) | 0.961 (0.863) | 0.998 (0.994) |
| Average precision (prior) | 0.34 (0.06) | 0.90 (0.65) |
| Brier skill score against the prior | 0.21 | 0.45 |
| ... first to fourth interval of the hour | 0.25 to 0.17 | 0.50 to 0.40 |
| ... by year, 2022 to 2026 | 0.29, 0.24, 0.09, 0.05, 0.18 | 0.53, 0.41, 0.42, 0.28, 0.44 |

Reliability:

- Up-spike probabilities between 0.1 and 0.5 are calibrated: at a mean of 0.27, 26% spiked.
- Below 0.05, up-spikes came about twice as often as predicted.
- Above 0.75, they came less often: at a mean of 0.83, 68% spiked.
- Down-spike probabilities above 0.75 averaged 0.93, and 87% spiked.

**Post-hoc diagnostics.** These are descriptive. Nothing was changed or re-run because of them.

- **A few days decide the result.** Averaged over the hubs, `mpc_spike` fell $0.62M behind
  `mpc_da` over the period. Its 10 worst days lost $1.25M and its 10 best gained $1.09M. Without
  the 10 worst days it would be $1.3/kW-year ahead; without both, $1.0 behind.
- **What separates them, on the three best and three worst days.** The mapping acts on a spike
  already under way, while every hour after the next keeps its DA price. So `mpc_spike` sells
  into the spike now; `mpc_da` keeps its energy for the DA peak.
  - It wins when the spike comes apart from the DA peak (examples at `HB_NORTH`).
    - On December 23, 2022 (Winter Storm Elliott), RT reached $2,400–3,900/MWh at 06:00–07:00.
      `mpc_da` waited for DA's morning peak (07:00–09:00), by the end of which RT had fallen to
      $77.
    - On September 6, 2023, RT reached $5,000 at 16:00, three hours before DA's peak.
    - On August 28, 2023, RT came in hundreds of dollars below DA all afternoon. `mpc_da` sold
      at $45–200 against DA's $690. `mpc_spike` saw the shortfall and held its charge for the RT
      spike at 19:15–20:00.
  - It loses when the spike builds into the DA peak. On July 13, 2022, at `HB_NORTH`, it sold
    most of the battery at $1,190–2,530 between 13:00 and 14:15. RT then stayed above $4,800
    from 14:30 to 18:15, when `mpc_da` sold all of its energy at DA's peak. That day `mpc_spike`
    earned $0.41M and `mpc_da` $0.94M. February 24, 2022, and July 10, 2023, went the same way:
    sales into an early, smaller spike left too little for the one at DA's peak.
- **During a spike, the forecast is far below the price that follows.** At `HB_NORTH` the
  up-spike side was active at 1.0% of decisions. There it added $50–171/MWh to DA on average,
  by year, while the realized spread averaged $155–497/MWh. Its probabilities averaged
  0.12–0.31, while 28–54% of those intervals spiked. The size it uses is a typical spike's, not
  the size of the spike in progress. On the worst days it even charged during spikes: at about
  $3,000/MWh on February 24, 2022 (09:30–09:45, forecast about $300), and at $1,383 on July 10,
  2023 (18:45, forecast about $130).
- **Perfect timing with typical sizes is worth little outside 2023.** Pooled, that is +0.7 in
  2022 and +1.4 from 2024 on. Knowing each spike's realized size as well is worth much more:
  +15.9 against +6.4 over the period.

## Reproducing (added 2026-11-07, after the test)

The averages over the hubs above were computed with a scratch script. After the test it became
`batt pool`, which reproduces all of them exactly.

Development stage, at each of the five hubs (add the sweep `mpc_oracle_spike_25,
mpc_oracle_spike_50,mpc_oracle_spike_250,mpc_oracle_spike_500,mpc_oracle_spike_1000` at
`HB_NORTH`):

```bash
batt backtest --location HB_WEST --start 2021-07-01 --end 2022-01-01 --policies perfect_rt,mpc_oracle_1h,mpc_oracle_spike,mpc_oracle_timing,mpc_da,mpc_spike_raw,mpc_spike_news,mpc_spike_regime
batt pool --start 2021-07-01 --end 2022-01-01 --policies perfect_rt,mpc_oracle_1h,mpc_oracle_spike,mpc_oracle_timing,mpc_spike_regime,mpc_spike_news,mpc_spike_raw
```

Test, at `HB_NORTH`:

```bash
batt backtest --location HB_NORTH --jobs 12 --policies perfect_rt,mpc_oracle_1h,mpc_oracle_spike,mpc_oracle_timing,mpc_da,mpc_spike,mpc_spike_raw,mpc_spike_news,mpc_spike_f,mpc_gbm_1h,mpc_spike_50,mpc_spike_250,mpc_oracle_spike_25,mpc_oracle_spike_50,mpc_oracle_spike_250,mpc_oracle_spike_500,mpc_oracle_spike_1000
batt nowcast
```

At `HB_WEST`, `HB_SOUTH`, `HB_HOUSTON`, and `HB_PAN`:

```bash
batt backtest --location HB_WEST --policies perfect_rt,mpc_oracle_1h,mpc_oracle_spike,mpc_oracle_timing,mpc_da,mpc_spike
```

Then:

```bash
batt pool --policies perfect_rt,mpc_oracle_1h,mpc_oracle_spike,mpc_oracle_timing,mpc_spike
batt pool --policies perfect_rt,mpc_oracle_1h,mpc_oracle_spike,mpc_oracle_timing,mpc_spike --since 2024-01-01
```

`batt pool` uses each hub's most recent run for the period and prints which folders it used.
Runtimes here (16 cores, 15.7 GB RAM): the `HB_NORTH` test run took 46 minutes with 12
processes. The other hubs took 18–24 minutes each and `batt nowcast` 19 minutes, two or three
of them at a time.
