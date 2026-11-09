# Real-time grid conditions: analysis plan

Written on 2026-11-07, before any model or policy used these data on the backtest period
(January 2022 to October 2026). The sections "Data", "Fixed design", "Development stage", and
"Test" are not edited after this date. Outcomes are added below them, dated.

## What was known before

- The backtest period has been used by every earlier milestone. The missing revenue is in the
  next hour: perfect knowledge of it is worth +$21.5/kW-year over `mpc_da`, averaged over the
  five trading hubs, and spikes beyond $100/MWh carry +$15.9 of it.
- The spike nowcast found that skill at predicting spikes did not become revenue. Its forecast
  sat far below the price during spikes, and selling into an early spike left too little for
  the DA peak ([spike plan](spike-nowcasting-plan.md)). That was learned on 2022–2026, so the
  design below states every choice in advance and does not try to fix those failures.
- The new data were inspected only for availability, columns, and publishing delays: a few
  rows on single days in 2021, 2023, and 2026, and two days of July 2021 to test the download.
  Nothing about how they relate to prices in 2022–2026 was looked at.

## Data

All from the ERCOT Public API archive, stored with the time each file was posted
(`batt ingest-conditions`, `batt.data.conditions`), from January 2021:

- **SCED** (NP6-323-CD), every real-time dispatch run, posted about 3 seconds after it runs.
  - The system-wide energy price: system lambda plus the real-time price adders.
  - Physical Responsive Capability (PRC, MW), ERCOT's measure of real-time reserves.
- **RTD look-ahead** (NP6-325-CD), every 5 minutes: ERCOT's indicative energy price for each of
  the next eleven 5-minute intervals.
- **Outages** (NP3-233-CD), every hour: MW of generation out of service in each hour of the
  coming week, wind and solar excluded.

The archive allows only 50 files per download request for the two 5-minute reports. So only
the run each decision would see is kept: the latest one posted by each decision time (every
15 minutes, 5 minutes before an interval starts). At a decision, that SCED run is about 5
minutes old, against 25 for the latest settlement price. That RTD run covers the next three
15-minute intervals but not the fourth.

ERCOT's real-time market redesign (RTC+B, December 5, 2025) changed both 5-minute reports:
PRC is no longer published in them, so it is missing from then on. The price is still the
system lambda plus the price adder that applies.

## Questions

1. Does ERCOT's own one-hour look-ahead, used directly, earn more than planning on DA prices?
2. Do reserves, 5-minute prices, the look-ahead, and outages improve the LightGBM price
   forecast and the spike nowcast? Is that worth revenue?

## Fixed design

**`mpc_rtd`.** It re-plans every 15 minutes, as `mpc_da` does. For each of the next
hour's intervals that the latest RTD run covers fully, the forecast is RTD's price averaged
over the interval, converted to the hub as chosen in the development stage. Every other
interval uses DA prices. There is no model and no training.

**Six features** (`conditions: true`), for an issue time and a target interval:

- `sced_spread`: the latest SCED price minus the DA price of the 15-minute interval it ran in.
- `prc`: the latest PRC.
- `prc_change`: the latest PRC minus that of the latest run public 15 minutes earlier.
- `rtd_spread`: RTD's price for the target interval minus its DA price (missing beyond the
  look-ahead).
- `outage_mw`: the latest published MW out for the target's hour.
- `outage_rev`: its change since the publication current at the DA bid deadline (10:00 the
  day before): news the DA price could not contain.

**Models.** `gbm_c`, `qgbm_c`, and `spike_c` are `gbm`, `qgbm`, and `spike` with the same
settings plus the six features. `spike_c` keeps the `regime` mapping and the $100 threshold.
The policies are `mpc_gbm_c` and `mpc_spike_c`.

## Development stage

**Open choices.** Two choices for `mpc_rtd`, giving four candidates:

- `mode`:
  - `all`: RTD's price for every covered interval.
  - `spikes`: RTD's price only where it is more than $100/MWh from DA, and DA elsewhere. This
    acts only on spikes, as the selected spike mapping did.
- `basis`, how RTD's system-wide price becomes a hub price:
  - `none`: use it as is.
  - `latest`: add the latest published hub price minus the SCED price from the same interval.

**Period.** July 1 to December 31, 2021, at the five trading hubs, before the backtest period.
Policies: `perfect_rt`, `mpc_oracle_1h`, `mpc_da`, and the four candidates.

**Selection rule.** The candidate with the highest revenue averaged over the five hubs becomes
`mpc_rtd` for the test. The `_c` models have no open choices and are not run here.

## Test

**Primary.**

- Measure: `mpc_rtd` minus `mpc_da`, $/kW-year, from January 2022 to October 2026.
- Pooling: daily revenue averaged over the five hubs, day by day (`batt pool`).
- Confidence interval: weekly moving-block bootstrap that resamples the same weeks at every hub,
  2,000 resamples.
- It counts as confirmed only if the 95% CI lies above zero. It is run once.

**Secondary (reported, not judged):**

1. `mpc_rtd` minus `mpc_da`:
   - per hub
   - per year
   - from 2024 on
   - before and after RTC+B (December 5, 2025)
2. At the five hubs, pooled the same way:
   - `mpc_spike_c` minus `mpc_spike`
   - `mpc_gbm_c` minus `mpc_gbm`
   - each of them minus `mpc_da`
3. At `HB_NORTH` (`batt forecast`):
   - `gbm_c` and `qgbm_c` against `gbm` and `qgbm`, by MAE and CRPS by lead time
   - RTD itself scored as a forecast (`rtd`)
4. At `HB_NORTH` (`batt nowcast`): `spike_c` against `spike` by ROC AUC, average precision, and
   Brier skill score, overall and by year.
5. The share of `gbm_c`'s and `spike_c`'s learned gain that goes to the six features.

Results after RTC+B are reported separately wherever they are shown by year. Before RTC+B the
models never saw PRC missing. After it, LightGBM routes the missing values down the branch
each split learned for missing data.

## Outcome of the development stage (2026-11-07)

Run as fixed above, once a missing month of data had been fetched (see "A gap in the first
run"). July 1 to December 31, 2021, in $/kW-year, averaged over the five hubs day by day
(`batt pool`), with 95% CIs against `mpc_da`:

| Policy | $/kW-year | minus `mpc_da` |
|---|---|---|
| `perfect_rt` | 44.2 | +17.3 [+13.3, +22.8] |
| `mpc_oracle_1h` | 36.0 | +9.1 [+6.2, +12.8] |
| `mpc_da` | 26.9 | |
| look-ahead, `spikes`, `none` | **24.7** | −2.2 [−4.3, −0.7] |
| look-ahead, `spikes`, `latest` | 23.0 | −3.9 [−7.2, −1.4] |
| look-ahead, `all`, `none` | 21.1 | −5.8 [−10.1, −2.3] |
| look-ahead, `all`, `latest` | 19.7 | −7.2 [−12.4, −3.5] |

**Selected: `mode: spikes`, `basis: none`**, the highest of the four. It is now the default
for `rtd` (`configs/default.yaml` and `LookAheadForecaster`), so `mpc_rtd` is this candidate in
the test. Every candidate earned less than `mpc_da` at every hub. The selected one was
−1.9 to −2.5 at the five hubs. The rule had no exception for this, and the test runs as fixed,
once. This stage makes a confirmed result unlikely.

In secondary 3, `rtd` is now the selected forecaster, which is DA except where the look-ahead
is more than $100 away. The look-ahead as published, which is what "RTD itself" meant, is also
scored, as `rtd_all` (added to the config with `mode: all`).

**A gap in the first run.** The first run of this stage had no SCED data for August 2021.
The download of that month failed with an expired sign-in (HTTP 401) exactly an hour after the
download started, and the download went on without it. gridstatus renews its token by its own
clock, which starts a moment after ERCOT's. Only the two `latest` candidates read SCED data.
With August stored they earned a little less (23.2 to 23.0 and 20.2 to 19.7). The `none`
candidates were unchanged, and so was the ranking. The first run is kept in
`results/superseded/grid-conditions-dev-2021-missing-aug-sced/`. The download now:

- signs in again five minutes early
- tries a failed month three times
- names the months that still failed

Backtests now refuse a period in which a report has a month with nothing stored.

**Why the look-ahead loses** (diagnostics on 2021, run after the selection; they change nothing
in the test):

- **Ordinary intervals.** It is a good forecast there, but not much better than the latest
  price. At HB_NORTH, its mean absolute error for the next interval is $6.4/MWh. DA's is $13.0,
  and the latest SCED price carried forward scores $7.0. Thirty to 45 minutes out it is worse
  than the latest prices ($10.4, against $9.9 for SCED and $9.8 for the hub). At HB_WEST and
  HB_PAN it is the system-wide price without those hubs' congestion, and the latest hub price
  beats it at every step.
- **Where it departs most from DA, it misleads.** At HB_NORTH, 169 of its forecasts put an
  interval more than $100 above DA, by $426 on average. The hub price came in $57 above DA on
  average, and more than $100 above in 21% of them (13–20% at the other hubs).
- **A few days decide the gap.** The worst 5 of 184 days hold 78% of the selected
  candidate's loss, pooled over the hubs. The worst was November 9, at HB_NORTH:
  - The look-ahead saw the evening spike late, and as if it would keep rising. At 17:25 it
    projected $935 for 17:45 and $979 for 18:00. The spike ended at 18:00, at $106.
  - `mpc_rtd` held its last 54 MWh for those prices and sold at $106 and $83.
  - `mpc_da` sold the same energy at $935 and $938, earning $42,000 more that day at HB_NORTH.
- **Mixing it with DA prices fails in a second way.** On October 20, DA expected an evening
  spike of $277–$400 from 16:00 to 20:00. At every decision the look-ahead put the next 45
  minutes at $75–$95. It was right that DA's spike would not come, but it missed the smaller
  one that did (18:15–18:45, $204–$251). So the plan kept waiting for DA's peak, which always
  lay further ahead.
  - `mpc_rtd` sold a fifth of its energy into the smaller spike and half after 20:00, at
    $52–$62.
  - `mpc_da`, following DA's plan, sold 40% of its energy into it.

## The data for the test (2026-11-07)

Downloaded in full after the development stage, from January 2021 to November 7, 2026, and
checked for availability only: publish times, not prices. Every report has data in every
month. In the test period, January 2022 to October 2026:

| Report | Stored | Out of |
|---|---|---|
| SCED: the latest run at each decision | 169,337 runs | 169,436 decisions (99.94%) |
| RTD: the latest run at each decision | 169,333 runs | 169,436 decisions (99.94%) |
| Outages: hourly publications | 42,322 | 42,359 hours (99.91%) |

A decision with no run posted since the decision before uses that decision's run. Two
problems turned up and were fixed before the test:

- **A file the archive will not serve.** The download of May 2026 failed three times. The
  archive lists the RTD run posted at 23:35:55 on May 5 but answers every request for it with
  HTTP 400. Now a batch the archive refuses is fetched one file at a time, a refused file is
  skipped, and the decision that would have used it (23:40) uses the run posted before it
  (23:30:56). More than five refused files in a report and month still fail the month.
- **Outages in May 2021.** Only 103 of the month's 744 publications had been stored. ERCOT
  added zonal columns to the outage report on 2021-05-27 at 17:00. In a download spanning that
  change, gridstatus computes each row's total from the zones, which left the older rows
  without one. Their total is now read from the report's own column, and the month was
  downloaded again (744 publications). These data reach only the `_c` models, which have not
  been run. The development stage did not use outages.

What is left as is:

- **Gaps.** Each 5-minute report has 11 stretches of 60 to 95 minutes with no new run. In 10
  of them ERCOT posted nothing, and decisions use the last run before, as they would have then.
  Nine of those are at about the same times in both reports.
- **The night clocks went back in 2025.** The eleventh gap in each report is in the repeated
  hour of 2025-11-02. That night ERCOT posted the hour's files slightly out of order, so their
  post times, which carry no daylight-saving flag, could not be split into the two passes
  through the hour, and the download read them all as the second pass. That can only make a
  run look later than it was, so a few decisions there use a run up to 75 minutes old. Nothing
  is seen early.

## Outcome of the test (2026-11-08)

Run once, as fixed, on November 7–8, 2026. The five hubs ran one after another, each with all its
policies in parallel. Then came `batt forecast` and `batt nowcast` at `HB_NORTH`, then
`batt pool`. Every step finished without errors. Policies that earlier milestones ran
(`perfect_rt`, `mpc_da`, `mpc_gbm`, `mpc_spike`) earned exactly what they earned then at
`HB_NORTH`, day by day.

Two additions made before the run record more but change no decision:

- each refit of `gbm_c` and `spike_c` logs the six features' share of the learned gain
  (secondary 5)
- the forecast and nowcast tables by year show the days from RTC+B on as their own periods

After the run, the yearly rows of `batt pool` and of the backtest tables were split at RTC+B the
same way. The split rows equal those from pooling only the days before or after RTC+B, which was
part of the run.

### Primary: not confirmed

`mpc_rtd` minus `mpc_da`, January 2022 to October 2026, averaged over the five hubs day by day:
**−$2.6/kW-year [−6.5, +0.5]**.

| $/kW-year | `HB_NORTH` | `HB_WEST` | `HB_SOUTH` | `HB_HOUSTON` | `HB_PAN` | Average |
|---|---:|---:|---:|---:|---:|---:|
| `perfect_rt` | 88.0 | 95.9 | 81.5 | 91.4 | 93.1 | 90.0 |
| `mpc_oracle_1h` | 72.3 | 78.9 | 67.0 | 75.5 | 76.1 | 73.9 |
| `mpc_da` | 52.3 | 56.9 | 47.0 | 52.8 | 53.3 | 52.5 |
| `mpc_rtd` | 49.3 | 54.4 | 45.0 | 50.1 | 50.6 | 49.9 |
| `mpc_spike` | 50.5 | 56.0 | 46.2 | 50.4 | 52.9 | 51.2 |
| `mpc_spike_c` | 51.3 | 55.7 | 47.5 | 52.5 | 52.6 | 51.9 |
| `mpc_gbm` | 46.3 | 51.5 | 42.2 | 46.1 | 46.9 | 46.6 |
| `mpc_gbm_c` | 45.1 | 50.3 | 41.4 | 47.2 | 46.0 | 46.0 |

### Secondary 1: `mpc_rtd` minus `mpc_da`

| Scope | $/kW-year (95% CI) |
|---|---|
| `HB_NORTH` | −3.0 [−7.3, +0.4] |
| `HB_WEST` | −2.5 [−6.4, +0.8] |
| `HB_SOUTH` | −2.0 [−5.3, +0.8] |
| `HB_HOUSTON` | −2.8 [−6.7, +0.6] |
| `HB_PAN` | −2.7 [−6.9, +0.7] |
| Average over the hubs, 2022 | −7.8 [−22.4, +1.9] |
| 2023 | +0.3 [−9.9, +9.5] |
| 2024 | −0.9 [−3.7, +2.5] |
| 2025, to December 4 | −2.0 [−5.4, +0.4] |
| December 5–31, 2025 (RTC+B) | −2.1 [−3.6, −0.7] |
| 2026, to October (RTC+B) | −2.6 [−5.1, −0.8] |
| From 2024 | −1.8 [−3.5, −0.2] |
| Before RTC+B | −2.6 [−7.3, +1.0] |
| From RTC+B | −2.6 [−4.8, −0.8] |

It lost at every hub, by $2.0 to $3.0. From 2024 on, and since RTC+B, the 95% CI lies below
zero.

### Secondary 2: the six features in the policies

| Averaged over the hubs, $/kW-year (95% CI) | Minus its base policy | Minus `mpc_da` |
|---|---|---|
| `mpc_gbm_c` (base `mpc_gbm`) | −0.6 [−2.9, +1.3] | −6.5 [−12.6, −2.0] |
| `mpc_spike_c` (base `mpc_spike`) | +0.7 [−0.2, +1.7] | −0.5 [−4.2, +2.6] |

| Minus its base policy, by period | `mpc_gbm_c` | `mpc_spike_c` |
|---|---|---|
| 2022 | −6.1 [−16.8, +0.8] | +0.0 [−2.8, +2.5] |
| 2023 | +3.4 [+0.2, +7.5] | +3.4 [+1.0, +6.9] |
| 2024 | +0.7 [+0.1, +1.4] | +0.2 [−0.2, +0.9] |
| 2025, to December 4 | −0.6 [−2.1, +0.3] | −0.4 [−1.7, +0.4] |
| December 5–31, 2025 | −0.3 [−1.5, +0.6] | −0.1 [−0.7, +0.3] |
| 2026, to October | −0.2 [−1.2, +0.9] | +0.2 [+0.0, +0.5] |

- `mpc_spike_c` beat `mpc_spike` at three hubs, by +1.3 [+0.4, +2.3] at `HB_SOUTH`,
  +2.1 [+0.5, +4.1] at `HB_HOUSTON`, and +0.8 [−0.0, +1.7] at `HB_NORTH`. It lost $0.3 at
  `HB_WEST` and at `HB_PAN`.
- `mpc_gbm_c` came within −1.2 to +1.1 of `mpc_gbm` at every hub.
- From 2024 on, against the base policy: `mpc_gbm_c` −0.0 [−0.6, +0.5], `mpc_spike_c`
  +0.0 [−0.5, +0.4]. Since RTC+B: −0.2 [−1.2, +0.8] and +0.2 [+0.0, +0.4].
- From 2024 on, against `mpc_da`: `mpc_gbm_c` −0.4 [−2.2, +1.2], `mpc_spike_c`
  +0.0 [−1.3, +1.5].

### Secondary 3: forecast accuracy at `HB_NORTH`

`batt forecast`, one forecast issued every hour, 2022 to October 2026. CRPS in $/MWh, which
equals the MAE for single-value forecasts:

| By lead time | 0–1 h | 1–4 h | 4–12 h | 12–24 h | 24–36 h |
|---|---:|---:|---:|---:|---:|
| `da` | 21.6 | 21.6 | 21.6 | 27.0 | 32.3 |
| `gbm` | 19.3 | 21.2 | 22.5 | 28.0 | 34.5 |
| `gbm_c` | 18.6 | 21.1 | 22.3 | 27.7 | 34.2 |
| `qgbm` | 13.0 | 14.8 | 16.0 | 20.1 | 24.4 |
| `qgbm_c` | 12.5 | 14.7 | 15.9 | 19.9 | 24.1 |
| `rtd`: DA, except where the look-ahead is more than $100 away | 26.9 | 21.6 | 21.6 | 27.0 | 32.3 |
| `rtd_all`: the look-ahead as published | 25.9 | 21.6 | 21.6 | 27.0 | 32.3 |

- The features help most in the first hour: 4% lower CRPS for `qgbm_c` and 4% lower MAE for
  `gbm_c`. Further out the gain is about 1%.
- `qgbm_c`'s central 50% and 80% ranges hold 50–52% and 83–84% of prices, as `qgbm`'s do.
- By period, over all lead times, `qgbm_c` beat `qgbm` in 2022 (CRPS 28.8 against 29.6). It was
  within 0.1 of it in every later period.
- `gbm_c` beat `gbm` in 2022, 2023, 2025 to December 4, and December 5–31 (10.2 against 11.7).
  It was worse in 2024 (14.6 against 14.3) and 2026 (13.7 against 13.4).
- The look-ahead scores worse than DA prices over the first hour, both as published and as
  `mpc_rtd` uses it. A forecast issued on the hour starts with an interval that starts on the
  hour, which is where the look-ahead errs most from 2025 on (see below).

### Secondary 4: the spike nowcast at `HB_NORTH`

| | Up-spikes, `spike` | Up-spikes, `spike_c` | Down-spikes, `spike` | Down-spikes, `spike_c` |
|---|---:|---:|---:|---:|
| ROC AUC | 0.961 | 0.971 | 0.998 | 0.998 |
| Average precision | 0.343 | 0.440 | 0.899 | 0.915 |
| Brier skill score against the prior | 0.207 | 0.269 | 0.448 | 0.510 |

- Up-spike average precision rose in every period:

  | Period | `spike` | `spike_c` |
  |---|---:|---:|
  | 2022 | 0.41 | 0.49 |
  | 2023 | 0.41 | 0.49 |
  | 2024 | 0.21 | 0.28 |
  | 2025, to December 4 | 0.13 | 0.23 |
  | December 5–31, 2025 | 0.21 | 0.36 |
  | 2026 | 0.36 | 0.51 |

- The Brier skill score rose in every period for both sides, with one exception: down-spikes in
  December 5–31, 2025. There were none then, and it fell from 0.05 to −0.25.

### Secondary 5: the features' share of the learned gain

LightGBM split gain, over the 58 monthly refits at `HB_NORTH`:

| Model | Mean | Range |
|---|---:|---:|
| `gbm_c` | 7.5% | 4.3–11.0% |
| `spike_c`, up-spikes | 19.9% | 11.1–37.9% |
| `spike_c`, down-spikes | 15.9% | 4.0–41.5% |

- The up-spike classifier leaned on them most in the refits of 2022 and of the RTC+B period
  (27% and 28% on average). It leaned on them least in 2024 (12%).
- The down-spike classifier's share fell from 38% in 2022 to 6–7% from 2025 on.

### The candidates not selected, at `HB_NORTH` (descriptive)

| Look-ahead candidate | $/kW-year | minus `mpc_da` (95% CI) |
|---|---:|---|
| `spikes`, `none` (selected: `mpc_rtd`) | 49.3 | −3.0 [−7.3, +0.4] |
| `all`, `none` | 48.0 | −4.4 [−8.1, −0.9] |
| `spikes`, `latest` | 45.4 | −6.9 [−11.0, −3.5] |
| `all`, `latest` | 42.4 | −9.9 [−14.3, −6.2] |

The selected candidate came first here too. The second and third swapped places.

### Why the look-ahead lost again

These diagnostics were run after the results. They change nothing in the test.

- **The data are what ERCOT published.** For one run checked by hand (October 21, 2025, 18:35),
  the stored price equals the archive's own file. Every stored run was posted after it ran. In
  each report, about 40 of 205,000 runs were posted more than 10 minutes late, most of them at
  the same moments in both reports: ERCOT's own delays.
- **On ordinary intervals it is no better than the latest price.** At `HB_NORTH`, consider the
  next interval at every decision.
  - In 2022–2024 its MAE beat DA's, as in 2021: 14.9 against 30.7, 24.9 against 37.0, and
    9.9 against 13.3.
  - The latest published hub price, 25 minutes old, scored better still in every period: 14.4,
    23.6, and 8.5 in those years.
- **From 2025 it forecast too high, mostly on the hour.**
  - For the next interval at `HB_NORTH`, its bias was +$8.8/MWh in 2025 and +$7.9 in 2026.
    DA's was +$1.1 and +$2.4.
  - Its MAE there was worse than DA's: 15.8 against 11.9 in 2025, and 14.4 against 13.7 in
    2026.
  - Across the three intervals it covers, the error sits in those that start on the hour. In
    2025 its MAE there was $45.4, against DA's $12.3. In intervals starting at :45 it was $9.8,
    against $12.5. In 2026 the same pairs were $45.5 against $14.0, and $12.2 against $13.9.
- **Its big up-moves seldom came true, and came in far smaller.** These are the intervals it put
  more than $100 above DA, which `mpc_rtd` acts on, at the five hubs:

  | Period | Decisions | Look-ahead minus DA | Realized minus DA | Realized > $100 above DA |
  |---|---:|---:|---:|---:|
  | 2022 | 1,065 | +$684 | +$453 | 39% |
  | 2023 | 764 | +$790 | +$256 | 28% |
  | 2024 | 543 | +$465 | +$120 | 21% |
  | 2025, to December 4 | 979 | +$967 | +$74 | 16% |
  | December 5–31, 2025 | 107 | +$442 | +$52 | 22% |
  | 2026 | 1,444 | +$725 | +$53 | 16% |

  - In 2021 at `HB_NORTH` the same figures were +$426, +$57, and 21%.
  - Its moves of more than $100 below DA mostly came true, 66–87% of the time.
  - In 2026 it put an interval of the next hour above $1,000/MWh at 673 decisions. The SCED
    price reached $1,000 in 4 of the runs stored that year, one per decision.
  - At 16 decisions in 2025–2026 it showed $5,000/MWh or more within the hour. The SCED price
    stayed below $1,000 at every one of them.
- **On the worst day, the spike the look-ahead showed was real.** July 13, 2022 holds 42% of
  the loss (−$5.3 of −$12.6/kW, averaged over the hubs).
  - At `HB_NORTH`, from 12:45 on, the look-ahead put the next 45 minutes at $1,257–5,001/MWh.
    DA prices for the afternoon were $487–891.
  - `mpc_rtd` sold 146 of its 200 MWh between 12:45 and 14:00, at $1,186–2,087. It sold the last
    54 MWh at 15:15–15:30, at about $5,050.
  - RT stayed above $4,800 from 14:30 to 18:15. `mpc_da` kept its energy for DA's peak and sold
    all of it from 15:00 to 16:45, at $5,018–5,291.
  - That day at `HB_NORTH`, `mpc_rtd` earned $0.38M and `mpc_da` $0.94M.
  - `mpc_spike` failed the same way that day. A signal that sees one hour ahead, with DA prices
    beyond it, makes a spike look short.
- **The rest of the loss is spread out.** Averaged over the hubs, `mpc_rtd` earned more than
  $5/MW less than `mpc_da` on 423 days, and more than $5/MW more on 154. Without its 10 worst
  and 10 best days, it still lost $9.9/kW over the period.

### A change in the data during the test period

On 2026-08-27, at about 16:05, ERCOT lengthened the look-ahead from 11 five-minute intervals
to 24 (two hours). All 24 are stored. The policy and the features read the first 12 (one
hour), so the test used the same next three 15-minute intervals throughout.

### What this means

Grid conditions carry news the DA price cannot contain, and the models used them. The six
features took 8–20% of the models' learned gain and improved the forecasts and the spike
classifier a little. That did not become revenue.

ERCOT's own look-ahead, used directly, lost to the DA plan, as it did in the development stage.
Its errors were the ones found there:

- big up-moves that seldom came true
- when a spike was real, a one-hour view that made the planner sell before the peak

The spike nowcast had the second failure too. What is missing is not whether a spike is coming,
but how long it will last.
