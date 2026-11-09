# Real-time grid conditions

Part of [Battery arbitrage on the Texas grid](../README.md).

At a decision the latest settlement price is 25 minutes old. ERCOT publishes faster signals.
Its real-time dispatch (SCED) runs every 5 minutes and posts the system price and reserves within
seconds. Its look-ahead (RTD) posts ERCOT's own price for each of the next eleven 5-minute
intervals. Can these earn part of the one-hour oracle's $21.5/kW-year? The design and the test
were written down before these data touched 2022–2026
([plan and dated outcomes](grid-conditions-plan.md)).

**Data.** `batt ingest-conditions` (`batt.data.conditions`) downloads them from the ERCOT Public
API archive, with the same free account as the [fundamentals](fundamentals.md#setup):

| Report | Kept | Published |
|---|---|---|
| SCED (NP6-323-CD) | System lambda plus the real-time price adders; Physical Responsive Capability (PRC, MW), ERCOT's measure of real-time reserves | every run, ~3 seconds after it |
| RTD look-ahead (NP6-325-CD) | ERCOT's indicative price for each of the next 11 five-minute intervals (24 from 2026-08-27) | every run, ~15 seconds after it |
| Hourly Resource Outage Capacity (NP3-233-CD) | MW out of service in each hour of the coming week, wind and solar apart | hourly |

- The archive serves at most 50 files per request for the two 5-minute reports. So only the run
  each 15-minute decision would see is kept: the latest one posted by then, a third of all
  runs.
  - January 2021 to November 2026 took about 10 hours.
  - The download is resumable, like `ingest-fundamentals`, and `--reports` picks reports.
- In the backtest period, 99.94% of decisions have a fresh SCED and RTD run, and 99.91% of hours
  have an outage publication. A decision with no new run since the one before uses that one.
- What the download handles:
  - The archive's sign-in lasts an hour, and is renewed five minutes early.
  - One RTD file (posted 2026-05-05, 23:35) is listed but refused with HTTP 400. It is skipped,
    and its decision uses the run before.
  - The outage report gained zonal columns on 2021-05-27. In a month spanning that day,
    gridstatus leaves the older rows without a total, so it is read from the report's own
    column.
  - RTC+B (December 5, 2025) removed PRC and the reserve adders from both 5-minute reports. PRC
    is missing from then on, and the price is the system lambda plus the reliability
    deployment adder.
  - On 2026-08-27 the look-ahead grew to two hours. The policy and the features read its first
    hour throughout.

**Design.**

- `mpc_rtd` re-plans every 15 minutes, like `mpc_da`. For each of the next hour's intervals that
  the latest look-ahead run covers fully (the next three), it uses the look-ahead's price where
  that is more than $100/MWh from DA, and DA elsewhere. There is no model and no training.
- Two choices were open:
  - use the look-ahead everywhere it reaches, or only where it is $100 from DA
  - use it as is, or add the latest gap between the hub price and the system price

  They were made on July to December 2021, at all five hubs, by revenue. Every candidate lost
  to `mpc_da` at every hub. The best, which was chosen, lost $2.2/kW-year [−4.3, −0.7].
- `gbm_c`, `qgbm_c`, and `spike_c` add six features to their base models:
  - the latest SCED price minus DA
  - PRC, and its change over 15 minutes
  - the look-ahead's price minus DA for the target interval
  - the MW out of service in the target hour, and its change since the DA bid deadline

**Test.** `mpc_rtd` minus `mpc_da` from January 2022 to October 2026. Daily revenue is averaged
over the five trading hubs, and the CIs resample the same weeks at every hub. It counts as
confirmed only if the 95% CI is above zero. It was run once.

**Result: not confirmed.** −$2.6/kW-year [−6.5, +0.5].

| $/kW-year, 2022 to Oct 2026 | `HB_NORTH` | `HB_WEST` | `HB_SOUTH` | `HB_HOUSTON` | `HB_PAN` | Average |
|---|---:|---:|---:|---:|---:|---:|
| `mpc_da` | 52.3 | 56.9 | 47.0 | 52.8 | 53.3 | 52.5 |
| `mpc_rtd` | 49.3 | 54.4 | 45.0 | 50.1 | 50.6 | 49.9 |
| `mpc_spike` | 50.5 | 56.0 | 46.2 | 50.4 | 52.9 | 51.2 |
| `mpc_spike_c` | 51.3 | 55.7 | 47.5 | 52.5 | 52.6 | 51.9 |
| `mpc_gbm` | 46.3 | 51.5 | 42.2 | 46.1 | 46.9 | 46.6 |
| `mpc_gbm_c` | 45.1 | 50.3 | 41.4 | 47.2 | 46.0 | 46.0 |

| Averaged over the hubs (95% CI) | 2022 to Oct 2026 | From 2024 | Since RTC+B |
|---|---|---|---|
| `mpc_rtd` − `mpc_da` | −2.6 [−6.5, +0.5] | −1.8 [−3.5, −0.2] | −2.6 [−4.8, −0.8] |
| `mpc_spike_c` − `mpc_spike` | +0.7 [−0.2, +1.7] | +0.0 [−0.5, +0.4] | +0.2 [0.0, +0.4] |
| `mpc_gbm_c` − `mpc_gbm` | −0.6 [−2.9, +1.3] | −0.0 [−0.6, +0.5] | −0.2 [−1.2, +0.8] |

`mpc_rtd` lost at every hub, by $2.0–3.0, though no single hub's CI excludes zero. By year it
was −7.8 in 2022, +0.3 in 2023, −0.9 in 2024, and −2.0 in 2025 up to RTC+B.

**The models used the new data, a little.** At `HB_NORTH`:

- The six features carry 7.5% of `gbm_c`'s learned gain and 16–20% of the spike classifiers'.
- Up-spike average precision rose from 0.34 to 0.44, and ROC AUC from 0.961 to 0.971. It rose
  in every year.
- First-hour CRPS fell from 13.0 to 12.5 for `qgbm_c`, and MAE from 19.3 to 18.6 for `gbm_c`.
  Further out the gain is about 1%.

**Why the look-ahead lost.** These diagnostics came after the test, and nothing was changed
because of them.

- The stored prices match ERCOT's archive, and no run was used before it was posted.
- On ordinary intervals it beat DA prices but not the latest published price, as in 2021. At
  `HB_NORTH` in 2024, its MAE for the next interval was $9.9/MWh. DA's was $13.3, and the
  latest hub price's, 25 minutes old, $8.5.
- Its big up-moves seldom came true. These are the intervals it put more than $100 above DA,
  at the five hubs:

  | Period | Look-ahead minus DA | Realized minus DA | Realized > $100 above DA |
  |---|---:|---:|---:|
  | 2022 | +$684 | +$453 | 39% |
  | 2023 | +$790 | +$256 | 28% |
  | 2024 | +$465 | +$120 | 21% |
  | 2025, to Dec 4 | +$967 | +$74 | 16% |
  | Dec 5–31, 2025 | +$442 | +$52 | 22% |
  | 2026 | +$725 | +$53 | 16% |

  In 2026 it put an interval of the next hour above $1,000/MWh at 673 decisions. The SCED price
  reached $1,000 in 4 of the runs stored that year.
- From 2025 its errors cluster on the hour. At `HB_NORTH`, in 2025 up to RTC+B and in 2026, its
  MAE in intervals that start on the hour was about $45/MWh, against DA's $12–14. In intervals
  starting at :45 it beat DA.
- On the worst day the spike was real. July 13, 2022 holds 42% of the loss. The look-ahead saw
  the spike 45 minutes ahead, but DA prices beyond that were far lower, so `mpc_rtd` sold most
  of the battery early, at $1,186–2,087/MWh. The price then stayed above $4,800 from 14:30 to
  18:15, and `mpc_da` sold everything into it. That day at `HB_NORTH`, `mpc_rtd` earned $0.38M
  and `mpc_da` $0.94M. The same day went against `mpc_spike`
  ([spike nowcasting](spike-nowcasting.md)).

To reproduce, run `batt ingest-conditions`. Then run
`batt backtest --location <hub> --policies <list>` at each hub, then `batt pool`,
`batt forecast`, and `batt nowcast --forecaster spike_c`. The exact lists are in the
[plan](grid-conditions-plan.md).
