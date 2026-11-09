# Fundamentals data

Part of [Battery arbitrage on the Texas grid](../README.md).

What moves real-time prices away from day-ahead prices is news that arrives after the DA bids
close at 10:00 the day before. Load comes in higher than forecast, the wind drops, clouds roll
over West Texas. ERCOT publishes its own forecasts of load, wind, and solar every hour.
`batt.data.fundamentals` keeps every one of those publications with its publish time:

| Series | ERCOT report | Published |
|---|---|---|
| `load_forecast` | Seven-Day Load Forecast by Model (NP3-565-CD): system total of the model in use | hourly at ~:30 |
| `wind_forecast`, `wind_actual` | Wind Power Production, Hourly Averaged (NP4-732-CD): STWPF and actual output | hourly at ~:55 |
| `solar_forecast`, `solar_actual` | Solar Power Production, Hourly Averaged (NP4-737-CD): STPPF and actual output | hourly at ~:55 |

Each publication keeps its next 48 hours. Actuals are public only from their first posting,
about 55 minutes after the hour ends. A decision at time *t* sees only what was published by
*t*, and model training follows the same rule. The leakage tests corrupt fundamentals published
after the cutoff, just as they corrupt prices.

The forecasters `gbm_f` and `qgbm_f`, and the policy `mpc_gbm_f`, add ten features for the
target hour:

- the latest load, wind, solar, and net-load (load minus wind minus solar) forecasts
- how much each forecast has moved since the DA bid deadline (`*_rev`): news the DA price could
  not contain
- how far the latest wind and solar actuals came in from their own bid-deadline forecasts
  (`*_err`)

On synthetic data where net-load surprises drive the RT–DA spread, these features cut forecast
error by 45% (`tests/test_fundamentals.py`).

## Setup

Free, and about 10 minutes:

1. Create an account at <https://apiexplorer.ercot.com>: Sign In, then "Sign up now", and
   confirm your email.
2. In the API Explorer, open Products > **Public API** > Subscribe. Copy the **primary key**
   from your Profile.
3. Copy `.env.example` to `.env` (git ignores it) and fill in `ERCOT_API_USERNAME` (your email),
   `ERCOT_API_PASSWORD`, and `ERCOT_PUBLIC_API_SUBSCRIPTION_KEY`.
4. Run the download:

   ```bash
   batt ingest-fundamentals
   ```

   It fetches history from the ERCOT archive one report-month at a time, about 720 publications
   per report each month.
   - January 2021 to November 2026 is 213 report-months, about 50 MB. That took about 2 hours
     here; ERCOT sometimes needs over a minute to package a month. The archive goes back to at
     least 2019.
   - It is resumable: stop it any time, and a re-run skips stored months. Each file is written
     under a temporary name first, so an interrupted run never leaves half a file behind.
   - `--reports wind` (or `load`, `solar`, or a comma-separated list) fetches only some reports,
     so separate runs can download them side by side.
   - `batt ingest-fundamentals --check` tests the credentials and the archive without
     downloading anything.

Without an account, `batt ingest-fundamentals --recent` fetches the last few days from ERCOT's
public website, enough to check the pipeline. Everything lands in `data/fundamentals/`.

The 2021–2026 history is complete: every month has at least 97% of its hourly publications for
every series, and the longest gap is 4 hours, an outage on ERCOT's side. On 2024-09-26 ERCOT
renamed the columns of the wind and solar reports; gridstatus maps the old and new names to the
same label, so a month spanning that day holds each column twice, and the parser merges them.
