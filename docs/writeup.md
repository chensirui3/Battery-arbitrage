# Why nothing beat re-planning on day-ahead prices

Part of [Battery arbitrage on the Texas grid](../README.md). Every number here comes from the
README's tables or from the plans in this folder.

## The question

A battery on ERCOT, the Texas grid, earns money by charging when power is cheap and discharging
when it is expensive. This one is 100 MW / 200 MWh, at the North Hub, from January 2022 to
October 2026. With perfect foresight of real-time prices it would have earned $88/kW-year. How
much of that can an operator capture who sees only what was public at each moment, and where
does the rest go?

## The setup

- **Data.** Every real-time and day-ahead price since 2021. ERCOT's hourly load, wind, and solar
  forecasts, as published. Its real-time dispatch runs, its one-hour price look-ahead, and its
  outage reports. Each value is stored with the time it became public.
- **Decisions.** Every 15 minutes, 5 minutes before the interval starts, a linear program
  re-plans the battery: about 170,000 decisions per backtest, at about 0.45 ms each.
- **Leakage tests.** Everything published after a cutoff is replaced with noise. No decision
  before the cutoff may change, and the same test must catch the oracles that do see the future.
- **Statistics.** Confidence intervals come from resampling whole weeks. Comparisons across
  ERCOT's five trading hubs resample the same weeks at every hub, since the hubs share weather
  and scarcity events.

## A baseline that would not lose

The simplest reasonable operator re-plans every 15 minutes as if real-time prices will equal the
day-ahead prices already published. It captures 59% of perfect foresight: $52 of $88/kW-year.
Planning once a day on the same prices captures 56%. The rolling plan wins in every year,
because once tomorrow's prices post at 13:30 it carries over only the energy that pays.

## Where the rest is

Oracles measure what information is worth. Each one re-plans like the baseline but knows part
of the future:

| Knows | Share of perfect foresight |
|---|---:|
| Real-time prices for the next 4 hours | 99% |
| ... for the next hour | 82% |
| ... only where they are spikes, more than $100/MWh from day-ahead | 77% |
| ... only when spikes come, with a typical size | 66% |

So the gap is in the next few hours, and mostly in spikes: their size as much as their timing.
It is also concentrated. The 10 best days out of 1,765 carry 23% of perfect-foresight revenue.

## What was tried

**Better forecasts.** The forecasters were AR(1), LightGBM on 21 point-in-time features refit
monthly, analog ensembles, and quantile LightGBM with conformal calibration. The probabilistic
forecasts score far better than day-ahead prices: quantile LightGBM's CRPS is 40% lower in the
first hour. But re-planning on the LightGBM forecast captured 53%, below the baseline
(−$6.0/kW-year [−10.4, −2.4]). The forecasts win on small everyday deviations, and a planner
that trades on those pays about 15% in round-trip losses plus $10/MWh of wear each time. None of
them predicts when a spike will start or how high it will go.

**Planning for uncertainty.** Two-stage stochastic MPC captured 55%. After its first 15 minutes,
each scenario's plan acts as if it knew that scenario's whole future, so it holds energy for
spikes it could never time. Sharing one plan across scenarios for longer removes that false
foresight: 58% when shared for one hour, 59% for four. That is about what planning on the
average scenario earns (60%), which ties the baseline. Stochastic dynamic programming models
what the battery will
actually learn, and when, and captured 52%.

**ERCOT's load, wind, and solar forecasts.** Real-time prices move away from day-ahead on news
that arrives after the day-ahead bids close. So ten features from ERCOT's own forecasts, as
published, were added to LightGBM, in a comparison fixed in advance. The model used them, for
about 20% of its learned gain, but revenue did not change detectably: −$1.3/kW-year
[−4.6, +1.3] against the same model without them. One week decided it. In 2022 the models had
only 2021 onward to learn from, Winter Storm Uri included, and had learned that real-time prices
clear far below day-ahead whenever day-ahead is high. In the heat wave of July 8–13, 2022 they did
not. A lead found from 2024 on was then tested at the other four hubs, in a test fixed in
advance: +$0.2 [−0.6, +1.0], not confirmed.

**A spike nowcast.** Two LightGBM classifiers predict, for each interval of the next hour,
whether the real-time price will end up more than $100/MWh above or below day-ahead. The skill
is real: up-spike ROC AUC 0.96, against 0.86 for a model that sees only what day-ahead prices
knew. How probabilities become prices was chosen on July to December 2021. The test was then
fixed in writing and run once, at all five hubs: −$1.3/kW-year [−4.2, +1.3], not confirmed. The
classifier knows a spike's odds but not its size. At HB_NORTH, averaged by year, its forecast
during spikes added $50–171/MWh to day-ahead, while the spread that followed averaged $155–497. Even a perfect classifier with
typical sizes is worth only +$6.4.

**ERCOT's own look-ahead and real-time grid conditions.** At a decision, the latest settlement
price is 25 minutes old. ERCOT publishes faster signals: each dispatch run's price and reserves
within seconds, and its own price look-ahead for the next hour. The look-ahead was used directly
where it is more than $100/MWh from day-ahead, as chosen on July to December 2021, and it
lost: −$2.6/kW-year
[−6.5, +0.5] over the five hubs, and −2.6 [−4.8, −0.8] since ERCOT's December 2025 market
redesign. Its big up-moves came true 16–39% of the time, depending on the year, and came in far
smaller. As model features, the grid conditions carried 8–20% of the models' learned gain and
improved the spike classifier (average precision 0.34 → 0.44), but not revenue: +$0.7
[−0.2, +1.7] for the spike nowcast.

## The common thread

The money is in spikes, and in how long they last. Every signal tried sees at most an hour
ahead, and the planner fills the rest of its horizon with day-ahead prices. When a real spike
starts, the planner sees high prices now and lower ones beyond the hour, so it sells into the
start.

July 13, 2022 shows it twice. The spike nowcast sold most of the battery at $1,190–2,530/MWh,
and the look-ahead policy at $1,186–2,087. The price then stayed above $4,800 for four hours,
through the day-ahead peak, where the baseline sold everything. That day holds 42% of the
look-ahead's five-year loss.

Re-planning on day-ahead prices wins by ignoring partial information. That costs it the spikes
that come apart from the day-ahead peak, and spares it the ones that build into it.

## How the work was kept honest

- **Point in time.** Every value carries its publish time, every model is fit inside the
  backtest on what was public then, and the leakage tests check both.
- **Fixed before running.** Model and planner settings were fixed before each backtest ran, with
  no tuning on its results. For the spike nowcast and the look-ahead, the open choices were made
  on July to December 2021, before the test period, and each test was written down and run
  once on 2022–2026. Analysis after a test is labeled as such and reported next to the
  original.
- **Checked against the data.** Claims in the README were checked against saved results, and
  the checks found real problems in the data pipeline:
  - a download that went on without a month after its sign-in expired, so the development
    stage was run again
  - an outage report whose format change left a month's totals blank
  - one archive file that ERCOT lists but refuses to serve

  The stored look-ahead prices were also checked against ERCOT's raw file.

## What would come next

- **A forward test.** Every test so far used 2022–2026, so a new idea needs data nobody has
  looked at. That period becomes the development set, and a frozen candidate is scored on data
  from November 2026 on, as it arrives.
- **Spike duration.** How long a spike in progress will last is where every attempt failed.
- **Decision-focused learning.** Train forecasters on the revenue their dispatch earns, not on
  forecast error, since better forecast errors never earned more here.
