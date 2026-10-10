# Machine Learning Documentation

## Model

A single **Gradient Boosting** regressor predicts the spot price from 24
features. It is a histogram-based GBM in the style of LightGBM, implemented
in pure NumPy (`NumpyGradientBoosting` in `ml/gbm.py`). LightGBM and
scikit-learn's `HistGradientBoostingRegressor` cannot be runtime
dependencies: neither has musllinux wheels, and Home Assistant OS and
Container are Alpine (musl) based.

```
NumpyGradientBoosting(          create_price_model() in ml/models.py
    n_estimators      = 200     boosting rounds (trees)
    learning_rate     = 0.05    shrinkage per tree
    max_depth         = 3       splits from the root to any leaf
    max_leaves        = 31      leaves per tree (max_depth 3 allows 8)
    min_samples_leaf  = 100     training rows per leaf (about one day of slots)
    l2_regularization = 1.0     λ on leaf values
    random_state      = 42      (the fit is deterministic)
)
```

With the **cross-border model** (option, #29) this model is stage 2 of a
two-stage model: it also gets one price column per neighbouring zone, from
a stage-1 model per neighbour (see
[Cross-Border Model](#cross-border-model-29)).

These values were chosen with the [backtest](#backtesting): with the current
inputs, deeper or less regularized trees (e.g. `max_depth` 6,
`learning_rate` 0.1, `min_samples_leaf` 20) fit the noise of single
weekday/slot cells and lose about 0.05 ct/kWh 1d MAE on a 30-day window.
Every installation uses these values; there is no per-installation
hyperparameter optimization (removed in #92, see
[Retraining](#retraining)).

How it is fitted (squared loss; each tree fits the residuals of the trees
before it):

1. **Binning.** Each feature is binned once per fit into at most 255 bins: one
   bin per distinct value when there are at most 255 of them (the time
   features), otherwise quantile bins (`np.quantile`, assigned with
   `np.searchsorted`). Missing values (NaN) get a dedicated extra bin.
2. **Split finding.** For every node, one `np.bincount` over the node's rows
   gives the residual sum and row count of every (feature, bin). Cumulative
   sums score every bin boundary of every feature at once, with the gain
   `G_L²/(n_L+λ) + G_R²/(n_R+λ) − G²/(n+λ)`. There is no loop over unique
   values. Only the smaller child of a split is histogrammed; the larger
   child's histogram is the parent's minus the smaller one's.
3. **Tree growth.** Trees grow leaf-wise: the leaf with the largest gain is
   split next, until the tree has `max_leaves` leaves, no leaf may deeper
   than `max_depth`, or no split keeps `min_samples_leaf` rows on both sides
   and reduces the loss. Leaf value = `learning_rate × G / (n + λ)`.
   Depth > 1 lets the model learn interactions (e.g. low wind **and** evening
   peak → price spike) that a sum of single-split trees cannot.
4. **Missing values.** Each split is scored twice, with the NaN rows sent left
   and right, and keeps the better direction as the node's default. A split
   whose node had no NaN rows sends NaN to the child with more training rows.
   NaN is never replaced by 0 inside the model.

A full fit on 180 days × 96 slots × 40 features (17,280 rows, 200 trees)
takes 0.6 s with the production hyperparameters and 1.1 s with `max_depth`
6 / 31 leaves / `min_samples_leaf` 20 (one aarch64 core, NumPy 2.5). The
old stump model searched every unique value of every feature and would need
hours for the same fit. Prediction walks all rows through each tree at once,
so `_generate_predictions` predicts every slot of a forecast in one call.

## Retraining

The model is retrained on all accumulated historical data **when its training
inputs have changed**, not on a fixed schedule (`ml/retraining.py`). Every
forecast run (startup, every 6 hours, and as soon as the 13:00-18:00
tomorrow-price check or the 15-minute update sees tomorrow's prices complete)
first stores today's known prices, then
compares two UTC timestamps:

- `last_data_update` — the newest of:
  - today's price-history entry being added or changed (a new day, a price
    correction, or tomorrow's prices extending the day)
  - a Nordpool prognosis row whose values changed (re-sending identical
    prognoses does not count)
  - an Open-Meteo zone weather row whose values changed (a revised or
    backfilled forecast)
  - with the cross-border model (#29), a neighbour's stored price or zone
    weather row that changed, so stage 2 retrains whenever any stage-1
    input changes
- `last_trained_at` — when the last successful training **started**, so data
  written during a training run triggers the next one

The model retrains if it is untrained or `last_data_update > last_trained_at`;
otherwise the existing model is reused. Each forecast run refreshes the
zone weather forecast, which Open-Meteo revises every few hours, so in
practice each scheduled run retrains; without new data (e.g. two runs back to
back) it does not. Forecast runs are
serialized so two retrains never overlap.

Trained trees, including the stage-1 models of the cross-border model, are
kept in memory only, so the first forecast after a restart always retrains
from the persisted history. The log line
`ML model trained in … s: holdout MAE=…` gives each training's duration.

**No hyperparameter optimization** (#92). Until #92 a weekly grid search
(`n_estimators` ∈ {100, 200, 300}, `learning_rate` ∈ {0.05, 0.1, 0.2},
`max_depth` ∈ {2, 3, 4, 6}) replaced the defaults per installation. It fitted
each candidate on the oldest 80 % of the window and scored the newest 20 %:
a month-long extrapolation (36 days at 180 days), not the 1-7 day forecast
the live model makes after every retrain. It chose deep, fast-learning
trees that forecast worse than the defaults. DK1, 180-day window, 365 daily
origins (2025-09-24 to 2026-09-23), zone weather, MAE / RMSE in EUR ct/kWh:

| Parameters                                       | 1d              | 2d              | 3d              |
| ------------------------------------------------ | --------------- | --------------- | --------------- |
| production defaults (200, 0.05, depth 3)         | **2.31 / 3.50** | **2.43 / 3.64** | **2.49 / 3.72** |
| a live DK1 instance's choice (300, 0.2, depth 6) | 2.36 / 3.56     | 2.55 / 3.81     | 2.61 / 3.92     |

A rolling-origin validation (4 origins in the window's last 2 weeks, each
fitted on the rows before it and scored on the next 1-3 days, defaults
kept unless beaten by 1 %) was tried as the replacement. Re-run at every
weekly origin of the same year (53 origins, 180-day window), it chose about
20 different parameter sets and was worse at 1d than the defaults (2.30 vs
2.22 MAE; 2d 2.31 vs 2.34, 3d 2.32 vs 2.37): a few days of validation
are too noisy to tune on. The optimization was removed instead, so every
installation uses the backtest-chosen `create_price_model()` defaults.
At startup the old results (`hpo_n_estimators`, `hpo_learning_rate`,
`hpo_max_depth`, `hpo_best_mae`, `hpo_counter`; `OBSOLETE_HPO_META_KEYS`)
are deleted from `meta` and never applied. Change the defaults only with a
backtest.

## Feature Vector (26 features)

| #   | Feature                    | Source               | Description                                                 |
| --- | -------------------------- | -------------------- | ----------------------------------------------------------- |
| 0   | `day_of_week`              | Time                 | 0=Mon, 6=Sun                                                |
| 1   | `is_weekend`               | Time                 | 1 if Saturday/Sunday                                        |
| 2   | `holiday`                  | Calendar             | 1 on Sundays/public holidays; share of states               |
| 3   | `slot_sin`                 | Time                 | sin(2π × local minute of day / 1440)                        |
| 4   | `slot_cos`                 | Time                 | cos(2π × local minute of day / 1440)                        |
| 5   | `morning_peak`             | Time                 | Seconds from 08:00 local time (negative before)             |
| 6   | `sun_elevation`            | Sun (zone centre)    | Sun elevation at the slot's middle (degrees)                |
| 7   | `sun_azimuth`              | Sun (zone centre)    | Sun azimuth at the slot's middle (degrees)                  |
| 8   | `since_sunrise`            | Sun (zone centre)    | Seconds from the day's sunrise to the slot                  |
| 9   | `since_sunset`             | Sun (zone centre)    | Seconds from the day's sunset to the slot                   |
| 10  | `consumption_forecast`     | Nordpool prognosis   | Demand prognosis for the slot's hour (MW)                   |
| 11  | `solar_generation`         | Nordpool prognosis   | Solar prognosis at the slot's hour start (MW)               |
| 12  | `wind_offshore`            | Nordpool prognosis   | Offshore wind prognosis, same hour start (MW)               |
| 13  | `wind_onshore`             | Nordpool prognosis   | Onshore wind prognosis, same hour start (MW)                |
| 14  | `net_demand`               | Derived              | consumption - solar - offshore - onshore (MW)               |
| 15  | `wind_share`               | Derived              | (offshore + onshore) / consumption                          |
| 16  | `load_forecast`            | ENTSO-E (API key)    | Week-ahead load forecast curve for the slot (MW)            |
| 17  | `gas_price`                | Instrat (#28)        | Gas price known before the slot's day (PLN/MWh)             |
| 18  | `unavailable_production`   | UMM / ENTSO-E (#123) | Production capacity announced unavailable for the slot (MW) |
| 19  | `unavailable_transmission` | UMM / ENTSO-E (#123) | Interconnector capacity announced unavailable (MW)          |
| 20  | `zone_wind`                | Open-Meteo zone      | Mean wind at 80 m over the zone's points (m/s)              |
| 21  | `zone_wind_power`          | Derived              | Mean power curve of the points' 80 m wind, 0-1              |
| 22  | `zone_temperature`         | Open-Meteo zone      | Mean temperature at 2 m (°C)                                |
| 23  | `zone_irradiance`          | Open-Meteo zone      | Mean global horizontal irradiance (W/m²)                    |
| 24  | `zone_pressure`            | Open-Meteo zone      | Mean sea-level pressure (hPa)                               |
| 25  | `zone_humidity`            | Open-Meteo zone      | Mean relative humidity at 2 m (%)                           |

Column order is `FEATURE_NAMES` in `ml/features.py`. With the cross-border
model (#29, option) the price model's input has one more column per
neighbour after these 26: `cross_price_<zone>`, stage 1's price for the
slot (see [Cross-Border Model](#cross-border-model-29)). The 26 stay the
canonical vector; stage 1 uses them too.

**Time of day and the sun** (#25). Since October 2025 the day-ahead market
clears every 15 minutes, and prices often step within an hour, so the time
features are per slot, not per hour: all four slots of an hour differ.
They follow the local wall clock, so 17:00 has the same values in winter,
in summer and on a DST day, and both passes of the repeated fall-back hour
share them. `slot_sin`/`slot_cos` place the slot on the 24-hour circle
(23:45 is next to 00:00); `morning_peak` is the time of day as a signed
distance to the 08:00 demand peak, in local time (EpexPredictor computes its
peak offsets in UTC, where they move by an hour at every DST change).
EpexPredictor also has an offset to 19:00; in local time that is
`morning_peak` shifted by 11 hours, which a tree model splits identically
(the backtest numbers are the same to two decimals), so it is not a
feature. Neither is `hour` any more: every hour boundary is a
`morning_peak` split too.

Solar production follows the sun rather than the clock, and sunrise moves by
over four hours between June and December in Denmark. The sun features
(`ml/sun.py`) describe the sun at the **zone centre**, the mean of the
region's `WEATHER_POINTS`, at the slot's middle: its elevation and azimuth,
and the seconds since that local day's sunrise and sunset (negative before
them). They are computed with `astral`, the library behind Home Assistant's
`sun.sun` entity, which ships with Home Assistant (no new dependency).
`sun.sun` itself cannot be the input: it only holds the current position and
the next events at the home location, while training needs every slot of the
window and prediction the next week. A day without sunrise or sunset (polar
day or night) has NaN for those two. The values depend only on the slot and
the region, so they are cached across retrains.

**Public holidays** (#26). On a public holiday offices and most industry
are closed, so demand and the price behave like on a Sunday, while
`day_of_week` says Thursday. `holiday` (`ml/public_holidays.py`) is 1 on
Sundays and public holidays and 0 on other days, from the slot's local date.
Sundays count so the model learns the effect from the Sundays in its window
and can apply it to a weekday holiday: a 60-day window often has no weekday
holiday at all. Where public holidays differ within the bidding zone, it is
the share of the zone's subdivisions with a holiday that day
(`HOLIDAY_SUBDIVISIONS` in `const.py`: Germany's 16 states, e.g. 9/16 on
Reformation Day). Other regions use their national calendar; France's
subdivisions in the package are overseas territories, outside the bidding
zone. Christmas Eve and New Year's Eve are at least 0.5, as half days. The
calendars come from the [`holidays`](https://pypi.org/project/holidays/)
package, the one Home Assistant's `workday` and `holiday` integrations use
(declared in `manifest.json` with a loose lower bound, so it resolves to
Home Assistant's version). They are built once per country and year, only
in the executor, where every feature row is built.

**Load forecast** (#30). Nordpool's consumption prognosis only exists for
today and tomorrow, so days 3-7 of the forecast had no demand input at all.
With an ENTSO-E API key (optional, in the config or options flow),
`load_forecast` is ENTSO-E's **week-ahead total load forecast** for the
bidding zone (`documentType` A65, `processType` A31): the forecast minimum
and maximum load of each day of the coming week. As in EpexPredictor, the
two values become a 15-minute curve (`load_curve()` in `api/entsoe_load.py`):
the minimum at 03:00, the maximum at 11:30 and 19:00 and
`(3 × max + min) / 4` at 14:30 local time, joined by a natural cubic spline
(NumPy; no pandas or SciPy). It is a separate feature rather than filling
`consumption_forecast` after Nordpool's horizon: the stored table
(`entsoe_load`) holds the week-ahead forecasts ENTSO-E published for the
training window's days too, so training and prediction see the same
source, as #17 and #23 require. Without a key, and in DE and NL (where
EpexPredictor's backtests found it hurts; `ENTSOE_LOAD_REGIONS` in
`const.py`), it is NaN in every row. A week-ahead forecast published once
a week may not reach day 7; those slots are NaN too.

ENTSO-E answers **one week-ahead document per request**, the ISO week of
`periodStart` (clipped to it), whatever `periodEnd` says, so the source
requests whole ISO weeks (`week_chunks()` in `time_series.py`, Monday to
Monday local time) and, for the curve's edges, the weeks on each side, each
week once per update (#114). The documents are variable sized blocks
(`curveType` A03): a day whose value equals the day before is left out and
`parse_entsoe_load()` repeats the previous value. A day published as `0`
(no forecast; DK1 had them on the Monday and Sunday of recent weeks) is
treated as missing, so the curve splits there instead of dipping below
zero. ENTSO-E also has whole weeks without data (e.g. DK1's weeks of
2026-09-07 and 2026-09-28); the feature is NaN for them.

**Gas price** (#28). Gas-fired plants often set the marginal price, so
the gas price level moves the electricity price. `gas_price`
(`ml/gas_price.py`) is the latest daily gas price dated **before the
slot's local day**: the price known the day before, so a training row
never sees its own day's price and a forecast days ahead gets the latest
published one, as in training. A price older than 14 days
(`GAS_LOOKBACK_DAYS`) is not used, so a source that stopped updating gives
NaN rather than a stale level. The source is Instrat's JSON API
(`api/gas_prices.py`, CC BY-NC 4.0): the TGE (Polish exchange) gas
day-ahead index in PLN/MWh, every day including weekends. It is a proxy for
the European gas price level; the German THE Day Ahead price, which
EpexPredictor scrapes from a Bundesnetzagentur page, scored the same in the
backtest, and scraping HTML breaks when the page changes. Only a level
matters to a tree model, so neither the currency nor the hub needs
converting. The regions in `GAS_PRICE_REGIONS` (`const.py`) fetch it; the
[backtest](#gas-price-28) chose them. Elsewhere, or when the source fails,
it is NaN.

**Zone weather** (#22). The local weather entity is one place, at 10 m,
about 48 hours ahead and hourly. The zone features describe
the whole bidding zone instead: Open-Meteo's 15-minute forecast at a few
fixed points per region (`WEATHER_POINTS` in `const.py`: wind and demand
centres and, where there is one, an offshore wind area; DK1 for example
samples North, West and South Jutland and Horns Rev), for 8 days ahead. They
are zone aggregates rather than one column per point, so every region has
the same vector whatever its number of points: the mean of each value, and
the mean turbine power curve of the points' wind (calm and windy points are
not averaged into a medium wind). The aggregation is `ZoneWeatherIndex`
(`ml/zone_weather.py`), the same in training and prediction. The backtest
chose aggregates over per-point columns (see [Backtesting](#backtesting)).

**One definition for training and prediction** (#17). Every row, for
training, prediction and the backtest, is built by
`build_feature_row(slot_start, SlotInputs, region)` in `ml/features.py` and turned
into the model input by `build_feature_vector()`. The two phases differ only
in where a slot's `SlotInputs` come from (see
[Training vs Prediction Segmentation](#training-vs-prediction-segmentation)).
Time features (0, 1, 3-5) come from `slot_time_features()`, `holiday` (2)
from `public_holiday()` with the region's calendar and sun features (6-9)
from `sun_features()` with the region's `zone_centre()`; derived features
(14, 15, 19) are computed from the slot's own inputs.

**Missing inputs are NaN.** An input that is unknown for a slot (no zone
weather stored for it, Nordpool prognoses only exist for today and
tomorrow, no ENTSO-E key) is `None` in the feature dict
and NaN in the model input, and so is every derived feature that needs it.
The price model handles NaN natively (see [Model](#model)). Nothing is
replaced by 0, 15 °C, 50 % humidity or the current observation. Rows whose
target price is missing are not training rows.

**Removed in #23**: the local weather entity's `wind_speed_mean`,
`wind_power_estimate`, `wind_direction`, `cloud_coverage`, `humidity` and
`temperature`. They were trained on measured weather (`weather_history`
snapshots) but predicted from the entity's forecast, so the model learned
"price given perfect weather knowledge" and was fed forecasts with an error
it had never seen. The entity has no archive of past forecasts to train on
instead, and the zone weather covers the same ground (and the whole zone).
Prediction rows still carry the entity's forecast in the feature dict,
where `store_prediction_for_learning` records it for the forecast-accuracy
score (see [Confidence Score](#confidence-score)); it is not a model input.

**Removed in #17**, because they meant different things in training and
prediction:

- `price_mean`: constant over all training rows (the mean of all history),
  but the mean of today's and tomorrow's prices at prediction. The previous
  day's mean price was tested as a consistent replacement: it lowered the
  30-day 1d MAE from 3.27 to 3.17 ct/kWh but raised 2d/3d MAE from 3.30/3.31
  to 4.16/4.13, because that day is not yet known two or more days ahead. It
  was not kept.
- `solar_radiation_mean` and `solar_power_estimate`: the inverter's
  instantaneous output (W) in training, Solcast's daily kWh estimate
  (constant over all slots) at prediction. The configured Solcast sensor
  only covers today, and predictions start where confirmed prices end
  (tomorrow or later), so a per-slot Solcast value would be unknown in every
  prediction row. The model's solar input is Nordpool's per-slot solar
  prognosis (`solar_generation`), the same source and unit in both phases.
  Irradiance came with #22 (`zone_irradiance`).

**Outages** (#123). A planned or unplanned outage of a large plant, or a
limitation on an interconnector, moves the price more than anything in the
weather or the prognoses, and it is not periodic, so bias correction cannot
learn it either. Nord Pool's **urgent market messages** (UMM API,
`ummapi.nordpoolgroup.com`, JSON, no key) announce them per unit with the
unavailable capacity in time periods. `unavailable_production` is the MW of
production and generation units in the region's area announced unavailable
for the slot, `unavailable_transmission` the MW of interconnector capacity
into or out of it (both directions, every connection of the area), each
the sum over the messages in force. Messages are revised in numbered
versions and can be dismissed, and the API serves every version with its
publication time, so the stored archive keeps them all (`umm_messages`,
`umm_periods`, `api/nordpool_umm.py`) and `OutageIndex` (`ml/outages.py`)
aggregates them **as known at an origin**: the latest version of each
message published by then, nothing for a dismissed one. Training rows use
the slot's day-ahead gate (12:00 CET the day before: what the auction that
set the price knew), prediction rows the forecast run's time, so a
revision published after the fact never reaches a training row. With
messages stored a slot nothing is announced for is 0; before the first
fetch, and outside `UMM_REGIONS`, the two features are NaN in both phases.
Nord Pool publishes UMMs for its own delivery areas (`UMM_AREAS`: DK, SE,
NO, FI and the Baltics), and the backtest can measure every one of them, but
the model uses them only where the backtest found they help: DK1 (see
[Nord Pool UMM outages](#nord-pool-umm-outages-123); in DK2 they lower the
day-1 error and raise the error at days 2-7).

DE, NL, BE and FR publish their unavailability on the **ENTSO-E
Transparency Platform** instead (#138, `api/entsoe_outages.py`, with the
ENTSO-E key): production and generation unit unavailability (`A77`/`A80`,
per bidding zone) and transmission unavailability (`A78`, per border and
direction, `ENTSOE_OUTAGE_BORDERS`), a ZIP of XML documents, at most 200 per
request (paged with `offset`, a window past the offset limit is halved).
`EntsoeOutageSource` stores them in the same tables through the shared
`OutageSource` cycle, so `OutageIndex` and the two features do not change:
a document is a message version (`mRID`, `revisionNumber`,
`createdDateTime` as its publication, `docStatus` A09 cancelled / A13
withdrawn as dismissed), and a plant's `Available_Period` points give the
unavailable MW as `nominalP − available`, one row per point segment. Two
differences from the UMMs: the platform publishes **no nominal capacity for
a grid asset** (only its available capacity), so in these zones
`unavailable_transmission` is the **number of transmission assets under a
limitation** (1.0 per asset, segments at the document's highest quantity
count nothing), not MW; and it serves **only the current revision** of a
document, so the first fetch stores each document once, at its latest
revision's publication time, and the archive gains the earlier versions of
later revisions only as the updates store them (a training row sees a
document from the stored revision's publication on — a conservative view,
never a leaked one). The model uses them in `ENTSOE_OUTAGE_REGIONS`, where
the backtest found they help (see
[ENTSO-E outages](#entso-e-outages-138)).

**Tested and not kept** (#119): lagged prices. The vector has no realised
price in it, and the naive "same slot last week" baseline was within
0.2 ct/kWh of a time-only model, so #119 tried origin-relative lags that
never leak the target: the same wall-clock slot on the slot's **reference
day** (the last day with known prices before the slot's day: the day before
a training row's day, the last known day at the origin for every forecast
day), the mean price of the week ending on it, the same slot one week
earlier, and a variant whose training rows lag 1-7 days with the lag's age
as a feature. The backtest ([Lagged prices](#lagged-prices-119)) found that
yesterday's slot price lowers the 1-day error by 4-6 % but raises the
3-7-day error, because the model learns from one-day-old lags and is given
up to seven-day-old ones, and that no variant lowered the MAE at every
horizon in DK1 and DK2. The experiment stays reproducible in the dev
backtest (`--lags`, `scripts/backtest_lags.py`); the integration has no
lagged price features.

## Cross-Border Model (#29)

European day-ahead markets are coupled: an interconnector pulls a zone's
price towards its neighbours' until it is congested. DK1 is linked to
Germany, the Netherlands (COBRA), NO2 (Skagerrak), SE3 (Konti-Skan) and DK2
(Great Belt), so a wind lull in Germany raises the DK1 price even when
Jutland is windy. The single-stage model only sees DK1's own inputs.

With the option **Cross-border model** (options flow, off by default; for
DK1 and DK2, the regions in `NEIGHBOURS` in `const.py`: DK1 → DE, NL, NO2,
SE3, DK2; DK2 → DK1, DE, SE4) the model is trained in two stages, after
EpexPredictor (`pricepredictor.py` `get_cross_features`, BSD-3-Clause,
reimplemented in `ml/cross_border.py`):

1. **Stage 1**: one price model per neighbour (`Stage1Model`), fitted on the
   neighbour's day-ahead prices (`neighbour_prices`, raw EUR/MWh) and rows
   from `build_feature_row` for the neighbour: its local time, holidays,
   sun position and Open-Meteo zone weather at its `WEATHER_POINTS`
   (`neighbour_feature_rows`). Nordpool and ENTSO-E inputs only exist for
   the region itself and are NaN. It is the production price model
   (`create_price_model()`).
2. **Stage 2**: the region's price model, with a `cross_price_<zone>` column
   per neighbour after the 26 features: stage 1's price for the slot.

**Stage-1 values in training rows are out of sample.** A model's fitted
values are much closer to the actual prices than its forecasts: on DK1's
neighbours (60-day window, weekly origins, mean over the five), the
stage-1 error on its own training rows is 1.3 ct/kWh, while the same
model's next-day forecast is off by 2.1 and an out-of-sample value (four
interleaved folds) by 1.9. Trained on fitted values, stage 2
would learn to trust the column more than any forecast deserves. So each
neighbour's stage 1 is `STAGE1_FOLDS` = 2 models, fitted on interleaved
local days (`local_day_fold`: the day's ordinal modulo 2): each training
row gets the price of the model that did not see its day. Prediction rows
get the mean of the two models, so no separate full fit is needed: two
half-size fits cost about one full one. Interleaved days rather than
contiguous blocks, because a block model extrapolates to the ends of the
window (see the backtest below).

Stage 1 reads the neighbours' prices and weather from storage in both
phases, like the region's own inputs: archived forecasts for past days, the
live forecast ahead. The forecast run refreshes the neighbours' prices up to
tomorrow and their weather from yesterday to the forecast's end; the
history backfill fills the training window; retention prunes them with the
rest (see [persistence](persistence.md#retention)).

**Missing neighbour data is NaN.** A neighbour with less than a week of
prices in a fold's training days (`MIN_STAGE1_ROWS`), or whose data fails to
load, has no stage-1 model and a NaN column, which the price model handles.
A new install therefore predicts at once and gains the columns when the
backfill has fetched the neighbours' history (which triggers a retrain).

**Cost.** Each training fits `2 × neighbours` extra models on the training
window and builds the neighbours' feature rows. With synthetic full
history, a DK1 training (stage 1, stage 2 and the holdout copy) takes 3.3 s
at a 60-day window and 9.6 s at the default 180 days on one aarch64 core of
the development container with the option, against 0.45 s and 1.1 s without
it (see [Training Window](#training-window) → Footprint). It was not
measured on a Raspberry Pi: a Pi 4 core is roughly 5-10 times slower, so
expect 50-100 s per training there at the default window, in the executor,
a few times a day (whenever an input changed); the
[live-accuracy procedure](#live-accuracy-94) measures it from the log. Prediction
only adds the stage-1 forecasts of the forecast week's slots. The database
grows by about 19 MB at 60 days and 60-70 MB at 180 days: the neighbours'
prices and their weather (DK1: 23 weather points besides its own 4). The
option is off by default
because of that cost and the extra requests (energy-charts and Open-Meteo,
per neighbour), not because of accuracy.

## Data Sources

| Source                                                | Type                | Resolution    | Used for                                                   |
| ----------------------------------------------------- | ------------------- | ------------- | ---------------------------------------------------------- |
| `sensor.stromligning_spotprice_ex_vat` (+ tomorrow)   | Raw spot price      | 15-min        | Training target, self-learning actuals (excl. VAT)         |
| `dayahead_prices` (SQLite, #27, #24)                  | Raw spot price      | 15-min        | `dayahead` source; the training window's history (all)     |
| `weather.get_forecasts`                               | Weather forecast    | Hourly        | Recorded with predictions (forecast accuracy)              |
| `weather.forecast_*` (state)                          | Current weather     | Every 15 min  | `weather_history` snapshots (forecast accuracy)            |
| `Nordpool Consumption API`                            | Demand forecast     | Hourly        | Market demand prognosis (MW), both phases                  |
| `Nordpool Production API`                             | Generation forecast | 15-min        | Solar, wind offshore/onshore (MW), both phases             |
| `sensor.solcast_*`                                    | Solar forecast      | Daily total   | Solar scaling factor only (not a model input)              |
| `sensor.power_inverter_*`                             | Actual solar        | Scalar        | Solar scaling factor only (not a model input)              |
| `weather_history` (SQLite)                            | Actual weather      | 15-min        | Scores the local forecast (confidence); not training       |
| `nordpool_prognoses` (SQLite)                         | Stored prognoses    | Hourly        | Training inputs                                            |
| Open-Meteo (`api.open-meteo.com`, #22)                | Zone weather        | 15-min        | `openmeteo_weather`: zone features, both phases            |
| Open-Meteo archive (`historical-forecast-api`, #23)   | Past zone forecasts | 15-min        | `openmeteo_weather` days before yesterday (training)       |
| ENTSO-E week-ahead load (A65/A31, API key, #30)       | Load forecast       | Daily min/max | `entsoe_load` curve: `load_forecast`, both phases          |
| Nord Pool UMM API (`ummapi.nordpoolgroup.com`, #123)  | Outage messages     | Per message   | `umm_messages`/`umm_periods`: `unavailable_*`, both phases |
| ENTSO-E outage documents (A77/A80/A78, API key, #138) | Outage documents    | Per document  | `umm_messages`/`umm_periods`: `unavailable_*`, DE/NL/BE/FR |
| Instrat TGE gas day-ahead index (#28)                 | Gas price           | Daily         | `gas_prices`: `gas_price`, both phases                     |
| `holidays` package (#26)                              | Public holidays     | Daily         | `holiday` feature, both phases                             |
| energy-charts.info, neighbouring zones (#29)          | Raw spot price      | 15-min        | `neighbour_prices`: stage-1 targets (cross-border)         |
| Open-Meteo at the neighbours' points (#29)            | Zone weather        | 15-min        | `openmeteo_weather`: stage-1 inputs (cross-border)         |

Wind speed is converted to m/s from the weather entity's `wind_speed_unit`
(default km/h) by `wind_speed_to_ms()` in `sensor_reader.py`, for the stored
snapshots and for the hourly forecast alike.

## Nordpool Prognoses

Two public APIs (no authentication) provide the market's own forecasts:

- **ConsumptionPrognoses**: Hourly demand forecast per delivery area
- **ProductionDataPrognoses**: 15-min generation forecast per type (Solar, WindOffshore, WindOnshore)

These are the same inputs used by market participants. They're stored in
`nordpool_prognoses` (one row per hour: the hour's consumption and the
production of its first quarter) by a gap-aware source (see
[persistence](persistence.md#time-series-sources)): each prediction run
re-fetches today's and tomorrow's delivery days (Nordpool revises them), and
a background task fills the missing days in `price_history` at startup and
after midnight. Without a login Nordpool only serves the last 7 delivery
days (older days answer 401), so the backfill reaches back
`NORDPOOL_HISTORY_DAYS` (6) days; older days only have the prognoses
earlier runs stored, and the window fills up as the integration runs.
Training reads
the stored rows; prediction reads today's and tomorrow's stored rows at the
same resolution: the hour's consumption and the production at the hour's
start, for all four slots of the hour.

**Availability differs between the phases** (#91). Almost every training
row has a stored prognosis once the integration has run for the whole
window (about 98 % of the rows of a 30- or 60-day window; a new install
only has the last 6 days, see above), but almost no prediction row
does: prognoses exist for today and tomorrow only, and a forecast starts
where the known prices end (`first_prediction_slot`). Once tomorrow's
prices are published (~13:00) the first predicted slot is the day after
tomorrow, so no predicted slot has one; before that, only tomorrow can
have one, if Nordpool has already published it. Training therefore adds
a **masked copy** of every row with a prognosis: the six Nordpool features
(`NORDPOOL_FEATURES`: 10-15) NaN, the same target price
(`with_masked_nordpool()` in `ml/features.py`). The model learns both
branches: without prognoses it matches a model without the features, and
it keeps most of the gain on day 1 of a morning forecast that has them
(see [Backtesting](#nordpool-prognoses-at-prediction-91)).

Derived features:

```
net_demand = consumption - solar - wind_offshore - wind_onshore
wind_share = (wind_offshore + wind_onshore) / consumption
```

## Training vs Prediction Segmentation

Training and prediction both use **forecasts** (#23): the model learns the
price from the same kind of weather input it is later given, with a similar
error, rather than from measured weather it never sees at prediction.

| Phase          | Zone weather (#22, #23)                                        | Nordpool source                                                |
| -------------- | -------------------------------------------------------------- | -------------------------------------------------------------- |
| **Training**   | `openmeteo_weather`: archived forecasts, and the last live one | `nordpool_prognoses` row for the hour, and a masked copy (#91) |
| **Prediction** | `openmeteo_weather`: the current live forecast                 | Live prognoses for the slot's hour (today and tomorrow only)   |

The gas price (#28) is stored per day in `gas_prices` (backfilled with the
training window plus 14 days, refreshed at every forecast run); both phases
take the latest price dated before the slot's day from it.

The ENTSO-E load forecast (#30) works the same way: `entsoe_load` holds, for
the training window's days, the week-ahead forecasts ENTSO-E published for
them (backfilled with the rest of the history), and from yesterday on the
latest forecast, re-fetched at every forecast run.

The outage messages (#123) are stored as **every published version** of
every message (`umm_messages`, `umm_periods`), so both phases rebuild what
was known at their origin: a training row takes the versions published by
its day's day-ahead gate (12:00 CET the day before, `day_ahead_gate()`),
a prediction row those published by the forecast run. Neither phase sees a
later revision or dismissal.
ENTSO-E's outage documents (#138) fill the same tables for DE, NL, BE and
FR; the platform only serves a document's current revision, so each
document enters at its latest revision's publication time and the archive
grows the version history from then on.

The zone weather is one stored table for both phases:

- Days before yesterday come from Open-Meteo's **archive of past forecasts**
  (`historical-forecast-api.open-meteo.com`), requested in whole UTC days,
  at most 90 per request. The background backfill fills the training window
  at setup and after midnight, so the model has zone weather for every
  training day from the first day on.
- From yesterday to 8 days ahead each forecast run re-fetches the **live
  forecast** (`api.open-meteo.com`); a past slot keeps the last forecast
  fetched for it.

The local weather entity is no longer a model input (see
[Feature Vector](#feature-vector-26-features)). Its snapshots
(`weather_history`) only score its forecast: the confidence's
forecast-error penalty compares the forecast recorded with a prediction
with the snapshot taken in the slot. They do not trigger a retrain.

Training reads both tables once per fit (`TrainingInputs` in
`ml/training_inputs.py`) and matches rows to slots on their UTC epoch.
Prediction matches prognoses on the slot's UTC hour, so Nordpool's UTC
timestamps line up with local slot times. Training never uses the current
forecast's values.

## Slot Timestamps and DST

A local day has 96 slots, but 92 on the spring-forward day (02:00-02:59 is
skipped) and 100 on the fall-back day (02:00-02:59 happens twice). Slot times
are therefore never built as `date + n × 15 min` in local wall-clock time:

- **Training rows** (`get_all_historical_prices`): slot _n_ of a stored day
  starts at local midnight converted to UTC plus _n_ × 15 min, converted back
  to local time (`slot_start_in_day()` in `time_slots.py`). Every slot after
  a DST change gets its real wall-clock time and UTC offset, so the time
  features and the `weather_history` / Nordpool lookups line up. A slot
  missing in the source (`null`, see the price grid in
  [stromligning_integration.md](stromligning_integration.md#price-grid)) is
  not a training row; the slots after it keep their times.
- **Self-learning**: the 15-minute update finds the current slot's price at
  `slot_index_in_day(now)`, its position counted the same way from local
  midnight (not `hour × 4 + minute // 15`). Predictions are matched to the
  slot's UTC instant, so the two passes of the repeated fall-back hour learn
  separately. The startup catch-up replay uses the same slot starts.
- **Clock**: `ml/` reads the time with `dt_util.now()` (Home Assistant's time
  zone), never the naive `datetime.now()`.

The bias-correction and error-metric slots stay keyed by local time of day
(0-95), so both passes of the repeated hour share their wall-clock slots.

## Training and Validation Split

`_train_models` builds one row per 15-minute slot of `price_history` (the
training window, see [Training Window](#training-window)), oldest first, and
uses the rows twice:

1. **Holdout validation.** A copy of the price model with the same
   hyperparameters is fitted on the oldest 80 % of the rows and scored on the
   newest 20 %. The holdout MAE and RMSE are logged
   (`ML model trained: holdout MAE=…, RMSE=…`) and kept with the training
   time in `meta` (`holdout_mae`, `holdout_rmse`, `holdout_trained_at`), so
   the learning-metrics sensor shows them across restarts (see
   [Self-Learning](self_learning.md#metrics-available)); the copy is then
   discarded. A failed training deletes them, so they never describe a model
   that is not in use.
2. **Live model.** `price_model` is fitted on 100 % of the rows. The most
   recent days are the most similar to the days being predicted, so they
   must be part of the model: fitting on the oldest 80 % only would ignore
   the newest ~36 of 180 days. The backtest's `current` row (see
   [Backtesting](#backtesting)) also fits on its whole window.

The split is chronological, never shuffled, so the holdout rows are always
later than the rows the copy was fitted on, as in a real forecast. The
Nordpool-masked copies (#91, see [Nordpool Prognoses](#nordpool-prognoses))
are added **after** the split, to each side separately, so both copies of a
row are on the same side; the holdout error covers the holdout rows with
and without their prognoses. The live model fits all rows and all their
copies. The copies roughly double the rows and the training time. With the
cross-border model the holdout copy gets the same stage-1 columns as the
live model: out of sample per day, but from stage-1 models fitted on the
whole window, so the neighbours' prices around the holdout days are
known to stage 1 and the holdout error is slightly optimistic. The extra
fit roughly doubles training time. Like the rest of training, it runs in the
executor.

## Training Window

The model trains on the last `training_days` of prices (option
**Training days**, 30-180, default **180**, #24, #121): `max_history_days`
on the predictor. The window also sets what is backfilled and how much
history is kept (window + 2 days, see
[persistence](persistence.md#retention)).

**Backfill.** At setup and after midnight a background task fills the
window's missing days, whatever the displayed price source: the day-ahead
spot price from energy-charts (ENTSO-E fallback, #27) is the same series as
Stromligning's `spotprice_ex_vat` sensor, converted with the day's ECB rate.
Then the zone weather (Open-Meteo's archive, #23) and the Nordpool
prognoses of those days are fetched, and the forecast is refreshed, so a new
install trains its ML model within minutes of setup instead of starting
from the heuristic. The sources only request what is missing, so an
interrupted backfill resumes where it stopped and a repeat is a no-op. The
heuristic stays the fallback while the backfill has not delivered (e.g. no
network).

**Why 180 days (#121).** The default was 60 days from the #24 sweep (zone
weather, no sun features, days 1-3 scored): 60 days was then the best
window at every horizon and 180 days (EpexPredictor's window) 0.13 ct/kWh
worse at 1d, because the model had no seasonal feature and older days
mostly added prices from another price level. The sun features (#25)
narrowed that to 0.03 at 1d and reversed it at 2d and 3d. The sweep below
re-ran the choice with the current feature set, on DK1 and DK2, 365 daily
origins from 2025-09-24 to 2026-09-23, retrained daily, `--horizon-days 7`,
NumPy GBM. Cells are MAE / RMSE in EUR ct/kWh; the last column is the mean
MAE over days 2-7 (after 13:00, day 1 is mostly known prices) and decides
the default. Four configurations: the single-stage model without and with
the gas price (`--gas`; on in production for every region in
`GAS_PRICE_REGIONS`, DK1 and DK2 included) and the cross-border model
(`--cross-border`, option off by default) without and with it. Recorded
2026-10-02.

**DK1, single-stage**

| Window | 1d          | 2d          | 3d          | 4d          | 5d          | 6d          | 7d          | 2-7d MAE |
| ------ | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | -------: |
| 30 d   | 2.32 / 3.56 | 2.52 / 3.79 | 2.60 / 3.87 | 2.65 / 3.94 | 2.64 / 3.95 | 2.67 / 3.96 | 2.72 / 4.02 |     2.63 |
| 60 d   | 2.29 / 3.46 | 2.46 / 3.66 | 2.52 / 3.73 | 2.54 / 3.77 | 2.54 / 3.76 | 2.57 / 3.79 | 2.58 / 3.81 |     2.54 |
| 90 d   | 2.30 / 3.46 | 2.44 / 3.62 | 2.49 / 3.68 | 2.51 / 3.71 | 2.53 / 3.73 | 2.55 / 3.76 | 2.57 / 3.78 |     2.52 |
| 120 d  | 2.29 / 3.46 | 2.41 / 3.60 | 2.46 / 3.67 | 2.49 / 3.70 | 2.50 / 3.71 | 2.51 / 3.73 | 2.53 / 3.76 | **2.48** |
| 180 d  | 2.31 / 3.50 | 2.43 / 3.64 | 2.49 / 3.72 | 2.51 / 3.73 | 2.52 / 3.74 | 2.53 / 3.75 | 2.55 / 3.79 |     2.50 |

**DK1, single-stage + gas**

| Window | 1d          | 2d          | 3d          | 4d          | 5d          | 6d          | 7d          | 2-7d MAE |
| ------ | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | -------: |
| 30 d   | 2.22 / 3.44 | 2.44 / 3.70 | 2.54 / 3.84 | 2.63 / 3.97 | 2.65 / 3.98 | 2.66 / 4.00 | 2.70 / 4.02 |     2.60 |
| 60 d   | 2.19 / 3.36 | 2.37 / 3.55 | 2.43 / 3.67 | 2.47 / 3.75 | 2.53 / 3.84 | 2.58 / 3.87 | 2.62 / 3.90 |     2.50 |
| 90 d   | 2.21 / 3.39 | 2.36 / 3.56 | 2.44 / 3.67 | 2.49 / 3.76 | 2.52 / 3.81 | 2.55 / 3.84 | 2.58 / 3.86 |     2.49 |
| 120 d  | 2.20 / 3.39 | 2.34 / 3.55 | 2.42 / 3.66 | 2.48 / 3.76 | 2.50 / 3.80 | 2.52 / 3.81 | 2.54 / 3.82 |     2.47 |
| 180 d  | 2.20 / 3.41 | 2.31 / 3.54 | 2.40 / 3.65 | 2.45 / 3.72 | 2.47 / 3.75 | 2.49 / 3.77 | 2.52 / 3.81 | **2.44** |

**DK1, cross-border**

| Window | 1d          | 2d          | 3d          | 4d          | 5d          | 6d          | 7d          | 2-7d MAE |
| ------ | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | -------: |
| 30 d   | 1.97 / 3.12 | 2.13 / 3.29 | 2.25 / 3.44 | 2.29 / 3.48 | 2.30 / 3.50 | 2.34 / 3.53 | 2.38 / 3.62 |     2.28 |
| 60 d   | 1.90 / 3.00 | 2.02 / 3.11 | 2.07 / 3.18 | 2.10 / 3.20 | 2.10 / 3.21 | 2.14 / 3.27 | 2.16 / 3.35 |     2.10 |
| 90 d   | 1.88 / 2.98 | 1.99 / 3.09 | 2.05 / 3.16 | 2.09 / 3.21 | 2.11 / 3.27 | 2.13 / 3.31 | 2.16 / 3.36 |     2.09 |
| 120 d  | 1.90 / 2.97 | 1.99 / 3.07 | 2.03 / 3.12 | 2.05 / 3.17 | 2.07 / 3.20 | 2.11 / 3.25 | 2.13 / 3.31 | **2.06** |
| 180 d  | 1.94 / 3.00 | 2.03 / 3.09 | 2.07 / 3.14 | 2.09 / 3.18 | 2.11 / 3.20 | 2.14 / 3.25 | 2.16 / 3.31 |     2.10 |

**DK1, cross-border + gas**

| Window | 1d          | 2d          | 3d          | 4d          | 5d          | 6d          | 7d          | 2-7d MAE |
| ------ | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | -------: |
| 30 d   | 1.92 / 3.06 | 2.09 / 3.24 | 2.22 / 3.42 | 2.26 / 3.47 | 2.27 / 3.49 | 2.33 / 3.54 | 2.37 / 3.63 |     2.26 |
| 60 d   | 1.84 / 2.92 | 1.97 / 3.05 | 2.05 / 3.16 | 2.08 / 3.20 | 2.11 / 3.25 | 2.14 / 3.30 | 2.17 / 3.37 |     2.09 |
| 90 d   | 1.77 / 2.86 | 1.89 / 2.98 | 1.97 / 3.09 | 2.01 / 3.15 | 2.05 / 3.25 | 2.09 / 3.29 | 2.12 / 3.36 |     2.02 |
| 120 d  | 1.79 / 2.85 | 1.90 / 2.98 | 1.97 / 3.07 | 2.00 / 3.13 | 2.04 / 3.20 | 2.07 / 3.24 | 2.08 / 3.29 |     2.01 |
| 180 d  | 1.79 / 2.86 | 1.88 / 2.96 | 1.94 / 3.02 | 1.98 / 3.08 | 1.98 / 3.12 | 2.01 / 3.16 | 2.04 / 3.22 | **1.97** |

**DK2, single-stage**

| Window | 1d          | 2d          | 3d          | 4d          | 5d          | 6d          | 7d          | 2-7d MAE |
| ------ | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | -------: |
| 30 d   | 2.55 / 3.85 | 2.76 / 4.10 | 2.83 / 4.16 | 2.86 / 4.21 | 2.87 / 4.23 | 2.86 / 4.23 | 2.89 / 4.27 |     2.85 |
| 60 d   | 2.52 / 3.78 | 2.69 / 3.99 | 2.73 / 4.04 | 2.75 / 4.05 | 2.76 / 4.07 | 2.76 / 4.06 | 2.79 / 4.11 |     2.75 |
| 90 d   | 2.50 / 3.73 | 2.65 / 3.90 | 2.69 / 3.95 | 2.70 / 3.97 | 2.73 / 4.00 | 2.74 / 4.00 | 2.75 / 4.03 |     2.71 |
| 120 d  | 2.52 / 3.76 | 2.65 / 3.91 | 2.68 / 3.96 | 2.70 / 3.98 | 2.72 / 4.00 | 2.73 / 4.01 | 2.74 / 4.04 | **2.70** |
| 180 d  | 2.55 / 3.83 | 2.65 / 3.96 | 2.69 / 4.00 | 2.71 / 4.02 | 2.73 / 4.04 | 2.74 / 4.06 | 2.76 / 4.08 |     2.71 |

**DK2, single-stage + gas**

| Window | 1d          | 2d          | 3d          | 4d          | 5d          | 6d          | 7d          | 2-7d MAE |
| ------ | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | -------: |
| 30 d   | 2.50 / 3.80 | 2.75 / 4.09 | 2.83 / 4.19 | 2.88 / 4.28 | 2.89 / 4.33 | 2.88 / 4.29 | 2.92 / 4.32 |     2.86 |
| 60 d   | 2.46 / 3.73 | 2.64 / 3.95 | 2.70 / 4.05 | 2.73 / 4.11 | 2.78 / 4.17 | 2.74 / 4.12 | 2.75 / 4.13 |     2.72 |
| 90 d   | 2.45 / 3.71 | 2.62 / 3.91 | 2.67 / 3.98 | 2.70 / 4.04 | 2.75 / 4.10 | 2.75 / 4.08 | 2.75 / 4.11 |     2.71 |
| 120 d  | 2.44 / 3.72 | 2.60 / 3.91 | 2.65 / 3.97 | 2.69 / 4.03 | 2.72 / 4.07 | 2.74 / 4.09 | 2.73 / 4.11 |     2.69 |
| 180 d  | 2.46 / 3.74 | 2.57 / 3.88 | 2.60 / 3.93 | 2.64 / 3.99 | 2.68 / 4.03 | 2.70 / 4.04 | 2.69 / 4.04 | **2.65** |

**DK2, cross-border**

| Window | 1d          | 2d          | 3d          | 4d          | 5d          | 6d          | 7d          | 2-7d MAE |
| ------ | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | -------: |
| 30 d   | 2.22 / 3.45 | 2.40 / 3.67 | 2.49 / 3.77 | 2.55 / 3.87 | 2.55 / 3.85 | 2.58 / 3.88 | 2.59 / 3.89 |     2.53 |
| 60 d   | 2.12 / 3.27 | 2.29 / 3.50 | 2.34 / 3.57 | 2.35 / 3.57 | 2.37 / 3.60 | 2.37 / 3.61 | 2.39 / 3.64 |     2.35 |
| 90 d   | 2.10 / 3.25 | 2.24 / 3.41 | 2.29 / 3.49 | 2.31 / 3.51 | 2.33 / 3.55 | 2.36 / 3.58 | 2.37 / 3.62 |     2.32 |
| 120 d  | 2.12 / 3.26 | 2.24 / 3.41 | 2.28 / 3.46 | 2.30 / 3.51 | 2.32 / 3.55 | 2.34 / 3.58 | 2.36 / 3.63 |     2.31 |
| 180 d  | 2.12 / 3.25 | 2.22 / 3.39 | 2.26 / 3.44 | 2.28 / 3.47 | 2.30 / 3.50 | 2.31 / 3.51 | 2.33 / 3.56 | **2.28** |

**DK2, cross-border + gas**

| Window | 1d          | 2d          | 3d          | 4d          | 5d          | 6d          | 7d          | 2-7d MAE |
| ------ | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | -------: |
| 30 d   | 2.17 / 3.38 | 2.38 / 3.63 | 2.46 / 3.74 | 2.51 / 3.83 | 2.53 / 3.85 | 2.56 / 3.87 | 2.57 / 3.87 |     2.50 |
| 60 d   | 2.02 / 3.16 | 2.20 / 3.36 | 2.24 / 3.45 | 2.27 / 3.48 | 2.30 / 3.52 | 2.31 / 3.52 | 2.32 / 3.58 |     2.27 |
| 90 d   | 1.99 / 3.13 | 2.11 / 3.29 | 2.19 / 3.41 | 2.22 / 3.42 | 2.26 / 3.49 | 2.27 / 3.49 | 2.29 / 3.54 |     2.22 |
| 120 d  | 1.97 / 3.12 | 2.09 / 3.27 | 2.15 / 3.35 | 2.18 / 3.38 | 2.21 / 3.44 | 2.23 / 3.45 | 2.23 / 3.49 |     2.18 |
| 180 d  | 1.95 / 3.07 | 2.03 / 3.18 | 2.08 / 3.23 | 2.11 / 3.28 | 2.12 / 3.32 | 2.15 / 3.34 | 2.15 / 3.37 | **2.11** |

The naive baseline (same slot last week) is 3.93-4.00 (DK1) and 4.12-4.20
(DK2) at every horizon.

- **With the gas price, the longest window wins everywhere.** 180 days has
  the lowest days-2-7 MAE in all four gas configurations: 0.06-0.07 under
  60 days with the single-stage model and 0.12-0.16 with the cross-border
  model, and the gain grows with the horizon (at 7d, 0.06-0.17). Day 1 is
  unchanged with the single-stage model (within 0.01) and 0.05-0.07 better
  with the cross-border model. The gas price carries the price level, so older
  days no longer add prices from another level, which was what made them
  harmful in #24.
- **Without the gas price, 120 days is best** on DK1 and on single-stage
  DK2, with 180 days within 0.01-0.04 of it, and 180 days is best on DK2
  with the cross-border model; every one of them beats 60 days by
  0.04-0.07. 180 days is 0.02-0.04 worse than 60 days at 1d there: without
  the price level, the oldest days still cost a little on the day the known
  prices dominate.
- **Every window from 90 days up beats 60 days** at days 2-7 in all eight
  tables, and 30 days is the worst everywhere: the 7-day horizon rewards
  more history than the 3-day one of #24 did.
- The default is therefore **180 days** in every configuration (the best
  window does not differ with the cross-border option, so it is one number,
  not tied to the option). The window stays selectable down to 30 days for
  installs that must keep the footprint small.

**LightGBM agrees** (the `lightgbm (reference)` row, `lightgbm==4.7.0`, same
period; cells are NumPy GBM / LightGBM MAE, for the production
configurations):

| Configuration           | Window | 1d          | 2d          | 3d          | 4d          | 5d          | 6d          | 7d          | 2-7d MAE    |
| ----------------------- | ------ | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- |
| DK1, single-stage + gas | 60 d   | 2.19 / 2.13 | 2.37 / 2.34 | 2.43 / 2.44 | 2.47 / 2.51 | 2.53 / 2.56 | 2.58 / 2.61 | 2.62 / 2.63 | 2.50 / 2.52 |
| DK1, single-stage + gas | 120 d  | 2.20 / 2.04 | 2.34 / 2.25 | 2.42 / 2.34 | 2.48 / 2.40 | 2.50 / 2.44 | 2.52 / 2.46 | 2.54 / 2.49 | 2.47 / 2.40 |
| DK1, single-stage + gas | 180 d  | 2.20 / 2.03 | 2.31 / 2.23 | 2.40 / 2.29 | 2.45 / 2.35 | 2.47 / 2.37 | 2.49 / 2.41 | 2.52 / 2.46 | 2.44 / 2.35 |
| DK2, single-stage + gas | 60 d   | 2.46 / 2.42 | 2.64 / 2.67 | 2.70 / 2.75 | 2.73 / 2.79 | 2.78 / 2.82 | 2.74 / 2.80 | 2.75 / 2.82 | 2.72 / 2.77 |
| DK2, single-stage + gas | 120 d  | 2.44 / 2.34 | 2.60 / 2.58 | 2.65 / 2.66 | 2.69 / 2.68 | 2.72 / 2.71 | 2.74 / 2.73 | 2.73 / 2.71 | 2.69 / 2.68 |
| DK2, single-stage + gas | 180 d  | 2.46 / 2.36 | 2.57 / 2.57 | 2.60 / 2.60 | 2.64 / 2.63 | 2.68 / 2.63 | 2.70 / 2.65 | 2.69 / 2.67 | 2.65 / 2.62 |
| DK1, cross-border + gas | 60 d   | 1.84 / 1.73 | 1.97 / 1.88 | 2.05 / 1.94 | 2.08 / 2.01 | 2.11 / 2.06 | 2.14 / 2.07 | 2.17 / 2.11 | 2.09 / 2.01 |
| DK1, cross-border + gas | 180 d  | 1.79 / 1.64 | 1.88 / 1.79 | 1.94 / 1.84 | 1.98 / 1.87 | 1.98 / 1.86 | 2.01 / 1.90 | 2.04 / 1.94 | 1.97 / 1.87 |
| DK2, cross-border + gas | 60 d   | 2.02 / 1.97 | 2.20 / 2.17 | 2.24 / 2.24 | 2.27 / 2.27 | 2.30 / 2.29 | 2.31 / 2.29 | 2.32 / 2.33 | 2.27 / 2.27 |
| DK2, cross-border + gas | 180 d  | 1.95 / 1.84 | 2.03 / 1.99 | 2.08 / 2.04 | 2.11 / 2.05 | 2.12 / 2.05 | 2.15 / 2.10 | 2.15 / 2.14 | 2.11 / 2.06 |

180 days lowers the LightGBM days-2-7 MAE under 60 days by 0.15-0.17 with
the single-stage model and 0.14-0.21 with the cross-border model, more than
the NumPy model gains, and ranks the windows the same way.

**Footprint.** One training (live fit plus holdout fit, with the
Nordpool-masked copies of #91) on synthetic full history (every slot of the
window: prices, the zone weather at DK1's four points and, with the option,
the five neighbours' prices and weather), on one aarch64 core of the
development container, recorded 2026-10-02:

| Window | Single-stage | Cross-border |
| ------ | -----------: | -----------: |
| 60 d   |       0.45 s |        3.3 s |
| 120 d  |       0.77 s |        6.4 s |
| 180 d  |        1.1 s |        9.6 s |

A Raspberry Pi 4 core is roughly 5-10 times slower, so expect 5-10 s per
training at the default there, and 50-100 s with the cross-border model, in
the executor, a few times a day (whenever an input changed, see
[Retraining](#retraining)); the [live-accuracy procedure](#live-accuracy-94)
measures it from the log. The database holds about 8 MB at 60 days and 20
MB at 180 days with the single-stage model; the cross-border model adds the
neighbours' prices and weather, about 19 MB at 60 days and 60-70 MB at
180 days. The rows in memory while training (34,560 × 24 at 180 days, with
the copies) are a few MB. A new install's backfill fetches three times the
days of the old default; the sources only request what is missing.

## Target: Raw Spot Price, VAT at Output

The model is trained on, learns from and predicts the **raw day-ahead spot
price excl. VAT and tariffs**, in currency/kWh (#16), read from Stromligning's
spot price sensors (`read_spot_prices()`; see
[Stromligning Integration](stromligning_integration.md#overview)).
`price_history`, stored predictions, error metrics and bias offsets are all in
that unit. Tariffs are time-of-use and seasonal; in the target they would be
learned as if they were market behaviour. The consumer price (tariffs, fees
and tax included) is never a model input.

Tariffs and VAT are applied once, at output (#39, #107): `PriceOutput`
(`price_output.py`) turns every exposed forecast price into
`(spot + tariff + surcharge) × (1 + VAT)` in the configured unit, optionally
averaged per local hour. The slot's tariff is Stromligning's consumer minus
spot price (`TariffSchedule`, `tariffs.py`); 0 without consumer prices. The
`Price Forecast (ML)` sensor applies it to its state and to every price
attribute, and says so with `includes_vat: true`, `includes_tariffs`, `vat`,
`surcharge` and `hourly_average`. The model, the stored predictions, the
error metrics and the bias correction never see a tariff, the surcharge or
VAT. See [Architecture → Price Output](architecture.md#price-output).

## Negative Prices

DK1 and DK2 regularly clear below zero on windy or sunny days. Nothing in the
pipeline clamps a price at 0 (#15): the model's output, the heuristic
fallback, the additive bias correction, the stored predictions and the sensor
attributes all keep negative values.

## Prediction Window

Predictions start at the current 15-minute slot (at 10:05, the 10:00 slot).
If the confirmed prices reach further, they start where the confirmed prices
end instead (e.g. 12:30), so no confirmed slot is predicted and none is
skipped. The ML path and the heuristic fallback both get this start from
`first_prediction_slot()` in `time_slots.py`, which rounds on the UTC
timeline so a DST change cannot shift it.

The `Price Forecast (ML)` sensor's state is the prediction for the slot that
contains now (`start <= now < end`, compared in UTC so the repeated hour on
the DST fall-back day picks the right slot), or the first future slot when
the predictions start later. Its `state_slot_start` attribute is that slot's
start. The state is unknown if every prediction is in the past, rather than
showing a stale slot. The sensor is polled, so the state moves to the next
slot within Home Assistant's polling interval.

## Heuristic Fallback

Until the price model is trained (e.g. during the first day of history),
predictions come from `_generate_heuristic_predictions` in `ml/models.py`:
the mean of the known prices times an hour-of-day factor, with confidence
falling 0.1 per day ahead (floor 0.3). The factor for a local hour is that
hour's mean price divided by the mean over all hours with prices
(`_extract_hourly_pattern`). It reads the aligned price series (today, then
tomorrow, on the 15-minute grid from local midnight) and takes each known
slot's real local hour from `slot_start_in_day()`, so a `None` slot or a
92/100-slot DST day never shifts the other slots. An hour without prices, or
every hour when the mean is not positive, gets the neutral factor 1.0.

## Solar Scaling Factor

A learned EMA ratio between Solcast's estimate for today and the actual
inverter output:

```
solar_scale = EMA(actual_power / solcast_estimate)
```

Updated every prediction run (`_update_solar_scale`) and persisted. Since
#17 it is **not applied to the price model**: the model no longer has a site
solar feature (see [Feature Vector](#feature-vector-26-features)), and a
factor applied to prediction rows only would make them differ from training
rows again.

## Confidence Score

Confidence adapts based on actual prediction accuracy per 15-minute slot.

**Phase 1 — Heuristic** (< 5 samples):

```
base = 0.80 - wind_penalty - solar_penalty - weekend - 0.05 × days_ahead
wind_penalty  = 0.20 if the slot has no wind forecast (wind_speed_mean unknown)
solar_penalty = 0.10 if the slot has no Nordpool solar prognosis
days_ahead = whole days until the slot starts (0 for the current slot)
Floor: 0.30
```

**Phase 2 — Learned** (≥ 5 samples):

```
confidence = max(0.10, 1.0 - (MAE / mean(|actual|)))
- volatility penalty: min(0.25, 0.3 × volatility_MAE / mean(|actual|))
- forecast_temp_error penalty (max -0.15)
- forecast_wind_error penalty (max -0.15)
```

The price scale is the slot's mean _absolute_ actual price (#15), so slots
that clear at or below zero still get a learned confidence; with only positive
prices it equals the mean price. Only a slot whose actual prices are all
exactly zero falls back to the heuristic.

## Self-Learning Loop

Every 15 minutes:

1. Read the current raw spot price (excl. VAT)
2. Look up every stored prediction for the current slot (all forecast runs)
3. Calculate error, update per-slot metrics
4. Update the slot's additive bias offsets, one per lead-time bucket, via EMA
   (see [self-learning](self_learning.md#bias-correction))
5. Compare stored forecast weather vs actual → forecast accuracy tracking
6. Remove matched predictions from pending queue
7. Add each error to its lead-time bucket (day 1/2/3/4+) → rolling 30-day
   MAE/RMSE per lead time (see [self-learning](self_learning.md))

**96 slots** (15-min intervals), not 24 hours. Each slot has independent
bias offsets, one per lead-time bucket (#118), and a prediction gets the
offset of its own lead time.

## Backtesting

`scripts/backtest.py` is a dev-only rolling backtest (not shipped with the
integration). It gives every model or feature change a reproducible
multi-day accuracy number. The method reimplements EpexPredictor's
`performance_testing.py` (BSD-3-Clause).

### Method

- **Rolling origin.** For every day _D_ in the test period, each model is
  retrained on the previous `--window-days` local days of prices (default
  180). It then forecasts the local days _D_, _D+1_ and _D+2_, which are
  scored separately as **1d, 2d and 3d ahead**. `--step-days` retrains less
  often.
- **Strict cutoff.** The horizon cutoff is local midnight at the start of
  _D_, after EpexPredictor's `DataStore.horizon_cutoff`. Models only receive
  `PriceSeries.between(window_start, cutoff)`: read-only copies of the slots
  that start strictly before the cutoff. Target slots come from the calendar
  (92, 96 or 100 on DST days), not from the data. `tests/test_backtest.py`
  replaces every price at or after the cutoff with 1e6 and with NaN, and
  asserts that no model's forecast changes.
- **Metrics.** MAE and RMSE per horizon, in EUR ct/kWh (EpexPredictor's
  unit). As in EpexPredictor, MAE is the mean of the daily MAEs and RMSE is
  the square root of the mean daily MSE. Slots without an actual price are
  skipped.
- **Prices.** Day-ahead prices come from energy-charts
  (`api.energy-charts.info`; © Bundesnetzagentur | SMARD.de, CC BY 4.0), the
  source #27 will add to the integration. Hourly prices from before
  2025-10-01 fill four quarter-hours. Complete months are cached in
  `.cache/backtest/` (every cache file is written atomically and an
  unreadable one is fetched again; a truncated response, a connection
  error, a timeout and an HTTP 5xx or 429 are retried up to five times). A month energy-charts has no prices for yet (HTTP 404,
  e.g. when `--horizon-days` reaches past the current month) is left empty
  and its slots are not scored. The OSF SQLite DB cannot be used, because it keeps only
  the training window's price history.

### Models

| Row                           | Model                                                                                                                                                                                    |
| ----------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `naive (same slot last week)` | Price of the same local wall-clock slot seven days earlier. The bar every model must clear.                                                                                              |
| `current (NumPy GBM)`         | `create_price_model()` fitted on rows from `build_feature_row()` + `build_feature_vector()`: the integration's own model and feature code.                                               |
| `lightgbm (reference)`        | LightGBM (500 rounds, learning rate 0.05, 31 leaves, seed 42, one thread) on the same rows. Dev-only (`requirements_backtest.txt`): LightGBM has no musllinux wheels, so it cannot ship. |

The `current` row measures the model and features, not the whole runtime
pipeline:

- **Zone weather from Open-Meteo's archive.** The zone features (18-23) come
  from Open-Meteo's historical forecast API (`historical-forecast-api`, 90
  days per request, cached in `.cache/backtest/`) at the region's
  `WEATHER_POINTS`, for training and target slots alike (`--weather none`
  leaves them out). The archive keeps the start of each model run, so a
  target slot days ahead gets weather about as good as a same-day forecast:
  the zone rows are **optimistic at 2-3 days ahead**, where the live
  forecast is less accurate.
- **No Nordpool history by default.** Features 10-15 have no source for a
  year of history (`nordpool_prognoses` keeps the training window plus 2
  days), so they are NaN in every row. `--nordpool-db PATH` reads the
  `nordpool_prognoses` table of a learning-database export (read-only; keep
  it in the git-ignored `.cache/live/`): training rows get their stored
  prognoses and, as in production, target rows none; `--nordpool-day1`
  gives day 1 its prognosis, as in a morning run. Training adds the masked
  copies (#91); `--nordpool-no-copies` reproduces the training before it.
  Only origins inside the export's history are meaningful.
- **ENTSO-E load only with a key.** `--load entsoe` (token in the
  `ENTSOE_API_KEY` environment variable, never cached or printed) adds
  feature 16 from ENTSO-E's week-ahead forecasts, one request per ISO week
  (ENTSO-E answers one week per request, #114), cached as daily min/max in
  `.cache/backtest/` once the week is over; otherwise it is NaN. A
  target day gets the forecast ENTSO-E keeps for it, which for the last
  days of a week can be newer than the origin (optimistic, like the
  weather archive).
- **Lagged prices only with `--lags`** (#119, tested and not kept). The
  columns of `scripts/backtest_lags.py` follow the features: computed from
  `PriceSeries.between(window_start, cutoff)`, the only prices a model
  receives, so a target row's reference day is the day before the origin,
  whatever its horizon, and a training row's the day before its own
  (`--lag-ages`: 1-7 days, hashed per day, next to `price_lag_days`).
- **Raw model output.** The per-slot, per-lead-time bias correction (which needs live
  self-learning state) is not applied. The hyperparameters are the
  production defaults, as in the integration.
- **Gas price only with `--gas`.** Feature 17 comes from Instrat's daily
  history (one request per month, cached in `.cache/backtest/`); every
  origin only sees prices dated before its horizon cutoff, so all target
  days get the latest price published before the forecast, as in
  production. Without the flag it is NaN.
- **Outages only with `--outages umm` or `--outages entsoe`.** Features
  18-19 come from Nord Pool's UMM API (every message version, one request
  series per calendar month, cached in `.cache/backtest/` once the month
  is over), for a region in `UMM_AREAS`, or from ENTSO-E's outage
  documents (#138, `ENTSOE_API_KEY`; the current revision of every document,
  per month and per query — the zone's A77/A80 and every border's A78 in
  both directions — paged with `offset`), for a region in
  `ENTSOE_OUTAGE_BORDERS`. Training rows get the messages published by
  their day-ahead gate and target rows those published before the horizon
  cutoff, exactly as in production: no row sees a later revision.
- **Cross-border model only with `--cross-border`.** The `current` and
  `lightgbm` rows then become two-stage models (#29) for a region in
  `NEIGHBOURS`: stage 1 is the integration's `Stage1Model` per neighbour,
  fitted on the neighbour's energy-charts prices from the same window, cut
  at the same horizon cutoff, with its Open-Meteo archive weather.

### Running

```bash
pip install -r requirements_backtest.txt   # optional LightGBM row
./scripts/quality.sh backtest --region DK1  # last 365 origins, 180-day window
python -m scripts.backtest --region DK1 --start 2025-09-21 --end 2026-09-20 --window-days 30
python -m scripts.backtest --region DK1 --window-days 60 --days holidays
python -m scripts.backtest --region DK1 --window-days 60 --cross-border
python -m scripts.backtest --region DK1 --window-days 60 --gas
python -m scripts.backtest --region DK1 --window-days 30 --start 2026-08-06 --end 2026-09-25 --nordpool-db .cache/live/dk1_live.db
ENTSOE_API_KEY=… python -m scripts.backtest --region DK1 --window-days 60 --horizon-days 7 --load entsoe
python -m scripts.backtest --region DK1 --window-days 60 --horizon-days 7 --lags price_same_slot_last_known_day [--lag-ages]
python -m scripts.backtest --region DK1 --window-days 60 --horizon-days 7 --outages umm
ENTSOE_API_KEY=… python -m scripts.backtest --region DE --window-days 60 --horizon-days 7 --outages entsoe
```

`--region` accepts every OSF region. `--horizon-days` scores more forecast
days per origin (default 3; 7 covers days 3-7 of the forecast). `--days holidays` scores only the
target days that are public holidays and not Sundays (and only runs the
origins that forecast one), to measure a change on the days the `holiday`
feature is for. Run the backtest before and after every
model or feature change, and put both tables in the PR.

### Live accuracy (#94)

The backtest is optimistic from day 2 on (archived weather, see above), so
choices that only show at longer lead times, such as the out-of-sample
stage-1 values of the cross-border model, need live numbers.
`scripts/live_report.py` (dev-only, not shipped) reads them from a
learning-database export:

- **Per lead-time bucket** (`lead_time_accuracy`, the last 30 days): samples,
  days, MAE, RMSE and bias (mean of predicted − actual) pooled over every
  sample, as the accuracy sensors show them, plus the mean daily MAE and the
  root of the mean daily MSE, the backtest's method. With more than one ISO
  week, the MAE per week and bucket too. The buckets are lead times from
  when the prediction was stored (`day_1` = 0-24 h), while the backtest's 1d
  is the local day after the origin (0-24 h after midnight), so `day_1`
  compares with the backtest's 1d and `day_2` with its 2d only roughly.
- **Evaluation** (`evaluation`, the last 7 days): the same metrics for the
  prediction made closest to 24 h ahead of each slot.
- **Training** (`meta`): the latest holdout MAE/RMSE and training sample
  count, and the `hpo_*` keys of an export from before #92.
- **Coverage**: the dates the data spans, with a warning below 14 days.
  `--since YYYY-MM-DD` keeps only slots from that local date on.

The tables are in the model's unit (raw spot price excl. VAT,
currency/kWh). `--eur-per-unit RATE` (EUR per currency unit, e.g. the
period's mean ECB rate) or `--currency DKK` (the ERM II central rate, or
`EUR`) reports EUR ct/kWh, the backtest's unit. The export is opened with
SQLite's `immutable=1`: the report never writes, migrates or checkpoints
it.

Procedure:

1. Update the Home Assistant instance to the current `main` and turn on the
   option under test (e.g. **Cross-border model**). Note the date: the
   database keeps the older version's lead-time sums for 30 days, and
   `--since` (the first local date on the new version) leaves them out.
2. Let it run for **14 days or more** on that version.
3. Export the database with SQLite's online backup, which is consistent
   while Home Assistant writes to it (a plain copy of the `.db` file misses
   what is still in the WAL):

   ```bash
   mkdir -p .cache/live
   sqlite3 /config/.storage/open_spot_forecast_DK1_learning.db ".backup '/tmp/dk1_live.db'"
   # copy /tmp/dk1_live.db from the HA host to .cache/live/dk1_live.db
   ```

   Keep exports in the git-ignored `.cache/live/`, **never in the
   repository root**: they hold weather snapshots and forecasts that reveal
   the installation's location.

4. Run the report, with the training times from the Home Assistant log
   (`ML model trained in … s: holdout MAE=…`, one line per training; see
   [Retraining](#retraining)):

   ```bash
   python -m scripts.live_report .cache/live/dk1_live.db --region DK1 --currency DKK --since 2026-10-01 --log .cache/live/home-assistant.log
   ```

5. Record the per-bucket MAE next to the backtest's 1d/2d/3d rows and, for
   the cross-border model, the median training time on the Raspberry Pi in
   its **Cost** paragraph (see [Cross-Border Model](#cross-border-model-29)).

**No live results are recorded yet.** The only DK1 export so far
(2026-09-28) held one day of self-learning, from a version with
hyperparameter optimization (#92) and without the cross-border model:
day 1 MAE 3.41 and day 2 4.62 EUR ct/kWh (133 and 155 samples, bias −1.75
and −3.78), too few days to compare with the backtest.

### Zone weather (#22)

DK1, 365 daily origins from 2025-09-24 to 2026-09-23, retrained daily, EUR
ct/kWh. `before` is `--weather none` (time features only, identical to the
model before #22); `after` adds the Open-Meteo zone features. Recorded
2026-09-26 with `lightgbm==4.7.0`.

| Window | Model                | before 1d MAE | after 1d MAE | before 2d / 3d MAE | after 2d / 3d MAE |
| ------ | -------------------- | ------------: | -----------: | -----------------: | ----------------: |
| 180 d  | current (NumPy GBM)  |          3.76 |     **2.53** |        3.78 / 3.82 |       2.64 / 2.70 |
| 180 d  | lightgbm (reference) |          3.78 |         2.52 |        3.78 / 3.82 |       2.69 / 2.75 |
| 30 d   | current (NumPy GBM)  |          3.30 |     **2.41** |        3.32 / 3.33 |       2.62 / 2.70 |
| 30 d   | lightgbm (reference) |          3.34 |         2.46 |        3.35 / 3.36 |       2.70 / 2.81 |

The naive baseline is 3.93. RMSE falls alike (30 days, 1d: 4.68 → 3.67).
DK2 with a 30-day window: 3.52 → 2.66 1d MAE (LightGBM 3.57 → 2.74).

- **The zone weather is the most valuable input so far**: 27 % lower 1d MAE
  with the production 30-day window, 33 % with 180 days. With weather the
  180-day window is no longer far behind the 30-day one, since the price
  level now follows the inputs rather than recency alone (#24).
- **2d/3d errors grow with the lead time now** (2.41 → 2.62 → 2.70), because
  the price depends on the weather. They are still optimistic: the archive
  gives target slots days ahead near-same-day forecasts.
- **Aggregates, not per-point columns.** On weekly origins, the zone
  aggregates and one column per point and value (time + 5 × 4 columns) tied
  at 1d (DK1 30 days: 2.41 vs 2.42; 180 days: 2.56 vs 2.57); per-point was
  0.05 worse at 2d and up to 0.12 better at 3d. The aggregates keep one
  vector for every region.
- **Training time** barely changes: a 180-day fit (20,252 rows, 23
  features) takes 0.38 s with the zone weather and 0.34 s without.

### Training on archived forecasts (#23)

Both models predict the targets from Open-Meteo's archived forecasts, as
production does; they differ only in the weather of the training rows:
ERA5 reanalysis (`archive-api.open-meteo.com`, the measured weather) or the
archived forecasts themselves. Hourly data with the same variables in both
(100 m wind, as ERA5 has no 80 m level), DK1, 365 daily origins from
2025-09-18 to 2026-09-17 (ERA5 lags about 5 days), EUR ct/kWh:

| Window | Training weather   | 1d MAE | 1d RMSE | 2d MAE | 2d RMSE | 3d MAE | 3d RMSE |
| ------ | ------------------ | -----: | ------: | -----: | ------: | -----: | ------: |
| 30 d   | actuals (ERA5)     |   2.40 |    3.65 |   2.62 |    3.89 |   2.70 |    3.99 |
| 30 d   | archived forecasts |   2.39 |    3.64 |   2.62 |    3.88 |   2.69 |    3.98 |
| 180 d  | actuals (ERA5)     |   2.51 |    3.71 |   2.61 |    3.83 |   2.65 |    3.87 |
| 180 d  | archived forecasts |   2.52 |    3.74 |   2.62 |    3.85 |   2.67 |    3.92 |

The two are within 0.01-0.05 ct/kWh of each other: 0.01 better on archived
forecasts with the 30-day window, 0.01-0.05 better on actuals with 180
days. This backtest cannot show the over-trust #23 is about, because its
target rows use the archive too, whose forecasts are near-same-day, while
at prediction 1-3 day forecasts are further off. What decided it is that
forecasts have an archive: the whole training window's zone weather is
available on the first day, from the same kind of source the model
predicts from, where measured weather would have to accumulate first.
The local weather entity's features were removed for the same reason (see
[Feature Vector](#feature-vector-26-features)); the backtest never had them
(no history), so its numbers do not change.

### Sun position and 15-minute time (#25)

DK1, 365 daily origins from 2025-09-24 to 2026-09-23, retrained daily, EUR
ct/kWh, with the zone weather. `before` is main before #25 (`hour`,
`hour_sin`, `hour_cos`); `after` has the 15-minute time and sun features.
Recorded 2026-09-27 with `lightgbm==4.7.0`.

| Window | Model                | before 1d MAE | after 1d MAE | before 2d / 3d MAE | after 2d / 3d MAE |
| ------ | -------------------- | ------------: | -----------: | -----------------: | ----------------: |
| 60 d   | current (NumPy GBM)  |          2.40 |     **2.29** |        2.57 / 2.62 |       2.46 / 2.52 |
| 60 d   | lightgbm (reference) |          2.45 |         2.28 |        2.65 / 2.72 |       2.48 / 2.55 |
| 180 d  | current (NumPy GBM)  |          2.53 |     **2.32** |        2.64 / 2.70 |       2.43 / 2.49 |

RMSE falls alike (60 days, 1d: 3.62 → 3.47; 180 days: 3.78 → 3.51). The
naive baseline is 3.93.

Which features carry it (NumPy GBM, 60 days, 1d / 2d / 3d MAE; each row
changes one thing from the `after` set):

| Variant                                        | 1d MAE | 2d MAE | 3d MAE |
| ---------------------------------------------- | -----: | -----: | -----: |
| after (21 features)                            |   2.29 |   2.46 |   2.52 |
| without the sun features (15-minute time only) |   2.37 |   2.53 |   2.58 |
| without `since_sunrise` / `since_sunset`       |   2.30 |   2.47 |   2.53 |
| without `sun_azimuth`                          |   2.29 |   2.46 |   2.52 |
| without `morning_peak`                         |   2.29 |   2.46 |   2.52 |
| with `hour` added back                         |   2.29 |   2.46 |   2.52 |
| with `evening_peak` (seconds from 19:00)       |   2.29 |   2.46 |   2.52 |

- **The sun does most of the work**: the 15-minute time alone gains 0.03 at
  1d, the sun features another 0.08. Solar output, and the price with it,
  follows sunrise and sun height, which move by hours over the year.
- **The longer window gains most** (0.21 at 1d with 180 days): the sun
  features tell the model the season, so older days no longer only add
  prices from another level.
- `hour` and `evening_peak` change nothing (a tree splits them exactly like
  `morning_peak`) and are not features. `sun_azimuth` and `morning_peak`
  only tie here; they stay as the issue's features, at negligible cost.

### Public holidays (#26)

DK1, 365 daily origins from 2025-09-24 to 2026-09-23, retrained daily, EUR
ct/kWh, with the zone weather and the #25 features. `before` has no
`holiday` feature. Recorded 2026-09-27 with `lightgbm==4.7.0`.

**Holiday target days** (`--days holidays`: the 10 public holidays in the
period that are not Sundays, including 24/12 and 31/12; 22 origins):

| Window | Model                | before 1d MAE | after 1d MAE | before 2d / 3d MAE | after 2d / 3d MAE |
| ------ | -------------------- | ------------: | -----------: | -----------------: | ----------------: |
| 60 d   | current (NumPy GBM)  |          2.97 |     **2.92** |        3.69 / 4.05 |       3.59 / 4.02 |
| 60 d   | lightgbm (reference) |          2.99 |         2.77 |        3.72 / 3.97 |       3.52 / 3.94 |
| 180 d  | current (NumPy GBM)  |          2.72 |     **2.65** |        3.21 / 3.46 |       3.17 / 3.40 |
| 180 d  | lightgbm (reference) |          2.89 |         2.74 |        3.51 / 3.77 |       3.46 / 3.75 |

The naive baseline is 3.86 on these days. **All days** (NumPy GBM): 60
days 2.29 / 2.46 / 2.52 before and after (1d RMSE 3.47 → 3.46), 180 days
2.32 → 2.31 at 1d and 2.43 / 2.49 at 2d / 3d before and after.

Variants on the holiday days (NumPy GBM, 60 days, 1d / 2d / 3d MAE):

| Variant                       | 1d MAE | 2d MAE | 3d MAE |
| ----------------------------- | -----: | -----: | -----: |
| no `holiday` feature          |   2.97 |   3.69 |   4.05 |
| after (24/12 and 31/12 = 0.5) |   2.92 |   3.59 |   4.02 |
| 24/12 and 31/12 = 0           |   2.92 |   3.66 |   4.08 |
| 24/12 and 31/12 = 1           |   2.88 |   3.58 |   4.02 |
| Sundays are 0 (holidays only) |   2.88 |   3.71 |   4.06 |
| `holiday` as the first column |   2.93 |   3.59 |   4.02 |

- The feature lowers the error on holidays in every setting, most with the
  longer window, which holds more holidays to learn from, and for LightGBM
  (0.22 at 1d with 60 days). It leaves the other days unchanged.
- Ten days make a small sample: the variants are within a few hundredths
  of each other. Counting Sundays helps at 2d / 3d, as the design intends,
  and the half-day value of 24/12 and 31/12 is not decided by these two
  days (in Denmark they are close to full holidays, 1 scores best; in
  Germany they are half days), so it stays at the issue's 0.5.

### Cross-border model (#29)

365 daily origins from 2025-09-24 to 2026-09-23, 60-day window, retrained
daily, EUR ct/kWh, with the zone weather and every earlier feature.
`before` is the single-stage model, `after` the two-stage model
(`--cross-border`). Recorded 2026-09-27 with `lightgbm==4.7.0`.

| Region | Model                | before 1d MAE | after 1d MAE | before 2d / 3d MAE | after 2d / 3d MAE |
| ------ | -------------------- | ------------: | -----------: | -----------------: | ----------------: |
| DK1    | current (NumPy GBM)  |          2.29 |     **1.90** |        2.46 / 2.52 |       2.02 / 2.07 |
| DK1    | lightgbm (reference) |          2.27 |         1.82 |        2.48 / 2.55 |       1.97 / 2.02 |
| DK2    | current (NumPy GBM)  |          2.52 |     **2.12** |        2.69 / 2.73 |       2.29 / 2.34 |
| DK2    | lightgbm (reference) |          2.54 |         2.09 |        2.74 / 2.81 |       2.28 / 2.35 |

RMSE falls alike (DK1 1d: 3.46 → 3.00; DK2 1d: 3.78 → 3.27). The naive
baseline is 3.93 (DK1) and 4.12 (DK2).

How stage 1 feeds stage 2 (DK1, NumPy GBM, same period; `s/origin` is the
fit time per origin on one core here, stage 1 included):

| Stage-1 values in training rows                 | 1d MAE | 2d MAE | 3d MAE | s/origin |
| ----------------------------------------------- | -----: | -----: | -----: | -------: |
| none (single-stage)                             |   2.29 |   2.46 |   2.52 |     0.17 |
| fitted, in sample (EpexPredictor)               |   1.87 |   2.02 |   2.07 |     1.20 |
| out of sample, 2 interleaved-day folds (chosen) |   1.90 |   2.03 |   2.07 |     1.60 |
| out of sample, 4 interleaved-day folds          |   1.88 |   2.01 |   2.06 |     3.36 |

On weekly origins (53) two more variants were tried: out of sample from 4
contiguous blocks of days scored 2.12 / 2.18 / 2.13 (a block's model
extrapolates to the window's ends), and the neighbours' zone weather as
direct columns instead of stage 1 (30 columns) 2.06 / 2.08 / 2.08, against
1.89 / 2.00 / 2.07 in sample and 2.17 / 2.35 / 2.48 single-stage. Germany
alone as a neighbour gave 1.97 / 2.13 / 2.14; adding DK1's own stage-1
model, as EpexPredictor does, changed nothing (1.88 / 2.02 / 2.08).

- **The neighbours are the largest gain since the zone weather**: 17-18 %
  lower MAE for DK1 and 14-16 % for DK2 at every horizon, for both learners.
- **In and out of sample tie in the backtest**, but the backtest cannot show
  the over-trust out-of-sample values guard against: its target rows get
  Open-Meteo's archive, near-same-day forecasts even days ahead, so stage 1
  forecasts the targets about as well as it fits its training rows. At
  prediction, days 2-7 have real forecasts; out-of-sample training values
  (error 1.9 against 1.3 fitted) match that. Two folds cost about one full
  fit.
- **Cost**: about 7 times the training time (see
  [Cross-Border Model](#cross-border-model-29)).

### Gas price (#28)

365 daily origins from 2025-09-24 to 2026-09-23, 60-day window, retrained
daily, EUR ct/kWh, NumPy GBM with every earlier feature. `before` is
without the gas price, `after` with it (`--gas`). Recorded 2026-09-27.

| Region | before 1d / 2d / 3d MAE | after 1d / 2d / 3d MAE | Gas price      |
| ------ | ----------------------: | ---------------------: | -------------- |
| DK1    |      2.29 / 2.46 / 2.52 |     2.19 / 2.37 / 2.43 | on             |
| DK2    |      2.52 / 2.69 / 2.73 |     2.46 / 2.64 / 2.70 | on             |
| DE     |      2.11 / 2.23 / 2.26 |     2.02 / 2.14 / 2.18 | on             |
| BE     |      2.27 / 2.40 / 2.45 |     2.15 / 2.31 / 2.36 | on             |
| NL     |      2.16 / 2.29 / 2.33 |     2.08 / 2.24 / 2.31 | on             |
| NO2    |      1.73 / 1.88 / 1.95 |     1.57 / 1.77 / 1.88 | on             |
| SE3    |      2.26 / 2.45 / 2.50 |     2.21 / 2.43 / 2.54 | off (3d worse) |
| SE4    |      2.84 / 3.06 / 3.13 |     2.76 / 2.99 / 3.08 | on             |
| FI     |      2.64 / 2.92 / 2.98 |     2.53 / 2.81 / 2.91 | on             |
| EE     |      3.85 / 4.04 / 4.07 |     3.87 / 4.12 / 4.17 | off (worse)    |
| LT     |      3.89 / 4.14 / 4.23 |     3.89 / 4.18 / 4.32 | off (worse)    |
| LV     |      3.83 / 4.06 / 4.12 |     3.80 / 4.13 / 4.25 | off (2d/3d)    |
| FR     |      2.58 / 2.78 / 2.89 |     2.43 / 2.67 / 2.77 | on             |

A region gets the gas price (`GAS_PRICE_REGIONS`) where it lowers the MAE
at every horizon. Which price, on DK1 / DE / BE (1d / 2d / 3d MAE, same
period):

| Gas price                                 | DK1                | DE                 | BE                 |
| ----------------------------------------- | ------------------ | ------------------ | ------------------ |
| none                                      | 2.29 / 2.46 / 2.52 | 2.11 / 2.23 / 2.26 | 2.27 / 2.40 / 2.45 |
| THE Day Ahead (Bundesnetzagentur, scrape) | 2.19 / 2.34 / 2.43 | 2.03 / 2.16 / 2.21 | 2.16 / 2.31 / 2.35 |
| THE Month+1 (Bundesnetzagentur, scrape)   | 2.22 / 2.39 / 2.46 | 2.02 / 2.16 / 2.22 | 2.15 / 2.29 / 2.34 |
| TGE day-ahead (Instrat API, chosen)       | 2.19 / 2.37 / 2.43 | 2.02 / 2.14 / 2.18 | 2.15 / 2.31 / 2.36 |

- **The gas price helps in most regions**, by 0.06-0.16 ct/kWh at 1d.
  EpexPredictor's backtests found it worse for DK and SE, with THE Month+1
  and its own setup; here THE Month+1 helps DK1 too. With a 60-day window
  the gas level explains part of the price level within the window. In EE
  and SE3 it does not help, nor in LT and LV at 2-3 days, so the Baltics
  and SE3 do without.
- **The three prices score alike**: only the European gas price level
  matters, not the hub. Instrat's API is machine-readable and complete
  (weekends included), where the THE prices would have to be scraped from
  HTML.
- **It adds to the cross-border model** (#29): DK1 two-stage scores
  1.84 / 1.97 / 2.05 with the gas price, against 1.90 / 2.02 / 2.07
  without (stage 1 does not use it).
- The gas price is known before the day, so unlike the zone weather it is
  not optimistic at 2-3 days ahead.

### Nordpool prognoses at prediction (#91)

DK1, `current (NumPy GBM)` with the zone weather, prognoses from a live
learning-database export (82 days, 2026-07-07 to 2026-09-27), daily origins
2026-08-06 to 2026-09-25 (30-day window, 51 origins) and 2026-09-05 to
2026-09-25 (60-day window, 21 origins), so every training window lies inside
the export. MAE in EUR ct/kWh, 1d / 2d / 3d. "→ without" scores target rows
without prognoses (production after ~13:00), "→ day 1 with" gives day 1 its
prognosis (morning runs). Recorded 2026-09-28.

| Training rows → prediction rows                | 30-day window          | 60-day window          |
| ---------------------------------------------- | ---------------------- | ---------------------- |
| no Nordpool features                           | 3.20 / 3.45 / 3.59     | 3.98 / 4.10 / 4.16     |
| with prognoses → without (before #91)          | 3.74 / 3.89 / 3.96     | 5.01 / 4.93 / 4.90     |
| with prognoses → day 1 with (before #91)       | 2.97 / 3.89 / 3.96     | 3.68 / 4.93 / 4.90     |
| with **and** without prognoses → without (#91) | **3.19 / 3.45 / 3.58** | **3.98 / 4.12 / 4.22** |
| with and without prognoses → day 1 with (#91)  | 3.10 / 3.45 / 3.58     | 3.74 / 4.12 / 4.22     |

Before #91 the model was 0.4-1.0 ct/kWh worse than without the features at
every horizon, because a split whose node saw no NaN rows sends NaN to the
child with more rows, an arbitrary branch. With the masked copies a forecast
without prognoses scores like the model without the features (1d equal at
both windows; 2d/3d within 0.06 ct/kWh at 60 days), and a morning forecast
keeps most of the day-1 gain (3.10 vs 3.20 at 30 days, 3.74 vs 3.98 at 60
days). Dropping features 10-15 would be as good after ~13:00 but lose that
gain.

```bash
python -m scripts.backtest --region DK1 --window-days 30 --start 2026-08-06 --end 2026-09-25 --models current --nordpool-db .cache/live/dk1_live.db [--nordpool-day1] [--nordpool-no-copies]
python -m scripts.backtest --region DK1 --window-days 60 --start 2026-09-05 --end 2026-09-25 --models current --nordpool-db .cache/live/dk1_live.db [--nordpool-day1] [--nordpool-no-copies]
```

### Nord Pool UMM outages (#123)

DK1 and DK2, 365 daily origins from 2025-09-30 to 2026-09-29, 60-day window
(the production window), `--horizon-days 7`, with the zone weather;
`--outages none` against `--outages umm`. Training rows get the messages
published by their day-ahead gate, target rows those published before the
horizon cutoff (local midnight of day 1). MAE in EUR ct/kWh, **without /
with** the two outage features. Recorded 2026-10-02.

| Zone | Model                | 1d          | 2d          | 3d          | 4d          | 5d          | 6d          | 7d          |
| ---- | -------------------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- |
| DK1  | current (NumPy GBM)  | 2.30 / 2.27 | 2.47 / 2.47 | 2.52 / 2.54 | 2.54 / 2.55 | 2.54 / 2.55 | 2.56 / 2.56 | 2.59 / 2.60 |
| DK1  | lightgbm (reference) | 2.29 / 2.19 | 2.51 / 2.48 | 2.58 / 2.56 | 2.60 / 2.58 | 2.62 / 2.60 | 2.63 / 2.59 | 2.67 / 2.63 |
| DK2  | current (NumPy GBM)  | 2.55 / 2.51 | 2.71 / 2.77 | 2.74 / 2.81 | 2.76 / 2.88 | 2.76 / 2.91 | 2.75 / 2.90 | 2.78 / 2.94 |
| DK2  | lightgbm (reference) | 2.58 / 2.46 | 2.76 / 2.79 | 2.83 / 2.88 | 2.85 / 2.93 | 2.82 / 2.96 | 2.83 / 2.95 | 2.87 / 3.00 |

The naive row is 3.97-4.00 in DK1 and unchanged per zone. What the numbers
say:

- **Day 1 improves everywhere**: −0.03 (DK1) and −0.04 (DK2) ct/kWh MAE
  for the production model, −0.10 and −0.12 for LightGBM, with the RMSE
  down as well (DK1 3.47 → 3.45, DK2 3.79 → 3.73). The day the auction has
  just cleared is the day whose outage state is complete.
- **Days 2-7 get worse in DK2** (+0.06 at 2d to +0.16 at 7d), for both
  learners, while **DK1 stays within ±0.02** (noise). A target slot days
  ahead gets the messages known at the origin, fewer than its own gate will
  have, while every training row has its gate's complete picture; the
  model learned the price effect of a complete outage state and applies it
  to an incomplete one. The same pattern as the lagged prices (#119: help
  at 1 day, hurt at 3-7), and the reason for the near-term model (#135).
- The production model gains less than LightGBM: depth-3 trees with 100
  samples per leaf see a sparse signal (a few outage episodes per 60-day
  window).

**Decision**: `UMM_REGIONS` is `{"DK1"}`: a lower day-1 error and no higher
error later. DK2 stays off (the day-1 gain does not pay for days 2-7), and
the other Nord Pool areas stay off until measured (`UMM_AREAS` lets the
backtest run them). Training rows with the outage state as of each lead
time (as the Nordpool-masked copies do for the prognoses, #91) is the
obvious next step for DK2.

```bash
python -m scripts.backtest --region DK1 --window-days 60 --horizon-days 7 --outages none
python -m scripts.backtest --region DK1 --window-days 60 --horizon-days 7 --outages umm
python -m scripts.backtest --region DK2 --window-days 60 --horizon-days 7 --outages umm
```

### ENTSO-E outages (#138)

DE, NL, BE and FR, 365 daily origins, 60-day window (the production
window), `--horizon-days 7`, with the zone weather; `--outages none`
against `--outages entsoe`. Training rows get the documents published (the
current revision's `createdDateTime`) by their day-ahead gate, target rows
those published before the horizon cutoff. MAE in EUR ct/kWh, **without /
with** the two outage features.

<!-- ENTSOE_OUTAGE_RESULTS -->

```bash
ENTSOE_API_KEY=… python -m scripts.backtest --region DE --window-days 60 --horizon-days 7 --outages none
ENTSOE_API_KEY=… python -m scripts.backtest --region DE --window-days 60 --horizon-days 7 --outages entsoe
```

### ENTSO-E load forecast (#30)

DK1, 365 daily origins from 2025-09-30 to 2026-09-29, 60-day window,
`--horizon-days 7`, with the zone weather; `--load none` against
`--load entsoe` (one request per ISO week, #114). MAE in EUR ct/kWh,
**without / with** the load feature. Recorded 2026-10-02.

| Model                | 1d          | 2d          | 3d          | 4d          | 5d          | 6d          | 7d          |
| -------------------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- |
| current (NumPy GBM)  | 2.30 / 2.30 | 2.47 / 2.46 | 2.52 / 2.52 | 2.54 / 2.55 | 2.54 / 2.54 | 2.57 / 2.58 | 2.59 / 2.60 |
| lightgbm (reference) | 2.29 / 2.27 | 2.51 / 2.49 | 2.58 / 2.58 | 2.60 / 2.62 | 2.62 / 2.62 | 2.63 / 2.63 | 2.67 / 2.66 |

The naive row is 3.98-4.00 at every horizon. **The feature neither helps
nor hurts DK1**: every cell moves by at most 0.02 ct/kWh, in both
directions, at days 3-7 as much as at days 1-2, so it is noise. DK1 stays
in `ENTSOE_LOAD_REGIONS` (the feature costs one request per week and
nothing in accuracy), but a key is not worth getting for it. Why it carries
so little, as far as the data shows:

- the week-ahead forecast is two numbers per day (minimum and maximum),
  and the demand shape they are stretched over is already in the time and
  sun features;
- ENTSO-E's DK1 series has gaps: 5 of the last 58 ISO weeks are not
  published at all (none from 2026-09-07 on) and the Monday and Sunday are
  `0` (dropped) in the weeks since 2026-08-10, so recent rows are NaN;
- the backtest gives a target day the forecast ENTSO-E keeps for it, which
  is at least as good as what a live forecast run would have had.

```bash
python -m scripts.backtest --region DK1 --window-days 60 --horizon-days 7 --load none
ENTSOE_API_KEY=… python -m scripts.backtest --region DK1 --window-days 60 --horizon-days 7 --load entsoe
```

### Lagged prices (#119)

DK1 and DK2, 365 daily origins from 2025-09-30 to 2026-09-29, 60-day window,
`--horizon-days 7`, retrained daily, with the zone weather. `before` is main
(24 features); the variants add lagged price columns computed from the
visible history (`--lags`, `scripts/backtest_lags.py`), relative to the
slot's **reference day**: the day before a training row's day, the day
before the origin for every target row. MAE in EUR ct/kWh, NumPy GBM.
Recorded 2026-10-02.

**DK1**

| Variant                                      |   1d |   2d |   3d |   4d |   5d |   6d |   7d |
| -------------------------------------------- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| before (24 features)                         | 2.30 | 2.47 | 2.52 | 2.54 | 2.54 | 2.56 | 2.59 |
| all three lags                               | 2.21 | 2.45 | 2.56 | 2.65 | 2.64 | 2.66 | 2.75 |
| `price_same_slot_last_known_day`             | 2.17 | 2.41 | 2.54 | 2.62 | 2.62 | 2.61 | 2.60 |
| `price_mean_last_7_known_days`               | 2.33 | 2.50 | 2.57 | 2.59 | 2.56 | 2.57 | 2.62 |
| `price_same_slot_last_week`                  | 2.31 | 2.48 | 2.53 | 2.53 | 2.55 | 2.57 | 2.59 |
| mean + last week                             | 2.32 | 2.52 | 2.58 | 2.58 | 2.57 | 2.58 | 2.63 |
| same slot, mixed ages 1-7 + `price_lag_days` | 2.35 | 2.44 | 2.53 | 2.57 | 2.59 | 2.53 | 2.60 |
| same slot, mixed ages + age + last week      | 2.34 | 2.42 | 2.50 | 2.55 | 2.58 | 2.54 | 2.64 |
| same slot, mixed ages, no age column         | 2.28 | 2.45 | 2.53 | 2.58 | 2.57 | 2.55 | 2.59 |

**DK2**

| Variant                                      |   1d |   2d |   3d |   4d |   5d |   6d |   7d |
| -------------------------------------------- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| before (24 features)                         | 2.55 | 2.71 | 2.74 | 2.76 | 2.76 | 2.75 | 2.78 |
| all three lags                               | 2.50 | 2.76 | 2.89 | 2.94 | 3.00 | 2.96 | 3.02 |
| `price_same_slot_last_known_day`             | 2.46 | 2.71 | 2.84 | 2.91 | 2.92 | 2.90 | 2.87 |
| `price_mean_last_7_known_days`               | 2.58 | 2.77 | 2.83 | 2.83 | 2.82 | 2.80 | 2.82 |
| `price_same_slot_last_week`                  | 2.55 | 2.72 | 2.75 | 2.77 | 2.78 | 2.78 | 2.80 |
| mean + last week                             | 2.58 | 2.80 | 2.84 | 2.83 | 2.84 | 2.82 | 2.84 |
| same slot, mixed ages 1-7 + `price_lag_days` | 2.56 | 2.69 | 2.76 | 2.78 | 2.80 | 2.79 | 2.82 |
| same slot, mixed ages + age + last week      | 2.55 | 2.67 | 2.71 | 2.76 | 2.79 | 2.79 | 2.89 |
| same slot, mixed ages, no age column         | 2.53 | 2.69 | 2.75 | 2.76 | 2.77 | 2.77 | 2.78 |

LightGBM with all three lags, before / after: DK1 2.29 / 2.15, 2.51 / 2.43,
2.58 / 2.58, 2.60 / 2.72, 2.62 / 2.71, 2.63 / 2.76, 2.67 / 2.84; DK2
2.58 / 2.51, 2.76 / 2.81, 2.83 / 2.92, 2.85 / 3.02, 2.82 / 3.07, 2.83 / 3.06,
2.87 / 3.11. The naive row is 3.97-4.00 (DK1) and 4.18-4.21 (DK2) at every
horizon.

What the numbers say:

- **Yesterday's slot price is a strong day-1 input and a harmful day-3+
  one.** Alone it lowers the 1d MAE by 6 % in DK1 (2.30 → 2.17) and 4 % in
  DK2 (2.55 → 2.46) and helps or ties at 2d, then raises the 3-7d MAE by up
  to 0.08 (DK1) and 0.17 (DK2). One model serves days 1-7: it learns from
  one-day-old lags and is given up to seven-day-old ones at prediction,
  which is the #17 `price_mean` finding per row.
- **The level (7-day mean) hurts everywhere**, by 0.02-0.09. It is constant
  over a day's 96 slots, so the trees can separate days with it and fit
  day-level noise, and at prediction it is up to a week stale.
- **Last week's slot price adds nothing**: within 0.01-0.03 of `before`,
  mostly above it. The model already has what the naive baseline knows.
- **Mixed-age training lags do not rescue it.** Lagging each training day by
  a hashed 1-7 days, with the age as a feature, makes day 1 _worse_ (only a
  seventh of the rows then show a one-day-old lag) and gains 0.03-0.05 at
  2-3 days. Without the age column it is the closest variant: 0.02 better
  at 1-2 days and within ±0.02 of `before` at 3-7 days in both regions,
  which is noise, not a gain.
- **The cross-border model shows the same shape.** DK1 two-stage
  (`--cross-border`) with yesterday's slot price alone scores
  1.85 / 2.01 / 2.07 / 2.12 / 2.14 / 2.13 / 2.19 against
  1.91 / 2.03 / 2.08 / 2.10 / 2.11 / 2.15 / 2.18 without it: better at
  1-3 days, worse at 4, 5 and 7. The neighbours' stage-1 prices already
  carry the level, so the swings are smaller, but the lag still does not
  lower the error at every horizon.
- **Nothing passes the issue's rule** (lower MAE at every horizon in both
  regions), so the integration keeps its 24 features. The day-1 gain is
  real and is the subject of #135 (a near-term model for forecast days
  1-2 next to the plain one for days 3-7).

```bash
python -m scripts.backtest --region DK1 --window-days 60 --horizon-days 7 --lags all
python -m scripts.backtest --region DK1 --window-days 60 --horizon-days 7 --lags price_same_slot_last_known_day,price_lag_days --lag-ages
```

### Baseline

DK1, 365 daily origins from 2025-09-21 to 2026-09-20, retrained daily. MAE
and RMSE in EUR ct/kWh. Recorded 2026-09-25 with `lightgbm==4.7.0`, after the
histogram GBM (#14). The `stumps (before #14)` row is the depth-1 model it
replaced, recorded 2026-09-24. Re-run after the feature rework (#17) with
identical results: the backtest has no weather or Nordpool history, and the
removed `price_mean` was constant within every window.

**180-day window** (EpexPredictor's setting and the target of #24):

| Model                       | 1d MAE | 1d RMSE | 2d MAE | 2d RMSE | 3d MAE | 3d RMSE |
| --------------------------- | -----: | ------: | -----: | ------: | -----: | ------: |
| naive (same slot last week) |   3.91 |    5.73 |   3.95 |    5.82 |   3.95 |    5.83 |
| stumps (before #14)         |   3.74 |    5.13 |   3.76 |    5.15 |   3.79 |    5.21 |
| current (NumPy GBM)         |   3.73 |    5.09 |   3.75 |    5.10 |   3.79 |    5.17 |
| lightgbm (reference)        |   3.74 |    5.10 |   3.75 |    5.10 |   3.79 |    5.17 |

**30-day window** (production's window until #24):

| Model                       | 1d MAE | 1d RMSE | 2d MAE | 2d RMSE | 3d MAE | 3d RMSE |
| --------------------------- | -----: | ------: | -----: | ------: | -----: | ------: |
| naive (same slot last week) |   3.91 |    5.73 |   3.95 |    5.82 |   3.95 |    5.83 |
| stumps (before #14)         |   3.27 |    4.57 |   3.32 |    4.65 |   3.34 |    4.69 |
| current (NumPy GBM)         |   3.27 |    4.62 |   3.30 |    4.67 |   3.31 |    4.70 |
| lightgbm (reference)        |   3.32 |    4.70 |   3.33 |    4.72 |   3.34 |    4.76 |

What the numbers say:

- **The learner is not the bottleneck yet.** On the same (time-only) rows,
  the histogram GBM is level with LightGBM with a 180-day window and 0.03 to
  0.05 ct/kWh better with a 30-day window. With LightGBM-like tree settings
  (`max_depth` 6, 31 leaves, `min_samples_leaf` 20, 200 trees at learning
  rate 0.1) it reproduced the LightGBM row to two decimals at both windows,
  including LightGBM's weaker 30-day result.
- **Deeper trees barely help time-only inputs.** Hour and weekday effects are
  close to additive, so the stumps were already a good fit. The depth-3
  trees match them at 1d MAE, gain up to 0.03 ct/kWh at 2d/3d, and trade a
  slightly higher 30-day 1d RMSE (4.62 vs 4.57) for a lower 180-day one.
  Depth matters once informative inputs (#17, #22, #23) create interactions
  such as low wind **and** evening peak.
- **A longer window hurts a time-only model.** Its only way to follow the
  price level is recency, so 30 days beats 180 days by about 13 % MAE. A
  180-day window (#24) should land together with or after the weather inputs
  (#22, #23), and be judged with this backtest.
- **Both GBMs beat the naive baseline**, by 16 % MAE at 30 days and 4 % at
  180 days. Accuracy barely changes from 1d to 3d, because nothing in the
  input gets staler with lead time.
- **EpexPredictor's README reports a DK1 1d MAE of 1.77 ct/kWh**, with
  weather and other market inputs. Its test period may differ (its backtest
  script at the referenced commit covers 2025-09-01 to 2026-09-01), but that
  is about half of OSF's best 3.27. Closing this gap is the job of the input
  issues (#22, #23, #29), with these tables as the before numbers.
