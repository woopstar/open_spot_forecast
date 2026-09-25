# Machine Learning Documentation

## Model

A single **Gradient Boosting** regressor predicts the spot price from 20
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

These values were chosen with the [backtest](#backtesting): with the current
inputs, deeper or less regularized trees (e.g. `max_depth` 6,
`learning_rate` 0.1, `min_samples_leaf` 20) fit the noise of single
weekday/slot cells and lose about 0.05 ct/kWh 1d MAE on a 30-day window.
Hyperparameter optimization can raise `max_depth` per installation.

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
  - a weather snapshot written to `weather_history` (every 15 minutes)
  - a Nordpool prognosis row whose values changed (re-sending identical
    prognoses does not count)
- `last_trained_at` — when the last successful training **started**, so data
  written during a training run triggers the next one

The model retrains if it is untrained or `last_data_update > last_trained_at`;
otherwise the existing model is reused. In an install with a weather sensor a
snapshot lands every 15 minutes, so in practice each scheduled run retrains;
without new data (e.g. two runs back to back) it does not. Forecast runs are
serialized so two retrains never overlap.

Trained trees are kept in memory only, so the first forecast after a restart
always retrains from the persisted history.

**Hyperparameter optimization** (a grid search over `n_estimators` ∈ {100,
200, 300}, `learning_rate` ∈ {0.05, 0.1, 0.2} and `max_depth` ∈ {2, 3, 4, 6})
runs once per 7 new days of price data. Each (`learning_rate`, `max_depth`)
pair is fitted once with 300 trees, and `staged_predict` scores it after 100,
200 and 300 trees, so the 36 candidates cost 12 fits. The day counter is
persisted as `hpo_counter` in the `meta` table, so it survives restarts. After
optimization the model is refitted with the best parameters in the same run.
The best parameters are stored as `hpo_n_estimators`, `hpo_learning_rate` and
`hpo_max_depth` in `meta` and restored at startup; parameters stored without
`hpo_max_depth` were tuned for the old depth-1 stump model and are ignored
until the next optimization.

## Feature Vector (17 features)

| #   | Feature                | Source             | Description                                   |
| --- | ---------------------- | ------------------ | --------------------------------------------- |
| 0   | `hour`                 | Time               | Hour of day (0-23)                            |
| 1   | `day_of_week`          | Time               | 0=Mon, 6=Sun                                  |
| 2   | `is_weekend`           | Time               | 1 if Saturday/Sunday                          |
| 3   | `hour_sin`             | Time               | sin(2π × hour / 24)                           |
| 4   | `hour_cos`             | Time               | cos(2π × hour / 24)                           |
| 5   | `wind_speed_mean`      | Weather entity     | Wind speed in the slot (m/s)                  |
| 6   | `wind_power_estimate`  | Derived            | Power curve(wind speed), 0-1                  |
| 7   | `wind_direction`       | Weather entity     | Wind bearing (0-360°)                         |
| 8   | `cloud_coverage`       | Weather entity     | Cloud cover (%)                               |
| 9   | `humidity`             | Weather entity     | Relative humidity (%)                         |
| 10  | `temperature`          | Weather entity     | Temperature in the slot                       |
| 11  | `consumption_forecast` | Nordpool prognosis | Demand prognosis for the slot's hour (MW)     |
| 12  | `solar_generation`     | Nordpool prognosis | Solar prognosis at the slot's hour start (MW) |
| 13  | `wind_offshore`        | Nordpool prognosis | Offshore wind prognosis, same hour start (MW) |
| 14  | `wind_onshore`         | Nordpool prognosis | Onshore wind prognosis, same hour start (MW)  |
| 15  | `net_demand`           | Derived            | consumption - solar - offshore - onshore (MW) |
| 16  | `wind_share`           | Derived            | (offshore + onshore) / consumption            |

Column order is `FEATURE_NAMES` in `ml/features.py`.

**One definition for training and prediction** (#17). Every row, for
training, prediction, hyperparameter search and the backtest, is built by
`build_feature_row(slot_start, SlotInputs)` in `ml/features.py` and turned
into the model input by `build_feature_vector()`. The two phases differ only
in where a slot's `SlotInputs` come from (see
[Training vs Prediction Segmentation](#training-vs-prediction-segmentation)).
Time features (0-4) come from `slot_time_features()`; derived features
(6, 15, 16) are computed from the slot's own inputs.

**Missing inputs are NaN.** An input that is unknown for a slot (no weather
forecast that far ahead, no stored snapshot for a training slot, Nordpool
prognoses only exist for today and tomorrow) is `None` in the feature dict
and NaN in the model input, and so is every derived feature that needs it.
The price model handles NaN natively (see [Model](#model)). Nothing is
replaced by 0, 15 °C, 50 % humidity or the current observation. Rows whose
target price is missing are not training rows.

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
  Irradiance arrives with #22.

## Data Sources

| Source                                              | Type                | Resolution   | Used for                                               |
| --------------------------------------------------- | ------------------- | ------------ | ------------------------------------------------------ |
| `sensor.stromligning_spotprice_ex_vat` (+ tomorrow) | Raw spot price      | 15-min       | Training target, self-learning actuals (excl. VAT)     |
| `weather.get_forecasts`                             | Weather forecast    | Hourly       | Prediction: per-slot wind (m/s), temp, cloud, humidity |
| `weather.forecast_*` (state)                        | Current weather     | Every 15 min | `weather_history` snapshots (training)                 |
| `Nordpool Consumption API`                          | Demand forecast     | Hourly       | Market demand prognosis (MW), both phases              |
| `Nordpool Production API`                           | Generation forecast | 15-min       | Solar, wind offshore/onshore (MW), both phases         |
| `sensor.solcast_*`                                  | Solar forecast      | Daily total  | Solar scaling factor only (not a model input)          |
| `sensor.power_inverter_*`                           | Actual solar        | Scalar       | Solar scaling factor only (not a model input)          |
| `weather_history` (SQLite)                          | Actual weather      | 15-min       | Training inputs                                        |
| `nordpool_prognoses` (SQLite)                       | Stored prognoses    | Hourly       | Training inputs                                        |

Wind speed is converted to m/s from the weather entity's `wind_speed_unit`
(default km/h) by `wind_speed_to_ms()` in `sensor_reader.py`, for the stored
snapshots and for the hourly forecast alike.

## Nordpool Prognoses

Two public APIs (no authentication) provide the market's own forecasts:

- **ConsumptionPrognoses**: Hourly demand forecast per delivery area
- **ProductionDataPrognoses**: 15-min generation forecast per type (Solar, WindOffshore, WindOnshore)

These are the same inputs used by market participants. They're fetched
before each prediction run (every 6 hours) for today and tomorrow, stored in
`nordpool_prognoses` (one row per hour: the hour's consumption and the
production of its first quarter), and backfilled for the days in
`price_history` at startup. Training reads those stored rows; prediction
reads the live prognoses at the same resolution: the hour's consumption and
the production at the hour's start, for all four slots of the hour.

Derived features:

```
net_demand = consumption - solar - wind_offshore - wind_onshore
wind_share = (wind_offshore + wind_onshore) / consumption
```

## Training vs Prediction Segmentation

| Phase          | Weather source                                    | Nordpool source                       | Purpose                 |
| -------------- | ------------------------------------------------- | ------------------------------------- | ----------------------- |
| **Training**   | `weather_history` snapshot taken in the slot      | `nordpool_prognoses` row for the hour | Learn real cause→effect |
| **Prediction** | `weather.get_forecasts` entry for the slot's hour | Live prognoses for the slot's hour    | Predict future price    |

Training reads both tables once per fit (`TrainingInputs` in
`ml/training_inputs.py`) and matches rows to slots on their UTC epoch.
Weather snapshots are stored with their UTC offset; older snapshots without
one are read as local time. Prediction matches forecasts and prognoses on
the slot's UTC hour, so HA's UTC forecast timestamps line up with local slot
times. Training never uses the current forecast's values.

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

`_train_models` builds one row per 15-minute slot of `price_history` (up to
30 days), oldest first, and uses the rows twice:

1. **Holdout validation.** A copy of the price model with the same
   hyperparameters is fitted on the oldest 80 % of the rows and scored on the
   newest 20 %. The holdout MAE and RMSE are logged
   (`ML model trained: holdout MAE=…, RMSE=…`); the copy is then discarded.
2. **Live model.** `price_model` is fitted on 100 % of the rows. The most
   recent days are the most similar to the days being predicted, so they
   must be part of the model: fitting on the oldest 80 % only would ignore
   the newest ~6 of 30 days. The backtest's `current` row (see
   [Backtesting](#backtesting)) also fits on its whole window.

The split is chronological, never shuffled, so the holdout rows are always
later than the rows the copy was fitted on, as in a real forecast. The extra
fit roughly doubles training time. Like the rest of training, it runs in the
executor.

Hyperparameter optimization compares its candidates on the same chronological
80/20 split, then replaces `price_model` with an unfitted model using the best
parameters, which the next training fits on all rows.

## Target: Raw Spot Price, VAT at Output

The model is trained on, learns from and predicts the **raw day-ahead spot
price excl. VAT and tariffs**, in currency/kWh (#16), read from Stromligning's
spot price sensors (`read_spot_prices()`; see
[Stromligning Integration](stromligning_integration.md#overview)).
`price_history`, stored predictions, error metrics and bias offsets are all in
that unit. Tariffs are time-of-use and seasonal; in the target they would be
learned as if they were market behaviour. The consumer price (tariffs, fees
and VAT included) is only displayed.

VAT is applied once, at output: the `Price Forecast (ML)` sensor adds the
configured VAT to its state and to every price attribute, and says so with
`includes_vat: true`, `includes_tariffs: false`. Tariffs at output are #39.

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
solar feature (see [Feature Vector](#feature-vector-17-features)), and a
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
4. Update the slot's additive bias offset via EMA (see
   [self-learning](self_learning.md#bias-correction))
5. Compare stored forecast weather vs actual → forecast accuracy tracking
6. Remove matched predictions from pending queue
7. Add each error to its lead-time bucket (day 1/2/3/4+) → rolling 30-day
   MAE/RMSE per lead time (see [self-learning](self_learning.md))

**96 slots** (15-min intervals), not 24 hours. Each slot has independent
bias correction.

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
  `.cache/backtest/`. The OSF SQLite DB cannot be used, because it keeps only
  30 days of `price_history`.

### Models

| Row                           | Model                                                                                                                                                                        |
| ----------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `naive (same slot last week)` | Price of the same local wall-clock slot seven days earlier. The bar every model must clear.                                                                                  |
| `current (NumPy GBM)`         | `create_price_model()` fitted on rows from `build_feature_row()` + `build_feature_vector()`: the integration's own model and feature code.                                   |
| `lightgbm (reference)`        | LightGBM (500 rounds, learning rate 0.05, 31 leaves, seed 42) on the same rows. Dev-only (`requirements_backtest.txt`): LightGBM has no musllinux wheels, so it cannot ship. |

The `current` row measures the model and features, not the whole runtime
pipeline:

- **No weather or Nordpool history.** Features 5-16 have no source for a
  year of history (`weather_history` and `nordpool_prognoses` keep 30 days).
  The backtest passes empty `SlotInputs`, so they are NaN in every row, and
  both GBMs see only the five time features. Historical weather forecasts
  arrive with #22 and #23.
- **Raw model output.** Per-slot bias correction (which needs live
  self-learning state) and
  hyperparameters restored from HPO are not applied.

### Running

```bash
pip install -r requirements_backtest.txt   # optional LightGBM row
./scripts/quality.sh backtest --region DK1  # last 365 origins, 180-day window
python -m scripts.backtest --region DK1 --start 2025-09-21 --end 2026-09-20 --window-days 30
```

`--region` accepts DK1, DK2, SE4, NL, BE, FR and DE. Run the backtest before
and after every model or feature change, and put both tables in the PR.

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

**30-day window** (production keeps `max_history_days = 30`):

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
