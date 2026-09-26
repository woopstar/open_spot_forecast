# Open Spot Forecast Repository Memory

This file is the canonical memory for the Open Spot Forecast (OSF) repository. Read it
before starting any work. It records the module responsibility map, canonical patterns,
feature-vector layout, and the rules that must never be re-invented inline.

## RTK CLI Rule (Mandatory)

Always prefix shell commands with `rtk` (e.g. `rtk git status`, `rtk pytest`). RTK filters
and compresses command output, saving 60-90% of tokens. Meta commands (`rtk gain`,
`rtk discover`, `rtk proxy <cmd>`) are run directly.

## Architecture — Module Responsibilities

### Component layer (`custom_components/open_spot_forecast/`)

| File                 | Responsibility                                                                                                                                             |
| -------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `const.py`           | `DOMAIN`, `CONF_*` keys, `REGIONS`, `PRICE_IN`, `PLATFORMS`, `UPDATE_SIGNAL`                                                                               |
| `config_flow.py`     | Two-step config flow (basic settings → sensor configuration) + options flow                                                                                |
| `sensor.py`          | Price sensors (current, today/tomorrow min/max/mean, ML prediction, confidence, learning metrics)                                                          |
| `accuracy_sensor.py` | Diagnostic forecast MAE/RMSE sensors per lead-time bucket (day 1/2/3/4+)                                                                                   |
| `binary_sensor.py`   | `TomorrowAvailableSensor`, `MLModelTrainedSensor`                                                                                                          |
| `sensor_reader.py`   | `SensorReader` — all external entity reads (Stromligning, weather, Solcast, Met.no)                                                                        |
| `price_series.py`    | `align_to_grid()` (prices by timestamp onto a day's 15-min grid), `is_invalid_price_series()`                                                              |
| `spot_prices.py`     | `ml_price_inputs()` (the model's raw spot prices + where they end), `extract_latest_known_timestamp()`                                                     |
| `time_slots.py`      | 15-min slot arithmetic (floor/ceil, first predicted slot, DST-aware day slots), component + ML                                                             |
| `tomorrow_prices.py` | `TomorrowPriceChecker` — re-reads prices every ~5 min from 13:00 local until tomorrow is complete                                                          |
| `__init__.py`        | Setup and unload: builds the predictor and `ForecastUpdater`, runs the initial fetch, registers timers                                                     |
| `updater.py`         | `ForecastUpdater` — update cycle (15-min / 6-hour / tomorrow poll / midnight), the one `run_forecast()` pipeline; `SensorEntities` (configured entity ids) |
| `history_updater.py` | `HistoryUpdaterMixin` — background backfill (day-ahead price days, Nordpool prognoses) and daily retention of stored history                               |
| `price_source.py`    | `PriceSettings` (price source, currency, VAT, ENTSO-E key) and `DayAheadPrices` (fetch, convert, history) for the `dayahead` source                        |
| `attribution.py`     | `price_attribution()` / `model_attribution()` and their entity mixins: every entity credits its data sources (#41)                                         |

### ML layer (`custom_components/open_spot_forecast/ml/`)

| File                    | Responsibility                                                                                                                      |
| ----------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| `predictor.py`          | `SpotPricePredictor` — composes `FeatureMixin` + `ModelMixin` + `LearningMixin` + `CatchUpMixin` + `LeadTimeMixin` + `RetrainMixin` |
| `features.py`           | `FeatureMixin` — feature extraction (wind, solar, time, Nordpool prognoses)                                                         |
| `zone_weather.py`       | `ZoneWeatherIndex` — Open-Meteo point rows aggregated per slot into the zone features (#22)                                         |
| `models.py`             | `ModelMixin` — training + prediction                                                                                                |
| `learning.py`           | `LearningMixin` — self-learning, bias correction, error metrics                                                                     |
| `catch_up.py`           | `CatchUpMixin` — startup replay of stored predictions against known prices (`catch_up_learning`)                                    |
| `gbm.py`                | `NumpyGradientBoosting` — the price model: histogram GBM (binned features, leaf-wise depth-limited trees, native NaN)               |
| `numpy_models.py`       | `NumpyRandomForest` and other legacy pure NumPy models                                                                              |
| `storage.py`            | `LearningStorage` — SQLite connection, write lock, schema and migrations; composes the storage mixins below                         |
| `storage_base.py`       | `StorageMixinBase` — type-only declarations (`_lock`, `_ensure_conn()`, `last_data_write`) shared by the storage mixins             |
| `prediction_storage.py` | `PredictionStorageMixin` — `predictions` table (pending predictions awaiting self-learning)                                         |
| `history_storage.py`    | `HistoryStorageMixin` — `weather_history`, `nordpool_prognoses` and `price_history` tables                                          |
| `series_storage.py`     | `SeriesStorageMixin` + `SeriesSpec` — generic time-series tables: stored grid points, change-detecting upsert, load, prune, state   |
| `state_storage.py`      | `LearningStateStorageMixin` — `error_metrics`, `bias_correction`, `volatility`, `meta`, bulk `save_all` / `load_all`                |
| `accuracy_storage.py`   | `LeadTimeAccuracyStorageMixin` — `lead_time_accuracy` table, mixed into `LearningStorage`                                           |
| `retraining.py`         | `RetrainMixin` — retrain when training data changed, HPO cadence                                                                    |
| `lead_time.py`          | `LeadTimeMixin` — lead-time bucketing + rolling MAE/RMSE per bucket                                                                 |

### API layer (`custom_components/open_spot_forecast/api/`)

| File                    | Responsibility                                                                                                          |
| ----------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| `nordpool_data.py`      | `fetch_consumption_prognosis`, `fetch_production_prognosis` — Nordpool public APIs                                      |
| `nordpool_prognoses.py` | `NordpoolPrognosisSource` — both prognoses as `nordpool_prognoses` rows, one request per missing CET delivery day       |
| `time_series_source.py` | `TimeSeriesSource` — gap-aware incremental updates shared by every upstream time series (#32)                           |
| `dayahead_prices.py`    | `DayAheadPriceSource` — energy-charts (+ ENTSO-E fallback) day-ahead prices as `dayahead_prices` rows (#27)             |
| `openmeteo_weather.py`  | `OpenMeteoWeatherSource` — Open-Meteo 15-min weather at the region's `WEATHER_POINTS` as `openmeteo_weather` rows (#22) |
| `exchange_rates.py`     | `ExchangeRates` — ECB EUR reference rates by day (DKK peg fallback)                                                     |
| `http.py`               | `async_get` — the one GET with retries/backoff/`Retry-After` for every API client; never logs URLs or params            |

## Canonical Patterns — Use These, Never Re-Invent

### Domain & config constants

`DOMAIN = "open_spot_forecast"`. All config keys (`CONF_REGION`, `CONF_CURRENCY`,
`CONF_VAT`, `CONF_PRECISION`, `CONF_PRICE_TYPE`, `CONF_STROMLIGNING_SENSOR`,
`CONF_ENABLE_ML_PREDICTION`, weather/solar sensor keys) live in `const.py`. Region
definitions and price-unit conversions (`PRICE_IN`) also live there. Never hard-code a
config key, region name, or price-unit factor elsewhere.

### Sensor reading

All external entity reads go through `SensorReader` in `sensor_reader.py`. Never call
`hass.states.get(...)` directly in platform or ML code. Methods:
`read_stromligning_sensor`, `read_stromligning_tomorrow_sensor`, `read_spot_prices`, `read_weather_sensors`,
`read_solcast_sensor`, `read_met_weather`.

**The ML model's prices are the raw day-ahead spot price excl. VAT and tariffs** (#16),
from `read_spot_prices()` (Stromligning's `spotprice_ex_vat` sensors) via `ml_price_inputs()`:
training target, self-learning actual and prediction. Stromligning's all-in consumer price
is display-only; never feed it to the model. VAT is added once, in `MLPredictionSensor`
(`_with_vat`); never add tariffs or VAT in `ml/`.

With the `dayahead` price source (#27) the model's prices are the stored day-ahead auction
prices (`dayahead_prices`, EUR/MWh) converted by `dayahead_spot_data()` / `dayahead_prices_by_day()`
(`spot_prices.py`) with the day's ECB rate: the same currency/kWh excl. VAT series. The displayed
prices are these with VAT (`with_vat()`); Stromligning is not read. Every HTTP client uses
`api/http.py` `async_get`; never add another retry loop.

A day's prices are one value per 15-min slot from local midnight (92/96/100), `None` for a
slot missing in the source: the readers place items by their own timestamps with
`align_to_grid()` (`price_series.py`; hourly expanded, gaps up to 4 slots filled). Use
`known_prices()` before min/max/mean and skip `None` slots in training and learning; never
guess the resolution from a list's length. `is_invalid_price_series()` rejects a day whose known
prices are all zero or not finite; the readers drop it and `store_daily_prices` / `predict`
refuse it. Never add an inline "all prices are 0" check.

### Time-series sources

Every upstream time series (Nordpool prognoses now; day-ahead prices and
Open-Meteo weather next) is a `TimeSeriesSource` subclass (`api/time_series_source.py`)
over a `SeriesSpec` table (`ml/series_storage.py`, created in `LearningStorage._create_schema`).
Subclasses implement `_fetch(start, end)` (rows, `[]` for "nothing", `None` for a failed
request) and may override `chunks()` / `max_request_span`, `refresh_from()` (revised
forecasts), `keys()` (keyed tables) and `retry_time()`. `async_update(start, end)` requests
only missing grid points, skips remembered holes until their retry time (the horizon is
open-ended), upserts (a change moves `last_data_write` → retrain) and persists the state in
`meta`. Range helpers (`missing_ranges`, `split_range`, …) and `SourceState` live in
`time_series.py`. Never re-fetch complete history outside a source's refresh window, and never
prune on a fixed day count: `ForecastUpdater.prune_history()` keeps the training window plus
`HISTORY_MARGIN_DAYS`.

### ML predictor

`SpotPricePredictor` in `ml/predictor.py` is the single ML predictor. It composes
`FeatureMixin`, `ModelMixin`, `LearningMixin`, `CatchUpMixin`, `LeadTimeMixin`, and `RetrainMixin`. Never re-implement
feature extraction, model training, or self-learning outside `ml/`.

### Retraining

The model retrains when its inputs change, not on a timer: `predict()` stores today's prices
via `record_training_prices()` and retrains only if `needs_retraining()` (untrained, or
`last_data_update > last_trained_at`). New training-data writes must move
`LearningStorage.last_data_write` (or `_prices_updated_at` for prices), or they will never
reach the model. HPO runs once per 7 new price days (`hpo_counter` in `meta`).

Pure helpers shared by training, prediction and the dev backtest — never inline them:
`build_feature_row()` (the one definition of every feature), `build_feature_vector()`
(model input row, column order `FEATURE_NAMES`) and `slot_time_features()` (per-slot time
features) in `ml/features.py`, and
`create_price_model()` (production GBM hyperparameters, also used by HPO) in `ml/models.py`.
The price model handles NaN inputs natively (each split learns where missing values go),
so a missing feature can reach it as NaN instead of an invented value.

### Update cycle

`ForecastUpdater` in `updater.py` owns every update path; `async_setup_entry` only builds it,
runs `async_initial_fetch()` and registers its callbacks. The forecast pipeline (weather →
Nordpool prognoses → known-data end → `predict` → `save_learning_data`) exists once, in
`ForecastUpdater.run_forecast()`, used by the initial fetch, the tomorrow-price refresh and the
6-hourly update. Never copy it into another callback.

### Model backtest

`scripts/backtest.py` (dev-only, never shipped) is the rolling 1/2/3-day-ahead MAE/RMSE
backtest: naive last-week baseline, current NumPy GBM, optional LightGBM reference
(`requirements_backtest.txt`, never `manifest.json`). It imports the helpers above and
enforces a strict horizon cutoff. Run `./scripts/quality.sh backtest` before and after any
model or feature change; the baseline is in `docs/ml_documentation.md` → Backtesting.

### Learning storage

`LearningStorage` in `ml/storage.py` is the single SQLite persistence layer. Never open a
raw `sqlite3` connection or write to the learning DB outside this class. Its table
operations live in mixins (`prediction_storage.py`, `history_storage.py`,
`state_storage.py`, `accuracy_storage.py`) that inherit `StorageMixinBase`
(`ml/storage_base.py`); add a new table's methods to the matching mixin (or a new one),
not to `storage.py`.

### Floating-point comparisons

Production code uses an epsilon guard (`abs(x) > 1e-9` instead of `x != 0`). Tests use
`pytest.approx()`.

## Feature Vector (17 features)

The canonical feature vector is defined in `docs/ml_documentation.md`. Every row, training
and prediction alike, comes from `build_feature_row(slot_start, SlotInputs)` in
`ml/features.py`, then `build_feature_vector()` in `FEATURE_NAMES` order. Only the inputs
differ: `TrainingInputs` (`ml/training_inputs.py`, stored zone weather + `nordpool_prognoses`,
matched by UTC epoch) vs `FeatureMixin._combine_features()` (stored/live forecasts, matched by
UTC hour). Both phases use forecasts (#23): past zone weather is Open-Meteo's archive of past
forecasts; the local weather entity is not a model input (its forecast is only recorded with
predictions, and `weather_history` snapshots score it and never trigger a retrain). The zone weather (#22) comes from the stored
`openmeteo_weather` rows in both phases, aggregated over the region's `WEATHER_POINTS` by
`ZoneWeatherIndex` (`ml/zone_weather.py`); never aggregate it inline. Unknown inputs are `None` → NaN; never fill in
0/15 °C/50 % or the current observation, and never copy prediction values into training rows.
Wind speed is m/s in both phases (`wind_speed_to_ms()` in `sensor_reader.py`).

| #   | Feature                | Source       |
| --- | ---------------------- | ------------ |
| 0   | `hour`                 | Time         |
| 1   | `day_of_week`          | Time         |
| 2   | `is_weekend`           | Time         |
| 3   | `hour_sin`             | Time         |
| 4   | `hour_cos`             | Time         |
| 5   | `consumption_forecast` | Nordpool API |
| 6   | `solar_generation`     | Nordpool API |
| 7   | `wind_offshore`        | Nordpool API |
| 8   | `wind_onshore`         | Nordpool API |
| 9   | `net_demand`           | Derived      |
| 10  | `wind_share`           | Derived      |
| 11  | `zone_wind`            | Open-Meteo   |
| 12  | `zone_wind_power`      | Open-Meteo   |
| 13  | `zone_temperature`     | Open-Meteo   |
| 14  | `zone_irradiance`      | Open-Meteo   |
| 15  | `zone_pressure`        | Open-Meteo   |
| 16  | `zone_humidity`        | Open-Meteo   |

Adding or removing a feature is a model change — see the `osf-ml-change` skill and update
`docs/ml_documentation.md`.

## Slot Granularity — 96 Slots

The system models **96 slots per day** (15-minute intervals), not 24 hours. Per-slot bias
correction and error metrics are keyed `0-95`. Never assume hourly (0-23) granularity.

Slot boundary arithmetic lives in `time_slots.py`: `floor_to_slot()`, `ceil_to_slot()`,
`first_prediction_slot()` (where predictions start), `slots_in_local_day()` (96, or 92/100 on
DST days), `tomorrow_prices_complete()` (the only "tomorrow is available" check), and
`slot_start_in_day()` / `slot_index_in_day()` (slot n of a local day and back, stepped in UTC
from local midnight). Stored slot timestamps are written with `utc_slot_key()` (UTC slot start,
`YYYY-MM-DDTHH:MM:SSZ`) and read with `parse_utc()` (naive = HA local time); storage lookups
compare instants (SQLite `julianday()` of the stored key against window bounds computed in
Python), never ISO strings with different offsets or a float julian-day distance; the
Nordpool lookup returns the row of the slot's UTC hour, as `TrainingInputs` does. Never build slot times as `date + n * 15 min` or index a day's prices
with `hour * 4 + minute // 15`, and never call naive `datetime.now()` — use `dt_util.now()`. They round on the UTC timeline, so
never round with `dt.replace(minute=...)` or add minutes to a local datetime inline.

## Bias Correction Formula

Each 15-minute slot has an additive offset (currency/kWh) learned via EMA (#15):

```
raw_bias = offset[slot] + mean_error      # mean_error = mean(predicted - actual)
offset[slot] = 0.9 * offset[slot] + 0.1 * raw_bias
corrected_price = raw_price - offset[slot]
```

`offset > 0` → model overpredicts → subtract. `offset < 0` → model underpredicts → add.
`mean_error` comes from stored (already corrected) predictions, so `offset + mean_error` is
the raw model's bias; an EMA of `mean_error` alone would settle at half the bias. The first
update sets the offset to `mean_error`. Prices can be negative: never clamp predictions at
0 and never divide by a price. Never invent a different correction scheme.

## Solar Scaling Factor

A learned EMA ratio between Solcast's estimate and actual inverter output:

```python
solar_scale = EMA(actual_power / solcast_estimate)
```

Updated every prediction run and persisted, but not applied to the price model since #17
(there is no site-solar feature; scaling only prediction rows would break consistency).

## Confidence Score

- **Phase 1 — Heuristic** (< 5 samples): `base = 0.80 - wind_penalty - solar_penalty -
weekend - days_ahead`, floor `0.30`.
- **Phase 2 — Learned** (≥ 5 samples): `confidence = max(0.10, 1.0 - (MAE / mean(|actual|)))`
  minus forecast temperature/wind error penalties (max `-0.15` each).

## Storage Schema

SQLite database at `/config/.storage/open_spot_forecast_{region}_learning.db`.

| Table                | Key                   | Content                                                                 |
| -------------------- | --------------------- | ----------------------------------------------------------------------- |
| `predictions`        | `id` (autoincrement)  | Pending predictions awaiting comparison                                 |
| `error_metrics`      | `hour` (0-95)         | Per-slot error arrays                                                   |
| `bias_correction`    | `hour` (0-95)         | Per-slot additive bias offsets (schema v5)                              |
| `spot_prices`        | `timestamp` (UTC key) | The model's price history per UTC slot (#24); `price_history` is legacy |
| `weather_history`    | `timestamp` (UTC key) | 15-min weather snapshots, keyed by UTC slot start (`…Z`)                |
| `meta`               | `key`                 | Training state, schema version                                          |
| `lead_time_accuracy` | `(date, bucket)`      | Daily per-lead-time error sums (rolling 30 days)                        |

Migrations are versioned in `meta.schema_version` and run once at startup: v5 resets the
multiplicative bias factors (`ml/bias_storage.py`), v6 discards consumer-price learning
data (`ml/spot_migration.py`), v7 rewrites weather snapshot timestamps as UTC slot keys
(`ml/weather_migration.py`), v8 stores the price history per UTC slot (`ml/price_storage.py`). The legacy JSON format
(`open_spot_forecast_DK1_learning.json`) only contributes its training state.

## File Size Rules

Hard limit: **30 KB AND 1000 lines** per file across the entire codebase. Both limits must
be satisfied. If a file exceeds either, split it before adding more features.

## Documentation Style

Docs live in `docs/` and are organized by responsibility, not by theme. The canonical docs
are `architecture.md`, `ml_documentation.md`, `self_learning.md`, `persistence.md`,
`stromligning_integration.md`, and `using_existing_sensors.md`. Any change that alters
behaviour must update the docs that describe it in the same PR.

## Sensor Wiring

To wire a new external entity into OSF, follow the full stack in order:

1. `const.py` — add a `CONF_*` key (and default entity-id string where sensible)
2. `config_flow.py` — add to the `sensors` step schema and options flow `init` step
3. `translations/en.json` — add `data` label for both `config.step.sensors` and `options.step.init`
4. `translations/da.json` — add the Danish translation
5. `sensor_reader.py` — add a `read_*` method on `SensorReader`
6. `updater.py` — add the entity to `SensorEntities` (and `sensor_config()`), read the value in
   `ForecastUpdater._read_weather()` and pass it into `weather_data`

Always check `docs/using_existing_sensors.md` first for the verified entity list.

## Testing Rules

- Use mock-based approach (`MagicMock`/`AsyncMock`) — compatible with Windows.
- Config flow tests: fresh install, reconfigure, abort-on-duplicate, validation errors.
- Entity tests: assert `unique_id`, `device_info`, `native_value`,
  `native_unit_of_measurement`, `state`, `extra_state_attributes`.
- Float comparisons: `pytest.approx()` in tests, epsilon guard in production.
- ML tests: feature extraction, model output shape, bias-correction convergence, storage
  round-trip, schema migration.

## Logging

- Use `%`-formatting for logging (never f-strings) to avoid evaluating the string when the
  log level is suppressed.
- No component/platform name in log messages (added automatically).
- No period at the end of log messages.
- Never log API keys, tokens, usernames, or passwords.
- Restrict `_LOGGER.info` — use `_LOGGER.debug` for non-user-facing details.

## GitHub Operations — `gh` CLI Is Available

The `gh` CLI is installed and authenticated in the devcontainer (`/usr/bin/gh`). Use it for
GitHub API operations (PRs, issues, reviews, branches, releases). GitHub MCP tools are not
present in every session — never assume they exist. Prefer `rtk gh ...` to cut output
tokens. Pass multiline bodies via `--body-file`, never as an inline shell argument.
