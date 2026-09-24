# Machine Learning Documentation

## Model

A single **Gradient Boosting** regressor predicts the spot price from 20
features. Implemented in pure NumPy (`numpy_models.py`) — no scikit-learn
dependency.

```
GradientBoosting(
    n_estimators = 200
    learning_rate = 0.1
    max_depth = 5          (decision stumps)
    random_state = 42
)
```

The model is retrained every 6 hours on all accumulated historical data.

## Feature Vector (20 features)

| #   | Feature                | Source         | Description                        |
| --- | ---------------------- | -------------- | ---------------------------------- |
| 0   | `hour`                 | Time           | Hour of day (0-23)                 |
| 1   | `day_of_week`          | Time           | 0=Mon, 6=Sun                       |
| 2   | `is_weekend`           | Time           | 1 if Saturday/Sunday               |
| 3   | `hour_sin`             | Time           | sin(2π × hour / 24)                |
| 4   | `hour_cos`             | Time           | cos(2π × hour / 24)                |
| 5   | `wind_speed_mean`      | Weather entity | Wind speed (m/s)                   |
| 6   | `wind_power_estimate`  | Derived        | Power curve(speed)                 |
| 7   | `wind_direction`       | Weather entity | Wind bearing (0-360°)              |
| 8   | `cloud_coverage`       | Weather entity | Cloud cover (%)                    |
| 9   | `humidity`             | Weather entity | Relative humidity (%)              |
| 10  | `solar_radiation_mean` | Solcast        | Average solar estimate             |
| 11  | `solar_power_estimate` | Solcast        | Total daily solar estimate, scaled |
| 12  | `price_mean`           | Stromligning   | Mean of all known prices           |
| 13  | `temperature`          | Weather entity | Temperature (°C) per slot          |
| 14  | `consumption_forecast` | Nordpool API   | DK1 demand forecast (MW)           |
| 15  | `solar_generation`     | Nordpool API   | Solar generation forecast (MW)     |
| 16  | `wind_offshore`        | Nordpool API   | Offshore wind forecast (MW)        |
| 17  | `wind_onshore`         | Nordpool API   | Onshore wind forecast (MW)         |
| 18  | `net_demand`           | Derived        | consumption - solar - wind (MW)    |
| 19  | `wind_share`           | Derived        | (offshore + onshore) / consumption |

Column order is `FEATURE_NAMES` in `ml/features.py`. Training, prediction,
hyperparameter search and the backtest all build rows with
`build_feature_vector()`, which fills a missing feature with 0 (humidity 50,
temperature 15) and coerces non-numeric values to 0. Time features (0-4) come
from `slot_time_features()`, which is shared by training and prediction.

## Data Sources

| Source                       | Type                | Resolution    | Used for                                |
| ---------------------------- | ------------------- | ------------- | --------------------------------------- |
| `sensor.stromligning_*`      | Confirmed prices    | 15-min        | Price history, self-learning target     |
| `weather.get_forecasts`      | Weather forecast    | Hourly (~48h) | Per-slot wind, temp, cloud, humidity    |
| `sensor.solcast_*`           | Solar forecast      | Hourly        | Solar features, scaled to actual output |
| `weather.forecast_*` (state) | Current weather     | Scalar        | Defaults when forecast unavailable      |
| `Nordpool Consumption API`   | Demand forecast     | Hourly        | Market demand prognosis (MW)            |
| `Nordpool Production API`    | Generation forecast | 15-min        | Solar, wind offshore/onshore (MW)       |
| `sensor.power_inverter_*`    | Actual solar        | Scalar        | Solar scaling factor calibration        |
| `weather_history` (SQLite)   | Actual weather      | 15-min        | Training with ground truth              |

## Nordpool Prognoses

Two public APIs (no authentication) provide the market's own forecasts:

- **ConsumptionPrognoses**: Hourly demand forecast per delivery area
- **ProductionDataPrognoses**: 15-min generation forecast per type (Solar, WindOffshore, WindOnshore)

These are the same inputs used by market participants. They're fetched
before each prediction run (every 6 hours) for today and tomorrow.

Derived features:

```
net_demand = consumption - solar - wind_offshore - wind_onshore
wind_share = (wind_offshore + wind_onshore) / consumption
```

## Training vs Prediction Segmentation

| Phase          | Weather source              | Nordpool source                         | Purpose                 |
| -------------- | --------------------------- | --------------------------------------- | ----------------------- |
| **Training**   | `weather_history` (actuals) | Not used (no historical NP data stored) | Learn real cause→effect |
| **Prediction** | `weather.get_forecasts`     | Nordpool APIs (live)                    | Predict future price    |

## Solar Scaling Factor

A learned EMA ratio between Solcast's estimate and actual inverter output:

```
solar_scale = EMA(actual_power / solcast_estimate)
```

Updated every prediction run. Applied to solar features before they enter
the price model.

## Confidence Score

Confidence adapts based on actual prediction accuracy per 15-minute slot.

**Phase 1 — Heuristic** (< 5 samples):

```
base = 0.80 - wind_penalty - solar_penalty - weekend - days_ahead
Floor: 0.30
```

**Phase 2 — Learned** (≥ 5 samples):

```
confidence = max(0.10, 1.0 - (MAE / mean_actual))
- forecast_temp_error penalty (max -0.15)
- forecast_wind_error penalty (max -0.15)
```

## Self-Learning Loop

Every 15 minutes:

1. Read current Stromligning price
2. Look up prediction made 24h ago for this timestamp
3. Calculate error, update per-slot metrics
4. Update per-slot bias correction via EMA
5. Compare stored forecast weather vs actual → forecast accuracy tracking
6. Remove matched prediction from pending queue

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
| `current (NumPy GBM)`         | `create_price_model()` fitted on rows from `slot_time_features()` + `build_feature_vector()`: the integration's own model and feature code.                                  |
| `lightgbm (reference)`        | LightGBM (500 rounds, learning rate 0.05, 31 leaves, seed 42) on the same rows. Dev-only (`requirements_backtest.txt`): LightGBM has no musllinux wheels, so it cannot ship. |

The `current` row measures the model and features, not the whole runtime
pipeline:

- **No weather or Nordpool history.** Features 5-11 and 13-19 have no
  historical source (`weather_history` and `nordpool_prognoses` hold 30 days
  of actuals, not forecasts). They keep `build_feature_vector()`'s constant
  defaults, and `price_mean` is constant within a window. Both GBMs therefore
  see only the five time features. Historical weather forecasts arrive with
  #22 and #23.
- **Raw model output.** Per-slot bias correction (which needs live
  self-learning state), clamping negative predictions to 0, and
  hyperparameters restored from HPO are not applied.
- **Full window.** `_train_models` fits on the oldest 80 % of its history and
  holds out the newest 20 % for its logged MAE. The backtest fits every model
  on the whole window.

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
and RMSE in EUR ct/kWh. Recorded 2026-09-24 with `lightgbm==4.7.0`.

**180-day window** (EpexPredictor's setting and the target of #24):

| Model                       | 1d MAE | 1d RMSE | 2d MAE | 2d RMSE | 3d MAE | 3d RMSE |
| --------------------------- | -----: | ------: | -----: | ------: | -----: | ------: |
| naive (same slot last week) |   3.91 |    5.73 |   3.95 |    5.82 |   3.95 |    5.83 |
| current (NumPy GBM)         |   3.74 |    5.13 |   3.76 |    5.15 |   3.79 |    5.21 |
| lightgbm (reference)        |   3.74 |    5.10 |   3.75 |    5.10 |   3.79 |    5.17 |

**30-day window** (production keeps `max_history_days = 30`):

| Model                       | 1d MAE | 1d RMSE | 2d MAE | 2d RMSE | 3d MAE | 3d RMSE |
| --------------------------- | -----: | ------: | -----: | ------: | -----: | ------: |
| naive (same slot last week) |   3.91 |    5.73 |   3.95 |    5.82 |   3.95 |    5.83 |
| current (NumPy GBM)         |   3.27 |    4.57 |   3.32 |    4.65 |   3.34 |    4.69 |
| lightgbm (reference)        |   3.32 |    4.70 |   3.33 |    4.72 |   3.34 |    4.76 |

What the numbers say:

- **The learner is not the bottleneck yet.** On the same (time-only) rows,
  the NumPy GBM is within 0.05 ct/kWh of LightGBM at every horizon with a
  180-day window, and slightly better with a 30-day window. Replacing the
  stumps (#14) will matter once informative inputs exist, not before.
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
