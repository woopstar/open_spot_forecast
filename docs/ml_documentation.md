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

## Retraining

The model is retrained on all accumulated historical data **when its training
inputs have changed**, not on a fixed schedule (`ml/retraining.py`). Every
forecast run (startup, every 6 hours, ~13:xx, and as soon as the 15-minute
update sees tomorrow's prices appear) first stores today's known prices, then
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

**Hyperparameter optimization** (a grid search over `n_estimators` and
`learning_rate`) runs once per 7 new days of price data. The day counter is
persisted as `hpo_counter` in the `meta` table, so it survives restarts. After
optimization the model is refitted with the best parameters in the same run.

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
