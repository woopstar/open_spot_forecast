# Open Spot Forecast — Architecture

## System Overview

Open Spot Forecast is a Home Assistant integration that predicts electricity
spot prices using machine learning, weather forecasts, and confirmed market
data. The system runs a self-learning loop that compares predictions against
actual prices and continuously improves accuracy via per-slot bias correction.

## Data Sources

| Source                                      | Type                             | Used for                                         |
| ------------------------------------------- | -------------------------------- | ------------------------------------------------ |
| `sensor.stromligning_current_price_vat`     | Confirmed prices (96/day)        | Price history, self-learning target              |
| `binary_sensor.stromligning_tomorrow_*`     | Tomorrow's prices when available | Known data window extension                      |
| `weather.forecast_mellemlokken_23` (state)  | Current weather snapshot         | Wind, temperature, humidity, cloud               |
| `weather.get_forecasts` (hourly)            | 48h weather forecast             | Per-slot wind/temp/cloud/humidity for prediction |
| `sensor.solcast_pv_forecast_forecast_today` | Solar generation forecast        | Solar scaling factor (not a model input)         |
| `sensor.power_inverter_input_total`         | Current solar production         | Solar scaling factor (not a model input)         |
| `sensor.metroair_330_outdoor_temperature`   | Actual outdoor temperature       | Historical temperature for training              |

## Component Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                     Home Assistant                               │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │                Open Spot Forecast                          │  │
│  │                                                            │  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────────────────────┐ │  │
│  │  │  Sensor  │  │  Binary  │  │      Config Flow         │ │  │
│  │  │  Entity  │  │  Sensor  │  │                          │ │  │
│  │  └────┬─────┘  └────┬─────┘  └──────────────────────────┘ │  │
│  │       │              │                                      │  │
│  │       └──────┬───────┘                                      │  │
│  │              │                                              │  │
│  │   ┌──────────▼──────────┐                                   │  │
│  │   │   Sensor Reader     │  ← reads HA entities directly    │  │
│  │   │  • Stromligning     │                                   │  │
│  │   │  • Weather entity   │                                   │  │
│  │   │  • Solcast          │                                   │  │
│  │   │  • weather.get_     │                                   │  │
│  │   │    forecasts         │                                   │  │
│  │   └──────────┬──────────┘                                   │  │
│  │              │                                              │  │
│  │   ┌──────────▼──────────┐                                   │  │
│  │   │   ML Predictor      │                                   │  │
│  │   │                     │                                   │  │
│  │   │  Price Model:       │  14 features → spot price        │  │
│  │   │  GradientBoosting   │  (pure numpy, no sklearn needed) │  │
│  │   │  200 trees, lr=0.1  │                                   │  │
│  │   │                     │                                   │  │
│  │   │  Feature Mixin      │  wind/solar/time extraction      │  │
│  │   │  Learning Mixin     │  self-learning + bias correction │  │
│  │   │  Model Mixin        │  training + prediction           │  │
│  │   └──────────┬──────────┘                                   │  │
│  │              │                                              │  │
│  │   ┌──────────▼──────────┐                                   │  │
│  │   │   Learning Storage  │  SQLite (persistent)             │  │
│  │   │                     │                                   │  │
│  │   │  • predictions      │  pending forecast → actual       │  │
│  │   │  • error_metrics    │  per-slot (0-95) tracking       │  │
│  │   │  • bias_correction  │  per-slot additive offsets      │  │
│  │   │  • price_history    │  daily prices for training      │  │
│  │   │  • weather_history  │  15-min weather snapshots       │  │
│  │   │  • meta             │  schema version, training state │  │
│  │   └─────────────────────┘                                   │  │
│  └────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

## Data Flow

```
Every 15 min ──→ Read Stromligning prices
             │   Read weather snapshot → store in weather_history
             │   Read tomorrow prices if available
             │   Self-learning: compare all predictions for the current
             │   slot vs actual, per-lead-time MAE/RMSE (day 1/2/3/4+)
             │   Tomorrow's prices just completed → refresh the forecast now
             │
Every 6 hours → Read weather forecast (weather.get_forecasts)
             │   Retrain model if training data changed since last training
             │   Generate 672 predictions (7 days × 96 slots)
             │   Store predictions in SQLite for future learning
             │   Apply per-slot bias corrections
             │
From 13:00 ──→ Re-read prices every ~5 min until tomorrow is complete
             │   (every slot of the next local day: 96, or 92/100 on DST days;
             │   gives up at 18:00 and tries again the next day)
             │   Once complete: extend known-data window, retrain model
             │   (price history changed), regenerate predictions
             │
Midnight ────→ Rotate tomorrow → today
```

## Model: Single Price Predictor

The system uses **one model** — a Gradient Boosting regressor that takes 17
features and directly predicts the spot price. Wind, solar, and temperature
are input features, not separate sub-models. Training and prediction rows
come from the same `build_feature_row()`; an unknown input is NaN (see
[ML Documentation](ml_documentation.md#feature-vector-17-features)).

```
Features (17):
  [hour, day_of_week, is_weekend, hour_sin, hour_cos,
   wind_speed_mean, wind_power_estimate, wind_direction,
   cloud_coverage, humidity, temperature,
   consumption_forecast, solar_generation, wind_offshore, wind_onshore,
   net_demand, wind_share]
                    │
                    ▼
  GradientBoosting (200 depth-limited trees)
                    │
                    ▼
               Spot Price (DKK/kWh)
```

A Carnot-style decomposition (separate wind/solar/consumption models feeding
into a price model) would require historical generation and consumption data
that isn't currently available.

## Training vs Prediction Segmentation

| Phase          | Weather source                            | Purpose                                                 |
| -------------- | ----------------------------------------- | ------------------------------------------------------- |
| **Training**   | `weather_history` (actual measurements)   | Learn real cause→effect: "when wind WAS X, price WAS Y" |
| **Prediction** | `weather.get_forecasts` (hourly forecast) | Predict future: "if wind WILL BE X, price should be Y"  |

Forecasts are ephemeral — pulled fresh each run. Actual measurements are
stored in `weather_history` (one snapshot every 15 minutes, kept 30 days)
and Nordpool prognoses in `nordpool_prognoses`; training matches both to
slots by UTC time. Both phases build their rows with the same function.

## No External API Dependencies

The integration reads everything from Home Assistant entities. No DMI API key,
no Nordpool API calls, no external HTTP requests. All data comes from HA's
built-in weather entity (Met.no), Stromligning sensors, Solcast, and inverter
power readings.
