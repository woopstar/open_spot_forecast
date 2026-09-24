---
name: osf-canonical-helpers
description: Activate at the start of any code change to ensure canonical helpers, constants, and patterns are used instead of re-implementing them inline.
---

# OSF Canonical Helpers & Patterns — Use These, Never Re-Invent

Activate this skill when writing any code that touches price prediction, feature
extraction, self-learning, storage, sensor reading, or configuration. These helpers
and patterns exist for a reason — **never re-implement them inline**.

## Canonical Modules

### 1. Domain & Config Constants — `const.py`

Location: `custom_components/open_spot_forecast/const.py`

```python
from custom_components.open_spot_forecast.const import (
    DOMAIN,
    CONF_REGION,
    CONF_CURRENCY,
    CONF_VAT,
    CONF_PRECISION,
    CONF_PRICE_TYPE,
    CONF_STROMLIGNING_SENSOR,
    CONF_ENABLE_ML_PREDICTION,
    REGIONS,
    PRICE_IN,
    PLATFORMS,
    UPDATE_SIGNAL,
)
```

All config keys, region definitions, price-unit conversions, and update signals
live here. Never hard-code a config key string, region name, or price-unit
factor elsewhere.

### 2. Sensor Reading — `SensorReader`

Location: `custom_components/open_spot_forecast/sensor_reader.py`

```python
from custom_components.open_spot_forecast.sensor_reader import SensorReader

reader = SensorReader(hass)
prices = reader.read_stromligning_sensor(entity_id)
weather = reader.read_weather_sensors(entity_id)
solar = reader.read_solcast_sensor(entity_id)
```

All reads of external HA entities (Stromligning, weather, Solcast, Met.no) go
through `SensorReader`. Never call `hass.states.get(...)` directly in platform
or ML code — add a reader method instead.

Validate a day of prices with `is_invalid_price_series()` from `price_series.py`
(all zero, or a missing/non-finite value, means invalid; some zero or negative
prices are fine). The Stromligning readers already drop invalid days, and
`store_daily_prices` / `predict` refuse them — never add an inline all-zero check.

### 3. ML Predictor — `SpotPricePredictor`

Location: `custom_components/open_spot_forecast/ml/predictor.py`

```python
from custom_components.open_spot_forecast.ml.predictor import SpotPricePredictor

predictor = SpotPricePredictor(hass, region, tz_name)
predictor.predict(weather_data, historical_prices, forecast_days=7)
```

`SpotPricePredictor` composes `FeatureMixin`, `ModelMixin`, and `LearningMixin`.
Never re-implement feature extraction, model training, or self-learning outside
`ml/`.

### 4. Learning Storage — `LearningStorage`

Location: `custom_components/open_spot_forecast/ml/storage.py`

```python
from custom_components.open_spot_forecast.ml.storage import LearningStorage

storage = LearningStorage(hass, region)
storage.insert_prediction(...)
storage.save_all()
```

All persistence goes through `LearningStorage` (SQLite). Never open a raw
`sqlite3` connection or write to the learning DB outside this class.

## Canonical Patterns

### Floating-Point Comparisons

```python
# Production code — epsilon guard (NEVER == or !=)
if abs(value) > 1e-9:        # instead of: if value != 0
if abs(a - b) < 1e-9:        # instead of: if a == b

# Test code — pytest.approx()
assert result == pytest.approx(expected, rel=1e-6)
```

### Slot Granularity — 96 Slots

The system models **96 slots per day** (15-minute intervals), not 24 hours.
Per-slot bias correction and error metrics are keyed `0-95`. Never assume
hourly (0-23) granularity.

Round to slot boundaries with `floor_to_slot()` / `ceil_to_slot()` and get the
first predicted slot from `first_prediction_slot()`, all in `time_slots.py`.
Count a local day's slots with `slots_in_local_day()` (92/96/100) and decide
whether tomorrow's prices are available with `tomorrow_prices_complete()` —
never compare a price-list length with a literal such as 23 or 96.
They work on the UTC timeline, so they stay correct across DST changes.

### Feature Vector — 20 Features

The canonical feature vector is defined in `docs/ml_documentation.md`. Model
input rows are built by `build_feature_vector()` (column order `FEATURE_NAMES`)
and time features by `slot_time_features()`, both in `ml/features.py`; the
production model comes from `create_price_model()` in `ml/models.py`. Never
inline a feature list. Adding/removing a feature is a model change — see the
`osf-ml-change` skill.

### Bias Correction Formula

```python
bias_ratio = 1.0 - (mean_error / mean_actual_price)
correction[slot] = 0.9 * old_correction + 0.1 * bias_ratio
```

This EMA is the canonical self-learning update. Never invent a different
correction scheme.

### Solar Scaling Factor

```python
solar_scale = EMA(actual_power / solcast_estimate)
```

Learned in `SpotPricePredictor.predict()`. Applied to solar features before they
enter the price model.

### Module Responsibilities (Know Where Code Lives)

| Layer      | Location                                    | Key files                                                                                  |
| ---------- | ------------------------------------------- | ------------------------------------------------------------------------------------------ |
| Component  | `custom_components/open_spot_forecast/`     | `__init__.py`, `config_flow.py`, `sensor.py`, `binary_sensor.py`, `const.py`               |
| Sensor I/O | `custom_components/open_spot_forecast/`     | `sensor_reader.py`                                                                         |
| ML         | `custom_components/open_spot_forecast/ml/`  | `predictor.py`, `features.py`, `models.py`, `learning.py`, `numpy_models.py`, `storage.py` |
| API        | `custom_components/open_spot_forecast/api/` | `nordpool_data.py`                                                                         |

### Utility Function Centralization

If a utility function is used in 2+ modules, it belongs in a shared module
(`const.py`, `sensor_reader.py`, or `ml/`). Before writing any utility:

1. Check existing modules for an implementation.
2. If found: import and reuse.
3. If not found AND used 2+ times: create it in the appropriate module with a
   public name and docstring.
4. If a one-off helper used in only one module: make it private
   (`_function_name()`), but refactor to a shared module if needs grow.

### File Size Limit

Hard limit: **30 KB AND 1000 lines** per file across the codebase. Check:
`wc -c custom_components/open_spot_forecast/**/*.py` and `wc -l custom_components/open_spot_forecast/**/*.py`.
