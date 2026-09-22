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

| File               | Responsibility                                                              |
| ------------------ | --------------------------------------------------------------------------- |
| `const.py`         | `DOMAIN`, `CONF_*` keys, `REGIONS`, `PRICE_IN`, `PLATFORMS`, `UPDATE_SIGNAL` |
| `config_flow.py`   | Two-step config flow (basic settings → sensor configuration) + options flow |
| `sensor.py`        | Price sensors (current, today/tomorrow min/max/mean, ML prediction, confidence, learning metrics) |
| `binary_sensor.py` | `TomorrowAvailableSensor`, `MLModelTrainedSensor`                           |
| `sensor_reader.py` | `SensorReader` — all external entity reads (Stromligning, weather, Solcast, Met.no) |
| `__init__.py`      | Setup, update cycle (15-min / 6-hour / daily / midnight), ML wiring         |

### ML layer (`custom_components/open_spot_forecast/ml/`)

| File             | Responsibility                                                              |
| ---------------- | --------------------------------------------------------------------------- |
| `predictor.py`   | `SpotPricePredictor` — composes `FeatureMixin` + `ModelMixin` + `LearningMixin` |
| `features.py`    | `FeatureMixin` — feature extraction (wind, solar, time, Nordpool prognoses) |
| `models.py`      | `ModelMixin` — training + prediction                                        |
| `learning.py`    | `LearningMixin` — self-learning, bias correction, error metrics             |
| `numpy_models.py`| `NumpyGradientBoosting`, `NumpyRandomForest` — pure NumPy models            |
| `storage.py`     | `LearningStorage` — SQLite persistence                                       |

### API layer (`custom_components/open_spot_forecast/api/`)

| File              | Responsibility                                                              |
| ----------------- | --------------------------------------------------------------------------- |
| `nordpool_data.py`| `fetch_consumption_prognosis`, `fetch_production_prognosis` — Nordpool public APIs |

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
`read_stromligning_sensor`, `read_stromligning_tomorrow_sensor`, `read_weather_sensors`,
`read_solcast_sensor`, `read_met_weather`.

### ML predictor

`SpotPricePredictor` in `ml/predictor.py` is the single ML predictor. It composes
`FeatureMixin`, `ModelMixin`, and `LearningMixin`. Never re-implement feature extraction,
model training, or self-learning outside `ml/`.

### Learning storage

`LearningStorage` in `ml/storage.py` is the single SQLite persistence layer. Never open a
raw `sqlite3` connection or write to the learning DB outside this class.

### Floating-point comparisons

Production code uses an epsilon guard (`abs(x) > 1e-9` instead of `x != 0`). Tests use
`pytest.approx()`.

## Feature Vector (20 features)

The canonical feature vector is defined in `docs/ml_documentation.md` and built by
`FeatureMixin._combine_features()`:

| #  | Feature                | Source          |
| -- | ---------------------- | --------------- |
| 0  | `hour`                 | Time            |
| 1  | `day_of_week`          | Time            |
| 2  | `is_weekend`           | Time            |
| 3  | `hour_sin`             | Time            |
| 4  | `hour_cos`             | Time            |
| 5  | `wind_speed_mean`      | Weather entity  |
| 6  | `wind_power_estimate`  | Derived         |
| 7  | `wind_direction`       | Weather entity  |
| 8  | `cloud_coverage`       | Weather entity  |
| 9  | `humidity`             | Weather entity  |
| 10 | `solar_radiation_mean` | Solcast         |
| 11 | `solar_power_estimate` | Solcast         |
| 12 | `price_mean`           | Stromligning    |
| 13 | `temperature`          | Weather entity  |
| 14 | `consumption_forecast` | Nordpool API    |
| 15 | `solar_generation`     | Nordpool API    |
| 16 | `wind_offshore`        | Nordpool API    |
| 17 | `wind_onshore`         | Nordpool API    |
| 18 | `net_demand`           | Derived         |
| 19 | `wind_share`           | Derived         |

Adding or removing a feature is a model change — see the `osf-ml-change` skill and update
`docs/ml_documentation.md`.

## Slot Granularity — 96 Slots

The system models **96 slots per day** (15-minute intervals), not 24 hours. Per-slot bias
correction and error metrics are keyed `0-95`. Never assume hourly (0-23) granularity.

## Bias Correction Formula

Each 15-minute slot has a multiplicative correction factor learned via EMA:

```
bias_ratio = 1.0 - (mean_error / mean_actual_price)
correction[slot] = 0.9 * old_correction + 0.1 * bias_ratio
```

`correction > 1.0` → model underpredicts → multiply up. `correction < 1.0` → model
overpredicts → multiply down. Never invent a different correction scheme.

## Solar Scaling Factor

A learned EMA ratio between Solcast's estimate and actual inverter output:

```python
solar_scale = EMA(actual_power / solcast_estimate)
```

Updated every prediction run and applied to solar features before they enter the price
model.

## Confidence Score

- **Phase 1 — Heuristic** (< 5 samples): `base = 0.80 - wind_penalty - solar_penalty -
  weekend - days_ahead`, floor `0.30`.
- **Phase 2 — Learned** (≥ 5 samples): `confidence = max(0.10, 1.0 - (MAE / mean_actual))`
  minus forecast temperature/wind error penalties (max `-0.15` each).

## Storage Schema

SQLite database at `/config/.storage/open_spot_forecast_{region}_learning.db`.

| Table             | Key                    | Content                                        |
| ----------------- | ---------------------- | ---------------------------------------------- |
| `predictions`     | `id` (autoincrement)   | Pending predictions awaiting comparison        |
| `error_metrics`   | `hour` (0-95)          | Per-slot error arrays                          |
| `bias_correction` | `hour` (0-95)          | Per-slot correction factors                    |
| `price_history`   | `date` (YYYY-MM-DD)    | Daily price arrays (96 values/day)             |
| `weather_history` | `timestamp` (ISO)      | 15-min weather snapshots                       |
| `meta`            | `key`                  | Training state, schema version                 |

Migrations are versioned in `meta.schema_version` and run once at startup. The legacy JSON
format (`open_spot_forecast_DK1_learning.json`) is auto-migrated on first startup.

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
6. `__init__.py` — read the value and pass it into `weather_data`

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