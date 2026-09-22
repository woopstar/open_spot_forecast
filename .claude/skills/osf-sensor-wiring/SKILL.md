---
name: osf-sensor-wiring
description: Activate when adding, modifying, or using any external sensor entity in OSF (Stromligning, Nordpool, weather, Solcast, Met.no). Follow the full wiring protocol from const.py through sensor_reader.py.
---

# OSF Sensor Wiring — External Entity Protocol

Activate this skill when you need to:

- Add a new external sensor entity (price, weather, solar) to OSF
- Use an existing external entity value
- Reference a new data source for the ML feature vector or price history

## Golden Rule

**Every external value consumed by OSF MUST be read through `SensorReader` in
`sensor_reader.py`.** Never call `hass.states.get(...)` directly in platform or
ML code, and never hard-code a numeric value that an entity reports.

## Step 1: Check `docs/USING_EXISTING_SENSORS.md` First

This is the canonical, verified list of entities OSF reads (Stromligning,
Met.no weather, Solcast, inverter power, temperature). Only fall back to
searching an upstream integration when an entity is not yet listed there.

## Step 2: If the Entity Is Already Wired

Re-use it. The entity should already flow through `const.py` → `config_flow.py`
→ `sensor_reader.py` → `__init__.py`.

## Step 3: If the Entity Is New — Wire It Through the Full Stack

Add it through the **full stack in this exact order**:

1. **`const.py`** — Add a `CONF_*` key (e.g. `CONF_WIND_SPEED_SENSOR`) and, if
   it has a sensible default, a default entity-id string.
2. **`config_flow.py`** — Add the field to the `sensors` step schema (and the
   options flow `init` step schema) using an entity `selector`.
3. **`translations/en.json`** — Add the `data` label (and `data_description`
   where helpful) for the new field in **both** `config.step.sensors` and
   `options.step.init`.
4. **`translations/da.json`** — Add the Danish translation, keeping it in sync
   with `en.json` (see `osf-translation-sync`).
5. **`sensor_reader.py`** — Add a `read_*` method on `SensorReader` that reads
   the entity state/attributes and returns a normalized dict.
6. **`__init__.py`** — Read the value during the update cycle and pass it into
   the `weather_data` dict consumed by `SpotPricePredictor.predict()`.

## Step 4: If the Entity Feeds the ML Feature Vector

If the new value becomes a model feature, you must also:

- Update `docs/ML_DOCUMENTATION.md` (feature table)
- Update `ml/features.py` (`_combine_features` / `_extract_*_features`)
- Add or update tests
- See the `osf-ml-change` skill for the full protocol

## Key Entity Mappings

| Source                          | Entity (example)                                  | Used for                          |
| ------------------------------- | ------------------------------------------------- | --------------------------------- |
| Stromligning (current price)    | `sensor.stromligning_current_price_vat`           | Confirmed consumer prices (96/day) |
| Stromligning (tomorrow)         | `binary_sensor.stromligning_tomorrow_spotprice_vat` | Tomorrow's price availability    |
| Met.no weather (state)          | `weather.forecast_*`                              | Current wind/temp/humidity/cloud  |
| Met.no weather (forecast)       | `weather.get_forecasts` service                   | 48h hourly forecast               |
| Solcast solar forecast          | `sensor.solcast_pv_forecast_forecast_today`       | Solar generation estimate         |
| Inverter solar production       | `sensor.power_inverter_input_total`               | Actual solar (training/scale)     |
| Outdoor temperature             | `sensor.metroair_330_outdoor_temperature`         | Actual temperature (training)     |

## Never Do This

- Never use a fixed numeric constant for a value an entity reports
- Never guess an entity ID — check `docs/USING_EXISTING_SENSORS.md` first
- Never skip a step in the wiring stack
- Never forget to update `translations/en.json` (and `da.json`) for both
  `config` and `options` steps
- Never read an entity directly in `ml/` code — go through `SensorReader`