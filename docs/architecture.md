# Open Spot Forecast — Architecture

## System Overview

Open Spot Forecast is a Home Assistant integration that predicts electricity
spot prices using machine learning, weather forecasts, and confirmed market
data. The system runs a self-learning loop that compares predictions against
actual prices and continuously improves accuracy via per-slot, per-lead-time
bias correction.

## Data Sources

| Source                                      | Type                             | Used for                                                     |
| ------------------------------------------- | -------------------------------- | ------------------------------------------------------------ |
| `sensor.stromligning_current_price_ex_vat`  | Consumer price excl. VAT         | Displayed prices; minus spot: the tariffs (#107)             |
| `sensor.stromligning_spotprice_ex_vat`      | Raw spot price excl. VAT         | Price history, self-learning target, prediction              |
| `binary_sensor.stromligning_tomorrow_*`     | Tomorrow's prices when available | Known data window extension                                  |
| `weather.forecast_mellemlokken_23` (state)  | Current weather snapshot         | Wind, temperature, humidity, cloud                           |
| `weather.get_forecasts` (hourly)            | 48h weather forecast             | Per-slot wind/temp/cloud/humidity for prediction             |
| `sensor.solcast_pv_forecast_forecast_today` | Solar generation forecast        | Solar scaling factor (not a model input)                     |
| `sensor.power_inverter_input_total`         | Current solar production         | Solar scaling factor (not a model input)                     |
| `sensor.metroair_330_outdoor_temperature`   | Actual outdoor temperature       | Historical temperature for training                          |
| energy-charts.info / ENTSO-E (`dayahead`)   | Day-ahead auction prices (#27)   | All prices, instead of the Stromligning sensors              |
| ECB reference rates                         | EUR exchange rates               | Day-ahead prices in DKK/SEK/NOK                              |
| Open-Meteo (`WEATHER_POINTS` per region)    | 15-min zone weather, 8 days      | Zone features, training and prediction (#22)                 |
| ENTSO-E week-ahead load (API key)           | Daily min/max load, next week    | `load_forecast` curve, both phases (#30)                     |
| energy-charts + Open-Meteo, neighbours      | Neighbours' prices and weather   | Stage-1 models, cross-border option (#29)                    |
| Instrat (TGE gas day-ahead index)           | Daily gas price                  | `gas_price`, both phases, some regions (#28)                 |
| Nord Pool UMM API (no key)                  | Outage messages, every version   | `unavailable_*`, both phases, `UMM_REGIONS` (#123)           |
| ENTSO-E outage documents (API key)          | Outage documents, DE/NL/BE/FR    | `unavailable_*`, both phases, `ENTSOE_OUTAGE_REGIONS` (#138) |

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
│  │   │  • spot_prices      │  price history per UTC slot     │  │
│  │   │  • weather_history  │  15-min weather snapshots       │  │
│  │   │  • meta             │  schema version, training state │  │
│  │   └─────────────────────┘                                   │  │
│  └────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

## Price Output

Every price an entity exposes is computed in one place, `PriceOutput`
(`price_output.py`, #39), from the options read by `PriceSettings.from_entry()`
(options first, then the entry's initial data):

| Option           | Default | Effect                                                                    |
| ---------------- | ------- | ------------------------------------------------------------------------- |
| `vat`            | 0.25    | VAT rate as a fraction                                                    |
| `surcharge`      | 0       | Fixed amount per unit (currency per `price_type`) added before VAT        |
| `price_type`     | kWh     | Unit of every exposed price: kWh, MWh or Wh                               |
| `precision`      | 3       | Decimals every exposed price is rounded to                                |
| `hourly_average` | false   | Mean of each local hour's four 15-min prices, for hourly-billed contracts |

- **Every price** is read excl. VAT and exposed as
  `total = (price + surcharge) × (1 + VAT)`, in the configured unit
  (`apply_price_components()`): Stromligning's consumer price, the day-ahead
  source's spot price and the ML forecast alike. `api_data["prices_today"/
"prices_tomorrow"]` hold the source's prices excl. VAT; the sensors convert
  them. VAT comes from the `vat` option only, never from the source.
- **Tariffs** (#107): a forecast slot's price is the predicted spot price
  plus the slot's tariff, Stromligning's consumer price minus its spot price
  (both excl. VAT, see [Tariffs](#tariffs)). So the forecast is on the same
  footing as the displayed consumer price. Stromligning's consumer price
  already holds the supplier's surcharge: keep `surcharge` at 0 with it.
- **`hourly_average`**: the current price is the current local hour's mean;
  today's/tomorrow's price lists hold one value per local hour (23/24/25 on
  DST days) and min/max/mean are taken over those; the forecast attribute has
  one entry per hour (`start`/`end` of the hour, mean price and confidence),
  and the `prediction_hours` window counts hours. Hours are grouped on the UTC
  timeline, so the repeated hour of the fall-back day is two entries.

Statistics are taken over the raw series, then converted: the conversion is
affine and increasing, so this equals converting first. The options flow has
no update listener: option changes apply after the integration reloads.

### Tariffs

`TariffSchedule` (`tariffs.py`, #107) is built with the spot prices
(`ForecastUpdater.read_spot_prices()`, `api_data["tariffs"]`) from the
consumer and spot prices already read:

- A slot's tariff is `consumer − spot`, for every slot of today and tomorrow
  where both are known (keyed by UTC slot start). That is every non-spot
  component: supplier surcharge, electricity tax, Energinet's net and system
  tariffs and the grid company's time-of-use tariff. Stromligning's separate
  tariff sensors report only the current value, and its distribution sensor
  has no tomorrow, so they are not read.
- A slot past the published prices takes the latest known day's tariff at
  the same local time of day (tariffs follow a fixed daily schedule, and the
  latest day already has e.g. the winter tariffs from 1 October); a time of
  day never seen (a skipped DST hour) takes the latest earlier one.
- Without consumer prices (no sensor, or the `dayahead` source) the schedule
  is empty: the tariff is 0 and `includes_tariffs` is false.

`PriceOutput.forecast()` adds the tariff per slot, before the hourly mean,
for the forecast sensor (state, `predictions`, `forecast_min/max/mean`) and
the `get_forecast` action; `evaluation()` adds it to both the predicted and
the actual price, so the error stays the spot price's. The model, the stored
predictions, the error metrics and the bias correction never see a tariff.

## Forecast Attributes

The `Price Forecast (ML)` sensor's `predictions` attribute holds the next
`prediction_hours` of the forecast, in one of two layouts (option
`attribute_format`, `forecast_attributes.py`, #38):

- **`detailed`** (default): a list of `{start, end, price, unit, confidence}`,
  about 125 bytes per 15-min slot; the window is capped at 72 hours.
- **`compact`**: parallel arrays, like EpexPredictor's short format:

  ```json
  {
    "interval_minutes": 15,
    "unit": "DKK/kWh",
    "s": [1790200800, 1790201700],
    "t": [1.234, 1.187],
    "c": [82, 80]
  }
  ```

  `s` is each interval's start in unix seconds, `t` its price, `c` its
  confidence in percent (like the Prediction Confidence sensor). About 20
  bytes per slot, so up to 168 hours fit.

Home Assistant's recorder does not store any of a state's attributes when
their JSON exceeds 16 KB (`RECORDER_MAX_ATTRIBUTES_BYTES`), and logs a
warning on every state write. The `predictions` attribute is therefore
excluded from the recorder (`_unrecorded_attributes`, #103): it is in the
live state for dashboards and automations, the sensor's state and its
other attributes are recorded, and the window is not shortened (the
detailed layout passes 16 KB beyond about 32 hours of 15-min slots). In
the compact layout `fit_compact()` still keeps the attributes within
15 KB (1 KB is left for the attributes Home Assistant adds) by dropping
whole hours from the end, e.g. with MWh prices at 6 decimals: the live
state has a size limit too. `hourly_metrics` of the Learning Metrics
sensor (96 slots of metrics) is excluded the same way. For the whole
forecast without a size limit use the `get_forecast` action.

An ApexCharts series over the compact layout:

```yaml
data_generator: |
  const p = entity.attributes.predictions;
  return p.s.map((s, i) => [s * 1000, p.t[i]]);
```

### Known prices and `known_until` (#40)

`known_until` (attribute of the forecast sensor, field of the action) is the
end of the last confirmed spot price slot, local ISO with offset
(`known_until()` in `spot_prices.py`); the model's predictions start there.

With the option `include_known_prices` (default off; the action's
`include_known` field) the `predictions` attribute is one continuous series
from the current slot: the confirmed spot prices up to `known_until`, then the
predictions (`with_known_prices()`). Every entry has a `source`, `actual`
(confidence 1.0) or `predicted`; an hour is `actual` only if all its slots
are. The compact layout adds `known_count`, the number of leading confirmed
entries. Predictions that start before the end of the confirmed prices (an
older forecast) are dropped, so no slot appears twice and there is no gap at
the boundary; a slot missing in the source stays missing. Confirmed prices go
through the same `PriceOutput` as the predictions, so both are
`(spot + tariff + surcharge) × (1 + VAT)`: a confirmed slot's tariff is its
own, so it equals the displayed consumer price (#107). The sensor's state stays the model's
prediction. The `today`/`tomorrow` lists are placed on their local day by the
spot data's `day`, so the series is right in the first second after midnight
too.

## Actions

`open_spot_forecast.get_forecast` (`services.py`, #37) is registered once in
`async_setup` (not per entry) with `SupportsResponse.ONLY`, so it validates
even while no entry is loaded. It finds the entry with Home Assistant's
`service.async_get_config_entry()` (`config_entry_id` may be left out when
one entry exists; unknown or unloaded entries raise a translated
`ServiceValidationError`) and returns `forecast_response()`:

| Field              | Content                                                                 |
| ------------------ | ----------------------------------------------------------------------- |
| `known_until`      | End of the last confirmed spot price slot (local ISO), or null          |
| `unit`             | e.g. `DKK/kWh`                                                          |
| `interval_minutes` | 15, or 60 with `hourly`                                                 |
| `forecast`         | `start`, `end`, `price`, `confidence` per interval, from `start` onward |

Fields: `start` (default now: the interval containing it), `hours` (default:
the whole forecast), `hourly` (default: the entry's `hourly_average`) and
`include_known` (default: the entry's `include_known_prices`: confirmed prices
from `start`, then the predictions, see above). With `evaluation: true` the
response also has `evaluation`: every kept slot's day-ahead prediction next to
its actual price (`start`, `end`, `predicted`, `actual`, `lead_hours`; see
[self-learning](self_learning.md#predicted-vs-actual-36)).
Prices go through the entry's `PriceOutput`, so the action and the forecast
sensor always agree; the action has no 16 KB attribute limit. Without the ML
model it raises `ml_prediction_disabled`.

`open_spot_forecast.reset_learning` (#132) is registered next to it with the
same `config_entry_id` lookup. It awaits `SpotPricePredictor.reset_learning()`
(clears the in-memory learned state and drops and recreates the learning
database), raises a translated `HomeAssistantError` when the database could
not be cleared, and sends `UPDATE_SIGNAL` so the learning and accuracy
entities refresh at once.

## Data Flow

```
Every 15 min ──→ Read Stromligning prices (consumer, and raw spot for the model)
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

Every path is a method of `ForecastUpdater` (`updater.py`): `new_quarter`,
`update_forecasts`, `check_tomorrow_prices` (via `TomorrowPriceChecker`) and
`new_day`; `async_setup_entry` builds the updater, runs its initial fetch and
registers these callbacks. Setup, the tomorrow-price refresh and the 6-hourly
update run the same forecast pipeline, `ForecastUpdater.run_forecast()`:
weather sensors and forecast → Nordpool prognoses (stored for training) →
known-data end → predict → save.

## Model: Single Price Predictor

The system uses **one model** — a Gradient Boosting regressor that takes 23
features and directly predicts the spot price. Wind, solar, and temperature
are input features, not separate sub-models. Training and prediction rows
come from the same `build_feature_row()`; an unknown input is NaN (see
[ML Documentation](ml_documentation.md#feature-vector-26-features)).

```
Features (23):
  [day_of_week, is_weekend, holiday, slot_sin, slot_cos, morning_peak,
   sun_elevation, sun_azimuth, since_sunrise, since_sunset,
   consumption_forecast, solar_generation, wind_offshore, wind_onshore,
   net_demand, wind_share, load_forecast,
   zone_wind, zone_wind_power, zone_temperature, zone_irradiance,
   zone_pressure, zone_humidity]
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

Both phases use **forecasts** (#23), so the model learns from the same kind
of weather input, with a similar error, as it predicts from:

| Phase          | Weather source                                                     | Purpose                                                        |
| -------------- | ------------------------------------------------------------------ | -------------------------------------------------------------- |
| **Training**   | `openmeteo_weather`: Open-Meteo's archived forecasts for past days | Learn "when the forecast said X, the price was Y"              |
| **Prediction** | `openmeteo_weather`: the live Open-Meteo forecast, 8 days ahead    | Predict: "the forecast says X, so the price should be about Y" |

The background backfill fills the training window's zone weather from the
archive at setup and after midnight; each forecast run refreshes the live
forecast from yesterday on. Nordpool prognoses are stored in
`nordpool_prognoses`; training matches both tables to slots by UTC time.
Both phases build their rows with the same function. Prognoses only exist
for today and tomorrow, so almost no prediction row has them; training adds
a copy of every row with them masked (#91), so the model also learns the
rows without them. The local weather
entity is no longer a model input: its snapshots (`weather_history`, one
every 15 minutes) score its forecast for the confidence.

## Attribution

Entities credit the external sources of their values in Home Assistant's
`attribution` (`attribution.py`, #41): the price entities
(`PriceAttributionMixin`) credit energy-charts.info with the licence it
reports for the zone, and ENTSO-E when its fallback is configured; the
model's entities (`ModelAttributionMixin`: forecast, confidence, learning
metrics, accuracy, model trained) add Open-Meteo (CC BY 4.0) when the zone
weather is used, Nord Pool, ENTSO-E when its load forecast is used, Instrat
in the regions with the gas price and the outage source in the regions that
use one: Nord Pool's UMMs (#123) or the ENTSO-E Transparency Platform's
outage documents (#138). The sources, their licences and the credit
to EpexPredictor are listed in the
[README](../README.md#data-sources-and-attribution).

## External APIs

Weather comes from Home Assistant entities: HA's built-in weather entity
(Met.no), Solcast and inverter power readings. Prices come from the
Stromligning sensors or, with the `dayahead` price source, from
energy-charts.info (ENTSO-E as fallback with an API key) and the ECB's
exchange rates (see
[price sources](stromligning_integration.md#price-sources)). The zone
weather comes from Open-Meteo (no key), one request for all of a region's
points, refreshed from yesterday to 8 days ahead on each forecast run. The other
external requests are Nordpool's public consumption and production
prognoses: today and tomorrow on each forecast run, older delivery days only
while they are incomplete and at most 6 days old (Nordpool answers 401
beyond a week without a login; see
[persistence](persistence.md#time-series-sources)). Every API client uses
the shared `api/http.py` (`async_get`: retries with backoff, `Retry-After`,
never logs a URL or its parameters). Nordpool gets a browser-like
`USER_AGENT` and Instrat `INSTRAT_USER_AGENT`: both sit behind Cloudflare,
which answers 403 to Home Assistant's default User-Agent. Nord Pool's UMM
API (#123, no key) accepts it; the source fetches the training window's
messages once (every version, paged) and then only new publications. The
ENTSO-E outage documents (#138, API key; DE, NL, BE and FR) follow the same
cycle: the zone's plant documents and every border's grid documents in both
directions, a ZIP of XML per request (at most 200 documents, paged with
`offset`), unzipped and parsed in the executor.
