# Quick Start Guide

## Installation (5 minutes)

1. Copy `custom_components/open_spot_forecast` to your Home Assistant `custom_components` directory
2. Restart Home Assistant
3. Go to **Settings** → **Devices & Services** → **Add Integration**
4. Search for "Open Spot Forecast"
5. Select your region (e.g., DK1, DK2)
6. Click **Submit**

## Configuration

### Basic Setup (No API Keys Required)

Works immediately with Nordpool public API:

- Current spot prices
- Today's min/max/mean
- Tomorrow's prices (after 13:00 CET)
- Basic ML predictions

### Recommended Setup

**Step 1: Configure Price Source**

Go to integration options and add:

```yaml
Consumer Price Sensor: sensor.stromligning_current_price_ex_vat
Consumer Price Tomorrow Sensor: binary_sensor.stromligning_tomorrow_available_ex_vat
Spot Price Sensor: sensor.stromligning_spotprice_ex_vat
Spot Price Tomorrow Sensor: binary_sensor.stromligning_tomorrow_spotprice_ex_vat
```

These are the defaults. All four are **disabled by default in Stromligning**:
enable them under Settings → Devices & services → Stromligning → Entities.
Always use the `_ex_vat` entities: Open Spot Forecast adds VAT itself.

**Why Stromligning?**

- Real consumer prices (includes tariffs; VAT added by OSF)
- The forecast includes each slot's tariffs too (consumer minus spot price)
- What you actually pay on your electricity bill
- Better for automation decisions

**Alternative**: Use Nordpool sensor for spot prices only

**Step 2: Configure Weather Sensors** (Optional but Recommended)

```yaml
Wind Speed Sensor: sensor.home_wind_speed
Wind Direction Sensor: sensor.home_wind_bearing
Solar Power Sensor: sensor.solcast_pv_forecast_power_now
Temperature Sensor: sensor.home_temperature
```

**Why Weather Data?**

- Improves ML prediction accuracy by 20-30%
- Wind and solar are major price drivers
- Helps predict price spikes

**Step 3: Enable ML Predictions**

```yaml
Enable ML Predictions: true
```

## Available Sensors

After installation, you'll have the entities below. Their IDs follow the
device name, `Open Spot Forecast <region>`: the list shows DK1, so for DK2
read `open_spot_forecast_dk2_...`. Check them under **Settings** → **Devices
& services** → **Open Spot Forecast**.

### Price Sensors

- `sensor.open_spot_forecast_dk1_current_spot_price` - Current price
- `sensor.open_spot_forecast_dk1_today_min_price` - Today's lowest
- `sensor.open_spot_forecast_dk1_today_max_price` - Today's highest
- `sensor.open_spot_forecast_dk1_today_mean_price` - Today's average
- `sensor.open_spot_forecast_dk1_tomorrow_min_price` - Tomorrow's lowest
- `sensor.open_spot_forecast_dk1_tomorrow_max_price` - Tomorrow's highest
- `sensor.open_spot_forecast_dk1_tomorrow_mean_price` - Tomorrow's average

### Forecast Sensors

- `sensor.open_spot_forecast_dk1_price_forecast_ml` - **ML-based 7-day forecast**
- `sensor.open_spot_forecast_dk1_prediction_confidence` - Prediction confidence (%)

### Learning Sensors

- `sensor.open_spot_forecast_dk1_learning_metrics` - Self-learning status

### Status Sensors

- `binary_sensor.open_spot_forecast_dk1_tomorrow_prices_available` - Tomorrow's prices ready
- `binary_sensor.open_spot_forecast_dk1_ml_model_trained` - ML model status

### Predbat Sensors (optional)

With the option **Predbat rate entities** (see [Predbat](#predbat)):

- `sensor.open_spot_forecast_dk1_predbat_import_today` - Consumer price today
- `sensor.open_spot_forecast_dk1_predbat_import_tomorrow` - Consumer price tomorrow and later
- `sensor.open_spot_forecast_dk1_predbat_export_today` - Raw spot price today
- `sensor.open_spot_forecast_dk1_predbat_export_tomorrow` - Raw spot price tomorrow and later

## Common Use Cases

### 1. Display Current Price

```yaml
type: entities
entities:
  - entity: sensor.open_spot_forecast_dk1_current_spot_price
    name: Current Electricity Price
    secondary_info: last-updated
```

### 2. Forecast Chart

See [Dashboard](#dashboard) for chart cards of the forecast, the
predicted-vs-actual evaluation and a cheapest-window script.

### 3. Charge EV During Cheap Hours

```yaml
automation:
  - alias: "Charge EV when price is low"
    trigger:
      - platform: time_pattern
        minutes: "/15"
    condition:
      - condition: numeric_state
        entity_id: sensor.open_spot_forecast_dk1_current_spot_price
        below: sensor.open_spot_forecast_dk1_today_mean_price
      - condition: numeric_state
        entity_id: sensor.open_spot_forecast_dk1_prediction_confidence
        above: 70
    action:
      - service: switch.turn_on
        target:
          entity_id: switch.ev_charger
```

### 4. Run Dishwasher at Cheapest Time

```yaml
automation:
  - alias: "Start dishwasher at cheapest hour"
    trigger:
      - platform: time_pattern
        minutes: "/15"
    condition:
      # The cheapest interval of the forecast (detailed attribute format)
      - condition: template
        value_template: >
          {% set forecast = state_attr('sensor.open_spot_forecast_dk1_price_forecast_ml', 'predictions') %}
          {% if forecast %}
            {% set cheapest = forecast | sort(attribute='price') | first %}
            {{ as_datetime(cheapest.start) <= now() < as_datetime(cheapest.end) }}
          {% else %}
            false
          {% endif %}
    action:
      - service: switch.turn_on
        target:
          entity_id: switch.dishwasher
```

### 5. Alert When Tomorrow's Prices Available

```yaml
automation:
  - alias: "Notify when tomorrow's prices are ready"
    trigger:
      - platform: state
        entity_id: binary_sensor.open_spot_forecast_dk1_tomorrow_prices_available
        to: "on"
    action:
      - service: notify.mobile_app
        data:
          message: "Tomorrow's electricity prices are now available!"
```

## Dashboard

The examples below were tested on Home Assistant 2026.9 with
[ApexCharts Card](https://github.com/RomRider/apexcharts-card) 2.2.3 and
[Plotly Graph Card](https://github.com/dbuezas/lovelace-plotly-graph-card)
3.3.5 (both from HACS). Entity IDs contain the region: replace `dk1` with
yours (check **Developer Tools** → **States**). Prices are the spot price with
each slot's tariffs (Stromligning source), surcharge and VAT.

The forecast sensor's `predictions` attribute has two layouts, set by the
option **Forecast attribute format**; use the card that matches it (a card for
the other layout shows no forecast line):

- **Detailed** (default): a list of `start`, `end`, `price`, `unit`,
  `confidence`, up to 72 hours.
- **Compact**: parallel arrays `s` (start, unix seconds), `t` (price) and `c`
  (confidence in percent), up to 168 hours.

### Forecast (detailed format)

Today's and tomorrow's confirmed prices, then the forecast as a dashed line:

```yaml
type: custom:apexcharts-card
graph_span: 3d
span:
  start: day
now:
  show: true
  label: Now
header:
  show: true
  title: Electricity price
series:
  - entity: sensor.open_spot_forecast_dk1_current_spot_price
    name: Confirmed
    curve: stepline
    extend_to: false
    data_generator: |
      const day = (prices, offset) => {
        prices = prices || [];
        const start = new Date();
        start.setHours(0, 0, 0, 0);
        start.setDate(start.getDate() + offset);
        const step = (prices.length > 25 ? 15 : 60) * 60000;
        return prices.map((p, i) => [start.getTime() + i * step, p]);
      };
      return day(entity.attributes.today_prices, 0).concat(
        day(entity.attributes.tomorrow_prices, 1),
      );
  - entity: sensor.open_spot_forecast_dk1_price_forecast_ml
    name: Forecast
    curve: stepline
    extend_to: false
    stroke_dash: 4
    data_generator: |
      return entity.attributes.predictions.map((p) => [
        new Date(p.start).getTime(),
        p.price,
      ]);
```

`today_prices` / `tomorrow_prices` hold one price per 15 minutes from local
midnight (one per hour with the **Hourly average prices** option); the
generator handles both. It steps in real time from midnight, so the 92- or
100-slot days when clocks change line up too.

### Forecast (compact format)

Price and confidence on two axes. With **Include confirmed prices in the
forecast attribute** on, the series starts with the confirmed prices
(confidence 100 %), so one series shows both:

```yaml
type: custom:apexcharts-card
graph_span: 48h
span:
  start: hour
now:
  show: true
  label: Now
header:
  show: true
  title: Price forecast
yaxis:
  - id: price
  - id: confidence
    opposite: true
    min: 0
    max: 100
series:
  - entity: sensor.open_spot_forecast_dk1_price_forecast_ml
    name: Price
    yaxis_id: price
    curve: stepline
    extend_to: false
    data_generator: |
      const f = entity.attributes.predictions;
      return f.s.map((s, i) => [s * 1000, f.t[i]]);
  - entity: sensor.open_spot_forecast_dk1_price_forecast_ml
    name: Confidence
    unit: "%"
    yaxis_id: confidence
    curve: stepline
    extend_to: false
    opacity: 0.4
    data_generator: |
      const f = entity.attributes.predictions;
      return f.s.map((s, i) => [s * 1000, f.c[i]]);
```

The same with Plotly Graph Card, from 4 hours ago to 26 hours ahead with a
"now" marker:

```yaml
type: custom:plotly-graph
hours_to_show: 30
time_offset: 26h
refresh_interval: 10
layout:
  yaxis9:
    fixedrange: true
    visible: false
    minallowed: 0
    maxallowed: 1
entities:
  - entity: sensor.open_spot_forecast_dk1_price_forecast_ml
    name: Forecast
    line:
      shape: hv
    filters:
      - fn: |-
          ({ meta }) => ({
            xs: meta.predictions.s.map((s) => new Date(s * 1000)),
            ys: meta.predictions.t,
          })
  - entity: ""
    name: Now
    yaxis: y9
    showlegend: false
    line:
      width: 1
      dash: dot
      color: orange
    x: $ex [Date.now(), Date.now()]
    "y": [0, 1]
```

### Predicted vs actual

The diagnostic **Forecast evaluation** sensor keeps, for the last 48 hours,
the prediction made about a day ahead next to the actual price (`s`, `t`
predicted, `a` actual); its state is the mean absolute error. It fills as
slots are scored, so it is empty for the first day after setup:

```yaml
type: custom:apexcharts-card
graph_span: 48h
span:
  end: hour
header:
  show: true
  title: Forecast vs actual (a day ahead)
series:
  - entity: sensor.open_spot_forecast_dk1_forecast_evaluation
    name: Actual
    curve: stepline
    extend_to: false
    data_generator: |
      const e = entity.attributes;
      return e.s.map((s, i) => [s * 1000, e.a[i]]);
  - entity: sensor.open_spot_forecast_dk1_forecast_evaluation
    name: Predicted
    curve: stepline
    extend_to: false
    stroke_dash: 4
    data_generator: |
      const e = entity.attributes;
      return e.s.map((s, i) => [s * 1000, e.t[i]]);
```

With Plotly Graph Card, use `meta.s`, `meta.t` and `meta.a` in a filter as in
the forecast example above.

### Cheapest window (`get_forecast` action)

A script that returns the start of the cheapest run of `hours` in the next
24 hours, from confirmed prices and the forecast. Add it to `scripts.yaml`:

```yaml
cheapest_window:
  alias: Cheapest window
  fields:
    hours:
      description: Length of the window in hours
      default: 3
      selector:
        number:
          min: 1
          max: 24
  sequence:
    - action: open_spot_forecast.get_forecast
      data:
        hours: 24
        include_known: true
      response_variable: forecast
    - variables:
        window: >
          {% set slots = forecast.forecast %}
          {% set n = ((hours | default(3)) * 60 / forecast.interval_minutes) | int %}
          {% set ns = namespace(total=none, start=none) %}
          {% for i in range(slots | length - n + 1) %}
            {% set total = slots[i:i + n] | map(attribute='price') | sum %}
            {% if ns.total is none or total < ns.total %}
              {% set ns.total = total %}
              {% set ns.start = slots[i].start %}
            {% endif %}
          {% endfor %}
          {{ {'start': ns.start, 'mean_price': (ns.total / n) | round(3) if n else none} }}
    - stop: Cheapest window found
      response_variable: window
```

Call it from an automation and use the result:

```yaml
- action: script.cheapest_window
  data:
    hours: 3
  response_variable: cheapest
- action: notify.notify
  data:
    message: >
      Cheapest 3 hours start {{ as_timestamp(cheapest.start) | timestamp_custom('%H:%M') }}
      at {{ cheapest.mean_price }} {{ state_attr('sensor.open_spot_forecast_dk1_price_forecast_ml', 'unit') }}
```

In a single-region setup the action needs no `config_entry_id`; with several
regions pass the entry of the one to read.

## evcc

[evcc](https://evcc.io/) has no Open Spot Forecast template, but its
[user-defined tariff](https://docs.evcc.io/en/user-defined-devices#tariff)
can read the forecast from Home Assistant's REST API: evcc sends a POST to the
`get_forecast` action with `?return_response` and converts the response to its
format with `jq`. Tested with evcc 0.316.1 (`evcc tariff` lists every slot at
the same price as the action).

1. In Home Assistant create a long-lived access token (**Profile** →
   **Security** → **Long-lived access tokens**). Treat it like a password: it
   grants full access to Home Assistant.
2. Add the grid tariff to `evcc.yaml`, with your Home Assistant address and
   the token:

```yaml
tariffs:
  currency: DKK
  grid:
    type: custom
    forecast:
      source: http
      uri: http://homeassistant.local:8123/api/services/open_spot_forecast/get_forecast?return_response
      method: POST
      headers:
        - Authorization: Bearer <long-lived access token>
        - Content-Type: application/json
      body: '{"include_known": true}'
      jq: >-
        [.service_response.forecast[] | {
          start: (.start | sub("(?<h>[+-][0-9]{2}):(?<m>[0-9]{2})$"; "\(.h)\(.m)") | strptime("%Y-%m-%dT%H:%M:%S%z") | mktime | todate),
          end: (.end | sub("(?<h>[+-][0-9]{2}):(?<m>[0-9]{2})$"; "\(.h)\(.m)") | strptime("%Y-%m-%dT%H:%M:%S%z") | mktime | todate),
          value: .price
        }] | tostring
```

3. Check it with `evcc tariff`: it lists the slots from now to about 7 days
   ahead.

Notes:

- `include_known: true` gives evcc the confirmed prices first, then the
  forecast. evcc wants UTC times (`...Z`); the `jq` converts the action's
  local times.
- `currency` must match the integration's currency, and the price unit must
  be kWh (evcc expects currency per kWh).
- With the Stromligning source the prices include the tariffs, VAT and the
  integration's surcharge: leave evcc's `charges`, `chargesZones` and `tax`
  at 0 (evcc applies `tax` to the whole price, which already has VAT). With
  the day-ahead source add grid tariffs (incl. VAT) with `charges` or
  `chargesZones`.
- With several regions add `"config_entry_id": "<entry id>"` to `body`.
- evcc reads the forecast hourly (`interval`, default `1h`).

## Predbat

[Predbat](https://springfall2008.github.io/batpred/) plans a home battery
from import and export rates. In Denmark it reads them from the
Stromligning integration's entities, which hold today and tomorrow only, so
after 13:00 it plans against at most about 35 hours of rates and repeats the
last known day beyond that. Open Spot Forecast can expose its 7-day forecast
in the same shape, so Predbat plans against the forecast instead.

1. In the integration's options enable **Predbat rate entities**. With the
   day-ahead price source there are no Stromligning tariffs: enter your grid
   tariff per kWh excl. VAT as **Predbat fixed grid tariff** (leave it 0 with
   the Stromligning source). Reload the integration.
2. Four sensors appear on the integration's device. Their entity IDs follow
   the device name (`Open Spot Forecast <region>`); check them under
   **Settings** → **Devices & services** → **Open Spot Forecast**.
3. Point Predbat's `apps.yaml` at them instead of the Stromligning entities
   (DK1 shown):

```yaml
metric_stromligning_import_today: sensor.open_spot_forecast_dk1_predbat_import_today
metric_stromligning_import_tomorrow: sensor.open_spot_forecast_dk1_predbat_import_tomorrow
metric_stromligning_export_today: sensor.open_spot_forecast_dk1_predbat_export_today
metric_stromligning_export_tomorrow: sensor.open_spot_forecast_dk1_predbat_export_tomorrow
```

Only `apps.yaml` changes: the integration keeps reading the Stromligning
entities for its own prices and tariffs.

What Predbat gets:

- **Attributes**: the today entities carry `prices_today`, the tomorrow
  entities `prices_tomorrow`, each a list of `{"start", "end", "price"}`
  per 15-minute interval (per hour with **Hourly average prices**) in local
  time, which is what Predbat reads from the Stromligning entities. The
  state is the current interval's price (today) or tomorrow's first price.
- **Import** is what you pay, per kWh: the confirmed prices, then the
  forecast, as `(spot + tariff + surcharge) × (1 + VAT)`. The tariff is
  each slot's Stromligning consumer price minus spot price (grid tariff,
  tax and fees, time of day included); days past the published prices
  repeat the latest day's tariff at the same time of day. With the
  day-ahead source the fixed tariff option is used instead.
- **Export** is the raw spot price excl. VAT, surcharge and tariffs: today's
  confirmed `spotprice_ex_vat` prices, then the model's raw forecast.
- **Unit**: `kr/kWh` for DKK, SEK and NOK. Predbat multiplies a price by
  100 (to øre) only when the unit contains `kr/`, which `DKK/kWh` would
  not. Other currencies are exposed in cents (`ct/kWh`), which Predbat uses
  as they are.
- **Horizon**: `prices_today` runs from local midnight; `prices_tomorrow`
  holds tomorrow and the following days as far as Home Assistant's 16 KB
  attribute limit allows: about 45 hours at 15-minute resolution (tomorrow
  and most of the day after), the whole 7-day forecast with **Hourly
  average prices**. Predbat keeps the rates for `forecast_days + 1` days
  from UTC midnight (`forecast_hours: 48` → today, tomorrow and the day
  after); raise `forecast_hours` in `apps.yaml` to plan further ahead.

Written against Predbat v9.3.3's Stromligning reader
(`apps/predbat/stromligning.py`, read 2026-10-02): the attribute names, the
entry shape, the `kr/` scaling and the horizon are taken from it. It has not
yet been confirmed on a running Predbat install; if you run one, check that
the plan shows the forecast's prices (in øre) through the day after tomorrow
and report back in the issue tracker.

## Understanding the Sensors

### Current Price Sensor

**Attributes**:

```yaml
region: DK1
currency: DKK
vat: 0.25
last_update: "2026-07-06T13:15:00"
today_prices: [234.5, 235.1, 236.8, ...] # 24 values
tomorrow_prices: [240.2, 241.5, ...] # 24 values (when available)
```

### ML Prediction Sensor

**Attributes**:

```yaml
predictions:
  - timestamp: "2026-07-06T14:00:00"
    predicted_price: 245.3
    confidence: 0.92
    model: "GradientBoosting"
  - timestamp: "2026-07-06T14:15:00"
    predicted_price: 248.7
    confidence: 0.91
    model: "GradientBoosting"
  # ... predictions for the configured window (default 48 hours = 192 slots;
  # configurable in 12-hour steps up to 72 hours to stay under Home
  # Assistant's 16 KB attribute limit)

prediction_min: 150.5
prediction_max: 450.2
prediction_mean: 280.3
mean_confidence: 0.85
total_predictions: 672
is_ml_model: true
training_samples: 720
```

### Confidence Score

- **0.9-1.0**: Very high confidence (short-term, good weather data)
- **0.7-0.9**: High confidence (1-2 days ahead)
- **0.5-0.7**: Medium confidence (3-5 days ahead)
- **0.3-0.5**: Low confidence (6-7 days ahead or missing data)

## Troubleshooting

### Tomorrow's Prices Not Available

**Problem**: `binary_sensor.open_spot_forecast_dk1_tomorrow_prices_available` is off

**Solution**:

- Nordpool publishes tomorrow's prices around 13:00 CET
- Wait until after 13:30 CET
- Check your internet connection
- Verify region is correct

### ML Predictions Not Working

**Problem**: ML prediction sensor shows "unavailable"

**Solution**:

1. Check if ML model is trained: `binary_sensor.open_spot_forecast_dk1_ml_model_trained`
2. If not trained, wait for 24+ hours of data collection
3. Configure weather sensors to improve predictions
4. Check Home Assistant logs for errors

### Prices Seem Wrong

**Problem**: Prices are much higher/lower than expected

**Solution**:

1. Check currency setting (DKK, EUR, SEK, NOK)
2. Verify VAT rate is correct for your country
3. Check price unit (kWh vs MWh)
4. If using Stromligning: prices include tariffs/VAT (higher than spot); 25 %
   too high means a `_vat` sensor is configured instead of `_ex_vat`
5. Compare with official Nordpool data

## Tips & Best Practices

### 1. Monitor Confidence Scores

Use confidence to decide when to act:

```yaml
condition:
  - condition: numeric_state
    entity_id: sensor.open_spot_forecast_dk1_prediction_confidence
    above: 70 # Only act on high-confidence predictions
```

### 2. Track Prediction Accuracy

Create a statistics sensor:

```yaml
sensor:
  - platform: statistics
    name: "ML Prediction Accuracy"
    entity_id: sensor.open_spot_forecast_dk1_price_forecast_ml
    state_characteristic: mean
    sampling_size: 100
```

### 3. Use Templates for Cheapest Intervals

Find the 3 cheapest intervals of the forecast (detailed attribute format):

```yaml
template:
  - sensor:
      - name: "Next 3 Cheap Intervals"
        state: >
          {% set forecast = state_attr('sensor.open_spot_forecast_dk1_price_forecast_ml', 'predictions') %}
          {% if forecast %}
            {% set cheapest = forecast | sort(attribute='price') | list %}
            {% for slot in cheapest[:3] %}
              {{ as_datetime(slot.start).strftime('%a %H:%M') }} {{ slot.price }} {{ slot.unit }}{{ ', ' if not loop.last }}
            {% endfor %}
          {% else %}
            unavailable
          {% endif %}
```

## Performance Expectations

### Prediction Accuracy

- **0-24 hours**: ±10-20 DKK/MWh (very accurate)
- **1-3 days**: ±20-40 DKK/MWh (good)
- **4-7 days**: ±40-80 DKK/MWh (reasonable)

### Update Frequency

- Current price: Every 15 minutes
- Tomorrow's prices: Once daily at ~13:10-13:40
- Weather data: Every 6 hours
- ML predictions: Every 6 hours

### Resource Usage

- Memory: ~50-100 MB
- CPU: Minimal (training takes ~10-30 seconds)
- Network: ~10-50 KB per update
- Storage: ~200 KB (learning data)

## Getting Help

### Check Logs

Enable debug logging in `configuration.yaml`:

```yaml
logger:
  default: info
  logs:
    custom_components.open_spot_forecast: debug
```

### Report Issues

When reporting issues, include:

1. Home Assistant version
2. Integration version
3. Region configured
4. Price source (Stromligning, Nordpool, or API)
5. Weather sensors configured
6. Relevant log entries
7. Screenshots of sensors

### Community Support

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: Questions and ideas
- **Home Assistant Forum**: Community help

## Next Steps

1. ✅ Install integration
2. ✅ Configure region
3. ✅ Add price source (Stromligning recommended)
4. ✅ Add weather sensors (optional but recommended)
5. ✅ Create dashboards
6. ✅ Set up automations
7. ✅ Monitor accuracy
8. ✅ Share feedback!

---

**Happy forecasting! 🌤️⚡**
