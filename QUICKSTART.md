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
Stromligning Sensor: sensor.stromligning_current_price_vat
```

**Why Stromligning?**
- Real consumer prices (includes tariffs + VAT)
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

After installation, you'll have:

### Price Sensors
- `sensor.open_spot_forecast_current_price` - Current price
- `sensor.open_spot_forecast_today_min` - Today's lowest
- `sensor.open_spot_forecast_today_max` - Today's highest
- `sensor.open_spot_forecast_today_mean` - Today's average
- `sensor.open_spot_forecast_tomorrow_min` - Tomorrow's lowest
- `sensor.open_spot_forecast_tomorrow_max` - Tomorrow's highest
- `sensor.open_spot_forecast_tomorrow_mean` - Tomorrow's average

### Forecast Sensors
- `sensor.open_spot_forecast_ml_prediction` - **ML-based 7-day forecast**
- `sensor.open_spot_forecast_prediction_confidence` - Prediction confidence (%)

### Learning Sensors
- `sensor.open_spot_forecast_learning_metrics` - Self-learning status

### Status Sensors
- `binary_sensor.open_spot_forecast_tomorrow_available` - Tomorrow's prices ready
- `binary_sensor.open_spot_forecast_ml_model_trained` - ML model status

## Common Use Cases

### 1. Display Current Price

```yaml
type: entities
entities:
  - entity: sensor.open_spot_forecast_current_price
    name: Current Electricity Price
    secondary_info: last-updated
```

### 2. 7-Day Forecast Chart

```yaml
type: custom:apexcharts-card
graph_span: 7d
header:
  show: true
  title: 7-Day Price Forecast
series:
  - entity: sensor.open_spot_forecast_ml_prediction
    type: line
    name: ML Prediction
    data_generator: |
      return entity.attributes.predictions.map(p => {
        return [new Date(p.timestamp).getTime(), p.predicted_price];
      });
```

### 3. Charge EV During Cheap Hours

```yaml
automation:
  - alias: "Charge EV when price is low"
    trigger:
      - platform: time_pattern
        minutes: "/15"
    condition:
      - condition: numeric_state
        entity_id: sensor.open_spot_forecast_current_price
        below: sensor.open_spot_forecast_today_mean
      - condition: numeric_state
        entity_id: sensor.open_spot_forecast_prediction_confidence
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
      - platform: time
        at: "18:00:00"
    condition:
      - condition: template
        value_template: >
          {% set prices = state_attr('sensor.open_spot_forecast_current_price', 'today_prices') %}
          {% if prices %}
            {% set current_hour = now().hour %}
            {% set sorted_hours = prices | sort(attribute='price') %}
            {% set cheapest_hour = sorted_hours[0].hour %}
            {{ current_hour == cheapest_hour }}
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
        entity_id: binary_sensor.open_spot_forecast_tomorrow_available
        to: "on"
    action:
      - service: notify.mobile_app
        data:
          message: "Tomorrow's electricity prices are now available!"
```

## Understanding the Sensors

### Current Price Sensor

**Attributes**:
```yaml
region: DK1
currency: DKK
vat: 0.25
last_update: "2026-07-06T13:15:00"
today_prices: [234.5, 235.1, 236.8, ...]  # 24 values
tomorrow_prices: [240.2, 241.5, ...]       # 24 values (when available)
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
  # ... 96 predictions for the next 24 hours (attributes are capped to stay
  # under Home Assistant's 16 KB attribute limit)

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

**Problem**: `binary_sensor.open_spot_forecast_tomorrow_available` is off

**Solution**:
- Nordpool publishes tomorrow's prices around 13:00 CET
- Wait until after 13:30 CET
- Check your internet connection
- Verify region is correct

### ML Predictions Not Working

**Problem**: ML prediction sensor shows "unavailable"

**Solution**:
1. Check if ML model is trained: `binary_sensor.open_spot_forecast_ml_model_trained`
2. If not trained, wait for 24+ hours of data collection
3. Configure weather sensors to improve predictions
4. Check Home Assistant logs for errors

### Prices Seem Wrong

**Problem**: Prices are much higher/lower than expected

**Solution**:
1. Check currency setting (DKK, EUR, SEK, NOK)
2. Verify VAT rate is correct for your country
3. Check price unit (kWh vs MWh)
4. If using Stromligning: prices include tariffs/VAT (higher than spot)
5. Compare with official Nordpool data

## Tips & Best Practices

### 1. Monitor Confidence Scores

Use confidence to decide when to act:
```yaml
condition:
  - condition: numeric_state
    entity_id: sensor.open_spot_forecast_prediction_confidence
    above: 70  # Only act on high-confidence predictions
```

### 2. Track Prediction Accuracy

Create a statistics sensor:
```yaml
sensor:
  - platform: statistics
    name: "ML Prediction Accuracy"
    entity_id: sensor.open_spot_forecast_ml_prediction
    state_characteristic: mean
    sampling_size: 100
```

### 3. Use Templates for Cheapest Hours

Find the next 3 cheapest hours:
```yaml
template:
  - sensor:
      - name: "Next 3 Cheap Hours"
        state: >
          {% set prices = state_attr('sensor.open_spot_forecast_current_price', 'today_prices') %}
          {% if prices %}
            {% set sorted = prices | sort(attribute='price') %}
            {% set cheapest = sorted[:3] %}
            {% for hour in cheapest %}
              {{ hour.hour }}:00 - {{ hour.price }} DKK/kWh
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
