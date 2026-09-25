# Stromligning Integration - Real Consumer Prices

## Overview

Open Spot Forecast now supports **Stromligning** as the primary price source, providing **real consumer prices** that include tariffs, fees, and VAT — not just spot prices.

## Why Stromligning is Better

### ✅ Real Consumer Prices

- **Includes tariffs** (grid fees, transmission fees)
- **Includes VAT** (25% in Denmark)
- **Includes other fees** (PSO, balance tariffs, etc.)
- **What you actually pay** on your electricity bill

### ✅ Accurate Cost Predictions

- ML model trained on real costs, not spot prices
- Predictions show what you'll actually pay
- Better for automation decisions (when to charge EV, run dishwasher, etc.)

### ✅ Already Configured

- You have Stromligning installed and working
- No duplicate API calls
- Leverages existing integration

### ✅ Region-Specific

- Handles local tariffs automatically
- Correct for your specific grid area
- Updated when tariffs change

## Priority System

Open Spot Forecast uses a **priority system** for price sources:

```
1. Stromligning (if configured) ← BEST: Real consumer prices
   ↓
2. Nordpool (if configured) ← GOOD: Spot prices only
   ↓
3. Nordpool API (fallback) ← OK: Direct API calls
```

## Configuration

### Step 1: Install Stromligning

Make sure you have Stromligning installed:

- Repository: https://github.com/MTrab/stromligning
- Configure it with your region and electricity provider

### Step 2: Find Your Stromligning Sensor

1. Go to **Developer Tools** → **States**
2. Search for `stromligning`
3. Look for: `sensor.stromligning_current_price_vat` (or similar)
4. Copy the entity ID

### Step 3: Configure Open Spot Forecast

In the integration options, enter:

```yaml
Stromligning Sensor: sensor.stromligning_current_price_vat
```

**Leave Nordpool Sensor empty** (Stromligning takes priority)

## What Stromligning Provides

### Sensor Data Structure

```python
{
    "current_price": 2.45,  # DKK/kWh (with tariffs and VAT)
    "today": [2.34, 2.35, 2.36, ...],  # 24 hourly prices
    "tomorrow": [2.40, 2.41, ...],      # 24 hourly prices (after 13:00)
    "raw_today": [
        {"start": "2026-07-06T00:00:00", "end": "2026-07-06T01:00:00", "value": 2.34},
        ...
    ],
    "raw_tomorrow": [...],
    "spot_price": 1.85,      # Just the spot price component
    "tariffs": 0.35,         # Tariff component
    "vat": 0.25,             # VAT rate (25%)
}
```

### Price Breakdown Example

For a price of **2.45 DKK/kWh**:

- Spot price: 1.85 DKK/kWh
- Tariffs: 0.35 DKK/kWh
- Subtotal: 2.20 DKK/kWh
- VAT (25%): 0.25 DKK/kWh
- **Total: 2.45 DKK/kWh** ← What you pay

## Benefits for ML Predictions

### Before (Spot Prices Only)

```
ML Model trained on: 185 DKK/MWh (spot price)
Prediction: 200 DKK/MWh
User sees: 200 DKK/MWh
Actual bill: 245 DKK/MWh (with tariffs/VAT)
❌ Prediction doesn't match reality
```

### After (Real Consumer Prices)

```
ML Model trained on: 2.45 DKK/kWh (real price)
Prediction: 2.50 DKK/kWh
User sees: 2.50 DKK/kWh
Actual bill: 2.45 DKK/kWh
✅ Prediction matches reality!
```

## Implementation Details

### Sensor Reader Method

```python
def read_stromligning_sensor(self, entity_id: str) -> dict:
    """Read Stromligning sensor data.

    Returns:
        Dictionary with real consumer prices (including tariffs/VAT)
    """
    result = {
        "current_price": None,
        "today": [],
        "tomorrow": [],
        "raw_today": [],
        "raw_tomorrow": [],
        "spot_price": None,
        "tariffs": None,
        "vat": None,
    }

    # Read sensor state and attributes
    state = self.hass.states.get(entity_id)
    result["current_price"] = float(state.state)
    result["today"] = state.attributes.get("today", [])
    result["tomorrow"] = state.attributes.get("tomorrow", [])
    result["spot_price"] = state.attributes.get("spot_price")
    result["tariffs"] = state.attributes.get("tariffs")
    result["vat"] = state.attributes.get("vat")

    return result
```

### Price Grid

The reader places each day's prices on that local day's 15-minute grid:
one value per slot from local midnight, 96 slots or 92/100 on a DST-change
day (`align_to_grid()` in `price_series.py`). Prices are placed by the
source's own timestamps, never by their position in the list:

- Each price item fills the slots from its `start` (or `timestamp`/`time`)
  to its `end`. Without an `end` it lasts the series' resolution, the
  smallest gap between consecutive starts: 15 minutes, or 60 for hourly
  prices, which are expanded to four slots.
- A gap of up to 4 slots between two known prices takes the earlier price.
  Longer gaps, and slots before the first or after the last known price
  (e.g. a partial publication), stay missing (`null` in `today_prices`).
  Missing slots are shown as missing, skipped by the min/max/mean sensors,
  and excluded from training and self-learning.
- Items dated outside today or tomorrow are ignored, also on the tomorrow
  sensor.
- A `today`/`tomorrow` attribute list has no timestamps, so it is used only
  when it is exactly one local day long: a price per 15-minute slot or per
  hour.
- If the sensor only reports its current price (e.g. around midnight), that
  price fills the current slot.

### Invalid Price Data

A price source that is failing (for example right after a Home Assistant
restart, or during an API hiccup) often reports 0 for every slot. The reader
checks each day's prices with `is_invalid_price_series()` (`price_series.py`)
and drops a day whose known prices are:

- all zero, or
- not all finite (NaN, inf).

A missing slot does not make a day invalid (see [Price Grid](#price-grid)).

A dropped day reads as "no data": Open Spot Forecast keeps its previous
prices for that day, and nothing is stored, trained on or learned from.
A warning is logged once per streak of bad reads, then at debug level until
the sensor reports valid prices again. If the sensor has no valid prices at
startup, it is read again on every 15-minute update.

Some zero or negative prices are normal (for example on windy, sunny days)
and keep a day valid. A price of exactly 0 is kept as a price, not skipped.

### Priority Logic

```python
# Priority 1: Stromligning (real consumer prices)
if stromligning_sensor:
    stromligning_data = sensor_reader.read_stromligning_sensor(stromligning_sensor)
    if stromligning_data["today"]:
        prices_today = stromligning_data["today"]
        prices_tomorrow = stromligning_data["tomorrow"]
    else:
        # Fall through to Nordpool
        pass

# Priority 2: Nordpool (spot prices only)
if not stromligning_sensor and nordpool_sensor:
    nordpool_data = sensor_reader.read_nordpool_sensor(nordpool_sensor)
    if nordpool_data["today"]:
        prices_today = nordpool_data["today"]
        prices_tomorrow = nordpool_data["tomorrow"]

# Priority 3: API fallback
elif not stromligning_sensor and not nordpool_sensor:
    await nordpool_api.update_prices()
    prices_today = nordpool_api.today
    prices_tomorrow = nordpool_api.tomorrow
```

## Comparison: Stromligning vs Nordpool

| Feature                  | Stromligning        | Nordpool            |
| ------------------------ | ------------------- | ------------------- |
| **Price Type**           | Real consumer price | Spot price only     |
| **Includes Tariffs**     | ✅ Yes              | ❌ No               |
| **Includes VAT**         | ✅ Yes              | ❌ No               |
| **Includes Fees**        | ✅ Yes              | ❌ No               |
| **Matches Bill**         | ✅ Yes              | ❌ No               |
| **Region-Specific**      | ✅ Yes              | ❌ No               |
| **Automation Decisions** | ✅ Accurate         | ⚠️ Needs conversion |
| **ML Training**          | ✅ Real costs       | ⚠️ Spot prices only |

## Example Use Cases

### 1. EV Charging Automation

**With Stromligning**:

```yaml
automation:
  - alias: "Charge EV when real price is low"
    condition:
      - condition: numeric_state
        entity_id: sensor.open_spot_forecast_ml_prediction
        below: 2.00 # DKK/kWh (real price you'll pay)
    action:
      - service: switch.turn_on
        target:
          entity_id: switch.ev_charger
```

**Result**: Charges when you'll actually pay less than 2.00 DKK/kWh

### 2. Dishwasher Scheduling

**With Stromligning**:

```yaml
automation:
  - alias: "Run dishwasher during cheapest 3 hours"
    trigger:
      - platform: time
        at: "18:00:00"
    condition:
      - condition: template
        value_template: >
          {% set prices = state_attr('sensor.open_spot_forecast_ml_prediction', 'predictions') %}
          {% set current_hour = now().hour %}
          {% set sorted = prices | sort(attribute='predicted_price') %}
          {% set cheapest_hours = sorted[:3] | map(attribute='hour') | list %}
          {{ current_hour in cheapest_hours }}
    action:
      - service: switch.turn_on
        target:
          entity_id: switch.dishwasher
```

**Result**: Runs during the 3 cheapest hours based on real costs

### 3. Price Forecast Dashboard

**With Stromligning**:

```yaml
type: custom:apexcharts-card
graph_span: 7d
header:
  title: Real Electricity Cost Forecast (incl. tariffs & VAT)
series:
  - entity: sensor.open_spot_forecast_ml_prediction
    type: line
    name: Predicted Cost
    data_generator: |
      return entity.attributes.predictions.map(p => {
        return [new Date(p.timestamp).getTime(), p.predicted_price];
      });
```

**Result**: Shows what you'll actually pay per kWh

## Troubleshooting

### Stromligning Sensor Not Found

**Problem**: "Stromligning sensor sensor.stromligning_current_price_vat not found"

**Solutions**:

1. Verify Stromligning integration is installed
2. Check entity ID in Developer Tools → States
3. Ensure sensor is not disabled
4. Restart Home Assistant

### Stromligning Has No Data

**Problem**: "Stromligning sensor has no data, falling back to Nordpool/API"

**Solutions**:

1. Check Stromligning integration is working
2. Verify sensor state is not "unknown" or "unavailable"
3. Check Stromligning integration logs
4. Wait for next update cycle

### Prices Seem High

**Problem**: Prices are higher than expected

**This is normal!** Stromligning shows real consumer prices:

- Spot price: ~1.85 DKK/kWh
- With tariffs/VAT: ~2.45 DKK/kWh
- **This is what you actually pay**

Check your electricity bill to verify.

## Migration from Nordpool

### Before (Nordpool)

```yaml
Nordpool Sensor: sensor.nordpool_kwh_dk1_dkk_3_10_025
```

**Result**: Spot prices only (185 DKK/MWh)

### After (Stromligning)

```yaml
Stromligning Sensor: sensor.stromligning_current_price_vat
```

**Result**: Real consumer prices (2.45 DKK/kWh = 2450 DKK/MWh)

### What Changes

1. **Price Scale**: DKK/MWh → DKK/kWh (divide by 1000)
2. **Price Value**: Spot → Real consumer (includes tariffs/VAT)
3. **ML Training**: Trained on real costs
4. **Predictions**: Show what you'll actually pay

## Summary

**Stromligning integration provides:**

✅ Real consumer prices (with tariffs/VAT)  
✅ Accurate cost predictions  
✅ Matches your electricity bill  
✅ Better for automation decisions  
✅ Region-specific tariffs  
✅ Already configured in your system  
✅ Priority 1 in Open Spot Forecast

**This is the best price source for Open Spot Forecast!** 💰✨

---

**Status**: ✅ Complete and integrated  
**Priority**: 1 (highest)  
**Fallback**: Nordpool sensor → Nordpool API  
**Configuration**: Via integration options UI  
**Documentation**: Complete
