# Stromligning Integration — Consumer and Spot Prices

## Overview

Open Spot Forecast reads two kinds of prices from the
[Stromligning](https://github.com/MTrab/stromligning) integration, and never
mixes them:

| Value              | Contains                                               | Stromligning entities (defaults)                                                               | Used for                                                               |
| ------------------ | ------------------------------------------------------ | ---------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------- |
| **Consumer price** | Spot price + supplier surcharge + tariffs + taxes, VAT | `sensor.stromligning_current_price_vat`, `binary_sensor.stromligning_tomorrow_spotprice_vat`   | Display: current price and today/tomorrow min/max/mean sensors         |
| **Raw spot price** | Day-ahead spot price only, excl. VAT                   | `sensor.stromligning_spotprice_ex_vat`, `binary_sensor.stromligning_tomorrow_spotprice_ex_vat` | The ML model: training target, self-learning actuals, prediction input |

**The ML model learns and predicts the raw spot price** (#16). Grid tariffs
are time-of-use and change with the season; in a tariff-inclusive target every
tariff change would look like market behaviour to the model and like a model
error to the bias correction. The spot price is what the market sets.

**VAT is applied once, at output.** The `Price Forecast (ML)` sensor shows the
predicted spot price with the configured VAT added, in its state and in every
price attribute (`predictions[].price`, `forecast_min/max/mean`). It does
**not** include tariffs, fees or taxes; adding them is #39. Its attributes say
so: `includes_vat: true`, `includes_tariffs: false`, `vat`. The
`Current Spot Price` sensor shows Stromligning's consumer price
(`includes_vat: true`, `includes_tariffs: true`), despite its name.

Error metrics (learning metrics, forecast MAE/RMSE sensors, bias offsets) are
in the model's unit: raw spot price excl. VAT, in currency/kWh.

The tomorrow sensors: the consumer tomorrow sensor's default,
`binary_sensor.stromligning_tomorrow_spotprice_vat`, holds tomorrow's **spot
price incl. VAT**, so the tomorrow min/max/mean sensors show that by default.
Choose `binary_sensor.stromligning_tomorrow_available_vat` to see tomorrow's
consumer price instead. The ML model always reads the spot price sensors.

## Configuration

1. Install and configure Stromligning (region and electricity provider).
2. In **Developer Tools → States**, find its entities. With the default
   integration name they are `sensor.stromligning_current_price_vat`,
   `binary_sensor.stromligning_tomorrow_spotprice_vat`,
   `sensor.stromligning_spotprice_ex_vat` and
   `binary_sensor.stromligning_tomorrow_spotprice_ex_vat`; another integration
   name changes the `stromligning_` prefix.
3. Enter them in Open Spot Forecast's sensor step (or its options):
   **Stromligning Sensor** and **Stromligning Tomorrow Sensor** (consumer
   price), **Spot Price Sensor** and **Spot Price Tomorrow Sensor** (raw spot
   price for the ML model). Existing installations use the spot defaults
   above until they are changed in the options.

If the spot price sensor has no prices, a warning is logged and the ML
forecast has no input; consumer prices are never used in its place.

## Upgrading from a Version That Trained on Consumer Prices

Earlier versions trained on the consumer price and then added VAT to the
forecast again (VAT twice). The stored price history, pending predictions,
error metrics, bias offsets and lead-time accuracy were all in that price and
cannot be converted (tariffs cannot be subtracted afterwards). They are
discarded once on upgrade (learning database schema v6, logged at info level;
see [persistence](persistence.md)), and the spot price history rebuilds from
the next readings: the model trains again as soon as today's spot prices are
read. Weather snapshots, Nordpool prognoses and hyperparameters are kept.

Automations that compare the ML forecast with a threshold need a new
threshold: the forecast no longer includes tariffs.

## Implementation Details

### Sensor Reader Methods

All reads go through `SensorReader` (`sensor_reader.py`):

- `read_stromligning_sensor(entity_id)` and
  `read_stromligning_tomorrow_sensor(entity_id)` read a Stromligning price
  sensor's `prices` attribute (items with `price`, `start`, `end`) onto the
  day's 15-minute grid (see below).
- `read_spot_prices(entity_id, tomorrow_entity_id)` reads the two spot price
  sensors with the same parsing and returns `today`, `tomorrow`, `raw_today`
  and `raw_tomorrow`. Stromligning fills their `prices` attribute from each
  price's `details.electricity.value`, the spot price excl. VAT.

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

### Price Sources

The price source is chosen when the integration is set up (**Price source**,
`price_source`, #27):

| Source                   | Regions            | Displayed price                         | The model's price                         |
| ------------------------ | ------------------ | --------------------------------------- | ----------------------------------------- |
| `stromligning` (default) | DK1, DK2           | Stromligning's consumer price (all-in)  | Stromligning's `spotprice_ex_vat` sensors |
| `dayahead`               | All 13 OSF regions | Day-ahead spot price + VAT (no tariffs) | The same day-ahead spot price, excl. VAT  |

The config flow rejects `stromligning` for a region outside Denmark
(`stromligning_region`). With `dayahead`, the Stromligning sensors are not
read, even if they are still configured.

**Day-ahead prices** (`api/dayahead_prices.py`) are the day-ahead auction
results:

- [energy-charts.info](https://api.energy-charts.info/) is asked first (no
  key). Its licence differs per zone and comes with every response: CC BY
  4.0 from Bundesnetzagentur | SMARD.de for e.g. DK1, DK2, NO2, NL, FR and
  DE-LU, and "private and internal use only" for e.g. SE3, FI, EE, LT and LV.
- The [ENTSO-E Transparency Platform](https://transparency.entsoe.eu/) is
  the fallback when a security token is configured (**ENTSO-E API key**):
  it is asked when energy-charts fails or does not cover a request, and
  energy-charts wins where both have a price. The token is stored in the
  config entry, sent as a query parameter and never logged.
- Requests span whole local days of the region, so zones east of UTC (e.g.
  FI and EE, whose day starts at 21:00/22:00 UTC) are not cut off. Hourly
  prices fill their four quarter-hours.
- Prices are stored raw in `dayahead_prices` (EUR/MWh per UTC 15-minute
  slot) through the gap-aware time-series source (#32): only missing slots
  are requested, tomorrow from 12:45 CET (when the auction results are due)
  and then every 5 minutes until it is complete.
- They are converted to the configured currency per kWh with the ECB
  reference rate of each day (the latest one on or before it, fetched from
  the ECB data API once a day, `api/exchange_rates.py`). If the ECB cannot
  be reached, DKK uses its ERM II central rate (7.46038); SEK and NOK have
  no prices until the ECB answers.
- With the ML model, the training window's missing days are backfilled at
  setup and after midnight and added to the price history, so the model
  trains on 30 days of prices from the first day.

Without the ML model the day-ahead prices are stored in the same learning
database, which is then opened for them alone.

## Comparison: Stromligning vs Day-Ahead Price

| Feature                  | Stromligning        | Day-ahead price              |
| ------------------------ | ------------------- | ---------------------------- |
| **Price Type**           | Real consumer price | Spot price + VAT             |
| **Includes Tariffs**     | ✅ Yes              | ❌ No                        |
| **Includes VAT**         | ✅ Yes              | ✅ Yes (configured VAT rate) |
| **Includes Fees**        | ✅ Yes              | ❌ No                        |
| **Matches Bill**         | ✅ Yes              | ❌ No                        |
| **Regions**              | DK1, DK2            | All OSF regions              |
| **History for training** | Accumulates daily   | Backfilled (training window) |

## Example Use Cases

### 1. EV Charging Automation

**With Stromligning**:

```yaml
automation:
  - alias: "Charge EV when real price is low"
    condition:
      - condition: numeric_state
        entity_id: sensor.open_spot_forecast_ml_prediction
        below: 1.00 # DKK/kWh: spot price incl. VAT, excl. tariffs
    action:
      - service: switch.turn_on
        target:
          entity_id: switch.ev_charger
```

**Result**: Charges when the forecast spot price (incl. VAT) is below 1.00
DKK/kWh. The forecast excludes tariffs, so compare it with a spot-price
threshold.

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

**Result**: Runs during the 3 cheapest hours of the forecast spot price

### 3. Price Forecast Dashboard

**With Stromligning**:

```yaml
type: custom:apexcharts-card
graph_span: 7d
header:
  title: Spot Price Forecast (incl. VAT, excl. tariffs)
series:
  - entity: sensor.open_spot_forecast_ml_prediction
    type: line
    name: Predicted Cost
    data_generator: |
      return entity.attributes.predictions.map(p => {
        return [new Date(p.timestamp).getTime(), p.predicted_price];
      });
```

**Result**: Shows the forecast spot price incl. VAT per kWh

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

**This is normal** for the consumer price sensors (current price, today and
tomorrow min/max/mean), which show what you pay:

- Spot price: ~1.85 DKK/kWh
- With tariffs/VAT: ~2.45 DKK/kWh

The ML forecast is lower: it is the spot price incl. VAT, without tariffs
(see [Overview](#overview)).

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
3. **Display**: The current price sensors show what you pay
4. **ML forecast**: Unchanged in kind: trained on and predicting the raw spot
   price (Stromligning's spot price sensors), VAT added once

## Summary

**Stromligning integration provides:**

✅ Real consumer prices (with tariffs/VAT) for display  
✅ The raw spot price the ML model learns and predicts  
✅ Matches your electricity bill (consumer price sensors)  
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
