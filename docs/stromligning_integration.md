# Stromligning Integration — Consumer and Spot Prices

## Overview

Open Spot Forecast reads two prices per 15-minute slot from the
[Stromligning](https://github.com/MTrab/stromligning) integration, both
**excl. VAT**:

| Value              | Contains                                        | Stromligning entities (defaults)                                                                   | Used for                                                               |
| ------------------ | ----------------------------------------------- | -------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------- |
| **Consumer price** | Spot price + supplier surcharge + tax + tariffs | `sensor.stromligning_current_price_ex_vat`, `binary_sensor.stromligning_tomorrow_available_ex_vat` | Display (current price, today/tomorrow min/max/mean) and the tariffs   |
| **Raw spot price** | Day-ahead spot price only                       | `sensor.stromligning_spotprice_ex_vat`, `binary_sensor.stromligning_tomorrow_spotprice_ex_vat`     | The ML model: training target, self-learning actuals, prediction input |

All four `_ex_vat` entities are **disabled by default** in Stromligning:
enable them under Settings → Devices & services → Stromligning → Entities.

[Predbat](https://springfall2008.github.io/batpred/) reads its Danish rates
from Stromligning's `_vat` / `spotprice_ex_vat` entities, which hold today
and tomorrow only. The optional Predbat rate entities expose OSF's 7-day
forecast in the same shape (import with tariffs and VAT, export the raw spot
price) as a drop-in for Predbat's `apps.yaml` (#124): see
[QUICKSTART.md → Predbat](../QUICKSTART.md#predbat).

**The ML model learns and predicts the raw spot price** (#16). Grid tariffs
are time-of-use and change with the season; in a tariff-inclusive target every
tariff change would look like market behaviour to the model and like a model
error to the bias correction. The spot price is what the market sets.

**The forecast adds each slot's tariff** (#107). For every slot of today and
tomorrow, consumer minus spot price is everything else on the bill, excl.
VAT. Example (DK1, one slot, DKK/kWh):

| Part                              | Stromligning sensor        | excl. VAT |
| --------------------------------- | -------------------------- | --------- |
| Spot price                        | Spotprice                  | 0.26      |
| Supplier surcharge                | Spotprice surcharge        | 0.012     |
| Electricity tax                   | Electricity tax            | 0.01      |
| Energinet net tariff              | Nettariff                  | 0.04      |
| Energinet system tariff           | Systemtariff               | 0.07      |
| Grid company tariff (time-of-use) | Provider transportexpenses | 0.10      |
| **Consumer price** (tariff: 0.23) | **Current**                | **0.49**  |

The `Price Forecast (ML)` sensor shows
`(spot + tariff + surcharge) × (1 + VAT)` in its state and every price
attribute (`predictions[].price`, `forecast_min/max/mean`), with
`includes_tariffs: true`. Days past tomorrow repeat the latest known day's
tariff at the same local time of day (`TariffSchedule`, `tariffs.py`), so a
new tariff season (e.g. the winter tariffs from 1 October) is picked up as
soon as its first day is published. The separate tariff sensors are not
read: they only report the current value (and the distribution sensor has no
tomorrow).

**VAT is applied once, at output**, from OSF's VAT option: to the consumer
prices and to the forecast alike (`(price + surcharge) × (1 + VAT)`, #39).
The consumer price already holds the supplier's surcharge, so keep the
**surcharge** option at 0 with Stromligning. See
[Architecture → Price Output](architecture.md#price-output).

Error metrics (learning metrics, forecast MAE/RMSE sensors, bias offsets) are
in the model's unit: raw spot price excl. VAT, in currency/kWh. The
`Forecast evaluation` sensor adds the slot's tariff to both its predicted and
its actual price, so its error is still the spot price's.

## Configuration

1. Install and configure Stromligning (region and electricity provider), and
   enable its four `_ex_vat` price entities (see above).
2. In **Developer Tools → States**, find them. With the default integration
   name they are `sensor.stromligning_current_price_ex_vat`,
   `binary_sensor.stromligning_tomorrow_available_ex_vat`,
   `sensor.stromligning_spotprice_ex_vat` and
   `binary_sensor.stromligning_tomorrow_spotprice_ex_vat`; another integration
   name changes the `stromligning_` prefix.
3. Open Spot Forecast uses these as defaults. Change them in its sensor step
   (or its options) if needed: **Consumer Price Sensor** and **Consumer
   Price Tomorrow Sensor**, **Spot Price Sensor** and **Spot Price Tomorrow
   Sensor**. Only choose `_ex_vat` entities: OSF adds VAT itself.

If the spot price sensor has no prices, a warning is logged and the ML
forecast has no input; consumer prices are never used in its place. Without
consumer prices the forecast has no tariffs (`includes_tariffs: false`).

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
  day's 15-minute grid (see below): the consumer price excl. VAT.
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
| `stromligning` (default) | DK1, DK2           | Consumer price + VAT; forecast + tariff | Stromligning's `spotprice_ex_vat` sensors |
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
  config entry, sent as a query parameter and never logged. It can be
  entered at setup or added later in the options, and it also gives the
  model ENTSO-E's week-ahead load forecast (see
  [ML documentation](ml_documentation.md#feature-vector-26-features), #30).
  The options take effect when the integration is reloaded.
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
- With the ML model, the training window's missing days (default 60, #24)
  are backfilled at setup and after midnight and added to the price history,
  so the model trains on its full window from the first day. This also
  happens with the Stromligning source: energy-charts' day-ahead spot price
  is the same series as Stromligning's `spotprice_ex_vat` sensor.

Without the ML model the day-ahead prices are stored in the same learning
database, which is then opened for them alone.

## Comparison: Stromligning vs Day-Ahead Price

| Feature                  | Stromligning                 | Day-ahead price              |
| ------------------------ | ---------------------------- | ---------------------------- |
| **Price Type**           | Real consumer price          | Spot price + VAT             |
| **Includes Tariffs**     | ✅ Yes (forecast too)        | ❌ No                        |
| **Includes VAT**         | ✅ Yes (configured VAT rate) | ✅ Yes (configured VAT rate) |
| **Includes Fees**        | ✅ Yes                       | ❌ No                        |
| **Matches Bill**         | ✅ Yes                       | ❌ No                        |
| **Regions**              | DK1, DK2                     | All OSF regions              |
| **History for training** | Accumulates daily            | Backfilled (training window) |

## Example Use Cases

### 1. EV Charging Automation

**With Stromligning**:

```yaml
automation:
  - alias: "Charge EV when real price is low"
    condition:
      - condition: numeric_state
        entity_id: sensor.open_spot_forecast_dk1_price_forecast_ml
        below: 2.00 # DKK/kWh: incl. tariffs and VAT, like your bill
    action:
      - service: switch.turn_on
        target:
          entity_id: switch.ev_charger
```

**Result**: Charges when the forecast price is below 2.00 DKK/kWh. With
Stromligning the forecast includes each slot's tariffs (#107), so compare it
with what you pay; with the day-ahead source it is the spot price incl. VAT.

### 2. Dishwasher Scheduling

**With Stromligning**:

```yaml
automation:
  - alias: "Run dishwasher during cheapest 3 hours"
    trigger:
      - platform: time_pattern
        minutes: "/15"
    condition:
      # One of the 3 cheapest intervals of the forecast (detailed attribute format)
      - condition: template
        value_template: >
          {% set forecast = state_attr('sensor.open_spot_forecast_dk1_price_forecast_ml', 'predictions') %}
          {% set cheapest = forecast | sort(attribute='price') | list %}
          {% set ns = namespace(hit=false) %}
          {% for slot in cheapest[:3] if as_datetime(slot.start) <= now() < as_datetime(slot.end) %}
            {% set ns.hit = true %}
          {% endfor %}
          {{ ns.hit }}
    action:
      - service: switch.turn_on
        target:
          entity_id: switch.dishwasher
```

**Result**: Runs during the 3 cheapest 15-minute intervals of the forecast
price, tariffs included (the grid tariff's evening peak counts)

### 3. Price Forecast Dashboard

**With Stromligning**:

```yaml
type: custom:apexcharts-card
graph_span: 3d
header:
  show: true
  title: Price Forecast (incl. tariffs and VAT)
series:
  - entity: sensor.open_spot_forecast_dk1_price_forecast_ml
    name: Predicted Cost
    curve: stepline
    extend_to: false
    data_generator: |
      return entity.attributes.predictions.map((p) => [
        new Date(p.start).getTime(),
        p.price,
      ]);
```

**Result**: Shows the forecast price incl. tariffs and VAT per kWh (detailed
attribute format). More cards, for the compact format and predicted vs
actual: [QUICKSTART.md → Dashboard](../QUICKSTART.md#dashboard).

## Troubleshooting

### Stromligning Sensor Not Found

**Problem**: "Stromligning sensor sensor.stromligning_current_price_ex_vat not found"

**Solutions**:

1. Verify Stromligning integration is installed
2. Check entity ID in Developer Tools → States
3. Ensure the sensor is enabled: Stromligning's `_ex_vat` entities are
   disabled by default
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

**This is normal**: the consumer price sensors (current price, today and
tomorrow min/max/mean) and the ML forecast show what you pay, tariffs and
VAT included (see [Overview](#overview)):

- Spot price: ~0.33 DKK/kWh incl. VAT
- With tariffs, tax and fees: ~0.62 DKK/kWh incl. VAT

The grid company's tariff peaks from 17:00 to 21:00, so evening prices are
much higher than the spot price alone.

### Prices Are 25 % Too High

The consumer or spot sensor is the `_vat` variant: OSF adds VAT itself.
Choose the `_ex_vat` entities (see [Configuration](#configuration)).

## Migration from Nordpool

### Before (Nordpool)

```yaml
Nordpool Sensor: sensor.nordpool_kwh_dk1_dkk_3_10_025
```

**Result**: Spot prices only (185 DKK/MWh)

### After (Stromligning)

```yaml
Consumer Price Sensor: sensor.stromligning_current_price_ex_vat
```

**Result**: Real consumer prices (0.62 DKK/kWh incl. VAT = 620 DKK/MWh)

### What Changes

1. **Price Scale**: DKK/MWh → DKK/kWh (divide by 1000)
2. **Price Value**: Spot → Real consumer (includes tariffs/VAT)
3. **Display**: The current price sensors show what you pay
4. **ML forecast**: trained on and predicting the raw spot price
   (Stromligning's spot price sensors); each slot's tariff and VAT are added
   once, at output

## Summary

**Stromligning integration provides:**

✅ Real consumer prices (with tariffs/VAT) for display  
✅ The raw spot price the ML model learns and predicts  
✅ Per-slot tariffs in the forecast  
✅ Matches your electricity bill (consumer price sensors and forecast)  
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
