# Using Existing Home Assistant Sensors

Open Spot Forecast reads everything from your existing HA entities. No
external API calls, no duplicate data fetching. Here's what's needed:

## Required

| Entity                                     | Integration                                           | Purpose                                    |
| ------------------------------------------ | ----------------------------------------------------- | ------------------------------------------ |
| `sensor.stromligning_current_price_ex_vat` | [Stromligning](https://github.com/MTrab/stromligning) | Consumer price excl. VAT (96/day): tariffs |
| `sensor.stromligning_spotprice_ex_vat`     | Stromligning                                          | Raw spot price excl. VAT: ML target        |
| `weather.forecast_mellemlokken_23`         | Built-in Met.no                                       | Current weather + 48h hourly forecast      |

Both Stromligning `_ex_vat` sensors are **disabled by default** in
Stromligning: enable them under Settings → Devices & services →
Stromligning → Entities.

## Recommended

| Entity                                                 | Integration                                             | Purpose                                |
| ------------------------------------------------------ | ------------------------------------------------------- | -------------------------------------- |
| `binary_sensor.stromligning_tomorrow_available_ex_vat` | Stromligning                                            | Tomorrow's consumer price (tariffs)    |
| `binary_sensor.stromligning_tomorrow_spotprice_ex_vat` | Stromligning                                            | Tomorrow's raw spot price (ML)         |
| `sensor.solcast_pv_forecast_forecast_today`            | [Solcast](https://github.com/BJReplay/ha-solcast-solar) | Solar generation forecast              |
| `sensor.power_inverter_input_total`                    | Your inverter                                           | Actual solar production (solar scale)  |
| `sensor.metroair_330_outdoor_temperature`              | MyUplink/Met.no                                         | Actual temperature (forecast accuracy) |

## Optional: External Forecasts

| Entity                                | Integration                                                       | Purpose                           |
| ------------------------------------- | ----------------------------------------------------------------- | --------------------------------- |
| `sensor.stromligning_forecasts_vat`   | Stromligning (enable its forecast option)                         | Stromligning's own price forecast |
| Your Energi Data Service price sensor | [Energi Data Service](https://github.com/MTrab/energidataservice) | Its `forecast` attribute (Carnot) |

Set them as **External forecast sensors** (config or options flow, several
allowed, none by default). They are never a model input: every forecast run
records what they show for the slots the ML forecast covers, and
self-learning scores them per lead time next to the ML forecast (the
`external` attribute of the forecast MAE/RMSE sensors). A sensor must list
its forecast in a `prices` (`start`, `price`) or `forecast` (`hour`, `price`)
attribute, in the terms this integration shows prices in (unit, tariffs,
VAT). See [Self-Learning](self_learning.md#external-forecasts-120).

## Weather Entity

A single `weather.*` entity scores the local weather forecast for the
confidence (its forecast error lowers the learned confidence). It is not a
model input since #23: the model's weather is Open-Meteo's zone forecast,
trained on archived forecasts (see
[ML Documentation](ml_documentation.md#training-vs-prediction-segmentation)).

**Current state** (read every 15 min into `weather_history`):

- `temperature`, `humidity`, `wind_speed`, `wind_bearing`, `cloud_coverage`

**Hourly forecast** (via `weather.get_forecasts` service):

- 48 hours of hourly predictions with all the above fields; the forecast for
  a slot is recorded with its prediction and compared with the slot's
  snapshot

No DMI API key needed — Met.no is built into Home Assistant and free.

## Solcast

Provides solar generation estimates via the Solcast Home Assistant integration.
The integration compares today's estimate with the inverter's actual output to
learn the solar scaling factor. It is not a price model input: the sensor only
covers today, while forecasts start tomorrow or later, so the model's solar
input is Nordpool's per-slot solar prognosis instead (see
[ML Documentation](ml_documentation.md#feature-vector-26-features)).

## Stromligning

The primary price source. Provides 96 consumer-price intervals per day
(includes tariffs, fees and tax) at 15-minute resolution, read excl. VAT
(`current_price_ex_vat`, `tomorrow_available_ex_vat`), which the price
sensors display with OSF's VAT. Its spot price sensors (`spotprice_ex_vat`,
today and tomorrow) provide the raw day-ahead spot price excl. VAT, which the
ML model learns and predicts. Consumer minus spot is each slot's tariff,
which the forecast adds to the predicted spot price (#107). See
[Stromligning Integration](stromligning_integration.md#overview).

The integration also reads the tomorrow availability binary sensor to know
when tomorrow's confirmed prices are published (~13:00 CET). From 13:00 local
it re-reads the prices every ~5 minutes until tomorrow is complete, i.e. has
a price for every 15-minute slot of the next local day (96, or 92/100 on a
DST-change day), and stops for the day at 18:00. Open Spot Forecast's own
`Tomorrow Prices Available` binary sensor turns on only then; its attributes
show `tomorrow_prices_count` against `tomorrow_slots_expected`.
