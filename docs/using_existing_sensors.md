# Using Existing Home Assistant Sensors

Open Spot Forecast reads everything from your existing HA entities. No
external API calls, no duplicate data fetching. Here's what's needed:

## Required

| Entity                                  | Integration                                           | Purpose                               |
| --------------------------------------- | ----------------------------------------------------- | ------------------------------------- |
| `sensor.stromligning_current_price_vat` | [Stromligning](https://github.com/MTrab/stromligning) | Confirmed consumer prices (96/day)    |
| `weather.forecast_mellemlokken_23`      | Built-in Met.no                                       | Current weather + 48h hourly forecast |

## Recommended

| Entity                                              | Integration                                             | Purpose                                |
| --------------------------------------------------- | ------------------------------------------------------- | -------------------------------------- |
| `binary_sensor.stromligning_tomorrow_spotprice_vat` | Stromligning                                            | Tomorrow's prices when available       |
| `sensor.solcast_pv_forecast_forecast_today`         | [Solcast](https://github.com/BJReplay/ha-solcast-solar) | Solar generation forecast              |
| `sensor.power_inverter_input_total`                 | Your inverter                                           | Actual solar production (for training) |
| `sensor.metroair_330_outdoor_temperature`           | MyUplink/Met.no                                         | Actual temperature (for training)      |

## Weather Entity

A single `weather.*` entity provides everything needed:

**Current state** (read every 15 min for history):

- `temperature`, `humidity`, `wind_speed`, `wind_bearing`, `cloud_coverage`

**Hourly forecast** (via `weather.get_forecasts` service):

- 48 hours of hourly predictions with all the above fields

No DMI API key needed — Met.no is built into Home Assistant and free.

## Solcast

Provides solar generation estimates via the Solcast Home Assistant integration.
The integration reads the `detailed_hourly` attribute for per-hour pv_estimate
values, which feed into the solar features.

## Stromligning

The primary price source. Provides 96 consumer-price intervals per day
(includes tariffs, fees, VAT) at 15-minute resolution. This is what you
actually pay — not the raw Nordpool spot price.

The integration also reads the tomorrow availability binary sensor to know
when tomorrow's confirmed prices are published (~13:00 CET).
