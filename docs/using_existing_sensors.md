# Using Existing Home Assistant Sensors

Open Spot Forecast reads everything from your existing HA entities. No
external API calls, no duplicate data fetching. Here's what's needed:

## Required

| Entity                                  | Integration                                           | Purpose                               |
| --------------------------------------- | ----------------------------------------------------- | ------------------------------------- |
| `sensor.stromligning_current_price_vat` | [Stromligning](https://github.com/MTrab/stromligning) | Confirmed consumer prices (96/day)    |
| `sensor.stromligning_spotprice_ex_vat`  | Stromligning                                          | Raw spot price excl. VAT: ML target   |
| `weather.forecast_mellemlokken_23`      | Built-in Met.no                                       | Current weather + 48h hourly forecast |

## Recommended

| Entity                                                 | Integration                                             | Purpose                                |
| ------------------------------------------------------ | ------------------------------------------------------- | -------------------------------------- |
| `binary_sensor.stromligning_tomorrow_spotprice_vat`    | Stromligning                                            | Tomorrow's prices when available       |
| `binary_sensor.stromligning_tomorrow_spotprice_ex_vat` | Stromligning                                            | Tomorrow's raw spot price (ML)         |
| `sensor.solcast_pv_forecast_forecast_today`            | [Solcast](https://github.com/BJReplay/ha-solcast-solar) | Solar generation forecast              |
| `sensor.power_inverter_input_total`                    | Your inverter                                           | Actual solar production (for training) |
| `sensor.metroair_330_outdoor_temperature`              | MyUplink/Met.no                                         | Actual temperature (for training)      |

## Weather Entity

A single `weather.*` entity provides everything needed:

**Current state** (read every 15 min for history):

- `temperature`, `humidity`, `wind_speed`, `wind_bearing`, `cloud_coverage`

**Hourly forecast** (via `weather.get_forecasts` service):

- 48 hours of hourly predictions with all the above fields

No DMI API key needed — Met.no is built into Home Assistant and free.

## Solcast

Provides solar generation estimates via the Solcast Home Assistant integration.
The integration compares today's estimate with the inverter's actual output to
learn the solar scaling factor. It is not a price model input: the sensor only
covers today, while forecasts start tomorrow or later, so the model's solar
input is Nordpool's per-slot solar prognosis instead (see
[ML Documentation](ml_documentation.md#feature-vector-17-features)).

## Stromligning

The primary price source. Provides 96 consumer-price intervals per day
(includes tariffs, fees, VAT) at 15-minute resolution, which the price
sensors display. Its spot price sensors (`spotprice_ex_vat`, today and
tomorrow) provide the raw day-ahead spot price excl. VAT, which the ML model
learns and predicts; the forecast sensor adds VAT once. See
[Stromligning Integration](stromligning_integration.md#overview).

The integration also reads the tomorrow availability binary sensor to know
when tomorrow's confirmed prices are published (~13:00 CET). From 13:00 local
it re-reads the prices every ~5 minutes until tomorrow is complete, i.e. has
a price for every 15-minute slot of the next local day (96, or 92/100 on a
DST-change day), and stops for the day at 18:00. Open Spot Forecast's own
`Tomorrow Prices Available` binary sensor turns on only then; its attributes
show `tomorrow_prices_count` against `tomorrow_slots_expected`.
