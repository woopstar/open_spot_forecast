# Open Spot Forecast for Home Assistant

[![GitHub Release][releases-shield]][releases]
[![GitHub Downloads][downloads-shield]][downloads]
[![License][license-shield]][license]
[![BuyMeCoffee][buymecoffeebadge]][buymecoffee]
[![codecov][codecov-shield]][codecov]

## Introduction

**Open Spot Forecast** is a Home Assistant integration that predicts electricity
spot prices using machine learning, weather forecasts, and confirmed market data.
It runs a self-learning loop that compares its predictions against actual prices
and continuously improves accuracy via per-slot bias correction.

---

## Features

### Price Forecasting

- **ML-based spot price prediction** — a Gradient Boosting regressor (200 trees)
  predicts the price up to 7 days ahead at 15-minute resolution (96 slots/day)
- **23-feature model** — time, local weather, zone weather from Open-Meteo
  (wind at 80 m, temperature, irradiance, pressure, humidity across the
  bidding zone), and market-demand features, implemented in pure NumPy (no
  scikit-learn dependency)
- **Real consumer prices** — Stromligning integration provides prices with
  tariffs, fees, and VAT (what you actually pay) for display
- **Spot price forecast** — the model learns the raw day-ahead spot price
  (Stromligning's spot price sensors) and the forecast adds VAT once; tariffs
  are not included

### Self-Learning

- **Per-slot bias correction** — an additive EMA offset per 15-minute slot,
  learned from prediction-vs-actual comparisons; negative prices are forecast
  as negative
- **Solar scaling factor** — a learned EMA ratio between Solcast's estimate and
  actual inverter output
- **Adaptive confidence** — heuristic confidence early on, switching to a
  learned confidence score once enough samples accumulate
- **Live accuracy per lead time** — rolling 30-day MAE and RMSE of the
  forecast 1, 2, 3 and 4+ days ahead, so you can see how far to trust it

### Sensors

- Current price, today/tomorrow min/max/mean
- ML prediction (7-day forecast) with per-slot confidence
- Prediction confidence and learning metrics
- Diagnostic forecast MAE/RMSE sensors per lead time (day 1/2/3/4+)
- Binary sensors for tomorrow's price availability and ML model training status

### Data Sources

- **Stromligning** — confirmed consumer prices (96/day) and the raw spot price
- **Day-ahead prices** — [energy-charts.info](https://energy-charts.info/)
  for every region without an extra integration, with the
  [ENTSO-E Transparency Platform](https://transparency.entsoe.eu/) as
  fallback (API key) and ECB exchange rates for DKK/SEK/NOK
- **Met.no weather** — current weather + 48h hourly forecast (built into HA)
- **Open-Meteo** — 15-minute weather at several points across the bidding
  zone, 8 days ahead (no key; weather data by Open-Meteo.com, CC BY 4.0)
- **Solcast** — solar generation forecast
- **Nordpool prognoses** — market demand and generation forecasts

---

## Quick Start

1. **Install** Open Spot Forecast via HACS or manually.
2. **Configure** your region (DK1, DK2, SE3, SE4, NO2, FI, EE, LT, LV, NL, BE, FR, DE).
3. **Choose a price source** — Stromligning's sensors (DK1/DK2, consumer
   prices with tariffs) or the day-ahead price (all regions, spot price +
   VAT; optionally an ENTSO-E API key as fallback).
4. **Add weather sensors** (optional but recommended) to improve ML accuracy.
5. **Let it learn** — accuracy improves as it accumulates history and self-corrects.

For detailed documentation, see the [`docs/`](docs/) directory.

---

## Requirements

To use this package, you need the following integrations:

- [Stromligning](https://github.com/MTrab/stromligning) — real consumer prices
  (DK1/DK2; not needed with the day-ahead price source)
- A weather entity (Met.no is built into Home Assistant and free)
- [Solcast](https://github.com/BJReplay/ha-solcast-solar) — solar forecast (optional)

---

## Installation

### Method 1: HACS (Home Assistant Community Store)

1. In HACS, go to **Integrations**.
2. Click the three dots in the top-right corner, and select **Custom repositories**.
3. Add this repository URL and select **Integration** as the category:
   `https://github.com/woopstar/open_spot_forecast`
4. Click **Add**.
5. The integration will now appear in HACS under the **Integrations** section. Click **Install**.
6. Restart Home Assistant.

### Method 2: Manual Installation

1. Copy the `open_spot_forecast` folder to your `custom_components` folder in your Home Assistant configuration.
2. Restart Home Assistant.
3. Add the integration via the Home Assistant integrations page and configure your settings.

---

## Removal

1. In Home Assistant, go to **Settings** -> **Devices & Services**.
2. Find the **Open Spot Forecast** integration, click the menu icon (three dots), and select **Delete**.
3. Restart Home Assistant.

### If installed via HACS

4. In HACS, go to **Integrations**, find Open Spot Forecast, click the menu icon (three dots), and select **Remove**.
5. Restart Home Assistant again.

### If installed manually

4. Delete the `custom_components/open_spot_forecast` folder from your Home Assistant configuration directory.
5. Restart Home Assistant.

---

## Documentation

Full documentation is available in the [`docs/`](docs/) directory:

- **[Architecture](docs/architecture.md)** — System overview, data sources, data flow
- **[ML Documentation](docs/ml_documentation.md)** — Model, 23-feature vector, confidence
- **[Self-Learning](docs/self_learning.md)** — Self-learning loop, bias correction
- **[Persistence](docs/persistence.md)** — SQLite storage schema and migrations
- **[Stromligning Integration](docs/stromligning_integration.md)** — Price sources (Stromligning, day-ahead)
- **[Using Existing Sensors](docs/using_existing_sensors.md)** — Sensor wiring reference

---

## Data Sources and Attribution

| Source                                                                       | Used for                                                       | Licence / terms                                                                                                                                                                                                                            |
| ---------------------------------------------------------------------------- | -------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| [Stromligning](https://github.com/MTrab/stromligning)                        | Consumer and spot prices (DK1/DK2, `stromligning` source)      | The Stromligning integration's terms                                                                                                                                                                                                       |
| [energy-charts.info](https://energy-charts.info/) (Fraunhofer ISE)           | Day-ahead prices (`dayahead` source), the backtest's prices    | Per zone, as the API reports it: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), Bundesnetzagentur \| SMARD.de (e.g. DK1, DK2, NO2, NL, FR, DE-LU); EPEX SPOT data for private and internal use only (e.g. SE3, FI, EE, LT, LV) |
| [ENTSO-E Transparency Platform](https://transparency.entsoe.eu/)             | Day-ahead price fallback (with an API key)                     | [ENTSO-E terms and conditions](https://transparency.entsoe.eu/content/static_content/Static%20content/terms%20and%20conditions/terms%20and%20conditions.html)                                                                              |
| [European Central Bank](https://data.ecb.europa.eu/)                         | EUR reference exchange rates (day-ahead prices in DKK/SEK/NOK) | ECB data, reusable with the source acknowledged                                                                                                                                                                                            |
| [Open-Meteo.com](https://open-meteo.com/)                                    | Zone weather forecasts; archived forecasts for the backtest    | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)                                                                                                                                                                                  |
| [Nord Pool](https://data.nordpoolgroup.com/)                                 | Consumption and production prognoses                           | Nord Pool's data portal terms                                                                                                                                                                                                              |
| Met.no (HA weather), [Solcast](https://github.com/BJReplay/ha-solcast-solar) | Local weather, solar forecast                                  | Their integrations' terms                                                                                                                                                                                                                  |

Every entity credits the sources its value comes from in Home Assistant's
**attribution** (the entity's more-info dialog): the price sensors the price
sources (energy-charts.info with the zone's licence, and ENTSO-E when its
fallback is configured), and the forecast, confidence, learning and accuracy
sensors everything the model learns from, e.g. _Prices: energy-charts.info
(CC BY 4.0, Bundesnetzagentur | SMARD.de) · Weather: Open-Meteo.com (CC BY
4.0) · Prognoses: Nord Pool_. Stromligning's prices are credited by the
Stromligning integration.

**Credit: [EpexPredictor](https://github.com/b3nn0/EpexPredictor)** (BSD-3-Clause)
inspired much of OSF's data and model design: gap-aware incremental data
stores, day-ahead prices from energy-charts.info with ENTSO-E as fallback,
weather from Open-Meteo at several points per bidding zone, training on
archived forecasts, and the rolling backtest with a LightGBM reference. OSF
(MIT) reimplements these ideas; no EpexPredictor code is copied, so no BSD-3
notice is needed. Code ported from it in the future must keep its copyright
notice and licence text (e.g. in a `NOTICE` file).

---

[releases-shield]: https://img.shields.io/github/v/release/woopstar/open_spot_forecast?style=for-the-badge
[releases]: https://github.com/woopstar/open_spot_forecast/releases
[downloads-shield]: https://img.shields.io/github/downloads/woopstar/open_spot_forecast/total.svg?style=for-the-badge
[downloads]: https://github.com/woopstar/open_spot_forecast/releases
[license-shield]: https://img.shields.io/github/license/woopstar/open_spot_forecast?style=for-the-badge
[license]: https://github.com/woopstar/open_spot_forecast/blob/main/LICENSE
[buymecoffeebadge]: https://img.shields.io/badge/buy%20me%20a%20coffee-donate-FFDD00.svg?style=for-the-badge&logo=buymeacoffee
[buymecoffee]: https://www.buymeacoffee.com/woopstar
[codecov-shield]: https://codecov.io/github/woopstar/open_spot_forecast/graph/badge.svg
[codecov]: https://codecov.io/github/woopstar/open_spot_forecast
