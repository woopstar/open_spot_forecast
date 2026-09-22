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
- **20-feature model** — time, wind, solar, temperature, and market-demand
  features, implemented in pure NumPy (no scikit-learn dependency)
- **Real consumer prices** — Stromligning integration provides prices with
  tariffs, fees, and VAT (what you actually pay), with Nordpool as fallback

### Self-Learning

- **Per-slot bias correction** — a multiplicative EMA correction factor per
  15-minute slot, learned from prediction-vs-actual comparisons
- **Solar scaling factor** — a learned EMA ratio between Solcast's estimate and
  actual inverter output
- **Adaptive confidence** — heuristic confidence early on, switching to a
  learned confidence score once enough samples accumulate

### Sensors

- Current price, today/tomorrow min/max/mean
- ML prediction (7-day forecast) with per-slot confidence
- Prediction confidence and learning metrics
- Binary sensors for tomorrow's price availability and ML model training status

### Data Sources

- **Stromligning** — confirmed consumer prices (96/day)
- **Met.no weather** — current weather + 48h hourly forecast (built into HA)
- **Solcast** — solar generation forecast
- **Nordpool prognoses** — market demand and generation forecasts

---

## Quick Start

1. **Install** Open Spot Forecast via HACS or manually.
2. **Configure** your region (DK1, DK2, SE3, SE4, NO2, FI, EE, LT, LV, NL, BE, FR, DE).
3. **Add a price source** — Stromligning (recommended) or Nordpool.
4. **Add weather sensors** (optional but recommended) to improve ML accuracy.
5. **Let it learn** — accuracy improves as it accumulates history and self-corrects.

For detailed documentation, see the [`docs/`](docs/) directory.

---

## Requirements

To use this package, you need the following integrations:

- [Stromligning](https://github.com/MTrab/stromligning) — real consumer prices (recommended)
- A weather entity (Met.no is built into Home Assistant and free)
- [Solcast](https://github.com/BJReplay/ha-solcast-solar) — solar forecast (optional)
- Any electricity price integration ([Nordpool](https://github.com/custom-components/nordpool), etc.)

---

## Installation

### Method 1: HACS (Home Assistant Community Store)

1. In HACS, go to **Integrations**.
2. Click the three dots in the top-right corner, and select **Custom repositories**.
3. Add this repository URL and select **Integration** as the category:
   `https://github.com/woopstar/openspotforecast`
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

- **[Architecture](docs/ARCHITECTURE.md)** — System overview, data sources, data flow
- **[ML Documentation](docs/ML_DOCUMENTATION.md)** — Model, 20-feature vector, confidence
- **[Self-Learning](docs/SELF_LEARNING.md)** — Self-learning loop, bias correction
- **[Persistence](docs/PERSISTENCE.md)** — SQLite storage schema and migrations
- **[Stromligning Integration](docs/STROMLIGNING_INTEGRATION.md)** — Price-source priority
- **[Using Existing Sensors](docs/USING_EXISTING_SENSORS.md)** — Sensor wiring reference

---

[releases-shield]: https://img.shields.io/github/v/release/woopstar/openspotforecast?style=for-the-badge
[releases]: https://github.com/woopstar/openspotforecast/releases
[downloads-shield]: https://img.shields.io/github/downloads/woopstar/openspotforecast/total.svg?style=for-the-badge
[downloads]: https://github.com/woopstar/openspotforecast/releases
[license-shield]: https://img.shields.io/github/license/woopstar/openspotforecast?style=for-the-badge
[license]: https://github.com/woopstar/openspotforecast/blob/main/LICENSE
[buymecoffeebadge]: https://img.shields.io/badge/buy%20me%20a%20coffee-donate-FFDD00.svg?style=for-the-badge&logo=buymeacoffee
[buymecoffee]: https://www.buymeacoffee.com/woopstar
[codecov-shield]: https://codecov.io/github/woopstar/openspotforecast/graph/badge.svg
[codecov]: https://codecov.io/github/woopstar/openspotforecast