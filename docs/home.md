# Open Spot Forecast

**Open Spot Forecast** is a Home Assistant integration that predicts electricity
spot prices using machine learning, weather forecasts, and confirmed market
data. It runs a self-learning loop that compares its predictions against actual
prices and continuously improves accuracy via per-slot bias correction.

## What It Does

- **ML-based spot price prediction** — a Gradient Boosting regressor (200 trees)
  predicts the price up to 7 days ahead at 15-minute resolution (96 slots/day).
- **17-feature model** — time, wind, solar, temperature, and market-demand
  features, implemented in pure NumPy (no scikit-learn dependency).
- **Real consumer prices** — Stromligning provides prices with tariffs, fees,
  and VAT (what you actually pay), with Nordpool as fallback.
- **Self-learning** — a per-slot additive bias-correction loop that
  improves accuracy as predictions are compared against confirmed prices.
- **No external API keys** — reads everything from your existing Home Assistant
  entities (Met.no weather, Stromligning, Solcast).

## Quick Start

1. **Install** Open Spot Forecast via HACS or manually.
2. **Configure** your region (DK1, DK2, SE3, SE4, NO2, FI, EE, LT, LV, NL, BE, FR, DE).
3. **Add a price source** — Stromligning (recommended) or Nordpool.
4. **Add weather sensors** (optional but recommended) to improve ML accuracy.
5. **Let it learn** — accuracy improves as it accumulates history and self-corrects.

## Documentation

- **[Architecture](architecture.md)** — System overview, data sources, data flow
- **[ML Documentation](ml_documentation.md)** — Model, 17-feature vector, confidence
- **[Self-Learning](self_learning.md)** — Self-learning loop, bias correction
- **[Persistence](persistence.md)** — SQLite storage schema and migrations
- **[Stromligning Integration](stromligning_integration.md)** — Price-source priority
- **[Using Existing Sensors](using_existing_sensors.md)** — Sensor wiring reference

See the [documentation index](index.md) for the full table of contents.
