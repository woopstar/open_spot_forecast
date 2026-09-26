"""Data-source attribution for the entities (#41).

Home Assistant shows an entity's ``attribution`` in its more-info dialog.
OSF credits the external sources an entity's value is derived from, as they
are configured and used:

* **Prices** (day-ahead source, #27): energy-charts.info with the licence it
  reports for the zone (CC BY 4.0 from Bundesnetzagentur | SMARD.de for e.g.
  DK1 and DK2; private and internal use only for EPEX SPOT data in e.g. SE3),
  and the ENTSO-E Transparency Platform when its fallback is configured.
  Stromligning's prices come from another integration, which credits them.
* **Weather** (zone weather, #22): Open-Meteo.com, CC BY 4.0.
* **Market prognoses**: Nord Pool.

Price entities credit the prices; the model's entities credit everything
the model learns from.
"""

from __future__ import annotations

from typing import Any

from .const import PRICE_SOURCE_DAYAHEAD

OPEN_METEO_ATTRIBUTION = "Weather: Open-Meteo.com (CC BY 4.0)"
NORD_POOL_ATTRIBUTION = "Prognoses: Nord Pool"
_SEPARATOR = " · "


def license_summary(license_info: str | None) -> str | None:
    """Return a short form of an energy-charts ``license_info`` text."""
    if not license_info:
        return None
    if license_info.startswith("CC BY 4.0"):
        provider = (
            license_info.split(" from ", 1)[1] if " from " in license_info else ""
        )
        return f"CC BY 4.0, {provider}" if provider else "CC BY 4.0"
    if "private and internal use only" in license_info:
        return "private and internal use only"
    return license_info


def price_attribution(api_data: dict[str, Any]) -> str | None:
    """Return the credit for the displayed prices, or None (Stromligning)."""
    if api_data.get("price_source") != PRICE_SOURCE_DAYAHEAD:
        return None
    summary = license_summary(api_data.get("price_license"))
    sources = [f"energy-charts.info ({summary})" if summary else "energy-charts.info"]
    if api_data.get("entsoe_fallback"):
        sources.append("ENTSO-E Transparency Platform")
    return "Prices: " + " / ".join(sources)


def model_attribution(api_data: dict[str, Any]) -> str | None:
    """Return the credit for the model's outputs: prices, weather, prognoses."""
    if api_data.get("ml_predictor") is None:
        return None
    parts = [price_attribution(api_data)]
    if api_data.get("zone_weather"):
        parts.append(OPEN_METEO_ATTRIBUTION)
    parts.append(NORD_POOL_ATTRIBUTION)
    return _SEPARATOR.join(part for part in parts if part)


class PriceAttributionMixin:
    """Credit the price sources in an entity's ``attribution``."""

    api_data: dict[str, Any]

    @property
    def attribution(self) -> str | None:
        """Return the price sources' credit."""
        return price_attribution(self.api_data)


class ModelAttributionMixin:
    """Credit the model's data sources in an entity's ``attribution``."""

    api_data: dict[str, Any]

    @property
    def attribution(self) -> str | None:
        """Return the model's data sources' credit."""
        return model_attribution(self.api_data)
