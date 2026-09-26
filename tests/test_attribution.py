"""Data-source attribution on the entities (#41)."""

from typing import Any
from unittest.mock import MagicMock, Mock

import pytest

from custom_components.open_spot_forecast.accuracy_sensor import (
    build_lead_time_accuracy_sensors,
)
from custom_components.open_spot_forecast.attribution import (
    license_summary,
    model_attribution,
    price_attribution,
)
from custom_components.open_spot_forecast.binary_sensor import (
    MLModelTrainedSensor,
    TomorrowAvailableSensor,
)
from custom_components.open_spot_forecast.sensor import (
    LearningMetricsSensor,
    MLPredictionSensor,
    PredictionConfidenceSensor,
    SpotPriceSensor,
    TodayMinSensor,
    TomorrowMeanSensor,
)

SMARD = (
    "CC BY 4.0 (creativecommons.org/licenses/by/4.0) from Bundesnetzagentur | SMARD.de"
)
EPEX = (
    "The data provided herein is for private and internal use only. The "
    "utilization of any data ... EPEX SPOT SE."
)
PRICES = "Prices: energy-charts.info (CC BY 4.0, Bundesnetzagentur | SMARD.de)"
WEATHER = "Weather: Open-Meteo.com (CC BY 4.0)"
PROGNOSES = "Prognoses: Nord Pool"


def _dayahead(**extra: Any) -> dict[str, Any]:
    return {"price_source": "dayahead", "price_license": SMARD} | extra


@pytest.mark.parametrize(
    ("license_info", "summary"),
    [
        (SMARD, "CC BY 4.0, Bundesnetzagentur | SMARD.de"),
        ("CC BY 4.0", "CC BY 4.0"),
        (EPEX, "private and internal use only"),
        ("Some other terms", "Some other terms"),
        (None, None),
        ("", None),
    ],
)
def test_license_summary(license_info: str | None, summary: str | None) -> None:
    assert license_summary(license_info) == summary


def test_price_attribution_follows_the_price_source() -> None:
    assert price_attribution({"price_source": "stromligning"}) is None
    assert price_attribution({"price_source": "dayahead"}) == (
        "Prices: energy-charts.info"
    )
    assert price_attribution(_dayahead()) == PRICES
    assert price_attribution(_dayahead(entsoe_fallback=True)) == (
        f"{PRICES} / ENTSO-E Transparency Platform"
    )
    assert price_attribution(_dayahead(price_license=EPEX)) == (
        "Prices: energy-charts.info (private and internal use only)"
    )


def test_model_attribution_credits_every_source_the_model_learns_from() -> None:
    model = Mock()

    assert model_attribution(_dayahead()) is None
    assert model_attribution(_dayahead(ml_predictor=model, zone_weather=True)) == (
        f"{PRICES} · {WEATHER} · {PROGNOSES}"
    )
    assert model_attribution({"ml_predictor": model, "zone_weather": False}) == (
        PROGNOSES
    )


def test_entities_show_the_attribution() -> None:
    hass, entry = Mock(), MagicMock()
    entry.entry_id = "test"
    api_data = _dayahead(ml_predictor=Mock(), zone_weather=True)
    price_entities = [
        SpotPriceSensor(hass, entry, api_data, "DK1", "DKK", 0.25, 3, "kWh"),
        TodayMinSensor(hass, entry, api_data, "DKK", 0.25, 3, "kWh"),
        TomorrowMeanSensor(hass, entry, api_data, "DKK", 0.25, 3, "kWh"),
        TomorrowAvailableSensor(hass, entry, api_data),
    ]
    model_entities = [
        MLPredictionSensor(hass, entry, api_data, "DKK", 0.25, 3, "kWh"),
        PredictionConfidenceSensor(hass, entry, api_data),
        LearningMetricsSensor(hass, entry, api_data),
        MLModelTrainedSensor(hass, entry, api_data),
        *build_lead_time_accuracy_sensors(hass, entry, api_data, "DKK", 3),
    ]

    assert {entity.attribution for entity in price_entities} == {PRICES}
    assert {entity.attribution for entity in model_entities} == {
        f"{PRICES} · {WEATHER} · {PROGNOSES}"
    }
    # Stromligning's prices are credited by the Stromligning integration
    api_data["price_source"] = "stromligning"
    assert price_entities[0].attribution is None
