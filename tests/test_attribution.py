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
from custom_components.open_spot_forecast.price_output import PriceOutput
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
UMM = "Outages: Nord Pool UMM"
ENTSOE_OUTAGES = "Outages: ENTSO-E Transparency Platform"


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
    # With an ENTSO-E key the model also learns from ENTSO-E's load forecast (#30)
    assert model_attribution({"ml_predictor": model, "entsoe_load": True}) == (
        f"{PROGNOSES} · Load forecast: ENTSO-E Transparency Platform"
    )
    # Outages: the active source's credit, Nord Pool's UMMs (#123) or the
    # ENTSO-E Transparency Platform's documents (#138)
    assert model_attribution({"ml_predictor": model, "outages": UMM}) == (
        f"{PROGNOSES} · Outages: Nord Pool UMM"
    )
    assert model_attribution({"ml_predictor": model, "outages": ENTSOE_OUTAGES}) == (
        f"{PROGNOSES} · Outages: ENTSO-E Transparency Platform"
    )
    assert model_attribution({"ml_predictor": model, "outages": None}) == PROGNOSES
    # The cross-border model also learns from the neighbours' prices (#29)
    assert model_attribution({"ml_predictor": model, "cross_border": True}) == (
        f"{PROGNOSES} · Neighbour prices: energy-charts.info"
    )


def test_entities_show_the_attribution() -> None:
    hass, entry = Mock(), MagicMock()
    entry.entry_id = "test"
    api_data = _dayahead(ml_predictor=Mock(), zone_weather=True)
    price_entities = [
        SpotPriceSensor(hass, entry, api_data, "DK1", "DKK", PriceOutput()),
        TodayMinSensor(hass, entry, api_data, "DKK", PriceOutput()),
        TomorrowMeanSensor(hass, entry, api_data, "DKK", PriceOutput()),
        TomorrowAvailableSensor(hass, entry, api_data),
    ]
    model_entities = [
        MLPredictionSensor(hass, entry, api_data, "DKK", PriceOutput()),
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


def test_the_model_credits_its_day_ahead_history_with_stromligning() -> None:
    """Stromligning shows the prices; the training history is day-ahead (#24)."""
    api_data = {
        "price_source": "stromligning",
        "price_license": SMARD,
        "history_prices": True,
        "ml_predictor": Mock(),
        "zone_weather": True,
    }

    assert price_attribution(api_data) is None
    assert model_attribution(api_data) == (
        "Price history: energy-charts.info (CC BY 4.0, Bundesnetzagentur | SMARD.de)"
        f" · {WEATHER} · {PROGNOSES}"
    )
