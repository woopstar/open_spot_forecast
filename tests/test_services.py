"""The ``open_spot_forecast.get_forecast`` action (#37)."""

from datetime import UTC, date, datetime
from typing import Any
from unittest.mock import MagicMock, Mock, patch
from zoneinfo import ZoneInfo

import pytest
import voluptuous as vol

from homeassistant.config_entries import ConfigEntryState
from homeassistant.core import SupportsResponse
from homeassistant.exceptions import ServiceValidationError

from custom_components.open_spot_forecast import CONFIG_SCHEMA, async_setup
from custom_components.open_spot_forecast.const import DOMAIN
from custom_components.open_spot_forecast.price_output import PriceOutput
from custom_components.open_spot_forecast.sensor import MLPredictionSensor
from custom_components.open_spot_forecast.services import (
    GET_FORECAST_SCHEMA,
    SERVICE_GET_FORECAST,
    _async_get_forecast,
    forecast_response,
)
from custom_components.open_spot_forecast.time_slots import (
    slot_start_in_day,
    slots_in_local_day,
)

pytestmark = pytest.mark.usefixtures("copenhagen_time_zone")

CPH = ZoneInfo("Europe/Copenhagen")
DAY = date(2026, 9, 24)
NOW = datetime(2026, 9, 24, 10, 20, tzinfo=CPH)


def _predictions(days: int = 7) -> list[dict[str, Any]]:
    """Return 15-min predictions from local midnight of DAY: price = slot number."""
    predictions: list[dict[str, Any]] = []
    for offset in range(days):
        day = date.fromordinal(DAY.toordinal() + offset)
        for index in range(slots_in_local_day(day)):
            predictions.append(
                {
                    "start": slot_start_in_day(day, index).isoformat(),
                    "end": slot_start_in_day(day, index + 1).isoformat(),
                    "price": float(len(predictions)),
                    "confidence": 0.8,
                }
            )
    return predictions


def _hass(
    entry_state: ConfigEntryState = ConfigEntryState.LOADED,
    ml_predictor: Any = None,
    options: dict[str, Any] | None = None,
) -> Mock:
    """Return a mock Home Assistant with one Open Spot Forecast entry."""
    entry = MagicMock()
    entry.entry_id = "entry"
    entry.domain = DOMAIN
    entry.title = "Open Spot Forecast DK1"
    entry.state = entry_state
    entry.data = {"region": "DK1", "currency": "DKK"}
    entry.options = options or {}
    hass = Mock()
    hass.config_entries.async_get_entry.side_effect = lambda entry_id: (
        entry if entry_id == "entry" else None
    )
    hass.config_entries.async_entries.return_value = [entry]
    hass.data = {
        DOMAIN: {
            "entry": {
                "ml_predictor": ml_predictor,
                "spot_data": {
                    "today": [1.0] * 96,
                    "raw_today": [{"start": "2026-09-24T21:45:00+00:00"}],
                },
            }
        }
    }
    return hass


def _predictor(predictions: list[dict[str, Any]]) -> MagicMock:
    predictor = MagicMock()
    predictor.predictions = predictions
    predictor.get_prediction_stats.return_value = {"mean_confidence": 0.8}
    return predictor


async def _call(hass: Mock, **data: Any) -> Any:
    call = Mock()
    call.hass = hass
    call.data = GET_FORECAST_SCHEMA(data)
    with patch("homeassistant.util.dt.utcnow", return_value=NOW.astimezone(UTC)):
        return await _async_get_forecast(call)


# --- Registration --------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_action_is_registered_once_in_async_setup() -> None:
    hass = Mock()

    assert await async_setup(hass, {}) is True

    hass.services.async_register.assert_called_once()
    args, kwargs = hass.services.async_register.call_args
    assert args[:2] == (DOMAIN, SERVICE_GET_FORECAST)
    assert kwargs["supports_response"] is SupportsResponse.ONLY
    # Config entries only: YAML is logged and ignored
    assert CONFIG_SCHEMA({}) == {}


def test_the_schema_validates_the_fields() -> None:
    data: dict[str, Any] = GET_FORECAST_SCHEMA(
        {"start": "2026-09-24 10:00", "hours": "24", "hourly": "true"}
    )

    assert data["start"] == datetime(2026, 9, 24, 10, 0)
    assert data["hours"] == 24
    assert data["hourly"] is True
    with pytest.raises(vol.Invalid):
        GET_FORECAST_SCHEMA({"hours": 0})
    with pytest.raises(vol.Invalid):
        GET_FORECAST_SCHEMA({"start": "not a time"})


# --- The response --------------------------------------------------------------------


def test_the_whole_forecast_is_returned() -> None:
    predictions = _predictions()

    response = forecast_response(
        predictions, PriceOutput(), "DKK", None, datetime(2026, 9, 24, 0, tzinfo=CPH)
    )

    # 7 days of 15-min slots: far beyond the sensor's 72-hour attribute cap
    assert len(response["forecast"]) == len(predictions) == 7 * 96
    assert response["unit"] == "DKK/kWh"
    assert response["interval_minutes"] == 15
    assert response["known_until"] is None
    assert response["forecast"][0] == {
        "start": "2026-09-24T00:00:00+02:00",
        "end": "2026-09-24T00:15:00+02:00",
        "price": pytest.approx(0.0),
        "confidence": 0.8,
    }
    assert response["forecast"][1]["price"] == pytest.approx(1.25)


def test_start_and_hours_select_a_window() -> None:
    response = forecast_response(
        _predictions(), PriceOutput(vat=0.0), "DKK", None, NOW, hours=24
    )

    # From the slot containing 10:20 (10:15, slot 41), 24 hours of slots
    forecast = response["forecast"]
    assert len(forecast) == 96
    assert forecast[0]["start"] == "2026-09-24T10:15:00+02:00"
    assert forecast[0]["price"] == pytest.approx(41.0)
    assert forecast[-1]["start"] == "2026-09-25T10:00:00+02:00"


def test_hourly_averages_the_four_slots() -> None:
    output = PriceOutput(vat=0.0, hourly_average=True)

    response = forecast_response(_predictions(), output, "DKK", None, NOW, hours=3)

    forecast = response["forecast"]
    assert response["interval_minutes"] == 60
    assert [entry["start"] for entry in forecast] == [
        "2026-09-24T10:00:00+02:00",
        "2026-09-24T11:00:00+02:00",
        "2026-09-24T12:00:00+02:00",
    ]
    # Slots 40-43
    assert forecast[0]["price"] == pytest.approx(41.5)
    assert forecast[0]["end"] == "2026-09-24T11:00:00+02:00"


def test_known_until_is_local_and_timezone_aware() -> None:
    known_until = datetime(2026, 9, 24, 22, 0, tzinfo=UTC)

    response = forecast_response([], PriceOutput(), "DKK", known_until, NOW)

    assert response["known_until"] == "2026-09-25T00:00:00+02:00"
    assert response["forecast"] == []


# --- The action ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_action_returns_the_entry_forecast() -> None:
    hass = _hass(ml_predictor=_predictor(_predictions()), options={"surcharge": 1.0})

    response = await _call(hass, config_entry_id="entry", hours=1)

    assert response["known_until"] == "2026-09-25T00:00:00+02:00"
    assert response["unit"] == "DKK/kWh"
    # (41 + 1) × 1.25: the entry's surcharge and VAT
    assert [entry["price"] for entry in response["forecast"]] == pytest.approx(
        [52.5, 53.75, 55.0, 56.25]
    )


@pytest.mark.asyncio
async def test_the_action_and_the_sensor_agree() -> None:
    predictor = _predictor(_predictions())
    hass = _hass(ml_predictor=predictor, options={"surcharge": 0.5})
    sensor = MLPredictionSensor(
        hass,
        hass.config_entries.async_get_entry("entry"),
        hass.data[DOMAIN]["entry"],
        "DKK",
        PriceOutput(surcharge=0.5),
        12,
    )

    response = await _call(hass, hours=12, start="2026-09-24 00:00")

    with patch("homeassistant.util.dt.utcnow", return_value=NOW.astimezone(UTC)):
        attributes = sensor.extra_state_attributes["predictions"]
    assert [entry["price"] for entry in response["forecast"]] == [
        entry["price"] for entry in attributes
    ]


@pytest.mark.asyncio
async def test_hourly_overrides_the_entry_option() -> None:
    hass = _hass(ml_predictor=_predictor(_predictions()), options={"vat": 0.0})

    hourly = await _call(hass, hourly=True, hours=2)
    slots = await _call(
        _hass(
            ml_predictor=_predictor(_predictions()),
            options={"vat": 0.0, "hourly_average": True},
        ),
        hourly=False,
        hours=2,
    )

    assert [entry["price"] for entry in hourly["forecast"]] == pytest.approx(
        [41.5, 45.5]
    )
    assert len(slots["forecast"]) == 8


@pytest.mark.asyncio
async def test_an_unknown_entry_is_a_translated_validation_error() -> None:
    hass = _hass(ml_predictor=_predictor([]))

    with pytest.raises(ServiceValidationError) as err:
        await _call(hass, config_entry_id="missing")

    assert err.value.translation_key == "service_config_entry_not_found"


@pytest.mark.asyncio
async def test_an_unloaded_entry_is_a_translated_validation_error() -> None:
    hass = _hass(entry_state=ConfigEntryState.SETUP_RETRY)

    with pytest.raises(ServiceValidationError) as err:
        await _call(hass, config_entry_id="entry")

    assert err.value.translation_key == "service_config_entry_not_loaded"


@pytest.mark.asyncio
async def test_without_ml_there_is_no_forecast() -> None:
    hass = _hass(ml_predictor=None)

    with pytest.raises(ServiceValidationError) as err:
        await _call(hass, config_entry_id="entry")

    assert err.value.translation_domain == DOMAIN
    assert err.value.translation_key == "ml_prediction_disabled"
    assert err.value.translation_placeholders == {
        "entry_title": "Open Spot Forecast DK1"
    }
