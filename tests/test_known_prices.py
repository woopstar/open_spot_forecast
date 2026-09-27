"""``known_until`` and the merged actual-then-forecast series (#40)."""

from datetime import UTC, date, datetime, timedelta
from typing import Any
from unittest.mock import MagicMock, Mock, patch
from zoneinfo import ZoneInfo

import pytest

from homeassistant.config_entries import ConfigEntryState

from custom_components.open_spot_forecast.config_flow import (
    OpenSpotForecastOptionsFlow,
)
from custom_components.open_spot_forecast.const import DOMAIN
from custom_components.open_spot_forecast.forecast_attributes import (
    compact_forecast,
    fit_compact,
)
from custom_components.open_spot_forecast.price_output import (
    PriceOutput,
    hourly_forecast,
    slot_forecast,
)
from custom_components.open_spot_forecast.sensor import MLPredictionSensor
from custom_components.open_spot_forecast.sensor_reader import SensorReader
from custom_components.open_spot_forecast.services import (
    GET_FORECAST_SCHEMA,
    _async_get_forecast,
)
from custom_components.open_spot_forecast.spot_prices import (
    known_until,
    with_known_prices,
)
from custom_components.open_spot_forecast.time_slots import (
    slot_start_in_day,
    slots_in_local_day,
)

pytestmark = pytest.mark.usefixtures("copenhagen_time_zone")

CPH = ZoneInfo("Europe/Copenhagen")
DAY = date(2026, 9, 24)
NOW = datetime(2026, 9, 24, 10, 20, tzinfo=CPH)
FALL_BACK = date(2026, 10, 25)


def _spot_data(day: date = DAY, tomorrow: bool = True) -> dict[str, Any]:
    """Return read_spot_prices()-shaped data: today (and tomorrow) known."""
    days = [day, day + timedelta(days=1)] if tomorrow else [day]
    lists = [
        [1.0 + index / 100 for index in range(slots_in_local_day(d))] for d in days
    ]
    raw = [
        [{"start": slot_start_in_day(d, i).isoformat()} for i in range(len(prices))]
        for d, prices in zip(days, lists)
    ]
    return {
        "today": lists[0],
        "tomorrow": lists[1] if tomorrow else [],
        "raw_today": raw[0],
        "raw_tomorrow": raw[1] if tomorrow else [],
        "day": day,
    }


def _predictions(first: date, days: int = 7) -> list[dict[str, Any]]:
    """Return predictions (price 50 + n) for ``days`` local days from ``first``."""
    predictions: list[dict[str, Any]] = []
    for offset in range(days):
        day = first + timedelta(days=offset)
        for index in range(slots_in_local_day(day)):
            predictions.append(
                {
                    "start": slot_start_in_day(day, index).isoformat(),
                    "end": slot_start_in_day(day, index + 1).isoformat(),
                    "price": 50.0 + len(predictions),
                    "confidence": 0.8,
                }
            )
    return predictions


def _assert_continuous(entries: list[dict[str, Any]]) -> None:
    starts = [datetime.fromisoformat(entry["start"]) for entry in entries]
    ends = [datetime.fromisoformat(entry["end"]) for entry in entries]
    assert all(end == start for end, start in zip(ends, starts[1:]))


# --- known_until ---------------------------------------------------------------------


def test_known_until_is_the_end_of_the_last_confirmed_slot() -> None:
    assert known_until(_spot_data()) == datetime(2026, 9, 25, 22, 0, tzinfo=UTC)
    assert known_until(_spot_data(tomorrow=False)) == datetime(
        2026, 9, 24, 22, 0, tzinfo=UTC
    )
    assert known_until(None) is None
    assert known_until({"today": []}) is None


def test_the_spot_data_knows_its_day() -> None:
    hass = Mock()
    hass.states.get.return_value = Mock(
        state="1.0",
        attributes={"prices": [{"start": "2026-09-24T10:00:00+02:00", "price": 1.0}]},
    )

    with patch("homeassistant.util.dt.now", return_value=NOW):
        spot = SensorReader(hass).read_spot_prices("sensor.spot", None)

    assert spot["day"] == DAY


# --- The merged series ---------------------------------------------------------------


def test_actual_prices_come_first_then_the_predictions() -> None:
    # Predictions made yesterday start at today's midnight, so they overlap
    # every confirmed slot of today and tomorrow
    merged = with_known_prices(_spot_data(), _predictions(DAY), NOW)

    sources = [entry["source"] for entry in merged]
    known = sources.count("actual")
    # From the 10:15 slot (41) to the end of tomorrow
    assert known == 96 - 41 + 96
    assert sources == ["actual"] * known + ["predicted"] * (len(merged) - known)
    assert merged[0]["start"] == "2026-09-24T10:15:00+02:00"
    assert merged[0]["price"] == pytest.approx(1.41)
    assert merged[0]["confidence"] == pytest.approx(1.0)
    # The first prediction starts where the confirmed prices end
    assert merged[known - 1]["end"] == "2026-09-26T00:00:00+02:00"
    assert merged[known]["start"] == "2026-09-26T00:00:00+02:00"
    _assert_continuous(merged)
    starts = [entry["start"] for entry in merged]
    assert len(starts) == len(set(starts))


def test_the_series_is_continuous_across_dst() -> None:
    since = datetime(2026, 10, 25, 0, 0, tzinfo=CPH)

    merged = with_known_prices(_spot_data(FALL_BACK), _predictions(FALL_BACK), since)

    assert [e["source"] for e in merged].count("actual") == 100 + 96
    _assert_continuous(merged)


def test_without_confirmed_prices_the_predictions_start_at_since() -> None:
    predictions = _predictions(DAY)

    merged = with_known_prices(None, predictions, NOW)

    assert merged[0]["start"] == "2026-09-24T10:15:00+02:00"
    assert {entry["source"] for entry in merged} == {"predicted"}
    assert len(merged) == len(predictions) - 41


def test_a_missing_day_means_today() -> None:
    spot = _spot_data(tomorrow=False)
    del spot["day"]

    with patch("homeassistant.util.dt.now", return_value=NOW):
        merged = with_known_prices(spot, [], NOW)

    assert merged[0]["start"] == "2026-09-24T10:15:00+02:00"


def test_a_missing_slot_is_not_invented() -> None:
    spot = _spot_data(tomorrow=False)
    spot["today"][50] = None

    merged = with_known_prices(spot, [], NOW)

    assert "2026-09-24T12:30:00+02:00" not in [entry["start"] for entry in merged]


def test_the_source_survives_the_output() -> None:
    merged = with_known_prices(
        _spot_data(tomorrow=False), _predictions(DAY + timedelta(days=1), 1), NOW
    )

    slots = slot_forecast(merged)
    hours = hourly_forecast(merged)

    assert slots[0]["source"] == "actual"
    assert slots[-1]["source"] == "predicted"
    assert hours[0]["source"] == "actual"
    assert hours[-1]["source"] == "predicted"
    # An hour is only actual if all its slots are
    mixed = [dict(merged[-1], source="actual"), merged[-2]]
    assert hourly_forecast(mixed)[0]["source"] == "predicted"
    # Without a marker nothing is added
    assert "source" not in slot_forecast(_predictions(DAY, 1))[0]


def test_compact_counts_the_confirmed_entries() -> None:
    merged = with_known_prices(
        _spot_data(tomorrow=False), _predictions(DAY + timedelta(days=1), 1), NOW
    )
    entries = PriceOutput().forecast(merged)

    compact = compact_forecast(entries, "DKK/kWh", 15)

    assert compact["known_count"] == 96 - 41
    assert compact["c"][0] == 100
    assert "known_count" not in compact_forecast(
        PriceOutput().forecast(_predictions(DAY, 1)), "DKK/kWh", 15
    )
    attributes = {"predictions": compact}
    fit_compact(attributes, "predictions", budget=300)
    assert attributes["predictions"]["known_count"] == len(compact["s"])
    assert compact_forecast(entries[:5], "DKK/kWh", 15)["known_count"] == 5


# --- The sensor ----------------------------------------------------------------------


def _sensor(
    include_known: bool, attribute_format: str = "detailed"
) -> MLPredictionSensor:
    predictor = MagicMock()
    predictor.predictions = _predictions(DAY)
    predictor.get_prediction_stats.return_value = {"mean_confidence": 0.8}
    api_data = {"ml_predictor": predictor, "spot_data": _spot_data()}
    return MLPredictionSensor(
        Mock(),
        MagicMock(entry_id="test"),
        api_data,
        "DKK",
        PriceOutput(),
        72,
        attribute_format,
        include_known,
    )


def _attributes(sensor: MLPredictionSensor) -> dict[str, Any]:
    with patch("homeassistant.util.dt.utcnow", return_value=NOW.astimezone(UTC)):
        return sensor.extra_state_attributes


def test_the_sensor_shows_known_until() -> None:
    attributes = _attributes(_sensor(include_known=False))

    assert attributes["known_until"] == "2026-09-26T00:00:00+02:00"
    assert datetime.fromisoformat(attributes["known_until"]).tzinfo is not None
    sensor = _sensor(include_known=False)
    sensor.api_data["spot_data"] = None
    assert _attributes(sensor)["known_until"] is None


def test_by_default_the_attribute_is_unchanged() -> None:
    predictions = _attributes(_sensor(include_known=False))["predictions"]

    assert predictions[0]["start"] == "2026-09-24T00:00:00+02:00"
    assert "source" not in predictions[0]


def test_with_the_option_the_attribute_starts_with_known_prices() -> None:
    sensor = _sensor(include_known=True)

    predictions = _attributes(sensor)["predictions"]

    assert len(predictions) == 72 * 4
    assert predictions[0]["start"] == "2026-09-24T10:15:00+02:00"
    assert predictions[0]["price"] == pytest.approx(round(1.41 * 1.25, 3))
    assert predictions[0]["source"] == "actual"
    boundary = [entry["source"] for entry in predictions].index("predicted")
    assert predictions[boundary]["start"] == "2026-09-26T00:00:00+02:00"
    _assert_continuous(predictions)
    # The state is still the model's forecast
    with patch("homeassistant.util.dt.utcnow", return_value=NOW.astimezone(UTC)):
        assert sensor.native_value == pytest.approx(round((50 + 41) * 1.25, 3))


def test_compact_attribute_has_the_known_count() -> None:
    compact = _attributes(_sensor(True, "compact"))["predictions"]

    assert compact["known_count"] == 96 - 41 + 96


# --- The action and the option -------------------------------------------------------


def _hass(options: dict[str, Any]) -> Mock:
    entry = MagicMock()
    entry.entry_id = "entry"
    entry.domain = DOMAIN
    entry.options = options
    entry.data = {"currency": "DKK"}
    hass = Mock()
    hass.config_entries.async_entries.return_value = [entry]
    entry.state = ConfigEntryState.LOADED
    predictor = MagicMock()
    predictor.predictions = _predictions(DAY)
    hass.data = {
        DOMAIN: {"entry": {"ml_predictor": predictor, "spot_data": _spot_data()}}
    }
    return hass


async def _call(hass: Mock, **data: Any) -> Any:
    call = Mock()
    call.hass = hass
    call.data = GET_FORECAST_SCHEMA(data)
    with patch("homeassistant.util.dt.utcnow", return_value=NOW.astimezone(UTC)):
        return await _async_get_forecast(call)


@pytest.mark.asyncio
async def test_the_action_includes_known_prices_on_request() -> None:
    merged = await _call(_hass({}), include_known=True, hours=48)
    default = await _call(_hass({}), hours=48)
    from_option = await _call(_hass({"include_known_prices": True}), hours=48)
    overridden = await _call(
        _hass({"include_known_prices": True}), include_known=False, hours=48
    )

    assert merged["known_until"] == "2026-09-26T00:00:00+02:00"
    assert merged["forecast"][0]["source"] == "actual"
    assert merged["forecast"][0]["start"] == "2026-09-24T10:15:00+02:00"
    assert len(merged["forecast"]) == 48 * 4
    assert from_option == merged
    assert "source" not in default["forecast"][0]
    assert overridden == default


@pytest.mark.asyncio
async def test_the_action_merges_from_its_start() -> None:
    response = await _call(
        _hass({}), include_known=True, start="2026-09-24 00:00", hours=1
    )

    assert [entry["source"] for entry in response["forecast"]] == ["actual"] * 4
    assert response["forecast"][0]["price"] == pytest.approx(1.25)


@pytest.mark.asyncio
async def test_the_options_flow_offers_include_known_prices() -> None:
    entry = MagicMock()
    entry.data = {"region": "DK1"}
    entry.options = {}
    hass = Mock()
    hass.config_entries.async_get_known_entry.return_value = entry
    flow: Any = OpenSpotForecastOptionsFlow()
    flow.hass = hass
    flow.handler = "test"
    flow.async_show_form = Mock(return_value={"type": "show_form"})

    await flow.async_step_init()

    schema = flow.async_show_form.call_args.kwargs["data_schema"].schema
    keys = {str(key): key for key in schema}
    assert keys["include_known_prices"].default() is False
