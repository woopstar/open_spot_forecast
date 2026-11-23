"""The fallback price sensor for days the day-ahead APIs lack (#270)."""

from collections.abc import Iterator
from datetime import date, datetime, timedelta
from typing import Any
from unittest.mock import MagicMock, Mock, patch
from zoneinfo import ZoneInfo

import pytest

from homeassistant.util import dt as dt_util

from custom_components.open_spot_forecast.config_flow import (
    OpenSpotForecastConfigFlow,
    OpenSpotForecastOptionsFlow,
)
from custom_components.open_spot_forecast.const import (
    CONF_FALLBACK_PRICE_SENSOR,
    CONF_REGION,
)
from custom_components.open_spot_forecast.price_series import known_prices
from custom_components.open_spot_forecast.sensor_entities import SensorEntities
from custom_components.open_spot_forecast.sensor_reader import SensorReader
from custom_components.open_spot_forecast.spot_prices import (
    fill_missing_days,
    ml_price_inputs,
)

AMS = ZoneInfo("Europe/Amsterdam")
TODAY = date(2026, 11, 20)
NOW = datetime(2026, 11, 20, 19, 0, tzinfo=AMS)
SENSOR = "sensor.nordpool_kwh_nl_eur_3_10_0"


@pytest.fixture(autouse=True)
def amsterdam() -> Iterator[None]:
    """Run at 19:00 in Amsterdam on 20 Nov 2026."""
    previous = dt_util.get_default_time_zone()
    dt_util.set_default_time_zone(AMS)
    with patch.object(dt_util, "now", return_value=NOW):
        yield
    dt_util.set_default_time_zone(previous)


def _hourly(day: date, price: float) -> list[dict[str, Any]]:
    """Nordpool's raw items for a day: hourly, with datetime start and end."""
    midnight = datetime(day.year, day.month, day.day, tzinfo=AMS)
    return [
        {
            "start": midnight + timedelta(hours=hour),
            "end": midnight + timedelta(hours=hour + 1),
            "value": price + hour / 100,
        }
        for hour in range(24)
    ]


def _reader(attributes: dict[str, Any] | None) -> SensorReader:
    hass = Mock()
    state = MagicMock()
    state.state = "0.1"
    state.attributes = attributes
    hass.states.get.side_effect = lambda entity_id: (
        state if attributes is not None and entity_id == SENSOR else None
    )
    return SensorReader(hass)


# --- SensorReader.read_price_sensor ---------------------------------------------


def test_nordpool_raw_today_and_tomorrow_are_read_per_slot() -> None:
    reader = _reader(
        {
            "currency": "EUR",
            "unit": "kWh",
            "raw_today": _hourly(TODAY, 0.10),
            "raw_tomorrow": _hourly(TODAY + timedelta(days=1), 0.20),
            "today": [0.0] * 24,
        }
    )

    data = reader.read_price_sensor(SENSOR)

    assert data["currency"] == "EUR"
    assert data["day"] == TODAY
    assert len(data["today"]) == 96
    assert data["today"][:5] == pytest.approx([0.10] * 4 + [0.11])
    assert data["tomorrow"][-1] == pytest.approx(0.43)
    assert len(data["raw_tomorrow"]) == 96
    # Predictions start at the end of tomorrow's last slot
    _prices, known_end = ml_price_inputs(data)
    assert known_end == datetime(2026, 11, 22, 0, 0, tzinfo=AMS)


@pytest.mark.parametrize(
    ("attributes", "expected"),
    [
        ({"unit": "MWh"}, 0.1),
        ({"unit": "Wh"}, 100_000.0),
        ({"unit": "kWh", "price_in_cents": True}, 1.0),
        ({"unit_of_measurement": "EUR/MWh"}, 0.1),
        ({}, 100.0),
    ],
)
def test_the_sensor_unit_is_converted_to_currency_per_kwh(
    attributes: dict[str, Any], expected: float
) -> None:
    reader = _reader({**attributes, "raw_today": _hourly(TODAY, 100.0)})

    assert reader.read_price_sensor(SENSOR)["today"][0] == pytest.approx(expected)


def test_a_prices_list_is_read_without_nordpool_attributes() -> None:
    items = [
        {"start": item["start"].isoformat(), "price": 0.3} for item in _hourly(TODAY, 0)
    ]
    data = _reader({"prices": items}).read_price_sensor(SENSOR)

    assert known_prices(data["today"]) == pytest.approx([0.3] * 96)
    assert data["tomorrow"] == []
    assert data["currency"] is None


@pytest.mark.parametrize(
    "attributes",
    [
        None,
        {"unit": "GJ", "raw_today": _hourly(TODAY, 0.1)},
        # All zero: a failing source, not prices
        {"raw_today": [{**item, "value": 0.0} for item in _hourly(TODAY, 0.0)]},
    ],
)
def test_no_usable_prices_read_as_empty_days(attributes: dict[str, Any] | None) -> None:
    data = _reader(attributes).read_price_sensor(SENSOR)

    assert data["today"] == []
    assert data["raw_today"] == []
    assert data["tomorrow"] == []


# --- fill_missing_days ----------------------------------------------------------


def _spot(
    today: list[float], tomorrow: list[float], day: date = TODAY
) -> dict[str, Any]:
    return {
        "today": today,
        "tomorrow": tomorrow,
        "raw_today": [{"start": "t"}] if today else [],
        "raw_tomorrow": [{"start": "m"}] if tomorrow else [],
        "day": day,
    }


def test_only_the_days_the_apis_lack_are_filled() -> None:
    spot = _spot([], [0.5] * 96)

    assert fill_missing_days(spot, _spot([0.1] * 96, [0.2] * 96)) == ["today"]

    assert spot["today"] == [0.1] * 96
    assert spot["raw_today"] == [{"start": "t"}]
    assert spot["tomorrow"] == [0.5] * 96


def test_a_fallback_for_another_day_is_not_used() -> None:
    spot = _spot([], [])

    assert (
        fill_missing_days(spot, _spot([0.1] * 96, [], TODAY - timedelta(days=1))) == []
    )
    assert spot["today"] == []


# --- Configuration --------------------------------------------------------------


def test_the_sensor_is_read_from_the_options_first() -> None:
    entry = MagicMock()
    entry.data = {CONF_FALLBACK_PRICE_SENSOR: "sensor.setup"}
    entry.options = {CONF_FALLBACK_PRICE_SENSOR: SENSOR}
    assert SensorEntities.from_entry(entry).fallback_price == SENSOR

    entry.options = {}
    entry.data = {}
    assert SensorEntities.from_entry(entry).fallback_price is None


def _key(schema: Any) -> Any:
    return next(k for k in schema if str(k) == CONF_FALLBACK_PRICE_SENSOR)


@pytest.mark.asyncio
async def test_both_flows_offer_the_sensor_without_a_default() -> None:
    flow: Any = OpenSpotForecastConfigFlow()
    flow.hass = Mock()
    flow.async_show_form = Mock(return_value={})
    await flow.async_step_sensors()
    key = _key(flow.async_show_form.call_args.kwargs["data_schema"].schema)
    assert key.description == {"suggested_value": None}

    entry = MagicMock()
    entry.data = {CONF_REGION: "NL"}
    entry.options = {CONF_FALLBACK_PRICE_SENSOR: SENSOR}
    options: Any = OpenSpotForecastOptionsFlow()
    options.hass = Mock()
    options.hass.config_entries.async_get_known_entry.return_value = entry
    options.handler = "entry"
    options.async_show_form = Mock(return_value={})
    await options.async_step_init()
    key = _key(options.async_show_form.call_args.kwargs["data_schema"].schema)
    assert key.description == {"suggested_value": SENSOR}
