"""Surcharge, unit and hourly averaging of every exposed price (#39).

``PriceOutput`` turns the raw spot price (currency/kWh excl. VAT) into what
the user pays, ``(spot + surcharge) × (1 + VAT)`` in the configured unit, and
optionally averages the four 15-minute prices of each local hour.
"""

from datetime import UTC, date, datetime, timedelta
from typing import Any
from unittest.mock import MagicMock, Mock, patch
from zoneinfo import ZoneInfo

import pytest

from custom_components.open_spot_forecast.config_flow import (
    OpenSpotForecastOptionsFlow,
)
from custom_components.open_spot_forecast.price_output import (
    PriceOutput,
    apply_price_components,
    hourly_averages,
    hourly_forecast,
    slot_forecast,
)
from custom_components.open_spot_forecast.price_source import PriceSettings
from custom_components.open_spot_forecast.sensor import (
    MLPredictionSensor,
    SpotPriceSensor,
    TodayMaxSensor,
    TodayMeanSensor,
    TodayMinSensor,
)
from custom_components.open_spot_forecast.time_slots import (
    slot_start_in_day,
    slots_in_local_day,
)

CPH = ZoneInfo("Europe/Copenhagen")
DAY = date(2026, 9, 24)
SPRING_FORWARD = date(2026, 3, 29)
FALL_BACK = date(2026, 10, 25)


def _entry(data: dict[str, Any], options: dict[str, Any] | None = None) -> MagicMock:
    entry = MagicMock()
    entry.entry_id = "test"
    entry.data = data
    entry.options = options or {}
    return entry


def _predictions(day: date, first: int = 0) -> list[dict[str, Any]]:
    """Return the model's predictions for a local day: price = slot index."""
    count = slots_in_local_day(day)
    return [
        {
            "start": slot_start_in_day(day, index).isoformat(),
            "end": slot_start_in_day(day, index + 1).isoformat(),
            "price": float(index),
            "confidence": 0.8 if index % 2 else 0.6,
        }
        for index in range(first, count)
    ]


# --- The formula ---------------------------------------------------------------------


def test_total_is_spot_plus_surcharge_then_vat() -> None:
    assert apply_price_components(1.0, 0.2, 0.25) == pytest.approx(1.5)
    # A negative spot price stays negative when the surcharge is smaller
    assert apply_price_components(-0.5, 0.1, 0.25) == pytest.approx(-0.5)


def test_convert_applies_the_components_once_and_rounds() -> None:
    output = PriceOutput(vat=0.25, surcharge=0.1, precision=3)

    assert output.convert(1.0) == pytest.approx(1.375)
    assert output.convert(0.12345) == pytest.approx(0.279)


def test_defaults_add_only_vat() -> None:
    """An install that never set the options sees the prices as before."""
    assert PriceOutput().convert(1.0) == pytest.approx(1.25)
    assert PriceOutput().interval_minutes == 15
    assert PriceOutput().unit("DKK") == "DKK/kWh"


@pytest.mark.parametrize(
    ("price_type", "expected"),
    [("kWh", 1.25), ("MWh", 1250.0), ("Wh", 0.00125), ("unknown", 1.25)],
)
def test_the_unit_converts_before_the_surcharge(
    price_type: str, expected: float
) -> None:
    output = PriceOutput(vat=0.25, price_type=price_type, precision=6)

    assert output.convert(1.0) == pytest.approx(expected)
    # The surcharge is per configured unit: 10 DKK/MWh on top of 1000 DKK/MWh
    with_surcharge = PriceOutput(vat=0.0, surcharge=10.0, price_type="MWh")
    assert with_surcharge.convert(1.0) == pytest.approx(1010.0)


# --- Hourly averages -----------------------------------------------------------------


@pytest.mark.usefixtures("copenhagen_time_zone")
@pytest.mark.parametrize(
    ("day", "hours"), [(DAY, 24), (SPRING_FORWARD, 23), (FALL_BACK, 25)]
)
def test_one_value_per_local_hour(day: date, hours: int) -> None:
    prices = [float(index) for index in range(slots_in_local_day(day))]

    averages = hourly_averages(prices)

    assert len(averages) == hours
    assert averages[0] == pytest.approx(1.5)
    assert averages[-1] == pytest.approx(len(prices) - 2.5)


def test_hourly_averages_skip_missing_slots() -> None:
    prices: list[float | None] = [1.0, None, 3.0, None, None, None, None, None, 5.0]

    assert hourly_averages(prices) == [pytest.approx(2.0), None, pytest.approx(5.0)]


@pytest.mark.usefixtures("copenhagen_time_zone")
def test_price_at_uses_the_current_hour_mean_on_a_dst_day() -> None:
    prices = [float(index) for index in range(slots_in_local_day(FALL_BACK))]
    output = PriceOutput(vat=0.0, hourly_average=True)
    # The second 02:00-03:00 (CET) is the fourth hour of the day: slots 12-15
    moment = datetime(2026, 10, 25, 1, 20, tzinfo=UTC)

    assert output.price_at(prices, moment) == pytest.approx(13.5)
    assert PriceOutput(vat=0.0).price_at(prices, moment) == pytest.approx(13.0)
    assert output.price_at(prices[:8], moment) is None


# --- The forecast --------------------------------------------------------------------


@pytest.mark.usefixtures("copenhagen_time_zone")
@pytest.mark.parametrize(
    ("day", "hours"), [(DAY, 24), (SPRING_FORWARD, 23), (FALL_BACK, 25)]
)
def test_hourly_forecast_has_one_entry_per_local_hour(day: date, hours: int) -> None:
    forecast = hourly_forecast(_predictions(day))

    assert len(forecast) == hours
    starts = [datetime.fromisoformat(entry["start"]) for entry in forecast]
    # Distinct instants, an hour apart, even across the DST change
    assert all((b - a).total_seconds() == 3600 for a, b in zip(starts, starts[1:]))
    assert forecast[0]["price"] == pytest.approx(1.5)
    assert forecast[0]["confidence"] == pytest.approx(0.7)
    assert datetime.fromisoformat(forecast[0]["end"]) == starts[0] + timedelta(hours=1)


@pytest.mark.usefixtures("copenhagen_time_zone")
def test_repeated_hour_keeps_both_offsets() -> None:
    forecast = hourly_forecast(_predictions(FALL_BACK))

    assert forecast[2]["start"] == "2026-10-25T02:00:00+02:00"
    assert forecast[3]["start"] == "2026-10-25T02:00:00+01:00"


@pytest.mark.usefixtures("copenhagen_time_zone")
def test_a_partial_first_hour_averages_its_predicted_slots() -> None:
    # Predictions from 10:30: the 10:00 hour has two predicted slots
    forecast = hourly_forecast(_predictions(DAY, first=42))

    assert forecast[0]["start"] == "2026-09-24T10:00:00+02:00"
    assert forecast[0]["price"] == pytest.approx(42.5)


def test_forecast_skips_unusable_predictions() -> None:
    predictions: list[dict[str, Any]] = [
        {"start": "2026-09-24T08:00:00+00:00", "price": 1.0},
        {"start": "2026-09-24T08:15:00+00:00", "price": None, "confidence": 0.9},
        {"start": "not-a-date", "price": 5.0},
        {"start": "2026-09-24T08:30:00+00:00", "price": 3.0, "confidence": 0.5},
    ]

    slots = slot_forecast(predictions)
    hours = hourly_forecast(predictions)

    assert [slot["price"] for slot in slots] == [1.0, 3.0]
    # A prediction without an end covers one slot
    assert datetime.fromisoformat(slots[0]["end"]) == datetime(
        2026, 9, 24, 8, 15, tzinfo=UTC
    )
    assert len(hours) == 1
    assert hours[0]["price"] == pytest.approx(2.0)
    assert hours[0]["confidence"] == pytest.approx(0.5)
    assert hourly_forecast(predictions[:1])[0]["confidence"] is None


def test_forecast_converts_every_price() -> None:
    predictions = [
        {"start": "2026-09-24T08:00:00+00:00", "price": 1.0, "confidence": 0.8},
        {"start": "2026-09-24T08:15:00+00:00", "price": 2.0, "confidence": 0.8},
    ]
    output = PriceOutput(vat=0.25, surcharge=0.2)

    assert [e["price"] for e in output.forecast(predictions)] == [
        pytest.approx(1.5),
        pytest.approx(2.75),
    ]
    hourly = PriceOutput(vat=0.25, surcharge=0.2, hourly_average=True)
    assert [e["price"] for e in hourly.forecast(predictions)] == [pytest.approx(2.125)]


# --- Settings ------------------------------------------------------------------------


def test_settings_read_the_output_options_over_the_data() -> None:
    settings = PriceSettings.from_entry(
        _entry(
            {"vat": 0.25, "price_type": "MWh", "precision": 2},
            {"vat": 0.19, "surcharge": 0.1, "hourly_average": True},
        )
    )

    assert settings.output == PriceOutput(
        vat=0.19, surcharge=0.1, price_type="MWh", precision=2, hourly_average=True
    )


def test_defaults_leave_existing_installs_unchanged() -> None:
    assert PriceSettings.from_entry(_entry({})).output == PriceOutput(
        vat=0.25, surcharge=0.0, price_type="kWh", precision=3, hourly_average=False
    )


@pytest.mark.asyncio
async def test_the_options_flow_offers_surcharge_and_hourly_average() -> None:
    entry = _entry({"region": "DK1"}, {"surcharge": 0.3})
    hass = Mock()
    hass.config_entries.async_get_known_entry.return_value = entry
    flow: Any = OpenSpotForecastOptionsFlow()
    flow.hass = hass
    flow.handler = "test"
    flow.async_show_form = Mock(return_value={"type": "show_form"})

    await flow.async_step_init()

    schema = flow.async_show_form.call_args.kwargs["data_schema"].schema
    keys = {str(key): key for key in schema}
    assert keys["surcharge"].default() == pytest.approx(0.3)
    assert keys["hourly_average"].default() is False
    assert schema[keys["surcharge"]]("0.5") == pytest.approx(0.5)


# --- Sensors -------------------------------------------------------------------------


def _ml_sensor(output: PriceOutput, predictions: list[dict[str, Any]]) -> Any:
    predictor = MagicMock()
    predictor.predictions = predictions
    predictor.get_prediction_stats.return_value = {"mean_confidence": 0.7}
    return MLPredictionSensor(
        Mock(), _entry({}), {"ml_predictor": predictor}, "DKK", output, 12
    )


@pytest.mark.usefixtures("copenhagen_time_zone")
def test_forecast_sensor_applies_the_surcharge_to_state_and_attributes() -> None:
    output = PriceOutput(vat=0.25, surcharge=1.0)
    sensor = _ml_sensor(output, _predictions(DAY))
    now = datetime(2026, 9, 24, 10, 20, tzinfo=CPH)

    with patch("homeassistant.util.dt.utcnow", return_value=now.astimezone(UTC)):
        state = sensor.native_value
        attrs = sensor.extra_state_attributes

    # Slot 41 (10:15-10:30): (41 + 1) × 1.25
    assert state == pytest.approx(52.5)
    assert len(attrs["predictions"]) == 12 * 4
    assert attrs["predictions"][41]["price"] == pytest.approx(52.5)
    assert attrs["surcharge"] == pytest.approx(1.0)
    assert attrs["hourly_average"] is False
    assert attrs["forecast_min"] == pytest.approx(1.25)
    assert attrs["forecast_max"] == pytest.approx(120.0)
    assert attrs["forecast_mean"] == pytest.approx((47.5 + 1) * 1.25)


@pytest.mark.usefixtures("copenhagen_time_zone")
def test_forecast_sensor_averages_per_hour() -> None:
    output = PriceOutput(vat=0.0, hourly_average=True)
    sensor = _ml_sensor(output, _predictions(DAY))
    now = datetime(2026, 9, 24, 10, 20, tzinfo=CPH)

    with patch("homeassistant.util.dt.utcnow", return_value=now.astimezone(UTC)):
        state = sensor.native_value
        attrs = sensor.extra_state_attributes

    # The 10:00 hour: slots 40-43
    assert state == pytest.approx(41.5)
    assert attrs["state_slot_start"] == "2026-09-24T10:00:00+02:00"
    # The window counts hours: 12 hourly entries
    assert len(attrs["predictions"]) == 12
    assert attrs["predictions"][1]["price"] == pytest.approx(5.5)
    assert attrs["forecast_min"] == pytest.approx(1.5)
    assert attrs["forecast_max"] == pytest.approx(93.5)


@pytest.mark.usefixtures("copenhagen_time_zone")
def test_dayahead_price_sensors_apply_the_surcharge() -> None:
    output = PriceOutput(vat=0.25, surcharge=0.2)
    today = [float(index) / 10 for index in range(96)]
    api_data = {"price_source": "dayahead", "prices_today": today}
    now = datetime(2026, 9, 24, 10, 20, tzinfo=CPH)
    current = SpotPriceSensor(Mock(), _entry({}), api_data, "DK1", "DKK", output)

    with patch("homeassistant.util.dt.now", return_value=now):
        assert current.native_value == pytest.approx((4.1 + 0.2) * 1.25)
    minimum = TodayMinSensor(Mock(), _entry({}), api_data, "DKK", output)
    assert minimum.native_value == pytest.approx(0.25)
    attrs = current.extra_state_attributes
    assert attrs["today_prices"][1] == pytest.approx(0.375)
    assert attrs["surcharge"] == pytest.approx(0.2)


@pytest.mark.usefixtures("copenhagen_time_zone")
def test_hourly_price_sensors_use_hour_means() -> None:
    """Stromligning's consumer prices (excl. VAT, #107) get the hour mean, then
    the surcharge and VAT like every price."""
    output = PriceOutput(vat=0.25, surcharge=0.2, hourly_average=True)
    # Slot prices alternate 0/4 in the first hour, then 10 for the rest
    today = [0.0, 4.0, 0.0, 4.0] + [10.0] * 92
    api_data = {"stromligning_data": {"current_price": 4.0, "today": today}}
    now = datetime(2026, 9, 24, 0, 20, tzinfo=CPH)
    current = SpotPriceSensor(Mock(), _entry({}), api_data, "DK1", "DKK", output)

    with patch("homeassistant.util.dt.now", return_value=now):
        assert current.native_value == pytest.approx((2.0 + 0.2) * 1.25)
    assert TodayMinSensor(Mock(), _entry({}), api_data, "DKK", output).native_value == (
        pytest.approx((2.0 + 0.2) * 1.25)
    )
    assert TodayMaxSensor(Mock(), _entry({}), api_data, "DKK", output).native_value == (
        pytest.approx((10.0 + 0.2) * 1.25)
    )
    mean = TodayMeanSensor(Mock(), _entry({}), api_data, "DKK", output)
    hour_mean = (2.0 + 23 * 10.0) / 24
    assert mean.native_value == pytest.approx(round((hour_mean + 0.2) * 1.25, 3))
    attrs = current.extra_state_attributes
    assert len(attrs["today_prices"]) == 24
    assert attrs["hourly_average"] is True
    assert attrs["surcharge"] == pytest.approx(0.2)
