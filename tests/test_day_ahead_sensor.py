"""The day-ahead prediction of the current slot, recorded as a sensor (#113).

The predictor caches, for the current and the coming slots, the stored
prediction made closest to a day ahead; the sensor's state is the current
slot's, so the recorder keeps the day-ahead forecast as a plain series.
"""

import sqlite3
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from homeassistant.components.sensor import SensorStateClass
from homeassistant.const import EntityCategory
from homeassistant.util import slugify as util_slugify

from custom_components.open_spot_forecast.const import (
    DAY_AHEAD_PREDICTION_HOURS,
    DOMAIN,
    UPDATE_SIGNAL,
    UPDATE_SIGNAL_FORECAST,
)
from custom_components.open_spot_forecast.day_ahead_sensor import (
    DayAheadPredictionSensor,
)
from custom_components.open_spot_forecast.ml.predictor import SpotPricePredictor
from custom_components.open_spot_forecast.price_output import PriceOutput
from custom_components.open_spot_forecast.tariffs import TariffSchedule
from custom_components.open_spot_forecast.time_slots import utc_slot_key
from custom_components.open_spot_forecast.updater import ForecastUpdater

pytestmark = pytest.mark.usefixtures("copenhagen_time_zone")

# 12:00 UTC is 14:00 in Copenhagen: a slot (and an hour) starts here
NOW = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)
QUARTER = timedelta(minutes=15)


@pytest.fixture
def predictor(tmp_path: Path) -> Iterator[SpotPricePredictor]:
    hass = Mock()
    hass.config.path.return_value = str(tmp_path / ".storage")
    predictor = SpotPricePredictor(hass, "DK1")
    yield predictor
    predictor.storage.close()


def _insert(
    predictor: SpotPricePredictor, slot: datetime, lead: float, price: float
) -> None:
    """Store a prediction for ``slot`` made ``lead`` hours before it, in local time."""
    local = slot.astimezone(predictor.tz)
    predictor.storage.insert_prediction(
        start=local.isoformat(),
        price=price,
        confidence=0.8,
        hour=local.hour,
        minute=local.minute,
        stored_at=(local - timedelta(hours=lead)).isoformat(),
    )


def _refresh(predictor: SpotPricePredictor, now: datetime = NOW) -> None:
    with patch("homeassistant.util.dt.utcnow", return_value=now):
        predictor.refresh_day_ahead_predictions()


# --- The predictor's cache -----------------------------------------------------------


def test_storage_returns_the_predictions_of_a_window(
    predictor: SpotPricePredictor,
) -> None:
    _insert(predictor, NOW - QUARTER, 24, 1.0)
    _insert(predictor, NOW, 30, 2.0)
    _insert(predictor, NOW, 20, 3.0)
    _insert(predictor, NOW + QUARTER, 24, 4.0)

    rows = predictor.storage.get_predictions_between(
        utc_slot_key(NOW), utc_slot_key(NOW + QUARTER)
    )

    # Local starts match the UTC window; the oldest prediction comes first
    assert [row["price"] for row in rows] == pytest.approx([2.0, 3.0])
    assert rows[0]["start"] == "2026-09-24T14:00:00+02:00"
    assert set(rows[0]) == {"start", "price", "stored_at"}


def test_the_cache_holds_each_coming_slots_day_ahead_prediction(
    predictor: SpotPricePredictor,
) -> None:
    _insert(predictor, NOW, 34, 1.0)
    _insert(predictor, NOW, 23, 2.0)
    _insert(predictor, NOW, 12, 3.0)
    _insert(predictor, NOW + QUARTER, 40, 4.0)
    # Past slots and slots beyond the window are left out
    _insert(predictor, NOW - QUARTER, 24, 5.0)
    _insert(predictor, NOW + timedelta(hours=DAY_AHEAD_PREDICTION_HOURS), 24, 6.0)

    # Refreshed in the middle of the current slot
    _refresh(predictor, NOW + timedelta(minutes=7))

    assert predictor.day_ahead_predictions == {
        "2026-09-24T12:00:00Z": pytest.approx(2.0),
        "2026-09-24T12:15:00Z": pytest.approx(4.0),
    }
    assert predictor.day_ahead_prediction(NOW) == pytest.approx(2.0)
    assert predictor.day_ahead_prediction(NOW + 2 * QUARTER) is None


def test_a_scored_slot_keeps_its_day_ahead_prediction(
    predictor: SpotPricePredictor,
) -> None:
    """Scoring deletes the stored predictions; the evaluation keeps the same one."""
    now = datetime.now(UTC).replace(second=0, microsecond=0)
    slot = now - timedelta(minutes=now.minute % 15)
    _insert(predictor, slot, 30, 1.0)
    _insert(predictor, slot, 22, 2.0)
    predictor.refresh_day_ahead_predictions()
    assert predictor.day_ahead_prediction(slot) == pytest.approx(2.0)

    local = slot.astimezone(predictor.tz)
    assert predictor.learn_from_actual_price(local.isoformat(), 2.5) is True
    # A restart after the slot was scored: nothing is stored for it any more
    predictor.refresh_day_ahead_predictions()

    assert predictor.day_ahead_predictions == {}
    assert predictor.day_ahead_prediction(slot) == pytest.approx(2.0)
    # Any moment of the slot, in any time zone
    assert predictor.day_ahead_prediction(
        local + timedelta(minutes=14)
    ) == pytest.approx(2.0)
    assert predictor.day_ahead_prediction(slot + QUARTER) is None
    assert predictor.day_ahead_prediction(slot - QUARTER) is None


def test_a_storage_error_keeps_the_cached_predictions(
    predictor: SpotPricePredictor, caplog: pytest.LogCaptureFixture
) -> None:
    predictor.day_ahead_predictions = {"2026-09-24T12:00:00Z": 1.0}

    with patch.object(
        predictor.storage,
        "get_predictions_between",
        side_effect=sqlite3.OperationalError("locked"),
    ):
        _refresh(predictor)

    assert predictor.day_ahead_predictions == {"2026-09-24T12:00:00Z": 1.0}
    assert "Failed to load the day-ahead predictions: locked" in caplog.text


def test_an_unreadable_start_is_skipped(predictor: SpotPricePredictor) -> None:
    rows = [
        {"start": "x", "price": 1.0, "stored_at": "y"},
        {
            "start": NOW.isoformat(),
            "price": 2.0,
            "stored_at": (NOW - timedelta(hours=24)).isoformat(),
        },
        # Stored after its slot started: not a forecast
        {
            "start": (NOW + QUARTER).isoformat(),
            "price": 3.0,
            "stored_at": (NOW + timedelta(hours=1)).isoformat(),
        },
    ]

    with patch.object(predictor.storage, "get_predictions_between", return_value=rows):
        _refresh(predictor)

    assert predictor.day_ahead_predictions == {
        "2026-09-24T12:00:00Z": pytest.approx(2.0)
    }


@pytest.mark.asyncio
async def test_every_forecast_run_reloads_the_cache() -> None:
    updater = ForecastUpdater.__new__(ForecastUpdater)
    updater.hass = Mock()
    updater.hass.async_add_executor_job = AsyncMock()
    predictor = MagicMock()
    predictor.save_learning_data = AsyncMock()
    updater.ml_predictor = predictor
    updater.weather = updater.load = None
    updater.sensors = Mock(external_forecasts=())
    updater.sensor_reader = updater.settings = Mock()
    updater.api_data = {"spot_data": None}

    with (
        patch.object(updater, "_read_weather", AsyncMock(return_value={"x": 1})),
        patch.object(updater, "_update_prognoses", AsyncMock()),
        patch.object(updater, "_update_ahead", AsyncMock()),
        patch.object(updater, "update_gas_price", AsyncMock()),
        patch.object(updater, "update_outages", AsyncMock()),
        patch.object(updater, "update_neighbours", AsyncMock()),
    ):
        await updater.run_forecast()

    jobs = [
        call.args[0] for call in updater.hass.async_add_executor_job.await_args_list
    ]
    assert jobs == [predictor.predict, predictor.refresh_day_ahead_predictions]


# --- The sensor ----------------------------------------------------------------------


def _sensor(
    predictions: dict[datetime, float] | None,
    output: PriceOutput | None = None,
    tariffs: TariffSchedule | None = None,
) -> DayAheadPredictionSensor:
    api_data: dict[str, Any] = {"tariffs": tariffs}
    if predictions is not None:
        predictor = Mock(is_trained=True)
        predictor.day_ahead_prediction = lambda slot: predictions.get(slot)
        api_data["ml_predictor"] = predictor
    return DayAheadPredictionSensor(
        Mock(),
        MagicMock(entry_id="test"),
        api_data,
        "DKK",
        output or PriceOutput(vat=0.25),
    )


def _value(sensor: DayAheadPredictionSensor, now: datetime) -> float | None:
    with patch("homeassistant.util.dt.utcnow", return_value=now):
        return sensor.native_value


def test_the_sensor_is_a_recorded_diagnostic_measurement() -> None:
    sensor = _sensor({})

    assert sensor.entity_category is EntityCategory.DIAGNOSTIC
    assert sensor.state_class is SensorStateClass.MEASUREMENT
    assert sensor.translation_key == "day_ahead_prediction"
    assert sensor.unique_id == f"{DOMAIN}_test_day_ahead_prediction"
    assert sensor.native_unit_of_measurement == "DKK/kWh"
    assert sensor.device_info == {"identifiers": {(DOMAIN, "test")}}
    assert sensor.extra_state_attributes == {"lead_hours": pytest.approx(24.0)}
    # Nothing keeps the state out of the recorder
    assert not getattr(sensor, "_unrecorded_attributes", frozenset())


def test_the_state_is_the_current_slots_prediction() -> None:
    sensor = _sensor({NOW: 1.0, NOW + QUARTER: 2.0, NOW + 3 * QUARTER: 4.0})

    # Converted like every price: VAT 25 %
    assert _value(sensor, NOW) == pytest.approx(1.25)
    assert _value(sensor, NOW + timedelta(minutes=14, seconds=59)) == pytest.approx(
        1.25
    )
    # The next slot's prediction from its first second
    assert _value(sensor, NOW + QUARTER) == pytest.approx(2.5)
    # A slot nothing was predicted for, then one that was
    assert _value(sensor, NOW + 2 * QUARTER) is None
    assert _value(sensor, NOW + 3 * QUARTER) == pytest.approx(5.0)


def test_negative_prices_and_other_units_are_converted() -> None:
    sensor = _sensor(
        {NOW: -0.1}, PriceOutput(vat=0.0, surcharge=10.0, price_type="MWh")
    )

    assert sensor.native_unit_of_measurement == "DKK/MWh"
    assert _value(sensor, NOW) == pytest.approx(-90.0)


def test_the_slots_tariff_is_added() -> None:
    tariffs = TariffSchedule({NOW: 0.4, NOW + QUARTER: 0.1})
    sensor = _sensor({NOW: 1.0, NOW + QUARTER: 1.0}, PriceOutput(vat=0.0), tariffs)

    assert _value(sensor, NOW) == pytest.approx(1.4)
    assert _value(sensor, NOW + QUARTER) == pytest.approx(1.1)


def test_with_hourly_average_the_state_is_the_hours_mean() -> None:
    predictions = {NOW + n * QUARTER: 1.0 + n for n in range(4)}
    # The next hour: only two of its slots were predicted
    predictions[NOW + 4 * QUARTER] = 10.0
    predictions[NOW + 6 * QUARTER] = 20.0
    sensor = _sensor(predictions, PriceOutput(vat=0.0, hourly_average=True))

    # The same value through the hour, changing on the hour
    for minutes in (0, 15, 44, 59):
        assert _value(sensor, NOW + timedelta(minutes=minutes)) == pytest.approx(2.5)
    assert _value(sensor, NOW + timedelta(minutes=60)) == pytest.approx(15.0)
    assert _value(sensor, NOW + timedelta(minutes=120)) is None


def test_without_a_model_or_a_prediction_the_state_is_unknown() -> None:
    assert _value(_sensor(None), NOW) is None
    assert _value(_sensor({}), NOW) is None


@pytest.mark.asyncio
async def test_the_state_is_written_on_slot_boundaries_and_updates() -> None:
    sensor = _sensor({})
    module = "custom_components.open_spot_forecast.day_ahead_sensor"

    with (
        patch.object(sensor, "async_on_remove") as on_remove,
        patch.object(sensor, "async_write_ha_state") as write,
        patch(f"{module}.async_dispatcher_connect") as connect,
        patch(f"{module}.async_track_time_change") as track,
    ):
        await sensor.async_added_to_hass()

        # After every forecast run and learning update (also the reset)
        assert [call.args[1:] for call in connect.call_args_list] == [
            (util_slugify(UPDATE_SIGNAL_FORECAST), write),
            (util_slugify(UPDATE_SIGNAL), write),
        ]
        # And when a slot starts: every quarter of an hour
        track.assert_called_once()
        assert list(track.call_args.kwargs["minute"]) == [0, 15, 30, 45]
        assert track.call_args.kwargs["second"] == 0
        # All three are removed with the entity
        assert [call.args[0] for call in on_remove.call_args_list] == [
            connect.return_value,
            connect.return_value,
            track.return_value,
        ]

        track.call_args.args[1](NOW)

        write.assert_called_once_with()
