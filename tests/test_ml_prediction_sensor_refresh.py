"""The ML forecast sensor builds its state off the event loop.

Home Assistant warned that a state write of the sensor took 0.9 seconds: the
properties walked a week of slots. ``async_update`` builds the state and the
attributes in an executor job and the properties return what it stored.
"""

from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components.open_spot_forecast.price_output import PriceOutput
from custom_components.open_spot_forecast.sensor import MLPredictionSensor

NOW = datetime(2026, 9, 24, 8, 5, tzinfo=UTC)
FIRST = datetime(2026, 9, 24, 8, 0, tzinfo=UTC)
SLOTS = 7 * 96
OUTPUT = PriceOutput(vat=0.25, surcharge=0.1, precision=4)


def _predictions(offset: float = 0.0) -> list[dict[str, Any]]:
    """A week of 15-minute predictions from the current slot on."""
    return [
        {
            "start": (FIRST + timedelta(minutes=15 * index)).isoformat(),
            "end": (FIRST + timedelta(minutes=15 * (index + 1))).isoformat(),
            "price": offset + 0.5 + (index % 96) / 100,
            "confidence": 0.7,
        }
        for index in range(SLOTS)
    ]


def _sensor() -> tuple[MLPredictionSensor, AsyncMock]:
    hass = MagicMock()
    executor = AsyncMock(side_effect=lambda job, *args: job(*args))
    hass.async_add_executor_job = executor
    predictor = MagicMock()
    predictor.predictions = _predictions()
    predictor.get_prediction_stats.return_value = {"total_predictions": SLOTS}
    sensor = MLPredictionSensor(
        hass,
        MagicMock(entry_id="test"),
        {"ml_predictor": predictor},
        "DKK",
        OUTPUT,
        168,
    )
    return sensor, executor


@pytest.mark.asyncio
async def test_the_state_is_built_in_an_executor_job() -> None:
    sensor, executor = _sensor()

    with patch("homeassistant.util.dt.utcnow", return_value=NOW):
        await sensor.async_update()

    executor.assert_awaited_once()
    assert executor.await_args is not None
    assert executor.await_args.args == (sensor._build_state, NOW)
    assert sensor.native_value == pytest.approx(OUTPUT.convert(0.5))
    assert sensor.extra_state_attributes["total_predictions"] == SLOTS


@pytest.mark.asyncio
async def test_a_state_write_only_reads_the_stored_state() -> None:
    sensor, _ = _sensor()
    with patch("homeassistant.util.dt.utcnow", return_value=NOW):
        await sensor.async_update()

    with (
        patch.object(sensor, "_state_value") as value,
        patch.object(sensor, "_state_attributes") as attributes,
    ):
        state = sensor.native_value
        attrs = sensor.extra_state_attributes

    value.assert_not_called()
    attributes.assert_not_called()
    assert state == pytest.approx(OUTPUT.convert(0.5))
    assert attrs["predictions"][0]["price"] == pytest.approx(OUTPUT.convert(0.5))


@pytest.mark.asyncio
async def test_a_new_forecast_is_built_before_it_is_written() -> None:
    sensor, executor = _sensor()
    with (
        patch("homeassistant.util.dt.utcnow", return_value=NOW),
        patch.object(sensor, "async_write_ha_state") as write,
    ):
        await sensor.async_update()
        sensor.api_data["ml_predictor"].predictions = _predictions(offset=1.0)

        await sensor._handle_update()

    assert executor.await_count == 2
    write.assert_called_once()
    assert sensor.native_value == pytest.approx(OUTPUT.convert(1.5))
