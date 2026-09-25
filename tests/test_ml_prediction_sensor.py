"""The ML price sensor shows the prediction for the current slot (#56).

Predictions start at the current 15-minute slot (#18). The sensor picks the
slot that contains now, compared in UTC, and falls back to the first future
slot. It used to compare aware starts with a naive ``datetime.now()``; the
TypeError was swallowed and ``predictions[0]`` was shown by accident.
"""

import re
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import pytest

from custom_components.open_spot_forecast.sensor import (
    MLPredictionSensor,
    current_prediction,
)
from custom_components.open_spot_forecast.time_slots import (
    slot_start_in_day,
    slots_in_local_day,
)

SENSOR_PY = (
    Path(__file__).parent.parent / "custom_components/open_spot_forecast/sensor.py"
)
VAT = 0.25
DAY = date(2026, 9, 24)
FALL_BACK = date(2026, 10, 25)


def _predictions(day: date, first: int = 0) -> list[dict[str, Any]]:
    """Return predictions like the model's: local ISO slot bounds, price = index."""
    count = slots_in_local_day(day)
    return [
        {
            "start": slot_start_in_day(day, index).isoformat(),
            "end": (
                slot_start_in_day(day, index + 1)
                if index + 1 < count
                else slot_start_in_day(day + timedelta(days=1), 0)
            ).isoformat(),
            "price": float(index),
            "confidence": 0.8,
        }
        for index in range(first, count)
    ]


def _sensor(predictions: list[dict[str, Any]]) -> MLPredictionSensor:
    predictor = MagicMock()
    predictor.predictions = predictions
    predictor.get_prediction_stats.return_value = {}
    return MLPredictionSensor(
        MagicMock(),
        MagicMock(entry_id="test"),
        {"ml_predictor": predictor},
        "DKK",
        VAT,
        4,
        "kWh",
    )


def _state_at(sensor: MLPredictionSensor, now: datetime) -> float | None:
    with patch("homeassistant.util.dt.utcnow", return_value=now.astimezone(UTC)):
        return sensor.native_value


@pytest.mark.usefixtures("copenhagen_time_zone")
def test_shows_the_current_slot_then_the_next_one() -> None:
    cph = ZoneInfo("Europe/Copenhagen")
    sensor = _sensor(_predictions(DAY, first=40))  # from 10:00 local

    # 10:07 is in the first slot (10:00-10:15), index 40
    assert _state_at(sensor, datetime(2026, 9, 24, 10, 7, tzinfo=cph)) == (
        pytest.approx(40 * (1 + VAT))
    )
    # One minute after that slot ended: the 10:15 slot, index 41
    assert _state_at(sensor, datetime(2026, 9, 24, 10, 16, tzinfo=cph)) == (
        pytest.approx(41 * (1 + VAT))
    )
    # A slot's end belongs to the next slot
    assert _state_at(sensor, datetime(2026, 9, 24, 10, 15, tzinfo=cph)) == (
        pytest.approx(41 * (1 + VAT))
    )


@pytest.mark.usefixtures("copenhagen_time_zone")
def test_shows_the_first_future_prediction_when_they_start_later() -> None:
    cph = ZoneInfo("Europe/Copenhagen")
    sensor = _sensor(_predictions(DAY, first=40))

    assert _state_at(sensor, datetime(2026, 9, 24, 9, 50, tzinfo=cph)) == (
        pytest.approx(40 * (1 + VAT))
    )


@pytest.mark.usefixtures("copenhagen_time_zone")
@pytest.mark.parametrize(
    ("utc_now", "expected_index", "expected_start"),
    [
        # First 02:15 (CEST, +02:00) is 00:15 UTC: slot 9
        (datetime(2026, 10, 25, 0, 20, tzinfo=UTC), 9, "2026-10-25T02:15:00+02:00"),
        # Second 02:15 (CET, +01:00) is 01:15 UTC: slot 13
        (datetime(2026, 10, 25, 1, 20, tzinfo=UTC), 13, "2026-10-25T02:15:00+01:00"),
    ],
)
def test_picks_the_right_slot_in_the_repeated_hour(
    utc_now: datetime, expected_index: int, expected_start: str
) -> None:
    predictions = _predictions(FALL_BACK)
    sensor = _sensor(predictions)

    assert _state_at(sensor, utc_now) == pytest.approx(expected_index * (1 + VAT))
    with patch("homeassistant.util.dt.utcnow", return_value=utc_now):
        assert sensor.extra_state_attributes["state_slot_start"] == expected_start


def test_prefers_the_earliest_future_prediction_whatever_the_order() -> None:
    now = datetime(2026, 9, 24, 8, 0, tzinfo=UTC)
    later = {"start": "2026-09-24T10:00:00+00:00", "price": 2.0}
    sooner = {"start": "2026-09-24T09:00:00+00:00", "price": 1.0}

    assert current_prediction([later, sooner], now) is sooner


def test_a_prediction_without_an_end_covers_one_slot() -> None:
    prediction = {"start": "2026-09-24T08:00:00+00:00", "price": 1.0}

    in_slot = datetime(2026, 9, 24, 8, 14, tzinfo=UTC)
    after_slot = datetime(2026, 9, 24, 8, 15, tzinfo=UTC)

    assert current_prediction([prediction], in_slot) is prediction
    assert current_prediction([prediction], after_slot) is None


def test_naive_timestamps_are_home_assistant_local_time() -> None:
    prediction = {"start": "2026-09-24T08:00:00", "end": "2026-09-24T08:15:00"}

    # Home Assistant's default time zone is UTC in tests
    assert (
        current_prediction([prediction], datetime(2026, 9, 24, 8, 5, tzinfo=UTC))
        is prediction
    )


def test_state_slot_start_is_none_without_a_usable_prediction() -> None:
    sensor = _sensor([{"start": "2020-01-01T00:00:00+00:00", "price": 1.0}])

    assert sensor.native_value is None
    assert sensor.extra_state_attributes["state_slot_start"] is None


def test_no_naive_now_in_sensor_py() -> None:
    """The sensor reads the clock through Home Assistant, never naively."""
    source = SENSOR_PY.read_text(encoding="utf-8")

    assert not re.search(r"datetime\.now\(\)|date\.today\(\)", source)
    assert "except ValueError, TypeError" not in source
