"""Tests for the shared price-slot time helpers and the prediction start (#18)."""

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import Mock, patch
from zoneinfo import ZoneInfo

import pytest

from custom_components.open_spot_forecast.ml.predictor import SpotPricePredictor
from custom_components.open_spot_forecast.time_slots import (
    ceil_to_slot,
    first_prediction_slot,
    floor_to_slot,
)

CPH = ZoneInfo("Europe/Copenhagen")
NOW = datetime(2026, 9, 24, 10, 5, 42, tzinfo=UTC)


@pytest.mark.parametrize(
    ("moment", "floor", "ceil"),
    [
        (
            NOW,
            datetime(2026, 9, 24, 10, 0, tzinfo=UTC),
            NOW.replace(minute=15, second=0),
        ),
        (
            datetime(2026, 9, 24, 10, 30, tzinfo=UTC),
            datetime(2026, 9, 24, 10, 30, tzinfo=UTC),
            datetime(2026, 9, 24, 10, 30, tzinfo=UTC),
        ),
        (
            datetime(2026, 9, 24, 23, 59, 59, 999999, tzinfo=UTC),
            datetime(2026, 9, 24, 23, 45, tzinfo=UTC),
            datetime(2026, 9, 25, 0, 0, tzinfo=UTC),
        ),
    ],
)
def test_floor_and_ceil_to_slot(
    moment: datetime, floor: datetime, ceil: datetime
) -> None:
    """Moments round to 15-minute boundaries; a boundary maps to itself."""
    assert floor_to_slot(moment) == floor
    assert ceil_to_slot(moment) == ceil


@pytest.mark.parametrize("fold", [0, 1])
def test_slot_rounding_in_the_repeated_dst_fall_back_hour(fold: int) -> None:
    """02:00-03:00 happens twice on 2026-10-25; each pass rounds within itself.

    fold=0 is the first pass (CEST, UTC+2), fold=1 the second (CET, UTC+1).
    Compared in UTC: PEP 495 makes a fold=1 time unequal to any other zone.
    """
    moment = datetime(2026, 10, 25, 2, 0, 30, tzinfo=CPH, fold=fold)
    utc_hour = 0 if fold == 0 else 1

    floored = floor_to_slot(moment)
    ceiled = ceil_to_slot(moment)

    assert floored.astimezone(UTC) == datetime(2026, 10, 25, utc_hour, 0, tzinfo=UTC)
    assert ceiled.astimezone(UTC) == datetime(2026, 10, 25, utc_hour, 15, tzinfo=UTC)
    assert floored.tzinfo is ceiled.tzinfo is CPH
    assert floored.utcoffset() == ceiled.utcoffset() == moment.utcoffset()


def test_floor_to_slot_hourly_interval() -> None:
    """The legacy hourly interval floors to the whole hour."""
    assert floor_to_slot(NOW, 60) == datetime(2026, 9, 24, 10, 0, tzinfo=UTC)
    assert ceil_to_slot(NOW, 60) == datetime(2026, 9, 24, 11, 0, tzinfo=UTC)


@pytest.mark.parametrize(
    ("known_end", "expected"),
    [
        # No confirmed prices: the current slot, not the next whole hour
        (None, datetime(2026, 9, 24, 10, 0, tzinfo=UTC)),
        # Confirmed prices end mid-hour: start right there
        (
            datetime(2026, 9, 24, 12, 30, tzinfo=UTC),
            datetime(2026, 9, 24, 12, 30, tzinfo=UTC),
        ),
        # Confirmed prices end inside a slot: the next boundary
        (
            datetime(2026, 9, 24, 12, 40, tzinfo=UTC),
            datetime(2026, 9, 24, 12, 45, tzinfo=UTC),
        ),
        # Confirmed prices ended in the past: the current slot
        (
            datetime(2026, 9, 24, 9, 30, tzinfo=UTC),
            datetime(2026, 9, 24, 10, 0, tzinfo=UTC),
        ),
    ],
)
def test_first_prediction_slot(known_end: datetime | None, expected: datetime) -> None:
    """Start at max(current slot, first boundary at or after known data)."""
    assert first_prediction_slot(NOW, known_end) == expected


def _predictor(tmp_path: Path) -> SpotPricePredictor:
    """Return a predictor backed by a throwaway SQLite store."""
    hass = Mock()
    hass.config.path.return_value = str(tmp_path / ".storage")
    return SpotPricePredictor(hass, "DK1")


def _first_start(predictions: list[dict]) -> datetime:
    return datetime.fromisoformat(predictions[0]["start"])


@pytest.mark.parametrize(
    ("known_end", "expected"),
    [
        (None, datetime(2026, 9, 24, 10, 0, tzinfo=UTC)),
        (
            datetime(2026, 9, 24, 12, 30, tzinfo=UTC),
            datetime(2026, 9, 24, 12, 30, tzinfo=UTC),
        ),
    ],
)
def test_ml_and_heuristic_paths_start_at_the_same_slot(
    tmp_path: Path, known_end: datetime | None, expected: datetime
) -> None:
    """At hh:05 both prediction paths start at the current slot or the known end."""
    predictor = _predictor(tmp_path)

    with patch("homeassistant.util.dt.utcnow", return_value=NOW):
        time_features = predictor._generate_time_features(1, 15, known_end)
        predictor._generate_heuristic_predictions([1.0] * 96, 1, 15, known_end)

    assert _first_start(time_features) == expected
    assert _first_start(predictor.predictions) == expected
    assert len(time_features) == len(predictor.predictions) == 96
