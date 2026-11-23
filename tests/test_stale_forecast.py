"""The last good forecast is kept, flagged stale, without spot prices (#277)."""

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, Mock, patch

import pytest

from custom_components.open_spot_forecast.const import FORECAST_STALE_AFTER_HOURS
from custom_components.open_spot_forecast.forecast_attributes import (
    forecast_freshness,
)
from custom_components.open_spot_forecast.ml.predictor import SpotPricePredictor
from custom_components.open_spot_forecast.price_output import PriceOutput
from custom_components.open_spot_forecast.sensor import MLPredictionSensor

NOW = datetime(2026, 9, 24, 10, 7, tzinfo=UTC)


def _slot(start: datetime, price: float) -> dict[str, Any]:
    return {
        "start": start.isoformat(),
        "end": (start + timedelta(minutes=15)).isoformat(),
        "price": price,
        "confidence": 0.8,
    }


def _forecast() -> list[dict[str, Any]]:
    """Slots from 09:45 to 10:45: the first one is over at NOW."""
    first = datetime(2026, 9, 24, 9, 45, tzinfo=UTC)
    return [_slot(first + timedelta(minutes=15 * n), float(n)) for n in range(4)]


def _predictor(tmp_path: Path) -> SpotPricePredictor:
    hass = Mock()
    hass.config.path.return_value = str(tmp_path / ".storage")
    return SpotPricePredictor(hass, "DK1")


@pytest.mark.parametrize("prices", [[], [None] * 96])
def test_no_prices_keep_the_forecast_without_its_past_slots(
    tmp_path: Path, prices: list[float | None]
) -> None:
    predictor = _predictor(tmp_path)
    previous = _forecast()
    predictor.predictions = previous

    with (
        patch("homeassistant.util.dt.utcnow", return_value=NOW),
        patch.object(predictor, "store_prediction_for_learning") as store,
        patch.object(predictor, "record_training_prices") as record,
    ):
        assert predictor.predict({}, prices, 1, 15) is False

    # 10:00-10:15 holds NOW, so it stays; 09:45-10:00 is over
    assert predictor.predictions == previous[1:]
    # A new list: the forecast sensor's cache is rebuilt
    assert predictor.predictions is not previous
    assert predictor.confidence_scores == [pytest.approx(0.8)] * 3
    # Stored when made: never scored twice by self-learning
    store.assert_not_called()
    record.assert_not_called()
    predictor.storage.close()


def test_a_new_forecast_is_reported(tmp_path: Path) -> None:
    predictor = _predictor(tmp_path)

    assert predictor.predict({}, [0.5] * 96, 1, 15) is True
    assert predictor.predictions
    predictor.storage.close()


def test_freshness_before_the_first_forecast() -> None:
    assert forecast_freshness({}, NOW) == {"generated_at": None, "stale": False}


def test_a_kept_or_old_forecast_is_stale() -> None:
    fresh = {"forecast_generated_at": NOW, "forecast_stale": False}
    assert forecast_freshness(fresh, NOW + timedelta(hours=6)) == {
        "generated_at": NOW.isoformat(),
        "stale": False,
    }
    assert forecast_freshness({**fresh, "forecast_stale": True}, NOW)["stale"]
    late = NOW + timedelta(hours=FORECAST_STALE_AFTER_HOURS, minutes=1)
    assert forecast_freshness(fresh, late)["stale"]


def test_the_forecast_sensor_shows_a_kept_forecast_flagged_stale() -> None:
    predictor = MagicMock()
    predictor.predictions = _forecast()[1:]
    predictor.get_prediction_stats.return_value = {}
    generated = NOW - timedelta(hours=7)
    sensor = MLPredictionSensor(
        MagicMock(),
        MagicMock(entry_id="test"),
        {
            "ml_predictor": predictor,
            "forecast_generated_at": generated,
            "forecast_stale": True,
        },
        "DKK",
        PriceOutput(vat=0.25, precision=4),
    )

    with patch("homeassistant.util.dt.utcnow", return_value=NOW):
        assert sensor.native_value == pytest.approx(1.25)
        attributes = sensor.extra_state_attributes
    assert attributes["generated_at"] == generated.isoformat()
    assert attributes["stale"] is True
