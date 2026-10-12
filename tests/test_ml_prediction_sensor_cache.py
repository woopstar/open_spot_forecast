"""The ML forecast sensor builds its forecast once per state write (#179)."""

from collections.abc import Iterable, Iterator
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from custom_components.open_spot_forecast.const import (
    ATTRIBUTE_FORMAT_COMPACT,
    DEFAULT_ATTRIBUTE_FORMAT,
)
from custom_components.open_spot_forecast.price_output import PriceOutput
from custom_components.open_spot_forecast.sensor import MLPredictionSensor
from custom_components.open_spot_forecast.tariffs import TariffSchedule

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


def _tariffs(tariff: float) -> TariffSchedule:
    return TariffSchedule({}, default=tariff)


def _sensor(
    attribute_format: str = DEFAULT_ATTRIBUTE_FORMAT,
    include_known: bool = False,
    tariffs: TariffSchedule | None = None,
) -> MLPredictionSensor:
    predictor = MagicMock()
    predictor.predictions = _predictions()
    predictor.get_prediction_stats.return_value = {
        "mean_confidence": 0.7,
        "total_predictions": SLOTS,
        "is_ml_model": True,
        "training_samples": 5760,
    }
    api_data: dict[str, Any] = {"ml_predictor": predictor, "tariffs": tariffs}
    return MLPredictionSensor(
        MagicMock(),
        MagicMock(entry_id="test"),
        api_data,
        "DKK",
        OUTPUT,
        168,
        attribute_format,
        include_known,
    )


def _write(sensor: MLPredictionSensor) -> tuple[float | None, dict[str, Any]]:
    """Read what one state write reads: the state, then the attributes."""
    with patch("homeassistant.util.dt.utcnow", return_value=NOW):
        return sensor.native_value, sensor.extra_state_attributes


@pytest.fixture
def series_builds() -> Iterator[list[int]]:
    """Count the passes over a forecast: the slots given to ``forecast_series``."""
    builds: list[int] = []
    real = PriceOutput.forecast_series

    def counted(
        output: PriceOutput,
        predictions: Iterable[dict[str, Any]],
        tariffs: TariffSchedule | None = None,
    ) -> list[dict[str, Any]]:
        predictions = list(predictions)
        builds.append(len(predictions))
        return real(output, predictions, tariffs)

    with patch.object(PriceOutput, "forecast_series", counted):
        yield builds


@pytest.mark.parametrize(
    "attribute_format", [DEFAULT_ATTRIBUTE_FORMAT, ATTRIBUTE_FORMAT_COMPACT]
)
def test_a_state_write_builds_the_forecast_once(
    attribute_format: str, series_builds: list[int]
) -> None:
    """State, attributes and statistics share one pass over the week."""
    sensor = _sensor(attribute_format)

    state, attributes = _write(sensor)

    assert series_builds == [SLOTS]
    assert state == pytest.approx(OUTPUT.convert(0.5))
    assert attributes["total_predictions"] == SLOTS
    assert attributes["forecast_min"] == pytest.approx(OUTPUT.convert(0.5))
    assert attributes["forecast_max"] == pytest.approx(OUTPUT.convert(1.45))


def test_the_known_prices_series_is_the_only_other_pass(
    series_builds: list[int],
) -> None:
    """With the confirmed prices in front, their series is built once as well."""
    sensor = _sensor(include_known=True)

    _write(sensor)

    assert len(series_builds) == 2
    assert series_builds[0] == SLOTS


def test_the_next_write_reuses_the_forecast_until_it_changes(
    series_builds: list[int],
) -> None:
    sensor = _sensor()
    first = _write(sensor)

    assert _write(sensor) == first
    assert series_builds == [SLOTS]

    # A forecast run replaces the predictions list
    sensor.api_data["ml_predictor"].predictions = _predictions(offset=1.0)
    state, attributes = _write(sensor)

    assert series_builds == [SLOTS, SLOTS]
    assert state == pytest.approx(OUTPUT.convert(1.5))
    assert attributes["forecast_max"] == pytest.approx(OUTPUT.convert(2.45))


def test_a_new_tariff_schedule_gives_new_prices(series_builds: list[int]) -> None:
    sensor = _sensor(tariffs=_tariffs(0.2))
    state, _ = _write(sensor)
    assert state == pytest.approx(OUTPUT.convert(0.7))

    sensor.api_data["tariffs"] = _tariffs(0.3)
    state, attributes = _write(sensor)

    assert series_builds == [SLOTS, SLOTS]
    assert state == pytest.approx(OUTPUT.convert(0.8))
    assert attributes["includes_tariffs"] is True


def test_a_list_that_is_still_being_filled_is_not_cached_as_complete(
    series_builds: list[int],
) -> None:
    """A write during a forecast run sees the slots so far, the next one all."""
    sensor = _sensor()
    predictions = _predictions()
    growing = predictions[:96]
    sensor.api_data["ml_predictor"].predictions = growing
    _, attributes = _write(sensor)
    assert attributes["forecast_max"] == pytest.approx(OUTPUT.convert(1.45))

    growing.extend({**p, "price": p["price"] + 1.0} for p in predictions[96:])
    _, attributes = _write(sensor)

    assert series_builds == [96, SLOTS]
    assert attributes["forecast_max"] == pytest.approx(OUTPUT.convert(2.45))


def test_the_cached_forecast_is_what_the_output_gives() -> None:
    """Caching changes when the forecast is built, not what it is."""
    sensor = _sensor(tariffs=_tariffs(0.2))
    predictions = sensor.api_data["ml_predictor"].predictions
    tariffs = sensor.api_data["tariffs"]

    series, forecast = sensor._cached_forecast()

    assert series == OUTPUT.forecast_series(predictions, tariffs)
    assert forecast == OUTPUT.forecast(predictions, tariffs)
    assert sensor._forecast() is forecast


def test_without_predictions_nothing_is_cached() -> None:
    sensor = _sensor()
    sensor.api_data["ml_predictor"].predictions = []

    assert sensor._cached_forecast() == ([], [])
    assert _write(sensor)[0] is None
    assert sensor._series_cache is None
