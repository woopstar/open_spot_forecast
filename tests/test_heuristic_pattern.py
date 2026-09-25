"""The heuristic fallback's hourly pattern uses each slot's real local hour (#57).

The known prices are a 15-minute grid from local midnight (92/96/100 slots a
day, None for a missing slot). The pattern used to read them by position, as
24 hourly prices per day, so "hour 0" of a 96-slot day was the mean of 00:00,
06:00, 12:00 and 18:00.
"""

from collections.abc import Iterator
from datetime import UTC, date, datetime
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from homeassistant.util import dt as dt_util

from custom_components.open_spot_forecast.ml.predictor import SpotPricePredictor

DAY = date(2026, 9, 24)


@pytest.fixture
def predictor(tmp_path: Path) -> Iterator[SpotPricePredictor]:
    hass = Mock()
    hass.config.path.return_value = str(tmp_path / ".storage")
    predictor = SpotPricePredictor(hass, "DK1")
    yield predictor
    predictor.storage.close()


def _day_prices(hours: list[int]) -> list[float | None]:
    """Return four 15-minute prices per listed local hour; price = 1 + hour."""
    return [1.0 + hour for hour in hours for _ in range(4)]


def _expected(hours_with_data: set[int]) -> list[float]:
    """Return the pattern of prices 1 + hour over the given hours."""
    mean = sum(1.0 + hour for hour in hours_with_data) / len(hours_with_data)
    return [
        (1.0 + hour) / mean if hour in hours_with_data else 1.0 for hour in range(24)
    ]


def test_96_slot_day_gives_the_per_hour_means(predictor: SpotPricePredictor) -> None:
    prices = _day_prices(list(range(24)))

    pattern = predictor._extract_hourly_pattern(prices, DAY)

    assert pattern == pytest.approx(_expected(set(range(24))))
    # The positional reading made hour 0 the mean of 00:00, 06:00, 12:00, 18:00
    assert pattern[0] == pytest.approx(1.0 / 12.5)


def test_missing_slots_do_not_shift_the_others(predictor: SpotPricePredictor) -> None:
    prices = _day_prices(list(range(24)))
    prices[5] = None  # 01:15
    prices[40:44] = [None] * 4  # the whole of 10:00-11:00

    pattern = predictor._extract_hourly_pattern(prices, DAY)

    assert pattern == pytest.approx(_expected(set(range(24)) - {10}))
    assert pattern[10] == pytest.approx(1.0)


@pytest.mark.usefixtures("copenhagen_time_zone")
@pytest.mark.parametrize(
    ("day", "hours"),
    [
        # Spring forward: 02:00-03:00 does not exist (92 slots)
        (date(2026, 3, 29), [0, 1, *range(3, 24)]),
        # Fall back: 02:00-03:00 happens twice (100 slots)
        (date(2026, 10, 25), [0, 1, 2, 2, *range(3, 24)]),
    ],
    ids=["92-slots", "100-slots"],
)
def test_dst_days_map_every_slot_to_its_local_hour(
    predictor: SpotPricePredictor, day: date, hours: list[int]
) -> None:
    prices = _day_prices(hours)

    pattern = predictor._extract_hourly_pattern(prices, day)

    assert pattern == pytest.approx(_expected(set(hours)))


def test_hours_without_data_are_neutral(predictor: SpotPricePredictor) -> None:
    prices = _day_prices(list(range(12)))  # until noon

    pattern = predictor._extract_hourly_pattern(prices, DAY)

    assert pattern[12:] == pytest.approx([1.0] * 12)
    assert pattern[:12] == pytest.approx(_expected(set(range(12)))[:12])


def test_series_continues_into_the_next_day(predictor: SpotPricePredictor) -> None:
    """Today then tomorrow: slot 96 is 00:00 of the next day."""
    prices = _day_prices(list(range(24))) + _day_prices(list(range(24)))

    pattern = predictor._extract_hourly_pattern(prices, DAY)

    assert pattern == pytest.approx(_expected(set(range(24))))


@pytest.mark.parametrize(
    "prices",
    [[], [None, None], [-1.0] * 96, [0.0] * 96],
    ids=["empty", "all-missing", "negative-mean", "zero-mean"],
)
def test_no_usable_mean_gives_a_neutral_pattern(
    predictor: SpotPricePredictor, prices: list[float | None]
) -> None:
    assert predictor._extract_hourly_pattern(prices, DAY) == pytest.approx([1.0] * 24)


@pytest.mark.usefixtures("copenhagen_time_zone")
def test_heuristic_predictions_follow_the_local_hour(
    predictor: SpotPricePredictor,
) -> None:
    """End to end: today's grid from local midnight, predictions by local hour."""
    now = datetime(2026, 9, 24, 8, 5, tzinfo=UTC)  # 10:05 local
    prices = _day_prices(list(range(24)))  # mean over hours = 12.5
    with (
        patch("homeassistant.util.dt.utcnow", return_value=now),
        patch("homeassistant.util.dt.now", return_value=dt_util.as_local(now)),
    ):
        predictor._generate_heuristic_predictions(prices, 1, 15)

    assert predictor.predictions
    for prediction in predictor.predictions:
        start = dt_util.parse_datetime(prediction["start"])
        assert start is not None
        assert prediction["price"] == pytest.approx(1.0 + start.hour)
