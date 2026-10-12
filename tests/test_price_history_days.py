"""Tests for the price history's shape: one entry per local day (#173)."""

from collections.abc import Callable, Iterator
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import Mock
from zoneinfo import ZoneInfo

import pytest

from custom_components.open_spot_forecast.ml.gbm import NumpyGradientBoosting
from custom_components.open_spot_forecast.ml.predictor import SpotPricePredictor
from custom_components.open_spot_forecast.ml.series_storage import OPENMETEO_WEATHER
from custom_components.open_spot_forecast.ml.zone_weather import zone_points
from custom_components.open_spot_forecast.price_series import split_local_days
from custom_components.open_spot_forecast.time_slots import slot_start_in_day

TZ = ZoneInfo("Europe/Copenhagen")
DAY_1 = [0.5 + slot / 1000 for slot in range(96)]
DAY_2 = [0.7 + slot / 1000 for slot in range(96)]
DAY_3 = [0.9 + slot / 1000 for slot in range(96)]


# --- split_local_days ---------------------------------------------------------------


def test_a_series_is_split_at_local_midnight() -> None:
    days = split_local_days(DAY_1 + DAY_2, date(2026, 9, 14), TZ)

    assert days == [(date(2026, 9, 14), DAY_1), (date(2026, 9, 15), DAY_2)]


@pytest.mark.parametrize(
    ("first_day", "slots"),
    [(date(2026, 3, 29), 92), (date(2026, 10, 25), 100), (date(2026, 9, 14), 96)],
    ids=["spring-forward", "fall-back", "normal"],
)
def test_each_day_takes_its_own_number_of_slots(first_day: date, slots: int) -> None:
    """The day after a DST change starts at its own midnight, not at slot 96."""
    prices: list[float | None] = [float(index) for index in range(slots + 96)]

    days = split_local_days(prices, first_day, TZ)

    assert [(day, len(values)) for day, values in days] == [
        (first_day, slots),
        (first_day + timedelta(days=1), 96),
    ]
    assert days[1][1][0] == pytest.approx(float(slots))


def test_a_series_that_ends_inside_a_day_keeps_the_short_day() -> None:
    days = split_local_days([*DAY_1, None, 0.3], date(2026, 9, 14), TZ)

    assert days == [(date(2026, 9, 14), DAY_1), (date(2026, 9, 15), [None, 0.3])]
    assert split_local_days([], date(2026, 9, 14), TZ) == []
    assert split_local_days(DAY_1[:10], date(2026, 9, 14), TZ) == [
        (date(2026, 9, 14), DAY_1[:10])
    ]


# --- The predictor's price history ---------------------------------------------------


async def _run_inline(func: Callable[..., Any], *args: Any) -> Any:
    return func(*args)


@pytest.fixture
def predictor(tmp_path: Path) -> Iterator[SpotPricePredictor]:
    """A DK1 predictor on a throwaway database, with a small model."""
    hass = Mock()
    hass.config.path.return_value = str(tmp_path / ".storage")
    hass.async_add_executor_job = _run_inline
    predictor = SpotPricePredictor(hass, "DK1")
    predictor.price_model = NumpyGradientBoosting(n_estimators=2)
    yield predictor
    predictor.storage.close()


def _over_midnight(predictor: SpotPricePredictor) -> None:
    """Store what three forecast runs store around a midnight.

    The afternoon of day 1 (today and tomorrow), the morning of day 2
    (today only) and its afternoon (today and tomorrow).
    """
    predictor.record_training_prices(DAY_1 + DAY_2, "2026-09-14")
    predictor.record_training_prices(DAY_2, "2026-09-15")
    predictor.record_training_prices(DAY_2 + DAY_3, "2026-09-15")


def test_tomorrows_prices_are_stored_as_their_own_day(
    predictor: SpotPricePredictor,
) -> None:
    _over_midnight(predictor)

    assert predictor.price_history == [
        {"date": "2026-09-14", "prices": DAY_1},
        {"date": "2026-09-15", "prices": DAY_2},
        {"date": "2026-09-16", "prices": DAY_3},
    ]


def test_every_slot_is_a_training_row_once(predictor: SpotPricePredictor) -> None:
    """Tomorrow's slots do not come back from the next day's entry."""
    _over_midnight(predictor)

    prices, features = predictor.get_all_historical_prices()

    starts = [feature["start"] for feature in features]
    assert len(prices) == len(starts) == len(set(starts)) == 3 * 96
    assert prices == pytest.approx(DAY_1 + DAY_2 + DAY_3)
    assert starts[96] == slot_start_in_day(date(2026, 9, 15), 0, TZ).isoformat()

    predictor._train_models()

    assert predictor.training_samples == 3 * 96


@pytest.mark.asyncio
async def test_the_history_in_memory_is_the_history_after_a_reload(
    predictor: SpotPricePredictor,
) -> None:
    """A restart does not change what the model trains on."""
    _over_midnight(predictor)
    in_memory = [dict(entry) for entry in predictor.price_history]

    await predictor.save_learning_data()
    predictor.price_history = []
    await predictor._load_learning_data()

    assert predictor.price_history == in_memory


def test_a_dst_day_keeps_its_slots_and_the_next_day_its_midnight(
    predictor: SpotPricePredictor,
) -> None:
    """92 slots on the spring-forward day, then tomorrow from its own midnight."""
    short_day = [0.4 + slot / 1000 for slot in range(92)]

    predictor.record_training_prices(short_day + DAY_2, "2026-03-29")

    assert [(e["date"], len(e["prices"])) for e in predictor.price_history] == [
        ("2026-03-29", 92),
        ("2026-03-30", 96),
    ]
    _prices, features = predictor.get_all_historical_prices()
    assert features[92]["start"] == datetime(2026, 3, 30, tzinfo=TZ).isoformat()


def test_only_a_new_or_changed_day_counts_as_new_training_data(
    predictor: SpotPricePredictor,
) -> None:
    """Tomorrow's prices are new data once; today's own list the next morning is not."""
    predictor.record_training_prices(DAY_1, "2026-09-14")
    first = predictor._prices_updated_at
    assert first is not None

    predictor.record_training_prices(DAY_1 + DAY_2, "2026-09-14")
    with_tomorrow = predictor._prices_updated_at
    assert with_tomorrow is not None and with_tomorrow > first

    # The same prices again, and the next morning's list of the new day alone
    predictor.record_training_prices(DAY_1 + DAY_2, "2026-09-14")
    predictor.record_training_prices(DAY_2, "2026-09-15")
    assert predictor._prices_updated_at == with_tomorrow

    corrected = [*DAY_2[:-1], DAY_2[-1] + 0.5]
    predictor.record_training_prices(DAY_1 + corrected, "2026-09-14")
    after_correction = predictor._prices_updated_at
    assert after_correction is not None and after_correction > with_tomorrow


def test_tomorrows_training_rows_get_their_stored_inputs(
    predictor: SpotPricePredictor,
) -> None:
    """The history's span reaches tomorrow, so its rows have the zone weather."""
    noon = slot_start_in_day(date(2026, 9, 15), 48, TZ)
    predictor.storage.upsert_series(
        OPENMETEO_WEATHER,
        [
            {
                "timestamp": noon.isoformat(),
                "point": point,
                "wind_80m": 9.0,
                "temperature": 14.0,
                "irradiance": 200.0,
                "pressure": 1012.0,
                "humidity": 70.0,
            }
            for point in zone_points("DK1")
        ],
    )
    predictor.record_training_prices(DAY_1 + DAY_2, "2026-09-14")

    _prices, features = predictor.get_all_historical_prices()

    assert features[96 + 48]["start"] == noon.isoformat()
    assert features[96 + 48]["zone_wind"] == pytest.approx(9.0)


def test_a_date_that_does_not_parse_stores_nothing(
    predictor: SpotPricePredictor,
) -> None:
    assert predictor.store_daily_prices(DAY_1, "tomorrow") is False
    assert predictor.store_daily_prices([], "2026-09-14") is False
    assert predictor.price_history == []
