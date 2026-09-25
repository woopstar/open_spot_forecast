"""Price series aligned to the 15-minute grid by timestamp (#33)."""

import math
import re
from collections.abc import Iterator
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, Mock, patch
from zoneinfo import ZoneInfo

import pytest

from homeassistant.util import dt as dt_util

from custom_components.open_spot_forecast.ml.predictor import SpotPricePredictor
from custom_components.open_spot_forecast.price_series import (
    align_to_grid,
    known_prices,
    same_prices,
)
from custom_components.open_spot_forecast.sensor import TodayMinSensor
from custom_components.open_spot_forecast.sensor_reader import SensorReader
from custom_components.open_spot_forecast.time_slots import (
    slot_index_in_day,
    slot_start_in_day,
    tomorrow_prices_complete,
)

CPH = ZoneInfo("Europe/Copenhagen")
DAY = date(2026, 9, 24)
SPRING = date(2026, 3, 29)
AUTUMN = date(2026, 10, 25)
INTEGRATION = Path(__file__).parents[1] / "custom_components" / "open_spot_forecast"

pytestmark = pytest.mark.usefixtures("copenhagen_time_zone")


def _quarter(day: date, index: int, price: float) -> tuple[datetime, None, float]:
    """A 15-minute sample without an end, starting at slot ``index``."""
    return slot_start_in_day(day, index), None, price


# --- align_to_grid ------------------------------------------------------------------


def test_one_missing_slot_keeps_every_other_slot_at_its_time() -> None:
    """A missing slot is filled; nothing after it shifts."""
    samples = [_quarter(DAY, i, float(i)) for i in range(96) if i != 10]

    values = align_to_grid(samples, DAY)

    assert len(values) == 96
    assert values[9:12] == pytest.approx([9.0, 9.0, 11.0])
    assert values[95] == pytest.approx(95.0)


@pytest.mark.parametrize(("day", "hours"), [(DAY, 24), (SPRING, 23), (AUTUMN, 25)])
def test_hourly_input_expands_to_four_slots(day: date, hours: int) -> None:
    """Hourly prices fill their hour's four slots, also on DST days."""
    samples = [
        (slot_start_in_day(day, hour, interval_minutes=60), None, float(hour))
        for hour in range(hours)
    ]

    values = align_to_grid(samples, day)

    assert values == pytest.approx([float(h) for h in range(hours) for _ in range(4)])


def test_sample_end_sets_its_duration() -> None:
    """An explicit end wins over the inferred resolution."""
    start = slot_start_in_day(DAY, 8)
    values = align_to_grid([(start, start + timedelta(hours=1), 3.0)], DAY)

    assert values[8:12] == pytest.approx([3.0] * 4)
    assert values[12] is None


def test_gaps_up_to_the_limit_are_filled_longer_ones_stay_missing() -> None:
    """A 4-slot gap takes the earlier price; a 5-slot gap stays None."""
    samples = [
        _quarter(DAY, i, 1.0) for i in range(96) if not (20 <= i < 24 or 50 <= i < 55)
    ]

    values = align_to_grid(samples, DAY)

    assert values[20:24] == pytest.approx([1.0] * 4)
    assert values[50:55] == [None] * 5


def test_leading_and_trailing_slots_are_not_invented() -> None:
    """A partial day is not padded before its first or after its last price."""
    samples = [_quarter(DAY, i, 2.0) for i in range(3, 50)]

    values = align_to_grid(samples, DAY)

    assert values[:3] == [None] * 3
    assert values[50:] == [None] * 46


def test_samples_outside_the_day_and_non_finite_prices_are_ignored() -> None:
    """Only this day's finite prices land on its grid; unordered input is fine."""
    samples = [
        _quarter(DAY, 3, 3.0),
        _quarter(DAY, 2, 2.0),
        _quarter(DAY, 1, math.nan),
        _quarter(DAY - timedelta(days=1), 5, 9.0),
        _quarter(DAY + timedelta(days=1), 5, 9.0),
        _quarter(DAY, 0, 1.0),
    ]

    values = align_to_grid(samples, DAY)

    assert values[:4] == pytest.approx([1.0, 1.0, 2.0, 3.0])  # NaN slot is a gap
    assert known_prices(values) == pytest.approx([1.0, 1.0, 2.0, 3.0])
    assert align_to_grid([], DAY) == []
    assert align_to_grid([_quarter(DAY + timedelta(days=1), 0, 1.0)], DAY) == []


def test_known_and_same_prices() -> None:
    """Missing slots are skipped and compared as missing."""
    assert known_prices([None, 1.0, None, 0.0]) == pytest.approx([1.0, 0.0])
    assert same_prices([1.0, None], [1.0 + 1e-12, None]) is True
    assert same_prices([1.0, None], [1.0, 2.0]) is False
    assert same_prices([1.0], [1.0, None]) is False


# --- Reader -------------------------------------------------------------------------


def test_reader_places_items_by_timestamp_not_position() -> None:
    """One missing item no longer shifts the rest of the day by a slot."""
    start = dt_util.start_of_local_day()
    items = [
        {"price": float(i), "start": (start + timedelta(minutes=15 * i)).isoformat()}
        for i in range(96)
        if i != 10
    ]
    hass = Mock()
    hass.states.get.return_value = Mock(state="1.0", attributes={"prices": items})

    today = SensorReader(hass).read_stromligning_sensor("sensor.strom")["today"]

    assert today[11] == pytest.approx(11.0)  # positionally it was item 12's price
    assert today[95] == pytest.approx(95.0)


# --- Downstream: completeness, sensors, training, self-learning ---------------------


def test_missing_slots_do_not_count_towards_complete() -> None:
    """Tomorrow with long gaps is not complete, however long the list is."""
    now = datetime(2026, 9, 24, 14, 0, tzinfo=CPH)

    assert tomorrow_prices_complete([1.0] * 96, now) is True
    assert tomorrow_prices_complete([1.0] * 90 + [None] * 6, now) is False


def test_price_sensor_ignores_missing_slots() -> None:
    """Min/max/mean sensors are computed over the known prices only."""
    sensor = TodayMinSensor.__new__(TodayMinSensor)
    sensor.api_data = {"stromligning_data": {"today": [None, 2.0, 1.5, None]}}
    sensor.precision = 2

    assert sensor.native_value == pytest.approx(1.5)


@pytest.fixture
def predictor(tmp_path: Path) -> Iterator[SpotPricePredictor]:
    """Predictor with a throwaway SQLite store."""
    hass = Mock()
    hass.config.path.return_value = str(tmp_path / ".storage")
    predictor = SpotPricePredictor(hass, "DK1")
    yield predictor
    predictor.storage.close()


def test_training_skips_missing_slots_but_keeps_later_times(
    predictor: SpotPricePredictor,
) -> None:
    """A long gap yields no training rows; the slots after it keep their times."""
    prices: list[float | None] = [1.0] * 96
    prices[40:46] = [None] * 6
    predictor.price_history = [{"date": "2026-09-24", "prices": prices}]

    rows, features = predictor.get_all_historical_prices()

    assert len(rows) == len(features) == 90
    assert features[40]["start"] == "2026-09-24T11:30:00+02:00"


def test_catch_up_learning_skips_missing_slots(predictor: SpotPricePredictor) -> None:
    """Catch-up has nothing to learn from a missing slot."""
    prices: list[float | None] = [10.0] * 96
    prices[40] = None
    predictor.price_history = [{"date": "2026-09-24", "prices": prices}]
    predictor.store_prediction_for_learning("2026-09-24T10:00:00+02:00", 11.0, 0.8)
    predictor.store_prediction_for_learning("2026-09-24T10:15:00+02:00", 11.0, 0.8)

    assert predictor.catch_up_learning() == 1
    assert predictor.storage.count_predictions() == 1


def test_changed_gap_is_new_training_data(predictor: SpotPricePredictor) -> None:
    """A slot filled in later counts as changed prices; identical ones don't."""
    with_gap: list[float | None] = [1.0] * 96
    with_gap[10:20] = [None] * 10

    predictor.record_training_prices(with_gap, "2026-09-24")
    first_update = predictor._prices_updated_at
    predictor.record_training_prices(list(with_gap), "2026-09-24")
    assert predictor._prices_updated_at == first_update

    predictor.record_training_prices([1.0] * 96, "2026-09-24")
    assert predictor._prices_updated_at != first_update


@pytest.mark.asyncio
@pytest.mark.parametrize("missing", [False, True])
async def test_quarter_update_skips_a_missing_slot(
    setup_entry: Any, missing: bool
) -> None:
    """End to end: self-learning is skipped when the current slot has no price."""
    now = datetime(2026, 9, 24, 10, 20, tzinfo=CPH)
    index = slot_index_in_day(now)
    today: list[float | None] = [1.0 + i for i in range(96)]
    if missing:
        today[index] = None
    reader = Mock()
    # Consumer prices are displayed; the model learns the spot prices
    reader.read_stromligning_sensor.return_value = {
        "today": [1.0] * 96,
        "tomorrow": [],
        "raw_today": [],
        "raw_tomorrow": [],
    }
    reader.read_spot_prices.return_value = {
        "today": today,
        "tomorrow": [],
        "raw_today": [],
        "raw_tomorrow": [],
    }
    reader.read_weather_sensors.return_value = {"temperature": 12.0}
    ml_predictor = Mock()
    ml_predictor._load_learning_data = AsyncMock()
    ml_predictor.save_learning_data = AsyncMock()
    ml_predictor.predictions = []
    ml_predictor.learn_from_actual_price.return_value = False

    _api_data, callbacks = await setup_entry(reader, ml_predictor)
    with patch("homeassistant.util.dt.now", return_value=now):
        await callbacks["new_quarter"](now)

    if missing:
        ml_predictor.learn_from_actual_price.assert_not_called()
    else:
        ml_predictor.learn_from_actual_price.assert_called_once_with(
            "2026-09-24T10:15:00+02:00", today[index]
        )


def test_no_resolution_detection_by_list_length() -> None:
    """Hourly vs 15-minute is never guessed from a price list's length."""
    guess = re.compile(r"intervals_per_hour|len\(\w*prices\w*\)\s*>\s*2[45]")
    sources = [
        INTEGRATION / "__init__.py",
        INTEGRATION / "updater.py",
        *sorted((INTEGRATION / "ml").glob("*.py")),
    ]
    offenders = [
        f"{path.name}:{number}"
        for path in sources
        for number, line in enumerate(path.read_text().splitlines(), 1)
        if guess.search(line)
    ]
    assert offenders == []
