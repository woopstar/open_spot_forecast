"""DST-safe slot timestamps on the Europe/Copenhagen change days (#19).

2026-03-29 lasts 23 hours (92 slots: 02:00-02:59 is skipped) and 2026-10-25
lasts 25 hours (100 slots: 02:00-02:59 happens twice, first at +02:00, then
at +01:00).
"""

import re
from collections.abc import Callable, Iterator
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock, patch
from zoneinfo import ZoneInfo

import pytest

from homeassistant.util import dt as dt_util

from custom_components.open_spot_forecast import async_setup_entry
from custom_components.open_spot_forecast.const import (
    CONF_ENABLE_ML_PREDICTION,
    CONF_REGION,
    CONF_STROMLIGNING_SENSOR,
    CONF_TEMPERATURE_SENSOR,
)
from custom_components.open_spot_forecast.ml.predictor import SpotPricePredictor
from custom_components.open_spot_forecast.time_slots import (
    slot_index_in_day,
    slot_start_in_day,
    slots_in_local_day,
)

CPH = ZoneInfo("Europe/Copenhagen")
SPRING = date(2026, 3, 29)
AUTUMN = date(2026, 10, 25)
ML_DIR = Path(__file__).parents[1] / "custom_components" / "open_spot_forecast" / "ml"

pytestmark = pytest.mark.usefixtures("copenhagen_time_zone")


@pytest.fixture
def predictor(tmp_path: Path) -> Iterator[SpotPricePredictor]:
    """Predictor for DK1 (Europe/Copenhagen) with a throwaway SQLite store."""
    hass = Mock()
    hass.config.path.return_value = str(tmp_path / ".storage")
    predictor = SpotPricePredictor(hass, "DK1")
    yield predictor
    predictor.storage.close()


# --- Slot timestamps --------------------------------------------------------------


@pytest.mark.parametrize(
    ("day", "index", "expected"),
    [
        (SPRING, 7, "2026-03-29T01:45:00+01:00"),
        (SPRING, 8, "2026-03-29T03:00:00+02:00"),  # 02:00 does not exist
        (SPRING, 91, "2026-03-29T23:45:00+02:00"),
        (AUTUMN, 8, "2026-10-25T02:00:00+02:00"),  # first 02:00
        (AUTUMN, 12, "2026-10-25T02:00:00+01:00"),  # second 02:00
        (AUTUMN, 99, "2026-10-25T23:45:00+01:00"),
    ],
)
def test_slot_start_in_day(day: date, index: int, expected: str) -> None:
    """Slot starts step in UTC, so they get the real local time and offset."""
    assert slot_start_in_day(day, index, CPH).isoformat() == expected


@pytest.mark.parametrize("day", [SPRING, AUTUMN, date(2026, 9, 24)])
def test_slot_index_is_the_inverse_of_slot_start(day: date) -> None:
    """Every slot of a 92/96/100-slot day maps back to its own index."""
    count = slots_in_local_day(day, tz=CPH)
    for index in range(count):
        # Add the 7 minutes in UTC: wall-clock arithmetic resets fold=1
        moment = slot_start_in_day(day, index, CPH).astimezone(UTC) + timedelta(
            minutes=7
        )
        assert slot_index_in_day(moment, CPH) == index


# --- Training rows ------------------------------------------------------------------


def test_training_rows_have_real_local_times_on_dst_days(
    predictor: SpotPricePredictor,
) -> None:
    """The 92- and 100-slot days get correct wall-clock times and offsets."""
    predictor.price_history = [
        {"date": "2026-03-29", "prices": [1.0] * 92},
        {"date": "2026-10-25", "prices": [1.0] * 100},
    ]

    prices, features = predictor.get_all_historical_prices()

    spring, autumn = features[:92], features[92:]
    assert len(prices) == 192
    assert spring[8]["start"] == "2026-03-29T03:00:00+02:00"
    assert spring[-1]["start"] == "2026-03-29T23:45:00+02:00"
    assert 2 not in {feature["hour"] for feature in spring}
    assert [feature["start"] for feature in autumn[8:13:4]] == [
        "2026-10-25T02:00:00+02:00",
        "2026-10-25T02:00:00+01:00",
    ]
    assert autumn[-1]["start"] == "2026-10-25T23:45:00+01:00"
    assert sum(1 for feature in autumn if feature["hour"] == 2) == 8


# --- Self-learning --------------------------------------------------------------------


def _store(predictor: SpotPricePredictor, start: str, price: float) -> None:
    """Store one pending prediction for ``start``."""
    predictor.store_prediction_for_learning(start, price, 0.8)


def test_learning_matches_only_the_right_pass_of_the_repeated_hour(
    predictor: SpotPricePredictor,
) -> None:
    """02:15 happens twice on the fall-back day; each pass learns on its own."""
    _store(predictor, "2026-10-25T02:15:00+02:00", 5.0)
    _store(predictor, "2026-10-25T02:15:00+01:00", 7.0)

    assert predictor.learn_from_actual_price("2026-10-25T02:15:00+01:00", 6.0)

    metrics = predictor.error_metrics[2 * 4 + 1]
    assert metrics["count"] == 1
    assert metrics["errors"] == [pytest.approx(1.0)]
    assert predictor.storage.count_predictions() == 1


def test_catch_up_learning_uses_real_slot_times(predictor: SpotPricePredictor) -> None:
    """Replayed prices pair each DST-day slot with the prediction for that instant."""
    predictor.price_history = [
        {"date": "2026-03-29", "prices": [10.0 + i for i in range(92)]},
        {"date": "2026-10-25", "prices": [10.0 + i for i in range(100)]},
    ]
    _store(predictor, "2026-03-29T03:00:00+02:00", 20.0)  # slot 8: actual 18
    _store(predictor, "2026-10-25T02:15:00+02:00", 20.0)  # slot 9: actual 19
    _store(predictor, "2026-10-25T02:15:00+01:00", 20.0)  # slot 13: actual 23

    assert predictor.catch_up_learning() == 3

    assert predictor.error_metrics[3 * 4]["errors"] == [pytest.approx(2.0)]
    assert sorted(predictor.error_metrics[2 * 4 + 1]["errors"]) == [
        pytest.approx(-3.0),
        pytest.approx(1.0),
    ]
    assert predictor.storage.count_predictions() == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("now", "expected_timestamp", "expected_index"),
    [
        (datetime(2026, 3, 29, 3, 20, tzinfo=CPH), "2026-03-29T03:15:00+02:00", 9),
        (
            datetime(2026, 10, 25, 2, 20, tzinfo=CPH),
            "2026-10-25T02:15:00+02:00",
            9,
        ),
        (
            datetime(2026, 10, 25, 2, 20, tzinfo=CPH, fold=1),
            "2026-10-25T02:15:00+01:00",
            13,
        ),
        (datetime(2026, 10, 25, 23, 50, tzinfo=CPH), "2026-10-25T23:45:00+01:00", 99),
    ],
)
async def test_quarter_update_learns_from_the_price_of_the_current_slot(
    tmp_path: Path, now: datetime, expected_timestamp: str, expected_index: int
) -> None:
    """End to end: the 15-minute update indexes today's prices by real instant."""
    day = now.date()
    today = [10.0 + i for i in range(slots_in_local_day(day, tz=CPH))]

    async def run_inline(func: Callable[..., Any], *args: Any) -> Any:
        return func(*args)

    hass = Mock()
    hass.data = {}
    hass.config.path.return_value = str(tmp_path / ".storage")
    hass.async_add_executor_job = run_inline
    hass.config_entries.async_forward_entry_setups = AsyncMock()
    entry = MagicMock()
    entry.entry_id = "test"
    entry.options = {}
    entry.data = {
        CONF_REGION: "DK1",
        CONF_ENABLE_ML_PREDICTION: True,
        CONF_STROMLIGNING_SENSOR: "sensor.stromligning_current_price_vat",
        CONF_TEMPERATURE_SENSOR: "sensor.outdoor_temperature",
    }
    reader = Mock()
    reader.read_stromligning_sensor.return_value = {
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
    callbacks: dict[str, Callable[..., Any]] = {}

    def track_time_change(_hass: Any, action: Any, **_kw: Any) -> Mock:
        callbacks[action.__name__] = action
        return Mock()

    module = "custom_components.open_spot_forecast"
    with (
        patch(f"{module}.async_get_integration", new=AsyncMock()),
        patch(f"{module}.SensorReader", return_value=reader),
        patch(f"{module}.SpotPricePredictor", return_value=ml_predictor),
        patch(f"{module}._fetch_nordpool_prognoses", new=AsyncMock(return_value=[])),
        patch(f"{module}.async_track_time_change", side_effect=track_time_change),
        patch(f"{module}.tomorrow_prices.async_track_point_in_utc_time"),
        patch(f"{module}.async_dispatcher_send"),
    ):
        assert await async_setup_entry(hass, entry) is True
        with patch("homeassistant.util.dt.now", return_value=now):
            await callbacks["new_quarter"](now)

    ml_predictor.learn_from_actual_price.assert_called_once_with(
        expected_timestamp, today[expected_index]
    )


# --- Clock -----------------------------------------------------------------------------


def test_no_naive_now_in_ml() -> None:
    """ml/ reads the clock through Home Assistant's time zone, never naively."""
    naive = re.compile(r"datetime\.now\(\)|date\.today\(\)")
    offenders = [
        f"{path.name}:{number}"
        for path in sorted(ML_DIR.glob("*.py"))
        for number, line in enumerate(path.read_text().splitlines(), 1)
        if naive.search(line)
        and not line.lstrip().startswith(("#", '"', "``"))
        and "``" not in line
    ]
    assert offenders == []


def test_heuristic_confidence_drops_with_days_ahead(
    predictor: SpotPricePredictor,
) -> None:
    """The documented days_ahead penalty applies (it silently never did)."""
    now = dt_util.now()
    feature = {"wind_speed_mean": 5.0, "solar_generation": 100.0}

    # The current slot started a few minutes ago: no bonus for "negative" days
    today = predictor._estimate_confidence(
        feature | {"start": (now - timedelta(minutes=10)).isoformat()}
    )
    later = predictor._estimate_confidence(
        feature | {"start": (now + timedelta(days=3, hours=1)).isoformat()}
    )

    assert today == pytest.approx(0.8)
    assert later == pytest.approx(0.8 - 3 * 0.05)
