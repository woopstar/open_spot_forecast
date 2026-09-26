"""Every slot's history lookup finds its own row, and nothing else (#46).

``find_weather_for_timestamp`` and ``find_nordpool_for_timestamp`` compared the
stored ISO strings with SQLite ``datetime()`` results (UTC, space separator):
a row on the same date never matched, and a window that crossed midnight
matched every row of the earlier date. #59 compares UTC instants. These tests
sweep every slot of a normal and both DST days, with the neighbouring days
stored, in each query format training and self-learning use, and check that
no row outside a lookup's window comes back.
"""

from collections.abc import Iterator
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from unittest.mock import Mock

import pytest

from homeassistant.util import dt as dt_util

from custom_components.open_spot_forecast.ml.predictor import SpotPricePredictor
from custom_components.open_spot_forecast.ml.series_storage import OPENMETEO_WEATHER
from custom_components.open_spot_forecast.ml.storage import LearningStorage
from custom_components.open_spot_forecast.ml.zone_weather import zone_points
from custom_components.open_spot_forecast.time_slots import (
    UTC_KEY_FORMAT,
    floor_to_slot,
    local_midnight,
    slot_start_in_day,
    slots_in_local_day,
    utc_slot_key,
)

# Home Assistant's time zone is Europe/Copenhagen, as in production
pytestmark = pytest.mark.usefixtures("copenhagen_time_zone")

NORMAL_DAY = date(2026, 9, 24)  # 96 slots
SPRING_FORWARD = date(2026, 3, 29)  # 92 slots
FALL_BACK = date(2026, 10, 25)  # 100 slots, 02:00-02:59 twice
DAYS = [NORMAL_DAY, SPRING_FORWARD, FALL_BACK]

# How a slot start is passed to a lookup
AWARE_LOCAL = "aware local"  # 2026-09-24T10:00:00+02:00, training and learning
UTC_Z = "UTC"  # 2026-09-24T08:00:00Z, the stored key format
NAIVE_LOCAL = "naive local"  # 2026-09-24T10:00:00, Home Assistant local time
QUERY_FORMATS = [AWARE_LOCAL, UTC_Z, NAIVE_LOCAL]

_ORIGIN = datetime(2026, 1, 1, tzinfo=UTC)
_SLOT = timedelta(minutes=15)
_HOUR = timedelta(hours=1)


def _hass(tmp_path: Path) -> Mock:
    hass = Mock()
    hass.config.path.return_value = str(tmp_path / ".storage")
    return hass


@pytest.fixture
def storage(tmp_path: Path) -> Iterator[LearningStorage]:
    store = LearningStorage(_hass(tmp_path), "DK1")
    yield store
    store.close()


def _value(start: datetime) -> float:
    """Return a value unique to the slot (or hour) beginning at ``start``."""
    return (start - _ORIGIN) / _SLOT


def _starts(first_day: date, last_day: date, step: timedelta) -> list[datetime]:
    """Return every ``step`` from ``first_day``'s local midnight to ``last_day``'s end."""
    moment = local_midnight(first_day).astimezone(UTC)
    end = local_midnight(last_day + timedelta(days=1)).astimezone(UTC)
    starts = []
    while moment < end:
        starts.append(moment)
        moment += step
    return starts


def _snapshot(storage: LearningStorage, start: datetime, temperature: float) -> None:
    """Store a snapshot the way the 15-minute update does, a second into the slot."""
    storage.insert_weather_snapshot(
        utc_slot_key(start + timedelta(seconds=1)),
        temperature,
        5.0,
        180.0,
        50.0,
        80.0,
        None,
    )


def _prognosis(storage: LearningStorage, hour: datetime, consumption: float) -> None:
    """Store an hour's prognosis keyed as Nordpool publishes it (UTC, ``Z``)."""
    storage.insert_nordpool_prognosis(
        hour.astimezone(UTC).strftime(UTC_KEY_FORMAT), consumption, 1.0, 2.0, 3.0
    )


def _store_days_around(storage: LearningStorage, day: date) -> None:
    """Store every snapshot and hourly prognosis of ``day`` and both neighbours."""
    around = (day - timedelta(days=1), day + timedelta(days=1))
    for start in _starts(*around, _SLOT):
        _snapshot(storage, start, _value(start))
    for hour in _starts(*around, _HOUR):
        _prognosis(storage, hour, _value(hour))


def _query(start: datetime, query_format: str) -> str:
    """Return a slot start as an ISO string in ``query_format``."""
    if query_format == AWARE_LOCAL:
        return start.isoformat()
    if query_format == UTC_Z:
        return utc_slot_key(start)
    return start.replace(tzinfo=None).isoformat()


def _meant(start: datetime, query_format: str) -> datetime:
    """Return the instant a query for ``start`` names.

    A naive time in the repeated fall-back hour is ambiguous and reads as its
    first pass, like every naive timestamp Home Assistant parses.
    """
    return start.replace(fold=0) if query_format == NAIVE_LOCAL else start


# --- Every slot of a day finds its own row -------------------------------------------


@pytest.mark.parametrize("query_format", QUERY_FORMATS)
@pytest.mark.parametrize("day", DAYS, ids=str)
def test_every_slot_finds_its_own_weather_snapshot(
    storage: LearningStorage, day: date, query_format: str
) -> None:
    """All 92/96/100 slots, midnight included, next to the adjacent days' rows."""
    _store_days_around(storage, day)
    starts = [slot_start_in_day(day, i) for i in range(slots_in_local_day(day))]

    found = [
        storage.find_weather_for_timestamp(_query(start, query_format))
        for start in starts
    ]

    assert [None if w is None else w["temperature"] for w in found] == pytest.approx(
        [_value(_meant(start, query_format)) for start in starts]
    )


@pytest.mark.parametrize("query_format", QUERY_FORMATS)
@pytest.mark.parametrize("day", DAYS, ids=str)
def test_every_slot_finds_its_hours_nordpool_row(
    storage: LearningStorage, day: date, query_format: str
) -> None:
    """Each slot reads its own hour's row, as training does, never the next one."""
    _store_days_around(storage, day)
    starts = [slot_start_in_day(day, i) for i in range(slots_in_local_day(day))]

    found = [
        storage.find_nordpool_for_timestamp(_query(start, query_format))
        for start in starts
    ]

    assert [None if n is None else n["consumption"] for n in found] == pytest.approx(
        [_value(floor_to_slot(_meant(start, query_format), 60)) for start in starts]
    )


# --- Nothing outside the window ------------------------------------------------------


def test_weather_lookup_returns_nothing_beyond_30_minutes(
    storage: LearningStorage,
) -> None:
    """Only 2026-09-24 is stored, without 12:00-13:45 local."""
    for index in range(96):
        if not 48 <= index < 56:
            start = slot_start_in_day(NORMAL_DAY, index)
            _snapshot(storage, start, _value(start))

    misses = [
        # Another day at the same time
        "2026-09-23T12:00:00+02:00",
        "2026-09-25T12:00:00+02:00",
        # Near midnight, beyond 30 minutes from the first (00:00) or last
        # (23:45) row; the old string range matched every row of the day here
        "2026-09-23T23:15:00+02:00",
        "2026-09-25T00:20:00+02:00",
        "2026-09-25T00:40:00+02:00",
        "2026-09-24T22:20:00Z",
        "2026-09-25T00:20:00",
        # Inside the gap, over 30 minutes from 11:45 and 14:00
        "2026-09-24T12:45:00+02:00",
        "2026-09-24T11:00:00Z",
        "2026-09-24T13:15:00",
    ]

    assert [storage.find_weather_for_timestamp(query) for query in misses] == [
        None
    ] * len(misses)


@pytest.mark.parametrize(
    ("offset", "inside"),
    [
        (timedelta(minutes=30), True),
        (-timedelta(minutes=30), True),
        (timedelta(minutes=30, seconds=1), False),
        (-timedelta(minutes=30, seconds=1), False),
    ],
)
def test_weather_window_edges_are_exact(
    storage: LearningStorage, offset: timedelta, inside: bool
) -> None:
    """A snapshot exactly 30 minutes away is found, one a second further is not.

    The window was a float distance in julian days, so rounding decided which
    rows on its edge were found (4 of these 12 at +30 minutes).
    """
    starts = [slot_start_in_day(NORMAL_DAY, index) for index in range(0, 96, 8)]
    for start in starts:
        _snapshot(storage, start, _value(start))

    found = [
        storage.find_weather_for_timestamp((start + offset).isoformat())
        for start in starts
    ]

    if inside:
        assert [None if w is None else w["temperature"] for w in found] == (
            pytest.approx([_value(start) for start in starts])
        )
    else:
        assert found == [None] * len(starts)


def test_nordpool_lookup_never_returns_another_hours_row(
    storage: LearningStorage,
) -> None:
    """Only the UTC hours of 2026-09-24 are stored, without 12:00."""
    first = datetime(2026, 9, 24, tzinfo=UTC)
    for hour in range(24):
        if hour != 12:
            _prognosis(storage, first + hour * _HOUR, float(hour))

    misses = [
        # The missing hour is not filled from 11:00 or 13:00
        "2026-09-24T12:00:00Z",
        "2026-09-24T12:45:00Z",
        "2026-09-24T14:30:00+02:00",
        # Across midnight, the neighbouring day's first or last row is not used
        "2026-09-23T23:30:00Z",
        "2026-09-24T01:45:00+02:00",
        "2026-09-25T00:00:00Z",
        "2026-09-25T02:00:00",
        # Another day at the same time
        "2026-09-23T13:00:00Z",
        "2026-09-25T13:00:00Z",
    ]
    hits = {
        "2026-09-24T11:59:59Z": 11.0,
        "2026-09-24T15:00:00+02:00": 13.0,
        "2026-09-24T00:00:00Z": 0.0,
        "2026-09-24T23:59:59Z": 23.0,
    }

    assert [storage.find_nordpool_for_timestamp(query) for query in misses] == [
        None
    ] * len(misses)
    for query, consumption in hits.items():
        row = storage.find_nordpool_for_timestamp(query)
        assert row is not None, query
        assert row["consumption"] == pytest.approx(consumption)


# --- Training and self-learning read the same rows -----------------------------------


@pytest.mark.parametrize("day", DAYS, ids=str)
def test_training_rows_carry_each_slots_stored_history(
    tmp_path: Path, day: date
) -> None:
    """Every training row gets its slot's zone weather and its hour's prognosis.

    The local snapshots are stored too, but are not training inputs (#23).
    """
    predictor = SpotPricePredictor(_hass(tmp_path), "DK1")
    try:
        _store_days_around(predictor.storage, day)
        around = (day - timedelta(days=1), day + timedelta(days=1))
        predictor.storage.upsert_series(
            OPENMETEO_WEATHER,
            [
                {
                    "timestamp": start.isoformat(),
                    "point": point,
                    "wind_80m": _value(start),
                }
                | {
                    "temperature": 1.0,
                    "irradiance": 0.0,
                    "pressure": 1.0,
                    "humidity": 1.0,
                }
                for start in _starts(*around, _SLOT)
                for point in zone_points("DK1")
            ],
        )
        count = slots_in_local_day(day)
        predictor.price_history = [{"date": day.isoformat(), "prices": [1.0] * count}]

        _, features = predictor.get_all_historical_prices()

        starts = [slot_start_in_day(day, i) for i in range(count)]
        assert [row["zone_wind"] for row in features] == pytest.approx(
            [_value(start) for start in starts]
        )
        assert {row["temperature"] for row in features} == {None}
        assert [row["consumption_forecast"] for row in features] == pytest.approx(
            [_value(floor_to_slot(start, 60)) for start in starts]
        )
    finally:
        predictor.storage.close()


@pytest.mark.parametrize("index", [0, 95])
def test_learning_measures_the_forecast_error_near_midnight(
    tmp_path: Path, index: int
) -> None:
    """The forecast is compared with the slot's own snapshot, not its neighbours'."""
    predictor = SpotPricePredictor(_hass(tmp_path), "DK1")
    try:
        slot = slot_start_in_day(NORMAL_DAY, index)
        _snapshot(predictor.storage, slot - _SLOT, 30.0)
        _snapshot(predictor.storage, slot, 12.0)
        _snapshot(predictor.storage, slot + _SLOT, 30.0)
        predictor.storage.insert_prediction(
            slot.isoformat(),
            0.5,
            0.8,
            slot.hour,
            slot.minute,
            dt_util.now().isoformat(),
            15.0,
            6.0,
        )

        assert predictor.learn_from_actual_price(slot.isoformat(), 0.4)

        metrics = predictor.error_metrics[index]
        assert metrics["forecast_temp_errors"] == pytest.approx([3.0])
        assert metrics["forecast_wind_errors"] == pytest.approx([1.0])
    finally:
        predictor.storage.close()
