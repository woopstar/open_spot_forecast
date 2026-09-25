"""Weather snapshots are keyed by their UTC slot start and found again (#59).

Snapshots were written as naive (later: offset) local ISO strings, while the
single-row lookups compared them against SQLite ``datetime()`` results (UTC,
space separator), so ``find_weather_for_timestamp`` and
``find_nordpool_for_timestamp`` never matched a row on the same date.
"""

import logging
import sqlite3
from collections.abc import Iterator
from contextlib import closing
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from unittest.mock import Mock
from zoneinfo import ZoneInfo

import pytest

from homeassistant.util import dt as dt_util

from custom_components.open_spot_forecast.ml.predictor import SpotPricePredictor
from custom_components.open_spot_forecast.ml.storage import LearningStorage
from custom_components.open_spot_forecast.ml.training_inputs import TrainingInputs
from custom_components.open_spot_forecast.ml.weather_migration import (
    WEATHER_UTC_SCHEMA_VERSION,
)
from custom_components.open_spot_forecast.time_slots import (
    parse_utc,
    slot_start_in_day,
    utc_slot_key,
)

CPH = ZoneInfo("Europe/Copenhagen")

# Home Assistant's time zone is Europe/Copenhagen, as in production
pytestmark = pytest.mark.usefixtures("copenhagen_time_zone")


def _hass(tmp_path: Path) -> Mock:
    hass = Mock()
    hass.config.path.return_value = str(tmp_path / ".storage")
    return hass


@pytest.fixture
def storage(tmp_path: Path) -> Iterator[LearningStorage]:
    store = LearningStorage(_hass(tmp_path), "DK1")
    yield store
    store.close()


def _snapshot(storage: LearningStorage, moment: datetime, temperature: float) -> None:
    """Store a snapshot the way the 15-minute update does."""
    storage.insert_weather_snapshot(
        utc_slot_key(moment), temperature, 5.0, 180.0, 50.0, 80.0, None
    )


# --- The stored format ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("moment", "key"),
    [
        (datetime(2026, 9, 24, 10, 0, 1, 123456, tzinfo=CPH), "2026-09-24T08:00:00Z"),
        (datetime(2026, 9, 24, 10, 14, 59, tzinfo=CPH), "2026-09-24T08:00:00Z"),
        (datetime(2026, 9, 24, 8, 15, tzinfo=UTC), "2026-09-24T08:15:00Z"),
        # Both passes of the repeated fall-back hour get their own key
        (datetime(2026, 10, 25, 2, 15, fold=0, tzinfo=CPH), "2026-10-25T00:15:00Z"),
        (datetime(2026, 10, 25, 2, 15, fold=1, tzinfo=CPH), "2026-10-25T01:15:00Z"),
    ],
)
def test_utc_slot_key(moment: datetime, key: str) -> None:
    assert utc_slot_key(moment) == key


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("2026-09-24T08:00:00Z", datetime(2026, 9, 24, 8, tzinfo=UTC)),
        ("2026-09-24T10:00:00+02:00", datetime(2026, 9, 24, 8, tzinfo=UTC)),
        # Naive = Home Assistant local time
        ("2026-09-24T10:00:00", datetime(2026, 9, 24, 8, tzinfo=UTC)),
        ("garbage", None),
        (None, None),
    ],
)
def test_parse_utc(value: str | None, expected: datetime | None) -> None:
    assert parse_utc(value) == expected


# --- Write → lookup ------------------------------------------------------------------


@pytest.mark.parametrize(
    "query",
    [
        "2026-09-24T10:00:00+02:00",  # the training/learning slot start
        "2026-09-24T08:00:00Z",  # UTC
        "2026-09-24T08:00:00+00:00",
        "2026-09-24T10:00:00",  # naive local
        "2026-09-24T10:20:00+02:00",  # within 30 minutes
    ],
)
def test_a_written_snapshot_is_found_for_its_slot(
    storage: LearningStorage, query: str
) -> None:
    # Written at 10:00:01 local by the 15-minute update
    _snapshot(storage, datetime(2026, 9, 24, 10, 0, 1, tzinfo=CPH), 12.5)

    weather = storage.find_weather_for_timestamp(query)

    assert weather is not None
    assert weather["temperature"] == pytest.approx(12.5)


def test_lookup_picks_the_closest_snapshot_within_30_minutes(
    storage: LearningStorage,
) -> None:
    for minute, temperature in ((0, 10.0), (15, 11.0), (30, 12.0)):
        _snapshot(storage, datetime(2026, 9, 24, 10, minute, tzinfo=CPH), temperature)

    closest = storage.find_weather_for_timestamp("2026-09-24T10:16:00+02:00")
    assert closest is not None
    assert closest["temperature"] == pytest.approx(11.0)
    assert storage.find_weather_for_timestamp("2026-09-24T11:01:00+02:00") is None
    assert storage.find_weather_for_timestamp("2026-09-24T09:29:00+02:00") is None
    assert storage.find_weather_for_timestamp("not a time") is None


@pytest.mark.parametrize(
    ("day", "index"),
    [
        # Spring forward: slot 8 is 03:00+02:00
        (date(2026, 3, 29), 8),
        # Fall back: slots 9 and 13 are both 02:15, at +02:00 and +01:00
        (date(2026, 10, 25), 9),
        (date(2026, 10, 25), 13),
    ],
)
def test_dst_day_slots_find_their_own_snapshot(
    storage: LearningStorage, day: date, index: int
) -> None:
    for slot in range(8, 16):
        _snapshot(storage, slot_start_in_day(day, slot), float(slot))

    weather = storage.find_weather_for_timestamp(
        slot_start_in_day(day, index).isoformat()
    )

    assert weather is not None
    assert weather["temperature"] == pytest.approx(float(index))


def test_nordpool_rows_are_found_on_their_own_date(storage: LearningStorage) -> None:
    """The Nordpool backfill's "already stored?" check found nothing before."""
    storage.insert_nordpool_prognosis("2026-09-24T12:00:00Z", 4000.0, 1.0, 2.0, 3.0)

    for query in ("2026-09-24T12:00:00Z", "2026-09-24T14:00:00+02:00"):
        row = storage.find_nordpool_for_timestamp(query)
        assert row is not None
        assert row["consumption"] == pytest.approx(4000.0)
    assert storage.find_nordpool_for_timestamp("2026-09-24T13:30:00Z") is None
    assert storage.find_nordpool_for_timestamp("") is None


def test_training_inputs_match_new_and_old_formats_alike() -> None:
    """Training (TrainingInputs, #61) already matched by UTC epoch.

    The new format gives it exactly the same inputs as the old local formats,
    so the model's training rows and holdout error are unchanged.
    """
    slot = datetime(2026, 9, 24, 10, 0, tzinfo=CPH)
    formats = [
        "2026-09-24T08:00:00Z",
        "2026-09-24T10:00:01.123456+02:00",
        "2026-09-24T10:00:01.123456",
    ]
    inputs = [
        TrainingInputs([{"timestamp": ts, "temperature": 12.5}], [], CPH).for_slot(slot)
        for ts in formats
    ]

    assert inputs[0].temperature == pytest.approx(12.5)
    assert inputs[1:] == [inputs[0], inputs[0]]


# --- Pruning -------------------------------------------------------------------------


def test_pruning_compares_instants(storage: LearningStorage) -> None:
    now = dt_util.utcnow()
    _snapshot(storage, now - timedelta(days=31), 1.0)
    _snapshot(storage, now - timedelta(days=29), 2.0)
    storage.insert_nordpool_prognosis(
        (now - timedelta(days=31)).strftime("%Y-%m-%dT%H:00:00Z"), 1.0, 0, 0, 0
    )
    storage.insert_nordpool_prognosis(
        (now - timedelta(days=29)).strftime("%Y-%m-%dT%H:00:00Z"), 2.0, 0, 0, 0
    )

    assert storage.delete_old_weather(30) == 1
    assert storage.delete_old_nordpool(30) == 1
    temperatures = [row["temperature"] for row in storage.load_weather_history()]
    consumption = [row["consumption"] for row in storage.load_nordpool_history()]
    assert temperatures == pytest.approx([2.0])
    assert consumption == pytest.approx([2.0])


# --- Self-learning uses the lookup ---------------------------------------------------


def test_learning_records_the_forecast_weather_error(tmp_path: Path) -> None:
    predictor = SpotPricePredictor(_hass(tmp_path), "DK1")
    try:
        slot = datetime(2026, 9, 24, 10, 0, tzinfo=CPH)
        predictor.storage.insert_prediction(
            slot.isoformat(), 0.5, 0.8, 10, 0, dt_util.now().isoformat(), 15.0, 6.0
        )
        _snapshot(predictor.storage, slot + timedelta(seconds=1), 12.0)

        assert predictor.learn_from_actual_price(slot.isoformat(), 0.4)

        metrics = predictor.error_metrics[40]
        assert metrics["forecast_temp_errors"] == pytest.approx([3.0])
        assert metrics["forecast_wind_errors"] == pytest.approx([1.0])
    finally:
        predictor.storage.close()


# --- Migration -----------------------------------------------------------------------


def _schema_version(db_path: Path) -> int:
    with closing(sqlite3.connect(db_path)) as conn:
        row = conn.execute(
            "SELECT value FROM meta WHERE key = 'schema_version'"
        ).fetchone()
    return int(row[0])


def test_existing_snapshots_are_rewritten_as_utc_slot_keys(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    storage = LearningStorage(_hass(tmp_path), "DK1")
    db_path = storage.db_path
    storage.close()
    with closing(sqlite3.connect(db_path)) as conn:
        conn.execute("UPDATE meta SET value = '6' WHERE key = 'schema_version'")
        conn.executemany(
            "INSERT INTO weather_history (timestamp, temperature, solar_power) "
            "VALUES (?, ?, ?)",
            [
                # Naive local, as written before #17
                ("2026-09-24T10:00:01.123456", 10.0, 100.0),
                # Local with offset and microseconds, as written since #17
                ("2026-09-24T10:15:01.654321+02:00", 11.0, None),
                # A later snapshot in the same slot is merged away
                ("2026-09-24T10:16:30+02:00", 99.0, None),
                # Naive in the repeated fall-back hour: the first pass
                ("2026-10-25T02:15:01", 13.0, None),
                ("garbage", 0.0, None),
            ],
        )
        conn.commit()

    with caplog.at_level(logging.INFO):
        migrated = LearningStorage(_hass(tmp_path), "DK1")
    try:
        rows = migrated.load_weather_history()
        assert [(row["timestamp"], row["temperature"]) for row in rows] == [
            ("2026-09-24T08:00:00Z", pytest.approx(10.0)),
            ("2026-09-24T08:15:00Z", pytest.approx(11.0)),
            ("2026-10-25T00:15:00Z", pytest.approx(13.0)),
        ]
        weather = migrated.find_weather_for_timestamp("2026-09-24T10:00:00+02:00")
        assert weather is not None
        assert weather["solar_power"] == pytest.approx(100.0)
        assert _schema_version(db_path) == WEATHER_UTC_SCHEMA_VERSION
        assert "rewrote 3, merged 1" in caplog.text
        assert "dropped 1 unreadable" in caplog.text
    finally:
        migrated.close()

    # Runs once: a snapshot written afterwards is left alone
    caplog.clear()
    with closing(sqlite3.connect(db_path)) as conn:
        conn.execute(
            "INSERT INTO weather_history (timestamp, temperature) VALUES (?, ?)",
            ("2026-09-24T10:30:00+02:00", 5.0),
        )
        conn.commit()
    with caplog.at_level(logging.INFO):
        reopened = LearningStorage(_hass(tmp_path), "DK1")
    reopened.close()
    assert "Weather snapshots are now keyed" not in caplog.text


def test_new_database_starts_at_the_utc_weather_schema(tmp_path: Path) -> None:
    storage = LearningStorage(_hass(tmp_path), "DK1")
    storage.close()

    assert _schema_version(storage.db_path) == WEATHER_UTC_SCHEMA_VERSION
