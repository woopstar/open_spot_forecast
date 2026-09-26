"""Generic time-series table access (#32): completeness, upserts, state."""

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import Mock

import pytest

from custom_components.open_spot_forecast.ml.series_storage import (
    NORDPOOL_PROGNOSES,
    SeriesSpec,
)
from custom_components.open_spot_forecast.ml.storage import LearningStorage

T0 = datetime(2026, 9, 24, 10, tzinfo=UTC)
HOUR = timedelta(hours=1)
KEYED = SeriesSpec("keyed", "keyed", ("value",), 15, key_column="point")


@pytest.fixture
def storage(tmp_path: Path) -> Iterator[LearningStorage]:
    hass = Mock()
    hass.config.path.return_value = str(tmp_path / ".storage")
    store = LearningStorage(hass, "DK1")
    store._ensure_conn().execute(
        "CREATE TABLE keyed (timestamp TEXT, point TEXT, value REAL, "
        "PRIMARY KEY (timestamp, point))"
    )
    yield store
    store.close()


def _row(moment: datetime, **values: float | None) -> dict:
    row: dict = {"timestamp": moment.isoformat(), "consumption": 4000.0}
    row.update({"solar": 1.0, "wind_offshore": 2.0, "wind_onshore": 3.0})
    row.update(values)
    return row


def test_only_complete_rows_count_as_stored(storage: LearningStorage) -> None:
    storage.upsert_series(
        NORDPOOL_PROGNOSES, [_row(T0), _row(T0 + HOUR, solar=None), _row(T0 + 3 * HOUR)]
    )

    stored = storage.series_timestamps(NORDPOOL_PROGNOSES, T0, T0 + 3 * HOUR)

    assert stored == [T0]


def test_a_partial_response_never_erases_stored_values(
    storage: LearningStorage,
) -> None:
    storage.upsert_series(NORDPOOL_PROGNOSES, [_row(T0)])

    changed = storage.upsert_series(NORDPOOL_PROGNOSES, [_row(T0, solar=None)])

    assert changed is False
    (row,) = storage.load_series(NORDPOOL_PROGNOSES, T0, T0 + HOUR)
    assert row["solar"] == pytest.approx(1.0)


def test_upserts_report_changes_and_move_last_data_write(
    storage: LearningStorage,
) -> None:
    assert storage.upsert_series(NORDPOOL_PROGNOSES, [_row(T0)]) is True
    storage.last_data_write = None

    assert storage.upsert_series(NORDPOOL_PROGNOSES, [_row(T0)]) is False
    assert storage.last_data_write is None

    assert storage.upsert_series(NORDPOOL_PROGNOSES, [_row(T0, solar=9.0)]) is True
    assert storage.last_data_write is not None


def test_timestamps_are_stored_as_utc_keys(storage: LearningStorage) -> None:
    storage.upsert_series(
        NORDPOOL_PROGNOSES,
        [
            _row(T0).copy() | {"timestamp": "2026-09-24T12:00:00+02:00"},
            {"timestamp": "garbage", "consumption": 1.0},
        ],
    )

    rows = storage.load_series(NORDPOOL_PROGNOSES, T0, T0 + HOUR)

    assert [row["timestamp"] for row in rows] == ["2026-09-24T10:00:00Z"]
    assert storage.upsert_series(NORDPOOL_PROGNOSES, []) is False


def test_keyed_points_need_a_row_for_every_key(storage: LearningStorage) -> None:
    quarter = timedelta(minutes=15)
    storage.upsert_series(
        KEYED,
        [
            {"timestamp": T0.isoformat(), "point": "a", "value": 1.0},
            {"timestamp": T0.isoformat(), "point": "b", "value": 2.0},
            {"timestamp": (T0 + quarter).isoformat(), "point": "a", "value": 1.0},
        ],
    )

    assert storage.series_timestamps(KEYED, T0, T0 + HOUR, ("a", "b")) == [T0]
    assert storage.series_timestamps(KEYED, T0, T0 + HOUR, ("a",)) == [T0, T0 + quarter]


def test_prune_and_source_state(storage: LearningStorage) -> None:
    storage.upsert_series(NORDPOOL_PROGNOSES, [_row(T0), _row(T0 + 2 * HOUR)])

    assert storage.prune_series(NORDPOOL_PROGNOSES, T0 + HOUR) == 1
    assert storage.load_source_state("nordpool") is None
    storage.save_source_state("nordpool", "[]")
    assert storage.load_source_state("nordpool") == "[]"
