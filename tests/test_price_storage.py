"""The model's price history stored per UTC slot (#24)."""

import json
import logging
import sqlite3
from collections.abc import Iterator
from contextlib import closing
from datetime import date
from pathlib import Path
from unittest.mock import Mock

import pytest

from custom_components.open_spot_forecast.ml.price_storage import (
    PRICE_ROWS_SCHEMA_VERSION,
    day_rows,
    delete_days_before,
    rows_to_days,
    write_changed_days,
)
from custom_components.open_spot_forecast.ml.storage import LearningStorage

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


def _day(day: str, slots: int, first: float = 0.0) -> dict:
    return {"date": day, "prices": [first + i for i in range(slots)]}


@pytest.mark.parametrize(
    ("day", "slots"), [("2026-09-24", 96), ("2026-03-29", 92), ("2026-10-25", 100)]
)
def test_days_round_trip_through_utc_rows(day: str, slots: int) -> None:
    """Every slot, incl. both passes of the repeated hour, keeps its place."""
    entry = _day(day, slots)
    entry["prices"][5] = None

    rows = day_rows(entry)

    assert len(rows) == slots - 1
    assert rows[0][0].endswith("Z")
    assert rows_to_days(rows) == [entry]


def test_rows_of_several_days_group_by_local_day() -> None:
    rows = day_rows(_day("2026-09-25", 96, 100.0)) + day_rows(_day("2026-09-24", 96))
    rows.append(("garbage", 1.0))

    days = rows_to_days(rows)

    assert [day["date"] for day in days] == ["2026-09-24", "2026-09-25"]
    assert days[1]["prices"][0] == pytest.approx(100.0)


def test_a_bulk_save_only_rewrites_changed_days(storage: LearningStorage) -> None:
    """180 days are not rewritten on every save: only new or changed ones."""
    history = [_day(f"2026-09-{d:02d}", 96) for d in range(1, 21)]
    conn = storage._ensure_conn()
    saved: dict[str, list[float | None]] = {}

    assert write_changed_days(conn, history, saved) == 20
    assert write_changed_days(conn, history, saved) == 0
    history[3]["prices"][10] = 99.0
    assert write_changed_days(conn, [*history, {"prices": [1.0]}], saved) == 1


def test_save_all_load_all_and_prune(storage: LearningStorage) -> None:
    history = [_day("2026-09-23", 96), _day("2026-09-24", 96)]
    storage.save_all({"price_history": history})

    loaded = storage.load_all()
    assert loaded is not None
    assert loaded["price_history"] == history
    assert storage.delete_old_prices("2026-09-24") == 96
    assert storage.load_price_history() == history[1:]
    assert delete_days_before(storage._ensure_conn(), date(2026, 9, 1)) == 0


def test_clear_all_empties_the_price_rows(storage: LearningStorage) -> None:
    storage.save_price_history([_day("2026-09-24", 96)])

    storage.clear_all()

    assert storage.load_price_history() == []
    # The change tracking forgot the cleared days: saving them writes again
    storage.save_price_history([_day("2026-09-24", 96)])
    assert len(storage.load_price_history()) == 1


def test_json_days_are_migrated_to_rows_once(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A v7 database's JSON days become rows; the legacy table is emptied."""
    storage = LearningStorage(_hass(tmp_path), "DK1")
    db_path = storage.db_path
    storage.close()
    with closing(sqlite3.connect(db_path)) as conn:
        conn.executemany(
            "INSERT INTO price_history (date, prices) VALUES (?, ?)",
            [
                ("2026-09-24", json.dumps(_day("2026-09-24", 96)["prices"])),
                ("2026-10-25", json.dumps(_day("2026-10-25", 100)["prices"])),
                ("2026-09-20", "not json"),
            ],
        )
        conn.execute("UPDATE meta SET value = '7' WHERE key = 'schema_version'")
        conn.commit()

    with caplog.at_level(logging.INFO):
        upgraded = LearningStorage(_hass(tmp_path), "DK1")
    try:
        days = upgraded.load_price_history()
        assert [day["date"] for day in days] == ["2026-09-24", "2026-10-25"]
        assert len(days[1]["prices"]) == 100
        assert "Stored 2 days of price history per UTC slot" in caplog.text
        with closing(sqlite3.connect(db_path)) as conn:
            legacy = conn.execute("SELECT COUNT(*) FROM price_history").fetchone()[0]
            version = conn.execute(
                "SELECT value FROM meta WHERE key = 'schema_version'"
            ).fetchone()[0]
        assert legacy == 0
        assert int(version) == PRICE_ROWS_SCHEMA_VERSION
    finally:
        upgraded.close()
