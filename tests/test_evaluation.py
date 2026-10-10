"""Predicted-vs-actual evaluation series (#36).

Self-learning removes a slot's predictions once it has scored them. The one
made closest to a day ahead is kept next to the actual price, and a
diagnostic sensor and the ``get_forecast`` action expose the recent series.
So are the snapshots at the other lead times (12 and 48 hours ahead, #113),
when a prediction was made close enough to them.
"""

import sqlite3
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock, patch
from zoneinfo import ZoneInfo

import pytest
import voluptuous as vol
import yaml

from homeassistant.config_entries import ConfigEntryState
from homeassistant.const import EntityCategory
from homeassistant.util import dt as dt_util

from custom_components.open_spot_forecast.const import (
    DOMAIN,
    EVALUATION_KEEP_DAYS,
    EVALUATION_LEAD_HOURS,
    EVALUATION_LEAD_TIMES,
    RECORDER_MAX_ATTRIBUTES_BYTES,
)
from custom_components.open_spot_forecast.evaluation_sensor import (
    ForecastEvaluationSensor,
)
from custom_components.open_spot_forecast.forecast_attributes import attributes_size
from custom_components.open_spot_forecast.ml.evaluation_storage import (
    EVALUATION_TABLE_SQL,
    migrate_evaluation_to_lead_times,
)
from custom_components.open_spot_forecast.ml.lead_time import (
    evaluation_prediction,
    evaluation_snapshots,
    snapshot_tolerance,
)
from custom_components.open_spot_forecast.ml.predictor import SpotPricePredictor
from custom_components.open_spot_forecast.ml.storage import LearningStorage
from custom_components.open_spot_forecast.price_output import PriceOutput
from custom_components.open_spot_forecast.services import (
    GET_FORECAST_SCHEMA,
    _async_get_forecast,
)
from custom_components.open_spot_forecast.time_slots import (
    slot_index_in_day,
    slots_in_local_day,
)

pytestmark = pytest.mark.usefixtures("copenhagen_time_zone")

CPH = ZoneInfo("Europe/Copenhagen")


def _hass(tmp_path: Path) -> Mock:
    hass = Mock()
    hass.config.path.return_value = str(tmp_path / ".storage")
    return hass


@pytest.fixture
def predictor(tmp_path: Path) -> Iterator[SpotPricePredictor]:
    predictor = SpotPricePredictor(_hass(tmp_path), "DK1")
    yield predictor
    predictor.storage.close()


def _slot(hours_ago: float = 1.0) -> datetime:
    """Return a recent local slot start."""
    moment = dt_util.now() - timedelta(hours=hours_ago)
    return moment.replace(minute=moment.minute // 15 * 15, second=0, microsecond=0)


def _insert(
    storage: LearningStorage, slot: datetime, lead: float, price: float
) -> None:
    storage.insert_prediction(
        start=slot.isoformat(),
        price=price,
        confidence=0.8,
        hour=slot.hour,
        minute=slot.minute,
        stored_at=(slot - timedelta(hours=lead)).isoformat(),
    )


def _prediction(lead: float, price: float = 1.0) -> dict[str, Any]:
    slot = datetime(2026, 9, 24, 12, 0, tzinfo=CPH)
    return {
        "start": slot.isoformat(),
        "stored_at": (slot - timedelta(hours=lead)).isoformat(),
        "price": price,
    }


# --- Choosing the prediction ---------------------------------------------------------


def test_the_prediction_closest_to_a_day_ahead_is_chosen() -> None:
    chosen = evaluation_prediction(
        [_prediction(34, 1.0), _prediction(22, 2.0), _prediction(28, 3.0)]
    )

    assert chosen is not None
    assert chosen[0]["price"] == pytest.approx(2.0)
    assert chosen[1] == pytest.approx(22.0)


def test_ties_go_to_the_later_prediction_and_bad_leads_are_skipped() -> None:
    chosen = evaluation_prediction(
        [_prediction(28, 1.0), _prediction(20, 2.0), _prediction(-1, 3.0)]
    )
    assert chosen is not None
    assert chosen[0]["price"] == pytest.approx(2.0)
    assert evaluation_prediction([_prediction(-1)]) is None
    assert evaluation_prediction([{"start": "x", "stored_at": "y"}]) is None


# --- A snapshot per lead time (#113) -------------------------------------------------


def test_the_day_ahead_lead_time_is_one_of_the_snapshots() -> None:
    assert EVALUATION_LEAD_HOURS in EVALUATION_LEAD_TIMES
    assert EVALUATION_LEAD_TIMES == (12.0, 24.0, 48.0)


def test_a_snapshot_may_be_half_the_gap_to_its_neighbour_away() -> None:
    assert snapshot_tolerance(12.0) == pytest.approx(6.0)
    assert snapshot_tolerance(48.0) == pytest.approx(12.0)
    # The day-ahead one is always kept, as before
    assert snapshot_tolerance(EVALUATION_LEAD_HOURS) is None


def test_each_lead_time_keeps_its_closest_prediction() -> None:
    snapshots = evaluation_snapshots(
        [
            _prediction(50, 1.0),
            _prediction(44, 2.0),
            _prediction(26, 3.0),
            _prediction(14, 4.0),
            _prediction(8, 5.0),
        ]
    )

    assert {target: chosen[0]["price"] for target, chosen in snapshots.items()} == {
        12.0: pytest.approx(4.0),
        24.0: pytest.approx(3.0),
        48.0: pytest.approx(1.0),
    }
    # Each with its real lead time
    assert snapshots[12.0][1] == pytest.approx(14.0)
    assert snapshots[48.0][1] == pytest.approx(50.0)


def test_a_lead_time_without_a_close_prediction_is_skipped() -> None:
    # 31 h ahead is neither a 12 h (6-18 h) nor a 48 h (36-60 h) snapshot
    snapshots = evaluation_snapshots([_prediction(31, 1.0)])

    assert list(snapshots) == [EVALUATION_LEAD_HOURS]
    assert snapshots[EVALUATION_LEAD_HOURS][1] == pytest.approx(31.0)
    # The edges of the tolerance still count
    assert set(evaluation_snapshots([_prediction(18), _prediction(36)])) == {
        12.0,
        24.0,
        48.0,
    }
    assert set(evaluation_snapshots([_prediction(18.5), _prediction(35.5)])) == {24.0}
    assert evaluation_snapshots([_prediction(-1)]) == {}


# --- Storage -------------------------------------------------------------------------


def test_storage_round_trip_replace_and_prune(tmp_path: Path) -> None:
    storage = LearningStorage(_hass(tmp_path), "DK1")
    try:
        storage.upsert_evaluation("2026-09-24T10:00:00Z", 1.0, 1.5, 24.0)
        storage.upsert_evaluation("2026-09-24T10:00:00Z", 1.2, 1.5, 22.0)
        storage.upsert_evaluation("2026-09-24T09:45:00Z", 0.9, 1.0, 26.0)

        rows = storage.get_evaluation("2026-09-24T00:00:00Z")
        assert [row["timestamp"] for row in rows] == [
            "2026-09-24T09:45:00Z",
            "2026-09-24T10:00:00Z",
        ]
        assert rows[1]["predicted"] == pytest.approx(1.2)
        assert rows[1]["lead_hours"] == pytest.approx(22.0)
        assert storage.delete_evaluation_before("2026-09-24T10:00:00Z") == 1
        storage.clear_all()
        assert storage.get_evaluation("2026-09-24T00:00:00Z") == []
    finally:
        storage.close()


def test_storage_keeps_a_row_per_slot_and_lead_time(tmp_path: Path) -> None:
    storage = LearningStorage(_hass(tmp_path), "DK1")
    try:
        storage.upsert_evaluation("2026-09-24T10:00:00Z", 1.0, 1.5, 23.0)
        storage.upsert_evaluation("2026-09-24T10:00:00Z", 1.1, 1.5, 13.0, 12.0)
        storage.upsert_evaluation("2026-09-24T10:00:00Z", 1.2, 1.5, 11.0, 12.0)
        storage.upsert_evaluation("2026-09-24T10:00:00Z", 1.3, 1.5, 47.0, 48.0)

        since = "2026-09-24T00:00:00Z"
        assert [row["predicted"] for row in storage.get_evaluation(since)] == [
            pytest.approx(1.0)
        ]
        twelve = storage.get_evaluation(since, 12.0)
        assert [row["predicted"] for row in twelve] == [pytest.approx(1.2)]
        assert twelve[0]["lead_hours"] == pytest.approx(11.0)
        assert storage.get_evaluation(since, 48.0)[0]["predicted"] == pytest.approx(1.3)
        # Pruning removes every lead time's row of a slot
        assert storage.delete_evaluation_before("2026-09-25T00:00:00Z") == 3
    finally:
        storage.close()


def test_a_table_from_before_the_lead_times_is_rebuilt(tmp_path: Path) -> None:
    storage = LearningStorage(_hass(tmp_path), "DK1")
    path = storage.db_path
    storage.close()
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """DROP TABLE evaluation;
               CREATE TABLE evaluation (
                   timestamp   TEXT    PRIMARY KEY,
                   predicted   REAL    NOT NULL,
                   actual      REAL    NOT NULL,
                   lead_hours  REAL    NOT NULL
               );
               INSERT INTO evaluation VALUES ('2026-09-24T10:00:00Z', 1.0, 1.5, 22.0);
            """
        )
    conn.close()

    # Opening the database migrates it; a second start changes nothing
    for kept_at_12_hours in (0, 1):
        storage = LearningStorage(_hass(tmp_path), "DK1")
        try:
            rows = storage.get_evaluation("2026-09-24T00:00:00Z")
            assert len(rows) == 1
            assert rows[0]["predicted"] == pytest.approx(1.0)
            assert rows[0]["lead_hours"] == pytest.approx(22.0)
            twelve = storage.get_evaluation("2026-09-24T00:00:00Z", 12.0)
            assert len(twelve) == kept_at_12_hours
            # The rebuilt table takes a second lead time for the same slot
            storage.upsert_evaluation("2026-09-24T10:00:00Z", 1.1, 1.5, 12.0, 12.0)
        finally:
            storage.close()


def test_the_migration_leaves_new_and_missing_tables_alone() -> None:
    conn = sqlite3.connect(":memory:")
    try:
        # No table yet (a new database): nothing to migrate
        migrate_evaluation_to_lead_times(conn)
        assert conn.execute("SELECT name FROM sqlite_master").fetchall() == []
        conn.execute(EVALUATION_TABLE_SQL)
        conn.execute(
            "INSERT INTO evaluation VALUES ('2026-09-24T10:00:00Z', 12.0, 1, 2, 11)"
        )
        migrate_evaluation_to_lead_times(conn)
        assert conn.execute("SELECT target_hours FROM evaluation").fetchall() == [
            (12.0,)
        ]
    finally:
        conn.close()


# --- Self-learning -------------------------------------------------------------------


def test_learning_keeps_the_day_ahead_prediction(predictor: SpotPricePredictor) -> None:
    slot = _slot()
    _insert(predictor.storage, slot, 34, 1.0)
    _insert(predictor.storage, slot, 23, 2.0)
    _insert(predictor.storage, slot, 12, 3.0)

    assert predictor.learn_from_actual_price(slot.isoformat(), 2.5) is True

    # The scored predictions are gone; the evaluation keeps the 23 h one
    assert predictor.storage.count_predictions() == 0
    assert predictor.evaluation == [
        {
            "start": slot.isoformat(),
            "end": (slot + timedelta(minutes=15)).isoformat(),
            "predicted": pytest.approx(2.0),
            "actual": pytest.approx(2.5),
            "lead_hours": pytest.approx(23.0),
        }
    ]


def test_learning_keeps_a_snapshot_per_lead_time(predictor: SpotPricePredictor) -> None:
    slot = _slot()
    _insert(predictor.storage, slot, 46, 1.0)
    _insert(predictor.storage, slot, 23, 2.0)
    _insert(predictor.storage, slot, 13, 3.0)

    assert predictor.learn_from_actual_price(slot.isoformat(), 2.5) is True

    snapshots = predictor.evaluation_snapshots
    assert predictor.evaluation is snapshots[EVALUATION_LEAD_HOURS]
    assert [
        (target, rows[0]["predicted"], rows[0]["lead_hours"])
        for target, rows in snapshots.items()
    ] == [
        (12.0, pytest.approx(3.0), pytest.approx(13.0)),
        (24.0, pytest.approx(2.0), pytest.approx(23.0)),
        (48.0, pytest.approx(1.0), pytest.approx(46.0)),
    ]
    assert all(rows[0]["actual"] == pytest.approx(2.5) for rows in snapshots.values())


def test_learning_skips_a_lead_time_nothing_was_predicted_near(
    predictor: SpotPricePredictor,
) -> None:
    slot = _slot()
    _insert(predictor.storage, slot, 31, 1.0)

    assert predictor.learn_from_actual_price(slot.isoformat(), 2.5) is True

    assert predictor.evaluation[0]["lead_hours"] == pytest.approx(31.0)
    assert predictor.evaluation_snapshots[12.0] == []
    assert predictor.evaluation_snapshots[48.0] == []
    assert predictor.storage.get_evaluation("2000-01-01T00:00:00Z", 12.0) == []


def test_the_evaluation_survives_a_restart(tmp_path: Path) -> None:
    first = SpotPricePredictor(_hass(tmp_path), "DK1")
    slot = _slot()
    _insert(first.storage, slot, 24, 2.0)
    first.learn_from_actual_price(slot.isoformat(), 2.5)
    first.storage.close()

    second = SpotPricePredictor(_hass(tmp_path), "DK1")
    try:
        assert second.evaluation == []
        second.refresh_evaluation()
        assert len(second.evaluation) == 1
    finally:
        second.storage.close()


def test_old_slots_are_not_kept(predictor: SpotPricePredictor) -> None:
    old = _slot(hours_ago=EVALUATION_KEEP_DAYS * 24 + 1)

    predictor.record_evaluation(old, [_prediction(24)], 1.0)

    assert predictor.storage.get_evaluation("2000-01-01T00:00:00Z") == []
    predictor.storage.upsert_evaluation("2000-01-01T00:00:00Z", 1.0, 1.0, 24.0)
    predictor.refresh_evaluation()
    assert predictor.storage.get_evaluation("2000-01-01T00:00:00Z") == []


def test_a_naive_slot_start_is_local_time(predictor: SpotPricePredictor) -> None:
    slot = _slot()
    prediction = {
        "start": slot.isoformat(),
        "stored_at": (slot - timedelta(hours=24)).isoformat(),
        "price": 1.0,
    }

    predictor.record_evaluation(slot.replace(tzinfo=None), [prediction], 1.1)

    assert predictor.evaluation[0]["start"] == slot.isoformat()


def test_catch_up_learning_records_the_evaluation(
    predictor: SpotPricePredictor,
) -> None:
    slot = _slot()
    day = slot.date()
    index = slot_index_in_day(slot)
    prices: list[float | None] = [None] * slots_in_local_day(day)
    prices[index] = 2.5
    predictor.price_history = [{"date": day.isoformat(), "prices": prices}]
    _insert(predictor.storage, slot, 25, 2.0)

    assert predictor.catch_up_learning() == 1

    assert predictor.evaluation[0]["predicted"] == pytest.approx(2.0)


def test_storage_errors_do_not_stop_learning(
    predictor: SpotPricePredictor, caplog: pytest.LogCaptureFixture
) -> None:
    with patch.object(
        predictor.storage,
        "upsert_evaluation",
        side_effect=sqlite3.OperationalError("locked"),
    ):
        predictor.record_evaluation(_slot(), [_prediction(24)], 1.0)

    assert "Failed to record the evaluation: locked" in caplog.text


@pytest.mark.asyncio
async def test_reset_learning_clears_the_evaluation() -> None:
    predictor = SpotPricePredictor.__new__(SpotPricePredictor)
    predictor.storage = Mock()
    predictor.storage.async_clear_storage = AsyncMock(return_value=True)
    predictor.evaluation = [{"start": "x"}]
    predictor.evaluation_snapshots = {12.0: [{"start": "x"}]}
    predictor.day_ahead_predictions = {"2026-09-24T10:00:00Z": 1.0}

    await predictor.reset_learning()

    assert predictor.evaluation == []
    assert predictor.evaluation_snapshots == {}
    assert predictor.day_ahead_predictions == {}


# --- The sensor ----------------------------------------------------------------------


def _rows(count: int, first: datetime) -> list[dict[str, Any]]:
    """Return evaluated slots from ``first``: predicted 1.0 + n/100, actual 1.0."""
    return [
        {
            "start": (first + timedelta(minutes=15 * n)).astimezone(CPH).isoformat(),
            "end": (first + timedelta(minutes=15 * (n + 1)))
            .astimezone(CPH)
            .isoformat(),
            "predicted": 1.0 + (n % 10) / 100,
            "actual": 1.0,
            "lead_hours": 24.0,
        }
        for n in range(count)
    ]


def _sensor(
    rows: list[dict[str, Any]] | None,
    snapshots: dict[float, list[dict[str, Any]]] | None = None,
) -> ForecastEvaluationSensor:
    api_data: dict[str, Any] = {}
    if rows is not None:
        api_data["ml_predictor"] = Mock(
            evaluation=rows, evaluation_snapshots=snapshots or {}, is_trained=True
        )
    return ForecastEvaluationSensor(
        Mock(), MagicMock(entry_id="test"), api_data, "DKK", PriceOutput(vat=0.25)
    )


NOW = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)


def _read(sensor: ForecastEvaluationSensor) -> tuple[Any, dict[str, Any]]:
    with patch("homeassistant.util.dt.utcnow", return_value=NOW):
        return sensor.native_value, sensor.extra_state_attributes


def test_the_sensor_is_a_translated_diagnostic_entity() -> None:
    sensor = _sensor([])

    assert sensor.entity_category is EntityCategory.DIAGNOSTIC
    assert sensor.translation_key == "forecast_evaluation"
    assert sensor.unique_id == f"{DOMAIN}_test_forecast_evaluation"
    assert sensor.native_unit_of_measurement == "DKK/kWh"


def test_the_sensor_shows_the_last_48_hours() -> None:
    # 3 days of slots up to now: only the last 48 hours are shown
    rows = _rows(3 * 96, NOW - timedelta(days=3))

    value, attributes = _read(_sensor(rows))

    assert attributes["samples"] == 48 * 4
    assert len(attributes["s"]) == len(attributes["t"]) == len(attributes["a"])
    assert attributes["s"][0] == int((NOW - timedelta(hours=48)).timestamp())
    # Converted like every price: VAT 25 %
    assert attributes["a"][0] == pytest.approx(1.25)
    errors = [t - a for t, a in zip(attributes["t"], attributes["a"])]
    assert value == pytest.approx(
        round(sum(abs(e) for e in errors) / len(errors), 3), abs=1e-3
    )
    assert attributes["bias"] == pytest.approx(value, abs=1e-3)
    assert attributes["lead_hours"] == pytest.approx(24.0)
    assert attributes["window_hours"] == 48


def test_the_attributes_stay_under_16_kb() -> None:
    # Every slot of the week has a snapshot at every lead time
    rows = _rows(7 * 96, NOW - timedelta(days=7))
    sensor = _sensor(rows, {12.0: rows, 48.0: rows})

    _, attributes = _read(sensor)

    assert len(attributes["t12"]) == len(attributes["t48"]) == 48 * 4
    assert None not in attributes["t12"]
    stored = {**attributes, "attribution": sensor.attribution, "friendly_name": "x"}
    assert attributes_size(stored) < RECORDER_MAX_ATTRIBUTES_BYTES


def test_the_other_lead_times_are_arrays_aligned_with_the_slots() -> None:
    rows = _rows(4, NOW - timedelta(hours=1))
    # 12 h ahead: the second and fourth slot only; 48 h ahead: none
    twelve = [{**rows[1], "predicted": 2.0}, {**rows[3], "predicted": 3.0}]

    _, attributes = _read(_sensor(rows, {12.0: twelve, 48.0: []}))

    # Converted like every price: VAT 25 %
    assert attributes["t12"] == [None, pytest.approx(2.5), None, pytest.approx(3.75)]
    assert attributes["t48"] == [None, None, None, None]
    # The day-ahead series is unchanged
    assert attributes["t"] == pytest.approx([1.25, 1.2625, 1.275, 1.2875], abs=1e-3)
    assert set(attributes) == {
        "interval_minutes",
        "unit",
        "lead_hours",
        "window_hours",
        "samples",
        "bias",
        "s",
        "t",
        "a",
        "t12",
        "t48",
    }


def test_without_data_the_sensor_is_unknown() -> None:
    for sensor in (_sensor(None), _sensor([])):
        value, attributes = _read(sensor)
        assert value is None
        assert attributes["samples"] == 0
        assert attributes["bias"] is None
        assert attributes["s"] == []
        assert attributes["t12"] == attributes["t48"] == []


# --- The action ----------------------------------------------------------------------


async def _call(evaluation: bool | None, target_hours: Any = None) -> Any:
    entry = MagicMock()
    entry.entry_id = "entry"
    entry.domain = DOMAIN
    entry.state = ConfigEntryState.LOADED
    entry.data = {"currency": "DKK"}
    entry.options = {"vat": 0.0}
    hass = Mock()
    hass.config_entries.async_entries.return_value = [entry]
    predictor = MagicMock()
    predictor.predictions = []
    predictor.evaluation = _rows(2, NOW - timedelta(hours=1))
    predictor.evaluation_snapshots = {
        12.0: [{**_rows(1, NOW - timedelta(hours=1))[0], "lead_hours": 13.0}],
        24.0: predictor.evaluation,
        48.0: [],
    }
    hass.data = {DOMAIN: {"entry": {"ml_predictor": predictor}}}
    call = Mock()
    call.hass = hass
    data = {} if evaluation is None else {"evaluation": evaluation}
    if target_hours is not None:
        data["target_hours"] = target_hours
    call.data = GET_FORECAST_SCHEMA(data)
    with patch("homeassistant.util.dt.utcnow", return_value=NOW):
        return await _async_get_forecast(call)


@pytest.mark.asyncio
async def test_the_action_returns_the_evaluation_on_request() -> None:
    response = await _call(True)

    assert response["evaluation"] == [
        {
            "start": "2026-09-24T13:00:00+02:00",
            "end": "2026-09-24T13:15:00+02:00",
            "predicted": pytest.approx(1.0),
            "actual": pytest.approx(1.0),
            "lead_hours": pytest.approx(24.0),
        },
        {
            "start": "2026-09-24T13:15:00+02:00",
            "end": "2026-09-24T13:30:00+02:00",
            "predicted": pytest.approx(1.01),
            "actual": pytest.approx(1.0),
            "lead_hours": pytest.approx(24.0),
        },
    ]
    assert "evaluation" not in await _call(None)
    assert "evaluation" not in await _call(False)


@pytest.mark.asyncio
async def test_the_action_returns_another_lead_time_on_request() -> None:
    day_ahead = (await _call(True))["evaluation"]

    # The selector sends the option as a string
    twelve = (await _call(True, "12"))["evaluation"]

    assert [row["lead_hours"] for row in twelve] == [pytest.approx(13.0)]
    assert twelve[0]["start"] == "2026-09-24T13:00:00+02:00"
    assert (await _call(True, 24))["evaluation"] == day_ahead
    assert (await _call(True, 48.0))["evaluation"] == []
    assert "evaluation" not in await _call(None, 12)
    with pytest.raises(vol.Invalid):
        await _call(True, 36)


def test_the_action_offers_every_lead_time() -> None:
    services = yaml.safe_load(
        (
            Path(__file__).parent.parent
            / "custom_components"
            / "open_spot_forecast"
            / "services.yaml"
        ).read_text()
    )

    options = services["get_forecast"]["fields"]["target_hours"]["selector"]["select"][
        "options"
    ]
    assert [float(option) for option in options] == list(EVALUATION_LEAD_TIMES)
