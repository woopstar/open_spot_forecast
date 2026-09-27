"""Predicted-vs-actual evaluation series (#36).

Self-learning removes a slot's predictions once it has scored them. The one
made closest to a day ahead is kept next to the actual price, and a
diagnostic sensor and the ``get_forecast`` action expose the recent series.
"""

import sqlite3
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock, patch
from zoneinfo import ZoneInfo

import pytest

from homeassistant.config_entries import ConfigEntryState
from homeassistant.const import EntityCategory
from homeassistant.util import dt as dt_util

from custom_components.open_spot_forecast.const import (
    DOMAIN,
    EVALUATION_KEEP_DAYS,
    RECORDER_MAX_ATTRIBUTES_BYTES,
)
from custom_components.open_spot_forecast.evaluation_sensor import (
    ForecastEvaluationSensor,
)
from custom_components.open_spot_forecast.forecast_attributes import attributes_size
from custom_components.open_spot_forecast.ml.lead_time import evaluation_prediction
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

    await predictor.reset_learning()

    assert predictor.evaluation == []


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


def _sensor(rows: list[dict[str, Any]] | None) -> ForecastEvaluationSensor:
    api_data: dict[str, Any] = {}
    if rows is not None:
        api_data["ml_predictor"] = Mock(evaluation=rows, is_trained=True)
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
    sensor = _sensor(_rows(7 * 96, NOW - timedelta(days=7)))

    _, attributes = _read(sensor)

    stored = {**attributes, "attribution": sensor.attribution, "friendly_name": "x"}
    assert attributes_size(stored) < RECORDER_MAX_ATTRIBUTES_BYTES


def test_without_data_the_sensor_is_unknown() -> None:
    for sensor in (_sensor(None), _sensor([])):
        value, attributes = _read(sensor)
        assert value is None
        assert attributes["samples"] == 0
        assert attributes["bias"] is None
        assert attributes["s"] == []


# --- The action ----------------------------------------------------------------------


async def _call(evaluation: bool | None) -> Any:
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
    hass.data = {DOMAIN: {"entry": {"ml_predictor": predictor}}}
    call = Mock()
    call.hass = hass
    call.data = GET_FORECAST_SCHEMA(
        {} if evaluation is None else {"evaluation": evaluation}
    )
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
