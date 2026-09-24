"""Tests for live forecast accuracy per lead time (bucketing and persistence)."""

import math
import sqlite3
from collections.abc import Iterator
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, Mock
from zoneinfo import ZoneInfo

import pytest

from homeassistant.util import dt as dt_util

from custom_components.open_spot_forecast.const import (
    LEAD_TIME_BUCKETS,
    LEAD_TIME_WINDOW_DAYS,
)
from custom_components.open_spot_forecast.ml.lead_time import (
    bucket_errors,
    lead_time_bucket,
    lead_time_hours,
    summarize_error_sums,
)
from custom_components.open_spot_forecast.ml.predictor import SpotPricePredictor
from custom_components.open_spot_forecast.ml.storage import LearningStorage

ACTUAL = 1.0


@pytest.fixture(autouse=True)
def copenhagen_time_zone() -> Iterator[None]:
    """Run with a non-UTC Home Assistant time zone, as in production."""
    original = dt_util.get_default_time_zone()
    dt_util.set_default_time_zone(ZoneInfo("Europe/Copenhagen"))
    yield
    dt_util.set_default_time_zone(original)


def _hass(tmp_path: Path) -> Mock:
    """Return a mock Home Assistant whose storage lives under tmp_path."""
    hass = Mock()
    hass.config.path.return_value = str(tmp_path / ".storage")
    return hass


def _slot() -> datetime:
    """Return today's 12:00 slot in Home Assistant's local time zone."""
    return dt_util.now().replace(hour=12, minute=0, second=0, microsecond=0)


def _insert(
    storage: LearningStorage, slot: datetime, lead_hours: float, price: float
) -> None:
    """Store a prediction for slot made lead_hours earlier (naive local stored_at)."""
    stored_at = (slot - timedelta(hours=lead_hours)).replace(tzinfo=None)
    storage.insert_prediction(
        start=slot.isoformat(),
        price=price,
        confidence=0.8,
        hour=slot.hour,
        minute=slot.minute,
        stored_at=stored_at.isoformat(),
    )


# --------------------------------------------------------------------------- #
# Bucketing
# --------------------------------------------------------------------------- #


def test_buckets_cover_day_one_to_three_and_beyond():
    """The 1d/2d/3d buckets exist and the last bucket is open-ended."""
    keys = [bucket for bucket, _ in LEAD_TIME_BUCKETS]
    assert keys == ["day_1", "day_2", "day_3", "day_4_plus"]
    assert math.isinf(LEAD_TIME_BUCKETS[-1][1])


@pytest.mark.parametrize(
    ("lead_hours", "expected"),
    [
        (0.0, "day_1"),
        (11.5, "day_1"),
        (23.99, "day_1"),
        (24.0, "day_2"),
        (47.99, "day_2"),
        (48.0, "day_3"),
        (71.99, "day_3"),
        (72.0, "day_4_plus"),
        (204.0, "day_4_plus"),
        (-0.25, None),
    ],
)
def test_lead_time_bucket_boundaries(lead_hours, expected):
    """Each bucket includes its lower bound and excludes its upper bound."""
    assert lead_time_bucket(lead_hours) == expected


def test_lead_time_hours_treats_naive_stored_at_as_local_time():
    """Naive stored_at (datetime.now()) is read in Home Assistant's time zone."""
    hours = lead_time_hours("2026-09-25T12:00:00+02:00", "2026-09-24T12:00:00")
    assert hours == pytest.approx(24.0)


def test_lead_time_hours_across_dst_change():
    """Lead time is real elapsed time, not wall-clock difference, across DST."""
    # Clocks go back at 03:00 CEST on 2026-10-25 (25-hour day).
    hours = lead_time_hours("2026-10-25T12:00:00+01:00", "2026-10-24T12:00:00")
    assert hours == pytest.approx(25.0)


@pytest.mark.parametrize(
    ("start", "stored_at"),
    [("not a date", "2026-09-24T12:00:00"), ("2026-09-25T12:00:00+02:00", "")],
)
def test_lead_time_hours_unparseable(start, stored_at):
    """Unparseable timestamps have no lead time."""
    assert lead_time_hours(start, stored_at) is None


def test_bucket_errors_groups_signed_errors_by_lead_time():
    """Errors are predicted - actual, grouped by bucket; invalid rows skipped."""
    start = "2026-09-25T12:00:00+02:00"
    predictions = [
        {"start": start, "stored_at": "2026-09-25T06:00:00", "price": 1.3},
        {"start": start, "stored_at": "2026-09-25T00:00:00", "price": 0.9},
        {"start": start, "stored_at": "2026-09-24T06:00:00", "price": 1.5},
        {"start": start, "stored_at": "2026-09-22T12:00:00", "price": 2.0},
        # Stored after the slot started: not a forecast.
        {"start": start, "stored_at": "2026-09-25T13:00:00", "price": 9.0},
        # Unparseable stored_at.
        {"start": start, "stored_at": "garbage", "price": 9.0},
    ]

    errors = bucket_errors(predictions, ACTUAL)

    assert set(errors) == {"day_1", "day_2", "day_4_plus"}
    assert errors["day_1"] == pytest.approx([0.3, -0.1])
    assert errors["day_2"] == pytest.approx([0.5])
    assert errors["day_4_plus"] == pytest.approx([1.0])


def test_summarize_error_sums_computes_mae_rmse_bias():
    """MAE, RMSE and bias are derived from the running sums."""
    # Errors +0.3 and -0.1: sum 0.2, abs 0.4, squares 0.1.
    summary = summarize_error_sums({"day_1": (2, 0.2, 0.4, 0.1), "day_2": (0, 0, 0, 0)})

    assert set(summary) == {"day_1"}
    assert summary["day_1"]["samples"] == 2
    assert summary["day_1"]["mae"] == pytest.approx(0.2)
    assert summary["day_1"]["rmse"] == pytest.approx(math.sqrt(0.05))
    assert summary["day_1"]["bias"] == pytest.approx(0.1)


# --------------------------------------------------------------------------- #
# Persistence (LearningStorage)
# --------------------------------------------------------------------------- #


def test_storage_accumulates_errors_per_date_and_bucket(tmp_path):
    """Repeated inserts for the same date and bucket add to the running sums."""
    storage = LearningStorage(_hass(tmp_path), "DK1")
    storage.add_lead_time_errors("2026-09-24", {"day_1": [0.3], "day_2": []})
    storage.add_lead_time_errors("2026-09-24", {"day_1": [-0.1]})
    storage.add_lead_time_errors("2026-09-25", {"day_1": [0.2], "day_3": [0.5]})
    storage.add_lead_time_errors("2026-09-25", {})

    sums = storage.get_lead_time_error_sums("2026-09-24")

    assert set(sums) == {"day_1", "day_3"}
    samples, sum_error, sum_abs, sum_sq = sums["day_1"]
    assert samples == 3
    assert sum_error == pytest.approx(0.4)
    assert sum_abs == pytest.approx(0.6)
    assert sum_sq == pytest.approx(0.14)
    assert storage.get_lead_time_error_sums("2026-09-25")["day_1"][0] == 1
    storage.close()


def test_storage_prunes_rows_before_cutoff(tmp_path):
    """Rows for slot dates before the cutoff are deleted."""
    storage = LearningStorage(_hass(tmp_path), "DK1")
    storage.add_lead_time_errors("2026-08-01", {"day_1": [0.1], "day_2": [0.2]})
    storage.add_lead_time_errors("2026-09-01", {"day_1": [0.3]})

    assert storage.delete_lead_time_accuracy_before("2026-09-01") == 2
    assert set(storage.get_lead_time_error_sums("2000-01-01")) == {"day_1"}
    storage.close()


def test_storage_survives_restart(tmp_path):
    """Accumulated sums are still there after closing and reopening the DB."""
    hass = _hass(tmp_path)
    storage = LearningStorage(hass, "DK1")
    storage.add_lead_time_errors("2026-09-24", {"day_2": [0.4, -0.2]})
    storage.close()

    reopened = LearningStorage(hass, "DK1")
    samples, sum_error, _, _ = reopened.get_lead_time_error_sums("2026-09-24")["day_2"]
    assert samples == 2
    assert sum_error == pytest.approx(0.2)
    reopened.close()


def test_clear_all_drops_lead_time_accuracy(tmp_path):
    """Resetting learning data also clears lead-time accuracy."""
    storage = LearningStorage(_hass(tmp_path), "DK1")
    storage.add_lead_time_errors("2026-09-24", {"day_1": [0.1]})

    storage.clear_all()

    assert storage.get_lead_time_error_sums("2000-01-01") == {}
    storage.close()


def test_predictions_are_kept_for_every_lead_time_bucket(tmp_path):
    """delete_old_predictions keeps rows until even day-4+ predictions can match.

    Forecasts reach 7 days past the end of the known prices, which can be up to
    ~1.5 days ahead, so a prediction can wait up to ~8.5 days for its slot.
    """
    predictor = SpotPricePredictor(_hass(tmp_path), "DK1")
    storage = predictor.storage
    now = datetime.now()
    for age_days in (9, predictor.max_history_days + 1):
        storage.insert_prediction(
            start=(now + timedelta(days=1)).isoformat(),
            price=1.0,
            confidence=0.5,
            hour=0,
            minute=0,
            stored_at=(now - timedelta(days=age_days)).isoformat(),
        )

    deleted = storage.delete_old_predictions(predictor.max_history_days)

    assert deleted == 1
    assert storage.count_predictions() == 1
    storage.close()


# --------------------------------------------------------------------------- #
# Self-learning integration (LeadTimeMixin)
# --------------------------------------------------------------------------- #


def test_learn_from_actual_price_records_every_lead_time(tmp_path):
    """All stored predictions for a slot are matched and bucketed by lead time."""
    predictor = SpotPricePredictor(_hass(tmp_path), "DK1")
    slot = _slot()
    _insert(predictor.storage, slot, 6, 1.3)  # day 1, error +0.3
    _insert(predictor.storage, slot, 18, 0.9)  # day 1, error -0.1
    _insert(predictor.storage, slot, 30, 1.5)  # day 2, error +0.5
    _insert(predictor.storage, slot, 54, 0.4)  # day 3, error -0.6
    _insert(predictor.storage, slot, 150, 2.0)  # day 4+, error +1.0

    assert predictor.learn_from_actual_price(slot.isoformat(), ACTUAL) is True

    accuracy = predictor.lead_time_accuracy
    assert set(accuracy) == {"day_1", "day_2", "day_3", "day_4_plus"}
    assert accuracy["day_1"]["samples"] == 2
    assert accuracy["day_1"]["mae"] == pytest.approx(0.2)
    assert accuracy["day_1"]["rmse"] == pytest.approx(math.sqrt(0.05))
    assert accuracy["day_1"]["bias"] == pytest.approx(0.1)
    assert accuracy["day_2"]["mae"] == pytest.approx(0.5)
    assert accuracy["day_3"]["rmse"] == pytest.approx(0.6)
    assert accuracy["day_3"]["bias"] == pytest.approx(-0.6)
    assert accuracy["day_4_plus"]["mae"] == pytest.approx(1.0)
    # Matched predictions are consumed.
    assert predictor.storage.count_predictions() == 0
    predictor.storage.close()


def test_lead_time_accuracy_survives_restart(tmp_path):
    """A new predictor on the same database reloads the rolling metrics."""
    hass = _hass(tmp_path)
    predictor = SpotPricePredictor(hass, "DK1")
    slot = _slot()
    _insert(predictor.storage, slot, 30, 1.5)
    predictor.learn_from_actual_price(slot.isoformat(), ACTUAL)
    predictor.storage.close()

    restarted = SpotPricePredictor(hass, "DK1")
    assert restarted.lead_time_accuracy == {}
    restarted.refresh_lead_time_accuracy()

    assert restarted.lead_time_accuracy["day_2"]["samples"] == 1
    assert restarted.lead_time_accuracy["day_2"]["mae"] == pytest.approx(0.5)
    restarted.storage.close()


def test_refresh_uses_rolling_window(tmp_path):
    """Only the last LEAD_TIME_WINDOW_DAYS days of slots count; older rows go."""
    predictor = SpotPricePredictor(_hass(tmp_path), "DK1")
    today = dt_util.now().date()
    oldest_inside = today - timedelta(days=LEAD_TIME_WINDOW_DAYS - 1)
    outside = today - timedelta(days=LEAD_TIME_WINDOW_DAYS)
    predictor.storage.add_lead_time_errors(today.isoformat(), {"day_1": [0.2]})
    predictor.storage.add_lead_time_errors(oldest_inside.isoformat(), {"day_1": [0.4]})
    predictor.storage.add_lead_time_errors(outside.isoformat(), {"day_1": [9.0]})

    predictor.refresh_lead_time_accuracy()

    assert predictor.lead_time_accuracy["day_1"]["samples"] == 2
    assert predictor.lead_time_accuracy["day_1"]["mae"] == pytest.approx(0.3)
    # The out-of-window row was pruned from the database.
    assert predictor.storage.get_lead_time_error_sums("2000-01-01")["day_1"][0] == 2
    predictor.storage.close()


def test_catch_up_learning_records_lead_time_accuracy(tmp_path):
    """The startup catch-up replay feeds the lead-time metrics too."""
    predictor = SpotPricePredictor(_hass(tmp_path), "DK1")
    slot = _slot()
    _insert(predictor.storage, slot, 30, 1.5)
    prices = [0.0] * 96
    prices[slot.hour * 4] = ACTUAL
    predictor.price_history = [{"date": slot.date().isoformat(), "prices": prices}]

    assert predictor.catch_up_learning() == 1

    assert predictor.lead_time_accuracy["day_2"]["mae"] == pytest.approx(0.5)
    predictor.storage.close()


def test_record_lead_time_accuracy_logs_storage_errors(caplog):
    """A storage failure is logged and does not break the learning loop."""
    predictor = SpotPricePredictor.__new__(SpotPricePredictor)
    predictor.storage = Mock()
    predictor.storage.add_lead_time_errors.side_effect = sqlite3.OperationalError(
        "disk I/O error"
    )
    predictions = [
        {
            "start": "2026-09-25T12:00:00+02:00",
            "stored_at": "2026-09-25T06:00:00",
            "price": 1.3,
        }
    ]

    predictor.record_lead_time_accuracy(predictions, ACTUAL)

    assert "Failed to record lead-time accuracy" in caplog.text


@pytest.mark.asyncio
async def test_reset_learning_clears_lead_time_accuracy():
    """reset_learning empties the cached lead-time summary."""
    predictor = SpotPricePredictor.__new__(SpotPricePredictor)
    predictor.storage = Mock()
    predictor.storage.async_clear_storage = AsyncMock(return_value=True)
    predictor.lead_time_accuracy = {"day_1": {"mae": 0.1, "samples": 1}}

    await predictor.reset_learning()

    assert predictor.lead_time_accuracy == {}
