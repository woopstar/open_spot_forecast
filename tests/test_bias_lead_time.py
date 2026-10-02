"""Bias offsets learned and applied per slot and lead-time bucket (issue #118)."""

import logging
import sqlite3
from collections.abc import Iterator
from contextlib import closing
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import numpy as np
import pytest

from homeassistant.util import dt as dt_util

from custom_components.open_spot_forecast.const import (
    BIAS_FALLBACK_BUCKET,
    LEAD_TIME_BUCKETS,
)
from custom_components.open_spot_forecast.ml.bias_storage import (
    LEAD_TIME_BIAS_SCHEMA_VERSION,
)
from custom_components.open_spot_forecast.ml.lead_time import prediction_bucket
from custom_components.open_spot_forecast.ml.learning import (
    MAX_ERROR_SAMPLES,
    add_matched_errors,
    new_slot_metrics,
)
from custom_components.open_spot_forecast.ml.predictor import SpotPricePredictor
from custom_components.open_spot_forecast.ml.storage import LearningStorage

SLOT = 48  # 12:00
BUCKETS = [bucket for bucket, _upper in LEAD_TIME_BUCKETS]


def _hass(tmp_path: Path) -> Mock:
    hass = Mock()
    hass.config.path.return_value = str(tmp_path / ".storage")
    return hass


@pytest.fixture
def predictor(tmp_path: Path) -> Iterator[SpotPricePredictor]:
    predictor = SpotPricePredictor(_hass(tmp_path), "DK1")
    yield predictor
    predictor.storage.close()


def _metrics(bucket_errors: dict[str, list[float]]) -> dict:
    """A slot's metrics with the given errors per bucket (pooled lists too)."""
    metrics = new_slot_metrics()
    for errors in bucket_errors.values():
        metrics["errors"] += errors
        metrics["abs_errors"] += [abs(e) for e in errors]
        metrics["pct_errors"] += [0.0] * len(errors)
        metrics["predictions"] += [1.0 + e for e in errors]
        metrics["actuals"] += [1.0] * len(errors)
        metrics["count"] += len(errors)
    metrics["bucket_errors"] = bucket_errors
    return metrics


def _prediction(slot_start: datetime, lead_hours: float, price: float) -> dict:
    stored_at = slot_start - timedelta(hours=lead_hours)
    return {
        "start": slot_start.isoformat(),
        "stored_at": stored_at.isoformat(),
        "price": price,
    }


def _tomorrow_noon() -> datetime:
    return dt_util.now().replace(
        hour=12, minute=0, second=0, microsecond=0
    ) + timedelta(days=1)


# --- Learning per bucket ---------------------------------------------------------


def test_each_bucket_learns_its_own_offset(predictor: SpotPricePredictor) -> None:
    predictor.error_metrics = {
        SLOT: _metrics({"day_1": [0.3] * 3, "day_2": [0.6] * 3, "day_3": [0.9] * 2})
    }

    predictor._update_bias_correction(SLOT)

    assert predictor.bias_correction == {
        SLOT: {"day_1": pytest.approx(0.3), "day_2": pytest.approx(0.6)}
    }


def test_the_ema_runs_per_bucket(predictor: SpotPricePredictor) -> None:
    """offset = 0.9 * old + 0.1 * (old + mean_error), per bucket."""
    predictor.bias_correction = {SLOT: {"day_1": 0.3, "day_2": 0.6}}
    predictor.error_metrics = {
        SLOT: _metrics({"day_1": [0.1] * 3, "day_2": [-0.2] * 4})
    }

    predictor._update_bias_correction(SLOT)

    assert predictor.bias_correction[SLOT]["day_1"] == pytest.approx(
        0.9 * 0.3 + 0.1 * (0.3 + 0.1)
    )
    assert predictor.bias_correction[SLOT]["day_2"] == pytest.approx(
        0.9 * 0.6 + 0.1 * (0.6 - 0.2)
    )


def test_a_bucket_without_an_offset_starts_from_the_fallback(
    predictor: SpotPricePredictor,
) -> None:
    """Its predictions had the day_1 offset subtracted, so the EMA continues from it."""
    predictor.bias_correction = {SLOT: {BIAS_FALLBACK_BUCKET: 0.2}}
    predictor.error_metrics = {SLOT: _metrics({"day_3": [0.1] * 3})}

    predictor._update_bias_correction(SLOT)

    assert predictor.bias_correction[SLOT]["day_3"] == pytest.approx(
        0.9 * 0.2 + 0.1 * (0.2 + 0.1)
    )
    assert predictor.bias_correction[SLOT][BIAS_FALLBACK_BUCKET] == pytest.approx(0.2)


def test_a_bucket_needs_three_samples(predictor: SpotPricePredictor) -> None:
    predictor.error_metrics = {SLOT: _metrics({"day_1": [1.0, 1.0], "day_2": []})}

    predictor._update_bias_correction(SLOT)

    assert predictor.bias_correction == {}


def test_metrics_without_bucket_errors_learn_nothing(
    predictor: SpotPricePredictor,
) -> None:
    """Metrics stored before #118 have no per-bucket errors yet."""
    predictor.error_metrics = {SLOT: {"errors": [0.5] * 5, "count": 5}}

    predictor._update_bias_correction(SLOT)

    assert predictor.bias_correction == {}


# --- Applying the offset of the prediction's lead time ----------------------------


def test_the_offset_of_the_prediction_bucket_is_applied(
    predictor: SpotPricePredictor,
) -> None:
    predictor.bias_correction = {SLOT: {"day_1": 0.2, "day_3": -0.4}}

    assert predictor.apply_bias_correction(1.0, SLOT, "day_1") == pytest.approx(0.8)
    assert predictor.apply_bias_correction(1.0, SLOT, "day_3") == pytest.approx(1.4)


def test_an_empty_bucket_falls_back_to_day_1(predictor: SpotPricePredictor) -> None:
    predictor.bias_correction = {SLOT: {BIAS_FALLBACK_BUCKET: 0.2}}

    assert predictor.apply_bias_correction(1.0, SLOT, "day_2") == pytest.approx(0.8)
    assert predictor.apply_bias_correction(1.0, SLOT, "day_4_plus") == pytest.approx(
        0.8
    )
    assert predictor._bias_offset(SLOT, "day_4_plus") == pytest.approx(0.2)


def test_without_a_fallback_the_prediction_is_unchanged(
    predictor: SpotPricePredictor,
) -> None:
    predictor.bias_correction = {SLOT: {"day_2": 0.2}}

    assert predictor.apply_bias_correction(1.0, SLOT, "day_3") == pytest.approx(1.0)
    assert predictor.apply_bias_correction(1.0, SLOT + 1, "day_2") == pytest.approx(1.0)
    assert predictor._bias_offset(SLOT, "day_3") is None


@pytest.mark.parametrize(
    ("lead_hours", "bucket"),
    [
        (-0.5, "day_1"),
        (0.0, "day_1"),
        (5.0, "day_1"),
        (24.0, "day_2"),
        (47.9, "day_2"),
        (60.0, "day_3"),
        (200.0, "day_4_plus"),
    ],
)
def test_prediction_bucket_follows_the_lead_time(
    lead_hours: float, bucket: str
) -> None:
    now = dt_util.now()
    start = now + timedelta(hours=lead_hours)

    assert prediction_bucket(start.isoformat(), now) == bucket


def test_prediction_bucket_of_an_unreadable_start_is_the_fallback() -> None:
    assert prediction_bucket("None", dt_util.now()) == BIAS_FALLBACK_BUCKET


def test_generated_predictions_get_the_offset_of_their_lead_time(
    predictor: SpotPricePredictor,
) -> None:
    """A run predicts the same slot a day apart: each gets its bucket's offset."""
    now = dt_util.now().replace(minute=0, second=0, microsecond=0)
    starts = [now + timedelta(hours=5), now + timedelta(hours=29)]
    slot = starts[0].hour * 4
    predictor.bias_correction = {slot: {"day_1": 0.1, "day_2": 0.5}}
    predictor.price_model = MagicMock()
    predictor.price_model.predict.side_effect = lambda rows: np.zeros(len(rows))
    predictor.is_trained = True
    features = predictor._combine_features(
        [{"start": start.isoformat()} for start in starts], {}
    )

    predictor._generate_predictions(features, 2, 60)

    assert [p["price"] for p in predictor.predictions] == [
        pytest.approx(-0.1),
        pytest.approx(-0.5),
    ]


# --- The learning loop -------------------------------------------------------------


def test_add_matched_errors_keeps_pooled_and_per_bucket_errors() -> None:
    start = _tomorrow_noon()
    metrics = new_slot_metrics()
    predictions = [
        _prediction(start, 3, 1.1),
        _prediction(start, 30, 1.4),
        _prediction(start, -1, 2.0),  # stored after the slot started: not a forecast
    ]

    add_matched_errors(metrics, predictions, 1.0)

    assert metrics["errors"] == pytest.approx([0.1, 0.4, 1.0])
    assert metrics["count"] == 3
    assert metrics["bucket_errors"] == {
        "day_1": [pytest.approx(0.1)],
        "day_2": [pytest.approx(0.4)],
    }


def test_add_matched_errors_bounds_every_list() -> None:
    start = _tomorrow_noon()
    metrics = new_slot_metrics()

    for _ in range(MAX_ERROR_SAMPLES + 10):
        add_matched_errors(metrics, [_prediction(start, 3, 1.1)], 1.0)

    assert len(metrics["errors"]) == MAX_ERROR_SAMPLES
    assert len(metrics["bucket_errors"]["day_1"]) == MAX_ERROR_SAMPLES
    assert metrics["count"] == MAX_ERROR_SAMPLES + 10


def test_learning_from_an_actual_price_updates_each_bucket(
    predictor: SpotPricePredictor,
) -> None:
    start = _tomorrow_noon()
    for lead_hours, price in [
        (3, 1.1),
        (6, 1.1),
        (9, 1.1),
        (30, 1.4),
        (33, 1.4),
        (36, 1.4),
    ]:
        predictor.storage.insert_prediction(
            start=start.isoformat(),
            price=price,
            confidence=0.8,
            hour=start.hour,
            minute=start.minute,
            stored_at=(start - timedelta(hours=lead_hours)).isoformat(),
        )

    assert predictor.learn_from_actual_price(start.isoformat(), 1.0)

    assert predictor.bias_correction == {
        SLOT: {"day_1": pytest.approx(0.1), "day_2": pytest.approx(0.4)}
    }
    assert predictor.error_metrics[SLOT]["bucket_errors"] == {
        "day_1": pytest.approx([0.1] * 3),
        "day_2": pytest.approx([0.4] * 3),
    }


def test_learning_metrics_report_the_offsets_per_bucket(
    predictor: SpotPricePredictor,
) -> None:
    predictor.error_metrics = {
        SLOT: _metrics({"day_1": [0.1] * 3}),
        50: _metrics({"day_2": [0.2] * 3}),
    }
    predictor.bias_correction = {SLOT: {"day_1": 0.1, "day_2": 0.3}, 50: {"day_2": 0.5}}

    metrics = predictor.get_learning_metrics()

    assert metrics["bias_corrections"] == 2
    assert metrics["bias_offsets"] == {
        "day_1": {"slots": 1, "mean_offset": pytest.approx(0.1)},
        "day_2": {"slots": 2, "mean_offset": pytest.approx(0.4)},
        "day_3": {"slots": 0, "mean_offset": None},
        "day_4_plus": {"slots": 0, "mean_offset": None},
    }
    slot_metrics = metrics["hourly_metrics"][str(SLOT)]
    assert slot_metrics["bias_correction"] == pytest.approx(0.1)
    assert slot_metrics["bias_offsets"] == {
        "day_1": pytest.approx(0.1),
        "day_2": pytest.approx(0.3),
    }
    # A slot without a day_1 offset reports 0.0 as before, plus its buckets
    assert metrics["hourly_metrics"]["50"]["bias_correction"] == pytest.approx(0.0)
    assert metrics["hourly_metrics"]["50"]["bias_offsets"] == {
        "day_2": pytest.approx(0.5)
    }


@pytest.mark.asyncio
async def test_reset_learning_clears_the_offsets(predictor: SpotPricePredictor) -> None:
    predictor.bias_correction = {SLOT: {"day_1": 0.1}}

    with patch.object(
        predictor.storage, "async_clear_storage", AsyncMock(return_value=True)
    ):
        await predictor.reset_learning()

    assert predictor.bias_correction == {}


# --- Migration ---------------------------------------------------------------------


def _bias_offsets(storage: LearningStorage) -> dict[int, dict[str, float]]:
    """Return the stored bias offsets through the bulk loader."""
    data = storage.load_all()
    assert data is not None
    offsets: dict[int, dict[str, float]] = data["bias_correction"]
    return offsets


def _schema_version(db_path: Path) -> int:
    with closing(sqlite3.connect(db_path)) as conn:
        row = conn.execute(
            "SELECT value FROM meta WHERE key = 'schema_version'"
        ).fetchone()
    return int(row[0])


def test_upgrade_keeps_the_pooled_offsets_as_day_1(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    storage = LearningStorage(_hass(tmp_path), "DK1")
    db_path = storage.db_path
    storage.close()
    with closing(sqlite3.connect(db_path)) as conn:
        # A v8 database: one offset per slot, no lead-time bucket column
        conn.executescript(
            "DROP TABLE bias_correction;"
            "CREATE TABLE bias_correction (hour INTEGER PRIMARY KEY, correction REAL NOT NULL);"
        )
        conn.executemany(
            "INSERT INTO bias_correction (hour, correction) VALUES (?, ?)",
            [(SLOT, 0.12), (SLOT + 1, -0.05)],
        )
        conn.execute("UPDATE meta SET value = '8' WHERE key = 'schema_version'")
        conn.commit()

    with caplog.at_level(logging.INFO):
        upgraded = LearningStorage(_hass(tmp_path), "DK1")
    try:
        assert _bias_offsets(upgraded) == {
            SLOT: {BIAS_FALLBACK_BUCKET: pytest.approx(0.12)},
            SLOT + 1: {BIAS_FALLBACK_BUCKET: pytest.approx(-0.05)},
        }
        assert _schema_version(db_path) >= LEAD_TIME_BIAS_SCHEMA_VERSION
        assert "kept 2 offsets as the day_1 offsets" in caplog.text

        # The other buckets are stored next to the migrated ones
        upgraded.save_all({"bias_correction": {SLOT: {"day_2": 0.3}}})
    finally:
        upgraded.close()
    reopened = LearningStorage(_hass(tmp_path), "DK1")
    try:
        assert _bias_offsets(reopened)[SLOT] == {
            BIAS_FALLBACK_BUCKET: pytest.approx(0.12),
            "day_2": pytest.approx(0.3),
        }
    finally:
        reopened.close()


def test_new_database_starts_at_the_lead_time_bias_schema(tmp_path: Path) -> None:
    storage = LearningStorage(_hass(tmp_path), "DK1")
    try:
        assert _schema_version(storage.db_path) >= LEAD_TIME_BIAS_SCHEMA_VERSION
        with closing(sqlite3.connect(storage.db_path)) as conn:
            columns = [c[1] for c in conn.execute("PRAGMA table_info(bias_correction)")]
        assert columns == ["hour", "bucket", "correction"]
    finally:
        storage.close()


def test_every_lead_time_bucket_can_hold_an_offset(tmp_path: Path) -> None:
    storage = LearningStorage(_hass(tmp_path), "DK1")
    try:
        storage.save_all({"bias_correction": {SLOT: dict.fromkeys(BUCKETS, 0.1)}})
        assert set(_bias_offsets(storage)[SLOT]) == set(BUCKETS)
    finally:
        storage.close()
