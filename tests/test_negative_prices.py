"""Negative prices end to end and additive bias correction (issue #15)."""

import json
import logging
import sqlite3
from collections.abc import Iterator
from contextlib import closing
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from homeassistant.util import dt as dt_util

from custom_components.open_spot_forecast.ml.bias_storage import (
    ADDITIVE_BIAS_SCHEMA_VERSION,
)
from custom_components.open_spot_forecast.ml.gbm import NumpyGradientBoosting
from custom_components.open_spot_forecast.ml.learning import percent_error
from custom_components.open_spot_forecast.ml.predictor import SpotPricePredictor
from custom_components.open_spot_forecast.ml.storage import LearningStorage
from custom_components.open_spot_forecast.sensor import MLPredictionSensor

SLOTS_PER_DAY = 96
VAT = 0.25
SLOT = 40


def _hass(tmp_path: Path) -> Mock:
    hass = Mock()
    hass.config.path.return_value = str(tmp_path / ".storage")
    return hass


@pytest.fixture
def predictor(tmp_path: Path) -> Iterator[SpotPricePredictor]:
    predictor = SpotPricePredictor(_hass(tmp_path), "DK1")
    yield predictor
    predictor.storage.close()


def _metrics(errors: list[float], actuals: list[float]) -> dict:
    return {
        "errors": errors,
        "abs_errors": [abs(e) for e in errors],
        "pct_errors": [0.0] * len(errors),
        "predictions": [a + e for a, e in zip(actuals, errors, strict=True)],
        "actuals": actuals,
        "count": len(errors),
    }


# --- Additive bias correction ----------------------------------------------------


def test_bias_correction_subtracts_the_offset(predictor: SpotPricePredictor) -> None:
    predictor.bias_correction = {SLOT: 0.2}

    assert predictor.apply_bias_correction(1.0, SLOT) == pytest.approx(0.8)
    assert predictor.apply_bias_correction(-0.5, SLOT) == pytest.approx(-0.7)
    assert predictor.apply_bias_correction(0.1, SLOT) == pytest.approx(-0.1)
    assert predictor.apply_bias_correction(1.0, SLOT + 1) == pytest.approx(1.0)


def test_zero_offset_never_changes_a_prediction(predictor: SpotPricePredictor) -> None:
    """A correct slot (offset 0) keeps negative and positive predictions as they are.

    A multiplicative factor below zero would flip the sign of every prediction.
    """
    predictor.error_metrics = {SLOT: _metrics([0.0] * 5, [-0.4] * 5)}
    predictor._update_bias_correction(SLOT)

    assert predictor.bias_correction[SLOT] == pytest.approx(0.0)
    assert predictor.apply_bias_correction(-0.4, SLOT) == pytest.approx(-0.4)
    assert predictor.apply_bias_correction(0.4, SLOT) == pytest.approx(0.4)


def test_first_update_sets_the_offset_then_follows_the_ema(
    predictor: SpotPricePredictor,
) -> None:
    """offset = 0.9 * old + 0.1 * (old + mean_error), starting at mean_error."""
    predictor.error_metrics = {SLOT: _metrics([0.3, 0.2, 0.4], [-1.0, -1.0, -1.0])}
    predictor._update_bias_correction(SLOT)
    assert predictor.bias_correction[SLOT] == pytest.approx(0.3)

    predictor.error_metrics = {SLOT: _metrics([0.1, 0.1, 0.1], [-1.0, -1.0, -1.0])}
    predictor._update_bias_correction(SLOT)
    assert predictor.bias_correction[SLOT] == pytest.approx(
        0.9 * 0.3 + 0.1 * (0.3 + 0.1)
    )


def test_update_needs_three_samples(predictor: SpotPricePredictor) -> None:
    predictor.error_metrics = {SLOT: _metrics([1.0, 1.0], [-1.0, -1.0])}

    predictor._update_bias_correction(SLOT)

    assert SLOT not in predictor.bias_correction


@pytest.mark.parametrize("actual", [-0.8, 0.0, 0.6])
def test_offset_converges_to_the_model_bias_whatever_the_price_sign(
    predictor: SpotPricePredictor, actual: float
) -> None:
    """With a constant model bias the corrected forecast converges to the actual.

    Errors are measured on corrected predictions, as in the learning loop. The
    offset stays bounded for all-negative and all-zero actuals (no division).
    """
    model_bias = 0.25
    rng = np.random.default_rng(0)
    errors: list[float] = []
    for _ in range(200):
        raw = actual + model_bias + float(rng.normal(0, 0.02))
        corrected = predictor.apply_bias_correction(raw, SLOT)
        errors = (errors + [corrected - actual])[-100:]
        predictor.error_metrics = {SLOT: _metrics(errors, [actual] * len(errors))}
        predictor._update_bias_correction(SLOT)
        assert abs(predictor.bias_correction.get(SLOT, 0.0)) < 1.0

    assert predictor.bias_correction[SLOT] == pytest.approx(model_bias, abs=0.03)
    assert predictor.apply_bias_correction(actual + model_bias, SLOT) == (
        pytest.approx(actual, abs=0.03)
    )


# --- Confidence -------------------------------------------------------------------


def test_confidence_is_learned_for_slots_with_negative_prices(
    predictor: SpotPricePredictor,
) -> None:
    """MAE is taken relative to the mean absolute price, not the mean price."""
    predictor.error_metrics = {SLOT: _metrics([0.1, -0.1] * 3, [-1.0] * 6)}
    feature = {"hour": SLOT // 4, "minute": 0}

    assert predictor._estimate_confidence(feature) == pytest.approx(0.9)


def test_confidence_with_mixed_sign_prices_uses_their_magnitude(
    predictor: SpotPricePredictor,
) -> None:
    """A slot whose mean price is 0 still gets a learned confidence."""
    predictor.error_metrics = {SLOT: _metrics([0.2] * 6, [-1.0, 1.0] * 3)}
    predictor.volatility_mae = {SLOT: 0.5}
    feature = {"hour": SLOT // 4, "minute": 0}

    # 1 - 0.2 / 1.0 = 0.8, minus min(0.25, 0.5 / 1.0 * 0.3) = 0.15
    assert predictor._estimate_confidence(feature) == pytest.approx(0.65)


def test_confidence_falls_back_to_the_heuristic_for_all_zero_prices(
    predictor: SpotPricePredictor,
) -> None:
    predictor.error_metrics = {SLOT: _metrics([0.1] * 6, [0.0] * 6)}
    feature = {"hour": SLOT // 4, "minute": 0, "wind_speed_mean": 5.0}

    # Heuristic: 0.8 - 0.1 (no solar prognosis)
    assert predictor._estimate_confidence(feature) == pytest.approx(0.7)


@pytest.mark.parametrize(
    ("error", "actual", "expected"),
    [(0.1, -0.5, 20.0), (-0.1, 0.5, -20.0), (0.3, 0.0, 0.0)],
)
def test_percent_error_uses_the_price_magnitude(
    error: float, actual: float, expected: float
) -> None:
    assert percent_error(error, actual) == pytest.approx(expected)


# --- Negative predictions end to end ----------------------------------------------


def _negative_history(predictor: SpotPricePredictor) -> None:
    """A week whose prices are negative from 10:00 to 16:00 every day."""
    profile = np.where(
        (np.arange(SLOTS_PER_DAY) >= 40) & (np.arange(SLOTS_PER_DAY) < 64), -0.3, 0.8
    )
    first = datetime(2026, 9, 14)
    predictor.price_history = [
        {
            "date": (first + timedelta(days=day)).strftime("%Y-%m-%d"),
            "prices": profile.tolist(),
        }
        for day in range(7)
    ]


def test_negative_prices_survive_model_bias_correction_and_sensor(
    predictor: SpotPricePredictor,
) -> None:
    """Model → bias correction → stored prediction → sensor attribute, all negative."""
    _negative_history(predictor)
    predictor.price_model = NumpyGradientBoosting(n_estimators=50, min_samples_leaf=20)
    with patch.object(predictor, "store_daily_prices"):
        predictor._train_models()
    predictor.bias_correction = {SLOT: 0.05}
    day = dt_util.now().replace(hour=0, minute=0, second=0, microsecond=0)
    features = predictor._combine_features(
        [
            {"start": (day + timedelta(minutes=15 * slot)).isoformat()}
            for slot in (SLOT, 20)
        ],
        {},
    )

    predictor._generate_predictions(features, 1, 15)

    negative, positive = (p["price"] for p in predictor.predictions)
    assert negative == pytest.approx(-0.3 - 0.05, abs=0.05)
    assert positive == pytest.approx(0.8, abs=0.05)
    stored = predictor.storage.find_predictions_for_timestamp(
        day.strftime("%Y-%m-%d"), SLOT // 4, 0
    )
    assert stored[0]["price"] == pytest.approx(negative)

    sensor = MLPredictionSensor(
        MagicMock(),
        MagicMock(entry_id="test"),
        {"ml_predictor": predictor},
        "DKK",
        VAT,
        4,
        "kWh",
    )
    attribute_prices = [
        p["price"] for p in sensor.extra_state_attributes["predictions"]
    ]
    assert attribute_prices[0] == pytest.approx(round(negative * (1 + VAT), 4))
    assert attribute_prices[0] < 0


def test_heuristic_predictions_can_be_negative(predictor: SpotPricePredictor) -> None:
    predictor._generate_heuristic_predictions([-0.5] * 48, 1, 60)

    assert predictor.predictions
    assert all(p["price"] == pytest.approx(-0.5) for p in predictor.predictions)


def test_learning_from_a_negative_actual_updates_the_offset(
    predictor: SpotPricePredictor,
) -> None:
    start = dt_util.now().replace(hour=10, minute=0, second=0, microsecond=0)
    for _ in range(3):
        predictor.store_prediction_for_learning(start.isoformat(), -0.1, 0.8)

    assert predictor.learn_from_actual_price(start.isoformat(), -0.4)

    slot = 40
    assert predictor.error_metrics[slot]["errors"] == pytest.approx([0.3] * 3)
    assert predictor.bias_correction[slot] == pytest.approx(0.3)
    assert predictor.apply_bias_correction(-0.1, slot) == pytest.approx(-0.4)


# --- Migration ----------------------------------------------------------------------


def _stored_bias(storage: LearningStorage) -> dict[int, float]:
    data = storage.load_all()
    assert data is not None
    bias: dict[int, float] = data["bias_correction"]
    return bias


def _schema_version(db_path: Path) -> int:
    with closing(sqlite3.connect(db_path)) as conn:
        row = conn.execute(
            "SELECT value FROM meta WHERE key = 'schema_version'"
        ).fetchone()
    return int(row[0])


def test_upgrade_resets_multiplicative_factors_once(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Stored factors cannot be converted to offsets; they are reset and logged."""
    storage = LearningStorage(_hass(tmp_path), "DK1")
    db_path = storage.db_path
    storage.close()
    with closing(sqlite3.connect(db_path)) as conn:
        conn.execute("UPDATE meta SET value = '4' WHERE key = 'schema_version'")
        conn.executemany(
            "INSERT OR REPLACE INTO bias_correction (hour, correction) VALUES (?, ?)",
            [(SLOT, 1.12), (SLOT + 1, 0.93)],
        )
        conn.commit()

    with caplog.at_level(logging.INFO):
        upgraded = LearningStorage(_hass(tmp_path), "DK1")
    assert _stored_bias(upgraded) == {}
    assert _schema_version(db_path) == ADDITIVE_BIAS_SCHEMA_VERSION
    assert "reset 2 multiplicative factors" in caplog.text

    upgraded.save_all({"bias_correction": {SLOT: 0.07}})
    upgraded.close()
    reopened = LearningStorage(_hass(tmp_path), "DK1")
    try:
        assert _stored_bias(reopened) == {SLOT: pytest.approx(0.07)}
    finally:
        reopened.close()


def test_new_database_starts_at_the_additive_schema(tmp_path: Path) -> None:
    storage = LearningStorage(_hass(tmp_path), "DK1")
    try:
        assert _schema_version(storage.db_path) == ADDITIVE_BIAS_SCHEMA_VERSION
    finally:
        storage.close()


def test_legacy_json_import_skips_multiplicative_factors(tmp_path: Path) -> None:
    storage_dir = tmp_path / ".storage"
    storage_dir.mkdir()
    (storage_dir / "open_spot_forecast_DK1_learning.json").write_text(
        json.dumps({"bias_correction": {"40": 1.2}, "training_samples": 5}),
        encoding="utf-8",
    )

    storage = LearningStorage(_hass(tmp_path), "DK1")
    try:
        assert _stored_bias(storage) == {}
    finally:
        storage.close()
