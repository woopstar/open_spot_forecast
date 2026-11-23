"""A retrained model is promoted only if it is not worse than the live one (#275).

The candidate is fitted into new objects. A candidate whose holdout MAE is
above the gate, whose predictions are not sane, or whose training fails
leaves the live price and near-term models serving, with their holdout
metrics, and the rejection is kept in meta and shown on the sensor.
"""

import logging
from collections.abc import Callable, Iterator
from datetime import timedelta
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from homeassistant.util import dt as dt_util

from custom_components.open_spot_forecast.ml.gbm import NumpyGradientBoosting
from custom_components.open_spot_forecast.ml.predictor import SpotPricePredictor
from custom_components.open_spot_forecast.ml.promotion import (
    PROMOTION_MAE_FACTOR,
    PROMOTION_MAE_FLOOR,
    PROMOTION_MIN_ROWS,
    REJECTED_ERROR,
    REJECTED_NEAR_CONSTANT,
    REJECTED_NON_FINITE,
    REJECTED_WORSE_MAE,
    previous_mae_to_beat,
    promotion_limit,
    rejection_reason,
)
from custom_components.open_spot_forecast.ml.training_info import (
    REJECTION_META_KEYS,
)
from custom_components.open_spot_forecast.sensor import LearningMetricsSensor

# A live model's holdout MAE no real candidate can match: the gate is the floor
TINY_MAE = 1e-6


def _hass(tmp_path: Path) -> Mock:
    async def run_inline(func: Callable[..., Any], *args: Any) -> Any:
        return func(*args)

    hass = Mock()
    hass.config.path.return_value = str(tmp_path / ".storage")
    hass.async_add_executor_job = run_inline
    hass.async_create_background_task = lambda coro, _name: coro.close()
    return hass


def _history(days: int) -> list[dict[str, Any]]:
    """Return days of noisy daily-profile prices, up to yesterday."""
    rng = np.random.default_rng(7)
    today = dt_util.now().date()
    profile = 0.5 + 0.1 * np.sin(2 * np.pi * np.arange(96) / 96)
    return [
        {
            "date": (today - timedelta(days=days - n)).isoformat(),
            "prices": (profile + rng.normal(0, 0.05, 96)).tolist(),
        }
        for n in range(days)
    ]


@pytest.fixture
def make_predictor(
    tmp_path: Path,
) -> Iterator[Callable[..., SpotPricePredictor]]:
    """Return a factory of DK1 predictors sharing one database (a restart)."""
    created: list[SpotPricePredictor] = []

    def make(days: int = 15) -> SpotPricePredictor:
        predictor = SpotPricePredictor(_hass(tmp_path), "DK1")
        predictor.price_model = NumpyGradientBoosting(
            n_estimators=10, min_samples_leaf=20
        )
        predictor.price_history = _history(days)
        created.append(predictor)
        return predictor

    yield make
    for predictor in created:
        predictor.storage.close()


def _train(predictor: SpotPricePredictor) -> None:
    """Train on the fixture history without appending today's prices."""
    with patch.object(predictor, "store_daily_prices"):
        predictor._train_models()


def _meta(predictor: SpotPricePredictor) -> dict[str, Any]:
    return predictor.storage.load_all() or {}


def _attributes(predictor: SpotPricePredictor) -> dict[str, Any]:
    sensor = LearningMetricsSensor(
        MagicMock(), MagicMock(entry_id="test"), {"ml_predictor": predictor}
    )
    return sensor.extra_state_attributes


def _prediction_rows(predictor: SpotPricePredictor) -> np.ndarray:
    _, features = predictor.get_all_historical_prices()
    return predictor._model_inputs(features[-96:])


# --- The gate -------------------------------------------------------------------


def test_promotion_limit_is_the_factor_or_the_floor() -> None:
    assert promotion_limit(0.2) == pytest.approx(0.2 * PROMOTION_MAE_FACTOR)
    assert promotion_limit(0.001) == pytest.approx(0.001 + PROMOTION_MAE_FLOOR)


@pytest.mark.parametrize(
    ("y_pred", "mae", "previous", "expected"),
    [
        ([0.1, 0.5, 0.9], 0.1, 0.1, None),
        ([0.1, 0.5, 0.9], 0.124, 0.1, None),
        ([0.1, 0.5, 0.9], 0.126, 0.1, REJECTED_WORSE_MAE),
        ([0.1, np.nan, 0.9], 0.1, 0.1, REJECTED_NON_FINITE),
        ([0.1, 0.5, 0.9], float("inf"), None, REJECTED_NON_FINITE),
        ([0.5, 0.5, 0.5], 0.1, 0.1, REJECTED_NEAR_CONSTANT),
        ([0.5, 0.5, 0.5], 0.9, None, None),
    ],
    ids=[
        "equal",
        "within-factor",
        "above-factor",
        "nan",
        "inf-without-model",
        "near-constant",
        "first-model",
    ],
)
def test_rejection_reason(
    y_pred: list[float], mae: float, previous: float | None, expected: str | None
) -> None:
    y_true = np.array([0.0, 0.5, 1.0])

    assert rejection_reason(y_true, np.array(y_pred), mae, previous) == expected


def test_constant_prices_are_not_near_constant_predictions() -> None:
    flat = np.full(3, 0.5)

    assert rejection_reason(flat, flat, 0.0, 0.1) is None


def test_the_mae_is_gated_only_with_a_live_model_and_enough_history() -> None:
    assert previous_mae_to_beat(True, 0.1, PROMOTION_MIN_ROWS) == pytest.approx(0.1)
    assert previous_mae_to_beat(False, 0.1, PROMOTION_MIN_ROWS) is None
    assert previous_mae_to_beat(True, 0.1, PROMOTION_MIN_ROWS - 1) is None
    assert previous_mae_to_beat(True, None, PROMOTION_MIN_ROWS) is None


# --- Rollback ---------------------------------------------------------------------


def test_a_worse_candidate_is_rejected_and_the_live_model_keeps_serving(
    make_predictor: Callable[..., SpotPricePredictor],
    caplog: pytest.LogCaptureFixture,
) -> None:
    predictor = make_predictor()
    _train(predictor)
    price_model, near_model = predictor.price_model, predictor.near_model
    gas_edges, samples = predictor.gas_edges, predictor.training_samples
    rows = _prediction_rows(predictor)
    before = price_model.predict(rows)
    predictor.holdout_mae = TINY_MAE
    # A bad data day: the newest day's prices are far off
    predictor.price_history[-1]["prices"] = [5.0] * 96

    with caplog.at_level(logging.WARNING):
        _train(predictor)

    assert predictor.is_trained is True
    assert predictor.price_model is price_model
    assert predictor.near_model is near_model
    assert predictor.gas_edges is gas_edges
    assert predictor.training_samples == samples
    np.testing.assert_allclose(predictor.price_model.predict(rows), before)
    assert predictor.holdout_mae == pytest.approx(TINY_MAE)
    assert predictor.last_retrain_rejected_reason == REJECTED_WORSE_MAE
    candidate_mae = predictor.last_retrain_rejected_mae
    assert candidate_mae is not None
    assert candidate_mae > promotion_limit(TINY_MAE)
    assert "Retrained price model rejected (worse_mae)" in caplog.text
    assert f"candidate holdout MAE={candidate_mae:.4f}" in caplog.text
    assert "live model's holdout MAE=0.0000" in caplog.text
    meta = _meta(predictor)
    assert meta["last_retrain_rejected_reason"] == REJECTED_WORSE_MAE
    assert float(meta["last_retrain_rejected_mae"]) == pytest.approx(candidate_mae)
    assert predictor.last_retrain_rejected_at is not None
    assert (
        meta["last_retrain_rejected_at"]
        == predictor.last_retrain_rejected_at.isoformat()
    )
    attributes = _attributes(predictor)
    assert attributes["last_retrain_rejected_reason"] == REJECTED_WORSE_MAE
    assert attributes["last_retrain_rejected_mae"] == pytest.approx(candidate_mae)
    assert attributes["last_retrain_rejected_at"] == meta["last_retrain_rejected_at"]


def test_a_good_candidate_replaces_the_live_models(
    make_predictor: Callable[..., SpotPricePredictor],
) -> None:
    predictor = make_predictor()
    _train(predictor)
    price_model, near_model = predictor.price_model, predictor.near_model

    _train(predictor)

    assert predictor.price_model is not price_model
    assert predictor.near_model is not near_model
    assert predictor.price_model.trees
    assert predictor.near_model.trees
    assert predictor.last_retrain_rejected_reason is None
    assert not set(REJECTION_META_KEYS) & set(_meta(predictor))


def test_a_failed_retrain_keeps_the_live_model(
    make_predictor: Callable[..., SpotPricePredictor],
) -> None:
    predictor = make_predictor()
    _train(predictor)
    price_model, near_model = predictor.price_model, predictor.near_model
    mae = predictor.holdout_mae

    with patch.object(NumpyGradientBoosting, "fit", side_effect=RuntimeError("no fit")):
        _train(predictor)

    assert predictor.is_trained is True
    assert predictor.price_model is price_model
    assert predictor.near_model is near_model
    assert predictor.holdout_mae == pytest.approx(mae)
    assert predictor.last_retrain_rejected_reason == REJECTED_ERROR
    assert predictor.last_retrain_rejected_mae is None
    meta = _meta(predictor)
    assert meta["last_retrain_rejected_reason"] == REJECTED_ERROR
    assert "last_retrain_rejected_mae" not in meta


def test_without_a_live_model_the_candidate_is_accepted(
    make_predictor: Callable[..., SpotPricePredictor],
) -> None:
    """After a restart the holdout MAE is restored but no model is in use."""
    predictor = make_predictor()
    predictor.is_trained = True
    predictor.holdout_mae = TINY_MAE

    _train(predictor)

    assert predictor.price_model.trees
    assert predictor.holdout_mae is not None
    assert predictor.holdout_mae > promotion_limit(TINY_MAE)
    assert predictor.last_retrain_rejected_reason is None


def test_a_short_history_skips_the_mae_gate(
    make_predictor: Callable[..., SpotPricePredictor],
) -> None:
    predictor = make_predictor(days=3)
    _train(predictor)
    price_model = predictor.price_model
    predictor.holdout_mae = TINY_MAE

    _train(predictor)

    assert predictor.price_model is not price_model
    assert predictor.last_retrain_rejected_reason is None


def test_a_rejected_candidate_is_not_retrained_on_every_run(
    make_predictor: Callable[..., SpotPricePredictor],
) -> None:
    predictor = make_predictor()
    with patch.object(predictor, "store_daily_prices"):
        predictor.retrain()
        first = predictor.last_trained_at
        predictor.holdout_mae = TINY_MAE
        predictor._prices_updated_at = dt_util.utcnow()
        predictor.retrain()

    assert predictor.last_retrain_rejected_reason == REJECTED_WORSE_MAE
    assert first is not None
    assert predictor.last_trained_at is not None
    assert predictor.last_trained_at >= first
    assert predictor.needs_retraining() is False


@pytest.mark.asyncio
async def test_the_rejection_survives_a_restart(
    make_predictor: Callable[..., SpotPricePredictor],
) -> None:
    first = make_predictor()
    _train(first)
    first.holdout_mae = TINY_MAE
    _train(first)
    await first.save_learning_data()
    first.storage.close()

    restarted = make_predictor()
    await restarted._load_learning_data()

    assert restarted.last_retrain_rejected_at == first.last_retrain_rejected_at
    assert restarted.last_retrain_rejected_reason == REJECTED_WORSE_MAE
    assert restarted.last_retrain_rejected_mae == pytest.approx(
        first.last_retrain_rejected_mae
    )


def test_no_rejection_is_shown_before_one(
    make_predictor: Callable[..., SpotPricePredictor],
) -> None:
    predictor = make_predictor()

    attributes = _attributes(predictor)

    for key in REJECTION_META_KEYS:
        assert attributes[key] is None
