"""Tests for ModelMixin training: live model fit, holdout validation and HPO."""

import logging
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

import numpy as np
import pytest

from custom_components.open_spot_forecast.ml import models
from custom_components.open_spot_forecast.ml.features import FEATURE_NAMES
from custom_components.open_spot_forecast.ml.gbm import NumpyGradientBoosting
from custom_components.open_spot_forecast.ml.models import (
    MIN_SAMPLES_LEAF,
    create_price_model,
)
from custom_components.open_spot_forecast.ml.predictor import SpotPricePredictor

SLOTS_PER_DAY = 96
BASE_LEVEL = 50.0
SHIFTED_LEVEL = 150.0


async def _run_inline(func: Callable[..., Any], *args: Any) -> Any:
    """Run an executor job inline."""
    return func(*args)


@pytest.fixture
def predictor(tmp_path: Path) -> Iterator[SpotPricePredictor]:
    """Return a predictor whose price history jumps in its final 20 %.

    Five consecutive weekdays (Mon 2026-09-14 to Fri 2026-09-18) of 96
    slots each, sharing one daily profile. Monday to Thursday trade around
    BASE_LEVEL and Friday around SHIFTED_LEVEL, so the chronological 80/20
    split falls exactly on the Friday boundary. The production model keeps
    MIN_SAMPLES_LEAF rows per leaf, more than one day, so it could never give
    the single Friday its own leaf; this model allows 20.
    """
    hass = Mock()
    hass.config.path.return_value = str(tmp_path / ".storage")
    hass.async_add_executor_job = _run_inline
    predictor = SpotPricePredictor(hass, "DK1")
    predictor.price_model = NumpyGradientBoosting(n_estimators=200, min_samples_leaf=20)

    profile = 10 * np.sin(2 * np.pi * np.arange(SLOTS_PER_DAY) / SLOTS_PER_DAY)
    levels = [BASE_LEVEL] * 4 + [SHIFTED_LEVEL]
    predictor.price_history = [
        {"date": f"2026-09-{14 + day}", "prices": (level + profile).tolist()}
        for day, level in enumerate(levels)
    ]

    yield predictor
    predictor.storage.close()


def _train(predictor: SpotPricePredictor) -> None:
    """Train on the fixture history without appending today's prices."""
    with patch.object(predictor, "store_daily_prices"):
        predictor._train_models()


def test_live_model_is_fitted_on_all_rows_and_follows_level_shift(
    predictor: SpotPricePredictor,
) -> None:
    """The live model sees 100 % of the rows, including the latest days.

    If it were fitted on the oldest 80 % only, it would never see Friday's
    new price level and would keep predicting the old one.
    """
    live_fit = Mock(wraps=predictor.price_model.fit)
    with patch.object(predictor.price_model, "fit", live_fit):
        _train(predictor)

    assert predictor.is_trained is True
    live_fit.assert_called_once()
    X, y = live_fit.call_args.args
    assert len(X) == len(y) == 5 * SLOTS_PER_DAY

    friday = predictor.price_model.predict(X[4 * SLOTS_PER_DAY :])
    assert float(np.mean(friday)) == pytest.approx(SHIFTED_LEVEL, abs=5.0)


def test_holdout_metrics_use_chronological_split(
    predictor: SpotPricePredictor, caplog: pytest.LogCaptureFixture
) -> None:
    """Holdout MAE/RMSE come from a model that never saw the final 20 %.

    Friday is the holdout, so a model fitted on Monday to Thursday misses
    the jump by the full level shift. Scoring the live model (which saw
    Friday) instead would log an error close to zero.
    """
    with caplog.at_level(logging.INFO):
        _train(predictor)

    record = next(
        r for r in caplog.records if r.getMessage().startswith("ML model trained")
    )
    assert isinstance(record.args, tuple)
    mae, rmse = record.args[:2]
    shift = SHIFTED_LEVEL - BASE_LEVEL
    assert mae == pytest.approx(shift, abs=5.0)
    assert rmse == pytest.approx(shift, abs=5.0)


# --- Hyperparameter optimization (issue #14) -------------------------------------


def test_hpo_searches_max_depth_and_persists_best_params(
    predictor: SpotPricePredictor, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The grid covers max_depth; the winner is applied unfitted and saved."""
    monkeypatch.setattr(models, "HPO_N_ESTIMATORS", (5, 10))
    monkeypatch.setattr(models, "HPO_LEARNING_RATES", (0.1,))
    monkeypatch.setattr(models, "HPO_MAX_DEPTHS", (1, 3))
    profile = 10 * np.sin(2 * np.pi * np.arange(SLOTS_PER_DAY) / SLOTS_PER_DAY)
    predictor.price_history = [
        {"date": f"2026-09-{1 + day:02d}", "prices": (BASE_LEVEL + profile).tolist()}
        for day in range(8)
    ]

    best = predictor._optimize_hyperparameters()

    assert best is not None
    assert best["n_estimators"] in (5, 10)
    assert best["max_depth"] in (1, 3)
    assert predictor.is_trained is False
    assert predictor.price_model.trees == []
    assert predictor.price_model.max_depth == best["max_depth"]
    assert predictor.price_model.n_estimators == best["n_estimators"]
    meta = predictor.storage.load_meta_dict()
    assert meta["hpo_max_depth"] == str(best["max_depth"])
    assert meta["hpo_n_estimators"] == str(best["n_estimators"])


def test_hpo_skipped_without_a_week_of_history(predictor: SpotPricePredictor) -> None:
    assert predictor._optimize_hyperparameters() is None


def _restart(predictor: SpotPricePredictor) -> SpotPricePredictor:
    """Return a new predictor on the same storage, as after a restart."""
    predictor.storage.close()
    return SpotPricePredictor(predictor.hass, "DK1")


@pytest.mark.asyncio
async def test_restore_applies_saved_hyperparameters(
    predictor: SpotPricePredictor,
) -> None:
    predictor.storage.save_meta_dict(
        {"hpo_n_estimators": "300", "hpo_learning_rate": "0.05", "hpo_max_depth": "4"}
    )
    restarted = _restart(predictor)
    try:
        await restarted._load_learning_data()

        assert restarted.price_model.n_estimators == 300
        assert restarted.price_model.learning_rate == pytest.approx(0.05)
        assert restarted.price_model.max_depth == 4
    finally:
        restarted.storage.close()


@pytest.mark.asyncio
async def test_restore_ignores_hyperparameters_tuned_for_stumps(
    predictor: SpotPricePredictor,
) -> None:
    """Parameters saved before max_depth existed fall back to the defaults."""
    predictor.storage.save_meta_dict(
        {"hpo_n_estimators": "300", "hpo_learning_rate": "0.2"}
    )
    restarted = _restart(predictor)
    try:
        await restarted._load_learning_data()

        assert restarted.price_model.get_params() == create_price_model().get_params()
    finally:
        restarted.storage.close()


def test_predictions_use_one_batch_model_call(predictor: SpotPricePredictor) -> None:
    """Every slot is predicted in one call, not one tree walk per slot."""
    _train(predictor)
    features = [{"hour": slot // 4, "minute": 15 * (slot % 4)} for slot in range(96)]
    batch = Mock(wraps=predictor.price_model.predict)

    with patch.object(predictor.price_model, "predict", batch):
        predictor._generate_predictions(features, 1, 15)

    batch.assert_called_once()
    assert batch.call_args.args[0].shape == (96, len(FEATURE_NAMES))
    assert len(predictor.predictions) == 96


def test_production_model_uses_depth_limited_trees() -> None:
    """The price model grows trees deeper than a stump, with day-sized leaves."""
    params = create_price_model().get_params()

    assert params["max_depth"] > 1
    assert params["min_samples_leaf"] == MIN_SAMPLES_LEAF
    assert min(models.HPO_MAX_DEPTHS) > 1
