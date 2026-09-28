"""Tests for ModelMixin training: live model fit, holdout validation and HPO."""

import logging
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from custom_components.open_spot_forecast.ml import models
from custom_components.open_spot_forecast.ml.features import (
    FEATURE_NAMES,
    NORDPOOL_FEATURES,
    with_masked_nordpool,
)
from custom_components.open_spot_forecast.ml.gbm import NumpyGradientBoosting
from custom_components.open_spot_forecast.ml.models import (
    MIN_SAMPLES_LEAF,
    create_price_model,
)
from custom_components.open_spot_forecast.ml.predictor import SpotPricePredictor
from custom_components.open_spot_forecast.ml.series_storage import NORDPOOL_PROGNOSES

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
    # The first argument is the training time
    mae, rmse = record.args[1:3]
    shift = SHIFTED_LEVEL - BASE_LEVEL
    assert mae == pytest.approx(shift, abs=5.0)
    assert rmse == pytest.approx(shift, abs=5.0)


# --- Nordpool-masked training copies (#91) ---------------------------------------

NORDPOOL_COLUMNS = [FEATURE_NAMES.index(name) for name in NORDPOOL_FEATURES]


def _store_prognoses(predictor: SpotPricePredictor, days: range) -> None:
    """Store an hourly Nordpool prognosis for every local hour of fixture days."""
    tz = ZoneInfo("Europe/Copenhagen")
    rows = [
        {
            "timestamp": datetime(2026, 9, 14 + day, hour, tzinfo=tz)
            .astimezone(UTC)
            .strftime("%Y-%m-%dT%H:%M:%SZ"),
            "consumption": 3000.0 + hour,
            "solar": 100.0,
            "wind_offshore": 800.0,
            "wind_onshore": 700.0,
        }
        for day in days
        for hour in range(24)
    ]
    predictor.storage.upsert_series(NORDPOOL_PROGNOSES, rows)


def _has_prognosis(X: np.ndarray) -> np.ndarray:
    return ~np.isnan(X[:, NORDPOOL_COLUMNS]).all(axis=1)


def test_masked_copies_mask_only_the_nordpool_columns() -> None:
    """Rows with a prognosis get a copy with it NaN and the same target."""
    X = np.arange(3 * (len(FEATURE_NAMES) + 1), dtype=float).reshape(3, -1)
    X[1, NORDPOOL_COLUMNS] = np.nan  # a row without prognoses is not copied
    y = np.array([1.0, 2.0, 3.0])

    rows, targets = with_masked_nordpool(X, y)

    assert rows.shape == (5, X.shape[1])
    np.testing.assert_array_equal(rows[:3], X)
    assert targets.tolist() == pytest.approx([1.0, 2.0, 3.0, 1.0, 3.0])
    assert np.isnan(rows[3:, NORDPOOL_COLUMNS]).all()
    others = [c for c in range(X.shape[1]) if c not in NORDPOOL_COLUMNS]
    np.testing.assert_array_equal(rows[3:, others], X[[0, 2]][:, others])


def test_masked_copies_skip_rows_without_prognoses() -> None:
    X = np.full((2, len(FEATURE_NAMES)), np.nan)
    y = np.array([1.0, 2.0])

    rows, targets = with_masked_nordpool(X, y)

    assert rows is X
    assert targets is y


def test_training_fits_masked_copies_on_each_side_of_the_split(
    predictor: SpotPricePredictor,
) -> None:
    """The holdout copy never fits a masked copy of a holdout row.

    Monday to Thursday are the fit side and Friday the holdout; with
    prognoses on every day, each side has its own rows twice, and the live
    model gets all five days twice.
    """
    _store_prognoses(predictor, range(5))
    fits: list[tuple[np.ndarray, np.ndarray]] = []
    predicted: list[np.ndarray] = []
    real_fit = NumpyGradientBoosting.fit
    real_predict = NumpyGradientBoosting.predict

    def spy_fit(model: NumpyGradientBoosting, X: np.ndarray, y: np.ndarray) -> None:
        fits.append((X, y))
        real_fit(model, X, y)

    def spy_predict(model: NumpyGradientBoosting, X: np.ndarray) -> np.ndarray:
        predicted.append(X)
        return real_predict(model, X)

    with (
        patch.object(NumpyGradientBoosting, "fit", spy_fit),
        patch.object(NumpyGradientBoosting, "predict", spy_predict),
    ):
        _train(predictor)

    (holdout_X, holdout_y), (live_X, live_y) = fits
    fit_days, all_days = 4 * SLOTS_PER_DAY, 5 * SLOTS_PER_DAY
    assert len(holdout_X) == len(holdout_y) == 2 * fit_days
    assert len(live_X) == len(live_y) == 2 * all_days
    # Both copies of a fit row have the fit side's (Monday-Thursday) targets
    assert float(np.max(holdout_y)) < SHIFTED_LEVEL - 20
    np.testing.assert_array_equal(holdout_y[:fit_days], holdout_y[fit_days:])
    assert _has_prognosis(holdout_X[:fit_days]).all()
    assert not _has_prognosis(holdout_X[fit_days:]).any()
    # The holdout score covers Friday with and without its prognosis
    (scored,) = predicted
    assert len(scored) == 2 * SLOTS_PER_DAY
    assert _has_prognosis(scored[:SLOTS_PER_DAY]).all()
    assert not _has_prognosis(scored[SLOTS_PER_DAY:]).any()


def test_prediction_rows_are_not_copied(predictor: SpotPricePredictor) -> None:
    """Prediction keeps one row per slot, prognoses as given (#91)."""
    _store_prognoses(predictor, range(5))
    _train(predictor)
    features = [
        {"consumption_forecast": 3000.0, "wind_offshore": 800.0} for _ in range(96)
    ]
    batch = Mock(wraps=predictor.price_model.predict)

    with patch.object(predictor.price_model, "predict", batch):
        predictor._generate_predictions(features, 1, 15)

    (rows,) = batch.call_args.args
    assert rows.shape == (96, len(FEATURE_NAMES))
    assert _has_prognosis(rows).all()


def test_the_masked_model_predicts_without_prognoses(
    predictor: SpotPricePredictor,
) -> None:
    """Without prognoses the model follows the other features, not a branch.

    Every fixture day has prognoses, so without the masked copies a
    prediction row without them would fall into whichever child had more
    training rows at every Nordpool split.
    """
    _store_prognoses(predictor, range(5))
    _train(predictor)
    all_prices, all_features = predictor.get_all_historical_prices()
    X = predictor._model_inputs(all_features)
    X[:, NORDPOOL_COLUMNS] = np.nan

    friday = predictor.price_model.predict(X[4 * SLOTS_PER_DAY :])

    assert float(np.mean(friday)) == pytest.approx(SHIFTED_LEVEL, abs=5.0)
    assert len(all_prices) == 5 * SLOTS_PER_DAY


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


def test_hpo_splits_before_adding_masked_copies(
    predictor: SpotPricePredictor, monkeypatch: pytest.MonkeyPatch
) -> None:
    """HPO fits on the oldest rows and their copies, never on a later row."""
    monkeypatch.setattr(models, "HPO_N_ESTIMATORS", (5,))
    monkeypatch.setattr(models, "HPO_LEARNING_RATES", (0.1,))
    monkeypatch.setattr(models, "HPO_MAX_DEPTHS", (2,))
    profile = 10 * np.sin(2 * np.pi * np.arange(SLOTS_PER_DAY) / SLOTS_PER_DAY)
    predictor.price_history = [
        {"date": f"2026-09-{14 + day}", "prices": (day + profile).tolist()}
        for day in range(10)
    ]
    _store_prognoses(predictor, range(10))
    fits: list[np.ndarray] = []
    real_fit = NumpyGradientBoosting.fit

    def spy_fit(model: NumpyGradientBoosting, X: np.ndarray, y: np.ndarray) -> None:
        fits.append(y)
        real_fit(model, X, y)

    with patch.object(NumpyGradientBoosting, "fit", spy_fit):
        assert predictor._optimize_hyperparameters() is not None

    (y,) = fits
    assert len(y) == 2 * 8 * SLOTS_PER_DAY
    # Day 8 and 9 (levels 8 and 9) are validation only
    assert float(np.max(y)) < 8 + 10


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
