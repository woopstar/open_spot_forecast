"""Tests for the near-term price model of the first forecast days (#135)."""

import logging
from collections.abc import Callable, Iterator
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, Mock, patch
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from custom_components.open_spot_forecast.const import NEAR_TERM_REGIONS
from custom_components.open_spot_forecast.ml.features import FEATURE_NAMES
from custom_components.open_spot_forecast.ml.gbm import NumpyGradientBoosting
from custom_components.open_spot_forecast.ml.predictor import SpotPricePredictor
from custom_components.open_spot_forecast.ml.price_lags import (
    NEAR_TERM_DAYS,
    PriceLagIndex,
)

TZ = ZoneInfo("Europe/Copenhagen")
FIRST = date(2026, 6, 1)
DAYS = 8
LAST = FIRST + timedelta(days=DAYS - 1)
SLOTS = 96
LAG = len(FEATURE_NAMES)  # the lag column follows the canonical features


def _price(day: int, slot: int) -> float:
    """A price unique to (day offset from FIRST, slot index)."""
    return day + slot / 1000


async def _run_inline(func: Callable[..., Any], *args: Any) -> Any:
    return func(*args)


def _predictor(tmp_path: Path, region: str) -> SpotPricePredictor:
    """A predictor with eight days of prices and a small model."""
    hass = Mock()
    hass.config.path.return_value = str(tmp_path / ".storage")
    hass.async_add_executor_job = _run_inline
    predictor = SpotPricePredictor(hass, region)
    predictor.price_model = NumpyGradientBoosting(n_estimators=5, min_samples_leaf=20)
    predictor.price_history = [
        {
            "date": (FIRST + timedelta(days=day)).isoformat(),
            "prices": [_price(day, slot) for slot in range(SLOTS)],
        }
        for day in range(DAYS)
    ]
    return predictor


@pytest.fixture
def predictor(tmp_path: Path) -> Iterator[SpotPricePredictor]:
    """A DK1 predictor: a region with the near-term model."""
    predictor = _predictor(tmp_path, "DK1")
    yield predictor
    predictor.storage.close()


@pytest.fixture
def one_model(tmp_path: Path) -> Iterator[SpotPricePredictor]:
    """An SE3 predictor: a region outside ``NEAR_TERM_REGIONS``."""
    assert "SE3" not in NEAR_TERM_REGIONS
    predictor = _predictor(tmp_path, "SE3")
    yield predictor
    predictor.storage.close()


def _recorded_fits(predictor: SpotPricePredictor) -> list[np.ndarray]:
    """Train and return the rows of every fit, in order."""
    rows: list[np.ndarray] = []
    real_fit = NumpyGradientBoosting.fit

    def fit(model: NumpyGradientBoosting, X: np.ndarray, y: np.ndarray) -> None:
        rows.append(X)
        real_fit(model, X, y)

    with patch.object(NumpyGradientBoosting, "fit", autospec=True, side_effect=fit):
        predictor._train_models()
    return rows


def _features(predictor: SpotPricePredictor, leads: range) -> list[dict]:
    """One 09:15 prediction row per day ``leads`` days after the last known day."""
    starts = [
        datetime.combine(LAST + timedelta(days=lead), datetime.min.time(), tzinfo=TZ)
        + timedelta(hours=9, minutes=15)
        for lead in leads
    ]
    return predictor._combine_features(
        [{"start": start.isoformat()} for start in starts], {}
    )


def _fake_model(price: float) -> MagicMock:
    """A fitted model that predicts ``price`` for every row."""
    model = MagicMock()
    model.trees = [object()]
    model.predict.side_effect = lambda rows: np.full(len(rows), price)
    return model


def test_training_fits_the_near_model_on_yesterdays_slot_price(
    predictor: SpotPricePredictor,
) -> None:
    """Holdout copy and price model keep their rows; the near model adds the lag."""
    holdout, plain, near = _recorded_fits(predictor)

    assert predictor.is_trained is True
    assert predictor.near_model.trees
    assert predictor.near_model.get_params() == predictor.price_model.get_params()
    assert holdout.shape[1] == plain.shape[1] == len(FEATURE_NAMES)
    assert near.shape == (DAYS * SLOTS, len(FEATURE_NAMES) + 1)
    np.testing.assert_array_equal(np.delete(near, LAG, axis=1), plain)
    # The first day has no day before it; every later row has yesterday's slot
    assert np.isnan(near[:SLOTS, LAG]).all()
    expected = [_price(day - 1, slot) for day in range(1, DAYS) for slot in range(96)]
    assert near[SLOTS:, LAG] == pytest.approx(expected)


def test_a_region_without_the_near_term_model_trains_and_predicts_with_one_model(
    one_model: SpotPricePredictor,
) -> None:
    """Outside NEAR_TERM_REGIONS there is no second fit and no routing."""
    fits = _recorded_fits(one_model)

    assert [rows.shape[1] for rows in fits] == [len(FEATURE_NAMES)] * 2
    assert one_model.is_trained is True
    assert not one_model.near_model.trees

    features = _features(one_model, range(1, 4))
    expected = one_model.price_model.predict(one_model._model_inputs(features))
    one_model._generate_predictions(features, 3, 15)

    assert [p["price"] for p in one_model.predictions] == pytest.approx(expected)


def test_the_first_forecast_days_come_from_the_near_model(
    predictor: SpotPricePredictor,
) -> None:
    """Days 1-2 after the last known day: near model; later days: price model."""
    predictor.is_trained = True
    predictor.price_model = _fake_model(1.0)
    predictor.near_model = _fake_model(2.0)
    features = _features(predictor, range(1, 8))

    predictor._generate_predictions(features, 7, 15)

    assert [p["price"] for p in predictor.predictions] == pytest.approx(
        [2.0] * NEAR_TERM_DAYS + [1.0] * (7 - NEAR_TERM_DAYS)
    )
    (near_rows,) = predictor.near_model.predict.call_args.args
    assert near_rows.shape == (NEAR_TERM_DAYS, len(FEATURE_NAMES) + 1)
    # Both days lag to the last known day's 09:15 slot, never to their own day
    assert near_rows[:, LAG] == pytest.approx([_price(DAYS - 1, 37)] * NEAR_TERM_DAYS)


def test_later_days_never_reach_the_near_model(predictor: SpotPricePredictor) -> None:
    predictor.is_trained = True
    predictor.price_model = _fake_model(1.0)
    predictor.near_model = _fake_model(2.0)

    predictor._generate_predictions(_features(predictor, range(3, 8)), 5, 15)

    predictor.near_model.predict.assert_not_called()
    assert [p["price"] for p in predictor.predictions] == pytest.approx([1.0] * 5)


def test_a_failed_near_fit_leaves_every_day_to_the_price_model(
    predictor: SpotPricePredictor, caplog: pytest.LogCaptureFixture
) -> None:
    """The forecast must not depend on the second model."""
    with (
        patch.object(predictor, "_near_term", side_effect=RuntimeError("no lag")),
        caplog.at_level(logging.WARNING),
    ):
        predictor._train_models()

    assert predictor.is_trained is True
    assert predictor.price_model.trees
    assert not predictor.near_model.trees
    assert "Near-term price model not trained" in caplog.text

    features = _features(predictor, range(1, 4))
    expected = predictor.price_model.predict(predictor._model_inputs(features))
    predictor._generate_predictions(features, 3, 15)

    assert [p["price"] for p in predictor.predictions] == pytest.approx(expected)


def test_a_failed_training_clears_the_near_model(
    predictor: SpotPricePredictor,
) -> None:
    predictor._train_models()
    assert predictor.near_model.trees

    with patch.object(predictor.price_model, "fit", side_effect=RuntimeError("no fit")):
        predictor._train_models()

    assert predictor.is_trained is False
    assert not predictor.near_model.trees


def test_a_forecast_run_stores_todays_prices_before_it_reads_the_lag(
    predictor: SpotPricePredictor,
) -> None:
    """``predict`` lags the first forecast day to the prices it was just given."""
    today = [9.0 + slot / 1000 for slot in range(SLOTS)]
    predictor.price_history = []
    near_inputs: list[np.ndarray] = []
    real_predict = NumpyGradientBoosting.predict

    def predict(model: NumpyGradientBoosting, X: np.ndarray) -> np.ndarray:
        if X.shape[1] == len(FEATURE_NAMES) + 1:
            near_inputs.append(X)
        return real_predict(model, X)

    with patch.object(
        NumpyGradientBoosting, "predict", autospec=True, side_effect=predict
    ):
        predictor.predict({}, today, forecast_days=4)

    lags = PriceLagIndex.from_days(predictor.price_history, predictor.tz)
    starts = [datetime.fromisoformat(p["start"]) for p in predictor.predictions]
    served = lags.near_term(starts, NEAR_TERM_DAYS)
    (rows,) = near_inputs
    assert 0 < len(rows) == int(served.sum()) < len(starts)
    np.testing.assert_array_equal(
        rows[:, LAG], lags.column([s for s, near in zip(starts, served) if near])
    )
    assert np.isin(rows[:, LAG][~np.isnan(rows[:, LAG])], today).all()
