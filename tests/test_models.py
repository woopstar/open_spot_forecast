"""Tests for ModelMixin training: live model fit and holdout validation."""

import logging
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest

from custom_components.open_spot_forecast.ml.predictor import SpotPricePredictor

SLOTS_PER_DAY = 96
BASE_LEVEL = 50.0
SHIFTED_LEVEL = 150.0


@pytest.fixture
def predictor(tmp_path: Path) -> Iterator[SpotPricePredictor]:
    """Return a predictor whose price history jumps in its final 20 %.

    Five consecutive weekdays (Mon 2026-09-14 to Fri 2026-09-18) of 96
    slots each, sharing one daily profile. Monday to Thursday trade around
    BASE_LEVEL and Friday around SHIFTED_LEVEL, so the chronological 80/20
    split falls exactly on the Friday boundary.
    """
    hass = Mock()
    hass.config.path.return_value = str(tmp_path / ".storage")
    predictor = SpotPricePredictor(hass, "DK1")

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
        predictor._train_models([], [])


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
