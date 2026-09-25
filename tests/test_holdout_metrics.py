"""The price model's holdout MAE/RMSE are persisted and shown (#58).

Each training fits a copy of the model on the oldest 80 % of the history and
scores it on the newest 20 %. The result is kept in ``meta`` and exposed as
learning-metrics sensor attributes, so it survives a restart.
"""

from collections.abc import Callable, Iterator
from datetime import timedelta
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, Mock, patch

import pytest

from homeassistant.util import dt as dt_util

from custom_components.open_spot_forecast.ml.models import HOLDOUT_META_KEYS
from custom_components.open_spot_forecast.ml.predictor import SpotPricePredictor
from custom_components.open_spot_forecast.sensor import LearningMetricsSensor


def _hass(tmp_path: Path) -> Mock:
    async def run_inline(func: Callable[..., Any], *args: Any) -> Any:
        return func(*args)

    hass = Mock()
    hass.config.path.return_value = str(tmp_path / ".storage")
    hass.async_add_executor_job = run_inline
    # The Nordpool backfill is a network task: never run it here
    hass.async_create_background_task = lambda coro, _name: coro.close()
    return hass


@pytest.fixture
def make_predictor(
    tmp_path: Path,
) -> Iterator[Callable[[], SpotPricePredictor]]:
    """Return a factory of predictors sharing one database (a restart)."""
    created: list[SpotPricePredictor] = []

    def make() -> SpotPricePredictor:
        predictor = SpotPricePredictor(_hass(tmp_path), "DK1")
        created.append(predictor)
        return predictor

    yield make
    for predictor in created:
        predictor.storage.close()


def _with_history(predictor: SpotPricePredictor, days: int = 3) -> None:
    """Give the predictor a few days of hour-shaped spot prices."""
    today = dt_util.now().date()
    predictor.price_history = [
        {
            "date": (today - timedelta(days=days - n)).isoformat(),
            "prices": [0.3 + 0.01 * (slot // 4) for slot in range(96)],
        }
        for n in range(days)
    ]


def _sensor_attributes(predictor: SpotPricePredictor) -> dict[str, Any]:
    sensor = LearningMetricsSensor(
        MagicMock(), MagicMock(entry_id="test"), {"ml_predictor": predictor}
    )
    return sensor.extra_state_attributes


def test_training_stores_the_holdout_metrics_in_meta(
    make_predictor: Callable[[], SpotPricePredictor],
) -> None:
    predictor = make_predictor()
    _with_history(predictor)
    before = dt_util.utcnow()

    predictor._train_models()

    assert predictor.is_trained
    assert predictor.holdout_mae is not None
    assert predictor.holdout_rmse is not None
    assert predictor.holdout_rmse >= predictor.holdout_mae >= 0
    assert predictor.holdout_trained_at is not None
    assert predictor.holdout_trained_at >= before
    meta = predictor.storage.load_meta_dict()
    assert float(meta["holdout_mae"]) == pytest.approx(predictor.holdout_mae)
    assert float(meta["holdout_rmse"]) == pytest.approx(predictor.holdout_rmse)
    assert meta["holdout_trained_at"] == predictor.holdout_trained_at.isoformat()


@pytest.mark.asyncio
async def test_holdout_metrics_survive_a_restart_and_show_as_attributes(
    make_predictor: Callable[[], SpotPricePredictor],
) -> None:
    first = make_predictor()
    _with_history(first)
    first._train_models()
    await first.save_learning_data()
    first.storage.close()

    restarted = make_predictor()
    await restarted._load_learning_data()

    assert restarted.holdout_mae == pytest.approx(first.holdout_mae)
    assert restarted.holdout_rmse == pytest.approx(first.holdout_rmse)
    assert restarted.holdout_trained_at == first.holdout_trained_at
    attributes = _sensor_attributes(restarted)
    assert attributes["holdout_mae"] == pytest.approx(first.holdout_mae)
    assert attributes["holdout_rmse"] == pytest.approx(first.holdout_rmse)
    assert first.holdout_trained_at is not None
    assert attributes["holdout_trained_at"] == first.holdout_trained_at.isoformat()


def test_holdout_metrics_are_none_before_the_first_training(
    make_predictor: Callable[[], SpotPricePredictor],
) -> None:
    predictor = make_predictor()

    metrics = predictor.get_learning_metrics()

    assert metrics["status"] == "idle"
    for key in HOLDOUT_META_KEYS:
        assert metrics[key] is None
        assert _sensor_attributes(predictor)[key] is None


def test_holdout_metrics_join_the_self_learning_metrics(
    make_predictor: Callable[[], SpotPricePredictor],
) -> None:
    predictor = make_predictor()
    _with_history(predictor)
    predictor._train_models()
    predictor.error_metrics = {
        40: {
            "errors": [0.1],
            "abs_errors": [0.1],
            "pct_errors": [10.0],
            "predictions": [1.1],
            "actuals": [1.0],
            "count": 1,
        }
    }

    metrics = predictor.get_learning_metrics()

    assert metrics["status"] == "learning"
    assert metrics["mae"] == pytest.approx(0.1)
    assert metrics["holdout_mae"] == pytest.approx(predictor.holdout_mae)


def test_a_failed_training_clears_the_holdout_metrics(
    make_predictor: Callable[[], SpotPricePredictor],
) -> None:
    predictor = make_predictor()
    _with_history(predictor)
    predictor._train_models()
    assert predictor.holdout_mae is not None

    with patch.object(predictor.price_model, "fit", side_effect=RuntimeError("no fit")):
        predictor._train_models()

    assert predictor.is_trained is False
    assert predictor.holdout_metrics() == dict.fromkeys(HOLDOUT_META_KEYS)
    assert not set(HOLDOUT_META_KEYS) & set(predictor.storage.load_meta_dict())


def test_a_storage_error_does_not_fail_the_training(
    make_predictor: Callable[[], SpotPricePredictor],
    caplog: pytest.LogCaptureFixture,
) -> None:
    predictor = make_predictor()
    _with_history(predictor)

    with patch.object(
        predictor.storage, "save_meta_dict", side_effect=OSError("disk full")
    ):
        predictor._train_models()

    assert predictor.is_trained
    assert predictor.holdout_mae is not None
    assert "Could not store the holdout metrics: disk full" in caplog.text


@pytest.mark.parametrize(
    "stored",
    [
        {},
        {"holdout_mae": "0.1", "holdout_rmse": "0.2"},
        {"holdout_mae": "x", "holdout_rmse": "0.2", "holdout_trained_at": "now"},
        {"holdout_mae": "nan", "holdout_rmse": "0.2", "holdout_trained_at": "now"},
        {
            "holdout_mae": "0.1",
            "holdout_rmse": "0.2",
            "holdout_trained_at": "not a time",
        },
    ],
    ids=["absent", "no-time", "bad-number", "nan", "bad-time"],
)
def test_incomplete_or_invalid_stored_metrics_restore_as_none(
    make_predictor: Callable[[], SpotPricePredictor], stored: dict[str, str]
) -> None:
    predictor = make_predictor()

    predictor._restore_holdout_metrics(stored)

    assert predictor.holdout_metrics() == dict.fromkeys(HOLDOUT_META_KEYS)


def test_restored_time_is_utc(
    make_predictor: Callable[[], SpotPricePredictor],
) -> None:
    predictor = make_predictor()

    predictor._restore_holdout_metrics(
        {
            "holdout_mae": "0.1",
            "holdout_rmse": "0.2",
            "holdout_trained_at": "2026-09-24T10:00:00+02:00",
        }
    )

    assert predictor.holdout_metrics() == {
        "holdout_mae": pytest.approx(0.1),
        "holdout_rmse": pytest.approx(0.2),
        "holdout_trained_at": "2026-09-24T08:00:00+00:00",
    }


def test_delete_meta_keys_leaves_other_keys(
    make_predictor: Callable[[], SpotPricePredictor],
) -> None:
    storage = make_predictor().storage
    storage.save_meta_dict({"holdout_mae": 0.1, "hpo_counter": 3})

    storage.delete_meta_keys(("holdout_mae", "missing"))

    meta = storage.load_meta_dict()
    assert "holdout_mae" not in meta
    assert meta["hpo_counter"] == "3"
