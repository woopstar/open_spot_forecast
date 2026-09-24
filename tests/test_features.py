"""Invariant tests for the canonical feature vector and time features (issue #34)."""

import math
import re
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from custom_components.open_spot_forecast.ml import models as models_module
from custom_components.open_spot_forecast.ml.features import (
    FEATURE_NAMES,
    build_feature_vector,
    slot_time_features,
)
from custom_components.open_spot_forecast.ml.models import create_price_model
from custom_components.open_spot_forecast.ml.predictor import SpotPricePredictor

TZ = ZoneInfo("Europe/Copenhagen")
ML_DOC = Path(__file__).parents[1] / "docs" / "ml_documentation.md"


def _predictor(tmp_path: Path) -> SpotPricePredictor:
    """Return a predictor backed by a throwaway SQLite store."""
    hass = Mock()
    hass.config.path.return_value = str(tmp_path / ".storage")
    return SpotPricePredictor(hass, "DK1")


class _SpyModel:
    """Stands in for the price model and records the rows it receives."""

    def __init__(self) -> None:
        # Hyperparameters _train_models copies into its holdout model
        self.n_estimators = 10
        self.learning_rate = 0.1
        self.random_state = 42
        self.trees: list[object] = []
        self.fit_rows = np.empty((0, 0))
        self.predict_rows: list[np.ndarray] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.fit_rows = np.asarray(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.predict_rows.append(np.asarray(X))
        return np.zeros(len(X))


@pytest.fixture
def built_rows(monkeypatch: pytest.MonkeyPatch) -> list[list[float]]:
    """Record every row ModelMixin builds through build_feature_vector."""
    rows: list[list[float]] = []

    def recording_builder(feature: dict) -> list[float]:
        row = build_feature_vector(feature)
        rows.append(row)
        return row

    monkeypatch.setattr(models_module, "build_feature_vector", recording_builder)
    return rows


def test_feature_vector_has_20_unique_features():
    """The model input stays at the 20 canonical features."""
    assert len(FEATURE_NAMES) == 20
    assert len(set(FEATURE_NAMES)) == 20


def test_feature_names_match_the_documented_vector():
    """Column order matches the table in docs/ml_documentation.md."""
    documented = re.findall(
        r"^\|\s*(\d+)\s*\|\s*`([a-z_]+)`", ML_DOC.read_text(encoding="utf-8"), re.M
    )

    assert [int(index) for index, _ in documented] == list(range(20))
    assert [name for _, name in documented] == list(FEATURE_NAMES)


def test_build_feature_vector_follows_feature_names_order():
    """Each feature lands in its documented column."""
    feature = {name: index for index, name in enumerate(FEATURE_NAMES)}

    assert build_feature_vector(feature) == pytest.approx(list(range(20)))


def test_build_feature_vector_defaults():
    """Missing features default to 0, except humidity 50 % and temperature 15 °C."""
    vector = dict(zip(FEATURE_NAMES, build_feature_vector({}), strict=True))

    assert vector.pop("humidity") == pytest.approx(50.0)
    assert vector.pop("temperature") == pytest.approx(15.0)
    assert list(vector.values()) == pytest.approx([0.0] * 18)


def test_build_feature_vector_sanitizes_values():
    """None and non-numeric values become 0.0; numeric strings are parsed."""
    vector = dict(
        zip(
            FEATURE_NAMES,
            build_feature_vector(
                {
                    "hour": None,
                    "humidity": "n/a",
                    "temperature": "21.5",
                    "is_weekend": True,
                    "wind_share": np.float64(0.25),
                    "price_mean": [1.0],
                }
            ),
            strict=True,
        )
    )

    assert vector["hour"] == pytest.approx(0.0)
    assert vector["humidity"] == pytest.approx(0.0)
    assert vector["temperature"] == pytest.approx(21.5)
    assert vector["is_weekend"] == pytest.approx(1.0)
    assert vector["wind_share"] == pytest.approx(0.25)
    assert vector["price_mean"] == pytest.approx(0.0)
    assert all(isinstance(value, float) for value in vector.values())


def test_slot_time_features_fields():
    """Time features describe the local slot start."""
    feature = slot_time_features(datetime(2026, 6, 6, 18, 45, tzinfo=TZ))  # Saturday

    assert feature["start"] == "2026-06-06T18:45:00+02:00"
    assert feature["end"] == "2026-06-06T19:00:00+02:00"
    assert (feature["hour"], feature["minute"]) == (18, 45)
    assert (feature["day_of_week"], feature["is_weekend"], feature["month"]) == (
        5,
        1,
        6,
    )
    assert feature["hour_sin"] == pytest.approx(math.sin(2 * math.pi * 18 / 24))
    assert feature["hour_cos"] == pytest.approx(math.cos(2 * math.pi * 18 / 24))
    assert feature["dow_sin"] == pytest.approx(math.sin(2 * math.pi * 5 / 7))


def test_slot_time_features_end_spans_one_interval_across_dst():
    """The slot ending as clocks fall back still lasts 15 real minutes."""
    feature = slot_time_features(datetime(2025, 10, 26, 2, 45, tzinfo=TZ))

    start = datetime.fromisoformat(feature["start"])
    end = datetime.fromisoformat(feature["end"])
    assert feature["end"] == "2025-10-26T02:00:00+01:00"
    assert end - start == timedelta(minutes=15)


def test_training_and_prediction_share_time_features(tmp_path: Path) -> None:
    """get_all_historical_prices uses the same time features as prediction."""
    predictor = _predictor(tmp_path)
    predictor.price_history = [{"date": "2026-06-01", "prices": [1.0] * 96}]

    _, features = predictor.get_all_historical_prices()

    assert len(features) == 96
    assert features[37] == slot_time_features(datetime(2026, 6, 1, 9, 15, tzinfo=TZ))


def test_training_rows_come_from_build_feature_vector(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, built_rows: list[list[float]]
) -> None:
    """_train_models fits the live model on all build_feature_vector rows."""
    predictor = _predictor(tmp_path)
    predictor.price_history = [
        {"date": day, "prices": [float(i % 7) for i in range(96)]}
        for day in ("2026-06-01", "2026-06-02")
    ]
    spy = _SpyModel()
    monkeypatch.setattr(predictor, "price_model", spy)
    monkeypatch.setattr(predictor, "store_daily_prices", lambda *args: None)

    predictor._train_models([], [{"temperature": 7.5}])

    assert len(built_rows) == 192
    np.testing.assert_array_equal(spy.fit_rows, np.array(built_rows))
    assert spy.fit_rows.shape[1] == len(FEATURE_NAMES)


def test_prediction_rows_come_from_build_feature_vector(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, built_rows: list[list[float]]
) -> None:
    """_generate_predictions predicts on build_feature_vector rows."""
    predictor = _predictor(tmp_path)
    spy = _SpyModel()
    monkeypatch.setattr(predictor, "price_model", spy)
    features = [
        slot_time_features(datetime(2026, 6, 1, 12, 15 * i, tzinfo=TZ))
        | {"price_mean": 0.9, "wind_speed_mean": 6.0}
        for i in range(4)
    ]

    predictor._generate_predictions(features, 1, 15)

    np.testing.assert_array_equal(np.vstack(spy.predict_rows), np.array(built_rows))
    assert len(predictor.predictions) == 4


def test_create_price_model_uses_documented_hyperparameters(tmp_path: Path) -> None:
    """The production price model is a fresh 200-tree GBM, lr 0.1, seed 42."""
    model = create_price_model()

    assert (model.n_estimators, model.learning_rate) == (200, pytest.approx(0.1))
    assert model.random_state == 42
    assert model.trees == []
    assert _predictor(tmp_path).price_model.n_estimators == 200
