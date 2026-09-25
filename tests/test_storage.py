"""LearningStorage split into per-table mixins (issue #53).

``LearningStorage`` keeps the connection, lock, schema and migrations; the
table operations live in mixins. These tests pin the public API and round-trip
every table through a real SQLite database.
"""

import json
from collections.abc import Callable, Iterator
from datetime import timedelta
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest

from homeassistant.util import dt as dt_util

from custom_components.open_spot_forecast.ml.accuracy_storage import (
    LeadTimeAccuracyStorageMixin,
)
from custom_components.open_spot_forecast.ml.history_storage import (
    HistoryStorageMixin,
)
from custom_components.open_spot_forecast.ml.prediction_storage import (
    PredictionStorageMixin,
)
from custom_components.open_spot_forecast.ml.state_storage import (
    LearningStateStorageMixin,
)
from custom_components.open_spot_forecast.ml.storage import LearningStorage

ML_DIR = Path(__file__).parent.parent / "custom_components/open_spot_forecast/ml"
MAX_FILE_BYTES = 30 * 1024
MAX_FILE_LINES = 1000


def _hass(tmp_path: Path) -> Mock:
    """Return a mock Home Assistant whose storage lives under tmp_path."""

    async def run_inline(func: Callable[..., Any], *args: Any) -> Any:
        return func(*args)

    hass = Mock()
    hass.config.path.return_value = str(tmp_path / ".storage")
    hass.async_add_executor_job = run_inline
    return hass


@pytest.fixture
def storage(tmp_path: Path) -> Iterator[LearningStorage]:
    """Return a LearningStorage on a fresh database, closed after the test."""
    store = LearningStorage(_hass(tmp_path), "DK1")
    yield store
    store.close()


# --- Module layout -------------------------------------------------------------------


@pytest.mark.parametrize("path", sorted(ML_DIR.glob("*.py")), ids=lambda p: p.name)
def test_ml_modules_stay_below_the_file_size_limit(path: Path) -> None:
    """Every ML module stays below 30 KB and 1000 lines."""
    assert path.stat().st_size < MAX_FILE_BYTES
    assert len(path.read_text(encoding="utf-8").splitlines()) < MAX_FILE_LINES


@pytest.mark.parametrize(
    ("mixin", "methods"),
    [
        (
            PredictionStorageMixin,
            (
                "insert_prediction",
                "delete_old_predictions",
                "find_predictions_for_timestamp",
                "get_pending_prediction_dates",
                "remove_prediction",
                "count_predictions",
            ),
        ),
        (
            HistoryStorageMixin,
            (
                "insert_weather_snapshot",
                "find_weather_for_timestamp",
                "delete_old_weather",
                "count_weather_snapshots",
                "load_weather_history",
                "insert_nordpool_prognosis",
                "insert_nordpool_prognoses_batch",
                "find_nordpool_for_timestamp",
                "delete_old_nordpool",
                "load_nordpool_history",
                "save_price_history",
                "load_price_history",
            ),
        ),
        (
            LearningStateStorageMixin,
            (
                "save_volatility",
                "load_volatility",
                "save_error_metrics",
                "load_error_metrics",
                "save_bias_correction",
                "load_bias_correction",
                "save_meta",
                "save_meta_dict",
                "load_meta_dict",
                "load_meta",
                "save_all",
                "async_save_all",
                "load_all",
                "async_load_all",
            ),
        ),
        (
            LeadTimeAccuracyStorageMixin,
            (
                "add_lead_time_errors",
                "get_lead_time_error_sums",
                "delete_lead_time_accuracy_before",
            ),
        ),
    ],
    ids=lambda value: value.__name__ if isinstance(value, type) else "",
)
def test_learning_storage_exposes_every_mixin_method(
    mixin: type, methods: tuple[str, ...]
) -> None:
    """The public API is unchanged: every method is reachable on LearningStorage."""
    assert issubclass(LearningStorage, mixin)
    for name in methods:
        assert getattr(LearningStorage, name) is getattr(mixin, name)


def test_connection_and_maintenance_stay_on_learning_storage() -> None:
    for name in ("checkpoint", "close", "clear_all", "async_clear_all"):
        assert name in vars(LearningStorage)
    assert "async_clear_storage" in vars(LearningStorage)


# --- Predictions ---------------------------------------------------------------------


def test_prediction_round_trip(storage: LearningStorage) -> None:
    stored_at = dt_util.now().isoformat()
    first = storage.insert_prediction(
        "2026-09-24T10:15:00+02:00", 0.42, 0.8, 10, 15, stored_at, 14.0, 5.5, 60.0
    )
    storage.insert_prediction("2026-09-25T00:00:00+02:00", 0.3, 0.7, 0, 0, stored_at)

    found = storage.find_predictions_for_timestamp("2026-09-24", 10, 15)
    assert len(found) == 1
    row = found[0]
    assert row["id"] == first
    assert row["price"] == pytest.approx(0.42)
    assert row["confidence"] == pytest.approx(0.8)
    assert (row["forecast_temp"], row["forecast_wind"], row["forecast_cloud"]) == (
        pytest.approx(14.0),
        pytest.approx(5.5),
        pytest.approx(60.0),
    )
    assert storage.find_predictions_for_timestamp("2026-09-24", 10, 30) == []
    assert storage.get_pending_prediction_dates() == ["2026-09-24", "2026-09-25"]
    assert storage.count_predictions() == 2

    assert first is not None
    storage.remove_prediction(first)
    assert storage.count_predictions() == 1


def test_delete_old_predictions_uses_stored_at(storage: LearningStorage) -> None:
    old = (dt_util.now() - timedelta(days=10)).isoformat()
    new = dt_util.now().isoformat()
    storage.insert_prediction("2026-09-14T10:00:00+02:00", 0.4, 0.8, 10, 0, old)
    storage.insert_prediction("2026-09-24T10:00:00+02:00", 0.4, 0.8, 10, 0, new)

    assert storage.delete_old_predictions(7) == 1
    assert storage.count_predictions() == 1


# --- Weather, Nordpool and price history ---------------------------------------------


def test_weather_snapshot_round_trip_moves_last_data_write(
    storage: LearningStorage,
) -> None:
    assert storage.last_data_write is None
    storage.insert_weather_snapshot(
        "2026-09-24T08:00:00Z", 12.5, 6.0, 270.0, 80.0, 90.0, 1500.0
    )

    assert storage.last_data_write is not None
    assert storage.count_weather_snapshots() == 1
    assert storage.load_weather_history() == [
        {
            "timestamp": "2026-09-24T08:00:00Z",
            "temperature": pytest.approx(12.5),
            "wind_speed": pytest.approx(6.0),
            "wind_direction": pytest.approx(270.0),
            "cloud_coverage": pytest.approx(80.0),
            "humidity": pytest.approx(90.0),
        }
    ]


def test_find_weather_and_nordpool_return_none_without_a_match(
    storage: LearningStorage,
) -> None:
    assert storage.find_weather_for_timestamp("2026-09-24T10:00:00Z") is None
    assert storage.find_nordpool_for_timestamp("2026-09-24T10:00:00Z") is None


def test_find_weather_and_nordpool_return_the_stored_values(
    storage: LearningStorage,
) -> None:
    # SQLite's datetime() returns "YYYY-MM-DD HH:MM:SS"; rows stored in that
    # form are matched by the unchanged lookup SQL.
    storage.insert_weather_snapshot(
        "2026-09-24 10:00:00", 12.5, 6.0, 270.0, 80.0, 90.0, None
    )
    storage.insert_nordpool_prognosis("2026-09-24 10:00:00", 4000.0, 10.0, 1.0, 2.0)

    weather = storage.find_weather_for_timestamp("2026-09-24 10:00:00")
    assert weather is not None
    assert weather["temperature"] == pytest.approx(12.5)
    assert weather["solar_power"] is None
    nordpool = storage.find_nordpool_for_timestamp("2026-09-24 10:00:00")
    assert nordpool == {
        "consumption": pytest.approx(4000.0),
        "solar": pytest.approx(10.0),
        "wind_offshore": pytest.approx(1.0),
        "wind_onshore": pytest.approx(2.0),
    }


def test_nordpool_batch_moves_last_data_write_only_on_change(
    storage: LearningStorage,
) -> None:
    entry = {
        "timestamp": "2026-09-24T08:00:00Z",
        "consumption": 4000.0,
        "solar": 10.0,
        "wind_offshore": 1.0,
        "wind_onshore": 2.0,
    }
    storage.insert_nordpool_prognoses_batch([entry])
    first_write = storage.last_data_write
    assert first_write is not None

    storage.last_data_write = None
    storage.insert_nordpool_prognoses_batch([entry])
    assert storage.last_data_write is None

    storage.insert_nordpool_prognoses_batch([{**entry, "solar": 20.0}])
    assert storage.last_data_write is not None
    rows = storage.load_nordpool_history()
    assert len(rows) == 1
    assert rows[0]["solar"] == pytest.approx(20.0)


def test_old_weather_and_nordpool_rows_are_pruned(storage: LearningStorage) -> None:
    old = (dt_util.now() - timedelta(days=40)).isoformat()
    new = dt_util.now().isoformat()
    for timestamp in (old, new):
        storage.insert_weather_snapshot(timestamp, 10.0, 5.0, None, None, None, None)
        storage.insert_nordpool_prognosis(timestamp, 4000.0, None, None, None)

    assert storage.delete_old_weather(30) == 1
    assert storage.delete_old_nordpool(30) == 1
    assert storage.count_weather_snapshots() == 1
    assert len(storage.load_nordpool_history()) == 1


def test_price_history_round_trip(storage: LearningStorage) -> None:
    storage.save_price_history(
        [
            {"date": "2026-09-25", "prices": [0.5, None, 0.7]},
            {"date": "2026-09-24", "prices": [0.4]},
            {"prices": [0.1]},
        ]
    )

    assert storage.load_price_history() == [
        {"date": "2026-09-24", "prices": [0.4]},
        {"date": "2026-09-25", "prices": [0.5, None, 0.7]},
    ]


# --- Learned state -------------------------------------------------------------------


def test_error_metrics_bias_and_volatility_round_trip(
    storage: LearningStorage,
) -> None:
    metrics = {
        40: {
            "errors": [0.1],
            "abs_errors": [0.1],
            "pct_errors": [10.0],
            "predictions": [1.1],
            "actuals": [1.0],
            "count": 1,
        }
    }
    storage.save_error_metrics(metrics)
    storage.save_bias_correction({40: -0.05})
    storage.save_volatility({40: 0.2})

    assert storage.load_error_metrics() == metrics
    assert storage.load_bias_correction() == {40: pytest.approx(-0.05)}
    assert storage.load_volatility() == {40: pytest.approx(0.2)}


def test_meta_round_trip(storage: LearningStorage) -> None:
    storage.save_meta(120, True)
    storage.save_meta_dict({"hpo_counter": 3, "hpo_best_mae": 0.12})

    meta = storage.load_meta()
    assert meta["training_samples"] == 120
    assert meta["is_trained"] is True
    assert meta["hpo_counter"] == "3"
    raw = storage.load_meta_dict()
    assert raw["is_trained"] == "1"
    assert raw["hpo_best_mae"] == "0.12"


@pytest.mark.asyncio
async def test_save_all_and_load_all_round_trip(storage: LearningStorage) -> None:
    saved = await storage.async_save_all(
        {
            "error_metrics": {5: {"errors": [0.2], "count": 1}},
            "bias_correction": {5: 0.02},
            "price_history": [{"date": "2026-09-24", "prices": [0.4, None]}],
            "prediction_history": [
                {"start": "2026-09-24T10:00:00+02:00", "price": 0.4, "hour": 10},
                {"price": 0.1},
            ],
            "volatility_mae": {5: 0.3},
            "solar_scale": 0.9,
            "solar_scale_samples": 12,
            "training_samples": 50,
            "is_trained": True,
        }
    )
    assert saved is True
    storage.save_meta_dict({"hpo_best_mae": 0.1})

    data = await storage.async_load_all()
    assert data is not None
    assert data["error_metrics"] == {5: {"errors": [pytest.approx(0.2)], "count": 1}}
    assert data["bias_correction"] == {5: pytest.approx(0.02)}
    assert data["price_history"] == [{"date": "2026-09-24", "prices": [0.4, None]}]
    assert data["prediction_count"] == 1
    assert data["volatility_mae"] == {5: pytest.approx(0.3)}
    assert data["solar_scale"] == pytest.approx(0.9)
    assert data["solar_scale_samples"] == 12
    assert data["training_samples"] == 50
    assert data["is_trained"] is True
    assert data["hpo_best_mae"] == "0.1"
    assert "schema_version" not in data


@pytest.mark.asyncio
async def test_failed_save_all_rolls_back(storage: LearningStorage) -> None:
    saved = await storage.async_save_all(
        {"bias_correction": {1: 0.1, 2: "not a number"}}
    )

    assert saved is False
    assert storage.load_bias_correction() == {}


@pytest.mark.asyncio
async def test_clear_storage_empties_every_table(storage: LearningStorage) -> None:
    storage.save_all({"bias_correction": {1: 0.1}, "training_samples": 5})
    storage.insert_prediction(
        "2026-09-24T10:00:00+02:00", 0.4, 0.8, 10, 0, dt_util.now().isoformat()
    )

    assert await storage.async_clear_storage() is True
    assert storage.count_predictions() == 0
    assert storage.load_bias_correction() == {}
    assert "training_samples" not in storage.load_meta_dict()


def test_legacy_json_training_state_is_imported(tmp_path: Path) -> None:
    storage_dir = tmp_path / ".storage"
    storage_dir.mkdir()
    json_path = storage_dir / "open_spot_forecast_DK1_learning.json"
    json_path.write_text(
        json.dumps({"training_samples": 7, "is_trained": True}), encoding="utf-8"
    )

    storage = LearningStorage(_hass(tmp_path), "DK1")
    try:
        meta = storage.load_meta()
        assert meta["training_samples"] == 7
        assert meta["is_trained"] is True
        assert not json_path.exists()
        assert json_path.with_suffix(".json.bak").exists()
    finally:
        storage.close()
