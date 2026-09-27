"""Tests for the two-stage cross-border price model (#29)."""

import logging
import math
from collections.abc import Callable, Iterator
from datetime import date, datetime, timedelta
from itertools import pairwise
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from custom_components.open_spot_forecast.const import NEIGHBOURS
from custom_components.open_spot_forecast.ml.cross_border import (
    MIN_STAGE1_ROWS,
    STAGE1_FOLDS,
    CrossBorderModels,
    Stage1Model,
    cross_price_name,
    local_day_fold,
    neighbour_feature_rows,
)
from custom_components.open_spot_forecast.ml.features import FEATURE_NAMES
from custom_components.open_spot_forecast.ml.predictor import SpotPricePredictor
from custom_components.open_spot_forecast.ml.series_storage import (
    OPENMETEO_WEATHER,
    neighbour_prices,
)
from custom_components.open_spot_forecast.ml.storage import LearningStorage
from custom_components.open_spot_forecast.ml.zone_weather import (
    ZoneWeatherIndex,
    zone_points,
)
from custom_components.open_spot_forecast.time_slots import slot_start_in_day

TZ = ZoneInfo("Europe/Copenhagen")
FIRST_DAY = date(2026, 9, 7)
DAYS = 16
SLOTS = 96


def _starts(first: date = FIRST_DAY, days: int = DAYS) -> list[datetime]:
    return [
        slot_start_in_day(first + timedelta(days=day), slot, TZ)
        for day in range(days)
        for slot in range(SLOTS)
    ]


def _wind(start: datetime) -> float:
    """A synthetic 80 m wind that changes by the hour and the day."""
    return 8 + 6 * math.sin(start.timestamp() / 7200) + start.day % 5


def _price(start: datetime) -> float:
    """A neighbour price driven by its wind and the time of day."""
    return 120 - 8 * _wind(start) + 20 * math.sin(2 * math.pi * start.hour / 24)


def _weather_rows(zone: str, starts: list[datetime]) -> list[dict[str, Any]]:
    return [
        {
            "timestamp": start.isoformat(),
            "point": point,
            "wind_80m": _wind(start),
            "temperature": 12.0,
            "irradiance": 100.0,
            "pressure": 1013.0,
            "humidity": 80.0,
        }
        for start in starts
        for point in zone_points(zone)
    ]


def _store_neighbour(
    storage: LearningStorage, zone: str, starts: list[datetime]
) -> None:
    storage.upsert_series(
        neighbour_prices(zone),
        [
            {"timestamp": start.isoformat(), "zone": zone, "price": _price(start)}
            for start in starts
        ],
    )
    storage.upsert_series(OPENMETEO_WEATHER, _weather_rows(zone, starts))


@pytest.fixture
def storage(tmp_path: Path) -> Iterator[LearningStorage]:
    hass = Mock()
    hass.config.path.return_value = str(tmp_path / ".storage")
    store = LearningStorage(hass, "DK1")
    yield store
    store.close()


# --- Names and folds -------------------------------------------------------------


def test_each_neighbour_has_a_named_column() -> None:
    assert cross_price_name("DE") == "cross_price_DE"
    assert NEIGHBOURS["DK1"] == ("DE", "NL", "NO2", "SE3", "DK2")
    assert NEIGHBOURS["DK2"] == ("DK1", "DE", "SE4")


def test_every_neighbour_is_a_region_with_weather_points() -> None:
    from custom_components.open_spot_forecast.const import REGIONS, WEATHER_POINTS

    for region, zones in NEIGHBOURS.items():
        assert region in REGIONS
        assert region not in zones
        assert all(zone in REGIONS and zone in WEATHER_POINTS for zone in zones)


def test_folds_are_whole_local_days_taking_turns() -> None:
    starts = _starts(days=4)
    folds = [local_day_fold(start) for start in starts]

    days = [set(folds[day * SLOTS : (day + 1) * SLOTS]) for day in range(4)]
    assert all(len(day) == 1 for day in days)
    assert all(a != b for a, b in pairwise(days))


def test_neighbour_rows_are_the_canonical_vector_in_the_zones_time() -> None:
    starts = _starts(days=1)
    weather = ZoneWeatherIndex(_weather_rows("DE", starts), zone_points("DE"))

    rows = neighbour_feature_rows("DE", starts, weather)

    assert rows.shape == (SLOTS, len(FEATURE_NAMES))
    wind = FEATURE_NAMES.index("zone_wind")
    assert rows[0, wind] == pytest.approx(_wind(starts[0]))
    # The region's own inputs are unknown for a neighbour
    assert np.isnan(rows[:, FEATURE_NAMES.index("consumption_forecast")]).all()
    # Without weather only the calendar and the sun are known
    assert np.isnan(neighbour_feature_rows("DE", starts, None)[:, wind]).all()


# --- Stage 1 -------------------------------------------------------------------


def _stage1_data(
    days: int = DAYS,
) -> tuple[list[datetime], np.ndarray, np.ndarray, np.ndarray]:
    starts = _starts(days=days)
    weather = ZoneWeatherIndex(_weather_rows("DE", starts), zone_points("DE"))
    rows = neighbour_feature_rows("DE", starts, weather)
    prices = np.array([_price(start) for start in starts])
    folds = np.array([local_day_fold(start) for start in starts])
    return starts, rows, prices, folds


def test_a_training_rows_value_never_depends_on_its_own_days_prices() -> None:
    """Out of sample: poisoning a fold's prices leaves that fold's values alone."""
    _, rows, prices, folds = _stage1_data()
    clean = Stage1Model()
    clean.fit(rows, prices, folds)
    poisoned = Stage1Model()
    poisoned.fit(rows, np.where(folds == 0, 1e6, prices), folds)

    in_fold = folds == 0
    assert np.allclose(
        clean.predict_out_of_sample(rows[in_fold], folds[in_fold]),
        poisoned.predict_out_of_sample(rows[in_fold], folds[in_fold]),
    )
    # The other fold's model did see them
    assert not np.allclose(
        clean.predict_out_of_sample(rows[~in_fold], folds[~in_fold]),
        poisoned.predict_out_of_sample(rows[~in_fold], folds[~in_fold]),
    )


def test_out_of_sample_values_are_further_off_than_fitted_ones() -> None:
    """What stage 2 trains on is as uncertain as a forecast, not a fit."""
    _, rows, prices, folds = _stage1_data()
    model = Stage1Model()
    model.fit(rows, prices, folds)

    out_of_sample = np.mean(np.abs(model.predict_out_of_sample(rows, folds) - prices))
    fitted = np.mean(np.abs(model.predict(rows) - prices))

    assert out_of_sample > fitted


def test_forecasts_are_the_mean_of_the_fold_models() -> None:
    _, rows, prices, folds = _stage1_data()
    model = Stage1Model()
    model.fit(rows, prices, folds)

    expected = np.mean(
        [fold_model.predict(rows[:10]) for fold_model in model.models if fold_model],
        axis=0,
    )
    assert model.predict(rows[:10]) == pytest.approx(expected)
    assert all(fold_model is not None for fold_model in model.models)


def test_too_little_history_leaves_stage_1_unfitted() -> None:
    days = (MIN_STAGE1_ROWS // SLOTS) * STAGE1_FOLDS - STAGE1_FOLDS
    _, rows, prices, folds = _stage1_data(days)
    model = Stage1Model()
    model.fit(rows, prices, folds)

    assert model.is_fitted is False
    assert np.isnan(model.predict(rows)).all()
    assert np.isnan(model.predict_out_of_sample(rows, folds)).all()


def test_missing_prices_are_not_training_rows() -> None:
    _, rows, prices, folds = _stage1_data()
    gappy = prices.copy()
    gappy[::20] = np.nan
    model = Stage1Model()
    model.fit(rows, gappy, folds)

    assert all(fold_model is not None for fold_model in model.models)
    assert np.isfinite(model.predict(rows)).all()


# --- Stored neighbour history --------------------------------------------------


def test_stage_1_is_fitted_from_the_stored_neighbour_history(
    storage: LearningStorage,
) -> None:
    starts = _starts()
    _store_neighbour(storage, "DE", starts)
    models = CrossBorderModels("DK1", TZ, storage)

    columns = models.fit(starts)

    assert columns.shape == (len(starts), len(NEIGHBOURS["DK1"]))
    de = models.zones.index("DE")
    assert np.isfinite(columns[:, de]).all()
    # Neighbours without stored history degrade to NaN
    others = [index for index in range(len(models.zones)) if index != de]
    assert np.isnan(columns[:, others]).all()
    # The column tracks the neighbour's price
    actual = np.array([_price(start) for start in starts])
    assert np.corrcoef(columns[:, de], actual)[0, 1] > 0.8


def test_forecasts_come_from_the_stored_weather_forecast(
    storage: LearningStorage,
) -> None:
    starts = _starts()
    ahead = _starts(FIRST_DAY + timedelta(days=DAYS), 2)
    _store_neighbour(storage, "DE", starts)
    storage.upsert_series(OPENMETEO_WEATHER, _weather_rows("DE", ahead))
    models = CrossBorderModels("DK1", TZ, storage)
    models.fit(starts)

    columns = models.predict(ahead)

    de = models.zones.index("DE")
    actual = np.array([_price(start) for start in ahead])
    assert np.mean(np.abs(columns[:, de] - actual)) < 10.0
    assert np.isnan(np.delete(columns, de, axis=1)).all()


def test_unfitted_models_and_empty_requests_give_nan(storage: LearningStorage) -> None:
    models = CrossBorderModels("DK1", TZ, storage)

    assert np.isnan(models.predict(_starts(days=1))).all()
    assert models.fit([]).shape == (0, 5)
    assert models.predict([]).shape == (0, 5)


def test_a_failing_neighbour_degrades_to_nan(
    storage: LearningStorage, caplog: pytest.LogCaptureFixture
) -> None:
    starts = _starts()
    _store_neighbour(storage, "DE", starts)
    models = CrossBorderModels("DK1", TZ, storage)
    models.fit(starts)
    load_series = storage.load_series

    def failing(spec: Any, *args: Any) -> Any:
        if spec.table == "openmeteo_weather":
            raise RuntimeError("disk I/O error")
        return load_series(spec, *args)

    with (
        patch.object(storage, "load_series", side_effect=failing),
        caplog.at_level(logging.WARNING),
    ):
        forecast = models.predict(starts[:96])
        refit = models.fit(starts)

    assert np.isnan(forecast).all()
    assert np.isnan(refit).all()
    assert "Stage-1 price forecast for DE failed" in caplog.text
    assert "Stage-1 price model for DE failed" in caplog.text
    # The failed refit leaves DE unfitted
    assert np.isnan(models.predict(starts[:96])).all()


# --- Stage 2: the predictor ------------------------------------------------------


async def _run_inline(func: Callable[..., Any], *args: Any) -> Any:
    return func(*args)


def _predictor(
    tmp_path: Path, region: str = "DK1", **kwargs: Any
) -> SpotPricePredictor:
    tmp_path.mkdir(exist_ok=True)
    hass = Mock()
    hass.config.path.return_value = str(tmp_path / ".storage")
    hass.async_add_executor_job = _run_inline
    return SpotPricePredictor(hass, region, **kwargs)


def test_the_option_turns_the_cross_border_model_on(tmp_path: Path) -> None:
    on = _predictor(tmp_path, cross_border=True)
    off = _predictor(tmp_path / "off")
    no_neighbours = _predictor(tmp_path / "se3", "SE3", cross_border=True)
    try:
        assert on.cross_border is not None
        assert on.cross_border.zones == NEIGHBOURS["DK1"]
        assert off.cross_border is None
        assert no_neighbours.cross_border is None
    finally:
        for predictor in (on, off, no_neighbours):
            predictor.storage.close()


@pytest.fixture
def two_stage(tmp_path: Path) -> Iterator[SpotPricePredictor]:
    """A DK1 predictor with the cross-border model and 16 days of prices."""
    predictor = _predictor(tmp_path, cross_border=True, training_days=30)
    predictor.price_history = [
        {
            "date": (FIRST_DAY + timedelta(days=day)).isoformat(),
            "prices": [
                0.5 + 0.1 * math.sin(2 * math.pi * slot / SLOTS)
                for slot in range(SLOTS)
            ],
        }
        for day in range(DAYS)
    ]
    yield predictor
    predictor.storage.close()


def test_stage_2_trains_on_one_column_per_neighbour(
    two_stage: SpotPricePredictor,
) -> None:
    _store_neighbour(two_stage.storage, "DE", _starts())
    fit = Mock(wraps=two_stage.price_model.fit)
    with patch.object(two_stage.price_model, "fit", fit):
        two_stage._train_models()

    assert two_stage.is_trained is True
    X, _ = fit.call_args.args
    assert X.shape == (DAYS * SLOTS, len(FEATURE_NAMES) + 5)
    assert np.isfinite(X[:, len(FEATURE_NAMES)]).all()  # cross_price_DE
    assert np.isnan(X[:, len(FEATURE_NAMES) + 1 :]).all()


def test_predictions_work_without_any_neighbour_data(
    two_stage: SpotPricePredictor,
) -> None:
    """Every stage-1 column is NaN until the neighbours are backfilled."""
    two_stage.predict({}, [0.5] * SLOTS, forecast_days=1)

    assert two_stage.is_trained is True
    assert len(two_stage.predictions) == SLOTS
    assert all(math.isfinite(p["price"]) for p in two_stage.predictions)


def test_predictions_use_the_stage_1_forecasts(two_stage: SpotPricePredictor) -> None:
    _store_neighbour(two_stage.storage, "DE", _starts())
    two_stage._train_models()
    features = two_stage._combine_features(two_stage._generate_time_features(1, 15), {})

    rows = two_stage._model_inputs(features)

    assert rows.shape == (len(features), len(FEATURE_NAMES) + 5)
    assert np.isfinite(rows[:, len(FEATURE_NAMES)]).all()
    two_stage._generate_predictions(features, 1, 15)
    assert len(two_stage.predictions) == len(features)


def test_new_neighbour_data_triggers_a_retrain(two_stage: SpotPricePredictor) -> None:
    """Stage 2 retrains when any stage-1 input changes (#31)."""
    two_stage.retrain()
    assert two_stage.needs_retraining() is False

    _store_neighbour(two_stage.storage, "NL", _starts(days=1))

    assert two_stage.needs_retraining() is True


def test_training_logs_its_duration(
    two_stage: SpotPricePredictor, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO):
        two_stage._train_models()

    assert "Stage-1 price models of DE, NL, NO2, SE3, DK2 fitted in" in caplog.text
    assert any(
        record.getMessage().startswith("ML model trained in")
        for record in caplog.records
    )
