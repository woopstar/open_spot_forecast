"""Tests for retraining the price model when its inputs change (issue #31)."""

from collections.abc import Callable, Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import numpy as np
import pytest

from custom_components.open_spot_forecast import async_setup_entry
from custom_components.open_spot_forecast.const import (
    CONF_ENABLE_ML_PREDICTION,
    CONF_REGION,
    CONF_STROMLIGNING_SENSOR,
    CONF_STROMLIGNING_TOMORROW_SENSOR,
    CONF_TEMPERATURE_SENSOR,
)
from custom_components.open_spot_forecast.ml.gbm import NumpyGradientBoosting
from custom_components.open_spot_forecast.ml.predictor import SpotPricePredictor
from custom_components.open_spot_forecast.ml.retraining import HPO_INTERVAL_DAYS
from custom_components.open_spot_forecast.ml.series_storage import (
    NORDPOOL_PROGNOSES,
    OPENMETEO_WEATHER,
)

TODAY = [1.0 + (slot % 24) / 10 for slot in range(96)]
TOMORROW = [2.0 + (slot % 24) / 10 for slot in range(96)]
# Raw spot prices excl. VAT and tariffs: what the model sees (#16)
SPOT_TODAY = [0.4 + (slot % 24) / 100 for slot in range(96)]
SPOT_TOMORROW = [0.6 + (slot % 24) / 100 for slot in range(96)]


class _Clock:
    """Deterministic UTC clock that advances one second on every reading.

    Keeps "data written after training" strictly newer than the training
    timestamp regardless of the platform clock resolution.
    """

    def __init__(self) -> None:
        self.now = datetime(2026, 9, 24, tzinfo=UTC)

    def __call__(self) -> datetime:
        self.now += timedelta(seconds=1)
        return self.now


def _make_hass(tmp_path: Path) -> Mock:
    """Return a mock Home Assistant whose storage lives in tmp_path."""
    hass = Mock()
    hass.config.path.return_value = str(tmp_path / ".storage")

    async def run_inline(func: Callable[..., Any], *args: Any) -> Any:
        return func(*args)

    hass.async_add_executor_job = run_inline
    return hass


@pytest.fixture
def predictor(tmp_path: Path) -> Iterator[SpotPricePredictor]:
    """Predictor with real SQLite storage, a tiny model and a fake clock."""
    with patch("homeassistant.util.dt.utcnow", new=_Clock()):
        predictor = SpotPricePredictor(_make_hass(tmp_path), "DK1")
        predictor.price_model = NumpyGradientBoosting(n_estimators=2)
        yield predictor
        predictor.storage.close()


def _spy_training(predictor: SpotPricePredictor) -> Any:
    """Patch _train_models with a spy that still trains."""
    return patch.object(predictor, "_train_models", wraps=predictor._train_models)


def _forecast(predictor: SpotPricePredictor, prices: list[float]) -> None:
    """Run one forecast: one day of hourly predictions."""
    predictor.predict({}, prices, 1, 60)


# --- Retrain decision ---------------------------------------------------------


def test_repeated_predict_without_new_data_trains_once(
    predictor: SpotPricePredictor,
) -> None:
    """Forecast runs with unchanged inputs reuse the trained model."""
    with _spy_training(predictor) as train:
        for _ in range(3):
            _forecast(predictor, TODAY)

    assert train.call_count == 1
    assert predictor.is_trained is True
    assert predictor.last_trained_at is not None
    assert predictor.needs_retraining() is False
    assert all(p["model"] == "GradientBoosting" for p in predictor.predictions)


def test_tomorrow_prices_trigger_retrain(predictor: SpotPricePredictor) -> None:
    """Tomorrow's prices extend the price history and retrain the model once."""
    with _spy_training(predictor) as train:
        _forecast(predictor, TODAY)
        _forecast(predictor, TODAY + TOMORROW)
        _forecast(predictor, TODAY + TOMORROW)

    assert train.call_count == 2
    assert predictor.training_samples == len(TODAY + TOMORROW)


def test_corrected_price_triggers_retrain(predictor: SpotPricePredictor) -> None:
    """A changed price for an already stored day counts as new data."""
    corrected = [*TODAY[:-1], TODAY[-1] + 0.5]
    with _spy_training(predictor) as train:
        _forecast(predictor, TODAY)
        _forecast(predictor, corrected)

    assert train.call_count == 2


def test_new_zone_weather_triggers_retrain_a_snapshot_does_not(
    predictor: SpotPricePredictor,
) -> None:
    """Changed zone weather is training data; a local snapshot is not (#23)."""
    with _spy_training(predictor) as train:
        _forecast(predictor, TODAY)
        predictor.storage.insert_weather_snapshot(
            datetime.now().isoformat(), 12.0, 6.5, 240.0, 75.0, 80.0, 0.0
        )
        _forecast(predictor, TODAY)
        assert train.call_count == 1

        predictor.storage.upsert_series(OPENMETEO_WEATHER, [_zone_row(11.0)])
        _forecast(predictor, TODAY)
        _forecast(predictor, TODAY)

    assert train.call_count == 2


def _zone_row(wind: float) -> dict:
    return {"timestamp": "2026-09-24T08:00:00Z", "point": "57.40,10.24"} | {
        "wind_80m": wind,
        "temperature": 12.0,
        "irradiance": 100.0,
        "pressure": 1013.0,
        "humidity": 80.0,
    }


def test_nordpool_rows_trigger_retrain_only_when_changed(
    predictor: SpotPricePredictor,
) -> None:
    """Re-sending identical prognoses is not new data; a changed row is."""
    row = {
        "timestamp": "2026-09-24T12:00:00Z",
        "consumption": 3100.0,
        "solar": None,
        "wind_offshore": 800.0,
        "wind_onshore": 1200.0,
    }
    predictor.storage.upsert_series(NORDPOOL_PROGNOSES, [row])

    with _spy_training(predictor) as train:
        _forecast(predictor, TODAY)
        predictor.storage.upsert_series(NORDPOOL_PROGNOSES, [row])
        _forecast(predictor, TODAY)
        assert train.call_count == 1

        # The per-type breakdown is published later than the total
        predictor.storage.upsert_series(NORDPOOL_PROGNOSES, [{**row, "solar": 450.0}])
        _forecast(predictor, TODAY)

    assert train.call_count == 2
    stored = (
        predictor.storage._ensure_conn()
        .execute(
            "SELECT solar, consumption FROM nordpool_prognoses WHERE timestamp = ?",
            (row["timestamp"],),
        )
        .fetchall()
    )
    assert len(stored) == 1
    assert stored[0][0] == pytest.approx(450.0)
    assert stored[0][1] == pytest.approx(3100.0)


def test_failed_training_is_retried_on_next_run(
    predictor: SpotPricePredictor,
) -> None:
    """A training error leaves the model untrained so the next run retries."""
    real_fit = predictor.price_model.fit
    fit_calls: list[int] = []

    def flaky_fit(X: np.ndarray, y: np.ndarray) -> None:
        fit_calls.append(len(y))
        if len(fit_calls) == 1:
            raise ValueError("boom")
        real_fit(X, y)

    with patch.object(predictor.price_model, "fit", side_effect=flaky_fit):
        _forecast(predictor, TODAY)
        assert predictor.is_trained is False
        assert predictor.last_trained_at is None

        _forecast(predictor, TODAY)

    assert len(fit_calls) == 2
    assert predictor.is_trained is True


def test_training_start_is_the_data_snapshot(predictor: SpotPricePredictor) -> None:
    """Data written while training runs is newer than last_trained_at."""
    real_train = predictor._train_models

    def train_with_concurrent_write() -> None:
        real_train()
        # Simulates a zone weather refresh landing mid-training
        predictor.storage.upsert_series(OPENMETEO_WEATHER, [_zone_row(4.0)])

    with patch.object(
        predictor, "_train_models", side_effect=train_with_concurrent_write
    ):
        _forecast(predictor, TODAY)

    assert predictor.needs_retraining() is True


def test_predict_holds_lock_while_running(predictor: SpotPricePredictor) -> None:
    """Overlapping forecast runs are serialized so a model is never co-fitted."""
    held: list[bool] = []

    def record_lock_state(*_args: Any) -> None:
        held.append(predictor._predict_lock.locked())

    with patch.object(predictor, "_predict", side_effect=record_lock_state):
        _forecast(predictor, TODAY)

    assert held == [True]
    assert predictor._predict_lock.locked() is False


# --- Hyperparameter optimization cadence --------------------------------------


def _record_days(predictor: SpotPricePredictor, days: int) -> None:
    """Store `days` distinct past days of prices."""
    for day in range(days):
        predictor.record_training_prices(TODAY, f"2026-08-{day + 1:02d}")


def test_hpo_counter_counts_new_days_and_is_persisted(
    predictor: SpotPricePredictor,
) -> None:
    """Only a new date bumps the counter; the value is stored in meta."""
    predictor.record_training_prices(TODAY, "2026-09-23")
    predictor.record_training_prices(TODAY, "2026-09-23")
    predictor.record_training_prices(TODAY + TOMORROW, "2026-09-23")
    predictor.record_training_prices(TODAY, "2026-09-24")

    assert predictor._hpo_counter == 2
    assert predictor.storage.load_meta_dict()["hpo_counter"] == "2"


@pytest.mark.asyncio
async def test_hpo_counter_survives_restart(
    predictor: SpotPricePredictor,
) -> None:
    """A restarted predictor restores the persisted HPO counter."""
    _record_days(predictor, 3)

    restarted = SpotPricePredictor(predictor.hass, "DK1")
    try:
        await restarted._load_learning_data()
        assert restarted._hpo_counter == 3
    finally:
        restarted.storage.close()


def test_restore_ignores_invalid_hpo_counter(predictor: SpotPricePredictor) -> None:
    """A corrupt meta value leaves the counter unchanged."""
    predictor._restore_hpo_counter({"hpo_counter": "not-a-number"})

    assert predictor._hpo_counter == 0


def _fake_hpo(predictor: SpotPricePredictor) -> Callable[[], dict]:
    """Mimic _optimize_hyperparameters: swap in an unfitted tuned model."""

    def optimize() -> dict:
        predictor.price_model = NumpyGradientBoosting(n_estimators=3)
        predictor.is_trained = False
        return {"n_estimators": 3, "learning_rate": 0.1}

    return optimize


def test_hpo_runs_after_interval_and_model_stays_trained(
    predictor: SpotPricePredictor,
) -> None:
    """HPO runs once enough new days accrue, then the tuned model is fitted."""
    _record_days(predictor, HPO_INTERVAL_DAYS)

    with (
        patch.object(
            predictor, "_optimize_hyperparameters", side_effect=_fake_hpo(predictor)
        ) as hpo,
        _spy_training(predictor) as train,
    ):
        _forecast(predictor, TODAY)

    hpo.assert_called_once()
    assert train.call_count == 2
    assert predictor.is_trained is True
    assert len(predictor.price_model.trees) == 3
    assert predictor._hpo_counter == 0
    assert predictor.storage.load_meta_dict()["hpo_counter"] == "0"
    assert all(p["model"] == "GradientBoosting" for p in predictor.predictions)


def test_hpo_not_due_before_interval(predictor: SpotPricePredictor) -> None:
    """Fewer than HPO_INTERVAL_DAYS new days never trigger optimization."""
    _record_days(predictor, HPO_INTERVAL_DAYS)
    predictor._hpo_counter = HPO_INTERVAL_DAYS - 2  # the forecast adds today

    with patch.object(predictor, "_optimize_hyperparameters") as hpo:
        _forecast(predictor, TODAY)

    hpo.assert_not_called()
    assert predictor._hpo_counter == HPO_INTERVAL_DAYS - 1


def test_hpo_counter_kept_when_optimization_skipped(
    predictor: SpotPricePredictor,
) -> None:
    """If optimization bails out (too little data), it is retried later."""
    _record_days(predictor, HPO_INTERVAL_DAYS)

    with (
        patch.object(predictor, "_optimize_hyperparameters", return_value=None),
        _spy_training(predictor) as train,
    ):
        _forecast(predictor, TODAY)

    assert train.call_count == 1
    assert predictor._hpo_counter >= HPO_INTERVAL_DAYS
    assert predictor.is_trained is True


# --- Model refit ----------------------------------------------------------------


def test_gradient_boosting_refit_replaces_previous_trees() -> None:
    """Refitting discards the old stumps instead of stacking new ones on top."""
    rng = np.random.default_rng(0)
    x_old, y_old = rng.normal(size=(40, 3)), rng.normal(size=40)
    x_new, y_new = rng.normal(size=(40, 3)), rng.normal(size=40)

    model = NumpyGradientBoosting(n_estimators=4)
    model.fit(x_old, y_old)
    model.fit(x_new, y_new)

    fresh = NumpyGradientBoosting(n_estimators=4)
    fresh.fit(x_new, y_new)

    assert len(model.trees) == 4
    np.testing.assert_allclose(model.predict(x_new), fresh.predict(x_new))


# --- Integration: tomorrow's prices refresh the forecast immediately -------------


def _stromligning(tomorrow: list[float]) -> dict[str, Any]:
    """Stromligning sensor reading with today's (and maybe tomorrow's) prices."""
    return {"today": TODAY, "tomorrow": tomorrow, "raw_today": [], "raw_tomorrow": []}


def _spot(tomorrow: list[float]) -> dict[str, Any]:
    """Spot price reading with today's (and maybe tomorrow's) raw spot prices."""
    return {
        "today": SPOT_TODAY,
        "tomorrow": tomorrow,
        "raw_today": [],
        "raw_tomorrow": [],
    }


def _tomorrow_sensor(tomorrow: list[float]) -> dict[str, Any]:
    """Stromligning tomorrow-sensor reading."""
    return {"available": bool(tomorrow), "tomorrow": tomorrow, "raw_tomorrow": []}


@pytest.mark.asyncio
async def test_tomorrow_prices_arrival_refreshes_forecast(tmp_path: Path) -> None:
    """The 15-minute update starts a forecast refresh when tomorrow appears."""
    hass = _make_hass(tmp_path)
    hass.data = {}
    hass.config_entries.async_forward_entry_setups = AsyncMock()
    entry = MagicMock()
    entry.entry_id = "test"
    entry.options = {}
    entry.data = {
        CONF_REGION: "DK1",
        CONF_ENABLE_ML_PREDICTION: True,
        CONF_STROMLIGNING_SENSOR: "sensor.stromligning_current_price_vat",
        CONF_STROMLIGNING_TOMORROW_SENSOR: "binary_sensor.stromligning_tomorrow",
        CONF_TEMPERATURE_SENSOR: "sensor.outdoor_temperature",
    }

    reader = Mock()
    reader.read_stromligning_sensor.return_value = _stromligning([])
    reader.read_stromligning_tomorrow_sensor.return_value = _tomorrow_sensor([])
    reader.read_spot_prices.return_value = _spot([])
    reader.read_weather_sensors.return_value = {"temperature": 12.0}

    ml_predictor = Mock()
    ml_predictor._load_learning_data = AsyncMock()
    ml_predictor.save_learning_data = AsyncMock()
    ml_predictor.predictions = []
    ml_predictor.learn_from_actual_price.return_value = False

    callbacks: dict[str, Callable[..., Any]] = {}

    def track_time_change(_hass: Any, action: Any, **_kw: Any) -> Mock:
        callbacks[action.__name__] = action
        return Mock()

    module = "custom_components.open_spot_forecast"
    with (
        patch(f"{module}.async_get_integration", new=AsyncMock()),
        patch(f"{module}.SensorReader", return_value=reader),
        patch(f"{module}.SpotPricePredictor", return_value=ml_predictor),
        patch(f"{module}.updater.NordpoolPrognosisSource", autospec=True),
        patch(f"{module}.updater.OpenMeteoWeatherSource", autospec=True),
        patch(f"{module}.async_track_time_change", side_effect=track_time_change),
        patch(f"{module}.tomorrow_prices.async_track_point_in_utc_time"),
        patch(f"{module}.updater.async_dispatcher_send"),
    ):
        assert await async_setup_entry(hass, entry) is True
        assert ml_predictor.predict.call_count == 1

        # The history backfill starts in the background at setup (#32)
        _hass_arg, backfill, name = entry.async_create_background_task.call_args.args
        assert name == "open_spot_forecast_history_backfill"
        backfill.close()
        entry.async_create_background_task.reset_mock()

        # Quarter without tomorrow's prices: no extra refresh
        await callbacks["new_quarter"](datetime.now())
        entry.async_create_background_task.assert_not_called()

        # Tomorrow's prices appear: a refresh is started in the background
        reader.read_stromligning_sensor.return_value = _stromligning(TOMORROW)
        reader.read_stromligning_tomorrow_sensor.return_value = _tomorrow_sensor(
            TOMORROW
        )
        reader.read_spot_prices.return_value = _spot(SPOT_TOMORROW)
        await callbacks["new_quarter"](datetime.now())
        entry.async_create_background_task.assert_called_once()
        _hass_arg, refresh, name = entry.async_create_background_task.call_args.args
        assert name == "open_spot_forecast_tomorrow_prices"

        await refresh
        assert ml_predictor.predict.call_count == 2
        # The model gets the raw spot prices, never the consumer prices
        assert ml_predictor.predict.call_args.args[1] == SPOT_TODAY + SPOT_TOMORROW

        # Already known on the next quarter: no second refresh
        await callbacks["new_quarter"](datetime.now())
        entry.async_create_background_task.assert_called_once()
