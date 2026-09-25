"""Tests for the shared price-series validation (#21)."""

import math
from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from homeassistant.util import dt as dt_util

from custom_components.open_spot_forecast import async_setup_entry
from custom_components.open_spot_forecast.const import (
    CONF_ENABLE_ML_PREDICTION,
    CONF_REGION,
    CONF_SPOT_PRICE_SENSOR,
    CONF_STROMLIGNING_SENSOR,
    CONF_TEMPERATURE_SENSOR,
    DOMAIN,
)
from custom_components.open_spot_forecast.ml.predictor import SpotPricePredictor
from custom_components.open_spot_forecast.price_series import is_invalid_price_series


@pytest.mark.parametrize(
    "prices",
    [
        [0.0] * 96,
        [0] * 24,
        [1e-12, -1e-12, 0.0],
        [0.0, None, 0.0],
        [1.2, math.nan, 1.4],
        [1.2, math.inf, 1.4],
    ],
    ids=["all-zero", "all-zero-int", "all-near-zero", "known-all-zero", "nan", "inf"],
)
def test_invalid_series(prices: list[float | None]) -> None:
    """All-zero days and days with a non-finite price are invalid."""
    assert is_invalid_price_series(prices) is True


@pytest.mark.parametrize(
    "prices",
    [
        [],
        [None, None],
        [1.2, None, 1.4],
        [1.2, 0.0, 1.4],
        [0.0] * 95 + [0.01],
        [-0.3, 0.0, 0.5],
        [-0.2] * 96,
    ],
    ids=[
        "empty",
        "no-known-price",
        "missing-slot",
        "some-zero",
        "one-non-zero",
        "negative-and-zero",
        "all-negative",
    ],
)
def test_valid_series(prices: list[float]) -> None:
    """Some zero or negative prices are normal; None is a missing slot."""
    assert is_invalid_price_series(prices) is False


# --- Where invalid days are stopped -----------------------------------------------


def _predictor(tmp_path: Path) -> SpotPricePredictor:
    """Return a predictor backed by a throwaway SQLite store."""
    hass = Mock()
    hass.config.path.return_value = str(tmp_path / ".storage")
    return SpotPricePredictor(hass, "DK1")


def test_invalid_day_is_not_stored_and_keeps_stored_prices(tmp_path: Path) -> None:
    """price_history never receives an all-zero day or loses good prices to one."""
    predictor = _predictor(tmp_path)
    good = [1.0 + slot / 100 for slot in range(96)]

    assert predictor.store_daily_prices(good, "2026-09-24") is True
    assert predictor.store_daily_prices([0.0] * 96, "2026-09-24") is False
    assert predictor.store_daily_prices([0.0] * 96, "2026-09-25") is False

    assert predictor.price_history == [{"date": "2026-09-24", "prices": good}]
    predictor.storage.close()


def test_invalid_day_does_not_trigger_retraining(tmp_path: Path) -> None:
    """Rejected prices neither count as a new HPO day nor as changed data."""
    predictor = _predictor(tmp_path)

    predictor.record_training_prices([0.0] * 96, "2026-09-24")

    assert predictor.price_history == []
    assert predictor._hpo_counter == 0
    assert predictor._prices_updated_at is None
    predictor.storage.close()


def test_predict_on_all_zero_prices_keeps_previous_predictions(
    tmp_path: Path,
) -> None:
    """All-zero known prices are neither stored, trained on nor predicted from."""
    predictor = _predictor(tmp_path)
    previous = [{"start": "2026-09-24T10:00:00+00:00", "price": 1.1}]
    predictor.predictions = list(previous)

    with (
        patch.object(predictor, "record_training_prices") as record,
        patch.object(predictor, "_train_models") as train,
    ):
        predictor.predict({}, [0.0] * 96, 1, 15)

    record.assert_not_called()
    train.assert_not_called()
    assert predictor.predictions == previous
    predictor.storage.close()


@pytest.mark.asyncio
async def test_all_zero_sensor_is_not_learned_from(tmp_path: Path) -> None:
    """End to end: the 15-minute update never learns from an all-zero day.

    A real SensorReader reads mocked Stromligning states: the consumer price
    (displayed) and the raw spot price (learned from, #16). While they report
    all zeros, today's prices stay as they were and self-learning is skipped;
    good prices are learned from, and a later all-zero read keeps them.
    """
    start = dt_util.start_of_local_day()

    def strom_state(prices: list[float]) -> Mock:
        items = [
            {"price": price, "start": (start + timedelta(minutes=15 * i)).isoformat()}
            for i, price in enumerate(prices)
        ]
        return Mock(state=str(prices[0]), attributes={"prices": items})

    good = [1.0 + slot / 100 for slot in range(96)]
    states = {
        "sensor.strom": strom_state([0.0] * 96),
        "sensor.spot": strom_state([0.0] * 96),
        "sensor.outdoor_temperature": Mock(state="12.0", attributes={}),
    }

    async def run_inline(func: Callable[..., Any], *args: Any) -> Any:
        return func(*args)

    hass = Mock()
    hass.data = {}
    hass.states.get.side_effect = states.get
    hass.async_add_executor_job = run_inline
    hass.config_entries.async_forward_entry_setups = AsyncMock()
    entry = MagicMock()
    entry.entry_id = "test"
    entry.options = {}
    entry.data = {
        CONF_REGION: "DK1",
        CONF_ENABLE_ML_PREDICTION: True,
        CONF_STROMLIGNING_SENSOR: "sensor.strom",
        CONF_SPOT_PRICE_SENSOR: "sensor.spot",
        CONF_TEMPERATURE_SENSOR: "sensor.outdoor_temperature",
    }
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
        patch(f"{module}.SpotPricePredictor", return_value=ml_predictor),
        patch(
            f"{module}.updater.fetch_nordpool_prognoses", new=AsyncMock(return_value=[])
        ),
        patch(f"{module}.async_track_time_change", side_effect=track_time_change),
        patch(f"{module}.tomorrow_prices.async_track_point_in_utc_time"),
        patch(f"{module}.updater.async_dispatcher_send"),
    ):
        assert await async_setup_entry(hass, entry) is True
        api_data = hass.data[DOMAIN]["test"]
        assert api_data["prices_today"] == []

        await callbacks["new_quarter"](datetime.now())
        ml_predictor.learn_from_actual_price.assert_not_called()

        states["sensor.strom"] = strom_state(good)
        states["sensor.spot"] = strom_state(good)
        await callbacks["new_quarter"](datetime.now())
        assert api_data["prices_today"] == pytest.approx(good)
        assert ml_predictor.learn_from_actual_price.call_count == 1

        states["sensor.strom"] = strom_state([0.0] * 96)
        states["sensor.spot"] = strom_state([0.0] * 96)
        await callbacks["new_quarter"](datetime.now())
        assert api_data["prices_today"] == pytest.approx(good)
        assert ml_predictor.learn_from_actual_price.call_count == 1
