"""Tests for the ML self-learning startup path."""

from unittest.mock import AsyncMock, Mock

import pytest

from custom_components.open_spot_forecast.ml.predictor import SpotPricePredictor


@pytest.mark.asyncio
async def test_load_learning_data_makes_no_network_calls():
    """Loading the learning state starts no background download.

    The Nordpool history backfill moved out of the ML layer into the
    update cycle's gap-aware source (#32, ``ForecastUpdater.backfill_history``).
    """
    predictor = SpotPricePredictor.__new__(SpotPricePredictor)

    predictor.hass = Mock()
    predictor.storage = Mock()
    predictor.storage.async_load_all = AsyncMock(
        return_value={
            "price_history": [{"date": "2026-09-20", "prices": [1.0, 2.0]}],
        }
    )
    predictor.storage.count_predictions = Mock(return_value=0)

    await predictor._load_learning_data()

    assert predictor.price_history == [{"date": "2026-09-20", "prices": [1.0, 2.0]}]
    predictor.hass.async_create_background_task.assert_not_called()
