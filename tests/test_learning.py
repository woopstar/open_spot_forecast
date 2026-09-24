"""Tests for the ML self-learning startup path."""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from custom_components.open_spot_forecast.ml.predictor import SpotPricePredictor


@pytest.mark.asyncio
async def test_load_learning_data_backfills_in_background():
    """The Nordpool backfill is scheduled as a background task, not awaited.

    A populated price_history triggers a Nordpool backfill that can issue
    many network calls. It must not block Home Assistant startup, so
    _load_learning_data hands the coroutine to async_create_background_task
    instead of awaiting it inline.
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

    with patch.object(
        predictor, "_backfill_nordpool_data", new=AsyncMock()
    ) as backfill:
        await predictor._load_learning_data()

    # The backfill coroutine was produced...
    backfill.assert_called_once()
    # ...and handed to Home Assistant as a background task rather than awaited.
    predictor.hass.async_create_background_task.assert_called_once()
    coro = predictor.hass.async_create_background_task.call_args.args[0]
    coro.close()  # Avoid "coroutine was never awaited" warning in tests.
