"""Shared pytest fixtures."""

from collections.abc import Awaitable, Callable, Iterator
from contextlib import ExitStack
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock, patch
from zoneinfo import ZoneInfo

import pytest

from homeassistant.util import dt as dt_util


@pytest.fixture
def copenhagen_time_zone() -> Iterator[ZoneInfo]:
    """Set Home Assistant's time zone to Europe/Copenhagen for one test."""
    zone = ZoneInfo("Europe/Copenhagen")
    previous = dt_util.get_default_time_zone()
    dt_util.set_default_time_zone(zone)
    yield zone
    dt_util.set_default_time_zone(previous)


@pytest.fixture
def setup_entry(
    tmp_path: Path,
) -> Iterator[Callable[[Any, Any], Awaitable[tuple[dict, dict[str, Any]]]]]:
    """Return an async ``setup(reader, ml_predictor)`` for ``async_setup_entry``.

    The integration runs against a mock Home Assistant with a Stromligning
    and a temperature sensor configured. ``reader`` replaces SensorReader and
    ``ml_predictor`` replaces SpotPricePredictor. ``setup`` returns the
    entry's ``api_data`` and its time-triggered callbacks by function name
    (e.g. ``"new_quarter"``). The patches stay active for the whole test.
    """
    from custom_components.open_spot_forecast import async_setup_entry
    from custom_components.open_spot_forecast.const import (
        CONF_ENABLE_ML_PREDICTION,
        CONF_REGION,
        CONF_STROMLIGNING_SENSOR,
        CONF_TEMPERATURE_SENSOR,
        DOMAIN,
    )

    callbacks: dict[str, Any] = {}

    def track_time_change(_hass: Any, action: Any, **_kw: Any) -> Mock:
        callbacks[action.__name__] = action
        return Mock()

    async def run_inline(func: Callable[..., Any], *args: Any) -> Any:
        return func(*args)

    module = "custom_components.open_spot_forecast"
    with ExitStack() as stack:
        stack.enter_context(patch(f"{module}.async_get_integration", new=AsyncMock()))
        stack.enter_context(
            patch(
                f"{module}.updater.fetch_nordpool_prognoses",
                new=AsyncMock(return_value=[]),
            )
        )
        stack.enter_context(
            patch(f"{module}.async_track_time_change", side_effect=track_time_change)
        )
        stack.enter_context(
            patch(f"{module}.tomorrow_prices.async_track_point_in_utc_time")
        )
        stack.enter_context(patch(f"{module}.updater.async_dispatcher_send"))

        async def setup(reader: Any, ml_predictor: Any) -> tuple[dict, dict[str, Any]]:
            stack.enter_context(patch(f"{module}.SensorReader", return_value=reader))
            stack.enter_context(
                patch(f"{module}.SpotPricePredictor", return_value=ml_predictor)
            )
            hass = Mock()
            hass.data = {}
            hass.config.path.return_value = str(tmp_path / ".storage")
            hass.async_add_executor_job = run_inline
            hass.config_entries.async_forward_entry_setups = AsyncMock()
            entry = MagicMock()
            entry.entry_id = "test"
            entry.options = {}
            entry.data = {
                CONF_REGION: "DK1",
                CONF_ENABLE_ML_PREDICTION: True,
                CONF_STROMLIGNING_SENSOR: "sensor.stromligning_current_price_vat",
                CONF_TEMPERATURE_SENSOR: "sensor.outdoor_temperature",
            }
            assert await async_setup_entry(hass, entry) is True
            return hass.data[DOMAIN]["test"], callbacks

        yield setup
