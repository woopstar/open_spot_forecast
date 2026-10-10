"""The cross-border model's neighbour data in the update cycle (#29)."""

from collections.abc import Callable, Iterator
from datetime import datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock, patch
from zoneinfo import ZoneInfo

import pytest

from custom_components.open_spot_forecast.api.dayahead_prices import (
    NeighbourPriceSource,
)
from custom_components.open_spot_forecast.api.openmeteo_weather import (
    OpenMeteoWeatherSource,
)
from custom_components.open_spot_forecast.history_updater import neighbour_sources
from custom_components.open_spot_forecast.price_source import PriceSettings
from custom_components.open_spot_forecast.updater import (
    FORECAST_DAYS,
    ForecastUpdater,
    SensorEntities,
)

HISTORY = "custom_components.open_spot_forecast.history_updater"
CPH = ZoneInfo("Europe/Copenhagen")
NOW = datetime(2026, 9, 24, 10, tzinfo=CPH)
ZONES = ("DE", "NO2")


def _predictor(zones: tuple[str, ...] | None) -> Mock:
    predictor = Mock()
    predictor.cross_border = None if zones is None else Mock(zones=zones)
    predictor.save_learning_data = AsyncMock()
    predictor.max_history_days = 30
    predictor.price_history = [{"date": "2026-09-20"}]
    predictor.storage.delete_old_weather.return_value = 0
    predictor.storage.delete_old_prices.return_value = 0
    # The region's own sources read this storage too; nothing is stored
    predictor.storage.load_series.return_value = []
    predictor.storage.load_umm_rows.return_value = []
    return predictor


def _source(name: str, changed: bool = False) -> Mock:
    source = Mock()
    source.spec.name = name
    source.async_update = AsyncMock(return_value=changed)
    source.async_prune = AsyncMock(return_value=0)
    return source


@pytest.fixture
def sources() -> Iterator[dict[str, Mock]]:
    """Patch the neighbour sources; return them by spec name as they are built."""
    built: dict[str, Mock] = {}

    def prices(_hass: Any, _storage: Any, zone: str, _key: Any) -> Mock:
        built[f"prices_{zone}"] = _source(f"neighbour_prices_{zone}")
        return built[f"prices_{zone}"]

    def weather(_hass: Any, _storage: Any, zone: str, neighbour: bool) -> Mock:
        assert neighbour is True
        built[f"weather_{zone}"] = _source(f"openmeteo_{zone}")
        return built[f"weather_{zone}"]

    with (
        patch(f"{HISTORY}.NeighbourPriceSource", side_effect=prices),
        patch(f"{HISTORY}.OpenMeteoWeatherSource", side_effect=weather),
    ):
        yield built


def _updater(predictor: Mock | None) -> ForecastUpdater:
    async def run_inline(func: Callable[..., Any], *args: Any) -> Any:
        return func(*args)

    hass = Mock()
    hass.async_add_executor_job = run_inline
    sensors = SensorEntities(*([None] * 9))
    updater = ForecastUpdater(
        hass,
        MagicMock(),
        {"region": "DK1", "sensor_config": sensors.sensor_config()},
        sensors,
        Mock(),
        predictor,
        PriceSettings("stromligning", "DKK", entsoe_api_key="token"),
        predictor.storage if predictor else Mock(),
    )
    # The gas price (#28) has its own tests
    updater.gas = None
    return updater


def test_neighbour_sources_exist_only_with_the_cross_border_model() -> None:
    hass = Mock()

    assert neighbour_sources(hass, None, None) == ([], [])
    assert neighbour_sources(hass, _predictor(None), None) == ([], [])

    prices, weather = neighbour_sources(hass, _predictor(ZONES), "token")

    assert [type(source) for source in prices] == [NeighbourPriceSource] * 2
    assert [type(source) for source in weather] == [OpenMeteoWeatherSource] * 2
    # One table each, with a source state per zone
    assert [source.spec.name for source in prices] == [
        "neighbour_prices_DE",
        "neighbour_prices_NO2",
    ]
    assert {source.spec.table for source in prices} == {"neighbour_prices"}
    assert [source.spec.name for source in weather] == ["openmeteo_DE", "openmeteo_NO2"]
    assert {source.spec.table for source in weather} == {"openmeteo_weather"}
    assert [source.keys() for source in prices] == [["DE"], ["NO2"]]


def test_the_updater_credits_the_neighbour_prices(sources: dict[str, Mock]) -> None:
    on = _updater(_predictor(ZONES))
    off = _updater(_predictor(None))

    assert on.api_data["cross_border"] is True
    assert off.api_data["cross_border"] is False
    assert off.neighbour_prices == off.neighbour_weather == []


@pytest.mark.asyncio
@pytest.mark.usefixtures("copenhagen_time_zone")
async def test_forecast_runs_refresh_the_neighbours(
    sources: dict[str, Mock], caplog: pytest.LogCaptureFixture
) -> None:
    """Prices up to tomorrow, the weather to the forecast's end, from yesterday."""
    updater = _updater(_predictor(ZONES))
    sources["prices_DE"].async_update.side_effect = RuntimeError("rate limited")

    with (
        patch("homeassistant.util.dt.now", return_value=NOW),
        patch.object(updater, "_read_weather", AsyncMock(return_value={})),
        patch.object(updater, "_update_prognoses", AsyncMock()),
        patch.object(updater, "_update_ahead", AsyncMock()),
    ):
        await updater.run_forecast()

    yesterday = datetime(2026, 9, 23, tzinfo=CPH)
    tomorrow_end = datetime(2026, 9, 26, tzinfo=CPH)
    forecast_end = datetime(2026, 9, 24, tzinfo=CPH) + timedelta(days=FORECAST_DAYS + 1)
    for zone in ZONES:
        sources[f"prices_{zone}"].async_update.assert_awaited_once_with(
            yesterday, tomorrow_end
        )
        sources[f"weather_{zone}"].async_update.assert_awaited_once_with(
            yesterday, forecast_end
        )
    # A failing neighbour is logged; the others are still refreshed
    assert "Could not update neighbour_prices_DE: rate limited" in caplog.text


@pytest.mark.asyncio
@pytest.mark.usefixtures("copenhagen_time_zone")
async def test_the_backfill_fills_the_neighbours_history(
    sources: dict[str, Mock],
) -> None:
    updater = _updater(_predictor(ZONES))
    updater.history_prices = None
    updater.weather = updater.load = None
    updater.nordpool = _source("nordpool")
    sources["weather_NO2"].async_update.return_value = True

    with (
        patch("homeassistant.util.dt.now", return_value=NOW),
        patch.object(ForecastUpdater, "refresh_forecast", autospec=True) as refresh,
    ):
        await updater.backfill_history()

    window = (datetime(2026, 9, 20, tzinfo=CPH), datetime(2026, 9, 24, tzinfo=CPH))
    for source in sources.values():
        source.async_update.assert_awaited_once_with(*window)
    # New neighbour history: stage 1 retrains on it at once
    refresh.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.usefixtures("copenhagen_time_zone")
async def test_the_neighbours_history_is_pruned_with_the_rest(
    sources: dict[str, Mock],
) -> None:
    updater = _updater(_predictor(ZONES))
    updater.history_prices = None
    updater.weather = updater.load = None
    updater.nordpool = _source("nordpool")

    with patch("homeassistant.util.dt.now", return_value=NOW):
        await updater.prune_history()

    cutoff = datetime(2026, 8, 23, tzinfo=CPH)
    for source in sources.values():
        source.async_prune.assert_awaited_once_with(cutoff)
