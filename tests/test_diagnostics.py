"""Config entry diagnostics and the ``Data sources`` sensor (#276).

The diagnostics hold the per-source health, the redacted config entry, the
forecast's freshness, the model's training info and the row counts. No API
key, URL or query parameter appears in them.
"""

import json
from collections.abc import Callable, Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, Mock, patch

import pytest

from homeassistant.const import EntityCategory

from custom_components.open_spot_forecast.const import (
    CONF_ENTSOE_API_KEY,
    CONF_REGION,
    DOMAIN,
    UPDATE_SIGNAL,
)
from custom_components.open_spot_forecast.diagnostics import (
    REDACTED,
    async_get_config_entry_diagnostics,
)
from custom_components.open_spot_forecast.ml.predictor import SpotPricePredictor
from custom_components.open_spot_forecast.sensor import async_setup_entry
from custom_components.open_spot_forecast.source_health import SourceHealth
from custom_components.open_spot_forecast.source_health_sensor import (
    DataSourcesSensor,
)

NOW = datetime(2026, 9, 24, 10, 0, tzinfo=UTC)
API_KEY = "0123abcd-secret-entsoe-key"


@pytest.fixture
def clock() -> Iterator[None]:
    with patch("homeassistant.util.dt.utcnow", return_value=NOW):
        yield


@pytest.fixture
def predictor(tmp_path: Path) -> Iterator[SpotPricePredictor]:
    hass = Mock()
    hass.config.path.return_value = str(tmp_path / ".storage")
    predictor = SpotPricePredictor(hass, "DK1")
    yield predictor
    predictor.storage.close()


def _entry() -> MagicMock:
    entry = MagicMock()
    entry.entry_id = "entry1"
    entry.version = 1
    entry.data = {CONF_REGION: "DK1", CONF_ENTSOE_API_KEY: API_KEY}
    entry.options = {CONF_ENTSOE_API_KEY: API_KEY, "vat": 0.25}
    return entry


def _hass(api_data: dict[str, Any] | None) -> Mock:
    async def run_inline(func: Callable[..., Any], *args: Any) -> Any:
        return func(*args)

    hass = Mock()
    hass.async_add_executor_job = run_inline
    hass.data = {DOMAIN: {"entry1": api_data}} if api_data is not None else {}
    return hass


def _health() -> SourceHealth:
    health = SourceHealth()
    health.record_success("openmeteo_weather")
    # A message carrying the key must not leak it
    health.record_failure("entsoe_load", RuntimeError(f"rejected key {API_KEY}"))
    return health


@pytest.mark.asyncio
async def test_the_diagnostics_show_health_config_and_training_info(
    predictor: SpotPricePredictor, clock: None
) -> None:
    api_data = {
        "ml_predictor": predictor,
        "source_health": _health(),
        "ml_predictions": [{"price": 0.5}] * 3,
        "forecast_generated_at": NOW - timedelta(hours=1),
        "forecast_stale": False,
        "tomorrow_available": True,
    }

    result = await async_get_config_entry_diagnostics(_hass(api_data), _entry())

    assert result["entry"]["data"][CONF_ENTSOE_API_KEY] == REDACTED
    assert result["entry"]["options"][CONF_ENTSOE_API_KEY] == REDACTED
    assert result["entry"]["options"]["vat"] == pytest.approx(0.25)
    assert result["entry"]["data"][CONF_REGION] == "DK1"
    sources = result["sources"]
    assert sources["openmeteo_weather"]["healthy"] is True
    assert sources["entsoe_load"]["healthy"] is False
    assert (
        sources["entsoe_load"]["last_error"] == f"RuntimeError: rejected key {REDACTED}"
    )
    assert result["forecast"] == {
        "generated_at": (NOW - timedelta(hours=1)).isoformat(),
        "stale": False,
        "predictions": 3,
        "tomorrow_available": True,
    }
    model = result["model"]
    assert model["is_trained"] is False
    assert model["holdout_mae"] is None
    assert model["model_options"]["training_days"] == predictor.max_history_days
    assert "training_seconds" in model
    assert result["row_counts"]["predictions"] == 0
    assert "meta" in result["row_counts"]
    # JSON-ready, and the key appears nowhere
    text = json.dumps(result)
    assert API_KEY not in text
    assert "http" not in text


@pytest.mark.asyncio
async def test_an_entry_that_is_not_loaded_shows_its_configuration() -> None:
    result = await async_get_config_entry_diagnostics(_hass(None), _entry())

    assert result["entry"]["data"][CONF_ENTSOE_API_KEY] == REDACTED
    assert result["sources"] == {}
    assert result["model"] is None
    assert result["row_counts"] == {}
    assert API_KEY not in json.dumps(result)


# --- The Data sources sensor ---------------------------------------------------------


def _sensor(api_data: dict[str, Any]) -> DataSourcesSensor:
    return DataSourcesSensor(Mock(), _entry(), api_data)


def test_the_sensor_counts_the_healthy_sources_and_lists_the_failing(
    clock: None,
) -> None:
    sensor = _sensor({"source_health": _health()})

    assert sensor.native_value == 1
    attributes = sensor.extra_state_attributes
    assert attributes["failing"] == ["entsoe_load"]
    assert attributes["sources"]["openmeteo_weather"] == {
        "healthy": True,
        "age_minutes": 0,
        "last_error": None,
        "data_until": None,
    }
    assert sensor.unique_id == "open_spot_forecast_entry1_data_sources"
    assert sensor.entity_category is EntityCategory.DIAGNOSTIC
    assert sensor.translation_key == "data_sources"
    assert sensor.device_info == {"identifiers": {(DOMAIN, "entry1")}}


def test_the_sensor_is_unknown_before_any_attempt() -> None:
    sensor = _sensor({"source_health": SourceHealth()})

    assert sensor.native_value is None
    assert sensor.extra_state_attributes == {"sources": {}, "failing": []}


@pytest.mark.asyncio
async def test_the_sensor_refreshes_on_the_update_signal() -> None:
    sensor = _sensor({"source_health": SourceHealth()})
    sensor.async_on_remove = Mock()  # type: ignore[method-assign]
    module = "custom_components.open_spot_forecast.source_health_sensor"
    with patch(f"{module}.async_dispatcher_connect") as connect:
        await sensor.async_added_to_hass()

    signals = [call.args[1] for call in connect.call_args_list]
    assert UPDATE_SIGNAL in signals or UPDATE_SIGNAL.lower() in signals
    assert sensor.async_on_remove.call_count == 2


@pytest.mark.asyncio
async def test_the_sensor_is_set_up_with_the_source_health() -> None:
    hass = Mock()
    entry = _entry()
    entry.data = {"region": "DK1", "currency": "DKK"}
    entry.options = {}
    hass.data = {DOMAIN: {"entry1": {"source_health": SourceHealth()}}}
    add = Mock()

    await async_setup_entry(hass, entry, add)

    sensors, _ = add.call_args[0]
    assert sum(isinstance(sensor, DataSourcesSensor) for sensor in sensors) == 1
