"""ForecastUpdater: the update cycle moved out of ``async_setup_entry`` (#54).

The forecast pipeline (weather → Nordpool → known-data end → predict → save)
exists once, in ``run_forecast``; the initial fetch, the tomorrow-price
refresh and the 6-hourly update all go through it.
"""

import re
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock, patch
from zoneinfo import ZoneInfo

import pytest

from homeassistant.util import dt as dt_util

from custom_components.open_spot_forecast.const import (
    CONF_SPOT_PRICE_SENSOR,
    CONF_STROMLIGNING_SENSOR,
    CONF_TEMPERATURE_SENSOR,
    DEFAULT_SPOT_PRICE_SENSOR,
    DEFAULT_SPOT_PRICE_TOMORROW_SENSOR,
    UPDATE_SIGNAL,
    UPDATE_SIGNAL_FORECAST,
)
from custom_components.open_spot_forecast.price_source import PriceSettings
from custom_components.open_spot_forecast.updater import (
    FORECAST_DAYS,
    INTERVAL_MINUTES,
    ForecastUpdater,
    SensorEntities,
)

MODULE = "custom_components.open_spot_forecast.updater"
INTEGRATION = Path(__file__).parent.parent / "custom_components/open_spot_forecast"
CPH = ZoneInfo("Europe/Copenhagen")
SPOT_TODAY = [0.5] * 96
KNOWN_END = datetime(2026, 9, 24, 22, 0, tzinfo=UTC)
NOW = datetime(2026, 9, 24, 10, 20, tzinfo=CPH)
ZONE_ROW = {"timestamp": "2026-09-24T08:00:00+00:00", "point": "57.40,10.24"}
YESTERDAY_ZONE_ROW = {"timestamp": "2026-09-22T21:45:00+00:00", "point": "x"}
NP_ROW = {
    "timestamp": "2026-09-24T10:00:00Z",
    "consumption": 4000.0,
    "solar": 50.0,
    "wind_offshore": 100.0,
    "wind_onshore": 200.0,
}


def _dayahead_spot(tomorrow: Sequence[float | None] = ()) -> dict[str, list]:
    """Day-ahead prices (DKK/kWh excl. VAT) as ``DayAheadPrices.async_read`` returns."""
    return {
        "today": [0.4] * 96,
        "tomorrow": list(tomorrow),
        "raw_today": [{"start": "2026-09-24T21:45:00+00:00"}],
        "raw_tomorrow": [],
    }


def _sensors(**overrides: str | None) -> SensorEntities:
    values: dict[str, str | None] = {
        "stromligning": "sensor.strom",
        "stromligning_tomorrow": None,
        "spot_price": "sensor.spot",
        "spot_price_tomorrow": "binary_sensor.spot_tomorrow",
        "wind_speed": None,
        "wind_direction": None,
        "solar_power": None,
        "solar_forecast": None,
        "temperature": "sensor.outdoor_temperature",
    }
    values.update(overrides)
    return SensorEntities(**values)


def _stromligning(today: list[float], tomorrow: list[float]) -> dict[str, Any]:
    return {"today": today, "tomorrow": tomorrow, "raw_today": [], "raw_tomorrow": []}


@dataclass
class Harness:
    """A ForecastUpdater on mocks, with the patched module functions."""

    updater: ForecastUpdater
    reader: Mock
    predictor: Mock
    entry: MagicMock
    api_data: dict[str, Any]
    nordpool: Mock
    nordpool_class: Mock
    dayahead: Mock
    weather: Mock
    read_forecast: AsyncMock
    dispatch: Mock


@pytest.fixture
def make() -> Iterator[Callable[..., Harness]]:
    """Return ``make(sensors=..., ml=True)`` building a Harness."""
    nordpool = Mock()
    nordpool.async_update = AsyncMock(return_value=True)
    nordpool.async_load = AsyncMock(return_value=[NP_ROW])
    nordpool.async_prune = AsyncMock(return_value=3)
    weather = Mock()
    weather.async_update = AsyncMock(return_value=True)
    weather.async_load = AsyncMock(return_value=[YESTERDAY_ZONE_ROW, ZONE_ROW])
    weather.async_prune = AsyncMock(return_value=7)
    dayahead = Mock()
    dayahead.async_read = AsyncMock(return_value=_dayahead_spot())
    dayahead.async_history = AsyncMock(return_value={})
    dayahead.async_prune = AsyncMock(return_value=5)
    read_forecast = AsyncMock(return_value=None)
    dispatch = Mock()
    with (
        patch(f"{MODULE}.NordpoolPrognosisSource", return_value=nordpool) as source,
        patch(f"{MODULE}.DayAheadPrices", return_value=dayahead),
        patch(f"{MODULE}.OpenMeteoWeatherSource", return_value=weather),
        patch(f"{MODULE}.async_read_weather_forecast", read_forecast),
        patch(f"{MODULE}.async_dispatcher_send", dispatch),
        patch(f"{MODULE}.ml_price_inputs", return_value=(SPOT_TODAY, KNOWN_END)),
    ):

        def build(
            sensors: SensorEntities | None = None,
            ml: bool = True,
            price_source: str = "stromligning",
        ) -> Harness:
            sensors = sensors or _sensors()

            async def run_inline(func: Callable[..., Any], *args: Any) -> Any:
                return func(*args)

            hass = Mock()
            hass.async_add_executor_job = run_inline
            entry = MagicMock()
            # A background task's coroutine is closed, not left un-awaited
            entry.async_create_background_task.side_effect = lambda _hass, coro, _name: (
                coro.close()
            )
            reader = Mock()
            reader.read_stromligning_sensor.return_value = _stromligning([1.0] * 96, [])
            reader.read_spot_prices.return_value = _stromligning(SPOT_TODAY, [])
            reader.read_weather_sensors.return_value = {"temperature": 12.0}
            predictor = Mock()
            predictor.save_learning_data = AsyncMock()
            predictor.predictions = [{"price": 0.5}]
            predictor.learn_from_actual_price.return_value = False
            predictor.max_history_days = 30
            predictor.price_history = []
            predictor.storage.delete_old_weather.return_value = 1
            predictor.storage.delete_old_prices.return_value = 2
            api_data: dict[str, Any] = {
                "region": "DK1",
                "prices_today": [],
                "prices_tomorrow": [],
                "ml_predictions": [],
                "weather_data": {},
                "tomorrow_available": False,
                "last_update": None,
                "sensor_config": sensors.sensor_config(),
            }
            updater = ForecastUpdater(
                hass,
                entry,
                api_data,
                sensors,
                reader,
                predictor if ml else None,
                PriceSettings(price_source, "DKK", 0.25),
                predictor.storage if ml else Mock(),
            )
            return Harness(
                updater,
                reader,
                predictor,
                entry,
                api_data,
                nordpool,
                source,
                dayahead,
                weather,
                read_forecast,
                dispatch,
            )

        yield build


def _signals(harness: Harness) -> list[str]:
    return [call.args[1] for call in harness.dispatch.call_args_list]


# --- Configuration -------------------------------------------------------------------


def test_sensor_entities_prefer_options_over_data() -> None:
    entry = MagicMock()
    entry.data = {
        CONF_STROMLIGNING_SENSOR: "sensor.strom_data",
        CONF_TEMPERATURE_SENSOR: "sensor.temp",
    }
    entry.options = {
        CONF_STROMLIGNING_SENSOR: "sensor.strom_options",
        CONF_SPOT_PRICE_SENSOR: "sensor.spot_options",
    }

    sensors = SensorEntities.from_entry(entry)

    assert sensors.stromligning == "sensor.strom_options"
    assert sensors.spot_price == "sensor.spot_options"
    assert sensors.spot_price_tomorrow == DEFAULT_SPOT_PRICE_TOMORROW_SENSOR
    assert sensors.temperature == "sensor.temp"
    assert sensors.wind_speed is None
    assert sensors.has_weather is True
    assert sensors.sensor_config() == {
        "stromligning_sensor": "sensor.strom_options",
        "wind_speed_sensor": None,
        "wind_direction_sensor": None,
        "solar_power_sensor": None,
        "solar_forecast_sensor": None,
        "temperature_sensor": "sensor.temp",
    }


def test_sensor_entities_defaults_without_configuration() -> None:
    entry = MagicMock()
    entry.data = {}
    entry.options = {}

    sensors = SensorEntities.from_entry(entry)

    assert sensors.spot_price == DEFAULT_SPOT_PRICE_SENSOR
    assert sensors.stromligning is None
    assert sensors.has_weather is False


# --- The forecast pipeline -----------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.usefixtures("copenhagen_time_zone")
async def test_run_forecast_reads_weather_stores_prognoses_predicts_and_saves(
    make: Callable[..., Harness],
) -> None:
    harness = make(
        _sensors(
            wind_speed="weather.home", solar_forecast="sensor.solcast_forecast_today"
        )
    )
    harness.reader.read_solcast_sensor.return_value = {"today": 5.0}
    forecast = [{"datetime": "2026-09-24T10:00:00+00:00", "wind_speed": 5.0}]
    harness.read_forecast.return_value = forecast

    with patch("homeassistant.util.dt.now", return_value=NOW):
        await harness.updater.run_forecast()

    weather = harness.api_data["weather_data"]
    assert weather["temperature"] == pytest.approx(12.0)
    assert weather["solcast_forecast"] == {"today": 5.0}
    assert weather["weather_forecast"] == forecast
    harness.read_forecast.assert_awaited_once_with(harness.updater.hass, "weather.home")
    # Only today's and tomorrow's missing prognoses are requested; the stored
    # rows are what the forecast reads
    harness.nordpool_class.assert_called_once_with(
        harness.updater.hass, harness.predictor.storage, "DK1"
    )
    today = datetime(2026, 9, 24, tzinfo=CPH)
    window = (today, today + timedelta(days=2))
    harness.nordpool.async_update.assert_awaited_once_with(*window)
    harness.nordpool.async_load.assert_awaited_once_with(*window)
    assert weather["consumption_prognosis"] == {"2026-09-24T10:00:00Z": 4000.0}
    assert weather["production_prognosis"] == [
        {
            "deliveryStart": "2026-09-24T10:00:00Z",
            "solar": 50.0,
            "wind_offshore": 100.0,
            "wind_onshore": 200.0,
        }
    ]
    harness.predictor.predict.assert_called_once_with(
        weather, SPOT_TODAY, FORECAST_DAYS, INTERVAL_MINUTES, KNOWN_END
    )
    assert harness.api_data["ml_predictions"] == [{"price": 0.5}]
    harness.predictor.save_learning_data.assert_awaited_once()


@pytest.mark.asyncio
async def test_run_forecast_skips_a_non_solcast_forecast_and_empty_weather_forecast(
    make: Callable[..., Harness],
) -> None:
    harness = make(_sensors(wind_speed="weather.home", solar_forecast="sensor.pv"))

    await harness.updater.run_forecast()

    harness.reader.read_solcast_sensor.assert_not_called()
    assert "weather_forecast" not in harness.api_data["weather_data"]
    harness.predictor.predict.assert_called_once()


@pytest.mark.asyncio
async def test_run_forecast_without_ml_only_reads_weather(
    make: Callable[..., Harness],
) -> None:
    """The prognoses only feed the model, and live in its database."""
    harness = make(ml=False)

    await harness.updater.run_forecast()

    harness.nordpool_class.assert_not_called()
    assert harness.api_data["weather_data"] == {"temperature": 12.0}
    harness.predictor.predict.assert_not_called()


@pytest.mark.asyncio
async def test_run_forecast_keeps_going_when_the_prognoses_fail(
    make: Callable[..., Harness], caplog: pytest.LogCaptureFixture
) -> None:
    """A failed update still reads what is stored; a failed read skips them."""
    harness = make()
    harness.reader.read_weather_sensors.side_effect = lambda _config: {
        "temperature": 12.0
    }
    harness.nordpool.async_update.side_effect = RuntimeError("offline")

    await harness.updater.run_forecast()

    assert "consumption_prognosis" in harness.api_data["weather_data"]
    harness.nordpool.async_load.side_effect = RuntimeError("locked")

    await harness.updater.run_forecast()

    assert "consumption_prognosis" not in harness.api_data["weather_data"]
    assert harness.predictor.predict.call_count == 2
    assert "Could not update the Nordpool prognoses: offline" in caplog.text
    assert "Could not read the stored Nordpool prognoses: locked" in caplog.text


@pytest.mark.asyncio
async def test_run_forecast_without_weather_or_prognoses_does_not_predict(
    make: Callable[..., Harness],
) -> None:
    harness = make(_sensors(temperature=None))
    harness.nordpool.async_load.return_value = []

    await harness.updater.run_forecast()

    harness.reader.read_weather_sensors.assert_not_called()
    assert harness.api_data["weather_data"] == {}
    harness.predictor.predict.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("caller", ["initial", "refresh", "six_hourly"])
async def test_every_forecast_path_runs_the_one_pipeline(
    make: Callable[..., Harness], caller: str
) -> None:
    harness = make()
    updater = harness.updater
    with patch.object(ForecastUpdater, "run_forecast", autospec=True) as run:
        if caller == "initial":
            await updater.async_initial_fetch()
        elif caller == "refresh":
            await updater.refresh_forecast()
        else:
            await updater.update_forecasts(dt_util.utcnow())

    run.assert_awaited_once_with(updater)


def test_the_pipeline_is_not_duplicated() -> None:
    """Only run_forecast predicts; __init__.py never calls the model itself."""
    updater_source = (INTEGRATION / "updater.py").read_text()
    init_source = (INTEGRATION / "__init__.py").read_text()

    assert len(re.findall(r"ml_predictor\.predict,", updater_source)) == 1
    assert len(re.findall(r"self\._update_prognoses\(", updater_source)) == 1
    assert not re.search(r"\.predict\b", init_source)
    assert "NordpoolPrognosisSource" not in init_source


# --- Initial fetch -------------------------------------------------------------------


@pytest.mark.asyncio
async def test_initial_fetch_loads_prices_and_the_first_forecast(
    make: Callable[..., Harness],
) -> None:
    harness = make()

    await harness.updater.async_initial_fetch()

    assert harness.api_data["prices_today"] == [1.0] * 96
    assert harness.api_data["price_source"] == "stromligning"
    assert harness.api_data["spot_data"]["today"] == SPOT_TODAY
    assert harness.api_data["tomorrow_available"] is False
    assert isinstance(harness.api_data["last_update"], datetime)
    harness.predictor.predict.assert_called_once()
    harness.dispatch.assert_not_called()


@pytest.mark.asyncio
async def test_initial_fetch_warns_about_missing_prices(
    make: Callable[..., Harness], caplog: pytest.LogCaptureFixture
) -> None:
    harness = make()
    harness.reader.read_stromligning_sensor.return_value = _stromligning([], [])
    harness.reader.read_spot_prices.return_value = _stromligning([], [])

    await harness.updater.async_initial_fetch()

    assert "Stromligning sensor has no valid prices yet" in caplog.text
    assert "Spot price sensor sensor.spot has no prices" in caplog.text
    assert harness.api_data["prices_today"] == []


@pytest.mark.asyncio
async def test_initial_fetch_without_a_price_sensor(
    make: Callable[..., Harness], caplog: pytest.LogCaptureFixture
) -> None:
    harness = make(_sensors(stromligning=None))

    await harness.updater.async_initial_fetch()

    assert "No price sensor configured" in caplog.text
    harness.reader.read_stromligning_sensor.assert_not_called()
    harness.reader.read_spot_prices.assert_called_once()


# --- Prices and tomorrow -------------------------------------------------------------


def test_read_prices_merges_the_tomorrow_sensor(make: Callable[..., Harness]) -> None:
    harness = make(_sensors(stromligning_tomorrow="binary_sensor.strom_tomorrow"))
    harness.reader.read_stromligning_tomorrow_sensor.return_value = {
        "available": True,
        "tomorrow": [2.0] * 96,
        "raw_tomorrow": [{"start": "x"}],
    }

    assert harness.updater.read_prices() is True

    assert harness.api_data["prices_tomorrow"] == [2.0] * 96
    assert harness.api_data["stromligning_data"]["raw_tomorrow"] == [{"start": "x"}]


def test_read_prices_keeps_the_old_prices_without_valid_ones(
    make: Callable[..., Harness],
) -> None:
    harness = make()
    harness.api_data["prices_today"] = [3.0]
    harness.reader.read_stromligning_sensor.return_value = _stromligning([], [])

    assert harness.updater.read_prices() is False
    assert harness.api_data["prices_today"] == [3.0]


@pytest.mark.asyncio
async def test_check_tomorrow_prices_refreshes_once_when_complete(
    make: Callable[..., Harness],
) -> None:
    harness = make()
    tomorrow = [2.0] * 100  # enough slots for any day
    harness.reader.read_stromligning_sensor.return_value = _stromligning(
        [1.0] * 96, tomorrow
    )
    with patch(f"{MODULE}.tomorrow_prices_complete", return_value=True):
        assert await harness.updater.check_tomorrow_prices() is True
        assert await harness.updater.check_tomorrow_prices() is True

    harness.entry.async_create_background_task.assert_called_once()
    _hass, refresh, name = harness.entry.async_create_background_task.call_args.args
    assert name == "open_spot_forecast_tomorrow_prices"
    refresh.close()
    assert _signals(harness) == [UPDATE_SIGNAL, UPDATE_SIGNAL]


@pytest.mark.asyncio
async def test_refresh_forecast_without_ml_only_rereads_prices(
    make: Callable[..., Harness],
) -> None:
    harness = make(ml=False)

    await harness.updater.refresh_forecast()

    harness.nordpool_class.assert_not_called()
    assert harness.api_data["prices_today"] == [1.0] * 96
    assert _signals(harness) == [UPDATE_SIGNAL]


# --- Timed updates -------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("ml", [True, False])
async def test_update_forecasts_signals_the_forecast_update(
    make: Callable[..., Harness], ml: bool
) -> None:
    harness = make(ml=ml)

    await harness.updater.update_forecasts(dt_util.utcnow())

    assert harness.predictor.predict.call_count == (1 if ml else 0)
    assert _signals(harness) == [UPDATE_SIGNAL_FORECAST]


@pytest.mark.asyncio
async def test_new_day_rotates_tomorrow_to_today(
    make: Callable[..., Harness],
) -> None:
    harness = make()
    harness.api_data["tomorrow_available"] = True
    harness.reader.read_stromligning_sensor.return_value = _stromligning(
        [1.0] * 96, [2.0] * 96
    )

    await harness.updater.new_day(dt_util.utcnow())

    assert harness.api_data["prices_today"] == [2.0] * 96
    assert harness.api_data["prices_tomorrow"] == []
    assert harness.api_data["tomorrow_available"] is False
    harness.reader.read_spot_prices.assert_called_once()
    assert _signals(harness) == [UPDATE_SIGNAL]


@pytest.mark.asyncio
@pytest.mark.parametrize("stromligning", ["sensor.strom", None])
async def test_new_day_without_tomorrow_clears_it(
    make: Callable[..., Harness], stromligning: str | None
) -> None:
    harness = make(_sensors(stromligning=stromligning))
    harness.api_data["prices_today"] = [1.0] * 96
    harness.api_data["prices_tomorrow"] = [2.0] * 96
    harness.api_data["tomorrow_available"] = True

    await harness.updater.new_day(dt_util.utcnow())

    assert harness.api_data["prices_today"] == [1.0] * 96
    assert harness.api_data["prices_tomorrow"] == []
    assert harness.api_data["tomorrow_available"] is False


@pytest.mark.asyncio
async def test_new_quarter_stores_a_weather_snapshot(
    make: Callable[..., Harness],
) -> None:
    harness = make(_sensors(wind_speed="sensor.wind"))
    now = datetime(2026, 9, 24, 10, 20, tzinfo=CPH)

    with (
        patch("homeassistant.util.dt.now", return_value=now),
        patch("homeassistant.util.dt.utcnow", return_value=now.astimezone(UTC)),
    ):
        await harness.updater.new_quarter(now)

    # Keyed by the UTC slot start (#59): 10:20 local is in the 08:15 UTC slot
    harness.predictor.storage.insert_weather_snapshot.assert_called_once_with(
        "2026-09-24T08:15:00Z", 12.0, None, None, None, None, None
    )
    # Pruning is daily and follows the training window (#32), not every
    # 100th snapshot
    harness.predictor.storage.delete_old_weather.assert_not_called()
    assert _signals(harness) == [UPDATE_SIGNAL]


@pytest.mark.asyncio
async def test_new_quarter_survives_a_failed_weather_snapshot(
    make: Callable[..., Harness],
) -> None:
    harness = make(_sensors(wind_speed="sensor.wind"))
    harness.predictor.storage.insert_weather_snapshot.side_effect = OSError("disk")

    await harness.updater.new_quarter(dt_util.utcnow())

    harness.predictor.storage.delete_old_weather.assert_not_called()
    assert _signals(harness) == [UPDATE_SIGNAL]


@pytest.mark.asyncio
@pytest.mark.usefixtures("copenhagen_time_zone")
async def test_new_quarter_learns_from_the_current_slot_and_saves(
    make: Callable[..., Harness],
) -> None:
    harness = make()
    spot = [float(i) for i in range(96)]
    harness.reader.read_spot_prices.return_value = _stromligning(spot, [])
    harness.predictor.learn_from_actual_price.return_value = True
    now = datetime(2026, 9, 24, 10, 20, tzinfo=CPH)

    with patch("homeassistant.util.dt.now", return_value=now):
        await harness.updater.new_quarter(now)

    harness.predictor.learn_from_actual_price.assert_called_once_with(
        "2026-09-24T10:15:00+02:00", pytest.approx(41.0)
    )
    harness.predictor.save_learning_data.assert_awaited_once()


@pytest.mark.asyncio
async def test_new_quarter_logs_a_learning_error(
    make: Callable[..., Harness], caplog: pytest.LogCaptureFixture
) -> None:
    harness = make()
    harness.predictor.learn_from_actual_price.side_effect = ValueError("boom")
    now = datetime(2026, 9, 24, 10, 20, tzinfo=CPH)

    with patch("homeassistant.util.dt.now", return_value=now):
        await harness.updater.new_quarter(now)

    assert "Self-learning update error: boom" in caplog.text
    assert _signals(harness) == [UPDATE_SIGNAL]


@pytest.mark.asyncio
async def test_new_quarter_reads_tomorrow_and_refreshes_when_it_arrives(
    make: Callable[..., Harness],
) -> None:
    harness = make(_sensors(stromligning_tomorrow="binary_sensor.strom_tomorrow"))
    harness.reader.read_stromligning_tomorrow_sensor.return_value = {
        "available": True,
        "tomorrow": [2.0] * 96,
        "raw_tomorrow": [],
    }

    with patch(f"{MODULE}.tomorrow_prices_complete", return_value=True):
        await harness.updater.new_quarter(dt_util.utcnow() + timedelta(minutes=1))

    assert harness.api_data["prices_tomorrow"] == [2.0] * 96
    assert harness.api_data["tomorrow_available"] is True
    harness.entry.async_create_background_task.assert_called_once()
    harness.entry.async_create_background_task.call_args.args[1].close()


# --- Stored history (#32) ------------------------------------------------------------


def _background_tasks(harness: Harness) -> list[str]:
    return [
        call.args[2]
        for call in harness.entry.async_create_background_task.call_args_list
    ]


@pytest.mark.asyncio
async def test_initial_fetch_backfills_the_history_in_the_background(
    make: Callable[..., Harness],
) -> None:
    harness = make()

    await harness.updater.async_initial_fetch()

    assert _background_tasks(harness) == ["open_spot_forecast_history_backfill"]


@pytest.mark.asyncio
@pytest.mark.usefixtures("copenhagen_time_zone")
async def test_backfill_covers_the_price_history_up_to_today(
    make: Callable[..., Harness],
) -> None:
    """Only the missing days are fetched (the source skips stored ones)."""
    harness = make()
    harness.predictor.price_history = [
        {"date": "2026-09-20"},
        {"date": "2026-09-18"},
        {"date": "not a date"},
    ]

    with (
        patch("homeassistant.util.dt.now", return_value=NOW),
        patch.object(ForecastUpdater, "refresh_forecast", autospec=True) as refresh,
    ):
        await harness.updater.backfill_history()

    window = (datetime(2026, 9, 18, tzinfo=CPH), datetime(2026, 9, 24, tzinfo=CPH))
    # The zone weather (Open-Meteo's archive, #23) and the prognoses
    harness.weather.async_update.assert_awaited_once_with(*window)
    harness.nordpool.async_update.assert_awaited_once_with(*window)
    # New history: the model retrains on it at once
    refresh.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.usefixtures("copenhagen_time_zone")
async def test_a_backfill_without_new_data_does_not_refresh(
    make: Callable[..., Harness], caplog: pytest.LogCaptureFixture
) -> None:
    harness = make()
    harness.predictor.price_history = [{"date": "2026-09-20"}]
    harness.nordpool.async_update.return_value = False
    harness.weather.async_update.side_effect = RuntimeError("archive down")

    with (
        patch("homeassistant.util.dt.now", return_value=NOW),
        patch.object(ForecastUpdater, "refresh_forecast", autospec=True) as refresh,
    ):
        await harness.updater.backfill_history()

    refresh.assert_not_awaited()
    assert "Open-Meteo history backfill failed: archive down" in caplog.text


@pytest.mark.asyncio
async def test_backfill_needs_price_history_and_logs_failures(
    make: Callable[..., Harness], caplog: pytest.LogCaptureFixture
) -> None:
    harness = make()

    await harness.updater.backfill_history()

    harness.nordpool.async_update.assert_not_awaited()
    harness.predictor.price_history = [{"date": "2026-09-20"}]
    harness.nordpool.async_update.side_effect = RuntimeError("offline")

    await harness.updater.backfill_history()

    assert "Nordpool history backfill failed: offline" in caplog.text


@pytest.mark.asyncio
@pytest.mark.usefixtures("copenhagen_time_zone")
async def test_new_day_prunes_beyond_the_training_window_and_backfills(
    make: Callable[..., Harness],
) -> None:
    """History is kept for the training window (30 days) plus 2 days."""
    harness = make()

    with patch("homeassistant.util.dt.now", return_value=NOW):
        await harness.updater.new_day(NOW)

    storage = harness.predictor.storage
    storage.delete_old_weather.assert_called_once_with(32)
    storage.delete_old_prices.assert_called_once_with("2026-08-23")
    harness.nordpool.async_prune.assert_awaited_once_with(
        datetime(2026, 8, 23, tzinfo=CPH)
    )
    assert _background_tasks(harness) == ["open_spot_forecast_history_backfill"]


@pytest.mark.asyncio
async def test_a_failed_prune_is_logged(
    make: Callable[..., Harness], caplog: pytest.LogCaptureFixture
) -> None:
    harness = make()
    harness.nordpool.async_prune.side_effect = RuntimeError("locked")

    await harness.updater.prune_history()

    assert "Could not prune the stored history: locked" in caplog.text


@pytest.mark.asyncio
async def test_without_ml_there_is_no_history_to_keep(
    make: Callable[..., Harness],
) -> None:
    harness = make(ml=False)

    await harness.updater.new_day(NOW)
    await harness.updater.backfill_history()

    harness.nordpool_class.assert_not_called()
    assert _background_tasks(harness) == []


# --- The day-ahead price source (#27) ------------------------------------------------


@pytest.mark.asyncio
async def test_dayahead_prices_are_shown_with_vat_and_are_the_models(
    make: Callable[..., Harness],
) -> None:
    harness = make(price_source="dayahead")

    assert await harness.updater.async_read_prices() is True

    api_data = harness.api_data
    assert api_data["spot_data"] == _dayahead_spot()
    assert api_data["prices_today"] == [pytest.approx(0.5)] * 96
    assert api_data["prices_tomorrow"] == []
    assert api_data["price_source"] == "dayahead"
    # Stromligning is not read, even with its sensor configured
    harness.reader.read_stromligning_sensor.assert_not_called()
    harness.updater.read_spot_prices()
    harness.reader.read_spot_prices.assert_not_called()


@pytest.mark.asyncio
async def test_without_todays_dayahead_prices_the_previous_are_kept(
    make: Callable[..., Harness], caplog: pytest.LogCaptureFixture
) -> None:
    harness = make(price_source="dayahead")
    harness.api_data["prices_today"] = [1.0]
    harness.dayahead.async_read.return_value = {"today": [], "tomorrow": []}

    assert await harness.updater.async_read_prices() is False
    harness.dayahead.async_read.side_effect = RuntimeError("locked")
    assert await harness.updater.async_read_prices() is False

    assert harness.api_data["prices_today"] == [1.0]
    assert "Could not read the day-ahead prices: locked" in caplog.text


@pytest.mark.asyncio
async def test_the_dayahead_source_needs_its_storage(
    make: Callable[..., Harness],
) -> None:
    harness = make(price_source="dayahead")
    updater = ForecastUpdater(
        harness.updater.hass,
        harness.entry,
        harness.api_data,
        _sensors(),
        harness.reader,
        None,
        PriceSettings("dayahead", "DKK", 0.25),
        None,
    )

    assert updater.dayahead is None


@pytest.mark.asyncio
async def test_initial_fetch_reads_dayahead_prices(
    make: Callable[..., Harness], caplog: pytest.LogCaptureFixture
) -> None:
    harness = make(price_source="dayahead")
    harness.dayahead.async_read.return_value = {"today": [], "tomorrow": []}

    await harness.updater.async_initial_fetch()

    assert "No day-ahead prices for today yet" in caplog.text
    assert "Spot price sensor" not in caplog.text


@pytest.mark.asyncio
async def test_quarters_and_midnight_read_dayahead_prices(
    make: Callable[..., Harness],
) -> None:
    harness = make(price_source="dayahead")
    harness.api_data["prices_tomorrow"] = [2.0] * 96

    await harness.updater.new_day(NOW)
    assert harness.api_data["prices_tomorrow"] == []

    # Tomorrow's prices are published: the forecast refreshes
    harness.dayahead.async_read.return_value = _dayahead_spot([0.8] * 96)
    await harness.updater.new_quarter(NOW)

    assert harness.api_data["tomorrow_available"] is True
    assert "open_spot_forecast_tomorrow_prices" in _background_tasks(harness)
    assert harness.dayahead.async_read.await_count == 2
    harness.reader.read_stromligning_sensor.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.usefixtures("copenhagen_time_zone")
async def test_the_backfill_adds_missing_dayahead_days_and_retrains(
    make: Callable[..., Harness],
) -> None:
    """The training window's missing price days become training data at once."""
    harness = make(price_source="dayahead")
    harness.predictor.price_history = [{"date": "2026-09-23"}]
    harness.dayahead.async_history.return_value = {
        date(2026, 9, 23): [0.3] * 96,
        date(2026, 9, 22): [0.2] * 96,
    }

    with (
        patch("homeassistant.util.dt.now", return_value=NOW),
        patch.object(ForecastUpdater, "refresh_forecast", autospec=True) as refresh,
    ):
        await harness.updater.backfill_history()

    harness.dayahead.async_history.assert_awaited_once_with(
        date(2026, 8, 25), date(2026, 9, 24)
    )
    harness.predictor.record_training_prices.assert_called_once_with(
        [0.2] * 96, "2026-09-22"
    )
    refresh.assert_awaited_once()


@pytest.mark.asyncio
async def test_a_failed_dayahead_backfill_is_logged(
    make: Callable[..., Harness], caplog: pytest.LogCaptureFixture
) -> None:
    harness = make(price_source="dayahead")
    harness.dayahead.async_history.side_effect = RuntimeError("offline")

    await harness.updater.backfill_history()

    assert "Day-ahead price backfill failed: offline" in caplog.text
    harness.predictor.record_training_prices.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.usefixtures("copenhagen_time_zone")
async def test_dayahead_prices_are_pruned_with_and_without_ml(
    make: Callable[..., Harness],
) -> None:
    harness = make(price_source="dayahead")
    with patch("homeassistant.util.dt.now", return_value=NOW):
        await harness.updater.prune_history()
    harness.dayahead.async_prune.assert_awaited_once_with(
        datetime(2026, 8, 23, tzinfo=CPH)
    )

    without_ml = make(ml=False, price_source="dayahead")
    with patch("homeassistant.util.dt.now", return_value=NOW):
        await without_ml.updater.prune_history()
    # Only the margin is kept without the model
    without_ml.dayahead.async_prune.assert_awaited_with(
        datetime(2026, 9, 22, tzinfo=CPH)
    )


# --- Open-Meteo zone weather (#22) ---------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.usefixtures("copenhagen_time_zone")
async def test_run_forecast_refreshes_the_zone_weather(
    make: Callable[..., Harness],
) -> None:
    """Yesterday to the forecast's end is refreshed; today on is attached."""
    harness = make()

    with patch("homeassistant.util.dt.now", return_value=NOW):
        await harness.updater.run_forecast()

    harness.weather.async_update.assert_awaited_once_with(
        datetime(2026, 9, 23, tzinfo=CPH), datetime(2026, 10, 2, tzinfo=CPH)
    )
    assert harness.api_data["weather_data"]["zone_weather"] == [ZONE_ROW]


@pytest.mark.asyncio
async def test_zone_weather_failures_keep_the_forecast_going(
    make: Callable[..., Harness], caplog: pytest.LogCaptureFixture
) -> None:
    harness = make()
    harness.weather.async_update.side_effect = RuntimeError("offline")
    harness.weather.async_load.side_effect = RuntimeError("locked")

    await harness.updater.run_forecast()

    assert "zone_weather" not in harness.api_data["weather_data"]
    assert "Could not update the Open-Meteo weather: offline" in caplog.text
    assert "Could not read the stored Open-Meteo weather: locked" in caplog.text
    harness.predictor.predict.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.usefixtures("copenhagen_time_zone")
async def test_zone_weather_is_pruned_with_the_history(
    make: Callable[..., Harness],
) -> None:
    harness = make()

    with patch("homeassistant.util.dt.now", return_value=NOW):
        await harness.updater.prune_history()

    harness.weather.async_prune.assert_awaited_once_with(
        datetime(2026, 8, 23, tzinfo=CPH)
    )


def test_zone_weather_needs_the_model(make: Callable[..., Harness]) -> None:
    assert make(ml=False).updater.weather is None


# --- Attribution (#41) ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_active_sources_are_recorded_for_attribution(
    make: Callable[..., Harness],
) -> None:
    harness = make(price_source="dayahead")
    harness.dayahead.license_info = "CC BY 4.0 from SMARD.de"

    await harness.updater.async_read_prices()

    assert harness.api_data["price_license"] == "CC BY 4.0 from SMARD.de"
    assert harness.api_data["zone_weather"] is True
    # No ENTSO-E key in the harness's settings
    assert harness.api_data["entsoe_fallback"] is False
    assert make(ml=False).api_data["zone_weather"] is False
