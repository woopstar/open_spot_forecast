"""Tests for polling tomorrow's prices until the full day is published (#20)."""

from collections.abc import Callable, Iterator
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock, patch
from zoneinfo import ZoneInfo

import pytest

from homeassistant.util import dt as dt_util

from custom_components.open_spot_forecast import async_setup_entry, async_unload_entry
from custom_components.open_spot_forecast.const import (
    CONF_ENABLE_ML_PREDICTION,
    CONF_REGION,
    CONF_STROMLIGNING_SENSOR,
    CONF_TEMPERATURE_SENSOR,
    DOMAIN,
)
from custom_components.open_spot_forecast.time_slots import (
    slots_in_local_day,
    tomorrow_prices_complete,
)
from custom_components.open_spot_forecast.tomorrow_prices import (
    TomorrowPriceChecker,
    next_tomorrow_check,
)

CPH = ZoneInfo("Europe/Copenhagen")
MODULE = "custom_components.open_spot_forecast.tomorrow_prices"

pytestmark = pytest.mark.usefixtures("copenhagen_time_zone")


def _cph(year: int, month: int, day: int, hour: int = 0, minute: int = 0) -> datetime:
    """Return a Europe/Copenhagen wall-clock time as a UTC datetime."""
    return datetime(year, month, day, hour, minute, tzinfo=CPH).astimezone(UTC)


# --- Completeness ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("day", "slots"),
    [
        (date(2026, 9, 25), 96),
        (date(2026, 3, 29), 92),  # spring forward: 23 hours
        (date(2026, 10, 25), 100),  # fall back: 25 hours
    ],
)
def test_slots_in_local_day(day: date, slots: int) -> None:
    """A local day has 96 slots, 92 or 100 on a DST-change day."""
    assert slots_in_local_day(day) == slots


@pytest.mark.parametrize(
    ("now", "count", "complete"),
    [
        (_cph(2026, 9, 24, 13, 30), 96, True),
        (_cph(2026, 9, 24, 13, 30), 95, False),
        (_cph(2026, 9, 24, 13, 30), 23, False),  # the old ">= 23" threshold
        (_cph(2026, 3, 28, 13, 30), 92, True),
        (_cph(2026, 3, 28, 13, 30), 91, False),
        (_cph(2026, 10, 24, 13, 30), 96, False),
        (_cph(2026, 10, 24, 13, 30), 100, True),
    ],
)
def test_tomorrow_prices_complete(now: datetime, count: int, complete: bool) -> None:
    """Tomorrow is complete only when every slot of the next local day is known."""
    assert tomorrow_prices_complete([1.0] * count, now) is complete


# --- Schedule -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("now", "complete", "expected"),
    [
        # Before 13:00: first check at 13:00
        (_cph(2026, 9, 24, 9, 0), False, _cph(2026, 9, 24, 13, 0)),
        # Published late: check again every 5 minutes
        (_cph(2026, 9, 24, 13, 2), False, _cph(2026, 9, 24, 13, 7)),
        (_cph(2026, 9, 24, 17, 58), False, _cph(2026, 9, 24, 18, 3)),
        # Cutoff reached: give up until tomorrow
        (_cph(2026, 9, 24, 18, 0), False, _cph(2026, 9, 25, 13, 0)),
        # Complete: nothing to do until tomorrow's publication
        (_cph(2026, 9, 24, 13, 5), True, _cph(2026, 9, 25, 13, 0)),
        # Across the fall-back change, 13:00 local is 12:00 UTC the next day
        (_cph(2026, 10, 24, 14, 0), True, datetime(2026, 10, 25, 12, 0, tzinfo=UTC)),
    ],
)
def test_next_tomorrow_check(now: datetime, complete: bool, expected: datetime) -> None:
    """Checks start at 13:00 local, repeat every 5 min, and stop when complete."""
    assert next_tomorrow_check(now, complete) == expected


# --- Checker --------------------------------------------------------------------


class _Scheduler:
    """Stands in for async_track_point_in_utc_time and records each schedule."""

    def __init__(self) -> None:
        self.times: list[datetime] = []
        self.unsubs: list[Mock] = []
        self.action: Callable[[datetime], Any] | None = None

    def __call__(self, _hass: Any, action: Any, when: datetime) -> Mock:
        self.times.append(when)
        self.action = action
        self.unsubs.append(Mock())
        return self.unsubs[-1]


@pytest.fixture
def scheduler() -> Iterator[_Scheduler]:
    """Patch the scheduler and remove jitter."""
    recorder = _Scheduler()
    with (
        patch(f"{MODULE}.async_track_point_in_utc_time", new=recorder),
        patch(f"{MODULE}.randint", return_value=0),
    ):
        yield recorder


async def _run_at(checker: TomorrowPriceChecker, moment: datetime) -> None:
    """Run the pending check as if the clock were at ``moment``."""
    with patch("homeassistant.util.dt.utcnow", return_value=moment):
        await checker._run(moment)


@pytest.mark.asyncio
async def test_late_publication_is_polled_until_complete(
    scheduler: _Scheduler,
) -> None:
    """Checks repeat every 5 minutes until complete, then wait for tomorrow."""
    check = AsyncMock(side_effect=[False, False, True])
    checker = TomorrowPriceChecker(Mock(), check)

    with patch("homeassistant.util.dt.utcnow", return_value=_cph(2026, 9, 24, 9)):
        checker.schedule(False)
    for minute in (0, 5, 10):
        await _run_at(checker, _cph(2026, 9, 24, 13, minute))

    assert check.await_count == 3
    assert scheduler.times == [
        _cph(2026, 9, 24, 13, 0),
        _cph(2026, 9, 24, 13, 5),
        _cph(2026, 9, 24, 13, 10),
        _cph(2026, 9, 25, 13, 0),
    ]


@pytest.mark.asyncio
async def test_cutoff_warns_and_waits_for_tomorrow(
    scheduler: _Scheduler, caplog: pytest.LogCaptureFixture
) -> None:
    """Still incomplete at 18:00: warn and check again the next day."""
    checker = TomorrowPriceChecker(Mock(), AsyncMock(return_value=False))

    await _run_at(checker, _cph(2026, 9, 24, 18, 1))

    assert scheduler.times == [_cph(2026, 9, 25, 13, 0)]
    assert "still incomplete after 18:00" in caplog.text


@pytest.mark.asyncio
async def test_failed_check_keeps_polling(
    scheduler: _Scheduler, caplog: pytest.LogCaptureFixture
) -> None:
    """An error in one check is logged and the next check is still scheduled."""
    checker = TomorrowPriceChecker(Mock(), AsyncMock(side_effect=RuntimeError))

    await _run_at(checker, _cph(2026, 9, 24, 14, 0))

    assert scheduler.times == [_cph(2026, 9, 24, 14, 5)]
    assert "Error checking for tomorrow's prices" in caplog.text


def test_schedule_replaces_and_cancel_removes_the_pending_check(
    scheduler: _Scheduler,
) -> None:
    """Only one check is ever pending, and cancel removes it."""
    checker = TomorrowPriceChecker(Mock(), AsyncMock())

    checker.schedule(False)
    checker.schedule(False)
    scheduler.unsubs[0].assert_called_once()

    checker.cancel()
    checker.cancel()
    scheduler.unsubs[1].assert_called_once()


# --- Integration ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_setup_polls_until_tomorrow_is_complete_and_cancels_on_unload(
    tmp_path: Path, scheduler: _Scheduler
) -> None:
    """End to end: partial prices are not "available"; completion refreshes once."""
    tomorrow = dt_util.now().date() + timedelta(days=1)
    full_day = [2.0] * slots_in_local_day(tomorrow)
    today = [1.0] * 96

    def reading(tomorrow_prices: list[float]) -> dict[str, Any]:
        return {
            "today": today,
            "tomorrow": tomorrow_prices,
            "raw_today": [],
            "raw_tomorrow": [],
        }

    async def run_inline(func: Callable[..., Any], *args: Any) -> Any:
        return func(*args)

    hass = Mock()
    hass.data = {}
    hass.config.path.return_value = str(tmp_path / ".storage")
    hass.async_add_executor_job = run_inline
    hass.config_entries.async_forward_entry_setups = AsyncMock()
    hass.config_entries.async_unload_platforms = AsyncMock(return_value=True)
    entry = MagicMock()
    entry.entry_id = "test"
    entry.options = {}
    entry.data = {
        CONF_REGION: "DK1",
        CONF_ENABLE_ML_PREDICTION: True,
        CONF_STROMLIGNING_SENSOR: "sensor.stromligning_current_price_vat",
        CONF_TEMPERATURE_SENSOR: "sensor.outdoor_temperature",
    }
    reader = Mock()
    reader.read_stromligning_sensor.return_value = reading(full_day[:23])
    reader.read_weather_sensors.return_value = {"temperature": 12.0}
    ml_predictor = Mock()
    ml_predictor._load_learning_data = AsyncMock()
    ml_predictor.save_learning_data = AsyncMock()
    ml_predictor.predictions = []

    module = "custom_components.open_spot_forecast"
    with (
        patch(f"{module}.async_get_integration", new=AsyncMock()),
        patch(f"{module}.SensorReader", return_value=reader),
        patch(f"{module}.SpotPricePredictor", return_value=ml_predictor),
        patch(f"{module}._fetch_nordpool_prognoses", new=AsyncMock(return_value=[])),
        patch(f"{module}.async_track_time_change", return_value=Mock()),
        patch(f"{module}.async_dispatcher_send"),
    ):
        assert await async_setup_entry(hass, entry) is True
        api_data = hass.data[DOMAIN]["test"]
        # 23 of tomorrow's slots used to count as available
        assert api_data["tomorrow_available"] is False
        assert len(scheduler.times) == 1
        assert scheduler.action is not None

        # Still partial at the next check: no refresh
        await scheduler.action(dt_util.utcnow())
        entry.async_create_background_task.assert_not_called()

        # Published in full: available, and the forecast refreshes once
        reader.read_stromligning_sensor.return_value = reading(full_day)
        await scheduler.action(dt_util.utcnow())
        assert api_data["tomorrow_available"] is True
        entry.async_create_background_task.assert_called_once()
        _hass_arg, refresh, name = entry.async_create_background_task.call_args.args
        assert name == "open_spot_forecast_tomorrow_prices"
        await refresh
        assert ml_predictor.predict.call_args.args[1] == today + full_day

        # Complete: the next check is at 13:00 local tomorrow, no second refresh
        await scheduler.action(dt_util.utcnow())
        entry.async_create_background_task.assert_called_once()
        assert scheduler.times[-1] == _cph(
            tomorrow.year, tomorrow.month, tomorrow.day, 13
        )

        # Unload cancels the pending check
        assert await async_unload_entry(hass, entry) is True
        scheduler.unsubs[-1].assert_called_once()
