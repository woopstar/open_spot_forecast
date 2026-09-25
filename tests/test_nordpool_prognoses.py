"""Nordpool prognoses fetched for the forecast and stored for training (#54).

``fetch_nordpool_prognoses`` moved from ``__init__.py`` to
``api/nordpool_prognoses.py`` unchanged.
"""

from collections.abc import Iterator
from datetime import date, datetime
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest

from custom_components.open_spot_forecast.api import fetch_nordpool_prognoses

MODULE = "custom_components.open_spot_forecast.api.nordpool_prognoses"
TODAY = date(2026, 9, 24)
TS = "2026-09-24T10:00:00Z"
TOMORROW_TS = "2026-09-25T10:00:00Z"


def _production(ts: str, solar: float) -> dict[str, Any]:
    return {
        "deliveryStart": ts,
        "solar": solar,
        "wind_offshore": 100.0,
        "wind_onshore": 200.0,
    }


@pytest.fixture
def clock() -> Iterator[Mock]:
    """Freeze the module's local clock; set ``clock.hour`` to move it."""
    fake_date = Mock()
    fake_date.today.return_value = TODAY
    fake_datetime = Mock()
    fake_datetime.hour = 14
    fake_datetime.now.side_effect = lambda: datetime(2026, 9, 24, fake_datetime.hour)
    with (
        patch(f"{MODULE}.date", fake_date),
        patch(f"{MODULE}.datetime", fake_datetime),
    ):
        yield fake_datetime


@pytest.fixture
def api() -> Iterator[tuple[AsyncMock, AsyncMock]]:
    """Patch both Nordpool endpoints; results are keyed by the requested day."""
    consumption = AsyncMock(
        side_effect=lambda day, _region: (
            ({TS: 4000.0}, "u1") if day == TODAY else ({TOMORROW_TS: 4100.0}, "u1")
        )
    )
    production = AsyncMock(
        side_effect=lambda day, _region: (
            ([_production(TS, 50.0)], "p1")
            if day == TODAY
            else ([_production(TOMORROW_TS, 60.0)], "p1")
        )
    )
    with (
        patch(f"{MODULE}.fetch_consumption_prognosis", consumption),
        patch(f"{MODULE}.fetch_production_prognosis", production),
    ):
        yield consumption, production


@pytest.mark.asyncio
async def test_fetches_today_and_tomorrow_and_combines_rows(
    clock: Mock, api: tuple[AsyncMock, AsyncMock]
) -> None:
    consumption, _production_api = api
    weather: dict[str, Any] = {}

    entries = await fetch_nordpool_prognoses("DK1", weather)

    assert [call.args for call in consumption.call_args_list] == [
        (TODAY, "DK1"),
        (date(2026, 9, 25), "DK1"),
    ]
    assert entries == [
        {
            "timestamp": TS,
            "consumption": 4000.0,
            "solar": 50.0,
            "wind_offshore": 100.0,
            "wind_onshore": 200.0,
        },
        {
            "timestamp": TOMORROW_TS,
            "consumption": 4100.0,
            "solar": 60.0,
            "wind_offshore": 100.0,
            "wind_onshore": 200.0,
        },
    ]
    assert weather["consumption_prognosis"] == {TS: 4000.0, TOMORROW_TS: 4100.0}
    assert len(weather["production_prognosis"]) == 2


@pytest.mark.asyncio
async def test_skips_tomorrow_before_13_with_a_cache(
    clock: Mock, api: tuple[AsyncMock, AsyncMock]
) -> None:
    consumption, _production_api = api
    clock.hour = 9
    api_data: dict[str, Any] = {}

    entries = await fetch_nordpool_prognoses("DK1", {}, api_data)

    assert [call.args[0] for call in consumption.call_args_list] == [TODAY]
    assert [entry["timestamp"] for entry in entries] == [TS]
    assert set(api_data["_nordpool_cache"]) == {"consumption", "production"}


@pytest.mark.asyncio
async def test_reuses_cached_data_while_updated_at_is_unchanged(
    clock: Mock, api: tuple[AsyncMock, AsyncMock]
) -> None:
    clock.hour = 9
    api_data: dict[str, Any] = {
        "_nordpool_cache": {
            "consumption": {"2026-09-24": {"updated_at": "u1", "data": {TS: 1.0}}},
            "production": {
                "2026-09-24": {"updated_at": "p1", "data": [_production(TS, 5.0)]}
            },
        }
    }

    entries = await fetch_nordpool_prognoses("DK1", {}, api_data)

    assert entries[0]["consumption"] == pytest.approx(1.0)
    assert entries[0]["solar"] == pytest.approx(5.0)


@pytest.mark.asyncio
async def test_replaces_cached_data_when_nordpool_updates_it(
    clock: Mock, api: tuple[AsyncMock, AsyncMock]
) -> None:
    clock.hour = 9
    cache: dict[str, Any] = {
        "consumption": {"2026-09-24": {"updated_at": "old", "data": {TS: 1.0}}},
        "production": {
            "2026-09-24": {"updated_at": "old", "data": [_production(TS, 5.0)]}
        },
    }

    entries = await fetch_nordpool_prognoses("DK1", {}, {"_nordpool_cache": cache})

    assert entries[0]["consumption"] == pytest.approx(4000.0)
    assert entries[0]["solar"] == pytest.approx(50.0)
    assert cache["consumption"]["2026-09-24"]["updated_at"] == "u1"
    assert cache["production"]["2026-09-24"]["updated_at"] == "p1"


@pytest.mark.asyncio
async def test_no_data_leaves_the_weather_untouched(clock: Mock) -> None:
    weather: dict[str, Any] = {}
    empty = AsyncMock(return_value=(None, None))
    with (
        patch(f"{MODULE}.fetch_consumption_prognosis", empty),
        patch(f"{MODULE}.fetch_production_prognosis", empty),
    ):
        entries = await fetch_nordpool_prognoses("DK1", weather)

    assert entries == []
    assert weather == {}
