"""Tests for the Nordpool data portal API client."""

from datetime import date
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components.open_spot_forecast.api.nordpool_data import (
    USER_AGENT,
    fetch_consumption_prognosis,
    fetch_production_prognosis,
)


def _response(status: int, payload: dict) -> MagicMock:
    """Build a mock aiohttp response with the given status and JSON body."""
    resp = MagicMock()
    resp.status = status
    resp.json = AsyncMock(return_value=payload)
    resp.__aenter__ = AsyncMock(return_value=resp)
    resp.__aexit__ = AsyncMock(return_value=False)
    return resp


def _session(response: MagicMock) -> MagicMock:
    """Build a mock aiohttp session that returns a single response."""
    session = MagicMock()
    # aiohttp's session.get() returns an async context manager synchronously,
    # so it must be a plain MagicMock (not AsyncMock).
    session.get = MagicMock(return_value=response)
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)
    return session


@pytest.mark.asyncio
async def test_consumption_prognosis_success():
    """A 200 response is parsed into an hourly volume dict."""
    payload = {
        "multiAreaEntries": [
            {
                "deliveryStart": "2026-09-22T00:00:00Z",
                "entryPerArea": {"DK1": {"volume": 2240}},
            },
            {
                "deliveryStart": "2026-09-22T01:00:00Z",
                "entryPerArea": {"DK1": {"volume": 2139}},
            },
        ]
    }
    with patch(
        "custom_components.open_spot_forecast.api.nordpool_data.aiohttp.ClientSession",
        return_value=_session(_response(200, payload)),
    ) as client_session:
        result = await fetch_consumption_prognosis(date(2026, 9, 22), "DK1")

    assert result == {
        "2026-09-22T00:00:00Z": 2240.0,
        "2026-09-22T01:00:00Z": 2139.0,
    }
    # A browser-like User-Agent must be sent to avoid Cloudflare bot detection.
    assert client_session.call_args.kwargs["headers"] == {"User-Agent": USER_AGENT}


@pytest.mark.asyncio
async def test_retry_on_transient_status_then_success():
    """A transient 429 is retried and eventually succeeds."""
    payload = {
        "multiAreaEntries": [
            {
                "deliveryStart": "2026-09-22T00:00:00Z",
                "entryPerArea": {"DK1": {"volume": 2240}},
            }
        ]
    }
    sessions = [
        _session(_response(429, {})),
        _session(_response(200, payload)),
    ]
    with (
        patch(
            "custom_components.open_spot_forecast.api.nordpool_data.aiohttp.ClientSession",
            side_effect=sessions,
        ),
        patch(
            "custom_components.open_spot_forecast.api.nordpool_data.asyncio.sleep",
            new=AsyncMock(),
        ),
    ):
        result = await fetch_consumption_prognosis(date(2026, 9, 22), "DK1")

    assert result == {"2026-09-22T00:00:00Z": 2240.0}


@pytest.mark.asyncio
async def test_persistent_failure_returns_none():
    """A persistent 401 exhausts retries and returns None."""
    sessions = [_session(_response(401, {})) for _ in range(4)]
    with (
        patch(
            "custom_components.open_spot_forecast.api.nordpool_data.aiohttp.ClientSession",
            side_effect=sessions,
        ),
        patch(
            "custom_components.open_spot_forecast.api.nordpool_data.asyncio.sleep",
            new=AsyncMock(),
        ),
    ):
        result = await fetch_consumption_prognosis(date(2026, 9, 22), "DK1")

    assert result is None


@pytest.mark.asyncio
async def test_production_prognosis_success():
    """A successful production response is parsed into interval dicts."""
    payload = {
        "content": [
            {
                "deliveryStart": "2026-09-22T00:00:00Z",
                "forecastByType": {
                    "Solar": {"dayAheadPrognosis": 0},
                    "WindOffshore": {"dayAheadPrognosis": 664.5},
                    "WindOnshore": {"dayAheadPrognosis": 715},
                },
                "totalDayAheadPrognosis": 1753.97,
            }
        ]
    }
    with patch(
        "custom_components.open_spot_forecast.api.nordpool_data.aiohttp.ClientSession",
        return_value=_session(_response(200, payload)),
    ):
        result = await fetch_production_prognosis(date(2026, 9, 22), "DK1")

    assert result == [
        {
            "deliveryStart": "2026-09-22T00:00:00Z",
            "solar": 0.0,
            "wind_offshore": 664.5,
            "wind_onshore": 715.0,
            "total": 1753.97,
        }
    ]
