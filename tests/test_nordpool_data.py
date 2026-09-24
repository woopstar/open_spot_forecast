"""Tests for the Nordpool data portal API client."""

from datetime import date
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components.open_spot_forecast.api.nordpool_data import (
    fetch_consumption_prognosis,
    fetch_production_prognosis,
)
from custom_components.open_spot_forecast.const import USER_AGENT


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
        "updatedAt": "2026-09-22T15:56:32Z",
        "multiAreaEntries": [
            {
                "deliveryStart": "2026-09-22T00:00:00Z",
                "entryPerArea": {"DK1": {"volume": 2240}},
            },
            {
                "deliveryStart": "2026-09-22T01:00:00Z",
                "entryPerArea": {"DK1": {"volume": 2139}},
            },
        ],
    }
    with patch(
        "custom_components.open_spot_forecast.api.nordpool_data.aiohttp.ClientSession",
        return_value=_session(_response(200, payload)),
    ) as client_session:
        result, updated_at = await fetch_consumption_prognosis(date(2026, 9, 22), "DK1")

    assert result == {
        "2026-09-22T00:00:00Z": 2240.0,
        "2026-09-22T01:00:00Z": 2139.0,
    }
    assert updated_at == "2026-09-22T15:56:32Z"
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
        result, _ = await fetch_consumption_prognosis(date(2026, 9, 22), "DK1")

    assert result == {"2026-09-22T00:00:00Z": 2240.0}


@pytest.mark.asyncio
async def test_persistent_failure_returns_none():
    """A persistent transient failure exhausts retries and returns None."""
    sessions = [_session(_response(503, {})) for _ in range(4)]
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
        result, updated_at = await fetch_consumption_prognosis(date(2026, 9, 22), "DK1")

    assert result is None
    assert updated_at is None


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [401, 403])
async def test_auth_error_not_retried(status):
    """An auth/forbidden response returns None immediately, without retrying."""
    with (
        patch(
            "custom_components.open_spot_forecast.api.nordpool_data.aiohttp.ClientSession",
            side_effect=[_session(_response(status, {}))],
        ) as client_session,
        patch(
            "custom_components.open_spot_forecast.api.nordpool_data.asyncio.sleep",
            new=AsyncMock(),
        ) as sleep_mock,
    ):
        result, _ = await fetch_consumption_prognosis(date(2026, 9, 22), "DK1")

    assert result is None
    client_session.assert_called_once()
    sleep_mock.assert_not_called()


@pytest.mark.asyncio
async def test_production_prognosis_success():
    """A successful production response is parsed into interval dicts."""
    payload = {
        "updatedAt": "2026-09-22T16:10:20Z",
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
        ],
    }
    with patch(
        "custom_components.open_spot_forecast.api.nordpool_data.aiohttp.ClientSession",
        return_value=_session(_response(200, payload)),
    ):
        result, updated_at = await fetch_production_prognosis(date(2026, 9, 22), "DK1")

    assert result == [
        {
            "deliveryStart": "2026-09-22T00:00:00Z",
            "solar": 0.0,
            "wind_offshore": 664.5,
            "wind_onshore": 715.0,
            "total": 1753.97,
        }
    ]
    assert updated_at == "2026-09-22T16:10:20Z"


@pytest.mark.asyncio
async def test_production_without_by_type_breakdown_returns_none():
    """A response with no per-type breakdown returns (None, updated_at).

    Nordpool publishes the day-ahead total before the Solar/WindOffshore/
    WindOnshore breakdown. Until that breakdown appears there is no per-type
    data, so the parsed result is None rather than fabricated zero values.
    """
    payload = {
        "updatedAt": "2026-09-22T15:05:05Z",
        "content": [
            {
                "forecastByType": {},
                "totalDayAheadPrognosis": 1026.21,
                "deliveryStart": "2026-09-22T22:00:00Z",
                "deliveryEnd": "2026-09-22T22:15:00Z",
            }
        ],
    }
    with patch(
        "custom_components.open_spot_forecast.api.nordpool_data.aiohttp.ClientSession",
        return_value=_session(_response(200, payload)),
    ):
        result, updated_at = await fetch_production_prognosis(date(2026, 9, 23), "DK1")

    assert result is None
    assert updated_at == "2026-09-22T15:05:05Z"


@pytest.mark.asyncio
async def test_non_retryable_status_returns_none():
    """A non-retryable 404 returns None immediately, without retrying."""
    with patch(
        "custom_components.open_spot_forecast.api.nordpool_data.aiohttp.ClientSession",
        return_value=_session(_response(404, {})),
    ):
        result, updated_at = await fetch_consumption_prognosis(date(2026, 9, 22), "DK1")

    assert result is None
    assert updated_at is None


@pytest.mark.asyncio
async def test_request_exception_retries_then_succeeds():
    """A raised exception is retried with backoff and eventually succeeds."""
    payload = {
        "multiAreaEntries": [
            {
                "deliveryStart": "2026-09-22T00:00:00Z",
                "entryPerArea": {"DK1": {"volume": 2240}},
            }
        ]
    }
    with (
        patch(
            "custom_components.open_spot_forecast.api.nordpool_data.aiohttp.ClientSession",
            side_effect=[Exception("boom"), _session(_response(200, payload))],
        ),
        patch(
            "custom_components.open_spot_forecast.api.nordpool_data.asyncio.sleep",
            new=AsyncMock(),
        ),
    ):
        result, _ = await fetch_consumption_prognosis(date(2026, 9, 22), "DK1")

    assert result == {"2026-09-22T00:00:00Z": 2240.0}


@pytest.mark.asyncio
async def test_request_exception_exhausts_retries():
    """A persistent exception exhausts retries and returns None."""
    with (
        patch(
            "custom_components.open_spot_forecast.api.nordpool_data.aiohttp.ClientSession",
            side_effect=[Exception("boom") for _ in range(4)],
        ),
        patch(
            "custom_components.open_spot_forecast.api.nordpool_data.asyncio.sleep",
            new=AsyncMock(),
        ),
    ):
        result, _ = await fetch_consumption_prognosis(date(2026, 9, 22), "DK1")

    assert result is None


@pytest.mark.asyncio
async def test_consumption_empty_result_returns_none():
    """A 200 with no volume entries returns None (not an empty dict)."""
    with patch(
        "custom_components.open_spot_forecast.api.nordpool_data.aiohttp.ClientSession",
        return_value=_session(_response(200, {"multiAreaEntries": []})),
    ):
        result, _ = await fetch_consumption_prognosis(date(2026, 9, 22), "DK1")

    assert result is None


@pytest.mark.asyncio
async def test_consumption_skips_entries_missing_start_or_volume():
    """Entries missing a start timestamp or volume are skipped."""
    payload = {
        "multiAreaEntries": [
            {
                "deliveryStart": "",
                "entryPerArea": {"DK1": {"volume": 100}},
            },
            {
                "deliveryStart": "2026-09-22T01:00:00Z",
                "entryPerArea": {"DK1": {}},
            },
        ]
    }
    with patch(
        "custom_components.open_spot_forecast.api.nordpool_data.aiohttp.ClientSession",
        return_value=_session(_response(200, payload)),
    ):
        result, _ = await fetch_consumption_prognosis(date(2026, 9, 22), "DK1")

    assert result is None


@pytest.mark.asyncio
async def test_production_empty_content_returns_none():
    """A 200 with no production content returns None."""
    with patch(
        "custom_components.open_spot_forecast.api.nordpool_data.aiohttp.ClientSession",
        return_value=_session(_response(200, {"content": []})),
    ):
        result, _ = await fetch_production_prognosis(date(2026, 9, 22), "DK1")

    assert result is None


@pytest.mark.asyncio
async def test_production_skips_entry_missing_start():
    """A production entry with no deliveryStart is skipped."""
    payload = {"content": [{"deliveryStart": None, "totalDayAheadPrognosis": 10}]}
    with patch(
        "custom_components.open_spot_forecast.api.nordpool_data.aiohttp.ClientSession",
        return_value=_session(_response(200, payload)),
    ):
        result, _ = await fetch_production_prognosis(date(2026, 9, 22), "DK1")

    assert result is None


@pytest.mark.asyncio
async def test_production_uses_defaults_when_prognosis_missing():
    """A missing per-type prognosis falls back to 0.0 for that type."""
    payload = {
        "content": [
            {
                "deliveryStart": "2026-09-22T00:00:00Z",
                "forecastByType": {"WindOffshore": {"dayAheadPrognosis": 715}},
                "totalDayAheadPrognosis": 1753.97,
            }
        ]
    }
    with patch(
        "custom_components.open_spot_forecast.api.nordpool_data.aiohttp.ClientSession",
        return_value=_session(_response(200, payload)),
    ):
        result, _ = await fetch_production_prognosis(date(2026, 9, 22), "DK1")

    assert result == [
        {
            "deliveryStart": "2026-09-22T00:00:00Z",
            "solar": 0.0,
            "wind_offshore": 715.0,
            "wind_onshore": 0.0,
            "total": 1753.97,
        }
    ]
