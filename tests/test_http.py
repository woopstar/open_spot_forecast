"""The shared HTTP GET with retries (#27)."""

from collections.abc import Iterator
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import aiohttp
import pytest

from custom_components.open_spot_forecast.api.http import (
    MAX_RETRY_AFTER,
    HttpResponse,
    async_get,
)

SLEEP = "custom_components.open_spot_forecast.api.http.asyncio.sleep"
SECRET = "s3cret-token"


def _response(
    status: int,
    text: str = "ok",
    headers: dict | None = None,
    body: bytes | None = None,
    encoding: str | Exception = "utf-8",
) -> MagicMock:
    resp = MagicMock()
    resp.status = status
    resp.headers = headers or {}
    resp.read = AsyncMock(return_value=text.encode() if body is None else body)
    resp.get_encoding = Mock(side_effect=[encoding])
    resp.__aenter__ = AsyncMock(return_value=resp)
    resp.__aexit__ = AsyncMock(return_value=False)
    return resp


def _session(*answers: MagicMock | Exception) -> MagicMock:
    session = MagicMock()
    session.get = MagicMock(side_effect=list(answers))
    return session


@pytest.fixture
def sleep() -> Iterator[AsyncMock]:
    with patch(SLEEP, new=AsyncMock()) as mock:
        yield mock


async def _get(session: MagicMock) -> HttpResponse | None:
    return await async_get(
        session, "https://example.test/api", "Test", params={"token": SECRET}
    )


@pytest.mark.asyncio
async def test_a_response_is_returned_with_its_text(sleep: AsyncMock) -> None:
    session = _session(_response(200, "body"))

    assert await _get(session) == HttpResponse(200, "body", b"body")
    assert session.get.call_args.kwargs["params"] == {"token": SECRET}
    sleep.assert_not_awaited()


@pytest.mark.asyncio
async def test_rate_limits_wait_for_retry_after(sleep: AsyncMock) -> None:
    session = _session(
        _response(429, headers={"Retry-After": "7"}),
        _response(429, headers={"Retry-After": "9999"}),
        _response(503, headers={"Retry-After": "soon"}),
        _response(200),
    )

    assert await _get(session) == HttpResponse(200, "ok", b"ok")
    assert [call.args[0] for call in sleep.await_args_list] == [
        7.0,
        MAX_RETRY_AFTER,
        4.0,
    ]


@pytest.mark.asyncio
async def test_the_last_transient_status_is_returned(sleep: AsyncMock) -> None:
    session = _session(*(_response(503) for _ in range(4)))

    response = await _get(session)

    assert response is not None
    assert response.status == 503
    assert sleep.await_count == 3


@pytest.mark.asyncio
async def test_other_statuses_are_not_retried(sleep: AsyncMock) -> None:
    assert await _get(_session(_response(401, "denied"))) == HttpResponse(
        401, "denied", b"denied"
    )
    sleep.assert_not_awaited()


@pytest.mark.asyncio
async def test_network_errors_are_retried_and_never_log_the_url(
    sleep: AsyncMock, caplog: pytest.LogCaptureFixture
) -> None:
    error = aiohttp.ClientError(f"https://example.test/api?token={SECRET}")
    assert await _get(_session(TimeoutError(), _response(200))) == HttpResponse(
        200, "ok", b"ok"
    )
    assert await _get(_session(*(error for _ in range(4)))) is None

    assert "Test API request failed: ClientError" in caplog.text
    assert SECRET not in caplog.text
    assert "example.test" not in caplog.text


@pytest.mark.asyncio
async def test_a_binary_body_is_kept_as_bytes_and_decoded_without_an_error(
    sleep: AsyncMock,
) -> None:
    # A ZIP (ENTSO-E's outage documents, #138): no charset, undecodable bytes
    zipped = b"PK\x03\x04\xff\xfe"
    assert await _get(
        _session(_response(200, body=zipped, encoding=RuntimeError("no charset")))
    ) == HttpResponse(200, zipped.decode("utf-8", errors="replace"), zipped)
    sleep.assert_not_awaited()
