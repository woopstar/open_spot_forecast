"""HTTP GET with retries and backoff, shared by every API client.

Transient failures (``RETRYABLE_STATUS`` and network errors) are retried up
to ``MAX_RETRIES`` times with exponential backoff; a 429's ``Retry-After``
(seconds) is honoured, up to ``MAX_RETRY_AFTER``. Other statuses, including
401/403, are returned at once. The URL and query parameters are never logged:
they can hold an API key.
"""

import asyncio
import logging
from collections.abc import Mapping
from dataclasses import dataclass

import aiohttp
from aiohttp import ClientTimeout

from ..const import MAX_RETRIES, RETRY_BASE_DELAY, RETRYABLE_STATUS

_LOGGER = logging.getLogger(__name__)

# Longest Retry-After wait honoured, in seconds
MAX_RETRY_AFTER = 60.0


@dataclass(frozen=True, slots=True)
class HttpResponse:
    """The final response of a request: status and body text."""

    status: int
    text: str


def _retry_delay(attempt: int, retry_after: str | None) -> float:
    """Return the wait before the next attempt (Retry-After seconds if given)."""
    try:
        return min(max(float(retry_after or ""), 0.0), MAX_RETRY_AFTER)
    except ValueError:
        return RETRY_BASE_DELAY * float(2**attempt)


async def async_get(
    session: aiohttp.ClientSession,
    url: str,
    label: str,
    *,
    params: Mapping[str, str] | None = None,
    headers: Mapping[str, str] | None = None,
    timeout: float = 30,
) -> HttpResponse | None:
    """GET a URL, retrying transient failures.

    Args:
        session: The aiohttp session (Home Assistant's shared one if possible).
        url: The URL, without secrets (those go in ``params``).
        label: Endpoint name for log messages.
        params: Query parameters.
        headers: Extra request headers.
        timeout: Total timeout per attempt, in seconds.

    Returns:
        The final response (any status), or None if every attempt failed
        without one.
    """
    for attempt in range(MAX_RETRIES + 1):
        retry = attempt < MAX_RETRIES
        try:
            async with session.get(
                url,
                params=params,
                headers=headers,
                timeout=ClientTimeout(total=timeout),
            ) as resp:
                if resp.status in RETRYABLE_STATUS and retry:
                    delay = _retry_delay(attempt, resp.headers.get("Retry-After"))
                    _LOGGER.warning(
                        "%s API returned %d (attempt %d/%d), retrying in %.1fs",
                        label,
                        resp.status,
                        attempt + 1,
                        MAX_RETRIES + 1,
                        delay,
                    )
                    await asyncio.sleep(delay)
                    continue
                return HttpResponse(resp.status, await resp.text())
        except (aiohttp.ClientError, TimeoutError) as err:
            if not retry:
                _LOGGER.warning("%s API request failed: %s", label, type(err).__name__)
                return None
            delay = RETRY_BASE_DELAY * float(2**attempt)
            _LOGGER.warning(
                "%s API request failed (%s), retrying in %.1fs",
                label,
                type(err).__name__,
                delay,
            )
            await asyncio.sleep(delay)
    return None
