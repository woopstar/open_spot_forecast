"""Nordpool data portal API client for consumption and production prognoses."""

import asyncio
import logging
from datetime import date
from typing import cast

import aiohttp
from aiohttp import ClientTimeout

_LOGGER = logging.getLogger(__name__)

NORDPOOL_API = "https://dataportal-api.nordpoolgroup.com/api"

# Nordpool's dataportal API sits behind Cloudflare bot protection. A browser-like
# User-Agent avoids being flagged as a script, which surfaces as HTTP 401/403.
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
)

# Status codes that indicate a transient failure worth retrying with backoff.
# 401/403 are included because Cloudflare bot protection returns them
# intermittently for otherwise-valid requests.
_RETRYABLE_STATUS = frozenset({401, 403, 429, 500, 502, 503, 504})

# Maximum retry attempts (in addition to the initial request).
_MAX_RETRIES = 3

# Initial backoff delay in seconds; doubles on each retry (1s, 2s, 4s).
_RETRY_BASE_DELAY = 1.0


async def _get_json(url: str, label: str) -> dict | None:
    """GET a Nordpool endpoint, retrying transient failures with backoff.

    Args:
        url: Full API URL to request.
        label: Human-readable endpoint name used in log messages.

    Returns:
        Parsed JSON body on success, or None if all attempts fail.
    """
    headers = {"User-Agent": USER_AGENT}
    for attempt in range(_MAX_RETRIES + 1):
        try:
            async with aiohttp.ClientSession(headers=headers) as session:  # noqa: SIM117
                async with session.get(url, timeout=ClientTimeout(total=30)) as resp:
                    if resp.status == 200:
                        return cast(dict, await resp.json())
                    if resp.status in _RETRYABLE_STATUS and attempt < _MAX_RETRIES:
                        delay = _RETRY_BASE_DELAY * (2**attempt)
                        _LOGGER.warning(
                            "%s API returned %d (attempt %d/%d), retrying in %.1fs",
                            label,
                            resp.status,
                            attempt + 1,
                            _MAX_RETRIES + 1,
                            delay,
                        )
                        await asyncio.sleep(delay)
                        continue
                    _LOGGER.warning("%s API returned %d", label, resp.status)
                    return None
        except Exception as err:
            if attempt < _MAX_RETRIES:
                delay = _RETRY_BASE_DELAY * (2**attempt)
                _LOGGER.warning(
                    "%s API request failed (%s), retrying in %.1fs",
                    label,
                    err,
                    delay,
                )
                await asyncio.sleep(delay)
                continue
            _LOGGER.warning("Failed to fetch %s: %s", label, err)
            return None
    return None


async def fetch_consumption_prognosis(
    target_date: date, area: str = "DK1"
) -> dict | None:
    """Fetch Nordpool consumption prognosis for a given date and area.

    Returns hourly consumption forecast in MW, keyed by ISO timestamp.
    Returns None on failure.
    """
    url = (
        f"{NORDPOOL_API}/ConsumptionPrognoses"
        f"?date={target_date.isoformat()}"
        f"&deliveryAreas={area}&locations="
    )
    data = await _get_json(url, "Consumption prognosis")
    if data is None:
        return None

    entries = data.get("multiAreaEntries", [])
    result: dict = {}
    for entry in entries:
        start = entry.get("deliveryStart", "")
        area_data = entry.get("entryPerArea", {}).get(area, {})
        volume = area_data.get("volume")
        if start and volume is not None:
            result[start] = float(volume)

    _LOGGER.info(
        "Fetched consumption prognosis for %s: %d hours",
        target_date,
        len(result),
    )
    return result if result else None


async def fetch_production_prognosis(
    target_date: date, area: str = "DK1"
) -> list[dict] | None:
    """Fetch Nordpool production data prognosis for a given date and area.

    Returns 15-minute resolution production forecasts in MW.
    Each entry: {deliveryStart, solar, wind_offshore, wind_onshore, total}
    Returns None on failure.
    """
    url = (
        f"{NORDPOOL_API}/ProductionDataPrognoses"
        f"?date={target_date.isoformat()}"
        f"&deliveryArea={area}&location="
    )
    data = await _get_json(url, "Production prognosis")
    if data is None:
        return None

    content = data.get("content", [])
    result: list[dict] = []
    for entry in content:
        start = entry.get("deliveryStart")
        if not start:
            continue
        forecast = entry.get("forecastByType", {})
        solar_data = forecast.get("Solar", {})
        wind_off = forecast.get("WindOffshore", {})
        wind_on = forecast.get("WindOnshore", {})

        result.append(
            {
                "deliveryStart": start,
                "solar": float(solar_data.get("dayAheadPrognosis", 0)),
                "wind_offshore": float(wind_off.get("dayAheadPrognosis", 0)),
                "wind_onshore": float(wind_on.get("dayAheadPrognosis", 0)),
                "total": float(entry.get("totalDayAheadPrognosis", 0)),
            }
        )

    _LOGGER.info(
        "Fetched production prognosis for %s: %d intervals",
        target_date,
        len(result),
    )
    return result if result else None
