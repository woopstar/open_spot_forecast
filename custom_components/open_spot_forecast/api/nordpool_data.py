"""Nordpool data portal API client for consumption and production prognoses."""

import json
import logging
from datetime import date
from typing import cast

import aiohttp

from ..const import NORDPOOL_API, USER_AGENT
from .http import async_get

_LOGGER = logging.getLogger(__name__)


async def _get_json(url: str, label: str) -> dict | None:
    """GET a Nordpool endpoint (retried by ``async_get``) and decode its JSON.

    Nordpool gets its own session: its Cloudflare protection needs the
    browser-like ``USER_AGENT``.

    Args:
        url: Full API URL to request.
        label: Human-readable endpoint name used in log messages.

    Returns:
        Parsed JSON body on success, or None if the request failed.
    """
    async with aiohttp.ClientSession(headers={"User-Agent": USER_AGENT}) as session:
        response = await async_get(session, url, label)
    if response is None:
        return None
    if response.status != 200:
        _LOGGER.warning("%s API returned %d", label, response.status)
        return None
    try:
        return cast(dict, json.loads(response.text))
    except ValueError:
        _LOGGER.warning("%s API returned invalid JSON", label)
        return None


async def fetch_consumption_prognosis(
    target_date: date, area: str = "DK1"
) -> tuple[dict | None, str | None]:
    """Fetch Nordpool consumption prognosis for a given date and area.

    Returns a tuple of (hourly consumption forecast in MW keyed by ISO
    timestamp, Nordpool's ``updatedAt`` timestamp). Both are None on failure.
    """
    url = (
        f"{NORDPOOL_API}/ConsumptionPrognoses"
        f"?date={target_date.isoformat()}"
        f"&deliveryAreas={area}&locations="
    )
    data = await _get_json(url, "Consumption prognosis")
    if data is None:
        return None, None

    updated_at = data.get("updatedAt")
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
    return (result if result else None), updated_at


async def fetch_production_prognosis(
    target_date: date, area: str = "DK1"
) -> tuple[list[dict] | None, str | None]:
    """Fetch Nordpool production data prognosis for a given date and area.

    Returns a tuple of (list of 15-minute production forecast entries,
    Nordpool's ``updatedAt`` timestamp). Each entry is
    {deliveryStart, solar, wind_offshore, wind_onshore, total}.

    Nordpool publishes the day-ahead total before the per-type breakdown
    (Solar/WindOffshore/WindOnshore). Until that breakdown appears there is
    no per-type data to report, so the list is None even though the total is
    available; the caller uses ``updatedAt`` to detect when the breakdown is
    published later.
    """
    url = (
        f"{NORDPOOL_API}/ProductionDataPrognoses"
        f"?date={target_date.isoformat()}"
        f"&deliveryArea={area}&location="
    )
    data = await _get_json(url, "Production prognosis")
    if data is None:
        return None, None

    updated_at = data.get("updatedAt")
    content = data.get("content", [])
    result: list[dict] = []
    has_breakdown = False
    for entry in content:
        start = entry.get("deliveryStart")
        if not start:
            continue
        forecast = entry.get("forecastByType", {})
        if not forecast:
            continue
        has_breakdown = True
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

    if not has_breakdown:
        _LOGGER.info(
            "Production prognosis for %s has no per-type breakdown yet",
            target_date,
        )
        return None, updated_at

    _LOGGER.info(
        "Fetched production prognosis for %s: %d intervals",
        target_date,
        len(result),
    )
    return (result if result else None), updated_at
