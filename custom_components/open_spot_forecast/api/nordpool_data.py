"""Nordpool data portal API client for consumption and production prognoses."""

import logging
from datetime import date

import aiohttp
from aiohttp import ClientTimeout

_LOGGER = logging.getLogger(__name__)

NORDPOOL_API = "https://dataportal-api.nordpoolgroup.com/api"


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
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=ClientTimeout(total=30)) as resp:
                if resp.status != 200:
                    _LOGGER.warning(
                        "Consumption prognosis API returned %d", resp.status
                    )
                    return None
                data = await resp.json()

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

    except Exception as err:
        _LOGGER.warning("Failed to fetch consumption prognosis: %s", err)
        return None


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
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=ClientTimeout(total=30)) as resp:
                if resp.status != 200:
                    _LOGGER.warning("Production prognosis API returned %d", resp.status)
                    return None
                data = await resp.json()

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

    except Exception as err:
        _LOGGER.warning("Failed to fetch production prognosis: %s", err)
        return None
