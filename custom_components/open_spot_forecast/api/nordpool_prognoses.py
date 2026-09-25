"""Fetch and cache Nordpool's consumption and production prognoses.

Combines the two Nordpool endpoints (``nordpool_data.py``) into the prognosis
data the forecast uses: attached to the weather data for prediction, and as
rows for the ``nordpool_prognoses`` table the model trains on.
"""

import logging
from datetime import date, datetime, timedelta

from .nordpool_data import fetch_consumption_prognosis, fetch_production_prognosis

_LOGGER = logging.getLogger(__name__)


async def fetch_nordpool_prognoses(
    region: str, weather_data: dict, api_data: dict | None = None
) -> list[dict]:
    """Fetch Nordpool consumption and production prognoses (with caching).

    Attaches 'consumption_prognosis' and 'production_prognosis' to
    weather_data if the API calls succeed. Also returns a list of
    entries suitable for storage in SQLite.

    Freshness is keyed on Nordpool's ``updatedAt`` timestamp rather than the
    wall-clock date. Day-ahead data is immutable once published, but the
    production per-type breakdown (Solar/WindOffshore/WindOnshore) is
    published later than the total, which surfaces as a later ``updatedAt``.
    We always fetch (calls are infrequent — startup + 6-hourly) and compare
    ``updatedAt`` so we pick up that late breakdown without guessing from the
    hour of day.

    Args:
        region: Price region (e.g. "DK1")
        weather_data: Dict to attach fetched data to
        api_data: Optional integration data dict for caching

    Returns:
        List of dicts with keys: timestamp, consumption, solar,
        wind_offshore, wind_onshore
    """
    today = date.today()
    tomorrow = today + timedelta(days=1)
    now = datetime.now()

    cache = api_data.get("_nordpool_cache", {}) if api_data is not None else {}
    consumption_cache = cache.setdefault("consumption", {})
    production_cache = cache.setdefault("production", {})

    # Always fetch today. Tomorrow's day-ahead total is published ~13:00 CET,
    # but the per-type breakdown arrives later; fetching before 13:00 only
    # returns empty data, so skip it then.
    dates_to_fetch = [today]
    if now.hour >= 13 or api_data is None:
        dates_to_fetch.append(tomorrow)

    stored_entries: list[dict] = []
    for target in dates_to_fetch:
        date_key = target.isoformat()

        consumption, cons_updated = await fetch_consumption_prognosis(target, region)
        production, prod_updated = await fetch_production_prognosis(target, region)

        # --- Consumption ---
        if consumption:
            cached = consumption_cache.get(date_key)
            if cached and cons_updated == cached.get("updated_at"):
                # Unchanged since the last fetch — reuse what we already parsed.
                consumption = cached.get("data") or {}
            else:
                consumption_cache[date_key] = {
                    "updated_at": cons_updated,
                    "data": consumption,
                }

            if "consumption_prognosis" not in weather_data:
                weather_data["consumption_prognosis"] = {}
            weather_data["consumption_prognosis"].update(consumption)

            # Build storage entries combining consumption + production
            for ts, cons in consumption.items():
                stored_entries.append(
                    {
                        "timestamp": ts,
                        "consumption": cons,
                        "solar": None,
                        "wind_offshore": None,
                        "wind_onshore": None,
                    }
                )

        # --- Production ---
        if production:
            cached = production_cache.get(date_key)
            if cached and prod_updated == cached.get("updated_at"):
                production = cached.get("data") or []
            else:
                production_cache[date_key] = {
                    "updated_at": prod_updated,
                    "data": production,
                }

            if "production_prognosis" not in weather_data:
                weather_data["production_prognosis"] = []
            weather_data["production_prognosis"].extend(production)

            # Update storage entries with production data
            prod_by_ts = {p["deliveryStart"]: p for p in production}
            for entry in stored_entries:
                ts = entry["timestamp"]
                if ts in prod_by_ts:
                    p = prod_by_ts[ts]
                    entry["solar"] = p.get("solar")
                    entry["wind_offshore"] = p.get("wind_offshore")
                    entry["wind_onshore"] = p.get("wind_onshore")

    if api_data is not None:
        api_data["_nordpool_cache"] = cache
        _LOGGER.info("Nordpool cache updated for %d dates", len(dates_to_fetch))

    return stored_entries
