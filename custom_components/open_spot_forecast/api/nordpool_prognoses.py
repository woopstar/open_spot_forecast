"""Nordpool's consumption and production prognoses as a time-series source.

``NordpoolPrognosisSource`` keeps the ``nordpool_prognoses`` table current
through the shared ``TimeSeriesSource`` (#32). An older delivery day is only
requested while one of its hours is missing a value, so complete history
costs no request. Today's and tomorrow's days are re-fetched on every update
(Nordpool revises them during the day; the upsert reports whether anything
changed). A day Nordpool has not published yet (tomorrow before ~13:00, or
the per-type production breakdown, which comes later than the total) is
asked for again after ``revalidate_after``. The same source fills the history
the model trains on and today's and tomorrow's rows the forecast reads
(``forecast_prognoses``).
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import TYPE_CHECKING, Any
from zoneinfo import ZoneInfo

from homeassistant.core import HomeAssistant

from ..const import NORDPOOL_MARKET_TZ
from ..ml.series_storage import NORDPOOL_PROGNOSES
from ..time_series import TimeRange
from ..time_slots import local_midnight
from .nordpool_data import fetch_consumption_prognosis, fetch_production_prognosis
from .time_series_source import TimeSeriesSource

if TYPE_CHECKING:
    from ..ml.storage import LearningStorage

_MARKET_TZ = ZoneInfo(NORDPOOL_MARKET_TZ)


def prognosis_rows(
    consumption: dict[str, float] | None, production: list[dict] | None
) -> list[dict[str, Any]]:
    """Combine a day's two prognoses into ``nordpool_prognoses`` rows.

    One row per consumption timestamp, with the production prognosis that
    starts at the same moment (None until Nordpool publishes the breakdown).
    """
    production_by_start = {p["deliveryStart"]: p for p in production or []}
    rows = []
    for timestamp, volume in (consumption or {}).items():
        produced = production_by_start.get(timestamp, {})
        rows.append(
            {
                "timestamp": timestamp,
                "consumption": volume,
                "solar": produced.get("solar"),
                "wind_offshore": produced.get("wind_offshore"),
                "wind_onshore": produced.get("wind_onshore"),
            }
        )
    return rows


def forecast_prognoses(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Return stored rows in the shape the prediction features read.

    ``consumption_prognosis`` maps a timestamp to MW; ``production_prognosis``
    lists ``deliveryStart``, ``solar``, ``wind_offshore`` and ``wind_onshore``
    (see ``FeatureMixin._combine_features``). Keys without data are left out.
    """
    prognoses: dict[str, Any] = {}
    consumption = {
        row["timestamp"]: row["consumption"]
        for row in rows
        if row.get("consumption") is not None
    }
    production = [
        {
            "deliveryStart": row["timestamp"],
            "solar": row.get("solar"),
            "wind_offshore": row.get("wind_offshore"),
            "wind_onshore": row.get("wind_onshore"),
        }
        for row in rows
        if row.get("solar") is not None
        or row.get("wind_offshore") is not None
        or row.get("wind_onshore") is not None
    ]
    if consumption:
        prognoses["consumption_prognosis"] = consumption
    if production:
        prognoses["production_prognosis"] = production
    return prognoses


def delivery_day_range(day: date) -> TimeRange:
    """Return Nordpool's delivery day (CET/CEST midnight to midnight) in UTC."""
    return (
        local_midnight(day, _MARKET_TZ).astimezone(UTC),
        local_midnight(day + timedelta(days=1), _MARKET_TZ).astimezone(UTC),
    )


class NordpoolPrognosisSource(TimeSeriesSource):
    """Nordpool's prognoses for one delivery area, fetched per delivery day."""

    spec = NORDPOOL_PROGNOSES
    # Nordpool's API sits behind Cloudflare, which blocks rapid requests
    request_interval = 1.0
    # Forecast runs are hours apart; this only stops a burst of runs (setup,
    # then the tomorrow-price refresh) from asking twice for an unpublished day
    revalidate_after = timedelta(minutes=15)

    def __init__(
        self,
        hass: HomeAssistant,
        storage: LearningStorage,
        region: str,
        horizon_cutoff: datetime | None = None,
    ) -> None:
        """Initialize the source for a delivery area (e.g. "DK1")."""
        super().__init__(hass, storage, horizon_cutoff)
        self.region = region

    def refresh_from(self, now: datetime) -> datetime | None:
        """Re-fetch from the start of today's delivery day (still revised)."""
        return delivery_day_range(now.astimezone(_MARKET_TZ).date())[0]

    def chunks(self, ranges: list[TimeRange]) -> list[TimeRange]:
        """Return one request per delivery day with a missing hour."""
        days: set[date] = set()
        for start, end in ranges:
            day = start.astimezone(_MARKET_TZ).date()
            while delivery_day_range(day)[0] < end:
                days.add(day)
                day += timedelta(days=1)
        return [delivery_day_range(day) for day in sorted(days)]

    async def _fetch(
        self, start: datetime, end: datetime
    ) -> list[dict[str, Any]] | None:
        """Fetch both prognoses for the delivery day starting at ``start``."""
        day = start.astimezone(_MARKET_TZ).date()
        consumption, updated_at = await fetch_consumption_prognosis(day, self.region)
        if consumption is None and updated_at is None:
            return None
        production, _ = await fetch_production_prognosis(day, self.region)
        return prognosis_rows(consumption, production)
