"""Per-source health of the update cycle (#276).

OSF reads about ten upstream sources (Nordpool, Open-Meteo, ENTSO-E,
Instrat, energy-charts, Nord Pool UMM, Stromligning's sensors). A failure
used to show up only as a WARNING line in the log. ``SourceHealth`` keeps,
per source, the last attempt, the last success, the last error and how far
its data reaches, so the config entry's diagnostics (``diagnostics.py``) and
the ``Data sources`` sensor (``source_health_sensor.py``) can show which
source fails and how old its data is.

It is recorded at the existing error-handling points only: the forecast
run's refresh (``async_refresh``), the price reads in ``updater.py`` and the
backfill in ``history_updater.py``. A source counts a request that got no
usable answer in ``failed_requests`` (``api/time_series_source.py``,
``api/outage_source.py``), so no API client needs a try/except of its own.

An error is kept as the exception class and a message without URLs or
query parameters (``describe_error``), as ``api/http.py`` logs them. Nothing
is persisted: the health starts empty after a restart.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any

from homeassistant.util import dt as dt_util

from .spot_prices import known_until

if TYPE_CHECKING:
    from .api.time_series_source import HistorySource

_LOGGER = logging.getLogger(__name__)

# The names of the price sources that are not a stored time series
STROMLIGNING_SOURCE = "stromligning"
SPOT_SENSOR_SOURCE = "spot_price_sensor"

# Longest error message kept
MAX_ERROR_LENGTH = 200

# A URL, and a secret-looking query parameter outside one
_URL = re.compile(r"\b[a-zA-Z][a-zA-Z0-9+.-]*://\S+")
_SECRET_PARAM = re.compile(r"(?i)\b(\w*(?:token|key|secret|password)\w*)=\S+")


def describe_error(error: BaseException | str) -> str:
    """Return an error as its class and message, without URLs or secrets.

    Args:
        error: The exception, or a message of OSF's own.

    Returns:
        ``"ClassName: message"`` (the class alone for an empty message),
        with every URL replaced by ``<url>`` and every ``…key=value`` /
        ``…token=value`` by ``…key=<redacted>``, at most
        ``MAX_ERROR_LENGTH`` characters.
    """
    if isinstance(error, BaseException):
        message = str(error)
        name = type(error).__name__
        text = f"{name}: {message}" if message else name
    else:
        text = error
    text = _URL.sub("<url>", text)
    text = _SECRET_PARAM.sub(r"\1=<redacted>", text)
    return text[:MAX_ERROR_LENGTH]


@dataclass(slots=True)
class SourceStatus:
    """What is known about one source's latest attempts."""

    last_attempt: datetime | None = None
    last_success: datetime | None = None
    last_error: str | None = None
    last_error_at: datetime | None = None
    # The latest moment the source's stored or read data covers
    data_until: datetime | None = None

    @property
    def healthy(self) -> bool:
        """Return whether the latest attempt succeeded."""
        return self.last_success is not None and self.last_success == self.last_attempt

    def age_minutes(self, now: datetime) -> int | None:
        """Return the whole minutes since the last success, None without one."""
        if self.last_success is None:
            return None
        return int((now - self.last_success).total_seconds() // 60)

    def as_dict(self, now: datetime) -> dict[str, Any]:
        """Return the status as JSON-ready values (ISO times, UTC)."""

        def iso(moment: datetime | None) -> str | None:
            return dt_util.as_utc(moment).isoformat() if moment else None

        return {
            "healthy": self.healthy,
            "last_attempt": iso(self.last_attempt),
            "last_success": iso(self.last_success),
            "age_minutes": self.age_minutes(now),
            "last_error": self.last_error,
            "last_error_at": iso(self.last_error_at),
            "data_until": iso(self.data_until),
        }


class SourceHealth:
    """The health of every source an update cycle used, by source name."""

    def __init__(self) -> None:
        """Start empty: a source appears at its first attempt."""
        self._sources: dict[str, SourceStatus] = {}

    def status(self, name: str) -> SourceStatus | None:
        """Return a source's status, None before its first attempt."""
        return self._sources.get(name)

    def _status(self, name: str, data_until: datetime | None) -> SourceStatus:
        status = self._sources.setdefault(name, SourceStatus())
        status.last_attempt = dt_util.utcnow()
        if data_until is not None:
            status.data_until = data_until
        return status

    def record_success(self, name: str, data_until: datetime | None = None) -> None:
        """Record a successful attempt (and where the data now ends, if known)."""
        status = self._status(name, data_until)
        status.last_success = status.last_attempt

    def record_failure(
        self,
        name: str,
        error: BaseException | str,
        data_until: datetime | None = None,
    ) -> None:
        """Record a failed attempt with its error (``describe_error``)."""
        status = self._status(name, data_until)
        status.last_error = describe_error(error)
        status.last_error_at = status.last_attempt

    def record_update(
        self, source: HistorySource, rows: list[dict[str, Any]] | None = None
    ) -> None:
        """Record a source's finished update: failed if a request failed.

        Args:
            source: The updated source; its ``failed_requests`` tells
                whether every request of the update was answered.
            rows: The stored rows read after the update, if any; their
                latest ``timestamp`` is where the data ends.
        """
        data_until = latest_timestamp(rows or [])
        if failed := source.failed_requests:
            message = f"{failed} request(s) failed"
            self.record_failure(source.source_name, message, data_until)
        else:
            self.record_success(source.source_name, data_until)

    def record_prices(
        self, name: str, data: dict[str, Any], failed_requests: int = 0
    ) -> None:
        """Record a price read: healthy if it has today's prices.

        Args:
            name: The price source.
            data: ``read_spot_prices()``-shaped data of the read.
            failed_requests: Requests of the read that got no answer.
        """
        if not isinstance(data, dict):
            data = {}
        data_until = known_until(data) if data.get("today") else None
        if failed_requests:
            self.record_failure(
                name, f"{failed_requests} request(s) failed", data_until
            )
        elif not data.get("today"):
            self.record_failure(name, "no prices for today", data_until)
        else:
            self.record_success(name, data_until)

    def record_source_prices(self, source: HistorySource, data: dict[str, Any]) -> None:
        """Record a price read from a stored source (the day-ahead APIs)."""
        self.record_prices(source.source_name, data, source.failed_requests)

    @property
    def healthy_count(self) -> int:
        """Return how many sources' latest attempt succeeded."""
        return sum(status.healthy for status in self._sources.values())

    def summary(self, now: datetime | None = None) -> dict[str, dict[str, Any]]:
        """Return every source's status (``SourceStatus.as_dict``), sorted by name."""
        now = now or dt_util.utcnow()
        return {
            name: self._sources[name].as_dict(now) for name in sorted(self._sources)
        }


def latest_timestamp(rows: list[dict[str, Any]]) -> datetime | None:
    """Return the latest ``timestamp`` of stored rows (UTC), None without one."""
    latest: datetime | None = None
    for row in rows if isinstance(rows, list) else []:
        value = row.get("timestamp") if isinstance(row, dict) else None
        moment = dt_util.parse_datetime(value) if isinstance(value, str) else None
        if moment is None:
            continue
        moment = dt_util.as_utc(moment)
        if latest is None or moment > latest:
            latest = moment
    return latest


async def async_refresh(
    health: SourceHealth,
    source: HistorySource,
    start: datetime,
    end: datetime,
    label: str,
) -> list[dict[str, Any]] | None:
    """Update a source for ``[start, end)``, record its health, return its rows.

    A failed update keeps what is stored; None if nothing can be read.
    """
    name = source.source_name
    try:
        await source.async_update(start, end)
    except Exception as err:
        _LOGGER.warning("Could not update the %s: %s", label, err)
        health.record_failure(name, err)
        updated = False
    else:
        updated = True
    try:
        rows = await source.async_load(start, end)
    except Exception as err:
        _LOGGER.warning("Could not read the stored %s: %s", label, err)
        if updated:
            health.record_failure(name, err)
        return None
    if updated:
        health.record_update(source, rows)
    elif (status := health.status(name)) and (until := latest_timestamp(rows)):
        status.data_until = until
    return rows
