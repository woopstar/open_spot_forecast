"""Nord Pool's urgent market messages: outages of plants and interconnectors (#123).

Nord Pool's UMM API (``NORDPOOL_UMM_API``, JSON, no key) lists every
message version published for a delivery area: ``messageId``, ``version``,
``publicationDate``, ``eventStatus`` (3 = dismissed), ``messageType`` (1 =
production, 3 = transmission), ``unavailabilityType`` (1 = unplanned, 2 =
planned) and the units with their ``installedCapacity`` and
``timePeriods[{eventStart, eventStop, unavailableCapacity}]``. A production
message names ``productionUnits`` or ``generationUnits`` (never both) with
an ``areaEic``; a transmission message names ``transmissionUnits`` with
``inAreaEic``/``outAreaEic``, and one message often covers several
connections, so the API's ``areas`` filter is wide: ``parse_umm_messages``
keeps only the units of the region's area. ``eventStartDate``/
``eventStopDate`` select the messages whose event overlaps the window;
``includeOutdated`` adds the superseded versions, which training needs to
see what was known at a past origin.

``NordpoolUmmSource`` keeps ``umm_messages``/``umm_periods`` current: the
events of a window once (every version, paged with ``skip``/``limit``), then
only what was published since the last update. Its coverage is kept in
``meta``; a failed request leaves it, so the request is repeated next time.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from homeassistant.core import HomeAssistant
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.util import dt as dt_util

from ..const import NORDPOOL_UMM_API, REGIONS
from ..ml.outages import (
    KIND_PRODUCTION,
    KIND_TRANSMISSION,
    UMM_PRODUCTION,
    UMM_TRANSMISSION,
)
from ..time_slots import UTC_KEY_FORMAT, parse_utc
from .http import async_get

if TYPE_CHECKING:
    from ..ml.storage import LearningStorage

_LOGGER = logging.getLogger(__name__)

# Messages per request; the API honours thousands, this keeps a page small
UMM_PAGE_SIZE = 500
# Pages per window before giving up (a broken ``total`` must not loop forever)
_MAX_PAGES = 200
_STATE_NAME = "umm_outages"
_QUERY_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


def _utc_key(value: Any) -> str | None:
    moment = parse_utc(value)
    return moment.strftime(UTC_KEY_FORMAT) if moment is not None else None


def _integer(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _capacity(value: Any) -> float | None:
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    return None


def _units(message: dict[str, Any], eic: str) -> list[tuple[str, str, dict[str, Any]]]:
    """Return the message's units in the area as ``(kind, unit key, unit)``."""
    units: list[tuple[str, str, dict[str, Any]]] = []
    for field in ("productionUnits", "generationUnits"):
        for unit in message.get(field) or []:
            if isinstance(unit, dict) and unit.get("areaEic") == eic:
                units.append(
                    (KIND_PRODUCTION, str(unit.get("eic") or unit.get("name")), unit)
                )
    for unit in message.get("transmissionUnits") or []:
        if isinstance(unit, dict) and eic in (
            unit.get("inAreaEic"),
            unit.get("outAreaEic"),
        ):
            key = f"{unit.get('inAreaEic')}>{unit.get('outAreaEic')}"
            units.append((KIND_TRANSMISSION, key, unit))
    return units


def parse_umm_messages(payload: Any, eic: str) -> list[dict[str, Any]]:
    """Return the stored rows of a ``/messages`` page for one area.

    One row per unit period in the area (``kind`` set), and one row without
    a period (``kind`` None) for a version that names no unit in the area,
    so it still supersedes earlier versions. Only production and
    transmission messages count; a message or period that lacks its key
    fields is skipped.

    Args:
        payload: The decoded response (``{"items": [...], "total": n}``).
        eic: The area's EIC code (``REGIONS[...]["entsoe"]``).

    Raises:
        TypeError: If the payload is not a page of message objects.
    """
    if not isinstance(payload, dict) or not isinstance(payload.get("items"), list):
        raise TypeError("UMM response is not a page of messages")
    rows: list[dict[str, Any]] = []
    for message in payload["items"]:
        if not isinstance(message, dict):
            raise TypeError("UMM message is not an object")
        message_type = _integer(message.get("messageType"))
        if message_type not in (UMM_PRODUCTION, UMM_TRANSMISSION):
            continue
        message_id = message.get("messageId")
        version = _integer(message.get("version"))
        published = _utc_key(message.get("publicationDate"))
        status = _integer(message.get("eventStatus"))
        if not isinstance(message_id, str) or None in (version, published, status):
            continue
        header: dict[str, Any] = {
            "message_id": message_id,
            "version": version,
            "published": published,
            "message_type": message_type,
            "unavailability_type": _integer(message.get("unavailabilityType")),
            "status": status,
        }
        periods: list[dict[str, Any]] = []
        for kind, unit_key, unit in _units(message, eic):
            for period in unit.get("timePeriods") or []:
                if not isinstance(period, dict):
                    continue
                start = _utc_key(period.get("eventStart"))
                stop = _utc_key(period.get("eventStop"))
                unavailable = _capacity(period.get("unavailableCapacity"))
                if start is None or stop is None or unavailable is None:
                    continue
                periods.append(
                    header
                    | {
                        "unit": unit_key,
                        "kind": kind,
                        "fuel_type": _integer(unit.get("fuelType")),
                        "event_start": start,
                        "event_stop": stop,
                        "unavailable_mw": unavailable,
                        "installed_mw": _capacity(unit.get("installedCapacity")),
                    }
                )
        rows.extend(periods or [header | {"kind": None}])
    return rows


def umm_query(
    eic: str,
    event_start: datetime,
    event_stop: datetime,
    published_until: datetime,
    published_from: datetime | None = None,
    skip: int = 0,
    limit: int = UMM_PAGE_SIZE,
) -> list[tuple[str, str]]:
    """Return the query of one page of an area's messages.

    Messages whose event overlaps ``[event_start, event_stop)``, every
    version (``includeOutdated``), published before ``published_until``
    (and from ``published_from`` on, for an incremental update). The list
    filters repeat their key, as the API expects.
    """
    query = [
        ("areas", eic),
        ("messageTypes", str(UMM_PRODUCTION)),
        ("messageTypes", str(UMM_TRANSMISSION)),
        ("includeOutdated", "true"),
        ("eventStartDate", event_start.astimezone(UTC).strftime(_QUERY_FORMAT)),
        ("eventStopDate", event_stop.astimezone(UTC).strftime(_QUERY_FORMAT)),
        (
            "publicationStopDate",
            published_until.astimezone(UTC).strftime(_QUERY_FORMAT),
        ),
        ("skip", str(skip)),
        ("limit", str(limit)),
    ]
    if published_from is not None:
        query.append(
            (
                "publicationStartDate",
                published_from.astimezone(UTC).strftime(_QUERY_FORMAT),
            )
        )
    return query


class NordpoolUmmSource:
    """The region's UMM outage messages, fetched incrementally."""

    def __init__(
        self, hass: HomeAssistant, storage: LearningStorage, region: str
    ) -> None:
        """Initialize the source for a region in ``REGIONS``; nothing is fetched yet.

        Args:
            hass: Home Assistant instance.
            storage: The learning database holding the UMM tables.
            region: The region (delivery area).
        """
        self.hass = hass
        self.storage = storage
        self.eic = str(REGIONS[region]["entsoe"])
        self._failed = False
        # Coverage: (events from, events until, publications until), or None
        self._state: tuple[datetime, datetime, datetime] | None = None
        self._state_loaded = False

    def _warn(self, message: str, *args: Any) -> None:
        """Log a failure once as a warning, repeats at debug level."""
        level = logging.DEBUG if self._failed else logging.WARNING
        self._failed = True
        _LOGGER.log(level, message, *args)

    async def _load_state(self) -> tuple[datetime, datetime, datetime] | None:
        if not self._state_loaded:
            self._state_loaded = True
            stored = await self.hass.async_add_executor_job(
                self.storage.load_source_state, _STATE_NAME
            )
            try:
                parts = [parse_utc(part) for part in json.loads(stored or "")]
            except ValueError, TypeError:
                parts = []
            if len(parts) == 3 and all(parts):
                self._state = (parts[0], parts[1], parts[2])  # type: ignore[assignment]
        return self._state

    async def _save_state(self, state: tuple[datetime, datetime, datetime]) -> None:
        self._state = state
        await self.hass.async_add_executor_job(
            self.storage.save_source_state,
            _STATE_NAME,
            json.dumps([moment.isoformat() for moment in state]),
        )

    async def async_update(self, start: datetime, end: datetime) -> bool:
        """Store the messages for events in ``[start, end)`` and new publications.

        The parts of the window not covered yet are fetched with every
        version; the covered part only with what was published since the
        last update. The coverage is saved when every request succeeded.

        Returns:
            Whether a new message version was stored.
        """
        now = dt_util.utcnow()
        state = await self._load_state()
        windows: list[tuple[datetime, datetime, datetime | None]] = []
        if state is None:
            windows.append((start, end, None))
        else:
            covered_from, covered_until, published_until = state
            if start < covered_from:
                windows.append((start, covered_from, None))
            if end > covered_until:
                windows.append((covered_until, end, None))
            windows.append((covered_from, covered_until, published_until))
        changed = False
        for event_start, event_stop, published_from in windows:
            if event_start >= event_stop:
                continue
            rows = await self._fetch(event_start, event_stop, now, published_from)
            if rows is None:
                return changed
            if rows:
                stored: bool = await self.hass.async_add_executor_job(
                    self.storage.upsert_umm_rows, rows
                )
                changed = changed or stored
        covered = (
            (start, end)
            if state is None
            else (min(start, state[0]), max(end, state[1]))
        )
        await self._save_state((covered[0], covered[1], now))
        _LOGGER.debug(
            "UMM outages: %d window(s) requested, data changed: %s",
            len(windows),
            changed,
        )
        return changed

    async def _fetch(
        self,
        event_start: datetime,
        event_stop: datetime,
        published_until: datetime,
        published_from: datetime | None,
    ) -> list[dict[str, Any]] | None:
        """Fetch every page of a window; None if a request failed."""
        rows: list[dict[str, Any]] = []
        skip = 0
        for _ in range(_MAX_PAGES):
            response = await async_get(
                async_get_clientsession(self.hass),
                NORDPOOL_UMM_API,
                "Nord Pool UMM",
                params=umm_query(
                    self.eic,
                    event_start,
                    event_stop,
                    published_until,
                    published_from,
                    skip,
                ),
            )
            if response is None or response.status != 200:
                status = response.status if response is not None else "no answer"
                self._warn("Outage messages unavailable (Nord Pool UMM: %s)", status)
                return None
            try:
                payload = json.loads(response.text)
                rows.extend(parse_umm_messages(payload, self.eic))
            except (ValueError, TypeError) as err:
                self._warn("Unusable Nord Pool UMM response: %s", err)
                return None
            items = payload.get("items") or []
            total = payload.get("total")
            skip += len(items)
            if not items or not isinstance(total, int) or skip >= total:
                break
        self._failed = False
        return rows

    async def async_load(self, start: datetime, end: datetime) -> list[dict[str, Any]]:
        """Return the stored versions with their periods overlapping ``[start, end)``."""
        return await self.hass.async_add_executor_job(
            self.storage.load_umm_rows, start, end
        )

    async def async_prune(self, before: datetime) -> int:
        """Delete periods that ended before ``before`` and versions left without any."""
        deleted: int = await self.hass.async_add_executor_job(
            self.storage.prune_umm_rows, before
        )
        state = await self._load_state()
        if state is not None and state[0] < before < state[1]:
            await self._save_state((before, state[1], state[2]))
        return deleted
