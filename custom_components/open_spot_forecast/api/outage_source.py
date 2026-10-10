"""The shared update cycle of an outage message archive (#123, #138).

Outage messages are not grid rows: a message is revised in numbered
versions, each with its publication time, and the archive keeps every
version so the model can rebuild what the market knew at any origin
(``OutageIndex``, ``ml/outages.py``). ``OutageSource`` keeps
``umm_messages``/``umm_periods`` current for one region: the events of a
window once, then only what was published since the last update. Its
coverage — the event window fetched and the publication time it is
complete up to — is kept in ``meta``; a failed request leaves it, so the
request is repeated next time.

Two sources implement ``_fetch``: ``NordpoolUmmSource`` (Nord Pool's UMM
API, its own delivery areas) and ``EntsoeOutageSource`` (the ENTSO-E
Transparency Platform's outage documents, #138). Both store the flat rows
``parse_umm_messages`` defines, so ``OutageIndex`` reads them alike.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import TYPE_CHECKING, Any

from homeassistant.core import HomeAssistant
from homeassistant.util import dt as dt_util

from ..const import REGIONS
from ..time_slots import parse_utc

if TYPE_CHECKING:
    from ..ml.storage import LearningStorage

_LOGGER = logging.getLogger(__name__)


class OutageSource(ABC):
    """A region's outage messages, fetched incrementally into the archive."""

    # The ``meta`` key of the coverage (``source_state_<name>``) and the log label
    state_name: str
    label: str
    # The entities' credit for the source (``attribution.py``)
    attribution: str

    def __init__(
        self, hass: HomeAssistant, storage: LearningStorage, region: str
    ) -> None:
        """Initialize the source for a region in ``REGIONS``; nothing is fetched yet.

        Args:
            hass: Home Assistant instance.
            storage: The learning database holding the outage tables.
            region: The region (delivery area or bidding zone).
        """
        self.hass = hass
        self.storage = storage
        self.region = region
        self.eic = str(REGIONS[region]["entsoe"])
        self._failed = False
        # Coverage: (events from, events until, publications until), or None
        self._state: tuple[datetime, datetime, datetime] | None = None
        self._state_loaded = False

    @abstractmethod
    async def _fetch(
        self,
        event_start: datetime,
        event_stop: datetime,
        published_until: datetime,
        published_from: datetime | None,
    ) -> list[dict[str, Any]] | None:
        """Fetch the rows of a window; None if a request failed.

        Every version of the messages whose event overlaps
        ``[event_start, event_stop)``, published before ``published_until``
        and, for an incremental update, from ``published_from`` on.
        """

    def _warn(self, message: str, *args: Any) -> None:
        """Log a failure once as a warning, repeats at debug level."""
        level = logging.DEBUG if self._failed else logging.WARNING
        self._failed = True
        _LOGGER.log(level, message, *args)

    async def _load_state(self) -> tuple[datetime, datetime, datetime] | None:
        if not self._state_loaded:
            self._state_loaded = True
            stored = await self.hass.async_add_executor_job(
                self.storage.load_source_state, self.state_name
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
            self.state_name,
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
            "%s outages: %d window(s) requested, data changed: %s",
            self.label,
            len(windows),
            changed,
        )
        return changed

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
