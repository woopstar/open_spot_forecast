"""Incremental, gap-aware updates of a time-series source (#32).

``TimeSeriesSource`` is the shared plumbing every upstream time series uses
(Nordpool prognoses now; day-ahead prices and Open-Meteo weather next). An
update of ``[start, end)``:

1. finds the grid points the table is missing (``series_timestamps``);
2. adds the source's refresh window (recent forecasts are still revised);
3. skips holes the source did not have at an earlier request until their
   retry time: its known horizon ("no data yet beyond T") and gaps in older
   history;
4. fetches the rest in request-sized chunks and upserts the rows, which
   reports whether anything changed (and so triggers a retrain);
5. records what is still missing as holes, persisted in ``meta``.

Complete data therefore costs no request at all. ``horizon_cutoff`` hides
everything after a moment, so a backtest cannot see the future.
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

from homeassistant.core import HomeAssistant
from homeassistant.util import dt as dt_util

from ..time_series import (
    SourceState,
    TimeRange,
    grid_range,
    merge_ranges,
    missing_ranges,
    split_range,
    subtract_ranges,
)

if TYPE_CHECKING:
    from ..ml.series_storage import SeriesSpec
    from ..ml.storage import LearningStorage

_LOGGER = logging.getLogger(__name__)


class TimeSeriesSource(ABC):
    """A table of upstream rows, updated by fetching only what is missing."""

    spec: SeriesSpec
    # Pause between the requests of one update (rate-limited APIs)
    request_interval: float = 0.0
    # When to ask again for a recent hole (e.g. tomorrow, not published yet)
    revalidate_after: timedelta = timedelta(hours=1)
    # ... and for a hole in older history, which the source may never fill
    history_retry_after: timedelta = timedelta(days=1)
    # A hole that ends after now minus this is recent
    recent_window: timedelta = timedelta(days=2)
    # Longest range one request may cover (None: the API takes any range)
    max_request_span: timedelta | None = None

    def __init__(
        self,
        hass: HomeAssistant,
        storage: LearningStorage,
        horizon_cutoff: datetime | None = None,
    ) -> None:
        """Initialize the source; nothing is read until the first update.

        Args:
            hass: Home Assistant instance (executor for the blocking storage).
            storage: The learning database holding the source's table.
            horizon_cutoff: Never request or return data from this moment on
                (backtests).
        """
        self.hass = hass
        self.storage = storage
        self.horizon_cutoff = horizon_cutoff
        self._state: SourceState | None = None

    # ------------------------------------------------------------------
    # Source-specific behaviour
    # ------------------------------------------------------------------

    @abstractmethod
    async def _fetch(
        self, start: datetime, end: datetime
    ) -> list[dict[str, Any]] | None:
        """Request ``[start, end)`` from upstream.

        Returns:
            Rows (``timestamp`` plus the spec's columns; more than asked for
            is fine), an empty list if upstream has nothing, or None if the
            request failed (it is retried at the next update).
        """

    def keys(self) -> Sequence[str]:
        """Return the keys every grid point needs a row for (keyed tables)."""
        return ()

    def refresh_from(self, now: datetime) -> datetime | None:
        """Return from when stored rows are re-fetched anyway (revised forecasts)."""
        return None

    def chunks(self, ranges: list[TimeRange]) -> list[TimeRange]:
        """Return the requests that cover the missing ranges.

        One per range, split into ``max_request_span`` pieces if it is set.
        """
        if self.max_request_span is None:
            return ranges
        span = self.max_request_span
        return [
            chunk for time_range in ranges for chunk in split_range(time_range, span)
        ]

    def retry_time(self, hole: TimeRange, now: datetime) -> datetime:
        """Return when a range the source did not have may be requested again."""
        if hole[1] > now - self.recent_window:
            return now + self.revalidate_after
        return now + self.history_retry_after

    # ------------------------------------------------------------------
    # The update
    # ------------------------------------------------------------------

    async def async_update(self, start: datetime, end: datetime) -> bool:
        """Fetch the parts of ``[start, end)`` that are missing.

        Returns:
            Whether any stored value changed.
        """
        now = dt_util.utcnow()
        if self.horizon_cutoff is not None:
            end = min(end, self.horizon_cutoff)
        start, end = grid_range(start, end, self.spec.step_minutes)
        if start >= end:
            return False
        state = await self._async_state()

        wanted = missing_ranges(
            await self._stored(start, end), start, end, self.spec.step_minutes
        )
        refresh = self.refresh_from(now)
        if refresh is not None and refresh < end:
            wanted = merge_ranges([*wanted, (max(start, refresh), end)])
        wanted = subtract_ranges(wanted, state.pending(now))
        if not wanted:
            return False

        changed = False
        attempted: list[TimeRange] = []
        for index, (chunk_start, chunk_end) in enumerate(self.chunks(wanted)):
            if index and self.request_interval:
                await asyncio.sleep(self.request_interval)
            rows = await self._fetch(chunk_start, chunk_end)
            if rows is None:
                continue
            attempted.append((chunk_start, chunk_end))
            if not rows:
                continue
            upserted: bool = await self.hass.async_add_executor_job(
                self.storage.upsert_series, self.spec, rows
            )
            changed = changed or upserted

        if attempted:
            still = missing_ranges(
                await self._stored(start, end), start, end, self.spec.step_minutes
            )
            # Missing up to the end of a requested future: the source's horizon
            open_from = still[-1][0] if still and still[-1][1] == end > now else None
            state.record(
                attempted, still, open_from, lambda hole: self.retry_time(hole, now)
            )
            await self._async_save_state()
        _LOGGER.debug(
            "%s: requested %d range(s), %d answered, data changed: %s",
            self.spec.name,
            len(wanted),
            len(attempted),
            changed,
        )
        return changed

    async def async_load(self, start: datetime, end: datetime) -> list[dict[str, Any]]:
        """Return the stored rows in ``[start, end)`` (never past the cutoff)."""
        if self.horizon_cutoff is not None:
            end = min(end, self.horizon_cutoff)
        return await self.hass.async_add_executor_job(
            self.storage.load_series, self.spec, start, end
        )

    async def async_prune(self, before: datetime) -> int:
        """Delete rows (and remembered holes) older than ``before``."""
        deleted = await self.hass.async_add_executor_job(
            self.storage.prune_series, self.spec, before
        )
        state = await self._async_state()
        state.prune(before)
        await self._async_save_state()
        return int(deleted)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _stored(self, start: datetime, end: datetime) -> list[datetime]:
        return await self.hass.async_add_executor_job(
            self.storage.series_timestamps, self.spec, start, end, self.keys()
        )

    async def _async_state(self) -> SourceState:
        if self._state is None:
            stored = await self.hass.async_add_executor_job(
                self.storage.load_source_state, self.spec.name
            )
            self._state = SourceState.from_json(stored)
        return self._state

    async def _async_save_state(self) -> None:
        if self._state is not None:
            await self.hass.async_add_executor_job(
                self.storage.save_source_state, self.spec.name, self._state.to_json()
            )
