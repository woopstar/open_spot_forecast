"""ENTSO-E's outage documents: plant and grid unavailability (#138).

DE, NL, BE and FR publish their REMIT unavailability on the ENTSO-E
Transparency Platform instead of Nord Pool's UMM API (#123). The outage
domain has three document types — ``A77`` production unit and ``A80``
generation unit unavailability, requested per bidding zone, and ``A78``
transmission unavailability, requested per border (``in_Domain`` and
``out_Domain`` are both mandatory, so each border is asked in both
directions, ``ENTSOE_OUTAGE_BORDERS``). ``periodStart``/``periodEnd``
select the documents whose event overlaps the window, and
``PeriodStartUpdate``/``PeriodEndUpdate`` those published in a window. A
response is a ZIP of XML documents, at most ``ENTSOE_DOCUMENT_LIMIT`` per
request, paged with ``offset`` up to ``ENTSOE_OFFSET_LIMIT``; a window with
more documents than that is halved. "No matching data" is an
acknowledgement document instead of a ZIP.

Each document is one message version: ``mRID`` + ``revisionNumber``,
``createdDateTime`` (its publication), ``docStatus`` A09 cancelled / A13
withdrawn, and a ``TimeSeries`` naming the unit (with its ``nominalP`` for
a plant) and ``Available_Period``s of the *available* quantity, a variable
block curve (``PT1M`` points where it changes). ``parse_entsoe_outages``
turns them into the flat rows of ``parse_umm_messages``: for a plant one
row per point segment with ``unavailable_mw = nominalP − available``; for a
grid asset — the platform publishes no nominal capacity — a row counting
**1.0** per asset under a limitation, so ``unavailable_transmission`` is the
number of restricted assets in these zones (docs/ml_documentation.md).

**The platform serves only the current revision of a document.** The
first fetch of a window therefore stores each document once, at the
publication time of its latest revision; from then on every update stores
what was published since (``OutageSource``), so the archive gains the
earlier versions of later revisions as they happen. A training row can only
see a document from the stored revision's publication on — a conservative
view, never a leaked one.
"""

from __future__ import annotations

import io
import logging
import xml.etree.ElementTree as ET  # nosec B405 — see api/entsoe.py
import zipfile
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, NamedTuple

from homeassistant.core import HomeAssistant

from ..attribution import ENTSOE_OUTAGE_ATTRIBUTION
from ..const import ENTSOE_OUTAGE_BORDERS
from ..ml.outages import (
    KIND_PRODUCTION,
    KIND_TRANSMISSION,
    UMM_ACTIVE,
    UMM_DISMISSED,
    UMM_PRODUCTION,
    UMM_TRANSMISSION,
)
from ..time_slots import UTC_KEY_FORMAT, parse_utc
from .entsoe import (
    async_entsoe_get_bytes,
    child,
    child_text,
    children,
    entsoe_period,
    local_name,
    period_bounds,
)
from .outage_source import OutageSource

if TYPE_CHECKING:
    from ..ml.storage import LearningStorage

_LOGGER = logging.getLogger(__name__)

# documentType: production unit, generation unit and transmission unavailability
DOCUMENT_KINDS: dict[str, str] = {
    "A77": KIND_PRODUCTION,
    "A80": KIND_PRODUCTION,
    "A78": KIND_TRANSMISSION,
}
_MESSAGE_TYPES = {KIND_PRODUCTION: UMM_PRODUCTION, KIND_TRANSMISSION: UMM_TRANSMISSION}
# businessType: planned (A53) and forced (A54), as UMM ``unavailabilityType``
_UNAVAILABILITY_TYPES = {"A53": 2, "A54": 1}
# docStatus: a cancelled or withdrawn document's outage does not take place
DISMISSED_STATUSES = frozenset({"A09", "A13"})
# Documents per response, and the largest ``offset`` the API accepts
ENTSOE_DOCUMENT_LIMIT = 200
ENTSOE_OFFSET_LIMIT = 4800
# A window this short is not halved further when it still overflows
_MIN_WINDOW = timedelta(days=1)
_NO_DATA = "No matching data"
_PRODUCTION_UNIT = "production_RegisteredResource"
_GENERATION_UNIT = f"{_PRODUCTION_UNIT}.pSRType.powerSystemResources"


class OutageResponse(NamedTuple):
    """A parsed response: how many documents it held and their rows."""

    documents: int
    rows: list[dict[str, Any]]


def _utc_key(value: Any) -> str | None:
    moment = parse_utc(value)
    return moment.strftime(UTC_KEY_FORMAT) if moment is not None else None


def _epoch_key(moment: datetime) -> str:
    return moment.astimezone(UTC).strftime(UTC_KEY_FORMAT)


def _float(value: str | None) -> float | None:
    try:
        return float(value) if value is not None else None
    except ValueError:
        return None


def entsoe_outage_queries(
    eic: str,
    borders: tuple[str, ...],
    event_start: datetime,
    event_stop: datetime,
    published_from: datetime | None = None,
    published_until: datetime | None = None,
) -> list[dict[str, str]]:
    """Return the queries a window needs, without ``offset`` and token.

    One per plant document type (A77, A80) for the zone, and one per border
    and direction for the grid (A78). Documents whose event overlaps
    ``[event_start, event_stop)``; with ``published_from`` only those
    published in ``[published_from, published_until)``.
    """
    period = entsoe_period(event_start, event_stop)
    if published_from is not None and published_until is not None:
        updated = entsoe_period(published_from, published_until)
        period |= {
            "PeriodStartUpdate": updated["periodStart"],
            "PeriodEndUpdate": updated["periodEnd"],
        }
    queries = [
        {"documentType": doc_type, "biddingZone_Domain": eic} | period
        for doc_type, kind in DOCUMENT_KINDS.items()
        if kind == KIND_PRODUCTION
    ]
    for other in borders:
        queries.append({"documentType": "A78", "in_Domain": eic, "out_Domain": other})
        queries.append({"documentType": "A78", "in_Domain": other, "out_Domain": eic})
    return [query | period for query in queries]


def outage_documents(body: bytes) -> list[bytes]:
    """Return the XML documents of a response.

    The members of a ZIP, a single unavailability document, or none for a
    "no matching data" acknowledgement.

    Raises:
        ValueError: If the body is neither, or the acknowledgement names
            another reason (a rejected request).
    """
    if body.startswith(b"PK"):
        try:
            with zipfile.ZipFile(io.BytesIO(body)) as archive:
                return [archive.read(name) for name in archive.namelist()]
        except zipfile.BadZipFile as err:
            raise ValueError("ENTSO-E outage response is not a ZIP") from err
    try:
        root = ET.fromstring(body)  # nosec B314
    except ET.ParseError as err:
        raise ValueError("ENTSO-E outage response is not XML") from err
    if local_name(root) != "Acknowledgement_MarketDocument":
        return [body]
    reason = child_text(root, "Reason", "text") or ""
    if reason.startswith(_NO_DATA):
        return []
    raise ValueError(f"ENTSO-E rejected the request: {reason or 'no reason given'}")


def _segments(
    series: ET.Element,
) -> list[tuple[str, str, float]]:
    """Return the series' ``(start, stop, available)`` segments, UTC keys.

    A variable block curve gives a point where the quantity changes; a
    segment runs to the next point or the period's end.
    """
    segments: list[tuple[str, str, float]] = []
    for period in children(series, "Available_Period"):
        bounds = period_bounds(period)
        if bounds is None:
            continue
        start, end, step = bounds
        if end is None:
            continue
        points: dict[int, float] = {}
        for point in children(period, "Point"):
            position = child_text(point, "position")
            quantity = _float(child_text(point, "quantity"))
            if position and position.isdigit() and quantity is not None:
                points[int(position)] = quantity
        positions = sorted(points)
        for index, number in enumerate(positions):
            segment_start = start + (number - 1) * step
            segment_end = (
                start + (positions[index + 1] - 1) * step
                if index + 1 < len(positions)
                else end
            )
            if segment_start < segment_end:
                segments.append(
                    (
                        _epoch_key(segment_start),
                        _epoch_key(segment_end),
                        points[number],
                    )
                )
    return segments


def _production_periods(series: ET.Element) -> list[dict[str, Any]]:
    """Return a plant series' periods: ``nominalP − available`` where positive."""
    nominal = _float(child_text(series, f"{_GENERATION_UNIT}.nominalP"))
    unit = child_text(series, f"{_GENERATION_UNIT}.mRID") or child_text(
        series, f"{_PRODUCTION_UNIT}.mRID"
    )
    if nominal is None or not unit:
        return []
    return [
        {
            "unit": unit,
            "event_start": start,
            "event_stop": stop,
            "unavailable_mw": nominal - available,
            "installed_mw": nominal,
        }
        for start, stop, available in _segments(series)
        if nominal - available > 1e-9
    ]


def _transmission_periods(series: ET.Element) -> list[dict[str, Any]]:
    """Return a grid series' periods: 1.0 per segment under a limitation.

    The platform gives the available capacity but no nominal one, so the
    row counts the asset. In a curve with several quantities the segments
    at the highest one are the asset's full capacity and count nothing.
    """
    asset = child(series, "Asset_RegisteredResource")
    unit = (
        (
            child_text(asset, "mRID") or child_text(asset, "name")
            if asset is not None
            else None
        )
        or f"{child_text(series, 'in_Domain.mRID')}>{child_text(series, 'out_Domain.mRID')}"
    )
    segments = _segments(series)
    highest = max((available for _, _, available in segments), default=0.0)
    limited = len({available for _, _, available in segments}) == 1
    return [
        {
            "unit": unit,
            "event_start": start,
            "event_stop": stop,
            "unavailable_mw": 1.0,
            "installed_mw": None,
        }
        for start, stop, available in segments
        if limited or available < highest - 1e-9
    ]


def parse_outage_document(document: bytes) -> list[dict[str, Any]]:
    """Return the stored rows of one unavailability document.

    One row per unit segment (``kind`` set), or one row without a period
    (``kind`` None) for a version that names no usable unit, so it still
    supersedes earlier versions. Other document types, and a document
    without its key fields, give no rows.

    Raises:
        ValueError: If the document is not XML.
    """
    try:
        root = ET.fromstring(document)  # nosec B314
    except ET.ParseError as err:
        raise ValueError("ENTSO-E outage document is not XML") from err
    kind = DOCUMENT_KINDS.get(child_text(root, "type") or "")
    message_id = child_text(root, "mRID")
    version = child_text(root, "revisionNumber")
    published = _utc_key(child_text(root, "createdDateTime"))
    if kind is None or not message_id or not version or not version.isdigit():
        return []
    if published is None:
        return []
    status = child_text(root, "docStatus", "value")
    series_list = list(children(root, "TimeSeries"))
    business = child_text(series_list[0], "businessType") if series_list else None
    header: dict[str, Any] = {
        "message_id": message_id,
        "version": int(version),
        "published": published,
        "message_type": _MESSAGE_TYPES[kind],
        "unavailability_type": _UNAVAILABILITY_TYPES.get(business or ""),
        "status": UMM_DISMISSED if status in DISMISSED_STATUSES else UMM_ACTIVE,
    }
    periods = [
        header | {"kind": kind, "fuel_type": None} | period
        for series in series_list
        for period in (
            _production_periods(series)
            if kind == KIND_PRODUCTION
            else _transmission_periods(series)
        )
    ]
    return periods or [header | {"kind": None}]


def parse_entsoe_outages(body: bytes) -> OutageResponse:
    """Return the document count and the rows of an outage response.

    Raises:
        ValueError: If the response is not a ZIP of XML documents, a single
            document or a "no matching data" acknowledgement.
    """
    documents = outage_documents(body)
    rows: list[dict[str, Any]] = []
    for document in documents:
        rows.extend(parse_outage_document(document))
    return OutageResponse(len(documents), rows)


class EntsoeOutageSource(OutageSource):
    """The zone's ENTSO-E outage documents, fetched incrementally."""

    state_name = "entsoe_outages"
    label = "ENTSO-E"
    attribution = ENTSOE_OUTAGE_ATTRIBUTION

    def __init__(
        self, hass: HomeAssistant, storage: LearningStorage, region: str, api_key: str
    ) -> None:
        """Initialize the source for a region in ``ENTSOE_OUTAGE_BORDERS``.

        Args:
            hass: Home Assistant instance.
            storage: The learning database holding the outage tables.
            region: The region (bidding zone).
            api_key: ENTSO-E security token.
        """
        super().__init__(hass, storage, region)
        self._api_key = api_key
        self.borders = ENTSOE_OUTAGE_BORDERS.get(region, ())

    async def _fetch(
        self,
        event_start: datetime,
        event_stop: datetime,
        published_until: datetime,
        published_from: datetime | None,
    ) -> list[dict[str, Any]] | None:
        """Fetch every query of a window, paged; None if a request failed."""
        rows: list[dict[str, Any]] = []
        for query in entsoe_outage_queries(
            self.eic,
            self.borders,
            event_start,
            event_stop,
            published_from,
            published_until,
        ):
            fetched = await self._documents(query, event_start, event_stop)
            if fetched is None:
                return None
            rows.extend(fetched)
        self._failed = False
        return rows

    async def _documents(
        self, query: dict[str, str], start: datetime, end: datetime
    ) -> list[dict[str, Any]] | None:
        """Fetch every page of one query; a window past the offset limit is halved."""
        rows: list[dict[str, Any]] = []
        for offset in range(0, ENTSOE_OFFSET_LIMIT + 1, ENTSOE_DOCUMENT_LIMIT):
            body = await async_entsoe_get_bytes(
                self.hass, self._api_key, query | {"offset": str(offset)}
            )
            if body is None:
                self._warn("Outage documents unavailable (ENTSO-E)")
                return None
            try:
                response: OutageResponse = await self.hass.async_add_executor_job(
                    parse_entsoe_outages, body
                )
            except ValueError as err:
                self._warn("Unusable ENTSO-E outage response: %s", err)
                return None
            rows.extend(response.rows)
            if response.documents < ENTSOE_DOCUMENT_LIMIT:
                return rows
        if end - start <= _MIN_WINDOW:
            _LOGGER.warning(
                "ENTSO-E %s outage documents: more than %d in one day, rest skipped",
                query["documentType"],
                ENTSOE_OFFSET_LIMIT + ENTSOE_DOCUMENT_LIMIT,
            )
            return rows
        middle = start + (end - start) / 2
        halves: list[dict[str, Any]] = []
        for half_start, half_end in ((start, middle), (middle, end)):
            half = await self._documents(
                query | entsoe_period(half_start, half_end), half_start, half_end
            )
            if half is None:
                return None
            halves.extend(half)
        return halves
