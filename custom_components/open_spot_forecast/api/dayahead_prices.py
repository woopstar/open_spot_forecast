"""Day-ahead auction prices: energy-charts.info, with ENTSO-E as fallback (#27).

``DayAheadPriceSource`` keeps the ``dayahead_prices`` table (raw EUR/MWh
per UTC 15-minute slot) current through the shared ``TimeSeriesSource``:
only missing slots are requested, and tomorrow is asked for again from
12:45 CET, when the auction results are due, then every few minutes.

* **energy-charts.info** (``/price?bzn=``, no key) is asked first. Its
  licence depends on the zone and comes with every response
  (``license_info``, kept for attribution).
* **ENTSO-E** (Transparency Platform, ``documentType=A44``) is asked when a
  security token is configured and energy-charts failed or did not cover the
  whole request. The token is sent as a query parameter and never logged.

Requests span whole local days of the region: a UTC-midnight window would
cut off the first hours of the day for zones east of UTC. Hourly prices
(before the move to 15-minute products) fill their four quarter-hours.
"""

from __future__ import annotations

import json
import logging

# The stdlib parser: its expat guards against entity expansion, and ENTSO-E is
# an official HTTPS source (defusedxml is not a dependency)
import xml.etree.ElementTree as ET  # nosec B405
from datetime import UTC, datetime, time, timedelta
from typing import TYPE_CHECKING, Any
from zoneinfo import ZoneInfo

from homeassistant.core import HomeAssistant
from homeassistant.helpers.aiohttp_client import async_get_clientsession

from ..const import ENERGY_CHARTS_API, ENTSOE_API, REGIONS
from ..ml.series_storage import DAYAHEAD_PRICES
from ..time_series import TimeRange, day_chunks, missing_ranges
from .http import async_get
from .time_series_source import TimeSeriesSource

if TYPE_CHECKING:
    from ..ml.storage import LearningStorage

_LOGGER = logging.getLogger(__name__)

ENERGY_CHARTS_UNIT = "EUR / MWh"
_SLOT = timedelta(minutes=15)
_HOUR = timedelta(hours=1)
# The day-ahead auction's results are published from about 12:45 CET
_AUCTION_TZ = ZoneInfo("Europe/Berlin")
_AUCTION_RESULTS = time(12, 45)
# One request covers at most this many local days
_MAX_REQUEST_DAYS = 31


def _quarter_rows(
    start: datetime, span: timedelta, price: float
) -> list[dict[str, Any]]:
    """Return one row per 15-minute slot of a price lasting ``span``."""
    quarters = max(1, min(span, _HOUR) // _SLOT)
    return [
        {"timestamp": (start + index * _SLOT).isoformat(), "price": price}
        for index in range(quarters)
    ]


def parse_energy_charts(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Return ``dayahead_prices`` rows from an energy-charts ``/price`` response.

    Each price lasts until the next timestamp (the series' resolution for the
    last one), at most an hour. Null prices are skipped.

    Raises:
        ValueError: If the unit is not EUR/MWh or the arrays do not match.
    """
    if payload.get("unit") != ENERGY_CHARTS_UNIT:
        raise ValueError(f"unexpected energy-charts unit: {payload.get('unit')!r}")
    starts = [int(value) for value in payload.get("unix_seconds") or []]
    prices = list(payload.get("price") or [])
    if len(starts) != len(prices):
        raise ValueError("energy-charts unix_seconds and price differ in length")
    rows: list[dict[str, Any]] = []
    for index, (start, price) in enumerate(zip(starts, prices, strict=True)):
        if price is None:
            continue
        if index + 1 < len(starts):
            span = starts[index + 1] - start
        else:
            span = start - starts[index - 1] if index else 900
        moment = datetime.fromtimestamp(start, UTC)
        rows.extend(_quarter_rows(moment, timedelta(seconds=span), float(price)))
    return rows


def _local_name(element: ET.Element) -> str:
    return element.tag.rsplit("}", 1)[-1]


def _child(element: ET.Element, name: str) -> ET.Element | None:
    return next((c for c in element if _local_name(c) == name), None)


def _child_text(element: ET.Element, *path: str) -> str | None:
    node: ET.Element | None = element
    for name in path:
        node = _child(node, name) if node is not None else None
    return node.text if node is not None else None


def _resolution(value: str | None) -> timedelta | None:
    """Return an ISO 8601 duration such as ``PT15M`` or ``PT60M``."""
    if not value or not value.startswith("PT"):
        return None
    number, unit = value[2:-1], value[-1]
    if not number.isdigit() or unit not in "HM":
        return None
    return (
        timedelta(hours=int(number)) if unit == "H" else timedelta(minutes=int(number))
    )


def parse_entsoe(text: str) -> list[dict[str, Any]]:
    """Return ``dayahead_prices`` rows from an ENTSO-E A44 document.

    Positions missing from a period repeat the previous price in an A03
    (variable-sized block) curve and are left out in an A01 curve. An
    acknowledgement document ("no matching data") has no rows.

    Raises:
        ValueError: If the text is not XML.
    """
    try:
        root = ET.fromstring(text)  # nosec B314
    except ET.ParseError as err:
        raise ValueError("ENTSO-E response is not XML") from err
    rows: list[dict[str, Any]] = []
    for series in (e for e in root.iter() if _local_name(e) == "TimeSeries"):
        fill_forward = _child_text(series, "curveType") == "A03"
        for period in (e for e in series if _local_name(e) == "Period"):
            start_text = _child_text(period, "timeInterval", "start")
            end_text = _child_text(period, "timeInterval", "end")
            step = _resolution(_child_text(period, "resolution"))
            if not start_text or not end_text or step is None:
                continue
            start = datetime.fromisoformat(start_text)
            end = datetime.fromisoformat(end_text)
            points: dict[int, float] = {}
            for point in (e for e in period if _local_name(e) == "Point"):
                index = _child_text(point, "position")
                amount = _child_text(point, "price.amount")
                if index and amount:
                    points[int(index)] = float(amount)
            price: float | None = None
            for position in range(1, (end - start) // step + 1):
                if position in points:
                    price = points[position]
                elif not fill_forward:
                    continue
                if price is not None:
                    moment = start + (position - 1) * step
                    rows.extend(_quarter_rows(moment, step, price))
    return rows


class DayAheadPriceSource(TimeSeriesSource):
    """Day-ahead prices for one region's bidding zone."""

    spec = DAYAHEAD_PRICES
    # energy-charts answers bursts of requests with 429
    request_interval = 2.0
    # After 12:45 CET, ask for tomorrow's prices every few minutes
    revalidate_after = timedelta(minutes=5)

    def __init__(
        self,
        hass: HomeAssistant,
        storage: LearningStorage,
        region: str,
        entsoe_api_key: str | None = None,
        horizon_cutoff: datetime | None = None,
    ) -> None:
        """Initialize the source for a region in ``REGIONS``.

        Args:
            hass: Home Assistant instance.
            storage: The learning database holding ``dayahead_prices``.
            region: The region (bidding zone).
            entsoe_api_key: ENTSO-E security token, enabling the fallback.
            horizon_cutoff: Never request or return data from this moment on.
        """
        super().__init__(hass, storage, horizon_cutoff)
        zone = REGIONS[region]
        self.zone = str(zone["energy_charts"])
        self.eic = str(zone["entsoe"])
        self.tz = ZoneInfo(str(zone["tz"]))
        self._entsoe_api_key = entsoe_api_key or None
        # The licence of the latest energy-charts response (attribution)
        self.license_info: str | None = None

    def retry_time(self, hole: TimeRange, now: datetime) -> datetime:
        """Ask for unpublished prices when the auction results are due."""
        if hole[1] <= now - self.recent_window:
            return now + self.history_retry_after
        local_now = now.astimezone(_AUCTION_TZ)
        results = datetime.combine(local_now.date(), _AUCTION_RESULTS, _AUCTION_TZ)
        if local_now < results:
            return results.astimezone(UTC)
        return now + self.revalidate_after

    def chunks(self, ranges: list[TimeRange]) -> list[TimeRange]:
        """Return requests of whole local days, at most ``_MAX_REQUEST_DAYS`` each."""
        return day_chunks(ranges, self.tz, _MAX_REQUEST_DAYS)

    async def _fetch(
        self, start: datetime, end: datetime
    ) -> list[dict[str, Any]] | None:
        """Fetch energy-charts, then ENTSO-E for what energy-charts lacks."""
        rows = await self._fetch_energy_charts(start, end)
        if rows and not self._uncovered(rows, start, end):
            return rows
        if self._entsoe_api_key is None:
            return rows
        fallback = await self._fetch_entsoe(start, end)
        if fallback is None:
            return rows
        # energy-charts wins where both have a price
        known = {row["timestamp"] for row in rows or []}
        return [*(rows or []), *(r for r in fallback if r["timestamp"] not in known)]

    @staticmethod
    def _uncovered(rows: list[dict[str, Any]], start: datetime, end: datetime) -> bool:
        stored = [datetime.fromisoformat(row["timestamp"]) for row in rows]
        return bool(missing_ranges(stored, start, end, DAYAHEAD_PRICES.step_minutes))

    async def _fetch_energy_charts(
        self, start: datetime, end: datetime
    ) -> list[dict[str, Any]] | None:
        # The end is inclusive: stop before the next day's first slot
        response = await async_get(
            async_get_clientsession(self.hass),
            ENERGY_CHARTS_API,
            "energy-charts",
            params={
                "bzn": self.zone,
                "start": start.astimezone(self.tz).isoformat(timespec="minutes"),
                "end": (end - _SLOT).astimezone(self.tz).isoformat(timespec="minutes"),
            },
        )
        if response is None or response.status != 200:
            if response is not None:
                _LOGGER.warning("energy-charts returned %d", response.status)
            return None
        try:
            payload = json.loads(response.text)
            rows = parse_energy_charts(payload)
        except ValueError as err:
            _LOGGER.warning("Unusable energy-charts response: %s", err)
            return None
        self.license_info = payload.get("license_info") or self.license_info
        return rows

    async def _fetch_entsoe(
        self, start: datetime, end: datetime
    ) -> list[dict[str, Any]] | None:
        response = await async_get(
            async_get_clientsession(self.hass),
            ENTSOE_API,
            "ENTSO-E",
            params={
                "documentType": "A44",
                "in_Domain": self.eic,
                "out_Domain": self.eic,
                "periodStart": start.astimezone(UTC).strftime("%Y%m%d%H%M"),
                "periodEnd": end.astimezone(UTC).strftime("%Y%m%d%H%M"),
                "securityToken": self._entsoe_api_key or "",
            },
        )
        # "No matching data" is an acknowledgement with status 200 or 400
        if response is None or response.status not in (200, 400):
            if response is not None:
                _LOGGER.warning("ENTSO-E returned %d", response.status)
            return None
        try:
            return parse_entsoe(response.text)
        except ValueError as err:
            _LOGGER.warning("Unusable ENTSO-E response: %s", err)
            return None
