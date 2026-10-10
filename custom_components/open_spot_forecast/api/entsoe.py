"""ENTSO-E Transparency Platform: the request and XML helpers (#27, #30).

OSF reads two ENTSO-E series: the day-ahead prices (``documentType=A44``,
the fallback of ``dayahead_prices``) and the week-ahead load forecast
(``A65``/``A31``, ``entsoe_load``). Both are XML documents of
``TimeSeries`` holding ``Period``s of ``Point``s. The outage documents
(``A77``/``A80``/``A78``, #138, ``entsoe_outages``) come as a ZIP of XML
documents, so ``async_entsoe_get_bytes`` returns the raw body. The
security token is a query parameter; ``async_get`` never logs URLs or
parameters.
"""

from __future__ import annotations

import logging

# The stdlib parser: its expat guards against entity expansion, and ENTSO-E is
# an official HTTPS source (defusedxml is not a dependency)
import xml.etree.ElementTree as ET  # nosec B405
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from typing import Any

from homeassistant.core import HomeAssistant
from homeassistant.helpers.aiohttp_client import async_get_clientsession

from ..const import ENTSOE_API
from .http import HttpResponse, async_get

_LOGGER = logging.getLogger(__name__)

_UNITS = {"M": timedelta(minutes=1), "H": timedelta(hours=1), "D": timedelta(days=1)}


def local_name(element: ET.Element) -> str:
    """Return an element's tag without its XML namespace."""
    return element.tag.rsplit("}", 1)[-1]


def child(element: ET.Element, name: str) -> ET.Element | None:
    """Return an element's first child named ``name``."""
    return next((c for c in element if local_name(c) == name), None)


def children(element: ET.Element, name: str) -> Iterator[ET.Element]:
    """Yield an element's children named ``name``."""
    return (c for c in element if local_name(c) == name)


def child_text(element: ET.Element, *path: str) -> str | None:
    """Return the text at a path of child names, or None if it is missing."""
    node: ET.Element | None = element
    for name in path:
        node = child(node, name) if node is not None else None
    return node.text if node is not None else None


def parse_resolution(value: str | None) -> timedelta | None:
    """Return an ISO 8601 duration such as ``PT15M``, ``PT60M`` or ``P1D``."""
    if not value or not value.startswith("P"):
        return None
    number, unit = value.removeprefix("PT").removeprefix("P")[:-1], value[-1]
    time_part = value.startswith("PT")
    if not number.isdigit() or unit not in _UNITS or time_part == (unit == "D"):
        return None
    return int(number) * _UNITS[unit]


def time_series(text: str) -> list[ET.Element]:
    """Return the ``TimeSeries`` elements of an ENTSO-E document.

    An acknowledgement ("no matching data") has none.

    Raises:
        ValueError: If the text is not XML.
    """
    try:
        root = ET.fromstring(text)  # nosec B314
    except ET.ParseError as err:
        raise ValueError("ENTSO-E response is not XML") from err
    return [e for e in root.iter() if local_name(e) == "TimeSeries"]


def period_bounds(
    period: ET.Element,
) -> tuple[datetime, datetime | None, timedelta] | None:
    """Return a ``Period``'s start, end (if given) and resolution, or None."""
    start_text = child_text(period, "timeInterval", "start")
    step = parse_resolution(child_text(period, "resolution"))
    if not start_text or step is None:
        return None
    end_text = child_text(period, "timeInterval", "end")
    end = datetime.fromisoformat(end_text) if end_text else None
    return datetime.fromisoformat(start_text), end, step


def entsoe_period(start: datetime, end: datetime) -> dict[str, str]:
    """Return the ``periodStart``/``periodEnd`` parameters (UTC, minutes)."""
    return {
        "periodStart": start.astimezone(UTC).strftime("%Y%m%d%H%M"),
        "periodEnd": end.astimezone(UTC).strftime("%Y%m%d%H%M"),
    }


async def _async_entsoe_response(
    hass: HomeAssistant, api_key: str, params: dict[str, Any]
) -> HttpResponse | None:
    """Request an ENTSO-E document; None if the request failed.

    "No matching data" is an acknowledgement document with status 200 or
    400, which the parsers read as no rows.
    """
    response = await async_get(
        async_get_clientsession(hass),
        ENTSOE_API,
        "ENTSO-E",
        params={**params, "securityToken": api_key},
    )
    if response is None or response.status not in (200, 400):
        if response is not None:
            _LOGGER.warning("ENTSO-E returned %d", response.status)
        return None
    return response


async def async_entsoe_get(
    hass: HomeAssistant, api_key: str, params: dict[str, Any]
) -> str | None:
    """Request an ENTSO-E document; return its text, or None if it failed."""
    response = await _async_entsoe_response(hass, api_key, params)
    return response.text if response is not None else None


async def async_entsoe_get_bytes(
    hass: HomeAssistant, api_key: str, params: dict[str, Any]
) -> bytes | None:
    """Request an ENTSO-E document; return its raw body (a ZIP or XML), or None."""
    response = await _async_entsoe_response(hass, api_key, params)
    return response.body if response is not None else None
