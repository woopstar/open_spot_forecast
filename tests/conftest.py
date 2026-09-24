"""Shared pytest fixtures."""

from collections.abc import Iterator
from zoneinfo import ZoneInfo

import pytest

from homeassistant.util import dt as dt_util


@pytest.fixture
def copenhagen_time_zone() -> Iterator[ZoneInfo]:
    """Set Home Assistant's time zone to Europe/Copenhagen for one test."""
    zone = ZoneInfo("Europe/Copenhagen")
    previous = dt_util.get_default_time_zone()
    dt_util.set_default_time_zone(zone)
    yield zone
    dt_util.set_default_time_zone(previous)
