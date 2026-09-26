"""The shared time-series source plumbing (#32), on a keyed forecast table.

A fake source over a keyed table (two sampling points per timestamp, like
multi-point weather) exercises the hooks the Nordpool source does not use:
a refresh window for revised forecasts, request-size limits and keys.
"""

from collections.abc import Callable, Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

import pytest

from custom_components.open_spot_forecast.api.time_series_source import (
    TimeSeriesSource,
)
from custom_components.open_spot_forecast.ml.series_storage import SeriesSpec
from custom_components.open_spot_forecast.ml.storage import LearningStorage

SPEC = SeriesSpec(
    name="test_weather",
    table="test_weather",
    columns=("wind", "temperature"),
    step_minutes=15,
    key_column="point",
)
POINTS = ("a", "b")
NOW = datetime(2026, 9, 24, 10, tzinfo=UTC)
QUARTER = timedelta(minutes=15)


class FakeWeather(TimeSeriesSource):
    """Returns both points for every requested quarter; records requests."""

    spec = SPEC
    max_request_span = timedelta(hours=1)

    def __init__(self, *args: Any, refresh: timedelta | None = None, **kw: Any) -> None:
        super().__init__(*args, **kw)
        self.refresh = refresh
        self.requests: list[tuple[datetime, datetime]] = []
        self.wind = 5.0

    def keys(self) -> tuple[str, ...]:
        return POINTS

    def refresh_from(self, now: datetime) -> datetime | None:
        return None if self.refresh is None else now - self.refresh

    async def _fetch(
        self, start: datetime, end: datetime
    ) -> list[dict[str, Any]] | None:
        self.requests.append((start, end))
        rows = []
        moment = start
        while moment < end:
            for point in POINTS:
                rows.append(
                    {
                        "timestamp": moment.isoformat(),
                        "point": point,
                        "wind": self.wind,
                        "temperature": 12.0,
                    }
                )
            moment += QUARTER
        return rows


@pytest.fixture
def storage(tmp_path: Path) -> Iterator[LearningStorage]:
    hass = Mock()
    hass.config.path.return_value = str(tmp_path / ".storage")
    store = LearningStorage(hass, "DK1")
    store._ensure_conn().execute(
        """CREATE TABLE test_weather (
               timestamp TEXT, point TEXT, wind REAL, temperature REAL,
               PRIMARY KEY (timestamp, point))"""
    )
    yield store
    store.close()


@pytest.fixture(autouse=True)
def clock() -> Iterator[None]:
    with patch("homeassistant.util.dt.utcnow", return_value=NOW):
        yield


def _source(storage: LearningStorage, **kwargs: Any) -> FakeWeather:
    async def run_inline(func: Callable[..., Any], *args: Any) -> Any:
        return func(*args)

    hass = Mock()
    hass.async_add_executor_job = run_inline
    return FakeWeather(hass, storage, **kwargs)


@pytest.mark.asyncio
async def test_requests_are_split_to_the_maximum_span(storage: LearningStorage) -> None:
    source = _source(storage)

    assert await source.async_update(NOW, NOW + timedelta(hours=2, minutes=30))

    assert source.requests == [
        (NOW, NOW + timedelta(hours=1)),
        (NOW + timedelta(hours=1), NOW + timedelta(hours=2)),
        (NOW + timedelta(hours=2), NOW + timedelta(hours=2, minutes=30)),
    ]


@pytest.mark.asyncio
async def test_a_point_missing_its_row_makes_the_timestamp_missing(
    storage: LearningStorage,
) -> None:
    source = _source(storage)
    await source.async_update(NOW, NOW + timedelta(hours=1))
    storage._ensure_conn().execute(
        "DELETE FROM test_weather WHERE point = 'b' AND timestamp = ?",
        ((NOW + QUARTER).strftime("%Y-%m-%dT%H:%M:%SZ"),),
    )
    source.requests.clear()

    assert await source.async_update(NOW, NOW + timedelta(hours=1)) is True

    assert source.requests == [(NOW + QUARTER, NOW + 2 * QUARTER)]


@pytest.mark.asyncio
async def test_the_refresh_window_is_fetched_again_and_reports_changes(
    storage: LearningStorage,
) -> None:
    """Revised forecasts from (now - refresh) on are re-fetched every update."""
    source = _source(storage, refresh=timedelta(hours=1))
    start = NOW - timedelta(hours=3)
    end = NOW + timedelta(hours=1)
    await source.async_update(start, end)
    source.requests.clear()

    # Unchanged forecast: re-fetched, but nothing changed
    assert await source.async_update(start, end) is False
    assert source.requests == [
        (NOW - timedelta(hours=1), NOW),
        (NOW, NOW + timedelta(hours=1)),
    ]

    source.wind = 7.0
    assert await source.async_update(start, end) is True
    winds = {row["wind"] for row in storage.load_series(SPEC, start, end)}
    assert winds == {5.0, 7.0}


@pytest.mark.asyncio
async def test_an_empty_or_backwards_range_does_nothing(
    storage: LearningStorage,
) -> None:
    source = _source(storage)

    assert await source.async_update(NOW, NOW) is False
    assert await source.async_update(NOW, NOW - QUARTER) is False
    assert source.requests == []
