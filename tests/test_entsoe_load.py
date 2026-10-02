"""ENTSO-E's week-ahead load forecast as a 15-minute curve (#30)."""

from collections.abc import Callable, Iterator
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, Mock, patch
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from custom_components.open_spot_forecast.api.entsoe import parse_resolution
from custom_components.open_spot_forecast.api.entsoe_load import (
    EntsoeLoadSource,
    load_curve,
    natural_cubic_spline,
    parse_entsoe_load,
)
from custom_components.open_spot_forecast.api.http import HttpResponse
from custom_components.open_spot_forecast.const import ENTSOE_LOAD_REGIONS
from custom_components.open_spot_forecast.ml.series_storage import ENTSOE_LOAD
from custom_components.open_spot_forecast.ml.storage import LearningStorage

ENTSOE_MODULE = "custom_components.open_spot_forecast.api.entsoe"
SOURCE = "custom_components.open_spot_forecast.api.time_series_source"
TZ = ZoneInfo("Europe/Copenhagen")
KEY = "entsoe-secret-token"
# Monday 2026-09-28 .. Sunday 2026-10-04 in Copenhagen (CEST, UTC+2)
WEEK_START = datetime(2026, 9, 27, 22, tzinfo=UTC)
NOW = datetime(2026, 9, 27, 10, tzinfo=UTC)
LOWS = [2100.0, 2150.0, 2200.0, 2250.0, 2300.0, 2000.0, 1900.0]
HIGHS = [3800.0, 3850.0, 3900.0, 3950.0, 4000.0, 3300.0, 3100.0]

SERIES = """
  <TimeSeries>
    <mRID>{business}</mRID>
    <businessType>{business}</businessType>
    <outBiddingZone_Domain.mRID codingScheme="A01">10YDK-1--------W</outBiddingZone_Domain.mRID>
    <quantity_Measure_Unit.name>MAW</quantity_Measure_Unit.name>
    <curveType>A01</curveType>
    <Period>
      <timeInterval><start>{start}</start><end>{end}</end></timeInterval>
      <resolution>{resolution}</resolution>
      {points}
    </Period>
  </TimeSeries>"""
DOCUMENT = """<?xml version="1.0" encoding="utf-8"?>
<GL_MarketDocument xmlns="urn:iec62325.351:tc57wg16:451-6:generationloaddocument:3:0">
  <type>A65</type>
  <process.processType>A31</process.processType>{series}
</GL_MarketDocument>"""
NO_DATA = """<?xml version="1.0" encoding="utf-8"?>
<Acknowledgement_MarketDocument xmlns="urn:iec62325.351:tc57wg16:451-1:acknowledgementdocument:7:0">
  <Reason><code>999</code><text>No matching data found</text></Reason>
</Acknowledgement_MarketDocument>"""


def _series(
    business: str,
    values: list[float],
    start: datetime = WEEK_START,
    resolution: str = "P1D",
    step: timedelta = timedelta(days=1),
) -> str:
    points = "".join(
        f"<Point><position>{i + 1}</position><quantity>{v}</quantity></Point>"
        for i, v in enumerate(values)
    )
    return SERIES.format(
        business=business,
        start=start.strftime("%Y-%m-%dT%H:%MZ"),
        end=(start + len(values) * step).strftime("%Y-%m-%dT%H:%MZ"),
        resolution=resolution,
        points=points,
    )


def _week(start: datetime = WEEK_START, days: int = 7) -> str:
    return DOCUMENT.format(
        series=_series("A60", LOWS[:days], start) + _series("A61", HIGHS[:days], start)
    )


def _local(day: date, hour: int, minute: int = 0) -> str:
    """The stored UTC key of a local wall-clock time."""
    moment = datetime.combine(day, time(hour, minute), TZ)
    return moment.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


# --- Parsing -------------------------------------------------------------------------


def test_daily_minimum_and_maximum_per_local_day() -> None:
    days = parse_entsoe_load(_week(), TZ)

    assert list(days) == [date(2026, 9, 28) + timedelta(days=n) for n in range(7)]
    assert days[date(2026, 9, 28)] == (LOWS[0], HIGHS[0])
    assert days[date(2026, 10, 4)] == (LOWS[6], HIGHS[6])


def test_a_day_needs_both_values_and_other_series_are_ignored() -> None:
    text = DOCUMENT.format(
        series=_series("A60", LOWS[:3])
        + _series("A61", HIGHS[:2])
        + _series("A04", [9999.0] * 3)
    )

    assert parse_entsoe_load(text, TZ) == {
        date(2026, 9, 28): (LOWS[0], HIGHS[0]),
        date(2026, 9, 29): (LOWS[1], HIGHS[1]),
    }


def test_finer_points_are_combined_per_local_day() -> None:
    """Hourly points: the day's lowest minimum and highest maximum."""
    hour = timedelta(hours=1)
    lows = [3000.0] * 24 + [2500.0] * 24
    highs = [3500.0] * 23 + [4100.0] + [3900.0] * 24
    text = DOCUMENT.format(
        series=_series("A60", lows, resolution="PT60M", step=hour)
        + _series("A61", highs, resolution="PT60M", step=hour)
    )

    assert parse_entsoe_load(text, TZ) == {
        date(2026, 9, 28): (3000.0, 4100.0),
        date(2026, 9, 29): (2500.0, 3900.0),
    }


def test_no_data_and_bad_documents() -> None:
    assert parse_entsoe_load(NO_DATA, TZ) == {}
    with pytest.raises(ValueError, match="not XML"):
        parse_entsoe_load("<html>", TZ)


def _blocks(business: str, points: dict[int, float], days: int = 7) -> str:
    """A variable sized block (A03) series: equal points are left out."""
    xml = "".join(
        f"<Point><position>{position}</position><quantity>{value}</quantity></Point>"
        for position, value in sorted(points.items())
    )
    return SERIES.format(
        business=business,
        start=WEEK_START.strftime("%Y-%m-%dT%H:%MZ"),
        end=(WEEK_START + timedelta(days=days)).strftime("%Y-%m-%dT%H:%MZ"),
        resolution="P1D",
        points=xml,
    ).replace("<curveType>A01</curveType>", "<curveType>A03</curveType>")


def test_a_left_out_block_position_repeats_the_previous_value() -> None:
    """ENTSO-E's live documents are A03: positions 2 and 7 are left out."""
    text = DOCUMENT.format(
        series=_blocks("A60", {1: 2100.0, 3: 2200.0, 4: 2250.0, 5: 2300.0, 6: 2000.0})
        + _blocks("A61", {1: 3800.0, 3: 3900.0, 4: 3950.0, 5: 4000.0, 6: 3300.0})
    )

    days = parse_entsoe_load(text, TZ)

    monday = date(2026, 9, 28)
    assert list(days) == [monday + timedelta(days=n) for n in range(7)]
    assert days[monday + timedelta(days=1)] == (2100.0, 3800.0)
    assert days[monday + timedelta(days=6)] == (2000.0, 3300.0)


def test_a_left_out_position_is_not_filled_in_a_point_curve() -> None:
    """In an A01 series every point is given: a gap stays a gap."""
    text = DOCUMENT.format(
        series=_series("A60", LOWS[:3]).replace(
            "<position>2</position>", "<position>5</position>"
        )
        + _series("A61", HIGHS[:3]).replace(
            "<position>2</position>", "<position>5</position>"
        )
    )

    assert list(parse_entsoe_load(text, TZ)) == [
        date(2026, 9, 28),
        date(2026, 9, 30),
        date(2026, 10, 2),
    ]


def test_a_day_without_a_positive_load_is_missing() -> None:
    """ENTSO-E publishes 0 for a day it has no forecast for (DK1, 2026-09-06)."""
    lows = [*LOWS[:5], 0.0, LOWS[6]]
    highs = [*HIGHS[:5], 0.0, HIGHS[6]]
    text = DOCUMENT.format(series=_series("A60", lows) + _series("A61", highs))

    days = parse_entsoe_load(text, TZ)

    assert date(2026, 10, 3) not in days
    assert len(days) == 6
    assert min(row["load"] for row in load_curve(days, TZ)) >= min(LOWS) - 1
    # ... and so is a day whose minimum is above its maximum
    swapped = DOCUMENT.format(
        series=_series("A60", HIGHS[:2]) + _series("A61", LOWS[:2])
    )
    assert parse_entsoe_load(swapped, TZ) == {}


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("PT15M", timedelta(minutes=15)),
        ("PT60M", timedelta(hours=1)),
        ("PT1H", timedelta(hours=1)),
        ("P1D", timedelta(days=1)),
        ("P7D", timedelta(days=7)),
        ("PT1D", None),
        ("P15M", None),
        ("PT15X", None),
        ("", None),
        (None, None),
    ],
)
def test_resolutions(value: str | None, expected: timedelta | None) -> None:
    assert parse_resolution(value) == expected


# --- The curve -----------------------------------------------------------------------


def test_the_spline_passes_through_its_points_and_keeps_straight_lines() -> None:
    x = np.array([0.0, 3.0, 4.5, 9.0, 13.0])
    y = np.array([5.0, 1.0, 7.0, 2.0, 4.0])

    assert natural_cubic_spline(x, y, x) == pytest.approx(y)
    at = np.linspace(0, 13, 27)
    assert natural_cubic_spline(x, 2 * x + 1, at) == pytest.approx(2 * at + 1)


def test_the_curve_hits_the_anchors_in_local_time() -> None:
    days = {date(2026, 9, 28): (2000.0, 4000.0), date(2026, 9, 29): (2400.0, 3600.0)}

    load = {row["timestamp"]: row["load"] for row in load_curve(days, TZ)}

    monday, tuesday = date(2026, 9, 28), date(2026, 9, 29)
    assert load[_local(monday, 3)] == pytest.approx(2000.0)
    assert load[_local(monday, 11, 30)] == pytest.approx(4000.0)
    assert load[_local(monday, 14, 30)] == pytest.approx((3 * 4000 + 2000) / 4)
    assert load[_local(monday, 19)] == pytest.approx(4000.0)
    assert load[_local(tuesday, 3)] == pytest.approx(2400.0)
    assert load[_local(tuesday, 19)] == pytest.approx(3600.0)
    # From the first anchor to the last, every 15 minutes
    assert min(load) == _local(monday, 3) and max(load) == _local(tuesday, 19)
    assert len(load) == (24 + 16) * 4 + 1
    # The night is the low, the evening peak the high
    assert load[_local(monday, 23)] < load[_local(monday, 19)]


def test_the_curve_follows_the_local_clock_on_dst_days() -> None:
    """On 2026-10-25 (25 hours) the anchors stay at local 03:00 and 19:00."""
    sunday = date(2026, 10, 25)
    days = {sunday + timedelta(days=n): (2000.0 + n, 3000.0 + n) for n in (-1, 0, 1)}

    load = {row["timestamp"]: row["load"] for row in load_curve(days, TZ)}

    assert load[_local(sunday, 3)] == pytest.approx(2000.0)
    assert load[_local(sunday, 19)] == pytest.approx(3000.0)
    assert _local(sunday, 19) == "2026-10-25T18:00:00Z"


def test_a_missing_day_splits_the_curve() -> None:
    days = {date(2026, 9, 28): (2000.0, 4000.0), date(2026, 9, 30): (2000.0, 4000.0)}

    load = {row["timestamp"] for row in load_curve(days, TZ)}

    assert _local(date(2026, 9, 28), 19) in load
    assert _local(date(2026, 9, 29), 12) not in load
    assert _local(date(2026, 9, 30), 3) in load
    assert load_curve({}, TZ) == []


# --- The source ----------------------------------------------------------------------


@pytest.fixture
def storage(tmp_path: Path) -> Iterator[LearningStorage]:
    hass = Mock()
    hass.config.path.return_value = str(tmp_path / ".storage")
    store = LearningStorage(hass, "DK1")
    yield store
    store.close()


def _period_start(params: dict) -> datetime:
    return datetime.strptime(params["periodStart"], "%Y%m%d%H%M").replace(tzinfo=UTC)


class FakeEntsoe:
    """ENTSO-E's answers, recording every request (one ISO week each)."""

    def __init__(self) -> None:
        self.answer: Callable[[dict], HttpResponse | None] = lambda p: HttpResponse(
            200, _week(_period_start(p), 7)
        )
        self.calls: list[tuple[str, dict]] = []

    async def get(self, _session: Any, _url: str, label: str, **kw: Any) -> Any:
        self.calls.append((label, kw["params"]))
        return self.answer(kw["params"])


@pytest.fixture
def entsoe() -> Iterator[FakeEntsoe]:
    fake = FakeEntsoe()
    with (
        patch(f"{ENTSOE_MODULE}.async_get", new=fake.get),
        patch(f"{ENTSOE_MODULE}.async_get_clientsession"),
        patch(f"{SOURCE}.asyncio.sleep", new=AsyncMock()),
        patch("homeassistant.util.dt.utcnow", return_value=NOW),
    ):
        yield fake


def _source(storage: LearningStorage) -> EntsoeLoadSource:
    async def run_inline(func: Callable[..., Any], *args: Any) -> Any:
        return func(*args)

    hass = Mock()
    hass.async_add_executor_job = run_inline
    return EntsoeLoadSource(hass, storage, "DK1", KEY)


def _request(monday: datetime, days: int = 7) -> tuple[str, dict]:
    return (
        "ENTSO-E",
        {
            "documentType": "A65",
            "processType": "A31",
            "outBiddingZone_Domain": "10YDK-1--------W",
            "periodStart": monday.strftime("%Y%m%d%H%M"),
            "periodEnd": (monday + timedelta(days=days)).strftime("%Y%m%d%H%M"),
            "securityToken": KEY,
        },
    )


@pytest.mark.asyncio
async def test_a_range_is_fetched_as_its_week_and_the_weeks_on_each_side(
    storage: LearningStorage, entsoe: FakeEntsoe
) -> None:
    """Monday to Wednesday: ENTSO-E answers one ISO week per request, so the
    whole week is stored; the curve needs the previous Sunday's and the next
    Monday's anchors, so the weeks on each side are fetched too."""
    start, end = WEEK_START, WEEK_START + timedelta(days=3)

    assert await _source(storage).async_update(start, end) is True

    week = timedelta(weeks=1)
    assert entsoe.calls == [
        _request(WEEK_START - week),
        _request(WEEK_START),
        _request(WEEK_START + week),
    ]
    rows = storage.load_series(ENTSOE_LOAD, start - week, end + week)
    assert rows[0]["timestamp"] == "2026-09-27T22:00:00Z"
    assert rows[-1]["timestamp"] == "2026-10-04T21:45:00Z"
    assert len(rows) == 7 * 96
    assert all(1500 < row["load"] < 4500 for row in rows)


@pytest.mark.asyncio
async def test_a_week_is_requested_once_per_update(
    storage: LearningStorage, entsoe: FakeEntsoe
) -> None:
    """Two weeks: the weeks shared by neighbouring requests are not fetched twice."""
    start, end = WEEK_START, WEEK_START + timedelta(days=10)

    assert await _source(storage).async_update(start, end) is True

    week = timedelta(weeks=1)
    assert [c[1]["periodStart"] for c in entsoe.calls] == [
        (WEEK_START + n * week).strftime("%Y%m%d%H%M") for n in (-1, 0, 1, 2)
    ]
    rows = storage.load_series(ENTSOE_LOAD, start, end)
    assert len(rows) == 10 * 96


@pytest.mark.asyncio
async def test_a_missing_neighbouring_week_ends_the_curve_at_its_anchors(
    storage: LearningStorage, entsoe: FakeEntsoe
) -> None:
    """Only the week itself is published: the curve runs from Monday 03:00
    to Sunday 19:00, and the edges stay holes to be asked for again."""
    entsoe.answer = lambda p: (
        HttpResponse(200, _week(WEEK_START, 7))
        if _period_start(p) == WEEK_START
        else HttpResponse(400, NO_DATA)
    )
    start, end = WEEK_START, WEEK_START + timedelta(weeks=1)

    assert await _source(storage).async_update(start, end) is True

    rows = storage.load_series(ENTSOE_LOAD, start, end)
    assert rows[0]["timestamp"] == _local(date(2026, 9, 28), 3)
    assert rows[-1]["timestamp"] == _local(date(2026, 10, 4), 19)
    assert len(rows) == 7 * 96 - 3 * 4 - 5 * 4 + 1


@pytest.mark.parametrize(
    "answer",
    [
        lambda _p: None,
        lambda _p: HttpResponse(401, ""),
        lambda _p: HttpResponse(200, "<x"),
    ],
)
@pytest.mark.asyncio
async def test_failed_requests_store_nothing(
    storage: LearningStorage,
    entsoe: FakeEntsoe,
    answer: Callable[[dict], HttpResponse | None],
) -> None:
    entsoe.answer = answer

    assert (
        await _source(storage).async_update(WEEK_START, WEEK_START + timedelta(days=1))
        is False
    )
    assert (
        storage.load_series(ENTSOE_LOAD, WEEK_START, WEEK_START + timedelta(days=1))
        == []
    )


@pytest.mark.asyncio
async def test_no_matching_data_is_a_hole_not_a_failure(
    storage: LearningStorage, entsoe: FakeEntsoe
) -> None:
    """ENTSO-E's 400 acknowledgement: nothing published, asked again later."""
    entsoe.answer = lambda _p: HttpResponse(400, NO_DATA)
    source = _source(storage)
    end = WEEK_START + timedelta(days=1)

    assert await source.async_update(WEEK_START, end) is False
    assert await source.async_update(WEEK_START, end) is False

    # The week and its neighbours once; the hole is not asked for again yet
    assert len(entsoe.calls) == 3


def test_the_source_refreshes_from_yesterday() -> None:
    source = EntsoeLoadSource(Mock(), Mock(), "DK2", KEY)

    assert source.refresh_from(NOW) == NOW - timedelta(days=1)
    assert source.eic == "10YDK-2--------M"


def test_germany_and_the_netherlands_do_not_use_it() -> None:
    assert {"DK1", "DK2", "SE3", "FI"} <= ENTSOE_LOAD_REGIONS
    assert not {"DE", "NL"} & ENTSOE_LOAD_REGIONS
