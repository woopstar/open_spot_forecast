"""Day-ahead prices from energy-charts.info with ENTSO-E as fallback (#27)."""

import json
from collections.abc import Callable, Iterator
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest

from custom_components.open_spot_forecast.api.dayahead_prices import (
    DayAheadPriceSource,
    parse_energy_charts,
    parse_entsoe,
)
from custom_components.open_spot_forecast.api.http import HttpResponse
from custom_components.open_spot_forecast.ml.series_storage import DAYAHEAD_PRICES
from custom_components.open_spot_forecast.ml.storage import LearningStorage

MODULE = "custom_components.open_spot_forecast.api.dayahead_prices"
SOURCE = "custom_components.open_spot_forecast.api.time_series_source"
KEY = "entsoe-secret-token"
# 2026-09-24 in Copenhagen (CEST): 22:00Z to 22:00Z
DAY_START = datetime(2026, 9, 23, 22, tzinfo=UTC)
DAY_END = datetime(2026, 9, 24, 22, tzinfo=UTC)
NOW = datetime(2026, 9, 24, 8, tzinfo=UTC)
QUARTER = timedelta(minutes=15)


def _payload(start: datetime, count: int, step: int = 900) -> dict[str, Any]:
    first = int(start.timestamp())
    return {
        "license_info": "CC BY 4.0 from Bundesnetzagentur | SMARD.de",
        "unix_seconds": [first + i * step for i in range(count)],
        "price": [50.0 + i for i in range(count)],
        "unit": "EUR / MWh",
    }


ENTSOE = """<?xml version="1.0" encoding="utf-8"?>
<Publication_MarketDocument xmlns="urn:iec62325.351:tc57wg16:451-3:publicationdocument:7:3">
  <TimeSeries>
    <curveType>{curve}</curveType>
    <Period>
      <timeInterval><start>{start}</start><end>{end}</end></timeInterval>
      <resolution>{resolution}</resolution>
      <Point><position>1</position><price.amount>50.5</price.amount></Point>
      <Point><position>3</position><price.amount>60</price.amount></Point>
    </Period>
    <Period>
      <timeInterval><start>{start}</start></timeInterval>
      <resolution>PT15M</resolution>
    </Period>
  </TimeSeries>
</Publication_MarketDocument>"""
NO_DATA = """<?xml version="1.0" encoding="utf-8"?>
<Acknowledgement_MarketDocument xmlns="urn:iec62325.351:tc57wg16:451-1:acknowledgementdocument:7:0">
  <Reason><code>999</code><text>No matching data found</text></Reason>
</Acknowledgement_MarketDocument>"""


def _entsoe(
    start: datetime, end: datetime, curve: str = "A03", res: str = "PT15M"
) -> str:
    return ENTSOE.format(
        curve=curve,
        resolution=res,
        start=start.strftime("%Y-%m-%dT%H:%MZ"),
        end=end.strftime("%Y-%m-%dT%H:%MZ"),
    )


# --- Parsing -------------------------------------------------------------------------


def test_energy_charts_hourly_prices_fill_their_quarters() -> None:
    start = DAY_START
    payload = _payload(start, 2, step=3600)
    payload["unix_seconds"].append(payload["unix_seconds"][-1] + 3600)
    payload["price"].append(None)

    rows = parse_energy_charts(payload)

    assert [row["timestamp"] for row in rows] == [
        (start + i * QUARTER).isoformat() for i in range(8)
    ]
    assert [row["price"] for row in rows] == [50.0] * 4 + [51.0] * 4


def test_energy_charts_single_price_and_bad_payloads() -> None:
    assert len(parse_energy_charts(_payload(DAY_START, 1))) == 1
    with pytest.raises(ValueError, match="unit"):
        parse_energy_charts({**_payload(DAY_START, 1), "unit": "EUR / kWh"})
    with pytest.raises(ValueError, match="differ in length"):
        parse_energy_charts({**_payload(DAY_START, 2), "price": [1.0]})


def test_entsoe_variable_blocks_repeat_the_previous_price() -> None:
    rows = parse_entsoe(_entsoe(DAY_START, DAY_START + 2 * QUARTER * 4))

    assert [row["price"] for row in rows] == [50.5, 50.5] + [60.0] * 6
    assert rows[0]["timestamp"] == DAY_START.isoformat()


def test_entsoe_fixed_blocks_leave_missing_positions_out() -> None:
    rows = parse_entsoe(
        _entsoe(DAY_START, DAY_START + timedelta(hours=3), curve="A01", res="PT60M")
    )

    # Hourly positions 1 and 3 fill their four quarters each
    starts = [row["timestamp"] for row in rows]
    assert starts[:4] == [(DAY_START + i * QUARTER).isoformat() for i in range(4)]
    assert starts[4] == (DAY_START + timedelta(hours=2)).isoformat()
    assert len(rows) == 8


@pytest.mark.parametrize("res", ["PT15X", "P1D", ""])
def test_entsoe_periods_without_a_known_resolution_are_skipped(res: str) -> None:
    assert parse_entsoe(_entsoe(DAY_START, DAY_END, res=res)) == []


def test_entsoe_no_data_and_garbage() -> None:
    assert parse_entsoe(NO_DATA) == []
    with pytest.raises(ValueError, match="not XML"):
        parse_entsoe("<html>")


# --- The source ----------------------------------------------------------------------


@pytest.fixture
def storage(tmp_path: Path) -> Iterator[LearningStorage]:
    hass = Mock()
    hass.config.path.return_value = str(tmp_path / ".storage")
    store = LearningStorage(hass, "DK1")
    yield store
    store.close()


def _source(
    storage: LearningStorage, region: str = "DK1", key: str | None = None
) -> DayAheadPriceSource:
    async def run_inline(func: Callable[..., Any], *args: Any) -> Any:
        return func(*args)

    hass = Mock()
    hass.async_add_executor_job = run_inline
    return DayAheadPriceSource(hass, storage, region, key)


class FakeApis:
    """energy-charts and ENTSO-E answers, recording every request."""

    def __init__(self) -> None:
        self.energy_charts: Callable[[dict], HttpResponse | None] = lambda p: (
            HttpResponse(200, json.dumps(_payload(_start(p), 96)))
        )
        self.entsoe: Callable[[dict], HttpResponse | None] = lambda p: HttpResponse(
            200, _entsoe(DAY_START, DAY_END)
        )
        self.calls: list[tuple[str, dict]] = []

    async def get(self, _session: Any, _url: str, label: str, **kw: Any) -> Any:
        self.calls.append((label, kw["params"]))
        answer = self.energy_charts if label == "energy-charts" else self.entsoe
        return answer(kw["params"])


def _start(params: dict) -> datetime:
    return datetime.fromisoformat(params["start"])


@pytest.fixture
def apis() -> Iterator[FakeApis]:
    fake = FakeApis()
    with (
        patch(f"{MODULE}.async_get", new=fake.get),
        patch(f"{MODULE}.async_get_clientsession"),
        patch(f"{SOURCE}.asyncio.sleep", new=AsyncMock()),
        patch("homeassistant.util.dt.utcnow", return_value=NOW),
    ):
        yield fake


@pytest.mark.asyncio
async def test_a_local_day_is_fetched_and_stored(
    storage: LearningStorage, apis: FakeApis
) -> None:
    source = _source(storage)

    assert await source.async_update(DAY_START, DAY_END) is True

    assert apis.calls == [
        (
            "energy-charts",
            {
                "bzn": "DK1",
                "start": "2026-09-24T00:00+02:00",
                "end": "2026-09-24T23:45+02:00",
            },
        )
    ]
    assert len(storage.load_series(DAYAHEAD_PRICES, DAY_START, DAY_END)) == 96
    assert source.license_info is not None
    assert "SMARD" in source.license_info
    # Complete: no second request
    assert await source.async_update(DAY_START, DAY_END) is False
    assert len(apis.calls) == 1


@pytest.mark.asyncio
async def test_requests_span_local_days_east_of_utc(
    storage: LearningStorage, apis: FakeApis
) -> None:
    """Estonia's day starts at 21:00 UTC; a UTC window would cut off 3 hours."""
    source = _source(storage, "EE")
    # One missing hour late in the Estonian day of 2026-09-24
    start = datetime(2026, 9, 24, 18, tzinfo=UTC)

    await source.async_update(start, start + timedelta(hours=1))

    assert apis.calls[0][1] == {
        "bzn": "EE",
        "start": "2026-09-24T00:00+03:00",
        "end": "2026-09-24T23:45+03:00",
    }


def test_requests_cover_whole_local_days_in_31_day_runs(
    storage: LearningStorage,
) -> None:
    source = _source(storage)
    hour = timedelta(hours=1)
    fall_back = datetime(2026, 10, 24, 22, tzinfo=UTC)  # 2026-10-25 00:00 CEST

    chunks = source.chunks(
        [
            (DAY_START + hour, DAY_START + 2 * hour),
            (fall_back + hour, fall_back + 2 * hour),
            (datetime(2026, 11, 1, tzinfo=UTC), datetime(2026, 12, 15, tzinfo=UTC)),
        ]
    )

    assert chunks[0] == (DAY_START, DAY_END)
    # The fall-back day lasts 25 hours
    assert chunks[1] == (fall_back, fall_back + timedelta(hours=25))
    assert chunks[2][0] == datetime(2026, 10, 31, 23, tzinfo=UTC)
    assert chunks[2][1] - chunks[2][0] == timedelta(days=31)
    assert chunks[3][1] == datetime(2026, 12, 15, 23, tzinfo=UTC)


def test_tomorrow_is_asked_for_when_the_auction_results_are_due(
    storage: LearningStorage,
) -> None:
    source = _source(storage)
    tomorrow = (DAY_END, DAY_END + timedelta(days=1))

    assert source.retry_time(tomorrow, NOW) == datetime(2026, 9, 24, 10, 45, tzinfo=UTC)
    afternoon = datetime(2026, 9, 24, 11, tzinfo=UTC)
    assert source.retry_time(tomorrow, afternoon) == afternoon + timedelta(minutes=5)
    old = (DAY_START - timedelta(days=10), DAY_START - timedelta(days=9))
    assert source.retry_time(old, NOW) == NOW + timedelta(days=1)


@pytest.mark.asyncio
async def test_entsoe_fills_in_when_energy_charts_fails(
    storage: LearningStorage, apis: FakeApis, caplog: pytest.LogCaptureFixture
) -> None:
    apis.energy_charts = lambda _p: HttpResponse(503, "")
    source = _source(storage, key=KEY)

    assert await source.async_update(DAY_START, DAY_END) is True

    label, params = apis.calls[1]
    assert label == "ENTSO-E"
    assert params["in_Domain"] == params["out_Domain"] == "10YDK-1--------W"
    assert params["periodStart"] == "202609232200"
    assert params["periodEnd"] == "202609242200"
    assert params["securityToken"] == KEY
    assert len(storage.load_series(DAYAHEAD_PRICES, DAY_START, DAY_END)) == 96
    assert KEY not in caplog.text


@pytest.mark.asyncio
async def test_energy_charts_wins_where_both_have_prices(
    storage: LearningStorage, apis: FakeApis
) -> None:
    """energy-charts has only half the day; ENTSO-E adds the rest."""
    apis.energy_charts = lambda p: HttpResponse(
        200, json.dumps(_payload(_start(p), 48))
    )
    source = _source(storage, key=KEY)

    await source.async_update(DAY_START, DAY_END)

    rows = storage.load_series(DAYAHEAD_PRICES, DAY_START, DAY_END)
    assert len(rows) == 96
    assert rows[0]["price"] == pytest.approx(50.0)  # energy-charts
    assert rows[47]["price"] == pytest.approx(97.0)
    assert rows[48]["price"] == pytest.approx(60.0)  # ENTSO-E


@pytest.mark.asyncio
async def test_without_a_key_there_is_no_fallback(
    storage: LearningStorage, apis: FakeApis
) -> None:
    apis.energy_charts = lambda _p: None
    source = _source(storage)

    assert await source.async_update(DAY_START, DAY_END) is False

    assert [label for label, _ in apis.calls] == ["energy-charts"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("entsoe", "stored"),
    [
        (HttpResponse(401, "Unauthorized"), 0),
        (HttpResponse(400, NO_DATA), 0),
        (HttpResponse(200, "<html>"), 0),
        (None, 0),
    ],
)
async def test_failed_or_empty_fallbacks_store_nothing(
    storage: LearningStorage,
    apis: FakeApis,
    caplog: pytest.LogCaptureFixture,
    entsoe: HttpResponse | None,
    stored: int,
) -> None:
    apis.energy_charts = lambda _p: HttpResponse(200, "not json")
    apis.entsoe = lambda _p: entsoe
    source = _source(storage, key=KEY)

    await source.async_update(DAY_START, DAY_END)

    assert len(storage.load_series(DAYAHEAD_PRICES, DAY_START, DAY_END)) == stored
    assert "Unusable energy-charts response" in caplog.text
    assert KEY not in caplog.text


@pytest.mark.asyncio
async def test_an_unexpected_energy_charts_status_is_logged(
    storage: LearningStorage, apis: FakeApis, caplog: pytest.LogCaptureFixture
) -> None:
    apis.energy_charts = lambda _p: HttpResponse(404, "")

    await _source(storage).async_update(DAY_START, DAY_END)

    assert "energy-charts returned 404" in caplog.text
    assert date(2026, 9, 24).isoformat() not in caplog.text
