"""ENTSO-E outage documents: parser, queries, index parity, source and backtest (#138)."""

from __future__ import annotations

import io
import json
import logging
import urllib.parse
import zipfile
from collections.abc import Callable, Iterator
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

import pytest

from custom_components.open_spot_forecast.api.entsoe_outages import (
    ENTSOE_DOCUMENT_LIMIT,
    EntsoeOutageSource,
    entsoe_outage_queries,
    outage_documents,
    parse_entsoe_outages,
    parse_outage_document,
)
from custom_components.open_spot_forecast.attribution import (
    ENTSOE_OUTAGE_ATTRIBUTION,
)
from custom_components.open_spot_forecast.const import (
    ENTSOE_OUTAGE_BORDERS,
    ENTSOE_OUTAGE_REGIONS,
    REGIONS,
    UMM_AREAS,
)
from custom_components.open_spot_forecast.ml.outages import OutageIndex
from custom_components.open_spot_forecast.ml.storage import LearningStorage
from scripts.backtest_entsoe_outages import load_entsoe_outages

MODULE = "custom_components.open_spot_forecast.api.entsoe_outages"
DE = "10Y1001A1001A82H"
DK1 = "10YDK-1--------W"
NL = "10YNL----------L"
NOW = datetime(2026, 10, 10, 12, tzinfo=UTC)
START, END = datetime(2026, 10, 1, tzinfo=UTC), datetime(2026, 10, 17, tzinfo=UTC)

_NS = "urn:iec62325.351:tc57wg16:451-6:outagedocument:3:0"


def _points(*points: tuple[int, float]) -> str:
    return "".join(
        f"<Point><position>{position}</position><quantity>{quantity}</quantity></Point>"
        for position, quantity in points
    )


def _period(start: str, end: str, *points: tuple[int, float]) -> str:
    return (
        "<Available_Period><timeInterval>"
        f"<start>{start}</start><end>{end}</end></timeInterval>"
        f"<resolution>PT1M</resolution>{_points(*points)}</Available_Period>"
    )


def _plant(
    *periods: str,
    nominal: float | None = 211.0,
    generation_unit: str | None = "11WD7VOLK5S-HKVM",
    business: str = "A53",
) -> str:
    """A production/generation unit series (A77: no generation unit)."""
    resource = "production_RegisteredResource"
    unit = f"{resource}.pSRType.powerSystemResources"
    parts = [
        f"<businessType>{business}</businessType>",
        f'<biddingZone_Domain.mRID codingScheme="A01">{DE}</biddingZone_Domain.mRID>',
        "<curveType>A03</curveType>",
        f'<{resource}.mRID codingScheme="A01">11WD7VOLK5S--KWI</{resource}.mRID>',
        f"<{resource}.name>Voelklingen</{resource}.name>",
    ]
    if generation_unit:
        parts.append(f'<{unit}.mRID codingScheme="A01">{generation_unit}</{unit}.mRID>')
    if nominal is not None:
        parts.append(f'<{unit}.nominalP unit="MAW">{nominal}</{unit}.nominalP>')
    return f"<TimeSeries><mRID>1</mRID>{''.join(parts)}{''.join(periods)}</TimeSeries>"


def _grid(*periods: str, asset: str | None = "11T0-0000-1516-C") -> str:
    """A transmission series between DE and DK1 (no nominal capacity)."""
    resource = (
        f'<Asset_RegisteredResource><mRID codingScheme="A01">{asset}</mRID>'
        "<name>AUDORF S-HANDEWITT</name>"
        "<asset_PSRType.psrType>B21</asset_PSRType.psrType></Asset_RegisteredResource>"
        if asset
        else ""
    )
    return (
        "<TimeSeries><mRID>1</mRID><businessType>A54</businessType>"
        f'<in_Domain.mRID codingScheme="A01">{DE}</in_Domain.mRID>'
        f'<out_Domain.mRID codingScheme="A01">{DK1}</out_Domain.mRID>'
        f"<curveType>A03</curveType>{resource}{''.join(periods)}</TimeSeries>"
    )


def _document(
    doc_type: str = "A80",
    message_id: str = "doc-a",
    revision: int = 1,
    created: str = "2026-10-05T13:29:50Z",
    status: str | None = None,
    *series: str,
) -> bytes:
    """One unavailability document, as the platform's ZIP members are."""
    doc_status = f"<docStatus><value>{status}</value></docStatus>" if status else ""
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        f'<Unavailability_MarketDocument xmlns="{_NS}">'
        f"<mRID>{message_id}</mRID><revisionNumber>{revision}</revisionNumber>"
        f"<type>{doc_type}</type><process.processType>A26</process.processType>"
        f"<createdDateTime>{created}</createdDateTime>"
        "<unavailability_Time_Period.timeInterval><start>2026-10-05T10:00Z</start>"
        "<end>2026-10-20T23:00Z</end></unavailability_Time_Period.timeInterval>"
        f"{doc_status}{''.join(series)}<Reason><code>A95</code></Reason>"
        "</Unavailability_MarketDocument>"
    ).encode()


def _zip(*documents: bytes) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for index, document in enumerate(documents):
            archive.writestr(f"{index + 1:03d}-UNAVAILABILITY.xml", document)
    return buffer.getvalue()


def _acknowledgement(reason: str) -> bytes:
    return (
        '<?xml version="1.0" encoding="UTF-8"?><Acknowledgement_MarketDocument '
        'xmlns="urn:iec62325.351:tc57wg16:451-1:acknowledgementdocument:7:0">'
        "<mRID>x</mRID><Reason><code>999</code>"
        f"<text>{reason}</text></Reason></Acknowledgement_MarketDocument>"
    ).encode()


NO_DATA = _acknowledgement(
    "No matching data found for Data item Unavailability of Production and "
    f"Generation Units [15.1.A&amp;B] (biddingZone_Domain: {DE})"
)
PLANT_PERIOD = _period("2026-10-05T10:00Z", "2026-10-07T10:00Z", (1, 197.0), (61, 0.0))
PLANT_DOC = _document(
    "A80", "doc-a", 1, "2026-10-05T13:29:50Z", None, _plant(PLANT_PERIOD)
)


# --- Parser --------------------------------------------------------------------------


def test_a_generation_unit_document_gives_one_row_per_point_segment() -> None:
    rows = parse_outage_document(PLANT_DOC)

    header = {
        "message_id": "doc-a",
        "version": 1,
        "published": "2026-10-05T13:29:50Z",
        "message_type": 1,
        "unavailability_type": 2,
        "status": 1,
        "kind": "production",
        "fuel_type": None,
        "unit": "11WD7VOLK5S-HKVM",
        "installed_mw": 211.0,
    }
    assert rows == [
        header
        | {
            "event_start": "2026-10-05T10:00:00Z",
            "event_stop": "2026-10-05T11:00:00Z",
            "unavailable_mw": pytest.approx(14.0),
        },
        header
        | {
            "event_start": "2026-10-05T11:00:00Z",
            "event_stop": "2026-10-07T10:00:00Z",
            "unavailable_mw": pytest.approx(211.0),
        },
    ]


def test_a_production_unit_document_names_the_unit_and_a_forced_one_its_type() -> None:
    period = _period("2026-10-05T10:00Z", "2026-10-06T10:00Z", (1, 2048.0))
    series = _plant(period, nominal=2142.0, generation_unit=None, business="A54")

    (row,) = parse_outage_document(
        _document("A77", "doc-b", 3, "2026-10-01T00:00Z", None, series)
    )

    assert row["unit"] == "11WD7VOLK5S--KWI"
    assert row["version"] == 3
    assert row["unavailability_type"] == 1
    assert row["unavailable_mw"] == pytest.approx(94.0)


def test_a_segment_at_full_capacity_and_a_unit_without_nominal_give_no_period() -> None:
    full = _period("2026-10-05T10:00Z", "2026-10-06T10:00Z", (1, 211.0))
    assert parse_outage_document(
        _document("A80", "doc-c", 1, NOW.isoformat(), None, _plant(full))
    ) == [
        {
            "message_id": "doc-c",
            "version": 1,
            "published": "2026-10-10T12:00:00Z",
            "message_type": 1,
            "unavailability_type": 2,
            "status": 1,
            "kind": None,
        }
    ]
    no_nominal = _plant(PLANT_PERIOD, nominal=None)
    rows = parse_outage_document(
        _document("A80", "doc-d", 1, NOW.isoformat(), None, no_nominal)
    )
    assert [row["kind"] for row in rows] == [None]


def test_a_grid_document_counts_the_asset_only_where_it_is_limited() -> None:
    constant = _period("2026-10-05T10:00Z", "2026-10-06T10:00Z", (1, 500.0))
    curve = _period(
        "2026-10-06T10:00Z",
        "2026-10-08T10:00Z",
        (1, 3500.0),
        (1441, 1875.0),
        (2161, 3500.0),
    )

    rows = parse_outage_document(
        _document("A78", "grid-a", 165, NOW.isoformat(), None, _grid(constant, curve))
    )

    assert [(r["event_start"], r["event_stop"], r["unavailable_mw"]) for r in rows] == [
        ("2026-10-05T10:00:00Z", "2026-10-06T10:00:00Z", 1.0),
        ("2026-10-07T10:00:00Z", "2026-10-07T22:00:00Z", 1.0),
    ]
    assert rows[0]["kind"] == "transmission"
    assert rows[0]["message_type"] == 3
    assert rows[0]["unavailability_type"] == 1
    assert rows[0]["unit"] == "11T0-0000-1516-C"
    assert rows[0]["installed_mw"] is None
    # A border without a named asset is keyed by its direction
    (row,) = parse_outage_document(
        _document(
            "A78", "grid-b", 1, NOW.isoformat(), None, _grid(constant, asset=None)
        )
    )
    assert row["unit"] == f"{DE}>{DK1}"


@pytest.mark.parametrize("status", ["A09", "A13"])
def test_a_cancelled_or_withdrawn_document_is_dismissed(status: str) -> None:
    rows = parse_outage_document(
        _document("A80", "doc-a", 4, NOW.isoformat(), status, _plant(PLANT_PERIOD))
    )

    assert {row["status"] for row in rows} == {3}
    assert len(rows) == 2


def test_other_documents_and_documents_missing_their_keys_give_no_rows() -> None:
    assert parse_outage_document(_document("A65", "load", 1, NOW.isoformat())) == []
    assert parse_outage_document(_document("A80", "", 1, NOW.isoformat())) == []
    assert parse_outage_document(_document("A80", "doc", 1, "yesterday")) == []
    assert parse_outage_document(_document("A80", "doc", "x", NOW.isoformat())) == []  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="not XML"):
        parse_outage_document(b"<broken")


def test_a_response_is_a_zip_a_single_document_or_an_acknowledgement() -> None:
    other = _document("A80", "doc-b", 1, NOW.isoformat(), "A09", _plant(PLANT_PERIOD))

    zipped = parse_entsoe_outages(_zip(PLANT_DOC, other))
    assert zipped.documents == 2
    assert [(r["message_id"], r["status"]) for r in zipped.rows] == [
        ("doc-a", 1),
        ("doc-a", 1),
        ("doc-b", 3),
        ("doc-b", 3),
    ]
    assert parse_entsoe_outages(PLANT_DOC) == (1, zipped.rows[:2])
    assert parse_entsoe_outages(NO_DATA) == (0, [])
    assert outage_documents(_zip()) == []


@pytest.mark.parametrize(
    ("body", "match"),
    [
        (b"PK\x03\x04 not an archive", "not a ZIP"),
        (b"{}", "not XML"),
        (
            _acknowledgement(
                "The number of instances (1237) exceeds the allowed maximum (200)"
            ),
            "rejected",
        ),
        (_acknowledgement(""), "no reason given"),
    ],
)
def test_unusable_responses_raise(body: bytes, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        parse_entsoe_outages(body)


def test_the_queries_cover_the_zone_and_every_border_in_both_directions() -> None:
    queries = entsoe_outage_queries(DE, (DK1, NL), START, END)

    assert [q["documentType"] for q in queries] == [
        "A77",
        "A80",
        "A78",
        "A78",
        "A78",
        "A78",
    ]
    assert queries[0] == {
        "documentType": "A77",
        "biddingZone_Domain": DE,
        "periodStart": "202610010000",
        "periodEnd": "202610170000",
    }
    assert queries[1]["biddingZone_Domain"] == DE
    assert [(q["in_Domain"], q["out_Domain"]) for q in queries[2:]] == [
        (DE, DK1),
        (DK1, DE),
        (DE, NL),
        (NL, DE),
    ]
    assert "PeriodStartUpdate" not in queries[0]

    since = entsoe_outage_queries(DE, (), START, END, NOW - timedelta(hours=1), NOW)
    assert len(since) == 2
    assert since[0]["PeriodStartUpdate"] == "202610101100"
    assert since[0]["PeriodEndUpdate"] == "202610101200"


# --- Index parity with the UMM rows ----------------------------------------------------


def test_the_index_sees_revisions_and_cancellations_like_umm_messages() -> None:
    revised = _document(
        "A80",
        "doc-a",
        2,
        "2026-10-06T08:00:00Z",
        None,
        _plant(_period("2026-10-05T10:00Z", "2026-10-07T10:00Z", (1, 100.0))),
    )
    cancelled = _document(
        "A80", "doc-a", 3, "2026-10-06T20:00:00Z", "A09", _plant(PLANT_PERIOD)
    )
    grid = _document(
        "A78",
        "grid",
        1,
        "2026-10-01T00:00:00Z",
        None,
        _grid(_period("2026-10-05T00:00Z", "2026-10-08T00:00Z", (1, 500.0))),
    )
    index = OutageIndex(
        parse_entsoe_outages(_zip(PLANT_DOC, revised, cancelled, grid)).rows
    )
    slot = datetime(2026, 10, 6, 12, tzinfo=UTC)

    # Revision 1 (14 then 211 MW), revision 2 (111 MW), then cancelled
    assert index.for_slot(slot, datetime(2026, 10, 5, 14, tzinfo=UTC)) == {
        "unavailable_production": pytest.approx(211.0),
        "unavailable_transmission": pytest.approx(1.0),
    }
    assert index.for_slot(slot, datetime(2026, 10, 6, 9, tzinfo=UTC)) == {
        "unavailable_production": pytest.approx(111.0),
        "unavailable_transmission": pytest.approx(1.0),
    }
    assert index.for_slot(slot, datetime(2026, 10, 7, tzinfo=UTC)) == {
        "unavailable_production": pytest.approx(0.0),
        "unavailable_transmission": pytest.approx(1.0),
    }
    # Before anything was published: nothing
    assert index.for_slot(slot, datetime(2026, 9, 1, tzinfo=UTC)) == {
        "unavailable_production": pytest.approx(0.0),
        "unavailable_transmission": pytest.approx(0.0),
    }


# --- The source ------------------------------------------------------------------------


@pytest.fixture
def storage(tmp_path: Path) -> Iterator[LearningStorage]:
    hass = Mock()
    hass.config.path.return_value = str(tmp_path)
    store = LearningStorage(hass, "DE")
    yield store
    store.close()


class FakeEntsoe:
    """The ENTSO-E API, answering by query and recording every request."""

    def __init__(self) -> None:
        self.calls: list[dict[str, str]] = []
        self.answer: Callable[[dict[str, str]], bytes | None] = lambda _q: NO_DATA

    async def get(
        self, _hass: Any, api_key: str, params: dict[str, Any]
    ) -> bytes | None:
        assert api_key == "token"
        self.calls.append(params)
        return self.answer(params)


@pytest.fixture
def entsoe() -> Iterator[FakeEntsoe]:
    fake = FakeEntsoe()
    with (
        patch(f"{MODULE}.async_entsoe_get_bytes", new=fake.get),
        patch("homeassistant.util.dt.utcnow", return_value=NOW),
    ):
        yield fake


def _source(storage: LearningStorage, region: str = "DE") -> EntsoeOutageSource:
    async def run_inline(func: Callable[..., Any], *args: Any) -> Any:
        return func(*args)

    hass = Mock()
    hass.async_add_executor_job = run_inline
    return EntsoeOutageSource(hass, storage, region, "token")


def _state(storage: LearningStorage) -> list[str]:
    return list(json.loads(storage.load_source_state("entsoe_outages") or ""))


@pytest.mark.asyncio
async def test_the_first_update_asks_every_query_of_the_window_and_stores_the_rows(
    storage: LearningStorage, entsoe: FakeEntsoe
) -> None:
    grid = _document(
        "A78",
        "grid",
        1,
        "2026-10-01T00:00:00Z",
        None,
        _grid(_period("2026-10-05T00:00Z", "2026-10-08T00:00Z", (1, 0.0))),
    )

    def answer(query: dict[str, str]) -> bytes:
        if query["documentType"] == "A80":
            return _zip(PLANT_DOC)
        if query.get("in_Domain") == DE and query["out_Domain"] == DK1:
            return grid
        return NO_DATA

    entsoe.answer = answer
    source = _source(storage)

    assert source.attribution == ENTSOE_OUTAGE_ATTRIBUTION
    assert await source.async_update(START, END) is True

    assert len(entsoe.calls) == 2 + 2 * len(ENTSOE_OUTAGE_BORDERS["DE"])
    assert all(call["offset"] == "0" for call in entsoe.calls)
    assert all("PeriodStartUpdate" not in call for call in entsoe.calls)
    assert entsoe.calls[0]["periodStart"] == "202610010000"
    rows = await source.async_load(START, END)
    assert {(r["message_id"], r["kind"]) for r in rows} == {
        ("doc-a", "production"),
        ("grid", "transmission"),
    }
    assert _state(storage) == [START.isoformat(), END.isoformat(), NOW.isoformat()]


@pytest.mark.asyncio
async def test_later_updates_fetch_the_publications_since_and_new_days_only(
    storage: LearningStorage, entsoe: FakeEntsoe
) -> None:
    source = _source(storage, "BE")
    await source.async_update(START, END)
    entsoe.calls.clear()
    later = NOW + timedelta(hours=6)

    with patch("homeassistant.util.dt.utcnow", return_value=later):
        assert await source.async_update(START, END + timedelta(days=1)) is False

    per_window = 2 + 2 * len(ENTSOE_OUTAGE_BORDERS["BE"])
    assert len(entsoe.calls) == 2 * per_window
    # The new day with every revision, then the covered window's publications
    new_day, covered = entsoe.calls[:per_window], entsoe.calls[per_window:]
    assert all(
        c["periodStart"] == "202610170000" and "PeriodStartUpdate" not in c
        for c in new_day
    )
    assert all(
        c["periodStart"] == "202610010000" and c["periodEnd"] == "202610170000"
        for c in covered
    )
    assert all(
        c["PeriodStartUpdate"] == "202610101200"
        and c["PeriodEndUpdate"] == "202610101800"
        for c in covered
    )
    assert _state(storage) == [
        START.isoformat(),
        (END + timedelta(days=1)).isoformat(),
        later.isoformat(),
    ]


@pytest.mark.asyncio
async def test_a_full_page_is_followed_by_the_next_offset_and_an_overflow_halves_the_window(
    storage: LearningStorage, entsoe: FakeEntsoe
) -> None:
    def answer(query: dict[str, str]) -> bytes:
        if query["documentType"] != "A80":
            return NO_DATA
        days = (int(query["periodEnd"][6:8]) - int(query["periodStart"][6:8])) or 1
        # Four documents per day: a 16-day window overflows the 2 x 2 limit
        documents = [
            _document(
                "A80",
                f"doc-{query['periodStart']}-{query['offset']}-{i}",
                1,
                NOW.isoformat(),
                None,
                _plant(PLANT_PERIOD),
            )
            for i in range(min(2, 4 * days - int(query["offset"])))
        ]
        return _zip(*documents)

    entsoe.answer = answer
    source = _source(storage, "FR")
    with (
        patch(f"{MODULE}.ENTSOE_DOCUMENT_LIMIT", 2),
        patch(f"{MODULE}.ENTSOE_OFFSET_LIMIT", 2),
    ):
        assert await source.async_update(START, START + timedelta(days=4)) is True

    a80 = [c for c in entsoe.calls if c["documentType"] == "A80"]
    # 4 days: offsets 0 and 2 full, halved into 2 + 2 days, each 0 and 2 full,
    # halved into 1-day windows of 4 documents: offsets 0 and 2 (the last page
    # is full too, but a day is never halved)
    windows = [(c["periodStart"], c["periodEnd"], c["offset"]) for c in a80]
    assert windows[:2] == [
        ("202610010000", "202610050000", "0"),
        ("202610010000", "202610050000", "2"),
    ]
    assert windows[2:4] == [
        ("202610010000", "202610030000", "0"),
        ("202610010000", "202610030000", "2"),
    ]
    assert windows[4:6] == [
        ("202610010000", "202610020000", "0"),
        ("202610010000", "202610020000", "2"),
    ]
    assert len(a80) == 2 + 2 * 2 + 4 * 2
    rows = await source.async_load(START, END)
    assert len({r["message_id"] for r in rows}) == 16


@pytest.mark.asyncio
async def test_a_failed_or_rejected_request_stores_nothing_and_is_repeated(
    storage: LearningStorage, entsoe: FakeEntsoe, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)
    entsoe.answer = lambda q: None if q["documentType"] == "A77" else _zip(PLANT_DOC)
    source = _source(storage, "NL")

    assert await source.async_update(START, END) is False
    assert len(entsoe.calls) == 1
    assert storage.load_source_state("entsoe_outages") is None
    assert await source.async_load(START, END) == []
    assert "Outage documents unavailable (ENTSO-E)" in caplog.text

    entsoe.answer = lambda q: _acknowledgement(
        "Mandatory parameter Out_Domain is missing."
    )
    assert await source.async_update(START, END) is False
    assert (
        "Unusable ENTSO-E outage response: ENTSO-E rejected the request" in caplog.text
    )
    assert storage.load_source_state("entsoe_outages") is None

    entsoe.answer = lambda q: _zip(PLANT_DOC) if q["documentType"] == "A80" else NO_DATA
    assert await source.async_update(START, END) is True
    assert _state(storage)[2] == NOW.isoformat()


@pytest.mark.asyncio
async def test_pruning_moves_the_covered_start(
    storage: LearningStorage, entsoe: FakeEntsoe
) -> None:
    source = _source(storage)
    await source.async_update(START, END)

    assert await source.async_prune(START + timedelta(days=3)) == 0

    assert _state(storage)[0] == (START + timedelta(days=3)).isoformat()


# --- Constants and the backtest ---------------------------------------------------------


def test_the_zones_publishing_on_entsoe_have_borders_and_the_model_uses_none_yet() -> (
    None
):
    assert set(ENTSOE_OUTAGE_BORDERS) == {"DE", "NL", "BE", "FR"}
    assert not set(ENTSOE_OUTAGE_BORDERS) & UMM_AREAS
    for region, borders in ENTSOE_OUTAGE_BORDERS.items():
        assert all(len(eic) == 16 for eic in borders)
        assert REGIONS[region]["entsoe"] not in borders
        assert len(set(borders)) == len(borders)
    # DE borders every OSF zone that neighbours it
    assert {
        REGIONS[z]["entsoe"] for z in ("DK1", "DK2", "SE4", "NO2", "NL", "BE", "FR")
    } <= set(ENTSOE_OUTAGE_BORDERS["DE"])
    assert set(ENTSOE_OUTAGE_BORDERS) >= ENTSOE_OUTAGE_REGIONS


def test_outages_are_loaded_per_month_paged_and_cached_for_the_backtest(
    tmp_path: Path,
) -> None:
    calls: list[dict[str, list[str]]] = []

    def fetch(url: str) -> bytes:
        query = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
        assert query["securityToken"] == ["token"]
        calls.append(query)
        if query["documentType"] == ["A80"] and query["periodStart"] == [
            "202605010000"
        ]:
            # May: a full page, then the rest
            if query["offset"] == ["0"]:
                return _zip(
                    *(
                        _document(
                            "A80",
                            f"may-{i}",
                            1,
                            "2026-05-01T00:00:00Z",
                            None,
                            _plant(PLANT_PERIOD),
                        )
                        for i in range(ENTSOE_DOCUMENT_LIMIT)
                    )
                )
            return _zip(PLANT_DOC)
        return NO_DATA

    span = (date(2026, 5, 20), date(2026, 6, 5))
    index = load_entsoe_outages(
        "BE", *span, tmp_path, "token", fetch, today=date(2026, 6, 3)
    )
    again = load_entsoe_outages(
        "BE", *span, tmp_path, "token", fetch, today=date(2026, 6, 3)
    )

    per_month = 2 + 2 * len(ENTSOE_OUTAGE_BORDERS["BE"])
    # May (one extra page) from the API, June too; then May from the cache
    assert len(calls) == (per_month + 1) + per_month + per_month
    assert (tmp_path / "entsoe_outages_BE_2026-05.json").exists()
    assert not (tmp_path / "entsoe_outages_BE_2026-06.json").exists()
    assert not any("token" in p.read_text() for p in tmp_path.glob("*.json"))
    at = datetime(2026, 10, 5, 12, tzinfo=UTC)
    assert index.for_slot(at, NOW) == again.for_slot(at, NOW)
    assert index.for_slot(at, NOW)["unavailable_production"] == pytest.approx(
        211.0 * (ENTSOE_DOCUMENT_LIMIT + 1)
    )
