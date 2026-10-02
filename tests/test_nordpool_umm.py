"""Nord Pool UMM outages: parser, per-origin aggregation, storage and source (#123)."""

import json
from collections.abc import Callable, Iterator
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

import pytest

from custom_components.open_spot_forecast.api.http import HttpResponse
from custom_components.open_spot_forecast.api.nordpool_umm import (
    UMM_PAGE_SIZE,
    NordpoolUmmSource,
    parse_umm_messages,
    umm_query,
)
from custom_components.open_spot_forecast.const import UMM_AREAS, UMM_REGIONS
from custom_components.open_spot_forecast.ml.outages import (
    OutageIndex,
    day_ahead_gate,
)
from custom_components.open_spot_forecast.ml.storage import LearningStorage

MODULE = "custom_components.open_spot_forecast.api.nordpool_umm"
DK1 = "10YDK-1--------W"
DK2 = "10YDK-2--------M"
SE3 = "10Y1001A1001A46L"
NOW = datetime(2026, 10, 2, 12, tzinfo=UTC)


def _period(start: str, stop: str, unavailable: int, available: int = 0) -> dict:
    return {
        "eventStart": f"{start}:00.0000000Z",
        "eventStop": f"{stop}:00.0000000Z",
        "unavailableCapacity": unavailable,
        "availableCapacity": available,
    }


def _message(
    message_id: str = "m1",
    version: int = 1,
    published: str = "2026-09-30T08:00",
    message_type: int = 1,
    status: int = 1,
    **units: list[dict],
) -> dict:
    """A message in the API's JSON shape (numeric enums, 7-digit fractions)."""
    return {
        "messageId": message_id,
        "version": version,
        "publicationDate": f"{published}:00.1234567Z",
        "isOutdated": False,
        "eventStatus": status,
        "messageType": message_type,
        "unavailabilityType": 2,
        "productionUnits": [],
        "generationUnits": [],
        "transmissionUnits": [],
        **units,
    }


HORNS_REV = {
    "name": "Horns Rev C",
    "eic": "45W000000000114R",
    "fuelType": 18,
    "areaEic": DK1,
    "installedCapacity": 407,
    "timePeriods": [_period("2026-10-05T06:30", "2026-10-05T12:00", 291, 116)],
}
NORDJYLLAND_B3 = {
    "name": "B3",
    "eic": "45W000000000016R",
    "productionUnitName": "Nordjyllandsværket",
    "fuelType": 5,
    "areaEic": DK1,
    "installedCapacity": 412,
    "timePeriods": [
        _period("2026-10-01T00:00", "2026-10-31T00:00", 72),
        _period("2026-10-31T00:00", "2026-11-02T00:00", 412),
    ],
}
SKAGERRAK = {
    "name": "NO2 → DK1",
    "inAreaEic": "10YNO-2--------T",
    "outAreaEic": DK1,
    "installedCapacity": 1700,
    "timePeriods": [_period("2026-10-06T04:00", "2026-10-08T15:00", 700, 1000)],
}
KONTI_SKAN = {
    "name": "SE3 → NO1",
    "inAreaEic": SE3,
    "outAreaEic": "10YNO-1--------2",
    "installedCapacity": 2095,
    "timePeriods": [_period("2026-10-06T04:00", "2026-10-08T15:00", 1595, 500)],
}


def _page(*messages: dict, total: int | None = None) -> dict:
    return {"items": list(messages), "total": len(messages) if total is None else total}


# --- Parsing -------------------------------------------------------------------------


def test_production_and_generation_units_in_the_area_become_period_rows() -> None:
    rows = parse_umm_messages(
        _page(
            _message("a", productionUnits=[HORNS_REV]),
            _message("b", 3, generationUnits=[NORDJYLLAND_B3]),
        ),
        DK1,
    )

    assert [(r["message_id"], r["version"], r["unit"]) for r in rows] == [
        ("a", 1, "45W000000000114R"),
        ("b", 3, "45W000000000016R"),
        ("b", 3, "45W000000000016R"),
    ]
    assert rows[0] == {
        "message_id": "a",
        "version": 1,
        "published": "2026-09-30T08:00:00Z",
        "message_type": 1,
        "unavailability_type": 2,
        "status": 1,
        "unit": "45W000000000114R",
        "kind": "production",
        "fuel_type": 18,
        "event_start": "2026-10-05T06:30:00Z",
        "event_stop": "2026-10-05T12:00:00Z",
        "unavailable_mw": 291.0,
        "installed_mw": 407.0,
    }
    assert [r["unavailable_mw"] for r in rows[1:]] == [72.0, 412.0]


def test_transmission_units_count_when_the_area_is_on_either_end() -> None:
    """One Nordic message lists many connections; only DK1's are kept."""
    message = _message("t", message_type=3, transmissionUnits=[SKAGERRAK, KONTI_SKAN])

    dk1 = parse_umm_messages(_page(message), DK1)
    se3 = parse_umm_messages(_page(message), SE3)

    assert [(r["kind"], r["unit"], r["unavailable_mw"]) for r in dk1] == [
        ("transmission", "10YNO-2--------T>10YDK-1--------W", 700.0)
    ]
    assert [r["unit"] for r in se3] == ["10Y1001A1001A46L>10YNO-1--------2"]
    assert dk1[0]["fuel_type"] is None


def test_a_version_without_a_unit_in_the_area_is_kept_as_a_header_row() -> None:
    """The API's area filter matches any unit of the message: a revision
    that dropped the area's unit must still supersede the earlier version."""
    other_area = {**HORNS_REV, "areaEic": DK2}
    rows = parse_umm_messages(
        _page(_message("a", 2, productionUnits=[other_area])), DK1
    )

    assert rows == [
        {
            "message_id": "a",
            "version": 2,
            "published": "2026-09-30T08:00:00Z",
            "message_type": 1,
            "unavailability_type": 2,
            "status": 1,
            "kind": None,
        }
    ]


def test_consumption_and_other_messages_and_broken_periods_are_skipped() -> None:
    broken = {**HORNS_REV, "timePeriods": [{"eventStart": None}, "x"]}
    rows = parse_umm_messages(
        _page(
            _message("c", message_type=2, consumptionUnits=[HORNS_REV]),
            _message("o", message_type=5),
            _message("p", productionUnits=[broken]),
            {"messageId": "no-version", "messageType": 1},
        ),
        DK1,
    )

    assert [(r["message_id"], r["kind"]) for r in rows] == [("p", None)]


def test_a_dismissed_version_keeps_its_status() -> None:
    (row,) = parse_umm_messages(
        _page(_message("a", 3, status=3, productionUnits=[HORNS_REV])), DK1
    )

    assert (row["version"], row["status"]) == (3, 3)


@pytest.mark.parametrize("payload", [[], {"items": None}, {"items": ["x"]}, "x"])
def test_unexpected_payloads_raise(payload: Any) -> None:
    with pytest.raises(TypeError):
        parse_umm_messages(payload, DK1)


def test_the_query_repeats_list_keys_and_bounds_the_publications() -> None:
    start = datetime(2026, 9, 1, tzinfo=UTC)
    query = umm_query(DK1, start, start + timedelta(days=30), NOW, skip=500)

    assert ("messageTypes", "1") in query and ("messageTypes", "3") in query
    assert ("includeOutdated", "true") in query
    assert ("eventStartDate", "2026-09-01T00:00:00Z") in query
    assert ("eventStopDate", "2026-10-01T00:00:00Z") in query
    assert ("publicationStopDate", "2026-10-02T12:00:00Z") in query
    assert ("skip", "500") in query and ("limit", str(UMM_PAGE_SIZE)) in query
    assert not [key for key, _ in query if key == "publicationStartDate"]
    assert ("publicationStartDate", "2026-10-02T12:00:00Z") in umm_query(
        DK1, start, start, NOW, published_from=NOW
    )


# --- Per-origin aggregation ----------------------------------------------------------


def _rows(*messages: dict, eic: str = DK1) -> list[dict]:
    return parse_umm_messages(_page(*messages), eic)


def _slot(day: int, hour: int) -> datetime:
    return datetime(2026, 10, day, hour, tzinfo=UTC)


def test_the_day_ahead_gate_is_noon_cet_the_day_before() -> None:
    assert day_ahead_gate(date(2026, 10, 2)).astimezone(UTC) == datetime(
        2026, 10, 1, 10, tzinfo=UTC
    )
    # Winter: CET is UTC+1
    assert day_ahead_gate(date(2026, 12, 2)).astimezone(UTC) == datetime(
        2026, 12, 1, 11, tzinfo=UTC
    )


def test_a_slot_sums_the_units_unavailable_in_it_by_kind() -> None:
    index = OutageIndex(
        _rows(
            _message("a", productionUnits=[HORNS_REV]),
            _message("b", generationUnits=[NORDJYLLAND_B3]),
            _message("t", message_type=3, transmissionUnits=[SKAGERRAK, KONTI_SKAN]),
        )
    )

    assert index.for_slot(_slot(5, 7), NOW) == {
        "unavailable_production": 291.0 + 72.0,
        "unavailable_transmission": 0.0,
    }
    assert index.for_slot(_slot(6, 12), NOW) == {
        "unavailable_production": 72.0,
        "unavailable_transmission": 700.0,
    }
    # The period's end is exclusive, its start inclusive
    assert index.for_slot(_slot(5, 12), NOW)["unavailable_production"] == 72.0
    assert index.for_slot(datetime(2026, 10, 5, 6, 30, tzinfo=UTC), NOW) == {
        "unavailable_production": 363.0,
        "unavailable_transmission": 0.0,
    }
    assert OutageIndex([]).for_slot(_slot(5, 7), NOW) == {
        "unavailable_production": 0.0,
        "unavailable_transmission": 0.0,
    }


def test_an_origin_sees_the_latest_version_published_by_then() -> None:
    """Revised: 291 MW announced on the 30th, cut to 100 MW on the 3rd."""
    revised = {
        **HORNS_REV,
        "timePeriods": [_period("2026-10-05T06:30", "2026-10-05T12:00", 100)],
    }
    index = OutageIndex(
        _rows(
            _message("a", 1, "2026-09-30T08:00", productionUnits=[HORNS_REV]),
            _message("a", 2, "2026-10-03T09:00", productionUnits=[revised]),
        )
    )
    slot = _slot(5, 8)

    assert index.for_slot(slot, datetime(2026, 9, 29, tzinfo=UTC)) == {
        "unavailable_production": 0.0,
        "unavailable_transmission": 0.0,
    }
    assert index.for_slot(slot, datetime(2026, 9, 30, 8, tzinfo=UTC))[
        "unavailable_production"
    ] == pytest.approx(291.0)
    assert index.for_slot(slot, datetime(2026, 10, 3, 8, 59, tzinfo=UTC))[
        "unavailable_production"
    ] == pytest.approx(291.0)
    assert index.for_slot(slot, datetime(2026, 10, 3, 9, tzinfo=UTC))[
        "unavailable_production"
    ] == pytest.approx(100.0)
    # Row order does not matter
    rows = _rows(
        _message("a", 2, "2026-10-03T09:00", productionUnits=[revised]),
        _message("a", 1, "2026-09-30T08:00", productionUnits=[HORNS_REV]),
    )
    assert OutageIndex(rows).for_slot(slot, datetime(2026, 10, 4, tzinfo=UTC))[
        "unavailable_production"
    ] == pytest.approx(100.0)


def test_a_dismissed_message_and_a_version_dropping_the_unit_count_nothing() -> None:
    dismissed = OutageIndex(
        _rows(
            _message("a", 1, "2026-09-30T08:00", productionUnits=[HORNS_REV]),
            _message("a", 2, "2026-10-01T08:00", status=3, productionUnits=[HORNS_REV]),
        )
    )
    dropped = OutageIndex(
        _rows(
            _message(
                "t",
                1,
                "2026-09-30T08:00",
                message_type=3,
                transmissionUnits=[SKAGERRAK],
            ),
            _message(
                "t",
                2,
                "2026-10-01T08:00",
                message_type=3,
                transmissionUnits=[KONTI_SKAN],
            ),
        )
    )

    before, after = (
        datetime(2026, 9, 30, 12, tzinfo=UTC),
        datetime(2026, 10, 1, 12, tzinfo=UTC),
    )
    assert dismissed.for_slot(_slot(5, 8), before)["unavailable_production"] == 291.0
    assert dismissed.for_slot(_slot(5, 8), after)["unavailable_production"] == 0.0
    assert dropped.for_slot(_slot(6, 12), before)["unavailable_transmission"] == 700.0
    assert dropped.for_slot(_slot(6, 12), after)["unavailable_transmission"] == 0.0


def test_a_period_given_twice_counts_once_and_bad_rows_are_ignored() -> None:
    rows = _rows(_message("a", productionUnits=[HORNS_REV]))
    index = OutageIndex(
        [
            *rows,
            *rows,
            {"message_id": "x", "version": "one"},
            {**rows[0], "message_id": "y", "unavailable_mw": -5},
            {**rows[0], "message_id": "z", "event_stop": rows[0]["event_start"]},
        ]
    )

    assert index.for_slot(_slot(5, 8), NOW)["unavailable_production"] == 291.0


# --- Storage -------------------------------------------------------------------------


@pytest.fixture
def storage(tmp_path: Path) -> Iterator[LearningStorage]:
    hass = Mock()
    hass.config.path.return_value = str(tmp_path / ".storage")
    store = LearningStorage(hass, "DK1")
    yield store
    store.close()


def test_rows_round_trip_and_a_stored_version_is_never_rewritten(
    storage: LearningStorage,
) -> None:
    rows = _rows(
        _message("a", productionUnits=[HORNS_REV]),
        _message("b", 2, productionUnits=[{**HORNS_REV, "areaEic": DK2}]),
    )
    october = (datetime(2026, 10, 1, tzinfo=UTC), datetime(2026, 11, 1, tzinfo=UTC))

    assert storage.upsert_umm_rows(rows) is True
    assert storage.last_data_write is not None
    assert storage.upsert_umm_rows(rows) is False
    assert storage.upsert_umm_rows([]) is False

    loaded = storage.load_umm_rows(*october)
    assert loaded[0] == rows[0]
    # The header row comes back with empty period columns
    assert loaded[1] == rows[1] | dict.fromkeys(
        (
            "unit",
            "fuel_type",
            "event_start",
            "event_stop",
            "unavailable_mw",
            "installed_mw",
        )
    )
    assert len(loaded) == 2
    # Outside the range every version still comes back, without its periods
    december = (datetime(2026, 12, 1, tzinfo=UTC), datetime(2027, 1, 1, tzinfo=UTC))
    assert [(r["message_id"], r["kind"]) for r in storage.load_umm_rows(*december)] == [
        ("a", None),
        ("b", None),
    ]


def test_pruning_drops_ended_periods_and_versions_left_without_any(
    storage: LearningStorage,
) -> None:
    storage.upsert_umm_rows(
        _rows(
            _message("a", 1, "2026-09-30T08:00", productionUnits=[HORNS_REV]),
            _message("b", 1, "2026-09-30T08:00", generationUnits=[NORDJYLLAND_B3]),
            _message("b", 2, "2026-10-20T08:00", generationUnits=[NORDJYLLAND_B3]),
        )
    )

    deleted = storage.prune_umm_rows(datetime(2026, 10, 15, tzinfo=UTC))

    assert deleted == 1
    rows = storage.load_umm_rows(
        datetime(2026, 1, 1, tzinfo=UTC), datetime(2027, 1, 1, tzinfo=UTC)
    )
    assert {(r["message_id"], r["version"]) for r in rows} == {("b", 1), ("b", 2)}
    assert all(r["event_stop"] >= "2026-10-15" for r in rows)


# --- The source ----------------------------------------------------------------------


class FakeUmm:
    """The UMM API, recording every request's query."""

    def __init__(self) -> None:
        self.pages: dict[int, dict] = {0: _page()}
        self.answer: Callable[[dict], HttpResponse | None] | None = None
        self.calls: list[dict[str, list[str]]] = []

    async def get(self, _session: Any, _url: str, label: str, **kw: Any) -> Any:
        query: dict[str, list[str]] = {}
        for key, value in kw["params"]:
            query.setdefault(key, []).append(value)
        self.calls.append(query)
        if self.answer is not None:
            return self.answer(query)
        return HttpResponse(200, json.dumps(self.pages[int(query["skip"][0])]))


@pytest.fixture
def umm() -> Iterator[FakeUmm]:
    fake = FakeUmm()
    with (
        patch(f"{MODULE}.async_get", new=fake.get),
        patch(f"{MODULE}.async_get_clientsession"),
        patch("homeassistant.util.dt.utcnow", return_value=NOW),
    ):
        yield fake


def _source(storage: LearningStorage) -> NordpoolUmmSource:
    async def run_inline(func: Callable[..., Any], *args: Any) -> Any:
        return func(*args)

    hass = Mock()
    hass.async_add_executor_job = run_inline
    return NordpoolUmmSource(hass, storage, "DK1")


START, END = datetime(2026, 9, 1, tzinfo=UTC), datetime(2026, 10, 10, tzinfo=UTC)


@pytest.mark.asyncio
async def test_the_first_update_fetches_every_version_of_the_window(
    storage: LearningStorage, umm: FakeUmm
) -> None:
    umm.pages = {
        0: _page(_message("a", productionUnits=[HORNS_REV]), total=2),
        1: _page(_message("a", 2, productionUnits=[HORNS_REV]), total=2),
    }
    source = _source(storage)

    assert await source.async_update(START, END) is True

    assert [c["skip"] for c in umm.calls] == [["0"], ["1"]]
    first = umm.calls[0]
    assert first["areas"] == [DK1]
    assert first["messageTypes"] == ["1", "3"]
    assert first["eventStartDate"] == ["2026-09-01T00:00:00Z"]
    assert first["eventStopDate"] == ["2026-10-10T00:00:00Z"]
    assert first["publicationStopDate"] == ["2026-10-02T12:00:00Z"]
    assert "publicationStartDate" not in first
    rows = storage.load_umm_rows(START, END)
    assert [(r["message_id"], r["version"]) for r in rows] == [("a", 1), ("a", 2)]
    assert json.loads(storage.load_source_state("umm_outages") or "") == [
        START.isoformat(),
        END.isoformat(),
        NOW.isoformat(),
    ]


@pytest.mark.asyncio
async def test_later_updates_fetch_new_publications_and_new_days_only(
    storage: LearningStorage, umm: FakeUmm
) -> None:
    source = _source(storage)
    await source.async_update(START, END)
    umm.calls.clear()
    later = NOW + timedelta(hours=6)

    with patch("homeassistant.util.dt.utcnow", return_value=later):
        changed = await source.async_update(START, END + timedelta(days=1))

    assert changed is False
    assert len(umm.calls) == 2
    new_days, since = umm.calls
    assert (new_days["eventStartDate"], new_days["eventStopDate"]) == (
        ["2026-10-10T00:00:00Z"],
        ["2026-10-11T00:00:00Z"],
    )
    assert "publicationStartDate" not in new_days
    assert since["eventStartDate"] == ["2026-09-01T00:00:00Z"]
    assert since["publicationStartDate"] == ["2026-10-02T12:00:00Z"]
    assert since["publicationStopDate"] == ["2026-10-02T18:00:00Z"]
    # Nothing new: the covered window grew, the publication mark moved on
    assert json.loads(storage.load_source_state("umm_outages") or "")[1:] == [
        (END + timedelta(days=1)).isoformat(),
        later.isoformat(),
    ]
    # A state written by an earlier run is read back by a new source
    umm.calls.clear()
    await _source(storage).async_update(START, END)
    assert len(umm.calls) == 1
    assert umm.calls[0]["publicationStartDate"] == [
        later.isoformat().replace("+00:00", "Z")
    ]


@pytest.mark.parametrize(
    "answer",
    [
        lambda _q: None,
        lambda _q: HttpResponse(403, "blocked"),
        lambda _q: HttpResponse(200, "<html>"),
        lambda _q: HttpResponse(200, "[]"),
    ],
)
@pytest.mark.asyncio
async def test_a_failed_request_stores_nothing_and_is_repeated_next_time(
    storage: LearningStorage,
    umm: FakeUmm,
    answer: Callable[[dict], HttpResponse | None],
) -> None:
    umm.answer = answer
    source = _source(storage)

    assert await source.async_update(START, END) is False
    umm.answer = None
    assert await source.async_update(START, END) is False

    assert storage.load_source_state("umm_outages") is not None
    # The window was asked for again in full
    assert "publicationStartDate" not in umm.calls[-1]
    assert umm.calls[-1]["eventStartDate"] == ["2026-09-01T00:00:00Z"]


@pytest.mark.asyncio
async def test_pruning_moves_the_covered_start(
    storage: LearningStorage, umm: FakeUmm
) -> None:
    source = _source(storage)
    await source.async_update(START, END)
    cutoff = datetime(2026, 9, 15, tzinfo=UTC)

    assert await source.async_prune(cutoff) == 0

    assert (
        json.loads(storage.load_source_state("umm_outages") or "")[0]
        == cutoff.isoformat()
    )
    umm.calls.clear()
    await source.async_update(START, END)
    # The pruned part is fetched again as history
    assert umm.calls[0]["eventStopDate"] == ["2026-09-15T00:00:00Z"]


def test_the_model_uses_it_where_the_backtest_found_it_helps() -> None:
    assert {"DK1", "DK2", "SE3", "SE4", "NO2", "FI", "EE", "LT", "LV"} == UMM_AREAS
    assert not {"DE", "NL", "BE", "FR"} & UMM_AREAS
    # DK2: a lower day-1 error, but days 2-7 got worse (docs/ml_documentation.md)
    assert {"DK1"} == UMM_REGIONS
    assert UMM_REGIONS <= UMM_AREAS
