"""OutageIndex: the per-day step functions give the sums of a loop (#167)."""

import random
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import patch

import pytest

from custom_components.open_spot_forecast.ml.features import floor_epoch
from custom_components.open_spot_forecast.ml.outages import (
    KIND_PRODUCTION,
    KIND_TRANSMISSION,
    OUTAGE_FEATURES,
    UMM_ACTIVE,
    UMM_DISMISSED,
    OutageIndex,
)

BASE = datetime(2026, 10, 1, tzinfo=UTC)
NOW = datetime(2026, 10, 2, 12, tzinfo=UTC)
MINUTE = timedelta(minutes=1)
DAY = timedelta(days=1)
DURATIONS = (15 * MINUTE, 50 * MINUTE, 6 * 60 * MINUTE, DAY, 3 * DAY, 40 * DAY)


def _row(
    message_id: str,
    start: datetime | None = None,
    stop: datetime | None = None,
    unavailable: float = 0.0,
    *,
    kind: str = KIND_PRODUCTION,
    version: int = 1,
    published: datetime = BASE,
    status: int = UMM_ACTIVE,
    unit: str = "unit",
) -> dict[str, Any]:
    """A stored row: a period, or a version without one (no ``start``)."""
    return {
        "message_id": message_id,
        "version": version,
        "published": published.isoformat(),
        "status": status,
        "kind": kind if start else None,
        "unit": unit if start else None,
        "event_start": start.isoformat() if start else None,
        "event_stop": stop.isoformat() if stop else None,
        "unavailable_mw": unavailable if start else None,
    }


def _loop(
    rows: list[dict[str, Any]], slot: datetime, origin: datetime
) -> dict[str, float]:
    """Return what a loop over every stored period sums (``for_slot`` before #167)."""
    known = int(origin.timestamp())
    latest: dict[str, tuple[int, int, bool]] = {}
    for row in rows:
        published = int(datetime.fromisoformat(row["published"]).timestamp())
        newer = (published, row["version"], row["status"] == UMM_DISMISSED)
        if published <= known and newer > latest.get(row["message_id"], (0, 0)):
            latest[row["message_id"]] = newer
    at = floor_epoch(slot, UTC)
    sums = dict.fromkeys(OUTAGE_FEATURES.values(), 0.0)
    seen: set[tuple[Any, ...]] = set()
    for row in rows:
        _, version, dismissed = latest.get(row["message_id"], (0, 0, True))
        key = (row["message_id"], row["unit"], row["event_start"], row["event_stop"])
        if dismissed or row["version"] != version or not row["kind"] or key in seen:
            continue
        seen.add(key)
        start = datetime.fromisoformat(row["event_start"]).timestamp()
        stop = datetime.fromisoformat(row["event_stop"]).timestamp()
        if start <= at < stop:
            sums[OUTAGE_FEATURES[row["kind"]]] += row["unavailable_mw"]
    return sums


def _archive(
    rng: random.Random,
) -> tuple[list[dict[str, Any]], list[datetime], list[datetime]]:
    """Return random rows, their publication moments and their period edges.

    Messages in one to four versions (some published at the same moment,
    some dismissed, some without a period), periods of 15 minutes to 40
    days off and on the slot grid, in quarter MW (their sums are exact in
    any order, so the stored order of the rows does not show).
    """
    publications = [BASE + timedelta(hours=rng.randrange(-240, 240)) for _ in range(30)]
    rows: list[dict[str, Any]] = []
    edges: list[datetime] = []
    for message in range(40):
        for version in range(1, rng.randint(1, 4) + 1):
            header: dict[str, Any] = {
                "version": version,
                "published": rng.choice(publications),
                "status": UMM_DISMISSED if rng.random() < 0.15 else UMM_ACTIVE,
            }
            if rng.random() < 0.1:
                rows.append(_row(f"m{message}", **header))
                continue
            for period in range(rng.randint(1, 3)):
                start = BASE + rng.randrange(-20 * 288, 20 * 288) * 5 * MINUTE
                stop = start + rng.choice(DURATIONS)
                edges += [start, stop]
                rows.append(
                    _row(
                        f"m{message}",
                        start,
                        stop,
                        rng.randint(1, 4000) / 4,
                        kind=rng.choice((KIND_PRODUCTION, KIND_TRANSMISSION)),
                        unit=f"u{period}",
                        **header,
                    )
                )
    rows += rng.sample(rows, 10)
    rng.shuffle(rows)
    return rows, publications, edges


@pytest.mark.parametrize("seed", range(8))
def test_a_slot_gets_the_sum_of_a_loop_over_the_periods(seed: int) -> None:
    rng = random.Random(seed)
    rows, publications, edges = _archive(rng)
    index = OutageIndex(rows)
    counted = 0

    for _ in range(400):
        # On and around a publication, on a period's edge and anywhere else
        origin = rng.choice(publications) + timedelta(seconds=rng.choice((-1, 0, 1)))
        slot = rng.choice(
            (
                rng.choice(edges),
                BASE + rng.randrange(-22 * 288, 22 * 288) * 5 * MINUTE,
            )
        )
        expected = _loop(rows, slot, origin)
        counted += any(expected.values())

        assert index.for_slot(slot, origin) == pytest.approx(expected, rel=0, abs=0)
    assert counted > 100


def test_a_period_counts_on_every_utc_day_it_covers() -> None:
    """Start inclusive and stop exclusive, also when they are midnights."""
    index = OutageIndex(
        [
            _row("a", BASE - 2 * DAY, BASE + DAY, 300.0),
            _row("b", BASE + 60 * MINUTE, BASE + 2 * DAY + 90 * MINUTE, 40.0),
            _row("t", BASE - 400 * DAY, BASE + 4000 * DAY, 2.0, kind=KIND_TRANSMISSION),
        ]
    )

    def production(slot: datetime) -> float:
        got = index.for_slot(slot, NOW)
        assert got["unavailable_transmission"] == pytest.approx(2.0)
        return got["unavailable_production"]

    assert production(BASE - 2 * DAY - 15 * MINUTE) == pytest.approx(0.0)
    assert production(BASE - 2 * DAY) == pytest.approx(300.0)
    assert production(BASE - 15 * MINUTE) == pytest.approx(300.0)
    assert production(BASE) == pytest.approx(300.0)
    assert production(BASE + 60 * MINUTE) == pytest.approx(340.0)
    assert production(BASE + DAY - 15 * MINUTE) == pytest.approx(340.0)
    assert production(BASE + DAY) == pytest.approx(40.0)
    assert production(BASE + 2 * DAY + 75 * MINUTE) == pytest.approx(40.0)
    assert production(BASE + 2 * DAY + 90 * MINUTE) == pytest.approx(0.0)
    assert production(BASE + 3 * DAY) == pytest.approx(0.0)


def test_a_slot_start_is_floored_to_its_quarter_and_a_naive_one_is_utc() -> None:
    index = OutageIndex([_row("a", BASE + 20 * MINUTE, BASE + 40 * MINUTE, 50.0)])

    # 00:15 is before the start at 00:20, 00:30 inside, 00:45 after the stop
    assert index.for_slot(BASE + 29 * MINUTE, NOW)[
        "unavailable_production"
    ] == pytest.approx(0.0)
    assert index.for_slot(BASE + 44 * MINUTE, NOW)[
        "unavailable_production"
    ] == pytest.approx(50.0)
    assert index.for_slot(
        (BASE + 30 * MINUTE).replace(tzinfo=None), NOW
    ) == index.for_slot(BASE + 30 * MINUTE, NOW)
    assert index.for_slot(BASE + 45 * MINUTE, NOW)[
        "unavailable_production"
    ] == pytest.approx(0.0)


def test_of_versions_published_together_the_higher_one_counts() -> None:
    rows = [
        _row("a", BASE, BASE + DAY, 100.0, version=2),
        _row("a", BASE, BASE + DAY, 250.0, version=3),
        _row("a", BASE, BASE + DAY, 400.0, version=1),
    ]

    for ordered in (rows, rows[::-1]):
        assert OutageIndex(ordered).for_slot(BASE, BASE)[
            "unavailable_production"
        ] == pytest.approx(250.0)


def test_fractional_capacities_leave_no_remainder_when_the_periods_end() -> None:
    """A level adds the periods in force; it is not a running total."""
    index = OutageIndex(
        [
            _row("a", BASE, BASE + 60 * MINUTE, 0.1),
            _row("b", BASE + 15 * MINUTE, BASE + 45 * MINUTE, 0.2),
            _row("c", BASE + 30 * MINUTE, BASE + 60 * MINUTE, 0.3),
        ]
    )

    def production(minutes: int) -> float:
        return index.for_slot(BASE + minutes * MINUTE, NOW)["unavailable_production"]

    assert production(30) == pytest.approx(0.1 + 0.2 + 0.3, rel=0, abs=0)
    assert production(45) == pytest.approx(0.1 + 0.3, rel=0, abs=0)
    assert production(60) == pytest.approx(0.0, abs=0)
    assert production(24 * 60) == pytest.approx(0.0, abs=0)


def test_the_slots_of_a_day_share_one_step_function_per_origin() -> None:
    """The periods are walked once per UTC day and origin, not once per slot."""
    index = OutageIndex(
        [_row(f"m{n}", BASE + n * MINUTE, BASE + DAY, 1.0) for n in range(0, 600, 5)]
    )
    slots = [BASE + quarter * 15 * MINUTE for quarter in range(2 * 96)]

    with patch.object(
        OutageIndex, "_profile", autospec=True, side_effect=OutageIndex._profile
    ) as built:
        first = [index.for_slot(slot, NOW)["unavailable_production"] for slot in slots]
        again = [index.for_slot(slot, NOW)["unavailable_production"] for slot in slots]
        index.for_slot(slots[0], NOW + DAY)

    assert built.call_count == 3
    assert first == pytest.approx(again)
    assert first[:5] == pytest.approx([1.0, 4.0, 7.0, 10.0, 13.0])
    assert first[96:] == pytest.approx([0.0] * 96)
