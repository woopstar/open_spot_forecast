"""Gap bookkeeping for time-series sources (#32): ranges and the source state."""

from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from custom_components.open_spot_forecast.time_series import (
    Hole,
    SourceState,
    TimeRange,
    grid_range,
    intersect_ranges,
    merge_ranges,
    missing_ranges,
    split_range,
    subtract_ranges,
)

T0 = datetime(2026, 9, 24, tzinfo=UTC)


def _t(hours: float) -> datetime:
    return T0 + timedelta(hours=hours)


# --- Ranges --------------------------------------------------------------------------


def test_grid_range_widens_to_the_grid_in_utc() -> None:
    cph = ZoneInfo("Europe/Copenhagen")
    start = datetime(2026, 9, 24, 2, 20, tzinfo=cph)  # 00:20 UTC

    assert grid_range(start, start + timedelta(minutes=50), 60) == (_t(0), _t(2))
    assert grid_range(start, start, 15) == (_t(0.25), _t(0.5))


def test_missing_ranges_are_runs_of_unknown_grid_points() -> None:
    known = [_t(1), _t(2), _t(5)]

    assert missing_ranges(known, _t(0), _t(7), 60) == [
        (_t(0), _t(1)),
        (_t(3), _t(5)),
        (_t(6), _t(7)),
    ]
    assert missing_ranges([_t(h) for h in range(4)], _t(0), _t(4), 60) == []
    assert missing_ranges([], _t(0), _t(1), 15) == [(_t(0), _t(1))]


def test_missing_ranges_match_known_points_in_any_time_zone() -> None:
    cph = ZoneInfo("Europe/Copenhagen")
    known = [datetime(2026, 9, 24, 3, tzinfo=cph)]  # 01:00 UTC

    assert missing_ranges(known, _t(0), _t(3), 60) == [
        (_t(0), _t(1)),
        (_t(2), _t(3)),
    ]


def test_merge_subtract_and_intersect() -> None:
    ranges = [(_t(4), _t(6)), (_t(0), _t(2)), (_t(1), _t(3)), (_t(5), _t(5))]

    assert merge_ranges(ranges) == [(_t(0), _t(3)), (_t(4), _t(6))]
    assert subtract_ranges(ranges, [(_t(1), _t(5)), (_t(9), _t(10))]) == [
        (_t(0), _t(1)),
        (_t(5), _t(6)),
    ]
    assert subtract_ranges([(_t(0), _t(4))], [(_t(1), _t(2))]) == [
        (_t(0), _t(1)),
        (_t(2), _t(4)),
    ]
    assert intersect_ranges(ranges, [(_t(2), _t(5))]) == [
        (_t(2), _t(3)),
        (_t(4), _t(5)),
    ]
    assert intersect_ranges(ranges, [(_t(7), _t(8))]) == []


def test_split_range_into_request_sized_chunks() -> None:
    assert split_range((_t(0), _t(5)), timedelta(hours=2)) == [
        (_t(0), _t(2)),
        (_t(2), _t(4)),
        (_t(4), _t(5)),
    ]
    assert split_range((_t(1), _t(1)), timedelta(hours=2)) == []


# --- Source state --------------------------------------------------------------------


def _retry(hours: float) -> Callable[[TimeRange], datetime]:
    return lambda _hole: _t(hours)


def test_the_horizon_is_open_ended_and_pending_until_its_retry() -> None:
    state = SourceState()

    state.record([(_t(0), _t(48))], [(_t(24), _t(48))], _t(24), _retry(1))

    assert state.horizon == _t(24)
    assert state.holes == [Hole(_t(24), None, _t(1))]
    pending = state.pending(_t(0.5))
    assert len(pending) == 1
    assert pending[0][0] == _t(24)
    assert pending[0][1] > _t(10_000)
    assert state.pending(_t(1)) == []


def test_a_retried_horizon_is_replaced_by_what_the_update_found() -> None:
    state = SourceState([Hole(_t(24), None, _t(1))])

    # Retried: the data arrived up to 36 h, the new horizon is there
    state.record([(_t(24), _t(48))], [(_t(36), _t(48))], _t(36), _retry(3))
    assert state.holes == [Hole(_t(36), None, _t(3))]

    # Retried again and complete: no horizon is left
    state.record([(_t(36), _t(48))], [], None, _retry(5))
    assert state.holes == []
    assert state.horizon is None


def test_an_update_elsewhere_keeps_the_horizon() -> None:
    state = SourceState([Hole(_t(24), None, _t(1))])

    state.record([(_t(-48), _t(0))], [(_t(-30), _t(-29))], None, _retry(24))

    assert state.holes == [Hole(_t(-30), _t(-29), _t(24)), Hole(_t(24), None, _t(1))]


def test_closed_holes_shrink_to_what_is_still_missing() -> None:
    state = SourceState([Hole(_t(0), _t(10), _t(1))])

    state.record([(_t(2), _t(4))], [(_t(3), _t(4))], None, _retry(9))

    assert state.holes == [
        Hole(_t(0), _t(2), _t(1)),
        Hole(_t(3), _t(4), _t(9)),
        Hole(_t(4), _t(10), _t(1)),
    ]


def test_prune_forgets_holes_before_the_cutoff() -> None:
    state = SourceState(
        [Hole(_t(0), _t(2), _t(9)), Hole(_t(3), _t(5), _t(9)), Hole(_t(6), None, _t(9))]
    )

    state.prune(_t(4))

    assert [hole.start for hole in state.holes] == [_t(3), _t(6)]


def test_the_state_round_trips_through_json() -> None:
    state = SourceState([Hole(_t(0), _t(2), _t(9)), Hole(_t(6), None, _t(1))])

    assert SourceState.from_json(state.to_json()) == state


@pytest.mark.parametrize(
    "stored",
    [
        None,
        "",
        "not json",
        '{"holes": []}',
        '[["bad", null, "2026-09-24T00:00:00+00:00"], "junk", [1, 2]]',
        '[["2026-09-24T00:00:00+00:00", "bad", "2026-09-24T00:00:00+00:00"]]',
    ],
)
def test_an_unreadable_state_is_empty(stored: str | None) -> None:
    assert SourceState.from_json(stored).holes == []
