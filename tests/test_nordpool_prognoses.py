"""Nordpool prognoses through the gap-aware time-series source (#32).

The source runs against a real ``LearningStorage``; only the two Nordpool
endpoints and the clock are replaced. Delivery days are CET/CEST days.
"""

from collections.abc import Callable, Iterator
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest

from custom_components.open_spot_forecast.api import (
    NordpoolPrognosisSource,
    forecast_prognoses,
)
from custom_components.open_spot_forecast.api.nordpool_prognoses import (
    delivery_day_range,
    prognosis_rows,
)
from custom_components.open_spot_forecast.ml.series_storage import NORDPOOL_PROGNOSES
from custom_components.open_spot_forecast.ml.storage import LearningStorage

MODULE = "custom_components.open_spot_forecast.api.nordpool_prognoses"
SOURCE = "custom_components.open_spot_forecast.api.time_series_source"
TODAY = date(2026, 9, 24)
TOMORROW = date(2026, 9, 25)
# Today's and tomorrow's delivery days: 2026-09-23T22:00Z to 2026-09-25T22:00Z
START = datetime(2026, 9, 23, 22, tzinfo=UTC)
END = datetime(2026, 9, 25, 22, tzinfo=UTC)
# 10:20 CEST, before tomorrow's prognoses are published
MORNING = datetime(2026, 9, 24, 8, 20, tzinfo=UTC)


def _hours(day: date) -> list[str]:
    start, _ = delivery_day_range(day)
    return [
        (start + timedelta(hours=h)).strftime("%Y-%m-%dT%H:%M:%SZ") for h in range(24)
    ]


def _production(timestamp: str) -> dict[str, Any]:
    return {
        "deliveryStart": timestamp,
        "solar": 50.0,
        "wind_offshore": 100.0,
        "wind_onshore": 200.0,
        "total": 350.0,
    }


class FakeNordpool:
    """Nordpool's two endpoints, publishing chosen days (with or without breakdown)."""

    def __init__(self) -> None:
        self.published: set[date] = {TODAY}
        self.breakdown: set[date] = {TODAY}
        self.failing: set[date] = set()
        # Added to every consumption value (Nordpool revising its prognosis)
        self.revision = 0.0
        self.consumption = AsyncMock(side_effect=self._consumption)
        self.production = AsyncMock(side_effect=self._production)

    async def _consumption(self, day: date, _area: str) -> tuple[Any, Any]:
        if day in self.failing:
            return None, None
        if day not in self.published:
            return None, "2026-09-24T07:00:00Z"
        values = enumerate(_hours(day))
        return {ts: 4000.0 + i + self.revision for i, ts in values}, "u1"

    async def _production(self, day: date, _area: str) -> tuple[Any, Any]:
        if day not in self.breakdown:
            return None, "u1"
        return [_production(ts) for ts in _hours(day)], "p1"

    @property
    def days(self) -> list[date]:
        """Return the delivery days requested so far."""
        return [call.args[0] for call in self.consumption.call_args_list]


@pytest.fixture
def nordpool() -> Iterator[FakeNordpool]:
    fake = FakeNordpool()
    with (
        patch(f"{MODULE}.fetch_consumption_prognosis", fake.consumption),
        patch(f"{MODULE}.fetch_production_prognosis", fake.production),
        patch(f"{SOURCE}.asyncio.sleep", new=AsyncMock()),
    ):
        yield fake


@pytest.fixture
def clock() -> Iterator[Callable[[datetime], None]]:
    """Freeze ``dt_util.utcnow``; call the fixture's value to move it."""
    now = {"value": MORNING}

    def move(moment: datetime) -> None:
        now["value"] = moment

    with patch("homeassistant.util.dt.utcnow", side_effect=lambda: now["value"]):
        yield move


@pytest.fixture
def storage(tmp_path: Path) -> Iterator[LearningStorage]:
    hass = Mock()
    hass.config.path.return_value = str(tmp_path / ".storage")
    store = LearningStorage(hass, "DK1")
    yield store
    store.close()


def _source(storage: LearningStorage, **kwargs: Any) -> NordpoolPrognosisSource:
    async def run_inline(func: Callable[..., Any], *args: Any) -> Any:
        return func(*args)

    hass = Mock()
    hass.async_add_executor_job = run_inline
    return NordpoolPrognosisSource(hass, storage, "DK1", **kwargs)


# --- Rows ----------------------------------------------------------------------------


def test_prognosis_rows_combine_consumption_with_the_production_at_its_start() -> None:
    rows = prognosis_rows(
        {"2026-09-24T10:00:00Z": 4000.0, "2026-09-24T11:00:00Z": 4100.0},
        [_production("2026-09-24T10:00:00Z"), _production("2026-09-24T10:15:00Z")],
    )

    assert rows == [
        {
            "timestamp": "2026-09-24T10:00:00Z",
            "consumption": 4000.0,
            "solar": 50.0,
            "wind_offshore": 100.0,
            "wind_onshore": 200.0,
        },
        {
            "timestamp": "2026-09-24T11:00:00Z",
            "consumption": 4100.0,
            "solar": None,
            "wind_offshore": None,
            "wind_onshore": None,
        },
    ]
    assert prognosis_rows(None, None) == []


def test_forecast_prognoses_have_the_shape_the_features_read() -> None:
    empty = {"solar": None, "wind_offshore": None, "wind_onshore": None}
    rows: list[dict[str, Any]] = [
        {"timestamp": "a", "consumption": 1.0, **empty, "solar": 2.0},
        {"timestamp": "b", "consumption": None, **empty},
    ]

    assert forecast_prognoses(rows) == {
        "consumption_prognosis": {"a": 1.0},
        "production_prognosis": [{"deliveryStart": "a", **empty, "solar": 2.0}],
    }
    assert forecast_prognoses([]) == {}


def test_one_request_per_cet_delivery_day(storage: LearningStorage) -> None:
    source = _source(storage)
    hour = timedelta(hours=1)

    chunks = source.chunks(
        [
            (START + 2 * hour, START + 3 * hour),
            (START + 5 * hour, START + 30 * hour),  # into tomorrow
        ]
    )

    assert chunks == [delivery_day_range(TODAY), delivery_day_range(TOMORROW)]
    # Winter: CET midnight is 23:00 UTC
    assert delivery_day_range(date(2026, 12, 1))[0] == datetime(
        2026, 11, 30, 23, tzinfo=UTC
    )


# --- Fetching only what is missing ---------------------------------------------------


@pytest.mark.asyncio
async def test_complete_history_costs_no_request(
    storage: LearningStorage, nordpool: FakeNordpool, clock: Callable[[datetime], None]
) -> None:
    history = [date(2026, 9, 20), date(2026, 9, 21)]
    nordpool.published.update(history)
    nordpool.breakdown.update(history)
    start, end = delivery_day_range(history[0])[0], delivery_day_range(history[1])[1]
    source = _source(storage)

    assert await source.async_update(start, end) is True
    assert nordpool.days == history
    assert len(storage.load_series(NORDPOOL_PROGNOSES, start, end)) == 48

    clock(MORNING + timedelta(hours=6))
    assert await source.async_update(start, end) is False

    assert nordpool.days == history
    assert nordpool.production.await_count == 2


@pytest.mark.asyncio
async def test_today_and_tomorrow_are_refreshed_and_revisions_detected(
    storage: LearningStorage, nordpool: FakeNordpool, clock: Callable[[datetime], None]
) -> None:
    """Nordpool revises the current days' prognoses, so they are re-fetched."""
    nordpool.published.add(TOMORROW)
    nordpool.breakdown.add(TOMORROW)
    source = _source(storage)
    await source.async_update(START, END)

    clock(MORNING + timedelta(hours=6))
    assert await source.async_update(START, END) is False
    nordpool.revision = 25.0
    assert await source.async_update(START, END) is True

    assert nordpool.days == [TODAY, TOMORROW] * 3
    rows = storage.load_series(NORDPOOL_PROGNOSES, START, END)
    assert rows[0]["consumption"] == pytest.approx(4025.0)


@pytest.mark.asyncio
async def test_unpublished_days_wait_for_revalidation(
    storage: LearningStorage, nordpool: FakeNordpool, clock: Callable[[datetime], None]
) -> None:
    """Tomorrow is not asked for again for 15 minutes, then found once published."""
    source = _source(storage)

    assert await source.async_update(START, END) is True
    assert nordpool.days == [TODAY, TOMORROW]
    state = await source._async_state()
    assert state.horizon == END - timedelta(days=1)

    clock(MORNING + timedelta(minutes=10))
    assert await source.async_update(START, END) is False
    # A longer request does not ask past the horizon either
    assert await source.async_update(START, END + timedelta(days=1)) is False
    assert nordpool.days == [TODAY, TOMORROW, TODAY, TODAY]

    nordpool.published.add(TOMORROW)
    nordpool.breakdown.add(TOMORROW)
    clock(MORNING + timedelta(minutes=16))
    assert await source.async_update(START, END) is True

    assert nordpool.days[-2:] == [TODAY, TOMORROW]
    assert (await source._async_state()).holes == []


@pytest.mark.asyncio
async def test_a_late_production_breakdown_is_fetched_and_changes_the_data(
    storage: LearningStorage, nordpool: FakeNordpool, clock: Callable[[datetime], None]
) -> None:
    """Rows without the breakdown are incomplete; filling it is new training data."""
    nordpool.breakdown.clear()
    source = _source(storage)
    today_end = END - timedelta(days=1)
    await source.async_update(START, today_end)
    rows = storage.load_series(NORDPOOL_PROGNOSES, START, END)
    assert {row["solar"] for row in rows} == {None}
    assert await source.async_update(START, today_end) is False

    storage.last_data_write = None
    nordpool.breakdown.add(TODAY)
    clock(MORNING + timedelta(minutes=16))
    assert await source.async_update(START, today_end) is True

    assert storage.last_data_write is not None
    rows = storage.load_series(NORDPOOL_PROGNOSES, START, END)
    assert {row["solar"] for row in rows} == {50.0}
    assert None not in {row["consumption"] for row in rows}


@pytest.mark.asyncio
async def test_a_failed_request_is_retried_at_the_next_update(
    storage: LearningStorage, nordpool: FakeNordpool, clock: Callable[[datetime], None]
) -> None:
    nordpool.failing.add(TODAY)
    source = _source(storage)
    today_end = END - timedelta(days=1)

    assert await source.async_update(START, today_end) is False
    assert (await source._async_state()).holes == []

    nordpool.failing.clear()
    assert await source.async_update(START, today_end) is True
    assert nordpool.days == [TODAY, TODAY]


@pytest.mark.asyncio
async def test_history_nordpool_lacks_is_retried_daily_not_hourly(
    storage: LearningStorage, nordpool: FakeNordpool, clock: Callable[[datetime], None]
) -> None:
    old_day = date(2026, 9, 10)
    start, end = delivery_day_range(old_day)
    source = _source(storage)

    await source.async_update(start, end)
    clock(MORNING + timedelta(hours=12))
    await source.async_update(start, end)
    assert nordpool.days == [old_day]

    clock(MORNING + timedelta(days=1, minutes=1))
    await source.async_update(start, end)
    assert nordpool.days == [old_day, old_day]


@pytest.mark.asyncio
async def test_requests_are_throttled_between_days(
    storage: LearningStorage, nordpool: FakeNordpool, clock: Callable[[datetime], None]
) -> None:
    nordpool.published.add(TOMORROW)
    with patch(f"{SOURCE}.asyncio.sleep", new=AsyncMock()) as sleep:
        await _source(storage).async_update(START, END)

    sleep.assert_awaited_once_with(1.0)


@pytest.mark.asyncio
async def test_the_horizon_cutoff_hides_the_future(
    storage: LearningStorage, nordpool: FakeNordpool, clock: Callable[[datetime], None]
) -> None:
    """A backtest source never requests or returns data from the cutoff on."""
    nordpool.published.add(TOMORROW)
    await _source(storage).async_update(START, END)

    rows = await _source(storage, horizon_cutoff=END - timedelta(days=1)).async_load(
        START, END
    )

    assert len(rows) == 24
    assert (
        await _source(storage, horizon_cutoff=START).async_update(START, END) is False
    )
    assert nordpool.days == [TODAY, TOMORROW]


@pytest.mark.asyncio
async def test_prune_drops_old_rows_and_holes(
    storage: LearningStorage, nordpool: FakeNordpool, clock: Callable[[datetime], None]
) -> None:
    source = _source(storage)
    old_start, old_end = delivery_day_range(date(2026, 9, 10))
    await source.async_update(old_start, old_end)
    await source.async_update(START, END)
    assert len((await source._async_state()).holes) == 2
    storage.upsert_series(
        NORDPOOL_PROGNOSES, [{"timestamp": "2026-09-01T10:00:00Z", "consumption": 1.0}]
    )

    assert await source.async_prune(START) == 1

    state = await source._async_state()
    assert [hole.start for hole in state.holes] == [END - timedelta(days=1)]


@pytest.mark.asyncio
async def test_the_state_survives_a_restart(
    storage: LearningStorage, nordpool: FakeNordpool, clock: Callable[[datetime], None]
) -> None:
    old_day = date(2026, 9, 10)
    await _source(storage).async_update(*delivery_day_range(old_day))

    restarted = _source(storage)
    assert await restarted.async_update(*delivery_day_range(old_day)) is False
    assert nordpool.days == [old_day]
