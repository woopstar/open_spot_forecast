"""Per-source health of the update cycle (#276).

Each source records its last attempt, last success, last error and how far
its data reaches at the existing error-handling points: the forecast run's
refresh, the price reads and the backfill. A failing source shows its error
and a growing age while the others stay healthy, and no error keeps a URL,
a query parameter or a key.
"""

from collections.abc import Callable, Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest

from custom_components.open_spot_forecast.api.time_series_source import (
    TimeSeriesSource,
)
from custom_components.open_spot_forecast.history_updater import HistoryUpdaterMixin
from custom_components.open_spot_forecast.ml.series_storage import SeriesSpec
from custom_components.open_spot_forecast.ml.storage import LearningStorage
from custom_components.open_spot_forecast.source_health import (
    MAX_ERROR_LENGTH,
    SPOT_SENSOR_SOURCE,
    STROMLIGNING_SOURCE,
    SourceHealth,
    async_refresh,
    describe_error,
    latest_timestamp,
)
from custom_components.open_spot_forecast.updater import ForecastUpdater

NOW = datetime(2026, 9, 24, 10, 0, tzinfo=UTC)
SPEC = SeriesSpec(
    name="test_series", table="test_series", columns=("value",), step_minutes=15
)
QUARTER = timedelta(minutes=15)
ROW = {"timestamp": "2026-09-24T12:00:00+00:00", "value": 1.0}
LATER_ROW = {"timestamp": "2026-09-25T21:45:00Z", "value": 2.0}


class Clock:
    """``dt_util.utcnow`` under the test's control."""

    def __init__(self) -> None:
        self.now = NOW

    def __call__(self) -> datetime:
        return self.now


@pytest.fixture
def clock() -> Iterator[Clock]:
    clock = Clock()
    with patch("homeassistant.util.dt.utcnow", clock):
        yield clock


class FakeSource:
    """A stored source whose update succeeds, fails or raises on demand."""

    def __init__(self, name: str, rows: list[dict[str, Any]] | None = None) -> None:
        self.source_name = name
        self.failed_requests = 0
        self.error: Exception | None = None
        self.rows = rows if rows is not None else [ROW]

    async def async_update(self, start: datetime, end: datetime) -> bool:
        del start, end
        if self.error is not None:
            raise self.error
        return True

    async def async_load(self, start: datetime, end: datetime) -> list[dict[str, Any]]:
        del start, end
        return self.rows

    async def async_prune(self, before: datetime) -> int:
        del before
        return 0


# --- Errors -------------------------------------------------------------------------


def test_an_error_keeps_its_class_and_message_without_urls_or_keys() -> None:
    err = ValueError(
        "400, message='Bad Request', url='https://web-api.tp.entsoe.eu/api"
        "?securityToken=abc123&documentType=A44'"
    )

    text = describe_error(err)

    assert text.startswith("ValueError: 400, message='Bad Request'")
    assert "entsoe.eu" not in text
    assert "abc123" not in text
    assert "<url>" in text


def test_a_secret_parameter_outside_a_url_is_redacted() -> None:
    text = describe_error("request with api_key=s3cret and securityToken=t0ken")

    assert "s3cret" not in text
    assert "t0ken" not in text
    assert "api_key=<redacted>" in text


def test_an_error_without_a_message_is_its_class_and_long_ones_are_cut() -> None:
    assert describe_error(TimeoutError()) == "TimeoutError"
    assert len(describe_error(RuntimeError("x" * 500))) == MAX_ERROR_LENGTH


# --- The registry -------------------------------------------------------------------


def test_a_failing_source_shows_its_error_and_growing_age(clock: Clock) -> None:
    health = SourceHealth()
    health.record_success("nordpool_prognoses")
    health.record_success("openmeteo_weather")

    clock.now = NOW + timedelta(minutes=30)
    health.record_failure("nordpool_prognoses", TimeoutError())
    health.record_success("openmeteo_weather")
    clock.now = NOW + timedelta(hours=2)
    health.record_failure("nordpool_prognoses", TimeoutError())

    summary = health.summary()
    failing = summary["nordpool_prognoses"]
    assert failing["healthy"] is False
    assert failing["last_error"] == "TimeoutError"
    assert failing["last_success"] == NOW.isoformat()
    assert failing["last_attempt"] == (NOW + timedelta(hours=2)).isoformat()
    assert failing["age_minutes"] == 120
    healthy = summary["openmeteo_weather"]
    assert healthy["healthy"] is True
    assert healthy["last_error"] is None
    assert healthy["age_minutes"] == 90
    assert health.healthy_count == 1


def test_a_source_without_a_success_has_no_age(clock: Clock) -> None:
    health = SourceHealth()
    health.record_failure("gas_prices", "no answer")

    status = health.summary()["gas_prices"]

    assert status["healthy"] is False
    assert status["age_minutes"] is None
    assert status["last_error_at"] == NOW.isoformat()


def test_a_recovered_source_is_healthy_and_keeps_its_last_error(clock: Clock) -> None:
    health = SourceHealth()
    health.record_failure("gas_prices", "no answer")
    clock.now = NOW + timedelta(minutes=15)
    health.record_success("gas_prices")

    status = health.summary()["gas_prices"]

    assert status["healthy"] is True
    assert status["last_error"] == "no answer"
    assert status["age_minutes"] == 0


def test_the_latest_row_is_where_the_data_ends() -> None:
    assert latest_timestamp([LATER_ROW, ROW, {"timestamp": None}]) == datetime(
        2026, 9, 25, 21, 45, tzinfo=UTC
    )
    assert latest_timestamp([]) is None


def test_a_price_read_is_healthy_with_todays_prices(clock: Clock) -> None:
    health = SourceHealth()
    health.record_prices(
        STROMLIGNING_SOURCE,
        {"today": [1.0], "raw_today": [{"start": "2026-09-24T21:45:00+00:00"}]},
    )
    health.record_prices(SPOT_SENSOR_SOURCE, {"today": [], "tomorrow": []})

    summary = health.summary()
    assert summary[STROMLIGNING_SOURCE]["healthy"] is True
    assert summary[STROMLIGNING_SOURCE]["data_until"] == "2026-09-24T22:00:00+00:00"
    assert summary[SPOT_SENSOR_SOURCE]["healthy"] is False
    assert summary[SPOT_SENSOR_SOURCE]["last_error"] == "no prices for today"


# --- The refresh of a stored source ---------------------------------------------


@pytest.mark.asyncio
async def test_a_refresh_records_success_and_where_the_rows_end(clock: Clock) -> None:
    health = SourceHealth()
    source = FakeSource("entsoe_load", [ROW, LATER_ROW])

    rows = await async_refresh(health, source, NOW, NOW + QUARTER, "load")

    assert rows == [ROW, LATER_ROW]
    status = health.summary()["entsoe_load"]
    assert status["healthy"] is True
    assert status["data_until"] == "2026-09-25T21:45:00+00:00"


@pytest.mark.asyncio
async def test_a_failed_request_is_a_failure_with_the_stored_rows(clock: Clock) -> None:
    health = SourceHealth()
    source = FakeSource("gas_prices")
    source.failed_requests = 2

    rows = await async_refresh(health, source, NOW, NOW + QUARTER, "gas price")

    assert rows == [ROW]
    status = health.summary()["gas_prices"]
    assert status["healthy"] is False
    assert status["last_error"] == "2 request(s) failed"
    assert status["data_until"] == "2026-09-24T12:00:00+00:00"


@pytest.mark.asyncio
async def test_a_raising_update_keeps_the_others_healthy(clock: Clock) -> None:
    health = SourceHealth()
    failing = FakeSource("nordpool_prognoses")
    failing.error = RuntimeError("boom at https://example.com/api?token=x")
    working = FakeSource("openmeteo_weather")

    assert await async_refresh(health, failing, NOW, NOW + QUARTER, "a") == [ROW]
    assert await async_refresh(health, working, NOW, NOW + QUARTER, "b") == [ROW]

    summary = health.summary()
    assert summary["nordpool_prognoses"]["healthy"] is False
    assert summary["nordpool_prognoses"]["last_error"] == "RuntimeError: boom at <url>"
    assert summary["openmeteo_weather"]["healthy"] is True


@pytest.mark.asyncio
async def test_a_failed_load_is_a_failure(clock: Clock) -> None:
    health = SourceHealth()
    source = FakeSource("entsoe_load")
    source.async_load = AsyncMock(side_effect=OSError("disk"))  # type: ignore[method-assign]

    assert await async_refresh(health, source, NOW, NOW + QUARTER, "load") is None
    assert health.summary()["entsoe_load"]["last_error"] == "OSError: disk"


# --- The sources count their failed requests ----------------------------------------


class FlakySeries(TimeSeriesSource):
    """Answers every other request; the rest fail (None)."""

    spec = SPEC

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.answer = False
        self.max_request_span = timedelta(hours=1)

    async def _fetch(
        self, start: datetime, end: datetime
    ) -> list[dict[str, Any]] | None:
        self.answer = not self.answer
        if not self.answer:
            return None
        return [
            {"timestamp": (start + QUARTER * n).isoformat(), "value": 1.0}
            for n in range(int((end - start) / QUARTER))
        ]


@pytest.fixture
def storage(tmp_path: Path) -> Iterator[LearningStorage]:
    hass = Mock()
    hass.config.path.return_value = str(tmp_path / ".storage")
    store = LearningStorage(hass, "DK1")
    store._ensure_conn().execute(
        "CREATE TABLE test_series (timestamp TEXT PRIMARY KEY, value REAL)"
    )
    yield store
    store.close()


def _inline_hass() -> Mock:
    async def run_inline(func: Callable[..., Any], *args: Any) -> Any:
        return func(*args)

    hass = Mock()
    hass.async_add_executor_job = run_inline
    return hass


@pytest.mark.asyncio
async def test_a_time_series_source_counts_its_failed_requests(
    storage: LearningStorage, clock: Clock
) -> None:
    source = FlakySeries(_inline_hass(), storage)
    assert source.source_name == "test_series"

    await source.async_update(NOW, NOW + timedelta(hours=4))
    assert source.failed_requests == 2

    # The answered hours are stored; the next update counts afresh
    source.answer = False
    await source.async_update(NOW, NOW + timedelta(hours=4))
    assert source.failed_requests == 1


@pytest.mark.asyncio
async def test_storage_counts_the_rows_of_every_table(
    storage: LearningStorage, clock: Clock
) -> None:
    source = FlakySeries(_inline_hass(), storage)
    await source.async_update(NOW, NOW + timedelta(hours=1))

    counts = storage.table_row_counts()

    assert counts["test_series"] == 4
    assert counts["predictions"] == 0
    assert "meta" in counts


# --- The update cycle records at its error-handling points --------------------------


def _updater(**attributes: Any) -> ForecastUpdater:
    updater = ForecastUpdater.__new__(ForecastUpdater)
    updater.health = SourceHealth()
    updater.api_data = {}
    for name, value in attributes.items():
        setattr(updater, name, value)
    return updater


def test_the_stromligning_and_spot_sensor_reads_are_recorded(clock: Clock) -> None:
    reader = Mock()
    reader.read_stromligning_sensor.return_value = {
        "today": [1.0] * 96,
        "tomorrow": [],
        "raw_today": [],
        "raw_tomorrow": [],
    }
    reader.read_spot_prices.return_value = {
        "today": [],
        "tomorrow": [],
        "raw_today": [],
        "raw_tomorrow": [],
    }
    sensors = Mock(stromligning_tomorrow=None)
    updater = _updater(
        sensor_reader=reader,
        sensors=sensors,
        ml_predictor=Mock(),
        dayahead=None,
        tariff_changes=None,
    )

    updater.read_prices()

    summary = updater.health.summary()
    assert summary[STROMLIGNING_SOURCE]["healthy"] is True
    assert summary[SPOT_SENSOR_SOURCE]["healthy"] is False


@pytest.mark.asyncio
async def test_a_failing_day_ahead_read_is_recorded(clock: Clock) -> None:
    dayahead = Mock()
    dayahead.source = FakeSource("dayahead_prices")
    dayahead.async_read = AsyncMock(side_effect=TimeoutError())
    sensors = Mock(fallback_price=None)
    updater = _updater(dayahead=dayahead, sensors=sensors)

    assert await updater.async_read_prices() is False
    status = updater.health.summary()["dayahead_prices"]
    assert status["healthy"] is False
    assert status["last_error"] == "TimeoutError"

    dayahead.async_read = AsyncMock(
        return_value={
            "today": [0.4] * 96,
            "tomorrow": [],
            "raw_today": [{"start": "2026-09-24T21:45:00+00:00"}],
            "raw_tomorrow": [],
        }
    )
    dayahead.license_info = None
    clock.now = NOW + timedelta(minutes=15)

    assert await updater.async_read_prices() is True
    assert updater.health.summary()["dayahead_prices"]["healthy"] is True


class _Backfill(HistoryUpdaterMixin):
    """The backfill's sources on fakes."""

    def __init__(self) -> None:
        self.health = SourceHealth()


@pytest.mark.asyncio
async def test_the_backfill_records_each_source(clock: Clock) -> None:
    backfill = _Backfill()
    good, bad = FakeSource("openmeteo_weather"), FakeSource("umm_outages")
    bad.error = ConnectionError("reset")

    first = NOW.date() - timedelta(days=3)
    assert await backfill._backfill_source(good, first, "Open-Meteo") is True
    assert await backfill._backfill_source(bad, first, "UMM") is False

    summary = backfill.health.summary()
    assert summary["openmeteo_weather"]["healthy"] is True
    assert summary["umm_outages"]["last_error"] == "ConnectionError: reset"


@pytest.mark.asyncio
async def test_the_neighbour_updates_are_recorded(clock: Clock) -> None:
    backfill = _Backfill()
    good, bad = FakeSource("neighbour_prices_SE3"), FakeSource("openmeteo_SE3")
    bad.error = TimeoutError()
    backfill.neighbour_prices = [good]  # type: ignore[list-item]
    backfill.neighbour_weather = [bad]  # type: ignore[list-item]
    bad.spec = Mock(name="spec")  # type: ignore[attr-defined]

    await backfill.update_neighbours(NOW + timedelta(days=7))

    summary = backfill.health.summary()
    assert summary["neighbour_prices_SE3"]["healthy"] is True
    assert summary["openmeteo_SE3"]["healthy"] is False
