"""Open-Meteo zone weather: the source, its parser and the aggregates (#22)."""

import json
from collections.abc import Callable, Iterator
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, Mock, patch
from zoneinfo import ZoneInfo

import pytest

from custom_components.open_spot_forecast.api.http import HttpResponse
from custom_components.open_spot_forecast.api.openmeteo_weather import (
    OpenMeteoWeatherSource,
    open_meteo_query,
    parse_open_meteo,
)
from custom_components.open_spot_forecast.const import (
    OPEN_METEO_API,
    OPEN_METEO_ARCHIVE_API,
    WEATHER_POINTS,
)
from custom_components.open_spot_forecast.ml.features import wind_power_curve
from custom_components.open_spot_forecast.ml.series_storage import OPENMETEO_WEATHER
from custom_components.open_spot_forecast.ml.storage import LearningStorage
from custom_components.open_spot_forecast.ml.zone_weather import (
    ZoneWeatherIndex,
    point_key,
    zone_aggregates,
    zone_points,
)

MODULE = "custom_components.open_spot_forecast.api.openmeteo_weather"
SOURCE = "custom_components.open_spot_forecast.api.time_series_source"
POINTS = ((57.40, 10.24), (56.20, 8.42))
NOW = datetime(2026, 9, 24, 8, 20, tzinfo=UTC)


def _forecast(first: datetime, count: int, wind: float = 8.0) -> dict[str, Any]:
    times = [
        (first + i * timedelta(minutes=15)).strftime("%Y-%m-%dT%H:%M")
        for i in range(count)
    ]
    return {
        "latitude": 57.397,
        "minutely_15": {
            "time": times,
            "wind_speed_80m": [wind] * count,
            "temperature_2m": [12.0] * count,
            "global_tilted_irradiance": [100.0] * count,
            "pressure_msl": [1013.0] * count,
            "relative_humidity_2m": [80] * count,
        },
    }


# --- Parsing -------------------------------------------------------------------------


def test_parse_rows_per_point_in_utc() -> None:
    first = datetime(2026, 9, 24, tzinfo=UTC)
    second = _forecast(first, 2, wind=5.0)
    second["minutely_15"]["temperature_2m"] = [None]  # shorter array
    rows = parse_open_meteo([_forecast(first, 2), second], POINTS)

    assert len(rows) == 4
    assert rows[0] == {
        "timestamp": "2026-09-24T00:00:00+00:00",
        "point": "57.40,10.24",
        "wind_80m": 8.0,
        "temperature": 12.0,
        "irradiance": 100.0,
        "pressure": 1013.0,
        "humidity": 80,
    }
    assert rows[2]["point"] == "56.20,8.42"
    assert rows[3]["temperature"] is None


def test_parse_a_single_point_and_skip_empty_slots() -> None:
    forecast = _forecast(datetime(2026, 9, 24, tzinfo=UTC), 2)
    for name in ("wind_speed_80m", "temperature_2m", "global_tilted_irradiance"):
        forecast["minutely_15"][name] = [1.0, None]
    for name in ("pressure_msl", "relative_humidity_2m"):
        forecast["minutely_15"][name] = [1.0, None]

    assert len(parse_open_meteo(forecast, POINTS[:1])) == 1
    with pytest.raises(ValueError, match="2 points"):
        parse_open_meteo([forecast], POINTS)


def test_query_lists_every_point_in_utc() -> None:
    query = open_meteo_query(POINTS, date(2026, 9, 23), date(2026, 10, 2))

    assert query["latitude"] == "57.40,56.20"
    assert query["longitude"] == "10.24,8.42"
    assert query["start_date"] == "2026-09-23"
    assert query["end_date"] == "2026-10-02"
    assert query["timezone"] == "UTC"
    assert query["wind_speed_unit"] == "ms"
    assert query["minutely_15"].split(",")[0] == "wind_speed_80m"


# --- Aggregates ----------------------------------------------------------------------


def test_zone_aggregates_average_the_points() -> None:
    rows: list[dict[str, Any]] = [
        {"wind_80m": 4.0, "temperature": 10.0, "irradiance": None},
        {"wind_80m": 14.0, "temperature": 14.0, "irradiance": 200.0, "pressure": 1000},
    ]

    zone = zone_aggregates(rows)

    assert zone["zone_wind"] == pytest.approx(9.0)
    # The mean of the power curve, not the curve of the mean wind
    assert zone["zone_wind_power"] == pytest.approx(
        (wind_power_curve(4.0) + wind_power_curve(14.0)) / 2
    )
    assert zone["zone_temperature"] == pytest.approx(12.0)
    assert zone["zone_irradiance"] == pytest.approx(200.0)
    assert zone["zone_pressure"] == pytest.approx(1000.0)
    assert zone["zone_humidity"] is None
    assert set(zone_aggregates([]).values()) == {None}


def test_the_index_matches_slots_and_the_regions_points() -> None:
    start = datetime(2026, 9, 24, 10, 15, tzinfo=ZoneInfo("Europe/Copenhagen"))
    rows = [
        {"timestamp": "2026-09-24T08:15:00Z", "point": "a", "wind_80m": 5.0},
        {"timestamp": "2026-09-24T08:15:00+00:00", "point": "b", "wind_80m": 7.0},
        {"timestamp": "2026-09-24T08:30:00Z", "point": "a", "wind_80m": 20.0},
        {"timestamp": "bad", "point": "a", "wind_80m": 1.0},
    ]

    assert ZoneWeatherIndex(rows).for_slot(start)["zone_wind"] == pytest.approx(6.0)
    only_a = ZoneWeatherIndex(rows, ["a"])
    assert only_a.for_slot(start)["zone_wind"] == pytest.approx(5.0)
    assert only_a.for_slot(start - timedelta(hours=1))["zone_wind"] is None


def test_every_region_has_sampling_points() -> None:
    from custom_components.open_spot_forecast.const import REGIONS

    assert set(WEATHER_POINTS) == set(REGIONS)
    assert zone_points("DK1")[0] == point_key((57.40, 10.24)) == "57.40,10.24"
    assert zone_points("XX") == []


# --- The source ----------------------------------------------------------------------


@pytest.fixture
def storage(tmp_path: Path) -> Iterator[LearningStorage]:
    hass = Mock()
    hass.config.path.return_value = str(tmp_path / ".storage")
    store = LearningStorage(hass, "DK1")
    yield store
    store.close()


def _source(storage: LearningStorage) -> OpenMeteoWeatherSource:
    async def run_inline(func: Callable[..., Any], *args: Any) -> Any:
        return func(*args)

    hass = Mock()
    hass.async_add_executor_job = run_inline
    return OpenMeteoWeatherSource(hass, storage, "DK1")


class FakeOpenMeteo:
    """Answers every requested UTC day for every point; records the queries."""

    def __init__(self, source: OpenMeteoWeatherSource) -> None:
        self.source = source
        self.queries: list[dict[str, str]] = []
        self.urls: list[str] = []
        self.wind = 8.0
        self.status = 200

    async def get(self, _session: Any, url: str, _label: str, **kw: Any) -> Any:
        # Requests are serialised across sources
        assert self.source._request_lock.locked()
        params = kw["params"]
        self.queries.append(params)
        self.urls.append(url)
        first = datetime.fromisoformat(params["start_date"]).replace(tzinfo=UTC)
        days = (date.fromisoformat(params["end_date"]) - first.date()).days + 1
        payload = [_forecast(first, days * 96, self.wind) for _ in self.source.points]
        return HttpResponse(self.status, json.dumps(payload))


@pytest.fixture
def api(
    storage: LearningStorage,
) -> Iterator[tuple[OpenMeteoWeatherSource, FakeOpenMeteo]]:
    source = _source(storage)
    fake = FakeOpenMeteo(source)
    with (
        patch(f"{MODULE}.async_get", new=fake.get),
        patch(f"{MODULE}.async_get_clientsession"),
        patch(f"{SOURCE}.asyncio.sleep", new=AsyncMock()),
        patch("homeassistant.util.dt.utcnow", return_value=NOW),
    ):
        yield source, fake


@pytest.mark.asyncio
async def test_the_zone_is_fetched_in_one_request_and_stored(
    api: tuple[OpenMeteoWeatherSource, FakeOpenMeteo], storage: LearningStorage
) -> None:
    source, fake = api
    start = datetime(2026, 9, 23, tzinfo=UTC)
    end = start + timedelta(days=9)

    assert await source.async_update(start, end) is True

    assert len(fake.queries) == 1
    assert (fake.queries[0]["start_date"], fake.queries[0]["end_date"]) == (
        "2026-09-23",
        "2026-10-01",
    )
    assert len(fake.queries[0]["latitude"].split(",")) == 4
    rows = storage.load_series(OPENMETEO_WEATHER, start, end)
    assert len(rows) == 9 * 96 * 4


@pytest.mark.asyncio
async def test_recent_forecasts_are_refreshed_older_ones_kept(
    api: tuple[OpenMeteoWeatherSource, FakeOpenMeteo], storage: LearningStorage
) -> None:
    """From yesterday on the forecast is re-fetched; complete older days are not."""
    source, fake = api
    start = datetime(2026, 9, 20, tzinfo=UTC)
    end = datetime(2026, 9, 26, tzinfo=UTC)
    await source.async_update(start, end)

    fake.wind = 12.0
    assert await source.async_update(start, end) is True

    assert fake.queries[1]["start_date"] == "2026-09-23"
    old = storage.load_series(OPENMETEO_WEATHER, start, start + timedelta(hours=1))
    new = storage.load_series(OPENMETEO_WEATHER, NOW, NOW + timedelta(hours=1))
    assert {row["wind_80m"] for row in old} == {8.0}
    assert {row["wind_80m"] for row in new} == {12.0}


@pytest.mark.asyncio
async def test_failures_are_logged_and_store_nothing(
    api: tuple[OpenMeteoWeatherSource, FakeOpenMeteo],
    storage: LearningStorage,
    caplog: pytest.LogCaptureFixture,
) -> None:
    source, fake = api
    fake.status = 500
    await source.async_update(NOW, NOW + timedelta(hours=1))

    # A failed request is retried at the next update
    wrong = AsyncMock(return_value=HttpResponse(200, "[{}]"))
    with patch(f"{MODULE}.async_get", new=wrong):
        await source.async_update(NOW, NOW + timedelta(hours=1))

    assert "Open-Meteo returned 500" in caplog.text
    assert "Unusable Open-Meteo response: Open-Meteo returned 1" in caplog.text
    assert storage.load_series(OPENMETEO_WEATHER, NOW, NOW + timedelta(days=1)) == []


def test_requests_are_whole_utc_days_up_to_90(storage: LearningStorage) -> None:
    source = _source(storage)
    start = datetime(2026, 1, 1, 5, tzinfo=UTC)

    with patch("homeassistant.util.dt.utcnow", return_value=NOW):
        chunks = source.chunks([(start, start + timedelta(days=100))])

    assert chunks == [
        (datetime(2026, 1, 1, tzinfo=UTC), datetime(2026, 4, 1, tzinfo=UTC)),
        (datetime(2026, 4, 1, tzinfo=UTC), datetime(2026, 4, 12, tzinfo=UTC)),
    ]
    assert source.refresh_from(NOW) == NOW - timedelta(days=1)


@pytest.mark.asyncio
async def test_days_before_yesterday_come_from_the_archive(
    api: tuple[OpenMeteoWeatherSource, FakeOpenMeteo], storage: LearningStorage
) -> None:
    """Training history is archived forecasts; the rest is the live forecast (#23)."""
    source, fake = api
    start = datetime(2026, 9, 1, tzinfo=UTC)
    end = datetime(2026, 9, 26, tzinfo=UTC)

    await source.async_update(start, end)

    assert fake.urls == [OPEN_METEO_ARCHIVE_API, OPEN_METEO_API]
    assert [(q["start_date"], q["end_date"]) for q in fake.queries] == [
        ("2026-09-01", "2026-09-22"),
        ("2026-09-23", "2026-09-25"),
    ]
    assert len(storage.load_series(OPENMETEO_WEATHER, start, end)) == 25 * 96 * 4
    assert OpenMeteoWeatherSource.archive_before(NOW) == datetime(
        2026, 9, 23, tzinfo=UTC
    )
