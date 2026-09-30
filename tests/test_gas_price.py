"""The natural-gas price feature (#28): source, lookup and update cycle."""

import json
import logging
from collections.abc import Callable, Iterator
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock, patch
from zoneinfo import ZoneInfo

import pytest

from custom_components.open_spot_forecast.api.gas_prices import (
    GasPriceSource,
    instrat_query,
    parse_instrat_gas,
)
from custom_components.open_spot_forecast.api.http import HttpResponse
from custom_components.open_spot_forecast.attribution import model_attribution
from custom_components.open_spot_forecast.const import (
    GAS_PRICE_REGIONS,
    INSTRAT_USER_AGENT,
)
from custom_components.open_spot_forecast.ml.gas_price import (
    GAS_LOOKBACK_DAYS,
    GasPriceIndex,
)
from custom_components.open_spot_forecast.ml.series_storage import GAS_PRICES
from custom_components.open_spot_forecast.ml.storage import LearningStorage
from custom_components.open_spot_forecast.price_source import PriceSettings
from custom_components.open_spot_forecast.updater import (
    ForecastUpdater,
    SensorEntities,
)

MODULE = "custom_components.open_spot_forecast.api.gas_prices"
UPDATER = "custom_components.open_spot_forecast.updater"
CPH = ZoneInfo("Europe/Copenhagen")
NOW = datetime(2026, 9, 24, 10, tzinfo=UTC)


def _row(day: str, price: float) -> dict[str, Any]:
    return {"timestamp": f"{day}T00:00:00Z", "price": price}


# --- The lookup -------------------------------------------------------------------


def test_a_day_gets_the_latest_price_dated_before_it() -> None:
    index = GasPriceIndex(
        [
            _row("2026-09-20", 300.0),
            _row("2026-09-22", 320.0),
            _row("2026-09-23", 330.0),
        ]
    )

    assert index.before(date(2026, 9, 23)) == pytest.approx(320.0)
    assert index.before(date(2026, 9, 22)) == pytest.approx(300.0)
    assert index.before(date(2026, 9, 21)) == pytest.approx(300.0)
    # A forecast days ahead gets the latest published price
    assert index.before(date(2026, 9, 30)) == pytest.approx(330.0)
    assert index.before(date(2026, 9, 20)) is None


def test_a_stale_price_is_not_used() -> None:
    index = GasPriceIndex([_row("2026-09-01", 300.0)])

    last_day = date(2026, 9, 1) + timedelta(days=GAS_LOOKBACK_DAYS)
    assert index.before(last_day) == pytest.approx(300.0)
    assert index.before(last_day + timedelta(days=1)) is None


def test_unusable_rows_are_skipped() -> None:
    index = GasPriceIndex(
        [{"timestamp": "garbage", "price": 1.0}, _row("2026-09-20", float("nan")), {}]
    )

    assert index.before(date(2026, 9, 25)) is None


# --- Instrat ----------------------------------------------------------------------


def test_instrat_entries_become_daily_rows() -> None:
    payload = [
        {"date": "2026-09-04T00:00:00Z", "indeks": "tgegasda", "price": 319.73},
        {"date": "2026-09-05T00:00:00Z", "price": 331},
        {"date": "2026-09-06T00:00:00Z", "price": None},
        {"date": "2026-09-07T00:00:00Z", "price": True},
        {"date": "not a date", "price": 1.0},
    ]

    assert parse_instrat_gas(payload) == [
        _row("2026-09-04", 319.73),
        _row("2026-09-05", 331),
    ]


@pytest.mark.parametrize("payload", [{"error": "x"}, "text", [1, 2]])
def test_an_unexpected_instrat_payload_is_rejected(payload: Any) -> None:
    with pytest.raises(TypeError):
        parse_instrat_gas(payload)


def test_the_query_asks_for_a_day_more_on_each_side() -> None:
    query = instrat_query(
        datetime(2026, 9, 5, tzinfo=UTC), datetime(2026, 9, 7, tzinfo=UTC)
    )

    assert query["date_from"] == "04-09-2026T00:00:00Z"
    assert query["date_to"] == "08-09-2026T00:00:00Z"
    assert query["aggregation_timeframe"] == "day"


# --- The source ---------------------------------------------------------------------


@pytest.fixture
def storage(tmp_path: Path) -> Iterator[LearningStorage]:
    hass = Mock()
    hass.config.path.return_value = str(tmp_path / ".storage")
    store = LearningStorage(hass, "DK1")
    yield store
    store.close()


def _source(storage: LearningStorage) -> GasPriceSource:
    async def run_inline(func: Callable[..., Any], *args: Any) -> Any:
        return func(*args)

    hass = Mock()
    hass.async_add_executor_job = run_inline
    return GasPriceSource(hass, storage)


class FakeInstrat:
    """Answers with every day of the query up to the day before NOW."""

    def __init__(self) -> None:
        self.status = 200
        self.text: str | None = None
        self.queries: list[dict[str, str]] = []
        self.headers: list[dict[str, str] | None] = []

    async def get(self, _session: Any, url: str, label: str, **kw: Any) -> Any:
        assert label == "Instrat"
        self.queries.append(kw["params"])
        self.headers.append(kw.get("headers"))
        if self.text is not None:
            return HttpResponse(self.status, self.text)
        first = datetime.strptime(kw["params"]["date_from"], "%d-%m-%YT%H:%M:%SZ")
        last = datetime.strptime(kw["params"]["date_to"], "%d-%m-%YT%H:%M:%SZ")
        days = [first + timedelta(days=n) for n in range((last - first).days)]
        payload = [
            {"date": f"{day:%Y-%m-%d}T00:00:00Z", "price": 300.0 + day.day}
            for day in days
            if day.date() < NOW.date()
        ]
        return HttpResponse(self.status, json.dumps(payload))


@pytest.fixture
def instrat() -> Iterator[FakeInstrat]:
    fake = FakeInstrat()
    with (
        patch(f"{MODULE}.async_get", new=fake.get),
        patch(f"{MODULE}.async_get_clientsession"),
        patch("homeassistant.util.dt.utcnow", return_value=NOW),
    ):
        yield fake


@pytest.mark.asyncio
async def test_the_days_of_a_range_are_stored(
    storage: LearningStorage, instrat: FakeInstrat
) -> None:
    source = _source(storage)
    start = datetime(2026, 9, 10, tzinfo=UTC)
    end = datetime(2026, 9, 25, tzinfo=UTC)

    assert await source.async_update(start, end) is True

    rows = storage.load_series(GAS_PRICES, start, end)
    # Only the range's days, up to the latest published one
    assert [row["timestamp"][:10] for row in rows] == [
        f"2026-09-{day}" for day in range(10, 24)
    ]
    assert rows[0]["price"] == pytest.approx(310.0)
    # Stored days are not asked for again; the unpublished day waits
    assert await source.async_update(start, end) is False
    assert len(instrat.queries) == 1


@pytest.mark.asyncio
async def test_requests_do_not_carry_home_assistants_user_agent(
    storage: LearningStorage, instrat: FakeInstrat
) -> None:
    """Instrat's Cloudflare answers 403 to ``HomeAssistant/... aiohttp/...``."""
    start = datetime(2026, 9, 10, tzinfo=UTC)

    await _source(storage).async_update(start, start + timedelta(days=5))

    assert instrat.headers == [{"User-Agent": INSTRAT_USER_AGENT}]
    assert "HomeAssistant" not in INSTRAT_USER_AGENT


@pytest.mark.asyncio
async def test_a_failing_source_warns_once_and_stores_nothing(
    storage: LearningStorage, instrat: FakeInstrat, caplog: pytest.LogCaptureFixture
) -> None:
    source = _source(storage)
    start = datetime(2026, 9, 10, tzinfo=UTC)
    instrat.status, instrat.text = 503, ""

    with caplog.at_level(logging.DEBUG):
        assert await source.async_update(start, start + timedelta(days=5)) is False
        assert await source.async_update(start, start + timedelta(days=5)) is False

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert [r.getMessage() for r in warnings] == [
        "Gas prices unavailable (Instrat: 503)"
    ]
    assert storage.load_series(GAS_PRICES, start, start + timedelta(days=5)) == []


@pytest.mark.asyncio
async def test_an_unusable_answer_is_logged(
    storage: LearningStorage, instrat: FakeInstrat, caplog: pytest.LogCaptureFixture
) -> None:
    source = _source(storage)
    instrat.text = json.dumps({"detail": "changed"})
    start = datetime(2026, 9, 10, tzinfo=UTC)

    assert await source.async_update(start, start + timedelta(days=5)) is False

    assert "Unusable Instrat gas prices" in caplog.text


@pytest.mark.asyncio
async def test_no_answer_at_all_is_logged(
    storage: LearningStorage, caplog: pytest.LogCaptureFixture
) -> None:
    source = _source(storage)
    start = datetime(2026, 9, 10, tzinfo=UTC)

    with (
        patch(f"{MODULE}.async_get", new=AsyncMock(return_value=None)),
        patch(f"{MODULE}.async_get_clientsession"),
    ):
        assert await source.async_update(start, start + timedelta(days=2)) is False

    assert "Gas prices unavailable (Instrat: no answer)" in caplog.text


# --- The update cycle -----------------------------------------------------------------


def _predictor() -> Mock:
    predictor = Mock()
    predictor.cross_border = None
    predictor.max_history_days = 30
    predictor.price_history = [{"date": "2026-09-20"}]
    predictor.storage.delete_old_weather.return_value = 0
    predictor.storage.delete_old_prices.return_value = 0
    return predictor


def _updater(region: str, predictor: Mock | None) -> ForecastUpdater:
    async def run_inline(func: Callable[..., Any], *args: Any) -> Any:
        return func(*args)

    hass = Mock()
    hass.async_add_executor_job = run_inline
    sensors = SensorEntities(*([None] * 9))
    return ForecastUpdater(
        hass,
        MagicMock(),
        {"region": region, "sensor_config": sensors.sensor_config()},
        sensors,
        Mock(),
        predictor,
        PriceSettings("dayahead", "EUR"),
        predictor.storage if predictor else Mock(),
    )


@pytest.fixture
def gas() -> Iterator[Mock]:
    source = Mock()
    source.async_update = AsyncMock(return_value=True)
    source.async_load = AsyncMock(return_value=[_row("2026-09-23", 310.0)])
    source.async_prune = AsyncMock(return_value=0)
    with patch(f"{UPDATER}.GasPriceSource", return_value=source):
        yield source


def test_only_regions_where_it_helps_fetch_the_gas_price(gas: Mock) -> None:
    assert "DK1" in GAS_PRICE_REGIONS
    assert "SE3" not in GAS_PRICE_REGIONS

    assert _updater("DK1", _predictor()).gas is gas
    assert _updater("DK1", _predictor()).api_data["gas_price"] is True
    assert _updater("SE3", _predictor()).gas is None
    assert _updater("DK1", None).gas is None


@pytest.mark.asyncio
@pytest.mark.usefixtures("copenhagen_time_zone")
async def test_forecast_runs_attach_the_recent_gas_prices(gas: Mock) -> None:
    updater = _updater("DK1", _predictor())
    weather_data: dict[str, Any] = {}

    with patch("homeassistant.util.dt.now", return_value=NOW.astimezone(CPH)):
        await updater.update_gas_price(weather_data)

    window = (
        datetime(2026, 9, 24, tzinfo=CPH) - timedelta(days=GAS_LOOKBACK_DAYS),
        datetime(2026, 9, 25, tzinfo=CPH),
    )
    gas.async_update.assert_awaited_once_with(*window)
    assert weather_data == {"gas_price": [_row("2026-09-23", 310.0)]}

    gas.async_load.return_value = []
    weather_data = {}
    await updater.update_gas_price(weather_data)
    assert weather_data == {}


@pytest.mark.asyncio
@pytest.mark.usefixtures("copenhagen_time_zone")
async def test_the_gas_history_is_backfilled_and_pruned(gas: Mock) -> None:
    """The training window's gas prices, plus the lookback before its first day."""
    updater = _updater("DK1", _predictor())
    updater.history_prices = updater.weather = updater.load = None
    updater.nordpool = Mock(async_update=AsyncMock(return_value=False))
    updater.nordpool.async_prune = AsyncMock(return_value=0)

    with (
        patch("homeassistant.util.dt.now", return_value=NOW.astimezone(CPH)),
        patch.object(ForecastUpdater, "refresh_forecast", autospec=True) as refresh,
    ):
        await updater.backfill_history()
        await updater.prune_history()

    first = datetime(2026, 9, 20, tzinfo=CPH) - timedelta(days=GAS_LOOKBACK_DAYS)
    gas.async_update.assert_awaited_once_with(first, datetime(2026, 9, 24, tzinfo=CPH))
    refresh.assert_awaited_once()
    gas.async_prune.assert_awaited_once_with(
        datetime(2026, 8, 23, tzinfo=CPH) - timedelta(days=GAS_LOOKBACK_DAYS)
    )


def test_the_model_credits_the_gas_price() -> None:
    assert model_attribution({"ml_predictor": Mock(), "gas_price": True}) == (
        "Prognoses: Nord Pool · Gas price: Instrat (CC BY-NC 4.0)"
    )


@pytest.mark.usefixtures("copenhagen_time_zone")
def test_predictions_use_the_attached_gas_prices(tmp_path: Path) -> None:
    """Every forecast slot gets the latest price published before its day."""
    from custom_components.open_spot_forecast.ml.predictor import SpotPricePredictor

    hass = Mock()
    hass.config.path.return_value = str(tmp_path / ".storage")
    predictor = SpotPricePredictor(hass, "DK1")
    rows: list[list[dict[str, Any]]] = []
    combine = predictor._combine_features

    def spy(*args: Any) -> list[dict[str, Any]]:
        rows.append(combine(*args))
        return rows[-1]

    today = datetime.now(CPH).date()
    gas_rows = [_row(str(today - timedelta(days=2)), 305.0)]
    try:
        with patch.object(predictor, "_combine_features", side_effect=spy):
            predictor.predict({"gas_price": gas_rows}, [0.5] * 96, forecast_days=2)
    finally:
        predictor.storage.close()

    assert rows[0]
    assert {row["gas_price"] for row in rows[0]} == {305.0}
