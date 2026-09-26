"""The configured price source and the day-ahead price path (#27)."""

import json
from collections.abc import Callable, Iterator
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from custom_components.open_spot_forecast.api.http import HttpResponse
from custom_components.open_spot_forecast.ml.series_storage import DAYAHEAD_PRICES
from custom_components.open_spot_forecast.ml.storage import LearningStorage
from custom_components.open_spot_forecast.price_source import (
    DayAheadPrices,
    PriceSettings,
    with_vat,
)
from custom_components.open_spot_forecast.spot_prices import (
    dayahead_prices_by_day,
    dayahead_spot_data,
)
from custom_components.open_spot_forecast.time_slots import local_midnight

pytestmark = pytest.mark.usefixtures("copenhagen_time_zone")

KEY = "entsoe-secret-token"
TODAY = date(2026, 9, 24)
NOW = datetime(2026, 9, 24, 8, tzinfo=UTC)
ECB_CSV = "TIME_PERIOD,OBS_VALUE\n2026-08-01,7.46\n2026-09-23,7.5\n"


def _entry(data: dict, options: dict | None = None) -> MagicMock:
    entry = MagicMock()
    entry.data = data
    entry.options = options or {}
    return entry


def _rows(start: datetime, count: int, price: float = 100.0) -> list[dict[str, Any]]:
    """Return ``count`` quarter-hour rows from ``start``, stepped in UTC."""
    first = start.astimezone(UTC)
    return [
        {"timestamp": (first + i * timedelta(minutes=15)).isoformat(), "price": price}
        for i in range(count)
    ]


# --- Settings ------------------------------------------------------------------------


def test_settings_prefer_options_and_hide_the_key() -> None:
    settings = PriceSettings.from_entry(
        _entry(
            {"price_source": "stromligning", "currency": "DKK", "vat": 0.25},
            {"price_source": "dayahead", "entsoe_api_key": KEY, "vat": 0.0},
        )
    )

    assert settings.dayahead
    assert settings.currency == "DKK"
    assert settings.vat == pytest.approx(0.0)
    assert settings.entsoe_api_key == KEY
    assert KEY not in repr(settings)


def test_settings_default_to_stromligning_without_a_key() -> None:
    settings = PriceSettings.from_entry(_entry({"entsoe_api_key": ""}))

    assert not settings.dayahead
    assert settings.entsoe_api_key is None
    assert settings.vat == pytest.approx(0.25)


def test_with_vat_keeps_missing_slots() -> None:
    assert with_vat([1.0, None], 0.25) == [pytest.approx(1.25), None]


# --- Converting stored prices --------------------------------------------------------


def test_spot_data_converts_eur_per_mwh_to_currency_per_kwh() -> None:
    today_start = local_midnight(TODAY)
    rows = _rows(today_start, 96) + _rows(today_start + timedelta(days=1), 48)

    data = dayahead_spot_data(rows, lambda _day: 7.5, TODAY)

    assert data["today"] == [pytest.approx(0.75)] * 96
    # Tomorrow's second half is missing (not published yet)
    assert data["tomorrow"][:48] == [pytest.approx(0.75)] * 48
    assert data["tomorrow"][48:] == [None] * 48
    assert data["raw_today"][0] == {"start": today_start.astimezone(UTC).isoformat()}
    assert len(data["raw_tomorrow"]) == 48


def test_spot_data_needs_a_rate_and_valid_prices() -> None:
    today_start = local_midnight(TODAY)
    rows = _rows(today_start, 96, price=0.0) + _rows(
        today_start + timedelta(days=1), 96
    )

    data = dayahead_spot_data(rows, lambda day: None if day > TODAY else 7.5, TODAY)

    assert data == {"today": [], "tomorrow": [], "raw_today": [], "raw_tomorrow": []}


def test_prices_by_day_follow_local_days_incl_dst() -> None:
    fall_back = date(2026, 10, 25)
    rows = _rows(local_midnight(fall_back), 100) + [{"timestamp": "bad", "price": 1.0}]

    days = dayahead_prices_by_day(rows, lambda _day: 1.0)

    assert list(days) == [fall_back]
    assert len(days[fall_back]) == 100


# --- The day-ahead path --------------------------------------------------------------


@pytest.fixture
def storage(tmp_path: Path) -> Iterator[LearningStorage]:
    hass = Mock()
    hass.config.path.return_value = str(tmp_path / ".storage")
    store = LearningStorage(hass, "DK1")
    yield store
    store.close()


@pytest.fixture
def prices(storage: LearningStorage) -> Iterator[DayAheadPrices]:
    """energy-charts publishes every requested day; the ECB answers."""

    async def get(_session: Any, _url: str, label: str, **kw: Any) -> HttpResponse:
        if label == "ECB exchange rate":
            return HttpResponse(200, ECB_CSV)
        start = datetime.fromisoformat(kw["params"]["start"])
        end = datetime.fromisoformat(kw["params"]["end"])
        count = int((end - start) / timedelta(minutes=15)) + 1
        first = int(start.timestamp())
        payload = {
            "unix_seconds": [first + 900 * i for i in range(count)],
            "price": [80.0] * count,
            "unit": "EUR / MWh",
        }
        return HttpResponse(200, json.dumps(payload))

    async def run_inline(func: Callable[..., Any], *args: Any) -> Any:
        return func(*args)

    hass = Mock()
    hass.async_add_executor_job = run_inline
    with (
        patch(
            "custom_components.open_spot_forecast.api.dayahead_prices.async_get",
            new=get,
        ),
        patch(
            "custom_components.open_spot_forecast.api.exchange_rates.async_get", new=get
        ),
        patch(
            "custom_components.open_spot_forecast.api.dayahead_prices.async_get_clientsession"
        ),
        patch(
            "custom_components.open_spot_forecast.api.exchange_rates.async_get_clientsession"
        ),
        patch(
            "custom_components.open_spot_forecast.api.time_series_source.asyncio.sleep",
            new=AsyncMock(),
        ),
        patch("homeassistant.util.dt.utcnow", return_value=NOW),
    ):
        yield DayAheadPrices(
            hass, storage, "DK1", PriceSettings("dayahead", "DKK", 0.25, KEY)
        )


@pytest.mark.asyncio
async def test_read_returns_today_and_tomorrow_in_the_currency(
    prices: DayAheadPrices,
) -> None:
    data = await prices.async_read(TODAY)

    # 80 EUR/MWh at 7.5 DKK/EUR = 0.6 DKK/kWh
    assert data["today"] == [pytest.approx(0.6)] * 96
    assert data["tomorrow"] == [pytest.approx(0.6)] * 96
    assert prices.license_info is None


@pytest.mark.asyncio
async def test_history_returns_every_day_before_today(
    prices: DayAheadPrices, storage: LearningStorage
) -> None:
    days = await prices.async_history(TODAY - timedelta(days=30), TODAY)

    assert list(days) == [TODAY - timedelta(days=d) for d in range(30, 0, -1)]
    # A day before the first ECB rate in the window takes the oldest known one
    assert days[TODAY - timedelta(days=30)][0] == pytest.approx(80 * 7.46 / 1000)
    assert (
        await prices.async_prune(local_midnight(TODAY - timedelta(days=10))) == 20 * 96
    )
    assert storage.load_series(DAYAHEAD_PRICES, local_midnight(TODAY), NOW) == []
