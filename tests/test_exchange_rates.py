"""ECB euro reference rates for day-ahead prices (#27)."""

from collections.abc import Iterator
from datetime import date
from unittest.mock import AsyncMock, Mock, patch

import pytest

from custom_components.open_spot_forecast.api.exchange_rates import (
    PEGGED_EUR_RATES,
    ExchangeRates,
    parse_ecb_csv,
)
from custom_components.open_spot_forecast.api.http import HttpResponse

MODULE = "custom_components.open_spot_forecast.api.exchange_rates"
CSV = (
    "KEY,FREQ,CURRENCY,CURRENCY_DENOM,EXR_TYPE,EXR_SUFFIX,TIME_PERIOD,OBS_VALUE,TITLE\n"
    'EXR.D.SEK.EUR.SP00.A,D,SEK,EUR,SP00,A,2026-09-24,10.7895,"Swedish krona, Euro"\n'
    'EXR.D.SEK.EUR.SP00.A,D,SEK,EUR,SP00,A,2026-09-25,10.84,"Swedish krona, Euro"\n'
    "EXR.D.SEK.EUR.SP00.A,D,SEK,EUR,SP00,A,bad,1.0,x\n"
    "EXR.D.SEK.EUR.SP00.A,D,SEK,EUR,SP00,A,2026-09-26,,x\n"
)
TODAY = date(2026, 9, 27)


@pytest.fixture
def get() -> Iterator[AsyncMock]:
    with (
        patch(f"{MODULE}.async_get", new=AsyncMock()) as mock,
        patch(f"{MODULE}.async_get_clientsession"),
    ):
        mock.return_value = HttpResponse(200, CSV)
        yield mock


def test_parse_ecb_csv_skips_unusable_rows() -> None:
    assert parse_ecb_csv(CSV) == {
        date(2026, 9, 24): pytest.approx(10.7895),
        date(2026, 9, 25): pytest.approx(10.84),
    }
    assert parse_ecb_csv("") == {}


@pytest.mark.asyncio
async def test_a_day_uses_its_rate_or_the_latest_before(get: AsyncMock) -> None:
    rates = ExchangeRates(Mock(), "SEK")

    await rates.async_update(date(2026, 9, 24), TODAY)

    assert rates.rate(date(2026, 9, 24)) == pytest.approx(10.7895)
    # A weekend day, and tomorrow, take the latest published rate
    assert rates.rate(date(2026, 9, 27)) == pytest.approx(10.84)
    assert rates.rate(date(2026, 9, 20)) is None
    call = get.await_args
    assert call is not None
    assert call.kwargs["params"] == {"startPeriod": "2026-09-17", "format": "csvdata"}
    assert call.args[1].endswith("/D.SEK.EUR.SP00.A")


@pytest.mark.asyncio
async def test_rates_are_fetched_once_a_day_unless_older_days_are_needed(
    get: AsyncMock,
) -> None:
    rates = ExchangeRates(Mock(), "SEK")

    await rates.async_update(date(2026, 9, 24), TODAY)
    await rates.async_update(date(2026, 9, 25), TODAY)
    assert get.await_count == 1

    await rates.async_update(date(2026, 8, 1), TODAY)
    await rates.async_update(date(2026, 9, 25), date(2026, 9, 28))
    assert get.await_count == 3


@pytest.mark.asyncio
async def test_euro_needs_no_rates(get: AsyncMock) -> None:
    rates = ExchangeRates(Mock(), "EUR")

    await rates.async_update(date(2026, 9, 24), TODAY)

    assert rates.rate(date(2026, 9, 24)) == pytest.approx(1.0)
    get.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response", [None, HttpResponse(500, ""), HttpResponse(200, "")]
)
async def test_without_the_ecb_only_the_pegged_krone_has_a_rate(
    get: AsyncMock, response: HttpResponse | None, caplog: pytest.LogCaptureFixture
) -> None:
    get.return_value = response
    krone = ExchangeRates(Mock(), "DKK")
    krona = ExchangeRates(Mock(), "SEK")

    await krone.async_update(date(2026, 9, 24), TODAY)
    await krona.async_update(date(2026, 9, 24), TODAY)

    assert krone.rate(date(2026, 9, 24)) == pytest.approx(PEGGED_EUR_RATES["DKK"])
    assert krona.rate(date(2026, 9, 24)) is None
    assert "Could not fetch the EUR/SEK exchange rate" in caplog.text
    # Not fetched: the next update asks again
    await krona.async_update(date(2026, 9, 24), TODAY)
    assert get.await_count == 3
