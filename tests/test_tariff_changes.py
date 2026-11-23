"""Dated national charges from Energi Data Service's price list (#278)."""

import json
from collections.abc import Iterator
from datetime import date, datetime
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch
from zoneinfo import ZoneInfo

import pytest

from custom_components.open_spot_forecast.api.http import HttpResponse
from custom_components.open_spot_forecast.api.tariff_changes import (
    ChargePeriod,
    NationalCharges,
    TariffChangeSource,
    parse_pricelist,
)
from custom_components.open_spot_forecast.const import NATIONAL_CHARGE_CODES

MODULE = "custom_components.open_spot_forecast.api.tariff_changes"
CPH = ZoneInfo("Europe/Copenhagen")
# Recorded 2026-11-23: Energinet's 40000, 41000 and EA-001 rows since
# 2024-12-01, plus a 40021 row (another code, skipped)
FIXTURE = Path(__file__).parent / "fixtures" / "datahub_pricelist.json"
TEXT = FIXTURE.read_text(encoding="utf-8")
TODAY = date(2026, 11, 23)


def _hourly(code: str, start: datetime, prices: list[float]) -> ChargePeriod:
    return ChargePeriod(code, start, None, tuple(prices))


def _flat(code: str, start: datetime, price: float) -> ChargePeriod:
    return ChargePeriod(code, start, None, (price,) + (None,) * 23)


@pytest.fixture
def get() -> Iterator[AsyncMock]:
    with (
        patch(f"{MODULE}.async_get", new=AsyncMock()) as mock,
        patch(f"{MODULE}.async_get_clientsession"),
    ):
        mock.return_value = HttpResponse(200, TEXT)
        yield mock


def test_the_recorded_price_list_parses_the_national_charges() -> None:
    periods = parse_pricelist(TEXT)

    assert len(periods) == 8
    assert {period.code for period in periods} == set(NATIONAL_CHARGE_CODES)
    tax = next(p for p in periods if p.code == "EA-001" and p.valid_from.year == 2026)
    # Danish local time, not UTC
    assert tax.valid_from == datetime(2026, 1, 1, tzinfo=CPH)
    assert tax.valid_to == datetime(2028, 1, 1, tzinfo=CPH)
    assert tax.prices[0] == pytest.approx(0.008)
    assert tax.prices[1] is None


def test_unusable_responses_give_no_periods() -> None:
    rows = [
        {"GLN_Number": "5790000432752", "ChargeType": "D03", "ChargeTypeCode": "40000"},
        {"GLN_Number": "1", "ChargeType": "D03", "ChargeTypeCode": "40000"},
        "not a row",
    ]
    for text in ("", "[]", "{}", '{"records": 1}', json.dumps({"records": rows})):
        assert parse_pricelist(text) == []


def test_the_charges_change_on_1_january_local_time() -> None:
    charges = NationalCharges(parse_pricelist(TEXT))

    # 2026: 0.043 + 0.072 + 0.008; 2027: 0.056 + 0.049 + 0.008
    assert charges.total_at(datetime(2026, 12, 31, 23, 45, tzinfo=CPH)) == (
        pytest.approx(0.123)
    )
    assert charges.total_at(datetime(2027, 1, 1, 0, 0, tzinfo=CPH)) == (
        pytest.approx(0.113)
    )
    assert charges.total_at(datetime(2024, 6, 1, tzinfo=CPH)) is None
    assert not NationalCharges()


def test_a_later_period_wins_over_an_overlapping_one() -> None:
    start = datetime(2026, 1, 1, tzinfo=CPH)
    periods = [_flat(code, start, 0.1) for code in NATIONAL_CHARGE_CODES]
    periods.append(_flat("EA-001", datetime(2027, 1, 1, tzinfo=CPH), 0.5))
    charges = NationalCharges(periods)

    assert charges.total_at(datetime(2026, 6, 1, tzinfo=CPH)) == pytest.approx(0.3)
    assert charges.total_at(datetime(2027, 6, 1, tzinfo=CPH)) == pytest.approx(0.7)


def test_hourly_prices_follow_the_local_hour_on_dst_days() -> None:
    """Price1 is 00-01 local; the repeated 02:00 hour takes Price3 twice."""
    start = datetime(2026, 1, 1, tzinfo=CPH)
    period = _hourly("40000", start, [float(hour) for hour in range(24)])
    first_two = datetime(2026, 10, 25, 0, 30, tzinfo=ZoneInfo("UTC"))
    second_two = datetime(2026, 10, 25, 1, 30, tzinfo=ZoneInfo("UTC"))

    assert period.price_at(first_two.astimezone(CPH)) == pytest.approx(2.0)
    assert period.price_at(second_two.astimezone(CPH)) == pytest.approx(2.0)
    # Spring forward: 01:59 CET is followed by 03:00 CEST
    after_gap = datetime(2026, 3, 29, 1, 0, tzinfo=ZoneInfo("UTC"))
    assert period.price_at(after_gap.astimezone(CPH)) == pytest.approx(3.0)


@pytest.mark.asyncio
async def test_the_source_fetches_once_a_day(get: AsyncMock) -> None:
    source = TariffChangeSource(Mock())

    assert await source.async_update(TODAY)
    assert not await source.async_update(TODAY)

    assert get.await_count == 1
    call = get.await_args
    assert call is not None
    params = call.kwargs["params"]
    assert params["start"] == "2023-11-21"
    assert json.loads(params["filter"]) == {
        "ChargeType": ["D03"],
        "GLN_Number": ["5790000432752"],
        "ChargeTypeCode": list(NATIONAL_CHARGE_CODES),
    }
    assert source.charges.total_at(datetime(2027, 1, 1, tzinfo=CPH)) == (
        pytest.approx(0.113)
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response", [None, HttpResponse(500, ""), HttpResponse(200, "{}")]
)
async def test_a_failed_fetch_keeps_the_charges_and_retries(
    get: AsyncMock, response: HttpResponse | None, caplog: pytest.LogCaptureFixture
) -> None:
    source = TariffChangeSource(Mock())
    await source.async_update(TODAY)
    charges = source.charges
    get.return_value = response

    assert not await source.async_update(date(2026, 11, 24))
    assert source.charges is charges
    assert not await source.async_update(date(2026, 11, 24))
    assert get.await_count == 3
    assert "energidataservice" not in caplog.text
