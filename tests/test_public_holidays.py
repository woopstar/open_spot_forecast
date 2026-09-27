"""Public-holiday feature (#26)."""

import math
from collections.abc import Iterator
from datetime import date, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import holidays
import pytest

from custom_components.open_spot_forecast.const import HOLIDAY_SUBDIVISIONS, REGIONS
from custom_components.open_spot_forecast.ml import public_holidays
from custom_components.open_spot_forecast.ml.features import (
    FEATURE_NAMES,
    SlotInputs,
    build_feature_row,
    build_feature_vector,
)
from custom_components.open_spot_forecast.ml.public_holidays import public_holiday

TZ = ZoneInfo("Europe/Copenhagen")
# Danish public holidays in 2026 that are not Sundays
DK_2026 = {
    date(2026, 1, 1): "Nytårsdag",
    date(2026, 4, 2): "Skærtorsdag",
    date(2026, 4, 3): "Langfredag",
    date(2026, 4, 6): "Anden påskedag",
    date(2026, 5, 14): "Kristi himmelfartsdag",
    date(2026, 5, 25): "Anden pinsedag",
    date(2026, 12, 25): "Juledag",
    date(2026, 12, 26): "Anden juledag",
}


@pytest.fixture(autouse=True)
def _fresh_cache() -> Iterator[None]:
    """Each test builds its own calendars."""
    public_holidays.public_holiday.cache_clear()
    public_holidays._calendars.cache_clear()
    yield
    public_holidays.public_holiday.cache_clear()
    public_holidays._calendars.cache_clear()


def _year(start: date) -> list[date]:
    return [start + timedelta(days=n) for n in range(365)]


def test_holiday_is_a_model_input() -> None:
    assert "holiday" in FEATURE_NAMES


@pytest.mark.parametrize("region", ["DK1", "DK2"])
def test_danish_public_holidays_2026(region: str) -> None:
    """1 on public holidays and Sundays, 0 on ordinary days; Saturdays are days."""
    for day in _year(date(2026, 1, 1)):
        value = public_holiday(day, region)
        if day in DK_2026 or day.weekday() == 6:
            assert value == pytest.approx(1.0), day
        elif (day.month, day.day) in {(12, 24), (12, 31)}:
            assert value == pytest.approx(0.5), day
        else:
            assert value == pytest.approx(0.0), day


def test_easter_sunday_and_whit_sunday_are_holidays_as_sundays() -> None:
    assert public_holiday(date(2026, 4, 5), "DK1") == pytest.approx(1.0)
    assert public_holiday(date(2026, 5, 24), "DK1") == pytest.approx(1.0)
    assert public_holiday(date(2026, 4, 4), "DK1") == pytest.approx(0.0)  # Saturday


@pytest.mark.parametrize(
    ("day", "share"),
    [
        (date(2026, 10, 3), 1.0),  # German Unity Day, national
        (date(2026, 1, 6), 3 / 16),  # Epiphany: BW, BY, ST
        (date(2026, 6, 4), 6 / 16),  # Corpus Christi: BW, BY, HE, NW, RP, SL
        (date(2026, 10, 31), 9 / 16),  # Reformation Day
        (date(2026, 11, 18), 1 / 16),  # Repentance and Prayer: SN
        (date(2026, 12, 24), 0.5),  # not a public holiday, a half day
        (date(2026, 11, 17), 0.0),
    ],
)
def test_german_holidays_are_the_share_of_states(day: date, share: float) -> None:
    assert public_holiday(day, "DE") == pytest.approx(share)


def test_a_half_day_that_is_a_public_holiday_stays_one() -> None:
    """Where 24/12 is a public holiday, it is not lowered to 0.5."""
    assert public_holiday(date(2026, 12, 24), "EE") == pytest.approx(1.0)


def test_every_region_has_a_supported_holiday_calendar() -> None:
    for region, zone in REGIONS.items():
        country = str(zone["holidays"])
        assert public_holiday(date(2026, 1, 1), region) == pytest.approx(1.0), region
        subdivisions = holidays.country_holidays(country).subdivisions
        assert set(HOLIDAY_SUBDIVISIONS.get(country, ())) <= set(subdivisions)
    assert len(HOLIDAY_SUBDIVISIONS["DE"]) == 16


def test_unknown_region_has_an_unknown_holiday() -> None:
    assert public_holiday(date(2026, 12, 25), None) is None
    assert public_holiday(date(2026, 12, 25), "XX") is None
    row = build_feature_row(datetime(2026, 12, 25, 12, tzinfo=TZ), SlotInputs(), None)
    vector = dict(zip(FEATURE_NAMES, build_feature_vector(row), strict=True))
    assert math.isnan(vector["holiday"])


def test_feature_rows_use_the_local_date() -> None:
    """00:15 on Christmas Day is still 24 December in UTC."""
    christmas = build_feature_row(
        datetime(2026, 12, 25, 0, 15, tzinfo=TZ), SlotInputs(), "DK1"
    )
    eve = build_feature_row(
        datetime(2026, 12, 24, 23, 45, tzinfo=TZ), SlotInputs(), "DK1"
    )

    assert christmas["holiday"] == pytest.approx(1.0)
    assert eve["holiday"] == pytest.approx(0.5)


def test_calendars_are_built_once_per_country_and_year(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    built: list[tuple[str, Any, Any]] = []
    real = holidays.country_holidays

    def counting(country: str, subdiv: Any = None, years: Any = None) -> Any:
        built.append((country, subdiv, years))
        return real(country, subdiv=subdiv, years=years)

    monkeypatch.setattr(public_holidays.holidays, "country_holidays", counting)

    for day in _year(date(2026, 1, 1)) + _year(date(2026, 7, 1)):
        public_holiday(day, "DK1")
        public_holiday(day, "DK2")
        public_holiday(day, "DE")

    assert built.count(("DK", None, 2026)) == 1
    assert built.count(("DK", None, 2027)) == 1
    assert sum(1 for country, _, year in built if (country, year) == ("DE", 2026)) == 16
    assert len(built) == 2 + 2 * 16
