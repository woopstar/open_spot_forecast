"""Tests for the backtest's lagged price columns (#119): origin-relative, leak-free."""

import math
from collections import Counter
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from custom_components.open_spot_forecast.ml.features import floor_epoch
from custom_components.open_spot_forecast.time_slots import (
    slot_start_in_day,
    slots_in_local_day,
)
from scripts.backtest_lags import (
    LAG_NAMES,
    LagConfig,
    PriceLagIndex,
    training_lead,
)

TZ = ZoneInfo("Europe/Copenhagen")
FIRST = date(2026, 6, 1)  # a Monday
SLOT = 37  # 09:15 local on a normal day


def _price(day: int, slot: int) -> float:
    """A price unique to (day offset from FIRST, slot index)."""
    return day * 100.0 + slot


def _prices(first: date = FIRST, days: int = 10) -> dict[int, float]:
    """Known prices by slot epoch: ``days`` local days, every slot known."""
    prices: dict[int, float] = {}
    for day in range(days):
        local_day = first + timedelta(days=day)
        for slot in range(slots_in_local_day(local_day, tz=TZ)):
            start = slot_start_in_day(local_day, slot, TZ)
            prices[floor_epoch(start, TZ)] = _price(day, slot)
    return prices


def _index(days: int = 10, **kwargs: bool) -> PriceLagIndex:
    return PriceLagIndex(_prices(days=days), TZ, **kwargs)


def _mean(days: range) -> float:
    """Mean of ``_price`` over whole (normal) days."""
    return float(np.mean([_price(day, slot) for day in days for slot in range(96)]))


def _slot(day: date, hour: int = 9, minute: int = 15) -> datetime:
    return datetime(day.year, day.month, day.day, hour, minute, tzinfo=TZ)


def test_training_rows_use_the_day_before_the_slot() -> None:
    """Inside the history a slot's reference day is the day before its own."""
    index = _index()

    lags = index.for_slot(_slot(FIRST + timedelta(days=8)))

    assert index.last_known_day == FIRST + timedelta(days=9)
    assert lags["price_same_slot_last_known_day"] == pytest.approx(_price(7, SLOT))
    assert lags["price_same_slot_last_week"] == pytest.approx(_price(1, SLOT))
    assert lags["price_mean_last_7_known_days"] == pytest.approx(_mean(range(1, 8)))
    assert lags["price_lag_days"] == pytest.approx(1.0)


@pytest.mark.parametrize("lead", range(1, 8))
def test_target_rows_share_the_last_known_day_at_every_horizon(lead: int) -> None:
    """A slot after the history gets the last known day, whatever its lead time."""
    index = _index()  # known until day 9
    start = _slot(FIRST + timedelta(days=9 + lead))

    lags = index.for_slot(start)

    assert index.reference_day(start) == FIRST + timedelta(days=9)
    assert lags["price_same_slot_last_known_day"] == pytest.approx(_price(9, SLOT))
    assert lags["price_mean_last_7_known_days"] == pytest.approx(_mean(range(3, 10)))
    # One week before day 9 + lead is day 2 + lead, known for leads up to 7
    assert lags["price_same_slot_last_week"] == pytest.approx(_price(2 + lead, SLOT))
    assert lags["price_lag_days"] == pytest.approx(float(lead))


def test_last_week_is_unknown_beyond_seven_days_ahead() -> None:
    """Eight days after the last known day, the slot a week before is unknown too."""
    lags = _index().for_slot(_slot(FIRST + timedelta(days=17)))

    assert lags["price_same_slot_last_week"] is None
    assert lags["price_same_slot_last_known_day"] == pytest.approx(_price(9, SLOT))


def test_prices_from_the_slots_day_or_later_never_reach_the_features() -> None:
    """Poisoning every price from a slot's day on leaves the slot's lags unchanged."""
    clean = _prices(days=12)
    cutoff = floor_epoch(_slot(FIRST + timedelta(days=8), 0, 0), TZ)
    poisoned = {epoch: (1e6 if epoch >= cutoff else p) for epoch, p in clean.items()}
    start = _slot(FIRST + timedelta(days=8))

    expected = PriceLagIndex(clean, TZ).for_slot(start)
    actual = PriceLagIndex(poisoned, TZ).for_slot(start)

    assert actual == expected
    assert all(value is not None and value < 1e5 for value in actual.values())


def test_mean_needs_every_one_of_the_seven_days() -> None:
    """A shorter history gives no 7-day mean rather than a shorter one."""
    index = _index(days=6)

    lags = index.for_slot(_slot(FIRST + timedelta(days=6)))
    week_later = index.for_slot(_slot(FIRST + timedelta(days=7)))

    assert lags["price_same_slot_last_known_day"] == pytest.approx(_price(5, SLOT))
    assert lags["price_mean_last_7_known_days"] is None
    assert lags["price_same_slot_last_week"] is None
    assert week_later["price_mean_last_7_known_days"] is None


def test_mean_skips_missing_slots_but_not_missing_days() -> None:
    """A NaN slot is left out of the mean; a day without prices voids it."""
    prices = _prices(days=8)
    missing = floor_epoch(_slot(FIRST + timedelta(days=3), 2, 30), TZ)
    prices[missing] = math.nan
    index = PriceLagIndex(prices, TZ)
    expected = np.mean(
        [_price(d, s) for d in range(7) for s in range(96) if (d, s) != (3, 10)]
    )

    lags = index.for_slot(_slot(FIRST + timedelta(days=7)))

    assert lags["price_mean_last_7_known_days"] == pytest.approx(expected)
    assert index.same_slot(_slot(FIRST, 2, 30), FIRST + timedelta(days=3)) is None


def test_same_slot_follows_the_local_wall_clock_across_dst() -> None:
    """08:00 the day after the spring-forward day looks up 08:00 of that day."""
    index = PriceLagIndex(_prices(date(2026, 3, 22), 9), TZ)
    start = _slot(date(2026, 3, 30), 8, 0)  # the day after 2026-03-29 (92 slots)

    lags = index.for_slot(start)

    # 00:00-02:00 CET are slots 0-7; 03:00 CEST is slot 8, so 08:00 is slot 28
    assert lags["price_same_slot_last_known_day"] == pytest.approx(_price(7, 28))
    assert lags["price_same_slot_last_week"] == pytest.approx(_price(1, 32))


def test_without_prices_every_lag_is_unknown() -> None:
    """An empty index has no reference day and only None features."""
    index = PriceLagIndex({}, TZ)

    assert index.last_known_day is None
    assert index.reference_day(_slot(FIRST)) is None
    assert index.for_slot(_slot(FIRST)) == dict.fromkeys(LAG_NAMES)


def test_non_finite_prices_are_not_prices() -> None:
    """NaN and inf in the index are unknown, like None; numbers are parsed."""
    index = PriceLagIndex({0: math.nan, 900: math.inf, 1800: "3.5"}, TZ)
    epoch = date(1970, 1, 1)

    assert index.same_slot(datetime.fromtimestamp(1800, TZ), epoch) == pytest.approx(
        3.5
    )
    assert index.same_slot(datetime.fromtimestamp(0, TZ), epoch) is None
    assert index.last_known_day == epoch


def test_training_lead_spreads_the_ages_independently_of_the_weekday() -> None:
    """Mixed-age training rows show every lag age 1-7, not one per weekday."""
    days = [FIRST + timedelta(days=offset) for offset in range(60)]
    leads = [training_lead(day) for day in days]

    assert set(leads) == set(range(1, 8))
    assert min(Counter(leads).values()) >= 5
    for weekday in range(7):
        on_weekday = {
            lead
            for day, lead in zip(days, leads, strict=True)
            if day.weekday() == weekday
        }
        assert len(on_weekday) > 1


def test_mixed_ages_lag_training_rows_by_their_lead_and_targets_by_the_horizon() -> (
    None
):
    """With mixed ages a training row's lag is training_lead days old."""
    index = _index(mixed_ages=True)
    day = FIRST + timedelta(days=8)
    lead = training_lead(day)

    training = index.for_slot(_slot(day))
    target = index.for_slot(_slot(FIRST + timedelta(days=12)))

    assert training["price_lag_days"] == pytest.approx(float(lead))
    assert training["price_same_slot_last_known_day"] == pytest.approx(
        _price(8 - lead, SLOT)
    )
    assert target["price_lag_days"] == pytest.approx(3.0)
    assert target["price_same_slot_last_known_day"] == pytest.approx(_price(9, SLOT))


def test_columns_follow_the_requested_names_with_nan_for_unknown() -> None:
    """The matrix has one column per name, NaN where the lag is unknown."""
    index = _index(days=3)
    starts = np.array(
        [
            floor_epoch(_slot(FIRST), TZ),  # first day: nothing before it
            floor_epoch(_slot(FIRST + timedelta(days=2)), TZ),
        ]
    )

    columns = index.columns(
        starts, ("price_same_slot_last_known_day", "price_lag_days")
    )

    assert columns.shape == (2, 2)
    assert math.isnan(columns[0, 0])
    assert columns[0, 1] == pytest.approx(1.0)
    assert columns[1, 0] == pytest.approx(_price(1, SLOT))


def test_lag_config_parses_all_or_a_subset_and_rejects_unknown_names() -> None:
    """``--lags all`` is every column; names are validated."""
    assert LagConfig.parse("all", False) == LagConfig(LAG_NAMES, False)
    assert LagConfig.parse(
        " price_lag_days, price_same_slot_last_week ", True
    ) == LagConfig(("price_lag_days", "price_same_slot_last_week"), True)
    with pytest.raises(ValueError, match="unknown lag feature"):
        LagConfig.parse("price_yesterday", False)
