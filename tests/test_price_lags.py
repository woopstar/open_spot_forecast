"""Tests for the near-term model's lagged price (ml/price_lags.py, #135)."""

import math
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from custom_components.open_spot_forecast.ml.features import FEATURE_NAMES
from custom_components.open_spot_forecast.ml.price_lags import (
    NEAR_TERM_DAYS,
    PriceLagIndex,
    near_term_inputs,
    slot_start,
)
from custom_components.open_spot_forecast.time_slots import slots_in_local_day

TZ = ZoneInfo("Europe/Copenhagen")
FIRST = date(2026, 6, 1)  # a Monday
DAYS = 10
LAST = FIRST + timedelta(days=DAYS - 1)
SLOT = 37  # 09:15 local on a normal day


def _price(day: int, slot: int) -> float:
    """A price unique to (day offset from FIRST, slot index)."""
    return day * 100.0 + slot


def _history(first: date = FIRST, days: int = DAYS) -> list[dict]:
    """The model's price history: one entry per local day, every slot known."""
    return [
        {
            "date": (first + timedelta(days=day)).isoformat(),
            "prices": [
                _price(day, slot)
                for slot in range(
                    slots_in_local_day(first + timedelta(days=day), tz=TZ)
                )
            ],
        }
        for day in range(days)
    ]


def _index() -> PriceLagIndex:
    return PriceLagIndex.from_days(_history(), TZ)


def _slot(day: date, hour: int = 9, minute: int = 15) -> datetime:
    return datetime(day.year, day.month, day.day, hour, minute, tzinfo=TZ)


def test_a_training_row_lags_to_the_day_before() -> None:
    index = _index()
    start = _slot(FIRST + timedelta(days=4))

    assert index.last_known_day == LAST
    assert index.reference_day(start) == FIRST + timedelta(days=3)
    assert index.lag_days(start) == 1
    assert index.column([start]) == pytest.approx([_price(3, SLOT)])


@pytest.mark.parametrize("lead", range(1, 8))
def test_prediction_rows_share_the_last_known_day_at_every_lead(lead: int) -> None:
    """A day-1 and a day-7 row both get the last known day's price in their slot."""
    index = _index()
    start = _slot(LAST + timedelta(days=lead))

    assert index.reference_day(start) == LAST
    assert index.lag_days(start) == lead
    assert index.column([start]) == pytest.approx([_price(DAYS - 1, SLOT)])


def test_only_the_first_forecast_days_are_near_term() -> None:
    index = _index()
    starts = [_slot(LAST + timedelta(days=lead)) for lead in range(1, 8)]

    served = index.near_term(starts, NEAR_TERM_DAYS)

    assert served.tolist() == [True] * NEAR_TERM_DAYS + [False] * (7 - NEAR_TERM_DAYS)
    # A slot inside the history lags one day, whichever day it is
    assert index.near_term([_slot(FIRST), _slot(LAST)], NEAR_TERM_DAYS).all()


def test_a_slot_never_sees_a_price_of_its_own_day_or_later() -> None:
    """Changing every price from a slot's day on leaves its lag unchanged."""
    history = _history()
    day = 6
    start = _slot(FIRST + timedelta(days=day))
    poisoned = [
        {**entry, "prices": [1e6] * len(entry["prices"])} if offset >= day else entry
        for offset, entry in enumerate(history)
    ]

    clean = PriceLagIndex.from_days(history, TZ).column([start])
    changed = PriceLagIndex.from_days(poisoned, TZ).column([start])

    assert clean == pytest.approx([_price(day - 1, SLOT)])
    assert changed == pytest.approx(clean)


def test_an_entry_reaching_into_the_next_day_keys_every_slot_by_its_own_start() -> None:
    """Today's entry holds tomorrow's prices once they are published."""
    history = _history()
    tomorrow = [_price(DAYS, slot) for slot in range(96)]
    history[-1] = {**history[-1], "prices": history[-1]["prices"] + tomorrow}
    index = PriceLagIndex.from_days(history, TZ)
    next_day = LAST + timedelta(days=1)

    assert index.last_known_day == next_day
    assert index.column([_slot(next_day + timedelta(days=1))]) == pytest.approx(
        [_price(DAYS, SLOT)]
    )
    assert index.column([_slot(next_day)]) == pytest.approx([_price(DAYS - 1, SLOT)])


def test_an_unknown_lag_is_nan() -> None:
    history = _history()
    history[3]["prices"][SLOT] = None
    history.append({"date": "not a date", "prices": [1.0]})
    index = PriceLagIndex.from_days(history, TZ)

    column = index.column(
        [
            _slot(FIRST),  # nothing before the first day
            _slot(FIRST + timedelta(days=4)),  # the slot is missing the day before
            None,  # a row without a start time
            _slot(FIRST + timedelta(days=5)),
        ]
    )

    assert np.isnan(column[:3]).all()
    assert column[3] == pytest.approx(_price(4, SLOT))
    assert index.near_term([None], NEAR_TERM_DAYS).tolist() == [False]


def test_without_prices_nothing_is_known_and_nothing_is_near_term() -> None:
    index = PriceLagIndex({0: math.nan, 900: None}, TZ)
    start = _slot(FIRST)

    assert index.last_known_day is None
    assert index.reference_day(start) is None
    assert index.lag_days(start) is None
    assert np.isnan(index.column([start])).all()
    assert not index.near_term([start], NEAR_TERM_DAYS).any()


def test_the_lag_is_the_same_wall_clock_slot_across_a_dst_change() -> None:
    """09:15 on the 92-slot day lags to 09:15 the day before, not to slot 37."""
    first = date(2026, 3, 22)
    index = PriceLagIndex.from_days(_history(first, 9), TZ)
    spring_forward = date(2026, 3, 29)

    after = index.column([_slot(spring_forward + timedelta(days=1))])
    on_the_day = index.column([_slot(spring_forward)])

    # 09:15 is slot 33 of the 92-slot day (02:00-03:00 does not exist)
    assert after == pytest.approx([_price(7, 33)])
    assert on_the_day == pytest.approx([_price(6, SLOT)])


def test_the_lag_column_follows_the_canonical_features() -> None:
    """Cross-border columns stay behind it, as in the backtest's rows."""
    rows = np.arange(2 * (len(FEATURE_NAMES) + 3), dtype=float).reshape(2, -1)
    lag = np.array([-1.0, -2.0])

    near = near_term_inputs(rows, lag)

    assert near.shape == (2, len(FEATURE_NAMES) + 4)
    np.testing.assert_array_equal(near[:, len(FEATURE_NAMES)], lag)
    np.testing.assert_array_equal(
        near[:, : len(FEATURE_NAMES)], rows[:, : len(FEATURE_NAMES)]
    )
    np.testing.assert_array_equal(
        near[:, len(FEATURE_NAMES) + 1 :], rows[:, len(FEATURE_NAMES) :]
    )


def test_slot_start_reads_a_feature_rows_start() -> None:
    start = _slot(FIRST)

    assert slot_start({"start": start.isoformat()}) == start
    assert slot_start({}) is None
    assert slot_start({"start": "soon"}) is None
