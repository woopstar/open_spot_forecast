"""Stromligning's per-slot tariffs in the forecast (#107).

The tariff of a slot is Stromligning's consumer price minus its spot price,
both excl. VAT. ``PriceOutput`` adds it to the predicted spot price, so the
forecast is on the same footing as the displayed consumer price; the model
never sees it.
"""

import re
from collections.abc import Sequence
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, Mock, patch
from zoneinfo import ZoneInfo

import pytest

from custom_components.open_spot_forecast.evaluation_sensor import (
    ForecastEvaluationSensor,
)
from custom_components.open_spot_forecast.price_output import (
    PriceOutput,
    with_tariffs,
)
from custom_components.open_spot_forecast.sensor import MLPredictionSensor
from custom_components.open_spot_forecast.services import forecast_response
from custom_components.open_spot_forecast.spot_prices import slot_prices
from custom_components.open_spot_forecast.tariffs import TariffSchedule
from custom_components.open_spot_forecast.time_slots import (
    slot_start_in_day,
    slots_in_local_day,
)
from custom_components.open_spot_forecast.updater import ForecastUpdater

CPH = ZoneInfo("Europe/Copenhagen")
DAY = date(2026, 9, 30)
TOMORROW = DAY + timedelta(days=1)
FALL_BACK = date(2026, 10, 25)
INTEGRATION = Path(__file__).parents[1] / "custom_components" / "open_spot_forecast"

pytestmark = pytest.mark.usefixtures("copenhagen_time_zone")


def _peak(day: date, index: int) -> float:
    """A time-of-use tariff: 0.40 from 17:00 to 21:00 local, 0.10 otherwise."""
    return 0.40 if 17 <= slot_start_in_day(day, index).hour < 21 else 0.10


def _data(
    today: Sequence[float | None], tomorrow: Sequence[float | None] = ()
) -> dict[str, Any]:
    """``read_spot_prices()``-shaped data for DAY."""
    return {"today": list(today), "tomorrow": list(tomorrow), "day": DAY}


def _day_prices(day: date, price: float = 0.3) -> list[float]:
    return [price] * slots_in_local_day(day)


def _consumer(day: date, spot: Sequence[float | None]) -> list[float | None]:
    return [
        None if price is None else price + _peak(day, index)
        for index, price in enumerate(spot)
    ]


def _schedule(
    today: bool = True, tomorrow: bool = False, day: date = DAY
) -> TariffSchedule:
    spot_today = _day_prices(day)
    spot_tomorrow = _day_prices(day + timedelta(days=1)) if tomorrow else []
    spot = {"today": spot_today, "tomorrow": spot_tomorrow, "day": day}
    consumer = {
        "today": _consumer(day, spot_today) if today else [],
        "tomorrow": _consumer(day + timedelta(days=1), spot_tomorrow),
        "day": day,
    }
    return TariffSchedule.from_prices(consumer, spot)


def _prediction(start: datetime, price: float = 0.5) -> dict[str, Any]:
    return {
        "start": start.isoformat(),
        "end": (start + timedelta(minutes=15)).isoformat(),
        "price": price,
        "confidence": 0.7,
    }


# --- The schedule -------------------------------------------------------------------


def test_the_tariff_is_consumer_minus_spot_per_slot() -> None:
    schedule = _schedule()

    assert len(schedule) == 96
    assert schedule.at(datetime(2026, 9, 30, 3, 0, tzinfo=CPH)) == pytest.approx(0.10)
    assert schedule.at(datetime(2026, 9, 30, 17, 0, tzinfo=CPH)) == pytest.approx(0.40)
    assert schedule.at(datetime(2026, 9, 30, 20, 45, tzinfo=CPH)) == pytest.approx(0.40)
    assert schedule.at(datetime(2026, 9, 30, 21, 0, tzinfo=CPH)) == pytest.approx(0.10)


def test_a_slot_missing_in_either_price_has_no_own_tariff() -> None:
    spot: list[float | None] = list(_day_prices(DAY))
    consumer = _consumer(DAY, spot)
    spot[4] = None
    consumer[8] = None

    schedule = TariffSchedule.from_prices(_data(consumer), _data(spot))

    assert len(schedule) == 94


def test_predicted_days_take_the_latest_days_time_of_day() -> None:
    """Tomorrow's (winter) tariffs win over today's for every later day."""
    spot_today, spot_tomorrow = _day_prices(DAY), _day_prices(TOMORROW)
    consumer_tomorrow = [price + 1.0 for price in spot_tomorrow]
    consumer_tomorrow = [
        price + _peak(TOMORROW, index) for index, price in enumerate(consumer_tomorrow)
    ]
    schedule = TariffSchedule.from_prices(
        _data(_consumer(DAY, spot_today), consumer_tomorrow),
        _data(spot_today, spot_tomorrow),
    )

    # Known slots keep their own tariff
    assert schedule.at(datetime(2026, 9, 30, 18, 0, tzinfo=CPH)) == pytest.approx(0.40)
    assert schedule.at(datetime(2026, 10, 1, 18, 0, tzinfo=CPH)) == pytest.approx(1.40)
    # Later days repeat the latest known day
    assert schedule.at(datetime(2026, 10, 4, 18, 0, tzinfo=CPH)) == pytest.approx(1.40)
    assert schedule.at(datetime(2026, 10, 4, 3, 15, tzinfo=CPH)) == pytest.approx(1.10)


def test_the_repeated_hour_of_a_dst_change_keeps_its_local_time() -> None:
    """The fall-back day's two 02:00 hours both take 02:00's tariff."""
    schedule = _schedule(day=DAY)
    first_two = slot_start_in_day(FALL_BACK, 8)
    second_two = slot_start_in_day(FALL_BACK, 12)

    assert first_two.hour == second_two.hour == 2
    # Same wall clock, an hour apart
    assert second_two.astimezone(UTC) - first_two.astimezone(UTC) == timedelta(hours=1)
    assert schedule.at(first_two) == pytest.approx(0.10)
    assert schedule.at(second_two) == pytest.approx(0.10)


def test_a_known_fall_back_day_keeps_both_repeated_hours() -> None:
    """Keyed by UTC instant, the two 02:00 hours of a known day stay distinct."""
    spot = _day_prices(FALL_BACK)
    consumer = [price + index / 1000 for index, price in enumerate(spot)]
    data = {"day": FALL_BACK}

    schedule = TariffSchedule.from_prices(
        {**data, "today": consumer}, {**data, "today": spot}
    )

    assert len(schedule) == 100
    assert schedule.at(slot_start_in_day(FALL_BACK, 8)) == pytest.approx(0.008)
    assert schedule.at(slot_start_in_day(FALL_BACK, 12)) == pytest.approx(0.012)


def test_a_time_of_day_never_seen_takes_the_latest_earlier_one() -> None:
    """E.g. a known 92-slot spring day has no 02:00-02:45: 01:45 stands in."""
    spot = _day_prices(DAY)[:4]
    schedule = TariffSchedule.from_prices(
        _data([price + 0.2 for price in spot]), _data(spot)
    )

    assert schedule.at(datetime(2026, 10, 2, 5, 30, tzinfo=CPH)) == pytest.approx(0.2)


def test_without_consumer_prices_the_schedule_is_empty() -> None:
    for consumer in (None, {}, _data([])):
        schedule = TariffSchedule.from_prices(consumer, _data(_day_prices(DAY)))

        assert not schedule
        assert schedule.at(datetime(2026, 9, 30, 18, 0, tzinfo=CPH)) == 0.0


def test_slot_prices_skip_missing_slots_and_step_into_tomorrow() -> None:
    slots = list(slot_prices(_data([1.0, None, 2.0], [3.0])))

    assert [price for _, _, price in slots] == [1.0, 2.0, 3.0]
    assert slots[0][0] == datetime(2026, 9, 30, 0, 0, tzinfo=CPH)
    assert slots[1][0] == datetime(2026, 9, 30, 0, 30, tzinfo=CPH)
    assert slots[2][0] == datetime(2026, 10, 1, 0, 0, tzinfo=CPH)
    assert slots[2][1] == datetime(2026, 10, 1, 0, 15, tzinfo=CPH)


# --- The output ---------------------------------------------------------------------


def test_the_forecast_adds_the_slots_tariff_before_vat() -> None:
    output = PriceOutput(vat=0.25, surcharge=0.0, precision=4)
    evening = datetime(2026, 10, 2, 18, 0, tzinfo=CPH)
    night = datetime(2026, 10, 2, 3, 0, tzinfo=CPH)

    forecast = output.forecast([_prediction(evening), _prediction(night)], _schedule())

    assert forecast[0]["price"] == pytest.approx((0.5 + 0.40) * 1.25)
    assert forecast[1]["price"] == pytest.approx((0.5 + 0.10) * 1.25)
    assert forecast[0]["confidence"] == pytest.approx(0.7)


def test_without_tariffs_the_forecast_is_the_spot_price() -> None:
    output = PriceOutput(vat=0.25, precision=4)
    predictions = [_prediction(datetime(2026, 10, 2, 18, 0, tzinfo=CPH))]

    for tariffs in (None, TariffSchedule({})):
        assert output.forecast(predictions, tariffs)[0]["price"] == pytest.approx(0.625)


def test_the_hourly_mean_includes_each_slots_tariff() -> None:
    """16:00-17:00 has no peak, 17:00 does: the hour 16:30-17:30 does not exist,
    but the hour 17:00 averages four peak slots."""
    output = PriceOutput(vat=0.0, precision=4, hourly_average=True)
    start = datetime(2026, 10, 2, 16, 30, tzinfo=CPH)
    predictions = [
        _prediction(start + timedelta(minutes=15 * index)) for index in range(6)
    ]

    hours = output.forecast(predictions, _schedule())

    assert [hour["price"] for hour in hours] == pytest.approx([0.6, 0.9])


def test_with_tariffs_leaves_the_predictions_unchanged() -> None:
    """The model's own predictions (and so the learning) never see a tariff."""
    predictions = [_prediction(datetime(2026, 10, 2, 18, 0, tzinfo=CPH))]

    with_tariffs(predictions, _schedule())

    assert predictions[0]["price"] == pytest.approx(0.5)


def test_the_evaluation_adds_the_tariff_to_both_prices() -> None:
    output = PriceOutput(vat=0.25, precision=4)
    start = datetime(2026, 9, 30, 18, 0, tzinfo=CPH)
    rows = [{"start": start.isoformat(), "predicted": 0.5, "actual": 0.3}]

    (row,) = output.evaluation(rows, _schedule())

    assert row["predicted"] == pytest.approx((0.5 + 0.40) * 1.25)
    assert row["actual"] == pytest.approx((0.3 + 0.40) * 1.25)
    # The error stays that of the spot price (times VAT)
    assert row["predicted"] - row["actual"] == pytest.approx(0.2 * 1.25)


def test_the_action_includes_the_tariffs() -> None:
    start = datetime(2026, 10, 2, 18, 0, tzinfo=CPH)

    response = forecast_response(
        [_prediction(start)],
        PriceOutput(vat=0.25, precision=4),
        "DKK",
        None,
        start,
        tariffs=_schedule(),
    )

    assert response["forecast"][0]["price"] == pytest.approx((0.5 + 0.40) * 1.25)


# --- The entities -------------------------------------------------------------------


def _entry() -> MagicMock:
    entry = MagicMock()
    entry.entry_id = "test"
    entry.options = {}
    return entry


def test_the_forecast_sensor_includes_the_tariffs() -> None:
    start = datetime(2026, 10, 2, 18, 0, tzinfo=CPH)
    predictor = MagicMock()
    predictor.predictions = [_prediction(start)]
    predictor.get_prediction_stats.return_value = {"mean_confidence": 0.7}
    api_data = {"ml_predictor": predictor, "tariffs": _schedule()}
    sensor = MLPredictionSensor(
        MagicMock(), _entry(), api_data, "DKK", PriceOutput(vat=0.25, precision=4)
    )

    with patch("homeassistant.util.dt.utcnow", return_value=start - timedelta(hours=1)):
        assert sensor.native_value == pytest.approx((0.5 + 0.40) * 1.25)
        attrs = sensor.extra_state_attributes

    assert attrs["includes_tariffs"] is True
    assert attrs["forecast_max"] == pytest.approx((0.5 + 0.40) * 1.25)
    assert attrs["predictions"][0]["price"] == pytest.approx((0.5 + 0.40) * 1.25)

    api_data["tariffs"] = TariffSchedule({})
    assert sensor.extra_state_attributes["includes_tariffs"] is False


def test_the_evaluation_sensor_includes_the_tariffs() -> None:
    start = datetime(2026, 9, 30, 18, 0, tzinfo=CPH)
    predictor = MagicMock()
    predictor.evaluation = [
        {"start": start.isoformat(), "predicted": 0.5, "actual": 0.3}
    ]
    api_data = {"ml_predictor": predictor, "tariffs": _schedule()}
    sensor = ForecastEvaluationSensor(
        MagicMock(), _entry(), api_data, "DKK", PriceOutput(vat=0.0, precision=4)
    )

    with patch("homeassistant.util.dt.utcnow", return_value=start):
        attrs = sensor.extra_state_attributes

    assert attrs["t"] == pytest.approx([0.9])
    assert attrs["a"] == pytest.approx([0.7])


def test_the_updater_builds_the_schedule_from_both_prices() -> None:
    """Read with the spot prices, from the consumer prices already read."""
    spot = _data(_day_prices(DAY))
    updater = ForecastUpdater.__new__(ForecastUpdater)
    updater.ml_predictor = Mock()
    updater.dayahead = None
    updater.sensors = Mock()
    updater.sensor_reader = Mock()
    updater.sensor_reader.read_spot_prices.return_value = spot
    updater.api_data = {
        "stromligning_data": _data(_consumer(DAY, spot["today"])),
    }

    updater.read_spot_prices()

    tariffs = updater.api_data["tariffs"]
    assert len(tariffs) == 96
    assert tariffs.at(datetime(2026, 9, 30, 18, 0, tzinfo=CPH)) == pytest.approx(0.40)


def test_the_model_never_sees_a_tariff() -> None:
    """Tariffs are output-only (#16): nothing in ml/ reads the schedule."""
    for path in (INTEGRATION / "ml").glob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert not re.search(r"TariffSchedule|\btariffs\.py|\"tariffs\"", text), (
            path.name
        )
