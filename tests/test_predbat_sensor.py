"""Predbat-compatible rate entities (#124).

Four optional sensors in the shape of Stromligning's: ``prices_today`` /
``prices_tomorrow`` of ``{"start", "end", "price"}``, the import price with
tariffs, surcharge and VAT, the export price the raw spot price, a unit
Predbat scales to øre, and the tomorrow entity carrying the following days
within the attribute budget.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import Any
from unittest.mock import MagicMock, Mock, patch
from zoneinfo import ZoneInfo

import pytest

from custom_components.open_spot_forecast.const import (
    CONF_PREDBAT_SENSORS,
    CONF_PREDBAT_TARIFF,
    DOMAIN,
    UPDATE_SIGNAL,
    UPDATE_SIGNAL_FORECAST,
)
from custom_components.open_spot_forecast.forecast_attributes import (
    ATTRIBUTE_BUDGET_BYTES,
    attributes_size,
    fit_entries,
)
from custom_components.open_spot_forecast.predbat_sensor import (
    PredbatRateSensor,
    build_predbat_sensors,
    predbat_unit,
)
from custom_components.open_spot_forecast.price_output import PriceOutput
from custom_components.open_spot_forecast.sensor import async_setup_entry
from custom_components.open_spot_forecast.tariffs import TariffSchedule
from custom_components.open_spot_forecast.time_slots import (
    slot_start_in_day,
    slots_in_local_day,
)

CPH = ZoneInfo("Europe/Copenhagen")
DAY = date(2026, 9, 24)
NOW = datetime(2026, 9, 24, 10, 7, tzinfo=CPH)
VAT = 0.25
SURCHARGE = 0.1
TARIFF = 0.5
SPOT_TODAY = 1.0
SPOT_TOMORROW = 2.0
PREDICTED = 3.0


def _spot_data(tomorrow: bool = True) -> dict[str, Any]:
    """Confirmed spot prices: 1.0 all of today, 2.0 all of tomorrow."""
    data: dict[str, Any] = {"day": DAY, "today": [], "tomorrow": []}
    for key, day, price in (
        ("today", DAY, SPOT_TODAY),
        ("tomorrow", DAY + timedelta(days=1), SPOT_TOMORROW),
    ):
        if key == "tomorrow" and not tomorrow:
            continue
        count = slots_in_local_day(day)
        data[key] = [price] * count
        data[f"raw_{key}"] = [
            {"start": slot_start_in_day(day, i).isoformat()} for i in range(count)
        ]
    return data


def _predictions(first_day: date, days: int, first_slot: int = 0) -> list[dict]:
    """Model predictions (raw spot 3.0) for ``days`` days from ``first_day``."""
    predictions = []
    for offset in range(days):
        day = first_day + timedelta(days=offset)
        for index in range(first_slot if offset == 0 else 0, slots_in_local_day(day)):
            predictions.append(
                {
                    "start": slot_start_in_day(day, index).isoformat(),
                    "end": slot_start_in_day(day, index + 1).isoformat(),
                    "price": PREDICTED,
                    "confidence": 0.8,
                }
            )
    return predictions


def _tariffs(spot_data: dict[str, Any]) -> TariffSchedule:
    consumer = {
        **spot_data,
        "today": [p + TARIFF for p in spot_data["today"]],
        "tomorrow": [p + TARIFF for p in spot_data["tomorrow"]],
    }
    return TariffSchedule.from_prices(consumer, spot_data)


def _api_data(
    predictions: list[dict] | None,
    spot_data: dict[str, Any] | None,
    tariffs: TariffSchedule | None = None,
) -> dict[str, Any]:
    api_data: dict[str, Any] = {"spot_data": spot_data, "tariffs": tariffs}
    if predictions is not None:
        predictor = MagicMock()
        predictor.predictions = predictions
        api_data["ml_predictor"] = predictor
    return api_data


def _sensor(
    api_data: dict[str, Any],
    kind: str,
    day: str,
    currency: str = "DKK",
    output: PriceOutput | None = None,
    fixed_tariff: float = 0.0,
) -> PredbatRateSensor:
    return PredbatRateSensor(
        MagicMock(),
        MagicMock(entry_id="test"),
        api_data,
        currency,
        output or PriceOutput(vat=VAT, surcharge=SURCHARGE, precision=4),
        kind,
        day,
        fixed_tariff,
    )


def _attrs(sensor: PredbatRateSensor, now: datetime = NOW) -> dict[str, Any]:
    with patch("homeassistant.util.dt.utcnow", return_value=now.astimezone(UTC)):
        return sensor.extra_state_attributes


def _state(sensor: PredbatRateSensor, now: datetime = NOW) -> float | None:
    with patch("homeassistant.util.dt.utcnow", return_value=now.astimezone(UTC)):
        return sensor.native_value


def _import_price(spot: float) -> float:
    return (spot + TARIFF + SURCHARGE) * (1 + VAT)


@pytest.fixture
def two_days() -> dict[str, Any]:
    """Confirmed today and tomorrow, forecast from the current slot for 7 days."""
    spot_data = _spot_data()
    return _api_data(
        _predictions(DAY, 7, first_slot=40), spot_data, _tariffs(spot_data)
    )


@pytest.mark.parametrize(
    ("currency", "unit", "factor"),
    [("DKK", "kr/kWh", 1.0), ("SEK", "kr/kWh", 1.0), ("NOK", "kr/kWh", 1.0)],
)
def test_krone_currencies_use_a_unit_predbat_scales_to_ore(
    currency: str, unit: str, factor: float
) -> None:
    """Predbat multiplies by 100 only if the unit contains ``kr/``."""
    assert predbat_unit(currency) == (unit, pytest.approx(factor))
    assert "kr/" in unit.lower()
    sensor = _sensor({}, "import", "today", currency=currency)
    assert sensor.native_unit_of_measurement == unit


@pytest.mark.usefixtures("copenhagen_time_zone")
def test_euro_prices_are_exposed_in_cents_with_a_unit_predbat_keeps() -> None:
    spot_data = _spot_data()
    sensor = _sensor(_api_data([], spot_data), "export", "today", currency="EUR")

    unit, factor = predbat_unit("EUR")
    assert "kr/" not in unit.lower()
    assert factor == pytest.approx(100.0)
    assert sensor.native_unit_of_measurement == unit
    assert _attrs(sensor)["prices_today"][0]["price"] == pytest.approx(100.0)


@pytest.mark.usefixtures("copenhagen_time_zone")
def test_entries_have_predbat_shape_in_local_time(two_days: dict[str, Any]) -> None:
    sensor = _sensor(two_days, "import", "today")

    entries = _attrs(sensor)["prices_today"]
    assert len(entries) == 96
    assert all(set(entry) == {"start", "end", "price"} for entry in entries)
    assert entries[0]["start"] == "2026-09-24T00:00:00+02:00"
    assert entries[0]["end"] == "2026-09-24T00:15:00+02:00"
    assert entries[-1]["end"] == "2026-09-25T00:00:00+02:00"
    assert all(isinstance(entry["price"], float) for entry in entries)


@pytest.mark.usefixtures("copenhagen_time_zone")
def test_today_and_tomorrow_split_at_local_midnight(two_days: dict[str, Any]) -> None:
    """Today holds today only; tomorrow holds tomorrow and the following days."""
    today = _attrs(_sensor(two_days, "export", "today"))["prices_today"]
    tomorrow = _attrs(_sensor(two_days, "export", "tomorrow"))["prices_tomorrow"]

    assert today[0]["start"] == "2026-09-24T00:00:00+02:00"
    assert today[-1]["start"] == "2026-09-24T23:45:00+02:00"
    assert tomorrow[0]["start"] == "2026-09-25T00:00:00+02:00"
    # Beyond tomorrow: the day after is in the tomorrow entity
    assert any(entry["start"].startswith("2026-09-26") for entry in tomorrow)
    assert not any(entry["start"].startswith("2026-09-24") for entry in tomorrow)


@pytest.mark.usefixtures("copenhagen_time_zone")
def test_confirmed_prices_come_first_then_the_forecast(
    two_days: dict[str, Any],
) -> None:
    """Confirmed slots win over predictions for the same slot; no gap, no overlap."""
    today = _attrs(_sensor(two_days, "export", "today"))["prices_today"]
    tomorrow = _attrs(_sensor(two_days, "export", "tomorrow"))["prices_tomorrow"]

    # The predictions start at 10:00 today but today is confirmed: 1.0 all day
    assert [entry["price"] for entry in today] == [pytest.approx(SPOT_TODAY)] * 96
    # Tomorrow is confirmed (2.0), the day after predicted (3.0)
    prices = [entry["price"] for entry in tomorrow]
    assert prices[:96] == [pytest.approx(SPOT_TOMORROW)] * 96
    assert prices[96] == pytest.approx(PREDICTED)
    starts = [entry["start"] for entry in tomorrow]
    assert starts == sorted(starts)
    assert len(starts) == len(set(starts))


@pytest.mark.usefixtures("copenhagen_time_zone")
def test_forecast_fills_today_after_the_confirmed_prices_end() -> None:
    """Without tomorrow's prices the forecast continues today's series."""
    spot_data = _spot_data(tomorrow=False)
    # Today's prices stop at 10:00 (slot 40); the forecast starts there
    spot_data["today"] = spot_data["today"][:40]
    spot_data["raw_today"] = spot_data["raw_today"][:40]
    api_data = _api_data(_predictions(DAY, 3, first_slot=40), spot_data)

    today = _attrs(_sensor(api_data, "export", "today"))["prices_today"]
    tomorrow = _attrs(_sensor(api_data, "export", "tomorrow"))["prices_tomorrow"]

    assert len(today) == 96
    assert [e["price"] for e in today[:40]] == [pytest.approx(SPOT_TODAY)] * 40
    assert [e["price"] for e in today[40:]] == [pytest.approx(PREDICTED)] * 56
    assert today[39]["end"] == today[40]["start"]
    # Two predicted days, trimmed to the attribute budget by whole hours
    assert 96 < len(tomorrow) <= 192
    assert len(tomorrow) % 4 == 0
    assert all(e["price"] == pytest.approx(PREDICTED) for e in tomorrow)


@pytest.mark.usefixtures("copenhagen_time_zone")
def test_import_is_the_consumer_price_and_export_the_raw_spot_price(
    two_days: dict[str, Any],
) -> None:
    import_today = _attrs(_sensor(two_days, "import", "today"))["prices_today"]
    import_tomorrow = _attrs(_sensor(two_days, "import", "tomorrow"))["prices_tomorrow"]
    export_tomorrow = _attrs(_sensor(two_days, "export", "tomorrow"))["prices_tomorrow"]

    # Import: (spot + tariff + surcharge) × (1 + VAT), confirmed and predicted
    assert import_today[0]["price"] == pytest.approx(_import_price(SPOT_TODAY))
    assert import_tomorrow[0]["price"] == pytest.approx(_import_price(SPOT_TOMORROW))
    # The day after tomorrow repeats tomorrow's tariff at the same time of day
    assert import_tomorrow[96]["price"] == pytest.approx(_import_price(PREDICTED))
    # Export: the raw spot price, nothing added
    assert export_tomorrow[0]["price"] == pytest.approx(SPOT_TOMORROW)
    assert export_tomorrow[96]["price"] == pytest.approx(PREDICTED)


@pytest.mark.usefixtures("copenhagen_time_zone")
def test_import_attributes_say_what_the_prices_contain(
    two_days: dict[str, Any],
) -> None:
    import_attrs = _attrs(_sensor(two_days, "import", "today"))
    export_attrs = _attrs(_sensor(two_days, "export", "today"))

    assert import_attrs["includes_vat"] is True
    assert import_attrs["includes_tariffs"] is True
    assert import_attrs["interval_minutes"] == 15
    assert import_attrs["known_until"] == "2026-09-26T00:00:00+02:00"
    assert export_attrs["includes_vat"] is False
    assert export_attrs["includes_tariffs"] is False


@pytest.mark.usefixtures("copenhagen_time_zone")
def test_fixed_tariff_applies_only_without_slot_tariffs() -> None:
    spot_data = _spot_data()
    without = _api_data(_predictions(DAY, 3, first_slot=40), spot_data)
    with_slots = _api_data(
        _predictions(DAY, 3, first_slot=40), spot_data, _tariffs(spot_data)
    )

    fixed = _attrs(_sensor(without, "import", "today", fixed_tariff=0.2))
    none = _attrs(_sensor(without, "import", "today"))
    slots = _attrs(_sensor(with_slots, "import", "today", fixed_tariff=0.2))
    export = _attrs(_sensor(without, "export", "today", fixed_tariff=0.2))

    assert fixed["prices_today"][0]["price"] == pytest.approx(
        (SPOT_TODAY + 0.2 + SURCHARGE) * (1 + VAT)
    )
    assert fixed["includes_tariffs"] is True
    assert none["prices_today"][0]["price"] == pytest.approx(
        (SPOT_TODAY + SURCHARGE) * (1 + VAT)
    )
    assert none["includes_tariffs"] is False
    assert slots["prices_today"][0]["price"] == pytest.approx(_import_price(SPOT_TODAY))
    # The export price never carries a tariff
    assert export["prices_today"][0]["price"] == pytest.approx(SPOT_TODAY)


@pytest.mark.usefixtures("copenhagen_time_zone")
def test_prices_are_always_per_kwh() -> None:
    """An entry showing MWh prices still gives Predbat kr/kWh; the surcharge follows."""
    spot_data = _spot_data()
    api_data = _api_data([], spot_data)
    output = PriceOutput(vat=VAT, surcharge=100.0, price_type="MWh", precision=4)

    import_today = _attrs(_sensor(api_data, "import", "today", output=output))
    export_today = _attrs(_sensor(api_data, "export", "today", output=output))

    assert import_today["prices_today"][0]["price"] == pytest.approx(
        (SPOT_TODAY + 0.1) * (1 + VAT)
    )
    assert export_today["prices_today"][0]["price"] == pytest.approx(SPOT_TODAY)


@pytest.mark.usefixtures("copenhagen_time_zone")
def test_hourly_average_gives_one_entry_per_hour(two_days: dict[str, Any]) -> None:
    output = PriceOutput(vat=VAT, precision=4, hourly_average=True)
    attrs = _attrs(_sensor(two_days, "export", "today", output=output))

    assert attrs["interval_minutes"] == 60
    assert len(attrs["prices_today"]) == 24
    assert attrs["prices_today"][0]["end"] == "2026-09-24T01:00:00+02:00"


@pytest.mark.usefixtures("copenhagen_time_zone")
def test_tomorrow_is_trimmed_to_the_attribute_budget_by_whole_hours(
    two_days: dict[str, Any],
) -> None:
    sensor = _sensor(two_days, "import", "tomorrow")
    attrs = _attrs(sensor)
    entries = attrs["prices_tomorrow"]

    assert attributes_size(attrs) <= ATTRIBUTE_BUDGET_BYTES
    # More than tomorrow alone, less than the whole 7-day forecast
    assert 96 < len(entries) < 6 * 96
    assert len(entries) % 4 == 0
    assert entries[-1]["end"].endswith(":00:00+02:00")


@pytest.mark.usefixtures("copenhagen_time_zone")
def test_hourly_tomorrow_fits_the_whole_forecast(two_days: dict[str, Any]) -> None:
    output = PriceOutput(vat=VAT, precision=4, hourly_average=True)
    attrs = _attrs(_sensor(two_days, "import", "tomorrow", output=output))

    assert attributes_size(attrs) <= ATTRIBUTE_BUDGET_BYTES
    # Tomorrow (confirmed) and six more days (forecast ends 2026-09-30 23:45)
    assert len(attrs["prices_tomorrow"]) == 6 * 24


@pytest.mark.usefixtures("copenhagen_time_zone")
def test_state_is_the_current_interval_today_and_the_first_one_tomorrow(
    two_days: dict[str, Any],
) -> None:
    assert _state(_sensor(two_days, "import", "today")) == pytest.approx(
        _import_price(SPOT_TODAY)
    )
    assert _state(_sensor(two_days, "export", "tomorrow")) == pytest.approx(
        SPOT_TOMORROW
    )


@pytest.mark.usefixtures("copenhagen_time_zone")
def test_state_is_none_without_prices() -> None:
    empty = _api_data(None, None)

    assert _state(_sensor(empty, "import", "today")) is None
    assert _state(_sensor(empty, "import", "tomorrow")) is None
    attrs = _attrs(_sensor(empty, "export", "tomorrow"))
    assert attrs["prices_tomorrow"] == []
    assert attrs["known_until"] is None


@pytest.mark.usefixtures("copenhagen_time_zone")
def test_without_the_model_only_the_confirmed_prices_are_exposed() -> None:
    api_data = _api_data(None, _spot_data())

    today = _attrs(_sensor(api_data, "export", "today"))["prices_today"]
    tomorrow = _attrs(_sensor(api_data, "export", "tomorrow"))["prices_tomorrow"]

    assert len(today) == 96
    assert len(tomorrow) == 96


def test_unique_ids_device_info_and_names() -> None:
    entry = MagicMock(entry_id="abc")
    entry.options = {CONF_PREDBAT_TARIFF: 0.3}
    sensors = build_predbat_sensors(MagicMock(), entry, {}, "DKK", PriceOutput())

    assert [s.unique_id for s in sensors] == [
        "open_spot_forecast_abc_predbat_import_today",
        "open_spot_forecast_abc_predbat_import_tomorrow",
        "open_spot_forecast_abc_predbat_export_today",
        "open_spot_forecast_abc_predbat_export_tomorrow",
    ]
    assert [s.translation_key for s in sensors] == [
        "predbat_import_today",
        "predbat_import_tomorrow",
        "predbat_export_today",
        "predbat_export_tomorrow",
    ]
    assert all(s.device_info == {"identifiers": {(DOMAIN, "abc")}} for s in sensors)
    assert all(s.has_entity_name for s in sensors)
    assert all(s.entity_category is None for s in sensors)
    assert all(s._fixed_tariff == pytest.approx(0.3) for s in sensors)
    assert sensors[0]._unrecorded_attributes == {"prices_today", "prices_tomorrow"}


@pytest.mark.asyncio
async def test_updates_on_price_and_forecast_signals() -> None:
    sensor = _sensor({}, "import", "today")
    with (
        patch.object(sensor, "async_on_remove") as on_remove,
        patch(
            "custom_components.open_spot_forecast.predbat_sensor.async_dispatcher_connect"
        ) as connect,
    ):
        await sensor.async_added_to_hass()

    signals = {call.args[1] for call in connect.call_args_list}
    assert signals == {UPDATE_SIGNAL, UPDATE_SIGNAL_FORECAST}
    assert on_remove.call_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled", [False, True])
async def test_option_adds_the_four_sensors(enabled: bool) -> None:
    hass = Mock()
    entry = MagicMock()
    entry.entry_id = "test"
    entry.data = {"region": "DK1", "currency": "DKK"}
    entry.options = {CONF_PREDBAT_SENSORS: enabled}
    hass.data = {DOMAIN: {"test": {"stromligning_data": {"current_price": 1.0}}}}
    async_add_entities = Mock()

    await async_setup_entry(hass, entry, async_add_entities)

    sensors = async_add_entities.call_args[0][0]
    predbat = [s for s in sensors if isinstance(s, PredbatRateSensor)]
    assert len(predbat) == (4 if enabled else 0)
    assert len(sensors) == 10 + len(predbat)


def test_fit_entries_trims_whole_hours_from_the_end() -> None:
    entries = [
        {"start": f"2026-09-25T{i // 4:02d}:{i % 4 * 15:02d}:00+02:00", "price": 1.0}
        for i in range(96)
    ]
    attrs = {"prices": entries, "other": "x"}
    budget = attributes_size({"prices": entries[:50], "other": "x"})

    fit_entries(attrs, "prices", 15, budget)

    assert attributes_size(attrs) <= budget
    assert len(attrs["prices"]) % 4 == 0
    assert 40 <= len(attrs["prices"]) <= 48


def test_fit_entries_can_empty_the_list() -> None:
    attrs = {"prices": [{"start": "x", "price": 1.0}]}

    fit_entries(attrs, "prices", 60, budget=1)

    assert attrs["prices"] == []


def test_fixed_tariff_schedule() -> None:
    schedule = TariffSchedule.fixed(0.2)

    assert schedule
    assert len(schedule) == 0
    assert schedule.at(datetime(2026, 9, 24, 10, tzinfo=UTC)) == pytest.approx(0.2)
    assert not TariffSchedule.fixed(0.0)


def test_price_output_per_kwh_and_raw_spot() -> None:
    output = PriceOutput(vat=VAT, surcharge=100.0, price_type="MWh", precision=2)

    per_kwh = output.per_kwh()
    assert per_kwh.price_type == "kWh"
    assert per_kwh.surcharge == pytest.approx(0.1)
    assert per_kwh.convert(1.0) == pytest.approx(1.38)
    assert PriceOutput().per_kwh() is not None

    raw = output.raw_spot()
    assert raw.vat == pytest.approx(0.0)
    assert raw.surcharge == pytest.approx(0.0)
    assert raw.convert(1.234) == pytest.approx(1.23)
