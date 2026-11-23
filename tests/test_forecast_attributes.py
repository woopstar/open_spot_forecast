"""The compact forecast attribute format (#38).

``compact`` exposes the forecast as parallel arrays (EpexPredictor's short
format plus confidence), so up to 168 hours fit the recorder's 16 KB limit.
``detailed`` stays the default.
"""

from datetime import UTC, date, datetime, timedelta
from typing import Any
from unittest.mock import MagicMock, Mock, patch
from zoneinfo import ZoneInfo

import pytest

from custom_components.open_spot_forecast.config_flow import (
    OpenSpotForecastOptionsFlow,
)
from custom_components.open_spot_forecast.const import (
    PREDICTION_HOURS_OPTIONS,
    RECORDER_MAX_ATTRIBUTES_BYTES,
)
from custom_components.open_spot_forecast.forecast_attributes import (
    attributes_size,
    compact_forecast,
    detailed_forecast,
    fit_compact,
)
from custom_components.open_spot_forecast.price_output import PriceOutput
from custom_components.open_spot_forecast.sensor import MLPredictionSensor
from custom_components.open_spot_forecast.time_slots import (
    slot_start_in_day,
    slots_in_local_day,
)

pytestmark = pytest.mark.usefixtures("copenhagen_time_zone")

CPH = ZoneInfo("Europe/Copenhagen")
START = datetime(2026, 9, 24, 0, 0, tzinfo=CPH)
NOW = datetime(2026, 9, 24, 0, 5, tzinfo=CPH)


def _predictions(slots: int) -> list[dict[str, Any]]:
    """Return 15-min predictions with realistic, varied DKK/kWh spot prices.

    Prices range over -1.3..3.5 with many decimals, so every converted price
    has the full precision (the worst case for the attribute size).
    """
    first = START.astimezone(UTC)
    predictions = []
    for index in range(slots):
        start = first + timedelta(minutes=15 * index)
        predictions.append(
            {
                "start": start.astimezone(CPH).isoformat(),
                "end": (start + timedelta(minutes=15)).astimezone(CPH).isoformat(),
                "price": -1.3 + (index * 0.0371937) % 4.8,
                "confidence": 0.3 + (index % 67) / 100,
            }
        )
    return predictions


def _sensor(
    slots: int,
    hours: int,
    attribute_format: str | None = None,
    output: PriceOutput | None = None,
) -> MLPredictionSensor:
    predictor = MagicMock()
    predictor.predictions = _predictions(slots)
    predictor.is_trained = True
    predictor.get_prediction_stats.return_value = {
        "mean_confidence": 0.62,
        "total_predictions": slots,
        "is_ml_model": True,
        "training_samples": 5760,
    }
    args: list[Any] = [
        Mock(),
        MagicMock(entry_id="test"),
        {"ml_predictor": predictor, "zone_weather": True},
        "DKK",
        output or PriceOutput(),
        hours,
    ]
    if attribute_format is not None:
        args.append(attribute_format)
    return MLPredictionSensor(*args)


def _attributes(sensor: MLPredictionSensor) -> dict[str, Any]:
    with patch("homeassistant.util.dt.utcnow", return_value=NOW.astimezone(UTC)):
        return sensor.extra_state_attributes


def _stored_attributes(sensor: MLPredictionSensor) -> dict[str, Any]:
    """Return the attributes plus the ones Home Assistant adds to the state."""
    return {
        **_attributes(sensor),
        "attribution": sensor.attribution,
        "device_class": "monetary",
        "friendly_name": "Open Spot Forecast DK1 Price Forecast (ML)",
        "icon": "mdi:brain",
        "unit_of_measurement": "DKK/kWh",
    }


# --- The layout ----------------------------------------------------------------------


def test_compact_is_parallel_arrays() -> None:
    entries = PriceOutput().forecast(_predictions(3))

    compact = compact_forecast(entries, "DKK/kWh", 15)

    assert compact["interval_minutes"] == 15
    assert compact["unit"] == "DKK/kWh"
    assert compact["s"] == [
        int(START.timestamp()),
        int(START.timestamp()) + 900,
        int(START.timestamp()) + 1800,
    ]
    assert compact["t"] == [entry["price"] for entry in entries]
    assert compact["c"] == [30, 31, 32]


def test_compact_keeps_the_arrays_aligned() -> None:
    entries = [
        {"start": "2026-09-24T00:00:00+02:00", "price": 1.0, "confidence": None},
        {"start": "garbage", "price": 2.0, "confidence": 0.5},
        {"start": "2026-09-24T00:30:00+02:00", "price": 3.0, "confidence": 0.555},
    ]

    compact = compact_forecast(entries, "DKK/kWh", 15)

    assert len(compact["s"]) == len(compact["t"]) == len(compact["c"]) == 2
    assert compact["t"] == [1.0, 3.0]
    assert compact["c"] == [None, 56]


def test_compact_is_at_least_five_times_smaller_per_slot() -> None:
    entries = PriceOutput().forecast(_predictions(96))

    detailed = attributes_size({"p": detailed_forecast(entries, "DKK/kWh")})
    compact = attributes_size({"p": compact_forecast(entries, "DKK/kWh", 15)})

    assert detailed / 96 >= 5 * (compact / 96)


# --- The sensor ----------------------------------------------------------------------


def test_detailed_stays_the_default() -> None:
    attributes = _attributes(_sensor(200, 48))

    predictions = attributes["predictions"]
    assert isinstance(predictions, list)
    assert len(predictions) == 192
    assert set(predictions[0]) == {"start", "end", "price", "unit", "confidence"}


def test_detailed_is_capped_at_72_hours() -> None:
    attributes = _attributes(_sensor(700, 168, "detailed"))

    assert len(attributes["predictions"]) == 72 * 4


def test_168_hours_of_compact_attributes_fit_16_kb() -> None:
    assert max(PREDICTION_HOURS_OPTIONS) == 168
    sensor = _sensor(700, 168, "compact")

    stored = _stored_attributes(sensor)

    compact = stored["predictions"]
    assert len(compact["s"]) == len(compact["t"]) == len(compact["c"]) == 168 * 4
    assert attributes_size(stored) < RECORDER_MAX_ATTRIBUTES_BYTES
    # The detailed format of the same window would not fit
    entries = PriceOutput().forecast(_predictions(168 * 4))
    assert (
        attributes_size({"predictions": detailed_forecast(entries, "DKK/kWh")})
        > RECORDER_MAX_ATTRIBUTES_BYTES
    )


def test_compact_hourly_has_one_entry_per_hour() -> None:
    output = PriceOutput(hourly_average=True)

    compact = _attributes(_sensor(700, 168, "compact", output))["predictions"]

    assert compact["interval_minutes"] == 60
    assert len(compact["s"]) == 168
    assert {b - a for a, b in zip(compact["s"], compact["s"][1:])} == {3600}


def test_an_oversized_compact_forecast_is_trimmed_by_whole_hours() -> None:
    # MWh with 6 decimals: every price is about twice as long
    output = PriceOutput(price_type="MWh", precision=6)
    sensor = _sensor(700, 168, "compact", output)

    stored = _stored_attributes(sensor)

    compact = stored["predictions"]
    assert 0 < len(compact["s"]) < 168 * 4
    assert len(compact["s"]) % 4 == 0
    assert len(compact["s"]) == len(compact["t"]) == len(compact["c"])
    assert attributes_size(stored) < RECORDER_MAX_ATTRIBUTES_BYTES


def test_fit_compact_can_trim_everything() -> None:
    attributes = {
        "predictions": compact_forecast(
            PriceOutput().forecast(_predictions(8)), "DKK/kWh", 15
        )
    }

    fit_compact(attributes, "predictions", budget=10)

    assert attributes["predictions"]["s"] == []
    assert attributes["predictions"]["t"] == []
    fit_compact(attributes, "predictions", budget=10)
    assert attributes["predictions"]["c"] == []


def test_the_state_is_the_same_in_both_formats() -> None:
    detailed = _sensor(200, 48, "detailed")
    compact = _sensor(200, 48, "compact")

    with patch("homeassistant.util.dt.utcnow", return_value=NOW.astimezone(UTC)):
        assert compact.native_value == pytest.approx(detailed.native_value)
    assert _attributes(compact)["predictions"]["t"] == [
        entry["price"] for entry in _attributes(detailed)["predictions"]
    ]


# --- DST change days (#279) ----------------------------------------------------------


def _day_predictions(day: date) -> list[dict[str, Any]]:
    """Return a prediction for every slot of a local day: price = slot index."""
    return [
        {
            "start": slot_start_in_day(day, index).isoformat(),
            "end": slot_start_in_day(day, index + 1).isoformat(),
            "price": float(index),
            "confidence": 0.5,
        }
        for index in range(slots_in_local_day(day))
    ]


def _local_times(entries: list[dict[str, Any]]) -> list[str]:
    return [entry["start"].split("T")[1] for entry in entries]


def test_compact_has_every_slot_of_a_dst_day(dst_day: tuple[date, int]) -> None:
    day, slots = dst_day
    entries = PriceOutput(vat=0.0).forecast(_day_predictions(day))

    compact = compact_forecast(entries, "DKK/kWh", 15)

    assert len(compact["s"]) == len(compact["t"]) == len(compact["c"]) == slots
    # Unix seconds, so no gap at the skipped hour and no overlap at the repeated one
    assert {b - a for a, b in zip(compact["s"], compact["s"][1:])} == {900}
    assert compact["t"] == pytest.approx([float(index) for index in range(slots)])


def test_detailed_keeps_both_passes_and_skips_the_missing_hour(
    dst_day: tuple[date, int],
) -> None:
    day, slots = dst_day
    detailed = detailed_forecast(
        PriceOutput(vat=0.0).forecast(_day_predictions(day)), "DKK/kWh"
    )

    assert len(detailed) == slots
    starts = [entry["start"] for entry in detailed]
    assert len(set(starts)) == slots
    times = _local_times(detailed)
    # None of 02:00-02:59 in spring; in autumn each of its slots twice
    assert sum(time.startswith("02:") for time in times) == slots - 92
    if slots == 100:
        assert times.count("02:15:00+02:00") == times.count("02:15:00+01:00") == 1


def test_compact_hourly_has_23_or_25_hours(dst_day: tuple[date, int]) -> None:
    day, slots = dst_day
    output = PriceOutput(vat=0.0, hourly_average=True)

    compact = compact_forecast(output.forecast(_day_predictions(day)), "DKK/kWh", 60)

    assert len(compact["s"]) == slots // 4
    assert {b - a for a, b in zip(compact["s"], compact["s"][1:])} == {3600}
    assert compact["t"][0] == pytest.approx(1.5)


def test_fit_compact_trims_whole_hours_of_a_dst_day(
    dst_day: tuple[date, int],
) -> None:
    day, slots = dst_day
    entries = PriceOutput(vat=0.0).forecast(_day_predictions(day))
    attributes = {"predictions": compact_forecast(entries, "DKK/kWh", 15)}
    untrimmed = attributes_size(attributes)

    # Room for the whole day: nothing is trimmed
    fit_compact(attributes, "predictions", budget=untrimmed)
    assert len(attributes["predictions"]["s"]) == slots

    # Room for about half the day: whole local hours are kept from the start
    fit_compact(attributes, "predictions", budget=untrimmed // 2)
    kept = attributes["predictions"]["s"]
    assert 0 < len(kept) < slots
    assert len(kept) % 4 == 0
    assert kept == compact_forecast(entries, "DKK/kWh", 15)["s"][: len(kept)]
    # Every kept hour starts on a local full hour
    hour_starts = [datetime.fromtimestamp(s, CPH) for s in kept[::4]]
    assert all(start.minute == 0 for start in hour_starts)
    # The trim is at the end of the day: the DST hour is kept as it was
    kept_times = [datetime.fromtimestamp(s, CPH) for s in kept]
    assert sum(start.hour == 2 for start in kept_times) == slots - 92
    assert attributes_size(attributes) <= untrimmed // 2


# --- The options flow ----------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_options_flow_offers_the_format() -> None:
    entry = MagicMock()
    entry.data = {"region": "DK1"}
    entry.options = {}
    hass = Mock()
    hass.config_entries.async_get_known_entry.return_value = entry
    flow: Any = OpenSpotForecastOptionsFlow()
    flow.hass = hass
    flow.handler = "test"
    flow.async_show_form = Mock(return_value={"type": "show_form"})

    await flow.async_step_init()

    schema = flow.async_show_form.call_args.kwargs["data_schema"].schema
    keys = {str(key): key for key in schema}
    assert keys["attribute_format"].default() == "detailed"
    assert schema[keys["attribute_format"]].config["options"] == [
        "detailed",
        "compact",
    ]
    assert 168 in schema[keys["prediction_hours"]].container
