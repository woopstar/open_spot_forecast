"""Training and prediction build identical feature rows (issue #17)."""

import math
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import Mock
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from custom_components.open_spot_forecast.ml.features import (
    FEATURE_NAMES,
    SlotInputs,
    build_feature_row,
    build_feature_vector,
    slot_time_features,
    wind_power_curve,
)
from custom_components.open_spot_forecast.ml.predictor import SpotPricePredictor
from custom_components.open_spot_forecast.ml.training_inputs import TrainingInputs

TZ = ZoneInfo("Europe/Copenhagen")
# Monday 2026-06-01 12:00 local = 10:00 UTC
SLOT = datetime(2026, 6, 1, 12, 0, tzinfo=TZ)
WEATHER = {
    "temperature": 12.0,
    "wind_speed": 8.0,
    "wind_direction": 240.0,
    "cloud_coverage": 75.0,
    "humidity": 80.0,
}
NORDPOOL = {
    "consumption": 3000.0,
    "solar": 500.0,
    "wind_offshore": 800.0,
    "wind_onshore": 700.0,
}
WEATHER_FEATURES = (
    "wind_speed_mean",
    "wind_power_estimate",
    "wind_direction",
    "cloud_coverage",
    "humidity",
    "temperature",
)
NORDPOOL_FEATURES = (
    "consumption_forecast",
    "solar_generation",
    "wind_offshore",
    "wind_onshore",
    "net_demand",
    "wind_share",
)


@pytest.fixture
def predictor(tmp_path: Path) -> Iterator[SpotPricePredictor]:
    """Predictor with real SQLite storage in Copenhagen time."""
    hass = Mock()
    hass.config.path.return_value = str(tmp_path / ".storage")
    predictor = SpotPricePredictor(hass, "DK1")
    yield predictor
    predictor.storage.close()


def _utc_iso(moment: datetime) -> str:
    return moment.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _store_history(predictor: SpotPricePredictor, start: datetime) -> None:
    """Store the snapshot taken 3 s into the slot and the hour's prognoses."""
    predictor.storage.insert_weather_snapshot(
        (start + timedelta(seconds=3)).isoformat(),
        WEATHER["temperature"],
        WEATHER["wind_speed"],
        WEATHER["wind_direction"],
        WEATHER["cloud_coverage"],
        WEATHER["humidity"],
        None,
    )
    predictor.storage.insert_nordpool_prognoses_batch(
        [{"timestamp": _utc_iso(start)} | NORDPOOL]
    )


def _live_data(start: datetime) -> dict:
    """The same values as live forecasts: HA's UTC forecast, Nordpool's UTC keys."""
    return {
        "weather_forecast": [
            {
                "datetime": start.astimezone(UTC).isoformat(),
                "temperature": WEATHER["temperature"],
                "wind_speed": WEATHER["wind_speed"],
                "wind_bearing": WEATHER["wind_direction"],
                "cloud_coverage": WEATHER["cloud_coverage"],
                "humidity": WEATHER["humidity"],
            }
        ],
        "consumption_prognosis": {_utc_iso(start): NORDPOOL["consumption"]},
        "production_prognosis": [
            {
                "deliveryStart": _utc_iso(start),
                "solar": NORDPOOL["solar"],
                "wind_offshore": NORDPOOL["wind_offshore"],
                "wind_onshore": NORDPOOL["wind_onshore"],
            }
        ],
    }


def _training_row(predictor: SpotPricePredictor, start: datetime) -> dict:
    predictor.price_history = [{"date": start.date().isoformat(), "prices": [1.0] * 96}]
    _, features = predictor.get_all_historical_prices()
    return next(f for f in features if f["start"] == start.isoformat())


def test_same_inputs_give_identical_training_and_prediction_rows(
    predictor: SpotPricePredictor,
) -> None:
    """A slot's stored actuals and its live forecast yield the same model row."""
    _store_history(predictor, SLOT)

    training = build_feature_vector(_training_row(predictor, SLOT))
    (prediction_row,) = predictor._combine_features(
        [slot_time_features(SLOT)], _live_data(SLOT)
    )
    prediction = build_feature_vector(prediction_row)

    assert not any(math.isnan(value) for value in training)
    assert training == prediction
    row = dict(zip(FEATURE_NAMES, training, strict=True))
    assert row["wind_power_estimate"] == pytest.approx(wind_power_curve(8.0))
    assert row["net_demand"] == pytest.approx(3000 - 500 - 800 - 700)
    assert row["wind_share"] == pytest.approx(1500 / 3000)


def test_every_quarter_of_an_hour_uses_the_hours_prognosis(
    predictor: SpotPricePredictor,
) -> None:
    """Nordpool is stored hourly, so prediction uses the hour's value for all slots."""
    live = _live_data(SLOT)
    live["production_prognosis"].append(
        {
            "deliveryStart": _utc_iso(SLOT + timedelta(minutes=15)),
            "solar": 9999.0,
            "wind_offshore": 9999.0,
            "wind_onshore": 9999.0,
        }
    )
    quarters = [slot_time_features(SLOT + timedelta(minutes=15 * i)) for i in range(4)]

    rows = predictor._combine_features(quarters, live)

    assert [row["solar_generation"] for row in rows] == [NORDPOOL["solar"]] * 4
    assert [row["consumption_forecast"] for row in rows] == [
        NORDPOOL["consumption"]
    ] * 4


def test_slots_beyond_the_forecasts_have_unknown_inputs(
    predictor: SpotPricePredictor,
) -> None:
    """Days without forecasts are NaN, not zero demand, 15 °C or 50 % humidity."""
    later = SLOT + timedelta(days=3)

    today, day_four = predictor._combine_features(
        [slot_time_features(SLOT), slot_time_features(later)], _live_data(SLOT)
    )
    vector = dict(zip(FEATURE_NAMES, build_feature_vector(day_four), strict=True))

    assert today["temperature"] == pytest.approx(WEATHER["temperature"])
    for name in WEATHER_FEATURES + NORDPOOL_FEATURES:
        assert day_four[name] is None
        assert math.isnan(vector[name])


def test_training_slots_without_stored_history_are_unknown(
    predictor: SpotPricePredictor,
) -> None:
    """Training does not borrow the current forecast for slots without history."""
    row = _training_row(predictor, SLOT)

    for name in WEATHER_FEATURES + NORDPOOL_FEATURES:
        assert row[name] is None


def test_no_training_feature_is_constant(predictor: SpotPricePredictor) -> None:
    """Every feature varies across the training rows when its inputs do."""
    first_day = datetime(2026, 6, 1, tzinfo=TZ)
    rng = np.random.default_rng(0)
    entries = []
    for day in range(7):
        date = (first_day + timedelta(days=day)).date()
        entries.append({"date": date.isoformat(), "prices": [1.0] * 96})
        for hour in range(24):
            start = datetime(date.year, date.month, date.day, hour, tzinfo=TZ)
            temperature, wind, direction, cloud, humidity = rng.uniform(1, 20, 5)
            predictor.storage.insert_weather_snapshot(
                start.isoformat(),
                float(temperature),
                float(wind),
                float(direction),
                float(cloud),
                float(humidity),
                None,
            )
            predictor.storage.insert_nordpool_prognoses_batch(
                [
                    {
                        "timestamp": _utc_iso(start),
                        "consumption": float(rng.uniform(2000, 4000)),
                        "solar": float(rng.uniform(0, 900)),
                        "wind_offshore": float(rng.uniform(0, 900)),
                        "wind_onshore": float(rng.uniform(0, 900)),
                    }
                ]
            )
    predictor.price_history = entries

    _, features = predictor.get_all_historical_prices()
    X = np.array([build_feature_vector(feature) for feature in features])

    for column, name in enumerate(FEATURE_NAMES):
        known = X[~np.isnan(X[:, column]), column]
        assert len(np.unique(known)) > 1, name


def test_training_inputs_read_naive_snapshots_as_local_time() -> None:
    """Snapshots stored before #17 have no offset; they are local wall-clock time."""
    naive = {"timestamp": "2026-06-01T12:00:03.123456"} | WEATHER
    aware = {"timestamp": "2026-06-01T12:15:02+02:00", "temperature": 13.0}
    inputs = TrainingInputs([naive, aware], [], TZ)

    assert inputs.for_slot(SLOT).temperature == pytest.approx(12.0)
    assert inputs.for_slot(SLOT + timedelta(minutes=15)).temperature == (
        pytest.approx(13.0)
    )
    assert inputs.for_slot(SLOT + timedelta(minutes=30)) == SlotInputs()


def test_training_inputs_ignore_unparsable_timestamps() -> None:
    inputs = TrainingInputs(
        [{"timestamp": "garbage", "temperature": 5.0}],
        [{"timestamp": None, "consumption": 1.0}],
        TZ,
    )

    assert inputs.for_slot(SLOT) == SlotInputs()


def test_history_loaders_round_trip(predictor: SpotPricePredictor) -> None:
    """Stored snapshots and prognoses come back in timestamp order."""
    _store_history(predictor, SLOT + timedelta(hours=1))
    _store_history(predictor, SLOT)

    weather = predictor.storage.load_weather_history()
    nordpool = predictor.storage.load_nordpool_history()

    assert [row["temperature"] for row in weather] == [12.0, 12.0]
    assert weather[0]["timestamp"] < weather[1]["timestamp"]
    assert nordpool[0] == {"timestamp": "2026-06-01T10:00:00Z"} | NORDPOOL


@pytest.mark.parametrize(
    ("inputs", "net_demand", "wind_share"),
    [
        (SlotInputs(consumption=100.0), None, None),
        (
            SlotInputs(consumption=0.0, wind_offshore=1.0, wind_onshore=1.0),
            None,
            None,
        ),
        (
            SlotInputs(
                consumption=100.0,
                wind_offshore=10.0,
                wind_onshore=20.0,
            ),
            None,
            0.3,
        ),
    ],
)
def test_derived_features_need_all_their_inputs(
    inputs: SlotInputs, net_demand: float | None, wind_share: float | None
) -> None:
    row = build_feature_row(SLOT, inputs)

    assert row["net_demand"] == net_demand
    assert row["wind_share"] == (
        pytest.approx(wind_share) if wind_share is not None else None
    )


def test_solar_scale_is_learned_but_not_a_model_input(
    predictor: SpotPricePredictor,
) -> None:
    """The site's Solcast ratio is tracked; no site-solar column reaches the model."""
    predictor._update_solar_scale(
        {"solar_power": 3.0, "solcast_forecast": {"estimate_today": 2.0}}
    )

    # First sample: alpha = min(0.3, 1 / samples) = 0.3
    assert predictor.solar_scale == pytest.approx(0.7 * 1.0 + 0.3 * 1.5)
    assert predictor._solar_scale_samples == 1
    assert "solar_power_estimate" not in FEATURE_NAMES
    assert "solar_radiation_mean" not in FEATURE_NAMES


@pytest.mark.parametrize(
    "weather_data",
    [
        {},
        {"solar_power": 3.0},
        {"solar_power": 3.0, "solcast_forecast": {"estimate_today": None}},
        {"solar_power": 100.0, "solar_forecast": {"estimate_today": 2.0}},
        {"solar_power": 0.0, "solcast_forecast": {"estimate_today": 2.0}},
    ],
)
def test_solar_scale_ignores_missing_or_implausible_readings(
    predictor: SpotPricePredictor, weather_data: dict
) -> None:
    predictor._update_solar_scale(weather_data)

    assert predictor.solar_scale == pytest.approx(1.0)
    assert predictor._solar_scale_samples == 0
