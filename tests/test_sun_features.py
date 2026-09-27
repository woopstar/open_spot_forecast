"""Sun-position and 15-minute time features (#25)."""

import math
from datetime import UTC, datetime, timedelta
from pathlib import Path
from statistics import fmean
from unittest.mock import Mock
from zoneinfo import ZoneInfo

import pytest

from custom_components.open_spot_forecast.const import WEATHER_POINTS
from custom_components.open_spot_forecast.ml.features import (
    FEATURE_NAMES,
    SlotInputs,
    build_feature_row,
    build_feature_vector,
    slot_time_features,
)
from custom_components.open_spot_forecast.ml.predictor import SpotPricePredictor
from custom_components.open_spot_forecast.ml.sun import (
    SUN_FEATURES,
    sun_features,
    zone_centre,
)

TZ = ZoneInfo("Europe/Copenhagen")
DK1 = zone_centre("DK1") or (0.0, 0.0)
LONGYEARBYEN = (78.22, 15.65)
TIME_FEATURES = ("slot_sin", "slot_cos", "morning_peak")


def _vector(start: datetime, region: str | None = "DK1") -> dict:
    row = build_feature_row(start, SlotInputs(), region)
    return dict(zip(FEATURE_NAMES, build_feature_vector(row), strict=True))


def _known(start: datetime, centre: tuple[float, float] = DK1) -> dict[str, float]:
    """Return ``sun_features`` that must all be known."""
    features = sun_features(start, centre)
    known = {name: value for name, value in features.items() if value is not None}
    assert known.keys() == features.keys()
    return known


def test_sun_features_are_model_inputs() -> None:
    assert set(SUN_FEATURES) <= set(FEATURE_NAMES)
    assert set(TIME_FEATURES) <= set(FEATURE_NAMES)
    # Seconds to 19:00 would be morning_peak shifted by 11 h: the same splits
    assert "evening_peak" not in FEATURE_NAMES


def test_zone_centre_is_the_mean_of_the_weather_points() -> None:
    points = WEATHER_POINTS["DK1"]

    assert zone_centre("DK1") == pytest.approx(
        (fmean(p[0] for p in points), fmean(p[1] for p in points))
    )
    assert zone_centre("XX") is None
    assert zone_centre(None) is None


def test_the_four_quarters_of_an_hour_differ() -> None:
    """17:00 and 17:45 no longer share every time feature."""
    starts = [datetime(2026, 1, 12, 17, 15 * i, tzinfo=TZ) for i in range(4)]
    time_columns = [
        tuple(vector[name] for name in (*TIME_FEATURES, *SUN_FEATURES))
        for vector in map(_vector, starts)
    ]

    assert len(set(time_columns)) == 4
    for name in TIME_FEATURES:
        assert (
            len({columns[TIME_FEATURES.index(name)] for columns in time_columns}) == 4
        )


@pytest.mark.parametrize(
    "day",
    [
        datetime(2026, 3, 29, tzinfo=TZ),  # spring forward, 92 slots
        datetime(2026, 6, 1, tzinfo=TZ),
        datetime(2026, 10, 25, tzinfo=TZ),  # fall back, 100 slots
        datetime(2026, 12, 1, tzinfo=TZ),
    ],
)
def test_time_of_day_features_follow_the_local_clock(day: datetime) -> None:
    """17:00 local has the same time features on DST days and in either season."""
    feature = slot_time_features(day.replace(hour=17))

    assert feature["morning_peak"] == 9 * 3600
    assert feature["slot_sin"] == pytest.approx(math.sin(2 * math.pi * 17 / 24))
    assert feature["slot_cos"] == pytest.approx(math.cos(2 * math.pi * 17 / 24))


def test_repeated_fall_back_hour_keeps_its_wall_clock_features() -> None:
    """Both 02:15 slots of the fall-back day are 02:15 on the local clock."""
    first = datetime(2026, 10, 25, 2, 15, tzinfo=TZ)
    second = (first.astimezone(UTC) + timedelta(hours=1)).astimezone(TZ)
    assert second.utcoffset() != first.utcoffset()

    one, two = slot_time_features(first), slot_time_features(second)

    assert {name: one[name] for name in TIME_FEATURES} == {
        name: two[name] for name in TIME_FEATURES
    }
    # The sun moved on in that real hour
    assert _vector(second)["sun_azimuth"] > _vector(first)["sun_azimuth"]


def test_sun_position_at_the_june_solstice() -> None:
    """At solar noon the sun is due south, 90 - latitude + 23.44 degrees high."""
    latitude, longitude = DK1
    # Solar noon in UTC is about 12:00 - longitude / 15 h (equation of time ~ -2 min)
    noon = datetime(2026, 6, 21, 12, 0, tzinfo=UTC) - timedelta(hours=longitude / 15)
    slot = noon - timedelta(minutes=7.5)

    features = _known(slot.astimezone(TZ))

    assert features["sun_elevation"] == pytest.approx(90 - latitude + 23.44, abs=0.3)
    assert features["sun_azimuth"] == pytest.approx(180, abs=1.5)


def test_since_sunrise_and_sunset_are_seconds_to_the_slot_middle() -> None:
    """Negative before the event, the day length apart, zero near sunrise."""
    day = [
        _known(datetime(2026, 6, 21, tzinfo=TZ) + timedelta(minutes=15 * i))
        for i in range(96)
    ]
    rise = [slot["since_sunrise"] for slot in day]
    set_ = [slot["since_sunset"] for slot in day]

    day_length = rise[0] - set_[0]
    assert 17 * 3600 < day_length < 17.7 * 3600  # Aarhus, midsummer
    assert all(
        r - s == pytest.approx(day_length) for r, s in zip(rise, set_, strict=True)
    )
    assert rise[0] < 0 < rise[-1]
    assert set_[0] < 0 < set_[-1]
    assert all(
        b - a == pytest.approx(900) for a, b in zip(rise, rise[1:], strict=False)
    )
    # The sunrise slot is within half a slot of the event
    assert min(abs(value) for value in rise) <= 450


def test_last_slot_of_the_day_uses_that_days_sunrise() -> None:
    features = _known(datetime(2026, 1, 12, 23, 45, tzinfo=TZ))

    assert 14 * 3600 < features["since_sunrise"] < 16 * 3600
    assert 6 * 3600 < features["since_sunset"] < 8 * 3600


@pytest.mark.parametrize("month", [6, 12])
def test_polar_day_and_night_have_unknown_sunrise_and_sunset(month: int) -> None:
    """No sunrise or sunset → NaN for the model; the position is still known."""
    features = sun_features(datetime(2026, month, 21, 12, tzinfo=TZ), LONGYEARBYEN)
    vector = dict(zip(FEATURE_NAMES, build_feature_vector(features), strict=True))

    assert features["since_sunrise"] is None
    assert features["since_sunset"] is None
    assert math.isnan(vector["since_sunrise"])
    assert math.isnan(vector["since_sunset"])
    assert (vector["sun_elevation"] > 0) == (month == 6)
    assert not math.isnan(vector["sun_azimuth"])


@pytest.mark.parametrize("region", [None, "XX"])
def test_unknown_region_gives_unknown_sun_features(region: str | None) -> None:
    vector = _vector(datetime(2026, 6, 1, 12, tzinfo=TZ), region)

    assert all(math.isnan(vector[name]) for name in SUN_FEATURES)
    assert vector["morning_peak"] == pytest.approx(4 * 3600)


@pytest.mark.parametrize(
    "start",
    [
        datetime(2026, 6, 1, 12, 0, tzinfo=TZ),
        datetime(2026, 3, 29, 1, 45, tzinfo=TZ),  # the slot before spring forward
        datetime(2026, 10, 25, 2, 30, tzinfo=TZ, fold=1),  # the repeated hour
    ],
)
def test_iso_round_trip_gives_the_same_row(start: datetime) -> None:
    """Prediction parses slot starts with a fixed offset; training has the zone."""
    parsed = datetime.fromisoformat(start.isoformat())

    assert _vector(parsed) == pytest.approx(_vector(start), nan_ok=True)


def test_training_and_prediction_rows_use_the_regions_centre(tmp_path: Path) -> None:
    hass = Mock()
    hass.config.path.return_value = str(tmp_path / ".storage")
    predictor = SpotPricePredictor(hass, "DK2")
    predictor.price_history = [{"date": "2026-06-01", "prices": [1.0] * 96}]
    start = datetime(2026, 6, 1, 9, 15, tzinfo=TZ)

    _, training = predictor.get_all_historical_prices()
    (prediction,) = predictor._combine_features([slot_time_features(start)], {})
    predictor.storage.close()

    expected = sun_features(start, zone_centre("DK2"))
    assert {name: training[37][name] for name in SUN_FEATURES} == expected
    assert {name: prediction[name] for name in SUN_FEATURES} == expected
    assert expected != sun_features(start, DK1)
