"""Feature engineering for the spot price predictor.

Every model input row, for training and for prediction, is built by
``build_feature_row`` from a slot start and that slot's ``SlotInputs``. The
two phases only differ in where the inputs come from: stored actuals and
prognoses for training (``ml/training_inputs.py``), live forecasts and
prognoses for prediction (``FeatureMixin._combine_features``). An input that
is not known for a slot is ``None`` in the row and NaN in the model input;
it is never replaced by an invented value.
"""

import logging
import math
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta, tzinfo
from typing import Any

import numpy as np

from homeassistant.util import dt as dt_util

from ..time_slots import first_prediction_slot
from .base import PredictorBase

_LOGGER = logging.getLogger(__name__)

# The canonical 17-feature model input, in column order (docs/ml_documentation.md).
FEATURE_NAMES: tuple[str, ...] = (
    "hour",
    "day_of_week",
    "is_weekend",
    "hour_sin",
    "hour_cos",
    "wind_speed_mean",
    "wind_power_estimate",
    "wind_direction",
    "cloud_coverage",
    "humidity",
    "temperature",
    "consumption_forecast",
    "solar_generation",
    "wind_offshore",
    "wind_onshore",
    "net_demand",
    "wind_share",
)

HOUR_SECONDS = 3600


@dataclass(frozen=True)
class SlotInputs:
    """Raw inputs for one 15-minute slot; ``None`` means unknown.

    Weather values are for the slot, wind speed in m/s (training: the stored
    snapshot, taken every 15 minutes; prediction: the hourly forecast for the
    slot's hour). Nordpool values are the day-ahead prognoses for the slot's
    hour, in MW.
    """

    temperature: float | None = None
    wind_speed: float | None = None
    wind_direction: float | None = None
    cloud_coverage: float | None = None
    humidity: float | None = None
    consumption: float | None = None
    solar_generation: float | None = None
    wind_offshore: float | None = None
    wind_onshore: float | None = None


def optional_float(value: Any) -> float | None:
    """Return ``value`` as a finite float, or None if missing or not a number."""
    if value is None:
        return None
    try:
        number = float(value)
    except ValueError, TypeError:
        return None
    return number if math.isfinite(number) else None


def floor_epoch(moment: datetime, tz: tzinfo, step_seconds: int = 900) -> int:
    """Return the UTC epoch of ``moment`` floored to ``step_seconds``.

    A naive ``moment`` is local time in ``tz``. Slots, forecasts and stored
    history are all matched on this one UTC key, so neither a DST change nor
    a timestamp's format can shift a value onto the wrong slot.
    """
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=tz)
    epoch = int(moment.timestamp())
    return epoch - epoch % step_seconds


def utc_epoch(timestamp: Any, tz: tzinfo, step_seconds: int = 900) -> int | None:
    """Return ``floor_epoch`` of an ISO timestamp, or None if it does not parse.

    A timestamp without an offset is local time in ``tz`` (weather snapshots
    were stored that way before #17).
    """
    try:
        moment = datetime.fromisoformat(str(timestamp))
    except ValueError:
        return None
    return floor_epoch(moment, tz, step_seconds)


def hour_epoch(timestamp: Any, tz: tzinfo) -> int | None:
    """Return the UTC epoch of the start of an ISO timestamp's hour."""
    return utc_epoch(timestamp, tz, HOUR_SECONDS)


def wind_power_curve(wind_speed: float) -> float:
    """Return a simplified turbine power curve (0-1) for a wind speed in m/s."""
    if wind_speed < 3:
        return 0.0
    if wind_speed < 12:
        return ((wind_speed - 3) / 9) ** 3
    if wind_speed < 25:
        return 1.0
    return 0.0


def _to_float(value: Any) -> float:
    """Coerce a feature value to float; None and non-numerics become NaN."""
    number = optional_float(value)
    return math.nan if number is None else number


def build_feature_vector(feature: dict[str, Any]) -> list[float]:
    """Return the model input row for one slot's feature dict.

    Training, prediction, hyperparameter search and the dev backtest
    (``scripts/backtest.py``) all build rows here, so ``FEATURE_NAMES`` is
    the single source of truth for the column order. A missing or unknown
    feature is NaN, which the price model handles natively.

    Args:
        feature: Feature dict for one slot, as produced by ``build_feature_row``.

    Returns:
        ``len(FEATURE_NAMES)`` floats in ``FEATURE_NAMES`` order.
    """
    return [_to_float(feature.get(name)) for name in FEATURE_NAMES]


def slot_time_features(start: datetime, interval_minutes: int = 15) -> dict[str, Any]:
    """Return the time features for the slot beginning at ``start``.

    Args:
        start: Timezone-aware slot start in the price region's local time.
        interval_minutes: Slot length; ``end`` is computed in UTC so a slot
            spanning a DST change still lasts ``interval_minutes``.

    Returns:
        Feature dict with ``start``/``end`` ISO strings and the time-of-day
        and day-of-week features used by the model.
    """
    end = (start.astimezone(UTC) + timedelta(minutes=interval_minutes)).astimezone(
        start.tzinfo
    )
    weekday = start.weekday()
    return {
        "start": start.isoformat(),
        "end": end.isoformat(),
        "hour": start.hour,
        "minute": start.minute,
        "day_of_week": weekday,
        "is_weekend": 1 if weekday >= 5 else 0,
        "month": start.month,
        "hour_sin": float(np.sin(2 * np.pi * start.hour / 24)),
        "hour_cos": float(np.cos(2 * np.pi * start.hour / 24)),
        "dow_sin": float(np.sin(2 * np.pi * weekday / 7)),
        "dow_cos": float(np.cos(2 * np.pi * weekday / 7)),
    }


def build_feature_row(
    start: datetime, inputs: SlotInputs, interval_minutes: int = 15
) -> dict[str, Any]:
    """Return the feature dict of one slot: its time features plus its inputs.

    The single definition of every model feature, used for training rows and
    for prediction rows alike. A derived feature is None when any input it
    needs is unknown.

    Args:
        start: Timezone-aware slot start in the price region's local time.
        inputs: The slot's raw inputs.
        interval_minutes: Slot length.

    Returns:
        ``slot_time_features(start)`` plus every non-time feature.
    """
    wind = inputs.wind_speed
    consumption = inputs.consumption
    offshore, onshore = inputs.wind_offshore, inputs.wind_onshore
    net_demand = None
    wind_share = None
    if consumption is not None and offshore is not None and onshore is not None:
        if inputs.solar_generation is not None:
            net_demand = consumption - inputs.solar_generation - offshore - onshore
        if consumption > 1e-9:
            wind_share = (offshore + onshore) / consumption

    return slot_time_features(start, interval_minutes) | {
        "wind_speed_mean": wind,
        "wind_power_estimate": wind_power_curve(wind) if wind is not None else None,
        "wind_direction": inputs.wind_direction,
        "cloud_coverage": inputs.cloud_coverage,
        "humidity": inputs.humidity,
        "temperature": inputs.temperature,
        "consumption_forecast": consumption,
        "solar_generation": inputs.solar_generation,
        "wind_offshore": offshore,
        "wind_onshore": onshore,
        "net_demand": net_demand,
        "wind_share": wind_share,
    }


class FeatureMixin(PredictorBase):
    """Feature extraction and engineering methods.

    Designed to be mixed into SpotPricePredictor — all attributes
    referenced via self (e.g. self.predictions) are provided by the
    owning class.
    """

    def _generate_time_features(
        self,
        forecast_days: int,
        interval_minutes: int,
        known_data_end_time: datetime | None = None,
    ) -> list[dict]:
        """Generate time-based features for predictions.

        Predictions start at the current slot, or at known_data_end_time
        when confirmed prices reach further (see first_prediction_slot), to
        avoid predicting prices we already have confirmed data for.

        Internal arithmetic is in UTC; the emitted ``start``/``end``
        timestamps and time-of-day fields are converted to the local time
        zone so they match the price source and HA's clock.
        """
        features = []
        start_time = first_prediction_slot(
            dt_util.utcnow(), known_data_end_time, interval_minutes
        )

        intervals_per_day = (24 * 60) // interval_minutes

        for day in range(forecast_days):
            for interval in range(intervals_per_day):
                dt = start_time + timedelta(
                    days=day, minutes=interval * interval_minutes
                )

                # Convert to local time for display and feature extraction
                features.append(
                    slot_time_features(dt_util.as_local(dt), interval_minutes)
                )

        return features

    def _combine_features(
        self, time_features: list[dict], weather_data: dict
    ) -> list[dict]:
        """Build the prediction feature rows from live forecasts and prognoses.

        Each slot is matched on its UTC hour to the hourly weather forecast
        (``weather_forecast``: datetime, temperature, wind_speed in m/s,
        wind_bearing, cloud_coverage, humidity), the Nordpool consumption
        prognosis (``consumption_prognosis``: UTC hour → MW) and the Nordpool
        production prognosis at the start of that hour
        (``production_prognosis``: deliveryStart, solar, wind_offshore,
        wind_onshore), the resolution training reads from storage. A slot
        outside a forecast's horizon gets None for those inputs, not a
        default or the current observation.
        """
        tz = self.tz
        weather_by_hour: dict[int, dict] = {}
        for entry in weather_data.get("weather_forecast") or []:
            key = hour_epoch(entry.get("datetime", ""), tz)
            if key is not None:
                weather_by_hour.setdefault(key, entry)
        consumption_by_hour: dict[int, Any] = {}
        consumption = weather_data.get("consumption_prognosis") or {}
        for timestamp, volume in consumption.items():
            key = hour_epoch(timestamp, tz)
            if key is not None:
                consumption_by_hour.setdefault(key, volume)
        production_by_start: dict[int, dict] = {}
        for entry in weather_data.get("production_prognosis") or []:
            key = utc_epoch(entry.get("deliveryStart", ""), tz, 1)
            if key is not None:
                production_by_start.setdefault(key, entry)

        combined = []
        for tf in time_features:
            start = datetime.fromisoformat(tf["start"])
            hour = floor_epoch(start, tz, HOUR_SECONDS)
            weather = weather_by_hour.get(hour, {})
            production = production_by_start.get(hour, {})
            inputs = SlotInputs(
                temperature=optional_float(weather.get("temperature")),
                wind_speed=optional_float(weather.get("wind_speed")),
                wind_direction=optional_float(weather.get("wind_bearing")),
                cloud_coverage=optional_float(weather.get("cloud_coverage")),
                humidity=optional_float(weather.get("humidity")),
                consumption=optional_float(consumption_by_hour.get(hour)),
                solar_generation=optional_float(production.get("solar")),
                wind_offshore=optional_float(production.get("wind_offshore")),
                wind_onshore=optional_float(production.get("wind_onshore")),
            )
            combined.append(build_feature_row(start, inputs))
        return combined
