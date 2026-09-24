"""Feature engineering for the spot price predictor."""

import logging
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np

from homeassistant.util import dt as dt_util

from ..time_slots import first_prediction_slot
from .base import PredictorBase

_LOGGER = logging.getLogger(__name__)

# The canonical 20-feature model input, in column order (docs/ml_documentation.md).
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
    "solar_radiation_mean",
    "solar_power_estimate",
    "price_mean",
    "temperature",
    "consumption_forecast",
    "solar_generation",
    "wind_offshore",
    "wind_onshore",
    "net_demand",
    "wind_share",
)

# Value for a feature missing from the feature dict; unlisted features default to 0.
_FEATURE_DEFAULTS: dict[str, float] = {"humidity": 50.0, "temperature": 15.0}


def _to_float(value: Any) -> float:
    """Coerce a feature value to float, mapping None and non-numerics to 0.0."""
    if value is None:
        return 0.0
    try:
        return float(value)
    except ValueError, TypeError:
        return 0.0


def build_feature_vector(feature: dict[str, Any]) -> list[float]:
    """Return the model input row for one slot's feature dict.

    Training, prediction, hyperparameter search and the dev backtest
    (``scripts/backtest.py``) all build rows here, so ``FEATURE_NAMES`` is
    the single source of truth for the column order.

    Args:
        feature: Feature dict for one slot, as produced by
            ``slot_time_features`` plus weather/price/Nordpool keys.

    Returns:
        20 floats in ``FEATURE_NAMES`` order.
    """
    return [
        _to_float(feature.get(name, _FEATURE_DEFAULTS.get(name, 0)))
        for name in FEATURE_NAMES
    ]


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


class FeatureMixin(PredictorBase):
    """Feature extraction and engineering methods.

    Designed to be mixed into SpotPricePredictor — all attributes
    referenced via self (e.g. self.predictions) are provided by the
    owning class.
    """

    def _extract_wind_features(self, weather_data: dict) -> dict:
        """Extract wind-related features from weather data."""
        wind_forecast = weather_data.get("wind_forecast", [])

        if wind_forecast:
            wind_speeds = [
                f.get("wind_speed", 0) for f in wind_forecast if f.get("wind_speed")
            ]
            if wind_speeds:
                wind_power = [self._wind_power_curve(v) for v in wind_speeds]
                return {
                    "wind_speed_mean": float(np.mean(wind_speeds)),
                    "wind_speed_max": float(np.max(wind_speeds)),
                    "wind_speed_std": float(np.std(wind_speeds)),
                    "wind_power_mean": float(np.mean(wind_power)),
                    "wind_power_max": float(np.max(wind_power)),
                    "wind_power_estimate": float(sum(wind_power)),
                    "wind_direction": float(weather_data.get("wind_direction", 0)),
                }

        # Fallback: use current wind speed from HA sensor (scalar)
        wind_speed = weather_data.get("wind_speed", 0)
        if isinstance(wind_speed, (int, float)) and wind_speed > 0:
            wind_power_scalar = self._wind_power_curve(wind_speed)
            return {
                "wind_speed_mean": float(wind_speed),
                "wind_speed_max": float(wind_speed),
                "wind_speed_std": 0.0,
                "wind_power_mean": float(wind_power_scalar),
                "wind_power_max": float(wind_power_scalar),
                "wind_power_estimate": float(wind_power_scalar),
                "wind_direction": float(weather_data.get("wind_direction", 0)),
            }

        return {
            "wind_speed_mean": 0,
            "wind_speed_max": 0,
            "wind_power_estimate": 0,
            "wind_direction": 0,
        }

    def _extract_solar_features(self, weather_data: dict) -> dict:
        """Extract solar-related features from weather data."""
        # Check both key names (solcast_forecast from HA sensor, solar_forecast from DMI)
        solar_forecast = weather_data.get(
            "solcast_forecast", weather_data.get("solar_forecast", [])
        )
        solar_power = weather_data.get("solar_power", 0)

        if not solar_forecast and not solar_power:
            return {"solar_radiation_mean": 0, "solar_power_estimate": 0}

        # Handle Solcast dict format
        if isinstance(solar_forecast, dict):
            detailed_hourly = solar_forecast.get("detailed_hourly", [])
            if detailed_hourly:
                radiation = [
                    h.get("pv_estimate", 0)
                    for h in detailed_hourly
                    if h.get("pv_estimate") is not None
                ]
                if radiation:
                    return {
                        "solar_radiation_mean": float(np.mean(radiation)),
                        "solar_radiation_max": float(np.max(radiation)),
                        "cloud_cover_mean": 0,
                        "solar_power_mean": float(np.mean(radiation)),
                        "solar_power_estimate": solar_forecast.get(
                            "estimate_today", float(sum(radiation))
                        ),
                    }

            estimate = solar_forecast.get("estimate_today", 0)
            return {
                "solar_radiation_mean": estimate / 24 if estimate else 0,
                "solar_radiation_max": estimate / 12 if estimate else 0,
                "cloud_cover_mean": 0,
                "solar_power_mean": estimate / 24 if estimate else 0,
                "solar_power_estimate": float(estimate),
            }

        # Handle list format (original DMI format)
        if isinstance(solar_forecast, list) and solar_forecast:
            radiation = [
                f.get("radiation", 0) for f in solar_forecast if f.get("radiation")
            ]
            cloud_cover = [
                f.get("cloud_cover", 0) for f in solar_forecast if f.get("cloud_cover")
            ]

            if not radiation:
                return {"solar_radiation_mean": 0, "solar_power_estimate": 0}

            avg_cloud = float(np.mean(cloud_cover)) if cloud_cover else 0
            solar_power_list = [r * 0.15 * (1 - avg_cloud * 0.7) for r in radiation]

            return {
                "solar_radiation_mean": float(np.mean(radiation)),
                "solar_radiation_max": float(np.max(radiation)),
                "cloud_cover_mean": avg_cloud,
                "solar_power_mean": float(np.mean(solar_power_list)),
                "solar_power_estimate": float(sum(solar_power_list)),
            }

        # Fallback to solar_power sensor
        if solar_power:
            return {
                "solar_radiation_mean": float(solar_power),
                "solar_radiation_max": float(solar_power),
                "cloud_cover_mean": 0,
                "solar_power_mean": float(solar_power),
                "solar_power_estimate": float(solar_power),
            }

        return {"solar_radiation_mean": 0, "solar_power_estimate": 0}

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
        self,
        wind_features: dict,
        solar_features: dict,
        time_features: list[dict],
        historical_prices: list[float],
        weather_data: dict,
    ) -> list[dict]:
        """Combine all features into a single feature set.

        When hourly weather forecast data is available in weather_data
        (key: 'weather_forecast'), each time slot gets its own matching
        wind/temperature/humidity/cloud values instead of sharing a
        single scalar.

        Nordpool prognoses (consumption, production) are matched per slot
        when available in weather_data keys:
          'consumption_prognosis' — hourly demand (MW)
          'production_prognosis'  — 15-min generation per type (MW)
        """
        combined = []
        weather_forecast = weather_data.get("weather_forecast")
        consumption_data = weather_data.get("consumption_prognosis", {})
        production_data = weather_data.get("production_prognosis", [])

        # Build production lookup by timestamp
        prod_by_ts: dict[str, dict] = {}
        for entry in production_data:
            ts = entry.get("deliveryStart", "")
            if ts:
                prod_by_ts[ts] = entry

        # Build a lookup: hour-key -> forecast entry
        forecast_by_hour: dict[str, dict] = {}
        if weather_forecast:
            for entry in weather_forecast:
                dt_str = entry.get("datetime", "")
                if dt_str:
                    hour_key = dt_str[:13] if len(dt_str) >= 13 else dt_str
                    forecast_by_hour[hour_key] = entry

        price_stats = {
            "price_mean": float(np.mean(historical_prices)) if historical_prices else 0,
            "price_std": float(np.std(historical_prices)) if historical_prices else 0,
            "price_min": float(np.min(historical_prices)) if historical_prices else 0,
            "price_max": float(np.max(historical_prices)) if historical_prices else 0,
        }

        default_temp = weather_data.get("temperature") or 15.0
        default_wind = float(weather_data.get("wind_speed") or 0)
        default_wind_dir = float(weather_data.get("wind_direction") or 0)

        for tf in time_features:
            start = tf.get("start", "")

            slot_temp = default_temp
            slot_wind = default_wind
            slot_wind_dir = default_wind_dir
            slot_cloud = 0.0
            slot_humidity = 50.0

            if start and forecast_by_hour:
                hour_key = start[:13] if len(start) >= 13 else start
                fc = forecast_by_hour.get(hour_key)
                if fc:
                    slot_temp = float(fc.get("temperature", slot_temp))
                    slot_wind = float(fc.get("wind_speed", slot_wind))
                    slot_wind_dir = float(fc.get("wind_bearing", slot_wind_dir))
                    slot_cloud = float(fc.get("cloud_coverage", 0))
                    slot_humidity = float(fc.get("humidity", 50))

            slot_wind_power = self._wind_power_curve(slot_wind)

            # --- Nordpool prognoses: match consumption (hourly) and production (15-min) ---
            consumption = 0.0
            solar_gen = 0.0
            wind_off = 0.0
            wind_on = 0.0

            # Nordpool returns UTC timestamps; convert the local prediction
            # start back to UTC so the lookups match.
            start_dt = dt_util.parse_datetime(start) if start else None
            utc_start = dt_util.as_utc(start_dt) if start_dt else None

            if utc_start and consumption_data:
                hour_key = utc_start.strftime("%Y-%m-%dT%H:00:00Z")
                consumption = consumption_data.get(hour_key, 0.0)

            if utc_start and prod_by_ts:
                prod = prod_by_ts.get(utc_start.strftime("%Y-%m-%dT%H:%M:%SZ"))
                if prod:
                    solar_gen = prod.get("solar", 0.0)
                    wind_off = prod.get("wind_offshore", 0.0)
                    wind_on = prod.get("wind_onshore", 0.0)

            net_demand = consumption - solar_gen - wind_off - wind_on
            wind_share = (wind_off + wind_on) / consumption if consumption > 0 else 0.0

            feature = {
                **tf,
                **wind_features,
                **solar_features,
                **price_stats,
                "wind_speed_mean": slot_wind,
                "wind_power_estimate": slot_wind_power,
                "wind_direction": slot_wind_dir,
                "cloud_coverage": slot_cloud,
                "humidity": slot_humidity,
                "temperature": slot_temp,
                "consumption_forecast": consumption,
                "solar_generation": solar_gen,
                "wind_offshore": wind_off,
                "wind_onshore": wind_on,
                "net_demand": net_demand,
                "wind_share": wind_share,
            }
            combined.append(feature)

        return combined

    def _wind_power_curve(self, wind_speed: float) -> float:
        """Simplified wind turbine power curve."""
        if wind_speed < 3:
            return 0.0
        elif wind_speed < 12:
            return ((wind_speed - 3) / 9) ** 3
        elif wind_speed < 25:
            return 1.0
        else:
            return 0.0
