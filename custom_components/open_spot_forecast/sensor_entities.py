"""The external entities a config entry reads (``SensorEntities``)."""

from __future__ import annotations

from dataclasses import dataclass, field

from homeassistant.config_entries import ConfigEntry

from .const import (
    CONF_CONSUMER_PRICE_SENSOR,
    CONF_CONSUMER_PRICE_TOMORROW_SENSOR,
    CONF_EXTERNAL_FORECAST_SENSORS,
    CONF_SOLAR_FORECAST_SENSOR,
    CONF_SOLAR_POWER_SENSOR,
    CONF_SPOT_PRICE_SENSOR,
    CONF_SPOT_PRICE_TOMORROW_SENSOR,
    CONF_TEMPERATURE_SENSOR,
    CONF_WIND_DIRECTION_SENSOR,
    CONF_WIND_SPEED_SENSOR,
    DEFAULT_CONSUMER_PRICE_SENSOR,
    DEFAULT_CONSUMER_PRICE_TOMORROW_SENSOR,
    DEFAULT_SPOT_PRICE_SENSOR,
    DEFAULT_SPOT_PRICE_TOMORROW_SENSOR,
)


def external_forecast_sensors(entry: ConfigEntry) -> tuple[str, ...]:
    """Return the sensors whose forecasts are recorded (#120), none by default.

    Options take precedence over the entry's initial data, so the list can
    be emptied after setup.
    """
    sensors = entry.options.get(
        CONF_EXTERNAL_FORECAST_SENSORS,
        entry.data.get(CONF_EXTERNAL_FORECAST_SENSORS),
    )
    if isinstance(sensors, str):
        return (sensors,)
    return tuple(sensors or ())


@dataclass(frozen=True, slots=True)
class SensorEntities:
    """The external entities a config entry reads.

    Options (reconfiguration) take precedence over the entry's initial data.
    """

    # Stromligning's consumer price excl. VAT, today and tomorrow (#107)
    stromligning: str | None
    stromligning_tomorrow: str | None
    # Raw spot price excl. VAT and tariffs: what the ML model learns (#16)
    spot_price: str | None
    spot_price_tomorrow: str | None
    wind_speed: str | None
    wind_direction: str | None
    solar_power: str | None
    solar_forecast: str | None
    temperature: str | None
    # Sensors whose forecast is recorded next to the model's and scored (#120)
    external_forecasts: tuple[str, ...] = field(default=(), kw_only=True)

    @classmethod
    def from_entry(cls, entry: ConfigEntry) -> SensorEntities:
        """Read the configured entity ids from a config entry."""

        def option(key: str, default: str | None = None) -> str | None:
            value: str | None = entry.options.get(key, entry.data.get(key, default))
            return value

        return cls(
            stromligning=option(
                CONF_CONSUMER_PRICE_SENSOR, DEFAULT_CONSUMER_PRICE_SENSOR
            ),
            stromligning_tomorrow=option(
                CONF_CONSUMER_PRICE_TOMORROW_SENSOR,
                DEFAULT_CONSUMER_PRICE_TOMORROW_SENSOR,
            ),
            spot_price=option(CONF_SPOT_PRICE_SENSOR, DEFAULT_SPOT_PRICE_SENSOR),
            spot_price_tomorrow=option(
                CONF_SPOT_PRICE_TOMORROW_SENSOR, DEFAULT_SPOT_PRICE_TOMORROW_SENSOR
            ),
            wind_speed=option(CONF_WIND_SPEED_SENSOR),
            wind_direction=option(CONF_WIND_DIRECTION_SENSOR),
            solar_power=option(CONF_SOLAR_POWER_SENSOR),
            solar_forecast=option(CONF_SOLAR_FORECAST_SENSOR),
            temperature=option(CONF_TEMPERATURE_SENSOR),
            external_forecasts=external_forecast_sensors(entry),
        )

    def sensor_config(self) -> dict[str, str | None]:
        """Return the ``api_data["sensor_config"]`` dict the weather reader uses."""
        return {
            "stromligning_sensor": self.stromligning,
            "wind_speed_sensor": self.wind_speed,
            "wind_direction_sensor": self.wind_direction,
            "solar_power_sensor": self.solar_power,
            "solar_forecast_sensor": self.solar_forecast,
            "temperature_sensor": self.temperature,
        }

    @property
    def has_weather(self) -> bool:
        """Return whether any weather or solar entity is configured."""
        return any(
            [
                self.wind_speed,
                self.wind_direction,
                self.solar_power,
                self.solar_forecast,
                self.temperature,
            ]
        )
