"""Config flow for Open Spot Forecast."""

import logging
from typing import Any

import voluptuous as vol
from homeassistant import config_entries
from homeassistant.core import callback
from homeassistant.helpers.selector import selector

from .const import (
    CONF_CURRENCY,
    CONF_ENABLE_ML_PREDICTION,
    CONF_PRECISION,
    CONF_PRICE_TYPE,
    CONF_REGION,
    CONF_SOLAR_FORECAST_SENSOR,
    CONF_SOLAR_POWER_SENSOR,
    CONF_STROMLIGNING_SENSOR,
    CONF_STROMLIGNING_TOMORROW_SENSOR,
    CONF_TEMPERATURE_SENSOR,
    CONF_VAT,
    CONF_WIND_DIRECTION_SENSOR,
    CONF_WIND_SPEED_SENSOR,
    DEFAULT_CURRENCY,
    DEFAULT_PRECISION,
    DEFAULT_PRICE_TYPE,
    DEFAULT_REGION,
    DEFAULT_VAT,
    DOMAIN,
    REGIONS,
)

_LOGGER = logging.getLogger(__name__)


class OpenSpotForecastConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Config flow for Open Spot Forecast."""

    VERSION = 1
    CONNECTION_CLASS = config_entries.CONN_CLASS_CLOUD_POLL

    def __init__(self):
        """Initialize the config flow."""
        self._errors = {}
        self._data = {}

    async def async_step_user(self, user_input: dict[str, Any] | None = None):
        """Handle the initial step - basic settings."""
        self._errors = {}

        if user_input is not None:
            # Validate region
            region = user_input.get(CONF_REGION)
            if region not in REGIONS:
                self._errors["base"] = "invalid_region"
            else:
                # Store data and move to step 2
                self._data.update(user_input)
                return await self.async_step_sensors()

        # Build the form for step 1
        regions_list = sorted(REGIONS.keys())
        currencies = sorted(list(set(r["currency"] for r in REGIONS.values())))
        price_types = ["kWh", "MWh", "Wh"]

        data_schema = vol.Schema(
            {
                vol.Required(CONF_REGION, default=DEFAULT_REGION): vol.In(regions_list),
                vol.Required(CONF_CURRENCY, default=DEFAULT_CURRENCY): vol.In(
                    currencies
                ),
                vol.Required(CONF_VAT, default=DEFAULT_VAT): vol.Coerce(float),
                vol.Required(CONF_PRECISION, default=DEFAULT_PRECISION): vol.Coerce(
                    int
                ),
                vol.Required(CONF_PRICE_TYPE, default=DEFAULT_PRICE_TYPE): vol.In(
                    price_types
                ),
            }
        )

        return self.async_show_form(
            step_id="user",
            data_schema=data_schema,
            errors=self._errors,
        )

    async def async_step_sensors(self, user_input: dict[str, Any] | None = None):
        """Handle the second step - sensor configuration."""
        self._errors = {}

        if user_input is not None:
            # Merge with data from step 1
            self._data.update(user_input)

            # Create the entry
            return self.async_create_entry(
                title=f"Open Spot Forecast {self._data.get(CONF_REGION)}",
                data=self._data,
            )

        # Build the form for step 2 with entity selectors
        data_schema = vol.Schema(
            {
                vol.Optional(
                    CONF_STROMLIGNING_SENSOR,
                    default=self._data.get(
                        CONF_STROMLIGNING_SENSOR,
                        "sensor.stromligning_current_price_vat",
                    ),
                ): selector({"entity": {"domain": "sensor"}}),
                vol.Optional(
                    CONF_STROMLIGNING_TOMORROW_SENSOR,
                    default=self._data.get(
                        CONF_STROMLIGNING_TOMORROW_SENSOR,
                        "binary_sensor.stromligning_tomorrow_spotprice_vat",
                    ),
                ): selector({"entity": {"domain": "binary_sensor"}}),
                vol.Optional(
                    CONF_WIND_SPEED_SENSOR,
                    default=self._data.get(
                        CONF_WIND_SPEED_SENSOR, "weather.forecast_mellemlokken_23"
                    ),
                ): selector({"entity": {"domain": ["sensor", "weather"]}}),
                vol.Optional(
                    CONF_WIND_DIRECTION_SENSOR,
                    default=self._data.get(
                        CONF_WIND_DIRECTION_SENSOR, "weather.forecast_mellemlokken_23"
                    ),
                ): selector({"entity": {"domain": ["sensor", "weather"]}}),
                vol.Optional(
                    CONF_SOLAR_POWER_SENSOR,
                    default=self._data.get(
                        CONF_SOLAR_POWER_SENSOR, "sensor.power_inverter_input_total"
                    ),
                ): selector({"entity": {"domain": "sensor"}}),
                vol.Optional(
                    CONF_SOLAR_FORECAST_SENSOR,
                    default=self._data.get(
                        CONF_SOLAR_FORECAST_SENSOR,
                        "sensor.solcast_pv_forecast_forecast_today",
                    ),
                ): selector({"entity": {"domain": "sensor"}}),
                vol.Optional(
                    CONF_TEMPERATURE_SENSOR,
                    default=self._data.get(
                        CONF_TEMPERATURE_SENSOR,
                        "sensor.metroair_330_outdoor_temperature",
                    ),
                ): selector({"entity": {"domain": ["sensor", "weather"]}}),
                vol.Optional(
                    CONF_ENABLE_ML_PREDICTION,
                    default=self._data.get(CONF_ENABLE_ML_PREDICTION, True),
                ): bool,
            }
        )

        return self.async_show_form(
            step_id="sensors",
            data_schema=data_schema,
            errors=self._errors,
            description_placeholders={
                "stromligning_info": "Real consumer prices with tariffs/VAT (recommended)",
                "weather_info": "Improves ML prediction accuracy",
            },
        )

    @staticmethod
    @callback
    def async_get_options_flow(config_entry):
        """Get the options flow."""
        return OpenSpotForecastOptionsFlow()
        return OpenSpotForecastOptionsFlow()


class OpenSpotForecastOptionsFlow(config_entries.OptionsFlow):
    """Options flow for Open Spot Forecast."""

    def __init__(self):
        """Initialize the options flow."""
        self._errors = {}

    async def async_step_init(self, user_input: dict[str, Any] | None = None):
        """Handle options step."""
        self._errors = {}

        if user_input is not None:
            return self.async_create_entry(
                title=self.config_entry.data.get(CONF_REGION, DEFAULT_REGION),
                data=user_input,
            )

        # Build the form with entity selectors
        data_schema = vol.Schema(
            {
                vol.Optional(
                    CONF_ENABLE_ML_PREDICTION,
                    default=self.config_entry.options.get(
                        CONF_ENABLE_ML_PREDICTION, True
                    ),
                ): bool,
                vol.Optional(
                    CONF_STROMLIGNING_SENSOR,
                    default=self.config_entry.options.get(
                        CONF_STROMLIGNING_SENSOR,
                        "sensor.stromligning_current_price_vat",
                    ),
                ): selector({"entity": {"domain": "sensor"}}),
                vol.Optional(
                    CONF_STROMLIGNING_TOMORROW_SENSOR,
                    default=self.config_entry.options.get(
                        CONF_STROMLIGNING_TOMORROW_SENSOR,
                        "binary_sensor.stromligning_tomorrow_spotprice_vat",
                    ),
                ): selector({"entity": {"domain": "binary_sensor"}}),
                vol.Optional(
                    CONF_WIND_SPEED_SENSOR,
                    default=self.config_entry.options.get(
                        CONF_WIND_SPEED_SENSOR, "weather.forecast_mellemlokken_23"
                    ),
                ): selector({"entity": {"domain": ["sensor", "weather"]}}),
                vol.Optional(
                    CONF_WIND_DIRECTION_SENSOR,
                    default=self.config_entry.options.get(
                        CONF_WIND_DIRECTION_SENSOR, "weather.forecast_mellemlokken_23"
                    ),
                ): selector({"entity": {"domain": ["sensor", "weather"]}}),
                vol.Optional(
                    CONF_SOLAR_POWER_SENSOR,
                    default=self.config_entry.options.get(
                        CONF_SOLAR_POWER_SENSOR, "sensor.power_inverter_input_total"
                    ),
                ): selector({"entity": {"domain": "sensor"}}),
                vol.Optional(
                    CONF_SOLAR_FORECAST_SENSOR,
                    default=self.config_entry.options.get(
                        CONF_SOLAR_FORECAST_SENSOR,
                        "sensor.solcast_pv_forecast_forecast_today",
                    ),
                ): selector({"entity": {"domain": "sensor"}}),
                vol.Optional(
                    CONF_TEMPERATURE_SENSOR,
                    default=self.config_entry.options.get(
                        CONF_TEMPERATURE_SENSOR,
                        "sensor.metroair_330_outdoor_temperature",
                    ),
                ): selector({"entity": {"domain": ["sensor", "weather"]}}),
                vol.Optional(
                    CONF_VAT,
                    default=self.config_entry.options.get(
                        CONF_VAT, self.config_entry.data.get(CONF_VAT, DEFAULT_VAT)
                    ),
                ): vol.Coerce(float),
                vol.Optional(
                    CONF_PRECISION,
                    default=self.config_entry.options.get(
                        CONF_PRECISION,
                        self.config_entry.data.get(CONF_PRECISION, DEFAULT_PRECISION),
                    ),
                ): vol.Coerce(int),
            }
        )

        return self.async_show_form(
            step_id="init",
            data_schema=data_schema,
            errors=self._errors,
        )
