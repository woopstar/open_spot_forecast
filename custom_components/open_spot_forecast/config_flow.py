"""Config flow for Open Spot Forecast."""

import logging
from typing import Any

import voluptuous as vol

from homeassistant import config_entries
from homeassistant.config_entries import ConfigFlowResult
from homeassistant.core import callback
from homeassistant.helpers.selector import (
    SelectSelector,
    SelectSelectorConfig,
    SelectSelectorMode,
    TextSelector,
    TextSelectorConfig,
    TextSelectorType,
    selector,
)

from .const import (
    CONF_CURRENCY,
    CONF_ENABLE_ML_PREDICTION,
    CONF_ENTSOE_API_KEY,
    CONF_PRECISION,
    CONF_PREDICTION_HOURS,
    CONF_PRICE_SOURCE,
    CONF_PRICE_TYPE,
    CONF_REGION,
    CONF_SOLAR_FORECAST_SENSOR,
    CONF_SOLAR_POWER_SENSOR,
    CONF_SPOT_PRICE_SENSOR,
    CONF_SPOT_PRICE_TOMORROW_SENSOR,
    CONF_STROMLIGNING_SENSOR,
    CONF_STROMLIGNING_TOMORROW_SENSOR,
    CONF_TEMPERATURE_SENSOR,
    CONF_TRAINING_DAYS,
    CONF_VAT,
    CONF_WIND_DIRECTION_SENSOR,
    CONF_WIND_SPEED_SENSOR,
    DEFAULT_CURRENCY,
    DEFAULT_PRECISION,
    DEFAULT_PREDICTION_HOURS,
    DEFAULT_PRICE_SOURCE,
    DEFAULT_PRICE_TYPE,
    DEFAULT_REGION,
    DEFAULT_SPOT_PRICE_SENSOR,
    DEFAULT_SPOT_PRICE_TOMORROW_SENSOR,
    DEFAULT_TRAINING_DAYS,
    DEFAULT_VAT,
    DOMAIN,
    PREDICTION_HOURS_OPTIONS,
    PRICE_SOURCE_STROMLIGNING,
    PRICE_SOURCES,
    REGIONS,
    STROMLIGNING_REGIONS,
    TRAINING_DAYS_OPTIONS,
)

_LOGGER = logging.getLogger(__name__)


def price_regions() -> list[str]:
    """Return the regions a price source covers (energy-charts or ENTSO-E)."""
    return sorted(
        region
        for region, zone in REGIONS.items()
        if zone.get("energy_charts") or zone.get("entsoe")
    )


def validate_price_source(user_input: dict[str, Any]) -> str | None:
    """Return the error key for an unusable region/price source, or None."""
    region = user_input.get(CONF_REGION)
    if region not in price_regions():
        return "invalid_region"
    source = user_input.get(CONF_PRICE_SOURCE, DEFAULT_PRICE_SOURCE)
    if source == PRICE_SOURCE_STROMLIGNING and region not in STROMLIGNING_REGIONS:
        return "stromligning_region"
    if not REGIONS[region].get("energy_charts") and not user_input.get(
        CONF_ENTSOE_API_KEY
    ):
        return "entsoe_key_required"
    return None


class OpenSpotForecastConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Config flow for Open Spot Forecast."""

    VERSION = 1
    CONNECTION_CLASS = config_entries.CONN_CLASS_CLOUD_POLL

    def __init__(self):
        """Initialize the config flow."""
        self._errors = {}
        self._data = {}

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Handle the initial step - basic settings."""
        self._errors = {}

        if user_input is not None:
            error = validate_price_source(user_input)
            if error:
                self._errors["base"] = error
            else:
                # Store data and move to step 2
                self._data.update(user_input)
                return await self.async_step_sensors()

        # Build the form for step 1: only regions with a price source
        regions_list = price_regions()
        currencies = sorted({str(r["currency"]) for r in REGIONS.values()})
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
                vol.Required(
                    CONF_PRICE_SOURCE, default=DEFAULT_PRICE_SOURCE
                ): SelectSelector(
                    SelectSelectorConfig(
                        options=list(PRICE_SOURCES),
                        mode=SelectSelectorMode.LIST,
                        translation_key=CONF_PRICE_SOURCE,
                    )
                ),
                vol.Optional(CONF_ENTSOE_API_KEY): TextSelector(
                    TextSelectorConfig(type=TextSelectorType.PASSWORD)
                ),
            }
        )

        return self.async_show_form(
            step_id="user",
            data_schema=data_schema,
            errors=self._errors,
        )

    async def async_step_sensors(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
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
                    CONF_SPOT_PRICE_SENSOR,
                    default=self._data.get(
                        CONF_SPOT_PRICE_SENSOR, DEFAULT_SPOT_PRICE_SENSOR
                    ),
                ): selector({"entity": {"domain": "sensor"}}),
                vol.Optional(
                    CONF_SPOT_PRICE_TOMORROW_SENSOR,
                    default=self._data.get(
                        CONF_SPOT_PRICE_TOMORROW_SENSOR,
                        DEFAULT_SPOT_PRICE_TOMORROW_SENSOR,
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
                vol.Optional(
                    CONF_PREDICTION_HOURS,
                    default=self._data.get(
                        CONF_PREDICTION_HOURS, DEFAULT_PREDICTION_HOURS
                    ),
                ): vol.In(PREDICTION_HOURS_OPTIONS),
                vol.Optional(
                    CONF_TRAINING_DAYS,
                    default=self._data.get(CONF_TRAINING_DAYS, DEFAULT_TRAINING_DAYS),
                ): vol.In(TRAINING_DAYS_OPTIONS),
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


class OpenSpotForecastOptionsFlow(config_entries.OptionsFlow):
    """Options flow for Open Spot Forecast."""

    def __init__(self):
        """Initialize the options flow."""
        self._errors = {}

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
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
                    CONF_PREDICTION_HOURS,
                    default=self.config_entry.options.get(
                        CONF_PREDICTION_HOURS,
                        self.config_entry.data.get(
                            CONF_PREDICTION_HOURS, DEFAULT_PREDICTION_HOURS
                        ),
                    ),
                ): vol.In(PREDICTION_HOURS_OPTIONS),
                vol.Optional(
                    CONF_TRAINING_DAYS,
                    default=self.config_entry.options.get(
                        CONF_TRAINING_DAYS,
                        self.config_entry.data.get(
                            CONF_TRAINING_DAYS, DEFAULT_TRAINING_DAYS
                        ),
                    ),
                ): vol.In(TRAINING_DAYS_OPTIONS),
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
                    CONF_SPOT_PRICE_SENSOR,
                    default=self.config_entry.options.get(
                        CONF_SPOT_PRICE_SENSOR,
                        self.config_entry.data.get(
                            CONF_SPOT_PRICE_SENSOR, DEFAULT_SPOT_PRICE_SENSOR
                        ),
                    ),
                ): selector({"entity": {"domain": "sensor"}}),
                vol.Optional(
                    CONF_SPOT_PRICE_TOMORROW_SENSOR,
                    default=self.config_entry.options.get(
                        CONF_SPOT_PRICE_TOMORROW_SENSOR,
                        self.config_entry.data.get(
                            CONF_SPOT_PRICE_TOMORROW_SENSOR,
                            DEFAULT_SPOT_PRICE_TOMORROW_SENSOR,
                        ),
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
