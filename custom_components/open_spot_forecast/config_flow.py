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
    ATTRIBUTE_FORMATS,
    CONF_ATTRIBUTE_FORMAT,
    CONF_CONSUMER_PRICE_SENSOR,
    CONF_CONSUMER_PRICE_TOMORROW_SENSOR,
    CONF_CROSS_BORDER,
    CONF_CURRENCY,
    CONF_ENABLE_ML_PREDICTION,
    CONF_ENTSOE_API_KEY,
    CONF_HOURLY_AVERAGE,
    CONF_INCLUDE_KNOWN_PRICES,
    CONF_PRECISION,
    CONF_PREDICTION_HOURS,
    CONF_PRICE_SOURCE,
    CONF_PRICE_TYPE,
    CONF_REGION,
    CONF_SOLAR_FORECAST_SENSOR,
    CONF_SOLAR_POWER_SENSOR,
    CONF_SPOT_PRICE_SENSOR,
    CONF_SPOT_PRICE_TOMORROW_SENSOR,
    CONF_SURCHARGE,
    CONF_TEMPERATURE_SENSOR,
    CONF_TRAINING_DAYS,
    CONF_VAT,
    CONF_WIND_DIRECTION_SENSOR,
    CONF_WIND_SPEED_SENSOR,
    DEFAULT_ATTRIBUTE_FORMAT,
    DEFAULT_CONSUMER_PRICE_SENSOR,
    DEFAULT_CONSUMER_PRICE_TOMORROW_SENSOR,
    DEFAULT_CROSS_BORDER,
    DEFAULT_CURRENCY,
    DEFAULT_HOURLY_AVERAGE,
    DEFAULT_INCLUDE_KNOWN_PRICES,
    DEFAULT_PRECISION,
    DEFAULT_PREDICTION_HOURS,
    DEFAULT_PRICE_SOURCE,
    DEFAULT_PRICE_TYPE,
    DEFAULT_REGION,
    DEFAULT_SPOT_PRICE_SENSOR,
    DEFAULT_SPOT_PRICE_TOMORROW_SENSOR,
    DEFAULT_SURCHARGE,
    DEFAULT_TRAINING_DAYS,
    DEFAULT_VAT,
    DOMAIN,
    NEIGHBOURS,
    PREDICTION_HOURS_OPTIONS,
    PRICE_SOURCE_STROMLIGNING,
    PRICE_SOURCES,
    PRICE_TYPES,
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


def training_days_selector() -> SelectSelector:
    """Return the training-window picker, with string options.

    ``vol.In`` over ints renders as a radio list below six options, and the
    frontend's radio group compares its (string) value to the (int) option
    values, so the saved choice was never shown as selected.
    """
    return SelectSelector(
        SelectSelectorConfig(
            options=[str(days) for days in TRAINING_DAYS_OPTIONS],
            mode=SelectSelectorMode.LIST,
        )
    )


def normalize_training_days(user_input: dict[str, Any]) -> dict[str, Any]:
    """Store the picked training window as an int, as before the selector."""
    if CONF_TRAINING_DAYS in user_input:
        user_input[CONF_TRAINING_DAYS] = int(user_input[CONF_TRAINING_DAYS])
    return user_input


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
                    list(PRICE_TYPES)
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
            self._data.update(normalize_training_days(user_input))

            # Create the entry
            return self.async_create_entry(
                title=f"Open Spot Forecast {self._data.get(CONF_REGION)}",
                data=self._data,
            )

        # Build the form for step 2 with entity selectors
        data_schema = vol.Schema(
            {
                vol.Optional(
                    CONF_CONSUMER_PRICE_SENSOR,
                    default=self._data.get(
                        CONF_CONSUMER_PRICE_SENSOR, DEFAULT_CONSUMER_PRICE_SENSOR
                    ),
                ): selector({"entity": {"domain": "sensor"}}),
                vol.Optional(
                    CONF_CONSUMER_PRICE_TOMORROW_SENSOR,
                    default=self._data.get(
                        CONF_CONSUMER_PRICE_TOMORROW_SENSOR,
                        DEFAULT_CONSUMER_PRICE_TOMORROW_SENSOR,
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
                    default=str(
                        self._data.get(CONF_TRAINING_DAYS, DEFAULT_TRAINING_DAYS)
                    ),
                ): training_days_selector(),
            }
        )

        return self.async_show_form(
            step_id="sensors",
            data_schema=data_schema,
            errors=self._errors,
            description_placeholders={
                "stromligning_info": "Consumer prices excl. VAT, with tariffs (recommended)",
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
                data=normalize_training_days(user_input),
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
                # Compact attributes fit up to 168 hours (#38)
                vol.Optional(
                    CONF_ATTRIBUTE_FORMAT,
                    default=self.config_entry.options.get(
                        CONF_ATTRIBUTE_FORMAT, DEFAULT_ATTRIBUTE_FORMAT
                    ),
                ): SelectSelector(
                    SelectSelectorConfig(
                        options=list(ATTRIBUTE_FORMATS),
                        mode=SelectSelectorMode.LIST,
                        translation_key=CONF_ATTRIBUTE_FORMAT,
                    )
                ),
                # Confirmed prices before the forecast (#40)
                vol.Optional(
                    CONF_INCLUDE_KNOWN_PRICES,
                    default=self.config_entry.options.get(
                        CONF_INCLUDE_KNOWN_PRICES, DEFAULT_INCLUDE_KNOWN_PRICES
                    ),
                ): bool,
                vol.Optional(
                    CONF_TRAINING_DAYS,
                    default=str(
                        self.config_entry.options.get(
                            CONF_TRAINING_DAYS,
                            self.config_entry.data.get(
                                CONF_TRAINING_DAYS, DEFAULT_TRAINING_DAYS
                            ),
                        )
                    ),
                ): training_days_selector(),
                # Two-stage cross-border model (#29), for regions with neighbours
                **(
                    {
                        vol.Optional(
                            CONF_CROSS_BORDER,
                            default=self.config_entry.options.get(
                                CONF_CROSS_BORDER, DEFAULT_CROSS_BORDER
                            ),
                        ): bool
                    }
                    if self.config_entry.data.get(CONF_REGION) in NEIGHBOURS
                    else {}
                ),
                # Added after setup too: enables ENTSO-E's load forecast (#30)
                vol.Optional(
                    CONF_ENTSOE_API_KEY,
                    description={
                        "suggested_value": self.config_entry.options.get(
                            CONF_ENTSOE_API_KEY,
                            self.config_entry.data.get(CONF_ENTSOE_API_KEY),
                        )
                    },
                ): TextSelector(TextSelectorConfig(type=TextSelectorType.PASSWORD)),
                vol.Optional(
                    CONF_CONSUMER_PRICE_SENSOR,
                    default=self.config_entry.options.get(
                        CONF_CONSUMER_PRICE_SENSOR,
                        self.config_entry.data.get(
                            CONF_CONSUMER_PRICE_SENSOR, DEFAULT_CONSUMER_PRICE_SENSOR
                        ),
                    ),
                ): selector({"entity": {"domain": "sensor"}}),
                vol.Optional(
                    CONF_CONSUMER_PRICE_TOMORROW_SENSOR,
                    default=self.config_entry.options.get(
                        CONF_CONSUMER_PRICE_TOMORROW_SENSOR,
                        self.config_entry.data.get(
                            CONF_CONSUMER_PRICE_TOMORROW_SENSOR,
                            DEFAULT_CONSUMER_PRICE_TOMORROW_SENSOR,
                        ),
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
                # Price output (#39): total = (spot + surcharge) × (1 + VAT)
                vol.Optional(
                    CONF_SURCHARGE,
                    default=self.config_entry.options.get(
                        CONF_SURCHARGE, DEFAULT_SURCHARGE
                    ),
                ): vol.Coerce(float),
                vol.Optional(
                    CONF_HOURLY_AVERAGE,
                    default=self.config_entry.options.get(
                        CONF_HOURLY_AVERAGE, DEFAULT_HOURLY_AVERAGE
                    ),
                ): bool,
            }
        )

        return self.async_show_form(
            step_id="init",
            data_schema=data_schema,
            errors=self._errors,
        )
