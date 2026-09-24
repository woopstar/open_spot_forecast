"""Constants for Open Spot Forecast integration."""

DOMAIN = "open_spot_forecast"

# Configuration keys
CONF_REGION = "region"
CONF_CURRENCY = "currency"
CONF_ENABLE_ML_PREDICTION = "enable_ml_prediction"
CONF_VAT = "vat"
CONF_PRICE_TYPE = "price_type"
CONF_PRECISION = "precision"

# Weather sensor configuration keys
CONF_WIND_SPEED_SENSOR = "wind_speed_sensor"
CONF_WIND_DIRECTION_SENSOR = "wind_direction_sensor"
CONF_SOLAR_POWER_SENSOR = "solar_power_sensor"
CONF_SOLAR_FORECAST_SENSOR = "solar_forecast_sensor"
CONF_TEMPERATURE_SENSOR = "temperature_sensor"

# Stromligning sensor configuration
CONF_STROMLIGNING_SENSOR = "stromligning_sensor"
CONF_STROMLIGNING_TOMORROW_SENSOR = "stromligning_tomorrow_sensor"

# Defaults
DEFAULT_NAME = "Open Spot Forecast"
DEFAULT_REGION = "DK1"
DEFAULT_CURRENCY = "DKK"
DEFAULT_VAT = 0.25
DEFAULT_PRECISION = 3
DEFAULT_PRICE_TYPE = "kWh"

# Platforms
PLATFORMS = ["sensor", "binary_sensor"]

# Update signals
UPDATE_SIGNAL = f"{DOMAIN}_signal_update"
UPDATE_SIGNAL_FORECAST = f"{DOMAIN}_signal_forecast_update"

# Nordpool API - REMOVED (replaced by Stromligning + weather entity)
# DMI API - REMOVED (replaced by weather entity via weather.get_forecasts)

# Supported regions
REGIONS = {
    "DK1": {
        "currency": "DKK",
        "country": "Denmark",
        "vat": 0.25,
        "tz": "Europe/Copenhagen",
    },
    "DK2": {
        "currency": "DKK",
        "country": "Denmark",
        "vat": 0.25,
        "tz": "Europe/Copenhagen",
    },
    "SE3": {
        "currency": "SEK",
        "country": "Sweden",
        "vat": 0.25,
        "tz": "Europe/Stockholm",
    },
    "SE4": {
        "currency": "SEK",
        "country": "Sweden",
        "vat": 0.25,
        "tz": "Europe/Stockholm",
    },
    "NO2": {"currency": "NOK", "country": "Norway", "vat": 0.25, "tz": "Europe/Oslo"},
    "FI": {
        "currency": "EUR",
        "country": "Finland",
        "vat": 0.255,
        "tz": "Europe/Helsinki",
    },
    "EE": {
        "currency": "EUR",
        "country": "Estonia",
        "vat": 0.24,
        "tz": "Europe/Tallinn",
    },
    "LT": {
        "currency": "EUR",
        "country": "Lithuania",
        "vat": 0.21,
        "tz": "Europe/Vilnius",
    },
    "LV": {"currency": "EUR", "country": "Latvia", "vat": 0.21, "tz": "Europe/Riga"},
    "NL": {
        "currency": "EUR",
        "country": "Netherlands",
        "vat": 0.21,
        "tz": "Europe/Amsterdam",
    },
    "BE": {
        "currency": "EUR",
        "country": "Belgium",
        "vat": 0.06,
        "tz": "Europe/Brussels",
    },
    "FR": {"currency": "EUR", "country": "France", "vat": 0.055, "tz": "Europe/Paris"},
    "DE": {"currency": "EUR", "country": "Germany", "vat": 0.19, "tz": "Europe/Berlin"},
}

# Nordpool dataportal API (consumption and production prognoses)
NORDPOOL_API = "https://dataportal-api.nordpoolgroup.com/api"

# Nordpool's dataportal API sits behind Cloudflare bot protection. A browser-like
# User-Agent avoids being flagged as a script, which surfaces as HTTP 401/403.
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
)

# Status codes that indicate a transient failure worth retrying with backoff.
# 429 (rate limit) and 5xx (server errors) may resolve on retry. 401/403 are
# deliberately excluded: they signal an auth/permission/bot-block failure that
# will not succeed on retry, and retrying them with backoff previously stalled
# startup for minutes when the Nordpool API refused our requests.
RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})

# Maximum retry attempts (in addition to the initial request).
MAX_RETRIES = 3

# Initial backoff delay in seconds; doubles on each retry (1s, 2s, 4s).
RETRY_BASE_DELAY = 1.0

# Price conversion
PRICE_IN = {"kWh": 1000, "MWh": 1, "Wh": 1000000}

# Startup message
STARTUP = """
-------------------------------------------------------------------
Open Spot Forecast
Version: %s
This is a custom integration
If you have any issues, open an issue here:
https://github.com/woopstar/open_spot_forecast/issues
-------------------------------------------------------------------
"""
