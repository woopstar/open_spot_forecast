"""Constants for Open Spot Forecast integration."""

DOMAIN = "open_spot_forecast"

# Configuration keys
CONF_REGION = "region"
CONF_CURRENCY = "currency"
CONF_ENABLE_ML_PREDICTION = "enable_ml_prediction"
CONF_VAT = "vat"
CONF_PRICE_TYPE = "price_type"
CONF_PRECISION = "precision"
CONF_PREDICTION_HOURS = "prediction_hours"
# Days of price history the model trains on (#24); history older than this
# (plus a margin) is pruned, and missing days are backfilled
CONF_TRAINING_DAYS = "training_days"

# Weather sensor configuration keys
CONF_WIND_SPEED_SENSOR = "wind_speed_sensor"
CONF_WIND_DIRECTION_SENSOR = "wind_direction_sensor"
CONF_SOLAR_POWER_SENSOR = "solar_power_sensor"
CONF_SOLAR_FORECAST_SENSOR = "solar_forecast_sensor"
CONF_TEMPERATURE_SENSOR = "temperature_sensor"

# Stromligning sensor configuration
CONF_STROMLIGNING_SENSOR = "stromligning_sensor"
CONF_STROMLIGNING_TOMORROW_SENSOR = "stromligning_tomorrow_sensor"

# Where prices come from (#27): Stromligning's sensors (DK1/DK2 only), or the
# day-ahead auction prices from energy-charts.info, with the ENTSO-E
# Transparency Platform as fallback when an API key is configured
CONF_PRICE_SOURCE = "price_source"
CONF_ENTSOE_API_KEY = "entsoe_api_key"
PRICE_SOURCE_STROMLIGNING = "stromligning"
PRICE_SOURCE_DAYAHEAD = "dayahead"
PRICE_SOURCES = (PRICE_SOURCE_STROMLIGNING, PRICE_SOURCE_DAYAHEAD)
DEFAULT_PRICE_SOURCE = PRICE_SOURCE_STROMLIGNING
STROMLIGNING_REGIONS = ("DK1", "DK2")

# Raw day-ahead spot price, excl. VAT and tariffs: the ML model's training and
# prediction target (#16). VAT is applied once, in the sensor layer.
CONF_SPOT_PRICE_SENSOR = "spot_price_sensor"
CONF_SPOT_PRICE_TOMORROW_SENSOR = "spot_price_tomorrow_sensor"
DEFAULT_SPOT_PRICE_SENSOR = "sensor.stromligning_spotprice_ex_vat"
DEFAULT_SPOT_PRICE_TOMORROW_SENSOR = (
    "binary_sensor.stromligning_tomorrow_spotprice_ex_vat"
)

# Defaults
DEFAULT_NAME = "Open Spot Forecast"
DEFAULT_REGION = "DK1"
DEFAULT_CURRENCY = "DKK"
DEFAULT_VAT = 0.25
DEFAULT_PRECISION = 3
DEFAULT_PRICE_TYPE = "kWh"
DEFAULT_PREDICTION_HOURS = 48
# The backtest's best window with the zone weather (docs/ml_documentation.md)
DEFAULT_TRAINING_DAYS = 60
TRAINING_DAYS_OPTIONS = [30, 60, 90, 120, 180]

# Platforms
PLATFORMS = ["sensor", "binary_sensor"]

# Update signals
UPDATE_SIGNAL = f"{DOMAIN}_signal_update"
UPDATE_SIGNAL_FORECAST = f"{DOMAIN}_signal_forecast_update"

# Nordpool API - REMOVED (replaced by Stromligning + weather entity)
# DMI API - REMOVED (replaced by weather entity via weather.get_forecasts)

# Supported regions. ``energy_charts`` is the bidding zone on energy-charts.info,
# ``entsoe`` its EIC code on the ENTSO-E Transparency Platform (#27)
REGIONS: dict[str, dict[str, str | float]] = {
    "DK1": {
        "currency": "DKK",
        "country": "Denmark",
        "vat": 0.25,
        "tz": "Europe/Copenhagen",
        "energy_charts": "DK1",
        "entsoe": "10YDK-1--------W",
    },
    "DK2": {
        "currency": "DKK",
        "country": "Denmark",
        "vat": 0.25,
        "tz": "Europe/Copenhagen",
        "energy_charts": "DK2",
        "entsoe": "10YDK-2--------M",
    },
    "SE3": {
        "currency": "SEK",
        "country": "Sweden",
        "vat": 0.25,
        "tz": "Europe/Stockholm",
        "energy_charts": "SE3",
        "entsoe": "10Y1001A1001A46L",
    },
    "SE4": {
        "currency": "SEK",
        "country": "Sweden",
        "vat": 0.25,
        "tz": "Europe/Stockholm",
        "energy_charts": "SE4",
        "entsoe": "10Y1001A1001A47J",
    },
    "NO2": {
        "currency": "NOK",
        "country": "Norway",
        "vat": 0.25,
        "tz": "Europe/Oslo",
        "energy_charts": "NO2",
        "entsoe": "10YNO-2--------T",
    },
    "FI": {
        "currency": "EUR",
        "country": "Finland",
        "vat": 0.255,
        "tz": "Europe/Helsinki",
        "energy_charts": "FI",
        "entsoe": "10YFI-1--------U",
    },
    "EE": {
        "currency": "EUR",
        "country": "Estonia",
        "vat": 0.24,
        "tz": "Europe/Tallinn",
        "energy_charts": "EE",
        "entsoe": "10Y1001A1001A39I",
    },
    "LT": {
        "currency": "EUR",
        "country": "Lithuania",
        "vat": 0.21,
        "tz": "Europe/Vilnius",
        "energy_charts": "LT",
        "entsoe": "10YLT-1001A0008Q",
    },
    "LV": {
        "currency": "EUR",
        "country": "Latvia",
        "vat": 0.21,
        "tz": "Europe/Riga",
        "energy_charts": "LV",
        "entsoe": "10YLV-1001A00074",
    },
    "NL": {
        "currency": "EUR",
        "country": "Netherlands",
        "vat": 0.21,
        "tz": "Europe/Amsterdam",
        "energy_charts": "NL",
        "entsoe": "10YNL----------L",
    },
    "BE": {
        "currency": "EUR",
        "country": "Belgium",
        "vat": 0.06,
        "tz": "Europe/Brussels",
        "energy_charts": "BE",
        "entsoe": "10YBE----------2",
    },
    "FR": {
        "currency": "EUR",
        "country": "France",
        "vat": 0.055,
        "tz": "Europe/Paris",
        "energy_charts": "FR",
        "entsoe": "10YFR-RTE------C",
    },
    "DE": {
        "currency": "EUR",
        "country": "Germany",
        "vat": 0.19,
        "tz": "Europe/Berlin",
        "energy_charts": "DE-LU",
        "entsoe": "10Y1001A1001A82H",
    },
}

# Prediction attribute window. The full 7-day forecast (672 slots) blows past
# Home Assistant's 16 KB attribute limit, so only the next N hours of predictions
# are surfaced as entity attributes. Configurable in 12-hour steps up to 72 hours.
PREDICTION_HOURS_OPTIONS = [12, 24, 36, 48, 60, 72]
SLOTS_PER_HOUR = 4  # 96 slots per day / 24 hours

# Live forecast accuracy per lead time (slot start - time the prediction was
# stored). Each bucket holds lead times below its upper bound in hours and above
# the previous bucket's bound. The last bucket is open-ended because forecasts
# reach 7 days past the end of the known prices.
LEAD_TIME_BUCKETS: tuple[tuple[str, float], ...] = (
    ("day_1", 24.0),
    ("day_2", 48.0),
    ("day_3", 72.0),
    ("day_4_plus", float("inf")),
)
# Rolling window (days of slots) the per-bucket MAE/RMSE are computed over.
LEAD_TIME_WINDOW_DAYS = 30

# Open-Meteo weather (#22): no key, 15-minute data, 16 days ahead. Each region
# is sampled at a few fixed points across its bidding zone (wind and demand
# centres, plus an offshore wind area where there is one), as lat/lon
OPEN_METEO_API = "https://api.open-meteo.com/v1/forecast"
# Open-Meteo's archive of past forecasts (#23): days before yesterday come
# from here, so training sees forecasts like the ones it predicts from
OPEN_METEO_ARCHIVE_API = "https://historical-forecast-api.open-meteo.com/v1/forecast"
WEATHER_POINTS: dict[str, tuple[tuple[float, float], ...]] = {
    "DK1": ((57.40, 10.24), (56.20, 8.42), (55.38, 9.60), (55.53, 7.91)),
    "DK2": ((55.98, 12.39), (54.91, 11.89), (55.12, 14.73), (55.05, 12.95)),
    "SE3": ((59.33, 18.06), (57.71, 11.97), (59.38, 13.50), (60.67, 17.14)),
    "SE4": ((55.60, 13.00), (56.66, 16.36), (56.03, 14.15), (56.67, 12.86)),
    "NO2": ((58.15, 8.00), (58.97, 5.73), (59.41, 5.27), (59.21, 9.61)),
    "FI": ((60.17, 24.94), (61.50, 23.76), (63.10, 21.62), (65.01, 25.47)),
    "EE": ((59.44, 24.75), (58.38, 26.72), (58.39, 24.50), (58.25, 22.50)),
    "LT": ((54.69, 25.28), (54.90, 23.90), (55.70, 21.13)),
    "LV": ((56.95, 24.11), (56.51, 21.01), (55.87, 26.54)),
    "NL": ((52.52, 6.08), (52.37, 4.90), (51.44, 5.48), (53.22, 6.57), (52.60, 4.10)),
    "BE": ((50.85, 4.35), (51.05, 3.72), (50.63, 5.57), (51.60, 2.90)),
    "FR": ((48.86, 2.35), (45.76, 4.84), (43.60, 1.44), (47.22, -1.55), (50.63, 3.06)),
    "DE": (
        (53.55, 9.99),
        (52.52, 13.40),
        (48.14, 11.58),
        (50.11, 8.68),
        (52.37, 9.73),
        (54.30, 6.60),
    ),
}

# Day-ahead price APIs (#27): energy-charts.info (no key; the licence is per
# zone and comes with each response) and ENTSO-E (API key, fallback)
ENERGY_CHARTS_API = "https://api.energy-charts.info/price"
ENTSOE_API = "https://web-api.tp.entsoe.eu/api"
# ECB euro reference rates, to convert EUR/MWh into the configured currency
ECB_RATES_API = "https://data-api.ecb.europa.eu/service/data/EXR"

# Nordpool dataportal API (consumption and production prognoses)
NORDPOOL_API = "https://dataportal-api.nordpoolgroup.com/api"
# Nordpool's delivery day (the API's ``date``) is the CET/CEST calendar day,
# for every delivery area
NORDPOOL_MARKET_TZ = "Europe/Berlin"

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
