"""API clients for Open Spot Forecast."""

from .nordpool_data import fetch_consumption_prognosis, fetch_production_prognosis
from .nordpool_prognoses import NordpoolPrognosisSource, forecast_prognoses

__all__ = [
    "NordpoolPrognosisSource",
    "fetch_consumption_prognosis",
    "fetch_production_prognosis",
    "forecast_prognoses",
]
