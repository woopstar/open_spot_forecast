"""API clients for Open Spot Forecast."""

from .nordpool_data import fetch_consumption_prognosis, fetch_production_prognosis
from .nordpool_prognoses import fetch_nordpool_prognoses

__all__ = [
    "fetch_consumption_prognosis",
    "fetch_nordpool_prognoses",
    "fetch_production_prognosis",
]
