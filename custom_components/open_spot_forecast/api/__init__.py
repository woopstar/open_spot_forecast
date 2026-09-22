"""API clients for Open Spot Forecast."""

from .nordpool_data import fetch_consumption_prognosis, fetch_production_prognosis

__all__ = ["fetch_consumption_prognosis", "fetch_production_prognosis"]
