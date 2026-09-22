"""Machine learning module for Open Spot Forecast."""

from .predictor import SpotPricePredictor
from .storage import LearningStorage

__all__ = ["SpotPricePredictor", "LearningStorage"]
