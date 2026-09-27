"""Two-stage cross-border price model (#29).

Day-ahead markets are coupled: an interconnector pulls a zone's price towards
its neighbours' until it is congested, so a wind lull in Germany raises the
DK1 price even when Jutland is windy. The region's own inputs cannot show
that. After EpexPredictor (``pricepredictor.py`` ``get_cross_features``,
BSD-3-Clause, reimplemented), the model is trained in two stages:

1. **Stage 1**: one price model per neighbouring zone (``NEIGHBOURS``), on
   that zone's day-ahead prices and rows from the same feature builder as
   the region's own (``build_feature_row`` with the zone's Open-Meteo
   weather, sun position and holidays).
2. **Stage 2**: the region's price model, with one ``cross_price_<zone>``
   column per neighbour: stage 1's price for the slot.

Stage-1 values in stage-2 training rows are out of sample. Each zone's stage
1 is ``STAGE1_FOLDS`` models, fitted on interleaved local days (the day's
ordinal modulo the fold count), and a training row gets the price of the
model that did not see its day. A fitted value is much closer to the real
price than a forecast is, so stage 2 would learn to trust the column more
than it deserves. Prediction rows get the mean of the fold models, so the
folds replace a separate full fit instead of adding to it.

A neighbour without enough stored prices gets a NaN column, which the price
model handles, so predictions work while its history is backfilled.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from datetime import datetime, timedelta, tzinfo
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

import numpy as np

from ..const import NEIGHBOURS, REGIONS
from .features import (
    FEATURE_NAMES,
    SlotInputs,
    build_feature_row,
    build_feature_vector,
    slot_values,
)
from .models import create_price_model
from .series_storage import OPENMETEO_WEATHER, neighbour_prices
from .zone_weather import ZoneWeatherIndex, zone_points

if TYPE_CHECKING:
    from .gbm import NumpyGradientBoosting
    from .storage import LearningStorage

_LOGGER = logging.getLogger(__name__)

# Stage-1 models per neighbour, each fitted on the days of the other folds.
# Two folds cost about one full fit (each sees half the rows); more folds
# did not lower the backtest error (docs/ml_documentation.md)
STAGE1_FOLDS = 2
# A fold model needs at least a week of 15-minute prices to be fitted
MIN_STAGE1_ROWS = 7 * 96
_SLOT = timedelta(minutes=15)


def cross_price_name(zone: str) -> str:
    """Return the stage-2 column name of a neighbour's stage-1 price."""
    return f"cross_price_{zone}"


def local_day_fold(start: datetime) -> int:
    """Return the stage-1 fold of a slot: its local day's ordinal modulo the folds."""
    return start.date().toordinal() % STAGE1_FOLDS


def neighbour_feature_rows(
    zone: str, starts: Sequence[datetime], weather: ZoneWeatherIndex | None
) -> np.ndarray:
    """Return the stage-1 model input of a neighbouring zone for each slot.

    The canonical feature vector from ``build_feature_row``, in the zone's
    local time, with the zone's weather, sun position and holidays. Inputs
    that only exist for the region itself (Nordpool prognoses, ENTSO-E
    load) are NaN.
    """
    tz = ZoneInfo(str(REGIONS[zone]["tz"]))
    rows = [
        build_feature_vector(
            build_feature_row(
                start.astimezone(tz),
                SlotInputs(**(weather.for_slot(start) if weather else {})),
                zone,
            )
        )
        for start in starts
    ]
    return np.array(rows, dtype=float).reshape(len(rows), len(FEATURE_NAMES))


class Stage1Model:
    """One neighbour's stage-1 price model: one model per fold of local days."""

    def __init__(self) -> None:
        """Start unfitted."""
        self.models: list[NumpyGradientBoosting | None] = [None] * STAGE1_FOLDS

    @property
    def is_fitted(self) -> bool:
        """Return whether any fold model is fitted."""
        return any(model is not None for model in self.models)

    def fit(self, rows: np.ndarray, prices: np.ndarray, folds: np.ndarray) -> None:
        """Fit each fold's model on the rows of the other folds.

        Args:
            rows: The zone's feature rows (``neighbour_feature_rows``).
            prices: The zone's price for each row; NaN rows are skipped.
            folds: Each row's ``local_day_fold``.
        """
        known = ~np.isnan(prices)
        for fold in range(STAGE1_FOLDS):
            train = known & (folds != fold)
            if int(train.sum()) < MIN_STAGE1_ROWS:
                self.models[fold] = None
                continue
            model = create_price_model()
            model.fit(rows[train], prices[train])
            self.models[fold] = model

    def predict_out_of_sample(self, rows: np.ndarray, folds: np.ndarray) -> np.ndarray:
        """Return each row's price from the model that did not see its day.

        NaN where that fold's model is not fitted.
        """
        result = np.full(len(rows), np.nan)
        for fold, model in enumerate(self.models):
            part = folds == fold
            if model is not None and part.any():
                result[part] = model.predict(rows[part])
        return result

    def predict(self, rows: np.ndarray) -> np.ndarray:
        """Return the mean price of the fitted fold models (NaN if none is)."""
        fitted = [model for model in self.models if model is not None]
        if not fitted or not len(rows):
            return np.full(len(rows), np.nan)
        return np.mean([model.predict(rows) for model in fitted], axis=0)


class CrossBorderModels:
    """A region's stage-1 models, fitted from the stored neighbour history.

    Neighbour prices come from ``neighbour_prices``, their weather from
    ``openmeteo_weather`` (archived forecasts for past days, the live
    forecast ahead), as for the region itself. Models are kept in memory
    only; every stage-2 training refits them.
    """

    def __init__(self, region: str, tz: tzinfo, storage: LearningStorage) -> None:
        """Initialize unfitted models for the region's ``NEIGHBOURS``."""
        self.tz = tz
        self.storage = storage
        self.zones: tuple[str, ...] = NEIGHBOURS.get(region, ())
        self._models = {zone: Stage1Model() for zone in self.zones}

    def fit(self, starts: Sequence[datetime]) -> np.ndarray:
        """Refit every neighbour's models on the stored history of ``starts``' span.

        Args:
            starts: The stage-2 training slots (timezone-aware).

        Returns:
            One out-of-sample column per neighbour for ``starts``; NaN for
            a neighbour without enough history.
        """
        began = time.monotonic()
        columns = np.full((len(starts), len(self.zones)), np.nan)
        if not starts:
            return columns
        folds = np.array(
            [local_day_fold(start.astimezone(self.tz)) for start in starts]
        )
        for index, zone in enumerate(self.zones):
            try:
                columns[:, index] = self._fit_zone(zone, starts, folds)
            except Exception as err:
                _LOGGER.warning("Stage-1 price model for %s failed: %s", zone, err)
                self._models[zone] = Stage1Model()
        _LOGGER.info(
            "Stage-1 price models of %s fitted in %.1f s",
            ", ".join(self.zones),
            time.monotonic() - began,
        )
        return columns

    def predict(self, starts: Sequence[datetime]) -> np.ndarray:
        """Return each neighbour's stage-1 price forecast for ``starts``.

        NaN for a neighbour whose models are not fitted.
        """
        columns = np.full((len(starts), len(self.zones)), np.nan)
        for index, zone in enumerate(self.zones):
            model = self._models[zone]
            if not starts or not model.is_fitted:
                continue
            try:
                weather = self._weather(zone, starts)
                columns[:, index] = model.predict(
                    neighbour_feature_rows(zone, starts, weather)
                )
            except Exception as err:
                _LOGGER.warning("Stage-1 price forecast for %s failed: %s", zone, err)
        return columns

    def _span(self, starts: Sequence[datetime]) -> tuple[datetime, datetime]:
        return min(starts), max(starts) + _SLOT

    def _weather(self, zone: str, starts: Sequence[datetime]) -> ZoneWeatherIndex:
        points = zone_points(zone)
        rows = self.storage.load_series(OPENMETEO_WEATHER, *self._span(starts), points)
        return ZoneWeatherIndex(rows, points)

    def _fit_zone(
        self, zone: str, starts: Sequence[datetime], folds: np.ndarray
    ) -> np.ndarray:
        """Fit one neighbour's model; return its out-of-sample column."""
        spec = neighbour_prices(zone)
        stored = self.storage.load_series(spec, *self._span(starts), (zone,))
        prices = slot_values(stored, "price", self.tz)
        model = Stage1Model()
        self._models[zone] = model
        if len(prices) < MIN_STAGE1_ROWS:
            _LOGGER.debug("Not enough %s prices for its stage-1 model", zone)
            return np.full(len(starts), np.nan)
        # The zone's priced slots and the region's are mostly the same (one
        # 15-minute grid): build each slot's row once
        own = [int(start.timestamp()) for start in starts]
        keys = sorted(set(prices) | set(own))
        index = {key: row for row, key in enumerate(keys)}
        rows = neighbour_feature_rows(
            zone,
            [datetime.fromtimestamp(key, self.tz) for key in keys],
            self._weather(zone, starts),
        )
        model.fit(
            rows[[index[key] for key in prices]],
            np.array(list(prices.values()), dtype=float),
            np.array(
                [local_day_fold(datetime.fromtimestamp(k, self.tz)) for k in prices]
            ),
        )
        if not model.is_fitted:
            return np.full(len(starts), np.nan)
        return model.predict_out_of_sample(rows[[index[key] for key in own]], folds)
