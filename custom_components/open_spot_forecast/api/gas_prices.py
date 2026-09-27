"""The daily natural-gas price (#28), from Instrat's API.

``GasPriceSource`` keeps the ``gas_prices`` table (one price per UTC day)
current through the shared ``TimeSeriesSource``. Sources considered:

* **Instrat** (``energy-api.instrat.pl``, used): a JSON API with the TGE
  (Polish exchange) gas day-ahead index, PLN/MWh, for every day of a date
  range, weekends included. Licence CC BY-NC 4.0 (energy.instrat.pl).
* Bundesnetzagentur's gas price page (THE Day Ahead and Month+1, the
  German hub; EpexPredictor's source): the data is embedded in the HTML as
  JavaScript and has to be scraped with a regex, which EpexPredictor's
  history shows breaking. It scored the same as Instrat in the backtest.
* energy-charts.info and ENTSO-E have no gas prices.

The price only sets a level, so the currency and the hub do not matter as
long as they follow the European gas price. A failed or unusable response
is logged once (then at debug level) and stores nothing, so the feature is
NaN and predictions go on.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

from homeassistant.core import HomeAssistant
from homeassistant.helpers.aiohttp_client import async_get_clientsession

from ..const import INSTRAT_GAS_API
from ..ml.series_storage import GAS_PRICES
from ..time_slots import UTC_KEY_FORMAT
from .http import async_get
from .time_series_source import TimeSeriesSource

if TYPE_CHECKING:
    from ..ml.storage import LearningStorage

_LOGGER = logging.getLogger(__name__)

_DAY = timedelta(days=1)
_QUERY_FORMAT = "%d-%m-%YT%H:%M:%SZ"


def instrat_query(start: datetime, end: datetime) -> dict[str, str]:
    """Return the query for the days of ``[start, end)`` and one on each side.

    The API's range is shifted by a time zone against its day labels, so a
    day more on each side makes sure every wanted day is in the answer.
    """
    return {
        "date_from": (start - _DAY).astimezone(UTC).strftime(_QUERY_FORMAT),
        "date_to": (end + _DAY).astimezone(UTC).strftime(_QUERY_FORMAT),
        "aggregation_timeframe": "day",
        "aggregation_type": "avg",
    }


def parse_instrat_gas(payload: Any) -> list[dict[str, Any]]:
    """Return ``gas_prices`` rows from an Instrat response.

    Each entry's ``date`` labels its day; entries without a numeric price
    are skipped.

    Raises:
        TypeError: If the payload is not a list of entries.
    """
    if not isinstance(payload, list):
        raise TypeError("Instrat gas prices are not a list")
    rows: list[dict[str, Any]] = []
    for entry in payload:
        if not isinstance(entry, dict):
            raise TypeError("Instrat gas price entry is not an object")
        price = entry.get("price")
        try:
            day = datetime.fromisoformat(str(entry.get("date"))[:10])
        except ValueError:
            continue
        if isinstance(price, int | float) and not isinstance(price, bool):
            moment = day.replace(tzinfo=UTC)
            rows.append({"timestamp": moment.strftime(UTC_KEY_FORMAT), "price": price})
    return rows


class GasPriceSource(TimeSeriesSource):
    """The daily natural-gas price."""

    spec = GAS_PRICES
    # Published once a day, a day or two late
    revalidate_after = timedelta(hours=6)

    def __init__(
        self,
        hass: HomeAssistant,
        storage: LearningStorage,
        horizon_cutoff: datetime | None = None,
    ) -> None:
        """Initialize the source; nothing is fetched yet."""
        super().__init__(hass, storage, horizon_cutoff)
        self._failed = False

    def _warn(self, message: str, *args: Any) -> None:
        """Log a failure once as a warning, repeats at debug level."""
        level = logging.DEBUG if self._failed else logging.WARNING
        self._failed = True
        _LOGGER.log(level, message, *args)

    async def _fetch(
        self, start: datetime, end: datetime
    ) -> list[dict[str, Any]] | None:
        """Fetch the days of ``[start, end)``."""
        response = await async_get(
            async_get_clientsession(self.hass),
            INSTRAT_GAS_API,
            "Instrat",
            params=instrat_query(start, end),
        )
        if response is None or response.status != 200:
            status = response.status if response is not None else "no answer"
            self._warn("Gas prices unavailable (Instrat: %s)", status)
            return None
        try:
            rows = parse_instrat_gas(json.loads(response.text))
        except (ValueError, TypeError) as err:
            self._warn("Unusable Instrat gas prices: %s", err)
            return None
        self._failed = False
        return [
            row
            for row in rows
            if start <= datetime.fromisoformat(row["timestamp"]) < end
        ]
