"""Nordpool prognoses from a learning-database export for the backtest (dev-only).

The public Nordpool prognosis APIs keep no long history, so the backtest reads
the ``nordpool_prognoses`` table of a live instance's learning database
(``--nordpool-db``, e.g. an export in the git-ignored ``.cache/live/``). The
export is opened read-only: the backtest never migrates or writes it.

As in production (#91), training rows get the stored prognosis of their hour,
while target rows get none: prognoses exist for today and tomorrow only, and a
forecast starts where the known prices end. ``day1`` gives the first forecast
day its stored prognosis, as in a morning run before tomorrow's prices.
"""

import sqlite3
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC
from pathlib import Path
from typing import Any

from custom_components.open_spot_forecast.ml.features import hour_epoch, optional_float

HOUR_SECONDS = 3600
# nordpool_prognoses column → SlotInputs field
_FIELDS = {
    "consumption": "consumption",
    "solar": "solar_generation",
    "wind_offshore": "wind_offshore",
    "wind_onshore": "wind_onshore",
}


@dataclass(frozen=True)
class NordpoolInputs:
    """Stored prognoses by UTC hour epoch, and how the backtest uses them.

    ``day1``: the first forecast day gets its stored prognosis. ``copies``:
    training adds the Nordpool-masked copy of every row, as production does
    since #91; False reproduces the training before it.
    """

    by_hour: Mapping[int, Mapping[str, float | None]]
    day1: bool = False
    copies: bool = True

    def slot_inputs(self, start: int, until: int | None = None) -> dict[str, Any]:
        """Return the ``SlotInputs`` fields of the slot starting at ``start``.

        Slots at or after ``until`` (UTC epoch) get no prognosis.
        """
        if until is not None and start >= until:
            return {}
        row = self.by_hour.get(start - start % HOUR_SECONDS)
        if row is None:
            return {}
        return {field: row.get(column) for column, field in _FIELDS.items()}


def load_nordpool_db(
    path: Path, day1: bool = False, copies: bool = True
) -> NordpoolInputs:
    """Read every ``nordpool_prognoses`` row of a learning-database export.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    if not path.is_file():
        raise FileNotFoundError(path)
    connection = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
    try:
        rows = connection.execute(
            "SELECT timestamp, consumption, solar, wind_offshore, wind_onshore"
            " FROM nordpool_prognoses ORDER BY timestamp"
        ).fetchall()
    finally:
        connection.close()
    by_hour: dict[int, dict[str, float | None]] = {}
    for timestamp, *values in rows:
        key = hour_epoch(timestamp, UTC)
        if key is not None:
            by_hour.setdefault(
                key,
                {
                    column: optional_float(value)
                    for column, value in zip(_FIELDS, values, strict=True)
                },
            )
    return NordpoolInputs(by_hour, day1=day1, copies=copies)
