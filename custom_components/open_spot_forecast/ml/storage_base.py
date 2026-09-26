"""Shared type declarations for the LearningStorage mixins.

``LearningStorage`` (``ml/storage.py``) owns the SQLite connection, the write
lock and the schema; its table operations live in mixins, one module per
group of tables. This base class declares the state and helpers the mixins
use via ``self`` once, so mypy and pyright see a consistent, fully-typed
``self``. It holds no runtime state or behaviour.
"""

import sqlite3
import threading
from datetime import datetime

from homeassistant.core import HomeAssistant


class StorageMixinBase:
    """Type-only base declaring the storage state shared by every mixin."""

    # --- Instance state, initialized in LearningStorage.__init__ ---
    hass: HomeAssistant
    _lock: threading.Lock
    # When training inputs (weather/Nordpool rows) last changed (UTC)
    last_data_write: datetime | None
    # Each price day as last written or read (spot_prices, #24)
    _saved_price_days: dict[str, list[float | None]]

    # --- Implemented by LearningStorage ---
    def _ensure_conn(self) -> sqlite3.Connection:
        raise NotImplementedError
