"""Schema v6: the model learns the raw spot price excl. VAT and tariffs (#16).

Older databases hold Stromligning's all-in consumer prices (tariffs, fees and
VAT included) in ``price_history``, predictions of that price, and error
metrics, bias offsets, volatility and lead-time accuracy learned from them.
Tariffs cannot be subtracted afterwards, so all of it is discarded once; the
spot price history rebuilds from the next readings. Weather snapshots,
Nordpool prognoses and hyperparameters are kept.
"""

import logging
import sqlite3

_LOGGER = logging.getLogger(__name__)

SPOT_PRICE_SCHEMA_VERSION = 6

# Tables whose rows are in (or were learned from) the old consumer price
_CLEAR_STATEMENTS = {
    "price_history": "DELETE FROM price_history",
    "predictions": "DELETE FROM predictions",
    "error_metrics": "DELETE FROM error_metrics",
    "bias_correction": "DELETE FROM bias_correction",
    "volatility": "DELETE FROM volatility",
    "lead_time_accuracy": "DELETE FROM lead_time_accuracy",
}


def migrate_to_spot_prices(conn: sqlite3.Connection) -> None:
    """Discard consumer-price learning data once (schema version < 6).

    Args:
        conn: Open learning database connection with every table created;
            the caller commits.
    """
    row = conn.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
    version = int(row[0]) if row else 0
    if version >= SPOT_PRICE_SCHEMA_VERSION:
        return

    cleared = {
        table: conn.execute(statement).rowcount
        for table, statement in _CLEAR_STATEMENTS.items()
    }
    conn.execute(
        "INSERT OR REPLACE INTO meta (key, value) VALUES ('schema_version', ?)",
        (str(SPOT_PRICE_SCHEMA_VERSION),),
    )
    if any(cleared.values()):
        _LOGGER.info(
            "The ML model now learns the raw spot price excl. VAT and tariffs: "
            "discarded %d days of consumer price history, %d pending predictions "
            "and the error metrics, bias offsets and accuracy learned from them",
            cleared["price_history"],
            cleared["predictions"],
        )
