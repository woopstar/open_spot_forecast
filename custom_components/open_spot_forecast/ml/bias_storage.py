"""Schema v5: bias correction became an additive offset per slot (#15).

Older databases hold multiplicative correction factors (around 1.0) in the
``bias_correction`` table. They cannot be converted into offsets in currency
per kWh, so they are reset once; the offsets are relearned from the next
matched predictions.
"""

import logging
import sqlite3

_LOGGER = logging.getLogger(__name__)

ADDITIVE_BIAS_SCHEMA_VERSION = 5


def migrate_bias_to_additive(conn: sqlite3.Connection) -> None:
    """Reset stored multiplicative bias factors once (schema version < 5).

    Args:
        conn: Open learning database connection; the caller commits.
    """
    row = conn.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
    version = int(row[0]) if row else 0
    if version >= ADDITIVE_BIAS_SCHEMA_VERSION:
        return

    cleared = conn.execute("DELETE FROM bias_correction").rowcount
    conn.execute(
        "INSERT OR REPLACE INTO meta (key, value) VALUES ('schema_version', ?)",
        (str(ADDITIVE_BIAS_SCHEMA_VERSION),),
    )
    if cleared:
        _LOGGER.info(
            "Bias correction is now an additive offset per slot: reset %d "
            "multiplicative factors, which are relearned from new matches",
            cleared,
        )
