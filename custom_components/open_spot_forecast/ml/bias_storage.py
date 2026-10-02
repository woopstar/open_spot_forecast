"""Versioned migrations of the ``bias_correction`` table.

Schema v5: bias correction became an additive offset per slot (#15). Older
databases hold multiplicative correction factors (around 1.0). They cannot be
converted into offsets in currency per kWh, so they are reset once; the
offsets are relearned from the next matched predictions.

Schema v9: the offset is learned per (slot, lead-time bucket) (#118). The
table gains a ``bucket`` column; the offsets learned before, pooled over
every lead time, are kept as the ``BIAS_FALLBACK_BUCKET`` offsets, which the
other buckets use until they have learned their own.
"""

import logging
import sqlite3

from ..const import BIAS_FALLBACK_BUCKET

_LOGGER = logging.getLogger(__name__)

ADDITIVE_BIAS_SCHEMA_VERSION = 5
LEAD_TIME_BIAS_SCHEMA_VERSION = 9

BIAS_TABLE_SQL = """
    CREATE TABLE IF NOT EXISTS bias_correction (
        hour        INTEGER NOT NULL,
        bucket      TEXT    NOT NULL,
        correction  REAL    NOT NULL,
        PRIMARY KEY (hour, bucket)
    )
"""


def _schema_version(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
    return int(row[0]) if row else 0


def _set_schema_version(conn: sqlite3.Connection, version: int) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO meta (key, value) VALUES ('schema_version', ?)",
        (str(version),),
    )


def migrate_bias_to_additive(conn: sqlite3.Connection) -> None:
    """Reset stored multiplicative bias factors once (schema version < 5).

    Args:
        conn: Open learning database connection; the caller commits.
    """
    if _schema_version(conn) >= ADDITIVE_BIAS_SCHEMA_VERSION:
        return

    cleared = conn.execute("DELETE FROM bias_correction").rowcount
    _set_schema_version(conn, ADDITIVE_BIAS_SCHEMA_VERSION)
    if cleared:
        _LOGGER.info(
            "Bias correction is now an additive offset per slot: reset %d "
            "multiplicative factors, which are relearned from new matches",
            cleared,
        )


def migrate_bias_to_lead_time_buckets(conn: sqlite3.Connection) -> None:
    """Key the bias offsets by (slot, lead-time bucket) once (schema version < 9).

    The offsets pooled over every lead time become the fallback bucket's.

    Args:
        conn: Open learning database connection with every table created;
            the caller commits.
    """
    if _schema_version(conn) >= LEAD_TIME_BIAS_SCHEMA_VERSION:
        return

    columns = [c[1] for c in conn.execute("PRAGMA table_info(bias_correction)")]
    kept = 0
    if "bucket" not in columns:
        conn.execute("ALTER TABLE bias_correction RENAME TO bias_correction_pooled")
        conn.execute(BIAS_TABLE_SQL)
        kept = conn.execute(
            "INSERT INTO bias_correction (hour, bucket, correction) "
            "SELECT hour, ?, correction FROM bias_correction_pooled",
            (BIAS_FALLBACK_BUCKET,),
        ).rowcount
        conn.execute("DROP TABLE bias_correction_pooled")
    _set_schema_version(conn, LEAD_TIME_BIAS_SCHEMA_VERSION)
    if kept:
        _LOGGER.info(
            "Bias correction is now learned per slot and lead time: kept %d "
            "offsets as the %s offsets until the other lead times have learned "
            "their own",
            kept,
            BIAS_FALLBACK_BUCKET,
        )
