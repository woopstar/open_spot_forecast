"""Migrate learning data from JSON file to SQLite database.

Usage:
    python migrate_to_sqlite.py [path_to_json]

The .db file is created next to the JSON file.

Note: The integration auto-migrates on first startup if the JSON file
is in .storage/ and the SQLite database is empty. This script is
useful for manual migration or testing.
"""

import json
import sqlite3
import sys
from pathlib import Path


def migrate(json_path: Path) -> None:
    """Migrate JSON learning data to SQLite."""
    db_path = json_path.with_suffix(".db")

    # Load JSON data
    print(f"Reading JSON from: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    region = data.get("region", "unknown")
    print(f"Region: {region}")
    print(f"Prediction entries: {len(data.get('prediction_history', []))}")
    print(f"Bias corrections: {len(data.get('bias_correction', {}))}")
    print(f"Error metrics: {len(data.get('error_metrics', {}))}")

    # Create SQLite database
    print(f"\nCreating SQLite database at: {db_path}")
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")

    conn.executescript("""
        CREATE TABLE IF NOT EXISTS predictions (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            start       TEXT    NOT NULL,
            price       REAL    NOT NULL,
            confidence  REAL,
            hour        INTEGER,
            minute      INTEGER,
            stored_at   TEXT    NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_predictions_start
            ON predictions(start);

        CREATE INDEX IF NOT EXISTS idx_predictions_stored_at
            ON predictions(stored_at);

        CREATE INDEX IF NOT EXISTS idx_predictions_hour_minute
            ON predictions(hour, minute);

        CREATE TABLE IF NOT EXISTS error_metrics (
            hour        INTEGER PRIMARY KEY,
            data        TEXT    NOT NULL
        );

        CREATE TABLE IF NOT EXISTS bias_correction (
            hour        INTEGER PRIMARY KEY,
            correction  REAL    NOT NULL
        );

        CREATE TABLE IF NOT EXISTS price_history (
            date        TEXT    PRIMARY KEY,
            prices      TEXT    NOT NULL
        );

        CREATE TABLE IF NOT EXISTS meta (
            key         TEXT    PRIMARY KEY,
            value       TEXT
        );
    """)

    # Import predictions (only new-format entries with "start" key)
    predictions = data.get("prediction_history", [])
    imported_preds = 0
    skipped_preds = 0

    for p in predictions:
        start = p.get("start", p.get("timestamp"))
        price = p.get("price", p.get("predicted_price", 0))
        if not start or not price:
            skipped_preds += 1
            continue

        try:
            conn.execute(
                """INSERT INTO predictions
                   (start, price, confidence, hour, minute, stored_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    start,
                    float(price),
                    p.get("confidence", 0.6),
                    p.get("hour", 0),
                    p.get("minute", 0),
                    p.get("stored_at", ""),
                ),
            )
            imported_preds += 1
        except Exception as err:
            print(f"  Skipping prediction (error): {err}")
            skipped_preds += 1

    print(f"  Imported: {imported_preds}, Skipped: {skipped_preds}")

    # Import error metrics
    error_metrics = data.get("error_metrics", {})
    imported_errors = 0
    for hour, metrics in error_metrics.items():
        data_json = json.dumps(
            {
                "errors": metrics.get("errors", []),
                "abs_errors": metrics.get("abs_errors", []),
                "pct_errors": metrics.get("pct_errors", []),
                "predictions": metrics.get("predictions", []),
                "actuals": metrics.get("actuals", []),
                "count": metrics.get("count", 0),
            }
        )
        conn.execute(
            "INSERT OR REPLACE INTO error_metrics (hour, data) VALUES (?, ?)",
            (int(hour), data_json),
        )
        imported_errors += 1
    print(f"  Error metrics imported: {imported_errors}")

    # Import bias corrections
    bias_correction = data.get("bias_correction", {})
    imported_bias = 0
    for hour, correction in bias_correction.items():
        conn.execute(
            "INSERT OR REPLACE INTO bias_correction (hour, correction) VALUES (?, ?)",
            (int(hour), float(correction)),
        )
        imported_bias += 1
    print(f"  Bias corrections imported: {imported_bias}")

    # Import price history
    price_history = data.get("price_history", [])
    imported_prices = 0
    for entry in price_history:
        date = entry.get("date")
        prices = entry.get("prices", [])
        if date:
            conn.execute(
                "INSERT OR REPLACE INTO price_history (date, prices) VALUES (?, ?)",
                (date, json.dumps(prices)),
            )
            imported_prices += 1
    print(f"  Price history days imported: {imported_prices}")

    # Import meta
    training_samples = data.get("training_samples", 0)
    is_trained = data.get("is_trained", False)
    conn.execute(
        "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
        ("training_samples", str(training_samples)),
    )
    conn.execute(
        "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
        ("is_trained", "1" if is_trained else "0"),
    )
    print(f"  Meta: training_samples={training_samples}, is_trained={is_trained}")

    conn.commit()
    conn.close()

    print(f"\nMigration complete. Database: {db_path}")
    print(f"File size: {db_path.stat().st_size / 1024:.1f} KB")

    # Compare sizes
    json_size = json_path.stat().st_size / 1024
    db_size = db_path.stat().st_size / 1024
    print("\nSize comparison:")
    print(f"  JSON:  {json_size:.1f} KB")
    print(f"  SQLite: {db_size:.1f} KB")
    print(f"  Reduction: {(1 - db_size / json_size) * 100:.0f}%")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        json_path = Path(sys.argv[1])
    else:
        json_path = Path("open_spot_forecast_DK1_learning.json")

    if not json_path.exists():
        print(f"Error: File not found: {json_path}")
        print(
            "Tip: The integration auto-migrates if the JSON file is in HA's .storage/ directory."
        )
        sys.exit(1)

    migrate(json_path)
    print(f"\nTo use in HA, ensure the .db file is in: .storage/{json_path.stem}.db")
