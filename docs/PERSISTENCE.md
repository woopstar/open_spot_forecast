# Persistence

All learning data is stored in a single SQLite database:

```
/config/.storage/open_spot_forecast_{region}_learning.db
```

## Schema

| Table | Key | Content |
|-------|-----|---------|
| `predictions` | `id` (autoincrement) | Pending predictions awaiting comparison with actual prices |
| `error_metrics` | `hour` (0-95 = 15-min slot) | Per-slot error arrays (errors, abs_errors, pct_errors, predictions, actuals) |
| `bias_correction` | `hour` (0-95) | Per-slot multiplicative correction factors |
| `price_history` | `date` (YYYY-MM-DD) | Daily price arrays (96 values per day) |
| `weather_history` | `timestamp` (ISO) | 15-min weather snapshots (temp, wind, cloud, humidity, solar) |
| `meta` | `key` | Training state, schema version |

## Connection Management

- Single persistent connection per integration lifetime
- WAL journal mode with `synchronous=FULL` for durability
- `threading.Lock` serializes all writes
- Auto-checkpoint after bulk prediction storage
- Clean shutdown via `PRAGMA wal_checkpoint(TRUNCATE)` before close

## Schema Migrations

Auto-migration runs at startup (no user intervention):

| Version | Change |
|---------|--------|
| v1 → v2 | Added `id` autoincrement to predictions (was `start PRIMARY KEY`) |
| v2 → v3 | Switched error_metrics/bias_correction from hour-based (0-23) to slot-based (0-95) |

The `meta` table tracks `schema_version` so migrations only run once.

## Data Flow

```
Predictions stored ──→ predictions table (individual INSERTs)
         │
         │  ~24 hours later
         ▼
Self-learning matches prediction → error_metrics updated
         │                         bias_correction updated
         │                         prediction removed from table
         │
Every 6 hours ──→ save_all() flushes error_metrics, bias_correction,
                  price_history, meta to disk
         │
Every 15 min ──→ weather snapshot inserted into weather_history
```

Predictions are stored individually as they're generated (672 per forecast run).
Error metrics and bias corrections are persisted in bulk via `save_all()`.

## JSON Legacy

The old JSON format (`open_spot_forecast_DK1_learning.json`) is auto-migrated
on first startup if the SQLite database is empty. After migration the JSON is
renamed to `.json.bak` and never used again.
