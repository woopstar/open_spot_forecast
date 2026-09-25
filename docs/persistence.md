# Persistence

All learning data is stored in a single SQLite database:

```
/config/.storage/open_spot_forecast_{region}_learning.db
```

## Schema

| Table                | Key                         | Content                                                                                                 |
| -------------------- | --------------------------- | ------------------------------------------------------------------------------------------------------- |
| `predictions`        | `id` (autoincrement)        | Pending predictions awaiting comparison with actual prices                                              |
| `error_metrics`      | `hour` (0-95 = 15-min slot) | Per-slot error arrays (errors, abs_errors, pct_errors, predictions, actuals)                            |
| `bias_correction`    | `hour` (0-95)               | Per-slot additive bias offsets (currency/kWh; column `correction`)                                      |
| `price_history`      | `date` (YYYY-MM-DD)         | Daily raw spot prices excl. VAT: one per 15-min slot from local midnight (92/96/100), `null` if missing |
| `weather_history`    | `timestamp` (ISO)           | 15-min weather snapshots (temp, wind m/s, cloud, humidity, solar), stored with UTC offset               |
| `meta`               | `key`                       | Training state, schema version, HPO params and `hpo_counter`                                            |
| `lead_time_accuracy` | `(date, bucket)`            | Per slot date and lead-time bucket: sample count and sums of error, absolute error and squared error    |

`price_history` never stores an invalid day (known prices all zero, or not
all finite; see `is_invalid_price_series()` in `price_series.py`), and
an invalid day never overwrites prices already stored for that date.

`lead_time_accuracy` is created with `CREATE TABLE IF NOT EXISTS` on every
startup, so existing databases gain it without a versioned migration. Rows
older than the 30-day rolling window are pruned whenever the metrics are
refreshed. The table is dropped and recreated by `clear_all()`.

## Code Layout

`LearningStorage` (`ml/storage.py`) is the only class that touches the
database. It owns the connection, the write lock, the schema and the
migrations, and inherits its table operations from one mixin per group of
tables:

| Module                     | Mixin                          | Tables                                                                   |
| -------------------------- | ------------------------------ | ------------------------------------------------------------------------ |
| `ml/prediction_storage.py` | `PredictionStorageMixin`       | `predictions`                                                            |
| `ml/history_storage.py`    | `HistoryStorageMixin`          | `weather_history`, `nordpool_prognoses`, `price_history`                 |
| `ml/state_storage.py`      | `LearningStateStorageMixin`    | `error_metrics`, `bias_correction`, `volatility`, `meta`, bulk save/load |
| `ml/accuracy_storage.py`   | `LeadTimeAccuracyStorageMixin` | `lead_time_accuracy`                                                     |

The mixins inherit `StorageMixinBase` (`ml/storage_base.py`), which declares
the shared `_lock`, `_ensure_conn()` and `last_data_write` for type checking
only. The versioned data migrations live next to them in `ml/bias_storage.py`
(v5) and `ml/spot_migration.py` (v6).

## Connection Management

- Single persistent connection per integration lifetime
- WAL journal mode with `synchronous=FULL` for durability
- `threading.Lock` serializes all writes
- Auto-checkpoint after bulk prediction storage
- Clean shutdown via `PRAGMA wal_checkpoint(TRUNCATE)` before close

## Schema Migrations

Auto-migration runs at startup (no user intervention):

| Version | Change                                                                             |
| ------- | ---------------------------------------------------------------------------------- |
| v1 → v2 | Added `id` autoincrement to predictions (was `start PRIMARY KEY`)                  |
| v2 → v3 | Switched error_metrics/bias_correction from hour-based (0-23) to slot-based (0-95) |
| v3 → v4 | Added forecast weather columns to predictions                                      |
| v4 → v5 | Reset `bias_correction`: multiplicative factors became additive offsets (#15)      |
| v5 → v6 | Discard consumer-price learning data: the model learns the raw spot price (#16)    |

The `meta` table tracks `schema_version` so migrations only run once.

The v6 migration deletes the rows of `price_history`, `predictions`,
`error_metrics`, `bias_correction`, `volatility` and `lead_time_accuracy`,
which held (or were learned from) Stromligning's consumer price, and logs how
many days and predictions it discarded. `weather_history`,
`nordpool_prognoses` and the hyperparameters in `meta` are kept.

## Data Flow

```
Predictions stored ──→ predictions table (individual INSERTs)
         │
         │  when the predicted slot arrives
         ▼
Self-learning matches prediction → error_metrics updated
         │                         bias_correction updated
         │                         lead_time_accuracy sums added (committed)
         │                         prediction removed from table
         │
Every 6 hours ──→ save_all() flushes error_metrics, bias_correction,
                  price_history, meta to disk
         │
Every 15 min ──→ weather snapshot inserted into weather_history
```

Predictions are stored individually as they're generated (672 per forecast run).
Error metrics and bias corrections are persisted in bulk via `save_all()`.

Nordpool prognosis rows are upserted and only rewritten when a value differs.
`LearningStorage.last_data_write` records (in memory) when a weather snapshot
or a changed Nordpool row was last written; the predictor uses it to decide
whether to retrain (see `docs/ml_documentation.md` → Retraining).

## JSON Legacy

The old JSON format (`open_spot_forecast_DK1_learning.json`) is auto-migrated
on first startup if the SQLite database is empty. Only its training state
is imported: its prices, predictions and error metrics are consumer prices
(#16), and its bias factors are multiplicative (#15). After
migration the JSON is renamed to `.json.bak` and never used again.
