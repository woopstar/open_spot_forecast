# Persistence

All learning data is stored in a single SQLite database:

```
/config/.storage/open_spot_forecast_{region}_learning.db
```

## Schema

| Table                | Key                         | Content                                                                                                         |
| -------------------- | --------------------------- | --------------------------------------------------------------------------------------------------------------- |
| `predictions`        | `id` (autoincrement)        | Pending predictions awaiting comparison with actual prices                                                      |
| `error_metrics`      | `hour` (0-95 = 15-min slot) | Per-slot error arrays (errors, abs_errors, pct_errors, predictions, actuals)                                    |
| `bias_correction`    | `hour` (0-95)               | Per-slot additive bias offsets (currency/kWh; column `correction`)                                              |
| `price_history`      | `date` (YYYY-MM-DD)         | Daily raw spot prices excl. VAT: one per 15-min slot from local midnight (92/96/100), `null` if missing         |
| `dayahead_prices`    | `timestamp` (UTC slot key)  | Raw day-ahead auction prices, EUR/MWh per 15-min slot (`dayahead` price source, #27)                            |
| `openmeteo_weather`  | `(timestamp, point)`        | Open-Meteo 15-min weather per sampling point (`lat,lon`): wind 80 m, temp, irradiance, pressure, humidity (#22) |
| `weather_history`    | `timestamp` (UTC slot key)  | 15-min weather snapshots (temp, wind m/s, cloud, humidity, solar), keyed `YYYY-MM-DDTHH:MM:SSZ`                 |
| `meta`               | `key`                       | Training state, schema version, HPO params, `hpo_counter`, the latest holdout metrics, source state             |
| `lead_time_accuracy` | `(date, bucket)`            | Per slot date and lead-time bucket: sample count and sums of error, absolute error and squared error            |

`price_history` never stores an invalid day (known prices all zero, or not
all finite; see `is_invalid_price_series()` in `price_series.py`), and
an invalid day never overwrites prices already stored for that date.

`meta` holds the latest successful training's holdout error: `holdout_mae`
and `holdout_rmse` (raw spot price excl. VAT, currency/kWh) and
`holdout_trained_at` (UTC ISO). They are written after every successful
training, deleted after a failed one, and restored at startup. `meta` is a
key/value table, so this needs no schema change.

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
| `ml/series_storage.py`     | `SeriesStorageMixin`           | Time-series source tables (`nordpool_prognoses`), source state in `meta` |
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
| v6 → v7 | Rewrite `weather_history` timestamps as UTC slot keys (#59)                        |

The `meta` table tracks `schema_version` so migrations only run once.

The v6 migration deletes the rows of `price_history`, `predictions`,
`error_metrics`, `bias_correction`, `volatility` and `lead_time_accuracy`,
which held (or were learned from) Stromligning's consumer price, and logs how
many days and predictions it discarded. `weather_history`,
`nordpool_prognoses` and the hyperparameters in `meta` are kept.

## Stored Timestamps

A weather snapshot is keyed by the UTC start of its 15-minute slot,
`YYYY-MM-DDTHH:MM:SSZ` (`utc_slot_key()` in `time_slots.py`), e.g. the
snapshot taken at 10:00:01 local (CEST) is `2026-09-24T08:00:00Z`. Nordpool
rows keep the UTC timestamps Nordpool publishes (`…Z`, one per hour).

`find_weather_for_timestamp` and `find_nordpool_for_timestamp` accept any ISO
timestamp (naive = Home Assistant local time) and normalize it to UTC:

- **Weather**: the snapshot closest to the timestamp, at most 30 minutes away
  (inclusive). A slot's own snapshot is 0 minutes from its start, so it wins
  over its neighbours; a slot without one falls back to a snapshot within 30
  minutes, even across midnight, and never further.
- **Nordpool**: the row of the timestamp's UTC hour (the first stored in the
  hour), the row training uses for all four slots of the hour. A missing hour
  returns None, not a neighbouring hour's row.

Window bounds are computed in Python and compared with the stored keys as
SQLite julian days on both sides, as is the pruning cutoff, so a row
exactly on a bound is always inside. Before #59 the lookups compared the
stored strings with SQLite `datetime()` results (UTC with a space separator),
which never matched a row on the same date: self-learning never recorded the
forecast weather error, and the Nordpool backfill re-fetched every stored day
at each startup. Until #46 the weather window was a float distance in julian
days, so rounding decided whether a snapshot exactly 30 minutes away was
found, and the Nordpool lookup took the nearest row within an hour, the next
hour's for every :45 slot. Training was not affected: it matches both tables
on the UTC epoch in Python (`TrainingInputs`).

The v7 migration rewrites the snapshots stored before: naive timestamps (as
written before #17) are read as Home Assistant local time, one in the
repeated hour of a DST fall-back day as its first pass, and the offset
timestamps written since as their instant. Each becomes its slot's UTC key;
when several fall into one slot the earliest is kept, as training does.
Unreadable timestamps are dropped. The migration logs how many snapshots it
rewrote, merged and dropped.

## Time-Series Sources

Upstream time series are stored through one shared layer (#32), so every
source fetches only what it is missing:

- `SeriesSpec` (`ml/series_storage.py`) describes a table on a fixed UTC
  grid: `timestamp` (`…Z`), optionally a key column (e.g. a sampling point),
  and value columns. `nordpool_prognoses` is `NORDPOOL_PROGNOSES` (hourly),
  `dayahead_prices` is `DAYAHEAD_PRICES` (15-minute, #27) and
  `openmeteo_weather` is `OPENMETEO_WEATHER` (15-minute, keyed by point,
  #22: a slot counts as stored when every point of the region has it).
- A grid point counts as stored when its row has a value in every column
  (and, for a keyed table, every expected key has such a row). Nordpool rows
  without the per-type production breakdown are therefore incomplete until
  it is published.
- `upsert_series` writes new values over old ones but never replaces a
  stored value with a missing one, and reports whether anything changed. A
  change moves `last_data_write`, which makes the model retrain.
- `TimeSeriesSource` (`api/time_series_source.py`) runs an update of a range:
  it requests only the missing grid points (grouped into requests: one per
  CET/CEST delivery day for Nordpool), plus an optional refresh window for
  revised forecasts, and records what upstream did not have as holes in
  `meta` (`source_state_<name>`, JSON). A hole reaching into the requested
  future is the source's horizon ("no data yet beyond T"): neither it nor
  anything after it is requested again before its retry time (for Nordpool
  15 minutes for recent data, 1 day for older history). A failed request records nothing,
  so it is retried at the next update. `horizon_cutoff` hides everything
  after a moment, for backtests.

Nordpool revises the current days' prognoses, so its refresh window starts
at today's delivery day: each forecast run re-fetches today and tomorrow
(the upsert tells whether anything changed) and reads them from the table.
Older days are only requested while incomplete, so complete history costs
no request. The history the model trains on (the stored price days up to
today) is filled by a background task at setup and after midnight, which
resumes where it stopped.

### Retention

History is kept for the training window (`max_history_days`, 30 days) plus
2 days. Once a day (at midnight) older `weather_history` snapshots,
`price_history` days, `nordpool_prognoses` rows, `openmeteo_weather` rows,
`dayahead_prices` rows and remembered holes are deleted. Without the ML model only `dayahead_prices` is
stored, and only the margin is kept. Before #32 only `weather_history` was pruned, to a fixed 30 days,
whenever the snapshot count was a multiple of 100.

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
         │
Each forecast ──→ today's and tomorrow's missing Nordpool prognoses fetched
         │
At midnight ──→ history older than the training window + 2 days pruned;
                missing Nordpool history backfilled in the background
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
