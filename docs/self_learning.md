# Self-Learning System

## Overview

The integration continuously improves by comparing its predictions against
actual confirmed prices. This self-learning loop runs every 15 minutes.

## How It Works

```
1. Model generates 672 predictions for the next 7 days
   → Stored in predictions table with timestamp, price, confidence

2. Every 15 minutes:
   a. Read current Stromligning price (actual confirmed price). If today's
      prices are all zero or have missing values, they are dropped (see
      [Invalid Price Data](stromligning_integration.md#invalid-price-data))
      and this update does not learn
   b. Look up every stored prediction for the current slot (today's date,
      same 15-minute slot), whatever forecast run made it
   c. If found:
      - Calculate error: predicted_price - actual_price
      - Update per-slot error metrics (MAE, bias, sample count)
      - Update per-slot bias correction factor
      - Add each error to its lead-time bucket (see below)
      - Remove the matched predictions from the queue
   d. If not found: skip (no forecast run covered this slot)

3. On next prediction run (every 6 hours):
   - Bias corrections are applied to raw model outputs
   - Corrected_price = raw_price × bias_correction[slot]
```

## Bias Correction

Each 15-minute slot (0-95) has a multiplicative correction factor learned via
exponential moving average:

```
bias_ratio = 1.0 - (mean_error / mean_actual_price)
correction[slot] = 0.9 × old_correction + 0.1 × bias_ratio
```

- `correction > 1.0` → model underpredicts → multiply up
- `correction < 1.0` → model overpredicts → multiply down
- `correction = 1.0` → no bias

Example: If the model consistently predicts 1.00 but actual is 1.30 for
slot 47 (11:45), the correction converges to ~1.30, and future predictions
for that slot get multiplied by 1.30.

## Slot Granularity

96 slots (15-minute intervals) instead of 24 hours. This captures intra-hour
price patterns — the 14:00-14:15 slot can have very different bias than
14:45-15:00.

## Learning Data Growth

- 672 predictions stored per forecast run (every 6 hours)
- ~2,688 new predictions daily
- A prediction is matched when its slot arrives
- Multiple forecast runs for the same timestamp = multiple learning samples
- Old predictions (stored >30 days ago) are pruned automatically. The longest
  lead time is ~8.5 days (7 days past the end of the known prices), so every
  prediction can be matched before it is pruned

## Lead-Time Accuracy

Each slot is predicted by several forecast runs, days apart. When a slot is
matched, every prediction's error is also bucketed by its lead time,
`slot_start - stored_at`:

| Bucket       | Lead time |
| ------------ | --------- |
| `day_1`      | 0-24 h    |
| `day_2`      | 24-48 h   |
| `day_3`      | 48-72 h   |
| `day_4_plus` | 72 h+     |

Predictions stored after their slot started (negative lead time) are not
counted. Naive `stored_at` values are read in Home Assistant's time zone.

Per bucket, the error sums are added to the `lead_time_accuracy` table, one
row per slot date (see [persistence](persistence.md)). MAE, RMSE and bias are
computed over a rolling window of the last 30 days of slots. Older rows are
pruned. The metrics survive restarts and are reloaded at startup. The startup
catch-up replay feeds them too, and `reset_learning()` clears them.

They are exposed as diagnostic sensors, one MAE and one RMSE sensor per
bucket (e.g. `sensor.open_spot_forecast_dk1_forecast_mae_day_1`). Values are
in the unit of the confirmed prices the model learns from (currency/kWh).
Attributes: `samples`, `bias` (mean signed error; positive = overpredicting)
and `window_days`.

## Metrics Available

Via `sensor.open_spot_forecast_dk1_learning_metrics`:

| Field                 | Description                                           |
| --------------------- | ----------------------------------------------------- |
| `status`              | "learning" when actively comparing predictions        |
| `total_samples`       | Total prediction→actual comparisons made              |
| `mae`                 | Mean absolute error across all slots                  |
| `rmse`                | Root mean square error                                |
| `mean_bias`           | Average systematic error (negative = underpredicting) |
| `slots_tracked`       | Number of 15-min slots with data                      |
| `pending_predictions` | Predictions still waiting for their slot to arrive    |
| `hourly_metrics`      | Per-slot MAE, bias, sample count, correction factor   |

## Reset

Call `reset_learning()` to clear all accumulated metrics and start fresh.
This drops all tables and recreates the schema.
