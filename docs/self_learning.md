# Self-Learning System

## Overview

The integration continuously improves by comparing its predictions against
actual confirmed prices. This self-learning loop runs every 15 minutes.

## How It Works

```
1. Model generates 672 predictions for the next 7 days
   → Stored in predictions table with timestamp, price, confidence

2. Every 15 minutes:
   a. Read the current raw spot price excl. VAT (the model's target; see
      [Target](ml_documentation.md#target-raw-spot-price-vat-at-output)). If today's
      prices are all zero, they are dropped (see
      [Invalid Price Data](stromligning_integration.md#invalid-price-data))
      and this update does not learn; neither does it when the current slot
      is missing in the source (see
      [Price Grid](stromligning_integration.md#price-grid))
   b. Look up every stored prediction for the current slot (today's date,
      same 15-minute slot, same UTC instant), whatever forecast run made it.
      The slot's price is found by its position from local midnight on the
      UTC timeline, so 92- and 100-slot DST days are indexed correctly (see
      [Slot Timestamps and DST](ml_documentation.md#slot-timestamps-and-dst))
   c. If found:
      - Calculate error: predicted_price - actual_price
      - Update per-slot error metrics (MAE, bias, sample count)
      - Update the slot's additive bias offset
      - Add each error to its lead-time bucket (see below)
      - Remove the matched predictions from the queue
   d. If not found: skip (no forecast run covered this slot)

3. On next prediction run (every 6 hours):
   - Bias corrections are applied to raw model outputs
   - corrected_price = raw_price - offset[slot] (no clamp: prices can be negative)
```

## Bias Correction

Each 15-minute slot (0-95) has an additive offset, in the price unit
(currency/kWh), learned via exponential moving average (#15):

```
mean_error   = mean(predicted - actual)            # the slot's last 100 errors
raw_bias     = offset[slot] + mean_error
offset[slot] = 0.9 × offset[slot] + 0.1 × raw_bias
corrected    = raw_prediction - offset[slot]
```

- `offset > 0` → model overpredicts → subtract
- `offset < 0` → model underpredicts → add
- `offset = 0` → no bias; the prediction is unchanged

The update runs once a slot has 3 samples; the first update sets the offset to
`mean_error`. Stored predictions already had the current offset subtracted, so
their `mean_error` is what is _left_ of the bias; `offset + mean_error` is the
raw model's bias. An EMA of `mean_error` alone would settle at half the bias
(a simulation with a week of forecast lag: 0.48 for a true bias of 1.0, versus
0.98 with this formula).

Example: If the model consistently predicts 1.00 but the actual price is 1.30
for slot 47 (11:45), the offset converges to about -0.30, and future
predictions for that slot get 0.30 added. The same works for a slot that clears
at -0.20 while the model predicts 0.10: the offset converges to 0.30 and the
corrected prediction to -0.20.

Being additive, the correction never divides by a price (it stays bounded when
prices are zero or negative) and never flips a prediction's sign by scaling.
Predictions are never clamped at 0 anywhere (model, heuristic fallback, bias
correction, sensor attributes): negative prices are the hours worth shifting
load into.

The multiplicative factors of older versions cannot be converted into offsets;
they are reset once on upgrade (schema v5, see [persistence](persistence.md)).

## Slot Granularity

96 slots (15-minute intervals) instead of 24 hours. This captures intra-hour
price patterns — the 14:00-14:15 slot can have very different bias than
14:45-15:00.

## Learning Data Growth

- 672 predictions stored per forecast run (every 6 hours)
- ~2,688 new predictions daily
- A prediction is matched when its slot arrives
- Multiple forecast runs for the same timestamp = multiple learning samples
- Predictions stored longer ago than the training window (default 60 days)
  are pruned automatically. The longest
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
counted. `stored_at` is written with its UTC offset (Home Assistant's time
zone); naive values from older versions are read in Home Assistant's time zone.

Per bucket, the error sums are added to the `lead_time_accuracy` table, one
row per slot date (see [persistence](persistence.md)). MAE, RMSE and bias are
computed over a rolling window of the last 30 days of slots. Older rows are
pruned. The metrics survive restarts and are reloaded at startup. The startup
catch-up replay feeds them too, and `reset_learning()` clears them.

They are exposed as diagnostic sensors, one MAE and one RMSE sensor per
bucket (e.g. `sensor.open_spot_forecast_dk1_forecast_mae_day_1`). Values are
in the unit the model learns: raw spot price excl. VAT (currency/kWh).
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
| `hourly_metrics`      | Per-slot MAE, bias, sample count, bias offset         |
| `holdout_mae`         | Latest training's holdout MAE (see below)             |
| `holdout_rmse`        | Latest training's holdout RMSE                        |
| `holdout_trained_at`  | When that training ran (UTC ISO)                      |

The holdout metrics measure how well the current model generalizes: each
training fits a copy of the model on the oldest 80 % of the price history
and scores it on the newest 20 % (see
[Training and Validation Split](ml_documentation.md#training-and-validation-split)).
They are in the model's unit (raw spot price excl. VAT, currency/kWh), are
persisted in `meta` so they survive a restart, and are `null` before the
first successful training or after a failed one. `mae` and `rmse` above are
the live errors of stored predictions against actual prices.

## Reset

Call `reset_learning()` to clear all accumulated metrics and start fresh.
This drops all tables and recreates the schema.
