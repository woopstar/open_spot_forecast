# Self-Learning System

## Overview

The integration continuously improves by comparing its predictions against
actual confirmed prices. This self-learning loop runs every 15 minutes.

## How It Works

```
1. Model generates 672 predictions for the next 7 days
   → Stored in predictions table with timestamp, price, confidence

2. Every 15 minutes:
   a. Read current Stromligning price (actual confirmed price)
   b. Look up the prediction made 24 hours ago for this exact timestamp
   c. If found:
      - Calculate error: predicted_price - actual_price
      - Update per-slot error metrics (MAE, bias, sample count)
      - Update per-slot bias correction factor
      - Remove the matched prediction from the queue
   d. If not found: skip (prediction hasn't aged 24h yet, or was never made)

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
- After 24h aging, the learning loop starts matching them
- Multiple forecast runs for the same timestamp = multiple learning samples
- Old predictions (>30 days) are pruned automatically

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
| `pending_predictions` | Predictions still waiting for their 24h aging         |
| `hourly_metrics`      | Per-slot MAE, bias, sample count, correction factor   |

## Reset

Call `reset_learning()` to clear all accumulated metrics and start fresh.
This drops all tables and recreates the schema.
