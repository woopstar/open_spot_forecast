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
      - Update the slot's additive bias offsets, one per lead-time bucket
      - Add each error to its lead-time bucket (see below)
      - Remove the matched predictions from the queue
   d. If not found: skip (no forecast run covered this slot)

3. On next prediction run (every 6 hours):
   - Bias corrections are applied to raw model outputs
   - corrected_price = raw_price - offset[slot][bucket], the bucket being the
     prediction's own lead time (no clamp: prices can be negative)
```

## Bias Correction

Each 15-minute slot (0-95) has an additive offset per lead-time bucket
(`day_1`, `day_2`, `day_3`, `day_4_plus`, see
[Lead-Time Accuracy](#lead-time-accuracy)), in the price unit (currency/kWh),
learned via exponential moving average (#15, #118):

```
mean_error           = mean(predicted - actual)   # the slot's last 100 errors in the bucket
raw_bias             = offset[slot][bucket] + mean_error
offset[slot][bucket] = 0.9 × offset[slot][bucket] + 0.1 × raw_bias
corrected            = raw_prediction - offset[slot][bucket]
```

- `offset > 0` → model overpredicts → subtract
- `offset < 0` → model underpredicts → add
- `offset = 0` → no bias; the prediction is unchanged

The update runs once a bucket of the slot has 3 samples; the first update
sets the offset to `mean_error`. Stored predictions already had the offset in
use subtracted, so their `mean_error` is what is _left_ of the bias;
`offset + mean_error` is the raw model's bias. An EMA of `mean_error` alone
would settle at half the bias (a simulation with a week of forecast lag: 0.48
for a true bias of 1.0, versus 0.98 with this formula).

A slot is predicted by many forecast runs, from under an hour to 8.5 days
ahead, and the bias differs by lead time: the first DK1 export had a day-1
bias of −1.75 and a day-2 bias of −3.78 ct/kWh (see
[Live accuracy](ml_documentation.md#live-accuracy-94)). One offset pooled over
every lead time under-corrected the long leads and over-corrected the short
ones, so since #118 each bucket learns its own, from its own errors: a matched
slot's errors are kept pooled (`errors`, for the MAE and the metrics) and per
bucket (`bucket_errors`, the last 100 per bucket) in `error_metrics`.

When a prediction is made, its lead time (`slot start − now`,
`prediction_bucket()` in `ml/lead_time.py`) picks the bucket whose offset is
subtracted; the slot already under way counts as `day_1`. A bucket that has
not learned its own offset yet uses the `day_1` offset (`BIAS_FALLBACK_BUCKET`
in `const.py`), and its first update starts the EMA from that offset, since
the predictions it learns from had it subtracted. Without any offset the
prediction is unchanged.

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
The offsets learned before #118, pooled over every lead time, are kept as the
`day_1` offsets (schema v9), so an upgraded installation corrects every lead
time as before until the other buckets have learned their own.

## Slot Granularity

96 slots (15-minute intervals) instead of 24 hours. This captures intra-hour
price patterns — the 14:00-14:15 slot can have very different bias than
14:45-15:00.

## Learning Data Growth

- 672 predictions stored per forecast run (every 6 hours)
- ~2,688 new predictions daily
- A prediction is matched when its slot arrives
- Multiple forecast runs for the same timestamp = multiple learning samples
- Predictions stored longer ago than the training window (default 180 days)
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

The same buckets key the bias offsets (see
[Bias Correction](#bias-correction)). Predictions stored after their slot
started (negative lead time) are not counted. `stored_at` is written with its UTC offset (Home Assistant's time
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

## Predicted vs Actual (#36)

Once a slot is scored its predictions are deleted, so the forecast sensor
never shows a slot whose price is known. To keep the comparison, the matched
prediction made closest to 24 hours before the slot (`evaluation_prediction()`
in `ml/lead_time.py`; ties go to the later one, negative lead times are
skipped) is stored next to the actual price in the `evaluation` table, by the
live loop and by the startup catch-up alike. Slots older than 7 days are not
kept. The predictor caches the series (`evaluation`), reloaded at startup.

Predictions for a slot stop once Nordpool publishes its price (around 13:00
the day before), so the "24 h" prediction's real lead time (`lead_hours`,
stored with every row) runs from about 11 h for the first slots of a day to
about 35 h for the last. To compare lead times honestly, a snapshot is also
kept at the other `EVALUATION_LEAD_TIMES` (12 h and 48 h ahead, #113):
`evaluation_snapshots()` picks, per lead time, the prediction made closest to
it, and skips a lead time when that prediction is further away than half the
gap to the neighbouring lead time (`snapshot_tolerance()`: a 12 h snapshot
was made 6-18 h ahead, a 48 h one 36-60 h ahead), so a 12 h snapshot is never
a prediction made 31 h ahead. The 24 h one is always kept, as before. A 12 h
snapshot therefore exists for the first hours of a day only. The predictor
caches every lead time's series in `evaluation_snapshots`.

It is exposed by the diagnostic `Forecast evaluation` sensor
(`evaluation_sensor.py`): its state is the mean absolute error over the last
48 hours, and its attributes hold the series as compact arrays, `s` (slot
start, unix seconds), `t` (predicted) and `a` (actual), plus `samples`, `bias`
(mean signed error; positive = too high), `lead_hours` and `window_hours`,
about 5 KB. Prices are converted like every exposed price (unit, surcharge,
VAT), with the slot's tariff (#107) added to both the predicted and the
actual price, so the error stays the spot price's. The other lead times'
predictions are further arrays aligned with `s`: `t12` and `t48`, `null`
where a slot has no such snapshot (about 7 KB in all). The `get_forecast`
action returns all kept slots with `evaluation: true`, and another lead
time's with `target_hours: 12` or `48`. An ApexCharts card over the sensor:

```yaml
type: custom:apexcharts-card
graph_span: 48h
series:
  - entity: sensor.open_spot_forecast_dk1_forecast_evaluation
    name: Predicted (day ahead)
    data_generator: |
      const e = entity.attributes;
      return e.s.map((s, i) => [s * 1000, e.t[i]]);
  - entity: sensor.open_spot_forecast_dk1_forecast_evaluation
    name: Actual
    data_generator: |
      const e = entity.attributes;
      return e.s.map((s, i) => [s * 1000, e.a[i]]);
```

### Day-ahead prediction in the recorder (#113)

The evaluation series lives in attributes, which Home Assistant's history and
long-term statistics never see, and covers 48 hours. The diagnostic
`Day-ahead prediction` sensor (`day_ahead_sensor.py`, `state_class:
measurement`) closes that gap: its state is the prediction made closest to
24 hours before **the slot that is current now**, converted like every exposed
price (the slot's tariff, unit, surcharge, VAT; the hour's mean with
`hourly_average`). The recorder therefore keeps the day-ahead forecast as a
plain series, for as long as it keeps history, and a history or statistics
graph over it and the actual price shows predicted against actual with no
attribute or 7-day limit.

The value is the one the evaluation keeps once the slot is scored:
`refresh_day_ahead_predictions()` (`ml/lead_time.py`) caches, after every
forecast run and at startup, `evaluation_prediction()` of the stored
predictions of each slot in the next `DAY_AHEAD_PREDICTION_HOURS` (48), and
`day_ahead_prediction()` falls back to the `evaluation` row of a slot that was
already scored (a restart in the middle of a slot). The state is written on
every slot boundary (minute 0/15/30/45) and after every forecast and learning
update; it is unknown for a slot that was never predicted, e.g. during the
first day after setup.

```yaml
type: history-graph
hours_to_show: 168
entities:
  - entity: sensor.open_spot_forecast_dk1_day_ahead_prediction
    name: Predicted a day ahead
  - entity: sensor.stromligning_current_price_vat
    name: Actual
```

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
| `bias_corrections`    | Number of slots with a learned bias offset            |
| `bias_offsets`        | Per lead-time bucket: slots with an own offset, mean  |
| `hourly_metrics`      | Per-slot MAE, bias, sample count, bias offsets (¹)    |
| `holdout_mae`         | Latest training's holdout MAE (see below)             |
| `holdout_rmse`        | Latest training's holdout RMSE                        |
| `holdout_trained_at`  | When that training ran (UTC ISO)                      |

(¹) In the live state only: Home Assistant's recorder does not store it
(`_unrecorded_attributes`, #103). With 96 slots it would push the
attributes past the recorder's 16 KB limit, and then none of them would be
recorded. The scalar metrics above are recorded.

The holdout metrics measure how well the current model generalizes: each
training fits a copy of the model on the oldest 80 % of the price history
and scores it on the newest 20 % (see
[Training and Validation Split](ml_documentation.md#training-and-validation-split)).
They are in the model's unit (raw spot price excl. VAT, currency/kWh), are
persisted in `meta` so they survive a restart, and are `null` before the
first successful training or after a failed one. `mae` and `rmse` above are
the live errors of stored predictions against actual prices.

## Reset

The `open_spot_forecast.reset_learning` action (`services.py`, #132) calls
`reset_learning()`, which clears the in-memory error metrics, bias offsets,
lead-time accuracy, evaluation and day-ahead predictions and drops every table of the learning
database — including the stored predictions and the price history the model
trains on — before recreating the schema. The learning and accuracy entities
are refreshed at once; the model retrains on the data collected afterwards.
