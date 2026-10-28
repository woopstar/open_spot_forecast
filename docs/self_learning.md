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
      - Add each prediction's raw error (its stored bias offset added back)
        to the window of its lead-time bucket, and set the slot's additive
        bias offsets from those windows (no bucket has a window since #206)
      - Add each error to its lead-time bucket (see below)
      - Remove the matched predictions from the queue
   d. If not found: skip (no forecast run covered this slot)

3. On next prediction run (every 6 hours):
   - Bias corrections are applied to raw model outputs
   - corrected_price = raw_price - offset[slot][bucket], the bucket being the
     prediction's own lead time (no clamp: prices can be negative); no
     bucket has an offset since #206, so this is the raw model output
   - The offset is stored with the prediction, so its raw error can be
     measured when the slot arrives
```

## Bias Correction

Each 15-minute slot (0-95) can have an additive offset per lead-time bucket
(see [Lead-Time Accuracy](#lead-time-accuracy)), in the price unit
(currency/kWh): the mean of the raw model's recent errors in that bucket (#15,
#118, #203). **No bucket is corrected since #206**: on 14 days of a live DK1
instance every trailing offset, per slot or pooled, raised the error (see
[Which buckets](#which-buckets)). `BIAS_WINDOW_DAYS` is empty, the
predictions are the raw model output and their stored `bias_offset` is `0.0`.
The mechanism below stays, measured and tested, so a bucket is corrected
again by listing it with its window.

```
raw_error            = stored_price + stored_bias_offset - actual
offset[slot][bucket] = mean(raw_error of the slot's last N measured days in the bucket)
corrected            = raw_prediction - offset[slot][bucket]
```

- `offset > 0` → model overpredicts → subtract
- `offset < 0` → model underpredicts → add
- no offset → the prediction is unchanged (every bucket since #206)

(`ml/bias.py`: `BIAS_WINDOW_DAYS`, `add_raw_errors()`, `window_offset()`,
`BiasMixin`.)

**Raw errors.** Every stored prediction keeps the offset that was subtracted
from it (`predictions.bias_offset`, `0.0` without one), so the raw model's
error is known whatever correction was in use, and the offset is simply its
mean. Nothing is smoothed on top: the previous offset has no say in the next.
With a constant bias the offset is that bias from the first update, and the
errors left after the correction are noise around zero.

**Window.** A slot's actual price arrives once a day, with one error per
forecast run that predicted it in the bucket. For a bucket listed in
`BIAS_WINDOW_DAYS` with N days they are kept per measured day as a count and
a sum (`raw_errors` in the slot's `error_metrics`), for the last N measured
days, none of them more than N days old (#203 shipped N = 4 for `day_1` and
`day_2`). The offset needs at least 3 errors in the window
(`MIN_BIAS_SAMPLES`). The offsets are set when their slot is matched and
again, for every slot, before each forecast run, so an offset never outlives
its window: after days without a match (the instance was off, the price
source was down) it is dropped, not kept. An instance upgraded to #206 drops
its offsets at its first forecast run the same way, and the windows of the
buckets that are no longer listed at the slot's next match.

### Which buckets

An error can only be measured when its slot arrives, a lead time after the
forecast that it corrects next: a `day_1` offset is used a day after its last
error was measured, a `day_2` offset two days after, a `day_3` offset three,
`day_4_plus` four to eight. An offset helps only if the raw model's bias lasts
longer than that.

#203 corrected `day_1` and `day_2` with 4-day windows, chosen on 6 days of the
first DK1 export with lead-time accuracy (2026-10-12), with the raw errors
estimated by adding the slow offsets of the time back. The first export with
stored offsets (2026-10-28, 14 days on v0.2.1, `--since 2026-10-15`, see
[Live accuracy](ml_documentation.md#live-accuracy-94)) measured them
exactly, and they raised the error of both buckets (MAE 2.21 with against
1.97 without for `day_1`, 2.60 against 2.24 for `day_2`, EUR ct/kWh; worse on
about 9 of the 14 days in each). The raw model's bias over those days was
small (−0.35 ct/kWh for `day_1`, −0.41 for `day_2`) next to how much it moved
from one day to the next (the daily `day_1` bias ranged from −4.0 to +3.9),
so a mean of earlier days mostly adds their noise.

Every alternative #206 scored lost to no correction as well. One offset per
bucket pooled over all 96 slots, a trailing mean of the raw daily bias
(`bias_accuracy`) measured a lead time earlier (EUR ct/kWh; n = the scored
days, a window takes the days it has; the RMSE is exact, from the daily sums
of the raw errors and their squares):

| RMSE                 | Lag    | No correction | Per slot, 4 days (live) | Pooled 1 day | 2 days | 3 days | 4 days | 7 days |
| -------------------- | ------ | ------------: | ----------------------: | -----------: | -----: | -----: | -----: | -----: |
| `day_1` (n = 13)     | 1 day  |          2.81 |                    3.20 |         3.55 |   3.32 |   3.17 |   3.14 |   2.97 |
| `day_2` (n = 12)     | 2 days |          2.99 |                    3.41 |         3.48 |   3.46 |   3.44 |   3.37 |   3.37 |
| `day_3` (n = 10)     | 3 days |          3.08 |                       — |         3.76 |   3.77 |   3.73 |   3.66 |   3.55 |
| `day_4_plus` (n = 8) | 4 days |          5.80 |                       — |         6.52 |   6.19 |   6.19 |   6.09 |   6.06 |

| Mean absolute daily bias left | No correction | Pooled 1 day | 2 days | 3 days | 4 days | 7 days |
| ----------------------------- | ------------: | -----------: | -----: | -----: | -----: | -----: |
| `day_1`                       |          1.63 |         2.04 |   1.73 |   1.76 |   1.78 |   1.68 |
| `day_2`                       |          1.72 |         2.25 |   2.12 |   1.99 |   2.01 |   2.04 |
| `day_3`                       |          1.60 |         2.40 |   2.35 |   2.35 |   2.28 |   1.95 |
| `day_4_plus`                  |          2.94 |         3.55 |   3.25 |   3.35 |   3.25 |   3.22 |

Per slot, on the raw errors per slot and day the export keeps (`raw_errors`,
the last 4 measured days of `day_1` and `day_2`; RMSE of each slot's daily
mean error, so the spread within a slot and day, the same for every
candidate, is left out; 386 `day_1` and 1,775 `day_2` errors):

| Offset                                                                 |            `day_1` |            `day_2` |
| ---------------------------------------------------------------------- | -----------------: | -----------------: |
| no correction                                                          |               4.00 |               3.89 |
| per slot (the slot's own window)                                       |               5.06 |               4.08 |
| pooled over all slots                                                  |               4.77 |               4.15 |
| per slot shrunk towards the pooled one, `n / (n + k)`, k = 3 / 10 / 30 | 4.80 / 4.74 / 4.75 | 4.21 / 4.15 / 4.13 |

So no window, no pooling and no shrinkage helps on this instance: none is
shipped. A later export re-checks it from `bias_accuracy` (the raw daily
sums are kept whatever is corrected); list a bucket in `BIAS_WINDOW_DAYS`
only where its trailing offset beats no correction there.

### Why raw errors, per lead time

**Why not a moving average.** Until #203 the offset was an exponential moving
average: once per slot and day, `offset += 0.1 × mean_error`, with
`mean_error` over the bucket's last 25 errors _left after the correction_
(#129). It moved a tenth of the remaining error a day, so it followed the bias
with a lag of about ten days, while the bias changes sign within a few. On
2026-10-13 the DK1 instance still had negative offsets from the
under-prediction of late September (08:00: −0.32 DKK/kWh against a measured
bias of +0.36) and added them to forecasts that were already too high: the
table's first row. A bucket without errors (e.g. `day_1` from 13:00 on, whose
prices are published before a forecast under 24 hours ahead is stored) kept an
offset from weeks before and lent it to the other buckets. The v10 migration
resets those offsets once (see [persistence](persistence.md)).

A slot is predicted by many forecast runs, from under an hour to 8.5 days
ahead, and the bias differs by lead time: the first DK1 export had a day-1
bias of −1.75 and a day-2 bias of −3.78 ct/kWh. So each bucket learns its own
offset, from its own errors (#118). When a prediction is made, its lead time
(`slot start − now`, `prediction_bucket()` in `ml/lead_time.py`) picks the
bucket; the slot already under way counts as `day_1`.

In DK1 and DK2 a prediction comes from one of two price models, by the age of the
lagged price it can use (the near-term model for the first two forecast
days, #135, see [ML Documentation](ml_documentation.md#near-term-model-135)).
Offsets stay per slot and lead-time bucket whichever model predicted: a
bucket's errors are those of the predictions stored in it, so its offset
follows the mix of models that fills it.

Example, with `day_1` listed with a 4-day window: if the model predicts 1.00
for slot 47 (11:45) a day ahead on four days running and the actual price is 1.30, the `day_1` offset is −0.30 and the
next day-ahead prediction for that slot gets 0.30 added. The same works for a
slot that clears at −0.20 while the model predicts 0.10: the offset is 0.30
and the corrected prediction −0.20.

Being additive, the correction never divides by a price (it stays bounded when
prices are zero or negative) and never flips a prediction's sign by scaling.
Predictions are never clamped at 0 anywhere (model, heuristic fallback, bias
correction, sensor attributes): negative prices are the hours worth shifting
load into.

### Measuring it

`bias_accuracy` holds, per slot date and lead-time bucket, the error sums of
the predictions that have a stored offset, twice: as stored (after the
correction) and for the raw model. `scripts/live_report.py` prints them side
by side ("Bias correction: with and without"): MAE and bias per bucket with
and without the correction, over the same predictions. For a bucket that is
not corrected (every bucket since #206) both are the raw model's. A
prediction stored before #203 has no offset and is in neither. The raw sums
per day and bucket are what a pooled offset is re-scored from, as above.

The multiplicative factors of older versions cannot be converted into offsets;
they are reset once on upgrade (schema v5, see [persistence](persistence.md)).
The moving-average offsets of before #203 are reset once too (schema v10).

## Slot Granularity

96 slots (15-minute intervals) instead of 24 hours. This captures intra-hour
price patterns — the 14:00-14:15 slot can have very different bias than
14:45-15:00.

## Learning Data Growth

- 672 predictions stored per forecast run (every 6 hours)
- ~2,688 new predictions daily
- A prediction is matched when its slot arrives
- Multiple forecast runs for the same timestamp = multiple learning samples
- With external forecast sensors (#120), each run also stores every source's
  forecast for the same slots (up to 672 rows per source), deleted when the
  slot is scored
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

## External Forecasts (#120)

Other integrations publish price forecasts too: Stromligning's forecast
sensor (`sensor.stromligning_forecasts_vat`, attribute `prices`) and Energi
Data Service's Carnot forecast (attribute `forecast`). They have no history
to backtest, so the only way to compare them with the model is to record them
live. With **External forecast sensors** set in the config or options flow
(none by default; nothing is read, stored or scored without one):

1. **Read** (`SensorReader.read_external_forecast()`): after every forecast
   run, each sensor's forecast, expanded to 15-minute slots by the items' own
   timestamps (an hourly price fills its four slots).
2. **Convert** (`external_forecasts.py`): the prices are taken to be shown in
   the terms of this entry's own prices, as on a shared chart: the configured
   unit, with the slot's tariff, the surcharge and VAT. `PriceOutput.to_spot()`
   (the inverse of `convert()`) and the slot's tariff (`TariffSchedule`, #107)
   turn them back into the raw spot price in currency/kWh, the exact reverse
   of how the model's forecast is exposed, so every source's errors are in
   the model's unit. Only the slots the model predicts are kept (from the end
   of the confirmed prices on): what a source shows for a slot whose price is
   known is no forecast.
3. **Store** (`store_external_forecasts()`, `ml/external_forecasts.py`): one
   `external_forecasts` row per source (the entity id), slot and reading
   (`stored_at`). Readings older than the training window are pruned, like
   stored predictions.
4. **Score** (`record_external_accuracy()`): when self-learning scores a slot
   (the live loop and the startup catch-up, right after the model's own
   lead-time accuracy), every stored forecast for it is bucketed by its lead
   time with the same `bucket_errors()` and added to the daily sums per source
   and bucket in `external_accuracy`; the rows are then deleted. A source is
   scored for the slots the model is scored for.
5. **Keep both errors of the slot** (#157): the daily sums cannot say what a
   blend of the model with a source would have scored, since that needs both
   errors of the same slot. So the slot's mean signed error per bucket is
   kept in `external_slot_errors` for every source, and for the model
   (source `model`) in the buckets a source has, over the same rolling 30
   days. The model's matched predictions are only read for this; nothing
   reads the table back in the integration.

The model never reads any of this: its error metrics, bias correction,
lead-time accuracy and predictions are the same with and without external
sensors (a test compares them).

The summary over the same rolling 30 days is cached in
`external_accuracy` (`{source: {bucket: mae, rmse, bias, samples}}`) and shown
by the forecast MAE/RMSE sensors: each has an `external` attribute with its
metric, `bias` and `samples` per source for its bucket, next to its own state.
`scripts/live_report.py` prints the full per-source table from an export
(see [Live accuracy](ml_documentation.md#live-accuracy-94)).

**Choosing the sensors.** Pick the variant that matches how this integration
shows prices: with the default VAT of 25 % that is Stromligning's
`forecasts_vat` sensor (consumer price with tariffs and VAT, enabled with
Stromligning's forecast option), and an Energi Data Service sensor whose cost
template and VAT give the same total. A source in other terms (no VAT, no
tariffs, øre instead of kr) is still recorded, but its `bias` shows the
constant difference and its MAE includes it.

**Blending is evaluated, not done** (#157). The forecast is the model's alone:
no option blends it with a source. Whether a blend would be better is measured
first, offline, by `scripts/live_report.py` from an export with 14 or more
days of `external_slot_errors` (see
[Live accuracy](ml_documentation.md#live-accuracy-94)): the model, each
source, their equal-weight mean and their inverse-MSE weighted mean over the
same slots. A blend is only built if it wins there, and only with sources
whose terms allow their forecast to be an input (see #115 for the dev-only
benchmark and the licence limits of other forecasts; Smartere Elforbrug's
forbid it).

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
| `training_seconds`    | How long that training took (s)                       |
| `stage1_seconds`      | Its cross-border stage-1 part (s), with that model    |
| `model_options`       | What shapes the model: cross-border, window, …        |
| `model_options_since` | First training with those options (UTC ISO)           |

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
lead-time accuracy, evaluation, day-ahead predictions and external forecast accuracy and drops every table of the learning
database — including the stored predictions and the price history the model
trains on — before recreating the schema. The learning and accuracy entities
are refreshed at once; the model retrains on the data collected afterwards.
