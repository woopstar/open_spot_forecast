---
name: osf-ml-change
description: Activate when making any change to the OSF ML layer — feature vector, model, self-learning, bias correction, confidence scoring, or storage schema.
---

# OSF ML Change — Model, Features & Self-Learning

Activate this skill when touching **any file under `custom_components/open_spot_forecast/ml/`**:
`predictor.py`, `features.py`, `models.py`, `learning.py`, `numpy_models.py`, or
`storage.py`.

## Step 1: Read the ML Documentation

**Always read `docs/ML_DOCUMENTATION.md` and `docs/ARCHITECTURE.md` before
touching any ML code.** These are the source of truth for the model, feature
vector, and data flow.

## Step 2: Verify These Invariants for Every ML Change

Every ML change must satisfy ALL of these invariants:

- [ ] **Feature vector** — 20 canonical features; any add/remove is a model change
- [ ] **Slot granularity** — 96 slots (15-min), never assume hourly (0-23)
- [ ] **Bias correction** — EMA formula `0.9 * old + 0.1 * bias_ratio`; never invent a new scheme
- [ ] **Solar scaling** — EMA of `actual_power / solcast_estimate`; applied before the price model
- [ ] **Confidence score** — heuristic (< 5 samples) → learned (≥ 5 samples); floor respected
- [ ] **Training vs prediction segmentation** — training uses `weather_history` actuals, prediction uses live forecasts
- [ ] **Storage schema** — `predictions`, `error_metrics`, `bias_correction`, `price_history`, `weather_history`, `meta`; migrations versioned
- [ ] **Float comparisons** — epsilon guard (`abs(x) > 1e-9`) in production, `pytest.approx()` in tests
- [ ] **No blocking calls** — offload model training via `hass.async_add_executor_job()`

## Step 3: Update the Docs If Semantics Change

If a change intentionally alters the model, feature vector, bias-correction
formula, confidence logic, or storage schema, **update the relevant doc in the
same PR**:

- `docs/ML_DOCUMENTATION.md` — model, features, data sources, confidence
- `docs/SELF_LEARNING.md` — learning loop, bias correction, metrics
- `docs/PERSISTENCE.md` — schema, migrations
- `docs/ARCHITECTURE.md` — data flow, component layout

Docs and implementation must never diverge silently.

## Step 4: Add or Update Tests

Add tests covering the affected invariants. Every ML change must have regression
test coverage for:

- Feature extraction correctness
- Model training / prediction output shape
- Bias correction convergence
- Storage round-trip (save/load)
- Schema migration

## Step 5: Check File Size

Hard limit: 30 KB AND 1000 lines per file. Check before PR:

```bash
wc -c custom_components/open_spot_forecast/ml/*.py
wc -l custom_components/open_spot_forecast/ml/*.py
```

## Definition of Done for ML Work

- [ ] `docs/ML_DOCUMENTATION.md` read and understood
- [ ] All invariants verified
- [ ] Docs updated if semantics changed
- [ ] Tests added or updated
- [ ] Docs and implementation are consistent
- [ ] Lint, typing, quality, and test checks pass