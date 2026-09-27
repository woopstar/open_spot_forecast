---
name: osf-pre-flight
description: Run before starting any OSF code change. Checkout main, pull latest, read repository memory and relevant docs, create a feature branch.
---

# OSF Pre-Flight — Start Any Code Change

Activate this skill **before writing any code** — when the user asks to fix a bug, implement a feature, or make any change to the OSF codebase.

## Step 1: Checkout Main and Pull Latest

```bash
git checkout main
git pull
```

## Step 2: Read Repository Memory

Read `.github/memories.md`. Pay special attention to:

- Module responsibility map (component, sensor I/O, ML, API layers)
- Canonical patterns (const.py constants, SensorReader, SpotPricePredictor, LearningStorage)
- Feature vector (23 features) and 96-slot granularity
- Bias-correction EMA and solar-scaling factor
- File size limits (30 KB AND 1000 lines)
- File organization patterns (by responsibility, not by theme)
- Sensor wiring protocol
- Testing and logging rules

## Step 3: Read Any Issue Being Solved

If this is issue-driven work, read the full GitHub issue before touching any code.

## Step 4: Create a Feature Branch

Format: `<type>/<issue-number>-<slug>`

| Type       | Use for                  |
| ---------- | ------------------------ |
| `feat`     | New features             |
| `fix`      | Bug fixes                |
| `chore`    | Repository/code chores   |
| `docs`     | Documentation updates    |
| `refactor` | Code refactoring         |
| `perf`     | Performance improvements |
| `test`     | Test additions/updates   |
| `ci`       | CI/CD changes            |

Examples: `fix/444-bias-correction-ema`, `feat/123-add-temperature-feature`

All branches MUST be based on main unless the user explicitly instructs otherwise.

## Step 5: Identify Relevant Documentation

Based on the change type, read these docs before touching code:

| Change touches                                | Must read                          |
| --------------------------------------------- | ---------------------------------- |
| ML model, feature vector, prediction logic    | `docs/ml_documentation.md`         |
| Self-learning, bias correction, error metrics | `docs/self_learning.md`            |
| SQLite storage, schema migrations             | `docs/persistence.md`              |
| Price sources (Stromligning, Nordpool)        | `docs/stromligning_integration.md` |
| External sensor entities                      | `docs/using_existing_sensors.md`   |
| System overview, data flow                    | `docs/architecture.md`             |

## Step 6: Understand the Affected Code

Search and read the relevant source files. Do not guess file paths — use `grep` and `glob` to locate them.

## Reminder: One Issue Per Branch

Solve **one issue only** per branch and PR. Do not combine multiple issues. Do not refactor unrelated code.
