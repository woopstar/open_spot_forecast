---
name: osf-doc-sync
description: Activate after every code change to verify all documentation files that describe the changed behaviour are updated and consistent with the implementation.
---

# OSF Documentation Sync — Keep Docs in Sync with Code

Activate this skill **after making code changes** and **before opening a PR**. Stale documentation causes confusion and bugs — every doc that describes changed behaviour must be updated in the same PR.

## Documentation Files — Check Every One

When behaviour changes, check **every** file below. If it describes something you changed, update it:

| File                              | When to check                                                                 |
| --------------------------------- | ----------------------------------------------------------------------------- |
| `docs/ARCHITECTURE.md`            | System overview, data sources, data flow, or component layout changed         |
| `docs/ML_DOCUMENTATION.md`        | Model, feature vector, data sources, or confidence logic changed              |
| `docs/SELF_LEARNING.md`           | Self-learning loop, bias correction, or metrics changed                       |
| `docs/PERSISTENCE.md`             | SQLite schema, migrations, or storage behaviour changed                       |
| `docs/STROMLIGNING_INTEGRATION.md`| Price-source priority or Stromligning behaviour changed                       |
| `docs/USING_EXISTING_SENSORS.md`  | Required/recommended entities or sensor wiring changed                        |
| `.github/memories.md`             | Canonical patterns, module map, open issues, or architectural decisions changed |
| `README.md`                       | User-facing features, descriptions, or links changed                          |
| `translations/{en,da}.json`       | Any user-facing string added, changed, or removed — see `osf-translation-sync` |

## Documentation Rules

### Spec-Implementation Consistency (Highest Priority)

- `docs/ML_DOCUMENTATION.md`, `docs/SELF_LEARNING.md`, and `docs/PERSISTENCE.md`
  describe the model, learning loop, and storage schema. They **must never
  diverge silently** from the implementation.
- If a change intentionally alters the feature vector, bias-correction formula,
  or storage schema, update the relevant doc in the same commit.

### Memories.md

- Module responsibility map must reflect current file layout
- Canonical patterns must be accurate
- Open issue numbers must be up to date
- New architectural decisions must be recorded

### Translations

- Every user-facing string (field labels, errors, aborts) must have a key in `translations/en.json`
- Boolean/switch fields must have translation entries
- `en.json` is the source of truth, but `da.json` must be kept in sync with it
  too — run the `osf-translation-sync` skill (backed by
  `scripts/validate_translations.py`) before opening the PR

### README.md

- User-facing feature descriptions must be accurate
- Links must resolve correctly
- Setup/configuration instructions must reflect current flows

## Verification Checklist

Before opening a PR:

- [ ] Read every docs file listed above
- [ ] For each file: does it describe something I changed? If yes, update it
- [ ] `docs/ML_DOCUMENTATION.md` consistent with the ML implementation
- [ ] `.github/memories.md` module map matches current file layout
- [ ] `translations/en.json` has entries for all new/changed user-facing strings
- [ ] No stale or misleading documentation remains

## Anti-Patterns to Avoid

- ❌ Updating the ML model but not `docs/ML_DOCUMENTATION.md`
- ❌ Adding a new config field but not `docs/USING_EXISTING_SENSORS.md`
- ❌ Changing a feature but leaving old behavior in `README.md`
- ❌ Adding a user-facing string but skipping `translations/en.json`
- ❌ Changing the storage schema but not `docs/PERSISTENCE.md`
- ❌ Recording a pattern in code but not in `.github/memories.md`