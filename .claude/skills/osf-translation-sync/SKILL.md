---
name: osf-translation-sync
description: Activate before opening or updating a pull request, and whenever a user-facing string (entity name, config/options flow label, selector option, error, service) is added, changed, or removed. Keeps translations/{en,da}.json in sync.
---

# OSF Translation Sync — Keep en/da in Sync

OSF ships two languages: **English** (`en.json`, source of truth) and **Danish**
(`da.json`), both under `custom_components/open_spot_forecast/translations/`. Every user-facing
string — config/options flow step titles, field labels, field descriptions,
error/abort messages, selector option labels, entity names, service
names/descriptions — must exist and be genuinely translated in both files.

Activate this skill:

- Before opening a new pull request (as part of `osf-pr-workflow`)
- Before updating an open PR after a follow-up commit that touches
  `translations/en.json`, `config_flow.py`, or any file wiring an entity's
  `translation_key`
- Any time you add, rename, or remove a config/options flow field, selector,
  entity `translation_key`, error key, abort reason, or service

## Step 1 — Run the validator

```bash
python3 scripts/validate_translations.py
```

This diffs `da.json` against `en.json` (the source of truth) and reports:

- **Missing keys** — present in English, absent in the target file. Hard error.
- **Stale keys** — present in the target file, absent in English (leftover from
  a removed/renamed field). Hard error.
- **Placeholder mismatches** — `{placeholder}` tokens differ between English and
  the target value. Hard error.
- **Untranslated values (warning)** — the target value is byte-identical to the
  English value. Not auto-failed, because short cognates are legitimate
  (`"VAT"`, `"API"`, and a few month names are spelled the same in Danish and
  English). Review each one: if it's a multi-word phrase or a sentence, it's
  almost always a missed translation, not a cognate.

The script exits non-zero if there are any hard errors.

## Step 2 — Fix what the validator finds

- **Missing keys**: translate the English value into Danish. Check whether the
  same field text already exists translated elsewhere in the file first —
  `config.step.X` and `options.step.X` frequently duplicate the exact same
  field label/description. Reuse the existing translation rather than
  re-translating from scratch; this is the fastest way to fix a batch of
  missing keys and keeps terminology consistent.
- **Stale keys**: confirm the key has no corresponding code (`grep -rn
"<leaf_key>" custom_components/open_spot_forecast`). If genuinely dead, delete it. If it's a
  key that's about to be reintroduced, leave a note in the PR instead of
  deleting.
- **Placeholder mismatches**: the target value must contain exactly the same
  `{placeholder}` tokens as English, in whatever order reads naturally in
  Danish. Never drop or rename a placeholder.
- **Untranslated warnings**: translate the ones that are real sentences/phrases.
  Leave single-word cognates and acronyms as-is (do not force a translation
  that doesn't exist, e.g. "VAT", "API", "Sensor").

## Step 3 — Terminology glossary

Use this glossary consistently so the same English concept always renders the
same way in Danish, instead of drifting between PRs:

| English               | Danish               |
| --------------------- | -------------------- |
| Spot price            | Spotpris             |
| Forecast              | Prognose             |
| Prediction            | Forudsigelse         |
| Confidence            | Konfidens            |
| Current price         | Aktuel pris          |
| Tomorrow's prices     | Morgendagens priser  |
| Wind speed            | Vindhastighed        |
| Wind direction        | Vindretning          |
| Solar power           | Solenergi            |
| Solar forecast        | Solprognose          |
| Temperature           | Temperatur           |
| Region                | Region               |
| Currency              | Valuta               |
| VAT rate              | Momssats             |
| Decimal precision     | Decimalpræcision     |
| Price unit            | Prisenhed            |
| Enable ML predictions | Aktivér ML-prognoser |

Keep product/protocol names and abbreviations untranslated everywhere:
`Open Spot Forecast`, `Stromligning`, `Nordpool`, `Solcast`, `Met.no`, `ML`,
`VAT`, `API`, entity/field acronyms like `SoC`.

**Tone**: imperative/infinitive, matching the English source ("Select the
sensor…", "Enable this option…"). Use informal _du_-form imperatives (Home
Assistant's own translation convention), not formal _De_.

## Step 4 — Re-run the validator

```bash
python3 scripts/validate_translations.py
```

Must report 0 missing, 0 stale, 0 placeholder mismatches before the PR is
opened. Then format:

```bash
npx --yes prettier@3.1.0 --write custom_components/open_spot_forecast/translations/*.json
```

## Anti-Patterns to Avoid

- ❌ Adding a new config/options flow field in English only
- ❌ Copy-pasting the English value into `da.json` as a placeholder and
  forgetting to translate it later
- ❌ Translating `config.step.X` but not the duplicate `options.step.X`
- ❌ Leaving a stale key behind after renaming a field
- ❌ Inventing new terminology instead of reusing the glossary in Step 3
