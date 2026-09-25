# AGENTS.md — Open Spot Forecast

This document is intended for AI coding agents (e.g., OpenAI Copilot, Claude Code) working in this
repository. It defines setup, constraints, workflow, safety rules, and quality expectations for
Open Spot Forecast (OSF) development.

Agents must follow this document strictly.

## Agent Objectives

- Implement and maintain Open Spot Forecast features for Home Assistant.
- Keep changes minimal, isolated, and testable.
- Prefer deterministic, explicit implementations over implicit or heuristic behavior.
- Never fabricate missing technical details.
- Preserve every met Home Assistant quality-scale rule and work toward Bronze, Silver, and Gold.

## No-Assumption Rule (Facts Only)

If required technical details are missing, the agent MUST:

- Locate the information inside this repository, Home Assistant documentation, or referenced
  dependencies, or
- Explicitly request clarification before implementing a dependent solution.

The agent must NOT:

- Invent API endpoints or protocols
- Guess authentication or service flows
- Introduce undocumented environment variables
- Fabricate integration requirements or constraints

If there is uncertainty, stop and request clarification.

## Repository Structure

- `custom_components/open_spot_forecast/` — Core integration, ML, and business logic
- `tests/` — Unit and integration tests
- `docs/` — Architecture documentation and design decisions
- `scripts/` — Development utilities and testing scripts
- `.github/` — GitHub Actions workflows and configurations

## Environment & Setup

The agent must use the exact versions defined in the project configuration files.

### Requirements

- Runtime: Python 3.14 (required - see `.python-version`)
- Follow versions specified in `requirements.txt` and/or `pyproject.toml`

## Sensor Wiring Rule (Mandatory)

**Every external value consumed by OSF MUST be read through `SensorReader` in
`sensor_reader.py`.** Never call `hass.states.get(...)` directly in platform or ML code.

The agent MUST:

1. **Before using any external entity**, check `docs/using_existing_sensors.md` in this
   repository first — it is the canonical, verified list of entities OSF reads (Stromligning,
   Met.no weather, Solcast, inverter power, temperature). Only fall back to searching an
   upstream integration repo when an entity is not yet listed there.
2. **If the entity already exists in OSF** (in `const.py`, `config_flow.py`, `sensor_reader.py`,
   and `__init__.py`): re-use it — never hard-code the value.
3. **If the entity is NOT yet wired into OSF**: add it through the full stack in this order:
   - `const.py` — add a `CONF_*` key (and a default entity-id string where sensible)
   - `config_flow.py` — add to the `sensors` step schema (and options flow `init` step)
   - `translations/en.json` — add `data` label (and `data_description`) for the new field in
     **both** `config.step.sensors` and `options.step.init`
   - `translations/da.json` — add the Danish translation, in sync with `en.json`
   - `sensor_reader.py` — add a `read_*` method on `SensorReader`
   - `__init__.py` — read the value during the update cycle and pass it into `weather_data`
4. **Never use a fixed numeric constant** for a value that an entity reports. Always source it
   from the live entity.

**Key entity mappings** (source → HA entity id pattern):

| Source                       | Entity (example)                                    | Used for                           |
| ---------------------------- | --------------------------------------------------- | ---------------------------------- |
| Stromligning (current price) | `sensor.stromligning_current_price_vat`             | Confirmed consumer prices (96/day) |
| Stromligning (tomorrow)      | `binary_sensor.stromligning_tomorrow_spotprice_vat` | Tomorrow's price availability      |
| Met.no weather (state)       | `weather.forecast_*`                                | Current wind/temp/humidity/cloud   |
| Met.no weather (forecast)    | `weather.get_forecasts` service                     | 48h hourly forecast                |
| Solcast solar forecast       | `sensor.solcast_pv_forecast_forecast_today`         | Solar generation estimate          |
| Inverter solar production    | `sensor.power_inverter_input_total`                 | Actual solar (training/scale)      |
| Outdoor temperature          | `sensor.metroair_330_outdoor_temperature`           | Actual temperature (training)      |

**Always check `docs/using_existing_sensors.md` first** before searching an upstream repo or
guessing an entity ID. If a new entity is confirmed to exist in HA, add it to
`docs/using_existing_sensors.md` as part of the same PR that wires it into OSF.

## ML Specification (Mandatory Reference)

The canonical definition of how the OSF ML layer must behave is in
**`docs/ml_documentation.md`** (with `docs/self_learning.md` and `docs/persistence.md`).

The agent MUST:

1. **Read `docs/ml_documentation.md` before touching any ML code** — model, feature vector,
   self-learning, bias correction, confidence scoring, or storage schema.
2. **Verify that every ML change is consistent with the docs** — 17-feature vector, 96-slot
   granularity, bias-correction EMA, solar-scaling factor, and confidence floor must all match.
3. **Update `docs/ml_documentation.md`** (and `docs/self_learning.md` / `docs/persistence.md`
   where relevant) whenever a change intentionally alters ML semantics. Docs and implementation
   must never be allowed to diverge silently.
4. **Add or update tests** that cover the affected invariants for every ML change.
5. **Include the doc check in the Definition of Done** — an ML change is not complete until both
   the docs and the tests are updated and passing.

Key invariants to verify for every ML PR:

- Feature vector stays at 17 canonical features (any add/remove is a model change).
- Slot granularity is 96 (15-min), never hourly (0-23).
- Bias correction is an additive offset: `offset = 0.9 * old + 0.1 * (old + mean_error)`, applied as `price - offset`.
- Predictions are never clamped at 0 (prices can be negative).
- Solar scaling uses the EMA of `actual_power / solcast_estimate` (learned, not a price-model input).
- Training and prediction rows both come from `build_feature_row()`; unknown inputs are NaN.
- Training uses `weather_history` actuals; prediction uses live forecasts.
- Storage schema (predictions, error_metrics, bias_correction, price_history, weather_history,
  meta) is versioned and migrations run once.

## OSF Development Rules

Spot-price forecasting is a data-driven system. The agent must:

- Avoid changes that require live market data validation unless:
  - Proper mocks are provided, or
  - A clear manual test plan is included.
- Model price and weather calculations conservatively and explicitly.
- Ensure network calls include reasonable timeouts.
- Avoid infinite retry loops.
- Handle disconnections and sensor unavailability gracefully.
- Document assumptions about sensor data accuracy and availability.

If credentials, API keys, or tokens are required:

- Never commit them.
- Never log them in plaintext.
- Always load them from environment variables or secure storage.
- Document required environment variables clearly.

## Logging & Error Handling

- Never log credentials, tokens, certificates, or sensitive identifiers.
- Prefer structured errors and logging where applicable.
- Fail explicitly rather than silently ignoring errors.
- Surface actionable error messages that help users understand and resolve issues.
- ALWAYS test for race conditions in relevant async/concurrent flows before considering a change
  complete.

## Code Standards

- Follow existing project formatting and naming conventions.
- Do not introduce large refactors in the same change as functional modifications unless explicitly
  requested.
- Keep commits small and focused.
- Avoid introducing new dependencies unless justified and discussed with the user.
- Apply Python style rules as defined in `pyproject.toml`.
- Run `./scripts/quality.sh lint` locally before committing.
- Run `./scripts/quality.sh typing` after lint — mypy type checking.
- Run `./scripts/quality.sh quality` after typing (pyright + vulture static checks).
- Run `./scripts/quality.sh translations` if any user-facing string changed (en/da sync).
- Run `./scripts/quality.sh test` to run the full test suite with coverage before opening a PR.
- See `CODE_QUALITY_STANDARDS.md` for full quality rules and conventions.
- **Never use `==` or `!=` to compare floating-point values.** In production code use an epsilon
  guard (e.g. `abs(x) > 1e-9` instead of `x != 0`). In tests always use `pytest.approx()`.

### Utility Function Centralization (No Duplication Rule)

Utility and helper functions must NEVER be duplicated across modules. Follow these rules:

**Rule: If a utility function is used in 2 or more modules, it belongs in a shared module**

1. **Before writing any utility function**, search existing code:

   - Check `const.py`, `sensor_reader.py`, and the `ml/` modules for similar functions
   - Search for regex patterns that might match the functionality

2. **If found**: Import and reuse the existing function

   - Never create a duplicate with a different name
   - Never create a local version in your module

3. **If NOT found AND will be used 2+ times**: Create in the appropriate shared module

   - Add to `const.py`, `sensor_reader.py`, or the relevant `ml/` module
   - Use public name (no leading underscore for functions meant to be reused)
   - Document with proper docstring
   - Import in all locations that need it

4. **If a one-off helper** that's ONLY used in one module:
   - Can be private (`_function_name()`) in that module
   - But if needs grow, refactor to a shared module immediately

## Home Assistant Compliance

The integration MUST comply with Home Assistant integration standards and developer guidelines.

The agent must:

- Follow Home Assistant architecture patterns for config entries, setup/unload flows, and platform
  forwarding.
- Implement entities according to Home Assistant entity model conventions (state, availability,
  device info, unique IDs, and naming).
- Use `DataUpdateCoordinator` where periodic or shared polling is required.
- Provide and maintain `config_flow`, diagnostics/repair handling (when relevant), and translations.
- Keep `manifest.json` and supported features aligned with Home Assistant requirements.
- Preserve every met Home Assistant quality-scale rule, close the tracked Bronze and Silver gaps,
  and move toward Gold where feasible.
- Add or update tests for behavior changes, especially setup flows, coordinator behavior, and entity
  state handling.

## Git Workflow

Branch naming convention:

```
feat/<issue-number>-<description>    - for introducing new features
fix/<issue-number>-<description>     - for fixing bugs
chore/<issue-number>-<description>   - for repository and code chores
docs/<issue-number>-<description>    - for documentation updates
refactor/<issue-number>-<description> - for code refactoring
```

All new branches MUST be based on the default branch (typically `main` or `master`), unless the user
explicitly instructs otherwise.

All code changes MUST start from a dedicated branch following the naming convention above.

The agent must NEVER push directly to the default branch and NEVER merge directly without explicit
user permission.

Before creating a commit, the agent MUST report the result of:

- `git status`
- Any local linting or formatting checks
- Relevant test runs for the change

## Pull Request Guidelines

### GitHub Operations — `gh` CLI (Primary), MCP Tools (When Available)

- **The `gh` CLI IS available and authenticated in the devcontainer** (`/usr/bin/gh`,
  installed via devcontainer features). Use it for GitHub API operations: PRs, issues,
  reviews, branches, releases.
- **GitHub MCP tools are not present in every session.** Never assume they exist. When
  both are available either path is fine; when they are not, `gh` is the only path.

| Operation                      | `gh` command                                           | MCP tool (when available)         |
| ------------------------------ | ------------------------------------------------------ | --------------------------------- |
| Create a PR                    | `gh pr create --base main --title ... --body-file -`   | `create_pull_request`             |
| Update a PR (title/body)       | `gh pr edit <n> --title ... --body-file -`             | `update_pull_request`             |
| Read a PR / diff / comments    | `gh pr view <n> --json ...` / `gh pr diff <n>`         | `pull_request_read`               |
| Review a PR                    | `gh pr review <n>`                                     | `pull_request_review_write`       |
| Create / update / close issues | `gh issue create` / `gh issue edit` / `gh issue close` | `issue_write` / `issue_read`      |
| List / search issues & PRs     | `gh issue list` / `gh search issues`                   | `list_issues` / `search_issues`   |
| Merge a PR                     | `gh pr merge <n>`                                      | `merge_pull_request`              |
| Create / list branches         | `git push -u origin <branch>`                          | `create_branch` / `list_branches` |

- **Multiline bodies:** pass `--body-file <path>`, or `--body-file -` and pipe a
  heredoc. Do not inline a long markdown body as a single shell argument.
- **Prefer `rtk gh ...`** — RTK filters `gh` output and cuts 26-87 % of the tokens.
- **Local git is still fine** for `git add` / `git commit` / `git checkout` / `git push`.

**REQUIRED: Code Quality Before Submission**

Before submitting a PR, the agent MUST:

- Run `./scripts/quality.sh lint` to format and lint all code, markdown, YAML and JSON
- Run `./scripts/quality.sh quality` after lint (runs pyright and vulture static checks)
- Run all tests locally: `./scripts/quality.sh test`
- Verify `git status` shows only intended changes
- Commit changes with: `git commit -m "<type>(<scope>): <description>"`

Each PR should include:

- A clear, descriptive title following Conventional Commits format
- A description of changes made
- Test strategy (automated test coverage or manual testing plan)
- Known limitations or open questions
- Any required configuration changes
- Reference to related GitHub issues using `Fixes #<issue-id>` if applicable

The agent must NOT merge a PR without explicit user permission.

### Keeping an Open PR Up to Date

If a PR already exists for the current branch and work continues on it, the agent MUST update the
PR after every meaningful commit:

- **Title** — keep it accurate to the current scope using Conventional Commits format.
- **Description** — reflect every change made since the PR was opened: new files, updated logic,
  additional tests, and any acceptance criteria that were added or completed.
- **Checklist** — tick off acceptance criteria that are now satisfied.
- Apply updates with `gh pr edit <n> --title ... --body-file <path>` (or the
  `update_pull_request` MCP tool when the session has it). Always pass the body via
  `--body-file`, never as an inline shell argument.
- Do NOT leave the PR description stale after follow-up commits.

Before merging any PR, the agent MUST ensure:

- All required CI/status checks are green/passing (including lint checks)
- Code review requirements are met (if applicable)
- Tests are passing locally and in CI

When a branch is merged, it should also be deleted locally and remotely after confirming changes are
available in the default branch.

## Security Constraints

The agent must NOT:

- Introduce telemetry without explicit approval
- Send user data to third-party services
- Add undocumented network endpoints
- Disable encryption for convenience
- Commit secrets or hardcoded credentials

All cloud endpoints or external integrations must be clearly documented.

## Testing Requirements

- Write unit tests for new logic and behavior changes.
- Test edge cases: missing sensors, unavailable entities, invalid data types, empty datasets.
- Document test scenarios and expected behaviors.
- Test concurrent or async operations for race conditions before marking changes complete.
- Include pytest or unittest fixtures for common test scenarios.

## When in Doubt

The agent must stop and request clarification regarding:

- ML model, feature-vector, or price-prediction logic or assumptions
- Home Assistant integration architecture decisions
- CI/CD expectations or tool configuration
- Required vs. optional features or breaking changes

## Definition of Done

A change is considered complete when:

- All relevant tests pass locally and in CI
- New behavior is covered by tests (where feasible)
- Code follows project style and conventions (enforced by ruff)
- **All lint checks pass** (`./scripts/quality.sh lint`)
- **Type checks pass** (`./scripts/quality.sh typing` — runs mypy)
- **Quality checks pass** (`./scripts/quality.sh quality` — runs pyright and vulture)
- **Tests pass** (`./scripts/quality.sh test` — runs pytest with coverage)
- Documentation is updated if configuration, API, or user-facing changes are made
- No secrets are committed
- All linting, formatting, type, quality, and test checks pass (`./scripts/quality.sh all` and CI)
- The implementation adheres strictly to the No-Assumption Rule
- The change aligns with Home Assistant integration standards
- Code quality is enhanced (no technical debt introduced)

@RTK.md
