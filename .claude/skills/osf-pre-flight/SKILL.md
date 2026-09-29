---
name: osf-pre-flight
description: Run before starting any OSF code change. Fetch the latest main, create a dedicated git worktree and feature branch, read repository memory and relevant docs.
---

# OSF Pre-Flight — Start Any Code Change

Activate this skill **before writing any code** — when the user asks to fix a bug, implement a feature, or make any change to the OSF codebase.

## Step 0: Never Switch the Shared Checkout

`/workspaces/osf` is the **shared main checkout**. Several agent sessions (and the
user) work in the same devcontainer at the same time, and they all see that working
tree, index and `HEAD`. Switching or rebasing it moves the branch under whoever else
is working there (on 2026-09-24 a session's feature branch, with uncommitted edits,
was rebased onto `origin/main` by another session's pre-flight).

In `/workspaces/osf`, **never** run `git checkout`, `git switch`, `git pull`,
`git rebase`, `git reset` or `git stash`. Read-only commands (`git fetch`,
`git log`, `git status`, `git worktree ...`) are fine.

## Step 1: Create a Worktree and Branch From the Latest Main

Every task gets its own linked worktree under `/workspaces/worktrees/` (a Docker
volume that survives container rebuilds; the same directory Zed and Claude Code use
for their worktrees). Branch format: `<type>/<issue-number>-<slug>`.

```bash
cd /workspaces/osf
git fetch origin
git worktree add /workspaces/worktrees/osf-<issue-number> -b <type>/<issue-number>-<slug> origin/main
cd /workspaces/worktrees/osf-<issue-number>
```

Do **all** work from that worktree: edits, `./scripts/quality.sh`, commits, push and
`gh pr ...`. Keep every shell command's working directory in the worktree (e.g. set
the tool's working directory, or `cd` into it first).

If the session already runs in a worktree it created (e.g. Claude Code's
`EnterWorktree` or a Zed parallel agent), branch there instead:
`git fetch origin && git switch -c <type>/<issue-number>-<slug> origin/main`.

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

All branches MUST be based on `origin/main` unless the user explicitly instructs otherwise.

No per-worktree setup is needed: Python packages are installed container-wide, git
hooks live in the common git dir, and `scripts/quality.sh` keeps its mypy, ruff and
pytest caches separate per worktree path.

## Step 2: Read Repository Memory

Read `.github/memories.md`. Pay special attention to:

- Module responsibility map (component, sensor I/O, ML, API layers)
- Canonical patterns (const.py constants, SensorReader, SpotPricePredictor, LearningStorage)
- Feature vector (24 features) and 96-slot granularity
- Bias-correction EMA and solar-scaling factor
- File size limits (30 KB AND 1000 lines)
- File organization patterns (by responsibility, not by theme)
- Sensor wiring protocol
- Testing and logging rules

## Step 3: Read Any Issue Being Solved

If this is issue-driven work, read the full GitHub issue before touching any code.

## Step 4: Identify Relevant Documentation

Based on the change type, read these docs before touching code:

| Change touches                                | Must read                          |
| --------------------------------------------- | ---------------------------------- |
| ML model, feature vector, prediction logic    | `docs/ml_documentation.md`         |
| Self-learning, bias correction, error metrics | `docs/self_learning.md`            |
| SQLite storage, schema migrations             | `docs/persistence.md`              |
| Price sources (Stromligning, Nordpool)        | `docs/stromligning_integration.md` |
| External sensor entities                      | `docs/using_existing_sensors.md`   |
| System overview, data flow                    | `docs/architecture.md`             |

## Step 5: Understand the Affected Code

Search and read the relevant source files. Do not guess file paths — use `grep` and `glob` to locate them.

## Step 6: Clean Up After the Merge

The worktree is disposable. After the PR is merged (see `osf-pr-workflow` →
Merge Rules), remove it and its local branch from the shared checkout:

```bash
cd /workspaces/osf
git worktree remove /workspaces/worktrees/osf-<issue-number>
git branch -D <type>/<issue-number>-<slug>
git fetch --prune origin
```

`git worktree remove` refuses to delete a worktree with uncommitted or untracked
changes; check them before adding `--force`. `git worktree list` shows what is left;
`git worktree prune` clears entries whose directory is already gone.

## Reminder: One Issue Per Branch

Solve **one issue only** per branch, worktree and PR. Do not combine multiple issues. Do not refactor unrelated code.
