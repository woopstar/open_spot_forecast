---
name: code-review
description: Activate when reviewing code changes, diffs, or pull requests to identify high-confidence bugs, regressions, security issues, and correctness problems.
---

# Code Review Skill

Activate this skill when the user asks for a code review, PR review, or diff review.

## Review Goals

- Focus on correctness, regressions, security, data loss, and broken tests.
- Report only high-confidence findings.
- Ignore style nits and trivial issues unless they hide a bug.
- Prefer the smallest actionable set of findings.

## Review Workflow

1. Read the diff and relevant surrounding code.
2. Trace call paths and invariants.
3. Check tests, types, and behavior changes.
4. Compare against repository conventions and docs when the change affects them.
5. Summarize findings with severity, rationale, and exact file and line references.

## OSF-Specific Checks

- If the change touches the ML model, feature vector, or prediction logic, read
  `docs/ml_documentation.md` and `docs/architecture.md` first.
- If the change touches self-learning, bias correction, or error metrics, read
  `docs/self_learning.md` first.
- If the change touches SQLite storage or schema migrations, read
  `docs/persistence.md` first.
- If the change touches price sources (Stromligning, Nordpool) or sensor wiring,
  read `docs/stromligning_integration.md` and `docs/using_existing_sensors.md` first.
- If the change affects PR workflow or release notes, check `.github/memories.md`
  and `docs/` for consistency.
- Verify affected tests exist or are updated.

## Output Format

- Start with an overall assessment.
- List findings sorted by severity.
- For each finding, include:
  - file path
  - line number or lines
  - concise explanation
  - why it matters
- If no issues are found, say so explicitly.