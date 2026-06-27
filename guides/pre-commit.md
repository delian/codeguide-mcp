# Pre-commit Framework Guidelines
Mandatory standards for fast, deterministic local git gates with the pre-commit framework: hook configuration, staged-file checks, secret scanning at commit time, and keeping local hooks in sync with CI. pre-commit 4.x, language-agnostic.

---
name: pre-commit
title: Pre-commit Framework Guidelines
version: 2.0
last_reviewed: 2026-06-05
kind: cross-cutting
tools: [pre-commit@4]
requires: []
recommends:
  - git
  - ci-cd
  - secure-coding
  - code-review
provides:
  - pre-commit-hooks
  - local-gates
  - hook-config
---

> 🧭 Authored per [`CONVENTIONS.md`](guides://CONVENTIONS.md): shared concerns are referenced, not restated. This guide owns the pre-commit framework and local git-hook gates; the *policy* behind each gate lives in its owner guide.

---

## 0. Prerequisites & References

This guide has no hard prerequisites. Fetch these when the task touches them — they own concerns this guide only *wires up*, and their rules are not repeated here.

> 📎 **RECOMMENDED — fetch when the task touches them:**
> - [`git.md`](guides://git.md) — the git-hook mechanism (`core.hooksPath`, hook lifecycle, branch policy). *(Binding: pre-commit installs into `.git/hooks/` or a managed hooks path.)*
> - [`ci-cd.md`](guides://ci-cd.md) — the same gates run server-side; this guide keeps local and CI configs identical. *(Binding: CI runs `pre-commit run --all-files`.)*
> - [`secure-coding.md`](guides://secure-coding.md) — secret-scanning and supply-chain policy. *(Binding: this guide adds the commit-time secret-scan hook.)*
> - [`code-review.md`](guides://code-review.md) — what humans review after the automated gate passes.

> 📎 **SEE ALSO:** the per-tool guides supply the *hook IDs and versions* a project pins — e.g. [`python.md`](guides://python.md) (ruff, bandit), [`typescript.md`](guides://typescript.md) / [`javascript.md`](guides://javascript.md) (eslint, prettier), [`go.md`](guides://go.md), [`rust.md`](guides://rust.md), [`dockerfile.md`](guides://dockerfile.md) (hadolint), [`bash.md`](guides://bash.md) (shellcheck), [`markdown.md`](guides://markdown.md), [`terraform.md`](guides://terraform.md). This guide does **not** restate their tool configs.

---

## 1. Core Philosophies

Principles unique to a *local commit-time gate*. Security, CI, git, and review policy come from §0.

- **Local mirror of CI, not a replacement.** Every gate that runs in CI MUST be runnable locally via the same `.pre-commit-config.yaml`; CI re-runs the identical hooks so a green local commit means a green pipeline. The gate is duplicated *for fast feedback*, never *for different rules* (see `ci-cd.md`).
- **Fast or it gets disabled.** Hooks on the `pre-commit` stage MUST be near-instant on a typical diff. Anything slow (full test suite, dependency audit, integration checks) moves to the `pre-push` stage or CI only. A hook that adds seconds to every commit will be bypassed by developers.
- **Staged-files only by default.** Hooks act on the **staged** files pre-commit passes them, not the whole tree. Whole-tree runs are reserved for `--all-files` (CI / first install). Hooks that ignore their file arguments (`pass_filenames: false`) MUST be the exception and MUST be scoped with `files:`/`types:`.
- **Deterministic & pinned.** Every `repo` is pinned to an immutable `rev` (tag/SHA). `pre-commit autoupdate` is the only way revs change, and that change is reviewed like any dependency bump.
- **Auto-fix, then re-verify.** Formatters and fixers modify files and exit non-zero so the commit aborts; the developer re-stages and re-commits. The gate never silently rewrites a passing commit.
- **Bypass is auditable.** `--no-verify` / `SKIP=` exist for genuine emergencies only and are surfaced in review; the server-side gate (`ci-cd.md`) catches anything bypassed locally.

**Verified Code**: code is not "done" until `pre-commit run --all-files` is clean and the same hooks pass in CI.

---

## 2. Requirements (MANDATORY, auditable)

RFC-2119 keywords. IDs `PC-<TOPIC>-<NN>`. Each row has a binary gate; rows binding a shared rule cite its owner.

| ID | Requirement | Verify | Gate |
|----|-------------|--------|------|
| PC-STRUCT-01 | A `.pre-commit-config.yaml` MUST exist at repo root and be valid | `pre-commit validate-config` | exit 0 |
| PC-STRUCT-02 | Hooks MUST be installed into git (commit + commit-msg + pre-push) | `pre-commit install -t pre-commit -t commit-msg -t pre-push` then inspect git hooks (see `git.md`) | hooks present |
| PC-DEP-01 | Every `repo` MUST pin an immutable `rev`; none floating/`HEAD` | `grep -nE '^\s*rev:' .pre-commit-config.yaml` review | all pinned |
| PC-CFG-01 | Whole-suite run MUST pass clean | `pre-commit run --all-files` | exit 0 |
| PC-SEC-01 | A commit-time secret scan MUST run on staged files (see `secure-coding.md`) | `pre-commit run gitleaks --all-files` (or detect-secrets) | 0 findings |
| PC-PERF-01 | `pre-commit`-stage hooks MUST be fast; slow checks live on `pre-push`/CI | review stage assignments; time a representative run | commit-stage run quick |
| PC-CI-01 | CI MUST run the identical config server-side (see `ci-cd.md`) | CI job runs `pre-commit run --all-files --show-diff-on-failure` | gate blocks merge |
| PC-SYNC-01 | Local hooks MUST match the project's lint/format/type/test gates (no drift) | compare hook IDs to language guide's §2 commands | no missing gate |
| PC-BRANCH-01 | Direct commits to protected branches MUST be blocked locally (see `git.md`) | `no-commit-to-branch` hook present | configured |
| PC-MSG-01 | Commit messages SHOULD be validated against the project convention | `conventional-pre-commit` (or commitlint) on `commit-msg` | exit 0 |
| PC-MNT-01 | Hook revs MUST be updated and reviewed on a schedule | `pre-commit autoupdate` in a reviewed PR | revs current |

> **Forbidden**: floating revs; slow hooks on the commit stage; secret-scan disabled; local config that diverges from CI; routine use of `--no-verify`/`SKIP` to land code.

---

## 3. Verification Protocol

Run before presenting any change that touches hooks or before a commit:

```bash
pre-commit validate-config                       # PC-STRUCT-01
pre-commit run --all-files --show-diff-on-failure # PC-CFG-01 (mirrors CI, PC-CI-01)
pre-commit run gitleaks --all-files              # PC-SEC-01 (secret scan)
```

If a hook fails: read the output, fix the root cause (or re-stage auto-fixes), re-run until green. Do not bypass. The *why* behind each gate lives in its §0 owner.

---

## 4. Config Anatomy

`.pre-commit-config.yaml` lives at repo root. The shape — top-level keys, stages, and selectors — is what this guide owns; the specific tool hooks come from each language/tool guide.

```yaml
# .pre-commit-config.yaml
minimum_pre_commit_version: "4.0.0"

default_install_hook_types: [pre-commit, commit-msg, pre-push]
default_stages: [pre-commit]        # modern stage name (NOT the legacy "commit")
fail_fast: false                     # run all hooks; report every failure at once

# Exclude generated/vendored paths once, globally (verbose regex).
exclude: |
  (?x)^(
    .*/node_modules/|
    dist/|build/|vendor/|
    .*\.min\.(js|css)$|
    .*-lock\.(json|yaml)$|uv\.lock$
  )

repos:
  - repo: https://github.com/<owner>/<hook-repo>
    rev: vX.Y.Z                       # PC-DEP-01: pinned, immutable
    hooks:
      - id: <hook-id>
        files: \.py$                  # narrow with files: and/or types:
        types_or: [python, pyi]
        stages: [pre-commit]          # or [pre-push] for slow hooks
```

Key selectors and what they mean:

| Key | Purpose |
|-----|---------|
| `files` / `exclude` | regex on the staged path; scope a hook to where it applies |
| `types` / `types_or` / `exclude_types` | identify-based file typing (`python`, `yaml`, `dockerfile`) — prefer over fragile path regexes |
| `stages` | when the hook runs: `pre-commit`, `commit-msg`, `pre-push`, `manual` |
| `pass_filenames: false` | hook ignores file args (whole-project tools); pair with a tight `files:`/`types:` trigger |
| `args` | static flags passed before filenames |
| `additional_dependencies` | extra packages installed into the hook's isolated env (plugins, type stubs) |
| `alias` | a name to invoke a specific hook with `pre-commit run <alias>` |

Top-level keys worth knowing: `minimum_pre_commit_version`, `default_language_version`, `default_install_hook_types`, `default_stages`, `fail_fast`, `exclude`, `ci` (for the pre-commit.ci hosted autofix/autoupdate service).

> Stage names modernized in pre-commit 3.2+/4.x: use `pre-commit`, `pre-push`, `pre-merge-commit`, `commit-msg`, `post-checkout`, `manual`. The bare `commit`/`push` aliases are deprecated — do not author new configs with them.

---

## 5. Hook Sources & Categories

Three hook source types, in order of preference:

1. **Published repo + `rev`** — versioned, isolated env, reproducible. Default choice. (`astral-sh/ruff-pre-commit`, `gitleaks/gitleaks`, `pre-commit/pre-commit-hooks`, `hadolint`, `shellcheck-py`, …)
2. **`repo: meta`** — pre-commit's built-in self-checks (`check-hooks-apply`, `check-useless-excludes`, `identity`). Cheap config hygiene; include them.
3. **`repo: local`** — project commands run from the dev environment (`language: system` / `script` / `docker_image`). Use when the project must run *its own pinned* toolchain (e.g. `uv run mypy`, `cargo clippy`) rather than a hook author's copy, so the version matches CI exactly.

Functional categories the gate should cover (order: cheapest/most-critical first). For each, the **hook ID and version come from the relevant language/tool guide** — do not hardcode them here:

- **Hygiene** — `pre-commit/pre-commit-hooks`: `trailing-whitespace`, `end-of-file-fixer`, `check-merge-conflict`, `check-added-large-files`, `check-yaml`/`-json`/`-toml`, `mixed-line-ending`, `detect-private-key`, `no-commit-to-branch` (PC-BRANCH-01).
- **Secrets** (PC-SEC-01, policy → `secure-coding.md`) — `gitleaks` or Yelp `detect-secrets` (with a committed `.secrets.baseline`). Always present.
- **Format** — the project formatter as a fixer (e.g. ruff-format, prettier, gofmt, rustfmt). See the language guide.
- **Lint** — the project linter (ruff, eslint, golangci-lint, clippy, shellcheck, hadolint, markdownlint). See the language guide.
- **Types** — type checker, usually `repo: local` so the version matches the project (mypy/pyright/tsc).
- **Commit message** (PC-MSG-01) — `conventional-pre-commit` or `commitlint` on the `commit-msg` stage.
- **Slow checks** (`pre-push` / CI only) — full test suite, dependency/CVE audit, doc-coverage. Never on the commit stage (PC-PERF-01).

---

## 6. Staging, Speed & Stage Assignment

How pre-commit interacts with the index is the core mechanic this guide owns:

- pre-commit runs against a **clean checkout of the staged content**: it stashes unstaged changes, runs hooks on what is staged, restores the stash. Only staged work is gated; a partially staged file is checked as staged, not as on disk.
- A fixer hook that modifies a file makes pre-commit **fail the commit**; the developer reviews the change, `git add`s it, and re-commits. This is intentional — never auto-`git add` fixer output inside a hook.
- Assign every hook a stage deliberately:

| Stage | Runs | Put here |
|-------|------|----------|
| `pre-commit` | every `git commit` | format, lint, type, hygiene, secret scan — fast, staged-files |
| `commit-msg` | after message entered | message-convention validation |
| `pre-push` | every `git push` | full unit suite, dependency audit, slower whole-repo checks |
| `manual` | only `pre-commit run --hook-stage manual <id>` | expensive opt-in tools |

Performance levers: scope with `files:`/`types:` so hooks no-op on irrelevant diffs; `exclude` generated/vendored paths globally; keep heavy work off the commit stage; prefer fast tools (ruff over flake8+isort, biome/prettier as configured by the language guide). `fail_fast: true` only when you want the first failure to stop the run — usually keep it `false` so one commit surfaces all problems.

---

## 7. Keeping Local Hooks in Sync With CI

The local gate and the server gate MUST be the **same file** producing the **same result** (PC-CI-01, PC-SYNC-01). Policy for the pipeline itself lives in [`ci-cd.md`](guides://ci-cd.md); the binding:

- CI installs pre-commit and runs `pre-commit run --all-files --show-diff-on-failure` against the checkout. A failure blocks the merge. This is the authority — local hooks are the fast preview, CI is enforcement.
- Cache the pre-commit env in CI keyed on `.pre-commit-config.yaml` so hook environments are not rebuilt every run.
- The hosted **pre-commit.ci** service (configured via the top-level `ci:` key) can auto-fix PRs and run weekly `autoupdate`; if used, it replaces a hand-written CI job but the rule is identical — same config, server-side enforcement.
- Drift detection: any lint/format/type/test gate listed in a language guide's §2 MUST have a corresponding hook (PC-SYNC-01). When CI gains a check, add the hook; when a hook changes, CI inherits it because it runs the same config.

```yaml
# Example CI invocation (engine-agnostic; see ci-cd.md / the engine's guide)
#   install:  pip install pre-commit   (or pipx/uv tool install)
#   run:      pre-commit run --all-files --show-diff-on-failure
#   cache:    key = hash(.pre-commit-config.yaml), path = ~/.cache/pre-commit
```

---

## 8. Lifecycle & Maintenance

```bash
# Install (idempotent; uses default_install_hook_types from config)
pre-commit install

# Or install specific hook types explicitly (PC-STRUCT-02)
pre-commit install -t pre-commit -t commit-msg -t pre-push

# First run / CI / whole-tree audit
pre-commit run --all-files --show-diff-on-failure   # PC-CFG-01

# Targeted runs
pre-commit run <hook-id>                 # one hook, staged files
pre-commit run --files path/to/file      # specific files
pre-commit run --hook-stage pre-push     # a stage's hooks

# Maintenance
pre-commit autoupdate                    # bump revs to latest tags (PC-MNT-01) — review the diff
pre-commit gc                            # drop unused cached hook envs
pre-commit clean                         # wipe the cache (force re-install of envs)
pre-commit validate-config               # PC-STRUCT-01
pre-commit validate-manifest             # validate a hook repo's .pre-commit-hooks.yaml

# Emergency bypass (auditable, discouraged — CI still enforces)
git commit --no-verify                   # skip all hooks
SKIP=mypy,pytest git commit              # skip named hooks
```

Maintenance rules: run `autoupdate` on a schedule and review the rev bumps like dependency upgrades; include `meta` hooks (`check-useless-excludes`, `check-hooks-apply`) so dead config is caught; secret-scan baselines (`.secrets.baseline`) are committed and refreshed deliberately, never to hide a real leak (see `secure-coding.md`).

---

## 9. Common Footguns

- **Floating revs** (`rev: HEAD` / branch name) → non-reproducible, breaks silently on upstream changes. Always pin a tag/SHA (PC-DEP-01).
- **Slow hook on the commit stage** → developers add `--no-verify` and the gate dies. Move it to `pre-push`/CI (PC-PERF-01).
- **`pass_filenames: false` without a `files:` trigger** → the hook runs on *every* commit regardless of relevance. Always scope the trigger.
- **Auto-`git add` inside a fixer hook** → silently rewrites the developer's intended commit. Let the hook fail and have the human re-stage.
- **Legacy stage names** (`stages: [commit]`/`[push]`) → deprecated; use `pre-commit`/`pre-push`.
- **Hooks that drift from CI** → "passes locally, fails in CI." Single config, both sides run it (PC-SYNC-01).
- **Hook version ≠ project version** (e.g. a ruff-pre-commit rev newer than the project's pinned ruff) → use `repo: local` with the project's own tool when exact-version parity matters.
- **Committing the secret-scan baseline to mask a leak** → forbidden; a baseline records known false positives, not real secrets.

---

## 10. Deployment Checklist

Generated from §2 — one box per requirement ID.

- [ ] PC-STRUCT-01 — valid `.pre-commit-config.yaml` at root
- [ ] PC-STRUCT-02 — hooks installed (pre-commit, commit-msg, pre-push)
- [ ] PC-DEP-01 — every repo pins an immutable `rev`
- [ ] PC-CFG-01 — `pre-commit run --all-files` clean
- [ ] PC-SEC-01 — commit-time secret scan present and clean (see `secure-coding.md`)
- [ ] PC-PERF-01 — commit-stage hooks fast; slow checks on pre-push/CI
- [ ] PC-CI-01 — CI runs the identical config and blocks merge (see `ci-cd.md`)
- [ ] PC-SYNC-01 — local hooks match the project's lint/format/type/test gates
- [ ] PC-BRANCH-01 — direct commits to protected branches blocked (see `git.md`)
- [ ] PC-MSG-01 — commit-message convention validated
- [ ] PC-MNT-01 — revs updated via reviewed `autoupdate`
- [ ] Agent ran every §3 command and documented any fixes

---
**End of Pre-commit Framework Guidelines**
