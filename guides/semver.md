# Semantic Versioning Guidelines
Mandatory standards for versioning, breaking-change policy, deprecation, changelogs, and commit-driven release automation per SemVer 2.0.0. semantic-release, commitizen, changesets, conventional-changelog, lerna/nx.

---
name: semver
title: Semantic Versioning Guidelines
version: 2.0
last_reviewed: 2026-06-05
kind: cross-cutting
tools: [semantic-release, commitizen, changesets, conventional-changelog, lerna, nx]
requires: []
recommends:
  - git
  - ci-cd
  - code-review
provides:
  - semver
  - breaking-change-policy
  - changelogs
  - conventional-commits
---

> 🧭 Authored per [`CONVENTIONS.md`](guides://CONVENTIONS.md): shared concerns are referenced, not restated. This guide owns versioning semantics, breaking-change/deprecation policy, the conventional-commits→version-bump mapping, and changelogs. It references `git.md` for tagging/branching mechanics and `ci-cd.md` for release automation.

---

## 0. Prerequisites & References

This is a cross-cutting guide with **no hard prerequisites**. Fetch the recommended guides when the task touches them.

> 📎 **RECOMMENDED — fetch when the task touches them:**
> - [`git.md`](guides://git.md) — commit-message hygiene, annotated tags, branching strategy. *(This guide owns the commit-type→version mapping; `git.md` owns the tagging/branching mechanics it triggers.)*
> - [`ci-cd.md`](guides://ci-cd.md) — the pipeline that runs the release: build, test gate, publish, tag push. *(This guide defines **what** the release does; `ci-cd.md` defines **where/how** it runs.)*
> - [`code-review.md`](guides://code-review.md) — reviewers gate breaking-change labels and changelog entries before merge.

> 📎 **SEE ALSO:** [`github.md`](guides://github.md) · [`gitlab.md`](guides://gitlab.md) — release/tag UIs and protected-branch rules.

Per-ecosystem version-range syntax and lockfile policy belong to the relevant language/datastore guide (e.g. [`nodejs.md`](guides://nodejs.md), [`python.md`](guides://python.md), [`rust.md`](guides://rust.md)) — this guide states the *policy*, not each ecosystem's `^`/`~`/`>=` spelling.

---

## 1. Core Philosophies: SEMVER-FIRST

Versioning-specific principles only. CI mechanics, git tagging, and review gates come from §0.

- **S**ingle source of truth: the version is **derived from the commit history**, not hand-edited. A human bumping a number by hand is a defect.
- **E**xplicit contract: a version number is a **promise** about API compatibility. The public API (exported symbols, wire formats, CLI flags, env contract, exit codes, on-disk/DB schema) is what is versioned — not internal code.
- **M**onotonic & immutable: a released version is never re-published with different bytes. Fix-forward with a new version; never reuse or move a tag.
- **V**isible impact: every release ships a changelog and (for MAJOR) a migration path. If a consumer can't tell what changed from the version + changelog, the release failed.
- **E**arly warning: nothing is removed without a prior deprecation in a MINOR release. Deprecate → warn → remove-on-MAJOR.
- **R**eproducible: anyone can regenerate the same version and notes from the same commits; releases run in CI (see `ci-cd.md`), not from a laptop.

**Verified Release**: No release ships until every gate in §2 is green.

---

## 2. Requirements (MANDATORY, auditable)

RFC-2119 keywords. IDs `SEMVER-<TOPIC>-<NN>`. Each row has a binary gate; rows binding a shared rule cite its owner.

| ID | Requirement | Verify | Gate |
|----|-------------|--------|------|
| SEMVER-VER-01 | Public version MUST be valid SemVer 2.0.0 `MAJOR.MINOR.PATCH[-prerelease][+build]` | regex `^\d+\.\d+\.\d+(-[0-9A-Za-z.-]+)?(\+[0-9A-Za-z.-]+)?$` | match |
| SEMVER-VER-02 | The released version MUST be derived from commit history, not hand-edited | `npx semantic-release --dry-run` (or tool dry-run) | computed bump == tag |
| SEMVER-BRK-01 | A breaking public-API change MUST bump MAJOR (or MINOR while 0.y.z) | API diff vs last tag (e.g. `api-extractor`, `cargo-semver-checks`, `revapi`) | no undeclared break |
| SEMVER-BRK-02 | Every breaking change MUST be flagged in its commit (`!` or `BREAKING CHANGE:` footer) | scan commits since last tag | breaks↔major consistent |
| SEMVER-DEP-01 | Nothing public MAY be removed without a prior deprecation in an earlier MINOR | grep removed symbols vs prior `@deprecated`/changelog | each removal pre-deprecated |
| SEMVER-CC-01 | Commits MUST follow Conventional Commits (see `git.md` for message hygiene) | `commitlint --from <last-tag>` | exit 0 |
| SEMVER-CHG-01 | A changelog MUST be updated for every release, Keep-a-Changelog format | release tool generates / diff `CHANGELOG.md` | entry present |
| SEMVER-CHG-02 | The MAJOR changelog entry MUST link a migration path for each breaking change | review `CHANGELOG.md` / release notes | link per break |
| SEMVER-TAG-01 | Each release MUST be an **annotated, signed** tag `vMAJOR.MINOR.PATCH` (see `git.md`) | `git tag -v vX.Y.Z` | valid signature |
| SEMVER-TAG-02 | A published version MUST NOT be re-published or its tag moved | registry immutability + `git tag` not force-updated | no overwrite |
| SEMVER-REL-01 | Releases MUST run in CI after the test gate passes (see `ci-cd.md`) | release job in pipeline | tag→build→publish atomic |

> **Forbidden**: hand-editing a version number, force-moving a release tag, re-publishing an existing version, removing public API without a prior deprecation, merging a breaking change without the `!`/`BREAKING CHANGE:` marker and a migration note, or releasing from a developer machine instead of CI.

---

## 3. Version Semantics

`MAJOR.MINOR.PATCH` — increment the **leftmost** that applies; reset everything to its right to 0.

| Bump | When (vs the **public API contract**) | Reset |
|------|----------------------------------------|-------|
| **MAJOR** `X.0.0` | Incompatible change: remove/rename a public symbol, change a signature/return type, change wire/serialization format, tighten input validation, change default behavior, drop a supported platform/runtime, raise a required dependency's floor across a major. | MINOR, PATCH → 0 |
| **MINOR** `x.Y.0` | Backward-compatible addition: new public symbol, new **optional** parameter, new opt-in config, new supported platform, marking something **deprecated** (still works). | PATCH → 0 |
| **PATCH** `x.y.Z` | Backward-compatible bug fix: correct wrong behavior to the documented contract, security patch with no API change, equivalent-behavior perf fix, internal refactor. | — |

The decision rule is one question: **"Could a consumer who upgrades without reading code be broken?"** Yes → MAJOR. New surface only → MINOR. Neither → PATCH.

> A **bug fix that changes documented behavior** is still MAJOR — restoring "correct" behavior can break consumers who depend on the old one. Document the trade-off in the changelog.

### 3.A `0.y.z` — Initial development
Anything MAY change at any time. Convention: bump **MINOR** for breaking changes, **PATCH** for everything else. Releasing `1.0.0` is the explicit commitment to stability — do it the moment the public API is something you'll stand behind. Avoid living on `0.x` indefinitely to dodge the compatibility contract.

### 3.B Pre-release & build metadata
`-prerelease` identifiers gate stability; `+build` is metadata only.

```
1.0.0-alpha < 1.0.0-alpha.1 < 1.0.0-beta < 1.0.0-rc.1 < 1.0.0
1.0.0+20260605   1.0.0-rc.1+sha.5114f85    # build metadata is IGNORED in precedence
```
- **alpha** — unstable, API may churn, internal.
- **beta** — feature-complete, API stabilizing, external testing.
- **rc** — release candidate; ship `X.Y.Z` if no blockers surface.

Precedence rules (SemVer §11): numeric identifiers compare numerically, alphanumeric lexically; **more** identifiers > a prefix-equal set with fewer; build metadata never affects ordering and two versions differing only in build metadata are **not** distinct releases.

---

## 4. Conventional Commits → Version Bump

This guide owns the **mapping**; `git.md` owns commit-message hygiene (format, scope, body wrapping, sign-off). The release tool reads the log and computes the bump — never decide it by hand.

| Commit | Bump |
|--------|------|
| `fix:` , `perf:` , `revert:` | **PATCH** |
| `feat:` | **MINOR** |
| any type with `!` (e.g. `feat!:`, `fix!:`) **or** a `BREAKING CHANGE:` footer | **MAJOR** |
| `docs:` `style:` `refactor:` `test:` `chore:` `ci:` `build:` | **none** (no release) |

```bash
feat(api): add cursor pagination to /users          # → MINOR

fix(auth): reject empty bearer tokens                # → PATCH

feat(api)!: return { data, meta } envelope           # → MAJOR
                                                     #   (! marks the break)

refactor(core): rename User → Account

BREAKING CHANGE: `User` is removed; import `Account`.   # → MAJOR via footer
Migration: see docs/migration-v3.md
```

- The bump for a release range is the **highest** triggered across all its commits.
- A `BREAKING CHANGE:` footer is MAJOR regardless of the commit type (even on a `fix:`).
- Enforce the convention in CI with `commitlint` (SEMVER-CC-01) and capture messages interactively with `commitizen` (`cz`) so contributors don't guess.

---

## 5. Breaking-Change & Deprecation Policy

This is the heart of the contract. **Removal is a two-release process: never remove in the same release you deprecate.**

### 5.A Deprecation lifecycle
1. **Deprecate (MINOR)** — keep the old API working; mark it deprecated in code, emit a one-time runtime warning, and document the replacement and the target removal version.
2. **Sustain** — the deprecated API keeps working for the rest of that MAJOR line; CI may warn but MUST NOT fail consumers.
3. **Remove (MAJOR)** — delete it in the next MAJOR, with a changelog entry and migration link (SEMVER-CHG-02).

Express the deprecation with each language's native idiom — `@deprecated` JSDoc/TSDoc, `warnings.warn(..., DeprecationWarning)`, `#[deprecated]`, `@Deprecated`, `Obsolete` — see the relevant language guide. Always state **what to use instead** and **when it disappears**, e.g. `@deprecated since 2.5.0 — use oauth2Auth(); removed in 3.0.0`.

### 5.B Communicating a breaking change
A break MUST be visible in three places, each progressively more detailed:
- **Commit** — the `!` marker and a `BREAKING CHANGE:` footer with a one-line migration (SEMVER-BRK-02).
- **Changelog** — under a `### BREAKING` / `### Removed` heading, with a link to the migration guide (SEMVER-CHG-02).
- **Migration guide** — before/after for each break, in `docs/migration-vN.md` or the GitHub/GitLab release notes.

### 5.C Reducing breakage
Prefer additive evolution over breaks: add a new method beside the old, accept the old shape via an adapter, version the surface (e.g. `/api/v2`, a new exported entry point) rather than mutating it in place. When a break is unavoidable, batch breaks into a single MAJOR rather than dribbling them out.

---

## 6. Changelogs

A release without a changelog is incomplete (SEMVER-CHG-01). Use **[Keep a Changelog](https://keepachangelog.com/en/1.1.0/)** sections, generated from commits — do not write them by hand.

```markdown
# Changelog
All notable changes are documented here.
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
and [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## [3.0.0] - 2026-03-01
### Added
- `oauth2Auth()` for token-based auth.
### Changed
- **BREAKING:** responses now use the `{ data, meta }` envelope.
### Removed
- **BREAKING:** `basicAuth()` removed — use `oauth2Auth()`. See [migration](./docs/migration-v3.md).
### Fixed
- Connection-pool leak under sustained load.
### Security
- Patched CVE-2026-XXXX in the transport dependency (see `secure-coding.md`).
```

Sections (in order): **Added, Changed, Deprecated, Removed, Fixed, Security**. Tag breaks with **BREAKING**. Keep an `[Unreleased]` heading at the top that the release step renames to the new version. The changelog is for **consumers** — describe impact and upgrade steps, not internal refactors.

---

## 7. Release Automation

The release is computed and executed by tooling in CI (SEMVER-VER-02, SEMVER-REL-01). This guide defines the policy; `ci-cd.md` owns the pipeline, and `git.md` owns the tagging mechanics.

**Pick one tool per repo:**
- **semantic-release** — fully automated: analyzes commits, bumps, generates notes/changelog, tags, publishes, creates the GitHub/GitLab release. Best for single-package, trunk-based repos.
- **changesets** — author-declared intent (`.changeset/*.md`) batched into a release PR; excellent for monorepos with **independent** package versions.
- **commit-and-tag-version / standard-version** — local bump + changelog + tag when you want a human to push the release.
- **lerna / nx** — monorepo orchestration (fixed or independent versioning) layered over the above.

```jsonc
// .releaserc — semantic-release, conventional-commits preset
{
  "branches": ["main", { "name": "next", "prerelease": "rc" }, { "name": "beta", "prerelease": true }],
  "plugins": [
    "@semantic-release/commit-analyzer",
    "@semantic-release/release-notes-generator",
    "@semantic-release/changelog",
    "@semantic-release/npm",
    ["@semantic-release/git", { "assets": ["CHANGELOG.md", "package.json"],
      "message": "chore(release): ${nextRelease.version} [skip ci]" }],
    "@semantic-release/github"
  ]
}
```

Release-flow rules:
- Branches map to channels: `main`→stable, `next`→`rc`, `beta`/`alpha`→pre-release (mirrors §3.B).
- The release job runs **after** the test/lint/security gates pass (see `ci-cd.md`) — a failing build MUST NOT tag or publish.
- Tag, build artifact, and registry publish are **one atomic step**; if publish fails, the tag is not pushed (SEMVER-TAG-02).
- Run `--dry-run` in PRs to surface the computed next version for reviewers (see `code-review.md`).
- Registries are immutable: npm/PyPI/crates/OCI reject re-publishing an existing version — rely on this rather than fighting it.

### 7.A Monorepos
Choose **fixed** (all packages share one version — simple, over-bumps untouched packages) or **independent** (per-package versions from per-package commit scopes — accurate, more bookkeeping). With changesets/lerna/nx, a package's version reflects only the commits that touch it; a breaking change in package A forces a MAJOR for A and a compatible bump for in-repo dependents.

---

## 8. Deployment Checklist

Generated from §2 — one box per requirement ID. No new requirements.

- [ ] SEMVER-VER-01 — version string is valid SemVer 2.0.0
- [ ] SEMVER-VER-02 — version derived from commits (tool dry-run matches), not hand-edited
- [ ] SEMVER-BRK-01 — API diff shows no undeclared break for the chosen bump
- [ ] SEMVER-BRK-02 — every breaking commit carries `!` / `BREAKING CHANGE:`
- [ ] SEMVER-DEP-01 — nothing removed without a prior MINOR deprecation
- [ ] SEMVER-CC-01 — `commitlint` clean since last tag
- [ ] SEMVER-CHG-01 — changelog updated (Keep a Changelog)
- [ ] SEMVER-CHG-02 — MAJOR entries link a migration path
- [ ] SEMVER-TAG-01 — annotated, signed `vX.Y.Z` tag (see `git.md`)
- [ ] SEMVER-TAG-02 — no re-published version / moved tag
- [ ] SEMVER-REL-01 — release ran in CI after the test gate (see `ci-cd.md`)

---
**End of Semantic Versioning Guidelines**
