# [TECHNOLOGY_NAME] Development Guidelines
Mandatory coding standards for [TECHNOLOGY_NAME]: [SHORT_GOALS_LIST]. [TOOL_LIST_WITH_VERSIONS].

---
name: [technology_name_lowercase]
title: [TECHNOLOGY_NAME] Development Guidelines
version: 1.0
last_reviewed: [YYYY-MM-DD]
kind: [language | framework | datastore | infra | cross-cutting | tooling]
tools: [[tool@version], [tool@version]]
requires:                 # REQUIRED references — fetched & applied; their content is NEVER duplicated below
  - tdd
  - [architecture_owner e.g. hexagonal]
  - secure-coding
  - error-handling
recommends:               # RECOMMENDED references — fetch when the task touches them
  - logging
  - observability
  - comments
  - semver
provides:                 # what THIS guide canonically owns (others reference, not copy)
  - [owned_concern_1]
  - [owned_concern_2]
---

> 🧭 **Authoring rule:** This guide follows [`CONVENTIONS.md`](guides://CONVENTIONS.md). It references shared concerns instead of restating them, and spends its tokens on what is unique to [TECHNOLOGY_NAME]. Do not duplicate content owned by a `requires`/`recommends` guide.

---

## 0. Prerequisites & References

Fetch and apply these **before** generating [TECHNOLOGY_NAME] code. This guide assumes their rules and does not repeat them.

> 📎 **REQUIRED — fetch & apply first:**
> - [`tdd.md`](guides://tdd.md) — test-first, Red-Green-Refactor, regression tests, coverage.
> - [`[architecture_owner].md`](guides://[architecture_owner].md) — layering, ports/adapters, dependency inversion.
> - [`secure-coding.md`](guides://secure-coding.md) — vulnerability scanning, supply chain, secrets.
> - [`error-handling.md`](guides://error-handling.md) — error strategy and propagation.

> 📎 **RECOMMENDED — fetch when the task touches them:**
> - [`logging.md`](guides://logging.md) · [`observability.md`](guides://observability.md) · [`comments.md`](guides://comments.md) · [`semver.md`](guides://semver.md) · [`env-config.md`](guides://env-config.md)

> 📎 **SEE ALSO:** [`designpatterns.md`](guides://designpatterns.md) · [`code-review.md`](guides://code-review.md) · [`ci-cd.md`](guides://ci-cd.md)

---

## 1. Core Philosophies: [ACRONYM]-FIRST

[TECHNOLOGY_NAME]-specific principles only. Cross-cutting principles (TDD, security, error handling) come from §0 — do **not** restate them here.

- **[LETTER_1]**[rest]: [language/framework-specific principle]
- **[LETTER_2]**[rest]: [language/framework-specific principle]
- **[LETTER_3]**[rest]: [language/framework-specific principle]
- **[LETTER_4]**[rest]: [language/framework-specific principle]

**Verified Code**: Agent-generated code MUST pass every gate in §2 before delivery.

---

## 2. Requirements (MANDATORY, auditable)

Each requirement is ID'd, uses RFC-2119 keywords, and has a binary gate. Rows that bind a shared rule cite its owner. IDs use `[PREFIX]-<TOPIC>-<NN>` (see `CONVENTIONS.md` §4).

| ID | Requirement | Verify | Gate |
|----|-------------|--------|------|
| [PFX]-TST-01 | Every feature MUST be test-first (see `tdd.md`) | `[test_command]` | exit 0, 0 skips |
| [PFX]-TST-02 | Each bug MUST get a regression test before the fix (see `tdd.md`) | `[test_command]` | failing→passing |
| [PFX]-FMT-01 | Code MUST be formatted | `[format_check_command]` | no diff |
| [PFX]-LINT-01 | Linter MUST pass clean | `[lint_command]` | exit 0 |
| [PFX]-TYP-01 | [Public APIs MUST be typed / N/A] | `[type_check_command]` | exit 0 |
| [PFX]-SEC-01 | 0 high/critical CVEs (see `secure-coding.md`) | `[security_scan_command]` | 0 high/critical |
| [PFX]-DEP-01 | Lockfile in sync & verified (see `secure-coding.md`) | `[lock_verify_command]` | verified |
| [PFX]-DOC-01 | Public APIs documented (see `comments.md`) | `[doc_command]` | builds clean |
| [PFX]-ARCH-01 | Layering respected (see `[architecture_owner].md`) | review / `[dep_lint]` | no inward→outward deps |

> Forbidden — never deliver code that: fails any gate above, ships implementation before its test (violates `tdd.md`), or fixes a bug without a regression test first.

---

## 3. Verification Protocol

[TECHNOLOGY_NAME]-specific commands the agent runs before presenting code. The *policy* behind each (why test-first, why scan) lives in the §0 references.

```bash
[format_check_command]      # PFX-FMT-01
[lint_command]              # PFX-LINT-01
[type_check_command]        # PFX-TYP-01 (if applicable)
[test_command]              # PFX-TST-01
[security_scan_command]     # PFX-SEC-01
[lock_verify_command]       # PFX-DEP-01
```

If a gate fails: read the error, find the root cause, fix, re-run. Do not present until every gate is green.

---

## 4. Project Structure

[TECHNOLOGY_NAME]'s idiomatic layout. Architectural *principles* (ports/adapters, dependency direction, acyclic deps) are owned by [`[architecture_owner].md`](guides://[architecture_owner].md) — this section only shows where they map in a [TECHNOLOGY_NAME] project.

```
project/
├── [src_dir]/             # source; domain inward, adapters outward (see architecture owner)
├── [test_dir]/            # tests (see tdd.md)
├── [config_file]          # dependency / build manifest
└── README.md
```

- Group by feature/domain, not by type.
- Keep modules small and single-responsibility.
- No circular dependencies.

---

## 5. [TECHNOLOGY_NAME] Specifics

**This is the heart of the guide — the unique value.** Idioms, language/framework features, ecosystem libraries, footguns, and patterns that are specific to [TECHNOLOGY_NAME] and exist in no shared guide.

### A. [Idiom / feature 1]
```[language]
[example showing the TECHNOLOGY, not a generic concept]
```

### B. [Idiom / feature 2]
```[language]
[example]
```

### C. [Common footguns & how to avoid them]
- [footgun 1] → [fix]
- [footgun 2] → [fix]

> For design patterns applied here, reference [`designpatterns.md`](guides://designpatterns.md) and show only the [TECHNOLOGY_NAME] binding.

---

## 6. Tooling & Dependencies

[PACKAGE_MANAGER] specifics for [TECHNOLOGY_NAME]. Security/supply-chain *policy* is owned by [`secure-coding.md`](guides://secure-coding.md); versioning policy by [`semver.md`](guides://semver.md).

```bash
[sync_command]              # install/sync from lockfile
[add_dependency_command]    # add a dependency (updates lockfile)
[update_command]            # update to latest secure versions
[lock_verify_command]       # PFX-DEP-01: verify integrity
```

---

## 7. Quick Reference

```bash
[build_command]    # build
[test_command]     # test
[lint_command]     # lint
[format_command]   # format
[run_command]      # run
```

---

## 8. Deployment Checklist

Generated from §2 — one box per requirement ID. No new requirements here.

- [ ] PFX-FMT-01 — formatted, no diff
- [ ] PFX-LINT-01 — linter clean
- [ ] PFX-TYP-01 — types check (if applicable)
- [ ] PFX-TST-01 / TST-02 — all tests pass, bugs have regression tests
- [ ] PFX-SEC-01 — 0 high/critical CVEs
- [ ] PFX-DEP-01 — lockfile in sync & verified
- [ ] PFX-DOC-01 — public APIs documented
- [ ] PFX-ARCH-01 — layering respected
- [ ] Agent ran every §3 command and documented any fixes

---
**End of [TECHNOLOGY_NAME] Guidelines**
