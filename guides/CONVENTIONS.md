# Guide Authoring Conventions
How every guide in this MCP server is structured so that guides stay modern, strict, extensive, and as small as possible by referencing — never duplicating — shared content.

---

> 📎 **This is a meta-guide.** It governs how guides are written. Authors MUST follow it; the agent does not need to fetch it to write user code.

## 1. The Prime Directive: Reference, Don't Duplicate

Every cross-cutting concern is **owned by exactly one canonical guide**. Any other guide that touches that concern MUST link to the owner and add only its own **specialization** — it MUST NOT restate the owner's rules.

- A rule lives in **one** place. Change it once, every guide inherits the change.
- A guide's tokens are spent on what is **unique** to its technology, not on re-explaining TDD/security/logging for the Nth time.
- "Could this paragraph be replaced by a reference to another guide?" — if yes, replace it.

### Canonical ownership map

| Concern | Canonical owner |
|---|---|
| Test-first / Red-Green-Refactor / coverage | `tdd.md` |
| Mutation testing | `mutmut.md` |
| Hexagonal / Ports & Adapters | `hexagonal.md` |
| Clean Architecture | `cleanarch.md` |
| Architecture styles (overview) | `architectures.md` |
| Microservices | `microservices.md` |
| Design patterns (GoF & friends) | `designpatterns.md` |
| Security, supply chain, secrets, CVEs | `secure-coding.md` |
| Error-handling strategy | `error-handling.md` |
| Structured logging | `logging.md` |
| Metrics, tracing, observability | `observability.md` |
| Code comments / API docs | `comments.md` |
| Markdown authoring | `markdown.md` |
| Architecture Decision Records | `adr.md` |
| Configuration & environment | `env-config.md` |
| Semantic versioning | `semver.md` |
| Code review | `code-review.md` |
| CI/CD | `ci-cd.md` |
| Git workflow | `git.md` |
| Feature flags | `feature-flags.md` |
| Performance | `performance.md` |
| Concurrency / parallelism | `parallelism.md` |
| TODO conventions | `todo.md` |
| Pre-commit hooks | `pre-commit.md` |
| API styles | `rest.md`, `graphql.md`, `grpc.md`, `openapi.md`, `websocket.md` |
| Auth | `oauth.md` |
| Accessibility | `accessibility.md` |
| End-to-end testing | `e2e-testing.md` |

> A **language/framework/datastore/infra** guide owns only what is unique to *that* technology (syntax, idioms, toolchain, ecosystem libraries). It NEVER owns a cross-cutting concern.

## 2. Reference Syntax (agent-actionable)

References use a fixed, machine-recognizable form so the agent reliably issues a `get_guide` / reads the `guides://` resource. Three strengths:

```markdown
> 📎 **REQUIRED — fetch & apply before writing code:** [`tdd.md`](guides://tdd.md)
> 📎 **RECOMMENDED — fetch if the task touches logging:** [`logging.md`](guides://logging.md)
> 📎 **SEE ALSO:** [`designpatterns.md`](guides://designpatterns.md)
```

- **REQUIRED** — a hard prerequisite. The agent MUST read it before generating code. Mirrors `requires:` in frontmatter.
- **RECOMMENDED** — fetch when the task involves that concern. Mirrors `recommends:`.
- **SEE ALSO** — optional depth; not auto-fetched.

Inline, mid-sentence references use a plain link: "...handle errors per [`error-handling.md`](guides://error-handling.md)." Every reference target MUST be the canonical owner from §1.

### The balanced rule (how much may a guide restate?)

When specializing a referenced concern, a guide MAY include **one sentence** reminding the reader of the rule, then the reference, then the technology-specific binding. It MUST NOT reproduce examples, checklists, or multi-paragraph explanations that live in the owner.

✅ Allowed:
> Tests are written first (Red-Green-Refactor — see [`tdd.md`](guides://tdd.md)).
> In Python the runner is `uv run pytest`; the coverage gate is `pytest --cov --cov-fail-under=100`.

❌ Not allowed: re-explaining what Red-Green-Refactor is, re-pasting the bug-fix workflow, re-listing generic security checks.

## 3. Frontmatter (machine-readable metadata)

> ⚠️ **Server constraint:** `extract_brief()` in `coding_guides_server/server.py` reads the brief from the line(s) **after the `# Title` and before the first `---`**. Therefore the **title + one-line brief come first**, and the metadata block follows. Do **not** put a YAML block at the very top of the file or the guide listing will show "No description available." (If the server is later patched to skip a leading frontmatter block, this ordering can change.)

```markdown
# Python Development Guidelines
Mandatory coding standards for Python: type-safe, documented, test-covered. Python 3.13+, uv, pytest, ruff, bandit, safety.

---
name: python
title: Python Development Guidelines
version: 2.0
last_reviewed: 2026-06-05
kind: language            # language | framework | datastore | infra | cross-cutting | tooling | meta
tools: [python@3.13, uv, pytest, ruff, bandit, safety]
requires:                 # REQUIRED references — fetched & applied; never duplicated here
  - tdd
  - hexagonal
  - secure-coding
  - error-handling
recommends:               # RECOMMENDED references — fetch when the task touches them
  - logging
  - observability
  - comments
  - semver
provides:                 # what THIS guide canonically owns (so others reference it, not copy)
  - pep8
  - type-hints
  - google-docstrings
---
```

`requires` + `recommends` form a walkable dependency graph: fetching one guide tells the agent in ~10 tokens which canonical guides to pull, instead of embedding their content. `name` values are guide filenames without `.md`.

## 4. Requirements: ID'd, RFC-2119, gated

Replace prose "MANDATORY" with a numbered, auditable requirements table. Each requirement is testable.

- **Keywords:** MUST / MUST NOT / SHOULD / SHOULD NOT / MAY (RFC 2119), uppercase.
- **ID:** `<TECH>-<TOPIC>-<NN>`, e.g. `PY-TST-01`. Topics: `TST` test, `TYP` types, `SEC` security, `DEP` dependencies, `FMT` format, `LINT` lint, `DOC` docs, `ARCH` architecture, `ERR` errors, `LOG` logging, `OBS` observability, `STRUCT` structure, `PERF` performance.
- Each row has a **Verify** command and a binary **Gate**.
- A requirement that merely binds a cross-cutting rule cites the owner in its text (e.g. "(see `tdd.md`)").

```markdown
| ID | Requirement | Verify | Gate |
|----|-------------|--------|------|
| PY-TST-01 | Every feature MUST be test-first (see `tdd.md`) | `uv run pytest` | exit 0, 0 skips |
| PY-TYP-01 | Public APIs MUST be fully type-hinted | `mypy --strict src/` | exit 0 |
| PY-SEC-01 | 0 high/critical CVEs (see `secure-coding.md`) | `safety check` | 0 high/critical |
```

The **Deployment Checklist** at the end of a guide is generated from these IDs — it does not invent new requirements.

## 5. Style

- **Modern:** pin current tool versions in `tools:`; set `last_reviewed`; prefer current idioms.
- **Strict:** RFC-2119 keywords; binary gates; no "consider" where a rule is meant.
- **Extensive but compact:** cover the technology's unique surface thoroughly; push everything shared to a reference. Code blocks illustrate the *technology*, not the *concept*.
- **No dead references:** every `guides://x.md` target must exist. New cross-cutting concern → create/extend its canonical owner, then reference it.

---
**End of Guide Authoring Conventions**
