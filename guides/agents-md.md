# AGENTS.md Guidelines
Mandatory standards for authoring and maintaining AGENTS.md — the repo-root instruction file that tells AI coding agents how to build, test, and conform to a project.

---
name: agents-md
title: AGENTS.md Guidelines
version: 2.0
last_reviewed: 2026-06-05
kind: cross-cutting
tools: []
requires: []
recommends:
  - coding-ai
  - markdown
  - comments
provides:
  - agents-md-format
  - agent-instructions
  - repo-conventions-file
---

> 🧭 Authored per [`CONVENTIONS.md`](guides://CONVENTIONS.md): shared concerns are referenced, not restated. This guide owns the AGENTS.md file convention — its structure, placement, scoping, content, and upkeep. How agents *consume* it is owned by [`coding-ai.md`](guides://coding-ai.md); how to *write* its prose is owned by [`markdown.md`](guides://markdown.md).

---

## 0. Prerequisites & References

> 📎 **RECOMMENDED — fetch when the task touches them:**
> - [`coding-ai.md`](guides://coding-ai.md) — how agents discover, read, and obey AGENTS.md; context-window discipline. *(AGENTS.md binding: this guide defines the file those rules consume.)*
> - [`markdown.md`](guides://markdown.md) — Markdown authoring: headings, tables, fenced code, link syntax used throughout AGENTS.md.
> - [`comments.md`](guides://comments.md) — code-level guidance; AGENTS.md states *repo-wide* conventions, not per-line comments.

> 📎 **SEE ALSO:** [`todo.md`](guides://todo.md) — task/state tracking (TODOS.md) referenced from AGENTS.md · [`tdd.md`](guides://tdd.md) — test-first policy AGENTS.md points at · [`git.md`](guides://git.md) · [`ci-cd.md`](guides://ci-cd.md) · [`adr.md`](guides://adr.md).

> **Scope note.** AGENTS.md is a *pointer and convention* file, not a copy of every policy. It names the stack, the commands, the do/don't rules, and links the canonical guides (TDD, architecture, language). It MUST NOT re-explain those guides — restating `tdd.md` inside AGENTS.md is the exact duplication this server forbids.

---

## 1. Core Philosophies

AGENTS.md-specific principles only. Test/architecture/logging policy is *referenced* from AGENTS.md, never defined in it.

- **Single source of truth at the root.** One `AGENTS.md` at the repo root is the first thing an agent reads. It is the contract; the agent's training priors lose to it.
- **Pointer, not encyclopedia.** State *what is true for this repo* (stack, commands, layout, conventions) and *link* the canonical guides. Keep it short enough to fit comfortably in context.
- **Executable over prose.** Every claim an agent must act on is a copy-pasteable command (`npm test`, `uv run pytest`) or a binary rule ("never edit `generated/`"), not a paragraph.
- **Current or deleted.** Stale instructions are worse than none — an agent will follow a wrong command confidently. Out-of-date content MUST be fixed or removed, not left "for history".
- **Scoped to where it applies.** Monorepo or sub-package specifics live in a nested AGENTS.md beside the code they govern (see §5).
- **Standard, not bespoke.** `AGENTS.md` is the cross-tool convention; do not invent per-tool filenames. Symlink legacy tool files to it (see §6).

---

## 2. Requirements (MANDATORY, auditable)

RFC-2119 keywords. IDs `AGT-<TOPIC>-<NN>`. Each row has a binary gate; rows binding a shared rule cite its owner.

| ID | Requirement | Verify | Gate |
|----|-------------|--------|------|
| AGT-STRUCT-01 | An `AGENTS.md` MUST exist at the repo root | `test -f AGENTS.md` | exit 0 |
| AGT-STRUCT-02 | It MUST contain the core sections: Overview, Setup/Build, Test, Conventions, Do/Don't (see §4) | `grep -Ei 'setup\|build\|test\|convention'` | all present |
| AGT-CMD-01 | Every build/test/lint command listed MUST actually run from a clean checkout | run each command | exit 0 |
| AGT-CMD-02 | Commands MUST be copy-pasteable (no `<placeholder>` left unfilled) | review / grep `<.*>` | no unresolved placeholders |
| AGT-REF-01 | Cross-cutting policy (TDD, architecture, language) MUST be *linked*, not restated (see `CONVENTIONS.md`) | review | no duplicated policy prose |
| AGT-REF-02 | Every referenced guide/file path MUST resolve (no dead links) | review / link check | all resolve |
| AGT-SCOPE-01 | Sub-package-specific rules MUST live in a nested AGENTS.md, not the root one | review | scoped correctly |
| AGT-FRESH-01 | AGENTS.md MUST be updated in the same change that alters build/test/layout it documents | review diff / CI | no stale commands |
| AGT-MD-01 | File MUST be valid Markdown (see `markdown.md`) | markdown linter | exit 0 |
| AGT-SIZE-01 | Root AGENTS.md SHOULD stay concise (≈ ≤ 300 lines); overflow → nested files or linked guides | `wc -l AGENTS.md` | within budget |

> **Forbidden**: copying the body of `tdd.md`/architecture/language guides into AGENTS.md; listing commands that don't run; leaving stale commands after a tooling change; inventing a bespoke per-tool instruction filename instead of `AGENTS.md`.

---

## 3. Verification Protocol

Run before committing AGENTS.md changes. Fix → re-run until clean.

```bash
test -f AGENTS.md                              # AGT-STRUCT-01
grep -Eiq 'test' AGENTS.md                     # AGT-STRUCT-02 (repeat per core section)
# AGT-CMD-01: from a clean checkout, execute each fenced setup/build/test command
grep -n '<[a-zA-Z].*>' AGENTS.md || true       # AGT-CMD-02: no unresolved placeholders
markdownlint AGENTS.md                          # AGT-MD-01 (see markdown.md)
wc -l AGENTS.md                                 # AGT-SIZE-01
```

The *why* behind referenced policies (why test-first, why this architecture) lives in their canonical guides; AGENTS.md only links them.

---

## 4. AGENTS.md Structure & Content (the owned core)

The heart of this guide. AGENTS.md is plain Markdown read top-to-bottom by an agent; keep the most load-bearing rules early. Recommended sections, in order:

### A. Header & overview
One or two sentences: what the project is, primary language(s)/framework, and the one rule that matters most ("100% coverage on `core/`", "never touch `vendor/`").

```markdown
# AGENTS.md
Payments service (Go 1.23, hexagonal). Domain in `internal/domain` must stay free of framework imports.
```

### B. Setup, build & run
Exact commands to get a working environment and build/run the app. Prefer the project's real toolchain over generic advice.

```markdown
## Setup & Build
- Install:   `make bootstrap`        # installs toolchain + deps
- Build:     `make build`
- Run:       `make run`              # serves on :8080
- Generate:  `make gen`              # regenerate `internal/gen/` after schema edits
```

### C. Test
How to run tests and what the coverage/quality gate is. Reference [`tdd.md`](guides://tdd.md) for the *policy* (test-first, regression-test-before-fix); list only the *commands* here.

```markdown
## Test
- All:       `make test`             # gate: exit 0, no skips
- One pkg:   `go test ./internal/domain/...`
- Coverage:  `make cover`            # gate: ≥ 90% on internal/
Test-first per tdd.md. Each bug gets a failing regression test before the fix.
```

### D. Conventions
Repo-wide rules an agent must follow: naming, formatting/lint command, import boundaries, commit style, branch policy. Point at the language/architecture guides instead of re-deriving them.

```markdown
## Conventions
- Format/lint: `make lint` must be clean before commit (see go.md).
- Architecture: hexagonal — domain imports nothing outward (see hexagonal.md).
- Commits: Conventional Commits; one logical change per commit (see git.md).
- Docs: exported symbols documented (see comments.md).
```

### E. Do / Don't
A short, blunt allow/deny list. This is where agents get the most leverage — concrete prohibitions prevent the most common wrong actions.

```markdown
## Do / Don't
- DO run `make test && make lint` before presenting any change.
- DO update this file in the same PR when build/test commands change.
- DON'T edit `internal/gen/**` — regenerate with `make gen`.
- DON'T add dependencies without updating the lockfile.
- DON'T commit secrets; config comes from env (see env-config.md).
```

### F. Project map (optional, recommended for large repos)
A short tree pointing to where things live, so the agent navigates instead of guessing.

```markdown
## Layout
- `cmd/`            entrypoints
- `internal/domain` business logic (no framework imports)
- `internal/adapter` db/http adapters
- `docs/decisions`  ADRs (see adr.md)
```

### G. State / task tracking (optional)
If the project tracks live task/test state, keep that in `TODOS.md` and *link* it — do not grow AGENTS.md into a changelog. State/Kanban/TDD-phase tracking is owned by [`todo.md`](guides://todo.md).

```markdown
## State
Active tasks and TDD phase tracked in [TODOS.md](./TODOS.md) (see todo.md).
```

> Keep prose minimal: tables and bullet lists per [`markdown.md`](guides://markdown.md) read faster for both humans and agents than paragraphs.

---

## 5. Placement & Scoping

- **Root file is mandatory** (AGT-STRUCT-01) and covers repo-wide truth.
- **Nested AGENTS.md** sit beside the code they govern. An agent applies the *closest* file to the files it is editing; nearer files take precedence on conflicts and add package-specific commands.

```
repo/
├── AGENTS.md                 # repo-wide: stack, global conventions
├── services/api/AGENTS.md    # api-only: its build/test/run, framework specifics
└── packages/ui/AGENTS.md     # ui-only: component conventions, storybook command
```

- Put a rule at the **narrowest scope where it is true**. Global rules go to the root; "this service uses Postgres, run `make db-up` first" goes in that service's file.
- Don't duplicate the root file's content in children — children add or override, they don't restate.

---

## 6. Compatibility & Migration

`AGENTS.md` is the cross-tool standard. Where a tool historically reads its own filename, point it at AGENTS.md rather than maintaining parallel content:

```bash
ln -s AGENTS.md CLAUDE.md      # legacy/tool-specific name → single source
```

- Keep **one** canonical file; symlink or thin-include the rest. Two files that drift is worse than one.
- When migrating existing scattered instructions, consolidate into AGENTS.md, replace policy prose with links to canonical guides, then verify every command (AGT-CMD-01).

---

## 7. Keeping It Current

Stale instructions actively mislead agents (AGT-FRESH-01).

- **Co-change rule.** Any PR that changes a build/test/lint command, the directory layout, or a convention MUST update AGENTS.md in the same PR. Treat it like updating a test.
- **CI guard.** Add a CI step that runs the commands AGENTS.md advertises (or at least lints links and placeholders) so drift fails the build (see [`ci-cd.md`](guides://ci-cd.md)).
- **Periodic review.** On a cadence, run every fenced command from a clean checkout; delete or fix anything that no longer works.
- **No "history" cruft.** AGENTS.md is not a changelog or decision log — decisions go to ADRs (see [`adr.md`](guides://adr.md)), task state to TODOS.md (see [`todo.md`](guides://todo.md)).

---

## 8. Deployment Checklist

Generated from §2 — one box per requirement ID.

- [ ] AGT-STRUCT-01 — `AGENTS.md` exists at repo root
- [ ] AGT-STRUCT-02 — core sections present (overview, setup/build, test, conventions, do/don't)
- [ ] AGT-CMD-01/02 — every listed command runs from a clean checkout; no unresolved placeholders
- [ ] AGT-REF-01/02 — cross-cutting policy linked not restated; all references resolve
- [ ] AGT-SCOPE-01 — sub-package rules live in nested AGENTS.md
- [ ] AGT-FRESH-01 — file updated in the same change as the build/test/layout it documents
- [ ] AGT-MD-01 — valid Markdown (see `markdown.md`)
- [ ] AGT-SIZE-01 — root file concise; overflow pushed to nested files or linked guides

---
**End of AGENTS.md Guidelines**
