# Architecture Decision Records (ADR) Guidelines
Mandatory standards for capturing architecture decisions as immutable, numbered, reviewable records. Markdown, adr-tools, MADR, Log4brains, Git.

---
name: adr
title: Architecture Decision Records (ADR) Guidelines
version: 2.0
last_reviewed: 2026-06-05
kind: cross-cutting
tools: [adr-tools, MADR, Log4brains]
requires: []
recommends:
  - markdown
  - comments
  - architectures
provides:
  - adr-format
  - adr-lifecycle
  - decision-records
---

> 🧭 Authored per [`CONVENTIONS.md`](guides://CONVENTIONS.md): shared concerns are referenced, not restated. This guide canonically **owns** the Architecture Decision Record discipline — format, lifecycle, numbering, storage, and the decision of when to write one. It does not re-teach Markdown authoring, the architectures it records, or how code comments link back to it.

---

## 0. Prerequisites & References

This guide is self-contained on ADR mechanics. The following are pulled in only when an ADR touches them.

> 📎 **RECOMMENDED — fetch when the task touches them:**
> - [`markdown.md`](guides://markdown.md) — ADRs are Markdown files; all authoring/formatting rules (headings, links, tables, lint) are owned there. *(ADR binding: one `#` H1, status as the second section.)*
> - [`architectures.md`](guides://architectures.md) — the architecture styles/patterns an ADR records (layered, event-driven, microservices, hexagonal, etc.). The ADR captures *the choice*; `architectures.md` explains *the options*.
> - [`comments.md`](guides://comments.md) — how to reference an ADR from source code/docstrings (e.g. `# See ADR-0012`). The linking convention is owned there.

> 📎 **SEE ALSO:** [`code-review.md`](guides://code-review.md) · [`git.md`](guides://git.md) · [`semver.md`](guides://semver.md) · [`designpatterns.md`](guides://designpatterns.md)

An ADR is a short Markdown document that captures **one** significant architectural decision together with its context and consequences. The collection of ADRs is the project's append-only **decision log**.

---

## 1. Core Philosophies: ADR-FIRST

ADR-specific principles only. Markdown style, the architectures being recorded, and code-linking come from §0.

- **A**ppend-only: an ADR is **immutable once Accepted**. You never edit the decision or rewrite history — you supersede it with a new ADR. Only the `Status` field (and links to superseding ADRs) may change after acceptance.
- **D**ecision-scoped: one ADR = one decision. If a document needs the word "and" to describe its decision, it is probably two ADRs.
- **R**ationale-bearing: record the **context and the reasoning**, not just the outcome. The value is the "why" and the rejected alternatives, so future readers don't re-litigate a settled question.
- **F**indable & numbered: every ADR has a monotonically increasing number, a stable filename, and an index entry. References use the `ADR-NNNN` form everywhere (code, PRs, other ADRs).
- **I**n-repo: ADRs live in version control next to the code they govern and ship through the same PR/review flow (see `code-review.md`, `git.md`) — not in a wiki that drifts.
- **R**eviewed-in-context: an ADR is `Proposed` in a PR, discussed, and only merged as `Accepted` after the agreed approval.

**Write an ADR when** a decision is *significant* and *hard to reverse*: it affects structure, dependencies, cross-team contracts, data/storage, security posture, or a public API/protocol; introduces or removes a major dependency or service boundary; or sets a project-wide convention. **Do not** write one for trivially reversible, local, or purely stylistic choices — note those inline or in code comments (see `comments.md`).

---

## 2. Requirements (MANDATORY, auditable)

RFC-2119 keywords. IDs `ADR-<TOPIC>-<NN>`. Each row has a binary gate; rows binding a shared rule cite its owner.

| ID | Requirement | Verify | Gate |
|----|-------------|--------|------|
| ADR-STRUCT-01 | Each ADR MUST be a single Markdown file named `NNNN-kebab-title.md`, zero-padded to 4 digits, in the ADR directory | `ls docs/**/decisions/` | all match `^[0-9]{4}-[a-z0-9-]+\.md$` |
| ADR-STRUCT-02 | ADR numbers MUST be unique and monotonic; numbers MUST NOT be reused or deleted | check for duplicate `NNNN` prefixes | 0 duplicates, 0 gaps from deletion |
| ADR-FMT-01 | Each ADR MUST contain at least `Title`, `Status`, `Date`, `Context`, `Decision`, `Consequences` sections (see `markdown.md` for formatting) | section grep / lint | all present |
| ADR-FMT-02 | Markdown MUST lint clean (see `markdown.md`) | `markdownlint docs/**/decisions/*.md` | exit 0 |
| ADR-LIFE-01 | Status MUST be one of `Proposed`, `Accepted`, `Rejected`, `Deprecated`, `Superseded` | grep `## Status` | value in set |
| ADR-LIFE-02 | An Accepted ADR's `Context`/`Decision`/`Consequences` MUST NOT be edited; changes go in a new superseding ADR | diff vs. accepted commit | only `Status`/links changed |
| ADR-LIFE-03 | A `Superseded` ADR MUST link to its superseding ADR, and the new ADR MUST link back (`Supersedes`) | grep both directions | bidirectional link present |
| ADR-DEC-01 | Each ADR MUST state ≥1 genuinely considered alternative and why it was rejected | review | non-empty Alternatives |
| ADR-IDX-01 | The decision-log index MUST list every ADR with id, title, status, date | regenerate & diff index | index matches files |
| ADR-REV-01 | Every new/changed ADR MUST go through PR review before merge (see `code-review.md`) | merged via PR | approved PR |

> **Forbidden**: editing the decision text of an Accepted ADR; deleting or renumbering an ADR; merging an ADR straight to `Accepted` without review; an ADR whose decision spans multiple unrelated choices; a dangling `Superseded` status with no link.

---

## 3. ADR Format

An ADR is small and skimmable — typically one screen. All Markdown rules (headings, tables, links, line length, lint) are owned by [`markdown.md`](guides://markdown.md); this section defines only the ADR-specific *shape*.

### A. Canonical template

```markdown
# ADR-NNNN: <Short imperative decision title>

## Status
Proposed            <!-- Proposed | Accepted | Rejected | Deprecated | Superseded by ADR-XXXX -->

## Date
YYYY-MM-DD          <!-- date of the latest status change -->

## Context
The forces at play: the problem, constraints (technical, business, team, regulatory),
and assumptions. State facts, not the decision. This is where future readers learn *why*.

## Decision
The choice, in active voice and full sentences: "We will …". Be specific about what is
in and out of scope. Name the architecture/pattern adopted and link the relevant guide
(see `architectures.md`).

## Consequences
What becomes easier and harder as a result — positive, negative, and neutral. Include
follow-on work and new risks. Honest trade-offs only; an ADR with no downsides is suspect.

## Alternatives Considered
Each realistic option that was rejected, with a one-line reason. Records the analysis so
the question isn't reopened later. (Required — see ADR-DEC-01.)

## References
Links to related ADRs (Supersedes / Superseded by / Related), RFCs, issues, and external sources.
```

Use the MADR (Markdown Any Decision Records) layout if the team prefers it — it adds optional `Deciders`, `Consulted`, and `Decision Drivers` fields. Pick **one** template and keep it consistent across the log.

### B. Short form

For low-ceremony decisions, a three-section form is acceptable: `Context` → `Decision` → `Consequences` (Nygard's original). Use the full template once a decision has non-trivial alternatives or cross-team impact.

### C. Title & writing rules

- Title is a **decision in imperative mood**: "Use PostgreSQL as the primary datastore", not "Database options".
- Keep prose tight: bullet trade-offs rather than essays. The depth lives in `Context` and `Alternatives`, not in restating what an architecture *is* — link [`architectures.md`](guides://architectures.md) instead of explaining event-driven/hexagonal/etc. from scratch.
- Frontmatter or a metadata block is optional; if the team's tooling (Log4brains, Docusaurus) needs it, keep it minimal.

---

## 4. Lifecycle & Immutability

This is the heart of the ADR discipline.

### A. Statuses

| Status | Meaning |
|--------|---------|
| **Proposed** | Drafted, under discussion in a PR. The only status that may be freely edited. |
| **Accepted** | Approved; implementation may proceed. The decision text is now **immutable**. |
| **Rejected** | Proposed but not adopted. Kept in the log — the analysis still has value. |
| **Deprecated** | Was Accepted, no longer recommended, but not replaced by a specific ADR. New code MUST NOT follow it. |
| **Superseded** | Replaced by a newer ADR. MUST link to the superseding ADR (ADR-LIFE-03). |

### B. Transitions

```
Proposed ──approve──▶ Accepted ──obsolete──▶ Deprecated
   │                     │
   └──reject──▶ Rejected └──replaced by ADR-XXXX──▶ Superseded
```

- **Proposed → Accepted / Rejected**: by the agreed approver(s) via PR review (ADR-REV-01).
- **Accepted → Superseded**: create a *new* ADR that states the replacement; set the old one to `Superseded by ADR-XXXX` and add the back-link. Never edit the old decision (ADR-LIFE-02).
- **Accepted → Deprecated**: when a decision is simply no longer valid and nothing replaces it 1:1.
- **Never delete or renumber.** The log is append-only; gaps and rewrites destroy the audit trail.

### C. Superseding (the only way to "change" a decision)

The new ADR carries a `## Supersedes` / `## References` link to `ADR-NNNN`; the old ADR's `## Status` becomes `Superseded by ADR-MMMM` with a link. Both directions are mandatory so the log is navigable from either end.

---

## 5. Numbering, Storage & Organization

### A. Numbering

- Sequential, zero-padded 4-digit integers starting at `0001` (ADR-0001 conventionally being "Record architecture decisions").
- The number is **permanent** — it is the stable identifier used in code, commits, and other ADRs. Never reuse a number even after rejection/supersession.
- On concurrent PRs that grab the same number, the later-merged PR renumbers before merge (the index regeneration in ADR-IDX-01 surfaces the clash).

### B. Storage layout

ADRs live in the repository under version control:

```
docs/
└── architecture/
    └── decisions/
        ├── README.md                              # generated index (ADR-IDX-01)
        ├── 0001-record-architecture-decisions.md
        ├── 0002-use-postgresql-as-primary-datastore.md
        ├── 0003-adopt-event-driven-order-processing.md
        └── adr-template.md
```

Common alternative roots: `docs/adr/`, `docs/decisions/`, or per-service `adr/` in a monorepo. Pick one location per repo/service and keep all ADRs there.

### C. Index

Maintain a generated index (the directory `README.md`) listing every ADR — id, title, status, date — so the log is browsable. Regenerate it from the files (tooling below) rather than hand-editing, to satisfy ADR-IDX-01:

```markdown
| ID | Title | Status | Date |
|----|-------|--------|------|
| [ADR-0001](0001-record-architecture-decisions.md) | Record architecture decisions | Accepted | 2026-01-04 |
| [ADR-0002](0002-use-postgresql-as-primary-datastore.md) | Use PostgreSQL as primary datastore | Accepted | 2026-01-12 |
| [ADR-0005](0005-use-mysql.md) | Use MySQL | Superseded | 2026-01-08 |
```

### D. Linking from code

Reference an ADR from source where the decision is enforced — e.g. `# Schema is JSONB per ADR-0002`. The exact comment/docstring convention is owned by [`comments.md`](guides://comments.md); ADRs just provide the stable `ADR-NNNN` token to cite.

---

## 6. Tooling

ADR tooling automates numbering, file creation, linking, and index generation. Use one; do not hand-number.

- **adr-tools** (shell): `adr init docs/architecture/decisions`, `adr new "Use PostgreSQL as primary datastore"` (auto-numbers + creates from template), `adr new -s 5 "Migrate to managed Postgres"` (supersedes ADR-5 and cross-links), `adr generate toc > README.md`, `adr list`.
- **MADR** template + `markdownlint` for the format/lint gates (ADR-FMT-01/02).
- **Log4brains**: a static-site/CLI tool that builds a browsable, searchable ADR knowledge base and validates structure in CI.

Wire the format, lint, and index checks (§2 IDs) into CI (see `ci-cd.md` from the project's stack) so a malformed or unindexed ADR fails the build. Markdown linting itself is owned by [`markdown.md`](guides://markdown.md).

---

## 7. Review Process

ADRs ship through the normal PR review flow (owned by [`code-review.md`](guides://code-review.md)); the ADR-specific review focus is:

- **Context** genuinely explains the problem and constraints — a reader unfamiliar with the issue can follow it.
- **Decision** is specific, scoped to one choice, and in active voice.
- **Consequences** are honest and balanced (real downsides listed).
- **Alternatives** are genuine, not strawmen, each with a rejection reason (ADR-DEC-01).
- **Links** to related/superseded ADRs are present and bidirectional (ADR-LIFE-03).
- Merge sets the status: `Accepted` or `Rejected`. An Accepted ADR is immutable thereafter.

---

## 8. Deployment Checklist

Generated from §2 — one box per requirement ID. No new requirements.

- [ ] ADR-STRUCT-01 — file named `NNNN-kebab-title.md`, 4-digit zero-padded, in the ADR dir
- [ ] ADR-STRUCT-02 — number unique, monotonic, never reused/deleted
- [ ] ADR-FMT-01 — Title/Status/Date/Context/Decision/Consequences all present
- [ ] ADR-FMT-02 — `markdownlint` clean (see `markdown.md`)
- [ ] ADR-LIFE-01 — Status in {Proposed, Accepted, Rejected, Deprecated, Superseded}
- [ ] ADR-LIFE-02 — Accepted ADR decision text unchanged (supersede instead of edit)
- [ ] ADR-LIFE-03 — Superseded/superseding ADRs cross-linked both ways
- [ ] ADR-DEC-01 — ≥1 genuine rejected alternative with reasoning
- [ ] ADR-IDX-01 — index regenerated and matches the files on disk
- [ ] ADR-REV-01 — ADR merged via approved PR (see `code-review.md`)

---

**End of Architecture Decision Records (ADR) Guidelines**
