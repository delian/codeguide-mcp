# TODO Comment Guidelines
Mandatory conventions for TODO/FIXME/HACK markers: tag taxonomy, required metadata (owner, issue link, date), lifecycle, debt tracking, and CI enforcement of stale and orphaned markers. Language-agnostic.

---
name: todo
title: TODO Comment Guidelines
version: 2.0
last_reviewed: 2026-06-05
kind: cross-cutting
tools: []
requires: []
recommends:
  - comments
  - git
  - code-review
provides:
  - todo-taxonomy
  - debt-tracking
  - todo-lifecycle
---

> 🧭 Authored per [`CONVENTIONS.md`](guides://CONVENTIONS.md): shared concerns are referenced, not restated. This guide owns only the convention for in-code deferral markers (TODO/FIXME/HACK and friends) — their syntax, required metadata, lifecycle, and enforcement.

---

## 0. Prerequisites & References

This guide is the canonical owner of TODO-comment conventions. It defers *general* comment policy, issue linking, and the review of debt to their owners.

> 📎 **RECOMMENDED — fetch when the task touches them:**
> - [`comments.md`](guides://comments.md) — general comment & API-doc policy. *(A TODO is a comment; obey that policy too. This guide adds only the deferral-marker rules.)*
> - [`git.md`](guides://git.md) — branch/commit/issue conventions used to *link* a marker to a tracked ticket and to date it via blame.
> - [`code-review.md`](guides://code-review.md) — how reviewers approve, reject, or escalate the debt a marker represents.

> 📎 **SEE ALSO:** [`semver.md`](guides://semver.md) *(removing deprecated paths flagged by markers)* · [`pre-commit.md`](guides://pre-commit.md) · [`ci-cd.md`](guides://ci-cd.md) *(where the §6 enforcement gates run)*.

A TODO marker is **not** a substitute for an issue tracker. It is a *pointer in the code* to work that is tracked elsewhere. The tracker is the source of truth; the marker is the cross-reference.

---

## 1. Core Philosophies

Principles unique to deferral markers. General comment hygiene comes from `comments.md`.

- **Every marker is a debt with an owner and a due-date.** An anonymous, undated TODO is invisible debt — forbidden (§2 `TODO-META-01`).
- **Markers point outward, never inward.** The *why/when/who* of the work lives in the tracker; the marker carries an ID that resolves there. Never write a paragraph of plan in a comment when an issue link will do.
- **A marker has a finite lifetime.** It is created, optionally escalated, and *resolved by deletion* — either the work is done or the marker is converted to a tracked ticket and removed. Markers do not accrete (§5 lifecycle).
- **The taxonomy is closed.** Use only the tags in §3. Inventing `// XXX2` or `// NOTE_TODO` defeats grep-based tooling and CI gates.
- **Severity is explicit, not implied.** `FIXME` and `HACK` are not "stronger TODOs" — they mean specific things (§3) and some are release-blocking (§4).
- **If it blocks release, it is not a TODO — it is a bug.** Markers track *deferred, acceptable* work. Anything that must not ship is an open ticket and a failing test (see `tdd.md`), not a comment.

**Verified Code**: agent-generated code MUST pass every gate in §2 before delivery — no orphaned, undated, or release-blocking markers.

---

## 2. Requirements (MANDATORY, auditable)

RFC-2119 keywords. IDs `TODO-<TOPIC>-<NN>`. Each row has a binary gate; rows binding a shared rule cite its owner.

| ID | Requirement | Verify | Gate |
|----|-------------|--------|------|
| TODO-TAX-01 | Markers MUST use only the closed taxonomy tags in §3 (`TODO`, `FIXME`, `HACK`, `XXX`, `OPTIMIZE`, `DEBT`, `SECURITY`, `DEPRECATED`) | grep for any other ALL-CAPS marker token | none found |
| TODO-META-01 | Every marker MUST carry an owner/ticket reference and an ISO date: `TAG(ref, YYYY-MM-DD)` | `scripts/lint-todos` (§6) | 0 markers missing metadata |
| TODO-META-02 | The `ref` MUST resolve to a real, open tracker item or a §3 reserved category (see `git.md` for the ID scheme) | `scripts/lint-todos --check-tracker` | 0 dangling refs |
| TODO-BLK-01 | No `FIXME`, `HACK`, or `SECURITY` marker MAY exist on a release/`main` build | grep on the release ref (§6) | 0 found on release |
| TODO-STALE-01 | No marker MAY be older than its tag's max age (§4) | `scripts/lint-todos --max-age` | 0 over-age markers |
| TODO-ORPH-01 | No marker MAY reference a closed/merged ticket (orphaned) | `scripts/lint-todos --check-tracker` | 0 orphans |
| TODO-DEBT-01 | `DEBT`/`SECURITY` markers MUST be mirrored by a tracked ticket and surface in the debt register (§5.C) | register diff vs grep | counts match |
| TODO-CMT-01 | Each marker MUST also satisfy general comment policy (see `comments.md`) | review | conforms |

> **Forbidden**: an undated or unowned marker; a marker used in place of opening a ticket for must-fix work; a `FIXME`/`HACK`/`SECURITY` left on a release branch; commenting-out code with a bare `// TODO remove`; suppressing the §6 linter to land a marker.

---

## 3. The Taxonomy (closed set)

Use exactly these tags. Each has one meaning. This is what graders, IDEs, and the §6 linter scan for.

| Tag | Meaning | Typical resolution | Release-blocking? |
|-----|---------|--------------------|--------------------|
| `TODO` | Planned, non-urgent work that is *fine to ship without*. | Implement, then delete the marker. | No |
| `FIXME` | Known-incorrect or fragile behavior that works *for now* but is wrong. | Fix and add a regression test (see `tdd.md`), then delete. | **Yes** |
| `HACK` | A deliberate, ugly workaround taken knowingly; correct approach is deferred. | Replace with the real solution. | **Yes** |
| `XXX` | "Danger / be careful here" — a sharp edge a future reader must understand before touching. | Refactor away the sharp edge, or promote to documented invariant. | No (but review) |
| `OPTIMIZE` | A known performance shortcut; correctness is fine (policy: `performance.md`). | Profile-justify, then optimize or delete. | No |
| `DEBT` | Tracked technical debt: a larger structural shortcut accepted on purpose. | Pay down per the debt register (§5.C). | No |
| `SECURITY` | A known security gap or unhardened path (policy: `secure-coding.md`). | Remediate before release; never ship. | **Yes** |
| `DEPRECATED` | Marks an API/path scheduled for removal (policy: `semver.md`). | Remove on the documented major; carry a removal date. | No |

Rules:
- **No synonyms, no nesting.** Not `BUG` (that is a ticket + failing test), not `NOTE`/`REVIEW` (use a normal comment or the PR — see `code-review.md`), not `TBD`, not `WIP`.
- **One tag per marker.** If something is both a hack and a security gap, it is `SECURITY` (the higher-severity tag wins).
- **Tag at column zero of the comment**, immediately after the comment leader, ALL-CAPS, so a single regex (`\b(TODO|FIXME|HACK|XXX|OPTIMIZE|DEBT|SECURITY|DEPRECATED)\b`) finds every marker.

---

## 4. Required Metadata & Format

Every marker is one canonical shape, independent of language comment syntax:

```
<leader> TAG(<ref>, <YYYY-MM-DD>): <imperative description>
<leader>   [optional continuation lines, indented]
```

- **`TAG`** — one of §3.
- **`ref`** — the link to the source of truth. One of:
  - a tracker ID — `#1423`, `JIRA-456`, `GH-88` (resolution + ID scheme owned by `git.md`/your tracker);
  - a reserved category for cross-cutting markers that have no single ticket — `SECURITY`, `DEBT`, `PERF`, `DEPRECATED` (these MUST still be mirrored in the debt register, §5.C, per `TODO-DEBT-01`).
  - Bare markers with no `ref` are forbidden (`TODO-META-01`).
- **`YYYY-MM-DD`** — the **creation date**, used by the staleness gate (`TODO-STALE-01`). Do not rely on git blame alone: rebases and reformatting reset blame; the in-text date is authoritative.
- **description** — imperative, one line. Detail belongs in the linked ticket, not the comment (see `comments.md`).

The comment *leader* is whatever the language uses (`//`, `#`, `--`, `/* */`, `<!-- -->`); only the leader changes between languages — the marker grammar does not. Show the idiom in the language guide, not here.

```text
// TODO(#1423, 2026-06-05): paginate the results endpoint
# FIXME(GH-88, 2026-05-30): race on concurrent writes — guarded by a coarse lock for now
-- HACK(#902, 2026-06-01): force index until the planner picks it itself
// SECURITY(SECURITY, 2026-06-02): rate-limit this route before public launch
// DEPRECATED(#770, 2026-06-04): remove in v3.0 — superseded by `parseV2` (see semver.md)
```

### Maximum age (staleness gate, `TODO-STALE-01`)

A marker older than its tag's ceiling fails CI. Tune per project, but the defaults are:

| Tag | Max age | Rationale |
|-----|---------|-----------|
| `FIXME`, `HACK`, `SECURITY` | release-blocking — never reach `main` | resolve before merge/release |
| `TODO`, `XXX`, `OPTIMIZE` | 90 days | force a decision: do it, ticket it as `DEBT`, or delete it |
| `DEBT` | tracked indefinitely, but MUST appear in the register (§5.C) | the register, not the calendar, governs |
| `DEPRECATED` | until the documented removal version | bound by `semver.md`, not by age |

Over-age `TODO`/`XXX`/`OPTIMIZE` markers are not silently extended — they are converted to `DEBT` (with a register entry) or deleted.

---

## 5. Lifecycle & Debt Tracking

A marker has exactly three legal end-states; "lingering" is not one of them.

### A. States

```
        create                escalate (optional)            resolve
  ──────────────►  ACTIVE  ──────────────────────────►  ESCALATED
                     │                                       │
                     │  done / obsolete                      │  ticketed as DEBT
                     ▼                                       ▼
                  DELETED  ◄───────────────────────────  CONVERTED→DEBT (register)
```

1. **Create** — added with full §4 metadata at the moment the shortcut is taken. Never "add the TODO later".
2. **Escalate** — a `TODO`/`XXX` that survives a review cycle or nears its max age is either deleted or promoted to `DEBT`/`FIXME` with a tracker entry. Severity only goes *up*.
3. **Resolve** — the work is done and the marker is **deleted in the same commit** as the fix (see `git.md` for linking the commit to the ticket). Resolution = deletion. A code change that "addresses" a marker without removing it has not resolved it.

### B. When a reviewer sees a marker (defer to `code-review.md`)

Review of the debt a marker represents is owned by [`code-review.md`](guides://code-review.md). The TODO-specific checks a reviewer applies: metadata present and valid (`TODO-META-01/02`), tag matches severity (§3), no release-blocking tag on the target branch (`TODO-BLK-01`), and any new `DEBT`/`SECURITY` marker has a register entry (`TODO-DEBT-01`). Reviewers reject markers that are really hidden bugs (those need a ticket + failing test, see `tdd.md`).

### C. The debt register

`DEBT` and `SECURITY` markers (and any escalated `TODO`) are mirrored in a single, version-controlled register so debt is visible at portfolio level, not buried in source. It is generated/reconciled from the markers by §6 tooling, so the two cannot drift (`TODO-DEBT-01`).

```markdown
# DEBT.md — Technical Debt Register (reconciled from code markers by CI)

| Marker ref | Tag | Location | Created | Owner | Ticket | Pay-down trigger |
|-----------|-----|----------|---------|-------|--------|------------------|
| DEBT | DEBT | adapters/db/legacy.go:88 | 2026-05-10 | @ana | #1102 | before sharding work |
| SECURITY | SECURITY | api/routes/upload.py:14 | 2026-06-02 | @lee | #1190 | before public launch |
```

The register links each entry to its tracker item (`git.md`); the *prioritization* of paying it down is a planning concern, not a code-comment concern.

---

## 6. Enforcement (CI & local)

Markers are only as good as the gate that audits them. The gates run in pre-commit (`pre-commit.md`) and CI (`ci-cd.md`); this guide owns *what* they check, not the pipeline plumbing.

A single scanner satisfies §2. Sketch (language-agnostic, ripgrep-based):

```bash
#!/usr/bin/env bash
# scripts/lint-todos — fails CI on any §2 violation. Tune MAX_AGE/branch as needed.
set -euo pipefail
TAGS='TODO|FIXME|HACK|XXX|OPTIMIZE|DEBT|SECURITY|DEPRECATED'
RELEASE_REF="${1:-origin/main}"

# Collect every marker: file:line:tag:ref:date
markers=$(rg -nP "\b(${TAGS})\b" -g '!{.git,node_modules,dist,vendor}' || true)

fail=0
while IFS= read -r line; do
  [ -z "$line" ] && continue
  # TODO-META-01: require TAG(ref, YYYY-MM-DD)
  if ! grep -qP "\b(${TAGS})\(\s*[^,]+,\s*\d{4}-\d{2}-\d{2}\s*\)" <<<"$line"; then
    echo "META: missing ref/date -> $line"; fail=1; continue
  fi
  # TODO-BLK-01: no release-blocking tag (checked against the release ref in CI)
  if grep -qP "\b(FIXME|HACK|SECURITY)\(" <<<"$line"; then
    echo "BLOCKER on $RELEASE_REF -> $line"; fail=1
  fi
  # TODO-STALE-01: enforce per-tag max age (date math omitted for brevity)
done <<<"$markers"

# TODO-TAX-01: catch ALL-CAPS marker-shaped tokens NOT in the taxonomy (e.g. NOTE/BUG/TBD/WIP)
if rg -nP '//?\s*(BUG|TBD|WIP|NOTE_TODO|REVIEW)\b' -g '!{.git,node_modules}' ; then
  echo "TAX: non-taxonomy marker"; fail=1
fi

exit "$fail"
```

- **`TODO-META-02` / `TODO-ORPH-01`** (`--check-tracker`): resolve each `ref` against the issue tracker's API and fail on dangling or already-closed tickets. The ID scheme and API are the tracker's / `git.md`'s concern.
- **`TODO-DEBT-01`**: diff the grep of `DEBT`/`SECURITY` markers against `DEBT.md` (§5.C); fail if either side has an unmatched row.
- IDE integration (TODO tree / problem-matcher) gives the same view locally; CI is the authority.

> Do **not** globally ignore the marker linter to land code. If a marker legitimately cannot be resolved now, escalate it to `DEBT` with a register entry — that is the sanctioned escape hatch, not a suppression comment.

---

## 7. Quick Reference

```text
TAG(ref, YYYY-MM-DD): imperative description     # the one canonical shape

TODO        plannable, ships fine        → implement & delete       (≤90d)
FIXME       wrong-but-works              → fix + regression test     (BLOCKS release)
HACK        knowing ugly workaround      → real fix                  (BLOCKS release)
XXX         sharp edge, be careful       → refactor / document       (≤90d)
OPTIMIZE    known perf shortcut          → profile then fix          (≤90d)
DEBT        accepted structural debt     → register + pay down       (tracked)
SECURITY    known gap                    → remediate                 (BLOCKS release)
DEPRECATED  scheduled removal            → delete on major (semver)  (version-bound)
```

```bash
rg -nP '\b(TODO|FIXME|HACK|XXX|OPTIMIZE|DEBT|SECURITY|DEPRECATED)\b'   # find all markers
scripts/lint-todos origin/main          # run all §2 gates locally
scripts/lint-todos --check-tracker      # resolve refs, catch orphans
scripts/lint-todos --max-age            # staleness gate
```

---

## 8. Deployment Checklist

Generated from §2 — one box per requirement ID. No new requirements.

- [ ] TODO-TAX-01 — only closed-taxonomy tags present
- [ ] TODO-META-01 — every marker has `(ref, YYYY-MM-DD)`
- [ ] TODO-META-02 — every `ref` resolves to a real tracker item (see `git.md`)
- [ ] TODO-BLK-01 — no `FIXME`/`HACK`/`SECURITY` on the release ref
- [ ] TODO-STALE-01 — no marker over its §4 max age
- [ ] TODO-ORPH-01 — no marker pointing at a closed ticket
- [ ] TODO-DEBT-01 — `DEBT`/`SECURITY` markers reconciled with `DEBT.md`
- [ ] TODO-CMT-01 — markers satisfy general comment policy (see `comments.md`)
- [ ] Agent ran `scripts/lint-todos` (§6) and documented any fixes

---
**End of TODO Comment Guidelines**
