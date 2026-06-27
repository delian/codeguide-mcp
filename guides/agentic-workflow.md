# Agentic Workflow & Engineering Discipline Guidelines
Mandatory process standards for how an AI agent works on a codebase: plan first, isolate parallel work, learn from corrections, verify before done, rubber-duck with a second reviewer, track tasks and docs. Governs HOW you work; the language/tool guides govern WHAT you write.

---
name: agentic-workflow
title: Agentic Workflow & Engineering Discipline Guidelines
version: 2.0
last_reviewed: 2026-06-05
kind: cross-cutting
tools: [git-worktree, plan-mode, subagents, lessons-docs]
requires: []
recommends:
  - tdd
  - code-review
  - git
  - comments
  - adr
  - coding-ai
  - agents-md
provides:
  - plan-mode-discipline
  - subagent-orchestration
  - lessons-learned-loop
  - verification-gate
  - rubber-duck-review
  - task-tracking-discipline
  - worktree-isolation
  - docs-and-changelog-discipline
---

> 🧭 **Read this on principle.** This guide is the canonical owner of *engineering workflow discipline* — the process every non-trivial task follows regardless of language or stack. The language/framework/datastore/infra guides own *what* you write; this owns *how* you work. Authored per [`CONVENTIONS.md`](guides://CONVENTIONS.md): cross-cutting concerns are referenced, not restated.
>
> A project's own `CLAUDE.md` / `AGENTS.md` / `CLAUDE.local.md` are authoritative and **override** this guide where they differ; treat this as the default discipline when the project is silent. Exact file names below (`docs/lessons.md`, `docs/todo.md`, `docs/LOG.md`, `docs/CHANGELOG.md`) are conventions — use the project's actual locations.

---

## 0. Prerequisites & References

This guide owns the *process*. The mechanics it depends on are owned elsewhere — fetch them when the task touches them.

> 📎 **RECOMMENDED — fetch when the task touches them:**
> - [`tdd.md`](guides://tdd.md) — test-first cycle, regression-test-before-fix, coverage (the verification gate, §6, builds on it).
> - [`code-review.md`](guides://code-review.md) — review checklist & etiquette (the rubber-duck review, §6, is an independent application of it).
> - [`git.md`](guides://git.md) — branching, commits, and **worktree isolation** for parallel agents (§4).
> - [`comments.md`](guides://comments.md) · [`adr.md`](guides://adr.md) — code/API docs and decision records (the docs surface, §8).
> - [`coding-ai.md`](guides://coding-ai.md) — AI-assisted coding: prompt hygiene, verifying AI output, trust calibration.
> - [`agents-md.md`](guides://agents-md.md) — the `AGENTS.md` repo-conventions file this discipline complements.

> 📎 **SEE ALSO:** [`semver.md`](guides://semver.md) (changelog/versioning) · [`env-config.md`](guides://env-config.md) (constants → config) · [`feature-flags.md`](guides://feature-flags.md) (opt-in defaults) · [`secure-coding.md`](guides://secure-coding.md) · [`ci-cd.md`](guides://ci-cd.md).

---

## 1. Core Philosophies

- **Plan before you build.** Any non-trivial task starts with a written plan; when the approach goes sideways, **stop and re-plan** rather than pushing through.
- **Learn in a loop.** Read prior lessons before planning; record every user correction as a reusable rule afterwards. The same mistake must not recur.
- **Prove it works.** Nothing is "done" without evidence — green tests, demonstrated behavior, a before/after diff. Every fix and feature ships with its own test.
- **Get a second opinion.** A different reviewer (ideally a different model / fresh context) rubber-ducks non-trivial changes before completion.
- **Isolate parallel work.** Multiple agents never share one working tree; each works in its own git worktree so no one's reset/clean/commit wipes another's in-flight work.
- **Minimal impact, root cause.** Touch only what is necessary, fix the real cause (no temporary patches), and prefer the elegant solution — without over-engineering trivial fixes.
- **Keep the record straight.** Tasks, docs, changelog, and engineering log stay synchronized with shipped reality, in the same change that ships the work.

**Verified Work**: A task is complete only when it satisfies every gate in §2.

---

## 2. Requirements (MANDATORY, auditable)

RFC-2119 keywords. IDs `WF-<TOPIC>-<NN>`. "Non-trivial" = 3+ steps, a new module/algorithm, a refactor, or any behavior change. Rows binding a shared rule cite its owner.

| ID | Requirement | Verify | Gate |
|----|-------------|--------|------|
| WF-PLAN-01 | Non-trivial tasks MUST be planned before implementation — a checklist plan written to the project's plan doc (e.g. `docs/todo.md`) | plan doc has the task as `- [ ]` items | plan exists before code |
| WF-PLAN-02 | When the approach goes sideways, you MUST stop and re-plan rather than continue pushing | review / session log | no thrash; re-plan recorded |
| WF-LESSON-01 | Before a non-trivial task, you MUST read recent `docs/lessons.md` entries and `docs/lessons-summary.md` and apply them in planning | review | lessons consulted |
| WF-LESSON-02 | After ANY user correction, you MUST record the pattern in `docs/lessons.md` and periodically re-summarize into `docs/lessons-summary.md` | `git diff docs/lessons.md` | correction → lesson captured |
| WF-SUB-01 | Research, exploration, and parallel analysis SHOULD be offloaded to subagents — one focused task each — to keep the main context clean | review | scoped subagents used |
| WF-ISO-01 | Parallel agents MUST each work in their own git worktree; the primary checkout MUST NOT `switch`/`reset --hard`/`clean` while another agent/worktree is active (see `git.md`) | review of git ops | no shared-tree collisions |
| WF-VERIFY-01 | A task MUST NOT be marked complete without proof it works — tests green, behavior demonstrated, before/after diff where relevant (see `tdd.md`) | test run + evidence | proof attached |
| WF-VERIFY-02 | Every bug fix and every feature MUST ship with its own test; bugs get a regression test that fails before the fix and passes after (see `tdd.md`) | test suite | test present, green |
| WF-RUBBER-01 | Non-trivial changes MUST be rubber-ducked by a second, independent reviewer (ideally a different model / fresh-context subagent) for bugs, edge cases, and assumptions before done; record the verdict | review record in `docs/LOG.md` | independent review done |
| WF-DOUBLE-01 | After every non-trivial change, you MUST double-check for bugs, duplication, and coding-standard compliance — consulting the relevant coding guides | review | checks run, issues resolved |
| WF-TRACK-01 | When work ships against a plan item, you MUST flip its checkbox `[ ]`→`[x]` (and update any STATUS line) in the SAME change | `git diff <plan-doc>` shows flipped item | checkboxes synced to reality |
| WF-LOG-01 | Changes MUST be recorded in the engineering journal (`docs/LOG.md`); any user-visible change MUST also get a `docs/CHANGELOG.md` entry (Keep a Changelog; see `semver.md`) | LOG/CHANGELOG diff | logged |
| WF-DOC-01 | Any functional change MUST update the docs surface: the README index (high-level, linking per-feature docs), the per-feature doc, and `docs/RESEARCH.md` for research (see `comments.md`/`markdown.md`/`adr.md`) | docs diff | docs updated |
| WF-CONST-01 | Significant constants MUST be exposed as configurable options with documented defaults — not hardcoded (see `env-config.md`/`feature-flags.md`) | grep / config review | no magic constants |
| WF-PROMPT-01 | LLM-facing prompts MUST be externalized as config-overridable template files, never inlined in source (a clearly-marked fallback constant is allowed) | grep for inline prompt bodies | externalized + override tested |
| WF-ARCH-01 | You MUST respect the project's documented locked architectural decisions (`CLAUDE.md`/`AGENTS.md`/ADRs) and not reintroduce superseded patterns (see `adr.md`) | review | no regression to legacy patterns |
| WF-SIMPLE-01 | Changes MUST be minimal-impact and address the root cause — no temporary patches; prefer the elegant solution for non-trivial work | review / diff size | minimal diff, root cause |

> **Forbidden**: marking a task done without proof; shipping a fix/feature with no test; pushing on after the plan breaks instead of re-planning; switching/resetting the primary checkout while parallel agents are active; `git add -A`/`git add .` when another agent has unrelated changes staged; hardcoding significant constants; inlining LLM prompts; leaving plan checkboxes, CHANGELOG, or docs out of sync with shipped code.

---

## 3. Plan First, Re-plan on Drift

- Enter plan mode for any non-trivial task; write the plan as a checklist to the project's plan doc (e.g. `docs/todo.md`) and verify it before implementing (WF-PLAN-01).
- Write specs up front to reduce ambiguity; the plan covers **verification steps**, not just building.
- If something goes sideways, **stop and re-plan immediately** — don't keep pushing a failing approach (WF-PLAN-02).

---

## 4. Subagents & Parallel Isolation

- Use subagents liberally to keep the main context clean: offload research, exploration, and parallel analysis; **one focused task per subagent** (WF-SUB-01). Throw more compute at hard problems via parallel subagents.
- **Every agent works in its own git worktree, never the shared primary checkout** (WF-ISO-01). A single shared tree lets one agent's `reset --hard`/`clean`/commit wipe another's in-flight work.
  - Subagents that edit files: give each its own worktree/branch (e.g. the Agent tool's `isolation: "worktree"`).
  - A standalone work stream: create a dedicated worktree off the main branch and do all editing/testing there.
- **The primary session checkout stays on its branch and never `switch`/`reset --hard`/`clean` while other agents are active** — a branch switch under a live session kills it. Land work by merging the task branch from the primary checkout *without* a checkout (`git -C <primary> merge --no-ff <task>`), then remove the worktree and delete the branch. Commit your own files with explicit paths, never `git add -A` when a parallel agent may have unrelated staged changes. Mechanics owned by [`git.md`](guides://git.md).

---

## 5. Self-Improvement Loop (lessons)

- **Before** any non-trivial task, read the recent `docs/lessons.md` entries and all of `docs/lessons-summary.md`, and apply them while planning (WF-LESSON-01).
- **After** any user correction, record the pattern in `docs/lessons.md` — write a rule for yourself that prevents the same mistake — and iterate on those rules (with user approval) until the mistake rate drops (WF-LESSON-02).
- **Periodically summarize** `docs/lessons.md` into `docs/lessons-summary.md` so the essential lessons stay small and processable.
- The `bug_hunt` and `deduplicate_code` workflows are built on this loop: read lessons first, hunt/fix, then update the lessons docs.

---

## 6. Verification Before Done (+ rubber-duck review)

A staff engineer must be able to approve the change. Before marking complete:

1. **Prove it works** (WF-VERIFY-01): run tests/linters/type-checks green, demonstrate behavior, and diff behavior between the baseline and your change where relevant. Math/algorithm implementations require unit tests.
2. **Test every change** (WF-VERIFY-02): each bug fix gets a regression test (fails before, passes after); each feature gets its own tests. Owned by [`tdd.md`](guides://tdd.md).
3. **Rubber-duck with a second, independent reviewer** (WF-RUBBER-01): have a *different* model / fresh-context subagent review the diff + intent and hunt for bugs, missed edge cases, and wrong assumptions — a different reviewer doesn't share the author's blind spots. Record the verdict and any resulting fix in `docs/LOG.md`.
4. **Double-check** (WF-DOUBLE-01): re-read end-to-end for bug classes (off-by-one, empty-collection/vacuous-truth, silent config drop, resume/restart edge cases), confirm no duplication an existing module already covers, and run the standards review against the relevant coding guides. Confirm a finding with a runnable probe before declaring it real.

Any fix surfaced here is applied and its pattern captured in `docs/lessons.md`.

---

## 7. Task & Change Tracking

- **Plan doc** (`docs/todo.md`): keep checkboxes synchronized with shipped reality — flip `[ ]`→`[x]` and update any STATUS line in the SAME change that ships the work, not a later pass (WF-TRACK-01). When one change spans multiple sections, update every touched section. Periodically audit unchecked items against git history.
- **Engineering journal** (`docs/LOG.md`): log changes and key decisions chronologically with file:line citations and rationale (WF-LOG-01).
- **User-facing changelog** (`docs/CHANGELOG.md`): for any user-visible change (feature, behavior/default change, deprecation, config knob, user-reported fix), add a Keep-a-Changelog entry under `## [Unreleased]`. Pure-internal refactors update the journal only. When in doubt, add the changelog entry. Versioning policy owned by [`semver.md`](guides://semver.md).

---

## 8. Documentation Surface

- **README as a high-level index**: keep it a bounded, link-rich index pointing to per-feature docs (e.g. `docs/<feature>.md`); every new feature gets a per-feature doc linked from the index, with no gaps in the table of contents (WF-DOC-01).
- **Per-feature docs** are the primary detail surface: how it runs, how it's used, examples, pros/cons, algorithm references.
- **Research** (`docs/RESEARCH.md`): record sources with short summaries and links; surface valuable outcomes as future plan items. Authoring conventions owned by [`comments.md`](guides://comments.md), [`markdown.md`](guides://markdown.md), and [`adr.md`](guides://adr.md) for decisions.

---

## 9. Engineering Principles

- **Simplicity & minimal impact** (WF-SIMPLE-01): the smallest change that solves the root cause; no temporary fixes; senior-developer standards.
- **Elegance, balanced**: for non-trivial changes, pause and ask "is there a more elegant way?"; if a fix feels hacky, redo it properly. Skip this for obvious one-liners — don't over-engineer.
- **Autonomous bug fixing**: given a bug report (logs, errors, failing tests), fix it — including failing CI — without hand-holding.
- **Constants are configuration** (WF-CONST-01): every significant constant is a documented, default-valued config option, not a hardcoded literal. Owned by [`env-config.md`](guides://env-config.md); runtime toggles by [`feature-flags.md`](guides://feature-flags.md).
- **Externalize LLM prompts** (WF-PROMPT-01): prompts are operator-tunable behavior, not code — keep them in config-overridable template files and test that an override is honored.
- **Respect locked architecture** (WF-ARCH-01): honor the project's documented architectural decisions and don't reintroduce superseded patterns; record new decisions as ADRs (see [`adr.md`](guides://adr.md)).

---

## 10. Done Checklist

Generated from §2 — one box per requirement ID. A task is complete only when all apply (skip rows that genuinely don't apply, and say so).

- [ ] WF-PLAN-01/02 — planned before building; re-planned if the approach drifted
- [ ] WF-LESSON-01/02 — lessons read before; corrections captured after
- [ ] WF-SUB-01 / WF-ISO-01 — scoped subagents; parallel work isolated in worktrees; primary checkout never switched/reset under live agents
- [ ] WF-VERIFY-01/02 — proof it works; every fix/feature has its own (regression) test, green
- [ ] WF-RUBBER-01 — independent second-reviewer rubber-duck done; verdict logged
- [ ] WF-DOUBLE-01 — bug/duplication/standards double-check run against the coding guides
- [ ] WF-TRACK-01 — plan checkboxes/STATUS synced in the same change
- [ ] WF-LOG-01 — `docs/LOG.md` updated; `docs/CHANGELOG.md` updated for user-visible changes
- [ ] WF-DOC-01 — README index + per-feature doc (+ RESEARCH.md) updated for functional changes
- [ ] WF-CONST-01 / WF-PROMPT-01 — significant constants configurable; LLM prompts externalized
- [ ] WF-ARCH-01 / WF-SIMPLE-01 — locked architecture respected; minimal, root-cause, elegant change

---
**End of Agentic Workflow & Engineering Discipline Guidelines**
