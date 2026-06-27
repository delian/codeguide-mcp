# Code Review Guidelines
Mandatory standards for human and automated code review: what to look for, review checklists, PR hygiene and size, reviewer etiquette, and approval gates. GitHub/GitLab/Bitbucket PRs, CI status checks, SAST/linters.

---
name: code-review
title: Code Review Guidelines
version: 2.0
last_reviewed: 2026-06-05
kind: cross-cutting
tools: []
requires: []
recommends:
  - git
  - tdd
  - secure-coding
  - comments
provides:
  - review-checklist
  - pr-hygiene
  - review-etiquette
  - approval-gates
---

> 🧭 Authored per [`CONVENTIONS.md`](guides://CONVENTIONS.md): shared concerns are referenced, not restated. This guide canonically owns code-review practice — checklists, PR hygiene, reviewer etiquette, and approval gates. It does not re-explain branch/PR mechanics (`git.md`), test policy (`tdd.md`), the security catalogue (`secure-coding.md`), or doc rules (`comments.md`).

---

## 0. Prerequisites & References

Code review is a gate that *enforces* other guides — it does not re-define their rules. The reviewer checks that the change complies with whatever guides apply to it.

> 📎 **RECOMMENDED — fetch when the task touches them:**
> - [`git.md`](guides://git.md) — branch/PR/merge workflow, commit hygiene, the mechanics a review sits on top of. *(This guide owns the *content* of review; `git.md` owns the *workflow* it runs in.)*
> - [`tdd.md`](guides://tdd.md) — test-first, regression-test-before-fix, coverage. The "tests present & adequate" gate below defers to it.
> - [`secure-coding.md`](guides://secure-coding.md) — the canonical vulnerability catalogue, secrets, supply chain. The security pass below points at it rather than re-listing checks.
> - [`comments.md`](guides://comments.md) — comment/API-doc policy. The documentation pass defers to it.

> 📎 **SEE ALSO:** [`ci-cd.md`](guides://ci-cd.md) (status checks/required gates) · [`pre-commit.md`](guides://pre-commit.md) (shift-left of automated checks) · [`error-handling.md`](guides://error-handling.md) · [`performance.md`](guides://performance.md) · [`adr.md`](guides://adr.md) (review of architectural decisions) · [`semver.md`](guides://semver.md) (breaking-change review)

---

## 1. Core Philosophies: REVIEW-FIRST

Code-review-specific principles only. Test, security, and doc *rules* come from §0.

- **R**espectful: critique the code, never the author; default to questions over verdicts.
- **E**ducational: every blocking comment explains *why* and links the authority (a guide, a standard, a measurable impact) — not "because I said so".
- **V**erifiable: feedback cites an objective criterion (a §0 guide, a failing test, a measured cost), so disagreements resolve on evidence, not opinion.
- **I**terative: small PRs reviewed within hours beat large PRs reviewed in days; size and turnaround are first-class quality levers.
- **E**fficient: machines catch format/lint/type/CVE/coverage; humans spend their attention on correctness, design, and security *judgment* — never re-litigate what a linter already enforces.
- **W**holistic: a review weighs correctness, security, design, performance, tests, and docs together — approval means the change is safe to merge, not merely that it compiles.

**Approval is an assertion, not a courtesy.** Approving a PR means: "I read this, I understand it, and I am willing to be on the hook for it in production."

---

## 2. Requirements (MANDATORY, auditable)

RFC-2119 keywords. IDs `CR-<TOPIC>-<NN>`. Each row has a binary gate; rows binding a shared rule cite its owner.

| ID | Requirement | Verify | Gate |
|----|-------------|--------|------|
| CR-PR-01 | Every change to a shared/protected branch MUST go through a reviewed PR (no direct pushes) | branch protection settings; `git log --merges` | direct pushes blocked |
| CR-PR-02 | A PR SHOULD change < 400 net lines; larger PRs MUST justify size or be split | `git diff --shortstat <base>...HEAD` | < 400, or documented |
| CR-PR-03 | Each PR MUST be one logical change with a complete description (what/why/how-to-verify) | PR body present & filled | template fields non-empty |
| CR-PR-04 | Author MUST self-review the full diff before requesting review | self-review comments / checklist box | box checked |
| CR-AUTO-01 | All automated gates (format, lint, type, tests, coverage, SAST, CVE) MUST be green before human review starts | CI status on the PR | all checks passing |
| CR-AUTO-02 | Automated-checkable concerns MUST NOT be raised as manual review comments (automate instead) | review comments vs linter scope | no style-nit comments where a linter exists |
| CR-RVW-01 | Review MUST cover, in priority order, correctness → security → design → performance → tests → docs → style | reviewer checklist (§4) | all areas considered |
| CR-RVW-02 | Tests MUST accompany behaviour changes and bug fixes (see `tdd.md`) | inspect diff for tests; CI | tests present, CI green |
| CR-RVW-03 | Security-relevant changes MUST be reviewed against the catalogue in `secure-coding.md` | security pass on diff | no open high/critical finding |
| CR-RVW-04 | Public API / behaviour changes MUST update docs (see `comments.md`) | docs diff present | docs updated |
| CR-CMT-01 | Every review comment MUST carry an intent prefix (`[MUST]/[SHOULD]/[COULD]/[NIT]/[QUESTION]/[PRAISE]`) | scan comments | all prefixed |
| CR-CMT-02 | Blocking (`[MUST]`) comments MUST state the why and a concrete fix/alternative | inspect `[MUST]` comments | each has rationale + suggestion |
| CR-GATE-01 | Merge MUST require ≥1 (≥2 for high-risk/security) approval and 0 unresolved `[MUST]` threads | branch protection + thread state | required approvals met, no open MUST |
| CR-GATE-02 | Approving means having actually read and understood the diff — no rubber-stamping | spot-check / review comments | substantive review evidence |
| CR-GATE-03 | A PR author MUST NOT self-approve or merge their own unreviewed change | platform setting | self-approval disabled |

> **Forbidden**: merging with failing CI or unresolved `[MUST]` threads; approving without reading; blocking a PR purely on personal style preference a linter does not enforce; bundling unrelated changes into one PR; landing a behaviour change with no test (violates `tdd.md`) or an API change with no doc update (violates `comments.md`).

---

## 3. The Review, Step by Step

A review is a pipeline. Do the cheap, automatable steps first so human attention is spent where only humans add value.

```
1. CI is green?            → if red, stop; author fixes before human review (CR-AUTO-01)
2. Context loaded          → read PR description + linked issue/ADR; know the goal
3. Scope sane?             → one logical change, < ~400 lines, else push back (CR-PR-02/03)
4. Priority-ordered pass   → correctness → security → design → perf → tests → docs → style (CR-RVW-01)
5. Comment with intent     → prefix every note; [MUST] gets why + fix (CR-CMT-01/02)
6. Verdict                 → Approve / Comment / Request-Changes (§6, CR-GATE-01)
```

Why this order: catching a security or correctness defect after you have spent the review budget bikeshedding variable names is the most common way real bugs ship. Review the things that can hurt production first.

---

## 4. The Review Checklist (this guide's core)

Run top-to-bottom. Items that bind another guide defer to it rather than re-stating its rules.

### A. Correctness — *does it do the right thing?*
- [ ] Code actually accomplishes the stated goal in the PR/issue.
- [ ] Edge/boundary cases handled: empty, null/None, zero, max, off-by-one, concurrency.
- [ ] Error paths are deliberate and propagated/handled per [`error-handling.md`](guides://error-handling.md) — no silently swallowed failures.
- [ ] No obvious logic inversions, wrong operators, or copy-paste mistakes.
- [ ] Backwards compatibility preserved, or the break is intentional and versioned (see [`semver.md`](guides://semver.md)).

### B. Security — *is it safe?*
Do not re-derive the catalogue; review the diff against [`secure-coding.md`](guides://secure-coding.md). On any user-input, auth, crypto, deserialization, or shell/SQL surface:
- [ ] Untrusted input validated/escaped at the boundary; queries parameterized; output encoded.
- [ ] AuthN/AuthZ checks present at the right layer; no missing permission checks.
- [ ] No secrets in code, logs, or fixtures; no sensitive data logged.
- [ ] New/updated dependencies justified and CVE-clean (see `secure-coding.md`).
- [ ] Any finding here is `[MUST]` and links the relevant `secure-coding.md` section.

### C. Design & architecture — *will we regret this in 6 months?*
- [ ] Fits existing patterns and layering; respects the project's architecture guide (e.g. `hexagonal.md`/`cleanarch.md`).
- [ ] Right abstraction level — not over-engineered, not a copy-paste of nearby code.
- [ ] Single responsibility; dependencies point the right way; no new circular deps.
- [ ] Public API/contract is minimal and hard to misuse; significant decisions captured in an [`adr.md`](guides://adr.md).

### D. Performance — *is it efficient enough?* (judgment, not premature optimization)
- [ ] No N+1 queries or unbounded fan-out introduced.
- [ ] Hot paths don't add accidental O(n²) (e.g. `list.includes` in a loop → set/map lookup).
- [ ] Large result sets paginated/streamed; resources (connections, files, handles) released.
- [ ] Defer deeper analysis to [`performance.md`](guides://performance.md); flag, measure, don't guess.

### E. Tests — *is it verified?* (policy owned by `tdd.md`)
- [ ] New behaviour has tests; each bug fix has a regression test (per [`tdd.md`](guides://tdd.md)).
- [ ] Tests assert behaviour, are deterministic (no flakiness), and aren't over-mocked into meaninglessness.
- [ ] Coverage gate from `tdd.md` / the language guide is met (a number alone is not the goal — relevance is).

### F. Documentation — *can the next person use it?* (policy owned by `comments.md`)
- [ ] Public APIs/behaviour changes documented per [`comments.md`](guides://comments.md); comments explain *why*, not *what*.
- [ ] README/changelog/migration notes updated when user-facing behaviour changes.

### G. Hygiene
- [ ] No dead code, commented-out blocks, debug prints, or leftover TODOs without a tracking ref (see [`todo.md`](guides://todo.md)).
- [ ] Naming clear and consistent; magic numbers named.

> Style/format/lint/type checks are **not** in this list on purpose — they are CR-AUTO-02 machine work. If you find yourself commenting on them, fix the linter config instead.

---

## 5. Feedback & Etiquette (this guide's core)

### A. Intent prefixes (CR-CMT-01)
Every comment starts with one, so the author instantly knows what blocks merge:

| Prefix | Meaning | Blocks merge? |
|--------|---------|---------------|
| `[MUST]` | Correctness/security/contract defect | Yes |
| `[SHOULD]` | Strong improvement; address or justify | Usually |
| `[COULD]` | Optional improvement | No |
| `[NIT]` | Trivial, author's discretion | No |
| `[QUESTION]` | Seeking understanding, not demanding change | No |
| `[PRAISE]` | Acknowledge good work | No |

### B. How to write a comment
A good blocking comment = **prefix + the problem + why it matters + a concrete fix.** Show, don't just tell — include a suggested snippet or a `suggestion` block when the platform supports it.

- ❌ "This is wrong." / "Why would you do this?" / "Didn't you read the style guide?"
- ✅ "[MUST] This dereferences `user` which can be `None` when the lookup misses, causing a crash on the not-found path — add a guard and raise a not-found error (see `error-handling.md`)."
- ✅ "[QUESTION] You chose a Map over a plain object here — was that for insertion-order or key-type reasons? Genuinely asking."
- ✅ "[PRAISE] The backoff strategy being injectable makes this trivially testable — nice."

Concept-level examples live here; do **not** paste language-specific vulnerability or N+1 snippets — name the idiom and link the owner (`secure-coding.md`, `performance.md`, or the language guide).

### C. Responding to feedback (author side)
- Acknowledge within ~1 business day; assume good intent.
- Resolve a thread only when addressed; don't resolve someone else's `[MUST]` thread for them.
- Disagree with evidence: "I chose X because <reason/measurement>; happy to change." Then **disagree-and-commit** if the reviewer holds — don't stall the PR on ego.

### D. Culture (encourage / discourage)
- Encourage: questions, explained reasoning, praising clever solutions, fast turnaround, small PRs.
- Discourage: nitpicking automatable style, condescension, blocking on personal preference, drive-by comments with no follow-through, and rubber-stamping.

---

## 6. Approval Gates & Workflow (this guide's core)

The review *verdict* and the merge *gate* — the mechanics of branches/merges themselves are owned by [`git.md`](guides://git.md).

### A. The three verdicts
| Verdict | Use when |
|---------|----------|
| **Request changes** | Any open `[MUST]`: security/correctness defect, missing required tests, architecture violation, failing CI the author must fix. |
| **Comment** | Non-blocking feedback/questions only; you're not gating but want input addressed. |
| **Approve** | All `[MUST]` resolved, `[SHOULD]`s addressed-or-justified, CI green, you understand and stand behind the change (CR-GATE-02). |

### B. Merge gate (CR-GATE-01)
A PR may merge only when **all** hold:
- Required automated checks green (CR-AUTO-01) — enforced as required status checks (see `ci-cd.md`).
- Required approvals present: **≥1** normally, **≥2** for security-sensitive, infra/migration, or public-API changes.
- Zero unresolved `[MUST]` threads.
- Author is not the sole approver (CR-GATE-03); self-merge of an unreviewed change is blocked by branch protection.

### C. Turnaround targets
| PR size | First review | Follow-up |
|---------|--------------|-----------|
| Small (< 100 LOC) | same day | same day |
| Medium (100–400 LOC) | within 24h | within 24h |
| Large (400+ LOC) | within 48h — *and a nudge to split* | within 24h |

Stale PRs accumulate conflicts and get superficial reviews; keep the loop tight (CR-PR-02).

---

## 7. Automated vs Human Review

The dividing line is CR-AUTO-01/02: machines gate the objective, humans judge the subjective. Wire the automated half as **required status checks** in CI ([`ci-cd.md`](guides://ci-cd.md)) and shift it left with [`pre-commit.md`](guides://pre-commit.md) so reviewers rarely see a red PR.

| Automate (no human comment) | Reserve for humans |
|---|---|
| Formatting, lint, import order | Logic correctness & edge cases |
| Type checking | Architecture & abstraction fit |
| SAST / secret scanning (see `secure-coding.md`) | Security *judgment* on novel surfaces |
| Dependency CVE audit | API design & ergonomics |
| Test execution & coverage gate (see `tdd.md`) | Test *quality* & relevance |
| Commit-message / PR-title lint | Naming clarity, "why" in comments |

If a human keeps flagging the same automatable issue, the fix is a new lint/check, not more comments.

---

## 8. Metrics (improve the process, don't game it)

Track to find bottlenecks, never as individual performance scores.

| Metric | Healthy target | Why |
|--------|----------------|-----|
| Time to first review | < 24h | tight feedback loop |
| Review cycles to merge | < 3 | efficient, clear feedback |
| PR size | < 400 LOC | reviewable depth (CR-PR-02) |
| Time to merge | < 48h | sustained velocity |
| Review coverage | 100% of merged code | nothing lands unreviewed (CR-PR-01) |

Do **not** optimize: comments-per-PR (rewards nitpicking), approvals-per-reviewer (rewards rubber-stamping), or speed at the expense of caught defects.

---

## 9. PR Description Template

One logical change, with enough context to review without a meeting (CR-PR-03):

```markdown
## What & Why
<!-- The change in one line, then the motivation / linked issue or ADR. -->
Closes #123

## Changes
- <bullet per logical change>

## How to verify
- <commands / steps / screenshots for UI>

## Checklist
- [ ] Self-reviewed the full diff (CR-PR-04)
- [ ] Tests added/updated (tdd.md)
- [ ] Docs updated if API/behaviour changed (comments.md)
- [ ] No secrets/debug code; CI green
- [ ] Breaking change? versioned + noted (semver.md)
```

---

## 10. Deployment Checklist

Generated from §2 — one box per requirement ID. No new requirements.

**Author**
- [ ] CR-PR-03 — one logical change, description complete (what/why/how-to-verify)
- [ ] CR-PR-02 — diff < 400 net lines or size justified
- [ ] CR-PR-04 — self-reviewed the full diff
- [ ] CR-RVW-02 — tests accompany behaviour/bug changes (`tdd.md`)
- [ ] CR-RVW-04 — docs updated for API/behaviour changes (`comments.md`)
- [ ] CR-AUTO-01 — all CI gates green before requesting review

**Reviewer**
- [ ] CR-RVW-01 — covered correctness → security → design → perf → tests → docs → style
- [ ] CR-RVW-03 — security-relevant change checked against `secure-coding.md`
- [ ] CR-CMT-01/02 — every comment prefixed; each `[MUST]` has why + concrete fix
- [ ] CR-AUTO-02 — raised no comment a linter should own
- [ ] CR-GATE-02 — actually read and understood the diff (no rubber-stamp)

**Merge gate**
- [ ] CR-PR-01 — change went through a reviewed PR (no direct push)
- [ ] CR-GATE-01 — required approvals met, 0 unresolved `[MUST]` threads, CI green
- [ ] CR-GATE-03 — not self-approved/self-merged unreviewed

---
**End of Code Review Guidelines**
