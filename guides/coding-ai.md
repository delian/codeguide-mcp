# AI-Assisted Coding Guidelines
Mandatory practice for coding with AI agents: prompt hygiene, verifying AI output, hallucination & supply-chain risk, trust calibration, and agent workflows. Claude Code, GitHub Copilot, Cursor, Windsurf, Cody, and any AI-assisted tool.

---
name: coding-ai
title: AI-Assisted Coding Guidelines
version: 2.0
last_reviewed: 2026-06-05
kind: cross-cutting
tools: []
requires: []
recommends:
  - agents-md
  - tdd
  - code-review
  - secure-coding
  - comments
provides:
  - ai-assisted-coding
  - ai-code-verification
  - prompt-hygiene
---

> 🧭 Authored per [`CONVENTIONS.md`](guides://CONVENTIONS.md): shared concerns are referenced, not restated. This guide owns only what is unique to working *with* an AI agent — how to prompt it, how to verify its output, and when to trust it.

---

## 0. Prerequisites & References

This guide governs the *human-agent loop*. It does not redefine testing, security, review, or docs — it binds them to AI-generated code.

> 📎 **RECOMMENDED — fetch when the task touches them:**
> - [`agents-md.md`](guides://agents-md.md) — the canonical owner of `AGENTS.md` / project context files. *(AI binding: context files are the #1 quality lever; this guide assumes you have one.)*
> - [`tdd.md`](guides://tdd.md) — test-first, Red-Green-Refactor, regression-test-before-fix, coverage. *(AI binding: the failing test is the agent's specification.)*
> - [`code-review.md`](guides://code-review.md) — review checklists, PR hygiene, approval gates. *(AI binding: AI diffs get the same or stricter review.)*
> - [`secure-coding.md`](guides://secure-coding.md) — SAST/SCA, secrets, supply chain, CVE policy. *(AI binding: AI output is untrusted third-party code.)*
> - [`comments.md`](guides://comments.md) — doc/comment policy for what the agent emits.

> 📎 **SEE ALSO:** [`ci-cd.md`](guides://ci-cd.md) · [`pre-commit.md`](guides://pre-commit.md) · [`git.md`](guides://git.md) · [`todo.md`](guides://todo.md) — the gates and hooks the agent's output must pass through.

---

## 1. Core Philosophies: VERIFIED

AI-specific principles only. Test-first, security scanning, review mechanics, and doc policy come from §0 — do **not** restate them here.

- **V**erify before trust: AI output is a *proposal*, not a fact. It is untrusted until a gate (test, type-check, scan, or a human who understands it) proves it correct.
- **E**ngineer the context: most failures are ambiguity, not model weakness. The right files, the right rules (`AGENTS.md`), and explicit acceptance criteria beat clever prompting.
- **R**eproducible instructions: a fresh agent session — or another developer — should produce equivalent results from the same prompt and context. If the result depends on chat history, write it down.
- **I**ncremental scope: one task, one bug, one function per prompt. Bound what the agent may touch; large unscoped prompts cause scope creep and context overflow.
- **F**ail fast with feedback loops: type checkers, linters, and tests are the agent's tightest, cheapest correction signal — give it the error verbatim and let it iterate, capped.
- **I**nstruct, don't hope: if you want a behavior, make it a rule, not a wish. Implicit expectations are silently violated.
- **E**scalate on uncertainty: high-risk or ambiguous work (architecture, crypto, schema, compliance) pauses for human judgment. Proportional autonomy — more agent freedom demands more guardrails.
- **D**iff is the truth: review the full diff, not just the new code. Agents "improve" things you didn't ask about.

**Verified Output**: AI-generated code MUST pass every gate in §2 before it is accepted or committed.

---

## 2. Requirements (MANDATORY, auditable)

RFC-2119 keywords. IDs `AI-<TOPIC>-<NN>`. Each row has a binary gate; rows binding a shared rule cite its owner. "Verify" gates run against the language toolchain — substitute the project's commands (see the relevant language guide).

| ID | Requirement | Verify | Gate |
|----|-------------|--------|------|
| AI-PROMPT-01 | Every codegen prompt MUST state acceptance criteria and a scope boundary (which files may/may not change) | review prompt before sending | criteria + scope present |
| AI-TST-01 | AI MUST be given (or asked to write) the failing test before the implementation (see `tdd.md`) | project test runner | test exists & fails first |
| AI-TST-02 | Each bug in AI code MUST get a reproducing regression test before the fix (see `tdd.md`) | project test runner | failing→passing |
| AI-TST-03 | AI changes MUST NOT decrease coverage (see `tdd.md`) | coverage tool with project gate | coverage ≥ baseline |
| AI-DEP-01 | Every new dependency the agent adds MUST be verified to exist & be legitimate (anti-slopsquatting; see `secure-coding.md`) | registry lookup (`npm info` / `pip index` / `cargo search`) | package real & maintained |
| AI-SEC-01 | AI output MUST pass SAST + secret scan with 0 high/critical (see `secure-coding.md`) | project SAST + secret scanner | 0 high/critical, 0 secrets |
| AI-SEC-02 | AI output MUST pass dependency CVE scan (see `secure-coding.md`) | project SCA scanner | 0 high/critical CVEs |
| AI-REV-01 | A human MUST read & understand the full diff before accept/merge — no rubber-stamping (see `code-review.md`) | human review of full diff | reviewer can explain every line |
| AI-SCOPE-01 | AI changes MUST stay within the requested scope; out-of-scope edits MUST be reverted | `git diff --stat` vs. prompt scope | no unrequested file changes |
| AI-CTX-01 | Project rules MUST live in a committed context file, not in chat-only memory (see `agents-md.md`) | file exists in VCS | `AGENTS.md` present & current |
| AI-MARK-01 | No agent artifacts MUST remain: stray `TODO: AI` / `FIXME: agent`, placeholder stubs, or unverified invented APIs | grep + review | none present |

> **Forbidden**: shipping AI code you cannot explain; accepting a hallucinated package; fixing an AI bug without a regression test first (violates `tdd.md`); pasting secrets into a prompt or context; letting an agent merge its own PR; overriding a security finding without human review (violates `secure-coding.md`).

---

## 3. Verification Protocol

Run, in order, before accepting AI output. Feed any failure's exact error back to the agent and let it iterate — **cap at ~3 cycles**, then intervene manually (more cycles usually means the prompt or context is wrong, not the code).

```text
1. PARSE/COMPILE   project build / type-check          # zero new errors or warnings
2. TEST            project test runner + coverage      # AI-TST-01/02/03
3. LINT/FORMAT     project linter + formatter          # matches project style
4. DEP VERIFY      registry lookup for each new dep    # AI-DEP-01 (run BEFORE install)
5. SECURITY        SAST + secret scan + SCA            # AI-SEC-01/02
6. DIFF REVIEW     git diff (full) read by a human     # AI-REV-01, AI-SCOPE-01
```

The concrete commands belong to the language/tooling guide (e.g. `python.md`, `typescript.md`, `secure-coding.md`); the *why* behind each gate lives in its §0 owner. Do not re-derive either here.

---

## 4. Prompt Hygiene (this guide's core)

The prompt is a specification. Quality of output is bounded by quality of input. The four patterns below cover almost every codegen interaction.

### A. Specification-first
State the contract before asking for code: inputs, outputs, error behavior, constraints, and *where* the code/tests go. Replace "build a login system" with an enumerated API + token lifetimes + rate limits + the existing model to reuse + the file paths + "write the tests first."

### B. Scope-bounded
Name the files the agent **may** modify and the ones it **must not**. This is the primary defense against scope creep (AI-SCOPE-01).

```text
MODIFY ONLY:  src/api/products.ts, src/services/product.ts, tests/products.test.ts
DO NOT TOUCH: schema/migrations, package manifest, shared utils, other endpoints
```

### C. Plan-first (for complex tasks)
Ask the agent to list the files it will touch, the changes per file, risks, and assumptions — and to **wait for approval** before coding. Cheaper to correct a plan than a diff.

### D. Incremental
Decompose: Cart model → shipping calc → tax calc → CheckoutService → endpoint, each "tests first." One task per prompt keeps the agent focused and each step independently verifiable.

### Prompt anti-patterns

| Anti-pattern | Why it fails | Fix |
|--------------|--------------|-----|
| "Make it work" / "fix all the bugs" | no success criteria, unbounded | acceptance tests, one bug + repro per prompt |
| "Clean up this file" | unbounded scope | specify exact changes + scope list |
| "Use best practices" | subjective; agent guesses | name the practice or cite the rule/guide |
| Dumping the whole repo | context overflow, diluted focus | 5–10 relevant files; let the agent read more |
| Multiple tasks in one prompt | agent loses focus mid-task | split into separate prompts |

### Context budget
Agents have finite, lossy context. Prioritize: (1) project rules/`AGENTS.md`, (2) the files being changed, (3) the test that defines "done", (4) types/interfaces, (5) callers/imports, (6) verbatim error messages, (7) external API docs. Prefer interfaces over full implementations; reference files by path and let the agent read what it needs rather than pasting everything.

---

## 5. Hallucination & Supply-Chain Risk (this guide's core)

The failure mode unique to AI codegen: **confident fabrication**. The model emits plausible code referencing things that do not exist.

### A. Hallucinated dependencies → "slopsquatting"
Agents invent package names. Attackers register those names, so an invented import can resolve to malicious code. This is the single highest-severity AI-specific risk.

- **Verify every new dependency BEFORE install** (AI-DEP-01): confirm it exists on the registry, check download counts, publish/last-release date, and maintainer history. Recent + low-downloads + name suspiciously close to a popular package = likely typo/slop-squat → reject.
- Cross-reference the import against the library's *official* docs, not the agent's claim.
- Supply-chain *policy* (lockfiles, SBOM, signing, CVE thresholds, license gates) is owned by [`secure-coding.md`](guides://secure-coding.md) — apply it; do not let "the AI added it" lower the bar.

### B. Hallucinated APIs
The agent calls methods/flags/config keys that don't exist or changed across versions (stale training data). Mitigation: pin versions in `AGENTS.md`, paste the relevant API reference into context, and let the type-checker/compiler catch the rest (AI-SEC/test gates).

### C. Plausible-but-wrong logic
Code compiles and reads correctly but mishandles edge cases or silently changes behavior. Mitigation: edge-case tests written *before* acceptance (see `tdd.md`); review the full diff for behavior you didn't request.

### D. AI-typical security defects
AI code skews toward insecure patterns (string-concatenated SQL, missing output encoding, leftover placeholder secrets, `cors:*`/`debug:true`, weak crypto, dropped auth checks). Encode the forbidden patterns as rules in `AGENTS.md` and enforce with SAST/secret scanning per [`secure-coding.md`](guides://secure-coding.md). Treat AI output as untrusted third-party code.

---

## 6. Trust Calibration: when to trust, distrust, escalate

Match scrutiny to risk. The agent's confidence is not evidence.

| Change type | Risk | Required oversight |
|-------------|------|--------------------|
| Formatting / typos / renames | Low | automated checks sufficient |
| Bug fix with regression test | Low–Med | quick human review + test verification |
| New isolated feature | Medium | full review + coverage check (see `code-review.md`) |
| Public API / interface change | High | full review + design discussion |
| Auth / payments / crypto | Critical | security-focused review; prefer vetted libs over AI crypto |
| DB schema / migrations | Critical | human approval + rollback plan |
| CI/CD / infra changes | Critical | platform-team review + dry run |

**Distrust by default for:** novel architecture (agents follow patterns, they don't invent them), security-critical cryptography, performance hot paths (benchmark — don't trust AI optimization claims), regulatory/compliance code, and undocumented legacy systems. Resolve highly ambiguous requirements with humans *first*, then bring in the agent.

**Watch-for list** (agent failure modes to pattern-match in review): hallucinated APIs, plausible-but-wrong logic, over-engineering you didn't ask for, copy-paste duplication instead of reuse, stale/deprecated patterns, silent behavior changes alongside the requested change, and happy-path-only error handling.

---

## 7. Agent Workflows for Codegen (this guide's core)

The verified loop, scaled by autonomy. Each step's *policy* lives in its §0 owner; this is the AI orchestration.

```text
PLAN → CONTEXT → TEST-FIRST → GENERATE → VERIFY → DIFF-REVIEW → ITERATE(≤3) → ACCEPT
```

- **Solo / interactive (Copilot, Cursor, Claude Code in-session):** define criteria → write/generate failing tests → prompt with context+tests+scope → generate → run §3 → review diff → iterate → commit. Guardrails: pre-commit hooks ([`pre-commit.md`](guides://pre-commit.md)) + `AGENTS.md`.
- **Team / shared:** ticket → branch → agent with shared `AGENTS.md` → CI gates ([`ci-cd.md`](guides://ci-cd.md)) → optional AI reviewer → human review (architecture, security, business correctness) → merge. Guardrails: CI required checks, CODEOWNERS, branch protection.
- **Autonomous (background coding agents, delegated tasks):** agent works on its own branch, opens a **draft** PR, CI + AI review run automatically, human reviews before merge.

  > **Autonomous agent boundaries — MUST:** new branch (never main), draft PR (never direct merge), run all tests before opening, scope to the assigned issue, justify any new dependency.
  > **MUST NOT:** touch CI/CD, schemas, or delete data without human approval; access production or secrets; merge its own PR.

- **Multi-agent (planner / implementer / tester / reviewer / verifier):** only the implementer holds write access to source; the others are read + run. A skeptical *verifier* re-checks claims (re-runs tests, hunts missed edge cases) rather than accepting "done" at face value. Define each role with explicit, least-privilege tool access.

### Continuous improvement
Track signals — first-pass acceptance rate, rework cycles, coverage delta (never negative), security findings per PR (target 0 high/critical), hallucinated deps per sprint (target 0), CI pass-rate on AI PRs. When a failure recurs, encode the fix as a rule in `AGENTS.md` and prune the file periodically (stale rules actively mislead — see `agents-md.md`).

---

## 8. Deployment Checklist

Generated from §2 — one box per requirement ID.

- [ ] AI-PROMPT-01 — prompt had acceptance criteria + scope boundary
- [ ] AI-TST-01/02/03 — tests first, bug regression tests added, coverage not decreased
- [ ] AI-DEP-01 — every new dependency verified real & maintained (anti-slopsquat)
- [ ] AI-SEC-01/02 — SAST + secret scan clean, dependency CVE scan clean
- [ ] AI-REV-01 — human read the full diff and can explain every line
- [ ] AI-SCOPE-01 — no out-of-scope file changes
- [ ] AI-CTX-01 — `AGENTS.md` present, committed, and current
- [ ] AI-MARK-01 — no stray agent artifacts / unverified invented APIs
- [ ] Ran every §3 verification step; capped iteration at 3 and documented manual fixes

---

**End of AI-Assisted Coding Guidelines**
