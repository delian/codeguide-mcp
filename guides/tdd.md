# Test-Driven Development (TDD) Guidelines
The canonical owner of test-first discipline: Red-Green-Refactor, regression-test-before-fix, test structure (AAA/GWT), test doubles, the test pyramid, coverage policy, and test naming — language-agnostic.

---
name: tdd
title: Test-Driven Development (TDD) Guidelines
version: 2.0
last_reviewed: 2026-06-05
kind: cross-cutting
tools: [xunit-style-runner, coverage-tool, ci-runner]
requires: []
recommends:
  - mutmut
  - ci-cd
  - code-review
  - e2e-testing
provides:
  - red-green-refactor
  - regression-test-first
  - coverage-policy
  - test-naming
  - test-doubles
  - test-pyramid
  - aaa-pattern
---

> 🧭 Authored per [`CONVENTIONS.md`](guides://CONVENTIONS.md). This guide **owns** test-first discipline; other guides reference it instead of restating it. It spends its tokens on TDD methodology and references neighbouring concerns (mutation testing, CI/CD, code review, e2e) rather than re-explaining them.

---

## 0. Prerequisites & References

This is a foundational cross-cutting guide; it has no hard prerequisites. It defines the test-first contract that language and framework guides bind to (each supplies its own runner, e.g. `uv run pytest`, `go test`, `vitest`, `cargo test`).

> 📎 **RECOMMENDED — fetch when the task touches them:**
> - [`mutmut.md`](guides://mutmut.md) — mutation testing. Coverage proves lines *ran*; mutation testing proves assertions *catch defects*. Do not measure test quality here.
> - [`ci-cd.md`](guides://ci-cd.md) — pipeline stages, quality gates, merge blocking, coverage trend enforcement. This guide states *what* must be gated; `ci-cd.md` owns *how* the pipeline runs it.
> - [`code-review.md`](guides://code-review.md) — human review of test quality and the test-first audit.
> - [`e2e-testing.md`](guides://e2e-testing.md) — end-to-end / browser / full-system workflow testing (the top of the pyramid). This guide keeps E2E to its role in the pyramid only.

> 📎 **SEE ALSO:** [`parallelism.md`](guides://parallelism.md) *(testing concurrent/race-condition code)* · [`todo.md`](guides://todo.md) *(tracking deferred test cases)* · [`comments.md`](guides://comments.md) *(documenting regression tests)*

---

## 1. Core Philosophies: TDD-FIRST

- **Red-Green-Refactor**: ALWAYS write a failing test → make it pass with minimal code → refactor with tests green. Never skip a phase.
- **Test before code**: the test exists and fails before any production line that satisfies it.
- **Regression shield**: every bug gets a failing test that reproduces it *before* the fix lands, and that test stays forever.
- **Test behaviour, not implementation**: assert observable outcomes through public APIs, not internal call counts or private state — so tests survive refactoring.
- **Fast feedback**: unit tests run in milliseconds; a slow suite stops being run.
- **Isolated & deterministic**: each test sets up its own state, depends on no other test, and never flakes.
- **Tests are documentation**: a test name and body explain intended behaviour better than a comment.
- **Coverage is a floor, not a goal**: high coverage is necessary but insufficient; assertion strength (mutation testing — see [`mutmut.md`](guides://mutmut.md)) is what proves the tests work.

**Verified Code**: agent-generated code MUST pass every gate in §2 before delivery, and the agent MUST have run the suite — never claim tests pass without executing them.

---

## 2. Requirements (MANDATORY, auditable)

RFC-2119 keywords. IDs `TDD-<TOPIC>-<NN>`. `<runner>`/`<coverage-cmd>` are supplied by the binding language guide. Each row has a binary gate.

| ID | Requirement | Verify | Gate |
|----|-------------|--------|------|
| TDD-TST-01 | Every behaviour MUST have a test written **before** its implementation | review git/commit order; `<runner>` | exit 0, 0 skips |
| TDD-TST-02 | A new test MUST be observed to FAIL for the right reason before code makes it pass | run test pre-implementation | red→green proven |
| TDD-TST-03 | Each bug MUST get a reproducing regression test **before** the fix | `<runner>` on the new test | failing→passing |
| TDD-TST-04 | Regression tests MUST stay in the suite permanently and reference the bug ID | grep test name/comment for issue ref | present |
| TDD-TST-05 | Tests MUST be independent of execution order | `<runner>` shuffled (e.g. random seed) | exit 0 |
| TDD-TST-06 | No test SHALL be skipped, `xfail`-ignored, or commented out without a linked tracking issue | grep for skip/xfail markers | 0 unjustified |
| TDD-TST-07 | Tests MUST assert behaviour, not implementation detail (no call-count assertions on collaborators unless the interaction *is* the contract) | review | no brittle interaction asserts |
| TDD-COV-01 | Line + branch coverage MUST meet the project floor (default ≥ 90%; critical paths 100%) | `<coverage-cmd>` | ≥ threshold |
| TDD-COV-02 | Critical paths (auth, payments, data-loss, security) MUST be 100% covered | `<coverage-cmd>` per-path | 100% |
| TDD-COV-03 | Test quality SHOULD be validated by mutation testing on critical modules (see `mutmut.md`) | per `mutmut.md` | ≥ kill-rate target |
| TDD-STRUCT-01 | Tests MUST follow Arrange-Act-Assert (or Given-When-Then), one behaviour per test | review | structured, single-behaviour |
| TDD-NAME-01 | Test names MUST describe behaviour: `<unit>_<scenario>_<expected>` or "should … when …" | review | descriptive |
| TDD-CI-01 | Tests MUST run on every commit and block merge on failure (see `ci-cd.md`) | CI config | gate enforced |

> **Forbidden — never deliver code that:** ships implementation before its test (TDD-TST-01); has a test that was never seen to fail (TDD-TST-02); fixes a bug without a prior regression test (TDD-TST-03); contains skipped/flaky/order-dependent tests; asserts on internal implementation detail; or chases a coverage number with assertion-free tests.

---

## 3. The Red-Green-Refactor Cycle

The atomic loop. Run it once per behaviour — one test at a time, never batch several behaviours into one cycle.

```
  ┌─────────────────────────────────────────────┐
  │ 🔴 RED      write ONE failing test           │
  │             run it → confirm it FAILS for    │
  │             the RIGHT reason (not a typo)    │
  ├─────────────────────────────────────────────┤
  │ 🟢 GREEN    write the MINIMAL code to pass   │
  │             run → confirm green. No extra    │
  │             features, no speculation.        │
  ├─────────────────────────────────────────────┤
  │ 🔵 REFACTOR improve structure/naming/dupes   │
  │             tests stay green; add NO new      │
  │             behaviour. Re-run after each step.│
  └────────────────────┬────────────────────────┘
                       └──► next behaviour
```

**RED — write a failing test (TDD-TST-01/02).** Write the test first; run it; confirm it fails, and that the failure message is the *expected assertion failure*, not a compile error or typo masquerading as red. Start with the simplest meaningful case.

**GREEN — minimal pass.** Write only enough code to satisfy the current test. Hard-coding a return value to get green is legitimate — the next test forces generalisation ("fake it till you make it"). Resist adding code "you'll need later"; that code has no test and violates TDD-TST-01.

**REFACTOR — clean with a safety net.** With the bar green, remove duplication, clarify names, extract methods, apply patterns. Re-run after every change. The rule that separates refactor from feature work: **no test changes its expected outcome during refactor.** If a test must change, you are adding behaviour — go back to RED.

**Triangulation.** When one example is not enough to drive the right general solution, add a second and third concrete test that force you to replace a constant with real logic.

**Queue, don't batch.** When you think of further cases mid-cycle, record them (a TODO list — see [`todo.md`](guides://todo.md)) and finish the current cycle first.

---

## 4. Regression Testing — Bug = Test (TDD-TST-03/04)

**Every bug is a missing test.** The fix is not the deliverable; the *failing-then-passing test* is.

```
🐛 bug found → ✍️ write a test that REPRODUCES it (RED, fails now)
            → ✅ confirm it fails for the bug's reason
            → 🔧 apply the minimal fix
            → 🟢 confirm the test passes
            → 🔒 keep the test forever, tagged with the issue ID
```

Why test-first even for bugs: a test you only see pass *after* the fix may pass for the wrong reason (or test nothing). Seeing it fail first proves it actually exercises the defect.

**Each regression test MUST** (TDD-TST-04) carry the issue/ticket reference in its name or a docstring, so the test's purpose survives the bug tracker. Keep the documentation minimal — the detailed "why" belongs in the commit/issue per [`comments.md`](guides://comments.md), not pasted into every test.

```pseudocode
TEST "divide raises on zero divisor — bug #1234"
  // Was: returned Infinity instead of raising. Fixed 2026-01-18.
  ASSERT THROWS DivByZero WHEN Calculator().divide(10, 0)
END TEST
```

Regression tests slot into whichever pyramid level reproduces the bug — a unit test for a logic bug, an integration test for a wiring bug, an E2E test (see [`e2e-testing.md`](guides://e2e-testing.md)) for a workflow bug. Reproduce at the **lowest** level that captures it.

---

## 5. Test Structure & Naming

### A. Arrange-Act-Assert / Given-When-Then (TDD-STRUCT-01)

Every test has three visually separated phases and exercises **one** behaviour.

```pseudocode
TEST "transfer moves funds and logs once"
  // ARRANGE  (GIVEN)
  source = Account(balance: 1000); target = Account(balance: 500)
  bank = Bank()
  // ACT      (WHEN)  — exactly one action under test
  bank.transfer(source, target, 200)
  // ASSERT   (THEN)
  ASSERT source.balance EQUALS 800
  ASSERT target.balance EQUALS 700
  ASSERT bank.log.length EQUALS 1
END TEST
```

AAA and GWT are the same shape; GWT phrasing suits behaviour/BDD specs. Multiple `assert` statements are fine when they verify one logical outcome; asserting two *different* behaviours means two tests.

### B. Naming (TDD-NAME-01)

Names read as specifications: `unit_scenario_expected` or "should <behaviour> when <condition>".

```
✅ divide_throws_when_divisor_is_zero
✅ "authenticate returns null when credentials are invalid"
❌ test1   ❌ addTest   ❌ "it works"
```

### C. Layout

Mirror the source tree; separate by speed/scope so fast tests can run alone.

```
tests/
├── unit/          # fast, isolated, mock external deps   (§6 base)
├── integration/   # real collaborators (db, cache)       (§6 middle)
├── e2e/           # full workflows — see e2e-testing.md   (§6 top)
└── fixtures/      # factories, builders, shared test data
```

---

## 6. The Test Pyramid & Coverage

### A. Pyramid

```
        ╱╲       E2E      — few, slow, brittle; full user journeys → e2e-testing.md
       ╱──╲      Integration — some, medium; component interaction, real deps
      ╱────╲     Unit     — many, fast, cheap; one function/method in isolation
     ╱──────╲
```

Push assertions **down** to the cheapest level that can prove them. Many unit tests, fewer integration, fewest E2E — an inverted pyramid (mostly E2E) is slow, flaky, and gives vague failures.

- **Unit** — one unit, no external systems, milliseconds; substitute collaborators with test doubles (§7).
- **Integration** — real collaborators (test DB/cache/queue); set up and tear down state per test so order independence (TDD-TST-05) holds.
- **E2E** — owned by [`e2e-testing.md`](guides://e2e-testing.md); keep them few and reserved for critical end-to-end paths.

### B. Coverage policy (TDD-COV-01/02)

- Default floor **≥ 90%** line + branch; **100%** on critical paths (auth, payments, data-loss, security).
- Coverage is a floor, not a target — never write assertion-free tests to inflate it (forbidden in §2).
- **Coverage measures execution, not correctness.** A line can be 100% covered by a test that asserts nothing. Validate assertion strength with **mutation testing** (TDD-COV-03 — see [`mutmut.md`](guides://mutmut.md)); a surviving mutant is a missing assertion. Do not configure or score mutation testing here — that is `mutmut.md`'s job.
- Enforce the floor and "coverage must not decrease" in CI (see [`ci-cd.md`](guides://ci-cd.md)).

---

## 7. Test Doubles

Use the narrowest double that isolates the unit. Prefer fakes/stubs over interaction-verifying mocks — the latter couple tests to implementation and break on refactor (see TDD-TST-07).

| Double | Purpose | Asserts on? |
|--------|---------|-------------|
| **Dummy** | Fills a required parameter; never used | nothing |
| **Stub** | Returns canned data to steer the path under test | the result |
| **Spy** | A stub that records calls for later inspection | recorded calls |
| **Mock** | Pre-programmed with expected interactions; fails if not met | the interaction |
| **Fake** | Lightweight working implementation (in-memory repo) | the result |

```pseudocode
// Stub — steer behaviour, assert on outcome (resilient)
db = StubDatabase({ "123": { name: "John" } })
ASSERT UserService(db).getUser("123").name EQUALS "John"

// Mock — only when the INTERACTION is the contract (e.g. "must send email")
email = MockEmailService()
UserService(email).resetPassword("u@x.com")
VERIFY email.send WAS_CALLED_ONCE WITH(to: "u@x.com", subject: CONTAINS "Reset")
```

**Rule of thumb:** assert on *return values and state* (stub/fake) by default; reach for a mock only when the side effect itself is the behaviour being specified. Don't mock types you don't own — wrap them behind a port and fake the port.

---

## 8. Anti-Patterns

| Anti-pattern | Why it hurts | Fix |
|---|---|---|
| Testing implementation details (call counts, private state) | Breaks on every refactor; tests nothing users see | Assert observable behaviour via the public API |
| Test interdependence / shared mutable state | Order-dependent, flaky; fails TDD-TST-05 | Each test builds its own fresh state |
| Excessive setup obscuring intent | Hides what's tested; slow | Factories/builders/fixtures (§5.C) |
| Multiple behaviours per test | Vague failures; violates STRUCT-01 | One behaviour per test |
| Over-mocking everything | Tests the mocks, not the code | Prefer fakes/stubs; let some integration be real |
| Coverage-driven assertion-free tests | Green that proves nothing | Mutation testing (`mutmut.md`) exposes them |
| Ignoring/skipping a red test | Hides regressions | Fix or delete; skips need a tracked issue (TDD-TST-06) |
| Flaky tests (time, ordering, network, randomness) | Erodes trust in the suite | Control the clock/seed; isolate I/O |

### Design smells revealed by hard-to-write tests

When a test is painful to write, the *design* — not the test — is usually wrong.

| Test pain | Design problem | Remedy |
|---|---|---|
| Huge setup | Too many dependencies | Dependency injection; smaller units |
| Can't test without a real DB | Coupled to infrastructure | Repository/port behind an interface |
| Need to test a private method | Hidden responsibility | Extract it to its own unit |
| Test breaks on every refactor | Asserting implementation | Test behaviour |

---

## 9. Agent TDD Protocol

For every code-generation task the agent MUST:

1. Decompose the requirement into testable behaviours.
2. For each: write one failing test, run it, confirm meaningful red (TDD-TST-02).
3. Write minimal code; run; confirm green.
4. Refactor with the suite green.
5. Repeat until the feature is complete.
6. Run the full suite + coverage and **paste/observe the actual result** before claiming success — never assert "tests pass" without running them.
7. For any bug: regression test first (§4).

Test quality and the test-first audit are also a human-review concern — see [`code-review.md`](guides://code-review.md).

---

## Deployment Checklist

Generated from §2 — one box per requirement ID.

- [ ] TDD-TST-01 — every behaviour was test-first (commit order shows test before impl)
- [ ] TDD-TST-02 — each new test was observed to fail for the right reason
- [ ] TDD-TST-03 — every fixed bug has a regression test written before the fix
- [ ] TDD-TST-04 — regression tests are permanent and reference their issue ID
- [ ] TDD-TST-05 — suite passes under shuffled/random order
- [ ] TDD-TST-06 — no unjustified skipped/xfail/commented tests
- [ ] TDD-TST-07 — tests assert behaviour, not implementation detail
- [ ] TDD-COV-01 — coverage meets the project floor (≥ 90% default)
- [ ] TDD-COV-02 — critical paths 100% covered
- [ ] TDD-COV-03 — mutation testing run on critical modules (see `mutmut.md`)
- [ ] TDD-STRUCT-01 — AAA/GWT, one behaviour per test
- [ ] TDD-NAME-01 — test names describe behaviour
- [ ] TDD-CI-01 — tests run on every commit and block merge (see `ci-cd.md`)
- [ ] Agent ran the full suite and observed real results before delivery

---
**End of Test-Driven Development (TDD) Guidelines**
