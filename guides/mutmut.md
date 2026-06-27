# Mutation Testing with mutmut Guidelines
Mandatory standards for mutation testing Python code with mutmut: prove assertion strength, kill survivors, gate on mutation score. Python 3.13+, uv, pytest, mutmut 3.x, coverage.

---
name: mutmut
title: Mutation Testing with mutmut Guidelines
version: 2.0
last_reviewed: 2026-06-05
kind: cross-cutting
tools: [mutmut@3, python@3.13, uv, pytest, coverage]
requires: []
recommends:
  - tdd
  - python
provides:
  - mutation-testing
  - mutation-score
---

> 🧭 Authored per [`CONVENTIONS.md`](guides://CONVENTIONS.md): shared concerns are referenced, not restated. This guide canonically owns **mutation testing** — others reference it instead of re-explaining survivors, scores, or kill workflow.

---

## 0. Prerequisites & References

mutmut measures *test-suite quality*; it presupposes a test-first workflow and a Python toolchain.

> 📎 **RECOMMENDED — fetch when the task touches them:**
> - [`tdd.md`](guides://tdd.md) — test-first, Red-Green-Refactor, and **coverage**. Coverage proves a line *ran*; mutation testing proves an assertion would *fail* if the line were wrong. mutmut is how you verify the test quality that `tdd.md` demands.
> - [`python.md`](guides://python.md) — the language binding: `uv`, `pytest`, `pyproject.toml`, project layout. All commands here run via `uv run`.

> 📎 **SEE ALSO:** [`ci-cd.md`](guides://ci-cd.md) — pipeline gating · [`performance.md`](guides://performance.md) — keeping the test suite fast enough to mutate.

This guide does **not** own coverage, test-first discipline, or the Python toolchain — it owns only what is unique to mutation testing.

---

## 1. Core Philosophies: MUTATION-FIRST

Mutation-testing-specific principles only. Test-first and coverage policy come from §0.

- **M**utants are test-suite bugs: a *surviving* mutant means a code change went undetected — the fault is in your assertions, not (usually) the code.
- **U**v-powered: every command runs via `uv run` (binding: `python.md`).
- **T**argeted runs: mutate specific modules and use coverage data to keep the feedback loop fast.
- **A**nalyze every survivor: each survivor is read, then either killed by a new test or documented as equivalent — never ignored.
- **T**est logic, not boilerplate: spend mutation budget on algorithms and business rules, not getters/glue.
- **E**quivalent mutants are explained: a mutant that cannot change observable behavior is documented or refactored away — not silenced to inflate the score.

**Verified Code**: code is "mutation-proof" only when it meets the §2 score gates with every non-equivalent survivor killed.

---

## 2. Requirements (MANDATORY, auditable)

RFC-2119 keywords. IDs `MUT-<TOPIC>-<NN>`. Each row has a binary gate.

| ID | Requirement | Verify | Gate |
|----|-------------|--------|------|
| MUT-TST-01 | The test suite MUST be green before any mutation run (see `tdd.md`) | `uv run pytest` | exit 0, 0 skips |
| MUT-RUN-01 | Mutation testing MUST run on changed business logic | `uv run mutmut run` | completes, results recorded |
| MUT-SCORE-01 | Core logic/algorithms MUST reach ≥ 90% mutation score | `uv run mutmut results` (see §4.C) | ≥ 90% on core modules |
| MUT-SCORE-02 | Total project mutation score MUST be ≥ 80% | `uv run mutmut results` | ≥ 80% |
| MUT-SURV-01 | Every surviving mutant MUST be killed or documented as equivalent | `uv run mutmut results` + review | no undocumented survivors |
| MUT-PRAGMA-01 | `# pragma: no mutate` MUST be limited to the allowed cases in §3.B | `grep -rn "pragma: no mutate"` + review | each occurrence justified |
| MUT-CI-01 | CI MUST fail when the mutation-score gate is not met (see `ci-cd.md`) | CI run on mutation job | red below threshold |

> **Forbidden**: shipping with a 0% mutation score, ignoring survivors in business logic, using `# pragma: no mutate` to hide a missing test, or skipping a mutation run after a logic refactor.

---

## 3. Configuration & Scope

### A. Configure in `pyproject.toml`

Single config source (binding: `python.md`):

```toml
[tool.mutmut]
paths_to_mutate = ["src/"]
tests_dir = "tests/"
runner = "uv run pytest -x"      # -x: stop on first kill → faster feedback
# mutmut 3.x discovers coverage automatically; keep tests fast (see performance.md)
```

### B. `# pragma: no mutate` — allowed cases only (MUT-PRAGMA-01)

Use it ONLY where a mutant is genuinely untestable or noise — never to mask a missing assertion:

```python
VERSION = "1.0.0"                       # pragma: no mutate  — metadata, no logic
logger.debug("starting process")        # pragma: no mutate  — logging, not behavior
```

Allowed: version/metadata strings, log statements, defensive branches impossible to trigger in unit tests. Everything else: write the test.

---

## 4. Reading & Acting on Results (owned)

### A. The kill workflow

A surviving mutant is a missing test. Drive it to green:

```bash
uv run mutmut run                 # run mutations (MUT-RUN-01)
uv run mutmut results             # list survivors with IDs
uv run mutmut show <id>           # see the exact source change that survived
```

For each survivor:
1. **Read** `mutmut show <id>` — understand which logic changed and was not caught.
2. **Write a failing test** that asserts the behavior the mutant broke (Red, per `tdd.md`).
3. **Re-run** `uv run mutmut run` and confirm the mutant is now killed.

Typical example: `return age >= 18` survives mutation to `return age > 18` because no test exercises the boundary. The fix is a boundary assertion (`assert is_adult(18) is True`), not more coverage — the line was already covered.

### B. Equivalent mutants (the hard case)

An *equivalent* mutant changes the source but cannot change observable behavior, so no test can kill it. Do not let it drag the score down:

- **Prefer to refactor it away.** An equivalent mutant often signals redundant or unclear code — e.g. `if x > 0` ⇄ `if x >= 1` for integers. Pick the canonical form and the ambiguity disappears.
- **If the code must stay**, document why the mutant is equivalent (comment or test note) so reviewers and CI accounting treat it as expected, not as an unaddressed survivor.

Equivalent mutants are the reason a 100% score is not always achievable or worth chasing — judge by §4.C targets, not perfection.

### C. Mutation score targets

Mutation score = killed ÷ (total − equivalent). Targets by component:

| Component | Target mutation score |
|----------------|-----------------------|
| Core logic / algorithms | 90% – 100% |
| API handlers / adapters | 70% – 80% |
| CLI / UI layer | 50% – 70% |
| Total project | ≥ 80% |

Gate the numbers that matter (MUT-SCORE-01/02): hold core logic to ≥ 90% even if peripheral layers pull the headline number around.

---

## 5. Performance & Scope Control (owned)

Mutation testing is inherently slow — it re-runs the suite once per mutant. Keep the loop usable:

1. **Lean on coverage**: mutmut skips mutating lines no test covers, so a covered, fast suite is the biggest lever.
2. **Narrow scope during development** — mutate only what you changed:
   ```bash
   uv run mutmut run --paths-to-mutate src/logic.py
   ```
3. **Keep tests fast**: mock external APIs and databases so each suite run is milliseconds, not seconds (see `performance.md`).
4. **Full sweep in CI, targeted runs locally**: developers mutate the module they touched; CI mutates the whole `paths_to_mutate` set.

---

## 6. CI Integration (owned)

Pipeline mechanics and stage policy are owned by [`ci-cd.md`](guides://ci-cd.md); the mutation-specific binding:

```yaml
- name: Mutation testing
  run: |
    uv run pytest                 # MUT-TST-01: suite must be green first
    uv run mutmut run             # MUT-RUN-01
    uv run mutmut results         # survivors + score
    uv run mutmut junit > mutation.xml   # publish to CI test reporting
```

Fail the job below the §4.C threshold (MUT-CI-01). mutmut exits non-zero when mutants survive; for a numeric score gate, parse `mutmut results` in a small wrapper script and exit non-zero under target. Publish `mutmut html` as a build artifact for survivor triage.

---

## 7. Quick Reference

```bash
uv run mutmut run                              # run mutation testing
uv run mutmut run --paths-to-mutate src/core/  # narrow scope
uv run mutmut results                          # summary + survivor IDs
uv run mutmut show <id>                         # diff for one mutant
uv run mutmut html                              # HTML survivor report
uv run mutmut junit > mutation.xml             # JUnit XML for CI
```

---

## 8. Deployment Checklist

Generated from §2 — one box per requirement ID. No new requirements here.

- [ ] MUT-TST-01 — test suite green before mutating
- [ ] MUT-RUN-01 — mutation run completed on changed logic
- [ ] MUT-SCORE-01 — core logic ≥ 90% mutation score
- [ ] MUT-SCORE-02 — total project ≥ 80% mutation score
- [ ] MUT-SURV-01 — every survivor killed or documented equivalent
- [ ] MUT-PRAGMA-01 — every `# pragma: no mutate` justified per §3.B
- [ ] MUT-CI-01 — CI fails below the score gate

---
**End of Mutation Testing with mutmut Guidelines**
