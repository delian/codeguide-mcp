# Z3 SMT Solver (Python) Guidelines
Mandatory standards for the Z3 SMT solver via Z3Py: declaring sorts, building constraints, checking sat/unsat/unknown, optimization, tactics, and solver performance. z3-solver 4.13+, Python 3.13+.

---
name: python-z3
title: Z3 SMT Solver (Python) Guidelines
version: 2.0
last_reviewed: 2026-06-05
kind: tooling
tools: [z3-solver@4.13, python@3.13]
requires: []
recommends:
  - python
  - performance
  - comments
provides:
  - z3-smt
  - constraint-modeling
  - sat-solving
  - optimization
---

> 🧭 Authored per [`CONVENTIONS.md`](guides://CONVENTIONS.md): shared concerns are referenced, not restated. This guide covers only what is unique to modelling and solving with Z3 in Python.

---

## 0. Prerequisites & References

Z3Py is a Python library, so the host project's Python standards apply in full. This guide does not repeat them.

> 📎 **RECOMMENDED — fetch when the task touches them:**
> - [`python.md`](guides://python.md) — the language binding: `uv` workflow, `mypy --strict`, `ruff`, `pytest`, packaging, security scans. **All Python gates (format, lint, type, test, CVE) apply to Z3 code unchanged.**
> - [`performance.md`](guides://performance.md) — measure-before-optimize, benchmarking, profiling. Bind it here: never claim a Z3 speedup you have not timed.
> - [`comments.md`](guides://comments.md) — docstring/API-doc policy. Bind it here: every constraint-building function documents the *model* it encodes (variables, their meaning, what sat/unsat means).

> 📎 **SEE ALSO:** [`parallelism.md`](guides://parallelism.md) *(multi-context solving)* · [`error-handling.md`](guides://error-handling.md) *(handling `unknown`/timeouts)*

Install: `uv add z3-solver` (the PyPI package ships the Z3 native library and Python bindings). Verify with `uv run python -c "import z3; print(z3.get_version_string())"`.

---

## 1. Core Philosophies: MODEL-FIRST

Z3-specific principles only. Test-first, typing, security, and packaging come from `python.md`.

- **M**odel explicitly: separate **constraint construction** (pure functions returning expressions) from **solving** (the `check()` call). Constraints become testable, composable, and cacheable.
- **O**utcome-checked: NEVER touch `solver.model()` before asserting `check() == sat`. Treat `sat`, `unsat`, and `unknown` as three distinct outcomes — `unknown` is not an error to swallow.
- **D**eclare sorts deliberately: pick the smallest theory that fits (Bool < BitVec < Int < Real < Array/quantifiers). The theory you choose dominates solve time.
- **E**liminate before solving: `simplify()`, propagate constants, and prefer quantifier-free encodings; reach for quantifiers and nonlinear arithmetic only when nothing weaker models the problem.
- **L**imit resources: every solver invocation that can run on adversarial or large input MUST set a `timeout` (and, where relevant, `max_memory`). An unbounded `check()` is a denial-of-service.

**Verified Code**: Z3 code MUST pass every gate in §2 before delivery.

---

## 2. Requirements (MANDATORY, auditable)

RFC-2119 keywords. IDs `Z3-<TOPIC>-<NN>`. Python toolchain rows defer to `python.md`; the rest are Z3-specific.

| ID | Requirement | Verify | Gate |
|----|-------------|--------|------|
| Z3-TST-01 | Every model MUST have tests covering `sat`, `unsat`, and (where reachable) `unknown` (see `tdd.md` via `python.md`) | `uv run pytest` | exit 0, 0 skips |
| Z3-CHK-01 | Code MUST assert `check() == sat` before calling `model()` | review / grep for `.model()` | no unguarded `.model()` |
| Z3-CHK-02 | `check()` results MUST handle `unknown` explicitly, not assume `unsat` | review | all three branches handled |
| Z3-LIM-01 | Solver calls on untrusted/large input MUST set a `timeout` | grep `set("timeout"` / `Solver(... timeout` | every external-input solve bounded |
| Z3-SORT-01 | Mixed `Int`/`Real` arithmetic MUST use explicit `ToReal`/`ToInt` | `uv run mypy --strict src/` + review | no implicit coercion |
| Z3-THR-01 | Z3 objects MUST NOT be shared across threads; use a per-thread `Context` or `.translate()` | review | no cross-context/thread sharing |
| Z3-DOC-01 | Each constraint builder MUST document its model (see `comments.md`) | `uv run python -m pydoc <module>` | docstrings parse, model described |
| Z3-PERF-01 | Any claimed optimization MUST be benchmarked (see `performance.md`) | timing test / `pytest-benchmark` | measured before/after |
| Z3-PY-01 | All `python.md` gates apply (format, lint, type, CVE, lock) | see `python.md` §2/§3 | each green |

> **Forbidden**: extracting a model without a `sat` check; treating `unknown` as `unsat`; an unbounded `check()` on external input; sharing solver/expr objects between threads; claiming a speedup with no measurement.

---

## 3. Verification Protocol

Run the **`python.md` §3 protocol** verbatim (`ruff format --check`, `ruff check`, `mypy --strict`, `pytest`, `bandit`, `pip-audit`, `uv lock --check`). Z3 adds no new tools — it is a library. The Z3-specific checks (Z3-CHK/LIM/SORT/THR) are enforced in review and the test suite, not by a separate binary.

If a gate fails: read the error, find the root cause, fix, re-run. Do not present until green.

---

## 4. Declaring Variables & Sorts

Pick the weakest sufficient theory — it is the single biggest lever on solve time.

```python
from z3 import (
    Int, Real, Bool, BitVec, Array, Const, Consts,
    IntSort, RealSort, BoolSort, BitVecSort, ArraySort,
    DeclareSort, EnumSort, Function,
)

x = Int("x")                     # unbounded integer (LIA/NIA)
r = Real("r")                    # rational (LRA/NRA)
b = Bool("b")                    # propositional
w = BitVec("w", 32)              # fixed-width, wraps mod 2**32 — models machine ints exactly
mem = Array("mem", IntSort(), BitVecSort(8))   # index sort → element sort

# Batch declaration
i, j, k = Ints("i j k")          # also Reals(...), Bools(...), BitVecs("a b", 16)

# Custom uninterpreted sort + enumerated sort
Node = DeclareSort("Node")
Color, (red, green, blue) = EnumSort("Color", ["red", "green", "blue"])
p, q = Consts("p q", Node)
```

**Sort selection guide:**
- **Bool** for pure SAT / propositional logic — fastest.
- **BitVec(n)** for machine arithmetic, overflow, bit manipulation, hashing — decidable and usually fast. Signed vs unsigned matters: use `ULT/UGT/ULE/UGE` and `URem/UDiv` for unsigned, the operators for signed.
- **Int/Real** for unbounded/symbolic arithmetic. Linear (LIA/LRA) is decidable and efficient; **nonlinear** (multiplying two variables) can be slow or `unknown`.
- **Array** for maps/memory; `Select(A, i)` / `Store(A, i, v)` (or `A[i]`).
- **Uninterpreted sort + `Function`** to abstract away values you don't need to reason about (see §8).

> **Footgun — Int vs Real:** `Int("x") + Real("y")` is ill-typed. Convert explicitly with `ToReal(x)` / `ToInt(r)` (Z3-SORT-01). Integer division and `%` follow Z3's Euclidean semantics, not Python's.

---

## 5. Building Constraints & the Solver API

```python
from z3 import Solver, And, Or, Not, Implies, If, Distinct, Sum, sat, unsat, unknown

x, y = Ints("x y")
s = Solver()
s.add(x > 0, y > 0)              # add accepts varargs or a list
s.add(And(x + y == 10, Implies(x > 5, y < 5)))
s.add(Distinct(x, y))           # pairwise !=  — the workhorse for "all different"

result = s.check()              # -> sat | unsat | unknown
if result == sat:
    m = s.model()
    print(m[x], m[y])           # model evaluation
    print(m.eval(x + y, model_completion=True))
elif result == unsat:
    ...                         # no assignment exists
else:                           # unknown: incomplete (timeout, nonlinear, quantifiers)
    ...                         # MUST NOT be treated as unsat (Z3-CHK-02)
```

Building blocks: `And/Or/Not/Implies/Xor`, `If(cond, then, else)` (functional ternary), `Distinct`, `Sum`, `Product`, comparisons. For "at most/least k true", use `AtMost(*bools, k)` / `AtLeast` / `PbLe`/`PbGe`/`PbEq` (pseudo-Boolean) rather than hand-rolled sums.

**Model extraction:** `m[x]` returns a Z3 value; convert with `.as_long()`, `.as_fraction()`, `.as_string()`, or `is_true(m[b])` for Bools. Use `m.eval(expr, model_completion=True)` to evaluate arbitrary expressions and fill unconstrained vars.

> Always close the loop in tests: re-substitute the model into the constraints and assert they hold. A solver bug or a modelling bug both surface here.

---

## 6. Incremental Solving: push / pop / assumptions

Reuse one solver across related queries — it keeps learned lemmas and is far cheaper than rebuilding.

```python
s = Solver()
s.add(base_constraints)

s.push()                        # checkpoint
s.add(hypothesis_a)
r1 = s.check()
s.pop()                         # discard hypothesis_a, keep base + lemmas

s.push()
s.add(hypothesis_b)
r2 = s.check()
s.pop()
```

**Assumption literals** are often better than push/pop for toggling constraints and are required for unsat cores:

```python
pa, pb = Bools("assume_a assume_b")
s.add(Implies(pa, hypothesis_a))
s.add(Implies(pb, hypothesis_b))
s.check(pa, pb)                 # solve under both; s.check(pa) toggles just a
```

---

## 7. Unsat Cores

When a query is `unsat`, extract the minimal conflicting subset to debug or explain it.

```python
s = Solver()
s.set(unsat_core=True)
s.assert_and_track(x > 10, "c_lower")     # name each tracked assertion
s.assert_and_track(x < 5,  "c_upper")
s.assert_and_track(y == x, "c_eq")
assert s.check() == unsat
print(s.unsat_core())          # [c_lower, c_upper] — c_eq is irrelevant
```

Use cores to pinpoint over-constrained models, drive minimal explanations, and build assumption-based incremental search.

---

## 8. Uninterpreted Functions (EUF)

Functions Z3 reasons about only by **consistency**: if `f(a) == b`, every `f(a)` equals `b`, with no assumption about implementation. Cheaper than expanding a concrete definition, and ideal for abstraction and equivalence checking.

```python
from z3 import Function, IntSort, ForAll

hash_fn = Function("hash", IntSort(), IntSort())   # abstract Int -> Int
a, b = Ints("a b")
s = Solver()
s.add(hash_fn(a) == hash_fn(b), a != b)            # sat: hash may collide
```

Use EUF to: model operations whose internals don't matter (`encrypt`, `read`/`write`, `transform`); check two implementations equivalent (assert `ForAll(x, impl1(x) == spec(x))`, then search for a counterexample where `impl2(x) != spec(x)` — `unsat` ⇒ equivalent); and build parameterized models where the function is declared once and reused across queries. Do **not** use EUF when correctness depends on the function's actual arithmetic — encode that as constraints instead.

---

## 9. Quantifiers

Quantifiers move you into semi-decidable territory and frequently yield `unknown`. Prefer quantifier-free encodings.

```python
from z3 import ForAll, Exists, Int, Array, IntSort

# AVOID when the domain is finite — unroll instead:
A = Array("A", IntSort(), IntSort())
finite = [A[i] >= 0 for i in range(n)]              # fast, quantifier-free

# Use quantifiers only for genuinely unbounded domains:
i = Int("i")
unbounded = ForAll(i, Implies(i >= 0, A[i] >= 0))
```

When you must quantify:
- Provide explicit **patterns/triggers** (`ForAll(xs, body, patterns=[...])`) to control instantiation.
- Try the **`qe`** (quantifier elimination) tactic to remove them before the SMT core runs (§10).
- Expect `unknown`; handle it (Z3-CHK-02) and consider bounding the domain.

---

## 10. Tactics

A tactic is a solving/simplification strategy; compose them into a pipeline and derive a solver.

```python
from z3 import Tactic, Then, simplify, And, Int

pipeline = Then("simplify", "propagate-values", "solve-eqs", "smt")
s = pipeline.solver()
s.add(constraints)
s.check()

# Domain-specialized one-shot tactics:
Tactic("qflia")   # quantifier-free linear integer arithmetic
Tactic("qfnra")   # quantifier-free nonlinear real arithmetic
Tactic("qfbv")    # quantifier-free bit-vectors
Tactic("sat")     # pure propositional
Tactic("qe")      # quantifier elimination

# Standalone simplification — cheap and almost always worth it before solving:
x = Int("x")
simplify(And(x + 5 > 10, x > 5, x > 3))            # -> x > 5  (redundancy removed)
```

List options with `describe_tactics()`. Picking the tactic that matches your theory (`qflia`, `qfbv`, …) routinely beats the default portfolio.

---

## 11. Optimization (Optimize)

`Optimize` extends the solver with objectives (MaxSMT / linear / pseudo-Boolean).

```python
from z3 import Optimize, Ints, If, Bool, sat

opt = Optimize()
x, y = Ints("x y")
opt.add(x + y <= 10, x >= 0, y >= 0)
h = opt.maximize(3 * y - 2 * x)        # also opt.minimize(...)
opt.add_soft(x == 5, weight=2)         # soft constraint (MaxSMT), optional weight/id
if opt.check() == sat:
    print(opt.model(), opt.upper(h))   # objective bound via the handle
```

- `maximize`/`minimize` return a **handle**; read the optimal bound with `opt.upper(h)`/`opt.lower(h)`.
- `add_soft` constraints may be violated; the solver minimizes total violated weight.
- Multiple objectives are optimized lexicographically by default (`opt.set("priority", "pareto")` for Pareto fronts).
- 0/1 selection problems (knapsack, assignment) model cleanly with `Bool` items and `Sum(If(item, value, 0) for ...)`.

---

## 12. Context Management & Thread Safety

Every Z3 object belongs to a `Context`. The default global context is fine single-threaded.

```python
from z3 import Context, Int, Solver

ctx = Context()                         # isolated context
x = Int("x", ctx)
s = Solver(ctx=ctx)
```

**Threads (Z3-THR-01):** NEVER share a solver or expression across threads. Give each thread its own `Context`, or `.translate(target_ctx)` objects into it. For CPU-bound parallel solving, prefer **processes** over threads. Cross-cutting concurrency policy lives in [`parallelism.md`](guides://parallelism.md); this is its Z3 binding.

```python
from concurrent.futures import ProcessPoolExecutor

def solve_case(payload):                # rebuild constraints inside the worker
    s = Solver()                        # fresh per-process context
    s.add(build(payload))
    s.set("timeout", 5000)
    return str(s.check())

with ProcessPoolExecutor() as ex:
    results = list(ex.map(solve_case, payloads))
```

---

## 13. Performance Tips

Optimize only against measurements (Z3-PERF-01, `performance.md`). High-leverage Z3 levers, roughly in order:

1. **Weaken the theory.** Bool/BitVec over Int over Real; quantifier-free over quantified; linear over nonlinear.
2. **`simplify()` and tactic pipelines** before the main solve (§10) — can collapse problem size dramatically.
3. **Incremental solving** (§6): one solver + push/pop or assumptions reuses learned clauses across related queries.
4. **Reuse declared variables and functions.** Build parameterized *generator* functions that declare sorts/`Function`s once and only vary the per-query constraints — avoids rebuilding identical AST.
5. **EUF** (§8) to abstract expensive subterms Z3 doesn't need to interpret.
6. **Bound the search:** `set("timeout", ms)`, `set("max_memory", mb)`, and handle `unknown` gracefully.
7. **Specialized tactics** (`qfbv`, `qflia`, …) matched to the problem.

Benchmark with a timing test or `pytest-benchmark`; record before/after numbers in the PR rather than asserting a "10-100x" speedup from memory.

---

## 14. Common Use Cases

Z3 is the engine behind many tasks; the value is in the *encoding*. Sketches (all model-construction; solve via §5):

- **Constraint satisfaction** (Sudoku, N-Queens, graph coloring): `Int`/`Bool` per cell + `Distinct`/`!=` + given values. `sat` ⇒ solution, `unsat` ⇒ no solution.
- **Scheduling**: start-time `Int`s, `start_i + dur_i <= start_j` for precedence, resource capacity via pseudo-Boolean sums; `Optimize` to minimize makespan.
- **Formal verification / equivalence**: assert the **negation** of the property and check for a counterexample — `unsat` ⇒ property holds, `sat` ⇒ `model()` is the counterexample. Use `BitVec` to match machine semantics exactly.
- **Symbolic execution**: path constraints as a conjunction of branch conditions; `check()` decides path feasibility; `model()` yields concrete inputs reaching that path. Use incremental push/pop per branch.
- **Optimization** (knapsack, assignment, MaxSMT): §11.

Keep each encoder a pure, documented function (Z3-DOC-01) returning `(variables, constraints)`, so it can be unit-tested independently of the solve.

---

## 15. Quick Reference

```python
from z3 import *

x = Int("x"); r = Real("r"); b = Bool("b")           # declarations
w = BitVec("w", 32); A = Array("A", IntSort(), IntSort())
f = Function("f", IntSort(), IntSort())               # uninterpreted fn

s = Solver(); s.add(x > 0, x < 10)                    # build
s.set("timeout", 5000)                                # bound (ms)
match s.check():                                      # sat | unsat | unknown
    case z3.CheckSatResult() if s.check() == sat:
        m = s.model(); m[x].as_long()
s.push(); s.add(x == 7); s.check(); s.pop()           # incremental
s.set(unsat_core=True); s.assert_and_track(x < 0, "neg"); s.unsat_core()

opt = Optimize(); h = opt.maximize(x); opt.upper(h)   # optimization
simplify(And(x > 5, x > 3))                           # -> x > 5
Then("simplify", "smt").solver()                      # tactic pipeline
```

---

## 16. Deployment Checklist

Generated from §2 — one box per requirement ID.

- [ ] Z3-TST-01 — tests cover sat / unsat / unknown
- [ ] Z3-CHK-01 — no `model()` without a preceding `sat` check
- [ ] Z3-CHK-02 — `unknown` handled explicitly, never as `unsat`
- [ ] Z3-LIM-01 — timeouts set on all external-input solves
- [ ] Z3-SORT-01 — Int/Real mixing uses explicit conversion
- [ ] Z3-THR-01 — no Z3 objects shared across threads (per-context / `.translate`)
- [ ] Z3-DOC-01 — every constraint builder documents its model
- [ ] Z3-PERF-01 — optimizations benchmarked, not assumed
- [ ] Z3-PY-01 — all `python.md` gates green (format, lint, type, test, CVE, lock)
- [ ] Agent ran the §3 (`python.md`) protocol and documented any fixes

---
**End of Z3 SMT Solver (Python) Guidelines**
