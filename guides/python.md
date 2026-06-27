# Python Development Guidelines
Mandatory coding standards for Python: type-safe, documented, test-covered. Python 3.13+, uv, pytest, ruff, mypy, Dynaconf, pydoc, bandit, pip-audit.

---
name: python
title: Python Development Guidelines
version: 2.0
last_reviewed: 2026-06-05
kind: language
tools: [python@3.13, uv, pytest, ruff, mypy, dynaconf, pydoc, bandit, pip-audit]
requires:
  - tdd
  - hexagonal
  - secure-coding
  - error-handling
recommends:
  - logging
  - observability
  - comments
  - env-config
  - semver
  - performance
  - pre-commit
  - ci-cd
provides:
  - pep8
  - type-hints
  - google-docstrings
  - uv-workflow
  - dynaconf-config
---

> 🧭 Authored per [`CONVENTIONS.md`](guides://CONVENTIONS.md): shared concerns are referenced, not restated. This guide covers only what is unique to Python.

---

## 0. Prerequisites & References

Fetch and apply these **before** generating Python code. Their rules are assumed below and not repeated.

> 📎 **REQUIRED — fetch & apply first:**
> - [`tdd.md`](guides://tdd.md) — test-first, Red-Green-Refactor, regression-test-before-fix, coverage. *(Python binding: runner is `uv run pytest`.)*
> - [`hexagonal.md`](guides://hexagonal.md) — layering, ports/adapters, dependency inversion.
> - [`secure-coding.md`](guides://secure-coding.md) — supply chain, secrets, CVE policy. *(Python binding: `bandit`, `pip-audit`.)*
> - [`error-handling.md`](guides://error-handling.md) — error strategy & propagation.

> 📎 **RECOMMENDED — fetch when the task touches them:**
> - [`comments.md`](guides://comments.md) — docstring/API-doc policy *(binding: Google-style docstrings, `pydoc`)*
> - [`env-config.md`](guides://env-config.md) — config policy *(binding: Dynaconf)*
> - [`logging.md`](guides://logging.md) · [`observability.md`](guides://observability.md) · [`performance.md`](guides://performance.md) · [`pre-commit.md`](guides://pre-commit.md) · [`ci-cd.md`](guides://ci-cd.md) · [`semver.md`](guides://semver.md)

> 📎 **SEE ALSO:** [`cleanarch.md`](guides://cleanarch.md) · [`designpatterns.md`](guides://designpatterns.md) · [`uv.md`](guides://uv.md) · [`poetry.md`](guides://poetry.md) *(only if the project mandates Poetry instead of uv)*

---

## 1. Core Philosophies: PYTHON-FIRST

Python-specific principles only. TDD, security, error handling, and architecture come from §0.

- **P**ackage management: `uv` only — every command runs via `uv run`; dependencies via `uv add`. Never `pip`/`poetry`/`pipenv` directly.
- **Y**ield & comprehensions: prefer comprehensions and generators over manual loops; generators for large/streamed data.
- **T**ype hints: strict, modern typing (`list[X]`, `X | None`) on every public signature; verified by a real type checker (`mypy --strict` or `pyright`), **not** the linter.
- **H**ints & docs: complete Google-style docstrings on public modules/classes/functions (policy: `comments.md`); API docs via `pydoc`.
- **O**utward config: Dynaconf for all configuration (policy: `env-config.md`); no hardcoded values.
- **N**on-negotiable quality: `ruff check` + `ruff format` clean, type check clean, `pytest` green at the §2 coverage gate.
- **S**ecurity scans: `bandit` and `pip-audit` on every delivery (policy: `secure-coding.md`).

**Verified Code**: Agent-generated Python MUST pass every gate in §2 before delivery.

---

## 2. Requirements (MANDATORY, auditable)

RFC-2119 keywords. IDs `PY-<TOPIC>-<NN>`. Each row has a binary gate; rows binding a shared rule cite its owner.

| ID | Requirement | Verify | Gate |
|----|-------------|--------|------|
| PY-TST-01 | Every feature MUST be test-first (see `tdd.md`) | `uv run pytest` | exit 0, 0 skips |
| PY-TST-02 | Each bug MUST get a regression test before the fix (see `tdd.md`) | `uv run pytest` | failing→passing |
| PY-TST-03 | Business logic coverage MUST be 100% | `uv run pytest --cov --cov-fail-under=100` | exit 0 |
| PY-FMT-01 | Code MUST be formatted | `uv run ruff format --check .` | no diff |
| PY-LINT-01 | Linter MUST pass clean | `uv run ruff check .` | exit 0 |
| PY-TYP-01 | Public APIs MUST be fully typed, modern syntax | `uv run mypy --strict src/` | exit 0 |
| PY-DOC-01 | Public modules/classes/functions MUST have docstrings (see `comments.md`) | `uv run python -m pydoc <module>` | parses, no missing |
| PY-CFG-01 | No hardcoded config; Dynaconf only (see `env-config.md`) | review / grep | no literals |
| PY-SEC-01 | 0 high/medium bandit findings (see `secure-coding.md`) | `uv run bandit -r src/` | 0 high/medium |
| PY-SEC-02 | 0 known CVEs in deps (see `secure-coding.md`) | `uv run pip-audit` | 0 vulnerabilities |
| PY-DEP-01 | Lockfile in sync & resolvable | `uv lock --check` | in sync |
| PY-ARCH-01 | Domain imports no adapter/framework code (see `hexagonal.md`) | review / import-linter | no inward→outward |

> **Forbidden**: shipping implementation before its test (violates `tdd.md`), fixing a bug without a regression test first, using `ruff` as a substitute for a type checker, mutable default arguments, or `pip install` in place of `uv add`.

---

## 3. Verification Protocol

Run, in order, before presenting code. Fix → re-run until every gate is green.

```bash
uv run ruff format --check .        # PY-FMT-01
uv run ruff check .                 # PY-LINT-01
uv run mypy --strict src/           # PY-TYP-01  (ruff does NOT type-check)
uv run pytest --cov --cov-fail-under=100   # PY-TST-01/03
uv run bandit -r src/               # PY-SEC-01
uv run pip-audit                    # PY-SEC-02
uv lock --check                     # PY-DEP-01
```

The *why* behind each gate lives in its §0 owner; do not re-derive it here.

---

## 4. Project Structure

Idiomatic `src/` layout. Architectural principles (dependency direction, ports/adapters, acyclic deps) are owned by [`hexagonal.md`](guides://hexagonal.md); below is only their Python mapping.

```
project/
├── src/<package>/
│   ├── domain/          # pure business logic — no framework/IO imports (PY-ARCH-01)
│   ├── application/     # use cases, orchestrates ports
│   ├── adapters/        # db/http/cli implementations of ports
│   └── __init__.py
├── tests/               # mirrors src/ (see tdd.md)
│   ├── unit/
│   └── integration/
├── config/              # Dynaconf settings (see §5.E, env-config.md)
├── pyproject.toml       # single source for deps, ruff, mypy, pytest
├── uv.lock              # committed lockfile
└── README.md
```

- Group by domain/feature, not by type.
- Enforce the import boundary with `import-linter` (`uv add --dev import-linter`).

---

## 5. Python Specifics

The unique value of this guide.

### A. Toolchain — uv only
```bash
uv init / uv sync               # create env + install from lockfile
uv add httpx                    # add dep (updates uv.lock)
uv add --dev pytest ruff mypy   # dev deps
uv add "numpy>=2.0,<3.0"        # version constraint
uv run <cmd>                    # run inside the project env (ALWAYS prefix)
```
Multi-package repos use uv workspaces (`[tool.uv.workspace] members = ["packages/*"]`) — one shared env, consistent versions.

### B. Modern typing (Python 3.13)
Use built-in generics and union syntax. **The linter is not a type checker** — gate types with `mypy --strict` or `pyright`.

```python
from collections.abc import Callable, Iterable, Sequence
from typing import Protocol, TypeVar

def stats(data: Sequence[float]) -> tuple[float, float, int]:
    return min(data), max(data), len(data)

def first[T](items: list[T]) -> T | None:        # PEP 695 type params (3.12+)
    return items[0] if items else None

class SignalSource(Protocol):                     # structural typing for ports
    def read(self, n: int) -> bytes: ...
```

```toml
# pyproject.toml — single config source
[project]
requires-python = ">=3.13"

[tool.ruff]
target-version = "py313"
line-length = 100
[tool.ruff.lint]
select = ["E", "W", "F", "I", "ANN", "B", "C4", "UP", "SIM", "RUF"]
# Do NOT add removed rules (ANN101/ANN102 were dropped in modern ruff).

[tool.mypy]
strict = true
python_version = "3.13"

[tool.pytest.ini_options]
addopts = "--cov --cov-fail-under=100"
```

### C. Comprehensions & generators
Prefer comprehensions over `append` loops; use generators for large/one-pass data.
```python
even_squares = [x*x for x in range(10) if x % 2 == 0]   # list
unique = {len(w) for w in words}                        # set
price_usd = {k: v*1.1 for k, v in prices.items()}       # dict
total = sum(x*x for x in range(1_000_000))              # generator — O(1) memory
```
Avoid: wrapping a generator in `list()` just to feed `sum/any/all`; comprehensions used only for side effects (use a `for` loop); nesting beyond ~2 levels (use a named loop).

### D. Functional idioms & immutability
Pure functions and immutable data by default; return new objects rather than mutating inputs.
```python
from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class Config:                       # immutable record
    host: str
    port: int

def add(items: tuple[str, ...], x: str) -> tuple[str, ...]:
    return (*items, x)              # no mutation
```
**Footgun — mutable default arguments:**
```python
def f(x: int, acc: list[int] | None = None) -> list[int]:
    acc = [] if acc is None else acc      # never `acc: list = []` in the signature
    return [*acc, x]
```
Reach for `functools.partial`, `reduce`, and `itertools` for composition — but a named function beats a clever lambda whenever logic exceeds one expression.

### E. Configuration — Dynaconf binding
Policy (layering, secrets, env separation) is owned by [`env-config.md`](guides://env-config.md). Python binding:
```python
# config/__init__.py
from pathlib import Path
from dynaconf import Dynaconf, Validator

BASE = Path(__file__).resolve().parent.parent
settings = Dynaconf(
    settings_files=[BASE / "config/defaults.toml", BASE / "config.toml"],
    environments=True,            # [development] / [production]
    envvar_prefix="MYAPP",        # MYAPP_* overrides
    secrets=str(BASE / ".secrets.toml"),   # git-ignored
    validators=[Validator("port", must_exist=True, gt=0)],
)
settings.validators.validate()    # fail fast on import
```
Never commit `.secrets.toml`. Never hardcode magic numbers — they go in `defaults.toml`.

### F. Docstrings — pydoc binding
Policy is owned by [`comments.md`](guides://comments.md). In Python: Google-style docstrings on every public module/class/function; verify with `uv run python -m pydoc <module>` (must parse) and browse with `uv run python -m pydoc -b`. Add a `docs` target to the build for HTML generation; keep generated HTML out of git.

---

## 6. Tooling & Dependencies

Security/supply-chain *policy* → [`secure-coding.md`](guides://secure-coding.md); versioning → [`semver.md`](guides://semver.md). Python binding:

```bash
uv sync                 # install from lockfile (reproducible)
uv add <pkg> / --dev    # add (updates uv.lock)
uv lock --upgrade       # update to latest resolvable versions
uv lock --check         # PY-DEP-01: lockfile in sync
uv run bandit -r src/   # PY-SEC-01: static security scan
uv run pip-audit        # PY-SEC-02: CVE scan against deps
```
Commit `uv.lock`. Pin or constrain direct deps; let uv resolve the graph.

---

## 7. Quick Reference

```bash
uv sync                              # setup
uv run pytest                        # test
uv run ruff check . && uv run ruff format .   # lint + format
uv run mypy --strict src/            # type check
uv run python -m <package>           # run
uv run python -m pydoc -b            # docs
```

---

## 8. Deployment Checklist

Generated from §2 — one box per requirement ID.

- [ ] PY-FMT-01 — `ruff format --check` clean
- [ ] PY-LINT-01 — `ruff check` clean
- [ ] PY-TYP-01 — `mypy --strict` clean (real type checker, not ruff)
- [ ] PY-TST-01/02/03 — tests pass, bugs have regression tests, coverage ≥ gate
- [ ] PY-DOC-01 — public APIs documented, pydoc parses
- [ ] PY-CFG-01 — no hardcoded config (Dynaconf)
- [ ] PY-SEC-01/02 — bandit clean, pip-audit 0 CVEs
- [ ] PY-DEP-01 — `uv.lock` in sync, committed
- [ ] PY-ARCH-01 — domain layer free of adapter/framework imports
- [ ] Agent ran every §3 command and documented any fixes

---
**End of Python Guidelines**
