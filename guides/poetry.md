# Poetry Project Management Guidelines
Mandatory standards for managing Python projects with Poetry: PEP 621 pyproject.toml, locked & reproducible installs, dependency groups, and the poetry-core build backend. Poetry 2.x, pyproject.toml, poetry.lock, poetry-core.

---
name: poetry
title: Poetry Project Management Guidelines
version: 2.0
last_reviewed: 2026-06-05
kind: tooling
tools: [poetry@2.x, poetry-core, pyproject.toml, poetry.lock]
requires: []
recommends:
  - python
  - uv
  - ci-cd
  - secure-coding
provides:
  - poetry-workflow
  - poetry-lockfiles
  - poetry-build
---

> 🧭 Authored per [`CONVENTIONS.md`](guides://CONVENTIONS.md): shared concerns are referenced, not restated. This guide covers only what is unique to Poetry — the project/dependency/build tool. Python language rules live in [`python.md`](guides://python.md).

---

## 0. Prerequisites & References

Poetry is a *tooling* guide: it owns the Poetry workflow, lockfile, and build backend. It does **not** own the language, testing, security, or CI rules — fetch those owners.

> 📎 **RECOMMENDED — fetch when the task touches them:**
> - [`python.md`](guides://python.md) — the language binding (types, layout, pytest/ruff/mypy). Poetry only manages its deps and venv.
> - [`secure-coding.md`](guides://secure-coding.md) — supply chain & CVE policy. *(Poetry binding: commit `poetry.lock`, `poetry check --lock`, `pip-audit`.)*
> - [`ci-cd.md`](guides://ci-cd.md) — pipeline policy. *(Poetry binding: `poetry install --no-root` then `poetry build`/`publish` in CI.)*
> - [`uv.md`](guides://uv.md) — **the faster modern alternative.** Many new projects now prefer uv (Rust-based, ~10–100× faster resolves, built-in Python management). Choose Poetry when you need its mature build/publish workflow or already standardized on it; otherwise evaluate uv first.

> 📎 **SEE ALSO:** [`semver.md`](guides://semver.md) — version-constraint policy · [`pre-commit.md`](guides://pre-commit.md) · [`env-config.md`](guides://env-config.md)

---

## 1. Core Philosophies: POETRY-FIRST

Poetry-specific principles only. Language, testing, architecture, and security policy come from §0.

- **P**EP 621 first: project metadata lives in the standard `[project]` table (Poetry 2.x); only Poetry-specific config (groups, sources, build settings) stays under `[tool.poetry]`.
- **O**ne lockfile, committed: `poetry.lock` is the source of truth for reproducible installs — always committed, never hand-edited.
- **E**very command through Poetry: dependencies via `poetry add`; tasks via `poetry run` (or an activated env). Never `pip install` into Poetry's venv.
- **T**iered dependency groups: separate `main`, `dev`, `test`, `docs` — only `main` ships in the built wheel.
- **R**esolve, then pin via lock: declare loose constraints (caret) in `pyproject.toml`; let the resolver pin exact versions in `poetry.lock`.
- **Y**our env is isolated: one virtualenv per project (prefer in-project `.venv`); the host interpreter stays clean.

**Verified Code**: Any Poetry project the agent delivers MUST pass every gate in §2.

---

## 2. Requirements (MANDATORY, auditable)

RFC-2119 keywords. IDs `POETRY-<TOPIC>-<NN>`. Each row has a binary gate; rows binding a shared rule cite its owner.

| ID | Requirement | Verify | Gate |
|----|-------------|--------|------|
| POETRY-STRUCT-01 | Project MUST use PEP 621 `[project]` table for metadata (Poetry 2.x) | review `pyproject.toml` | `[project]` present |
| POETRY-DEP-01 | `poetry.lock` MUST be committed and in sync with `pyproject.toml` | `poetry check --lock` | exit 0 |
| POETRY-DEP-02 | `pyproject.toml` MUST be valid | `poetry check` | exit 0 |
| POETRY-DEP-03 | Direct deps MUST declare bounded constraints, not `*` (see `semver.md`) | grep `pyproject.toml` | no `"*"` |
| POETRY-DEP-04 | Dev/test/docs tools MUST live in non-`main` groups | review groups | none in `[project.dependencies]` |
| POETRY-INST-01 | Install MUST be reproducible from the lockfile | `poetry install --sync` | exit 0 |
| POETRY-SEC-01 | 0 known CVEs in resolved deps (see `secure-coding.md`) | `poetry run pip-audit` | 0 vulnerabilities |
| POETRY-BUILD-01 | Package MUST build with `poetry-core` backend | `poetry build` | sdist + wheel produced |
| POETRY-ENV-01 | Project MUST use an isolated venv; no global installs | `poetry env info --path` | venv exists |

> **Forbidden**: editing `poetry.lock` by hand; committing without `poetry check --lock` passing; `pip install` into the Poetry venv; unbounded `*` constraints on direct deps; shipping dev/test deps in the `main` group.

---

## 3. Verification Protocol

Run, in order, before presenting a Poetry project. Fix → re-run until every gate is green.

```bash
poetry check                  # POETRY-DEP-02: pyproject validity
poetry check --lock           # POETRY-DEP-01: lock in sync with pyproject
poetry install --sync         # POETRY-INST-01: reproducible env (prunes stray deps)
poetry run pip-audit          # POETRY-SEC-01: CVE scan (see secure-coding.md)
poetry build                  # POETRY-BUILD-01: sdist + wheel
```

Language gates (ruff/mypy/pytest) run *inside* the env via `poetry run …` — their policy is owned by [`python.md`](guides://python.md), not here.

---

## 4. Project Layout

A Poetry project is a standard Python `src/` layout plus two Poetry artifacts. Architecture/layout principles are owned by [`python.md`](guides://python.md); Poetry only adds the manifest and lockfile.

```
project/
├── src/<package>/        # source (layout/architecture: see python.md)
├── tests/                # tests (see tdd.md via python.md)
├── pyproject.toml        # PEP 621 [project] + [tool.poetry] (single manifest)
├── poetry.lock           # COMMITTED, resolver-generated, never hand-edited
└── README.md
```

- `poetry new <name> --src` scaffolds the `src/` layout; `poetry init` adds Poetry to an existing project.
- Multi-package monorepos: use **path dependencies** (`package = { path = "../other", develop = true }`) per sub-package, each with its own `pyproject.toml`. (Poetry has no first-class workspace concept — this is where [`uv.md`](guides://uv.md) workspaces are often preferred.)

---

## 5. Poetry Specifics

The unique value of this guide.

### A. `pyproject.toml` — PEP 621 + `[tool.poetry]` (Poetry 2.x)

Poetry 2.0 (Jan 2025) aligns with PEP 621: metadata goes in the standard `[project]` table. Keep only Poetry-specific concerns under `[tool.poetry]`.

```toml
[project]                              # PEP 621 — the modern, portable way
name = "myapp"
version = "1.0.0"
description = "A production-ready Python application"
authors = [{ name = "Your Name", email = "you@example.com" }]
readme = "README.md"
license = "MIT"
requires-python = ">=3.13"
dependencies = [                       # PEP 508 strings for runtime deps
  "pydantic>=2.6,<3.0",
  "httpx>=0.27,<0.28",
]

[project.scripts]                      # console entry points → `poetry run serve`
serve = "myapp.main:serve"

[tool.poetry]                          # Poetry-only settings (packages, source, etc.)
packages = [{ include = "myapp", from = "src" }]

[build-system]                         # poetry-core is the build backend
requires = ["poetry-core>=2.0"]
build-backend = "poetry.core.masonry.api"
```

> Legacy projects use `[tool.poetry]` for *all* metadata (Poetry 1.x style, with `version = "^3.13"` for python). On Poetry 2.x, migrate metadata to `[project]`; `poetry check` warns about the mix. Do not declare a field in both tables.

### B. Dependency groups

Only `main` (`[project.dependencies]`) ships in the wheel. Tooling goes in named groups, optional groups are not installed by default.

```toml
[tool.poetry.group.dev.dependencies]
ruff = "^0.6"
mypy = "^1.11"

[tool.poetry.group.test.dependencies]
pytest = "^8.3"
pytest-cov = "^5.0"

[tool.poetry.group.docs]
optional = true                        # skipped unless --with docs
[tool.poetry.group.docs.dependencies]
mkdocs-material = "^9.5"
```

```bash
poetry add httpx                       # add to main
poetry add --group dev ruff mypy       # add to a group
poetry install --only main             # production install (no dev tools)
poetry install --with docs             # include an optional group
poetry install --without dev,test      # exclude groups
```

### C. Version constraints

| Constraint | Means | Use |
|---|---|---|
| `^2.6.0` | `>=2.6.0,<3.0.0` | default for libraries/apps (allows compatible updates) |
| `~2.6.0` | `>=2.6.0,<2.7.0` | patch-only updates |
| `>=2.6,<3` | explicit range | when caret semantics are wrong |
| `==2.6.0` | exact pin | sparingly (reproducibility comes from the lockfile, not pins) |
| `*` | any version | **forbidden** on direct deps (POETRY-DEP-03) |

Extras: `httpx = { version = "^0.27", extras = ["http2"] }`. Constraint *policy* (when to widen/narrow) is owned by [`semver.md`](guides://semver.md).

### D. Lockfile discipline

`poetry.lock` records exact versions + hashes for every transitive dep → byte-identical installs everywhere. It is **resolver-generated**: never edit by hand.

```bash
poetry lock                  # (re)resolve and write poetry.lock
poetry lock --no-update      # re-lock after editing pyproject WITHOUT bumping pins
poetry check --lock          # POETRY-DEP-01: lock matches pyproject (CI gate)
poetry update                # bump within constraints, then re-lock
poetry update requests       # update one package only
poetry show --outdated       # what could move
```

Lockfile integrity = supply-chain integrity (hashes pin artifacts) — see [`secure-coding.md`](guides://secure-coding.md). Always commit `poetry.lock`; review its diff on dependency changes.

### E. Virtualenv management

One isolated venv per project. Prefer in-project `.venv` for discoverability and editor integration.

```bash
poetry config virtualenvs.in-project true   # create .venv/ inside the project
poetry env use 3.13                          # bind a specific interpreter
poetry env info --path                       # POETRY-ENV-01: locate the venv
poetry run <cmd>                             # run a command inside the env (no activation)
poetry env activate                          # print/activate the env (Poetry 2.x)
poetry env remove --all                      # tear down envs
```

> `poetry shell` was removed from core in Poetry 2.0. Use `poetry run` (preferred, no activation), `poetry env activate`, or install the `poetry-plugin-shell` plugin if you need the old behavior.

### F. Build & publish

```bash
poetry build                 # POETRY-BUILD-01: → dist/*.tar.gz (sdist) + *.whl (wheel)
poetry config pypi-token.pypi <token>    # store a PyPI token
poetry publish               # upload dist/ to PyPI
poetry publish --build       # build + publish in one step
poetry publish -r my-repo    # publish to a configured private repo
```

Configure a private index once: `poetry config repositories.my-repo https://pypi.example.com/`. The build backend is `poetry-core` (declared in `[build-system]`) — a standalone, PEP 517 backend with no runtime dependency on Poetry itself, so consumers build your sdist without installing Poetry.

### G. Plugins

Poetry 2.x ships a slimmer core; several former built-ins are now plugins:

- `poetry-plugin-export` — `poetry export -f requirements.txt --output requirements.txt` (for tools that consume `requirements.txt`).
- `poetry-plugin-shell` — restores `poetry shell`.
- Install with `poetry self add <plugin>`; manage with `poetry self show plugins`.

---

## 6. Quick Reference

```bash
poetry new myapp --src       # scaffold src/ project   |  poetry init  (existing project)
poetry install --sync        # reproducible install from poetry.lock
poetry add <pkg> [--group g] # add dependency, update lock
poetry update [<pkg>]        # bump within constraints
poetry check --lock          # lock ↔ pyproject in sync
poetry run pytest            # run tasks inside the env (see python.md)
poetry build && poetry publish   # package & release
```

---

## 7. Deployment Checklist

Generated from §2 — one box per requirement ID.

- [ ] POETRY-STRUCT-01 — metadata in PEP 621 `[project]` table
- [ ] POETRY-DEP-01 — `poetry.lock` committed and in sync (`poetry check --lock`)
- [ ] POETRY-DEP-02 — `poetry check` clean
- [ ] POETRY-DEP-03 — no `*` constraints on direct deps
- [ ] POETRY-DEP-04 — dev/test/docs tools in non-`main` groups
- [ ] POETRY-INST-01 — `poetry install --sync` reproducible
- [ ] POETRY-SEC-01 — `pip-audit` reports 0 CVEs
- [ ] POETRY-BUILD-01 — `poetry build` produces sdist + wheel
- [ ] POETRY-ENV-01 — isolated venv (no global installs)
- [ ] Agent ran every §3 command and documented any fixes

---
**End of Poetry Project Management Guidelines**
