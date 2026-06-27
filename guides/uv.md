# uv Development Guidelines
Mandatory standards for the uv Python package & project manager: reproducible installs, lockfiles, workspaces, tool management. uv (latest), pyproject.toml, uv.lock.

---
name: uv
title: uv Development Guidelines
version: 2.0
last_reviewed: 2026-06-05
kind: tooling
tools: [uv, pyproject.toml, uv.lock]
requires: []
recommends:
  - python
  - ci-cd
  - secure-coding
provides:
  - uv-workflow
  - python-packaging
  - uv-lockfiles
  - uv-workspaces
---

> 🧭 Authored per [`CONVENTIONS.md`](guides://CONVENTIONS.md): shared concerns are referenced, not restated. This guide owns only the **uv toolchain** — the Python *language* is owned by [`python.md`](guides://python.md), CI policy by [`ci-cd.md`](guides://ci-cd.md), supply-chain policy by [`secure-coding.md`](guides://secure-coding.md).

---

## 0. Prerequisites & References

uv is the toolchain binding for Python. Fetch the language and cross-cutting owners when the task touches them; do not re-derive their rules here.

> 📎 **RECOMMENDED — fetch when the task touches them:**
> - [`python.md`](guides://python.md) — the Python language, idioms, typing, project layout. uv is *how* you run/install; Python is *what* you write. *(All `uv run …` commands in `python.md` assume this guide.)*
> - [`secure-coding.md`](guides://secure-coding.md) — supply-chain, lockfile integrity, CVE policy. *(uv binding: `uv.lock` hashes, `--locked`/`--frozen`, `uv lock --check`.)*
> - [`ci-cd.md`](guides://ci-cd.md) — pipeline policy, caching, reproducibility. *(uv binding: `astral-sh/setup-uv`, `UV_CACHE_DIR`, `uv sync --frozen`.)*

> 📎 **SEE ALSO:** [`poetry.md`](guides://poetry.md) *(legacy alternative — migrate to uv)* · [`semver.md`](guides://semver.md) · [`pre-commit.md`](guides://pre-commit.md) · [`dockerfile.md`](guides://dockerfile.md)

---

## 1. Core Philosophies: UV-FIRST

uv-specific principles only. TDD, security, architecture, and Python idioms come from §0.

- **One tool, one source of truth:** uv replaces pip, pip-tools, pipenv, virtualenv, pyenv, and pipx. `pyproject.toml` declares intent; `uv.lock` pins reality. Never edit `uv.lock` by hand.
- **Always `uv run`:** never call a project's interpreter or tools directly. `uv run` guarantees the environment is synced first, so commands run against the locked graph.
- **Reproducible by default:** commit `uv.lock`. In CI, containers, and releases use `--frozen`/`--locked` so the lockfile is law, never silently mutated.
- **uv owns Python itself:** `uv python` installs and pins interpreters — no pyenv. `.python-version` pins the project's interpreter.
- **Ephemeral tools, isolated:** developer tools (ruff, mypy, pre-commit) run via `uvx`/`uv tool`, isolated from project dependencies.

**Verified Code:** any uv project the agent generates MUST pass every gate in §2 before delivery.

---

## 2. Requirements (MANDATORY, auditable)

RFC-2119 keywords. IDs `UV-<TOPIC>-<NN>`. Each row has a binary gate; rows binding a shared rule cite its owner.

| ID | Requirement | Verify | Gate |
|----|-------------|--------|------|
| UV-DEP-01 | `uv.lock` MUST exist and be committed | `git ls-files uv.lock` | tracked |
| UV-DEP-02 | Lockfile MUST be in sync with `pyproject.toml` | `uv lock --check` | exit 0 |
| UV-DEP-03 | Installs MUST resolve from the lockfile, never mutate it in CI/build (see `secure-coding.md`) | `uv sync --frozen` | exit 0, no lock change |
| UV-DEP-04 | Direct deps MUST be added via `uv add`, never hand-edited or `pip install` | review / `git diff pyproject.toml` | uv-managed only |
| UV-SEC-01 | Lockfile MUST carry hashes for integrity (see `secure-coding.md`) | `grep -c "hash = " uv.lock` | > 0 |
| UV-PY-01 | Interpreter version MUST be pinned | test `-f .python-version` | present |
| UV-BLD-01 | Buildable projects MUST declare a `[build-system]` and build clean | `uv build` | wheel + sdist produced |
| UV-WS-01 | Workspace members MUST resolve to one shared lockfile (see `python.md` §4 for layering) | `uv lock --check` at root | single `uv.lock` |
| UV-RUN-01 | Project commands MUST run via `uv run` (synced env) | review / grep CI scripts | no bare `python`/`pytest` |

> **Forbidden**: editing `uv.lock` by hand; `pip install` into a uv project; committing without re-running `uv lock`; using `uv sync` (unfrozen) in CI/release builds; installing dev tools into project deps instead of `uv tool`/`uvx`.

---

## 3. Verification Protocol

Run before presenting a uv project. Fix → re-run until every gate is green.

```bash
uv lock --check                 # UV-DEP-02: lockfile in sync with pyproject.toml
uv sync --frozen                # UV-DEP-03: reproducible install, lock unchanged
git ls-files uv.lock            # UV-DEP-01: lockfile committed
test -f .python-version         # UV-PY-01: interpreter pinned
uv build                        # UV-BLD-01: builds (skip for app-only repos)
```

The *why* behind reproducibility and integrity lives in [`secure-coding.md`](guides://secure-coding.md); CI wiring in [`ci-cd.md`](guides://ci-cd.md).

---

## 4. Project Manifest & Lockfile

uv is **`pyproject.toml`-native** (PEP 621). The manifest is the only thing you edit; the lockfile is generated.

```toml
# pyproject.toml
[project]
name = "myapp"
version = "1.0.0"
requires-python = ">=3.13"
dependencies = [
    "httpx>=0.28",
    "pydantic>=2.9",
]

[build-system]                  # required only if the package is built/published
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.uv]
package = true                  # false → app-only repo, no wheel built
```

- **`uv.lock`** is a cross-platform, hashed, fully-resolved snapshot of the *entire* dependency graph. Commit it. Never hand-edit. It is uv-specific (not `requirements.txt`); regenerate with `uv lock`.
- **`requires-python`** drives resolution; uv resolves a graph valid for the whole range.
- Pin/constrain **direct** deps in `[project.dependencies]`; let uv resolve the transitive graph.

### Dependency groups (PEP 735)

Separate non-shipping deps with `[dependency-groups]`. The `dev` group is special — synced by default.

```toml
[dependency-groups]
dev   = ["pytest>=8", "mypy>=1.11"]
test  = ["pytest-cov>=5", "faker>=22"]
docs  = ["sphinx>=8"]
```

```bash
uv add --dev pytest             # add to the dev group
uv add --group docs sphinx      # add to a named group
uv sync --group test            # include the test group
uv sync --no-dev                # production: omit dev group
uv sync --all-groups            # everything (e.g. full CI matrix)
```

Optional **extras** (`[project.optional-dependencies]`) are for *consumers* of your package (`pip install myapp[redis]`); **groups** are for *your* workflow and are never published. Use extras for shippable optional features, groups for tooling.

---

## 5. uv Workflow (the heart of the guide)

### A. Project lifecycle

```bash
uv init my-app                  # new project: pyproject.toml, .python-version, src/, README
uv init --lib my-lib            # library layout (src/, build-system)
uv init --app                   # application layout (default)
uv add httpx                    # add dep → updates pyproject.toml + uv.lock + syncs
uv add "numpy>=2.0,<3.0"        # with a constraint
uv add --dev pytest ruff mypy   # dev-group deps
uv remove httpx                 # drop a dep (re-resolves)
uv sync                         # install/refresh .venv from the lockfile
uv run <cmd>                    # run inside the project env (ALWAYS prefix)
uv run pytest                   # e.g. tests — synced first, no manual activate
```

`uv run` auto-creates `.venv`, syncs it to the lock, then executes — no `source .venv/bin/activate` needed. Never invoke the interpreter directly.

### B. Lockfile commands

```bash
uv lock                         # (re)generate uv.lock from pyproject.toml
uv lock --upgrade               # bump all deps to latest resolvable
uv lock --upgrade-package httpx # bump one package
uv lock --check                 # UV-DEP-02: assert lock matches pyproject.toml
uv sync --frozen                # install from lock; error if lock is stale (no edit)
uv sync --locked                # assert lock is up-to-date, then install
uv export --no-dev > requirements.txt   # compat export for tools that need it
```

`--frozen` (don't even look at `pyproject.toml`) and `--locked` (verify lock is current, then install) are the **reproducibility switches** — mandatory in CI, Docker, and release builds (UV-DEP-03).

### C. Python version management (no pyenv)

```bash
uv python install 3.13          # download a managed interpreter
uv python list                  # list available / installed
uv python pin 3.13              # write .python-version (UV-PY-01)
uv run --python 3.12 pytest     # one-off run under another version
```

uv fetches and caches standalone interpreters; `.python-version` pins the project. CI matrices simply `uv python install ${{ matrix.python-version }}`.

### D. Tool management — `uvx` / `uv tool`

Run developer tools isolated from project deps. `uvx` = `uv tool run`.

```bash
uvx ruff check .                # ephemeral, cached, isolated run
uvx ruff@0.6.0 check .          # pin a tool version for a run
uv tool install ruff            # persistent install on PATH (replaces pipx)
uv tool install pre-commit
uv tool list / upgrade / uninstall
```

Keep linters/formatters/scaffolders **out** of `[dependency-groups]` when they don't need the project's environment — install them with `uv tool`. Tools that must see project code (e.g. `pytest`, `mypy`) stay in the dev group and run via `uv run`.

### E. Inline script dependencies (PEP 723)

Single-file scripts declare their own deps; `uv run` builds a throwaway env.

```python
# /// script
# requires-python = ">=3.13"
# dependencies = ["httpx", "rich"]
# ///
import httpx
from rich import print
```

```bash
uv run script.py                # resolves the inline deps, runs, discards env
uv add --script script.py httpx # edit the inline metadata block
```

---

## 6. Workspaces (multi-package repos)

A workspace is several packages sharing **one lockfile and one resolved environment** — consistent versions across members, fast local cross-edits. Use it for monorepos and layered architectures. The *architecture* (which layer may depend on which) is owned by [`python.md`](guides://python.md) §4 and its `hexagonal.md` reference — uv only enforces the *resolution*.

```toml
# root pyproject.toml
[tool.uv.workspace]
members = ["packages/*"]
exclude = ["packages/legacy"]

[tool.uv.sources]               # resolve member deps from the workspace, not PyPI
myapp-domain = { workspace = true }
```

```bash
uv sync                         # builds the shared env for all members
uv add --package myapp-api fastapi      # add a dep to one member
uv run --package myapp-api uvicorn ...  # run in a member's context
uv lock --check                 # UV-WS-01: single root lockfile in sync
```

- Members declare normal `[project]` manifests; intra-repo deps resolve via `[tool.uv.sources] … { workspace = true }`.
- One `uv.lock` at the root governs every member — never per-member lockfiles.
- `[tool.uv.sources]` also pins to a local `path`, `git`, or branch when a dep isn't on PyPI.

---

## 7. Building & Publishing

```bash
uv build                        # UV-BLD-01: build sdist + wheel into dist/
uv build --wheel                # wheel only
uv publish                      # upload dist/* (token via UV_PUBLISH_TOKEN)
uv publish --index testpypi     # publish to a configured alternate index
```

- Building requires a `[build-system]` (e.g. `hatchling`); `[tool.uv].package = false` marks app-only repos that never build a wheel.
- Versioning policy (when to bump major/minor/patch) is owned by [`semver.md`](guides://semver.md).
- Never put real tokens in `pyproject.toml`; pass via env/CI secrets (see `secure-coding.md`).

---

## 8. CI/CD & Containers (binding only)

Pipeline policy, caching, and reproducibility rules are owned by [`ci-cd.md`](guides://ci-cd.md). uv binding:

```yaml
# GitHub Actions
- uses: astral-sh/setup-uv@v6
  with: { enable-cache: true }   # caches ~/.cache/uv keyed on uv.lock
- run: uv python install 3.13
- run: uv sync --frozen --all-groups   # UV-DEP-03: reproducible
- run: uv run pytest
```

```dockerfile
# Multi-stage: copy the static uv binary, install frozen, then a slim runtime.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev          # prod: no dev group, lock is law
```

- Cache `UV_CACHE_DIR` (`~/.cache/uv`) keyed on `uv.lock` for fast, deterministic installs.
- Always `--frozen` in CI/build so a stale or tampered manifest fails loudly instead of silently re-resolving.
- Container/Dockerfile specifics → [`dockerfile.md`](guides://dockerfile.md).

---

## 9. Migration from pip / Poetry

| From | Action | uv equivalent |
|------|--------|---------------|
| `requirements.txt` | `uv init` then `uv add -r requirements.txt` | `[project.dependencies]` + `uv.lock` |
| `pip install X` | `uv add X` | manifest + lock updated atomically |
| `pip install -e .` | `uv sync` | editable by default for the project |
| `python -m venv && pip` | `uv venv` / `uv sync` | managed `.venv`, no activate |
| `pyenv` | `uv python install/pin` | managed interpreters + `.python-version` |
| `pipx install X` | `uv tool install X` | isolated global tools |
| Poetry `[tool.poetry.dependencies]` | move to PEP 621 `[project]`, `uv lock` | standard `pyproject.toml` + `uv.lock` |
| Poetry `poetry.lock` | regenerate with `uv lock` | `uv.lock` (delete `poetry.lock`) |
| Poetry dev-deps groups | `[dependency-groups]` (PEP 735) | `uv add --dev` / `--group` |

After migrating: delete `requirements*.txt` / `poetry.lock` / `setup.py` once `uv.lock` is committed and `uv sync --frozen` is green. See [`poetry.md`](guides://poetry.md) only if a project still mandates Poetry.

---

## 10. Footguns

- **Hand-editing `uv.lock`** → always regenerate via `uv add`/`uv lock`; a hand-edited lock loses hash integrity (UV-SEC-01).
- **`uv pip install`** (the pip-compat shim) bypasses the manifest/lock → use `uv add` for project deps; reserve `uv pip` for throwaway/legacy interop.
- **Bare `pytest`/`python` in CI** → not synced to the lock; always `uv run` (UV-RUN-01).
- **`uv sync` (unfrozen) in CI** → may silently re-resolve and mutate the lock; use `--frozen`/`--locked` (UV-DEP-03).
- **Per-member lockfiles in a workspace** → only the root `uv.lock` is authoritative (UV-WS-01).
- **Forgetting to commit `uv.lock` with a `pyproject.toml` change** → `uv lock --check` fails for everyone else.

---

## 11. Quick Reference

```bash
uv init / uv add X / uv sync         # create · add dep · install from lock
uv run <cmd>                         # run in the project env (always prefix)
uv lock --check && uv sync --frozen  # CI gate: in-sync + reproducible
uv python install 3.13 && uv python pin 3.13   # manage interpreter
uvx ruff check . ; uv tool install pre-commit  # ephemeral / global tools
uv build && uv publish               # package & release
uv export --no-dev > requirements.txt          # compat export
```

---

## 12. Deployment Checklist

Generated from §2 — one box per requirement ID. No new requirements.

- [ ] UV-DEP-01 — `uv.lock` exists and is committed
- [ ] UV-DEP-02 — `uv lock --check` in sync with `pyproject.toml`
- [ ] UV-DEP-03 — `uv sync --frozen` reproducible, lock unchanged in CI/build
- [ ] UV-DEP-04 — all direct deps added via `uv add` (no hand edits / `pip install`)
- [ ] UV-SEC-01 — lockfile carries hashes (integrity)
- [ ] UV-PY-01 — interpreter pinned in `.python-version`
- [ ] UV-BLD-01 — `uv build` produces wheel + sdist (if buildable)
- [ ] UV-WS-01 — workspace resolves to one root lockfile
- [ ] UV-RUN-01 — project commands run via `uv run`
- [ ] Agent ran every §3 command and documented any fixes

---
**End of uv Guidelines**
