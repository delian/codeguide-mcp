# GNU Make Guidelines
Mandatory standards for writing modern, maintainable Makefiles — as a task runner and as a build system. GNU Make 4.x, POSIX shell.

---
name: make
title: GNU Make Guidelines
version: 2.0
last_reviewed: 2026-06-05
kind: tooling
tools: [gnu-make@4.4, posix-sh]
requires: []
recommends:
  - ci-cd
  - c
  - cpp
  - cmake
provides:
  - makefiles
  - make-task-runner
  - pattern-rules
  - phony-targets
---

> 🧭 Authored per [`CONVENTIONS.md`](guides://CONVENTIONS.md): shared concerns are referenced, not restated. This guide covers only what is unique to GNU Make.

---

## 0. Prerequisites & References

GNU Make has no hard prerequisites — it is a tool, not a language stack. Fetch the references below **when the task touches them**.

> 📎 **RECOMMENDED — fetch when the task touches them:**
> - [`ci-cd.md`](guides://ci-cd.md) — pipeline policy. *(Make binding: phony targets such as `make ci`, `make lint`, `make test` are the stable CI entrypoints; the pipeline calls Make, not raw tool invocations.)*
> - [`c.md`](guides://c.md) · [`cpp.md`](guides://cpp.md) — when Make drives a C/C++ compile; the compiler flags and toolchain rules live there.
> - [`cmake.md`](guides://cmake.md) — **graduate to CMake** once a C/C++ build needs cross-platform generation, package discovery, or multi-config output (see §7). Make remains the task-runner wrapper around it.

> 📎 **SEE ALSO:** [`secure-coding.md`](guides://secure-coding.md) (shell-injection, quoting) · [`error-handling.md`](guides://error-handling.md) · [`comments.md`](guides://comments.md) (`##` self-documenting targets) · [`tdd.md`](guides://tdd.md) (if Make orchestrates a test suite) · [`bash.md`](guides://bash.md) (recipe shell scripting).

---

## 1. Core Philosophies

GNU Make-specific principles only. Cross-cutting policy (CI, security, testing) comes from §0.

- **Two jobs, one tool.** Make is either a **build system** (files-from-files, incremental, dependency-tracked) or a **task runner** (named commands: `make test`, `make lint`, `make deploy`). The common case today is the task runner. Know which you are writing — the rules differ (file targets vs `.PHONY`).
- **Let Make do the graph.** Express *what depends on what*; never hand-script the order. A correct prerequisite graph gives you incremental rebuilds and `-j` parallelism for free.
- **Declarative over imperative.** A recipe is a last resort, not the structure. Reach for built-in functions (`$(wildcard)`, `$(patsubst)`, `$(foreach)`) before shelling out.
- **Portability is deliberate.** Pin `SHELL`, prefer POSIX utilities, and assume GNU Make (not BSD/`pmake`) — declare it. Recursive `make` across directories is an anti-pattern (Miller, *Recursive Make Considered Harmful*): prefer `include`d sub-makefiles sharing one graph.
- **Self-documenting.** Every public target carries a `## description` so `make help` is the front door.

**Verified Makefiles**: agent-generated Makefiles MUST pass every gate in §2 before delivery.

---

## 2. Requirements (MANDATORY, auditable)

RFC-2119 keywords. IDs `MAKE-<TOPIC>-<NN>`. Each row has a binary gate; rows binding a shared rule cite its owner.

| ID | Requirement | Verify | Gate |
|----|-------------|--------|------|
| MAKE-SYN-01 | Makefile MUST parse with no undefined-variable warnings | `make --warn-undefined-variables -n` | exit 0, no warnings |
| MAKE-TAB-01 | Recipe lines MUST be indented with a literal TAB, not spaces | `cat -A Makefile` / `make -n` | no "missing separator" |
| MAKE-PHONY-01 | Every non-file (task) target MUST be declared `.PHONY` | review / grep `.PHONY` | all task targets listed |
| MAKE-GOAL-01 | `.DEFAULT_GOAL` MUST be set to a safe target (typically `help`) | `make` with no args | no destructive default |
| MAKE-ERR-01 | `.DELETE_ON_ERROR:` MUST be present (no half-written targets) | `grep '^.DELETE_ON_ERROR:' Makefile` | present |
| MAKE-DEP-01 | File targets MUST declare real prerequisites (incremental, no over-build) | run target twice | 2nd run: "Nothing to be done" |
| MAKE-PAR-01 | Build MUST be parallel-safe | `make -j$(nproc)` from clean | succeeds, no races |
| MAKE-SEC-01 | Shell-expanded variables MUST be quoted; `SHELL` pinned (see `secure-coding.md`) | review / `grep '^SHELL' Makefile` | quoted; SHELL set |
| MAKE-DOC-01 | Public targets MUST have `##` help text (see `comments.md`) | `make help` | every target listed |
| MAKE-STRUCT-01 | Makefiles > ~100 lines SHOULD split into `make/*.mk` includes | review | main file thin |

> **Forbidden**: a recipe indented with spaces; a task target (`test`, `clean`, `deploy`) that is not `.PHONY` (a same-named file silently disables it); recursive `$(MAKE) -C subdir` when a shared-graph `include` would do; unquoted `rm -rf $(DIR)` (empty var → `rm -rf /`); a default goal that builds or deploys instead of showing help.

---

## 3. Verification Protocol

Run, in order, before presenting a Makefile. Fix → re-run until clean.

```bash
make --warn-undefined-variables -n        # MAKE-SYN-01: parses, no undefined vars
cat -A Makefile | grep -nP '^ +\S'        # MAKE-TAB-01: should match NOTHING (spaces in recipe)
make help                                 # MAKE-GOAL-01/DOC-01: default goal + documented targets
make clean && make && make                # MAKE-DEP-01: 2nd build == "Nothing to be done"
make clean && make -j"$(nproc)"           # MAKE-PAR-01: parallel build is race-free
```

The *why* behind security/test/CI gates lives in their §0 owners; do not re-derive it here.

---

## 4. Anatomy: targets, prerequisites, recipes

```makefile
target: prerequisite1 prerequisite2     # rule head: what depends on what
	recipe-line                         # TAB-indented shell; only runs if a prereq is newer
```

- **Target** — usually a file to produce; for a task runner, a phony name.
- **Prerequisite** — a file or target that must be up-to-date first. The graph is built from these.
- **Recipe** — shell commands, each on its own **TAB-indented** line. Each line runs in its **own** shell (no state carries across lines unless joined with `\`).
- Prefix a recipe line with `@` to silence the echo, `-` to ignore its exit status. Default goal is the **first** target unless `.DEFAULT_GOAL` overrides it.

---

## 5. GNU Make Specifics — the unique value

### A. Variable flavors: `=` vs `:=` vs `?=` vs `+=`
The single most common source of subtle bugs.

```makefile
A := $(shell date)     # simple/immediate: expanded ONCE, now. Use this by default.
B  = $(shell date)     # recursive/deferred: re-expanded on EVERY use (B differs per reference!)
C ?= fallback          # set only if not already defined (env/CLI override wins)
D := base
D += more              # append (keeps the flavor of the original)
E := $(A:.c=.o)        # substitution reference: .c → .o
```

Rule of thumb: use `:=` (predictable, cheaper) unless you specifically need lazy evaluation (e.g. a variable that references another defined later). `make -p` dumps every variable and its flavor.

### B. Automatic variables (inside recipes)
```makefile
build/%.o: src/%.c
	$(CC) $(CFLAGS) -c $< -o $@      # $< first prereq, $@ the target
build/app: $(OBJECTS)
	$(CC) $^ -o $@                   # $^ all prereqs (dedup'd), $? only newer prereqs
```
`$@` target · `$<` first prerequisite · `$^` all prerequisites · `$?` prerequisites newer than target · `$*` the `%` stem. In a multi-line `for` loop, escape Make's `$` as `$$` to reach the shell's variable.

### C. Pattern rules & static patterns
One rule for a whole class of files — Make infers the stem `%`.

```makefile
OBJECTS := $(patsubst src/%.c,build/%.o,$(wildcard src/*.c))

build/%.o: src/%.c | build           # implicit pattern rule
	$(CC) $(CFLAGS) -c $< -o $@

# Static pattern: same idea, scoped to an explicit target list (clearer, safer)
$(OBJECTS): build/%.o: src/%.c | build
	$(CC) $(CFLAGS) -c $< -o $@
```

### D. Phony targets — the task-runner core
A phony target names an action, not a file. Without `.PHONY`, a file called `test` or `clean` in the repo silently shadows the target.

```makefile
.PHONY: test lint fmt clean ci

test: ## Run the test suite
	pytest

lint: ## Static analysis
	ruff check .

ci: lint test ## What CI runs (see ci-cd.md) — the stable pipeline entrypoint
```

CI calls `make ci`; the pipeline definition stays small and the commands live in one place (policy: [`ci-cd.md`](guides://ci-cd.md)).

### E. Order-only prerequisites & directories
A `|`-separated prereq must exist but its timestamp does **not** trigger a rebuild — the idiom for output directories.

```makefile
build/%.o: src/%.c | build      # build/ must exist; its mtime is ignored
	$(CC) -c $< -o $@
build:
	mkdir -p $@
```

### F. Automatic header dependencies (C/C++)
Don't hand-maintain header lists — let the compiler emit them. (Compiler flag policy: [`c.md`](guides://c.md) / [`cpp.md`](guides://cpp.md).)

```makefile
DEPFLAGS := -MMD -MP
build/%.o: src/%.c | build
	$(CC) $(CFLAGS) $(DEPFLAGS) -c $< -o $@
-include $(OBJECTS:.o=.d)        # leading '-' : don't fail before the .d files exist
```

### G. Functions & `define`
```makefile
SOURCES := $(sort $(wildcard src/*.c))            # sort → deterministic order
NAMES   := $(foreach f,$(SOURCES),$(notdir $(f)))
GIT_TAG := $(shell git describe --tags --always)

define compile           # multi-line canned recipe; invoke with $(call ...)
	@echo "  CC    $(notdir $2)"
	$(CC) $(CFLAGS) -c $1 -o $2
endef
build/%.o: src/%.c | build
	$(call compile,$<,$@)
```
Common functions: `$(wildcard)`, `$(patsubst)`/`$(subst)`, `$(filter)`/`$(filter-out)`, `$(sort)`, `$(foreach)`, `$(if)`, `$(shell)`, `$(notdir)`/`$(basename)`/`$(addprefix)`.

### H. Self-documenting `help` (the default goal)
```makefile
.DEFAULT_GOAL := help
.PHONY: help
help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-18s\033[0m %s\n",$$1,$$2}'
```
This `grep|awk` over `$(MAKEFILE_LIST)` is the canonical pattern; targets document themselves via the `## text` after the prerequisites.

### I. Safety header (put at the top of every Makefile)
```makefile
SHELL := bash                  # pin the shell; never trust the user's $SHELL
.SHELLFLAGS := -eu -o pipefail -c   # fail fast inside recipes (use /bin/sh + -eu if portability matters)
.DELETE_ON_ERROR:              # MAKE-ERR-01: remove a target if its recipe fails
MAKEFLAGS += --warn-undefined-variables --no-builtin-rules
```

### J. Common footguns → fixes
- **Spaces instead of a TAB** → `*** missing separator. Stop.` Recipes need a literal TAB. (`cat -A` shows `^I` for tab.)
- **Forgot `.PHONY`** → target stops running once a same-named file appears. Declare every task target.
- **`=` where you meant `:=`** → variable re-runs `$(shell …)` on every reference; performance and correctness surprises.
- **Each recipe line is a fresh shell** → `cd foo` then a separate line is back in the original dir. Join with `\` or `cd foo && …` on one logical line.
- **Unquoted variable in `rm -rf`** → empty/relative expansion is catastrophic; always `rm -rf "$(DIR)"` (see `secure-coding.md`).
- **Recursive `make -C` per directory** → loses the global dependency graph, breaks `-j`, hides errors with default `-k`. Prefer a single graph via `include` (Recursive Make Considered Harmful).
- **`-j` race conditions** → two recipes writing the same file, or a missing prereq edge. Fix the graph, don't add `.NOTPARALLEL` to paper over it.
- **Relying on built-in implicit rules** → surprising behavior; disable with `--no-builtin-rules` / `MAKEFLAGS`.

---

## 6. Parallelism

Make parallelizes independent branches of the graph automatically.

```bash
make -j"$(nproc)"          # one job per core
make -j -l 8               # unbounded jobs but back off above load 8
make -O                    # group each target's output (readable parallel logs)
```

`-j` correctness depends entirely on **accurate prerequisites** (MAKE-PAR-01). If a target reads a file another target writes, that edge MUST be in the graph. `.NOTPARALLEL:` (whole file) or `target: .NOTPARALLEL` is a last resort, not a fix. Concurrency *theory* is owned by [`parallelism.md`](guides://parallelism.md); here it is just "model true data dependencies as prerequisites."

---

## 7. Task runner vs build system — and when to graduate

| Use Make as… | When | Targets |
|---|---|---|
| **Task runner** | Any project needing memorable command aliases (`make test`, `make up`, `make deploy`) | all `.PHONY` |
| **Build system** | Small/medium C/C++ or codegen where files derive from files | file targets + pattern rules |

**Graduate to [`cmake.md`](guides://cmake.md)** when a C/C++ build needs: cross-platform generation, dependency/package discovery (`find_package`), multiple build configs, IDE project files, or installed/exported targets. Keep a thin Makefile as the human-facing wrapper (`make build` → `cmake --build build`). Pure non-compiled task running (lint/test/format/deploy aliases) never needs to graduate — Make is the right tool.

For large Makefiles, split by concern and keep the root file thin (MAKE-STRUCT-01):

```
project/
├── Makefile            # includes + .DEFAULT_GOAL only
└── make/
    ├── config.mk       # variables (:=)
    ├── build.mk        # pattern rules, file targets
    ├── test.mk         # phony task targets
    └── help.mk         # the help target
```

```makefile
# Makefile — thin orchestrator
include make/config.mk make/build.mk make/test.mk make/help.mk
.DEFAULT_GOAL := help
```

Use `include` (shared graph), not `$(MAKE) -C` (recursive, fragmented graph).

---

## 8. Quick Reference

```bash
make                       # run .DEFAULT_GOAL (help)
make test                  # run the test target
make -n build              # dry run: print recipes without executing
make -j"$(nproc)"          # parallel build
make -p | less             # dump database: all rules, vars, implicit rules
make -d target             # full trace of why target is (not) rebuilt
make --warn-undefined-variables -n   # lint: undefined vars + parse
make CC=clang build        # override a variable from the CLI
```

```makefile
# Idiom cheat sheet
VAR := value                       # immediate (default choice)
VAR ?= value                       # default unless overridden
$@ $< $^ $? $*                     # auto vars: target, 1st prereq, all, newer, stem
$(patsubst %.c,%.o,$(SRCS))        # transform a list
target: prereq | order-only        # '|' = exists-but-mtime-ignored
.PHONY: name                       # task target, not a file
.DELETE_ON_ERROR:                  # remove target on recipe failure
```

---

## 9. Deployment Checklist

Generated from §2 — one box per requirement ID.

- [ ] MAKE-SYN-01 — `make --warn-undefined-variables -n` clean, no warnings
- [ ] MAKE-TAB-01 — recipes TAB-indented (`cat -A` shows `^I`, no spaces)
- [ ] MAKE-PHONY-01 — every task target declared `.PHONY`
- [ ] MAKE-GOAL-01 — `.DEFAULT_GOAL` set to a safe target (help)
- [ ] MAKE-ERR-01 — `.DELETE_ON_ERROR:` present
- [ ] MAKE-DEP-01 — second build reports "Nothing to be done"
- [ ] MAKE-PAR-01 — `make -j$(nproc)` from clean is race-free
- [ ] MAKE-SEC-01 — shell variables quoted, `SHELL` pinned (see `secure-coding.md`)
- [ ] MAKE-DOC-01 — `make help` lists every public target (see `comments.md`)
- [ ] MAKE-STRUCT-01 — large Makefiles split into `make/*.mk`
- [ ] Agent ran every §3 command and documented any fixes

---
**End of GNU Make Guidelines**
