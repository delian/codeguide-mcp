# Fish Shell Scripting Guidelines
Mandatory standards for fish: deliberately non-POSIX, friendly syntax, autoloaded functions, scoped variables. fish 3.7+, fish_indent, fishtape.

---
name: fish
title: Fish Shell Scripting Guidelines
version: 2.0
last_reviewed: 2026-06-05
kind: language
tools: [fish@3.7, fish_indent, fishtape]
requires:
  - secure-coding
  - error-handling
recommends:
  - bash
  - zsh
  - comments
provides:
  - fish-syntax
  - fish-functions
  - fish-variables
  - fish-vs-posix
---

> 🧭 Authored per [`CONVENTIONS.md`](guides://CONVENTIONS.md): shared concerns are referenced, not restated. This guide covers only what is unique to fish.

---

## 0. Prerequisites & References

Fetch and apply these **before** generating fish code. Their rules are assumed below and not repeated.

> 📎 **REQUIRED — fetch & apply first:**
> - [`secure-coding.md`](guides://secure-coding.md) — input sanitization, supply chain, secrets. *(fish binding: vet Fisher plugins, never `curl | source` untrusted code, quote/`--`-terminate command args.)*
> - [`error-handling.md`](guides://error-handling.md) — error strategy & propagation. *(fish binding: integer `$status`, `; and` / `; or`, `return N` — see §7.)*

> 📎 **RECOMMENDED — fetch when the task touches them:**
> - [`bash.md`](guides://bash.md) — **the portability counterpart. fish is NOT POSIX; any script that must run on arbitrary machines or as `/bin/sh` belongs in bash, not fish (see §1, FISH-PORT-01).**
> - [`zsh.md`](guides://zsh.md) — the other interactive shell; contrast its bash-compatible posture with fish's clean break.
> - [`comments.md`](guides://comments.md) — doc policy *(binding: `--description` on every function, see §6).*

> 📎 **SEE ALSO:** [`tdd.md`](guides://tdd.md) (test-first; fish runner is `fishtape`) · [`hexagonal.md`](guides://hexagonal.md) (layering) · [`logging.md`](guides://logging.md)

---

## 1. Core Philosophies: FISH-FIRST

fish-specific principles only. Security, error-handling, and test-first policy come from §0 — do **not** restate them here.

🐟 **The prime fact: fish deliberately abandons POSIX for a clean, predictable language.** It is not bash with nicer defaults — it is a different language. Do not port bash idioms; write fish.

- **F**riendly syntax: assign with `set`, never `=`; substitute with `(cmd)`, never `$(cmd)` or backticks; block with `end`, never `fi`/`done`/`esac`.
- **I**mmutable scoping: declare the narrowest scope — `set -l` (local) by default, `-g` (global) only when shared, `-U` (universal) only for persisted user config.
- **S**ane expansion: no word-splitting and no glob-on-expansion of variables — lists are real, so quoting gymnastics are unnecessary (but still quote untrusted data per `secure-coding.md`).
- **H**armony with built-ins: prefer `string`, `math`, `argparse`, `path`, `count`, `contains` over `sed`/`awk`/`cut`/`expr`/external tools.
- **Interactive-first**: choose fish for interactive use, prompts, completions, and personal tooling — **not** for portable system scripts (those go to `bash.md`).

**Verified Code**: Agent-generated fish MUST pass every gate in §2 before delivery.

---

## 2. Requirements (MANDATORY, auditable)

RFC-2119 keywords. IDs `FISH-<TOPIC>-<NN>`. Each row has a binary gate; rows binding a shared rule cite its owner.

| ID | Requirement | Verify | Gate |
|----|-------------|--------|------|
| FISH-SYN-01 | Script MUST parse as valid fish | `fish -n script.fish` | exit 0 |
| FISH-FMT-01 | Code MUST be fish_indent-formatted | `fish_indent --check script.fish` | no diff |
| FISH-PORT-01 | fish MUST NOT be used where POSIX/portability is required; use bash instead (see `bash.md`) | review: no `#!/bin/sh`/`#!/bin/bash` shebang on a `.fish`; target is interactive/personal | justified |
| FISH-SYN-02 | MUST use fish syntax, never bash/zsh constructs (`$(...)`, `` ` ``, `[[ ]]`, `=`-assignment, `()` funcs) | `! grep -nE '\$\(|`|\[\[|\bfi\b|\bdone\b' script.fish` | no matches |
| FISH-VAR-01 | Variables MUST use the narrowest scope (`set -l` default) | review / `grep 'set -g'` justified | scoped |
| FISH-ERR-01 | Failures MUST propagate via `$status` / `; and` / `; or` / `return` (see `error-handling.md`) | review | no silent failure |
| FISH-TST-01 | Logic MUST be test-first with fishtape (see `tdd.md`) | `fishtape tests/*.fish` | exit 0 |
| FISH-TST-02 | Each bug MUST get a regression test before the fix (see `tdd.md`) | `fishtape tests/*.fish` | failing→passing |
| FISH-DOC-01 | Public functions MUST carry `--description` (see `comments.md`) | `functions -D name` / review | present |
| FISH-SEC-01 | No `eval`/`curl\|source` on untrusted input; inputs sanitized (see `secure-coding.md`) | `! grep -nE 'eval|curl.*\| *source' script.fish` | none / justified |

> **Forbidden**: shipping implementation before its test (violates `tdd.md`), fixing a bug without a regression test first, writing bash-in-fish (`$(...)`, `[[ ]]`, `=`-assignment), or reaching for `sed`/`awk` where `string` suffices.

---

## 3. Verification Protocol

Run, in order, before presenting code. Fix → re-run until every gate is green.

```fish
fish -n script.fish                 # FISH-SYN-01  parse check
fish_indent --check script.fish     # FISH-FMT-01  (auto-fix: fish_indent -w script.fish)
fishtape tests/*.fish               # FISH-TST-01/02
```

`fish --no-execute` is the same as `fish -n`. Test against fish **3.7+** (`status fish-path`, `fish --version`); features like `path`, `string pad`, and `set --function` assume modern fish. The *why* behind test-first/security gates lives in their §0 owners.

---

## 4. Project Structure & Autoloading

fish's defining structural feature is **autoloading**: a function named `foo` is loaded on first use from any `foo.fish` on `$fish_function_path` (default includes `~/.config/fish/functions/`). One function per file, file named exactly after the function. `conf.d/*.fish` runs at startup (alphabetical); `completions/foo.fish` is autoloaded when `foo` is completed. Architectural layering (ports/adapters) is owned by [`hexagonal.md`](guides://hexagonal.md); below is only its fish mapping.

```
~/.config/fish/
├── config.fish            # interactive startup; guard machine code with `status is-interactive`
├── conf.d/                # autoloaded at startup, alphabetical (e.g. 00-env.fish, 50-aliases.fish)
├── functions/             # one autoloaded function per file: name.fish
└── completions/           # one completion spec per command: cmd.fish

# A standalone tool/plugin mirrors the same tree so Fisher can install it.
```

- A standalone script gets `#!/usr/bin/env fish`; reusable logic belongs in `functions/` (autoloaded, independently testable), not in monolithic scripts.
- Resolve a script's own dir with `set -l dir (path dirname (status filename))` (fish 3.7) or `(dirname (status filename))`.

---

## 5. Fish Specifics — the unique surface

### A. Syntax that breaks from POSIX (the load-bearing differences)
```fish
set name "Ada"                 # assignment — NOT name="Ada"
set files (ls *.txt)           # command substitution — NOT $(...) and NOT backticks
echo $name                     # expansion; $ only when reading
echo "count: "(count $files)   # () inline; no word-splitting on $files

if test -f $f; and test -r $f  # test / [ ; there is no [[ ]]
    echo ok
end                            # every block ends with `end` (not fi/done/esac)

switch $animal
    case cat dog; echo pet
    case '*';    echo other
end
```
Lists are first-class and **1-indexed**: `$files[1]`, `$files[-1]`, slices `$files[2..-1]`, `count $files`, `contains x $files`. There is no `${#arr[@]}` and no `${arr[0]}`.

### B. Variable scopes (a fish-defining concept)
```fish
set -l x val      # local: this block/function only (DEFAULT — prefer this)
set -g x val      # global: this shell session
set -U x val      # universal: persisted across sessions & shells (use sparingly — survives restarts)
set -x  PATH …    # exported to child processes (combine: set -gx, set -lx)
set -a list more  # append;  set -p prepend
set -e x          # erase;   set -q x  → status 0 if set (query)
```
`-U` writes to disk immediately and is shared by every running fish — ideal for `$EDITOR`, dangerous for transient state. `$PATH`, `$fish_function_path` etc. are lists, not colon-strings.

### C. Functions, `argparse`, and `$argv`
```fish
function greet --description 'Greet by name' --argument-names who
    set -q who[1]; or set who world
    echo "Hello, $who"
end
```
Arguments arrive in `$argv` (1-indexed). Parse options with built-in **`argparse`** — not manual loops:
```fish
function deploy --description 'Deploy a build'
    argparse h/help 'e/env=' v/verbose -- $argv; or return        # `or return` on parse failure
    set -q _flag_help; and begin; printf 'usage: deploy -e ENV [-v]\n'; return 0; end
    set -q _flag_env;  or begin; echo 'deploy: --env required' >&2; return 2; end
    test -n "$_flag_verbose"; and echo "env=$_flag_env argv=$argv"
end
```
Each `x/long` flag sets `$_flag_x`; `=` means it takes a value. Process text with `string` (`string match -rq`, `string split`, `string replace`, `string trim`, `string pad`) and arithmetic with `math`, not external tools.

### D. Abbreviations vs aliases vs functions
- **`abbr`** — expands inline *before* execution; the real command lands in history and is editable. **Preferred for interactive shortcuts.** `abbr -a gco git checkout`; fish 3.7 supports function-driven `--function` abbreviations.
- **`alias`** — sugar that *defines a wrapper function*; use only for genuine command wrapping (e.g. `alias ls 'ls --color=auto'`), and prefer a real autoloaded function for anything non-trivial.
- A named function in `functions/` is the right home for multi-line logic — testable and autoloaded.

### E. Event handlers & completions (fish-native extensibility)
```fish
function _on_pwd --on-variable PWD          # also: --on-event, --on-signal INT, --on-job-exit
    test -f .nvmrc; and nvm use 2>/dev/null
end

complete -c deploy -s e -l env -d 'Target env' -xa 'staging prod'   # completions/deploy.fish
```
Handlers fire on variable changes, named `emit`ted events, signals, and process/job exit — fish's substitute for trap-heavy bash patterns.

### F. config.fish & startup
Keep `config.fish` fast and idempotent. Guard interactive-only setup with `status is-interactive`; put environment/`$PATH` edits in `conf.d/` so they apply to non-interactive invocations too. Use `fish_add_path` instead of hand-editing `$PATH`.

### G. Common footguns → fix
- `var=value` → `set var value`. `$(cmd)` / `` `cmd` `` → `(cmd)`.
- `$arr[0]` → `$arr[1]` (1-indexed). `${#arr[@]}` → `count $arr`.
- `[[ … ]]` / `&&` / `||` in conditionals → `test`/`[`, and `; and`/`; or`.
- `set -e` (bash "exit on error") → **not a thing**; in fish `set -e` *erases a variable*. Propagate errors with `; or return` (see `error-handling.md`).
- Splitting a captured value on spaces unexpectedly → it won't; fish doesn't word-split. Use `string split` when you *want* splitting.
- Reading a missing `$argv[1]` → guard with `set -q argv[1]`.

---

## 6. Functions Documentation & Testing

Documentation policy is owned by [`comments.md`](guides://comments.md); fish binding: every public function carries `--description` (surfaced by `functions -D`, completion, and `help`), plus a short header comment for arguments/returns when non-obvious. Test-first policy is owned by [`tdd.md`](guides://tdd.md); fish runner is **fishtape** (TAP output):
```fish
# tests/test_greet.fish
source (path dirname (status filename))/../functions/greet.fish
@test "greet defaults to world" (greet) = "Hello, world"
@test "greet uses argument"     (greet Ada) = "Hello, Ada"
```
Run `fishtape tests/*.fish` (exit 0 = pass). `littlecheck` is the alternative for output-matching CLI tests. Write the test first, watch it fail, then implement (FISH-TST-01/02).

---

## 7. Error Handling & Tooling

Strategy is owned by [`error-handling.md`](guides://error-handling.md); fish binding: every command sets integer `$status` (0 = success). Bind control flow to it — capture immediately (`set -l rc $status`) since the next command overwrites it:
```fish
risky; or begin; echo 'failed' >&2; return 1; end     # handle failure
validate $f; or return $status                        # propagate
set -l out (cmd); or set out fallback                 # default on failure
```
**Dependencies** use Fisher; commit `~/.config/fish/fish_plugins`. Supply-chain *policy* (vet sources, pin, never `curl | source` untrusted code) is owned by [`secure-coding.md`](guides://secure-coding.md):
```fish
fisher install jorgebucaran/nvm.fish    # add (review the code first)
fisher update                           # update pinned set
fisher list                             # audit installed plugins
```

---

## 8. Quick Reference

```fish
fish -n script.fish                 # parse check          (FISH-SYN-01)
fish_indent -w script.fish          # format               (FISH-FMT-01)
fishtape tests/*.fish               # test                 (FISH-TST-01/02)
fish script.fish                    # run
set -l x v   /  set -gx PATH …      # scope / export
argparse h/help 'e/env=' -- $argv   # parse options
string match -rq PATTERN -- $s      # match;  math 1 + 2   # arithmetic
```

---

## 9. Deployment Checklist

Generated from §2 — one box per requirement ID.

- [ ] FISH-SYN-01 — `fish -n` parses clean
- [ ] FISH-SYN-02 — no bash/zsh constructs (`$(...)`, `[[ ]]`, `=`-assignment, `()` funcs)
- [ ] FISH-FMT-01 — `fish_indent --check` no diff
- [ ] FISH-PORT-01 — fish is the right tool (interactive/personal); portable scripts went to bash
- [ ] FISH-VAR-01 — narrowest scope used (`set -l` default)
- [ ] FISH-ERR-01 — failures propagate via `$status`/`and`/`or`/`return`
- [ ] FISH-TST-01/02 — fishtape green, bugs have regression tests
- [ ] FISH-DOC-01 — public functions carry `--description`
- [ ] FISH-SEC-01 — no `eval`/`curl|source` on untrusted input; inputs sanitized
- [ ] Agent ran every §3 command and documented any fixes

---
**End of Fish Shell Scripting Guidelines**
