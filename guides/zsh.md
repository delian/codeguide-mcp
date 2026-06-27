# Zsh Shell Scripting Guidelines
Mandatory standards for zsh scripts and interactive config: when to choose zsh, the zsh/bash differences that bite, glob qualifiers, parameter-expansion flags, and the completion system. Zsh 5.9, zsh -n, shellcheck (portable subset only), shfmt.

---
name: zsh
title: Zsh Shell Scripting Guidelines
version: 2.0
last_reviewed: 2026-06-05
kind: language
tools: [zsh@5.9, shellcheck, shfmt, bats]
requires:
  - secure-coding
  - error-handling
recommends:
  - bash
  - fish
  - comments
provides:
  - zsh-vs-bash
  - glob-qualifiers
  - zsh-expansion
  - completion-system
---

> 🧭 Authored per [`CONVENTIONS.md`](guides://CONVENTIONS.md): shared concerns are referenced, not restated. This guide covers only what is unique to **zsh** — the shared shell fundamentals live in [`bash.md`](guides://bash.md).

---

## 0. Prerequisites & References

Fetch and apply these **before** writing zsh code. Their rules are assumed below and not repeated.

> 📎 **REQUIRED — fetch & apply first:**
> - [`secure-coding.md`](guides://secure-coding.md) — input sanitization, secrets, supply chain. *(zsh binding: never put secrets in `.zshrc`/plugin files; vet plugin sources.)*
> - [`error-handling.md`](guides://error-handling.md) — error strategy & propagation. *(zsh binding: `setopt ERR_EXIT NO_UNSET PIPE_FAIL`, `TRAPZERR`, `always` blocks — see §7.)*

> 📎 **RECOMMENDED — fetch when the task touches them:**
> - [`bash.md`](guides://bash.md) — **the shared shell fundamentals**: hexagonal structure for scripts, strict mode, quoting, `getopt`/`getopts`, traps/`mktemp`, bats testing, shellcheck/shfmt. This guide does **not** restate them; it documents only where zsh *differs*.
> - [`comments.md`](guides://comments.md) — function/usage documentation policy.
> - [`fish.md`](guides://fish.md) — the other "human-first" shell; reference when comparing interactive shells.

> 📎 **SEE ALSO:** [`tdd.md`](guides://tdd.md) · [`git.md`](guides://git.md) *(dotfiles in version control)*

---

## 1. Core Philosophies

Zsh-specific principles only. Quoting, strict-mode rationale, modular/hexagonal layout, and testing come from §0 — do **not** restate them here.

- **Pick the shell on purpose.** A POSIX-portable automation script that must run in CI, containers, or other people's machines belongs in **bash** (`#!/usr/bin/env bash`, see [`bash.md`](guides://bash.md)). Reach for **zsh** when the script is genuinely zsh-targeted (your dotfiles toolchain, a macOS-default-shell tool, completion/prompt code) **or** when zsh's globbing/expansion eliminates an external `find`/`sed`/`awk` pipeline.
- **No accidental dialects.** Declare the target. A zsh script uses `#!/usr/bin/env zsh` and may freely use zsh features; do not write a `#!/bin/sh`/bash script that silently relies on zsh behavior, and do not claim portability you have not tested.
- **Lean on what makes zsh zsh.** Glob qualifiers, expansion flags, and `zmodload`/`zsh/*` modules exist to replace fragile external-command pipelines — use them in zsh scripts instead of shelling out.
- **`emulate -L zsh` in functions.** Library functions that must behave predictably regardless of the user's `setopt` state start with `emulate -L zsh` (locally resets options, auto-restored on return).
- **Config is code.** `.zshrc`/`.zshenv` are programs that run on every shell start: keep them fast, idempotent, secret-free, and in version control (see `git.md`).

**Verified Code**: Agent-generated zsh MUST pass every gate in §2 before delivery.

---

## 2. Requirements (MANDATORY, auditable)

RFC-2119 keywords. IDs `ZSH-<TOPIC>-<NN>`. Each row has a binary gate; rows binding a shared rule cite its owner.

| ID | Requirement | Verify | Gate |
|----|-------------|--------|------|
| ZSH-SYN-01 | Scripts MUST parse clean in zsh | `zsh -n script.zsh` | exit 0 |
| ZSH-SYN-02 | Strict-mode parse MUST pass | `zsh -o ERR_EXIT -o NO_UNSET -o PIPE_FAIL -n script.zsh` | exit 0 |
| ZSH-SHEBANG-01 | A zsh script MUST declare `#!/usr/bin/env zsh`; portable scripts MUST NOT (use `bash`) | `head -1` review | correct interpreter |
| ZSH-LINT-01 | The **bash-portable** subset MUST pass shellcheck; pure-zsh scripts are exempt (shellcheck has no zsh dialect) | `shellcheck -s bash file` *(portable only)* | exit 0 |
| ZSH-FMT-01 | Portable scripts MUST be `shfmt`-clean (zsh-only syntax may be excluded) | `shfmt -d file` | no diff |
| ZSH-ERR-01 | Scripts MUST enable strict mode & handle errors (see `error-handling.md`) | grep `setopt ERR_EXIT NO_UNSET PIPE_FAIL` / `TRAPZERR` | present |
| ZSH-TST-01 | Logic MUST have tests; bug fixes get a regression test first (see `tdd.md`, harness in `bash.md`) | `zsh -f tests/*.ztst` / `bats tests/` | exit 0 |
| ZSH-SEC-01 | No secrets in source/config; user input sanitized (see `secure-coding.md`) | review / grep | 0 secrets |
| ZSH-SEC-02 | Interactive plugins MUST come from vetted sources, pinned (see `secure-coding.md`) | review plugin manifest | all vetted |
| ZSH-RC-01 | `.zshrc`/`.zshenv` MUST be fast, idempotent, secret-free | `time zsh -ic exit` | < ~200 ms, no secrets |

> **Forbidden**: shipping a script whose shebang lies about its dialect; relying on shellcheck to validate pure-zsh syntax; fixing a bug without a regression test first (violates `tdd.md`); hardcoding secrets in `.zshrc`.

---

## 3. Zsh vs Bash — the differences that bite

The single highest-value section: behaviors that silently change meaning when a bash habit meets zsh.

| Behavior | bash | zsh | Consequence |
|---|---|---|---|
| Array indexing | 0-based | **1-based** | `${a[1]}` is the *first* element in zsh, second in bash |
| Unquoted `$var` with spaces | word-splits | **no split** | zsh does not word-split parameter expansions by default |
| Unquoted glob, no match | left literal | **error** (or empty with `NULL_GLOB`) | `rm *.log` aborts if nothing matches |
| `$var` splitting in `for` | splits on IFS | iterates as **one word** | port bash loops carefully |
| `${a[@]}` vs `$a` | `$a` = first elem | **`$a` = whole array joined** | different scalar semantics |
| `read a b c` | splits line | needs `read -A`/explicit | array reads differ |

```zsh
# Get 0-based, bash-like arrays for a block of ported code:
setopt KSH_ARRAYS              # 0-based indexing + ${a[@]} semantics like bash
# Get bash-style word splitting where a script truly needs it:
setopt SH_WORD_SPLIT          # opt-in only; prefer explicit arrays instead
```

Prefer **adapting to zsh semantics** (explicit arrays, 1-based logic) over forcing bash emulation. If a script must run in both shells unmodified, write it for **bash** and keep it in a `.sh` file per [`bash.md`](guides://bash.md) — do not author a "dual-dialect" hybrid.

---

## 4. Glob qualifiers & extended globbing

Zsh's defining feature: filter and sort matches **in the glob itself**, replacing most `find`/`ls | grep` pipelines. Enable extended patterns once:

```zsh
setopt EXTENDED_GLOB NULL_GLOB        # rich patterns; empty glob → empty list, not error
```

**Glob qualifiers** — the `(...)` suffix selects by type/permission/time/size:

```zsh
print -l *(.)            # regular files only
print -l *(/)            # directories only
print -l *(@)            # symlinks;  *(*) executables;  *(.x) exec-by-owner
print -l *(.om[1])       # newest regular file (o=order, m=mtime, [1]=first)
print -l *(.L0)          # zero-length files;  *(Lm+10) larger than 10 MB
print -l **/*.log(.mh-24)  # *.log modified in the last 24 h, recursive (**)
print -l *(.N)            # N = NULL_GLOB for this glob only
files=( **/*.txt~*/node_modules/*(.) )   # recursive, regular, excluding a subtree
```

**Extended-glob operators** (need `EXTENDED_GLOB`):

```zsh
print -l ^*.txt          # everything NOT matching *.txt
print -l *.txt~*test*    # *.txt except names containing "test"
print -l (foo|bar)*.c    # alternation
print -l (#i)*.PNG       # case-insensitive flag
print -l file<1-100>.dat # numeric range
```

> These qualifiers are **zsh-only**. A portable script must use `find` instead (see [`bash.md`](guides://bash.md)) — that is a reason to keep portable scripts in bash rather than half-porting qualifiers.

---

## 5. Parameter expansion flags

Zsh puts transformations *inside* `${...}` via `(flag)` prefixes — no `tr`/`sed`/`awk` subprocess:

```zsh
${(U)var}  ${(L)var}      # upper / lower case        (bash: ${var^^} ${var,,})
${(C)var}                  # Capitalize Words
${var:u}  ${var:l}         # modifier form of up/low
${(j:,:)array}             # join array with ","        (bash: IFS subshell trick)
${(s:,:)str}               # split "str" on ","          → array
${(f)text}                 # split on newlines
${(o)array}  ${(O)array}   # sort ascending / descending
${(u)array}                # unique
${(@kv)assoc}              # key,value pairs of an assoc array
${(P)name}                 # indirect: value of the var *named* by $name (bash: ${!name})
${(q)var}  ${(Q)var}       # shell-quote / unquote
${#array}  ${(w)#str}      # element count / word count
${(@)array}                # preserve empty elements (like "${a[@]}")
```

Filename modifiers chain on any expansion (and on `$0`):

```zsh
${0:A}      # absolute, symlink-resolved path of the script
${0:A:h}    # its directory   (the zsh-native SCRIPT_DIR)
${0:t}      # tail/basename     ${file:r} root (no ext)   ${file:e} extension
```

Combine flags: `${(j:, :)$(<file)}`, `${(f)"$(< list.txt)"}`, `${(Oa)array}` (reverse). These compose where bash needs a pipeline — the core reason to write the script in zsh at all.

---

## 6. Completion system (compsys)

Programmable completion is canonical zsh territory. Initialize once, then ship `_command` functions on `$fpath`.

```zsh
# In .zshrc — enable the modern completion system
autoload -Uz compinit && compinit          # add -C to skip the slow security audit on trusted machines
zstyle ':completion:*' menu select          # arrow-key menu
zstyle ':completion:*' matcher-list 'm:{a-z}={A-Za-z}'   # case-insensitive
zstyle ':completion:*' use-cache on
```

A completion function lives in a file named `_mytool` on `$fpath`, first line `#compdef mytool`:

```zsh
#compdef mytool
_mytool() {
  _arguments -s \
    '(-h --help)'{-h,--help}'[show help]' \
    '-v[verbose]' \
    '--output=[output file]:file:_files' \
    '1:command:(build test deploy)' \
    '*::args:->args'
}
_mytool "$@"
```

Key building blocks: `_arguments` (option spec + state machine), `_describe`/`_values` (tagged candidate lists), `_files`/`_path_files` (file completion), `compadd` (low-level). Put custom completions in a versioned `~/.zsh/completions` dir added to `fpath` **before** `compinit`. Regenerate the dump after changes: `rm -f ~/.zcompdump; compinit`.

---

## 7. Scripting idioms & error handling

Strict-mode and the error *strategy* are owned by [`error-handling.md`](guides://error-handling.md); the zsh bindings:

```zsh
#!/usr/bin/env zsh
emulate -L zsh                         # predictable options inside functions/libs
setopt ERR_EXIT NO_UNSET PIPE_FAIL     # ≈ bash `set -euo pipefail`
setopt EXTENDED_GLOB

readonly SCRIPT_DIR="${0:A:h}"         # zsh-native (no BASH_SOURCE dance)

# ZERR trap: runs on any non-zero status (zsh-only)
TRAPZERR() { print -u2 "error at ${funcfiletrace[1]}"; }

# try/finally: the `always` block runs even on error (zsh-only)
{
  risky_step_1
  risky_step_2
} always {
  cleanup            # guaranteed
}
```

- **Argument parsing.** For zsh-only tools, `zparseopts` is idiomatic and the right call: `zparseopts -D -E -F - h=help v=verbose o:=output`. For anything that must also run in bash, use `getopt`/`getopts` per [`bash.md`](guides://bash.md). Choose by the script's declared dialect — not by reflex.
- **Read a file into an array of lines:** `lines=( ${(f)"$(<file)"} )` — no loop.
- **Associative arrays:** `typeset -A m=(host localhost port 8080)`; iterate `for k v in ${(@kv)m}; do …; done` (zsh-native key/value loop).
- **Numeric/test context:** `(( count > 0 ))` and `[[ … ]]` work as in bash; zsh adds `[[ -o option ]]` to test `setopt` state.
- **Modules over subprocesses:** `zmodload zsh/datetime` (`$EPOCHREALTIME`), `zsh/mathfunc`, `zsh/stat` replace `date`/`bc`/`stat` forks.

---

## 8. Interactive config (.zshrc) hygiene

`.zshenv` → `.zprofile` → `.zshrc` → `.zlogin` load in that order; put PATH/env in `.zshenv` (or `.zprofile` for login-only), interactive setup (prompt, keybindings, completion, aliases) in `.zshrc`. Keep startup lean and secret-free.

```zsh
# Profile startup cost; keep it well under ~200 ms (ZSH-RC-01)
time zsh -ic exit
zmodload zsh/zprof    # add `zprof` at the end of .zshrc to see per-function cost
```

- **Plugins:** prefer a thin manager (`zinit`, `antidote`, or git submodules) over a heavyweight framework if startup matters; pin commits and vet sources (ZSH-SEC-02, see [`secure-coding.md`](guides://secure-coding.md)). The two essentials are `zsh-autosuggestions` and `zsh-syntax-highlighting` (load highlighting **last**).
- **Prompt:** use the native `prompt` system (`autoload -Uz promptinit; promptinit`) or a prompt tool (e.g. Starship); enable substitution with `setopt PROMPT_SUBST` and keep command substitutions in `PROMPT` cheap (they run every redraw).
- **History:** `setopt HIST_IGNORE_ALL_DUPS SHARE_HISTORY HIST_REDUCE_BLANKS`; set `HISTFILE`/`HISTSIZE`/`SAVEHIST`; add `HIST_IGNORE_SPACE` so space-prefixed commands (secrets) are never written.
- **Never** echo tokens, run `eval "$(curl …)"`, or store credentials in any startup file — source them from a secret manager.
- **Version-control** dotfiles (see [`git.md`](guides://git.md)); make every block idempotent so re-sourcing is safe.

---

## 9. Deployment Checklist

Generated from §2 — one box per requirement ID.

- [ ] ZSH-SYN-01/02 — `zsh -n` clean, strict-mode parse clean
- [ ] ZSH-SHEBANG-01 — shebang matches the actual dialect
- [ ] ZSH-LINT-01 — portable subset passes `shellcheck -s bash` (pure-zsh exempt)
- [ ] ZSH-FMT-01 — portable code `shfmt`-clean
- [ ] ZSH-ERR-01 — strict mode + error handling present
- [ ] ZSH-TST-01 — tests pass; bugs have regression tests first
- [ ] ZSH-SEC-01/02 — no secrets; plugins vetted & pinned
- [ ] ZSH-RC-01 — startup fast, idempotent, secret-free
- [ ] Agent ran every §2 verify command and documented any fixes

---
**End of Zsh Shell Scripting Guidelines**
