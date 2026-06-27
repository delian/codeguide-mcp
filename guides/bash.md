# Bash Shell Scripting Guidelines
Mandatory coding standards for Bash: strict-mode, quoted, shellcheck-clean, portable, testable scripts. Bash 5.2+, shellcheck, shfmt, bats.

---
name: bash
title: Bash Shell Scripting Guidelines
version: 2.0
last_reviewed: 2026-06-05
kind: language
tools: [bash@5.2, shellcheck, shfmt, bats]
requires:
  - secure-coding
  - error-handling
recommends:
  - zsh
  - fish
  - comments
  - ci-cd
provides:
  - bash-strict-mode
  - quoting
  - shellcheck
  - traps
  - bash-arrays
---

> 🧭 Authored per [`CONVENTIONS.md`](guides://CONVENTIONS.md): shared concerns are referenced, not restated. This guide covers only what is unique to Bash.

---

## 0. Prerequisites & References

Fetch and apply these **before** generating Bash code. Their rules are assumed below and not repeated.

> 📎 **REQUIRED — fetch & apply first:**
> - [`secure-coding.md`](guides://secure-coding.md) — injection, secrets, supply chain. *(Bash binding: quoting prevents word-splitting/glob injection; never `curl | bash`; `shellcheck` is the static scanner.)*
> - [`error-handling.md`](guides://error-handling.md) — error strategy & propagation. *(Bash binding: `set -euo pipefail` + `trap` for fail-fast and cleanup.)*

> 📎 **RECOMMENDED — fetch when the task touches them:**
> - [`zsh.md`](guides://zsh.md) · [`fish.md`](guides://fish.md) — sibling shells; consult when a script must also run under, or interoperate with, them.
> - [`comments.md`](guides://comments.md) — function/usage-doc policy *(binding: header block + `# Arguments/Returns/Outputs` per function)*.
> - [`ci-cd.md`](guides://ci-cd.md) — running scripts as pipeline steps; gate `shellcheck`/`shfmt`/`bats` in CI.

> 📎 **SEE ALSO:** [`tdd.md`](guides://tdd.md) *(test-first via `bats`)* · [`make.md`](guides://make.md) *(orchestrating script targets)* · [`python.md`](guides://python.md) *(reach for it when a script outgrows Bash — see §7)*.

---

## 1. Core Philosophies: BASH-FIRST

Bash-specific principles only. Security, error handling, and testing policy come from §0.

- **B**ootstrapped safety: every script opens with a strict-mode preamble (`set -euo pipefail`) before any logic runs.
- **A**lways quoted: every parameter expansion and command substitution is double-quoted unless word-splitting is explicitly wanted. Unquoted `$var` is the #1 Bash bug.
- **S**hellcheck-clean: `shellcheck` is mandatory and non-negotiable — a script that does not pass is not delivered.
- **H**onest scope: Bash is glue for processes and files. Past ~100 lines, branching data structures, or arithmetic-heavy logic, reach for a real language (§7).
- **F**ail loud, clean up: check exit codes, emit actionable `stderr` messages, and release temp files/locks via `trap`.

**Verified Code**: Agent-generated Bash MUST pass every gate in §2 before delivery.

---

## 2. Requirements (MANDATORY, auditable)

RFC-2119 keywords. IDs `BASH-<TOPIC>-<NN>`. Each row has a binary gate; rows binding a shared rule cite its owner.

| ID | Requirement | Verify | Gate |
|----|-------------|--------|------|
| BASH-SAFE-01 | Every script MUST start with `set -euo pipefail` (see `error-handling.md`) | `head -5 script.sh \| grep -q 'set -euo pipefail'` | present |
| BASH-SYN-01 | Script MUST parse without errors | `bash -n script.sh` | exit 0 |
| BASH-LINT-01 | `shellcheck` MUST pass clean — no warnings | `shellcheck -x -S style script.sh` | exit 0 |
| BASH-FMT-01 | Code MUST be `shfmt`-formatted | `shfmt -d -i 2 -ci script.sh` | no diff |
| BASH-QUOTE-01 | All expansions MUST be quoted (no SC2086/SC2046) | `shellcheck script.sh` | 0 SC2086/SC2046 |
| BASH-PORT-01 | Shebang MUST match the dialect used; bashisms only under `#!/usr/bin/env bash` | `shellcheck -s bash script.sh` | exit 0 |
| BASH-SEC-01 | No `eval`/`curl\|bash` on untrusted input; no hardcoded secrets (see `secure-coding.md`) | `shellcheck` + review / `grep -nE 'eval\|curl.*\| *(ba)?sh'` | none |
| BASH-ERR-01 | Temp files/locks MUST be released via `trap ... EXIT` (see `error-handling.md`) | review / `grep -q 'trap ' script.sh` | cleanup present |
| BASH-TST-01 | Non-trivial logic MUST have `bats` tests (see `tdd.md`) | `bats tests/` | exit 0, 0 skips |
| BASH-DOC-01 | Script + each public function MUST carry a header/usage block (see `comments.md`) | review / `--help` runs | documented |

> **Forbidden**: shipping a script that fails any gate above; unquoted expansions; parsing `ls` output; `eval` on untrusted input; `curl … | bash`; secrets in source; using `set -e` as a substitute for checking critical exit codes (it is silenced inside `if`/`&&`/`||` and command substitution).

---

## 3. Verification Protocol

Run, in order, before presenting code. Fix → re-run until every gate is green.

```bash
shfmt -d -i 2 -ci script.sh         # BASH-FMT-01
bash -n script.sh                   # BASH-SYN-01
shellcheck -x script.sh             # BASH-LINT-01 / QUOTE-01 / PORT-01
bats tests/                         # BASH-TST-01 (if tests exist)
./script.sh --help                  # BASH-DOC-01 smoke test
```

The *why* behind each gate lives in its §0 owner; do not re-derive it here.

---

## 4. Project Structure

Idiomatic library/CLI layout. Keep the entrypoint thin; put reusable functions in sourced libraries so they are unit-testable with `bats`.

```
project/
├── bin/<tool>              # executable entrypoint: arg-parse + main(), thin
├── lib/                    # sourced function libraries (no top-level side effects)
│   └── *.sh
├── tests/                  # bats specs mirroring lib/ (see tdd.md)
│   └── *.bats
└── README.md
```

- A sourced library MUST NOT run code at load time (guard `main` with `[[ "${BASH_SOURCE[0]}" == "$0" ]] && main "$@"`), so tests can `source` it cleanly.
- Group functions by responsibility; one small, single-purpose function per task.

---

## 5. Bash Specifics

The unique value of this guide.

### A. Strict-mode preamble
Every script opens with the same safe header. This is the Bash binding of `error-handling.md` (fail-fast) and `secure-coding.md` (predictable splitting).

```bash
#!/usr/bin/env bash
set -euo pipefail              # -e: exit on error; -u: error on unset var; pipefail: catch failures mid-pipe
IFS=$'\n\t'                    # split only on newline/tab, never spaces

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
```

**`set -e` is not a safety net** — it is suppressed inside `if`, `while`, `&&`, `||`, `!`, and `$(...)`. Check exit codes explicitly where correctness depends on them:

```bash
if ! cp "$src" "$dst"; then
  printf 'copy failed: %s -> %s\n' "$src" "$dst" >&2
  exit 1
fi
```

### B. Quoting — the #1 footgun
Always double-quote expansions; word-splitting and globbing happen *after* expansion, on unquoted values only.

```bash
process "$file"               # ✅ one argument, spaces preserved
process $file                 # ❌ splits + globs — SC2086

for f in "$@"; do ... done    # ✅ each positional intact
rm -- "${files[@]}"           # ✅ array expansion, one elem per word
cp -- "$src" "$dst"           # ✅ `--` stops a leading-dash filename being read as a flag
```

`"$@"` (each arg quoted) vs `"$*"` (all args joined by IFS) — use `"$@"` to forward arguments. Never iterate over `$(ls)`; glob instead (`for f in ./*.txt`) or `find … -print0 | while IFS= read -r -d '' f`.

### C. Tests, conditionals & arithmetic
Prefer `[[ ]]` over `[ ]` in Bash: no word-splitting inside, supports `=~`, `&&`, `<`, pattern matching.

```bash
[[ -f "$file" && -r "$file" ]]        # no quoting hazards, unlike POSIX [ ]
[[ "$name" == user_* ]]               # glob match
[[ "$line" =~ ^([0-9]+):(.*)$ ]] && id="${BASH_REMATCH[1]}"   # regex + captures
(( count > 0 )) && echo "non-empty"   # arithmetic context — bare names, no $
```

Use `[ ]`/`test` only when the script targets POSIX `sh` (§F).

### D. Arrays & parameter expansion
Bash has real indexed and associative arrays. Use them instead of space-delimited strings.

```bash
declare -a files=(src/*.sh)           # indexed; glob expands to elements
declare -A seen=([a]=1 [b]=2)         # associative (Bash 4+)
echo "${#files[@]}"                   # length
printf '%s\n' "${files[@]}"           # iterate safely

cmd=(rsync -a --delete)               # build argv as an array — never a string
cmd+=(--exclude '*.tmp')
"${cmd[@]}" "$src" "$dst"             # safe expansion, no eval
```

Parameter expansion replaces external tools (`basename`, `sed`, `cut`) and subshells:

```bash
"${var:-default}"     # default if unset/empty       "${var:?msg}"   # error if unset
"${path##*/}"         # basename                      "${path%/*}"    # dirname
"${name%.tar.gz}"     # strip suffix                  "${s//foo/bar}" # global replace
"${var,,}" / "${var^^}"   # lower / upper (Bash 4+)   "${#var}"       # length
```

### E. Functions, traps & exit codes
Localize state; document the contract (policy: `comments.md`); clean up with `trap`.

```bash
# Download a URL to a path.
# Arguments: $1 url, $2 dest
# Returns:   0 ok, non-zero on failure
download() {
  local url="$1" dest="$2"
  curl -fsSL --retry 3 -o "$dest" -- "$url"   # -f: fail on HTTP error
}

# Single EXIT trap owns all cleanup — fires on success, error, and signal.
work() {
  local tmp; tmp="$(mktemp)"
  trap 'rm -f -- "$tmp"' RETURN          # function-scoped cleanup
  ...
}
trap 'rm -rf -- "$workdir"' EXIT          # script-scoped cleanup
```

Exit codes are the API: `0` success, `1`–`125` app errors, `2` for usage, `>128` = `128+signal`. `return`/`exit` with a meaningful code; write diagnostics to `stderr` via `>&2` (use `printf`, not `echo`, for anything with `\`, `-`, or variables).

### F. Here-docs & portability
```bash
cat <<'EOF' > config            # quoted 'EOF' = literal, NO expansion (safe for secrets/templates)
$HOME stays literal
EOF
cat <<EOF                       # unquoted = expand variables
running as $USER
EOF
indented() { cat <<-EOF         # <<- strips leading TABS (only tabs) for indentation
	indented body
	EOF
}
```

**Bash vs POSIX `sh`:** arrays, `[[ ]]`, `local`, `${x^^}`, `<<<`, and `function` are *bashisms* — they break under `dash`/`/bin/sh`. Pick one and declare it in the shebang (`#!/usr/bin/env bash` for bashisms; `#!/bin/sh` for POSIX), then lint with the matching dialect (`shellcheck -s bash` / `-s sh`). For sibling-shell behavior see [`zsh.md`](guides://zsh.md) and [`fish.md`](guides://fish.md); do not assume Bash arrays/expansions port to them unchanged.

### G. Security binding
Policy is owned by [`secure-coding.md`](guides://secure-coding.md). Bash specifics:
- Never `eval` untrusted data; build commands as arrays (§D), not strings.
- Never `curl … | bash`. Download, checksum-verify, inspect, then run.
- Quote everything (§B) — unquoted expansion is a command/path-injection vector.
- Secrets come from the environment or a secret manager, never source; redact in `set -x` traces.

---

## 6. Tooling

No package manager; Bash scripts depend on system binaries. Pin the toolchain and check dependencies up front.

```bash
shellcheck --version            # BASH-LINT-01 (>= 0.10)
shfmt --version                 # BASH-FMT-01 (>= 3.8)
bats --version                  # BASH-TST-01 (bats-core >= 1.11)

# Fail fast if a required command is missing
for cmd in jq curl; do
  command -v "$cmd" >/dev/null 2>&1 || { printf 'missing dependency: %s\n' "$cmd" >&2; exit 1; }
done
```

Add a `.shellcheckrc` (e.g. `enable=all`, `severity=style`) and an `.editorconfig`/`shfmt` flag set (`-i 2 -ci`) so local and CI gates agree. Wire `shellcheck`, `shfmt -d`, and `bats` as required CI steps (see [`ci-cd.md`](guides://ci-cd.md)).

---

## 7. When NOT to Use Bash

Bash excels at gluing commands, file plumbing, and short automation. Switch to a real language (e.g. [`python.md`](guides://python.md), [`go.md`](guides://go.md)) when the script:

- exceeds ~100–200 lines, or grows nested data structures / parsing logic;
- does non-trivial arithmetic, floating point, or string algorithms;
- needs structured error handling, real types, libraries, or unit-testable modules beyond `bats`;
- manipulates JSON/YAML/CSV beyond a single `jq` filter, or makes structured API calls;
- must be portable across OSes where shell quirks bite.

Choosing the right tool is part of the standard — do not force Bash past its scope.

---

## 8. Quick Reference

```bash
# Verify
shfmt -d -i 2 -ci script.sh          # format check
bash -n script.sh                    # syntax
shellcheck -x script.sh              # lint (mandatory)
bats tests/                          # test
# Fix
shfmt -w -i 2 -ci script.sh          # auto-format
bash -x script.sh                    # trace execution while debugging
```

```bash
# Strict-mode header (copy verbatim)
#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'
trap 'rm -rf -- "${tmp:-}"' EXIT
```

---

## 9. Deployment Checklist

Generated from §2 — one box per requirement ID.

- [ ] BASH-SAFE-01 — `set -euo pipefail` preamble present
- [ ] BASH-SYN-01 — `bash -n` parses clean
- [ ] BASH-LINT-01 — `shellcheck` clean (mandatory)
- [ ] BASH-FMT-01 — `shfmt -d` shows no diff
- [ ] BASH-QUOTE-01 — no unquoted expansions (no SC2086/SC2046)
- [ ] BASH-PORT-01 — shebang matches dialect; bashisms only under `bash`
- [ ] BASH-SEC-01 — no `eval`/`curl|bash` on untrusted input, no secrets in source
- [ ] BASH-ERR-01 — temp files/locks cleaned up via `trap`
- [ ] BASH-TST-01 — `bats` tests pass (non-trivial logic)
- [ ] BASH-DOC-01 — script + functions documented, `--help` works
- [ ] Agent ran every §3 command and documented any fixes

---
**End of Bash Shell Scripting Guidelines**
