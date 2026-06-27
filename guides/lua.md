# Lua Development Guidelines
Mandatory coding standards for Lua: minimal, local-scoped, table-driven, test-covered, portable across PUC-Lua and LuaJIT. Lua 5.4, LuaJIT, luacheck, busted, stylua, LuaRocks, LDoc.

---
name: lua
title: Lua Development Guidelines
version: 2.0
last_reviewed: 2026-06-05
kind: language
tools: [lua@5.4, luajit, luacheck, busted, stylua, luarocks, ldoc]
requires:
  - tdd
  - secure-coding
  - error-handling
recommends:
  - comments
  - performance
provides:
  - lua-tables
  - metatables
  - coroutines
  - pcall-errors
---

> 🧭 Authored per [`CONVENTIONS.md`](guides://CONVENTIONS.md): shared concerns are referenced, not restated. This guide covers only what is unique to Lua.

---

## 0. Prerequisites & References

Fetch and apply these **before** generating Lua code. Their rules are assumed below and not repeated.

> 📎 **REQUIRED — fetch & apply first:**
> - [`tdd.md`](guides://tdd.md) — test-first, Red-Green-Refactor, regression-test-before-fix, coverage. *(Lua binding: runner is `busted`; coverage via `busted --coverage` + `luacov`.)*
> - [`secure-coding.md`](guides://secure-coding.md) — supply chain, secrets, CVE policy. *(Lua binding: pin rockspec versions; avoid `load`/`loadstring`/`os.execute` on untrusted input.)*
> - [`error-handling.md`](guides://error-handling.md) — error strategy & propagation. *(Lua binding: `error`/`pcall`/`xpcall`; return `nil, err` for expected failures.)*

> 📎 **RECOMMENDED — fetch when the task touches them:**
> - [`comments.md`](guides://comments.md) — API-doc policy *(binding: LDoc `---` doc comments)*
> - [`performance.md`](guides://performance.md) — performance policy *(binding: LuaJIT, local caching, table pre-sizing)*

> 📎 **SEE ALSO:** [`architectures.md`](guides://architectures.md) · [`hexagonal.md`](guides://hexagonal.md) *(if the host app mandates ports/adapters)* · [`semver.md`](guides://semver.md) *(rockspec version policy)*

---

## 1. Core Philosophies: LUA-FIRST

Lua-specific principles only. TDD, security, and error handling come from §0.

- **L**ocal by default: every variable, function, and required module is `local`. A bare assignment creates a global — luacheck must flag it. Cache hot globals (`local floor = math.floor`).
- **U**niform data model: the **table** is Lua's one and only data structure — array, map, object, namespace, and set are all tables. Master it before anything else.
- **A**ssume embedding: Lua is a guest in a C/host process. Code MUST NOT crash the host: protect IO and untrusted boundaries with `pcall`/`xpcall` (policy: `error-handling.md`).
- **F**ail loud or return `nil, err`: programmer errors `error()`; expected failures return `nil, message` (the stdlib convention, e.g. `io.open`). Never silently swallow.
- **I**diomatic & minimal: prefer the small standard library and language idioms over frameworks; no global state; one module = one file returning one value.
- **R**untime contracts: Lua is dynamically typed — validate at boundaries with `assert`/`type` and document contracts in LDoc (policy: `comments.md`).
- **S**ame code, two runtimes: keep code portable across PUC-Lua 5.4 and LuaJIT (≈5.1 + extensions); gate version- or FFI-specific paths explicitly.

**Verified Code**: Agent-generated Lua MUST pass every gate in §2 before delivery.

---

## 2. Requirements (MANDATORY, auditable)

RFC-2119 keywords. IDs `LUA-<TOPIC>-<NN>`. Each row has a binary gate; rows binding a shared rule cite its owner.

| ID | Requirement | Verify | Gate |
|----|-------------|--------|------|
| LUA-TST-01 | Every feature MUST be test-first (see `tdd.md`) | `busted` | exit 0, 0 pending |
| LUA-TST-02 | Each bug MUST get a regression test before the fix (see `tdd.md`) | `busted` | failing→passing |
| LUA-TST-03 | Business-logic coverage MUST meet the project gate | `busted --coverage && luacov` | ≥ threshold |
| LUA-SYN-01 | Every file MUST compile | `luac -p <files>` (or `luajit -bl`) | exit 0 |
| LUA-FMT-01 | Code MUST be formatted | `stylua --check .` | no diff |
| LUA-LINT-01 | Linter MUST pass clean (no globals, no unused) | `luacheck .` | exit 0, 0 warnings |
| LUA-SEC-01 | No `load`/`loadstring`/`dofile` on untrusted input; no unsanitized `os.execute`/`io.popen` (see `secure-coding.md`) | `luacheck .` + review | 0 findings |
| LUA-SEC-02 | Dependencies pinned & scanned (see `secure-coding.md`) | rockspec review / `trivy fs .` | exact pins, 0 high/critical |
| LUA-ERR-01 | Host/IO boundaries protected; expected failures return `nil, err` (see `error-handling.md`) | review / tests | no unguarded boundary |
| LUA-DOC-01 | Public modules/functions MUST have LDoc (see `comments.md`) | `ldoc -f markdown .` | builds, no warnings |

> **Forbidden**: shipping implementation before its test (violates `tdd.md`), fixing a bug without a regression test first, creating accidental globals, modifying a sequence with `table.remove` while iterating it with `ipairs`, or comparing `1`-based and `0`-based indices across the C boundary without translation.

---

## 3. Verification Protocol

Run, in order, before presenting code. Fix → re-run until every gate is green.

```bash
stylua --check .                 # LUA-FMT-01
luacheck .                       # LUA-LINT-01 / LUA-SEC-01
luac -p $(find . -name '*.lua')  # LUA-SYN-01  (use luajit -bl for LuaJIT)
busted --coverage                # LUA-TST-01/02/03
luacov                           # LUA-TST-03: coverage report
ldoc -f markdown .               # LUA-DOC-01
```

The *why* behind each gate lives in its §0 owner; do not re-derive it here.

---

## 4. Project Structure

Idiomatic LuaRocks layout. Architectural *principles* (dependency direction, ports/adapters) are owned by [`architectures.md`](guides://architectures.md) / [`hexagonal.md`](guides://hexagonal.md); below is only the Lua mapping.

```
project/
├── src/<package>/
│   ├── init.lua          # module root: returns one table
│   └── <feature>.lua     # one module per file
├── spec/                 # busted specs, mirror src/ (see tdd.md)
│   └── <feature>_spec.lua
├── <package>-1.0.0-1.rockspec   # deps (pinned) + build.modules map
├── .luacheckrc           # lint config (std, globals)
├── stylua.toml           # format config
├── .luacov               # coverage config
└── config.ld             # LDoc config
```

- One module per file; the file `return`s a single value (usually a table). Name it `local M = {}` … `return M`.
- Map every module path to a file in `build.modules` of the rockspec — `require("pkg.feature")` must resolve.
- Keep modules side-effect-free at load time: requiring a module must not perform IO.

---

## 5. Lua Specifics

The unique value of this guide.

### A. Tables — the one data structure

A table is simultaneously an array (sequence) and a hash. The **sequence** part is the contiguous integer keys `1..n`.

```lua
local t = { 10, 20, 30, name = "x" }   -- array part + hash part in one table
#t            -- 3  (length operator: only valid on a borderless sequence)
t[#t + 1] = 40                          -- append idiom
```

- **`#` is undefined when the sequence has holes.** `{1, nil, 3}` may report length 1 *or* 3. Never `nil` out a middle element and then use `#`; track length explicitly or use `table.remove`.
- `pairs` iterates *all* keys in unspecified order; `ipairs` iterates the sequence `1..n` and stops at the first `nil`.
- **Do not insert/remove during iteration.** Removing while iterating with `ipairs` skips elements or errors; build a new table or iterate indices downward (`for i = #t, 1, -1`).

### B. 1-based indexing footguns

Lua sequences start at **1**, not 0. `t[0]` is just another hash key, not part of the sequence.

```lua
for i = 1, #t do ... end          -- canonical sequence loop
string.sub(s, 1, 1)               -- first char (1-based, inclusive both ends)
string.find(s, "x")               -- returns 1-based start, end (or nil)
```

When crossing the **C API / FFI boundary**, C arrays are 0-based — translate indices explicitly (`c_array[i - 1]`). Off-by-one across this boundary is the most common embedding bug (LUA-SEC/ERR boundary).

### C. Scoping & closures

`local` is lexically scoped; a function captures *upvalues* by reference, giving closures and private state.

```lua
local function counter()
  local n = 0
  return function() n = n + 1; return n end   -- n is a private upvalue
end
```

- A bare name without `local` writes a **global** — the #1 source of action-at-a-distance bugs. luacheck (`LUA-LINT-01`) must reject undeclared globals.
- Closures replace classes for simple encapsulation; use the metatable pattern (§D) when you need shared methods + `self`.

### D. Metatables & metamethods

Metatables give tables operator overloading, inheritance, and proxy behaviour. Prototype-based OOP is `__index`:

```lua
local Vec = {}
Vec.__index = Vec                       -- method lookup falls back to Vec
function Vec.new(x, y) return setmetatable({ x = x, y = y }, Vec) end
function Vec:len() return math.sqrt(self.x^2 + self.y^2) end   -- self via `:`
function Vec.__add(a, b) return Vec.new(a.x + b.x, a.y + b.y) end
function Vec.__tostring(v) return ("(%g,%g)"):format(v.x, v.y) end
```

Key metamethods: `__index`/`__newindex` (read/write fallback — inheritance & read-only proxies), `__add`/`__eq`/`__lt`/`__call`, `__tostring`, `__gc` (finalizers, 5.4), `__mode` (weak tables for caches: `__mode = "k"|"v"|"kv"`). Inheritance chains: set the subclass's metatable's `__index` to the superclass.

### E. Modules & `require`

```lua
-- src/mypkg/math.lua
local M = {}
function M.add(a, b) return a + b end
return M                                 -- always return one value
```

```lua
local mathx = require("mypkg.math")      -- cached: runs the file once, reuses result
```

- `require` searches `package.path` (Lua) and `package.cpath` (C), caches in `package.loaded`. A module's top level runs **once**.
- Never rely on the deprecated global `module()` function (removed/forbidden). Never set globals from a module.
- Dots in module names map to directory separators via the path templates.

### F. Coroutines

Cooperative, single-threaded "stackful" coroutines — for generators, iterators, async/IO scheduling, and state machines. They yield/resume; they are **not** OS threads (no preemption, no parallelism — for that see [`performance.md`](guides://performance.md) on multiple Lua states).

```lua
local function range(n)                  -- generator
  return coroutine.wrap(function()
    for i = 1, n do coroutine.yield(i) end
  end)
end
for i in range(3) do print(i) end        -- 1 2 3
```

- `coroutine.create` + `resume`/`yield`/`status` for full control; `coroutine.wrap` wraps a coroutine as an iterator function (errors propagate instead of returning `false, err`).
- `resume` is protected (returns `false, err` on error) — `wrap` is not, so wrap-driven errors raise. Choose per call site.
- A coroutine that yields across a C call is only safe on Lua 5.4 / LuaJIT with `lua_callk`-aware hosts — verify before yielding through C.

### G. Error handling — pcall/error binding

Policy (when to raise vs. return, propagation, context) is owned by [`error-handling.md`](guides://error-handling.md). Lua binding:

```lua
local ok, result = pcall(may_fail, arg)              -- catch; ok=false → result is the error
if not ok then return nil, ("load failed: %s"):format(result) end

local ok2, res2 = xpcall(may_fail, debug.traceback)  -- attach a stack trace at throw site
```

- `error(msg)` raises; `error(msg, 2)` blames the **caller** (better messages for argument validation). `error(table)` raises a structured error object — pcall returns it unchanged, enabling typed errors.
- **Stdlib convention:** expected failures return `nil, message` (e.g. `io.open`, `tonumber`); reserve `error()`/`assert()` for programmer/contract violations and protect host boundaries with `pcall`.
- `assert(v, msg)` raises `msg` when `v` is falsy — concise contract checks; remember `assert` evaluates its message argument eagerly.

### H. Standard library essentials

Small and portable — prefer it over dependencies. `string` (pattern matching — **Lua patterns, not regex**: `%a %d %s`, anchors `^ $`, `-` lazy, captures `()`; `string.format`, `gsub`, `gmatch`), `table` (`insert`, `remove`, `concat` for O(n) joins, `sort`, `unpack`/`table.unpack`), `math`, `os`/`io` (sandbox these at untrusted boundaries — see `secure-coding.md`), `utf8` (5.3+). Build strings with `table.concat`, never repeated `..` in a loop.

### I. LuaJIT vs PUC-Lua

| | PUC-Lua 5.4 | LuaJIT |
|---|---|---|
| Language base | 5.4 (integers, `goto`, `<close>`, bitwise ops) | 5.1 + select 5.2/5.3 extensions |
| Speed | reference interpreter | JIT-compiled, often 10–100× hot loops |
| FFI | none (C modules only) | `ffi` library — call C directly, struct cdata |
| Integers | true 64-bit integer subtype | all numbers are doubles (no integer subtype) |

- Write to the **common subset** unless the project targets one runtime. Guard 5.4-only syntax (`//`, `<close>`, `goto`, bitwise `&`/`|`) and LuaJIT-only `ffi`/`bit` behind explicit detection.
- Performance work (object pooling, table pre-sizing, avoiding allocations in hot loops, NYI-trace awareness on LuaJIT) is owned by [`performance.md`](guides://performance.md) — apply its rules; do not premature-optimize PUC code that should run on LuaJIT.

### J. Embedding & the C API (basics)

Lua's reason for existing is embedding. The C side drives a virtual **stack** (1-based, like Lua sequences):

```c
lua_State *L = luaL_newstate();
luaL_openlibs(L);
if (luaL_dofile(L, "script.lua") != LUA_OK)   /* protected: never longjmps past you */
    fprintf(stderr, "lua: %s\n", lua_tostring(L, -1));   /* error on top of stack */
lua_close(L);
```

- Cross the boundary only through `luaL_*`/`lua_*`; every value passes via the stack — push args, `lua_pcall`, read results, then `lua_settop`/balance the stack.
- **Always call into Lua with `lua_pcall`** (not `lua_call`) from C so a Lua `error` cannot `longjmp` past host cleanup. This is the C-side mirror of LUA-ERR-01.
- Expose C functions as `lua_CFunction` (return count, args on stack); register with a `luaL_Reg` array. On LuaJIT prefer `ffi` over hand-written C bindings for speed and less glue.

### K. Common footguns → fixes

- Accidental global (missing `local`) → enable luacheck `std`+`globals`; treat unset-global warnings as errors.
- `#t` on a table with holes → don't store `nil` in sequences; track count or use `table.remove`.
- `table.remove` inside `ipairs` → iterate downward or rebuild the table (see §5.A).
- `a == b` on tables compares **identity**, not contents → define `__eq` or compare fields explicitly.
- `0`-based assumptions / C-boundary indexing → remember Lua is 1-based; translate at the FFI/C edge.
- `string.find(s, ".")` matches *any* char → it's a pattern; escape with `%.` or pass `true` as the plain flag.
- Numeric `for` with float step / 5.4 integer-vs-float (`3 == 3.0` true, but `math.type` differs) → be explicit when keys or equality depend on subtype.

---

## 6. Tooling & Dependencies

Security/supply-chain *policy* → [`secure-coding.md`](guides://secure-coding.md); versioning → [`semver.md`](guides://semver.md). Lua binding:

```bash
luarocks install --only-deps <pkg>-<ver>.rockspec   # install pinned deps
luarocks make                                       # build/install this project from its rockspec
luarocks list                                        # installed rocks + versions
trivy fs --scanners vuln .                            # LUA-SEC-02: CVE scan (no native LuaRocks audit)
```

- **LuaRocks has no lockfile.** Pin **exact** versions in `dependencies` (e.g. `"luasocket == 3.1.0-1"`), not open ranges — this is the reproducibility gate (LUA-SEC-02).
- A `.rockspec` declares `dependencies` and a `build.modules` map (module name → file). Commit it; tag releases.
- Use a per-project tree (`luarocks --tree ./.rocks` or `luarocks --local`) to avoid polluting the system; never `sudo luarocks install`.

---

## 7. Quick Reference

```bash
stylua .                       # format
luacheck .                     # lint (globals, unused, security)
luac -p file.lua               # syntax/compile check
busted --coverage && luacov    # test + coverage
ldoc -f markdown .             # docs
luarocks make                  # build/install from rockspec
```

```lua
local M = {}                         -- module
function M.f() end
return M

local C = {}; C.__index = C          -- class via metatable
function C.new() return setmetatable({}, C) end

local ok, err = pcall(risky)         -- protected call
local v = (obj or {}).field or DEF   -- safe nil access
for i = 1, #t do end                 -- 1-based sequence loop
```

---

## 8. Deployment Checklist

Generated from §2 — one box per requirement ID.

- [ ] LUA-FMT-01 — `stylua --check` clean
- [ ] LUA-LINT-01 — `luacheck` clean (no globals, no unused)
- [ ] LUA-SYN-01 — every file compiles (`luac -p` / `luajit -bl`)
- [ ] LUA-TST-01/02/03 — tests pass, bugs have regression tests, coverage ≥ gate
- [ ] LUA-SEC-01 — no `load`/`loadstring`/`dofile` or `os.execute`/`io.popen` on untrusted input
- [ ] LUA-SEC-02 — deps pinned exactly in rockspec, CVE scan clean
- [ ] LUA-ERR-01 — host/IO boundaries protected; expected failures return `nil, err`
- [ ] LUA-DOC-01 — public modules/functions have LDoc, docs build clean
- [ ] Agent ran every §3 command and documented any fixes

---
**End of Lua Guidelines**
