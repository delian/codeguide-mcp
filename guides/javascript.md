# JavaScript Development Guidelines
Mandatory coding standards for modern JavaScript: ESM-only, strict-mode, type-checked via JSDoc, test-covered. ES2023+, Node 22+, ESLint 9 / Biome, Vitest.

---
name: javascript
title: JavaScript Development Guidelines
version: 2.0
last_reviewed: 2026-06-05
kind: language
tools: [node@22, eslint@9, biome@2, vitest@3, prettier@3]
requires:
  - tdd
  - secure-coding
  - error-handling
recommends:
  - typescript
  - nodejs
  - comments
  - performance
provides:
  - modern-es
  - esm
  - async-await
  - js-footguns
---

> 🧭 Authored per [`CONVENTIONS.md`](guides://CONVENTIONS.md): shared concerns are referenced, not restated. This guide covers only what is unique to JavaScript the language.

---

## 0. Prerequisites & References

Fetch and apply these **before** generating JavaScript code. Their rules are assumed below and not repeated.

> 📎 **REQUIRED — fetch & apply first:**
> - [`tdd.md`](guides://tdd.md) — test-first, Red-Green-Refactor, regression-test-before-fix, coverage. *(JS binding: runner is Vitest — `npx vitest run`.)*
> - [`secure-coding.md`](guides://secure-coding.md) — supply chain, secrets, CVE policy. *(JS binding: `npm audit`, no `eval`/`Function`, prototype-pollution guards.)*
> - [`error-handling.md`](guides://error-handling.md) — error strategy & propagation. *(JS binding: `Error`-only throws, async rejection handling, see §6.)*

> 📎 **RECOMMENDED — fetch when the task touches them:**
> - [`typescript.md`](guides://typescript.md) — **strongly recommended for any non-trivial codebase**; prefer TS over plain JS + JSDoc when the project allows a build step.
> - [`nodejs.md`](guides://nodejs.md) — runtime APIs, module resolution, package.json fields, when JS runs on Node.
> - [`comments.md`](guides://comments.md) — doc policy *(binding: JSDoc on public exports)*.
> - [`performance.md`](guides://performance.md) — hot-path and allocation guidance.

> 📎 **SEE ALSO:** [`reactjs.md`](guides://reactjs.md) · [`deno.md`](guides://deno.md) · [`designpatterns.md`](guides://designpatterns.md) · [`semver.md`](guides://semver.md)

---

## 1. Core Philosophies: MODERN-JS-FIRST

JavaScript-specific principles only. TDD, security, and error-handling strategy come from §0 — do not restate them.

- **M**odules: ESM exclusively (`import`/`export`). No CommonJS `require`/`module.exports` in new code.
- **O**nly `const`/`let`: `const` by default, `let` when reassigned, **never** `var`.
- **D**eterministic equality: `===`/`!==` only; explicit conversions; no implicit coercion footguns (§5).
- **E**xplicit async: `async`/`await` over raw `.then()` chains; never callback-style for new I/O; never an unhandled rejection.
- **R**euse the platform: prefer built-ins (`Array.prototype.at`, `Object.hasOwn`, `structuredClone`, `Map.groupBy`, `Intl`) over utility libraries.
- **N**on-negotiable quality: lint clean, formatted, type-checked (JSDoc + `tsc --checkJs` or TypeScript), Vitest green at the §2 coverage gate.

> **Strongly recommended:** adopt **TypeScript** ([`typescript.md`](guides://typescript.md)) for any code beyond a small script. Plain JS below assumes you are deliberately build-free; even then, type-check with `tsc --checkJs`.

**Verified Code**: Agent-generated JavaScript MUST pass every gate in §2 before delivery.

---

## 2. Requirements (MANDATORY, auditable)

RFC-2119 keywords. IDs `JS-<TOPIC>-<NN>`. Each row has a binary gate; rows binding a shared rule cite its owner.

| ID | Requirement | Verify | Gate |
|----|-------------|--------|------|
| JS-TST-01 | Every feature MUST be test-first (see `tdd.md`) | `npx vitest run` | exit 0, 0 skips |
| JS-TST-02 | Each bug MUST get a regression test before the fix (see `tdd.md`) | `npx vitest run` | failing→passing |
| JS-TST-03 | Business-logic coverage MUST meet the project gate | `npx vitest run --coverage` | ≥ threshold |
| JS-FMT-01 | Code MUST be formatted | `npx prettier --check .` or `npx biome format .` | no diff |
| JS-LINT-01 | Linter MUST pass clean | `npx eslint .` or `npx biome lint .` | exit 0 |
| JS-TYP-01 | Public APIs MUST be JSDoc-typed and type-check | `npx tsc --checkJs --noEmit` | exit 0 |
| JS-ESM-01 | Sources MUST be ESM (no `require`/`module.exports`) | `"type":"module"` + grep | no CJS in src |
| JS-STRICT-01 | No `var`; `===` only; no `eval`/`Function` (see `secure-coding.md`) | `npx eslint .` (`no-var`,`eqeqeq`,`no-eval`) | exit 0 |
| JS-ERR-01 | Reject/throw `Error` objects; no unhandled rejections (see `error-handling.md`) | eslint `no-throw-literal`, review | exit 0 |
| JS-DOC-01 | Public exports documented with JSDoc (see `comments.md`) | `npx eslint .` (`jsdoc/*`) | exit 0 |
| JS-SEC-01 | 0 high/critical CVEs in deps (see `secure-coding.md`) | `npm audit --audit-level=high` | 0 high/critical |
| JS-DEP-01 | Lockfile in sync & verified (see `secure-coding.md`) | `npm ci` / `npm audit signatures` | installs clean |

> **Forbidden**: shipping implementation before its test (violates `tdd.md`), fixing a bug without a regression test first, `var`, `==`/`!=` (except the deliberate `== null` idiom), `eval`/`new Function`, swallowing a rejected promise, or throwing non-`Error` values.

---

## 3. Verification Protocol

Run, in order, before presenting code. Fix → re-run until every gate is green.

```bash
npx prettier --check .            # JS-FMT-01   (or: biome format .)
npx eslint .                      # JS-LINT-01/STRICT-01/ERR-01/DOC-01  (or: biome lint .)
npx tsc --checkJs --noEmit        # JS-TYP-01   (ESLint does NOT type-check)
npx vitest run --coverage         # JS-TST-01/02/03
npm audit --audit-level=high      # JS-SEC-01
npm ci                            # JS-DEP-01   (lockfile reproducible)
```

The *why* behind each gate lives in its §0 owner; do not re-derive it here.

---

## 4. Project Structure

Idiomatic ESM layout. Architectural principles (dependency direction, boundaries) are owned by the project's architecture guide; below is only the JS mapping.

```
project/
├── src/
│   ├── domain/           # pure logic — no I/O imports
│   ├── services/         # use cases / orchestration
│   ├── adapters/         # http/db/fs implementations
│   └── index.js          # entry; "exports" mapped in package.json
├── test/                 # mirrors src/ (see tdd.md); *.test.js
├── package.json          # "type":"module", "exports", scripts, engines
├── eslint.config.js      # flat config (ESLint 9) — or biome.json
├── vitest.config.js
└── package-lock.json     # committed lockfile
```

- `package.json` MUST set `"type": "module"` and an `"engines": { "node": ">=22" }` floor.
- Define the public surface with the `"exports"` map; avoid deep imports into internals.
- Group by feature/domain, not by file type. No circular imports.

---

## 5. JavaScript Language Specifics

The unique value of this guide — language semantics that bite even experienced engineers.

### A. Modules (ESM)
ESM is static, strict-by-default, and asynchronously graph-loaded.

```js
import { readFile } from 'node:fs/promises';   // named
import config from './config.js';              // default; extension REQUIRED in ESM
export const VERSION = '2.0';
export { parse } from './parse.js';            // re-export
const mod = await import('./plugin.js');       // dynamic, code-split, lazy
```

- File extensions are mandatory in relative specifiers; `node:` prefix for built-ins.
- ESM has no `__dirname`/`require`; use `import.meta.url` + `node:url` `fileURLToPath`.
- Top-level `await` is allowed in modules. There is no `module.exports`/`require` interop in new code.

### B. Scope, closures & the TDZ
`let`/`const` are block-scoped and hoisted into a **temporal dead zone** — referencing before declaration throws. `const` binds the variable, not the value (objects stay mutable; freeze with `Object.freeze`/`structuredClone`).

```js
// Closure-per-iteration: `let` gives each iteration its own binding.
const handlers = [];
for (let i = 0; i < 3; i++) handlers.push(() => i);   // 0,1,2  (with var: 3,3,3)
```

### C. Equality, coercion & nullish (footguns)
Use `===`. The one sanctioned loose check is `x == null` (true for `null` **and** `undefined`).

```js
0 == '';        // true   — coercion trap
NaN === NaN;    // false  — use Number.isNaN(x)
[] == false;    // true   — never rely on it
typeof null;    // 'object'  (historical bug)
0 ?? 'd';       // 0       — ?? only falls back on null/undefined
0 || 'd';       // 'd'     — || falls back on any falsy
a?.b?.();       // optional chaining short-circuits to undefined
```

- Compare numbers with `Number.isNaN` / `Object.is`; never `=== NaN`.
- Reach for `??`/`?.` not `||`/`&&` when `0`/`''`/`false` are valid values.

### D. Destructuring, spread & rest
```js
const { host = 'localhost', port = 3000, ...extra } = config;   // defaults + rest
const [first, , third = 0] = items;                              // skip, default
const merged = { ...base, ...override };                         // shallow merge
fn(...args);                                                     // spread call
const { a: { b } = {} } = obj;                                   // nested + guard
```
Spread is **shallow** — nested objects are shared references. Deep-copy with `structuredClone(x)`.

### E. Async, promises & concurrency
`async`/`await` for sequencing; combinators for fan-out. Always handle rejection; bind concurrency policy to [`error-handling.md`](guides://error-handling.md), perf to [`performance.md`](guides://performance.md).

```js
const [u, posts] = await Promise.all([fetchUser(id), fetchPosts(id)]);  // fail-fast
const results = await Promise.allSettled(tasks);                        // collect all
const { promise, resolve, reject } = Promise.withResolvers();           // ES2024
```

- Never `await` inside a `for` loop when items are independent — use `Promise.all` (perf).
- A floating promise = silent failure. `await` it, `return` it, or `.catch()` it.
- Cancellation is via `AbortController`/`AbortSignal`, not ad-hoc flags.
- CPU-bound work belongs on `worker_threads`, not the event loop (see [`nodejs.md`](guides://nodejs.md), [`parallelism.md`](guides://parallelism.md)).

### F. Iterators & generators
```js
function* range(n) { for (let i = 0; i < n; i++) yield i; }     // lazy
async function* lines(stream) { for await (const c of stream) yield c; }  // async iter
const set = new Set([1, 1, 2]);                                 // iterables
```
Any object with `[Symbol.iterator]` works in `for…of`/spread. Generators give lazy, O(1)-memory pipelines; `for await…of` consumes async iterables (streams).

### G. Prototypes, classes & `this`
Classes are prototype sugar. `this` is bound at **call time** — arrow methods or explicit `.bind` for callbacks.

```js
class Cache {
  #store = new Map();                         // true private field
  static from(obj) { return Object.assign(new Cache(), obj); }
  get size() { return this.#store.size; }     // accessor
  has = (k) => this.#store.has(k);            // arrow field → `this` always bound
}
```
- Prefer `Object.hasOwn(o, k)` over `o.hasOwnProperty`; `Object.create(null)` for map-like objects to avoid prototype keys.
- Prototype pollution is a security issue — never assign from untrusted keys to `__proto__`/`constructor`/`prototype` (see [`secure-coding.md`](guides://secure-coding.md)).

### H. Strict mode & modern built-ins
ESM and class bodies are **always** strict — no implicit globals, no silent assignment failures, no octal/`with`. Prefer current platform APIs over libraries: `structuredClone`, `Array.prototype.{at,findLast,toSorted,toReversed,with}` (ES2023, non-mutating), `Object.groupBy`/`Map.groupBy` (ES2024), `Intl.*` for i18n, `URL`/`URLSearchParams`, `fetch`, `crypto.randomUUID()`.

---

## 6. Errors & Documentation Bindings

Strategy is owned by [`error-handling.md`](guides://error-handling.md); doc policy by [`comments.md`](guides://comments.md). JavaScript bindings only:

```js
class NotFoundError extends Error {
  constructor(id) { super(`User ${id} not found`); this.name = 'NotFoundError'; }
}
try {
  await save(user);
} catch (err) {
  throw new Error('save failed', { cause: err });   // ES2022 error cause — preserve chain
}
```

- Throw/reject **`Error` instances only** (carry `name`, `message`, `cause`, `stack`); never throw strings.
- `catch (err)` binding is `unknown`-shaped — narrow with `err instanceof X` before using.
- JSDoc on every public export so `tsc --checkJs` and editors get types without a `.ts` file:

```js
/**
 * @param {string} id
 * @param {{ timeout?: number, signal?: AbortSignal }} [opts]
 * @returns {Promise<User|null>}
 * @throws {NotFoundError}
 */
export async function fetchUser(id, opts = {}) { /* … */ }
```

---

## 7. Tooling & Dependencies

Lint/format with **ESLint 9 (flat config)** or **Biome** (faster, lint+format in one). Security/supply-chain *policy* → [`secure-coding.md`](guides://secure-coding.md); versioning → [`semver.md`](guides://semver.md).

```js
// eslint.config.js — flat config (ESLint 9)
import js from '@eslint/js';
export default [
  js.configs.recommended,
  {
    languageOptions: { ecmaVersion: 2023, sourceType: 'module' },
    rules: {
      'no-var': 'error',
      'prefer-const': 'error',
      eqeqeq: ['error', 'always', { null: 'ignore' }],  // allow `== null`
      'no-eval': 'error',
      'no-implied-eval': 'error',
      'no-throw-literal': 'error',
      'require-await': 'error',
      'no-floating-promises': 'error',
    },
  },
];
```

```bash
npm ci                      # JS-DEP-01: reproducible install from lockfile
npm install <pkg>           # add dep (updates package-lock.json)
npm update                  # update within semver ranges
npm audit --audit-level=high  # JS-SEC-01: CVE scan
npm audit signatures        # registry signature verification
```
Commit `package-lock.json`. Set `"engines": { "node": ">=22" }`. Prefer zero/few-dependency native solutions to shrink the supply-chain surface.

---

## 8. Quick Reference

```bash
npm ci                              # setup
npx vitest run                      # test     (vitest --watch for TDD loop)
npx eslint . && npx prettier -w .   # lint + format   (or: npx biome check --write .)
npx tsc --checkJs --noEmit          # type check via JSDoc
node src/index.js                   # run
npm audit --audit-level=high        # CVE scan
```

```js
// idiom cheat-sheet
x ?? d;  a?.b;  arr.at(-1);  arr.toSorted();  Object.hasOwn(o,k);
structuredClone(o);  Map.groupBy(items, fn);  Promise.allSettled(ps);
const { a, ...rest } = obj;  [...new Set(arr)];  crypto.randomUUID();
```

---

## 9. Deployment Checklist

Generated from §2 — one box per requirement ID.

- [ ] JS-FMT-01 — formatted, no diff
- [ ] JS-LINT-01 — ESLint/Biome clean
- [ ] JS-TYP-01 — `tsc --checkJs` clean (real type check, not the linter)
- [ ] JS-ESM-01 — ESM only, `"type":"module"`, no CJS in src
- [ ] JS-STRICT-01 — no `var`, `===` only, no `eval`/`Function`
- [ ] JS-ERR-01 — `Error`-only throws, no unhandled rejections
- [ ] JS-TST-01/02/03 — tests pass, bugs have regression tests, coverage ≥ gate
- [ ] JS-DOC-01 — public exports JSDoc-documented
- [ ] JS-SEC-01 — `npm audit` 0 high/critical CVEs
- [ ] JS-DEP-01 — `package-lock.json` in sync & committed
- [ ] Agent ran every §3 command and documented any fixes

---
**End of JavaScript Guidelines**
