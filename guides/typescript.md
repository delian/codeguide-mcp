# TypeScript Development Guidelines
Mandatory coding standards for TypeScript: strict types, ESM, runtime-validated boundaries, test-covered. TypeScript 5.8+, tsc, typescript-eslint/Biome, Vitest, tsx, tsup.

---
name: typescript
title: TypeScript Development Guidelines
version: 2.0
last_reviewed: 2026-06-05
kind: language
tools: [typescript@5.8, tsc, typescript-eslint@8, biome@2, vitest@3, tsx, tsup, npm]
requires:
  - tdd
  - secure-coding
  - error-handling
recommends:
  - nodejs
  - javascript
  - zod
  - hexagonal
  - comments
  - semver
provides:
  - ts-type-system
  - strict-config
  - generics
  - type-narrowing
  - esm
---

> 🧭 Authored per [`CONVENTIONS.md`](guides://CONVENTIONS.md): shared concerns are referenced, not restated. This guide covers only what is unique to TypeScript — the type system, strict compiler config, ESM, declaration output, and the TS toolchain.

---

## 0. Prerequisites & References

Fetch and apply these **before** generating TypeScript code. Their rules are assumed below and not repeated.

> 📎 **REQUIRED — fetch & apply first:**
> - [`tdd.md`](guides://tdd.md) — test-first, Red-Green-Refactor, regression-test-before-fix, coverage. *(TS binding: runner is `vitest run --coverage`.)*
> - [`secure-coding.md`](guides://secure-coding.md) — supply chain, secrets, CVE policy. *(TS binding: `npm audit`, `npm audit signatures`.)*
> - [`error-handling.md`](guides://error-handling.md) — error strategy & propagation. *(TS binding: discriminated-union `Result<T,E>` and typed `Error` subclasses; §6.)*

> 📎 **RECOMMENDED — fetch when the task touches them:**
> - [`zod.md`](guides://zod.md) — runtime validation/parsing at I/O boundaries (TS types are erased at runtime; §5.G).
> - [`nodejs.md`](guides://nodejs.md) · [`javascript.md`](guides://javascript.md) — the runtime, async/await, Promise, FP idioms TypeScript compiles to. *Do not re-derive JS semantics here.*
> - [`comments.md`](guides://comments.md) — API-doc policy *(binding: TSDoc + TypeDoc).*
> - [`hexagonal.md`](guides://hexagonal.md) — layering, ports/adapters, dependency inversion.
> - [`semver.md`](guides://semver.md) — versioning of published packages and their `.d.ts` surface.

> 📎 **SEE ALSO:** [`reactjs.md`](guides://reactjs.md) · [`nextjs.md`](guides://nextjs.md) · [`deno.md`](guides://deno.md) · [`designpatterns.md`](guides://designpatterns.md) · [`code-review.md`](guides://code-review.md) · [`ci-cd.md`](guides://ci-cd.md)

---

## 1. Core Philosophies: TYPESCRIPT-FIRST

TypeScript-specific principles only. TDD, security, error handling, async/FP idioms, and architecture come from §0.

- **T**ype-complete: every public signature is explicitly typed; the compiler under `strict` is the contract. The linter is **not** a type checker — gate types with `tsc --noEmit`.
- **Y**ank `any`: `any` is banned; use `unknown` at boundaries and narrow. `@ts-ignore`/`@ts-expect-error` only with a justifying comment.
- **P**arse, don't trust: types vanish at runtime; validate all external input with a schema (`zod.md`) before it enters typed code.
- **E**SM-only: native ES modules, `verbatimModuleSyntax`, `import type` for type-only imports; ship `.d.ts` declarations for libraries.
- **S**afe modelling: make illegal states unrepresentable — discriminated unions, branded types, `readonly`, `satisfies`.
- **N**arrow over assert: prefer type guards and control-flow narrowing to `as` casts; reserve casts for proven-safe boundaries.

**Verified Code**: Agent-generated TypeScript MUST pass every gate in §2 before delivery.

---

## 2. Requirements (MANDATORY, auditable)

RFC-2119 keywords. IDs `TS-<TOPIC>-<NN>`. Each row has a binary gate; rows binding a shared rule cite its owner.

| ID | Requirement | Verify | Gate |
|----|-------------|--------|------|
| TS-TST-01 | Every feature MUST be test-first (see `tdd.md`) | `vitest run` | exit 0, 0 skips |
| TS-TST-02 | Each bug MUST get a regression test before the fix (see `tdd.md`) | `vitest run` | failing→passing |
| TS-TST-03 | Business-logic coverage MUST meet the project gate | `vitest run --coverage` | ≥ threshold |
| TS-TYP-01 | Code MUST type-check under `strict` (+`noUncheckedIndexedAccess`) | `tsc --noEmit` | exit 0 |
| TS-TYP-02 | No `any`, no unjustified `@ts-ignore`/`@ts-expect-error` | `eslint .` (no-explicit-any, ban-ts-comment) | exit 0 |
| TS-MOD-01 | ESM + type-only imports enforced | `tsc --noEmit` (`verbatimModuleSyntax`) | exit 0 |
| TS-VAL-01 | External input MUST be schema-validated at boundaries (see `zod.md`) | review / grep for unparsed `JSON.parse`/`req.body` | validated |
| TS-ERR-01 | Errors typed; no thrown non-`Error` values (see `error-handling.md`) | `eslint .` (only-throw-error) | exit 0 |
| TS-FMT-01 | Code MUST be formatted | `biome format --check .` *or* `prettier --check .` | no diff |
| TS-LINT-01 | Type-aware linter MUST pass clean | `eslint .` *or* `biome check .` | exit 0 |
| TS-DOC-01 | Public exports MUST have TSDoc (see `comments.md`) | `typedoc --validation.notDocumented` | no warnings |
| TS-SEC-01 | 0 high/critical CVEs in deps (see `secure-coding.md`) | `npm audit --audit-level=high` | 0 high/critical |
| TS-DEP-01 | Lockfile in sync & signatures verified (see `secure-coding.md`) | `npm ci` / `npm audit signatures` | in sync, verified |
| TS-ARCH-01 | Domain imports no adapter/framework code (see `hexagonal.md`) | review / `eslint` import boundaries | no inward→outward |

> **Forbidden**: shipping implementation before its test (violates `tdd.md`), fixing a bug without a regression test first, using `any` or unjustified `@ts-ignore` to silence the compiler, trusting `unknown` external data without parsing it (violates `zod.md`), throwing non-`Error` values, or using `eslint`/`biome` as a substitute for `tsc`.

---

## 3. Verification Protocol

Run, in order, before presenting code. Fix → re-run until every gate is green.

```bash
biome format --check .          # TS-FMT-01   (or: prettier --check .)
eslint .                        # TS-LINT-01/TYP-02/ERR-01  (or: biome check .)
tsc --noEmit                    # TS-TYP-01/MOD-01  (lint does NOT type-check)
vitest run --coverage           # TS-TST-01/03
npm audit --audit-level=high    # TS-SEC-01
npm audit signatures            # TS-DEP-01
typedoc --validation.notDocumented true --emit none   # TS-DOC-01
```

The *why* behind each gate lives in its §0 owner; do not re-derive it here.

---

## 4. Project Structure

Idiomatic ESM `src/` layout. Architectural principles (dependency direction, ports/adapters, acyclic deps) are owned by [`hexagonal.md`](guides://hexagonal.md); below is only their TypeScript mapping.

```
project/
├── src/
│   ├── domain/          # pure business logic — no framework/IO imports (TS-ARCH-01)
│   ├── application/     # use cases, orchestrates ports
│   ├── adapters/        # db/http/cli implementations of ports
│   └── index.ts         # public entry; only this is in tsup `entry`
├── tests/               # mirrors src/ (see tdd.md) — *.test.ts
├── tsconfig.json        # strict compiler config (§5.A)
├── tsconfig.build.json  # extends base, emits .d.ts for libraries
├── biome.json / eslint.config.ts   # one formatter+linter, type-aware rules
├── tsup.config.ts       # build (libraries)
├── vitest.config.ts
├── package.json         # "type": "module", "exports" map, types field
└── package-lock.json    # committed lockfile
```

- `"type": "module"` in `package.json`; use the `"exports"` field (not legacy `"main"`/`"typings"`) to publish ESM + `.d.ts`.
- Group by domain/feature, not by type. Enforce the import boundary with an ESLint `no-restricted-imports`/boundaries rule.

---

## 5. TypeScript Specifics

The unique value of this guide: the type system and toolchain.

### A. Strict compiler config — the contract

`strict` is the floor, not the ceiling. Enable the extra safety flags; these are what separate "TypeScript" from "JS with annotations".

```jsonc
// tsconfig.json
{
  "compilerOptions": {
    "target": "ES2023",
    "module": "NodeNext",            // or "Preserve" + bundler
    "moduleResolution": "NodeNext",
    "lib": ["ES2023"],

    "strict": true,                  // implies noImplicitAny, strictNullChecks, etc.
    "noUncheckedIndexedAccess": true,// arr[i] is T | undefined — catches the #1 footgun
    "exactOptionalPropertyTypes": true,
    "noImplicitOverride": true,
    "noFallthroughCasesInSwitch": true,
    "noImplicitReturns": true,
    "noPropertyAccessFromIndexSignature": true,

    "verbatimModuleSyntax": true,    // forces explicit `import type`; clean ESM emit
    "isolatedModules": true,         // safe for single-file transpilers (tsx/esbuild/swc)
    "esModuleInterop": true,
    "resolveJsonModule": true,

    "declaration": true,             // emit .d.ts (libraries)
    "declarationMap": true,
    "sourceMap": true,
    "skipLibCheck": true,
    "forceConsistentCasingInFileNames": true
  },
  "include": ["src"]
}
```

- `useUnknownInCatchVariables` is on under `strict` — `catch (e)` gives `unknown`; narrow before use.
- Do **not** loosen flags to make code compile. Fix the type instead.

### B. `unknown` vs `any`, narrowing, and guards

`any` disables checking and is contagious; `unknown` keeps the value opaque until proven. Boundaries return `unknown`, then narrow.

```typescript
function len(x: unknown): number {
  if (typeof x === "string" || Array.isArray(x)) return x.length; // narrowed
  throw new TypeError("expected string or array");
}

// User-defined type guard
function isUser(v: unknown): v is User {
  return typeof v === "object" && v !== null && "id" in v && typeof v.id === "string";
}

// Assertion function — narrows after the call
function assertDefined<T>(v: T | null | undefined, msg: string): asserts v is T {
  if (v == null) throw new Error(msg);
}
```

Prefer narrowing (`typeof`, `instanceof`, `in`, discriminant checks, guards) over `as`. Reserve `as` for boundaries you have just validated; never `as any` then `as Target`.

### C. Discriminated unions — make illegal states unrepresentable

The single most valuable TS modelling tool. A shared literal discriminant gives exhaustive, narrowed handling.

```typescript
type Fetch<T> =
  | { status: "idle" }
  | { status: "loading" }
  | { status: "ok"; data: T }
  | { status: "err"; error: AppError };

function render(s: Fetch<User>): string {
  switch (s.status) {
    case "idle":    return "Ready";
    case "loading": return "…";
    case "ok":      return s.data.name;       // data exists here
    case "err":     return s.error.message;   // error exists here
    default:        return assertNever(s);    // compile error if a case is added
  }
}
function assertNever(x: never): never { throw new Error(`unhandled: ${JSON.stringify(x)}`); }
```

### D. Generics, utility, conditional & mapped types

Generics carry types through; constrain them with `extends`. Use the standard utility types before hand-rolling.

```typescript
// Constrained generic + keyof
function pluck<T, K extends keyof T>(obj: T, key: K): T[K] { return obj[key]; }

// Mapped + conditional type
type Mutable<T> = { -readonly [K in keyof T]: T[K] };
type Nullable<T> = { [K in keyof T]: T[K] | null };
type Awaited2<T> = T extends Promise<infer U> ? U : T;   // `infer` extracts

// Template-literal types
type EventName = `on${Capitalize<"click" | "focus">}`;   // "onClick" | "onFocus"
```

Reach for built-ins: `Partial`, `Required`, `Readonly`, `Pick`, `Omit`, `Record`, `Exclude`, `Extract`, `ReturnType`, `Parameters`, `NonNullable`, `Awaited`. Keep conditional/mapped chains shallow — if a type needs a comment to read, extract named aliases.

### E. `satisfies` and branded types

`satisfies` validates a value against a type **without widening** it — you keep the precise literal type and still get the check.

```typescript
const config = {
  port: 8080,
  host: "0.0.0.0",
} satisfies Record<string, string | number>;
// config.port is number (not string|number); typo in keys still errors.
```

Branded (nominal) types stop primitive obsession — a raw `string` can't be passed where a validated value is required:

```typescript
type UserId = string & { readonly __brand: "UserId" };
const toUserId = (s: string): UserId => {
  if (!/^[0-9a-f-]{36}$/.test(s)) throw new TypeError("bad id");
  return s as UserId;            // the ONE sanctioned cast — right after validation
};
```

### F. ESM, type-only imports & declaration files

- Native ESM only. With `verbatimModuleSyntax`, type-only imports MUST be marked so they are erased and never emitted as runtime `import`:
  ```typescript
  import type { User } from "./user.js";          // erased
  import { createUser, type CreateInput } from "./user.js"; // mixed: inline `type`
  ```
- Use `.js` specifiers in import paths under `NodeNext` (they resolve to `.ts` at build) — this is the common ESM footgun.
- Libraries ship declarations: `declaration: true` produces `.d.ts`; expose them via `package.json` `"exports": { ".": { "types": "./dist/index.d.ts", "import": "./dist/index.js" } }`.
- For ambient module/global types use a dedicated `*.d.ts`; never put runtime code in a declaration file.

### G. Runtime validation at the boundary — zod binding

Types are erased at compile time, so `JSON.parse`, `fetch`, `req.body`, env vars, and DB rows are `unknown` no matter what you annotate. Validation policy/schema design is owned by [`zod.md`](guides://zod.md). TS binding: parse at the edge and **infer** the static type from the schema so there is one source of truth.

```typescript
import { z } from "zod";
const User = z.object({ id: z.string().uuid(), email: z.email() });
type User = z.infer<typeof User>;                 // type derived from schema
const user = User.parse(await res.json());        // throws on bad shape → typed User
```

Never cast external data (`as User`) in place of parsing it.

### H. Common footguns → fix
- `arr[i]` typed as `T` not `T | undefined` → enable `noUncheckedIndexedAccess`.
- Casting away errors (`x as any`, `as unknown as T`) → narrow or validate instead.
- `enum` (emits runtime code, awkward under `isolatedModules`) → prefer union of string literals or `as const` objects.
- Non-null `!` assertions hiding real nulls → guard or `assertDefined`.
- Missing `.js` extension in ESM import paths → add it.
- `catch (e)` treating `e` as `Error` → it's `unknown`; check `e instanceof Error` (see `error-handling.md`).

---

## 6. Errors & Async — TypeScript binding

Strategy (when to throw vs. return, propagation, retries) is owned by [`error-handling.md`](guides://error-handling.md); async/Promise/FP semantics by [`javascript.md`](guides://javascript.md) / [`nodejs.md`](guides://nodejs.md). TypeScript adds the *typing*:

- Model recoverable failures as a discriminated-union `Result<T, E>` (§5.C) rather than throwing across layers; reserve `throw` for truly exceptional cases.
- Subclass `Error` for typed domain errors; set `name` and use the `cause` option. Throw only `Error` instances (enforced by `eslint` `only-throw-error`).
- Type async functions explicitly: `Promise<User | null>`, not inferred. `catch` variables are `unknown` — narrow with `instanceof` before access.

```typescript
class NotFoundError extends Error {
  readonly code = "NOT_FOUND" as const;
  constructor(id: string, opts?: { cause?: unknown }) {
    super(`not found: ${id}`, opts);
    this.name = "NotFoundError";
  }
}
type Result<T, E = Error> = { ok: true; value: T } | { ok: false; error: E };
```

---

## 7. Tooling & Dependencies

Security/supply-chain *policy* → [`secure-coding.md`](guides://secure-coding.md); versioning → [`semver.md`](guides://semver.md). TypeScript binding:

```bash
npm ci                  # TS-DEP-01: install exactly from lockfile (reproducible)
npm install <pkg>       # add dep (updates package-lock.json)
npm audit --audit-level=high   # TS-SEC-01: CVE scan
npm audit signatures    # TS-DEP-01: verify registry signatures
```

- One formatter + one linter. Modern choice: **Biome** (`biome check`) for speed, or **typescript-eslint v8** with `eslint.config.ts` (flat config) and **type-aware** rules (`recommendedTypeChecked`) plus **Prettier**. Do not run both ESLint and Biome as linters.
- **Build:** apps run directly with `tsx`/Node 20+ `--experimental-strip-types`; libraries bundle with **tsup** (esbuild) emitting ESM + `.d.ts`. `tsc` remains the *type-check* gate (`--noEmit`) even when esbuild does the transpile.
- Audit `@types/*` packages too — they are real dependencies and can drift from their runtime counterparts.
- Commit `package-lock.json`. Pin/constrain direct deps; let the resolver handle the graph.

---

## 8. Quick Reference

```bash
tsc --noEmit                         # type check (the real gate)
vitest run --coverage                # test
eslint .            # or: biome check .   # lint
biome format --write .   # or: prettier --write .   # format
tsx src/index.ts                     # run (no build step)
tsup                                 # build lib → dist (ESM + .d.ts)
typedoc                              # API docs (see comments.md)
```

Type-pattern cheat sheet:

```typescript
type Result<T, E = Error> = { ok: true; value: T } | { ok: false; error: E };  // discriminated union
type UserId = string & { readonly __brand: "UserId" };                          // branded
function isUser(v: unknown): v is User { /* … */ }                              // guard
const cfg = { port: 8080 } satisfies AppConfig;                                 // checked, not widened
type DeepReadonly<T> = { readonly [K in keyof T]: DeepReadonly<T[K]> };          // mapped/recursive
```

---

## 9. Deployment Checklist

Generated from §2 — one box per requirement ID.

- [ ] TS-TYP-01 — `tsc --noEmit` clean under `strict` + `noUncheckedIndexedAccess`
- [ ] TS-TYP-02 — no `any`, no unjustified `@ts-ignore`/`@ts-expect-error`
- [ ] TS-MOD-01 — ESM + `import type` enforced (`verbatimModuleSyntax`)
- [ ] TS-VAL-01 — external input schema-validated at boundaries (zod)
- [ ] TS-ERR-01 — errors typed; only `Error` instances thrown
- [ ] TS-FMT-01 — formatter check clean
- [ ] TS-LINT-01 — type-aware linter clean (not a substitute for `tsc`)
- [ ] TS-TST-01/02/03 — tests pass, bugs have regression tests, coverage ≥ gate
- [ ] TS-DOC-01 — public exports have TSDoc, TypeDoc validates
- [ ] TS-SEC-01 — `npm audit` 0 high/critical
- [ ] TS-DEP-01 — `package-lock.json` in sync, signatures verified, committed
- [ ] TS-ARCH-01 — domain layer free of adapter/framework imports
- [ ] Agent ran every §3 command and documented any fixes

---
**End of TypeScript Guidelines**
