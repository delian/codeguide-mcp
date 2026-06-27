# Deno Development Guidelines
Mandatory standards for Deno: secure-by-default permissions, native TypeScript, JSR-first deps, built-in tooling. Deno 2.x, TypeScript 5.x, JSR, @std.

---
name: deno
title: Deno Development Guidelines
version: 2.0
last_reviewed: 2026-06-05
kind: language
tools: [deno@2.4, typescript@5.7, jsr, "@std"]
requires:
  - tdd
  - secure-coding
  - error-handling
recommends:
  - typescript
  - javascript
  - nodejs
  - comments
provides:
  - deno-permissions
  - native-ts
  - jsr
  - deno-tooling
  - web-standard-apis
---

> 🧭 Authored per [`CONVENTIONS.md`](guides://CONVENTIONS.md): shared concerns are referenced, not restated. This guide covers only what is unique to the **Deno runtime** — not the TypeScript language itself.

---

## 0. Prerequisites & References

Fetch and apply these **before** generating Deno code. Their rules are assumed below and not repeated.

> 📎 **REQUIRED — fetch & apply first:**
> - [`tdd.md`](guides://tdd.md) — test-first, Red-Green-Refactor, regression-test-before-fix, coverage. *(Deno binding: `Deno.test` / `deno test`; BDD via `jsr:@std/testing/bdd`.)*
> - [`secure-coding.md`](guides://secure-coding.md) — supply chain, secrets, CVE policy. *(Deno binding: the permission model below + frozen `deno.lock` ARE Deno's enforcement of this policy.)*
> - [`error-handling.md`](guides://error-handling.md) — error strategy & propagation. *(Deno binding: `Result<T,E>` discriminated unions or thrown `Error` subclasses; `Deno.errors.*` for runtime faults.)*

> 📎 **RECOMMENDED — fetch when the task touches them:**
> - [`typescript.md`](guides://typescript.md) — the language. Deno runs TS natively; **all type/idiom rules live there**, not here.
> - [`nodejs.md`](guides://nodejs.md) — for `node:`/`npm:` compat and contrast (see §5.F).
> - [`javascript.md`](guides://javascript.md) · [`comments.md`](guides://comments.md) *(JSDoc; verified by `deno doc --lint`)*

> 📎 **SEE ALSO:** [`zod.md`](guides://zod.md) · [`rest.md`](guides://rest.md) · [`oauth.md`](guides://oauth.md) · [`env-config.md`](guides://env-config.md) · [`performance.md`](guides://performance.md) · [`parallelism.md`](guides://parallelism.md) · [`observability.md`](guides://observability.md) · [`ci-cd.md`](guides://ci-cd.md)

---

## 1. Core Philosophies: DENO

Deno-runtime principles only. TDD, security policy, and error strategy come from §0; TypeScript syntax comes from [`typescript.md`](guides://typescript.md).

- **D**efault-deny permissions: code accesses nothing (fs, net, env, subprocess, FFI) without an explicit `--allow-*` grant. Least privilege is Deno's binding of [`secure-coding.md`](guides://secure-coding.md) — never `-A`/`--allow-all`.
- **E**SM + native TypeScript: no transpile/build step; `deno check` type-checks `.ts` directly. URL/JSR/npm specifiers replace `node_modules` + `package.json`.
- **N**ative tooling, zero config: one toolchain — `deno fmt`, `deno lint`, `deno test`, `deno bench`, `deno check`, `deno doc`, `deno compile` — configured by a single optional `deno.json`.
- **O**pen web standards first: prefer Web Platform APIs (`fetch`, `Request`/`Response`, `URL`, Web Crypto, Streams, `Deno.serve`) over runtime-specific shims, so code is portable to browsers/edge.

**Verified Code**: Agent-generated Deno code MUST pass every gate in §2 before delivery.

---

## 2. Requirements (MANDATORY, auditable)

RFC-2119 keywords. IDs `DENO-<TOPIC>-<NN>`. Each row has a binary gate; rows binding a shared rule cite its owner.

| ID | Requirement | Verify | Gate |
|----|-------------|--------|------|
| DENO-TST-01 | Every feature MUST be test-first (see `tdd.md`) | `deno test` | exit 0, 0 ignored |
| DENO-TST-02 | Each bug MUST get a regression test before the fix (see `tdd.md`) | `deno test` | failing→passing |
| DENO-FMT-01 | Code MUST be formatted | `deno fmt --check` | no diff |
| DENO-LINT-01 | Linter MUST pass clean | `deno lint` | exit 0 |
| DENO-TYP-01 | All modules MUST type-check (native TS, `strict`) | `deno check .` | exit 0 |
| DENO-DOC-01 | Public APIs MUST have JSDoc (see `comments.md`) | `deno doc --lint mod.ts` | exit 0 |
| DENO-SEC-01 | Tasks/CI MUST use least-privilege flags; no `-A`/`--allow-all` (see `secure-coding.md`) | grep `deno.json` tasks + CI | none present |
| DENO-SEC-02 | 0 high/critical CVEs in deps (see `secure-coding.md`) | scan `deno.lock` per `secure-coding.md` | 0 high/critical |
| DENO-DEP-01 | `deno.lock` committed; CI installs frozen (see `secure-coding.md`) | `deno install --frozen` | no lockfile changes |
| DENO-CFG-01 | No hardcoded secrets/config; validated env (see `env-config.md`) | review / grep | no literals |

> **Forbidden**: shipping implementation before its test (violates `tdd.md`), fixing a bug without a regression test first, `--allow-all`/`-A` in committed tasks or CI, committing `.env`/secrets, importing un-pinned remote URLs without a committed `deno.lock`, or `any` to silence `deno check`.

> ⚠️ **Correction:** there is **no** `deno audit` subcommand. Supply-chain integrity comes from the **frozen lockfile** (`--frozen`) + **least-privilege permissions**; CVE scanning of `npm:`/`jsr:` deps follows [`secure-coding.md`](guides://secure-coding.md) tooling.

---

## 3. Verification Protocol

Run, in order, before presenting code. Fix → re-run until every gate is green.

```bash
deno fmt --check          # DENO-FMT-01
deno lint                 # DENO-LINT-01
deno check .              # DENO-TYP-01  (native type check — strict)
deno test                 # DENO-TST-01/02
deno doc --lint mod.ts    # DENO-DOC-01  (flags missing/!invalid JSDoc)
deno install --frozen     # DENO-DEP-01  (errors if deno.lock would change)
```

The *why* behind each gate lives in its §0 owner; do not re-derive it here.

---

## 4. Project Structure

Idiomatic Deno layout — flat, ESM, `deno.json`-driven. Architectural *principles* (layering, dependency direction) are owned by the architecture guides; this is only their Deno mapping.

```
project/
├── src/
│   ├── domain/          # pure logic — no Deno.* / IO imports
│   ├── services/        # use cases
│   └── adapters/        # http/db/kv implementations
├── tests/               # *_test.ts, mirrors src/ (see tdd.md)
├── mod.ts               # library entry (JSR `exports`)
├── main.ts              # app entry point
├── deno.json            # tasks, imports (import map), compilerOptions, lock
├── deno.lock            # committed lockfile (DENO-DEP-01)
└── README.md
```

- Test files are `*_test.ts` / `*.test.ts` and are discovered automatically by `deno test`.
- Use snake_case filenames (the `@std` convention); group by feature, not by type.

---

## 5. Deno Specifics

The unique value of this guide.

### A. Permission model — Deno's security mechanism

This is how Deno *enforces* [`secure-coding.md`](guides://secure-coding.md). Grant the **narrowest** scope, never the bare flag (a bare `--allow-net` permits *every* host).

```bash
# ✅ scoped to exactly what the program needs
deno run --allow-net=api.example.com:443 --allow-read=./data --allow-env=PORT main.ts
# ❌ never ship these
deno run -A main.ts            # all permissions
deno run --allow-net main.ts   # every host
```

| Operation | Flag (scoped) |
|-----------|---------------|
| Read file | `--allow-read=./data` |
| Write file | `--allow-write=./logs` |
| HTTP server | `--allow-net=:8000` |
| HTTP client | `--allow-net=api.example.com` |
| Env vars | `--allow-env=PORT,DATABASE_URL` |
| Subprocess | `--allow-run=git` |
| FFI (unsafe) | `--allow-ffi=./native.so` |
| System info | `--allow-sys=osRelease` |

Pin permissions per task in `deno.json` so they are reviewable and reproducible:

```json
{
  "tasks": {
    "start": "deno run --allow-net=:8000 --allow-read=./public main.ts",
    "test": "deno test --allow-read=./fixtures"
  }
}
```

Permissions are queryable/requestable at runtime and scopable per test:

```ts
const s = await Deno.permissions.query({ name: "read", path: "./data" });
if (s.state !== "granted") throw new Error("read ./data required");

Deno.test({ name: "no net", permissions: { net: false }, fn: async () => {
  await assertRejects(() => fetch("https://x"), Deno.errors.NotCapable);
}});
```

### B. `deno.json`, import maps & module specifiers

`deno.json` replaces `package.json`. The `imports` field is an **import map** — bare specifiers resolve to JSR, npm, `node:`, or local paths. Prefer **JSR** for Deno-native packages, `npm:` for the wider ecosystem.

```jsonc
{
  "imports": {
    "@std/assert": "jsr:@std/assert@1",   // Deno standard library (audited, versioned)
    "@std/http":   "jsr:@std/http@1",
    "hono":        "jsr:@hono/hono@^4",    // JSR third-party
    "zod":         "npm:zod@^3.23"         // npm via the npm: bridge
  },
  "compilerOptions": { "strict": true },   // strict is the default; keep it
  "lock": true                             // maintain deno.lock
}
```

```ts
import { assertEquals } from "@std/assert";   // via import map
import { z } from "zod";                       // npm: package, no node_modules step
```

- Pin versions in the import map; the resolved graph is locked in `deno.lock`.
- The `@std` library (`jsr:@std/*`) covers assert, http, path, fs, crypto, async, testing, cli, etc. — reach for it before adding a third-party dep.

### C. Native TypeScript

Deno executes `.ts` directly and type-checks with `deno check` (strict by default). **The language rules — typing, generics, idioms — are owned by [`typescript.md`](guides://typescript.md); do not restate them.** Deno-specific notes only:

- `deno check .` is the type gate; `deno test` and `deno run` type-check by default (skip with `--no-check` only for throwaway scripts).
- Global runtime APIs (`Deno.*`) are typed out of the box; no `@types/*` packages.

### D. Built-in tooling

One toolchain, no plugins:

```bash
deno test --coverage=cov   # test runner + coverage (deno coverage cov)
deno bench                 # benchmarks (*_bench.ts, Deno.bench)
deno fmt / deno lint       # formatter + linter (config in deno.json)
deno doc --lint mod.ts     # JSDoc completeness/validity (DENO-DOC-01)
deno compile --allow-net=:8000 -o app main.ts   # standalone self-contained binary
deno run --env-file=.env main.ts                 # native .env loading
```

### E. Web-standard APIs

Prefer the platform over runtime shims — same code runs on Deno Deploy/edge and (mostly) in browsers:

```ts
Deno.serve({ port: 8000 }, (req: Request): Response =>
  Response.json({ ok: true, path: new URL(req.url).pathname }));

const buf = await crypto.subtle.digest("SHA-256", new TextEncoder().encode("x")); // Web Crypto
```

Validate request bodies with a schema validator (see [`zod.md`](guides://zod.md)); shape errors per [`error-handling.md`](guides://error-handling.md). Built-in **Deno KV** (`Deno.openKv()`) gives an ACID key-value store with atomic transactions for simple persistence; for SQL use `node:sqlite` or an ecosystem driver.

### F. Node compatibility & contrast

Deno runs much of the Node ecosystem via `node:` builtins and `npm:` specifiers — but **without** an implicit `node_modules`/ambient permissions. Node-specific runtime rules live in [`nodejs.md`](guides://nodejs.md); the Deno bindings:

```ts
import { Buffer } from "node:buffer";
import { DatabaseSync } from "node:sqlite";   // built-in SQLite
import express from "npm:express@^4";         // npm package, still permission-gated
```

- `npm:` packages obey the same `--allow-*` model — a transitive dep cannot silently read the disk.
- Use `node:` for Node builtins explicitly; bare `"fs"`/`"path"` do not resolve.

### G. Footguns

- Bare `--allow-net` / `--allow-read` (no scope) ⇒ effectively unrestricted → always scope to host/path.
- `--allow-all` in a committed task or CI → fails DENO-SEC-01; pin per-task flags.
- Un-pinned remote URL imports without a committed `deno.lock` → non-reproducible & supply-chain risk.
- Catching `error` typed as `unknown` then accessing `.message` → narrow with `instanceof Error` (see `error-handling.md`).
- Top-level `await` is allowed in ESM — but a hanging promise blocks startup; guard long-running setup.

---

## 6. Tooling & Dependencies

Security/supply-chain *policy* → [`secure-coding.md`](guides://secure-coding.md); versioning → [`semver.md`](guides://semver.md). Deno binding:

```bash
deno add jsr:@std/http          # add JSR dep (updates deno.json + deno.lock)
deno add npm:zod                # add npm dep
deno install                    # cache/resolve all deps from lockfile
deno install --frozen           # DENO-DEP-01: error if deno.lock would change (use in CI)
deno outdated --update          # bump to latest resolvable versions
```

Commit `deno.lock`. Pin direct deps in the import map; let Deno lock the transitive graph. There is no native CVE scanner — audit deps per `secure-coding.md`.

---

## 7. Quick Reference

```bash
deno run --allow-net=:8000 main.ts    # run (scoped perms)
deno test                             # test
deno lint && deno fmt                 # lint + format
deno check .                          # type check
deno compile -o app main.ts           # standalone binary
deno doc --html mod.ts                # generate API docs
```

---

## 8. Deployment Checklist

Generated from §2 — one box per requirement ID.

- [ ] DENO-FMT-01 — `deno fmt --check` clean
- [ ] DENO-LINT-01 — `deno lint` clean
- [ ] DENO-TYP-01 — `deno check .` clean (strict)
- [ ] DENO-TST-01/02 — tests pass, 0 ignored, bugs have regression tests
- [ ] DENO-DOC-01 — `deno doc --lint` clean, public APIs have JSDoc
- [ ] DENO-SEC-01 — least-privilege flags only, no `-A`/`--allow-all`
- [ ] DENO-SEC-02 — 0 high/critical CVEs in deps
- [ ] DENO-DEP-01 — `deno.lock` committed, `deno install --frozen` clean
- [ ] DENO-CFG-01 — no hardcoded secrets, env validated
- [ ] Agent ran every §3 command and documented any fixes

---
**End of Deno Guidelines**
