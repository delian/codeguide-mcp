# Node.js Runtime Guidelines
Mandatory standards for the Node.js runtime: event-loop-aware, async-first, ESM-native, gracefully shut down. Node.js 24 LTS (22 LTS floor), npm/pnpm, node:test/vitest, ESM.

---
name: nodejs
title: Node.js Runtime Guidelines
version: 2.0
last_reviewed: 2026-06-05
kind: language
tools: [node@24-lts, node@22-lts, npm@10, pnpm@9, node:test, vitest@3, tsx, biome]
requires:
  - tdd
  - secure-coding
  - error-handling
recommends:
  - typescript
  - javascript
  - hexagonal
  - microservices
  - observability
  - logging
  - comments
provides:
  - node-runtime
  - event-loop
  - streams
  - npm-esm
  - graceful-shutdown
---

> 🧭 Authored per [`CONVENTIONS.md`](guides://CONVENTIONS.md): shared concerns are referenced, not restated. This guide covers only what is unique to the **Node.js runtime** — the *language* (syntax, types, async semantics) lives in [`typescript.md`](guides://typescript.md) / [`javascript.md`](guides://javascript.md).

---

## 0. Prerequisites & References

Fetch and apply these **before** generating Node.js code. Their rules are assumed below and not repeated.

> 📎 **REQUIRED — fetch & apply first:**
> - [`tdd.md`](guides://tdd.md) — test-first, Red-Green-Refactor, regression-test-before-fix, coverage. *(Node binding: runner is `node --test` or `vitest`.)*
> - [`secure-coding.md`](guides://secure-coding.md) — supply chain, secrets, CVE policy. *(Node binding: `npm audit`, lockfile + `npm ci`.)*
> - [`error-handling.md`](guides://error-handling.md) — error strategy & propagation. *(Node binding: `unhandledRejection`/`uncaughtException` process hooks.)*

> 📎 **RECOMMENDED — fetch when the task touches them:**
> - [`typescript.md`](guides://typescript.md) — types, `strict`, `tsconfig`, JSDoc. *(All language-level typing rules live here, not here.)*
> - [`javascript.md`](guides://javascript.md) — async/await, iterators, modules, language idioms.
> - [`hexagonal.md`](guides://hexagonal.md) · [`microservices.md`](guides://microservices.md) — layering, service boundaries.
> - [`logging.md`](guides://logging.md) · [`observability.md`](guides://observability.md) — structured logs/metrics/traces. *(Node binding: `pino` + OpenTelemetry SDK.)*
> - [`comments.md`](guides://comments.md) — doc policy *(binding: JSDoc + TypeDoc)*

> 📎 **SEE ALSO:** [`fastify.md`](guides://fastify.md) *(Node web framework, builds on this guide)* · [`parallelism.md`](guides://parallelism.md) *(worker_threads/cluster)* · [`env-config.md`](guides://env-config.md) · [`performance.md`](guides://performance.md) · [`rest.md`](guides://rest.md) · [`pre-commit.md`](guides://pre-commit.md) · [`ci-cd.md`](guides://ci-cd.md)

---

## 1. Core Philosophies: NODE-FIRST

Node.js runtime principles only. TDD, security, error strategy, and architecture come from §0; language/type rules come from `typescript.md`/`javascript.md`.

- **N**on-blocking: never block the single-threaded event loop — no sync FS/crypto/`JSON.parse` of huge payloads on the request path; offload CPU work to `worker_threads`.
- **O**ne module system — ESM: `"type": "module"`, `node:` protocol for built-ins, explicit `.js` extensions, top-level `await`. No `require`/CJS in new code.
- **D**eps minimal & native-first: prefer built-ins (`fetch`, `node:test`, `node:sqlite`, Web Crypto, streams) over packages; every dependency is a supply-chain liability (`secure-coding.md`).
- **E**vent-loop & stream literacy: understand microtask vs. macrotask ordering; stream large data with backpressure rather than buffering it all in memory.
- **Async-first & resilient**: async/await over callbacks; bind `unhandledRejection`/`uncaughtException`; shut down gracefully on `SIGTERM`/`SIGINT`.

**Verified Code**: Agent-generated Node.js MUST pass every gate in §2 before delivery.

---

## 2. Requirements (MANDATORY, auditable)

RFC-2119 keywords. IDs `NODE-<TOPIC>-<NN>`. Each row has a binary gate; rows binding a shared rule cite its owner.

| ID | Requirement | Verify | Gate |
|----|-------------|--------|------|
| NODE-TST-01 | Every feature MUST be test-first (see `tdd.md`) | `node --test` or `npx vitest run` | exit 0, 0 skips |
| NODE-TST-02 | Each bug MUST get a regression test before the fix (see `tdd.md`) | `npx vitest run` | failing→passing |
| NODE-TST-03 | Business-logic coverage MUST meet the project gate | `npx vitest run --coverage` (or `node --test --experimental-test-coverage`) | ≥ threshold |
| NODE-ESM-01 | Package MUST be ESM; built-ins imported via `node:`; relative imports carry `.js` | `node --check` + review / lint | ESM only, no bare CJS |
| NODE-FMT-01 | Code MUST be formatted | `npx biome format .` (or `prettier --check`) | no diff |
| NODE-LINT-01 | Linter MUST pass clean | `npx biome check .` (or `eslint .`) | exit 0 |
| NODE-TYP-01 | TS sources MUST type-check strict (see `typescript.md`) | `tsc --noEmit` | exit 0 |
| NODE-ERR-01 | Process MUST bind `unhandledRejection` + `uncaughtException` and exit non-zero on fatal (see `error-handling.md`) | review / grep `process.on` | handlers present |
| NODE-RUN-01 | MUST NOT block the event loop; CPU-bound work offloaded (see `parallelism.md`) | review / `--prof` / event-loop-delay metric | no sync hotspots |
| NODE-SHUT-01 | MUST handle `SIGTERM`/`SIGINT`: stop intake, drain, close resources, exit | review / integration test | clean drain, exit 0 |
| NODE-SEC-01 | 0 high/critical CVEs in deps (see `secure-coding.md`) | `npm audit --audit-level=high` | 0 high/critical |
| NODE-DEP-01 | Lockfile committed & installs are frozen | `npm ci` (or `pnpm i --frozen-lockfile`) | resolves, no drift |
| NODE-DOC-01 | Public exports documented with JSDoc (see `comments.md`) | `typedoc --validation.notDocumented` | 0 undocumented |
| NODE-CFG-01 | No hardcoded config/secrets; read from env, validated at startup | review / grep | no literals |

> **Forbidden**: blocking the event loop with sync I/O/CPU on the hot path; swallowing a rejected promise; shipping CommonJS in new code; `npm install` (unpinned) in CI instead of `npm ci`; committing without a regression test for a fixed bug (violates `tdd.md`); leaving `unhandledRejection` unbound.

---

## 3. Verification Protocol

Run, in order, before presenting code. Fix → re-run until every gate is green.

```bash
npx biome check .                        # NODE-FMT-01 / NODE-LINT-01
tsc --noEmit                             # NODE-TYP-01  (TS projects; lint does NOT type-check)
npx vitest run --coverage                # NODE-TST-01/03  (or: node --test)
npm audit --audit-level=high             # NODE-SEC-01
npm ci                                   # NODE-DEP-01  (frozen install from lockfile)
typedoc --validation.notDocumented true --emit none   # NODE-DOC-01
```

The *why* behind each gate lives in its §0 owner; do not re-derive it here.

---

## 4. Project Structure

Idiomatic `src/` layout. Architectural principles (dependency direction, ports/adapters, service boundaries) are owned by [`hexagonal.md`](guides://hexagonal.md) / [`microservices.md`](guides://microservices.md); below is only their Node mapping.

```
project/
├── src/
│   ├── domain/          # pure logic — no node:/IO imports (NODE-RUN-01 keeps it sync-safe)
│   ├── application/     # use cases, orchestrates ports
│   ├── adapters/        # http/db/queue/cli — the only place node: built-ins + deps live
│   └── index.ts         # entry: wire deps, bind process hooks, start, register shutdown
├── test/                # mirrors src/ (see tdd.md); *.test.ts
├── package.json         # "type": "module", "engines", "exports", scripts
├── tsconfig.json        # strict (see typescript.md)
├── biome.json           # lint + format
└── package-lock.json    # committed lockfile (NODE-DEP-01)
```

- Group by feature/domain, not by technical type.
- Keep `node:` built-ins and third-party deps at the adapter edge so the domain stays portable and unit-testable without I/O.

---

## 5. Node.js Runtime Specifics

The unique value of this guide. Language syntax/types are in `typescript.md`/`javascript.md`; these are runtime concerns.

### A. The event loop — never block it
Node runs JS on **one thread**; the libuv event loop interleaves I/O. A synchronous CPU burst or sync syscall stalls *every* connection. Know the ordering: each loop tick drains all **microtasks** (`Promise` callbacks, `queueMicrotask`) before the next **macrotask** (timers, I/O, `setImmediate`); `process.nextTick` runs before other microtasks (use sparingly — it can starve I/O).

```js
// Macrotask phases: timers → pending → poll(I/O) → check(setImmediate) → close
setTimeout(() => console.log('timeout'), 0);
setImmediate(() => console.log('immediate'));   // fires after poll, vs timer's min-delay
Promise.resolve().then(() => console.log('microtask'));   // before either macrotask
process.nextTick(() => console.log('nextTick'));          // before other microtasks
```

Rules:
- Prefer async built-ins: `node:fs/promises`, not `fs.readFileSync`, on a live server.
- Never `JSON.parse`/`JSON.stringify` or hash/compress large payloads inline — stream it or move it to a worker.
- Watch event-loop lag with `perf_hooks.monitorEventLoopDelay()`; alert when p99 climbs.

### B. ESM & `package.json`
ESM is the runtime contract, not just syntax.
```jsonc
// package.json
{
  "type": "module",
  "engines": { "node": ">=22" },
  "exports": { ".": "./dist/index.js", "./adapters": "./dist/adapters/index.js" },
  "imports": { "#config": "./dist/config.js" }   // internal subpath aliases
}
```
```js
import { readFile } from 'node:fs/promises';   // built-ins MUST use the node: protocol
import { UserService } from './user.service.js';   // relative imports keep the .js extension
const cfg = await loadConfig();                // top-level await is allowed in ESM
```
- `exports` is the public API surface — anything not listed is unreachable to consumers (encapsulation).
- ESM has no `__dirname`: use `import.meta.dirname` (Node 20.11+) or `fileURLToPath(import.meta.url)`.
- Interop: `import` a CJS package's default; you cannot named-import a CJS module's bindings reliably. Avoid `require` in new code; use `module.createRequire` only at a forced boundary.
- Run/develop TypeScript directly with `tsx` (`tsx watch src/index.ts`) or Node's native type-stripping (`node --experimental-strip-types`, stabilizing in 24).

### C. Streams & backpressure
Streams move large data with bounded memory. **Always honor backpressure** — use `pipeline`, which propagates errors and respects the consumer's pace; never manual `.pipe()` chains (they leak on error) or `data` handlers that ignore `.write()` returning `false`.
```js
import { pipeline } from 'node:stream/promises';
import { createReadStream, createWriteStream } from 'node:fs';
import { createGzip } from 'node:zlib';

await pipeline(
  createReadStream('big.csv'),
  createGzip(),
  createWriteStream('big.csv.gz'),
);   // errors reject; streams auto-destroy on failure
```
- Use async generators as ergonomic transforms; `Readable.from(asyncIterable)` bridges them into the stream world.
- Web Streams (`ReadableStream`) interop with `fetch` bodies; convert via `stream.Readable.fromWeb`/`toWeb`.

### D. EventEmitter
The native pub/sub primitive behind streams, servers, and process. Emitters are **synchronous** (listeners run in registration order on `emit`) and unbounded.
```js
import { EventEmitter, once } from 'node:events';
const bus = new EventEmitter();
bus.setMaxListeners(20);                 // guard the leak warning deliberately
bus.on('error', (e) => log.error(e));    // an 'error' event with no listener CRASHES the process
const [payload] = await once(bus, 'ready');   // await an event without callback nesting
```
- Always attach an `'error'` listener to long-lived emitters.
- Remove listeners (`off`/`removeListener`) on teardown to avoid leaks; an ever-growing listener count is the classic Node memory leak.

### E. Concurrency: worker_threads & cluster
Policy is owned by [`parallelism.md`](guides://parallelism.md). Node binding:
- **`worker_threads`** for CPU-bound work (parsing, crypto, image/PDF processing) — keeps the main loop responsive. Share memory via `SharedArrayBuffer`/`MessageChannel`; pool workers (e.g. `piscina`) rather than spawning per task.
- **`cluster`** / running N processes behind a load balancer for scaling stateless I/O-bound services across cores. Modern deployments often prefer one process per container + an orchestrator over in-app `cluster`.
- I/O concurrency needs no threads — that's what the event loop is for; bound it with `Promise.all` over batches, not unbounded fan-out.

### F. Process lifecycle: fatal errors & graceful shutdown
Error *strategy* is owned by [`error-handling.md`](guides://error-handling.md); the runtime binding is the process hooks. An unhandled rejection or uncaught exception leaves the process in an undefined state — **log and exit non-zero**, let the orchestrator restart.
```js
process.on('unhandledRejection', (reason) => { log.fatal({ reason }, 'unhandledRejection'); shutdown(1); });
process.on('uncaughtException', (err) => { log.fatal({ err }, 'uncaughtException'); shutdown(1); });

let shuttingDown = false;
async function shutdown(code = 0) {
  if (shuttingDown) return; shuttingDown = true;
  server.close();                       // 1. stop accepting new work
  await drainInFlight();                // 2. let in-flight requests finish (with a timeout)
  await Promise.allSettled([db.end(), queue.close()]);   // 3. release resources
  process.exit(code);                   // 4. exit; force-kill after a deadline if drain hangs
}
for (const sig of ['SIGTERM', 'SIGINT']) process.on(sig, () => shutdown(0));
```
Never resume normal operation after `uncaughtException`. Expose `/health` (liveness) and `/ready` (readiness flips false at shutdown start so the LB drains you).

### G. Native runtime APIs — prefer over dependencies
```js
// Native fetch (18+) — drop axios/node-fetch
const res = await fetch(url, { signal: AbortSignal.timeout(5000) });

// Native test runner (20+) — zero-dependency
import { test } from 'node:test'; import assert from 'node:assert/strict';

// Native env file (20.6+) — drop dotenv at runtime
// node --env-file=.env src/index.js

// Native SQLite (22.5+, stabilizing) — embedded store without a driver
import { DatabaseSync } from 'node:sqlite';

// Web Crypto — hashing/HMAC/random without 'crypto-js'
const digest = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(data));
```
Every avoided dependency is one fewer CVE surface (NODE-SEC-01) and faster cold starts.

### H. Configuration & secrets — env binding
Layering/secret policy is owned by [`env-config.md`](guides://env-config.md). Node binding: read from `process.env` (loaded via `--env-file` or the platform), **validate and coerce once at startup**, fail fast, then pass a typed config object inward — never read `process.env` deep in the code.
```js
import { z } from 'zod';
const env = z.object({
  NODE_ENV: z.enum(['development', 'production', 'test']),
  PORT: z.coerce.number().int().positive().default(3000),
  DATABASE_URL: z.string().url(),
}).parse(process.env);   // throws on boot if misconfigured (NODE-CFG-01)
```
Secrets come from the environment/secret manager, never the repo (`secure-coding.md`).

---

## 6. Testing — node:test & Vitest

Test-first policy and coverage discipline are owned by [`tdd.md`](guides://tdd.md). Node bindings:

- **`node --test`** — zero-dependency, ideal for libraries and runtime/integration tests close to production. Uses `node:test` + `node:assert/strict`; built-in mocking via `mock`, coverage via `--experimental-test-coverage`, watch via `--watch`.
- **Vitest 3** — fast ESM-native runner for app code; richer mocking (`vi.fn`/`vi.mock`), snapshots, `--coverage` (v8), and `expectTypeOf` for type-level tests. Use it where you want a Jest-like DX.

```js
import { test, mock } from 'node:test';
import assert from 'node:assert/strict';

test('createUser rejects duplicate email', async () => {
  const repo = { findByEmail: mock.fn(async () => ({ id: '1' })) };
  const svc = new UserService(repo);
  await assert.rejects(() => svc.createUser({ email: 'dup@x.io' }), /exists/);
});
```
- Unit-test the domain with fakes/mocks at the port (no real I/O); integration-test adapters against a real datastore (e.g. Testcontainers). End-to-end flows: see [`e2e-testing.md`](guides://e2e-testing.md).
- A bug fix starts with a failing regression test naming the issue (NODE-TST-02, see `tdd.md`).

---

## 7. Tooling & Dependencies

Security/supply-chain *policy* → [`secure-coding.md`](guides://secure-coding.md); versioning → [`semver.md`](guides://semver.md). Node binding:

```bash
npm ci                         # NODE-DEP-01: reproducible, frozen install from package-lock.json
npm install <pkg>              # add a dep (updates lockfile) — commit the lockfile
npm outdated / npm update      # surface & apply in-range updates
npm audit --audit-level=high   # NODE-SEC-01: CVE scan (pnpm: pnpm audit)
npm pkg set scripts.x=...      # edit package.json scripts programmatically
```
- Pin `engines.node`; develop and CI on the same LTS (24, floor 22). Run `corepack enable` to pin the package-manager version.
- `pnpm` is a valid alternative (strict node_modules, fast, disk-efficient) — use `pnpm i --frozen-lockfile` in CI. Do not mix lockfiles. Avoid Yarn v1 (unmaintained).
- Vet new deps: prefer maintained, typed, small-footprint packages; check install scripts. Lockfile + `npm ci` is the supply-chain gate.

---

## 8. Quick Reference

```bash
npm ci                                  # setup (frozen)
npm run dev                             # tsx watch src/index.ts
node --test            # or: npx vitest run --coverage   # test
npx biome check .                       # lint + format
tsc --noEmit                            # type check (TS)
npm audit --audit-level=high            # CVE scan
node --env-file=.env dist/index.js      # run (native env file)
```

---

## 9. Deployment Checklist

Generated from §2 — one box per requirement ID.

- [ ] NODE-FMT-01 / NODE-LINT-01 — Biome (or ESLint/Prettier) clean
- [ ] NODE-TYP-01 — `tsc --noEmit` clean (real type checker, not the linter)
- [ ] NODE-TST-01/02/03 — tests pass, bugs have regression tests, coverage ≥ gate
- [ ] NODE-ESM-01 — ESM only, `node:` built-ins, `.js` import extensions
- [ ] NODE-ERR-01 — `unhandledRejection` + `uncaughtException` bound, exit non-zero on fatal
- [ ] NODE-RUN-01 — event loop not blocked; CPU work offloaded to workers
- [ ] NODE-SHUT-01 — graceful shutdown drains and closes resources on SIGTERM/SIGINT
- [ ] NODE-SEC-01 — `npm audit` 0 high/critical CVEs
- [ ] NODE-DEP-01 — lockfile committed, `npm ci`/`--frozen-lockfile` resolves
- [ ] NODE-DOC-01 — public exports documented (TypeDoc validates)
- [ ] NODE-CFG-01 — no hardcoded config/secrets; env validated at startup
- [ ] Agent ran every §3 command and documented any fixes

---
**End of Node.js Runtime Guidelines**
