# Fastify (Node.js Web Framework) Guidelines
Mandatory standards for Node.js web servers built Fastify-first: plugin-encapsulated, schema-validated, async-everywhere, gracefully shut down. Fastify 5.x, Node 22 LTS, TypeScript 5.x.

---
name: fastify
title: Fastify (Node.js Web Framework) Guidelines
version: 2.0
last_reviewed: 2026-06-05
kind: framework
tools: [fastify@5, node@22-lts, typescript@5, "@fastify/type-provider-typebox", "@sinclair/typebox", fastify-type-provider-zod, vitest@3]
requires:
  - nodejs
  - tdd
  - secure-coding
recommends:
  - typescript
  - rest
  - openapi
  - error-handling
  - observability
  - zod
provides:
  - fastify-plugins
  - fastify-encapsulation
  - fastify-hooks-lifecycle
  - schema-validation-serialization
  - fastify-error-handler
  - fastify-inject-testing
  - graceful-shutdown-http
---

> 🧭 Authored per [`CONVENTIONS.md`](guides://CONVENTIONS.md): shared concerns are referenced, not restated. This guide covers only what is unique to **building HTTP servers with Fastify**. The Node.js *runtime* (event loop, streams, ESM, process lifecycle) lives in [`nodejs.md`](guides://nodejs.md); HTTP/REST *semantics* live in [`rest.md`](guides://rest.md).

---

## 0. Prerequisites & References

Fetch and apply these **before** generating Fastify code. Their rules are assumed below and not repeated.

> 📎 **REQUIRED — fetch & apply first:**
> - [`nodejs.md`](guides://nodejs.md) — the runtime. Event loop, streams, ESM, `process` hooks, native APIs, graceful-shutdown drain are owned there; this guide only binds them to the HTTP server. **Do not restate runtime rules.**
> - [`tdd.md`](guides://tdd.md) — test-first, Red-Green-Refactor, regression-test-before-fix, coverage. *(Fastify binding: `fastify.inject()` for in-process HTTP tests, runner is `vitest`/`node --test`.)*
> - [`secure-coding.md`](guides://secure-coding.md) — supply chain, secrets, CVE policy, and web hardening (headers, rate limiting, CORS). *(Fastify binding: `@fastify/helmet`, `@fastify/rate-limit`, `@fastify/cors`.)*

> 📎 **RECOMMENDED — fetch when the task touches them:**
> - [`typescript.md`](guides://typescript.md) — `strict`, `tsconfig`, generics. *(Type providers below depend on strict TS.)*
> - [`rest.md`](guides://rest.md) — HTTP verbs, status codes, resource design, idempotency, pagination. *(Routes implement these; don't redefine them here.)*
> - [`openapi.md`](guides://openapi.md) — OpenAPI/Swagger generation. *(Fastify binding: `@fastify/swagger` derives the spec from route JSON Schemas.)*
> - [`zod.md`](guides://zod.md) — request/response validation schemas. *(Fastify binding: `fastify-type-provider-zod`.)*
> - [`error-handling.md`](guides://error-handling.md) — error strategy & taxonomy. *(Fastify binding: `setErrorHandler`.)*
> - [`observability.md`](guides://observability.md) — logs/metrics/traces. *(Fastify binding: built-in `pino` logger + OpenTelemetry instrumentation.)*

> 📎 **SEE ALSO:** [`oauth.md`](guides://oauth.md) *(auth flows behind `@fastify/jwt`/`@fastify/oauth2`)* · [`postgresql.md`](guides://postgresql.md) · [`sql.md`](guides://sql.md) *(data access lives in a datastore guide, not here)* · [`semver.md`](guides://semver.md) · [`ci-cd.md`](guides://ci-cd.md)

---

## 1. Core Philosophies: FASTIFY-FIRST

Fastify-specific principles only. Runtime concerns come from [`nodejs.md`](guides://nodejs.md); REST design from [`rest.md`](guides://rest.md); validation/security/errors from §0.

- **F**astify by default: Fastify is the recommended framework for new Node HTTP services — schema-first, fast, and structured. Express is legacy (see §7); reach for it only to maintain existing code.
- **A**ll routes schema-validated: every route declares a JSON Schema (or TypeBox/Zod via a type provider) for `body`/`querystring`/`params`/`headers` and `response`. No unvalidated input reaches a handler; response serialization is schema-driven (fast + leak-proof).
- **S**elf-contained plugins: every feature is an encapsulated plugin. Shared services attach via `decorate`; cross-cutting setup via `register`. Encapsulation is the unit of structure, isolation, and testing.
- **T**yped end-to-end: a type provider (TypeBox or Zod) makes `request.body`/`reply` fully typed from the same schema that validates them — one source of truth, no `as` casts.
- **I**diomatic async: every handler, hook, and plugin is `async`. Return the payload (or `reply.send`) — never mix callbacks with `async`. Throw to signal errors; let `setErrorHandler` shape the response.
- **F**ail-fast lifecycle: build the app in a `buildApp()` factory, `await app.ready()` in tests, bind the runtime's signal handlers (`nodejs.md`) to `app.close()` for graceful drain.
- **Y**ield to the runtime: don't block the event loop in a handler (`nodejs.md`); stream large responses; keep CPU work in workers.

**Verified Code**: Agent-generated Fastify code MUST pass every gate in §2 before delivery.

---

## 2. Requirements (MANDATORY, auditable)

RFC-2119 keywords. IDs `FST-<TOPIC>-<NN>`. Each row has a binary gate; rows binding a shared rule cite its owner.

| ID | Requirement | Verify | Gate |
|----|-------------|--------|------|
| FST-TST-01 | Every route/feature MUST be test-first via `fastify.inject()` (see `tdd.md`) | `npx vitest run` (or `node --test`) | exit 0, 0 skips |
| FST-TST-02 | Each bug MUST get a failing `inject()` regression test before the fix (see `tdd.md`) | `npx vitest run` | failing→passing |
| FST-VAL-01 | Every route MUST declare a schema for all untrusted input (`body`/`query`/`params`/`headers`) | grep routes / review; `app.ready()` validates schemas | no schemaless route |
| FST-VAL-02 | Every route MUST declare a `response` schema so serialization is schema-bound (no field leakage) | review / contract test | response schemas present |
| FST-STRUCT-01 | Features MUST be encapsulated plugins; shared services via `decorate`, not module globals | review / `fastify-plugin` usage | no global singletons |
| FST-ERR-01 | A single `setErrorHandler` MUST map errors to status + safe body; no stack/internals leak (see `error-handling.md`) | review + test 4xx/5xx bodies | handler present, no leak |
| FST-SEC-01 | Security headers, rate limiting, and CORS MUST be applied (see `secure-coding.md`) | grep `@fastify/helmet`/`rate-limit`/`cors` + test headers | plugins registered |
| FST-SEC-02 | 0 high/critical CVEs in deps (see `secure-coding.md`) | `npm audit --audit-level=high` | 0 high/critical |
| FST-SHUT-01 | Server MUST close gracefully on `SIGTERM`/`SIGINT` via `app.close()` (runtime hooks: `nodejs.md`) | integration test / signal drill | clean drain, exit 0 |
| FST-LOG-01 | MUST use Fastify's built-in `pino` logger with `requestId`; no `console.log` (see `observability.md`) | grep `console.` / review | structured logs only |
| FST-TYP-01 | TS routes MUST be strict-typed via a type provider; no `any`/`as` on request/reply (see `typescript.md`) | `tsc --noEmit` | exit 0 |
| FST-DOC-01 | Public HTTP API SHOULD expose an OpenAPI spec from route schemas (see `openapi.md`) | `@fastify/swagger` route / generated spec | spec served/emitted |

> **Forbidden**: a route handler with no input schema; building the response object by hand instead of a `response` schema; `try/catch` in every handler instead of a central `setErrorHandler`; registering shared state as a module-level singleton instead of `decorate`; calling `app.listen` without binding `app.close()` to signals; blocking the event loop in a handler (violates `nodejs.md`); shipping Express patterns in new services without justification.

---

## 3. Verification Protocol

Run, in order, before presenting code. Fix → re-run until every gate is green. Runtime-level gates (lint/format/ESM/audit/lockfile) are owned by [`nodejs.md`](guides://nodejs.md) §3 — run those too.

```bash
npx biome check .                  # format + lint (nodejs.md)
tsc --noEmit                       # FST-TYP-01  (type provider gives typed request/reply)
npx vitest run --coverage          # FST-TST-01/02  (fastify.inject in-process tests)
npm audit --audit-level=high       # FST-SEC-02
node -e "import('./dist/app.js').then(m=>m.buildApp().ready())"  # schemas/plugins compile (FST-VAL-*)
```

The *why* behind each gate lives in its §0 owner; do not re-derive it here.

---

## 4. Application Structure — the `buildApp()` factory

Architectural layering (domain/application/adapters) is owned by [`nodejs.md`](guides://nodejs.md)/`hexagonal.md`. The Fastify-specific rule: **never instantiate the server at import time.** Export a factory so tests can build, `inject`, and `close` it without binding a port.

```
src/
├── app.ts              # buildApp(): registers plugins, hooks, error handler — NO listen()
├── server.ts           # entry: buildApp().listen() + bind SIGTERM/SIGINT → app.close() (nodejs.md)
├── plugins/            # cross-cutting, app-wide (fastify-plugin wrapped): db, auth, swagger, sensible
├── routes/             # feature plugins: users/, orders/ — each encapsulated, schema-first
└── schemas/            # shared TypeBox/Zod schemas ($ref-able)
test/                   # *.test.ts — buildApp() + app.inject() (see tdd.md)
```

```ts
// app.ts
import Fastify, { type FastifyInstance } from 'fastify';

export async function buildApp(opts = {}): Promise<FastifyInstance> {
  const app = Fastify({ logger: true, ...opts });   // built-in pino (FST-LOG-01)
  await app.register(securityPlugins);              // helmet/cors/rate-limit (secure-coding.md)
  await app.register(swaggerPlugin);                // OpenAPI from schemas (openapi.md)
  await app.register(usersRoutes, { prefix: '/api/users' });
  app.setErrorHandler(errorHandler);                // FST-ERR-01
  return app;
}
```

```ts
// server.ts — the ONLY place that listens; signal handling owned by nodejs.md
const app = await buildApp();
await app.listen({ port: env.PORT, host: '0.0.0.0' });
for (const sig of ['SIGTERM', 'SIGINT'] as const)
  process.on(sig, () => app.close().then(() => process.exit(0)));
```

---

## 5. Fastify Specifics — the unique value of this guide

### A. Plugins & encapsulation
A Fastify plugin is `async (app, opts) => { … }`. **Encapsulation** is the core model: anything a plugin `register`s, `decorate`s, or hooks into is visible only to that plugin and its children — not to siblings or parents. This gives natural isolation and scoping.

- To **share** a decorator/hook with the whole app, wrap the plugin with `fastify-plugin` (`fp`) to break encapsulation deliberately.
- Use `prefix` on route plugins to scope URLs; compose plugins instead of one giant file.

```ts
import fp from 'fastify-plugin';

// Shared across the app → fp(): decorator visible everywhere
export default fp(async (app) => {
  app.decorate('db', await connect(env.DATABASE_URL));   // app.db is now typed & global
  app.addHook('onClose', async (a) => a.db.end());       // released on app.close() (FST-SHUT-01)
});

// Feature plugin → NOT fp(): encapsulated, only its own routes see local hooks
export async function usersRoutes(app: FastifyInstance) {
  app.addHook('preHandler', requireAuth);    // applies only inside this plugin
  app.get('/:id', { schema: getUserSchema }, async (req) => app.db.users.find(req.params.id));
}
```

### B. Decorators
Attach reusable services/values to `app`, `request`, or `reply` once — typed, no per-request allocation.
```ts
app.decorate('config', env);                          // app.config
app.decorateRequest('user', null);                    // declare shape; set in a hook
app.addHook('preHandler', async (req) => { req.user = await authenticate(req); });
```
Declare the decorator's type via module augmentation (`declare module 'fastify' { interface FastifyInstance { db: Db } }`) so `app.db`/`request.user` are strictly typed (FST-TYP-01).

### C. Hooks & the request lifecycle
Hooks run at defined lifecycle points; use them instead of Express-style middleware chains. Order per request:

`onRequest` → `preParsing` → `preValidation` → `preHandler` → `preSerialization` → `onSend` → `onResponse`
(plus app-level `onReady`, `onClose`, and `onError`).

- **`onRequest`** — auth/rate-limit gates (no body yet).
- **`preHandler`** — authorization, load-resource, per-route guards.
- **`preSerialization`/`onSend`** — shape/redact the outgoing payload.
- **`onError`** — observe errors (the *response* is shaped by `setErrorHandler`, see §D).
Hooks are encapsulated (§A): a hook added inside a plugin applies only within it.

### D. Schema-based validation & serialization
This is Fastify's defining feature and a hard requirement (FST-VAL-01/02). One schema both **validates input** (rejects bad requests with 400 automatically) and **serializes output** (fast `fast-json-stringify`, and only declared fields are emitted — preventing accidental leakage of internal fields).

Pick **one** type provider so schemas also produce static types:

```ts
// TypeBox provider — JSON-Schema-native, zero runtime coercion overhead
import { Type } from '@sinclair/typebox';
import type { TypeBoxTypeProvider } from '@fastify/type-provider-typebox';

const app = Fastify().withTypeProvider<TypeBoxTypeProvider>();

app.post('/api/users', {
  schema: {
    body: Type.Object({
      email: Type.String({ format: 'email' }),
      name: Type.String({ minLength: 1, maxLength: 100 }),
    }),
    response: {
      201: Type.Object({ id: Type.String({ format: 'uuid' }), email: Type.String() }),
    },
  },
}, async (req, reply) => {
  // req.body is typed { email: string; name: string } — validated before we get here
  const user = await createUser(req.body);
  return reply.code(201).send(user);   // serialized against the 201 schema
});
```

```ts
// Zod provider — when you already standardize on Zod (zod.md owns schema design)
import { serializerCompiler, validatorCompiler, type ZodTypeProvider } from 'fastify-type-provider-zod';
app.setValidatorCompiler(validatorCompiler);
app.setSerializerCompiler(serializerCompiler);
const typed = app.withTypeProvider<ZodTypeProvider>();
typed.post('/api/users', { schema: { body: CreateUserSchema, response: { 201: UserSchema } } }, handler);
```

*Schema authoring* (refinements, transforms, error messages) is owned by [`zod.md`](guides://zod.md); *REST status/verb choices* by [`rest.md`](guides://rest.md). Register shared schemas with `app.addSchema()` and `$ref` them to avoid duplication.

### E. Error handling — one `setErrorHandler`
Strategy/taxonomy is owned by [`error-handling.md`](guides://error-handling.md). The Fastify binding: a single, app-wide handler maps thrown errors to status + safe body. Handlers **throw**; they do not build error responses inline.
```ts
app.setErrorHandler((err, req, reply) => {
  if (err.validation) return reply.code(400).send({ error: 'ValidationError', details: err.validation });
  const status = (err as AppError).statusCode ?? 500;
  req.log.error({ err }, 'request failed');                 // log full detail server-side
  reply.code(status).send({ error: err.name, message: status < 500 ? err.message : 'Internal Server Error' });
});                                                          // never leak stack/internals (FST-ERR-01)
app.setNotFoundHandler((req, reply) => reply.code(404).send({ error: 'NotFound' }));
```
`@fastify/sensible` adds `httpErrors` helpers (`throw app.httpErrors.notFound()`); validation errors are produced automatically by the schemas in §D.

### F. Plugin ecosystem (scoped `@fastify/*`)
Prefer the maintained first-party scoped plugins over hand-rolled middleware:
| Need | Plugin | Owner of the *policy* |
|------|--------|----------------------|
| Security headers | `@fastify/helmet` | `secure-coding.md` |
| CORS | `@fastify/cors` | `secure-coding.md` |
| Rate limiting | `@fastify/rate-limit` | `secure-coding.md` |
| OpenAPI/Swagger UI | `@fastify/swagger` (+ `@fastify/swagger-ui`) | `openapi.md` |
| JWT / OAuth2 | `@fastify/jwt` · `@fastify/oauth2` | `oauth.md` |
| Cookies / sessions | `@fastify/cookie` · `@fastify/session` | `secure-coding.md` |
| Multipart uploads | `@fastify/multipart` | — |
| Static files | `@fastify/static` | — |
| Utility errors | `@fastify/sensible` | `error-handling.md` |

### G. Security binding
Web-hardening *policy* (which headers, CORS rules, limits, input sanitization) is owned by [`secure-coding.md`](guides://secure-coding.md). Fastify binding (FST-SEC-01):
```ts
await app.register(import('@fastify/helmet'));
await app.register(import('@fastify/cors'), { origin: env.ALLOWED_ORIGINS });
await app.register(import('@fastify/rate-limit'), { max: 100, timeWindow: '1 minute' });
```
Input validation (§D) is your primary injection defense; parameterized queries belong to the datastore layer (`postgresql.md`/`sql.md`), not the handler.

### H. OpenAPI generation
Don't hand-write a spec. `@fastify/swagger` derives OpenAPI directly from the same route JSON Schemas used for validation — the spec can't drift from the implementation. Spec/contract *policy* is owned by [`openapi.md`](guides://openapi.md).
```ts
await app.register(import('@fastify/swagger'), { openapi: { info: { title: 'API', version: '1.0.0' } } });
await app.register(import('@fastify/swagger-ui'), { routePath: '/docs' });
```

### I. Performance
Fastify is fast by design; keep it that way: rely on schema serialization (don't `JSON.stringify` manually), reuse decorators instead of per-request allocation, set `keepAliveTimeout`, and stream large payloads (`nodejs.md`). General performance budgets/profiling are owned by [`nodejs.md`](guides://nodejs.md) and `performance.md`.

### J. Data access — out of scope here
Persistence is **not** a web-framework concern. Wire your repository/datastore as an encapsulated plugin (`app.decorate('db', …)`, closed in an `onClose` hook), but keep the actual queries, migrations, and ORM/driver choice in the datastore guide: [`postgresql.md`](guides://postgresql.md) / [`sql.md`](guides://sql.md). The domain talks to a port; the Fastify plugin is just the adapter wiring (`hexagonal.md` via `nodejs.md`). Do not embed ORM modeling in route handlers.

---

## 6. Testing — `fastify.inject()`
Test-first policy and coverage discipline are owned by [`tdd.md`](guides://tdd.md). Fastify binding: `app.inject()` dispatches a simulated HTTP request **in-process** (no port, no network) — fast, deterministic, and the canonical way to test routes, hooks, schemas, and the error handler.
```ts
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { buildApp } from '../src/app.js';

test('POST /api/users rejects an invalid email (schema 400)', async () => {
  const app = await buildApp();
  await app.ready();
  const res = await app.inject({ method: 'POST', url: '/api/users', payload: { email: 'nope', name: 'A' } });
  assert.equal(res.statusCode, 400);                 // validation schema enforced (FST-VAL-01)
  await app.close();
});
```
- Build a fresh app per test (or per suite) and `await app.close()` to release decorated resources.
- Unit-test domain logic with port fakes (no `inject`); use `inject` for the HTTP contract; real-datastore integration via Testcontainers; full flows → [`e2e-testing.md`](guides://e2e-testing.md).
- A bug fix starts with a failing `inject` regression test naming the issue (FST-TST-02, see `tdd.md`).

---

## 7. Migrating from Express / Express-legacy
Express is the older, minimalist framework. It still dominates existing codebases, but for **new** Node services Fastify is recommended. Key differences when porting:

| Express | Fastify | Why Fastify |
|---------|---------|-------------|
| `app.use(mw)` middleware chain | Lifecycle **hooks** + encapsulated plugins | Scoped, ordered, testable; no implicit global chain |
| Manual `req.body` parsing + ad-hoc validation middleware | Declarative **schemas** per route | Validation + typed input + fast serialization from one source |
| `res.json(obj)` (serializes everything) | `response` schema serialization | Only declared fields emitted — no accidental leakage |
| `try/catch` + `next(err)` in every handler; `asyncHandler` wrappers needed | `async` handlers that **throw** + one `setErrorHandler` | Async errors are caught natively; no wrapper boilerplate |
| `console`/`morgan` logging bolted on | Built-in `pino` with `requestId` | Structured logging by default (FST-LOG-01) |
| Plugin = npm middleware grab-bag | First-party `@fastify/*` scoped plugins | Maintained, encapsulation-aware |

Migration path: run Express inside Fastify via `@fastify/express` (compatibility shim) to port route-by-route, then replace middleware with hooks and add schemas. Note Express 5 modernized async error propagation, but it still lacks schema-based validation/serialization and encapsulation. If maintaining Express, at minimum: validate every input (e.g. Zod middleware, see `zod.md`), centralize errors in one error-handling middleware (`error-handling.md`), and add `helmet`/`cors`/rate-limit (`secure-coding.md`).

---

## 8. Quick Reference
```bash
npm ci                                   # setup (frozen — nodejs.md)
npm run dev                              # tsx watch src/server.ts
npx vitest run --coverage                # test (fastify.inject)
tsc --noEmit                             # type check (type provider)
npm audit --audit-level=high            # CVE scan
node --env-file=.env dist/server.js      # run (native env file — nodejs.md)
```

---

## 9. Deployment Checklist
Generated from §2 — one box per requirement ID.

- [ ] FST-TST-01/02 — routes are test-first via `inject`; bugs have failing-first regression tests
- [ ] FST-VAL-01 — every route validates `body`/`query`/`params`/`headers` with a schema
- [ ] FST-VAL-02 — every route has a `response` schema (serialization is schema-bound)
- [ ] FST-STRUCT-01 — features are encapsulated plugins; shared services via `decorate`
- [ ] FST-ERR-01 — single `setErrorHandler`; no stack/internals leaked
- [ ] FST-SEC-01 — helmet + CORS + rate-limit registered
- [ ] FST-SEC-02 — `npm audit` 0 high/critical CVEs
- [ ] FST-SHUT-01 — `app.close()` bound to SIGTERM/SIGINT; resources drained
- [ ] FST-LOG-01 — built-in pino with `requestId`; no `console.log`
- [ ] FST-TYP-01 — `tsc --noEmit` clean; typed via a type provider, no `any`/`as`
- [ ] FST-DOC-01 — OpenAPI spec served/emitted from route schemas
- [ ] Agent ran every §3 command (plus nodejs.md §3) and documented any fixes

---
**End of Fastify Guidelines**
