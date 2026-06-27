# Zod Development Guidelines
Mandatory standards for Zod schema validation in TypeScript: schemas as the single source of truth, inferred types, and parse-at-the-boundary. Zod 4.x (3.x compatible), TypeScript 5.x.

---
name: zod
title: Zod Development Guidelines
version: 2.0
last_reviewed: 2026-06-05
kind: cross-cutting
tools: [zod@4, typescript@5]
requires: []
recommends:
  - typescript
  - rest
  - openapi
  - error-handling
provides:
  - zod-schemas
  - runtime-validation
  - type-inference
  - boundary-validation
---

> 🧭 Authored per [`CONVENTIONS.md`](guides://CONVENTIONS.md): shared concerns are referenced, not restated. This guide owns Zod schema definition, parsing, inference, composition, refinement/transform, and boundary validation — nothing else.

---

## 0. Prerequisites & References

Zod is a TypeScript runtime-validation library. It has no hard prerequisites, but its value depends on the type system and the boundaries it guards.

> 📎 **RECOMMENDED — fetch when the task touches them:**
> - [`typescript.md`](guides://typescript.md) — the type system Zod infers into. Zod owns the *schema*; TypeScript owns generics, `strict` mode, `unknown` vs `any`, and `tsconfig`.
> - [`error-handling.md`](guides://error-handling.md) — what to *do* with a validation failure (error taxonomy, propagation, mapping to responses). Zod owns producing the `ZodError`; it does not own your app's error strategy.
> - [`rest.md`](guides://rest.md) · [`openapi.md`](guides://openapi.md) — API-boundary contracts. Zod validates request/response payloads and can emit JSON Schema for an OpenAPI document; the wire contract itself is owned there.

> 📎 **SEE ALSO:** [`nodejs.md`](guides://nodejs.md) · [`nextjs.md`](guides://nextjs.md) · [`reactjs.md`](guides://reactjs.md) · [`env-config.md`](guides://env-config.md) — common places Zod parses input (forms, env, route handlers).

---

## 1. Core Philosophies: VALID

Zod-specific principles only. Test strategy, error-response strategy, and the type system come from §0.

- **V**alidate at the boundary: every datum entering the program from outside (HTTP body/query/params, env, files, message queues, LLM output, `localStorage`, `JSON.parse`) MUST be parsed by a schema. Inside the boundary, data is trusted and typed.
- **A**utomate types: the schema is the single source of truth. Derive the type with `z.infer` — never hand-write a parallel `interface`/`type` for the same shape.
- **L**ayer schemas: compose large schemas from small named primitives via `.extend()`, `.pick()`, `.omit()`, `.partial()`, and unions. No copy-pasted field lists.
- **I**nfer, don't assert: `unknown` in, typed out. Never use `as` to launder unvalidated data into a typed shape — that is the exact bug Zod exists to prevent.
- **D**ecide parse vs safeParse deliberately: `.parse()` to fail fast on programmer-trusted invariants; `.safeParse()` at user/network boundaries where a failure is an expected, handleable outcome.

**Verified Code**: Agent-generated Zod code MUST pass every gate in §2 before delivery.

---

## 2. Requirements (MANDATORY, auditable)

RFC-2119 keywords. IDs `ZOD-<TOPIC>-<NN>`. Each row has a binary gate; rows binding a shared rule cite its owner.

| ID | Requirement | Verify | Gate |
|----|-------------|--------|------|
| ZOD-TYP-01 | Types MUST be inferred from schemas (`z.infer`), never duplicated as a parallel `interface`/`type` (see `typescript.md`) | review / grep for hand-written shapes | no duplicate shape |
| ZOD-TYP-02 | Schema-derived code MUST compile under `strict` (see `typescript.md`) | `tsc --noEmit` | exit 0 |
| ZOD-BND-01 | All external input MUST be parsed by a schema at the boundary | review of every boundary (route/env/file/IPC) | no unparsed external data |
| ZOD-BND-02 | `unknown` MUST be the input type at boundaries; `as` MUST NOT be used to bypass parsing | grep `as ` near boundaries | none |
| ZOD-ERR-01 | `.safeParse()` failures MUST be handled, not swallowed; mapping follows the app's error strategy (see `error-handling.md`) | review / test error path | failure surfaced |
| ZOD-PRS-01 | User/network boundaries MUST use `.safeParse()` (handleable) not bare `.parse()` in a try/catch as control flow | review | safeParse at edges |
| ZOD-STR-01 | Schemas parsing untrusted input MUST reject unknown keys (`.strict()`); `.passthrough()` MUST NOT be used on untrusted input | grep `.passthrough()` | none on untrusted |
| ZOD-ANY-01 | `z.any()`/`z.unknown()` MUST NOT remain in a schema without a subsequent refinement narrowing it | grep `z.any()` / `z.unknown()` | none unrefined |
| ZOD-TST-01 | Every schema MUST be tested for both a valid and an invalid input (see `tdd.md`) | test runner | exit 0 |

> **Forbidden**: laundering unvalidated data with `as`; keeping a hand-written type beside its schema; `.passthrough()` on untrusted input; catching and silently discarding a `ZodError`; using `z.any()` to paper over a type mismatch.

---

## 3. Verification Protocol

Run before presenting code. Fix → re-run until green. (Concrete runner/linter come from the project's language guide, e.g. [`typescript.md`](guides://typescript.md) / [`nodejs.md`](guides://nodejs.md).)

```bash
tsc --noEmit            # ZOD-TYP-02: all z.infer<> shapes resolve under strict
<test-runner> run       # ZOD-TST-01: valid + invalid cases per schema
```

The *why* behind test-first and error strategy lives in their §0 owners; do not re-derive it here.

---

## 4. Project Structure

Schemas are a first-class layer. Co-locate the schema with the boundary it guards, or centralize a `schemas/` module — but never scatter the same shape across files.

```
src/
├── schemas/
│   ├── primitives.ts     # reusable leaves: Email, UUID, Slug, ISODate…
│   └── <entity>.schema.ts # one domain entity per file; CRUD variants derived
├── <boundary>/           # routes/env/ipc — parse here, pass typed data inward
└── ...
```

- One schema per domain entity; derive `Create`/`Update`/`ListItem` variants from it — do not redefine.
- Extract shared leaves (email, id, date) to `primitives.ts` and reference them.

---

## 5. Zod Specifics

The unique value of this guide.

### A. Schema, type, and the inference rule

The schema is the source of truth; the type is derived. `z.infer` gives the **output** type (after transforms); `z.input` gives the type accepted **before** transforms/coercion.

```ts
import { z } from "zod";

const User = z.object({
  id: z.uuid(),
  email: z.email(),
  role: z.enum(["admin", "user", "viewer"]),
  createdAt: z.coerce.date(),        // input: string|number|Date → output: Date
});

type User = z.infer<typeof User>;    // { id: string; email: string; role: ...; createdAt: Date }
type UserInput = z.input<typeof User>; // createdAt: string | number | Date
```

In Zod 4, common string formats are top-level (`z.email()`, `z.uuid()`, `z.url()`, `z.iso.datetime()`) rather than `z.string().email()` (still supported but deprecated). Enums are `z.enum([...])`; for an existing TS `enum`, `z.enum(MyEnum)` (Zod 4 unified `nativeEnum` into `enum`).

### B. Parse vs safeParse — fail-fast vs handle

```ts
// Trusted invariant, failure is a bug → throw (fail fast)
const cfg = ConfigSchema.parse(loadedConfig);

// User/network input, failure is expected → branch on .success
const r = CreateUser.safeParse(req.body);
if (!r.success) return reply(r.error);   // see error-handling.md for the mapping
doWork(r.data);                          // r.data is fully typed

// Async schemas (refinements/transforms that await) → parseAsync / safeParseAsync
const out = await Schema.parseAsync(payload);
```

`.safeParse` returns a discriminated union `{ success: true, data } | { success: false, error }` — TypeScript narrows it, so no casts are needed. Do **not** wrap `.parse()` in a try/catch and use the catch as normal control flow at a boundary; use `.safeParse()` (ZOD-PRS-01).

### C. Composition — derive, never duplicate

```ts
const Base = z.object({ id: z.uuid(), createdAt: z.coerce.date() });

const User = Base.extend({ email: z.email(), name: z.string().min(1) });

const CreateUser = User.omit({ id: true, createdAt: true });  // POST body
const UpdateUser = CreateUser.partial();                       // PATCH body
const ListItem  = User.pick({ id: true, name: true });
```

`.extend` / `.pick` / `.omit` / `.partial` / `.required` / `.merge` cover every CRUD variant from one base. Unknown-key policy: `z.object` strips unknown keys by default; `.strict()` rejects them (use on untrusted input — ZOD-STR-01); `.loose()` keeps them.

### D. Refinements, transforms, and pipes

- `.refine()` / `.superRefine()` (Zod 4: `.check()`): custom predicates with a message and `path` — for cross-field rules (`password === confirm`).
- `.transform()`: reshape *after* a successful parse (changes the output type; not reversible, so it can't emit JSON Schema — see §E).
- `.pipe()`: chain one schema's output into another's input for multi-step validation.

```ts
const Signup = z.object({ pw: z.string().min(8), confirm: z.string() })
  .refine((d) => d.pw === d.confirm, { message: "Passwords differ", path: ["confirm"] });

const PortFromEnv = z.string()
  .transform((s) => Number(s))
  .pipe(z.number().int().min(1).max(65535));
```

**Branded types** give nominal typing so two `string`-shaped ids are not interchangeable:

```ts
const UserId = z.uuid().brand<"UserId">();
type UserId = z.infer<typeof UserId>;     // string & brand; not assignable from a raw string
```

### E. Boundary recipes

**Env (parse once at startup, fail fast):**
```ts
export const env = z.object({
  NODE_ENV: z.enum(["development", "production", "test"]).default("development"),
  PORT: z.coerce.number().int().min(1).max(65535).default(3000),
  DATABASE_URL: z.url(),
}).parse(process.env);   // config policy/layering: see env-config.md
```

**HTTP boundary (framework-agnostic):** parse `body`/`query`/`params` with `.safeParse()`, and on failure return the structured issues mapped per [`error-handling.md`](guides://error-handling.md). For query/form strings use `z.coerce.*`. Validate responses too in tests so the wire contract (owned by [`rest.md`](guides://rest.md)) can't silently drift.

**Forms:** `.safeParse(formData)` then surface field-level messages via `z.treeifyError`/`z.flattenError` (§F). Most form libraries (React Hook Form, etc.) accept a Zod resolver directly.

**OpenAPI / JSON Schema:** Zod 4 ships `z.toJSONSchema(schema)` (first-party). Use it to generate the OpenAPI component for a route's schema so one definition drives runtime validation *and* the published contract. Pure shapes serialize; `.transform()`/`.pipe()`/`.brand()` do not — keep an AI/wire schema pure and add runtime-only refinements in a separate `.extend()` applied after parsing.

```ts
const json = z.toJSONSchema(User);   // feed into the OpenAPI document (see openapi.md)
```

### F. Error shaping (producing, not strategizing)

Zod produces a `ZodError` with an `.issues[]` array (`{ code, path, message }`). Zod owns turning a failed parse into these issues; mapping them to an HTTP status / user message is owned by [`error-handling.md`](guides://error-handling.md).

```ts
if (!r.success) {
  z.treeifyError(r.error);   // nested, mirrors data shape — good for forms
  z.flattenError(r.error);   // { formErrors, fieldErrors } — flat
  // or map r.error.issues → your app's error DTO
}
```

Attach human messages at definition time: `z.string().min(8, "At least 8 characters")`, `z.email("Enter a valid email")`. Set a localized/global default via `z.config({ customError })`.

### G. Recursive & advanced types

```ts
const Category = z.object({
  name: z.string(),
  get children() { return z.array(Category); },   // Zod 4 getter form for recursion
});
```

Collections: `z.array(T).min(1)`, `z.tuple([...])`, `z.record(z.string(), V)`, `z.map`, `z.set`. Polymorphism: `z.discriminatedUnion("type", [...])` for O(1) tag dispatch (prefer over `z.union` when a literal discriminator exists). Modifiers: `.optional()`, `.nullable()`, `.nullish()`, `.default(v)`, `.catch(fallback)`.

### H. Common footguns

- `as` to satisfy a type instead of parsing → defeats the entire library (ZOD-BND-02).
- A hand-written `interface` next to the schema → drifts; use `z.infer` (ZOD-TYP-01).
- `.passthrough()`/`.loose()` on untrusted input → smuggles unvalidated fields (ZOD-STR-01).
- Forgetting `z.coerce.*` on query/env strings → `"3000"` fails a `z.number()` (it doesn't auto-coerce).
- `.transform()` inside a schema you then serialize to JSON Schema → the transform is silently dropped server-side.
- `z.any()`/`z.unknown()` left unrefined → a typed hole (ZOD-ANY-01).
- Catching `ZodError` and swallowing it → loses the failure (ZOD-ERR-01).

---

## 6. Tooling & Dependencies

Supply-chain/CVE *policy* is owned by [`secure-coding.md`](guides://secure-coding.md); versioning by [`semver.md`](guides://semver.md); the package manager and `tsconfig` by the project's language guide ([`nodejs.md`](guides://nodejs.md) / [`typescript.md`](guides://typescript.md)).

```bash
npm install zod          # v4 is current; v3 API is largely source-compatible
```

- `tsconfig` MUST set `"strict": true` — without it, `z.infer` types degrade and `unknown` handling breaks (owned by `typescript.md`).
- Review the changelog before a Zod major bump; schema behavior (default unknown-key handling, format APIs) changed between v3 and v4.
- Optional ecosystem: `zod-validation-error` (prettier messages), framework resolvers (RHF, etc.).

---

## 7. Quick Reference

```ts
// Schema → type
const S = z.object({ id: z.uuid(), n: z.string().min(1) });
type S = z.infer<typeof S>;            // output type
type SIn = z.input<typeof S>;          // input (pre-transform) type

// Parse
S.parse(x);                            // throws ZodError (fail fast)
S.safeParse(x);                        // { success, data | error } (handle)
await S.parseAsync(x);                 // async refinements/transforms

// Compose
S.extend({}); S.pick({}); S.omit({}); S.partial(); S.merge(Other);
S.strict();                            // reject unknown keys (untrusted input)

// Coerce / refine / transform
z.coerce.number(); z.coerce.date();    // edge string inputs
S.refine(pred, { message, path });     // cross-field rule
S.transform(fn);                       // reshape output
A.pipe(B);                             // multi-step

// Interop
z.toJSONSchema(S);                     // OpenAPI / JSON Schema (pure shapes only)
z.treeifyError(err); z.flattenError(err);
```

---

## 8. Deployment Checklist

Generated from §2 — one box per requirement ID.

- [ ] ZOD-TYP-01 — types inferred via `z.infer`, no parallel hand-written shapes
- [ ] ZOD-TYP-02 — `tsc --noEmit` clean under `strict`
- [ ] ZOD-BND-01 — every external input parsed at its boundary
- [ ] ZOD-BND-02 — `unknown` in, no `as` laundering
- [ ] ZOD-ERR-01 — safeParse failures handled, not swallowed
- [ ] ZOD-PRS-01 — `.safeParse()` at user/network edges
- [ ] ZOD-STR-01 — `.strict()` on untrusted input, no `.passthrough()`
- [ ] ZOD-ANY-01 — no unrefined `z.any()`/`z.unknown()`
- [ ] ZOD-TST-01 — each schema tested with a valid and an invalid input
- [ ] Agent ran every §3 command and documented any fixes

---
**End of Zod Guidelines**
