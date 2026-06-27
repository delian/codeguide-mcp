# Mongoose ODM Guidelines
Mandatory standards for the Mongoose ODM: typed schemas, models, validation, virtuals, middleware, and population. Mongoose 8.x, async/await only. Mongoose 8, Node 22 LTS, TypeScript 5.x.

---
name: mongoose
title: Mongoose ODM Guidelines
version: 2.0
last_reviewed: 2026-06-05
kind: tooling
tools: [mongoose@8, node@22, typescript@5]
requires:
  - nodejs
  - mongodb
recommends:
  - typescript
  - secure-coding
  - error-handling
  - zod
provides:
  - mongoose-schemas
  - mongoose-models
  - odm-validation
  - mongoose-population
  - mongoose-middleware
---

> 🧭 Authored per [`CONVENTIONS.md`](guides://CONVENTIONS.md): shared concerns are referenced, not restated. This guide covers only what is unique to Mongoose. The database — document modeling, the embed-vs-reference decision, indexing/ESR, aggregation, transactions semantics — is owned by `mongodb.md`; the runtime by `nodejs.md`.

---

## 0. Prerequisites & References

Fetch and apply these **before** generating Mongoose code. Their rules are assumed below and not repeated.

> 📎 **REQUIRED — fetch & apply first:**
> - [`mongodb.md`](guides://mongodb.md) — the database. Document modeling, **embed vs reference**, the 16 MB limit, indexing & the **ESR rule**, aggregation, write/read concern, transactions, change streams. Do **not** re-reason about these here; Mongoose only declares them — the *why* lives there.
> - [`nodejs.md`](guides://nodejs.md) — the runtime: module system, async/await, top-level await, error propagation, the driver connection lifecycle.

> 📎 **RECOMMENDED — fetch when the task touches them:**
> - [`typescript.md`](guides://typescript.md) — typed schemas/models. *(Binding: Mongoose 8 infers types via `InferSchemaType`; hydrated docs are `HydratedDocument<T>`. All `strict` rules apply — see §9.)*
> - [`secure-coding.md`](guides://secure-coding.md) — injection, secrets, CVE policy. *(Binding: NoSQL/operator-injection defense, query sanitization, `strictQuery`, never trust user-supplied query objects — see §10.)*
> - [`error-handling.md`](guides://error-handling.md) — error strategy. *(Binding: catch `ValidationError`/`CastError`/duplicate-key `E11000` at a defined boundary — see §3, §10.)*
> - [`zod.md`](guides://zod.md) — edge validation. *(Binding: validate untrusted input with Zod at the API edge; schema validators are the last-line DB invariant, not input parsing — see §3.D.)*

> 📎 **SEE ALSO:** [`tdd.md`](guides://tdd.md) *(test models/hooks against `mongodb-memory-server` or a Compose replica set)* · [`performance.md`](guides://performance.md) · [`observability.md`](guides://observability.md)

---

## 1. Core Philosophies: ODM-DISCIPLINE

Mongoose-specific principles only. Document modeling, indexing, injection policy, and the runtime come from §0.

- **The schema is a contract, not the model design.** Mongoose enforces *structure* over a schemaless server; it does **not** decide embed-vs-reference for you — that decision is owned by `mongodb.md`. Mongoose declares the model; MongoDB modeling rules still govern it.
- **Types are inferred, never hand-written twice.** Define the schema once; derive the TS type with `InferSchemaType` / `HydratedDocument`. A drifting hand-maintained interface is a defect (§9).
- **async/await only.** Mongoose 8 **dropped callback support** entirely — every query, hook, and connection is a promise. No `cb`-style APIs, no `.exec(cb)`.
- **Reads are lean by default.** `.lean()` is the single biggest perf lever; only hydrate when you need virtuals, getters, or `.save()` (§6).
- **Populate is a join you opted into.** Each `populate()` is an extra query — guard against N+1 and prefer the modeling patterns in `mongodb.md` (extended-reference) on hot paths (§5).
- **Indexes are declared, never auto-built in prod.** Set `autoIndex: false` in production; build indexes deliberately (§4).

**Verified Code**: Agent-generated Mongoose code MUST pass every gate in §2 before delivery.

---

## 2. Requirements (MANDATORY, auditable)

RFC-2119 keywords. IDs `MONGOOSE-<TOPIC>-<NN>`. Each row has a binary gate; rows binding a shared rule cite its owner.

| ID | Requirement | Verify | Gate |
|----|-------------|--------|------|
| MONGOOSE-SCHEMA-01 | Schemas MUST declare explicit types, `required`, and `enum`/range validators for every business field (no untyped `Mixed` for known shapes) | review schema vs domain | typed, validated |
| MONGOOSE-TYP-01 | Document/model types MUST be inferred (`InferSchemaType`/`HydratedDocument`), not hand-duplicated (see `typescript.md`) | `grep -rn "InferSchemaType" src/` ; `tsc --noEmit` | inferred, exit 0 |
| MONGOOSE-ASYNC-01 | All Mongoose calls MUST use async/await; no callback APIs (removed in v8) | `grep -rEn '\.(find\w*|save|exec|create)\([^)]*,\s*function' src/` | no matches |
| MONGOOSE-VAL-01 | Server-side validators MUST run on updates: `runValidators: true` on `findOneAndUpdate`/`updateOne` (off by default) | `grep -rn "runValidators" src/` ; review | present on updates |
| MONGOOSE-IDX-01 | Indexes MUST be declared in the schema and follow ESR (see `mongodb.md`); `autoIndex` MUST be `false` in production | `grep -rn "autoIndex" src/config` ; review | `autoIndex:false` in prod |
| MONGOOSE-LEAN-01 | Read-only queries that don't need a hydrated doc MUST use `.lean()` (see `performance.md`) | review hot read paths | lean on read paths |
| MONGOOSE-POP-01 | `populate()` MUST select only needed fields and MUST NOT run inside a per-document loop (N+1) | review ; query-count test | no N+1, projected |
| MONGOOSE-CONN-01 | A single connection MUST be established at startup (top-level await `connect`) and reused; `strictQuery` set explicitly | review bootstrap | one connection, strictQuery set |
| MONGOOSE-TX-01 | Multi-document invariants MUST use a session/transaction; the session MUST be passed to every op (see `mongodb.md`) | review ; `grep -rn "session" src/` | session threaded |
| MONGOOSE-SEC-01 | User input MUST NOT be passed as a raw query/update object; cast & sanitize, reject `$`/`.` keys (see `secure-coding.md`) | review ; input validated with Zod (`zod.md`) | sanitized at edge |
| MONGOOSE-ERR-01 | `ValidationError`/`CastError`/`E11000` duplicate-key MUST be handled at a defined boundary (see `error-handling.md`) | review error mapping | handled |
| MONGOOSE-TST-01 | Schemas, validators, and hooks MUST be test-first against `mongodb-memory-server`/replica set (see `tdd.md`) | `npm test` | exit 0, 0 skips |

> **Forbidden**: callback-style Mongoose calls (gone in v8); hand-maintained document interfaces parallel to the schema; `findOneAndUpdate` without `runValidators` where validators exist; `autoIndex:true` in production; hydrating large read result sets instead of `.lean()`; `populate()` in a loop; passing `req.body`/`req.query` straight into `find`/`update`; multi-doc writes without a shared session.

---

## 3. Schemas, SchemaTypes & Validation

The first half of Mongoose's unique value. The schema declares structure, defaults, and last-line validation over MongoDB's schemaless server — it does **not** replace `$jsonSchema` server validation (owned by `mongodb.md`, MONGO-MODEL-02), nor edge input validation (Zod, §3.D).

### A. Schema definition & SchemaTypes

```typescript
import { Schema, model, InferSchemaType, HydratedDocument } from "mongoose";

const addressSchema = new Schema({
  type:    { type: String, enum: ["home", "work", "other"], default: "home" },
  street:  { type: String, required: true },
  city:    { type: String, required: true },
  country: { type: String, required: true },
  isDefault: { type: Boolean, default: false },
}, { _id: false });                       // subdocument, no own _id

const userSchema = new Schema({
  email: {
    type: String,
    required: [true, "Email is required"],
    unique: true,                          // builds a unique INDEX, not a validator (see note)
    lowercase: true, trim: true,
    match: [/^\S+@\S+\.\S+$/, "Invalid email format"],
  },
  passwordHash: { type: String, required: true, select: false }, // excluded from queries by default
  profile: {
    firstName: { type: String, maxlength: 100 },
    lastName:  { type: String, maxlength: 100 },
  },
  addresses: [addressSchema],
  status:    { type: String, enum: ["active", "inactive", "suspended"], default: "active", index: true },
  balance:   { type: Schema.Types.Decimal128, default: "0" }, // money → Decimal128, never Number
  deletedAt: { type: Date, default: null },
}, {
  timestamps: true,                        // auto createdAt / updatedAt
  toJSON: { virtuals: true }, toObject: { virtuals: true },
});
```

SchemaTypes: `String`, `Number`, `Date`, `Boolean`, `Buffer`, `ObjectId` (refs), `Decimal128` (money — never `Number`), `Map`, `UUID`, and `Mixed` (use sparingly — opaque, change-tracking requires `markModified`). `unique: true` builds an **index**, not a validator: duplicate inserts fail with a driver `E11000` error, handled per §10/`error-handling.md`, not a `ValidationError`.

### B. Models & the connection

```typescript
type User = InferSchemaType<typeof userSchema>;     // §9: derive, don't duplicate
export const User = model<User>("User", userSchema);
```

```typescript
// bootstrap — top-level await, one connection reused process-wide (MONGOOSE-CONN-01)
import mongoose from "mongoose";
mongoose.set("strictQuery", true);                  // reject unknown query fields (see §10)
await mongoose.connect(process.env.MONGODB_URI!, {  // URI from env (see env-config in mongodb.md)
  autoIndex: process.env.NODE_ENV !== "production", // MONGOOSE-IDX-01 footgun — see §4
  serverSelectionTimeoutMS: 5000,
});
```

Connect once at startup and reuse the global connection; Mongoose pools internally (`maxPoolSize`). Never call `connect` per request. Write/read concern, retryable writes, and replica-set topology are owned by `mongodb.md` — set them on the URI.

### C. Built-in & custom validators

Built-in: `required`, `min`/`max` (Number/Date), `minlength`/`maxlength`, `enum`, `match`. Custom validators are sync or async functions returning/​resolving boolean:

```typescript
userSchema.path("email").validate(
  async (v: string) => !(await User.exists({ email: v })),
  "Email already in use",                   // prefer the unique index (E11000) over this race-prone check
);
userSchema.path("balance").validate((v) => Number(v) >= 0, "Balance cannot be negative");
```

**Validators run on `save()` and `create()` but NOT on `findOneAndUpdate`/`updateOne` by default** — you MUST pass `runValidators: true` (MONGOOSE-VAL-01):

```typescript
await User.findByIdAndUpdate(id, { status: "inactive" },
  { new: true, runValidators: true });      // without runValidators, enum/required are skipped
```

A failed validator throws `ValidationError` (per-path `.errors`); handle it at the boundary per `error-handling.md` (§10).

### D. Validation altitude — Mongoose validators vs Zod (when to use which)

Two layers, different jobs — do not conflate them:

- **Zod at the edge** (owned by `zod.md`) — parse and reject *untrusted external input* (`req.body`, query params) **before** it reaches the model. Produces typed, trusted data and good 400-level error messages. This is your injection boundary (§10).
- **Mongoose schema validators** — the *last-line database invariant* (enum membership, ranges, required relationships) that must hold for every write path, including internal ones. They are not a substitute for input parsing and they are not enforced by the server (use `$jsonSchema` for that, see `mongodb.md`).

Rule: validate shape/format with Zod at ingress; keep Mongoose validators for invariants you want enforced on every code path. Don't duplicate elaborate format regex in both layers — own format at the edge, own invariants at the model.

---

## 4. Indexes (declare in schema; the autoIndex footgun)

Index *strategy* — the ESR rule, compound/partial/TTL/text indexes, covered queries — is owned by `mongodb.md` (§4 there). Mongoose only **declares** them; the ordering rules still apply.

```typescript
userSchema.index({ status: 1, createdAt: -1 });          // compound — ESR order per mongodb.md
userSchema.index({ email: 1 }, { unique: true });
userSchema.index({ deletedAt: 1 }, { expireAfterSeconds: 0 }); // TTL
```

**The `autoIndex` production footgun (MONGOOSE-IDX-01):** by default Mongoose calls `createIndex` for every declared index on model init. In production this runs on every deploy/restart, can foreground-build on large collections, and silently blocks. Set `autoIndex: false` in production and build indexes deliberately — via a migration, `Model.syncIndexes()` in a controlled job, or out-of-band `createIndex` with the concurrency options from `mongodb.md`. Keep `autoIndex: true` only in dev/test for convenience.

---

## 5. Population (and the populate-vs-embed decision)

`populate()` replaces stored `ObjectId` refs with the referenced documents — a client-side join executed as **additional queries**.

```typescript
const orderSchema = new Schema({
  userId: { type: Schema.Types.ObjectId, ref: "User", required: true, index: true },
  total:  { type: Schema.Types.Decimal128, required: true },
  status: { type: String, enum: ["pending", "shipped", "delivered"], default: "pending" },
}, { timestamps: true });

// project the populated fields; never populate the whole doc you don't need
const orders = await Order.find({ status: "pending" })
  .populate("userId", "email profile.firstName")   // selective
  .sort({ createdAt: -1 })
  .lean();                                          // §6: plain objects, faster
```

**The N+1 / cost reality (MONGOOSE-POP-01):** each `populate()` is at least one extra query (`$in` batched across the result set — good), but `populate()` **inside a loop over documents** fires one query per document (bad). Deeply nested populate multiplies queries. Select only needed fields; cap depth; verify with a query-count test.

**populate vs embed is a *modeling* decision owned by `mongodb.md`** (embed for bounded, read-together, owned data; reference + populate for large/independent/high-churn relations). Do not re-derive it here — Mongoose merely *implements* whichever the model dictates. On hot read paths prefer the **extended-reference** pattern (copy the few hot fields onto the parent, per `mongodb.md` §3.B) over a runtime `populate()`.

---

## 6. Lean queries (the #1 perf lever)

By default every query returns **hydrated documents** — full Mongoose objects with change tracking, getters, virtuals, and `.save()`. That machinery is pure overhead for read-only data.

```typescript
const users = await User.find({ status: "active" }).lean();  // plain JS objects — far less CPU/memory
```

`.lean()` skips hydration: no virtuals, no getters/setters, no `.save()`, no document middleware on the results. Use it for **every read you only serialize or inspect** (MONGOOSE-LEAN-01) — list endpoints, reports, exports. Hydrate (omit `.lean()`) only when you need to mutate-and-save, or need a virtual/getter. For large result sets `.lean()` is often a multiple-x latency and memory win (rationale: `performance.md`). For huge scans, stream with `.cursor()` instead of materializing an array.

---

## 7. Virtuals, getters & setters

Computed properties not persisted to MongoDB.

```typescript
userSchema.virtual("fullName").get(function () {
  return `${this.profile?.firstName ?? ""} ${this.profile?.lastName ?? ""}`.trim();
});
// virtual populate — a reverse reference without storing an array of ids
userSchema.virtual("orders", { ref: "Order", localField: "_id", foreignField: "userId" });
```

Enable `toJSON: { virtuals: true }` (done in §3.A) so virtuals serialize. **Virtuals do not exist on `.lean()` results** and **cannot be queried** (they aren't stored) — if you need to filter on it, store/compute it (the *computed pattern*, owned by `mongodb.md`). Getters/setters transform on access/assignment (`set: (v) => v.trim()`); keep them pure and cheap — heavy logic belongs in a method or service, not a getter.

---

## 8. Middleware / hooks (pre/post)

Hooks run logic around document, query, aggregate, and model operations. **All hooks are async/await in v8** (no `next` callback required — return a promise).

```typescript
import bcrypt from "bcrypt";

// document pre-save: hash only when changed
userSchema.pre("save", async function () {
  if (this.isModified("passwordHash")) {
    this.passwordHash = await bcrypt.hash(this.passwordHash, 12);
  }
});

// query middleware: enforce soft-delete filter on every find
userSchema.pre(/^find/, function () {
  this.where({ deletedAt: null });
});

// post hook: map the driver duplicate-key error to a domain error (see error-handling.md)
userSchema.post("save", function (err: any, _doc: unknown, next: (e?: Error) => void) {
  if (err?.code === 11000) next(new Error("DUPLICATE_EMAIL"));
  else next(err);
});
```

Hook scope matters: **document** middleware (`save`, `validate`, `remove`) binds `this` to the doc; **query** middleware (`find`, `findOneAndUpdate`, `updateOne`) binds `this` to the query — and **document hooks do NOT fire on `findOneAndUpdate`/`updateOne`** (those go through query middleware). Keep hooks fast, deterministic, and free of cross-collection writes that should be in a transaction (§/`mongodb.md`). Methods (`schema.methods`), statics (`schema.statics`), and query helpers (`schema.query`) attach behavior to instances/models/chains respectively.

---

## 9. TypeScript integration

Policy owned by `typescript.md`; Mongoose-8 binding here. **Infer, don't duplicate** (MONGOOSE-TYP-01):

```typescript
import { Schema, model, InferSchemaType, HydratedDocument } from "mongoose";

const userSchema = new Schema({ /* ...as §3.A... */ });

type User = InferSchemaType<typeof userSchema>;     // raw stored shape
type UserDoc = HydratedDocument<User>;              // a live document (has _id, methods, .save())

export const User = model<User>("User", userSchema);

async function deactivate(id: string): Promise<UserDoc | null> {
  return User.findByIdAndUpdate(id, { status: "inactive" }, { new: true, runValidators: true });
}
```

`InferSchemaType` derives the document type from the single schema source — a hand-written parallel interface drifts and is a defect. For methods/statics/virtuals that inference can't see, declare them via the generic type params on `Schema`/`model` (e.g. an explicit methods interface) rather than re-typing the data fields. Run `tsc --noEmit` under `strict` (gate from `typescript.md`).

---

## 10. Security & error-handling bindings

Policy owned by `secure-coding.md` and `error-handling.md`; Mongoose bindings (MONGOOSE-SEC-01, MONGOOSE-ERR-01):

- **NoSQL / operator injection.** Never pass `req.body`/`req.query` directly into `find`/`update` — a `{ "$gt": "" }` or `{ "$where": "..." }` payload becomes a query operator. Validate and **cast input with Zod at the edge** (`zod.md`) to fixed types, reject keys starting with `$` or containing `.`. Mongoose casting helps but is not a sanitizer.
- **`strictQuery`.** Set `mongoose.set("strictQuery", true)` so unknown fields in filter objects are stripped rather than passed through — reduces injection surface and silent mis-queries (MONGOOSE-CONN-01).
- **`select: false`** on secrets (password hashes, tokens) so they're excluded by default; opt in explicitly (`.select("+passwordHash")`) only where needed.
- **Error mapping.** `ValidationError` (per-path `.errors`) and `CastError` (bad ObjectId/type) come from Mongoose; duplicate-key `E11000` comes from the driver (`unique` index, not a validator). Map each to a domain error at a defined boundary per `error-handling.md` — don't leak raw Mongoose errors to clients.

Injection/sanitization rationale and the secrets/CVE policy live in `secure-coding.md`; the connection-string-from-env rule and DB-level RBAC live in `mongodb.md`.

---

## 11. Quick Reference

```typescript
// schema → inferred type → model
const s = new Schema({ email: { type: String, required: true, unique: true } }, { timestamps: true });
type T = InferSchemaType<typeof s>;
export const M = model<T>("M", s);

// connect once (top-level await), strictQuery, autoIndex off in prod
mongoose.set("strictQuery", true);
await mongoose.connect(process.env.MONGODB_URI!, { autoIndex: process.env.NODE_ENV !== "production" });

// read fast (lean), write with validators, populate selectively
await M.find({ status: "active" }).lean();
await M.findByIdAndUpdate(id, patch, { new: true, runValidators: true });
await Order.find().populate("userId", "email").lean();

// transaction (session threaded to every op — semantics owned by mongodb.md)
const session = await mongoose.startSession();
await session.withTransaction(async () => {
  await A.updateOne({ _id }, { $inc: { n: -1 } }, { session });
  await B.updateOne({ _id }, { $inc: { n:  1 } }, { session });
});
await session.endSession();
```

---

## 12. Deployment Checklist

Generated from §2 — one box per requirement ID. No new requirements here.

- [ ] MONGOOSE-SCHEMA-01 — every business field typed + validated; no opaque `Mixed`
- [ ] MONGOOSE-TYP-01 — document/model types inferred (`InferSchemaType`/`HydratedDocument`), `tsc` clean
- [ ] MONGOOSE-ASYNC-01 — async/await only; no callback APIs (removed in v8)
- [ ] MONGOOSE-VAL-01 — `runValidators: true` on update operations
- [ ] MONGOOSE-IDX-01 — indexes declared in schema (ESR per `mongodb.md`); `autoIndex:false` in prod
- [ ] MONGOOSE-LEAN-01 — read-only queries use `.lean()`
- [ ] MONGOOSE-POP-01 — `populate()` projected and not in a per-doc loop (no N+1)
- [ ] MONGOOSE-CONN-01 — single connection at startup (top-level await), `strictQuery` set
- [ ] MONGOOSE-TX-01 — multi-doc invariants use a threaded session/transaction
- [ ] MONGOOSE-SEC-01 — input cast/sanitized at the edge (Zod), no raw query objects (see `secure-coding.md`)
- [ ] MONGOOSE-ERR-01 — `ValidationError`/`CastError`/`E11000` mapped at a boundary (see `error-handling.md`)
- [ ] MONGOOSE-TST-01 — schema/validator/hook tests pass against `mongodb-memory-server`/replica set
- [ ] Agent ran the §2 verify commands and documented any fixes

---
**End of Mongoose ODM Guidelines**
