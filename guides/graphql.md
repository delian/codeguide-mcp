# GraphQL API Guidelines
Mandatory standards for designing and implementing GraphQL APIs: schema-first design, N+1-safe resolvers, cursor pagination, and query-cost security. Apollo Server 4, GraphQL Yoga, graphql-js, DataLoader, GraphQL Code Generator, Apollo Federation.

---
name: graphql
title: GraphQL API Guidelines
version: 2.0
last_reviewed: 2026-06-05
kind: cross-cutting
tools: [apollo-server@4, graphql-yoga@5, graphql-js@16, dataloader@2, graphql-codegen@5, apollo-federation@2, graphql-armor]
requires: []
recommends:
  - rest
  - oauth
  - secure-coding
  - error-handling
  - observability
provides:
  - graphql-schema-design
  - resolvers
  - dataloader
  - query-security
---

> 🧭 Authored per [`CONVENTIONS.md`](guides://CONVENTIONS.md): shared concerns are referenced, not restated. This guide owns GraphQL's unique surface — schema, resolvers, batching, pagination, and query-cost defense — and references the owners of auth, security, error strategy, and observability.

---

## 0. Prerequisites & References

GraphQL is an API *style*; it sits on top of cross-cutting concerns owned elsewhere. Apply those guides; this one does not repeat them.

> 📎 **RECOMMENDED — fetch when the task touches them:**
> - [`oauth.md`](guides://oauth.md) — authentication & token validation. *(GraphQL binding: validate the token once in context-building; resolvers read the authenticated principal — never re-parse `Authorization` per field.)*
> - [`secure-coding.md`](guides://secure-coding.md) — input validation, injection, secrets, dependency CVEs. *(GraphQL-specific binding owned here: query depth/complexity/introspection — see §6.)*
> - [`error-handling.md`](guides://error-handling.md) — failure taxonomy, retries, fail-fast vs degrade. *(GraphQL binding: how that taxonomy maps onto the `errors[]` envelope — see §5.)*
> - [`rest.md`](guides://rest.md) — when REST is the better fit (see §1, "REST vs GraphQL").
> - [`observability.md`](guides://observability.md) — tracing/metrics. *(GraphQL binding: per-resolver spans + trace propagation through context — see §9.)*

> 📎 **SEE ALSO:** [`grpc.md`](guides://grpc.md) · [`openapi.md`](guides://openapi.md) · [`websocket.md`](guides://websocket.md) *(subscription transport)* · [`designpatterns.md`](guides://designpatterns.md) · [`ci-cd.md`](guides://ci-cd.md)

---

## 1. Core Philosophies: GRAPHQL-FIRST

GraphQL-specific principles only. Auth, generic security, error strategy, and tracing come from §0.

- **G**raph thinking: model relationships between types, not request/response endpoints. The schema is the contract and the documentation.
- **R**esolver efficiency: every list/relation field is N+1-prone — batch through DataLoader by default (see §3.B).
- **A**uthorization in the data layer: enforce on the resolver/field that touches data, not at a gateway edge alone, so nested paths cannot bypass it.
- **P**agination always: list fields MUST return a paginated Connection, never an unbounded array.
- **H**uman-readable schema: descriptive type/field names, `description` strings, `@deprecated(reason:)` over breaking removals.
- **Q**uery-cost ceiling: bound depth, complexity, and aliases so one request cannot DoS the server (see §6).
- **L**ayered: keep typeDefs, resolvers, and data sources separate; resolvers orchestrate, data sources do I/O.

**REST vs GraphQL (choose deliberately).** Prefer GraphQL when clients need flexible, nested, client-shaped reads across many related types, or to collapse many round-trips into one. Prefer REST (see [`rest.md`](guides://rest.md)) for simple CRUD, cache-by-URL/CDN semantics, file up/download, and webhooks. Do not expose both styles over the same domain without a single shared service layer beneath them.

**Verified Code**: Agent-generated GraphQL MUST pass every gate in §2 before delivery.

---

## 2. Requirements (MANDATORY, auditable)

RFC-2119 keywords. IDs `GQL-<TOPIC>-<NN>`. Each row has a binary gate; rows binding a shared rule cite its owner.

| ID | Requirement | Verify | Gate |
|----|-------------|--------|------|
| GQL-SCHEMA-01 | Schema MUST build without errors | `graphql-inspector validate` / `buildSchema()` test | exit 0 |
| GQL-SCHEMA-02 | Schema changes MUST NOT break clients without a deprecation cycle | `graphql-inspector diff <old> <new>` | no `BREAKING` |
| GQL-SCHEMA-03 | Every list field MUST be a paginated Connection (no unbounded `[T!]!` collections) | schema lint / review | no unpaginated lists |
| GQL-SCHEMA-04 | Mutations MUST return a payload type carrying `userErrors`, not throw for expected domain failures | review / schema lint | payload pattern present |
| GQL-RESOLVE-01 | Relation/list resolvers MUST batch via DataLoader (no per-item DB call) | N+1 test (assert query count) / `apollo` trace | no N+1 |
| GQL-AUTHZ-01 | Every non-public field/resolver MUST enforce authorization (see `oauth.md`) | authz tests per field | unauthorized → denied |
| GQL-SEC-01 | Query **depth** MUST be capped | depth-limit rule test with over-deep query | rejected |
| GQL-SEC-02 | Query **complexity/cost** MUST be capped | complexity test over the limit | rejected |
| GQL-SEC-03 | Introspection MUST be disabled in production (see `secure-coding.md`) | prod config check / introspection query | denied in prod |
| GQL-SEC-04 | Input MUST be validated beyond scalar types (see `secure-coding.md`) | validation unit tests | invalid → rejected |
| GQL-ERR-01 | Errors MUST use stable `extensions.code`; internals MUST NOT leak in prod (see `error-handling.md`) | `formatError` test in prod mode | no stack/SQL leaked |
| GQL-OBS-01 | Each request MUST carry a trace id propagated to resolvers/logs (see `observability.md`) | trace span assertion | trace id present |
| GQL-TST-01 | Schema, resolver, and authz behavior MUST be tested first (see `tdd.md`) | `<test_command>` | exit 0, 0 skips |

> **Forbidden**: returning unbounded lists; resolvers that issue one query per parent (N+1); enabling introspection or unbounded query depth/complexity in production; leaking internal error messages/stack traces to clients; performing auth only at the gateway and trusting it inside resolvers.

---

## 3. Resolvers & the N+1 Problem

The single most important GraphQL implementation concern. A resolver is `(parent, args, context, info)`; field resolvers fire once **per parent object**, so a list of N parents naively triggers N child queries.

### A. Resolver structure

Resolvers stay thin: authorize, delegate to a data source or loader, shape the result. No business logic, no raw SQL inline.

```typescript
export const userResolvers: Resolvers<Context> = {
  Query: {
    user: (_p, { id }, ctx) => ctx.dataSources.users.byId(id),   // authz via field on User, below
    me: (_p, _a, ctx) => ctx.auth.userId ? ctx.loaders.user.load(ctx.auth.userId) : null,
  },
  User: {
    // Relation resolvers go through loaders — NOT direct queries.
    posts: (user, args, ctx) => ctx.loaders.postsByAuthor.load(user.id),
  },
};
```

### B. DataLoader — batch + per-request cache (GQL-RESOLVE-01)

DataLoader coalesces all `.load(key)` calls made in one tick into a single batch function call. **Create loaders per request** (in context-building), never module-global — a shared loader would cache across users and leak data.

```typescript
import DataLoader from "dataloader";

export const createLoaders = (db: Db) => ({
  user: new DataLoader<string, User | null>(async (ids) => {
    const rows = await db.users.whereIdIn(ids as string[]); // ONE query for all ids
    const map = new Map(rows.map((u) => [u.id, u]));
    return ids.map((id) => map.get(id) ?? null);            // MUST return in input order, same length
  }),
});

// context (built once per request):
async function context({ req }): Promise<Context> {
  return { auth: await authenticate(req), loaders: createLoaders(db), dataSources, traceId: traceIdOf(req) };
}
```

Loader rules (the common footguns):
- The batch function MUST resolve to an array **the same length and order** as `keys`; map misses to `null`/`Error`, never drop them.
- One loader **per key shape** (`userById`, `postsByAuthorId`). A loader keyed by an object needs a `cacheKeyFn`.
- For one-to-many relations, return arrays per key (each element is the list for that parent), e.g. `postsByAuthor.load(authorId) → Post[]`.
- Prime the cache (`loader.prime(id, row)`) when a parent query already fetched children, to skip a redundant batch.

### C. Detecting N+1
Assert query counts in tests (`expect(db.queryCount).toBe(1)` for a list-of-N query), and inspect Apollo/Yoga traces for repeated identical resolvers. A green N+1 test is the gate for GQL-RESOLVE-01.

---

## 4. Schema Design

The contract. Design it before resolvers (schema-first), validate it in CI, and evolve it without breaking clients.

### A. Types, inputs, enums

```graphql
type User {
  id: ID!
  email: String!
  displayName: String!
  role: UserRole!
  createdAt: DateTime!
  posts(first: Int, after: String): PostConnection!   # paginated relation (GQL-SCHEMA-03)
}

enum UserRole { ADMIN MODERATOR USER GUEST }

scalar DateTime          # define & validate custom scalars in resolvers
scalar Email

input CreateUserInput {  # mutation args go in a single Input type
  email: Email!
  displayName: String!
  role: UserRole = USER
}
```

Conventions: Types/inputs `PascalCase` (inputs end `Input`), fields/args `camelCase`, enum values `SCREAMING_SNAKE_CASE`. Nullability is a design decision — mark a field `!` only when it can never legitimately be null; over-using `!` makes one failed field null the whole parent.

### B. Mutations return payloads, not bare types (GQL-SCHEMA-04)

Expected domain failures (validation, conflict, not-found) belong **in the payload** as `userErrors`, so a partial success is representable and clients get typed, localizable messages. Reserve thrown GraphQL errors (the top-level `errors[]`) for unexpected/transport failures (see §5).

```graphql
type CreateUserPayload {
  user: User
  userErrors: [UserError!]!
}
type UserError { field: [String!]! message: String! code: ErrorCode! }
enum ErrorCode { VALIDATION CONFLICT NOT_FOUND FORBIDDEN }

type Mutation { createUser(input: CreateUserInput!): CreateUserPayload! }
```

### C. Evolution without breaking (GQL-SCHEMA-02)
GraphQL has no URL versions — you evolve one schema. Add fields/args additively; never remove or retype a field clients use without `@deprecated(reason: "use X")` and a migration window. Gate every PR with a schema diff (`graphql-inspector diff`) that fails on `BREAKING` changes.

---

## 5. Errors in GraphQL

Error *strategy* (taxonomy, retries, fail-fast) is owned by [`error-handling.md`](guides://error-handling.md). What is **GraphQL-specific** is the dual channel and the response envelope:

- **Top-level `errors[]`** — execution/transport failures. Each carries `message`, `path`, and a stable machine code in `extensions.code` (`UNAUTHENTICATED`, `FORBIDDEN`, `BAD_USER_INPUT`, `INTERNAL_SERVER_ERROR`). A single response MAY contain **both** `data` (partial) and `errors`.
- **Typed `userErrors` in mutation payloads** — expected domain outcomes (see §4.B). Prefer these for anything a client should branch on.

```typescript
import { GraphQLError } from "graphql";
export const forbidden = (msg = "Not authorized") =>
  new GraphQLError(msg, { extensions: { code: "FORBIDDEN" } });

// server-side scrubbing — internals MUST NOT leak in prod (GQL-ERR-01)
const server = new ApolloServer({
  typeDefs, resolvers,
  formatError: (formatted, err) => {
    log.error({ err, code: formatted.extensions?.code });        // log full detail server-side
    if (process.env.NODE_ENV === "production" &&
        formatted.extensions?.code === "INTERNAL_SERVER_ERROR") {
      return { message: "Internal error", extensions: { code: "INTERNAL_SERVER_ERROR" } };
    }
    return formatted;
  },
});
```

Stable `extensions.code` values are part of your contract — clients branch on them. Never put SQL, stack traces, or internal hostnames in a client-facing message.

---

## 6. Query-Cost Security (the GraphQL attack surface)

This is the binding that [`secure-coding.md`](guides://secure-coding.md) defers to GraphQL. A single legitimate-looking query can be exponentially expensive because clients control the shape. Defend with **all** of:

- **Depth limit (GQL-SEC-01)** — reject queries nested beyond a cap (e.g. 10) to stop cyclic `friends{friends{...}}` blowups.
- **Complexity/cost limit (GQL-SEC-02)** — assign per-field cost (list fields multiply by their `first`/limit arg) and reject over a budget. This catches wide queries that depth alone misses.
- **Introspection off in prod (GQL-SEC-03)** — `introspection: false`; introspection hands attackers your whole schema. Keep it on in dev/staging.
- **Pagination caps** — clamp `first`/`last` to a max (e.g. ≤ 100); a Connection arg is still attacker-controlled.
- **Disable batching / cap aliases** — array-batched operations and aliased duplicate fields multiply cost; cap or disable.
- **Persisted queries / allowlist (prod)** — accept only pre-registered operation hashes so arbitrary queries cannot run at all; the strongest control where the client set is known.
- **Timeouts & rate limits** — bound resolver/request time; rate-limit by principal.

```typescript
// graphql-armor bundles depth, cost, alias, and introspection guards (Apollo/Yoga plugin)
import { ApolloArmor } from "@escape.tech/graphql-armor";
const armor = new ApolloArmor({
  maxDepth: { n: 10 },
  costLimit: { maxCost: 1000 },
  blockFieldSuggestion: { enabled: true },   // don't hint field names on typos
});
const server = new ApolloServer({
  typeDefs, resolvers,
  introspection: process.env.NODE_ENV !== "production",   // GQL-SEC-03
  ...armor.protect(),                                     // GQL-SEC-01/02
});
```

Authentication (token validation) and authorization policy are owned by [`oauth.md`](guides://oauth.md); enforce authz at the field/resolver that reads data (GQL-AUTHZ-01) — a `@auth(requires:)` schema directive or a per-field guard — so nested paths cannot route around it.

---

## 7. Pagination — Relay Connections

The owned pattern for all list fields (GQL-SCHEMA-03). Cursor-based, not offset-based: cursors stay stable when rows are inserted/deleted between pages, where offsets skip or duplicate.

```graphql
type PostConnection { edges: [PostEdge!]! pageInfo: PageInfo! totalCount: Int! }
type PostEdge { cursor: String! node: Post! }
type PageInfo { hasNextPage: Boolean! hasPreviousPage: Boolean! startCursor: String endCursor: String }
```

Implementation rules:
- Forward paginate with `first`/`after`, backward with `last`/`before`; reject `first` and `last` together.
- A cursor is an **opaque** server token (base64-encode the sort key; never expose a raw DB id/offset).
- Over-fetch by one (`take: first + 1`) to compute `hasNextPage` without a second count query.
- Clamp `first`/`last` to a maximum (ties into GQL-SEC-02); default a sane page size when omitted.
- Sort on a stable, unique key (e.g. `(createdAt, id)`); keyset/seek pagination scales where `OFFSET` does not (see [`performance.md`](guides://performance.md)).

---

## 8. Subscriptions & Federation

### A. Subscriptions (real-time)
Use for server-pushed events (new message, status change). Transport is `graphql-ws` over WebSocket (the legacy `subscriptions-transport-ws` is unmaintained — do not use it); WebSocket concerns are owned by [`websocket.md`](guides://websocket.md). Authorize on **subscribe** and filter per event so a subscriber only receives authorized payloads; in multi-instance deployments back the PubSub with Redis/Kafka, not in-memory.

```typescript
Subscription: {
  orderUpdated: {
    subscribe: withFilter(
      (_p, _a, ctx) => ctx.pubsub.asyncIterator(["ORDER_UPDATED"]),
      (payload, vars, ctx) =>
        payload.orderUpdated.id === vars.orderId &&
        payload.orderUpdated.ownerId === ctx.auth.userId,   // authz in the filter
    ),
  },
},
```

### B. Federation (Apollo Federation 2)
Compose multiple subgraphs into one supergraph rather than building a hand-rolled stitched monolith. Each subgraph owns its types; `@key` declares an entity's identity and `__resolveReference` resolves it for the gateway. Keep authz and DataLoader **inside each subgraph** — the gateway does not enforce them. Validate composition in CI (`rover supergraph compose` / schema checks) before publishing.

---

## 9. Observability & Tooling

Tracing/metrics policy is owned by [`observability.md`](guides://observability.md). GraphQL binding (GQL-OBS-01): extract/generate a trace id in context-building and propagate it to every resolver and log line; emit a span per resolver (Apollo OpenTelemetry plugin or Yoga's tracing) so you can see which field is slow. Metrics worth recording: per-operation latency, per-resolver error rate, and request complexity score.

**Code generation.** Generate resolver and operation types from the schema (GraphQL Code Generator: `typescript`, `typescript-resolvers`, `typescript-operations`) so resolver signatures and client queries stay type-checked against the schema. Treat generated files as build artifacts (regenerate in CI; keep out of hand-edits).

```bash
graphql-codegen                      # regenerate types from schema + documents
graphql-inspector validate schema.graphql
graphql-inspector diff <base> <head> # GQL-SCHEMA-02: fail on BREAKING
```

---

## 10. Deployment Checklist

Generated from §2 — one box per requirement ID. No new requirements here.

- [ ] GQL-SCHEMA-01 — schema builds clean
- [ ] GQL-SCHEMA-02 — no breaking schema diff without deprecation
- [ ] GQL-SCHEMA-03 — all list fields are paginated Connections
- [ ] GQL-SCHEMA-04 — mutations return payloads with `userErrors`
- [ ] GQL-RESOLVE-01 — relation resolvers batch via DataLoader (no N+1)
- [ ] GQL-AUTHZ-01 — every non-public field enforces authorization
- [ ] GQL-SEC-01/02 — depth and complexity/cost capped
- [ ] GQL-SEC-03 — introspection disabled in production
- [ ] GQL-SEC-04 — input validated beyond scalar types
- [ ] GQL-ERR-01 — stable error codes; no internal leakage in prod
- [ ] GQL-OBS-01 — trace id propagated to resolvers and logs
- [ ] GQL-TST-01 — schema/resolver/authz tested first
- [ ] Agent ran every verification command and documented any fixes

---
**End of GraphQL API Guidelines**
