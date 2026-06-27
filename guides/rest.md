# REST API Design Guidelines
Mandatory standards for designing REST/HTTP APIs: resource modeling, HTTP method/status semantics, versioning, pagination & filtering, idempotency, content negotiation, and HATEOAS. Language-agnostic; HTTP/1.1 & HTTP/2, OpenAPI 3.1, RFC 7231/9110, RFC 7807/9457.

---
name: rest
title: REST API Design Guidelines
version: 2.0
last_reviewed: 2026-06-05
kind: cross-cutting
tools: []
requires: []
recommends:
  - openapi
  - oauth
  - secure-coding
  - error-handling
  - observability
  - semver
provides:
  - resource-modeling
  - http-semantics
  - api-versioning
  - pagination
  - idempotency
---

> 🧭 Authored per [`CONVENTIONS.md`](guides://CONVENTIONS.md): shared concerns are referenced, not restated. This guide owns REST/HTTP *design*; the schema/spec, auth, error-body, and versioning-policy details live in the guides referenced in §0.

---

## 0. Prerequisites & References

REST design decisions bind to several cross-cutting concerns. This guide owns the **HTTP/REST shape** of each; the policy lives in the owner. Do not restate the owner's content.

> 📎 **RECOMMENDED — fetch when the task touches them:**
> - [`openapi.md`](guides://openapi.md) — the contract: schema definitions, `$ref` reuse, examples, spec linting. REST owns *what* the resources/methods are; OpenAPI owns *how the spec is written*. Every API in this guide is described by an OpenAPI 3.1 document.
> - [`oauth.md`](guides://oauth.md) — authentication & token flows. REST owns only the transport binding (`Authorization: Bearer`, `WWW-Authenticate`, 401 vs 403).
> - [`secure-coding.md`](guides://secure-coding.md) — input validation, injection, secrets, transport security. REST owns only the API-surface binding (IDs in URLs, TLS-only, CORS).
> - [`error-handling.md`](guides://error-handling.md) — error strategy & taxonomy. REST owns only the *wire mapping* (status code ⇄ error body, RFC 9457 `problem+json`).
> - [`observability.md`](guides://observability.md) — tracing/metrics. REST owns only header propagation (`traceparent`, rate-limit headers).
> - [`semver.md`](guides://semver.md) — what is a breaking change. REST owns only how a version is *expressed* on the wire (URL/header/media-type).

> 📎 **SEE ALSO:** [`graphql.md`](guides://graphql.md) · [`grpc.md`](guides://grpc.md) · [`websocket.md`](guides://websocket.md) (sibling API styles) · [`microservices.md`](guides://microservices.md) · [`hexagonal.md`](guides://hexagonal.md) (route handlers are adapters) · [`logging.md`](guides://logging.md)

---

## 1. Core Philosophies: REST-FIRST

REST/HTTP design principles only. Test-first, security, error strategy, and architecture come from §0.

- **R**esource-oriented: model **nouns** (resources), not verbs (RPC). State changes are transfers of resource representations.
- **E**xplicit HTTP semantics: methods, status codes, and headers carry meaning — never overload `200` for errors or `GET` for mutations.
- **S**tateless: each request carries everything needed to serve it; no server-side session affinity. Auth travels per-request (see `oauth.md`).
- **T**yped contract: every request/response is described in OpenAPI 3.1 (see `openapi.md`); the spec is the source of truth.
- **C**acheable & uniform: identical interface across resources; responses declare cacheability (`Cache-Control`, `ETag`).
- **I**dempotent & safe by method: `GET/HEAD/OPTIONS` safe; `PUT/DELETE` idempotent; `POST` made retry-safe with `Idempotency-Key`.

**Verified APIs**: every endpoint MUST satisfy each gate in §2 before delivery.

---

## 2. Requirements (MANDATORY, auditable)

RFC-2119 keywords. IDs `REST-<TOPIC>-<NN>`. Each row has a binary gate; rows binding a shared rule cite its owner.

| ID | Requirement | Verify | Gate |
|----|-------------|--------|------|
| REST-RES-01 | Paths MUST name plural resource collections (nouns), not actions | review / Spectral ruleset | no verb-in-path |
| REST-RES-02 | Resource IDs in URLs MUST be opaque & non-enumerable (no sequential ints, no PII) (see `secure-coding.md`) | review / grep paths | no `/{int}` or PII segments |
| REST-HTTP-01 | Each operation MUST use the correct method & success status per §4 | contract tests | methods/codes match spec |
| REST-HTTP-02 | `GET/HEAD/OPTIONS` MUST be safe (no state change); `PUT/DELETE` MUST be idempotent | contract/integration tests | repeat call = same state |
| REST-IDEM-01 | Non-idempotent `POST` that creates resources MUST accept `Idempotency-Key` | integration test (double-submit) | one resource created |
| REST-VER-01 | API MUST carry an explicit version; a breaking change MUST bump it (see `semver.md`) | review / diff old vs new spec | version present & bumped |
| REST-PAGE-01 | Collection endpoints MUST paginate and bound page size (default + max) | contract test | unbounded list rejected |
| REST-ERR-01 | Errors MUST use the correct 4xx/5xx code and a single machine-readable body (RFC 9457) (see `error-handling.md`) | contract test | code + `problem+json` |
| REST-NEG-01 | Bodies MUST be JSON (`application/json`); server MUST honor `Accept`/`Content-Type` or return 406/415 | contract test | correct negotiation |
| REST-SEC-01 | TLS-only; protected endpoints declare a security scheme; no secrets in URL/query (see `secure-coding.md`, `oauth.md`) | review / spec lint | https + scheme + clean URLs |
| REST-DOC-01 | Every operation MUST be in the OpenAPI 3.1 spec with examples (see `openapi.md`) | `spectral lint` / `swagger-cli validate` | exit 0 |
| REST-OBS-01 | Responses MUST propagate a trace id and expose rate-limit headers (see `observability.md`) | integration test | `traceparent` echoed |

> **Forbidden**: tunneling actions through `GET` query params; returning `200` with an error body; sequential/PII identifiers in URLs; unbounded collection responses; a breaking change without a version bump; bespoke per-endpoint error shapes.

---

## 3. Resource Modeling

The heart of REST: turn a domain into a small set of addressable resources with a uniform interface.

### A. Nouns, plural, hierarchical
```
/v1/organizations                         # collection
/v1/organizations/{orgId}                 # member
/v1/organizations/{orgId}/projects        # sub-collection (containment)
/v1/organizations/{orgId}/projects/{projectId}/tasks/{taskId}
```
- Collections are **plural** nouns; a member is `collection/{id}`.
- Nest only to express **ownership/containment**; keep depth ≤ 3. Beyond that, link instead of nesting (see §8 HATEOAS).
- Put **primary identifiers in the path** (uniquely identifies + makes `GET` cacheable); put **filters/pagination/sorting in the query string** (§6).

### B. IDs are opaque (REST-RES-02)
URL IDs MUST be non-enumerable and carry no PII — sequential integers enable enumeration attacks (the *why* is owned by [`secure-coding.md`](guides://secure-coding.md)). Use UUIDv4/v7, ULID, or NanoID:
```
✅  /v1/users/01ARZ3NDEKTSV4RRFFQ69G5FAV          # ULID
❌  /v1/users/42                                   # enumerable
❌  /v1/users/alice@example.com                    # PII + cache/log leak
```

### C. Actions that aren't CRUD
When an operation is a genuine verb (not state-transfer), model it as a **sub-resource controller** under the owning resource, via `POST`:
```
POST /v1/orders/{orderId}/cancellation        # an event/command resource
POST /v1/invoices/{invoiceId}/payments        # creates a payment under the invoice
POST /v1/articles/{articleId}/publish         # state transition (acceptable when no resource fits)
```
Prefer modeling the *result* as a resource (`/payments`) over an imperative verb where one exists.

---

## 4. HTTP Method & Status Semantics

The uniform interface (RFC 9110). Method choice and status code are part of the contract — REST-HTTP-01/02.

| Method | Use | Safe | Idempotent | Body | Success |
|--------|-----|------|------------|------|---------|
| `GET` | Read resource/collection | yes | yes | no | `200` (`206` for ranges) |
| `HEAD` | Headers only (existence, `ETag`) | yes | yes | no | `200` |
| `OPTIONS` | Allowed methods / CORS preflight | yes | yes | no | `200`/`204` |
| `POST` | Create under a collection; non-idempotent action | no | no | yes | `201` (+`Location`) / `202` async |
| `PUT` | Create-or-**replace** at a known URL | no | yes | yes | `200`/`204` (`201` if created) |
| `PATCH` | **Partial** update (JSON Merge Patch RFC 7396 or JSON Patch RFC 6902) | no | no* | yes | `200`/`204` |
| `DELETE` | Remove resource | no | yes | optional | `204` (`200` w/ body) |

\* `PATCH` semantics depend on the patch document; make it idempotent where feasible.

**Status code mapping (canonical):**

| Class | Code | When |
|-------|------|------|
| 2xx | `200` ok · `201` created · `202` accepted (async) · `204` no content · `206` partial |
| 3xx | `301`/`308` moved · `304` not modified (conditional `GET`) |
| 4xx | `400` malformed · `401` unauthenticated · `403` authenticated-but-forbidden · `404` not found · `405` method not allowed (set `Allow`) · `406` can't satisfy `Accept` · `409` conflict (version/duplicate) · `410` gone · `412` precondition failed · `415` unsupported media type · `422` semantic validation failed · `428` precondition required · `429` rate limited (set `Retry-After`) |
| 5xx | `500` unexpected · `502` bad gateway · `503` unavailable (set `Retry-After`) · `504` upstream timeout |

Distinctions agents get wrong: `400` (syntactically malformed) vs `422` (well-formed but semantically invalid); `401` (who are you?) vs `403` (you may not); `409` (state conflict) vs `412`/`428` (conditional-header precondition).

---

## 5. Idempotency & Concurrency (REST-IDEM-01, REST-HTTP-02)

REST owns making unreliable networks safe to retry.

### A. Idempotency keys for `POST`
Creating `POST`s MUST accept a client-supplied `Idempotency-Key` (UUID). The server stores `key → first response` for a TTL and **replays** the stored response on retry instead of creating a duplicate:
```
POST /v1/payments
Idempotency-Key: 8f3a...   →  201 (first time)
POST /v1/payments                                       (retry, same key)
Idempotency-Key: 8f3a...   →  201 (replayed, no new payment)
```
Same key + **different** body → `422`. `PUT`/`DELETE` are already idempotent and need no key.

### B. Optimistic concurrency (lost-update prevention)
Use conditional requests (RFC 9110 §13). Server returns an `ETag`; client echoes it on mutation:
```
GET   /v1/users/{id}         → 200  ETag: "v7"
PUT   /v1/users/{id}
If-Match: "v7"               → 200  (or 412 Precondition Failed if it changed)
```
Require `If-Match` on mutations to forbid blind overwrites; respond `428 Precondition Required` when it's missing. `If-None-Match` + `304` powers client caching for `GET`.

---

## 6. Pagination, Filtering, Sorting (REST-PAGE-01)

Collection endpoints MUST paginate with a bounded page size. Primary IDs stay in the path; these controls go in the query string.

```
?limit=20&cursor=eyJpZCI6...     # cursor pagination — REQUIRED for large/changing sets
?page=2&limit=20                 # page/offset — acceptable for small, stable sets
?sort=-created_at,name           # '-' = descending; comma-separated precedence
?status=active,pending           # multi-value filter (OR within a field)
?created_after=2026-01-01        # range filter
?fields=id,name                  # sparse fieldset
?expand=owner                    # embed related resource
```

- **Cursor (keyset) pagination is preferred** over offset: stable under inserts/deletes and O(1) at depth. Offset pagination skips/duplicates rows when the set changes and degrades on large offsets.
- Always send a `limit` **default** and a hard **max**; reject or clamp larger values.
- Envelope a collection so clients get metadata + navigation links (§8):
```json
{ "data": [ ... ],
  "pagination": { "limit": 20, "next_cursor": "eyJ...", "has_more": true },
  "links": { "self": "/v1/users?limit=20", "next": "/v1/users?limit=20&cursor=eyJ..." } }
```

---

## 7. Content Negotiation & Caching (REST-NEG-01)

- Default representation is `application/json`; honor the `Accept` header, returning `406` when it can't be satisfied and `415` for an unsupported request `Content-Type`.
- Use **vendor/versioned media types** when negotiating versions by header: `Accept: application/vnd.example.v2+json` (§9).
- `PATCH` request bodies declare their patch media type: `application/merge-patch+json` or `application/json-patch+json`.
- Caching is part of the contract: send `Cache-Control` (e.g. `public, max-age=300` or `no-store` for private data) and `ETag`/`Last-Modified` on cacheable `GET`s; support conditional `GET` (`If-None-Match` → `304`). PII-bearing responses MUST be `no-store`.
- Compression negotiated via `Accept-Encoding`/`Content-Encoding` (`gzip`/`br`).

---

## 8. HATEOAS & Hypermedia

The "uniform interface" includes hypermedia: responses link to valid next states so clients aren't hard-coded to URL templates.

```json
{ "id": "01ARZ...", "status": "pending",
  "_links": {
    "self":    { "href": "/v1/orders/01ARZ..." },
    "cancel":  { "href": "/v1/orders/01ARZ.../cancellation", "method": "POST" },
    "payment": { "href": "/v1/orders/01ARZ.../payments",     "method": "POST" } } }
```

- Provide at minimum a `self` link; include state-transition links that are currently valid (omit `cancel` once shipped).
- Pick **one** hypermedia convention and apply it consistently — HAL (`_links`/`_embedded`), JSON:API (`links`/`relationships`), or a documented custom shape.
- Full HATEOAS is a SHOULD, not a MUST: link collections (pagination) and key state transitions even when not adopting a formal hypermedia format.

---

## 9. Versioning (REST-VER-01)

What counts as a breaking change is owned by [`semver.md`](guides://semver.md). REST owns only **how the version rides on the wire**:

| Strategy | Form | Trade-off |
|----------|------|-----------|
| **URL path** (preferred) | `/v1/users`, `/v2/users` | Most visible, cache- & proxy-friendly, trivial to route. Coarse-grained. |
| Custom header | `API-Version: 2` | Keeps URLs clean; harder to test/cache; easy for clients to forget. |
| Media type | `Accept: application/vnd.example.v2+json` | Pure REST; granular per-representation; highest client friction. |

- Use a **major** version number in the URL; additive (non-breaking) changes ship without a bump (new optional fields, new endpoints). Breaking changes (removing/renaming fields, changing types, tightening validation, changing status codes) REQUIRE a new version.
- Run old and new versions in parallel during deprecation; signal sunset with the `Deprecation` and `Sunset` response headers (RFC 8594).

---

## 10. Error Responses on the Wire (REST-ERR-01)

Error *taxonomy/strategy* is owned by [`error-handling.md`](guides://error-handling.md). REST owns the **wire mapping**: pick the correct status code (§4) and return **one** consistent machine-readable body. Use RFC 9457 `application/problem+json`:

```json
{
  "type": "https://example.com/problems/validation",
  "title": "Request validation failed",
  "status": 422,
  "detail": "email is not a valid address",
  "instance": "/v1/users",
  "errors": [ { "field": "email", "message": "invalid format" } ]
}
```

- One error schema across the whole API (define once in OpenAPI `components`, `$ref` everywhere — see `openapi.md`).
- The `status` field MUST equal the HTTP status code. Never return `200` with an embedded error.
- For `429`/`503`, include `Retry-After`; for `405`, include `Allow`. Do not leak stack traces or internal identifiers in `detail`.

---

## 11. Security Bindings

Policy is owned by [`secure-coding.md`](guides://secure-coding.md) and [`oauth.md`](guides://oauth.md). REST owns only these API-surface bindings:

- **TLS only** — reject plaintext; set `Strict-Transport-Security`.
- **Auth transport** — `Authorization: Bearer <token>` (see `oauth.md`); `401` + `WWW-Authenticate` when missing/invalid, `403` when authenticated but unauthorized.
- **No secrets in URLs** — tokens/passwords/PII never in path or query (they leak via logs, history, referrers, caches); IDs are opaque (§3.B).
- **CORS** — explicit allow-list of origins/methods/headers; never reflect `Origin` with credentials.
- **Rate limiting** — `429` + `X-RateLimit-Limit`/`-Remaining`/`-Reset` and `Retry-After`.
- **Input validation** — validate every body/param against its schema at the boundary (the *how* of validation is owned by `secure-coding.md`; the *schema* by `openapi.md`); reject unknown fields (`additionalProperties: false`).

---

## 12. Observability Bindings (REST-OBS-01)

Tracing/metrics policy is owned by [`observability.md`](guides://observability.md); structured logs by [`logging.md`](guides://logging.md). REST owns the header contract: accept an inbound trace context, generate one if absent, propagate it to downstreams, and echo it back.

```
traceparent: 00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01   # W3C Trace Context (preferred)
X-Request-ID: <uuid>          # per-request id, echoed in the response
```

---

## 13. Quick Reference

```
# Collections & members
GET    /v1/things            # list (paginated, §6)        200
POST   /v1/things            # create                      201 + Location
GET    /v1/things/{id}       # read                        200 / 304
PUT    /v1/things/{id}       # replace (If-Match)          200/204 / 412
PATCH  /v1/things/{id}       # partial (merge-patch+json)  200/204
DELETE /v1/things/{id}       # delete                      204

# Sub-resources & actions
GET    /v1/things/{id}/parts          # sub-collection
POST   /v1/things/{id}/cancellation   # action-as-resource

# Headers that carry contract meaning
Authorization: Bearer <token>        Idempotency-Key: <uuid>
ETag / If-Match / If-None-Match      Cache-Control / Retry-After
Accept / Content-Type                traceparent / X-Request-ID
```

Spec validation/lint commands are owned by [`openapi.md`](guides://openapi.md) (`spectral lint`, `swagger-cli validate`).

---

## 14. Deployment Checklist

Generated from §2 — one box per requirement ID. No new requirements.

- [ ] REST-RES-01 — paths are plural resource nouns, no verbs
- [ ] REST-RES-02 — opaque, non-enumerable IDs; no PII in URLs
- [ ] REST-HTTP-01 — correct method + success status per operation
- [ ] REST-HTTP-02 — safe methods don't mutate; PUT/DELETE idempotent
- [ ] REST-IDEM-01 — creating POSTs honor `Idempotency-Key`
- [ ] REST-VER-01 — explicit version; breaking change bumps it (see `semver.md`)
- [ ] REST-PAGE-01 — collections paginate with default + max page size
- [ ] REST-ERR-01 — correct 4xx/5xx + single `problem+json` body (see `error-handling.md`)
- [ ] REST-NEG-01 — JSON bodies; `Accept`/`Content-Type` honored (406/415)
- [ ] REST-SEC-01 — TLS-only, security scheme declared, clean URLs (see `secure-coding.md`, `oauth.md`)
- [ ] REST-DOC-01 — every operation in OpenAPI 3.1 with examples (see `openapi.md`)
- [ ] REST-OBS-01 — trace id propagated, rate-limit headers present (see `observability.md`)
- [ ] Agent verified the spec (see `openapi.md`) and ran contract tests

---
**End of REST API Design Guidelines**
