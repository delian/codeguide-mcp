# OpenAPI Specification Guidelines
Mandatory standards for contract-first API specs: structured documents, reusable components, rich examples, declared security, code generation, Spectral linting. OpenAPI 3.1.1, Spectral 6, Redocly CLI, oasdiff, openapi-generator.

---
name: openapi
title: OpenAPI Specification Guidelines
version: 2.0
last_reviewed: 2026-06-05
kind: cross-cutting
tools: [openapi@3.1.1, spectral@6, redocly-cli, oasdiff, openapi-generator]
requires: []
recommends:
  - rest
  - semver
  - secure-coding
  - oauth
provides:
  - openapi-spec
  - contract-first
  - schema-components
  - spec-linting
---

> 🧭 Authored per [`CONVENTIONS.md`](guides://CONVENTIONS.md): shared concerns are referenced, not restated. This guide owns the **OpenAPI document** — its structure, component reuse, examples, security-scheme declaration, code generation, and linting. The REST design the spec *describes* lives elsewhere.

---

## 0. Prerequisites & References

This guide describes how to author the OpenAPI **document**. The API's *design* and its cross-cutting concerns are owned elsewhere — fetch those when the task touches them; do not redesign them here.

> 📎 **RECOMMENDED — fetch when the task touches them:**
> - [`rest.md`](guides://rest.md) — REST resource design, status codes, pagination strategy, the error model. *(OpenAPI binding: the spec **documents** these decisions; it does not invent them.)*
> - [`semver.md`](guides://semver.md) — versioning & breaking-change policy. *(Binding: `info.version` is the spec's SemVer; breaking changes are gated by `oasdiff`.)*
> - [`secure-coding.md`](guides://secure-coding.md) — auth, secrets, transport, threat model. *(Binding: `securitySchemes` + `security` declare, never weaken, that policy.)*
> - [`oauth.md`](guides://oauth.md) — OAuth 2.x / OIDC flows and scopes. *(Binding: `type: oauth2` scheme flows mirror the authorization server's real flows.)*

> 📎 **SEE ALSO:** [`graphql.md`](guides://graphql.md) · [`grpc.md`](guides://grpc.md) · [`websocket.md`](guides://websocket.md) *(sibling API styles)* · [`ci-cd.md`](guides://ci-cd.md) *(where lint/diff gates run)* · [`zod.md`](guides://zod.md) *(schema generation in TS clients)*

---

## 1. Core Philosophies: OPENAPI-FIRST

OpenAPI-document principles only. REST semantics, auth design, and versioning policy come from §0 — do **not** restate them here.

- **O**ne source of truth: the spec is the **contract**. Server and client are generated or validated *against* it, never hand-drifted away from it (`provides: contract-first`).
- **P**ortable structure: split large specs into `$ref`'d files; every reusable shape lives once in `components` and is referenced (`provides: schema-components`).
- **E**xamples everywhere: every operation and schema carries realistic, schema-valid examples — they power try-it-out, mock servers, and contract tests.
- **N**ormative & precise: schemas pin types, formats, bounds, and `required`; no untyped `object` blobs leak into the contract.
- **A**utomatable: the spec drives codegen, mock servers, and CI gates; it MUST be machine-valid (parses + lints clean).
- **P**inned & versioned: `info.version` follows SemVer (see `semver.md`); breaking changes are detected mechanically, never by eyeball.
- **I**dentity & security declared: every security scheme the API enforces is declared and applied; nothing is implicit.

**Verified Spec**: An agent-authored OpenAPI document MUST pass every gate in §2 before delivery.

---

## 2. Requirements (MANDATORY, auditable)

RFC-2119 keywords. IDs `OAPI-<TOPIC>-<NN>`. Each row has a binary gate; rows binding a shared rule cite its owner.

| ID | Requirement | Verify | Gate |
|----|-------------|--------|------|
| OAPI-STRUCT-01 | Document MUST be valid OpenAPI 3.1.x | `redocly lint openapi.yaml` (or `spectral lint`) | 0 errors |
| OAPI-LINT-01 | Spectral ruleset MUST pass clean | `spectral lint openapi.yaml --fail-severity=error` | exit 0 |
| OAPI-LINT-02 | Every operation MUST have a unique camelCase `operationId`, `summary`, and `tags` | spectral rule (§8) | 0 errors |
| OAPI-STRUCT-02 | Reusable shapes MUST live in `components` and be `$ref`'d (no inline duplication) | review / `spectral` `no-inline-schema` rule | no dup inline schemas |
| OAPI-DOC-01 | Every operation, parameter, and schema field MUST have a `description` | spectral `*-description` rules | 0 errors |
| OAPI-EX-01 | Every request body, response, and named schema MUST carry a schema-valid `example`/`examples` | `spectral` + `oas3-valid-media-example` | 0 invalid examples |
| OAPI-SEC-01 | All enforced auth MUST be declared in `securitySchemes` and applied via `security` (see `secure-coding.md`, `oauth.md`) | review / spectral `oas3-operation-security-defined` | every non-public op secured |
| OAPI-SEC-02 | Secret-bearing fields MUST be `writeOnly`; server-set fields `readOnly`; no secrets in examples (see `secure-coding.md`) | review / grep examples | no secret leakage |
| OAPI-VER-01 | `info.version` MUST be SemVer and bumped per change class (see `semver.md`) | review / CI | matches change class |
| OAPI-VER-02 | A change MUST NOT introduce a breaking diff without a major bump (see `semver.md`) | `oasdiff breaking old.yaml new.yaml` | no breaking unless major |
| OAPI-GEN-01 | Generated server/client MUST regenerate cleanly from the current spec | `openapi-generator-cli generate ...` | exit 0, no drift |

> **Forbidden**: hand-editing generated stubs instead of the spec; inlining a schema already in `components`; shipping an operation without `operationId`/security/examples; merging a breaking diff without a major version bump; placing real tokens, keys, or PII in examples.

---

## 3. Validation Protocol (contract-first loop)

This guide owns the spec-as-test loop. Author/edit the document, then run — in order — and fix until every gate is green. The *why* of versioning and security lives in the §0 owners.

```bash
redocly lint openapi.yaml                                  # OAPI-STRUCT-01 (bundle-aware, follows $ref)
spectral lint openapi.yaml --fail-severity=error           # OAPI-LINT-01/02, DOC-01, EX-01, SEC-01
oasdiff breaking --fail-on ERR last-released.yaml openapi.yaml   # OAPI-VER-02 (run in CI vs the released spec)
openapi-generator-cli generate -i openapi.yaml -g <target> -o build/gen   # OAPI-GEN-01
```

Contract-first workflow:

1. **Design the spec first** — agree the contract before any implementation (this is the OpenAPI analogue of test-first; the lint/diff gates above *are* the failing→passing loop).
2. **Generate** server stubs and client SDKs from the spec (§6). Implementation fills the stubs; it never redefines the contract.
3. **Validate continuously** — run §3 in CI (see [`ci-cd.md`](guides://ci-cd.md)); a red gate blocks merge.

---

## 4. Document Structure

The single entry document. `paths` *document* the REST design owned by [`rest.md`](guides://rest.md); this section shows the OpenAPI **encoding**, not the design rules.

```yaml
# openapi.yaml
openapi: 3.1.1

info:
  title: Orders API
  version: 1.4.0                 # SemVer — see semver.md
  summary: Manage customer orders and fulfilment.
  description: |
    Long-form docs. Rate limiting, pagination, and idempotency
    conventions are described in rest.md and summarised here.
  contact: { name: API Support, email: api@example.com }
  license: { name: Apache-2.0, identifier: Apache-2.0 }   # SPDX id (3.1)

servers:
  - url: https://api.example.com/v1     # major version in path — see semver.md/rest.md
    description: Production

tags:
  - name: Orders
    description: Order lifecycle operations.

security:
  - bearerAuth: []                # default; operations may override (see §7)

paths: {}                         # see §5
components: {}                    # see §5–§7 — the reuse hub
webhooks: {}                      # see §8 (3.1 native)
```

### File organization (split + bundle)

Large specs split across files and are bundled for tooling that needs a single document:

```
api/
├── openapi.yaml              # entry: info, servers, tags, security, $ref-only paths/components
├── paths/                    # one file per resource group
│   └── orders.yaml
├── components/
│   ├── schemas/              # the canonical shapes (OAPI-STRUCT-02)
│   ├── parameters/
│   ├── responses/
│   └── securitySchemes.yaml
└── examples/
```

```yaml
# openapi.yaml — references only; no inline definitions
paths:
  /orders:
    $ref: './paths/orders.yaml#/orders'
components:
  schemas:
    Order: { $ref: './components/schemas/order.yaml' }
```

Bundle for single-file consumers: `redocly bundle openapi.yaml -o dist/openapi.yaml`.

---

## 5. Components: the reuse hub (`provides: schema-components`)

Define each shape **once** under `components` and `$ref` it everywhere (OAPI-STRUCT-02). This is the core value of the format.

### A. Object schemas — precise, bounded, with modifiers

```yaml
components:
  schemas:
    Order:
      type: object
      description: A customer order.
      required: [id, customerId, status, total, createdAt]
      properties:
        id:        { type: string, format: uuid, readOnly: true,
                     examples: ['123e4567-e89b-12d3-a456-426614174000'] }
        customerId:{ type: string, format: uuid }
        status:    { type: string, enum: [pending, paid, shipped, delivered, cancelled],
                     default: pending }
        total:     { type: number, format: double, minimum: 0 }
        couponCode:{ type: [string, 'null'], maxLength: 32 }   # 3.1: nullable via type array
        createdAt: { type: string, format: date-time, readOnly: true }
      additionalProperties: false
```

3.1 notes: it is a strict superset of **JSON Schema 2020-12** — use `type: [T, 'null']` (not the removed `nullable: true`), `examples` arrays (not singular `example`) in schemas, and `$ref` siblings (`$ref` may now sit beside `description`).

### B. Collection + pagination envelope

The pagination *strategy* is owned by [`rest.md`](guides://rest.md); the spec encodes whatever it chose:

```yaml
    OrderList:
      type: object
      required: [data, page]
      properties:
        data: { type: array, items: { $ref: '#/components/schemas/Order' } }
        page: { $ref: '#/components/schemas/Page' }
    Page:
      type: object
      required: [limit, hasMore]
      properties:
        limit:      { type: integer, minimum: 1, maximum: 100, default: 20 }
        nextCursor: { type: [string, 'null'] }
        hasMore:    { type: boolean }
```

### C. Composition & polymorphism

Use `allOf` for mixins, `oneOf`/`anyOf` for variants, and always pair a `oneOf` with a `discriminator` so codegen emits tagged unions:

```yaml
    Notification:
      oneOf:
        - $ref: '#/components/schemas/EmailNotification'
        - $ref: '#/components/schemas/SmsNotification'
      discriminator:
        propertyName: channel
        mapping:
          email: '#/components/schemas/EmailNotification'
          sms:   '#/components/schemas/SmsNotification'
```

### D. Reusable parameters

```yaml
components:
  parameters:
    OrderId:
      name: orderId
      in: path
      required: true
      description: Order identifier.
      schema: { type: string, format: uuid }
    Limit:
      name: limit
      in: query
      description: Max items per page.
      schema: { type: integer, minimum: 1, maximum: 100, default: 20 }
    Cursor:
      name: cursor
      in: query
      description: Opaque pagination cursor from the previous response.
      schema: { type: string }
```

### E. Reusable responses & the error model

The error **model** (shape, when each fires) is a REST design concern — own it in [`rest.md`](guides://rest.md); prefer **RFC 9457 `application/problem+json`**. OpenAPI just declares it once and references it:

```yaml
components:
  responses:
    NotFound:
      description: Resource not found.
      content:
        application/problem+json:
          schema: { $ref: '#/components/schemas/Problem' }
          example: { type: 'about:blank', title: Not Found, status: 404,
                     detail: 'Order ord_456 does not exist' }
  schemas:
    Problem:                       # RFC 9457
      type: object
      required: [type, title, status]
      properties:
        type:   { type: string, format: uri }
        title:  { type: string }
        status: { type: integer }
        detail: { type: string }
        instance: { type: string, format: uri }
```

### F. Operation skeleton (ties it together)

```yaml
paths:
  /orders/{orderId}:
    parameters: [ { $ref: '#/components/parameters/OrderId' } ]
    get:
      operationId: getOrder        # unique, camelCase (OAPI-LINT-02)
      summary: Get an order
      description: Returns a single order by id.
      tags: [Orders]
      responses:
        '200':
          description: The order.
          content:
            application/json:
              schema: { $ref: '#/components/schemas/Order' }
              examples:
                default: { $ref: '#/components/examples/OrderExample' }
        '404': { $ref: '#/components/responses/NotFound' }
```

---

## 6. Code generation (`provides: contract-first`)

The spec is the source; generated artefacts are build output (never hand-edited — OAPI-GEN-01).

```bash
# Client SDK / server stubs
openapi-generator-cli generate -i openapi.yaml -g typescript-fetch -o build/client
openapi-generator-cli generate -i openapi.yaml -g go-server       -o build/server

# Mock server straight from the spec (drives consumer dev before the API exists)
prism mock openapi.yaml

# Type-only generation for TS consumers
npx openapi-typescript openapi.yaml -o src/api-types.ts
```

Rules:
- Generated code lives in a build/ output dir and is git-ignored (or committed only as a reviewed artefact); regeneration MUST be reproducible in CI.
- Implementation extends generated stubs via hooks/interfaces — it never forks the contract.
- When the spec changes, regenerate; a non-empty `git diff` on hand-written code that *should* be generated is a defect.
- For TS request/response validation derived from the contract, generate Zod schemas (see [`zod.md`](guides://zod.md)).

---

## 7. Security schemes — declare, don't design (`provides`)

The auth **policy** is owned by [`secure-coding.md`](guides://secure-coding.md); the **flows/scopes** by [`oauth.md`](guides://oauth.md). OpenAPI's job is to *declare* exactly what the API enforces and *apply* it (OAPI-SEC-01). Declaring a weaker scheme than the API enforces is a defect.

```yaml
components:
  securitySchemes:
    bearerAuth:                    # JWT bearer
      type: http
      scheme: bearer
      bearerFormat: JWT
    apiKeyAuth:
      type: apiKey
      in: header
      name: X-API-Key
    oidc:                          # prefer OpenID Connect discovery when available
      type: openIdConnect
      openIdConnectUrl: https://auth.example.com/.well-known/openid-configuration
    oauth2:                        # flows MUST mirror the real authorization server — see oauth.md
      type: oauth2
      flows:
        authorizationCode:
          authorizationUrl: https://auth.example.com/oauth/authorize
          tokenUrl:         https://auth.example.com/oauth/token
          scopes:
            read:orders: Read orders
            write:orders: Create and modify orders

security:                          # global default
  - bearerAuth: []

paths:
  /public/health:
    get:
      security: []                 # explicit opt-out for public ops
  /admin/orders:
    get:
      security:
        - oauth2: [read:orders]    # scope-gated (AND across schemes, OR across list items)
```

Mark secret-bearing request fields `writeOnly` and server-set fields `readOnly`; never put real credentials, tokens, or PII in examples (OAPI-SEC-02).

---

## 8. Webhooks & callbacks (3.1)

3.1 makes outbound events first-class via top-level `webhooks` (provider-initiated, no fixed URL) — distinct from per-operation `callbacks` (tied to a prior request).

```yaml
webhooks:
  orderShipped:
    post:
      operationId: orderShippedWebhook
      summary: Sent when an order ships.
      requestBody:
        required: true
        content:
          application/json:
            schema: { $ref: '#/components/schemas/Event' }
      responses:
        '200': { description: Acknowledged. }
```

Document the signature header consumers must verify (binding to [`secure-coding.md`](guides://secure-coding.md)).

---

## 9. Linting with Spectral (`provides: spec-linting`)

A committed, extended ruleset is the enforcement engine for §2. Start from `spectral:oas` and add project rules:

```yaml
# .spectral.yaml
extends: [[spectral:oas, all]]
rules:
  operation-operationId: error           # OAPI-LINT-02
  operation-operationId-unique: error
  operation-tag-defined: error
  operation-description: error           # OAPI-DOC-01
  oas3-valid-media-type-example: error   # OAPI-EX-01
  oas3-operation-security-defined: error # OAPI-SEC-01

  operation-id-camel-case:
    description: operationId must be camelCase.
    severity: error
    given: "$.paths[*][get,put,post,delete,patch]"
    then: { field: operationId, function: casing, functionOptions: { type: camel } }

  schema-must-have-example:
    description: Named component schemas must carry an example.
    severity: warn
    given: "$.components.schemas[*]"
    then: { field: examples, function: defined }
```

Run `spectral lint openapi.yaml --fail-severity=error` locally and in CI (see [`ci-cd.md`](guides://ci-cd.md)).

---

## 10. Quick Reference

```bash
redocly lint openapi.yaml                            # OAPI-STRUCT-01: structural validity
redocly bundle openapi.yaml -o dist/openapi.yaml     # single-file bundle for tooling
spectral lint openapi.yaml --fail-severity=error     # OAPI-LINT/DOC/EX/SEC gates
oasdiff breaking --fail-on ERR old.yaml new.yaml      # OAPI-VER-02: breaking-change gate
openapi-generator-cli generate -i openapi.yaml -g <target> -o build/   # OAPI-GEN-01
prism mock openapi.yaml                              # spin a mock server from the contract
redocly preview-docs openapi.yaml                    # rendered docs preview
```

```yaml
# 3.1 cheat-sheet
type: [string, 'null']    # nullable (replaces nullable: true)
examples: ['v']           # schema examples are an array in 3.1
const: fixed              # single-value constraint
format: uuid|email|uri|date-time|date|password
readOnly: true            # server-set; writeOnly: true for secrets-in
$ref: '#/components/...'  # may now have sibling keywords
```

---

## 11. Deployment Checklist

Generated from §2 — one box per requirement ID.

- [ ] OAPI-STRUCT-01 — valid OpenAPI 3.1.x (`redocly lint` clean)
- [ ] OAPI-STRUCT-02 — reusable shapes in `components`, no inline duplication
- [ ] OAPI-LINT-01/02 — Spectral clean; every op has unique camelCase `operationId`, summary, tags
- [ ] OAPI-DOC-01 — every operation/parameter/field described
- [ ] OAPI-EX-01 — schema-valid examples on bodies, responses, named schemas
- [ ] OAPI-SEC-01 — all enforced auth declared and applied (see `secure-coding.md`, `oauth.md`)
- [ ] OAPI-SEC-02 — `writeOnly`/`readOnly` set; no secrets/PII in examples
- [ ] OAPI-VER-01/02 — `info.version` SemVer-correct; no unbumped breaking diff (`oasdiff`)
- [ ] OAPI-GEN-01 — server/client regenerate cleanly from the spec
- [ ] Agent ran every §3 command and documented any fixes

---
**End of OpenAPI Specification Guidelines**
