# Error Handling Guidelines
Canonical, language-agnostic strategy for failure: error vs exception models, Result/Either types, propagation vs recovery, retries/timeouts/circuit-breakers, fail-fast, error taxonomies, and user-facing vs internal errors.

---
name: error-handling
title: Error Handling Guidelines
version: 2.0
last_reviewed: 2026-06-05
kind: cross-cutting
tools: [language-agnostic]
requires: []
recommends:
  - logging
  - observability
  - secure-coding
  - tdd
provides:
  - error-taxonomy
  - retry-timeout-policy
  - result-types
  - fail-fast
---

> 🧭 Authored per [`CONVENTIONS.md`](guides://CONVENTIONS.md): this is the **canonical owner** of error-handling strategy. Language guides bind these rules to their syntax; they MUST NOT restate them. Adjacent concerns (logging, metrics, secrets, testing) are referenced, not duplicated.

---

## 0. Prerequisites & References

This guide owns the **failure model**. It deliberately stops at the boundary of four neighbours and references them instead of repeating their rules.

> 📎 **RECOMMENDED — fetch when the task touches them:**
> - [`logging.md`](guides://logging.md) — **how** to record errors (structured fields, levels, correlation IDs). This guide decides *which level* an error class maps to; `logging.md` owns the emit mechanics.
> - [`observability.md`](guides://observability.md) — error **metrics, traces, alerting** (error-rate SLOs, span status, exemplars). This guide names the signals; `observability.md` owns their export.
> - [`secure-coding.md`](guides://secure-coding.md) — **not leaking sensitive data** in errors (PII, secrets, stack traces, internal topology). This guide mandates the user/internal split; `secure-coding.md` owns the data-classification rules it enforces.
> - [`tdd.md`](guides://tdd.md) — **testing error paths** and the regression-test-before-fix workflow. This guide requires error paths be tested; `tdd.md` owns the Red-Green-Refactor mechanics.

> 📎 **SEE ALSO:** [`rest.md`](guides://rest.md) · [`grpc.md`](guides://grpc.md) · [`graphql.md`](guides://graphql.md) — protocol-level error/status mapping. [`microservices.md`](guides://microservices.md) — cross-service failure propagation. [`parallelism.md`](guides://parallelism.md) — cancellation & timeout semantics under concurrency.

---

## 1. Core Philosophies: ERROR-FIRST

Cross-cutting failure principles. Language idioms (Go `error`, Rust `Result`, Python exceptions, TS unions) are *bindings* of these — they belong in the language guide, not here.

- **E**xplicit: Failure modes are part of the API contract — encoded in the type signature (Result/Either, checked return, typed exception), never hidden behind a sentinel `null`, a swallowed `catch`, or a magic `-1`.
- **R**ecoverable-vs-not: Every error is classified once as *recoverable* (handle/retry/degrade) or *unrecoverable* (fail fast, alert) — see §2's taxonomy. The class, not the call site, decides the response.
- **R**ich context: Errors carry a stable machine code, a human message, and a preserved cause chain. Context is *added* on the way up (wrapping), never *replaced* (which destroys the root cause).
- **O**bservable: Every error is countable and traceable. This guide assigns levels/signals; emission is delegated to [`logging.md`](guides://logging.md) and [`observability.md`](guides://observability.md).
- **R**eportable safely: Users get an actionable, non-technical message + a correlation ID; internals (stack traces, queries, topology, PII) stay server-side per [`secure-coding.md`](guides://secure-coding.md).

**Verified Code**: Agent-generated code MUST satisfy every gate in §2 before delivery.

---

## 2. Requirements (MANDATORY, auditable)

RFC-2119 keywords. IDs `ERR-<TOPIC>-<NN>`. Each row has a binary gate; rows binding a neighbouring concern cite its owner. Topics: `TAX` taxonomy, `MODEL` error/exception model, `CTX` context/wrapping, `RES` resilience (retry/timeout/breaker), `FAIL` fail-fast, `MSG` user-facing, `LOG`/`OBS`/`SEC`/`TST` delegated.

| ID | Requirement | Verify | Gate |
|----|-------------|--------|------|
| ERR-TAX-01 | Every error MUST be classified *recoverable* or *unrecoverable*, and into a stable category (user / not-found / conflict / auth / transient / programming / config / system) with a stable machine code | code review / error-catalog doc | every error type maps to one category + code |
| ERR-TAX-02 | A single typed error hierarchy (or error enum) per service MUST exist; ad-hoc `throw "string"` / untyped maps MUST NOT be used | grep for raw string/`Error()`/`panic` without type | no untyped raises in domain code |
| ERR-MODEL-01 | Expected, domain-level failures (not-found, validation, conflict) SHOULD be modelled as values (Result/Either/`(T, error)`), not exceptions, in the language's idiom (binding: language guide) | review | domain APIs return typed failures |
| ERR-MODEL-02 | Exceptions/panics MUST be reserved for *unexpected* or *unrecoverable* conditions; control flow MUST NOT depend on catching exceptions for the happy-path branch | review | no exceptions-as-control-flow |
| ERR-MODEL-03 | `catch`/`except`/`recover` MUST be specific; a catch-all MUST re-raise or convert — never silently swallow (no empty `catch {}`, no bare `except:`) | lint (e.g. ruff `BLE`, eslint `no-empty`) / review | 0 silent swallows |
| ERR-CTX-01 | Errors MUST preserve the original cause when wrapped (`raise … from e`, `%w`, `cause:`); the chain MUST be reconstructable | review / test asserts `__cause__`/`errors.Is` | root cause recoverable |
| ERR-CTX-02 | `try`/error-scope blocks MUST be narrow — one fallible operation per block — so the failing step is unambiguous | review | no whole-function try blocks |
| ERR-RES-01 | Every outbound call to an external dependency (network/DB/queue) MUST have an explicit timeout; unbounded waits MUST NOT ship | grep for client calls without timeout/deadline | 100% have timeout |
| ERR-RES-02 | Retries MUST be applied ONLY to transient errors, MUST be bounded, and MUST use exponential backoff **with jitter**; non-idempotent operations MUST NOT be retried without an idempotency key | review / config | bounded + jitter + idempotent-only |
| ERR-RES-03 | Calls to a failure-prone external dependency MUST be protected by a circuit breaker (or equivalent bulkhead) so a downstream outage cannot exhaust callers | review / resilience config | breaker present on critical deps |
| ERR-RES-04 | Degraded paths (fallback/cache/partial response) SHOULD be defined for non-critical dependencies so their failure does not fail the whole request | review | non-critical failures isolated |
| ERR-FAIL-01 | Unrecoverable startup/config errors MUST fail fast at boot (validate config/secrets on startup) rather than failing lazily mid-request | startup smoke test | bad config → no boot |
| ERR-FAIL-02 | Programming errors (invariant/assertion violations) MUST NOT be caught-and-continued; they surface, alert, and (where safe) crash-restart | review | no swallowed invariant breaks |
| ERR-MSG-01 | User-facing messages MUST be actionable and free of internal detail; each error response MUST carry a correlation/request ID; validation errors MUST include field-level detail | response inspection / contract test | safe message + ID |
| ERR-MSG-02 | Every service MUST have a centralized top-level handler mapping error categories → protocol status (e.g. HTTP 4xx/5xx, gRPC codes) so no raw exception escapes the boundary (binding: `rest.md`/`grpc.md`/`graphql.md`) | integration test hitting error paths | no unmapped 500/leak |
| ERR-LOG-01 | Every error MUST be logged exactly once, at the level its category dictates (user/expected → INFO, transient/auth → WARN, unexpected/system → ERROR), via structured logging (owner: `logging.md`) | log inspection / review | correct level, no double-log |
| ERR-OBS-01 | Errors MUST be counted by category and surfaced on traces (span status/error attribute) for SLO/alerting (owner: `observability.md`) | metrics/trace check | error metric + span status present |
| ERR-SEC-01 | Stack traces, secrets, PII, SQL, and internal topology MUST NOT appear in any user-facing response (owner: `secure-coding.md`) | response scan / review | 0 sensitive leaks |
| ERR-TST-01 | Error/recovery paths (incl. retry exhaustion, breaker-open, timeout, fallback) MUST be covered by tests; each bug fix MUST add a failing regression test BEFORE the fix (owner: `tdd.md`) | test suite + coverage | error paths covered, exit 0 |

> **Forbidden**: silently swallowing exceptions; catching `Exception`/`Throwable`/bare-`except` then ignoring; using exceptions as happy-path control flow; retrying non-idempotent operations or non-transient errors; retries without a cap, backoff, or jitter; unbounded I/O without a timeout; leaking stack traces/PII/secrets to users; logging the same error at multiple layers; fixing a bug without a regression test first (violates `tdd.md`).

---

## 3. Error Taxonomy (the model this guide owns)

Classify once; the class drives level, retry, status, and alert. This replaces per-call-site decisions.

```
RECOVERABLE — handle, possibly retry, degrade gracefully
├── User errors          → reject with actionable message   (4xx, INFO)
│     validation · missing field · business-rule violation
├── Expected failures    → typed value result               (4xx, INFO)
│     not-found · duplicate/conflict · permission-denied
└── Transient errors     → retry w/ backoff+jitter, then breaker (5xx/503, WARN)
      timeout · service-unavailable · rate-limited · network blip

UNRECOVERABLE — fail fast, alert, do NOT retry
├── Programming errors   → surface + crash/restart          (500, ERROR + alert)
│     null/None deref · type error · broken invariant/assertion
├── Configuration errors → fail at boot (ERR-FAIL-01)        (boot failure, ERROR)
│     missing/invalid config · bad credentials · malformed settings
└── System failures      → fail fast + alert                (500/503, ERROR + alert)
      OOM · disk full · hardware/host loss
```

**Category → response matrix** (the single source of truth each language/protocol guide binds to):

| Category | Recoverable | Retry | Log level (→`logging.md`) | Typical HTTP / gRPC | User detail |
|---|---|---|---|---|---|
| Validation / user | yes | no | INFO | 400 / INVALID_ARGUMENT | field-level detail |
| Authentication | yes | no | WARN | 401 / UNAUTHENTICATED | "log in to continue" |
| Authorization | yes | no | WARN | 403 / PERMISSION_DENIED | "not permitted" |
| Not-found | yes | no | INFO | 404 / NOT_FOUND | generic, no detail |
| Conflict / duplicate | yes | no | INFO | 409 / ALREADY_EXISTS | generic |
| Rate-limited | yes | yes (respect `Retry-After`) | WARN | 429 / RESOURCE_EXHAUSTED | "try again later" |
| Transient / dependency down | yes | yes (backoff+jitter, then breaker) | WARN | 503 / UNAVAILABLE | "temporarily unavailable" |
| Programming / invariant | no | no | ERROR + alert | 500 / INTERNAL | generic + correlation ID |
| Config / system | no | no | ERROR + alert | 500 / 503 | generic + correlation ID |

> The hierarchy is a *model*, not a code dump. Each language guide shows its idiomatic encoding: Python typed-exception tree, Go sentinel + `errors.As`, Rust `enum` + `thiserror`, TS discriminated `AppError`. Those bindings live there, not here.

---

## 4. Error vs. Exception Model — choosing the encoding

Two encodings; pick per failure class, not per language fashion.

- **Errors-as-values** (Result/Either/`(T, error)`): expected, recoverable, domain failures. They are part of the signature, force the caller to handle them, and read linearly. Preferred for domain/use-case layers (ERR-MODEL-01). Idioms: Rust `Result`/`?`, Go `(T, error)`, TS discriminated union, Python return-or-raise at the edge.
- **Exceptions/panics**: unexpected or unrecoverable conditions, and crossing wide layers where threading a value is impractical. Reserve for the abnormal (ERR-MODEL-02). One centralized boundary handler converts them to safe responses (ERR-MSG-02).

Rules that hold regardless of encoding:
- Never use exceptions for the happy path or normal control flow.
- A function's failure modes are documented/typed; callers don't guess.
- At the I/O boundary, convert between models deliberately (e.g. wrap a thrown driver error into a typed domain `Result`), preserving the cause (ERR-CTX-01).

---

## 5. Propagation vs. Recovery

Decide at each layer: **handle here**, **wrap and propagate**, or **let pass**.

- **Recover** only where you have enough context to do something meaningful (translate a not-found into a domain result, supply a fallback, retry a transient). Recovering "everything" everywhere hides bugs.
- **Wrap & propagate** when you can add context but not resolve: attach the operation + identifiers, preserve the cause, re-raise/return (ERR-CTX-01). Add context *once per layer*, not per line.
- **Let pass** when the current layer adds nothing — don't catch just to re-raise unchanged.
- **Narrow scope** (ERR-CTX-02): wrap a single fallible operation so the failing step is unambiguous; never a whole function.
- **Side-effect isolation**: a failure in a non-essential side effect (e.g. notification after a successful write) is logged and degraded, not propagated to fail the primary operation (ERR-RES-04).

---

## 6. Resilience: Timeouts, Retries, Circuit Breakers

Owned here; language guides bind to a library (e.g. Tenacity, `failsafe`, resilience4j, Polly, Go `context`).

- **Timeouts first (ERR-RES-01)**: every outbound call gets a deadline. Set timeouts to a realistic budget; propagate a *deadline/cancellation* through the call chain (a 30s request must not spawn a 60s downstream wait). An unbounded wait is a latent outage.
- **Retries (ERR-RES-02)**: transient-only, bounded attempts, **exponential backoff with jitter** (jitter prevents synchronized retry storms / thundering herd). Honour `Retry-After`/rate-limit hints. Retry only idempotent operations, or guard with an idempotency key. Cap total retry time within the caller's deadline. Never retry validation/auth/not-found.
- **Circuit breakers (ERR-RES-03)**: CLOSED → (failures ≥ threshold) → OPEN (fast-fail, no calls) → after cooldown HALF-OPEN (probe) → CLOSED on success. Protects callers from a stuck dependency and gives it room to recover; combine with retries (breaker wraps the retried call).
- **Bulkheads & load-shedding**: isolate resource pools per dependency so one slow downstream can't exhaust all threads/connections; shed load (429) instead of queueing unboundedly.
- **Graceful degradation (ERR-RES-04)**: define fallbacks (cached/stale data, partial response, default) for non-critical dependencies so their failure degrades, not denies.

> Concurrency-level cancellation/deadline propagation semantics: see [`parallelism.md`](guides://parallelism.md). Cross-service propagation of these policies: see [`microservices.md`](guides://microservices.md).

---

## 7. Fail-Fast

- **Validate at boot (ERR-FAIL-01)**: config, secrets, schema, and connectivity preconditions are checked at startup; an invalid environment prevents the process from serving — not a 500 on the first request hours later.
- **Assert invariants (ERR-FAIL-02)**: programming errors (broken invariants, impossible states) must surface loudly — alert and, where safe, crash so a supervisor restarts a clean process. Do not catch-and-continue past corruption; a fast crash beats serving corrupt state.
- **No defensive over-catching**: don't wrap invariant violations in a generic handler that returns 200/empty — that masks bugs the taxonomy classifies as unrecoverable.

---

## 8. User-Facing vs. Internal Errors

The split this guide mandates (ERR-MSG-01, ERR-SEC-01); the *data-classification* rules behind "sensitive" are owned by [`secure-coding.md`](guides://secure-coding.md).

- **User-facing**: actionable, non-technical, stable per category, plus a correlation/request ID for support. Validation errors carry structured field-level detail. Example contract:
  ```json
  { "error": "VALIDATION_ERROR",
    "message": "Please correct the highlighted fields.",
    "request_id": "req_01H...",
    "fields": { "email": ["Enter a valid email address."],
                "password": ["Must be at least 12 characters."] } }
  ```
- **Internal**: full type, cause chain, stack trace, parameters — logged server-side per [`logging.md`](guides://logging.md), surfaced as metrics/traces per [`observability.md`](guides://observability.md). NEVER returned to the client.
- **Boundary handler (ERR-MSG-02)**: a single top-level handler per service maps category → status and serializes the safe shape; no raw exception, stack trace, SQL, secret, PII, or topology hint escapes (ERR-SEC-01). Protocol-specific status mapping: bind to [`rest.md`](guides://rest.md) / [`grpc.md`](guides://grpc.md) / [`graphql.md`](guides://graphql.md).

---

## 9. Logging, Metrics & Testing of Errors (delegated)

This guide assigns *what* and *which level/signal*; the *how* lives in the owners — do not restate them.

- **Logging (ERR-LOG-01 → [`logging.md`](guides://logging.md))**: each error logged once, structured, at the category's level (§3 matrix). Avoid double-logging the same error at multiple layers (log-and-throw is an anti-pattern: log at the boundary *or* propagate, not both).
- **Observability (ERR-OBS-01 → [`observability.md`](guides://observability.md))**: error count by category feeds error-rate SLOs/alerts; the current span gets error status + attributes; correlation/request ID ties logs↔traces.
- **Secure data (ERR-SEC-01 → [`secure-coding.md`](guides://secure-coding.md))**: error payloads/logs are scrubbed of secrets/PII per its classification rules.
- **Testing (ERR-TST-01 → [`tdd.md`](guides://tdd.md))**: error/recovery paths are first-class tests — assert the right typed error, status, retry-exhaustion, breaker-open, timeout, and fallback. Every bug fix begins with a failing regression test (Red-Green-Refactor mechanics owned by `tdd.md`).

---

## Deployment Checklist

Generated from §2 — one box per requirement ID. No new requirements here.

- [ ] ERR-TAX-01/02 — every error classified + stable code; single typed hierarchy, no untyped raises
- [ ] ERR-MODEL-01/02/03 — expected failures as values; exceptions only for the abnormal; no silent swallow
- [ ] ERR-CTX-01/02 — cause chain preserved; try-scopes narrow
- [ ] ERR-RES-01 — every external call has a timeout/deadline
- [ ] ERR-RES-02 — retries bounded, backoff+jitter, transient + idempotent only
- [ ] ERR-RES-03 — circuit breaker on critical dependencies
- [ ] ERR-RES-04 — graceful degradation / fallback for non-critical deps
- [ ] ERR-FAIL-01/02 — fail-fast on bad config at boot; invariant breaks not swallowed
- [ ] ERR-MSG-01/02 — safe actionable messages + correlation ID; centralized boundary mapping
- [ ] ERR-LOG-01 — logged once at correct level (see `logging.md`)
- [ ] ERR-OBS-01 — error metrics + span status (see `observability.md`)
- [ ] ERR-SEC-01 — no stack traces/PII/secrets/topology leaked (see `secure-coding.md`)
- [ ] ERR-TST-01 — error paths tested; bug fixes have a regression test first (see `tdd.md`)
- [ ] Agent ran the project's gate commands and documented any fixes

---
**End of Error Handling Guidelines**
