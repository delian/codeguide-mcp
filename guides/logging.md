# Logging Guidelines
Mandatory, language-agnostic standards for application logging: structured/JSON events, log levels, correlation/trace IDs, context propagation, log hygiene, sampling, and retention.

---
name: logging
title: Logging Guidelines
version: 2.0
last_reviewed: 2026-06-05
kind: cross-cutting
tools: [structured-logging, json-logs, opentelemetry-logs]
requires: []
recommends:
  - observability
  - secure-coding
  - error-handling
provides:
  - structured-logging
  - log-levels
  - correlation-ids
  - log-hygiene
---

> 🧭 Authored per [`CONVENTIONS.md`](guides://CONVENTIONS.md): this is the canonical owner of **structured logging**. Other guides reference it instead of restating log rules. It owns log levels, structured events, correlation IDs, context propagation, hygiene, sampling, and retention — and references the owners of metrics/tracing, secrets/PII, and error strategy rather than duplicating them.

---

## 0. Prerequisites & References

Logging is **one pillar of observability**, not all of it. This guide owns log events; it does not restate how metrics or traces work, how secrets are classified, or what the error-handling strategy is.

> 📎 **RECOMMENDED — fetch when the task touches them:**
> - [`observability.md`](guides://observability.md) — metrics, distributed tracing, the three-pillars model. *Logging binding: a log line carries `trace_id`/`span_id` so logs join the trace; metrics and span instrumentation are owned there, not here.*
> - [`secure-coding.md`](guides://secure-coding.md) — what counts as a secret/PII, classification, retention-of-secrets policy. *Logging binding: never serialize those classes into a log event (§2 LOG-HYG rows).*
> - [`error-handling.md`](guides://error-handling.md) — error strategy, wrapping, propagation, when to retry vs. fail. *Logging binding: an error is logged once at the boundary that handles it, with cause chain and stack (§2 LOG-ERR-01); the decision to swallow/re-raise belongs there.*

> 📎 **SEE ALSO:** [`env-config.md`](guides://env-config.md) (log level/format from config) · [`microservices.md`](guides://microservices.md) (cross-service propagation) · [`kubernetes.md`](guides://kubernetes.md) · [`docker-compose.md`](guides://docker-compose.md) (stdout collection).

Language guides (e.g. [`python.md`](guides://python.md), [`go.md`](guides://go.md), [`nodejs.md`](guides://nodejs.md), [`java.md`](guides://java.md)) own only the *binding* — which library and API implements these rules in that ecosystem.

---

## 1. Core Philosophies: LOG-FIRST

Principles unique to logging. Tracing/metrics philosophy comes from `observability.md`; secret classification from `secure-coding.md`.

- **L**evels mean something: every event maps to exactly one level by the §3 decision matrix; level is the primary alerting and retention axis.
- **O**bservable intent: a log answers *what happened and why*, in fields — not prose. Log facts and identifiers, not narration.
- **G**reppable: emit **structured key/value events** (JSON in prod), never interpolated free-text. Stable event names, stable field names.
- **F**ast & non-blocking: logging MUST NOT add latency on the hot path — no synchronous network/disk writes in request handling (§2 LOG-PERF).
- **I**dentifiable: every request-scoped log carries a propagated correlation/`trace_id`, set once at the boundary via context, not threaded by hand.
- **R**edacted by default: secrets/PII never reach a log sink — enforced by a pipeline processor, not developer discipline (see `secure-coding.md`).
- **S**tandardized: one schema, one timestamp format (RFC 3339 / ISO 8601, UTC), identical field names across all services.
- **T**o stdout: apps log to **stdout/stderr as a stream**; rotation, shipping, retention, and storage are the platform's job, not the app's.

**Verified Code**: agent-generated logging MUST satisfy every gate in §2 before delivery.

---

## 2. Requirements (MANDATORY, auditable)

RFC-2119 keywords. IDs `LOG-<TOPIC>-<NN>`. Each row has a binary gate; rows binding a shared rule cite its owner.

| ID | Requirement | Verify | Gate |
|----|-------------|--------|------|
| LOG-FMT-01 | Production logs MUST be machine-parseable structured records (one JSON object per line) | Pipe sample to `jq .` | every line parses |
| LOG-FMT-02 | Free-text/printf-style log calls (string concatenation/interpolation as the message) MUST NOT be used; pass fields as key/values | lint / grep for `log.*+`, f-string in log call | 0 hits |
| LOG-FMT-03 | Timestamps MUST be RFC 3339 / ISO 8601 in UTC with millisecond precision | inspect emitter config / sample | matches `…Z` |
| LOG-FMT-04 | Every record MUST carry `timestamp`, `level`, `service`, `message`/`event` | `jq 'select(.level==null or .service==null)'` | empty |
| LOG-LVL-01 | Each event MUST use exactly the level defined by the §3 matrix | review against §3 | no misclassification |
| LOG-LVL-02 | Runtime level MUST be config-driven (env/config), not hardcoded; default `INFO` in prod (see `env-config.md`) | grep for hardcoded level; check config | level from config |
| LOG-LVL-03 | `DEBUG`/`TRACE` MUST be disabled by default in production | inspect prod config | not enabled |
| LOG-CID-01 | Every request-scoped log MUST include a propagated correlation id (`trace_id`/`correlation_id`) | sample request logs | id present & stable per request |
| LOG-CID-02 | Correlation/trace context MUST be carried implicitly (context/ALS/MDC), not passed as a parameter through business code | review call sites | no logger threading |
| LOG-CID-03 | Inbound trace context MUST be honored and outbound calls MUST propagate it (W3C `traceparent`/`X-Correlation-ID`) (see `observability.md`) | trace a 2-service call | same id end-to-end |
| LOG-HYG-01 | Secrets MUST NOT be logged (passwords, tokens, keys, connection strings) (see `secure-coding.md`) | redaction test + grep | 0 secrets in sink |
| LOG-HYG-02 | PII MUST NOT be logged in cleartext; use IDs, masked, or hashed values (see `secure-coding.md`) | redaction test | 0 raw PII |
| LOG-HYG-03 | A centralized redaction processor MUST run in the pipeline (deny-list of fields + value scanning), not be left to call sites | inspect pipeline config | processor present & tested |
| LOG-ERR-01 | A handled error MUST be logged once, at the handling boundary, with `error_type`, `error_message`, cause chain, and stack (see `error-handling.md`) | review; assert single log per error | no log-and-rethrow dup |
| LOG-PERF-01 | Logging MUST be non-blocking on the request path; no synchronous remote sink writes inline | review handler/sink config | async/buffered sink |
| LOG-PERF-02 | High-volume/diagnostic logs MUST be sampled or rate-limited (not per-event in hot loops) | review hot paths | sampling applied |
| LOG-OUT-01 | Apps MUST write logs to stdout/stderr; the app MUST NOT manage files, rotation, or shipping | inspect handlers | stream-only sink |
| LOG-RET-01 | Retention MUST be defined per level/environment at the platform (not unbounded) | inspect retention policy | policy exists |

> **Forbidden**: logging a secret or raw PII (violates `secure-coding.md`); log-and-rethrow that double-logs the same error (violates `error-handling.md`); free-text-interpolated messages; hardcoded log level; synchronous remote logging on the hot path; the app writing/rotating its own log files; re-implementing metrics or span creation here (owned by `observability.md`).

---

## 3. Log Levels (the level matrix)

Level is the contract between developers and operators. One event → one level. This matrix is canonical; language guides do not redefine it.

| Level | Trigger | Action required | Examples | Prod volume |
|-------|---------|-----------------|----------|-------------|
| **FATAL/CRITICAL** | Process cannot continue and will exit | Immediate page | cannot bind port; OOM; corrupt schema after failed migration; required config absent at boot | 0–1 per incident |
| **ERROR** | An operation failed; the request/job could not be fulfilled | Investigate within alert SLA | charge failed after retries; downstream 5xx after retry budget; unhandled exception at a boundary | < 0.1% of requests (ideal) |
| **WARN** | Unexpected but handled; degraded, not failed | Trend-watch, next business day | retry succeeded; cache miss → DB fallback; deprecated endpoint hit; quota at 80% | moderate (watch trends) |
| **INFO** | Normal, business-relevant milestone | None | service started; user logged in; order placed; scheduled job processed N records | a few per request/event |
| **DEBUG** | Technical detail for diagnosis | Viewed only while investigating | resolved config path; cache key/hit; chosen branch; outbound call target | off by default in prod |
| **TRACE** | Extremely fine execution flow | Local/dev only | function entry/exit; loop iteration; raw payload bytes | never in prod |

Decision tree: *Will the process exit?* → FATAL. *Did an operation fail?* → ERROR. *Unexpected but handled?* → WARN. *Normal milestone?* → INFO. *Diagnostic detail?* → DEBUG. *Byte/iteration-level?* → TRACE.

Rules: choose the level by **consequence**, not by how much you care; never use ERROR for an expected/handled condition (that hides real failures); WARN is for the alertable trend, not the satisfying log. Logging an error does **not** decide whether to retry/swallow/re-raise — that is `error-handling.md`.

---

## 4. Structured Event Schema

A log record is a flat (or shallowly grouped) JSON object with **stable names**. Treat the schema as an API: renaming a field breaks dashboards and alerts.

```json
{
  "timestamp": "2026-06-05T10:30:45.123Z",
  "level": "INFO",
  "service": "order-service",
  "version": "1.2.3",
  "environment": "production",
  "event": "order_placed",
  "message": "Order placed successfully",
  "trace_id": "4bf92f3577b34da6a3ce929d0e0e4736",
  "span_id": "00f067aa0ba902b7",
  "user_id": "USR-456",
  "order_id": "ORD-123",
  "duration_ms": 245
}
```

**Field tiers** (names are normative; values illustrative):

- **Always**: `timestamp` (RFC 3339 UTC), `level`, `service`, `version`, `environment`, `event` (stable machine name) + `message` (human gloss).
- **Request-scoped**: `trace_id`, `span_id`, `correlation_id`/`request_id`, `user_id` (or other non-PII subject id), `session_id`.
- **Errors**: `error_type`, `error_message`, `error_code`, `stack_trace`, `cause`. The error *strategy* is owned by `error-handling.md`; these fields are how an error is *represented* in a log.
- **Performance**: `duration_ms`, `status_code`, `method`, `path`.

Schema discipline:
- Prefer a stable `event` name (e.g. `payment_declined`) over encoding facts in `message`; humans read `message`, machines group by `event`.
- Use consistent units and suffixes (`_ms`, `_bytes`, `_count`); never mix seconds and milliseconds under one name.
- Bind service/version/environment **once** at logger init; bind request fields **once** at the boundary; do not re-pass them per call.
- Group only when it aids querying (e.g. an `http` group); deep nesting hurts indexing in most sinks.

---

## 5. Correlation & Context Propagation

This is the highest-value, logging-owned mechanic: a single id that joins every log of one logical request, across threads and services. Trace/span *semantics* and sampling of traces are owned by `observability.md`; here we own getting the id **into every log line implicitly**.

### A. Set once, at the boundary
At the inbound edge (HTTP/gRPC/queue consumer), extract an existing correlation id from headers, or mint one; store it in the ambient request context, and bind it to the logger context so **every** subsequent log inherits it. Never accept logs that thread a `logger`/`trace_id` argument through business functions (LOG-CID-02).

Ambient-context primitive per ecosystem (binding owned by the language guide):
- Python: `contextvars` (e.g. `structlog.contextvars`) bound in middleware.
- Go: `context.Context` carrying the id; a context-aware `*slog.Logger`.
- Node/TS: `AsyncLocalStorage` holding the request-scoped logger.
- JVM: SLF4J **MDC** (thread/scope-local).

### B. Align with W3C Trace Context
Prefer the standard `traceparent`/`tracestate` headers and OpenTelemetry's `trace_id`/`span_id` so logs and traces share one id space and can be pivoted between in the backend. If you also accept a human-friendly `X-Correlation-ID`, map it onto the same field; do not invent a third id.

### C. Propagate outbound
Any outbound call (HTTP client, message publish, job enqueue) MUST forward the correlation/trace headers so the next service continues the same id (LOG-CID-03). Cross-service propagation patterns live in `microservices.md`; the **logging requirement** is that the id is unbroken end-to-end.

> OpenTelemetry note: when an OTel SDK is present, install its log-correlation hook so emitted logs are stamped with the active span's `trace_id`/`span_id` automatically. The SDK setup itself is `observability.md`; logging just consumes the active context.

---

## 6. Log Hygiene — Secrets & PII

`secure-coding.md` owns *what* is a secret/PII and the data-classification policy. This guide owns *that it never lands in a log sink* and *how the pipeline guarantees it*.

- **Never log** credentials, tokens (JWT/refresh/session), API/private keys, connection strings, full PANs, CVV, SSN/national IDs, full address, DOB, health/biometric data (LOG-HYG-01/02).
- **Log the safe surrogate instead**: a stable id (`user_id`), a masked value (`****1234`, `j***@example.com`), a non-reversible hash for correlation, or a non-identifying derivative (e.g. email domain).
- **Enforce centrally** (LOG-HYG-03): a redaction processor in the logging pipeline runs on every record before the sink — a deny-list of field names (case-insensitive) plus value scanning for embedded secrets in URLs/connection strings/`Authorization` headers. Call-site discipline is a backstop, not the control.
- **Test it**: a redaction unit test feeds known secrets/PII through the pipeline and asserts they are absent from output — this is the gate for LOG-HYG-01/02/03.
- **Defense in depth**: redact at the source even though the sink may also scrub; a secret should never have existed in the stream.

Implementation is a single pipeline stage (structlog processor / `slog.Handler` wrapper / pino redact paths / Logback masking converter) — the language guide supplies the exact hook.

---

## 7. Performance: Async, Sampling, Rate-Limiting

Observability MUST NOT degrade the system it observes (LOG-PERF-01/02).

- **Non-blocking sinks**: write to stdout/an in-process buffer; ship asynchronously. Never make a synchronous network call to a log aggregator on the request path.
- **Guard expensive payloads**: gate costly field construction behind a level check so it is skipped when the level is disabled.
- **Sample high-volume events**: for chatty diagnostic logs, emit a fraction (e.g. 1–10%) and stamp the record with the sample rate so downstream counts can be scaled. Always keep WARN/ERROR/FATAL at 100%.
- **Rate-limit repeats**: collapse log storms (e.g. "logged N times in the last 10s") instead of one line per event in a tight loop.
- **No logging inside hot loops** at INFO+; aggregate and log a summary.

Trace sampling (head/tail) is owned by `observability.md` — keep log sampling and trace sampling decisions aligned but don't restate trace policy here.

---

## 8. Error Logging

The error-handling *strategy* (wrap, propagate, retry, fail) is owned by `error-handling.md`. The **logging** rules:

- Log a handled error **once**, at the boundary that actually handles it, with `error_type`, `error_message`, `error_code`, the full `cause` chain, and `stack_trace` (LOG-ERR-01). Do **not** log-and-rethrow at every layer — that produces N copies of one failure and pollutes error-rate metrics.
- Attach the request context (already bound via §5) plus operation-specific fields (`order_id`, `operation`).
- Stack traces: include for ERROR/FATAL. They are generally not needed (and noisy) for WARN.
- Use the level matrix: a caught, recovered condition is WARN, not ERROR.
- Deriving alerting from error logs (rates, thresholds) is an alerting/observability concern — define those rules in `observability.md`, not as bespoke in-app counters.

---

## 9. Output, Shipping & Retention

- **Twelve-factor logs** (LOG-OUT-01): the app writes an event stream to stdout/stderr and is done. It does not open files, rotate, or push to Elasticsearch/Loki/CloudWatch directly. The platform (sidecar/agent/driver — promtail, Fluent Bit, Filebeat, the container runtime, CloudWatch agent) collects, parses, ships, and stores.
- **Parsing**: because records are already JSON (LOG-FMT-01), collectors index fields directly with no fragile regex; map `level`, `service`, `trace_id` to searchable labels/fields.
- **Retention** (LOG-RET-01): set per level and environment at the platform, e.g. prod ERROR 90d / WARN 30d / INFO 14d / DEBUG off; tighten in lower environments. Retention is a storage/compliance policy, configured where logs are stored — not in application code.
- **Local files only as a fallback**: if a runtime genuinely cannot stream (rare), use the platform's rotating handler — but prefer fixing the deployment so stdout is collected.

---

## Deployment Checklist

Generated from §2 — one box per requirement ID. No new requirements.

- [ ] LOG-FMT-01/02 — JSON-per-line, no interpolated free-text messages
- [ ] LOG-FMT-03 — timestamps RFC 3339 / ISO 8601, UTC, ms precision
- [ ] LOG-FMT-04 — `timestamp`/`level`/`service`/`event` present on every record
- [ ] LOG-LVL-01/02/03 — correct levels; level from config; DEBUG/TRACE off in prod
- [ ] LOG-CID-01/02 — correlation/trace id on every request log, carried via context (not threaded)
- [ ] LOG-CID-03 — inbound honored, outbound propagated end-to-end (see `observability.md`)
- [ ] LOG-HYG-01/02 — no secrets, no raw PII (see `secure-coding.md`)
- [ ] LOG-HYG-03 — centralized redaction processor present and unit-tested
- [ ] LOG-ERR-01 — errors logged once at the boundary with cause chain + stack (see `error-handling.md`)
- [ ] LOG-PERF-01/02 — non-blocking sink; high-volume logs sampled/rate-limited
- [ ] LOG-OUT-01 — app streams to stdout; platform handles rotation/shipping
- [ ] LOG-RET-01 — per-level/per-env retention policy defined at the platform

---
**End of Logging Guidelines**
