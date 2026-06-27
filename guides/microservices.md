# Microservices Architecture Guidelines
Mandatory architectural standards for microservices: service boundaries, inter-service communication, distributed data, sagas, and resilience topology. Language- and runtime-agnostic; owns architecture, references transport/messaging/deployment/observability guides.

---
name: microservices
title: Microservices Architecture Guidelines
version: 2.0
last_reviewed: 2026-06-05
kind: cross-cutting
tools: []
requires: []
recommends:
  - hexagonal
  - observability
  - error-handling
  - rest
  - grpc
  - kafka
  - kubernetes
  - secure-coding
provides:
  - service-boundaries
  - inter-service-comms
  - sagas
  - resilience-patterns
---

> 🧭 Authored per [`CONVENTIONS.md`](guides://CONVENTIONS.md): shared concerns are referenced, not restated. This guide owns the *distributed-system topology* — how services are split, how they talk, how they stay consistent, and how they survive failure. Transport syntax, messaging internals, deployment mechanics, observability instrumentation, and per-service internal structure live in their canonical owners.

---

## 0. Prerequisites & References

This guide describes architecture-level decisions. The mechanics it relies on are owned elsewhere — fetch them when the task touches them. This guide does not restate their rules.

> 📎 **RECOMMENDED — fetch when the task touches them:**
> - [`hexagonal.md`](guides://hexagonal.md) — the internal structure of *each* service (domain/application/adapters, ports, dependency inversion). A microservice is a hexagon; this guide never re-describes its layers.
> - [`rest.md`](guides://rest.md) · [`grpc.md`](guides://grpc.md) — synchronous transport design (resource modeling, status codes, `.proto` contracts, streaming, versioning). This guide chooses *when* to use each; the owners define *how*.
> - [`kafka.md`](guides://kafka.md) — asynchronous messaging mechanics (topics, partitions, delivery/ordering guarantees, exactly-once, consumer groups). This guide owns the event-driven *topology*; Kafka owns the broker.
> - [`observability.md`](guides://observability.md) — metrics (RED/USE), distributed tracing, SLI/SLO, dashboards, alerting, health-check semantics. This guide says *what to trace across boundaries*; the owner says *how to instrument*.
> - [`error-handling.md`](guides://error-handling.md) — retry, timeout, and circuit-breaker *strategy* and error taxonomies. This guide binds those policies to the service topology only.
> - [`kubernetes.md`](guides://kubernetes.md) — deployment, probes, autoscaling, rollout strategies, resource limits. This guide specifies deployment *requirements*; the owner specifies manifests.
> - [`secure-coding.md`](guides://secure-coding.md) — input validation, secrets management, supply-chain, authn/authz hygiene. This guide adds only the zero-trust *between-services* binding.

> 📎 **SEE ALSO:** [`architectures.md`](guides://architectures.md) (is microservices even the right style?) · [`istio.md`](guides://istio.md) (service mesh) · [`oauth.md`](guides://oauth.md) (token issuance) · [`env-config.md`](guides://env-config.md) · [`ci-cd.md`](guides://ci-cd.md) · [`semver.md`](guides://semver.md) (API versioning) · [`e2e-testing.md`](guides://e2e-testing.md) · [`designpatterns.md`](guides://designpatterns.md)

---

## 1. Core Philosophies

Microservices-specific principles only. TDD, security, error strategy, transport, and observability instrumentation come from the §0 references — not restated here.

- **Boundaries before code.** A service exists because it owns a **bounded context** with its own ubiquitous language and lifecycle — not because of a technical layer. Get the seam wrong and every other decision compounds the mistake.
- **Independent deployability is the litmus test.** If two services must release together, they are one service wearing two hats (a *distributed monolith*). Autonomy of deployment, scaling, and data is the definition — size is incidental.
- **Data is owned, never shared.** Each service is the sole writer of its data store. Other services reach it only through its API or its published events. A shared database is the single most common cause of coupling.
- **The network is hostile.** Every remote call can be slow, fail, duplicate, or arrive out of order. Resilience (timeout → retry → circuit-breaker → fallback) and idempotency are mandatory, not optional polish.
- **Embrace eventual consistency.** Cross-service atomic transactions do not exist; use sagas and the outbox pattern. Model business invariants that tolerate temporary inconsistency.
- **Conway's Law is a constraint, not a footnote.** Architecture mirrors org structure: one team owns each service end-to-end (build, run, on-call).

**Verified Architecture**: Agent-generated systems MUST satisfy every gate in §2 before delivery.

---

## 2. Requirements (MANDATORY, auditable)

RFC-2119 keywords. IDs `MS-<TOPIC>-<NN>`. Each row has a binary gate; rows binding a shared rule cite its owner. `ARCH`=boundaries, `COMM`=communication, `DATA`=data ownership, `OBS`=observability, `SEC`=security.

| ID | Requirement | Verify | Gate |
|----|-------------|--------|------|
| MS-ARCH-01 | Each service MUST map to exactly one bounded context / business capability | architecture review against context map | one capability per service |
| MS-ARCH-02 | Each service MUST be independently deployable (no lock-step release) | deploy one service alone in CI; no coordinated rollout required | succeeds in isolation |
| MS-ARCH-03 | Each service MUST be owned by a single team | service catalog has an `owner` per service | every service owned |
| MS-DATA-01 | Each service MUST be the sole writer of its data store; no shared schema | review DB credentials/grants per service | no cross-service writes |
| MS-DATA-02 | Cross-service writes MUST use a saga + transactional outbox, never a distributed XA transaction | review write paths; outbox table present | no 2PC; outbox used |
| MS-COMM-01 | Every synchronous inter-service call MUST set an explicit timeout (see `error-handling.md`) | grep clients / contract review | no unbounded calls |
| MS-COMM-02 | Synchronous calls on critical paths MUST be wrapped in a circuit breaker with a fallback (see `error-handling.md`) | review client config / mesh policy | breaker + fallback present |
| MS-COMM-03 | Retries MUST target only idempotent operations and use exponential backoff + jitter (see `error-handling.md`) | review retry policy | no retry on non-idempotent |
| MS-COMM-04 | Async consumers MUST be idempotent (dedupe by event/message id) | contract review + duplicate-delivery test | duplicates absorbed |
| MS-COMM-05 | Inter-service APIs MUST be contract-tested before deploy (consumer-driven) | run contract suite (e.g. Pact) in CI | provider verifies contracts |
| MS-COMM-06 | API changes MUST be backward-compatible or versioned (see `semver.md`, `rest.md`/`grpc.md`) | contract diff / proto compat check | no breaking change unversioned |
| MS-COMM-07 | Services MUST be located via service discovery, never hardcoded host:port | grep config for literal IPs/ports | no hardcoded endpoints |
| MS-OBS-01 | Trace context MUST propagate across every service boundary (see `observability.md`) | trace a request end-to-end | single connected trace |
| MS-OBS-02 | Each service MUST expose liveness, readiness, and startup health endpoints (see `kubernetes.md`, `observability.md`) | probe `/health/{live,ready,startup}` | correct 200/503 |
| MS-SEC-01 | Service-to-service traffic MUST be mutually authenticated and encrypted (mTLS) (see `secure-coding.md`, `istio.md`) | inspect mesh/TLS config | mTLS enforced |
| MS-SEC-02 | All external traffic MUST enter through an API gateway performing authn + rate limiting (see `secure-coding.md`, `oauth.md`) | review ingress / gateway config | no direct service ingress |

> **Forbidden**: a shared database across services (violates MS-DATA-01); a distributed XA/2PC transaction across services (MS-DATA-02); any remote call without a timeout (MS-COMM-01); retrying a non-idempotent operation (MS-COMM-03); hardcoded service URLs (MS-COMM-07); a service that cannot be deployed without releasing another (MS-ARCH-02). Test-first development and bug-regression tests are governed by [`tdd.md`](guides://tdd.md) and apply to all service code.

---

## 3. Service Boundaries (owned)

The most consequential decision in the architecture. A boundary is a **bounded context** (DDD): a cohesive model with one ubiquitous language, one owning team, and one data store.

### Drawing the boundary
- **Align to business capability, not technical layer.** `OrderService`, `PaymentService`, `InventoryService` — never `ValidationService` or `DatabaseService`. The name is a business noun.
- **Use the bounded context as the unit.** Two contexts that share no invariants and change for different reasons are two services. Two pieces of data that must change together in one transaction belong in *one* service.
- **Decompose by these axes**, in priority order: bounded context → rate of change → team ownership → scaling profile. A subdomain that changes daily should not be welded to one that changes yearly.
- **Right-size by autonomy, not lines of code.** "Just right" = a single team understands it, owns its data, deploys it independently. Too small (nano-service) multiplies network hops and distributed-failure surface for no autonomy gain; too large (mini-monolith) hosts multiple contexts and forces coordinated change.

### Anti-corruption layer
When a context must consume another's model, translate at the edge with an **anti-corruption layer** (an inbound adapter — structure owned by [`hexagonal.md`](guides://hexagonal.md)) so a neighbor's model never leaks into your domain.

### Boundary validation checklist
- [ ] Single business capability; service name is a business noun.
- [ ] Changes to this capability rarely require changes to others.
- [ ] Owns its data store; no other service writes it (MS-DATA-01).
- [ ] One team owns build/run/on-call (MS-ARCH-03).
- [ ] Deployable and scalable independently (MS-ARCH-02).
- [ ] Not a single CRUD table dressed as a service.

---

## 4. Inter-Service Communication (owned topology)

This guide owns *which communication style to use and how services are wired*. The transport mechanics belong to the owners: REST/gRPC ([`rest.md`](guides://rest.md), [`grpc.md`](guides://grpc.md)) for synchronous, Kafka ([`kafka.md`](guides://kafka.md)) for asynchronous messaging.

### Synchronous vs. asynchronous — the decision
| Need | Choose | Owner of mechanics |
|------|--------|--------------------|
| Caller blocks for an immediate answer (queries, validations) | Synchronous request/response | `rest.md` (external/public), `grpc.md` (internal, high-throughput, streaming) |
| Fire-and-forget, decoupling, fan-out to many consumers | Asynchronous events (pub/sub) | `kafka.md` |
| Work distribution / load leveling, exactly-one consumer | Asynchronous queue | `kafka.md` (or broker of record) |

**Default to asynchronous for cross-context state propagation.** Synchronous coupling compounds latency (sum of hops) and erodes availability (product of uptimes). Reserve synchronous calls for genuine query/command paths that need an answer now.

### Event-driven topology (owned)
- **Events are immutable, past-tense business facts**: `OrderPlaced`, `PaymentCaptured`, `InventoryReserved` — never imperative (`CreateOrder`). The producer asserts a fact; consumers decide what to do.
- **Carry correlation and causation IDs** on every event so a business transaction is traceable across services (the *propagation* is owned here; the *tracing backend* by [`observability.md`](guides://observability.md)).
- **Choreography vs. orchestration** for multi-service flows — see §5.
- **Consumers MUST be idempotent** (MS-COMM-04): dedupe on event id, because at-least-once delivery is the realistic default. Ordering guarantees are a Kafka concern; design consumers to tolerate reordering where the topic does not guarantee it.
- **Schema evolution**: events are a public contract. Evolve additively; version the schema; never break existing consumers (MS-COMM-06, governed by [`semver.md`](guides://semver.md); registry/encoding mechanics by [`kafka.md`](guides://kafka.md)).

### Contracts (owned obligation, mechanics referenced)
Inter-service contracts MUST be explicit and tested *before* deploy (MS-COMM-05). Use **consumer-driven contracts** (e.g. Pact): the consumer declares expectations, the provider's CI verifies them, and a breaking change blocks the deploy. The contract *artifact* is OpenAPI/Protobuf/AsyncAPI — owned by [`rest.md`](guides://rest.md)/[`grpc.md`](guides://grpc.md)/[`kafka.md`](guides://kafka.md); the *obligation to test it across the boundary* is owned here.

### Service discovery (owned)
Never hardcode `host:port` (MS-COMM-07). Resolve by logical service name through one of:
- **DNS-based** (Kubernetes Service DNS) — the default; deployment mechanics in [`kubernetes.md`](guides://kubernetes.md).
- **Service registry** (Consul, Eureka) — for non-K8s or cross-cluster.
- **Service mesh** (Istio, Linkerd) — discovery + mTLS + traffic policy in the data plane; see [`istio.md`](guides://istio.md).

### Avoid chatty coupling
Design **coarse-grained** APIs: return the data a use case needs in one call rather than N fine-grained calls. For client-facing aggregation, use a **Backend-for-Frontend (BFF)** or API-composition layer at the edge — and parallelize independent downstream calls rather than chaining them.

---

## 5. Distributed Data & Transactions (owned)

Each service owns its data (MS-DATA-01). Consistency across services is therefore *eventual*, coordinated by sagas — never a distributed XA/2PC transaction (MS-DATA-02).

### Saga pattern (owned)
A saga is a sequence of local transactions, each publishing an event that triggers the next; failure triggers **compensating** transactions that semantically undo prior steps.

- **Choreography** — services react to each other's events with no central coordinator.
  - Use when: few steps, loose coupling desired.
  - Cost: the end-to-end flow is implicit and emergent — hard to see and debug. Mitigate with distributed tracing ([`observability.md`](guides://observability.md)).
- **Orchestration** — a saga orchestrator issues commands and tracks state through the steps and compensations.
  - Use when: complex multi-step workflows that need an explicit, auditable flow.
  - Cost: the orchestrator is a coupling point and must itself be resilient and persistent.

**Compensation is semantic, not a rollback**: `PaymentCaptured` is compensated by `PaymentRefunded`, not by deleting a row. Design every step's inverse up front. Steps and their compensations MUST be idempotent (retried under failure).

### Transactional outbox (owned, mandatory for reliable events)
Writing to the database *and* publishing an event are two systems — they cannot share one transaction. The **outbox** solves the dual-write problem (MS-DATA-02):
1. In the same local DB transaction, write the business change **and** an `outbox` row.
2. A relay (poller or change-data-capture, e.g. Debezium) reads the outbox and publishes to the broker (broker mechanics: [`kafka.md`](guides://kafka.md)).
3. Mark the outbox row published. At-least-once delivery results → consumers must be idempotent (MS-COMM-04).

This makes "state changed" and "event emitted" atomic, eliminating lost or phantom events.

### Cross-service reads
You cannot `JOIN` across service databases. Choose:
- **API composition** — an aggregator queries each service and assembles the result. Simple; latency grows with fan-out; keep it at the edge (BFF).
- **CQRS** — maintain a denormalized read model fed by events, separate from the write model. Use for divergent read/write patterns and complex queries; accept eventual consistency.
- **Event-carried state / replicated read model** — a service keeps a local, read-only projection built from another service's events. Use for high-read, low-staleness-tolerance paths; the projection is owned and rebuildable from the event log.
- **Event sourcing** — persist state as an append-only event log; derive current state by replay. Use when you need a full audit trail, temporal queries, or replay-based recovery; it pairs naturally with CQRS. Significant complexity — adopt deliberately.

---

## 6. Resilience Topology (owned binding)

The *strategy* for timeouts, retries, and circuit breakers is owned by [`error-handling.md`](guides://error-handling.md). This guide owns only how those policies bind to a **service topology**.

- **Timeout layering** (MS-COMM-01): upstream timeout MUST exceed the sum of downstream timeout + its retries — `gateway > service > datastore`. An unbounded call is forbidden; it converts one slow dependency into a system-wide thread/connection exhaustion.
- **Circuit breaker placement** (MS-COMM-02): one breaker **per downstream dependency** (not one global breaker), so a failing `PaymentService` cannot open the circuit to a healthy `InventoryService`. Open → fail fast to a fallback.
- **Bulkhead isolation**: give each downstream dependency its own connection/thread pool (or concurrency semaphore). Exhausting the payment pool must not starve order or inventory calls. This is the topology-level expression of fault isolation.
- **Fallbacks** (MS-COMM-02): every critical path needs a degradation path — cached/stale value, default, an alternative provider, or a graceful "fail silent" for non-critical dependencies (e.g. analytics). The fallback must be simpler and more reliable than the primary.
- **Retry safety** (MS-COMM-03): retry only idempotent operations, with exponential backoff **plus jitter** to avoid synchronized retry storms (thundering herd). Combine with the breaker so a downed dependency is not hammered.
- **Idempotency keys**: non-idempotent operations (e.g. `POST /payments`) that may be retried by callers MUST accept an idempotency key so duplicate delivery is absorbed.

Where a service mesh is present, these policies (timeout, retry, breaker, mTLS) can be enforced in the data plane rather than in app code — see [`istio.md`](guides://istio.md).

---

## 7. Edge & Security Topology (owned binding)

Security *controls* (validation, secrets, authn/authz, crypto) are owned by [`secure-coding.md`](guides://secure-coding.md); token issuance by [`oauth.md`](guides://oauth.md). This guide owns the **distributed trust topology**.

- **API gateway is the single front door** (MS-SEC-02): all external traffic enters through a gateway that handles authentication, rate limiting, TLS termination, and routing. No service is directly internet-reachable. Keep business logic *out* of the gateway.
- **Zero trust between services** (MS-SEC-01): the network perimeter is not a trust boundary. Every inter-service call is mutually authenticated and encrypted — **mTLS**, typically enforced by the service mesh ([`istio.md`](guides://istio.md)). Assume breach; minimize blast radius via segmentation and least-privilege service identities.
- **Service identity**: each service has its own identity (mesh certificate or short-lived JWT), not a shared credential. Authorize on identity + claims; never trust an unauthenticated upstream.
- **Secrets**: per-service, rotated, from a secrets manager — never hardcoded or in plain config (policy in [`secure-coding.md`](guides://secure-coding.md)).

---

## 8. Deployment & Operations (referenced binding)

Deployment mechanics, probes, autoscaling, and rollout strategies are owned by [`kubernetes.md`](guides://kubernetes.md) (and [`ci-cd.md`](guides://ci-cd.md) for pipelines). This guide states the microservices-specific *requirements*:

- **Independently deployable** (MS-ARCH-02): each service builds, tests, and ships on its own pipeline and cadence. No lock-step releases.
- **Health probes** (MS-OBS-02): every service exposes liveness (process up — fast, no dependencies), readiness (can serve traffic — checks critical dependencies), and startup (initialization complete) endpoints. Readiness must exclude non-critical dependencies, or a non-essential outage will pull a healthy service from rotation. Semantics/instrumentation owned by [`observability.md`](guides://observability.md); probe wiring by [`kubernetes.md`](guides://kubernetes.md).
- **Graceful shutdown**: drain in-flight requests and commit/ack in-flight messages on SIGTERM before exit.
- **Progressive delivery**: prefer canary or blue-green for risky changes; decouple deploy from release with feature flags ([`feature-flags.md`](guides://feature-flags.md)). Strategy mechanics in [`kubernetes.md`](guides://kubernetes.md).
- **Service & event catalog**: maintain a catalog of every service (owner, API spec, dependencies) and every published event (schema, producer, consumers) — the operational basis for MS-ARCH-03 and MS-COMM-06.

---

## 9. Testing Across Boundaries

Test-first development, regression-before-fix, and coverage are owned by [`tdd.md`](guides://tdd.md); cross-system journeys by [`e2e-testing.md`](guides://e2e-testing.md). The microservices-specific shape of the test pyramid:

- **Unit** (many) — domain logic in isolation; all I/O mocked.
- **Component** (some) — one service in isolation; real DB via test containers, downstream services mocked. Tests the service's own API behavior.
- **Contract** (some, mandatory) — consumer-driven contracts verify cross-service compatibility *without* spinning up both services (MS-COMM-05). This is the layer that catches breaking changes before deploy.
- **Integration** (some) — service against real adapters (DB, broker test instance).
- **E2E** (few) — critical business journeys across services only; owned by [`e2e-testing.md`](guides://e2e-testing.md).

Test the failure modes that only exist in distributed systems: duplicate/out-of-order event delivery, downstream timeout → fallback, circuit-breaker open state, saga compensation, and partial failure. **Chaos experiments** (kill instances, inject latency, partition the network) validate that the resilience topology in §6 actually holds under failure — run them in non-production first, with a hypothesis and a rollback plan.

---

## 10. Anti-Patterns (PROHIBITED)

- **Distributed monolith** — services that deploy together, share a database, or chain synchronously. Violates MS-ARCH-02. Fix: decouple via events, give each its own data, draw real boundaries (§3).
- **Shared database** — multiple services reading/writing the same schema. Violates MS-DATA-01. Fix: database per service; share via API or events.
- **Distributed transaction (2PC/XA across services)** — Violates MS-DATA-02. Fix: saga + outbox (§5).
- **Chatty communication** — many fine-grained calls per use case. Fix: coarse-grained APIs, BFF, parallelize (§4).
- **Synchronous chains** — `Client → A → B → C → D`; latency and failure compound. Fix: async where possible, parallelize, aggregate at the edge.
- **Hardcoded service locations** — IPs/ports in config. Violates MS-COMM-07. Fix: service discovery (§4).
- **Dual write** — writing the DB and publishing an event in separate steps; one can fail leaving them inconsistent. Fix: transactional outbox (§5).
- **Non-idempotent consumer** — breaks under at-least-once delivery. Violates MS-COMM-04. Fix: dedupe on event id.

---

## 11. Deployment Checklist

Generated from §2 — one box per requirement ID. No new requirements.

- [ ] MS-ARCH-01 — each service = one bounded context
- [ ] MS-ARCH-02 — independently deployable, no lock-step releases
- [ ] MS-ARCH-03 — single owning team per service (catalog)
- [ ] MS-DATA-01 — sole writer of its data store; no shared schema
- [ ] MS-DATA-02 — saga + outbox for cross-service writes; no 2PC
- [ ] MS-COMM-01 — explicit timeout on every sync call (see `error-handling.md`)
- [ ] MS-COMM-02 — circuit breaker + fallback on critical sync paths
- [ ] MS-COMM-03 — retries idempotent-only, backoff + jitter
- [ ] MS-COMM-04 — async consumers idempotent (dedupe by id)
- [ ] MS-COMM-05 — consumer-driven contract tests pass in CI
- [ ] MS-COMM-06 — API/event changes backward-compatible or versioned (see `semver.md`)
- [ ] MS-COMM-07 — service discovery, no hardcoded endpoints
- [ ] MS-OBS-01 — trace context propagated across boundaries (see `observability.md`)
- [ ] MS-OBS-02 — liveness/readiness/startup endpoints (see `kubernetes.md`)
- [ ] MS-SEC-01 — mTLS between services (see `secure-coding.md`, `istio.md`)
- [ ] MS-SEC-02 — gateway-fronted external traffic with authn + rate limiting (see `oauth.md`)
- [ ] Agent verified boundaries, resilience topology, and observability before delivery

---
**End of Microservices Architecture Guidelines**
