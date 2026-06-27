# Hexagonal Architecture Guidelines
Language-agnostic standards for Hexagonal (Ports & Adapters) architecture: domain/application/infrastructure layering, dependency inversion, driving/driven ports, swappable adapters, isolated testing seams.

---
name: hexagonal
title: Hexagonal Architecture Guidelines
version: 2.0
last_reviewed: 2026-06-05
kind: cross-cutting
tools: []
requires: []
recommends:
  - tdd
  - designpatterns
  - cleanarch
  - microservices
  - architectures
provides:
  - ports-and-adapters
  - dependency-inversion
  - layer-boundaries
---

> 🧭 Authored per [`CONVENTIONS.md`](guides://CONVENTIONS.md): this guide canonically OWNS Hexagonal / Ports & Adapters / dependency inversion / domain–application–infrastructure layering. It references — never restates — testing (`tdd.md`), GoF patterns (`designpatterns.md`), Clean Architecture (`cleanarch.md`), distributed concerns (`microservices.md`), and the style overview (`architectures.md`).

---

## 0. Prerequisites & References

This guide is language-agnostic and has **no hard prerequisites** — it defines the layering rules that language/framework guides bind to. Fetch the recommended guides when the task touches them.

> 📎 **RECOMMENDED — fetch when the task touches them:**
> - [`tdd.md`](guides://tdd.md) — test-first, Red-Green-Refactor, regression-test-before-fix, coverage. *(This guide owns only the per-layer test **seam** strategy; the cycle and coverage policy live there.)*
> - [`designpatterns.md`](guides://designpatterns.md) — Repository, Adapter, Dependency-Injection, Factory, Strategy as GoF patterns. *(This guide shows where they sit in the hexagon; the pattern mechanics live there.)*
> - [`cleanarch.md`](guides://cleanarch.md) — Clean Architecture / Onion: the same Dependency Rule expressed as concentric rings + entities/use-cases/interface-adapters naming.
> - [`microservices.md`](guides://microservices.md) — applying the hexagon per service, anti-corruption layers, transport between bounded contexts.
> - [`architectures.md`](guides://architectures.md) — where hexagonal sits among layered/event-driven/CQRS/monolith-vs-microservice styles.

> 📎 **SEE ALSO:** [`adr.md`](guides://adr.md) (record boundary/port decisions) · [`error-handling.md`](guides://error-handling.md) (mapping adapter faults to domain errors) · [`rest.md`](guides://rest.md) · [`grpc.md`](guides://grpc.md) · [`graphql.md`](guides://graphql.md) · [`kafka.md`](guides://kafka.md) (driving/driven adapter transports)

> Relationship to Clean Architecture: hexagonal and Clean Architecture share one rule — **source-code dependencies point inward toward the domain**. They differ only in vocabulary (ports/adapters vs. boundaries/interface-adapters) and ring count. Do not duplicate `cleanarch.md`; pick one vocabulary per codebase and reference the other.

---

## 1. Core Philosophies: HEXAGONAL

Architecture-specific principles only. Test cycle, patterns, and distribution come from §0.

- **H**exagonal boundaries: three regions — domain, application, infrastructure — with a single, strict dependency direction.
- **E**xplicit ports: every crossing of a boundary goes through a named interface, never an ad-hoc call.
- **X**changeable adapters: any adapter can be replaced (real DB ↔ in-memory, Stripe ↔ Adyen) without touching domain or application.
- **A**pplication orchestration: the application layer coordinates domain objects and driven ports; it holds no business rules.
- **G**uarded domain: the domain has zero dependencies on frameworks, IO, transport, or persistence.
- **O**utward implementation, inward dependency: ports are *declared* inward (domain/application), *implemented* outward (infrastructure); the dependency points in via inversion.
- **N**o leaky abstractions: port signatures speak the domain's language — no SQL, HTTP, ORM, or vendor types cross a boundary.
- **A**gnostic core: the domain is technology-, framework-, and database-agnostic and remains compilable in isolation.
- **L**ayered test seams: each layer is exercised at the right seam — pure domain unit tests, mocked-port application tests, real-system integration tests for adapters.

**Verified Architecture**: agent-generated architecture MUST pass every gate in §2 (dependency direction, port placement, domain purity, test seams) before delivery.

---

## 2. Requirements (MANDATORY, auditable)

RFC-2119 keywords. IDs `HEX-<TOPIC>-<NN>`. Each row has a binary gate; rows binding a shared rule cite its owner.

| ID | Requirement | Verify | Gate |
|----|-------------|--------|------|
| HEX-ARCH-01 | Source dependencies MUST point inward only (Infra→App→Domain); no inner layer imports an outer one | architecture linter (ArchUnit / dependency-cruiser / import-linter / deptrac) | 0 inward-violations |
| HEX-ARCH-02 | The domain layer MUST NOT import any framework, ORM, transport, IO, or vendor library | linter rule on `domain/` imports | 0 forbidden imports |
| HEX-ARCH-03 | The application layer MUST NOT import any concrete adapter or infrastructure module | linter rule on `application/` imports | 0 infra imports |
| HEX-ARCH-04 | There MUST be no circular dependencies across or within layers | static cycle analysis | 0 cycles |
| HEX-PORT-01 | Every boundary crossing MUST go through a port interface; ports MUST be declared in domain (driven persistence) or application (other driven + all driving) | review / linter | no direct cross-layer calls |
| HEX-PORT-02 | Port signatures MUST use only domain types; no SQL/HTTP/ORM/vendor types in any port | review | 0 leaked infra types |
| HEX-PORT-03 | Each driven port MUST have ≥1 adapter AND a test double (in-memory/fake) usable without external systems | test inventory | double exists |
| HEX-DOM-01 | Domain entities/aggregates MUST encapsulate state and enforce invariants in behavior methods (no public setters, no anemic model) | review | no public mutators |
| HEX-DOM-02 | Value objects MUST be immutable and self-validating (invalid state unconstructable) | review / unit tests | construction validates |
| HEX-ADP-01 | Adapters MUST contain only translation/IO; zero business logic; MUST NOT call other adapters directly | review | logic-free adapters |
| HEX-ADP-02 | Driven adapters MUST map external faults to domain/application errors (see `error-handling.md`); no infra exception escapes inward | review / tests | faults mapped |
| HEX-TST-01 | Each layer MUST be tested at its seam (domain: no mocks; application: mocked ports; adapters: integration) — test-first per `tdd.md` | `tdd.md` runner | per-layer suites pass |
| HEX-TST-02 | Each bug MUST get a regression test in the owning layer before the fix (see `tdd.md`) | `tdd.md` runner | failing→passing |

> **Forbidden**: business logic in adapters or application services; framework annotations/ORM mappings on domain types; ports that expose SQL/HTTP/vendor types; an outer→inner import; a driven port with no test double; shipping a fix before its regression test (violates `tdd.md`).

---

## 3. The Layers & the Dependency Rule

Three regions, dependencies inward only. This is the load-bearing rule of the architecture.

```
┌─ INFRASTRUCTURE ──────────────────────────────────────────┐
│  driving adapters (REST/CLI/gRPC/consumers)               │
│  driven adapters (DB/HTTP clients/publishers/storage)     │
│  config, DI wiring, migrations                            │
│   ┌─ APPLICATION ───────────────────────────────────────┐ │
│   │  use cases / application services (orchestration)    │ │
│   │  driving ports (use-case interfaces)                 │ │
│   │  driven ports (gateway interfaces), DTOs, commands   │ │
│   │   ┌─ DOMAIN ─────────────────────────────────────┐   │ │
│   │   │  entities, value objects, aggregates,        │   │ │
│   │   │  domain services, domain events,             │   │ │
│   │   │  repository INTERFACES (driven persistence)  │   │ │
│   │   └──────────────────────────────────────────────┘   │ │
│   └──────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────┘
        dependencies flow INWARD only  →  (HEX-ARCH-01)
```

| Layer | Owns | MUST contain | MUST NOT contain | May depend on |
|---|---|---|---|---|
| **Domain** (inner) | business rules | entities, value objects, aggregates, domain services/events, repository **interfaces** | framework/ORM/HTTP/IO/vendor imports, persistence concerns, DTOs | only itself + stdlib value types |
| **Application** (middle) | orchestration | use cases/app services, driving ports, driven ports, DTOs, commands/queries | business rules, concrete adapters, infra imports | domain only |
| **Infrastructure** (outer) | technical concerns | driving + driven adapters, config/DI, persistence, migrations | business rules, leaking infra types inward | application + domain |

**Dependency inversion is what makes "inward" possible at runtime** (HEX-PORT-01): the application *declares* a driven port; infrastructure *implements* it; DI wiring in infrastructure injects the implementation. The compile-time arrow still points inward (infra → application interface), so the domain stays pure while still "calling out" to a database.

Repository **interfaces** are the one port type that conventionally lives in `domain/` (they are part of the ubiquitous domain language); their **implementations** live in infrastructure. All other ports live in `application/`.

---

## 4. Ports

A port is a typed contract at a boundary. Two kinds, by who initiates the call.

### A. Driving (primary / inbound) ports
The API the application *exposes*. The outside world drives the app through them.

```
External actor → [driving adapter] → (driving port) → application service
  HTTP request  → REST controller   → PlaceOrderUseCase
  CLI invocation→ CLI handler        → PlaceOrderUseCase
  Kafka message → message consumer   → ProcessPaymentUseCase
```
Driving ports are usually the use-case interfaces themselves (one method per use case). They live in `application/ports/driving/`.

### B. Driven (secondary / outbound) ports
The contracts the application *requires*. The app drives external systems through them.

```
application service → (driven port) → [driven adapter] → external system
  PlaceOrderUseCase  → OrderRepository → SqlOrderRepository → PostgreSQL
  PlaceOrderUseCase  → PaymentGateway  → StripeGateway      → Stripe API
  PlaceOrderUseCase  → Notifier        → SesNotifier        → email
```
Driven ports live in `application/ports/driven/` (persistence repository interfaces in `domain/repositories/`).

### C. Port design rules (HEX-PORT-02)
- Signatures use **domain types only** (`Money`, `OrderId`, `Email`) — never `ResultSet`, `HttpResponse`, `StripeCharge`, or pagination/framework types.
- Express intent in domain language: `findActiveByCustomer(id)`, not `findByQuery(sql)`.
- Return domain objects or a domain `Result`/error; map technical faults at the adapter (HEX-ADP-02, see [`error-handling.md`](guides://error-handling.md)).
- Keep ports narrow (Interface Segregation): many focused ports beat one god-interface.
- Every driven port ships a test double so the application is testable without infrastructure (HEX-PORT-03).

```
✅  interface OrderRepository { save(order: Order); findById(id: OrderId): Order? }
❌  interface OrderRepository { findBySql(sql: string); save(o: Order, conn: DbConnection) }   // leaks SQL + connection
```

> Repository/Adapter/Strategy/Factory as named GoF patterns are owned by [`designpatterns.md`](guides://designpatterns.md) — apply them here, but reference the mechanics rather than re-explaining them.

---

## 5. Adapters

An adapter binds a port to a concrete technology. It is the *only* place technology-specific code lives (HEX-ADP-01).

### A. Driving adapters (inbound)
Translate an external protocol into an application call, and the result back out. Responsibilities: deserialize input; validate **format** (not business rules); map to a command/DTO; invoke the use case; serialize the result; map application/domain errors to protocol responses (e.g. domain `OrderNotFound` → HTTP 404). One adapter may drive several use cases. They never touch the database or domain objects directly, and never call another adapter.

### B. Driven adapters (outbound)
Implement a driven port using a specific technology. Responsibilities: accept domain objects; translate to the external format (SQL row, HTTP body, message); execute the call with technical resilience (retries, timeouts, pooling); translate responses back to domain objects; map external faults to domain/application errors (HEX-ADP-02). They expose **no** external type upward — a persistence record (`UserRecord` with ORM annotations) lives entirely in infrastructure with `toDomain()` / `fromDomain()` mappers; the domain `User` stays annotation-free.

```
✅  class SqlOrderRepository implements OrderRepository {
        save(o: Order) { db.insert(OrderRecord.fromDomain(o)) }   // mapping only
    }
❌  class Order { save() { db.insert(this) } }                    // domain knows the DB — forbidden (HEX-ARCH-02)
```

---

## 6. Domain & Application Design

### A. Domain layer (the protected core)
- **Entities / aggregates**: identity + lifecycle; private state; behavior methods enforce invariants; equality by identity; factory methods for complex creation (HEX-DOM-01). The aggregate root is the consistency boundary and the only entry point to its internals.
- **Value objects**: immutable, equality by value, self-validating so an invalid instance cannot exist; replace primitives (`CustomerId`, `Money`, `Email`) to make illegal states unrepresentable (HEX-DOM-02).
- **Domain services**: stateless, named for a domain concept (`PricingService`), used when logic spans entities or needs a port; operate only on passed-in domain objects.
- **Domain events**: immutable records of something that happened; raised by the domain, dispatched by application/infrastructure.

The litmus test: the entire domain package compiles and its tests run with **zero** external dependencies (HEX-ARCH-02). Rich behavior, not an anemic bag of getters/setters.

### B. Application layer (orchestration)
A use case / application service is a thin coordinator with **one** public method (Single Responsibility): validate the command's format, load aggregates via driven ports, invoke domain behavior (which enforces the rules), persist via ports, emit events/notifications, return a DTO. It contains **no** business logic — pricing, eligibility, and invariants belong to the domain. Transactions are demarcated here (or delegated to infrastructure).

```
PlaceOrderUseCase.execute(cmd):                  ports injected (driven):
  customer = customers.findById(cmd.customerId)  ─ CustomerRepository
  order    = Order.create(customer, cmd.items)   ─ (domain enforces rules)
  payment  = gateway.charge(order.total)         ─ PaymentGateway
  order.markPaid(payment); orders.save(order)    ─ OrderRepository
  notifier.orderPlaced(order); return OrderDTO(order)  ─ Notifier
```
A service that calculates tax, validates inventory, opens a DB connection, or carries five public methods is a god-service — split it (one use case per class) and push the rules down into the domain.

---

## 7. Testing Seams

This guide owns *where* to test each layer; the cycle, coverage gates, and regression-before-fix policy are owned by [`tdd.md`](guides://tdd.md) (HEX-TST-01/02). Develop outward — Domain → Application → Infrastructure — writing the failing test first at each seam.

| Layer | Seam / test type | Mocking | Focus |
|---|---|---|---|
| Domain | unit | **none** (pure) | invariants, value-object validation, domain-service logic, event emission |
| Application | unit | **all driven ports** (use the fakes from HEX-PORT-03) | orchestration order, port interactions, error mapping, transaction boundaries |
| Infrastructure | integration | minimal (real/containerized systems, sandbox APIs) | adapter conforms to its port contract, real persistence/transport, retries & timeouts |

The architecture exists to make this cheap: the most critical code (domain) is the fastest and mock-free to test, because dependency inversion keeps IO at the edges. A bug is fixed in the **layer that owns it** — a violated invariant is a domain test, an orchestration slip an application test, a timeout-handling gap an adapter integration test — each with a regression test written first (see `tdd.md`).

---

## 8. Project Structure

The architecture must be visible in the tree. Two layouts; pick one.

**Layered (single bounded context):**
```
src/
├── domain/           # entities, value-objects, aggregates, services, events,
│                     # exceptions, repositories/ (INTERFACES only) — pure (HEX-ARCH-02)
├── application/      # services/ (use cases), ports/{driving,driven}/, dto/, commands/, queries/
└── infrastructure/   # adapters/{driving,driven}/, config/ (DI), migrations/
tests/                # unit/{domain,application}/, integration/adapters/, e2e/   (see tdd.md)
```

**Feature / modular monolith (multiple bounded contexts):** each module carries its own `domain/ application/ infrastructure/`, with a `shared/` kernel for cross-context value objects.
```
src/modules/{orders,users,payments}/{domain,application,infrastructure}/
src/shared/{domain,infrastructure}/
```
Module boundaries are themselves ports — communicate between contexts via interfaces/anti-corruption layers, not by reaching into another module's domain. When contexts become separate deployables, this is where [`microservices.md`](guides://microservices.md) takes over (the per-service internal architecture stays hexagonal).

---

## 9. Enforcement & Verification

Make the Dependency Rule executable in CI — review alone does not scale (HEX-ARCH-01..04).

```bash
# 1. per-layer test suites at their seams (see tdd.md for the runner & coverage gates)
<runner> tests/unit/domain/          # HEX-TST-01: fast, no mocks
<runner> tests/unit/application/     # HEX-TST-01: ports mocked
<runner> tests/integration/          # HEX-TST-01: real/containerized systems

# 2. dependency-direction & purity linting (HEX-ARCH-01..04, HEX-ARCH-02)
#    ArchUnit (Java/Kotlin) · dependency-cruiser (JS/TS) ·
#    import-linter (Python) · deptrac (PHP) · go-arch-lint (Go)

# 3. cycle check (HEX-ARCH-04) — included in the linters above
```

Architecture-review questions (fail any → not done):
- Can the domain be tested with **zero** infrastructure?
- Can the database be swapped by changing **one** adapter, with no domain/application edits?
- Can a new driving adapter (CLI alongside REST) be added without modifying existing code?
- Is every driven port declared inward and implemented outward, using domain types only?
- Is there **any** business logic outside the domain?

Record boundary, port, and adapter-swap decisions as ADRs (see [`adr.md`](guides://adr.md)).

---

## 10. Anti-Patterns (PROHIBITED)

| Anti-pattern | Why it breaks the hexagon | Fix |
|---|---|---|
| Domain imports infrastructure (`order.save()` hits the DB) | reverses the Dependency Rule | declare a repository port; implement it in infra (HEX-ARCH-01/02) |
| Anemic domain (getters/setters, rules in services) | logic leaks to application | move invariants into entity behavior (HEX-DOM-01) |
| Leaky port (`findBySql`, ORM annotations on domain types) | infra vocabulary crosses inward | domain-typed signatures; mappers in infra (HEX-PORT-02) |
| Business logic in an adapter | rules become untestable & duplicated per transport | adapters translate only; rules in domain (HEX-ADP-01) |
| God application service (CRUD + tax + inventory + reports) | violates SRP, hides use cases | one use case per class (§6.B) |
| Adapter calls another adapter | hidden coupling outside the app | route through an application service |
| Driven port with no test double | application can't be unit-tested | ship a fake per port (HEX-PORT-03) |

---

## Deployment Checklist

Generated from §2 — one box per requirement ID. No new requirements.

- [ ] HEX-ARCH-01 — dependencies point inward only (linter clean)
- [ ] HEX-ARCH-02 — domain free of framework/ORM/transport/IO/vendor imports
- [ ] HEX-ARCH-03 — application imports no concrete adapter/infrastructure
- [ ] HEX-ARCH-04 — zero circular dependencies
- [ ] HEX-PORT-01 — every boundary crossing goes through a correctly-placed port
- [ ] HEX-PORT-02 — port signatures use domain types only
- [ ] HEX-PORT-03 — each driven port has an adapter and a test double
- [ ] HEX-DOM-01 — entities/aggregates encapsulate state & enforce invariants
- [ ] HEX-DOM-02 — value objects immutable & self-validating
- [ ] HEX-ADP-01 — adapters logic-free; no adapter-to-adapter calls
- [ ] HEX-ADP-02 — external faults mapped to domain/application errors (see `error-handling.md`)
- [ ] HEX-TST-01 — each layer tested at its seam, test-first (see `tdd.md`)
- [ ] HEX-TST-02 — every bug has a regression test in its owning layer before the fix (see `tdd.md`)
- [ ] Agent ran every §9 command and documented any fixes

---
**End of Hexagonal Architecture Guidelines**
