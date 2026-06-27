# Clean Architecture Guidelines
Language-agnostic standards for Clean Architecture (Uncle Bob): the Dependency Rule, concentric entities/use-cases/interface-adapters/frameworks layers, explicit input/output boundaries, screaming architecture, framework independence.

---
name: cleanarch
title: Clean Architecture Guidelines
version: 2.0
last_reviewed: 2026-06-05
kind: cross-cutting
tools: []
requires: []
recommends:
  - hexagonal
  - designpatterns
  - microservices
  - architectures
  - tdd
provides:
  - dependency-rule
  - use-cases
  - entities
  - screaming-architecture
---

> 🧭 Authored per [`CONVENTIONS.md`](guides://CONVENTIONS.md): this guide canonically OWNS Clean Architecture — the Dependency Rule, the four concentric layers (entities, use cases, interface adapters, frameworks & drivers), input/output boundaries, the interactor/presenter model, and screaming architecture. It references — never restates — ports/adapters mechanics (`hexagonal.md`), GoF pattern mechanics (`designpatterns.md`), the test cycle (`tdd.md`), and distributed/style concerns (`microservices.md`, `architectures.md`).

---

## 0. Prerequisites & References

This guide is language-agnostic and has **no hard prerequisites** — it defines the architectural rules that language/framework guides bind to. Fetch the recommended guides when the task touches them.

> 📎 **RECOMMENDED — fetch when the task touches them:**
> - [`hexagonal.md`](guides://hexagonal.md) — Ports & Adapters: the SAME Dependency Rule in different vocabulary. **Pick ONE vocabulary per codebase** (Clean's boundaries/interactors/presenters OR hexagonal's ports/adapters) and reference the other; do not mix or duplicate. Port/adapter mechanics live there.
> - [`designpatterns.md`](guides://designpatterns.md) — Repository, Gateway, Factory, Strategy, Decorator, Dependency Injection as GoF patterns. *(This guide shows where they sit in the layers; the pattern mechanics live there.)*
> - [`tdd.md`](guides://tdd.md) — test-first, Red-Green-Refactor, regression-test-before-fix, coverage. *(This guide owns only the per-layer test **seam** strategy; the cycle and coverage policy live there.)*
> - [`microservices.md`](guides://microservices.md) — applying Clean Architecture per service, anti-corruption layers, transport between bounded contexts.
> - [`architectures.md`](guides://architectures.md) — where Clean Architecture sits among layered/onion/event-driven/CQRS styles.

> 📎 **SEE ALSO:** [`adr.md`](guides://adr.md) (record boundary/layer decisions) · [`error-handling.md`](guides://error-handling.md) (mapping adapter faults to domain errors) · [`rest.md`](guides://rest.md) · [`grpc.md`](guides://grpc.md) · [`graphql.md`](guides://graphql.md) (controller/presenter transports)

---

## 1. Core Philosophies: CLEAN-ARCH

Architecture-specific principles only. The test cycle, pattern mechanics, and distribution concerns come from §0.

- **C**oncentric layers: entities → use cases → interface adapters → frameworks & drivers, innermost most stable.
- **L**ayer independence: an inner circle knows **nothing** about an outer circle — not its names, types, or existence.
- **E**ntities first: enterprise-wide business rules sit at the center, sharable across applications, least likely to change.
- **A**pplication use cases: application-specific rules are explicit, named classes that orchestrate entities and drive the design.
- **N**o framework coupling: frameworks (web, ORM, UI) are plugins at the edge, replaceable without touching business rules.
- **A**bstraction boundaries: crossings go only through interfaces declared by the inner layer (dependency inversion).
- **R**eversible decisions: defer and isolate framework/DB choices so they stay swappable — maximize decisions *not yet made*.
- **C**ontrolled data flow: simple data structures cross boundaries in both directions; **dependencies** cross inward only.
- **H**idden details: implementation details (SQL, HTTP, vendor types) are invisible to the business rules.

**Verified Architecture**: agent-generated architecture MUST pass every gate in §2 (dependency direction, boundary placement, domain purity, test seams) before delivery.

---

## 2. Requirements (MANDATORY, auditable)

RFC-2119 keywords. IDs `CA-<TOPIC>-<NN>`. Each row has a binary gate; rows binding a shared rule cite its owner.

| ID | Requirement | Verify | Gate |
|----|-------------|--------|------|
| CA-ARCH-01 | Source dependencies MUST point inward only; no inner layer references an outer layer's names/types | arch-lint (ArchUnit / import-linter / dependency-cruiser / deptrac) in CI | no inward→outward edge |
| CA-ARCH-02 | Entities (innermost) MUST have ZERO framework/IO/persistence/UI imports | arch-lint rule `domain -> (nothing external)` | 0 violations |
| CA-ARCH-03 | Use cases MUST depend only on entities + interfaces they declare; no framework or adapter imports | arch-lint rule `usecases -> entities only` | 0 violations |
| CA-ARCH-04 | Outer-layer dependencies MUST be inverted: interfaces (repositories, gateways, output boundaries) declared in inner layer, implemented in outer | review / arch-lint | interfaces inner, impls outer |
| CA-ARCH-05 | Only DTOs/value objects/primitives MAY cross a boundary; framework types (HTTP req, ORM rows) MUST NOT | review / arch-lint type rule | no framework type at boundary |
| CA-ARCH-06 | No circular dependencies between modules/layers | arch-lint cycle check | 0 cycles |
| CA-STRUCT-01 | Each use case MUST be an explicit, single-purpose unit (one application operation) | review / directory listing | one op per use case |
| CA-STRUCT-02 | Structure MUST scream the domain (feature/use-case names visible), not the framework | review of top-level `src/` tree | domain-named, not type-named |
| CA-STRUCT-03 | Controllers/presenters/repositories MUST contain NO business logic (data conversion only) | review | no rules in adapters |
| CA-STRUCT-04 | Entities MUST be rich (behavior + invariants), not anemic; no public setters bypassing invariants | review | behavior present, invariants enforced |
| CA-TST-01 | Entities MUST be unit-tested with no mocks; use cases with mocked outer interfaces; adapters via integration (seams per §7; cycle per `tdd.md`) | test runner per layer | each layer green at its seam |
| CA-TST-02 | Each bug MUST get a regression test at the layer it lives in, before the fix (see `tdd.md`) | test runner | failing→passing |

> **Forbidden**: an inner layer importing an outer one; framework/ORM/HTTP types in entities or use cases; business logic in controllers, presenters, or repositories; anemic entities with logic in external "services"; skipping the use-case layer (controller → repository directly); fixing a bug without a regression test first (violates `tdd.md`).

---

## 3. The Dependency Rule (the one rule everything serves)

**Source-code dependencies MUST point inward only. Nothing in an inner circle may know anything about something in an outer circle — not its functions, classes, variables, or any named entity** (Robert C. Martin).

```
Frameworks ──▶ Interface ──▶ Use Cases ──▶ Entities
& Drivers      Adapters      (Application   (Enterprise
(outermost,   (Controllers,   Business       Business
 volatile)     Presenters,    Rules)         Rules)
               Gateways,                     (innermost,
               Repositories)                  stable, abstract)

Dependencies flow INWARD only. Inner = more stable, more abstract.
```

**Data flow vs. dependency direction** — the two are independent and must not be conflated:
- *Data* flows both ways: a controller passes a request DTO inward; the use case returns a response DTO outward through the output boundary.
- *Dependencies* point inward only: the controller and presenter depend on the use case's boundary interfaces; the use case depends on nothing outward.

**Dependency inversion at every boundary.** When an inner layer needs something an outer layer provides (a database, a payment API), the inner layer **declares the interface** and the outer layer **implements** it. The dependency arrow then points inward to the abstraction, not outward to the concrete class:

```
Use Case ──▶ <<interface>> Repository   (declared in the use-case layer)
                   ▲
                   │ implements
            SqlOrderRepository           (lives in the adapter layer)
```

This is the mechanism shared with hexagonal's "ports declared inward, adapters implemented outward" — see [`hexagonal.md`](guides://hexagonal.md); choose one vocabulary and don't restate both.

---

## 4. The Four Concentric Layers

| Layer | Contains | Depends on | Volatility |
|-------|----------|-----------|------------|
| **Entities** (Enterprise Business Rules) | Domain objects, value objects, domain services, domain events, enterprise policies | Nothing | Lowest |
| **Use Cases** (Application Business Rules) | Interactors, input/output boundary interfaces, request/response DTOs, repository & gateway *interfaces* | Entities only | Low |
| **Interface Adapters** | Controllers, presenters, view models, gateway impls, repository impls, mappers | Use cases, entities | High |
| **Frameworks & Drivers** | Web/ORM/UI framework code, DB drivers, DI/composition root, external clients, config wiring | All inner layers | Highest |

**Entities** — enterprise-wide critical rules and data structures, sharable across multiple applications, framework/DB/UI-agnostic. No operational change to any one application should affect them.

**Use cases** — application-specific rules. They orchestrate entities to achieve a goal, define the input/output boundaries, and declare (but never implement) the interfaces for repositories and gateways. They contain no framework code and no presentation logic.

**Interface adapters** — convert data between the use-case format and external formats. Controllers adapt input → request DTO; presenters adapt response DTO → view model; repositories/gateways implement inner-layer interfaces against real infrastructure. **No business logic** — conversion only.

**Frameworks & drivers** — the volatile edge: the web server, ORM setup, UI framework, and the composition root that wires everything. Designed so an entire framework can be swapped without touching inner layers.

---

## 5. Directory Structure (screaming architecture)

The top of the tree MUST reveal **what the system does**, not which framework it uses. Two valid layouts:

**A. Layered (layer-first).** Best when one bounded context dominates:

```
src/
├── domain/                 # ENTITIES: entities/, value-objects/, services/, events/, policies/
├── application/            # USE CASES: use-cases/, boundaries/{input,output}/, dto/,
│                           #            interfaces/{repositories,gateways}/   (interfaces only)
├── adapters/               # INTERFACE ADAPTERS: controllers/{http,cli,graphql}/, presenters/,
│                           #                     gateways/, persistence/{sql,memory}/
└── infrastructure/         # FRAMEWORKS & DRIVERS: web/, database/, config/ (DI, composition root)
tests/  → unit/{domain,application}/  integration/adapters/  e2e/
```

**B. Screaming / feature-first.** Preferred at scale — the directory *screams* the domain. Each bounded context owns its four layers; a `shared/` kernel holds cross-context value objects and infra:

```
src/
├── orders/      → domain/  application/  adapters/  infrastructure/
├── customers/   → domain/  application/  adapters/  infrastructure/
├── payments/    → domain/  application/  adapters/  infrastructure/
└── shared/      → domain/ (Money, Email)   infrastructure/ (Database, Logging)
```

Group by feature/use case, not by technical type. Reading `src/` should let a newcomer name the system's capabilities without opening a file.

---

## 6. Entities & Value Objects

**Entities** encapsulate identity, state, behavior, and invariants. All state changes go through behavior methods that re-check invariants; no public setters; complex construction via factory methods; significant changes may raise domain events. They are **rich**, never anemic.

```
Order (entity)
├── Identity:   orderId: OrderId
├── State:      customerId, items, status, totalAmount, placedAt   (encapsulated)
├── Invariants: ≥1 item · total = Σ items · same currency · cannot modify completed order
├── Behavior:   addItem() · removeItem() · submit() · cancel(reason) · complete()
└── Factory:    Order.create(customerId, items)  → validates, returns Order
```

**Anemic-model anti-pattern (forbidden):** an `Order` with only getters/setters while an external `OrderService` holds the rules. Rules belong *inside* the entity.

**Value objects** are immutable, compared by attribute (not identity), self-validating on construction, and side-effect-free (operations return new instances). Replace primitives with them: `Money` (amount + currency), `Email`, `Address`, `DateRange`, `Quantity`, `OrderId`, `PhoneNumber`. Typed value objects make whole classes of bug (mixed currency, invalid email) unrepresentable.

---

## 7. Use Cases: Interactors & Boundaries

A use case is the application's unit of behavior, expressed through four parts plus its interactor:

```
PlaceOrder use case
├── Input Boundary (interface):  execute(PlaceOrderRequest): void          ← called by controllers
├── Output Boundary (interface): presentSuccess(resp) · presentValidationError(errs)
│                                 · presentNotFound(msg) · presentError(err)  ← implemented by presenters
├── Request model (DTO):         primitives/simple DTOs only — no framework types
├── Response model (DTO):        primitives/simple DTOs only
└── Interactor (impl of Input Boundary):
        depends on → OrderRepository, CustomerRepository, PaymentGateway, OutputBoundary
        (ALL interfaces declared in this layer)
```

**Interactor responsibilities**, in order: validate the request → load entities via repository interfaces → invoke entity behavior → apply application rules → call gateway interfaces (payment, notification) → persist → present via the output boundary. It contains **no** framework code and **no** presentation logic; every dependency is an interface.

**Why a separate output boundary (Clean's distinction from a plain port).** The use case pushes a result *out* through the presenter interface rather than *returning* a value the controller formats. This lets the presenter own all formatting (status codes, currency/date formatting, error shapes) and supports multiple presentation formats (JSON, XML, HTML) behind one use case. If your codebase uses hexagonal vocabulary, this collapses into a single driven port — see [`hexagonal.md`](guides://hexagonal.md); don't model it both ways.

---

## 8. Interface Adapters: Controllers, Presenters, Gateways/Repositories

- **Controller** — extract data from the framework request → map to the request DTO → call `inputBoundary.execute(request)` → return the presenter's view model. Thin; framework code isolated here; no business logic.
- **Presenter** — implements the output boundary; receives the response DTO and builds a view model (formatting only: dates, currency, error codes). One presenter per output format.
- **Repository / Gateway impl** — implements an interface declared in the use-case layer; maps entity ↔ external representation; holds DB/HTTP/vendor code and error handling; **no** business logic. One interface, many impls behind it: `SqlOrderRepository` (prod), `InMemoryOrderRepository` (tests), `CachedOrderRepository` (decorator). These are GoF patterns — see [`designpatterns.md`](guides://designpatterns.md) for Repository/Gateway/Adapter/Decorator mechanics.

**Composition root** (frameworks layer) is the only place that knows concrete types: it constructs infrastructure, then adapters that implement inner interfaces, then presenters, then interactors (injecting the adapters), then controllers, then wires routes. All `new` of concrete dependencies happens here, at the outermost edge.

---

## 9. Testing Seams (cycle & coverage owned by `tdd.md`)

The Red-Green-Refactor cycle, coverage gates, and regression-test-before-fix policy are owned by [`tdd.md`](guides://tdd.md). Clean Architecture owns only **where the test seam sits per layer** and the inside-out order:

| Layer | Seam | Mocks | Speed |
|-------|------|-------|-------|
| Entities | public methods / factories | none (pure) | fastest, most numerous |
| Use cases | input boundary | mock repository/gateway interfaces only | fast, numerous |
| Adapters | interface contract | real/containerized infra | slower, fewer |
| System | HTTP/CLI entry | none | slowest, fewest |

Rules: test inside-out (entities, then use cases, then adapters) so inner layers are stable before outer ones depend on them; test **through boundaries**, never private methods or internal state; mock **only** at layer boundaries. The CA-TST IDs in §2 bind these seams; the cycle behind them lives in `tdd.md`.

---

## 10. Clean Architecture vs. Hexagonal — pick one

Clean Architecture and Hexagonal (Ports & Adapters) enforce the **same rule** — source dependencies point inward toward the domain. They differ only in vocabulary and granularity:

| Aspect | Clean Architecture | Hexagonal |
|--------|--------------------|-----------|
| Rings/regions | 4 concentric layers | 3 regions (domain, application, infrastructure) |
| Primary lens | use cases & boundaries | ports & adapters |
| Entities | their own layer | part of the domain |
| Output | explicit presenter / output boundary | a driven port (presenter optional) |
| Coiners | Robert C. Martin | Alistair Cockburn |

**Choose ONE vocabulary per codebase and reference the other** — do not model boundaries as both layers and ports, and do not duplicate `hexagonal.md` here. Reach for Clean when you have rich use-case orchestration, explicit input/output boundaries, or multiple presentation formats; reach for hexagonal when you want fewer regions and a sharp infrastructure-isolation focus (common in microservices — see [`microservices.md`](guides://microservices.md)). For how both sit among other styles, see [`architectures.md`](guides://architectures.md).

---

## 11. Why This Architecture Works

- **The Dependency Rule protects business logic from churn:** databases, UI, and frameworks can be replaced and only outer-layer code changes; the core stays stable.
- **Testability without infrastructure:** entities and use cases have no framework/DB dependencies, so the most important code is also the fastest and easiest to test.
- **Framework independence reduces migration risk:** all vendor-specific code lives at the edge, so upgrades/replacements touch adapters only.
- **Explicit use cases make the system self-documenting:** named use-case classes reveal capabilities from the directory tree (screaming architecture).
- **Concentric layers enforce separation of concerns:** the boundary between entities, use cases, adapters, and frameworks prevents rules, persistence, and presentation from entangling.

> "The center of your application is not the database, nor the frameworks. The center is the use cases of your application." — Robert C. Martin

---

## 12. Deployment Checklist

Generated from §2 — one box per requirement ID. No new requirements.

- [ ] CA-ARCH-01 — dependencies point inward only (arch-lint in CI, no inward→outward edge)
- [ ] CA-ARCH-02 — entities have zero framework/IO/persistence/UI imports
- [ ] CA-ARCH-03 — use cases depend only on entities + declared interfaces
- [ ] CA-ARCH-04 — interfaces declared inner, implemented outer (dependency inversion)
- [ ] CA-ARCH-05 — only DTOs/value objects cross boundaries; no framework types
- [ ] CA-ARCH-06 — no circular dependencies (cycle check clean)
- [ ] CA-STRUCT-01 — each use case is single-purpose
- [ ] CA-STRUCT-02 — structure screams the domain, not the framework
- [ ] CA-STRUCT-03 — controllers/presenters/repositories carry no business logic
- [ ] CA-STRUCT-04 — entities are rich (behavior + invariants), not anemic
- [ ] CA-TST-01 — entity/use-case/adapter tests at the correct seam (see `tdd.md`)
- [ ] CA-TST-02 — every bug has a regression test at its layer before the fix (see `tdd.md`)
- [ ] Agent ran the arch-lint and per-layer test commands and documented any fixes

---
**End of Clean Architecture Guidelines**
