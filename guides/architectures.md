# Software Architecture Styles Guidelines
How to position, compare, and choose software architecture styles — internal structure, deployment, communication, and data — by trade-off, not by hype. Language-agnostic; centred on ADRs, the C4 Model, DDD, and fitness functions.

---
name: architectures
title: Software Architecture Styles Guidelines
version: 2.0
last_reviewed: 2026-06-05
kind: cross-cutting
tools: []
requires: []
recommends:
  - hexagonal
  - cleanarch
  - microservices
  - designpatterns
  - observability
provides:
  - style-selection
  - tradeoff-analysis
  - architecture-overview
---

> 🧭 Authored per [`CONVENTIONS.md`](guides://CONVENTIONS.md): this guide OWNS the overview, comparison, and selection of architecture *styles*. It deliberately does **not** restate the deep mechanics of styles that have their own canonical guide — it references them and keeps only positioning, trade-offs, and decision guidance.

---

## 0. Prerequisites & References

This is the entry point for "which architecture should this system use?" Once a style is chosen, fetch its owning guide for the mechanics.

> 📎 **RECOMMENDED — fetch when the task settles on that style:**
> - [`hexagonal.md`](guides://hexagonal.md) — Ports & Adapters mechanics, dependency inversion, adapter wiring. *(This guide only positions it vs. Clean/Onion/Layered.)*
> - [`cleanarch.md`](guides://cleanarch.md) — the four concentric layers, Screaming Architecture, use-case/entity split.
> - [`microservices.md`](guides://microservices.md) — service decomposition, data ownership, inter-service contracts, deployment topology.
> - [`designpatterns.md`](guides://designpatterns.md) — GoF & friends used to realise a style (Strategy, Factory, Adapter, Repository).
> - [`observability.md`](guides://observability.md) — metrics, tracing, and the fitness-function instrumentation a distributed style demands.

> 📎 **SEE ALSO:** [`adr.md`](guides://adr.md) — record every style decision · [`tdd.md`](guides://tdd.md) — architectural fitness tests are written test-first · [`kafka.md`](guides://kafka.md) — event-streaming substrate · [`kubernetes.md`](guides://kubernetes.md) — runtime for microservices/serverless · [`rest.md`](guides://rest.md) · [`grpc.md`](guides://grpc.md) · [`graphql.md`](guides://graphql.md) — synchronous communication contracts · [`error-handling.md`](guides://error-handling.md) · [`performance.md`](guides://performance.md) · [`parallelism.md`](guides://parallelism.md) · [`secure-coding.md`](guides://secure-coding.md).

---

## 1. Core Philosophies: ARCHITECTURE-FIRST

Principles unique to *choosing and positioning* architecture. The mechanics of any chosen style live in its owning guide (§0); test-first discipline lives in [`tdd.md`](guides://tdd.md).

- **A**lign with business: architecture serves business goals, not résumés.
- **R**ight-size complexity: match architecture complexity to *problem* complexity (see the alignment matrix in §7).
- **C**hange-friendly: design for evolution, not a perfect final state.
- **H**uman-centric: fit the architecture to team size, skill, and cognitive load (Conway's Law is real — the system mirrors the org chart).
- **I**nformed trade-offs: every choice has costs; make them explicit and write them down.
- **T**estable by design: prefer styles whose boundaries enable isolated testing.
- **E**volvable boundaries: pick boundaries that can shift as the domain is understood.
- **R**eversible decisions: prefer two-way-door choices; defer one-way doors until forced.
- **U**ndocumented is unfinished: a style choice without an ADR (see [`adr.md`](guides://adr.md)) does not exist.

**Orthogonality is the key insight** (§3): internal structure, deployment, communication, and data are *separate axes*. You combine them — e.g. Microservices (deployment) + Hexagonal (internal) + Events (communication) + CQRS (data). Never conflate "should it be microservices?" with "should it be hexagonal?".

---

## 2. Requirements (MANDATORY, auditable)

RFC-2119 keywords. IDs `ARCH-<TOPIC>-<NN>`. These govern the *act of choosing and enforcing* a style; per-style implementation rules are gated in that style's owning guide.

| ID | Requirement | Verify | Gate |
|----|-------------|--------|------|
| ARCH-STRUCT-01 | Each system MUST declare its style on all four axes (internal / deployment / communication / data) | review of the system's ADR | four axes named |
| ARCH-STRUCT-02 | Style complexity MUST be justified against problem complexity (no microservices/CQRS without a demonstrated driver) | review vs. §7 alignment matrix | justification present |
| ARCH-STRUCT-03 | Module/layer dependencies MUST flow as the chosen style mandates; no cycles (see owning style guide) | `pydeps`/`madge --circular` / import-linter | 0 cycles, 0 violations |
| ARCH-ARCH-01 | Every significant style decision MUST be recorded as an ADR (see `adr.md`) | ADR exists in `docs/adr/` | one ADR per decision |
| ARCH-TST-01 | Architectural constraints MUST have automated fitness tests, written test-first (see `tdd.md`) | `pytest tests/architecture/` (or lang equiv) | exit 0 |
| ARCH-TST-02 | Each architectural violation bug MUST get a regression fitness test before the fix (see `tdd.md`) | run the new test pre-fix | failing→passing |
| ARCH-OBS-01 | Distributed styles MUST emit the metrics/traces their fitness functions assert on (see `observability.md`) | dashboards/traces exist | SLO signals present |
| ARCH-STRUCT-04 | Service/module boundaries MUST be independently deployable iff the deployment style claims so (no distributed monolith) | service independence test | each deploys alone |

> **Forbidden**: adopting a distributed style to learn it (résumé-driven development); copying a hyperscaler's architecture without their constraints (cargo cult); shipping "microservices" that share a database or must deploy together (distributed monolith); choosing a style without an ADR.

---

## 3. The Four Orthogonal Axes

Architecture styles answer four *independent* questions. Choose one option per axis and combine.

| Axis | Question | Options |
|------|----------|---------|
| **Internal structure** | How is code organised inside a unit? | Layered · Hexagonal · Clean · Onion · Vertical Slice |
| **Deployment** | How is it packaged and run? | Monolith · Modular Monolith · Microservices · Serverless |
| **Communication** | How do parts talk? | Request/Reply (REST/gRPC/GraphQL) · Events · Message Queue · Streaming |
| **Data** | How is state managed? | CRUD · CQRS · Event Sourcing |

```
EXAMPLE COMBINATION:
  Microservices (deployment)
    └─ each service uses Hexagonal (internal)
         └─ services communicate via Events (communication)
              └─ complex reporting uses CQRS (data)
```

The sections below give the *positioning* of each option — what it's for, when it wins, what it costs. Deep mechanics of Hexagonal, Clean, and Microservices are in their own guides (§0).

---

## 4. Internal Structure Styles

How you organise code *within* a deployable unit. Hexagonal, Clean, and Onion share one DNA — **dependency inversion** (dependencies point inward, the domain knows nothing of infrastructure). They differ mainly in prescriptiveness and vocabulary.

### A. Layered (N-Tier)
Presentation → Business → Data Access → Database, each depending on the one below.
- ✅ Simplest to grasp; fine for CRUD and prototypes.
- ❌ DB changes ripple upward; business logic leaks; hard to test in isolation; trends toward an anemic domain model.
- **Best for:** simple CRUD apps, MVPs, teams new to structured architecture.

### B. Hexagonal (Ports & Adapters)
Domain at the centre, **ports** (interfaces) as boundaries, **adapters** translating external↔internal; driving (input) vs. driven (output) sides.
- ✅ Highly testable (mock adapters); multiple I/O channels; framework-independent.
- ❌ Up-front interface ceremony; needs comfort with dependency inversion.
- **Best for:** long-lived apps needing high testability and many I/O channels.
- 📎 Mechanics, port/adapter wiring, and language bindings: [`hexagonal.md`](guides://hexagonal.md).

### C. Clean (Uncle Bob)
Four explicit concentric layers — Entities → Use Cases → Interface Adapters → Frameworks & Drivers — with source dependencies pointing strictly inward, plus "Screaming Architecture" (directory structure reveals *what the app does*, not which framework it uses).
- vs. Hexagonal: more prescriptive (fixed layer count), separates Entities from Use Cases, explicit Presenter pattern.
- **Best for:** large teams wanting a standardised, governed layer scheme.
- 📎 Layer responsibilities and mechanics: [`cleanarch.md`](guides://cleanarch.md).

### D. Onion (Palermo)
Domain Model at the absolute core, then Domain Services (pure, stateless), Application Services (orchestration), Infrastructure outermost. Core MUST compile without infrastructure. Strong DDD alignment; sharpest Domain-vs-Application service split.
- **Best for:** complex enterprise domains with strong DDD experience and 10-year horizons.

### E. Vertical Slice
Organise by **feature**, not layer: each slice (e.g. `CreateOrder/` with its command, handler, validator, endpoint) cuts through all layers and is independent. "Code that changes together lives together"; duplication is preferred over the wrong abstraction.
- ✅ Fast feature work, low cross-feature coupling, good for brownfield refactors.
- ❌ Can duplicate; heavily interconnected features still want a shared domain.
- **Best for:** CRUD-heavy or feature-driven apps; discovering boundaries before extraction.

### F. The "inverted family" at a glance

| Aspect | Layered | Hexagonal | Clean | Onion | Vertical Slice |
|--------|---------|-----------|-------|-------|----------------|
| Core focus | Technical layers | Boundaries & testing | Standardised layers | Pure domain model | Feature independence |
| Dependency rule | Top→down | Inward to domain | Inward to entities | Inward to domain model | Within the slice |
| Prescriptive | Low | Less | More | Medium | Low |
| DDD alignment | Weak | Compatible | Compatible | Strong | Compatible |
| Complexity | Low | Medium | High | High | Medium |
| Testability | Low | High | High | High | Medium |
| Best team size | 1-5 | 3-15 | 10-50+ | 10-50+ | 5-20 |

---

## 5. Deployment & Communication Styles

How you package/run the system, and how its parts talk.

### Deployment

| Style | What it is | Choose when |
|-------|-----------|-------------|
| **Monolith** | One deployable artifact, one process, often one DB. ACID across modules, trivial debugging; scales all-or-nothing. | New project, team < 10, simple domain, need fast time-to-market. |
| **Modular Monolith** | Monolith with **strict** module boundaries: modules talk only via public APIs, own their schema, never query each other's tables. The common "sweet spot". | Growing complexity, team 5-30, want clean boundaries + simple ops, planning eventual extraction. |
| **Microservices** | Small, independently deployable services, each owning its data, talking over the network. | Large org, multiple autonomous teams, independent scaling, mature DevOps, clear bounded contexts. 📎 [`microservices.md`](guides://microservices.md) |
| **Serverless** | Functions as the deployment unit; pay-per-invocation, auto-scale to zero, event-driven, stateless. | Event-driven or spiky workloads, cost-sensitive, simple stateless operations. |

**Extraction path** (the recommended evolution): Monolith → Modular Monolith (clean module boundaries) → extract a module to a service → replace in-process calls with network calls → split its schema. Start simple; extract only when boundaries are proven and a driver exists.

### Communication

| Style | Coupling | Consistency | Best for | Cost |
|-------|----------|-------------|----------|------|
| **Request/Reply** (REST/gRPC/GraphQL) | Tight (caller blocks) | Strong | Simple, immediate-feedback APIs | Cascading failures; callee must be up. 📎 [`rest.md`](guides://rest.md) · [`grpc.md`](guides://grpc.md) · [`graphql.md`](guides://graphql.md) |
| **Event-Driven** (pub/sub of facts: `OrderPlaced`) | Loose | Eventual | Decoupled, scalable, resilient reactions | Eventual consistency; harder debugging; ordering. |
| **Message Queue** (commands: `ProcessPayment`) | Loose | At-least-once delivery | Load leveling, parallel work, buffered spikes | Idempotency required; message-broker ops. |
| **Event Streaming** (immutable ordered log; Kafka) | Loose | Eventual, replayable | Event sourcing, audit, real-time analytics, state rebuild | Offset/retention management. 📎 [`kafka.md`](guides://kafka.md) |

The defining difference between a queue and a stream: a queue **deletes** a message once consumed; a stream **retains** it, and consumers track their own offset and can replay.

---

## 6. Data Styles

How state is read and written.

| Style | What it is | Best for | Trade-offs |
|-------|-----------|----------|------------|
| **CRUD** | One model for reads and writes, one schema, one DB. | Simple apps, similar read/write patterns, small volumes. | Read and write needs eventually diverge. |
| **CQRS** | Separate **write model** (rich, normalised, validated) and **read model** (denormalised, query-optimised), kept in sync by projection. | Read-heavy systems, complex queries that don't fit the write model, independent read/write scaling. | Eventual consistency between models; sync complexity. |
| **Event Sourcing** | Store **events**, not current state; state = replay of events. | Audit/compliance, temporal queries, debug-by-replay, rebuildable read models. | Event-schema evolution, long-stream performance, learning curve. |

CQRS and Event Sourcing pair naturally: events are the write side, projections build the read models.

---

## 7. Selecting a Style (decision framework)

This is the canonical decision content this guide owns.

### A. Requirements → recommendation

| If the project is… | Internal | Deployment | Communication | Data |
|--------------------|----------|------------|---------------|------|
| Simple CRUD / prototype | Layered | Monolith | REST | CRUD |
| Standard business app | Hexagonal | Monolith / Modular Monolith | REST | CRUD |
| Growing product | Hexagonal per module | Modular Monolith | REST + internal Events | CRUD + read replicas |
| Complex domain, long lifecycle | Clean / Onion + DDD | Modular Monolith → Microservices | Events + gRPC | CQRS for key domains |
| High scale / many teams | Hexagonal per service | Microservices | Event Streaming | CQRS + Event Sourcing |
| Variable / spiky traffic | Vertical Slice or Hexagonal | Serverless | Events | CRUD or Event Sourcing |

### B. Complexity alignment (the central rule)

Architecture complexity MUST match problem complexity (ARCH-STRUCT-02):

```
                 Simple problem   Medium problem   Complex problem
Layered          ✅ perfect fit   ❌ under-built   ❌ dangerous
Hexagonal        ❌ over-built    ✅ perfect fit   ❌ may struggle
Microservices    ❌ massively     ❌ over-         ✅ appropriate
                    over-built       engineered       complexity
```

### C. Team-size heuristic

| Team | Deployment default |
|------|--------------------|
| 1-5 | Monolith (Hexagonal internals optional) |
| 5-15 | Modular Monolith |
| 15-50 | Modular Monolith or *careful* Microservices |
| 50+ | Microservices (a team per service) |

### D. Decision questions to answer in the ADR

- **Deployment:** How many teams? Do parts scale independently? DevOps maturity? Infra budget?
- **Internal:** How complex is the domain logic? How important is isolated testability? How volatile is the infrastructure choice?
- **Communication:** Do we need immediate responses? Can we tolerate eventual consistency? Do we need replay/audit?
- **Data:** Do read and write patterns differ? Do we need full history? What consistency is acceptable?
- **Evolution:** How long will this live? How uncertain are the requirements? How cheap is later refactoring?

### E. Decision flow

```
Prototype/MVP? ─yes→ Layered + Monolith + REST + CRUD (keep it simple)
   │no
Clear domain boundaries? ─no→ Vertical Slice + Modular Monolith (discover boundaries)
   │yes
>3 teams working independently? ─no→ Hexagonal + Modular Monolith + REST/Events (extraction-ready)
   │yes
   └→ Microservices + Event-Driven (+ CQRS if read/write diverge)
```

### F. Proven combinations

- **Startup:** Layered/Hexagonal · Monolith · REST · CRUD — MVP, < 5 devs, need speed.
- **Growth:** Hexagonal per module · Modular Monolith · REST + internal Events · CRUD + read replicas — 5-30 devs, eventual extraction.
- **Enterprise:** Clean/Hexagonal per service · Microservices · gRPC internal + Events external · CQRS + Event Sourcing for key domains — 30+ devs, compliance.
- **Real-time:** Hexagonal per service · Microservices/Serverless · Event Streaming (Kafka) · Event Sourcing + CQRS — replay/audit needs.

---

## 8. Resilience & Consistency Patterns (distributed styles)

Once a distributed deployment/communication style is chosen, these patterns are table stakes. They are *named* here for selection; their implementation belongs to the relevant service/infra guides and [`error-handling.md`](guides://error-handling.md).

- **Circuit Breaker** — stop calling a failing dependency (`CLOSED → OPEN → HALF-OPEN`) to prevent cascading failure.
- **Retry with backoff** — exponential backoff *with jitter* (`base * 2^attempt`); only for idempotent operations.
- **Bulkhead** — isolate resource pools so one overloaded dependency can't sink the rest.
- **Timeout** — fail fast and release resources; never wait forever.
- **Fallback** — degrade gracefully (cached/default value, backup service) when the primary fails.
- **Saga** — model a distributed transaction as local transactions with compensating actions; choreography (events) for loose coupling, orchestration (central coordinator) for visibility.
- **Outbox** — write the domain change and an outbox row in one DB transaction; a separate process relays the row to the broker, guaranteeing the event is published iff the state changed.

> Distributed styles only work with observability: every saga step, retry, and breaker state transition must be traceable (see [`observability.md`](guides://observability.md), ARCH-OBS-01).

---

## 9. Anti-Patterns to Avoid

- **Distributed monolith** — services that share a database or must deploy together: all the cost of microservices, none of the benefit. (Gated by ARCH-STRUCT-04.)
- **Résumé-driven development** — choosing a style to learn it rather than to solve the problem.
- **Premature microservices** — splitting before bounded contexts are understood; start modular-monolith, extract later.
- **Big ball of mud** — no boundaries; everything depends on everything. Enforce boundaries even in a monolith.
- **Golden hammer** — one architecture for every problem regardless of fit.
- **Cargo cult** — copying Netflix/Amazon topology without their scale or constraints.
- **Analysis paralysis** — over-deliberating; make reversible decisions quickly and iterate.

### Architecture smells → reconsider

| Smell | Consider |
|-------|----------|
| Changing one feature touches 5+ files across layers | Vertical Slice |
| Tests need full DB/infra to run | Ports & Adapters (Hexagonal) |
| Two teams blocked on each other | Module boundaries / Microservices |
| Must deploy everything to change one part | Modular Monolith / Microservices |
| Can't say where new code belongs | Clean Architecture layers |
| Same bug recurs | Add architectural fitness tests |
| Schema drives every decision | Domain-first (DDD) |
| Network calls everywhere, cascading failures | Event-driven + circuit breakers |
| Simple changes take weeks | *Reduce* architecture complexity |

---

## 10. Documenting & Enforcing the Choice

- **ADRs** — record every significant style decision (status, context, decision, consequences). Owned by [`adr.md`](guides://adr.md); ARCH-ARCH-01 requires one per decision.
- **C4 model** — communicate at four zoom levels: System Context → Container → Component → Code (last only where complex).
- **Quality attributes** — state non-functional targets the style must meet (e.g. p95 < 200 ms, 99.9% uptime, 10× traffic headroom) and how the architecture achieves them.
- **Fitness functions** — encode the constraints as automated tests (dependency direction, no cycles, module coupling, service independence) run in CI (ARCH-TST-01/02). Write them test-first per [`tdd.md`](guides://tdd.md).

```bash
# Validation tooling (examples; pick the language's equivalent)
pydeps --cluster --max-bacon 2 src/      # dependency graph (Python)
madge --circular src/                     # circular imports (JS/TS)
pytest tests/architecture/ -v             # fitness tests (ARCH-TST-01)
radon cc src/ -a -s ; radon mi src/ -s    # complexity / maintainability
```

---

## 11. Deployment Checklist

Generated from §2 — one box per requirement ID. No new requirements.

- [ ] ARCH-STRUCT-01 — style declared on all four axes (internal/deployment/communication/data)
- [ ] ARCH-STRUCT-02 — complexity justified against problem complexity (§7)
- [ ] ARCH-STRUCT-03 — dependency direction correct, 0 cycles/violations
- [ ] ARCH-STRUCT-04 — deployable units are independent as the style claims (no distributed monolith)
- [ ] ARCH-ARCH-01 — one ADR per significant decision (see `adr.md`)
- [ ] ARCH-TST-01 — architectural fitness tests exist and pass, written test-first (see `tdd.md`)
- [ ] ARCH-TST-02 — every architectural bug has a pre-fix regression test
- [ ] ARCH-OBS-01 — distributed styles emit the metrics/traces their fitness functions need (see `observability.md`)

---

**End of Software Architecture Styles Guidelines**
