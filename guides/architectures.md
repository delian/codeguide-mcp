# Software Architecture Reference Guide

This document provides a comprehensive overview of software architecture patterns, their relationships, trade-offs, and guidance for selecting the right architecture based on project requirements. This guide is language-agnostic and focuses on architectural principles and decision-making.

---

**Agent Profile**: The Software Architect
**Role**: Senior Solutions Architect & System Design Expert
**Objective**: Guide architectural decisions by understanding trade-offs, selecting appropriate patterns, and designing systems that balance maintainability, scalability, and team capabilities.
**Tools**: Architecture Decision Records (ADRs), C4 Model, UML, Domain-Driven Design (DDD), SOLID principles.

---

## 1. Core Philosophies: ARCHITECTURE-FIRST

The agent must adhere to the **ARCHITECTURE-FIRST** principles for every architectural decision:

- **A**lign with Business: Architecture serves business goals, not the reverse
- **R**ight-Size Complexity: Match architecture complexity to problem complexity
- **C**hange-Friendly: Design for evolution, not perfection
- **H**uman-Centric: Consider team skills, size, and cognitive load
- **I**nformed Trade-offs: Every decision has trade-offs; make them explicit
- **T**estable by Design: Architecture should enable, not hinder, testing
- **E**volvable Boundaries: Define boundaries that can shift as understanding grows
- **C**onsistent Patterns: Apply patterns consistently within a bounded context
- **T**echnically Sound: Base decisions on engineering principles, not hype
- **U**ndocumented is Unfinished: Architecture decisions must be recorded
- **R**eversible Decisions: Prefer decisions that can be changed later
- **E**xplicit Dependencies: Make all dependencies visible and intentional

**Additional Principles:**

- **Test-Driven Development (TDD)**: Architecture should support TDD at all levels
- **Fitness Functions**: Define measurable architectural characteristics
- **Evolutionary Architecture**: Build for change, not for a final state
- **Conway's Law Awareness**: Organizational structure influences architecture

---

## 2. The Architecture Landscape

### A. Architecture Categories Overview

```
ARCHITECTURE TAXONOMY:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  INTERNAL STRUCTURE          DEPLOYMENT & SCALE       COMMUNICATION     │
│  (How code is organized)     (How it runs)            (How parts talk)  │
│                                                                         │
│  ┌─────────────────────┐    ┌─────────────────────┐  ┌─────────────────┐│
│  │ • Layered (N-Tier)  │    │ • Monolith          │  │ • Request/Reply ││
│  │ • Hexagonal         │    │ • Modular Monolith  │  │ • Event-Driven  ││
│  │ • Clean             │    │ • Microservices     │  │ • Message Queue ││
│  │ • Onion             │    │ • Serverless        │  │ • Pub/Sub       ││
│  │ • Vertical Slice    │    │ • Service-Oriented  │  │ • Streaming     ││
│  │ • Feature-Based     │    │ • Distributed       │  │ • CQRS          ││
│  └─────────────────────┘    └─────────────────────┘  └─────────────────┘│
│                                                                         │
│  These are ORTHOGONAL - you combine them:                               │
│  Example: Microservices (deployment) + Hexagonal (internal) +           │
│           Event-Driven (communication)                                  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### B. Key Insight: Orthogonal Concerns

**CRITICAL: These architecture types solve different problems and can be combined.**

```
COMBINING ARCHITECTURES:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  Question                      │ Architecture Category                  │
│  ──────────────────────────────┼──────────────────────────────────────  │
│  How do I organize my code?    │ Hexagonal, Clean, Layered, Onion      │
│  How do I deploy my system?    │ Monolith, Microservices, Serverless   │
│  How do my components talk?    │ REST, Events, Messages, gRPC          │
│  How do I handle data?         │ CRUD, CQRS, Event Sourcing            │
│                                                                         │
│  EXAMPLE COMBINATION:                                                   │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                                                                   │   │
│  │  Microservices Deployment                                        │   │
│  │       │                                                          │   │
│  │       └── Each service uses Hexagonal Architecture               │   │
│  │              │                                                   │   │
│  │              └── Services communicate via Events                 │   │
│  │                     │                                            │   │
│  │                     └── Complex queries use CQRS                 │   │
│  │                                                                   │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Internal Structure Architectures

These define how you organize code within a deployable unit.

### A. Layered Architecture (N-Tier)

```
LAYERED ARCHITECTURE:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  The Traditional Approach - Linear Dependencies                         │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    PRESENTATION LAYER                            │   │
│  │                    (Controllers, Views, APIs)                    │   │
│  └───────────────────────────┬─────────────────────────────────────┘   │
│                              │ depends on                              │
│                              ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    BUSINESS LOGIC LAYER                          │   │
│  │                    (Services, Business Rules)                    │   │
│  └───────────────────────────┬─────────────────────────────────────┘   │
│                              │ depends on                              │
│                              ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    DATA ACCESS LAYER                             │   │
│  │                    (Repositories, ORM, Queries)                  │   │
│  └───────────────────────────┬─────────────────────────────────────┘   │
│                              │ depends on                              │
│                              ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    DATABASE                                       │   │
│  │                    (Schema is foundation)                         │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

CHARACTERISTICS:
  ✅ Simple to understand and implement
  ✅ Clear separation of concerns by technical function
  ✅ Works well for CRUD-focused applications

  ❌ Database changes ripple upward
  ❌ Business logic often leaks into other layers
  ❌ Tight coupling to infrastructure
  ❌ Difficult to test business logic in isolation
  ❌ Tends toward "Anemic Domain Model"

BEST FOR:
  • Simple CRUD applications
  • Prototypes and MVPs
  • Small applications with limited complexity
  • Teams new to structured architecture
```

### B. Hexagonal Architecture (Ports & Adapters)

```
HEXAGONAL ARCHITECTURE:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  Focus: Protect business logic from external concerns via boundaries    │
│                                                                         │
│                    DRIVING SIDE                                         │
│                    (Primary/Input)                                      │
│                         │                                               │
│     ┌───────────────────┼───────────────────┐                          │
│     │ REST    │ CLI     │ Tests   │ Events  │  ← Driving Adapters      │
│     │ Controller        │                   │                          │
│     └───────────────────┼───────────────────┘                          │
│                         │                                               │
│                    ┌────▼────┐                                         │
│                    │  PORTS  │  ← Interfaces (Driving)                 │
│                    └────┬────┘                                         │
│                         │                                               │
│     ┌───────────────────┼───────────────────────────────────┐          │
│     │                   ▼                                    │          │
│     │     ┌─────────────────────────────────────┐           │          │
│     │     │         DOMAIN / CORE               │           │          │
│     │     │  • Entities                         │           │          │
│     │     │  • Value Objects                    │           │          │
│     │     │  • Domain Services                  │           │          │
│     │     │  • Business Rules                   │           │          │
│     │     │  (NO external dependencies)         │           │          │
│     │     └─────────────────────────────────────┘           │          │
│     │                                                        │          │
│     │     ┌─────────────────────────────────────┐           │          │
│     │     │      APPLICATION SERVICES           │           │          │
│     │     │  • Use Cases / Interactors          │           │          │
│     │     │  • Orchestration                    │           │          │
│     │     └─────────────────────────────────────┘           │          │
│     │                                                        │          │
│     └───────────────────┼───────────────────────────────────┘          │
│                         │                                               │
│                    ┌────▼────┐                                         │
│                    │  PORTS  │  ← Interfaces (Driven)                  │
│                    └────┬────┘                                         │
│                         │                                               │
│     ┌───────────────────┼───────────────────┐                          │
│     │ Database  │ Email │ Payment │ Queue   │  ← Driven Adapters       │
│     │ Adapter           │                   │                          │
│     └───────────────────┼───────────────────┘                          │
│                         │                                               │
│                    DRIVEN SIDE                                          │
│                    (Secondary/Output)                                   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

KEY CONCEPTS:
  • Ports: Interfaces that define how the core interacts with the world
  • Adapters: Implementations that translate between external and internal
  • Driving: Things that drive your application (user, API, tests)
  • Driven: Things your application drives (database, email, payments)

DEPENDENCY RULE:
  Dependencies point INWARD. The core knows nothing about adapters.

BEST FOR:
  • Applications needing high testability
  • Systems with multiple input/output channels
  • Long-lived applications requiring flexibility
  • Teams comfortable with dependency inversion
```

### C. Clean Architecture (Uncle Bob)

```
CLEAN ARCHITECTURE:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  Focus: Explicit layers with clear responsibilities                     │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    FRAMEWORKS & DRIVERS                          │   │
│  │  (Web, UI, Database, Devices, External Interfaces)               │   │
│  │                                                                   │   │
│  │    ┌─────────────────────────────────────────────────────────┐   │   │
│  │    │                  INTERFACE ADAPTERS                      │   │   │
│  │    │  (Controllers, Gateways, Presenters)                     │   │   │
│  │    │                                                          │   │   │
│  │    │    ┌─────────────────────────────────────────────────┐   │   │   │
│  │    │    │              USE CASES                           │   │   │   │
│  │    │    │  (Application Business Rules)                    │   │   │   │
│  │    │    │                                                  │   │   │   │
│  │    │    │    ┌─────────────────────────────────────────┐   │   │   │   │
│  │    │    │    │              ENTITIES                   │   │   │   │   │
│  │    │    │    │  (Enterprise Business Rules)            │   │   │   │   │
│  │    │    │    └─────────────────────────────────────────┘   │   │   │   │
│  │    │    │                                                  │   │   │   │
│  │    │    └─────────────────────────────────────────────────┘   │   │   │
│  │    │                                                          │   │   │
│  │    └─────────────────────────────────────────────────────────┘   │   │
│  │                                                                   │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  DEPENDENCY RULE: Source code dependencies point INWARD only           │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

LAYER RESPONSIBILITIES:

Layer               │ Contains                    │ Depends On
────────────────────┼─────────────────────────────┼─────────────
Entities            │ Business objects, rules     │ Nothing
Use Cases           │ Application logic, flows    │ Entities
Interface Adapters  │ Controllers, presenters     │ Use Cases
Frameworks          │ Web, DB, external tools     │ All inner

KEY FEATURE - "Screaming Architecture":
  The directory structure should reveal WHAT the app does,
  not WHICH framework it uses.

  ✅ CORRECT:
  src/
  ├── healthcare/
  │   ├── patient/
  │   ├── appointment/
  │   └── prescription/

  ❌ WRONG:
  src/
  ├── controllers/
  ├── services/
  ├── repositories/

DIFFERENCE FROM HEXAGONAL:
  • More prescriptive about layer count (4 explicit layers)
  • Separates Entities from Use Cases
  • Explicit Presenter pattern for output
  • Stronger emphasis on standardized structure

BEST FOR:
  • Teams wanting strict, standardized layer definitions
  • Large applications with many developers
  • Enterprise systems requiring governance
```

### D. Onion Architecture (Jeffrey Palermo)

```
ONION ARCHITECTURE:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  Focus: Domain Model as the absolute core                               │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                         INFRASTRUCTURE                           │   │
│  │  (Database, File System, External Services, UI)                  │   │
│  │                                                                   │   │
│  │    ┌─────────────────────────────────────────────────────────┐   │   │
│  │    │                    APPLICATION SERVICES                  │   │   │
│  │    │  (Orchestration, Use Cases, Application Logic)           │   │   │
│  │    │                                                          │   │   │
│  │    │    ┌─────────────────────────────────────────────────┐   │   │   │
│  │    │    │                DOMAIN SERVICES                   │   │   │   │
│  │    │    │  (Pure Business Logic, Stateless)                │   │   │   │
│  │    │    │                                                  │   │   │   │
│  │    │    │    ┌─────────────────────────────────────────┐   │   │   │   │
│  │    │    │    │              DOMAIN MODEL               │   │   │   │   │
│  │    │    │    │  (Entities, Value Objects, Aggregates)  │   │   │   │   │
│  │    │    │    └─────────────────────────────────────────┘   │   │   │   │
│  │    │    │                                                  │   │   │   │
│  │    │    └─────────────────────────────────────────────────┘   │   │   │
│  │    │                                                          │   │   │
│  │    └─────────────────────────────────────────────────────────┘   │   │
│  │                                                                   │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

KEY DISTINCTION FROM OTHERS:
  • Emphasizes "Application Core" must compile WITHOUT infrastructure
  • Strong distinction between Domain Services (pure) and
    Application Services (orchestration)
  • Heavy DDD (Domain-Driven Design) influence

DOMAIN SERVICES vs APPLICATION SERVICES:

Domain Services (Pure Logic):
  • Stateless operations on domain objects
  • No infrastructure dependencies
  • Example: PricingService.calculateDiscount(order, customer)

Application Services (Orchestration):
  • Coordinate multiple domain operations
  • May call infrastructure (via interfaces)
  • Example: PlaceOrderService.execute(command)

BEST FOR:
  • Complex enterprise applications
  • Rich business rules requiring DDD
  • Long-lived systems (10+ years)
  • Teams with strong DDD experience
```

### E. Vertical Slice Architecture

```
VERTICAL SLICE ARCHITECTURE:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  Focus: Organize by FEATURE, not by LAYER                               │
│                                                                         │
│  TRADITIONAL (Layered):          VERTICAL SLICE:                        │
│                                                                         │
│  src/                            src/                                   │
│  ├── Controllers/                ├── Features/                          │
│  │   ├── OrderController         │   ├── CreateOrder/                   │
│  │   └── CustomerController      │   │   ├── CreateOrderCommand         │
│  ├── Services/                   │   │   ├── CreateOrderHandler         │
│  │   ├── OrderService            │   │   ├── CreateOrderValidator       │
│  │   └── CustomerService         │   │   └── CreateOrderEndpoint        │
│  ├── Repositories/               │   ├── GetOrder/                      │
│  │   ├── OrderRepository         │   │   ├── GetOrderQuery              │
│  │   └── CustomerRepository      │   │   ├── GetOrderHandler            │
│  └── Models/                     │   │   └── GetOrderEndpoint           │
│      ├── Order                   │   └── CancelOrder/                   │
│      └── Customer                │       ├── CancelOrderCommand         │
│                                  │       └── ...                        │
│                                  └── Shared/                            │
│                                      └── Domain/                        │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

PRINCIPLES:
  1. Each feature is a complete slice through all layers
  2. Features are independent - change one without affecting others
  3. Code that changes together lives together
  4. Minimize shared code (duplication > wrong abstraction)

SLICE STRUCTURE:

CreateOrder/
├── CreateOrderCommand.cs      # Input DTO
├── CreateOrderHandler.cs      # Business logic + persistence
├── CreateOrderValidator.cs    # Validation rules
├── CreateOrderEndpoint.cs     # HTTP endpoint
└── CreateOrderResponse.cs     # Output DTO

WHEN A SLICE HANDLES EVERYTHING:

Request → Endpoint → Handler → Database → Response
           │           │
           │           └── Contains: validation, business rules,
           │                        persistence, response mapping
           └── Maps HTTP to/from command/response

BEST FOR:
  • CRUD-heavy applications
  • Rapid feature development
  • Teams that struggle with layer coupling
  • Applications with independent features
  • Brownfield refactoring of legacy systems

CAUTION:
  • Can lead to duplication
  • Complex domain logic may need shared domain layer
  • Not ideal for heavily interconnected features
```

### F. Architecture Comparison Matrix

```
INTERNAL ARCHITECTURE COMPARISON:

┌──────────────────┬───────────────────────────────────────────────────────┐
│ Architecture     │ Key Characteristics                                   │
├──────────────────┼───────────────────────────────────────────────────────┤
│ Layered          │ • Linear dependencies (top-down)                      │
│                  │ • Database is foundation                              │
│                  │ • Simple, widely understood                           │
│                  │ • Risk: Anemic domain, tight coupling                 │
├──────────────────┼───────────────────────────────────────────────────────┤
│ Hexagonal        │ • Domain at center, ports as boundaries               │
│                  │ • Adapters translate external ↔ internal              │
│                  │ • Highly testable (mock adapters)                     │
│                  │ • Focus: Boundary protection                          │
├──────────────────┼───────────────────────────────────────────────────────┤
│ Clean            │ • 4 explicit concentric layers                        │
│                  │ • Entities separate from Use Cases                    │
│                  │ • "Screaming Architecture" principle                  │
│                  │ • Focus: Standardization                              │
├──────────────────┼───────────────────────────────────────────────────────┤
│ Onion            │ • Domain Model + Domain Services at core              │
│                  │ • Core must compile without infrastructure            │
│                  │ • Strong DDD alignment                                │
│                  │ • Focus: Pure domain model                            │
├──────────────────┼───────────────────────────────────────────────────────┤
│ Vertical Slice   │ • Organize by feature, not layer                      │
│                  │ • Each slice is independent                           │
│                  │ • Minimize shared abstractions                        │
│                  │ • Focus: Feature independence                         │
└──────────────────┴───────────────────────────────────────────────────────┘

THE "INVERTED FAMILY" COMPARISON:
(Hexagonal, Clean, Onion share the same DNA: Dependency Inversion)

┌────────────────┬──────────────┬───────────────────┬──────────────────────┐
│ Aspect         │ Hexagonal    │ Clean             │ Onion                │
├────────────────┼──────────────┼───────────────────┼──────────────────────┤
│ Visual         │ Hexagon      │ Concentric circles│ Onion layers         │
│ Terminology    │ Ports,       │ Entities, Use     │ Domain Model,        │
│                │ Adapters     │ Cases, Gateways   │ Domain Services      │
│ Primary Focus  │ Boundaries & │ Standardized      │ Pure domain model    │
│                │ Testing      │ layers            │                      │
│ Prescriptive   │ Less         │ More              │ Medium               │
│ DDD Alignment  │ Compatible   │ Compatible        │ Strong               │
│ Origin         │ Cockburn     │ Martin (Uncle Bob)│ Palermo              │
└────────────────┴──────────────┴───────────────────┴──────────────────────┘
```

---

## 4. Deployment Architectures

These define how you package and deploy your system.

### A. Monolith

```
MONOLITHIC ARCHITECTURE:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  Single deployable unit containing all functionality                    │
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │                        MONOLITH                                    │ │
│  │                                                                    │ │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐          │ │
│  │  │  Orders  │  │ Customers│  │ Payments │  │ Inventory│          │ │
│  │  │  Module  │  │  Module  │  │  Module  │  │  Module  │          │ │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘          │ │
│  │                                                                    │ │
│  │  ┌──────────────────────────────────────────────────────────────┐ │ │
│  │  │                    SHARED DATABASE                            │ │ │
│  │  └──────────────────────────────────────────────────────────────┘ │ │
│  │                                                                    │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

CHARACTERISTICS:
  ✅ Simple deployment (one artifact)
  ✅ Simple development environment
  ✅ Easy debugging (single process)
  ✅ No network latency between modules
  ✅ ACID transactions across modules

  ❌ Scaling is all-or-nothing
  ❌ Single point of failure
  ❌ Long build/test times as it grows
  ❌ Technology lock-in
  ❌ Difficult for large teams

BEST FOR:
  • Startups and MVPs
  • Small to medium applications
  • Small teams (< 10 developers)
  • Simple domains
  • Budget constraints
```

### B. Modular Monolith

```
MODULAR MONOLITH ARCHITECTURE:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  Monolith with strict module boundaries (best of both worlds)           │
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │                     MODULAR MONOLITH                               │ │
│  │                                                                    │ │
│  │  ┌─────────────────┐    ┌─────────────────┐                       │ │
│  │  │  Orders Module  │    │ Customers Module│                       │ │
│  │  │ ┌─────────────┐ │    │ ┌─────────────┐ │                       │ │
│  │  │ │   Domain    │ │    │ │   Domain    │ │                       │ │
│  │  │ ├─────────────┤ │    │ ├─────────────┤ │                       │ │
│  │  │ │ Application │ │    │ │ Application │ │                       │ │
│  │  │ ├─────────────┤ │    │ ├─────────────┤ │                       │ │
│  │  │ │Infrastructure│ │◄──┤ │Infrastructure│ │  ← Communication     │ │
│  │  │ ├─────────────┤ │    │ ├─────────────┤ │    via PUBLIC API    │ │
│  │  │ │ Public API  │ │    │ │ Public API  │ │    only              │ │
│  │  │ └─────────────┘ │    │ └─────────────┘ │                       │ │
│  │  │       │         │    │       │         │                       │ │
│  │  │       ▼         │    │       ▼         │                       │ │
│  │  │  [Own Schema]   │    │  [Own Schema]   │  ← Logical separation │ │
│  │  └─────────────────┘    └─────────────────┘                       │ │
│  │                                                                    │ │
│  │  ┌──────────────────────────────────────────────────────────────┐ │ │
│  │  │                    SHARED DATABASE                            │ │ │
│  │  │         (but modules only access OWN schema)                  │ │ │
│  │  └──────────────────────────────────────────────────────────────┘ │ │
│  │                                                                    │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

KEY RULES:
  1. Modules can ONLY communicate through their PUBLIC APIs
  2. Modules CANNOT access each other's internal classes
  3. Modules CANNOT directly query each other's database tables
  4. Each module has its own schema/namespace in the database

WHY THIS IS THE "SWEET SPOT":
  • Simple deployment like a monolith
  • Clean boundaries like microservices
  • Easy to extract to microservices later
  • Single database transaction when needed
  • Lower operational complexity

EXTRACTION PATH TO MICROSERVICES:
  1. Module already has clean boundaries
  2. Extract module to separate service
  3. Replace in-process calls with network calls
  4. Separate the database schema

BEST FOR:
  • Growing startups (most common recommendation)
  • Medium complexity domains
  • Teams wanting microservice benefits without overhead
  • Path to eventual microservices
```

### C. Microservices

```
MICROSERVICES ARCHITECTURE:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  Small, independently deployable services                               │
│                                                                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐   │
│  │   Orders    │  │  Customers  │  │  Payments   │  │  Inventory  │   │
│  │   Service   │  │   Service   │  │   Service   │  │   Service   │   │
│  │             │  │             │  │             │  │             │   │
│  │ ┌─────────┐ │  │ ┌─────────┐ │  │ ┌─────────┐ │  │ ┌─────────┐ │   │
│  │ │Hexagonal│ │  │ │Hexagonal│ │  │ │  Clean  │ │  │ │ Layered │ │   │
│  │ │Internal │ │  │ │Internal │ │  │ │Internal │ │  │ │Internal │ │   │
│  │ └─────────┘ │  │ └─────────┘ │  │ └─────────┘ │  │ └─────────┘ │   │
│  │      │      │  │      │      │  │      │      │  │      │      │   │
│  │      ▼      │  │      ▼      │  │      ▼      │  │      ▼      │   │
│  │ [Own DB]    │  │ [Own DB]    │  │ [Own DB]    │  │ [Own DB]    │   │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘   │
│         │                │                │                │          │
│         └────────────────┴────────────────┴────────────────┘          │
│                                  │                                     │
│                         Network Communication                          │
│                    (REST, gRPC, Events, Messages)                      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

KEY CHARACTERISTICS:
  • Each service is independently deployable
  • Each service owns its data (no shared databases)
  • Services communicate over the network
  • Different services can use different technologies
  • Teams can work independently

ANTI-PATTERN - Distributed Monolith:
  ❌ Services that share a database
  ❌ Services that must be deployed together
  ❌ Synchronous call chains across services
  ❌ Tight coupling between services

PREREQUISITES:
  • Mature DevOps (CI/CD, monitoring, logging)
  • Experienced team
  • Well-understood domain boundaries
  • Infrastructure for distributed systems

TRADE-OFFS:
  ✅ Independent scaling
  ✅ Technology flexibility
  ✅ Team autonomy
  ✅ Fault isolation

  ❌ Network latency
  ❌ Distributed transactions complexity
  ❌ Operational complexity
  ❌ Testing complexity
  ❌ Debugging difficulty

BEST FOR:
  • Large organizations with multiple teams
  • Systems needing independent scaling
  • Complex domains with clear boundaries
  • Teams with strong DevOps capabilities
```

### D. Serverless Architecture

```
SERVERLESS ARCHITECTURE:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  Functions as the unit of deployment                                    │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                        API GATEWAY                                │  │
│  │            (Routing, Authentication, Rate Limiting)               │  │
│  └─────┬────────────────┬────────────────┬────────────────┬─────────┘  │
│        │                │                │                │            │
│        ▼                ▼                ▼                ▼            │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐         │
│  │ Function │    │ Function │    │ Function │    │ Function │         │
│  │ Create   │    │ Get      │    │ Update   │    │ Delete   │         │
│  │ Order    │    │ Order    │    │ Order    │    │ Order    │         │
│  └────┬─────┘    └────┬─────┘    └────┬─────┘    └────┬─────┘         │
│       │               │               │               │                │
│       └───────────────┴───────────────┴───────────────┘                │
│                              │                                          │
│                              ▼                                          │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                    MANAGED SERVICES                               │  │
│  │  (DynamoDB, S3, SQS, SNS, Aurora Serverless, etc.)               │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

CHARACTERISTICS:
  • Pay per invocation (no idle costs)
  • Auto-scaling to zero and to infinity
  • No server management
  • Event-driven by nature
  • Stateless functions

CHALLENGES:
  • Cold start latency
  • Vendor lock-in
  • Limited execution time
  • Complex debugging
  • State management

BEST FOR:
  • Event-driven workloads
  • Variable/unpredictable traffic
  • Simple APIs with independent endpoints
  • Cost-sensitive applications
  • Background processing
```

### E. Deployment Architecture Comparison

```
DEPLOYMENT ARCHITECTURE SELECTION:

┌──────────────────┬───────────────────────────────────────────────────────┐
│ Architecture     │ Choose When                                           │
├──────────────────┼───────────────────────────────────────────────────────┤
│ Monolith         │ • Starting new project                                │
│                  │ • Team < 10 developers                                │
│                  │ • Simple domain                                       │
│                  │ • Need fast time-to-market                            │
├──────────────────┼───────────────────────────────────────────────────────┤
│ Modular Monolith │ • Growing complexity                                  │
│                  │ • Want microservice benefits without overhead         │
│                  │ • Planning for eventual extraction                    │
│                  │ • Team 10-30 developers                               │
├──────────────────┼───────────────────────────────────────────────────────┤
│ Microservices    │ • Large organization                                  │
│                  │ • Multiple autonomous teams                           │
│                  │ • Need independent scaling                            │
│                  │ • Strong DevOps maturity                              │
│                  │ • Clear bounded contexts                              │
├──────────────────┼───────────────────────────────────────────────────────┤
│ Serverless       │ • Event-driven workloads                              │
│                  │ • Highly variable traffic                             │
│                  │ • Cost optimization critical                          │
│                  │ • Simple, stateless operations                        │
└──────────────────┴───────────────────────────────────────────────────────┘
```

---

## 5. Communication Architectures

These define how components communicate with each other.

### A. Request/Response (Synchronous)

```
REQUEST/RESPONSE PATTERN:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  Service A ──────────────────────────────► Service B                   │
│            │        Request                     │                       │
│            │                                    │                       │
│            │                                    │ Process               │
│            │                                    │                       │
│            ◄──────────────────────────────────  │                       │
│                    Response                     │                       │
│                                                                         │
│  Caller WAITS for response (blocking)                                   │
│                                                                         │
│  PROTOCOLS:                                                             │
│    • REST over HTTP (most common)                                       │
│    • gRPC (high performance, typed)                                     │
│    • GraphQL (flexible queries)                                         │
│                                                                         │
│  TRADE-OFFS:                                                            │
│    ✅ Simple to understand and debug                                    │
│    ✅ Immediate feedback                                                │
│    ✅ Strong consistency                                                │
│    ❌ Tight coupling (caller waits)                                     │
│    ❌ Cascading failures                                                │
│    ❌ Service availability required                                     │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### B. Event-Driven (Asynchronous)

```
EVENT-DRIVEN ARCHITECTURE:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  EVENTS = Facts that happened (past tense)                              │
│  "OrderPlaced", "PaymentReceived", "InventoryReserved"                  │
│                                                                         │
│  ┌─────────────┐                              ┌─────────────┐          │
│  │   Orders    │     OrderPlaced              │  Payments   │          │
│  │   Service   │  ─────────────────────────►  │   Service   │          │
│  └─────────────┘           │                  └─────────────┘          │
│                            │                                            │
│                            │                  ┌─────────────┐          │
│                            └─────────────────►│  Inventory  │          │
│                            │                  │   Service   │          │
│                            │                  └─────────────┘          │
│                            │                                            │
│                            │                  ┌─────────────┐          │
│                            └─────────────────►│  Analytics  │          │
│                                               │   Service   │          │
│                                               └─────────────┘          │
│                                                                         │
│  CHARACTERISTICS:                                                       │
│    • Publisher doesn't know about subscribers                           │
│    • Loose coupling between services                                    │
│    • Asynchronous processing                                            │
│    • Natural audit log (event history)                                  │
│                                                                         │
│  TRADE-OFFS:                                                            │
│    ✅ Loose coupling                                                    │
│    ✅ High scalability                                                  │
│    ✅ Resilience (no immediate dependencies)                            │
│    ❌ Eventual consistency                                              │
│    ❌ Complex debugging                                                 │
│    ❌ Event ordering challenges                                         │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### C. Message Queue (Command Pattern)

```
MESSAGE QUEUE PATTERN:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  COMMANDS = Instructions to do something                                │
│  "ProcessPayment", "SendEmail", "GenerateReport"                        │
│                                                                         │
│  ┌─────────────┐    ┌──────────────────┐    ┌─────────────┐            │
│  │  Producer   │───►│   Message Queue   │───►│  Consumer   │            │
│  │  (Sender)   │    │                   │    │ (Processor) │            │
│  └─────────────┘    │  [Message 1]      │    └─────────────┘            │
│                     │  [Message 2]      │                               │
│                     │  [Message 3]      │    ┌─────────────┐            │
│                     │  ...              │───►│  Consumer   │            │
│                     │                   │    │ (Processor) │            │
│                     └──────────────────┘    └─────────────┘            │
│                                                                         │
│  KEY FEATURES:                                                          │
│    • Messages persist until processed                                   │
│    • Multiple consumers can process in parallel                         │
│    • Guaranteed delivery (at-least-once)                                │
│    • Load leveling (buffer traffic spikes)                              │
│                                                                         │
│  COMMON IMPLEMENTATIONS:                                                │
│    • RabbitMQ (traditional message broker)                              │
│    • Amazon SQS (managed queue)                                         │
│    • Azure Service Bus                                                  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### D. Event Streaming (Kafka Pattern)

```
EVENT STREAMING:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  Events stored as an immutable, ordered log (commit log)                │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                         KAFKA TOPIC                               │  │
│  │                                                                   │  │
│  │  ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐              │  │
│  │  │  0  │  1  │  2  │  3  │  4  │  5  │  6  │ ... │              │  │
│  │  └─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘              │  │
│  │                   │         │         │                          │  │
│  │                   │         │         │                          │  │
│  │              Consumer A  Consumer B  Consumer C                  │  │
│  │              (offset 3)  (offset 5)  (offset 2)                  │  │
│  │                                                                   │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  KEY DIFFERENCES FROM MESSAGE QUEUE:                                    │
│    • Events are NOT deleted after consumption                           │
│    • Consumers track their own position (offset)                        │
│    • Replay possible from any point                                     │
│    • Event log as source of truth                                       │
│                                                                         │
│  USE CASES:                                                             │
│    • Event sourcing                                                     │
│    • State reconstruction                                               │
│    • Audit trails                                                       │
│    • Real-time analytics                                                │
│    • Stream processing                                                  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Data Architectures

These define how data is managed and accessed.

### A. Traditional CRUD

```
CRUD PATTERN:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  Single model for both reading and writing                              │
│                                                                         │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                 │
│  │   Create    │    │    Read     │    │   Update    │                 │
│  │  (INSERT)   │    │  (SELECT)   │    │  (UPDATE)   │                 │
│  └──────┬──────┘    └──────┬──────┘    └──────┬──────┘                 │
│         │                  │                  │                         │
│         └──────────────────┴──────────────────┘                         │
│                            │                                            │
│                            ▼                                            │
│         ┌──────────────────────────────────────┐                       │
│         │            SINGLE MODEL              │                       │
│         │                                      │                       │
│         │     Used for BOTH queries AND        │                       │
│         │     commands (same schema)           │                       │
│         │                                      │                       │
│         └──────────────────────────────────────┘                       │
│                            │                                            │
│                            ▼                                            │
│         ┌──────────────────────────────────────┐                       │
│         │          SINGLE DATABASE             │                       │
│         └──────────────────────────────────────┘                       │
│                                                                         │
│  BEST FOR:                                                              │
│    • Simple applications                                                │
│    • Similar read/write patterns                                        │
│    • Small data volumes                                                 │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### B. CQRS (Command Query Responsibility Segregation)

```
CQRS PATTERN:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  SEPARATE models for reading and writing                                │
│                                                                         │
│        COMMANDS (Write)                    QUERIES (Read)               │
│        ┌─────────────────┐                ┌─────────────────┐          │
│        │ CreateOrder     │                │ GetOrderDetails │          │
│        │ UpdateOrder     │                │ ListOrders      │          │
│        │ CancelOrder     │                │ SearchOrders    │          │
│        └────────┬────────┘                └────────┬────────┘          │
│                 │                                  │                    │
│                 ▼                                  ▼                    │
│        ┌─────────────────┐                ┌─────────────────┐          │
│        │   WRITE MODEL   │                │   READ MODEL    │          │
│        │                 │                │                 │          │
│        │ • Rich domain   │  ──────────►   │ • Denormalized  │          │
│        │ • Validation    │   Projection   │ • Optimized for │          │
│        │ • Business rules│                │   queries       │          │
│        │ • Normalized    │                │ • Pre-computed  │          │
│        └────────┬────────┘                └────────┬────────┘          │
│                 │                                  │                    │
│                 ▼                                  ▼                    │
│        ┌─────────────────┐                ┌─────────────────┐          │
│        │  WRITE DATABASE │                │  READ DATABASE  │          │
│        │  (PostgreSQL)   │                │  (Elasticsearch)│          │
│        └─────────────────┘                └─────────────────┘          │
│                                                                         │
│  WHY USE CQRS:                                                          │
│    • Reads and writes have different performance requirements           │
│    • Complex queries that don't fit the write model                     │
│    • Scale reads and writes independently                               │
│    • Optimize each model for its purpose                                │
│                                                                         │
│  TRADE-OFFS:                                                            │
│    ✅ Optimized read performance                                        │
│    ✅ Independent scaling                                               │
│    ✅ Simpler queries                                                   │
│    ❌ Eventual consistency between models                               │
│    ❌ Increased complexity                                              │
│    ❌ Data synchronization challenges                                   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### C. Event Sourcing

```
EVENT SOURCING PATTERN:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  Store EVENTS, not current state                                        │
│  Current state = replay all events                                      │
│                                                                         │
│  TRADITIONAL:                      EVENT SOURCING:                      │
│  ┌──────────────────┐              ┌──────────────────────────────────┐│
│  │ orders           │              │ events                           ││
│  │─────────────────-│              │──────────────────────────────────││
│  │ id    │ status   │              │ id │ type          │ data       ││
│  │───────┼──────────│              │────┼───────────────┼────────────││
│  │ 123   │ shipped  │              │ 1  │ OrderCreated  │ {id:123,...}│
│  │       │          │              │ 2  │ ItemAdded     │ {item:...} ││
│  └──────────────────┘              │ 3  │ PaymentMade   │ {amount:..}││
│                                    │ 4  │ OrderShipped  │ {date:...} ││
│  Only current state                └──────────────────────────────────┘│
│                                    Full history preserved              │
│                                                                         │
│  REPLAYING EVENTS TO GET STATE:                                         │
│                                                                         │
│  OrderCreated ──► ItemAdded ──► PaymentMade ──► OrderShipped           │
│       │               │              │               │                  │
│       ▼               ▼              ▼               ▼                  │
│  Order{               Order{         Order{         Order{             │
│    status:created      +item         status:paid    status:shipped     │
│  }                   }              }              }                    │
│                                                                         │
│  BENEFITS:                                                              │
│    • Complete audit trail                                               │
│    • Temporal queries ("what was the state on date X?")                 │
│    • Debug by replaying events                                          │
│    • Rebuild read models from scratch                                   │
│                                                                         │
│  CHALLENGES:                                                            │
│    • Event schema evolution                                             │
│    • Performance (long event streams)                                   │
│    • Complexity                                                         │
│    • Learning curve                                                     │
│                                                                         │
│  OFTEN COMBINED WITH CQRS:                                              │
│    Events (write) ──► Projections ──► Read Models (query)              │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 7. Resilience Patterns

Essential patterns for distributed systems.

### A. Core Resilience Patterns

```
RESILIENCE PATTERNS:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  CIRCUIT BREAKER                                                        │
│  ─────────────────                                                      │
│  Prevent cascading failures by stopping calls to failing services       │
│                                                                         │
│  States: CLOSED ──(failures)──► OPEN ──(timeout)──► HALF-OPEN          │
│             │                     │                      │              │
│             │                     │                      │              │
│        (success)            (reject all)          (test one call)       │
│             │                     │                      │              │
│             ◄─────────────────────┴──────────────────────┘              │
│                                                                         │
│  RETRY WITH BACKOFF                                                     │
│  ──────────────────                                                     │
│  Retry failed operations with increasing delays                         │
│                                                                         │
│  Attempt 1 ──(fail)──► wait 1s ──► Attempt 2 ──(fail)──► wait 2s ──►...│
│                                                                         │
│  Exponential backoff: wait = base * 2^attempt (+ jitter)               │
│                                                                         │
│  BULKHEAD                                                               │
│  ────────                                                               │
│  Isolate failures to prevent spreading                                  │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │   │
│  │  │ Orders Pool  │  │ Payments Pool│  │ Reports Pool │          │   │
│  │  │ (10 threads) │  │ (5 threads)  │  │ (3 threads)  │          │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘          │   │
│  │  If Payments fails, Orders and Reports continue working         │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  TIMEOUT                                                                │
│  ───────                                                                │
│  Don't wait forever; fail fast and release resources                    │
│                                                                         │
│  FALLBACK                                                               │
│  ────────                                                               │
│  Provide alternative behavior when primary fails                        │
│  • Return cached data                                                   │
│  • Return default value                                                 │
│  • Call backup service                                                  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### B. Consistency Patterns

```
DISTRIBUTED CONSISTENCY:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  SAGA PATTERN                                                           │
│  ────────────                                                           │
│  Distributed transaction as a sequence of local transactions            │
│                                                                         │
│  Choreography (Events):                                                 │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐             │
│  │ Order   │───►│ Payment │───►│Inventory│───►│Shipping │             │
│  │ Service │    │ Service │    │ Service │    │ Service │             │
│  └─────────┘    └─────────┘    └─────────┘    └─────────┘             │
│     OrderPlaced   PaymentMade   InventoryReserved  ShipmentCreated     │
│                                                                         │
│  If any step fails, compensating transactions roll back:                │
│  InventoryReserved ──(fail)──► RefundPayment ──► CancelOrder           │
│                                                                         │
│  Orchestration (Central Coordinator):                                   │
│                    ┌───────────────────┐                               │
│                    │    Orchestrator   │                               │
│                    │   (Saga Manager)  │                               │
│                    └─────────┬─────────┘                               │
│                              │                                          │
│         ┌────────────────────┼────────────────────┐                    │
│         ▼                    ▼                    ▼                    │
│    ┌─────────┐         ┌─────────┐         ┌─────────┐                │
│    │ Payment │         │Inventory│         │Shipping │                │
│    │ Service │         │ Service │         │ Service │                │
│    └─────────┘         └─────────┘         └─────────┘                │
│                                                                         │
│  OUTBOX PATTERN                                                         │
│  ──────────────                                                         │
│  Ensure event publishing and database update are atomic                 │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ Transaction:                                                     │   │
│  │   1. UPDATE orders SET status = 'completed'                      │   │
│  │   2. INSERT INTO outbox (event_type, payload)                    │   │
│  │      VALUES ('OrderCompleted', '{...}')                          │   │
│  │ COMMIT                                                           │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  Separate process reads outbox and publishes to message broker          │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 8. Architecture Decision Framework

### A. Selection Criteria

```
ARCHITECTURE SELECTION MATRIX:

┌──────────────────────────────────────────────────────────────────────────┐
│ If your project is...           │ Recommended Architecture              │
├─────────────────────────────────┼───────────────────────────────────────┤
│ Simple CRUD / Prototype         │ Layered (MVC)                         │
│                                 │ Keep it simple, don't over-engineer   │
├─────────────────────────────────┼───────────────────────────────────────┤
│ Standard Business Application   │ Hexagonal or Modular Monolith         │
│                                 │ Balance of structure and speed        │
├─────────────────────────────────┼───────────────────────────────────────┤
│ Complex Domain / Long Lifecycle │ Clean Architecture + DDD              │
│                                 │ Strict rules to prevent rot           │
├─────────────────────────────────┼───────────────────────────────────────┤
│ High Scale / Distributed        │ Microservices                         │
│                                 │ With Hexagonal internals per service  │
├─────────────────────────────────┼───────────────────────────────────────┤
│ High Concurrency / Real-time    │ Event-Driven + CQRS                   │
│                                 │ Optimize for throughput               │
├─────────────────────────────────┼───────────────────────────────────────┤
│ Variable Traffic / Cost Focus   │ Serverless                            │
│                                 │ Pay only for usage                    │
└─────────────────────────────────┴───────────────────────────────────────┘

TEAM SIZE CONSIDERATIONS:

┌──────────────────┬───────────────────────────────────────────────────────┐
│ Team Size        │ Recommended Approach                                  │
├──────────────────┼───────────────────────────────────────────────────────┤
│ 1-5 developers   │ Monolith (potentially with Hexagonal internal)        │
│ 5-15 developers  │ Modular Monolith                                      │
│ 15-50 developers │ Modular Monolith or careful Microservices             │
│ 50+ developers   │ Microservices (team per service)                      │
└──────────────────┴───────────────────────────────────────────────────────┘

COMPLEXITY ALIGNMENT:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  Architecture complexity should match problem complexity                │
│                                                                         │
│                                                                         │
│  Architecture   │ Simple         Medium         Complex                 │
│  Complexity     │ Problem        Problem        Problem                 │
│  ──────────────────────────────────────────────────────────────────     │
│                 │                                                       │
│  Simple         │ ✅ Perfect     ❌ Under-      ❌ Dangerous            │
│  (Layered)      │    fit         engineered                             │
│                 │                                                       │
│  Medium         │ ❌ Over-       ✅ Perfect     ❌ May struggle         │
│  (Hexagonal)    │ engineered        fit                                 │
│                 │                                                       │
│  Complex        │ ❌ Massively   ❌ Over-       ✅ Appropriate          │
│  (Microservices)│ over-engineered engineered       complexity           │
│                 │                                                       │
└─────────────────────────────────────────────────────────────────────────┘
```

### B. Decision Questions

```
ARCHITECTURE DECISION QUESTIONS:

1. DEPLOYMENT
   □ How many teams will work on this?
   □ Do different parts need to scale independently?
   □ What is our DevOps maturity?
   □ What is our budget for infrastructure?

2. INTERNAL STRUCTURE
   □ How complex is the business logic?
   □ How important is testability?
   □ How likely are infrastructure changes?
   □ What is the team's experience level?

3. COMMUNICATION
   □ Do we need immediate responses?
   □ Can we tolerate eventual consistency?
   □ What is our reliability requirement?
   □ Do we need to replay/audit events?

4. DATA
   □ Are read and write patterns similar?
   □ Do we need historical data?
   □ What consistency model is acceptable?
   □ What are the performance requirements?

5. EVOLUTION
   □ How long will this system live?
   □ How much will requirements change?
   □ What is our ability to refactor later?
   □ What are the most uncertain areas?
```

---

## 9. Anti-Patterns to Avoid

### A. Common Mistakes

```
ARCHITECTURE ANTI-PATTERNS:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  ❌ DISTRIBUTED MONOLITH                                                │
│  ──────────────────────                                                 │
│  Microservices that share a database or must deploy together            │
│  → You have all the complexity of microservices with none of the        │
│    benefits                                                             │
│                                                                         │
│  ❌ RESUME-DRIVEN DEVELOPMENT                                           │
│  ────────────────────────                                               │
│  Choosing architecture to learn new technologies, not to solve problems │
│  → The best architecture is the simplest one that works                 │
│                                                                         │
│  ❌ PREMATURE MICROSERVICES                                             │
│  ──────────────────────────                                             │
│  Starting with microservices before understanding domain boundaries     │
│  → Start with a modular monolith, extract when boundaries are clear     │
│                                                                         │
│  ❌ BIG BALL OF MUD                                                     │
│  ───────────────────                                                    │
│  No clear structure, everything depends on everything                   │
│  → Enforce boundaries, even in a monolith                               │
│                                                                         │
│  ❌ GOLDEN HAMMER                                                       │
│  ───────────────                                                        │
│  Using the same architecture for every problem                          │
│  → Different problems need different solutions                          │
│                                                                         │
│  ❌ CARGO CULT ARCHITECTURE                                             │
│  ──────────────────────────                                             │
│  Copying Netflix/Amazon architecture without their context              │
│  → You're not Netflix. Solve YOUR problems.                             │
│                                                                         │
│  ❌ ANALYSIS PARALYSIS                                                  │
│  ────────────────────                                                   │
│  Spending too long deciding on architecture                             │
│  → Make reversible decisions quickly, iterate                           │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 10. Architecture Documentation

### A. Essential Documentation

```
ARCHITECTURE DOCUMENTATION:

1. ARCHITECTURE DECISION RECORDS (ADRs)
   Document every significant architecture decision

   Template:
   ┌─────────────────────────────────────────────────────────────────────┐
   │ # ADR-001: Use Hexagonal Architecture                               │
   │                                                                     │
   │ ## Status: Accepted                                                 │
   │                                                                     │
   │ ## Context                                                          │
   │ We need to choose an internal architecture for our services...      │
   │                                                                     │
   │ ## Decision                                                         │
   │ We will use Hexagonal Architecture because...                       │
   │                                                                     │
   │ ## Consequences                                                     │
   │ Positive: High testability, framework independence...               │
   │ Negative: Steeper learning curve for junior developers...           │
   └─────────────────────────────────────────────────────────────────────┘

2. C4 MODEL DIAGRAMS
   Four levels of abstraction:

   Level 1: System Context
   └── How does system interact with users and other systems?

   Level 2: Container
   └── What are the major deployable units?

   Level 3: Component
   └── What are the major components within each container?

   Level 4: Code
   └── Class diagrams (only for complex areas)

3. QUALITY ATTRIBUTES
   Document non-functional requirements and how architecture addresses them:
   • Performance: Response time < 200ms for 95th percentile
   • Scalability: Support 10x current traffic within 1 hour
   • Availability: 99.9% uptime
   • Security: All data encrypted at rest and in transit
```

---

## 11. Summary

### Core Recommendations

| Scenario | Internal Structure | Deployment | Communication |
|----------|-------------------|------------|---------------|
| Startup/MVP | Layered or Hexagonal | Monolith | REST |
| Growing Product | Hexagonal | Modular Monolith | REST + Events |
| Enterprise | Clean/Onion | Microservices | Events + gRPC |
| High Scale | Hexagonal per service | Microservices | Event Streaming |
| Variable Load | Vertical Slice | Serverless | Events |

### Golden Rules

1. **Start simple**: Begin with a monolith, extract later
2. **Align complexity**: Match architecture complexity to problem complexity
3. **Consider the team**: Architecture should fit team skills and size
4. **Document decisions**: Use ADRs to record why, not just what
5. **Design for change**: Make boundaries clear for future evolution
6. **Test everything**: Architecture should enable, not hinder, testing

### Remember

> "The goal of software architecture is to minimize the human resources required to build and maintain the required system." — Robert C. Martin

> "Architecture is about the important stuff. Whatever that is." — Ralph Johnson

> "The best architecture is the simplest one that solves the problem."

---

## Related Guides

- **[hexagonal.md](hexagonal.md)**: Detailed Hexagonal Architecture implementation guide
- **[cleanarch.md](cleanarch.md)**: Clean Architecture implementation guide
- **[microservices.md](microservices.md)**: Microservices architecture patterns
- **[kafka.md](kafka.md)**: Event streaming with Apache Kafka
- **[kubernetes.md](kubernetes.md)**: Kubernetes deployment for various architectures
- **[tdd.md](tdd.md)**: Test-Driven Development across all architectures
