# Hexagonal Architecture Guidelines
Mandatory architectural standards and development practices for hexagonal architecture (also known as Ports and Adapters) with emphasis on clean separation of concerns, testability, and maintainability. This guide is language-agnostic and focuses on architectural principles. Any programming language, DDD principles, SOLID principles, Clean Architecture concepts.

---

**Agent Profile**: The Hexagonal Architect
**Role**: Senior Software Architect & Domain-Driven Design Specialist
**Objective**: Generate production-ready, well-structured, maintainable systems using hexagonal architecture with clear boundaries, dependency inversion, and testable components.
**Tools**: Any programming language, DDD principles, SOLID principles, Clean Architecture concepts.

---

## 1. Core Philosophies: HEXAGONAL

The agent must adhere to the **HEXAGONAL** standard for every architectural implementation:

- **H**exagonal Boundaries: Clear separation between domain, application, and infrastructure
- **E**xplicit Ports: Well-defined interfaces for all external interactions
- **X**changeable Adapters: Infrastructure components are replaceable without domain changes
- **A**pplication Services: Orchestration layer coordinating domain and ports
- **G**uarded Domain: Domain logic has zero external dependencies
- **O**utward Dependencies: Dependencies point inward (infrastructure depends on domain, never reverse)
- **N**o Leaky Abstractions: Domain concepts never leak into infrastructure terminology
- **A**gnostic Core: Domain is technology-agnostic, framework-agnostic, database-agnostic
- **L**ayered Testing: Each layer testable in isolation

**Additional Principles:**

- **Test-Driven Development (TDD)**: ALWAYS write tests BEFORE implementation (Red-Green-Refactor cycle mandatory)
- **Regression Shield**: EVERY bug discovered MUST receive a test BEFORE fixing to prevent regression
- **Dependency Inversion**: High-level modules depend on abstractions, not concrete implementations
- **Single Responsibility**: Each component has one reason to change
- **Interface Segregation**: Many specific interfaces over one general-purpose interface

**Verified Architecture**: Agent-generated architecture MUST be validated for proper layer separation, dependency direction, and testability before delivery.

---

## 2. Architectural Layers (MANDATORY)

### A. The Three Concentric Layers

**CRITICAL: All hexagonal systems MUST have these three layers with strict dependency rules.**

```
┌─────────────────────────────────────────────────────────────┐
│                     INFRASTRUCTURE                          │
│  (Adapters: Controllers, Repositories, External Services)   │
│                                                             │
│    ┌─────────────────────────────────────────────────┐     │
│    │                 APPLICATION                      │     │
│    │    (Use Cases, Application Services, Ports)      │     │
│    │                                                  │     │
│    │    ┌─────────────────────────────────────┐      │     │
│    │    │            DOMAIN                    │      │     │
│    │    │   (Entities, Value Objects,          │      │     │
│    │    │    Domain Services, Domain Events)   │      │     │
│    │    └─────────────────────────────────────┘      │     │
│    │                                                  │     │
│    └─────────────────────────────────────────────────┘     │
│                                                             │
└─────────────────────────────────────────────────────────────┘

Dependencies flow INWARD only: Infrastructure → Application → Domain
```

### B. Layer Responsibilities

#### 1. Domain Layer (Innermost)

**Purpose**: Contains pure business logic, rules, and domain concepts.

**Contains**:
- **Entities**: Objects with identity that persist over time
- **Value Objects**: Immutable objects defined by their attributes
- **Domain Services**: Stateless operations that don't belong to entities
- **Domain Events**: Records of something significant that happened
- **Aggregates**: Clusters of entities treated as a unit
- **Repository Interfaces**: Abstractions for persistence (NOT implementations)

**Rules**:
- ZERO dependencies on external frameworks, libraries, or infrastructure
- ZERO imports from application or infrastructure layers
- Pure business logic only
- Technology-agnostic (no HTTP, SQL, JSON concepts)

#### 2. Application Layer (Middle)

**Purpose**: Orchestrates domain objects to perform use cases.

**Contains**:
- **Use Cases / Application Services**: Coordinate domain operations
- **Port Interfaces**: Define contracts for external interactions
- **DTOs**: Data Transfer Objects for input/output
- **Command/Query Objects**: Represent user intentions
- **Application Events**: Cross-cutting application concerns

**Rules**:
- Depends ONLY on domain layer
- Defines ports (interfaces) that infrastructure will implement
- No direct infrastructure dependencies
- Orchestrates, does not contain business logic

#### 3. Infrastructure Layer (Outermost)

**Purpose**: Implements technical concerns and external integrations.

**Contains**:
- **Driving Adapters**: Controllers, CLI handlers, message consumers, scheduled tasks
- **Driven Adapters**: Repository implementations, external API clients, message publishers
- **Configuration**: Framework setup, dependency injection, environment config
- **Persistence**: Database schemas, ORM mappings, migrations

**Rules**:
- Implements interfaces defined in application layer
- Contains ALL framework-specific code
- Contains ALL external service integrations
- Can depend on any layer (dependencies flow inward)

---

## 2A. Test-Driven Development (TDD) Protocol (MANDATORY)

**CRITICAL: Follow the Red-Green-Refactor cycle for ALL new code in every layer.**

### TDD Cycle for Hexagonal Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    TDD IN HEXAGONAL ARCHITECTURE                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   1. 🔴 RED: Write a failing test first                         │
│      │                                                          │
│      │  Start with DOMAIN layer tests (innermost)               │
│      │  • Entity behavior tests                                 │
│      │  • Value object validation tests                         │
│      │  • Domain service tests                                  │
│      ▼                                                          │
│   2. 🟢 GREEN: Write minimal code to make it pass               │
│      │                                                          │
│      │  Implement ONLY what the test requires                   │
│      │  • No extra features                                     │
│      │  • No premature optimization                             │
│      ▼                                                          │
│   3. 🔵 REFACTOR: Improve code while keeping tests green        │
│      │                                                          │
│      │  Clean up implementation                                 │
│      │  • Extract value objects                                 │
│      │  • Apply domain patterns                                 │
│      ▼                                                          │
│   4. 🔄 REPEAT: Move outward through layers                     │
│                                                                  │
│      Domain → Application → Infrastructure                       │
│      (Each layer tested in isolation)                           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Example TDD Workflow: Creating a Port and Adapter

**Scenario**: Implement a PaymentGateway driven port with a Stripe adapter.

```
STEP 1: 🔴 RED - Write Domain Test First
────────────────────────────────────────
// Domain Layer Test
test "Order processes payment successfully" {
    order = Order.create(customerId, items)
    paymentResult = PaymentResult.success(transactionId)

    order.markAsPaid(paymentResult)

    assert order.status == OrderStatus.PAID
    assert order.paymentTransactionId == transactionId
}
// RUN: FAILS - Order.markAsPaid() doesn't exist

STEP 2: 🟢 GREEN - Implement Domain Logic
─────────────────────────────────────────
// Domain Layer
class Order {
    markAsPaid(result: PaymentResult): void {
        this.ensureCanBePaid()
        this.status = OrderStatus.PAID
        this.paymentTransactionId = result.transactionId
        this.addEvent(new OrderPaid(this.id, result))
    }
}
// RUN: PASSES

STEP 3: 🔴 RED - Write Application Layer Test (with Port)
─────────────────────────────────────────────────────────
// Application Layer Test (mocking the driven port)
test "PlaceOrderUseCase processes payment via gateway" {
    mockPaymentGateway = mock(PaymentGateway)
    mockPaymentGateway.processPayment(any, any)
        .returns(PaymentResult.success("txn_123"))

    useCase = PlaceOrderUseCase(
        orderRepository: mockOrderRepo,
        paymentGateway: mockPaymentGateway
    )

    result = useCase.execute(placeOrderCommand)

    assert result.isSuccess
    verify mockPaymentGateway.processPayment(
        amount: Money(100, "USD"),
        method: any
    )
}
// RUN: FAILS - PaymentGateway port doesn't exist

STEP 4: 🟢 GREEN - Define Port Interface
────────────────────────────────────────
// Application Layer - Driven Port
interface PaymentGateway {
    processPayment(amount: Money, method: PaymentMethod): PaymentResult
    refund(paymentId: PaymentId, amount: Money): RefundResult
}
// RUN: PASSES (with mock implementation)

STEP 5: 🔴 RED - Write Integration Test for Adapter
───────────────────────────────────────────────────
// Infrastructure Layer Test (integration)
test "StripePaymentGateway processes payment" {
    gateway = StripePaymentGateway(testApiKey)

    result = gateway.processPayment(
        Money(1000, "USD"),
        PaymentMethod.card(testCard)
    )

    assert result.isSuccess
    assert result.transactionId.startsWith("txn_")
}
// RUN: FAILS - StripePaymentGateway doesn't exist

STEP 6: 🟢 GREEN - Implement Adapter
────────────────────────────────────
// Infrastructure Layer - Driven Adapter
class StripePaymentGateway implements PaymentGateway {
    processPayment(amount: Money, method: PaymentMethod): PaymentResult {
        stripeCharge = this.stripeClient.charges.create({
            amount: amount.cents,
            currency: amount.currency.code,
            source: this.toStripeSource(method)
        })
        return PaymentResult.success(stripeCharge.id)
    }
}
// RUN: PASSES
```

### TDD Rules by Layer

```
DOMAIN LAYER TDD:
├── NO mocking required (pure domain logic)
├── Test entities, value objects, domain services
├── Focus on business rule validation
└── Fast, isolated tests

APPLICATION LAYER TDD:
├── MOCK all driven ports (repositories, gateways)
├── Test use case orchestration
├── Verify port interactions
└── Test error handling and edge cases

INFRASTRUCTURE LAYER TDD:
├── INTEGRATION tests with real external systems
├── Use test databases, sandbox APIs
├── Test adapters implement ports correctly
└── Test technical error handling (retries, timeouts)
```

---

## 2B. Bug Fix Protocol (MANDATORY)

**CRITICAL: Every bug MUST receive a regression test BEFORE fixing.**

### Bug Fix Workflow for Hexagonal Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│               BUG FIX WORKFLOW IN HEXAGONAL ARCHITECTURE         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   1. 🐛 BUG REPORTED/DISCOVERED                                 │
│      │                                                          │
│      │  Identify which layer contains the bug:                  │
│      │  • Domain bug? (business logic error)                    │
│      │  • Application bug? (orchestration error)                │
│      │  • Infrastructure bug? (adapter/integration error)       │
│      ▼                                                          │
│   2. 📝 WRITE FAILING TEST (in appropriate layer)               │
│      │                                                          │
│      │  Test MUST reproduce the exact bug behavior              │
│      │  • Domain bug → Unit test in domain                      │
│      │  • Port contract bug → Application test with mock        │
│      │  • Adapter bug → Integration test                        │
│      ▼                                                          │
│   3. ✅ VERIFY TEST FAILS for the right reason                  │
│      │                                                          │
│      │  Confirm the test captures the bug accurately            │
│      ▼                                                          │
│   4. 🔧 FIX THE BUG (in the correct layer)                      │
│      │                                                          │
│      │  Fix should be minimal and focused                       │
│      │  • Respect layer boundaries                              │
│      │  • Don't leak abstractions                               │
│      ▼                                                          │
│   5. ✅ VERIFY TEST PASSES                                       │
│      │                                                          │
│      │  Run all tests to ensure no regressions                  │
│      ▼                                                          │
│   6. 📄 DOCUMENT in test comments (include bug ID)              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Example Bug Fix: Domain Layer Bug

```
BUG REPORT #BUG-142:
"Orders with zero items can be completed, violating business rules"

LAYER IDENTIFICATION:
This is a DOMAIN BUG - business invariant not enforced

STEP 1: Write failing test in Domain layer
──────────────────────────────────────────
// tests/unit/domain/OrderTest
test "BUG-142: Cannot complete order with zero items" {
    order = Order.create(customerId, items: [])

    // This should throw but currently doesn't
    assertThrows(EmptyOrderError) {
        order.complete()
    }
}
// RUN: FAILS - order.complete() succeeds with zero items

STEP 2: Fix the bug in Domain layer
───────────────────────────────────
// domain/model/entities/Order
class Order {
    complete(): void {
        // BUG-142 FIX: Add invariant check
        if (this.items.isEmpty()) {
            throw new EmptyOrderError(
                "Cannot complete order with no items"
            )
        }
        this.ensureCanComplete()
        this.status = OrderStatus.COMPLETED
        this.addEvent(new OrderCompleted(this.id))
    }
}
// RUN: PASSES - bug fixed, regression prevented
```

### Example Bug Fix: Port/Adapter Integration Bug

```
BUG REPORT #BUG-287:
"Payment gateway timeout not handled, causes 500 errors"

LAYER IDENTIFICATION:
This is an INFRASTRUCTURE BUG - adapter error handling

STEP 1: Write failing integration test
──────────────────────────────────────
// tests/integration/adapters/StripePaymentGatewayTest
test "BUG-287: Handles gateway timeout gracefully" {
    gateway = StripePaymentGateway(
        apiKey: testKey,
        timeout: 1ms  // Force timeout
    )

    result = gateway.processPayment(
        Money(1000, "USD"),
        testPaymentMethod
    )

    // Should return failure, not throw exception
    assert result.isFailure
    assert result.error instanceof PaymentTimeoutError
}
// RUN: FAILS - throws unhandled timeout exception

STEP 2: Fix in Infrastructure layer
───────────────────────────────────
// infrastructure/adapters/driven/StripePaymentGateway
class StripePaymentGateway implements PaymentGateway {
    processPayment(amount: Money, method: PaymentMethod): PaymentResult {
        try {
            stripeCharge = this.stripeClient.charges.create(...)
            return PaymentResult.success(stripeCharge.id)
        } catch (TimeoutException e) {
            // BUG-287 FIX: Handle timeout gracefully
            return PaymentResult.failure(
                new PaymentTimeoutError("Payment gateway timed out")
            )
        }
    }
}
// RUN: PASSES

STEP 3: Add Application layer test for proper handling
──────────────────────────────────────────────────────
// tests/unit/application/PlaceOrderUseCaseTest
test "BUG-287: Use case handles payment timeout" {
    mockGateway = mock(PaymentGateway)
    mockGateway.processPayment(any, any)
        .returns(PaymentResult.failure(PaymentTimeoutError()))

    useCase = PlaceOrderUseCase(gateway: mockGateway, ...)

    result = useCase.execute(command)

    assert result.isFailure
    assert result.error.message.contains("payment")
}
// Ensures application layer handles adapter errors correctly
```

### Bug Fix Checklist

```
BUG FIX VERIFICATION:
□ Bug isolated to correct layer (Domain/Application/Infrastructure)
□ Failing test written BEFORE fix
□ Test reproduces exact bug behavior
□ Fix implemented in appropriate layer
□ Fix respects layer boundaries (no dependency violations)
□ All existing tests still pass
□ Bug ID documented in test comments
□ No new technical debt introduced
```

---

## 3. Ports and Adapters (MANDATORY)

### A. Port Types

**CRITICAL: Distinguish between driving (primary) and driven (secondary) ports.**

#### Driving Ports (Primary)

**Definition**: How the outside world interacts with our application.

**Purpose**: Define the API that the application exposes to external actors.

**Examples**:
- Use case interfaces
- Application service interfaces
- Command handlers
- Query handlers

```
DRIVING PORTS (Primary)
External Actor → [Driving Adapter] → [Driving Port] → Application

Examples:
- HTTP Controller → REST API Port → Use Case
- CLI Handler → Command Port → Use Case
- Message Consumer → Event Port → Use Case
- Scheduler → Task Port → Use Case
```

#### Driven Ports (Secondary)

**Definition**: How our application interacts with external systems.

**Purpose**: Define abstractions for external dependencies the application needs.

**Examples**:
- Repository interfaces
- External service interfaces
- Notification interfaces
- File storage interfaces

```
DRIVEN PORTS (Secondary)
Application → [Driven Port] → [Driven Adapter] → External System

Examples:
- Use Case → Repository Port → Database Adapter
- Use Case → Email Port → SMTP Adapter
- Use Case → Payment Port → Stripe Adapter
- Use Case → Storage Port → S3 Adapter
```

### B. Adapter Implementation Rules

#### Driving Adapters (Inbound)

**CRITICAL: Driving adapters translate external requests into application calls.**

```
┌──────────────────────────────────────────────────────────┐
│ DRIVING ADAPTER RESPONSIBILITIES                         │
├──────────────────────────────────────────────────────────┤
│ 1. Receive external input (HTTP, CLI, messages, etc.)    │
│ 2. Validate input format (NOT business validation)       │
│ 3. Transform to application DTOs/Commands                │
│ 4. Call application service/use case                     │
│ 5. Transform result to external format                   │
│ 6. Handle technical errors (NOT business errors)         │
└──────────────────────────────────────────────────────────┘
```

#### Driven Adapters (Outbound)

**CRITICAL: Driven adapters implement port interfaces defined in application layer.**

```
┌──────────────────────────────────────────────────────────┐
│ DRIVEN ADAPTER RESPONSIBILITIES                          │
├──────────────────────────────────────────────────────────┤
│ 1. Implement port interface from application layer       │
│ 2. Translate domain objects to external format           │
│ 3. Interact with external system                         │
│ 4. Translate external responses to domain objects        │
│ 5. Handle technical failures (retries, timeouts)         │
│ 6. Map external errors to domain exceptions              │
└──────────────────────────────────────────────────────────┘
```

---

## 4. Directory Structure (MANDATORY)

### A. Standard Hexagonal Layout

**CRITICAL: Follow this directory structure for clear layer separation.**

```
project/
├── src/
│   ├── domain/                    # DOMAIN LAYER (innermost)
│   │   ├── model/                 # Domain models
│   │   │   ├── entities/          # Entities with identity
│   │   │   │   ├── User
│   │   │   │   ├── Order
│   │   │   │   └── Product
│   │   │   ├── value-objects/     # Immutable value types
│   │   │   │   ├── Email
│   │   │   │   ├── Money
│   │   │   │   └── Address
│   │   │   └── aggregates/        # Aggregate roots
│   │   │       └── OrderAggregate
│   │   ├── services/              # Domain services
│   │   │   ├── PricingService
│   │   │   └── InventoryService
│   │   ├── events/                # Domain events
│   │   │   ├── OrderPlaced
│   │   │   └── PaymentReceived
│   │   ├── exceptions/            # Domain exceptions
│   │   │   ├── InsufficientFunds
│   │   │   └── InvalidOrder
│   │   └── repositories/          # Repository INTERFACES only
│   │       ├── UserRepository
│   │       └── OrderRepository
│   │
│   ├── application/               # APPLICATION LAYER (middle)
│   │   ├── ports/                 # Port definitions
│   │   │   ├── driving/           # Inbound ports (use case interfaces)
│   │   │   │   ├── CreateOrderUseCase
│   │   │   │   └── GetOrderQuery
│   │   │   └── driven/            # Outbound ports (external dependencies)
│   │   │       ├── PaymentGateway
│   │   │       ├── NotificationService
│   │   │       └── EmailSender
│   │   ├── services/              # Application services (use case implementations)
│   │   │   ├── OrderService
│   │   │   └── UserService
│   │   ├── dto/                   # Data Transfer Objects
│   │   │   ├── CreateOrderRequest
│   │   │   ├── OrderResponse
│   │   │   └── UserDTO
│   │   ├── commands/              # Command objects
│   │   │   ├── CreateOrderCommand
│   │   │   └── UpdateUserCommand
│   │   ├── queries/               # Query objects
│   │   │   ├── GetOrderByIdQuery
│   │   │   └── ListOrdersQuery
│   │   └── exceptions/            # Application exceptions
│   │       ├── OrderNotFound
│   │       └── ValidationError
│   │
│   └── infrastructure/            # INFRASTRUCTURE LAYER (outermost)
│       ├── adapters/
│       │   ├── driving/           # Inbound adapters
│       │   │   ├── rest/          # REST API controllers
│       │   │   │   ├── OrderController
│       │   │   │   └── UserController
│       │   │   ├── graphql/       # GraphQL resolvers
│       │   │   ├── cli/           # CLI command handlers
│       │   │   ├── grpc/          # gRPC service implementations
│       │   │   └── messaging/     # Message consumers (Kafka, RabbitMQ)
│       │   │       └── OrderEventConsumer
│       │   └── driven/            # Outbound adapters
│       │       ├── persistence/   # Database implementations
│       │       │   ├── UserRepositoryImpl
│       │       │   ├── OrderRepositoryImpl
│       │       │   └── orm/       # ORM configurations
│       │       ├── external/      # External API clients
│       │       │   ├── StripePaymentGateway
│       │       │   └── TwilioNotificationService
│       │       ├── messaging/     # Message publishers
│       │       │   └── KafkaEventPublisher
│       │       └── storage/       # File storage
│       │           └── S3StorageAdapter
│       ├── config/                # Configuration
│       │   ├── DependencyInjection
│       │   ├── DatabaseConfig
│       │   └── ExternalServicesConfig
│       └── migrations/            # Database migrations
│
├── tests/
│   ├── unit/
│   │   ├── domain/                # Domain unit tests
│   │   └── application/           # Application unit tests
│   ├── integration/
│   │   └── adapters/              # Adapter integration tests
│   └── e2e/                       # End-to-end tests
│
└── docs/
    ├── architecture/
    │   ├── decisions/             # Architecture Decision Records (ADRs)
    │   └── diagrams/
    └── api/
```

### B. Alternative: Feature-Based Structure

For larger applications, organize by feature/module while maintaining hexagonal layers:

```
project/
├── src/
│   ├── modules/
│   │   ├── orders/                # Order module (bounded context)
│   │   │   ├── domain/
│   │   │   ├── application/
│   │   │   └── infrastructure/
│   │   ├── users/                 # User module (bounded context)
│   │   │   ├── domain/
│   │   │   ├── application/
│   │   │   └── infrastructure/
│   │   └── payments/              # Payment module (bounded context)
│   │       ├── domain/
│   │       ├── application/
│   │       └── infrastructure/
│   └── shared/                    # Shared kernel
│       ├── domain/
│       └── infrastructure/
```

---

## 5. Dependency Rules (MANDATORY)

### A. The Dependency Rule

**CRITICAL: Dependencies MUST point inward. This is the fundamental rule of hexagonal architecture.**

```
ALLOWED:
  Infrastructure → Application → Domain
  Infrastructure → Domain
  Application → Domain

FORBIDDEN:
  Domain → Application
  Domain → Infrastructure
  Application → Infrastructure
```

### B. Import Rules by Layer

#### Domain Layer Imports

```
✅ ALLOWED in Domain:
- Other domain classes (entities, value objects, domain services)
- Standard library types (strings, numbers, collections, dates)
- Domain-specific exceptions

❌ FORBIDDEN in Domain:
- Application layer imports
- Infrastructure layer imports
- Framework imports (Spring, Django, Express, etc.)
- ORM imports (Hibernate, SQLAlchemy, Prisma, etc.)
- HTTP/REST imports
- Database driver imports
- External library imports (except pure utilities)
```

#### Application Layer Imports

```
✅ ALLOWED in Application:
- Domain layer (entities, value objects, repository interfaces)
- Other application classes (DTOs, commands, queries)
- Port interfaces (both driving and driven)

❌ FORBIDDEN in Application:
- Infrastructure layer imports
- Framework-specific imports
- Concrete adapter implementations
- Database-specific imports
- HTTP-specific imports
```

#### Infrastructure Layer Imports

```
✅ ALLOWED in Infrastructure:
- Domain layer
- Application layer
- Framework imports
- Database/ORM imports
- External library imports
- Any technical dependency
```

### C. Dependency Verification

**CRITICAL: Always verify dependency direction before delivery.**

```
Verification Checklist:
□ Domain has ZERO imports from application or infrastructure
□ Application has ZERO imports from infrastructure
□ All port interfaces are defined in application layer
□ All port implementations are in infrastructure layer
□ Repository interfaces are in domain, implementations in infrastructure
□ No framework annotations/decorators in domain layer
□ No database types (IDs, timestamps) leak into domain
```

---

## 6. Domain Layer Design (MANDATORY)

### A. Entity Design

**CRITICAL: Entities have identity and lifecycle. Design them to protect invariants.**

#### Entity Rules

```
ENTITY REQUIREMENTS:
1. Unique identity (ID) that persists over time
2. Encapsulated state (no public setters)
3. Business methods that enforce invariants
4. Factory methods for complex creation
5. Equality based on identity, not attributes
```

#### Entity Pattern

```
✅ CORRECT - Well-Designed Entity

Entity: Order
├── Identity: OrderId (value object)
├── State: (private/protected)
│   ├── customerId: CustomerId
│   ├── items: List<OrderItem>
│   ├── status: OrderStatus
│   ├── totalAmount: Money
│   └── createdAt: DateTime
├── Invariants:
│   ├── Order must have at least one item
│   ├── Total must equal sum of item prices
│   └── Cannot modify completed orders
├── Behavior:
│   ├── addItem(product, quantity) → validates & updates total
│   ├── removeItem(itemId) → validates & updates total
│   ├── complete() → validates & changes status
│   └── cancel(reason) → validates & changes status
└── Factory:
    └── create(customerId, items) → validates & returns Order

❌ WRONG - Anemic Entity

Entity: Order
├── Public fields with getters/setters
├── No validation in setters
├── All logic in external services
└── Direct field manipulation allowed
```

### B. Value Object Design

**CRITICAL: Value objects are immutable and defined by their attributes.**

#### Value Object Rules

```
VALUE OBJECT REQUIREMENTS:
1. Immutable (no state changes after creation)
2. Equality based on all attributes
3. Self-validating (invalid state impossible)
4. No identity (no ID field)
5. Replaceable (swap entire object, don't modify)
```

#### Value Object Examples

```
✅ CORRECT - Well-Designed Value Objects

ValueObject: Email
├── Validation: Valid email format on creation
├── Immutable: Cannot change after creation
├── Behavior: domain(), localPart(), toString()
└── Equality: Two emails equal if string matches

ValueObject: Money
├── Properties: amount (decimal), currency (Currency)
├── Validation: Amount >= 0, valid currency
├── Immutable: Cannot change amount or currency
├── Behavior: add(Money), subtract(Money), multiply(factor)
└── Equality: Same amount AND same currency

ValueObject: Address
├── Properties: street, city, postalCode, country
├── Validation: All required fields present
├── Immutable: Create new Address for changes
└── Equality: All fields match

❌ WRONG - Mutable "Value Object"

ValueObject: Money
├── Mutable amount field
├── setAmount() method
└── No validation
```

### C. Domain Service Design

**CRITICAL: Domain services contain logic that doesn't belong to a single entity.**

```
DOMAIN SERVICE CRITERIA:
Use a domain service when the operation:
1. Involves multiple entities/aggregates
2. Requires external information (via port)
3. Doesn't naturally fit in any entity
4. Is a significant domain concept

DOMAIN SERVICE RULES:
1. Stateless (no instance state)
2. Named after domain concept (PricingService, not PriceCalculator)
3. Operates on domain objects only
4. May use repository interfaces (defined in domain)
```

---

## 7. Application Layer Design (MANDATORY)

### A. Use Case / Application Service Design

**CRITICAL: Application services orchestrate domain objects to fulfill use cases.**

#### Application Service Rules

```
APPLICATION SERVICE REQUIREMENTS:
1. One public method per use case (Single Responsibility)
2. Orchestrates domain objects and ports
3. Handles transactions (or delegates to infrastructure)
4. Transforms DTOs to/from domain objects
5. Contains NO business logic (delegate to domain)
6. Thin layer - mostly coordination
```

#### Use Case Pattern

```
✅ CORRECT - Well-Designed Use Case

UseCase: PlaceOrder
├── Input: PlaceOrderCommand (DTO)
│   ├── customerId: string
│   ├── items: List<OrderItemDTO>
│   └── shippingAddress: AddressDTO
├── Output: OrderResult (DTO)
│   ├── orderId: string
│   ├── status: string
│   └── estimatedDelivery: date
├── Dependencies (Ports):
│   ├── OrderRepository (driven port)
│   ├── CustomerRepository (driven port)
│   ├── PaymentGateway (driven port)
│   └── NotificationService (driven port)
├── Flow:
│   1. Validate command (format validation)
│   2. Load customer from repository
│   3. Create Order entity (domain validates business rules)
│   4. Process payment via gateway
│   5. Save order to repository
│   6. Send notification
│   7. Return OrderResult
└── Error Handling:
    ├── CustomerNotFound → Application exception
    ├── InsufficientFunds → Domain exception (from Order)
    └── PaymentFailed → Map to application exception

❌ WRONG - Fat Application Service

UseCase: PlaceOrder
├── Contains business validation logic
├── Calculates prices (should be in domain)
├── Directly uses database connection
├── Has multiple public methods
└── 500+ lines of code
```

### B. Port Interface Design

**CRITICAL: Ports define contracts that adapters must fulfill.**

#### Driven Port Design

```
✅ CORRECT - Well-Designed Driven Port

Port: PaymentGateway
├── Purpose: Process payments with external provider
├── Methods:
│   ├── processPayment(amount: Money, method: PaymentMethod) → PaymentResult
│   ├── refund(paymentId: PaymentId, amount: Money) → RefundResult
│   └── getPaymentStatus(paymentId: PaymentId) → PaymentStatus
├── Uses Domain Types: Money, PaymentMethod, PaymentResult
├── No Infrastructure Types: No "StripeCharge", no HTTP concepts
└── Error Handling: Returns Result type or throws domain exceptions

Port: OrderRepository
├── Purpose: Persist and retrieve orders
├── Methods:
│   ├── save(order: Order) → void
│   ├── findById(id: OrderId) → Order?
│   ├── findByCustomer(customerId: CustomerId) → List<Order>
│   └── delete(id: OrderId) → void
├── Uses Domain Types: Order, OrderId, CustomerId
└── No Database Types: No SQL, no "findByQuery", no pagination details

❌ WRONG - Leaky Port Interface

Port: OrderRepository
├── findBySqlQuery(sql: string) → List<Order>  // Leaks SQL
├── save(order: Order, connection: DbConnection) // Leaks DB connection
└── findWithPagination(page: Page<Order>) // Leaks framework type
```

---

## 8. Infrastructure Layer Design (MANDATORY)

### A. Driving Adapter Design

**CRITICAL: Driving adapters translate external protocols to application calls.**

```
DRIVING ADAPTER PATTERN:

External Request                    Application
      │                                 │
      ▼                                 │
┌─────────────────┐                     │
│ Driving Adapter │                     │
├─────────────────┤                     │
│ 1. Receive      │ HTTP/CLI/Message    │
│ 2. Deserialize  │ JSON/Args/Event     │
│ 3. Validate     │ Format only         │
│ 4. Transform    │ To Command/Query ───┼──▶ Application Service
│ 5. Call         │ Use Case            │
│ 6. Transform    │ From Result ◀───────┼─── Result/DTO
│ 7. Serialize    │ To HTTP/Response    │
│ 8. Respond      │                     │
└─────────────────┘                     │
```

#### Driving Adapter Rules

```
DRIVING ADAPTER REQUIREMENTS:
1. Handle protocol-specific concerns (HTTP status, headers, etc.)
2. Validate input FORMAT (not business rules)
3. Transform external types to application DTOs
4. Call application service (one adapter may call multiple use cases)
5. Transform results to external format
6. Handle technical errors (500s, timeouts)
7. Map application exceptions to protocol responses (404, 422, etc.)

DRIVING ADAPTER FORBIDDEN:
1. Business logic
2. Direct domain object manipulation
3. Direct database access
4. Calling other adapters directly
```

### B. Driven Adapter Design

**CRITICAL: Driven adapters implement port interfaces using specific technologies.**

```
DRIVEN ADAPTER PATTERN:

Application                     External System
     │                                │
     │                                │
     ▼                                │
┌─────────────────┐                   │
│  Driven Adapter │                   │
├─────────────────┤                   │
│ 1. Receive      │ Domain objects    │
│ 2. Transform    │ To external format┼──▶ Database/API/Queue
│ 3. Execute      │ External call     │
│ 4. Handle       │ Errors/Retries    │
│ 5. Transform    │ From external ◀───┼─── External response
│ 6. Return       │ Domain objects    │
└─────────────────┘                   │
     │                                │
     ▼                                │
Port Interface                        │
(Defined in Application)              │
```

#### Driven Adapter Rules

```
DRIVEN ADAPTER REQUIREMENTS:
1. Implement port interface from application layer
2. Contain ALL technology-specific code
3. Transform domain objects to external format
4. Transform external responses to domain objects
5. Handle technical concerns (retries, timeouts, connection pooling)
6. Map external errors to domain exceptions

DRIVEN ADAPTER FORBIDDEN:
1. Business logic
2. Exposing external types to application layer
3. Leaking infrastructure concepts upward
```

---

## 9. Testing Strategy (MANDATORY)

### A. Test Pyramid for Hexagonal Architecture

**CRITICAL: Each layer has its own testing strategy.**

```
                    ┌─────────┐
                   /   E2E    \           Few, slow, expensive
                  /   Tests    \          Test entire system
                 /──────────────\
                /   Integration  \        Some, medium speed
               /     Tests        \       Test adapters + external
              /────────────────────\
             /      Unit Tests      \     Many, fast, cheap
            /  Domain + Application  \    Test business logic
           /──────────────────────────\
```

### B. Testing by Layer

#### Domain Layer Testing

```
DOMAIN TESTS:
├── Type: Unit tests
├── Scope: Entities, value objects, domain services
├── Dependencies: None (pure domain only)
├── Mocking: None needed (no external dependencies)
└── Coverage: 90%+ required

Test Categories:
1. Entity invariant tests
2. Value object validation tests
3. Domain service logic tests
4. Domain event generation tests
5. Aggregate consistency tests
```

#### Application Layer Testing

```
APPLICATION TESTS:
├── Type: Unit tests with mocks
├── Scope: Use cases, application services
├── Dependencies: Mocked ports
├── Mocking: All driven ports (repositories, gateways)
└── Coverage: 80%+ required

Test Categories:
1. Use case happy path tests
2. Use case error handling tests
3. Port interaction verification
4. Transaction boundary tests
5. DTO transformation tests
```

#### Infrastructure Layer Testing

```
INFRASTRUCTURE TESTS:
├── Type: Integration tests
├── Scope: Adapters (driving and driven)
├── Dependencies: Real external systems (test instances)
├── Mocking: Minimal (test actual integration)
└── Coverage: 70%+ required

Test Categories:
1. Repository tests (with test database)
2. External API client tests (with test server/sandbox)
3. Controller tests (with test HTTP client)
4. Message handler tests (with test broker)
5. Configuration tests
```

### C. TDD Protocol for Hexagonal Architecture

**CRITICAL: Follow Red-Green-Refactor for all layers.**

```
TDD WORKFLOW:

1. 🔴 RED: Write failing test
   ├── Start with domain layer
   ├── Define expected behavior
   └── Test should fail (code doesn't exist)

2. 🟢 GREEN: Write minimal implementation
   ├── Implement just enough to pass
   ├── No optimization
   └── Test must pass

3. 🔵 REFACTOR: Improve code quality
   ├── Clean up implementation
   ├── Maintain passing tests
   └── Apply design patterns

4. 🔄 REPEAT: Move to next layer
   ├── Domain → Application → Infrastructure
   └── Each layer tested in isolation
```

---

## 10. Common Anti-Patterns (PROHIBITED)

### A. Dependency Violations

```
❌ PROHIBITED: Domain Depends on Infrastructure

// Domain layer
class Order {
    save() {
        database.insert(this);  // WRONG: Domain knows about database
    }
}

✅ CORRECT: Infrastructure Depends on Domain

// Domain layer
interface OrderRepository {
    save(order: Order): void;
}

// Infrastructure layer
class SqlOrderRepository implements OrderRepository {
    save(order: Order): void {
        database.insert(this.toRecord(order));
    }
}
```

### B. Anemic Domain Model

```
❌ PROHIBITED: Logic Outside Domain

// Domain layer - Anemic
class Order {
    id: OrderId;
    items: List<OrderItem>;
    status: string;
    // Only getters and setters, no behavior
}

// Application layer - Fat service
class OrderService {
    complete(order: Order) {
        if (order.items.isEmpty()) throw Error;  // WRONG: Business logic here
        if (order.status !== 'pending') throw Error;
        order.status = 'completed';  // WRONG: Direct mutation
        order.completedAt = now();
    }
}

✅ CORRECT: Rich Domain Model

// Domain layer - Rich
class Order {
    private items: List<OrderItem>;
    private status: OrderStatus;

    complete(): void {
        this.ensureCanComplete();  // Validates invariants
        this.status = OrderStatus.COMPLETED;
        this.addEvent(new OrderCompleted(this.id));
    }

    private ensureCanComplete(): void {
        if (this.items.isEmpty()) {
            throw new EmptyOrderError();
        }
        if (this.status !== OrderStatus.PENDING) {
            throw new InvalidOrderStateError();
        }
    }
}

// Application layer - Thin orchestration
class OrderService {
    complete(orderId: OrderId): void {
        order = this.orderRepository.findById(orderId);
        order.complete();  // Domain handles business logic
        this.orderRepository.save(order);
    }
}
```

### C. Leaky Abstractions

```
❌ PROHIBITED: Infrastructure Leaks to Domain

// Domain layer
class User {
    @Column("user_name")     // WRONG: ORM annotation in domain
    name: string;

    @JsonProperty("email")    // WRONG: Serialization in domain
    email: Email;
}

// Application layer port
interface UserRepository {
    findByQuery(query: SqlQuery): User[];  // WRONG: SQL in port
}

✅ CORRECT: Clean Abstractions

// Domain layer - Pure
class User {
    name: string;
    email: Email;
}

// Application layer port - Domain language
interface UserRepository {
    findByEmail(email: Email): User?;
    findActive(): User[];
}

// Infrastructure layer - Contains all technical details
@Entity("users")
class UserRecord {
    @Column("user_name")
    name: string;

    toUser(): User { ... }
    static fromUser(user: User): UserRecord { ... }
}
```

### D. God Services

```
❌ PROHIBITED: Monolithic Application Service

class OrderService {
    createOrder() { ... }
    updateOrder() { ... }
    deleteOrder() { ... }
    getOrder() { ... }
    listOrders() { ... }
    calculateTax() { ... }      // Should be domain
    validateInventory() { ... } // Should be domain
    sendNotification() { ... }  // Should be separate port
    generateReport() { ... }    // Should be separate use case
    // 1000+ lines..
}

✅ CORRECT: Focused Use Cases

class CreateOrderUseCase {
    execute(command: CreateOrderCommand): OrderResult { ... }
}

class GetOrderQuery {
    execute(query: GetOrderByIdQuery): OrderDTO { ... }
}

class CancelOrderUseCase {
    execute(command: CancelOrderCommand): void { ... }
}
```

---

## 11. Verification Checklist (MANDATORY)

### A. Architecture Verification Protocol

**CRITICAL: Verify architecture before delivery.**

```
VERIFICATION CHECKLIST:

□ Layer Separation
  □ Domain has no imports from application or infrastructure
  □ Application has no imports from infrastructure
  □ Each layer is in its own directory/package/module

□ Dependency Direction
  □ All dependencies point inward
  □ Domain depends on nothing external
  □ Infrastructure implements interfaces from application

□ Port Design
  □ All driven ports defined in application layer
  □ Ports use domain types (not infrastructure types)
  □ No leaky abstractions in port interfaces

□ Domain Design
  □ Entities have behavior (not just data)
  □ Value objects are immutable
  □ Domain services are stateless
  □ No framework annotations in domain

□ Application Design
  □ Use cases are single-purpose
  □ No business logic in application layer
  □ DTOs are used for input/output

□ Infrastructure Design
  □ Adapters implement port interfaces
  □ All framework code is in infrastructure
  □ No business logic in adapters

□ Testing
  □ Domain tests have no mocks
  □ Application tests mock all ports
  □ Integration tests verify adapters
  □ 80%+ coverage achieved
```

### B. Code Review Checklist

```
REVIEW QUESTIONS:

1. Can I test the domain without any infrastructure?
2. Can I swap the database without changing domain/application?
3. Can I add a CLI adapter without changing existing code?
4. Is there any business logic outside the domain?
5. Are port interfaces using domain language?
6. Do adapters contain only technical concerns?
7. Are dependencies flowing in the correct direction?
```

---

## 12. Migration Strategy

### A. Migrating to Hexagonal Architecture

**For existing applications, migrate incrementally:**

```
MIGRATION PHASES:

Phase 1: Identify Boundaries
├── Map existing components to layers
├── Identify domain concepts
└── Document current dependencies

Phase 2: Extract Domain
├── Create domain layer directory
├── Move/create entities and value objects
├── Remove infrastructure dependencies from domain
└── Add domain tests

Phase 3: Define Ports
├── Create application layer
├── Define port interfaces
├── Create application services
└── Add application tests

Phase 4: Implement Adapters
├── Create infrastructure layer
├── Implement driven adapters
├── Refactor driving adapters
└── Add integration tests

Phase 5: Clean Up
├── Remove circular dependencies
├── Verify dependency direction
├── Complete test coverage
└── Document architecture
```

---

## 13. Summary

### Core Principles

1. **Domain at the center**: Business logic is protected and technology-agnostic
2. **Dependencies point inward**: Infrastructure depends on domain, never reverse
3. **Ports define contracts**: Application layer defines interfaces for external interactions
4. **Adapters are replaceable**: Infrastructure components can be swapped without affecting business logic
5. **Test in isolation**: Each layer is testable independently

### Key Benefits

- **Testability**: Domain and application logic testable without infrastructure
- **Flexibility**: Easy to swap databases, frameworks, external services
- **Maintainability**: Clear boundaries and responsibilities
- **Understandability**: Architecture reflects business domains
- **Evolvability**: Easy to add new adapters and features

### Remember

> "The center of your application is not the database. Nor is it one or more of the frameworks you may be using. The center of your application is the use cases of your application."
> — Robert C. Martin (Uncle Bob)

---

## Quick Reference

### Hexagonal Architecture at a Glance

```
┌─────────────────────────────────────────────────────────────────┐
│                     QUICK REFERENCE                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  LAYER HIERARCHY (Dependencies flow INWARD only):               │
│                                                                  │
│    Infrastructure → Application → Domain                         │
│         │                │            │                          │
│    Adapters         Ports/Use     Entities                       │
│    Config           Cases         Value Objects                  │
│    DB/HTTP          DTOs          Domain Services                │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  PORT TYPES:                                                     │
│                                                                  │
│  DRIVING (Primary/Inbound):    DRIVEN (Secondary/Outbound):     │
│  • Use case interfaces          • Repository interfaces          │
│  • Command handlers             • External service interfaces    │
│  • Query handlers               • Notification interfaces        │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ADAPTER TYPES:                                                  │
│                                                                  │
│  DRIVING (Inbound):             DRIVEN (Outbound):              │
│  • REST Controllers             • Database Repositories          │
│  • CLI Handlers                 • External API Clients           │
│  • Message Consumers            • Message Publishers             │
│  • GraphQL Resolvers            • File Storage Adapters          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Common Patterns

```
PATTERN                         USE CASE
─────────────────────────────────────────────────────────────────
Repository Interface            Abstract persistence in domain
  (Domain) + Impl (Infra)       Swap databases without domain changes

Port Interface                  Define contracts for external systems
  (Application)                 Decouple from specific implementations

Driving Adapter                 Translate external protocols to app calls
  (Infrastructure)              Support multiple entry points (REST, CLI)

Driven Adapter                  Implement port with specific technology
  (Infrastructure)              Hide external system complexity

Application Service             Orchestrate domain operations
  (Application)                 Single use case per service

Domain Service                  Cross-entity business logic
  (Domain)                      Stateless, operates on domain objects

Value Object                    Immutable, self-validating types
  (Domain)                      Replace primitives with domain concepts

Aggregate Root                  Consistency boundary
  (Domain)                      Single entry point for related entities
```

### Directory Structure Quick Reference

```
project/
├── src/
│   ├── domain/                 # INNERMOST - Zero external dependencies
│   │   ├── model/              # Entities, Value Objects, Aggregates
│   │   ├── services/           # Domain Services (stateless)
│   │   ├── events/             # Domain Events
│   │   └── repositories/       # Repository INTERFACES only
│   │
│   ├── application/            # MIDDLE - Depends only on Domain
│   │   ├── ports/
│   │   │   ├── driving/        # Inbound port interfaces
│   │   │   └── driven/         # Outbound port interfaces
│   │   ├── services/           # Use case implementations
│   │   └── dto/                # Data Transfer Objects
│   │
│   └── infrastructure/         # OUTERMOST - Implements all ports
│       ├── adapters/
│       │   ├── driving/        # REST, CLI, GraphQL, Messaging
│       │   └── driven/         # Repositories, External APIs
│       └── config/             # DI, Database, Framework config
│
└── tests/
    ├── unit/domain/            # No mocks needed
    ├── unit/application/       # Mock all ports
    └── integration/adapters/   # Real external systems (test instances)
```

### TDD Quick Reference

```
LAYER           TEST TYPE       MOCKING                 FOCUS
─────────────────────────────────────────────────────────────────
Domain          Unit            None                    Business rules
Application     Unit            All driven ports        Orchestration
Infrastructure  Integration     Minimal                 External systems
```

### Common Anti-Patterns to Avoid

```
❌ ANTI-PATTERN                  ✅ CORRECT APPROACH
─────────────────────────────────────────────────────────────────
Domain imports infrastructure   Domain has zero external imports
Application calls DB directly   Application uses repository port
Framework annotations in domain Keep domain pure, annotations in infra
Business logic in adapters      Adapters only translate/delegate
Fat application services        One public method per use case
Anemic domain model             Entities have behavior, not just data
Leaky port interfaces           Ports use domain types only
Skipping TDD for "simple" code  TDD for ALL code, especially domain
```

### Verification Commands

```bash
# Typical verification workflow (language-agnostic)

# 1. Run domain unit tests (should be fast, no external deps)
[test-runner] tests/unit/domain/

# 2. Run application unit tests (with mocked ports)
[test-runner] tests/unit/application/

# 3. Run integration tests (may need test containers/databases)
[test-runner] tests/integration/

# 4. Verify dependency direction (use architecture linting tools)
# Example tools: ArchUnit (Java), dependency-cruiser (JS),
# import-linter (Python), deptrac (PHP)

# 5. Check test coverage
[coverage-tool] --threshold=80
```

### Key Questions for Architecture Review

```
VALIDATION CHECKLIST:
□ Can domain be tested without ANY infrastructure?
□ Can I swap the database by only changing one adapter?
□ Can I add a CLI adapter without modifying existing code?
□ Are all port interfaces defined in the application layer?
□ Do adapters implement ports, not call each other?
□ Is there ANY business logic outside the domain layer?
□ Do dependencies flow inward (never outward)?
□ Are value objects used instead of primitives?
□ Does every bug have a regression test?
```

---

## 14. Why This Configuration Works

- **Technology swaps become trivial**: Because all external dependencies are hidden behind port interfaces, replacing a database (PostgreSQL to MongoDB), messaging system (RabbitMQ to Kafka), or payment provider (Stripe to Adyen) requires only writing a new adapter without touching any business logic.
- **Domain logic is tested in isolation**: The domain layer has zero external dependencies, so its tests are pure unit tests that run in milliseconds with no mocks, containers, or network calls. This makes the most critical code in the system also the most thoroughly and cheaply testable.
- **Multiple entry points coexist naturally**: Adding a CLI, GraphQL API, or event consumer alongside an existing REST API requires only a new driving adapter. Each adapter translates its protocol to the same application service calls, eliminating code duplication across interfaces.
- **Port interfaces enforce clean contracts**: Defining driven ports in the application layer using domain language (not SQL queries or HTTP concepts) creates contracts that are naturally resistant to leaky abstractions, keeping infrastructure details from polluting business logic.
- **Incremental migration is practical**: The layered approach allows existing applications to adopt hexagonal architecture one module at a time, extracting domain logic inward and wrapping infrastructure outward, rather than requiring a risky full rewrite.

---

## 15. Implementation Checklist

### Port and Adapter Compliance
- [ ] **Ports defined in application layer**: All port interfaces live in the application layer, not infrastructure
- [ ] **Ports use domain types only**: Port method signatures reference domain objects, not SQL types, HTTP objects, or framework classes
- [ ] **Driving adapters translate protocols**: REST controllers, CLI handlers, and message consumers only translate and delegate to application services
- [ ] **Driven adapters implement ports**: Each external system interaction (database, API, messaging) implements a port interface
- [ ] **Adapters do not call each other**: No direct adapter-to-adapter dependencies; all communication flows through application services

### Dependency Direction
- [ ] **Domain has zero external imports**: No framework, ORM, HTTP, or library references in domain layer code
- [ ] **Application depends only on domain**: Application layer imports only from the domain layer
- [ ] **Infrastructure depends inward**: Infrastructure layer imports from application and domain, never the reverse
- [ ] **Dependency direction automated**: Architecture linting tool (ArchUnit, dependency-cruiser, import-linter, deptrac) enforces rules in CI
- [ ] **No circular dependencies**: Static analysis confirms zero circular imports across all layers

### Testing Verification
- [ ] **Domain tests have no mocks**: Domain unit tests run with plain objects, zero external dependencies
- [ ] **Application tests mock all driven ports**: Use case tests substitute port interfaces, exercising orchestration logic
- [ ] **Integration tests use real infrastructure**: Adapter tests run against real or containerized databases and services
- [ ] **TDD cycle followed**: All new code developed via Red-Green-Refactor
- [ ] **Bug fixes include regression tests**: Every resolved defect has a failing-then-passing test

### Code Quality
- [ ] **One use case per application service**: Each service class has a single public method for one operation
- [ ] **Value objects replace primitives**: Domain concepts (CustomerId, Money, Email) are typed, not raw strings or numbers
- [ ] **Aggregate roots enforce invariants**: Entity modifications go through aggregate root methods that validate business rules
- [ ] **Domain services are stateless**: Cross-entity logic in domain services operates only on passed-in domain objects
- [ ] **Directory structure reflects architecture**: Folder names (domain, application, infrastructure) make the architecture visible

### Documentation
- [ ] **Port catalog maintained**: All driving and driven ports listed with their purpose and implementing adapters
- [ ] **Adapter swappability verified**: At least one adapter has been swapped (e.g., in-memory to database) to prove the architecture works
- [ ] **Layer responsibilities documented**: Each layer's purpose, allowed dependencies, and testing strategy described
- [ ] **Architecture diagram current**: Hexagon diagram updated to reflect current ports, adapters, and domain components

---

## Related Guides

- **[cleanarch.md](cleanarch.md)**: Clean Architecture - Robert C. Martin's complementary architectural pattern
- **[microservices.md](microservices.md)**: Microservices Architecture - applying hexagonal architecture to distributed systems
- **[tdd.md](tdd.md)**: Test-Driven Development - essential practice for implementing hexagonal architecture
- **[rest.md](rest.md)**: REST API Design - designing APIs that work with hexagonal boundaries


**End of Hexagonal Architecture Guidelines**
