# Hexagonal Architecture Guidelines
This document provides mandatory architectural standards and development practices for hexagonal architecture (also known as Ports and Adapters) with emphasis on clean separation of concerns, testability, and maintainability. This guide is language-agnostic and focuses on architectural principles.

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
    // 1000+ lines...
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

## Related Guides

- **[cleanarch.md](cleanarch.md)**: Clean Architecture - Robert C. Martin's complementary architectural pattern
- **[microservices.md](microservices.md)**: Microservices Architecture - applying hexagonal architecture to distributed systems
- **[tdd.md](tdd.md)**: Test-Driven Development - essential practice for implementing hexagonal architecture
- **[rest.md](rest.md)**: REST API Design - designing APIs that work with hexagonal boundaries
