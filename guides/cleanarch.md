# Clean Architecture Guidelines

This document provides mandatory architectural standards and development practices for Clean Architecture as defined by Robert C. Martin (Uncle Bob). Emphasis on the Dependency Rule, use case-driven design, and framework independence. This guide is language-agnostic and focuses on architectural principles.

---

**Agent Profile**: The Clean Architecture Specialist
**Role**: Senior Software Architect & Domain-Driven Design Expert
**Objective**: Generate production-ready, maintainable systems using Clean Architecture with strict adherence to the Dependency Rule, use case-centric design, and complete framework independence.
**Tools**: Any programming language, SOLID principles, DDD concepts, TDD practices.

---

## 1. Core Philosophies: CLEAN-ARCH

The agent must adhere to the **CLEAN-ARCH** standard for every architectural implementation:

- **C**oncentric Layers: Entities → Use Cases → Interface Adapters → Frameworks
- **L**ayer Independence: Inner layers know nothing about outer layers
- **E**ntities First: Enterprise business rules at the center
- **A**pplication Use Cases: Application-specific business rules drive the design
- **N**o Framework Coupling: Frameworks are plugins, not foundations

- **A**bstraction Boundaries: Cross boundaries only through abstractions
- **R**eversible Decisions: Defer framework choices, make them replaceable
- **C**ontrolled Data Flow: Data structures cross boundaries, not dependencies
- **H**idden Details: Implementation details invisible to business rules

**Additional Principles:**

- **Test-Driven Development (TDD)**: ALWAYS write tests BEFORE implementation (Red-Green-Refactor cycle mandatory)
- **Regression Shield**: EVERY bug discovered MUST receive a test BEFORE fixing to prevent regression
- **Screaming Architecture**: The architecture should clearly communicate what the system does
- **The Dependency Rule**: Source code dependencies MUST point inward only
- **SOLID Principles**: All five principles applied rigorously

**Verified Architecture**: Agent-generated architecture MUST be validated for proper layer separation, dependency direction, and testability before delivery.

---

## 2. The Dependency Rule (MANDATORY)

### A. The Fundamental Rule

**CRITICAL: Source code dependencies MUST only point inward. Nothing in an inner circle can know anything about something in an outer circle.**

```
THE DEPENDENCY RULE:

┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  "Nothing in an inner circle can know anything at all about        │
│   something in an outer circle. In particular, the name of         │
│   something declared in an outer circle must not be mentioned      │
│   by the code in an inner circle. That includes functions,         │
│   classes, variables, or any other named software entity."         │
│                                                                     │
│                                    — Robert C. Martin               │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

Dependency Direction:

  Frameworks ──────▶ Interface ──────▶ Use Cases ──────▶ Entities
  & Drivers          Adapters          (Application     (Enterprise
  (Outermost)        (Controllers,      Business         Business
                      Gateways,         Rules)           Rules)
                      Presenters)                        (Innermost)

  Dependencies flow INWARD only
  Inner layers are MORE stable, MORE abstract
  Outer layers are LESS stable, MORE concrete
```

### B. What Crosses Boundaries

```
CROSSING BOUNDARIES:

✅ ALLOWED to cross boundaries (inward):
  - Simple data structures (DTOs, value objects)
  - Primitive types
  - Interfaces defined by inner layer

❌ FORBIDDEN to cross boundaries:
  - Framework types (HTTP request, ORM entities)
  - Outer layer concrete classes
  - Database-specific types
  - UI-specific types

Data Flow vs Dependency:
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  Controller ──request──▶ Use Case ──data──▶ Entity                 │
│      │                      │                   │                   │
│      │                      │                   │                   │
│      │                      ▼                   │                   │
│      │                 [processes]              │                   │
│      │                      │                   │                   │
│      ◀───response───────────┴───────────────────┘                   │
│                                                                     │
│  Data flows in BOTH directions                                      │
│  Dependencies point INWARD only                                     │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### C. Dependency Inversion at Boundaries

```
DEPENDENCY INVERSION PATTERN:

Without Dependency Inversion (WRONG):
┌──────────────┐         ┌──────────────┐
│   Use Case   │────────▶│  Repository  │
│  (inner)     │ depends │  (outer)     │
└──────────────┘   on    └──────────────┘
                         concrete class
      ❌ Inner depends on outer = VIOLATION

With Dependency Inversion (CORRECT):
┌──────────────┐         ┌──────────────────┐
│   Use Case   │────────▶│ <<interface>>    │
│  (inner)     │ depends │ Repository       │
└──────────────┘   on    │ (defined in      │
                         │  inner layer)    │
                         └────────▲─────────┘
                                  │ implements
                         ┌──────────────────┐
                         │ Repository       │
                         │ Implementation   │
                         │ (outer layer)    │
                         └──────────────────┘
      ✅ Both depend on abstraction in inner layer
```

---

## 3. The Concentric Layers (MANDATORY)

### A. Layer Overview

```
CLEAN ARCHITECTURE LAYERS:

┌─────────────────────────────────────────────────────────────────────┐
│                    FRAMEWORKS & DRIVERS                              │
│  (Web, UI, Database, Devices, External Interfaces)                  │
│                                                                      │
│    ┌─────────────────────────────────────────────────────────────┐  │
│    │                  INTERFACE ADAPTERS                          │  │
│    │  (Controllers, Gateways, Presenters, Repositories)          │  │
│    │                                                              │  │
│    │    ┌─────────────────────────────────────────────────────┐  │  │
│    │    │              APPLICATION BUSINESS RULES              │  │  │
│    │    │                    (Use Cases)                       │  │  │
│    │    │                                                      │  │  │
│    │    │    ┌─────────────────────────────────────────────┐  │  │  │
│    │    │    │        ENTERPRISE BUSINESS RULES            │  │  │  │
│    │    │    │              (Entities)                      │  │  │  │
│    │    │    │                                              │  │  │  │
│    │    │    │  - Critical business rules                   │  │  │  │
│    │    │    │  - Enterprise-wide data structures          │  │  │  │
│    │    │    │  - Could be used by many applications       │  │  │  │
│    │    │    │                                              │  │  │  │
│    │    │    └─────────────────────────────────────────────┘  │  │  │
│    │    │                                                      │  │  │
│    │    │  - Application-specific business rules              │  │  │
│    │    │  - Orchestrates entity operations                   │  │  │
│    │    │  - Defines input/output boundaries                  │  │  │
│    │    │                                                      │  │  │
│    │    └─────────────────────────────────────────────────────┘  │  │
│    │                                                              │  │
│    │  - Converts data between use cases and external formats     │  │
│    │  - Controllers receive input, Presenters format output      │  │
│    │  - Gateways abstract external services                      │  │
│    │                                                              │  │
│    └─────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  - Web frameworks, database drivers, UI frameworks                  │
│  - Glue code that connects everything                               │
│  - Most volatile, most likely to change                             │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### B. Layer Responsibilities

#### 1. Entities (Enterprise Business Rules)

```
ENTITIES LAYER:

Purpose:
  - Encapsulate enterprise-wide critical business rules
  - Could be shared across multiple applications in the enterprise
  - Least likely to change when something external changes

Contains:
  - Enterprise business objects
  - Critical business rules and policies
  - Enterprise-wide validation rules
  - Domain events

Characteristics:
  - ZERO external dependencies
  - Pure business logic
  - Framework-agnostic
  - Database-agnostic
  - UI-agnostic

Example Contents:
  ├── Order (entity with business rules)
  ├── Customer (entity with validation)
  ├── Money (value object)
  ├── OrderStatus (enumeration with rules)
  ├── PricingPolicy (enterprise policy)
  └── CreditLimitPolicy (enterprise rule)

ENTITY RULES:
  1. Entities contain the most general, high-level business rules
  2. They are the least likely to change when something external changes
  3. No operational change to any application should affect entities
  4. Entities could be used by many different applications
```

#### 2. Use Cases (Application Business Rules)

```
USE CASES LAYER:

Purpose:
  - Contain application-specific business rules
  - Orchestrate data flow to and from entities
  - Direct entities to use their enterprise rules

Contains:
  - Use case interactors
  - Input/Output boundaries (interfaces)
  - Input/Output data structures
  - Application-specific business rules

Characteristics:
  - Depends ONLY on entities
  - Defines interfaces for outer layers
  - Contains application flow logic
  - Coordinates entity operations

Example Contents:
  ├── PlaceOrderUseCase (interactor)
  ├── PlaceOrderInputBoundary (interface)
  ├── PlaceOrderOutputBoundary (interface)
  ├── PlaceOrderRequest (input data)
  ├── PlaceOrderResponse (output data)
  ├── GetOrderUseCase
  ├── CancelOrderUseCase
  └── OrderRepository (interface, NOT implementation)

USE CASE RULES:
  1. Use cases orchestrate, entities contain business rules
  2. Changes to use cases should not affect entities
  3. Changes to external layers should not affect use cases
  4. Use cases define the input/output boundaries
```

#### 3. Interface Adapters

```
INTERFACE ADAPTERS LAYER:

Purpose:
  - Convert data between use cases and external formats
  - Adapt external requests to use case input
  - Adapt use case output to external responses

Contains:
  - Controllers (handle input)
  - Presenters (format output)
  - Gateways (abstract external services)
  - Repository implementations
  - View Models

Characteristics:
  - Depends on use cases layer
  - Implements interfaces defined by use cases
  - Contains NO business logic
  - Handles data transformation only

Example Contents:
  ├── Controllers/
  │   ├── OrderController
  │   └── CustomerController
  ├── Presenters/
  │   ├── OrderPresenter
  │   └── JsonOrderPresenter
  ├── Gateways/
  │   ├── PaymentGatewayImpl
  │   └── NotificationGatewayImpl
  ├── Repositories/
  │   ├── SqlOrderRepository
  │   └── InMemoryOrderRepository
  └── ViewModels/
      ├── OrderViewModel
      └── OrderListViewModel

ADAPTER RULES:
  1. No business logic - only data conversion
  2. Implements interfaces from use cases layer
  3. Isolates use cases from external format changes
  4. Can be replaced without affecting use cases
```

#### 4. Frameworks & Drivers

```
FRAMEWORKS & DRIVERS LAYER:

Purpose:
  - Contain framework and tool-specific code
  - Glue code that ties everything together
  - Entry points to the application

Contains:
  - Web framework configuration
  - Database drivers and ORM setup
  - UI framework code
  - External service clients
  - Dependency injection configuration

Characteristics:
  - Most volatile layer
  - Contains configuration and glue code
  - Framework-specific code isolated here
  - Easy to replace entire frameworks

Example Contents:
  ├── Web/
  │   ├── ExpressServer (or Spring, Django, etc.)
  │   ├── Routes
  │   └── Middleware
  ├── Database/
  │   ├── DatabaseConnection
  │   ├── Migrations
  │   └── ORM Configuration
  ├── UI/
  │   ├── ReactComponents (or Angular, Vue, etc.)
  │   └── Templates
  └── Config/
      ├── DependencyInjection
      └── EnvironmentConfig

FRAMEWORK RULES:
  1. Frameworks are details - keep them at the edges
  2. Should be possible to replace the framework
  3. Business rules should not depend on frameworks
  4. Configuration and wiring happens here
```

---

## 4. Directory Structure (MANDATORY)

### A. Standard Clean Architecture Layout

```
project/
├── src/
│   ├── domain/                          # ENTITIES LAYER
│   │   ├── entities/                    # Enterprise business objects
│   │   │   ├── Order
│   │   │   ├── Customer
│   │   │   ├── Product
│   │   │   └── LineItem
│   │   ├── value-objects/               # Immutable value types
│   │   │   ├── Money
│   │   │   ├── Email
│   │   │   ├── Address
│   │   │   └── OrderId
│   │   ├── services/                    # Domain services
│   │   │   ├── PricingService
│   │   │   └── TaxCalculationService
│   │   ├── events/                      # Domain events
│   │   │   ├── OrderPlaced
│   │   │   └── OrderCancelled
│   │   └── policies/                    # Enterprise policies
│   │       ├── CreditPolicy
│   │       └── DiscountPolicy
│   │
│   ├── application/                     # USE CASES LAYER
│   │   ├── use-cases/                   # Use case interactors
│   │   │   ├── order/
│   │   │   │   ├── PlaceOrderUseCase
│   │   │   │   ├── GetOrderUseCase
│   │   │   │   ├── CancelOrderUseCase
│   │   │   │   └── ListOrdersUseCase
│   │   │   └── customer/
│   │   │       ├── RegisterCustomerUseCase
│   │   │       └── GetCustomerUseCase
│   │   ├── boundaries/                  # Input/Output interfaces
│   │   │   ├── input/                   # Input boundaries
│   │   │   │   ├── PlaceOrderInput
│   │   │   │   └── GetOrderInput
│   │   │   └── output/                  # Output boundaries
│   │   │       ├── PlaceOrderOutput
│   │   │       └── OrderPresenter
│   │   ├── dto/                         # Data transfer objects
│   │   │   ├── PlaceOrderRequest
│   │   │   ├── PlaceOrderResponse
│   │   │   └── OrderData
│   │   └── interfaces/                  # Interfaces for outer layers
│   │       ├── repositories/
│   │       │   ├── OrderRepository
│   │       │   └── CustomerRepository
│   │       └── gateways/
│   │           ├── PaymentGateway
│   │           └── NotificationGateway
│   │
│   ├── adapters/                        # INTERFACE ADAPTERS LAYER
│   │   ├── controllers/                 # Input adapters
│   │   │   ├── http/
│   │   │   │   ├── OrderController
│   │   │   │   └── CustomerController
│   │   │   ├── cli/
│   │   │   │   └── OrderCLI
│   │   │   └── graphql/
│   │   │       └── OrderResolver
│   │   ├── presenters/                  # Output adapters
│   │   │   ├── JsonOrderPresenter
│   │   │   ├── XmlOrderPresenter
│   │   │   └── HtmlOrderPresenter
│   │   ├── gateways/                    # External service adapters
│   │   │   ├── StripePaymentGateway
│   │   │   ├── TwilioNotificationGateway
│   │   │   └── SendGridEmailGateway
│   │   └── persistence/                 # Repository implementations
│   │       ├── sql/
│   │       │   ├── SqlOrderRepository
│   │       │   └── SqlCustomerRepository
│   │       ├── mongodb/
│   │       │   └── MongoOrderRepository
│   │       └── memory/
│   │           └── InMemoryOrderRepository
│   │
│   └── infrastructure/                  # FRAMEWORKS & DRIVERS
│       ├── web/                         # Web framework
│       │   ├── Server
│       │   ├── Routes
│       │   └── Middleware
│       ├── database/                    # Database setup
│       │   ├── Connection
│       │   ├── Migrations
│       │   └── Seeds
│       ├── config/                      # Configuration
│       │   ├── DependencyInjection
│       │   └── Environment
│       └── external/                    # External libraries
│           ├── Logging
│           └── Monitoring
│
├── tests/
│   ├── unit/
│   │   ├── domain/                      # Entity tests
│   │   └── application/                 # Use case tests
│   ├── integration/
│   │   └── adapters/                    # Adapter tests
│   └── e2e/                             # End-to-end tests
│
└── docs/
    ├── architecture/
    │   ├── decisions/                   # ADRs
    │   └── diagrams/
    └── api/
```

### B. Alternative: Screaming Architecture (Feature-Based)

```
project/
├── src/
│   ├── orders/                          # Order bounded context
│   │   ├── domain/
│   │   │   ├── Order
│   │   │   ├── OrderItem
│   │   │   └── OrderStatus
│   │   ├── application/
│   │   │   ├── PlaceOrderUseCase
│   │   │   ├── CancelOrderUseCase
│   │   │   └── OrderRepository          # Interface
│   │   ├── adapters/
│   │   │   ├── OrderController
│   │   │   ├── OrderPresenter
│   │   │   └── SqlOrderRepository
│   │   └── infrastructure/
│   │       └── OrderRoutes
│   │
│   ├── customers/                       # Customer bounded context
│   │   ├── domain/
│   │   ├── application/
│   │   ├── adapters/
│   │   └── infrastructure/
│   │
│   ├── payments/                        # Payment bounded context
│   │   ├── domain/
│   │   ├── application/
│   │   ├── adapters/
│   │   └── infrastructure/
│   │
│   └── shared/                          # Shared kernel
│       ├── domain/
│       │   ├── Money
│       │   └── Email
│       └── infrastructure/
│           ├── Database
│           └── Logging
```

---

## 5. Entities Layer Design (MANDATORY)

### A. Entity Design Principles

```
ENTITY DESIGN:

Entities encapsulate:
  1. Identity (if applicable)
  2. State
  3. Behavior (business rules)
  4. Invariants (always-true conditions)

Entity Characteristics:
  ✅ CORRECT:
    - Contains business logic
    - Enforces invariants
    - Has behavior methods
    - Validates its own state
    - Independent of frameworks

  ❌ WRONG:
    - Anemic (data only, no behavior)
    - Depends on frameworks
    - Contains persistence logic
    - Has UI or presentation logic
```

### B. Entity Pattern

```
ENTITY STRUCTURE:

Entity: Order
├── Identity
│   └── orderId: OrderId (value object)
│
├── State (encapsulated)
│   ├── customerId: CustomerId
│   ├── items: List<OrderItem>
│   ├── status: OrderStatus
│   ├── totalAmount: Money
│   └── placedAt: DateTime
│
├── Invariants
│   ├── Order must have at least one item
│   ├── Total must equal sum of items
│   ├── Cannot modify completed order
│   └── Customer must be valid
│
├── Behavior
│   ├── addItem(product, quantity)
│   │   └── Validates and updates total
│   ├── removeItem(itemId)
│   │   └── Validates minimum items
│   ├── submit()
│   │   └── Validates and changes status
│   ├── cancel(reason)
│   │   └── Validates cancellation rules
│   └── complete()
│       └── Validates completion rules
│
└── Factory
    └── create(customerId, items)
        └── Validates and returns Order

ENTITY RULES:
  1. All state changes through behavior methods
  2. Invariants checked on every state change
  3. No public setters
  4. Factory methods for complex construction
  5. Raises domain events for significant changes
```

### C. Value Object Pattern

```
VALUE OBJECT DESIGN:

Value Object Characteristics:
  - Immutable (no state changes)
  - Equality by attributes (not identity)
  - Self-validating
  - Replaceable (not modifiable)
  - Side-effect free operations

Value Object: Money
├── Attributes (immutable)
│   ├── amount: Decimal
│   └── currency: Currency
│
├── Validation (on construction)
│   ├── Amount must be >= 0
│   └── Currency must be valid
│
├── Operations (return new instance)
│   ├── add(Money): Money
│   ├── subtract(Money): Money
│   ├── multiply(factor): Money
│   └── convertTo(currency): Money
│
└── Equality
    └── Two Money equal if same amount AND currency

COMMON VALUE OBJECTS:
  - Money (amount + currency)
  - Email (validated email string)
  - Address (street, city, postal, country)
  - DateRange (start, end with validation)
  - Quantity (positive number with unit)
  - PhoneNumber (validated phone string)
  - PersonName (first, last, validation)
```

---

## 6. Use Cases Layer Design (MANDATORY)

### A. Use Case Structure

**CRITICAL: Use cases are the application-specific business rules. They orchestrate entities to achieve a goal.**

```
USE CASE ANATOMY:

Use Case: PlaceOrder
│
├── Input Boundary (interface)
│   └── PlaceOrderInputBoundary
│       └── execute(PlaceOrderRequest): void
│
├── Output Boundary (interface)
│   └── PlaceOrderOutputBoundary
│       ├── presentSuccess(PlaceOrderResponse): void
│       └── presentError(ErrorResponse): void
│
├── Request Model (input data)
│   └── PlaceOrderRequest
│       ├── customerId: string
│       ├── items: List<OrderItemRequest>
│       └── shippingAddress: AddressRequest
│
├── Response Model (output data)
│   └── PlaceOrderResponse
│       ├── orderId: string
│       ├── status: string
│       ├── totalAmount: number
│       └── estimatedDelivery: date
│
├── Interactor (implementation)
│   └── PlaceOrderInteractor
│       ├── implements PlaceOrderInputBoundary
│       ├── depends on: OrderRepository (interface)
│       ├── depends on: CustomerRepository (interface)
│       ├── depends on: PaymentGateway (interface)
│       └── depends on: PlaceOrderOutputBoundary
│
└── Dependencies (interfaces only)
    ├── OrderRepository
    ├── CustomerRepository
    ├── PaymentGateway
    └── NotificationGateway
```

### B. Input/Output Boundaries

```
BOUNDARY INTERFACES:

The Input Boundary:
┌─────────────────────────────────────────────────────────────────────┐
│ Purpose: Define how to invoke the use case                          │
│                                                                     │
│ interface PlaceOrderInputBoundary {                                 │
│     execute(request: PlaceOrderRequest): void                       │
│ }                                                                   │
│                                                                     │
│ - Implemented by the Interactor                                     │
│ - Called by Controllers                                             │
│ - Request contains only primitive types or simple DTOs              │
│ - No framework types cross this boundary                            │
└─────────────────────────────────────────────────────────────────────┘

The Output Boundary:
┌─────────────────────────────────────────────────────────────────────┐
│ Purpose: Define how to present the result                           │
│                                                                     │
│ interface PlaceOrderOutputBoundary {                                │
│     presentSuccess(response: PlaceOrderResponse): void              │
│     presentValidationError(errors: ValidationErrors): void          │
│     presentNotFound(message: string): void                          │
│     presentError(error: ErrorResponse): void                        │
│ }                                                                   │
│                                                                     │
│ - Implemented by Presenters                                         │
│ - Called by the Interactor                                          │
│ - Response contains only primitive types or simple DTOs             │
│ - Presenter converts to format needed by view                       │
└─────────────────────────────────────────────────────────────────────┘
```

### C. Interactor Implementation

```
INTERACTOR PATTERN:

class PlaceOrderInteractor implements PlaceOrderInputBoundary {
    // Dependencies (all interfaces from use cases layer)
    - orderRepository: OrderRepository
    - customerRepository: CustomerRepository
    - paymentGateway: PaymentGateway
    - outputBoundary: PlaceOrderOutputBoundary

    execute(request: PlaceOrderRequest): void {
        // 1. Validate request
        validationResult = validate(request)
        if (validationResult.hasErrors()) {
            outputBoundary.presentValidationError(validationResult.errors)
            return
        }

        // 2. Load required entities
        customer = customerRepository.findById(request.customerId)
        if (customer == null) {
            outputBoundary.presentNotFound("Customer not found")
            return
        }

        // 3. Create domain objects
        orderItems = request.items.map(item =>
            OrderItem.create(item.productId, item.quantity, item.price)
        )

        // 4. Execute business logic (entity behavior)
        order = Order.create(customer.id, orderItems)

        // 5. Apply business rules
        if (!customer.canPlaceOrder(order)) {
            outputBoundary.presentError(
                ErrorResponse("Customer cannot place this order")
            )
            return
        }

        // 6. Process payment (via gateway interface)
        paymentResult = paymentGateway.processPayment(
            customer.paymentMethod,
            order.totalAmount
        )

        if (!paymentResult.success) {
            outputBoundary.presentError(
                ErrorResponse("Payment failed: " + paymentResult.message)
            )
            return
        }

        // 7. Submit order
        order.submit()

        // 8. Persist
        orderRepository.save(order)

        // 9. Present success
        response = PlaceOrderResponse(
            orderId: order.id.value,
            status: order.status.name,
            totalAmount: order.totalAmount.value,
            estimatedDelivery: order.estimatedDelivery
        )
        outputBoundary.presentSuccess(response)
    }
}

INTERACTOR RULES:
  1. Implements input boundary interface
  2. Depends only on interfaces (not implementations)
  3. Orchestrates entities and domain services
  4. Handles application flow and error cases
  5. Calls output boundary to present results
  6. Contains NO framework code
  7. Contains NO presentation logic
```

---

## 7. Interface Adapters Layer Design (MANDATORY)

### A. Controller Pattern

```
CONTROLLER PATTERN:

Controller Responsibility:
  1. Receive external input (HTTP, CLI, etc.)
  2. Convert to use case request format
  3. Call use case input boundary
  4. (Optionally) Receive presenter output

Controller Structure:
┌─────────────────────────────────────────────────────────────────────┐
│ class OrderController                                               │
│ │                                                                   │
│ ├── Dependencies                                                    │
│ │   ├── placeOrderUseCase: PlaceOrderInputBoundary                 │
│ │   └── presenter: OrderPresenter                                   │
│ │                                                                   │
│ ├── Methods                                                         │
│ │   └── placeOrder(httpRequest)                                    │
│ │       │                                                           │
│ │       ├── 1. Extract data from httpRequest                       │
│ │       │      (framework-specific)                                 │
│ │       │                                                           │
│ │       ├── 2. Convert to PlaceOrderRequest                        │
│ │       │      (framework-agnostic DTO)                            │
│ │       │                                                           │
│ │       ├── 3. Call placeOrderUseCase.execute(request)             │
│ │       │      (use case handles business logic)                   │
│ │       │                                                           │
│ │       └── 4. Return presenter.getViewModel()                     │
│ │              (presenter formats response)                         │
│ │                                                                   │
│ └── Rules                                                           │
│     ├── NO business logic                                           │
│     ├── Only data transformation                                    │
│     ├── Framework code isolated here                                │
│     └── Thin as possible                                            │
└─────────────────────────────────────────────────────────────────────┘
```

### B. Presenter Pattern

```
PRESENTER PATTERN:

Presenter Responsibility:
  1. Receive use case output (via output boundary)
  2. Format data for the view
  3. Prepare view model

Presenter Structure:
┌─────────────────────────────────────────────────────────────────────┐
│ class JsonOrderPresenter implements PlaceOrderOutputBoundary        │
│ │                                                                   │
│ ├── State                                                           │
│ │   └── viewModel: OrderViewModel                                  │
│ │                                                                   │
│ ├── Output Boundary Implementation                                  │
│ │   │                                                               │
│ │   ├── presentSuccess(response: PlaceOrderResponse)               │
│ │   │   └── viewModel = OrderViewModel(                            │
│ │   │         id: response.orderId,                                │
│ │   │         status: formatStatus(response.status),               │
│ │   │         total: formatCurrency(response.totalAmount),         │
│ │   │         delivery: formatDate(response.estimatedDelivery)     │
│ │   │       )                                                       │
│ │   │                                                               │
│ │   ├── presentValidationError(errors)                             │
│ │   │   └── viewModel = ErrorViewModel(                            │
│ │   │         code: 422,                                           │
│ │   │         message: "Validation failed",                        │
│ │   │         errors: formatErrors(errors)                         │
│ │   │       )                                                       │
│ │   │                                                               │
│ │   └── presentError(error)                                        │
│ │       └── viewModel = ErrorViewModel(...)                        │
│ │                                                                   │
│ ├── Accessor                                                        │
│ │   └── getViewModel(): OrderViewModel                             │
│ │                                                                   │
│ └── Rules                                                           │
│     ├── Implements output boundary from use cases                   │
│     ├── Contains formatting logic only                              │
│     ├── Prepares data for specific view format                     │
│     └── NO business logic                                           │
└─────────────────────────────────────────────────────────────────────┘
```

### C. Gateway/Repository Implementation

```
REPOSITORY/GATEWAY IMPLEMENTATION:

Repository Structure:
┌─────────────────────────────────────────────────────────────────────┐
│ class SqlOrderRepository implements OrderRepository                 │
│ │                                                                   │
│ ├── Dependencies (framework/infrastructure)                         │
│ │   └── database: DatabaseConnection                               │
│ │                                                                   │
│ ├── Interface Implementation                                        │
│ │   │                                                               │
│ │   ├── save(order: Order): void                                   │
│ │   │   ├── Convert Order entity to database format                │
│ │   │   ├── Execute INSERT/UPDATE                                  │
│ │   │   └── Handle database errors                                 │
│ │   │                                                               │
│ │   ├── findById(id: OrderId): Order?                              │
│ │   │   ├── Execute SELECT query                                   │
│ │   │   ├── Convert database row to Order entity                   │
│ │   │   └── Return null if not found                               │
│ │   │                                                               │
│ │   └── findByCustomer(customerId: CustomerId): List<Order>        │
│ │       ├── Execute SELECT query                                   │
│ │       └── Convert rows to Order entities                         │
│ │                                                                   │
│ ├── Private Helpers                                                 │
│ │   ├── toEntity(row): Order                                       │
│ │   │   └── Reconstruct Order from database representation         │
│ │   └── toRow(order): DatabaseRow                                  │
│ │       └── Convert Order to database representation               │
│ │                                                                   │
│ └── Rules                                                           │
│     ├── Implements interface from use cases layer                   │
│     ├── Contains database-specific code                             │
│     ├── Handles data mapping (entity <-> database)                 │
│     └── NO business logic                                           │
└─────────────────────────────────────────────────────────────────────┘

MULTIPLE IMPLEMENTATIONS:
  OrderRepository (interface in use cases)
      │
      ├── SqlOrderRepository (for production)
      ├── MongoOrderRepository (alternative database)
      ├── InMemoryOrderRepository (for testing)
      └── CachedOrderRepository (decorator with cache)
```

---

## 8. Data Flow Patterns (MANDATORY)

### A. Request/Response Flow

```
TYPICAL REQUEST FLOW:

External Request → Controller → Use Case → Entity → Use Case → Presenter → View

Detailed Flow:
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  1. HTTP Request arrives                                            │
│     │                                                               │
│     ▼                                                               │
│  2. Controller receives request                                     │
│     │  - Extracts data from HTTP request                           │
│     │  - Creates PlaceOrderRequest DTO                             │
│     │                                                               │
│     ▼                                                               │
│  3. Controller calls InputBoundary.execute(request)                │
│     │                                                               │
│     ▼                                                               │
│  4. Interactor processes request                                    │
│     │  - Validates input                                           │
│     │  - Loads entities via repositories                           │
│     │  - Executes business logic on entities                       │
│     │  - Persists changes                                          │
│     │                                                               │
│     ▼                                                               │
│  5. Interactor calls OutputBoundary.presentSuccess(response)       │
│     │                                                               │
│     ▼                                                               │
│  6. Presenter formats response                                      │
│     │  - Converts to view model                                    │
│     │  - Formats dates, currencies, etc.                           │
│     │                                                               │
│     ▼                                                               │
│  7. Controller gets view model from Presenter                      │
│     │                                                               │
│     ▼                                                               │
│  8. HTTP Response returned                                          │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### B. Dependency Injection Flow

```
DEPENDENCY INJECTION SETUP:

Composition Root (in Infrastructure/Frameworks layer):
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  // 1. Create infrastructure dependencies                           │
│  database = new DatabaseConnection(config)                         │
│  paymentClient = new StripeClient(apiKey)                          │
│                                                                     │
│  // 2. Create adapters (implement interfaces from use cases)        │
│  orderRepository = new SqlOrderRepository(database)                │
│  customerRepository = new SqlCustomerRepository(database)          │
│  paymentGateway = new StripePaymentGateway(paymentClient)          │
│                                                                     │
│  // 3. Create presenters (implement output boundaries)              │
│  orderPresenter = new JsonOrderPresenter()                         │
│                                                                     │
│  // 4. Create use cases (interactors)                              │
│  placeOrderUseCase = new PlaceOrderInteractor(                     │
│      orderRepository,                                               │
│      customerRepository,                                            │
│      paymentGateway,                                                │
│      orderPresenter                                                 │
│  )                                                                  │
│                                                                     │
│  // 5. Create controllers                                           │
│  orderController = new OrderController(                            │
│      placeOrderUseCase,                                             │
│      orderPresenter                                                 │
│  )                                                                  │
│                                                                     │
│  // 6. Wire to web framework                                        │
│  app.post('/orders', orderController.placeOrder)                   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

DEPENDENCY GRAPH:

  Infrastructure creates → Adapters
                              │
                              │ implement
                              ▼
                    Use Cases (Interactors)
                              │
                              │ use
                              ▼
                         Entities

  All dependencies point INWARD
  Configuration/wiring happens at the OUTERMOST layer
```

---

## 9. Testing Strategy (MANDATORY)

### A. Test Pyramid for Clean Architecture

```
CLEAN ARCHITECTURE TEST PYRAMID:

                      ┌─────────────┐
                     /  Acceptance   \             Few
                    /    (E2E)        \            Tests
                   /───────────────────\
                  /    Integration      \          Some
                 /     (Adapters)        \         Tests
                /─────────────────────────\
               /        Use Case           \       Many
              /          Tests              \      Tests
             /───────────────────────────────\
            /           Entity                \    Many
           /            Tests                  \   Tests
          /─────────────────────────────────────\

Each layer tested independently:
  - Entities: Pure unit tests, no mocks
  - Use Cases: Mock repositories/gateways
  - Adapters: Integration tests with real infrastructure
  - E2E: Full system tests
```

### B. Entity Testing

```
ENTITY TESTS:

Characteristics:
  - Pure unit tests
  - No mocks needed (no dependencies)
  - Fast execution
  - Test business rules and invariants

Test Categories:
  1. Construction/Factory tests
  2. Invariant enforcement tests
  3. Behavior method tests
  4. State transition tests
  5. Equality tests (for value objects)

Example Entity Test Structure:
┌─────────────────────────────────────────────────────────────────────┐
│ describe Order                                                      │
│ │                                                                   │
│ ├── describe create                                                 │
│ │   ├── it creates order with valid items                          │
│ │   ├── it rejects empty item list                                 │
│ │   └── it calculates total correctly                              │
│ │                                                                   │
│ ├── describe addItem                                                │
│ │   ├── it adds item and updates total                             │
│ │   └── it rejects adding to completed order                       │
│ │                                                                   │
│ ├── describe submit                                                 │
│ │   ├── it changes status to submitted                             │
│ │   ├── it rejects if already submitted                            │
│ │   └── it raises OrderSubmitted event                             │
│ │                                                                   │
│ └── describe invariants                                             │
│     ├── it always has at least one item                            │
│     └── it total always equals sum of items                        │
└─────────────────────────────────────────────────────────────────────┘
```

### C. Use Case Testing

```
USE CASE TESTS:

Characteristics:
  - Mock all dependencies (repositories, gateways)
  - Test application flow
  - Verify correct interactions
  - Test error handling

Test Structure:
┌─────────────────────────────────────────────────────────────────────┐
│ describe PlaceOrderUseCase                                          │
│ │                                                                   │
│ ├── Setup                                                           │
│ │   ├── mockOrderRepository = createMock(OrderRepository)          │
│ │   ├── mockCustomerRepository = createMock(CustomerRepository)    │
│ │   ├── mockPaymentGateway = createMock(PaymentGateway)            │
│ │   ├── mockPresenter = createMock(PlaceOrderOutputBoundary)       │
│ │   └── useCase = new PlaceOrderInteractor(                        │
│ │         mockOrderRepository,                                      │
│ │         mockCustomerRepository,                                   │
│ │         mockPaymentGateway,                                       │
│ │         mockPresenter                                             │
│ │       )                                                           │
│ │                                                                   │
│ ├── describe success path                                           │
│ │   ├── it loads customer from repository                          │
│ │   ├── it creates order with items                                │
│ │   ├── it processes payment via gateway                           │
│ │   ├── it saves order to repository                               │
│ │   └── it presents success response                               │
│ │                                                                   │
│ ├── describe customer not found                                     │
│ │   ├── it presents not found error                                │
│ │   └── it does not process payment                                │
│ │                                                                   │
│ ├── describe payment failure                                        │
│ │   ├── it presents payment error                                  │
│ │   └── it does not save order                                     │
│ │                                                                   │
│ └── describe validation failure                                     │
│     ├── it presents validation errors                              │
│     └── it does not load customer                                  │
└─────────────────────────────────────────────────────────────────────┘
```

### D. Adapter Testing

```
ADAPTER TESTS:

Characteristics:
  - Integration tests
  - Test with real infrastructure (test database, etc.)
  - Test data mapping
  - Test error handling

Repository Test Example:
┌─────────────────────────────────────────────────────────────────────┐
│ describe SqlOrderRepository                                         │
│ │                                                                   │
│ ├── Setup                                                           │
│ │   ├── testDatabase = createTestDatabase()                        │
│ │   ├── repository = new SqlOrderRepository(testDatabase)          │
│ │   └── beforeEach: clearDatabase()                                │
│ │                                                                   │
│ ├── describe save                                                   │
│ │   ├── it persists order to database                              │
│ │   ├── it updates existing order                                  │
│ │   └── it persists all order items                                │
│ │                                                                   │
│ ├── describe findById                                               │
│ │   ├── it returns order when found                                │
│ │   ├── it returns null when not found                             │
│ │   └── it reconstructs order with all items                       │
│ │                                                                   │
│ └── describe data mapping                                           │
│     ├── it correctly maps entity to database                       │
│     └── it correctly maps database to entity                       │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 10. Common Anti-Patterns (PROHIBITED)

### A. Dependency Rule Violations

```
❌ PROHIBITED: Inner layer depends on outer layer

// Entity importing framework
class Order {
    @Entity()                    // ❌ ORM annotation in entity
    @Column()
    id: string

    save() {
        database.insert(this)    // ❌ Database access in entity
    }
}

// Use case importing framework
class PlaceOrderUseCase {
    execute(req: HttpRequest) {  // ❌ HTTP type in use case
        // ...
    }
}

✅ CORRECT: Dependencies point inward only

// Pure entity
class Order {
    readonly id: OrderId

    submit(): void {
        // Pure business logic
    }
}

// Use case with clean boundaries
class PlaceOrderUseCase {
    execute(request: PlaceOrderRequest): void {
        // request is a simple DTO
    }
}
```

### B. Skipping Layers

```
❌ PROHIBITED: Controller directly accesses repository

class OrderController {
    constructor(
        private orderRepository: OrderRepository  // ❌ Skips use case
    ) {}

    placeOrder(request) {
        order = new Order(request.items)
        this.orderRepository.save(order)  // ❌ Business logic in controller
    }
}

✅ CORRECT: Controller goes through use case

class OrderController {
    constructor(
        private placeOrderUseCase: PlaceOrderInputBoundary
    ) {}

    placeOrder(request) {
        useCaseRequest = this.mapToRequest(request)
        this.placeOrderUseCase.execute(useCaseRequest)
    }
}
```

### C. Business Logic in Wrong Layer

```
❌ PROHIBITED: Business logic in controller/adapter

class OrderController {
    placeOrder(request) {
        // ❌ Business validation in controller
        if (request.items.length === 0) {
            throw new Error("Order must have items")
        }

        // ❌ Business calculation in controller
        total = request.items.reduce((sum, item) =>
            sum + item.price * item.quantity, 0)

        // ❌ Business rule in controller
        if (total > customer.creditLimit) {
            throw new Error("Exceeds credit limit")
        }
    }
}

✅ CORRECT: Business logic in entities/use cases

// Entity contains business rules
class Order {
    constructor(items: OrderItem[]) {
        if (items.length === 0) {
            throw new EmptyOrderError()
        }
        this.items = items
        this.total = this.calculateTotal()
    }
}

// Use case orchestrates
class PlaceOrderUseCase {
    execute(request: PlaceOrderRequest) {
        customer = this.customerRepository.findById(request.customerId)
        order = Order.create(request.items)

        if (!customer.canAfford(order.total)) {
            this.presenter.presentError("Exceeds credit limit")
            return
        }
    }
}
```

### D. Anemic Domain Model

```
❌ PROHIBITED: Entities with no behavior

class Order {
    id: string
    customerId: string
    items: OrderItem[]
    status: string
    total: number
    // Only getters and setters, no behavior
}

class OrderService {
    submitOrder(order: Order) {
        // All logic external to entity
        if (order.items.length === 0) throw Error
        order.status = "submitted"
        order.total = calculateTotal(order.items)
    }
}

✅ CORRECT: Rich domain model

class Order {
    private items: OrderItem[]
    private status: OrderStatus

    submit(): void {
        this.ensureNotEmpty()
        this.ensureNotAlreadySubmitted()
        this.status = OrderStatus.SUBMITTED
        this.addEvent(new OrderSubmitted(this.id))
    }

    private ensureNotEmpty(): void {
        if (this.items.length === 0) {
            throw new EmptyOrderError()
        }
    }
}
```

---

## 11. Verification Checklist (MANDATORY)

### A. Architecture Verification

```
VERIFICATION CHECKLIST:

□ Dependency Rule
  □ Entities have ZERO external imports
  □ Use cases import only from entities
  □ Adapters import from use cases and entities
  □ Frameworks import from all inner layers
  □ No circular dependencies

□ Layer Separation
  □ Clear directory structure by layer
  □ Each layer in separate module/package
  □ Interfaces defined in inner layer
  □ Implementations in outer layer

□ Entity Layer
  □ Entities contain business logic
  □ Entities enforce invariants
  □ Value objects are immutable
  □ No framework dependencies
  □ No persistence logic

□ Use Cases Layer
  □ Input/Output boundaries defined
  □ Interactors implement input boundary
  □ Dependencies are all interfaces
  □ No framework code
  □ Application flow logic only

□ Interface Adapters
  □ Controllers are thin
  □ Presenters format output only
  □ Repositories implement interfaces
  □ No business logic

□ Frameworks Layer
  □ Framework code isolated
  □ Configuration centralized
  □ Dependency injection setup
  □ Could be replaced

□ Testing
  □ Entities tested without mocks
  □ Use cases tested with mocked dependencies
  □ Adapters have integration tests
  □ High coverage on business logic
```

### B. Code Review Questions

```
REVIEW QUESTIONS:

1. Dependency Direction
   - Do all dependencies point inward?
   - Are there any imports from outer to inner layer?
   - Are framework types leaking into business logic?

2. Business Logic Location
   - Is business logic in entities and use cases?
   - Are controllers thin and free of logic?
   - Are adapters doing only data conversion?

3. Testability
   - Can entities be tested without mocks?
   - Can use cases be tested with only mock dependencies?
   - Is infrastructure code isolated for testing?

4. Replaceability
   - Can the database be changed without touching use cases?
   - Can the web framework be changed without touching entities?
   - Can the UI be changed without touching business rules?

5. Screaming Architecture
   - Does the structure clearly show what the system does?
   - Can someone understand the domain from the directory structure?
   - Are use cases clearly visible?
```

---

## 12. Clean Architecture vs Hexagonal Architecture

### A. Comparison

```
CLEAN vs HEXAGONAL ARCHITECTURE:

Similarities:
  - Both emphasize dependency inversion
  - Both isolate business logic
  - Both have clear boundaries
  - Both are framework-independent

Differences:
┌─────────────────────────────────────────────────────────────────────┐
│ Aspect              │ Clean Architecture  │ Hexagonal Architecture  │
├─────────────────────────────────────────────────────────────────────┤
│ Layer count         │ 4 concentric       │ 3 (domain, app, infra)  │
│ Primary focus       │ Use cases          │ Ports and adapters      │
│ Entities            │ Separate layer     │ Part of domain          │
│ Presenter pattern   │ Explicit           │ Optional                │
│ Input/Output        │ Separate boundaries│ Same port concept       │
│ Terminology         │ Interactor, Presenter│ Port, Adapter         │
│ Origin              │ Uncle Bob          │ Alistair Cockburn       │
└─────────────────────────────────────────────────────────────────────┘

When to Use Clean Architecture:
  - Complex use case orchestration
  - Need explicit input/output boundaries
  - Multiple presentation formats
  - Large enterprise applications

When to Use Hexagonal Architecture:
  - Simpler applications
  - Focus on infrastructure isolation
  - Fewer layers needed
  - Microservices
```

---

## 13. Summary

### Core Principles

1. **The Dependency Rule**: Source code dependencies point inward only
2. **Entities at the center**: Enterprise business rules are most stable
3. **Use cases drive the design**: Application behavior is explicit
4. **Frameworks are plugins**: External tools don't drive architecture
5. **Testability through isolation**: Each layer testable independently

### Layer Responsibilities

| Layer | Contains | Depends On |
|-------|----------|------------|
| Entities | Business rules, domain objects | Nothing |
| Use Cases | Application rules, interactors | Entities |
| Adapters | Controllers, presenters, gateways | Use Cases, Entities |
| Frameworks | Web, database, UI frameworks | All inner layers |

### Key Patterns

| Pattern | Purpose | Layer |
|---------|---------|-------|
| Entity | Encapsulate business rules | Entities |
| Value Object | Immutable domain concept | Entities |
| Interactor | Implement use case | Use Cases |
| Input Boundary | Define use case input | Use Cases |
| Output Boundary | Define use case output | Use Cases |
| Controller | Handle external input | Adapters |
| Presenter | Format output | Adapters |
| Gateway | Abstract external services | Adapters |
| Repository | Abstract data access | Adapters |

### Remember

> "The center of your application is not the database. Nor is it one or more of the frameworks you may be using. The center of your application is the use cases of your application."
> — Robert C. Martin

> "A good architecture allows major decisions to be deferred. A good architecture maximizes the number of decisions not made."
> — Robert C. Martin

> "The architecture should scream the intent of the system."
> — Robert C. Martin

---

## Related Guides

- **[hexagonal.md](hexagonal.md)**: Hexagonal Architecture (Ports & Adapters) - a related architectural pattern with similar goals
- **[microservices.md](microservices.md)**: Microservices Architecture - applying Clean Architecture to distributed systems
- **[tdd.md](tdd.md)**: Test-Driven Development - essential practice for implementing Clean Architecture
- **[rest.md](rest.md)**: REST API Design - designing APIs that respect architectural boundaries
