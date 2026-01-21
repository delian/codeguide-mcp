# Microservices Architecture Guidelines
This document provides mandatory architectural standards and development practices for microservices architecture with emphasis on service autonomy, resilience, observability, and maintainability. This guide is language-agnostic and focuses on architectural principles.

---

**Agent Profile**: The Microservices Architect
**Role**: Senior Distributed Systems Architect & Platform Engineer
**Objective**: Generate production-ready, resilient, observable, and maintainable distributed systems using microservices architecture with clear service boundaries, proper communication patterns, and operational excellence.
**Tools**: Any programming language, container orchestration, message brokers, API gateways, service meshes, observability platforms.

---

## 1. Core Philosophies: MICROSERVICES

The agent must adhere to the **MICROSERVICES** standard for every architectural implementation:

- **M**odular Services: Small, focused services with single business capability
- **I**ndependent Deployment: Each service deployable without coordinating with others
- **C**ommunication Patterns: Well-defined sync and async communication strategies
- **R**esilience First: Design for failure with circuit breakers, retries, and fallbacks
- **O**bservability Built-in: Logging, metrics, and distributed tracing from day one
- **S**ecurity at Every Layer: Zero-trust, service-to-service authentication, secrets management
- **E**ventual Consistency: Embrace distributed data and asynchronous patterns
- **R**ight-Sized Services: Not too big (monolith), not too small (nano-services)
- **V**ersioned APIs: Backward-compatible changes, semantic versioning
- **I**solated Data: Database per service, no shared databases
- **C**ontainerized: Immutable infrastructure, container-first deployment
- **E**volvable Design: Services can be rewritten, replaced, or retired independently
- **S**calable Independently: Each service scales based on its own needs

**Additional Principles:**

- **Test-Driven Development (TDD)**: ALWAYS write tests BEFORE implementation (Red-Green-Refactor cycle mandatory)
- **Regression Shield**: EVERY bug discovered MUST receive a test BEFORE fixing to prevent regression
- **Domain-Driven Design**: Service boundaries align with bounded contexts
- **Infrastructure as Code**: All infrastructure is version-controlled and reproducible
- **GitOps**: Declarative infrastructure with Git as single source of truth

**Verified Architecture**: Agent-generated architecture MUST be validated for proper service boundaries, resilience patterns, and observability before delivery.

---

## 2. Service Boundaries (MANDATORY)

### A. Defining Service Boundaries

**CRITICAL: Service boundaries must align with business capabilities, not technical layers.**

```
SERVICE BOUNDARY PRINCIPLES:

1. Business Capability Alignment
   └── Each service owns ONE business capability
   └── Example: OrderService, PaymentService, InventoryService

2. Bounded Context (DDD)
   └── Services map to bounded contexts
   └── Each context has its own ubiquitous language
   └── Contexts communicate through well-defined interfaces

3. Team Ownership
   └── One team owns one or more services
   └── Team can deploy independently
   └── Conway's Law: Architecture mirrors organization

4. Data Ownership
   └── Each service owns its data
   └── No direct database sharing between services
   └── Data accessed only through service APIs
```

### B. Service Sizing Guidelines

```
┌─────────────────────────────────────────────────────────────────┐
│                    SERVICE SIZE SPECTRUM                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  TOO SMALL          JUST RIGHT              TOO LARGE           │
│  (Nano-service)     (Microservice)          (Mini-monolith)     │
│                                                                  │
│  ┌───┐              ┌─────────┐             ┌────────────────┐  │
│  │ • │              │ ••••••• │             │ •••••••••••••• │  │
│  └───┘              │ ••••••• │             │ •••••••••••••• │  │
│                     └─────────┘             │ •••••••••••••• │  │
│                                             └────────────────┘  │
│                                                                  │
│  Problems:          Characteristics:        Problems:            │
│  - Too many services- Single business       - Multiple bounded   │
│  - Network overhead   capability              contexts           │
│  - Distributed       - 1-2 week to build   - Shared database    │
│    complexity       - Team can understand  - Coordinated        │
│  - Hard to trace    - Independent deploy     deployments        │
│                     - Own database         - Hard to scale      │
│                     - 2-pizza team           independently      │
└─────────────────────────────────────────────────────────────────┘
```

### C. Service Boundary Checklist

```
SERVICE BOUNDARY VALIDATION:

□ Single Business Capability
  □ Service has one clear business purpose
  □ Service name is a business noun (OrderService, not OrderCreatorAndValidator)
  □ Changes to one capability don't require changes to others

□ Independent Lifecycle
  □ Service can be deployed independently
  □ Service can be scaled independently
  □ Service can be rewritten without affecting others

□ Data Autonomy
  □ Service owns its data store
  □ No shared database with other services
  □ Data accessed only through APIs

□ Team Ownership
  □ Single team owns the service
  □ Team has full responsibility (build, run, maintain)
  □ No cross-team coordination for deployments

□ Right Size
  □ Can be understood by one team
  □ Can be rewritten in 2-4 weeks
  □ Not just a single CRUD operation
```

---

## 3. Communication Patterns (MANDATORY)

### A. Synchronous Communication

**Use synchronous communication when you need an immediate response.**

```
SYNCHRONOUS PATTERNS:

┌─────────────────────────────────────────────────────────────┐
│                    REQUEST-RESPONSE                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Service A ──────────────────────────────────▶ Service B    │
│            │         Request                   │             │
│            │                                   │             │
│            ◀──────────────────────────────────│             │
│                      Response                                │
│                                                              │
│  Protocols: HTTP/REST, gRPC, GraphQL                        │
│  Use when: Need immediate response, simple queries          │
│  Avoid when: Long-running operations, unreliable network    │
└─────────────────────────────────────────────────────────────┘

SYNCHRONOUS COMMUNICATION RULES:
1. Always set timeouts (connection, read, write)
2. Implement circuit breakers
3. Use retries with exponential backoff
4. Design for partial failure
5. Consider caching for read-heavy patterns
```

#### REST API Design

```
REST API STANDARDS:

Resource Naming:
  ✅ CORRECT: /orders, /orders/{id}, /orders/{id}/items
  ❌ WRONG:   /getOrders, /order_list, /orderById

HTTP Methods:
  GET    → Read (idempotent, cacheable)
  POST   → Create (not idempotent)
  PUT    → Full update (idempotent)
  PATCH  → Partial update (not idempotent)
  DELETE → Remove (idempotent)

Status Codes:
  2xx → Success (200 OK, 201 Created, 204 No Content)
  4xx → Client error (400 Bad Request, 404 Not Found, 422 Unprocessable)
  5xx → Server error (500 Internal, 503 Service Unavailable)

Response Format:
  {
    "data": { ... },           // Response payload
    "meta": {                  // Metadata
      "requestId": "...",
      "timestamp": "..."
    },
    "errors": [ ... ]          // Error details (if any)
  }
```

#### gRPC Design

```
gRPC STANDARDS:

Service Definition:
  ✅ CORRECT: Define clear service contracts in .proto files
  ✅ CORRECT: Use streaming for large data or real-time updates
  ✅ CORRECT: Version your proto files

Message Design:
  ✅ CORRECT: Use well-defined message types
  ✅ CORRECT: Reserve field numbers for backward compatibility
  ❌ WRONG:   Use generic maps for everything

Error Handling:
  ✅ CORRECT: Use standard gRPC status codes
  ✅ CORRECT: Include error details in metadata
  ❌ WRONG:   Return errors in response body with OK status
```

### B. Asynchronous Communication

**Use asynchronous communication for decoupling and resilience.**

```
ASYNCHRONOUS PATTERNS:

┌─────────────────────────────────────────────────────────────┐
│                    EVENT-DRIVEN                              │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Service A ──────▶ Message Broker ──────▶ Service B         │
│            (publish)             (subscribe)                 │
│                          │                                   │
│                          └──────▶ Service C                  │
│                               (subscribe)                    │
│                                                              │
│  Patterns: Pub/Sub, Event Sourcing, CQRS                    │
│  Use when: Decoupling, scalability, eventual consistency    │
│  Technologies: Kafka, RabbitMQ, AWS SNS/SQS, NATS          │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                    MESSAGE QUEUE                             │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Service A ──────▶ Queue ──────▶ Service B                  │
│            (send)         (receive)                          │
│                                                              │
│  Pattern: Point-to-point, work queue                        │
│  Use when: Task distribution, load leveling                 │
│  Guarantee: Each message processed by exactly one consumer  │
└─────────────────────────────────────────────────────────────┘
```

#### Event Design

```
EVENT STANDARDS:

Event Structure:
{
  "eventId": "uuid",              // Unique event identifier
  "eventType": "OrderPlaced",     // Event type (past tense verb)
  "aggregateType": "Order",       // Source aggregate
  "aggregateId": "order-123",     // Source aggregate ID
  "timestamp": "ISO-8601",        // When event occurred
  "version": 1,                   // Event schema version
  "correlationId": "uuid",        // Request correlation
  "causationId": "uuid",          // Causing event ID
  "data": {                       // Event payload
    "orderId": "order-123",
    "customerId": "customer-456",
    "totalAmount": 99.99
  },
  "metadata": {                   // Additional context
    "userId": "user-789",
    "source": "order-service"
  }
}

Event Naming:
  ✅ CORRECT: OrderPlaced, PaymentReceived, InventoryReserved (past tense)
  ❌ WRONG:   CreateOrder, ProcessPayment, ReserveInventory (imperative)

Event Types:
  1. Domain Events  → Business facts (OrderPlaced, CustomerRegistered)
  2. Integration Events → Cross-service communication
  3. System Events → Technical events (ServiceStarted, HealthCheckFailed)
```

### C. Communication Pattern Selection

```
PATTERN SELECTION MATRIX:

┌────────────────────┬──────────────┬───────────────┬─────────────────┐
│ Requirement        │ Sync (REST)  │ Sync (gRPC)   │ Async (Events)  │
├────────────────────┼──────────────┼───────────────┼─────────────────┤
│ Immediate response │ ✅           │ ✅            │ ❌              │
│ High throughput    │ ❌           │ ✅            │ ✅              │
│ Loose coupling     │ ❌           │ ❌            │ ✅              │
│ Guaranteed delivery│ ❌           │ ❌            │ ✅              │
│ Simple debugging   │ ✅           │ ✅            │ ❌              │
│ Streaming data     │ ❌           │ ✅            │ ✅              │
│ External clients   │ ✅           │ ❌            │ ❌              │
│ Real-time updates  │ ❌           │ ✅            │ ✅              │
└────────────────────┴──────────────┴───────────────┴─────────────────┘

DECISION GUIDE:
- Need response now? → Synchronous
- Fire and forget? → Asynchronous
- Multiple consumers? → Pub/Sub events
- Load leveling? → Message queue
- External API? → REST
- Internal high-perf? → gRPC
- Event sourcing? → Event stream
```

---

## 4. Data Management (MANDATORY)

### A. Database Per Service

**CRITICAL: Each service MUST own its data. No shared databases.**

```
DATABASE PER SERVICE PRINCIPLE:

✅ CORRECT Architecture:
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ Order       │    │ Payment     │    │ Inventory   │
│ Service     │    │ Service     │    │ Service     │
└──────┬──────┘    └──────┬──────┘    └──────┬──────┘
       │                  │                  │
       ▼                  ▼                  ▼
  ┌─────────┐        ┌─────────┐        ┌─────────┐
  │ Order   │        │ Payment │        │Inventory│
  │ DB      │        │ DB      │        │ DB      │
  └─────────┘        └─────────┘        └─────────┘

❌ WRONG Architecture:
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ Order       │    │ Payment     │    │ Inventory   │
│ Service     │    │ Service     │    │ Service     │
└──────┬──────┘    └──────┬──────┘    └──────┬──────┘
       │                  │                  │
       └──────────────────┼──────────────────┘
                          ▼
                    ┌───────────┐
                    │ Shared    │
                    │ Database  │  ← NEVER DO THIS
                    └───────────┘
```

### B. Data Consistency Patterns

```
CONSISTENCY PATTERNS:

1. SAGA Pattern (Choreography)
   ┌─────────┐  event   ┌─────────┐  event   ┌─────────┐
   │ Order   │ ──────▶  │ Payment │ ──────▶  │Inventory│
   │ Service │          │ Service │          │ Service │
   └─────────┘          └─────────┘          └─────────┘
        │                    │                    │
        │    ◀── compensating events ───         │
        │         (on failure)                   │

   Use when: Long-running transactions across services
   Pros: Loose coupling, resilient
   Cons: Complex compensation logic

2. SAGA Pattern (Orchestration)
   ┌─────────────────────────────────────────────────────┐
   │                  SAGA Orchestrator                   │
   │  (coordinates all steps and compensations)          │
   └───────────┬───────────────┬───────────────┬─────────┘
               │               │               │
               ▼               ▼               ▼
          ┌─────────┐    ┌─────────┐    ┌─────────┐
          │ Order   │    │ Payment │    │Inventory│
          └─────────┘    └─────────┘    └─────────┘

   Use when: Complex workflows needing central coordination
   Pros: Clear flow, easier to understand
   Cons: Single point of failure, coupling to orchestrator

3. Event Sourcing
   All changes stored as events, current state derived from event replay

   Use when: Need audit trail, temporal queries, CQRS
   Pros: Complete history, debugging, replay
   Cons: Complexity, eventual consistency

4. Transactional Outbox
   ┌─────────────────────────────────────────┐
   │ Service Database                         │
   │  ┌─────────────┐  ┌─────────────────┐   │
   │  │ Business    │  │ Outbox Table    │   │
   │  │ Tables      │  │ (pending events)│   │
   │  └─────────────┘  └────────┬────────┘   │
   └────────────────────────────┼────────────┘
                                │
                     ┌──────────▼──────────┐
                     │ Message Relay       │
                     │ (polls and publishes│
                     └──────────┬──────────┘
                                │
                                ▼
                        Message Broker

   Use when: Need reliable event publishing with transactions
   Pros: Atomic operations, guaranteed delivery
   Cons: Additional complexity
```

### C. Data Query Patterns

```
CROSS-SERVICE DATA PATTERNS:

1. API Composition
   ┌──────────────────────────────────────────────────────┐
   │                 API Gateway / BFF                     │
   │  (aggregates data from multiple services)            │
   └───────────┬───────────────┬───────────────┬──────────┘
               │               │               │
               ▼               ▼               ▼
          ┌─────────┐    ┌─────────┐    ┌─────────┐
          │ Order   │    │ Customer│    │ Product │
          └─────────┘    └─────────┘    └─────────┘

   Use when: Simple aggregation, few services
   Cons: Latency, complexity grows with services

2. CQRS (Command Query Responsibility Segregation)

   Commands ──▶ Write Model ──▶ Event Store
                                    │
                                    ▼
   Queries ◀── Read Model ◀── Projections

   Use when: Different read/write patterns, complex queries
   Pros: Optimized read models, scalability
   Cons: Eventual consistency, complexity

3. Data Replication (Read Replicas)

   Service A publishes events → Service B maintains read replica

   Use when: High-read scenarios, need local data copy
   Cons: Stale data, storage overhead
```

---

## 5. Resilience Patterns (MANDATORY)

### A. Circuit Breaker

**CRITICAL: Prevent cascade failures with circuit breakers.**

```
CIRCUIT BREAKER PATTERN:

States:
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│    ┌────────┐        failures       ┌────────┐             │
│    │ CLOSED │ ──────────────────▶   │  OPEN  │             │
│    │        │   (threshold reached) │        │             │
│    └────┬───┘                       └────┬───┘             │
│         │                                │                  │
│         │ success                        │ timeout          │
│         │                                │                  │
│         │           ┌──────────┐         │                  │
│         └───────────│HALF-OPEN │◀────────┘                  │
│                     │          │                            │
│                     └──────────┘                            │
│                      │        │                             │
│              success │        │ failure                     │
│                      ▼        ▼                             │
│                   CLOSED    OPEN                            │
│                                                             │
└─────────────────────────────────────────────────────────────┘

Configuration:
- Failure threshold: Number of failures before opening (e.g., 5)
- Success threshold: Successes in half-open to close (e.g., 3)
- Timeout: Time before attempting recovery (e.g., 30s)
- Window: Time window for counting failures (e.g., 60s)

CIRCUIT BREAKER RULES:
1. Apply to all external service calls
2. Configure thresholds based on service SLAs
3. Provide fallback responses when open
4. Monitor circuit state (metrics/alerts)
5. Log state transitions
```

### B. Retry with Backoff

```
RETRY PATTERN:

Exponential Backoff:
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  Request ──▶ Fail ──▶ Wait 1s ──▶ Retry                    │
│                                     │                       │
│                                    Fail                     │
│                                     │                       │
│                              Wait 2s ──▶ Retry              │
│                                           │                 │
│                                          Fail               │
│                                           │                 │
│                                    Wait 4s ──▶ Retry        │
│                                                 │           │
│                                                ...          │
│                                                 │           │
│                                          Max retries        │
│                                          exceeded           │
│                                                 │           │
│                                              Fail           │
│                                                             │
└─────────────────────────────────────────────────────────────┘

With Jitter (prevent thundering herd):
  delay = min(cap, base * 2^attempt) + random(0, 1000ms)

RETRY RULES:
1. Only retry idempotent operations (GET, PUT, DELETE)
2. Don't retry client errors (4xx except 429)
3. Do retry server errors (5xx) and timeouts
4. Add jitter to prevent thundering herd
5. Set maximum retry count
6. Use circuit breaker in conjunction
```

### C. Bulkhead

```
BULKHEAD PATTERN:

Isolate failures to prevent resource exhaustion:

┌─────────────────────────────────────────────────────────────┐
│                       Service                                │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐  │
│  │ Thread Pool A   │  │ Thread Pool B   │  │ Thread Pool │  │
│  │ (Order API)     │  │ (Payment API)   │  │ C (Inventory│  │
│  │ [10 threads]    │  │ [5 threads]     │  │ [5 threads] │  │
│  └────────┬────────┘  └────────┬────────┘  └──────┬──────┘  │
│           │                    │                   │         │
│           ▼                    ▼                   ▼         │
│      Order Service      Payment Service     Inventory Svc   │
│                                                              │
│  If Payment Service is slow/failing:                        │
│  - Only Thread Pool B exhausted                             │
│  - Order and Inventory calls unaffected                     │
└─────────────────────────────────────────────────────────────┘

BULKHEAD TYPES:
1. Thread pool isolation (separate pools per dependency)
2. Connection pool isolation (separate connections)
3. Semaphore isolation (limit concurrent calls)

BULKHEAD RULES:
1. Isolate calls to different external services
2. Size pools based on expected load and SLAs
3. Monitor pool utilization
4. Fail fast when pool exhausted
```

### D. Timeout

```
TIMEOUT CONFIGURATION:

Timeout Layers:
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  Client ──▶ [Gateway Timeout: 30s]                         │
│                    │                                        │
│                    ▼                                        │
│            API Gateway ──▶ [Service Timeout: 10s]          │
│                                │                            │
│                                ▼                            │
│                        Service A ──▶ [DB Timeout: 5s]      │
│                                          │                  │
│                                          ▼                  │
│                                      Database               │
│                                                             │
│  Rule: Upstream timeout > Downstream timeout                │
│        (Gateway > Service > Database)                       │
│                                                             │
└─────────────────────────────────────────────────────────────┘

TIMEOUT TYPES:
1. Connection timeout: Time to establish connection (1-5s)
2. Read timeout: Time to receive response (5-30s)
3. Write timeout: Time to send request (5-10s)
4. Idle timeout: Time before closing idle connection

TIMEOUT RULES:
1. Always set timeouts (never wait forever)
2. Upstream timeouts > downstream timeouts
3. Include buffer for retries in upstream timeout
4. Monitor timeout rates
5. Adjust based on P99 latency
```

### E. Fallback

```
FALLBACK STRATEGIES:

1. Default Value
   if (serviceCall.failed()) {
     return defaultValue;  // e.g., empty list, cached value
   }

2. Cached Response
   if (serviceCall.failed()) {
     return cache.get(key);  // Return stale data
   }

3. Graceful Degradation
   if (recommendationService.failed()) {
     return popularItems;  // Show popular instead of personalized
   }

4. Fail Silent
   if (analyticsService.failed()) {
     // Log and continue without analytics
     log.warn("Analytics unavailable");
   }

5. Alternative Service
   if (primaryPaymentGateway.failed()) {
     return backupPaymentGateway.process(payment);
   }

FALLBACK RULES:
1. Always have a fallback for critical paths
2. Fallback should be simpler/more reliable than primary
3. Monitor fallback usage
4. Test fallbacks regularly
5. Consider business impact of degraded responses
```

---

## 6. Observability (MANDATORY)

### A. The Three Pillars

**CRITICAL: All services MUST implement logging, metrics, and tracing.**

```
OBSERVABILITY PILLARS:

┌─────────────────────────────────────────────────────────────┐
│                                                             │
│     LOGS              METRICS            TRACES             │
│     (Events)          (Aggregates)       (Requests)         │
│                                                             │
│  ┌───────────┐      ┌───────────┐      ┌───────────┐       │
│  │ What      │      │ What      │      │ What      │       │
│  │ happened  │      │ is the    │      │ is the    │       │
│  │ (detail)  │      │ state     │      │ flow      │       │
│  └───────────┘      │ (numbers) │      │ (journey) │       │
│                     └───────────┘      └───────────┘       │
│                                                             │
│  Use for:           Use for:           Use for:            │
│  - Debugging        - Alerting         - Debugging         │
│  - Audit            - Dashboards       - Performance       │
│  - Forensics        - Capacity         - Dependencies      │
│                       planning                              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### B. Structured Logging

```
LOGGING STANDARDS:

Log Format (JSON):
{
  "timestamp": "2024-01-15T10:30:00.000Z",
  "level": "INFO",
  "service": "order-service",
  "instance": "order-service-abc123",
  "traceId": "abc123",
  "spanId": "def456",
  "userId": "user-789",
  "message": "Order placed successfully",
  "context": {
    "orderId": "order-123",
    "amount": 99.99,
    "items": 3
  }
}

Log Levels:
  ERROR → Something failed, needs attention
  WARN  → Unexpected but handled, potential issue
  INFO  → Business events, state changes
  DEBUG → Detailed diagnostic (not in production)
  TRACE → Very detailed (development only)

LOGGING RULES:
1. Use structured logging (JSON)
2. Include correlation IDs (traceId, spanId)
3. Include business context (orderId, userId)
4. Don't log sensitive data (passwords, tokens, PII)
5. Log at service boundaries (entry/exit)
6. Use appropriate log levels
7. Include timing information
```

### C. Metrics

```
METRICS STANDARDS:

Metric Types:
┌─────────────┬────────────────────────────────────────────────┐
│ Type        │ Use Case                                       │
├─────────────┼────────────────────────────────────────────────┤
│ Counter     │ Total requests, errors, events                 │
│ Gauge       │ Current connections, queue size, memory        │
│ Histogram   │ Request duration, response sizes               │
│ Summary     │ Quantiles (P50, P95, P99)                      │
└─────────────┴────────────────────────────────────────────────┘

Required Metrics (RED Method):
- Rate: Requests per second
- Errors: Error rate/count
- Duration: Request latency (P50, P95, P99)

Additional Metrics (USE Method for resources):
- Utilization: Percent of resource used
- Saturation: Queue depth, waiting
- Errors: Error count

Naming Convention:
  {service}_{component}_{metric}_{unit}

  Examples:
  - order_service_http_requests_total
  - order_service_http_request_duration_seconds
  - order_service_db_connections_active
  - order_service_queue_messages_pending

METRICS RULES:
1. Use standard naming conventions
2. Include labels (method, status, endpoint)
3. Export RED metrics for all services
4. Set up dashboards for key metrics
5. Configure alerts for anomalies
```

### D. Distributed Tracing

```
DISTRIBUTED TRACING:

Trace Structure:
┌─────────────────────────────────────────────────────────────┐
│ Trace ID: abc123                                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│ ├── Span: API Gateway (span-1)         [0ms────────100ms]  │
│ │   └── Span: Order Service (span-2)   [10ms───────90ms]   │
│ │       ├── Span: DB Query (span-3)    [20ms──40ms]        │
│ │       └── Span: Payment Svc (span-4) [50ms─────80ms]     │
│ │           └── Span: Stripe API (5)   [55ms──75ms]        │
│                                                             │
└─────────────────────────────────────────────────────────────┘

Span Attributes:
{
  "traceId": "abc123",
  "spanId": "span-2",
  "parentSpanId": "span-1",
  "operationName": "POST /orders",
  "serviceName": "order-service",
  "startTime": "...",
  "duration": 80,
  "status": "OK",
  "tags": {
    "http.method": "POST",
    "http.url": "/orders",
    "http.status_code": 201,
    "user.id": "user-789"
  }
}

TRACING RULES:
1. Propagate trace context across all calls
2. Create spans for all external calls (HTTP, DB, Queue)
3. Add meaningful tags to spans
4. Sample appropriately (100% in dev, 1-10% in prod)
5. Link async operations with trace context
```

### E. Health Checks

```
HEALTH CHECK ENDPOINTS:

Liveness Check: /health/live
  Purpose: Is the service running?
  Checks: Process is responsive
  Response: 200 OK or 503 Service Unavailable

  Used by: Container orchestrator to restart unhealthy containers

  {
    "status": "UP"
  }

Readiness Check: /health/ready
  Purpose: Can the service handle traffic?
  Checks: Dependencies available (DB, cache, downstream services)
  Response: 200 OK or 503 Service Unavailable

  Used by: Load balancer to route traffic

  {
    "status": "UP",
    "checks": {
      "database": { "status": "UP", "latency": "5ms" },
      "cache": { "status": "UP", "latency": "1ms" },
      "payment-service": { "status": "UP", "latency": "50ms" }
    }
  }

Startup Check: /health/startup
  Purpose: Has the service finished initializing?
  Checks: Migrations complete, caches warmed, connections established
  Response: 200 OK or 503 Service Unavailable

  Used by: Orchestrator to know when to start liveness/readiness checks

HEALTH CHECK RULES:
1. Liveness: Only check process health (fast, no dependencies)
2. Readiness: Check critical dependencies
3. Don't include non-critical dependencies in readiness
4. Set appropriate timeouts for health checks
5. Cache dependency health status (don't check every request)
```

---

## 7. API Gateway & Service Mesh (MANDATORY)

### A. API Gateway

**CRITICAL: Use an API gateway for external traffic.**

```
API GATEWAY RESPONSIBILITIES:

┌─────────────────────────────────────────────────────────────┐
│                       API Gateway                            │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  External     ┌──────────────────────────────┐              │
│  Clients ──▶  │ • Authentication/Authorization│              │
│               │ • Rate Limiting               │              │
│               │ • Request/Response Transform  │              │
│               │ • SSL Termination             │              │
│               │ • Load Balancing              │  ──▶ Services│
│               │ • Caching                     │              │
│               │ • Request Routing             │              │
│               │ • API Versioning              │              │
│               │ • Circuit Breaking            │              │
│               │ • Logging/Monitoring          │              │
│               └──────────────────────────────┘              │
│                                                              │
└─────────────────────────────────────────────────────────────┘

API GATEWAY RULES:
1. All external traffic goes through gateway
2. Perform authentication at gateway level
3. Rate limit by client/API key
4. Transform public API to internal protocols
5. Don't put business logic in gateway
6. Monitor gateway as critical infrastructure
```

### B. Service Mesh

```
SERVICE MESH (for internal traffic):

┌─────────────────────────────────────────────────────────────┐
│                                                              │
│  ┌─────────────┐         ┌─────────────┐                    │
│  │  Service A  │         │  Service B  │                    │
│  │  ┌───────┐  │ mTLS    │  ┌───────┐  │                    │
│  │  │ App   │  │  ───▶   │  │ App   │  │                    │
│  │  └───┬───┘  │         │  └───┬───┘  │                    │
│  │      │      │         │      │      │                    │
│  │  ┌───▼───┐  │         │  ┌───▼───┐  │                    │
│  │  │ Sidecar│  │◀───────▶│  │ Sidecar│  │                    │
│  │  │ Proxy │  │ Service │  │ Proxy │  │                    │
│  │  └───────┘  │  Mesh   │  └───────┘  │                    │
│  └─────────────┘ Traffic └─────────────┘                    │
│                                                              │
│  Service Mesh Provides:                                      │
│  • mTLS (mutual TLS)                                        │
│  • Load balancing                                           │
│  • Circuit breaking                                         │
│  • Retries                                                  │
│  • Observability                                            │
│  • Traffic management                                       │
│  • Access control                                           │
│                                                              │
└─────────────────────────────────────────────────────────────┘

SERVICE MESH RULES:
1. Use for service-to-service communication
2. Offload cross-cutting concerns from services
3. Implement mTLS for zero-trust security
4. Use for traffic shifting (canary, blue-green)
5. Monitor mesh control plane
```

---

## 8. Security (MANDATORY)

### A. Zero Trust Architecture

**CRITICAL: Never trust, always verify.**

```
ZERO TRUST PRINCIPLES:

1. Verify Explicitly
   - Authenticate every request
   - Validate all inputs
   - Check authorization at every layer

2. Least Privilege Access
   - Minimal permissions
   - Just-in-time access
   - Role-based access control (RBAC)

3. Assume Breach
   - Segment network
   - Encrypt all traffic
   - Log everything
   - Minimize blast radius

SECURITY LAYERS:
┌─────────────────────────────────────────────────────────────┐
│ Layer              │ Security Measures                      │
├─────────────────────────────────────────────────────────────┤
│ Network            │ • Firewall rules                       │
│                    │ • Network segmentation                 │
│                    │ • DDoS protection                      │
├─────────────────────────────────────────────────────────────┤
│ Transport          │ • TLS 1.3 everywhere                   │
│                    │ • mTLS between services                │
│                    │ • Certificate rotation                 │
├─────────────────────────────────────────────────────────────┤
│ Application        │ • Authentication (OAuth2, OIDC)        │
│                    │ • Authorization (RBAC, ABAC)           │
│                    │ • Input validation                     │
│                    │ • Output encoding                      │
├─────────────────────────────────────────────────────────────┤
│ Data               │ • Encryption at rest                   │
│                    │ • Encryption in transit                │
│                    │ • Data masking                         │
│                    │ • Access auditing                      │
└─────────────────────────────────────────────────────────────┘
```

### B. Authentication & Authorization

```
SERVICE-TO-SERVICE AUTHENTICATION:

1. Mutual TLS (mTLS)
   - Both client and server present certificates
   - Identity verified at transport layer
   - Managed by service mesh typically

2. JWT Tokens
   - Service identity in token claims
   - Short-lived tokens
   - Validated by receiving service

3. API Keys
   - For internal service identification
   - Rotate regularly
   - Different keys for different environments

AUTHORIZATION PATTERNS:

1. Centralized Authorization
   └── All services call central auth service
   └── Pros: Consistent, auditable
   └── Cons: Latency, single point of failure

2. Distributed Authorization
   └── Each service makes own decisions
   └── Policies distributed to services
   └── Pros: Performance, resilience
   └── Cons: Consistency challenges

3. Token-Based (JWT)
   └── Claims embedded in token
   └── Service validates token and claims
   └── Pros: No additional calls
   └── Cons: Token size, revocation complexity
```

### C. Secrets Management

```
SECRETS MANAGEMENT:

✅ CORRECT Practices:
- Use dedicated secrets manager (Vault, AWS Secrets Manager)
- Rotate secrets automatically
- Never commit secrets to version control
- Use different secrets per environment
- Audit secret access
- Encrypt secrets at rest and in transit

❌ PROHIBITED Practices:
- Hardcoded secrets in code
- Secrets in environment variables (plain text)
- Shared secrets across services
- Long-lived secrets without rotation
- Secrets in configuration files

SECRETS HIERARCHY:
┌─────────────────────────────────────────────────────────────┐
│                                                              │
│  Application ──▶ Secrets SDK ──▶ Secrets Manager           │
│                                        │                    │
│                              ┌─────────▼─────────┐          │
│                              │ Encrypted Storage │          │
│                              │ (KMS backed)      │          │
│                              └───────────────────┘          │
│                                                              │
│  Secret Types:                                               │
│  • Database credentials                                      │
│  • API keys                                                  │
│  • Encryption keys                                           │
│  • TLS certificates                                          │
│  • Service account tokens                                    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 9. Deployment Patterns (MANDATORY)

### A. Container-First Deployment

**CRITICAL: All services MUST be containerized.**

```
CONTAINERIZATION STANDARDS:

Image Requirements:
- Base image: Minimal (Alpine, Distroless)
- Non-root user
- Multi-stage builds
- Proper signal handling
- Health check endpoints

Image Tagging:
  ✅ CORRECT:
    - myservice:v1.2.3 (semantic version)
    - myservice:abc123 (git commit)
    - myservice:20240115-abc123 (date + commit)

  ❌ WRONG:
    - myservice:latest (mutable, not reproducible)
    - myservice:dev (ambiguous)

Container Configuration:
- Resource limits (CPU, memory)
- Liveness/readiness probes
- Graceful shutdown handling
- Environment-specific configuration via env vars
```

### B. Deployment Strategies

```
DEPLOYMENT STRATEGIES:

1. Rolling Deployment
   ┌─────────────────────────────────────────────────────────┐
   │ v1  v1  v1  v1  →  v2  v1  v1  v1  →  v2  v2  v1  v1   │
   │ →  v2  v2  v2  v1  →  v2  v2  v2  v2                   │
   └─────────────────────────────────────────────────────────┘
   Use: Standard deployments
   Pros: Zero downtime, gradual rollout
   Cons: Multiple versions running simultaneously

2. Blue-Green Deployment
   ┌─────────────────────────────────────────────────────────┐
   │                    Load Balancer                         │
   │                         │                                │
   │           ┌─────────────┴─────────────┐                  │
   │           ▼                           ▼                  │
   │     ┌──────────┐               ┌──────────┐             │
   │     │  Blue    │               │  Green   │             │
   │     │  (v1)    │  ◀── switch ──│  (v2)    │             │
   │     │  ACTIVE  │               │  STANDBY │             │
   │     └──────────┘               └──────────┘             │
   └─────────────────────────────────────────────────────────┘
   Use: Critical services, need instant rollback
   Pros: Instant switch, easy rollback
   Cons: Double infrastructure cost

3. Canary Deployment
   ┌─────────────────────────────────────────────────────────┐
   │                    Load Balancer                         │
   │                         │                                │
   │           ┌────────────────────────────┐                 │
   │           │ 95%                    5%  │                 │
   │           ▼                        ▼   │                 │
   │     ┌──────────┐            ┌──────────┐                │
   │     │  Stable  │            │  Canary  │                │
   │     │  (v1)    │            │  (v2)    │                │
   │     └──────────┘            └──────────┘                │
   └─────────────────────────────────────────────────────────┘
   Use: Validating new versions with real traffic
   Pros: Low risk, real-world validation
   Cons: Complex traffic management

4. Feature Flags
   ┌─────────────────────────────────────────────────────────┐
   │                                                          │
   │  if (featureFlag.isEnabled("new-checkout")) {           │
   │    return newCheckoutFlow();                             │
   │  } else {                                                │
   │    return oldCheckoutFlow();                             │
   │  }                                                       │
   │                                                          │
   └─────────────────────────────────────────────────────────┘
   Use: Gradual feature rollout, A/B testing
   Pros: Decouple deploy from release, instant toggle
   Cons: Code complexity, technical debt
```

### C. GitOps Workflow

```
GITOPS PRINCIPLES:

┌─────────────────────────────────────────────────────────────┐
│                                                              │
│  Developer ──▶ Git Repo ──▶ CI Pipeline ──▶ Container       │
│     │           (code)       (build/test)   Registry        │
│     │                                          │            │
│     │                                          ▼            │
│     └──▶ Config Repo ──▶ GitOps Operator ──▶ Kubernetes    │
│           (manifests)      (Argo/Flux)                      │
│                                                              │
│  Principles:                                                 │
│  1. Git is the single source of truth                       │
│  2. Declarative desired state                               │
│  3. Automated synchronization                               │
│  4. All changes through pull requests                       │
│                                                              │
└─────────────────────────────────────────────────────────────┘

GITOPS RULES:
1. Infrastructure as code in Git
2. No manual changes to production
3. Pull request for all changes
4. Automated drift detection
5. Audit trail via Git history
```

---

## 10. Testing Strategy (MANDATORY)

### A. Test Pyramid for Microservices

```
MICROSERVICES TEST PYRAMID:

                      ┌───────┐
                     /   E2E   \              Few
                    /   Tests   \             (10%)
                   /─────────────\
                  /  Contract     \           Some
                 /   Tests         \          (20%)
                /───────────────────\
               /   Integration       \        Some
              /     Tests             \       (20%)
             /─────────────────────────\
            /       Component           \     Some
           /         Tests               \    (20%)
          /───────────────────────────────\
         /           Unit                  \  Many
        /            Tests                  \ (30%)
       /─────────────────────────────────────\
```

### B. Test Types

```
UNIT TESTS:
  Scope: Single class/function
  Dependencies: Mocked
  Speed: Milliseconds
  Coverage: 80%+ of code

COMPONENT TESTS:
  Scope: Single service in isolation
  Dependencies: Mocked external services, real DB (test container)
  Speed: Seconds
  Coverage: Service API and behavior

INTEGRATION TESTS:
  Scope: Service with real dependencies
  Dependencies: Real databases, real message brokers (test instances)
  Speed: Seconds to minutes
  Coverage: Data access, external integrations

CONTRACT TESTS:
  Scope: API contracts between services
  Types:
    - Consumer-driven contracts (Pact)
    - Provider verification
  Speed: Seconds
  Coverage: API compatibility

E2E TESTS:
  Scope: Complete user journeys across services
  Environment: Production-like
  Speed: Minutes
  Coverage: Critical business flows only
```

### C. Contract Testing

```
CONTRACT TESTING FLOW:

Consumer-Driven Contracts:
┌─────────────────────────────────────────────────────────────┐
│                                                              │
│  1. Consumer defines expected interactions (contract)        │
│                                                              │
│     Consumer ──▶ "I expect GET /users/123 returns {name}"   │
│                                                              │
│  2. Contract published to broker                             │
│                                                              │
│     Contract ──▶ Contract Broker                            │
│                                                              │
│  3. Provider verifies it meets contracts                     │
│                                                              │
│     Provider ◀── Contract Broker                            │
│        │                                                     │
│        └── Runs contract tests against provider             │
│                                                              │
│  4. Deployment only if contracts satisfied                   │
│                                                              │
│     ✅ Contracts pass → Deploy allowed                       │
│     ❌ Contracts fail → Deploy blocked                       │
│                                                              │
└─────────────────────────────────────────────────────────────┘

CONTRACT TESTING RULES:
1. Consumers define contracts
2. Contracts version-controlled
3. Provider CI verifies contracts
4. Breaking changes detected before deployment
5. Contracts are part of definition of done
```

### D. Chaos Engineering

```
CHAOS ENGINEERING PRINCIPLES:

1. Define Steady State
   └── What does "normal" look like? (latency, error rate, throughput)

2. Hypothesize
   └── "System will handle 50% of order-service instances failing"

3. Introduce Chaos
   └── Kill instances, inject latency, corrupt network

4. Observe
   └── Did system maintain steady state?

5. Learn
   └── Fix weaknesses, improve resilience

CHAOS EXPERIMENTS:
┌──────────────────────────┬──────────────────────────────────┐
│ Experiment               │ Tests                             │
├──────────────────────────┼──────────────────────────────────┤
│ Kill service instance    │ Auto-scaling, load balancing     │
│ Increase latency         │ Timeouts, circuit breakers       │
│ Network partition        │ Fallbacks, graceful degradation  │
│ DNS failure              │ Caching, retry logic             │
│ Database failure         │ Failover, read replicas          │
│ Disk full                │ Alerting, auto-remediation       │
│ CPU/Memory exhaustion    │ Resource limits, auto-scaling    │
└──────────────────────────┴──────────────────────────────────┘

CHAOS RULES:
1. Start in non-production
2. Start small (single instance)
3. Have rollback plan
4. Monitor during experiments
5. Document learnings
```

---

## 11. Common Anti-Patterns (PROHIBITED)

### A. Distributed Monolith

```
❌ PROHIBITED: Distributed Monolith

Symptoms:
- Services must deploy together
- Shared database between services
- Synchronous chains of calls
- Tight coupling via shared libraries
- Changes require coordination

┌─────────────────────────────────────────────────────────────┐
│                                                              │
│  Service A ──sync──▶ Service B ──sync──▶ Service C          │
│      │                   │                   │               │
│      └───────────────────┼───────────────────┘               │
│                          ▼                                   │
│                   Shared Database  ← WRONG                  │
│                                                              │
└─────────────────────────────────────────────────────────────┘

✅ CORRECT: True Microservices

- Independent deployments
- Database per service
- Async communication where possible
- Loose coupling
- Teams work independently
```

### B. Chatty Services

```
❌ PROHIBITED: Chatty Communication

Service A makes many calls to Service B for single operation:

  A ──▶ B.getUserName(userId)
  A ──▶ B.getUserEmail(userId)
  A ──▶ B.getUserPhone(userId)
  A ──▶ B.getUserAddress(userId)

  Result: 4 network round trips, high latency

✅ CORRECT: Coarse-Grained APIs

  A ──▶ B.getUser(userId)

  Result: 1 network round trip, returns complete user

RULES:
1. Design APIs for common use cases
2. Batch operations where possible
3. Consider BFF (Backend for Frontend) pattern
4. Cache frequently accessed data
```

### C. Shared Database

```
❌ PROHIBITED: Shared Database

Services directly accessing another service's tables:

  Order Service ──▶ ┌──────────────────┐
                    │                  │
  User Service  ──▶ │ Shared Database  │
                    │                  │
  Payment Service ──▶└──────────────────┘

Problems:
- Tight coupling
- Schema changes affect all services
- No clear data ownership
- Cannot scale independently

✅ CORRECT: Database Per Service

  Order Service ──▶ Order DB
  User Service  ──▶ User DB
  Payment Service ──▶ Payment DB

Data sharing via:
- APIs
- Events
- Data replication (if needed)
```

### D. Synchronous Chains

```
❌ PROHIBITED: Long Synchronous Chains

Client ──▶ A ──▶ B ──▶ C ──▶ D ──▶ E

Problems:
- Latency compounds (sum of all services)
- Reliability decreases (product of uptimes)
- Single point of failure anywhere breaks chain
- Difficult to scale

If each service: 100ms latency, 99.9% uptime
  Chain latency: 500ms
  Chain uptime: 99.5% (0.999^5)

✅ CORRECT: Minimize Synchronous Depth

1. Async where possible:
   Client ──▶ A ──event──▶ B ──event──▶ C

2. Parallel calls:
   Client ──▶ A ──┬──▶ B
                  └──▶ C

3. Aggregate at edge:
   Client ──▶ BFF ──┬──▶ Service 1
                    ├──▶ Service 2
                    └──▶ Service 3
```

### E. Hardcoded Service Locations

```
❌ PROHIBITED: Hardcoded URLs

config:
  user_service_url: "http://192.168.1.50:8080"
  payment_service_url: "http://192.168.1.51:8080"

Problems:
- Cannot scale dynamically
- No failover
- Environment-specific configuration
- Manual updates required

✅ CORRECT: Service Discovery

config:
  user_service: "user-service"      # Service name
  payment_service: "payment-service"

Resolution via:
- DNS-based discovery (Kubernetes)
- Service registry (Consul, Eureka)
- Service mesh (Istio, Linkerd)
```

---

## 12. Verification Checklist (MANDATORY)

### A. Architecture Verification Protocol

**CRITICAL: Verify architecture before delivery.**

```
VERIFICATION CHECKLIST:

□ Service Boundaries
  □ Services align with business capabilities
  □ Each service has single responsibility
  □ Services can be deployed independently
  □ Teams can work on services independently

□ Data Management
  □ Each service owns its data
  □ No shared databases
  □ Data consistency patterns defined
  □ Event schemas versioned

□ Communication
  □ Sync vs async patterns chosen appropriately
  □ API contracts defined (OpenAPI, Proto)
  □ Events documented with schemas
  □ Circuit breakers implemented

□ Resilience
  □ Timeouts configured for all calls
  □ Retries with backoff implemented
  □ Fallbacks defined for critical paths
  □ Bulkheads isolate failures

□ Observability
  □ Structured logging implemented
  □ Metrics exported (RED method)
  □ Distributed tracing enabled
  □ Health endpoints implemented

□ Security
  □ Authentication at gateway
  □ Service-to-service auth (mTLS/JWT)
  □ Secrets in secrets manager
  □ No sensitive data logged

□ Deployment
  □ Services containerized
  □ Infrastructure as code
  □ CI/CD pipelines defined
  □ Deployment strategy chosen

□ Testing
  □ Unit tests (80%+ coverage)
  □ Contract tests for APIs
  □ Integration tests for adapters
  □ E2E tests for critical flows
```

### B. Service Readiness Checklist

```
BEFORE A SERVICE GOES TO PRODUCTION:

□ Functionality
  □ All acceptance criteria met
  □ API documented (OpenAPI/AsyncAPI)
  □ Error responses standardized

□ Resilience
  □ Circuit breakers configured
  □ Timeouts set
  □ Retry policies defined
  □ Fallbacks implemented
  □ Graceful shutdown handling

□ Observability
  □ Logs structured and correlated
  □ Metrics exposed
  □ Tracing enabled
  □ Dashboards created
  □ Alerts configured

□ Security
  □ Authentication implemented
  □ Authorization rules defined
  □ Input validation
  □ Secrets externalized
  □ Security scan passed

□ Operations
  □ Health checks implemented
  □ Runbook documented
  □ Capacity planning done
  □ Disaster recovery tested
  □ On-call rotation defined
```

---

## 13. Summary

### Core Principles

1. **Service autonomy**: Services are independently deployable and scalable
2. **Data ownership**: Each service owns its data, no shared databases
3. **Resilience first**: Design for failure with circuit breakers, retries, fallbacks
4. **Observability built-in**: Logging, metrics, and tracing from day one
5. **Security at every layer**: Zero trust, encrypt everything, authenticate always

### Key Patterns

| Category | Patterns |
|----------|----------|
| Communication | REST, gRPC, Events, Message Queues |
| Data | Database per service, SAGA, Event Sourcing, CQRS |
| Resilience | Circuit Breaker, Retry, Bulkhead, Timeout, Fallback |
| Deployment | Rolling, Blue-Green, Canary, Feature Flags |
| Testing | Unit, Component, Integration, Contract, E2E, Chaos |

### Remember

> "Microservices are not about the size of the service, but about the scope of its responsibility and its ability to change independently."

> "If you can't deploy your service independently, you don't have microservices - you have a distributed monolith."

> "Design for failure. Embrace eventual consistency. Monitor everything."

---

## Related Guides

- **[kubernetes.md](kubernetes.md)**: Kubernetes deployment and orchestration for microservices
- **[istio.md](istio.md)**: Istio service mesh for microservices communication and security
- **[kafka.md](kafka.md)**: Apache Kafka for event-driven microservices communication
- **[hexagonal.md](hexagonal.md)**: Hexagonal Architecture - structuring individual microservices
- **[cleanarch.md](cleanarch.md)**: Clean Architecture - internal structure for microservices
- **[rest.md](rest.md)**: REST API Design - designing synchronous microservice APIs
- **[tdd.md](tdd.md)**: Test-Driven Development - testing microservices
