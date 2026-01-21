# Software Design Patterns Reference Guide

This document provides a comprehensive overview of essential software design patterns, their modern implementations, comparisons, and guidance for selecting the right pattern. This guide is language-agnostic with examples in pseudocode.

---

**Agent Profile**: The Design Pattern Expert
**Role**: Senior Software Engineer & Pattern Specialist
**Objective**: Apply appropriate design patterns to solve common software problems while avoiding over-engineering and pattern abuse.
**Tools**: SOLID principles, Gang of Four patterns, modern functional patterns, domain-driven design patterns.

---

## 1. Core Philosophies: PATTERNS-FIRST

The agent must adhere to the **PATTERNS-FIRST** principles for every pattern decision:

- **P**roblem-Driven: Apply patterns to solve real problems, not for their own sake
- **A**ppropriate Scope: Use the simplest pattern that solves the problem
- **T**estability: Patterns should enhance, not hinder, testability
- **T**ransparency: Code should be readable; patterns shouldn't obscure intent
- **E**volution: Prefer patterns that allow code to evolve
- **R**efactor To: Introduce patterns through refactoring when needed, not upfront
- **N**aming Matters: Use pattern names in code when they clarify intent
- **S**implicity First: Sometimes no pattern is the best pattern

**Key Principle:**

> "A pattern is a solution to a problem in a context. If you don't have the problem, you don't need the solution."

---

## 2. Pattern Categories Overview

```
DESIGN PATTERN TAXONOMY:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  CREATIONAL              STRUCTURAL             BEHAVIORAL              │
│  (Object Creation)       (Composition)          (Communication)         │
│                                                                         │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐     │
│  │ • Factory       │    │ • Adapter       │    │ • Strategy      │     │
│  │ • Builder       │    │ • Decorator     │    │ • Observer      │     │
│  │ • Singleton     │    │ • Facade        │    │ • Command       │     │
│  │ • Prototype     │    │ • Composite     │    │ • State         │     │
│  │                 │    │ • Proxy         │    │ • Template      │     │
│  └─────────────────┘    └─────────────────┘    │ • Chain of Resp │     │
│                                                 └─────────────────┘     │
│                                                                         │
│  PRESENTATION            ARCHITECTURAL          FUNCTIONAL              │
│  (UI Patterns)           (Application)          (Modern FP)             │
│                                                                         │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐     │
│  │ • MVC           │    │ • Repository    │    │ • Result/Either │     │
│  │ • MVP           │    │ • Unit of Work  │    │ • Option/Maybe  │     │
│  │ • MVVM          │    │ • Specification │    │ • Pipe/Compose  │     │
│  │ • MVI           │    │ • CQRS          │    │ • Monad         │     │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘     │
│                                                                         │
│  FOUNDATIONAL TECHNIQUES (Enable all patterns above)                    │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ • Dependency Injection (DI)    • Inversion of Control (IoC)     │   │
│  │ • Dependency Inversion (DIP)   • Composition Root               │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Foundational Techniques

These are not classic "patterns" but fundamental techniques and principles that enable good design and make patterns work effectively.

### A. Dependency Inversion Principle (DIP)

```
DEPENDENCY INVERSION PRINCIPLE:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  The "D" in SOLID - A principle, not a pattern                          │
│                                                                         │
│  DEFINITION:                                                            │
│    1. High-level modules should not depend on low-level modules.        │
│       Both should depend on abstractions.                               │
│    2. Abstractions should not depend on details.                        │
│       Details should depend on abstractions.                            │
│                                                                         │
│  WITHOUT DIP:                        WITH DIP:                          │
│                                                                         │
│  ┌─────────────────┐                ┌─────────────────┐                │
│  │  OrderService   │                │  OrderService   │                │
│  │  (high-level)   │                │  (high-level)   │                │
│  └────────┬────────┘                └────────┬────────┘                │
│           │ depends on                       │ depends on              │
│           ▼                                  ▼                         │
│  ┌─────────────────┐                ┌─────────────────┐                │
│  │ PostgresRepo    │                │ <<interface>>   │                │
│  │ (low-level)     │                │ OrderRepository │                │
│  └─────────────────┘                └────────▲────────┘                │
│                                              │ implements              │
│  High-level depends on                ┌──────┴──────┐                  │
│  low-level concrete class             │             │                  │
│  ❌ Tight coupling              ┌─────┴─────┐ ┌─────┴─────┐            │
│                                 │PostgresRepo│ │MongoRepo  │            │
│                                 └───────────┘ └───────────┘            │
│                                                                         │
│                                 Both depend on abstraction             │
│                                 ✅ Loose coupling                      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

WHY "INVERSION"?
  Traditional: High-level → Low-level (top-down dependency)
  Inverted:    Both → Abstraction (dependency points to abstraction)

  The low-level module's dependency is "inverted" - instead of
  high-level depending on it, IT depends on an interface defined
  by the high-level module's needs.
```

### B. Inversion of Control (IoC)

```
INVERSION OF CONTROL:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  A paradigm where control flow is inverted                              │
│  "Don't call us, we'll call you" (Hollywood Principle)                  │
│                                                                         │
│  TRADITIONAL CONTROL:               INVERTED CONTROL:                   │
│                                                                         │
│  // Your code controls flow         // Framework controls flow          │
│  main() {                           // You provide hooks/callbacks      │
│    data = readInput()                                                   │
│    result = process(data)           @Controller                         │
│    writeOutput(result)              class OrderController {             │
│  }                                    @Get("/orders/:id")               │
│                                       getOrder(id) {                    │
│  You call the libraries               return orderService.get(id)       │
│                                       }                                 │
│                                     }                                   │
│                                                                         │
│                                     Framework calls YOUR code           │
│                                                                         │
│  FORMS OF IoC:                                                          │
│                                                                         │
│  1. Dependency Injection - Dependencies provided to you                 │
│  2. Event-driven/Callbacks - You register, framework calls              │
│  3. Template Method - Framework calls your overridden methods           │
│  4. Service Locator - You ask container for dependencies                │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### C. Dependency Injection (DI)

```
DEPENDENCY INJECTION:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  A TECHNIQUE to achieve IoC and implement DIP                           │
│  "Don't create your dependencies, receive them"                         │
│                                                                         │
│  WITHOUT DI:                        WITH DI:                            │
│                                                                         │
│  class OrderService {               class OrderService {                │
│    constructor() {                    constructor(                      │
│      // Creates own dependencies        repo: OrderRepository,          │
│      this.repo = new PostgresRepo()     email: EmailService,            │
│      this.email = new SmtpEmail()       logger: Logger                  │
│      this.logger = new FileLogger()   ) {                               │
│    }                                    this.repo = repo                │
│  }                                      this.email = email              │
│                                         this.logger = logger            │
│  ❌ Hard to test                      }                                 │
│  ❌ Hard to change implementations  }                                   │
│  ❌ Hidden dependencies                                                 │
│                                     ✅ Easy to test (inject mocks)      │
│                                     ✅ Easy to change (inject different)│
│                                     ✅ Explicit dependencies            │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

THREE TYPES OF INJECTION:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  1. CONSTRUCTOR INJECTION (Preferred)                                   │
│  ─────────────────────────────────────                                  │
│                                                                         │
│  class OrderService {                                                   │
│      constructor(                                                       │
│          private repo: OrderRepository,    // Required dependency       │
│          private email: EmailService       // Required dependency       │
│      ) {}                                                               │
│  }                                                                      │
│                                                                         │
│  ✅ Dependencies are explicit and required                              │
│  ✅ Object is fully initialized after construction                      │
│  ✅ Immutable dependencies (can be readonly)                            │
│  ✅ Easy to see all dependencies at a glance                            │
│                                                                         │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                         │
│  2. SETTER/PROPERTY INJECTION (For optional dependencies)               │
│  ─────────────────────────────────────────────────────────              │
│                                                                         │
│  class OrderService {                                                   │
│      private logger?: Logger                                            │
│                                                                         │
│      setLogger(logger: Logger) {      // Optional dependency            │
│          this.logger = logger                                           │
│      }                                                                  │
│  }                                                                      │
│                                                                         │
│  ⚠️  Use sparingly - only for truly optional dependencies               │
│  ❌ Object may be in incomplete state                                   │
│  ❌ Dependency can be changed after construction                        │
│                                                                         │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                         │
│  3. INTERFACE INJECTION (Rare)                                          │
│  ─────────────────────────────                                          │
│                                                                         │
│  interface LoggerAware {                                                │
│      setLogger(logger: Logger): void                                    │
│  }                                                                      │
│                                                                         │
│  class OrderService implements LoggerAware {                            │
│      setLogger(logger: Logger) { ... }                                  │
│  }                                                                      │
│                                                                         │
│  ⚠️  Rarely used - adds complexity without much benefit                 │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

RECOMMENDATION: Use Constructor Injection by default.
                Use Setter Injection only for optional dependencies.
```

### D. DI Containers (IoC Containers)

```
DEPENDENCY INJECTION CONTAINERS:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  A container/framework that automates dependency injection              │
│                                                                         │
│  MANUAL DI (Pure DI):               WITH DI CONTAINER:                  │
│                                                                         │
│  // You wire everything             // Container wires for you          │
│  const logger = new Logger()        container.register(Logger)          │
│  const config = new Config()        container.register(Config)          │
│  const db = new Database(config)    container.register(Database)        │
│  const repo = new UserRepo(db)      container.register(UserRepository)  │
│  const email = new EmailSvc(config) container.register(EmailService)    │
│  const userSvc = new UserService(   container.register(UserService)     │
│    repo, email, logger                                                  │
│  )                                  // Container resolves dependencies  │
│                                     const userSvc = container           │
│  Tedious for large apps               .resolve(UserService)             │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

DI CONTAINER FEATURES:

1. REGISTRATION (Tell container what exists)
   container.register(OrderRepository, PostgresOrderRepository)
   container.register(EmailService, SmtpEmailService)

2. RESOLUTION (Container creates with dependencies)
   const service = container.resolve(OrderService)
   // Container sees OrderService needs OrderRepository and EmailService
   // Creates PostgresOrderRepository, SmtpEmailService, injects them

3. LIFETIME MANAGEMENT
   container.registerSingleton(DatabaseConnection)  // One instance
   container.registerTransient(RequestHandler)      // New each time
   container.registerScoped(UserContext)            // One per scope/request

EXAMPLE - Modern DI Container:

// Registration (typically at app startup)
@injectable()
class PostgresOrderRepository implements OrderRepository {
    constructor(private db: DatabaseConnection) {}
    // ...
}

@injectable()
class OrderService {
    constructor(
        private repo: OrderRepository,
        private email: EmailService
    ) {}
}

// Configuration
container.register(OrderRepository, PostgresOrderRepository)
container.register(EmailService, SmtpEmailService)
container.registerSingleton(DatabaseConnection)

// Resolution (container figures out the graph)
const orderService = container.resolve(OrderService)

// What container does internally:
// 1. OrderService needs OrderRepository and EmailService
// 2. OrderRepository → PostgresOrderRepository needs DatabaseConnection
// 3. DatabaseConnection is singleton, create once or reuse
// 4. Create PostgresOrderRepository with DatabaseConnection
// 5. Create SmtpEmailService
// 6. Create OrderService with both dependencies

POPULAR DI CONTAINERS:
  • TypeScript/JS: InversifyJS, tsyringe, NestJS DI
  • Java: Spring, Guice, Dagger
  • C#: Microsoft.Extensions.DI, Autofac, Ninject
  • Python: dependency-injector, inject
  • Go: wire, dig, fx
```

### E. Composition Root

```
COMPOSITION ROOT:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  The ONE place where all dependencies are wired together                │
│  (As close to application entry point as possible)                      │
│                                                                         │
│  APPLICATION STRUCTURE:                                                 │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                         main() / startup                         │   │
│  │  ┌───────────────────────────────────────────────────────────┐  │   │
│  │  │              COMPOSITION ROOT                              │  │   │
│  │  │                                                            │  │   │
│  │  │  // All wiring happens HERE and ONLY here                  │  │   │
│  │  │  const config = loadConfig()                               │  │   │
│  │  │  const db = new DatabaseConnection(config.db)              │  │   │
│  │  │  const orderRepo = new PostgresOrderRepo(db)               │  │   │
│  │  │  const emailService = new SmtpEmailService(config.smtp)    │  │   │
│  │  │  const orderService = new OrderService(orderRepo, email)   │  │   │
│  │  │  const orderController = new OrderController(orderService) │  │   │
│  │  │                                                            │  │   │
│  │  │  // Or with DI container:                                  │  │   │
│  │  │  container.register(...)                                   │  │   │
│  │  │  container.register(...)                                   │  │   │
│  │  │                                                            │  │   │
│  │  └───────────────────────────────────────────────────────────┘  │   │
│  │                              │                                   │   │
│  │                              ▼                                   │   │
│  │                    Start Application                             │   │
│  │                    (everything is wired)                         │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  REST OF APPLICATION: No "new" for services, only receive dependencies │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

EXAMPLE - Express.js Application:

// composition-root.ts (or main.ts)
function createApp(): Express {
    // === COMPOSITION ROOT ===

    // Infrastructure
    const config = loadConfig()
    const db = new PostgresConnection(config.database)
    const cache = new RedisCache(config.redis)

    // Repositories
    const userRepo = new PostgresUserRepository(db)
    const orderRepo = new PostgresOrderRepository(db)

    // External Services
    const emailService = new SendGridEmailService(config.sendgrid)
    const paymentGateway = new StripePaymentGateway(config.stripe)

    // Application Services
    const userService = new UserService(userRepo, emailService)
    const orderService = new OrderService(orderRepo, paymentGateway, emailService)

    // Controllers
    const userController = new UserController(userService)
    const orderController = new OrderController(orderService)

    // Express App
    const app = express()
    app.use('/users', userController.router)
    app.use('/orders', orderController.router)

    return app
}

// index.ts
const app = createApp()
app.listen(3000)

RULES:
  ✅ All object construction in composition root
  ✅ Rest of code receives dependencies via constructors
  ❌ No "new ServiceClass()" scattered through codebase
  ❌ No service locator calls in business logic
```

### F. DI vs Service Locator

```
DEPENDENCY INJECTION vs SERVICE LOCATOR:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  Both achieve IoC, but DI is preferred                                  │
│                                                                         │
│  SERVICE LOCATOR (Anti-pattern):    DEPENDENCY INJECTION:               │
│                                                                         │
│  class OrderService {               class OrderService {                │
│    doSomething() {                    constructor(                      │
│      // ASK for dependency              private repo: OrderRepository,  │
│      const repo = ServiceLocator        private email: EmailService     │
│        .get(OrderRepository)          ) {}                              │
│      const email = ServiceLocator                                       │
│        .get(EmailService)             doSomething() {                   │
│                                         // Already HAVE dependencies    │
│      repo.save(...)                     this.repo.save(...)             │
│      email.send(...)                    this.email.send(...)            │
│    }                                  }                                 │
│  }                                  }                                   │
│                                                                         │
│  ❌ Hidden dependencies             ✅ Explicit dependencies            │
│  ❌ Global state                    ✅ No global state                  │
│  ❌ Hard to test                    ✅ Easy to test                     │
│  ❌ Runtime errors if missing       ✅ Compile-time safety              │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

WHY SERVICE LOCATOR IS AN ANTI-PATTERN:

1. HIDDEN DEPENDENCIES
   // What does this class need? Have to read ALL the code to find out.
   class OrderService {
       process() {
           const x = Locator.get(SomeService)      // Hidden!
           const y = Locator.get(AnotherService)   // Hidden!
           // ... 500 lines later ...
           const z = Locator.get(YetAnother)       // Hidden!
       }
   }

2. TESTING DIFFICULTY
   // Must configure global locator before testing
   beforeEach(() => {
       ServiceLocator.register(OrderRepository, mockRepo)
       ServiceLocator.register(EmailService, mockEmail)
       // Easy to forget one!
   })

3. WITH DI - Everything is explicit:
   // Constructor shows ALL dependencies immediately
   class OrderService {
       constructor(
           private repo: OrderRepository,      // Visible!
           private email: EmailService,        // Visible!
           private payment: PaymentGateway     // Visible!
       ) {}
   }

   // Testing - must provide all dependencies (compiler enforces)
   const service = new OrderService(mockRepo, mockEmail, mockPayment)

WHEN SERVICE LOCATOR IS ACCEPTABLE:
  • Legacy code migration (temporary)
  • Framework internals (hidden from user code)
  • Plugin systems with dynamic loading
```

### G. DI Best Practices

```
DEPENDENCY INJECTION BEST PRACTICES:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  ✅ DO                                                                  │
│  ────                                                                   │
│                                                                         │
│  1. Use constructor injection by default                                │
│     constructor(private repo: OrderRepository) {}                       │
│                                                                         │
│  2. Depend on abstractions (interfaces), not concretions                │
│     constructor(repo: OrderRepository)     // ✅ Interface              │
│     constructor(repo: PostgresOrderRepo)   // ❌ Concrete               │
│                                                                         │
│  3. Keep constructors simple (assignment only)                          │
│     constructor(private repo: OrderRepository) {                        │
│         this.repo = repo  // Just assign                                │
│         // NO complex logic, NO calls to other services                 │
│     }                                                                   │
│                                                                         │
│  4. Single composition root at application entry                        │
│                                                                         │
│  5. Make dependencies explicit and required                             │
│                                                                         │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                         │
│  ❌ DON'T                                                               │
│  ────────                                                               │
│                                                                         │
│  1. Use Service Locator in business logic                               │
│     const repo = Container.get(OrderRepository)  // ❌ Hidden           │
│                                                                         │
│  2. Inject the container itself                                         │
│     constructor(private container: Container)    // ❌ Service Locator  │
│                                                                         │
│  3. Create dependencies inside classes (except value objects)           │
│     this.repo = new PostgresOrderRepository()    // ❌ Tight coupling   │
│                                                                         │
│  4. Have too many dependencies (code smell)                             │
│     constructor(a, b, c, d, e, f, g, h, i, j)   // ❌ Class does too much│
│     // If > 3-4 dependencies, class may need splitting                  │
│                                                                         │
│  5. Inject dependencies you don't directly use                          │
│     constructor(repo: OrderRepository) {                                │
│         this.thing = repo.getThing()  // ❌ Just inject Thing directly  │
│     }                                                                   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

WHEN NOT TO USE DI:

  • Value objects and data classes
    new Money(100, "USD")           // ✅ OK to create directly
    new Email("user@example.com")   // ✅ OK to create directly

  • Simple utilities with no dependencies
    StringUtils.capitalize(str)     // ✅ Static utility is fine

  • Factories (they exist to create things)
    factory.create(type)            // ✅ Factory's job is to create
```

### H. The Complete Picture

```
HOW IT ALL FITS TOGETHER:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  SOLID PRINCIPLES                                                       │
│       │                                                                 │
│       │ includes                                                        │
│       ▼                                                                 │
│  DEPENDENCY INVERSION PRINCIPLE (DIP)                                   │
│  "Depend on abstractions, not concretions"                              │
│       │                                                                 │
│       │ leads to                                                        │
│       ▼                                                                 │
│  INVERSION OF CONTROL (IoC)                                             │
│  "Don't call us, we'll call you"                                        │
│       │                                                                 │
│       │ implemented via                                                 │
│       ▼                                                                 │
│  DEPENDENCY INJECTION (DI)                                              │
│  "Receive dependencies, don't create them"                              │
│       │                                                                 │
│       ├─────────────────────────────┐                                   │
│       │                             │                                   │
│       ▼                             ▼                                   │
│  PURE/MANUAL DI              DI CONTAINER                               │
│  (Wire by hand)              (Automated wiring)                         │
│       │                             │                                   │
│       └──────────────┬──────────────┘                                   │
│                      │                                                  │
│                      ▼                                                  │
│              COMPOSITION ROOT                                           │
│              (Single wiring location)                                   │
│                      │                                                  │
│                      │ enables                                          │
│                      ▼                                                  │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                                                                   │   │
│  │  • Testability (inject mocks)                                    │   │
│  │  • Flexibility (swap implementations)                            │   │
│  │  • Maintainability (loose coupling)                              │   │
│  │  • All the design patterns work properly                         │   │
│  │    (Strategy, Repository, Adapter, etc.)                         │   │
│  │                                                                   │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

SUMMARY:

| Term | What It Is | Level |
|------|------------|-------|
| DIP | SOLID Principle | Design Principle |
| IoC | Paradigm | Architectural Concept |
| DI | Technique | Implementation Technique |
| DI Container | Tool | Framework/Library |
| Composition Root | Location | Architectural Pattern |
| Service Locator | Anti-pattern | Avoid in business code |
```

---

## 4. Creational Patterns

### A. Factory Pattern

```
FACTORY PATTERN:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  PROBLEM: Object creation logic is complex or needs to be centralized   │
│                                                                         │
│  WITHOUT FACTORY:                   WITH FACTORY:                       │
│                                                                         │
│  // Scattered creation logic        // Centralized creation             │
│  if (type == "pdf") {               document = DocumentFactory          │
│    doc = new PdfDocument()            .create(type)                     │
│    doc.setRenderer(pdfRenderer)                                         │
│    doc.setParser(pdfParser)         // Factory handles complexity       │
│  } else if (type == "word") {                                           │
│    doc = new WordDocument()                                             │
│    doc.setRenderer(wordRenderer)                                        │
│    // ... repeated everywhere                                           │
│  }                                                                      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

MODERN IMPLEMENTATION:

// Simple Factory (most common)
class NotificationFactory {
    create(type: string, config: Config): Notification {
        switch (type) {
            case "email": return new EmailNotification(config)
            case "sms": return new SmsNotification(config)
            case "push": return new PushNotification(config)
            default: throw new UnknownNotificationType(type)
        }
    }
}

// Factory with Registration (extensible)
class NotificationFactory {
    private creators: Map<string, () => Notification> = new Map()

    register(type: string, creator: () => Notification) {
        this.creators.set(type, creator)
    }

    create(type: string): Notification {
        const creator = this.creators.get(type)
        if (!creator) throw new UnknownNotificationType(type)
        return creator()
    }
}

// Usage
factory.register("email", () => new EmailNotification())
factory.register("slack", () => new SlackNotification())  // Easy to extend

// FUNCTIONAL ALTERNATIVE (Modern):
const notificationCreators = {
    email: (config) => new EmailNotification(config),
    sms: (config) => new SmsNotification(config),
    push: (config) => new PushNotification(config),
}

const createNotification = (type, config) =>
    notificationCreators[type]?.(config)
    ?? throw new UnknownNotificationType(type)

WHEN TO USE:
  ✅ Complex object creation with multiple steps
  ✅ Creation logic needs to be reused across codebase
  ✅ Need to decouple client from concrete classes
  ✅ Object families that should be created together

WHEN TO AVOID:
  ❌ Simple object creation (just use constructor)
  ❌ Only one implementation exists
  ❌ Adding complexity without benefit
```

### B. Builder Pattern

```
BUILDER PATTERN:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  PROBLEM: Object has many optional parameters or complex construction   │
│                                                                         │
│  WITHOUT BUILDER:                   WITH BUILDER:                       │
│                                                                         │
│  // Telescoping constructors        // Fluent, readable                 │
│  new Email(to, from, subject,       Email.builder()                     │
│    body, cc, bcc, attachments,        .to("user@example.com")           │
│    headers, priority, null,           .from("app@example.com")          │
│    null, replyTo, ...)                .subject("Hello")                 │
│                                       .body("Content")                  │
│  // What is that 7th null?            .priority(HIGH)                   │
│                                       .build()                          │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

MODERN IMPLEMENTATION:

class HttpRequestBuilder {
    private method: string = "GET"
    private url: string
    private headers: Map<string, string> = new Map()
    private body: any = null
    private timeout: number = 30000

    static create(): HttpRequestBuilder {
        return new HttpRequestBuilder()
    }

    withMethod(method: string): this {
        this.method = method
        return this
    }

    withUrl(url: string): this {
        this.url = url
        return this
    }

    withHeader(key: string, value: string): this {
        this.headers.set(key, value)
        return this
    }

    withBody(body: any): this {
        this.body = body
        return this
    }

    withTimeout(ms: number): this {
        this.timeout = ms
        return this
    }

    build(): HttpRequest {
        if (!this.url) throw new Error("URL is required")
        return new HttpRequest(
            this.method,
            this.url,
            this.headers,
            this.body,
            this.timeout
        )
    }
}

// Usage
const request = HttpRequestBuilder.create()
    .withMethod("POST")
    .withUrl("https://api.example.com/users")
    .withHeader("Content-Type", "application/json")
    .withHeader("Authorization", "Bearer token")
    .withBody({ name: "John" })
    .withTimeout(5000)
    .build()

// IMMUTABLE BUILDER (Modern - Functional Style):
class QueryBuilder {
    private constructor(private readonly params: QueryParams) {}

    static create(): QueryBuilder {
        return new QueryBuilder({ table: "", conditions: [], limit: null })
    }

    from(table: string): QueryBuilder {
        return new QueryBuilder({ ...this.params, table })
    }

    where(condition: Condition): QueryBuilder {
        return new QueryBuilder({
            ...this.params,
            conditions: [...this.params.conditions, condition]
        })
    }

    limit(n: number): QueryBuilder {
        return new QueryBuilder({ ...this.params, limit: n })
    }

    build(): Query {
        return new Query(this.params)
    }
}

WHEN TO USE:
  ✅ Objects with many optional parameters
  ✅ Object construction has multiple steps
  ✅ Need immutable objects with complex creation
  ✅ Want fluent, readable construction API

WHEN TO AVOID:
  ❌ Simple objects with few parameters
  ❌ All parameters are required
  ❌ Object is mutable and can be modified after creation
```

### C. Singleton Pattern

```
SINGLETON PATTERN:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  PROBLEM: Need exactly one instance of a class, globally accessible     │
│                                                                         │
│  ⚠️  WARNING: Often overused and considered an anti-pattern             │
│                                                                         │
│  PROBLEMS WITH SINGLETON:                                               │
│    • Global state (hidden dependencies)                                 │
│    • Hard to test (can't inject mocks)                                  │
│    • Tight coupling                                                     │
│    • Concurrency issues                                                 │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

CLASSIC SINGLETON (Avoid):

class DatabaseConnection {
    private static instance: DatabaseConnection

    private constructor() {}

    static getInstance(): DatabaseConnection {
        if (!DatabaseConnection.instance) {
            DatabaseConnection.instance = new DatabaseConnection()
        }
        return DatabaseConnection.instance
    }
}

// Usage - creates hidden dependency
class UserRepository {
    findById(id: string): User {
        const db = DatabaseConnection.getInstance()  // Hidden dependency!
        return db.query("SELECT * FROM users WHERE id = ?", id)
    }
}

MODERN ALTERNATIVE - Dependency Injection (Preferred):

// Register as singleton in DI container
container.registerSingleton(DatabaseConnection)

// Inject as dependency (explicit, testable)
class UserRepository {
    constructor(private db: DatabaseConnection) {}  // Explicit dependency

    findById(id: string): User {
        return this.db.query("SELECT * FROM users WHERE id = ?", id)
    }
}

// In tests - easy to mock
const mockDb = createMock<DatabaseConnection>()
const repo = new UserRepository(mockDb)

WHEN SINGLETON IS ACCEPTABLE:
  ✅ Logging (truly global, stateless)
  ✅ Configuration (read-only after initialization)
  ✅ Connection pools (managed resource)
  ✅ Caches (with proper invalidation)

MODERN APPROACH:
  • Use DI container to manage lifetime (singleton scope)
  • Inject dependencies explicitly
  • Singleton lifetime, not singleton pattern

WHEN TO AVOID:
  ❌ Any case where DI is available
  ❌ When you need to test the class
  ❌ When the "single instance" requirement isn't real
  ❌ For business logic classes
```

### D. Creational Patterns Comparison

```
CREATIONAL PATTERNS COMPARISON:

┌──────────────┬─────────────────────────────────────────────────────────┐
│ Pattern      │ Use When                                                │
├──────────────┼─────────────────────────────────────────────────────────┤
│ Factory      │ • Need to decouple creation from usage                  │
│              │ • Multiple types share an interface                     │
│              │ • Creation logic is complex                             │
├──────────────┼─────────────────────────────────────────────────────────┤
│ Builder      │ • Many optional parameters                              │
│              │ • Step-by-step construction                             │
│              │ • Want fluent API                                       │
├──────────────┼─────────────────────────────────────────────────────────┤
│ Singleton    │ • Truly need ONE instance (rare)                        │
│              │ • Prefer DI container singleton scope instead           │
├──────────────┼─────────────────────────────────────────────────────────┤
│ Prototype    │ • Cloning is cheaper than creating                      │
│              │ • Objects have many shared configurations               │
└──────────────┴─────────────────────────────────────────────────────────┘
```

---

## 4. Structural Patterns

### A. Adapter Pattern

```
ADAPTER PATTERN:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  PROBLEM: Interface mismatch between what you have and what you need    │
│                                                                         │
│  ┌─────────────┐                              ┌─────────────┐           │
│  │   Client    │ ── expects Interface A ──►  │   Legacy    │           │
│  │             │                              │   System    │           │
│  └─────────────┘                              │ (Interface B)│           │
│                                               └─────────────┘           │
│                                                                         │
│  SOLUTION:                                                              │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                 │
│  │   Client    │───►│   Adapter   │───►│   Legacy    │                 │
│  │             │    │  (A → B)    │    │   System    │                 │
│  └─────────────┘    └─────────────┘    └─────────────┘                 │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

MODERN IMPLEMENTATION:

// Your application expects this interface
interface PaymentGateway {
    charge(amount: Money, card: Card): PaymentResult
    refund(transactionId: string, amount: Money): RefundResult
}

// Third-party SDK has different interface
class StripeSDK {
    createCharge(params: StripeChargeParams): StripeCharge { ... }
    createRefund(chargeId: string, params: StripeRefundParams): StripeRefund { ... }
}

// Adapter translates between them
class StripeAdapter implements PaymentGateway {
    constructor(private stripe: StripeSDK) {}

    charge(amount: Money, card: Card): PaymentResult {
        const stripeParams = this.toStripeChargeParams(amount, card)
        const stripeCharge = this.stripe.createCharge(stripeParams)
        return this.toPaymentResult(stripeCharge)
    }

    refund(transactionId: string, amount: Money): RefundResult {
        const stripeParams = this.toStripeRefundParams(amount)
        const stripeRefund = this.stripe.createRefund(transactionId, stripeParams)
        return this.toRefundResult(stripeRefund)
    }

    private toStripeChargeParams(amount: Money, card: Card): StripeChargeParams {
        return {
            amount: amount.cents,
            currency: amount.currency.toLowerCase(),
            source: card.token,
        }
    }

    private toPaymentResult(charge: StripeCharge): PaymentResult {
        return {
            transactionId: charge.id,
            status: charge.status === "succeeded" ? "success" : "failed",
            amount: Money.fromCents(charge.amount, charge.currency),
        }
    }
}

// Usage - client doesn't know about Stripe
class CheckoutService {
    constructor(private paymentGateway: PaymentGateway) {}

    processPayment(order: Order, card: Card): void {
        const result = this.paymentGateway.charge(order.total, card)
        // ...
    }
}

// Easy to swap payment providers
const checkout = new CheckoutService(new StripeAdapter(stripe))
// or
const checkout = new CheckoutService(new PayPalAdapter(paypal))

WHEN TO USE:
  ✅ Integrating third-party libraries
  ✅ Working with legacy code
  ✅ Need to swap implementations
  ✅ Interface doesn't match your domain

WHEN TO AVOID:
  ❌ Interfaces are already compatible
  ❌ You control both sides (just change the interface)
```

### B. Decorator Pattern

```
DECORATOR PATTERN:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  PROBLEM: Need to add behavior to objects dynamically without           │
│           modifying the class or using inheritance                      │
│                                                                         │
│  ┌─────────────┐                                                       │
│  │   Base      │                                                       │
│  │  Component  │                                                       │
│  └──────┬──────┘                                                       │
│         │                                                               │
│  ┌──────┴──────┐                                                       │
│  │  Decorator  │ ──wraps──► Component                                  │
│  │  (adds      │                                                       │
│  │  behavior)  │                                                       │
│  └─────────────┘                                                       │
│                                                                         │
│  Decorators can be stacked:                                             │
│  Logging(Caching(Retry(HttpClient)))                                   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

MODERN IMPLEMENTATION:

// Base interface
interface HttpClient {
    request(url: string, options: RequestOptions): Response
}

// Base implementation
class BasicHttpClient implements HttpClient {
    request(url: string, options: RequestOptions): Response {
        return fetch(url, options)
    }
}

// Decorator base
abstract class HttpClientDecorator implements HttpClient {
    constructor(protected wrapped: HttpClient) {}

    request(url: string, options: RequestOptions): Response {
        return this.wrapped.request(url, options)
    }
}

// Logging decorator
class LoggingHttpClient extends HttpClientDecorator {
    constructor(wrapped: HttpClient, private logger: Logger) {
        super(wrapped)
    }

    request(url: string, options: RequestOptions): Response {
        this.logger.info(`Request: ${options.method} ${url}`)
        const start = Date.now()

        const response = super.request(url, options)

        this.logger.info(`Response: ${response.status} in ${Date.now() - start}ms`)
        return response
    }
}

// Retry decorator
class RetryHttpClient extends HttpClientDecorator {
    constructor(wrapped: HttpClient, private maxRetries: number = 3) {
        super(wrapped)
    }

    request(url: string, options: RequestOptions): Response {
        let lastError: Error

        for (let attempt = 1; attempt <= this.maxRetries; attempt++) {
            try {
                return super.request(url, options)
            } catch (error) {
                lastError = error
                if (attempt < this.maxRetries) {
                    sleep(exponentialBackoff(attempt))
                }
            }
        }

        throw lastError
    }
}

// Caching decorator
class CachingHttpClient extends HttpClientDecorator {
    constructor(wrapped: HttpClient, private cache: Cache) {
        super(wrapped)
    }

    request(url: string, options: RequestOptions): Response {
        if (options.method === "GET") {
            const cached = this.cache.get(url)
            if (cached) return cached
        }

        const response = super.request(url, options)

        if (options.method === "GET") {
            this.cache.set(url, response)
        }

        return response
    }
}

// Stack decorators
const client = new LoggingHttpClient(
    new RetryHttpClient(
        new CachingHttpClient(
            new BasicHttpClient(),
            cache
        ),
        3
    ),
    logger
)

// Request flows through: Logging → Retry → Caching → Basic

FUNCTIONAL ALTERNATIVE (Modern):

// Using higher-order functions
const withLogging = (client: HttpClient, logger: Logger): HttpClient => ({
    request: (url, options) => {
        logger.info(`Request: ${url}`)
        const response = client.request(url, options)
        logger.info(`Response: ${response.status}`)
        return response
    }
})

const withRetry = (client: HttpClient, maxRetries: number): HttpClient => ({
    request: (url, options) => {
        for (let i = 0; i < maxRetries; i++) {
            try {
                return client.request(url, options)
            } catch (e) {
                if (i === maxRetries - 1) throw e
            }
        }
    }
})

// Compose
const client = withLogging(
    withRetry(
        new BasicHttpClient(),
        3
    ),
    logger
)

WHEN TO USE:
  ✅ Add responsibilities dynamically
  ✅ Behavior composition over inheritance
  ✅ Single Responsibility (each decorator = one concern)
  ✅ Cross-cutting concerns (logging, caching, retry)

WHEN TO AVOID:
  ❌ Only one combination is needed (just use inheritance)
  ❌ Order of decorators doesn't matter (may indicate wrong pattern)
```

### C. Facade Pattern

```
FACADE PATTERN:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  PROBLEM: Complex subsystem with many classes, client needs simple API  │
│                                                                         │
│  WITHOUT FACADE:                    WITH FACADE:                        │
│                                                                         │
│  // Client knows all subsystems     // Client knows only facade         │
│  videoDecoder.decode(file)          videoConverter.convert(             │
│  audioDecoder.decode(file)            file,                             │
│  subtitleParser.parse(file)           outputFormat                      │
│  encoder.encode(video, audio)       )                                   │
│  muxer.mux(encoded, subtitles)                                          │
│  writer.write(output)               // Facade handles complexity        │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

MODERN IMPLEMENTATION:

// Complex subsystem classes
class InventoryService {
    checkStock(productId: string): number { ... }
    reserveStock(productId: string, quantity: number): void { ... }
}

class PaymentService {
    authorize(amount: Money, paymentMethod: PaymentMethod): AuthResult { ... }
    capture(authId: string): CaptureResult { ... }
}

class ShippingService {
    calculateShipping(address: Address, items: Item[]): Money { ... }
    createShipment(orderId: string, address: Address): Shipment { ... }
}

class NotificationService {
    sendOrderConfirmation(email: string, order: Order): void { ... }
    sendShippingNotification(email: string, tracking: string): void { ... }
}

// Facade provides simple interface
class OrderFacade {
    constructor(
        private inventory: InventoryService,
        private payment: PaymentService,
        private shipping: ShippingService,
        private notification: NotificationService
    ) {}

    placeOrder(cart: Cart, payment: PaymentMethod, address: Address): OrderResult {
        // 1. Check inventory
        for (const item of cart.items) {
            const stock = this.inventory.checkStock(item.productId)
            if (stock < item.quantity) {
                return OrderResult.outOfStock(item.productId)
            }
        }

        // 2. Calculate shipping
        const shippingCost = this.shipping.calculateShipping(address, cart.items)
        const total = cart.subtotal.add(shippingCost)

        // 3. Process payment
        const auth = this.payment.authorize(total, payment)
        if (!auth.success) {
            return OrderResult.paymentFailed(auth.error)
        }

        // 4. Reserve inventory
        for (const item of cart.items) {
            this.inventory.reserveStock(item.productId, item.quantity)
        }

        // 5. Capture payment
        this.payment.capture(auth.id)

        // 6. Create shipment
        const order = Order.create(cart, address, auth.id)
        const shipment = this.shipping.createShipment(order.id, address)

        // 7. Send notification
        this.notification.sendOrderConfirmation(cart.customer.email, order)

        return OrderResult.success(order, shipment)
    }
}

// Client code is simple
class CheckoutController {
    constructor(private orderFacade: OrderFacade) {}

    checkout(request: CheckoutRequest): CheckoutResponse {
        const result = this.orderFacade.placeOrder(
            request.cart,
            request.paymentMethod,
            request.shippingAddress
        )
        return this.toResponse(result)
    }
}

WHEN TO USE:
  ✅ Simplify complex subsystems
  ✅ Decouple client from subsystem details
  ✅ Provide entry point to layered system
  ✅ Wrap legacy systems

WHEN TO AVOID:
  ❌ Subsystem is already simple
  ❌ Client needs fine-grained control
  ❌ Facade becomes a "god class" (break it up)
```

### D. Structural Patterns Comparison

```
STRUCTURAL PATTERNS COMPARISON:

┌──────────────┬─────────────────────────────────────────────────────────┐
│ Pattern      │ Use When                                                │
├──────────────┼─────────────────────────────────────────────────────────┤
│ Adapter      │ • Interface mismatch                                    │
│              │ • Integrating external libraries                        │
│              │ • Wrapping legacy code                                  │
├──────────────┼─────────────────────────────────────────────────────────┤
│ Decorator    │ • Add behavior dynamically                              │
│              │ • Compose functionality                                 │
│              │ • Cross-cutting concerns                                │
├──────────────┼─────────────────────────────────────────────────────────┤
│ Facade       │ • Simplify complex subsystems                           │
│              │ • Provide unified interface                             │
│              │ • Hide complexity                                       │
├──────────────┼─────────────────────────────────────────────────────────┤
│ Composite    │ • Tree structures                                       │
│              │ • Part-whole hierarchies                                │
│              │ • Uniform treatment of leaves and composites            │
├──────────────┼─────────────────────────────────────────────────────────┤
│ Proxy        │ • Lazy loading                                          │
│              │ • Access control                                        │
│              │ • Remote objects                                        │
│              │ • Caching                                               │
└──────────────┴─────────────────────────────────────────────────────────┘
```

---

## 5. Behavioral Patterns

### A. Strategy Pattern

```
STRATEGY PATTERN:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  PROBLEM: Multiple algorithms/behaviors that should be interchangeable  │
│                                                                         │
│  WITHOUT STRATEGY:                  WITH STRATEGY:                      │
│                                                                         │
│  class Shipping {                   class Shipping {                    │
│    calculate(order, method) {         constructor(strategy) {           │
│      if (method == "ground") {          this.strategy = strategy        │
│        // ground logic                }                                 │
│      } else if (method == "air") {    calculate(order) {                │
│        // air logic                     return this.strategy            │
│      } else if (method == "sea") {        .calculate(order)             │
│        // sea logic                   }                                 │
│      }                              }                                   │
│      // Growing if-else chain                                           │
│    }                                // Add new strategies without       │
│  }                                  // modifying Shipping class         │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

MODERN IMPLEMENTATION:

// Strategy interface
interface PricingStrategy {
    calculatePrice(order: Order): Money
}

// Concrete strategies
class RegularPricing implements PricingStrategy {
    calculatePrice(order: Order): Money {
        return order.items.reduce(
            (total, item) => total.add(item.price.multiply(item.quantity)),
            Money.zero()
        )
    }
}

class PremiumPricing implements PricingStrategy {
    constructor(private discountPercent: number) {}

    calculatePrice(order: Order): Money {
        const subtotal = new RegularPricing().calculatePrice(order)
        return subtotal.multiply(1 - this.discountPercent / 100)
    }
}

class BulkPricing implements PricingStrategy {
    calculatePrice(order: Order): Money {
        return order.items.reduce((total, item) => {
            const unitPrice = item.quantity >= 100
                ? item.price.multiply(0.8)  // 20% off for bulk
                : item.price
            return total.add(unitPrice.multiply(item.quantity))
        }, Money.zero())
    }
}

// Context
class OrderService {
    constructor(private pricingStrategy: PricingStrategy) {}

    // Strategy can be changed at runtime
    setPricingStrategy(strategy: PricingStrategy): void {
        this.pricingStrategy = strategy
    }

    calculateTotal(order: Order): Money {
        return this.pricingStrategy.calculatePrice(order)
    }
}

// Usage
const orderService = new OrderService(new RegularPricing())
let total = orderService.calculateTotal(order)

// Customer upgrades to premium
orderService.setPricingStrategy(new PremiumPricing(15))
total = orderService.calculateTotal(order)

FUNCTIONAL ALTERNATIVE (Modern):

// Strategies as functions
type PricingStrategy = (order: Order) => Money

const regularPricing: PricingStrategy = (order) =>
    order.items.reduce(
        (total, item) => total.add(item.price.multiply(item.quantity)),
        Money.zero()
    )

const premiumPricing = (discountPercent: number): PricingStrategy =>
    (order) => regularPricing(order).multiply(1 - discountPercent / 100)

const bulkPricing: PricingStrategy = (order) =>
    order.items.reduce((total, item) => {
        const unitPrice = item.quantity >= 100
            ? item.price.multiply(0.8)
            : item.price
        return total.add(unitPrice.multiply(item.quantity))
    }, Money.zero())

// Usage - just pass functions
const calculateOrder = (order: Order, pricing: PricingStrategy): Money =>
    pricing(order)

calculateOrder(order, regularPricing)
calculateOrder(order, premiumPricing(15))
calculateOrder(order, bulkPricing)

WHEN TO USE:
  ✅ Multiple algorithms for same task
  ✅ Need to switch algorithms at runtime
  ✅ Avoid complex conditionals
  ✅ Algorithms need to be testable in isolation

WHEN TO AVOID:
  ❌ Only one algorithm (over-engineering)
  ❌ Algorithms never change
  ❌ Simple conditional logic (if/else is fine)
```

### B. Observer Pattern

```
OBSERVER PATTERN:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  PROBLEM: Objects need to be notified when another object changes       │
│                                                                         │
│  ┌─────────────┐        notifies        ┌─────────────┐                │
│  │   Subject   │ ─────────────────────► │  Observer   │                │
│  │ (Publisher) │                        │(Subscriber) │                │
│  └─────────────┘                        └─────────────┘                │
│        │                                      ▲                         │
│        │                                      │                         │
│        └───────────────────────────┬──────────┘                         │
│                                    │                                    │
│                              ┌─────────────┐                           │
│                              │  Observer   │                           │
│                              │(Subscriber) │                           │
│                              └─────────────┘                           │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

MODERN IMPLEMENTATION:

// Event types
interface OrderEvent {
    type: "placed" | "shipped" | "delivered" | "cancelled"
    order: Order
    timestamp: Date
}

// Observer interface
interface OrderEventHandler {
    handle(event: OrderEvent): void
}

// Subject (Publisher)
class OrderEventPublisher {
    private handlers: Set<OrderEventHandler> = new Set()

    subscribe(handler: OrderEventHandler): void {
        this.handlers.add(handler)
    }

    unsubscribe(handler: OrderEventHandler): void {
        this.handlers.delete(handler)
    }

    publish(event: OrderEvent): void {
        for (const handler of this.handlers) {
            handler.handle(event)
        }
    }
}

// Concrete observers
class EmailNotificationHandler implements OrderEventHandler {
    constructor(private emailService: EmailService) {}

    handle(event: OrderEvent): void {
        switch (event.type) {
            case "placed":
                this.emailService.send(
                    event.order.customer.email,
                    "Order Confirmation",
                    `Your order ${event.order.id} has been placed.`
                )
                break
            case "shipped":
                this.emailService.send(
                    event.order.customer.email,
                    "Order Shipped",
                    `Your order ${event.order.id} is on its way!`
                )
                break
        }
    }
}

class InventoryHandler implements OrderEventHandler {
    constructor(private inventoryService: InventoryService) {}

    handle(event: OrderEvent): void {
        if (event.type === "placed") {
            for (const item of event.order.items) {
                this.inventoryService.reserve(item.productId, item.quantity)
            }
        } else if (event.type === "cancelled") {
            for (const item of event.order.items) {
                this.inventoryService.release(item.productId, item.quantity)
            }
        }
    }
}

class AnalyticsHandler implements OrderEventHandler {
    constructor(private analytics: AnalyticsService) {}

    handle(event: OrderEvent): void {
        this.analytics.track("order_event", {
            type: event.type,
            orderId: event.order.id,
            total: event.order.total.amount,
        })
    }
}

// Setup
const publisher = new OrderEventPublisher()
publisher.subscribe(new EmailNotificationHandler(emailService))
publisher.subscribe(new InventoryHandler(inventoryService))
publisher.subscribe(new AnalyticsHandler(analytics))

// Usage
class OrderService {
    constructor(private publisher: OrderEventPublisher) {}

    placeOrder(order: Order): void {
        // ... order logic

        this.publisher.publish({
            type: "placed",
            order,
            timestamp: new Date()
        })
    }
}

MODERN ALTERNATIVES:

// 1. Event Emitter (Node.js style)
class OrderEvents extends EventEmitter {
    emitPlaced(order: Order) {
        this.emit("placed", order)
    }
}

orderEvents.on("placed", (order) => sendEmail(order))
orderEvents.on("placed", (order) => updateInventory(order))

// 2. Reactive Streams (RxJS style)
const orderPlaced$ = new Subject<Order>()

orderPlaced$.subscribe(order => sendEmail(order))
orderPlaced$.subscribe(order => updateInventory(order))
orderPlaced$
    .pipe(filter(o => o.total > 1000))
    .subscribe(order => notifyVIP(order))

// 3. Message Broker (for distributed systems)
// See Kafka, RabbitMQ patterns

WHEN TO USE:
  ✅ One-to-many dependency between objects
  ✅ Decoupled communication
  ✅ Event-driven systems
  ✅ UI state changes

WHEN TO AVOID:
  ❌ Observers need results from subject
  ❌ Order of notification matters
  ❌ Simple direct calls suffice
```

### C. Command Pattern

```
COMMAND PATTERN:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  PROBLEM: Need to parameterize, queue, or undo operations               │
│                                                                         │
│  COMMAND = Encapsulated operation (object representing an action)       │
│                                                                         │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                 │
│  │   Invoker   │───►│   Command   │───►│  Receiver   │                 │
│  │(Button/API) │    │ (Operation) │    │  (Service)  │                 │
│  └─────────────┘    └─────────────┘    └─────────────┘                 │
│                                                                         │
│  Benefits:                                                              │
│    • Operations as first-class objects                                  │
│    • Undo/Redo support                                                  │
│    • Command queuing                                                    │
│    • Transaction-like behavior                                          │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

MODERN IMPLEMENTATION (CQRS-style):

// Command interface
interface Command<TResult> {
    // Marker interface - commands are data objects
}

// Command handler interface
interface CommandHandler<TCommand extends Command<TResult>, TResult> {
    handle(command: TCommand): TResult
}

// Concrete commands (pure data)
class CreateUserCommand implements Command<User> {
    constructor(
        public readonly email: string,
        public readonly name: string,
        public readonly role: Role
    ) {}
}

class UpdateUserEmailCommand implements Command<void> {
    constructor(
        public readonly userId: string,
        public readonly newEmail: string
    ) {}
}

class DeleteUserCommand implements Command<void> {
    constructor(public readonly userId: string) {}
}

// Command handlers (contain logic)
class CreateUserHandler implements CommandHandler<CreateUserCommand, User> {
    constructor(
        private userRepository: UserRepository,
        private emailService: EmailService
    ) {}

    handle(command: CreateUserCommand): User {
        // Validate
        if (this.userRepository.existsByEmail(command.email)) {
            throw new EmailAlreadyExistsError(command.email)
        }

        // Create
        const user = User.create(
            command.email,
            command.name,
            command.role
        )

        // Persist
        this.userRepository.save(user)

        // Side effects
        this.emailService.sendWelcome(user)

        return user
    }
}

// Command dispatcher (mediator)
class CommandBus {
    private handlers: Map<string, CommandHandler<any, any>> = new Map()

    register<T extends Command<R>, R>(
        commandType: new (...args: any[]) => T,
        handler: CommandHandler<T, R>
    ): void {
        this.handlers.set(commandType.name, handler)
    }

    dispatch<T extends Command<R>, R>(command: T): R {
        const handler = this.handlers.get(command.constructor.name)
        if (!handler) {
            throw new NoHandlerFoundError(command.constructor.name)
        }
        return handler.handle(command)
    }
}

// Usage
const bus = new CommandBus()
bus.register(CreateUserCommand, new CreateUserHandler(repo, email))

// In controller
class UserController {
    constructor(private commandBus: CommandBus) {}

    createUser(request: CreateUserRequest): Response {
        const command = new CreateUserCommand(
            request.email,
            request.name,
            request.role
        )

        const user = this.commandBus.dispatch(command)

        return Response.created(user)
    }
}

WITH UNDO SUPPORT:

interface UndoableCommand<TResult> extends Command<TResult> {
    undo(): void
}

class MoveFileCommand implements UndoableCommand<void> {
    private previousPath: string | null = null

    constructor(
        private fileSystem: FileSystem,
        private sourcePath: string,
        private destPath: string
    ) {}

    execute(): void {
        this.previousPath = this.sourcePath
        this.fileSystem.move(this.sourcePath, this.destPath)
    }

    undo(): void {
        if (this.previousPath) {
            this.fileSystem.move(this.destPath, this.previousPath)
        }
    }
}

class CommandHistory {
    private history: UndoableCommand<any>[] = []

    execute(command: UndoableCommand<any>): void {
        command.execute()
        this.history.push(command)
    }

    undo(): void {
        const command = this.history.pop()
        command?.undo()
    }
}

WHEN TO USE:
  ✅ CQRS (separate commands from queries)
  ✅ Undo/Redo functionality
  ✅ Operation queuing
  ✅ Audit logging (commands are data)
  ✅ Decouple invoker from operation

WHEN TO AVOID:
  ❌ Simple operations without these needs
  ❌ No benefit from decoupling
```

### D. State Pattern

```
STATE PATTERN:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  PROBLEM: Object behavior depends on its state, with many conditionals  │
│                                                                         │
│  WITHOUT STATE PATTERN:             WITH STATE PATTERN:                 │
│                                                                         │
│  class Order {                      class Order {                       │
│    ship() {                           constructor() {                   │
│      if (state == PENDING) {            this.state = new PendingState() │
│        // can't ship                  }                                 │
│      } else if (state == PAID) {      ship() {                          │
│        // do shipping                   this.state.ship(this)           │
│        state = SHIPPED                }                                 │
│      } else if (state == SHIPPED) {   setState(state) {                 │
│        // already shipped               this.state = state              │
│      }                                }                                 │
│      // Growing conditionals        }                                   │
│    }                                                                    │
│  }                                  // Each state handles its behavior │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

MODERN IMPLEMENTATION:

// State interface
interface OrderState {
    pay(order: Order): void
    ship(order: Order): void
    deliver(order: Order): void
    cancel(order: Order): void
    getName(): string
}

// Concrete states
class PendingState implements OrderState {
    pay(order: Order): void {
        order.processPayment()
        order.setState(new PaidState())
    }

    ship(order: Order): void {
        throw new InvalidOperationError("Cannot ship unpaid order")
    }

    deliver(order: Order): void {
        throw new InvalidOperationError("Cannot deliver unpaid order")
    }

    cancel(order: Order): void {
        order.setState(new CancelledState())
    }

    getName(): string { return "pending" }
}

class PaidState implements OrderState {
    pay(order: Order): void {
        throw new InvalidOperationError("Order already paid")
    }

    ship(order: Order): void {
        order.createShipment()
        order.setState(new ShippedState())
    }

    deliver(order: Order): void {
        throw new InvalidOperationError("Order not shipped yet")
    }

    cancel(order: Order): void {
        order.refundPayment()
        order.setState(new CancelledState())
    }

    getName(): string { return "paid" }
}

class ShippedState implements OrderState {
    pay(order: Order): void {
        throw new InvalidOperationError("Order already paid")
    }

    ship(order: Order): void {
        throw new InvalidOperationError("Order already shipped")
    }

    deliver(order: Order): void {
        order.confirmDelivery()
        order.setState(new DeliveredState())
    }

    cancel(order: Order): void {
        throw new InvalidOperationError("Cannot cancel shipped order")
    }

    getName(): string { return "shipped" }
}

class DeliveredState implements OrderState {
    pay(order: Order): void {
        throw new InvalidOperationError("Order already complete")
    }

    ship(order: Order): void {
        throw new InvalidOperationError("Order already complete")
    }

    deliver(order: Order): void {
        throw new InvalidOperationError("Order already delivered")
    }

    cancel(order: Order): void {
        throw new InvalidOperationError("Cannot cancel delivered order")
    }

    getName(): string { return "delivered" }
}

// Context
class Order {
    private state: OrderState = new PendingState()

    setState(state: OrderState): void {
        console.log(`Order ${this.id}: ${this.state.getName()} → ${state.getName()}`)
        this.state = state
    }

    pay(): void { this.state.pay(this) }
    ship(): void { this.state.ship(this) }
    deliver(): void { this.state.deliver(this) }
    cancel(): void { this.state.cancel(this) }

    // Internal methods called by states
    processPayment(): void { /* ... */ }
    createShipment(): void { /* ... */ }
    confirmDelivery(): void { /* ... */ }
    refundPayment(): void { /* ... */ }
}

STATE MACHINE ALTERNATIVE (Modern):

// Using a state machine library
const orderMachine = createMachine({
    id: "order",
    initial: "pending",
    states: {
        pending: {
            on: {
                PAY: "paid",
                CANCEL: "cancelled"
            }
        },
        paid: {
            on: {
                SHIP: "shipped",
                CANCEL: { target: "cancelled", actions: "refund" }
            }
        },
        shipped: {
            on: {
                DELIVER: "delivered"
            }
        },
        delivered: { type: "final" },
        cancelled: { type: "final" }
    }
})

WHEN TO USE:
  ✅ Object behavior depends on state
  ✅ Many state-dependent conditionals
  ✅ State transitions are complex
  ✅ Need state machine visualization

WHEN TO AVOID:
  ❌ Few states with simple transitions
  ❌ States don't significantly change behavior
```

### E. Behavioral Patterns Comparison

```
BEHAVIORAL PATTERNS COMPARISON:

┌──────────────┬─────────────────────────────────────────────────────────┐
│ Pattern      │ Use When                                                │
├──────────────┼─────────────────────────────────────────────────────────┤
│ Strategy     │ • Multiple interchangeable algorithms                   │
│              │ • Need to select algorithm at runtime                   │
│              │ • Avoid complex conditionals                            │
├──────────────┼─────────────────────────────────────────────────────────┤
│ Observer     │ • One-to-many notifications                             │
│              │ • Event-driven communication                            │
│              │ • Decoupled publishers/subscribers                      │
├──────────────┼─────────────────────────────────────────────────────────┤
│ Command      │ • Encapsulate operations as objects                     │
│              │ • Undo/Redo support                                     │
│              │ • Queue or log operations                               │
│              │ • CQRS pattern                                          │
├──────────────┼─────────────────────────────────────────────────────────┤
│ State        │ • Behavior depends on state                             │
│              │ • Complex state transitions                             │
│              │ • State machines                                        │
├──────────────┼─────────────────────────────────────────────────────────┤
│ Template     │ • Algorithm skeleton with customizable steps            │
│ Method       │ • Framework hooks                                       │
│              │ • Inversion of control                                  │
├──────────────┼─────────────────────────────────────────────────────────┤
│ Chain of     │ • Multiple handlers for a request                       │
│ Responsibility│ • Handler determined at runtime                        │
│              │ • Middleware pipelines                                  │
└──────────────┴─────────────────────────────────────────────────────────┘
```

---

## 6. Presentation Patterns (MVC Family)

### A. MVC (Model-View-Controller)

```
MVC PATTERN:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  ORIGIN: Smalltalk GUI (1979) - Trygve Reenskaug                        │
│  SCOPE: Presentation/UI layer (NOT an application architecture)         │
│                                                                         │
│  CLASSIC MVC (Desktop):                                                 │
│                                                                         │
│         User Input                                                      │
│              │                                                          │
│              ▼                                                          │
│      ┌────────────────┐                                                │
│      │   Controller   │────────────────┐                               │
│      │ (Input Handler)│                │                               │
│      └───────┬────────┘                │ Updates                       │
│              │ Updates                 │                               │
│              ▼                         ▼                               │
│      ┌────────────────┐        ┌────────────────┐                      │
│      │     Model      │◄───────│     View       │                      │
│      │    (Data)      │ Reads  │    (Display)   │                      │
│      └───────┬────────┘        └────────────────┘                      │
│              │                         ▲                               │
│              │ Notifies (Observer)     │                               │
│              └─────────────────────────┘                               │
│                                                                         │
│  WEB MVC (Request/Response):                                            │
│                                                                         │
│      HTTP Request                                                       │
│              │                                                          │
│              ▼                                                          │
│      ┌────────────────┐                                                │
│      │   Controller   │                                                │
│      │(Request Handler)│                                                │
│      └───────┬────────┘                                                │
│              │                                                          │
│         Uses │                                                          │
│              ▼                                                          │
│      ┌────────────────┐                                                │
│      │     Model      │                                                │
│      │ (Domain/Data)  │                                                │
│      └───────┬────────┘                                                │
│              │                                                          │
│        Passes│ to                                                       │
│              ▼                                                          │
│      ┌────────────────┐                                                │
│      │     View       │                                                │
│      │  (Template)    │                                                │
│      └───────┬────────┘                                                │
│              │                                                          │
│              ▼                                                          │
│      HTTP Response                                                      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

COMPONENTS:

Model:
  • Business logic and data
  • Independent of presentation
  • Notifies views of changes (in classic MVC)

View:
  • Displays data to user
  • Receives user input
  • Multiple views can display same model

Controller:
  • Handles user input
  • Updates model
  • Selects view

WHEN TO USE:
  ✅ Web applications with server-side rendering
  ✅ Traditional request/response cycle
  ✅ Clear separation of presentation logic
  ✅ Multiple views for same data

LIMITATIONS:
  ❌ View and Controller often tightly coupled
  ❌ "Massive View Controller" problem
  ❌ Doesn't scale well for complex UIs
  ❌ Bidirectional data flow can be confusing
```

### B. MVP (Model-View-Presenter)

```
MVP PATTERN:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  IMPROVEMENT: View is passive (dumb), Presenter contains all logic      │
│                                                                         │
│         User Input                                                      │
│              │                                                          │
│              ▼                                                          │
│      ┌────────────────┐                                                │
│      │      View      │                                                │
│      │   (Passive)    │                                                │
│      └───────┬────────┘                                                │
│              │ Delegates to                                             │
│              ▼                                                          │
│      ┌────────────────┐                                                │
│      │   Presenter    │                                                │
│      │(Presentation   │                                                │
│      │    Logic)      │                                                │
│      └───────┬────────┘                                                │
│              │                                                          │
│         Uses │                                                          │
│              ▼                                                          │
│      ┌────────────────┐                                                │
│      │     Model      │                                                │
│      │    (Data)      │                                                │
│      └────────────────┘                                                │
│                                                                         │
│  KEY DIFFERENCE FROM MVC:                                               │
│    • View is completely passive (no logic)                              │
│    • Presenter updates View directly                                    │
│    • View has reference to Presenter                                    │
│    • Better testability (mock the View interface)                       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

IMPLEMENTATION:

// View interface (for testing)
interface UserView {
    showLoading(): void
    hideLoading(): void
    showUser(user: UserViewModel): void
    showError(message: string): void
}

// Presenter (testable without UI)
class UserPresenter {
    constructor(
        private view: UserView,
        private userService: UserService
    ) {}

    async loadUser(userId: string): Promise<void> {
        this.view.showLoading()

        try {
            const user = await this.userService.getUser(userId)
            const viewModel = this.toViewModel(user)
            this.view.showUser(viewModel)
        } catch (error) {
            this.view.showError("Failed to load user")
        } finally {
            this.view.hideLoading()
        }
    }

    private toViewModel(user: User): UserViewModel {
        return {
            displayName: `${user.firstName} ${user.lastName}`,
            email: user.email,
            memberSince: formatDate(user.createdAt),
        }
    }
}

// View implementation (thin, no logic)
class UserActivity implements UserView {
    private presenter: UserPresenter

    onCreate() {
        this.presenter = new UserPresenter(this, userService)
        this.presenter.loadUser(userId)
    }

    showLoading(): void {
        this.loadingSpinner.visible = true
    }

    hideLoading(): void {
        this.loadingSpinner.visible = false
    }

    showUser(user: UserViewModel): void {
        this.nameText.text = user.displayName
        this.emailText.text = user.email
    }

    showError(message: string): void {
        this.errorText.text = message
    }
}

WHEN TO USE:
  ✅ Android development (traditional)
  ✅ Need highly testable presentation logic
  ✅ Complex view logic
  ✅ Windows Forms, WPF (without MVVM)

WHEN TO AVOID:
  ❌ Simple views with little logic
  ❌ Reactive frameworks available (use MVVM)
```

### C. MVVM (Model-View-ViewModel)

```
MVVM PATTERN:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  IMPROVEMENT: Data binding eliminates manual View updates               │
│                                                                         │
│      ┌────────────────┐                                                │
│      │      View      │                                                │
│      │  (Declarative) │                                                │
│      └───────┬────────┘                                                │
│              │                                                          │
│              │ Data Binding (automatic sync)                            │
│              │  ↑↓                                                      │
│              │                                                          │
│      ┌───────┴────────┐                                                │
│      │   ViewModel    │                                                │
│      │ (Presentation  │                                                │
│      │     State)     │                                                │
│      └───────┬────────┘                                                │
│              │                                                          │
│         Uses │                                                          │
│              ▼                                                          │
│      ┌────────────────┐                                                │
│      │     Model      │                                                │
│      │    (Data)      │                                                │
│      └────────────────┘                                                │
│                                                                         │
│  KEY FEATURES:                                                          │
│    • Two-way data binding                                               │
│    • ViewModel doesn't know about View                                  │
│    • Reactive/Observable properties                                     │
│    • View is declarative                                                │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

MODERN IMPLEMENTATION (React-style):

// ViewModel (React: custom hook)
function useUserViewModel(userId: string) {
    const [state, setState] = useState<UserState>({
        loading: true,
        user: null,
        error: null,
    })

    useEffect(() => {
        loadUser()
    }, [userId])

    async function loadUser() {
        setState(s => ({ ...s, loading: true, error: null }))

        try {
            const user = await userService.getUser(userId)
            setState({
                loading: false,
                user: toViewModel(user),
                error: null,
            })
        } catch (error) {
            setState({
                loading: false,
                user: null,
                error: "Failed to load user",
            })
        }
    }

    function updateEmail(email: string) {
        // Handle user input
    }

    return {
        ...state,
        updateEmail,
        refresh: loadUser,
    }
}

// View (declarative, no logic)
function UserProfile({ userId }: Props) {
    const { loading, user, error, updateEmail } = useUserViewModel(userId)

    if (loading) return <Spinner />
    if (error) return <Error message={error} />

    return (
        <div>
            <h1>{user.displayName}</h1>
            <input
                value={user.email}
                onChange={e => updateEmail(e.target.value)}
            />
        </div>
    )
}

// Angular-style
@Component({
    template: `
        <div *ngIf="loading">Loading...</div>
        <div *ngIf="user">
            <h1>{{ user.displayName }}</h1>
            <input [(ngModel)]="user.email" />
        </div>
    `
})
class UserComponent {
    loading = true
    user: UserViewModel | null = null

    constructor(private userService: UserService) {}

    ngOnInit() {
        this.loadUser()
    }

    async loadUser() {
        this.loading = true
        this.user = await this.userService.getUser(this.userId)
        this.loading = false
    }
}

WHEN TO USE:
  ✅ Frameworks with data binding (Angular, Vue, React, SwiftUI)
  ✅ Complex UI state
  ✅ Need reactive updates
  ✅ Form-heavy applications

WHEN TO AVOID:
  ❌ No data binding support
  ❌ Very simple UIs
```

### D. MVI (Model-View-Intent)

```
MVI PATTERN:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  IMPROVEMENT: Unidirectional data flow, immutable state                 │
│                                                                         │
│      ┌────────────────┐                                                │
│      │      View      │                                                │
│      │   (Renders     │                                                │
│      │    State)      │                                                │
│      └───────┬────────┘                                                │
│              │ User Intent                                              │
│              ▼                                                          │
│      ┌────────────────┐                                                │
│      │    Intent      │                                                │
│      │  (User Action) │                                                │
│      └───────┬────────┘                                                │
│              │                                                          │
│              ▼                                                          │
│      ┌────────────────┐                                                │
│      │    Reducer     │                                                │
│      │ (State Machine)│                                                │
│      └───────┬────────┘                                                │
│              │ New State                                                │
│              ▼                                                          │
│      ┌────────────────┐                                                │
│      │     Model      │ ──────────────────┐                            │
│      │ (Immutable     │                   │                            │
│      │    State)      │                   │                            │
│      └────────────────┘                   │                            │
│              │                            │                            │
│              │ Renders                    │                            │
│              └────────────────────────────┘                            │
│                                                                         │
│  DATA FLOW: Intent → Reducer → State → View → Intent (cycle)           │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

MODERN IMPLEMENTATION (Redux-style):

// State (immutable)
interface UserState {
    readonly loading: boolean
    readonly user: User | null
    readonly error: string | null
}

const initialState: UserState = {
    loading: false,
    user: null,
    error: null,
}

// Intents (Actions)
type UserIntent =
    | { type: "LOAD_USER"; userId: string }
    | { type: "LOAD_USER_SUCCESS"; user: User }
    | { type: "LOAD_USER_FAILURE"; error: string }
    | { type: "UPDATE_EMAIL"; email: string }

// Reducer (pure function: state + intent → new state)
function userReducer(state: UserState, intent: UserIntent): UserState {
    switch (intent.type) {
        case "LOAD_USER":
            return { ...state, loading: true, error: null }

        case "LOAD_USER_SUCCESS":
            return { loading: false, user: intent.user, error: null }

        case "LOAD_USER_FAILURE":
            return { loading: false, user: null, error: intent.error }

        case "UPDATE_EMAIL":
            return state.user
                ? { ...state, user: { ...state.user, email: intent.email } }
                : state

        default:
            return state
    }
}

// Side Effects (async operations)
async function loadUserEffect(
    userId: string,
    dispatch: (intent: UserIntent) => void
) {
    dispatch({ type: "LOAD_USER", userId })

    try {
        const user = await userService.getUser(userId)
        dispatch({ type: "LOAD_USER_SUCCESS", user })
    } catch (error) {
        dispatch({ type: "LOAD_USER_FAILURE", error: error.message })
    }
}

// View (pure render of state)
function UserView({ state, dispatch }: Props) {
    if (state.loading) return <Spinner />
    if (state.error) return <Error message={state.error} />
    if (!state.user) return null

    return (
        <div>
            <h1>{state.user.name}</h1>
            <input
                value={state.user.email}
                onChange={e => dispatch({
                    type: "UPDATE_EMAIL",
                    email: e.target.value
                })}
            />
        </div>
    )
}

WHEN TO USE:
  ✅ Complex state management
  ✅ Need time-travel debugging
  ✅ Predictable state changes
  ✅ Multiple components share state

WHEN TO AVOID:
  ❌ Simple applications (overkill)
  ❌ Team unfamiliar with functional concepts
```

### E. Presentation Patterns Comparison

```
PRESENTATION PATTERNS COMPARISON:

┌─────────┬────────────────┬────────────────┬────────────────┬────────────┐
│ Aspect  │     MVC        │     MVP        │     MVVM       │    MVI     │
├─────────┼────────────────┼────────────────┼────────────────┼────────────┤
│Data Flow│ Bidirectional  │ Bidirectional  │ Two-way binding│Unidirection│
│         │                │                │                │            │
│View Role│ Some logic     │ Passive (dumb) │ Declarative    │ Pure render│
│         │                │                │                │            │
│Testabil-│ Medium         │ High           │ High           │ Very High  │
│ity      │                │                │                │            │
│Complex- │ Low            │ Medium         │ Medium         │ High       │
│ity      │                │                │                │            │
│State    │ Mutable        │ Mutable        │ Observable     │ Immutable  │
│         │                │                │                │            │
│Best For │ Server-side    │ Android        │ Angular, Vue   │ Redux,     │
│         │ web apps       │ (traditional)  │ React, SwiftUI │ complex UI │
└─────────┴────────────────┴────────────────┴────────────────┴────────────┘

SELECTION GUIDE:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  Server-rendered web app? ──► MVC                                       │
│                                                                         │
│  Need highly testable UI? ──► MVP                                       │
│                                                                         │
│  Using reactive framework? ──► MVVM                                     │
│                                                                         │
│  Complex state, need predictability? ──► MVI                            │
│                                                                         │
│  Simple UI, small team? ──► MVC or MVVM (whatever is idiomatic)        │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 7. Architectural Patterns

### A. Repository Pattern

```
REPOSITORY PATTERN:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  PROBLEM: Decouple domain logic from data access details                │
│                                                                         │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                 │
│  │   Domain    │───►│ Repository  │───►│  Database   │                 │
│  │   Logic     │    │ (Interface) │    │ (or any     │                 │
│  │             │    │             │    │  storage)   │                 │
│  └─────────────┘    └─────────────┘    └─────────────┘                 │
│                           ▲                                             │
│                           │ Implements                                  │
│                     ┌─────────────┐                                    │
│                     │ SQL Repo    │                                    │
│                     │ Mongo Repo  │                                    │
│                     │ Memory Repo │                                    │
│                     └─────────────┘                                    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

MODERN IMPLEMENTATION:

// Repository interface (in domain/application layer)
interface UserRepository {
    findById(id: UserId): Promise<User | null>
    findByEmail(email: Email): Promise<User | null>
    findActive(): Promise<User[]>
    save(user: User): Promise<void>
    delete(id: UserId): Promise<void>
}

// SQL implementation
class PostgresUserRepository implements UserRepository {
    constructor(private db: Database) {}

    async findById(id: UserId): Promise<User | null> {
        const row = await this.db.query(
            "SELECT * FROM users WHERE id = $1",
            [id.value]
        )
        return row ? this.toDomain(row) : null
    }

    async findByEmail(email: Email): Promise<User | null> {
        const row = await this.db.query(
            "SELECT * FROM users WHERE email = $1",
            [email.value]
        )
        return row ? this.toDomain(row) : null
    }

    async save(user: User): Promise<void> {
        await this.db.query(
            `INSERT INTO users (id, email, name, status, created_at)
             VALUES ($1, $2, $3, $4, $5)
             ON CONFLICT (id) DO UPDATE SET
               email = $2, name = $3, status = $4`,
            [user.id.value, user.email.value, user.name, user.status, user.createdAt]
        )
    }

    private toDomain(row: UserRow): User {
        return User.reconstitute({
            id: new UserId(row.id),
            email: new Email(row.email),
            name: row.name,
            status: row.status,
            createdAt: row.created_at,
        })
    }
}

// In-memory implementation (for testing)
class InMemoryUserRepository implements UserRepository {
    private users: Map<string, User> = new Map()

    async findById(id: UserId): Promise<User | null> {
        return this.users.get(id.value) ?? null
    }

    async save(user: User): Promise<void> {
        this.users.set(user.id.value, user)
    }

    // ... other methods
}

// Usage in domain service
class UserService {
    constructor(private userRepository: UserRepository) {}

    async registerUser(email: Email, name: string): Promise<User> {
        const existing = await this.userRepository.findByEmail(email)
        if (existing) {
            throw new EmailAlreadyExistsError(email)
        }

        const user = User.create(email, name)
        await this.userRepository.save(user)
        return user
    }
}

WHEN TO USE:
  ✅ Domain-driven design
  ✅ Need to swap storage implementations
  ✅ Unit testing domain logic
  ✅ Complex query requirements

WHEN TO AVOID:
  ❌ Simple CRUD with no business logic
  ❌ Only one storage type ever
```

### B. Specification Pattern

```
SPECIFICATION PATTERN:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  PROBLEM: Complex query logic scattered across code                     │
│                                                                         │
│  SOLUTION: Encapsulate query criteria as composable objects             │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

MODERN IMPLEMENTATION:

// Base specification
interface Specification<T> {
    isSatisfiedBy(item: T): boolean
    and(other: Specification<T>): Specification<T>
    or(other: Specification<T>): Specification<T>
    not(): Specification<T>
}

abstract class CompositeSpecification<T> implements Specification<T> {
    abstract isSatisfiedBy(item: T): boolean

    and(other: Specification<T>): Specification<T> {
        return new AndSpecification(this, other)
    }

    or(other: Specification<T>): Specification<T> {
        return new OrSpecification(this, other)
    }

    not(): Specification<T> {
        return new NotSpecification(this)
    }
}

// Concrete specifications
class ActiveUserSpecification extends CompositeSpecification<User> {
    isSatisfiedBy(user: User): boolean {
        return user.status === "active"
    }
}

class PremiumUserSpecification extends CompositeSpecification<User> {
    isSatisfiedBy(user: User): boolean {
        return user.subscription === "premium"
    }
}

class RecentlyActiveSpecification extends CompositeSpecification<User> {
    constructor(private days: number) { super() }

    isSatisfiedBy(user: User): boolean {
        const cutoff = Date.now() - this.days * 24 * 60 * 60 * 1000
        return user.lastLoginAt.getTime() > cutoff
    }
}

// Usage - compose specifications
const activePremiumUsers = new ActiveUserSpecification()
    .and(new PremiumUserSpecification())

const targetUsers = new ActiveUserSpecification()
    .and(new PremiumUserSpecification())
    .and(new RecentlyActiveSpecification(30))

// Filter in memory
const filtered = users.filter(u => targetUsers.isSatisfiedBy(u))

// Or translate to SQL
class UserRepository {
    findBySpecification(spec: Specification<User>): Promise<User[]> {
        const sql = this.specificationToSql(spec)
        return this.db.query(sql)
    }
}

WHEN TO USE:
  ✅ Complex, reusable query criteria
  ✅ Domain-driven design
  ✅ Combinable business rules
  ✅ Query building in repositories
```

---

## 8. Functional Patterns

### A. Result/Either Pattern

```
RESULT/EITHER PATTERN:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  PROBLEM: Error handling with exceptions is implicit and error-prone    │
│                                                                         │
│  SOLUTION: Return explicit success/failure types                        │
│                                                                         │
│  Result<T, E> = Success<T> | Failure<E>                                │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

MODERN IMPLEMENTATION:

// Result type
type Result<T, E> = Success<T> | Failure<E>

class Success<T> {
    readonly isSuccess = true
    readonly isFailure = false

    constructor(public readonly value: T) {}

    map<U>(fn: (value: T) => U): Result<U, never> {
        return new Success(fn(this.value))
    }

    flatMap<U, E>(fn: (value: T) => Result<U, E>): Result<U, E> {
        return fn(this.value)
    }
}

class Failure<E> {
    readonly isSuccess = false
    readonly isFailure = true

    constructor(public readonly error: E) {}

    map<U>(_fn: (value: never) => U): Result<U, E> {
        return this as unknown as Result<U, E>
    }

    flatMap<U, F>(_fn: (value: never) => Result<U, F>): Result<U, E | F> {
        return this as unknown as Result<U, E | F>
    }
}

// Helper functions
const success = <T>(value: T): Result<T, never> => new Success(value)
const failure = <E>(error: E): Result<never, E> => new Failure(error)

// Error types
type UserError =
    | { type: "NOT_FOUND"; userId: string }
    | { type: "EMAIL_TAKEN"; email: string }
    | { type: "INVALID_EMAIL"; email: string }

// Usage
class UserService {
    async createUser(
        email: string,
        name: string
    ): Promise<Result<User, UserError>> {
        // Validate email
        if (!isValidEmail(email)) {
            return failure({ type: "INVALID_EMAIL", email })
        }

        // Check if email taken
        const existing = await this.userRepo.findByEmail(email)
        if (existing) {
            return failure({ type: "EMAIL_TAKEN", email })
        }

        // Create user
        const user = User.create(email, name)
        await this.userRepo.save(user)

        return success(user)
    }
}

// Caller must handle both cases
async function handleCreateUser(email: string, name: string) {
    const result = await userService.createUser(email, name)

    if (result.isSuccess) {
        console.log("User created:", result.value.id)
    } else {
        switch (result.error.type) {
            case "INVALID_EMAIL":
                console.log("Invalid email format")
                break
            case "EMAIL_TAKEN":
                console.log("Email already registered")
                break
        }
    }
}

// Chaining results
async function registerAndNotify(
    email: string,
    name: string
): Promise<Result<void, UserError | NotificationError>> {
    const userResult = await userService.createUser(email, name)

    return userResult.flatMap(user =>
        notificationService.sendWelcome(user)
    )
}

WHEN TO USE:
  ✅ Expected failure cases (validation, not found)
  ✅ Functional programming style
  ✅ Explicit error handling
  ✅ Chaining operations that can fail

WHEN TO AVOID:
  ❌ Unexpected errors (use exceptions)
  ❌ Team unfamiliar with functional patterns
  ❌ Simple cases where null/undefined suffices
```

### B. Option/Maybe Pattern

```
OPTION/MAYBE PATTERN:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  PROBLEM: Null/undefined checks are error-prone and verbose             │
│                                                                         │
│  SOLUTION: Wrap optional values in a container type                     │
│                                                                         │
│  Option<T> = Some<T> | None                                            │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

IMPLEMENTATION:

type Option<T> = Some<T> | None

class Some<T> {
    readonly isSome = true
    readonly isNone = false

    constructor(public readonly value: T) {}

    map<U>(fn: (value: T) => U): Option<U> {
        return new Some(fn(this.value))
    }

    flatMap<U>(fn: (value: T) => Option<U>): Option<U> {
        return fn(this.value)
    }

    getOrElse(_defaultValue: T): T {
        return this.value
    }
}

class None {
    readonly isSome = false
    readonly isNone = true

    map<U>(_fn: (value: never) => U): Option<U> {
        return this as unknown as Option<U>
    }

    flatMap<U>(_fn: (value: never) => Option<U>): Option<U> {
        return this as unknown as Option<U>
    }

    getOrElse<T>(defaultValue: T): T {
        return defaultValue
    }
}

const some = <T>(value: T): Option<T> => new Some(value)
const none: Option<never> = new None()

// Usage
function findUser(id: string): Option<User> {
    const user = database.find(id)
    return user ? some(user) : none
}

// Safe chaining
const displayName = findUser("123")
    .map(user => user.profile)
    .map(profile => profile.displayName)
    .getOrElse("Anonymous")

// Without Option (error-prone)
const user = findUser("123")
const displayName = user?.profile?.displayName ?? "Anonymous"  // Optional chaining helps

MODERN NOTE:
  In many languages, optional chaining (?.) and nullish coalescing (??)
  reduce the need for explicit Option types. Use Option when you want
  to enforce handling of absent values or chain transformations.
```

---

## 9. Pattern Selection Guide

### A. Decision Matrix

```
PATTERN SELECTION MATRIX:

┌──────────────────────────────────────────────────────────────────────────┐
│ Problem                              │ Pattern(s) to Consider            │
├──────────────────────────────────────┼───────────────────────────────────┤
│ Complex object creation              │ Factory, Builder                  │
│ Many optional parameters             │ Builder                           │
│ Object families                      │ Abstract Factory                  │
│ Interface mismatch                   │ Adapter                           │
│ Add behavior dynamically             │ Decorator                         │
│ Simplify complex subsystem           │ Facade                            │
│ Multiple algorithms                  │ Strategy                          │
│ Notify multiple objects              │ Observer                          │
│ Encapsulate operations               │ Command                           │
│ Behavior depends on state            │ State                             │
│ Decouple data access                 │ Repository                        │
│ Complex queries                      │ Specification                     │
│ Explicit error handling              │ Result/Either                     │
│ UI state management                  │ MVC, MVP, MVVM, MVI               │
└──────────────────────────────────────┴───────────────────────────────────┘
```

### B. Anti-Patterns to Avoid

```
PATTERN ANTI-PATTERNS:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  ❌ PATTERN FEVER                                                       │
│  ────────────────                                                       │
│  Using patterns everywhere, even where not needed                       │
│  → Simple code is better than clever code                               │
│                                                                         │
│  ❌ WRONG PATTERN                                                       │
│  ───────────────                                                        │
│  Using a pattern that doesn't fit the problem                           │
│  → Understand the problem before choosing a pattern                     │
│                                                                         │
│  ❌ PATTERN NAME OBSESSION                                              │
│  ──────────────────────                                                 │
│  Naming everything after patterns even when it's just good design       │
│  → Patterns are discovered, not forced                                  │
│                                                                         │
│  ❌ OVER-ABSTRACTION                                                    │
│  ─────────────────                                                      │
│  Creating interfaces and abstractions "just in case"                    │
│  → YAGNI (You Aren't Gonna Need It)                                    │
│                                                                         │
│  ❌ SINGLETON ABUSE                                                     │
│  ────────────────                                                       │
│  Using Singleton for everything that "should be one"                    │
│  → Use dependency injection instead                                     │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 10. Summary

### Key Takeaways

1. **Patterns solve specific problems** - Don't use them without the problem
2. **Start simple** - Add patterns when complexity demands it
3. **Refactor to patterns** - Introduce patterns through refactoring
4. **Modern languages reduce need** - Functional features, optional chaining, etc.
5. **Composition over inheritance** - Prefer Strategy, Decorator over class hierarchies
6. **Dependency injection** - Makes patterns like Singleton unnecessary

### Pattern Quick Reference

| Category | Pattern | One-Line Summary |
|----------|---------|------------------|
| Creational | Factory | Centralize object creation |
| Creational | Builder | Step-by-step construction |
| Structural | Adapter | Convert interface A to B |
| Structural | Decorator | Add behavior dynamically |
| Structural | Facade | Simplify complex systems |
| Behavioral | Strategy | Interchangeable algorithms |
| Behavioral | Observer | Publish-subscribe notifications |
| Behavioral | Command | Encapsulate operations as objects |
| Behavioral | State | Behavior based on state |
| Presentation | MVC | Separate Model, View, Controller |
| Presentation | MVVM | Data binding with ViewModel |
| Presentation | MVI | Unidirectional data flow |
| Architectural | Repository | Abstract data access |
| Functional | Result | Explicit success/failure |

### Remember

> "Design patterns should not be applied indiscriminately. Often they achieve flexibility and variability by introducing additional levels of indirection, and that can complicate a design and/or cost you some performance." — Gang of Four

> "When you have a hammer, everything looks like a nail. When you know patterns, everything looks like a pattern opportunity. Resist the urge."

---

## Related Guides

- **[architectures.md](architectures.md)**: Application architectures that use these patterns
- **[hexagonal.md](hexagonal.md)**: Hexagonal Architecture using Adapter, Repository patterns
- **[cleanarch.md](cleanarch.md)**: Clean Architecture using these patterns
- **[tdd.md](tdd.md)**: Test-Driven Development - patterns enable testability
- **[typescript.md](typescript.md)**: TypeScript implementations of these patterns
