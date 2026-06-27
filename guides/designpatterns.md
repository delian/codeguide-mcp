# Software Design Patterns Guidelines
Canonical catalog of software design patterns — GoF creational/structural/behavioral, presentation, and modern functional patterns — with selection guidance and anti-patterns. Language-agnostic.

---
name: designpatterns
title: Software Design Patterns Guidelines
version: 2.0
last_reviewed: 2026-06-05
kind: cross-cutting
tools: []
requires: []
recommends:
  - hexagonal
  - cleanarch
  - error-handling
  - tdd
provides:
  - gof-patterns
  - creational-patterns
  - structural-patterns
  - behavioral-patterns
  - presentation-patterns
  - pattern-selection
---

> 🧭 Authored per [`CONVENTIONS.md`](guides://CONVENTIONS.md): shared concerns are referenced, not restated. This guide owns the design-pattern catalog; architecture, error strategy, and test workflow live in their owners.

---

## 0. Prerequisites & References

This guide is the canonical owner of design patterns. Other guides reference it for pattern definitions and show only their language binding. The concerns below are owned elsewhere — fetch them when the task touches them; do not re-derive them here.

> 📎 **RECOMMENDED — fetch when the task touches them:**
> - [`hexagonal.md`](guides://hexagonal.md) — Ports & Adapters, dependency inversion, dependency direction. *(Adapter/Repository patterns map directly to hexagonal ports.)*
> - [`cleanarch.md`](guides://cleanarch.md) — layered application architecture; where these patterns sit in a layered system.
> - [`error-handling.md`](guides://error-handling.md) — error strategy, propagation, and the canonical Result/Either and Option/Maybe rules.
> - [`tdd.md`](guides://tdd.md) — Red-Green-Refactor, regression-test-before-fix. *(Patterns EMERGE from refactoring; do not design them upfront.)*

> 📎 **SEE ALSO:** [`architectures.md`](guides://architectures.md) · [`microservices.md`](guides://microservices.md) · [`parallelism.md`](guides://parallelism.md) *(concurrency-safe pattern variants)* · the language guides (`python.md`, `typescript.md`, `java.md`, `csharp.md`, `go.md`, `rust.md`, …) for idiomatic per-language realizations.

Examples below use a neutral, TypeScript-like pseudocode chosen for readability; each pattern ends with a **language-variation** note rather than re-listing the same code in N languages. For a concrete binding, fetch the relevant language guide.

---

## 1. Core Philosophies: PATTERNS-FIRST

Design-pattern-specific principles only. Test workflow (`tdd.md`), error strategy (`error-handling.md`), and dependency direction (`hexagonal.md`/`cleanarch.md`) come from the references in §0.

- **P**roblem-driven: a pattern is a solution to a problem in a context; no problem → no pattern.
- **A**ppropriate scope: choose the simplest pattern that solves the problem; "no pattern" is a valid choice.
- **T**estability: a pattern MUST improve, never hinder, testability.
- **T**ransparency: the pattern must clarify intent, not obscure it.
- **E**volution: prefer patterns that let code change cheaply (composition over inheritance).
- **R**efactor-to: introduce patterns by refactoring under green tests (see `tdd.md`), not as upfront framework-building.
- **N**aming: use the pattern's name in code (`PricingStrategy`, `OrderEventPublisher`) when it communicates the role.
- **S**implicity-first: modern language features (closures, pattern matching, `?.`, `??`, sum types) replace many classic OOP patterns — reach for the feature before the pattern.

> "A pattern is a solution to a problem in a context. If you don't have the problem, you don't need the solution."

---

## 2. Requirements (MANDATORY, auditable)

RFC-2119 keywords. IDs `DP-<TOPIC>-<NN>`. Each row has a binary gate; rows binding a shared rule cite its owner. These are reviewer/auditor checks — design patterns have no compiler gate, so verification is review- or test-based.

| ID | Requirement | Verify | Gate |
|----|-------------|--------|------|
| DP-FIT-01 | Every applied pattern MUST address a present, named problem — not a speculative one | Code review / ADR (see `adr.md`) | reviewer confirms problem |
| DP-FIT-02 | The simplest pattern that solves the problem MUST be chosen; "no pattern" preferred when sufficient | Code review | reviewer confirms |
| DP-EMERGE-01 | Patterns MUST be introduced by refactoring under green tests, not designed upfront (see `tdd.md`) | `tdd.md` test run | tests green before & after refactor |
| DP-TST-01 | A pattern's behavior MUST be covered by tests asserting behavior, not internal wiring (see `tdd.md`) | language test runner | exit 0, behavior asserted |
| DP-COMP-01 | New variant behavior MUST extend via composition/new type (Open-Closed), not by editing existing classes | Code review / diff | no edits to existing strategy/handler |
| DP-DI-01 | Dependencies MUST be injected, not located; no Service Locator in business logic; no global Singleton access | grep for `getInstance(`/`ServiceLocator`/container in domain | none in business code |
| DP-DIP-01 | High-level modules MUST depend on abstractions, not concretions (see `hexagonal.md`) | review / dep-linter | no high→concrete deps |
| DP-OBS-01 | Observer/event subjects MUST iterate a snapshot of subscribers and isolate subscriber failures | Unit test: unsubscribe-during-notify; failing observer | no concurrent-mod, chain not broken |
| DP-ERR-01 | Result/Either & Option/Maybe usage MUST follow the canonical rules (see `error-handling.md`) | review | conforms to `error-handling.md` |
| DP-DOC-01 | Non-obvious pattern choice MUST be recorded (ADR or code comment) (see `adr.md`, `comments.md`) | review | rationale present |

> **Forbidden**: pattern fever (applying patterns without a problem), forcing the wrong pattern, naming everything after a pattern, "just-in-case" abstraction (YAGNI), Singleton for anything DI can provide, or designing a pattern before a failing test demands it.

---

## 3. Foundational Techniques (DI, IoC, DIP)

These enable every pattern below. The **principle** of dependency inversion and dependency direction is owned by [`hexagonal.md`](guides://hexagonal.md) and [`cleanarch.md`](guides://cleanarch.md) — fetch them for the rules. This section gives only the pattern-mechanics the catalog relies on.

| Term | What it is | Level |
|------|-----------|-------|
| **DIP** | "Depend on abstractions, not concretions" — the *D* in SOLID | Design principle (owner: `hexagonal.md`) |
| **IoC** | "Don't call us, we'll call you" — framework/control-flow inversion | Paradigm |
| **DI** | "Receive your dependencies, don't create them" | Implementation technique |
| **DI container** | Tool that automates wiring (lifetimes: singleton/transient/scoped) | Framework/library |
| **Composition Root** | The one place near the entry point where the object graph is wired | Structural pattern |
| **Service Locator** | "Ask a global for dependencies" | **Anti-pattern** — avoid in business code |

**Dependency Injection — prefer constructor injection:**
```ts
class OrderService {
  constructor(                       // all dependencies explicit & required
    private readonly repo: OrderRepository,   // depend on the abstraction
    private readonly email: EmailService,
  ) {}
}
```
- **Constructor injection** (default): required deps, object fully initialized, immutable wiring.
- **Setter injection**: only for genuinely optional deps; leaves the object temporarily incomplete.
- **Interface injection**: rare; adds ceremony for little benefit.

**Composition Root** — wire the whole graph in one place (`main`/startup), nowhere else; the rest of the app only *receives* dependencies:
```ts
function createApp() {                // === COMPOSITION ROOT ===
  const config = loadConfig()
  const db     = new PostgresConnection(config.db)
  const repo   = new PostgresOrderRepository(db)
  const email  = new SmtpEmailService(config.smtp)
  const svc    = new OrderService(repo, email)
  return new OrderController(svc)
}
```
A DI container automates this (`container.register(OrderRepository, PostgresOrderRepository)` then `container.resolve(OrderService)`), managing singleton/transient/scoped lifetimes — but it is still configured **only at the composition root**.

> **Service Locator is an anti-pattern in business logic** (DP-DI-01): it hides dependencies (you must read all the code to know what a class needs), makes tests fragile (global must be configured first), and turns missing deps into runtime errors. Inject explicitly instead. Acceptable only in framework internals, plugin loaders, or temporary legacy migration.

**Language variation:** containers are ecosystem-specific — Spring/Guice/Dagger (Java), Microsoft.Extensions.DI/Autofac (C#), InversifyJS/tsyringe/NestJS (TS), wire/fx/dig (Go), `dependency-injector` (Python). Many small apps need no container — pure/manual DI at the composition root is enough.

---

## 4. Creational Patterns

How objects are created, decoupling clients from concrete classes. `provides: creational-patterns`.

### 4.1 Factory (Factory Method / Simple Factory / Abstract Factory)

**Problem:** object creation is complex, branchy, or must be reused; the client should not name concrete classes.

```ts
// Registration-based factory — extensible without editing existing code (Open-Closed)
class NotificationFactory {
  private creators = new Map<string, (c: Config) => Notification>()
  register(type: string, make: (c: Config) => Notification) { this.creators.set(type, make) }
  create(type: string, c: Config): Notification {
    const make = this.creators.get(type)
    if (!make) throw new UnknownNotificationType(type)
    return make(c)
  }
}
factory.register("email", c => new EmailNotification(c))
factory.register("slack", c => new SlackNotification(c))   // extend, don't modify
```
- **Simple Factory**: one method maps a key → product (a `switch`/map). Most common.
- **Factory Method**: subclasses decide which product to instantiate (`abstract createButton()`).
- **Abstract Factory**: a family of related products created together (`WidgetFactory.button()`, `.checkbox()`), so a whole family swaps consistently (e.g. light/dark theme).

**Use when** creation is complex/reused, you must decouple from concrete types, or a product *family* must stay consistent. **Avoid** for trivial construction — just call the constructor.

**Language variation:** in functional/dynamic languages a factory is often just a map of constructor functions (`{email: makeEmail, sms: makeSms}[type](config)`); no class needed.

### 4.2 Builder

**Problem:** an object has many optional parameters or multi-step construction; telescoping constructors are unreadable.

```ts
const request = HttpRequest.builder()
  .method("POST").url("https://api/users")
  .header("Authorization", "Bearer …")
  .body({ name: "John" }).timeout(5000)
  .build()                              // build() validates required fields, throws if missing
```
Prefer an **immutable builder** (each step returns a new builder with `{...params, field}`) when the result must be immutable. **Use when** many optional params or fluent construction adds clarity. **Avoid** when all params are required and few.

**Language variation:** Python uses keyword args / `@dataclass`; Kotlin/Scala named+default params; Go the functional-options idiom (`New(WithTimeout(5*time.Second))`). These often make an explicit Builder unnecessary.

### 4.3 Prototype

**Problem:** creating an object fresh is expensive or its configuration is complex; cloning an existing instance is cheaper.

```ts
interface Cloneable<T> { clone(): T }
const base = new GameEnemy({ hp: 100, weapons, ai })   // expensive to assemble
const variant = base.clone()                            // copy, then tweak
variant.hp = 150
```
Beware shallow vs deep copy (shared mutable references). **Use when** cloning beats constructing, or for many objects sharing a base config. **Language variation:** JS `structuredClone`, Python `copy.deepcopy`, C# `MemberwiseClone`, Rust `#[derive(Clone)]` — usually no hand-written pattern needed.

### 4.4 Singleton

> ⚠️ Usually an anti-pattern: it is hidden global state, hard to test (cannot inject a mock), tightly coupled, and concurrency-prone (race in lazy init).

```ts
// PREFER: singleton *lifetime* via DI, not the Singleton *pattern*
container.registerSingleton(DatabaseConnection)
class UserRepository {
  constructor(private readonly db: DatabaseConnection) {}   // explicit, mockable
}
```
A true global instance is acceptable only for genuinely global, stateless/read-only resources (logger, immutable config, connection pool). Even then, model it as a **singleton-scoped DI registration**, not a `getInstance()` global (DP-DI-01).

### Creational comparison

| Pattern | Use when |
|---|---|
| Factory | decouple creation from use; multiple types share an interface; complex creation |
| Abstract Factory | a *family* of related products must be created consistently |
| Builder | many optional params; step-by-step / fluent; immutable result |
| Prototype | cloning is cheaper than constructing; many shared base configs |
| Singleton | truly need ONE instance (rare) — prefer DI singleton scope |

---

## 5. Structural Patterns

How objects are composed into larger structures. `provides: structural-patterns`.

### 5.1 Adapter

**Problem:** the interface you have ≠ the interface you need (third-party SDK, legacy system).

```ts
interface PaymentGateway { charge(amount: Money, card: Card): PaymentResult }

class StripeAdapter implements PaymentGateway {       // adapts StripeSDK → your port
  constructor(private readonly stripe: StripeSDK) {}
  charge(amount: Money, card: Card): PaymentResult {
    const c = this.stripe.createCharge({ amount: amount.cents, currency: amount.currency, source: card.token })
    return { transactionId: c.id, status: c.status === "succeeded" ? "success" : "failed" }
  }
}
const checkout = new CheckoutService(new StripeAdapter(stripe))  // swap providers freely
```
> Adapter is the structural realization of a **port adapter** in Ports & Adapters — see [`hexagonal.md`](guides://hexagonal.md). The interface (`PaymentGateway`) is the port; `StripeAdapter` is the driven adapter.

**Use when** integrating external libs/legacy code or swapping implementations. **Avoid** when you control both sides (just change the interface).

### 5.2 Decorator

**Problem:** add behavior to an object dynamically, composably, without subclassing every combination.

```ts
abstract class HttpClientDecorator implements HttpClient {
  constructor(protected readonly wrapped: HttpClient) {}
  request(url: string, o: RequestOptions) { return this.wrapped.request(url, o) }
}
class LoggingHttpClient extends HttpClientDecorator { /* log around super.request */ }
class RetryHttpClient   extends HttpClientDecorator { /* retry around super.request */ }
class CachingHttpClient extends HttpClientDecorator { /* cache GETs */ }

// Stack — request flows Logging → Retry → Caching → Basic
const client = new LoggingHttpClient(new RetryHttpClient(new CachingHttpClient(new BasicHttpClient(), cache), 3), logger)
```
Each decorator = one cross-cutting concern (logging, retry, caching) → Single Responsibility, composable. **Order can matter** — make it explicit. **Avoid** when only one fixed combination exists (use inheritance) .

**Language variation:** with higher-order functions a decorator is just function composition (`withLogging(withRetry(base))`); Python has `@decorator` syntax; many languages favor middleware chains for the same effect.

### 5.3 Facade

**Problem:** a complex subsystem exposes many classes; the client wants one simple entry point.

```ts
class OrderFacade {                       // orchestrates inventory, payment, shipping, notification
  constructor(private inv: InventoryService, private pay: PaymentService,
              private ship: ShippingService, private notify: NotificationService) {}
  placeOrder(cart: Cart, method: PaymentMethod, addr: Address): OrderResult {
    // check stock → calc shipping → authorize → reserve → capture → ship → notify
  }
}
```
**Use when** simplifying a subsystem or providing a layered entry point. **Avoid** letting it grow into a god-object — if it does too much, split it.

### 5.4 Composite

**Problem:** clients should treat individual objects and compositions of objects uniformly (tree/part-whole hierarchies).

```ts
interface FsNode { size(): number }                       // component
class File implements FsNode { constructor(private bytes: number) {} size() { return this.bytes } }
class Directory implements FsNode {                        // composite
  private children: FsNode[] = []
  add(n: FsNode) { this.children.push(n) }
  size() { return this.children.reduce((t, c) => t + c.size(), 0) }   // recurse uniformly
}
```
**Use for** trees (filesystems, UI component trees, org charts) where leaves and branches share an interface.

### 5.5 Proxy

**Problem:** control access to an object — lazy loading, access control, caching, remoting — behind the same interface.

```ts
class LazyImage implements Image {                 // virtual proxy: defer expensive load
  private real: RealImage | null = null
  constructor(private readonly path: string) {}
  render() { (this.real ??= new RealImage(this.path)).render() }
}
```
Variants: **virtual** (lazy init), **protection** (access checks), **remote** (RPC stub), **caching/smart** proxy. Distinguish from Decorator: Proxy controls *access* to the same interface; Decorator *adds behavior*.

### 5.6 Bridge & Flyweight (briefly)

- **Bridge**: split an abstraction from its implementation so both vary independently (`Shape` ↔ `Renderer`: `Circle(VectorRenderer)` vs `Circle(RasterRenderer)`). Prevents a class explosion (`VectorCircle`, `RasterCircle`, …).
- **Flyweight**: share immutable intrinsic state across many objects to save memory (glyphs, tile types, particle sprites); pass extrinsic state (position) as method args. Reach for it only under real memory pressure.

### Structural comparison

| Pattern | Use when |
|---|---|
| Adapter | interface mismatch; wrap external/legacy code |
| Decorator | add behavior dynamically/composably; cross-cutting concerns |
| Facade | simplify/unify a complex subsystem |
| Composite | tree / part-whole; uniform leaf & branch treatment |
| Proxy | control access: lazy, protection, remote, caching |
| Bridge | abstraction & implementation must vary independently |
| Flyweight | many objects share immutable state; memory pressure |

---

## 6. Behavioral Patterns

How objects communicate and distribute responsibility. `provides: behavioral-patterns`.

### 6.1 Strategy

**Problem:** several interchangeable algorithms; growing `if/else`/`switch` over a "kind".

```ts
interface PricingStrategy { calculate(order: Order): Money }
class RegularPricing implements PricingStrategy { calculate(o) { /* … */ } }
class BulkPricing    implements PricingStrategy { calculate(o) { /* 20% off ≥100 units */ } }

class OrderService {
  constructor(private strategy: PricingStrategy) {}
  setStrategy(s: PricingStrategy) { this.strategy = s }   // swap at runtime
  total(o: Order) { return this.strategy.calculate(o) }
}
```
**Use when** multiple algorithms must be selected at runtime or tested in isolation; it replaces conditional chains and satisfies Open-Closed. **Avoid** when there is only one algorithm.

**Language variation:** in FP-leaning languages a strategy is just a function value (`type PricingStrategy = (o: Order) => Money`); pass the function, skip the interface and classes entirely.

### 6.2 Observer (Publish-Subscribe)

**Problem:** one object's change must notify many others, decoupled.

```ts
class OrderEventPublisher {
  private handlers = new Set<OrderEventHandler>()
  subscribe(h: OrderEventHandler)   { this.handlers.add(h) }
  unsubscribe(h: OrderEventHandler) { this.handlers.delete(h) }
  publish(e: OrderEvent) {
    for (const h of [...this.handlers]) {   // DP-OBS-01: snapshot — safe if a handler unsubscribes
      try { h.handle(e) } catch (err) { this.log.error(err) }  // isolate failures; don't break the chain
    }
  }
}
```
> **Two recurring bugs (DP-OBS-01):** (1) a subscriber that unsubscribes *during* notification mutates the list mid-iteration → iterate a copy; (2) one observer throwing aborts the rest → catch per-observer. Also watch for memory leaks from never-unsubscribed handlers.

**Use for** one-to-many, event-driven decoupling, UI state. **Avoid** when the subject needs results back, or strict notification ordering is required.

**Language variation / modern alternatives:** Node `EventEmitter` (`emitter.on("placed", …)`), reactive streams (RxJS `Subject`, with `pipe(filter(…))`), or — across process boundaries — a message broker (see [`kafka.md`](guides://kafka.md), [`microservices.md`](guides://microservices.md)).

### 6.3 Command

**Problem:** turn an operation into a first-class object so it can be queued, logged, undone, or dispatched.

```ts
interface CommandHandler<C, R> { handle(c: C): R }
class CreateUserCommand { constructor(readonly email: string, readonly name: string) {} }   // pure data
class CreateUserHandler implements CommandHandler<CreateUserCommand, User> { handle(c) { /* … */ } }

class CommandBus {                       // dispatcher / mediator
  private handlers = new Map<string, CommandHandler<any, any>>()
  register(type: Function, h: CommandHandler<any, any>) { this.handlers.set(type.name, h) }
  dispatch<R>(c: object): R { return this.handlers.get(c.constructor.name)!.handle(c) }
}
```
For **undo/redo**, add `undo()` and keep a `CommandHistory` stack. Command underpins **CQRS** (commands separate from queries) and audit logs (commands are serializable data). **Use when** you need queuing, undo, audit, or invoker/operation decoupling.

### 6.4 State

**Problem:** behavior depends on an object's state, expressed as sprawling conditionals; transitions are complex.

```ts
interface OrderState { pay(o: Order): void; ship(o: Order): void; cancel(o: Order): void }
class PendingState implements OrderState {
  pay(o)    { o.processPayment(); o.setState(new PaidState()) }
  ship(o)   { throw new InvalidOperation("cannot ship unpaid order") }
  cancel(o) { o.setState(new CancelledState()) }
}
class PaidState implements OrderState { ship(o){ o.createShipment(); o.setState(new ShippedState()) } /* … */ }

class Order {
  private state: OrderState = new PendingState()
  setState(s: OrderState) { this.state = s }
  pay()  { this.state.pay(this) }       // delegate to current state
  ship() { this.state.ship(this) }
}
```
Each state owns its transitions and forbids invalid ones, removing conditionals. **Use when** behavior is state-dependent with many transitions. **Avoid** for a couple of trivial states.

**Modern alternative:** a declarative state-machine spec (`{ pending: { on: { PAY: "paid" } }, … }`, XState-style) for visualizable, validated transitions.

### 6.5 Template Method, Chain of Responsibility, and other GoF behaviorals (briefly)

- **Template Method**: a base class fixes the algorithm skeleton and lets subclasses fill steps (`prepare(); process(); finish()` with abstract `process()`). A common form of IoC ("don't call us, we'll call you"). FP equivalent: pass the varying step as a function.
- **Chain of Responsibility**: pass a request along a chain until one handler handles it (middleware pipelines, validation chains, escalation). Each link decides to handle or forward.
- **Mediator**: centralize many-to-many object communication in one mediator (the `CommandBus` above is one); reduces tangled direct references.
- **Iterator**: provide sequential access without exposing internals — built into most modern languages (`for…of`, generators/`yield`, `Iterable`/`Iterator`). Rarely hand-written today.
- **Memento**: capture/restore an object's state without breaking encapsulation (undo snapshots).
- **Visitor**: add operations to a fixed object structure without modifying it (AST traversal, double dispatch). Powerful but heavy; sum types + pattern matching are the modern substitute.

### Behavioral comparison

| Pattern | Use when |
|---|---|
| Strategy | interchangeable algorithms, chosen at runtime |
| Observer | one-to-many event notification; decoupled pub/sub |
| Command | operations as objects: queue, undo, audit, CQRS |
| State | behavior depends on state; complex transitions |
| Template Method | fixed algorithm skeleton, variable steps |
| Chain of Responsibility | a request handled by one of a chain (middleware) |
| Mediator | centralize many-to-many communication |
| Iterator | sequential traversal — usually language-built-in |
| Visitor | new operations over a stable structure (AST) |

---

## 7. Presentation Patterns (MVC Family)

UI/presentation-layer separation. These govern presentation only — they are **not** application architectures (architecture: [`cleanarch.md`](guides://cleanarch.md), [`hexagonal.md`](guides://hexagonal.md)). `provides: presentation-patterns`.

- **MVC** (Model-View-Controller): Controller handles input, updates Model, selects View; Model notifies Views. Origin: Smalltalk GUI, 1979. Fits server-rendered request/response web apps. Pitfall: bidirectional flow and "massive view controllers".
- **MVP** (Model-View-Presenter): View is **passive** (an interface with `showLoading/showUser/showError`), Presenter holds all presentation logic and is unit-testable without a UI. Traditional Android, WinForms/WPF.
- **MVVM** (Model-View-ViewModel): a ViewModel exposes observable state; **data binding** syncs View↔ViewModel automatically; the ViewModel doesn't know the View. Frameworks with binding (Angular, Vue, SwiftUI; React via hooks).
- **MVI** (Model-View-Intent): strictly **unidirectional** flow — `Intent → Reducer → immutable State → View → Intent`. Predictable, time-travel-debuggable. Redux/Elm-style; best for complex shared state.

| Aspect | MVC | MVP | MVVM | MVI |
|---|---|---|---|---|
| Data flow | bidirectional | bidirectional | two-way binding | unidirectional |
| View role | some logic | passive | declarative | pure render |
| State | mutable | mutable | observable | immutable |
| Testability | medium | high | high | very high |
| Best for | server-rendered web | traditional Android | Angular/Vue/SwiftUI | Redux / complex UI |

**Selection:** server-rendered → MVC; need highly testable UI logic → MVP; reactive/binding framework → MVVM; complex predictable state → MVI. For per-framework realizations see [`reactjs.md`](guides://reactjs.md), [`angular.md`](guides://angular.md), [`svelte.md`](guides://svelte.md), [`android.md`](guides://android.md), [`flutter.md`](guides://flutter.md), [`ios.md`](guides://ios.md).

---

## 8. Architectural & Functional Patterns (owned elsewhere — bindings only)

A few catalog patterns are canonically owned by other guides. They appear here for completeness; fetch the owner for the rules.

### 8.1 Repository & Specification

- **Repository**: an interface in the domain/application layer (`UserRepository` with `findById/save/…`) abstracts persistence; implementations (`PostgresUserRepository`, `InMemoryUserRepository` for tests) live in adapters. This is the persistence **port** of Ports & Adapters — see [`hexagonal.md`](guides://hexagonal.md) for layering rules and [`cleanarch.md`](guides://cleanarch.md) for where it sits. Use it to swap storage and to test domain logic against an in-memory fake. Avoid for trivial CRUD with no domain logic (it adds a layer for nothing).
- **Specification**: encapsulate a business predicate as a composable object with `isSatisfiedBy(item)` plus `and`/`or`/`not`, so rules combine (`active.and(premium).and(recentlyActive(30))`) and can be evaluated in memory or translated to a query. Use for reusable, combinable business rules; for query/persistence concerns defer to the datastore guide.

### 8.2 Result/Either & Option/Maybe — see `error-handling.md`

These are the canonical functional error/absence types: `Result<T,E> = Success<T> | Failure<E>` and `Option<T> = Some<T> | None`, with `map`/`flatMap` for chaining. **The rules for when to return a Result vs. throw, and how to model error types, are owned by [`error-handling.md`](guides://error-handling.md)** — do not re-derive them here.

Pattern note (mechanics only): return an explicit `Result` for **expected** failures (validation, not-found) and chain with `flatMap`; reserve exceptions for **unexpected** errors. For mere absence, modern languages' optional chaining (`?.`) and nullish coalescing (`??`) often replace an explicit `Option` — use `Option`/`Maybe` when you want to *force* handling or chain transformations. Sum types + pattern matching (Rust `Result`/`Option`, Kotlin sealed classes, TS discriminated unions, Scala `Either`) are the idiomatic realization.

---

## 9. Pattern Selection Guide

| Problem | Pattern(s) to consider |
|---|---|
| Complex object creation | Factory, Builder |
| Many optional parameters | Builder |
| Families of related objects | Abstract Factory |
| Cheap copy of a configured object | Prototype |
| Interface mismatch / external SDK | Adapter |
| Add behavior dynamically | Decorator |
| Simplify a complex subsystem | Facade |
| Tree / part-whole structure | Composite |
| Control access (lazy/remote/cache) | Proxy |
| Interchangeable algorithms | Strategy |
| Notify many objects | Observer |
| Operation as object (undo/queue/CQRS) | Command |
| Behavior depends on state | State |
| Fixed skeleton, variable steps | Template Method |
| Request handled by one of a chain | Chain of Responsibility |
| Decouple data access | Repository (see `hexagonal.md`) |
| Composable business rules | Specification |
| Explicit success/failure | Result/Either (see `error-handling.md`) |
| UI state management | MVC / MVP / MVVM / MVI |

### Anti-patterns to avoid

- **Pattern fever** — patterns everywhere; simple code beats clever code.
- **Wrong pattern** — forcing a pattern that doesn't fit; understand the problem first.
- **Pattern-name obsession** — naming everything after a pattern when it's just good design.
- **Over-abstraction** — interfaces "just in case" (YAGNI).
- **Singleton abuse** — global state for everything "that should be one"; use DI.
- **God-object Facade** — a facade that grows into a do-everything class.
- **Premature abstraction** — extracting an interface before the second implementation exists (let it emerge — see `tdd.md`).

> "Patterns achieve flexibility by adding indirection, which can complicate a design and cost performance." — Gang of Four. Discover patterns through refactoring; don't force them.

---

## 10. Deployment Checklist

Generated from §2 — one box per requirement ID. No new requirements.

- [ ] DP-FIT-01 — every pattern addresses a present, named problem (ADR/review)
- [ ] DP-FIT-02 — simplest sufficient pattern chosen ("no pattern" preferred where it suffices)
- [ ] DP-EMERGE-01 — patterns introduced by refactoring under green tests (see `tdd.md`)
- [ ] DP-TST-01 — pattern behavior tested (behavior, not wiring; see `tdd.md`)
- [ ] DP-COMP-01 — new variants extend via composition/new type, not edits (Open-Closed)
- [ ] DP-DI-01 — dependencies injected; no Service Locator / global Singleton in business logic
- [ ] DP-DIP-01 — high-level modules depend on abstractions (see `hexagonal.md`)
- [ ] DP-OBS-01 — observers iterate a snapshot and isolate subscriber failures
- [ ] DP-ERR-01 — Result/Either & Option/Maybe follow `error-handling.md`
- [ ] DP-DOC-01 — non-obvious pattern choice recorded (ADR/comment)

---
**End of Software Design Patterns Guidelines**
