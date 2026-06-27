# Swift Development Guidelines
Mandatory coding standards for Swift: value-typed, optional-safe, protocol-oriented, concurrency-safe. Swift 6.x, SwiftPM, Swift Testing, swiftformat, swiftlint.

---
name: swift
title: Swift Development Guidelines
version: 2.0
last_reviewed: 2026-06-05
kind: language
tools: [swift@6.0, swiftpm, swift-testing, xctest, swiftformat, swiftlint]
requires:
  - tdd
  - secure-coding
  - error-handling
recommends:
  - ios
  - hexagonal
  - comments
  - parallelism
provides:
  - value-types
  - optionals
  - protocol-oriented
  - swift-concurrency
  - arc-memory
---

> 🧭 Authored per [`CONVENTIONS.md`](guides://CONVENTIONS.md): shared concerns are referenced, not restated. This guide covers only what is unique to Swift the language. Apple-platform UI (UIKit/SwiftUI) lives in [`ios.md`](guides://ios.md).

---

## 0. Prerequisites & References

Fetch and apply these **before** generating Swift code. Their rules are assumed below and not repeated.

> 📎 **REQUIRED — fetch & apply first:**
> - [`tdd.md`](guides://tdd.md) — test-first, Red-Green-Refactor, regression-test-before-fix, coverage. *(Swift binding: runner is `swift test`; frameworks are Swift Testing / XCTest.)*
> - [`secure-coding.md`](guides://secure-coding.md) — supply chain, secrets, CVE policy. *(Swift binding: pin SPM deps, scan with Snyk/OWASP/Trivy, Keychain for secrets.)*
> - [`error-handling.md`](guides://error-handling.md) — error strategy & propagation. *(Swift binding: typed `throws` vs `Result`, §5.E.)*

> 📎 **RECOMMENDED — fetch when the task touches them:**
> - [`ios.md`](guides://ios.md) — SwiftUI/UIKit, app lifecycle, Apple-platform UI *(this guide stays framework-agnostic; all UI rules live there)*
> - [`hexagonal.md`](guides://hexagonal.md) — layering, ports/adapters, dependency inversion *(Swift binding: protocols as ports, §5.C)*
> - [`parallelism.md`](guides://parallelism.md) — concurrency strategy, data races *(Swift binding: actors, async/await, `Sendable`, §5.F)*
> - [`comments.md`](guides://comments.md) — doc-comment policy *(binding: `///` DocC markup)*

> 📎 **SEE ALSO:** [`designpatterns.md`](guides://designpatterns.md) · [`code-review.md`](guides://code-review.md) · [`ci-cd.md`](guides://ci-cd.md) · [`semver.md`](guides://semver.md)

---

## 1. Core Philosophies: SWIFT-FIRST

Swift-specific principles only. TDD, security, error-handling strategy, and concurrency policy come from §0.

- **S**afe optionals: model absence with `Optional`; unwrap with `if let`/`guard let`/`??`. Never force-unwrap (`!`) outside tests or proven invariants.
- **W**ell-named APIs: follow the Swift API Design Guidelines — names read at the call site like prose; argument labels clarify role, not type.
- **I**mmutable by default: `let` over `var`; value types (`struct`/`enum`) over `class`; mutate by returning new values.
- **F**irst-class protocols: program to protocols with default implementations and generics; reference types only for identity, inheritance, or Obj-C interop.
- **T**yped concurrency: structured `async`/`await`, actors for shared mutable state, `Sendable` checked under Swift 6 strict concurrency.

**Verified Code**: Agent-generated Swift MUST pass every gate in §2 before delivery.

---

## 2. Requirements (MANDATORY, auditable)

RFC-2119 keywords. IDs `SWIFT-<TOPIC>-<NN>`. Each row has a binary gate; rows binding a shared rule cite its owner.

| ID | Requirement | Verify | Gate |
|----|-------------|--------|------|
| SWIFT-TST-01 | Every feature MUST be test-first (see `tdd.md`) | `swift test` | exit 0, 0 skips |
| SWIFT-TST-02 | Each bug MUST get a regression test before the fix (see `tdd.md`) | `swift test` | failing→passing |
| SWIFT-TST-03 | Business logic coverage MUST meet the project gate | `swift test --enable-code-coverage` | ≥ gate |
| SWIFT-FMT-01 | Code MUST be formatted | `swiftformat --lint .` | no diff |
| SWIFT-LINT-01 | Linter MUST pass clean | `swiftlint --strict` | 0 violations |
| SWIFT-TYP-01 | Build MUST be warning-free under strict concurrency | `swift build -Xswiftc -warnings-as-errors` | exit 0 |
| SWIFT-SAFE-01 | No force-unwrap/force-try/force-cast in non-test code | `swiftlint` (`force_unwrapping`, `force_try`, `force_cast`) | 0 violations |
| SWIFT-CONC-01 | Concurrency MUST be data-race-safe (see `parallelism.md`) | `swift build` (Swift 6 strict concurrency) | 0 `Sendable` warnings |
| SWIFT-DOC-01 | Public APIs MUST have doc comments (see `comments.md`) | `swift package generate-documentation` | builds clean |
| SWIFT-SEC-01 | 0 known CVEs in deps (see `secure-coding.md`) | `snyk test` / `trivy fs .` | 0 high/critical |
| SWIFT-DEP-01 | Lockfile in sync & committed | `swift package resolve` + `git diff --exit-code Package.resolved` | no diff |

> **Forbidden**: shipping implementation before its test (violates `tdd.md`), fixing a bug without a regression test first, force-unwrapping optionals in production to silence a failure, `@unchecked Sendable` without a documented invariant, or disabling App Transport Security.

---

## 3. Verification Protocol

Run, in order, before presenting code. Fix → re-run until every gate is green.

```bash
swiftformat --lint .                          # SWIFT-FMT-01
swiftlint --strict                            # SWIFT-LINT-01 / SWIFT-SAFE-01
swift build -Xswiftc -warnings-as-errors      # SWIFT-TYP-01 / SWIFT-CONC-01 (strict concurrency)
swift test --enable-code-coverage             # SWIFT-TST-01/03
swift package resolve                         # SWIFT-DEP-01 (then check Package.resolved is unchanged)
snyk test          # or: trivy fs .           # SWIFT-SEC-01
```

The *why* behind each gate lives in its §0 owner; do not re-derive it here.

---

## 4. Project Structure

Idiomatic SwiftPM layout. Architectural principles (dependency direction, ports/adapters, acyclic deps) are owned by [`hexagonal.md`](guides://hexagonal.md); below is only their Swift mapping.

```
MyPackage/
├── Package.swift            # manifest: targets, deps, platforms
├── Package.resolved         # committed lockfile (SWIFT-DEP-01)
├── Sources/
│   ├── Domain/              # pure value types & protocols — no UIKit/SwiftUI/IO imports
│   ├── Application/         # use cases; depends on protocol "ports"
│   └── Adapters/            # URLSession/persistence/CLI implementations of ports
├── Tests/
│   └── MyPackageTests/      # mirrors Sources/ (see tdd.md)
└── README.md
```

- One public `protocol` per port; domain depends on the protocol, adapters conform to it (dependency inversion).
- Group by feature/domain, not by type. No circular module dependencies — split targets to enforce boundaries at compile time.
- Keep Apple-platform UI targets thin and out of `Domain`; UI guidance is owned by [`ios.md`](guides://ios.md).

---

## 5. Swift Specifics

The unique value of this guide.

### A. Value vs reference types

Default to `struct`/`enum`; reach for `class` only for identity, inheritance, deinit-based cleanup, or Obj-C interop. Value types give copy semantics, no shared mutable state, and free thread-safety.

```swift
struct User: Codable, Equatable, Sendable {   // value model — copied, never aliased
    let id: UUID
    var name: String
    var email: String
}

final class NetworkManager {                  // class: shared identity + lifecycle
    static let shared = NetworkManager()
    private init() {}
}
```

Mark every reference type `final` unless it is explicitly designed for subclassing — it enables devirtualization and signals intent. Prefer `enum` for closed sets of states (`enum LoadingState { case idle, loading, loaded(Data), failed(Error) }`).

### B. Optionals & safety

Model absence with `Optional`; unwrap explicitly. Never force-unwrap in production.

```swift
guard let order, !order.items.isEmpty else {   // early-exit, keeps happy path un-nested
    throw OrderError.empty
}
let name = user?.profile?.name ?? "Anonymous"  // optional chaining + nil-coalescing
if let token = session?.token, token.isValid { /* ... */ }
```

Footguns → fixes:
- Force-unwrap `value!` → `guard let` / `??`. Banned by SWIFT-SAFE-01.
- Implicitly-unwrapped `var x: T!` → only for IBOutlets or two-phase init that is *guaranteed* set before use.
- `try?` swallowing errors → use `do/catch` unless `nil` truly means "absent" (see `error-handling.md`).

### C. Protocol-oriented programming

Compose behavior from focused protocols with default implementations; use them as ports for dependency injection (the Swift binding of [`hexagonal.md`](guides://hexagonal.md)).

```swift
protocol UserRepository: Sendable {                 // a "port"
    func user(id: UUID) async throws -> User
}

protocol Identifiable { associatedtype ID: Hashable; var id: ID { get } }

extension Sequence where Element: Identifiable {    // protocol-constrained extension
    func indexed() -> [Element.ID: Element] {
        Dictionary(uniqueKeysWithValues: map { ($0.id, $0) })
    }
}
```

Use `some`/`any` deliberately: `some P` (opaque, static dispatch, one concrete type) for returns; `any P` (existential, dynamic dispatch) only when you must store heterogeneous conformers. Prefer protocol witnesses over class inheritance.

### D. Generics

Constrain with `where`; use `associatedtype` and primary associated types (`Collection<Element>`) for expressive, type-safe APIs.

```swift
func first<C: Collection>(in c: C, where p: (C.Element) -> Bool) -> C.Element? {
    c.first(where: p)
}

func decode<T: Decodable>(_ type: T.Type, from data: Data) throws -> T {
    try JSONDecoder().decode(T.self, from: data)
}
```

### E. Error handling — throws vs Result

Strategy (when to recover vs propagate) is owned by [`error-handling.md`](guides://error-handling.md). Swift binding: prefer `throws` for synchronous/`async` flows; use `Result` only to *store* or transport an outcome (e.g. completion handlers, caching a fetch). Model domain errors as `enum: Error`; conform to `LocalizedError` only at the boundary that shows them to a user.

```swift
enum NetworkError: Error, Equatable {
    case unauthorized, timeout, badStatus(Int)
}

func fetchUser(id: UUID) async throws -> User {
    let (data, response) = try await session.data(from: endpoint(id))
    guard let http = response as? HTTPURLResponse else { throw NetworkError.timeout }
    switch http.statusCode {
    case 200...299: return try JSONDecoder().decode(User.self, from: data)
    case 401:       throw NetworkError.unauthorized
    default:        throw NetworkError.badStatus(http.statusCode)
    }
}
```

Swift 6 supports typed throws (`func f() throws(NetworkError)`) — use it where the error set is closed and stable. Catch specific cases (`catch NetworkError.unauthorized`) before a generic `catch`.

### F. Structured concurrency

Policy (race-freedom, cancellation, structured lifetimes) is owned by [`parallelism.md`](guides://parallelism.md). Swift binding under Swift 6 **strict concurrency**:

```swift
// Parallel children, structured: both awaited or both cancelled together.
func profile(id: UUID) async throws -> Profile {
    async let user = fetchUser(id: id)
    async let posts = fetchPosts(of: id)
    return try await Profile(user: user, posts: posts)
}

// Actor: serializes access to mutable state — no manual locks.
actor ImageCache {
    private var store: [URL: Data] = [:]
    func data(for url: URL) -> Data? { store[url] }
    func insert(_ d: Data, for url: URL) { store[url] = d }
}

// Dynamic fan-out with cooperative cancellation.
func download(_ urls: [URL]) async throws -> [Data] {
    try await withThrowingTaskGroup(of: Data.self) { group in
        for url in urls { group.addTask { try await fetch(url) } }
        return try await group.reduce(into: []) { $0.append($1) }
    }
}
```

Rules: cross actor/`Task` boundaries only with `Sendable` types; honor cancellation (`try Task.checkCancellation()` / `Task.isCancelled`); isolate UI state to `@MainActor` (UI specifics → `ios.md`); never `@unchecked Sendable` without a written invariant. Reserve raw `Task.detached` and GCD (`DispatchQueue`) for legacy interop.

### G. ARC & memory

ARC is deterministic — break reference cycles explicitly. Use `[weak self]` when the captured object may outlive the closure (callbacks, timers, stored closures); `[unowned self]` only when lifetime is *guaranteed* (e.g. child→parent back-reference).

```swift
timer = Timer.scheduledTimer(withTimeInterval: 1, repeats: true) { [weak self] _ in
    self?.tick()                       // no retain cycle
}
deinit { timer?.invalidate() }
```

Footguns: strong `self` in an escaping closure stored on `self` → cycle; capturing the whole object when one field suffices (`let id = user.id`) → capture the field. Value types do not participate in cycles — another reason to prefer them.

### H. Codable

Synthesize `Codable` on value types; override `CodingKeys` for wire-format mapping, custom `init(from:)` only when the JSON shape diverges.

```swift
struct Event: Codable {
    let id: UUID
    let createdAt: Date
    enum CodingKeys: String, CodingKey { case id, createdAt = "created_at" }
}
let decoder = JSONDecoder()
decoder.dateDecodingStrategy = .iso8601          // configure once, reuse
decoder.keyDecodingStrategy = .convertFromSnakeCase   // alternative to CodingKeys
```

### I. Testing — Swift Testing & XCTest

Test policy (Red-Green-Refactor, regression-first) is owned by [`tdd.md`](guides://tdd.md). Swift binding: prefer the modern **Swift Testing** framework (`@Test`, `#expect`, `#require`) for new code; XCTest remains valid for existing suites and UI tests. Inject protocol dependencies for fakes/mocks (§5.C).

```swift
import Testing
@testable import MyPackage

@Test func validatesEmail() throws {
    #expect(try EmailValidator.validate("a@b.com") == "a@b.com")
}

@Test(arguments: ["invalid", "", "a @b.com"])     // parameterized — one case each
func rejectsBadEmail(_ input: String) {
    #expect(throws: EmailValidationError.self) { try EmailValidator.validate(input) }
}
```

Run a subset with `swift test --filter`. For async code, `await` directly inside `@Test`; no expectations boilerplate.

> For design patterns applied in Swift, reference [`designpatterns.md`](guides://designpatterns.md) and show only the Swift binding (protocol witnesses, enums-with-associated-values, value-type strategies).

---

## 6. Tooling & Dependencies

Security/supply-chain *policy* → [`secure-coding.md`](guides://secure-coding.md); versioning → [`semver.md`](guides://semver.md). Swift binding via SwiftPM:

```bash
swift build                       # compile
swift test                        # run tests
swift package resolve             # resolve deps → Package.resolved (commit it)
swift package update              # update to latest allowed versions
swift package show-dependencies   # inspect the graph
snyk test    # or trivy fs .      # SWIFT-SEC-01: CVE scan (SPM has no native audit)
```

Pin direct deps tightly in `Package.swift` (`.upToNextMinor(from:)` or `.exact(_:)` for security-sensitive packages); avoid `.upToNextMajor` for transitive-heavy deps. Commit `Package.resolved`. Review new packages before adding. Store secrets in the Keychain, never in source.

```swift
// Package.swift
// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "MyApp",
    platforms: [.iOS(.v17), .macOS(.v14)],
    dependencies: [
        .package(url: "https://github.com/apple/swift-log.git", .upToNextMinor(from: "1.6.0")),
    ],
    targets: [
        .target(name: "MyApp", dependencies: [.product(name: "Logging", package: "swift-log")],
                swiftSettings: [.enableUpcomingFeature("StrictConcurrency")]),
        .testTarget(name: "MyAppTests", dependencies: ["MyApp"]),
    ]
)
```

---

## 7. Quick Reference

```bash
swift build                          # build
swift test                           # test
swiftlint --strict                   # lint
swiftformat .                        # format
swift run                            # run executable
swift package generate-documentation # DocC docs
```

```swift
guard let value else { return }      // optional unwrap, early exit
let x = optional ?? fallback         // nil-coalescing
async let a = work(); try await a    // structured parallel child
actor Box { private var v = 0 }      // race-safe mutable state
```

---

## 8. Deployment Checklist

Generated from §2 — one box per requirement ID.

- [ ] SWIFT-FMT-01 — `swiftformat --lint` clean
- [ ] SWIFT-LINT-01 — `swiftlint --strict` clean
- [ ] SWIFT-TYP-01 — builds warning-free (`-warnings-as-errors`)
- [ ] SWIFT-SAFE-01 — no force-unwrap/force-try/force-cast in production
- [ ] SWIFT-CONC-01 — Swift 6 strict concurrency, 0 `Sendable` warnings
- [ ] SWIFT-TST-01/02/03 — tests pass, bugs have regression tests, coverage ≥ gate
- [ ] SWIFT-DOC-01 — public APIs documented, DocC builds
- [ ] SWIFT-SEC-01 — 0 high/critical CVEs in deps
- [ ] SWIFT-DEP-01 — `Package.resolved` in sync, committed
- [ ] Agent ran every §3 command and documented any fixes

---
**End of Swift Guidelines**
