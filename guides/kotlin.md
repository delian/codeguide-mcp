# Kotlin Development Guidelines
Mandatory coding standards for Kotlin: null-safe, immutable, coroutine-driven, minimal boilerplate. Kotlin 2.x, Gradle (Kotlin DSL), JUnit 5 / Kotest, detekt, ktlint, kotlinx-coroutines, kotlinx-serialization, Dokka.

---
name: kotlin
title: Kotlin Development Guidelines
version: 2.0
last_reviewed: 2026-06-05
kind: language
tools: [kotlin@2.1, gradle@8, junit@5, kotest@5.9, detekt@1.23, ktlint@1.4, dokka@2]
requires:
  - tdd
  - hexagonal
  - secure-coding
  - error-handling
recommends:
  - java
  - designpatterns
  - parallelism
  - comments
provides:
  - null-safety
  - coroutines-flows
  - data-sealed-classes
  - kotlin-idioms
---

> 🧭 Authored per [`CONVENTIONS.md`](guides://CONVENTIONS.md): shared concerns are referenced, not restated. This guide covers only what is unique to Kotlin.

---

## 0. Prerequisites & References

Fetch and apply these **before** generating Kotlin code. Their rules are assumed below and not repeated.

> 📎 **REQUIRED — fetch & apply first:**
> - [`tdd.md`](guides://tdd.md) — test-first, Red-Green-Refactor, regression-test-before-fix, coverage. *(Kotlin binding: `./gradlew test` with Kotest/JUnit 5; coroutine tests use `runTest`.)*
> - [`hexagonal.md`](guides://hexagonal.md) — layering, ports/adapters, dependency inversion. *(Kotlin mapping in §4.)*
> - [`secure-coding.md`](guides://secure-coding.md) — supply chain, secrets, CVE policy. *(Kotlin binding: `detekt`, `dependencyCheckAnalyze`, Gradle dependency locking.)*
> - [`error-handling.md`](guides://error-handling.md) — error vs exception models, Result/Either, propagation. *(Kotlin binding in §6: sealed-class results, `runCatching`/`kotlin.Result`, exceptions across the boundary.)*

> 📎 **RECOMMENDED — fetch when the task touches them:**
> - [`java.md`](guides://java.md) — JVM toolchain, GC, Java interop, virtual threads *(Kotlin runs on the JVM and consumes Java libraries — see §7.)*
> - [`parallelism.md`](guides://parallelism.md) — concurrency models, races, structured concurrency *(Kotlin binding: coroutines/Flow in §8.)*
> - [`designpatterns.md`](guides://designpatterns.md) — GoF & functional patterns *(show only the Kotlin binding; many patterns collapse to idioms — see §5.D.)*
> - [`comments.md`](guides://comments.md) — API-doc policy *(binding: KDoc + Dokka.)*

> 📎 **SEE ALSO:** [`mutmut.md`](guides://mutmut.md) (mutation-testing concept; Kotlin tool is Pitest) · [`performance.md`](guides://performance.md) · [`semver.md`](guides://semver.md) · [`ci-cd.md`](guides://ci-cd.md) · [`pre-commit.md`](guides://pre-commit.md)

---

## 1. Core Philosophies: MINIMAL-KOTLIN

Kotlin-specific principles only. TDD, security, error handling, and architecture come from §0.

- **M**inimal boilerplate: top-level functions and `data`/`value` classes over ceremony classes; let the compiler synthesize `equals`/`hashCode`/`copy`.
- **I**mmutable by default: `val` over `var`; read-only collection types (`List`, `Map`) at APIs; `data class copy()` instead of in-place mutation.
- **N**ull safety is a type-system feature: non-null by default; nullability is explicit (`T?`) and resolved with `?.`/`?:`/`let`, never `!!`.
- **I**diomatic expressions: `when`/`if`/`try` as expressions; scope functions (`let`/`run`/`also`/`apply`/`with`) used for their distinct intent, not stacked.
- **M**odern concurrency: structured `suspend`/coroutines and cold `Flow` over threads and callbacks (binding for `parallelism.md`).
- **A**rchitecture: hexagonal ports/adapters (owned by `hexagonal.md`); domain is pure Kotlin with no framework imports.
- **L**azy where it pays: `Sequence` for multi-step pipelines on large data; `by lazy` for expensive once-only init.

**Verified Code**: Agent-generated Kotlin MUST pass every gate in §2 before delivery.

---

## 2. Requirements (MANDATORY, auditable)

RFC-2119 keywords. IDs `KT-<TOPIC>-<NN>`. Each row has a binary gate; rows binding a shared rule cite its owner.

| ID | Requirement | Verify | Gate |
|----|-------------|--------|------|
| KT-TST-01 | Every feature MUST be test-first (see `tdd.md`) | `./gradlew test` | exit 0, 0 skips |
| KT-TST-02 | Each bug MUST get a regression test before the fix (see `tdd.md`) | `./gradlew test` | failing→passing |
| KT-TST-03 | Business-logic coverage MUST meet the project gate | `./gradlew koverVerify` | exit 0 |
| KT-FMT-01 | Code MUST be formatted | `./gradlew ktlintCheck` | no diff |
| KT-LINT-01 | Static analysis MUST pass clean | `./gradlew detekt` | exit 0 |
| KT-NULL-01 | No `!!`, no platform-type leaks across public APIs | `detekt` (`UnsafeCallOnNullableType`) / review | 0 findings |
| KT-COMPILE-01 | Code MUST compile warning-free (K2) | `./gradlew compileKotlin -Pwerror` | exit 0, 0 warnings |
| KT-DOC-01 | Public APIs MUST have KDoc (see `comments.md`) | `./gradlew dokkaHtml` | builds, 0 missing-doc warnings |
| KT-SEC-01 | 0 high/critical dependency CVEs (see `secure-coding.md`) | `./gradlew dependencyCheckAnalyze` | 0 high/critical |
| KT-DEP-01 | Lockfile in sync & verified (see `secure-coding.md`) | `./gradlew dependencies --write-locks` then diff | no change, committed |
| KT-ARCH-01 | Domain imports no adapter/framework code (see `hexagonal.md`) | Konsist / ArchUnit / review | no inward→outward |
| KT-ERR-01 | Failures modeled deliberately (see `error-handling.md`) | review | sealed result or typed exception at boundary |
| KT-CONC-01 | Coroutines structured & cancellation-cooperative (see `parallelism.md`) | review / `runTest` | no `GlobalScope`, cancellation honored |

> **Forbidden**: shipping implementation before its test (violates `tdd.md`), fixing a bug without a regression test first, `!!` to silence nullability, `GlobalScope.launch`, swallowing `CancellationException`, blocking calls inside `suspend` without `Dispatchers.IO`, or `@Disabled`/ignored tests to make a build green.

---

## 3. Verification Protocol

Run, in order, before presenting code. Fix → re-run until every gate is green.

```bash
./gradlew ktlintCheck                 # KT-FMT-01
./gradlew detekt                      # KT-LINT-01, KT-NULL-01
./gradlew compileKotlin -Pwerror      # KT-COMPILE-01 (K2, warnings as errors)
./gradlew test koverVerify            # KT-TST-01/02/03
./gradlew dokkaHtml                   # KT-DOC-01
./gradlew dependencyCheckAnalyze      # KT-SEC-01
./gradlew dependencies --write-locks  # KT-DEP-01 (then verify no diff)
```

The *why* behind each gate lives in its §0 owner; do not re-derive it here.

---

## 4. Project Structure

Idiomatic Gradle/JVM layout. Architectural principles (dependency direction, ports/adapters, acyclic deps) are owned by [`hexagonal.md`](guides://hexagonal.md); below is only their Kotlin mapping.

```
project/
├── build.gradle.kts            # Kotlin DSL build script
├── settings.gradle.kts
├── gradle/libs.versions.toml   # version catalog (single source of dep versions)
├── gradle.lockfile             # committed dependency lock (KT-DEP-01)
└── src/
    ├── main/kotlin/com/example/
    │   ├── domain/             # pure business logic — no framework/IO imports (KT-ARCH-01)
    │   │   ├── model/          # data/sealed classes, value classes
    │   │   └── port/           # in/out port interfaces (use `in`/`out` as escaped pkg names)
    │   ├── application/        # use cases, orchestrate ports (suspend functions)
    │   └── adapter/
    │       ├── in/             # driving adapters (web, cli)
    │       └── out/            # driven adapters (persistence, external)
    └── test/kotlin/com/example/   # mirrors main/ (see tdd.md)
```

- Group by domain/feature, not by type.
- Enforce the import boundary with **Konsist** or **ArchUnit** in a test (`KT-ARCH-01`), not just convention.
- Put domain logic in **top-level functions**; reserve classes for state/identity.

---

## 5. Kotlin Specifics

The unique value of this guide.

### A. Null safety — the type system, not runtime checks
Non-null is the default; `T?` is the explicit opt-in. Resolve with safe-call `?.`, Elvis `?:`, and `let`. **`!!` is banned** (`KT-NULL-01`) — it converts a type-system guarantee back into a runtime NPE.

```kotlin
fun displayName(user: User?): String =
    user?.name?.takeIf { it.isNotBlank() } ?: "Anonymous"

val cfg = lookup(key) ?: error("missing config: $key")   // fail fast instead of !!
val len: Int = nullable?.length ?: 0                       // never nullable!!.length
```

Guard **platform types** from Java interop at the boundary: annotate Java with `@Nullable`/`@NotNull` or wrap immediately so a `String!` never propagates. Enable `-Xjsr305=strict` so JSR-305 annotations are enforced.

### B. Data, sealed, and value classes
```kotlin
data class User(val id: UserId, val name: String, val email: Email)   // equals/hashCode/copy free

@JvmInline value class UserId(val raw: String)            // type-safe, zero-alloc wrapper over String
@JvmInline value class Email(val raw: String) {
    init { require("@" in raw) { "invalid email: $raw" } } // validate at construction
}

sealed interface PaymentState {                            // closed hierarchy → exhaustive `when`
    data object Pending : PaymentState
    data class Settled(val txId: String) : PaymentState
    data class Failed(val reason: String) : PaymentState
}

fun describe(s: PaymentState): String = when (s) {         // no `else` needed; new case = compile error
    PaymentState.Pending   -> "pending"
    is PaymentState.Settled -> "ok ${s.txId}"
    is PaymentState.Failed  -> "failed: ${s.reason}"
}
```
Prefer `data object`/`sealed interface` over enum when cases carry data. Value classes eliminate primitive-obsession bugs (passing a raw `String` user-id where an order-id was meant).

### C. Extension functions & scope functions
Extensions add behavior to types you don't own (incl. Java/stdlib) without inheritance or wrappers:
```kotlin
fun String.toSlug(): String = trim().lowercase().replace(Regex("\\s+"), "-")
val List<User>.active: List<User> get() = filter { it.isActive }
```
Use scope functions by their distinct contract — do **not** nest them into puzzles:

| Function | Receiver | Returns | Use for |
|---|---|---|---|
| `let` | `it` | lambda result | null-safe transform of a value |
| `run` | `this` | lambda result | configure-and-compute on a receiver |
| `apply` | `this` | the receiver | builder-style mutation, return the object |
| `also` | `it` | the receiver | side effects (logging) in a chain |
| `with` | `this` | lambda result | grouped calls on an existing object |

### D. Idioms that replace patterns
Reference [`designpatterns.md`](guides://designpatterns.md) for the catalog; in Kotlin many patterns collapse to language features — show only the binding:
- **Singleton** → `object`. **Strategy** → a function-typed parameter `(In) -> Out`. **Builder** → named/default args + `apply`. **Decorator** → extension functions or delegation `by`. **Observer** → `Flow`/`StateFlow`. **Visitor** → `sealed` + exhaustive `when`.
- Delegation: `class Repo(store: Store) : Store by store` forwards automatically; `val x by lazy { ... }` and `by Delegates.observable(...)` for properties.

### E. Immutability & collections
`val` for every binding that isn't reassigned; expose read-only `List`/`Map`/`Set` at APIs (the concrete type may be a `MutableList` internally — never leak it). Build new values with `copy()` and collection operators (`map`/`filter`/`associate`) rather than mutating.
```kotlin
fun promote(u: User): User = u.copy(role = Role.ADMIN)     // new object, no mutation
```
**Footgun:** `List` is read-only, not immutable — if you hand out a `MutableList` upcast to `List`, callers can still mutate via the original reference. Return `.toList()` to defensively copy when the source is mutable.

### F. KDoc — Dokka binding
Policy is owned by [`comments.md`](guides://comments.md). In Kotlin: KDoc (`/** ... */` with `@param`/`@return`/`@throws`/`@sample`) on every public/protected declaration; generate with `./gradlew dokkaHtml` (Dokka 2.x), keep generated HTML out of git. `@sample` references a real function so examples can't rot.

---

## 6. Error Handling — Kotlin binding

Strategy (error vs exception, propagate vs recover, retries/timeouts) is owned by [`error-handling.md`](guides://error-handling.md). Kotlin mechanics:

- **Expected, domain failures** → model as a `sealed`/`Result` type and force the caller to handle every case via exhaustive `when` (no checked exceptions exist in Kotlin, so the type system is your only compile-time guarantee).
  ```kotlin
  sealed interface Outcome<out T> {
      data class Ok<T>(val value: T) : Outcome<T>
      data class Err(val error: DomainError) : Outcome<Nothing>
  }
  ```
- **`kotlin.Result` / `runCatching`** → fine for wrapping a single throwing call and folding it (`runCatching { parse(x) }.getOrElse { default }`), but do **not** use it as a domain error channel or store it in fields (it cannot represent typed domain errors and swallows specifics). Prefer a domain sealed type for business errors.
- **Exceptions** → for truly exceptional, unrecoverable conditions and at the system boundary. Validate invariants with `require`/`check`/`error` (throw `IllegalArgumentException`/`IllegalStateException`). **Never catch `CancellationException`** to swallow it (see §8) — rethrow it.
- Use `try` as an expression: `val n = try { s.toInt() } catch (e: NumberFormatException) { 0 }`.

---

## 7. JVM & Java Interop — Java binding

JVM toolchain, GC, build target, and virtual threads are owned by [`java.md`](guides://java.md). Kotlin specifics:

- Pin the toolchain: `kotlin { jvmToolchain(21) }` — reproducible across machines.
- **Nullability across the boundary** is the main interop risk: Java returns *platform types* (`String!`) with no null info. Treat every Java return as nullable unless annotated; resolve immediately (§5.A). Annotate your Kotlin public API for Java consumers with `@JvmStatic`, `@JvmOverloads`, `@JvmName`, `@Throws` where the Java side must see them.
- `SAM` conversion lets you pass a Kotlin lambda where Java expects a functional interface. Kotlin `suspend` functions are **not** callable from Java — expose a `CompletableFuture`/blocking façade at the boundary if Java must call in.

---

## 8. Concurrency — Coroutines & Flow (parallelism binding)

Concurrency *policy* (races, deadlocks, structured concurrency, cancellation) is owned by [`parallelism.md`](guides://parallelism.md). Kotlin binding:

- **Structured concurrency only**: launch within a `coroutineScope`/`supervisorScope` or an injected `CoroutineScope` tied to a lifecycle. **`GlobalScope` is banned** (`KT-CONC-01`) — it leaks work that outlives its caller.
  ```kotlin
  suspend fun loadDashboard(id: UserId): Dashboard = coroutineScope {
      val profile = async { profilePort.fetch(id) }     // concurrent
      val orders  = async { orderPort.recent(id) }
      Dashboard(profile.await(), orders.await())          // child failure cancels siblings
  }
  ```
- **Cancellation is cooperative**: suspend functions check it automatically; in CPU loops call `ensureActive()`/`yield()`. Always rethrow `CancellationException` (don't catch-all swallow it).
- **Dispatchers**: `Dispatchers.IO` for blocking IO, `Default` for CPU work. Never block a coroutine thread — wrap blocking JVM calls in `withContext(Dispatchers.IO)`.
- **`Flow`** for cold async streams; `StateFlow`/`SharedFlow` for hot state/events. Set the dispatcher with `flowOn`, handle errors with `catch`, and collect inside a scope. Use `runTest` + `kotlinx-coroutines-test` (virtual time) for deterministic coroutine tests.

---

## 9. Tooling & Dependencies

Security/supply-chain *policy* → [`secure-coding.md`](guides://secure-coding.md); versioning → [`semver.md`](guides://semver.md). Kotlin/Gradle binding (Kotlin DSL, version catalog, dependency locking):

```kotlin
// build.gradle.kts
plugins {
    kotlin("jvm") version "2.1.0"
    kotlin("plugin.serialization") version "2.1.0"
    id("io.gitlab.arturbosch.detekt") version "1.23.7"
    id("org.jlleitschuh.gradle.ktlint") version "12.1.1"
    id("org.jetbrains.dokka") version "2.0.0"
    id("org.jetbrains.kotlinx.kover") version "0.9.1"   // coverage (KT-TST-03)
    id("org.owasp.dependencycheck") version "11.1.0"    // KT-SEC-01
}

kotlin {
    jvmToolchain(21)
    compilerOptions { allWarningsAsErrors = true }       // KT-COMPILE-01
}

dependencyLocking { lockAllConfigurations() }            // KT-DEP-01: reproducible builds

dependencies {
    implementation(libs.kotlinx.coroutines.core)         // versions from gradle/libs.versions.toml
    implementation(libs.kotlinx.serialization.json)
    testImplementation(libs.kotest.runner.junit5)
    testImplementation(libs.kotlinx.coroutines.test)
}

tasks.test { useJUnitPlatform() }                        // Kotest/JUnit 5
```

```bash
./gradlew dependencies --write-locks   # KT-DEP-01: regenerate lockfile after a dep change (commit it)
./gradlew dependencyCheckAnalyze       # KT-SEC-01: CVE scan
./gradlew detekt                       # KT-LINT-01 static analysis
```
Keep all versions in `gradle/libs.versions.toml`; commit `gradle.lockfile`. Use BOMs for coordinated library families.

---

## 10. Quick Reference

```bash
./gradlew build                                   # compile + test + checks
./gradlew test koverVerify                        # test + coverage gate
./gradlew ktlintCheck detekt                      # lint + static analysis
./gradlew ktlintFormat                            # auto-format
./gradlew dokkaHtml                               # API docs
./gradlew run                                      # run
```

```kotlin
// Idiom cheat-sheet
val id = UserId("u-1")                            // value class, type-safe
when (state) { is A -> …; is B -> … }             // exhaustive over sealed
user?.email?.substringAfter("@") ?: "unknown"     // null-safe chain, no !!
data.asSequence().filter { … }.map { … }.take(n)  // lazy pipeline, large data
coroutineScope { async { … }.await() }            // structured concurrency
class Repo(s: Store) : Store by s                  // delegation, no boilerplate
```

---

## 11. Deployment Checklist

Generated from §2 — one box per requirement ID.

- [ ] KT-FMT-01 — `ktlintCheck` clean
- [ ] KT-LINT-01 — `detekt` clean
- [ ] KT-NULL-01 — no `!!`, platform types resolved at the boundary
- [ ] KT-COMPILE-01 — K2 compile warning-free (`-Pwerror`)
- [ ] KT-TST-01/02/03 — tests pass, bugs have regression tests, coverage ≥ gate
- [ ] KT-DOC-01 — public APIs have KDoc, Dokka builds clean
- [ ] KT-SEC-01 — 0 high/critical CVEs
- [ ] KT-DEP-01 — `gradle.lockfile` in sync & committed
- [ ] KT-ARCH-01 — domain free of adapter/framework imports (Konsist/ArchUnit)
- [ ] KT-ERR-01 — failures modeled deliberately (sealed result / typed exception)
- [ ] KT-CONC-01 — structured coroutines, cancellation honored, no `GlobalScope`
- [ ] Agent ran every §3 command and documented any fixes

---
**End of Kotlin Guidelines**
