# Java Development Guidelines
Mandatory coding standards for modern Java: minimal, immutable, test-covered, framework-independent domains. Java 21 LTS (25 where available), Gradle (preferred) / Maven, JUnit 5, SpotBugs, Error Prone, Checkstyle.

---
name: java
title: Java Development Guidelines
version: 2.0
last_reviewed: 2026-06-05
kind: language
tools: [java@21, gradle@8, maven@3.9, junit@5, spotbugs, error-prone, checkstyle, jacoco, owasp-dependency-check]
requires:
  - tdd
  - hexagonal
  - secure-coding
  - error-handling
recommends:
  - designpatterns
  - logging
  - observability
  - comments
  - semver
provides:
  - modern-java
  - records-sealed
  - virtual-threads
  - streams-optional
  - java-build
---

> 🧭 Authored per [`CONVENTIONS.md`](guides://CONVENTIONS.md): shared concerns are referenced, not restated. This guide covers only what is unique to Java.

---

## 0. Prerequisites & References

Fetch and apply these **before** generating Java code. Their rules are assumed below and not repeated.

> 📎 **REQUIRED — fetch & apply first:**
> - [`tdd.md`](guides://tdd.md) — test-first, Red-Green-Refactor, regression-test-before-fix, coverage. *(Java binding: JUnit 5 via `./gradlew test` / `mvn test`; coverage gate via JaCoCo.)*
> - [`hexagonal.md`](guides://hexagonal.md) — layering, ports/adapters, dependency inversion. *(Java binding: ports are interfaces; constructor injection; domain package free of framework imports.)*
> - [`secure-coding.md`](guides://secure-coding.md) — supply chain, secrets, CVE policy. *(Java binding: OWASP Dependency-Check, dependency locking, signed artifacts.)*
> - [`error-handling.md`](guides://error-handling.md) — error strategy & propagation. *(Java binding: §6 checked-vs-unchecked rules.)*

> 📎 **RECOMMENDED — fetch when the task touches them:**
> - [`designpatterns.md`](guides://designpatterns.md) — GoF & friends *(binding: §7 shows the modern-Java form — lambdas, sealed types, enums.)*
> - [`comments.md`](guides://comments.md) — API-doc policy *(binding: JavaDoc on public APIs.)*
> - [`logging.md`](guides://logging.md) · [`observability.md`](guides://observability.md) *(binding: SLF4J facade; Micrometer for metrics/tracing.)*
> - [`parallelism.md`](guides://parallelism.md) — concurrency policy *(binding: virtual threads, structured concurrency.)*
> - [`semver.md`](guides://semver.md)

> 📎 **SEE ALSO:** [`kotlin.md`](guides://kotlin.md) · [`scala.md`](guides://scala.md) · [`performance.md`](guides://performance.md) · [`cleanarch.md`](guides://cleanarch.md) · [`ci-cd.md`](guides://ci-cd.md)

---

## 1. Core Philosophies: JAVA-FIRST

Java-specific principles only. TDD, security, error handling, architecture, and design patterns come from §0.

- **J**VM-modern: target the current LTS (Java 21; use 25 LTS where the platform allows). Adopt records, sealed types, pattern matching, and switch expressions over legacy boilerplate.
- **A**bsence of mutation: immutable data by default — `record` carriers, `final` fields, `List.copyOf`/`Map.copyOf` for defensive snapshots.
- **V**alue expressiveness: streams, `Optional`, and functional interfaces over manual loops and null returns — without sacrificing readability.
- **A**nalysis-clean: zero SpotBugs / Error Prone / Checkstyle findings; compile with `-Xlint:all -Werror`.

**Verified Code**: Agent-generated Java MUST pass every gate in §2 before delivery.

---

## 2. Requirements (MANDATORY, auditable)

RFC-2119 keywords. IDs `JAVA-<TOPIC>-<NN>`. Each row has a binary gate; rows binding a shared rule cite its owner. Commands show Gradle then Maven.

| ID | Requirement | Verify | Gate |
|----|-------------|--------|------|
| JAVA-TST-01 | Every feature MUST be test-first (see `tdd.md`) | `./gradlew test` / `mvn test` | exit 0, 0 skips |
| JAVA-TST-02 | Each bug MUST get a regression test before the fix (see `tdd.md`) | `./gradlew test` / `mvn test` | failing→passing |
| JAVA-TST-03 | Domain/business-logic coverage MUST meet the project gate | `./gradlew jacocoTestCoverageVerification` | meets threshold |
| JAVA-FMT-01 | Code MUST be formatted | `./gradlew spotlessCheck` / `mvn spotless:check` | no diff |
| JAVA-LINT-01 | Style checks MUST pass clean | `./gradlew checkstyleMain` / `mvn checkstyle:check` | 0 violations |
| JAVA-LINT-02 | Static bug analysis MUST pass clean | `./gradlew spotbugsMain` + Error Prone on `compileJava` | 0 findings |
| JAVA-WARN-01 | Code MUST compile warning-clean | `javac -Xlint:all -Werror` (build flag) | exit 0 |
| JAVA-NULL-01 | Public APIs MUST declare nullability; no unchecked null deref | `@NonNull`/`@Nullable` + Error Prone/NullAway | 0 NPE findings |
| JAVA-DOC-01 | Public APIs MUST have JavaDoc (see `comments.md`) | `./gradlew javadoc` / `mvn javadoc:javadoc` | builds, 0 warnings |
| JAVA-SEC-01 | 0 high/critical CVEs in deps (see `secure-coding.md`) | `./gradlew dependencyCheckAnalyze` | 0 HIGH/CRITICAL |
| JAVA-DEP-01 | Lockfile in sync & verified (see `secure-coding.md`) | `./gradlew dependencies --write-locks` then diff | no drift |
| JAVA-ARCH-01 | Domain imports no adapter/framework code (see `hexagonal.md`) | ArchUnit test / review | no inward→outward |

> **Forbidden**: shipping implementation before its test (violates `tdd.md`); fixing a bug without a regression test first; `@Disabled` on a failing test; suppressing warnings instead of fixing the root cause; mutable static state; returning `null` from a method that could return `Optional`; checked exceptions across a port boundary (see `error-handling.md`).

---

## 3. Verification Protocol

Run, in order, before presenting code. Fix → re-run until every gate is green. (Gradle shown; Maven equivalents in parentheses.)

```bash
./gradlew spotlessCheck                 # JAVA-FMT-01   (mvn spotless:check)
./gradlew checkstyleMain spotbugsMain   # JAVA-LINT-01/02
./gradlew compileJava                   # JAVA-WARN-01/NULL-01 (-Xlint:all -Werror + Error Prone)
./gradlew test jacocoTestCoverageVerification   # JAVA-TST-01/02/03  (mvn verify)
./gradlew javadoc                       # JAVA-DOC-01   (mvn javadoc:javadoc)
./gradlew dependencyCheckAnalyze        # JAVA-SEC-01
./gradlew dependencies --write-locks    # JAVA-DEP-01 (then verify no diff)
```

The *why* behind each gate lives in its §0 owner; do not re-derive it here.

---

## 4. Project Structure

Idiomatic Maven/Gradle layout. Architectural principles (dependency direction, ports/adapters, acyclic deps) are owned by [`hexagonal.md`](guides://hexagonal.md); below is only their Java mapping.

```
project/
├── src/main/java/com/example/
│   ├── domain/          # pure business logic — no framework/Spring/JPA imports (JAVA-ARCH-01)
│   │   ├── model/       # records & sealed types
│   │   └── port/        # in/ (use-case interfaces) + out/ (repository interfaces)
│   ├── application/     # use-case implementations; orchestrate ports
│   └── adapter/         # in/ (web, cli) + out/ (persistence, external) implementing ports
├── src/test/java/...    # mirrors main/ (see tdd.md); ArchUnit guards boundaries
├── build.gradle.kts     # or pom.xml — single build/dep manifest
├── gradle/libs.versions.toml   # version catalog (Gradle)
└── README.md
```

- Group by domain/feature, not by technical layer-as-package-root.
- Enforce the import boundary with an **ArchUnit** test (`noClasses().that().resideInAPackage("..domain..").should().dependOnClassesThat().resideInAPackage("..adapter..")`).
- Ports are interfaces in `domain`; adapters depend inward only. Wire with **constructor injection** (no field `@Autowired`).

---

## 5. Java Specifics

The unique value of this guide. Modern Java (Java 21 LTS+) idioms that exist in no shared guide.

### A. Records — immutable data carriers
Replace JavaBeans/POJO boilerplate. Validate in the compact constructor; add derived behaviour as methods.

```java
public record User(Long id, String username, String email, Instant createdAt) {
    public User {                                   // compact constructor: validate + normalize
        Objects.requireNonNull(username, "username");
        if (!email.contains("@")) throw new IllegalArgumentException("invalid email");
    }
    public User withEmail(String newEmail) { return new User(id, username, newEmail, createdAt); }
}
```
Use records for DTOs, value objects, and multi-return tuples. Do **not** add mutable fields; for builders of wide configs use a static nested `Builder` (records have no setters).

### B. Sealed types + exhaustive pattern matching
Model closed hierarchies (results, states, AST nodes) with `sealed`; the compiler then proves the `switch` is exhaustive — no `default` needed.

```java
sealed interface Shape permits Circle, Rectangle {}
record Circle(double r) implements Shape {}
record Rectangle(double w, double h) implements Shape {}

double area(Shape s) {
    return switch (s) {                              // exhaustive: no default branch
        case Circle c        -> Math.PI * c.r() * c.r();
        case Rectangle r     -> r.w() * r.h();
    };
}
```
Record deconstruction patterns and guards (`case Circle c when c.r() > 0 -> ...`) replace `instanceof`-cast chains. Prefer a `sealed` `Result`/`Either` to nulls for expected failure modes (the error *strategy* is owned by [`error-handling.md`](guides://error-handling.md)).

### C. Streams, Optional & functional idioms
Declarative pipelines over manual loops; `Optional` instead of `null` returns.

```java
List<String> activeEmails = users.stream()
    .filter(User::isActive)
    .map(User::email)
    .filter(e -> e != null && !e.isBlank())
    .distinct().sorted()
    .toList();                                       // unmodifiable (Java 16+)

Optional<User> u = repo.findById(id);                // never return null
String name = u.map(User::username).orElse("anon");
```
Footguns: never call `Optional.get()` without `isPresent`; don't use `Optional` for fields/parameters/collections (use it for return types); avoid side effects inside `map`/`filter`; reach for `parallelStream()` only for CPU-bound work on large, splittable sources — never for I/O (use virtual threads, §D). Concurrency *policy* is owned by [`parallelism.md`](guides://parallelism.md).

### D. Virtual threads & structured concurrency
For I/O-bound fan-out, use virtual threads (Java 21+) — cheap, one-per-task — instead of bounded platform-thread pools. Use `StructuredTaskScope` to bound lifetimes and propagate cancellation/errors.

```java
List<Result> results;
try (var scope = new StructuredTaskScope.ShutdownOnFailure()) {
    var tasks = items.stream()
        .map(it -> scope.fork(() -> fetch(it)))      // each runs on a virtual thread
        .toList();
    scope.join().throwIfFailed();                    // bounded; fails fast
    results = tasks.stream().map(Subtask::get).toList();
}
```
Or `Executors.newVirtualThreadPerTaskExecutor()` with `CompletableFuture` for ad-hoc async. Do **not** pool virtual threads, and avoid `synchronized` blocks around blocking I/O on them (pinning) — use `ReentrantLock`.

### E. Null-safety
Annotate APIs with JSR-305 / JSpecify (`@NonNull`/`@Nullable`) and enforce with **Error Prone + NullAway** at compile time (JAVA-NULL-01). Prefer `Optional` returns, `Objects.requireNonNull` at boundaries, and `record` invariants over scattered null checks.

### F. Modern language conveniences
Text blocks (`"""`) for multi-line literals (SQL/JSON); `var` for local inference where the type is obvious; enhanced `switch` (arrow form, no fall-through); enum singletons for thread-safe, serialization-safe singletons.

> For the generic GoF catalogue, reference [`designpatterns.md`](guides://designpatterns.md). The Java binding: prefer **lambdas/functional interfaces** over Strategy classes, **enums** for fixed strategy registries, **sealed interfaces** for state/visitor hierarchies, and **`java.util.concurrent.Flow`** (or Reactor if already in-stack) over the deprecated `java.util.Observable`.

---

## 6. Error Handling — Java binding

The error *strategy* (when to fail fast, wrap, propagate, or recover) is owned by [`error-handling.md`](guides://error-handling.md). Java-specific bindings only:

- **Checked vs unchecked:** use **unchecked** exceptions for programming errors and domain rule violations (`IllegalArgumentException`, `IllegalStateException`, custom `RuntimeException` subtypes). Reserve **checked** exceptions for recoverable, caller-actionable conditions — and do not leak them across a port/use-case boundary; wrap at the adapter edge.
- **Never** swallow (`catch (Exception e) {}`), catch `Throwable`/`Error`, or use exceptions for control flow.
- Always clean up with **try-with-resources** (`AutoCloseable`); never rely on `finally` for closing.
- Preserve cause chains (`throw new XException("context", e)`); add context, don't discard the original.
- For *expected* failure modes prefer a `sealed Result`/`Optional` (§5.B/C) over throwing.

---

## 7. Tooling & Build

Security/supply-chain *policy* → [`secure-coding.md`](guides://secure-coding.md); versioning → [`semver.md`](guides://semver.md). **Prefer Gradle (Kotlin DSL)**; use Maven only when the project already uses it.

```kotlin
// build.gradle.kts
plugins { java; jacoco; checkstyle; id("com.github.spotbugs") version "..."; id("com.diffplug.spotless") version "..." }

java { toolchain { languageVersion = JavaLanguageVersion.of(21) } }   // pin LTS

dependencies {
    implementation(platform("org.springframework.boot:spring-boot-dependencies:3.x"))  // BOM
    errorprone("com.uber.nullaway:nullaway:...")
    testImplementation("org.junit.jupiter:junit-jupiter")
    testImplementation("com.tngtech.archunit:archunit-junit5:...")     // JAVA-ARCH-01
}

tasks.test { useJUnitPlatform(); finalizedBy(tasks.jacocoTestReport) }
tasks.compileJava { options.compilerArgs.addAll(listOf("-parameters", "-Xlint:all", "-Werror")) }  // JAVA-WARN-01
dependencyLocking { lockAllConfigurations() }                          // JAVA-DEP-01
```

Build commands:

```bash
./gradlew dependencies --write-locks    # JAVA-DEP-01: update + verify lockfiles (gradle.lockfile)
./gradlew dependencyCheckAnalyze        # JAVA-SEC-01: OWASP CVE scan, 0 HIGH/CRITICAL
./gradlew spotbugsMain checkstyleMain   # JAVA-LINT-01/02
```

- Use a **version catalog** (`gradle/libs.versions.toml`) or a **BOM**; pin direct deps, let the resolver fix the graph.
- Commit lockfiles. Verify artifact checksums/signatures for supply-chain integrity (policy: `secure-coding.md`).
- Logging via the **SLF4J** facade (binding for [`logging.md`](guides://logging.md)); metrics/tracing via **Micrometer** (binding for [`observability.md`](guides://observability.md)) — do not log to `System.out`.

---

## 8. Quick Reference

```bash
./gradlew build                              # build (mvn package)
./gradlew test jacocoTestReport              # test + coverage (mvn verify)
./gradlew checkstyleMain spotbugsMain        # lint + static analysis
./gradlew spotlessApply                      # format
./gradlew dependencyCheckAnalyze             # CVE scan
./gradlew javadoc                            # API docs
./gradlew bootRun  /  java -jar app.jar      # run
```

```java
// modern-Java cheat sheet
record Point(int x, int y) {}                                   // immutable carrier
sealed interface Shape permits Circle, Square {}                // closed hierarchy
var msg = switch (s) { case Circle c -> "round"; case Square q -> "square"; };  // exhaustive
try (var scope = new StructuredTaskScope.ShutdownOnFailure()) { /* virtual threads */ }
String sql = """
    SELECT * FROM users WHERE active = true""";                 // text block
```

---

## 9. Deployment Checklist

Generated from §2 — one box per requirement ID. No new requirements here.

- [ ] JAVA-FMT-01 — `spotlessCheck` clean, no diff
- [ ] JAVA-LINT-01/02 — Checkstyle + SpotBugs/Error Prone clean
- [ ] JAVA-WARN-01 — compiles `-Xlint:all -Werror`
- [ ] JAVA-NULL-01 — nullability annotated, NullAway clean
- [ ] JAVA-TST-01/02/03 — tests pass, bugs have regression tests, coverage meets gate
- [ ] JAVA-DOC-01 — public APIs have JavaDoc, builds 0 warnings
- [ ] JAVA-SEC-01 — OWASP Dependency-Check 0 HIGH/CRITICAL
- [ ] JAVA-DEP-01 — lockfiles in sync & committed
- [ ] JAVA-ARCH-01 — domain free of adapter/framework imports (ArchUnit)
- [ ] Agent ran every §3 command and documented any fixes

---
**End of Java Guidelines**
