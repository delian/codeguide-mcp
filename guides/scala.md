# Scala Development Guidelines
Mandatory coding standards for Scala: type-safe, functional, effect-managed. Scala 3, sbt, ScalaTest/munit, scalafmt, scalafix, Cats Effect/ZIO.

---
name: scala
title: Scala Development Guidelines
version: 2.0
last_reviewed: 2026-06-05
kind: language
tools: [scala@3.7, sbt@1.10, scalatest@3.2, munit@1.0, scalafmt@3.8, scalafix@0.13]
requires:
  - tdd
  - secure-coding
  - error-handling
recommends:
  - hexagonal
  - java
  - designpatterns
  - parallelism
  - comments
provides:
  - scala3
  - fp-idioms
  - adts-pattern-matching
  - type-classes
  - effect-systems
---

> 🧭 Authored per [`CONVENTIONS.md`](guides://CONVENTIONS.md): shared concerns are referenced, not restated. This guide covers only what is unique to Scala.

---

## 0. Prerequisites & References

Fetch and apply these **before** generating Scala code. Their rules are assumed below and not repeated.

> 📎 **REQUIRED — fetch & apply first:**
> - [`tdd.md`](guides://tdd.md) — test-first, Red-Green-Refactor, regression-test-before-fix, coverage. *(Scala binding: runner is `sbt test`; ScalaTest or munit.)*
> - [`secure-coding.md`](guides://secure-coding.md) — supply chain, secrets, CVE policy. *(Scala binding: `sbt-dependency-check`, `sbt-updates`.)*
> - [`error-handling.md`](guides://error-handling.md) — error strategy & propagation. *(Scala binding: `Either`/`Try`/`Validated`, typed effect errors; never throw across pure boundaries.)*

> 📎 **RECOMMENDED — fetch when the task touches them:**
> - [`hexagonal.md`](guides://hexagonal.md) — ports/adapters *(binding: traits as ports, tagless final, adapter wiring at the edge)*
> - [`java.md`](guides://java.md) — JVM interop *(binding: wrapping Java APIs that throw/return null)*
> - [`parallelism.md`](guides://parallelism.md) — concurrency *(binding: fibers, `parTraverseN`, `Ref`, structured concurrency)*
> - [`designpatterns.md`](guides://designpatterns.md) — FP patterns *(binding: type classes, smart constructors, Reader)*
> - [`comments.md`](guides://comments.md) — Scaladoc on public APIs

> 📎 **SEE ALSO:** [`logging.md`](guides://logging.md) · [`observability.md`](guides://observability.md) · [`env-config.md`](guides://env-config.md) · [`semver.md`](guides://semver.md)

---

## 1. Core Philosophies: SCALA-FIRST

Scala-specific principles only. TDD, security, error handling, and architecture come from §0.

- **S**afe by types: make illegal states unrepresentable — ADTs, opaque types, smart constructors, exhaustive matches. The compiler is the first test.
- **C**omposable: build from small pure functions and total functions; prefer combinators (`map`/`flatMap`/`traverse`) over manual control flow.
- **A**lgebraic: model the domain with `enum`/sealed ADTs and pattern matching; encode behaviour with type classes (`given`/`using`), not inheritance.
- **L**azy & immutable: `val` over `var`, immutable collections by default; `lazy val`, `view`, and `LazyList` for deferred/streamed work.
- **A**bstract over effects: side effects are values (`IO`/`ZIO`), described and composed — never run eagerly inside pure code.

**Verified Code**: Agent-generated Scala MUST pass every gate in §2 before delivery.

---

## 2. Requirements (MANDATORY, auditable)

RFC-2119 keywords. IDs `SCALA-<TOPIC>-<NN>`. Each row has a binary gate; rows binding a shared rule cite its owner.

| ID | Requirement | Verify | Gate |
|----|-------------|--------|------|
| SCALA-TST-01 | Every feature MUST be test-first (see `tdd.md`) | `sbt test` | exit 0, 0 ignored |
| SCALA-TST-02 | Each bug MUST get a regression test before the fix (see `tdd.md`) | `sbt test` | failing→passing |
| SCALA-TST-03 | Business-logic coverage MUST meet the project gate | `sbt coverage test coverageReport` | ≥ threshold |
| SCALA-FMT-01 | Code MUST be formatted | `sbt scalafmtCheckAll` | no diff |
| SCALA-LINT-01 | Linter MUST pass clean | `sbt "scalafixAll --check"` | exit 0 |
| SCALA-TYP-01 | Compile MUST be warning-free | `sbt compile` with `-Werror` | exit 0 |
| SCALA-TYP-02 | Pattern matches MUST be exhaustive; no silent catch-all (see `error-handling.md`) | `-Werror` + review | no `case _ => ()` dropping variants |
| SCALA-ERR-01 | No partial ops in prod code: `.get`, `.head`, `asInstanceOf`, `null` (see `error-handling.md`) | `scalafix` Disable rule | 0 findings |
| SCALA-SEC-01 | 0 high/critical CVEs in deps (see `secure-coding.md`) | `sbt dependencyCheck` | 0 high/critical |
| SCALA-DEP-01 | Dependency versions pinned & resolvable | `sbt update` | exit 0, no `latest.*` |
| SCALA-DOC-01 | Public APIs documented (see `comments.md`) | `sbt doc` | builds clean |
| SCALA-ARCH-01 | Domain imports no adapter/framework code (see `hexagonal.md`) | review / module deps | no inward→outward |

> **Forbidden**: shipping implementation before its test (violates `tdd.md`), fixing a bug without a regression test first, `.get`/`asInstanceOf`/`null` to dodge the type checker, throwing exceptions across pure boundaries, or `var`/mutable collections where an immutable value suffices.

---

## 3. Verification Protocol

Run, in order, before presenting code. Fix → re-run until every gate is green.

```bash
sbt scalafmtCheckAll        # SCALA-FMT-01
sbt "scalafixAll --check"   # SCALA-LINT-01
sbt compile                 # SCALA-TYP-01/02  (-Werror turns warnings into failures)
sbt test                    # SCALA-TST-01/02
sbt coverage test coverageReport   # SCALA-TST-03 (sbt-scoverage)
sbt dependencyCheck         # SCALA-SEC-01
```

The *why* behind each gate lives in its §0 owner; do not re-derive it here.

---

## 4. Project Structure

Idiomatic multi-module sbt layout. Architectural principles (dependency direction, ports/adapters, acyclic deps) are owned by [`hexagonal.md`](guides://hexagonal.md); below is only their Scala mapping.

```
my-app/
├── build.sbt                 # versions, scalacOptions, module graph
├── project/
│   ├── build.properties      # sbt.version=1.10.x
│   └── plugins.sbt           # scalafix, scoverage, dependency-check, sbt-updates
├── .scalafmt.conf            # formatter config (version pinned)
├── modules/
│   ├── domain/   src/main/scala  # pure models, ADTs, ports (traits) — no framework imports (SCALA-ARCH-01)
│   ├── core/     src/main/scala  # shared types, config, effect aliases
│   ├── infra/    src/main/scala  # adapters: db/http/messaging implement ports
│   ├── api/      src/main/scala  # routes, codecs (DTOs separate from domain)
│   └── app/      src/main/scala  # Main, dependency wiring
└── */src/test/scala            # tests mirror sources (see tdd.md)
```

- Group by domain/feature; `domain` depends on nothing framework-shaped.
- The module graph (`dependsOn`) enforces the dependency direction — domain is a leaf.

---

## 5. Scala Specifics

The unique value of this guide. Scala 3 syntax throughout (`given`/`using`, `enum`, significant indentation).

### A. ADTs — `enum` and sealed hierarchies

Model domains as closed sets; the compiler then proves matches exhaustive (SCALA-TYP-02).

```scala
enum PaymentMethod:
  case CreditCard(number: String, expiry: String, cvv: String)
  case PayPal(email: String)
  case Crypto(wallet: String, currency: CryptoCurrency)

enum Tree[+A]:                     // parameterized / recursive ADT
  case Leaf(value: A)
  case Node(left: Tree[A], right: Tree[A])

case class User(id: UserId, name: String, email: Email, role: Role = Role.User):
  def withEmail(e: Email): User = copy(email = e)   // immutable update
```

Prefer composition of small case classes over one large class. Use `copy` for updates — never mutate.

### B. Opaque types & smart constructors

Zero-cost domain types: a `UserId` is a `Long` at runtime but a distinct type at compile time, so it cannot be swapped for an `Email`.

```scala
opaque type UserId = Long
object UserId:
  def apply(v: Long): UserId = v
  extension (id: UserId) def value: Long = id

opaque type Email = String
object Email:
  def from(v: String): Either[String, Email] =          // smart constructor validates
    if v.matches("""^[^\s@]+@[^\s@]+\.[^\s@]+$""") then Right(v.toLowerCase)
    else Left("Invalid email")
  extension (e: Email) def value: String = e

// Private-ctor case class is the alternative when you need methods + structural copy
final case class OrderQuantity private (value: Int)
object OrderQuantity:
  def from(v: Int): Either[String, OrderQuantity] =
    Either.cond(v >= 1 && v <= 10000, OrderQuantity(v), s"out of range: $v")
```

Validation lives in the constructor; the rest of the code receives only valid values. (Pattern owned by [`designpatterns.md`](guides://designpatterns.md).)

### C. `given`/`using` & type classes

Encode behaviour with type classes, not subtyping. `given` defines an instance, `using` requests one, `summon`/context bounds retrieve it.

```scala
trait JsonEncoder[A]:
  def encode(a: A): Json
  extension (a: A) def toJson: Json = encode(a)

object JsonEncoder:
  def apply[A](using e: JsonEncoder[A]): JsonEncoder[A] = e   // summoner
  given JsonEncoder[Int]    = i => Json.Num(i)
  given JsonEncoder[String] = s => Json.Str(s)
  given [A](using e: JsonEncoder[A]): JsonEncoder[List[A]] =  // derived instance
    xs => Json.Arr(xs.map(e.encode))

def sorted[A: Ordering](xs: List[A]): List[A] = xs.sorted    // context bound = `using Ordering[A]`
```

Scala 3 can `derives` type classes (`case class U(...) derives JsonCodec`). Be explicit on given imports (`import Foo.given`). Do not make arbitrary values `given` just for convenience — type classes only.

### D. Option / Either / Try

Total functions return `Option`/`Either`; never `null`. `Try` only wraps Java/throwing APIs at the boundary (see [`java.md`](guides://java.md)). Strategy is owned by [`error-handling.md`](guides://error-handling.md) — Scala bindings:

```scala
// Option: combinators, never .get
findUser(id).map(_.name).getOrElse("Unknown")
findUser(id).toRight(UserError.NotFound(id))             // lift into Either

// Either: model domain errors as an enum, chain with for-comprehension (short-circuits)
enum UserError:
  case NotFound(id: UserId); case InvalidEmail(s: String); case AlreadyExists(e: Email)

def createUser(name: String, raw: String): Either[UserError, User] =
  for
    email <- Email.from(raw).left.map(_ => UserError.InvalidEmail(raw))
    _     <- ensureUnique(email)
    user  <- save(User(UserId(nextId()), name, email))
  yield user

// Try: only at the Java/throwing boundary
def parse(raw: String): Either[ConfigError, Config] =
  Try(ConfigFactory.parseString(raw)).toEither.left.map(e => ConfigError.Parse(e.getMessage))

// Validated (Cats): ACCUMULATE errors instead of short-circuiting (e.g. form validation)
(validName(f.name), validEmail(f.email), validAge(f.age)).mapN(ValidatedUser.apply)
```

### E. Pattern matching

Exhaustive by default (SCALA-TYP-02). Use guards, destructuring, `@`-binds, typed/union patterns. Avoid a catch-all `_` that silently drops ADT variants.

```scala
def status(m: PaymentMethod): Int = m match       // compiler errors if a case is missing
  case PaymentMethod.CreditCard(_, _, _) => process(...)
  case PaymentMethod.PayPal(email)       => process(email)
  case PaymentMethod.Crypto(w, c)        => process(w, c)

// union types (Scala 3) + type patterns
type JsonValue = String | Int | Boolean | Null
def render(v: JsonValue): String = v match
  case s: String  => s""""$s""""
  case n: Int     => n.toString
  case b: Boolean => b.toString
  case null       => "null"

order match
  case Order(_, _, Customer(_, ContactInfo(email, _, _))) => Some(email)  // nested destructure
```

### F. Immutable collections

Immutable by default; chain transformations; defer with `view`/`LazyList`.

```scala
val emails = users.filter(_.isActive).map(_.email).distinct.sorted
val admins = users.collect { case u if u.isAdmin => u.email }   // filter + map in one pass
val byCustomer: Map[CustomerId, List[Order]] = orders.groupBy(_.customerId)

largeList.view.filter(p).map(f).take(100).toList   // lazy: one traversal, materialize at end
lazy val fibs: LazyList[BigInt] = BigInt(0) #:: BigInt(1) #:: fibs.zip(fibs.tail).map(_ + _)
```

Footgun: `largeList.filter(p).map(f).take(10)` builds full intermediate collections — use `.view` or `.iterator`. Reach for `@tailrec` on recursive accumulators to avoid stack overflow.

### G. Effect systems — Cats Effect & ZIO

Side effects are values. The two mainstream effect systems are **Cats Effect** (`IO`) and **ZIO** (`ZIO[R, E, A]`); pick one per project and stay consistent. Concurrency policy is owned by [`parallelism.md`](guides://parallelism.md).

```scala
// Cats Effect: compose with for-comprehension; manage lifecycles with Resource
import cats.effect.*, cats.syntax.all.*

def register(name: String, email: Email): IO[User] =
  for
    user  <- IO.pure(User(UserId(nextId()), name, email))
    saved <- saveUser(user)
    _     <- sendEmail(email, "Welcome", s"Hi $name")
  yield saved

def transactor: Resource[IO, Transactor[IO]] =          // acquire/release, leak-free
  HikariTransactor.newHikariTransactor[IO](driver, url, user, pass, pool)

(fetchUser(a), fetchOrders(a)).parTupled                // structured concurrency: both or neither
items.parTraverseN(10)(process)                         // bounded parallelism
Ref.of[IO, Int](0)                                      // thread-safe shared state

object Main extends IOApp.Simple:
  def run: IO[Unit] = transactor.use(server).void

// ZIO: typed error channel E and environment R; ZLayer for wiring
import zio.*
def find(id: UserId): ZIO[Database, UserError, User] =
  ZIO.serviceWithZIO[Database](_.get(id)).someOrFail(UserError.NotFound(id))
```

Tagless final (`F[_]: Monad`) abstracts over the concrete effect so services are testable with a pure interpreter (pattern owned by [`designpatterns.md`](guides://designpatterns.md)).

---

## 6. Testing

Test-first policy and coverage are owned by [`tdd.md`](guides://tdd.md). Scala bindings:

```scala
// ScalaTest — expressive matchers, table- and property-driven checks
class UserServiceSpec extends AnyFlatSpec with Matchers with EitherValues:
  "createUser" should "reject invalid email" in:
    service.createUser("Al", "bad").left.value shouldBe a[UserError.InvalidEmail]

// munit + munit-cats-effect — lightweight, returns IO directly
class UserSuite extends CatsEffectSuite:
  test("createUser persists"):
    service.createUser("Al", Email.from("al@x.io").value).map(u => assertEquals(u.name, "Al"))

// ScalaCheck — property-based: generators + algebraic laws
property("json roundtrips") = forAll(genUser): u =>
  JsonDecoder[User].decode(u.toJson) == Right(u)
```

Use tagless-final / in-memory `Ref`-backed implementations of ports instead of mocking frameworks. Cats Effect tests assert on `IO`; munit-cats-effect or ScalaTest's `AsyncIOSpec` run them.

---

## 7. Tooling & Dependencies

`sbt` is the build tool. Security/supply-chain *policy* → [`secure-coding.md`](guides://secure-coding.md); versioning → [`semver.md`](guides://semver.md). Scala binding:

```scala
// build.sbt — fail on warnings, unused, and value discards
ThisBuild / scalaVersion := "3.7.1"
ThisBuild / scalacOptions ++= Seq(
  "-deprecation", "-feature", "-unchecked",
  "-Werror",                 // SCALA-TYP-01: warnings are errors
  "-Wunused:all", "-Wvalue-discard"
)
libraryDependencies ++= Seq(
  "org.typelevel" %% "cats-effect" % "3.5.7",
  "org.scalatest" %% "scalatest"   % "3.2.19" % Test,
  "org.scalameta" %% "munit"       % "1.0.4"  % Test
)
```

```scala
// project/plugins.sbt
addSbtPlugin("ch.epfl.scala"     % "sbt-scalafix"         % "0.13.0")
addSbtPlugin("org.scoverage"     % "sbt-scoverage"        % "2.2.2")
addSbtPlugin("net.vonbuchholtz"  % "sbt-dependency-check" % "5.1.0")
addSbtPlugin("com.timushev.sbt"  % "sbt-updates"          % "0.6.4")
```

```bash
sbt update                  # resolve & pin (SCALA-DEP-01); reject latest.* ranges
sbt dependencyUpdates       # surface newer secure versions
sbt dependencyTree          # inspect transitive deps for audit
sbt dependencyCheck         # SCALA-SEC-01: CVE scan
```

Formatting is `.scalafmt.conf` (pin `version`); linting is scalafix rules (`DisableSyntax` to ban `.get`/`null`/`asInstanceOf`, `OrganizeImports`, `RemoveUnused`).

---

## 8. Quick Reference

```bash
sbt compile                 # build (-Werror)
sbt test                    # test
sbt "scalafixAll --check"   # lint
sbt scalafmtAll             # format
sbt run                     # run
sbt console                 # REPL on the project classpath
sbt doc                     # Scaladoc
```

```scala
opt.map(f).getOrElse(d)         // Option: transform or default
opt.toRight(err)                // Option -> Either
either.flatMap(f).leftMap(g)    // Either: chain, transform error
xs.collect { case p => ... }    // filter + map
xs.traverse(f) / xs.parTraverseN(n)(f)   // effectful / bounded-concurrent
io.handleErrorWith(f); io.timeout(d); (a, b).parTupled   // Cats Effect
given x: T = v  /  def f(using x: T)  /  summon[T]        // type classes
opaque type Id = Long; enum E: case A, B                  // domain types / ADTs
```

---

## 9. Deployment Checklist

Generated from §2 — one box per requirement ID.

- [ ] SCALA-FMT-01 — `scalafmtCheckAll` clean
- [ ] SCALA-LINT-01 — `scalafixAll --check` clean
- [ ] SCALA-TYP-01/02 — compiles under `-Werror`, matches exhaustive
- [ ] SCALA-TST-01/02/03 — tests pass, bugs have regression tests, coverage ≥ gate
- [ ] SCALA-ERR-01 — no `.get`/`.head`/`asInstanceOf`/`null` in prod code
- [ ] SCALA-SEC-01 — `dependencyCheck` 0 high/critical CVEs
- [ ] SCALA-DEP-01 — versions pinned, `sbt update` resolves
- [ ] SCALA-DOC-01 — public APIs documented, `sbt doc` builds
- [ ] SCALA-ARCH-01 — domain layer free of adapter/framework imports
- [ ] Agent ran every §3 command and documented any fixes

---
**End of Scala Guidelines**
