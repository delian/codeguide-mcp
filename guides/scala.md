# Scala Development Guidelines
Mandatory standards for Scala development, following functional programming principles and community best practices. Scala 3, sbt, Metals, ScalaTest, Cats, ZIO.

---

**Agent Profile**: The Scala Expert
**Role**: Senior Scala Developer & Functional Programming Architect
**Objective**: Generate type-safe, functional, and performant Scala code following both OOP and FP best practices.
**Tools**: Scala 3, sbt, Metals, ScalaTest, Cats, ZIO.

---

## 1. Core Philosophies: SCALA-FIRST

**Test-Driven Development (TDD)**: ALWAYS write tests BEFORE implementation (Red-Green-Refactor cycle mandatory).
**Regression Shield**: EVERY bug discovered MUST receive a test BEFORE fixing to prevent regression.

- **S**afe: Leverage the type system for compile-time safety
- **C**omposable: Build from small, reusable components
- **A**lgebraic: Use ADTs and pattern matching
- **L**azy: Prefer lazy evaluation where appropriate
- **A**synchronous: Handle effects properly with IO monads

---

## 2. Type System (MANDATORY)

### A. Type Definitions

```scala
// Algebraic Data Types (ADTs) with enum (Scala 3)
enum PaymentMethod:
  case CreditCard(number: String, expiry: String, cvv: String)
  case PayPal(email: String)
  case BankTransfer(accountNumber: String, routingNumber: String)
  case Crypto(walletAddress: String, currency: CryptoCurrency)

enum CryptoCurrency:
  case Bitcoin, Ethereum, Litecoin

// Sealed traits for Scala 2 style ADTs
sealed trait Result[+E, +A]
case class Success[A](value: A) extends Result[Nothing, A]
case class Failure[E](error: E) extends Result[E, Nothing]

// Opaque types for type safety without runtime overhead
opaque type UserId = Long
object UserId:
  def apply(value: Long): UserId = value
  extension (id: UserId) def value: Long = id

opaque type Email = String
object Email:
  def apply(value: String): Either[String, Email] =
    if value.contains("@") then Right(value)
    else Left("Invalid email format")

  extension (email: Email) def value: String = email

// Type aliases for clarity
type ErrorOr[A] = Either[String, A]
type AsyncResult[A] = Future[Either[Throwable, A]]
```

### B. Case Classes

```scala
// Immutable data with case classes
case class User(
  id: UserId,
  name: String,
  email: Email,
  role: Role = Role.User,
  createdAt: Instant = Instant.now()
):
  def withEmail(newEmail: Email): User = copy(email = newEmail)
  def isAdmin: Boolean = role == Role.Admin

enum Role:
  case Admin, Moderator, User, Guest

// Avoid large case classes - use composition
case class Address(
  street: String,
  city: String,
  country: String,
  postalCode: String
)

case class ContactInfo(
  email: Email,
  phone: Option[String],
  address: Option[Address]
)

case class UserProfile(
  user: User,
  contact: ContactInfo,
  preferences: UserPreferences
)
```

---

## 2A. Test-Driven Development (TDD) Protocol (MANDATORY)

**CRITICAL: Follow the Red-Green-Refactor cycle for ALL new code.**

### TDD Cycle

```
1. RED: Write a failing test first
   ↓
2. GREEN: Write minimal code to make it pass
   ↓
3. REFACTOR: Improve code while keeping tests green
   ↓
   Repeat
```

### Example TDD Workflow for Scala

```scala
// Step 1: RED - Write failing test first
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class EmailValidatorSpec extends AnyFlatSpec with Matchers:

  "EmailValidator.validate" should "return Right for a valid email" in:
    EmailValidator.validate("user@example.com") shouldBe Right("user@example.com")

  it should "return Left for an email without @" in:
    EmailValidator.validate("invalid-email").isLeft shouldBe true

  it should "return Left for an empty string" in:
    EmailValidator.validate("").isLeft shouldBe true

// Run: sbt test
// FAILS - EmailValidator object does not exist

// Step 2: GREEN - Write minimal implementation
object EmailValidator:
  def validate(email: String): Either[String, String] =
    if email.contains("@") then Right(email)
    else Left("Invalid email format")

// Run: sbt test
// PASSES - all tests pass

// Step 3: REFACTOR - Improve with regex validation
object EmailValidator:
  private val emailRegex = """^[^\s@]+@[^\s@]+\.[^\s@]+$""".r

  def validate(email: String): Either[String, String] =
    emailRegex.findFirstIn(email) match
      case Some(_) => Right(email.toLowerCase)
      case None    => Left("Invalid email format")
// Tests still pass
```

---

## 2B. Bug Fix Protocol (MANDATORY)

**CRITICAL: Every bug MUST receive a regression test BEFORE fixing.**

### Bug Fix Workflow

```
1. Bug Reported/Discovered
   ↓
2. Write a test that REPRODUCES the bug (test will FAIL)
   ↓
3. Verify the test fails for the right reason
   ↓
4. Fix the bug (make the test pass)
   ↓
5. Verify the test now PASSES
   ↓
6. Document the bug in test comments (include bug ID)
   ↓
7. Deploy with confidence (regression prevented)
```

### Example Bug Fix

```scala
// Bug Report #1042: EmailValidator accepts emails with spaces like "user @example.com"

// Step 1-2: Write test that reproduces the bug
class EmailValidatorSpec extends AnyFlatSpec with Matchers:

  // Regression test for Bug #1042
  "EmailValidator.validate" should "reject emails containing spaces" in:
    EmailValidator.validate("user @example.com").isLeft shouldBe true
    EmailValidator.validate(" user@example.com").isLeft shouldBe true
    EmailValidator.validate("user@example.com ").isLeft shouldBe true

// Run: sbt test
// FAILS - validate returns Right for emails with spaces

// Step 3: Fix the bug
object EmailValidator:
  private val emailRegex = """^[^\s@]+@[^\s@]+\.[^\s@]+$""".r

  def validate(email: String): Either[String, String] =
    val trimmed = email.trim
    if trimmed != email then Left("Invalid email format: contains whitespace")
    else
      emailRegex.findFirstIn(email) match
        case Some(_) => Right(email.toLowerCase)
        case None    => Left("Invalid email format")

// Run: sbt test
// PASSES - bug fixed, regression prevented
```

### Prohibited Practices for Bug Fixes

**NEVER:**
- Fix a bug without adding a regression test first
- Write implementation before writing tests (violates TDD)
- Skip the Red-Green-Refactor cycle
- Commit code with failing tests
- Remove tests to make code pass
- Use `.get` on `Option`/`Either` or `asInstanceOf` to work around type errors in tests

---

## 3. Functional Patterns (MANDATORY)

### A. Option Handling

```scala
// ✅ CORRECT: Use map, flatMap, getOrElse
def findUser(id: UserId): Option[User] = ???

val userName: String = findUser(userId)
  .map(_.name)
  .getOrElse("Unknown")

val userEmail: Option[String] = for
  user <- findUser(userId)
  email <- user.contactEmail
yield email.value

// ✅ CORRECT: Use fold for transformation
val greeting: String = findUser(userId).fold(
  "Hello, Guest"
)(user => s"Hello, ${user.name}")

// ✅ CORRECT: Use pattern matching when appropriate
findUser(userId) match
  case Some(user) if user.isAdmin => handleAdmin(user)
  case Some(user) => handleRegularUser(user)
  case None => handleGuest()

// ❌ WRONG: Using .get
val user = findUser(userId).get  // Throws on None!

// ❌ WRONG: Manual null checks
if findUser(userId) != null then ..
```

### B. Either for Error Handling

```scala
// Define domain errors
enum UserError:
  case NotFound(id: UserId)
  case InvalidEmail(email: String)
  case AlreadyExists(email: Email)
  case Unauthorized

type UserResult[A] = Either[UserError, A]

// ✅ CORRECT: Chain operations with for-comprehension
def createUser(name: String, emailStr: String): UserResult[User] =
  for
    email <- Email(emailStr).left.map(_ => UserError.InvalidEmail(emailStr))
    _ <- checkEmailNotExists(email)
    user <- saveUser(User(UserId(generateId()), name, email))
  yield user

// ✅ CORRECT: Handle errors at boundaries
createUser("Alice", "alice@example.com") match
  case Right(user) =>
    println(s"Created user: ${user.id}")
  case Left(UserError.InvalidEmail(e)) =>
    println(s"Invalid email: $e")
  case Left(UserError.AlreadyExists(e)) =>
    println(s"Email already registered: ${e.value}")
  case Left(error) =>
    println(s"Error: $error")

// ✅ CORRECT: Transform errors
def processUser(id: UserId): Either[AppError, ProcessedUser] =
  findUser(id)
    .toRight(UserError.NotFound(id))
    .flatMap(validateUser)
    .flatMap(processValidUser)
    .leftMap(userError => AppError.fromUserError(userError))
```

### C. Collections

```scala
// ✅ CORRECT: Use immutable collections by default
val numbers: List[Int] = List(1, 2, 3, 4, 5)
val names: Vector[String] = Vector("Alice", "Bob", "Charlie")
val ages: Map[String, Int] = Map("Alice" -> 30, "Bob" -> 25)

// ✅ CORRECT: Chain transformations
val result = users
  .filter(_.isActive)
  .map(_.email)
  .distinct
  .sorted

// ✅ CORRECT: Use collect for filter + map
val adminEmails: List[Email] = users.collect:
  case user if user.isAdmin => user.email

// ✅ CORRECT: Use foldLeft for complex aggregations
val totalByCategory: Map[Category, BigDecimal] =
  orders.foldLeft(Map.empty[Category, BigDecimal]): (acc, order) =>
    val current = acc.getOrElse(order.category, BigDecimal(0))
    acc.updated(order.category, current + order.amount)

// ✅ CORRECT: Use groupBy and mapValues
val ordersByCustomer: Map[CustomerId, List[Order]] =
  orders.groupBy(_.customerId)

val orderCountByCustomer: Map[CustomerId, Int] =
  orders.groupBy(_.customerId).view.mapValues(_.size).toMap

// ✅ CORRECT: Parallel collections for CPU-intensive work
import scala.collection.parallel.CollectionConverters.*
val processed = largeList.par.map(expensiveComputation).toList
```

---

## 4. Effects and IO (MANDATORY)

### A. Cats Effect

```scala
import cats.effect.*
import cats.syntax.all.*

// Define effectful operations
def fetchUser(id: UserId): IO[Option[User]] = ???
def saveUser(user: User): IO[User] = ???
def sendEmail(to: Email, subject: String, body: String): IO[Unit] = ???

// ✅ CORRECT: Compose effects with for-comprehension
def registerUser(name: String, email: Email): IO[User] =
  for
    _ <- IO.println(s"Registering user: $name")
    user = User(UserId(generateId()), name, email)
    saved <- saveUser(user)
    _ <- sendEmail(email, "Welcome!", s"Hello $name, welcome!")
    _ <- IO.println(s"User registered: ${saved.id}")
  yield saved

// ✅ CORRECT: Error handling with IO
def processUserSafe(id: UserId): IO[Either[UserError, ProcessedUser]] =
  fetchUser(id)
    .flatMap:
      case Some(user) => processUser(user).map(Right(_))
      case None => IO.pure(Left(UserError.NotFound(id)))
    .handleErrorWith: error =>
      IO.println(s"Error processing user: $error") *>
      IO.pure(Left(UserError.Internal(error.getMessage)))

// ✅ CORRECT: Resource management
def withDatabaseConnection[A](f: Connection => IO[A]): IO[A] =
  Resource
    .make(IO(createConnection()))(conn => IO(conn.close()))
    .use(f)

// ✅ CORRECT: Concurrent operations
def fetchAllData(userId: UserId): IO[(User, List[Order], List[Notification])] =
  (
    fetchUser(userId).flatMap(_.liftTo[IO](UserError.NotFound(userId))),
    fetchOrders(userId),
    fetchNotifications(userId)
  ).parTupled

// ✅ CORRECT: Retry with exponential backoff
import cats.effect.std.Random
import scala.concurrent.duration.*

def retryWithBackoff[A](
  action: IO[A],
  maxRetries: Int = 3,
  initialDelay: FiniteDuration = 1.second
): IO[A] =
  action.handleErrorWith: error =>
    if maxRetries > 0 then
      IO.sleep(initialDelay) *>
        retryWithBackoff(action, maxRetries - 1, initialDelay * 2)
    else
      IO.raiseError(error)
```

### B. ZIO Alternative

```scala
import zio.*

// Define services as traits
trait UserService:
  def findUser(id: UserId): IO[UserError, User]
  def createUser(name: String, email: Email): IO[UserError, User]

// Implement with ZIO
object UserServiceLive:
  val layer: ZLayer[Database, Nothing, UserService] =
    ZLayer:
      for
        db <- ZIO.service[Database]
      yield new UserService:
        def findUser(id: UserId): IO[UserError, User] =
          db.query(s"SELECT * FROM users WHERE id = ${id.value}")
            .mapError(_ => UserError.NotFound(id))
            .flatMap:
              case Some(row) => ZIO.succeed(rowToUser(row))
              case None => ZIO.fail(UserError.NotFound(id))

        def createUser(name: String, email: Email): IO[UserError, User] =
          for
            _ <- checkEmailNotExists(email)
            user = User(UserId(generateId()), name, email)
            _ <- db.insert("users", user)
          yield user

// Use the service
def program: ZIO[UserService, UserError, Unit] =
  for
    userService <- ZIO.service[UserService]
    user <- userService.createUser("Alice", Email("alice@example.com").toOption.get)
    _ <- Console.printLine(s"Created: ${user.name}")
  yield ()
```

---

## 5. Pattern Matching (MANDATORY)

### A. Exhaustive Matching

```scala
// ✅ CORRECT: Match all cases
def processPayment(method: PaymentMethod): IO[PaymentResult] =
  method match
    case PaymentMethod.CreditCard(number, expiry, cvv) =>
      processCreditCard(number, expiry, cvv)
    case PaymentMethod.PayPal(email) =>
      processPayPal(email)
    case PaymentMethod.BankTransfer(account, routing) =>
      processBankTransfer(account, routing)
    case PaymentMethod.Crypto(wallet, currency) =>
      processCrypto(wallet, currency)

// ✅ CORRECT: Use guards for additional conditions
def greet(user: User): String =
  user match
    case User(_, name, _, Role.Admin, _) =>
      s"Welcome, Administrator $name"
    case User(_, name, _, _, created) if created.isAfter(recentThreshold) =>
      s"Welcome, new user $name!"
    case User(_, name, _, _, _) =>
      s"Hello, $name"

// ✅ CORRECT: Destructure nested structures
case class Order(id: OrderId, items: List[OrderItem], customer: Customer)
case class Customer(name: String, contact: ContactInfo)

def getCustomerEmail(order: Order): Option[Email] =
  order match
    case Order(_, _, Customer(_, ContactInfo(email, _, _))) =>
      Some(email)
```

### B. Extractors

```scala
// Custom extractors
object Even:
  def unapply(n: Int): Boolean = n % 2 == 0

object Positive:
  def unapply(n: Int): Option[Int] =
    if n > 0 then Some(n) else None

// Use in pattern matching
def classify(n: Int): String = n match
  case 0 => "zero"
  case Even() & Positive(p) => s"positive even: $p"
  case Even() => "negative even"
  case Positive(p) => s"positive odd: $p"
  case _ => "negative odd"

// Email extractor
object ValidEmail:
  private val emailRegex = """^[^@]+@[^@]+\.[^@]+$""".r

  def unapply(s: String): Option[Email] =
    emailRegex.findFirstIn(s).flatMap(Email(_).toOption)

def processInput(input: String): Unit = input match
  case ValidEmail(email) => sendWelcome(email)
  case _ => println("Invalid email")
```

---

## 6. Implicits and Type Classes (MANDATORY)

### A. Type Classes

```scala
// Define type class
trait JsonEncoder[A]:
  def encode(value: A): Json

  extension (a: A)
    def toJson: Json = encode(a)

object JsonEncoder:
  // Summoner
  def apply[A](using encoder: JsonEncoder[A]): JsonEncoder[A] = encoder

  // Instances
  given JsonEncoder[String] with
    def encode(value: String): Json = Json.Str(value)

  given JsonEncoder[Int] with
    def encode(value: Int): Json = Json.Num(value)

  given JsonEncoder[Boolean] with
    def encode(value: Boolean): Json = Json.Bool(value)

  // Derive for Option
  given [A](using encoder: JsonEncoder[A]): JsonEncoder[Option[A]] with
    def encode(value: Option[A]): Json = value match
      case Some(a) => encoder.encode(a)
      case None => Json.Null

  // Derive for List
  given [A](using encoder: JsonEncoder[A]): JsonEncoder[List[A]] with
    def encode(value: List[A]): Json =
      Json.Arr(value.map(encoder.encode))

// Use type class
case class User(name: String, age: Int)

given JsonEncoder[User] with
  def encode(user: User): Json = Json.Obj(
    "name" -> user.name.toJson,
    "age" -> user.age.toJson
  )

// Usage
val json = User("Alice", 30).toJson
```

### B. Context Functions

```scala
// Define context
case class RequestContext(
  userId: UserId,
  traceId: TraceId,
  permissions: Set[Permission]
)

// Context function type
type Contextual[A] = RequestContext ?=> A

// Use context
def getCurrentUserId: Contextual[UserId] =
  summon[RequestContext].userId

def hasPermission(permission: Permission): Contextual[Boolean] =
  summon[RequestContext].permissions.contains(permission)

def authorizedAction[A](permission: Permission)(action: => A): Contextual[Option[A]] =
  if hasPermission(permission) then Some(action)
  else None

// Provide context
given RequestContext = RequestContext(
  UserId(123),
  TraceId("abc"),
  Set(Permission.Read, Permission.Write)
)

val userId = getCurrentUserId  // UserId(123)
val canWrite = hasPermission(Permission.Write)  // true
```

---

## 7. Testing (MANDATORY)

### A. ScalaTest

```scala
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import org.scalatest.EitherValues

class UserServiceSpec extends AnyFlatSpec with Matchers with EitherValues:

  "UserService.createUser" should "create a user with valid input" in:
    val service = new UserService(mockRepo)

    val result = service.createUser("Alice", "alice@example.com")

    result.isRight shouldBe true
    result.value.name shouldBe "Alice"
    result.value.email.value shouldBe "alice@example.com"

  it should "return InvalidEmail error for invalid email" in:
    val service = new UserService(mockRepo)

    val result = service.createUser("Alice", "invalid-email")

    result.isLeft shouldBe true
    result.left.value shouldBe a[UserError.InvalidEmail]

  it should "return AlreadyExists error for duplicate email" in:
    val repo = new MockUserRepository(
      existingEmails = Set("alice@example.com")
    )
    val service = new UserService(repo)

    val result = service.createUser("Alice", "alice@example.com")

    result shouldBe Left(UserError.AlreadyExists(Email("alice@example.com").toOption.get))

class UserValidationSpec extends AnyFlatSpec with Matchers with TableDrivenPropertyChecks:

  val validEmails = Table(
    "email",
    "user@example.com",
    "test.user@domain.co.uk",
    "name+tag@example.org"
  )

  val invalidEmails = Table(
    "email",
    "invalid",
    "@nodomain.com",
    "noat.com",
    ""
  )

  "Email validation" should "accept valid emails" in:
    forAll(validEmails): email =>
      Email(email).isRight shouldBe true

  it should "reject invalid emails" in:
    forAll(invalidEmails): email =>
      Email(email).isLeft shouldBe true
```

### B. Cats Effect Testing

```scala
import cats.effect.testing.scalatest.AsyncIOSpec
import org.scalatest.freespec.AsyncFreeSpec
import org.scalatest.matchers.should.Matchers

class UserServiceIOSpec extends AsyncFreeSpec with AsyncIOSpec with Matchers:

  "UserService" - {
    "fetchUser" - {
      "should return user when exists" in:
        val service = UserService.make(testRepo)

        service.fetchUser(UserId(1)).asserting: result =>
          result shouldBe defined
          result.get.name shouldBe "Test User"

      "should return None when not exists" in:
        val service = UserService.make(emptyRepo)

        service.fetchUser(UserId(999)).asserting: result =>
          result shouldBe None
    }

    "createUser" - {
      "should save and return new user" in:
        val service = UserService.make(testRepo)

        val io = for
          user <- service.createUser("New User", Email("new@example.com").toOption.get)
          fetched <- service.fetchUser(user.id)
        yield (user, fetched)

        io.asserting: (created, fetched) =>
          fetched shouldBe defined
          fetched.get.id shouldBe created.id
    }
  }
```

---

## 8. Concurrency (MANDATORY)

### A. Fibers and Parallelism

```scala
import cats.effect.*
import cats.syntax.all.*

// ✅ CORRECT: Parallel independent operations
def fetchDashboardData(userId: UserId): IO[Dashboard] =
  (
    fetchUserProfile(userId),
    fetchRecentOrders(userId),
    fetchNotifications(userId),
    fetchRecommendations(userId)
  ).parMapN(Dashboard.apply)

// ✅ CORRECT: Race operations
def fetchWithFallback[A](primary: IO[A], fallback: IO[A]): IO[A] =
  primary.timeoutTo(5.seconds, fallback)

// ✅ CORRECT: Concurrent processing with bounded parallelism
def processItems(items: List[Item]): IO[List[Result]] =
  items.parTraverseN(10)(processItem)  // Max 10 concurrent

// ✅ CORRECT: Fiber cancellation
def searchWithTimeout(query: String): IO[Option[SearchResult]] =
  search(query)
    .timeout(10.seconds)
    .handleError(_ => None)

// ✅ CORRECT: Resource-safe concurrent operations
def processWithResources(items: List[Item]): IO[Unit] =
  Stream
    .emits(items)
    .covary[IO]
    .parEvalMapUnordered(maxConcurrent = 5): item =>
      Resource
        .make(acquireConnection())(releaseConnection)
        .use(conn => processItem(item, conn))
    .compile
    .drain
```

### B. Refs and Concurrent State

```scala
import cats.effect.{IO, Ref}

// ✅ CORRECT: Thread-safe state with Ref
def createCounter: IO[Counter] =
  Ref.of[IO, Int](0).map: ref =>
    new Counter:
      def increment: IO[Unit] = ref.update(_ + 1)
      def decrement: IO[Unit] = ref.update(_ - 1)
      def get: IO[Int] = ref.get

// ✅ CORRECT: Atomic updates
def rateLimiter(maxRequests: Int, window: FiniteDuration): IO[RateLimiter] =
  for
    requests <- Ref.of[IO, List[Instant]](Nil)
  yield new RateLimiter:
    def tryAcquire: IO[Boolean] =
      requests.modify: timestamps =>
        val now = Instant.now()
        val cutoff = now.minusMillis(window.toMillis)
        val recent = timestamps.filter(_.isAfter(cutoff))
        if recent.size < maxRequests then
          (now :: recent, true)
        else
          (recent, false)
```

---

## 9. Build Configuration (MANDATORY)

### A. build.sbt

```scala
// build.sbt
ThisBuild / scalaVersion := "3.3.1"
ThisBuild / organization := "com.example"
ThisBuild / version := "0.1.0-SNAPSHOT"

// Compiler options
ThisBuild / scalacOptions ++= Seq(
  "-deprecation",
  "-feature",
  "-unchecked",
  "-Xfatal-warnings",
  "-Wunused:all",
  "-Wvalue-discard"
)

lazy val root = (project in file("."))
  .settings(
    name := "my-app",
    libraryDependencies ++= Seq(
      // Core
      "org.typelevel" %% "cats-core" % "2.10.0",
      "org.typelevel" %% "cats-effect" % "3.5.2",

      // HTTP
      "org.http4s" %% "http4s-ember-server" % "0.23.23",
      "org.http4s" %% "http4s-dsl" % "0.23.23",
      "org.http4s" %% "http4s-circe" % "0.23.23",

      // JSON
      "io.circe" %% "circe-core" % "0.14.6",
      "io.circe" %% "circe-generic" % "0.14.6",

      // Database
      "org.tpolecat" %% "doobie-core" % "1.0.0-RC4",
      "org.tpolecat" %% "doobie-hikari" % "1.0.0-RC4",
      "org.tpolecat" %% "doobie-postgres" % "1.0.0-RC4",

      // Testing
      "org.scalatest" %% "scalatest" % "3.2.17" % Test,
      "org.typelevel" %% "cats-effect-testing-scalatest" % "1.5.0" % Test
    )
  )
```

---

## 10. Deployment Checklist

### Code Quality
- [ ] No compiler warnings
- [ ] Scalafmt formatting applied
- [ ] All tests passing
- [ ] No deprecated API usage

### Type Safety
- [ ] Exhaustive pattern matching
- [ ] Proper error types defined
- [ ] No unsafe operations (.get, asInstanceOf)

### Performance
- [ ] Proper use of lazy vals
- [ ] Tail-recursive functions where needed
- [ ] Efficient collection operations

### Effects
- [ ] Resource cleanup handled
- [ ] Cancellation supported
- [ ] Error handling comprehensive

---

## 11. Quick Reference

```scala
// Option
option.map(f)
option.flatMap(f)
option.getOrElse(default)
option.fold(default)(f)
option.toRight(error)  // Either

// Either
either.map(f)
either.flatMap(f)
either.leftMap(f)
either.fold(onLeft, onRight)
either.toOption

// List
list.map(f)
list.flatMap(f)
list.filter(p)
list.collect { case ... => }
list.foldLeft(z)(f)
list.groupBy(f)

// IO (Cats Effect)
IO.pure(value)
IO.delay(computation)
IO.raiseError(error)
io.flatMap(f)
io.handleErrorWith(f)
io.attempt  // IO[Either[Throwable, A]]
(io1, io2).parTupled

// Pattern matching
x match
  case Pattern(a, b) => ..
  case _ if guard => ..
  case _ => ..
```

---

**Last Updated:** 2026-01-31
**Version:** 1.0
**Maintainer:** Scala Team


**End of Scala Development Guidelines**
