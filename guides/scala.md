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

// Opaque types with bounds for subtyping relationships
opaque type NonEmptyString <: String = String
object NonEmptyString:
  def apply(value: String): Either[String, NonEmptyString] =
    if value.nonEmpty then Right(value)
    else Left("String must not be empty")

  def unsafeFrom(value: String): NonEmptyString =
    require(value.nonEmpty, "String must not be empty")
    value

  extension (s: NonEmptyString) def value: String = s

// Opaque types with numeric precision
opaque type Percentage = Double
object Percentage:
  def apply(value: Double): Either[String, Percentage] =
    if value >= 0.0 && value <= 100.0 then Right(value)
    else Left(s"Percentage must be between 0 and 100, got $value")

  val Zero: Percentage = 0.0
  val Full: Percentage = 100.0

  extension (p: Percentage)
    def value: Double = p
    def complement: Percentage = 100.0 - p
    def of(total: BigDecimal): BigDecimal = total * p / 100.0

// Type aliases for clarity
type ErrorOr[A] = Either[String, A]
type AsyncResult[A] = Future[Either[Throwable, A]]
```

### C. Extension Methods

```scala
// ✅ CORRECT: Group related extension methods
extension (s: String)
  def toSlug: String =
    s.toLowerCase.replaceAll("[^a-z0-9]+", "-").stripPrefix("-").stripSuffix("-")

  def truncate(maxLength: Int): String =
    if s.length <= maxLength then s
    else s.take(maxLength - 3) + "..."

  def toOption: Option[String] =
    Option.when(s.nonEmpty)(s)

// ✅ CORRECT: Extension methods with type parameters and context bounds
extension [A](list: List[A])
  def partitionMap[B, C](f: A => Either[B, C]): (List[B], List[C]) =
    list.foldRight((List.empty[B], List.empty[C])): (a, acc) =>
      f(a) match
        case Left(b)  => (b :: acc._1, acc._2)
        case Right(c) => (acc._1, c :: acc._2)

  def groupByNel[K](f: A => K): Map[K, List[A]] =
    list.groupBy(f).view.mapValues(_.toList).toMap

// ✅ CORRECT: Extension methods with given constraints
extension [A](option: Option[A])
  def toEither[E](error: => E): Either[E, A] =
    option match
      case Some(a) => Right(a)
      case None    => Left(error)

  def orRaise[F[_]](error: => Throwable)(using ae: cats.ApplicativeError[F, Throwable]): F[A] =
    option match
      case Some(a) => ae.pure(a)
      case None    => ae.raiseError(error)

// ✅ CORRECT: Extension methods on domain types
extension (user: User)
  def fullDisplayName: String =
    s"${user.name} (${user.role})"

  def hasPermission(permission: Permission): Boolean =
    user.role match
      case Role.Admin     => true
      case Role.Moderator => permission != Permission.ManageUsers
      case Role.User      => permission == Permission.Read || permission == Permission.Write
      case Role.Guest     => permission == Permission.Read

// ❌ WRONG: Extension methods that should be regular methods on the type
// Don't add core business logic as extensions - put it in the type itself
```

### D. Given/Using Clauses

```scala
// ✅ CORRECT: Define given instances for type classes
trait Ordering[A]:
  def compare(x: A, y: A): Int

given Ordering[Int] with
  def compare(x: Int, y: Int): Int = x - y

given Ordering[String] with
  def compare(x: String, y: String): Int = x.compareTo(y)

// ✅ CORRECT: Derived given instances
given [A](using ord: Ordering[A]): Ordering[List[A]] with
  def compare(xs: List[A], ys: List[A]): Int =
    (xs, ys) match
      case (Nil, Nil) => 0
      case (Nil, _)   => -1
      case (_, Nil)   => 1
      case (x :: xtail, y :: ytail) =>
        val c = ord.compare(x, y)
        if c != 0 then c else compare(xtail, ytail)

// ✅ CORRECT: Using clauses for dependency injection
def sortItems[A](items: List[A])(using ord: Ordering[A]): List[A] =
  items.sortWith((a, b) => ord.compare(a, b) < 0)

// ✅ CORRECT: Named given with alias
given userOrdering: Ordering[User] with
  def compare(x: User, y: User): Int =
    x.name.compareTo(y.name)

// ✅ CORRECT: Given imports - be explicit about what you import
import scala.math.Ordering.given           // import all given Ordering instances
import MyCodecs.given                       // import all given from MyCodecs
import MyCodecs.{given JsonEncoder[User]}   // import specific given

// ✅ CORRECT: Context bounds (shorthand for using)
def max[A: Ordering](x: A, y: A): A =
  val ord = summon[Ordering[A]]
  if ord.compare(x, y) >= 0 then x else y

// ❌ WRONG: Overusing implicits for non-type-class patterns
// Don't make random values implicit just for convenience
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

// ✅ CORRECT: Use LazyList for potentially infinite or expensive sequences
val fibonacci: LazyList[BigInt] =
  BigInt(0) #:: BigInt(1) #:: fibonacci.zip(fibonacci.tail).map(_ + _)

val first20Fibs = fibonacci.take(20).toList

// ✅ CORRECT: Use view for lazy intermediate transformations on large collections
val result = largeList.view
  .filter(_.isActive)
  .map(_.transform)
  .take(100)
  .toList  // Only materializes at the end

// ❌ WRONG: Creating intermediate collections unnecessarily
// val result = largeList.filter(p).map(f).take(10) // 3 full traversals
```

### D. Error Handling Patterns

```scala
// ✅ CORRECT: Use Try for wrapping Java APIs that throw exceptions
import scala.util.{Try, Success, Failure}

def parseConfig(raw: String): Either[ConfigError, Config] =
  Try(ConfigFactory.parseString(raw)) match
    case Success(parsed) => Right(Config.fromParsed(parsed))
    case Failure(ex)     => Left(ConfigError.ParseFailed(ex.getMessage))

// ✅ CORRECT: Convert between error types
def toAppError(userError: UserError): AppError = userError match
  case UserError.NotFound(id)     => AppError.ResourceNotFound("user", id.value.toString)
  case UserError.InvalidEmail(e)  => AppError.ValidationFailed(s"Invalid email: $e")
  case UserError.AlreadyExists(e) => AppError.Conflict(s"Email already registered: ${e.value}")
  case UserError.Unauthorized     => AppError.Forbidden("Insufficient permissions")

// ✅ CORRECT: Enum-based error hierarchies with Scala 3
enum AppError:
  case ValidationFailed(message: String, field: Option[String] = None)
  case ResourceNotFound(resourceType: String, id: String)
  case Conflict(message: String)
  case Forbidden(reason: String)
  case Internal(message: String, cause: Option[Throwable] = None)
  case RateLimited(retryAfter: scala.concurrent.duration.FiniteDuration)

  def toHttpStatus: Int = this match
    case _: ValidationFailed => 400
    case _: ResourceNotFound => 404
    case _: Conflict         => 409
    case _: Forbidden        => 403
    case _: Internal         => 500
    case _: RateLimited      => 429

  def toMessage: String = this match
    case ValidationFailed(msg, Some(field)) => s"Validation failed for $field: $msg"
    case ValidationFailed(msg, None)        => s"Validation failed: $msg"
    case ResourceNotFound(tpe, id)          => s"$tpe not found: $id"
    case Conflict(msg)                      => msg
    case Forbidden(reason)                  => s"Forbidden: $reason"
    case Internal(msg, _)                   => s"Internal error: $msg"
    case RateLimited(d)                     => s"Rate limited. Retry after ${d.toSeconds}s"

// ✅ CORRECT: Validated for accumulating errors (Cats)
import cats.data.ValidatedNel
import cats.syntax.all.*

case class RegistrationForm(name: String, email: String, age: Int)

def validateName(name: String): ValidatedNel[String, String] =
  if name.nonEmpty then name.validNel
  else "Name must not be empty".invalidNel

def validateEmail(email: String): ValidatedNel[String, Email] =
  Email(email).leftMap(_ => "Invalid email format").toValidatedNel

def validateAge(age: Int): ValidatedNel[String, Int] =
  if age >= 18 && age <= 150 then age.validNel
  else s"Age must be between 18 and 150, got $age".invalidNel

def validateRegistration(form: RegistrationForm): ValidatedNel[String, ValidatedUser] =
  (
    validateName(form.name),
    validateEmail(form.email),
    validateAge(form.age)
  ).mapN(ValidatedUser.apply)

// Usage: accumulates ALL errors instead of short-circuiting on first
val result = validateRegistration(RegistrationForm("", "bad", 10))
// Invalid(NonEmptyList("Name must not be empty", "Invalid email format", "Age must be between 18 and 150, got 10"))
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

// ✅ CORRECT: Resource composition
def appResources: Resource[IO, AppResources] =
  for
    config  <- Resource.eval(loadConfig)
    pool    <- ExecutionContexts.fixedThreadPool[IO](config.dbPoolSize)
    xa      <- HikariTransactor.newHikariTransactor[IO](
                 config.dbDriver, config.dbUrl, config.dbUser, config.dbPass, pool
               )
    client  <- EmberClientBuilder.default[IO].build
    cache   <- Resource.eval(Ref.of[IO, Map[String, CacheEntry]](Map.empty))
  yield AppResources(xa, client, cache)

// ✅ CORRECT: Bracket pattern for cleanup
def withTempFile[A](prefix: String)(use: Path => IO[A]): IO[A] =
  IO.blocking(Files.createTempFile(prefix, ".tmp"))
    .bracket(use)(path => IO.blocking(Files.deleteIfExists(path)).void)

// ✅ CORRECT: IO.both for structured concurrency (neither leaks on failure)
def fetchPair(id1: UserId, id2: UserId): IO[(User, User)] =
  (fetchUser(id1), fetchUser(id2)).parTupled
  // If either fails, the other is cancelled automatically

// ✅ CORRECT: Supervisor for background tasks
import cats.effect.std.Supervisor

def withBackgroundTasks: Resource[IO, Unit] =
  for
    supervisor <- Supervisor[IO]
    _          <- Resource.eval(supervisor.supervise(periodicCleanup))
    _          <- Resource.eval(supervisor.supervise(metricsReporter))
  yield ()

// ✅ CORRECT: IOApp as application entry point
object Main extends IOApp.Simple:
  val run: IO[Unit] =
    appResources.use: resources =>
      for
        _      <- IO.println("Starting application...")
        server <- buildServer(resources)
        _      <- server.useForever
      yield ()
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

// ✅ CORRECT: ZIO error handling with typed errors
def processOrder(orderId: OrderId): ZIO[OrderService & PaymentService, OrderError, Receipt] =
  for
    orderService   <- ZIO.service[OrderService]
    paymentService <- ZIO.service[PaymentService]
    order          <- orderService.findOrder(orderId)
                        .someOrFail(OrderError.NotFound(orderId))
    _              <- ZIO.when(order.isPaid)(ZIO.fail(OrderError.AlreadyPaid(orderId)))
    receipt        <- paymentService.charge(order)
                        .mapError(e => OrderError.PaymentFailed(orderId, e.getMessage))
    _              <- orderService.markPaid(orderId, receipt.id)
  yield receipt

// ✅ CORRECT: ZIO layers composition
object AppLayer:
  val live: ZLayer[Any, Config.Error, UserService & OrderService & PaymentService] =
    ZLayer.make[UserService & OrderService & PaymentService](
      ConfigLive.layer,
      DatabaseLive.layer,
      UserServiceLive.layer,
      OrderServiceLive.layer,
      PaymentServiceLive.layer,
      HttpClientLive.layer
    )

// ✅ CORRECT: ZIO resource management with Scope
def managedConnection: ZIO[Scope, Throwable, Connection] =
  ZIO.acquireRelease(
    ZIO.attempt(dataSource.getConnection)
  )(conn => ZIO.succeed(conn.close()))

// ✅ CORRECT: ZIO retry scheduling
import zio.Schedule

val retryPolicy = Schedule.exponential(1.second) &&
  Schedule.recurs(5) &&
  Schedule.recurWhile[Throwable] {
    case _: java.net.ConnectException => true
    case _: java.util.concurrent.TimeoutException => true
    case _ => false
  }

def callExternalApi(request: Request): ZIO[HttpClient, ApiError, Response] =
  httpClient.send(request)
    .retry(retryPolicy)
    .mapError(ApiError.ConnectionFailed(_))
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

### C. Pattern Matching Best Practices

```scala
// ✅ CORRECT: Use @unchecked only when you can prove exhaustiveness
// The compiler cannot always verify exhaustiveness for complex patterns
(list: @unchecked) match
  case head :: tail => processNonEmpty(head, tail)
  // We know the list is non-empty because we checked earlier

// ✅ CORRECT: Type-safe pattern matching with union types (Scala 3)
type JsonValue = String | Int | Double | Boolean | Null
def stringify(value: JsonValue): String = value match
  case s: String  => s""""$s""""
  case n: Int     => n.toString
  case d: Double  => d.toString
  case b: Boolean => b.toString
  case null       => "null"

// ✅ CORRECT: Pattern matching on tuples
def handleCoordinates(point: (Double, Double)): String = point match
  case (0.0, 0.0)           => "origin"
  case (x, 0.0)             => s"on x-axis at $x"
  case (0.0, y)             => s"on y-axis at $y"
  case (x, y) if x == y     => s"on diagonal at $x"
  case (x, y)               => s"point ($x, $y)"

// ✅ CORRECT: Bind matched values with @ for further use
def processTree[A](tree: Tree[A]): String = tree match
  case leaf @ Leaf(value)                => s"Leaf: $value"
  case node @ Node(left, right) if node.depth > 10 => "Deep node"
  case Node(Leaf(l), Leaf(r))            => s"Shallow node: $l, $r"
  case Node(left, right)                 => s"Node with depth ${left.depth + right.depth}"

// ✅ CORRECT: Pattern matching in val declarations and for-comprehensions
val (first, rest) = list.splitAt(1)

val pairs = for
  case (key, value: String) <- config.entries  // Irrefutable patterns with case
yield (key, value.trim)

// ✅ CORRECT: Match types (Scala 3 advanced feature)
type Elem[X] = X match
  case String      => Char
  case Array[t]    => t
  case Iterable[t] => t

def firstElem[X](x: X): Elem[X] = x match
  case s: String      => s.charAt(0)
  case a: Array[_]    => a(0)
  case i: Iterable[_] => i.head

// ❌ WRONG: Catch-all that silently drops cases
def process(cmd: Command): Unit = cmd match
  case Command.Start => start()
  case _ => ()  // Silently ignores Stop, Restart, etc.

// ✅ CORRECT: Handle all cases explicitly
def process(cmd: Command): Unit = cmd match
  case Command.Start   => start()
  case Command.Stop    => stop()
  case Command.Restart => restart()
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

### B. Type Class Derivation and Advanced Patterns

```scala
// ✅ CORRECT: Type class with contravariant functor
trait JsonDecoder[A]:
  def decode(json: Json): Either[DecodeError, A]

object JsonDecoder:
  def apply[A](using decoder: JsonDecoder[A]): JsonDecoder[A] = decoder

  // Create instances from functions
  def instance[A](f: Json => Either[DecodeError, A]): JsonDecoder[A] =
    new JsonDecoder[A]:
      def decode(json: Json): Either[DecodeError, A] = f(json)

  given JsonDecoder[String] = instance:
    case Json.Str(s) => Right(s)
    case other        => Left(DecodeError.TypeMismatch("String", other))

  given JsonDecoder[Int] = instance:
    case Json.Num(n) => Right(n.toInt)
    case other        => Left(DecodeError.TypeMismatch("Int", other))

  given [A](using dec: JsonDecoder[A]): JsonDecoder[List[A]] = instance:
    case Json.Arr(items) =>
      items.traverse(dec.decode)
    case other =>
      Left(DecodeError.TypeMismatch("Array", other))

// ✅ CORRECT: Combine encoder and decoder into a Codec
trait JsonCodec[A] extends JsonEncoder[A] with JsonDecoder[A]

object JsonCodec:
  def from[A](enc: JsonEncoder[A], dec: JsonDecoder[A]): JsonCodec[A] =
    new JsonCodec[A]:
      def encode(value: A): Json = enc.encode(value)
      def decode(json: Json): Either[DecodeError, A] = dec.decode(json)

// ✅ CORRECT: Show type class for display
trait Show[A]:
  def show(a: A): String

  extension (a: A) def display: String = show(a)

object Show:
  def apply[A](using s: Show[A]): Show[A] = s

  def from[A](f: A => String): Show[A] =
    new Show[A]:
      def show(a: A): String = f(a)

  given Show[String] = from(identity)
  given Show[Int] = from(_.toString)
  given Show[UserId] = from(id => s"UserId(${id.value})")
  given Show[Email] = from(e => e.value)

  given [A](using s: Show[A]): Show[List[A]] = from: list =>
    list.map(s.show).mkString("[", ", ", "]")

  given Show[User] = from: u =>
    s"User(${u.name}, ${u.email.display}, ${u.role})"

// ✅ CORRECT: Eq type class for type-safe equality
trait Eq[A]:
  def eqv(x: A, y: A): Boolean

  extension (x: A)
    def ===(y: A): Boolean = eqv(x, y)
    def =!=(y: A): Boolean = !eqv(x, y)

object Eq:
  def from[A](f: (A, A) => Boolean): Eq[A] =
    new Eq[A]:
      def eqv(x: A, y: A): Boolean = f(x, y)

  given Eq[String] = from(_ == _)
  given Eq[Int] = from(_ == _)
  given Eq[UserId] = from((a, b) => a.value == b.value)
```

### C. Context Functions

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

### C. MUnit Testing

```scala
import munit.CatsEffectSuite

class UserServiceMunitSpec extends CatsEffectSuite:

  val testUser = User(UserId(1), "Test", Email.unsafeFrom("test@example.com"))

  test("createUser returns created user"):
    val service = UserService.make(InMemoryRepo())
    for
      user <- service.createUser("Alice", Email.unsafeFrom("alice@example.com"))
    yield
      assertEquals(user.name, "Alice")
      assertEquals(user.email.value, "alice@example.com")

  test("createUser fails for duplicate email"):
    val repo = InMemoryRepo(existing = List(testUser))
    val service = UserService.make(repo)
    interceptMessageIO[UserAlreadyExistsException]("User already exists"):
      service.createUser("Bob", testUser.email)

  // Fixtures for resource management in tests
  val databaseFixture = ResourceSuiteLocalFixture(
    "database",
    Resource.make(IO(TestDatabase.create()))(db => IO(db.close()))
  )

  override def munitFixtures = List(databaseFixture)

  test("query returns results from database"):
    val db = databaseFixture()
    for
      _      <- IO(db.insert(testUser))
      result <- IO(db.findById(testUser.id))
    yield assertEquals(result, Some(testUser))
```

### D. ScalaCheck Property-Based Testing

```scala
import org.scalacheck.{Gen, Prop, Properties}
import org.scalacheck.Prop.forAll
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import org.scalatestplus.scalacheck.ScalaCheckPropertyChecks

// ✅ CORRECT: Define generators for domain types
object Generators:
  val genNonEmptyString: Gen[String] =
    Gen.nonEmptyListOf(Gen.alphaNumChar).map(_.mkString)

  val genEmail: Gen[Email] =
    for
      local  <- genNonEmptyString
      domain <- genNonEmptyString
      tld    <- Gen.oneOf("com", "org", "net", "io")
    yield Email.unsafeFrom(s"$local@$domain.$tld")

  val genRole: Gen[Role] =
    Gen.oneOf(Role.Admin, Role.Moderator, Role.User, Role.Guest)

  val genUserId: Gen[UserId] =
    Gen.posNum[Long].map(UserId(_))

  val genUser: Gen[User] =
    for
      id    <- genUserId
      name  <- genNonEmptyString
      email <- genEmail
      role  <- genRole
    yield User(id, name, email, role)

  val genPercentage: Gen[Double] =
    Gen.choose(0.0, 100.0)

  val genMoney: Gen[BigDecimal] =
    Gen.choose(0L, 1000000L).map(cents => BigDecimal(cents) / 100)

// ✅ CORRECT: Property-based tests with ScalaTest integration
class MoneySpec extends AnyFlatSpec with Matchers with ScalaCheckPropertyChecks:
  import Generators.*

  "Money addition" should "be commutative" in:
    forAll(genMoney, genMoney): (a, b) =>
      (a + b) shouldBe (b + a)

  it should "be associative" in:
    forAll(genMoney, genMoney, genMoney): (a, b, c) =>
      ((a + b) + c) shouldBe (a + (b + c))

  it should "have zero as identity" in:
    forAll(genMoney): a =>
      (a + BigDecimal(0)) shouldBe a

// ✅ CORRECT: Property-based testing with ScalaCheck Properties
object UserProperties extends Properties("User"):
  import Generators.*

  property("serialization roundtrip") = forAll(genUser): user =>
    val json = user.toJson
    val decoded = JsonDecoder[User].decode(json)
    decoded == Right(user)

  property("role permissions are monotonic") = forAll(genUser): user =>
    user.role match
      case Role.Admin => user.hasPermission(Permission.Read) &&
                         user.hasPermission(Permission.Write) &&
                         user.hasPermission(Permission.ManageUsers)
      case Role.Moderator => user.hasPermission(Permission.Read) &&
                             user.hasPermission(Permission.Write)
      case Role.User  => user.hasPermission(Permission.Read)
      case Role.Guest => user.hasPermission(Permission.Read)

// ✅ CORRECT: Shrinking for better error messages
import org.scalacheck.Shrink

given Shrink[User] = Shrink: user =>
  Shrink.shrink(user.name).map(n => user.copy(name = n))
```

### E. Table-Driven Tests

```scala
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import org.scalatest.prop.TableDrivenPropertyChecks

class DiscountCalculatorSpec extends AnyFlatSpec with Matchers with TableDrivenPropertyChecks:

  val discountScenarios = Table(
    ("description",       "orderTotal", "customerType", "expectedDiscount"),
    ("no discount",       100.0,        "regular",      0.0),
    ("bulk discount",     500.0,        "regular",      25.0),
    ("VIP discount",      100.0,        "vip",          10.0),
    ("VIP bulk discount", 500.0,        "vip",          75.0),
    ("employee discount", 200.0,        "employee",     40.0),
  )

  "DiscountCalculator" should "calculate correct discounts" in:
    forAll(discountScenarios): (desc, total, customerType, expected) =>
      withClue(s"Scenario: $desc"):
        DiscountCalculator.calculate(total, customerType) shouldBe expected

  val httpStatusMappings = Table(
    ("error",                                   "expectedStatus"),
    (AppError.ValidationFailed("bad input"),     400),
    (AppError.ResourceNotFound("user", "123"),   404),
    (AppError.Conflict("duplicate"),             409),
    (AppError.Forbidden("nope"),                 403),
    (AppError.Internal("boom"),                  500),
  )

  "AppError.toHttpStatus" should "map errors to correct HTTP status codes" in:
    forAll(httpStatusMappings): (error, expectedStatus) =>
      error.toHttpStatus shouldBe expectedStatus
```

### F. Async Testing Patterns

```scala
import cats.effect.IO
import cats.effect.testing.scalatest.AsyncIOSpec
import org.scalatest.freespec.AsyncFreeSpec
import org.scalatest.matchers.should.Matchers
import scala.concurrent.duration.*

class TimeoutServiceSpec extends AsyncFreeSpec with AsyncIOSpec with Matchers:

  "TimeoutService" - {
    "should complete within timeout" in:
      val service = TimeoutService.make(timeout = 5.seconds)
      service.process("fast-request")
        .asserting: result =>
          result.isSuccess shouldBe true

    "should fail when operation exceeds timeout" in:
      val service = TimeoutService.make(timeout = 100.millis)
      service.process("slow-request")
        .assertNoException  // Expect graceful timeout handling, not exception

    "should handle concurrent requests" in:
      val service = TimeoutService.make(timeout = 5.seconds)
      val requests = List.fill(100)(service.process("concurrent-req"))
      requests.parSequence
        .asserting: results =>
          results.size shouldBe 100
          results.forall(_.isSuccess) shouldBe true

    "should cancel in-flight request on fiber cancellation" in:
      val service = TimeoutService.make(timeout = 5.seconds)
      for
        ref    <- cats.effect.Ref.of[IO, Boolean](false)
        fiber  <- service.processWithCallback("req", () => ref.set(true)).start
        _      <- IO.sleep(50.millis)
        _      <- fiber.cancel
        wasSet <- ref.get
      yield assert(!wasSet, "Callback should not have fired after cancellation")
  }

// Test helpers
trait TestSupport:
  def withTestService[A](f: UserService[IO] => IO[A]): IO[A] =
    for
      ref <- cats.effect.Ref.of[IO, Map[UserId, User]](Map.empty)
      service = InMemoryUserService(ref)
      result <- f(service)
    yield result

  class InMemoryUserService(store: cats.effect.Ref[IO, Map[UserId, User]]) extends UserService[IO]:
    def findUser(id: UserId): IO[Option[User]] =
      store.get.map(_.get(id))

    def saveUser(user: User): IO[User] =
      store.update(_ + (user.id -> user)).as(user)

    def deleteUser(id: UserId): IO[Boolean] =
      store.modify: m =>
        if m.contains(id) then (m - id, true)
        else (m, false)
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
ThisBuild / scalaVersion := "3.8.0"
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

### B. Multi-Module sbt Project

```scala
// build.sbt - Multi-module project structure
ThisBuild / scalaVersion := "3.8.0"
ThisBuild / organization := "com.example"
ThisBuild / version := "0.1.0-SNAPSHOT"

ThisBuild / scalacOptions ++= Seq(
  "-deprecation",
  "-feature",
  "-unchecked",
  "-Xfatal-warnings",
  "-Wunused:all",
  "-Wvalue-discard"
)

// Shared dependency versions
lazy val versions = new {
  val catsEffect = "3.5.2"
  val http4s     = "0.23.23"
  val circe      = "0.14.6"
  val doobie     = "1.0.0-RC4"
  val pureconfig = "0.17.4"
  val log4cats   = "2.6.0"
  val scalatest  = "3.2.17"
  val munit      = "1.0.0"
}

lazy val root = (project in file("."))
  .aggregate(core, domain, infrastructure, api, app)
  .settings(
    name := "my-app",
    publish / skip := true
  )

// Domain layer: pure business logic, no framework dependencies
lazy val domain = (project in file("modules/domain"))
  .settings(
    name := "my-app-domain",
    libraryDependencies ++= Seq(
      "org.typelevel" %% "cats-core"  % "2.10.0",
      "org.scalatest" %% "scalatest"  % versions.scalatest % Test,
    )
  )

// Core layer: shared types, configs, and utilities
lazy val core = (project in file("modules/core"))
  .dependsOn(domain)
  .settings(
    name := "my-app-core",
    libraryDependencies ++= Seq(
      "org.typelevel"         %% "cats-effect"    % versions.catsEffect,
      "org.typelevel"         %% "log4cats-slf4j" % versions.log4cats,
      "com.github.pureconfig" %% "pureconfig-core" % versions.pureconfig,
    )
  )

// Infrastructure layer: database, HTTP clients, external integrations
lazy val infrastructure = (project in file("modules/infrastructure"))
  .dependsOn(core)
  .settings(
    name := "my-app-infra",
    libraryDependencies ++= Seq(
      "org.tpolecat" %% "doobie-core"     % versions.doobie,
      "org.tpolecat" %% "doobie-hikari"   % versions.doobie,
      "org.tpolecat" %% "doobie-postgres" % versions.doobie,
      "org.http4s"   %% "http4s-ember-client" % versions.http4s,
      "org.http4s"   %% "http4s-circe"    % versions.http4s,
      "io.circe"     %% "circe-core"      % versions.circe,
      "io.circe"     %% "circe-generic"   % versions.circe,
      "org.scalameta" %% "munit"          % versions.munit % Test,
      "org.typelevel" %% "munit-cats-effect" % "2.0.0" % Test,
    )
  )

// API layer: HTTP routes, request/response codecs
lazy val api = (project in file("modules/api"))
  .dependsOn(core, infrastructure)
  .settings(
    name := "my-app-api",
    libraryDependencies ++= Seq(
      "org.http4s" %% "http4s-ember-server" % versions.http4s,
      "org.http4s" %% "http4s-dsl"          % versions.http4s,
      "org.http4s" %% "http4s-circe"        % versions.http4s,
    )
  )

// App layer: wiring, main entry point
lazy val app = (project in file("modules/app"))
  .dependsOn(api, infrastructure)
  .enablePlugins(JavaAppPackaging, DockerPlugin)
  .settings(
    name := "my-app-runner",
    Docker / packageName := "my-app",
    Docker / version := version.value,
    dockerBaseImage := "eclipse-temurin:21-jre-alpine",
    dockerExposedPorts := Seq(8080),
  )
```

### C. Project Directory Layout

```
my-app/
├── build.sbt
├── project/
│   ├── build.properties          # sbt.version=1.9.7
│   └── plugins.sbt               # sbt plugins
├── modules/
│   ├── domain/
│   │   └── src/
│   │       ├── main/scala/com/example/domain/
│   │       │   ├── models/        # Case classes, ADTs, value objects
│   │       │   │   ├── User.scala
│   │       │   │   ├── Order.scala
│   │       │   │   └── Payment.scala
│   │       │   ├── errors/        # Domain error types
│   │       │   │   └── DomainError.scala
│   │       │   ├── services/      # Domain service traits (ports)
│   │       │   │   ├── UserService.scala
│   │       │   │   └── OrderService.scala
│   │       │   └── validation/    # Business rule validators
│   │       │       └── Validators.scala
│   │       └── test/scala/com/example/domain/
│   │           ├── models/
│   │           └── validation/
│   ├── core/
│   │   └── src/main/scala/com/example/core/
│   │       ├── config/            # Configuration case classes
│   │       │   └── AppConfig.scala
│   │       ├── logging/           # Logging utilities
│   │       │   └── LoggingMiddleware.scala
│   │       └── types/             # Shared opaque types, type aliases
│   │           └── CommonTypes.scala
│   ├── infrastructure/
│   │   └── src/main/scala/com/example/infra/
│   │       ├── persistence/       # Database repositories
│   │       │   ├── UserRepository.scala
│   │       │   └── DoobieUserRepository.scala
│   │       ├── http/              # HTTP client adapters
│   │       │   └── PaymentGatewayClient.scala
│   │       └── messaging/         # Message queue adapters
│   │           └── KafkaProducer.scala
│   ├── api/
│   │   └── src/main/scala/com/example/api/
│   │       ├── routes/            # HTTP route definitions
│   │       │   ├── UserRoutes.scala
│   │       │   └── OrderRoutes.scala
│   │       ├── codecs/            # JSON encoders/decoders
│   │       │   └── JsonCodecs.scala
│   │       └── middleware/        # HTTP middleware
│   │           ├── AuthMiddleware.scala
│   │           └── CorsMiddleware.scala
│   └── app/
│       └── src/main/scala/com/example/
│           └── Main.scala         # Application entry point, wiring
└── docker-compose.yml
```

### D. Configuration with PureConfig and HOCON

```scala
// modules/core/src/main/scala/com/example/core/config/AppConfig.scala

import pureconfig.*
import pureconfig.generic.derivation.default.*
import scala.concurrent.duration.FiniteDuration

// ✅ CORRECT: Typed configuration with PureConfig
case class AppConfig(
  server: ServerConfig,
  database: DatabaseConfig,
  auth: AuthConfig,
  features: FeatureFlags
) derives ConfigReader

case class ServerConfig(
  host: String,
  port: Int,
  idleTimeout: FiniteDuration,
  shutdownTimeout: FiniteDuration
) derives ConfigReader

case class DatabaseConfig(
  driver: String,
  url: String,
  user: String,
  password: String,
  poolSize: Int,
  connectionTimeout: FiniteDuration
) derives ConfigReader

case class AuthConfig(
  jwtSecret: String,
  tokenExpiry: FiniteDuration,
  refreshTokenExpiry: FiniteDuration,
  issuer: String
) derives ConfigReader

case class FeatureFlags(
  enableNewCheckout: Boolean,
  enableBetaFeatures: Boolean,
  maxUploadSizeMb: Int
) derives ConfigReader

object AppConfig:
  def load: Either[pureconfig.error.ConfigReaderFailures, AppConfig] =
    ConfigSource.default.load[AppConfig]

  def loadOrThrow: AppConfig =
    ConfigSource.default.loadOrThrow[AppConfig]

  // Load with environment override
  def loadWithEnv(env: String): Either[pureconfig.error.ConfigReaderFailures, AppConfig] =
    ConfigSource.default
      .withFallback(ConfigSource.resources(s"application.$env.conf"))
      .load[AppConfig]
```

```hocon
# src/main/resources/application.conf
server {
  host = "0.0.0.0"
  host = ${?SERVER_HOST}
  port = 8080
  port = ${?SERVER_PORT}
  idle-timeout = 60s
  shutdown-timeout = 30s
}

database {
  driver = "org.postgresql.Driver"
  url = "jdbc:postgresql://localhost:5432/myapp"
  url = ${?DATABASE_URL}
  user = "postgres"
  user = ${?DATABASE_USER}
  password = "postgres"
  password = ${?DATABASE_PASSWORD}
  pool-size = 10
  pool-size = ${?DATABASE_POOL_SIZE}
  connection-timeout = 30s
}

auth {
  jwt-secret = "change-me-in-production"
  jwt-secret = ${?JWT_SECRET}
  token-expiry = 1h
  refresh-token-expiry = 30d
  issuer = "my-app"
}

features {
  enable-new-checkout = false
  enable-new-checkout = ${?FEATURE_NEW_CHECKOUT}
  enable-beta-features = false
  max-upload-size-mb = 10
}
```

```hocon
# src/main/resources/application.prod.conf
# Production overrides - inherits from application.conf
database {
  pool-size = 32
  connection-timeout = 10s
}

server {
  idle-timeout = 120s
}

features {
  enable-beta-features = false
}
```

### E. Logging and Observability

```scala
// ✅ CORRECT: Structured logging with log4cats
import org.typelevel.log4cats.{Logger, SelfAwareStructuredLogger}
import org.typelevel.log4cats.slf4j.Slf4jLogger

// Create logger instance
given SelfAwareStructuredLogger[IO] = Slf4jLogger.getLogger[IO]

// ✅ CORRECT: Contextual logging with MDC
import org.typelevel.log4cats.extras.LogLevel

def processRequest[F[_]: Logger: Monad](request: Request): F[Response] =
  for
    _      <- Logger[F].info(s"Processing request: ${request.method} ${request.path}")
    result <- handleRequest(request)
    _      <- Logger[F].info(s"Request completed: ${result.status}")
  yield result

// ✅ CORRECT: Structured logging with context
def processOrder[F[_]: Logger: Monad](orderId: OrderId, userId: UserId): F[Unit] =
  val ctx = Map("orderId" -> orderId.toString, "userId" -> userId.value.toString)
  for
    _ <- Logger[F].info(ctx)(s"Starting order processing")
    _ <- validateOrder(orderId)
    _ <- Logger[F].info(ctx)(s"Order validated successfully")
    _ <- chargePayment(orderId)
    _ <- Logger[F].info(ctx)(s"Payment charged")
    _ <- Logger[F].info(ctx)(s"Order processing complete")
  yield ()

// ✅ CORRECT: Error logging with context
def handleError[F[_]: Logger: ApplicativeError[*[_], Throwable]](
  operation: String,
  error: Throwable
): F[Unit] =
  error match
    case e: java.net.ConnectException =>
      Logger[F].warn(e)(s"Connection failed during $operation, will retry")
    case e: java.util.concurrent.TimeoutException =>
      Logger[F].warn(e)(s"Timeout during $operation")
    case e =>
      Logger[F].error(e)(s"Unexpected error during $operation")

// ✅ CORRECT: Logging middleware for HTTP
import org.http4s.HttpApp

def loggingMiddleware[F[_]: Logger: Temporal](app: HttpApp[F]): HttpApp[F] =
  HttpApp: request =>
    for
      start    <- Temporal[F].monotonic
      _        <- Logger[F].info(s">>> ${request.method} ${request.uri}")
      response <- app.run(request)
      end      <- Temporal[F].monotonic
      elapsed  = end - start
      _        <- Logger[F].info(
                    s"<<< ${request.method} ${request.uri} -> ${response.status} (${elapsed.toMillis}ms)"
                  )
    yield response
```

```xml
<!-- src/main/resources/logback.xml -->
<configuration>
  <appender name="STDOUT" class="ch.qos.logback.core.ConsoleAppender">
    <encoder class="net.logstash.logback.encoder.LogstashEncoder">
      <includeMdcKeyName>orderId</includeMdcKeyName>
      <includeMdcKeyName>userId</includeMdcKeyName>
      <includeMdcKeyName>traceId</includeMdcKeyName>
    </encoder>
  </appender>

  <root level="INFO">
    <appender-ref ref="STDOUT" />
  </root>

  <logger name="com.example" level="DEBUG" />
  <logger name="org.http4s" level="INFO" />
  <logger name="doobie" level="INFO" />
</configuration>
```

### F. Metrics and Health Checks

```scala
// ✅ CORRECT: Health check endpoint
import cats.effect.IO
import io.circe.syntax.*

case class HealthStatus(
  status: String,
  database: ComponentHealth,
  cache: ComponentHealth,
  uptime: Long
) derives io.circe.Encoder.AsObject

case class ComponentHealth(
  status: String,
  latencyMs: Option[Long] = None,
  message: Option[String] = None
) derives io.circe.Encoder.AsObject

def healthCheck(db: Database[IO], cache: Cache[IO]): IO[HealthStatus] =
  for
    startTime  <- IO.monotonic
    dbHealth   <- checkDatabase(db).handleError(e =>
                    ComponentHealth("unhealthy", message = Some(e.getMessage)))
    cacheHealth <- checkCache(cache).handleError(e =>
                    ComponentHealth("unhealthy", message = Some(e.getMessage)))
    uptime     <- IO.monotonic.map(now => (now - startTime).toSeconds)
    overall    = if dbHealth.status == "healthy" && cacheHealth.status == "healthy"
                 then "healthy" else "degraded"
  yield HealthStatus(overall, dbHealth, cacheHealth, uptime)

def checkDatabase(db: Database[IO]): IO[ComponentHealth] =
  for
    start   <- IO.monotonic
    _       <- db.query("SELECT 1")
    elapsed <- IO.monotonic.map(now => (now - start).toMillis)
  yield ComponentHealth("healthy", latencyMs = Some(elapsed))
```

---

## 10. Security & Dependency Management (MANDATORY)

### A. Automated Dependency Management

**Use sbt to manage and pin dependencies:**

```bash
# Compile and resolve dependencies
sbt compile

# Add a new dependency (edit build.sbt, then)
sbt update

# Check for dependency updates
sbt dependencyUpdates

# Show dependency tree
sbt dependencyTree
```

### B. Vulnerability Scanning & Security

**Mandatory security checks for ALL Scala projects:**

1. **Vulnerability Scan**:
   ```bash
   # Scan for known vulnerabilities (requires sbt-dependency-check plugin)
   sbt dependencyCheck
   ```
   - Agents MUST fix all HIGH/CRITICAL vulnerabilities before delivery.

2. **Supply Chain Audit**:
   - Verify pinned versions in `build.sbt`
   - Audit licenses for compliance
   - Use `sbt dependencyTree` to inspect transitive dependencies

### C. Dependency File

```scala
// build.sbt
ThisBuild / scalaVersion := "3.8.0"

libraryDependencies ++= Seq(
  "org.typelevel" %% "cats-core"   % "2.10.0",
  "org.typelevel" %% "cats-effect" % "3.5.2",
  "org.scalatest" %% "scalatest"   % "3.2.17" % Test
)

// project/plugins.sbt
addSbtPlugin("net.vonbuchholtz" % "sbt-dependency-check" % "5.1.0")
addSbtPlugin("com.timushev.sbt" % "sbt-updates"          % "0.6.4")
```

---

## 11. Design Patterns (MANDATORY)

### A. Tagless Final

```scala
// ✅ CORRECT: Define algebras as traits parameterized by effect type
trait UserRepository[F[_]]:
  def findById(id: UserId): F[Option[User]]
  def findByEmail(email: Email): F[Option[User]]
  def save(user: User): F[User]
  def delete(id: UserId): F[Boolean]

trait NotificationService[F[_]]:
  def sendEmail(to: Email, subject: String, body: String): F[Unit]
  def sendPush(userId: UserId, message: String): F[Unit]

// ✅ CORRECT: Business logic uses abstract F[_] with required constraints
class UserRegistration[F[_]: Monad](
  repo: UserRepository[F],
  notifications: NotificationService[F],
  logger: Logger[F]
):
  def register(name: String, email: Email): F[Either[RegistrationError, User]] =
    for
      existing <- repo.findByEmail(email)
      result <- existing match
        case Some(_) =>
          logger.warn(s"Registration attempt with existing email: ${email.value}") *>
            Monad[F].pure(Left(RegistrationError.EmailTaken(email)))
        case None =>
          val user = User(UserId(generateId()), name, email)
          for
            saved <- repo.save(user)
            _     <- notifications.sendEmail(email, "Welcome!", s"Hello $name!")
            _     <- logger.info(s"User registered: ${saved.id}")
          yield Right(saved)
    yield result

// ✅ CORRECT: Production implementation with IO
class DoobieUserRepository(xa: Transactor[IO]) extends UserRepository[IO]:
  def findById(id: UserId): IO[Option[User]] =
    sql"SELECT id, name, email, role FROM users WHERE id = ${id.value}"
      .query[User]
      .option
      .transact(xa)

  def findByEmail(email: Email): IO[Option[User]] =
    sql"SELECT id, name, email, role FROM users WHERE email = ${email.value}"
      .query[User]
      .option
      .transact(xa)

  def save(user: User): IO[User] =
    sql"INSERT INTO users (id, name, email, role) VALUES (${user.id.value}, ${user.name}, ${user.email.value}, ${user.role.toString})"
      .update
      .run
      .transact(xa)
      .as(user)

  def delete(id: UserId): IO[Boolean] =
    sql"DELETE FROM users WHERE id = ${id.value}"
      .update
      .run
      .transact(xa)
      .map(_ > 0)

// ✅ CORRECT: Test implementation - pure, no database needed
class InMemoryUserRepository(ref: Ref[IO, Map[UserId, User]]) extends UserRepository[IO]:
  def findById(id: UserId): IO[Option[User]] =
    ref.get.map(_.get(id))

  def findByEmail(email: Email): IO[Option[User]] =
    ref.get.map(_.values.find(_.email == email))

  def save(user: User): IO[User] =
    ref.update(_ + (user.id -> user)).as(user)

  def delete(id: UserId): IO[Boolean] =
    ref.modify: m =>
      if m.contains(id) then (m - id, true)
      else (m, false)
```

### B. Hexagonal / Clean Architecture

```scala
// ✅ CORRECT: Ports (domain traits) - define contracts, no framework dependencies

// Primary port (driven by external actors - API, CLI, etc.)
trait OrderManagement[F[_]]:
  def placeOrder(request: PlaceOrderRequest): F[Either[OrderError, Order]]
  def cancelOrder(orderId: OrderId): F[Either[OrderError, Unit]]
  def getOrderStatus(orderId: OrderId): F[Either[OrderError, OrderStatus]]

// Secondary ports (driven by the application - DB, external APIs, etc.)
trait OrderRepository[F[_]]:
  def save(order: Order): F[Order]
  def findById(id: OrderId): F[Option[Order]]
  def updateStatus(id: OrderId, status: OrderStatus): F[Unit]

trait PaymentGateway[F[_]]:
  def charge(amount: BigDecimal, method: PaymentMethod): F[Either[PaymentError, PaymentReceipt]]
  def refund(receiptId: ReceiptId): F[Either[PaymentError, Unit]]

trait InventoryService[F[_]]:
  def reserve(items: List[OrderItem]): F[Either[InventoryError, ReservationId]]
  def release(reservationId: ReservationId): F[Unit]

// ✅ CORRECT: Domain model - pure case classes, no annotations
case class Order(
  id: OrderId,
  customerId: CustomerId,
  items: List[OrderItem],
  status: OrderStatus,
  total: BigDecimal,
  createdAt: Instant
):
  def canBeCancelled: Boolean = status match
    case OrderStatus.Pending | OrderStatus.Confirmed => true
    case _ => false

  def itemCount: Int = items.map(_.quantity).sum

enum OrderStatus:
  case Pending, Confirmed, Shipped, Delivered, Cancelled

case class OrderItem(
  productId: ProductId,
  quantity: Int,
  unitPrice: BigDecimal
):
  def subtotal: BigDecimal = unitPrice * quantity

// ✅ CORRECT: Application service (use case implementation)
class OrderService[F[_]: Monad: Logger](
  orders: OrderRepository[F],
  payments: PaymentGateway[F],
  inventory: InventoryService[F]
) extends OrderManagement[F]:

  def placeOrder(request: PlaceOrderRequest): F[Either[OrderError, Order]] =
    (for
      reservation <- inventory.reserve(request.items)
                       .flatMap(_.leftMap(e => OrderError.InsufficientStock(e.message)).liftTo[F])
      receipt     <- payments.charge(request.total, request.paymentMethod)
                       .flatMap(_.leftMap(e => OrderError.PaymentFailed(e.message)).liftTo[F])
                       .onError(_ => inventory.release(reservation).void)
      order       = Order(
                      id = OrderId.generate(),
                      customerId = request.customerId,
                      items = request.items,
                      status = OrderStatus.Confirmed,
                      total = request.total,
                      createdAt = Instant.now()
                    )
      saved       <- orders.save(order)
      _           <- Logger[F].info(s"Order placed: ${saved.id}")
    yield Right(saved))
      .handleErrorWith: e =>
        Logger[F].error(e)(s"Failed to place order") *>
          Monad[F].pure(Left(OrderError.Internal(e.getMessage)))

  def cancelOrder(orderId: OrderId): F[Either[OrderError, Unit]] =
    for
      maybeOrder <- orders.findById(orderId)
      result <- maybeOrder match
        case None => Monad[F].pure(Left(OrderError.NotFound(orderId)))
        case Some(order) if !order.canBeCancelled =>
          Monad[F].pure(Left(OrderError.CannotCancel(orderId, order.status)))
        case Some(order) =>
          orders.updateStatus(orderId, OrderStatus.Cancelled).map(Right(_))
    yield result

  def getOrderStatus(orderId: OrderId): F[Either[OrderError, OrderStatus]] =
    orders.findById(orderId).map:
      case Some(order) => Right(order.status)
      case None        => Left(OrderError.NotFound(orderId))

// ✅ CORRECT: Adapter wiring at the edge of the application
object AppWiring:
  def make(config: AppConfig): Resource[IO, OrderManagement[IO]] =
    for
      xa          <- HikariTransactor.newHikariTransactor[IO](
                       config.database.driver, config.database.url,
                       config.database.user, config.database.password,
                       ExecutionContexts.synchronous
                     )
      httpClient  <- EmberClientBuilder.default[IO].build
      orderRepo    = DoobieOrderRepository(xa)
      paymentGw    = StripePaymentGateway(httpClient, config.auth.stripeKey)
      inventorySvc = HttpInventoryService(httpClient, config.inventoryUrl)
      given Logger[IO] = Slf4jLogger.getLogger[IO]
    yield OrderService[IO](orderRepo, paymentGw, inventorySvc)
```

### C. Smart Constructor Pattern

```scala
// ✅ CORRECT: Prevent invalid state via private constructors + factory methods
final case class PositiveInt private (value: Int):
  def +(other: PositiveInt): PositiveInt =
    PositiveInt(value + other.value)  // Safe: sum of positives is positive

object PositiveInt:
  def from(value: Int): Either[String, PositiveInt] =
    if value > 0 then Right(new PositiveInt(value))
    else Left(s"Expected positive integer, got $value")

  def unsafeFrom(value: Int): PositiveInt =
    from(value).fold(msg => throw new IllegalArgumentException(msg), identity)

final case class NonEmptyList[A] private (head: A, tail: List[A]):
  def toList: List[A] = head :: tail
  def map[B](f: A => B): NonEmptyList[B] = NonEmptyList(f(head), tail.map(f))
  def size: Int = 1 + tail.size

object NonEmptyList:
  def of[A](head: A, tail: A*): NonEmptyList[A] =
    NonEmptyList(head, tail.toList)

  def fromList[A](list: List[A]): Option[NonEmptyList[A]] =
    list match
      case head :: tail => Some(NonEmptyList(head, tail))
      case Nil          => None

// ✅ CORRECT: Refined types for business rules
final case class OrderQuantity private (value: Int)
object OrderQuantity:
  def from(value: Int): Either[String, OrderQuantity] =
    if value >= 1 && value <= 10000 then Right(OrderQuantity(value))
    else Left(s"Order quantity must be between 1 and 10000, got $value")
```

### D. Dependency Injection with Reader Pattern

```scala
// ✅ CORRECT: Reader monad for dependency injection (lightweight alternative)
import cats.data.ReaderT
import cats.effect.IO

case class AppEnv(
  userRepo: UserRepository[IO],
  orderRepo: OrderRepository[IO],
  logger: Logger[IO],
  config: AppConfig
)

type AppIO[A] = ReaderT[IO, AppEnv, A]

// Use in services
def findUserOrFail(id: UserId): AppIO[User] =
  ReaderT: env =>
    env.userRepo.findById(id).flatMap:
      case Some(user) => IO.pure(user)
      case None       => IO.raiseError(new NoSuchElementException(s"User $id not found"))

def logAndProcess(id: UserId): AppIO[ProcessedUser] =
  for
    env  <- ReaderT.ask[IO, AppEnv]
    _    <- ReaderT.liftF(env.logger.info(s"Processing user $id"))
    user <- findUserOrFail(id)
    result <- ReaderT.liftF(processUser(user))
  yield result

// Run with environment
val program: IO[ProcessedUser] = logAndProcess(UserId(1)).run(appEnv)
```

---

## 12. Architecture: HTTP API Pattern (RECOMMENDED)

### A. http4s Routes

```scala
import org.http4s.*
import org.http4s.dsl.io.*
import org.http4s.circe.*
import org.http4s.circe.CirceEntityCodec.given
import io.circe.generic.auto.*

// ✅ CORRECT: Define routes as a function from services to HttpRoutes
def userRoutes(userService: UserService[IO]): HttpRoutes[IO] =
  HttpRoutes.of[IO]:
    case GET -> Root / "users" / LongVar(id) =>
      userService.findById(UserId(id)).flatMap:
        case Some(user) => Ok(user.toResponse)
        case None       => NotFound(ErrorResponse("User not found"))

    case req @ POST -> Root / "users" =>
      for
        body     <- req.as[CreateUserRequest]
        result   <- userService.create(body.name, body.email)
        response <- result match
          case Right(user) => Created(user.toResponse)
          case Left(UserError.InvalidEmail(e)) =>
            BadRequest(ErrorResponse(s"Invalid email: $e"))
          case Left(UserError.AlreadyExists(_)) =>
            Conflict(ErrorResponse("Email already registered"))
          case Left(error) =>
            InternalServerError(ErrorResponse(error.toMessage))
      yield response

    case DELETE -> Root / "users" / LongVar(id) =>
      userService.delete(UserId(id)).flatMap:
        case true  => NoContent()
        case false => NotFound(ErrorResponse("User not found"))

// ✅ CORRECT: Compose routes
def allRoutes(services: Services[IO]): HttpRoutes[IO] =
  userRoutes(services.userService) <+>
    orderRoutes(services.orderService) <+>
    healthRoutes(services.healthService)

// ✅ CORRECT: Request/Response DTOs separate from domain
case class CreateUserRequest(name: String, email: String)
case class UserResponse(id: Long, name: String, email: String, role: String)
case class ErrorResponse(message: String, code: Option[String] = None)

extension (user: User)
  def toResponse: UserResponse =
    UserResponse(user.id.value, user.name, user.email.value, user.role.toString)
```

### B. Authentication Middleware

```scala
import org.http4s.*
import org.http4s.server.AuthMiddleware
import org.http4s.headers.Authorization
import cats.data.{Kleisli, OptionT}

case class AuthenticatedUser(id: UserId, role: Role)

// ✅ CORRECT: Auth middleware extracts user from JWT
def authMiddleware(jwtService: JwtService[IO]): AuthMiddleware[IO, AuthenticatedUser] =
  val authUser: Kleisli[OptionT[IO, *], Request[IO], AuthenticatedUser] =
    Kleisli: request =>
      OptionT:
        request.headers.get[Authorization] match
          case Some(Authorization(Credentials.Token(AuthScheme.Bearer, token))) =>
            jwtService.validate(token).map(_.toOption)
          case _ =>
            IO.pure(None)

  AuthMiddleware(authUser)

// Use authenticated routes
def protectedRoutes(
  userService: UserService[IO]
): AuthedRoutes[AuthenticatedUser, IO] =
  AuthedRoutes.of:
    case GET -> Root / "me" as user =>
      userService.findById(user.id).flatMap:
        case Some(u) => Ok(u.toResponse)
        case None    => NotFound()

    case req @ PUT -> Root / "me" / "profile" as user =>
      for
        body     <- req.req.as[UpdateProfileRequest]
        result   <- userService.updateProfile(user.id, body)
        response <- result.fold(
          err => BadRequest(ErrorResponse(err.toMessage)),
          _   => Ok("Profile updated")
        )
      yield response
```

---

## 13. Deployment Checklist

### Code Quality
- [ ] No compiler warnings (`-Xfatal-warnings` enabled)
- [ ] Scalafmt formatting applied (`sbt scalafmtCheckAll`)
- [ ] Scalafix linting passed (`sbt scalafix --check`)
- [ ] All tests passing (unit, integration, property-based)
- [ ] No deprecated API usage
- [ ] Code coverage meets threshold (>= 80%)

### Type Safety
- [ ] Exhaustive pattern matching (no catch-all `_` that silently drops cases)
- [ ] Proper error types defined (enum-based error hierarchies)
- [ ] No unsafe operations (`.get`, `asInstanceOf`, `null`)
- [ ] Opaque types used for domain identifiers
- [ ] Smart constructors validate invariants

### Performance
- [ ] Proper use of `lazy val` for expensive initialization
- [ ] `@tailrec` annotation on tail-recursive functions
- [ ] `.view` used for lazy intermediate collection transformations
- [ ] Bounded parallelism for concurrent operations (`parTraverseN`)
- [ ] No accidental blocking on the compute thread pool

### Effects & Resources
- [ ] All resources managed via `Resource.make` or `bracket`
- [ ] Fiber cancellation supported (no uncancelable blocks unless necessary)
- [ ] Error handling comprehensive (typed errors, no swallowed exceptions)
- [ ] Timeouts configured for all external calls
- [ ] Retry policies with backoff for transient failures

### Observability
- [ ] Structured logging with context (trace ID, user ID)
- [ ] Health check endpoint implemented
- [ ] Key business metrics tracked
- [ ] Error rates monitored

### Security
- [ ] Secrets loaded from environment variables, not hardcoded
- [ ] Dependencies scanned for vulnerabilities (`sbt dependencyCheck`)
- [ ] Input validation at API boundaries
- [ ] Authentication and authorization enforced on protected routes

---

## 14. Quick Reference

```scala
// Option
option.map(f)                // Transform value if present
option.flatMap(f)            // Chain optional operations
option.getOrElse(default)    // Unwrap with fallback
option.fold(default)(f)      // Transform or default
option.toRight(error)        // Convert to Either
option.filter(predicate)     // None if predicate fails
option.collect { case ... }  // Partial function + filter
option.orElse(fallback)      // Try alternative Option

// Either
either.map(f)                // Transform right side
either.flatMap(f)            // Chain Either operations
either.leftMap(f)            // Transform left (error) side
either.fold(onLeft, onRight) // Handle both cases
either.toOption              // Discard error info
either.swap                  // Swap Left and Right
either.merge                 // Get value when both sides same type

// List
list.map(f)                  // Transform each element
list.flatMap(f)              // Transform + flatten
list.filter(p)               // Keep elements matching predicate
list.collect { case ... => } // Filter + transform in one step
list.foldLeft(z)(f)          // Reduce left to right
list.groupBy(f)              // Group into Map by key
list.traverse(f)             // Apply effectful function (Cats)
list.parTraverse(f)          // Apply concurrently (Cats Effect)
list.parTraverseN(n)(f)      // Bounded concurrent traversal

// IO (Cats Effect)
IO.pure(value)               // Lift pure value
IO.delay(computation)        // Suspend side-effecting computation
IO.blocking(computation)     // Run on blocking thread pool
IO.raiseError(error)         // Create failed IO
IO.ref(initial)              // Create thread-safe mutable reference
IO.sleep(duration)           // Suspend for a duration
io.flatMap(f)                // Chain IO operations
io.handleErrorWith(f)        // Recover from errors
io.attempt                   // IO[Either[Throwable, A]]
io.timeout(duration)         // Fail if exceeds duration
io.timeoutTo(duration, fb)   // Fallback if exceeds duration
io.guarantee(finalizer)      // Run finalizer regardless of outcome
io.onCancel(handler)         // Run handler if cancelled
(io1, io2).parTupled         // Run concurrently, collect results
(io1, io2).parMapN(f)        // Run concurrently, combine results
IO.race(io1, io2)            // Run concurrently, take winner

// Resource (Cats Effect)
Resource.make(acquire)(release)   // Bracket-style resource
Resource.eval(io)                 // Lift IO into Resource
resource.use(f)                   // Use resource, then release
resource.flatMap(f)               // Chain resources
(r1, r2).tupled                   // Combine resources

// Scala 3 syntax
given instance: MyType = value    // Define given instance
def f(using x: MyType): Unit     // Accept context parameter
summon[MyType]                    // Retrieve given instance
extension (x: MyType)            // Add methods to existing type
  def newMethod: Result = ...
opaque type Name = Underlying     // Zero-cost type wrapper
enum Color:                       // ADT definition
  case Red, Green, Blue
enum Tree[+A]:                    // Parameterized ADT
  case Leaf(value: A)
  case Node(left: Tree[A], right: Tree[A])

// Pattern matching
x match
  case Pattern(a, b) => ..        // Destructure
  case _ if guard => ..           // Guard clause
  case p @ Pattern(_) => ..      // Bind matched value
  case _: SpecificType => ..     // Type check
  case _ => ..                    // Catch-all (use sparingly)

// For-comprehension
for
  a <- optionA                    // flatMap
  b <- optionB if b > 0          // withFilter
  c = a + b                       // map (no <-)
yield c                           // Final map

// sbt commands
// sbt compile                    // Compile all modules
// sbt test                       // Run all tests
// sbt "testOnly *UserSpec"       // Run specific test
// sbt scalafmtAll                // Format all sources
// sbt dependencyUpdates          // Check for newer versions
// sbt dependencyTree             // Show dependency graph
// sbt docker:publishLocal        // Build Docker image locally
// sbt "project api" console      // REPL with api module on classpath
```

---

## 15. Why This Configuration Works

1. **Exhaustive Pattern Matching with ADTs**: Defining domain models as sealed traits and enums forces the compiler to verify every case is handled. Adding a new variant produces compile errors at every incomplete match, making illegal states unrepresentable and refactoring safe.

2. **Either for Error Handling Over Exceptions**: Using `Either[Error, A]` makes error paths visible in function signatures, composable via for-comprehensions, and impossible to accidentally ignore. This eliminates the hidden control flow that try/catch introduces.

3. **Cats Effect IO for Referential Transparency**: Wrapping side effects in `IO` ensures that programs are descriptions of computation rather than eager execution. This makes concurrent code composable, testable without mocking, and safe from resource leaks via `Resource.make`.

4. **Opaque Types for Domain Safety**: Opaque types like `UserId` and `Email` provide compile-time type safety with zero runtime overhead. Passing a `UserId` where an `Email` is expected is a compile error, preventing entire categories of data-mixing bugs.

5. **sbt with Scalafmt and -Xfatal-warnings**: Treating all compiler warnings as errors ensures deprecated APIs, unused imports, and unchecked casts are fixed immediately rather than accumulating. Combined with Scalafmt auto-formatting, code quality is enforced mechanically.

6. **Tagless Final for Testability**: Abstracting over the effect type `F[_]` makes services testable with synchronous interpreters and swappable for different runtimes. Production uses `IO`, tests use in-memory implementations -- no mocking frameworks required.

7. **Hexagonal Architecture with Traits as Ports**: Defining service boundaries as traits decouples business logic from infrastructure. Database, HTTP, and messaging adapters implement these traits, making the domain layer framework-agnostic and independently testable.

8. **Smart Constructors for Domain Integrity**: Using private constructors with factory methods (`from`, `apply`) that validate invariants guarantees that invalid domain objects cannot exist at runtime. This pushes validation to the edges and keeps the core logic clean.

9. **PureConfig with HOCON for Type-Safe Configuration**: Deriving configuration readers from case classes ensures that misconfigured applications fail fast at startup with clear error messages, rather than silently misbehaving at runtime with wrong values.

10. **Property-Based Testing with ScalaCheck**: Generating random inputs to verify algebraic properties (commutativity, associativity, roundtrip serialization) catches edge cases that example-based tests miss. Combined with shrinking, failures produce minimal reproducible examples.

---

**Last Updated:** 2026-04-13
**Version:** 2.0
**Maintainer:** Scala Team


**End of Scala Development Guidelines**
