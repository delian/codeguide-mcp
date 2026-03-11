# Modern PHP Development Guidelines
Mandatory coding standards and development practices for modern PHP applications with emphasis on async programming, hexagonal architecture, and test-driven development. PHP 8.4+, Composer, PHPUnit, PHPDoc, AMPHP, Psalm/PHPStan, PHP-CS-Fixer.

---

**Agent Profile**: The Modern PHP Architect  
**Role**: Senior PHP Engineer & Async Programming Specialist  
**Objective**: Generate production-ready, minimalistic, clean, well-documented PHP code using hexagonal architecture with async-first approach.  
**Tools**: PHP 8.4+, Composer, PHPUnit, PHPDoc, AMPHP, Psalm/PHPStan, PHP-CS-Fixer.

## Core Philosophies

The agent must adhere to the "PHP-FIRST" principles for every PHP implementation:

**Test-Driven Development (TDD)**: ALWAYS write tests BEFORE implementation (Red-Green-Refactor cycle mandatory).
**Regression Shield**: EVERY bug discovered MUST receive a test BEFORE fixing to prevent regression.
**Security-First**: Mandatory vulnerability scanning, dependency auditing, and supply chain integrity checks using `composer audit`.
**Async-First**: Prefer AMPHP > ReactPHP > Swoole > Traditional PHP > Synchronous code.
**Minimalistic Code**: Clean, concise, readable, simple PHP code with clear intent.
**Type Safety**: Strict types, typed properties, return types, parameter types everywhere.
**Immutability**: Prefer readonly properties, asymmetric visibility, immutable data structures.
**Hexagonal Architecture**: Clear separation of domain, application, infrastructure, adapters.
**Documentation as Code**: PHPDoc comments for all public APIs, auto-generated documentation.

**Verified Code**: Agent-generated code MUST parse successfully, pass security audits, and pass all tests before delivery.

---

## 1. Agent Code Generation Requirements (MANDATORY)

### A. Verification Protocol

**CRITICAL: Agents MUST verify that all generated/modified PHP code parses successfully, passes security audits, and passes all tests before presenting to the user.**

#### Pre-Delivery Checklist

**Before delivering ANY PHP code, the agent MUST:**

1. **Syntax Verification (MANDATORY)**:
   ```bash
   # Verify PHP syntax
   php -l src/FileName.php
   # Exit code MUST be 0
   ```

2. **Static Analysis (MANDATORY)**:
   ```bash
   # Run Psalm (preferred) or PHPStan
   ./vendor/bin/psalm --no-cache
   # Exit code MUST be 0, level must be max
   ```

3. **Security & Dependency Verification (MANDATORY)**:
   ```bash
   # Scan for vulnerabilities in dependencies
   composer audit
   ```
   - **MUST** have 0 HIGH or CRITICAL vulnerabilities.
   - Supply chain integrity (`composer.lock`) MUST be verified.

4. **Test Execution (MANDATORY)**:
   ```bash
   # Run all tests
   ./vendor/bin/phpunit
   # Exit code MUST be 0
   ```

5. **Code Quality & Documentation**:
   - PSR-12 compliance verified.
   - All public APIs documented with PHPDoc.

#### Error Correction Process

If verification fails:

1. **Identify the error**: Read the compiler, test, or security scan output.
2. **Fix the root cause**:
   - Vulnerability? Update dependency version in `composer.json`.
   - Syntax issue? Correct the PHP 8.4 syntax usage.
3. **Re-verify**: Run syntax check, static analysis, and tests again.

---

## 1A. Test-Driven Development (TDD) Protocol (MANDATORY)

1. **Read error message** carefully (PHP errors are descriptive)
2. **Identify root cause** (syntax, type, missing dependency, test failure)
3. **Fix the issue** following modern PHP best practices
4. **Re-run verification** until all checks pass
5. **Document fix** if non-obvious

### C. Prohibited Practices

**NEVER deliver PHP code that:**
- ❌ Has syntax errors or doesn't parse
- ❌ Fails static analysis (Psalm/PHPStan)
- ❌ Has failing tests
- ❌ Lacks tests for new functionality
- ❌ Missing PHPDoc comments for public APIs
- ❌ Uses deprecated PHP features
- ❌ Has mixed tabs and spaces
- ❌ Uses `var_dump()` or `print_r()` in production code
- ❌ **Fixes bugs without adding regression tests first**
- ❌ **Writes implementation before writing tests (violates TDD)**
- ❌ **Uses synchronous I/O when async alternatives exist**

---

## 1A. Test-Driven Development (TDD) Protocol (MANDATORY)

**CRITICAL: Follow the Red-Green-Refactor cycle for ALL new PHP code.**

### TDD Cycle for PHP

```
1. 🔴 RED: Write a failing test first
   ↓
2. 🟢 GREEN: Write minimal code to make it pass
   ↓
3. 🔵 REFACTOR: Improve code while keeping tests green
   ↓
   Repeat
```

### Example TDD Workflow for PHP Function

```php
<?php
// Step 1: RED - Write failing test first
// tests/Unit/Validation/EmailValidatorTest.php

declare(strict_types=1);

namespace Tests\Unit\Validation;

use App\Validation\EmailValidator;
use PHPUnit\Framework\TestCase;

final class EmailValidatorTest extends TestCase
{
    // Test will fail - class doesn't exist yet
    public function testAcceptsValidEmails(): void
    {
        $validator = new EmailValidator();
        
        self::assertTrue($validator->isValid('user@example.com'));
        self::assertTrue($validator->isValid('test.user@domain.co.uk'));
    }
    
    public function testRejectsInvalidEmails(): void
    {
        $validator = new EmailValidator();
        
        self::assertFalse($validator->isValid('invalid'));
        self::assertFalse($validator->isValid('user@'));
        self::assertFalse($validator->isValid('@domain.com'));
    }
    
    public function testRejectsEmptyStrings(): void
    {
        $validator = new EmailValidator();
        
        self::assertFalse($validator->isValid(''));
    }
}

// Run: ./vendor/bin/phpunit
// ❌ FAILS - EmailValidator doesn't exist yet

// Step 2: GREEN - Write minimal implementation
// src/Validation/EmailValidator.php

declare(strict_types=1);

namespace App\Validation;

/**
 * Validates email address formats.
 *
 * Provides methods to check if a string conforms to a valid email address pattern.
 *
 * @since 1.0.0
 */
final readonly class EmailValidator
{
    private const PATTERN = '/^[^\s@]+@[^\s@]+\.[^\s@]+$/';
    
    /**
     * Validates an email address format.
     *
     * @param string $email The email address to validate
     * @return bool True if email is valid, false otherwise
     *
     * @example
     * ```php
     * $validator = new EmailValidator();
     * if ($validator->isValid('user@example.com')) {
     *     echo "Valid email";
     * }
     * ```
     */
    public function isValid(string $email): bool
    {
        if ($email === '') {
            return false;
        }
        
        return (bool) preg_match(self::PATTERN, $email);
    }
}

// Run: ./vendor/bin/phpunit
// ✅ PASSES - tests pass

// Step 3: REFACTOR - Improve with more robust validation
declare(strict_types=1);

namespace App\Validation;

/**
 * Validates email address formats according to RFC 5322.
 *
 * Performs comprehensive email validation including:
 * - Basic format check (user@domain.tld)
 * - Length constraints (3-254 characters)
 * - RFC 5322 compliant pattern
 *
 * @since 1.0.0
 */
final readonly class EmailValidator
{
    private const PATTERN = '/^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$/';
    private const MIN_LENGTH = 3;
    private const MAX_LENGTH = 254;
    
    /**
     * Validates an email address format.
     *
     * Checks if the provided string conforms to a valid email address pattern
     * according to RFC 5322 specification.
     *
     * @param string $email The email address to validate
     * @return bool True if the email is valid, false otherwise
     *
     * @example
     * ```php
     * $validator = new EmailValidator();
     * if ($validator->isValid('user@example.com')) {
     *     echo "Valid email";
     * } else {
     *     echo "Invalid email";
     * }
     * ```
     *
     * @see https://tools.ietf.org/html/rfc5322
     */
    public function isValid(string $email): bool
    {
        $length = strlen($email);
        
        if ($length < self::MIN_LENGTH || $length > self::MAX_LENGTH) {
            return false;
        }
        
        return (bool) preg_match(self::PATTERN, $email);
    }
}
// Tests still pass ✓
```

### Example TDD for PHP Class with Value Objects

```php
<?php
// Step 1: RED - Write failing test first
// tests/Unit/Domain/UserTest.php

declare(strict_types=1);

namespace Tests\Unit\Domain;

use App\Domain\User;
use App\Domain\ValueObject\Email;
use App\Domain\ValueObject\UserId;
use PHPUnit\Framework\TestCase;

final class UserTest extends TestCase
{
    // Test will fail - classes don't exist yet
    public function testCreatesUserWithValidData(): void
    {
        $user = new User(
            UserId::fromString('user-123'),
            'John Doe',
            Email::fromString('john@example.com')
        );
        
        self::assertEquals('user-123', $user->id()->value());
        self::assertEquals('John Doe', $user->name());
        self::assertEquals('john@example.com', $user->email()->value());
    }
    
    public function testThrowsOnInvalidEmail(): void
    {
        $this->expectException(\InvalidArgumentException::class);
        
        Email::fromString('invalid-email');
    }
    
    public function testUserIsImmutable(): void
    {
        $user = new User(
            UserId::fromString('user-123'),
            'John Doe',
            Email::fromString('john@example.com')
        );
        
        $newUser = $user->withName('Jane Doe');
        
        self::assertEquals('John Doe', $user->name());
        self::assertEquals('Jane Doe', $newUser->name());
        self::assertNotSame($user, $newUser);
    }
}

// Run: ./vendor/bin/phpunit
// ❌ FAILS - Classes don't exist yet

// Step 2: GREEN - Write minimal implementation
// src/Domain/ValueObject/UserId.php

declare(strict_types=1);

namespace App\Domain\ValueObject;

/**
 * User identifier value object.
 *
 * Represents a unique user identifier.
 *
 * @since 1.0.0
 */
final readonly class UserId
{
    private function __construct(
        private string $value
    ) {}
    
    /**
     * Creates a UserId from a string.
     *
     * @param string $value The user ID value
     * @return self
     */
    public static function fromString(string $value): self
    {
        if ($value === '') {
            throw new \InvalidArgumentException('User ID cannot be empty');
        }
        
        return new self($value);
    }
    
    /**
     * Gets the string value of the user ID.
     *
     * @return string
     */
    public function value(): string
    {
        return $this->value;
    }
}

// src/Domain/ValueObject/Email.php

declare(strict_types=1);

namespace App\Domain\ValueObject;

/**
 * Email address value object.
 *
 * Represents a validated email address.
 *
 * @since 1.0.0
 */
final readonly class Email
{
    private const PATTERN = '/^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$/';
    
    private function __construct(
        private string $value
    ) {}
    
    /**
     * Creates an Email from a string.
     *
     * @param string $email The email address
     * @return self
     * @throws \InvalidArgumentException If email format is invalid
     */
    public static function fromString(string $email): self
    {
        if (!preg_match(self::PATTERN, $email)) {
            throw new \InvalidArgumentException(
                sprintf('Invalid email format: %s', $email)
            );
        }
        
        return new self($email);
    }
    
    /**
     * Gets the string value of the email.
     *
     * @return string
     */
    public function value(): string
    {
        return $this->value;
    }
}

// src/Domain/User.php

declare(strict_types=1);

namespace App\Domain;

use App\Domain\ValueObject\Email;
use App\Domain\ValueObject\UserId;

/**
 * User domain entity.
 *
 * Represents an immutable user in the system.
 *
 * @since 1.0.0
 */
final readonly class User
{
    /**
     * Creates a new user.
     *
     * @param UserId $id The user identifier
     * @param string $name The user's name
     * @param Email $email The user's email
     */
    public function __construct(
        private UserId $id,
        private string $name,
        private Email $email
    ) {
        if ($name === '') {
            throw new \InvalidArgumentException('Name cannot be empty');
        }
    }
    
    /**
     * Gets the user ID.
     *
     * @return UserId
     */
    public function id(): UserId
    {
        return $this->id;
    }
    
    /**
     * Gets the user's name.
     *
     * @return string
     */
    public function name(): string
    {
        return $this->name;
    }
    
    /**
     * Gets the user's email.
     *
     * @return Email
     */
    public function email(): Email
    {
        return $this->email;
    }
    
    /**
     * Creates a copy of this user with a different name.
     *
     * @param string $name The new name
     * @return self
     */
    public function withName(string $name): self
    {
        return new self($this->id, $name, $this->email);
    }
}

// Run: ./vendor/bin/phpunit
// ✅ PASSES - tests pass

// Step 3: REFACTOR - Already clean due to readonly properties and value objects
// Tests still pass ✓
```

---

## 1B. Bug Fix Protocol for PHP (MANDATORY)

**CRITICAL: Every PHP bug MUST receive a regression test BEFORE fixing.**

### Bug Fix Workflow for PHP

```
1. 🐛 Bug Reported/Discovered
   ↓
2. ✍️ Write a test that REPRODUCES the bug (test will FAIL)
   ↓
3. ✅ Verify the test fails for the right reason
   ↓
4. 🔧 Fix the bug (make the test pass)
   ↓
5. 🟢 Verify the test now PASSES
   ↓
6. 📝 Document the bug in test comments (include bug ID)
   ↓
7. 🚀 Deploy with confidence (regression prevented)
```

### Example Bug Fix: NullPointerException

```php
<?php
// Bug Report #3421: getUserName() crashes when user is null

// Step 1-2: Write test that reproduces the bug
// tests/Unit/Service/UserServiceTest.php

declare(strict_types=1);

namespace Tests\Unit\Service;

use App\Service\UserService;
use PHPUnit\Framework\TestCase;

final class UserServiceTest extends TestCase
{
    /**
     * Bug #3421: getUserName crashes when user is null.
     * Discovered: 2026-01-18
     * This test prevents regression.
     *
     * @test
     */
    public function getUserName_returnsNull_whenUserIsNull_Bug3421(): void
    {
        $service = new UserService();
        
        // Should return null, not crash
        $result = $service->getUserName(null);
        
        self::assertNull($result);
    }
    
    public function testGetUserNameReturnsNameWhenUserExists(): void
    {
        $service = new UserService();
        $user = new User(
            UserId::fromString('123'),
            'John Doe',
            Email::fromString('john@example.com')
        );
        
        $result = $service->getUserName($user);
        
        self::assertEquals('John Doe', $result);
    }
}

// Run: ./vendor/bin/phpunit
// ❌ FAILS - TypeError: User::name() called on null

// Step 3: Fix the bug
// src/Service/UserService.php

declare(strict_types=1);

namespace App\Service;

use App\Domain\User;

/**
 * Service for user-related operations.
 *
 * @since 1.0.0
 */
final readonly class UserService
{
    /**
     * Gets the user's name.
     *
     * Bug Fix #3421: Now properly handles null users by returning
     * null instead of causing a TypeError.
     *
     * @param User|null $user The user (may be null)
     * @return string|null The user's name, or null if user is null
     */
    public function getUserName(?User $user): ?string
    {
        // FIX: Check for null before accessing user
        return $user?->name();
    }
}

// Run: ./vendor/bin/phpunit
// ✅ PASSES - bug fixed, regression prevented ✓
```

### Example Bug Fix: Async Race Condition

```php
<?php
// Bug Report #3422: Race condition in async user creation

// Step 1-2: Write test that reproduces the bug
// tests/Integration/Repository/UserRepositoryTest.php

declare(strict_types=1);

namespace Tests\Integration\Repository;

use Amp\PHPUnit\AsyncTestCase;
use App\Repository\UserRepository;
use function Amp\async;
use function Amp\Future\await;

final class UserRepositoryTest extends AsyncTestCase
{
    /**
     * Bug #3422: Race condition when creating users concurrently.
     * Discovered: 2026-01-18
     * This test prevents regression.
     *
     * @test
     */
    public function createUser_handlesC oncurrentCallsCorrectly_Bug3422(): void
    {
        $repository = new UserRepository($this->getConnection());
        $email = 'test@example.com';
        
        // Trigger multiple concurrent creation attempts
        $futures = [
            async(fn() => $repository->createUser($email, 'User 1')),
            async(fn() => $repository->createUser($email, 'User 2')),
            async(fn() => $repository->createUser($email, 'User 3')),
        ];
        
        $results = await($futures);
        
        // Only one should succeed, others should throw duplicate key error
        $successful = array_filter($results, fn($r) => $r !== null);
        
        self::assertCount(1, $successful, 'Only one concurrent create should succeed');
    }
}

// Run: ./vendor/bin/phpunit
// ❌ FAILS - Multiple users created with same email

// Step 3: Fix the bug
// src/Repository/UserRepository.php

declare(strict_types=1);

namespace App\Repository;

use Amp\Mysql\MysqlTransaction;
use Amp\Sync\LocalMutex;

/**
 * Repository for user persistence.
 *
 * @since 1.0.0
 */
final class UserRepository
{
    private LocalMutex $mutex;
    
    public function __construct(
        private readonly MysqlConnection $connection
    ) {
        $this->mutex = new LocalMutex();
    }
    
    /**
     * Creates a new user.
     *
     * Bug Fix #3422: Now properly handles concurrent creation attempts
     * by using a mutex lock to serialize access.
     *
     * @param string $email The user email (must be unique)
     * @param string $name The user name
     * @return User The created user
     * @throws \RuntimeException If user with email already exists
     */
    public function createUser(string $email, string $name): User
    {
        // FIX: Acquire lock before checking/creating
        $lock = $this->mutex->acquire();
        
        try {
            // Check if user exists
            $existing = $this->findByEmail($email);
            if ($existing !== null) {
                throw new \RuntimeException("User with email $email already exists");
            }
            
            // Create user
            $userId = $this->generateId();
            $this->connection->execute(
                'INSERT INTO users (id, email, name) VALUES (?, ?, ?)',
                [$userId, $email, $name]
            );
            
            return new User(
                UserId::fromString($userId),
                $name,
                Email::fromString($email)
            );
        } finally {
            $lock->release();
        }
    }
}

// Run: ./vendor/bin/phpunit
// ✅ PASSES - bug fixed, race condition resolved, regression prevented ✓
```

### Prohibited Practices for PHP Bug Fixes

**NEVER:**
- ❌ Fix a bug without adding a regression test first
- ❌ Write implementation before writing tests (violates TDD)
- ❌ Skip the Red-Green-Refactor cycle
- ❌ Commit code with failing tests
- ❌ Remove tests to make code pass
- ❌ Use `@group ignore` or similar to skip failing tests
- ❌ Suppress errors with `@` operator instead of fixing

**ALWAYS:**
- ✅ Write a test that reproduces the bug first
- ✅ Verify the test fails before fixing
- ✅ Document bug ID in test PHPDoc
- ✅ Run `./vendor/bin/phpunit` after fix
- ✅ Run static analysis after fix
- ✅ Ensure fix doesn't introduce new issues
- ✅ Keep tests in codebase permanently

---

## 2. Async Programming Hierarchy (MANDATORY)

### A. Async Framework Preference Order

**ALWAYS prefer async frameworks in this order:**

1. **AMPHP** (HIGHEST PRIORITY) - Modern, fiber-based, comprehensive
2. **ReactPHP** (FALLBACK) - Mature, stable, good ecosystem
3. **Swoole** (FALLBACK) - High performance, PHP extension required
4. **Traditional PHP** (FALLBACK) - Traditional async patterns
5. **Synchronous PHP** (LAST RESORT) - Only when async not needed

### B. AMPHP Examples (PREFERRED)

```php
<?php
declare(strict_types=1);

namespace App\Http;

use Amp\Http\Server\HttpServer;
use Amp\Http\Server\Request;
use Amp\Http\Server\RequestHandler\ClosureRequestHandler;
use Amp\Http\Server\Response;
use Amp\Http\Status;
use Amp\Socket;
use Psr\Log\LoggerInterface;
use function Amp\trapSignal;

/**
 * HTTP server using AMPHP.
 *
 * Provides non-blocking HTTP request handling with fibers.
 *
 * @since 1.0.0
 */
final class Server
{
    public function __construct(
        private readonly LoggerInterface $logger,
        private readonly RequestHandler $handler
    ) {}
    
    /**
     * Starts the HTTP server.
     *
     * @param string $host The host to bind to
     * @param int $port The port to listen on
     * @return void
     */
    public function start(string $host = '0.0.0.0', int $port = 8080): void
    {
        // Create socket server
        $sockets = [
            Socket\listen(sprintf('%s:%d', $host, $port)),
        ];
        
        // Create request handler
        $requestHandler = new ClosureRequestHandler(
            function (Request $request): Response {
                return $this->handler->handle($request);
            }
        );
        
        // Create HTTP server
        $server = new HttpServer($sockets, $requestHandler, $this->logger);
        
        // Start server
        $server->start();
        
        $this->logger->info(sprintf('Server started on %s:%d', $host, $port));
        
        // Wait for SIGINT or SIGTERM
        $signal = trapSignal([SIGINT, SIGTERM]);
        
        $this->logger->info(sprintf('Received signal %d, stopping...', $signal));
        
        // Stop server gracefully
        $server->stop();
    }
}

/**
 * Example async database query with AMPHP.
 *
 * @since 1.0.0
 */
final readonly class UserRepository
{
    public function __construct(
        private MysqlConnection $connection
    ) {}
    
    /**
     * Finds a user by ID asynchronously.
     *
     * @param string $userId The user ID
     * @return User|null The user if found, null otherwise
     */
    public function findById(string $userId): ?User
    {
        $result = $this->connection->execute(
            'SELECT id, name, email FROM users WHERE id = ?',
            [$userId]
        );
        
        $row = $result->fetchRow();
        
        if ($row === null) {
            return null;
        }
        
        return User::fromArray($row);
    }
    
    /**
     * Finds multiple users concurrently.
     *
     * @param list<string> $userIds The user IDs to find
     * @return list<User> The found users
     */
    public function findMany(array $userIds): array
    {
        // Execute queries concurrently
        $futures = [];
        foreach ($userIds as $userId) {
            $futures[] = async(fn() => $this->findById($userId));
        }
        
        // Wait for all queries to complete
        $users = await($futures);
        
        // Filter out nulls
        return array_filter($users);
    }
}

/**
 * Example parallel async operations.
 *
 * @since 1.0.0
 */
final readonly class DashboardService
{
    public function __construct(
        private UserRepository $userRepository,
        private OrderRepository $orderRepository,
        private StatsRepository $statsRepository
    ) {}
    
    /**
     * Loads dashboard data in parallel.
     *
     * @param string $userId The user ID
     * @return DashboardData The dashboard data
     */
    public function loadDashboard(string $userId): DashboardData
    {
        // Execute all queries in parallel
        [$user, $orders, $stats] = await([
            async(fn() => $this->userRepository->findById($userId)),
            async(fn() => $this->orderRepository->findByUserId($userId)),
            async(fn() => $this->statsRepository->getUserStats($userId)),
        ]);
        
        return new DashboardData($user, $orders, $stats);
    }
}
```

### C. Fibers Usage (PHP 8.1+)

```php
<?php
declare(strict_types=1);

namespace App\Async;

use Fiber;

/**
 * Cooperative task scheduler using Fibers.
 *
 * Demonstrates fiber-based concurrency for I/O-bound operations.
 *
 * @since 1.0.0
 */
final class TaskScheduler
{
    /** @var list<Fiber> */
    private array $fibers = [];
    
    /**
     * Schedules a task for execution.
     *
     * @param callable $task The task to execute
     * @return void
     */
    public function schedule(callable $task): void
    {
        $this->fibers[] = new Fiber($task);
    }
    
    /**
     * Runs all scheduled tasks cooperatively.
     *
     * @return void
     */
    public function run(): void
    {
        while (!empty($this->fibers)) {
            foreach ($this->fibers as $key => $fiber) {
                if (!$fiber->isStarted()) {
                    $fiber->start();
                } elseif ($fiber->isSuspended()) {
                    $fiber->resume();
                }
                
                if ($fiber->isTerminated()) {
                    unset($this->fibers[$key]);
                }
            }
        }
    }
}

/**
 * Example async operation with Fiber suspension.
 *
 * @since 1.0.0
 */
function asyncSleep(float $seconds): void
{
    // Suspend current fiber
    $fiber = Fiber::getCurrent();
    if ($fiber === null) {
        // Not in a fiber, use regular sleep
        usleep((int) ($seconds * 1_000_000));
        return;
    }
    
    // Schedule resume
    $wakeTime = microtime(true) + $seconds;
    Fiber::suspend($wakeTime);
}

/**
 * Example usage of Fiber-based async.
 *
 * @return void
 */
function example(): void
{
    $scheduler = new TaskScheduler();
    
    // Schedule multiple tasks
    $scheduler->schedule(function (): void {
        echo "Task 1 start\n";
        asyncSleep(0.1);
        echo "Task 1 end\n";
    });
    
    $scheduler->schedule(function (): void {
        echo "Task 2 start\n";
        asyncSleep(0.05);
        echo "Task 2 end\n";
    });
    
    // Run all tasks cooperatively
    $scheduler->run();
}
```

---

## 2A. TDD Protocol (MANDATORY)

**CRITICAL: Follow the Red-Green-Refactor cycle for ALL new PHP code.**

### TDD Cycle Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        RED-GREEN-REFACTOR CYCLE                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│    ┌──────────┐         ┌──────────┐         ┌──────────┐              │
│    │   RED    │         │  GREEN   │         │ REFACTOR │              │
│    │  Write   │ ──────► │  Write   │ ──────► │ Improve  │              │
│    │ Failing  │         │ Minimal  │         │   Code   │              │
│    │   Test   │         │   Code   │         │          │              │
│    └──────────┘         └──────────┘         └──────────┘              │
│         │                                          │                    │
│         │                                          │                    │
│         └──────────────────────────────────────────┘                    │
│                          REPEAT                                         │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│  Step 1: RED     │ Write a test that fails (class/method doesn't exist) │
│  Step 2: GREEN   │ Write the minimum code to make the test pass         │
│  Step 3: REFACTOR│ Improve code quality while keeping tests green       │
└─────────────────────────────────────────────────────────────────────────┘
```

### Example TDD Workflow for PHP using PHPUnit

**Scenario: Building a Password Strength Validator**

#### Step 1: RED - Write Failing Test First

```php
<?php
// tests/Unit/Security/PasswordStrengthValidatorTest.php

declare(strict_types=1);

namespace Tests\Unit\Security;

use App\Security\PasswordStrengthValidator;
use App\Security\PasswordStrength;
use PHPUnit\Framework\TestCase;

/**
 * @covers \App\Security\PasswordStrengthValidator
 */
final class PasswordStrengthValidatorTest extends TestCase
{
    private PasswordStrengthValidator $validator;

    protected function setUp(): void
    {
        $this->validator = new PasswordStrengthValidator();
    }

    public function testWeakPasswordReturnsWeakStrength(): void
    {
        $result = $this->validator->evaluate('abc');

        self::assertEquals(PasswordStrength::Weak, $result);
    }

    public function testMediumPasswordReturnsMediumStrength(): void
    {
        $result = $this->validator->evaluate('Password1');

        self::assertEquals(PasswordStrength::Medium, $result);
    }

    public function testStrongPasswordReturnsStrongStrength(): void
    {
        $result = $this->validator->evaluate('MyP@ssw0rd!123');

        self::assertEquals(PasswordStrength::Strong, $result);
    }

    /**
     * @dataProvider weakPasswordsProvider
     */
    public function testIdentifiesWeakPasswords(string $password): void
    {
        $result = $this->validator->evaluate($password);

        self::assertEquals(PasswordStrength::Weak, $result);
    }

    /**
     * @return array<string, array{string}>
     */
    public static function weakPasswordsProvider(): array
    {
        return [
            'too short' => ['ab'],
            'only lowercase' => ['abcdefgh'],
            'only numbers' => ['12345678'],
            'common password' => ['password'],
        ];
    }
}
```

```bash
# Run the test
./vendor/bin/phpunit tests/Unit/Security/PasswordStrengthValidatorTest.php

# Output:
# Error: Class "App\Security\PasswordStrengthValidator" not found
# ❌ RED - Test fails because class doesn't exist
```

#### Step 2: GREEN - Write Minimal Implementation

```php
<?php
// src/Security/PasswordStrength.php

declare(strict_types=1);

namespace App\Security;

/**
 * Enum representing password strength levels.
 */
enum PasswordStrength: string
{
    case Weak = 'weak';
    case Medium = 'medium';
    case Strong = 'strong';
}
```

```php
<?php
// src/Security/PasswordStrengthValidator.php

declare(strict_types=1);

namespace App\Security;

/**
 * Validates password strength.
 *
 * @since 1.0.0
 */
final readonly class PasswordStrengthValidator
{
    private const MIN_LENGTH = 8;
    private const COMMON_PASSWORDS = ['password', '12345678', 'qwerty'];

    /**
     * Evaluates the strength of a password.
     *
     * @param string $password The password to evaluate
     * @return PasswordStrength The evaluated strength level
     */
    public function evaluate(string $password): PasswordStrength
    {
        $length = strlen($password);

        // Check for weak passwords
        if ($length < self::MIN_LENGTH || in_array(strtolower($password), self::COMMON_PASSWORDS, true)) {
            return PasswordStrength::Weak;
        }

        $hasLower = preg_match('/[a-z]/', $password) === 1;
        $hasUpper = preg_match('/[A-Z]/', $password) === 1;
        $hasNumber = preg_match('/[0-9]/', $password) === 1;
        $hasSpecial = preg_match('/[^a-zA-Z0-9]/', $password) === 1;

        // Check for only one character type
        $types = array_filter([$hasLower, $hasUpper, $hasNumber, $hasSpecial]);
        if (count($types) <= 1) {
            return PasswordStrength::Weak;
        }

        // Strong: has all four character types and sufficient length
        if ($hasLower && $hasUpper && $hasNumber && $hasSpecial && $length >= 12) {
            return PasswordStrength::Strong;
        }

        return PasswordStrength::Medium;
    }
}
```

```bash
# Run the test
./vendor/bin/phpunit tests/Unit/Security/PasswordStrengthValidatorTest.php

# Output:
# OK (7 tests, 7 assertions)
# ✅ GREEN - All tests pass
```

#### Step 3: REFACTOR - Improve Code Quality

```php
<?php
// src/Security/PasswordStrengthValidator.php (Refactored)

declare(strict_types=1);

namespace App\Security;

/**
 * Evaluates password strength based on multiple criteria.
 *
 * Criteria evaluated:
 * - Minimum length (8 characters)
 * - Character diversity (lowercase, uppercase, numbers, special)
 * - Common password blacklist
 *
 * @since 1.0.0
 */
final readonly class PasswordStrengthValidator
{
    private const MIN_LENGTH_WEAK = 8;
    private const MIN_LENGTH_STRONG = 12;

    /** @var list<string> */
    private const COMMON_PASSWORDS = [
        'password', '12345678', 'qwerty', 'letmein', 'welcome',
        'admin', 'login', 'passw0rd', 'abc123', 'iloveyou',
    ];

    /**
     * Evaluates the strength of a password.
     *
     * @param string $password The password to evaluate
     * @return PasswordStrength The evaluated strength level
     *
     * @example
     * ```php
     * $validator = new PasswordStrengthValidator();
     * $strength = $validator->evaluate('MyP@ssw0rd!');
     * if ($strength === PasswordStrength::Strong) {
     *     echo "Password is strong!";
     * }
     * ```
     */
    public function evaluate(string $password): PasswordStrength
    {
        if ($this->isWeakPassword($password)) {
            return PasswordStrength::Weak;
        }

        if ($this->isStrongPassword($password)) {
            return PasswordStrength::Strong;
        }

        return PasswordStrength::Medium;
    }

    /**
     * Checks if password meets weak criteria.
     */
    private function isWeakPassword(string $password): bool
    {
        return strlen($password) < self::MIN_LENGTH_WEAK
            || $this->isCommonPassword($password)
            || $this->getCharacterTypeCount($password) <= 1;
    }

    /**
     * Checks if password meets strong criteria.
     */
    private function isStrongPassword(string $password): bool
    {
        return strlen($password) >= self::MIN_LENGTH_STRONG
            && $this->getCharacterTypeCount($password) >= 4;
    }

    /**
     * Checks if password is in the common passwords list.
     */
    private function isCommonPassword(string $password): bool
    {
        return in_array(strtolower($password), self::COMMON_PASSWORDS, true);
    }

    /**
     * Counts the number of character types present in the password.
     */
    private function getCharacterTypeCount(string $password): int
    {
        return array_sum([
            (int) (preg_match('/[a-z]/', $password) === 1),
            (int) (preg_match('/[A-Z]/', $password) === 1),
            (int) (preg_match('/[0-9]/', $password) === 1),
            (int) (preg_match('/[^a-zA-Z0-9]/', $password) === 1),
        ]);
    }
}
```

```bash
# Run the test
./vendor/bin/phpunit tests/Unit/Security/PasswordStrengthValidatorTest.php

# Output:
# OK (7 tests, 7 assertions)
# ✅ REFACTOR - Tests still pass, code is cleaner
```

### Visual Step-by-Step TDD Example

```
┌───────────────────────────────────────────────────────────────────────────┐
│                    TDD WORKFLOW: PasswordStrengthValidator                │
├───────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  STEP 1: RED                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │ ❌ Write test: PasswordStrengthValidatorTest.php                    │  │
│  │    - testWeakPasswordReturnsWeakStrength()                          │  │
│  │    - testMediumPasswordReturnsMediumStrength()                      │  │
│  │    - testStrongPasswordReturnsStrongStrength()                      │  │
│  │                                                                     │  │
│  │ $ ./vendor/bin/phpunit                                              │  │
│  │ Result: Error - Class not found                                     │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│                              │                                            │
│                              ▼                                            │
│  STEP 2: GREEN                                                            │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │ ✅ Create: PasswordStrength.php (enum)                              │  │
│  │ ✅ Create: PasswordStrengthValidator.php (minimal implementation)   │  │
│  │                                                                     │  │
│  │ $ ./vendor/bin/phpunit                                              │  │
│  │ Result: OK (7 tests, 7 assertions)                                  │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│                              │                                            │
│                              ▼                                            │
│  STEP 3: REFACTOR                                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │ 🔄 Refactor: Extract private methods                                │  │
│  │    - isWeakPassword()                                               │  │
│  │    - isStrongPassword()                                             │  │
│  │    - isCommonPassword()                                             │  │
│  │    - getCharacterTypeCount()                                        │  │
│  │ 🔄 Add: PHPDoc documentation and examples                           │  │
│  │                                                                     │  │
│  │ $ ./vendor/bin/phpunit                                              │  │
│  │ Result: OK (7 tests, 7 assertions) ✓                                │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│                              │                                            │
│                              ▼                                            │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │                    REPEAT FOR NEXT FEATURE                          │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│                                                                           │
└───────────────────────────────────────────────────────────────────────────┘
```

### TDD Best Practices for PHP

| Practice | Description |
|----------|-------------|
| **One assertion per test** | Each test should verify one specific behavior |
| **Descriptive test names** | Use `testMethodName_condition_expectedResult` naming |
| **Data providers** | Use `@dataProvider` for testing multiple inputs |
| **Arrange-Act-Assert** | Structure tests with clear setup, action, and verification |
| **Test edge cases** | Include boundary conditions and error scenarios |
| **Keep tests fast** | Unit tests should run in milliseconds |

---

## 2B. Bug Fix Protocol (MANDATORY)

**CRITICAL: Every PHP bug MUST receive a regression test BEFORE fixing.**

### Bug Fix Workflow Diagram

```
┌───────────────────────────────────────────────────────────────────────────┐
│                         BUG FIX WORKFLOW                                  │
├───────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  ┌─────────────┐    ┌──────────────┐    ┌──────────────┐                 │
│  │     BUG     │    │   WRITE      │    │   VERIFY     │                 │
│  │  REPORTED   │───►│ REGRESSION   │───►│  TEST FAILS  │                 │
│  │             │    │    TEST      │    │  (RIGHT WAY) │                 │
│  └─────────────┘    └──────────────┘    └──────────────┘                 │
│                                                │                          │
│                                                ▼                          │
│  ┌─────────────┐    ┌──────────────┐    ┌──────────────┐                 │
│  │  DOCUMENT   │◄───│   VERIFY     │◄───│   FIX THE    │                 │
│  │  & DEPLOY   │    │ TEST PASSES  │    │     BUG      │                 │
│  │             │    │              │    │              │                 │
│  └─────────────┘    └──────────────┘    └──────────────┘                 │
│                                                                           │
├───────────────────────────────────────────────────────────────────────────┤
│                          WORKFLOW DETAILS                                 │
├─────────────────┬─────────────────────────────────────────────────────────┤
│ 1. Bug Reported │ Document: ID, description, reproduction steps          │
│ 2. Write Test   │ Create PHPUnit test that reproduces the bug            │
│ 3. Verify Fail  │ Run test, confirm it fails for the expected reason     │
│ 4. Fix Bug      │ Implement the fix in the source code                   │
│ 5. Verify Pass  │ Run test, confirm it now passes                        │
│ 6. Document     │ Add bug ID in PHPDoc, explain the fix                  │
└─────────────────┴─────────────────────────────────────────────────────────┘
```

### Example Bug Fix with Regression Test using PHPUnit

**Bug Report #4521: ArrayHelper::flatten() causes infinite loop on circular references**

#### Step 1: Document the Bug

```
Bug ID: #4521
Reported: 2026-01-20
Severity: Critical
Description: ArrayHelper::flatten() enters infinite loop when array contains
             circular references, causing memory exhaustion.
Reproduction: Create array with self-reference, call flatten()
```

#### Step 2: Write Regression Test (Test Will FAIL)

```php
<?php
// tests/Unit/Helper/ArrayHelperFlattenBugTest.php

declare(strict_types=1);

namespace Tests\Unit\Helper;

use App\Helper\ArrayHelper;
use App\Exception\CircularReferenceException;
use PHPUnit\Framework\TestCase;

/**
 * Regression tests for ArrayHelper::flatten() bug fixes.
 *
 * @covers \App\Helper\ArrayHelper
 */
final class ArrayHelperFlattenBugTest extends TestCase
{
    /**
     * Bug #4521: flatten() causes infinite loop on circular references.
     * Discovered: 2026-01-20
     *
     * The flatten method should detect circular references and throw
     * an exception instead of entering an infinite loop.
     *
     * @test
     */
    public function flatten_throwsException_whenCircularReferenceDetected_Bug4521(): void
    {
        // Arrange: Create array with circular reference
        $array = ['a' => 1, 'b' => []];
        $array['b']['circular'] = &$array;

        // Assert: Expect exception
        $this->expectException(CircularReferenceException::class);
        $this->expectExceptionMessage('Circular reference detected');

        // Act: Attempt to flatten
        ArrayHelper::flatten($array);
    }

    /**
     * Bug #4521: Ensure normal nested arrays still work after fix.
     *
     * @test
     */
    public function flatten_worksCorrectly_withDeeplyNestedArrays_Bug4521(): void
    {
        $array = [
            'level1' => [
                'level2' => [
                    'level3' => [
                        'value' => 'deep',
                    ],
                ],
            ],
        ];

        $result = ArrayHelper::flatten($array);

        self::assertEquals(['level1.level2.level3.value' => 'deep'], $result);
    }

    /**
     * Bug #4521: Ensure performance with maximum recursion depth.
     *
     * @test
     */
    public function flatten_respectsMaxDepth_toPreventStackOverflow_Bug4521(): void
    {
        // Create deeply nested array (100 levels)
        $array = ['value' => 'leaf'];
        for ($i = 0; $i < 100; $i++) {
            $array = ['nested' => $array];
        }

        $this->expectException(CircularReferenceException::class);
        $this->expectExceptionMessage('Maximum recursion depth exceeded');

        ArrayHelper::flatten($array, maxDepth: 50);
    }
}
```

```bash
# Run the test
./vendor/bin/phpunit tests/Unit/Helper/ArrayHelperFlattenBugTest.php

# Output:
# 1) flatten_throwsException_whenCircularReferenceDetected_Bug4521
#    Failed asserting that exception of type "CircularReferenceException" is thrown.
#    (Test times out / memory exhausted)
# ❌ RED - Test fails because bug exists
```

#### Step 3: Verify Test Fails for the Right Reason

```bash
# The test either:
# - Times out (infinite loop)
# - Runs out of memory
# - Does not throw the expected exception

# This confirms the bug exists and our test reproduces it
```

#### Step 4: Fix the Bug

```php
<?php
// src/Exception/CircularReferenceException.php

declare(strict_types=1);

namespace App\Exception;

/**
 * Exception thrown when a circular reference is detected.
 */
final class CircularReferenceException extends \RuntimeException
{
    public static function detected(): self
    {
        return new self('Circular reference detected in array structure');
    }

    public static function maxDepthExceeded(int $maxDepth): self
    {
        return new self(
            sprintf('Maximum recursion depth exceeded (max: %d)', $maxDepth)
        );
    }
}
```

```php
<?php
// src/Helper/ArrayHelper.php

declare(strict_types=1);

namespace App\Helper;

use App\Exception\CircularReferenceException;

/**
 * Helper class for array operations.
 *
 * @since 1.0.0
 */
final class ArrayHelper
{
    private const DEFAULT_MAX_DEPTH = 100;

    /**
     * Flattens a nested array into a single-level array with dot notation keys.
     *
     * Bug Fix #4521: Now detects circular references and respects maximum
     * recursion depth to prevent infinite loops and stack overflows.
     *
     * @param array<mixed> $array The array to flatten
     * @param string $prefix The prefix for keys (used internally)
     * @param int $maxDepth Maximum recursion depth (default: 100)
     * @return array<string, mixed> The flattened array
     *
     * @throws CircularReferenceException If circular reference detected or max depth exceeded
     *
     * @example
     * ```php
     * $nested = ['user' => ['name' => 'John', 'email' => 'john@example.com']];
     * $flat = ArrayHelper::flatten($nested);
     * // Result: ['user.name' => 'John', 'user.email' => 'john@example.com']
     * ```
     */
    public static function flatten(
        array $array,
        string $prefix = '',
        int $maxDepth = self::DEFAULT_MAX_DEPTH
    ): array {
        // Track seen arrays to detect circular references
        static $seen = [];

        // Reset tracking on initial call
        if ($prefix === '') {
            $seen = [];
        }

        // Check max depth
        $currentDepth = substr_count($prefix, '.');
        if ($currentDepth > $maxDepth) {
            $seen = []; // Reset before throwing
            throw CircularReferenceException::maxDepthExceeded($maxDepth);
        }

        // Get unique ID for this array instance
        $arrayId = self::getArrayId($array);

        // Check for circular reference
        if (isset($seen[$arrayId])) {
            $seen = []; // Reset before throwing
            throw CircularReferenceException::detected();
        }

        $seen[$arrayId] = true;

        $result = [];

        foreach ($array as $key => $value) {
            $newKey = $prefix === '' ? (string) $key : $prefix . '.' . $key;

            if (is_array($value) && !empty($value)) {
                $result = array_merge($result, self::flatten($value, $newKey, $maxDepth));
            } else {
                $result[$newKey] = $value;
            }
        }

        // Clean up tracking for this array
        unset($seen[$arrayId]);

        return $result;
    }

    /**
     * Gets a unique identifier for an array instance.
     *
     * @param array<mixed> $array The array
     * @return string The unique identifier
     */
    private static function getArrayId(array &$array): string
    {
        // Use a marker to detect the same array instance
        $marker = '__flatten_marker_' . spl_object_id(new \stdClass());

        // Check if we've seen this exact array reference
        if (isset($array[$marker])) {
            return $array[$marker];
        }

        // Mark this array
        $id = uniqid('array_', true);
        $array[$marker] = $id;

        // Clean up marker after getting ID
        $result = $id;
        unset($array[$marker]);

        return $result;
    }
}
```

#### Step 5: Verify Test Passes

```bash
# Run the test
./vendor/bin/phpunit tests/Unit/Helper/ArrayHelperFlattenBugTest.php

# Output:
# OK (3 tests, 3 assertions)
# ✅ GREEN - Bug fixed, regression test passes
```

#### Step 6: Document and Verify All Tests

```bash
# Run full test suite to ensure no regressions
./vendor/bin/phpunit

# Run static analysis
./vendor/bin/phpstan analyse src tests --level max

# Verify all checks pass before committing
# ✅ All tests pass
# ✅ Static analysis passes
# ✅ Bug #4521 fixed with regression test
```

### Bug Fix Checklist

```
┌───────────────────────────────────────────────────────────────────────────┐
│                      BUG FIX VERIFICATION CHECKLIST                       │
├───────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  Before Fixing:                                                           │
│  [ ] Bug documented with ID, description, and reproduction steps          │
│  [ ] Regression test written that reproduces the bug                      │
│  [ ] Test verified to FAIL before fix                                     │
│                                                                           │
│  After Fixing:                                                            │
│  [ ] Regression test now PASSES                                           │
│  [ ] All existing tests still pass                                        │
│  [ ] Bug ID documented in PHPDoc comment                                  │
│  [ ] Fix explanation added to code comments                               │
│  [ ] Static analysis passes (Psalm/PHPStan)                               │
│  [ ] Code style check passes (PHP-CS-Fixer)                               │
│                                                                           │
│  Prohibited Actions:                                                      │
│  [ ] DO NOT fix bugs without regression tests                             │
│  [ ] DO NOT delete tests to make code pass                                │
│  [ ] DO NOT use @group ignore to skip failing tests                       │
│  [ ] DO NOT suppress errors with @ operator                               │
│                                                                           │
└───────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Hexagonal Architecture (MANDATORY)

### A. Directory Structure

```
project-root/
├── composer.json
├── composer.lock
├── phpunit.xml
├── psalm.xml
├── .php-cs-fixer.php
│
├── src/
│   ├── Domain/                    # Core domain (no dependencies)
│   │   ├── Entity/
│   │   │   └── User.php
│   │   ├── ValueObject/
│   │   │   ├── UserId.php
│   │   │   └── Email.php
│   │   ├── Repository/            # Repository interfaces (ports)
│   │   │   └── UserRepositoryInterface.php
│   │   └── Service/
│   │       └── UserDomainService.php
│   │
│   ├── Application/               # Use cases
│   │   ├── Command/
│   │   │   ├── CreateUserCommand.php
│   │   │   └── CreateUserHandler.php
│   │   ├── Query/
│   │   │   ├── GetUserQuery.php
│   │   │   └── GetUserHandler.php
│   │   └── Service/
│   │       └── UserApplicationService.php
│   │
│   ├── Infrastructure/            # External dependencies
│   │   ├── Persistence/
│   │   │   ├── Mysql/
│   │   │   │   └── MysqlUserRepository.php
│   │   │   └── Redis/
│   │   │       └── RedisCacheRepository.php
│   │   ├── Messaging/
│   │   │   └── AmqpEventBus.php
│   │   └── Http/
│   │       └── GuzzleHttpClient.php
│   │
│   └── Adapter/                   # Adapters (primary/driving)
│       ├── Http/
│       │   ├── Controller/
│       │   │   └── UserController.php
│       │   └── Middleware/
│       │       └── AuthenticationMiddleware.php
│       ├── Console/
│       │   └── Command/
│       │       └── CreateUserCommand.php
│       └── Api/
│           └── GraphQL/
│               └── UserResolver.php
│
└── tests/
    ├── Unit/
    │   ├── Domain/
    │   ├── Application/
    │   └── Infrastructure/
    ├── Integration/
    │   └── Infrastructure/
    └── E2E/
        └── Http/
```

### B. Domain Layer Example

```php
<?php
declare(strict_types=1);

namespace App\Domain\Entity;

use App\Domain\ValueObject\Email;
use App\Domain\ValueObject\UserId;

/**
 * User domain entity.
 *
 * Represents a user in the domain model with all business rules.
 * This is the core of the hexagonal architecture - no external dependencies.
 *
 * @since 1.0.0
 */
final readonly class User
{
    /**
     * Creates a new user entity.
     *
     * @param UserId $id The unique user identifier
     * @param string $name The user's full name
     * @param Email $email The user's email address
     * @param \DateTimeImmutable $createdAt When the user was created
     */
    public function __construct(
        private UserId $id,
        private string $name,
        private Email $email,
        private \DateTimeImmutable $createdAt
    ) {
        if ($name === '' || strlen($name) < 2) {
            throw new \DomainException('Name must be at least 2 characters');
        }
    }
    
    /**
     * Creates a new user (factory method).
     *
     * @param string $name The user's name
     * @param string $email The user's email
     * @return self
     */
    public static function create(string $name, string $email): self
    {
        return new self(
            UserId::generate(),
            $name,
            Email::fromString($email),
            new \DateTimeImmutable()
        );
    }
    
    public function id(): UserId
    {
        return $this->id;
    }
    
    public function name(): string
    {
        return $this->name;
    }
    
    public function email(): Email
    {
        return $this->email;
    }
    
    public function createdAt(): \DateTimeImmutable
    {
        return $this->createdAt;
    }
    
    /**
     * Changes the user's name.
     *
     * @param string $newName The new name
     * @return self A new user instance with the updated name
     */
    public function changeName(string $newName): self
    {
        return new self(
            $this->id,
            $newName,
            $this->email,
            $this->createdAt
        );
    }
}

/**
 * Repository interface (port) - defines contract.
 *
 * This is a port in hexagonal architecture - the domain defines
 * what it needs, infrastructure provides the implementation.
 *
 * @since 1.0.0
 */
interface UserRepositoryInterface
{
    /**
     * Finds a user by ID.
     *
     * @param UserId $id The user ID
     * @return User|null The user if found, null otherwise
     */
    public function findById(UserId $id): ?User;
    
    /**
     * Saves a user.
     *
     * @param User $user The user to save
     * @return void
     */
    public function save(User $user): void;
    
    /**
     * Finds a user by email.
     *
     * @param Email $email The email to search for
     * @return User|null The user if found, null otherwise
     */
    public function findByEmail(Email $email): ?User;
}
```

### C. Application Layer Example

```php
<?php
declare(strict_types=1);

namespace App\Application\Command;

/**
 * Command to create a new user.
 *
 * Represents the intent to create a user (CQRS pattern).
 *
 * @since 1.0.0
 */
final readonly class CreateUserCommand
{
    public function __construct(
        public string $name,
        public string $email
    ) {}
}

/**
 * Handler for CreateUserCommand.
 *
 * Application layer - orchestrates domain operations.
 *
 * @since 1.0.0
 */
final readonly class CreateUserHandler
{
    public function __construct(
        private UserRepositoryInterface $userRepository,
        private EventBusInterface $eventBus
    ) {}
    
    /**
     * Handles the create user command.
     *
     * @param CreateUserCommand $command The command to handle
     * @return User The created user
     * @throws \DomainException If user with email already exists
     */
    public function handle(CreateUserCommand $command): User
    {
        // Check if user exists
        $existingUser = $this->userRepository->findByEmail(
            Email::fromString($command->email)
        );
        
        if ($existingUser !== null) {
            throw new \DomainException('User with this email already exists');
        }
        
        // Create user (domain logic)
        $user = User::create($command->name, $command->email);
        
        // Save user (infrastructure)
        $this->userRepository->save($user);
        
        // Publish event
        $this->eventBus->publish(
            new UserCreatedEvent($user->id(), $user->email())
        );
        
        return $user;
    }
}
```

### D. Infrastructure Layer Example

```php
<?php
declare(strict_types=1);

namespace App\Infrastructure\Persistence\Mysql;

use Amp\Mysql\MysqlConnection;
use App\Domain\Entity\User;
use App\Domain\Repository\UserRepositoryInterface;
use App\Domain\ValueObject\Email;
use App\Domain\ValueObject\UserId;

/**
 * MySQL implementation of UserRepository.
 *
 * Infrastructure layer - implements the domain port.
 * Uses AMPHP for async database operations.
 *
 * @since 1.0.0
 */
final readonly class MysqlUserRepository implements UserRepositoryInterface
{
    public function __construct(
        private MysqlConnection $connection
    ) {}
    
    public function findById(UserId $id): ?User
    {
        $result = $this->connection->execute(
            'SELECT id, name, email, created_at FROM users WHERE id = ?',
            [$id->value()]
        );
        
        $row = $result->fetchRow();
        
        if ($row === null) {
            return null;
        }
        
        return $this->hydrateUser($row);
    }
    
    public function save(User $user): void
    {
        $this->connection->execute(
            'INSERT INTO users (id, name, email, created_at) 
             VALUES (?, ?, ?, ?)
             ON DUPLICATE KEY UPDATE name = VALUES(name), email = VALUES(email)',
            [
                $user->id()->value(),
                $user->name(),
                $user->email()->value(),
                $user->createdAt()->format('Y-m-d H:i:s'),
            ]
        );
    }
    
    public function findByEmail(Email $email): ?User
    {
        $result = $this->connection->execute(
            'SELECT id, name, email, created_at FROM users WHERE email = ?',
            [$email->value()]
        );
        
        $row = $result->fetchRow();
        
        if ($row === null) {
            return null;
        }
        
        return $this->hydrateUser($row);
    }
    
    /**
     * Hydrates a user from database row.
     *
     * @param array<string, mixed> $row The database row
     * @return User The hydrated user
     */
    private function hydrateUser(array $row): User
    {
        return new User(
            UserId::fromString($row['id']),
            $row['name'],
            Email::fromString($row['email']),
            new \DateTimeImmutable($row['created_at'])
        );
    }
}
```

### E. Adapter Layer Example

```php
<?php
declare(strict_types=1);

namespace App\Adapter\Http\Controller;

use Amp\Http\Server\Request;
use Amp\Http\Server\Response;
use Amp\Http\Status;
use App\Application\Command\CreateUserCommand;
use App\Application\Command\CreateUserHandler;

/**
 * HTTP controller for user operations.
 *
 * Adapter layer - translates HTTP to application commands.
 *
 * @since 1.0.0
 */
final readonly class UserController
{
    public function __construct(
        private CreateUserHandler $createUserHandler
    ) {}
    
    /**
     * Creates a new user from HTTP request.
     *
     * POST /users
     * Body: {"name": "John Doe", "email": "john@example.com"}
     *
     * @param Request $request The HTTP request
     * @return Response The HTTP response
     */
    public function create(Request $request): Response
    {
        try {
            // Parse request body
            $body = json_decode(
                $request->getBody()->buffer(),
                true,
                512,
                JSON_THROW_ON_ERROR
            );
            
            // Validate input
            if (!isset($body['name'], $body['email'])) {
                return new Response(
                    Status::BAD_REQUEST,
                    ['content-type' => 'application/json'],
                    json_encode(['error' => 'Missing required fields'])
                );
            }
            
            // Create command
            $command = new CreateUserCommand(
                $body['name'],
                $body['email']
            );
            
            // Execute command
            $user = $this->createUserHandler->handle($command);
            
            // Return response
            return new Response(
                Status::CREATED,
                ['content-type' => 'application/json'],
                json_encode([
                    'id' => $user->id()->value(),
                    'name' => $user->name(),
                    'email' => $user->email()->value(),
                ])
            );
            
        } catch (\DomainException $e) {
            return new Response(
                Status::CONFLICT,
                ['content-type' => 'application/json'],
                json_encode(['error' => $e->getMessage()])
            );
        } catch (\Throwable $e) {
            return new Response(
                Status::INTERNAL_SERVER_ERROR,
                ['content-type' => 'application/json'],
                json_encode(['error' => 'Internal server error'])
            );
        }
    }
}
```

---

## 4. Documentation as Code (MANDATORY)

### A. PHPDoc Standards

**ALL public classes, methods, properties, and constants MUST have complete PHPDoc comments.**

```php
<?php
declare(strict_types=1);

namespace App\Service;

use App\Domain\Entity\User;
use App\Domain\Repository\UserRepositoryInterface;
use App\Domain\ValueObject\Email;
use Psr\Log\LoggerInterface;

/**
 * Service for user authentication operations.
 *
 * Provides methods to authenticate users, manage sessions, and handle
 * authentication-related business logic. All operations are logged for
 * security audit purposes.
 *
 * @since 1.0.0
 * @see User
 * @see UserRepositoryInterface
 *
 * @example
 * ```php
 * $service = new AuthService($userRepository, $logger);
 * $user = $service->authenticate('user@example.com', 'password');
 * if ($user !== null) {
 *     echo "Authentication successful";
 * }
 * ```
 */
final readonly class AuthService
{
    /**
     * Maximum number of failed login attempts before account lockout.
     */
    private const MAX_FAILED_ATTEMPTS = 5;
    
    /**
     * Account lockout duration in seconds.
     */
    private const LOCKOUT_DURATION = 900; // 15 minutes
    
    /**
     * Creates a new authentication service.
     *
     * @param UserRepositoryInterface $userRepository Repository for user data access
     * @param LoggerInterface $logger Logger for security audit trail
     */
    public function __construct(
        private UserRepositoryInterface $userRepository,
        private LoggerInterface $logger
    ) {}
    
    /**
     * Authenticates a user with email and password.
     *
     * Verifies the provided credentials against stored user data.
     * Implements rate limiting to prevent brute force attacks.
     * Logs all authentication attempts for security auditing.
     *
     * @param string $email The user's email address
     * @param string $password The user's password (plain text)
     * @return User|null The authenticated user, or null if authentication failed
     *
     * @throws \DomainException If account is locked due to too many failed attempts
     * @throws \InvalidArgumentException If email format is invalid
     *
     * @example
     * ```php
     * try {
     *     $user = $authService->authenticate('user@example.com', 'secret123');
     *     if ($user !== null) {
     *         // Authentication successful
     *         $_SESSION['user_id'] = $user->id()->value();
     *     } else {
     *         // Invalid credentials
     *         echo "Invalid email or password";
     *     }
     * } catch (\DomainException $e) {
     *     // Account locked
     *     echo "Account temporarily locked: " . $e->getMessage();
     * }
     * ```
     *
     * @psalm-return User|null
     */
    public function authenticate(string $email, string $password): ?User
    {
        $this->logger->info('Authentication attempt', ['email' => $email]);
        
        // Validate email format
        $emailVO = Email::fromString($email);
        
        // Find user
        $user = $this->userRepository->findByEmail($emailVO);
        
        if ($user === null) {
            $this->logger->warning('Authentication failed: user not found', [
                'email' => $email,
            ]);
            return null;
        }
        
        // Check if account is locked
        if ($this->isAccountLocked($user)) {
            $this->logger->warning('Authentication failed: account locked', [
                'user_id' => $user->id()->value(),
            ]);
            throw new \DomainException('Account temporarily locked. Try again later.');
        }
        
        // Verify password
        if (!password_verify($password, $user->passwordHash())) {
            $this->handleFailedAttempt($user);
            $this->logger->warning('Authentication failed: invalid password', [
                'user_id' => $user->id()->value(),
            ]);
            return null;
        }
        
        // Authentication successful
        $this->resetFailedAttempts($user);
        $this->logger->info('Authentication successful', [
            'user_id' => $user->id()->value(),
        ]);
        
        return $user;
    }
    
    /**
     * Checks if an account is currently locked.
     *
     * @param User $user The user to check
     * @return bool True if account is locked, false otherwise
     */
    private function isAccountLocked(User $user): bool
    {
        // Implementation details..
        return false;
    }
    
    /**
     * Handles a failed authentication attempt.
     *
     * Increments the failed attempt counter and locks the account
     * if the maximum number of attempts is reached.
     *
     * @param User $user The user who failed authentication
     * @return void
     */
    private function handleFailedAttempt(User $user): void
    {
        // Implementation details..
    }
    
    /**
     * Resets the failed authentication attempt counter.
     *
     * Called after successful authentication to clear any
     * previous failed attempts.
     *
     * @param User $user The user to reset
     * @return void
     */
    private function resetFailedAttempts(User $user): void
    {
        // Implementation details..
    }
}
```

### B. PHPDoc Generation

**Configure PHPDocumentor:**

```xml
<!-- phpdoc.xml -->
<?xml version="1.0" encoding="UTF-8" ?>
<phpdocumentor
    configVersion="3"
    xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
    xmlns="https://www.phpdoc.org"
    xsi:schemaLocation="https://www.phpdoc.org/schemas/phpdoc.xsd"
>
    <title>My PHP Application</title>
    <paths>
        <output>docs/api</output>
        <cache>.phpdoc/cache</cache>
    </paths>
    <version number="1.0.0">
        <folder>latest</folder>
        <api>
            <source dsn=".">
                <path>src</path>
            </source>
            <output>docs/api</output>
            <ignore hidden="true" symlinks="true">
                <path>vendor/**</path>
            </ignore>
            <extensions>
                <extension>php</extension>
            </extensions>
            <visibility>public</visibility>
            <visibility>protected</visibility>
            <default-package-name>App</default-package-name>
            <markers>
                <item>TODO</item>
                <item>FIXME</item>
            </markers>
        </api>
    </version>
    <template name="default"/>
</phpdocumentor>
```

```bash
# Generate documentation
./vendor/bin/phpdoc run
```

---

## 5. Testing Requirements (MANDATORY)

### A. PHPUnit Configuration

```xml
<!-- phpunit.xml -->
<?xml version="1.0" encoding="UTF-8"?>
<phpunit
    xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
    xsi:noNamespaceSchemaLocation="vendor/phpunit/phpunit/phpunit.xsd"
    bootstrap="vendor/autoload.php"
    colors="true"
    failOnRisky="true"
    failOnWarning="true"
    stopOnFailure="false"
    executionOrder="random"
    beStrictAboutOutputDuringTests="true"
    beStrictAboutTodoAnnotatedTests="true"
    cacheDirectory=".phpunit.cache"
>
    <testsuites>
        <testsuite name="Unit">
            <directory>tests/Unit</directory>
        </testsuite>
        <testsuite name="Integration">
            <directory>tests/Integration</directory>
        </testsuite>
        <testsuite name="E2E">
            <directory>tests/E2E</directory>
        </testsuite>
    </testsuites>

    <source>
        <include>
            <directory suffix=".php">src</directory>
        </include>
        <exclude>
            <directory>src/Infrastructure/Migration</directory>
        </exclude>
    </source>

    <coverage
        includeUncoveredFiles="true"
        processUncoveredFiles="true"
        pathCoverage="false"
        ignoreDeprecatedCodeUnits="true"
        disableCodeCoverageIgnore="false"
    >
        <report>
            <html outputDirectory="coverage/html"/>
            <text outputFile="php://stdout" showOnlySummary="true"/>
        </report>
    </coverage>

    <php>
        <env name="APP_ENV" value="test"/>
        <ini name="error_reporting" value="-1"/>
        <ini name="display_errors" value="1"/>
        <ini name="display_startup_errors" value="1"/>
        <ini name="memory_limit" value="-1"/>
    </php>
</phpunit>
```

### B. Test Organization

```php
<?php
declare(strict_types=1);

namespace Tests\Unit\Domain\Entity;

use App\Domain\Entity\User;
use App\Domain\ValueObject\Email;
use App\Domain\ValueObject\UserId;
use PHPUnit\Framework\TestCase;

/**
 * Test suite for User entity.
 *
 * @covers \App\Domain\Entity\User
 */
final class UserTest extends TestCase
{
    public function testCreateUserWithValidData(): void
    {
        $user = User::create('John Doe', 'john@example.com');
        
        self::assertInstanceOf(User::class, $user);
        self::assertEquals('John Doe', $user->name());
        self::assertEquals('john@example.com', $user->email()->value());
        self::assertInstanceOf(\DateTimeImmutable::class, $user->createdAt());
    }
    
    /**
     * @dataProvider invalidNamesProvider
     */
    public function testThrowsOnInvalidName(string $invalidName): void
    {
        $this->expectException(\DomainException::class);
        $this->expectExceptionMessage('Name must be at least 2 characters');
        
        User::create($invalidName, 'john@example.com');
    }
    
    /**
     * @return array<string, array{string}>
     */
    public static function invalidNamesProvider(): array
    {
        return [
            'empty string' => [''],
            'single character' => ['J'],
            'whitespace only' => ['  '],
        ];
    }
    
    public function testUserIsImmutable(): void
    {
        $original = User::create('John Doe', 'john@example.com');
        $modified = $original->changeName('Jane Doe');
        
        self::assertEquals('John Doe', $original->name());
        self::assertEquals('Jane Doe', $modified->name());
        self::assertNotSame($original, $modified);
    }
}
```

---

## 6. Deployment Checklist

### Pre-Commit Checklist
- [ ] **Syntax check passes**: `php -l` succeeds for all files
- [ ] **Tests written first**: TDD cycle followed
- [ ] **All tests pass**: `./vendor/bin/phpunit` succeeds
- [ ] **Coverage maintained**: >80% code coverage
- [ ] **Static analysis passes**: Psalm/PHPStan level max
- [ ] **Code style compliant**: PHP-CS-Fixer passes
- [ ] **PHPDoc complete**: All public APIs documented
- [ ] **No debug code**: No `var_dump()`, `dd()`, etc.
- [ ] **Async preference**: AMPHP used where applicable
- [ ] **For bug fixes**: Regression test included

### Pre-Deploy Checklist
- [ ] **Composer dependencies**: `composer validate` passes
- [ ] **Autoload optimized**: `composer dump-autoload --optimize`
- [ ] **OPcache enabled**: Production PHP configuration
- [ ] **Error handling**: Proper exception handling
- [ ] **Logging configured**: PSR-3 logger integrated
- [ ] **Environment variables**: `.env` properly configured
- [ ] **Database migrations**: All migrations applied
- [ ] **Health check**: Application responds correctly

---

## 7. Why This Configuration Works

1. **TDD First**: Tests before code ensures quality and prevents regressions (40-80% fewer bugs).
2. **Async with AMPHP**: Non-blocking I/O scales to thousands of concurrent connections.
3. **Fibers**: Native cooperative multitasking without callbacks, clean async code.
4. **Hexagonal Architecture**: Clear boundaries, testable code, changeable infrastructure.
5. **Readonly Properties**: Immutability prevents bugs, easier to reason about code.
6. **Strict Types**: Catches type errors at compile time, not runtime.
7. **PHPDoc**: Generated documentation stays in sync with code.
8. **Static Analysis**: Psalm/PHPStan catch bugs before tests run.
9. **Value Objects**: Domain concepts explicit, validation centralized.
10. **Regression Tests**: Every bug gets a test, building safety net over time.

---

## Quick Reference

### Common Commands

```bash
# ─────────────────────────────────────────────────────────────────────────────
# COMPOSER - Dependency Management
# ─────────────────────────────────────────────────────────────────────────────
composer install              # Install dependencies from composer.lock
composer update               # Update dependencies to latest versions
composer require vendor/pkg   # Add a new dependency
composer require --dev pkg    # Add a dev dependency
composer dump-autoload -o     # Optimize autoloader for production
composer validate             # Validate composer.json
composer check-platform-reqs  # Check PHP version and extensions

# ─────────────────────────────────────────────────────────────────────────────
# PHPUNIT - Testing
# ─────────────────────────────────────────────────────────────────────────────
./vendor/bin/phpunit                           # Run all tests
./vendor/bin/phpunit tests/Unit                # Run unit tests only
./vendor/bin/phpunit tests/Integration         # Run integration tests only
./vendor/bin/phpunit --filter=TestClassName    # Run specific test class
./vendor/bin/phpunit --filter=testMethodName   # Run specific test method
./vendor/bin/phpunit --group=regression        # Run tests in a group

# With coverage (requires Xdebug or PCOV)
XDEBUG_MODE=coverage ./vendor/bin/phpunit --coverage-text
XDEBUG_MODE=coverage ./vendor/bin/phpunit --coverage-html=coverage

# ─────────────────────────────────────────────────────────────────────────────
# PHPSTAN - Static Analysis
# ─────────────────────────────────────────────────────────────────────────────
./vendor/bin/phpstan analyse src tests         # Analyze source and tests
./vendor/bin/phpstan analyse --level max       # Maximum strictness
./vendor/bin/phpstan analyse --generate-baseline  # Generate baseline for legacy code
./vendor/bin/phpstan clear-result-cache       # Clear analysis cache

# ─────────────────────────────────────────────────────────────────────────────
# PSALM - Static Analysis (Alternative)
# ─────────────────────────────────────────────────────────────────────────────
./vendor/bin/psalm                             # Run Psalm analysis
./vendor/bin/psalm --no-cache                  # Run without cache
./vendor/bin/psalm --set-baseline=baseline.xml # Create baseline
./vendor/bin/psalm --show-info=true            # Show all issues including info

# ─────────────────────────────────────────────────────────────────────────────
# PHP-CS-FIXER - Code Style
# ─────────────────────────────────────────────────────────────────────────────
./vendor/bin/php-cs-fixer fix                  # Auto-fix code style
./vendor/bin/php-cs-fixer fix --dry-run        # Preview changes only
./vendor/bin/php-cs-fixer fix --diff           # Show diff of changes
./vendor/bin/php-cs-fixer fix src/             # Fix specific directory

# ─────────────────────────────────────────────────────────────────────────────
# PHP - Syntax and Execution
# ─────────────────────────────────────────────────────────────────────────────
php -l src/File.php                            # Syntax check single file
find src/ -name "*.php" -exec php -l {} \;     # Syntax check all files
php -S localhost:8080 -t public/               # Built-in development server
php bin/console                                # Run console command (Symfony)
php artisan                                    # Run console command (Laravel)

# ─────────────────────────────────────────────────────────────────────────────
# AMPHP - Async Development Server
# ─────────────────────────────────────────────────────────────────────────────
php bin/server.php                             # Run AMPHP server
php bin/worker.php                             # Run background worker
```

### PHP Patterns Cheat Sheet

```php
<?php
// ═══════════════════════════════════════════════════════════════════════════
// VALUE OBJECTS (Immutable, Self-Validating)
// ═══════════════════════════════════════════════════════════════════════════
final readonly class Email
{
    private function __construct(private string $value) {}

    public static function fromString(string $email): self
    {
        if (!filter_var($email, FILTER_VALIDATE_EMAIL)) {
            throw new \InvalidArgumentException("Invalid email: {$email}");
        }
        return new self($email);
    }

    public function value(): string { return $this->value; }
    public function equals(self $other): bool { return $this->value === $other->value; }
}

// ═══════════════════════════════════════════════════════════════════════════
// ENUMS (Type-Safe Constants)
// ═══════════════════════════════════════════════════════════════════════════
enum Status: string
{
    case Pending = 'pending';
    case Active = 'active';
    case Inactive = 'inactive';

    public function isActive(): bool { return $this === self::Active; }
}

// ═══════════════════════════════════════════════════════════════════════════
// RESULT TYPE (Error Handling Without Exceptions)
// ═══════════════════════════════════════════════════════════════════════════
/**
 * @template T
 * @template E
 */
final readonly class Result
{
    /** @param T|null $value @param E|null $error */
    private function __construct(
        private mixed $value,
        private mixed $error,
        private bool $isSuccess
    ) {}

    /** @param T $value @return self<T, never> */
    public static function ok(mixed $value): self
    {
        return new self($value, null, true);
    }

    /** @param E $error @return self<never, E> */
    public static function err(mixed $error): self
    {
        return new self(null, $error, false);
    }

    public function isOk(): bool { return $this->isSuccess; }
    public function isErr(): bool { return !$this->isSuccess; }
    /** @return T */ public function unwrap(): mixed { return $this->value; }
    /** @return E */ public function error(): mixed { return $this->error; }
}

// ═══════════════════════════════════════════════════════════════════════════
// REPOSITORY INTERFACE (Port in Hexagonal Architecture)
// ═══════════════════════════════════════════════════════════════════════════
interface UserRepositoryInterface
{
    public function findById(UserId $id): ?User;
    public function findByEmail(Email $email): ?User;
    public function save(User $user): void;
    public function delete(UserId $id): void;
}

// ═══════════════════════════════════════════════════════════════════════════
// COMMAND/HANDLER PATTERN (CQRS)
// ═══════════════════════════════════════════════════════════════════════════
final readonly class CreateUserCommand
{
    public function __construct(
        public string $name,
        public string $email
    ) {}
}

final readonly class CreateUserHandler
{
    public function __construct(
        private UserRepositoryInterface $repository,
        private EventBusInterface $eventBus
    ) {}

    public function handle(CreateUserCommand $cmd): User
    {
        $user = User::create($cmd->name, $cmd->email);
        $this->repository->save($user);
        $this->eventBus->publish(new UserCreatedEvent($user->id()));
        return $user;
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// ASYNC PATTERNS (AMPHP)
// ═══════════════════════════════════════════════════════════════════════════
use function Amp\async;
use function Amp\Future\await;

// Parallel execution
[$user, $orders, $stats] = await([
    async(fn() => $userRepo->findById($userId)),
    async(fn() => $orderRepo->findByUser($userId)),
    async(fn() => $statsRepo->getForUser($userId)),
]);

// Sequential with async
$user = $userRepo->findById($userId);  // Runs async, suspends fiber
$orders = $orderRepo->findByUser($user->id());  // Runs after user found

// ═══════════════════════════════════════════════════════════════════════════
// DEPENDENCY INJECTION (Constructor Injection)
// ═══════════════════════════════════════════════════════════════════════════
final readonly class UserService
{
    public function __construct(
        private UserRepositoryInterface $userRepository,  // Interface, not concrete
        private LoggerInterface $logger,                  // PSR-3 Logger
        private EventBusInterface $eventBus               // Event publishing
    ) {}
}

// ═══════════════════════════════════════════════════════════════════════════
// TESTING PATTERNS
// ═══════════════════════════════════════════════════════════════════════════
final class UserServiceTest extends TestCase
{
    // Arrange-Act-Assert pattern
    public function testCreateUserReturnsUser(): void
    {
        // Arrange
        $repository = $this->createMock(UserRepositoryInterface::class);
        $repository->method('findByEmail')->willReturn(null);
        $service = new UserService($repository);

        // Act
        $user = $service->create('John', 'john@example.com');

        // Assert
        self::assertEquals('John', $user->name());
    }

    // Data provider pattern
    /** @dataProvider invalidEmailsProvider */
    public function testRejectsInvalidEmails(string $email): void
    {
        $this->expectException(\InvalidArgumentException::class);
        Email::fromString($email);
    }

    public static function invalidEmailsProvider(): array
    {
        return [
            'no at symbol' => ['invalid'],
            'no domain' => ['user@'],
            'no local part' => ['@domain.com'],
        ];
    }
}
```

### Project Structure

```
┌───────────────────────────────────────────────────────────────────────────┐
│                    STANDARD PHP PROJECT STRUCTURE                         │
├───────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  project-root/                                                            │
│  │                                                                        │
│  ├── bin/                         # Executable scripts                    │
│  │   ├── console                  # CLI entry point                       │
│  │   └── server.php               # AMPHP server entry point              │
│  │                                                                        │
│  ├── config/                      # Configuration files                   │
│  │   ├── services.php             # DI container configuration            │
│  │   ├── routes.php               # HTTP routes                           │
│  │   └── packages/                # Package-specific configs              │
│  │                                                                        │
│  ├── public/                      # Web root (if applicable)              │
│  │   └── index.php                # Front controller                      │
│  │                                                                        │
│  ├── src/                         # Source code (PSR-4: App\)             │
│  │   │                                                                    │
│  │   ├── Domain/                  # Core domain (no dependencies)         │
│  │   │   ├── Entity/              # Domain entities                       │
│  │   │   │   └── User.php                                                 │
│  │   │   ├── ValueObject/         # Value objects                         │
│  │   │   │   ├── UserId.php                                               │
│  │   │   │   └── Email.php                                                │
│  │   │   ├── Repository/          # Repository interfaces (ports)         │
│  │   │   │   └── UserRepositoryInterface.php                              │
│  │   │   ├── Service/             # Domain services                       │
│  │   │   └── Event/               # Domain events                         │
│  │   │                                                                    │
│  │   ├── Application/             # Use cases & orchestration             │
│  │   │   ├── Command/             # Commands (write operations)           │
│  │   │   │   ├── CreateUserCommand.php                                    │
│  │   │   │   └── CreateUserHandler.php                                    │
│  │   │   ├── Query/               # Queries (read operations)             │
│  │   │   │   ├── GetUserQuery.php                                         │
│  │   │   │   └── GetUserHandler.php                                       │
│  │   │   └── Service/             # Application services                  │
│  │   │                                                                    │
│  │   ├── Infrastructure/          # External implementations              │
│  │   │   ├── Persistence/         # Database implementations              │
│  │   │   │   ├── Mysql/                                                   │
│  │   │   │   │   └── MysqlUserRepository.php                              │
│  │   │   │   └── Redis/                                                   │
│  │   │   │       └── RedisCache.php                                       │
│  │   │   ├── Messaging/           # Queue/event implementations           │
│  │   │   │   └── AmqpEventBus.php                                         │
│  │   │   └── Http/                # HTTP client implementations           │
│  │   │                                                                    │
│  │   └── Adapter/                 # Driving adapters (entry points)       │
│  │       ├── Http/                # HTTP controllers                      │
│  │       │   ├── Controller/                                              │
│  │       │   │   └── UserController.php                                   │
│  │       │   └── Middleware/                                              │
│  │       ├── Console/             # CLI commands                          │
│  │       │   └── Command/                                                 │
│  │       └── Api/                 # API handlers (GraphQL, gRPC)          │
│  │                                                                        │
│  ├── tests/                       # Test code                             │
│  │   ├── Unit/                    # Unit tests (mirror src/ structure)    │
│  │   │   ├── Domain/                                                      │
│  │   │   ├── Application/                                                 │
│  │   │   └── Infrastructure/                                              │
│  │   ├── Integration/             # Integration tests                     │
│  │   │   └── Infrastructure/                                              │
│  │   ├── E2E/                     # End-to-end tests                      │
│  │   │   └── Http/                                                        │
│  │   └── Fixtures/                # Test data fixtures                    │
│  │                                                                        │
│  ├── var/                         # Generated files (gitignored)          │
│  │   ├── cache/                   # Cache files                           │
│  │   └── log/                     # Log files                             │
│  │                                                                        │
│  ├── vendor/                      # Composer dependencies (gitignored)    │
│  │                                                                        │
│  ├── .env                         # Environment variables (gitignored)    │
│  ├── .env.example                 # Example environment file              │
│  ├── .gitignore                                                           │
│  ├── .php-cs-fixer.php            # PHP-CS-Fixer configuration            │
│  ├── composer.json                # Composer dependencies                 │
│  ├── composer.lock                # Locked dependency versions            │
│  ├── phpstan.neon                 # PHPStan configuration                 │
│  ├── phpunit.xml                  # PHPUnit configuration                 │
│  ├── psalm.xml                    # Psalm configuration (alternative)     │
│  └── README.md                    # Project documentation                 │
│                                                                           │
└───────────────────────────────────────────────────────────────────────────┘

LAYER DEPENDENCIES (Hexagonal Architecture):

  ┌─────────────────────────────────────────────────────────────────────────┐
  │                                                                         │
  │    Adapter ─────► Application ─────► Domain ◄───── Infrastructure      │
  │    (HTTP,        (Commands,         (Entities,     (MySQL, Redis,      │
  │     CLI)          Queries)           Ports)         External APIs)     │
  │                                                                         │
  │    ════════════════════════════════════════════════════════════════    │
  │    Direction of dependencies: Outside → Inside                         │
  │    Domain has NO external dependencies                                  │
  │    Infrastructure IMPLEMENTS Domain interfaces                          │
  │                                                                         │
  └─────────────────────────────────────────────────────────────────────────┘
```

### Configuration Files Quick Reference

```php
<?php
// .php-cs-fixer.php - Code Style Configuration
return (new PhpCsFixer\Config())
    ->setRules([
        '@PSR12' => true,
        '@PHP83Migration' => true,
        'strict_param' => true,
        'declare_strict_types' => true,
        'array_syntax' => ['syntax' => 'short'],
        'ordered_imports' => ['sort_algorithm' => 'alpha'],
        'no_unused_imports' => true,
        'single_quote' => true,
        'trailing_comma_in_multiline' => true,
    ])
    ->setFinder(
        PhpCsFixer\Finder::create()
            ->in(__DIR__ . '/src')
            ->in(__DIR__ . '/tests')
    );
```

```yaml
# phpstan.neon - Static Analysis Configuration
parameters:
    level: max
    paths:
        - src
        - tests
    excludePaths:
        - src/Infrastructure/Migration
    checkMissingIterableValueType: true
    checkGenericClassInNonGenericObjectType: true
    treatPhpDocTypesAsCertain: false
```

```json
// composer.json - Dependency Management
{
    "require": {
        "php": "^8.3",
        "amphp/amp": "^3.0",
        "amphp/http-server": "^3.0",
        "amphp/mysql": "^3.0",
        "psr/log": "^3.0"
    },
    "require-dev": {
        "phpunit/phpunit": "^11.0",
        "phpstan/phpstan": "^1.10",
        "friendsofphp/php-cs-fixer": "^3.0",
        "amphp/phpunit-util": "^3.0"
    },
    "autoload": {
        "psr-4": { "App\\": "src/" }
    },
    "autoload-dev": {
        "psr-4": { "Tests\\": "tests/" }
    }
}
```

---

## 11. Security & Dependency Management (MANDATORY)

### A. Automated Dependency Management

**Use Composer with lockfiles and automation for consistent environments:**

```json
// composer.json
{
    "config": {
        "allow-plugins": {
            "php-http/discovery": true
        },
        "sort-packages": true
    },
    "require": {
        "php": "^8.4"
    }
}
```

- **Lockfiles**: ALWAYS commit `composer.lock`. Use `composer install --no-dev` in production.
- **Dependency Auditing**: Integrate `composer audit` into CI to catch vulnerabilities.

### B. Vulnerability Scanning & Security

**Mandatory security checks for ALL PHP projects:**

1. **Vulnerability Scan**:
   ```bash
   # Scan for known vulnerabilities in dependencies
   composer audit
   ```
   - Agents MUST fix all discoverable high/critical vulnerabilities before presentation.

2. **Supply Chain Audit**:
   - Verify package hashes in `composer.lock`.
   - Use `composer validate --strict` to ensure config integrity.

### C. Dependency File

```json
// Example composer.json
{
    "require": {
        "amphp/amp": "^3.0",
        "psr/log": "^3.0"
    },
    "require-dev": {
        "phpunit/phpunit": "^11.5",
        "vimeo/psalm": "^6.0"
    }
}
```

---

## 12. Deployment Checklist

### Agent-Generated Code Verification (MANDATORY)

#### Build & Compilation
- [ ] Code parses: `php -l` returns 0 for all files
- [ ] No syntax errors or deprecated features
- [ ] PHP 8.4 features used (Property hooks, Asymmetric visibility)
- [ ] Code formatted: `php-cs-fixer fix --dry-run` passes

#### Testing
- [ ] All tests pass: `phpunit` returns exit code 0
- [ ] Reasonable coverage: `XDEBUG_MODE=coverage` shows >80%
- [ ] Async operations verified (AMPHP/Fibers)

#### Security
- [ ] Dependency scan passes: `composer audit` shows 0 vulnerabilities
- [ ] Supply chain verified: `composer.lock` is in sync
- [ ] Secrets check: No hardcoded API keys or passwords in `.env`
- [ ] Static analysis: `psalm` or `phpstan` passes at max level

#### Code Quality
- [ ] No unused imports or dead code
- [ ] Readonly classes/properties used for immutable data
- [ ] Project structure follows hexagonal layout

#### Documentation
- [ ] All public APIs have PHPDoc comments
- [ ] Documentation follows PSR-5/PSR-19 conventions
- [ ] Examples provided for complex APIs

#### Architecture
- [ ] Hexagonal architecture followed (Ports and Adapters)
- [ ] Dependency injection used via constructor
- [ ] Async-first design where I/O is involved

#### Agent Workflow Completed
- [ ] Agent verified code parses/builds successfully
- [ ] Agent ran all tests and verified they pass
- [ ] Agent ran security scans and verified 0 high vulnerabilities
- [ ] Agent verified documentation and PHPDoc

---

## 13. Why This Configuration Works

**PHP 8.4 Property Hooks**:
- Dramatically reduces boilerplate by allowing validation and transformation logic directly inside property definitions, eliminating the need for many manual getters and setters.

**AMPHP & Fibers**:
- Enables high-performance, non-blocking I/O without the "callback hell" of traditional async patterns, allowing PHP to scale to thousands of concurrent connections.

**Hexagonal Architecture**:
- Decouples core business logic from infrastructure (like the DB or HTTP framework), making the application highly testable and resilient to technology changes.

---

## 14. Quick Reference

### Common Commands

```bash
# Build/Install
composer install

# Test
./vendor/bin/phpunit

# Security Scan
composer audit

# Static Analysis
./vendor/bin/psalm

# Lint and Format
./vendor/bin/php-cs-fixer fix
```

### Modern PHP 8.4 Patterns Cheat Sheet

```php
// Property Hooks (PHP 8.4)
public string $name {
    set => strlen($value) < 2 ? throw new Error() : $value;
    get => strtoupper($value);
}

// Asymmetric Visibility (PHP 8.4)
public private(set) string $id;

// New without parentheses
$name = new User()->name;

// Array find functions
$user = array_find($users, fn($u) => $u->id === $id);
```

---

**Last Updated:** 2026-02-06
**Version:** 1.2
**Maintainer:** PHP Team


**End of Modern PHP Development Guidelines**
