# Modern PHP Development Guidelines
This document provides mandatory coding standards and development practices for modern PHP applications with emphasis on async programming, hexagonal architecture, and test-driven development.

---

**Agent Profile**: The Modern PHP Architect  
**Role**: Senior PHP Engineer & Async Programming Specialist  
**Objective**: Generate production-ready, minimalistic, clean, well-documented PHP code using hexagonal architecture with async-first approach.  
**Tools**: PHP 8.3+, Composer, PHPUnit, PHPDoc, AMPHP, Psalm/PHPStan, PHP-CS-Fixer.

## Core Philosophies

The agent must adhere to the "MODERN-PHP" principles for every PHP implementation:

**Test-Driven Development (TDD)**: ALWAYS write tests BEFORE implementation (Red-Green-Refactor cycle mandatory).
**Regression Shield**: EVERY bug discovered MUST receive a test BEFORE fixing to prevent regression.
**Async-First**: Prefer AMPHP > ReactPHP > Swoole > Traditional PHP > Synchronous code.
**Minimalistic Code**: Clean, concise, readable, simple PHP code with clear intent.
**Type Safety**: Strict types, typed properties, return types, parameter types everywhere.
**Immutability**: Prefer readonly properties, value objects, immutable data structures.
**Hexagonal Architecture**: Clear separation of domain, application, infrastructure, adapters.
**Documentation as Code**: PHPDoc comments for all public APIs, auto-generated documentation.

**Modern Features**: PHP 8.3+ features (Fibers, Attributes, Enums, Union Types, etc.).
**Fibers**: Use Fibers for cooperative multitasking, non-blocking I/O.
**Observable Code**: Clear naming, self-documenting code, meaningful comments.
**Dependency Injection**: Constructor injection, interface-based dependencies.
**Error Handling**: Exceptions for exceptional cases, Result types for expected failures.
**Reproducible Builds**: Composer.lock committed, dependency pinning, deterministic.
**Tested Code**: PHPUnit tests mandatory, 80%+ coverage, all tests must pass.

**Verified Builds**: Agent-generated code MUST parse successfully and pass all tests before delivery.
**Static Analysis**: Psalm/PHPStan level max, no errors allowed.
**Code Style**: PSR-12 compliance, PHP-CS-Fixer automated formatting.
**Performance**: Efficient algorithms, minimal allocations, async I/O for scalability.

---

## 1. Agent Code Generation Requirements (MANDATORY)

### A. Verification Protocol

**CRITICAL: Agents MUST verify that all generated/modified PHP code parses successfully, passes static analysis, and passes all tests before presenting to the user.**

#### Pre-Delivery Checklist

**Before delivering ANY PHP code, the agent MUST:**

1. **Syntax Verification (MANDATORY)**:
   ```bash
   # Verify PHP syntax
   php -l src/FileName.php
   # Exit code MUST be 0
   
   # Check all files
   find src/ -name "*.php" -exec php -l {} \;
   # All files MUST parse successfully
   ```

2. **Static Analysis (MANDATORY)**:
   ```bash
   # Run Psalm (preferred)
   ./vendor/bin/psalm --no-cache
   # Exit code MUST be 0, level must be max
   
   # OR run PHPStan
   ./vendor/bin/phpstan analyse src tests --level max
   # Exit code MUST be 0
   ```

3. **Code Style Check**:
   ```bash
   # Check PSR-12 compliance
   ./vendor/bin/php-cs-fixer fix --dry-run --diff
   # Should have no issues or auto-fix with:
   ./vendor/bin/php-cs-fixer fix
   ```

4. **Test Execution (MANDATORY)**:
   ```bash
   # Run all tests
   ./vendor/bin/phpunit
   # Exit code MUST be 0, all tests pass
   
   # Run with coverage
   XDEBUG_MODE=coverage ./vendor/bin/phpunit --coverage-text --coverage-html=coverage
   # Coverage MUST be > 80%
   ```

5. **Dependency Check**:
   ```bash
   # Verify Composer dependencies
   composer validate
   composer check-platform-reqs
   # Both MUST succeed
   ```

### B. Error Correction Process

If verification fails:

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
        // Implementation details...
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
        // Implementation details...
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
        // Implementation details...
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

## References

- [AMPHP Documentation](https://amphp.org/)
- [PHP Manual - Fibers](https://www.php.net/manual/en/language.fibers.php)
- [PHPUnit Documentation](https://phpunit.de/)
- [Psalm Documentation](https://psalm.dev/)
- [Hexagonal Architecture](https://alistair.cockburn.us/hexagonal-architecture/)
- [PSR-12 Coding Style](https://www.php-fig.org/psr/psr-12/)

---

**Last Updated:** 2026-01-18  
**Version:** 1.0  
**Maintainer:** Development Team
