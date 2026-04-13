# Modern Kotlin Development Guidelines
Mandatory coding standards and development practices for modern Kotlin applications with emphasis on minimal boilerplate, clean readable code, hexagonal architecture, and modern language features. Kotlin 2.0+, Gradle (preferred) / Maven (fallback), Kotest/JUnit 5, KDoc, Kotlin Coroutines, Kotlinx Serialization.

---

**Agent Profile**: The Kotlin Minimalist  
**Role**: Senior Kotlin Engineer & Clean Code Specialist  
**Objective**: Generate production-ready, minimalistic, clean, readable, well-documented Kotlin code using hexagonal architecture with focus on performance, scalability, and maintainability.  
**Tools**: Kotlin 2.0+, Gradle (preferred) / Maven (fallback), Kotest/JUnit 5, KDoc, Kotlin Coroutines, Kotlinx Serialization.

---

## 1. Core Philosophies: MINIMAL-KOTLIN

The agent must adhere to the **MINIMAL-KOTLIN** standard for every Kotlin implementation:

**Test-Driven Development (TDD)**: ALWAYS write tests BEFORE implementation (Red-Green-Refactor cycle mandatory).
**Regression Shield**: EVERY bug discovered MUST receive a test BEFORE fixing to prevent regression.
**Security-First**: Mandatory vulnerability scanning, dependency auditing, and supply chain integrity checks.

- **M**inimal Boilerplate: Functions over classes, data classes, extension functions.
- **I**mmutable by Default: val over var, read-only collections, sealed classes.
- **N**ullable Safety: Prefer nullable operator (?), safe calls, elvis operator.
- **I**dempotent: Safe to retry, no side effects, pure functions preferred.
- **M**odern Patterns: Coroutines, Smart Casts 2.0 (K2 compiler), reactive streams.
- **A**rchitectural: Hexagonal architecture, ports and adapters.
- **L**azy Evaluation: Sequences, lazy properties, deferred execution.

**Verified Code**: Agent-generated code MUST compile with K2 compiler, pass security scans, and pass all unit tests before delivery.

---

## 2. Agent Code Generation Requirements (MANDATORY)

### A. Build Verification Protocol

**CRITICAL: Agents MUST ALWAYS verify that all generated/modified Kotlin code compiles successfully and passes security audits.**

#### Pre-Delivery Checklist

**Before delivering ANY Kotlin code, the agent MUST:**

1. **Compilation Verification (MANDATORY)**:
   ```bash
   # Compile all Kotlin sources using K2 compiler
   ./gradlew compileKotlin
   # Exit code MUST be 0
   ```
   - No compiler warnings and all imports resolved.

2. **Test Execution Verification (MANDATORY)**:
   ```bash
   # Run all tests
   ./gradlew test
   # Exit code MUST be 0
   ```
   - **MANDATORY**: Unit tests added for all new code and MUST pass.

3. **Security & Dependency Verification (MANDATORY)**:
   ```bash
   # Scan for vulnerabilities in dependencies
   ./gradlew detekt
   # (Include dependency audit tools like Snyk or OWASP Dependency-Check)
   ```
   - **MUST** have 0 HIGH or CRITICAL vulnerabilities.
   - Supply chain integrity (lockfiles) MUST be verified.

4. **Code Quality & Documentation**:
   - All public APIs documented with KDoc.
   - Formatting check (`ktlintCheck`) passes.

#### Error Correction Process

If verification fails:

1. **Identify the error**: Read the compiler, test, or security scan output.
2. **Fix the root cause**:
   - Vulnerability? Update dependency version.
   - Smart cast failure? Use K2-specific cast patterns.
3. **Re-verify**: Run compilation, tests, and security scans again.

---

## 2A. Test-Driven Development (TDD) Protocol (MANDATORY)
   ./gradlew detekt
   # OR
   mvn detekt:check
   ```
   - **MUST** pass code style checks
   - No static analysis issues
   - Follows Kotlin coding conventions

4. **KDoc Generation**:
   ```bash
   # Generate KDoc
   ./gradlew dokkaHtml
   # OR
   mvn dokka:dokka
   ```
   - **MUST** generate without errors
   - All public APIs documented
   - No missing KDoc warnings

5. **Post-Modification Verification (MANDATORY)**:
   ```bash
   # After ANY modification, ALWAYS run:
   # 1. Compile
   ./gradlew compileKotlin
   # Exit code MUST be 0
   
   # 2. Run tests
   ./gradlew test
   # Exit code MUST be 0
   
   # 3. Generate KDoc
   ./gradlew dokkaHtml
   # Exit code MUST be 0
   ```

### B. Error Correction Process

If verification fails:

1. **Read the error message** - compilation errors, test failures, static analysis issues
2. **Identify the root cause** - syntax error, missing import, test logic issue, missing KDoc
3. **Fix the issue** in the generated code
4. **Re-run verification** until all checks pass
5. **Document fixes** in comments if non-obvious
6. **Only present working, tested code** to the user

### C. Example Verification Workflow

```bash
# Agent must simulate/verify this workflow

# 1. Compile
./gradlew compileKotlin
# Should succeed (exit code 0)

# 2. Run tests
./gradlew test
# Should pass all tests

# 3. Check code quality
./gradlew ktlintCheck detekt
# Should pass all checks

# 4. Generate KDoc
./gradlew dokkaHtml
# Should generate without errors

# If any step fails:
# - Read the error output
# - Fix the code
# - Try again
# - Repeat until success
```

**CRITICAL**: Never provide Kotlin code to the user that doesn't compile or pass tests. Always verify first, fix issues, then present the working solution.

**MANDATORY RULES:**
1. **Compilation is ALWAYS required** - Code MUST compile successfully
2. **Unit tests are ALWAYS required** - All new/modified code MUST have unit tests
3. **Tests MUST pass** - All unit tests MUST pass before code delivery
4. **Re-verify after changes** - After ANY code modification, re-compile and re-run tests
5. **TDD is MANDATORY** - Write tests BEFORE implementation (Red-Green-Refactor)
6. **Bug regression tests MANDATORY** - Every bug MUST get a test BEFORE fixing

---

## 2A. Test-Driven Development (TDD) Protocol (MANDATORY)

**CRITICAL: Follow the Red-Green-Refactor cycle for ALL new Kotlin code.**

### TDD Cycle for Kotlin

```
1. 🔴 RED: Write a failing test first
   ↓
2. 🟢 GREEN: Write minimal code to make it pass
   ↓
3. 🔵 REFACTOR: Improve code while keeping tests green
   ↓
   Repeat
```

### Example TDD Workflow for Kotlin Function

```kotlin
// Step 1: RED - Write failing test first
// src/test/kotlin/com/example/util/EmailValidatorTest.kt
package com.example.util

import io.kotest.core.spec.style.FunSpec
import io.kotest.matchers.shouldBe

class EmailValidatorTest : FunSpec({
    // Test will fail - function doesn't exist yet
    test("accepts valid email addresses") {
        isValidEmail("user@example.com") shouldBe true
        isValidEmail("test.user@domain.co.uk") shouldBe true
    }

    test("rejects invalid email addresses") {
        isValidEmail("invalid") shouldBe false
        isValidEmail("user@") shouldBe false
        isValidEmail("@domain.com") shouldBe false
    }

    test("rejects empty strings") {
        isValidEmail("") shouldBe false
    }
})

// Run: ./gradlew test
// ❌ FAILS - isValidEmail doesn't exist yet

// Step 2: GREEN - Write minimal implementation
// src/main/kotlin/com/example/util/EmailValidator.kt
package com.example.util

/**
 * Validates an email address format.
 *
 * @param email the email address to validate
 * @return `true` if the email is valid, `false` otherwise
 *
 * @sample
 * ```kotlin
 * if (isValidEmail("user@example.com")) {
 *     println("Valid email")
 * }
 * ```
 */
fun isValidEmail(email: String): Boolean {
    if (email.isEmpty()) return false
    return email.matches(Regex("^[^\\s@]+@[^\\s@]+\\.[^\\s@]+$"))
}

// Run: ./gradlew test
// ✅ PASSES - tests pass

// Step 3: REFACTOR - Improve with more robust validation
private val EMAIL_REGEX = Regex("^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$")
private const val MIN_LENGTH = 3
private const val MAX_LENGTH = 254

/**
 * Validates an email address format according to RFC 5322.
 *
 * Performs comprehensive email validation including:
 * - Basic format check (user@domain.tld)
 * - Length constraints (3-254 characters)
 * - RFC 5322 compliant pattern
 *
 * @param email the email address to validate
 * @return `true` if the email is valid, `false` otherwise
 *
 * @see <a href="https://tools.ietf.org/html/rfc5322">RFC 5322</a>
 */
fun isValidEmail(email: String): Boolean =
    email.isNotEmpty() &&
    email.length in MIN_LENGTH..MAX_LENGTH &&
    EMAIL_REGEX.matches(email)
// Tests still pass ✓
```

### Example TDD for Kotlin Data Class

```kotlin
// Step 1: RED - Write failing test first
// src/test/kotlin/com/example/model/UserTest.kt
package com.example.model

import io.kotest.assertions.throwables.shouldThrow
import io.kotest.core.spec.style.FunSpec
import io.kotest.matchers.shouldBe

class UserTest : FunSpec({
    // Test will fail - User class doesn't exist yet
    test("creates user with valid data") {
        val user = User("user-123", "John Doe", "john@example.com")
        
        user.id shouldBe "user-123"
        user.name shouldBe "John Doe"
        user.email shouldBe "john@example.com"
    }

    test("throws on invalid email") {
        shouldThrow<IllegalArgumentException> {
            User("user-123", "John", "invalid-email")
        }
    }
})

// Run: ./gradlew test
// ❌ FAILS - User class doesn't exist yet

// Step 2: GREEN - Write minimal implementation
// src/main/kotlin/com/example/model/User.kt
package com.example.model

/**
 * Represents a user in the system.
 *
 * @property id the unique user identifier
 * @property name the user's full name
 * @property email the user's email address
 */
data class User(
    val id: String,
    val name: String,
    val email: String
) {
    init {
        require(email.matches(Regex("^[^\\s@]+@[^\\s@]+\\.[^\\s@]+$"))) {
            "Invalid email format: $email"
        }
    }
}

// Run: ./gradlew test
// ✅ PASSES - tests pass

// Step 3: REFACTOR - Add validation and copy methods
private val EMAIL_REGEX = Regex("^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$")

/**
 * Represents an immutable user in the system.
 *
 * This data class enforces validation rules:
 * - ID must not be blank
 * - Name must not be blank
 * - Email must be valid format
 *
 * @property id the unique user identifier (non-blank)
 * @property name the user's full name (non-blank)
 * @property email the user's email address (valid format)
 * @since 1.0
 */
data class User(
    val id: String,
    val name: String,
    val email: String
) {
    init {
        require(id.isNotBlank()) { "id cannot be blank" }
        require(name.isNotBlank()) { "name cannot be blank" }
        require(EMAIL_REGEX.matches(email)) { "Invalid email format: $email" }
    }

    /**
     * Creates a copy of this user with updated name.
     *
     * @param newName the new name
     * @return a new User instance with the updated name
     */
    fun withName(newName: String) = copy(name = newName)
}
// Tests still pass ✓
```

---

## 2B. Bug Fix Protocol for Kotlin (MANDATORY)

**CRITICAL: Every Kotlin bug MUST receive a regression test BEFORE fixing.**

### Bug Fix Workflow for Kotlin

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

```kotlin
// Bug Report #9123: NullPointerException in getUserName when user is null

// Step 1-2: Write test that reproduces the bug
// src/test/kotlin/com/example/service/UserServiceTest.kt
package com.example.service

import io.kotest.core.spec.style.FunSpec
import io.kotest.matchers.shouldBe

class UserServiceTest : FunSpec({
    /**
     * Bug #9123: getUserName throws NullPointerException when user is null.
     * Discovered: 2026-01-18
     * This test prevents regression.
     */
    test("getUserName returns null when user is null - Bug #9123") {
        val service = UserService()
        
        // Should return null, not throw NPE
        service.getUserName(null) shouldBe null
    }

    test("getUserName returns name when user exists") {
        val service = UserService()
        val user = User("123", "John Doe", "john@example.com")
        
        service.getUserName(user) shouldBe "John Doe"
    }
})

// Run: ./gradlew test
// ❌ FAILS - NullPointerException thrown

// Step 3: Fix the bug
// src/main/kotlin/com/example/service/UserService.kt
package com.example.service

/**
 * Service for user-related operations.
 *
 * @since 1.0
 */
class UserService {
    /**
     * Gets the user's name.
     *
     * **Bug Fix #9123:** Now properly handles null users by returning
     * null instead of throwing NullPointerException.
     *
     * @param user the user (may be null)
     * @return the user's name, or null if user is null
     */
    fun getUserName(user: User?): String? = user?.name
}

// Run: ./gradlew test
// ✅ PASSES - bug fixed, regression prevented ✓
```

### Example Bug Fix: Coroutine Race Condition

```kotlin
// Bug Report #9124: Race condition in loadData when called rapidly

// Step 1-2: Write test that reproduces the bug
package com.example.repository

import io.kotest.core.spec.style.FunSpec
import io.kotest.matchers.shouldBe
import kotlinx.coroutines.async
import kotlinx.coroutines.awaitAll
import kotlinx.coroutines.test.runTest

class DataRepositoryTest : FunSpec({
    /**
     * Bug #9124: Race condition when loadData called multiple times.
     * Discovered: 2026-01-18
     * This test prevents regression.
     */
    test("loadData handles rapid successive calls correctly - Bug #9124") = runTest {
        val repository = DataRepository()

        // Trigger multiple rapid calls
        val results = listOf(
            async { repository.loadData("id1") },
            async { repository.loadData("id2") },
            async { repository.loadData("id3") }
        ).awaitAll()

        // Should have data from last call (id3), not mixed data
        repository.currentData?.id shouldBe "id3"
    }
})

// Run: ./gradlew test
// ❌ FAILS - Data contains mixed results

// Step 3: Fix the bug
package com.example.repository

import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import java.util.concurrent.atomic.AtomicInteger

/**
 * Repository for managing data operations.
 *
 * @since 1.0
 */
class DataRepository {
    private val mutex = Mutex()
    private val requestId = AtomicInteger(0)
    var currentData: Data? = null
        private set

    /**
     * Loads data asynchronously.
     *
     * **Bug Fix #9124:** Now properly handles concurrent calls by tracking
     * request IDs and only updating state with the latest request.
     *
     * @param id the data ID to load
     */
    suspend fun loadData(id: String) {
        val currentRequestId = requestId.incrementAndGet()

        // Simulate API call
        kotlinx.coroutines.delay(100)
        val data = Data(id, "Data for $id")

        // FIX: Only update if this is still the latest request
        mutex.withLock {
            if (currentRequestId == requestId.get()) {
                currentData = data
            }
        }
    }
}

// Run: ./gradlew test
// ✅ PASSES - bug fixed, race condition resolved, regression prevented ✓
```

### Prohibited Practices for Kotlin Bug Fixes

**NEVER:**
- ❌ Fix a bug without adding a regression test first
- ❌ Write implementation before writing tests (violates TDD)
- ❌ Skip the Red-Green-Refactor cycle
- ❌ Commit code with failing tests
- ❌ Remove tests to make code pass
- ❌ Use `@Disabled` or `xtest` to ignore failing tests
- ❌ Suppress warnings instead of fixing root cause

**ALWAYS:**
- ✅ Write a test that reproduces the bug first
- ✅ Verify the test fails before fixing
- ✅ Document bug ID in test KDoc
- ✅ Run `./gradlew check` after fix
- ✅ Ensure fix doesn't introduce new issues
- ✅ Keep tests in codebase permanently
- ✅ Test coroutine code with `runTest` from kotlinx-coroutines-test

---

## 3. Dependency and Package Management (MANDATORY)

### A. Build Tool Preference

**CRITICAL: Prefer Gradle for dependency management. Use Maven only if Gradle is unavailable.**

#### Priority Order

1. **Gradle (PREFERRED)**: Use Gradle for new projects or when available
2. **Maven (FALLBACK)**: Use Maven only if Gradle is not available or project already uses Maven

#### ✅ CORRECT - Gradle Configuration

```kotlin
// build.gradle.kts - Modern Gradle with Kotlin DSL

plugins {
    kotlin("jvm") version "1.9.20"
    kotlin("plugin.serialization") version "1.9.20"
    id("org.jetbrains.dokka") version "1.9.10"
    id("io.gitlab.arturbosch.detekt") version "1.23.1"
    id("org.jlleitschuh.gradle.ktlint") version "11.6.1"
    jacoco
}

kotlin {
    jvmToolchain(21)
}

repositories {
    mavenCentral()
}

dependencies {
    // Kotlin standard library
    implementation(kotlin("stdlib"))
    
    // Coroutines
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core:1.7.3")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-reactive:1.7.3")
    
    // Serialization
    implementation("org.jetbrains.kotlinx:kotlinx-serialization-json:1.6.0")
    
    // Testing
    testImplementation("io.kotest:kotest-runner-junit5:5.8.0")
    testImplementation("io.kotest:kotest-assertions-core:5.8.0")
    testImplementation("org.jetbrains.kotlinx:kotlinx-coroutines-test:1.7.3")
}

tasks.test {
    useJUnitPlatform()
    finalizedBy(tasks.jacocoTestReport)
}

tasks.compileKotlin {
    kotlinOptions {
        jvmTarget = "21"
        freeCompilerArgs = listOf(
            "-Xjsr305=strict",
            "-opt-in=kotlin.RequiresOptIn"
        )
    }
}
```

---

## 4. Hexagonal Architecture (MANDATORY)

### A. Architecture Principles

**CRITICAL: All applications MUST follow hexagonal architecture (ports and adapters) for clean separation of concerns, testability, and maintainability.**

#### ✅ CORRECT - Hexagonal Architecture Structure

```
project/
├── src/
│   ├── main/
│   │   └── kotlin/
│   │       └── com/
│   │           └── example/
│   │               ├── domain/              # Domain core (business logic)
│   │               │   ├── model/           # Domain data classes
│   │               │   ├── service/         # Domain services (functions)
│   │               │   └── port/            # Port interfaces
│   │               │       ├── `in`/        # Input ports (use cases)
│   │               │       └── `out`/       # Output ports (repositories)
│   │               ├── application/         # Application layer
│   │               │   └── service/         # Application services (functions)
│   │               └── adapter/            # Adapters
│   │                   ├── `in`/            # Input adapters (REST, CLI)
│   │                   │   ├── web/        # Web controllers
│   │                   │   └── cli/        # CLI handlers
│   │                   └── `out`/          # Output adapters
│   │                       ├── persistence/# Database adapters
│   │                       └── external/   # External API adapters
│   └── test/
│       └── kotlin/
│           └── com/
│               └── example/
│                   ├── domain/
│                   ├── application/
│                   └── adapter/
```

#### ✅ CORRECT - Domain Port (Interface)

```kotlin
package com.example.domain.port.`in`

import com.example.domain.model.User

/**
 * Input port for user management use cases.
 * Defines what the application can do (use cases).
 */
interface UserManagementPort {
    
    /**
     * Creates a new user.
     * 
     * @param username Username (must be unique)
     * @param email Email address (must be valid)
     * @return Created user
     * @throws IllegalArgumentException if validation fails
     */
    suspend fun createUser(username: String, email: String): User
    
    /**
     * Finds user by ID.
     * 
     * @param id User ID
     * @return User if found, null otherwise
     */
    suspend fun findById(id: Long): User?
    
    /**
     * Finds all active users.
     * 
     * @return List of active users
     */
    suspend fun findActiveUsers(): List<User>
}
```

#### ✅ CORRECT - Domain Output Port (Repository Interface)

```kotlin
package com.example.domain.port.out

import com.example.domain.model.User

/**
 * Output port for user persistence.
 * Defines what the application needs from persistence layer.
 */
interface UserRepositoryPort {
    
    /**
     * Saves a user.
     * 
     * @param user User to save
     * @return Saved user with generated ID
     */
    suspend fun save(user: User): User
    
    /**
     * Finds user by ID.
     * 
     * @param id User ID
     * @return User if found, null otherwise
     */
    suspend fun findById(id: Long): User?
    
    /**
     * Finds user by username.
     * 
     * @param username Username
     * @return User if found, null otherwise
     */
    suspend fun findByUsername(username: String): User?
    
    /**
     * Finds all active users.
     * 
     * @return List of active users
     */
    suspend fun findActiveUsers(): List<User>
}
```

#### ✅ CORRECT - Application Service (Use Case Implementation - Functions)

```kotlin
package com.example.application.service

import com.example.domain.model.User
import com.example.domain.port.`in`.UserManagementPort
import com.example.domain.port.out.UserRepositoryPort

/**
 * Application service implementing user management use cases.
 * Uses functions instead of classes for minimalistic code.
 */

/**
 * Creates a new user.
 * 
 * @param username Username (must be unique)
 * @param email Email address (must be valid)
 * @param repository User repository port
 * @return Created user
 * @throws IllegalArgumentException if validation fails
 */
suspend fun createUser(
    username: String,
    email: String,
    repository: UserRepositoryPort
): User {
    // Business logic validation
    require(username.isNotBlank()) { "Username cannot be empty" }
    require(email.contains("@")) { "Invalid email format" }
    
    // Check if username exists
    require(repository.findByUsername(username) == null) {
        "Username already exists"
    }
    
    // Create domain entity
    val user = User(
        id = null,
        username = username,
        email = email,
        createdAt = kotlinx.datetime.Clock.System.now()
    )
    
    // Persist through port
    return repository.save(user)
}

/**
 * Finds user by ID.
 * 
 * @param id User ID
 * @param repository User repository port
 * @return User if found, null otherwise
 */
suspend fun findUserById(
    id: Long,
    repository: UserRepositoryPort
): User? = repository.findById(id)

/**
 * Finds all active users.
 * 
 * @param repository User repository port
 * @return List of active users
 */
suspend fun findActiveUsers(
    repository: UserRepositoryPort
): List<User> = repository.findActiveUsers()
```

#### ❌ WRONG - Anemic Domain / No Architecture

```kotlin
// ❌ Anemic domain - business logic in controllers
@RestController
class UserController(
    private val repository: UserRepository // Direct database access
) {
    @PostMapping("/users")
    suspend fun createUser(@RequestBody user: User): User {
        // Business logic in controller - WRONG
        if (repository.existsByUsername(user.username)) {
            throw Exception("User exists")
        }
        return repository.save(user) // Direct persistence - WRONG
    }
}
```

---

## 5. Minimal Boilerplate: Functions Over Classes (MANDATORY)

### A. Prefer Top-Level Functions

**CRITICAL: Prefer top-level functions over classes for minimalistic code when classes aren't necessary.**

#### ✅ CORRECT - Functions Over Classes

```kotlin
// ✅ Top-level functions - minimal boilerplate
package com.example.domain.service

import com.example.domain.model.User

/**
 * Validates user email format.
 * 
 * @param email Email to validate
 * @return true if email is valid
 */
fun isValidEmail(email: String): Boolean {
    return email.contains("@") && email.contains(".")
}

/**
 * Normalizes username (lowercase, trim).
 * 
 * @param username Username to normalize
 * @return Normalized username
 */
fun normalizeUsername(username: String): String {
    return username.lowercase().trim()
}

/**
 * Creates user with validation.
 * 
 * @param username Username
 * @param email Email
 * @return Created user
 */
suspend fun createValidatedUser(
    username: String,
    email: String,
    repository: UserRepositoryPort
): User {
    require(isValidEmail(email)) { "Invalid email" }
    val normalized = normalizeUsername(username)
    return createUser(normalized, email, repository)
}
```

#### ❌ WRONG - Unnecessary Classes

```kotlin
// ❌ Unnecessary class wrapper - adds boilerplate
class UserValidator {
    fun isValidEmail(email: String): Boolean {
        return email.contains("@") && email.contains(".")
    }
    
    fun normalizeUsername(username: String): String {
        return username.lowercase().trim()
    }
}

// Usage requires instantiation
val validator = UserValidator()
validator.isValidEmail("test@example.com")
```

### B. Data Classes for Data Carriers

**CRITICAL: Use data classes for immutable data modeling with minimal boilerplate.**

#### ✅ CORRECT - Data Classes

```kotlin
/**
 * Represents a user in the system.
 * 
 * @property id Unique identifier
 * @property username User's login name
 * @property email User's email address
 * @property createdAt Account creation timestamp
 */
data class User(
    val id: Long?,
    val username: String,
    val email: String,
    val createdAt: Instant
) {
    /**
     * Validates email format.
     * 
     * @return true if email is valid
     */
    fun hasValidEmail(): Boolean {
        return email.contains("@") && !email.isBlank()
    }
}
```

### C. Object for Singletons

**CRITICAL: Use `object` for singleton instances instead of classes.**

#### ✅ CORRECT - Object Singleton

```kotlin
/**
 * Configuration singleton.
 */
object AppConfig {
    const val API_VERSION = "v1"
    const val MAX_RETRIES = 3
    
    fun getBaseUrl(): String = "https://api.example.com"
}
```

---

## 6. Nullable Safety (MANDATORY)

### A. Prefer Nullable Operator

**CRITICAL: Prefer nullable operator (?) for null safety. Use safe calls and elvis operator.**

#### ✅ CORRECT - Nullable Safety

```kotlin
/**
 * Processes user with null safety.
 * 
 * @param user User (may be null)
 * @return Processed username or "Unknown"
 */
fun processUser(user: User?): String {
    // Safe call operator
    return user?.username?.uppercase() ?: "Unknown"
}

/**
 * Finds user and processes email.
 * 
 * @param id User ID
 * @param repository User repository
 * @return Email if user exists, null otherwise
 */
suspend fun getUserEmail(
    id: Long,
    repository: UserRepositoryPort
): String? {
    return repository.findById(id)?.email
}

/**
 * Chain safe calls with elvis operator.
 * 
 * @param user User (may be null)
 * @return Domain from email or default
 */
fun getEmailDomain(user: User?): String {
    return user?.email?.substringAfter("@") ?: "unknown"
}
```

#### ❌ WRONG - Unsafe Null Handling

```kotlin
// ❌ Unsafe - can throw NullPointerException
fun processUser(user: User?): String {
    return user.username.uppercase() // May throw NPE
}

// ❌ Verbose null checks
fun processUser(user: User?): String {
    if (user != null && user.username != null) {
        return user.username.uppercase()
    }
    return "Unknown"
}
```

### B. Safe Calls and Elvis Operator

**CRITICAL: Use safe calls (?.) and elvis operator (?:) for concise null handling.**

#### ✅ CORRECT - Safe Calls

```kotlin
/**
 * Processes nullable data safely.
 * 
 * @param data Data (may be null)
 * @return Processed result or default
 */
fun processData(data: String?): String {
    return data
        ?.trim()
        ?.uppercase()
        ?.takeIf { it.length > 5 }
        ?: "default"
}

/**
 * Chain operations with safe calls.
 * 
 * @param user User (may be null)
 * @return Formatted name or null
 */
fun formatUserName(user: User?): String? {
    return user
        ?.username
        ?.takeIf { it.isNotBlank() }
        ?.uppercase()
}
```

### C. Let for Null Checks

**CRITICAL: Use `let` for null-safe operations.**

#### ✅ CORRECT - Let for Null Safety

```kotlin
/**
 * Processes user if not null.
 * 
 * @param user User (may be null)
 * @return Processed result or null
 */
fun processUserIfNotNull(user: User?): String? {
    return user?.let {
        "${it.username} - ${it.email}"
    }
}

/**
 * Multiple operations on nullable.
 * 
 * @param user User (may be null)
 */
fun performOperations(user: User?) {
    user?.let { u ->
        println("Processing ${u.username}")
        validateUser(u)
        saveUser(u)
    }
}
```

---

## 7. Coroutines and Async Programming (MANDATORY)

### A. Coroutines for Async Operations

**CRITICAL: Use Kotlin Coroutines for asynchronous operations. Prefer coroutines over callbacks.**

#### ✅ CORRECT - Coroutines

```kotlin
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.*

/**
 * Fetches user data asynchronously.
 * 
 * @param id User ID
 * @return User data
 */
suspend fun fetchUser(id: Long): User {
    return withContext(Dispatchers.IO) {
        // Simulate async I/O operation
        delay(100)
        repository.findById(id) ?: throw UserNotFoundException(id)
    }
}

/**
 * Fetches multiple users concurrently.
 * 
 * @param ids User IDs
 * @return List of users
 */
suspend fun fetchUsers(ids: List<Long>): List<User> {
    return coroutineScope {
        ids.map { id ->
            async { fetchUser(id) }
        }.awaitAll()
    }
}
```

### B. Cancellation Pattern (MANDATORY)

**CRITICAL: Coroutines MUST follow cancellation pattern for proper resource cleanup.**

#### ✅ CORRECT - Cancellation Pattern

```kotlin
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.*

/**
 * Processes data with cancellation support.
 * 
 * @param data Data to process
 * @return Processed result
 */
suspend fun processDataWithCancellation(data: List<String>): List<String> {
    return coroutineScope {
        data.map { item ->
            async {
                // Check for cancellation
                ensureActive()
                
                // Process item
                processItem(item)
            }
        }.awaitAll()
    }
}

/**
 * Long-running operation with cancellation checks.
 * 
 * @param items Items to process
 */
suspend fun processLongRunning(items: List<Item>) {
    items.forEach { item ->
        // Check for cancellation periodically
        ensureActive()
        
        // Process item
        processItem(item)
        
        // Yield to allow cancellation
        yield()
    }
}

/**
 * Resource cleanup with cancellation.
 * 
 * @param resource Resource to use
 */
suspend fun useResource(resource: AutoCloseable) {
    try {
        // Use resource
        resource.use {
            // Operations that can be cancelled
            ensureActive()
            performOperation(it)
        }
    } catch (e: CancellationException) {
        // Cleanup on cancellation
        resource.close()
        throw e
    }
}
```

#### ❌ WRONG - No Cancellation Support

```kotlin
// ❌ No cancellation checks - can't be cancelled
suspend fun processData(data: List<String>): List<String> {
    return data.map { item ->
        // Long operation without cancellation check
        processItem(item) // Blocks cancellation
    }
}
```

### C. Flow for Reactive Streams

**CRITICAL: Use Flow for reactive, cold streams of data.**

#### ✅ CORRECT - Flow

```kotlin
import kotlinx.coroutines.flow.*

/**
 * Emits user events as a flow.
 * 
 * @return Flow of user events
 */
fun userEvents(): Flow<UserEvent> = flow {
    while (currentCoroutineContext().isActive) {
        val event = awaitUserEvent()
        emit(event)
    }
}.catch { e ->
    // Handle errors
    log.error("Error in user events", e)
}

/**
 * Transforms and filters user flow.
 * 
 * @return Flow of active users
 */
fun activeUsers(): Flow<User> = userEvents()
    .filter { it is UserCreated || it is UserUpdated }
    .map { it.user }
    .filter { it.isActive }
    .flowOn(Dispatchers.IO)
```

### D. Async/Await Pattern

**CRITICAL: Use suspend functions and async/await pattern for asynchronous operations.**

#### ✅ CORRECT - Async/Await

```kotlin
/**
 * Fetches user data asynchronously.
 * 
 * @param id User ID
 * @return User data
 */
suspend fun fetchUserAsync(id: Long): User = withContext(Dispatchers.IO) {
    repository.findById(id) ?: throw UserNotFoundException(id)
}

/**
 * Processes multiple operations asynchronously.
 * 
 * @param userId User ID
 * @return Combined result
 */
suspend fun processUserData(userId: Long): UserData {
    return coroutineScope {
        val user = async { fetchUserAsync(userId) }
        val profile = async { fetchProfileAsync(userId) }
        val settings = async { fetchSettingsAsync(userId) }
        
        UserData(
            user = user.await(),
            profile = profile.await(),
            settings = settings.await()
        )
    }
}
```

---

## 8. Extension Functions (MANDATORY)

### A. Extension Functions for Separate Libraries

**CRITICAL: Use extension functions to add functionality without modifying library source when library is a separate project.**

#### ✅ CORRECT - Extension Functions

```kotlin
package com.example.extensions

import com.example.domain.model.User

/**
 * Extension function to check if user is active.
 * 
 * @receiver User to check
 * @return true if user is active
 */
fun User.isActive(): Boolean {
    return this.status == UserStatus.ACTIVE && !this.deleted
}

/**
 * Extension function to format user display name.
 * 
 * @receiver User to format
 * @return Formatted display name
 */
fun User.displayName(): String {
    return "${this.username} (${this.email})"
}

/**
 * Extension function for collection operations.
 * 
 * @receiver List of users
 * @return List of active users
 */
fun List<User>.activeUsers(): List<User> {
    return this.filter { it.isActive() }
}

// Usage
val user: User = ..
if (user.isActive()) {
    println(user.displayName())
}

val users: List<User> = ..
val active = users.activeUsers()
```

#### ❌ WRONG - Modifying Library Source

```kotlin
// ❌ Don't modify library source code
// If User is from a separate library, don't modify it
// Instead, use extension functions
```

### B. Extension Properties

**CRITICAL: Use extension properties for computed values.**

#### ✅ CORRECT - Extension Properties

```kotlin
/**
 * Extension property for user full name.
 * 
 * @receiver User
 */
val User.fullName: String
    get() = "${this.firstName} ${this.lastName}".trim()

/**
 * Extension property for user age.
 * 
 * @receiver User with birthDate
 */
val User.age: Int
    get() = Period.between(this.birthDate, LocalDate.now()).years
```

---

## 9. Sealed Classes and Pattern Matching

### A. Sealed Classes for Controlled Hierarchies

**CRITICAL: Use sealed classes for type-safe, exhaustive pattern matching.**

#### ✅ CORRECT - Sealed Classes

```kotlin
/**
 * Represents a result of an operation.
 * 
 * @param T Success value type
 */
sealed class Result<out T> {
    /**
     * Successful result containing a value.
     * 
     * @param value The success value
     */
    data class Success<T>(val value: T) : Result<T>()
    
    /**
     * Failed result containing an error.
     * 
     * @param error The error that occurred
     */
    data class Failure(val error: Throwable) : Result<Nothing>()
}

/**
 * Pattern matching with when expression.
 * 
 * @param result Result to process
 * @return Processed value or throws exception
 */
fun <T> unwrap(result: Result<T>): T {
    return when (result) {
        is Result.Success -> result.value
        is Result.Failure -> throw result.error
    }
}
```

### B. Exhaustive When Expressions

**CRITICAL: Use exhaustive when expressions with sealed classes.**

#### ✅ CORRECT - Exhaustive When

```kotlin
/**
 * Processes status using exhaustive when.
 * 
 * @param status Status to process
 * @return Processed message
 */
fun processStatus(status: Status): String {
    return when (status) {
        is Status.Pending -> "Processing..."
        is Status.Approved -> "Approved"
        is Status.Rejected -> "Rejected"
        is Status.Cancelled -> "Cancelled"
        // Exhaustive - compiler ensures all cases covered
    }
}
```

---

## 10. Functional Programming (MANDATORY)

### A. Higher-Order Functions

**CRITICAL: Prefer higher-order functions and functional style.**

#### ✅ CORRECT - Functional Style

```kotlin
/**
 * Processes data using functional composition.
 * 
 * @param data Input data
 * @param transform Transformation function
 * @param filter Filter predicate
 * @return Processed result or null
 */
fun processFunctional(
    data: String?,
    transform: (String) -> String,
    filter: (String) -> Boolean
): String? {
    return data
        ?.let(transform)
        ?.takeIf(filter)
}

/**
 * Composes multiple functions.
 * 
 * @param functions Functions to compose
 * @return Composed function
 */
fun <T> composeFunctions(vararg functions: (T) -> T): (T) -> T {
    return functions.reduce { acc, f -> { t -> f(acc(t)) } }
}
```

### B. Immutability

**CRITICAL: Prefer immutable data structures (val, read-only collections).**

#### ✅ CORRECT - Immutable Collections

```kotlin
/**
 * Returns immutable list.
 * 
 * @param items Items to wrap
 * @return Immutable list
 */
fun getImmutableList(items: List<String>): List<String> {
    return items.toList() // Creates immutable copy
}

/**
 * Returns immutable map.
 * 
 * @param map Map to wrap
 * @return Immutable map
 */
fun getImmutableMap(map: Map<String, String>): Map<String, String> {
    return map.toMap() // Creates immutable copy
}

// Use val for immutable variables
val users: List<User> = listOf(user1, user2) // Immutable
// users.add(user3) // Compile error - list is immutable
```

---

## 11. KDoc Documentation (MANDATORY)

### A. Complete API Documentation

**CRITICAL: All public APIs MUST have complete KDoc comments for auto-generated documentation.**

#### ✅ CORRECT - Complete KDoc

```kotlin
/**
 * Service for processing user data.
 * 
 * This service provides operations for:
 * - User creation and validation
 * - User data retrieval
 * - User status management
 * 
 * All operations are thread-safe and can be used concurrently.
 * 
 * @author John Doe
 * @since 1.0
 */
object UserService {
    
    /**
     * Creates a new user with the specified details.
     * 
     * Validates the user data before creation:
     * - Username must be unique
     * - Email must be valid format
     * - Password must meet security requirements
     * 
     * @param username User's login name (must be unique, not null)
     * @param email User's email address (must be valid format, not null)
     * @return Created user with generated ID
     * @throws IllegalArgumentException if validation fails
     * @throws DuplicateUserException if username already exists
     * @since 1.0
     */
    suspend fun createUser(username: String, email: String): User {
        // Implementation
    }
    
    /**
     * Finds user by ID.
     * 
     * @param id User ID (must be positive)
     * @return User if found, null otherwise
     * @throws IllegalArgumentException if id is not positive
     */
    suspend fun findById(id: Long): User? {
        // Implementation
    }
}
```

---

## 12. Testing Requirements (MANDATORY)

### A. Unit Testing (MANDATORY - ALWAYS REQUIRED)

**CRITICAL: All new/modified code MUST have unit tests. Unit tests MUST pass before code delivery. This is non-negotiable.**

**MANDATORY RULES:**
1. **Unit tests are ALWAYS required** for all new code
2. **Unit tests are ALWAYS required** for all modified code
3. **All unit tests MUST pass** before code delivery
4. **After ANY code change**, re-run tests to verify they still pass
5. **Minimum 80% code coverage** for business logic

#### ✅ CORRECT - Kotest Tests

```kotlin
import io.kotest.core.spec.style.StringSpec
import io.kotest.matchers.shouldBe
import io.kotest.matchers.shouldNotBe
import io.kotest.matchers.throwable.shouldThrow
import kotlinx.coroutines.test.runTest

/**
 * Tests for UserService.
 */
class UserServiceTest : StringSpec({
    
    "createUser with valid data should return user" {
        runTest {
            // Given
            val username = "testuser"
            val email = "test@example.com"
            
            // When
            val result = createUser(username, email, mockRepository)
            
            // Then
            result shouldNotBe null
            result.username shouldBe username
            result.email shouldBe email
            result.id shouldNotBe null
        }
    }
    
    "createUser with duplicate username should throw exception" {
        runTest {
            // Given
            createUser("user", "email@example.com", mockRepository)
            
            // When/Then
            shouldThrow<IllegalArgumentException> {
                createUser("user", "other@example.com", mockRepository)
            }
        }
    }
    
    "findById with existing user should return user" {
        runTest {
            // Given
            val user = createUser("user", "email@example.com", mockRepository)
            
            // When
            val result = findUserById(user.id!!, mockRepository)
            
            // Then
            result shouldNotBe null
            result?.id shouldBe user.id
        }
    }
})
```

---

## 13. Performance Optimization

### A. Sequences for Large Datasets

**CRITICAL: Use sequences for lazy evaluation of large datasets.**

#### ✅ CORRECT - Sequences

```kotlin
/**
 * Processes large dataset using sequence (lazy evaluation).
 * 
 * @param data Large dataset
 * @return Processed results
 */
fun processLargeDataset(data: List<Data>): List<String> {
    return data.asSequence()
        .filter { it.isValid() }
        .map { it.transform() }
        .take(1000) // Limit results
        .toList()
}
```

### B. Inline Functions

**CRITICAL: Use inline functions for performance-critical code.**

#### ✅ CORRECT - Inline Functions

```kotlin
/**
 * Inline function for performance-critical operations.
 * 
 * @param block Operation to execute
 * @return Result of operation
 */
inline fun <T> measureTime(block: () -> T): T {
    val start = System.currentTimeMillis()
    return try {
        block()
    } finally {
        val duration = System.currentTimeMillis() - start
        println("Operation took $duration ms")
    }
}
```

---

## 14. Security & Dependency Management (MANDATORY)

### A. Automated Dependency Management

**Use Gradle with version catalogs and lockfiles for automated management:**

```kotlin
// gradle/libs.versions.toml
[versions]
kotlin = "2.0.20"
coroutines = "1.9.0"

[libraries]
kotlinx-coroutines-core = { group = "org.jetbrains.kotlinx", name = "kotlinx-coroutines-core", version.ref = "coroutines" }
```

- **Lockfiles**: Enable Gradle dependency locking (`dependencyLocking { lockAllConfigurations() }`) to ensure reproducible and secure builds.
- **BOMs**: Use Kotlin and library-specific BOMs to maintain version consistency.

### B. Vulnerability Scanning & Security

**Mandatory security checks for ALL Kotlin projects:**

1. **Vulnerability Scan**:
   ```bash
   # Scan for CVEs in dependencies
   ./gradlew dependencyCheckAnalyze
   ```
   - Agents MUST ensure 0 HIGH or CRITICAL vulnerabilities remain.

2. **Supply Chain Audit**:
   - Verify artifact checksums and signatures.
   - Use `checksum-dependency-check` to verify no tampered JARs are used.

### C. Dependency File

```kotlin
// build.gradle.kts example
dependencies {
    implementation(libs.kotlinx.coroutines.core)
    testImplementation(kotlin("test"))
}
```

---

## 15. Deployment Checklist

### Agent-Generated Code Verification (MANDATORY)

#### Build & Compilation
- [ ] Code compiles: `./gradlew compileKotlin` returns exit code 0
- [ ] No compilation errors or warnings (Werror=true)
- [ ] Kotlin 2.0 features used (K2 compiler enabled)
- [ ] Code formatted: `./gradlew ktlintCheck` passes

#### Testing
- [ ] All tests pass: `./gradlew test` returns exit code 0
- [ ] Reasonable coverage: `jacocoTestReport` shows >80%
- [ ] Integration tests pass (using Kotest/Mockk)

#### Security
- [ ] Dependency scan passes: 0 vulnerabilities found
- [ ] Supply chain verified: Lockfiles in sync
- [ ] Secrets check: No hardcoded secrets in code or resources
- [ ] Static analysis: `detekt` passes with 0 issues

#### Code Quality
- [ ] No unused dependencies or imports
- [ ] Data classes used for all DTOs and models
- [ ] Small, focused top-level functions instead of unnecessary classes

#### Documentation
- [ ] All public APIs have KDoc comments
- [ ] Documentation follows conventions
- [ ] Examples provided for complex APIs

#### Architecture
- [ ] Hexagonal architecture followed (Ports and Adapters)
- [ ] Dependency injection used for all components
- [ ] Coroutines used for all async/I/O operations

#### Agent Workflow Completed
- [ ] Agent verified code compiles/builds successfully
- [ ] Agent ran all tests and verified they pass
- [ ] Agent ran security scans and verified 0 high vulnerabilities
- [ ] Agent verified documentation and KDoc

---

## 16. Why This Configuration Works

**Kotlin 2.0 K2 Compiler**:
- Provides massive performance improvements in compilation speed and smarter analysis, catching more potential bugs at compile-time.

**Smart Casts 2.0**:
- Significantly reduces boilerplate by allowing the compiler to track types through more complex logic flows, making code safer and more readable.

**Coroutines with Cancellation**:
- Native support for structured concurrency ensures that resources are always cleaned up, and no "ghost" tasks continue running after a scope is closed.

---

## 17. Quick Reference

### Common Commands

```bash
# Build
./gradlew build

# Test with coverage
./gradlew test jacocoTestReport

# Security scan
./gradlew dependencyCheckAnalyze

# Lint and Format
./gradlew ktlintCheck detekt ktlintFormat

# Run
./gradlew run
```

### Modern Kotlin Patterns Cheat Sheet

```kotlin
// Smart Casts 2.0 (Kotlin 2.0)
fun handle(x: Any) {
    if (x is String || x is Int) {
        // x is smart-cast to Any in 1.9, but K2 handles more cases
    }
}

// Guard Conditions
when (status) {
    is Status.Active if status.priority > 10 -> handleHighPriority()
    is Status.Active -> handleNormal()
}

// Native Multi-Error Joining
val err = errors.join() // Collective error handling patterns

// Value Classes
@JvmInline value class UserId(val id: String)
```

---

**Last Updated:** 2026-02-06
**Version:** 1.1
**Maintainer:** Kotlin Team


**End of Modern Kotlin Development Guidelines**
