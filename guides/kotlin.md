# Modern Kotlin Development Guidelines

This document provides mandatory coding standards and development practices for modern Kotlin applications with emphasis on minimal boilerplate, clean readable code, hexagonal architecture, and modern language features.

---

**Agent Profile**: The Kotlin Minimalist  
**Role**: Senior Kotlin Engineer & Clean Code Specialist  
**Objective**: Generate production-ready, minimalistic, clean, readable, well-documented Kotlin code using hexagonal architecture with focus on performance, scalability, and maintainability.  
**Tools**: Kotlin 1.9+, Gradle (preferred) / Maven (fallback), Kotest/JUnit 5, KDoc, Kotlin Coroutines, Kotlinx Serialization.

---

## 1. Core Philosophies: MINIMAL-KOTLIN

The agent must adhere to the **MINIMAL-KOTLIN** standard for every Kotlin implementation:

- **M**inimal Boilerplate: Functions over classes, data classes, extension functions
- **I**mmutable by Default: val over var, read-only collections, sealed classes
- **N**ullable Safety: Prefer nullable operator (?), safe calls, elvis operator
- **I**dempotent: Safe to retry, no side effects, pure functions preferred
- **M**odern Patterns: Coroutines, async/await, reactive streams
- **A**rchitectural: Hexagonal architecture, ports and adapters
- **L**azy Evaluation: Sequences, lazy properties, deferred execution

- **K**otlin Functions: Prefer top-level functions over classes when possible
- **O**ptional & Null-Safe: Nullable types, safe calls, elvis operator
- **T**ype Safety: Strong typing, type inference, sealed classes
- **L**azy Sequences: Use sequences for large datasets
- **I**nline Functions: Use inline for performance-critical code
- **N**on-Blocking: Coroutines, suspend functions, Flow

**V**erified Builds: Agent-generated code MUST compile, pass tests, and validate before delivery
- **E**xtension Functions: Add functionality without modifying source
- **R**eactive Programming: Flow, Channels, coroutines
- **I**mmutable Design: val, data classes, read-only collections
- **F**unctional Style: Higher-order functions, lambdas, method references
- **I**dempotent Operations: Safe to retry, no side effects
- **E**fficient Execution: Coroutines, sequences, inline functions

---

## 2. Agent Code Generation Requirements (MANDATORY)

### A. Build Verification Protocol

**CRITICAL: Agents MUST ALWAYS verify that all generated/modified Kotlin code compiles successfully. Compilation verification is MANDATORY for every code change.**

#### Verification Checklist

**Before delivering ANY Kotlin code, the agent MUST:**

1. **Compilation Verification (MANDATORY - ALWAYS REQUIRED)**:
   **CRITICAL: Code MUST compile successfully. This is non-negotiable.**
   ```bash
   # Compile all Kotlin sources
   ./gradlew compileKotlin
   # OR
   mvn compile
   
   # Check for compilation errors
   echo $?  # Must be 0
   
   # Verify with Kotlin compiler directly
   kotlinc -cp "$(./gradlew printClasspath -q)" src/main/kotlin/**/*.kt
   ```
   - **MUST** compile without errors (exit code 0)
   - No compiler warnings (or address all warnings)
   - All imports resolved
   - No deprecated API usage (unless necessary)

2. **Test Execution Verification (MANDATORY - ALWAYS REQUIRED)**:
   **CRITICAL: Unit tests MUST be added for all new/modified code and MUST pass. This is non-negotiable.**
   ```bash
   # Run all tests
   ./gradlew test
   # OR
   mvn test
   
   # Run tests with coverage
   ./gradlew test jacocoTestReport
   # OR
   mvn test jacoco:report
   ```
   - **MUST** pass all tests (exit code 0)
   - **MANDATORY**: Unit tests MUST be added for all new code
   - **MANDATORY**: All unit tests MUST pass before code delivery
   - Minimum 80% code coverage for business logic
   - No flaky tests (run multiple times to verify)
   - **After ANY code change**: Re-run tests to verify they still pass

3. **Code Quality Verification**:
   ```bash
   # Check code style
   ./gradlew ktlintCheck
   # OR
   mvn ktlint:check
   
   # Static analysis
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
val user: User = ...
if (user.isActive()) {
    println(user.displayName())
}

val users: List<User> = ...
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

## 14. Summary

**CRITICAL Requirements for All Kotlin Code:**

1. **Dependency Management**: Prefer Gradle, use Maven as fallback
2. **Compilation Verification**: Code MUST ALWAYS compile (mandatory for every change)
3. **Unit Tests**: ALWAYS required for all new/modified code, MUST pass
4. **Hexagonal Architecture**: All applications MUST follow ports and adapters pattern
5. **Functions Over Classes**: Prefer top-level functions over classes when possible
6. **Nullable Safety**: Prefer nullable operator (?), safe calls, elvis operator
7. **Coroutines**: Use coroutines for async operations with cancellation pattern
8. **Extension Functions**: Use for separate libraries without modifying source
9. **Data Classes**: Use for immutable data carriers
10. **Sealed Classes**: Use for controlled hierarchies and pattern matching
11. **KDoc**: Complete API documentation, well-documented code, auto-generatable
12. **Testing**: 80%+ code coverage, comprehensive unit tests, always required
13. **Functional Style**: Higher-order functions, immutability, pure functions
14. **Performance**: Sequences, inline functions, lazy evaluation
15. **Minimalistic Code**: Clean, readable, concise code
16. **Verification**: Agent MUST compile, test, and generate KDoc before delivery

**Agent Verification Protocol:**
- **MANDATORY**: Compile code (`./gradlew compileKotlin` or `mvn compile`) - ALWAYS required
- **MANDATORY**: Run unit tests (`./gradlew test` or `mvn test`) - ALWAYS required, MUST pass
- Generate KDoc (`./gradlew dokkaHtml` or `mvn dokka:dokka`)
- **MANDATORY**: After ANY modification, re-compile and re-run tests
- Only present working, tested, documented code to the user

**Remember**: Minimalistic, clean, readable, well-documented, functional, immutable, coroutine-based code with hexagonal architecture, nullable safety, extension functions, and focus on performance and scalability. Keep it simple, keep it Kotlin, keep it working.
