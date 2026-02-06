# Modern Java Development Guidelines
Mandatory coding standards and development practices for modern Java applications with emphasis on performance, portability, minimalistic code, and modern language features. Java 21+, Gradle (preferred) / Maven (fallback), JUnit 5, JavaDoc, Project Reactor, Virtual Threads.

---

**Agent Profile**: The Java Modernist  
**Role**: Senior Java Engineer & Performance Specialist  
**Objective**: Generate production-ready, minimalistic, clean, readable, well-documented Java code using hexagonal architecture with focus on performance, scalability, and memory footprint.  
**Tools**: Java 21+, Gradle (preferred) / Maven (fallback), JUnit 5, JavaDoc, Project Reactor, Virtual Threads.

---

## 1. Core Philosophies: JAVA-FIRST

The agent must adhere to the **JAVA-FIRST** principles for every Java implementation:

- **Test-Driven Development (TDD)**: ALWAYS write tests BEFORE implementation (Red-Green-Refactor cycle mandatory)
- **Regression Shield**: EVERY bug discovered MUST receive a test BEFORE fixing to prevent regression
- **M**inimalistic Code: Less verbose, concise, clean, readable, expressive code
- **O**bject-Oriented Modern: Records, sealed classes, pattern matching
- **D**ata-Centric: Immutable data carriers, records over POJOs
- **E**xpressiveness: Functional programming, streams, lambdas
- **R**eactive & Async: Futures, coroutines, async/await patterns
- **N**on-Blocking: Virtual threads, reactive streams, Project Reactor

- **J**avaDoc Documentation: Complete API documentation, well-documented code, auto-generatable
- **A**sync-First: Prefer async patterns when applicable
- **V**irtual Threads: Use virtual threads for concurrent operations
- **A**rchitectural Patterns: Hexagonal architecture, Repository, Facade, Decorator, Strategy
- **P**erformance: Optimize for speed, memory efficiency, scalability, minimal memory footprint
- **O**ptional & Null-Safe: Use Optional, null-safety patterns
- **R**ecords & Sealed: Immutable data, controlled hierarchies
- **T**esting: Comprehensive unit tests, always required, 80%+ coverage, must pass

**V**erified Builds: Agent-generated code MUST ALWAYS compile, pass unit tests, and validate before delivery
- **E**rror Handling: Explicit error handling, Result types where applicable
- **R**eactive Patterns: Project Reactor, Flow API for event handling
- **I**mmutable Design: Records, final fields, defensive copying
- **F**unctional Style: Streams, lambdas, method references
- **I**dempotent Operations: Safe to retry, no side effects
- **E**fficient Execution: Virtual threads, parallel streams, async I/O

---

## 2. Agent Code Generation Requirements (MANDATORY)

### A. Build Verification Protocol

**CRITICAL: Agents MUST ALWAYS verify that all generated/modified Java code compiles successfully. Compilation verification is MANDATORY for every code change.**

#### Verification Checklist

**Before delivering ANY Java code, the agent MUST:**

1. **Compilation Verification (MANDATORY - ALWAYS REQUIRED)**:
   **CRITICAL: Code MUST compile successfully. This is non-negotiable.**
   ```bash
   # Compile all Java sources
   mvn compile
   # OR
   ./gradlew compileJava
   
   # Check for compilation errors
   echo $?  # Must be 0
   
   # Verify with Java compiler directly
   javac -cp "$(mvn dependency:build-classpath -q -Dmdep.outputFile=/dev/stdout)" src/main/java/**/*.java
   ```
   - **MUST** compile without errors (exit code 0)
   - No compiler warnings (or address all warnings)
   - All imports resolved
   - No deprecated API usage (unless necessary)

2. **Test Execution Verification**:
   ```bash
   # Run all tests
   mvn test
   # OR
   ./gradlew test
   
   # Run tests with coverage
   mvn test jacoco:report
   # OR
   ./gradlew test jacocoTestReport
   ```
   - **MUST** pass all tests (exit code 0)
   - Minimum 80% code coverage for business logic
   - No flaky tests (run multiple times to verify)

3. **Code Quality Verification**:
   ```bash
   # Check code style
   mvn checkstyle:check
   # OR
   ./gradlew checkstyleMain
   
   # Static analysis
   mvn spotbugs:check
   # OR
   ./gradlew spotbugsMain
   ```
   - **MUST** pass code style checks
   - No static analysis issues
   - Follows project coding standards

4. **JavaDoc Generation**:
   ```bash
   # Generate JavaDoc
   mvn javadoc:javadoc
   # OR
   ./gradlew javadoc
   ```
   - **MUST** generate without errors
   - All public APIs documented
   - No missing JavaDoc warnings

5. **Post-Modification Verification (MANDATORY)**:
   ```bash
   # After ANY modification, ALWAYS run:
   # 1. Compile
   mvn compile
   # Exit code MUST be 0
   
   # 2. Run tests
   mvn test
   # Exit code MUST be 0
   
   # 3. Generate JavaDoc
   mvn javadoc:javadoc
   # Exit code MUST be 0
   ```

### B. Error Correction Process

If verification fails:

1. **Read the error message** - compilation errors, test failures, static analysis issues
2. **Identify the root cause** - syntax error, missing import, test logic issue, missing JavaDoc
3. **Fix the issue** in the generated code
4. **Re-run verification** until all checks pass
5. **Document fixes** in comments if non-obvious
6. **Only present working, tested code** to the user

### C. Example Verification Workflow

```bash
# Agent must simulate/verify this workflow

# 1. Compile
mvn compile
# Should succeed (exit code 0)

# 2. Run tests
mvn test
# Should pass all tests

# 3. Check code quality
mvn checkstyle:check spotbugs:check
# Should pass all checks

# 4. Generate JavaDoc
mvn javadoc:javadoc
# Should generate without errors

# If any step fails:
# - Read the error output
# - Fix the code
# - Try again
# - Repeat until success
```

**CRITICAL**: Never provide Java code to the user that doesn't compile or pass tests. Always verify first, fix issues, then present the working solution.

**MANDATORY RULES:**
1. **Compilation is ALWAYS required** - Code MUST compile successfully
2. **Unit tests are ALWAYS required** - All new/modified code MUST have unit tests
3. **Tests MUST pass** - All unit tests MUST pass before code delivery
4. **Re-verify after changes** - After ANY code modification, re-compile and re-run tests
5. **TDD is MANDATORY** - Write tests BEFORE implementation (Red-Green-Refactor)
6. **Bug regression tests MANDATORY** - Every bug MUST get a test BEFORE fixing

---

## 2A. Test-Driven Development (TDD) Protocol (MANDATORY)

**CRITICAL: Follow the Red-Green-Refactor cycle for ALL new Java code.**

### TDD Cycle for Java

```
1. 🔴 RED: Write a failing test first
   ↓
2. 🟢 GREEN: Write minimal code to make it pass
   ↓
3. 🔵 REFACTOR: Improve code while keeping tests green
   ↓
   Repeat
```

### Example TDD Workflow for Java Method

```java
// Step 1: RED - Write failing test first
// src/test/java/com/example/util/EmailValidatorTest.java
package com.example.util;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class EmailValidatorTest {
    // Test will fail - method doesn't exist yet
    @Test
    void acceptsValidEmails() {
        assertTrue(EmailValidator.isValid("user@example.com"));
        assertTrue(EmailValidator.isValid("test.user@domain.co.uk"));
    }

    @Test
    void rejectsInvalidEmails() {
        assertFalse(EmailValidator.isValid("invalid"));
        assertFalse(EmailValidator.isValid("user@"));
        assertFalse(EmailValidator.isValid("@domain.com"));
    }

    @Test
    void rejectsEmptyStrings() {
        assertFalse(EmailValidator.isValid(""));
        assertFalse(EmailValidator.isValid(null));
    }
}

// Run: mvn test or ./gradlew test
// ❌ FAILS - EmailValidator doesn't exist yet

// Step 2: GREEN - Write minimal implementation
// src/main/java/com/example/util/EmailValidator.java
package com.example.util;

import java.util.regex.Pattern;

/**
 * Validates email address formats.
 *
 * <p>Provides methods to check if a string conforms to a valid email address pattern.
 *
 * @since 1.0
 */
public final class EmailValidator {
    private static final Pattern EMAIL_PATTERN = 
        Pattern.compile("^[^\\s@]+@[^\\s@]+\\.[^\\s@]+$");

    private EmailValidator() {
        throw new AssertionError("Utility class");
    }

    /**
     * Validates an email address format.
     *
     * @param email the email address to validate
     * @return {@code true} if the email is valid, {@code false} otherwise
     */
    public static boolean isValid(String email) {
        if (email == null || email.isEmpty()) {
            return false;
        }
        return EMAIL_PATTERN.matcher(email).matches();
    }
}

// Run: mvn test
// ✅ PASSES - tests pass

// Step 3: REFACTOR - Improve with more robust validation
/**
 * Validates email address formats according to RFC 5322.
 *
 * <p>Provides comprehensive email validation including:
 * <ul>
 *   <li>Basic format check (user@domain.tld)</li>
 *   <li>Length constraints (3-254 characters)</li>
 *   <li>RFC 5322 compliant pattern</li>
 * </ul>
 *
 * @since 1.0
 */
public final class EmailValidator {
    private static final Pattern EMAIL_PATTERN = Pattern.compile(
        "^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$"
    );
    private static final int MIN_LENGTH = 3;
    private static final int MAX_LENGTH = 254;

    private EmailValidator() {
        throw new AssertionError("Utility class");
    }

    /**
     * Validates an email address format.
     *
     * <p>Checks if the provided string conforms to a valid email address pattern
     * according to RFC 5322 specification.
     *
     * @param email the email address to validate
     * @return {@code true} if the email is valid, {@code false} otherwise
     * @see <a href="https://tools.ietf.org/html/rfc5322">RFC 5322</a>
     */
    public static boolean isValid(String email) {
        if (email == null || email.isEmpty()) {
            return false;
        }

        if (email.length() < MIN_LENGTH || email.length() > MAX_LENGTH) {
            return false;
        }

        return EMAIL_PATTERN.matcher(email).matches();
    }
}
// Tests still pass ✓
```

### Example TDD for Java Record

```java
// Step 1: RED - Write failing test first
// src/test/java/com/example/model/UserTest.java
package com.example.model;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class UserTest {
    // Test will fail - User record doesn't exist yet
    @Test
    void createsUserWithValidData() {
        var user = new User("user-123", "John Doe", "john@example.com");
        
        assertEquals("user-123", user.id());
        assertEquals("John Doe", user.name());
        assertEquals("john@example.com", user.email());
    }

    @Test
    void throwsOnNullId() {
        assertThrows(NullPointerException.class, () -> 
            new User(null, "John", "john@example.com")
        );
    }

    @Test
    void throwsOnInvalidEmail() {
        assertThrows(IllegalArgumentException.class, () -> 
            new User("user-123", "John", "invalid-email")
        );
    }
}

// Run: mvn test
// ❌ FAILS - User record doesn't exist yet

// Step 2: GREEN - Write minimal implementation
// src/main/java/com/example/model/User.java
package com.example.model;

import java.util.Objects;

/**
 * Represents a user in the system.
 *
 * @param id    the unique user identifier
 * @param name  the user's full name
 * @param email the user's email address
 */
public record User(String id, String name, String email) {
    public User {
        Objects.requireNonNull(id, "id cannot be null");
        Objects.requireNonNull(name, "name cannot be null");
        Objects.requireNonNull(email, "email cannot be null");
        
        if (!email.matches("^[^\\s@]+@[^\\s@]+\\.[^\\s@]+$")) {
            throw new IllegalArgumentException("Invalid email format");
        }
    }
}

// Run: mvn test
// ✅ PASSES - tests pass

// Step 3: REFACTOR - Add validation and factory methods
/**
 * Represents an immutable user in the system.
 *
 * <p>This record enforces validation rules:
 * <ul>
 *   <li>ID must not be null or empty</li>
 *   <li>Name must not be null or empty</li>
 *   <li>Email must be valid format</li>
 * </ul>
 *
 * @param id    the unique user identifier (non-null, non-empty)
 * @param name  the user's full name (non-null, non-empty)
 * @param email the user's email address (non-null, valid format)
 * @since 1.0
 */
public record User(String id, String name, String email) {
    private static final Pattern EMAIL_PATTERN = Pattern.compile(
        "^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$"
    );

    /**
     * Compact constructor with validation.
     *
     * @throws NullPointerException     if any parameter is null
     * @throws IllegalArgumentException if email format is invalid or any field is empty
     */
    public User {
        Objects.requireNonNull(id, "id cannot be null");
        Objects.requireNonNull(name, "name cannot be null");
        Objects.requireNonNull(email, "email cannot be null");

        if (id.isBlank()) {
            throw new IllegalArgumentException("id cannot be empty");
        }
        if (name.isBlank()) {
            throw new IllegalArgumentException("name cannot be empty");
        }
        if (!EMAIL_PATTERN.matcher(email).matches()) {
            throw new IllegalArgumentException("Invalid email format: " + email);
        }
    }

    /**
     * Creates a new user with updated name.
     *
     * @param newName the new name
     * @return a new User instance with the updated name
     */
    public User withName(String newName) {
        return new User(id, newName, email);
    }
}
// Tests still pass ✓
```

---

## 2B. Bug Fix Protocol for Java (MANDATORY)

**CRITICAL: Every Java bug MUST receive a regression test BEFORE fixing.**

### Bug Fix Workflow for Java

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

```java
// Bug Report #7821: NullPointerException in getUserName when user is null

// Step 1-2: Write test that reproduces the bug
// src/test/java/com/example/service/UserServiceTest.java
package com.example.service;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class UserServiceTest {
    /**
     * Bug #7821: getUserName throws NullPointerException when user is null.
     * Discovered: 2026-01-18
     * This test prevents regression.
     */
    @Test
    void getUserName_returnsEmptyOptional_whenUserIsNull_Bug7821() {
        var service = new UserService();
        
        // Should return empty Optional, not throw NPE
        var result = service.getUserName(null);
        
        assertTrue(result.isEmpty());
    }

    @Test
    void getUserName_returnsName_whenUserExists() {
        var service = new UserService();
        var user = new User("123", "John Doe", "john@example.com");
        
        var result = service.getUserName(user);
        
        assertTrue(result.isPresent());
        assertEquals("John Doe", result.get());
    }
}

// Run: mvn test
// ❌ FAILS - NullPointerException thrown

// Step 3: Fix the bug
// src/main/java/com/example/service/UserService.java
package com.example.service;

import java.util.Optional;

/**
 * Service for user-related operations.
 *
 * @since 1.0
 */
public class UserService {
    /**
     * Gets the user's name.
     *
     * <p><b>Bug Fix #7821:</b> Now properly handles null users by returning
     * empty Optional instead of throwing NullPointerException.
     *
     * @param user the user (may be null)
     * @return an Optional containing the user's name, or empty if user is null
     */
    public Optional<String> getUserName(User user) {
        // FIX: Check for null before accessing user
        return Optional.ofNullable(user)
                       .map(User::name);
    }
}

// Run: mvn test
// ✅ PASSES - bug fixed, regression prevented ✓
```

### Example Bug Fix: Concurrent Modification

```java
// Bug Report #7822: ConcurrentModificationException in removeInactiveUsers

// Step 1-2: Write test that reproduces the bug
package com.example.service;

import org.junit.jupiter.api.Test;
import java.util.*;
import static org.junit.jupiter.api.Assertions.*;

class UserManagerTest {
    /**
     * Bug #7822: ConcurrentModificationException when removing inactive users.
     * Discovered: 2026-01-18
     * This test prevents regression.
     */
    @Test
    void removeInactiveUsers_doesNotThrowConcurrentModification_Bug7822() {
        var manager = new UserManager();
        
        // Add multiple users
        manager.addUser(new User("1", "John", "john@example.com", false));
        manager.addUser(new User("2", "Jane", "jane@example.com", true));
        manager.addUser(new User("3", "Bob", "bob@example.com", false));
        
        // Should not throw ConcurrentModificationException
        assertDoesNotThrow(() -> manager.removeInactiveUsers());
        
        // Should only have active users left
        assertEquals(1, manager.getUserCount());
    }
}

// Run: mvn test
// ❌ FAILS - ConcurrentModificationException thrown

// Step 3: Fix the bug
package com.example.service;

import java.util.*;

/**
 * Manages a collection of users.
 *
 * @since 1.0
 */
public class UserManager {
    private final List<User> users = new ArrayList<>();

    public void addUser(User user) {
        users.add(user);
    }

    /**
     * Removes all inactive users from the collection.
     *
     * <p><b>Bug Fix #7822:</b> Now uses iterator for safe removal during
     * iteration, preventing ConcurrentModificationException.
     */
    public void removeInactiveUsers() {
        // FIX: Use iterator for safe removal
        // OLD (buggy) code:
        // for (User user : users) {
        //     if (!user.isActive()) {
        //         users.remove(user);  // ConcurrentModificationException!
        //     }
        // }
        
        // NEW (fixed) code:
        users.removeIf(user -> !user.isActive());
    }

    public int getUserCount() {
        return users.size();
    }
}

// Run: mvn test
// ✅ PASSES - bug fixed, regression prevented ✓
```

### Prohibited Practices for Java Bug Fixes

**NEVER:**
- ❌ Fix a bug without adding a regression test first
- ❌ Write implementation before writing tests (violates TDD)
- ❌ Skip the Red-Green-Refactor cycle
- ❌ Commit code with failing tests
- ❌ Remove tests to make code pass
- ❌ Use `@Disabled` to ignore failing tests
- ❌ Suppress warnings instead of fixing root cause

**ALWAYS:**
- ✅ Write a test that reproduces the bug first
- ✅ Verify the test fails before fixing
- ✅ Document bug ID in test JavaDoc
- ✅ Run `mvn verify` or `./gradlew check` after fix
- ✅ Ensure fix doesn't introduce new issues
- ✅ Keep tests in codebase permanently
- ✅ Test with different JVM versions if applicable

---

## 3. Dependency and Package Management (MANDATORY)

### A. Build Tool Preference

**CRITICAL: Prefer Gradle for dependency management. Use Maven only if Gradle is unavailable.**

#### Priority Order

1. **Gradle (PREFERRED)**: Use Gradle for new projects or when available
2. **Maven (FALLBACK)**: Use Maven only if Gradle is not available or project already uses Maven

#### ✅ CORRECT - Gradle Configuration

```gradle
// build.gradle.kts - Modern Gradle with Kotlin DSL

plugins {
    java
    `java-library`
    jacoco
    checkstyle
}

java {
    toolchain {
        languageVersion = JavaLanguageVersion.of(21)
    }
}

repositories {
    mavenCentral()
}

dependencies {
    // Core dependencies
    implementation("org.springframework.boot:spring-boot-starter-web:3.2.0")
    implementation("org.springframework.boot:spring-boot-starter-data-jpa:3.2.0")
    
    // Reactive
    implementation("io.projectreactor:reactor-core:3.6.0")
    
    // Testing
    testImplementation("org.springframework.boot:spring-boot-starter-test:3.2.0")
    testImplementation("org.junit.jupiter:junit-jupiter:5.10.0")
    testRuntimeOnly("org.junit.platform:junit-platform-launcher")
}

tasks.test {
    useJUnitPlatform()
    finalizedBy(tasks.jacocoTestReport)
}

tasks.jacocoTestReport {
    dependsOn(tasks.test)
    reports {
        xml.required = true
        html.required = true
    }
}

tasks.compileJava {
    options.compilerArgs.addAll(listOf(
        "-parameters",
        "-Xlint:all",
        "-Werror"
    ))
}
```

#### ✅ CORRECT - Maven Configuration (Fallback)

```xml
<!-- pom.xml - Maven configuration when Gradle unavailable -->

<project>
    <modelVersion>4.0.0</modelVersion>
    
    <groupId>com.example</groupId>
    <artifactId>myproject</artifactId>
    <version>1.0.0</version>
    
    <properties>
        <java.version>21</java.version>
        <maven.compiler.source>21</maven.compiler.source>
        <maven.compiler.target>21</maven.compiler.target>
        <project.build.sourceEncoding>UTF-8</project.build.sourceEncoding>
        
        <spring-boot.version>3.2.0</spring-boot.version>
        <junit.version>5.10.0</junit.version>
    </properties>
    
    <dependencies>
        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter-web</artifactId>
            <version>${spring-boot.version}</version>
        </dependency>
        
        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter-test</artifactId>
            <version>${spring-boot.version}</version>
            <scope>test</scope>
        </dependency>
    </dependencies>
    
    <build>
        <plugins>
            <plugin>
                <groupId>org.apache.maven.plugins</groupId>
                <artifactId>maven-compiler-plugin</artifactId>
                <version>3.11.0</version>
                <configuration>
                    <source>21</source>
                    <target>21</target>
                    <parameters>true</parameters>
                </configuration>
            </plugin>
            
            <plugin>
                <groupId>org.apache.maven.plugins</groupId>
                <artifactId>maven-surefire-plugin</artifactId>
                <version>3.2.0</version>
            </plugin>
        </plugins>
    </build>
</project>
```

### B. Dependency Management Best Practices

**CRITICAL: Manage dependencies efficiently for performance and build speed.**

#### ✅ CORRECT - Efficient Dependency Management

```gradle
// build.gradle.kts - Performance-optimized

configurations {
    // Use implementation instead of compile (better performance)
    implementation {
        resolutionStrategy {
            // Fail on version conflicts
            failOnVersionConflict()
            // Cache dependencies
            cacheDynamicVersionsFor(10, "minutes")
            cacheChangingModulesFor(0, "seconds")
        }
    }
}

dependencies {
    // Use specific versions (not dynamic) for reproducibility
    implementation("org.springframework.boot:spring-boot-starter-web:3.2.0")
    
    // Exclude transitive dependencies when not needed
    implementation("com.example:library:1.0.0") {
        exclude(group = "org.slf4j", module = "slf4j-api")
    }
    
    // Use BOM for version management
    implementation(platform("org.springframework.boot:spring-boot-dependencies:3.2.0"))
}
```

### C. Build Performance

**CRITICAL: Optimize build performance for efficiency.**

```gradle
// settings.gradle.kts - Build performance optimization

// Enable build cache
buildCache {
    local {
        enabled = true
    }
}

// Parallel execution
org.gradle.parallel = true
org.gradle.caching = true
org.gradle.configureondemand = true
```

---

## 4. Hexagonal Architecture (MANDATORY)

### A. Architecture Principles

**CRITICAL: All applications MUST follow hexagonal architecture (ports and adapters) for clean separation of concerns, testability, and maintainability.**

#### Core Concepts

1. **Domain Core**: Business logic in the center, framework-independent
2. **Ports**: Interfaces defining application boundaries (input/output)
3. **Adapters**: Implementations connecting to external systems
4. **Dependency Inversion**: Dependencies point inward toward domain

#### ✅ CORRECT - Hexagonal Architecture Structure

```
project/
├── src/
│   ├── main/
│   │   └── java/
│   │       └── com/
│   │           └── example/
│   │               ├── domain/              # Domain core (business logic)
│   │               │   ├── model/           # Domain entities/records
│   │               │   ├── service/         # Domain services
│   │               │   └── port/            # Port interfaces
│   │               │       ├── in/           # Input ports (use cases)
│   │               │       └── out/         # Output ports (repositories)
│   │               ├── application/         # Application layer
│   │               │   └── service/        # Application services
│   │               └── adapter/             # Adapters
│   │                   ├── in/              # Input adapters (REST, CLI)
│   │                   │   ├── web/         # Web controllers
│   │                   │   └── cli/         # CLI handlers
│   │                   └── out/             # Output adapters
│   │                       ├── persistence/ # Database adapters
│   │                       └── external/    # External API adapters
│   └── test/
│       └── java/
│           └── com/
│               └── example/
│                   ├── domain/
│                   ├── application/
│                   └── adapter/
```

#### ✅ CORRECT - Domain Port (Interface)

```java
package com.example.domain.port.in;

import com.example.domain.model.User;

import java.util.Optional;
import java.util.List;

/**
 * Input port for user management use cases.
 * Defines what the application can do (use cases).
 */
public interface UserManagementPort {
    
    /**
     * Creates a new user.
     * 
     * @param username Username (must be unique)
     * @param email Email address (must be valid)
     * @return Created user
     * @throws IllegalArgumentException if validation fails
     */
    User createUser(String username, String email);
    
    /**
     * Finds user by ID.
     * 
     * @param id User ID
     * @return Optional user if found
     */
    Optional<User> findById(Long id);
    
    /**
     * Finds all active users.
     * 
     * @return List of active users
     */
    List<User> findActiveUsers();
}
```

#### ✅ CORRECT - Domain Output Port (Repository Interface)

```java
package com.example.domain.port.out;

import com.example.domain.model.User;

import java.util.Optional;
import java.util.List;

/**
 * Output port for user persistence.
 * Defines what the application needs from persistence layer.
 */
public interface UserRepositoryPort {
    
    /**
     * Saves a user.
     * 
     * @param user User to save
     * @return Saved user with generated ID
     */
    User save(User user);
    
    /**
     * Finds user by ID.
     * 
     * @param id User ID
     * @return Optional user if found
     */
    Optional<User> findById(Long id);
    
    /**
     * Finds user by username.
     * 
     * @param username Username
     * @return Optional user if found
     */
    Optional<User> findByUsername(String username);
    
    /**
     * Finds all active users.
     * 
     * @return List of active users
     */
    List<User> findActiveUsers();
}
```

#### ✅ CORRECT - Application Service (Use Case Implementation)

```java
package com.example.application.service;

import com.example.domain.model.User;
import com.example.domain.port.in.UserManagementPort;
import com.example.domain.port.out.UserRepositoryPort;

import java.util.List;
import java.util.Optional;

/**
 * Application service implementing user management use cases.
 * Orchestrates domain logic and coordinates with ports.
 */
public class UserManagementService implements UserManagementPort {
    
    private final UserRepositoryPort userRepository;
    
    public UserManagementService(UserRepositoryPort userRepository) {
        this.userRepository = userRepository;
    }
    
    @Override
    public User createUser(String username, String email) {
        // Business logic validation
        if (username == null || username.isBlank()) {
            throw new IllegalArgumentException("Username cannot be empty");
        }
        if (email == null || !email.contains("@")) {
            throw new IllegalArgumentException("Invalid email format");
        }
        
        // Check if username exists
        if (userRepository.findByUsername(username).isPresent()) {
            throw new IllegalArgumentException("Username already exists");
        }
        
        // Create domain entity
        User user = new User(null, username, email, java.time.Instant.now());
        
        // Persist through port
        return userRepository.save(user);
    }
    
    @Override
    public Optional<User> findById(Long id) {
        return userRepository.findById(id);
    }
    
    @Override
    public List<User> findActiveUsers() {
        return userRepository.findActiveUsers();
    }
}
```

#### ✅ CORRECT - Input Adapter (REST Controller)

```java
package com.example.adapter.in.web;

import com.example.domain.port.in.UserManagementPort;
import com.example.domain.model.User;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;

/**
 * REST adapter for user management.
 * Maps HTTP requests to use cases.
 */
@RestController
@RequestMapping("/api/users")
public class UserController {
    
    private final UserManagementPort userManagement;
    
    public UserController(UserManagementPort userManagement) {
        this.userManagement = userManagement;
    }
    
    @PostMapping
    public ResponseEntity<User> createUser(@RequestBody CreateUserRequest request) {
        User user = userManagement.createUser(request.username(), request.email());
        return ResponseEntity.status(HttpStatus.CREATED).body(user);
    }
    
    @GetMapping("/{id}")
    public ResponseEntity<User> getUser(@PathVariable Long id) {
        return userManagement.findById(id)
            .map(ResponseEntity::ok)
            .orElse(ResponseEntity.notFound().build());
    }
    
    @GetMapping("/active")
    public List<User> getActiveUsers() {
        return userManagement.findActiveUsers();
    }
}
```

#### ✅ CORRECT - Output Adapter (Persistence)

```java
package com.example.adapter.out.persistence;

import com.example.domain.model.User;
import com.example.domain.port.out.UserRepositoryPort;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Optional;

/**
 * JPA adapter implementing user repository port.
 * Maps domain entities to database entities.
 */
@Repository
public class UserJpaAdapter implements UserRepositoryPort {
    
    private final UserJpaRepository jpaRepository;
    private final UserEntityMapper mapper;
    
    public UserJpaAdapter(UserJpaRepository jpaRepository, UserEntityMapper mapper) {
        this.jpaRepository = jpaRepository;
        this.mapper = mapper;
    }
    
    @Override
    public User save(User user) {
        UserEntity entity = mapper.toEntity(user);
        UserEntity saved = jpaRepository.save(entity);
        return mapper.toDomain(saved);
    }
    
    @Override
    public Optional<User> findById(Long id) {
        return jpaRepository.findById(id)
            .map(mapper::toDomain);
    }
    
    @Override
    public Optional<User> findByUsername(String username) {
        return jpaRepository.findByUsername(username)
            .map(mapper::toDomain);
    }
    
    @Override
    public List<User> findActiveUsers() {
        return jpaRepository.findByActiveTrue().stream()
            .map(mapper::toDomain)
            .toList();
    }
}
```

#### ❌ WRONG - Anemic Domain / No Architecture

```java
// ❌ Anemic domain - business logic in controllers
@RestController
public class UserController {
    @Autowired
    private UserRepository repository; // Direct database access
    
    @PostMapping("/users")
    public User createUser(@RequestBody User user) {
        // Business logic in controller - WRONG
        if (repository.existsByUsername(user.getUsername())) {
            throw new Exception("User exists");
        }
        return repository.save(user); // Direct persistence - WRONG
    }
}
```

### B. Hexagonal Architecture Benefits

1. **Testability**: Domain logic can be tested without frameworks
2. **Maintainability**: Clear separation of concerns
3. **Flexibility**: Easy to swap adapters (database, web framework)
4. **Performance**: Domain logic is framework-independent, can be optimized
5. **Scalability**: Clean boundaries enable horizontal scaling

---

## 5. Data-Centric & Immutable Patterns (MANDATORY)

### A. Records for Data Carriers

**CRITICAL: Use Records instead of verbose JavaBeans/POJOs for immutable data modeling.**

#### ✅ CORRECT - Using Records

```java
/**
 * Represents a user in the system.
 * 
 * @param id Unique identifier
 * @param username User's login name
 * @param email User's email address
 * @param createdAt Account creation timestamp
 */
public record User(
    Long id,
    String username,
    String email,
    Instant createdAt
) {
    /**
     * Validates email format.
     * 
     * @return true if email is valid
     */
    public boolean hasValidEmail() {
        return email != null && email.contains("@");
    }
}
```

#### ❌ WRONG - Verbose POJO

```java
// ❌ Verbose, boilerplate-heavy POJO
public class User {
    private Long id;
    private String username;
    private String email;
    private Instant createdAt;
    
    public User(Long id, String username, String email, Instant createdAt) {
        this.id = id;
        this.username = username;
        this.email = email;
        this.createdAt = createdAt;
    }
    
    // Getters, setters, equals, hashCode, toString..
    // 50+ lines of boilerplate
}
```

### B. Sealed Classes/Interfaces

**CRITICAL: Use Sealed Classes/Interfaces for controlled hierarchies and exhaustive pattern matching.**

#### ✅ CORRECT - Sealed Classes

```java
/**
 * Represents a result of an operation.
 * 
 * @param <T> The success value type
 * @param <E> The error type
 */
public sealed interface Result<T, E extends Exception> 
    permits Result.Success, Result.Failure {
    
    /**
     * Successful result containing a value.
     * 
     * @param <T> Value type
     * @param <E> Error type
     * @param value The success value
     */
    record Success<T, E extends Exception>(T value) implements Result<T, E> {}
    
    /**
     * Failed result containing an error.
     * 
     * @param <T> Value type
     * @param <E> Error type
     * @param error The error that occurred
     */
    record Failure<T, E extends Exception>(E error) implements Result<T, E> {}
    
    /**
     * Pattern matching with switch expressions.
     * 
     * @param result The result to process
     * @param <T> Value type
     * @param <E> Error type
     * @return Processed value or throws exception
     * @throws E if result is a failure
     */
    static <T, E extends Exception> T unwrap(Result<T, E> result) throws E {
        return switch (result) {
            case Success<T, E>(T value) -> value;
            case Failure<T, E>(E error) -> throw error;
        };
    }
}
```

#### ❌ WRONG - Open Inheritance

```java
// ❌ Open hierarchy - any class can extend
public interface Result<T, E> {
    // No control over implementations
}

// ❌ Can't guarantee exhaustive pattern matching
// ❌ Can't prevent unauthorized implementations
```

### C. Builder Pattern

**CRITICAL: Use Builder Pattern for complex object construction, preferably with Lombok @Builder.**

#### ✅ CORRECT - Builder with Lombok

```java
import lombok.Builder;
import lombok.Value;

/**
 * Configuration for database connection.
 */
@Value
@Builder
public class DatabaseConfig {
    String host;
    int port;
    String database;
    String username;
    String password;
    @Builder.Default
    int connectionTimeout = 30;
    @Builder.Default
    boolean sslEnabled = false;
}

// Usage
DatabaseConfig config = DatabaseConfig.builder()
    .host("localhost")
    .port(5432)
    .database("mydb")
    .username("user")
    .password("pass")
    .build();
```

#### ✅ CORRECT - Manual Builder (if Lombok unavailable)

```java
/**
 * Configuration for database connection.
 */
public final class DatabaseConfig {
    private final String host;
    private final int port;
    private final String database;
    private final String username;
    private final String password;
    private final int connectionTimeout;
    private final boolean sslEnabled;
    
    private DatabaseConfig(Builder builder) {
        this.host = builder.host;
        this.port = builder.port;
        this.database = builder.database;
        this.username = builder.username;
        this.password = builder.password;
        this.connectionTimeout = builder.connectionTimeout;
        this.sslEnabled = builder.sslEnabled;
    }
    
    public static Builder builder() {
        return new Builder();
    }
    
    // Getters..
    
    public static final class Builder {
        private String host;
        private int port;
        private String database;
        private String username;
        private String password;
        private int connectionTimeout = 30;
        private boolean sslEnabled = false;
        
        public Builder host(String host) {
            this.host = host;
            return this;
        }
        
        // Other setters..
        
        public DatabaseConfig build() {
            // Validation
            if (host == null || host.isBlank()) {
                throw new IllegalArgumentException("Host is required");
            }
            return new DatabaseConfig(this);
        }
    }
}
```

---

## 6. Behavioral Patterns (Modernized)

### A. Strategy Pattern (Functional Style)

**CRITICAL: Use lambda expressions or functional interfaces instead of concrete strategy classes.**

#### ✅ CORRECT - Functional Strategy

```java
import java.util.function.Function;

/**
 * Payment processing service using functional strategy pattern.
 */
public class PaymentService {
    
    /**
     * Processes payment using the provided strategy.
     * 
     * @param amount Payment amount
     * @param processor Payment processing strategy
     * @return Payment result
     */
    public PaymentResult processPayment(
            BigDecimal amount,
            Function<BigDecimal, PaymentResult> processor
    ) {
        return processor.apply(amount);
    }
    
    // Usage with lambdas
    public void example() {
        // Credit card strategy
        processPayment(amount, amt -> 
            new PaymentResult(processCreditCard(amt))
        );
        
        // PayPal strategy
        processPayment(amount, amt -> 
            new PaymentResult(processPayPal(amt))
        );
    }
}
```

#### ❌ WRONG - Class-Based Strategy

```java
// ❌ Verbose class-based strategy
public interface PaymentStrategy {
    PaymentResult process(BigDecimal amount);
}

public class CreditCardStrategy implements PaymentStrategy {
    @Override
    public PaymentResult process(BigDecimal amount) {
        // Implementation
    }
}
// More boilerplate..
```

### B. Enum-Based Registry Strategy

**CRITICAL: Use Enums to store and manage strategy implementations for compile-time safety.**

#### ✅ CORRECT - Enum Registry

```java
/**
 * Payment processing strategies as enum registry.
 */
public enum PaymentStrategy {
    CREDIT_CARD(amount -> processCreditCard(amount)),
    PAYPAL(amount -> processPayPal(amount)),
    BANK_TRANSFER(amount -> processBankTransfer(amount));
    
    private final Function<BigDecimal, PaymentResult> processor;
    
    PaymentStrategy(Function<BigDecimal, PaymentResult> processor) {
        this.processor = processor;
    }
    
    /**
     * Processes payment using this strategy.
     * 
     * @param amount Payment amount
     * @return Payment result
     */
    public PaymentResult process(BigDecimal amount) {
        return processor.apply(amount);
    }
    
    // Usage
    PaymentResult result = PaymentStrategy.CREDIT_CARD.process(amount);
}
```

### C. Command Pattern

**CRITICAL: Use Command Pattern for auditing, undo functionality, and multi-step operations.**

#### ✅ CORRECT - Command Pattern

```java
/**
 * Command interface for reversible operations.
 * 
 * @param <T> Result type
 */
@FunctionalInterface
public interface Command<T> {
    /**
     * Executes the command.
     * 
     * @return Command result
     */
    T execute();
    
    /**
     * Reverses the command (undo).
     * 
     * @return Undo result
     */
    default Command<T> undo() {
        return () -> {
            throw new UnsupportedOperationException("Undo not supported");
        };
    }
}

/**
 * Command executor with audit trail.
 */
public class CommandExecutor {
    private final List<Command<?>> history = new ArrayList<>();
    
    /**
     * Executes command and records in history.
     * 
     * @param command Command to execute
     * @param <T> Result type
     * @return Command result
     */
    public <T> T execute(Command<T> command) {
        T result = command.execute();
        history.add(command);
        return result;
    }
    
    /**
     * Undoes last command.
     */
    public void undo() {
        if (!history.isEmpty()) {
            Command<?> last = history.remove(history.size() - 1);
            last.undo().execute();
        }
    }
}
```

### D. Observer/Reactive Pattern

**CRITICAL: Use Project Reactor or Java Flow API instead of java.util.Observable.**

#### ✅ CORRECT - Project Reactor

```java
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;

/**
 * Event publisher using Project Reactor.
 */
public class EventPublisher {
    private final Flux<Event> eventStream;
    
    /**
     * Publishes event to subscribers.
     * 
     * @param event Event to publish
     * @return Mono that completes when event is published
     */
    public Mono<Void> publish(Event event) {
        return Mono.fromRunnable(() -> 
            eventStream.doOnNext(e -> processEvent(e))
        ).then();
    }
    
    /**
     * Subscribes to event stream.
     * 
     * @param handler Event handler
     * @return Disposable subscription
     */
    public Disposable subscribe(Function<Event, Mono<Void>> handler) {
        return eventStream
            .flatMap(handler)
            .subscribe();
    }
}
```

#### ✅ CORRECT - Java Flow API

```java
import java.util.concurrent.Flow;

/**
 * Event publisher using Java Flow API.
 */
public class FlowEventPublisher implements Flow.Publisher<Event> {
    private final List<Flow.Subscriber<? super Event>> subscribers = 
        new CopyOnWriteArrayList<>();
    
    @Override
    public void subscribe(Flow.Subscriber<? super Event> subscriber) {
        subscribers.add(subscriber);
        subscriber.onSubscribe(new Flow.Subscription() {
            @Override
            public void request(long n) {
                // Handle backpressure
            }
            
            @Override
            public void cancel() {
                subscribers.remove(subscriber);
            }
        });
    }
    
    /**
     * Publishes event to all subscribers.
     * 
     * @param event Event to publish
     */
    public void publish(Event event) {
        subscribers.forEach(sub -> {
            try {
                sub.onNext(event);
            } catch (Exception e) {
                sub.onError(e);
            }
        });
    }
}
```

---

## 7. Structural & Architectural Patterns

### A. Repository Pattern

**CRITICAL: Use Repository Pattern with Spring Data JPA for decoupling business logic from data access.**

#### ✅ CORRECT - Spring Data Repository

```java
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Optional;

/**
 * Repository for user data access operations.
 */
@Repository
public interface UserRepository extends JpaRepository<User, Long> {
    
    /**
     * Finds user by username.
     * 
     * @param username Username to search for
     * @return Optional user if found
     */
    Optional<User> findByUsername(String username);
    
    /**
     * Finds all active users.
     * 
     * @return List of active users
     */
    @Query("SELECT u FROM User u WHERE u.active = true")
    List<User> findActiveUsers();
    
    /**
     * Checks if email exists.
     * 
     * @param email Email to check
     * @return true if email exists
     */
    boolean existsByEmail(String email);
}
```

### B. Facade Pattern

**CRITICAL: Use Facade Pattern to provide simplified interface to complex subsystems.**

#### ✅ CORRECT - Service Facade

```java
import org.springframework.stereotype.Service;

/**
 * Facade for order processing subsystem.
 * Simplifies complex order operations behind a single interface.
 */
@Service
public class OrderServiceFacade {
    private final InventoryService inventoryService;
    private final PaymentService paymentService;
    private final ShippingService shippingService;
    private final NotificationService notificationService;
    
    /**
     * Processes complete order workflow.
     * 
     * @param order Order to process
     * @return Processing result
     */
    public CompletableFuture<OrderResult> processOrder(Order order) {
        return CompletableFuture
            .supplyAsync(() -> inventoryService.reserveItems(order))
            .thenCompose(reservation -> 
                paymentService.processPayment(order, reservation))
            .thenCompose(payment -> 
                shippingService.scheduleShipping(order, payment))
            .thenApply(shipping -> {
                notificationService.sendConfirmation(order);
                return new OrderResult(order, shipping);
            });
    }
}
```

### C. Decorator Pattern

**CRITICAL: Use Decorator Pattern for adding behavior dynamically at runtime.**

#### ✅ CORRECT - Decorator Pattern

```java
/**
 * Base data processor interface.
 */
public interface DataProcessor {
    /**
     * Processes data.
     * 
     * @param data Data to process
     * @return Processed data
     */
    String process(String data);
}

/**
 * Base processor implementation.
 */
public class BasicProcessor implements DataProcessor {
    @Override
    public String process(String data) {
        return data.trim();
    }
}

/**
 * Encryption decorator.
 */
public class EncryptionDecorator implements DataProcessor {
    private final DataProcessor processor;
    private final EncryptionService encryptionService;
    
    public EncryptionDecorator(DataProcessor processor, 
                               EncryptionService encryptionService) {
        this.processor = processor;
        this.encryptionService = encryptionService;
    }
    
    @Override
    public String process(String data) {
        String processed = processor.process(data);
        return encryptionService.encrypt(processed);
    }
}

/**
 * Compression decorator.
 */
public class CompressionDecorator implements DataProcessor {
    private final DataProcessor processor;
    
    public CompressionDecorator(DataProcessor processor) {
        this.processor = processor;
    }
    
    @Override
    public String process(String data) {
        String processed = processor.process(data);
        return compress(processed);
    }
    
    // Usage: new CompressionDecorator(
    //     new EncryptionDecorator(
    //         new BasicProcessor(), encryptionService))
}
```

---

## 8. Modern Language Features

### A. Switch Expressions & Pattern Matching

**CRITICAL: Use Switch Expressions and Pattern Matching to reduce verbosity and enhance safety.**

#### ✅ CORRECT - Switch Expressions

```java
/**
 * Processes status using switch expression.
 * 
 * @param status Status to process
 * @return Processed message
 */
public String processStatus(Status status) {
    return switch (status) {
        case PENDING -> "Processing...";
        case APPROVED -> "Approved";
        case REJECTED -> "Rejected";
        case CANCELLED -> "Cancelled";
    };
}

/**
 * Pattern matching with sealed classes.
 * 
 * @param result Result to process
 * @param <T> Value type
 * @return Processed value
 */
public <T> T processResult(Result<T, Exception> result) {
    return switch (result) {
        case Result.Success<T, Exception>(T value) -> value;
        case Result.Failure<T, Exception>(Exception error) -> {
            log.error("Operation failed", error);
            throw new ProcessingException(error);
        }
    };
}
```

#### ❌ WRONG - Verbose If-Else

```java
// ❌ Verbose if-else chain
public String processStatus(Status status) {
    if (status == Status.PENDING) {
        return "Processing...";
    } else if (status == Status.APPROVED) {
        return "Approved";
    } else if (status == Status.REJECTED) {
        return "Rejected";
    } else if (status == Status.CANCELLED) {
        return "Cancelled";
    } else {
        throw new IllegalArgumentException("Unknown status");
    }
}
```

### B. Streams API

**CRITICAL: Use Streams API for functional, declarative data processing.**

#### ✅ CORRECT - Streams API

```java
import java.util.stream.Stream;
import java.util.List;

/**
 * Processes users using streams.
 * 
 * @param users List of users
 * @return List of active user emails
 */
public List<String> getActiveUserEmails(List<User> users) {
    return users.stream()
        .filter(User::isActive)
        .map(User::email)
        .filter(email -> email != null && !email.isBlank())
        .distinct()
        .sorted()
        .toList();
}

/**
 * Parallel processing for large datasets.
 * 
 * @param items Items to process
 * @return Processed items
 */
public List<ProcessedItem> processInParallel(List<Item> items) {
    return items.parallelStream()
        .map(this::processItem)
        .filter(ProcessedItem::isValid)
        .toList();
}
```

### C. Enum Singletons

**CRITICAL: Use Enum Singletons for thread-safe, serialization-resistant singleton pattern.**

#### ✅ CORRECT - Enum Singleton

```java
/**
 * Thread-safe singleton using enum pattern.
 * Most robust implementation - serialization-safe, reflection-safe.
 */
public enum DatabaseConnection {
    INSTANCE;
    
    private Connection connection;
    
    /**
     * Gets database connection.
     * 
     * @return Database connection
     */
    public Connection getConnection() {
        if (connection == null) {
            connection = createConnection();
        }
        return connection;
    }
    
    private Connection createConnection() {
        // Connection creation logic
        return null; // Placeholder
    }
}

// Usage
Connection conn = DatabaseConnection.INSTANCE.getConnection();
```

---

## 9. Async/Await & Virtual Threads (MANDATORY)

### A. Virtual Threads

**CRITICAL: Use Virtual Threads (Java 21+) for concurrent operations instead of platform threads.**

#### ✅ CORRECT - Virtual Threads

```java
import java.util.concurrent.Executors;
import java.util.concurrent.CompletableFuture;
import java.util.List;

/**
 * Service using virtual threads for concurrent operations.
 */
public class ConcurrentService {
    
    /**
     * Processes items concurrently using virtual threads.
     * 
     * @param items Items to process
     * @return List of processed results
     */
    public List<ProcessedItem> processConcurrently(List<Item> items) {
        try (var executor = Executors.newVirtualThreadPerTaskExecutor()) {
            return items.stream()
                .map(item -> CompletableFuture.supplyAsync(
                    () -> processItem(item), executor))
                .map(CompletableFuture::join)
                .toList();
        }
    }
    
    /**
     * Processes item (blocking I/O operation).
     * 
     * @param item Item to process
     * @return Processed item
     */
    private ProcessedItem processItem(Item item) {
        // Blocking I/O - virtual threads handle this efficiently
        return fetchData(item)
            .thenApply(this::transform)
            .join();
    }
}
```

#### ❌ WRONG - Platform Threads

```java
// ❌ Using platform threads (wasteful for I/O-bound operations)
ExecutorService executor = Executors.newFixedThreadPool(100);
// Limited scalability, high memory overhead
```

### B. CompletableFuture & Async Patterns

**CRITICAL: Use CompletableFuture for async operations when virtual threads aren't applicable.**

#### ✅ CORRECT - CompletableFuture

```java
import java.util.concurrent.CompletableFuture;
import java.util.List;

/**
 * Async service using CompletableFuture.
 */
public class AsyncService {
    
    /**
     * Processes data asynchronously.
     * 
     * @param data Data to process
     * @return CompletableFuture with result
     */
    public CompletableFuture<Result> processAsync(String data) {
        return CompletableFuture
            .supplyAsync(() -> fetchData(data))
            .thenApply(this::transform)
            .thenCompose(this::validate)
            .exceptionally(this::handleError);
    }
    
    /**
     * Processes multiple items concurrently.
     * 
     * @param items Items to process
     * @return CompletableFuture with all results
     */
    public CompletableFuture<List<Result>> processAllAsync(
            List<String> items) {
        List<CompletableFuture<Result>> futures = items.stream()
            .map(this::processAsync)
            .toList();
        
        return CompletableFuture.allOf(
            futures.toArray(new CompletableFuture[0]))
            .thenApply(v -> futures.stream()
                .map(CompletableFuture::join)
                .toList());
    }
}
```

### C. Project Reactor (Reactive)

**CRITICAL: Use Project Reactor for reactive, non-blocking event-driven programming.**

#### ✅ CORRECT - Project Reactor

```java
import reactor.core.publisher.Mono;
import reactor.core.publisher.Flux;
import reactor.core.scheduler.Schedulers;

/**
 * Reactive service using Project Reactor.
 */
public class ReactiveService {
    
    /**
     * Processes data reactively.
     * 
     * @param data Data to process
     * @return Mono with result
     */
    public Mono<Result> processReactive(String data) {
        return Mono.fromCallable(() -> fetchData(data))
            .subscribeOn(Schedulers.boundedElastic())
            .map(this::transform)
            .flatMap(this::validate)
            .onErrorResume(this::handleError);
    }
    
    /**
     * Processes stream of items reactively.
     * 
     * @param items Stream of items
     * @return Flux of processed results
     */
    public Flux<Result> processStream(Flux<String> items) {
        return items
            .flatMap(this::processReactive)
            .filter(Result::isValid)
            .buffer(100)
            .flatMap(this::batchProcess);
    }
}
```

---

## 10. Functional Programming (MANDATORY)

### A. Prefer Functional Style

**CRITICAL: Prefer functional programming patterns over imperative code.**

#### ✅ CORRECT - Functional Style

```java
import java.util.function.Function;
import java.util.function.Predicate;
import java.util.List;
import java.util.Optional;

/**
 * Functional data processing.
 */
public class FunctionalService {
    
    /**
     * Processes data using functional composition.
     * 
     * @param data Input data
     * @param transform Transformation function
     * @param filter Filter predicate
     * @return Optional result
     */
    public Optional<String> processFunctional(
            String data,
            Function<String, String> transform,
            Predicate<String> filter) {
        return Optional.ofNullable(data)
            .map(transform)
            .filter(filter)
            .map(String::toUpperCase);
    }
    
    /**
     * Composes multiple functions.
     * 
     * @param functions Functions to compose
     * @return Composed function
     */
    @SafeVarargs
    public final Function<String, String> composeFunctions(
            Function<String, String>... functions) {
        return Stream.of(functions)
            .reduce(Function.identity(), Function::andThen);
    }
}
```

#### ❌ WRONG - Imperative Style

```java
// ❌ Verbose imperative code
public String processImperative(String data) {
    if (data == null) {
        return null;
    }
    String transformed = transform(data);
    if (!filter(transformed)) {
        return null;
    }
    return transformed.toUpperCase();
}
```

### B. Immutability

**CRITICAL: Prefer immutable data structures and defensive copying.**

#### ✅ CORRECT - Immutable Collections

```java
import java.util.Collections;
import java.util.List;
import java.util.Map;

/**
 * Service with immutable data handling.
 */
public class ImmutableService {
    
    /**
     * Returns immutable list.
     * 
     * @param items Items to wrap
     * @return Immutable list
     */
    public List<String> getImmutableList(List<String> items) {
        return List.copyOf(items); // Java 10+
    }
    
    /**
     * Returns immutable map.
     * 
     * @param map Map to wrap
     * @return Immutable map
     */
    public Map<String, String> getImmutableMap(Map<String, String> map) {
        return Map.copyOf(map); // Java 10+
    }
    
    /**
     * Defensive copying for mutable objects.
     * 
     * @param config Config to copy
     * @return Defensive copy
     */
    public Config createDefensiveCopy(Config config) {
        return new Config(
            config.host(),
            config.port(),
            List.copyOf(config.allowedUsers())
        );
    }
}
```

---

## 11. JavaDoc Documentation (MANDATORY)

### A. Complete API Documentation

**CRITICAL: All public APIs MUST have complete JavaDoc comments for auto-generated documentation.**

#### ✅ CORRECT - Complete JavaDoc

```java
/**
 * Service for processing user data.
 * 
 * <p>This service provides operations for:
 * <ul>
 *   <li>User creation and validation</li>
 *   <li>User data retrieval</li>
 *   <li>User status management</li>
 * </ul>
 * 
 * <p>All operations are thread-safe and can be used concurrently.
 * 
 * @author John Doe
 * @version 1.0
 * @since 1.0
 */
public class UserService {
    
    /**
     * Creates a new user with the specified details.
     * 
     * <p>Validates the user data before creation:
     * <ul>
     *   <li>Username must be unique</li>
     *   <li>Email must be valid format</li>
     *   <li>Password must meet security requirements</li>
     * </ul>
     * 
     * @param username User's login name (must be unique, not null)
     * @param email User's email address (must be valid format, not null)
     * @param password User's password (must meet requirements, not null)
     * @return Created user with generated ID
     * @throws IllegalArgumentException if validation fails
     * @throws DuplicateUserException if username already exists
     * @since 1.0
     */
    public User createUser(String username, String email, String password) {
        // Implementation
    }
    
    /**
     * Finds user by ID.
     * 
     * @param id User ID (must be positive)
     * @return Optional user if found, empty otherwise
     * @throws IllegalArgumentException if id is not positive
     */
    public Optional<User> findById(Long id) {
        // Implementation
    }
}
```

#### ❌ WRONG - Missing JavaDoc

```java
// ❌ No JavaDoc - cannot generate documentation
public class UserService {
    public User createUser(String username, String email, String password) {
        // No documentation
    }
}
```

### B. JavaDoc Generation

**CRITICAL: JavaDoc MUST be generatable without errors.**

```bash
# Generate JavaDoc
mvn javadoc:javadoc
# OR
./gradlew javadoc

# Verify no warnings
# All public classes, methods, fields must be documented
```

---

## 12. Performance, Scalability & Memory Optimization (MANDATORY)

**CRITICAL: All code MUST be optimized for performance, scalability, and minimal memory footprint.**

### A. Performance Principles

1. **Minimalistic Code**: Less code = faster execution, less memory
2. **Efficient Algorithms**: Choose optimal data structures and algorithms
3. **Lazy Evaluation**: Process only what's needed
4. **Caching**: Cache expensive computations when appropriate
5. **Resource Management**: Proper cleanup, use try-with-resources

### B. Memory Footprint Optimization

**CRITICAL: Minimize memory usage for scalability.**

#### ✅ CORRECT - Memory-Efficient Patterns

```java
// ✅ Use records (minimal memory overhead)
public record User(Long id, String username, String email) {}

// ✅ Use streams for large datasets (lazy, memory-efficient)
public List<String> processLargeDataset(List<Data> data) {
    return data.stream()
        .filter(this::isValid)
        .map(this::transform)
        .limit(1000) // Limit results to prevent memory issues
        .toList();
}

// ✅ Use primitive collections when possible
IntList ids = new IntArrayList(); // More memory-efficient than List<Integer>

// ✅ Clear references when done
public void processAndCleanup(List<LargeObject> objects) {
    try {
        processObjects(objects);
    } finally {
        objects.clear(); // Help GC
    }
}
```

#### ❌ WRONG - Memory Waste

```java
// ❌ Unnecessary object creation
for (int i = 0; i < 1000000; i++) {
    String result = new String("constant"); // Creates new object each time
}

// ❌ Loading entire dataset into memory
List<Data> allData = repository.findAll(); // May be millions of records
// Should use pagination or streaming
```

### C. Scalability Patterns

**CRITICAL: Design for horizontal and vertical scaling.**

#### ✅ CORRECT - Scalable Patterns

```java
// ✅ Stateless services (easily scalable)
@Service
public class UserService {
    // No instance state - can scale horizontally
    public User processUser(User user) {
        return transform(user);
    }
}

// ✅ Use virtual threads for I/O-bound operations (scales to millions)
try (var executor = Executors.newVirtualThreadPerTaskExecutor()) {
    List<CompletableFuture<Result>> futures = items.stream()
        .map(item -> CompletableFuture.supplyAsync(
            () -> fetchFromDatabase(item), executor))
        .toList();
    
    return futures.stream()
        .map(CompletableFuture::join)
        .toList();
}

// ✅ Pagination for large datasets
public Page<User> findUsers(Pageable pageable) {
    return repository.findAll(pageable);
}
```

### D. Virtual Threads for I/O-Bound Operations

**CRITICAL: Use Virtual Threads for I/O-bound concurrent operations (scales to millions).**

```java
// ✅ CORRECT - Virtual threads for I/O (minimal memory, maximum scalability)
try (var executor = Executors.newVirtualThreadPerTaskExecutor()) {
    List<CompletableFuture<Result>> futures = items.stream()
        .map(item -> CompletableFuture.supplyAsync(
            () -> fetchFromDatabase(item), executor))
        .toList();
    
    return futures.stream()
        .map(CompletableFuture::join)
        .toList();
}
```

### E. Parallel Streams for CPU-Bound Operations

**CRITICAL: Use parallel streams for CPU-intensive operations.**

```java
// ✅ CORRECT - Parallel streams for CPU-bound work
List<ProcessedItem> results = items.parallelStream()
    .map(this::cpuIntensiveProcessing)
    .filter(ProcessedItem::isValid)
    .toList();
```

### F. Lazy Evaluation

**CRITICAL: Use lazy evaluation to avoid unnecessary computations and memory usage.**

```java
// ✅ CORRECT - Lazy evaluation (processes only what's needed)
Optional<String> result = data.stream()
    .filter(this::expensiveFilter)
    .map(this::expensiveTransform)
    .findFirst(); // Only processes until first match

// ✅ CORRECT - Stream processing (doesn't load all into memory)
public void processLargeFile(Path file) throws IOException {
    Files.lines(file)
        .filter(this::isValid)
        .map(this::transform)
        .forEach(this::process); // Processes line by line, not all at once
}
```

### G. Performance Checklist

**All code MUST:**

- [ ] Use virtual threads for I/O-bound operations
- [ ] Use parallel streams for CPU-bound operations
- [ ] Minimize object creation (reuse when possible)
- [ ] Use lazy evaluation (streams, Optional)
- [ ] Clear large collections when done
- [ ] Use pagination for large datasets
- [ ] Prefer records over classes for data carriers
- [ ] Use primitive collections when possible
- [ ] Avoid unnecessary copying (defensive copying only when needed)
- [ ] Profile and measure performance-critical code

---

## 13. Testing Requirements (MANDATORY)

### A. Unit Testing (MANDATORY - ALWAYS REQUIRED)

**CRITICAL: All new/modified code MUST have unit tests. Unit tests MUST pass before code delivery. This is non-negotiable.**

**MANDATORY RULES:**
1. **Unit tests are ALWAYS required** for all new code
2. **Unit tests are ALWAYS required** for all modified code
3. **All unit tests MUST pass** before code delivery
4. **After ANY code change**, re-run tests to verify they still pass
5. **Minimum 80% code coverage** for business logic

#### ✅ CORRECT - JUnit 5 Tests

```java
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for UserService.
 */
class UserServiceTest {
    private UserService userService;
    
    @BeforeEach
    void setUp() {
        userService = new UserService();
    }
    
    @Test
    void createUser_ValidData_ReturnsUser() {
        // Given
        String username = "testuser";
        String email = "test@example.com";
        String password = "SecurePass123!";
        
        // When
        User result = userService.createUser(username, email, password);
        
        // Then
        assertNotNull(result);
        assertEquals(username, result.username());
        assertEquals(email, result.email());
        assertNotNull(result.id());
    }
    
    @Test
    void createUser_DuplicateUsername_ThrowsException() {
        // Given
        userService.createUser("user", "email@example.com", "pass");
        
        // When/Then
        assertThrows(DuplicateUserException.class, () ->
            userService.createUser("user", "other@example.com", "pass")
        );
    }
}
```

---

## 14. Summary

**CRITICAL Requirements for All Java Code:**

1. **Dependency Management**: Prefer Gradle, use Maven as fallback
2. **Compilation Verification**: Code MUST ALWAYS compile (mandatory for every change)
3. **Unit Tests**: ALWAYS required for all new/modified code, MUST pass
4. **Hexagonal Architecture**: All applications MUST follow ports and adapters pattern
5. **Records over POJOs**: Use records for immutable data carriers
6. **Sealed Classes**: Use sealed classes/interfaces for controlled hierarchies
7. **Builder Pattern**: Use builders for complex object construction
8. **Functional Patterns**: Prefer functional programming, streams, lambdas
9. **Virtual Threads**: Use virtual threads for concurrent I/O operations
10. **Async/Await**: Use CompletableFuture or Project Reactor for async operations
11. **JavaDoc**: Complete API documentation, well-documented code, auto-generatable
12. **Testing**: 80%+ code coverage, comprehensive unit tests, always required
13. **Modern Patterns**: Strategy (functional), Command, Observer (reactive)
14. **Modern Features**: Switch expressions, pattern matching, streams API
15. **Immutability**: Prefer immutable data structures
16. **Performance**: Optimize for speed, scalability, minimal memory footprint
17. **Minimalistic Code**: Clean, readable, concise code
18. **Verification**: Agent MUST compile, test, and generate JavaDoc before delivery

**Agent Verification Protocol:**
- **MANDATORY**: Compile code (`mvn compile` or `./gradlew compileJava`) - ALWAYS required
- **MANDATORY**: Run unit tests (`mvn test` or `./gradlew test`) - ALWAYS required, MUST pass
- Generate JavaDoc (`mvn javadoc:javadoc` or `./gradlew javadoc`)
- **MANDATORY**: After ANY modification, re-compile and re-run tests
- Only present working, tested, documented code to the user

**Remember**: Minimalistic, clean, readable, well-documented, functional, immutable, async-first code with hexagonal architecture, virtual threads, modern patterns, focus on performance, scalability, and minimal memory footprint. Keep it simple, keep it modern, keep it working.

---

## 15. Quick Reference

### Common Commands

```bash
# Build (Gradle)
./gradlew build
./gradlew compileJava

# Test
./gradlew test
./gradlew test --tests "MyTest"

# Build (Maven)
mvn compile
mvn package

# Test (Maven)
mvn test

# Format
mvn spotless:apply
./gradlew spotlessApply

# Documentation
mvn javadoc:javadoc
./gradlew javadoc
```

### Modern Java Patterns

```java
// Records (immutable data)
record User(Long id, String name, String email) {}

// Sealed classes
sealed interface Result permits Success, Failure {}
record Success(Data data) implements Result {}
record Failure(String error) implements Result {}

// Pattern matching
if (obj instanceof User u) { use(u.name()); }

// Switch expressions
String result = switch (status) {
    case ACTIVE -> "active";
    case PENDING -> "pending";
    default -> "unknown";
};

// Streams
list.stream()
    .filter(x -> x.active())
    .map(User::name)
    .toList();

// Virtual threads
Thread.startVirtualThread(() -> process());
```

### build.gradle.kts Template

```kotlin
plugins {
    java
    id("org.springframework.boot") version "3.2.0"
}

java { toolchain { languageVersion = JavaLanguageVersion.of(21) } }

dependencies {
    implementation("org.springframework.boot:spring-boot-starter-web")
    testImplementation("org.springframework.boot:spring-boot-starter-test")
}

tasks.test { useJUnitPlatform() }
```

### Project Structure

```
my_project/
├── build.gradle.kts
├── src/
│   ├── main/java/com/example/
│   │   ├── domain/          # Domain models
│   │   ├── port/            # Interfaces
│   │   └── adapter/         # Implementations
│   └── test/java/
└── docs/
```

---

## References

- [Java Documentation](https://docs.oracle.com/en/java/)
- [Spring Boot Reference](https://docs.spring.io/spring-boot/docs/current/reference/)
- [JUnit 5 User Guide](https://junit.org/junit5/docs/current/user-guide/)


**End of Modern Java Development Guidelines**
