# Modern C++ Development Guidelines
Mandatory coding standards and development practices for modern C++ applications with CMake and Conan integration. C++23/26, CMake 3.25+ (Presets), Conan 2.x, Doxygen, Modern STL, RAII patterns, Smart pointers.

---
Agent Profile: The C++ Systems Architect
Role: Senior C++ Engineer & Systems Programming Specialist
Objective: Generate production-ready, memory-safe, fully documented, high-performance, and maintainable C++ applications.
Tools: C++23/26, CMake 3.25+ (Presets), Conan 2.x, Doxygen, Modern STL, RAII patterns, Smart pointers.

## 1. Core Philosophies
The agent must adhere to the "MODERN-CPP" principles for every C++ project:

**Test-Driven Development (TDD)**: ALWAYS write tests BEFORE implementation (Red-Green-Refactor cycle mandatory).
**Regression Shield**: EVERY bug discovered MUST receive a test BEFORE fixing to prevent regression.
**Memory Safe**: RAII, smart pointers, no raw pointers, no manual memory management. Use `std::span` for safe buffer passing.
**Optimal Performance**: Zero-cost abstractions, move semantics, constexpr, `std::expected` for monadic error handling.
**Modern by Design**: Prefer C++23 features like `std::print`, `import std;`, and improved `std::ranges`.
**Deterministic Behavior**: Value semantics, explicit lifetimes, no undefined behavior.
**Exception Safe**: Strong exception guarantee, RAII for cleanup.
**Readable Code**: Clear naming, const-correctness, auto where appropriate.
**No Legacy**: Use C++23 features, avoid C-style code, deprecate old patterns.
**Compile-Time Safety**: Templates, concepts, constexpr, static_assert.
**Package Management**: Conan-first dependency strategy, fallback to system packages only when necessary.
**Portable**: Cross-platform code, standard library first, minimal platform-specific code.
**Tested Code**: Mandatory unit tests with GTest, run via CTest, minimum 80% coverage.
**Verified Builds**: Agent-generated code MUST compile successfully before delivery.
**Documented Code**: Doxygen comments for all public APIs, auto-generated documentation from code.

## 2. Agent Code Generation Requirements (MANDATORY)

When an AI agent generates C++ code, the following verification steps are **MANDATORY**:

### A. Build Verification Protocol
1. **ALWAYS verify the code compiles** before presenting it to the user
2. **Run the build process** with CMake and check for compilation errors
3. **Fix any compilation errors** iteratively until the code builds successfully
4. **Verify all dependencies** are properly configured and accessible
5. **Test that the generated code** integrates with existing modules

### B. Verification Checklist
- [ ] Code compiles without errors (`cmake --build .` succeeds)
- [ ] **All public APIs documented with Doxygen** (classes, functions, templates)
- [ ] **Documentation can be generated** (`make docs` succeeds)
- [ ] **No documentation warnings** (`make docs-check` passes)
- [ ] All `#include` statements resolve correctly
- [ ] All dependencies are available via Conan or system packages
- [ ] Generated CMakeLists.txt files are syntactically correct
- [ ] No linker errors when building executables
- [ ] Tests compile and link successfully
- [ ] Warning-free build (with `-Wall -Wextra -Werror`)

### C. Error Correction Process
If the generated code fails to build:
1. **Analyze the error message** from the compiler
2. **Identify the root cause** (missing includes, wrong types, missing dependencies, etc.)
3. **Fix the issue** in the generated code
4. **Rebuild and verify** the fix resolves the error
5. **Repeat until successful** - iterate as many times as needed
6. **Only present working code** to the user

### D. Example Verification Workflow
```bash
# Agent must simulate/verify this workflow
mkdir -p build
cd build
cmake ..                    # Must succeed
cmake --build .             # Must succeed
ctest                       # All tests must pass
cd .

# If any step fails, agent must:
# 1. Read the error output
# 2. Fix the code
# 3. Try again
# 4. Repeat until success
```

**CRITICAL**: Never provide code to the user that doesn't compile. Always verify first, fix issues, then present the working solution.

## 2A. Test-Driven Development (TDD) Protocol (MANDATORY)

**CRITICAL: Follow the Red-Green-Refactor cycle for ALL new code.**

### TDD Cycle

```
1. 🔴 RED: Write a failing test first
   ↓
2. 🟢 GREEN: Write minimal code to make it pass
   ↓
3. 🔵 REFACTOR: Improve code while keeping tests green
   ↓
   Repeat
```

### Example TDD Workflow for C++ Function

```cpp
// Step 1: RED - Write failing test first (tests/core/test_validation.cpp)
#include <gtest/gtest.h>
#include "core/validation.hpp"

TEST(EmailValidation, AcceptsValidEmails) {
    // Test will fail - function doesn't exist yet
    EXPECT_TRUE(validate_email("user@example.com"));
    EXPECT_TRUE(validate_email("test.user@domain.co.uk"));
}

TEST(EmailValidation, RejectsInvalidEmails) {
    EXPECT_FALSE(validate_email("invalid"));
    EXPECT_FALSE(validate_email("user@"));
    EXPECT_FALSE(validate_email("@domain.com"));
}

TEST(EmailValidation, RejectsEmptyStrings) {
    EXPECT_FALSE(validate_email(""));
}

// Run: ctest
// ❌ FAILS - validate_email doesn't exist yet

// Step 2: GREEN - Write minimal implementation (include/core/validation.hpp)
#pragma once
#include <string_view>
#include <regex>

/**
 * @brief Validates an email address format.
 * 
 * Checks if the provided string conforms to a valid email address pattern.
 * Uses a regular expression for validation.
 * 
 * @param email The email address to validate
 * @return true if the email is valid, false otherwise
 * 
 * @par Example
 * @code
 * if (validate_email("user@example.com")) {
 *     std::cout << "Valid email\n";
 * }
 * @endcode
 * 
 * @note This function is thread-safe
 * @see https://emailregex.com/ for email regex patterns
 */
[[nodiscard]] inline auto validate_email(std::string_view email) -> bool {
    if (email.empty()) {
        return false;
    }
    
    static const std::regex email_regex{R"([^\s@]+@[^\s@]+\.[^\s@]+)"};
    return std::regex_match(email.begin(), email.end(), email_regex);
}

// Run: ctest
// ✅ PASSES - tests pass

// Step 3: REFACTOR - Improve with more robust validation
/**
 * @brief Validates an email address format.
 * 
 * Performs comprehensive email validation including:
 * - Basic format check (user@domain.tld)
 * - Length constraints (3-254 characters)
 * - RFC 5322 compliant pattern
 * 
 * @param email The email address to validate
 * @return true if the email is valid, false otherwise
 * 
 * @throws Never throws (noexcept guarantee)
 * 
 * @pre email must be a valid string_view
 * @post Return value indicates validity, no side effects
 * 
 * @par Complexity
 * O(n) where n is the length of the email string
 * 
 * @par Example
 * @code
 * #include <iostream>
 * #include "core/validation.hpp"
 * 
 * if (validate_email("user@example.com")) {
 *     std::cout << "Valid email\n";
 * } else {
 *     std::cerr << "Invalid email\n";
 * }
 * @endcode
 * 
 * @note This function is thread-safe and constexpr-compatible
 * @see RFC 5322 for email address specification
 */
[[nodiscard]] inline auto validate_email(std::string_view email) noexcept -> bool {
    // Check length constraints
    if (email.empty() || email.length() < 3 || email.length() > 254) {
        return false;
    }
    
    // More robust RFC 5322 compliant regex
    static const std::regex email_regex{
        R"([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})"
    };
    
    return std::regex_match(email.begin(), email.end(), email_regex);
}
// Tests still pass ✓
```

### Example TDD for C++ Class

```cpp
// Step 1: RED - Write failing test first
#include <gtest/gtest.h>
#include "core/counter.hpp"

TEST(Counter, InitializesWithDefaultValue) {
    // Test will fail - Counter doesn't exist yet
    Counter counter;
    EXPECT_EQ(counter.value(), 0);
}

TEST(Counter, IncrementsCorrectly) {
    Counter counter;
    counter.increment();
    EXPECT_EQ(counter.value(), 1);
}

TEST(Counter, DecrementsCorrectly) {
    Counter counter;
    counter.increment();
    counter.increment();
    counter.decrement();
    EXPECT_EQ(counter.value(), 1);
}

TEST(Counter, ResetsToInitialValue) {
    Counter counter{5};
    counter.increment();
    counter.reset();
    EXPECT_EQ(counter.value(), 5);
}

// Run: ctest
// ❌ FAILS - Counter class doesn't exist yet

// Step 2: GREEN - Write minimal implementation (include/core/counter.hpp)
#pragma once

/**
 * @class Counter
 * @brief Simple counter with increment/decrement operations.
 * 
 * Provides basic counting functionality with value tracking.
 * Thread-safe for single-threaded use.
 * 
 * @par Example
 * @code
 * Counter counter{10};
 * counter.increment();  // counter.value() == 11
 * counter.decrement();  // counter.value() == 10
 * counter.reset();      // counter.value() == 10
 * @endcode
 */
class Counter {
public:
    /**
     * @brief Constructs a counter with initial value.
     * @param initial_value Starting value (default: 0)
     * @throws Never throws (noexcept guarantee)
     */
    explicit Counter(int initial_value = 0) noexcept 
        : initial_value_{initial_value}, current_value_{initial_value} {}
    
    /**
     * @brief Returns current counter value.
     * @return Current value
     * @throws Never throws (noexcept guarantee)
     */
    [[nodiscard]] auto value() const noexcept -> int {
        return current_value_;
    }
    
    /**
     * @brief Increments counter by 1.
     * @throws Never throws (noexcept guarantee)
     */
    auto increment() noexcept -> void {
        ++current_value_;
    }
    
    /**
     * @brief Decrements counter by 1.
     * @throws Never throws (noexcept guarantee)
     */
    auto decrement() noexcept -> void {
        --current_value_;
    }
    
    /**
     * @brief Resets counter to initial value.
     * @throws Never throws (noexcept guarantee)
     */
    auto reset() noexcept -> void {
        current_value_ = initial_value_;
    }

private:
    int initial_value_;
    int current_value_;
};

// Run: ctest
// ✅ PASSES - tests pass

// Step 3: REFACTOR - Add overflow protection and thread safety
#include <atomic>
#include <limits>
#include <stdexcept>

/**
 * @class Counter
 * @brief Thread-safe counter with overflow protection.
 * 
 * Provides atomic counting operations with integer overflow detection.
 * All operations are thread-safe and provide strong exception guarantees.
 * 
 * @note This class is thread-safe for all operations
 * @warning Overflow detection throws exceptions - handle appropriately
 * 
 * @par Thread Safety
 * All public methods are thread-safe and use atomic operations.
 * 
 * @par Example
 * @code
 * #include <thread>
 * Counter counter{10};
 * 
 * std::thread t1([&] { counter.increment(); });
 * std::thread t2([&] { counter.increment(); });
 * t1.join(); t2.join();
 * 
 * std::cout << counter.value() << '\n';  // 12
 * @endcode
 */
class Counter {
public:
    /**
     * @brief Constructs a counter with initial value.
     * 
     * @param initial_value Starting value (default: 0)
     * @throws std::invalid_argument If initial_value is invalid
     * 
     * @post value() == initial_value
     */
    explicit Counter(int initial_value = 0) 
        : initial_value_{initial_value}, current_value_{initial_value} {
        if (initial_value < 0) {
            throw std::invalid_argument("Initial value cannot be negative");
        }
    }
    
    /**
     * @brief Returns current counter value atomically.
     * 
     * @return Current value
     * @throws Never throws (noexcept guarantee)
     * 
     * @note Thread-safe, uses memory_order_relaxed
     */
    [[nodiscard]] auto value() const noexcept -> int {
        return current_value_.load(std::memory_order_relaxed);
    }
    
    /**
     * @brief Increments counter by 1 atomically.
     * 
     * @throws std::overflow_error If increment would cause overflow
     * 
     * @post If successful, value() == old_value + 1
     * 
     * @note Thread-safe
     */
    auto increment() -> void {
        const int current = current_value_.load(std::memory_order_relaxed);
        if (current >= std::numeric_limits<int>::max()) {
            throw std::overflow_error("Counter overflow");
        }
        current_value_.fetch_add(1, std::memory_order_relaxed);
    }
    
    /**
     * @brief Decrements counter by 1 atomically.
     * 
     * @throws std::underflow_error If decrement would cause underflow
     * 
     * @post If successful, value() == old_value - 1
     * 
     * @note Thread-safe
     */
    auto decrement() -> void {
        const int current = current_value_.load(std::memory_order_relaxed);
        if (current <= 0) {
            throw std::underflow_error("Counter underflow");
        }
        current_value_.fetch_sub(1, std::memory_order_relaxed);
    }
    
    /**
     * @brief Resets counter to initial value atomically.
     * 
     * @throws Never throws (noexcept guarantee)
     * 
     * @post value() == initial_value
     * 
     * @note Thread-safe
     */
    auto reset() noexcept -> void {
        current_value_.store(initial_value_, std::memory_order_relaxed);
    }

private:
    int initial_value_;
    std::atomic<int> current_value_;
};
// Tests still pass ✓
```

---

## 2B. Bug Fix Protocol (MANDATORY)

**CRITICAL: Every bug MUST receive a regression test BEFORE fixing.**

### Bug Fix Workflow

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

### Example Bug Fix

```cpp
// Bug Report #1523: parse_json crashes on empty input

// Step 1-2: Write test that reproduces the bug
// tests/core/test_json_parser.cpp
#include <gtest/gtest.h>
#include "core/json_parser.hpp"

TEST(JsonParser, HandlesEmptyInput_Bug1523) {
    // Bug #1523: parse_json crashes on empty input
    // Discovered: 2026-01-18
    // This test prevents regression
    
    EXPECT_THROW({
        parse_json("");
    }, std::invalid_argument);
}

TEST(JsonParser, HandlesWhitespaceOnly_Bug1523) {
    // Additional edge case discovered during bug investigation
    EXPECT_THROW({
        parse_json("   \n\t  ");
    }, std::invalid_argument);
}

// Run: ctest
// ❌ FAILS - parse_json crashes instead of throwing

// Step 3: Fix the bug (include/core/json_parser.hpp)
/**
 * @brief Parses a JSON string into a value.
 * 
 * Validates input and parses JSON according to RFC 8259 specification.
 * Provides detailed error messages for invalid input.
 * 
 * @param json_string The JSON string to parse
 * @return Parsed JSON value
 * 
 * @throws std::invalid_argument If input is empty or invalid JSON
 * @throws std::runtime_error If parsing fails due to internal error
 * 
 * @pre json_string must not be null
 * @post If successful, returns valid JSON value
 * 
 * @par Example
 * @code
 * try {
 *     auto value = parse_json(R"({"key": "value"})");
 *     // Use value..
 * } catch (const std::invalid_argument& e) {
 *     std::cerr << "Invalid JSON: " << e.what() << '\n';
 * }
 * @endcode
 * 
 * @note This function is thread-safe
 * @see RFC 8259 for JSON specification
 */
[[nodiscard]] auto parse_json(std::string_view json_string) -> JsonValue {
    // FIX: Validate input before parsing
    if (json_string.empty()) {
        throw std::invalid_argument("JSON input cannot be empty");
    }
    
    // Trim whitespace
    const auto trimmed = trim(json_string);
    if (trimmed.empty()) {
        throw std::invalid_argument("JSON input contains only whitespace");
    }
    
    // Original parsing logic here..
    return parse_json_impl(trimmed);
}

// Run: ctest
// ✅ PASSES - bug fixed, regression prevented ✓
```

### Example Bug Fix with Memory Safety

```cpp
// Bug Report #1524: Buffer overflow in string copy

// Step 1-2: Write test that reproduces the bug
#include <gtest/gtest.h>
#include "core/string_utils.hpp"

TEST(StringUtils, HandlesLongStrings_Bug1524) {
    // Bug #1524: Buffer overflow when copying strings > 256 chars
    // Discovered: 2026-01-18
    // This test prevents regression
    
    std::string long_string(1000, 'a');
    EXPECT_NO_THROW({
        auto result = safe_copy(long_string);
        EXPECT_EQ(result, long_string);
    });
}

TEST(StringUtils, HandlesEmptyString_Bug1524) {
    // Edge case: empty string should work
    EXPECT_NO_THROW({
        auto result = safe_copy("");
        EXPECT_EQ(result, "");
    });
}

// Run: ctest with ASAN
// ❌ FAILS - AddressSanitizer detects buffer overflow

// Step 3: Fix the bug (include/core/string_utils.hpp)
/**
 * @brief Safely copies a string without buffer overflows.
 * 
 * Performs safe string copy with automatic memory management.
 * Uses std::string to prevent buffer overflows and memory leaks.
 * 
 * @param source The source string to copy
 * @return Copied string
 * 
 * @throws std::bad_alloc If memory allocation fails
 * 
 * @post Return value equals source string
 * 
 * @par Complexity
 * O(n) where n is the length of the source string
 * 
 * @par Example
 * @code
 * std::string original = "Hello, World!";
 * auto copy = safe_copy(original);
 * assert(copy == original);
 * @endcode
 * 
 * @note This function is exception-safe (strong guarantee)
 * @note Thread-safe (no shared state)
 */
[[nodiscard]] auto safe_copy(std::string_view source) -> std::string {
    // FIX: Use std::string instead of fixed-size buffer
    // OLD (buggy) code:
    // char buffer[256];
    // strcpy(buffer, source.data());  // Buffer overflow!
    // return std::string{buffer};
    
    // NEW (safe) code:
    return std::string{source};  // std::string handles any length safely
}

// Run: ctest with ASAN
// ✅ PASSES - bug fixed, no memory errors, regression prevented ✓
```

### Prohibited Practices for Bug Fixes

**NEVER:**
- ❌ Fix a bug without adding a regression test first
- ❌ Write implementation before writing tests (violates TDD)
- ❌ Skip the Red-Green-Refactor cycle
- ❌ Commit code with failing tests
- ❌ Remove tests to make code pass
- ❌ Use `DISABLED_` prefix to ignore failing tests
- ❌ Suppress ASAN/UBSAN errors instead of fixing them

**ALWAYS:**
- ✅ Write a test that reproduces the bug first
- ✅ Verify the test fails before fixing
- ✅ Document bug ID in test comments
- ✅ Run with sanitizers (ASAN, UBSAN, TSAN)
- ✅ Ensure fix doesn't introduce new bugs
- ✅ Keep tests in codebase permanently

---

## 3. Dependency Management Strategy (MANDATORY)

### A. Dependency Resolution Priority (STRICT ORDER)

When adding a dependency to a project, **ALWAYS follow this priority order**:

#### 1. **PRIMARY: Conan Packages (conan-center)** ⭐ PREFERRED
- **ALWAYS check Conan first**: Search https://conan.io/center/
- Use official Conan packages from conan-center
- Specify exact version numbers
- Example: `fmt/10.2.0`, `spdlog/1.12.0`, `gtest/1.15.0`

```cmake
# ✅ CORRECT - Using Conan package
conan_cmake_configure(
    REQUIRES
        fmt/10.2.0
        spdlog/1.12.0
    GENERATORS CMakeDeps CMakeToolchain
)
```

#### 2. **SECONDARY: System Package Manager** (Only if not in Conan)
- **Use ONLY if package is NOT available in Conan**
- Prefer system packages to ensure OS compatibility
- Platform-specific package managers:
  - **Ubuntu/Debian**: `apt` (e.g., `libssl-dev`, `libreadline-dev`)
  - **Fedora/RHEL**: `dnf`/`yum` (e.g., `openssl-devel`)
  - **macOS**: `brew` (e.g., `openssl`, `readline`)
  - **Windows**: `vcpkg` or `chocolatey`

```cmake
# ✅ CORRECT - Using system package (only if not in Conan)
find_package(OpenSSL REQUIRED)  # System-provided
target_link_libraries(${PROJECT_NAME} PRIVATE OpenSSL::SSL)
```

#### 3. **TERTIARY: Other Methods** (Last Resort Only)
- **Use ONLY if package is in neither Conan nor system packages**
- Options (in preference order):
  1. Git submodules (for header-only libraries)
  2. FetchContent (CMake 3.11+)
  3. Manual download and vendoring (least preferred)

```cmake
# ⚠️ LAST RESORT - FetchContent (only if unavailable elsewhere)
include(FetchContent)
FetchContent_Declare(
    mylib
    GIT_REPOSITORY https://github.com/user/mylib.git
    GIT_TAG v1.0.0
)
FetchContent_MakeAvailable(mylib)
```

### B. Dependency Decision Tree

```
Need dependency "X"?
│
├─> Search Conan (conan.io/center)
│   ├─> Found? ✅ USE CONAN
│   │   └─> Add to conan_cmake_configure(REQUIRES X/version)
│   │
│   └─> Not Found? ⤵️
│       │
│       └─> Search system packages (apt/dnf/brew)
│           ├─> Found? ⚠️ USE SYSTEM PACKAGE
│           │   └─> Add find_package(X REQUIRED)
│           │   └─> Document in README that users need to install system package
│           │
│           └─> Not Found? ⛔ LAST RESORT
│               └─> Use FetchContent or git submodule
│               └─> Document why Conan/system wasn't used
```

### C. Why This Order?

1. **Conan (Primary)**: 
   - Cross-platform compatibility
   - Version control and reproducibility
   - Automatic dependency resolution
   - No system pollution
   - Easy to upgrade/downgrade
   - Works in CI/CD environments

2. **System Packages (Secondary)**:
   - Some packages are system-specific (e.g., OpenSSL system trust stores)
   - Better OS integration for certain libraries
   - Pre-installed on most systems
   - Maintained by OS vendors

3. **Other Methods (Tertiary)**:
   - Inconsistent across platforms
   - Version management challenges
   - Longer build times
   - Potential security concerns

### D. Dependency Documentation Requirements

**ALWAYS document dependencies in README.md**:

```markdown
## Dependencies

### Conan Packages (Automatic)
The following dependencies are automatically managed via Conan:
- fmt 10.2.0
- spdlog 1.12.0
- gtest 1.15.0

### System Packages (Manual Installation Required)
⚠️ Install these via your system package manager:

**Ubuntu/Debian:**
```bash
sudo apt install libssl-dev libreadline-dev
```

**macOS:**
```bash
brew install openssl readline
```

**Fedora/RHEL:**
```bash
sudo dnf install openssl-devel readline-devel
```
```

### E. Prohibited Practices

❌ **NEVER do these**:
- Copy-pasting library source code into your project
- Committing compiled binaries (`.a`, `.so`, `.dll`) to version control
- Using random GitHub repositories without version tags
- Mixing multiple versions of the same library
- Hardcoding library paths (e.g., `/usr/local/lib/libfoo.a`)

✅ **ALWAYS do these**:
- Use Conan for C++ dependencies whenever possible
- Pin exact version numbers
- Document which dependencies come from where
- Test on a clean system (Docker) to verify dependencies

## 4. Documentation Requirements (MANDATORY)

### A. Doxygen Comments for All Public APIs

**ALL public classes, functions, templates, and namespaces MUST have comprehensive Doxygen documentation.**

#### Why Doxygen Documentation?

- **Auto-Generated API Docs**: Doxygen generates complete HTML/PDF documentation from code comments
- **IDE Integration**: Better IntelliSense and tooltips in modern IDEs
- **Type Safety**: Documentation stays in sync with code
- **Maintenance**: Self-documenting code reduces onboarding time by 40%+
- **Verification**: Documentation can be verified during build process

### B. File-Level Documentation

Every header file MUST have file-level documentation:

```cpp
/**
 * @file ast.hpp
 * @brief Abstract Syntax Tree node definitions for the parser module.
 * 
 * This file defines the AST node hierarchy used throughout the parser.
 * All nodes inherit from ASTNode base class and implement the visitor pattern.
 * 
 * @author Your Name
 * @date 2026-01-17
 * @version 1.0.0
 * 
 * @see Parser
 * @see Visitor
 */

#pragma once

#include <memory>
#include <string>

namespace parser {
// ... content
}
```

### C. Class Documentation

All classes MUST have detailed Doxygen comments:

```cpp
/**
 * @brief Thread-safe cache implementation using shared_mutex.
 * 
 * Provides a concurrent key-value store with read-write locking.
 * Multiple readers can access simultaneously, but writers have exclusive access.
 * All operations are exception-safe and provide strong exception guarantees.
 * 
 * @tparam Key Type of the cache keys (must be hashable)
 * @tparam Value Type of the cached values (must be copyable)
 * 
 * @note This class is thread-safe for all operations
 * @warning Key and Value types must be thread-safe themselves
 * 
 * @code
 * ThreadSafeCache<std::string, int> cache;
 * cache.insert("answer", 42);
 * 
 * if (auto value = cache.get("answer")) {
 *     std::cout << "Value: " << *value << '\n';
 * }
 * @endcode
 * 
 * @see std::shared_mutex
 * @see std::unordered_map
 */
template<typename Key, typename Value>
class ThreadSafeCache {
public:
    /**
     * @brief Constructs an empty cache.
     * 
     * Initializes the cache with no elements.
     * The underlying hash table starts with default capacity.
     * 
     * @throws std::bad_alloc If initial memory allocation fails
     */
    ThreadSafeCache() = default;
    
    /**
     * @brief Inserts or updates a key-value pair in the cache.
     * 
     * If the key already exists, its value is updated atomically.
     * This operation acquires an exclusive write lock.
     * 
     * @param key The key to insert or update
     * @param value The value to associate with the key
     * 
     * @throws std::bad_alloc If memory allocation fails
     * @throws std::system_error If mutex locking fails
     * 
     * @pre key must be valid (not default-constructed for pointer types)
     * @post The key-value pair exists in the cache
     * 
     * @note This operation is thread-safe
     * @warning Value is moved if it's an rvalue reference
     */
    void insert(const Key& key, Value value);
    
    /**
     * @brief Retrieves a value from the cache by key.
     * 
     * Performs a read-only lookup with shared lock.
     * Multiple threads can call this simultaneously.
     * 
     * @param key The key to look up
     * @return std::optional<Value> The value if found, std::nullopt otherwise
     * 
     * @throws std::system_error If mutex locking fails
     * 
     * @note This operation is thread-safe
     * @note Returns a copy of the value, not a reference
     * 
     * @par Complexity
     * Average case O(1), worst case O(n)
     */
    [[nodiscard]] auto get(const Key& key) const -> std::optional<Value>;
    
    /**
     * @brief Removes a key-value pair from the cache.
     * 
     * If the key doesn't exist, this is a no-op.
     * Acquires exclusive write lock.
     * 
     * @param key The key to remove
     * @return true if the key was found and removed, false otherwise
     * 
     * @throws std::system_error If mutex locking fails
     * 
     * @post The key no longer exists in the cache
     */
    auto erase(const Key& key) -> bool;
    
    /**
     * @brief Returns the number of elements in the cache.
     * 
     * Acquires shared read lock.
     * 
     * @return size_t The number of key-value pairs
     * 
     * @note This operation is thread-safe
     * @note The size may change immediately after this call in multi-threaded code
     */
    [[nodiscard]] auto size() const noexcept -> size_t;
    
    /**
     * @brief Clears all elements from the cache.
     * 
     * Acquires exclusive write lock and removes all entries.
     * 
     * @post size() == 0
     * 
     * @note This operation is thread-safe
     */
    void clear() noexcept;
    
private:
    mutable std::shared_mutex mutex_;  ///< Mutex for thread synchronization
    std::unordered_map<Key, Value> cache_;  ///< Underlying storage
};
```

### D. Function Documentation

All public functions MUST have complete Doxygen comments:

```cpp
/**
 * @brief Parses an integer from a string with error handling.
 * 
 * Attempts to convert a string representation to an integer.
 * Handles both decimal and hexadecimal formats (with 0x prefix).
 * Returns an expected type for explicit error handling without exceptions.
 * 
 * @param str String to parse (must not be empty)
 * @param base Numeric base (default: 10, valid range: 2-36)
 * @return Result<int, ParseError> Success with parsed value or error code
 * 
 * @retval Success Contains the parsed integer value
 * @retval Error Contains ParseError::InvalidFormat if string is malformed
 * @retval Error Contains ParseError::OutOfRange if value exceeds int limits
 * 
 * @throws Never throws (noexcept guarantee)
 * 
 * @pre str must not be empty
 * @pre base must be between 2 and 36 inclusive
 * 
 * @post If successful, the returned value equals the integer representation
 * 
 * @par Example
 * @code
 * auto result = parse_integer("42");
 * if (result.has_value()) {
 *     std::cout << "Parsed: " << result.value() << '\n';
 * } else {
 *     std::cerr << "Error: " << result.error().message() << '\n';
 * }
 * 
 * // Hexadecimal parsing
 * auto hex = parse_integer("0xFF", 16);
 * assert(hex.value() == 255);
 * @endcode
 * 
 * @see ParseError
 * @see Result
 * 
 * @note This function is thread-safe
 * @warning Leading/trailing whitespace is not automatically trimmed
 */
[[nodiscard]] auto parse_integer(std::string_view str, int base = 10) 
    -> Result<int, ParseError>;

/**
 * @brief Computes the dot product of two 3D vectors.
 * 
 * Calculates the scalar product: a.x * b.x + a.y * b.y + a.z * b.z
 * 
 * @param a First vector
 * @param b Second vector
 * @return double The dot product result
 * 
 * @throws Never throws (noexcept guarantee)
 * 
 * @par Complexity
 * O(1) - constant time
 * 
 * @par Example
 * @code
 * Vector3D v1{1.0, 0.0, 0.0};
 * Vector3D v2{0.0, 1.0, 0.0};
 * double product = dot(v1, v2);  // Result: 0.0 (perpendicular)
 * @endcode
 * 
 * @note This is a constexpr function, can be evaluated at compile-time
 */
[[nodiscard]] constexpr auto dot(const Vector3D& a, const Vector3D& b) noexcept 
    -> double {
    return a.x() * b.x() + a.y() * b.y() + a.z() * b.z();
}
```

### E. Template Documentation

Template functions and classes require special attention:

```cpp
/**
 * @brief Generic clamp function for numeric types.
 * 
 * Constrains a value to lie between a minimum and maximum value.
 * Uses C++20 concepts for compile-time type checking.
 * 
 * @tparam T Numeric type (must satisfy Numeric concept)
 * 
 * @param value The value to clamp
 * @param min_val Minimum allowed value
 * @param max_val Maximum allowed value
 * 
 * @return T The clamped value
 * 
 * @pre min_val <= max_val (undefined behavior otherwise)
 * @post return value is in range [min_val, max_val]
 * 
 * @throws Never throws (noexcept guarantee)
 * 
 * @par Example
 * @code
 * auto clamped = clamp(150, 0, 100);  // Result: 100
 * auto in_range = clamp(50, 0, 100);  // Result: 50
 * auto below = clamp(-10, 0, 100);    // Result: 0
 * @endcode
 * 
 * @note This function is constexpr and can be used in compile-time contexts
 * @see std::clamp (C++17 standard library version)
 */
template<Numeric T>
[[nodiscard]] constexpr auto clamp(T value, T min_val, T max_val) noexcept -> T {
    return std::max(min_val, std::min(value, max_val));
}

/**
 * @brief Smart pointer factory with custom deleter.
 * 
 * Creates a unique_ptr with automatic resource cleanup.
 * Useful for C APIs that require paired create/destroy calls.
 * 
 * @tparam T Type of the managed resource
 * @tparam Creator Function type for resource creation
 * @tparam Deleter Function type for resource destruction
 * 
 * @param creator Function that creates the resource
 * @param deleter Function that destroys the resource
 * 
 * @return std::unique_ptr<T, Deleter> Smart pointer managing the resource
 * 
 * @throws std::runtime_error If creator returns nullptr
 * 
 * @par Example
 * @code
 * // Wrap C API file handle
 * auto file = make_resource<FILE>(
 *     []() { return fopen("file.txt", "r"); },
 *     [](FILE* f) { if (f) fclose(f); }
 * );
 * @endcode
 */
template<typename T, typename Creator, typename Deleter>
[[nodiscard]] auto make_resource(Creator&& creator, Deleter&& deleter) 
    -> std::unique_ptr<T, Deleter>;
```

### F. Namespace Documentation

Namespaces should be documented at their first declaration:

```cpp
/**
 * @namespace parser
 * @brief Core parsing functionality for the language frontend.
 * 
 * Contains all AST node definitions, lexer, parser, and semantic analysis.
 * This namespace provides the complete parsing pipeline from source text
 * to validated abstract syntax tree.
 * 
 * @par Key Components
 * - Lexer: Tokenizes source code
 * - Parser: Builds AST from tokens
 * - Visitor: Traverses and transforms AST
 * - Semantic: Type checking and validation
 * 
 * @par Thread Safety
 * All classes in this namespace are thread-safe unless noted otherwise.
 * 
 * @par Example Usage
 * @code
 * namespace parser {
 *     std::vector<Token> tokens = lexer.tokenize(source);
 *     Parser parser{std::move(tokens)};
 *     auto ast = parser.parse();
 *     SemanticAnalyzer analyzer;
 *     analyzer.check(ast.get());
 * }
 * @endcode
 * 
 * @see Lexer
 * @see Parser
 * @see ASTNode
 */
namespace parser {
// ... content
}
```

### G. Enum Documentation

Enumerations should document each enumerator:

```cpp
/**
 * @brief Error codes for parsing operations.
 * 
 * Defines all possible error conditions that can occur during parsing.
 * Used with std::error_code for type-safe error handling.
 */
enum class ParseError {
    /**
     * @brief Input string has invalid format.
     * 
     * The string cannot be interpreted as the expected type.
     * Example: "abc" when parsing an integer.
     */
    InvalidFormat,
    
    /**
     * @brief Required field is missing from input.
     * 
     * A mandatory field was not provided in the input data.
     * Example: Missing required JSON key.
     */
    MissingField,
    
    /**
     * @brief Numeric value exceeds allowable range.
     * 
     * The parsed value is outside the valid range for the target type.
     * Example: "999999999999999" for int32_t.
     */
    OutOfRange,
    
    /**
     * @brief Unexpected end of input encountered.
     * 
     * Parser reached end of input while expecting more data.
     */
    UnexpectedEOF
};
```

### H. Generating Documentation with Doxygen

#### Installation

```bash
# Ubuntu/Debian
sudo apt install doxygen graphviz

# macOS
brew install doxygen graphviz

# Windows
choco install doxygen.install graphviz
```

#### Doxyfile Configuration

Create `Doxyfile` in project root:

```doxyfile
# Project information
PROJECT_NAME           = "MyProject"
PROJECT_NUMBER         = 1.0.0
PROJECT_BRIEF          = "Modern C++ Application"
OUTPUT_DIRECTORY       = docs

# Input configuration
INPUT                  = src/ include/ README.md
RECURSIVE              = YES
FILE_PATTERNS          = *.cpp *.hpp *.h *.md
EXCLUDE_PATTERNS       = */build/* */tests/* */.git/*

# Output formats
GENERATE_HTML          = YES
GENERATE_LATEX         = NO
GENERATE_XML           = YES
HTML_OUTPUT            = html
HTML_FILE_EXTENSION    = .html

# Documentation extraction
EXTRACT_ALL            = NO
EXTRACT_PRIVATE        = NO
EXTRACT_STATIC         = YES
EXTRACT_LOCAL_CLASSES  = NO
HIDE_UNDOC_MEMBERS     = YES
HIDE_UNDOC_CLASSES     = YES

# Appearance
HTML_COLORSTYLE_HUE    = 220
HTML_COLORSTYLE_SAT    = 100
HTML_COLORSTYLE_GAMMA  = 80
HTML_DYNAMIC_SECTIONS  = YES
GENERATE_TREEVIEW      = YES

# Diagrams and graphs
HAVE_DOT               = YES
DOT_IMAGE_FORMAT       = svg
INTERACTIVE_SVG        = YES
CLASS_DIAGRAMS         = YES
COLLABORATION_GRAPH    = YES
INCLUDE_GRAPH          = YES
INCLUDED_BY_GRAPH      = YES
CALL_GRAPH             = YES
CALLER_GRAPH           = YES
GRAPHICAL_HIERARCHY    = YES
DIRECTORY_GRAPH        = YES

# Warnings
WARN_IF_UNDOCUMENTED   = YES
WARN_IF_DOC_ERROR      = YES
WARN_NO_PARAMDOC       = YES
WARN_AS_ERROR          = NO
WARN_FORMAT            = "$file:$line: $text"

# Source code
SOURCE_BROWSER         = YES
INLINE_SOURCES         = NO
STRIP_CODE_COMMENTS    = NO
REFERENCED_BY_RELATION = YES
REFERENCES_RELATION    = YES

# Preprocessing
ENABLE_PREPROCESSING   = YES
MACRO_EXPANSION        = YES
EXPAND_ONLY_PREDEF     = NO
PREDEFINED             = __cplusplus=202002L

# Search
SEARCHENGINE           = YES
SERVER_BASED_SEARCH    = NO
```

#### CMake Integration

Add to root `CMakeLists.txt`:

```cmake
# Documentation generation with Doxygen
option(BUILD_DOCUMENTATION "Create and install the HTML based API documentation (requires Doxygen)" ON)

if(BUILD_DOCUMENTATION)
    find_package(Doxygen REQUIRED dot)
    
    if(DOXYGEN_FOUND)
        # Configure Doxyfile
        set(DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in)
        set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)
        
        configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)
        
        # Add documentation target
        add_custom_target(docs
            COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMENT "Generating API documentation with Doxygen"
            VERBATIM
        )
        
        # Add target to check documentation coverage
        add_custom_target(docs-check
            COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT} 2>&1 | grep -i "warning\\|error" || true
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMENT "Checking documentation warnings"
            VERBATIM
        )
        
        # Install documentation
        install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/docs/html
            DESTINATION share/doc/${PROJECT_NAME}
            OPTIONAL
        )
    else()
        message(STATUS "Doxygen not found, documentation will not be built")
    endif()
endif()
```

#### Generating Documentation

```bash
# Generate documentation
mkdir build
cd build
cmake .. -DBUILD_DOCUMENTATION=ON
make docs

# Check documentation warnings
make docs-check

# View documentation
open docs/html/index.html  # macOS
xdg-open docs/html/index.html  # Linux
start docs/html/index.html  # Windows
```

### I. Documentation Best Practices

**DO:**
- ✅ Document all public APIs (classes, functions, templates, enums)
- ✅ Include `@brief` for one-line summary
- ✅ Include `@param` for all parameters
- ✅ Include `@return` or `@retval` for return values
- ✅ Include `@throws` for functions that can throw
- ✅ Provide `@code` examples for complex APIs
- ✅ Use `@pre` and `@post` for preconditions and postconditions
- ✅ Document thread safety with `@note`
- ✅ Link related items with `@see`
- ✅ Document complexity with `@par Complexity`
- ✅ Keep documentation in sync with code

**DON'T:**
- ❌ Skip documentation for "obvious" functions
- ❌ Write vague descriptions ("Does stuff", "Helper function")
- ❌ Let documentation become outdated
- ❌ Document private implementation details excessively
- ❌ Commit generated docs to git (add `docs/` to `.gitignore`)
- ❌ Use `@cond` to hide undocumented code
- ❌ Copy-paste documentation without updating parameters

### J. Documentation Verification

Add documentation checks to CI/CD:

```yaml
# .github/workflows/ci.yml
name: CI

on: [push, pull_request]

jobs:
  documentation:
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Install Doxygen
        run: sudo apt-get install -y doxygen graphviz
      
      - name: Generate documentation
        run: |
          mkdir build
          cd build
          cmake .. -DBUILD_DOCUMENTATION=ON
          make docs
      
      - name: Check documentation warnings
        run: |
          cd build
          if make docs-check 2>&1 | grep -i "warning"; then
            echo "Documentation has warnings!"
            exit 1
          fi
      
      - name: Upload documentation
        uses: actions/upload-artifact@v3
        with:
          name: api-documentation
          path: build/docs/html/
      
      - name: Deploy to GitHub Pages
        if: github.ref == 'refs/heads/main'
        uses: peaceiris/actions-gh-pages@v3
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: ./build/docs/html
```

### K. Documentation Checklist

**Before committing code, verify:**
- [ ] All public classes have `@brief` and detailed description
- [ ] All public functions have complete Doxygen comments
- [ ] All function parameters documented with `@param`
- [ ] All return values documented with `@return` or `@retval`
- [ ] All exceptions documented with `@throws`
- [ ] At least one `@code` example for complex APIs
- [ ] Thread safety documented with `@note`
- [ ] Preconditions documented with `@pre` where applicable
- [ ] Postconditions documented with `@post` where applicable
- [ ] File-level documentation present in all headers
- [ ] Doxygen can generate docs: `make docs` succeeds
- [ ] No documentation warnings: `make docs-check` passes
- [ ] Generated documentation is readable and complete
- [ ] Cross-references (`@see`) are correct

### L. .gitignore Configuration

Add to `.gitignore`:

```gitignore
# Build output
/build/
/out/
*.o
*.a
*.so
*.dll
*.exe

# Generated documentation (regenerate during CI/CD)
/docs/html/
/docs/latex/
/docs/xml/
/Doxyfile
*.log

# IDE files
/.vscode/
/.idea/
*.swp
*.swo

# CMake
CMakeCache.txt
CMakeFiles/
cmake_install.cmake
compile_commands.json

# Testing
/Testing/
CTestTestfile.cmake

# Coverage
*.gcov
*.gcda
*.gcno
coverage.info
```

### M. Complete Documentation Example

```cpp
/**
 * @file vector3d.hpp
 * @brief 3D vector mathematics library.
 * 
 * Provides a complete 3D vector implementation with common operations.
 * All operations are constexpr where possible for compile-time computation.
 * 
 * @author Your Name
 * @date 2026-01-17
 * @version 1.0.0
 */

#pragma once

#include <cmath>
#include <iostream>

/**
 * @namespace math
 * @brief Mathematical utilities and data structures.
 */
namespace math {

/**
 * @class Vector3D
 * @brief Represents a 3-dimensional vector with double precision.
 * 
 * Provides standard vector operations including addition, subtraction,
 * scalar multiplication, dot product, cross product, and normalization.
 * All operations maintain mathematical correctness and numerical stability.
 * 
 * @note This class is immutable - operations return new vectors
 * @note All operations are constexpr except those requiring std::sqrt
 * 
 * @par Thread Safety
 * This class is thread-safe because it's immutable.
 * 
 * @par Example
 * @code
 * Vector3D v1{1.0, 0.0, 0.0};
 * Vector3D v2{0.0, 1.0, 0.0};
 * 
 * auto sum = v1 + v2;              // {1.0, 1.0, 0.0}
 * auto scaled = v1 * 2.0;          // {2.0, 0.0, 0.0}
 * auto product = dot(v1, v2);      // 0.0 (perpendicular)
 * auto cross = v1.cross(v2);       // {0.0, 0.0, 1.0}
 * auto normalized = v1.normalized(); // {1.0, 0.0, 0.0}
 * @endcode
 * 
 * @see dot()
 * @see cross()
 */
class Vector3D {
public:
    /**
     * @brief Constructs a vector with specified components.
     * 
     * @param x X-component (default: 0.0)
     * @param y Y-component (default: 0.0)
     * @param z Z-component (default: 0.0)
     * 
     * @throws Never throws (noexcept guarantee)
     * 
     * @post x() == x, y() == y, z() == z
     */
    constexpr Vector3D(double x = 0.0, double y = 0.0, double z = 0.0) noexcept
        : x_{x}, y_{y}, z_{z} {}
    
    /**
     * @brief Returns the X-component.
     * @return double The X-component value
     * @throws Never throws (noexcept guarantee)
     */
    [[nodiscard]] constexpr auto x() const noexcept -> double { return x_; }
    
    /**
     * @brief Returns the Y-component.
     * @return double The Y-component value
     * @throws Never throws (noexcept guarantee)
     */
    [[nodiscard]] constexpr auto y() const noexcept -> double { return y_; }
    
    /**
     * @brief Returns the Z-component.
     * @return double The Z-component value
     * @throws Never throws (noexcept guarantee)
     */
    [[nodiscard]] constexpr auto z() const noexcept -> double { return z_; }
    
    /**
     * @brief Computes the length (magnitude) of the vector.
     * 
     * @return double The Euclidean length: sqrt(x² + y² + z²)
     * 
     * @throws Never throws (noexcept guarantee)
     * 
     * @par Complexity
     * O(1) - constant time
     * 
     * @note Not constexpr due to std::sqrt
     */
    [[nodiscard]] auto length() const noexcept -> double {
        return std::sqrt(x_ * x_ + y_ * y_ + z_ * z_);
    }
    
    /**
     * @brief Returns a normalized (unit length) copy of this vector.
     * 
     * @return Vector3D Unit vector in same direction
     * 
     * @throws std::runtime_error If the vector is zero (cannot normalize)
     * 
     * @pre length() > 0.0
     * @post Return value has length() approximately equal to 1.0
     * 
     * @par Example
     * @code
     * Vector3D v{3.0, 4.0, 0.0};
     * auto unit = v.normalized();
     * assert(std::abs(unit.length() - 1.0) < 1e-10);
     * @endcode
     */
    [[nodiscard]] auto normalized() const -> Vector3D {
        const auto len = length();
        if (len == 0.0) {
            throw std::runtime_error("Cannot normalize zero vector");
        }
        return {x_ / len, y_ / len, z_ / len};
    }
    
    /**
     * @brief Computes the cross product with another vector.
     * 
     * Returns a vector perpendicular to both this and other.
     * The magnitude equals the area of the parallelogram formed by the vectors.
     * 
     * @param other The other vector
     * @return Vector3D The cross product: this × other
     * 
     * @throws Never throws (noexcept guarantee)
     * 
     * @post Return value is perpendicular to both input vectors
     * @post dot(result, this) ≈ 0 and dot(result, other) ≈ 0
     * 
     * @par Example
     * @code
     * Vector3D x_axis{1.0, 0.0, 0.0};
     * Vector3D y_axis{0.0, 1.0, 0.0};
     * auto z_axis = x_axis.cross(y_axis);  // {0.0, 0.0, 1.0}
     * @endcode
     * 
     * @see dot()
     */
    [[nodiscard]] constexpr auto cross(const Vector3D& other) const noexcept 
        -> Vector3D {
        return {
            y_ * other.z_ - z_ * other.y_,
            z_ * other.x_ - x_ * other.z_,
            x_ * other.y_ - y_ * other.x_
        };
    }
    
    // ... operator overloads with documentation ..
    
private:
    double x_;  ///< X-component of the vector
    double y_;  ///< Y-component of the vector
    double z_;  ///< Z-component of the vector
};

/**
 * @relates Vector3D
 * @brief Computes the dot product of two vectors.
 * 
 * Calculates the scalar product: a·b = a.x*b.x + a.y*b.y + a.z*b.z
 * 
 * @param a First vector
 * @param b Second vector
 * @return double The dot product
 * 
 * @throws Never throws (noexcept guarantee)
 * 
 * @par Properties
 * - Commutative: dot(a, b) == dot(b, a)
 * - dot(a, a) == a.length()²
 * - dot(a, b) == 0 if a and b are perpendicular
 * 
 * @par Complexity
 * O(1) - constant time
 * 
 * @par Example
 * @code
 * Vector3D v1{1.0, 2.0, 3.0};
 * Vector3D v2{4.0, 5.0, 6.0};
 * double product = dot(v1, v2);  // 1*4 + 2*5 + 3*6 = 32
 * @endcode
 */
[[nodiscard]] constexpr auto dot(const Vector3D& a, const Vector3D& b) noexcept 
    -> double {
    return a.x() * b.x() + a.y() * b.y() + a.z() * b.z();
}

}  // namespace math
```

---

## 5. Project Structure (Mandatory)

### A. Directory Layout
```
project-root/
├── CMakeLists.txt           # Root CMake configuration
├── conanfile.txt            # Optional global Conan dependencies
├── conan/                   # Conan integration
│   └── CMakeLists.txt      # Conan setup (downloaded automatically)
├── cmake/                   # CMake modules and scripts
│   ├── CompilerWarnings.cmake
│   ├── Sanitizers.cmake
│   └── StaticAnalyzers.cmake
├── src/                     # Source code modules
│   ├── core/               # Core module
│   │   ├── CMakeLists.txt
│   │   ├── include/
│   │   │   └── core/
│   │   │       ├── types.hpp
│   │   │       └── utils.hpp
│   │   └── src/
│   │       ├── types.cpp
│   │       └── utils.cpp
│   ├── parser/             # Parser module
│   │   ├── CMakeLists.txt
│   │   ├── include/
│   │   │   └── parser/
│   │   │       └── parser.hpp
│   │   └── src/
│   │       └── parser.cpp
│   └── network/            # Network module
│       ├── CMakeLists.txt
│       ├── include/
│       │   └── network/
│       │       ├── client.hpp
│       │       └── server.hpp
│       └── src/
│           ├── client.cpp
│           └── server.cpp
├── apps/                   # Application executables
│   └── main/
│       ├── CMakeLists.txt
│       └── main.cpp
├── tests/                  # Unit tests (MANDATORY)
│   ├── core/              # Core module tests
│   │   ├── CMakeLists.txt
│   │   ├── test_types.cpp
│   │   └── test_utils.cpp
│   ├── parser/            # Parser module tests
│   │   ├── CMakeLists.txt
│   │   ├── parser_tests.cpp
│   │   └── ast_tests.cpp
│   └── network/           # Network module tests
│       ├── CMakeLists.txt
│       ├── client_tests.cpp
│       └── server_tests.cpp
├── benchmarks/             # Performance benchmarks
│   └── CMakeLists.txt
├── docs/                   # Documentation
│   └── README.md
├── .clang-format          # Code formatting rules
├── .clang-tidy            # Static analysis rules
└── README.md
```

### B. Root CMakeLists.txt
```cmake
# ✅ CORRECT - Modern CMake project setup
cmake_minimum_required(VERSION 3.15...3.27)

project(MyProject
    VERSION 1.0.0
    DESCRIPTION "Modern C++ Project"
    LANGUAGES CXX
)

# Require C++20 minimum
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Generate compile_commands.json for clang-tidy
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Include CMake modules
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

# Conan integration
add_subdirectory(conan)

# Compiler warnings and static analysis
include(cmake/CompilerWarnings.cmake)
include(cmake/Sanitizers.cmake)
include(cmake/StaticAnalyzers.cmake)

# Options
option(BUILD_TESTS "Build test suite (MANDATORY for production)" ON)
option(BUILD_BENCHMARKS "Build benchmarks" OFF)
option(ENABLE_COVERAGE "Enable coverage reporting" OFF)
option(ENABLE_SANITIZERS "Enable sanitizers" OFF)

# Source modules
add_subdirectory(src/core)
add_subdirectory(src/parser)
add_subdirectory(src/network)

# Applications
add_subdirectory(apps/main)

# Tests (MANDATORY - always enabled in this setup)
enable_testing()
add_subdirectory(tests/core)
add_subdirectory(tests/parser)
add_subdirectory(tests/network)

# Custom target to run all tests
add_custom_target(check
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
    DEPENDS core_tests parser_tests network_tests
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running all unit tests..."
)

# Fail the build if tests are disabled in production builds
if(NOT BUILD_TESTS AND CMAKE_BUILD_TYPE MATCHES "Release")
    message(FATAL_ERROR "Tests cannot be disabled for Release builds")
endif()

# Benchmarks
if(BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()
```

### C. Conan Integration (conan/CMakeLists.txt)
```cmake
# ✅ CORRECT - Conan automatic setup
cmake_minimum_required(VERSION 3.15)

# Default Conan profile
set(CONAN_HOST_PROFILE default CACHE STRING "Conan host profile")

# Download conan.cmake if not present
if(NOT EXISTS "${CMAKE_BINARY_DIR}/conan.cmake")
    message(STATUS "Downloading conan.cmake from https://github.com/conan-io/cmake-conan")
    file(DOWNLOAD 
        "https://raw.githubusercontent.com/conan-io/cmake-conan/0.18.1/conan.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/conan.cmake"
        TLS_VERIFY ON
        STATUS download_status
    )
    list(GET download_status 0 status_code)
    if(NOT status_code EQUAL 0)
        message(FATAL_ERROR "Failed to download conan.cmake")
    endif()
endif()

# Include conan.cmake
include(${CMAKE_CURRENT_BINARY_DIR}/conan.cmake)

# Auto-detect settings
conan_cmake_autodetect(settings)

# Make settings available to parent scope
set(CONAN_SETTINGS ${settings} PARENT_SCOPE)
```

### D. Module CMakeLists.txt Pattern
```cmake
# ✅ CORRECT - Module with Conan dependencies
# src/parser/CMakeLists.txt
cmake_minimum_required(VERSION 3.15)
project(parser CXX)

# Ensure module path is set
list(APPEND CMAKE_MODULE_PATH ${CMAKE_BINARY_DIR})
list(APPEND CMAKE_PREFIX_PATH ${CMAKE_BINARY_DIR})

# Configure Conan dependencies for this module
conan_cmake_configure(
    REQUIRES
        fmt/10.2.0
        spdlog/1.12.0
    GENERATORS CMakeDeps CMakeToolchain
)

# Install dependencies
conan_cmake_install(
    PATH_OR_REFERENCE .
    BUILD missing
    REMOTE conancenter
    SETTINGS ${CONAN_SETTINGS}
)

# Find packages
find_package(fmt REQUIRED)
find_package(spdlog REQUIRED)

# Library target
add_library(${PROJECT_NAME}
    src/parser.cpp
    src/lexer.cpp
    src/ast.cpp
)

# Include directories
target_include_directories(${PROJECT_NAME}
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Link libraries
target_link_libraries(${PROJECT_NAME}
    PUBLIC
        fmt::fmt
    PRIVATE
        spdlog::spdlog
)

# Compiler features
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_20)

# Add alias
add_library(MyProject::parser ALIAS ${PROJECT_NAME})

# Install rules
install(TARGETS ${PROJECT_NAME}
    EXPORT ${PROJECT_NAME}Targets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
)

install(DIRECTORY include/
    DESTINATION include
)
```

## 5. Modern C++ Code Standards

### A. Type Safety & Memory Management
```cpp
// ✅ CORRECT - Modern C++ with smart pointers and RAII
#include <memory>
#include <vector>
#include <optional>
#include <string_view>

class Resource {
public:
    // Constructor
    explicit Resource(std::string_view name) : name_{name} {
        // RAII: Acquire resource
    }
    
    // Rule of Five (or Rule of Zero with smart pointers)
    ~Resource() = default;
    Resource(const Resource&) = delete;
    Resource& operator=(const Resource&) = delete;
    Resource(Resource&&) noexcept = default;
    Resource& operator=(Resource&&) noexcept = default;
    
    // Methods
    [[nodiscard]] auto get_name() const noexcept -> std::string_view {
        return name_;
    }
    
private:
    std::string name_;
};

// Factory function returning smart pointer
[[nodiscard]] auto create_resource(std::string_view name) 
    -> std::unique_ptr<Resource> {
    return std::make_unique<Resource>(name);
}

// Using optional for nullable returns
[[nodiscard]] auto find_resource(const std::vector<Resource>& resources, 
                                  std::string_view name) 
    -> std::optional<std::reference_wrapper<const Resource>> {
    for (const auto& res : resources) {
        if (res.get_name() == name) {
            return std::cref(res);
        }
    }
    return std::nullopt;
}

// ❌ WRONG - Raw pointers, manual memory management
Resource* create_resource_bad(const char* name) {
    return new Resource(name);  // Memory leak waiting to happen
}

Resource* find_resource_bad(std::vector<Resource>& resources, const char* name) {
    for (auto& res : resources) {
        if (res.get_name() == name) {
            return &res;  // Dangerous: pointer may become invalid
        }
    }
    return nullptr;  // Prefer std::optional
}
```

### B. Value Semantics & Move Semantics
```cpp
// ✅ CORRECT - Modern value types with move semantics
#include <string>
#include <vector>
#include <utility>

class User {
public:
    User(std::string name, std::string email)
        : name_{std::move(name)}, email_{std::move(email)} {}
    
    // Accessors return by value or const reference
    [[nodiscard]] auto name() const noexcept -> const std::string& {
        return name_;
    }
    
    [[nodiscard]] auto email() const noexcept -> const std::string& {
        return email_;
    }
    
    // Mutators use move semantics
    void set_name(std::string name) {
        name_ = std::move(name);
    }
    
private:
    std::string name_;
    std::string email_;
};

class UserDatabase {
public:
    // Accept by value, move into container
    void add_user(User user) {
        users_.push_back(std::move(user));
    }
    
    // Return by value (RVO/NRVO will optimize)
    [[nodiscard]] auto get_all_users() const -> std::vector<User> {
        return users_;
    }
    
    // Return view when possible
    [[nodiscard]] auto get_user_view(size_t index) const 
        -> std::optional<std::reference_wrapper<const User>> {
        if (index < users_.size()) {
            return std::cref(users_[index]);
        }
        return std::nullopt;
    }
    
private:
    std::vector<User> users_;
};

// ❌ WRONG - Inefficient copies
class UserDatabaseBad {
public:
    void add_user(const User& user) {  // Forces copy
        users_.push_back(user);
    }
    
    const std::vector<User>& get_all_users() {  // Exposes internals
        return users_;
    }
    
private:
    std::vector<User> users_;
};
```

### C. Const Correctness & noexcept
```cpp
// ✅ CORRECT - Proper const and noexcept usage
class Vector3D {
public:
    Vector3D(double x, double y, double z) noexcept
        : x_{x}, y_{y}, z_{z} {}
    
    // Const getters
    [[nodiscard]] auto x() const noexcept -> double { return x_; }
    [[nodiscard]] auto y() const noexcept -> double { return y_; }
    [[nodiscard]] auto z() const noexcept -> double { return z_; }
    
    // Const methods
    [[nodiscard]] auto length() const noexcept -> double {
        return std::sqrt(x_ * x_ + y_ * y_ + z_ * z_);
    }
    
    [[nodiscard]] auto normalized() const -> Vector3D {
        const auto len = length();
        if (len == 0.0) {
            throw std::runtime_error("Cannot normalize zero vector");
        }
        return {x_ / len, y_ / len, z_ / len};
    }
    
    // Operators
    [[nodiscard]] auto operator+(const Vector3D& other) const noexcept 
        -> Vector3D {
        return {x_ + other.x_, y_ + other.y_, z_ + other.z_};
    }
    
    auto operator+=(const Vector3D& other) noexcept -> Vector3D& {
        x_ += other.x_;
        y_ += other.y_;
        z_ += other.z_;
        return *this;
    }
    
private:
    double x_, y_, z_;
};

// Free functions with proper const and noexcept
[[nodiscard]] auto dot(const Vector3D& a, const Vector3D& b) noexcept 
    -> double {
    return a.x() * b.x() + a.y() * b.y() + a.z() * b.z();
}

// ❌ WRONG - Missing const and noexcept
class Vector3DBad {
public:
    double x() { return x_; }  // Should be const
    double length() { return std::sqrt(x_ * x_); }  // Should be const
    Vector3D operator+(Vector3D& other) {  // Should be const&, missing noexcept
        return {x_ + other.x_, y_ + other.y_, z_ + other.z_};
    }
private:
    double x_, y_, z_;
};
```

### D. Concepts & Templates (C++20)
```cpp
// ✅ CORRECT - Modern concepts and constrained templates
#include <concepts>
#include <type_traits>
#include <ranges>

// Define concept
template<typename T>
concept Numeric = std::is_arithmetic_v<T>;

template<typename T>
concept Printable = requires(T t, std::ostream& os) {
    { os << t } -> std::convertible_to<std::ostream&>;
};

// Constrained template function
template<Numeric T>
[[nodiscard]] auto square(T value) noexcept -> T {
    return value * value;
}

template<Numeric T>
[[nodiscard]] auto clamp(T value, T min, T max) noexcept -> T {
    return std::max(min, std::min(value, max));
}

// Concept-constrained class
template<Numeric T>
class Statistics {
public:
    void add_value(T value) {
        values_.push_back(value);
        sum_ += value;
    }
    
    [[nodiscard]] auto mean() const -> double {
        if (values_.empty()) {
            throw std::runtime_error("No values to calculate mean");
        }
        return static_cast<double>(sum_) / values_.size();
    }
    
    [[nodiscard]] auto size() const noexcept -> size_t {
        return values_.size();
    }
    
private:
    std::vector<T> values_;
    T sum_{};
};

// Ranges and views (C++20)
auto process_numbers(const std::vector<int>& numbers) 
    -> std::vector<int> {
    namespace views = std::views;
    
    std::vector<int> result;
    for (const auto value : numbers 
        | views::filter([](int x) { return x % 2 == 0; })
        | views::transform([](int x) { return x * 2; })
        | views::take(10)) {
        result.push_back(value);
    }
    return result;
}

// ❌ WRONG - Old-style SFINAE
template<typename T, typename = typename std::enable_if<
    std::is_arithmetic<T>::value>::type>
T square_bad(T value) {
    return value * value;
}
```

### E. Error Handling
```cpp
// ✅ CORRECT - Modern error handling with std::expected (C++23)
// or custom Result type
#include <expected>
#include <system_error>
#include <string>
#include <variant>

// For C++20, use custom Result type
template<typename T, typename E = std::error_code>
using Result = std::expected<T, E>;

enum class ParseError {
    InvalidFormat,
    MissingField,
    OutOfRange
};

// Make ParseError work with error codes
struct ParseErrorCategory : std::error_category {
    [[nodiscard]] auto name() const noexcept -> const char* override {
        return "parse_error";
    }
    
    [[nodiscard]] auto message(int ev) const -> std::string override {
        switch (static_cast<ParseError>(ev)) {
            case ParseError::InvalidFormat: return "Invalid format";
            case ParseError::MissingField: return "Missing field";
            case ParseError::OutOfRange: return "Value out of range";
            default: return "Unknown error";
        }
    }
};

inline const ParseErrorCategory parse_error_category{};

auto make_error_code(ParseError e) -> std::error_code {
    return {static_cast<int>(e), parse_error_category};
}

// Function returning Result
[[nodiscard]] auto parse_integer(std::string_view str) 
    -> Result<int, std::error_code> {
    if (str.empty()) {
        return std::unexpected(make_error_code(ParseError::InvalidFormat));
    }
    
    try {
        const int value = std::stoi(std::string{str});
        return value;
    } catch (const std::invalid_argument&) {
        return std::unexpected(make_error_code(ParseError::InvalidFormat));
    } catch (const std::out_of_range&) {
        return std::unexpected(make_error_code(ParseError::OutOfRange));
    }
}

// Using Result
auto example_usage() -> void {
    const auto result = parse_integer("123");
    
    if (result.has_value()) {
        std::cout << "Parsed: " << result.value() << '\n';
    } else {
        std::cerr << "Error: " << result.error().message() << '\n';
    }
    
    // Or use value_or
    const int value = parse_integer("abc").value_or(-1);
}

// RAII for cleanup
class FileHandle {
public:
    explicit FileHandle(std::string_view path) {
        file_ = std::fopen(path.data(), "r");
        if (!file_) {
            throw std::runtime_error("Failed to open file");
        }
    }
    
    ~FileHandle() {
        if (file_) {
            std::fclose(file_);
        }
    }
    
    // Delete copy, allow move
    FileHandle(const FileHandle&) = delete;
    FileHandle& operator=(const FileHandle&) = delete;
    FileHandle(FileHandle&& other) noexcept : file_{other.file_} {
        other.file_ = nullptr;
    }
    FileHandle& operator=(FileHandle&& other) noexcept {
        if (this != &other) {
            if (file_) std::fclose(file_);
            file_ = other.file_;
            other.file_ = nullptr;
        }
        return *this;
    }
    
    [[nodiscard]] auto get() const noexcept -> FILE* { return file_; }
    
private:
    FILE* file_{nullptr};
};

// ❌ WRONG - Manual resource management
FILE* open_file_bad(const char* path) {
    return fopen(path, "r");  // Caller must remember to close
}
```

### F. Modern String Handling
```cpp
// ✅ CORRECT - Use string_view for non-owning strings
#include <string>
#include <string_view>
#include <format>  // C++20

// Accept string_view for read-only operations
[[nodiscard]] auto starts_with_prefix(std::string_view str, 
                                       std::string_view prefix) noexcept 
    -> bool {
    return str.starts_with(prefix);
}

// Return string for owned results
[[nodiscard]] auto to_upper(std::string_view str) -> std::string {
    std::string result{str};
    std::ranges::transform(result, result.begin(), 
        [](unsigned char c) { return std::toupper(c); });
    return result;
}

// Modern string formatting (C++20)
[[nodiscard]] auto format_user(std::string_view name, int age) 
    -> std::string {
    return std::format("User: {} (age: {})", name, age);
}

// String splitting with views
auto split_string(std::string_view str, char delimiter) 
    -> std::vector<std::string_view> {
    std::vector<std::string_view> result;
    size_t start = 0;
    size_t end = str.find(delimiter);
    
    while (end != std::string_view::npos) {
        result.push_back(str.substr(start, end - start));
        start = end + 1;
        end = str.find(delimiter, start);
    }
    result.push_back(str.substr(start));
    return result;
}

// ❌ WRONG - Inefficient string operations
std::string concat_bad(const std::string& a, const std::string& b) {
    return a + b;  // Multiple allocations
}

bool starts_with_bad(const std::string& str, const std::string& prefix) {
    return str.substr(0, prefix.length()) == prefix;  // Unnecessary allocation
}
```

### G. Concurrency & Threading
```cpp
// ✅ CORRECT - Modern concurrency with C++20/23
#include <thread>
#include <mutex>
#include <shared_mutex>
#include <atomic>
#include <future>
#include <latch>
#include <barrier>
#include <semaphore>

// Thread-safe counter with atomic
class AtomicCounter {
public:
    void increment() noexcept {
        count_.fetch_add(1, std::memory_order_relaxed);
    }
    
    [[nodiscard]] auto get() const noexcept -> int {
        return count_.load(std::memory_order_relaxed);
    }
    
private:
    std::atomic<int> count_{0};
};

// Thread-safe cache with shared_mutex
template<typename Key, typename Value>
class ThreadSafeCache {
public:
    void insert(const Key& key, Value value) {
        std::unique_lock lock{mutex_};
        cache_[key] = std::move(value);
    }
    
    [[nodiscard]] auto get(const Key& key) const 
        -> std::optional<Value> {
        std::shared_lock lock{mutex_};
        if (const auto it = cache_.find(key); it != cache_.end()) {
            return it->second;
        }
        return std::nullopt;
    }
    
private:
    mutable std::shared_mutex mutex_;
    std::unordered_map<Key, Value> cache_;
};

// Async task with future
[[nodiscard]] auto async_computation(int value) -> std::future<int> {
    return std::async(std::launch::async, [value]() {
        // Simulate heavy computation
        std::this_thread::sleep_for(std::chrono::seconds(1));
        return value * 2;
    });
}

// Parallel algorithm (C++17)
auto parallel_transform(std::vector<int>& data) -> void {
    std::transform(std::execution::par, 
                   data.begin(), data.end(), 
                   data.begin(),
                   [](int x) { return x * 2; });
}

// Coordination with latch (C++20)
auto parallel_work() -> void {
    constexpr int num_threads = 4;
    std::latch done{num_threads};
    std::vector<std::jthread> threads;
    
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([&done, i]() {
            // Do work
            std::this_thread::sleep_for(std::chrono::milliseconds(100 * i));
            done.count_down();
        });
    }
    
    done.wait();  // Wait for all threads
}

// ❌ WRONG - Manual mutex management
class BadThreadSafe {
public:
    void increment() {
        mutex_.lock();  // What if exception is thrown?
        ++count_;
        mutex_.unlock();
    }
    
private:
    std::mutex mutex_;
    int count_{0};
};
```

## 6. Module CMakeLists.txt Examples

### A. Library Module with Conan
```cmake
# ✅ CORRECT - Network module
# src/network/CMakeLists.txt
cmake_minimum_required(VERSION 3.15)
project(network CXX)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_BINARY_DIR})
list(APPEND CMAKE_PREFIX_PATH ${CMAKE_BINARY_DIR})

# Conan dependencies
conan_cmake_configure(
    REQUIRES
        asio/1.28.0
        openssl/3.1.4
        fmt/10.2.0
    GENERATORS CMakeDeps CMakeToolchain
)

conan_cmake_install(
    PATH_OR_REFERENCE .
    BUILD missing
    REMOTE conancenter
    SETTINGS ${CONAN_SETTINGS}
)

# Find packages
find_package(asio REQUIRED)
find_package(OpenSSL REQUIRED)
find_package(fmt REQUIRED)

# Library
add_library(${PROJECT_NAME}
    src/client.cpp
    src/server.cpp
    src/connection.cpp
)

target_include_directories(${PROJECT_NAME}
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(${PROJECT_NAME}
    PUBLIC
        asio::asio
        OpenSSL::SSL
    PRIVATE
        fmt::fmt
)

target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_20)

# Compiler warnings
if(MSVC)
    target_compile_options(${PROJECT_NAME} PRIVATE /W4 /WX)
else()
    target_compile_options(${PROJECT_NAME} PRIVATE 
        -Wall -Wextra -Wpedantic -Werror
        -Wshadow -Wnon-virtual-dtor -Wcast-align
        -Wunused -Woverloaded-virtual -Wconversion
        -Wsign-conversion -Wformat=2
    )
endif()

add_library(MyProject::network ALIAS ${PROJECT_NAME})
```

### B. Executable Module
```cmake
# ✅ CORRECT - Application executable
# apps/main/CMakeLists.txt
cmake_minimum_required(VERSION 3.15)
project(myapp CXX)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_BINARY_DIR})
list(APPEND CMAKE_PREFIX_PATH ${CMAKE_BINARY_DIR})

# Conan dependencies (CLI-specific)
conan_cmake_configure(
    REQUIRES
        cli11/2.3.2
        spdlog/1.12.0
    GENERATORS CMakeDeps CMakeToolchain
)

conan_cmake_install(
    PATH_OR_REFERENCE .
    BUILD missing
    REMOTE conancenter
    SETTINGS ${CONAN_SETTINGS}
)

find_package(CLI11 REQUIRED)
find_package(spdlog REQUIRED)

# Executable
add_executable(${PROJECT_NAME}
    main.cpp
)

# Link with project libraries
target_link_libraries(${PROJECT_NAME}
    PRIVATE
        MyProject::core
        MyProject::parser
        MyProject::network
        CLI11::CLI11
        spdlog::spdlog
)

target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_20)

# Install
install(TARGETS ${PROJECT_NAME}
    RUNTIME DESTINATION bin
)
```

### C. Test Module
```cmake
# ✅ CORRECT - Test suite with GTest
# tests/core/CMakeLists.txt
cmake_minimum_required(VERSION 3.10)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_BINARY_DIR})
list(APPEND CMAKE_PREFIX_PATH ${CMAKE_CURRENT_BINARY_DIR})

# Prevent overriding parent project settings on Windows
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

set(target core_tests)

# Conan test dependencies
conan_cmake_configure(
    REQUIRES
        gtest/1.15.0
    GENERATORS CMakeDeps CMakeToolchain
)

conan_cmake_install(
    PATH_OR_REFERENCE .
    BUILD missing
    REMOTE conancenter
    SETTINGS ${CONAN_SETTINGS}
)

find_package(GTest REQUIRED)

# Enable testing
enable_testing()

# Test executable for core module
add_executable(${target}
    test_types.cpp
    test_utils.cpp
)

target_link_libraries(${target}
    PRIVATE
        MyProject::core
        GTest::gtest
        GTest::gtest_main
)

# Discover and register tests with CTest
include(GoogleTest)
gtest_discover_tests(${target})

# Coverage support
if(ENABLE_COVERAGE)
    target_compile_options(${target} PRIVATE --coverage)
    target_link_options(${target} PRIVATE --coverage)
endif()
```

## 7. Security Best Practices

### A. Buffer Safety
```cpp
// ✅ CORRECT - Safe buffer handling
#include <array>
#include <span>
#include <vector>

// Use std::array for fixed-size buffers
auto process_fixed_buffer() -> void {
    std::array<char, 256> buffer{};  // Automatic bounds checking
    
    // Safe access
    if (buffer.size() > 10) {
        buffer.at(10) = 'x';  // Throws on out-of-bounds
    }
}

// Use span for non-owning views
auto process_data(std::span<const int> data) -> int {
    int sum = 0;
    for (const auto value : data) {  // Range-based for is safe
        sum += value;
    }
    return sum;
}

// Use vector for dynamic buffers
[[nodiscard]] auto read_file_safe(std::string_view path) 
    -> std::vector<char> {
    std::ifstream file{path.data(), std::ios::binary};
    if (!file) {
        throw std::runtime_error("Failed to open file");
    }
    
    return std::vector<char>{
        std::istreambuf_iterator<char>{file},
        std::istreambuf_iterator<char>{}
    };
}

// ❌ WRONG - C-style unsafe buffers
void process_bad() {
    char buffer[256];  // Uninitialized
    buffer[256] = 'x';  // Buffer overflow - undefined behavior
    strcpy(buffer, "data");  // Unsafe, no bounds checking
}
```

### B. Integer Safety
```cpp
// ✅ CORRECT - Safe integer operations
#include <limits>
#include <stdexcept>

template<typename T>
[[nodiscard]] auto safe_add(T a, T b) -> T 
    requires std::is_integral_v<T> {
    if (a > 0 && b > std::numeric_limits<T>::max() - a) {
        throw std::overflow_error("Integer overflow in addition");
    }
    if (a < 0 && b < std::numeric_limits<T>::min() - a) {
        throw std::overflow_error("Integer underflow in addition");
    }
    return a + b;
}

template<typename T>
[[nodiscard]] auto safe_multiply(T a, T b) -> T 
    requires std::is_integral_v<T> {
    if (a > 0 && b > 0 && a > std::numeric_limits<T>::max() / b) {
        throw std::overflow_error("Integer overflow in multiplication");
    }
    if (a < 0 && b < 0 && a < std::numeric_limits<T>::max() / b) {
        throw std::overflow_error("Integer overflow in multiplication");
    }
    return a * b;
}

// Use unsigned types for sizes and indices
[[nodiscard]] auto calculate_buffer_size(size_t element_count, 
                                          size_t element_size) -> size_t {
    return safe_multiply(element_count, element_size);
}

// ❌ WRONG - Unchecked arithmetic
int add_bad(int a, int b) {
    return a + b;  // Can overflow
}
```

### C. Input Validation
```cpp
// ✅ CORRECT - Thorough input validation
#include <regex>
#include <cctype>

[[nodiscard]] auto validate_email(std::string_view email) -> bool {
    static const std::regex pattern{
        R"([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})"
    };
    return std::regex_match(email.begin(), email.end(), pattern);
}

[[nodiscard]] auto sanitize_filename(std::string_view filename) 
    -> std::string {
    std::string result;
    result.reserve(filename.size());
    
    for (const char c : filename) {
        if (std::isalnum(static_cast<unsigned char>(c)) || 
            c == '.' || c == '_' || c == '-') {
            result.push_back(c);
        }
    }
    
    return result;
}

[[nodiscard]] auto validate_port(int port) -> bool {
    return port >= 1 && port <= 65535;
}

// ❌ WRONG - No validation
void use_input_bad(const std::string& user_input) {
    std::system(user_input.c_str());  // NEVER DO THIS - command injection!
}
```

## 8. Complete Example: Modern C++ Module

### A. Header File (include/parser/ast.hpp)
```cpp
#pragma once

#include <memory>
#include <string>
#include <vector>
#include <variant>
#include <optional>

namespace parser {

// Forward declarations
class Visitor;

// Base AST node with virtual dispatch
class ASTNode {
public:
    virtual ~ASTNode() = default;
    
    // Visitor pattern
    virtual auto accept(Visitor& visitor) -> void = 0;
    
    // RAII ensures cleanup
    ASTNode() = default;
    ASTNode(const ASTNode&) = delete;
    ASTNode& operator=(const ASTNode&) = delete;
    ASTNode(ASTNode&&) noexcept = default;
    ASTNode& operator=(ASTNode&&) noexcept = default;
};

// Concrete nodes
class NumberLiteral final : public ASTNode {
public:
    explicit NumberLiteral(double value) noexcept : value_{value} {}
    
    [[nodiscard]] auto value() const noexcept -> double { return value_; }
    
    auto accept(Visitor& visitor) -> void override;
    
private:
    double value_;
};

class BinaryOperation final : public ASTNode {
public:
    enum class Op { Add, Subtract, Multiply, Divide };
    
    BinaryOperation(Op op, 
                   std::unique_ptr<ASTNode> left,
                   std::unique_ptr<ASTNode> right)
        : op_{op}, left_{std::move(left)}, right_{std::move(right)} {}
    
    [[nodiscard]] auto op() const noexcept -> Op { return op_; }
    [[nodiscard]] auto left() const noexcept -> const ASTNode& { return *left_; }
    [[nodiscard]] auto right() const noexcept -> const ASTNode& { return *right_; }
    
    auto accept(Visitor& visitor) -> void override;
    
private:
    Op op_;
    std::unique_ptr<ASTNode> left_;
    std::unique_ptr<ASTNode> right_;
};

// Visitor interface
class Visitor {
public:
    virtual ~Visitor() = default;
    
    virtual auto visit(const NumberLiteral& node) -> void = 0;
    virtual auto visit(const BinaryOperation& node) -> void = 0;
};

// Parser class
class Parser {
public:
    using Token = std::variant<double, char>;
    
    explicit Parser(std::vector<Token> tokens)
        : tokens_{std::move(tokens)}, position_{0} {}
    
    [[nodiscard]] auto parse() -> std::unique_ptr<ASTNode>;
    
private:
    std::vector<Token> tokens_;
    size_t position_;
    
    [[nodiscard]] auto parse_expression() -> std::unique_ptr<ASTNode>;
    [[nodiscard]] auto parse_term() -> std::unique_ptr<ASTNode>;
    [[nodiscard]] auto parse_factor() -> std::unique_ptr<ASTNode>;
    
    [[nodiscard]] auto peek() const -> std::optional<Token>;
    [[nodiscard]] auto consume() -> std::optional<Token>;
};

} // namespace parser
```

### B. Implementation File (src/ast.cpp)
```cpp
#include "parser/ast.hpp"
#include <stdexcept>
#include <format>

namespace parser {

auto NumberLiteral::accept(Visitor& visitor) -> void {
    visitor.visit(*this);
}

auto BinaryOperation::accept(Visitor& visitor) -> void {
    visitor.visit(*this);
}

auto Parser::parse() -> std::unique_ptr<ASTNode> {
    return parse_expression();
}

auto Parser::parse_expression() -> std::unique_ptr<ASTNode> {
    auto left = parse_term();
    
    while (const auto token = peek()) {
        if (const auto* op = std::get_if<char>(&*token);
            op && (*op == '+' || *op == '-')) {
            consume();
            auto right = parse_term();
            
            const auto operation = *op == '+' 
                ? BinaryOperation::Op::Add 
                : BinaryOperation::Op::Subtract;
            
            left = std::make_unique<BinaryOperation>(
                operation, std::move(left), std::move(right)
            );
        } else {
            break;
        }
    }
    
    return left;
}

auto Parser::parse_term() -> std::unique_ptr<ASTNode> {
    auto left = parse_factor();
    
    while (const auto token = peek()) {
        if (const auto* op = std::get_if<char>(&*token);
            op && (*op == '*' || *op == '/')) {
            consume();
            auto right = parse_factor();
            
            const auto operation = *op == '*'
                ? BinaryOperation::Op::Multiply
                : BinaryOperation::Op::Divide;
            
            left = std::make_unique<BinaryOperation>(
                operation, std::move(left), std::move(right)
            );
        } else {
            break;
        }
    }
    
    return left;
}

auto Parser::parse_factor() -> std::unique_ptr<ASTNode> {
    const auto token = consume();
    if (!token) {
        throw std::runtime_error("Unexpected end of input");
    }
    
    if (const auto* num = std::get_if<double>(&*token)) {
        return std::make_unique<NumberLiteral>(*num);
    }
    
    throw std::runtime_error(
        std::format("Expected number, got operator")
    );
}

auto Parser::peek() const -> std::optional<Token> {
    if (position_ < tokens_.size()) {
        return tokens_[position_];
    }
    return std::nullopt;
}

auto Parser::consume() -> std::optional<Token> {
    if (position_ < tokens_.size()) {
        return tokens_[position_++];
    }
    return std::nullopt;
}

} // namespace parser
```

## 9. Unit Testing with GTest and CTest (MANDATORY)

**ALL modules MUST have unit tests.** Tests are not optional - they are a required part of the development process.

### A. Test Requirements
* **Minimum 80% code coverage** for all business logic
* **Tests MUST run via CTest** (`ctest` command)
* **Use Google Test (GTest)** framework via Conan
* **Each module has its own test suite** in `tests/module_name/`
* **Tests MUST pass** before code review/merge
* **Fast execution**: Unit tests should run in < 1 second per test file

### B. Test Module CMakeLists.txt Pattern
```cmake
# ✅ CORRECT - Test suite with GTest and CTest
# tests/parser/CMakeLists.txt
cmake_minimum_required(VERSION 3.10)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_BINARY_DIR})
list(APPEND CMAKE_PREFIX_PATH ${CMAKE_CURRENT_BINARY_DIR})

# Prevent overriding parent project settings on Windows
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

set(target parser_tests)

# Conan dependencies for testing
conan_cmake_configure(
    REQUIRES
        gtest/1.15.0
    GENERATORS CMakeDeps CMakeToolchain
)

conan_cmake_install(
    PATH_OR_REFERENCE .
    BUILD missing
    REMOTE conancenter
    SETTINGS ${CONAN_SETTINGS}
)

# Find GTest package
find_package(GTest REQUIRED)

# Enable testing
enable_testing()

# Create test executable
add_executable(${target}
    parser_tests.cpp
    ast_tests.cpp
)

# Link with module under test and GTest
target_link_libraries(${target}
    PRIVATE
        MyProject::parser
        GTest::gtest
        GTest::gtest_main
)

# Discover and register tests with CTest
include(GoogleTest)
gtest_discover_tests(${target})
```

### C. Test File Structure
```cpp
// ✅ CORRECT - Modern GTest structure
// tests/parser/parser_tests.cpp
#include <gtest/gtest.h>
#include "parser/ast.hpp"

using namespace parser;

// Test fixture for shared setup
class ParserTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Common setup code
    }
    
    void TearDown() override {
        // Cleanup code
    }
    
    // Helper methods
    auto create_simple_tokens() -> std::vector<Parser::Token> {
        return {10.0, '+', 5.0};
    }
};

// Basic value test
TEST(NumberLiteralTest, StoresValueCorrectly) {
    const NumberLiteral literal{42.0};
    EXPECT_DOUBLE_EQ(literal.value(), 42.0);
}

// Test with fixture
TEST_F(ParserTest, ParsesSimpleExpression) {
    auto tokens = create_simple_tokens();
    Parser parser{std::move(tokens)};
    
    auto ast = parser.parse();
    ASSERT_NE(ast, nullptr);
    
    const auto* bin_op = dynamic_cast<const BinaryOperation*>(ast.get());
    ASSERT_NE(bin_op, nullptr);
    EXPECT_EQ(bin_op->op(), BinaryOperation::Op::Add);
}

// Multiple assertions
TEST(BinaryOperationTest, ConstructsCorrectly) {
    auto left = std::make_unique<NumberLiteral>(10.0);
    auto right = std::make_unique<NumberLiteral>(5.0);
    
    BinaryOperation op{
        BinaryOperation::Op::Add,
        std::move(left),
        std::move(right)
    };
    
    EXPECT_EQ(op.op(), BinaryOperation::Op::Add);
    
    const auto& left_node = dynamic_cast<const NumberLiteral&>(op.left());
    EXPECT_DOUBLE_EQ(left_node.value(), 10.0);
    
    const auto& right_node = dynamic_cast<const NumberLiteral&>(op.right());
    EXPECT_DOUBLE_EQ(right_node.value(), 5.0);
}

// Exception testing
TEST(ParserTest, ThrowsOnInvalidInput) {
    std::vector<Parser::Token> tokens{'+', 5.0};
    Parser parser{std::move(tokens)};
    
    EXPECT_THROW(parser.parse(), std::runtime_error);
}

// Parameterized tests
class OperatorTest : public ::testing::TestWithParam<std::tuple<char, BinaryOperation::Op>> {};

TEST_P(OperatorTest, ParsesOperatorCorrectly) {
    auto [op_char, expected_op] = GetParam();
    
    std::vector<Parser::Token> tokens{1.0, op_char, 2.0};
    Parser parser{std::move(tokens)};
    
    auto ast = parser.parse();
    const auto* bin_op = dynamic_cast<const BinaryOperation*>(ast.get());
    
    ASSERT_NE(bin_op, nullptr);
    EXPECT_EQ(bin_op->op(), expected_op);
}

INSTANTIATE_TEST_SUITE_P(
    Operators,
    OperatorTest,
    ::testing::Values(
        std::make_tuple('+', BinaryOperation::Op::Add),
        std::make_tuple('-', BinaryOperation::Op::Subtract),
        std::make_tuple('*', BinaryOperation::Op::Multiply),
        std::make_tuple('/', BinaryOperation::Op::Divide)
    )
);

// Death tests (for testing crashes/aborts)
TEST(ParserDeathTest, AssertsOnNullPointer) {
    EXPECT_DEATH({
        Parser* parser = nullptr;
        parser->parse();  // Should crash
    }, "");
}

// Performance test (not run by default)
TEST(ParserPerformanceTest, DISABLED_HandlesLargeExpression) {
    std::vector<Parser::Token> tokens;
    for (int i = 0; i < 1000; ++i) {
        tokens.push_back(1.0);
        tokens.push_back('+');
    }
    tokens.push_back(1.0);
    
    Parser parser{std::move(tokens)};
    
    auto start = std::chrono::high_resolution_clock::now();
    auto ast = parser.parse();
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    EXPECT_LT(duration.count(), 100);  // Should complete in < 100ms
}
```

### D. Running Tests
```bash
# Build and run all tests
cd build
cmake .
cmake --build .
ctest

# Run tests with verbose output
ctest --verbose

# Run tests with output on failure
ctest --output-on-failure

# Run specific test
ctest -R parser_tests

# Run tests in parallel
ctest -j8

# Generate coverage report (if ENABLE_COVERAGE=ON)
ctest
lcov --capture --directory . --output-file coverage.info
lcov --remove coverage.info '/usr/*' --output-file coverage.info
lcov --list coverage.info
```

### E. Test Organization Best Practices
```cpp
// ✅ CORRECT - Well-organized tests
namespace {  // Anonymous namespace for test helpers

// Test helper functions
auto create_test_user() -> User {
    return User{"test@example.com", "Test User"};
}

// Test constants
constexpr double kPi = 3.14159265359;
constexpr int kMaxIterations = 1000;

}  // namespace

// Group related tests
TEST(MathUtilsTest, AddPositiveNumbers) {
    EXPECT_EQ(add(2, 3), 5);
}

TEST(MathUtilsTest, AddNegativeNumbers) {
    EXPECT_EQ(add(-2, -3), -5);
}

TEST(MathUtilsTest, AddMixedNumbers) {
    EXPECT_EQ(add(-2, 3), 1);
}

// Use descriptive test names
TEST(UserValidation, AcceptsValidEmail) {
    EXPECT_TRUE(is_valid_email("user@example.com"));
}

TEST(UserValidation, RejectsEmailWithoutAtSign) {
    EXPECT_FALSE(is_valid_email("userexample.com"));
}

TEST(UserValidation, RejectsEmailWithoutDomain) {
    EXPECT_FALSE(is_valid_email("user@"));
}
```

### F. Mock Objects (for testing with dependencies)
```cpp
// ✅ CORRECT - Using Google Mock
#include <gmock/gmock.h>
#include <gtest/gtest.h>

// Interface to mock
class Database {
public:
    virtual ~Database() = default;
    virtual auto get_user(int id) -> std::optional<User> = 0;
    virtual auto save_user(const User& user) -> bool = 0;
};

// Mock implementation
class MockDatabase : public Database {
public:
    MOCK_METHOD(std::optional<User>, get_user, (int id), (override));
    MOCK_METHOD(bool, save_user, (const User& user), (override));
};

// Service under test
class UserService {
public:
    explicit UserService(Database& db) : db_{db} {}
    
    auto load_user(int id) -> std::optional<User> {
        return db_.get_user(id);
    }
    
private:
    Database& db_;
};

// Test using mock
TEST(UserServiceTest, LoadsUserFromDatabase) {
    MockDatabase mock_db;
    User expected_user{"test@example.com", "Test User"};
    
    // Set expectations
    EXPECT_CALL(mock_db, get_user(123))
        .WillOnce(::testing::Return(expected_user));
    
    UserService service{mock_db};
    auto user = service.load_user(123);
    
    ASSERT_TRUE(user.has_value());
    EXPECT_EQ(user->email(), "test@example.com");
}
```

### G. Integration with Root CMakeLists.txt
```cmake
# Root CMakeLists.txt
if(BUILD_TESTS)
    enable_testing()
    
    # Add test subdirectories
    add_subdirectory(tests/core)
    add_subdirectory(tests/parser)
    add_subdirectory(tests/network)
    
    # Custom target to run all tests
    add_custom_target(check
        COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
        DEPENDS core_tests parser_tests network_tests
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        COMMENT "Running all unit tests..."
    )
endif()
```

## 10. CMake Helper Modules

### A. Compiler Warnings (cmake/CompilerWarnings.cmake)
```cmake
# ✅ CORRECT - Comprehensive warning flags
function(set_project_warnings target_name)
    set(MSVC_WARNINGS
        /W4          # Baseline reasonable warnings
        /w14242      # Conversion warning
        /w14254      # 'operator': conversion from 'type1:field_bits' to 'type2:field_bits'
        /w14263      # 'function': member function does not override any base class virtual member function
        /w14265      # 'classname': class has virtual functions, but destructor is not virtual
        /w14287      # 'operator': unsigned/negative constant mismatch
        /we4289      # Loop control variable declared in the for-loop is used outside
        /w14296      # 'operator': expression is always 'boolean_value'
        /w14311      # 'variable': pointer truncation from 'type1' to 'type2'
        /w14545      # Expression before comma evaluates to a function
        /w14546      # Function call before comma missing argument list
        /w14547      # 'operator': operator before comma has no effect
        /w14549      # 'operator': operator before comma has no effect
        /w14555      # Expression has no effect
        /w14619      # Pragma warning: there is no warning number 'number'
        /w14640      # Enable warning on thread un-safe static member initialization
        /w14826      # Conversion from 'type1' to 'type_2' is sign-extended
        /w14905      # Wide string literal cast to 'LPSTR'
        /w14906      # String literal cast to 'LPWSTR'
        /w14928      # Illegal copy-initialization
    )

    set(GCC_CLANG_WARNINGS
        -Wall
        -Wextra
        -Wpedantic
        -Wshadow
        -Wnon-virtual-dtor
        -Wcast-align
        -Wunused
        -Woverloaded-virtual
        -Wconversion
        -Wsign-conversion
        -Wnull-dereference
        -Wdouble-promotion
        -Wformat=2
        -Wimplicit-fallthrough
    )

    set(GCC_WARNINGS
        ${GCC_CLANG_WARNINGS}
        -Wmisleading-indentation
        -Wduplicated-cond
        -Wduplicated-branches
        -Wlogical-op
        -Wuseless-cast
    )

    set(CLANG_WARNINGS
        ${GCC_CLANG_WARNINGS}
        -Wmost
        -Weverything
        -Wno-c++98-compat
        -Wno-c++98-compat-pedantic
        -Wno-padded
    )

    if(MSVC)
        set(PROJECT_WARNINGS ${MSVC_WARNINGS})
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
        set(PROJECT_WARNINGS ${GCC_WARNINGS})
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        set(PROJECT_WARNINGS ${CLANG_WARNINGS})
    endif()

    target_compile_options(${target_name} INTERFACE ${PROJECT_WARNINGS})
endfunction()
```

### B. Sanitizers (cmake/Sanitizers.cmake)
```cmake
# ✅ CORRECT - Runtime sanitizers
function(enable_sanitizers target_name)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        option(ENABLE_SANITIZER_ADDRESS "Enable address sanitizer" FALSE)
        option(ENABLE_SANITIZER_LEAK "Enable leak sanitizer" FALSE)
        option(ENABLE_SANITIZER_UNDEFINED "Enable undefined behavior sanitizer" FALSE)
        option(ENABLE_SANITIZER_THREAD "Enable thread sanitizer" FALSE)
        option(ENABLE_SANITIZER_MEMORY "Enable memory sanitizer" FALSE)

        set(SANITIZERS "")

        if(ENABLE_SANITIZER_ADDRESS)
            list(APPEND SANITIZERS "address")
        endif()

        if(ENABLE_SANITIZER_LEAK)
            list(APPEND SANITIZERS "leak")
        endif()

        if(ENABLE_SANITIZER_UNDEFINED)
            list(APPEND SANITIZERS "undefined")
        endif()

        if(ENABLE_SANITIZER_THREAD)
            if("address" IN_LIST SANITIZERS OR "leak" IN_LIST SANITIZERS)
                message(WARNING "Thread sanitizer cannot be used with Address or Leak sanitizer")
            else()
                list(APPEND SANITIZERS "thread")
            endif()
        endif()

        if(ENABLE_SANITIZER_MEMORY)
            if("address" IN_LIST SANITIZERS 
               OR "thread" IN_LIST SANITIZERS 
               OR "leak" IN_LIST SANITIZERS)
                message(WARNING "Memory sanitizer cannot be used with other sanitizers")
            else()
                list(APPEND SANITIZERS "memory")
            endif()
        endif()

        list(JOIN SANITIZERS "," SANITIZER_LIST)

        if(SANITIZER_LIST)
            if(NOT "${SANITIZER_LIST}" STREQUAL "")
                target_compile_options(${target_name} INTERFACE 
                    -fsanitize=${SANITIZER_LIST}
                    -fno-omit-frame-pointer
                )
                target_link_options(${target_name} INTERFACE 
                    -fsanitize=${SANITIZER_LIST}
                )
            endif()
        endif()
    endif()
endfunction()
```

## 11. Deployment Checklist

### Pre-Production Validation

#### Agent Code Generation Verification (MANDATORY - For AI-Generated Code)
- [ ] **Agent verified build succeeds** - Code was tested to compile before delivery
- [ ] **All public APIs documented** - Doxygen comments added for all exported symbols
- [ ] **Documentation can be generated** - `make docs` succeeds without errors
- [ ] **No documentation warnings** - `make docs-check` passes
- [ ] **All compilation errors fixed** - Agent iterated until clean build
- [ ] **Dependencies properly configured** - All Conan packages or system packages available
- [ ] **CMakeLists.txt syntax verified** - No CMake configuration errors

#### Testing (MANDATORY - Cannot proceed without these)
- [ ] **All unit tests exist** - Every module has tests in `tests/module_name/`
- [ ] **All unit tests passing** - `ctest` returns 0 exit code
- [ ] **Test coverage ≥ 80%** - Use `--coverage` flags and verify with lcov/gcov
- [ ] **Tests run via CTest** - All tests discovered with `gtest_discover_tests()`
- [ ] **No skipped/disabled tests** - All `DISABLED_` tests have documented reasons
- [ ] **Tests are fast** - Each test file completes in < 1 second
- [ ] **Integration tests passing** (if applicable)

#### Code Quality
- [ ] **All public APIs have Doxygen documentation** - Classes, functions, templates
- [ ] **Documentation complete** - Includes @brief, @param, @return, @throws
- [ ] **Documentation examples provided** - @code blocks for complex APIs
- [ ] **API documentation generated** - `make docs` creates readable output
- [ ] All compiler warnings resolved (`-Wall -Wextra -Werror`)
- [ ] Static analysis passing (clang-tidy)
- [ ] No raw pointers in public APIs
- [ ] RAII for all resources
- [ ] Move semantics used throughout
- [ ] const-correctness enforced
- [ ] All code formatted with clang-format

#### Security & Safety
- [ ] No memory leaks (Valgrind/ASAN clean)
- [ ] No undefined behavior (UBSAN clean)
- [ ] Thread sanitizer clean (if multithreaded)
- [ ] Input validation on all external inputs
- [ ] Buffer overflows impossible (std::array, std::vector, span)
- [ ] Integer overflow checks in critical paths

#### Build & Dependencies
- [ ] All dependencies from Conan, no system dependencies
- [ ] Clean build from scratch succeeds
- [ ] Release build optimized (`-O3` or `-O2`)
- [ ] Debug symbols available for production debugging
- [ ] No warnings in Release build

#### Runtime
- [ ] Crash handler installed
- [ ] Logging configured
- [ ] Error messages are meaningful
- [ ] Resource cleanup verified (no file descriptors leaked)

### Running the Complete Validation

```bash
# Complete validation script
#!/bin/bash
set -e

# Clean build
rm -rf build
mkdir build
cd build

# Configure with all checks enabled
cmake .. \
    -DBUILD_TESTS=ON \
    -DENABLE_COVERAGE=ON \
    -DENABLE_SANITIZER_ADDRESS=ON \
    -DENABLE_SANITIZER_UNDEFINED=ON \
    -DCMAKE_BUILD_TYPE=Debug

# Build
cmake --build . -j$(nproc)

# Run tests
ctest --output-on-failure

# Check coverage
lcov --capture --directory . --output-file coverage.info
lcov --remove coverage.info '/usr/*' '*/tests/*' --output-file coverage.info
lcov --list coverage.info

COVERAGE=$(lcov --summary coverage.info | grep lines | awk '{print $2}' | sed 's/%//')
if (( $(echo "$COVERAGE < 80" | bc -l) )); then
    echo "ERROR: Coverage is $COVERAGE%, minimum is 80%"
    exit 1
fi

echo "✅ All validation checks passed!"
```

---

## Why This Configuration Works

1. **Conan-First Dependency Strategy**: Ensures reproducible builds across all platforms, eliminates "works on my machine" problems, provides version locking, and simplifies CI/CD. Fallback to system packages only when necessary maintains OS integration for system-critical libraries.

2. **Agent Build Verification**: Requiring agents to verify code compilation before delivery eliminates non-compiling code submissions, reduces developer frustration, and ensures all examples and generated code actually work. The iterative fix-and-rebuild cycle catches errors early.

3. **Doxygen + API Documentation**: Auto-generated documentation from code, always in sync, reduces onboarding time by 40%+, better IDE IntelliSense, enables API discoverability. Documentation as code ensures it never becomes outdated.

4. **Conan in CMake**: Single command build, reproducible builds, dependency isolation per module, no manual dependency management.

5. **Modular Structure**: Small files, clear responsibilities, easy testing, parallel builds.

6. **C++20/23 Features**: Modern safety, better performance, cleaner code.

7. **Smart Pointers**: Automatic memory management, no leaks, clear ownership.

8. **RAII**: Resource safety, exception safety, deterministic cleanup.

9. **Move Semantics**: Zero-copy where possible, explicit ownership transfer.

10. **Concepts**: Compile-time constraints, better error messages, self-documenting.

11. **std::expected**: Explicit error handling, no exceptions for expected errors.

12. **Const Correctness**: Prevents bugs, enables optimizations, documents intent.

13. **Comprehensive Warnings**: Catches bugs at compile-time, prevents undefined behavior.

14. **Mandatory Testing with GTest**: Industry-standard framework, excellent IDE integration, parameterized tests, mock support. CTest integration enables running tests as part of the build process and CI/CD pipelines. Tests catch regressions before they reach production.

15. **Per-Module Testing**: Each module manages its own test dependencies through Conan, enabling independent testing and parallel test execution.

---

## Quick Reference

### Common Commands

```bash
# Configure & Build
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel

# Test
ctest --test-dir build --output-on-failure
cmake --build build --target test

# Format & Lint
clang-format -i src/**/*.cpp src/**/*.hpp
clang-tidy src/**/*.cpp --fix

# Documentation
doxygen Doxyfile
cmake --build build --target docs

# Conan dependencies
conan install . --build=missing
conan lock create .
```

### CMakeLists.txt Template

```cmake
cmake_minimum_required(VERSION 3.20)
project(MyProject VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(GTest REQUIRED)

add_executable(${PROJECT_NAME} src/main.cpp)
target_include_directories(${PROJECT_NAME} PRIVATE include)

enable_testing()
add_executable(tests tests/test_main.cpp)
target_link_libraries(tests PRIVATE GTest::gtest_main)
add_test(NAME unit_tests COMMAND tests)
```

### Modern C++ Patterns

```cpp
// Smart pointers
auto ptr = std::make_unique<MyClass>(args...);
auto shared = std::make_shared<MyClass>(args...);

// Optional
std::optional<T> findById(int id);
if (auto result = findById(1)) { use(*result); }

// Expected (C++23)
std::expected<T, Error> process();

// Ranges
auto result = items | std::views::filter(pred)
                    | std::views::transform(fn);

// Structured bindings
auto [key, value] = std::pair{1, "one"};
```

### Project Structure

```
my_project/
├── CMakeLists.txt
├── conanfile.txt
├── include/
│   └── myproject/
│       └── *.hpp
├── src/
│   └── *.cpp
├── tests/
│   └── test_*.cpp
└── docs/
    └── Doxyfile
```

---

## References

- [C++ Core Guidelines](https://isocpp.github.io/CppCoreGuidelines/)
- [Modern CMake Guide](https://cliutils.gitlab.io/modern-cmake/)
- [Conan Documentation](https://docs.conan.io/)
- [C++20/23 Features](https://en.cppreference.com/)
- [CERT C++ Coding Standard](https://wiki.sei.cmu.edu/confluence/x/nNYxBQ)
- [CppCon Talks](https://www.youtube.com/user/CppCon)


**End of Modern C++ Development Guidelines**
