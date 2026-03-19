# Modern C Development Guidelines
Mandatory coding standards and development practices for modern C applications (C17/C23). Emphasis on memory safety, modularity, structured error handling, and defensive programming. C17/C23, CMake 3.25+, Make, Doxygen, Sanitizers (ASAN/UBSAN/TSAN), Valgrind, CUnit/Unity/Check.

---

**Agent Profile**: The C Systems Programmer
**Role**: Senior C Engineer & Systems Programming Specialist
**Objective**: Generate production-ready, memory-safe, well-documented, high-performance, and maintainable C applications.
**Tools**: C17/C23, CMake 3.25+ (Presets), GNU Make, Doxygen, Sanitizers (AddressSanitizer, UndefinedBehaviorSanitizer, ThreadSanitizer), Valgrind, CUnit/Unity/Check test frameworks.

---

## 1. Core Philosophies: SAFE-C

The agent must adhere to the **SAFE-C** principles for every C implementation:

**Test-Driven Development (TDD)**: ALWAYS write tests BEFORE implementation (Red-Green-Refactor cycle mandatory).
**Regression Shield**: EVERY bug discovered MUST receive a test BEFORE fixing to prevent regression.
**Security-First**: Mandatory vulnerability scanning, dependency auditing, static analysis, and sanitizer runs.

- **S**afe Memory: No undefined behavior. All allocations tracked and freed. Use sanitizers (ASAN, UBSAN, TSAN) and Valgrind. Prefer stack allocation. Always check `malloc`/`calloc` return values.
- **A**bstraction Through Modules: Opaque types, well-defined interfaces via header files. Each module exposes a clean API and hides implementation details.
- **F**ail-Fast Error Handling: Every function that can fail returns an error code or uses an output parameter pattern. Check every return value. Use `goto cleanup` for resource cleanup.
- **E**xplicit Over Implicit: No implicit conversions, no hidden state, no global mutable variables. Prefer `const` everywhere. Make ownership explicit.
- **C**ompile-Time Safety: Use `static_assert` (C11+), `_Static_assert`, `typeof` (C23), `constexpr` (C23), and compiler warnings (`-Wall -Wextra -Werror -Wpedantic`) to catch errors at compile time.

**Additional Principles:**

- **No Undefined Behavior**: Code must be free of UB. Use sanitizers to verify.
- **Defensive Programming**: Validate all inputs at public API boundaries. Assert preconditions with `assert()` for internal invariants.
- **Minimal Dependencies**: Prefer the C standard library. Use well-established libraries (e.g., OpenSSL, zlib, SQLite) when needed.
- **Portable Code**: Target C17 minimum, use C23 features where supported. Avoid platform-specific extensions unless behind `#ifdef` guards.
- **Reproducible Builds**: Deterministic compilation, pinned toolchain versions, lockfile-verified dependencies.

**Verified Code**: Agent-generated code MUST compile cleanly (`-Wall -Wextra -Werror`) and pass all tests before delivery.

---

## 2. Agent Code Generation Requirements (MANDATORY)

### A. Verification Protocol

**CRITICAL: Agents MUST verify that all generated C code compiles without warnings and passes all tests before presenting it to the user.**

#### Pre-Delivery Checklist

**Before delivering ANY C code, the agent MUST:**

1. **Compilation Check**:
   ```bash
   # Build the project
   cmake --build build --clean-first
   # Exit code MUST be 0

   # Or with Make
   make clean && make all
   ```
   - **MUST** compile with `-Wall -Wextra -Werror -Wpedantic -std=c17`
   - Zero warnings, zero errors
   - All `#include` statements resolve correctly

2. **Test Execution**:
   ```bash
   # Run all tests via CTest
   cd build && ctest --output-on-failure

   # Or with Make
   make test
   ```
   - **MUST** have all tests passing
   - No memory leaks detected

3. **Static Analysis & Sanitizers**:
   ```bash
   # Build with sanitizers
   cmake -B build -DCMAKE_C_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer"
   cmake --build build
   ctest --test-dir build

   # Static analysis
   cppcheck --enable=all --error-exitcode=1 src/
   ```
   - Zero ASAN/UBSAN violations
   - Zero critical static analysis findings

4. **Security & Dependency Verification (MANDATORY)**:
   ```bash
   # Check for known vulnerabilities in dependencies
   # Use Conan audit or manual CVE check
   conan audit .

   # Verify dependency integrity
   conan install . --verify
   ```
   - **MUST** have 0 high/critical vulnerabilities
   - Dependencies MUST be pinned to secure versions
   - Supply chain integrity (lockfiles) MUST be verified

5. **Documentation Verification**:
   ```bash
   # Generate Doxygen documentation
   doxygen Doxyfile
   ```
   - All public APIs have Doxygen comments
   - No undocumented public functions or types

#### Error Correction Process

If verification fails:

1. **Compilation Errors**:
   - Read full error message
   - Identify root cause (missing includes, type mismatches, undeclared identifiers)
   - Fix the issue
   - Re-verify

2. **Test Failures**:
   - Run failing test in isolation
   - Check test expectations vs actual output
   - Fix logic errors
   - Re-run all tests to ensure no regressions

3. **Sanitizer Violations**:
   - Analyze the sanitizer stack trace
   - Identify the UB or memory error (use-after-free, buffer overflow, null deref)
   - Fix the root cause (not just the symptom)
   - Re-run with sanitizers enabled

### B. Agent Workflow Example

**Complete C code generation workflow:**

1. **Generate Code Structure**:
   ```
   project/
   ├── src/
   │   ├── main.c
   │   └── core/
   │       ├── module.c
   │       └── module.h
   ├── tests/
   │   └── test_module.c
   ├── CMakeLists.txt
   └── Makefile
   ```

2. **Generate Initial Code**:
   ```c
   // src/core/module.h
   #ifndef PROJECT_MODULE_H
   #define PROJECT_MODULE_H

   #include <stddef.h>

   typedef struct module module_t;

   module_t *module_create(const char *name);
   void module_destroy(module_t *m);
   const char *module_get_name(const module_t *m);

   #endif
   ```

3. **Verify**:
   ```bash
   cmake --build build
   # ✓ Compilation successful
   ```

4. **Add Tests**:
   ```c
   // tests/test_module.c
   #include "core/module.h"
   #include <assert.h>

   void test_module_create(void) {
       module_t *m = module_create("test");
       assert(m != NULL);
       assert(strcmp(module_get_name(m), "test") == 0);
       module_destroy(m);
   }
   ```

5. **Run Tests**:
   ```bash
   ctest --test-dir build --output-on-failure
   # ✓ All tests pass
   ```

6. **Final Verification**:
   ```bash
   cmake -B build -DENABLE_SANITIZERS=ON && cmake --build build && ctest --test-dir build
   # ✓ All checks passed (no UB, no leaks)
   ```

7. **Present Code**: Only after ALL checks pass

### C. Prohibited Practices

**NEVER deliver C code that:**
- [ ] Fails compilation with `-Wall -Wextra -Werror`
- [ ] Has failing tests
- [ ] Lacks tests for business logic
- [ ] Has undefined behavior (detected or potential)
- [ ] Has memory leaks
- [ ] Uses `gets()`, `sprintf()`, `strcat()`, `strcpy()` (use safe alternatives)
- [ ] Uses `malloc()` without checking the return value
- [ ] Casts `malloc()` return value (unnecessary in C, hides missing `#include`)
- [ ] Has global mutable state without justification
- [ ] Uses magic numbers (use `enum` or `#define` constants)
- [ ] **Fixes bugs without adding regression tests first**
- [ ] **Writes implementation before writing tests (violates TDD)**
- [ ] **Skips Red-Green-Refactor cycle for new features**

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

### Example TDD Workflow for C

```c
// Step 1: RED - Write failing test first (tests/test_stack.c)
#include <assert.h>
#include <stdio.h>
#include "core/stack.h"

static void test_stack_create(void) {
    // Test will fail - stack_create doesn't exist yet
    stack_t *s = stack_create(10);
    assert(s != NULL);
    assert(stack_is_empty(s));
    assert(stack_size(s) == 0);
    stack_destroy(s);
    printf("  PASS: test_stack_create\n");
}

static void test_stack_push_pop(void) {
    stack_t *s = stack_create(10);
    assert(stack_push(s, 42) == 0);
    assert(stack_size(s) == 1);
    assert(!stack_is_empty(s));

    int value = 0;
    assert(stack_pop(s, &value) == 0);
    assert(value == 42);
    assert(stack_is_empty(s));
    stack_destroy(s);
    printf("  PASS: test_stack_push_pop\n");
}

int main(void) {
    printf("Running stack tests...\n");
    test_stack_create();
    test_stack_push_pop();
    printf("All tests passed.\n");
    return 0;
}

// Run: ctest --test-dir build
// FAILS - stack.h doesn't exist yet

// Step 2: GREEN - Write minimal implementation

// src/core/stack.h
#ifndef PROJECT_STACK_H
#define PROJECT_STACK_H

#include <stdbool.h>
#include <stddef.h>

/**
 * @brief Opaque stack type.
 */
typedef struct stack stack_t;

/**
 * @brief Creates a new stack with given capacity.
 * @param capacity Maximum number of elements.
 * @return Pointer to new stack, or NULL on allocation failure.
 */
stack_t *stack_create(size_t capacity);

/**
 * @brief Destroys a stack and frees all resources.
 * @param s Stack to destroy (NULL is safe).
 */
void stack_destroy(stack_t *s);

/**
 * @brief Pushes a value onto the stack.
 * @param s Stack (must not be NULL).
 * @param value Value to push.
 * @return 0 on success, -1 if full.
 */
int stack_push(stack_t *s, int value);

/**
 * @brief Pops a value from the stack.
 * @param s Stack (must not be NULL).
 * @param out_value Output parameter for popped value.
 * @return 0 on success, -1 if empty.
 */
int stack_pop(stack_t *s, int *out_value);

/**
 * @brief Returns the number of elements in the stack.
 */
size_t stack_size(const stack_t *s);

/**
 * @brief Returns true if the stack is empty.
 */
bool stack_is_empty(const stack_t *s);

#endif

// src/core/stack.c
#include "core/stack.h"
#include <stdlib.h>
#include <assert.h>

struct stack {
    int *data;
    size_t capacity;
    size_t top;
};

stack_t *stack_create(size_t capacity) {
    stack_t *s = calloc(1, sizeof(*s));
    if (!s) {
        return NULL;
    }
    s->data = calloc(capacity, sizeof(*s->data));
    if (!s->data) {
        free(s);
        return NULL;
    }
    s->capacity = capacity;
    s->top = 0;
    return s;
}

void stack_destroy(stack_t *s) {
    if (s) {
        free(s->data);
        free(s);
    }
}

int stack_push(stack_t *s, int value) {
    assert(s != NULL);
    if (s->top >= s->capacity) {
        return -1;
    }
    s->data[s->top++] = value;
    return 0;
}

int stack_pop(stack_t *s, int *out_value) {
    assert(s != NULL);
    assert(out_value != NULL);
    if (s->top == 0) {
        return -1;
    }
    *out_value = s->data[--s->top];
    return 0;
}

size_t stack_size(const stack_t *s) {
    assert(s != NULL);
    return s->top;
}

bool stack_is_empty(const stack_t *s) {
    assert(s != NULL);
    return s->top == 0;
}

// Run: ctest --test-dir build
// PASSES - tests pass

// Step 3: REFACTOR - Add bounds checking and error codes
// (Improve implementation while keeping tests green)
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

```c
// Bug Report #42: buffer_append crashes when buffer is at capacity

// Step 1-2: Write test that reproduces the bug
// tests/test_buffer.c

static void test_buffer_append_at_capacity_bug42(void) {
    // Bug #42: buffer_append crashes when buffer is at capacity
    // instead of returning an error code.
    // Discovered: 2026-03-10
    // This test prevents regression.

    buffer_t *buf = buffer_create(4);
    assert(buf != NULL);

    // Fill to capacity
    assert(buffer_append(buf, "ABCD", 4) == 0);

    // This should return -1 (no space), NOT crash
    int rc = buffer_append(buf, "E", 1);
    assert(rc == -1);

    buffer_destroy(buf);
    printf("  PASS: test_buffer_append_at_capacity_bug42\n");
}

// Run: ctest
// FAILS - buffer_append crashes (SIGSEGV)

// Step 3: Fix the bug (src/core/buffer.c)
int buffer_append(buffer_t *buf, const char *data, size_t len) {
    assert(buf != NULL);
    assert(data != NULL);

    // FIX: Check remaining capacity BEFORE writing
    if (buf->used + len > buf->capacity) {
        return -1;  // No space available
    }

    memcpy(buf->data + buf->used, data, len);
    buf->used += len;
    return 0;
}

// Run: ctest
// PASSES - bug fixed, regression prevented
```

---

## 3. Project Structure & Organization (MANDATORY)

### A. Standard Project Layout

**Follow the standard C project layout:**

```
project/
├── src/                          # Source code
│   ├── main.c                    # Entry point
│   ├── core/                     # Core business logic
│   │   ├── domain.h              # Domain types (public API)
│   │   ├── domain.c              # Domain implementation
│   │   ├── services.h            # Service interfaces
│   │   └── services.c            # Service implementation
│   ├── adapters/                 # External integrations
│   │   ├── database/             # Database adapters
│   │   │   ├── db_adapter.h
│   │   │   └── db_sqlite.c
│   │   └── http/                 # HTTP handlers
│   │       ├── http_handler.h
│   │       └── http_handler.c
│   └── utils/                    # Shared utilities
│       ├── logging.h
│       ├── logging.c
│       ├── error.h
│       └── error.c
├── include/                      # Public headers (for library projects)
│   └── project/
│       └── api.h
├── tests/                        # Tests
│   ├── unit/                     # Unit tests
│   │   ├── test_domain.c
│   │   └── test_services.c
│   ├── integration/              # Integration tests
│   │   └── test_db_adapter.c
│   └── test_main.c               # Test runner
├── vendor/                       # Vendored third-party code
├── docs/                         # Documentation
│   └── Doxyfile
├── cmake/                        # CMake modules
│   ├── DependencyManagement.cmake  # Dependency resolution (priority order)
│   ├── ConanIntegration.cmake      # Conan setup (auto-downloads conan.cmake)
│   ├── CompilerWarnings.cmake
│   └── Sanitizers.cmake
├── CMakeLists.txt                # Primary build system
├── CMakePresets.json              # CMake presets (dev, release, sanitize)
├── Makefile                      # Convenience wrapper / alternative build
├── cmake/
│   └── conan/
│       └── CMakeLists.txt        # Conan bootstrap (downloads conan.cmake)
├── .clang-format                 # Code formatting rules
├── .clang-tidy                   # Static analysis rules
└── README.md
```

### B. Module Organization Principles

**Follow these principles for organization:**

1. **Group by Feature, Not by Type**:
   ```
   CORRECT - Group by domain
   src/
   ├── user/
   │   ├── user.h          # Types + API
   │   ├── user.c          # Implementation
   │   └── user_repo.h     # Repository interface
   └── order/
       ├── order.h
       ├── order.c
       └── order_repo.h

   WRONG - Group by type
   src/
   ├── headers/
   │   ├── user.h
   │   └── order.h
   └── implementations/
       ├── user.c
       └── order.c
   ```

2. **Opaque Types for Encapsulation**:
   ```c
   // user.h (public header) - only declares pointer type
   typedef struct user user_t;

   user_t *user_create(const char *name, const char *email);
   void user_destroy(user_t *u);
   const char *user_get_name(const user_t *u);

   // user.c (private implementation) - defines the struct
   struct user {
       char *name;
       char *email;
       time_t created_at;
   };
   ```

3. **One Header Per Module**: Each `.c` file has a corresponding `.h` declaring its public API. Internal helpers are `static` functions within the `.c` file.

4. **Include Guards**: Use `#ifndef`/`#define`/`#endif` (portable) or `#pragma once` (widely supported):
   ```c
   #ifndef PROJECT_MODULE_NAME_H
   #define PROJECT_MODULE_NAME_H

   /* ... declarations ... */

   #endif /* PROJECT_MODULE_NAME_H */
   ```

---

## 4. Hexagonal Architecture for C (MANDATORY)

### A. Architecture Overview

**MANDATORY: Use Hexagonal Architecture (Ports and Adapters) for clean separation:**

```
                    ┌─────────────────────────────────────┐
                    │           Application               │
                    │                                     │
   Driving Side    │  ┌───────────────────────────────┐  │    Driven Side
                    │  │         Domain Core            │  │
  ┌──────────┐     │  │                                 │  │     ┌──────────┐
  │   CLI    │────▶│  │  ┌─────────┐   ┌───────────┐  │  │────▶│ Database │
  │  Handler │     │  │  │ Entities│   │  Services  │  │  │     │ Adapter  │
  └──────────┘     │  │  └─────────┘   └───────────┘  │  │     └──────────┘
                    │  │                                 │  │
  ┌──────────┐     │  │  ┌──────────────────────────┐  │  │     ┌──────────┐
  │   HTTP   │────▶│  │  │   Port Interfaces (.h)   │  │  │────▶│  File    │
  │  Handler │     │  │  └──────────────────────────┘  │  │     │  System  │
  └──────────┘     │  │                                 │  │     └──────────┘
                    │  └───────────────────────────────┘  │
                    │                                     │
                    └─────────────────────────────────────┘

  Adapters              Ports (Header Interfaces)            Adapters
  (Driving)             (Domain Core Logic)                  (Driven)
```

### B. Implementation in C

**Ports are header files defining function pointer tables (vtables). Adapters implement them.**

```c
// === PORT (Interface) === //
// src/ports/repository.h
#ifndef PROJECT_REPOSITORY_PORT_H
#define PROJECT_REPOSITORY_PORT_H

#include "core/user.h"

/**
 * @brief Repository port - interface for user persistence.
 *
 * Adapters implement this interface to provide concrete storage
 * (SQLite, PostgreSQL, in-memory, file, etc.).
 */
typedef struct user_repository {
    int (*save)(void *ctx, const user_t *user);
    user_t *(*find_by_id)(void *ctx, int id);
    int (*delete)(void *ctx, int id);
    void (*destroy)(void *ctx);
    void *ctx;  /**< Adapter-specific context (opaque) */
} user_repository_t;

#endif

// === ADAPTER (Concrete Implementation) === //
// src/adapters/sqlite_user_repo.h
#ifndef PROJECT_SQLITE_USER_REPO_H
#define PROJECT_SQLITE_USER_REPO_H

#include "ports/repository.h"

/**
 * @brief Creates a SQLite-backed user repository.
 * @param db_path Path to the SQLite database file.
 * @return Initialized repository, or NULL on failure.
 */
user_repository_t *sqlite_user_repo_create(const char *db_path);

#endif

// src/adapters/sqlite_user_repo.c
#include "adapters/sqlite_user_repo.h"
#include <sqlite3.h>
#include <stdlib.h>

typedef struct {
    sqlite3 *db;
} sqlite_ctx_t;

static int sqlite_save(void *ctx, const user_t *user) {
    sqlite_ctx_t *sctx = (sqlite_ctx_t *)ctx;
    // ... SQLite INSERT implementation ...
    return 0;
}

static user_t *sqlite_find_by_id(void *ctx, int id) {
    sqlite_ctx_t *sctx = (sqlite_ctx_t *)ctx;
    // ... SQLite SELECT implementation ...
    return NULL;
}

static int sqlite_delete(void *ctx, int id) {
    sqlite_ctx_t *sctx = (sqlite_ctx_t *)ctx;
    // ... SQLite DELETE implementation ...
    return 0;
}

static void sqlite_destroy(void *ctx) {
    sqlite_ctx_t *sctx = (sqlite_ctx_t *)ctx;
    if (sctx) {
        sqlite3_close(sctx->db);
        free(sctx);
    }
}

user_repository_t *sqlite_user_repo_create(const char *db_path) {
    user_repository_t *repo = calloc(1, sizeof(*repo));
    if (!repo) return NULL;

    sqlite_ctx_t *sctx = calloc(1, sizeof(*sctx));
    if (!sctx) { free(repo); return NULL; }

    if (sqlite3_open(db_path, &sctx->db) != SQLITE_OK) {
        free(sctx);
        free(repo);
        return NULL;
    }

    repo->save = sqlite_save;
    repo->find_by_id = sqlite_find_by_id;
    repo->delete = sqlite_delete;
    repo->destroy = sqlite_destroy;
    repo->ctx = sctx;
    return repo;
}

// === DOMAIN SERVICE (Uses ports, not adapters) === //
// src/core/user_service.c
#include "core/user_service.h"
#include "ports/repository.h"

int user_service_register(user_repository_t *repo, const char *name, const char *email) {
    // Domain logic: validate, create, persist
    if (!name || !email) return -1;

    user_t *user = user_create(name, email);
    if (!user) return -1;

    int rc = repo->save(repo->ctx, user);
    user_destroy(user);
    return rc;
}
```

**Benefits:**
- Swap database backends without changing business logic
- Test domain logic with in-memory mock repositories
- Clear dependency direction: adapters depend on ports, never the reverse

---

## 5. Design Patterns for C (MANDATORY)

### A. Opaque Pointer Pattern (Information Hiding)

**Use opaque pointers for encapsulation (C's equivalent of private classes):**

```c
// widget.h - Public API (users see only the pointer type)
#ifndef WIDGET_H
#define WIDGET_H

#include <stddef.h>

/**
 * @brief Opaque widget handle.
 */
typedef struct widget widget_t;

widget_t *widget_create(const char *label, int width, int height);
void widget_destroy(widget_t *w);
int widget_set_label(widget_t *w, const char *label);
const char *widget_get_label(const widget_t *w);

#endif

// widget.c - Private implementation (struct definition hidden)
#include "widget.h"
#include <stdlib.h>
#include <string.h>

struct widget {
    char *label;
    int width;
    int height;
};

widget_t *widget_create(const char *label, int width, int height) {
    widget_t *w = calloc(1, sizeof(*w));
    if (!w) return NULL;

    w->label = strdup(label);
    if (!w->label) { free(w); return NULL; }

    w->width = width;
    w->height = height;
    return w;
}

void widget_destroy(widget_t *w) {
    if (w) {
        free(w->label);
        free(w);
    }
}
```

### B. Constructor/Destructor Pattern (Resource Management)

**Use `create`/`destroy` pairs for RAII-like resource management:**

```c
/**
 * @brief Resource lifecycle pattern.
 *
 * Every resource-owning type MUST provide:
 *   - type_create()  → allocates and initializes
 *   - type_destroy() → releases all resources (NULL-safe)
 *
 * Use goto-cleanup for multi-resource functions.
 */

int process_data(const char *input_path, const char *output_path) {
    int result = -1;
    FILE *input = NULL;
    FILE *output = NULL;
    char *buffer = NULL;

    input = fopen(input_path, "r");
    if (!input) {
        goto cleanup;
    }

    output = fopen(output_path, "w");
    if (!output) {
        goto cleanup;
    }

    buffer = malloc(BUFFER_SIZE);
    if (!buffer) {
        goto cleanup;
    }

    // ... process data ...
    result = 0;  // Success

cleanup:
    free(buffer);
    if (output) fclose(output);
    if (input) fclose(input);
    return result;
}
```

### C. Vtable Pattern (Runtime Polymorphism)

**Use function pointer tables for interface-based programming:**

```c
// logger.h - Logger interface
typedef struct logger {
    void (*log)(void *ctx, const char *level, const char *msg);
    void (*destroy)(void *ctx);
    void *ctx;
} logger_t;

// console_logger.c - Console implementation
static void console_log(void *ctx, const char *level, const char *msg) {
    (void)ctx;
    fprintf(stderr, "[%s] %s\n", level, msg);
}

logger_t *console_logger_create(void) {
    logger_t *l = calloc(1, sizeof(*l));
    if (!l) return NULL;
    l->log = console_log;
    l->destroy = NULL;
    l->ctx = NULL;
    return l;
}

// file_logger.c - File implementation
typedef struct {
    FILE *fp;
} file_logger_ctx_t;

static void file_log(void *ctx, const char *level, const char *msg) {
    file_logger_ctx_t *fctx = ctx;
    fprintf(fctx->fp, "[%s] %s\n", level, msg);
}

static void file_logger_destroy(void *ctx) {
    file_logger_ctx_t *fctx = ctx;
    if (fctx) {
        fclose(fctx->fp);
        free(fctx);
    }
}

logger_t *file_logger_create(const char *path) {
    logger_t *l = calloc(1, sizeof(*l));
    if (!l) return NULL;

    file_logger_ctx_t *fctx = calloc(1, sizeof(*fctx));
    if (!fctx) { free(l); return NULL; }

    fctx->fp = fopen(path, "a");
    if (!fctx->fp) { free(fctx); free(l); return NULL; }

    l->log = file_log;
    l->destroy = file_logger_destroy;
    l->ctx = fctx;
    return l;
}

// Usage (polymorphic - works with any logger)
void application_run(logger_t *log) {
    log->log(log->ctx, "INFO", "Application started");
}
```

### D. Error Code + Output Parameter Pattern

**Standard pattern for functions that return results and errors:**

```c
/**
 * @brief Parses an integer from a string.
 * @param str Input string (must not be NULL).
 * @param out_value Output: parsed integer value.
 * @return 0 on success, -1 on invalid input, -2 on overflow.
 */
int parse_int(const char *str, int *out_value) {
    assert(str != NULL);
    assert(out_value != NULL);

    char *endptr = NULL;
    errno = 0;
    long val = strtol(str, &endptr, 10);

    if (endptr == str || *endptr != '\0') {
        return -1;  // Invalid input
    }
    if (errno == ERANGE || val > INT_MAX || val < INT_MIN) {
        return -2;  // Overflow
    }

    *out_value = (int)val;
    return 0;
}
```

---

## 6. Configuration & Environment (MANDATORY)

### A. Configuration Management

**Use a structured configuration approach with environment variables and config files:**

```c
// config.h
#ifndef PROJECT_CONFIG_H
#define PROJECT_CONFIG_H

#include <stdbool.h>
#include <stddef.h>

/**
 * @brief Application configuration.
 */
typedef struct config {
    char *db_path;
    int port;
    int log_level;
    bool debug_mode;
    size_t max_connections;
} config_t;

/**
 * @brief Loads configuration from environment variables.
 * @param cfg Output configuration struct.
 * @return 0 on success, -1 on error.
 */
int config_load_from_env(config_t *cfg);

/**
 * @brief Frees resources owned by configuration.
 */
void config_cleanup(config_t *cfg);

#endif

// config.c
#include "config.h"
#include <stdlib.h>
#include <string.h>

static const char *getenv_or(const char *name, const char *fallback) {
    const char *val = getenv(name);
    return val ? val : fallback;
}

int config_load_from_env(config_t *cfg) {
    if (!cfg) return -1;

    cfg->db_path = strdup(getenv_or("APP_DB_PATH", "data.db"));
    if (!cfg->db_path) return -1;

    cfg->port = atoi(getenv_or("APP_PORT", "8080"));
    cfg->log_level = atoi(getenv_or("APP_LOG_LEVEL", "2"));
    cfg->debug_mode = getenv("APP_DEBUG") != NULL;
    cfg->max_connections = (size_t)atoi(getenv_or("APP_MAX_CONN", "100"));
    return 0;
}

void config_cleanup(config_t *cfg) {
    if (cfg) {
        free(cfg->db_path);
        cfg->db_path = NULL;
    }
}
```

### B. Environment Variables

**Required environment variables:**

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `APP_DB_PATH` | Path to the database file | `data.db` | No |
| `APP_PORT` | Server listening port | `8080` | No |
| `APP_LOG_LEVEL` | Log verbosity (0=OFF, 1=ERROR, 2=WARN, 3=INFO, 4=DEBUG) | `2` | No |
| `APP_DEBUG` | Enable debug mode (any value = on) | unset | No |
| `APP_MAX_CONN` | Maximum concurrent connections | `100` | No |

---

## 7. Logging & Observability (MANDATORY)

### A. Structured Logging

**Use structured, leveled logging:**

```c
// logging.h
#ifndef PROJECT_LOGGING_H
#define PROJECT_LOGGING_H

#include <stdio.h>
#include <time.h>

typedef enum {
    LOG_LEVEL_OFF = 0,
    LOG_LEVEL_ERROR,
    LOG_LEVEL_WARN,
    LOG_LEVEL_INFO,
    LOG_LEVEL_DEBUG,
} log_level_t;

/**
 * @brief Sets the global minimum log level.
 */
void log_set_level(log_level_t level);

/**
 * @brief Sets the log output stream (default: stderr).
 */
void log_set_output(FILE *stream);

/**
 * @brief Structured log macro with timestamp, level, file, and line.
 */
#define LOG_ERROR(fmt, ...) \
    log_write(LOG_LEVEL_ERROR, __FILE__, __LINE__, fmt, ##__VA_ARGS__)
#define LOG_WARN(fmt, ...)  \
    log_write(LOG_LEVEL_WARN,  __FILE__, __LINE__, fmt, ##__VA_ARGS__)
#define LOG_INFO(fmt, ...)  \
    log_write(LOG_LEVEL_INFO,  __FILE__, __LINE__, fmt, ##__VA_ARGS__)
#define LOG_DEBUG(fmt, ...) \
    log_write(LOG_LEVEL_DEBUG, __FILE__, __LINE__, fmt, ##__VA_ARGS__)

/**
 * @brief Internal logging function (use LOG_* macros instead).
 */
void log_write(log_level_t level, const char *file, int line,
               const char *fmt, ...) __attribute__((format(printf, 4, 5)));

#endif
```

### B. Log Output Format

**Structured log format for machine-parseable output:**

```
2026-03-11T10:15:30Z [INFO ] src/core/server.c:42  Server started on port 8080
2026-03-11T10:15:31Z [ERROR] src/adapters/db.c:87  Database connection failed: SQLITE_CANTOPEN
2026-03-11T10:15:31Z [DEBUG] src/core/config.c:23  Config loaded: port=8080, debug=true
```

---

## 8. Testing (MANDATORY)

### A. Unit Tests

**Use CUnit, Unity, Check, or simple assert-based test harness for comprehensive coverage:**

```c
// tests/unit/test_user.c
#include <assert.h>
#include <string.h>
#include <stdio.h>
#include "core/user.h"

/* ---- Test Fixtures ---- */

static void test_user_create_valid(void) {
    user_t *u = user_create("Alice", "alice@example.com");
    assert(u != NULL);
    assert(strcmp(user_get_name(u), "Alice") == 0);
    assert(strcmp(user_get_email(u), "alice@example.com") == 0);
    user_destroy(u);
    printf("  PASS: test_user_create_valid\n");
}

static void test_user_create_null_name_returns_null(void) {
    user_t *u = user_create(NULL, "alice@example.com");
    assert(u == NULL);
    printf("  PASS: test_user_create_null_name_returns_null\n");
}

static void test_user_create_empty_email_returns_null(void) {
    user_t *u = user_create("Alice", "");
    assert(u == NULL);
    printf("  PASS: test_user_create_empty_email_returns_null\n");
}

static void test_user_destroy_null_is_safe(void) {
    user_destroy(NULL);  // Must not crash
    printf("  PASS: test_user_destroy_null_is_safe\n");
}

/* ---- Test Runner ---- */

int main(void) {
    printf("Running user tests...\n");
    test_user_create_valid();
    test_user_create_null_name_returns_null();
    test_user_create_empty_email_returns_null();
    test_user_destroy_null_is_safe();
    printf("All user tests passed.\n");
    return 0;
}
```

### B. Integration Tests

```c
// tests/integration/test_sqlite_repo.c
#include <assert.h>
#include <stdio.h>
#include <unistd.h>
#include "adapters/sqlite_user_repo.h"
#include "core/user.h"

static void test_sqlite_round_trip(void) {
    const char *db_path = "/tmp/test_repo.db";
    unlink(db_path);  // Clean slate

    user_repository_t *repo = sqlite_user_repo_create(db_path);
    assert(repo != NULL);

    // Save a user
    user_t *u = user_create("Bob", "bob@test.com");
    assert(u != NULL);
    assert(repo->save(repo->ctx, u) == 0);

    // Retrieve the user
    user_t *found = repo->find_by_id(repo->ctx, 1);
    assert(found != NULL);
    assert(strcmp(user_get_name(found), "Bob") == 0);

    user_destroy(found);
    user_destroy(u);
    repo->destroy(repo->ctx);
    free(repo);
    unlink(db_path);
    printf("  PASS: test_sqlite_round_trip\n");
}

int main(void) {
    printf("Running integration tests...\n");
    test_sqlite_round_trip();
    printf("All integration tests passed.\n");
    return 0;
}
```

### C. Test Coverage Requirements

- Minimum coverage: **80%** for business logic
- Critical paths: **100%** coverage (memory management, error handling)
- All public APIs must have tests
- All tests must run under **AddressSanitizer** and **UndefinedBehaviorSanitizer**
- Memory leak detection with **Valgrind** or ASAN LeakSanitizer

### D. Testing with Sanitizers

```bash
# Build with AddressSanitizer + UndefinedBehaviorSanitizer
cmake -B build-san \
    -DCMAKE_C_COMPILER=gcc \
    -DCMAKE_C_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer -g -O1" \
    -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=address,undefined"
cmake --build build-san
ctest --test-dir build-san --output-on-failure

# Build with ThreadSanitizer (for concurrent code)
cmake -B build-tsan \
    -DCMAKE_C_FLAGS="-fsanitize=thread -fno-omit-frame-pointer -g -O1" \
    -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=thread"
cmake --build build-tsan
ctest --test-dir build-tsan --output-on-failure

# Valgrind memory check
valgrind --leak-check=full --error-exitcode=1 ./build/tests/test_runner
```

---

## 9. Error Handling (MANDATORY)

### A. Error Handling Strategy

**C has no exceptions. Use return codes, output parameters, and `goto cleanup` consistently.**

#### Error Code Convention

```c
// error.h - Project-wide error codes
#ifndef PROJECT_ERROR_H
#define PROJECT_ERROR_H

/**
 * @brief Standard error codes for the project.
 *
 * All functions returning int use these codes:
 *   0  = success
 *   <0 = error (specific code indicates the type)
 */
typedef enum {
    ERR_OK            =  0,
    ERR_NULL_ARG      = -1,
    ERR_OUT_OF_MEMORY = -2,
    ERR_INVALID_INPUT = -3,
    ERR_NOT_FOUND     = -4,
    ERR_IO_FAILURE    = -5,
    ERR_OVERFLOW      = -6,
    ERR_PERMISSION    = -7,
} error_code_t;

/**
 * @brief Returns a human-readable error message.
 */
const char *error_to_string(error_code_t err);

#endif
```

#### Goto-Cleanup Pattern

```c
/**
 * @brief Reads a file, processes it, and writes output.
 * @return ERR_OK on success, negative error code on failure.
 */
int process_file(const char *in_path, const char *out_path) {
    if (!in_path || !out_path) {
        return ERR_NULL_ARG;
    }

    int result = ERR_IO_FAILURE;
    FILE *in = NULL;
    FILE *out = NULL;
    char *buffer = NULL;

    in = fopen(in_path, "rb");
    if (!in) goto cleanup;

    out = fopen(out_path, "wb");
    if (!out) goto cleanup;

    // Get file size
    fseek(in, 0, SEEK_END);
    long size = ftell(in);
    if (size < 0) goto cleanup;
    rewind(in);

    buffer = malloc((size_t)size);
    if (!buffer) { result = ERR_OUT_OF_MEMORY; goto cleanup; }

    if (fread(buffer, 1, (size_t)size, in) != (size_t)size) goto cleanup;

    // ... process buffer ...

    if (fwrite(buffer, 1, (size_t)size, out) != (size_t)size) goto cleanup;

    result = ERR_OK;

cleanup:
    free(buffer);
    if (out) fclose(out);
    if (in) fclose(in);
    return result;
}
```

### B. Common Error Patterns

| Pattern | When to Use | Example |
|---------|-------------|---------|
| Return code (int) | Functions that can fail | `int save(const user_t *u)` → 0 or negative |
| Output parameter | Functions returning a value + error | `int parse(const char *s, int *out)` |
| `goto cleanup` | Functions acquiring multiple resources | File I/O, multi-allocation |
| `errno` | Wrapping libc calls | Check after `fopen`, `malloc` on some systems |
| `assert()` | Internal invariants (debug only) | `assert(ptr != NULL)` for programming errors |
| `static_assert` | Compile-time checks | `static_assert(sizeof(int) >= 4, "need 32-bit int")` |

---

## 10. Documentation (MANDATORY)

### A. Code Documentation

**Follow Doxygen conventions for all public APIs:**

```c
/**
 * @file stack.h
 * @brief Thread-safe bounded stack data structure.
 * @author Project Team
 * @date 2026-03-11
 *
 * Provides a fixed-capacity LIFO stack with O(1) push/pop operations.
 * Not thread-safe unless external synchronization is used.
 *
 * @par Example
 * @code
 * stack_t *s = stack_create(100);
 * stack_push(s, 42);
 *
 * int val;
 * stack_pop(s, &val);  // val == 42
 * stack_destroy(s);
 * @endcode
 */

/**
 * @brief Creates a new stack with the given capacity.
 *
 * Allocates memory for a stack that can hold up to @p capacity elements.
 * The caller is responsible for calling stack_destroy() when done.
 *
 * @param[in] capacity Maximum number of elements (must be > 0).
 * @return Pointer to the new stack, or NULL if allocation fails.
 *
 * @pre capacity > 0
 * @post Returned stack is empty (stack_is_empty() == true).
 *
 * @note The caller owns the returned pointer and must call stack_destroy().
 *
 * @see stack_destroy
 */
stack_t *stack_create(size_t capacity);
```

### B. Generate Documentation

```bash
# Generate Doxygen documentation
doxygen docs/Doxyfile

# View documentation
open docs/html/index.html
```

---

## 11. Security & Dependency Management (MANDATORY)

### A. Safe Standard Library Usage

**NEVER use unsafe functions. Use safe alternatives:**

| Unsafe Function | Safe Alternative | Reason |
|----------------|------------------|--------|
| `gets()` | `fgets()` | Removed in C11 — unbounded read |
| `sprintf()` | `snprintf()` | Buffer overflow risk |
| `strcpy()` | `strncpy()` or `strlcpy()` or manual length check | No bounds checking |
| `strcat()` | `strncat()` or `snprintf()` | No bounds checking |
| `scanf("%s", ...)` | `scanf("%255s", ...)` with width | Unbounded read |
| `atoi()` | `strtol()` with error checking | No error detection |

### B. Memory Safety Rules

```c
// RULE 1: Always check allocation return values
char *buf = malloc(size);
if (!buf) {
    return ERR_OUT_OF_MEMORY;
}

// RULE 2: Don't cast malloc return (C, not C++)
int *arr = malloc(n * sizeof(*arr));  // sizeof(*arr), NOT sizeof(int)

// RULE 3: Use calloc for zero-initialized memory
struct data *d = calloc(1, sizeof(*d));

// RULE 4: Always pair allocations with frees (create/destroy pattern)
// RULE 5: Set pointers to NULL after free to prevent use-after-free
free(ptr);
ptr = NULL;

// RULE 6: Use sizeof(*ptr) not sizeof(type) to avoid type mismatch
widget_t *w = malloc(sizeof(*w));  // Correct: adapts if type changes

// RULE 7: Check for integer overflow before allocation
if (count > SIZE_MAX / sizeof(*arr)) {
    return ERR_OVERFLOW;
}
arr = malloc(count * sizeof(*arr));
```

### C. Dependency Management Strategy (STRICT ORDER)

**CMake is the single tool orchestrating ALL dependency management** (as defined in cmake.md and conan.md). Conan is bootstrapped and invoked automatically from within CMake — the user never runs `conan install` or maintains external dependency files (`conanfile.txt`, `conanfile.py`). Each module manages its own Conan dependencies independently via `conan_cmake_configure()` + `conan_cmake_install()` in its own `CMakeLists.txt`.

**Dependency Resolution Priority (MANDATORY):**

#### 1. **C Standard Library** — ALWAYS prefer first
- `<stdlib.h>`, `<string.h>`, `<stdio.h>`, `<stdint.h>`, etc.
- No external dependency needed for standard functionality.

#### 2. **PRIMARY: Conan Packages (conan-center)** ⭐ PREFERRED
- **ALWAYS check Conan first**: Search https://conan.io/center/
- Use official Conan packages from conan-center
- Specify exact version numbers
- **Conan MUST be used from within CMake** — each module calls `conan_cmake_configure()` + `conan_cmake_install()` directly in its own `CMakeLists.txt` (see conan.md for full pattern)
- Example: `openssl/3.2.0`, `zlib/1.3.1`, `sqlite3/3.45.0`, `cjson/1.7.17`

```cmake
# ✅ CORRECT - Per-module Conan dependencies (PRIMARY pattern from conan.md)
# Each module's CMakeLists.txt declares its own dependencies independently

# Set module-local paths for Conan-generated files
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_BINARY_DIR})
list(APPEND CMAKE_PREFIX_PATH ${CMAKE_CURRENT_BINARY_DIR})

# Declare and install this module's Conan dependencies
conan_cmake_configure(
    REQUIRES openssl/3.2.0 zlib/1.3.1 cjson/1.7.17
    GENERATORS CMakeDeps CMakeToolchain
)
conan_cmake_install(
    PATH_OR_REFERENCE .
    BUILD missing
    REMOTE conancenter
    SETTINGS ${CONAN_SETTINGS}
)

# Then use standard CMake find_package (Conan generates config files)
find_package(OpenSSL REQUIRED)
find_package(ZLIB REQUIRED)
find_package(cJSON REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE OpenSSL::SSL ZLIB::ZLIB cjson::cjson)
```

> **Alternative**: The `add_conan_dependencies()` wrapper from `cmake/ConanIntegration.cmake` is also acceptable — it wraps the above calls into a convenience function. See conan.md for both patterns.

#### Conan Bootstrap (cmake/conan/CMakeLists.txt)

**PRIMARY pattern (from conan.md). Bootstraps Conan once via `add_subdirectory(cmake/conan)` in the root CMakeLists.txt.**

```cmake
# cmake/conan/CMakeLists.txt - Conan Bootstrap
# Purpose: Download conan.cmake and autodetect settings ONCE.
#          All modules then use conan_cmake_configure() / conan_cmake_install()
#          independently in their own CMakeLists.txt.

cmake_minimum_required(VERSION 3.15)

# Download conan.cmake if not already cached
if(NOT EXISTS "${CMAKE_BINARY_DIR}/conan.cmake")
    message(STATUS "Downloading conan.cmake from https://github.com/conan-io/cmake-conan")
    file(DOWNLOAD
        "https://raw.githubusercontent.com/conan-io/cmake-conan/0.18.1/conan.cmake"
        "${CMAKE_BINARY_DIR}/conan.cmake"
        TLS_VERIFY ON
    )
endif()

include(${CMAKE_BINARY_DIR}/conan.cmake)

# Autodetect compiler, OS, architecture, build type
conan_cmake_autodetect(settings)

# Make settings available to all subdirectories
set(CONAN_SETTINGS ${settings} CACHE INTERNAL "Conan autodetected settings")
```

#### 3. **SECONDARY: System Packages** (Only if not in Conan)
- **Use ONLY if package is NOT available in Conan**
- `pkg-config` or `find_package()` based dependencies
- Platform-specific package managers:
  - **Ubuntu/Debian**: `apt` (e.g., `libssl-dev`)
  - **Fedora/RHEL**: `dnf`/`yum`
  - **macOS**: `brew`

```cmake
# ✅ CORRECT - System package (only if not in Conan)
find_package(PkgConfig REQUIRED)
pkg_check_modules(SYSTEMD REQUIRED libsystemd)
target_link_libraries(${PROJECT_NAME} PRIVATE ${SYSTEMD_LIBRARIES})
```

#### 4. **TERTIARY: Other Methods** (Last Resort Only)
- **Use ONLY if package is in neither Conan nor system packages**
- Options (in preference order):
  1. CMake FetchContent (for small libraries from Git)
  2. Vendored source — copy into `vendor/` (pin version, track upstream)

```cmake
# ⚠️ LAST RESORT - FetchContent (only if unavailable elsewhere)
include(FetchContent)
FetchContent_Declare(
    cjson
    GIT_REPOSITORY https://github.com/DaveGamble/cJSON.git
    GIT_TAG        v1.7.17
)
FetchContent_MakeAvailable(cjson)
```

### Dependency Decision Tree

```
Need dependency "X"?
│
├─> In C Standard Library? ✅ USE STDLIB (no dependency needed)
│
├─> Search Conan (conan.io/center)
│   ├─> Found? ✅ USE CONAN (per-module conan_cmake_configure/install)
│   │   └─> Add to conan_cmake_configure(REQUIRES X/version ...) in the module's CMakeLists.txt
│   │
│   └─> Not Found? ⤵️
│       │
│       └─> Search system packages (apt/dnf/brew/pkg-config)
│           ├─> Found? ⚠️ USE SYSTEM PACKAGE
│           │   └─> Add find_package(X REQUIRED) or pkg_check_modules()
│           │   └─> Document in README that users need to install system package
│           │
│           └─> Not Found? ⛔ LAST RESORT
│               └─> Use FetchContent or vendor the source
│               └─> Document why Conan/system wasn't used
```

### Prohibited Practices

❌ **NEVER do these**:
- Running `conan install` outside of CMake as a separate manual step — CMake is the single orchestrator
- Maintaining a centralized `conanfile.txt` or `conanfile.py` — dependencies live in each module's `CMakeLists.txt`
- Copy-pasting library source code into your project without vendoring properly
- Committing compiled binaries (`.a`, `.so`, `.dll`) to version control
- Using random GitHub repositories without version tags
- Hardcoding library paths (e.g., `/usr/local/lib/libfoo.a`)

✅ **ALWAYS do these**:
- Use per-module `conan_cmake_configure()` + `conan_cmake_install()` in each module's `CMakeLists.txt` (or the `add_conan_dependencies()` wrapper)
- Bootstrap Conan via `add_subdirectory(cmake/conan)` in the root `CMakeLists.txt` (see conan.md)
- Pin exact version numbers
- Document which dependencies come from where
- Test on a clean system (Docker) to verify dependencies

### D. Vulnerability Scanning

```bash
# Scan dependencies for known CVEs (Conan)
conan audit .

# Static analysis for security issues
cppcheck --enable=all --error-exitcode=1 --suppress=missingInclude src/

# Clang-Tidy for security-focused checks
clang-tidy src/**/*.c -checks='-*,bugprone-*,cert-*,clang-analyzer-*'

# Build and run with sanitizers (catches UB, buffer overflows, use-after-free)
cmake -B build -DCMAKE_C_FLAGS="-fsanitize=address,undefined" && cmake --build build && ctest --test-dir build
```

---

## 12. Modern C Features (C17/C23)

### A. C17 Baseline

C17 (ISO/IEC 9899:2018) is the minimum standard. Key features to use:

```c
// static_assert without message (C11, but use it)
static_assert(sizeof(void *) >= 4);

// _Generic for type-safe macros
#define print_value(x) _Generic((x), \
    int:    printf("%d\n", (x)),     \
    double: printf("%f\n", (x)),     \
    char *: printf("%s\n", (x))      \
)

// _Alignas / _Alignof for alignment control
_Alignas(64) char cache_line[64];

// _Noreturn for functions that don't return
_Noreturn void fatal_error(const char *msg) {
    fprintf(stderr, "FATAL: %s\n", msg);
    abort();
}

// Anonymous structs and unions (C11)
typedef struct {
    union {
        struct { float x, y, z; };
        float v[3];
    };
} vec3_t;
```

### B. C23 Modern Features

C23 (ISO/IEC 9899:2024) — use when compiler support is available (GCC 15+, Clang 18+):

```c
// nullptr - type-safe null pointer constant
int *ptr = nullptr;
if (ptr == nullptr) { /* ... */ }

// constexpr - compile-time constants
constexpr int MAX_BUFFER_SIZE = 4096;
constexpr double PI = 3.14159265358979323846;

// typeof / typeof_unqual - type inference
int x = 42;
typeof(x) y = x;  // y is int

// auto type inference (C23)
auto result = some_function();

// [[nodiscard]] - warn if return value is ignored
[[nodiscard]] int critical_operation(void);

// [[maybe_unused]] - suppress unused warnings
void callback([[maybe_unused]] void *ctx) { /* ... */ }

// [[deprecated]] - mark deprecated APIs
[[deprecated("Use new_api() instead")]]
void old_api(void);

// Binary literals
int mask = 0b11110000;

// Digit separators for readability
long population = 8'000'000'000L;

// Empty initializer
struct point { int x; int y; };
struct point origin = {};  // Zero-initialized

// static_assert is now a keyword (no header needed)
static_assert(sizeof(int) == 4, "Expected 32-bit int");

// bool, true, false are keywords (no <stdbool.h> needed)
bool flag = true;

// Improved enums with fixed underlying type
enum color : unsigned char { RED, GREEN, BLUE };
```

### C. Feature Detection

```c
// Use __STDC_VERSION__ to detect standard level
#if __STDC_VERSION__ >= 202311L
    // C23 features available
    #define HAS_NULLPTR 1
    #define HAS_CONSTEXPR 1
    #define HAS_TYPEOF 1
#elif __STDC_VERSION__ >= 201710L
    // C17 features available
    #define HAS_NULLPTR 0
    #define HAS_CONSTEXPR 0
    #define HAS_TYPEOF 0
#endif

// Compiler-specific feature macros
#ifdef __has_c_attribute
    #if __has_c_attribute(nodiscard)
        #define NODISCARD [[nodiscard]]
    #else
        #define NODISCARD
    #endif
#else
    #define NODISCARD
#endif
```

---

## 13. Build System Integration (MANDATORY)

### A. CMakeLists.txt Template

```cmake
cmake_minimum_required(VERSION 3.25)
project(myproject VERSION 1.0.0 LANGUAGES C)

# ── C Standard ──────────────────────────────────────────────────────────
set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

# ── CMake Modules (following cmake.md pattern) ────────────────────────
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

# Bootstrap Conan (downloads conan.cmake, autodetects settings)
# Makes conan_cmake_configure/conan_cmake_install available to all modules
add_subdirectory(cmake/conan)

# ── Compiler Warnings ──────────────────────────────────────────────────
add_compile_options(
    -Wall -Wextra -Werror -Wpedantic
    -Wshadow -Wconversion -Wsign-conversion
    -Wstrict-prototypes -Wmissing-prototypes
    -Wdouble-promotion -Wformat=2
    -Wnull-dereference -Wuninitialized
)

# ── Sanitizer Option ──────────────────────────────────────────────────
option(ENABLE_SANITIZERS "Enable ASAN + UBSAN" OFF)
if(ENABLE_SANITIZERS)
    add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer)
    add_link_options(-fsanitize=address,undefined)
endif()

# ── Source Library ─────────────────────────────────────────────────────
add_library(core STATIC
    src/core/domain.c
    src/core/services.c
    src/utils/logging.c
    src/utils/error.c
)
target_include_directories(core PUBLIC src/)

# ── Main Executable ───────────────────────────────────────────────────
add_executable(app src/main.c)
target_link_libraries(app PRIVATE core)

# ── Testing ───────────────────────────────────────────────────────────
enable_testing()

add_executable(test_domain tests/unit/test_domain.c)
target_link_libraries(test_domain PRIVATE core)
add_test(NAME test_domain COMMAND test_domain)

add_executable(test_services tests/unit/test_services.c)
target_link_libraries(test_services PRIVATE core)
add_test(NAME test_services COMMAND test_services)

# ── Doxygen Documentation ────────────────────────────────────────────
find_package(Doxygen)
if(DOXYGEN_FOUND)
    set(DOXYGEN_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/docs)
    doxygen_add_docs(docs src/ include/ COMMENT "Generate API docs")
endif()
```

### B. CMake Presets

```json
{
    "version": 6,
    "configurePresets": [
        {
            "name": "dev",
            "binaryDir": "build/dev",
            "cacheVariables": {
                "CMAKE_BUILD_TYPE": "Debug",
                "CMAKE_C_STANDARD": "17",
                "ENABLE_SANITIZERS": "ON"
            }
        },
        {
            "name": "release",
            "binaryDir": "build/release",
            "cacheVariables": {
                "CMAKE_BUILD_TYPE": "Release",
                "CMAKE_C_STANDARD": "17"
            }
        },
        {
            "name": "sanitize",
            "binaryDir": "build/sanitize",
            "cacheVariables": {
                "CMAKE_BUILD_TYPE": "Debug",
                "ENABLE_SANITIZERS": "ON"
            }
        }
    ],
    "buildPresets": [
        { "name": "dev", "configurePreset": "dev" },
        { "name": "release", "configurePreset": "release" }
    ],
    "testPresets": [
        { "name": "dev", "configurePreset": "dev", "output": { "outputOnFailure": true } }
    ]
}
```

### C. Makefile Wrapper

```makefile
# Makefile - Convenience wrapper around CMake
.PHONY: all build test clean format lint docs

BUILD_DIR ?= build/dev
PRESET    ?= dev

all: build

build:
	cmake --preset $(PRESET)
	cmake --build $(BUILD_DIR) -j$(nproc)

test: build
	ctest --test-dir $(BUILD_DIR) --output-on-failure

clean:
	rm -rf build/

format:
	find src/ tests/ -name '*.c' -o -name '*.h' | xargs clang-format -i

lint:
	cppcheck --enable=all --error-exitcode=1 --suppress=missingInclude src/
	clang-tidy src/**/*.c -- -Isrc/

sanitize:
	cmake --preset sanitize
	cmake --build build/sanitize -j$(nproc)
	ctest --test-dir build/sanitize --output-on-failure

valgrind: build
	valgrind --leak-check=full --error-exitcode=1 ./$(BUILD_DIR)/app

docs:
	cmake --build $(BUILD_DIR) --target docs

# ── Convenience targets ───────────────────────────────────────────────
check: format lint test sanitize  ## Run all checks (format, lint, test, sanitize)

release:
	cmake --preset release
	cmake --build build/release -j$(nproc)
```

---

## 14. Deployment Checklist

### Agent-Generated Code Verification (MANDATORY)

**If code was generated/modified by an agent, verify BEFORE delivery:**

#### Build & Compilation
- [ ] Code compiles: `cmake --build build` returns exit code 0
- [ ] No compilation warnings with `-Wall -Wextra -Werror -Wpedantic`
- [ ] All `#include` statements resolve correctly
- [ ] Code formatted: `clang-format --dry-run --Werror` produces no changes

#### Testing
- [ ] All tests pass: `ctest --test-dir build` returns exit code 0
- [ ] Reasonable coverage: `gcov` / `lcov` shows >80% for business logic
- [ ] Integration tests pass (if applicable)
- [ ] Tests run under ASAN/UBSAN without violations

#### Security
- [ ] Dependency scan passes: 0 known vulnerabilities
- [ ] No unsafe function usage (`gets`, `sprintf`, `strcpy`, `strcat`)
- [ ] No hardcoded secrets or sensitive data
- [ ] Static analysis: `cppcheck` and `clang-tidy` pass
- [ ] All inputs validated at API boundaries

#### Memory Safety
- [ ] No memory leaks (Valgrind or ASAN LeakSanitizer)
- [ ] No use-after-free, double-free, buffer overflows (ASAN)
- [ ] No undefined behavior (UBSAN)
- [ ] No data races (TSAN, for threaded code)
- [ ] All `malloc`/`calloc` return values checked
- [ ] All resources have matching `free`/`fclose`/`destroy`

#### Code Quality
- [ ] Linter passes: `cppcheck --enable=all`
- [ ] No unused variables or functions
- [ ] No global mutable state (unless justified)
- [ ] Opaque types used for encapsulation
- [ ] Consistent error handling (error codes + goto cleanup)
- [ ] `const` used wherever applicable

#### Documentation
- [ ] All public APIs have Doxygen comments
- [ ] `@param`, `@return`, `@pre`, `@post` documented
- [ ] Code examples provided for complex APIs

#### Architecture
- [ ] Hexagonal architecture followed (ports and adapters)
- [ ] Dependencies point inward only
- [ ] Modules use opaque types for encapsulation
- [ ] No circular dependencies between modules

#### Agent Workflow Completed
- [ ] Agent verified code compiles with `-Wall -Wextra -Werror`
- [ ] Agent ran all tests and verified they pass
- [ ] Agent ran sanitizers (ASAN + UBSAN)
- [ ] Agent ran static analysis
- [ ] Agent verified Doxygen documentation generates
- [ ] Agent documented any fixes made during verification

---

## 15. Why This Configuration Works

**Memory Safety Without Garbage Collection**:
- The combination of opaque types, `create`/`destroy` patterns, `goto cleanup`, and sanitizer enforcement provides memory safety comparable to managed languages while retaining C's zero-overhead performance. Every memory issue is caught before delivery through ASAN, UBSAN, and Valgrind.

**Testability Through Architecture**:
- Hexagonal architecture with vtable-based ports makes every module independently testable. Mock repositories and loggers can be injected without modifying business logic. TDD ensures tests exist before code.

**Modern Standards With Backwards Compatibility**:
- Targeting C17 as baseline with C23 features behind feature detection macros provides access to modern safety improvements (`nullptr`, `constexpr`, `[[nodiscard]]`) while maintaining portability to older toolchains.

**Compile-Time Error Prevention**:
- Strict compiler flags (`-Wall -Wextra -Werror -Wpedantic -Wshadow -Wconversion`) combined with `static_assert`, `clang-tidy`, and `cppcheck` catch entire categories of bugs at compile time rather than runtime.

**Reproducible Builds**:
- CMake Presets, Conan lockfiles, and pinned dependency versions ensure every developer and CI system produces identical builds. CMake is the single orchestrator via `cmake/conan/CMakeLists.txt` bootstrap + per-module `conan_cmake_configure()`/`conan_cmake_install()` — no standalone `conan install` steps, no centralized dependency files. No "works on my machine" failures. See conan.md for the full pattern.

---

## 16. Quick Reference

### Common Commands

```bash
# Configure (development with sanitizers)
cmake --preset dev

# Build
cmake --build build/dev -j$(nproc)

# Test
ctest --test-dir build/dev --output-on-failure

# Lint
cppcheck --enable=all --error-exitcode=1 src/

# Format
clang-format -i src/**/*.c src/**/*.h

# Static Analysis
clang-tidy src/**/*.c -- -Isrc/

# Sanitizer Run
make sanitize

# Memory Check
valgrind --leak-check=full ./build/dev/app

# Documentation
cmake --build build/dev --target docs

# Full Verification
make check

# Release Build
make release
```

### Coding Style Quick Reference

```c
/* ── Naming Conventions ────────────────────────────────────────────── */
// Types:          snake_case_t      (user_t, buffer_t, config_t)
// Functions:      module_verb_noun  (user_create, stack_push, config_load)
// Constants:      UPPER_SNAKE_CASE  (MAX_BUFFER_SIZE, ERR_NOT_FOUND)
// Variables:      snake_case        (user_count, file_path, is_valid)
// Macros:         UPPER_SNAKE_CASE  (LOG_ERROR, ARRAY_SIZE)
// Enum values:    UPPER_SNAKE_CASE  (LOG_LEVEL_DEBUG, COLOR_RED)
// File names:     snake_case.c/.h   (user_service.c, user_service.h)

/* ── Common Macros ─────────────────────────────────────────────────── */
#define ARRAY_SIZE(arr) (sizeof(arr) / sizeof((arr)[0]))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define UNUSED(x) ((void)(x))

/* ── Function Signature Conventions ────────────────────────────────── */
// Constructors:  type_t *type_create(args...)     → NULL on failure
// Destructors:   void type_destroy(type_t *t)     → NULL-safe
// Getters:       const X *type_get_field(const type_t *t)
// Setters:       int type_set_field(type_t *t, X value) → 0 or error
// Actions:       int type_verb(type_t *t, ...)    → 0 or error
```

### .clang-format Template

```yaml
BasedOnStyle: LLVM
IndentWidth: 4
ColumnLimit: 100
BreakBeforeBraces: Linux
AllowShortFunctionsOnASingleLine: None
AllowShortIfStatementsOnASingleLine: false
AlwaysBreakAfterReturnType: None
SortIncludes: CaseSensitive
IncludeBlocks: Preserve
```

---

**End of Modern C Development Guidelines**
