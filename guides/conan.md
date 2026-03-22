# Conan Dependency Management Guidelines (CMake-Orchestrated)
Mandatory standards and best practices for Conan package management in C/C++ projects. CMake is the single orchestrator — Conan is bootstrapped, configured, and executed entirely from within CMake. Each module manages its own dependencies independently. The user never needs to run Conan directly or maintain external dependency files. Conan 2.x, CMake 3.15+, conan.cmake, CMakeDeps, CMakeToolchain.

---

**Agent Profile**: The Conan Package Manager Expert
**Role**: Senior C/C++ Build & Dependency Management Specialist
**Objective**: Generate production-ready, reproducible, secure dependency configurations using Conan orchestrated entirely from within CMake, with decentralized per-module dependency management.
**Tools**: Conan 2.x, CMake 3.15+, conan.cmake (cmake-conan), CMakeDeps/CMakeToolchain generators, conan audit.

---

## 1. Core Philosophies: CMAKE-ONLY

The agent must adhere to the **CMAKE-ONLY** principles for every Conan integration:

**Test-Driven Development (TDD)**: ALWAYS write tests BEFORE implementation (Red-Green-Refactor cycle mandatory).
**Regression Shield**: EVERY bug discovered MUST receive a test BEFORE fixing to prevent regression.
**Security-First**: Mandatory vulnerability scanning (`conan audit scan`), dependency auditing, and supply chain integrity checks.

- **C**Make is the ONLY Tool: The user runs `cmake` and `cmake --build` — nothing else. Conan is bootstrapped and invoked automatically from within CMake. Zero external commands.
- **M**odule-Local Dependencies: Each library/directory/module declares and manages its OWN Conan dependencies independently via `conan_cmake_configure()` + `conan_cmake_install()` in its own `CMakeLists.txt`.
- **A**uto-Bootstrapped: `conan.cmake` is automatically downloaded during CMake configure. No pre-installation of cmake-conan required.
- **K**nown Versions Only: Pin exact dependency versions (e.g., `fmt/10.2.0`, not `fmt/[>=10.0]`).
- **E**very Module Self-Contained: No centralized `conanfile.txt` or `conanfile.py`. Dependencies live where they are used — in each module's `CMakeLists.txt`.
- **O**paque to the User: The developer does not need to know Conan exists, what dependencies are needed, or how to use Conan. `cmake --build build` just works.
- **N**o External Files: No `conanfile.txt`, no `conanfile.py`, no manual `conan install`. CMake is the single source of truth.
- **L**ocal Scope: Different modules can use different versions of the same dependency without conflict.
- **Y**ield Pre-Built Binaries: Leverage Conan's binary-first approach — download pre-built binaries when available, build from source only when necessary (`BUILD missing`).

**Additional Principles:**

- Conan is a build implementation detail, not a user-facing tool
- The `add_subdirectory()` pattern bootstraps Conan once, then each module uses it independently
- No global dependency list — dependencies are co-located with the code that uses them
- Clean separation: changing one module's dependencies never affects another module

**Verified Dependencies**: Agent-generated code MUST resolve all Conan dependencies and compile successfully from a clean `cmake` configure before delivery.

---

## 2. Agent Code Generation Requirements (MANDATORY)

### A. Verification Protocol

**CRITICAL: Agents MUST verify that all generated CMake+Conan configurations resolve, build, and pass tests before presenting them to the user.**

#### Pre-Delivery Checklist

**Before delivering ANY CMake+Conan code, the agent MUST:**

1. **Build Verification (the ONLY required check)**:
   ```bash
   # This is ALL the user should ever need to run:
   cmake -B build
   cmake --build build
   # Exit code MUST be 0

   # Conan is bootstrapped and all dependencies resolved automatically
   # during the cmake configure step above
   ```
   - **MUST** configure without errors (conan.cmake downloaded, dependencies resolved)
   - **MUST** build without errors (all libraries linked correctly)
   - No manual `conan install` step required

2. **Test Execution**:
   ```bash
   cd build && ctest --output-on-failure
   # Exit code MUST be 0
   ```
   - All tests pass with Conan-provided test dependencies (e.g., GTest)

3. **Security & Dependency Verification**:
   ```bash
   # Scan for known vulnerabilities (CVEs) in all project dependencies
   conan audit scan .

   # Verify specific packages
   conan audit list "openssl/3.2.0"
   ```
   - **MUST** have 0 high/critical vulnerabilities
   - Dependencies MUST be pinned to exact secure versions

4. **Clean Build Verification**:
   ```bash
   # Verify it works from scratch (no cached state)
   rm -rf build
   cmake -B build
   cmake --build build
   ctest --test-dir build --output-on-failure
   # ALL steps MUST succeed
   ```

#### Error Correction Process

If verification fails:

1. **Conan Download Errors**:
   - Verify internet connectivity
   - Check that the conan.cmake download URL is correct and accessible
   - Verify TLS is enabled

2. **Dependency Resolution Errors**:
   - Check package name and version on https://conan.io/center/
   - Verify the `REMOTE conancenter` is accessible
   - Check for version conflicts between modules
   - Ensure `BUILD missing` is specified for source-only packages

3. **CMake Integration Errors**:
   - Verify `CMAKE_MODULE_PATH` and `CMAKE_PREFIX_PATH` include `${CMAKE_CURRENT_BINARY_DIR}`
   - Check that `find_package()` names match Conan package names
   - Verify generators are `CMakeDeps CMakeToolchain`

### B. Agent Workflow Example

**Complete CMake+Conan project generation workflow:**

1. **Generate Project Structure**:
   ```
   project/
   ├── CMakeLists.txt              # Root: bootstraps Conan, adds subdirectories
   ├── cmake/
   │   └── conan/
   │       └── CMakeLists.txt      # Conan bootstrap (downloads conan.cmake)
   ├── src/
   │   ├── core/
   │   │   ├── CMakeLists.txt      # Core lib + its own Conan deps
   │   │   ├── include/core/
   │   │   └── src/
   │   ├── network/
   │   │   ├── CMakeLists.txt      # Network lib + its own Conan deps
   │   │   ├── include/network/
   │   │   └── src/
   │   └── app/
   │       ├── CMakeLists.txt      # App executable
   │       └── main.cpp
   └── tests/
       ├── CMakeLists.txt          # Tests + GTest via Conan
       └── test_core.cpp
   ```

2. **Verify Build**:
   ```bash
   cmake -B build
   cmake --build build
   # ✓ Conan bootstrapped, all dependencies resolved and built
   ```

3. **Run Tests**:
   ```bash
   ctest --test-dir build --output-on-failure
   # ✓ All tests pass
   ```

4. **Security Scan**:
   ```bash
   conan audit scan .
   # ✓ No critical/high vulnerabilities
   ```

5. **Present Code**: Only after ALL checks pass

### C. Prohibited Practices

**NEVER deliver CMake+Conan code that:**
- [ ] Requires the user to run `conan install` manually
- [ ] Requires a `conanfile.txt` or `conanfile.py` for the consumer project
- [ ] Uses centralized dependency lists instead of per-module declarations
- [ ] Fails to auto-download `conan.cmake` during CMake configure
- [ ] Has unresolved Conan dependencies after `cmake -B build`
- [ ] Requires the user to know Conan exists or how to use it
- [ ] Uses Conan 1.x-only generators (`cmake`, `cmake_find_package`, `cmake_paths`)
- [ ] Hardcodes paths to Conan cache directories
- [ ] Has known high/critical CVEs in dependencies
- [ ] **Fixes bugs without adding regression tests first**
- [ ] **Writes implementation before writing tests (violates TDD)**

---

## 2A. Test-Driven Development (TDD) Protocol (MANDATORY)

**CRITICAL: Follow the Red-Green-Refactor cycle for ALL new code.**

### TDD Cycle

```
1. RED: Write a failing test first (test may need a new Conan dependency)
   ↓
2. GREEN: Add the dependency to the module's CMakeLists.txt via conan_cmake_configure()
   ↓
3. REFACTOR: Clean up, pin versions, verify security
   ↓
   Repeat
```

### Example TDD Workflow

```cmake
# Step 1: RED - Write test that needs nlohmann_json (tests/CMakeLists.txt)
# conan_cmake_configure(REQUIRES nlohmann_json/3.11.3 gtest/1.15.0 ...)
# Test file: tests/test_json.cpp uses #include <nlohmann/json.hpp>
# Run: cmake --build build → FAILS - nlohmann/json.hpp not found

# Step 2: GREEN - Add to tests/CMakeLists.txt:
#   conan_cmake_configure(REQUIRES nlohmann_json/3.11.3 gtest/1.15.0
#                         GENERATORS CMakeDeps CMakeToolchain)
#   conan_cmake_install(PATH_OR_REFERENCE . BUILD missing REMOTE conancenter SETTINGS ${settings})
#   find_package(nlohmann_json REQUIRED)
# Run: cmake --build build && ctest → PASSES

# Step 3: REFACTOR - Pin version, scan for CVEs
# conan audit list "nlohmann_json/3.11.3"
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

---

## 3. Project Structure & Organization (MANDATORY)

### A. Standard Project Layout

**Every module manages its own Conan dependencies in its own CMakeLists.txt:**

```
project/
├── CMakeLists.txt                  # Root CMake: bootstraps Conan, orchestrates modules
├── cmake/
│   └── conan/
│       └── CMakeLists.txt          # Conan bootstrap: downloads conan.cmake, autodetects settings
├── src/
│   ├── core/                       # Core library module
│   │   ├── CMakeLists.txt          # Declares: fmt/10.2.0, spdlog/1.12.0
│   │   ├── include/core/
│   │   │   └── types.hpp
│   │   └── src/
│   │       └── types.cpp
│   ├── parser/                     # Parser library module
│   │   ├── CMakeLists.txt          # Declares: readline/8.2, + uses BISON/FLEX
│   │   ├── parser.y
│   │   ├── lexer.l
│   │   └── ast.c
│   ├── network/                    # Network library module
│   │   ├── CMakeLists.txt          # Declares: openssl/3.2.0, libcurl/8.5.0
│   │   ├── include/network/
│   │   └── src/
│   └── app/                        # Application executable
│       ├── CMakeLists.txt          # Links to core, parser, network — no own Conan deps
│       └── main.cpp
├── tests/                          # Tests
│   ├── CMakeLists.txt              # Declares: gtest/1.15.0
│   ├── test_core.cpp
│   └── test_parser.cpp
├── .clang-format
├── .clang-tidy
└── README.md
```

**Key principles:**
- NO `conanfile.txt` or `conanfile.py` in the project root
- Each module's `CMakeLists.txt` is the single source of truth for that module's dependencies
- The `cmake/conan/CMakeLists.txt` bootstraps Conan once; modules use it independently
- A developer only needs to run `cmake -B build && cmake --build build`

### B. Module Organization Principles

1. **Dependencies Live Where They Are Used**:
   ```
   ✅ CORRECT — Each module declares its own dependencies
   src/core/CMakeLists.txt       → conan_cmake_configure(REQUIRES fmt/10.2.0)
   src/network/CMakeLists.txt    → conan_cmake_configure(REQUIRES openssl/3.2.0)
   tests/CMakeLists.txt          → conan_cmake_configure(REQUIRES gtest/1.15.0)

   ❌ WRONG — Centralized dependency file
   conanfile.txt                 → [requires] fmt/10.2.0 openssl/3.2.0 gtest/1.15.0
   ```

2. **Modules Are Self-Contained**: Adding or removing a module does not affect other modules' dependencies.

3. **Different Versions Are Possible**: Module A can use `fmt/10.2.0` while Module B uses `fmt/9.1.0` if needed — each has its own Conan install scope.

---

## 4. CMake-Orchestrated Conan Integration (MANDATORY)

### A. Architecture Overview

**MANDATORY: CMake is the ONLY tool. Conan is an invisible implementation detail.**

```
User Workflow (what the developer sees):
═══════════════════════════════════════════════════════════════

  $ cmake -B build           ← ONLY command needed to configure
  $ cmake --build build      ← ONLY command needed to build
  $ ctest --test-dir build   ← ONLY command needed to test

  That's it. Nothing else. No conan commands. No dependency files.

Internal Workflow (what happens automatically):
═══════════════════════════════════════════════════════════════

  cmake -B build
       │
       ├──► Root CMakeLists.txt
       │    └── add_subdirectory(cmake/conan)
       │         │
       │         ├── Downloads conan.cmake (if not cached)
       │         ├── include(conan.cmake)
       │         └── conan_cmake_autodetect(settings)
       │              └── ${settings} available to all subdirectories
       │
       ├──► add_subdirectory(src/core)
       │    └── conan_cmake_configure(REQUIRES fmt/10.2.0 ...)
       │    └── conan_cmake_install(... SETTINGS ${settings})
       │    └── find_package(fmt REQUIRED)
       │    └── target_link_libraries(core PRIVATE fmt::fmt)
       │
       ├──► add_subdirectory(src/network)
       │    └── conan_cmake_configure(REQUIRES openssl/3.2.0 ...)
       │    └── conan_cmake_install(... SETTINGS ${settings})
       │    └── find_package(OpenSSL REQUIRED)
       │    └── target_link_libraries(network PRIVATE OpenSSL::SSL)
       │
       ├──► add_subdirectory(src/app)
       │    └── target_link_libraries(app core network)
       │
       └──► add_subdirectory(tests)
            └── conan_cmake_configure(REQUIRES gtest/1.15.0 ...)
            └── conan_cmake_install(... SETTINGS ${settings})
            └── find_package(GTest REQUIRED)
            └── gtest_discover_tests(...)
```

### B. Conan Bootstrap (cmake/conan/CMakeLists.txt)

**This file is included once via `add_subdirectory()` and bootstraps Conan for the entire project. Each module then uses `conan_cmake_configure()` / `conan_cmake_install()` independently.**

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

**Alternative: Function-based bootstrap (cmake/ConanIntegration.cmake)**

This is the pattern used in cpp.md, c.md, and cmake.md — a CMake module with `setup_conan()` and `add_conan_dependencies()` wrapper functions:

```cmake
# cmake/ConanIntegration.cmake - Conan package management (function-based)
# Purpose: Setup Conan from within CMake (no manual conan install steps)
# Usage: include(ConanIntegration) then setup_conan() in root CMakeLists.txt
#        Then use add_conan_dependencies() in any module CMakeLists.txt

function(setup_conan)
    # Check if Conan is available
    find_program(CONAN_CMD conan)
    if(NOT CONAN_CMD)
        message(WARNING "Conan not found, skipping Conan integration")
        return()
    endif()

    # Download conan.cmake if needed
    if(NOT EXISTS "${CMAKE_BINARY_DIR}/conan.cmake")
        file(DOWNLOAD
            "https://raw.githubusercontent.com/conan-io/cmake-conan/0.18.1/conan.cmake"
            "${CMAKE_BINARY_DIR}/conan.cmake"
            TLS_VERIFY ON
        )
    endif()

    include(${CMAKE_BINARY_DIR}/conan.cmake)

    # Configure Conan
    conan_cmake_autodetect(settings)
    set(CONAN_SETTINGS ${settings} PARENT_SCOPE)
    set(CONAN_AVAILABLE TRUE PARENT_SCOPE)
endfunction()

function(add_conan_dependencies)
    cmake_parse_arguments(CONAN "" "" "REQUIRES" ${ARGN})

    conan_cmake_configure(
        REQUIRES ${CONAN_REQUIRES}
        GENERATORS CMakeDeps CMakeToolchain
    )

    conan_cmake_install(
        PATH_OR_REFERENCE .
        BUILD missing
        REMOTE conancenter
        SETTINGS ${CONAN_SETTINGS}
    )
endfunction()
```

Both patterns achieve the same goal. The `add_subdirectory()` pattern is simpler and more explicit; the function-based pattern provides a cleaner API. Choose one and use it consistently.

### C. Root CMakeLists.txt

```cmake
# CMakeLists.txt - Root orchestrator
# Purpose: Bootstrap Conan, then add all modules.
#          The user runs ONLY cmake — no conan commands needed.

cmake_minimum_required(VERSION 3.15)
project(MyProject C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 17)

# ── Bootstrap Conan (downloads conan.cmake, autodetects settings) ────
add_subdirectory(cmake/conan)

# ── Source modules (each manages its own Conan dependencies) ─────────
add_subdirectory(src/core)
add_subdirectory(src/parser)
add_subdirectory(src/network)

# ── Application ──────────────────────────────────────────────────────
add_subdirectory(src/app)

# ── Tests ────────────────────────────────────────────────────────────
enable_testing()
add_subdirectory(tests)
```

### D. Module CMakeLists.txt Pattern (MANDATORY)

**Every module that needs external dependencies declares them inline:**

```cmake
# src/core/CMakeLists.txt - Core library
# Purpose: Core types and utilities. Dependencies: fmt, spdlog.

cmake_minimum_required(VERSION 3.15)
project(core CXX)

# ── Conan paths for this module ──────────────────────────────────────
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_BINARY_DIR})
list(APPEND CMAKE_PREFIX_PATH ${CMAKE_CURRENT_BINARY_DIR})

# ── Declare and install THIS module's dependencies ───────────────────
conan_cmake_configure(REQUIRES
    fmt/10.2.0
    spdlog/1.12.0
    GENERATORS CMakeDeps CMakeToolchain
)
conan_cmake_install(PATH_OR_REFERENCE .
    BUILD missing
    REMOTE conancenter
    SETTINGS ${settings}
)

# ── Find packages (Conan generated config files above) ───────────────
find_package(fmt REQUIRED)
find_package(spdlog REQUIRED)

# ── Library target ───────────────────────────────────────────────────
add_library(${PROJECT_NAME}
    src/types.cpp
    src/utils.cpp
)

target_include_directories(${PROJECT_NAME}
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(${PROJECT_NAME}
    PRIVATE
        fmt::fmt
        spdlog::spdlog
)

target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_20)
```

### E. More Module Examples

**Network module with its own dependencies:**

```cmake
# src/network/CMakeLists.txt - Network library
# Purpose: HTTP client/server. Dependencies: openssl, libcurl.

cmake_minimum_required(VERSION 3.15)
project(network CXX)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_BINARY_DIR})
list(APPEND CMAKE_PREFIX_PATH ${CMAKE_CURRENT_BINARY_DIR})

# This module's own dependencies (separate from core, parser, etc.)
conan_cmake_configure(REQUIRES
    openssl/3.2.0
    libcurl/8.5.0
    GENERATORS CMakeDeps CMakeToolchain
)
conan_cmake_install(PATH_OR_REFERENCE .
    BUILD missing
    REMOTE conancenter
    SETTINGS ${settings}
)

find_package(OpenSSL REQUIRED)
find_package(CURL REQUIRED)

add_library(${PROJECT_NAME}
    src/client.cpp
    src/server.cpp
)

target_include_directories(${PROJECT_NAME}
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(${PROJECT_NAME}
    PRIVATE
        OpenSSL::SSL
        OpenSSL::Crypto
        CURL::libcurl
)
```

**Parser module with Conan + system tools:**

```cmake
# src/parser/CMakeLists.txt - Parser library
# Purpose: Language parser using BISON/FLEX. Dependencies: readline (Conan).

cmake_minimum_required(VERSION 3.15)
project(parser C)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_BINARY_DIR})
list(APPEND CMAKE_PREFIX_PATH ${CMAKE_CURRENT_BINARY_DIR})

set(BISON_PARSER_OUT ${CMAKE_CURRENT_SOURCE_DIR}/temp/bison_parser.c)
set(BISON_HEADER_OUT ${CMAKE_CURRENT_SOURCE_DIR}/temp/bison_parser.h)
set(FLEX_OUT ${CMAKE_CURRENT_SOURCE_DIR}/temp/flex_lexer.c)

# This module's Conan dependencies
conan_cmake_configure(REQUIRES
    readline/8.2
    GENERATORS CMakeDeps CMakeToolchain
)
conan_cmake_install(PATH_OR_REFERENCE .
    BUILD missing
    REMOTE conancenter
    SETTINGS ${settings}
)

find_package(readline REQUIRED)

# System tools (not from Conan — BISON and FLEX must be installed)
find_package(BISON REQUIRED)
bison_target(${PROJECT_NAME} parser.y ${BISON_PARSER_OUT}
    DEFINES_FILE ${BISON_HEADER_OUT}
    COMPILE_FLAGS -Wcounterexamples
    VERBOSE
)

find_package(FLEX REQUIRED)
flex_target(lexer lexer.l ${FLEX_OUT})
add_flex_bison_dependency(lexer ${PROJECT_NAME})

add_library(${PROJECT_NAME}
    ${FLEX_lexer_OUTPUTS}
    ${BISON_parser_OUTPUTS}
    ast.c
)

target_link_libraries(${PROJECT_NAME} PRIVATE ${readline_LIBRARIES})
target_include_directories(${PROJECT_NAME}
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/temp
        ${readline_INCLUDE_DIRS}
)
```

**Test module with GTest from Conan:**

```cmake
# tests/CMakeLists.txt - Test suite
# Purpose: Unit tests. Dependencies: gtest (from Conan).

cmake_minimum_required(VERSION 3.15)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_BINARY_DIR})
list(APPEND CMAKE_PREFIX_PATH ${CMAKE_CURRENT_BINARY_DIR})

# Prevent overriding parent project compiler/linker settings on Windows
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

# Test dependencies (from Conan)
conan_cmake_configure(REQUIRES
    gtest/1.15.0
    GENERATORS CMakeDeps CMakeToolchain
)
conan_cmake_install(PATH_OR_REFERENCE .
    BUILD missing
    REMOTE conancenter
    SETTINGS ${settings}
)

find_package(GTest REQUIRED)

# ── Core tests ───────────────────────────────────────────────────────
set(target core_tests)
add_executable(${target} test_core.cpp)
target_link_libraries(${target} core gtest::gtest)
include(GoogleTest)
gtest_discover_tests(${target})

# ── Parser tests ─────────────────────────────────────────────────────
set(target parser_tests)
add_executable(${target} test_parser.cpp)
target_link_libraries(${target} parser gtest::gtest)
gtest_discover_tests(${target})
```

**Application module (no own Conan deps — just links to libraries):**

```cmake
# src/app/CMakeLists.txt - Main application
# Purpose: Application executable. No Conan deps — uses project libraries.

cmake_minimum_required(VERSION 3.15)

add_executable(app main.cpp)
target_link_libraries(app PRIVATE core parser network)
install(TARGETS app DESTINATION bin)
```

### F. The Complete Pattern (Visual Summary)

```
┌─────────────────────────────────────────────────────────────────┐
│ Root CMakeLists.txt                                              │
│                                                                  │
│   add_subdirectory(cmake/conan)    ◄── Bootstrap ONCE            │
│       │                                                          │
│       ├── Downloads conan.cmake                                  │
│       ├── include(conan.cmake)                                   │
│       └── conan_cmake_autodetect(settings)                       │
│                                                                  │
│   add_subdirectory(src/core)       ◄── Module manages own deps   │
│       │                                                          │
│       ├── conan_cmake_configure(REQUIRES fmt/10.2.0 spdlog/...)  │
│       ├── conan_cmake_install(... SETTINGS ${settings})          │
│       ├── find_package(fmt REQUIRED)                             │
│       └── target_link_libraries(core fmt::fmt spdlog::spdlog)    │
│                                                                  │
│   add_subdirectory(src/network)    ◄── Module manages own deps   │
│       │                                                          │
│       ├── conan_cmake_configure(REQUIRES openssl/3.2.0 curl/...) │
│       ├── conan_cmake_install(... SETTINGS ${settings})          │
│       ├── find_package(OpenSSL REQUIRED)                         │
│       └── target_link_libraries(network OpenSSL::SSL CURL::...)  │
│                                                                  │
│   add_subdirectory(src/app)        ◄── No Conan deps, just links │
│       └── target_link_libraries(app core network)                │
│                                                                  │
│   add_subdirectory(tests)          ◄── Test deps from Conan      │
│       ├── conan_cmake_configure(REQUIRES gtest/1.15.0)           │
│       ├── conan_cmake_install(... SETTINGS ${settings})          │
│       └── gtest_discover_tests(core_tests)                       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 5. Design Patterns (MANDATORY)

### A. The Three-Line Conan Pattern

**Every module that needs Conan dependencies follows this exact three-step pattern:**

```cmake
# Step 1: Set paths so find_package() looks in the right place
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_BINARY_DIR})
list(APPEND CMAKE_PREFIX_PATH ${CMAKE_CURRENT_BINARY_DIR})

# Step 2: Declare and install dependencies
conan_cmake_configure(REQUIRES
    <package>/<version>
    GENERATORS CMakeDeps CMakeToolchain
)
conan_cmake_install(PATH_OR_REFERENCE .
    BUILD missing
    REMOTE conancenter
    SETTINGS ${settings}
)

# Step 3: Use standard CMake find_package()
find_package(<Package> REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE <Package>::<target>)
```

**Benefits:**
- Consistent across all modules
- Self-documenting — you can see exactly what each module needs
- No hidden dependencies — everything is explicit

### B. Wrapper Function Pattern (add_conan_dependencies)

**For cleaner syntax, use the wrapper from cmake/ConanIntegration.cmake (as defined in cpp.md and c.md):**

```cmake
# Using the wrapper function (setup_conan() must have been called first)
add_conan_dependencies(
    REQUIRES
        fmt/10.2.0
        spdlog/1.12.0
)

# Then standard find_package
find_package(fmt REQUIRED)
find_package(spdlog REQUIRED)
```

This is syntactic sugar over the three-line pattern. The raw `conan_cmake_configure()` + `conan_cmake_install()` pattern and the `add_conan_dependencies()` wrapper are both acceptable. Choose one and use it consistently throughout the project.

### C. Mixed Dependencies (Conan + System)

**Some modules may mix Conan packages with system-installed tools:**

```cmake
# Conan-managed dependencies
conan_cmake_configure(REQUIRES readline/8.2 GENERATORS CMakeDeps CMakeToolchain)
conan_cmake_install(PATH_OR_REFERENCE . BUILD missing REMOTE conancenter SETTINGS ${settings})
find_package(readline REQUIRED)

# System-installed tools (not from Conan)
find_package(BISON REQUIRED)
find_package(FLEX REQUIRED)
```

**Priority order for each dependency:**
1. Conan (conan.io/center) — PREFERRED
2. System packages (apt/dnf/brew) — only if not in Conan
3. FetchContent / vendored — last resort

---

## 6. Configuration & Environment (MANDATORY)

### A. Conan Settings Autodetection

The `conan_cmake_autodetect(settings)` call (in the bootstrap) detects:
- Operating system
- Compiler and version
- Architecture
- Build type (from `CMAKE_BUILD_TYPE`)

These settings are passed to every `conan_cmake_install()` call via `SETTINGS ${settings}`.

### B. Overriding Build Type

```cmake
# In root CMakeLists.txt or cmake/conan/CMakeLists.txt
# Force a specific build type for Conan
set(CMAKE_BUILD_TYPE Release)  # Before conan_cmake_autodetect()

# Or override per-module
conan_cmake_install(PATH_OR_REFERENCE .
    BUILD missing
    REMOTE conancenter
    SETTINGS ${settings}
    SETTINGS build_type=Release  # Override for this module
)
```

### C. Custom Remotes

```cmake
# Use a private Artifactory remote instead of (or in addition to) conancenter
conan_cmake_install(PATH_OR_REFERENCE .
    BUILD missing
    REMOTE mycompany-conan      # Use custom remote
    SETTINGS ${settings}
)
```

### D. Conan Profile Integration

While the user never runs `conan` directly, developers and CI systems can configure Conan profiles for the autodetect to use:

```bash
# Set up a profile (one-time, or in CI setup)
conan profile detect

# Profiles are used automatically by conan_cmake_autodetect()
# The developer still only runs cmake — profiles customize behavior
```

---

## 7. Generators (MANDATORY)

### A. Required Generators

**ALWAYS use `CMakeDeps` and `CMakeToolchain` together:**

| Generator | Purpose |
|-----------|---------|
| `CMakeDeps` | Generates `<pkg>-config.cmake` files so `find_package()` works |
| `CMakeToolchain` | Generates toolchain file with compiler/OS/arch settings |

```cmake
# ✅ CORRECT — Always use both generators
conan_cmake_configure(REQUIRES
    fmt/10.2.0
    GENERATORS CMakeDeps CMakeToolchain
)
```

### B. Prohibited Generators

**NEVER use these deprecated/Conan 1.x generators:**

| Prohibited | Reason |
|-----------|--------|
| `cmake` | Conan 1.x only, not supported in Conan 2.x |
| `cmake_find_package` | Replaced by `CMakeDeps` |
| `cmake_find_package_multi` | Replaced by `CMakeDeps` |
| `cmake_paths` | Replaced by `CMakeToolchain` |

---

## 8. Security & Vulnerability Scanning (MANDATORY)

### A. Conan Audit

**MANDATORY: Scan all dependencies for known CVEs before delivery.**

```bash
# Scan entire dependency graph
conan audit scan .

# Scan with lower severity threshold (include medium)
conan audit scan . --severity-level=7.0

# List CVEs for a specific package
conan audit list "openssl/3.2.0"

# Output in JSON for CI pipelines
conan audit scan . --format=json > audit-report.json
```

### B. CI/CD Security Pipeline

```yaml
# .github/workflows/security.yml
name: Dependency Security Scan
on: [push, pull_request]

jobs:
  audit:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Install Conan
        run: pip install conan
      - name: Scan for vulnerabilities
        run: conan audit scan . --severity-level=9.0
```

### C. Version Pinning Rules

```cmake
# ✅ CORRECT — Exact version pinned
conan_cmake_configure(REQUIRES
    fmt/10.2.0
    openssl/3.2.0
    GENERATORS CMakeDeps CMakeToolchain
)

# ❌ WRONG — Version range (non-deterministic without lockfile)
conan_cmake_configure(REQUIRES
    fmt/[>=10.0]
    GENERATORS CMakeDeps CMakeToolchain
)
```

---

## 9. Testing (MANDATORY)

### A. Test Dependencies via Conan

**Test frameworks are just another Conan dependency, declared in the test module's CMakeLists.txt:**

```cmake
# tests/CMakeLists.txt
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_BINARY_DIR})
list(APPEND CMAKE_PREFIX_PATH ${CMAKE_CURRENT_BINARY_DIR})

conan_cmake_configure(REQUIRES
    gtest/1.15.0
    GENERATORS CMakeDeps CMakeToolchain
)
conan_cmake_install(PATH_OR_REFERENCE .
    BUILD missing
    REMOTE conancenter
    SETTINGS ${settings}
)

find_package(GTest REQUIRED)

add_executable(unit_tests test_core.cpp test_parser.cpp)
target_link_libraries(unit_tests core parser gtest::gtest)

include(GoogleTest)
gtest_discover_tests(unit_tests)
```

### B. Test Coverage Requirements

- Minimum coverage: 80% for business logic
- Critical paths: 100% coverage
- All public APIs must have tests
- All tests must link against Conan-provided dependencies correctly

---

## 10. Error Handling (MANDATORY)

### A. Common Errors and Fixes

| Error | Cause | Fix |
|-------|-------|-----|
| `conan.cmake` download fails | No internet or URL changed | Check connectivity; update URL to latest cmake-conan release |
| `find_package(X) not found` | Missing `CMAKE_MODULE_PATH`/`CMAKE_PREFIX_PATH` | Add `list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_BINARY_DIR})` |
| `ERROR: Package 'X/Y' not found` | Wrong package name or version | Search https://conan.io/center/ for correct name |
| `BUILD missing` fails | No pre-built binary, source build error | Check compiler compatibility; try `BUILD always` to force rebuild |
| `settings` variable empty | Bootstrap not run before module | Ensure `add_subdirectory(cmake/conan)` comes first in root CMakeLists.txt |
| Version conflict between modules | Two modules require incompatible versions | Accept or resolve — Conan installs per-module so conflicts are rare |

### B. Debugging

```cmake
# Enable verbose Conan output in cmake/conan/CMakeLists.txt
set(CONAN_CMAKE_SILENT_OUTPUT OFF)

# Or from command line
cmake -B build -DCONAN_CMAKE_SILENT_OUTPUT=OFF
```

---

## 11. Documentation (MANDATORY)

### A. Module Documentation

**Each module's CMakeLists.txt MUST document its dependencies:**

```cmake
# src/network/CMakeLists.txt - Network library
# Purpose: HTTP client/server with TLS support
# Dependencies (via Conan):
#   - openssl/3.2.0  — TLS/SSL implementation
#   - libcurl/8.5.0  — HTTP client
# System dependencies: none
```

### B. README Documentation

**Document the build process — emphasize that only cmake is needed:**

```markdown
## Building

```bash
# Prerequisites: CMake 3.15+, Conan 2.x, C/C++ compiler
# That's it — all library dependencies are managed automatically.

cmake -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

No manual dependency installation required.
All C/C++ library dependencies are automatically downloaded
and built via Conan during the CMake configure step.
```

---

## 12. Deployment Checklist

### Agent-Generated Code Verification (MANDATORY)

**If code was generated/modified by an agent, verify BEFORE delivery:**

#### Build (from clean state)
- [ ] `rm -rf build && cmake -B build` succeeds (Conan bootstraps, all deps resolve)
- [ ] `cmake --build build` succeeds (all targets compile and link)
- [ ] No manual `conan install` step needed
- [ ] No `conanfile.txt` or `conanfile.py` required

#### Per-Module Dependencies
- [ ] Each module declares its own dependencies in its own CMakeLists.txt
- [ ] No centralized dependency list
- [ ] `conan_cmake_configure()` uses exact pinned versions
- [ ] `GENERATORS CMakeDeps CMakeToolchain` specified
- [ ] `CMAKE_MODULE_PATH` and `CMAKE_PREFIX_PATH` set to `${CMAKE_CURRENT_BINARY_DIR}`

#### Conan Bootstrap
- [ ] `cmake/conan/CMakeLists.txt` downloads conan.cmake automatically
- [ ] `conan_cmake_autodetect(settings)` called once
- [ ] `${settings}` available to all subdirectories

#### Testing
- [ ] `ctest --test-dir build` passes all tests
- [ ] Test dependencies (GTest, etc.) managed via Conan in tests/CMakeLists.txt

#### Security
- [ ] `conan audit scan .` passes with 0 high/critical CVEs
- [ ] All dependency versions are pinned (no ranges)
- [ ] No known vulnerable versions used

#### Code Quality
- [ ] Each module CMakeLists.txt has a purpose comment
- [ ] Dependencies are documented (what and why)
- [ ] Prohibited generators not used (no `cmake`, `cmake_find_package`)

#### Agent Workflow Completed
- [ ] Agent verified clean build from scratch
- [ ] Agent verified all tests pass
- [ ] Agent ran security scan
- [ ] Agent documented any fixes made during verification

---

## 13. Why This Configuration Works

**CMake as the Only Tool**:
- The developer never leaves CMake. `cmake -B build && cmake --build build` handles everything — Conan is bootstrapped, dependencies downloaded, binaries cached, packages linked. Zero context switching, zero external commands, zero dependency files to maintain.

**Decentralized Dependencies**:
- Each module declares its own dependencies where they are used. Adding a new module with new dependencies requires editing only that module's CMakeLists.txt. Removing a module removes its dependencies. No centralized file to keep in sync. No dependency conflicts between unrelated modules.

**Opaque to the User**:
- A new developer clones the repo and runs `cmake -B build && cmake --build build`. They don't need to know Conan exists, what packages to install, or how to configure a package manager. The build just works.

**Binary-First Performance**:
- Conan downloads pre-built binaries when available (matched by OS, compiler, arch, build type). A 30-minute compilation becomes a 30-second download. Source builds happen only via `BUILD missing` when no matching binary exists.

**Self-Documenting**:
- Looking at any module's CMakeLists.txt tells you exactly what external libraries it needs, what versions, and how they're linked. No searching through separate dependency files or lockfiles.

---

## 14. Quick Reference

### Common Commands (User Perspective)

```bash
# ═══════════════════════════════════════════════════════════════════
# BUILD — This is ALL the user needs to know
# ═══════════════════════════════════════════════════════════════════

# Configure (Conan bootstraps automatically)
cmake -B build

# Build
cmake --build build

# Test
ctest --test-dir build --output-on-failure

# Clean rebuild
rm -rf build && cmake -B build && cmake --build build
```

### Module CMakeLists.txt Template

```cmake
# src/<module>/CMakeLists.txt - <Module Name>
# Purpose: <description>
# Dependencies (via Conan): <package>/<version>, ...

cmake_minimum_required(VERSION 3.15)
project(<module> CXX)

# ── Conan dependency resolution ──────────────────────────────────
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_BINARY_DIR})
list(APPEND CMAKE_PREFIX_PATH ${CMAKE_CURRENT_BINARY_DIR})

conan_cmake_configure(REQUIRES
    <package1>/<version>
    <package2>/<version>
    GENERATORS CMakeDeps CMakeToolchain
)
conan_cmake_install(PATH_OR_REFERENCE .
    BUILD missing
    REMOTE conancenter
    SETTINGS ${settings}
)

find_package(<Package1> REQUIRED)
find_package(<Package2> REQUIRED)

# ── Library target ───────────────────────────────────────────────
add_library(${PROJECT_NAME}
    src/impl.cpp
)

target_include_directories(${PROJECT_NAME}
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(${PROJECT_NAME}
    PRIVATE
        <Package1>::<target>
        <Package2>::<target>
)
```

### Dependency Priority Order

```
Need dependency "X"?
│
├─> In C/C++ Standard Library? ✅ USE STDLIB (no dependency needed)
│
├─> Search Conan (conan.io/center)
│   ├─> Found? ✅ USE CONAN (via conan_cmake_configure in module CMakeLists.txt)
│   │   └─> Add to module's conan_cmake_configure(REQUIRES X/version)
│   │   └─> find_package(X REQUIRED) + target_link_libraries(...)
│   │
│   └─> Not Found? ⤵️
│       │
│       └─> Search system packages (apt/dnf/brew)
│           ├─> Found? ⚠️ USE SYSTEM (find_package only, no Conan)
│           │   └─> Document system dependency in README
│           │
│           └─> Not Found? ⛔ LAST RESORT
│               └─> FetchContent or vendor source
│               └─> Document why Conan/system wasn't used
```

### Prohibited Practices

```
❌ NEVER:
  • Require the user to run `conan install` manually
  • Use conanfile.txt or conanfile.py for consumer projects
  • Centralize dependencies in one file instead of per-module
  • Use Conan 1.x generators (cmake, cmake_find_package, cmake_paths)
  • Use version ranges without pinning
  • Skip security scanning (conan audit scan)
  • Hardcode Conan cache paths
  • Require the user to know Conan exists

✅ ALWAYS:
  • Bootstrap Conan from cmake/conan/CMakeLists.txt (or setup_conan())
  • Declare dependencies per-module in each CMakeLists.txt
  • Use CMakeDeps + CMakeToolchain generators
  • Pin exact versions (fmt/10.2.0, not fmt/[>=10.0])
  • Use BUILD missing for source-only packages
  • Use REMOTE conancenter (or your private remote)
  • Pass ${settings} (or ${CONAN_SETTINGS}) to every conan_cmake_install()
  • Run conan audit scan before delivery
  • Document each module's dependencies in CMakeLists.txt comments
```

### Build Automation (Makefile Wrapper)

```makefile
# Makefile - Convenience wrapper (optional)
.PHONY: all build test clean audit

BUILD_DIR ?= build

all: build

build:
	cmake -B $(BUILD_DIR)
	cmake --build $(BUILD_DIR) -j$$(nproc)

test: build
	ctest --test-dir $(BUILD_DIR) --output-on-failure

clean:
	rm -rf $(BUILD_DIR)

audit:
	conan audit scan . --severity-level=7.0

check: build test audit  ## Full verification
```

---

**End of Conan Dependency Management Guidelines (CMake-Orchestrated)**
