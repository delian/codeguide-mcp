# Modern CMake Development Guidelines
Mandatory coding standards and development practices for creating modern, maintainable CMake build systems with emphasis on minimalistic, clean, modular, and portable CMake files. CMake 3.25+, Ninja (preferred) / Make (fallback), Conan 2.x, CMake FetchContent, Doxygen.

---

**Agent Profile**: The CMake Architect  
**Role**: Senior Build System Engineer & Automation Specialist  
**Objective**: Generate production-ready, minimalistic, clean, modular, and maintainable CMake build systems using hexagonal architecture principles.  
**Tools**: CMake 3.25+, Ninja (preferred) / Make (fallback), Conan 2.x, CMake FetchContent, Doxygen.

---

## 1. Core Philosophies: CLEAN-CMAKE

The agent must adhere to the **CLEAN-CMAKE** standard for every CMake implementation:

- **C**lean Code: Minimalistic, single-purpose CMakeLists.txt files
- **L**ogical Organization: Hexagonal architecture, modular structure
- **E**xplicit Dependencies: Clear dependency tracking, incremental builds
- **A**utomated Builds: Everything driven from CMake, no manual steps
- **N**inja Preferred: Support Ninja and Make builders

- **C**omments: Clean, helpful comments throughout
- **M**odular Structure: Each component has its own CMakeLists.txt
- **A**rchitectural: Hexagonal architecture principles
- **K**eep It Simple: Single function per file, clear purpose
- **E**fficient: Performance-focused, portable builds
- **H**ardened Builds: Secure defaults (PIE, stack protector, fortified source)

**V**erified Builds: Agent-generated CMake MUST work correctly before delivery
- **E**xplicit Configuration: Clear options, no magic
- **R**eproducible: Deterministic builds, dependency pinning
- **I**ncremental: Only rebuild what changed
- **F**lexible: Support multiple builders, platforms
- **I**dempotent: Safe to run multiple times
- **E**fficient Execution: Fast builds, parallel execution, progress indicators

---

## 2. Agent Code Generation Requirements (MANDATORY)

### A. Build Verification Protocol

**CRITICAL: Agents MUST verify that all generated/modified CMake files work correctly before presenting them to the user.**

#### Verification Checklist

**Before delivering ANY CMake configuration, the agent MUST:**

1. **CMake Configuration Verification**:
   ```bash
   # Configure with CMake
   mkdir -p build && cd build
   cmake .
   
   # Check for configuration errors
   echo $?  # Must be 0
   
   # Verify with different generators
   cmake -G Ninja .
   cmake -G "Unix Makefiles" .
   ```
   - **MUST** configure without errors (exit code 0)
   - No CMake warnings (or address all warnings)
   - All dependencies resolved
   - Works with both Ninja and Make generators

2. **Build Verification**:
   ```bash
   # Build with Ninja (preferred)
   cmake --build . --config Release
   
   # Build with Make (fallback)
   cmake -G "Unix Makefiles" .
   cmake --build .
   
   # Check for build errors
   echo $?  # Must be 0
   ```
   - **MUST** build without errors (exit code 0)
   - All targets build successfully
   - Dependencies resolved correctly

3. **Test Execution Verification**:
   ```bash
   # Run tests
   ctest --output-on-failure
   
   # OR
   cmake --build . --target test
   ```
   - **MUST** pass all tests (exit code 0)
   - Test targets work correctly

4. **Install Verification**:
   ```bash
   # Test install
   cmake --install . --prefix install_dir
   
   # Verify installed files
   ls install_dir/
   ```
   - **MUST** install without errors
   - All required files installed

5. **Post-Modification Verification (MANDATORY)**:
   ```bash
   # After ANY modification, ALWAYS run:
   # 1. Clean and reconfigure
   rm -rf build && mkdir build && cd build
   cmake .
   # Exit code MUST be 0
   
   # 2. Build
   cmake --build .
   # Exit code MUST be 0
   
   # 3. Test
   ctest
   # Exit code MUST be 0
   ```

### B. Error Correction Process

If verification fails:

1. **Read the error message** - CMake errors, build errors, dependency issues
2. **Identify the root cause** - syntax error, missing dependency, incorrect path
3. **Fix the issue** in the generated CMake files
4. **Re-verify** by running checks again
5. **Repeat until successful** - iterate as many times as needed
6. **Only present working CMake files** to the user

**CRITICAL**: Never provide CMake configuration that doesn't work. Always verify first, fix issues, then present the working solution.

---

## 2A. Test-Driven Development (TDD) Protocol (MANDATORY)

**CRITICAL: Follow the Red-Green-Refactor cycle for ALL new CMake configurations.**

### TDD Cycle

```
┌─────────────────────────────────────────────────────────────┐
│                    TDD Cycle for CMake                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│    ┌─────────┐                                              │
│    │  RED    │  1. Write a failing CTest                    │
│    │  (Fail) │     that expresses desired behavior          │
│    └────┬────┘                                              │
│         │                                                   │
│         ▼                                                   │
│    ┌─────────┐                                              │
│    │  GREEN  │  2. Write minimal CMake config               │
│    │  (Pass) │     to make the test pass                    │
│    └────┬────┘                                              │
│         │                                                   │
│         ▼                                                   │
│    ┌─────────┐                                              │
│    │REFACTOR │  3. Improve CMake structure                  │
│    │(Improve)│     while keeping tests green                │
│    └────┬────┘                                              │
│         │                                                   │
│         └──────────────► Repeat                             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Example TDD Workflow for CMake

**Scenario**: Add a new library target with dependencies

#### Step 1: RED - Write a failing test first

```cmake
# tests/core/CMakeLists.txt - Write test BEFORE implementation

cmake_minimum_required(VERSION 3.15)

# Test that core library exists and links correctly
add_executable(test_core_exists test_core_exists.cpp)

# This will FAIL - core library doesn't exist yet
target_link_libraries(test_core_exists PRIVATE core)

# Register test
add_test(NAME core_library_exists COMMAND test_core_exists)
```

```cpp
// tests/core/test_core_exists.cpp
#include "core/types.h"  // Will fail - header doesn't exist yet

int main() {
    // Minimal test to verify library links
    return 0;
}
```

```bash
# Run: cmake .. && cmake --build .
# FAILS - target 'core' does not exist
# CMake Error: Cannot specify link libraries for target "test_core_exists"
# which is not built by this project.
```

#### Step 2: GREEN - Write minimal implementation

```cmake
# src/core/CMakeLists.txt - Minimal implementation to pass

cmake_minimum_required(VERSION 3.15)

# Minimal library to satisfy test
add_library(core
    src/types.cpp
)

target_include_directories(core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
```

```bash
# Run: cmake .. && cmake --build . && ctest
# PASSES - tests pass, library links correctly
# Test project /build
#     Start 1: core_library_exists
# 1/1 Test #1: core_library_exists ..........   Passed    0.01 sec
```

#### Step 3: REFACTOR - Improve CMake structure

```cmake
# src/core/CMakeLists.txt - Refactored with proper structure

# CMakeLists.txt - Core module
# Purpose: Builds the core library with fundamental types and utilities

cmake_minimum_required(VERSION 3.15)

# Include CMake modules from separate directory
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
include(CompilerWarnings)

# Library target with all sources
add_library(core
    src/types.cpp
    src/utils.cpp
)

# Public include directory
target_include_directories(core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

# Apply compiler warnings
set_target_warnings(core)

# C++ standard
target_compile_features(core PUBLIC cxx_std_17)
```

```bash
# Run: cmake .. && cmake --build . && ctest
# PASSES - tests still pass, structure improved
```

### Visual Step-by-Step TDD Example

```
┌──────────────────────────────────────────────────────────────────────┐
│ TDD Example: Adding FetchContent Dependency                          │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│ STEP 1: RED - Write test expecting fmt library                       │
│ ─────────────────────────────────────────────                        │
│ tests/CMakeLists.txt:                                                │
│   add_executable(test_fmt test_fmt.cpp)                              │
│   target_link_libraries(test_fmt fmt::fmt)  # ❌ FAILS               │
│                                                                      │
│ $ cmake .. && cmake --build .                                        │
│ CMake Error: Target 'fmt::fmt' not found                             │
│                                                                      │
├──────────────────────────────────────────────────────────────────────┤
│ STEP 2: GREEN - Add minimal FetchContent                             │
│ ─────────────────────────────────────────────                        │
│ cmake/DependencyManagement.cmake:                                    │
│   include(FetchContent)                                              │
│   FetchContent_Declare(fmt                                           │
│       GIT_REPOSITORY https://github.com/fmtlib/fmt.git               │
│       GIT_TAG 10.2.0)                                                │
│   FetchContent_MakeAvailable(fmt)                                    │
│                                                                      │
│ $ cmake .. && cmake --build . && ctest                               │
│ ✓ All tests pass                                                     │
│                                                                      │
├──────────────────────────────────────────────────────────────────────┤
│ STEP 3: REFACTOR - Improve with find_package fallback                │
│ ─────────────────────────────────────────────────────                │
│ cmake/DependencyManagement.cmake:                                    │
│   # Try system first, fallback to FetchContent                       │
│   find_package(fmt QUIET)                                            │
│   if(NOT fmt_FOUND)                                                  │
│       include(FetchContent)                                          │
│       FetchContent_Declare(fmt ...)                                  │
│       FetchContent_MakeAvailable(fmt)                                │
│   endif()                                                            │
│                                                                      │
│ $ cmake .. && cmake --build . && ctest                               │
│ ✓ Tests still pass, better dependency management                     │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### TDD Verification Checklist

Before completing each TDD cycle:

1. **RED Phase**:
   - [ ] Test is written before implementation
   - [ ] Test clearly expresses expected behavior
   - [ ] Test FAILS for the right reason (not syntax errors)
   - [ ] `ctest --output-on-failure` shows expected failure

2. **GREEN Phase**:
   - [ ] Implementation is minimal (just enough to pass)
   - [ ] `cmake --build .` succeeds (exit code 0)
   - [ ] `ctest` passes (exit code 0)
   - [ ] No warnings in build output

3. **REFACTOR Phase**:
   - [ ] Code structure improved
   - [ ] All tests still pass after refactoring
   - [ ] No duplicate CMake code
   - [ ] Follows hexagonal architecture principles

---

## 2B. Bug Fix Protocol (MANDATORY)

**CRITICAL: Every CMake bug MUST receive a regression test BEFORE fixing.**

### Bug Fix Workflow

```
┌─────────────────────────────────────────────────────────────┐
│                  Bug Fix Workflow for CMake                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│    ┌──────────────────┐                                     │
│    │  1. BUG REPORTED │                                     │
│    │  Build fails on  │                                     │
│    │  certain config  │                                     │
│    └────────┬─────────┘                                     │
│             │                                               │
│             ▼                                               │
│    ┌──────────────────┐                                     │
│    │ 2. WRITE TEST    │                                     │
│    │ That REPRODUCES  │  ← Test MUST fail                   │
│    │ the bug          │                                     │
│    └────────┬─────────┘                                     │
│             │                                               │
│             ▼                                               │
│    ┌──────────────────┐                                     │
│    │ 3. VERIFY TEST   │                                     │
│    │ Fails for the    │  ← Confirms bug exists              │
│    │ RIGHT reason     │                                     │
│    └────────┬─────────┘                                     │
│             │                                               │
│             ▼                                               │
│    ┌──────────────────┐                                     │
│    │ 4. FIX THE BUG   │                                     │
│    │ Minimal change   │  ← Test now passes                  │
│    │ to CMake files   │                                     │
│    └────────┬─────────┘                                     │
│             │                                               │
│             ▼                                               │
│    ┌──────────────────┐                                     │
│    │ 5. VERIFY ALL    │                                     │
│    │ Run full test    │  ← No regressions                   │
│    │ suite with ctest │                                     │
│    └────────┬─────────┘                                     │
│             │                                               │
│             ▼                                               │
│    ┌──────────────────┐                                     │
│    │ 6. DOCUMENT      │                                     │
│    │ Add bug ID to    │  ← Prevents future issues           │
│    │ test comments    │                                     │
│    └──────────────────┘                                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Example Bug Fix: Missing Include Directory

**Bug Report #42**: Build fails when using core library headers from another module.

#### Step 1-2: Write test that reproduces the bug

```cmake
# tests/integration/CMakeLists.txt
# Regression test for Bug #42: Missing include directory propagation

cmake_minimum_required(VERSION 3.15)

# Test that core headers are accessible when linking
add_executable(test_bug_42_include_propagation
    test_include_propagation.cpp
)

# Link to core - should automatically get include directories
target_link_libraries(test_bug_42_include_propagation
    PRIVATE
        core
)

# Register regression test
add_test(
    NAME regression_bug_42_include_propagation
    COMMAND test_bug_42_include_propagation
)
```

```cpp
// tests/integration/test_include_propagation.cpp
// Regression test for Bug #42

#include "core/types.h"  // Should work via target_link_libraries

int main() {
    // If this compiles, the bug is fixed
    return 0;
}
```

```bash
# Run: cmake .. && cmake --build .
# FAILS - fatal error: 'core/types.h' file not found
# This confirms Bug #42 exists
```

#### Step 3: Verify test fails for the right reason

```bash
# The error message confirms the bug:
# fatal error: 'core/types.h' file not found
# This is exactly what Bug #42 reported
```

#### Step 4: Fix the bug

```cmake
# src/core/CMakeLists.txt - BEFORE (buggy)
add_library(core src/types.cpp)
# ❌ Missing: include directories not propagated

# src/core/CMakeLists.txt - AFTER (fixed)
add_library(core src/types.cpp)

# ✅ Fix Bug #42: Properly propagate include directories
target_include_directories(core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)
```

```bash
# Run: cmake .. && cmake --build . && ctest
# PASSES - Bug #42 fixed
# Test #1: regression_bug_42_include_propagation ... Passed
```

#### Step 5-6: Verify and document

```cmake
# tests/integration/CMakeLists.txt
# Updated with documentation

# Regression test for Bug #42: Missing include directory propagation
# Bug: When linking to 'core', include directories were not propagated
# Fix: Added PUBLIC target_include_directories to src/core/CMakeLists.txt
# Date: 2024-01-15
# Verified: Test passes after fix
```

### Example Bug Fix: Generator-Specific Build Failure

**Bug Report #87**: Build fails with Ninja but works with Make.

#### Step 1-2: Write test that reproduces the bug

```cmake
# tests/generator/CMakeLists.txt
# Regression test for Bug #87: Ninja generator failure

cmake_minimum_required(VERSION 3.15)

# Test custom command works with all generators
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/generated_config.h
    COMMAND ${CMAKE_COMMAND} -E echo "// Generated" > ${CMAKE_CURRENT_BINARY_DIR}/generated_config.h
    COMMENT "Generating config header"
)

add_custom_target(generate_config
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/generated_config.h
)

# Test executable that depends on generated file
add_executable(test_bug_87_generator test_generator.cpp)
add_dependencies(test_bug_87_generator generate_config)
target_include_directories(test_bug_87_generator
    PRIVATE ${CMAKE_CURRENT_BINARY_DIR}
)

add_test(NAME regression_bug_87_generator COMMAND test_bug_87_generator)
```

```bash
# Run with Ninja: cmake -G Ninja .. && cmake --build .
# FAILS with Ninja - dependency not tracked correctly
# ninja: error: 'generated_config.h', needed by 'test_bug_87_generator', missing
```

#### Step 4: Fix the bug

```cmake
# BEFORE (buggy) - Custom command output not properly linked
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/generated_config.h
    COMMAND ${CMAKE_COMMAND} -E echo "// Generated" > generated_config.h
    # ❌ Missing: WORKING_DIRECTORY and proper dependency chain
)

# AFTER (fixed) - Proper dependency tracking for all generators
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/generated_config.h
    COMMAND ${CMAKE_COMMAND} -E echo "// Generated" > ${CMAKE_CURRENT_BINARY_DIR}/generated_config.h
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    VERBATIM  # ✅ Fix Bug #87: Ensures command works across generators
    COMMENT "Generating config header"
)

# ✅ Properly link generated file to target
target_sources(test_bug_87_generator
    PRIVATE
        ${CMAKE_CURRENT_BINARY_DIR}/generated_config.h
)
```

```bash
# Run with both generators:
# cmake -G Ninja .. && cmake --build . && ctest
# cmake -G "Unix Makefiles" .. && cmake --build . && ctest
# PASSES - Bug #87 fixed for all generators
```

### Bug Fix Verification Checklist

Before completing a bug fix:

1. **Reproduction**:
   - [ ] Bug is clearly documented (ID, description, steps to reproduce)
   - [ ] Regression test written BEFORE attempting fix
   - [ ] Test fails and reproduces the exact bug behavior
   - [ ] Failure message matches reported bug symptoms

2. **Fix**:
   - [ ] Fix is minimal and targeted
   - [ ] Fix addresses root cause, not just symptoms
   - [ ] Fix works with both Ninja and Make generators
   - [ ] Fix doesn't break existing functionality

3. **Verification**:
   - [ ] Regression test now passes
   - [ ] All existing tests still pass (`ctest`)
   - [ ] Build works with all supported generators
   - [ ] No new warnings introduced

4. **Documentation**:
   - [ ] Bug ID referenced in test comments
   - [ ] Fix description added to test file
   - [ ] Date of fix recorded
   - [ ] Related CMake files documented

---

## 3. Hexagonal Architecture for CMake (MANDATORY)

### A. Architecture Principles

**CRITICAL: All CMake projects MUST follow hexagonal architecture principles with clear separation of concerns.**

#### Core Concepts

1. **Root CMakeLists.txt**: Orchestrates modules, minimal logic
2. **Module CMakeLists.txt**: Each component has its own file
3. **CMake Modules**: Reusable functions in separate directory
4. **Dependency Management**: Centralized, clear dependencies

#### ✅ CORRECT - Hexagonal CMake Structure

```
project/
├── CMakeLists.txt              # Root (orchestrates, minimal)
├── cmake/                       # CMake modules (MANDATORY separate directory)
│   ├── CompilerWarnings.cmake
│   ├── DependencyManagement.cmake
│   ├── BuildOptions.cmake
│   ├── Testing.cmake
│   └── Install.cmake
├── src/                         # Source modules
│   ├── core/
│   │   └── CMakeLists.txt      # Core module
│   ├── parser/
│   │   └── CMakeLists.txt      # Parser module
│   └── network/
│       └── CMakeLists.txt      # Network module
├── apps/                        # Applications
│   └── main/
│       └── CMakeLists.txt      # Main app
├── tests/                       # Tests
│   ├── core/
│   │   └── CMakeLists.txt      # Core tests
│   └── parser/
│       └── CMakeLists.txt      # Parser tests
└── docs/                        # Documentation
    └── CMakeLists.txt          # Docs generation
```

#### ❌ WRONG - Monolithic CMake

```
project/
├── CMakeLists.txt              # ❌ Everything in one file (1000+ lines)
└── src/
    └── ...                     # ❌ No module CMakeLists.txt files
```

### B. Root CMakeLists.txt (Minimalistic)

**CRITICAL: Root CMakeLists.txt MUST be minimalistic and only orchestrate modules.**

#### ✅ CORRECT - Clean Root CMakeLists.txt

```cmake
# CMakeLists.txt - Root orchestrator (minimal, clean)
# Project: MyProject
# Description: Modern C++ application with hexagonal architecture

cmake_minimum_required(VERSION 3.15...3.27)

project(MyProject
    VERSION 1.0.0
    DESCRIPTION "Modern C++ Application"
    LANGUAGES CXX
)

# Include CMake modules from separate directory
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

# Include configuration modules
include(cmake/BuildOptions.cmake)
include(cmake/CompilerWarnings.cmake)
include(cmake/DependencyManagement.cmake)

# Add source modules
add_subdirectory(src/core)
add_subdirectory(src/parser)
add_subdirectory(src/network)

# Add applications
add_subdirectory(apps/main)

# Add tests
if(BUILD_TESTING)
    include(cmake/Testing.cmake)
    add_subdirectory(tests/core)
    add_subdirectory(tests/parser)
endif()

# Add documentation
if(BUILD_DOCUMENTATION)
    add_subdirectory(docs)
endif()
```

#### ❌ WRONG - Monolithic Root

```cmake
# ❌ Everything in root - hard to maintain
cmake_minimum_required(VERSION 3.15)
project(MyProject)

# 500+ lines of configuration, dependencies, targets..
# Should be split into modules
```

---

## 4. Modular CMakeLists.txt (MANDATORY)

### A. Single Purpose Per File

**CRITICAL: Each CMakeLists.txt file MUST have a single, clear purpose.**

#### ✅ CORRECT - Single Purpose Module with Per-Module Conan Dependencies

```cmake
# src/core/CMakeLists.txt - Core module (single purpose)
# Purpose: Build the core library with its own Conan dependencies

cmake_minimum_required(VERSION 3.15)

# Set module-local paths for Conan-generated config files
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_BINARY_DIR})
list(APPEND CMAKE_PREFIX_PATH ${CMAKE_CURRENT_BINARY_DIR})

# Declare and install THIS MODULE's Conan dependencies
conan_cmake_configure(
    REQUIRES fmt/10.2.0 spdlog/1.12.0
    GENERATORS CMakeDeps CMakeToolchain
)
conan_cmake_install(
    PATH_OR_REFERENCE .
    BUILD missing
    REMOTE conancenter
    SETTINGS ${CONAN_SETTINGS}
)

# Find packages (Conan generates CMake config files)
find_package(fmt REQUIRED)
find_package(spdlog REQUIRED)

# Library target
add_library(core
    src/types.cpp
    src/utils.cpp
)

target_include_directories(core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(core
    PUBLIC
        fmt::fmt
        spdlog::spdlog
)

# Apply project-wide settings
apply_project_settings(core)
```

### B. Module Organization

**CRITICAL: Every component MUST have its own CMakeLists.txt file.**

#### ✅ CORRECT - Per-Component CMakeLists.txt

```
src/
├── core/
│   ├── CMakeLists.txt          # ✅ Core module
│   ├── include/
│   └── src/
├── parser/
│   ├── CMakeLists.txt          # ✅ Parser module
│   ├── include/
│   └── src/
└── network/
    ├── CMakeLists.txt          # ✅ Network module
    ├── include/
    └── src/
```

#### ❌ WRONG - Missing Module Files

```
src/
├── core/
│   └── src/                    # ❌ No CMakeLists.txt
├── parser/
│   └── src/                    # ❌ No CMakeLists.txt
└── CMakeLists.txt              # ❌ Everything in one file
```

---

## 5. CMake Modules Directory (MANDATORY)

### A. Separate Directory for Modules

**CRITICAL: All CMake modules, functions, and utilities MUST be in a separate `cmake/` directory.**

#### ✅ CORRECT - Modules in Separate Directory

```
project/
├── CMakeLists.txt
├── cmake/                       # ✅ Separate directory
│   ├── BuildOptions.cmake
│   ├── CompilerWarnings.cmake
│   ├── DependencyManagement.cmake
│   ├── Testing.cmake
│   └── Install.cmake
└── src/
```

#### ❌ WRONG - Modules in Root

```
project/
├── CMakeLists.txt
├── BuildOptions.cmake          # ❌ Should be in cmake/
├── CompilerWarnings.cmake      # ❌ Should be in cmake/
└── src/
```

### B. Module Examples

#### BuildOptions.cmake

```cmake
# cmake/BuildOptions.cmake - Build configuration options
# Purpose: Define and configure build options

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Build options
option(BUILD_TESTING "Build test suite" ON)
option(BUILD_DOCUMENTATION "Build documentation" OFF)
option(BUILD_EXAMPLES "Build example programs" OFF)
option(ENABLE_COVERAGE "Enable code coverage" OFF)
option(ENABLE_SANITIZERS "Enable sanitizers" OFF)

# Verbose mode
option(CMAKE_VERBOSE_MAKEFILE "Verbose build output" OFF)

# Debug mode
option(CMAKE_DEBUG_MODE "Debug CMake configuration" OFF)

# Generator selection
if(CMAKE_DEBUG_MODE)
    set(CMAKE_VERBOSE_MAKEFILE ON)
endif()

# Export options
set(BUILD_OPTIONS_LOADED TRUE PARENT_SCOPE)
```

#### CompilerWarnings.cmake

```cmake
# cmake/CompilerWarnings.cmake - Compiler warning configuration
# Purpose: Set compiler warnings for targets

function(set_target_warnings target_name)
    set(MSVC_WARNINGS
        /W4
        /WX
    )
    
    set(GCC_CLANG_WARNINGS
        -Wall
        -Wextra
        -Wpedantic
        -Werror
    )
    
    if(MSVC)
        target_compile_options(${target_name} PRIVATE ${MSVC_WARNINGS})
    else()
        target_compile_options(${target_name} PRIVATE ${GCC_CLANG_WARNINGS})
    endif()
endfunction()
```

#### DependencyManagement.cmake

```cmake
# cmake/DependencyManagement.cmake - Dependency management
# Purpose: Manage dependencies with priority: CMake → Conan → System → FetchContent

# Priority order:
# 1. CMake find_package (embedded tools)
# 2. Conan packages
# 3. System packages
# 4. FetchContent (download)

function(configure_dependencies)
    cmake_parse_arguments(DEPS "" "" "REQUIRES" ${ARGN})
    
    foreach(dep ${DEPS_REQUIRES})
        # Try CMake find_package first
        find_package(${dep} QUIET)
        
        if(NOT ${dep}_FOUND)
            # Try Conan
            find_package(${dep} REQUIRED CONAN)
        endif()
        
        if(NOT ${dep}_FOUND)
            # Fallback to system package
            find_package(${dep} REQUIRED)
        endif()
    endforeach()
endfunction()
```

---

## 6. Package Management Strategy (MANDATORY)

### A. Dependency Priority Order

**CRITICAL: Follow strict priority order for dependency management (as defined in cpp.md and conan.md).**

#### Priority Order

1. **Conan Packages (conan-center)** ⭐ PREFERRED: Use per-module `conan_cmake_configure()` + `conan_cmake_install()` (see conan.md)
2. **System Packages**: Use system package manager as fallback (only if not in Conan)
3. **FetchContent/Download**: Last resort, download from internet

Each module declares and manages its own Conan dependencies independently in its own `CMakeLists.txt`. No centralized `conanfile.txt` or `conanfile.py`. The user runs `cmake -B build && cmake --build build` — nothing else.

### B. Conan Bootstrap (cmake/conan/CMakeLists.txt)

**PRIMARY pattern (from conan.md). Bootstraps Conan once via `add_subdirectory(cmake/conan)` in root CMakeLists.txt.**

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

### B2. Alternative: Function-Based Wrapper (cmake/ConanIntegration.cmake)

**Optional convenience wrapper. Use `include(ConanIntegration)` + `setup_conan()` in root, then `add_conan_dependencies()` in modules.**

```cmake
# cmake/ConanIntegration.cmake - Conan package management (wrapper)

function(setup_conan)
    find_program(CONAN_CMD conan)
    if(NOT CONAN_CMD)
        message(WARNING "Conan not found, skipping Conan integration")
        return()
    endif()

    if(NOT EXISTS "${CMAKE_BINARY_DIR}/conan.cmake")
        file(DOWNLOAD
            "https://raw.githubusercontent.com/conan-io/cmake-conan/0.18.1/conan.cmake"
            "${CMAKE_BINARY_DIR}/conan.cmake"
            TLS_VERIFY ON
        )
    endif()

    include(${CMAKE_BINARY_DIR}/conan.cmake)
    conan_cmake_autodetect(settings)
    set(CONAN_SETTINGS ${settings} PARENT_SCOPE)
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

---

## 7. Builder Support: Ninja and Make (MANDATORY)

### A. Generator Detection

**CRITICAL: CMake MUST support both Ninja (preferred) and Make builders.**

#### ✅ CORRECT - Generator Support

```cmake
# cmake/GeneratorSupport.cmake - Builder configuration

# Detect generator
if(CMAKE_GENERATOR MATCHES "Ninja")
    set(USING_NINJA TRUE)
    message(STATUS "Using Ninja generator")
elseif(CMAKE_GENERATOR MATCHES "Unix Makefiles")
    set(USING_MAKE TRUE)
    message(STATUS "Using Make generator")
else()
    message(STATUS "Using generator: ${CMAKE_GENERATOR}")
endif()

# Set build tool
if(USING_NINJA)
    find_program(NINJA_CMD ninja)
    if(NINJA_CMD)
        set(CMAKE_MAKE_PROGRAM ${NINJA_CMD})
    endif()
endif()
```

### B. Progress Indicators

**CRITICAL: Enable progress indicators when available.**

```cmake
# Enable progress output
if(CMAKE_GENERATOR MATCHES "Ninja")
    # Ninja shows progress by default
    set(CMAKE_BUILD_PARALLEL_LEVEL ${CMAKE_BUILD_PARALLEL_LEVEL})
elseif(CMAKE_GENERATOR MATCHES "Unix Makefiles")
    # Make: use -j for parallel, show progress if available
    if(NOT CMAKE_BUILD_PARALLEL_LEVEL)
        include(ProcessorCount)
        ProcessorCount(N)
        set(CMAKE_BUILD_PARALLEL_LEVEL ${N})
    endif()
endif()
```

---

## 8. Verbose and Debug Modes (MANDATORY)

### A. Verbose Mode

**CRITICAL: CMake MUST support verbose mode for troubleshooting.**

#### ✅ CORRECT - Verbose Mode Support

```cmake
# cmake/BuildOptions.cmake

# Verbose mode option
option(CMAKE_VERBOSE_MAKEFILE "Show verbose build output" OFF)

# Enable verbose if requested
if(CMAKE_VERBOSE_MAKEFILE)
    set(CMAKE_VERBOSE_MAKEFILE ON)
    message(STATUS "Verbose mode enabled")
endif()

# Usage in build
# cmake -DCMAKE_VERBOSE_MAKEFILE=ON .
# OR
# cmake --build . --verbose
```

### B. Debug Mode

**CRITICAL: CMake MUST support debug mode for detailed information.**

#### ✅ CORRECT - Debug Mode Support

```cmake
# cmake/BuildOptions.cmake

# Debug mode option
option(CMAKE_DEBUG_MODE "Show debug information" OFF)

# Debug output function
function(debug_message message)
    if(CMAKE_DEBUG_MODE)
        message(STATUS "[DEBUG] ${message}")
    endif()
endfunction()

# Usage
debug_message("Configuring dependencies...")
debug_message("CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")
debug_message("CMAKE_GENERATOR: ${CMAKE_GENERATOR}")

# Usage in build
# cmake -DCMAKE_DEBUG_MODE=ON .
```

---

## 9. Testing Support (MANDATORY)

### A. CTest Integration

**CRITICAL: CMake MUST support testing via CTest.**

#### ✅ CORRECT - Testing Module

```cmake
# cmake/Testing.cmake - Testing configuration
# Purpose: Configure CTest and testing infrastructure

# Enable testing
enable_testing()

# CTest configuration
set(CTEST_OUTPUT_ON_FAILURE ON)
set(CTEST_PROGRESS_OUTPUT ON)

# Test discovery
function(add_module_tests module_name)
    # Tests are added in module's CMakeLists.txt
    # This function configures test infrastructure
    message(STATUS "Configuring tests for ${module_name}")
endfunction()

# Custom test target
add_custom_target(test
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure --progress
    DEPENDS all
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running all tests..."
)
```

### B. Test Module Example

```cmake
# tests/core/CMakeLists.txt - Test configuration

cmake_minimum_required(VERSION 3.15)

# Test executable
add_executable(core_tests
    test_types.cpp
    test_utils.cpp
)

target_link_libraries(core_tests
    PRIVATE
        core
        GTest::gtest
        GTest::gtest_main
)

# Register with CTest
include(GoogleTest)
gtest_discover_tests(core_tests)
```

---

## 10. Install Support (MANDATORY)

### A. Install Configuration

**CRITICAL: CMake MUST support installation of targets and files.**

#### ✅ CORRECT - Install Module

```cmake
# cmake/Install.cmake - Installation configuration
# Purpose: Configure installation rules

# Install directories
set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install" CACHE PATH "Install prefix")

# Install targets
function(install_library target_name)
    install(TARGETS ${target_name}
        EXPORT ${target_name}Targets
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin
        INCLUDES DESTINATION include
    )
    
    install(DIRECTORY include/
        DESTINATION include
        FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
    )
endfunction()

# Install executables
function(install_executable target_name)
    install(TARGETS ${target_name}
        RUNTIME DESTINATION bin
    )
endfunction()
```

### B. Install Target

```cmake
# Root CMakeLists.txt

# Install target
install(TARGETS core parser network
    EXPORT MyProjectTargets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
)

# Install headers
install(DIRECTORY include/
    DESTINATION include
    FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
)

# Usage
# cmake --install . --prefix install_dir
```

---

## 11. Incremental Builds and Dependency Tracking (MANDATORY)

### A. Proper Dependency Declaration

**CRITICAL: Declare dependencies correctly to enable incremental builds.**

#### ✅ CORRECT - Proper Dependencies

```cmake
# src/parser/CMakeLists.txt

# Library with proper dependencies
add_library(parser
    src/parser.cpp
    src/lexer.cpp
)

# Public dependencies (propagate to consumers)
target_link_libraries(parser
    PUBLIC
        core              # Parser depends on core
        fmt::fmt          # Public dependency
)

# Private dependencies (internal only)
target_link_libraries(parser
    PRIVATE
        spdlog::spdlog    # Only parser uses spdlog
)

# Include directories
target_include_directories(parser
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)
```

### B. Dependency Graph

**CRITICAL: CMake automatically tracks dependencies for incremental builds.**

```cmake
# CMake tracks:
# - Source file changes → recompile only changed files
# - Header changes → recompile dependent sources
# - Dependency changes → relink affected targets

# Example dependency chain:
# core (library)
#   └── parser (depends on core)
#       └── main (depends on parser)
#
# If core changes → parser and main rebuild
# If parser changes → only parser and main rebuild
# If main changes → only main rebuilds
```

---

## 12. Clean Comments (MANDATORY)

### A. Comment Style

**CRITICAL: All CMake files MUST contain clean, helpful comments.**

#### ✅ CORRECT - Clean Comments

```cmake
# CMakeLists.txt - Core module
# Purpose: Build the core library with minimal dependencies

cmake_minimum_required(VERSION 3.15)

# Include CMake modules from separate directory
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

# Configure dependencies for this module
include(DependencyManagement)
configure_dependencies(
    REQUIRES
        fmt/10.2.0
)

# Library target
add_library(core
    src/types.cpp
    src/utils.cpp
)

# Public include directory (consumers can use)
target_include_directories(core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

# Link dependencies
target_link_libraries(core
    PUBLIC
        fmt::fmt
)
```

#### ❌ WRONG - Poor Comments

```cmake
# ❌ No comments - unclear purpose
cmake_minimum_required(VERSION 3.15)
add_library(core src/types.cpp)
target_link_libraries(core fmt::fmt)

# ❌ Obvious comments
add_library(core src/types.cpp)  # Adds library named core

# ❌ Outdated comments
# Old build system - TODO: update
add_library(core src/types.cpp)
```

### B. File Header Comments

**CRITICAL: Each CMakeLists.txt MUST have a header comment explaining its purpose.**

```cmake
# CMakeLists.txt - Core module
# 
# Purpose: Builds the core library providing fundamental types and utilities.
# Dependencies: fmt (for formatting)
# Output: core library (static or shared based on BUILD_SHARED_LIBS)
#
# This module is the foundation for other modules in the project.
```

---

## 13. Everything Driven from CMake (MANDATORY)

### A. No Shell Scripts Required

**CRITICAL: All build operations MUST be driven from CMake. No shell scripts or manual steps required.**

#### ✅ CORRECT - CMake-Only Build

```cmake
# Root CMakeLists.txt - Complete automation

# Configure step (automatic dependency resolution)
# User runs: cmake .
# CMake handles: dependency detection, configuration, setup

# Build step (automatic)
# User runs: cmake --build .
# CMake handles: compilation, linking, dependency order

# Test step (automatic)
# User runs: ctest
# CMake handles: test discovery, execution, reporting

# Install step (automatic)
# User runs: cmake --install .
# CMake handles: file installation, directory creation
```

#### ❌ WRONG - Manual Steps Required

```bash
# ❌ Requires shell script
#!/bin/bash
# build.sh - Should not be needed
./configure.sh
make
make install

# ❌ Manual dependency installation
# User must run: apt install libfoo-dev
# Should be handled by CMake/Conan
```

### B. Custom Targets for Common Tasks

**CRITICAL: Provide custom targets for all common operations.**

```cmake
# Root CMakeLists.txt

# Build all
add_custom_target(all
    DEPENDS core parser network main
    COMMENT "Building all targets"
)

# Clean build
add_custom_target(clean-all
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_BINARY_DIR}
    COMMENT "Cleaning build directory"
)

# Format code (if clang-format available)
find_program(CLANG_FORMAT clang-format)
if(CLANG_FORMAT)
    add_custom_target(format
        COMMAND ${CLANG_FORMAT} -i ${SOURCES}
        COMMENT "Formatting source code"
    )
endif()

# Documentation
if(BUILD_DOCUMENTATION)
    add_custom_target(docs
        COMMAND ${DOXYGEN} Doxyfile
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        COMMENT "Generating documentation"
    )
endif()
```

---

## 14. Portability (MANDATORY)

### A. Cross-Platform Support

**CRITICAL: CMake files MUST be portable across platforms.**

#### ✅ CORRECT - Portable CMake

```cmake
# cmake/PlatformSupport.cmake - Platform configuration

# Detect platform
if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    set(IS_WINDOWS TRUE)
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(IS_LINUX TRUE)
elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    set(IS_MACOS TRUE)
endif()

# Platform-specific settings
if(IS_WINDOWS)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
else()
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
endif()

# ✅ MODERN - Use block() for scope isolation (CMake 3.25+)
block(PROPAGATE result)
    set(local_var "internal")
    set(result "success")
endblock()

# ✅ MODERN - Use cmake_path for path manipulation (CMake 3.20+)
cmake_path(SET my_path "/path/to/file.txt")
cmake_path(GET my_path FILENAME my_filename) # "file.txt"

# Use CMake path functions (portable)
function(get_relative_path base_path file_path)
    file(RELATIVE_PATH relative ${base_path} ${file_path})
    return(PROPAGATE relative)
endfunction()
```

### B. Path Handling

**CRITICAL: Use CMake path functions for portability.**

```cmake
# ✅ CORRECT - Portable paths
set(INCLUDE_DIR "${CMAKE_SOURCE_DIR}/include")
set(LIB_DIR "${CMAKE_BINARY_DIR}/lib")

# ❌ WRONG - Platform-specific paths
set(INCLUDE_DIR "/usr/local/include")  # Unix-specific
set(LIB_DIR "C:\\libs")                # Windows-specific
```

---

## 15. Performance Optimization (MANDATORY)

### A. Parallel Builds

**CRITICAL: Enable parallel builds for performance.**

```cmake
# cmake/Performance.cmake - Build performance

# Detect number of processors
include(ProcessorCount)
ProcessorCount(N)

# Set parallel build level
if(NOT CMAKE_BUILD_PARALLEL_LEVEL)
    set(CMAKE_BUILD_PARALLEL_LEVEL ${N} CACHE STRING "Number of parallel jobs")
endif()

# Enable parallel builds
if(CMAKE_GENERATOR MATCHES "Ninja")
    # Ninja uses parallel by default
    set(CMAKE_BUILD_PARALLEL_LEVEL ${N})
elseif(CMAKE_GENERATOR MATCHES "Unix Makefiles")
    # Make: use -j flag
    set(CMAKE_BUILD_PARALLEL_LEVEL ${N})
endif()

message(STATUS "Build parallel level: ${CMAKE_BUILD_PARALLEL_LEVEL}")
```

### B. Build Cache

**CRITICAL: Enable build caching when available.**

```cmake
# Enable CMake build cache
set(CMAKE_CACHE_DEFAULT_PATH "${CMAKE_BINARY_DIR}/.cmake_cache")

# Use ccache if available
find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    set(CMAKE_C_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
    set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
    message(STATUS "Using ccache for faster builds")
endif()
```

---

## 16. Progress Indicators (MANDATORY)

### A. Build Progress

**CRITICAL: Show progress during builds when available.**

```cmake
# cmake/Progress.cmake - Progress indicators

# Enable progress output
if(CMAKE_GENERATOR MATCHES "Ninja")
    # Ninja shows progress by default
    set(CMAKE_BUILD_PROGRESS ON)
elseif(CMAKE_GENERATOR MATCHES "Unix Makefiles")
    # Make: use progress indicator if available
    find_program(MAKE_PROGRESS make)
    if(MAKE_PROGRESS)
        set(CMAKE_MAKE_PROGRAM "${MAKE_PROGRESS} --progress")
    endif()
endif()

# Custom progress function
function(show_progress message current total)
    if(CMAKE_VERBOSE_MAKEFILE)
        message(STATUS "[${current}/${total}] ${message}")
    else()
        # Show progress bar if possible
        math(EXPR percent "${current} * 100 / ${total}")
        message(STATUS "[${percent}%] ${message}")
    endif()
endfunction()
```

---

## 17. Complete Example: Modular CMake Project

### A. Project Structure

```
project/
├── CMakeLists.txt              # Root (minimal): bootstraps Conan, orchestrates modules
├── cmake/                      # CMake modules
│   ├── conan/
│   │   └── CMakeLists.txt      # Conan bootstrap (downloads conan.cmake)
│   ├── BuildOptions.cmake
│   ├── CompilerWarnings.cmake
│   ├── Testing.cmake
│   ├── Install.cmake
│   └── Performance.cmake
├── src/
│   ├── core/
│   │   ├── CMakeLists.txt      # Core lib + its own Conan deps
│   │   ├── include/core/
│   │   └── src/
│   └── parser/
│       ├── CMakeLists.txt      # Parser lib + its own Conan deps
│       ├── include/parser/
│       └── src/
├── apps/
│   └── main/
│       └── CMakeLists.txt
└── tests/
    ├── core/
    │   └── CMakeLists.txt      # Core tests + GTest via Conan
    └── parser/
        └── CMakeLists.txt      # Parser tests + GTest via Conan
```

### B. Root CMakeLists.txt

```cmake
# CMakeLists.txt - Root orchestrator
# Purpose: Orchestrates all modules with minimal logic

cmake_minimum_required(VERSION 3.15...3.27)

project(MyProject
    VERSION 1.0.0
    DESCRIPTION "Modern C++ Application"
    LANGUAGES CXX
)

# Include CMake modules from separate directory
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

# Bootstrap Conan (downloads conan.cmake, autodetects settings)
# Makes conan_cmake_configure/conan_cmake_install available to all modules
add_subdirectory(cmake/conan)

# Load configuration modules
include(cmake/BuildOptions.cmake)
include(cmake/CompilerWarnings.cmake)
include(cmake/Performance.cmake)

# Add source modules (each manages its own Conan dependencies)
add_subdirectory(src/core)
add_subdirectory(src/parser)

# Add applications
add_subdirectory(apps/main)

# Add tests
if(BUILD_TESTING)
    include(cmake/Testing.cmake)
    add_subdirectory(tests/core)
    add_subdirectory(tests/parser)
endif()

# Install configuration
if(INSTALL_ENABLED)
    include(cmake/Install.cmake)
endif()
```

### C. Module CMakeLists.txt

```cmake
# src/core/CMakeLists.txt - Core module
# Purpose: Builds the core library with its own Conan dependencies

cmake_minimum_required(VERSION 3.15)

# Set module-local paths for Conan-generated config files
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_BINARY_DIR})
list(APPEND CMAKE_PREFIX_PATH ${CMAKE_CURRENT_BINARY_DIR})

# Declare and install THIS MODULE's Conan dependencies
conan_cmake_configure(
    REQUIRES fmt/10.2.0
    GENERATORS CMakeDeps CMakeToolchain
)
conan_cmake_install(
    PATH_OR_REFERENCE .
    BUILD missing
    REMOTE conancenter
    SETTINGS ${CONAN_SETTINGS}
)

# Find packages (Conan generates CMake config files)
find_package(fmt REQUIRED)

# Library target
add_library(core
    src/types.cpp
    src/utils.cpp
)

# Include directories
target_include_directories(core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

# Link dependencies
target_link_libraries(core
    PUBLIC
        fmt::fmt
)

# Apply project settings
include(CompilerWarnings)
set_target_warnings(core)

# C++ standard
target_compile_features(core PUBLIC cxx_std_20)
```

---

## 18. Security & Dependency Management (MANDATORY)

### A. Infrastructure Security Scanning

```cmake
# Static analysis integration in CMake
# cppcheck - general static analysis
find_program(CPPCHECK cppcheck)
if(CPPCHECK)
    set(CMAKE_CXX_CPPCHECK
        "${CPPCHECK}"
        "--enable=warning,performance,portability"
        "--suppress=missingIncludeSystem"
        "--inline-suppr"
        "--inconclusive"
        "--error-exitcode=1"
    )
endif()

# clang-tidy - LLVM static analyzer
find_program(CLANG_TIDY clang-tidy)
if(CLANG_TIDY)
    set(CMAKE_CXX_CLANG_TIDY
        "${CLANG_TIDY}"
        "-checks=-*,bugprone-*,cert-*,clang-analyzer-*,concurrency-*,cppcoreguidelines-*,modernize-*,performance-*,readability-*"
        "--warnings-as-errors=*"
    )
endif()
```

```bash
# Run static analysis tools manually
cppcheck --enable=all --error-exitcode=1 --suppress=missingIncludeSystem src/
clang-tidy src/*.cpp -- -Iinclude/

# flawfinder - security-focused C/C++ scanner
flawfinder --minlevel=2 --error-level=4 src/
flawfinder --columns --context --sarif src/ > flawfinder-report.sarif
```

### B. Vulnerability Scanning

```bash
# If using Conan package manager - audit dependencies
conan audit scan .
conan audit scan . --format json > audit-report.json

# For vcpkg projects - check for known vulnerabilities manually
# vcpkg does not have built-in audit; use OSV-Scanner instead
osv-scanner --lockfile=vcpkg.json

# Scan build artifacts and binaries
trivy fs --scanners vuln .
```

### C. Policy & Compliance

```cmake
# AddressSanitizer - runtime memory error detection
option(ENABLE_ASAN "Enable AddressSanitizer" OFF)
if(ENABLE_ASAN)
    add_compile_options(-fsanitize=address -fno-omit-frame-pointer)
    add_link_options(-fsanitize=address)
endif()

# UndefinedBehaviorSanitizer - runtime UB detection
option(ENABLE_UBSAN "Enable UndefinedBehaviorSanitizer" OFF)
if(ENABLE_UBSAN)
    add_compile_options(-fsanitize=undefined -fno-omit-frame-pointer)
    add_link_options(-fsanitize=undefined)
endif()

# ThreadSanitizer - data race detection
option(ENABLE_TSAN "Enable ThreadSanitizer" OFF)
if(ENABLE_TSAN)
    add_compile_options(-fsanitize=thread)
    add_link_options(-fsanitize=thread)
endif()

# Hardening flags for production builds
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    add_compile_options(
        -D_FORTIFY_SOURCE=2      # Buffer overflow detection
        -fstack-protector-strong  # Stack smashing protection
        -fPIE                     # Position-independent executable
    )
    add_link_options(
        -pie                      # Position-independent executable
        -Wl,-z,relro,-z,now       # Full RELRO
    )
endif()
```

---

## 19. Quick Reference

### Common Commands

```bash
# ═══════════════════════════════════════════════════════════════════
# CMake Configuration
# ═══════════════════════════════════════════════════════════════════

# Basic configuration (creates build files)
cmake -B build                          # Configure in 'build' directory
cmake -B build -S .                     # Explicit source directory
cmake -B build -G Ninja                 # Use Ninja generator (preferred)
cmake -B build -G "Unix Makefiles"      # Use Make generator

# Configuration with options
cmake -B build -DCMAKE_BUILD_TYPE=Release           # Release build
cmake -B build -DCMAKE_BUILD_TYPE=Debug             # Debug build
cmake -B build -DBUILD_TESTING=ON                   # Enable tests
cmake -B build -DCMAKE_INSTALL_PREFIX=/usr/local    # Set install path
cmake -B build -DCMAKE_VERBOSE_MAKEFILE=ON          # Verbose output

# ═══════════════════════════════════════════════════════════════════
# Building
# ═══════════════════════════════════════════════════════════════════

# Build all targets
cmake --build build                     # Build using default config
cmake --build build --config Release    # Build Release configuration
cmake --build build --config Debug      # Build Debug configuration
cmake --build build -j $(nproc)         # Parallel build (all cores)
cmake --build build --verbose           # Verbose build output

# Build specific target
cmake --build build --target mylib      # Build specific target
cmake --build build --target clean      # Clean build artifacts
cmake --build build --target all        # Build all targets

# ═══════════════════════════════════════════════════════════════════
# Testing with CTest
# ═══════════════════════════════════════════════════════════════════

# Run tests
ctest --test-dir build                  # Run all tests
ctest --test-dir build --output-on-failure    # Show output on failure
ctest --test-dir build -V               # Verbose test output
ctest --test-dir build -VV              # Extra verbose output
ctest --test-dir build -j $(nproc)      # Parallel test execution

# Test filtering
ctest --test-dir build -R "unit_"       # Run tests matching pattern
ctest --test-dir build -E "integration" # Exclude tests matching pattern
ctest --test-dir build -L "fast"        # Run tests with label
ctest --test-dir build --rerun-failed   # Rerun only failed tests

# Test reporting
ctest --test-dir build --progress       # Show progress
ctest --test-dir build --output-junit result.xml  # JUnit XML output

# ═══════════════════════════════════════════════════════════════════
# Installation
# ═══════════════════════════════════════════════════════════════════

cmake --install build                   # Install to CMAKE_INSTALL_PREFIX
cmake --install build --prefix /opt/myapp   # Install to custom prefix
cmake --install build --config Release  # Install Release build
cmake --install build --component Runtime   # Install specific component

# ═══════════════════════════════════════════════════════════════════
# Interactive Configuration
# ═══════════════════════════════════════════════════════════════════

ccmake -B build                         # Curses-based configuration UI
cmake-gui -B build                      # Qt-based configuration GUI

# ═══════════════════════════════════════════════════════════════════
# Debugging & Inspection
# ═══════════════════════════════════════════════════════════════════

# Print variables and properties
cmake -B build -LAH                     # List all cached variables
cmake -B build --trace                  # Trace CMake execution
cmake -B build --trace-expand           # Trace with variable expansion
cmake -B build --debug-output           # Debug output

# Graphviz dependency graph
cmake -B build --graphviz=deps.dot      # Generate dependency graph
dot -Tpng deps.dot -o deps.png          # Convert to PNG

# ═══════════════════════════════════════════════════════════════════
# Package Management
# ═══════════════════════════════════════════════════════════════════

# Conan (preferred for C++ dependencies)
conan install . --output-folder=build --build=missing
cmake -B build -DCMAKE_TOOLCHAIN_FILE=build/conan_toolchain.cmake

# vcpkg
cmake -B build -DCMAKE_TOOLCHAIN_FILE=[vcpkg-root]/scripts/buildsystems/vcpkg.cmake
```

### CMake Patterns Cheat Sheet

```cmake
# ═══════════════════════════════════════════════════════════════════
# TARGET PATTERNS
# ═══════════════════════════════════════════════════════════════════

# Library target (most common)
add_library(mylib
    src/file1.cpp
    src/file2.cpp
)

# Library types
add_library(mylib_static STATIC src/lib.cpp)    # Static library (.a)
add_library(mylib_shared SHARED src/lib.cpp)    # Shared library (.so)
add_library(mylib_object OBJECT src/lib.cpp)    # Object library
add_library(mylib_interface INTERFACE)           # Header-only library

# Executable target
add_executable(myapp
    src/main.cpp
)

# Alias target (for namespaced exports)
add_library(MyProject::mylib ALIAS mylib)

# ═══════════════════════════════════════════════════════════════════
# INCLUDE DIRECTORIES
# ═══════════════════════════════════════════════════════════════════

# Modern generator expressions (CORRECT)
target_include_directories(mylib
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# ═══════════════════════════════════════════════════════════════════
# LINKING LIBRARIES
# ═══════════════════════════════════════════════════════════════════

# Link visibility
target_link_libraries(mylib
    PUBLIC      # Propagates to consumers
        fmt::fmt
    PRIVATE     # Only for this target
        spdlog::spdlog
    INTERFACE   # Only for consumers, not this target
        Boost::headers
)

# ═══════════════════════════════════════════════════════════════════
# VARIABLES
# ═══════════════════════════════════════════════════════════════════

# Important built-in variables
CMAKE_SOURCE_DIR          # Top-level source directory
CMAKE_BINARY_DIR          # Top-level build directory
CMAKE_CURRENT_SOURCE_DIR  # Current CMakeLists.txt source directory
CMAKE_CURRENT_BINARY_DIR  # Current CMakeLists.txt build directory
PROJECT_SOURCE_DIR        # Current project source directory
PROJECT_BINARY_DIR        # Current project build directory

# Setting variables
set(MY_VAR "value")                     # Set variable
set(MY_LIST "a" "b" "c")                # Set list
set(MY_CACHE "default" CACHE STRING "Description")  # Cache variable
option(MY_OPTION "Description" ON)      # Boolean option

# ═══════════════════════════════════════════════════════════════════
# FIND_PACKAGE PATTERNS
# ═══════════════════════════════════════════════════════════════════

# Basic find_package
find_package(fmt REQUIRED)              # Required dependency
find_package(Boost QUIET)               # Optional, no error if missing
find_package(OpenSSL 1.1 REQUIRED)      # Version requirement

# Find package with components
find_package(Qt6 REQUIRED COMPONENTS Core Widgets)
target_link_libraries(myapp Qt6::Core Qt6::Widgets)

# Config vs Module mode
find_package(MyLib CONFIG REQUIRED)     # Use MyLibConfig.cmake
find_package(MyLib MODULE REQUIRED)     # Use FindMyLib.cmake

# Fallback pattern
find_package(fmt QUIET)
if(NOT fmt_FOUND)
    include(FetchContent)
    FetchContent_Declare(fmt
        GIT_REPOSITORY https://github.com/fmtlib/fmt.git
        GIT_TAG 10.2.0
    )
    FetchContent_MakeAvailable(fmt)
endif()

# ═══════════════════════════════════════════════════════════════════
# FETCHCONTENT PATTERNS
# ═══════════════════════════════════════════════════════════════════

include(FetchContent)

# From Git repository
FetchContent_Declare(json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.3
)

# From URL archive
FetchContent_Declare(catch2
    URL https://github.com/catchorg/Catch2/archive/v3.4.0.tar.gz
    URL_HASH SHA256=..
)

# Make available (downloads and configures)
FetchContent_MakeAvailable(json catch2)

# ═══════════════════════════════════════════════════════════════════
# COMPILER FEATURES & OPTIONS
# ═══════════════════════════════════════════════════════════════════

# C++ standard (modern approach)
target_compile_features(mylib PUBLIC cxx_std_17)
target_compile_features(mylib PUBLIC cxx_std_20)

# Compiler warnings
target_compile_options(mylib PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic>
    $<$<CXX_COMPILER_ID:MSVC>:/W4>
)

# Preprocessor definitions
target_compile_definitions(mylib
    PUBLIC
        MY_PUBLIC_DEFINE
    PRIVATE
        MY_PRIVATE_DEFINE=1
        $<$<CONFIG:Debug>:DEBUG_MODE>
)

# ═══════════════════════════════════════════════════════════════════
# TESTING PATTERNS
# ═══════════════════════════════════════════════════════════════════

# Enable testing
enable_testing()

# Basic test
add_test(NAME my_test COMMAND my_test_exe)

# Test with arguments
add_test(NAME my_test COMMAND my_test_exe --arg1 --arg2)

# Test with working directory
add_test(NAME my_test COMMAND my_test_exe)
set_tests_properties(my_test PROPERTIES
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests
)

# GoogleTest integration
include(GoogleTest)
gtest_discover_tests(my_test_exe)

# Catch2 integration
include(Catch)
catch_discover_tests(my_test_exe)

# ═══════════════════════════════════════════════════════════════════
# INSTALL PATTERNS
# ═══════════════════════════════════════════════════════════════════

# Install targets
install(TARGETS mylib myapp
    EXPORT MyProjectTargets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
)

# Install headers
install(DIRECTORY include/
    DESTINATION include
    FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
)

# Install CMake config
install(EXPORT MyProjectTargets
    FILE MyProjectTargets.cmake
    NAMESPACE MyProject::
    DESTINATION lib/cmake/MyProject
)
```

### Project Structure

```
# ═══════════════════════════════════════════════════════════════════
# RECOMMENDED CMAKE PROJECT STRUCTURE
# ═══════════════════════════════════════════════════════════════════

project/
├── CMakeLists.txt              # Root orchestrator (minimal)
│
├── cmake/                      # CMake modules (MANDATORY separate dir)
│   ├── BuildOptions.cmake      # Build configuration options
│   ├── CompilerWarnings.cmake  # Compiler warning configuration
│   ├── DependencyManagement.cmake  # Dependency resolution
│   ├── ConanIntegration.cmake  # Conan package manager
│   ├── Testing.cmake           # CTest configuration
│   ├── Install.cmake           # Installation rules
│   └── Performance.cmake       # Parallel builds, ccache
│
├── src/                        # Source modules
│   ├── core/                   # Core module
│   │   ├── CMakeLists.txt     # Module build file
│   │   ├── include/           # Public headers
│   │   │   └── core/
│   │   │       ├── types.h
│   │   │       └── utils.h
│   │   └── src/               # Implementation
│   │       ├── types.cpp
│   │       └── utils.cpp
│   │
│   ├── parser/                 # Parser module
│   │   ├── CMakeLists.txt
│   │   ├── include/parser/
│   │   └── src/
│   │
│   └── network/                # Network module
│       ├── CMakeLists.txt
│       ├── include/network/
│       └── src/
│
├── apps/                       # Applications
│   └── main/
│       ├── CMakeLists.txt
│       └── main.cpp
│
├── tests/                      # Tests (mirror src structure)
│   ├── CMakeLists.txt         # Test configuration
│   ├── core/
│   │   ├── CMakeLists.txt
│   │   └── test_types.cpp
│   ├── parser/
│   │   └── CMakeLists.txt
│   └── integration/            # Integration tests
│       └── CMakeLists.txt
│
├── docs/                       # Documentation
│   ├── CMakeLists.txt         # Doxygen config
│   └── Doxyfile.in
│
├── examples/                   # Example programs
│   └── CMakeLists.txt
│
├── conanfile.txt              # Conan dependencies (if using Conan)
├── CMakePresets.json          # CMake presets (optional)
└── README.md

# ═══════════════════════════════════════════════════════════════════
# KEY FILES EXPLAINED
# ═══════════════════════════════════════════════════════════════════

# Root CMakeLists.txt (minimal orchestrator)
cmake_minimum_required(VERSION 3.15...3.27)
project(MyProject VERSION 1.0.0 LANGUAGES CXX)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
include(cmake/BuildOptions.cmake)
include(cmake/CompilerWarnings.cmake)
include(cmake/DependencyManagement.cmake)
add_subdirectory(src/core)
add_subdirectory(src/parser)
add_subdirectory(apps/main)
if(BUILD_TESTING)
    enable_testing()
    add_subdirectory(tests)
endif()

# Module CMakeLists.txt (single purpose)
# src/core/CMakeLists.txt
add_library(core src/types.cpp src/utils.cpp)
target_include_directories(core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
target_compile_features(core PUBLIC cxx_std_17)

# Test CMakeLists.txt
# tests/core/CMakeLists.txt
add_executable(test_core test_types.cpp)
target_link_libraries(test_core PRIVATE core GTest::gtest_main)
include(GoogleTest)
gtest_discover_tests(test_core)
```

---

## 20. Summary

**CRITICAL Requirements for All CMake Files:**

1. **Modular Structure**: Each component has its own CMakeLists.txt
2. **CMake Modules**: All modules in separate `cmake/` directory
3. **Minimalistic Root**: Root CMakeLists.txt only orchestrates
4. **Package Management**: Conan (per-module) → System → FetchContent priority (see conan.md)
5. **Builder Support**: Ninja (preferred) and Make generators
6. **Verbose/Debug Modes**: Support for troubleshooting
7. **Testing**: CTest integration, test targets
8. **Install**: Complete install support
9. **Incremental Builds**: Proper dependency tracking
10. **Progress Indicators**: Show build progress when available
11. **CMake-Only**: Everything driven from CMake, no shell scripts
12. **Portability**: Cross-platform support
13. **Performance**: Parallel builds, caching
14. **Clean Comments**: Helpful, clear comments throughout
15. **Verification**: Agent MUST test CMake before delivery

**Agent Verification Protocol:**
- Configure CMake (`cmake ..`) - MUST succeed
- Build (`cmake --build .`) - MUST succeed
- Test (`ctest`) - MUST pass
- Install (`cmake --install .`) - MUST succeed
- **MANDATORY**: After ANY modification, re-verify all steps
- Only present working CMake files to the user

**Remember**: Minimalistic, modular, clean, portable CMake files with hexagonal architecture, everything driven from CMake, support for Ninja and Make, verbose/debug modes, and proper dependency management. Keep it simple, keep it clean, keep it working.

---

## 21. Deployment Checklist

### Build Configuration
- [ ] Minimum CMake version specified (`cmake_minimum_required`)
- [ ] Project version defined in `project()` command
- [ ] Release and Debug build types configured correctly
- [ ] Compiler warnings enabled (`-Wall -Wextra -Wpedantic` or MSVC equivalents)
- [ ] Position-independent code enabled for shared libraries

### Dependencies
- [ ] Conan (per-module) used as primary package manager
- [ ] System packages used as fallback with `find_package()`
- [ ] FetchContent used only as last resort
- [ ] All dependency versions pinned and reproducible

### Testing
- [ ] CTest integration enabled with `enable_testing()`
- [ ] Unit tests discoverable via `gtest_discover_tests()` or equivalent
- [ ] Test targets build without errors
- [ ] All tests pass (`ctest --output-on-failure`)
- [ ] Code coverage target configured for CI

### Installation
- [ ] `install()` targets defined for all public components
- [ ] CMake package config files generated for downstream consumers
- [ ] `CMAKE_INSTALL_PREFIX` documented and tested
- [ ] Component-based install separates Runtime/Development/Documentation

### Cross-Platform
- [ ] Builds succeed with both Ninja and Make generators
- [ ] Windows (MSVC), Linux (GCC/Clang), macOS (AppleClang) tested
- [ ] No platform-specific assumptions without generator expressions
- [ ] CI pipeline validates all target platforms

---

## 22. Why This Configuration Works

1. **Modern CMake Targets**: Using `target_*` commands instead of global variables (`include_directories`, `link_libraries`) scopes settings precisely and prevents unintended leakage between targets.

2. **Ninja as Preferred Generator**: Ninja builds are 10-20% faster than Make due to parallel dependency resolution and minimal stat calls on incremental rebuilds.

3. **Conan-First Dependency Management**: Conan provides hermetic, versioned C/C++ packages with prebuilt binaries, eliminating the "works on my machine" problem for native dependencies.

4. **One CMakeLists.txt per Component**: Each library and executable has its own `CMakeLists.txt`, making the build graph modular and enabling independent development.

5. **CTest Integration**: Native CTest support enables uniform test execution across all platforms, with JUnit XML output for CI pipeline integration.

6. **Generator Expressions**: Using `$<CONFIG:Release>` and `$<PLATFORM_ID:Linux>` instead of `if()` blocks keeps configuration declarative and correct for multi-config generators.

7. **Install and Export Support**: Generating `FooConfig.cmake` files allows downstream projects to consume libraries with a simple `find_package(Foo)`.

8. **Verbose and Debug Modes**: `CMAKE_VERBOSE_MAKEFILE` and `--trace-expand` provide full build transparency for diagnosing configuration issues.

9. **Incremental Prefix Numbering**: `001-network.cmake`, `002-compute.cmake` ordering prevents ambiguity in file processing order and simplifies code review.

10. **Hexagonal Architecture Alignment**: Separating core logic, adapters, and ports into distinct CMake targets enforces architectural boundaries at the build system level.

**End of Modern CMake Development Guidelines**
