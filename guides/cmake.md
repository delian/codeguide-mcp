# Modern CMake Development Guidelines

This document provides mandatory coding standards and development practices for creating modern, maintainable CMake build systems with emphasis on minimalistic, clean, modular, and portable CMake files.

---

**Agent Profile**: The CMake Architect  
**Role**: Senior Build System Engineer & Automation Specialist  
**Objective**: Generate production-ready, minimalistic, clean, modular, and maintainable CMake build systems using hexagonal architecture principles.  
**Tools**: CMake 3.15+, Ninja (preferred) / Make (fallback), Conan 2.x, CMake FetchContent, Doxygen.

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
   cmake ..
   
   # Check for configuration errors
   echo $?  # Must be 0
   
   # Verify with different generators
   cmake -G Ninja ..
   cmake -G "Unix Makefiles" ..
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
   cmake -G "Unix Makefiles" ..
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
   cmake ..
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

# 500+ lines of configuration, dependencies, targets...
# Should be split into modules
```

---

## 4. Modular CMakeLists.txt (MANDATORY)

### A. Single Purpose Per File

**CRITICAL: Each CMakeLists.txt file MUST have a single, clear purpose.**

#### ✅ CORRECT - Single Purpose Module

```cmake
# src/core/CMakeLists.txt - Core module (single purpose)
# Purpose: Build the core library

cmake_minimum_required(VERSION 3.15)

# Include CMake modules
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
include(DependencyManagement)

# Configure dependencies for this module
configure_dependencies(
    REQUIRES
        fmt/10.2.0
        spdlog/1.12.0
)

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

**CRITICAL: Follow strict priority order for dependency management as defined in cpp.md.**

#### Priority Order

1. **CMake Embedded Tools** (PREFERRED): Use CMake's built-in find_package
2. **Conan Packages**: Use Conan when CMake packages unavailable
3. **System Packages**: Use system package manager as fallback
4. **FetchContent/Download**: Last resort, download from internet

#### ✅ CORRECT - Dependency Management Function

```cmake
# cmake/DependencyManagement.cmake - Dependency resolution

function(resolve_dependency package_name)
    # 1. Try CMake find_package (embedded tools)
    find_package(${package_name} QUIET)
    if(${package_name}_FOUND)
        message(STATUS "Found ${package_name} via CMake")
        return()
    endif()
    
    # 2. Try Conan
    if(CONAN_AVAILABLE)
        find_package(${package_name} REQUIRED CONAN)
        if(${package_name}_FOUND)
            message(STATUS "Found ${package_name} via Conan")
            return()
        endif()
    endif()
    
    # 3. Try system package
    find_package(${package_name} REQUIRED)
    if(${package_name}_FOUND)
        message(STATUS "Found ${package_name} via system package")
        return()
    endif()
    
    # 4. Last resort: FetchContent
    message(STATUS "Fetching ${package_name} from internet")
    include(FetchContent)
    FetchContent_Declare(${package_name}
        GIT_REPOSITORY https://github.com/example/${package_name}.git
        GIT_TAG v1.0.0
    )
    FetchContent_MakeAvailable(${package_name})
endfunction()
```

### B. Conan Integration

**CRITICAL: Prefer Conan for C++ dependencies when CMake packages unavailable.**

```cmake
# cmake/ConanIntegration.cmake - Conan package management

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
# cmake -DCMAKE_VERBOSE_MAKEFILE=ON ..
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
# cmake -DCMAKE_DEBUG_MODE=ON ..
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
# User runs: cmake ..
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
├── CMakeLists.txt              # Root (minimal)
├── cmake/                      # CMake modules
│   ├── BuildOptions.cmake
│   ├── CompilerWarnings.cmake
│   ├── DependencyManagement.cmake
│   ├── ConanIntegration.cmake
│   ├── Testing.cmake
│   ├── Install.cmake
│   └── Performance.cmake
├── src/
│   ├── core/
│   │   ├── CMakeLists.txt
│   │   ├── include/core/
│   │   └── src/
│   └── parser/
│       ├── CMakeLists.txt
│       ├── include/parser/
│       └── src/
├── apps/
│   └── main/
│       └── CMakeLists.txt
└── tests/
    ├── core/
    │   └── CMakeLists.txt
    └── parser/
        └── CMakeLists.txt
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

# Load configuration modules
include(cmake/BuildOptions.cmake)
include(cmake/CompilerWarnings.cmake)
include(cmake/DependencyManagement.cmake)
include(cmake/Performance.cmake)

# Add source modules
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
# Purpose: Builds the core library

cmake_minimum_required(VERSION 3.15)

# Include modules
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
include(DependencyManagement)

# Configure dependencies
configure_dependencies(
    REQUIRES
        fmt/10.2.0
)

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

## 18. Summary

**CRITICAL Requirements for All CMake Files:**

1. **Modular Structure**: Each component has its own CMakeLists.txt
2. **CMake Modules**: All modules in separate `cmake/` directory
3. **Minimalistic Root**: Root CMakeLists.txt only orchestrates
4. **Package Management**: CMake → Conan → System → FetchContent priority
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
