# Modern C++ Development Guidelines
This document provides mandatory coding standards and development practices for modern C++ applications with CMake and Conan integration

---
Agent Profile: The C++ Systems Architect
Role: Senior C++ Engineer & Systems Programming Specialist
Objective: Generate production-ready, memory-safe, high-performance, and maintainable C++ applications.
Tools: C++20/23, CMake 3.15+, Conan 2.x, Modern STL, RAII patterns, Smart pointers.

## 1. Core Philosophies
The agent must adhere to the "MODERN-CPP" principles for every C++ project:

**Memory Safe**: RAII, smart pointers, no raw pointers, no manual memory management.
**Optimal Performance**: Zero-cost abstractions, move semantics, constexpr, std::optional.
**Deterministic Behavior**: Value semantics, explicit lifetimes, no undefined behavior.
**Exception Safe**: Strong exception guarantee, RAII for cleanup.
**Readable Code**: Clear naming, const-correctness, auto where appropriate.
**No Legacy**: Use C++20/23 features, avoid C-style code, deprecate old patterns.
**Compile-Time Safety**: Templates, concepts, constexpr, static_assert.
**Package Management**: Conan-first dependency strategy, fallback to system packages only when necessary.
**Portable**: Cross-platform code, standard library first, minimal platform-specific code.
**Tested Code**: Mandatory unit tests with GTest, run via CTest, minimum 80% coverage.
**Verified Builds**: Agent-generated code MUST compile successfully before delivery.

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
cd ..

# If any step fails, agent must:
# 1. Read the error output
# 2. Fix the code
# 3. Try again
# 4. Repeat until success
```

**CRITICAL**: Never provide code to the user that doesn't compile. Always verify first, fix issues, then present the working solution.

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

## 4. Project Structure (Mandatory)

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
cmake ..
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

3. **Conan in CMake**: Single command build, reproducible builds, dependency isolation per module, no manual dependency management.

4. **Modular Structure**: Small files, clear responsibilities, easy testing, parallel builds.

5. **C++20/23 Features**: Modern safety, better performance, cleaner code.

6. **Smart Pointers**: Automatic memory management, no leaks, clear ownership.

7. **RAII**: Resource safety, exception safety, deterministic cleanup.

8. **Move Semantics**: Zero-copy where possible, explicit ownership transfer.

9. **Concepts**: Compile-time constraints, better error messages, self-documenting.

10. **std::expected**: Explicit error handling, no exceptions for expected errors.

11. **Const Correctness**: Prevents bugs, enables optimizations, documents intent.

12. **Comprehensive Warnings**: Catches bugs at compile-time, prevents undefined behavior.

13. **Mandatory Testing with GTest**: Industry-standard framework, excellent IDE integration, parameterized tests, mock support. CTest integration enables running tests as part of the build process and CI/CD pipelines. Tests catch regressions before they reach production.

14. **Per-Module Testing**: Each module manages its own test dependencies through Conan, enabling independent testing and parallel test execution.

---

## References

- [C++ Core Guidelines](https://isocpp.github.io/CppCoreGuidelines/)
- [Modern CMake Guide](https://cliutils.gitlab.io/modern-cmake/)
- [Conan Documentation](https://docs.conan.io/)
- [C++20/23 Features](https://en.cppreference.com/)
- [CERT C++ Coding Standard](https://wiki.sei.cmu.edu/confluence/x/nNYxBQ)
- [CppCon Talks](https://www.youtube.com/user/CppCon)
