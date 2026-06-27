# Conan Dependency Management Guidelines (CMake-Orchestrated)
Mandatory standards for CMake-orchestrated Conan in C/C++: CMake is the only tool — Conan is bootstrapped and run from within CMake, each module declares its own dependencies, no external conanfile. Conan 2.x, cmake-conan, CMakeDeps/CMakeToolchain.

---
name: conan
title: Conan Dependency Management Guidelines (CMake-Orchestrated)
version: 2.0
last_reviewed: 2026-06-05
kind: tooling
tools: [conan@2, cmake@3.15+, cmake-conan@0.18.1, CMakeDeps, CMakeToolchain, conan-audit]
requires: []
recommends:
  - cmake
  - cpp
  - c
  - secure-coding
  - ci-cd
provides:
  - cmake-orchestrated-conan
  - per-module-dependencies
  - conan-bootstrap
  - conan-audit
---

> 🧭 Authored per [`CONVENTIONS.md`](guides://CONVENTIONS.md): shared concerns are referenced, not restated. This guide covers only what is unique to running Conan *from within* CMake with decentralized, per-module dependency management.

---

## 0. Prerequisites & References

Conan here is an implementation detail of the CMake build. Fetch these when the task touches them; this guide does not restate their rules.

> 📎 **RECOMMENDED — fetch when the task touches them:**
> - [`cmake.md`](guides://cmake.md) — the build system that orchestrates Conan. Targets, `target_link_libraries`, install/export, presets, and `find_package` usage are owned there; this guide covers only how each module pulls its Conan deps.
> - [`cpp.md`](guides://cpp.md) · [`c.md`](guides://c.md) — the languages being built (toolchain, `cppstd`, sanitizers, warnings).
> - [`secure-coding.md`](guides://secure-coding.md) — supply-chain / CVE policy *(binding: `conan audit scan`, pinned versions, authenticated remotes)*.
> - [`ci-cd.md`](guides://ci-cd.md) — pipeline & cache policy *(binding: the audit job, build-once/cache binaries)*.

> 📎 **SEE ALSO:** [`tdd.md`](guides://tdd.md) — test-first cycle (a failing test may pull a new Conan dep into a module's `CMakeLists.txt`); [`semver.md`](guides://semver.md) — pinned-version semantics.

---

## 1. Core Philosophies: CMAKE-ONLY

The single rule behind every decision: **CMake is the only tool the developer runs; Conan is bootstrapped, configured, and executed entirely from within CMake.** Cross-cutting concerns (test-first, CVE policy) come from §0.

- **C**Make is the only tool: the developer runs `cmake -B build` and `cmake --build build` — nothing else. Conan is invoked automatically during configure. Zero external commands.
- **M**odule-local dependencies: each library/module declares and installs its **own** Conan deps via `conan_cmake_configure()` + `conan_cmake_install()` in its **own** `CMakeLists.txt`.
- **A**uto-bootstrapped: `conan.cmake` (cmake-conan) is downloaded during configure (pinned URL, `TLS_VERIFY ON`). No pre-installation of cmake-conan.
- **K**nown versions only: pin exact versions (`fmt/10.2.0`, never `fmt/[>=10.0]`).
- **E**very module self-contained: **no** central `conanfile.txt` / `conanfile.py`. Dependencies live where they are used.
- **O**paque to the user: a new developer need not know Conan exists; `cmake --build build` just works.
- **N**o external files: no `conanfile.txt`, no `conanfile.py`, no manual `conan install`. The CMake tree is the single source of truth.
- **L**ocal scope: different modules may use different versions of the same dependency without conflict — each has its own install scope.
- **Y**ield pre-built binaries: `BUILD missing` downloads a pre-built binary when one matches (OS/compiler/arch/build_type); builds from source only when none exists.

**Additional principles:** Conan is a build implementation detail, not a user-facing tool; the `add_subdirectory()` pattern bootstraps Conan once, then each module uses it independently; changing one module's dependencies never affects another.

**Verified Config**: Agent-generated CMake+Conan MUST resolve all dependencies and build from a **clean** `cmake` configure (no manual Conan step) before delivery, and pass every gate in §2.

---

## 2. Requirements (MANDATORY, auditable)

RFC-2119 keywords. IDs `CONAN-<TOPIC>-<NN>`. Each row has a binary gate; rows binding a shared rule cite its owner.

| ID | Requirement | Verify | Gate |
|----|-------------|--------|------|
| CONAN-BUILD-01 | A clean `cmake -B build && cmake --build build` MUST resolve every dependency and build — no manual `conan install`, no consumer `conanfile` | `rm -rf build && cmake -B build && cmake --build build` | exit 0 |
| CONAN-MOD-01 | Every module's deps MUST be declared in **its own** `CMakeLists.txt` via `conan_cmake_configure()` + `conan_cmake_install()`; there MUST be no centralized `conanfile.txt`/`conanfile.py` | grep tree | per-module decls; no root conanfile |
| CONAN-BOOT-01 | `conan.cmake` MUST be auto-downloaded during configure from a **pinned** cmake-conan release with `TLS_VERIFY ON` | review `cmake/conan/CMakeLists.txt` | pinned URL + TLS |
| CONAN-PIN-01 | Every `REQUIRES` MUST pin an exact version — no ranges | grep for `/[` ranges | 0 unpinned refs |
| CONAN-GEN-01 | MUST generate with `CMakeDeps` + `CMakeToolchain`; the Conan 1.x generators (`cmake`, `cmake_find_package`, `cmake_paths`) MUST NOT appear | grep generators | only 2.x generators |
| CONAN-SEC-01 | 0 high/critical CVEs across the dependency graph (see `secure-coding.md`) | `conan audit scan .` | 0 high/critical |
| CONAN-USER-01 | The consumer workflow MUST require only CMake; docs/README MUST NOT instruct the user to run any `conan` command | review README / module docs | "only cmake" documented |

> **Forbidden**: requiring a manual `conan install`; a central `conanfile.txt`/`conanfile.py` for the consumer; centralized dependency lists instead of per-module declarations; unpinned ranges; Conan 1.x generators; hardcoded Conan cache paths; shipping a graph with known high/critical CVEs.

---

## 3. Verification Protocol

Run, in order, before presenting any CMake+Conan config. Fix → re-run until every gate is green.

```bash
rm -rf build                                   # start clean
cmake -B build                                 # CONAN-BOOT-01 + BUILD-01: bootstraps conan.cmake, resolves all module deps
cmake --build build                            # BUILD-01: all targets compile & link
ctest --test-dir build --output-on-failure     # tests link Conan-provided deps (e.g. GTest)
conan audit scan .                             # CONAN-SEC-01: CVE scan of the whole graph
```

The clean rebuild is the contract: if it does not succeed end-to-end with no manual Conan step, the config is defective. The *why* behind TDD and CVE policy lives in their §0 owners.

---

## 4. Project Structure

Every module manages its own Conan dependencies in its own `CMakeLists.txt`. There is **no** project-root conanfile.

```
project/
├── CMakeLists.txt                  # Root: bootstraps Conan, then add_subdirectory() each module
├── cmake/
│   └── conan/
│       └── CMakeLists.txt          # Bootstrap: downloads conan.cmake, autodetects settings ONCE
├── src/
│   ├── core/
│   │   ├── CMakeLists.txt          # Declares: fmt/10.2.0, spdlog/1.12.0
│   │   ├── include/core/types.hpp
│   │   └── src/types.cpp
│   ├── parser/
│   │   ├── CMakeLists.txt          # Declares: readline/8.2  (+ system BISON/FLEX)
│   │   ├── parser.y · lexer.l · ast.c
│   ├── network/
│   │   ├── CMakeLists.txt          # Declares: openssl/3.2.0, libcurl/8.5.0
│   │   └── include/network/ · src/
│   └── app/
│       ├── CMakeLists.txt          # Links core/parser/network — no own Conan deps
│       └── main.cpp
└── tests/
    ├── CMakeLists.txt              # Declares: gtest/1.15.0
    └── test_core.cpp · test_parser.cpp
```

- **No** `conanfile.txt`/`conanfile.py` in the root — dependencies are co-located with the code that uses them.
- Each module's `CMakeLists.txt` is the single source of truth for *that module's* dependencies.
- `cmake/conan/CMakeLists.txt` bootstraps Conan once; modules use it independently.
- Adding/removing a module touches only that module; modules may even use different versions of the same package (CMAKE-ONLY "L"ocal scope).

---

## 5. CMake-Orchestrated Conan Integration

The unique value of this guide.

### A. Architecture overview

```
What the developer sees:          What happens automatically (cmake -B build):
─────────────────────────         ─────────────────────────────────────────────
$ cmake -B build                  Root CMakeLists.txt
$ cmake --build build               └─ add_subdirectory(cmake/conan)   ← bootstrap ONCE
$ ctest --test-dir build               ├─ download conan.cmake (pinned, TLS)
                                       ├─ include(conan.cmake)
That's it. No conan commands.          └─ conan_cmake_autodetect(settings)
No conanfile. No manual install.    └─ add_subdirectory(src/core)      ← module owns its deps
                                       ├─ conan_cmake_configure(REQUIRES fmt/10.2.0 spdlog/1.12.0 …)
                                       ├─ conan_cmake_install(… SETTINGS ${settings})
                                       ├─ find_package(fmt REQUIRED)
                                       └─ target_link_libraries(core PRIVATE fmt::fmt …)
                                    └─ add_subdirectory(src/network)   ← a DIFFERENT dep set
                                       └─ conan_cmake_configure(REQUIRES openssl/3.2.0 libcurl/8.5.0 …)
                                    └─ add_subdirectory(src/app)       ← no Conan deps, just links
                                    └─ add_subdirectory(tests)         ← gtest from Conan
```

### B. Conan bootstrap — `cmake/conan/CMakeLists.txt`

Included once via `add_subdirectory()`; downloads `conan.cmake` and autodetects settings. Each module then calls `conan_cmake_configure()` / `conan_cmake_install()` independently.

```cmake
# cmake/conan/CMakeLists.txt — Conan bootstrap (runs once during configure)
cmake_minimum_required(VERSION 3.15)

# Download conan.cmake (cmake-conan) if not already cached — pinned release, TLS verified (CONAN-BOOT-01)
if(NOT EXISTS "${CMAKE_BINARY_DIR}/conan.cmake")
    message(STATUS "Downloading conan.cmake from cmake-conan")
    file(DOWNLOAD
        "https://raw.githubusercontent.com/conan-io/cmake-conan/0.18.1/conan.cmake"
        "${CMAKE_BINARY_DIR}/conan.cmake"
        TLS_VERIFY ON)
endif()

include(${CMAKE_BINARY_DIR}/conan.cmake)

conan_cmake_autodetect(settings)                  # detect os/compiler/arch/build_type
set(CONAN_SETTINGS ${settings} CACHE INTERNAL "Conan autodetected settings")
```

**Function-based variant (`cmake/ConanIntegration.cmake`)** — a cleaner API; `setup_conan()` in the root, then `add_conan_dependencies(REQUIRES …)` in any module. Pick one style and use it consistently.

```cmake
function(setup_conan)
    if(NOT EXISTS "${CMAKE_BINARY_DIR}/conan.cmake")
        file(DOWNLOAD
            "https://raw.githubusercontent.com/conan-io/cmake-conan/0.18.1/conan.cmake"
            "${CMAKE_BINARY_DIR}/conan.cmake" TLS_VERIFY ON)
    endif()
    include(${CMAKE_BINARY_DIR}/conan.cmake)
    conan_cmake_autodetect(settings)
    set(CONAN_SETTINGS ${settings} PARENT_SCOPE)
endfunction()

function(add_conan_dependencies)
    cmake_parse_arguments(C "" "" "REQUIRES" ${ARGN})
    conan_cmake_configure(REQUIRES ${C_REQUIRES} GENERATORS CMakeDeps CMakeToolchain)
    conan_cmake_install(PATH_OR_REFERENCE . BUILD missing REMOTE conancenter SETTINGS ${CONAN_SETTINGS})
endfunction()
```

### C. Root `CMakeLists.txt`

```cmake
# CMakeLists.txt — Root orchestrator. The user runs ONLY cmake.
cmake_minimum_required(VERSION 3.15)
project(MyProject C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 17)

add_subdirectory(cmake/conan)        # bootstrap Conan (download + autodetect) — must come first
add_subdirectory(src/core)           # each module manages its own Conan dependencies
add_subdirectory(src/parser)
add_subdirectory(src/network)
add_subdirectory(src/app)
enable_testing()
add_subdirectory(tests)
```

### D. Module pattern (MANDATORY) — `src/core/CMakeLists.txt`

Every module that needs external libraries declares them inline, then uses ordinary `find_package`:

```cmake
# src/core/CMakeLists.txt — Core library. Dependencies (via Conan): fmt, spdlog.
cmake_minimum_required(VERSION 3.15)
project(core CXX)

# ── Conan paths for THIS module ──────────────────────────────────────
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_BINARY_DIR})
list(APPEND CMAKE_PREFIX_PATH ${CMAKE_CURRENT_BINARY_DIR})

# ── Declare and install THIS module's dependencies (pinned) ──────────
conan_cmake_configure(REQUIRES
    fmt/10.2.0
    spdlog/1.12.0
    GENERATORS CMakeDeps CMakeToolchain)
conan_cmake_install(PATH_OR_REFERENCE .
    BUILD missing
    REMOTE conancenter
    SETTINGS ${settings})

# ── Standard CMake from here ─────────────────────────────────────────
find_package(fmt REQUIRED)
find_package(spdlog REQUIRED)

add_library(${PROJECT_NAME} src/types.cpp src/utils.cpp)
target_include_directories(${PROJECT_NAME}
    PUBLIC  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:include>)
target_link_libraries(${PROJECT_NAME} PRIVATE fmt::fmt spdlog::spdlog)
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_20)
```

### E. More module examples

**Network library — a different dependency set:**

```cmake
# src/network/CMakeLists.txt — Dependencies (via Conan): openssl, libcurl.
cmake_minimum_required(VERSION 3.15)
project(network CXX)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_BINARY_DIR})
list(APPEND CMAKE_PREFIX_PATH ${CMAKE_CURRENT_BINARY_DIR})

conan_cmake_configure(REQUIRES openssl/3.2.0 libcurl/8.5.0 GENERATORS CMakeDeps CMakeToolchain)
conan_cmake_install(PATH_OR_REFERENCE . BUILD missing REMOTE conancenter SETTINGS ${settings})

find_package(OpenSSL REQUIRED)
find_package(CURL REQUIRED)
add_library(${PROJECT_NAME} src/client.cpp src/server.cpp)
target_include_directories(${PROJECT_NAME}
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>)
target_link_libraries(${PROJECT_NAME} PRIVATE OpenSSL::SSL OpenSSL::Crypto CURL::libcurl)
```

**Parser library — Conan package mixed with system tools (BISON/FLEX):**

```cmake
# src/parser/CMakeLists.txt — Conan: readline. System: BISON, FLEX.
cmake_minimum_required(VERSION 3.15)
project(parser C)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_BINARY_DIR})
list(APPEND CMAKE_PREFIX_PATH ${CMAKE_CURRENT_BINARY_DIR})

conan_cmake_configure(REQUIRES readline/8.2 GENERATORS CMakeDeps CMakeToolchain)
conan_cmake_install(PATH_OR_REFERENCE . BUILD missing REMOTE conancenter SETTINGS ${settings})
find_package(readline REQUIRED)

# System tools — NOT from Conan (must be installed on the build machine)
find_package(BISON REQUIRED)
bison_target(${PROJECT_NAME} parser.y ${CMAKE_CURRENT_BINARY_DIR}/parser.c
    DEFINES_FILE ${CMAKE_CURRENT_BINARY_DIR}/parser.h COMPILE_FLAGS -Wcounterexamples)
find_package(FLEX REQUIRED)
flex_target(lexer lexer.l ${CMAKE_CURRENT_BINARY_DIR}/lexer.c)
add_flex_bison_dependency(lexer ${PROJECT_NAME})

add_library(${PROJECT_NAME} ${FLEX_lexer_OUTPUTS} ${BISON_parser_OUTPUTS} ast.c)
target_link_libraries(${PROJECT_NAME} PRIVATE readline::readline)
```

**Test module — GTest from Conan:**

```cmake
# tests/CMakeLists.txt — Dependencies (via Conan): gtest.
cmake_minimum_required(VERSION 3.15)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_BINARY_DIR})
list(APPEND CMAKE_PREFIX_PATH ${CMAKE_CURRENT_BINARY_DIR})
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)   # Windows: don't override parent CRT

conan_cmake_configure(REQUIRES gtest/1.15.0 GENERATORS CMakeDeps CMakeToolchain)
conan_cmake_install(PATH_OR_REFERENCE . BUILD missing REMOTE conancenter SETTINGS ${settings})
find_package(GTest REQUIRED)

add_executable(core_tests test_core.cpp)
target_link_libraries(core_tests PRIVATE core GTest::gtest_main)
include(GoogleTest)
gtest_discover_tests(core_tests)
```

**Application — no Conan deps, just links the project libraries:**

```cmake
# src/app/CMakeLists.txt
cmake_minimum_required(VERSION 3.15)
add_executable(app main.cpp)
target_link_libraries(app PRIVATE core parser network)
install(TARGETS app DESTINATION bin)
```

---

## 6. Patterns

### A. The three-step module pattern

Every module that needs Conan deps follows the same three steps — self-documenting, no hidden dependencies:

```cmake
# 1) point find_package() at this module's Conan output
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_BINARY_DIR})
list(APPEND CMAKE_PREFIX_PATH ${CMAKE_CURRENT_BINARY_DIR})
# 2) declare + install (pinned, both generators, BUILD missing)
conan_cmake_configure(REQUIRES <pkg>/<version> GENERATORS CMakeDeps CMakeToolchain)
conan_cmake_install(PATH_OR_REFERENCE . BUILD missing REMOTE conancenter SETTINGS ${settings})
# 3) ordinary CMake
find_package(<Pkg> REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE <Pkg>::<target>)
```

The `add_conan_dependencies(REQUIRES …)` wrapper (§5.B) is syntactic sugar over steps 1–2; either is acceptable.

### B. Dependency priority order

```
Need dependency X?
├─ In the C/C++ standard library?  → use it (no dependency)
├─ On conan.io/center?             → USE CONAN (conan_cmake_configure in the module)  ← preferred
├─ System package (apt/dnf/brew)?  → find_package only; document it in README
└─ Neither?                        → FetchContent / vendored source (last resort), document why
```

---

## 7. Generators

Always generate with **both**:

| Generator | Purpose |
|---|---|
| `CMakeDeps` | emits `<pkg>-config.cmake` so `find_package()` resolves the package |
| `CMakeToolchain` | emits the toolchain (compiler/OS/arch/build_type/options) |

**Prohibited (Conan 1.x — do not exist in Conan 2.x):** `cmake`, `cmake_find_package`, `cmake_find_package_multi`, `cmake_paths` (CONAN-GEN-01).

---

## 8. Security & Supply Chain

CVE, integrity, and secrets *policy* is owned by [`secure-coding.md`](guides://secure-coding.md). Conan binding:

```bash
conan audit scan .                       # CONAN-SEC-01: scan the whole graph for CVEs
conan audit scan . --severity-level=7.0  # tighten the threshold (include medium)
conan audit list "openssl/3.2.0"         # known CVEs for one ref
conan audit scan . --format=json > audit-report.json   # for CI
```

- Gate the pipeline on 0 high/critical findings (the CI security job belongs to [`ci-cd.md`](guides://ci-cd.md)).
- Pin exact versions (CONAN-PIN-01) so an audited graph stays audited; consume from authenticated remotes only.

---

## 9. Configuration

- **Autodetect**: `conan_cmake_autodetect(settings)` (in the bootstrap) detects OS, compiler+version, arch, and `build_type` (from `CMAKE_BUILD_TYPE`); `${settings}` is passed to every `conan_cmake_install()`.
- **Override build type**: set `CMAKE_BUILD_TYPE` before autodetect, or pass `SETTINGS build_type=Release` to a specific `conan_cmake_install()`.
- **Custom/private remote**: `conan_cmake_install(… REMOTE mycorp …)` to use Artifactory instead of (or alongside) `conancenter`; pin the remote and keep credentials in env, never in VCS (policy: `secure-coding.md`).
- **Profiles**: developers/CI may run `conan profile detect` once; autodetect picks it up. The developer still runs only `cmake`.

---

## 10. Quick Reference

```bash
# Everything the user ever runs:
cmake -B build                                   # configure (Conan bootstraps + resolves automatically)
cmake --build build                              # build
ctest --test-dir build --output-on-failure       # test
rm -rf build && cmake -B build && cmake --build build   # clean rebuild
conan audit scan .                               # CVE scan
```

```cmake
# Module template — src/<module>/CMakeLists.txt
# Dependencies (via Conan): <pkg>/<version>, …
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_BINARY_DIR})
list(APPEND CMAKE_PREFIX_PATH ${CMAKE_CURRENT_BINARY_DIR})
conan_cmake_configure(REQUIRES <pkg>/<version> GENERATORS CMakeDeps CMakeToolchain)
conan_cmake_install(PATH_OR_REFERENCE . BUILD missing REMOTE conancenter SETTINGS ${settings})
find_package(<Pkg> REQUIRED)
add_library(${PROJECT_NAME} src/impl.cpp)
target_link_libraries(${PROJECT_NAME} PRIVATE <Pkg>::<target>)
```

> **NEVER**: require `conan install`; use a consumer `conanfile.txt`/`conanfile.py`; centralize deps; use 1.x generators; use version ranges. **ALWAYS**: bootstrap from `cmake/conan`; declare deps per module; `CMakeDeps`+`CMakeToolchain`; pin exact versions; `BUILD missing`; pass `${settings}`; `conan audit scan` before delivery.

---

## 11. Deployment Checklist

Generated from §2 — one box per requirement ID.

- [ ] CONAN-BUILD-01 — clean `rm -rf build && cmake -B build && cmake --build build` resolves all deps and builds (no manual Conan step)
- [ ] CONAN-MOD-01 — each module declares its own deps in its own `CMakeLists.txt`; no central conanfile
- [ ] CONAN-BOOT-01 — `conan.cmake` auto-downloaded from a pinned cmake-conan release with `TLS_VERIFY ON`
- [ ] CONAN-PIN-01 — exact pinned versions, no ranges
- [ ] CONAN-GEN-01 — `CMakeDeps` + `CMakeToolchain` only; no 1.x generators
- [ ] CONAN-SEC-01 — `conan audit scan` 0 high/critical CVEs
- [ ] CONAN-USER-01 — consumer workflow is CMake-only; README does not instruct any `conan` command
- [ ] Agent ran every §3 command from a clean tree and documented any fixes

---
**End of Conan Dependency Management Guidelines (CMake-Orchestrated)**
