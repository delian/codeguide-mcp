# CMake Build System Guidelines
Mandatory standards for modern, target-based CMake build systems: portable, reproducible, presets-driven, packageable. CMake 3.28+, CMakePresets, Ninja, ctest.

---
name: cmake
title: CMake Build System Guidelines
version: 2.0
last_reviewed: 2026-06-05
kind: tooling
tools: [cmake@3.28, ninja, ctest, cpack]
requires: []
recommends:
  - cpp
  - c
  - conan
  - make
  - ci-cd
provides:
  - modern-cmake
  - target-based-builds
  - cmake-presets
  - cmake-packaging
---

> 🧭 Authored per [`CONVENTIONS.md`](guides://CONVENTIONS.md): shared concerns are referenced, not restated. This guide covers only what is unique to CMake.

---

## 0. Prerequisites & References

CMake builds C/C++ (and others). The *language* rules — compiler flags policy, warnings-as-errors, sanitizers, static analysis — are owned by the language guides; this guide owns only how the **build system** is wired.

> 📎 **RECOMMENDED — fetch when the task touches them:**
> - [`cpp.md`](guides://cpp.md) / [`c.md`](guides://c.md) — the languages CMake compiles; warning sets, sanitizers, standards. *(CMake binding: `target_compile_features`, `target_compile_options`.)*
> - [`conan.md`](guides://conan.md) — **CMake-orchestrated** dependency management: CMake is the only tool, Conan is bootstrapped and run from within CMake, and **each module declares its own deps** in its own `CMakeLists.txt` (`conan_cmake_configure`/`conan_cmake_install`) — no external conanfile. *(CMake binding: `add_subdirectory(cmake/conan)` bootstrap, per-module `find_package`.)*
> - [`make.md`](guides://make.md) — simpler hand-written alternative, and the `Unix Makefiles` generator.
> - [`ci-cd.md`](guides://ci-cd.md) — configure/build/test/package in CI via presets.

> 📎 **SEE ALSO:** [`tdd.md`](guides://tdd.md) *(ctest binding for test-first)* · [`secure-coding.md`](guides://secure-coding.md) *(hardening flags, dependency provenance)* · [`semver.md`](guides://semver.md) *(the `VERSION` in `project()` and exported configs)*

---

## 1. Core Philosophies: TARGETS-FIRST

CMake-specific principles only. Language and dependency policy come from §0.

- **T**argets, not variables: model the build as a graph of targets carrying their own `INTERFACE`/`PUBLIC`/`PRIVATE` usage requirements. Never configure with global `include_directories`, `link_libraries`, `add_definitions`, or `CMAKE_CXX_FLAGS`.
- **A**bstract usage requirements: a consumer that links a target inherits its public include dirs, compile features, and definitions automatically. Get the visibility right and consumers need zero extra wiring.
- **R**eproducible & out-of-source: every build lives in a throwaway binary dir (`build/`); the source tree is never polluted. Versions are pinned; configuration is declarative via `CMakePresets.json`.
- **G**enerator-agnostic: author so the same `CMakeLists.txt` works under Ninja, Make, MSBuild, and Xcode. Use generator expressions instead of `if(CMAKE_BUILD_TYPE ...)` for per-config logic.
- **E**xportable: libraries ship an install/export interface (`FooConfig.cmake`) so downstream projects consume them with `find_package(Foo)` — the same way in the build tree and after install.

**Verified Builds**: Agent-generated CMake MUST pass every gate in §2 before delivery.

---

## 2. Requirements (MANDATORY, auditable)

RFC-2119 keywords. IDs `CMAKE-<TOPIC>-<NN>`. Each row has a binary gate; rows binding a shared rule cite its owner.

| ID | Requirement | Verify | Gate |
|----|-------------|--------|------|
| CMAKE-STRUCT-01 | Build MUST be out-of-source (no artifacts in the source tree) | `git status --porcelain` after configure | no build files tracked |
| CMAKE-STRUCT-02 | `cmake_minimum_required` MUST be ≥ 3.28 and `project()` MUST set `VERSION` + `LANGUAGES` (see `semver.md`) | configure | no deprecation warning |
| CMAKE-TGT-01 | Usage requirements MUST be set per-target with explicit `PUBLIC`/`PRIVATE`/`INTERFACE`; no global `include_directories`/`link_libraries`/`add_definitions` | `grep -rnE '^\s*(include_directories|link_libraries|add_definitions)\(' .` | no matches |
| CMAKE-TGT-02 | Include dirs MUST use `$<BUILD_INTERFACE>`/`$<INSTALL_INTERFACE>` generator expressions | review / grep | no raw absolute paths |
| CMAKE-PRESET-01 | A `CMakePresets.json` MUST define configure/build/test presets | `cmake --list-presets` | ≥1 preset listed |
| CMAKE-BUILD-01 | Project MUST configure & build cleanly under Ninja | `cmake --preset ci && cmake --build --preset ci` | exit 0, 0 warnings |
| CMAKE-TST-01 | Tests MUST be registered with ctest and pass (see `tdd.md`) | `ctest --preset ci --output-on-failure` | exit 0, 0 skips |
| CMAKE-DEP-01 | External deps MUST be pinned (`find_package` version / `FetchContent` `GIT_TAG`/`URL_HASH`, see `conan.md`) | review / grep | no floating refs |
| CMAKE-PKG-01 | Installable libraries MUST export a versioned package config (`find_package(Foo)` works post-install) | `cmake --install build --prefix /tmp/p && find_package` smoke | config found |
| CMAKE-WARN-01 | Compiler warnings MUST be enabled on first-party targets (policy in `cpp.md`/`c.md`) | review / grep `target_compile_options` | `-Wall -Wextra` (or `/W4`) |

> **Forbidden**: in-source builds; global directory-scope commands (`include_directories`, `link_libraries`, `add_definitions`, mutating `CMAKE_CXX_FLAGS`); unpinned `FetchContent`/`find_package`; hardcoded absolute paths; raising required version above what features actually need.

---

## 3. Verification Protocol

Run before presenting any CMake change. Fix → re-run until green.

```bash
cmake --preset ci                                 # CMAKE-STRUCT/PRESET/DEP
cmake --build --preset ci                         # CMAKE-BUILD-01 (Ninja)
ctest --preset ci --output-on-failure             # CMAKE-TST-01
cmake --install build/ci --prefix /tmp/pkgsmoke   # CMAKE-PKG-01
grep -rnE '^\s*(include_directories|link_libraries|add_definitions)\(' . # CMAKE-TGT-01
```

The *why* behind each gate lives in its §0 owner; do not re-derive it here.

---

## 4. Project Structure

Out-of-source layout. The root orchestrates; each library/app owns its own `CMakeLists.txt` and declares its own usage requirements.

```
project/
├── CMakeLists.txt          # root: project(), options, add_subdirectory only
├── CMakePresets.json       # configure/build/test/package presets (committed)
├── cmake/                  # reusable modules + FooConfig.cmake.in template
├── src/<lib>/CMakeLists.txt  # one target per component, public headers under include/<lib>/
├── apps/<app>/CMakeLists.txt # executables
├── tests/                  # ctest-registered tests (see tdd.md)
├── build/                  # generated, git-ignored — NEVER committed
└── README.md
```

- Model by target, not by directory: `add_subdirectory` only collects targets — it MUST NOT push global state up via `include_directories`/`add_definitions`.
- Public headers live under `include/<target>/` and are exposed via `$<BUILD_INTERFACE>`/`$<INSTALL_INTERFACE>`, never via a raw source path.
- `build/` is disposable; everything needed to recreate it is in version control.

---

## 5. CMake Specifics

The unique value of this guide.

### A. Targets & usage requirements (PUBLIC / PRIVATE / INTERFACE)

The whole model. A property is **PRIVATE** (used to build this target only), **INTERFACE** (imposed on consumers only — header-only libs), or **PUBLIC** (both). Get visibility right and the graph wires itself.

```cmake
add_library(core
    src/types.cpp
    src/parser.cpp            # one component = one target
)
add_library(MyProj::core ALIAS core)   # namespaced alias = same name in-tree and installed

target_include_directories(core
    PUBLIC                              # consumers see these headers...
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>    # ...both before and after install
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src # internal-only
)

target_compile_features(core PUBLIC cxx_std_20)   # propagates the standard
target_link_libraries(core
    PUBLIC  fmt::fmt        # appears in core's public API → propagates
    PRIVATE spdlog::spdlog  # implementation detail → hidden from consumers
)
```

A consumer just does `target_link_libraries(app PRIVATE MyProj::core)` and inherits the includes, the C++20 requirement, and `fmt`. No global state, no leakage.

### B. Avoid global/directory commands

| ❌ Global (banned) | ✅ Target-scoped |
|---|---|
| `include_directories(inc)` | `target_include_directories(t PUBLIC ...)` |
| `link_libraries(fmt)` | `target_link_libraries(t PRIVATE fmt::fmt)` |
| `add_definitions(-DX)` | `target_compile_definitions(t PRIVATE X)` |
| `set(CMAKE_CXX_FLAGS "-Wall")` | `target_compile_options(t PRIVATE -Wall)` |

Global commands leak settings into every target in the directory and below — the single biggest source of fragile, non-composable builds. Compiler warning *sets* themselves are owned by [`cpp.md`](guides://cpp.md)/[`c.md`](guides://c.md); apply them per target.

### C. Generator expressions

Per-config and per-platform logic that resolves at build time — correct even for multi-config generators (Ninja Multi-Config, MSBuild, Xcode) where `CMAKE_BUILD_TYPE` is empty at configure time.

```cmake
target_compile_options(core PRIVATE
    $<$<CONFIG:Debug>:-O0 -g>
    $<$<CONFIG:Release>:-O2>
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic>
    $<$<CXX_COMPILER_ID:MSVC>:/W4>
)
target_compile_definitions(core PRIVATE $<$<CONFIG:Debug>:DEBUG_BUILD>)
```

Prefer this over `if(CMAKE_BUILD_TYPE STREQUAL ...)`, which silently does nothing under multi-config generators.

### D. Dependencies: find_package vs FetchContent

Order of preference: a real package manager ([`conan.md`](guides://conan.md) — provides a toolchain file so `find_package` "just works") → system `find_package` → `FetchContent` as a self-contained fallback. **Always pin.**

```cmake
# Preferred: package manager / system, via find_package (CMAKE-DEP-01: version pinned)
find_package(fmt 10.2 REQUIRED)

# Fallback: FetchContent — pin GIT_TAG to a commit/tag, or URL_HASH for archives
include(FetchContent)
FetchContent_Declare(json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG        v3.11.3        # never a branch
)
FetchContent_MakeAvailable(json)  # exposes nlohmann_json::nlohmann_json target
```

`find_package` consumes prebuilt/installed packages (fast, cache-friendly, hermetic with Conan); `FetchContent` builds the dependency as part of your tree (zero-setup for consumers, but slower and recompiled). Both yield imported targets you link the same way.

**CMake as the single source of truth (Conan orchestrated from CMake).** To make CMake the only tool — a developer runs *only* `cmake -B build && cmake --build build`, with no external conanfile — bootstrap Conan from within CMake (`add_subdirectory(cmake/conan)` downloads `conan.cmake` during configure), then have **each module declare and install its own dependencies** in its own `CMakeLists.txt` via `conan_cmake_configure()`/`conan_cmake_install()` + an ordinary `find_package()`. Different libraries thus carry entirely different, independently-versioned dependency sets. The full CMAKE-ONLY pattern — bootstrap, root orchestrator, and concrete per-module `src/<module>/CMakeLists.txt` examples — is owned by [`conan.md`](guides://conan.md); bind to it rather than restating it here.

### E. CMakePresets.json

The canonical, committed way to capture configure/build/test/package options — replaces ad-hoc shell flags and makes CI and local builds identical (CMAKE-PRESET-01).

```json
{
  "version": 6,
  "configurePresets": [
    {
      "name": "ci",
      "generator": "Ninja",
      "binaryDir": "${sourceDir}/build/ci",
      "toolchainFile": "${sourceDir}/build/conan_toolchain.cmake",
      "cacheVariables": {
        "CMAKE_BUILD_TYPE": "Release",
        "BUILD_TESTING": "ON"
      }
    }
  ],
  "buildPresets": [{ "name": "ci", "configurePreset": "ci" }],
  "testPresets": [
    { "name": "ci", "configurePreset": "ci",
      "output": { "outputOnFailure": true } }
  ]
}
```

Use `CMakeUserPresets.json` (git-ignored) for personal overrides. CI invokes the exact same presets (see [`ci-cd.md`](guides://ci-cd.md)).

### F. Toolchain files

A toolchain file (`-DCMAKE_TOOLCHAIN_FILE=...` or `toolchainFile` in a preset) sets compiler, sysroot, and target system **before** the project is configured — the mechanism for cross-compilation and for package managers. Conan emits `conan_toolchain.cmake`; reference it from the preset rather than hand-rolling flags.

```cmake
# Minimal cross toolchain (e.g. aarch64 Linux)
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)
set(CMAKE_C_COMPILER   aarch64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)
```

### G. ctest

Register tests so any runner/CI invokes them uniformly (test-first policy → [`tdd.md`](guides://tdd.md)).

```cmake
include(CTest)                         # adds BUILD_TESTING option + enable_testing()
add_executable(core_test test_core.cpp)
target_link_libraries(core_test PRIVATE MyProj::core GTest::gtest_main)
include(GoogleTest)
gtest_discover_tests(core_test)        # one ctest case per TEST() — auto-discovered
```

Run with `ctest --preset ci --output-on-failure`. Use labels (`set_tests_properties(... LABELS fast)`) and `-L/-R/--rerun-failed` to slice the suite; emit `--output-junit` for CI.

### H. Install, export & packaging

Make a library consumable downstream via `find_package` — identically from the build tree and after install (CMAKE-PKG-01).

```cmake
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

install(TARGETS core
    EXPORT  MyProjTargets
    LIBRARY  DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE  DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME  DESTINATION ${CMAKE_INSTALL_BINDIR}
)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(EXPORT MyProjTargets
    NAMESPACE MyProj::                 # consumers link MyProj::core
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/MyProj
    FILE MyProjTargets.cmake
)

write_basic_package_version_file(
    MyProjConfigVersion.cmake
    VERSION ${PROJECT_VERSION}         # from project(VERSION ...) — see semver.md
    COMPATIBILITY SameMajorVersion
)
configure_package_config_file(
    cmake/MyProjConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/MyProjConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/MyProj
)
install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/MyProjConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/MyProjConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/MyProj
)
```

`MyProjConfig.cmake.in` calls `include("${CMAKE_CURRENT_LIST_DIR}/MyProjTargets.cmake")` and re-finds public dependencies (`find_dependency(fmt 10.2)`). Add CPack (`include(CPack)`) for distributable archives/installers when you ship binaries.

### I. Footguns

- **Raising `cmake_minimum_required` arbitrarily** → tie the floor to features you actually use; bumping it breaks consumers for no gain.
- **`file(GLOB)` for sources** → CMake won't notice added/removed files without re-running; list sources explicitly.
- **Linking by raw path/name** (`target_link_libraries(t /usr/lib/libfmt.a)`) → link imported targets so usage requirements propagate.
- **`if(CMAKE_BUILD_TYPE ...)` under multi-config generators** → empty at configure time; use generator expressions (§C).
- **In-source build** (`cmake .`) → always `cmake -B build` / a preset; in-source pollution is hard to undo.

---

## 6. Tooling & Dependencies

Dependency-manager policy is owned by [`conan.md`](guides://conan.md); supply-chain/provenance by [`secure-coding.md`](guides://secure-coding.md). CMake binding:

```bash
conan install . --output-folder=build --build=missing   # emits conan_toolchain.cmake
cmake --preset ci          # toolchainFile points at it; find_package resolves deps
cmake --build --preset ci  # Ninja, parallel by default
```

Pin every external reference (CMAKE-DEP-01). Optionally enable `ccache`/`sccache` via `CMAKE_CXX_COMPILER_LAUNCHER` for faster rebuilds.

---

## 7. Quick Reference

```bash
cmake --preset ci                          # configure (out-of-source, Ninja)
cmake --build --preset ci                  # build
cmake --build --preset ci --target core    # build one target
ctest --preset ci --output-on-failure      # test
cmake --install build/ci --prefix /opt/x   # install + export package config
cmake --list-presets                       # discover presets
cpack --config build/ci/CPackConfig.cmake  # package (if CPack configured)
```

---

## 8. Deployment Checklist

Generated from §2 — one box per requirement ID.

- [ ] CMAKE-STRUCT-01 — out-of-source; no build artifacts tracked
- [ ] CMAKE-STRUCT-02 — `cmake_minimum_required ≥ 3.28`, `project()` sets VERSION + LANGUAGES
- [ ] CMAKE-TGT-01 — per-target usage requirements; no global directory commands
- [ ] CMAKE-TGT-02 — includes use `$<BUILD_INTERFACE>`/`$<INSTALL_INTERFACE>`
- [ ] CMAKE-PRESET-01 — `CMakePresets.json` defines configure/build/test presets
- [ ] CMAKE-BUILD-01 — configures & builds clean under Ninja, 0 warnings
- [ ] CMAKE-TST-01 — ctest registered and green, 0 skips
- [ ] CMAKE-DEP-01 — all external deps pinned
- [ ] CMAKE-PKG-01 — installable libs export a versioned `find_package` config
- [ ] CMAKE-WARN-01 — warnings enabled per target (policy in cpp.md/c.md)
- [ ] Agent ran every §3 command and documented any fixes

---
**End of CMake Build System Guidelines**
