# C Development Guidelines
Mandatory coding standards for modern C: memory-safe, UB-free, sanitizer-verified, modular. C23/C17, gcc 14+, clang 18+, CMake 3.25+, clang-tidy, cppcheck, valgrind, ASan/UBSan/TSan.

---
name: c
title: C Development Guidelines
version: 2.0
last_reviewed: 2026-06-05
kind: language
tools: [c23, c17, gcc@14, clang@18, cmake@3.25, clang-tidy, cppcheck, valgrind, asan, ubsan, tsan]
requires:
  - tdd
  - secure-coding
  - error-handling
recommends:
  - cmake
  - make
  - conan
  - performance
  - parallelism
  - comments
  - logging
provides:
  - c-memory-management
  - undefined-behavior
  - c-build
  - c-sanitizers
---

> 🧭 Authored per [`CONVENTIONS.md`](guides://CONVENTIONS.md): shared concerns are referenced, not restated. This guide covers only what is unique to C — memory ownership, undefined behavior, the C standard, the toolchain, and C error idioms.

---

## 0. Prerequisites & References

Fetch and apply these **before** generating C code. Their rules are assumed below and not repeated.

> 📎 **REQUIRED — fetch & apply first:**
> - [`tdd.md`](guides://tdd.md) — test-first, Red-Green-Refactor, regression-test-before-fix, coverage. *(C binding: assert-based harness, Unity/Check/CMocka, driven by `ctest`; every test runs under ASan+UBSan.)*
> - [`secure-coding.md`](guides://secure-coding.md) — memory-safety policy, CWE classes, supply chain, secrets, CVE gate. *(C binding: banned unsafe libc calls, integer-overflow checks, dependency CVE scan via Conan; see §6.)*
> - [`error-handling.md`](guides://error-handling.md) — error strategy & propagation. *(C binding: return-code + output-parameter + `errno`; see §5.)*

> 📎 **RECOMMENDED — fetch when the task touches them:**
> - [`cmake.md`](guides://cmake.md) · [`make.md`](guides://make.md) — build orchestration. *(C binding: CMake is the single orchestrator; Make is a thin wrapper; see §7.)*
> - [`conan.md`](guides://conan.md) — dependency resolution. *(C binding: per-module `conan_cmake_configure`/`install`; see §6.)*
> - [`parallelism.md`](guides://parallelism.md) — concurrency model. *(C binding: C11 `<threads.h>` / POSIX `pthreads`; verify with TSan.)*
> - [`comments.md`](guides://comments.md) — API-doc policy *(binding: Doxygen on every public header.)*
> - [`logging.md`](guides://logging.md) · [`performance.md`](guides://performance.md)

> 📎 **SEE ALSO:** [`hexagonal.md`](guides://hexagonal.md) *(ports/adapters mapped to C via vtables — §4.B)* · [`designpatterns.md`](guides://designpatterns.md) · [`code-review.md`](guides://code-review.md) · [`ci-cd.md`](guides://ci-cd.md) · [`env-config.md`](guides://env-config.md)

---

## 1. Core Philosophies: SAFE-C

C-specific principles only. TDD, security policy, and error strategy come from §0 — do **not** restate them here.

- **S**afe memory: explicit ownership for every allocation; every `malloc`/`calloc`/`realloc` checked; every owning resource has a `destroy`/`free`/`fclose` on every path. Verified by ASan + LeakSanitizer + Valgrind, not by inspection.
- **A**bstraction through modules: opaque `struct` types, one public header per `.c`, internal helpers `static`. The struct definition lives only in the `.c`.
- **F**ail-fast: functions that can fail return an error code or use an output parameter; `goto cleanup` unwinds multi-resource functions in reverse acquisition order. (Strategy → [`error-handling.md`](guides://error-handling.md).)
- **E**xplicit over implicit: `const`-correct; no implicit narrowing (`-Wconversion`); no hidden global mutable state; `sizeof(*ptr)`, never `sizeof(type)`.
- **C**ompile-time safety: maximal warnings as errors, `static_assert`, fixed-width `<stdint.h>` types, and C23 `[[nodiscard]]`/`constexpr` to push errors to compile time.

**No Undefined Behavior** is the prime directive: signed overflow, OOB access, use-after-free, data races, strict-aliasing and alignment violations are all UB and MUST be eliminated, not merely hidden. UBSan + ASan + TSan are the proof.

**Verified Code**: Agent-generated C MUST pass every gate in §2 before delivery.

---

## 2. Requirements (MANDATORY, auditable)

RFC-2119 keywords. IDs `C-<TOPIC>-<NN>`. Each row has a binary gate; rows binding a shared rule cite its owner.

| ID | Requirement | Verify | Gate |
|----|-------------|--------|------|
| C-TST-01 | Every feature MUST be test-first (see `tdd.md`) | `ctest --test-dir build` | exit 0, 0 skips |
| C-TST-02 | Each bug MUST get a regression test before the fix (see `tdd.md`) | `ctest --test-dir build` | failing→passing |
| C-TST-03 | Tests MUST run clean under ASan+UBSan | `ctest --test-dir build-san --output-on-failure` | 0 violations |
| C-FMT-01 | Code MUST be formatted | `clang-format --dry-run --Werror src/ tests/` | no diff |
| C-WARN-01 | MUST compile clean at max warnings | `cmake --build build` (`-Wall -Wextra -Werror -Wpedantic -Wshadow -Wconversion`) | exit 0, 0 warnings |
| C-LINT-01 | Static analysis MUST pass | `cppcheck --enable=all --error-exitcode=1 src/` && `clang-tidy` | exit 0 |
| C-MEM-01 | No leaks / use-after-free / double-free / OOB | `valgrind --leak-check=full --error-exitcode=1 ./build/<exe>` or ASan | 0 errors |
| C-UB-01 | No undefined behavior | UBSan-built test run | 0 UBSan reports |
| C-MEM-02 | Every allocation return value MUST be checked | `clang-tidy -checks=clang-analyzer-*` / review | no unchecked alloc |
| C-SEC-01 | No banned unsafe libc calls (see `secure-coding.md`) | `grep -nE '\b(gets|strcpy|strcat|sprintf)\s*\('` | no matches |
| C-SEC-02 | 0 high/critical CVEs in deps (see `secure-coding.md`) | `conan audit .` | 0 high/critical |
| C-DEP-01 | Dependencies pinned & integrity-verified (see `conan.md`) | `conan install . --verify` / `conan.lock` | verified |
| C-DOC-01 | Public headers MUST have Doxygen comments (see `comments.md`) | `doxygen Doxyfile` | 0 undocumented |
| C-ERR-01 | Fallible functions MUST signal failure via return code/out-param (see `error-handling.md`) | review | no silent failure |

> **Forbidden**: shipping implementation before its test (violates `tdd.md`); fixing a bug without a regression test first; `gets`/`strcpy`/`strcat`/`sprintf`/unbounded `scanf("%s")`; casting `malloc`'s return; `sizeof(type)` where `sizeof(*ptr)` is meant; magic numbers; unjustified global mutable state; ignoring a fallible return value.

### Verification Protocol

Run, in order, before presenting code. Fix → re-run until every gate is green. The *why* lives in the §0 owners.

```bash
clang-format --dry-run --Werror src/ tests/            # C-FMT-01
cmake --build build                                    # C-WARN-01
cppcheck --enable=all --error-exitcode=1 src/          # C-LINT-01
clang-tidy src/**/*.c -- -Isrc/                        # C-LINT-01
ctest --test-dir build-san --output-on-failure         # C-TST-01/03, C-MEM-01, C-UB-01 (ASan+UBSan build)
valgrind --leak-check=full --error-exitcode=1 ./build/app   # C-MEM-01
conan audit . && conan install . --verify              # C-SEC-02, C-DEP-01
doxygen Doxyfile                                       # C-DOC-01
```

---

## 3. Project Structure

Idiomatic C layout. Architectural *principles* (ports/adapters, dependency direction, acyclic deps) are owned by [`hexagonal.md`](guides://hexagonal.md); below is only their C mapping.

```
project/
├── src/
│   ├── main.c
│   ├── core/            # domain types + services — no adapter includes
│   ├── ports/           # interface headers (vtables) — see §4.B
│   ├── adapters/        # db/http/file implementations of ports
│   └── utils/           # logging, error helpers
├── include/<project>/   # public headers (library projects)
├── tests/{unit,integration}/   # assert/Unity/Check harnesses (see tdd.md)
├── cmake/               # CompilerWarnings.cmake, Sanitizers.cmake, conan/
├── CMakeLists.txt · CMakePresets.json · Makefile
├── .clang-format · .clang-tidy · Doxyfile
└── README.md
```

- Group by feature/domain, not by type (`user/user.{c,h}`, not `headers/` + `impl/`).
- One public header per `.c`; struct definitions stay in the `.c` (opaque types — §4.A).
- Include guards: `#ifndef PROJECT_MODULE_H` … `#endif` (portable) or `#pragma once`.
- No circular includes; the domain layer includes no adapter/framework headers.

---

## 4. Memory Management & Ownership (C's core discipline)

This is the heart of the guide. C has no GC, no destructors, no borrow checker — ownership is a **convention you enforce mechanically**.

### A. Opaque types + create/destroy (RAII-by-convention)

Every resource-owning type exposes a `*_create` (allocates+initializes, returns `NULL` on failure) and a NULL-safe `*_destroy`. The struct is defined only in the `.c`, so callers cannot reach inside.

```c
// widget.h — public: callers see only the handle
typedef struct widget widget_t;
widget_t *widget_create(const char *label);   // NULL on OOM
void      widget_destroy(widget_t *w);          // NULL-safe

// widget.c — private: definition + lifecycle
struct widget { char *label; };

widget_t *widget_create(const char *label) {
    widget_t *w = calloc(1, sizeof(*w));        // sizeof(*w), not sizeof(widget_t)
    if (!w) return NULL;
    w->label = strdup(label);
    if (!w->label) { free(w); return NULL; }    // partial-init rollback
    return w;
}
void widget_destroy(widget_t *w) {
    if (!w) return;                              // NULL-safe by contract
    free(w->label);
    free(w);
}
```

### B. Ownership rules (mechanical, enforceable)

1. **Document ownership in the signature/Doxygen** — does the function take ownership, borrow, or return a new owned pointer? "Returns a `const char *` owned by the object" vs "caller must `free`".
2. **One owner at a time.** Transfer ownership explicitly; null the source after a move if it could be reused.
3. **`sizeof(*ptr)`, never `sizeof(type)`** — survives type changes, prevents undersized allocs.
4. **Never cast `malloc`'s return** in C — it hides a missing `<stdlib.h>` and masks errors.
5. **`calloc` for zeroed memory**; never assume `malloc` zeroes.
6. **Set freed pointers to `NULL`** when the storage outlives the free, to turn use-after-free into a null deref.
7. **`goto cleanup`** for multi-resource functions; release in reverse acquisition order (see §5).
8. **Check integer overflow before sizing**: `if (n > SIZE_MAX / sizeof(*p)) return ERR_OVERFLOW;` before `malloc(n * sizeof(*p))`. Prefer `calloc(n, sizeof(*p))`, which checks internally.

### C. Pointers, arrays & strings — the footgun surface

- **Arrays decay to pointers**; `sizeof` only gives the element count *in the defining scope*. Pass length explicitly (`f(arr, n)`), never rely on `sizeof(param)`.
- **`strncpy` does NOT NUL-terminate** when the source fills the buffer. Prefer `snprintf(dst, sizeof dst, "%s", src)` or a checked `memcpy` + manual terminator.
- **No pointer arithmetic past one-past-the-end**; comparing/dereferencing such pointers is UB.
- **Flexible array members** (`struct { size_t n; T items[]; }`) for header+payload allocations; size with `offsetof`.
- **`const`-correctness**: borrowed inputs are `const T *`; only the owner mutates.
- **`restrict`** where two pointer params provably never alias — enables optimization and documents intent.

### D. Undefined behavior — the non-negotiables

UB is not "implementation-defined"; the compiler may assume it never happens and miscompile around it. Eliminate, do not paper over:

- **Signed integer overflow** is UB (unsigned wraps). Use `<stdint.h>` widths and checked arithmetic (`__builtin_add_overflow`, or C23 `<stdckdint.h>` `ckd_add`).
- **Out-of-bounds / use-after-free / double-free** → ASan.
- **Reading uninitialized memory**, **null deref**, **misaligned access**, **strict-aliasing violations** (type-pun via `memcpy` or a `union`, never via incompatible pointer casts) → UBSan.
- **Data races** → TSan (see [`parallelism.md`](guides://parallelism.md); C11 `<threads.h>` / pthreads).
- **Shifting by ≥ width**, **`INT_MIN / -1`**, **modifying a string literal**, **`%n` / format mismatches** (`-Wformat=2`).

Sanitizers are the proof of absence at test time; max warnings + `clang-tidy`/`cppcheck` catch many statically (§6).

---

## 5. Error Handling — C bindings

Strategy (when to fail fast, how to propagate, what to log) is owned by [`error-handling.md`](guides://error-handling.md). C has no exceptions; the bindings are:

- **Return code** for fallible functions: `0` (or a positive count) on success, a negative `error_code_t` on failure. Define a project-wide enum (`ERR_OK=0, ERR_NULL_ARG=-1, ERR_OOM=-2, …`) with an `error_to_string()`.
- **Output parameter** when a function returns both a value and a status: `int parse_int(const char *s, int *out)`.
- **`errno`** when wrapping libc: set `errno = 0` before the call, check the documented sentinel, then read `errno`/`strerror`.
- **`assert()`** for *internal invariants / programming errors only* (compiled out with `NDEBUG`) — never for validating external input. Validate external input with real checks that return error codes.
- **`goto cleanup`** for resource unwinding:

```c
int process_file(const char *in, const char *out) {
    if (!in || !out) return ERR_NULL_ARG;
    int rc = ERR_IO_FAILURE;
    FILE *fi = NULL, *fo = NULL; char *buf = NULL;

    fi = fopen(in, "rb");          if (!fi) goto cleanup;
    fo = fopen(out, "wb");         if (!fo) goto cleanup;
    buf = malloc(BUF_SIZE);        if (!buf) { rc = ERR_OOM; goto cleanup; }
    /* ... work ... */
    rc = ERR_OK;
cleanup:                            // reverse acquisition order
    free(buf);
    if (fo) fclose(fo);
    if (fi) fclose(fi);
    return rc;
}
```

`[[nodiscard]]` (C23) on functions whose return MUST be checked makes C-ERR-01 a compile-time gate.

---

## 6. Toolchain, Safety Tooling & Dependencies

Build orchestration → [`cmake.md`](guides://cmake.md)/[`make.md`](guides://make.md); dependency resolution → [`conan.md`](guides://conan.md); supply-chain & CVE *policy* → [`secure-coding.md`](guides://secure-coding.md). C bindings only below.

### A. Compiler & standard

- **Baseline C17**, opt into **C23** behind `__STDC_VERSION__ >= 202311L` feature detection. Set `CMAKE_C_STANDARD 17/23`, `STANDARD_REQUIRED ON`, `EXTENSIONS OFF`.
- **Mandatory warning set** (C-WARN-01): `-Wall -Wextra -Werror -Wpedantic -Wshadow -Wconversion -Wsign-conversion -Wstrict-prototypes -Wmissing-prototypes -Wdouble-promotion -Wformat=2 -Wnull-dereference`.

### B. Sanitizers & static analysis (C owns these gates)

```bash
# ASan + UBSan build (test gate C-TST-03 / C-MEM-01 / C-UB-01)
cmake -B build-san -DCMAKE_C_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer -g -O1" \
                   -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=address,undefined"
cmake --build build-san && ctest --test-dir build-san --output-on-failure

cmake -B build-tsan -DCMAKE_C_FLAGS="-fsanitize=thread -g -O1"   # concurrent code → see parallelism.md
valgrind --leak-check=full --error-exitcode=1 ./build/app        # complementary leak/UB check

cppcheck --enable=all --error-exitcode=1 --suppress=missingInclude src/        # C-LINT-01
clang-tidy src/**/*.c -checks='bugprone-*,cert-*,clang-analyzer-*,misc-*' -- -Isrc/
```

**Safety-critical (MISRA-style) hardening** when the domain demands it: enable `clang-tidy` `cert-*`, forbid dynamic allocation after init, single function exit where mandated, no recursion, fixed-width types only, and document any deviation. Treat MISRA as an additional `clang-tidy`/`cppcheck` profile layered on the gates above, not a replacement.

### C. Banned functions (C-SEC-01)

`gets` (removed in C11), `strcpy`/`strcat` (no bounds), `sprintf` (overflow), unbounded `scanf("%s")`, `atoi` (no error detection). Use `fgets`, `snprintf`, width-limited `scanf("%255s")`, bounds-checked `memcpy`, and `strtol` with `errno`/`endptr` checking.

### D. Dependencies via Conan, orchestrated by CMake

CMake is the **single orchestrator**; Conan is bootstrapped and invoked from within CMake — never run `conan install` standalone, never hand-maintain `conanfile.txt`. Resolution priority: **C stdlib → Conan (conan-center, pinned `pkg/x.y.z`) → system package (`pkg-config`/`find_package`) → FetchContent/vendored (last resort, pinned tag)**. Each module declares its own deps:

```cmake
conan_cmake_configure(REQUIRES openssl/3.2.0 zlib/1.3.1 GENERATORS CMakeDeps CMakeToolchain)
conan_cmake_install(PATH_OR_REFERENCE . BUILD missing REMOTE conancenter SETTINGS ${CONAN_SETTINGS})
find_package(OpenSSL REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE OpenSSL::SSL ZLIB::ZLIB)
```

Pin exact versions; commit the lock; never commit compiled artifacts. Full bootstrap pattern → [`conan.md`](guides://conan.md).

---

## 7. Build System (CMake binding)

Policy → [`cmake.md`](guides://cmake.md); the C-specific skeleton:

```cmake
cmake_minimum_required(VERSION 3.25)
project(myproject VERSION 1.0.0 LANGUAGES C)
set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)
add_subdirectory(cmake/conan)                       # bootstrap Conan once
include(CompilerWarnings)                            # the §6.A warning set
option(ENABLE_SANITIZERS "ASan+UBSan" OFF)
if(ENABLE_SANITIZERS)
    add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer)
    add_link_options(-fsanitize=address,undefined)
endif()
add_library(core STATIC src/core/domain.c src/utils/error.c)
target_include_directories(core PUBLIC src/)
add_executable(app src/main.c); target_link_libraries(app PRIVATE core)
enable_testing()
add_executable(test_domain tests/unit/test_domain.c); target_link_libraries(test_domain PRIVATE core)
add_test(NAME test_domain COMMAND test_domain)
```

`CMakePresets.json` defines `dev` (Debug+sanitizers), `release`, `sanitize`. A `Makefile` is a thin convenience wrapper (`build`/`test`/`lint`/`sanitize`/`valgrind`/`check`) over `cmake --preset` — see [`make.md`](guides://make.md).

---

## 8. Modern C Features (C17 → C23)

Use C23 behind `#if __STDC_VERSION__ >= 202311L`.

| Feature | Standard | Use |
|---|---|---|
| `static_assert` (keyword) | C23 (C11 via `<assert.h>`) | compile-time invariants |
| `nullptr` | C23 | type-safe null constant |
| `constexpr` | C23 | true compile-time constants (replaces many `#define`) |
| `typeof` / `auto` | C23 | type inference in macros/locals |
| `[[nodiscard]]` / `[[maybe_unused]]` / `[[deprecated]]` | C23 | enforce/annotate at compile time |
| `<stdckdint.h>` `ckd_add/sub/mul` | C23 | checked integer arithmetic (UB-safe) |
| `enum E : type` | C23 | fixed underlying type |
| `bool`/`true`/`false` keywords | C23 (else `<stdbool.h>`) | — |
| `_Generic` | C11 | type-safe macros |
| `_Alignas`/`_Alignof`, `_Noreturn` | C11 | alignment, no-return |

```c
#if __STDC_VERSION__ >= 202311L
  #define NODISCARD [[nodiscard]]
#else
  #define NODISCARD
#endif
NODISCARD int critical_operation(void);
```

Always pun types via `memcpy`/`union`, never via incompatible pointer casts (strict aliasing). Use fixed-width (`int32_t`, `size_t`, `uintptr_t`) — never assume `int` is 32-bit beyond a `static_assert`.

---

## 9. Concurrency, Logging & Docs — bindings

- **Concurrency** (policy → [`parallelism.md`](guides://parallelism.md)): C11 `<threads.h>` (`thrd_*`, `mtx_*`, `_Atomic`) or POSIX `pthreads`; `<stdatomic.h>` for lock-free; every threaded build verified under **TSan**. No data races (UB).
- **Logging** (policy → [`logging.md`](guides://logging.md)): leveled, structured macros that capture `__FILE__`/`__LINE__` and an ISO-8601 timestamp; annotate variadic log functions with `__attribute__((format(printf, n, m)))` so `-Wformat=2` checks call sites.
- **Docs** (policy → [`comments.md`](guides://comments.md)): Doxygen on every public header — `@brief`, `@param[in/out]`, `@return`, `@pre`, `@post`, ownership note; generate via the `docs` CMake target; C-DOC-01 gates on zero undocumented public symbols.

### Naming & idiom quick reference

```c
// Types: snake_case_t  | Functions: module_verb_noun()  | Macros/consts/enum: UPPER_SNAKE
// Constructors: type_t *type_create(...)  → NULL on failure
// Destructors:  void type_destroy(type_t*) → NULL-safe
// Getters:      const X *type_get_x(const type_t*)
// Actions:      int type_verb(type_t*, ...) → 0 / negative error code
#define ARRAY_SIZE(a) (sizeof(a) / sizeof((a)[0]))
#define UNUSED(x) ((void)(x))
```

---

## 10. Deployment Checklist

Generated from §2 — one box per requirement ID. No new requirements.

- [ ] C-FMT-01 — `clang-format --dry-run --Werror` clean
- [ ] C-WARN-01 — compiles clean at `-Wall -Wextra -Werror -Wpedantic -Wshadow -Wconversion`
- [ ] C-LINT-01 — `cppcheck` + `clang-tidy` clean
- [ ] C-TST-01/02/03 — tests pass, bugs have regression tests, green under ASan+UBSan
- [ ] C-MEM-01 — no leaks/UAF/double-free/OOB (Valgrind or ASan)
- [ ] C-UB-01 — no undefined behavior (UBSan)
- [ ] C-MEM-02 — every allocation return value checked
- [ ] C-SEC-01 — no banned unsafe libc calls
- [ ] C-SEC-02 — `conan audit` 0 high/critical CVEs
- [ ] C-DEP-01 — dependencies pinned, lock verified
- [ ] C-DOC-01 — public headers documented, Doxygen clean
- [ ] C-ERR-01 — fallible functions signal failure (return code / out-param)
- [ ] Agent ran every Verification Protocol command and documented any fixes

---
**End of C Guidelines**
