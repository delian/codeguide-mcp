# Modern C++ Development Guidelines
Mandatory coding standards for modern C++: memory-safe, value-semantic, zero-cost, fully verified. C++23/20, GCC 14 / Clang 18, CMake 3.28+ (Presets), Conan 2.x, clang-tidy, clang-format, ASan/UBSan, GoogleTest.

---
name: cpp
title: Modern C++ Development Guidelines
version: 2.0
last_reviewed: 2026-06-05
kind: language
tools: [c++23, gcc@14, clang@18, cmake@3.28, conan@2, clang-tidy@18, clang-format@18, gtest@1.15, asan, ubsan]
requires:
  - tdd
  - secure-coding
  - error-handling
recommends:
  - hexagonal
  - designpatterns
  - cmake
  - conan
  - performance
  - parallelism
  - comments
provides:
  - modern-cpp
  - raii
  - smart-pointers
  - templates-concepts
  - value-semantics
---

> 🧭 Authored per [`CONVENTIONS.md`](guides://CONVENTIONS.md): shared concerns are referenced, not restated. This guide covers only what is unique to C++.

---

## 0. Prerequisites & References

Fetch and apply these **before** generating C++ code. Their rules are assumed below and not repeated.

> 📎 **REQUIRED — fetch & apply first:**
> - [`tdd.md`](guides://tdd.md) — test-first, Red-Green-Refactor, regression-test-before-fix, coverage. *(C++ binding: GoogleTest + CTest; run `ctest`.)*
> - [`secure-coding.md`](guides://secure-coding.md) — memory safety, supply chain, CVE policy. *(C++ binding: RAII/no raw owning pointers; ASan/UBSan; `conan audit`.)*
> - [`error-handling.md`](guides://error-handling.md) — error strategy & propagation. *(C++ binding: exceptions vs `std::expected` — see §6.)*

> 📎 **RECOMMENDED — fetch when the task touches them:**
> - [`hexagonal.md`](guides://hexagonal.md) — layering/ports/adapters *(C++ binding: abstract interfaces or concepts as ports; §3)*
> - [`designpatterns.md`](guides://designpatterns.md) — pattern mechanics *(C++ binding: RAII, CRTP, type erasure — §7)*
> - [`cmake.md`](guides://cmake.md) · [`conan.md`](guides://conan.md) — build orchestration & dependency policy *(C++ binding: §8)*
> - [`parallelism.md`](guides://parallelism.md) — concurrency model *(C++ binding: `std::jthread`, atomics, `std::execution` — §5.H)*
> - [`comments.md`](guides://comments.md) — API-doc policy *(C++ binding: Doxygen on public APIs)*
> - [`performance.md`](guides://performance.md) — perf policy *(C++ binding: move semantics, `constexpr`, zero-cost abstractions)*

> 📎 **SEE ALSO:** [`c.md`](guides://c.md) *(only for C interop / `extern "C"` boundaries)* · [`code-review.md`](guides://code-review.md) · [`ci-cd.md`](guides://ci-cd.md) · [`pre-commit.md`](guides://pre-commit.md)

---

## 1. Core Philosophies: MODERN-CPP

C++-specific principles only. TDD, security, error strategy, and architecture come from §0.

- **M**emory-safe by construction: RAII owns every resource; no `new`/`delete`/`malloc` in user code; no owning raw pointers. Pass non-owning buffers as `std::span`/`std::string_view`.
- **O**wnership is explicit: `unique_ptr` for sole ownership, `shared_ptr` only when ownership is genuinely shared, raw pointer/reference for non-owning access.
- **D**eterministic lifetimes: value semantics by default; the Rule of Zero — let the compiler generate special members; reach for the Rule of Five only when managing a resource directly.
- **E**xpress intent in the type system: concepts over SFINAE, `enum class`, strong types, `const`/`constexpr`/`consteval`/`noexcept` correctness.
- **R**anges & STL first: prefer `std::ranges` algorithms and views over hand-written loops; never reimplement what the STL provides.
- **N**o legacy C: no C-style casts, C arrays, `printf`-family, or manual buffers; use `std::format`/`std::print`, `std::array`/`std::vector`, `static_cast`.
- **C**ompile-time over run-time: `constexpr`/`consteval`, `static_assert`, concepts to move checks and computation to compile time (zero-cost — see `performance.md`).
- **P**ortable & warning-clean: standard library first; builds clean under `-Wall -Wextra -Wpedantic -Werror` and clang-tidy.
- **P**redictable errors: exceptions for truly exceptional conditions, `std::expected` for expected failures (policy in `error-handling.md`, binding in §6).

**Verified Code**: Agent-generated C++ MUST pass every gate in §2 before delivery.

---

## 2. Requirements (MANDATORY, auditable)

RFC-2119 keywords. IDs `CPP-<TOPIC>-<NN>`. Each row has a binary gate; rows binding a shared rule cite its owner.

| ID | Requirement | Verify | Gate |
|----|-------------|--------|------|
| CPP-TST-01 | Every feature MUST be test-first (see `tdd.md`) | `ctest --test-dir build --output-on-failure` | exit 0, 0 skips |
| CPP-TST-02 | Each bug MUST get a regression test before the fix (see `tdd.md`) | `ctest --test-dir build` | failing→passing |
| CPP-TST-03 | Business-logic line coverage MUST be ≥ 90% | `gcovr`/`llvm-cov` against the gate | ≥ 90% |
| CPP-FMT-01 | Code MUST be clang-format clean | `clang-format --dry-run --Werror $(git ls-files '*.cpp' '*.hpp')` | no diff |
| CPP-LINT-01 | clang-tidy MUST pass clean | `clang-tidy -p build $(git ls-files '*.cpp')` | 0 warnings |
| CPP-WARN-01 | Build MUST be warning-free | build with `-Wall -Wextra -Wpedantic -Werror` (`/W4 /WX` on MSVC) | exit 0 |
| CPP-MEM-01 | No owning raw pointers / no manual `new`/`delete` in user code (see `secure-coding.md`) | clang-tidy `cppcoreguidelines-owning-memory`, review | 0 findings |
| CPP-SAN-01 | ASan + UBSan clean on the test suite (see `secure-coding.md`) | tests built `-fsanitize=address,undefined`, run `ctest` | 0 errors |
| CPP-ERR-01 | Error path uses exceptions or `std::expected` per policy (see `error-handling.md`) | review | no raw error codes/`errno` leakage |
| CPP-SEC-01 | 0 high/critical CVEs in deps (see `secure-coding.md`) | `conan audit scan .` | 0 high/critical |
| CPP-DEP-01 | Build & deps reproducible from manifest (see `conan.md`/`cmake.md`) | clean `cmake --preset … && cmake --build` in Docker | succeeds |
| CPP-DOC-01 | Public APIs documented with Doxygen (see `comments.md`) | `doxygen` / `--target docs` | 0 warnings |
| CPP-ARCH-01 | Domain layer free of framework/IO deps (see `hexagonal.md`) | review / link-time deps | no inward→outward |

> **Forbidden**: shipping implementation before its test (violates `tdd.md`); fixing a bug without a regression test first; owning raw pointers, `new`/`delete`, or C-style casts in user code; `strcpy`/`sprintf`/`gets` and fixed C buffers; suppressing ASan/UBSan instead of fixing; `DISABLED_` tests without a documented reason.

---

## 3. Verification Protocol

Run, in order, before presenting code. Fix → re-run until every gate is green.

```bash
cmake --preset debug                                    # configure (Presets, exports compile_commands.json)
cmake --build --preset debug                            # CPP-WARN-01 (-Werror)
clang-format --dry-run --Werror $(git ls-files '*.cpp' '*.hpp')   # CPP-FMT-01
clang-tidy -p build $(git ls-files '*.cpp')             # CPP-LINT-01, CPP-MEM-01
ctest --preset debug --output-on-failure                # CPP-TST-01/02; built with -fsanitize=address,undefined → CPP-SAN-01
gcovr --fail-under-line 90 build                        # CPP-TST-03
conan audit scan .                                      # CPP-SEC-01
```

The *why* behind each gate lives in its §0 owner; do not re-derive it here.

---

## 4. Project Structure

Idiomatic modular C++ layout. Architectural *principles* (dependency direction, ports/adapters, acyclic deps) are owned by [`hexagonal.md`](guides://hexagonal.md); below is only their C++ mapping. CMake/Conan wiring is owned by [`cmake.md`](guides://cmake.md) / [`conan.md`](guides://conan.md).

```
project/
├── CMakeLists.txt          # root: project(), C++ standard, add_subdirectory (see cmake.md)
├── CMakePresets.json       # configure/build/test presets
├── conanfile.py            # dependency manifest (see conan.md)
├── cmake/                  # CompilerWarnings/Sanitizers/StaticAnalyzers modules
├── src/<module>/
│   ├── include/<module>/   # public headers (the port surface) — domain pure, no IO
│   └── src/                # implementation (adapters live here)
├── apps/                   # executables (composition root)
├── tests/<module>/         # GoogleTest suites, mirror src/ (see tdd.md)
├── .clang-format
└── .clang-tidy
```

- Group by feature/domain, not by type; one public header dir per module forms its interface.
- Domain types and use cases depend only on abstractions (interfaces/concepts), never on adapters (CPP-ARCH-01).
- No circular dependencies between modules; enforce with target-level CMake deps.

---

## 5. C++ Specifics — the unique value

### A. RAII & ownership (the central idiom)
Every resource is owned by an object whose destructor releases it; cleanup is deterministic and exception-safe — no `try/finally`, no leaks. Prefer the **Rule of Zero**: design types so the compiler-generated special members are correct (hold members by value / smart pointer). Define the **Rule of Five** only when wrapping a raw resource directly.

```cpp
// Rule of Zero: nothing to write — members manage themselves.
class Session {
  std::string id_;
  std::vector<std::byte> buffer_;
  std::unique_ptr<Connection> conn_;   // owns; move-only propagates automatically
};

// Rule of Five only when wrapping a C resource directly — prefer a unique_ptr deleter instead:
using FilePtr = std::unique_ptr<std::FILE, decltype([](std::FILE* f){ if (f) std::fclose(f); })>;
[[nodiscard]] auto open_file(const char* path) -> FilePtr {
  FilePtr f{std::fopen(path, "rb")};
  if (!f) throw std::runtime_error("open failed");      // RAII closes on any later throw
  return f;
}
```

### B. Smart pointers — choosing ownership
- `std::unique_ptr<T>` — default for heap ownership; zero overhead; move-only. Create with `std::make_unique`.
- `std::shared_ptr<T>` — only when ownership is genuinely shared; create with `std::make_shared`. Break cycles with `std::weak_ptr`.
- Raw pointer / reference / `std::span` / `std::string_view` — **non-owning** access; never `delete` them.
- Never `new`/`delete` in user code (CPP-MEM-01). Custom cleanup → `unique_ptr` with a deleter (see §5.A).

```cpp
auto widget = std::make_unique<Widget>(args...);   // sole owner
void use(const Widget& w);                          // borrow, no ownership transfer
use(*widget);
```

### C. Value & move semantics
Default to value types; let RVO/NRVO elide copies. Take sink parameters **by value and `std::move`** into place; expose accessors by `const&` for large members. Mark moves `noexcept` so containers can use them.

```cpp
class User {
public:
  User(std::string name, std::string email)               // by value
      : name_{std::move(name)}, email_{std::move(email)} {}
  [[nodiscard]] auto name() const noexcept -> const std::string& { return name_; }
private:
  std::string name_, email_;
};
db.add(std::move(user));                                   // transfer, no copy
```

### D. const / constexpr / noexcept correctness
Mark every non-mutating member `const`; compute at compile time with `constexpr`/`consteval`; mark `noexcept` where no exception can escape (especially moves, swaps, destructors). Use `[[nodiscard]]` on functions whose result must be used.

```cpp
[[nodiscard]] constexpr auto dot(Vec3 a, Vec3 b) noexcept -> double {
  return a.x*b.x + a.y*b.y + a.z*b.z;                      // usable in compile-time contexts
}
static_assert(dot({1,0,0}, {0,1,0}) == 0.0);
```

### E. Templates & concepts (C++20)
Constrain templates with **concepts**, never legacy `enable_if`/SFINAE — clearer errors and intent. Use `requires` clauses for ad-hoc constraints.

```cpp
template <typename T>
concept Numeric = std::integral<T> || std::floating_point<T>;

template <Numeric T>
[[nodiscard]] constexpr auto clamp(T v, T lo, T hi) noexcept -> T {
  return std::min(std::max(v, lo), hi);
}
```

### F. Ranges & the STL
Prefer `std::ranges` algorithms and lazy views; they compose without temporaries and read top-to-bottom. Reach for an STL algorithm before writing a raw loop.

```cpp
auto evens_doubled = data
  | std::views::filter([](int x){ return x % 2 == 0; })
  | std::views::transform([](int x){ return x * 2; })
  | std::views::take(10);
std::ranges::sort(items, std::less{}, &Item::key);          // projection
```

### G. Modern strings & formatting
`std::string_view` for non-owning read-only params, `std::string` for owned results. Format with `std::format` / `std::print` (C++23) — never `printf`/`sprintf`.

```cpp
[[nodiscard]] auto greet(std::string_view name, int age) -> std::string {
  return std::format("{} is {} years old", name, age);
}
std::print("{:>8.2f}\n", value);                            // C++23 std::print
```

### H. Concurrency binding
Model and policy (data races, happens-before, lock discipline) are owned by [`parallelism.md`](guides://parallelism.md). C++ tools: `std::jthread` (auto-join, `stop_token`) over bare `std::thread`; `std::atomic` with explicit memory order; `std::scoped_lock`/`std::shared_mutex` (never manual lock/unlock — not exception-safe); `std::latch`/`std::barrier`/`std::counting_semaphore` for coordination; `std::async`/`std::future` and parallel algorithms (`std::execution::par`). Verify under TSan (`-fsanitize=thread`).

```cpp
std::jthread worker{[](std::stop_token st){
  while (!st.stop_requested()) { /* ... */ }
}};                                                         // joins & requests stop on destruction
```

### I. Common footguns → fixes
- Dangling `string_view`/`span` into a temporary → bind to a named owner; never return a view of a local.
- Returning reference to a local / member after move → return by value or check lifetimes.
- `shared_ptr` cycles → `weak_ptr` for back-references.
- Object slicing (storing derived in a base-by-value container) → store `unique_ptr<Base>` or use `std::variant`.
- Iterator/reference invalidation after `push_back`/`erase` → re-acquire after mutation; `reserve` up front.
- Integer signedness/overflow in indexing → use `std::ssize`, `size_t`, or checked arithmetic (policy: `secure-coding.md`).
- Forgetting `virtual` destructor on a polymorphic base → declare `virtual ~Base() = default;`.

---

## 6. Error Handling — C++ binding

Strategy (when to fail, how to propagate, what to log) is owned by [`error-handling.md`](guides://error-handling.md). C++ mechanics:

- **Exceptions** for genuinely exceptional / unrecoverable conditions and constructor failure. Throw `std::exception`-derived types by value, catch by `const&`. Provide the **strong exception guarantee** via RAII and copy-and-swap. Mark functions `noexcept` only when they truly cannot throw.
- **`std::expected<T, E>`** (C++23) for *expected* failures in a function's normal contract (parsing, lookups, validation) — no control-flow-by-exception, no out-params, no sentinel returns.
- Never use C-style error codes/`errno` or bare `int` returns in user APIs (CPP-ERR-01).

```cpp
enum class ParseError { Empty, BadFormat, OutOfRange };

[[nodiscard]] auto parse_int(std::string_view s) -> std::expected<int, ParseError> {
  if (s.empty()) return std::unexpected(ParseError::Empty);
  int value{};
  const auto [ptr, ec] = std::from_chars(s.data(), s.data() + s.size(), value);
  if (ec == std::errc::invalid_argument)   return std::unexpected(ParseError::BadFormat);
  if (ec == std::errc::result_out_of_range) return std::unexpected(ParseError::OutOfRange);
  return value;
}

// Monadic composition (C++23):
auto port = parse_int(arg).and_then(validate_port).value_or(8080);
```

---

## 7. Design Patterns — C++ binding

Pattern mechanics are owned by [`designpatterns.md`](guides://designpatterns.md). Show only the C++-idiomatic realization:

- **RAII / Scope Guard** — the canonical C++ resource pattern (§5.A); cleanup via destructors, not GC.
- **Strategy / type erasure** — store behavior as `std::function`, a concept-constrained template, or a hand-rolled type-erased wrapper (à la `std::any`/`std::function`).
- **CRTP** — static polymorphism for zero-cost interfaces: `template <class D> struct Base { void f(){ static_cast<D*>(this)->impl(); } };`.
- **Visitor** — `std::variant` + `std::visit` over an `enum`-tagged hierarchy; prefer it to virtual `accept()` for closed sets.
- **Factory** — return `std::unique_ptr<Interface>`; the composition root wires implementations (ports/adapters, see `hexagonal.md`).
- **Pimpl** — `std::unique_ptr<Impl>` to break compile-time coupling and stabilize ABI.

---

## 8. Tooling, Build & Dependencies — C++ binding

Build orchestration is owned by [`cmake.md`](guides://cmake.md); dependency/supply-chain policy by [`conan.md`](guides://conan.md) and [`secure-coding.md`](guides://secure-coding.md); versioning by [`semver.md`](guides://semver.md). C++ essentials only:

- **CMake 3.28+ with Presets**: `CMakePresets.json` drives configure/build/test; set `CMAKE_CXX_STANDARD 23`, `CXX_STANDARD_REQUIRED ON`, `CXX_EXTENSIONS OFF`, `CMAKE_EXPORT_COMPILE_COMMANDS ON`. Modern target-based CMake — `target_link_libraries`/`target_compile_features`, no global `include_directories`.
- **Conan 2.x** as the package manager; pin exact versions; integrate via the CMakeToolchain/CMakeDeps generators so `cmake --preset` drives the whole build. Commit the lockfile; scan with `conan audit`. (Full wiring → `conan.md`.)
- **Warnings as errors**: `-Wall -Wextra -Wpedantic -Werror -Wshadow -Wconversion -Wsign-conversion` (GCC/Clang) or `/W4 /WX` (MSVC).
- **clang-tidy** (`-p build`) with `cppcoreguidelines-*`, `modernize-*`, `bugprone-*`, `performance-*`; **clang-format** for layout.
- **Sanitizers**: ASan+UBSan on the default test build (CPP-SAN-01); TSan for multithreaded code; MSan separately (they are mutually exclusive with ASan/TSan).
- **GoogleTest + CTest**: `gtest_discover_tests()`; run via `ctest` (binding for `tdd.md`).

```bash
conan install . -s build_type=Debug --build=missing   # resolve deps (toolchain for CMake)
cmake --preset debug                                   # configure via preset
cmake --build --preset debug                           # warning-clean build
ctest --preset debug --output-on-failure               # tests + sanitizers
```

---

## 9. Quick Reference

```bash
cmake --preset debug && cmake --build --preset debug   # configure + build
ctest --preset debug --output-on-failure               # test (with ASan/UBSan)
clang-tidy -p build $(git ls-files '*.cpp')            # lint
clang-format -i $(git ls-files '*.cpp' '*.hpp')        # format
cmake --build build --target docs                      # Doxygen
conan audit scan .                                     # CVE scan
```

```cpp
auto p   = std::make_unique<T>(args...);               // sole ownership
auto opt = find(id);  if (opt) use(*opt);              // std::optional
std::expected<T, E> r = parse(s);                      // expected error
auto v = items | std::views::filter(f) | std::views::transform(g);   // ranges
auto [k, val] = pair;                                  // structured bindings
```

---

## 10. Deployment Checklist

Generated from §2 — one box per requirement ID.

- [ ] CPP-FMT-01 — clang-format clean, no diff
- [ ] CPP-LINT-01 — clang-tidy clean
- [ ] CPP-WARN-01 — warning-free build (`-Werror` / `/WX`)
- [ ] CPP-TST-01/02/03 — tests pass, bugs have regression tests, coverage ≥ 90%
- [ ] CPP-MEM-01 — no owning raw pointers / no manual `new`/`delete`
- [ ] CPP-SAN-01 — ASan + UBSan clean (TSan if multithreaded)
- [ ] CPP-ERR-01 — errors via exceptions / `std::expected` per policy
- [ ] CPP-SEC-01 — `conan audit` 0 high/critical CVEs
- [ ] CPP-DEP-01 — reproducible build from manifest (clean Docker build)
- [ ] CPP-DOC-01 — public APIs Doxygen-documented, no warnings
- [ ] CPP-ARCH-01 — domain layer free of framework/IO deps
- [ ] Agent ran every §3 command and documented any fixes

---
**End of Modern C++ Guidelines**
