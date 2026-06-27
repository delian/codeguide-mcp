# Zig Development Guidelines
Mandatory coding standards for Zig: explicit allocators, error unions, comptime, no hidden control flow. Zig 0.14.x, zig build, zig test, zig fmt.

---
name: zig
title: Zig Development Guidelines
version: 2.0
last_reviewed: 2026-06-05
kind: language
tools: [zig@0.14, zig-build, zig-test, zig-fmt]
requires:
  - tdd
  - secure-coding
  - error-handling
recommends:
  - c
  - performance
  - parallelism
  - comments
provides:
  - explicit-allocators
  - error-unions
  - comptime
  - zig-build
  - c-interop
---

> 🧭 Authored per [`CONVENTIONS.md`](guides://CONVENTIONS.md): shared concerns are referenced, not restated. This guide covers only what is unique to Zig.

> ⚠️ **Zig is pre-1.0.** The language and standard library change between minor releases (e.g. the `std.ArrayList` managed→unmanaged shift, `std.builtin.Type` tag renames, build-system signatures). Pin the exact compiler in `build.zig.zon` (`minimum_zig_version`) and treat every upgrade as a breaking change: re-run all gates in §2 against the new toolchain. This guide targets **0.14.x**.

---

## 0. Prerequisites & References

Fetch and apply these **before** generating Zig code. Their rules are assumed below and not repeated.

> 📎 **REQUIRED — fetch & apply first:**
> - [`tdd.md`](guides://tdd.md) — test-first, Red-Green-Refactor, regression-test-before-fix, coverage. *(Zig binding: tests live in `test "..." {}` blocks in the source file; runner is `zig build test`.)*
> - [`error-handling.md`](guides://error-handling.md) — error strategy & propagation. *(Zig owns this strongly: error unions `!T`, error sets, `try`/`catch`/`errdefer` are the canonical mechanism — see §5.B.)*
> - [`secure-coding.md`](guides://secure-coding.md) — supply chain, secrets, CVE policy, **memory safety**. *(Zig binding: build/test in safety-checked modes; verify `build.zig.zon` hashes; detect leaks with `std.testing.allocator`.)*

> 📎 **RECOMMENDED — fetch when the task touches them:**
> - [`c.md`](guides://c.md) — C interop & manual memory semantics *(binding: `@cImport`, `translate-c`, `std.heap.c_allocator`, `extern` — see §5.H)*
> - [`performance.md`](guides://performance.md) — data-oriented design, cache layout, SIMD.
> - [`parallelism.md`](guides://parallelism.md) — threading & concurrency *(binding: `std.Thread`, `std.Thread.Pool`, atomics, `std.Thread.Mutex`)*
> - [`comments.md`](guides://comments.md) — doc-comment/API-doc policy *(binding: `///` item docs, `//!` module docs, `zig build` `-femit-docs`)*

> 📎 **SEE ALSO:** [`hexagonal.md`](guides://hexagonal.md) · [`designpatterns.md`](guides://designpatterns.md) · [`semver.md`](guides://semver.md)

---

## 1. Core Philosophies: ZIG-FIRST

Zig-specific principles only. TDD, security/memory-safety, and error strategy come from §0.

- **Z**ero hidden control flow: no hidden allocations, no hidden control flow, no operator overloading, no exceptions. If a function allocates, it takes an `Allocator`. If it can fail, it returns `!T`.
- **I**ntentional memory: allocators are passed explicitly; every allocation is paired with `defer`/`errdefer`; ownership (who frees) is documented. Prefer arenas for batch/scoped lifetimes.
- **G**enerics via comptime: types are values at compile time; use `comptime` for generic data structures, type introspection, and zero-cost specialization — not runtime reflection.
- **F**ail loud & explicit: error unions over sentinels; optionals (`?T`) over null pointers; explicit casts (`@intCast`, `@ptrCast`); `unreachable` only with a proven invariant.

**Verified Code**: Agent-generated Zig MUST pass every gate in §2 before delivery.

---

## 2. Requirements (MANDATORY, auditable)

RFC-2119 keywords. IDs `ZIG-<TOPIC>-<NN>`. Each row has a binary gate; rows binding a shared rule cite its owner.

| ID | Requirement | Verify | Gate |
|----|-------------|--------|------|
| ZIG-TST-01 | Every feature MUST be test-first (see `tdd.md`) | `zig build test` | exit 0, 0 skips |
| ZIG-TST-02 | Each bug MUST get a regression test before the fix (see `tdd.md`) | `zig build test` | failing→passing |
| ZIG-BUILD-01 | Code MUST compile clean | `zig build` | exit 0, no warnings |
| ZIG-FMT-01 | Code MUST be formatted | `zig fmt --check .` | no diff |
| ZIG-MEM-01 | No memory leaks; tests MUST use `std.testing.allocator` (see `secure-coding.md`) | `zig build test` | 0 leaks reported |
| ZIG-SAFE-01 | MUST build/test in a safety-checked mode; no UB; `unreachable` only with a stated invariant (see `secure-coding.md`) | `zig build test -Doptimize=ReleaseSafe` | exit 0 |
| ZIG-ERR-01 | No error union may be silently discarded; alloc-then-fail paths MUST use `errdefer` (see `error-handling.md`) | `zig build` + review | exit 0, no `catch unreachable`/`catch undefined` without justification |
| ZIG-SEC-01 | Dependency hashes MUST be pinned & verifiable (see `secure-coding.md`) | `zig build --fetch` | 0 hash mismatches |
| ZIG-DEP-01 | `build.zig.zon` committed; `minimum_zig_version` pinned | review | present & pinned |
| ZIG-DOC-01 | Public (`pub`) items MUST have `///` doc comments (see `comments.md`) | `zig build docs` | builds clean |

> **Forbidden**: shipping implementation before its test (violates `tdd.md`); fixing a bug without a regression test first; hidden/global allocators; an allocation without a paired `defer`/`errdefer`; `catch unreachable` on a recoverable error; ignoring an `error{}` return; `@ptrCast` without matching alignment handling.

---

## 3. Verification Protocol

Run, in order, before presenting code. Fix → re-run until every gate is green.

```bash
zig fmt --check .                       # ZIG-FMT-01
zig build                               # ZIG-BUILD-01
zig build test                          # ZIG-TST-01/02, ZIG-MEM-01 (testing.allocator)
zig build test -Doptimize=ReleaseSafe   # ZIG-SAFE-01 (safety checks retained)
zig build --fetch                       # ZIG-SEC-01: verify dependency hashes
zig build docs                          # ZIG-DOC-01 (if a docs step is defined)
```

The *why* behind each gate lives in its §0 owner; do not re-derive it here.

---

## 4. Project Structure

Idiomatic Zig layout. Architectural *principles* (layering, ports/adapters, dependency direction) are owned by [`hexagonal.md`](guides://hexagonal.md); below is only their Zig mapping.

```
project/
├── build.zig            # build graph (the canonical entry point)
├── build.zig.zon        # package manifest + pinned dependency hashes
├── src/
│   ├── main.zig         # binary entry point (I/O, allocator setup)
│   ├── root.zig         # library root (re-exports public API)
│   └── <feature>/       # group by domain/feature, not by type
└── tests/               # integration tests importing the library module
```

- Unit tests live **inside** the source file as `test` blocks (next to the code they cover). Integration tests go under `tests/` and import the built module.
- Wire test files into `zig build test` so every block runs (a `test` block in an un-referenced file never executes).
- Domain code stays free of OS/IO imports; push `std.fs`/`std.net`/C calls to the edges (see `hexagonal.md`).

---

## 5. Zig Specifics

The unique value of this guide.

### A. Explicit allocators & ownership

No function allocates from a hidden global. Pass `std.mem.Allocator`; pair every allocation with `defer` (always free) or `errdefer` (free only on the error path); document who owns the result.

```zig
const std = @import("std");

/// Caller owns the returned slice and must free it with `allocator`.
pub fn join(allocator: std.mem.Allocator, parts: []const []const u8, sep: u8) ![]u8 {
    var list = std.ArrayList(u8).init(allocator);
    errdefer list.deinit();                 // freed only if we return an error
    for (parts, 0..) |p, i| {
        if (i != 0) try list.append(sep);
        try list.appendSlice(p);
    }
    return list.toOwnedSlice();             // ownership transfers to caller
}
```

Pick the allocator deliberately:
- `std.heap.GeneralPurposeAllocator(.{})` — app default; reports leaks/double-frees on `deinit()`.
- `std.heap.ArenaAllocator` — scoped/batch lifetimes (per request, per frame); free everything at once with one `arena.deinit()`, no per-object `defer`.
- `std.testing.allocator` — in tests; **fails the test on any leak** (this is how ZIG-MEM-01 is enforced).
- `std.heap.c_allocator` / `std.heap.page_allocator` — C interop / page-granular only.

```zig
pub fn handleRequest(parent: std.mem.Allocator, req: Request) !Response {
    var arena = std.heap.ArenaAllocator.init(parent);
    defer arena.deinit();                   // one free for the whole request
    const a = arena.allocator();
    const user = try loadUser(a, req.id);   // no individual defer needed
    return try render(a, user);
}
```

### B. Error unions, error sets & errdefer (canonical — see `error-handling.md`)

Zig is the strong owner of this idiom. Errors are values in an **error set**; a fallible function returns `!T` (error-union). `try` propagates; `catch` handles; `errdefer` unwinds allocations on the error path only.

```zig
pub const ParseError = error{ Empty, BadDigit } || std.mem.Allocator.Error;  // set composition

pub fn parse(allocator: std.mem.Allocator, s: []const u8) ParseError!Parsed {
    if (s.len == 0) return error.Empty;
    const buf = try allocator.dupe(u8, s);
    errdefer allocator.free(buf);           // freed iff a later step fails
    try validate(buf);                      // propagates BadDigit
    return Parsed{ .raw = buf };
}

// Handle specific errors; never blanket-swallow:
const v = parse(a, input) catch |err| switch (err) {
    error.Empty => return defaultValue,
    else => return err,                     // re-propagate the rest
};
```

- Prefer a named, explicit error set over inferred `!T` on public APIs — it documents the contract.
- `catch unreachable` / `catch undefined` are forbidden on recoverable errors (ZIG-ERR-01); use only where an invariant *proves* the error cannot occur, and say why in a comment.
- Strategy (when to recover vs. fail-fast, error taxonomy) is owned by [`error-handling.md`](guides://error-handling.md); the syntax above is the Zig binding.

### C. Optionals — no null pointers

`?T` replaces nullable pointers. Unwrap with `if (x) |v|`, `x orelse default`, or `x.?` (only when proven non-null).

```zig
fn lookup(map: *const Map, key: []const u8) ?u32 { return map.get(key); }

const port = lookup(&cfg, "port") orelse 8080;      // default
if (lookup(&cfg, "id")) |id| use(id);               // safe unwrap
```

### D. defer / errdefer

`defer` runs at scope exit in LIFO order (always). `errdefer` runs only when the scope returns an error. Place each immediately after the resource is acquired so cleanup can't be forgotten.

```zig
const file = try std.fs.cwd().openFile(path, .{});
defer file.close();                          // always closes
const data = try file.readToEndAlloc(a, max);
defer a.free(data);                          // always frees
```

### E. comptime — generics & type introspection

Types are first-class comptime values. Generic containers are functions returning a `type`. `@typeInfo` enables compile-time reflection with zero runtime cost.

```zig
pub fn Stack(comptime T: type) type {
    return struct {
        const Self = @This();
        items: std.ArrayList(T),
        pub fn init(a: std.mem.Allocator) Self { return .{ .items = std.ArrayList(T).init(a) }; }
        pub fn deinit(self: *Self) void { self.items.deinit(); }
        pub fn push(self: *Self, v: T) !void { try self.items.append(v); }
        pub fn pop(self: *Self) ?T { return self.items.popOrNull(); }
    };
}
```

```zig
// 0.14 tag names are lowercase; keyword tags use @"...":
pub fn fieldNames(comptime T: type) []const []const u8 {
    const info = @typeInfo(T);
    if (info != .@"struct") @compileError("expected a struct, got " ++ @typeName(T));
    comptime var names: []const []const u8 = &.{};
    inline for (info.@"struct".fields) |f| names = names ++ &[_][]const u8{f.name};
    return names;
}
```

> The `.Struct`→`.@"struct"`, `.Int`→`.int` rename and `async`/`await` **removal** are exactly the kind of pre-1.0 churn flagged at the top — verify reflection and concurrency code against your pinned compiler.

### F. Slices & packed structs

A slice (`[]T`) is a pointer+length — bounds-checked in safety modes; prefer it over raw pointers. Use sentinel-terminated slices (`[:0]const u8`) for C strings. `packed struct` gives a guaranteed bit layout for protocols/registers; `extern struct` gives C ABI layout.

```zig
const Flags = packed struct(u8) { read: bool, write: bool, exec: bool, _pad: u5 = 0 };
const rgba = packed struct(u32) { r: u8, g: u8, b: u8, a: u8 };
fn sum(xs: []const i64) i64 { var t: i64 = 0; for (xs) |x| t += x; return t; }  // bounds-checked
```

### G. The build system — `build.zig` (0.14)

`build.zig` is a normal Zig program describing a build graph; use `b.path(...)` (the old `.{ .path = "..." }` anonymous-struct form is removed). Define `run`, `test`, and `docs` steps so §3 commands work.

```zig
const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const exe = b.addExecutable(.{
        .name = "myapp",
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });
    b.installArtifact(exe);

    const run = b.addRunArtifact(exe);
    if (b.args) |args| run.addArgs(args);
    b.step("run", "Run the app").dependOn(&run.step);

    const tests = b.addTest(.{ .root_source_file = b.path("src/root.zig"), .target = target, .optimize = optimize });
    b.step("test", "Run unit tests").dependOn(&b.addRunArtifact(tests).step);

    const docs = b.addInstallDirectory(.{
        .source_dir = tests.getEmittedDocs(),
        .install_dir = .prefix,
        .install_subdir = "docs",
    });
    b.step("docs", "Build API docs").dependOn(&docs.step);
}
```

Build/test in `-Doptimize=Debug` or `ReleaseSafe` during development (safety checks: bounds, overflow, null/undefined). `ReleaseFast`/`ReleaseSmall` drop those checks — use only for shipping artifacts that already passed ZIG-SAFE-01.

### H. C interop (see `c.md`)

Zig consumes C headers directly via `@cImport`/`@cInclude` (or `zig translate-c`) and links C libraries from `build.zig`. Manual-memory and C-string semantics are owned by [`c.md`](guides://c.md); the Zig binding:

```zig
const c = @cImport({
    @cInclude("sqlite3.h");
});
// build.zig: exe.linkSystemLibrary("sqlite3"); exe.linkLibC();
```

- Use `std.heap.c_allocator` when memory crosses the C boundary; match `malloc`/`free` ownership.
- Bridge pointers with `@ptrCast` **plus** `@alignCast` when alignment differs; pass `[:0]const u8` for `const char *`.
- Export Zig to C with `export fn` / `extern`. Keep the `@cImport` surface in one adapter module, not scattered through the domain.

---

## 6. Tooling & Dependencies

Supply-chain *policy* → [`secure-coding.md`](guides://secure-coding.md); versioning → [`semver.md`](guides://semver.md). Zig binding via `build.zig.zon`:

```zig
.{
    .name = "myapp",
    .version = "0.1.0",
    .minimum_zig_version = "0.14.0",          // ZIG-DEP-01: pin the compiler
    .dependencies = .{
        .mylib = .{
            .url = "https://github.com/example/mylib/archive/<commit>.tar.gz",
            .hash = "1220...",                 // ZIG-SEC-01: cryptographic hash, verified on fetch
        },
    },
    .paths = .{ "build.zig", "build.zig.zon", "src" },
}
```

```bash
zig fetch --save <url>   # add a dependency (writes the verified hash into build.zig.zon)
zig build --fetch        # ZIG-SEC-01: fetch all deps & verify hashes match
```

- Commit `build.zig.zon` (it is the lockfile/source of truth for hashes).
- Wire deps into modules in `build.zig`: `exe.root_module.addImport("mylib", dep.module("mylib"));`
- Prefer dependencies pinned to a commit/tag with a hash over moving `main` tarballs.

---

## 7. Quick Reference

```bash
zig build                               # build
zig build run -- <args>                 # run
zig build test                          # test (+ leak detection via testing.allocator)
zig build test -Doptimize=ReleaseSafe   # test with safety checks, optimized
zig fmt .                               # format
zig fmt --check .                       # format gate (CI)
zig fetch --save <url>                  # add a pinned dependency
zig build --fetch                       # verify dependency hashes
```

---

## 8. Deployment Checklist

Generated from §2 — one box per requirement ID.

- [ ] ZIG-BUILD-01 — `zig build` clean, no warnings
- [ ] ZIG-FMT-01 — `zig fmt --check .` clean
- [ ] ZIG-TST-01/02 — tests pass, bugs have regression tests, 0 skips
- [ ] ZIG-MEM-01 — `std.testing.allocator` reports 0 leaks
- [ ] ZIG-SAFE-01 — `ReleaseSafe` build/test passes; no unjustified `unreachable`
- [ ] ZIG-ERR-01 — no discarded error unions; `errdefer` on alloc-then-fail paths
- [ ] ZIG-SEC-01 — `zig build --fetch` verifies all dependency hashes
- [ ] ZIG-DEP-01 — `build.zig.zon` committed, `minimum_zig_version` pinned
- [ ] ZIG-DOC-01 — public items documented, `zig build docs` clean
- [ ] Agent ran every §3 command and documented any fixes

---
**End of Zig Guidelines**
