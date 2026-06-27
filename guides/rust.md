# Rust Development Guidelines
Mandatory coding standards for Rust: memory-safe, type-driven, test-covered, zero-cost. Rust stable 1.85+ (2024 edition), cargo, clippy, rustfmt, cargo-audit, cargo-deny.

---
name: rust
title: Rust Development Guidelines
version: 2.0
last_reviewed: 2026-06-05
kind: language
tools: [rust@1.85, cargo, clippy, rustfmt, cargo-audit, cargo-deny, cargo-nextest]
requires:
  - tdd
  - secure-coding
  - error-handling
recommends:
  - hexagonal
  - performance
  - parallelism
  - comments
  - semver
provides:
  - ownership-borrowing
  - traits-generics
  - result-option
  - cargo-workflow
  - unsafe-discipline
---

> 🧭 Authored per [`CONVENTIONS.md`](guides://CONVENTIONS.md): shared concerns are referenced, not restated. This guide covers only what is unique to Rust.

---

## 0. Prerequisites & References

Fetch and apply these **before** generating Rust code. Their rules are assumed below and not repeated.

> 📎 **REQUIRED — fetch & apply first:**
> - [`tdd.md`](guides://tdd.md) — test-first, Red-Green-Refactor, regression-test-before-fix, coverage. *(Rust binding: `cargo test` / `cargo nextest run`; doctests via `cargo test --doc`.)*
> - [`secure-coding.md`](guides://secure-coding.md) — supply chain, secrets, CVE policy. *(Rust binding: `cargo audit`, `cargo deny check`.)*
> - [`error-handling.md`](guides://error-handling.md) — error strategy & propagation. *(Rust owns the `Result`/`Option`/`?` mechanics — see §5; the strategy of what to model and where to surface errors comes from here.)*

> 📎 **RECOMMENDED — fetch when the task touches them:**
> - [`hexagonal.md`](guides://hexagonal.md) — layering, ports/adapters, dependency inversion *(binding: ports are `trait`s, adapters are `impl`s; see §6)*
> - [`parallelism.md`](guides://parallelism.md) — concurrency model *(binding: `Send`/`Sync`, `async`/tokio, `rayon`; see §7)*
> - [`comments.md`](guides://comments.md) — API-doc policy *(binding: `///` rustdoc, runnable doctests)*
> - [`performance.md`](guides://performance.md) · [`semver.md`](guides://semver.md) *(binding: crate `MAJOR.MINOR.PATCH`, pre-1.0 caveats)*

> 📎 **SEE ALSO:** [`designpatterns.md`](guides://designpatterns.md) · [`cleanarch.md`](guides://cleanarch.md) · [`code-review.md`](guides://code-review.md) · [`ci-cd.md`](guides://ci-cd.md)

---

## 1. Core Philosophies: RUST-FIRST

Rust-specific principles only. TDD, security, error strategy, and architecture come from §0.

- **R**AII & ownership: every value has one owner; resources (files, locks, sockets) free deterministically via `Drop`. No GC, no manual `free`. Borrow instead of clone; reach for `.clone()` only when a measured need exists.
- **U**nsafe is rare and justified: `#![forbid(unsafe_code)]` by default; any `unsafe` block carries a `// SAFETY:` comment proving its invariants (§8).
- **S**trong types over primitives: newtypes wrap domain primitives; enums model closed state sets so illegal states are unrepresentable; `match` is exhaustive.
- **T**raits, not inheritance: behavior is composed through traits and generics (static dispatch) or trait objects (`dyn`, dynamic dispatch); abstraction stays zero-cost where possible.

Plus: `Result`/`Option` for all fallibility (no `panic!` in library code); immutable bindings by default (`let`, not `let mut`); iterators/combinators over index loops; `async`/await for I/O.

**Verified Code**: Agent-generated Rust MUST pass every gate in §2 before delivery.

---

## 2. Requirements (MANDATORY, auditable)

RFC-2119 keywords. IDs `RUST-<TOPIC>-<NN>`. Each row has a binary gate; rows binding a shared rule cite its owner.

| ID | Requirement | Verify | Gate |
|----|-------------|--------|------|
| RUST-TST-01 | Every feature MUST be test-first (see `tdd.md`) | `cargo test` | exit 0, 0 ignored |
| RUST-TST-02 | Each bug MUST get a regression test before the fix (see `tdd.md`) | `cargo test` | failing→passing |
| RUST-TST-03 | Doctests in public-API examples MUST run and pass | `cargo test --doc` | exit 0 |
| RUST-FMT-01 | Code MUST be formatted | `cargo fmt --check` | no diff |
| RUST-LINT-01 | Clippy MUST pass with warnings denied | `cargo clippy --all-targets --all-features -- -D warnings` | exit 0 |
| RUST-TYP-01 | Code MUST compile clean on all targets/features | `cargo check --all-targets --all-features` | exit 0 |
| RUST-ERR-01 | Library code MUST NOT `panic!`/`unwrap`/`expect` on fallible paths (see `error-handling.md`) | `cargo clippy` w/ `unwrap_used`,`expect_used` denied | 0 findings in `src/` |
| RUST-SAFE-01 | `unsafe` MUST be forbidden, or each block MUST carry a `// SAFETY:` proof | grep `unsafe` / `#![forbid(unsafe_code)]` | every block justified |
| RUST-DOC-01 | Public items MUST be documented (see `comments.md`) | `RUSTDOCFLAGS="-D warnings" cargo doc --no-deps` | exit 0, 0 missing-docs |
| RUST-SEC-01 | 0 unpatched advisories in deps (see `secure-coding.md`) | `cargo audit` | 0 vulnerabilities |
| RUST-SEC-02 | License/source/ban policy MUST hold (see `secure-coding.md`) | `cargo deny check` | exit 0 |
| RUST-DEP-01 | `Cargo.lock` committed & in sync | `cargo update --locked` (or `--frozen` build) | no change |
| RUST-ARCH-01 | Domain layer MUST NOT depend on adapter/framework crates (see `hexagonal.md`) | review / `cargo-modules` deps | no inward→outward |

> **Forbidden**: shipping implementation before its test (violates `tdd.md`); fixing a bug without a regression test first; `unwrap`/`expect`/`panic!`/indexing-that-can-panic on fallible paths in library code; `unsafe` without a `// SAFETY:` comment; ignoring a `Result` with `let _ =` (use `?` or handle it); gratuitous `.clone()` to dodge the borrow checker.

---

## 3. Verification Protocol

Run, in order, before presenting code. Fix → re-run until every gate is green.

```bash
cargo fmt --check                                            # RUST-FMT-01
cargo clippy --all-targets --all-features -- -D warnings     # RUST-LINT-01, RUST-ERR-01
cargo check --all-targets --all-features                     # RUST-TYP-01
cargo test                                                   # RUST-TST-01/02
cargo test --doc                                             # RUST-TST-03
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps               # RUST-DOC-01
cargo audit                                                  # RUST-SEC-01
cargo deny check                                             # RUST-SEC-02
```

Rust's compiler errors are precise: read the message, identify the cause (type / lifetime / ownership / trait-bound), fix idiomatically, re-run. The *why* behind each gate lives in its §0 owner; do not re-derive it here.

---

## 4. Project Structure

Idiomatic cargo layout. Architectural principles (dependency direction, ports/adapters, acyclic deps) are owned by [`hexagonal.md`](guides://hexagonal.md); below is only their Rust mapping.

```
project/
├── Cargo.toml            # manifest: deps, lints, profiles, [workspace]
├── Cargo.lock            # committed (libs included for reproducible CI)
├── deny.toml             # cargo-deny: advisories, licenses, bans (RUST-SEC-02)
├── rustfmt.toml          # formatting overrides
├── clippy.toml           # lint thresholds
├── src/
│   ├── lib.rs            # crate root; #![forbid(unsafe_code)]; module decls + re-exports
│   ├── main.rs           # binary entry (thin: parse args → call lib)
│   ├── domain/           # pure logic — no tokio/sqlx/http imports (RUST-ARCH-01)
│   ├── application/      # use cases; depends on ports (traits)
│   └── adapters/         # db/http/cli impls of the ports
├── tests/                # integration tests (black-box, see tdd.md)
├── benches/              # criterion benchmarks (see performance.md)
└── examples/             # runnable examples (compiled by CI)
```

- Modules map to files/dirs (`mod foo;` → `foo.rs` or `foo/mod.rs`). Keep visibility tight: default private, expose via `pub`/`pub(crate)`.
- Multi-crate repos use a **cargo workspace** (`[workspace] members = ["crates/*"]`) sharing one `Cargo.lock` and `target/`.
- Enforce the domain→adapter boundary by crate split (workspace) or review; the *rule* is owned by `hexagonal.md`.

---

## 5. Rust Specifics

The unique value of this guide.

### A. Ownership, borrowing & lifetimes

Move semantics by default; borrow (`&T`/`&mut T`) instead of cloning; exactly one mutable XOR many shared borrows at a time (enforced at compile time).

```rust
fn longest<'a>(a: &'a str, b: &'a str) -> &'a str {   // explicit lifetime: output ties to inputs
    if a.len() >= b.len() { a } else { b }
}

fn total(items: &[Item]) -> u64 {                     // borrow the slice; caller keeps ownership
    items.iter().map(|i| i.qty).sum()
}
```

- Prefer `&str` over `&String`, `&[T]` over `&Vec<T>` in signatures (accept the broadest borrow).
- Return owned values or use lifetimes; do **not** return a reference into a local.
- Shared ownership: `Rc<T>` (single-thread) / `Arc<T>` (threads); interior mutability: `Cell`/`RefCell` (single-thread, `RefCell` panics on aliasing violation at runtime) / `Mutex`/`RwLock` (threads). Don't reach for these to silence the borrow checker — restructure first.
- **Footgun:** fighting the borrow checker with `.clone()` everywhere. Restructure ownership (split borrows, pass indices, scope the borrow) before cloning.

### B. The type system: newtypes & enums make illegal states unrepresentable

```rust
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct UserId(String);                 // newtype: not interchangeable with any other String

#[derive(Debug, Clone, PartialEq)]
pub enum Connection {                       // closed set — `match` is exhaustive, no `default` escape
    Disconnected,
    Connecting { since: Instant },
    Open { socket: TcpStream },
}
```

- Newtype every domain primitive (IDs, money, email) to kill primitive obsession at compile time.
- Model state machines as enums with data; an unreachable arm is a compile error, not a runtime check.
- Use the **typestate pattern** (state encoded in a generic param) to make invalid method calls fail to compile.

### C. Traits & generics — zero-cost abstraction

```rust
pub trait Repository {                                   // a port (see hexagonal.md)
    fn get(&self, id: &UserId) -> Result<User, RepoError>;
}

fn load<R: Repository>(repo: &R, id: &UserId) -> Result<User, RepoError> {  // static dispatch, monomorphized
    repo.get(id)
}

fn load_dyn(repo: &dyn Repository, id: &UserId) -> Result<User, RepoError> { // dynamic dispatch (vtable)
    repo.get(id)
}
```

- Prefer generics + trait bounds (static dispatch, inlinable, zero-cost) for hot paths; use `dyn Trait` (boxed/`&dyn`) when you need heterogeneous collections or to break monomorphization bloat.
- Implement standard traits via `derive` (`Debug`, `Clone`, `PartialEq`, `Hash`, `Default`, `serde::{Serialize,Deserialize}`); implement `Display`, `From`/`TryFrom`, `Iterator` by hand where it reads naturally.
- Lean on the std trait taxonomy: `From`/`Into` for conversions, `TryFrom` for fallible ones, `AsRef`/`Borrow` for flexible APIs, `Deref` only for smart pointers (never to fake inheritance).

### D. `Result`, `Option` & the `?` operator (Rust owns these mechanics)

The *strategy* (what to model as an error, where to surface it) is owned by [`error-handling.md`](guides://error-handling.md). Rust's mechanics:

```rust
fn parse_port(s: &str) -> Result<u16, ConfigError> {
    let n: u16 = s.parse()?;                 // `?` converts the error via `From` and early-returns
    if n == 0 { return Err(ConfigError::ZeroPort); }
    Ok(n)
}

fn first_admin(users: &[User]) -> Option<&User> {
    users.iter().find(|u| u.is_admin())      // Option, not null
}
```

- **Libraries**: define a typed error enum with [`thiserror`](https://docs.rs/thiserror) `2.x` (`#[derive(Error)]`, `#[from]` for source conversion, `#[error("…")]` messages). Callers can `match` on variants.
- **Applications / binaries**: use [`anyhow`](https://docs.rs/anyhow) `1.x` `Result<T>` + `.context("…")` to add context while propagating; reserve `unwrap`/`expect` for tests, `build.rs`, and `main` startup where a panic is the right behavior.
- `?` works on `Result` and `Option` and auto-converts errors through `From`. Convert with `.ok_or(e)?`, `.map_err(…)?`, `.context(…)?`.
- **Footgun:** `unwrap`/`expect` in library code (gate `RUST-ERR-01`); array/slice indexing `v[i]` panics — prefer `.get(i)` returning `Option`.

```rust
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("port must not be zero")]
    ZeroPort,
    #[error("invalid number")]
    Parse(#[from] std::num::ParseIntError),   // `?` on parse() lands here
}
```

### E. Iterators, closures & immutability

Bindings are immutable unless `mut`. Prefer lazy iterator chains (fused, often zero-cost) over manual loops; they express intent and avoid bounds checks.

```rust
let active: Vec<&str> = users.iter()
    .filter(|u| u.is_active())
    .map(|u| u.name.as_str())
    .collect();

let total: u64 = orders.iter().map(Order::total).sum();
let (ok, bad): (Vec<_>, Vec<_>) = results.into_iter().partition(Result::is_ok);
```

- Reach for `filter_map`, `fold`/`try_fold`, `flat_map`, `scan`, `zip`, `chain` instead of hand-rolled accumulation. Use `collect::<Result<Vec<_>, _>>()` to short-circuit on the first error.
- Closures capture by reference by default; add `move` to take ownership (required for threads/`async` tasks).
- **Footgun:** wrapping a lazy iterator in `.collect::<Vec<_>>()` only to immediately iterate again — keep it lazy.

### F. Async / await

Async I/O is owned conceptually by [`parallelism.md`](guides://parallelism.md); Rust binding:

```rust
use tokio::time::{timeout, Duration};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let (a, b) = tokio::try_join!(fetch_a(), fetch_b())?;   // concurrent, fail-fast
    let u = timeout(Duration::from_secs(5), fetch_user()).await??;  // bound the wait
    Ok(())
}
```

- Pick **one** runtime (usually `tokio` with the features you use, not `["full"]` in libraries). Don't block the executor: no `std::thread::sleep`, no blocking file/CPU work on async tasks — use `tokio::task::spawn_blocking`.
- **Native `async fn` in traits is stable since Rust 1.75** — prefer it over the `async-trait` crate for new code; reach for `async-trait` only when you need `dyn`-compatible (object-safe) async trait objects.
- Concurrency: `tokio::join!`/`try_join!` for fixed sets, `JoinSet`/`FuturesUnordered` for dynamic sets, `tokio::select!` for racing. CPU-bound parallelism → `rayon` (`par_iter`), not async.
- **Footgun:** holding a `std::sync::Mutex` guard across an `.await` (deadlock risk) — use `tokio::sync::Mutex` or release before awaiting.

---

## 6. Tooling, Cargo & Dependencies

Security/supply-chain *policy* → [`secure-coding.md`](guides://secure-coding.md); versioning → [`semver.md`](guides://semver.md). Rust binding:

```bash
cargo build / cargo run            # build / run (add --release for optimized)
cargo check                        # fast type-check without codegen
cargo add <crate> --features x     # add dep (updates Cargo.toml + Cargo.lock)
cargo add --dev <crate>            # dev-dependency
cargo update                       # bump within semver ranges
cargo build --locked               # RUST-DEP-01: fail if Cargo.lock would change
cargo audit                        # RUST-SEC-01: RustSec advisory scan
cargo deny check                   # RUST-SEC-02: advisories + licenses + bans
cargo nextest run                  # faster test runner (optional, drop-in)
```

Set edition and a minimum supported Rust version; deny `unsafe` and surface lints at the manifest level:

```toml
[package]
edition = "2024"            # current stable edition (Rust 1.85+)
rust-version = "1.85"       # MSRV — checked by CI

[lints.rust]
unsafe_code = "forbid"      # RUST-SAFE-01 default posture
missing_docs = "warn"       # RUST-DOC-01

[lints.clippy]
all = { level = "warn", priority = -1 }
unwrap_used = "deny"        # RUST-ERR-01
expect_used = "deny"
panic = "deny"

[profile.release]
lto = true
codegen-units = 1
strip = true
```

- Commit `Cargo.lock` for binaries **and** libraries (reproducible CI); let cargo resolve the graph from constrained direct deps.
- Audit the dependency surface: minimize crates, prefer well-maintained ones, run `cargo tree` to spot duplicate/transitive bloat. Default to safe, pure-Rust crates; FFI/`*-sys` crates add build and `unsafe` surface — justify them.

---

## 7. Quick Reference

```bash
cargo build                                                  # build
cargo test && cargo test --doc                              # test + doctests
cargo clippy --all-targets --all-features -- -D warnings    # lint
cargo fmt                                                    # format
cargo run                                                    # run
cargo doc --no-deps --open                                  # docs
```

```rust
let v = result?;                       // propagate error (From-converted)
let v = option.ok_or(Err::Missing)?;   // Option → Result, then propagate
let v = slice.get(i);                  // Option, never panics (vs slice[i])
match state { S::A => …, S::B(x) => …, } // exhaustive — compiler enforces all arms
```

---

## 8. Unsafe Discipline

`unsafe` does not turn off the borrow checker — it unlocks five operations (deref raw pointers, call `unsafe` fns, access `static mut`/union fields, implement `unsafe` traits). The compiler can no longer verify memory/thread safety; **you** uphold the invariants.

- Keep `#![forbid(unsafe_code)]` at the crate root unless a measured need exists. When needed, scope `unsafe` to the smallest block and wrap it in a safe, well-documented API.
- Every `unsafe` block MUST carry a `// SAFETY:` comment stating the invariant being upheld and why it holds (gate `RUST-SAFE-01`).
- Prefer a vetted safe abstraction (e.g. `bytemuck`, `zerocopy`) over hand-written `unsafe`. Validate `unsafe` code with `cargo +nightly miri test` where feasible.

```rust
// SAFETY: `ptr` is non-null and aligned (just allocated above), and we hold
// the only reference, so no aliasing or use-after-free is possible here.
let value = unsafe { *ptr };
```

---

## 9. Deployment Checklist

Generated from §2 — one box per requirement ID.

- [ ] RUST-FMT-01 — `cargo fmt --check` clean
- [ ] RUST-LINT-01 — `cargo clippy --all-targets --all-features -- -D warnings` clean
- [ ] RUST-TYP-01 — `cargo check --all-targets --all-features` clean
- [ ] RUST-TST-01/02 — tests pass, bugs have regression tests, 0 ignored
- [ ] RUST-TST-03 — doctests pass
- [ ] RUST-ERR-01 — no `unwrap`/`expect`/`panic!` on fallible paths in `src/`
- [ ] RUST-SAFE-01 — `unsafe` forbidden, or every block has a `// SAFETY:` proof
- [ ] RUST-DOC-01 — public items documented, `cargo doc` clean under `-D warnings`
- [ ] RUST-SEC-01/02 — `cargo audit` and `cargo deny check` clean
- [ ] RUST-DEP-01 — `Cargo.lock` committed and in sync (`--locked`)
- [ ] RUST-ARCH-01 — domain layer free of adapter/framework deps
- [ ] Agent ran every §3 command and documented any fixes

---
**End of Rust Guidelines**
