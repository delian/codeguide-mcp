# Go Development Guidelines
Mandatory coding standards for Go: idiomatic, test-covered, error-as-values, concurrent-safe. Go 1.23+, go modules, go test, gofmt, go vet, golangci-lint, govulncheck.

---
name: go
title: Go Development Guidelines
version: 2.0
last_reviewed: 2026-06-05
kind: language
tools: [go@1.23, golangci-lint, go-vet, govulncheck, gofumpt]
requires:
  - tdd
  - secure-coding
  - error-handling
recommends:
  - hexagonal
  - microservices
  - parallelism
  - observability
  - comments
provides:
  - idiomatic-go
  - error-values
  - goroutines-channels
  - go-interfaces
  - go-modules
---

> 🧭 Authored per [`CONVENTIONS.md`](guides://CONVENTIONS.md): shared concerns are referenced, not restated. This guide covers only what is unique to Go.

---

## 0. Prerequisites & References

Fetch and apply these **before** generating Go code. Their rules are assumed below and not repeated.

> 📎 **REQUIRED — fetch & apply first:**
> - [`tdd.md`](guides://tdd.md) — test-first, Red-Green-Refactor, regression-test-before-fix, coverage. *(Go binding: runner is `go test ./...`; table-driven tests; `-race` mandatory.)*
> - [`secure-coding.md`](guides://secure-coding.md) — supply chain, secrets, CVE policy. *(Go binding: `govulncheck`, `go mod verify`.)*
> - [`error-handling.md`](guides://error-handling.md) — general error strategy & propagation. *(Go owns errors-as-values; see §6.)*

> 📎 **RECOMMENDED — fetch when the task touches them:**
> - [`hexagonal.md`](guides://hexagonal.md) · [`microservices.md`](guides://microservices.md) — service structure, ports/adapters, dependency direction *(binding: `internal/`, `cmd/`, ports as interfaces — see §4)*.
> - [`parallelism.md`](guides://parallelism.md) — concurrency model *(binding: goroutines, channels, `context`, `errgroup` — see §5)*.
> - [`observability.md`](guides://observability.md) — logging/metrics/tracing *(binding: `log/slog`, OpenTelemetry)*.
> - [`comments.md`](guides://comments.md) — doc-comment policy *(binding: godoc, doc starts with the symbol name)*.

> 📎 **SEE ALSO:** [`designpatterns.md`](guides://designpatterns.md) · [`code-review.md`](guides://code-review.md) · [`ci-cd.md`](guides://ci-cd.md) · [`grpc.md`](guides://grpc.md) · [`rest.md`](guides://rest.md)

---

## 1. Core Philosophies: GO-FIRST

Go-specific principles only. TDD, security, error strategy, and architecture come from §0.

- **G**ofmt non-negotiable: code is `gofmt`/`gofumpt`-clean and `goimports`-ordered; there is one true format, no debate.
- **O**bvious over clever: clear control flow, no magic, no `init()` side effects, no global mutable state; the [Go proverbs](https://go-proverbs.github.io/) hold ("clear is better than clever").
- **F**ail with values: errors are returned, not thrown; `panic` only in `main`/truly-unrecoverable paths; every error is checked (see §6).
- **I**nterfaces small & implicit: 1–3 methods, defined by the *consumer*; **accept interfaces, return structs**.
- **R**eturn-time concurrency safety: every concurrent path is `go test -race`-clean; goroutines have a clear owner and lifetime tied to a `context` (see §5).
- **S**imple dependencies: standard library first; explicit dependency injection (no DI frameworks); modules pinned and verified.
- **T**ested by table: table-driven `_test.go` beside the code, mocks via small interfaces, examples that run as tests.

**Verified Code**: Agent-generated Go MUST pass every gate in §2 before delivery.

---

## 2. Requirements (MANDATORY, auditable)

RFC-2119 keywords. IDs `GO-<TOPIC>-<NN>`. Each row has a binary gate; rows binding a shared rule cite its owner.

| ID | Requirement | Verify | Gate |
|----|-------------|--------|------|
| GO-TST-01 | Every feature MUST be test-first (see `tdd.md`) | `go test ./...` | exit 0, 0 skips |
| GO-TST-02 | Each bug MUST get a regression test before the fix (see `tdd.md`) | `go test ./...` | failing→passing |
| GO-TST-03 | Concurrent code MUST be race-clean | `go test -race ./...` | exit 0, no data races |
| GO-FMT-01 | Code MUST be gofmt-formatted | `gofmt -l .` (or `gofumpt -l .`) | no output |
| GO-VET-01 | `go vet` MUST pass clean | `go vet ./...` | exit 0 |
| GO-LINT-01 | Linter MUST pass clean | `golangci-lint run` | exit 0 |
| GO-BUILD-01 | All packages MUST compile | `go build ./...` | exit 0 |
| GO-ERR-01 | Errors MUST be checked & wrapped with `%w` (see `error-handling.md`) | `golangci-lint run` (errcheck, errorlint) | 0 findings |
| GO-DOC-01 | Exported symbols MUST have doc comments (see `comments.md`) | `golangci-lint run` (revive/godot) | 0 findings |
| GO-SEC-01 | 0 known CVEs in module graph (see `secure-coding.md`) | `govulncheck ./...` | 0 vulnerabilities |
| GO-DEP-01 | Module graph MUST be tidy & verified (see `secure-coding.md`) | `go mod tidy -diff && go mod verify` | no diff, verified |
| GO-ARCH-01 | Domain imports no adapter/framework code (see `hexagonal.md`) | review / depguard (golangci-lint) | no inward→outward |

> **Forbidden**: shipping implementation before its test (violates `tdd.md`), fixing a bug without a regression test first, ignoring an error without `_ =` + comment, `panic` in library code, `init()` for side effects, global mutable application state, exported structs with mutable public fields used as config (use functional options), `fmt.Println` for logging (use `slog`), or starting a goroutine with no defined owner/lifetime.

---

## 3. Verification Protocol

Run, in order, before presenting code. Fix → re-run until every gate is green.

```bash
gofmt -l .                  # GO-FMT-01  (gofumpt -l . for stricter)
go build ./...              # GO-BUILD-01
go vet ./...                # GO-VET-01
golangci-lint run           # GO-LINT-01, GO-ERR-01, GO-DOC-01, GO-ARCH-01 (depguard)
go test -race ./...         # GO-TST-01/02/03
govulncheck ./...           # GO-SEC-01
go mod tidy -diff && go mod verify   # GO-DEP-01
```

The *why* behind each gate lives in its §0 owner; do not re-derive it here.

---

## 4. Project Structure

Idiomatic Go layout. Architectural *principles* (ports/adapters, dependency direction, acyclic deps) are owned by [`hexagonal.md`](guides://hexagonal.md) and [`microservices.md`](guides://microservices.md); below is only their Go mapping.

```
project/
├── cmd/<app>/main.go    # entry point — minimal: wire deps, call internal
├── internal/            # private code; not importable externally
│   ├── <domain>/        # group by feature/domain, not by type
│   │   ├── user.go              # domain entity + behaviour (pure, no IO)
│   │   ├── service.go           # use case; depends on port interfaces
│   │   ├── repository.go        # PORT: interface defined by the consumer
│   │   └── service_test.go      # tests beside code (see tdd.md)
│   └── adapter/         # port implementations: postgres, http, grpc, redis…
├── pkg/                 # ONLY genuinely reusable libraries for external import
├── go.mod / go.sum      # module manifest + checksums (committed)
└── go.work              # optional: multi-module workspace
```

- Group by domain/feature, not by `models/`, `services/`, `handlers/`.
- `internal/` for everything app-private; `pkg/` only for code you'd publish.
- Ports (interfaces) are declared where they are **used** (consumer side), not where implemented — this keeps the domain free of adapter imports (GO-ARCH-01). Enforce with golangci-lint `depguard`.
- No circular package dependencies (compiler enforces; break cycles with an interface).
- `cmd/<app>/main.go` is thin: construct the dependency graph explicitly (no DI framework) and start.

---

## 5. Concurrency (Go binding)

Concurrency *policy* (when to parallelize, backpressure, cancellation semantics) is owned by [`parallelism.md`](guides://parallelism.md). Go owns the *mechanism*: goroutines, channels, `context`, and the memory model.

- **"Don't communicate by sharing memory; share memory by communicating."** Prefer channels to pass ownership of data; use `sync.Mutex`/`atomic` only for small, local shared state.
- **Every goroutine has an owner and an exit path.** Never start a goroutine you cannot stop. Lifetime is tied to a `context.Context`; the leaf selects on `<-ctx.Done()`.
- **`context.Context` is the first parameter** of any blocking/IO call and is threaded down the stack — never stored in a struct, never `nil` (use `context.TODO()`).
- **Channels:** the sender closes, never the receiver; a closed channel signals "no more values"; `select` with `ctx.Done()` for cancellation; buffered channels only when you can justify the size.
- **`errgroup`** for fan-out where any error should cancel siblings: `g, ctx := errgroup.WithContext(ctx)`, bound with `g.SetLimit(n)`, collect via `g.Wait()`.

```go
func ProcessAll(ctx context.Context, items []Item) error {
	g, ctx := errgroup.WithContext(ctx)
	g.SetLimit(runtime.GOMAXPROCS(0)) // bounded concurrency
	for _, it := range items {
		it := it // pre-1.22 loop-var capture; unnecessary on Go 1.22+
		g.Go(func() error { return process(ctx, it) })
	}
	return g.Wait() // first error cancels ctx; the rest unwind
}
```

- **Footguns:** loop-variable capture (fixed in Go 1.22+, but be explicit when targeting older); leaking goroutines blocked on a channel no one drains; data races on shared maps (always `go test -race`, GO-TST-03); `WaitGroup.Add` inside the goroutine (add before `go`).

---

## 6. Errors as Values (Go binding — Go owns this strongly)

General strategy (when to wrap, where to handle, fail-fast vs. recover) is owned by [`error-handling.md`](guides://error-handling.md). Go's **error-value binding** is canonical here and the value of this guide.

- **Check every error at the call site.** `if err != nil { return ... }` is the norm; ignore only with an explicit `_ = f()` plus a comment justifying why.
- **Wrap with `%w` to preserve the chain**, adding context as you go up the stack: `fmt.Errorf("get user %s: %w", id, err)`. Wrap once per layer; do not double-log-and-wrap.
- **Inspect with `errors.Is` / `errors.As`**, never `==` on wrapped errors and never string-matching:

```go
var ErrNotFound = errors.New("not found")       // sentinel for expected conditions

type ValidationError struct{ Field, Msg string } // typed error for rich context
func (e *ValidationError) Error() string { return e.Field + ": " + e.Msg }

// caller:
if errors.Is(err, ErrNotFound) { /* expected: 404 */ }
var ve *ValidationError
if errors.As(err, &ve) { /* read ve.Field */ }
```

- **`errors.Join`** to aggregate multiple non-fatal failures (e.g. validating many fields) into one error.
- **Sentinels vs. typed errors:** use a sentinel (`errors.New`) when callers only need identity; a typed error when they need data. Export them so callers can match.
- **No exceptions:** `panic`/`recover` is not control flow. `panic` only for programmer bugs or unrecoverable startup failures; a top-level `recover` in a server's request handler may convert a panic into a 500 + logged stack, nothing more.
- **`defer` for cleanup**, and capture deferred-close errors when they matter: `defer func() { err = errors.Join(err, f.Close()) }()` with a named return.

---

## 7. Go Specifics

The unique value of this guide.

### A. Interfaces — small, implicit, consumer-defined
Interfaces are satisfied structurally (no `implements`). Keep them tiny (`io.Reader` is one method) and declare them where consumed. **Accept interfaces, return concrete structs** so callers stay decoupled while constructors keep full type info.

```go
// defined in the package that USES it, not the one that implements it
type UserStore interface {
	ByID(ctx context.Context, id string) (*User, error)
}

func NewService(s UserStore) *Service { return &Service{store: s} } // accept iface, return struct
```
Avoid premature interfaces ("one implementation" → use the struct); avoid large "god" interfaces.

### B. Functional options for configuration
Backward-compatible, self-documenting optional config without breaking call sites or exporting mutable fields.

```go
type Server struct{ addr string; readTimeout time.Duration }
type Option func(*Server)

func WithAddr(a string) Option        { return func(s *Server) { s.addr = a } }
func WithReadTimeout(d time.Duration) Option { return func(s *Server) { s.readTimeout = d } }

func NewServer(opts ...Option) *Server {
	s := &Server{addr: ":8080", readTimeout: 10 * time.Second} // defaults
	for _, opt := range opts {
		opt(s)
	}
	return s
}
```

### C. Generics (Go 1.18+)
Use type parameters to remove `interface{}`+cast boilerplate for containers and algorithms — not as a substitute for interfaces. Constrain with `comparable`, `golang.org/x/exp/constraints`, or a custom constraint interface.

```go
func Map[T, U any](in []T, f func(T) U) []U {
	out := make([]U, len(in))
	for i, v := range in {
		out[i] = f(v)
	}
	return out
}
```
Reach for generics when the only thing varying is the type; prefer a plain interface when behaviour varies.

### D. Range-over-func iterators (Go 1.23+)
Custom iteration via `iter.Seq[T]` / `iter.Seq2[K,V]`, consumable with `for v := range seq`.

```go
func (s *Service) All(ctx context.Context) iter.Seq[*User] {
	return func(yield func(*User) bool) {
		for _, u := range s.cache {
			if !yield(u) { return } // consumer broke out
		}
	}
}
```

### E. Standard-library first
Prefer the stdlib: `net/http` (with `http.ServeMux` path patterns, Go 1.22+) over a framework when it suffices; `log/slog` for structured logging (see `observability.md`); `html/template` (auto-escaping) for HTML, never `text/template`; `embed` for static assets; `context` for cancellation. Add a dependency only when the stdlib genuinely falls short.

### F. Footguns
- Nil map writes panic — `make(map[...]...)` before writing.
- `append` may alias backing arrays — copy when you must not mutate the source.
- A non-nil interface holding a nil pointer is `!= nil` — return concrete `nil` or a typed sentinel, not a typed-nil pointer as `error`.
- Deferred calls run at function return, not block exit — beware deferring in a loop.
- Slices/maps passed to a struct are shared, not copied — clone if the caller may mutate.

---

## 8. Tooling & Dependencies

Security/supply-chain *policy* → [`secure-coding.md`](guides://secure-coding.md); versioning → [`semver.md`](guides://semver.md). Go binding:

```bash
go mod download           # install from go.sum (reproducible)
go get example.com/pkg@v1.2.3   # add/pin a dependency
go get -u ./... && go mod tidy  # update to latest, prune unused
go mod tidy -diff         # GO-DEP-01: tidy is a no-op (CI-safe)
go mod verify             # GO-DEP-01: checksums match go.sum
govulncheck ./...         # GO-SEC-01: CVE scan (call-graph aware)
```

- Commit `go.mod` **and** `go.sum`. Pin direct deps; let the toolchain resolve the rest.
- Track build/dev tools with the **`tool` directive in `go.mod`** (Go 1.24+) and run via `go tool <name>`; the legacy `tools.go` `//go:build tools` pattern is the fallback on older toolchains.
- `golangci-lint` is the meta-linter (it runs `errcheck`, `errorlint`, `govet`, `staticcheck`, `revive`, `depguard`, etc.); configure in `.golangci.yml` and keep it under version control.

---

## 9. Quick Reference

```bash
go build ./...                      # build
go test -race ./...                 # test (race-clean)
go vet ./... && golangci-lint run   # lint
gofmt -w . # or gofumpt -w .        # format
go doc ./...                        # docs
go run ./cmd/<app>                  # run
```

---

## 10. Deployment Checklist

Generated from §2 — one box per requirement ID.

- [ ] GO-FMT-01 — `gofmt -l .` clean
- [ ] GO-BUILD-01 — `go build ./...` exit 0
- [ ] GO-VET-01 — `go vet ./...` clean
- [ ] GO-LINT-01 — `golangci-lint run` clean
- [ ] GO-TST-01/02/03 — tests pass, bugs have regression tests, `-race` clean
- [ ] GO-ERR-01 — all errors checked & wrapped with `%w`
- [ ] GO-DOC-01 — exported symbols documented (godoc)
- [ ] GO-SEC-01 — `govulncheck` 0 vulnerabilities
- [ ] GO-DEP-01 — `go mod tidy -diff` no diff, `go mod verify` ok, `go.sum` committed
- [ ] GO-ARCH-01 — domain layer free of adapter/framework imports
- [ ] Agent ran every §3 command and documented any fixes

---
**End of Go Guidelines**
