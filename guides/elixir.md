# Elixir Development Guidelines
Mandatory coding standards for Elixir: functional, immutable, fault-tolerant OTP on the BEAM. Elixir 1.18+, mix, ExUnit, Dialyzer, Credo.

---
name: elixir
title: Elixir Development Guidelines
version: 2.0
last_reviewed: 2026-06-05
kind: language
tools: [elixir@1.18, mix, ex_unit, dialyzer, credo, mix_audit]
requires:
  - tdd
  - secure-coding
  - error-handling
recommends:
  - hexagonal
  - observability
  - parallelism
  - comments
provides:
  - otp
  - let-it-crash
  - pattern-matching
  - beam-concurrency
  - tagged-tuples
---

> 🧭 Authored per [`CONVENTIONS.md`](guides://CONVENTIONS.md): shared concerns are referenced, not restated. This guide covers only what is unique to Elixir and the BEAM/OTP runtime.

---

## 0. Prerequisites & References

Fetch and apply these **before** generating Elixir code. Their rules are assumed below and not repeated.

> 📎 **REQUIRED — fetch & apply first:**
> - [`tdd.md`](guides://tdd.md) — test-first, Red-Green-Refactor, regression-test-before-fix, coverage. *(Elixir binding: runner is `mix test`; doctests double as executable docs.)*
> - [`secure-coding.md`](guides://secure-coding.md) — supply chain, secrets, CVE policy. *(Elixir binding: `mix deps.audit`, `mix hex.audit`.)*
> - [`error-handling.md`](guides://error-handling.md) — error vs exception model, Result types, fail-fast. *(Elixir binding: tagged tuples `{:ok, _}`/`{:error, _}` for expected failures; "let it crash" + supervisors for the unexpected — see §6–§7.)*

> 📎 **RECOMMENDED — fetch when the task touches them:**
> - [`parallelism.md`](guides://parallelism.md) — concurrency model, races, structured concurrency. *(Elixir binding: BEAM processes, GenServer, Task, supervision — see §7.)*
> - [`hexagonal.md`](guides://hexagonal.md) — layering, ports/adapters, dependency inversion. *(Elixir binding: contexts + behaviours as ports.)*
> - [`observability.md`](guides://observability.md) — metrics, tracing, health. *(Elixir binding: `:telemetry` + `Telemetry.Metrics` — see §10.)*
> - [`comments.md`](guides://comments.md) — doc policy *(binding: `@moduledoc`/`@doc`, ExDoc).*

> 📎 **SEE ALSO:** [`phoenix.md`](guides://phoenix.md) *(the Phoenix framework builds on this guide)* · [`designpatterns.md`](guides://designpatterns.md) · [`env-config.md`](guides://env-config.md) · [`microservices.md`](guides://microservices.md) · [`code-review.md`](guides://code-review.md)

---

## 1. Core Philosophies: ELIXIR-FIRST

Elixir-specific principles only. TDD, security, error strategy, and architecture come from §0.

- **E**xplicit results: model expected outcomes as tagged tuples (`{:ok, _}`/`{:error, _}`); reserve raising/crashing for the unexpected.
- **L**et it crash: do not defensively guard against bugs — isolate them in processes and let supervisors restart to a known-good state (policy: `error-handling.md`).
- **I**mmutable: all data is immutable; transform with new values, never mutate in place.
- **X**-process concurrency: model concurrency with lightweight BEAM processes and OTP behaviours, not shared memory or locks (policy: `parallelism.md`).
- **I**diomatic flow: pattern matching, multiple function clauses, guards, the pipe operator, and `with` over nested conditionals.
- **R**igor: `mix format`, `mix credo --strict`, and `mix dialyzer` (with `@spec`s) all green before delivery.

**Verified Code**: Agent-generated Elixir MUST pass every gate in §2 before delivery.

---

## 2. Requirements (MANDATORY, auditable)

RFC-2119 keywords. IDs `EX-<TOPIC>-<NN>`. Each row has a binary gate; rows binding a shared rule cite its owner.

| ID | Requirement | Verify | Gate |
|----|-------------|--------|------|
| EX-TST-01 | Every feature MUST be test-first (see `tdd.md`) | `mix test` | exit 0, 0 skips/excludes |
| EX-TST-02 | Each bug MUST get a regression test before the fix (see `tdd.md`) | `mix test` | failing→passing |
| EX-TST-03 | Public functions with `@doc` examples MUST run as doctests | `mix test` | doctests pass |
| EX-FMT-01 | Code MUST be formatted | `mix format --check-formatted` | no diff |
| EX-LINT-01 | Credo MUST pass strict | `mix credo --strict` | exit 0, 0 issues |
| EX-TYP-01 | Public functions MUST have `@spec`; Dialyzer clean | `mix dialyzer` | exit 0, 0 warnings |
| EX-ERR-01 | Expected failures MUST use tagged tuples; no defensive guarding of bugs (see `error-handling.md`) | review / Credo | tagged tuples, let-it-crash |
| EX-OTP-01 | Stateful/long-lived processes MUST run under a supervisor (see `parallelism.md`) | review | no orphan `spawn`/`start_link` |
| EX-DOC-01 | Public modules/functions MUST have `@moduledoc`/`@doc` (see `comments.md`) | `mix docs` / Credo | builds clean, no missing |
| EX-SEC-01 | 0 known CVEs in deps (see `secure-coding.md`) | `mix deps.audit` | 0 vulnerabilities |
| EX-SEC-02 | No retired Hex packages (see `secure-coding.md`) | `mix hex.audit` | clean |
| EX-DEP-01 | Lockfile in sync & verified | `mix deps.get --check-locked` | in sync |
| EX-ARCH-01 | Context/boundary deps respected (see `hexagonal.md`) | review / Boundary | no inward→outward deps |

> **Forbidden**: shipping implementation before its test (violates `tdd.md`), fixing a bug without a regression test first, `rescue`-ing or `try`-wrapping code to hide a crash that a supervisor should handle, unsupervised long-lived processes, leaving `IO.inspect`/`dbg` in delivered code, or functions without `@spec` on public APIs.

---

## 3. Verification Protocol

Run, in order, before presenting code. Fix → re-run until every gate is green.

```bash
mix format --check-formatted    # EX-FMT-01
mix credo --strict              # EX-LINT-01
mix dialyzer                    # EX-TYP-01  (Credo does NOT type-check)
mix test                        # EX-TST-01/02/03 (includes doctests)
mix deps.audit                  # EX-SEC-01
mix hex.audit                   # EX-SEC-02
mix deps.get --check-locked     # EX-DEP-01
```

The *why* behind each gate lives in its §0 owner; do not re-derive it here.

---

## 4. Project Structure

Idiomatic mix layout. Architectural principles (dependency direction, ports/adapters, acyclic deps) are owned by [`hexagonal.md`](guides://hexagonal.md); below is only their Elixir mapping. Group by **context** (bounded domain), not by technical type.

```
my_app/
├── lib/
│   └── my_app/
│       ├── application.ex     # OTP Application + root supervision tree
│       ├── accounts/          # context (bounded domain): pure logic + behaviours = ports
│       │   ├── accounts.ex    # context module = public API boundary
│       │   └── user.ex
│       ├── orders/            # another context
│       └── workers/           # GenServers / Task / Oban workers (adapters)
├── test/
│   ├── my_app/                # mirrors lib/ (see tdd.md)
│   └── test_helper.exs
├── config/                    # config.exs, runtime.exs (secrets → runtime, see env-config.md)
├── .credo.exs
├── mix.exs                    # deps, releases, dialyzer/credo config
└── mix.lock                   # committed lockfile
```

- A context module is the only public entry point to its domain; other contexts call it, never its internals.
- Enforce boundaries with the `boundary` library (compile-time checks) for EX-ARCH-01.

---

## 5. Functional Core: Immutability, Pattern Matching, Pipes

The unique value of this guide starts here.

### A. Immutability & pattern matching
Data is immutable; "updating" returns a new term. Destructure and branch by matching in function heads with multiple clauses and guards instead of conditionals.

```elixir
# Multiple clauses + guards replace if/else chains
def classify(n) when is_integer(n) and n > 0, do: :positive
def classify(0), do: :zero
def classify(n) when is_integer(n), do: :negative

# Destructure structs/maps in the head; pin (^) matches an existing value
def greet(%User{role: :admin, name: name}), do: "Welcome, admin #{name}"
def greet(%User{name: name}), do: "Hello, #{name}"

# Immutable "update" — %{m | k: v} returns a new map (key must already exist)
new_state = %{state | processed: state.processed + 1}
```

### B. Pipe operator
Thread a value through transformations left-to-right. Pipe only chains of ≥2 calls; use the capture operator `&`, never a bare anonymous fn, on the right.

```elixir
users
|> Enum.filter(&active?/1)
|> Enum.map(&format/1)
|> Enum.sort_by(& &1.name)

result = process(data)          # single call: NO pipe
data |> process() |> dbg()      # debug with dbg/1, not chained tap/IO.inspect (strip before delivery)
```

### C. `with` for happy-path chaining
Compose fallible steps that each return tagged tuples; the first non-match short-circuits to `else`. Flattens what would be nested `case`.

```elixir
def create_order(params) do
  with {:ok, user}  <- get_user(params.user_id),
       {:ok, items} <- validate_items(params.items),
       {:ok, order} <- Orders.insert(user, items) do
    {:ok, order}
  else
    {:error, :user_not_found} -> {:error, :user_not_found}
    {:error, reason}          -> {:error, reason}
  end
end
```

---

## 6. Error Handling: Tagged Tuples + Let It Crash

Policy (error vs exception, fail-fast, recover-where) is owned by [`error-handling.md`](guides://error-handling.md). Elixir binding:

- **Expected, recoverable** outcomes → tagged tuples. Function returning a value: `{:ok, value} | {:error, reason}`. The `reason` SHOULD be a structured atom or tuple (`:not_found`, `{:invalid, field}`), not a bare string.
- **`!` (bang) variants** raise on failure for the "this should always succeed" call site (`fetch!`, `get_user!`). Provide a non-bang tagged-tuple variant alongside.
- **Unexpected / programmer error** → do **not** rescue it. Let the process crash; its supervisor restarts a clean state ("let it crash"). Reserve `try/rescue` for genuinely exceptional foreign failures (e.g. a library that raises) at a boundary, not for normal control flow.
- Define exception structs with `defexception` for raise-paths; keep them few and meaningful.

```elixir
@spec find_user(binary()) :: {:ok, User.t()} | {:error, :not_found}
def find_user(id) do
  case Repo.get(User, id) do
    nil  -> {:error, :not_found}
    user -> {:ok, user}
  end
end

@spec find_user!(binary()) :: User.t()   # bang variant for must-succeed call sites
def find_user!(id), do: Repo.get!(User, id)
```

---

## 7. OTP & BEAM Concurrency

Concurrency *policy* (race conditions, structured concurrency, deadlock avoidance) is owned by [`parallelism.md`](guides://parallelism.md). The BEAM removes shared-memory hazards: processes share **nothing**, communicate by **message passing**, and are scheduled preemptively. This section owns the OTP idioms.

### A. GenServer — encapsulated stateful process
Separate the **client API** (runs in the caller) from the **server callbacks** (run in the process). Annotate every callback with `@impl true`.

```elixir
defmodule MyApp.Workers.Counter do
  use GenServer

  # Client API
  def start_link(opts), do: GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  @spec bump() :: :ok
  def bump, do: GenServer.cast(__MODULE__, :bump)       # async, no reply
  @spec count() :: non_neg_integer()
  def count, do: GenServer.call(__MODULE__, :count)     # sync, waits for reply

  # Server callbacks
  @impl true
  def init(_opts), do: {:ok, 0, {:continue, :warm}}     # defer expensive init off the supervisor
  @impl true
  def handle_continue(:warm, state), do: {:noreply, state}
  @impl true
  def handle_cast(:bump, count), do: {:noreply, count + 1}
  @impl true
  def handle_call(:count, _from, count), do: {:reply, count, count}
end
```

Rules: keep `init/1` fast (use `{:continue, _}` for heavy work); never put a long blocking call inside a `handle_call` that others wait on; prefer GenServer over `Agent` once logic or `handle_info` is needed.

### B. Supervisors — the recovery strategy
Every long-lived/stateful process MUST be started under a supervisor (EX-OTP-01), normally the application tree.

```elixir
defmodule MyApp.Application do
  use Application

  @impl true
  def start(_type, _args) do
    children = [
      {Registry, keys: :unique, name: MyApp.Registry},
      {DynamicSupervisor, strategy: :one_for_one, name: MyApp.DynSup},
      {Task.Supervisor, name: MyApp.TaskSup},
      MyApp.Workers.Counter
    ]
    Supervisor.start_link(children, strategy: :one_for_one, name: MyApp.Supervisor,
      max_restarts: 5, max_seconds: 10)
  end
end
```

Restart strategies: `:one_for_one` (independent children), `:rest_for_one` (restart the failed child + those started after it), `:one_for_all` (tightly coupled — restart all). Always set `max_restarts`/`max_seconds` so a crash loop escalates instead of spinning.

### C. Dynamic processes & Tasks
- **DynamicSupervisor + Registry**: per-tenant/per-session processes started on demand and looked up by key via `{:via, Registry, {MyApp.Registry, key}}`.
- **`Task.Supervisor`**: concurrent work with results — `async_nolink/2` + `Task.await_many/2`; fire-and-forget — `start_child/2`. Use `Task.yield/2 || Task.shutdown/1` to bound latency instead of unbounded `await`.

```elixir
results =
  items
  |> Enum.map(&Task.Supervisor.async_nolink(MyApp.TaskSup, fn -> work(&1) end))
  |> Task.await_many(5_000)
```

---

## 8. Typespecs, Dialyzer & Credo

Elixir is dynamically typed; rigor comes from typespecs verified by Dialyzer's success typing plus Credo's static analysis.

```elixir
@type t :: %__MODULE__{id: binary(), email: String.t(), role: atom()}

@spec normalize(String.t()) :: {:ok, String.t()} | {:error, :invalid}
def normalize(email) when is_binary(email), do: ...
```

- Put `@spec` on **every public function** (EX-TYP-01) and `@type t` on public structs; private helpers as needed for inference.
- Run `mix dialyzer` (via `:dialyxir`); cache the PLT in CI. It finds contract violations, impossible patterns, and unreachable clauses — it is **not** a substitute for Credo or tests.
- `mix credo --strict` enforces consistency, complexity limits, and readability (e.g. missing `@moduledoc`). Configure in `.credo.exs`; do not silence checks to pass the gate.

---

## 9. Testing with ExUnit & Doctests

Test *policy* (Red-Green-Refactor, pyramid, doubles, coverage) is owned by [`tdd.md`](guides://tdd.md). Elixir binding:

```elixir
defmodule MyApp.EmailTest do
  use ExUnit.Case, async: true          # async: true for isolated, side-effect-free tests
  doctest MyApp.Email                   # runs the @doc examples as tests (EX-TST-03)

  describe "normalize/1" do
    test "lowercases a valid address" do
      assert {:ok, "a@b.com"} = MyApp.Email.normalize("A@B.com")
    end

    test "rejects input without @" do
      assert {:error, :invalid} = MyApp.Email.normalize("nope")
    end
  end
end
```

- **Doctests**: executable `@doc` examples keep docs honest and count as tests — wire them with `doctest Module`.
- **`async: true`** wherever tests don't share mutable global state, for parallel speed.
- **Mox** for behaviour-based mocks (`Mox.defmock(.., for: Behaviour)`, `setup :verify_on_exit!`) — mock the *behaviour/port*, never a concrete module.
- **StreamData** (`ExUnitProperties`) for property-based tests of pure functions and invariants.
- Coverage via `mix test --cover` (or `excoveralls`) at the `tdd.md` gate.

---

## 10. Observability & Ecosystem

### A. Telemetry (observability binding)
Instrumentation policy (metrics, tracing, SLIs) is owned by [`observability.md`](guides://observability.md). On the BEAM the standard is `:telemetry`: libraries emit events; you attach handlers and aggregate with `Telemetry.Metrics` + `telemetry_poller`. Emit custom events with `:telemetry.execute([:my_app, :orders, :created], %{duration: d}, metadata)` and export to your backend (e.g. OpenTelemetry/Prometheus per the owner).

### B. Ecosystem awareness (reach for, don't reinvent)
- **Oban** — persistent, DB-backed background jobs that survive restarts; prefer over bespoke GenServer queues for durable work.
- **Broadway / GenStage** — concurrent, back-pressured data pipelines (SQS/Kafka/RabbitMQ).
- **Phoenix / Ecto** — web and DB layers; their own conventions are out of scope here. Keep web/persistence concerns in adapter modules at the context edge (see `hexagonal.md`).
- **Nx / Bumblebee** — numerical computing and pre-trained ML models on the BEAM.

### C. Releases & runtime config
Ship with `mix release` (self-contained artifact, no build tools on target). Read all secrets/env-specific values in `config/runtime.exs` at boot — never at compile time — and `raise` on missing required vars (config policy: [`env-config.md`](guides://env-config.md)). Run migrations via a release `eval` module, not `mix`, in production.

---

## 11. Tooling & Dependencies

Security/supply-chain *policy* → [`secure-coding.md`](guides://secure-coding.md). Elixir binding:

```bash
mix deps.get                    # install from mix.lock (reproducible)
mix deps.get --check-locked     # EX-DEP-01: lockfile in sync
mix hex.outdated                # see available upgrades
mix deps.update --all           # update + relock
mix deps.audit                  # EX-SEC-01: CVE scan (mix_audit)
mix hex.audit                   # EX-SEC-02: retired-package check
```

Commit `mix.lock`. Add `{:mix_audit, "~> 2.1", only: [:dev, :test], runtime: false}`, `{:credo, .., runtime: false}`, and `{:dialyxir, .., runtime: false}` as dev/test deps.

---

## 12. Quick Reference

```bash
mix deps.get                         # setup
mix test                             # test (+ doctests)
mix credo --strict                   # lint
mix dialyzer                         # type/contract check
mix format                           # format
mix release                          # build deployable artifact
iex -S mix                           # REPL with project loaded
```

---

## 13. Deployment Checklist

Generated from §2 — one box per requirement ID.

- [ ] EX-FMT-01 — `mix format --check-formatted` clean
- [ ] EX-LINT-01 — `mix credo --strict` clean
- [ ] EX-TYP-01 — `@spec` on public APIs, `mix dialyzer` clean
- [ ] EX-TST-01/02/03 — tests pass, bugs have regression tests, doctests run
- [ ] EX-ERR-01 — tagged tuples for expected failures, let-it-crash for the rest
- [ ] EX-OTP-01 — all long-lived processes under a supervisor
- [ ] EX-DOC-01 — public modules/functions documented, `mix docs` clean
- [ ] EX-SEC-01/02 — `mix deps.audit` 0 CVEs, `mix hex.audit` clean
- [ ] EX-DEP-01 — `mix.lock` in sync, committed
- [ ] EX-ARCH-01 — context boundaries respected, no leaking internals
- [ ] No `IO.inspect`/`dbg` left in delivered code; secrets only in `runtime.exs`
- [ ] Agent ran every §3 command and documented any fixes

---
**End of Elixir Guidelines**
