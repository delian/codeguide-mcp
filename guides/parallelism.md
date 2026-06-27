# Concurrency & Parallelism Guidelines
Mandatory, language-agnostic standards for safe, correct concurrent and parallel code: concurrency models, race conditions, synchronization, deadlock avoidance, and structured concurrency. Tool-agnostic; language primitives live in the language guides.

---
name: parallelism
title: Concurrency & Parallelism Guidelines
version: 2.0
last_reviewed: 2026-06-05
kind: cross-cutting
tools: []
requires: []
recommends:
  - performance
  - error-handling
  - observability
provides:
  - concurrency-models
  - race-conditions
  - synchronization
  - structured-concurrency
---

> 🧭 Authored per [`CONVENTIONS.md`](guides://CONVENTIONS.md): shared concerns are referenced, not restated. This guide owns concurrency & parallelism as a cross-cutting concern; it names idioms and defers concrete primitives to the language guides.

---

## 0. Prerequisites & References

This guide is the canonical owner of concurrency & parallelism. It defines the models, hazards, and discipline; it does **not** restate performance theory, error strategy, or per-language syntax.

> 📎 **RECOMMENDED — fetch when the task touches them:**
> - [`performance.md`](guides://performance.md) — concurrency is a *perf lever*: measure before parallelizing, model throughput/latency, contention budgets.
> - [`error-handling.md`](guides://error-handling.md) — cancellation, timeouts, propagation of failures across tasks, retry/backoff policy.
> - [`observability.md`](guides://observability.md) — tracing across task/thread boundaries, contention/queue-depth metrics.

> 📎 **SEE ALSO — language-specific concurrency primitives (fetch the one(s) you target):**
> - [`go.md`](guides://go.md) — goroutines, channels, `select`, `sync`, `context`, the race detector.
> - [`rust.md`](guides://rust.md) — `Send`/`Sync`, ownership-enforced freedom from data races, `tokio`/`async`, `Arc<Mutex<…>>`, `rayon`.
> - [`java.md`](guides://java.md) — `java.util.concurrent`, virtual threads (Project Loom), `CompletableFuture`, `StructuredTaskScope`, the JMM.
> - [`python.md`](guides://python.md) — `asyncio`, `concurrent.futures`, `multiprocessing`, the GIL and free-threaded builds.
> - [`cpp.md`](guides://cpp.md) · [`c.md`](guides://c.md) — `std::atomic`, memory orderings, `std::jthread`, threads + sanitizers.
> - [`kotlin.md`](guides://kotlin.md) · [`elixir.md`](guides://elixir.md) · [`scala.md`](guides://scala.md) — coroutines / BEAM actors / effect systems.

---

## 1. Core Philosophies

Concurrency-specific principles only. Performance theory, error strategy, and language syntax come from the §0 references.

- **Don't, until you must.** Sequential code has no races. Parallelize only when profiling proves the need (see `performance.md`); concurrency is a cost, not a feature.
- **Make races impossible by design, not by careful coding.** Prefer immutability, ownership, and message passing so that the *type system or structure* forbids the bug — review and luck do not scale.
- **Climb the hierarchy as little as possible.** Choose the highest-level model that meets the need (§3). Lower levels (raw locks, lock-free, multiprocessing) are progressively more dangerous.
- **Share by communicating.** Move ownership through channels/queues rather than sharing mutable memory behind locks.
- **Structure task lifetimes.** Every spawned task lives inside a scope that owns its cancellation and cleanup — no orphaned tasks/threads/goroutines.
- **Bound everything.** Bounded queues, bounded pools, timeouts on every blocking call. Unbounded concurrency is a latent OOM and a DoS vector.
- **Correctness over speed.** A safe, slightly slower design beats a fast one with a non-deterministic data race.

**Verified Code**: Agent-generated concurrent code MUST pass every gate in §2 (race detector clean, deadlock-free, bounded, tested) before delivery.

---

## 2. Requirements (MANDATORY, auditable)

RFC-2119 keywords. IDs `PAR-<TOPIC>-<NN>`. Binary gates; rows binding a shared rule cite its owner. Verification is largely language-tool-driven — bind these to the concrete tool in the relevant language guide.

| ID | Requirement | Verify | Gate |
|----|-------------|--------|------|
| PAR-STRUCT-01 | Concurrency MUST only be introduced after a sequential baseline is profiled and shown insufficient (see `performance.md`) | profile/benchmark attached to change | baseline recorded |
| PAR-STRUCT-02 | The chosen concurrency level MUST be the highest in the §3 hierarchy that meets the need, and documented | design note / PR description | level justified |
| PAR-RACE-01 | Code MUST be free of data races | language race detector (e.g. `go test -race`, TSan, `cargo`+Miri/loom) | 0 races |
| PAR-RACE-02 | Shared mutable state MUST be immutable, owned by one task, or fully synchronized — never partially locked across a read-modify-write | review / static analysis | no torn RMW |
| PAR-SYNC-01 | All locks acquired in >1 site MUST follow one documented global ordering (no lock inversion) | review / lock-order linter | single order |
| PAR-SYNC-02 | Every blocking/awaiting call MUST have a timeout or be cancellation-bound (see `error-handling.md`) | review / grep for unbounded `wait`/`join`/`recv` | no infinite waits |
| PAR-STRUCT-03 | Spawned tasks MUST be scoped (structured concurrency); no detached/orphaned tasks | review / leak test | 0 leaked tasks |
| PAR-BOUND-01 | Queues and worker pools MUST be bounded; producers experience backpressure | review / config | finite limits set |
| PAR-SEC-01 | Concurrent file/resource access MUST avoid TOCTOU; locks MUST NOT be held across I/O (see `secure-coding.md`) | review | atomic ops, no lock-over-I/O |
| PAR-TST-01 | Concurrent logic MUST have stress tests (high task counts, repeated runs) and a CI race-detector job (see `tdd.md`) | CI job | race job green |
| PAR-TST-02 | Each concurrency bug MUST get a deterministic regression test before the fix (see `tdd.md`) | test run | failing→passing |
| PAR-OBS-01 | Contention, queue depth, and task latency SHOULD be observable (see `observability.md`) | metrics present | dashboards/metrics exist |

> **Forbidden**: shipping concurrent code with a dirty race detector; holding a lock across network/disk I/O; unbounded task spawning; `join`/`await`/`recv` without a timeout or cancellation; "fixing" a race by adding `sleep`; double-checked locking outside a language's blessed safe pattern.

---

## 3. The Concurrency Hierarchy (choose the highest level that fits)

Always choose the highest level of abstraction that meets your needs; each step down adds danger and cost. Map the chosen level to the concrete primitive in your language guide (§0 SEE ALSO).

| Lvl | Model | Use when | Why / caveat |
|----|-------|----------|--------------|
| 1 | **Sequential** | Performance is adequate; problem is inherently serial | No races possible. Default. |
| 2 | **Async / await** (cooperative) | I/O-bound; thousands of concurrent ops mostly *waiting* | Low overhead, mostly single-threaded → few data races. One blocking call stalls the loop; doesn't use multiple cores. |
| 3 | **Thread pool + immutable/copied data** | CPU-bound, independent work units | Uses all cores, no synchronization needed. Cost: data copying, pool tuning. |
| 4 | **Threads/tasks + message passing** | Workers must communicate | Channels give clear ownership; no explicit locks. Cost: channel deadlocks, error plumbing. |
| 5 | **Threads + locks** | Must share mutable state, message passing impractical | Last resort. Races, deadlocks, contention; hard to test. Minimize critical sections. |
| 6 | **Lock-free / atomics** | Proven contention bottleneck, expert present | Highest throughput, no blocking. Extremely error-prone, platform-subtle — gate behind benchmarks. |
| 7 | **Multiprocessing** | Memory isolation, fault isolation, GIL bypass | Complete isolation. Highest overhead, slow IPC, large footprint. |

**Decision sketch:** need concurrency? → no: stay sequential. → I/O-bound? → async/await. → CPU-bound + data can be immutable/copied? → thread pool. → workers must talk? → channels. → must share mutable state? → locks (minimize scope) → contention proven? → lock-free. → need isolation / bypass a global lock? → processes.

**Pool sizing:** CPU-bound ≈ `num_cores` (or `-1`); I/O-bound ≈ `num_cores × (1 + wait/compute)`; mixed → separate pools for CPU vs I/O work so a slow I/O task can't starve compute (tune empirically — see `performance.md`).

---

## 4. Async / Await (the preferred model for I/O)

Cooperative concurrency on an event loop: a task yields control at each `await`, so one OS thread serves thousands of waiting operations. Bind to your runtime (`asyncio`, `tokio`, JS event loop, virtual threads) via the language guide.

**Idioms to apply:**
- **Launch independent work concurrently, then join.** Sequential `await a; await b;` when `a` and `b` are independent is the #1 async anti-pattern — gather/`join!`/`Promise.all` them instead.
- **Never block the loop.** CPU-heavy work or a synchronous blocking call inside an async task freezes every other task → offload to a thread/process pool (`run_in_executor`, `spawn_blocking`).
- **Always `await`.** A fire-and-forget call that returns an un-awaited future runs detached or never — violates PAR-STRUCT-03.
- **Timeout / cancel everything.** Wrap awaits in a timeout; propagate cancellation. Failure-aggregation (`Promise.allSettled`, `join_all` with results) so one failure doesn't silently drop siblings — policy in `error-handling.md`.
- **Backpressure with a semaphore.** Cap in-flight operations rather than spawning one task per input.

> Cancellation, timeout, and retry/backoff *policy* is owned by [`error-handling.md`](guides://error-handling.md); this guide only mandates that async tasks be cancellation-bound (PAR-SYNC-02).

---

## 5. Shared State & Race Conditions

**Shared mutable state is the root cause of most concurrency bugs.** In priority order:

1. **No shared state** — each task owns its data (best; impossible to race).
2. **Shared immutable state** — read-only data is freely shareable.
3. **Message passing** — transfer ownership through a channel; one owner at a time.
4. **Shared mutable + synchronization** — last resort; protect with a lock/atomic.

**Race-condition taxonomy and the fix:**

| Hazard | Fix |
|--------|-----|
| Check-then-act (e.g. `if not exists: create`) | Atomic compare-and-swap, or hold the lock across both steps |
| Read-modify-write (`x = x + 1`) | Atomic `fetch_add`, or one lock around the whole RMW (PAR-RACE-02) |
| Lazy initialization | `once` / `call_once` / lazy-static idiom |
| Partial critical section | Make the entire invariant-preserving operation one critical section — releasing mid-sequence reintroduces the race |

A critical section MUST cover the *whole* read-compute-write that preserves an invariant; locking the read and the write separately (releasing in between) is still a race. Prefer thread-local storage when per-worker state suffices — no synchronization needed.

> Ownership-based languages (Rust) push much of this into the compiler (`Send`/`Sync`); GC/JVM/CLR languages rely on the memory model + `concurrent` collections — see the language guide.

---

## 6. Synchronization Primitives & Memory Visibility

Pick the lightest primitive that expresses the constraint; map it to the language API in §0.

| Primitive | Purpose |
|-----------|---------|
| Mutex / Lock | Exclusive access to a resource |
| RWLock | Many readers **or** one writer |
| Semaphore | Cap concurrent access count (backpressure) |
| Condition variable | Wait until a predicate becomes true (always re-check in a `while`) |
| Atomic | Lock-free single-variable RMW / flags / CAS |
| Channel / Queue | Message passing, ownership transfer |
| Barrier | Rendezvous N tasks at a point |

**Memory visibility:** without synchronization, one thread's writes may never become visible to another, and the compiler/CPU may reorder them — this is undefined behavior, not "eventually consistent." Establish *happens-before* via a lock (release→acquire), an atomic with acquire/release ordering, or a memory fence. Never busy-wait on a plain (non-atomic) flag.

**Atomics vs locks:** use atomics for a single variable / simple RMW / low contention / wait-free needs; use a lock when multiple variables must stay mutually consistent, the operation is complex, or clarity matters. Detailed memory-ordering semantics (`acquire`/`release`/`seq_cst`) belong to `cpp.md`/`rust.md`.

---

## 7. Deadlock & Liveness

A deadlock needs all four Coffman conditions simultaneously — **break at least one**: mutual exclusion, hold-and-wait, no-preemption, circular wait.

**Prevention (in preference order):**
1. **Eliminate locks** — message passing / immutability has no lock cycle.
2. **Global lock ordering** — always acquire locks in one documented order (e.g. by resource ID). This breaks *circular wait* and is mandated by PAR-SYNC-01. The classic `transfer(from, to)` bug is fixed by locking the two accounts in ID order, not argument order.
3. **Try-lock with timeout + backoff** — acquire-or-release-and-retry breaks *hold-and-wait*; pair with randomized backoff to avoid livelock.
4. **Single coarse lock** — when contention is low, one lock can't deadlock.

Also guard against **livelock** (threads busy but not progressing — add backoff/jitter), **starvation** (use fair locks / queues), and **lost wakeups** (signal under the lock, wait in a loop).

---

## 8. Structured Concurrency

Bind every task's lifetime to a lexical scope that owns its children: the scope does not exit until all child tasks complete or are cancelled, and a child failure cancels its siblings and propagates upward. This makes leaks impossible (PAR-STRUCT-03) and turns "where did this task go?" into a structural guarantee.

- Use the language's scope construct: Go `errgroup`/`context`, Kotlin `coroutineScope`, Java `StructuredTaskScope`, Trio/`asyncio.TaskGroup` nurseries, Swift `withTaskGroup`.
- Cancellation flows down the scope tree via a cancellation token / `context`; tasks MUST check it cooperatively (never hard-kill threads — that leaves locks/resources in an undefined state).
- On scope exit (normal, error, or cancel) all resources acquired inside are released. This is the concurrency analogue of RAII / `try/finally`.

> The *failure* semantics (what to do when a child errors — retry, fail-fast, partial result) are policy from [`error-handling.md`](guides://error-handling.md); structured concurrency is the *mechanism* that guarantees cleanup and cancellation reach every task.

---

## 9. Concurrency Patterns

Name the pattern; implement with the lightest §6 primitive in your language.

- **Producer–Consumer** — a *bounded* queue between producers and consumers; full queue blocks producers (backpressure), empty queue blocks consumers. Bounding is mandatory (PAR-BOUND-01).
- **Fan-Out / Fan-In** — distribute work to N workers, then merge results (parallel map/reduce). Ideal at hierarchy Level 3 with immutable inputs.
- **Pipeline** — stages connected by channels; each stage is single-purpose and can scale independently. Watch the slowest stage (the bottleneck).
- **Worker Pool** — fixed workers pulling tasks from a shared queue; caps resource usage and amortizes thread creation.
- **Actor Model** — isolated state per actor, mutated only via its mailbox; no shared memory, so no data races by construction (BEAM/Akka/etc.).
- **Read-Write Lock** — many concurrent readers or one writer for read-mostly shared data; beware writer starvation.

> For the general GoF patterns these compose with, see [`designpatterns.md`](guides://designpatterns.md); show only the concurrency binding.

---

## 10. Concurrency-Related Security

Concurrency opens specific vulnerability classes (full policy in [`secure-coding.md`](guides://secure-coding.md); here are the concurrency-specific bindings — PAR-SEC-01):

- **TOCTOU** — a check then a use, with an attacker-controllable gap (file replaced/symlinked in between). Fix: operate on a single atomic handle (open with `O_NOFOLLOW`, `openat`, fd-based checks) rather than re-resolving a path.
- **Resource-exhaustion / DoS** — unbounded task or connection spawning lets an attacker exhaust threads/memory. Fix: bound with a semaphore/pool and shed load when full (ties to PAR-BOUND-01).
- **Lock-based DoS** — holding a lock across slow I/O lets a slow client stall every other thread. Fix: do I/O outside the lock; lock only the in-memory mutation.

---

## 11. Testing Concurrent Code

Concurrency bugs are non-deterministic, so testing is probabilistic — design for detection (binds PAR-TST-01/02; test-first policy from [`tdd.md`](guides://tdd.md)).

- **Race / sanitizer job in CI** — TSan, `go test -race`, Helgrind, Rust `loom`/Miri. A dirty detector fails the build (PAR-RACE-01).
- **Stress tests** — many threads × many iterations; assert the invariant (e.g. final counter == expected). Vary thread counts (1, 2, 4, 8, 16+) to surface different interleavings.
- **Repeat + jitter** — run hot tests hundreds/thousands of times; inject random delays to perturb timing.
- **Invariant & happens-before checks** — assert pre/post invariants around critical sections; use barriers to assert visibility after a synchronization point.
- **Deadlock detection** — every blocking test has a timeout; a hang is a failure, not a hang.
- **Shutdown & error paths** — verify clean termination, no leaked tasks, resources released on timeout/exception (ties to PAR-STRUCT-03).
- **Regression-first** — reproduce a reported race deterministically (model-checker like `loom`, or controlled scheduling) before fixing it (PAR-TST-02).

---

## 12. Performance of Concurrent Code

Concurrency is a performance lever, not free speed — measure, don't assume (full methodology in [`performance.md`](guides://performance.md)). Concurrency-specific pitfalls:

- **Lock contention** — a single global lock serializes everything. Shard locks by key, or go lock-free, only when contention is *measured*.
- **False sharing** — independent variables on the same cache line ping-pong between cores. Pad hot per-thread data to cache-line boundaries.
- **Over-synchronization** — locking each step of a parallel reduce destroys the parallelism. Accumulate per-thread, reduce once at the end.
- **Amdahl's law** — the serial fraction caps speedup; shrinking critical sections matters more than adding threads. (Theory: `performance.md`.)
- **Oversubscription** — more threads than cores adds context-switch overhead without throughput; size pools per §3.

> Surface contention, queue depth, and per-task latency as metrics so regressions are visible in production (see `observability.md`, PAR-OBS-01).

---

## Deployment Checklist

Generated from §2 — one box per requirement ID. No new requirements.

- [ ] PAR-STRUCT-01 — sequential baseline profiled before parallelizing
- [ ] PAR-STRUCT-02 — concurrency level chosen from §3 hierarchy and justified
- [ ] PAR-STRUCT-03 — all tasks scoped (structured concurrency); no orphans
- [ ] PAR-RACE-01 — race detector clean
- [ ] PAR-RACE-02 — shared mutable state immutable/owned/fully synchronized
- [ ] PAR-SYNC-01 — single documented global lock order
- [ ] PAR-SYNC-02 — every blocking/awaiting call has a timeout or is cancellation-bound
- [ ] PAR-BOUND-01 — queues and pools bounded; backpressure present
- [ ] PAR-SEC-01 — no TOCTOU; no lock held across I/O
- [ ] PAR-TST-01 — stress tests + CI race-detector job green
- [ ] PAR-TST-02 — each concurrency bug has a regression test before its fix
- [ ] PAR-OBS-01 — contention / queue depth / latency observable

---
**End of Concurrency & Parallelism Guidelines**
