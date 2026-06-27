# Performance Engineering Guidelines
Mandatory, language-agnostic standards for fast software: measure first, set budgets, optimize the proven bottleneck. Profilers, load generators (k6, Locust, wrk), perf budgets, p50/p95/p99 latency.

---
name: performance
title: Performance Engineering Guidelines
version: 2.0
last_reviewed: 2026-06-05
kind: cross-cutting
tools: [k6, locust, wrk, lighthouse-ci, async-profiler, py-spy, pprof, perf]
requires: []
recommends:
  - observability
  - parallelism
  - error-handling
provides:
  - profiling
  - perf-budgets
  - caching-strategy
  - load-testing
---

> 🧭 Authored per [`CONVENTIONS.md`](guides://CONVENTIONS.md): shared concerns are referenced, not restated. This guide owns performance engineering as a discipline; it never restates concurrency, metrics/SLO, or error-budget rules owned elsewhere.

---

## 0. Prerequisites & References

This guide is language-agnostic. The numbers you optimize *toward* (SLOs, percentiles, alert thresholds) and the *mechanisms* you optimize *with* (concurrency, batching) live in the guides below — fetch them when the task touches them.

> 📎 **RECOMMENDED — fetch when the task touches them:**
> - [`observability.md`](guides://observability.md) — metrics, tracing, **SLOs and percentiles** that performance targets. This guide sets the budget; observability *measures and alerts* on it. Do not redefine metric/trace plumbing here.
> - [`parallelism.md`](guides://parallelism.md) — concurrency/async/threads for **throughput**. This guide decides *when* parallelism is the right lever; parallelism.md owns *how* to do it safely.
> - [`error-handling.md`](guides://error-handling.md) — timeouts, retries, backoff, circuit breakers, and **error budgets** (the failure side of an SLO). Cite it for failure-mode tuning.

> 📎 **SEE ALSO (technology-specific tuning lives in the owning guide, not here):**
> - Datastores: [`postgresql.md`](guides://postgresql.md) · [`mysql-mariadb.md`](guides://mysql-mariadb.md) · [`redis.md`](guides://redis.md) · [`mongodb.md`](guides://mongodb.md) · [`sql.md`](guides://sql.md) — indexing, query plans, connection pools.
> - Languages: [`python.md`](guides://python.md) · [`go.md`](guides://go.md) · [`rust.md`](guides://rust.md) · [`java.md`](guides://java.md) · [`javascript.md`](guides://javascript.md) — allocation, GC, profiler invocation.
> - Frontend/web: [`reactjs.md`](guides://reactjs.md) · [`nextjs.md`](guides://nextjs.md) · [`css.md`](guides://css.md) · [`html.md`](guides://html.md) — render path, bundle splitting, Core Web Vitals binding.
> - Infra: [`kubernetes.md`](guides://kubernetes.md) · [`docker-compose.md`](guides://docker-compose.md) · [`aws.md`](guides://aws.md) — resource limits, autoscaling, CDN/edge caching.
> - [`ci-cd.md`](guides://ci-cd.md) — where budget gates (§9) run.

---

## 1. Core Philosophies: PERF-FIRST

Performance-specific principles only. Concurrency, metrics, and error budgets come from §0.

- **P**rove it: **measure before you optimize, and after.** No change ships on intuition — it ships on a before/after number from a profiler or load test. Premature optimization of un-profiled code is forbidden.
- **E**liminate the dominant cost first: optimize the single biggest contributor (Amdahl's law). A 10× speedup of code that is 5% of runtime buys ~5%; fix the 80% first.
- **R**ight complexity: choose the correct **algorithm and data structure** (Big-O) before micro-tuning constants. No accidental O(n²) on hot paths; no O(n) lookups where a hash/index gives O(1)/O(log n).
- **F**lat tails: optimize for the **percentiles users feel** (p95/p99), not the mean. A good average with a fat tail is a slow product.
- **I**n a budget: every hot path has an explicit, enforced **performance budget** (§9). Regressions fail CI, not production.
- **R**euse, don't recompute: cache deliberately (§5) with explicit TTL/invalidation; pool expensive resources; batch I/O.
- **S**tream, don't slurp: bound memory — process in chunks/generators; never load an unbounded result set into RAM.
- **T**est under load: capacity is proven by a load test against budget (§7), not assumed.

**Verified Code**: agent-generated code MUST pass every gate in §2 before delivery.

---

## 2. Requirements (MANDATORY, auditable)

RFC-2119 keywords. IDs `PERF-<TOPIC>-<NN>`. Each row has a binary gate; rows binding a shared rule cite its owner.

| ID | Requirement | Verify | Gate |
|----|-------------|--------|------|
| PERF-PROF-01 | Any optimization MUST be justified by a profiler/benchmark measurement (before & after) | attach profile/bench diff to the change | both numbers present, after ≤ before |
| PERF-PROF-02 | Hot paths MUST NOT contain unintended super-linear complexity | code review / complexity analysis on identified hot path | no accidental O(n²)+ on hot path |
| PERF-BUD-01 | Every user-facing path MUST have a documented performance budget (latency + size) | budget file exists (`performance-budget.yml`) | budget defined for each path |
| PERF-BUD-02 | Budgets MUST be enforced in CI; a regression fails the build (see `ci-cd.md`) | budget gate job (e.g. Lighthouse CI / k6 thresholds) | job present, fails on breach |
| PERF-LAT-01 | Service latency MUST be tracked and asserted at p95/p99, not mean (see `observability.md`) | load-test thresholds on `p(95)`/`p(99)` | thresholds defined & met |
| PERF-LOAD-01 | Capacity-relevant changes MUST pass a load test at target concurrency before release | `k6 run` / `locust` against budget | thresholds pass |
| PERF-CACHE-01 | Every cache MUST have a bounded size (eviction) and explicit TTL + invalidation rule | review cache config | max-size + TTL + invalidation defined |
| PERF-MEM-01 | Large/unbounded datasets MUST be streamed/paginated, not fully materialized | review for `fetch_all`/unbounded list returns | no unbounded materialization |
| PERF-IO-01 | Independent I/O MUST be batched or parallelized; no N+1 and no serial-when-independent calls (see `parallelism.md`) | review / query log / trace | no N+1, independents concurrent |
| PERF-REG-01 | No performance regression on hot paths is merged without sign-off | benchmark CI / flame-graph diff | within budget tolerance |

> **Forbidden**: optimizing code you have not profiled; shipping a cache without eviction or invalidation; loading unbounded result sets into memory; making independent remote calls serially; asserting only on mean latency; merging a budget regression.

---

## 3. The Optimization Loop (measure → fix → re-measure)

The only sanctioned workflow. Skipping step 1 or 5 violates `PERF-PROF-01`.

1. **Set the target.** A number tied to an SLO (owned by [`observability.md`](guides://observability.md)) or a budget (§9): "checkout p95 < 200 ms", "import job < 5 min for 1 M rows".
2. **Measure the baseline.** Profile or load-test the *real* path with *representative* data. Synthetic micro-benchmarks lie about cache, I/O, and contention.
3. **Find the dominant cost.** Read the flame graph / top-N. Optimize the widest frame first (Amdahl). Common culprits: §8.
4. **Form one hypothesis, change one thing.** Algorithm/data structure first (§4), then I/O batching/caching (§5/§6), then constants/allocations last.
5. **Re-measure on the same harness.** Keep the before/after delta. If it didn't move the target number, revert it — complexity without a win is a regression.
6. **Lock it in.** Add a benchmark or budget assertion (§9) so the win can't silently regress (`PERF-REG-01`).

### Profiling — pick the tool for the question
Wall-clock vs CPU vs allocation vs lock contention are different questions; use a profiler that answers the one you have. Invocation is language-specific — see the language guide.

| Question | Tool family (see language guide for exact invocation) |
|---|---|
| Where is CPU time spent? | sampling CPU profiler → flame graph (`py-spy`, `pprof`, `async-profiler`, `perf`) |
| Why is wall-clock > CPU? | tracing/span profiler; check I/O waits & lock waits (see `observability.md` tracing) |
| Where is memory allocated/retained? | heap/allocation profiler; leak = retained set grows without bound |
| Where do requests spend time across services? | distributed tracing (owned by `observability.md`) |
| What is the throughput ceiling? | load generator (§7) |

> Profile in a **production-like** build/config (optimizations on, realistic data volume). A debug-build profile is a different program.

---

## 4. Algorithmic Performance (Big-O first)

The largest, cheapest wins are almost always algorithmic, not micro-optimizations. Get the complexity right before touching constants.

- **Know the cost of your operations.** Membership/lookup: hash set/map → O(1) avg vs list scan → O(n). Sorted-range queries: tree/B-tree index → O(log n). Avoid repeatedly scanning a collection inside a loop (the classic accidental O(n²)).
- **Pick the data structure for the access pattern**, not by habit: dict/map for keyed lookup; set for dedup/membership; heap/priority-queue for top-k; ring buffer for bounded streams; trie/index for prefix/range.
- **Reduce work, don't just speed it up.** Precompute, memoize pure functions, hoist invariants out of loops, short-circuit (`exists`/`any` over `count`), and prune early.
- **Mind constant factors only after Big-O is right.** Cache locality, branch prediction, and allocation count matter on truly hot inner loops — but a better algorithm usually dominates a tuned bad one.
- **Bound everything.** Unbounded recursion, growth, or fan-out is a latent O(∞). Cap depth, page results, and limit concurrency (limits owned by `parallelism.md`).

> Database query complexity (indexes, query plans, keyset vs OFFSET pagination, EXISTS vs COUNT) is the same principle applied to data and is owned by the datastore guides ([`sql.md`](guides://sql.md), [`postgresql.md`](guides://postgresql.md)). Don't restate query tuning here — name the bottleneck and reference them.

---

## 5. Caching Strategy

Caching trades memory and freshness for latency. **The hard part is invalidation, not lookup** — so every cache obeys `PERF-CACHE-01`: bounded size + explicit TTL + a defined invalidation rule.

### Patterns
- **Cache-aside (lazy):** app checks cache, on miss loads from source and populates. Most common; tolerate the first-request miss.
- **Read-through / write-through:** cache sits in front of the store and handles load/write itself. Write-through keeps cache and store consistent at write cost.
- **Write-behind:** buffer writes, flush asynchronously. Highest write throughput, weakest durability — only with a durable buffer.
- **Tiered (L1 in-process → L2 shared → origin):** L1 (per-process, tiny TTL, fastest) backed by L2 (shared, e.g. Redis) backed by the store. Promote on L2 hit; size-cap L1.

### Non-negotiables
- **Eviction is mandatory.** An unbounded cache is a memory leak (§8). Use LRU/LFU/size or TTL eviction. (LRU is a few lines around an ordered map — name the idiom; the language guide shows the binding.)
- **Invalidation is explicit.** On the write path, invalidate or update the key. Prefer short TTL + event-driven invalidation over hoping TTL is "short enough".
- **Key deterministically.** Stable, collision-free keys; include the version/schema so a deploy doesn't serve stale shapes.
- **Guard the stampede.** On expiry of a hot key, use a single-flight lock or staggered/jittered TTL so N requests don't all recompute at once.
- **HTTP/CDN caching** (immutable hashed assets `Cache-Control: public, immutable`, `ETag`/`Last-Modified`, edge TTLs) is the same strategy at the network tier — bind it in the web/infra guides, don't reimplement it in app code.

---

## 6. Latency vs Throughput, and the Tail

These are different goals and sometimes conflict (batching helps throughput but adds latency).

- **Latency** = time for one operation. Optimize the **tail (p95/p99/p99.9)** — that's what users and SLOs feel. Tracking only the mean hides the slow requests (`PERF-LAT-01`).
- **Throughput** = operations per unit time. Raise it with concurrency, batching, and pooling — concurrency mechanics are owned by [`parallelism.md`](guides://parallelism.md); this guide only decides *that* it's the right lever.
- **Latency reducers:** remove serial dependencies (run independent I/O concurrently — `PERF-IO-01`), cache (§5), reduce payload/serialization, keep connections warm via pooling, fail fast with timeouts (owned by [`error-handling.md`](guides://error-handling.md)).
- **Throughput raisers:** batch/bulk I/O (one `executemany`/`COPY` beats N inserts), pool connections (size to the workload, never one-per-request), backpressure to protect downstreams.
- **Little's Law:** `concurrency = throughput × latency`. To hit a throughput target at a fixed latency you need a known minimum concurrency/pool size — size pools and worker counts from this, not by guessing.
- **Tail-tolerance:** hedged requests, timeouts, and load shedding cap the tail; their policy lives in [`error-handling.md`](guides://error-handling.md).

---

## 7. Load & Stress Testing

Capacity is proven, not assumed (`PERF-LOAD-01`). Define thresholds as the **gate** (matching the budget), ramp realistically, and assert on percentiles.

- **Test types:** *load* (expected peak), *stress* (beyond peak, find the knee), *soak* (hours, find leaks/degradation), *spike* (sudden surge, test autoscaling/shedding).
- **Model real users:** realistic ramp stages, think-time between actions, and a representative request mix — not a single endpoint hammered flat-out.
- **Assert percentiles + error rate**, and fail the run on breach.

```javascript
// k6 — gate on tail latency and error rate (run: k6 run --vus 50 --duration 5m load.js)
import http from 'k6/http';
import { check, sleep } from 'k6';

export const options = {
  stages: [
    { duration: '1m', target: 10 },   // warm up
    { duration: '3m', target: 50 },   // expected peak
    { duration: '2m', target: 100 },  // stress past peak
    { duration: '1m', target: 0 },    // cool down
  ],
  thresholds: {                       // these ARE the gate (PERF-LAT-01 / PERF-LOAD-01)
    http_req_duration: ['p(95)<200', 'p(99)<500'],
    http_req_failed:   ['rate<0.01'],
  },
};

export default function () {
  const res = http.get(`${__ENV.BASE_URL}/api/products`);
  check(res, { 'status 200': (r) => r.status === 200 });
  sleep(Math.random() * 3);           // think time
}
```

> The metrics this emits feed the dashboards/alerts owned by [`observability.md`](guides://observability.md); the latency targets should equal the service SLO. Don't define the SLO here — reference it.

---

## 8. Common Bottlenecks & Anti-Patterns

The usual suspects, ranked by how often they dominate a flame graph. Each has a one-line fix; the deep fix lives in the named owner guide.

- **N+1 queries / chatty I/O** — a query (or RPC) per row in a loop. Fix: one JOIN or batched/`IN` query. *(query tuning → datastore guides.)*
- **Serial independent calls** — `await a; await b; await c` when they don't depend on each other. Fix: run concurrently (`Promise.all`/`gather`); mechanics → [`parallelism.md`](guides://parallelism.md). Total latency drops from sum to max.
- **Missing/incorrect index** — full scan whose cost grows with table size. Fix: index the WHERE/JOIN/ORDER BY columns → datastore guide.
- **Unbatched writes** — N single INSERTs instead of one bulk/`COPY`. Fix: batch.
- **No pagination** — unbounded result set → memory blowup and timeouts. Fix: keyset pagination + max page size (`PERF-MEM-01`).
- **Over-serialization** — dumping entire ORM graphs (all relations) to JSON. Fix: explicit DTO/schema with only needed fields.
- **Unbounded cache / listener / buffer = memory leak** — set that only grows. Fix: bounded LRU/TTL; remove listeners on teardown (`PERF-CACHE-01`).
- **Main-thread / event-loop blocking** — long synchronous CPU work stalls everything. Fix: chunk + yield, or offload to a worker/thread (→ [`parallelism.md`](guides://parallelism.md)).
- **Connection churn** — opening a new connection/client per request. Fix: a reused, size-bounded pool.
- **Premature/unmeasured optimization** — clever code with no profile behind it. Fix: §3; revert if it didn't move the target.

---

## 9. Performance Budgets

A budget is a contract per path, enforced in CI (`PERF-BUD-01/02`) so performance can't erode commit by commit. Set tighter budgets for conversion-/revenue-critical paths.

```yaml
# performance-budget.yml — backend services + web pages share one file
budgets:
  api.checkout:                 # tightest: revenue path
    p50_latency_ms: 50
    p95_latency_ms: 200
    p99_latency_ms: 500
    error_rate_pct: 0.1
    max_response_kb: 200
  api.product_list:
    p95_latency_ms: 200
    max_response_kb: 500
  web.homepage:                 # Core Web Vitals (thresholds owned by web guides)
    lcp_ms: 2500
    inp_ms: 200
    cls: 0.1
    total_js_kb: 200            # compressed
    total_weight_kb: 1000
```

**Enforce in CI** (gate job per `ci-cd.md`):
- **Backend:** k6/Locust thresholds (§7) fail the pipeline on breach.
- **Web:** Lighthouse CI assertions on LCP/INP/CLS + bundle-size limits (`bundlesize`/`size-limit`). The Core Web Vitals *thresholds and frontend tuning* (critical CSS, code splitting, image/font optimization, resource hints) are owned by the web guides — this guide owns the *budget-and-gate discipline*, not the CSS.

```javascript
// lighthouserc.js — web budget gate (assertions = the gate)
module.exports = {
  ci: {
    collect: { numberOfRuns: 5, url: ['http://localhost:3000/', 'http://localhost:3000/checkout'] },
    assert: {
      assertions: {
        'largest-contentful-paint': ['error', { maxNumericValue: 2500 }],
        'cumulative-layout-shift':  ['error', { maxNumericValue: 0.1 }],
        'interactive':              ['error', { maxNumericValue: 3500 }],
        'resource-summary:script:size': ['error', { maxNumericValue: 200000 }],
      },
    },
  },
};
```

> Measure real users too: capture field metrics (RUM / `PerformanceObserver`, Web Vitals) and compare against the budget — but the *collection, dashboards, and alerting* are owned by [`observability.md`](guides://observability.md). Budgets define the line; observability watches it.

---

## 10. Deployment Checklist

Generated from §2 — one box per requirement ID. No new requirements.

- [ ] PERF-PROF-01 — change backed by before/after profile or benchmark
- [ ] PERF-PROF-02 — hot paths free of accidental super-linear complexity
- [ ] PERF-BUD-01 — every user-facing path has a documented budget
- [ ] PERF-BUD-02 — budgets enforced in CI; regression fails the build (see `ci-cd.md`)
- [ ] PERF-LAT-01 — latency asserted at p95/p99, not mean (see `observability.md`)
- [ ] PERF-LOAD-01 — capacity changes pass a load test at target concurrency
- [ ] PERF-CACHE-01 — every cache bounded with TTL + invalidation rule
- [ ] PERF-MEM-01 — large datasets streamed/paginated, never fully materialized
- [ ] PERF-IO-01 — no N+1; independent I/O batched/parallelized (see `parallelism.md`)
- [ ] PERF-REG-01 — no hot-path regression merged without sign-off
- [ ] Agent ran the §3 optimization loop and attached the measurement delta

---

**End of Performance Engineering Guidelines**
