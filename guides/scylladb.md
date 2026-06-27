# ScyllaDB Development Guidelines
Mandatory standards for ScyllaDB: shard-per-core data modeling, shard-aware drivers, tablets, and low-latency operations. ScyllaDB 6.x, CQL, shard-aware drivers, tablets, Alternator, ScyllaDB Manager, ScyllaDB Monitoring Stack.

---
name: scylladb
title: ScyllaDB Development Guidelines
version: 2.0
last_reviewed: 2026-06-05
kind: datastore
tools: [scylladb@6.x, cql, shard-aware-drivers, tablets, scylla-manager, scylla-monitoring-stack, cqlsh]
requires:
  - secure-coding
  - error-handling
recommends:
  - observability
  - performance
  - cassandra
  - env-config
provides:
  - shard-per-core
  - shard-aware-drivers
  - scylla-tablets
  - dynamodb-alternator
  - scylla-cdc
---

> 🧭 Authored per [`CONVENTIONS.md`](guides://CONVENTIONS.md): shared concerns are referenced, not restated. ScyllaDB is a C++ rewrite of Apache Cassandra and is **CQL- and driver-compatible**, so data modeling, consistency, and compaction policy are owned by [`cassandra.md`](guides://cassandra.md). This guide covers only what is **unique to Scylla**: the shard-per-core (Seastar) architecture, shard-aware drivers, tablets, Alternator, CDC, and the resulting performance/operational model.

---

## 0. Prerequisites & References

Fetch and apply these **before** generating ScyllaDB code or schema. Their rules are assumed below and not repeated.

> 📎 **REQUIRED — fetch & apply first:**
> - [`secure-coding.md`](guides://secure-coding.md) — secrets, supply chain, CVE policy. *(Scylla binding: prepared statements (never string-built CQL), Authenticator/Authorizer, TLS client+inter-node, RBAC.)*
> - [`error-handling.md`](guides://error-handling.md) — error strategy & propagation. *(Scylla binding: idempotent retries, consistency-level fallback, `WriteTimeout`/`ReadTimeout`/`Unavailable` handling — see §5.)*

> 📎 **RECOMMENDED — fetch when the task touches them:**
> - [`cassandra.md`](guides://cassandra.md) — **CQL syntax, query-driven data modeling, partition & clustering keys, consistency levels (LOCAL_QUORUM), denormalization, compaction strategies, LWT, secondary indexes & materialized views.** Scylla is CQL-compatible; model exactly as Cassandra does. This guide notes only where Scylla **differs** (§4).
> - [`performance.md`](guides://performance.md) — perf methodology *(binding: shard-per-core tuning, §3)*
> - [`observability.md`](guides://observability.md) — metrics/tracing policy *(binding: ScyllaDB Monitoring Stack, per-shard metrics, §6)*
> - [`env-config.md`](guides://env-config.md) — externalize endpoints/credentials; never hardcode contact points or secrets.

> 📎 **SEE ALSO:** [`kubernetes.md`](guides://kubernetes.md) · [`docker-compose.md`](guides://docker-compose.md) (ScyllaDB Operator / local clusters) · [`redis.md`](guides://redis.md) · [`postgresql.md`](guides://postgresql.md) (alternatives when access patterns are not query-known in advance).

---

## 1. Core Philosophies: SCYLLA-FIRST

Scylla-specific principles only. CQL/modeling come from `cassandra.md`; security, errors, and perf method from §0.

- **S**hard-aware: every client uses a **shard-aware** driver so requests route directly to the CPU core (shard) owning the data — token-awareness alone is not enough on Scylla.
- **C**ore-local design: one CPU core = one shard with private memory, I/O queue, and network; design assumes **shared-nothing, lock-free** execution. A hot partition saturates a single core, not the node.
- **Y**our query patterns: schema is driven by queries, not entities (owned by `cassandra.md`) — restated here only as a reminder.
- **L**atency over knobs: rely on Scylla's **self-tuning** (autonomous I/O & CPU schedulers) and **workload prioritization** instead of hand-tuning dozens of JVM/GC flags — there is no JVM and no GC pause.
- **L**inear & elastic: scale by adding shards/nodes; prefer **tablets** (elastic, per-table sharding) over legacy vnodes for fast, incremental scaling.
- **A**utonomous ops: lean on ScyllaDB Manager (repair/backup) and the Monitoring Stack rather than bespoke tooling.

**Verified Code**: Agent-generated schema and client code MUST pass every gate in §2 before delivery.

---

## 2. Requirements (MANDATORY, auditable)

RFC-2119 keywords. IDs `SCYLLA-<TOPIC>-<NN>`. Each row has a binary gate; rows binding a shared rule cite its owner.

| ID | Requirement | Verify | Gate |
|----|-------------|--------|------|
| SCYLLA-TST-01 | Schema/query changes MUST be test-first against a real Scylla (testcontainers/CCM), incl. a regression test per data bug (see `cassandra.md`, `error-handling.md`) | run CQL integration suite | exit 0, 0 skips |
| SCYLLA-DRV-01 | Clients MUST use a **shard-aware** driver build/version with token-aware + DC-aware load balancing | check driver version + `nodetool`/dashboard shard-connection metric | shard-aware connections > 0 |
| SCYLLA-DRV-02 | All CQL MUST use prepared, parameterized statements; never string-built CQL (see `secure-coding.md`) | grep for f-strings/concat in CQL; review | none found |
| SCYLLA-MODEL-01 | Tables MUST be modeled per query (partition/clustering keys, denormalized), partitions bounded (≤ ~10MB / ~100k rows) (see `cassandra.md`) | `nodetool tablehistograms` partition size | within bound |
| SCYLLA-TBL-01 | New keyspaces SHOULD use **tablets** (`replication = {...'enabled': true}`) unless a feature requires vnodes | `DESCRIBE KEYSPACE` / system.tablets | tablets enabled or justified |
| SCYLLA-CONS-01 | Production reads & writes MUST use LOCAL_QUORUM (or stricter) with idempotent retry policy (see `cassandra.md`, `error-handling.md`) | review execution profiles | LOCAL_QUORUM + retry set |
| SCYLLA-SEC-01 | AuthN/AuthZ (RBAC) enabled; client-to-node and inter-node TLS on (see `secure-coding.md`) | `cqlsh` w/o creds fails; `openssl s_client` | auth required, TLS on |
| SCYLLA-SEC-02 | Credentials/endpoints externalized, 0 high/critical CVEs in driver deps (see `secure-coding.md`, `env-config.md`) | dependency audit; grep for literals | 0 high/critical, no literals |
| SCYLLA-OBS-01 | ScyllaDB Monitoring Stack scraping per-shard metrics; p99 latency & alerts wired (see `observability.md`) | Prometheus targets up, dashboards load | metrics present |
| SCYLLA-OPS-01 | Repairs & backups scheduled via ScyllaDB Manager (never manual `nodetool repair`) | `sctool task list` | scheduled tasks exist |

> **Forbidden**: token-aware-only drivers on Scylla (lose shard routing); string-built CQL; unbounded/hot partitions; manual `nodetool repair` in production; `ALLOW FILTERING` on large tables; hardcoded contact points or credentials.

---

## 3. Architecture: Shard-per-Core (the unique value)

Scylla is built on the **Seastar** framework: an asynchronous, shared-nothing, thread-per-core C++ runtime. Each CPU core is a **shard** that owns a slice of the node's data with its **own** memory allocator, I/O queue, network queues, and scheduler. There are **no locks** and effectively **no cross-shard communication** on the hot path.

```
Node
├── Shard 0 (CPU0): private RAM · I/O queue · CQL handler · token sub-ranges
├── Shard 1 (CPU1): private RAM · I/O queue · CQL handler · token sub-ranges
└── Shard N (CPUn): ...
```

**Why it matters for you:**
- **Shard-aware drivers** (§4.A) connect to the exact shard owning a partition → no intra-node hop. Token-awareness picks the node; shard-awareness picks the core. Use both.
- **No JVM, no GC pauses.** C++ + Seastar gives consistent sub-millisecond p99 under load; you tune for the OS/disk, not for GC. Do **not** port Cassandra heap/GC settings.
- **A hot partition is a hot core.** A single oversized or high-traffic partition saturates one shard while others idle → design keys to spread load across shards (modeling rules in `cassandra.md`).
- **Self-tuning / autonomous schedulers.** Scylla's I/O scheduler benchmarks the disk (`scylla_io_setup`/iotune) and the CPU scheduler balances foreground vs background (compaction, repair, streaming) work automatically. Avoid manual throttle knobs.
- **Workload prioritization** (Enterprise): assign `SERVICE LEVEL`s so OLTP traffic preempts analytics/batch on shared shards instead of competing blindly.
- **Heat-weighted load balancing**: replicas with warm cache are preferred for reads, smoothing tail latency after a node restart (cold cache nodes get less traffic until warm).

> Per-shard tuning methodology (CPU pinning, NUMA, disk benchmarking) is perf work — apply [`performance.md`](guides://performance.md). Scylla binding: run `scylla_setup`/`iotune` so the I/O scheduler is calibrated; pin shards to physical cores; isolate IRQs.

---

## 4. Where Scylla Differs from Cassandra

CQL, modeling, consistency, and compaction are **owned by [`cassandra.md`](guides://cassandra.md)** — do not relearn them here. The deltas:

### A. Shard-aware drivers
Use the ScyllaDB driver forks (or upstream DataStax drivers that support shard-awareness): `scylla-driver` (Python), `scylla-cpp-driver`, `gocql`/ScyllaDB fork (Go), Scylla Rust/Java/Node drivers. They auto-discover the shard-aware port (default 19042) — **do not** list it in contact points.

```python
# Python — ScyllaDB shard-aware driver (drop-in for cassandra-driver)
from cassandra.cluster import Cluster, ExecutionProfile, EXEC_PROFILE_DEFAULT
from cassandra.policies import TokenAwarePolicy, DCAwareRoundRobinPolicy

profile = ExecutionProfile(
    load_balancing_policy=TokenAwarePolicy(DCAwareRoundRobinPolicy(local_dc="DC1")),
    consistency_level=ConsistencyLevel.LOCAL_QUORUM,   # SCYLLA-CONS-01
)
cluster = Cluster(
    ["10.0.0.1", "10.0.0.2", "10.0.0.3"],              # plain CQL port; shard port auto-discovered
    execution_profiles={EXEC_PROFILE_DEFAULT: profile},
    protocol_version=4,
)
session = cluster.connect("production")
stmt = session.prepare("SELECT * FROM users WHERE user_id = ?")  # SCYLLA-DRV-02
session.execute(stmt, (user_id,))
```
Shard-awareness eliminates the intra-node hop a token-aware-only driver still pays. Verify via the `scylla_database_connections{shard=...}` spread or driver logs.

### B. Tablets (elastic) vs vnodes
Modern Scylla (6.x default for new keyspaces) replaces static **vnodes** with **tablets**: data is split into per-table tablets that the cluster **splits/merges and migrates automatically** based on size and load.
- **Elastic scaling**: new nodes receive tablets and serve traffic in minutes, not after full streaming of vnode ranges.
- **Per-table distribution**: small and huge tables are balanced independently.
- Enable on keyspace creation; some features (e.g. certain materialized-view/secondary-index or LWT combinations, depending on version) may still require vnode keyspaces — check the version matrix before disabling.
```sql
CREATE KEYSPACE production WITH replication = {
  'class': 'NetworkTopologyStrategy', 'DC1': 3
} AND tablets = {'enabled': true};
```

### C. Alternator — DynamoDB-compatible API
Scylla exposes a **DynamoDB-compatible HTTP API** ("Alternator") alongside CQL. Existing AWS SDK / DynamoDB apps point at the Alternator endpoint with **no code change** — useful for migrating off DynamoDB or running it on-prem. Enable with `alternator_port` in `scylla.yaml`; the same data is *not* shared between the CQL and Alternator interfaces (separate tables/keyspaces), so pick one API per dataset.

### D. CDC — Change Data Capture
Scylla CDC streams row-level changes into a **readable CQL table** (`<table>_scylla_cdc_log`) — query it with ordinary CQL, no external connector required, and integrate via the Scylla CDC source connectors (Kafka).
```sql
CREATE TABLE orders (...) WITH cdc = {'enabled': true, 'ttl': 86400};
-- consume changes:
SELECT * FROM orders_scylla_cdc_log WHERE "cdc$stream_id" = ? ;
```
Use CDC for outbox/event-sourcing and downstream sync instead of dual-writes.

### E. Operational deltas
- **Compaction**: same strategies as Cassandra (owned by `cassandra.md`); Scylla adds **Incremental Compaction Strategy (ICS)** (Enterprise) to cut STCS space amplification — prefer it for large STCS-style workloads.
- **Repairs/backups**: use **ScyllaDB Manager** (`sctool`) for scheduled repair (row-level) and backup to object storage — never run manual `nodetool repair`.
- **No JVM**: ignore Cassandra heap/GC/`jvm.options` guidance entirely.

---

## 5. Errors, Retries & Consistency Binding

Strategy is owned by [`error-handling.md`](guides://error-handling.md); consistency semantics by [`cassandra.md`](guides://cassandra.md). Scylla binding:
- Make writes **idempotent** (deterministic IDs/TTLs) so timeouts can be retried safely.
- Use a retry policy that retries on the **next host** for `Unavailable`/`WriteTimeout` (for idempotent ops) and surfaces non-retryable errors.
- Prefer `LOCAL_QUORUM`; only fall back to a weaker CL deliberately and never silently below quorum for critical writes. Speculative execution can trim tail latency for read-heavy idempotent paths.

---

## 6. Monitoring & Operations

Metrics/tracing **policy** is owned by [`observability.md`](guides://observability.md). Scylla binding — the **ScyllaDB Monitoring Stack** (Prometheus + Grafana + bundled dashboards):

```bash
git clone https://github.com/scylladb/scylla-monitoring && cd scylla-monitoring
./start-all.sh -d data_dir            # Prometheus + Grafana + Alertmanager + dashboards
```
- Scrape **per-shard** metrics — a node-level average hides a single hot shard (the most common Scylla pathology).
- Watch: `scylla_storage_proxy_coordinator_*` p99 read/write latency, per-shard CPU/reactor utilization, pending/active compactions, cache hit ratio, tablet migrations, foreground/background scheduler shares.
- Manager (`sctool`): schedule weekly repair and regular backups; verify with `sctool task list` (SCYLLA-OPS-01).

```bash
# nodetool (diagnostics only — not for routine repair)
nodetool status                 # cluster/ownership
nodetool tablehistograms ks tbl # partition size & latency distribution
nodetool toppartitions ks tbl   # find hot partitions (hot shards)
```

---

## 7. When to Choose Scylla over Cassandra (honest guidance)

**Choose Scylla when:** you need predictable low p99 at high throughput; GC pauses hurt you on Cassandra/JVM; you want fewer nodes for the same load (denser hardware, shard-per-core); you want autonomous tuning + integrated Manager/Monitoring; you need elastic scaling (tablets) or a DynamoDB-compatible on-prem store (Alternator).

**Stay on / choose Cassandra when:** you depend on a Cassandra-only feature or integration not yet in Scylla; your team's tooling/operational expertise is deeply Cassandra-specific; you need the broader Cassandra ecosystem/connectors for a niche feature. Migration is usually low-friction because CQL, SSTables, and drivers are compatible — but **validate feature parity for your exact version** rather than trusting headline throughput numbers.

**Neither, if** access patterns aren't known in advance / you need ad-hoc joins or strong multi-row transactions → use a relational store ([`postgresql.md`](guides://postgresql.md)) or a different model.

---

## 8. Deployment Checklist

Generated from §2 — one box per requirement ID.

- [ ] SCYLLA-TST-01 — schema/query changes test-first; data bugs have regression tests
- [ ] SCYLLA-DRV-01 — shard-aware driver, token + DC-aware LB, shard connections observed
- [ ] SCYLLA-DRV-02 — prepared/parameterized CQL only, no string-built queries
- [ ] SCYLLA-MODEL-01 — query-driven schema, partitions bounded (see `cassandra.md`)
- [ ] SCYLLA-TBL-01 — tablets enabled on new keyspaces (or justified)
- [ ] SCYLLA-CONS-01 — LOCAL_QUORUM + idempotent retry policy
- [ ] SCYLLA-SEC-01 — RBAC auth + client/inter-node TLS on
- [ ] SCYLLA-SEC-02 — credentials/endpoints externalized, 0 high/critical CVEs
- [ ] SCYLLA-OBS-01 — Monitoring Stack scraping per-shard metrics, p99 alerts
- [ ] SCYLLA-OPS-01 — repairs & backups scheduled via ScyllaDB Manager
- [ ] Agent calibrated I/O scheduler (`scylla_setup`/iotune) and verified per-shard latency

---
**End of ScyllaDB Guidelines**
