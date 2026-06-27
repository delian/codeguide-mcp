# Apache Cassandra Development Guidelines
Mandatory standards for Apache Cassandra: query-driven data modeling, bounded partitions, tunable consistency, and workload-matched compaction. Cassandra 5.0, CQL, cqlsh, nodetool, SAI.

---
name: cassandra
title: Apache Cassandra Development Guidelines
version: 2.0
last_reviewed: 2026-06-05
kind: datastore
tools: [cassandra@5.0, cql, cqlsh, nodetool, sai]
requires:
  - secure-coding
  - error-handling
recommends:
  - observability
  - performance
  - scylladb
  - env-config
provides:
  - wide-column-model
  - query-driven-modeling
  - partition-clustering-keys
  - tunable-consistency
  - compaction-strategies
---

> 🧭 Authored per [`CONVENTIONS.md`](guides://CONVENTIONS.md): shared concerns are referenced, not restated. This guide covers only what is unique to Cassandra.

> **Cassandra vs ScyllaDB:** [ScyllaDB](guides://scylladb.md) is a C++ (Seastar) rewrite of Cassandra that is **CQL- and driver-compatible** — the data modeling, consistency, and compaction rules below apply to both. Scylla owns only its shard-per-core differences; design with this guide, then read `scylladb.md` if deploying on Scylla.

---

## 0. Prerequisites & References

Fetch and apply these **before** designing schemas or writing CQL. Their rules are assumed below and not repeated.

> 📎 **REQUIRED — fetch & apply first:**
> - [`secure-coding.md`](guides://secure-coding.md) — auth, RBAC, secrets, injection, CVE policy. *(Cassandra binding: `PasswordAuthenticator` + `CassandraAuthorizer`, role least-privilege, client/internode TLS, parameterized prepared statements — §9.)*
> - [`error-handling.md`](guides://error-handling.md) — error strategy & propagation. *(Cassandra binding: consistency-level choice, idempotent retries, timeout/unavailable handling, speculative execution — §6, §8.)*

> 📎 **RECOMMENDED — fetch when the task touches them:**
> - [`observability.md`](guides://observability.md) — metrics/tracing policy *(binding: `nodetool`, JMX/Prometheus metrics, query tracing — §9)*
> - [`performance.md`](guides://performance.md) — perf policy *(binding: partition sizing, compaction throughput, JVM/G1GC heap — §5, §7)*
> - [`scylladb.md`](guides://scylladb.md) — the drop-in C++ rewrite; read when choosing the engine or porting
> - [`env-config.md`](guides://env-config.md) — config policy *(binding: contact points/credentials in env/secret store, never hardcoded)*

> 📎 **SEE ALSO:** [`tdd.md`](guides://tdd.md) (test schema + CQL against a Compose/`ccm` cluster; regression-test data bugs before fixing) · [`kafka.md`](guides://kafka.md) (CDC / event ingestion) · [`docker-compose.md`](guides://docker-compose.md) · [`kubernetes.md`](guides://kubernetes.md) (K8ssandra / cass-operator)

---

## 1. Core Philosophies: CASSANDRA-FIRST

Cassandra-specific principles only. Security, error handling, observability come from §0.

- **C**onsistency is tunable, not free: pick R/W levels per query so `R + W > RF` where you need strong reads; default `LOCAL_QUORUM`.
- **A**ccess pattern is the schema: model one table **per query**, denormalize, duplicate freely. There are **no joins** — the read shape dictates the table.
- **S**ingle-partition reads: every hot query must hit exactly one partition; multi-partition scans and `ALLOW FILTERING` are red flags, not options.
- **S**ize the partition: keep partitions bounded (target < 100 MB, < 100k rows); time-bucket or add to the partition key before they grow unbounded or go hot.
- **A**nti-entropy on schedule: deletes create tombstones — run `repair` inside `gc_grace_seconds` or deleted data resurrects.
- **N**o relational reflexes: masterless ring, eventual consistency, immutable LSM storage. Prefer inserts + TTL over update/delete churn.
- **D**istribute by design: `NetworkTopologyStrategy`, RF ≥ 3, rack/DC awareness — no single point of failure.

**Verified Code**: Agent-generated schemas and CQL MUST pass every gate in §2 before delivery.

---

## 2. Requirements (MANDATORY, auditable)

RFC-2119 keywords. IDs `CASS-<TOPIC>-<NN>`. Each row has a binary gate; rows binding a shared rule cite its owner.

| ID | Requirement | Verify | Gate |
|----|-------------|--------|------|
| CASS-MODEL-01 | Every table MUST be designed for a specific query; no joins/normalized-then-joined access (denormalize) | design review vs §3 | one table per query |
| CASS-MODEL-02 | Every production query MUST restrict the full partition key; no `ALLOW FILTERING` | grep CQL / trace | no `ALLOW FILTERING`, partition key bound |
| CASS-PART-01 | Partitions MUST be bounded (< 100 MB, < ~100k rows); unbounded time-series MUST be bucketed | `nodetool tablestats` / `tablehistograms` | max partition < 100 MB |
| CASS-CONS-01 | RF ≥ 3 with `NetworkTopologyStrategy`; never `SimpleStrategy`/RF 1 in prod | `DESCRIBE KEYSPACE` | NTS, RF ≥ 3 |
| CASS-CONS-02 | Reads/writes MUST use an explicit consistency level (default `LOCAL_QUORUM`); `R+W>RF` where strong (see `error-handling.md`) | driver config / trace | explicit CL, no `ANY` |
| CASS-COMP-01 | Compaction strategy MUST match workload (UCS default; TWCS for time-series+TTL) | `DESCRIBE TABLE` | strategy justified |
| CASS-TOMB-01 | `gc_grace_seconds` MUST be honored by scheduled repair; no queue/delete-heavy patterns | repair schedule + `tombstone_warn_threshold` | repair < gc_grace, no warns |
| CASS-IDX-01 | Indexing MUST prefer a denormalized table or SAI; legacy 2i only for low-cardinality local lookups | `DESCRIBE TABLE` / review | SAI or table, justified |
| CASS-LWT-01 | LWT (`IF`/Paxos) used only for genuine compare-and-set; never as a perf default | review | scoped to CAS |
| CASS-SEC-01 | Auth + authorization on, TLS (client + internode), least-privilege roles, prepared statements (see `secure-coding.md`) | `cassandra.yaml` / `LIST ROLES` | auth/TLS on, scoped roles |
| CASS-CFG-01 | Contact points & credentials from env/secret store, never hardcoded (see `env-config.md`) | grep source | no literals |
| CASS-OBS-01 | Cluster + table metrics exported and alerted (see `observability.md`) | JMX/Prometheus exporter live | metrics scraped |

> **Forbidden**: `ALLOW FILTERING` on a hot path; unbounded partitions; `SimpleStrategy` or RF 1 in production; `CONSISTENCY ANY` (silent data loss); using Cassandra as a queue (tombstone storm); multi-partition `IN`/logged batches as a "performance" trick; LWT as the default write path; secondary index on a high-cardinality column; hardcoded contact points/credentials.

---

## 3. Data Modeling — query-driven & wide-column (the central skill)

Cassandra is a **wide-column** store: a *partition* (one partition-key value) holds many *rows*, each an ordered set of *cells*, distributed across the ring by hashing the partition key. You model **for queries**, not for entities — denormalize and store the same data in as many tables as you have query shapes.

**The method:** list the application's queries first → design one table per query so each is a single-partition read → choose the partition key for even distribution + access locality → choose clustering keys for on-disk sort order. Joins, subqueries, and ad-hoc `WHERE` do not exist; if a new query appears, you add a new table (or a [materialized view](#8-indexing--search), §8).

```cql
-- Query: "latest posts by a user." Single-partition, no join, user data denormalized in.
CREATE TABLE posts_by_user (
    user_id    UUID,
    created_at TIMESTAMP,
    post_id    TIMEUUID,
    username   TEXT,        -- denormalized copy; duplication is expected
    title      TEXT,
    body       TEXT,
    PRIMARY KEY ((user_id), created_at, post_id)
) WITH CLUSTERING ORDER BY (created_at DESC, post_id ASC);

SELECT * FROM posts_by_user WHERE user_id = ? LIMIT 20;   -- newest 20, one partition
```

Data types worth knowing: `UUID`/`TIMEUUID` (the latter encodes time and sorts), `TIMESTAMP`/`DATE`/`TIME`, `DECIMAL` for money, collections `SET`/`LIST`/`MAP` (keep < ~100 items — each is read whole and edits create tombstones), UDTs (`FROZEN<...>`), and `VECTOR<FLOAT, n>` for embeddings (§8). Prefer writing immutable rows with TTL over updating in place.

---

## 4. Partition & Clustering Key Design

`PRIMARY KEY ((partition_key…), clustering_col…)`. The **partition key** is the unit of distribution *and* of access (one query → ideally one partition). The **clustering columns** define row order within the partition and enable range/slice queries.

**Partition key goals:** (1) spread data evenly across the ring, (2) make each query a single-partition read, (3) keep the partition bounded.

```cql
-- ❌ Unbounded: one partition per sensor grows forever, eventually hot & huge
PRIMARY KEY (sensor_id, ts)

-- ✅ Time-bucketed composite partition key: each partition is one day, bounded
CREATE TABLE sensor_data_by_day (
    sensor_id TEXT,
    day       DATE,                       -- bucket → bounds the partition
    ts        TIMESTAMP,
    value     DOUBLE,
    PRIMARY KEY ((sensor_id, day), ts)
) WITH CLUSTERING ORDER BY (ts DESC);

SELECT * FROM sensor_data_by_day
WHERE sensor_id = ? AND day = '2026-06-27'
  AND ts >= ? AND ts < ?;                 -- range on clustering col, one partition
```

Footguns: **hot partitions** (low-cardinality or monotonic partition key concentrates load on one node) → add to the composite key or bucket; **large partitions** degrade reads/compaction → bucket; clustering columns can only be range-queried left-to-right (no skipping). Check sizes with `nodetool tablestats <ks>.<table>` and `tablehistograms` (CASS-PART-01).

---

## 5. CQL — SQL-like, NOT SQL

CQL borrows SQL syntax but is a different model. What it does **not** have: joins, subqueries, `OR`, arbitrary `WHERE`, aggregations across partitions, or referential integrity. `WHERE` must constrain the partition key (and then clustering columns in order).

- `ALLOW FILTERING` makes a forbidden query *run* by scanning — it is a **red flag**, not a fix. It signals the table is wrong for the query; build a new table or a SAI index instead (CASS-MODEL-02).
- `INSERT` and `UPDATE` are both **upserts** (last-write-wins by timestamp) — no "row exists" error without LWT.
- Always use **prepared statements** with bind parameters — never string-concatenate values (perf + injection, see `secure-coding.md`).

```cql
-- ❌ full-cluster scan, unpredictable latency
SELECT * FROM users WHERE age > 25 ALLOW FILTERING;
-- ✅ model the query: a table (or SAI index) keyed for it
```

`BATCH` is for **atomicity within a single partition**, not throughput. A multi-partition logged batch is *slower* than concurrent async writes and stresses the coordinator — use the driver's async/`execute_concurrent` API for bulk loads (CASS forbidden list).

---

## 6. Architecture, Consistency & the Read/Write Path

**Masterless ring.** Every node is a peer; data is placed by **consistent hashing** of the partition key over a token ring, gossip propagates membership. Any node can be the **coordinator** for a request and forwards to the replicas. `NetworkTopologyStrategy` places RF replicas across racks/DCs; **RF ≥ 3** in production (CASS-CONS-01).

**Tunable consistency.** Per query you choose how many replicas must ack. Strong consistency holds when `R + W > RF`. With RF = 3, `LOCAL_QUORUM` (2) for both read and write gives strong consistency in the local DC while tolerating one node down — the production default.

| Level | Meaning | Use |
|-------|---------|-----|
| `LOCAL_QUORUM` | quorum in local DC | **production default** |
| `QUORUM` | quorum across all DCs | cross-DC strong consistency |
| `EACH_QUORUM` | quorum in *every* DC | multi-DC strong writes |
| `LOCAL_ONE` / `ONE` | one replica | read-heavy, eventual-OK |
| `ALL` | all replicas | avoid (no fault tolerance) |
| `ANY` | hinted-handoff counts | **never** (silent data loss) |

Consistency, idempotent retry, and timeout/`Unavailable` handling are an error-handling concern — see [`error-handling.md`](guides://error-handling.md); the binding is: make writes idempotent so retries are safe, and tune driver speculative execution rather than blanket-retrying non-idempotent ops.

**Write path (LSM):** write → append to **commit log** (durability) + **memtable** (memory) → ack. Memtable flushes to an immutable **SSTable** on disk. **Compaction** later merges SSTables, drops overwritten cells and expired tombstones.

**Read path:** merge the memtable + relevant SSTables (bloom filters/partition index narrow the set), reconcile by write timestamp. On a quorum read, the coordinator detects divergent replicas and issues a **read repair**. Background `nodetool repair` (anti-entropy) is still required to converge cold data (§9).

---

## 7. Compaction Strategies (pick per workload)

Compaction governs read amplification, space amplification, and tombstone reclamation — choosing it per table is a first-class modeling decision (CASS-COMP-01).

| Strategy | Best for | Notes |
|----------|----------|-------|
| **UCS** (UnifiedCompactionStrategy) | general default on 5.0 | runtime-tunable via `scaling_parameters` (`T4` balanced, `L4` read-opt, `N` write-opt) — no rewrite to retune |
| **TWCS** (TimeWindow) | time-series **with TTL** | data grouped by time window; whole expired SSTables dropped cheaply — the right call for metrics/events/logs |
| **LCS** (Leveled) | read-heavy, update-heavy | low read amplification, higher write/IO cost; on 5.0 prefer UCS |
| **STCS** (SizeTiered) | write-heavy / legacy default | low write cost, high read amplification; migrate to UCS |

```cql
-- Time-series: TWCS + table-level TTL → cheap expiry, no tombstone scan storms
CREATE TABLE events_by_day (...)
WITH compaction = {'class':'TimeWindowCompactionStrategy',
                   'compaction_window_unit':'DAYS','compaction_window_size':1}
 AND default_time_to_live = 2592000;     -- 30 days

-- General table on 5.0
... WITH compaction = {'class':'UnifiedCompactionStrategy','scaling_parameters':'T4'};
```

Monitor with `nodetool compactionstats`; compaction throughput tuning is a perf concern (`performance.md`).

---

## 8. Indexing & Search

**Default to a denormalized table** for any new access path — it is always a single-partition read. Reach for an index only when duplication is impractical.

- **SAI — Storage-Attached Indexing (5.0, preferred).** A genuine improvement over legacy 2i: multiple SAI indexes per table, numeric range + text (optionally case-insensitive) + collection predicates, far lower disk overhead, AND across indexes. Still queries across the cluster, so keep it secondary to good partitioning.
  ```cql
  CREATE INDEX ON products (category) USING 'sai';
  CREATE INDEX ON products (name) USING 'sai' WITH OPTIONS = {'case_sensitive':'false'};
  SELECT * FROM products WHERE category = 'Electronics' AND price > 100 AND price < 1000;
  ```
- **Legacy secondary index (2i) — usually wrong.** Local per node (scatter-gather across all nodes), bad on high-cardinality columns, no ranges. Acceptable only for low-cardinality local lookups; otherwise use SAI or a table (CASS-IDX-01).
- **Materialized views — caution.** Server-maintained denormalized tables; convenient but add write amplification and are eventually consistent (and can drift). Prefer SAI or an app-maintained table; reserve MVs for stable, well-understood views.
- **Vector search (5.0).** `VECTOR<FLOAT, n>` + a SAI index enables ANN similarity for RAG/embeddings:
  ```cql
  CREATE TABLE documents (id UUID PRIMARY KEY, content TEXT, embedding VECTOR<FLOAT, 1536>);
  CREATE INDEX ON documents (embedding) USING 'sai';
  SELECT id, content FROM documents ORDER BY embedding ANN OF ? LIMIT 10;
  ```

---

## 9. Cassandra-Specific Operations: LWT, Counters, TTL, Tombstones

- **Lightweight transactions (LWT).** `IF NOT EXISTS` / `IF col = ?` give linearizable compare-and-set via **Paxos** — 3–4× the latency of a normal write (extra round trips) and contention-prone. Use *only* for genuine CAS (unique registration, optimistic locking), never as a default; single-partition only. Read with `SERIAL`/`LOCAL_SERIAL`. Cassandra 4.1+ Paxos v2 needs periodic `nodetool repair --paxos-only` (CASS-LWT-01).
- **Counters.** `counter` columns are a distinct type with their own (non-idempotent) write path — keep them in dedicated tables, never mix with regular columns, and never expect exactly-once under retries.
- **TTL.** Per-write or `default_time_to_live` expires data automatically — the idiomatic way to bound time-series. Expiry creates a tombstone, so pair TTL with **TWCS** so whole expired SSTables drop without tombstone scans.
- **Tombstones & the `gc_grace_seconds` footgun.** Deletes (and TTL expiry, and `null` writes, and collection updates) write **tombstones**, not in-place removals. They persist for `gc_grace_seconds` (default 10 days) so deletes propagate to all replicas; if `repair` does not run within that window, **deleted data resurrects**. Reading across many tombstones is slow and can fail at `tombstone_failure_threshold`. **Never build a queue** on Cassandra (delete-after-read = tombstone storm). Schedule repair (Reaper or `nodetool repair`) inside the gc window (CASS-TOMB-01).

Security (`secure-coding.md`): enable `PasswordAuthenticator` + `CassandraAuthorizer`, grant least-privilege roles (`GRANT SELECT/MODIFY ON ...`), rotate the default `cassandra` superuser, enable client + internode TLS, and use prepared statements (no concatenated CQL). 5.0 adds dynamic data masking (`MASKED WITH`) for PII.

Observability (`observability.md`): expose JMX → Prometheus (`org.apache.cassandra.metrics`), watch p99 read/write latency, pending compactions, dropped messages, hinted handoffs, and tombstone scans; debug single queries with `TRACING ON`. Operate via `nodetool` (`status`, `tablestats`, `tpstats`, `compactionstats`, `repair`).

---

## 10. Cassandra 5.0 Highlights & Engine Choice

- **SAI** replaces legacy 2i (§8) and powers **vector search**.
- **UCS** is the general-purpose, runtime-tunable compaction default (§7).
- **Trie memtables & BTI (trie-indexed) SSTables** — smaller indexes, faster reads, less GC pressure; default in 5.0, run `nodetool upgradesstables` after upgrade.
- **JDK 17**, dynamic data masking, and stronger guardrails (table/keyspace/page-size thresholds in `cassandra.yaml`).
- **Engine choice:** if you need the same CQL/data model with lower tail latency and higher per-node throughput, evaluate [`scylladb.md`](guides://scylladb.md) — a drop-in C++ rewrite. The modeling discipline here is identical.

---

## 11. Deployment Checklist

Generated from §2 — one box per requirement ID. No new requirements here.

- [ ] CASS-MODEL-01 — one table per query, denormalized (no joins)
- [ ] CASS-MODEL-02 — every prod query bounds the partition key; no `ALLOW FILTERING`
- [ ] CASS-PART-01 — partitions bounded (< 100 MB), time-series bucketed
- [ ] CASS-CONS-01 — `NetworkTopologyStrategy`, RF ≥ 3
- [ ] CASS-CONS-02 — explicit consistency (default `LOCAL_QUORUM`), `R+W>RF` where strong
- [ ] CASS-COMP-01 — compaction matches workload (UCS / TWCS)
- [ ] CASS-TOMB-01 — repair scheduled inside `gc_grace_seconds`; no queue pattern
- [ ] CASS-IDX-01 — denormalized table or SAI; 2i only low-cardinality
- [ ] CASS-LWT-01 — LWT scoped to genuine compare-and-set
- [ ] CASS-SEC-01 — auth/authz + TLS + least-privilege roles + prepared statements
- [ ] CASS-CFG-01 — contact points/credentials from env/secret store
- [ ] CASS-OBS-01 — cluster/table metrics exported and alerted

---
**End of Apache Cassandra Guidelines**
