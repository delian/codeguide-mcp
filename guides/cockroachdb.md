# CockroachDB Development Guidelines
Mandatory standards for CockroachDB: distributed-SQL schema design, hotspot-free keys, SERIALIZABLE transaction-retry handling, and multi-region topology. CockroachDB 24.x, cockroach CLI, DB Console, EXPLAIN ANALYZE.

---
name: cockroachdb
title: CockroachDB Development Guidelines
version: 2.0
last_reviewed: 2026-06-05
kind: datastore
tools: [cockroachdb@24.3, cockroach-cli, dbconsole, "explain-analyze"]
requires:
  - sql
  - secure-coding
  - error-handling
recommends:
  - observability
  - performance
  - postgresql
  - kubernetes
provides:
  - distributed-sql
  - serializable-retries
  - multi-region
  - hotspot-avoidance
  - hash-sharded-indexes
---

> 🧭 Authored per [`CONVENTIONS.md`](guides://CONVENTIONS.md): shared concerns are referenced, not restated. This guide covers only what is unique to CockroachDB as a distributed SQL datastore.

---

## 0. Prerequisites & References

Fetch and apply these **before** writing CockroachDB schema or queries. Their rules are assumed below and not repeated.

> 📎 **REQUIRED — fetch & apply first:**
> - [`sql.md`](guides://sql.md) — general SQL: normalization, query design, joins, CTEs, window functions, transaction theory. *(CRDB binding: write standard ANSI/PostgreSQL SQL; this guide only covers the distributed divergences.)*
> - [`secure-coding.md`](guides://secure-coding.md) — secrets, supply chain, CVE policy. *(CRDB binding: TLS client-cert auth, never `--insecure` in prod, parameterized queries, no credentials in connection strings.)*
> - [`error-handling.md`](guides://error-handling.md) — error strategy & propagation. *(CRDB binding: the `40001` serialization-failure retry loop in §6 is mandatory — a CRDB transaction can be told to retry and the application owns that loop.)*

> 📎 **RECOMMENDED — fetch when the task touches them:**
> - [`postgresql.md`](guides://postgresql.md) — CockroachDB speaks the PostgreSQL wire protocol; reuse pg drivers/tools, but see §3 for where behaviour **diverges**.
> - [`performance.md`](guides://performance.md) · [`observability.md`](guides://observability.md) — query latency, plan analysis, metrics. *(CRDB binding: `EXPLAIN ANALYZE`, Statements page, Prometheus endpoint.)*
> - [`kubernetes.md`](guides://kubernetes.md) — common deploy target (CockroachDB Operator / Helm / StatefulSet).

> 📎 **SEE ALSO:** [`tdd.md`](guides://tdd.md) · [`sqlalchemy-alembic.md`](guides://sqlalchemy-alembic.md) · [`kafka.md`](guides://kafka.md) *(changefeed sink)* · [`terraform.md`](guides://terraform.md)

---

## 1. Core Philosophies: DISTRIBUTED-FIRST

CockroachDB-specific principles only. SQL design, security, and error strategy come from §0.

- **D**istributed by default: every table is split into ~512 MiB **ranges**, each Raft-replicated (default 3×) across nodes. Design for data that lives on many machines, not one.
- **I**dempotent retries: SERIALIZABLE is the default isolation; conflicting transactions are aborted with SQLSTATE `40001` and **must be retried by the application**. Every write transaction is wrapped in a retry loop (§6).
- **S**hard the hot key: monotonic primary keys (sequences, `now()`, `unique_rowid()`) funnel all writes to one range = a hotspot. Default to random UUIDs or **hash-sharded** keys (§4, §5).
- **T**opology-aware: in multi-region clusters, pin data with `LOCALITY` (regional-by-row / regional-by-table / global) and a survival goal — latency is a schema decision (§7).
- **R**ead the plan: confirm index usage and avoid full-cluster scans with `EXPLAIN ANALYZE` before shipping a query (policy: `performance.md`).
- **I**ntegrity over denormalization: rely on real ACID transactions and foreign keys; denormalize only with measured justification.

**Verified Code**: Agent-generated schema and queries MUST pass every gate in §2 before delivery.

---

## 2. Requirements (MANDATORY, auditable)

RFC-2119 keywords. IDs `CRDB-<TOPIC>-<NN>`. Each row has a binary gate; rows binding a shared rule cite its owner.

| ID | Requirement | Verify | Gate |
|----|-------------|--------|------|
| CRDB-TST-01 | Schema/query changes MUST be test-first against a real CRDB (see `tdd.md`) | run the suite (`cockroach demo` / testcontainers) | exit 0, 0 skips |
| CRDB-TXN-01 | Every write transaction MUST be wrapped in a `40001` retry loop (see `error-handling.md`) | grep/review; inject a forced retry | retries, then succeeds |
| CRDB-TXN-02 | Code MUST NOT assume a transaction commits without retry; no client-side `BEGIN…COMMIT` without retry handling | review | no bare multi-stmt txns |
| CRDB-PK-01 | High-write tables MUST NOT use a sequential/monotonic primary key | `SHOW CREATE TABLE`; review | UUID or hash-sharded |
| CRDB-PK-02 | Monotonic access patterns (timestamps, FKs) MUST use a hash-sharded index where they are the leading key | `SHOW INDEXES`; review | `USING HASH` present |
| CRDB-IDX-01 | Hot-path queries MUST use an index, not a full scan | `EXPLAIN ANALYZE <query>` | no `full scan` on hot path |
| CRDB-MR-01 | Multi-region DBs MUST declare `PRIMARY REGION`, a survival goal, and per-table `LOCALITY` | `SHOW REGIONS`; `SHOW CREATE TABLE` | all set explicitly |
| CRDB-SCHEMA-01 | Schema changes MUST use online DDL; no app-blocking migrations | `SHOW JOBS`; review | job completes, no lock wait |
| CRDB-COMPAT-01 | No reliance on PG features CRDB diverges on without verifying (see §3) | run DDL/DML on target version | executes on 24.x |
| CRDB-SEC-01 | Prod clusters & clients MUST use TLS certs; never `--insecure` (see `secure-coding.md`) | inspect start flags / connstring | `sslmode=verify-full` |
| CRDB-SEC-02 | No secrets in connection strings/SQL; 0 high/critical CVEs in drivers (see `secure-coding.md`) | secret scan; `pip-audit`/`npm audit`/`govulncheck` | 0 findings |
| CRDB-OBS-01 | Clusters MUST export metrics & SLO alerts (see `observability.md`) | scrape `/_status/vars`; check alerts | Prometheus target up |
| CRDB-BACKUP-01 | Managed/self-hosted clusters MUST have scheduled backups + a tested restore | `SHOW SCHEDULES`; restore drill | backup green, restore ok |

> **Forbidden**: shipping a write transaction without a retry loop; a sequential PK on a high-throughput table; `--insecure` or plaintext credentials in production; a query whose plan shows a full scan on a hot path; a multi-region table with no explicit `LOCALITY`.

---

## 3. PostgreSQL Wire Compatibility — and Where It Diverges

CockroachDB speaks the **PostgreSQL wire protocol**: connect with `psql`, pgx, psycopg, JDBC, SQLAlchemy, pgAdmin — driver code is unchanged. General SQL is owned by [`sql.md`](guides://sql.md); pg-specific idioms by [`postgresql.md`](guides://postgresql.md). What follows is **only the divergence** you must design around.

| Area | PostgreSQL | CockroachDB 24.x | Action |
|---|---|---|---|
| Default isolation | READ COMMITTED | **SERIALIZABLE** (READ COMMITTED is opt-in, GA in 23.2+) | Implement retry loops (§6) |
| Topology | single primary | distributed ranges + Raft | Design keys for distribution (§4) |
| `SERIAL`/sequences | fast, monotonic | works but **hotspots**; `unique_rowid()` is non-contiguous | Use `UUID`/hash-shard (§4) |
| Stored procedures / UDFs / triggers | long-standing | UDFs (22.2+), **stored procedures (24.1+)**, **triggers (24.3+)** now exist but feature-incomplete | Verify on target version before relying |
| `gen_random_uuid()` / `JSONB` / arrays | yes | yes | Portable |
| Foreign keys / CHECK / `ON DELETE CASCADE` | yes | yes (online) | Portable |
| Extensions (PostGIS, pgvector, etc.) | rich ecosystem | spatial & vector are **built-in**; arbitrary C extensions are **not** loadable | Don't assume `CREATE EXTENSION <arbitrary>` |
| Perf characteristics | single-node latency | every write = cross-node Raft consensus; small txns favoured, chatty round-trips punished | Batch; co-locate; use follower reads (§8) |

> **Rule:** treat CRDB as PostgreSQL-*compatible*, not PostgreSQL-*identical*. Run your DDL/DML against the **target CRDB version** in CI — compatibility is per-feature and per-version (CRDB-COMPAT-01).

Type note: CockroachDB's canonical string type is `STRING` (alias of `VARCHAR`/`TEXT`); `INT` defaults to 64-bit (`INT8`). Prefer `DECIMAL` for money, `TIMESTAMPTZ` over `TIMESTAMP`, `JSONB` over `JSON`.

---

## 4. Schema & Primary-Key Design (hotspot avoidance)

The single most important CRDB design decision is the **primary key**, because it determines how rows map to ranges and therefore where write load lands.

**Default: random UUID** — spreads writes uniformly across ranges.
```sql
CREATE TABLE users (
    id    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email STRING UNIQUE NOT NULL,
    name  STRING NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
```

**Avoid: monotonic keys on high-write tables** — a `SERIAL`/sequence/`now()` leading key sends every new row to the *last* range = a single-range write bottleneck.
```sql
-- ❌ all inserts hammer one range
CREATE TABLE events (id SERIAL PRIMARY KEY, ...);
```

**When you need ordered/sequential access — hash-sharded** (§5) so the key is still range-distributed while remaining indexable by value.

**Composite keys** co-locate related rows for efficient range scans and as the basis for `REGIONAL BY ROW` partitioning:
```sql
CREATE TABLE metrics (
    tenant STRING,
    ts     TIMESTAMPTZ,
    name   STRING,
    value  FLOAT,
    PRIMARY KEY (tenant, ts, name)   -- scans within a tenant stay on few ranges
);
```

**Schema evolution is online** (CRDB-SCHEMA-01): `ADD COLUMN`, `ADD COLUMN … DEFAULT`, `CREATE INDEX`, FK changes, type changes run as background **jobs** without taking the table offline. Track them with `SHOW JOBS`. Prefer FKs with `ON DELETE CASCADE` to co-locate related data (the old *interleaved tables* feature was removed in v21.1 — do not use it).

---

## 5. Indexes & Hash-Sharded Indexes

Index design is owned by [`sql.md`](guides://sql.md)/[`performance.md`](guides://performance.md); the CRDB-specific tools:

- **`STORING`/covering indexes** — include non-key columns so the query is answered from the index alone, avoiding a primary-index lookup:
  ```sql
  CREATE INDEX idx_orders_user ON orders (user_id) STORING (status, total);
  ```
- **Hash-sharded indexes** — the CRDB answer to write hotspots on monotonic columns (timestamps, FK fan-in, sequential IDs). It prepends a computed shard column so writes spread across `bucket_count` ranges while range scans by value still work:
  ```sql
  -- modern 24.x syntax (bucket_count optional; auto-tuned if omitted)
  CREATE INDEX ON orders (created_at) USING HASH WITH (bucket_count = 16);

  -- or a hash-sharded PRIMARY KEY for a sequential id
  CREATE TABLE ledger (
      id INT8 PRIMARY KEY USING HASH,
      amount DECIMAL(18,2) NOT NULL,
      created_at TIMESTAMPTZ NOT NULL DEFAULT now()
  );
  ```
  > Legacy syntax `USING HASH WITH BUCKET_COUNT = 8` is deprecated — use `WITH (bucket_count = N)`.
- **Partial & expression indexes** are supported and reduce index size for selective predicates.

Always confirm the optimizer actually uses the index with `EXPLAIN ANALYZE` (CRDB-IDX-01) — a plan showing `full scan` on a large table is a defect.

---

## 6. Transactions, Isolation & the 40001 Retry Loop

**This is the section that most distinguishes CockroachDB from single-node SQL.** Transaction theory is owned by [`sql.md`](guides://sql.md); the error-handling contract by [`error-handling.md`](guides://error-handling.md). The CRDB binding:

CockroachDB is **SERIALIZABLE by default**. To preserve serializability it may abort a transaction at `COMMIT` (or mid-flight) with SQLSTATE **`40001` (`serialization_failure`, "restart transaction")**. This is **expected, not an error to surface** — the application MUST retry the whole transaction (with backoff), because earlier statements may now be invalid.

```python
# psycopg / psycopg2 — own the retry loop
import time, psycopg2
from psycopg2 import errorcodes

def run_txn(conn, op, max_retries=5):
    for attempt in range(max_retries):
        try:
            with conn.cursor() as cur:
                op(cur)
            conn.commit()
            return
        except psycopg2.errors.SerializationFailure:   # SQLSTATE 40001
            conn.rollback()
            if attempt == max_retries - 1:
                raise
            time.sleep(0.1 * 2 ** attempt)             # exponential backoff
        except psycopg2.Error:
            conn.rollback()
            raise                                       # non-retryable: propagate

def transfer(cur):
    cur.execute("UPDATE accounts SET balance = balance - 100 WHERE id = 'alice'")
    cur.execute("UPDATE accounts SET balance = balance + 100 WHERE id = 'bob'")
```

Prefer the **maintained helpers** that implement this loop for you:
- Python/SQLAlchemy: `sqlalchemy-cockroachdb`'s `run_transaction(...)`.
- Go: `github.com/cockroachdb/cockroach-go/v2/crdb` (`crdb.ExecuteTx`).
- Node: `@cockroachdb/...` retry wrappers or a hand-written loop keyed on `code === '40001'`.

Rules:
- **Keep transactions small and short** — fewer statements = fewer conflicts = fewer retries.
- **Do not put non-idempotent side effects** (emails, external calls) inside the retried block; the body can run multiple times.
- `READ COMMITTED` (GA 23.2+) reduces retries for contended workloads but **weakens guarantees** — opt in deliberately with `SET TRANSACTION ISOLATION LEVEL READ COMMITTED`, never silently.
- Use `SELECT … FOR UPDATE` to take explicit locks and reduce serialization conflicts on read-modify-write paths; `SAVEPOINT`/`ROLLBACK TO SAVEPOINT` for nested partial rollback.

---

## 7. Multi-Region Topology

Native multi-region is a core CRDB differentiator. Declare topology with high-level **abstractions** (preferred over manual zone configs).

**1. Add regions & pick a survival goal** (latency vs. fault tolerance trade-off):
```sql
ALTER DATABASE app SET PRIMARY REGION 'us-east1';
ALTER DATABASE app ADD REGION 'us-west1';
ALTER DATABASE app ADD REGION 'europe-west1';

ALTER DATABASE app SURVIVE ZONE FAILURE;     -- default; tolerate an AZ loss (3+ nodes/region)
-- or
ALTER DATABASE app SURVIVE REGION FAILURE;   -- tolerate a whole region loss (needs 3+ regions; costlier writes)
```
Nodes are tagged at startup with `--locality=region=us-east1,zone=us-east1-a`.

**2. Set table `LOCALITY`** — choose per access pattern:

| Locality | Use when | Cost |
|---|---|---|
| `REGIONAL BY ROW` | rows belong to a home region (per-user/per-tenant data) | fast local R/W in home region; cross-region access slower |
| `REGIONAL BY TABLE IN '<region>'` | whole table is owned by one region | fast in that region only |
| `GLOBAL` | read-mostly reference data read everywhere | fast reads everywhere; writes pay cross-region consensus |

```sql
-- per-row homing; CRDB manages the hidden crdb_region column
ALTER TABLE users SET LOCALITY REGIONAL BY ROW;
ALTER TABLE catalog SET LOCALITY GLOBAL;
ALTER TABLE eu_invoices SET LOCALITY REGIONAL BY TABLE IN 'europe-west1';
```

Prefer these abstractions over hand-rolled `PARTITION BY LIST … CONFIGURE ZONE USING constraints/lease_preferences`; drop to raw zone configs only for needs the abstractions can't express. CockroachDB automatically moves leaseholders toward query load ("follow-the-workload") — no config needed.

---

## 8. Performance, Reads & Distribution

Policy is owned by [`performance.md`](guides://performance.md). CRDB-specific levers:

- **Follower reads** — read a slightly-stale (~few s) value from the nearest replica, offloading the leaseholder. Ideal for catalogs/analytics:
  ```sql
  SELECT * FROM products AS OF SYSTEM TIME follower_read_timestamp() WHERE active;
  ```
- **`AS OF SYSTEM TIME`** (time-travel) — query historical data within the GC window (default 25h, set by `gc.ttlseconds`): audits, point-in-time recovery, bounded-staleness reads (`AS OF SYSTEM TIME '-30s'`).
- **Batch & co-locate** — each write costs a Raft round-trip; prefer multi-row `INSERT`/`UPSERT`, keep related rows on the same ranges (composite keys / regional-by-row), and minimize cross-range transactions.
- **Inspect** — `EXPLAIN ANALYZE` for plans + actual row counts; the DB Console **Statements** page and `crdb_internal.node_statement_statistics` for hot/slow queries; `SHOW RANGES FROM TABLE t` to see distribution and spot hot ranges.
- **Connection pooling is mandatory** — CRDB connections are not free; use a pooler (PgBouncer or the driver's pool) sized to the cluster.

CockroachDB is **not** an OLAP/analytics engine, a document store, or a graph DB — route those workloads to the right tool (see §0 of the relevant datastore guides). Storage engine is **Pebble** (the RocksDB era ended in v21).

---

## 9. Operations: Security, Backup, Observability, Deploy

These bind cross-cutting owners to CRDB; do not restate the owners.

- **Security** (`secure-coding.md`): run with TLS certs — node + client certs via `cockroach cert create-*`; clients use `sslmode=verify-full`. **Never `--insecure` in production.** Use RBAC (`CREATE ROLE`, `GRANT`), keep the DB Console (`:8080`) and SQL port (`:26257`) behind a VPN/bastion, enable enterprise audit logging where required, and encrypt at rest. No credentials in connection strings or committed config (CRDB-SEC-01/02).
- **Backup & recovery**: `BACKUP INTO 's3://…'` + `CREATE SCHEDULE` for periodic full+incremental; restore with `RESTORE FROM LATEST IN '…'`. A backup is only valid once a **restore has been drilled** (CRDB-BACKUP-01). CRDB self-heals from node loss (Raft re-replication) — backups guard against logical/operator errors, not node failure.
- **Observability** (`observability.md`): scrape Prometheus metrics from `/_status/vars` (and `/metrics`), watch the DB Console, and alert on SLOs (p99 latency, retry rate, replica under-replication, disk). **Changefeeds** (CDC) stream row changes to Kafka/cloud storage for downstream consumers: `CREATE CHANGEFEED FOR TABLE orders INTO 'kafka://…' WITH format=json, diff` (see [`kafka.md`](guides://kafka.md)).
- **Deployment** (`kubernetes.md`): the common target is the **CockroachDB Kubernetes Operator** or Helm chart over a StatefulSet with persistent volumes and a headless service; otherwise `cockroach start --join=…` across nodes and a load balancer (HAProxy/cloud LB) in front of `:26257`. CockroachDB Cloud (Serverless/Dedicated) is the managed option. Upgrade one node at a time (drain → upgrade binary → rejoin), then finalize.

---

## 10. Migration & Drivers (compat checklist)

Migrating from PostgreSQL/MySQL (policy in [`sql.md`](guides://sql.md)):
- Replace `SERIAL`/`AUTO_INCREMENT` PKs with `UUID DEFAULT gen_random_uuid()` or hash-sharded keys (§4) — this is the #1 source of hotspots.
- Re-verify any **triggers / stored procedures / UDFs**: these now exist (24.x) but are feature-incomplete vs. PostgreSQL — test on the target version (CRDB-COMPAT-01).
- Use `IMPORT INTO` / `COPY` for bulk load; for zero-downtime cutover use **dual-write + backfill**, validate row counts/checksums, then flip reads.
- Drivers are the standard PG ones (psycopg, pgx, JDBC, SQLAlchemy via `sqlalchemy-cockroachdb`) — wire-protocol compatible; just add the retry loop (§6).

---

## 11. Quick Reference

```bash
cockroach demo                                   # local in-memory cluster for tests
cockroach sql --certs-dir=certs --host=<n>:26257 # SQL shell (TLS)
cockroach node status --certs-dir=certs          # cluster/node health
cockroach cert create-client <user> --certs-dir=certs --ca-key=ca.key
cockroach node drain <id> --certs-dir=certs      # graceful drain for maintenance/upgrade
```
```sql
EXPLAIN ANALYZE <query>;                          -- plan + actual rows (CRDB-IDX-01)
SHOW RANGES FROM TABLE t;                          -- data distribution / hotspots
SHOW JOBS;                                          -- track online schema changes
SHOW REGIONS FROM DATABASE app;                    -- multi-region topology
BACKUP INTO 's3://bucket/app?AUTH=implicit';       -- full backup
```

---

## 12. Deployment Checklist

Generated from §2 — one box per requirement ID. No new requirements here.

- [ ] CRDB-TST-01 — schema/query changes test-first against a real CRDB
- [ ] CRDB-TXN-01/02 — every write transaction has a `40001` retry loop; no bare multi-statement txns
- [ ] CRDB-PK-01/02 — no monotonic PK on high-write tables; hash-sharded where monotonic access is needed
- [ ] CRDB-IDX-01 — hot-path queries index-backed (`EXPLAIN ANALYZE`, no full scans)
- [ ] CRDB-MR-01 — multi-region DBs declare PRIMARY REGION + survival goal + table LOCALITY
- [ ] CRDB-SCHEMA-01 — schema changes run as online jobs, no app-blocking locks
- [ ] CRDB-COMPAT-01 — PG divergences verified on the target CRDB version
- [ ] CRDB-SEC-01/02 — TLS certs, no `--insecure`, no secrets in connstrings, 0 high/critical CVEs
- [ ] CRDB-OBS-01 — metrics scraped, SLO alerts configured
- [ ] CRDB-BACKUP-01 — scheduled backups + a tested restore
- [ ] Agent ran every verification command and documented any fixes

---
**End of CockroachDB Guidelines**
