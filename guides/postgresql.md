# PostgreSQL Development Guidelines
Mandatory standards for PostgreSQL schema design, query optimization, and operation. PostgreSQL 17, psql, pg_stat_statements, PgBouncer.

---
name: postgresql
title: PostgreSQL Development Guidelines
version: 2.0
last_reviewed: 2026-06-05
kind: datastore
tools: [postgresql@17, psql, pg_stat_statements, pgbouncer, pgvector, pg_partman]
requires:
  - sql
  - secure-coding
  - error-handling
recommends:
  - observability
  - performance
  - docker-compose
  - env-config
  - sqlalchemy-alembic
  - timescaledb
provides:
  - postgres-jsonb
  - postgres-indexing
  - mvcc-vacuum
  - postgres-extensions
  - rls
  - skip-locked
---

> 🧭 Authored per [`CONVENTIONS.md`](guides://CONVENTIONS.md): shared concerns are referenced, not restated. This guide covers only what is unique to PostgreSQL.

---

## 0. Prerequisites & References

Fetch and apply these **before** generating PostgreSQL code. Their rules are assumed below and not repeated.

> 📎 **REQUIRED — fetch & apply first:**
> - [`sql.md`](guides://sql.md) — relational design, normalization, joins, ANSI isolation levels, set operations. *(This guide binds PG specifics on top; it does NOT restate general SQL.)*
> - [`secure-coding.md`](guides://secure-coding.md) — least privilege, secrets, supply chain. *(PG binding: roles, RLS, `scram-sha-256`, TLS — §6.)*
> - [`error-handling.md`](guides://error-handling.md) — error strategy & propagation. *(PG binding: SQLSTATE classes, constraint violations, serialization-failure retries — §7.)*

> 📎 **RECOMMENDED — fetch when the task touches them:**
> - [`observability.md`](guides://observability.md) — metrics/monitoring policy *(binding: `pg_stat_statements`, `pg_stat_*` views — §8)*
> - [`performance.md`](guides://performance.md) — perf methodology *(binding: EXPLAIN ANALYZE, planner stats — §4)*
> - [`env-config.md`](guides://env-config.md) — connection/config policy *(binding: DSN, `sslmode`, PgBouncer — §9)*
> - [`sqlalchemy-alembic.md`](guides://sqlalchemy-alembic.md) — common Python access/migration layer
> - [`docker-compose.md`](guides://docker-compose.md) — local Postgres + PgBouncer stack
> - [`timescaledb.md`](guides://timescaledb.md) — time-series extension on top of Postgres

> 📎 **SEE ALSO:** [`tdd.md`](guides://tdd.md) *(pgTAP / app-level tests)* · [`sqlc.md`](guides://sqlc.md) · [`go.md`](guides://go.md) · [`python.md`](guides://python.md)

---

## 1. Core Philosophies: POSTGRES-FIRST

PostgreSQL-specific principles only. SQL design, security, and error strategy come from §0.

- **P**lan-verified: every non-trivial query is checked with `EXPLAIN (ANALYZE, BUFFERS)` before delivery; no Seq Scan on a large table where an index applies.
- **O**wned types: use PG's rich type system (`BIGINT … IDENTITY`, `TIMESTAMPTZ`, `NUMERIC`, `JSONB`, arrays, ranges, enums, `INET`, `UUID`) — never the lossy fallback.
- **S**ecured rows: sensitive/multi-tenant tables run with Row-Level Security and `FORCE ROW LEVEL SECURITY`; apps connect via least-privilege roles, never superuser.
- **T**ransaction-aware: pick the right isolation level, use `SKIP LOCKED` for queues, and treat serialization failures as retryable (see `error-handling.md`).
- **G**roomed storage: respect MVCC — monitor dead tuples, tune autovacuum per-table, avoid bloat and XID wraparound.
- **R**eplicable: schema and access designed for logical replication, HA, and connection pooling (PgBouncer transaction mode).
- **E**xtensions: reach for the ecosystem (`pg_stat_statements`, `pg_trgm`, `pgvector`, PostGIS, `pg_cron`, `pg_partman`) instead of reinventing.
- **S**tandard DDL: `GENERATED ALWAYS AS IDENTITY`, declarative partitioning, generated columns — never legacy `SERIAL`.

**Verified Code**: Agent-generated schema and queries MUST pass every gate in §2 before delivery.

---

## 2. Requirements (MANDATORY, auditable)

RFC-2119 keywords. IDs `PG-<TOPIC>-<NN>`. Each row has a binary gate; rows binding a shared rule cite its owner.

| ID | Requirement | Verify | Gate |
|----|-------------|--------|------|
| PG-TST-01 | Schema changes & functions MUST be test-first (pgTAP or app-level) (see `tdd.md`) | run test suite | exit 0, 0 skips |
| PG-TST-02 | Each bug MUST get a regression test before the fix (see `tdd.md`) | run test suite | failing→passing |
| PG-TYP-01 | Columns MUST use exact/native types: `BIGINT … IDENTITY` PK, `TIMESTAMPTZ`, `NUMERIC` for money, `JSONB` not `JSON`, `INET`, enum — never `SERIAL`/`FLOAT` money/`TIMESTAMP` | grep DDL / review | no banned types |
| PG-IDX-01 | Queries on large tables MUST be index-backed (right access method) | `EXPLAIN` plan | no unjustified Seq Scan |
| PG-PERF-01 | Non-trivial queries MUST be plan-checked (see `performance.md`) | `EXPLAIN (ANALYZE, BUFFERS)` | no red flags (§4) |
| PG-SEC-01 | Sensitive/multi-tenant tables MUST have RLS enabled + `FORCE` (see `secure-coding.md`) | query `pg_class.relrowsecurity` | RLS on all PII tables |
| PG-SEC-02 | App roles MUST be least-privilege (no SUPERUSER/CREATEDB); TLS + `scram-sha-256` enforced (see `secure-coding.md`) | `\du`, `pg_hba.conf` review | 0 over-privileged logins |
| PG-SEC-03 | No hardcoded credentials/DSN; config from env (see `env-config.md`) | grep / review | no literals |
| PG-TXN-01 | Serialization/deadlock failures (SQLSTATE 40001/40P01) MUST be retried (see `error-handling.md`) | code review | retry wrapper present |
| PG-VAC-01 | High-churn tables MUST have tuned autovacuum; bloat & XID age monitored | `pg_stat_user_tables` | dead_pct in budget |
| PG-MIG-01 | All DDL MUST ship as versioned, reviewed migrations | migration tool status | applied & in sync |
| PG-DOC-01 | Tables/columns SHOULD carry `COMMENT ON` (see `comments.md`) | `\d+` | comments present |

> **Forbidden**: `SERIAL`/`BIGSERIAL` in new schema, `FLOAT`/`real` for money, `JSON` where `JSONB` fits, session-level `SET` for RLS context under pooling, `VACUUM FULL` on a live hot table, app login as superuser, deploying DDL outside a migration.

---

## 3. Verification Protocol

Run before presenting code. Fix → re-run until green.

```sql
EXPLAIN (FORMAT TEXT) <query>;                 -- PG-IDX/PERF: parses + valid plan
EXPLAIN (ANALYZE, BUFFERS, SETTINGS) <query>;  -- PG-PERF-01: inspect for §4 red flags
SELECT relname FROM pg_class                    -- PG-SEC-01: RLS audit
  WHERE relkind='r' AND relrowsecurity=false AND relname = ANY(<sensitive_tables>);
SELECT rolname FROM pg_roles                    -- PG-SEC-02: privilege audit
  WHERE rolsuper OR rolcreatedb;
```

The *why* behind each gate lives in its §0 owner; do not re-derive it here.

---

## 4. Query Planning & Optimization

PG-specific tuning. General join/normalization theory is owned by [`sql.md`](guides://sql.md); methodology by [`performance.md`](guides://performance.md).

### A. Reading EXPLAIN
Always use `EXPLAIN (ANALYZE, BUFFERS)` (add `SETTINGS`, `VERBOSE`, `WAL`, or `FORMAT JSON` as needed). Compare **estimated vs actual rows** and multiply `actual time × loops`.

Red flags → fix:
- Seq Scan on a large table → add/choose index (§5).
- `rows` estimate far from `actual` → stale stats, run `ANALYZE` (raise `default_statistics_target` for skewed columns).
- Nested Loop with huge `loops` → missing index or wrong join; expect Hash/Merge Join.
- `Sort Method: external merge` → raise `work_mem` or add an ordering index.
- `Buffers: shared read` ≫ `shared hit` → cold cache / I/O bound.

### B. Idioms that keep indexes usable
- Replace `col OR col2` with `UNION` of two index scans; replace `NOT IN (subquery)` with `NOT EXISTS`.
- Don't wrap an indexed column in a function — add a matching **expression index** instead (§5).
- Keyset (seek) pagination over `OFFSET` for deep pages: `WHERE (created_at, id) < ($1,$2) ORDER BY created_at DESC, id DESC LIMIT n`.
- CTEs are inlined since PG12; use `WITH MATERIALIZED` to fence an expensive shared subquery, `NOT MATERIALIZED` to force push-down. Always bound `RECURSIVE` CTEs (depth/cycle).
- **Materialized views** for expensive aggregates; add a UNIQUE index so you can `REFRESH … CONCURRENTLY` (schedule via `pg_cron`).
- Mark pure SQL/plpgsql functions `IMMUTABLE`/`STABLE` and `PARALLEL SAFE` so the planner can index/parallelize them. Parallel query is disabled for SERIALIZABLE, cursors, and writes to the target table.

---

## 5. Data Types & Indexing (`provides: postgres-indexing`)

### A. Type choices
```sql
id          BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY  -- not SERIAL; blocks manual IDs
id          UUID DEFAULT gen_random_uuid()                   -- distributed generation
amount      NUMERIC(12,2)                                    -- money; never FLOAT
created_at  TIMESTAMPTZ NOT NULL DEFAULT now()               -- never naive TIMESTAMP
ip          INET            -- ranges/CIDR aware
status      order_status    -- CREATE TYPE … AS ENUM (…); add values with ALTER TYPE
tags        TEXT[]          -- small lists; query with @>, &&, ANY
period      TSTZRANGE       -- ranges; enforce non-overlap via EXCLUDE … USING gist
prefs       JSONB NOT NULL DEFAULT '{}'                      -- never JSON
total       NUMERIC GENERATED ALWAYS AS (qty * price) STORED -- generated column
```
Override identity only for migration: `INSERT … OVERRIDING SYSTEM VALUE`. Use `GENERATED BY DEFAULT AS IDENTITY` when imports must set IDs.

### B. Index access methods — pick by shape
| Access method | Use for |
|---|---|
| **B-tree** (default) | equality, ranges, `ORDER BY`, `IN`, `IS NULL` |
| **Hash** | equality only on large values |
| **GIN** | `JSONB` (`@>`,`?`), arrays, `tsvector` FTS, `pg_trgm`. `jsonb_path_ops` = smaller/faster but `@>` only |
| **GiST** | geometry/PostGIS, range types, exclusion constraints, KNN (`<->`) |
| **BRIN** | append-only huge tables where the column correlates with physical order (tiny index) |

```sql
CREATE INDEX ix_orders_user_status ON orders(user_id, status, created_at DESC);  -- composite: order matches WHERE/ORDER BY
CREATE INDEX ix_orders_cover ON orders(user_id) INCLUDE (status, total_amount);  -- covering / index-only scan
CREATE INDEX ix_orders_pending ON orders(created_at) WHERE status='pending';     -- partial
CREATE UNIQUE INDEX uq_users_email_live ON users(email) WHERE deleted_at IS NULL;-- partial unique (soft delete)
CREATE INDEX ix_users_email_lower ON users(LOWER(email));                        -- expression
CREATE INDEX ix_logs_ts_brin ON logs USING brin(created_at) WITH (pages_per_range=32);
```
Build hot indexes with `CREATE INDEX CONCURRENTLY` to avoid write locks. Drop unused indexes (`pg_stat_user_indexes.idx_scan = 0`, excluding constraint-backing ones).

---

## 6. JSONB (`provides: postgres-jsonb`)

Use JSONB for variable/sparse schema, external payloads, or blob-read data. Use **normalized columns** when a field is filtered/joined/aggregated, needs a FK, or a unique constraint. Anti-pattern: JSONB only to dodge migrations — if you filter a field in most queries, promote it to a column.

```sql
profile->>'name'                       -- text; ->  keeps JSONB; #>>'{a,b}' path text
profile @> '{"settings":{"theme":"dark"}}'   -- containment (GIN-indexable)
profile ? 'name'  /  ?| ARRAY[…]  /  ?& ARRAY[…]   -- key existence
jsonb_path_query(profile,'$.age ? (@ > 25)')       -- SQL/JSON path (PG12+)
JSON_VALUE / JSON_QUERY / JSON_EXISTS / JSON_TABLE -- SQL/JSON standard (PG16/17)
```
Index: `USING gin(profile)` for full operator set; `gin(profile jsonb_path_ops)` for `@>`-only; expression index `((profile->>'name'))` for equality on a known field; `JSON_TABLE` to shred JSON into rows.
Modify: `jsonb_set(profile,'{a,b}','"x"')`, merge `profile || '{"k":v}'`, delete `profile - 'k'` / `profile #- '{a,b}'`.

---

## 7. Transactions, Concurrency & Queues (`provides: skip-locked`)

ANSI isolation semantics are owned by [`sql.md`](guides://sql.md); error/retry strategy by [`error-handling.md`](guides://error-handling.md). PG bindings:

- Default is **Read Committed**; use `REPEATABLE READ` for a stable snapshot, `SERIALIZABLE` for true isolation — the latter can abort with **SQLSTATE 40001**; deadlocks raise **40P01**. Both MUST be caught and retried with backoff (PG-TXN-01).
- **Row locks**: `FOR UPDATE` / `FOR NO KEY UPDATE`; `NOWAIT` to fail fast.
- **Job queue pattern** — concurrent workers pull without contention:
```sql
SELECT id FROM jobs
WHERE status='pending'
ORDER BY created_at
FOR UPDATE SKIP LOCKED
LIMIT 1;
```
- **Upserts**: `INSERT … ON CONFLICT (key) DO UPDATE SET … / DO NOTHING`. For multi-action (insert+update+delete) use **`MERGE`** (PG15+, with conditional `WHEN MATCHED AND …`).
- **Advisory locks** for app-level mutual exclusion: `pg_advisory_xact_lock(key)` (auto-released at commit) or `pg_try_advisory_lock` (non-blocking). Note: session-level advisory locks don't survive PgBouncer transaction pooling.

---

## 8. MVCC, Vacuum & Monitoring (`provides: mvcc-vacuum`, `postgres-extensions`)

PostgreSQL keeps row versions (MVCC); `UPDATE`/`DELETE` leave dead tuples that **autovacuum** must reclaim. Monitoring policy is owned by [`observability.md`](guides://observability.md); PG bindings:

```sql
-- bloat & vacuum freshness
SELECT relname, n_live_tup, n_dead_tup,
       round(100.0*n_dead_tup/NULLIF(n_live_tup+n_dead_tup,0),2) AS dead_pct,
       last_autovacuum FROM pg_stat_user_tables ORDER BY n_dead_tup DESC;

-- XID wraparound risk
SELECT relname, age(relfrozenxid) FROM pg_class
WHERE relkind='r' AND age(relfrozenxid) > 100000000 ORDER BY 2 DESC;
```
- Default scale factor (0.2) vacuums too late on big tables — tune per-table: `ALTER TABLE orders SET (autovacuum_vacuum_scale_factor=0.01, autovacuum_vacuum_threshold=1000);`
- `VACUUM ANALYZE` reclaims + refreshes stats; `VACUUM FULL` rewrites and **locks** (use `pg_repack` online instead). Watch `pg_stat_progress_vacuum`.
- **`pg_stat_statements`** (set in `shared_preload_libraries`) is mandatory for prod: rank by `total_exec_time`, `mean_exec_time` (calls>100), and cache-hit ratio (`shared_blks_hit` vs `read`). Reset periodically.
- Size/usage: `pg_total_relation_size`, `pg_stat_user_indexes` (find unused indexes).

### Extensions to reach for
`pg_trgm` (fuzzy `ILIKE`/`similarity`, GIN `gin_trgm_ops` or GiST KNN) · **`pgvector`** (embeddings; `vector(n)` with HNSW `vector_cosine_ops` or IVFFlat; operators `<->` L2, `<#>` inner, `<=>` cosine) · **PostGIS** (geography/geometry, `gist`, `ST_DWithin`, KNN) · `pg_cron` (in-DB scheduling) · `pg_partman` (auto partition lifecycle). Audit extension provenance per [`secure-coding.md`](guides://secure-coding.md).

---

## 9. Partitioning, Replication, Pooling & Security

### A. Declarative partitioning
Partition when a table exceeds ~100M rows / 100GB **and** queries filter on the key; skip for small tables or cross-partition unique needs.
```sql
CREATE TABLE events (…, created_at TIMESTAMPTZ NOT NULL) PARTITION BY RANGE (created_at);
CREATE TABLE events_2026_01 PARTITION OF events FOR VALUES FROM ('2026-01-01') TO ('2026-02-01');
CREATE TABLE events_default PARTITION OF events DEFAULT;
```
Also `PARTITION BY LIST (region)` / `HASH (id)`. Maintenance: `DETACH PARTITION … CONCURRENTLY`, then archive/drop; automate with `pg_partman` + `pg_cron`. For time-series, prefer [`timescaledb.md`](guides://timescaledb.md) hypertables (`time_bucket`, continuous aggregates, retention/compression policies).

### B. Logical replication & HA
```sql
-- wal_level = logical
CREATE PUBLICATION pub FOR TABLE orders WHERE (region='us-east');   -- row filter (PG15+), column lists too
CREATE SUBSCRIPTION sub CONNECTION '…' PUBLICATION pub;
ALTER SUBSCRIPTION sub SET (streaming='parallel');                 -- PG16+ parallel apply
```
Monitor lag via `pg_replication_slots` (`pg_wal_lsn_diff`). Physical: `archive_mode=on` + WAL archiving / streaming replicas for PITR & failover.

### C. Connection pooling — PgBouncer
Front the DB with **PgBouncer** (apps connect to it, not Postgres). `pool_mode=transaction` is the default for web apps — but forbids session state: no cross-transaction prepared statements, `LISTEN/NOTIFY`, session advisory locks, or session-level `SET`. Size: `max_connections ≈ cores×4` (SSD); don't over-provision (~10MB/conn). Watch `SHOW POOLS` (`cl_waiting`, `maxwait` near 0). App pool config (DSN, `sslmode`) belongs in env per [`env-config.md`](guides://env-config.md); SQLAlchemy/psycopg pools are covered in [`sqlalchemy-alembic.md`](guides://sqlalchemy-alembic.md).

### D. Security bindings (`provides: rls`)
Policy is owned by [`secure-coding.md`](guides://secure-coding.md). PG specifics:
```sql
ALTER TABLE documents ENABLE ROW LEVEL SECURITY;
ALTER TABLE documents FORCE ROW LEVEL SECURITY;   -- owners bypass RLS otherwise
CREATE POLICY tenant_isolation ON documents FOR ALL
  USING (tenant_id = current_setting('app.tenant_id')::bigint)
  WITH CHECK (tenant_id = current_setting('app.tenant_id')::bigint);
-- per-transaction context (pooling-safe):
BEGIN; SET LOCAL app.tenant_id = '456'; … COMMIT;   -- NEVER plain SET under pooling
```
Roles: build a `NOLOGIN` privilege hierarchy (`app_readonly` → `app_readwrite` → `app_admin`) with `ALTER DEFAULT PRIVILEGES`; grant to `LOGIN` service roles; set `CONNECTION LIMIT`/`VALID UNTIL`. Use column-level `GRANT (col,…)` or views to hide PII. Enforce TLS (`hostssl … scram-sha-256` in `pg_hba.conf`; `sslmode=verify-full` in DSN).

---

## 10. Migrations & Tooling

All DDL ships as **versioned, reviewed migrations** (PG-MIG-01) — Alembic ([`sqlalchemy-alembic.md`](guides://sqlalchemy-alembic.md)), Flyway, Liquibase, Atlas, or sqlc-adjacent tooling. Commit schema/lockfiles; never edit production schema by hand. Run zero-downtime DDL with `CONCURRENTLY` (indexes), `NOT VALID` + later `VALIDATE CONSTRAINT`, and additive-then-backfill column changes. Local dev/CI stacks (Postgres + PgBouncer) via [`docker-compose.md`](guides://docker-compose.md).

---

## 11. Quick Reference

```sql
EXPLAIN (ANALYZE, BUFFERS) <query>;                         -- plan check
SELECT * FROM pg_stat_activity;                             -- sessions
SELECT pg_terminate_backend(<pid>);                         -- kill backend
INSERT … ON CONFLICT (k) DO UPDATE SET …;                   -- upsert
SELECT … FOR UPDATE SKIP LOCKED LIMIT 1;                    -- queue pull
MERGE INTO t USING s ON t.id=s.id WHEN MATCHED THEN UPDATE … WHEN NOT MATCHED THEN INSERT …;
SELECT * FROM JSON_TABLE(doc,'$.items[*]' COLUMNS (id INT PATH '$.id'));  -- PG17
```

---

## 12. Deployment Checklist

Generated from §2 — one box per requirement ID.

- [ ] PG-TST-01/02 — schema/functions test-first; bugs have regression tests
- [ ] PG-TYP-01 — native/exact types only (IDENTITY, TIMESTAMPTZ, NUMERIC, JSONB)
- [ ] PG-IDX-01 — large-table queries index-backed, correct access method
- [ ] PG-PERF-01 — `EXPLAIN (ANALYZE, BUFFERS)` shows no red flags
- [ ] PG-SEC-01 — RLS enabled + FORCE on sensitive/multi-tenant tables
- [ ] PG-SEC-02 — least-privilege roles; TLS + scram-sha-256 enforced
- [ ] PG-SEC-03 — no hardcoded credentials/DSN
- [ ] PG-TXN-01 — serialization/deadlock retries in place
- [ ] PG-VAC-01 — autovacuum tuned; bloat & XID age monitored
- [ ] PG-MIG-01 — all DDL via versioned, reviewed migrations
- [ ] PG-DOC-01 — tables/columns commented
- [ ] Agent ran every §3 command and documented any fixes

---
**End of PostgreSQL Guidelines**
