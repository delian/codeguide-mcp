# MySQL & MariaDB Development Guidelines
Mandatory standards for MySQL/MariaDB OLTP design: InnoDB-first schemas, utf8mb4, correct indexing, safe transactions, and replication. MySQL 8.4 LTS, MariaDB 11.x, mysqldump/mydumper, pt-toolkit.

---
name: mysql-mariadb
title: MySQL & MariaDB Development Guidelines
version: 2.0
last_reviewed: 2026-06-05
kind: datastore
tools: [mysql@8.4-lts, mariadb@11.4, mysqldump, mydumper, percona-toolkit, proxysql]
requires:
  - sql
  - secure-coding
  - error-handling
recommends:
  - observability
  - performance
  - docker-compose
  - env-config
provides:
  - innodb
  - mysql-indexing
  - utf8mb4
  - mysql-replication
  - mariadb-divergence
---

> 🧭 Authored per [`CONVENTIONS.md`](guides://CONVENTIONS.md): shared concerns are referenced, not restated. This guide covers only what is unique to MySQL and MariaDB. General relational design, normalization, and portable SQL live in [`sql.md`](guides://sql.md).

---

## 0. Prerequisites & References

Fetch and apply these **before** generating MySQL/MariaDB code. Their rules are assumed below and not repeated.

> 📎 **REQUIRED — fetch & apply first:**
> - [`sql.md`](guides://sql.md) — relational modeling, normalization, joins, ANSI SQL, migrations, generic EXPLAIN literacy. *(This guide adds only the MySQL/MariaDB-specific layer.)*
> - [`secure-coding.md`](guides://secure-coding.md) — injection prevention, secrets, least privilege. *(Binding: parameterized/prepared statements via the driver; `GRANT` least-privilege; never concatenate input.)*
> - [`error-handling.md`](guides://error-handling.md) — error strategy & propagation. *(Binding: deadlock `1213` / lock-wait-timeout `1205` retry with backoff.)*

> 📎 **RECOMMENDED — fetch when the task touches them:**
> - [`observability.md`](guides://observability.md) — metrics/tracing *(binding: `performance_schema`, slow query log, `pt-query-digest`)*
> - [`performance.md`](guides://performance.md) · [`env-config.md`](guides://env-config.md) *(binding: DSN/credentials from env, never hardcoded)* · [`docker-compose.md`](guides://docker-compose.md) *(local dev stacks)*

> 📎 **SEE ALSO:** [`tdd.md`](guides://tdd.md) (test-first schema/queries) · [`postgresql.md`](guides://postgresql.md) (when choosing an RDBMS) · [`redis.md`](guides://redis.md) (cache-aside; query cache is gone) · [`sqlc.md`](guides://sqlc.md) · [`sqlalchemy-alembic.md`](guides://sqlalchemy-alembic.md)

---

## 1. Core Philosophies: OLTP-FIRST

MySQL/MariaDB-specific principles only. TDD, security, error handling, and generic SQL come from §0.

- **O**nly InnoDB: every table `ENGINE=InnoDB`. No MyISAM in new code (no transactions, table locks, no crash recovery).
- **L**ean clustered PK: every table has a small, **monotonic** PK (`BIGINT UNSIGNED AUTO_INCREMENT` or ordered UUID) — InnoDB clusters rows by PK and copies it into every secondary index.
- **T**ext is utf8mb4: never `utf8` (= `utf8mb3`, no emoji/astral chars). `utf8mb4` charset + an explicit collation everywhere.
- **P**arameterize always: prepared/bound statements only (policy: `secure-coding.md`). No string-built SQL.
- **F**ail-safe transactions: InnoDB defaults to REPEATABLE READ + gap locks; keep transactions short, lock rows in a consistent order, and retry deadlocks.

**Verified Code**: Agent-generated SQL MUST pass every gate in §2 before delivery.

---

## 2. Requirements (MANDATORY, auditable)

RFC-2119 keywords. IDs `MY-<TOPIC>-<NN>`. Each row has a binary gate; rows binding a shared rule cite its owner.

| ID | Requirement | Verify | Gate |
|----|-------------|--------|------|
| MY-ENG-01 | Every table MUST be `ENGINE=InnoDB` | `SELECT table_name FROM information_schema.tables WHERE engine<>'InnoDB' AND table_schema=DATABASE()` | 0 rows |
| MY-CHR-01 | Every DB/table/column MUST be `utf8mb4` (never `utf8`/`utf8mb3`/`latin1`) | `SELECT * FROM information_schema.columns WHERE character_set_name NOT IN('utf8mb4',NULL) AND table_schema=DATABASE()` | 0 rows |
| MY-PK-01 | Every table MUST have an explicit, small, monotonic PRIMARY KEY | `SELECT t.table_name FROM information_schema.tables t LEFT JOIN information_schema.statistics s ON s.table_name=t.table_name AND s.index_name='PRIMARY' WHERE s.index_name IS NULL AND t.table_schema=DATABASE()` | 0 rows |
| MY-IDX-01 | Every FK and every WHERE/JOIN/ORDER-BY hot column MUST be index-covered; no full scans on hot paths | `EXPLAIN FORMAT=JSON <query>` | no `type: ALL` on hot path |
| MY-SEC-01 | All queries MUST be parameterized; no string-built SQL (see `secure-coding.md`) | code review / grep for f-string/concat SQL | none found |
| MY-SEC-02 | App user MUST have least privilege (no `SUPER`/`FILE`/`ALL`) (see `secure-coding.md`) | `SHOW GRANTS FOR CURRENT_USER()` | scoped to needed DB+verbs |
| MY-TXN-01 | Write transactions MUST be short and retry deadlock `1213`/lock-wait `1205` (see `error-handling.md`) | code review / chaos test | retry loop present |
| MY-MIG-01 | Schema changes MUST be reversible and run online (`ALGORITHM=INSTANT/INPLACE` or `pt-online-schema-change`) | migration dry-run | no blocking `COPY` on large tables |
| MY-CFG-01 | Credentials/DSN from env, never hardcoded (see `env-config.md`) | grep source | no literals |
| MY-BAK-01 | Backups MUST be tested-restorable; binlog enabled for PITR | restore drill | restore succeeds |
| MY-DIV-01 | Code MUST be portable across the target engine(s) or fence engine-specific syntax (see §8) | run suite on MySQL **and** MariaDB if both targeted | both pass |

> **Forbidden**: MyISAM in new code; `utf8`/`utf8mb3`; `FLOAT`/`DOUBLE` for money (use `DECIMAL`); functions wrapping indexed columns in WHERE; `OFFSET` deep pagination; ignoring deadlock errors; storing the query-cache-era `query_cache_*` settings (removed).

---

## 3. Verification Protocol

Run before presenting schema/queries. Fix → re-run until green.

```sql
EXPLAIN FORMAT=JSON <query>;          -- MY-IDX-01: inspect access type, key, rows, cost
SHOW CREATE TABLE <t>;                 -- MY-ENG-01/CHR-01: engine + charset + collation
SHOW ENGINE INNODB STATUS\G            -- MY-TXN-01: latest deadlock, lock waits
SHOW GRANTS FOR CURRENT_USER();        -- MY-SEC-02: least privilege
```
```bash
mysqldump --single-transaction --routines --triggers db | gzip > db.sql.gz   # MY-BAK-01 (logical)
pt-query-digest /var/log/mysql/slow.log    # observability: top offenders
pt-online-schema-change --alter "ADD COLUMN ..." D=db,t=tbl --execute        # MY-MIG-01
```
The *why* behind each gate lives in its §0 owner; do not re-derive it here.

---

## 4. InnoDB: the only engine

InnoDB is the canonical owned concern. Key mental model that drives every schema decision:

- **Clustered index = the table.** Rows are stored *in PK order* inside a B+tree. There is no heap. The PK choice is the single biggest physical-design decision.
- **Every secondary index stores the PK** as its row pointer. A wide PK (e.g. a random `CHAR(36)` UUID or a multi-column natural key) bloats *every* index and slows every lookup → prefer a narrow `BIGINT UNSIGNED AUTO_INCREMENT`.
- **Monotonic PKs avoid page splits.** Random PKs (v4 UUID) scatter inserts across the tree, causing splits, fragmentation, and a cold buffer pool. If you need UUIDs, use a **time-ordered** form (`UUID_TO_BIN(UUID(), 1)` swaps the time fields; MariaDB 11 has a native `UUID` type that is byte-ordered) stored as `BINARY(16)`, not `CHAR(36)`.
- **Buffer pool is the cache.** `innodb_buffer_pool_size` ≈ 70–80% of RAM on a dedicated server is the most impactful tuning knob; aim for a buffer-pool hit ratio > 99%.
- **MVCC**: readers don't block writers. Undo logs serve consistent snapshots.

```sql
CREATE TABLE users (
  id         BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,   -- narrow, monotonic clustered key
  email      VARCHAR(255) NOT NULL,
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  UNIQUE KEY uq_email (email)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci ROW_FORMAT=DYNAMIC;
```

Engine config that matters (`my.cnf`): `innodb_buffer_pool_size`, `innodb_redo_log_capacity` (8.0.30+; was `innodb_log_file_size`), `innodb_flush_log_at_trx_commit=1` + `sync_binlog=1` (durable; use `2`/`0` only when you can lose ~1s on OS crash), `innodb_flush_method=O_DIRECT`, `innodb_io_capacity` matched to storage, `innodb_file_per_table=ON`. Per-connection buffers (`sort_buffer_size`, `join_buffer_size`) are allocated *per session* — keep small. Legacy engines: **MyISAM** is forbidden (no ACID, table locks); MariaDB **Aria** is a crash-safe MyISAM but still table-locked — use only for internal temp/system tables.

---

## 5. Data types & gotchas

MySQL's type system has sharp edges that no generic SQL guide covers:

- **`utf8mb4`, never `utf8`.** In MySQL `utf8` aliases `utf8mb3` (3-byte, no emoji). Set server `character_set_server=utf8mb4` and pick a collation: MySQL 8 default `utf8mb4_0900_ai_ci` (Unicode 9, accent/case-insensitive); MariaDB uses `utf8mb4_uca1400_ai_ci` (11.x) or `utf8mb4_unicode_ci`. A utf8mb4 `VARCHAR(255)` can be 1020 bytes — watch index key-length limits (3072 bytes for InnoDB).
- **`TIMESTAMP` vs `DATETIME`.** `TIMESTAMP` is 4 bytes, stored UTC, converted by `time_zone`, range 1970–2038 (the 2038 problem). `DATETIME` is 8 bytes, no timezone conversion, range 1000–9999. Rule: `TIMESTAMP` for "moment things happened" with `DEFAULT CURRENT_TIMESTAMP [ON UPDATE CURRENT_TIMESTAMP]`; `DATETIME` for far-future/historical/wall-clock values. Store UTC; convert at the edges.
- **`DECIMAL` for money** — never `FLOAT`/`DOUBLE` (binary rounding). `DECIMAL(12,2)`.
- **Implicit conversions are silent and dangerous.** Comparing a string column to a number (`WHERE phone = 123`) casts the *column* to a number, **disables the index**, and can match unexpected rows. `0 = 'abc'` is TRUE. Comparing different collations/charsets across a join forces a conversion that kills index use. Always compare like-for-like types.
- **Integers:** `BIGINT UNSIGNED` for IDs; `TINYINT(1)`/`BOOLEAN` for flags. The display-width `(N)` is cosmetic and deprecated in MySQL 8.
- **`ENUM`** is compact and fast but reordering values is a metadata change and cross-engine portability is poor; prefer a lookup table or `CHECK` constraint for evolving sets.
- **`JSON`**: native binary type (see §6).

---

## 6. JSON

MySQL has a **native binary `JSON`** type (validated, normalized, with path indexes). MariaDB's `JSON` is an *alias for `LONGTEXT`* with a `CHECK(JSON_VALID(...))` — same functions, different storage and performance; this is a key divergence (§8).

```sql
SELECT id, profile->>'$.city'  AS city,        -- ->> unquotes; -> keeps JSON
       JSON_EXTRACT(profile,'$.tags')          AS tags
FROM users
WHERE JSON_CONTAINS(profile->'$.roles', '"admin"');

UPDATE users SET profile = JSON_SET(profile,'$.city','Sofia') WHERE id = 1;
```

Index JSON via **generated columns** or **functional indexes** (MySQL 8 / MariaDB 10.5+):
```sql
ALTER TABLE users
  ADD COLUMN city VARCHAR(64)
    GENERATED ALWAYS AS (profile->>'$.city') STORED,   -- or VIRTUAL (computed on read)
  ADD INDEX idx_city (city);

-- MySQL 8: functional index directly
CREATE INDEX idx_city ON users ((CAST(profile->>'$.city' AS CHAR(64))));
-- MySQL 8.0.17+: multi-valued index over a JSON array
CREATE INDEX idx_tags ON products ((CAST(tags->'$[*]' AS CHAR(50) ARRAY)));
```
Use JSON for genuinely schemaless/sparse attributes — not as a substitute for normalized columns you query and join on. For document-heavy workloads, reconsider the datastore.

---

## 7. Indexing (canonical: `mysql-indexing`)

Generic index theory is in `sql.md`; the MySQL/InnoDB specifics:

- **Leftmost-prefix rule.** A composite index `(a, b, c)` serves `WHERE a`, `a,b`, `a,b,c`, and `ORDER BY a,b` — but **not** a query that starts at `b`. Order columns: equality predicates first, then the range/sort column last.
- **Covering indexes.** If the index contains every column the query reads, InnoDB never touches the clustered table — `EXPLAIN` shows `Using index`. Add trailing columns to turn a hot query covering.
- **Prefix indexes** for long strings: `INDEX (email(20))` indexes the first 20 chars — saves space but cannot be covering and weakens uniqueness/sorting.
- **Functional / expression indexes** (MySQL 8, MariaDB 10.5+): `CREATE INDEX ix ON t ((LOWER(email)))` — lets `WHERE LOWER(email)=?` use an index. The fix for the "function on indexed column disables the index" footgun.
- **Invisible indexes** (`ALTER TABLE t ALTER INDEX ix INVISIBLE`): optimizer ignores it but still maintains it — test-drop a suspected-unused index safely before `DROP`.
- **Descending indexes** (MySQL 8): `INDEX (created_at DESC)` truly stores descending for `ORDER BY ... DESC LIMIT`.
- **Don't index:** low-cardinality columns (boolean/status alone), tiny tables, or columns written far more than read. Find unused indexes via `sys.schema_unused_indexes` / `performance_schema.table_io_waits_summary_by_index_usage`.
- **Full-text** (`FULLTEXT ... MATCH ... AGAINST`) and **spatial** (`SPATIAL INDEX` + `ST_*`) indexes exist on InnoDB but are niche — for serious search/geo use a dedicated engine.

---

## 8. Query optimization & the optimizer

- **`EXPLAIN FORMAT=JSON`** is the primary tool — it shows `access_type`, chosen `key`, estimated `rows`, filtered %, and **`query_cost`**. Read `type` worst→best: `ALL` (full scan) → `index` → `range` → `ref` → `eq_ref` → `const`. `EXPLAIN ANALYZE` (MySQL 8.0.18+) runs the query and reports *actual* time/rows per iterator.
- **Index-friendly predicates:** never wrap an indexed column in a function (`YEAR(created_at)=2024` → use a half-open range `>= '2024-01-01' AND < '2025-01-01'`). Avoid leading `%` in `LIKE`.
- **Keyset (seek) pagination**, not `LIMIT n OFFSET big` — OFFSET scans and discards. Use `WHERE id > :last ORDER BY id LIMIT n`.
- **`SELECT` only needed columns** (enables covering indexes; avoids fetching off-page BLOB/TEXT).
- **Optimizer hints** when the plan is wrong: `/*+ JOIN_ORDER(...) */`, `/*+ INDEX(t ix) */`, `STRAIGHT_JOIN`. Keep `ANALYZE TABLE` stats fresh so the cost model is accurate. MySQL 8 has a histogram option (`ANALYZE TABLE ... UPDATE HISTOGRAM`) for skewed non-indexed columns.
- The **query cache is gone** (removed MySQL 8.0, MariaDB 10.6 default-off) — cache in the app (cache-aside with Redis/Memcached), not the server.
- **Window functions** (`ROW_NUMBER`/`RANK`/`SUM() OVER (PARTITION BY ... ORDER BY ...)`) and **CTEs** incl. `WITH RECURSIVE` are available (MySQL 8.0+, MariaDB 10.2+) — prefer them over correlated subqueries and emulated row-numbering.

---

## 9. Transactions, isolation & locking

- **Default isolation is REPEATABLE READ** (stricter than most engines' READ COMMITTED). InnoDB uses **gap locks / next-key locks** under RR to prevent phantoms, so range `UPDATE`/`SELECT ... FOR UPDATE` can lock rows that don't exist yet — a common deadlock source. Switch to `READ COMMITTED` for high-write OLTP if gap-lock contention hurts and phantoms are tolerable.
- **Row locks** (InnoDB, automatic) lock *index records*, not rows — an `UPDATE` without a usable index escalates to locking many records. Always update/delete via an indexed predicate.
- **Locking reads:** `SELECT ... FOR UPDATE` (exclusive), `FOR SHARE` (shared). `FOR UPDATE SKIP LOCKED` and `FOR UPDATE NOWAIT` (MySQL 8 / MariaDB 10.6) implement work-queue patterns without blocking.
- **Deadlocks are normal** — InnoDB detects them and rolls back the cheaper transaction with error `1213`; `1205` is lock-wait-timeout. Applications **MUST** retry with exponential backoff (policy: `error-handling.md`). Prevent them by locking rows in a consistent order and keeping transactions short.

```python
# Deadlock-safe transfer (driver-agnostic shape; parameterized per secure-coding.md)
for attempt in range(3):
    try:
        with conn.begin():
            a, b = sorted((from_id, to_id))                       # consistent lock order
            cur.execute("SELECT 1 FROM accounts WHERE id=%s FOR UPDATE", (a,))
            cur.execute("SELECT 1 FROM accounts WHERE id=%s FOR UPDATE", (b,))
            cur.execute("UPDATE accounts SET bal=bal-%s WHERE id=%s", (amt, from_id))
            cur.execute("UPDATE accounts SET bal=bal+%s WHERE id=%s", (amt, to_id))
        break
    except DeadlockError as e:                                    # errno 1213 / 1205
        if attempt == 2: raise
        time.sleep(0.1 * 2 ** attempt)
```
Inspect the last deadlock with `SHOW ENGINE INNODB STATUS\G` (LATEST DETECTED DEADLOCK).

---

## 10. Replication & HA

The canonical owned concern `mysql-replication`:

- **Binlog formats:** use `binlog_format=ROW` (deterministic, safe) — not `STATEMENT`. Enable **GTIDs** (`gtid_mode=ON`, `enforce_gtid_consistency=ON`) so replicas auto-position (`CHANGE REPLICATION SOURCE TO ... SOURCE_AUTO_POSITION=1`) and failover is sane.
- **Async (default):** primary doesn't wait for replicas → fast, but failover can lose the last committed transactions. Monitor lag (`SHOW REPLICA STATUS\G` → `Seconds_Behind_Source`; MySQL 8 renamed `MASTER`→`SOURCE`, `SLAVE`→`REPLICA`, kept old aliases).
- **Semi-synchronous:** primary waits for ≥1 replica to ack receipt (`rpl_semi_sync_source_enabled`) → bounded data loss at some latency cost.
- **Multi-primary / synchronous:** **MySQL Group Replication** (built-in, Paxos-based, single- or multi-primary) vs **MariaDB Galera Cluster** (`wsrep_*`, synchronous certification-based, true multi-master). These are **not interchangeable** — picking one ties you to that engine. Front read/write splitting with **ProxySQL** (query-aware routing) or HAProxy + a VIP.
- Replicas serve read scaling; set `read_only`/`super_read_only` on them.

---

## 11. Security binding

Policy is owned by [`secure-coding.md`](guides://secure-coding.md). MySQL/MariaDB specifics:

- **Parameterized statements only** — the driver's `?`/`%s` placeholders, never string interpolation (MY-SEC-01).
- **Least-privilege grants** scoped to DB + verbs; one role per app. Audit dangerous grants:
```sql
CREATE USER 'app'@'10.%' IDENTIFIED BY :secret;          -- from secrets, per env-config.md
GRANT SELECT, INSERT, UPDATE, DELETE ON app.* TO 'app'@'10.%';   -- no SUPER/FILE/ALL
SELECT user,host FROM mysql.user WHERE Super_priv='Y' OR File_priv='Y';   -- should be admins only
```
- **TLS in transit** (`require_secure_transport=ON`), **TDE at rest** (`innodb` keyring) for sensitive data, and `validate_password` component for password policy.
- MySQL 8 uses `caching_sha2_password` by default (not `mysql_native_password`) — ensure drivers/TLS support it.

---

## 12. Operations: backup, migrations, monitoring

- **Backups (MY-BAK-01):** logical via `mysqldump --single-transaction --routines --triggers` (consistent on InnoDB) or parallel `mydumper`; physical/hot via Percona **XtraBackup** (MySQL) / **mariabackup** (MariaDB/Galera). Keep **binary logs** for point-in-time recovery. A backup is unproven until a restore drill succeeds.
- **Online schema changes (MY-MIG-01):** prefer native `ALGORITHM=INSTANT` (add column, etc. — MySQL 8 / MariaDB 10.3+) or `INPLACE`; for large tables that would `COPY`-lock, use **`pt-online-schema-change`** or `gh-ost`. Every migration is reversible (see `sql.md` migration policy).
- **Observability** (policy: `observability.md`): enable the **slow query log** (`long_query_time`, `log_queries_not_using_indexes`) and digest it with `pt-query-digest`; query **`performance_schema`** (`events_statements_summary_by_digest`) and the `sys` schema for top consumers; track buffer-pool hit ratio, `Threads_running`, replica lag. Percona PMM is a ready dashboard.
- **Local dev:** spin MySQL/MariaDB via `docker-compose.md` with a named volume and `MYSQL_*` env vars; never bake credentials into images.

---

## 13. MySQL vs MariaDB divergence (canonical: `mariadb-divergence`)

Once interchangeable forks, they have **diverged significantly** — do not assume drop-in compatibility. Pick one as the target; if you must support both, fence engine-specific syntax and run the test suite against both (MY-DIV-01).

| Area | MySQL 8.4 LTS / 9.x | MariaDB 11.x |
|---|---|---|
| `JSON` | Native binary type, validated | Alias for `LONGTEXT` + `JSON_VALID` CHECK |
| Synchronous cluster | Group Replication (built-in) | Galera (built-in, `wsrep`) |
| Thread pool | Enterprise only | Built-in (community) |
| Auth default | `caching_sha2_password` | `mysql_native_password` / ed25519 |
| Sequences | No (use `AUTO_INCREMENT`) | `CREATE SEQUENCE` (Oracle-style) |
| System-versioned (temporal) tables | No | `WITH SYSTEM VERSIONING` + `FOR SYSTEM_TIME` |
| Native `UUID` type | No (`BINARY(16)` + `UUID_TO_BIN`) | `UUID` type (byte-ordered) |
| Invisible/INVISIBLE columns | No | Yes |
| Vector search | `VECTOR` type + distance fns (8.4+) | `VECTOR` + `VEC_DISTANCE` (11.7+; different API) |
| JS stored programs | MySQL 9.0 `LANGUAGE JAVASCRIPT` | No |
| CTEs / window functions | 8.0+ | 10.2+ (compatible) |

Migration: MySQL→MariaDB is usually smooth **except JSON** (different storage/perf) and auth plugins. MariaDB→MySQL means dropping sequences, temporal tables, invisible columns, and replacing Galera with Group Replication. Always re-test JSON and replication paths.

---

## 14. Deployment Checklist

Generated from §2 — one box per requirement ID. No new requirements.

- [ ] MY-ENG-01 — all tables `ENGINE=InnoDB`, no MyISAM
- [ ] MY-CHR-01 — `utf8mb4` + explicit collation across DB/table/column
- [ ] MY-PK-01 — every table has a small, monotonic PRIMARY KEY
- [ ] MY-IDX-01 — hot WHERE/JOIN/ORDER-BY/FK columns indexed; `EXPLAIN` shows no `ALL` on hot paths
- [ ] MY-SEC-01 — all SQL parameterized (see `secure-coding.md`)
- [ ] MY-SEC-02 — app user least-privilege, no SUPER/FILE/ALL
- [ ] MY-TXN-01 — transactions short; deadlock/lock-wait retried with backoff (see `error-handling.md`)
- [ ] MY-MIG-01 — schema changes reversible & online (INSTANT/INPLACE or pt-osc)
- [ ] MY-CFG-01 — credentials/DSN from env (see `env-config.md`)
- [ ] MY-BAK-01 — backups restore-tested; binlog on for PITR
- [ ] MY-DIV-01 — engine-specific syntax fenced; suite passes on target engine(s)
- [ ] Agent ran every §3 verification command and documented any fixes

---
**End of MySQL & MariaDB Guidelines**
