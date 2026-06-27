# DuckDB Development Guidelines
Mandatory standards for DuckDB: in-process OLAP, columnar/vectorized, direct file querying, friendly SQL, Arrow zero-copy. DuckDB 1.x, Python/CLI, httpfs.

---
name: duckdb
title: DuckDB Development Guidelines
version: 2.0
last_reviewed: 2026-06-05
kind: datastore
tools: [duckdb@1.1, python@3.13, httpfs, pip-audit]
requires:
  - sql
  - secure-coding
recommends:
  - performance
  - python
  - error-handling
provides:
  - duckdb-olap
  - columnar-vectorized
  - direct-file-query
  - duckdb-dialect
  - arrow-integration
---

> 🧭 Authored per [`CONVENTIONS.md`](guides://CONVENTIONS.md): shared concerns are referenced, not restated. This guide covers only what is unique to DuckDB.

---

## 0. Prerequisites & References

Fetch and apply these **before** generating DuckDB code. Their rules are assumed below and not repeated.

> 📎 **REQUIRED — fetch & apply first:**
> - [`sql.md`](guides://sql.md) — general SQL: query design, joins, window-function and CTE semantics, indexing/normalization theory. *(DuckDB binding: a PostgreSQL-like dialect with analytical extensions — only the DuckDB-specific syntax lives here.)*
> - [`secure-coding.md`](guides://secure-coding.md) — supply chain, secrets, untrusted input. *(DuckDB binding: parameterized SQL, sandboxing untrusted file/extension/path access, `CREATE SECRET`, `pip-audit`.)*

> 📎 **RECOMMENDED — fetch when the task touches them:**
> - [`performance.md`](guides://performance.md) — perf method; DuckDB binding is columnar/vectorized execution and pushdown (§7).
> - [`python.md`](guides://python.md) — the dominant integration: pandas/Polars/Arrow, `uv`, `pip-audit` (§6).
> - [`error-handling.md`](guides://error-handling.md) — error strategy at the DB boundary.

> 📎 **SEE ALSO:** [`tdd.md`](guides://tdd.md) · [`sqlite.md`](guides://sqlite.md) · [`postgresql.md`](guides://postgresql.md) *(scanner/migration source)* · [`parallelism.md`](guides://parallelism.md)

---

## 1. Core Philosophies: OLAP-FIRST

DuckDB-specific principles only. SQL design, security, and error handling come from §0.

- **O**LAP, not OLTP: DuckDB is an *embedded analytical* engine — "SQLite for analytics". Reach for it for scans/aggregations/joins over columnar data; **not** for high-concurrency row-level writes (§3).
- **L**et data stay where it is: query Parquet/CSV/JSON/Arrow **in place** with pushdown; import into a `.duckdb` file only when repeated queries justify it (§4, §8).
- **A**rrow/columnar end to end: move data zero-copy via Arrow; prefer Parquet over CSV; lean on vectorized execution and automatic compression rather than manual tuning (§6, §7).
- **P**arameterize everything: never interpolate user input into SQL or file paths — use `?`/`$name` placeholders (§2, `secure-coding.md`).
- **F**riendly SQL: use DuckDB's analytical dialect (`QUALIFY`, `PIVOT`, `ASOF JOIN`, `SAMPLE`, `SELECT * EXCLUDE/REPLACE`, list/struct types) instead of verbose ANSI workarounds (§5).

**Verified Code**: Agent-generated DuckDB code MUST use parameterized SQL and pass every gate in §2 before delivery.

---

## 2. Requirements (MANDATORY, auditable)

RFC-2119 keywords. IDs `DUCK-<TOPIC>-<NN>`. Each row has a binary gate; rows binding a shared rule cite its owner.

| ID | Requirement | Verify | Gate |
|----|-------------|--------|------|
| DUCK-TST-01 | Every query/transform MUST be test-first against an in-memory DB (see `tdd.md`) | `pytest` (fixture `duckdb.connect(':memory:')`) | exit 0, 0 skips |
| DUCK-TST-02 | Each bug MUST get a regression test before the fix (see `tdd.md`) | `pytest` | failing→passing |
| DUCK-SEC-01 | All dynamic values MUST use `?`/`$name` placeholders — never f-string/concat SQL (see `secure-coding.md`, `sql.md`) | grep for f-string SQL / review | 0 interpolated SQL |
| DUCK-SEC-02 | Untrusted contexts MUST disable filesystem/network escape | `SELECT current_setting('enable_external_access')` | `false` when untrusted |
| DUCK-SEC-03 | Unsigned extensions MUST be disallowed in production | `SELECT current_setting('allow_unsigned_extensions')` | `false` |
| DUCK-SEC-04 | Cloud credentials MUST come from `CREATE SECRET`/env/IAM, never literals (see `secure-coding.md`) | grep for keys / review | 0 hardcoded secrets |
| DUCK-DEP-01 | 0 known CVEs in `duckdb` + host deps (see `secure-coding.md`) | `pip-audit` | 0 vulnerabilities |
| DUCK-VER-01 | DuckDB MUST be pinned to a 1.x release; storage format compatible | `duckdb --version` / lockfile | pinned ≥1.0 |
| DUCK-PERF-01 | Repeated analytics MUST use Parquet (not CSV) and column pruning (see `performance.md`) | review / `EXPLAIN` | no `SELECT *` over CSV in hot path |
| DUCK-CONC-01 | Multi-process access MUST open extra connections `read_only=True` (single writer) | review | 1 writer, N readers |

> **Forbidden**: interpolating user input into SQL or file paths; shipping a transform before its test (violates `tdd.md`); loading unsigned/unofficial extensions in production; using DuckDB as an OLTP store for concurrent writers; hardcoding S3/GCS/Azure credentials.

---

## 3. Verification Protocol

Run before presenting code. Fix → re-run until every gate is green.

```bash
pytest                                   # DUCK-TST-01/02 (in-memory fixtures)
grep -rnE 'execute\(f"|execute\(.*\+|\.format\(' src/   # DUCK-SEC-01: no string-built SQL
pip-audit                                # DUCK-DEP-01
duckdb --version                         # DUCK-VER-01: pinned 1.x
```
```sql
SELECT current_setting('enable_external_access');     -- DUCK-SEC-02 (false if untrusted)
SELECT current_setting('allow_unsigned_extensions');  -- DUCK-SEC-03 (false in prod)
```
The *why* behind each gate lives in its §0 owner; do not re-derive it here.

---

## 4. The OLAP Model — When DuckDB Fits

DuckDB is an **in-process, columnar, vectorized** SQL engine: no server, single library/binary, full ACID. It stores columns contiguously (great compression + read only the columns you touch) and processes them in ~2048-row vectors with SIMD across all cores.

**Use DuckDB for:** interactive analysis (notebooks, ad-hoc SQL over CSV/Parquet), ETL/format conversion, BI/dashboards, data-lake/lakehouse queries over object storage, embedded analytics in desktop/CLI apps, and fast analytical test fixtures.

**Do NOT use DuckDB for** (reach for the noted alternative):

| Not a fit | Why | Use instead |
|---|---|---|
| High-concurrency OLTP / many small writes | Single-writer, MVCC tuned for bulk | `postgresql.md`, `mysql-mariadb` |
| Distributed / multi-node scale-out | Single-node only, no clustering | ClickHouse, BigQuery, Snowflake |
| Real-time streaming | Batch-oriented | Kafka + Flink |
| Multi-process concurrent writers | One writer at a time | a client-server RDBMS |

Readers never block the single writer (WAL); concurrency within one process scales across threads.

---

## 5. Direct File Querying (the killer feature)

Query files **in place** — the path *is* the table. No `CREATE TABLE`/load step needed.

```sql
SELECT * FROM 'data.parquet';                 -- single file
SELECT * FROM 'data/**/*.parquet';            -- recursive glob, read in parallel
SELECT category, SUM(amount) FROM 's3://bkt/sales/*.parquet' GROUP BY category;
SELECT * FROM read_csv('d.csv', header=true, sample_size=200000, ignore_errors=true);
SELECT * FROM read_json('events.ndjson', format='newline_delimited');
SELECT * FROM read_parquet('y=*/m=*/f.parquet', hive_partitioning=true);  -- partition cols from path
SELECT * FROM read_csv('logs/*.csv', union_by_name=true);  -- align differing schemas
```

- **Pushdown is automatic:** projection (read only referenced columns) and predicate pushdown (skip Parquet row groups via min/max stats). Filter early; select only needed columns. Inspect with `EXPLAIN ANALYZE`.
- **Hive partitioning:** layout as `year=2024/month=01/…`; a `WHERE year=2024` prunes whole directories.
- **Write/convert** with `COPY`:
  ```sql
  COPY (SELECT * FROM 'in.csv') TO 'out.parquet' (FORMAT PARQUET, COMPRESSION zstd);
  COPY tbl TO 'out' (FORMAT PARQUET, PARTITION_BY (year, month));  -- partitioned dataset
  ```
- **Attach external databases** (zero ETL): `ATTACH 'pg.conn' AS pg (TYPE POSTGRES);` then join `pg.users` to a local Parquet file in one query. Same for `TYPE SQLITE`, `TYPE MYSQL`.
- **Persist vs in-memory:** `duckdb.connect(':memory:')` is ephemeral; `connect('analytics.duckdb')` persists and lets you `CREATE TABLE … AS SELECT …` once and query many times — far faster than re-scanning files per query.
- **Larger-than-memory:** DuckDB spills to disk automatically. Set `SET memory_limit='16GB'; SET temp_directory='/nvme/tmp';` so out-of-core joins/sorts/aggregations land on fast storage.

---

## 6. The DuckDB SQL Dialect

A PostgreSQL-like dialect plus analytical "friendly SQL". General join/CTE/window *semantics* are owned by [`sql.md`](guides://sql.md); below is only what DuckDB adds.

### A. Friendly SQL ergonomics
```sql
SELECT * EXCLUDE (ssn, password) FROM users;          -- drop columns from *
SELECT * REPLACE (UPPER(name) AS name) FROM users;    -- transform in place
SELECT COLUMNS('sales_.*') FROM t;                    -- regex column selection
SELECT region, SUM(sales) FROM t GROUP BY ALL;        -- group by all non-aggregates
SELECT region, product, SUM(sales) FROM t
QUALIFY ROW_NUMBER() OVER (PARTITION BY region ORDER BY SUM(sales) DESC) <= 5;  -- filter window results, no subquery
FROM t SELECT count(*);                               -- FROM-first is valid
```

### B. Analytical operators
```sql
PIVOT sales ON year USING SUM(amount);                -- long→wide, no manual CASE
UNPIVOT wide ON q1, q2, q3, q4 INTO NAME quarter VALUE amount;

-- ASOF JOIN: match each row to the most recent row in another table (time-series alignment)
SELECT t.*, p.price
FROM trades t ASOF JOIN prices p
  ON t.symbol = p.symbol AND t.ts >= p.ts;

SELECT * FROM big USING SAMPLE 5% (BERNOULLI);        -- fast sampling (also SYSTEM, N ROWS)
SELECT APPROX_COUNT_DISTINCT(user_id), APPROX_QUANTILE(amount, 0.5) FROM events;  -- HLL/t-digest
GROUP BY CUBE (a, b) / ROLLUP (a, b) / GROUPING SETS ((a),(b),());
```

### C. Nested types (list / struct / map / union)
```sql
SELECT tags[1], len(tags), list_contains(tags, 'x'),  -- LIST (1-indexed)
       [x*2 FOR x IN tags IF x > 0] AS doubled         -- list comprehension
FROM t;
SELECT person.address.city FROM t;                     -- STRUCT dot access
SELECT attrs['height'] FROM t;                         -- MAP lookup
SELECT data->>'$.user.name', json_extract(data, '$.tags') FROM events;  -- JSON path
```
Use the smallest exact integer type (`UTINYINT`…`UHUGEINT`), `DECIMAL(p,s)` for money, and rich temporal types (`TIMESTAMP`, `TIMESTAMPTZ`, `INTERVAL`). Compression (dictionary/RLE/bit-packing/FSST) is automatic — do not hand-tune it.

> Generic analytical *patterns* (cohort, funnel, retention, moving averages) are just CTEs + window functions — build them per [`sql.md`](guides://sql.md). DuckDB's only specialization is the friendlier syntax above.

---

## 7. Python / Arrow / Polars / pandas Integration

Python is the dominant integration; toolchain policy (`uv`, `pip-audit`) is owned by [`python.md`](guides://python.md).

```python
import duckdb
con = duckdb.connect("analytics.duckdb")          # or ":memory:"; use as a context manager

# Replacement scans: a DataFrame in scope IS a table — no registration
import pandas as pd, polars as pl
res = duckdb.sql("SELECT * FROM my_df WHERE age > 25").df()   # pandas in, pandas out
res = duckdb.sql("SELECT * FROM pl_df WHERE v > 15").pl()     # Polars zero-copy
arrow_tbl = con.execute("SELECT * FROM 'big.parquet'").arrow()  # Arrow, zero-copy

# Parameterized — the ONLY safe way to pass values (DUCK-SEC-01)
con.execute("SELECT * FROM users WHERE name = ? AND age > ?", ["Alice", 25])

# Bulk load via Arrow/DataFrame beats row-by-row executemany
con.execute("INSERT INTO t SELECT * FROM my_df")
con.register("v", my_df); con.unregister("v")     # explicit registration when names collide
con.create_function("tax", lambda x: x * 1.2, return_type="DOUBLE")  # Python UDF
```

- **Zero-copy** Arrow/Polars bridges make DuckDB the fastest path from raw files to a DataFrame; prefer `.arrow()`/`.pl()` over `.fetchall()` for large results.
- **Concurrency:** one writer; give each reader thread its own `connect(path, read_only=True)` connection (DUCK-CONC-01). Do not share a connection across threads. See [`parallelism.md`](guides://parallelism.md).
- **CLI:** `duckdb mydb.duckdb`, or one-shot `duckdb -c "COPY (FROM 'in.csv') TO 'out.parquet'"`.

---

## 8. Extensions

Extensions add formats and connectors; load only what you need, from the **official** repo, with unsigned loading disabled in prod (DUCK-SEC-03).

```sql
INSTALL httpfs; LOAD httpfs;        -- S3/GCS/Azure/HTTP(S) object storage
INSTALL iceberg; INSTALL delta;     -- open table formats (lakehouse)
INSTALL postgres; INSTALL sqlite; INSTALL mysql;   -- live scanners / ATTACH
INSTALL fts; INSTALL spatial; INSTALL icu;         -- search / geo / i18n + collation
SELECT extension_name, installed, loaded FROM duckdb_extensions();
```

**Cloud credentials — use the secret manager, not `SET s3_*` literals** (DUCK-SEC-04):
```sql
CREATE SECRET s3 (TYPE S3, PROVIDER credential_chain, REGION 'us-east-1');  -- IAM/instance role
-- or PROVIDER config with KEY_ID/SECRET sourced from env, never committed
SELECT * FROM 's3://bucket/data/**/*.parquet';
```
Prefer IAM roles / `credential_chain` over static keys. Pin extension versions in production.

---

## 9. Performance Bindings

Method is owned by [`performance.md`](guides://performance.md); measure with `EXPLAIN ANALYZE` and `PRAGMA enable_profiling`. DuckDB-specific levers:

- **Format:** Parquet (compressed, stats, column pruning) over CSV — often 10–100× on scans (DUCK-PERF-01).
- **Read less:** select needed columns only; filter to enable predicate/partition pushdown.
- **Persist hot data:** `CREATE TABLE AS SELECT` once instead of re-scanning files each query.
- **Resources:** `SET threads`, `SET memory_limit`, `SET temp_directory` (fast SSD/NVMe for spills); `SET preserve_insertion_order=false` for large unordered loads.
- **Approximate** (`APPROX_COUNT_DISTINCT`, `APPROX_QUANTILE`) and `USING SAMPLE` when exactness isn't required.

---

## 10. Quick Reference

```bash
duckdb mydb.duckdb                                   # CLI, persistent
duckdb -c "SELECT * FROM 'data.parquet' LIMIT 10"    # one-shot file query
duckdb -c "COPY (FROM 'in.csv') TO 'out.parquet' (FORMAT PARQUET)"   # convert
pytest                                               # tests (in-memory fixtures)
pip-audit                                            # CVE scan
```
```python
import duckdb
con = duckdb.connect(":memory:")
con.execute("SELECT * FROM 'data/**/*.parquet' WHERE year = ?", [2024]).pl()
```

---

## 11. Deployment Checklist

Generated from §2 — one box per requirement ID.

- [ ] DUCK-TST-01/02 — transforms tested first (in-memory), bugs have regression tests
- [ ] DUCK-SEC-01 — all dynamic values parameterized; no f-string/concat SQL
- [ ] DUCK-SEC-02 — `enable_external_access=false` in untrusted contexts
- [ ] DUCK-SEC-03 — `allow_unsigned_extensions=false` in production; extensions from official repo
- [ ] DUCK-SEC-04 — credentials via `CREATE SECRET`/env/IAM, none hardcoded
- [ ] DUCK-DEP-01 — `pip-audit` clean (0 CVEs)
- [ ] DUCK-VER-01 — DuckDB pinned to a 1.x release
- [ ] DUCK-PERF-01 — Parquet + column pruning in hot paths
- [ ] DUCK-CONC-01 — single writer; reader connections `read_only=True`
- [ ] Agent ran every §3 command and documented any fixes

---
**End of DuckDB Guidelines**
