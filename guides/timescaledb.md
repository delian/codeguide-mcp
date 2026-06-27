# TimescaleDB Development Guidelines
Mandatory standards for time-series data on PostgreSQL with TimescaleDB: hypertables, chunk sizing, columnar compression (hypercore), continuous aggregates, retention/tiering, and time-series query functions. TimescaleDB 2.x on PostgreSQL 17.

---
name: timescaledb
title: TimescaleDB Development Guidelines
version: 2.0
last_reviewed: 2026-06-05
kind: datastore
tools: [timescaledb@2.x, postgresql@17, timescaledb-toolkit, timescaledb-tune, psql]
requires:
  - sql
  - postgresql
  - secure-coding
recommends:
  - observability
  - performance
  - error-handling
provides:
  - hypertables
  - continuous-aggregates
  - timescale-compression
  - retention-policies
  - time-bucket
---

> 🧭 Authored per [`CONVENTIONS.md`](guides://CONVENTIONS.md): shared concerns are referenced, not restated. TimescaleDB **is** PostgreSQL — this guide covers only what the extension adds. Everything about types, indexing internals, EXPLAIN, RLS, roles, transactions, replication, and `pg_dump` lives in [`postgresql.md`](guides://postgresql.md) and is **not** repeated here.

---

## 0. Prerequisites & References

Fetch and apply these **before** generating TimescaleDB code. Their rules are assumed below.

> 📎 **REQUIRED — fetch & apply first:**
> - [`postgresql.md`](guides://postgresql.md) — the base engine: data types (`TIMESTAMPTZ`, `NUMERIC`, `JSONB`), index types (B-tree/BRIN/GIN), EXPLAIN, RLS, roles, transactions, partitioning, `pg_dump`. *(TimescaleDB binding: hypertables sit on top of all of it.)*
> - [`sql.md`](guides://sql.md) — portable SQL style, set-based thinking, parameterization.
> - [`secure-coding.md`](guides://secure-coding.md) — secrets, supply chain, CVE policy. *(Binding: pin the extension + image version; least-privilege DB roles.)*

> 📎 **RECOMMENDED — fetch when the task touches them:**
> - [`observability.md`](guides://observability.md) — TimescaleDB is most often the **store** for metrics/traces; also monitor the DB itself (`pg_stat_statements`, job stats).
> - [`performance.md`](guides://performance.md) — methodology for tuning chunk size, compression, and memory.
> - [`error-handling.md`](guides://error-handling.md) — handle decompression/late-data/retry errors at the application edge.

> 📎 **SEE ALSO:** [`tdd.md`](guides://tdd.md) *(pgTAP / app-level tests, owner of test-first)* · [`influxdb.md`](guides://influxdb.md) *(dedicated TSDB alternative)* · [`sqlalchemy-alembic.md`](guides://sqlalchemy-alembic.md) *(ORM migrations)*

---

## 1. Core Philosophies: TIMESCALE-FIRST

TimescaleDB-specific principles only. Schema, security, and tuning *policy* come from §0.

- **T**ime is the partition key: every time-series table is a **hypertable** partitioned by a `TIMESTAMPTZ` (or other time) dimension; queries always carry a time predicate.
- **I**ngest hot, read cold cheaply: recent chunks stay in the rowstore for fast writes; aged chunks roll into the **columnstore** (hypercore) for 90%+ compression.
- **M**aterialize, don't recompute: dashboards read **continuous aggregates**, never raw rows.
- **E**xpire by chunk, not by row: drop old data with `drop_chunks`/retention policies (metadata op) — never `DELETE`.
- **L**et the planner skip chunks: chunk exclusion + `time_bucket` + min/max chunk metadata do the work; help them with the right segmentby/orderby.
- **N**ative SQL only: it is still PostgreSQL — joins, `JSONB`, window functions, foreign keys all work; no proprietary query language.

**Verified Code**: Agent-generated schema, policies, and queries MUST pass every gate in §2 before delivery.

---

## 2. Requirements (MANDATORY, auditable)

RFC-2119 keywords. IDs `TSDB-<TOPIC>-<NN>`. Rows binding a shared rule cite its owner.

| ID | Requirement | Verify | Gate |
|----|-------------|--------|------|
| TSDB-STRUCT-01 | Time-series tables MUST be hypertables, not plain tables | `SELECT * FROM timescaledb_information.hypertables` | table present |
| TSDB-STRUCT-02 | Any PK/UNIQUE constraint MUST include the time partitioning column | `\d <table>` / review | time col in key |
| TSDB-PART-01 | `chunk_time_interval` MUST size a chunk (+indexes) at ≈25% of RAM | chunk-size query (§4) | within target band |
| TSDB-COMP-01 | Chunks past the hot window MUST have a columnstore/compression policy | `timescaledb_information.jobs` | policy exists |
| TSDB-COMP-02 | `segmentby`/`orderby` MUST match common filter + sort columns | review against query set | aligned |
| TSDB-RET-01 | Raw hypertables MUST have a retention policy or a documented exception (see `adr.md`) | `timescaledb_information.jobs` | policy or ADR |
| TSDB-CAGG-01 | Dashboard/report queries MUST read continuous aggregates, not raw rows | `EXPLAIN` / review | reads cagg |
| TSDB-QRY-01 | Time-series queries MUST carry a time predicate enabling chunk exclusion | `EXPLAIN (ANALYZE)` | "Chunks excluded" > 0, no full scan |
| TSDB-QRY-02 | Bucketed aggregation MUST use `time_bucket`, not `date_trunc` | grep / review | `time_bucket` only |
| TSDB-SEC-01 | RLS, least-privilege roles, SSL per `postgresql.md` + `secure-coding.md` | postgresql.md §8 checks | 0 gaps |
| TSDB-SEC-02 | Extension + image version pinned, 0 high/critical CVEs (see `secure-coding.md`) | version scan | 0 high/critical |
| TSDB-OPS-01 | Backup/restore MUST be TimescaleDB-aware (`timescaledb_pre/post_restore`, no parallel restore) | review backup scripts | correct procedure |
| TSDB-TST-01 | Every feature MUST be test-first via pgTAP/app tests (see `tdd.md`) | test runner | exit 0, 0 skips |

> **Forbidden**: deleting old data with `DELETE` instead of `drop_chunks`; querying raw hypertables for dashboards; time-series queries with no time predicate; mutating compressed/columnstore chunks without converting back first; `pg_restore -j` against a TimescaleDB dump.

---

## 3. Verification Protocol

Run before presenting schema/queries. Fix → re-run until green.

```sql
-- TSDB-STRUCT-01/02: confirm hypertable + dimensions
SELECT hypertable_name, num_dimensions FROM timescaledb_information.hypertables;
SELECT * FROM timescaledb_information.dimensions WHERE hypertable_name = '<t>';

-- TSDB-PART-01: chunk sizes vs 25% of shared memory
SELECT chunk_name, pg_size_pretty(total_bytes) FROM chunk_compression_stats('<t>');

-- TSDB-COMP-01/RET-01: policies exist
SELECT proc_name, hypertable_name, schedule_interval, config
FROM timescaledb_information.jobs WHERE hypertable_name = '<t>';

-- TSDB-QRY-01: chunk exclusion is happening
EXPLAIN (ANALYZE, BUFFERS) <query>;   -- expect "Chunks excluded during startup: N"
```

The *why* behind security/perf/test gates lives in the §0 owners; do not re-derive it here.

---

## 4. Hypertables & Chunks (the core abstraction)

A **hypertable** is a logical table automatically split into **chunks** — real child PostgreSQL tables, each covering a time range, with their own indexes and independent compression. This is PostgreSQL declarative partitioning (see `postgresql.md`) with automatic chunk creation, chunk-exclusion planning, and time-series policies layered on.

### A. Create (modern dimension API, TimescaleDB 2.x)

```sql
-- Preferred: declare the hypertable + columnstore at CREATE time (2.18+)
CREATE TABLE conditions (
    time        TIMESTAMPTZ NOT NULL,
    device_id   INTEGER     NOT NULL,
    temperature DOUBLE PRECISION,
    humidity    DOUBLE PRECISION
) WITH (
    tsdb.hypertable,
    tsdb.partition_column = 'time',
    tsdb.segmentby = 'device_id',
    tsdb.orderby   = 'time DESC'
);

-- Or convert an existing table with the dimension builders:
SELECT create_hypertable('conditions', by_range('time', INTERVAL '1 day'));
SELECT add_dimension('conditions', by_hash('device_id', 4));   -- optional space partition

-- Legacy positional form still works but prefer by_range/by_hash:
-- SELECT create_hypertable('conditions', 'time', chunk_time_interval => INTERVAL '1 day');
```

- **PK/UNIQUE must include the time column** (TSDB-STRUCT-02): `PRIMARY KEY (device_id, time)`. A unique key that excludes the partitioning column is rejected.
- Create indexes **after** the hypertable; they propagate to every chunk. `create_hypertable` adds a `time DESC` index automatically.
- `migrate_data => TRUE` rechunks existing rows (slow; lock-heavy — do it in a maintenance window).

### B. Chunk sizing — the 25% rule (TSDB-PART-01)

Size `chunk_time_interval` so that **one chunk plus its indexes ≈ 25% of RAM**, keeping the active write-set in `shared_buffers`.

```
target_chunk_bytes ≈ 0.25 * RAM
chunk_interval     ≈ target_chunk_bytes / ingest_bytes_per_unit_time
```

| Ingest rate | Typical interval |
|---|---|
| High (≥1M rows/s) | 1–12 h |
| Standard (10k–100k rows/s) | 1–7 days |
| Low (<10k rows/s) | 7–30 days |

Default is 7 days. Adjust with `SELECT set_chunk_time_interval('conditions', INTERVAL '12 hours');` (applies to *new* chunks only). Inspect with `timescaledb_information.chunks` (`range_start/end`, `total_bytes`).

### C. Space partitioning

A `by_hash` dimension distributes a chunk's rows across N partitions. Use it for parallelism on multi-disk single nodes or to balance load — **not** to speed up `WHERE device_id = X` (hash partitioning gives no range pruning). Most single-node deployments need only the time dimension.

### D. Manual chunk ops

```sql
SELECT show_chunks('conditions', older_than => INTERVAL '90 days');
SELECT drop_chunks('conditions', older_than => INTERVAL '90 days');   -- metadata op, instant
```

---

## 5. Compression — Hypercore Columnstore (the big win)

Aged chunks are converted from row storage to a **columnar** format ("hypercore" / columnstore), typically **90–95% smaller**, with analytical scans often *faster* because fewer pages are read. Reads are transparent — no query change needed. Per-column algorithms (delta, delta-of-delta, Gorilla XOR for floats, dictionary, run-length, LZ) are chosen automatically.

### A. Enable + policy (modern API)

```sql
-- Configure columnstore (if not set at CREATE time):
ALTER TABLE conditions SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'device_id',     -- group rows queried/grouped together
    timescaledb.compress_orderby   = 'time DESC'       -- order within a compressed batch
);

-- Auto-convert chunks older than the hot window (TSDB-COMP-01):
CALL add_columnstore_policy('conditions', after => INTERVAL '7 days');
-- (equivalent classic form: SELECT add_compression_policy('conditions', INTERVAL '7 days');)
```

### B. Choosing segmentby / orderby (TSDB-COMP-02)

- **segmentby** → the low/medium-cardinality columns you filter or `GROUP BY` (e.g. `device_id`, `symbol`). Segmented values are stored once per batch and enable predicate push-down. Never segment by a high-cardinality column (UUID, raw timestamp) — it defeats batching.
- **orderby** → almost always `time DESC`; add secondary sort columns you frequently order by.
- Bloom + min/max sparse indexes are built on new columnstore chunks automatically and power chunk skipping.

### C. Manual conversion + mutating compressed data

```sql
SELECT convert_to_columnstore(c)  FROM show_chunks('conditions', older_than => INTERVAL '7 days') c;
SELECT convert_to_rowstore('_timescaledb_internal._hyper_1_42_chunk');  -- before bulk UPDATE/DELETE
```

Modern TimescaleDB allows `INSERT ... ON CONFLICT`, `UPDATE`, and `DELETE` directly on columnstore chunks (auto round-trips internally), but **large** backfills are far cheaper if you `convert_to_rowstore` first, modify, then re-convert. Treat compressed chunks as effectively append-mostly.

---

## 6. Continuous Aggregates (incremental materialized views)

A continuous aggregate is a materialized view over a hypertable that **refreshes only the changed time ranges** instead of recomputing — orders of magnitude faster than querying raw data, and the canonical way to serve dashboards (TSDB-CAGG-01).

### A. Define + refresh policy

```sql
CREATE MATERIALIZED VIEW conditions_hourly
WITH (timescaledb.continuous) AS
SELECT time_bucket('1 hour', time) AS bucket,
       device_id,
       avg(temperature) AS avg_temp,
       max(temperature) AS max_temp,
       count(*)         AS n
FROM conditions
GROUP BY bucket, device_id
WITH NO DATA;                       -- backfill via the policy, not at DDL time

SELECT add_continuous_aggregate_policy('conditions_hourly',
    start_offset      => INTERVAL '3 hours',    -- how far back each run refreshes
    end_offset        => INTERVAL '1 hour',     -- leave recent, still-filling buckets alone
    schedule_interval => INTERVAL '30 minutes');
```

`end_offset` exists to avoid materializing incomplete buckets and to wait for late-arriving data.

### B. Real-time aggregation

By default a cagg is `materialized_only=false`: a query transparently **unions** the materialized buckets with a live aggregate over the not-yet-materialized tail, so reads are always current. Set `materialized_only=true` for strictly precomputed (faster, possibly stale) reads.

### C. Hierarchical caggs & compression

Build a cagg **on top of another cagg** for cheap multi-resolution rollups (1 min → 1 h → 1 day), each with its own retention. Caggs are hypertables too, so compress and retain them:

```sql
CALL add_columnstore_policy('conditions_hourly', after => INTERVAL '30 days');
SELECT add_retention_policy('conditions_hourly', INTERVAL '2 years');
CALL refresh_continuous_aggregate('conditions_hourly', '2026-01-01', '2026-02-01');  -- manual range
```

Caggs may use `FILTER`, `first()`/`last()`, and toolkit aggregates (§8). Window functions are not allowed directly in the cagg definition — wrap the bucketed aggregate in an outer view.

---

## 7. Retention & Tiering (lifecycle by chunk)

Expire raw data by **dropping whole chunks** (TSDB-RET-01) — instant metadata operation, no vacuum churn:

```sql
SELECT add_retention_policy('conditions', INTERVAL '90 days');   -- runs daily by default
```

Typical tiered lifecycle — keep raw briefly, downsample into ever-coarser caggs kept ever-longer:

| Tier | Age | Storage |
|---|---|---|
| Hot | 0–7 d | rowstore hypertable, uncompressed |
| Warm | 7–90 d | columnstore (compressed) |
| Cool | 90 d–2 y | continuous aggregates only; raw chunks dropped |
| Cold/archive | 2 y+ | exported to object storage (or Timescale tiered storage) before drop |

Set the **retention window past the cagg `start_offset`** so a refresh never needs rows already dropped. Tune any job's cadence with `alter_job(<job_id>, schedule_interval => INTERVAL '6 hours')`.

---

## 8. Time-Series Query Functions & Hyperfunctions

The reason to use TimescaleDB over hand-rolled Postgres partitioning. `time_bucket`, gap-filling, and `first/last` are in core; the rest ship in the **`timescaledb_toolkit`** extension (`CREATE EXTENSION timescaledb_toolkit;`).

### A. Bucketing, gap-filling, LOCF, interpolation (core)

```sql
-- time_bucket: chunk-aware downsampling (always prefer over date_trunc — TSDB-QRY-02)
SELECT time_bucket('5 minutes', time) AS bucket, device_id, avg(temperature)
FROM conditions
WHERE time > now() - INTERVAL '24 hours'
GROUP BY bucket, device_id;

-- Gap-filling produces a row for every bucket even when no data exists;
-- locf() carries the last value forward, interpolate() draws a straight line.
SELECT time_bucket_gapfill('1 hour', time) AS bucket,
       device_id,
       locf(avg(temperature))        AS temp_locf,
       interpolate(avg(temperature)) AS temp_interp
FROM conditions
WHERE time > now() - INTERVAL '7 days' AND time < now()
GROUP BY bucket, device_id
ORDER BY bucket;

-- first()/last(): value of one column at the min/max of another (e.g. latest reading per device)
SELECT device_id, last(temperature, time) AS latest_temp
FROM conditions WHERE time > now() - INTERVAL '1 hour'
GROUP BY device_id;
```

`time_bucket_gapfill` requires bounded `time` predicates so it knows the range to fill.

### B. Toolkit hyperfunctions (aggregate → accessor pattern)

Two-step aggregates: an aggregate builds a partial state (storable in a cagg), accessors read summaries from it cheaply.

```sql
-- Percentiles without storing raw data (uddsketch/t-digest)
SELECT approx_percentile(0.95, percentile_agg(response_ms)) AS p95 FROM requests;

-- Counters that reset (network/cpu): rate, delta, irate over monotonic series
SELECT device_id, rate(counter_agg(time, bytes_total)) FROM net GROUP BY device_id;

-- Statistical aggregate: one pass, many accessors
SELECT average(s), stddev(s) FROM (SELECT stats_agg(temperature) s FROM conditions) q;

-- OHLC candlesticks, time-weighted averages, hyperloglog distinct counts, state_agg, etc.
SELECT candlestick_agg(time, price, volume) FROM ticks GROUP BY time_bucket('1 day', time);
```

Store the `*_agg` partial in a continuous aggregate, then apply accessors at query time — this is the idiomatic, cheap way to serve percentile/rate/OHLC dashboards.

---

## 9. Indexing & Query Optimization

Index *mechanics* (B-tree/BRIN/GIN, partial, covering, `EXPLAIN` reading) are owned by [`postgresql.md`](guides://postgresql.md). TimescaleDB-specific bindings:

- **Default**: `(<segment cols>, time DESC)` covers the dominant "latest rows for an entity" pattern. The auto-created `(time DESC)` index covers time-range scans.
- **BRIN** on `time` is tiny and effective because chunks are physically time-ordered — good for huge append-only hypertables.
- **Chunk exclusion (TSDB-QRY-01)** is the headline optimization: a `WHERE` predicate on the time (or space) dimension lets the planner skip whole chunks at plan/startup time. Verify with `EXPLAIN (ANALYZE)` → `Chunks excluded during startup: N`. A query with no time predicate scans every chunk — forbidden for production paths.
- **Columnstore chunk skipping**: min/max + bloom sparse indexes on columnstore chunks skip batches by `segmentby`/`orderby` value; enable extra ranges with `enable_chunk_skipping`.
- Indexes are per-chunk: a `REINDEX`/new index only touches the chunks you target; build concurrently on large hypertables.

---

## 10. Writing Data

```sql
-- Batch inserts (multi-row VALUES or COPY); 1k–10k rows/statement is the sweet spot.
INSERT INTO conditions (time, device_id, temperature) VALUES
  ('2026-06-27T10:00Z', 1, 21.5), ('2026-06-27T10:00Z', 2, 19.8) /* ... */;
```

- Bulk-load with `COPY` or the parallel `timescaledb-parallel-copy` tool — far faster than row-by-row.
- Out-of-order / late inserts go to the correct chunk automatically (chunks are created on demand); extreme back-dating across many old chunks is slow — batch by time range.
- Upserts use `INSERT ... ON CONFLICT (time, device_id) DO UPDATE`; the conflict target must match a unique index that includes the time column.
- Application-side write errors, retries, and idempotency: see [`error-handling.md`](guides://error-handling.md).

---

## 11. When to Use TimescaleDB

| Choose… | When |
|---|---|
| **Plain PostgreSQL** | Low-volume time data (< ~10M rows), no compression/retention needs — a `created_at` B-tree index is enough. |
| **Native PG declarative partitioning** | You need partitioning but not time-series policies/hyperfunctions, and you already manage partitions (or `pg_partman`). |
| **TimescaleDB** | High-ingest time-series + you want compression, automatic chunking, continuous aggregates, retention policies, and hyperfunctions — **while keeping full SQL, joins, and the Postgres ecosystem**. |
| **Dedicated TSDB** (InfluxDB, etc. — see [`influxdb.md`](guides://influxdb.md)) | Extreme cardinality / metrics-only workloads where you do not need relational joins, transactions, or SQL, and a purpose-built push/query stack fits better. |

TimescaleDB's edge: it is *just Postgres*, so your ORM, migrations ([`sqlalchemy-alembic.md`](guides://sqlalchemy-alembic.md)), RLS, and dimension-table joins all keep working.

---

## 12. Operations Bindings (defer to owners)

These are mostly standard PostgreSQL ops — follow [`postgresql.md`](guides://postgresql.md) and [`secure-coding.md`](guides://secure-coding.md). TimescaleDB-specific deltas only:

- **Backup/restore (TSDB-OPS-01)**: `pg_dump` captures hypertables, chunks, caggs, and policies. On restore wrap with `SELECT timescaledb_pre_restore();` … restore … `SELECT timescaledb_post_restore();` and **do not use parallel restore** (`-j`) — it corrupts the catalog. Physical/base backups and PITR work normally.
- **Tuning**: run `timescaledb-tune` to set `shared_buffers`, `work_mem`, `effective_cache_size`, and `timescaledb.max_background_workers`. Methodology → [`performance.md`](guides://performance.md).
- **Monitoring**: `pg_stat_statements` plus `timescaledb_information.jobs`/`job_stats` (policy success/failure). When TimescaleDB is the metrics store *and* the thing being monitored, wire both into [`observability.md`](guides://observability.md).
- **Version upgrades**: `ALTER EXTENSION timescaledb UPDATE;` then restart; pin the extension and the `timescale/timescaledb:*-pg17` image (TSDB-SEC-02).
- **HA / replication / multi-node**: streaming replication and Patroni work as in `postgresql.md`. Note: built-in multi-node distributed hypertables are **deprecated/removed** in current TimescaleDB — scale up + columnstore, or use Timescale Cloud, rather than self-managed multi-node.

---

## 13. Deployment Checklist

Generated from §2 — one box per requirement ID.

- [ ] TSDB-STRUCT-01 — time-series tables are hypertables
- [ ] TSDB-STRUCT-02 — PK/UNIQUE includes the time column
- [ ] TSDB-PART-01 — chunk interval sized to ≈25% RAM
- [ ] TSDB-COMP-01/02 — columnstore policy set; segmentby/orderby match queries
- [ ] TSDB-RET-01 — retention policy (or documented ADR exception)
- [ ] TSDB-CAGG-01 — dashboards read continuous aggregates, not raw
- [ ] TSDB-QRY-01 — time predicate present; chunk exclusion verified in EXPLAIN
- [ ] TSDB-QRY-02 — bucketing uses `time_bucket`
- [ ] TSDB-SEC-01/02 — RLS/roles/SSL per postgresql.md; version pinned, 0 high/critical CVEs
- [ ] TSDB-OPS-01 — TimescaleDB-aware backup/restore (pre/post_restore, no `-j`)
- [ ] TSDB-TST-01 — features test-first (pgTAP/app), tests green
- [ ] Agent ran every §3 query and documented any fixes

---
**End of TimescaleDB Guidelines**
