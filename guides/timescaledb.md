# TimescaleDB Development Guidelines
Mandatory coding standards and development practices for TimescaleDB development. Comprehensive best practices for TimescaleDB 2.x covering architecture, data modeling, performance optimization, compression, and production deployment (Updated 2026).

---

**Agent Profile**: The TimescaleDB Expert
**Role**: Senior Time-Series Database Engineer & PostgreSQL Specialist
**Objective**: Generate production-ready, performant, and maintainable time-series solutions using TimescaleDB.
**Tools**: TimescaleDB 2.x, PostgreSQL 12–17, hypertables, continuous aggregates, compression, retention policies

---

## 1. Core Philosophies: TIMESCALE-FIRST

The agent must adhere to the **TIMESCALE-FIRST** principles for every TimescaleDB implementation:

**Test-Driven Development (TDD)**: ALWAYS write tests BEFORE implementation (Red-Green-Refactor cycle mandatory).
**Regression Shield**: EVERY bug discovered MUST receive a test BEFORE fixing to prevent regression.

- **T**ime-series first: Design schemas with a time dimension; use hypertables for all time-ordered data.
- **I**ndexes and chunks: Use chunk-aligned indexes; size chunks for your query and retention patterns.
- **M**aterialize wisely: Use continuous aggregates for heavy rollups; refresh policies to balance freshness and cost.
- **E**nable compression: Compress older chunks; tune segment-by and order-by for query and compression ratio.
- **S**ecurity: Least privilege; secure connections; no secrets in SQL; follow PostgreSQL and TimescaleDB security docs.
- **C**onsistency: Use transactions; understand time bucketing and time zones; test retention and compression.
- **A**vailability: Plan for HA (replication, backups); test restore and failover.
- **L**oad and query: Optimize writes (batch, unlogged when safe); optimize reads (chunk exclusion, indexes).
- **E**xplain and tune: Use EXPLAIN (ANALYZE, BUFFERS); tune work_mem, shared_buffers, and TimescaleDB params.
**Verified Code**: Agent-generated code MUST use parameterized SQL, handle errors, and pass tests before delivery.

---

## Table of Contents

1. [Core Philosophies: TIMESCALE-FIRST](#1-core-philosophies-timescale-first)
2. [TimescaleDB Architecture](#2-timescaledb-architecture)
3. [Data Modeling for Time-Series](#3-data-modeling-for-time-series)
4. [Hypertable Creation and Configuration](#4-hypertable-creation-and-configuration)
5. [Chunk Sizing and Time Partitioning](#5-chunk-sizing-and-time-partitioning)
6. [Compression Strategies](#6-compression-strategies)
7. [Continuous Aggregates](#7-continuous-aggregates)
8. [Retention Policies and Data Lifecycle](#8-retention-policies-and-data-lifecycle)
9. [Write Optimization](#9-write-optimization)
10. [Query Optimization](#10-query-optimization)
11. [Indexing Best Practices](#11-indexing-best-practices)
12. [Memory Optimization and Cache Tuning](#12-memory-optimization-and-cache-tuning)
13. [Performance Tuning](#13-performance-tuning)
14. [Security](#14-security)
15. [High Availability](#15-high-availability)
16. [Backup and Restore](#16-backup-and-restore)
17. [Monitoring and Operations](#17-monitoring-and-operations)
18. [Docker Deployment](#18-docker-deployment)
19. [Kubernetes Deployment](#19-kubernetes-deployment)
20. [Migration Strategies](#20-migration-strategies)
21. [Multi-Node and Distributed Hypertables](#21-multi-node-and-distributed-hypertables)

## 2. TimescaleDB Architecture

### Core Concepts

TimescaleDB is a PostgreSQL extension that transforms PostgreSQL into a time-series database through hypertables - an abstraction over standard PostgreSQL tables that enables automatic partitioning.

```yaml
# TimescaleDB Architecture Components
Extension: PostgreSQL extension (installed per database)
Storage: Chunks (physical PostgreSQL tables)
Partitioning: Automatic time-based partitioning
Compatibility: Full PostgreSQL SQL support
Version: TimescaleDB 2.x (latest as of 2026)
PostgreSQL: Compatible with PostgreSQL 12-17 (17.1 not recommended)
```

### Hypertables and Chunks

```text
Hypertable (Logical Table)
├── Chunk 1: 2024-01-01 to 2024-01-08 (7 days)
├── Chunk 2: 2024-01-08 to 2024-01-15 (7 days)
├── Chunk 3: 2024-01-15 to 2024-01-22 (7 days)
└── Chunk N: 2024-MM-DD to 2024-MM-DD (7 days)

Each chunk:
- Physical PostgreSQL table
- Separate indexes (built per chunk)
- Independent compression
- Automatic maintenance
```

**Key Benefits:**
```yaml
Automatic_Partitioning:
  - No manual partition management
  - Time-based by default
  - Optional space partitioning

Query_Performance:
  - Chunk exclusion (skips irrelevant chunks)
  - Recent data stays in memory
  - Parallel query execution

Data_Management:
  - Drop old chunks instantly (vs slow DELETE)
  - Compress historical chunks
  - Automatic retention policies
```

### PostgreSQL Compatibility

```sql
-- TimescaleDB is fully PostgreSQL-compatible
-- All PostgreSQL features work:

-- Standard PostgreSQL queries
SELECT * FROM metrics WHERE time > NOW() - INTERVAL '1 hour';

-- JOINs with regular tables
SELECT m.*, d.name
FROM metrics m
JOIN devices d ON m.device_id = d.id;

-- Extensions (PostGIS, pg_stat_statements, etc.)
CREATE EXTENSION postgis;
SELECT ST_Distance(location, 'POINT(0 0)') FROM sensors;

-- Full-text search
CREATE INDEX idx_fts ON events USING gin(to_tsvector('english', description));
```

## 2A. Test-Driven Development (TDD) Protocol (MANDATORY)

**CRITICAL: Follow the Red-Green-Refactor cycle for ALL new code.**

### TDD Cycle

```
1. RED: Write a failing test first
   ↓
2. GREEN: Write minimal code to make it pass
   ↓
3. REFACTOR: Improve code while keeping tests green
   ↓
   Repeat
```

### Example TDD Workflow for TimescaleDB

```python
# Step 1: RED - Write failing test
import pytest
import psycopg2
from datetime import datetime, timedelta, timezone

@pytest.fixture
def db_conn():
    conn = psycopg2.connect("dbname=testdb user=postgres host=localhost")
    conn.autocommit = True
    cur = conn.cursor()
    # Setup: create hypertable and continuous aggregate
    cur.execute("DROP TABLE IF EXISTS sensor_data CASCADE;")
    cur.execute("""
        CREATE TABLE sensor_data (
            time TIMESTAMPTZ NOT NULL,
            sensor_id INTEGER NOT NULL,
            temperature DOUBLE PRECISION,
            humidity DOUBLE PRECISION
        );
    """)
    cur.execute("SELECT create_hypertable('sensor_data', 'time');")
    yield conn
    cur.execute("DROP TABLE IF EXISTS sensor_data CASCADE;")
    conn.close()

def test_continuous_aggregate_hourly_avg(db_conn):
    """Test that continuous aggregate correctly computes hourly averages."""
    cur = db_conn.cursor()

    # Create continuous aggregate
    cur.execute("""
        CREATE MATERIALIZED VIEW sensor_hourly
        WITH (timescaledb.continuous) AS
        SELECT
            time_bucket('1 hour', time) AS bucket,
            sensor_id,
            AVG(temperature) AS avg_temp,
            AVG(humidity) AS avg_humidity
        FROM sensor_data
        GROUP BY bucket, sensor_id
        WITH NO DATA;
    """)

    # Insert test data: 4 readings in one hour for sensor 1
    base_time = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
    temps = [20.0, 22.0, 24.0, 26.0]  # avg = 23.0
    for i, temp in enumerate(temps):
        cur.execute(
            "INSERT INTO sensor_data (time, sensor_id, temperature, humidity) VALUES (%s, %s, %s, %s)",
            (base_time + timedelta(minutes=i * 15), 1, temp, 50.0)
        )

    # Refresh the continuous aggregate
    cur.execute("""
        CALL refresh_continuous_aggregate('sensor_hourly', '2024-01-15 10:00:00+00', '2024-01-15 11:00:00+00');
    """)

    # Verify
    cur.execute("""
        SELECT avg_temp FROM sensor_hourly
        WHERE sensor_id = 1 AND bucket = '2024-01-15 10:00:00+00'
    """)
    result = cur.fetchone()
    assert result is not None, "Continuous aggregate returned no rows"
    assert result[0] == 23.0, f"Expected avg_temp=23.0, got {result[0]}"

# FAILS - continuous aggregate not yet created in production code

# Step 2: GREEN - Implement continuous aggregate creation
def create_sensor_hourly_aggregate(conn):
    cur = conn.cursor()
    cur.execute("""
        CREATE MATERIALIZED VIEW IF NOT EXISTS sensor_hourly
        WITH (timescaledb.continuous) AS
        SELECT
            time_bucket('1 hour', time) AS bucket,
            sensor_id,
            AVG(temperature) AS avg_temp,
            AVG(humidity) AS avg_humidity
        FROM sensor_data
        GROUP BY bucket, sensor_id
        WITH NO DATA;
    """)

# PASSES

# Step 3: REFACTOR - Add refresh policy, compression on aggregate
def create_sensor_hourly_aggregate(conn):
    cur = conn.cursor()
    cur.execute("""
        CREATE MATERIALIZED VIEW IF NOT EXISTS sensor_hourly
        WITH (timescaledb.continuous) AS
        SELECT
            time_bucket('1 hour', time) AS bucket,
            sensor_id,
            AVG(temperature) AS avg_temp,
            AVG(humidity) AS avg_humidity,
            COUNT(*) AS sample_count
        FROM sensor_data
        GROUP BY bucket, sensor_id
        WITH NO DATA;
    """)
    cur.execute("""
        SELECT add_continuous_aggregate_policy('sensor_hourly',
            start_offset => INTERVAL '3 hours',
            end_offset => INTERVAL '1 hour',
            schedule_interval => INTERVAL '1 hour');
    """)
```

---

## 2B. Bug Fix Protocol (MANDATORY)

**CRITICAL: Every bug MUST receive a regression test BEFORE fixing.**

### Bug Fix Workflow

```
1. Bug Reported/Discovered
   ↓
2. Write a test that REPRODUCES the bug (test will FAIL)
   ↓
3. Verify the test fails for the right reason
   ↓
4. Fix the bug (make the test pass)
   ↓
5. Verify the test now PASSES
   ↓
6. Document the bug in test comments (include bug ID)
   ↓
7. Deploy with confidence (regression prevented)
```

### Example Bug Fix

```python
# Bug Report: BUG-4015 - Compressed chunks return incorrect results
# when querying with a time range that spans a compression boundary.

import pytest
import psycopg2
from datetime import datetime, timedelta, timezone

def test_bug_4015_compressed_chunk_boundary_query(db_conn):
    """Regression test: Queries spanning compressed/uncompressed chunks must return all rows."""
    cur = db_conn.cursor()

    # Enable compression on the hypertable
    cur.execute("""
        ALTER TABLE sensor_data SET (
            timescaledb.compress,
            timescaledb.compress_segmentby = 'sensor_id',
            timescaledb.compress_orderby = 'time DESC'
        );
    """)

    # Insert data across two chunks (assuming 7-day chunk interval)
    base_time = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    for day in range(14):  # 14 days of data across 2 chunks
        cur.execute(
            "INSERT INTO sensor_data (time, sensor_id, temperature, humidity) VALUES (%s, %s, %s, %s)",
            (base_time + timedelta(days=day), 1, 20.0 + day, 50.0)
        )

    # Compress only the first chunk (older data)
    cur.execute("""
        SELECT compress_chunk(c.chunk_name)
        FROM timescaledb_information.chunks c
        WHERE c.hypertable_name = 'sensor_data'
        ORDER BY c.range_start
        LIMIT 1;
    """)

    # Query spanning both compressed and uncompressed chunks
    cur.execute("""
        SELECT COUNT(*) FROM sensor_data
        WHERE sensor_id = 1
          AND time >= '2024-01-01'::timestamptz
          AND time < '2024-01-15'::timestamptz
    """)
    result = cur.fetchone()
    assert result[0] == 14, (
        f"BUG-4015: Expected 14 rows spanning compressed/uncompressed chunks, got {result[0]}"
    )

# Fix: Ensured compression policy uses segment_by that aligns with query filters,
# and verified decompress_chunk is not needed for standard SELECT queries.
```

### Prohibited Practices for Bug Fixes

**NEVER:**
- Fix a bug without adding a regression test first
- Write implementation before writing tests (violates TDD)
- Skip the Red-Green-Refactor cycle
- Commit code with failing tests
- Remove tests to make code pass
- Modify production schema without migration tests

---

## 3. Data Modeling for Time-Series

### Schema Design Principles

```sql
-- Best Practice: Include time column + metadata + metrics
CREATE TABLE sensor_data (
    time        TIMESTAMPTZ NOT NULL,           -- Time column (REQUIRED)
    sensor_id   INTEGER NOT NULL,                -- Device identifier
    location    TEXT NOT NULL,                   -- Metadata
    temperature DOUBLE PRECISION,                -- Metric
    humidity    DOUBLE PRECISION,                -- Metric
    pressure    DOUBLE PRECISION                 -- Metric
);

-- Convert to hypertable (7-day chunks by default)
SELECT create_hypertable('sensor_data', 'time');
```

### Data Type Selection

```sql
-- Time columns: ALWAYS use TIMESTAMPTZ
CREATE TABLE metrics (
    time TIMESTAMPTZ NOT NULL,                  -- ✅ CORRECT: Timezone-aware
    -- time TIMESTAMP NOT NULL,                 -- ❌ WRONG: No timezone info
    value DOUBLE PRECISION
);

-- Numeric data: Use appropriate precision
CREATE TABLE measurements (
    time TIMESTAMPTZ NOT NULL,
    -- Integer metrics
    count INTEGER,                               -- ✅ Counters, IDs
    -- Floating point
    temperature DOUBLE PRECISION,                -- ✅ Scientific measurements
    -- Exact decimals (for financial data)
    price NUMERIC(12, 2),                        -- ✅ Money values
    -- Boolean
    is_active BOOLEAN                            -- ✅ Status flags
);
```

### Narrow vs Wide Tables

```yaml
# Decision matrix for table design
Narrow_Tables:
  Structure: Few columns, many rows
  Use_When:
    - Metrics change frequently
    - Different sampling rates
    - Need flexible schema
  Example: "metric_name, time, value, tags"

Wide_Tables:
  Structure: Many columns, fewer rows
  Use_When:
    - Fixed set of metrics
    - Same sampling rate
    - Related measurements
  Example: "time, temp, humidity, pressure, wind_speed"

Recommendation:
  - Wide tables for related metrics (sensor readings)
  - Narrow tables for heterogeneous data (logs, events)
  - TimescaleDB handles both efficiently
```

### Table Design Examples

```sql
-- Example 1: Wide table (recommended for sensors)
CREATE TABLE weather_stations (
    time            TIMESTAMPTZ NOT NULL,
    station_id      INTEGER NOT NULL,
    temperature     DOUBLE PRECISION,
    humidity        DOUBLE PRECISION,
    pressure        DOUBLE PRECISION,
    wind_speed      DOUBLE PRECISION,
    wind_direction  DOUBLE PRECISION,
    rainfall        DOUBLE PRECISION
);

SELECT create_hypertable('weather_stations', 'time');
CREATE INDEX ON weather_stations (station_id, time DESC);

-- Example 2: Narrow table (for flexible metrics)
CREATE TABLE application_metrics (
    time        TIMESTAMPTZ NOT NULL,
    metric_name TEXT NOT NULL,
    value       DOUBLE PRECISION,
    tags        JSONB
);

SELECT create_hypertable('application_metrics', 'time');
CREATE INDEX ON application_metrics (metric_name, time DESC);
CREATE INDEX ON application_metrics USING gin(tags);
```

### Metadata Tables

```sql
-- Best Practice: Store metadata in separate tables
CREATE TABLE devices (
    device_id   SERIAL PRIMARY KEY,
    device_name TEXT NOT NULL,
    location    TEXT,
    install_date DATE,
    metadata    JSONB
);

CREATE TABLE device_metrics (
    time        TIMESTAMPTZ NOT NULL,
    device_id   INTEGER NOT NULL REFERENCES devices(device_id),
    cpu_usage   DOUBLE PRECISION,
    memory_used BIGINT,
    disk_io     BIGINT
);

SELECT create_hypertable('device_metrics', 'time');

-- Efficient JOINs with metadata
SELECT d.device_name, d.location, m.cpu_usage
FROM device_metrics m
JOIN devices d ON m.device_id = d.device_id
WHERE m.time > NOW() - INTERVAL '1 hour';
```

## 4. Hypertable Creation and Configuration

### Creating Hypertables

```sql
-- Basic hypertable creation
CREATE TABLE conditions (
    time        TIMESTAMPTZ NOT NULL,
    location    TEXT NOT NULL,
    temperature DOUBLE PRECISION,
    humidity    DOUBLE PRECISION
);

-- Convert to hypertable (must be called BEFORE inserting data)
SELECT create_hypertable('conditions', 'time');

-- With chunk interval specification
SELECT create_hypertable(
    'conditions',
    'time',
    chunk_time_interval => INTERVAL '1 day'  -- Override 7-day default
);

-- With space partitioning (for distributed hypertables)
SELECT create_hypertable(
    'conditions',
    'time',
    partitioning_column => 'location',
    number_partitions => 4,                   -- Hash partitioning
    chunk_time_interval => INTERVAL '1 day'
);
```

### Constraints and Indexes

```sql
-- Primary key must include time column
CREATE TABLE metrics (
    time        TIMESTAMPTZ NOT NULL,
    device_id   INTEGER NOT NULL,
    value       DOUBLE PRECISION,
    PRIMARY KEY (time, device_id)             -- ✅ CORRECT
    -- PRIMARY KEY (device_id)                -- ❌ ERROR: must include time
);

SELECT create_hypertable('metrics', 'time');

-- Create indexes AFTER hypertable creation
CREATE INDEX ON metrics (device_id, time DESC);
CREATE INDEX ON metrics (time DESC, device_id);  -- Different query patterns

-- Partial index for recent data
CREATE INDEX ON metrics (device_id, time DESC)
WHERE time > NOW() - INTERVAL '7 days';
```

### Hypertable Options

```sql
-- Disable compression initially (enable later)
SELECT create_hypertable(
    'metrics',
    'time',
    if_not_exists => TRUE,
    migrate_data => TRUE                       -- Migrate existing data
);

-- Check hypertable configuration
SELECT * FROM timescaledb_information.hypertables
WHERE hypertable_name = 'metrics';

-- Modify chunk interval after creation
SELECT set_chunk_time_interval('metrics', INTERVAL '12 hours');
```

## 5. Chunk Sizing and Time Partitioning

### The 25% Memory Rule

**Best Practice:** Set `chunk_time_interval` so that one chunk (including indexes) occupies **25% of main memory**.

```yaml
# Chunk sizing calculation
Available_Memory: 32 GB
Target_Chunk_Size: 8 GB (25% of memory)
Data_Rate: 100 MB/hour
Estimated_Chunk_Interval: 8000 MB / 100 MB/hour = 80 hours ≈ 3 days

Configuration:
  SELECT set_chunk_time_interval('metrics', INTERVAL '3 days');
```

### Default Chunk Intervals

```sql
-- Default: 7 days for new hypertables
-- Check current setting:
SELECT h.table_name, h.chunk_sizing_func_name,
       d.interval_length
FROM _timescaledb_catalog.hypertable h
JOIN _timescaledb_catalog.dimension d ON h.id = d.hypertable_id;

-- Common intervals by use case:
-- High ingestion (1M+ points/sec): 12-24 hours
SELECT set_chunk_time_interval('high_volume_metrics', INTERVAL '12 hours');

-- Standard ingestion (10K-100K points/sec): 1-7 days
SELECT set_chunk_time_interval('standard_metrics', INTERVAL '1 day');

-- Low ingestion (<10K points/sec): 7-30 days
SELECT set_chunk_time_interval('low_volume_metrics', INTERVAL '7 days');
```

### Monitoring Chunk Size

```sql
-- View chunk sizes
SELECT chunk_schema || '.' || chunk_name AS chunk,
       pg_size_pretty(total_bytes) AS total_size,
       pg_size_pretty(table_bytes) AS table_size,
       pg_size_pretty(index_bytes) AS index_size,
       range_start, range_end
FROM timescaledb_information.chunks
WHERE hypertable_name = 'metrics'
ORDER BY range_start DESC;

-- Check if chunks fit 25% rule
SELECT chunk_schema || '.' || chunk_name AS chunk,
       total_bytes / 1024 / 1024 AS size_mb,
       (SELECT setting::bigint FROM pg_settings WHERE name = 'shared_buffers') * 8 / 1024 AS shared_buffers_mb,
       CASE
           WHEN total_bytes < (SELECT setting::bigint FROM pg_settings WHERE name = 'shared_buffers') * 8 * 0.25
           THEN 'OK'
           ELSE 'TOO LARGE'
       END AS status
FROM timescaledb_information.chunks
WHERE hypertable_name = 'metrics';
```

### Space Partitioning

```sql
-- Use space partitioning for:
-- 1. Distributed hypertables (multi-node)
-- 2. High parallelism needs
-- 3. Device/location-based queries

-- Space partitions = number of data nodes (distributed)
-- Or: 1-4 partitions for single node (rarely needed)
SELECT create_hypertable(
    'distributed_metrics',
    'time',
    partitioning_column => 'device_id',
    number_partitions => 4,                    -- Matches number of data nodes
    chunk_time_interval => INTERVAL '1 day'
);

-- Important: Space partitioning uses HASH
-- Not range-based, so doesn't help with "WHERE device_id = X" queries
-- Primarily for load balancing across nodes
```

### Chunk Management

```sql
-- Show chunk information
SELECT show_chunks('metrics');

-- Drop old chunks manually
SELECT drop_chunks('metrics', INTERVAL '90 days');

-- Compress specific chunk
SELECT compress_chunk('_timescaledb_internal._hyper_1_1_chunk');

-- Decompress chunk (for modifications)
SELECT decompress_chunk('_timescaledb_internal._hyper_1_1_chunk');

-- Recompress after updates
SELECT compress_chunk('_timescaledb_internal._hyper_1_1_chunk', if_not_compressed => TRUE);
```

## 6. Compression Strategies

### Native Columnar Compression

TimescaleDB provides native columnar compression achieving **90%+ compression ratios** on typical time-series data.

```yaml
# Compression results (2026 production data)
Compression_Ratios:
  Typical: 91-92% reduction
  Storage_Saved: 10x space savings
  Best_Case: 90x compression (with hypercore)

Performance:
  Query_Speed: Maintained or improved
  Decompression: Automatic and transparent
  Write_Path: Uncompressed chunks, compress later
```

### Enabling Compression

```sql
-- Step 1: Enable compression on hypertable
ALTER TABLE metrics SET (
    timescaledb.compress,
    timescaledb.compress_orderby = 'time DESC',
    timescaledb.compress_segmentby = 'device_id'
);

-- Step 2: Create compression policy (auto-compress old chunks)
SELECT add_compression_policy('metrics', INTERVAL '7 days');

-- This means: Compress chunks older than 7 days
```

### Compression Configuration Parameters

```sql
-- compress_orderby: Orders data within compressed batches
-- Use DESC for time to optimize recent data queries
ALTER TABLE sensor_data SET (
    timescaledb.compress,
    timescaledb.compress_orderby = 'time DESC'
);

-- compress_segmentby: Groups similar data together
-- Improves compression ratio and query performance
ALTER TABLE sensor_data SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'sensor_id, location',
    timescaledb.compress_orderby = 'time DESC'
);

-- Multiple orderby columns
ALTER TABLE metrics SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'device_id',
    timescaledb.compress_orderby = 'time DESC, metric_id'
);
```

### Compression Best Practices

```yaml
Segmentby_Selection:
  Use_For:
    - Columns in WHERE clauses
    - GROUP BY columns
    - Low-to-medium cardinality columns
  Avoid:
    - High cardinality columns (UUIDs, timestamps)
    - Columns with unique values per row

Orderby_Selection:
  Primary: time DESC (always)
  Secondary: Frequently sorted columns

Batch_Size:
  Default: 1000 rows per batch
  Impact: Larger batches = better compression
  Sparse_Data: May result in smaller batches, reduced compression

Compression_Timing:
  Immediate: Not recommended (high overhead)
  Recommended: 1-7 days after chunk creation
  Conservative: After chunk is fully written (no more inserts)
```

### Compression Policies

```sql
-- Auto-compress chunks older than 7 days
SELECT add_compression_policy('metrics', INTERVAL '7 days');

-- Compress more aggressively (1 day)
SELECT add_compression_policy('high_volume_data', INTERVAL '1 day');

-- Remove compression policy
SELECT remove_compression_policy('metrics');

-- Modify compression policy
SELECT remove_compression_policy('metrics');
SELECT add_compression_policy('metrics', INTERVAL '3 days');

-- View compression policies
SELECT * FROM timescaledb_information.jobs
WHERE proc_name = 'policy_compression';
```

### Manual Compression

```sql
-- Compress all eligible chunks
SELECT compress_chunk(chunk)
FROM show_chunks('metrics', older_than => INTERVAL '7 days') AS chunk;

-- Compress specific chunk
SELECT compress_chunk('_timescaledb_internal._hyper_1_1_chunk');

-- Check compression status
SELECT chunk_schema || '.' || chunk_name AS chunk,
       pg_size_pretty(before_compression_total_bytes) AS before,
       pg_size_pretty(after_compression_total_bytes) AS after,
       100 - (after_compression_total_bytes::float / before_compression_total_bytes * 100) AS compression_pct
FROM chunk_compression_stats('metrics')
ORDER BY chunk;
```

### Compression Algorithms

TimescaleDB uses **seven specialized compression algorithms** optimized for different data types:

```yaml
# Compression algorithms (automatic selection)
Algorithms:
  Dictionary_Encoding: For repeated values
  Delta_Encoding: For sequential values (timestamps)
  Run_Length_Encoding: For consecutive identical values
  LZ_Compression: General purpose
  Frame_of_Reference: For bounded numeric ranges
  XOR: For floating point (time-series optimization)
  Gorilla: For scientific/sensor data (double precision)

Selection: Automatic based on data patterns
Optimization: Per-column, per-batch
```

### Decompression and Modifications

```sql
-- Compressed chunks are read-only
-- To modify data, decompress first:

-- Decompress specific chunk
SELECT decompress_chunk('_timescaledb_internal._hyper_1_1_chunk');

-- Modify data
UPDATE metrics
SET value = value * 1.1
WHERE time BETWEEN '2024-01-01' AND '2024-01-02';

-- Recompress
SELECT compress_chunk('_timescaledb_internal._hyper_1_1_chunk');

-- Alternative: Use INSERT ON CONFLICT for updates
-- (works on compressed chunks via decompression-recompression)
INSERT INTO metrics (time, device_id, value)
VALUES ('2024-01-01 10:00:00', 1, 25.5)
ON CONFLICT (time, device_id) DO UPDATE
SET value = EXCLUDED.value;
```

## 7. Continuous Aggregates

### What are Continuous Aggregates?

Continuous aggregates are automatically updated materialized views optimized for time-series data. They precompute aggregations incrementally, drastically improving query performance.

```yaml
# Benefits
Performance: 100x-1000x faster than raw queries
Incremental: Only processes new/changed data
Real_Time: Can include latest raw data
Storage: Trades storage for query speed
Automatic: Refresh policies handle updates
```

### Creating Continuous Aggregates

```sql
-- Basic continuous aggregate (5-minute averages)
CREATE MATERIALIZED VIEW conditions_5min
WITH (timescaledb.continuous) AS
SELECT time_bucket('5 minutes', time) AS bucket,
       location,
       AVG(temperature) AS avg_temp,
       MAX(temperature) AS max_temp,
       MIN(temperature) AS min_temp
FROM conditions
GROUP BY bucket, location;

-- Create refresh policy (auto-update every 30 minutes)
SELECT add_continuous_aggregate_policy('conditions_5min',
    start_offset => INTERVAL '3 hours',         -- Backfill window
    end_offset => INTERVAL '1 hour',            -- Exclude recent data
    schedule_interval => INTERVAL '30 minutes'  -- Refresh frequency
);
```

### Real-Time Aggregation

```sql
-- Enable real-time aggregation (includes non-materialized data)
CREATE MATERIALIZED VIEW conditions_hourly
WITH (timescaledb.continuous, timescaledb.materialized_only=false) AS
SELECT time_bucket('1 hour', time) AS bucket,
       device_id,
       AVG(value) AS avg_value,
       COUNT(*) AS num_readings
FROM metrics
GROUP BY bucket, device_id;

-- When querying:
-- materialized_only=false: Returns materialized + raw recent data
-- materialized_only=true: Only returns materialized data (faster, may be stale)
```

### Refresh Policies

```sql
-- Continuous aggregate refresh policy parameters:
-- start_offset: How far back to refresh (from now)
-- end_offset: How far forward to exclude (from now)
-- schedule_interval: How often to run refresh

-- Example: Refresh last 3 hours every 30 minutes, excluding last hour
SELECT add_continuous_aggregate_policy('conditions_5min',
    start_offset => INTERVAL '3 hours',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '30 minutes'
);

-- Why exclude end_offset?
-- - Avoids refreshing incomplete time buckets
-- - Prevents unnecessary re-computation
-- - Waits for late-arriving data

-- Typical configurations:
-- High-frequency data (seconds): end_offset = 5-15 minutes
-- Medium-frequency (minutes): end_offset = 1 hour
-- Low-frequency (hours): end_offset = 1 day
```

### Multiple Levels of Aggregation

```sql
-- Layer 1: 1-minute aggregates from raw data
CREATE MATERIALIZED VIEW metrics_1min
WITH (timescaledb.continuous) AS
SELECT time_bucket('1 minute', time) AS bucket,
       device_id,
       AVG(value) AS avg_value,
       MAX(value) AS max_value,
       MIN(value) AS min_value
FROM raw_metrics
GROUP BY bucket, device_id;

-- Layer 2: 1-hour aggregates from 1-minute aggregates
CREATE MATERIALIZED VIEW metrics_1hour
WITH (timescaledb.continuous) AS
SELECT time_bucket('1 hour', bucket) AS bucket,
       device_id,
       AVG(avg_value) AS avg_value,  -- Note: Nested aggregation
       MAX(max_value) AS max_value,
       MIN(min_value) AS min_value
FROM metrics_1min
GROUP BY bucket, device_id;

-- Layer 3: 1-day aggregates from 1-hour aggregates
CREATE MATERIALIZED VIEW metrics_1day
WITH (timescaledb.continuous) AS
SELECT time_bucket('1 day', bucket) AS bucket,
       device_id,
       AVG(avg_value) AS avg_value,
       MAX(max_value) AS max_value,
       MIN(min_value) AS min_value
FROM metrics_1hour
GROUP BY bucket, device_id;
```

### Querying Continuous Aggregates

```sql
-- Query like a regular table (automatic optimization)
SELECT * FROM conditions_5min
WHERE bucket >= NOW() - INTERVAL '1 day'
  AND location = 'office'
ORDER BY bucket DESC;

-- Force materialized_only (ignores recent raw data)
SELECT * FROM conditions_5min
WHERE bucket >= NOW() - INTERVAL '1 day'
  AND timescaledb.materialized_only = true;

-- Join with dimension tables
SELECT c.bucket, d.device_name, c.avg_temp
FROM conditions_5min c
JOIN devices d ON c.device_id = d.id
WHERE c.bucket > NOW() - INTERVAL '6 hours';
```

### Maintenance and Operations

```sql
-- View continuous aggregate info
SELECT * FROM timescaledb_information.continuous_aggregates;

-- View refresh policies
SELECT * FROM timescaledb_information.jobs
WHERE proc_name = 'policy_refresh_continuous_aggregate';

-- Manual refresh (force update)
CALL refresh_continuous_aggregate('conditions_5min',
    '2024-01-01',
    '2024-01-31'
);

-- Drop continuous aggregate
DROP MATERIALIZED VIEW conditions_5min;

-- Alter refresh policy
SELECT remove_continuous_aggregate_policy('conditions_5min');
SELECT add_continuous_aggregate_policy('conditions_5min',
    start_offset => INTERVAL '6 hours',
    end_offset => INTERVAL '30 minutes',
    schedule_interval => INTERVAL '15 minutes'
);
```

### Advanced Continuous Aggregates

```sql
-- With FILTER clause
CREATE MATERIALIZED VIEW error_counts
WITH (timescaledb.continuous) AS
SELECT time_bucket('10 minutes', time) AS bucket,
       service_name,
       COUNT(*) FILTER (WHERE status = 'error') AS error_count,
       COUNT(*) FILTER (WHERE status = 'warning') AS warning_count,
       COUNT(*) AS total_count
FROM logs
GROUP BY bucket, service_name;

-- With window functions (use subquery)
CREATE MATERIALIZED VIEW metrics_with_delta
WITH (timescaledb.continuous) AS
SELECT bucket, device_id, avg_value,
       avg_value - LAG(avg_value) OVER (PARTITION BY device_id ORDER BY bucket) AS delta
FROM (
    SELECT time_bucket('1 hour', time) AS bucket,
           device_id,
           AVG(value) AS avg_value
    FROM metrics
    GROUP BY bucket, device_id
) subq;

-- Percentile aggregation (approximation)
CREATE MATERIALIZED VIEW response_times
WITH (timescaledb.continuous) AS
SELECT time_bucket('5 minutes', time) AS bucket,
       endpoint,
       percentile_agg(response_time) AS pct_agg  -- Approximate percentile
FROM requests
GROUP BY bucket, endpoint;

-- Query percentiles
SELECT bucket, endpoint,
       approx_percentile(0.50, pct_agg) AS p50,
       approx_percentile(0.95, pct_agg) AS p95,
       approx_percentile(0.99, pct_agg) AS p99
FROM response_times
WHERE bucket > NOW() - INTERVAL '1 day';
```

## 8. Retention Policies and Data Lifecycle

### Automatic Data Retention

```sql
-- Add retention policy (drop chunks older than 90 days)
SELECT add_retention_policy('metrics', INTERVAL '90 days');

-- This automatically:
-- 1. Identifies chunks older than 90 days
-- 2. Drops entire chunks (fast, no DELETE)
-- 3. Runs on schedule (default: daily)

-- View retention policies
SELECT * FROM timescaledb_information.jobs
WHERE proc_name = 'policy_retention';

-- Remove retention policy
SELECT remove_retention_policy('metrics');
```

### Data Lifecycle Strategy

```yaml
# Typical data lifecycle
Hot_Data:
  Age: 0-7 days
  Storage: Uncompressed
  Performance: Fastest writes and queries
  Location: Recent chunks

Warm_Data:
  Age: 7-90 days
  Storage: Compressed (native columnar)
  Performance: Fast queries, slower writes (decompress needed)
  Retention: Compressed chunks

Cold_Data:
  Age: 90+ days
  Storage: Continuous aggregates only
  Performance: Very fast (pre-aggregated)
  Raw_Data: Dropped via retention policy

Archive_Data:
  Age: 1+ years
  Storage: External (S3, data lake)
  Access: Rare, via batch queries
  Method: Export before dropping
```

### Multi-Tier Data Management

```sql
-- Layer 1: Raw data retention (7 days)
SELECT add_retention_policy('raw_metrics', INTERVAL '7 days');

-- Layer 2: 1-minute aggregates (90 days)
CREATE MATERIALIZED VIEW metrics_1min
WITH (timescaledb.continuous) AS
SELECT time_bucket('1 minute', time) AS bucket,
       device_id,
       AVG(value) AS avg_value
FROM raw_metrics
GROUP BY bucket, device_id;

SELECT add_retention_policy('metrics_1min', INTERVAL '90 days');

-- Layer 3: 1-hour aggregates (2 years)
CREATE MATERIALIZED VIEW metrics_1hour
WITH (timescaledb.continuous) AS
SELECT time_bucket('1 hour', bucket) AS bucket,
       device_id,
       AVG(avg_value) AS avg_value
FROM metrics_1min
GROUP BY bucket, device_id;

SELECT add_retention_policy('metrics_1hour', INTERVAL '2 years');

-- Layer 4: Daily aggregates (forever)
CREATE MATERIALIZED VIEW metrics_daily
WITH (timescaledb.continuous) AS
SELECT time_bucket('1 day', bucket) AS bucket,
       device_id,
       AVG(avg_value) AS avg_value
FROM metrics_1hour
GROUP BY bucket, device_id;
-- No retention policy = keep forever
```

### Manual Chunk Dropping

```sql
-- Drop chunks manually (for one-time cleanup)
SELECT drop_chunks('metrics', OLDER_THAN => INTERVAL '1 year');

-- Drop specific time range
SELECT drop_chunks('metrics',
    older_than => '2023-12-31 23:59:59'::timestamptz,
    newer_than => '2023-01-01 00:00:00'::timestamptz
);

-- Dry run (see what would be dropped)
SELECT show_chunks('metrics', older_than => INTERVAL '90 days');
```

### Retention Policy Scheduling

```sql
-- View job schedule
SELECT job_id, schedule_interval, retry_period
FROM timescaledb_information.jobs
WHERE proc_name = 'policy_retention';

-- Modify job schedule (run every 6 hours instead of daily)
SELECT alter_job(job_id, schedule_interval => INTERVAL '6 hours')
FROM timescaledb_information.jobs
WHERE proc_name = 'policy_retention'
  AND hypertable_name = 'metrics';
```

## 9. Write Optimization

### Batching Writes

```sql
-- Bad: Row-by-row inserts (slow)
INSERT INTO metrics (time, device_id, value) VALUES ('2024-01-01 10:00:00', 1, 25.5);
INSERT INTO metrics (time, device_id, value) VALUES ('2024-01-01 10:00:01', 1, 25.6);
-- ... 1000 more inserts

-- Good: Batch inserts (100x-1000x faster)
INSERT INTO metrics (time, device_id, value) VALUES
    ('2024-01-01 10:00:00', 1, 25.5),
    ('2024-01-01 10:00:01', 1, 25.6),
    ('2024-01-01 10:00:02', 1, 25.7)
    -- ... 1000+ rows
;

-- Best: COPY command (fastest for bulk loading)
COPY metrics (time, device_id, value) FROM STDIN CSV;
2024-01-01 10:00:00,1,25.5
2024-01-01 10:00:01,1,25.6
-- ... millions of rows
\.
```

### Optimal Batch Sizes

```yaml
# Batch size recommendations (2026)
Batch_Sizes:
  Small_Batches: 100-500 rows
    Use: Low-latency requirements, small transactions

  Medium_Batches: 1,000-5,000 rows
    Use: Balanced throughput and latency

  Large_Batches: 10,000-50,000 rows
    Use: High-throughput ingestion, batch ETL

  Bulk_Loading: 100,000+ rows
    Use: Initial data loading, COPY command

Performance_Impact:
  Throughput: 50,000+ inserts/second on modest hardware
  Network: Batching reduces round trips
  Transactions: Fewer transaction commits
```

### Parallel Inserts

```sql
-- TimescaleDB supports parallel inserts
-- Use multiple connections for higher throughput

-- Connection 1:
INSERT INTO metrics SELECT * FROM staging_table_1;

-- Connection 2 (parallel):
INSERT INTO metrics SELECT * FROM staging_table_2;

-- Connection 3 (parallel):
INSERT INTO metrics SELECT * FROM staging_table_3;

-- Important: Each INSERT targets different chunks for best performance
-- Avoid concurrent inserts to the same chunk (causes lock contention)
```

### Using COPY for Bulk Loading

```bash
# timescaledb-parallel-copy tool
# Parallelizes PostgreSQL COPY for maximum throughput

# Install
go install github.com/timescale/timescaledb-parallel-copy/cmd/timescaledb-parallel-copy@latest

# Usage (default: 5000 rows per batch)
timescaledb-parallel-copy \
    --connection "host=localhost user=postgres password=pwd dbname=mydb" \
    --table metrics \
    --file data.csv \
    --workers 4 \
    --reporting-period 1s

# Performance: Can achieve 1M+ rows/second
```

### Write Performance Configuration

```sql
-- PostgreSQL settings for write optimization
-- postgresql.conf

-- Shared buffers (25% of RAM)
shared_buffers = 8GB

-- Write-ahead log
wal_buffers = 16MB
wal_writer_delay = 200ms

-- Checkpoints
checkpoint_completion_target = 0.9
max_wal_size = 4GB
min_wal_size = 1GB

-- Background workers
max_worker_processes = 16
timescaledb.max_background_workers = 8

-- Memory for maintenance
maintenance_work_mem = 2GB
max_parallel_maintenance_workers = 4
```

### Pre-creating Chunks

```sql
-- Avoid lock contention by pre-creating chunks
-- Useful for high-ingestion workloads

-- Pre-create next 7 days of chunks
SELECT create_chunk('metrics',
    '["2024-01-01 00:00:00","2024-01-02 00:00:00"]'::jsonb
);
SELECT create_chunk('metrics',
    '["2024-01-02 00:00:00","2024-01-03 00:00:00"]'::jsonb
);
-- ... etc

-- Or use a function to pre-create multiple chunks
DO $$
DECLARE
    i INT;
    start_time TIMESTAMPTZ := NOW();
BEGIN
    FOR i IN 1..7 LOOP
        PERFORM create_chunk('metrics',
            jsonb_build_array(
                start_time + (i - 1) * INTERVAL '1 day',
                start_time + i * INTERVAL '1 day'
            )
        );
    END LOOP;
END $$;
```

### Upserts and Conflicts

```sql
-- Efficient upsert (INSERT ... ON CONFLICT)
INSERT INTO metrics (time, device_id, value)
VALUES ('2024-01-01 10:00:00', 1, 25.5)
ON CONFLICT (time, device_id) DO UPDATE
SET value = EXCLUDED.value;

-- Batch upserts
INSERT INTO metrics (time, device_id, value)
VALUES
    ('2024-01-01 10:00:00', 1, 25.5),
    ('2024-01-01 10:00:01', 2, 30.2),
    ('2024-01-01 10:00:02', 3, 22.8)
ON CONFLICT (time, device_id) DO UPDATE
SET value = EXCLUDED.value;

-- Note: Upserts on compressed chunks require decompression
-- Avoid frequent updates to compressed data
```

## 10. Query Optimization

### Chunk Exclusion

TimescaleDB's query planner automatically excludes chunks that don't match query constraints:

```sql
-- Good: Time filter enables chunk exclusion
SELECT * FROM metrics
WHERE time > NOW() - INTERVAL '1 hour'  -- Only scans recent chunks
  AND device_id = 1;

-- Bad: No time filter (scans all chunks)
SELECT * FROM metrics
WHERE device_id = 1;  -- Scans entire hypertable

-- Query plan shows chunk exclusion:
EXPLAIN (ANALYZE) SELECT * FROM metrics
WHERE time > NOW() - INTERVAL '1 hour';

-- Look for "Chunks excluded during startup: N"
```

### Time-Based Queries

```sql
-- Best practices for time queries
-- ✅ Use time ranges (enables chunk exclusion)
SELECT * FROM metrics
WHERE time >= '2024-01-01' AND time < '2024-01-08';

-- ✅ Use NOW() - INTERVAL
SELECT * FROM metrics
WHERE time > NOW() - INTERVAL '24 hours';

-- ✅ Combine time + other filters
SELECT AVG(value) FROM metrics
WHERE time > NOW() - INTERVAL '1 hour'
  AND device_id IN (1, 2, 3)
GROUP BY device_id;

-- ❌ Avoid: No time filter
SELECT AVG(value) FROM metrics
WHERE device_id = 1;  -- Scans all historical data
```

### Indexing for Query Performance

```sql
-- Index strategy: time + frequently filtered columns
CREATE INDEX ON metrics (device_id, time DESC);

-- For queries like:
SELECT * FROM metrics
WHERE device_id = 1
  AND time > NOW() - INTERVAL '1 hour'
ORDER BY time DESC;

-- Composite index considerations
-- Order matters:
-- Index (device_id, time) - Good for "WHERE device_id = X AND time > Y"
-- Index (time, device_id) - Good for "WHERE time > Y" (without device filter)

-- Create both if needed:
CREATE INDEX ON metrics (device_id, time DESC);
CREATE INDEX ON metrics (time DESC);
```

### Aggregation Optimization

```sql
-- Use time_bucket for efficient time-based aggregation
SELECT time_bucket('5 minutes', time) AS bucket,
       device_id,
       AVG(value) AS avg_value
FROM metrics
WHERE time > NOW() - INTERVAL '24 hours'
GROUP BY bucket, device_id
ORDER BY bucket DESC;

-- time_bucket benefits:
-- - Optimized for TimescaleDB chunk structure
-- - Faster than DATE_TRUNC
-- - Supports custom intervals

-- Compare to slow version:
SELECT DATE_TRUNC('minute', time) AS minute,  -- Slower
       AVG(value)
FROM metrics
GROUP BY minute;
```

### Chunk-Skipping Indexes

```sql
-- Enable chunk-skipping on non-time columns (TimescaleDB 2.x+)
-- Tracks min/max values per chunk for fast exclusion

-- Create chunk-skipping index
CREATE INDEX ON metrics (device_id, time DESC)
INCLUDE (value);  -- Track value range

-- Enable skipscans
SELECT create_hypertable('metrics', 'time',
    chunk_time_interval => INTERVAL '1 day'
);

-- Queries benefit from chunk-skipping:
SELECT * FROM metrics
WHERE device_id = 5        -- Skips chunks without device_id=5
  AND value > 100;         -- Skips chunks with max value < 100

-- Check effectiveness:
EXPLAIN (ANALYZE, BUFFERS) SELECT ...
-- Look for "Heap Blocks: exact=X" (fewer = better)
```

### Join Optimization

```sql
-- Best practice: Join with dimension tables
SELECT m.time, d.device_name, m.value
FROM metrics m
JOIN devices d ON m.device_id = d.id
WHERE m.time > NOW() - INTERVAL '1 hour'
ORDER BY m.time DESC;

-- Tip: Index foreign keys
CREATE INDEX ON metrics (device_id);

-- For large joins, use continuous aggregates
CREATE MATERIALIZED VIEW device_stats
WITH (timescaledb.continuous) AS
SELECT time_bucket('1 hour', m.time) AS bucket,
       d.device_name,
       AVG(m.value) AS avg_value
FROM metrics m
JOIN devices d ON m.device_id = d.id
GROUP BY bucket, d.device_name;
```

### Query Performance Analysis

```sql
-- Use EXPLAIN ANALYZE
EXPLAIN (ANALYZE, BUFFERS, VERBOSE)
SELECT time_bucket('5 minutes', time) AS bucket,
       AVG(value)
FROM metrics
WHERE time > NOW() - INTERVAL '1 day'
  AND device_id = 1
GROUP BY bucket;

-- Key metrics to watch:
-- - Execution Time: Total query time
-- - Planning Time: Query planning overhead
-- - Shared Buffers Hit: Cache effectiveness
-- - Chunks excluded: Chunk exclusion working
-- - Seq Scan vs Index Scan: Index usage

-- Enable query logging
-- postgresql.conf:
log_min_duration_statement = 1000  # Log queries > 1 second
log_statement = 'all'               # Log all statements (dev only)
```

## 11. Indexing Best Practices

### Time-Based Indexes

```sql
-- Primary index pattern: time column
-- TimescaleDB automatically creates time index on hypertable creation

-- Additional time-based indexes
CREATE INDEX ON metrics (time DESC);          -- Reverse chronological
CREATE INDEX ON metrics (time ASC);           -- Chronological (rare)
CREATE INDEX ON metrics (device_id, time DESC);  -- Composite

-- Descending time indexes (DESC) preferred for:
-- - Recent data queries (most common)
-- - ORDER BY time DESC
-- - Latest value queries
```

### BRIN Indexes

BRIN (Block Range Index) indexes are ideal for large, time-ordered tables:

```sql
-- BRIN index on time column (space-efficient)
CREATE INDEX ON metrics USING brin(time) WITH (pages_per_range = 128);

-- Benefits:
-- - Very small index size (< 1% of B-tree)
-- - Fast creation
-- - Good for time-ordered data

-- Use cases:
-- - Very large tables (100GB+)
-- - Time-ordered inserts
-- - Range queries on time

-- Note: TimescaleDB chunks are already time-partitioned
-- BRIN typically not needed on time column
-- More useful for non-time columns with natural ordering
```

### GiST Indexes

```sql
-- GiST indexes for spatial/geometric data
CREATE EXTENSION postgis;

CREATE TABLE sensor_locations (
    time        TIMESTAMPTZ NOT NULL,
    sensor_id   INTEGER,
    location    GEOMETRY(Point, 4326),
    value       DOUBLE PRECISION
);

SELECT create_hypertable('sensor_locations', 'time');

-- GiST index on location
CREATE INDEX ON sensor_locations USING gist(location);

-- Spatial queries
SELECT * FROM sensor_locations
WHERE time > NOW() - INTERVAL '1 hour'
  AND ST_DWithin(location, ST_MakePoint(-73.935242, 40.730610)::geography, 1000);

-- GiST also useful for:
-- - Range types
-- - Full-text search (with tsvector)
-- - Network addresses (inet types)
```

### Partial Indexes

```sql
-- Index only recent/hot data
CREATE INDEX ON metrics (device_id, time DESC)
WHERE time > NOW() - INTERVAL '30 days';

-- Index specific conditions
CREATE INDEX ON events (time DESC)
WHERE severity = 'error';

-- Benefits:
-- - Smaller index size
-- - Faster maintenance
-- - Reduced I/O

-- Combine with retention policies:
-- If data is dropped after 90 days, index only last 90 days
CREATE INDEX ON metrics (device_id, time DESC)
WHERE time > NOW() - INTERVAL '90 days';
```

### Covering Indexes (INCLUDE)

```sql
-- Include non-key columns to avoid table lookups
CREATE INDEX ON metrics (device_id, time DESC)
INCLUDE (value, status);

-- This allows index-only scans for:
SELECT device_id, time, value, status
FROM metrics
WHERE device_id = 1
  AND time > NOW() - INTERVAL '1 hour';

-- Benefits:
-- - Faster queries (no heap access)
-- - Reduced I/O
-- - Better cache efficiency
```

### Index Maintenance

```sql
-- Monitor index usage
SELECT schemaname, tablename, indexname,
       idx_scan, idx_tup_read, idx_tup_fetch
FROM pg_stat_user_indexes
WHERE schemaname NOT LIKE 'pg_%'
ORDER BY idx_scan ASC;

-- Find unused indexes (idx_scan = 0)
SELECT schemaname || '.' || tablename AS table,
       indexname AS index,
       pg_size_pretty(pg_relation_size(indexrelid)) AS size
FROM pg_stat_user_indexes
WHERE idx_scan = 0
  AND indexrelname NOT LIKE '%pkey'
ORDER BY pg_relation_size(indexrelid) DESC;

-- Drop unused indexes
DROP INDEX IF EXISTS unused_index_name;

-- Reindex (after heavy updates/deletes)
REINDEX INDEX metrics_device_id_time_idx;
REINDEX TABLE metrics;  -- All indexes

-- Automatic reindexing (pg_cron)
CREATE EXTENSION pg_cron;
SELECT cron.schedule('0 2 * * 0', 'REINDEX TABLE metrics');  -- Weekly
```

### Index Best Practices Summary

```yaml
Index_Strategy:
  Time_Column:
    - TimescaleDB auto-creates (no manual index needed)
    - Use DESC for recent data queries

  Foreign_Keys:
    - Always index foreign key columns
    - Example: device_id, location_id

  Composite_Indexes:
    - Order: Most selective column first
    - Common pattern: (device_id, time DESC)

  Covering_Indexes:
    - Use INCLUDE for index-only scans
    - Balance: Larger index vs faster queries

  Partial_Indexes:
    - Index recent data only
    - Match retention policy window

  Avoid:
    - Too many indexes (slow inserts)
    - Unused indexes (waste space)
    - Redundant indexes (e.g., (a,b) + (a))

Per_Chunk_Indexes:
  Note: Each chunk has separate indexes
  Impact: N chunks × M indexes = total index count
  Guideline: Keep indexes minimal (3-5 per hypertable)
```

## 12. Memory Optimization and Cache Tuning

### PostgreSQL Memory Configuration

```sql
-- postgresql.conf - Memory settings for TimescaleDB

-- Shared buffers (25% of RAM, max 16GB)
shared_buffers = 8GB              -- For 32GB system

-- Effective cache size (50-75% of RAM)
effective_cache_size = 24GB       -- OS + PostgreSQL cache estimate

-- Work memory (per operation)
work_mem = 64MB                   -- For sorts, hashes
maintenance_work_mem = 2GB        -- For VACUUM, CREATE INDEX

-- WAL buffers
wal_buffers = 16MB                -- -1 = auto (1/32 of shared_buffers)
```

### timescaledb-tune

```bash
# Automated configuration tuning
# Install timescaledb-tune
sudo apt install timescaledb-tools

# Run tuning wizard
sudo timescaledb-tune

# Sample output for 32GB RAM, 8 CPU system:
# shared_buffers = 8GB (was 128MB)
# effective_cache_size = 24GB (was 4GB)
# maintenance_work_mem = 2GB (was 64MB)
# work_mem = 52MB (was 4MB)
# max_worker_processes = 11 (was 8)
# max_parallel_workers = 8 (was 8)
# timescaledb.max_background_workers = 8 (was 8)

# Apply changes and restart PostgreSQL
sudo systemctl restart postgresql
```

### Chunk Cache Optimization

```yaml
# Chunk caching strategy
Shared_Buffers:
  Purpose: In-memory cache for chunks
  Sizing: Recent chunks should fit in cache
  Target: Latest 25% of data in memory

Calculation:
  Total_Data: 400 GB
  Chunk_Size: 8 GB per chunk (1 day)
  Active_Chunks: 7 chunks (last 7 days) = 56 GB
  Shared_Buffers: 64 GB (can hold active data)

Configuration:
  shared_buffers = 64GB
  effective_cache_size = 192GB  # Total available cache
```

### Background Worker Configuration

```sql
-- postgresql.conf

-- Total worker processes
max_worker_processes = 16

-- Parallel query workers
max_parallel_workers_per_gather = 4
max_parallel_workers = 8

-- TimescaleDB background workers
timescaledb.max_background_workers = 8

-- Background worker pool for:
-- - Compression jobs
-- - Retention policies
-- - Continuous aggregate refreshes
-- - Reorder policies

-- Recommendation:
-- max_worker_processes >= max_parallel_workers + timescaledb.max_background_workers + 3
-- Example: 16 >= 8 + 8 + 3 = 19 (need to increase to 20)
```

### Query Memory Tuning

```sql
-- Per-query memory limits
SET work_mem = '256MB';  -- For current session

-- For specific heavy queries
SET work_mem = '1GB';
SELECT time_bucket('1 hour', time) AS bucket,
       device_id,
       percentile_cont(0.95) WITHIN GROUP (ORDER BY value) AS p95
FROM metrics
WHERE time > NOW() - INTERVAL '30 days'
GROUP BY bucket, device_id;
RESET work_mem;

-- Maintenance operations
SET maintenance_work_mem = '4GB';
CREATE INDEX ON large_table (time DESC);
RESET maintenance_work_mem;
```

### Connection Pooling

```yaml
# PgBouncer configuration
pgbouncer.ini:
  [databases]
  timescaledb = host=localhost port=5432 dbname=timescaledb

  [pgbouncer]
  listen_addr = 0.0.0.0
  listen_port = 6432
  auth_type = md5
  auth_file = /etc/pgbouncer/userlist.txt
  pool_mode = transaction        # Recommended for TimescaleDB
  max_client_conn = 1000
  default_pool_size = 25
  min_pool_size = 5
  reserve_pool_size = 5
  max_db_connections = 100

Pool_Modes:
  transaction: Best for TimescaleDB (reuses connections per transaction)
  session: Use if using prepared statements or temp tables
  statement: Rarely needed
```

### Vacuum and Autovacuum

```sql
-- postgresql.conf - Autovacuum tuning

-- Enable autovacuum (default: on)
autovacuum = on

-- Autovacuum parameters
autovacuum_max_workers = 4
autovacuum_naptime = 10s          -- Check interval

-- Per-table thresholds
autovacuum_vacuum_scale_factor = 0.05    -- Vacuum at 5% dead tuples
autovacuum_analyze_scale_factor = 0.02   -- Analyze at 2% changes

-- Cost-based vacuum delay (prevent I/O spikes)
autovacuum_vacuum_cost_delay = 2ms
autovacuum_vacuum_cost_limit = 400

-- Manual vacuum (for hypertables)
VACUUM (ANALYZE, VERBOSE) metrics;

-- Per-chunk vacuum
VACUUM (ANALYZE) _timescaledb_internal._hyper_1_1_chunk;
```

### Memory Monitoring

```sql
-- Check buffer cache hit ratio (target: >99%)
SELECT
    sum(heap_blks_read) as heap_read,
    sum(heap_blks_hit)  as heap_hit,
    sum(heap_blks_hit) / (sum(heap_blks_hit) + sum(heap_blks_read)) AS ratio
FROM pg_statio_user_tables;

-- Per-table cache statistics
SELECT schemaname, tablename,
       heap_blks_hit,
       heap_blks_read,
       heap_blks_hit::float / (heap_blks_hit + heap_blks_read) AS cache_hit_ratio
FROM pg_statio_user_tables
WHERE schemaname NOT IN ('pg_catalog', 'information_schema')
  AND heap_blks_read > 0
ORDER BY cache_hit_ratio ASC;

-- Shared buffer usage
SELECT * FROM pg_buffercache_summary();
```

## 13. Performance Tuning

### PostgreSQL + TimescaleDB Configuration

```sql
-- postgresql.conf - Production settings (32GB RAM, 8 CPU)

# Memory
shared_buffers = 8GB
effective_cache_size = 24GB
work_mem = 64MB
maintenance_work_mem = 2GB
wal_buffers = 16MB

# Checkpoints
checkpoint_completion_target = 0.9
max_wal_size = 4GB
min_wal_size = 1GB
checkpoint_timeout = 15min

# Query Planning
random_page_cost = 1.1              # For SSD
effective_io_concurrency = 200      # For SSD
default_statistics_target = 100

# Workers
max_worker_processes = 20
max_parallel_workers_per_gather = 4
max_parallel_workers = 8
timescaledb.max_background_workers = 8

# WAL
wal_level = replica
max_wal_senders = 3
wal_keep_size = 1GB

# Logging (for performance analysis)
log_min_duration_statement = 1000   # Log slow queries (>1s)
log_checkpoints = on
log_connections = on
log_disconnections = on
log_lock_waits = on
```

### I/O and Storage Optimization

```yaml
# Storage recommendations
Disk_Type:
  Production: NVMe SSD
  Minimum: SATA SSD
  Avoid: HDD (too slow for time-series)

File_System:
  Recommended: ext4, xfs
  Mount_Options:
    - noatime (reduce write overhead)
    - nodiratime
    - discard (for SSD TRIM)

Tablespace_Strategy:
  Hot_Data: NVMe SSD (recent chunks)
  Warm_Data: SATA SSD (compressed chunks)
  Cold_Data: S3 / object storage (archives)

# Example tablespace usage
CREATE TABLESPACE fast_storage LOCATION '/mnt/nvme';
CREATE TABLESPACE slow_storage LOCATION '/mnt/sata';

# Move old chunks to slow storage
SELECT move_chunk(
    chunk => '_timescaledb_internal._hyper_1_1_chunk',
    destination_tablespace => 'slow_storage'
);
```

### Parallel Query Execution

```sql
-- Enable parallelism for large queries
SET max_parallel_workers_per_gather = 4;

-- Parallel-friendly queries
SELECT time_bucket('1 hour', time) AS bucket,
       AVG(value) AS avg_value
FROM large_table
WHERE time > NOW() - INTERVAL '30 days'
GROUP BY bucket;

-- Check if parallel execution is used
EXPLAIN (ANALYZE)
SELECT COUNT(*) FROM large_table
WHERE time > NOW() - INTERVAL '7 days';

-- Look for "Parallel Seq Scan" or "Gather" nodes
```

### Query Plan Optimization

```sql
-- Update table statistics (for better query plans)
ANALYZE metrics;

-- Per-chunk statistics
ANALYZE _timescaledb_internal._hyper_1_1_chunk;

-- Increase statistics target for important columns
ALTER TABLE metrics ALTER COLUMN device_id SET STATISTICS 500;
ANALYZE metrics;

-- Disable sequential scans (force index usage - for testing)
SET enable_seqscan = off;
EXPLAIN SELECT ... ;
RESET enable_seqscan;
```

### Compression Performance Tuning

```sql
-- Monitor compression job performance
SELECT * FROM timescaledb_information.job_stats
WHERE job_id IN (
    SELECT job_id FROM timescaledb_information.jobs
    WHERE proc_name = 'policy_compression'
);

-- Tune compression job interval
SELECT alter_job(job_id, schedule_interval => INTERVAL '6 hours')
FROM timescaledb_information.jobs
WHERE proc_name = 'policy_compression'
  AND hypertable_name = 'metrics';

-- Compress during low-traffic periods
SELECT alter_job(job_id,
    schedule_interval => INTERVAL '1 day',
    next_start => '2024-01-01 02:00:00'  -- 2 AM daily
)
FROM timescaledb_information.jobs
WHERE proc_name = 'policy_compression';
```

### Low-Latency Configuration

```yaml
# Optimized for low write latency
PostgreSQL_Settings:
  synchronous_commit: off           # Trade durability for speed (careful!)
  wal_writer_delay: 10ms            # Flush WAL more frequently
  commit_delay: 0                   # No artificial delay

TimescaleDB_Settings:
  chunk_time_interval: 1-4 hours    # Smaller chunks for faster inserts
  Compression: Disabled on hot data # Avoid decompression overhead

Application:
  Batch_Size: 1,000-5,000 rows     # Balance latency vs throughput
  Connection_Pooling: Required      # Reuse connections
  Prepared_Statements: Recommended  # Reduce parsing overhead
```

### High-Throughput Configuration

```yaml
# Optimized for maximum write throughput
PostgreSQL_Settings:
  shared_buffers: 16GB+             # Large cache
  checkpoint_timeout: 30min         # Less frequent checkpoints
  max_wal_size: 8GB+                # More WAL before checkpoint

TimescaleDB_Settings:
  chunk_time_interval: 1-7 days    # Larger chunks
  Compression_Policy: 7+ days       # Compress older data

Application:
  Batch_Size: 10,000-100,000 rows  # Large batches
  Parallel_Writers: 4-16 connections # Concurrent inserts
  COPY_Command: Use for bulk loading # Fastest insert method
```

## 14. Security

### Authentication and SSL/TLS

```sql
-- postgresql.conf - Require SSL connections
ssl = on
ssl_cert_file = '/etc/postgresql/ssl/server.crt'
ssl_key_file = '/etc/postgresql/ssl/server.key'
ssl_ca_file = '/etc/postgresql/ssl/root.crt'

-- pg_hba.conf - Enforce SSL
hostssl    all    all    0.0.0.0/0    scram-sha-256

# Connection string with SSL
postgresql://user:password@host:5432/dbname?sslmode=verify-full
```

### Row-Level Security (RLS)

```sql
-- Enable RLS on hypertable
ALTER TABLE sensor_data ENABLE ROW LEVEL SECURITY;

-- Policy: Users see only their own devices
CREATE POLICY tenant_isolation ON sensor_data
    FOR SELECT
    USING (device_id IN (
        SELECT device_id FROM user_devices
        WHERE user_id = current_user
    ));

-- Policy: Admins see everything
CREATE POLICY admin_access ON sensor_data
    FOR ALL
    USING (current_user = 'admin');

-- Note: RLS works on hypertables
-- Applied to all chunks automatically
```

**IMPORTANT RLS Limitation (2026):**

Based on GitHub issue #7830, there is a known issue where RLS policies on hypertables are not automatically propagated to chunks. Users can bypass security by querying chunks directly. Workaround:

```sql
-- Workaround: Apply RLS to chunks via trigger
CREATE OR REPLACE FUNCTION apply_rls_to_new_chunks()
RETURNS EVENT_TRIGGER AS $$
DECLARE
    chunk RECORD;
BEGIN
    FOR chunk IN
        SELECT format('%I.%I', chunk_schema, chunk_name) AS chunk_name
        FROM timescaledb_information.chunks
        WHERE hypertable_name = 'sensor_data'
    LOOP
        EXECUTE format('ALTER TABLE %s ENABLE ROW LEVEL SECURITY', chunk.chunk_name);
        -- Re-apply policies to chunk
    END LOOP;
END;
$$ LANGUAGE plpgsql;

CREATE EVENT TRIGGER apply_rls_on_chunk_creation
    ON ddl_command_end
    WHEN TAG IN ('CREATE TABLE')
    EXECUTE FUNCTION apply_rls_to_new_chunks();
```

### User Roles and Permissions

```sql
-- Create read-only user
CREATE USER readonly_user WITH PASSWORD 'secure_password';
GRANT CONNECT ON DATABASE timescaledb TO readonly_user;
GRANT USAGE ON SCHEMA public TO readonly_user;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO readonly_user;

-- Create write user
CREATE USER writer_user WITH PASSWORD 'secure_password';
GRANT CONNECT ON DATABASE timescaledb TO writer_user;
GRANT USAGE ON SCHEMA public TO writer_user;
GRANT SELECT, INSERT ON ALL TABLES IN SCHEMA public TO writer_user;

-- Create admin user
CREATE USER admin_user WITH PASSWORD 'secure_password';
GRANT ALL PRIVILEGES ON DATABASE timescaledb TO admin_user;

-- Grant access to TimescaleDB internals (for admin)
GRANT USAGE ON SCHEMA _timescaledb_internal TO admin_user;
GRANT SELECT ON ALL TABLES IN SCHEMA _timescaledb_internal TO admin_user;
```

### Encryption

```yaml
# Encryption layers
At_Rest:
  Method: Full disk encryption (LUKS, dm-crypt)
  Filesystem: Encrypted tablespaces
  Backup: Encrypted backup files

In_Transit:
  SSL/TLS: Required for all connections
  Minimum: TLS 1.2
  Certificate: Signed by trusted CA (production)

Application:
  Column_Encryption: pgcrypto extension
  Sensitive_Data: Encrypt before storing

Example_Column_Encryption:
  CREATE EXTENSION pgcrypto;

  INSERT INTO users (email, encrypted_ssn)
  VALUES ('user@example.com',
          pgp_sym_encrypt('123-45-6789', 'encryption_key'));

  SELECT email, pgp_sym_decrypt(encrypted_ssn::bytea, 'encryption_key')
  FROM users;
```

### Network Security

```yaml
# Network isolation
Firewall:
  PostgreSQL_Port: 5432 (restrict to application servers only)
  Allow_List: Specific IP addresses/subnets
  Deny_Default: Block all other traffic

VPC:
  Private_Subnet: Database instances
  Public_Subnet: Application load balancers only
  No_Direct_Access: Database not internet-facing

pg_hba.conf:
  # Local connections
  local   all   all   scram-sha-256

  # Remote connections (SSL required)
  hostssl all   all   10.0.0.0/8   scram-sha-256
  hostssl all   all   172.16.0.0/12   scram-sha-256

  # Reject non-SSL
  hostnossl all   all   0.0.0.0/0   reject
```

### Audit Logging

```sql
-- Enable query logging
-- postgresql.conf
log_statement = 'all'                    # Log all queries (dev/audit)
log_connections = on
log_disconnections = on
log_duration = on

-- pgaudit extension for detailed auditing
CREATE EXTENSION pgaudit;

-- Configure audit logging
ALTER SYSTEM SET pgaudit.log = 'write, ddl';
SELECT pg_reload_conf();

-- Audit logs include:
-- - All DML operations (INSERT, UPDATE, DELETE)
-- - DDL operations (CREATE, ALTER, DROP)
-- - User connections/disconnections
-- - Failed login attempts
```

## 15. High Availability

### Streaming Replication (Recommended)

```yaml
# PostgreSQL streaming replication
Architecture:
  Primary: Write node (accepts all writes)
  Replicas: Read nodes (asynch replication)
  Replication: WAL streaming

Configuration:
  Primary: Sends WAL to replicas
  Replicas: Apply WAL continuously
  Lag: Typically < 1 second
```

**Primary Configuration:**

```sql
-- postgresql.conf (primary)
wal_level = replica
max_wal_senders = 3
wal_keep_size = 1GB
hot_standby = on

-- Create replication user
CREATE USER replicator WITH REPLICATION ENCRYPTED PASSWORD 'repl_password';

-- pg_hba.conf (primary)
host    replication    replicator    <replica_ip>/32    scram-sha-256
```

**Replica Configuration:**

```bash
# Stop PostgreSQL on replica
sudo systemctl stop postgresql

# Remove old data directory
sudo rm -rf /var/lib/postgresql/14/main/*

# Base backup from primary
pg_basebackup -h <primary_ip> -D /var/lib/postgresql/14/main -U replicator -P -v -R -X stream -C -S replica_slot

# Start replica
sudo systemctl start postgresql
```

```sql
-- postgresql.conf (replica)
hot_standby = on
primary_conninfo = 'host=<primary_ip> port=5432 user=replicator password=repl_password'
primary_slot_name = 'replica_slot'

-- Check replication status (on primary)
SELECT * FROM pg_stat_replication;

-- Check replication lag (on primary)
SELECT client_addr,
       application_name,
       state,
       sync_state,
       pg_wal_lsn_diff(pg_current_wal_lsn(), replay_lsn) AS lag_bytes,
       write_lag,
       flush_lag,
       replay_lag
FROM pg_stat_replication;
```

### Patroni for Automatic Failover

```yaml
# Patroni: HA solution for PostgreSQL
Features:
  - Automatic failover
  - Leader election (via etcd/consul/zookeeper)
  - Health checks
  - Automatic replica promotion

Architecture:
  Primary: Active PostgreSQL instance
  Replicas: Standby instances
  DCS: Distributed configuration store (etcd)

Installation:
  pip install patroni[etcd]

Configuration:
  Cluster: TimescaleDB cluster
  Nodes: 3 (1 primary, 2 replicas)
  DCS: etcd cluster (3 nodes)
```

**Patroni Configuration (patroni.yml):**

```yaml
scope: timescaledb-cluster
name: node1

restapi:
  listen: 0.0.0.0:8008
  connect_address: node1:8008

etcd:
  hosts: etcd1:2379,etcd2:2379,etcd3:2379

bootstrap:
  dcs:
    ttl: 30
    loop_wait: 10
    retry_timeout: 10
    maximum_lag_on_failover: 1048576

  initdb:
    - encoding: UTF8
    - data-checksums

  pg_hba:
    - host replication replicator 0.0.0.0/0 md5
    - host all all 0.0.0.0/0 md5

postgresql:
  listen: 0.0.0.0:5432
  connect_address: node1:5432
  data_dir: /var/lib/postgresql/14/main
  bin_dir: /usr/lib/postgresql/14/bin

  authentication:
    replication:
      username: replicator
      password: repl_password
    superuser:
      username: postgres
      password: postgres_password

  parameters:
    shared_buffers: 8GB
    effective_cache_size: 24GB
    wal_level: replica
    max_wal_senders: 10
    wal_keep_size: 1GB
```

### Logical Replication (Not Recommended)

**Important:** Logical replication is **not recommended** for TimescaleDB due to:
- Requires schema synchronization
- Partition root tables not supported
- Complex chunk management
- Better alternatives available (streaming replication)

### TimescaleDB-Specific HA Considerations

```yaml
Hypertables:
  - Replicate normally via streaming replication
  - Chunk structure maintained on replicas
  - Compression state preserved

Continuous_Aggregates:
  - Replicate to standby
  - Refresh policies run on primary only
  - Materialized views copied to replicas

Background_Jobs:
  - Run on primary only
  - Compression, retention policies
  - Continuous aggregate refreshes

Failover:
  - Promote replica to primary
  - Background jobs start on new primary
  - Update client connection strings
```

## 16. Backup and Restore

### Backup Strategies

```yaml
# Backup methods
Logical_Backup:
  Tool: pg_dump / pg_restore
  Format: SQL or custom format
  Pros: Version-independent, selective restore
  Cons: Slow for large databases

Physical_Backup:
  Tool: pg_basebackup
  Format: Binary file copy
  Pros: Fast, full cluster backup
  Cons: Version-specific, all-or-nothing restore

Continuous_Archiving:
  Tool: WAL archiving + base backup
  Format: WAL files + base backup
  Pros: Point-in-time recovery
  Cons: Complex setup
```

### pg_dump Backup

```bash
# Full database backup (compressed custom format)
pg_dump -Fc -Z9 -d timescaledb -U postgres > timescaledb_backup.dump

# Schema-only backup
pg_dump -s -d timescaledb > schema_only.sql

# Specific hypertable backup
pg_dump -t metrics -d timescaledb > metrics_backup.sql

# Important: pg_dump includes:
# - Hypertable definitions
# - All chunks
# - Indexes
# - Continuous aggregates
# - Policies (compression, retention, etc.)

# Restore
pg_restore -d timescaledb_new timescaledb_backup.dump

# Restore with parallel jobs (faster)
pg_restore -j 4 -d timescaledb_new timescaledb_backup.dump

# CRITICAL: Do NOT use -j with TimescaleDB (catalog issues)
# Use single-threaded restore for TimescaleDB databases
```

### Physical Backup (pg_basebackup)

```bash
# Stop writes (optional but recommended)
psql -c "SELECT pg_start_backup('backup_label', false, false);"

# Base backup
pg_basebackup -h localhost -D /backup/postgres -U postgres -P -v -X fetch

# Resume writes
psql -c "SELECT pg_stop_backup();"

# Restore:
# 1. Stop PostgreSQL
sudo systemctl stop postgresql

# 2. Replace data directory
sudo rm -rf /var/lib/postgresql/14/main
sudo cp -a /backup/postgres /var/lib/postgresql/14/main
sudo chown -R postgres:postgres /var/lib/postgresql/14/main

# 3. Start PostgreSQL
sudo systemctl start postgresql
```

### Continuous Archiving (WAL)

```sql
-- postgresql.conf
wal_level = replica
archive_mode = on
archive_command = 'cp %p /backup/wal_archive/%f'
archive_timeout = 300  # Force archive every 5 minutes

-- Take base backup
SELECT pg_start_backup('backup_label', false, false);
-- Copy data directory
SELECT pg_stop_backup();

-- Restore procedure (recovery.conf / postgresql.auto.conf):
restore_command = 'cp /backup/wal_archive/%f %p'
recovery_target_time = '2024-01-01 10:00:00'
```

### Backup Best Practices

```yaml
Scheduling:
  Frequency:
    Production: Daily minimum (full backup)
    Critical: Hourly incremental + WAL archiving

  Timing:
    - During low-traffic periods
    - Avoid compression job schedules
    - Coordinate with maintenance windows

Retention:
  Daily: 7-30 days
  Weekly: 3-6 months
  Monthly: 1-2 years

Storage:
  Location: Off-server (S3, NAS, tape)
  Encryption: Encrypted backups required
  Verification: Test restores monthly

Automation:
  Tools: pg_cron, cron, Kubernetes CronJobs
  Monitoring: Alert on backup failures
  Validation: Check backup integrity
```

### Backup Automation Script

```bash
#!/bin/bash
# TimescaleDB backup script

BACKUP_DIR="/backups/timescaledb"
DATE=$(date +%Y%m%d-%H%M%S)
DATABASE="timescaledb"
RETENTION_DAYS=30

# Create backup directory
mkdir -p "${BACKUP_DIR}"

# Perform backup
pg_dump -Fc -Z9 -d "${DATABASE}" > "${BACKUP_DIR}/backup_${DATE}.dump"

# Upload to S3
aws s3 cp "${BACKUP_DIR}/backup_${DATE}.dump" \
    s3://my-backups/timescaledb/backup_${DATE}.dump

# Clean old local backups
find "${BACKUP_DIR}" -name "backup_*.dump" -mtime +${RETENTION_DAYS} -delete

# Verify backup
pg_restore --list "${BACKUP_DIR}/backup_${DATE}.dump" > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "Backup successful: backup_${DATE}.dump"
else
    echo "Backup verification failed!" >&2
    # Send alert
    exit 1
fi
```

### Selective Restore

```bash
# Restore specific table
pg_restore -d timescaledb_new -t metrics timescaledb_backup.dump

# Restore excluding data (schema only)
pg_restore -s -d timescaledb_new timescaledb_backup.dump

# List backup contents
pg_restore --list timescaledb_backup.dump

# Restore specific objects
pg_restore -d timescaledb_new -L restore_list.txt timescaledb_backup.dump
```

## 17. Monitoring and Operations

### pg_stat_statements

```sql
-- Enable extension
CREATE EXTENSION pg_stat_statements;

-- postgresql.conf
shared_preload_libraries = 'timescaledb,pg_stat_statements'
pg_stat_statements.track = all

-- Restart PostgreSQL
sudo systemctl restart postgresql

-- Find slow queries
SELECT calls,
       total_exec_time / 1000 AS total_time_seconds,
       mean_exec_time / 1000 AS mean_time_seconds,
       query
FROM pg_stat_statements
ORDER BY mean_exec_time DESC
LIMIT 20;

-- Find queries by total time
SELECT calls,
       total_exec_time / 1000 AS total_time_seconds,
       100.0 * total_exec_time / SUM(total_exec_time) OVER () AS percent_total,
       query
FROM pg_stat_statements
ORDER BY total_exec_time DESC
LIMIT 20;

-- Reset statistics
SELECT pg_stat_statements_reset();
```

### TimescaleDB Metrics

```sql
-- Hypertable statistics
SELECT * FROM timescaledb_information.hypertables;

-- Chunk information
SELECT hypertable_name,
       COUNT(*) AS num_chunks,
       pg_size_pretty(SUM(total_bytes)) AS total_size
FROM timescaledb_information.chunks
GROUP BY hypertable_name;

-- Compression statistics
SELECT hypertable_schema || '.' || hypertable_name AS hypertable,
       pg_size_pretty(before_compression_total_bytes) AS before,
       pg_size_pretty(after_compression_total_bytes) AS after,
       100 - (100 * after_compression_total_bytes / before_compression_total_bytes) AS compression_ratio
FROM hypertable_compression_stats('metrics');

-- Job statistics
SELECT job_id,
       application_name,
       schedule_interval,
       last_run_started_at,
       last_successful_finish,
       total_runs,
       total_successes,
       total_failures
FROM timescaledb_information.job_stats
ORDER BY last_run_started_at DESC;
```

### Database Size Monitoring

```sql
-- Database size
SELECT pg_database.datname,
       pg_size_pretty(pg_database_size(pg_database.datname)) AS size
FROM pg_database
ORDER BY pg_database_size(pg_database.datname) DESC;

-- Table sizes (including hypertables)
SELECT schemaname || '.' || tablename AS table,
       pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS total_size,
       pg_size_pretty(pg_relation_size(schemaname||'.'||tablename)) AS table_size,
       pg_size_pretty(pg_indexes_size(schemaname||'.'||tablename)) AS index_size
FROM pg_tables
WHERE schemaname NOT IN ('pg_catalog', 'information_schema', '_timescaledb_internal')
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;

-- Chunk sizes
SELECT chunk_schema || '.' || chunk_name AS chunk,
       pg_size_pretty(total_bytes) AS size,
       compression_status,
       range_start,
       range_end
FROM timescaledb_information.chunks
WHERE hypertable_name = 'metrics'
ORDER BY range_start DESC;
```

### Connection Monitoring

```sql
-- Active connections
SELECT pid, usename, application_name, client_addr, state,
       NOW() - query_start AS duration,
       query
FROM pg_stat_activity
WHERE state != 'idle'
ORDER BY duration DESC;

-- Connection count by database
SELECT datname, count(*) AS connections
FROM pg_stat_activity
GROUP BY datname
ORDER BY connections DESC;

-- Long-running queries
SELECT pid, usename, NOW() - query_start AS duration, query
FROM pg_stat_activity
WHERE state = 'active'
  AND NOW() - query_start > INTERVAL '5 minutes'
ORDER BY duration DESC;

-- Kill long-running query
SELECT pg_terminate_backend(pid)
FROM pg_stat_activity
WHERE pid = <pid>;
```

### Performance Metrics

```sql
-- Cache hit ratio (target: >99%)
SELECT
    sum(heap_blks_read) AS heap_read,
    sum(heap_blks_hit) AS heap_hit,
    round(sum(heap_blks_hit) / (sum(heap_blks_hit) + sum(heap_blks_read)), 4) AS cache_hit_ratio
FROM pg_statio_user_tables;

-- Index usage
SELECT schemaname, tablename, indexname,
       idx_scan,
       pg_size_pretty(pg_relation_size(indexrelid)) AS index_size
FROM pg_stat_user_indexes
WHERE schemaname NOT LIKE 'pg_%'
ORDER BY idx_scan ASC;

-- Table I/O statistics
SELECT schemaname, tablename,
       heap_blks_read,
       heap_blks_hit,
       idx_blks_read,
       idx_blks_hit
FROM pg_statio_user_tables
WHERE schemaname NOT IN ('pg_catalog', 'information_schema')
ORDER BY heap_blks_read DESC;
```

### Grafana Dashboard Setup

```yaml
# Prometheus + Grafana monitoring
Components:
  postgres_exporter: Exports PostgreSQL metrics
  Prometheus: Collects and stores metrics
  Grafana: Visualizes metrics

Installation:
  # Install postgres_exporter
  docker run -d \
    --name postgres_exporter \
    -p 9187:9187 \
    -e DATA_SOURCE_NAME="postgresql://user:password@localhost:5432/timescaledb?sslmode=disable" \
    prometheuscommunity/postgres-exporter

  # Configure Prometheus (prometheus.yml)
  scrape_configs:
    - job_name: 'postgresql'
      static_configs:
        - targets: ['postgres_exporter:9187']

  # Import Grafana dashboard
  Dashboard_ID: 9628 (PostgreSQL Database)
  TimescaleDB_Metrics: Custom queries
```

### Alert Configuration

```yaml
# Prometheus alerts (alerts.yml)
groups:
  - name: timescaledb
    rules:
      - alert: HighDatabaseSize
        expr: pg_database_size_bytes > 100e9
        for: 5m
        annotations:
          summary: "Database size exceeds 100GB"

      - alert: SlowQueries
        expr: rate(pg_stat_statements_mean_exec_time[5m]) > 1000
        for: 5m
        annotations:
          summary: "Slow queries detected (>1s average)"

      - alert: LowCacheHitRatio
        expr: pg_stat_database_blks_hit / (pg_stat_database_blks_hit + pg_stat_database_blks_read) < 0.99
        for: 10m
        annotations:
          summary: "Cache hit ratio below 99%"

      - alert: HighConnectionCount
        expr: pg_stat_activity_count > 100
        for: 5m
        annotations:
          summary: "More than 100 active connections"
```

## 18. Docker Deployment

### Official Docker Image

```bash
# Pull official TimescaleDB image
docker pull timescale/timescaledb:latest-pg17

# Latest version tags (2026):
# - timescale/timescaledb:latest-pg17 (PostgreSQL 17)
# - timescale/timescaledb:latest-pg16 (PostgreSQL 16)
# - timescale/timescaledb:2.14.2-pg16 (specific version)

# Note: PostgreSQL 17.1, 16.5, 15.9, 14.14 not recommended
# Use 17.2+, 16.6+, 15.10+, 14.15+ instead
```

### Basic Docker Deployment

```bash
# Run TimescaleDB container
docker run -d \
  --name timescaledb \
  -p 5432:5432 \
  -e POSTGRES_PASSWORD=password \
  -e POSTGRES_DB=timescaledb \
  -v timescaledb-data:/var/lib/postgresql/data \
  timescale/timescaledb:latest-pg16

# Connect to database
docker exec -it timescaledb psql -U postgres -d timescaledb

# Create hypertable
CREATE TABLE metrics (
    time TIMESTAMPTZ NOT NULL,
    device_id INTEGER,
    value DOUBLE PRECISION
);

SELECT create_hypertable('metrics', 'time');
```

### Docker Compose Configuration

```yaml
# docker-compose.yml
version: '3.8'

services:
  timescaledb:
    image: timescale/timescaledb:latest-pg16
    container_name: timescaledb
    restart: unless-stopped
    ports:
      - "5432:5432"
    environment:
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD_FILE: /run/secrets/postgres_password
      POSTGRES_DB: timescaledb
      # TimescaleDB tuning
      TS_TUNE_MEMORY: 8GB
      TS_TUNE_NUM_CPUS: 4
      TS_TUNE_MAX_BG_WORKERS: 8
    volumes:
      - timescaledb-data:/var/lib/postgresql/data
      - ./init:/docker-entrypoint-initdb.d
    secrets:
      - postgres_password
    networks:
      - timescale-network
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 10s
      timeout: 5s
      retries: 5
    deploy:
      resources:
        limits:
          cpus: '4'
          memory: 8G
        reservations:
          cpus: '2'
          memory: 4G

volumes:
  timescaledb-data:
    driver: local

secrets:
  postgres_password:
    file: ./secrets/postgres_password.txt

networks:
  timescale-network:
    driver: bridge
```

### Production Docker Configuration

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  timescaledb:
    image: timescale/timescaledb:2.14.2-pg16
    container_name: timescaledb-prod
    restart: always
    ports:
      - "127.0.0.1:5432:5432"  # Bind to localhost only
    environment:
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD_FILE: /run/secrets/postgres_password
      POSTGRES_DB: timescaledb
      POSTGRES_INITDB_ARGS: "-E UTF8 --data-checksums"
      # Tuning parameters
      TS_TUNE_MEMORY: 16GB
      TS_TUNE_NUM_CPUS: 8
      TS_TUNE_MAX_BG_WORKERS: 16
    volumes:
      - /data/timescaledb:/var/lib/postgresql/data
      - ./postgresql.conf:/etc/postgresql/postgresql.conf:ro
      - ./init-scripts:/docker-entrypoint-initdb.d:ro
    secrets:
      - postgres_password
    user: "70:70"  # postgres user in container
    security_opt:
      - no-new-privileges:true
    read_only: false  # PostgreSQL needs write access
    tmpfs:
      - /tmp
      - /run
    networks:
      - timescale-network
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 10s
      timeout: 5s
      retries: 5
      start_period: 30s
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
    deploy:
      resources:
        limits:
          cpus: '8'
          memory: 16G
        reservations:
          cpus: '4'
          memory: 8G

volumes:
  timescaledb-data:
    driver: local
    driver_opts:
      type: none
      o: bind
      device: /data/timescaledb

secrets:
  postgres_password:
    file: /secure/postgres_password.txt

networks:
  timescale-network:
    driver: bridge
    internal: true
```

### Initialization Scripts

```sql
-- init/01-create-extensions.sql
CREATE EXTENSION IF NOT EXISTS timescaledb;
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;
CREATE EXTENSION IF NOT EXISTS postgis;

-- init/02-create-hypertables.sql
CREATE TABLE IF NOT EXISTS metrics (
    time TIMESTAMPTZ NOT NULL,
    device_id INTEGER NOT NULL,
    value DOUBLE PRECISION
);

SELECT create_hypertable('metrics', 'time', if_not_exists => TRUE);

-- Create indexes
CREATE INDEX IF NOT EXISTS metrics_device_time_idx
ON metrics (device_id, time DESC);

-- Enable compression
ALTER TABLE metrics SET (
    timescaledb.compress,
    timescaledb.compress_orderby = 'time DESC',
    timescaledb.compress_segmentby = 'device_id'
);

SELECT add_compression_policy('metrics', INTERVAL '7 days');
SELECT add_retention_policy('metrics', INTERVAL '90 days');
```

### Docker Backup Strategy

```bash
#!/bin/bash
# Backup TimescaleDB Docker container

CONTAINER="timescaledb"
BACKUP_DIR="/backups"
DATE=$(date +%Y%m%d-%H%M%S)

# Backup using pg_dump
docker exec ${CONTAINER} pg_dump -U postgres -Fc timescaledb > \
    ${BACKUP_DIR}/timescaledb_${DATE}.dump

# Upload to S3
aws s3 cp ${BACKUP_DIR}/timescaledb_${DATE}.dump \
    s3://my-backups/timescaledb/

# Clean old backups
find ${BACKUP_DIR} -name "timescaledb_*.dump" -mtime +7 -delete
```

### Docker Performance Tips

```yaml
Storage:
  Volume_Driver: local (direct mount for performance)
  Avoid: Docker overlay2 for data volumes
  Recommended: Bind mount to SSD path

Memory:
  Set_Limits: Match available RAM
  Leave_Headroom: 20-30% for OS

Networking:
  Use_Host_Network: For maximum performance (dev only)
  Bridge_Network: For production (isolated)

Configuration:
  Tune_Parameters: Use TS_TUNE_* environment variables
  Custom_Config: Mount postgresql.conf
```

## 19. Kubernetes Deployment

### Helm Chart Deployment

```bash
# Add TimescaleDB Helm repository
helm repo add timescale https://charts.timescale.com
helm repo update

# Install TimescaleDB
helm install timescaledb timescale/timescaledb-single \
  --namespace timescaledb \
  --create-namespace \
  --set persistence.enabled=true \
  --set persistence.size=100Gi \
  --set resources.requests.memory=8Gi \
  --set resources.requests.cpu=4 \
  --set resources.limits.memory=16Gi \
  --set resources.limits.cpu=8

# Note: Timescale recommends using PostgreSQL operators (2026)
# Consider using CloudNativePG or other operators instead
```

### StatefulSet Deployment

```yaml
# timescaledb-statefulset.yaml
apiVersion: v1
kind: Service
metadata:
  name: timescaledb
  namespace: timescaledb
spec:
  ports:
  - port: 5432
    name: postgresql
  clusterIP: None
  selector:
    app: timescaledb
---
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: timescaledb
  namespace: timescaledb
spec:
  serviceName: timescaledb
  replicas: 1
  selector:
    matchLabels:
      app: timescaledb
  template:
    metadata:
      labels:
        app: timescaledb
    spec:
      containers:
      - name: timescaledb
        image: timescale/timescaledb:latest-pg16
        ports:
        - containerPort: 5432
          name: postgresql
        env:
        - name: POSTGRES_USER
          value: "postgres"
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: timescaledb-credentials
              key: password
        - name: POSTGRES_DB
          value: "timescaledb"
        - name: TS_TUNE_MEMORY
          value: "8GB"
        - name: TS_TUNE_NUM_CPUS
          value: "4"
        - name: TS_TUNE_MAX_BG_WORKERS
          value: "8"
        resources:
          requests:
            memory: "8Gi"
            cpu: "4"
          limits:
            memory: "16Gi"
            cpu: "8"
        volumeMounts:
        - name: timescaledb-data
          mountPath: /var/lib/postgresql/data
        - name: init-scripts
          mountPath: /docker-entrypoint-initdb.d
        livenessProbe:
          exec:
            command:
            - pg_isready
            - -U
            - postgres
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
        readinessProbe:
          exec:
            command:
            - pg_isready
            - -U
            - postgres
          initialDelaySeconds: 5
          periodSeconds: 5
          timeoutSeconds: 3
      securityContext:
        fsGroup: 70
        runAsUser: 70
        runAsNonRoot: true
      volumes:
      - name: init-scripts
        configMap:
          name: timescaledb-init
  volumeClaimTemplates:
  - metadata:
      name: timescaledb-data
    spec:
      accessModes: [ "ReadWriteOnce" ]
      storageClassName: fast-ssd
      resources:
        requests:
          storage: 100Gi
```

### Secrets Management

```yaml
# timescaledb-secrets.yaml
apiVersion: v1
kind: Secret
metadata:
  name: timescaledb-credentials
  namespace: timescaledb
type: Opaque
stringData:
  password: "change-me-in-production"

---
# Using External Secrets Operator (recommended)
apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata:
  name: timescaledb-credentials
  namespace: timescaledb
spec:
  refreshInterval: 1h
  secretStoreRef:
    name: vault-backend
    kind: SecretStore
  target:
    name: timescaledb-credentials
    creationPolicy: Owner
  data:
  - secretKey: password
    remoteRef:
      key: secret/timescaledb
      property: password
```

### ConfigMap for Initialization

```yaml
# timescaledb-configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: timescaledb-init
  namespace: timescaledb
data:
  01-extensions.sql: |
    CREATE EXTENSION IF NOT EXISTS timescaledb;
    CREATE EXTENSION IF NOT EXISTS pg_stat_statements;

  02-hypertables.sql: |
    CREATE TABLE IF NOT EXISTS metrics (
        time TIMESTAMPTZ NOT NULL,
        device_id INTEGER NOT NULL,
        value DOUBLE PRECISION
    );

    SELECT create_hypertable('metrics', 'time', if_not_exists => TRUE);

    ALTER TABLE metrics SET (
        timescaledb.compress,
        timescaledb.compress_orderby = 'time DESC',
        timescaledb.compress_segmentby = 'device_id'
    );

    SELECT add_compression_policy('metrics', INTERVAL '7 days');
```

### Service and Ingress

```yaml
# timescaledb-service.yaml
apiVersion: v1
kind: Service
metadata:
  name: timescaledb-external
  namespace: timescaledb
spec:
  type: LoadBalancer
  ports:
  - port: 5432
    targetPort: 5432
    protocol: TCP
  selector:
    app: timescaledb
```

### Backup CronJob

```yaml
# timescaledb-backup-cronjob.yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: timescaledb-backup
  namespace: timescaledb
spec:
  schedule: "0 2 * * *"  # Daily at 2 AM
  successfulJobsHistoryLimit: 3
  failedJobsHistoryLimit: 1
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: backup
            image: timescale/timescaledb:latest-pg16
            command:
            - /bin/bash
            - -c
            - |
              BACKUP_FILE="/backups/timescaledb_$(date +%Y%m%d-%H%M%S).dump"
              pg_dump -h timescaledb -U postgres -Fc -Z9 timescaledb > "$BACKUP_FILE"
              # Upload to S3 (requires aws-cli in image)
              # aws s3 cp "$BACKUP_FILE" s3://my-backups/timescaledb/
            env:
            - name: PGPASSWORD
              valueFrom:
                secretKeyRef:
                  name: timescaledb-credentials
                  key: password
            volumeMounts:
            - name: backup-storage
              mountPath: /backups
          restartPolicy: OnFailure
          volumes:
          - name: backup-storage
            persistentVolumeClaim:
              claimName: timescaledb-backup-pvc
```

### CloudNativePG Operator (Recommended 2026)

```yaml
# Using CloudNativePG with TimescaleDB
# Install operator:
# kubectl apply -f https://raw.githubusercontent.com/cloudnative-pg/cloudnative-pg/main/releases/cnpg-latest.yaml

apiVersion: postgresql.cnpg.io/v1
kind: Cluster
metadata:
  name: timescaledb-cluster
  namespace: timescaledb
spec:
  instances: 3
  imageName: timescale/timescaledb:latest-pg16

  postgresql:
    shared_preload_libraries:
      - timescaledb
    parameters:
      shared_buffers: "8GB"
      effective_cache_size: "24GB"
      work_mem: "64MB"
      maintenance_work_mem: "2GB"
      timescaledb.max_background_workers: "8"

  bootstrap:
    initdb:
      database: timescaledb
      owner: app
      postInitSQL:
        - CREATE EXTENSION IF NOT EXISTS timescaledb;
        - CREATE EXTENSION IF NOT EXISTS pg_stat_statements;

  storage:
    size: 100Gi
    storageClass: fast-ssd

  resources:
    requests:
      memory: "8Gi"
      cpu: "4"
    limits:
      memory: "16Gi"
      cpu: "8"

  backup:
    barmanObjectStore:
      destinationPath: s3://my-backups/timescaledb/
      s3Credentials:
        accessKeyId:
          name: s3-credentials
          key: ACCESS_KEY_ID
        secretAccessKey:
          name: s3-credentials
          key: SECRET_ACCESS_KEY
    retentionPolicy: "30d"
```

## 20. Migration Strategies

### PostgreSQL to TimescaleDB Migration

```sql
-- Step 1: Install TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Step 2: Existing table with time-series data
-- (Example: Standard PostgreSQL table)
CREATE TABLE sensor_readings (
    reading_time TIMESTAMPTZ NOT NULL,
    sensor_id INTEGER NOT NULL,
    temperature DOUBLE PRECISION,
    humidity DOUBLE PRECISION
);

-- Data already exists (millions of rows)

-- Step 3: Convert to hypertable (with data migration)
SELECT create_hypertable(
    'sensor_readings',
    'reading_time',
    migrate_data => TRUE,           -- Migrate existing data
    chunk_time_interval => INTERVAL '1 day'
);

-- Step 4: Create indexes (AFTER hypertable conversion)
CREATE INDEX ON sensor_readings (sensor_id, reading_time DESC);

-- Step 5: Enable compression
ALTER TABLE sensor_readings SET (
    timescaledb.compress,
    timescaledb.compress_orderby = 'reading_time DESC',
    timescaledb.compress_segmentby = 'sensor_id'
);

SELECT add_compression_policy('sensor_readings', INTERVAL '7 days');
```

### Zero-Downtime Migration

```yaml
# Strategy: Dual-write approach
Phase_1_Preparation:
  - Deploy TimescaleDB alongside PostgreSQL
  - Set up replication or dual-write logic
  - Test application with TimescaleDB

Phase_2_Backfill:
  - Export historical data from PostgreSQL
  - Import into TimescaleDB hypertables
  - Verify data consistency

Phase_3_Validation:
  - Run read queries against both databases
  - Compare results and performance
  - Monitor for discrepancies

Phase_4_Cutover:
  - Switch reads to TimescaleDB
  - Stop writes to PostgreSQL
  - Monitor performance and errors

Phase_5_Cleanup:
  - Remove dual-write logic
  - Decommission PostgreSQL instance
  - Update documentation
```

### Version Upgrade Strategy

```yaml
# TimescaleDB version upgrades
Pre_Upgrade:
  - Backup database (full backup)
  - Review release notes for breaking changes
  - Test upgrade in non-production environment
  - Check PostgreSQL version compatibility

Important_2026_Notes:
  PostgreSQL_Compatibility:
    Not_Recommended: 17.1, 16.5, 15.9, 14.14, 13.17, 12.21
    Use_Instead: 17.2+, 16.6+, 15.10+, 14.15+, 13.18+, 12.22+

  PostgreSQL_15_Support:
    Status: Deprecated (June 2026 removal)
    Action: Plan migration to PostgreSQL 16+

Upgrade_Process:
  1_Backup: pg_dump backup before upgrade
  2_Update_Extension: ALTER EXTENSION timescaledb UPDATE;
  3_Verify: SELECT * FROM timescaledb_information.hypertables;
  4_Test: Run application test suite
  5_Monitor: Watch for performance regressions
```

### Migration from InfluxDB to TimescaleDB

```python
# Example migration script (Python)
from influxdb_client import InfluxDBClient
import psycopg2
from datetime import datetime

# Connect to InfluxDB
influx_client = InfluxDBClient(url="http://localhost:8086",
                                token="my-token",
                                org="my-org")
query_api = influx_client.query_api()

# Connect to TimescaleDB
pg_conn = psycopg2.connect(
    host="localhost",
    database="timescaledb",
    user="postgres",
    password="password"
)
pg_cursor = pg_conn.cursor()

# Create hypertable in TimescaleDB
pg_cursor.execute("""
    CREATE TABLE IF NOT EXISTS metrics (
        time TIMESTAMPTZ NOT NULL,
        measurement TEXT NOT NULL,
        tag_device TEXT,
        tag_location TEXT,
        field_value DOUBLE PRECISION
    );
    SELECT create_hypertable('metrics', 'time', if_not_exists => TRUE);
    CREATE INDEX ON metrics (tag_device, time DESC);
""")
pg_conn.commit()

# Query data from InfluxDB
query = '''
from(bucket: "my-bucket")
  |> range(start: -30d)
  |> filter(fn: (r) => r._measurement == "cpu_usage")
'''
tables = query_api.query(query)

# Insert into TimescaleDB (batched)
batch = []
for table in tables:
    for record in table.records:
        batch.append((
            record.get_time(),
            record.get_measurement(),
            record.values.get("host"),
            record.values.get("location"),
            record.get_value()
        ))

        if len(batch) >= 10000:
            pg_cursor.executemany(
                "INSERT INTO metrics VALUES (%s, %s, %s, %s, %s)",
                batch
            )
            pg_conn.commit()
            batch = []

# Insert remaining
if batch:
    pg_cursor.executemany(
        "INSERT INTO metrics VALUES (%s, %s, %s, %s, %s)",
        batch
    )
    pg_conn.commit()

pg_cursor.close()
pg_conn.close()
influx_client.close()
```

## 21. Multi-Node and Distributed Hypertables

### Multi-Node Architecture (TimescaleDB 2.x)

```yaml
# Note: Multi-node capabilities introduced in TimescaleDB 2.0
Components:
  Access_Node:
    Role: Query coordinator and metadata storage
    Handles: Client connections, query distribution

  Data_Nodes:
    Role: Store chunks of distributed hypertables
    Handles: Data storage, local query execution

Architecture:
  Client -> Access Node -> Data Nodes (1-N)
  Access Node aggregates results from Data Nodes
```

### Creating Distributed Hypertables

```sql
-- On Access Node:
-- Step 1: Add data nodes
SELECT add_data_node('node1', host => 'datanode1.example.com', port => 5432);
SELECT add_data_node('node2', host => 'datanode2.example.com', port => 5432);
SELECT add_data_node('node3', host => 'datanode3.example.com', port => 5432);

-- Step 2: Create distributed hypertable
CREATE TABLE distributed_metrics (
    time TIMESTAMPTZ NOT NULL,
    device_id INTEGER NOT NULL,
    location TEXT NOT NULL,
    value DOUBLE PRECISION
);

SELECT create_distributed_hypertable(
    'distributed_metrics',
    'time',
    'device_id',                    -- Space partitioning column
    number_partitions => 3,         -- Match number of data nodes
    chunk_time_interval => INTERVAL '1 day'
);

-- Data is now automatically distributed across data nodes
```

### Space Partitioning Strategy

```yaml
# Partitioning recommendations
Number_of_Partitions:
  General_Rule: Match number of data nodes
  Example_3_Nodes: number_partitions => 3

Partitioning_Column:
  Choose: High cardinality, evenly distributed
  Examples: device_id, customer_id, sensor_id
  Avoid: Low cardinality (location with few values)

Benefits:
  - Load balancing across nodes
  - Parallel query execution
  - Horizontal scalability
  - Data locality (related data on same node)
```

### Data Node Management

```sql
-- View data nodes
SELECT * FROM timescaledb_information.data_nodes;

-- View chunk distribution
SELECT hypertable_name,
       node_name,
       COUNT(*) AS num_chunks
FROM timescaledb_information.chunks
GROUP BY hypertable_name, node_name
ORDER BY hypertable_name, node_name;

-- Attach/detach data nodes
SELECT attach_data_node('node4', hypertable => 'distributed_metrics');
SELECT detach_data_node('node4', hypertable => 'distributed_metrics');

-- Drop data node (removes node and all its data)
SELECT delete_data_node('node4');
```

### Distributed Queries

```sql
-- Queries on distributed hypertables work transparently
SELECT time_bucket('1 hour', time) AS bucket,
       location,
       AVG(value) AS avg_value
FROM distributed_metrics
WHERE time > NOW() - INTERVAL '24 hours'
GROUP BY bucket, location
ORDER BY bucket DESC;

-- Access node:
-- 1. Sends query to all data nodes
-- 2. Data nodes execute locally on their chunks
-- 3. Access node aggregates results
-- 4. Returns final result to client
```

### Limitations and Considerations

```yaml
# Multi-node limitations (TimescaleDB 2.x)
Current_Limitations:
  - Continuous aggregates: Access node only (not distributed)
  - Compression policies: Must compress on each node separately
  - Joins: Limited support for distributed joins
  - Background jobs: Run on access node only

Workarounds:
  - Use access node for continuous aggregates
  - Coordinate compression across nodes
  - Perform joins on access node

Best_Practices:
  - Use space partitioning for even distribution
  - Monitor data node health
  - Ensure consistent schemas across nodes
  - Plan for network latency between nodes
```

### Scaling Out (Adding Nodes)

```sql
-- Step 1: Deploy new data node (same PostgreSQL + TimescaleDB version)

-- Step 2: Add data node to cluster
SELECT add_data_node('node4', host => 'datanode4.example.com');

-- Step 3: Attach to hypertable
SELECT attach_data_node('node4', hypertable => 'distributed_metrics');

-- Step 4: (Optional) Rebalance data
-- New chunks will use new node automatically
-- Old chunks stay on original nodes unless rebalanced

-- TimescaleDB automatically adjusts space partitions
-- for new chunks to include the new node
```

### Multi-Node Best Practices

```yaml
Network:
  Low_Latency: Required between access and data nodes
  Bandwidth: High bandwidth for data transfer
  Location: Same data center/region recommended

Data_Distribution:
  Even_Distribution: Choose good partitioning column
  Monitor_Skew: Watch for unbalanced data nodes

Fault_Tolerance:
  Replication: Use PostgreSQL streaming replication per node
  Backup: Backup each data node independently
  Monitoring: Monitor all nodes (access + data)

Performance:
  Parallel_Queries: Leverage multi-node parallelism
  Data_Locality: Partition for co-located data
  Network_Overhead: Minimize data transfer
```

---

## Additional Resources

### Official Documentation

- [TimescaleDB Documentation](https://docs.timescale.com/)
- [TimescaleDB GitHub Repository](https://github.com/timescale/timescaledb)
- [Timescale Blog](https://www.timescale.com/blog/)

### Related Guides

- See `postgresql.md` for PostgreSQL fundamentals
- See `docker-compose.md` for Docker orchestration patterns
- See `kubernetes.md` for Kubernetes deployment strategies
- See `sql.md` for SQL best practices

---

**Last Updated:** 2026-02-06
**TimescaleDB Version:** 2.x (2.14+)
**PostgreSQL Versions:** 12-17 (avoid 17.1, 16.5, 15.9, 14.14)
**Target Audience:** Backend Engineers, DBAs, DevOps Engineers, SREs

---

## Sources

- [Timescale Documentation: About hypertables](https://docs.timescale.com/use-timescale/latest/hypertables/about-hypertables/)
- [TimescaleDB Compression: 90% Reduction in Production](https://dev.to/polliog/timescaledb-compression-from-150gb-to-15gb-90-reduction-real-production-data-bnj)
- [How to Use TimescaleDB Continuous Aggregates (2026)](https://oneuptime.com/blog/post/2026-01-27-timescaledb-continuous-aggregates/view)
- [Timescale Tips: Testing Your Chunk Size](https://www.tigerdata.com/blog/timescale-cloud-tips-testing-your-chunk-size)
- [13 Tips to Improve PostgreSQL Insert Performance](https://www.tigerdata.com/blog/13-tips-to-improve-postgresql-insert-performance)
- [Boost Postgres Performance with Chunk-Skipping Indexes](https://www.tigerdata.com/blog/boost-postgres-performance-by-7x-with-chunk-skipping-indexes)
- [TimescaleDB Tuning Tool](https://github.com/timescale/timescaledb-tune)
- [TimescaleDB Security Documentation](https://docs.timescale.com/use-timescale/latest/security/)
- [PostgreSQL Database Replication Guide](https://www.tigerdata.com/learn/postgresql-database-replication-guide)
- [TimescaleDB Backup and Restore](https://docs.timescale.com/self-hosted/latest/backup-and-restore/)
- [Best Practices for Time-Series Data Modeling](https://www.timescale.com/blog/best-practices-for-time-series-data-modeling-narrow-medium-or-wide-table-layout-2/)
- [TimescaleDB Helm Charts](https://github.com/timescale/helm-charts)

---

**End of TimescaleDB Development Guidelines**
