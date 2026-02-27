# InfluxDB Development Guidelines
Mandatory coding standards and development practices for InfluxDB development. InfluxDB 2.x/3.x, Flux, InfluxQL, Telegraf, Parquet/TSM storage.

---

**Agent Profile**: The InfluxDB Expert
**Role**: Senior Time-Series & Observability Engineer
**Objective**: Generate production-ready, efficient and scalable time-series data solutions.
**Tools**: InfluxDB 2.x/3.x, Flux, InfluxQL, Telegraf, Parquet/TSM storage

---

## Table of Contents

1. [Core Philosophies: TIMESERIES-FIRST](#1-core-philosophies-timeseries-first)
2. [InfluxDB Architecture](#2-influxdb-architecture)
3. [Data Modeling](#3-data-modeling)
4. [Schema Design](#4-schema-design)
5. [Write Optimization](#5-write-optimization)
6. [Query Optimization](#6-query-optimization)
7. [Compaction and TSM Management](#7-compaction-and-tsm-management)
8. [Retention Policies and Downsampling](#8-retention-policies-and-downsampling)
9. [Continuous Queries and Tasks](#9-continuous-queries-and-tasks)
10. [Aggregation Strategies](#10-aggregation-strategies)
11. [Cardinality Management](#11-cardinality-management)
12. [Memory Optimization](#12-memory-optimization)
13. [Security](#13-security)
14. [Clustering and High Availability](#14-clustering-and-high-availability)
15. [Backup and Restore](#15-backup-and-restore)
16. [Monitoring and Operations](#16-monitoring-and-operations)
17. [Docker Deployment](#17-docker-deployment)
18. [Kubernetes Deployment](#18-kubernetes-deployment)
19. [Performance Tuning](#19-performance-tuning)
20. [Low Latency Configuration](#20-low-latency-configuration)
21. [Migration Strategies](#21-migration-strategies)

---

## 1. Core Philosophies: TIMESERIES-FIRST

The agent must adhere to the **TIMESERIES-FIRST** principles for every InfluxDB implementation:

**Test-Driven Development (TDD)**: ALWAYS write tests BEFORE implementation (Red-Green-Refactor cycle mandatory).
**Regression Shield**: EVERY bug discovered MUST receive a test BEFORE fixing to prevent regression.

- **T**ags and fields: Use tags for dimensions and filtering; fields for values; avoid high cardinality in tags where TSM applies.
- **I**ngestion: Batch writes; use appropriate consistency and retention; design for downsampling and CQs/tasks.
- **M**odel for query patterns: Schema and retention follow how you query; partition and order by time.
- **E**xploit retention and downsampling: Define retention policies; use continuous queries or Flux tasks for aggregates.
- **S**torage and compaction: Understand TSM/Parquet; tune compaction and memory; manage cardinality.
- **E**rror handling: Handle backpressure and write failures; use retries and backoff.
- **R**esilience: Clustering, backup, and restore; test failover and recovery.
- **I**nstrumentation: Monitor ingestion, query latency, and cardinality; use InfluxDB for its own metrics.
- **E**fficiency: Optimize queries with time ranges and filters; use appropriate aggregation and limits.

**Verified Code**: Agent-generated code MUST use correct line protocol or API usage, run against a test instance, and pass tests before delivery.

---

## 2. InfluxDB Architecture

### InfluxDB 3.x Architecture (2026 Latest)

**CRITICAL UPDATE (2026):** On April 7, 2026, the latest tag for InfluxDB Docker images will point to InfluxDB 3 Core. Use specific version tags to avoid unexpected upgrades.

#### Core Architecture Changes

```yaml
# InfluxDB 3.x uses Apache Parquet instead of TSM
Storage Engine: Apache Parquet (columnar format)
Built With: Rust + FDAP stack (Flight, DataFusion, Arrow, Parquet)
Cardinality Support: Infinite (major improvement over TSM)
Performance: 10x ingestion, 100x query speed on high cardinality
```

#### Storage Components

**Object Store:**
- Stores time series data in Apache Parquet format
- Each Parquet file represents a partition (default: daily)
- Data is sorted, encoded, and compressed
- Supports S3, Azure Blob Storage, Google Cloud Storage, MinIO

**Architecture Benefits:**
```text
✓ Unlimited tag cardinality (vs TSM limitations)
✓ Real-time queries on columnar data
✓ 45x faster ingestion than Enterprise v1
✓ Reduced storage costs
✓ Better compression efficiency
```

### InfluxDB 2.x Architecture (TSM-Based)

#### TSM Storage Engine

**Time-Structured Merge Tree (TSM):**
```text
Write Path:
1. Data arrives → Write-Ahead Log (WAL) for durability
2. In-memory cache holds recent writes
3. Periodic snapshots write cache to TSM files
4. Compaction merges TSM files into optimized structures

Compaction Levels:
- Snapshots: Cache → TSM
- Level 1-4: Progressive compaction for optimization
- Index Optimization: Splits series across new TSM files
- Full Compaction: Runs when shard is cold or after deletes
```

**TSM File Structure:**
```text
TSM File = Compressed columnar data + Index
- Series keys indexed for fast lookups
- Tag values indexed (important for cardinality)
- Field values NOT indexed
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

### Example TDD Workflow for InfluxDB

```python
# Step 1: RED - Write failing test
import pytest
from datetime import datetime, timedelta, timezone
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS

@pytest.fixture
def influx_client():
    client = InfluxDBClient(url="http://localhost:8086", token="test-token", org="test-org")
    yield client
    # Cleanup: delete test bucket data
    delete_api = client.delete_api()
    delete_api.delete(
        start=datetime(1970, 1, 1, tzinfo=timezone.utc),
        stop=datetime(2100, 1, 1, tzinfo=timezone.utc),
        predicate="",
        bucket="test-bucket",
        org="test-org"
    )
    client.close()

def test_hourly_mean_cpu_aggregation(influx_client):
    """Test that hourly mean aggregation correctly averages CPU usage per host."""
    write_api = influx_client.write_api(write_options=SYNCHRONOUS)

    # Write test data: 4 points across 1 hour for one host
    base_time = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
    values = [40.0, 60.0, 80.0, 20.0]  # mean = 50.0
    for i, val in enumerate(values):
        point = Point("cpu_usage") \
            .tag("host", "server-01") \
            .field("value", val) \
            .time(base_time + timedelta(minutes=i * 15))
        write_api.write(bucket="test-bucket", record=point)

    # Query hourly mean
    query_api = influx_client.query_api()
    result = query_api.query(f'''
        from(bucket: "test-bucket")
          |> range(start: 2024-01-15T10:00:00Z, stop: 2024-01-15T11:00:00Z)
          |> filter(fn: (r) => r._measurement == "cpu_usage")
          |> filter(fn: (r) => r.host == "server-01")
          |> aggregateWindow(every: 1h, fn: mean, createEmpty: false)
    ''')

    records = result[0].records
    assert len(records) == 1
    assert records[0].get_value() == 50.0

# FAILS - aggregation query or write path not yet implemented in production code

# Step 2: GREEN - Implement the aggregation function
def get_hourly_cpu_mean(client, bucket, host, start, stop):
    query_api = client.query_api()
    result = query_api.query(f'''
        from(bucket: "{bucket}")
          |> range(start: {start}, stop: {stop})
          |> filter(fn: (r) => r._measurement == "cpu_usage")
          |> filter(fn: (r) => r.host == "{host}")
          |> aggregateWindow(every: 1h, fn: mean, createEmpty: false)
    ''')
    return [{"time": r.get_time(), "mean": r.get_value()} for table in result for r in table.records]

# PASSES

# Step 3: REFACTOR - Add parameterized bucket, error handling, multiple hosts
def get_hourly_cpu_mean(client, bucket, hosts, start, stop):
    host_filter = " or ".join([f'r.host == "{h}"' for h in hosts])
    query_api = client.query_api()
    result = query_api.query(f'''
        from(bucket: "{bucket}")
          |> range(start: {start}, stop: {stop})
          |> filter(fn: (r) => r._measurement == "cpu_usage")
          |> filter(fn: (r) => {host_filter})
          |> aggregateWindow(every: 1h, fn: mean, createEmpty: false)
          |> group(columns: ["host"])
    ''')
    return {
        table.records[0].values["host"]: [
            {"time": r.get_time(), "mean": r.get_value()} for r in table.records
        ]
        for table in result
    }
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
# Bug Report: BUG-3091 - Downsampling task drops data points written
# during the last 10 seconds of each hour due to incorrect range boundary.

import pytest
from datetime import datetime, timezone
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS

def test_bug_3091_downsample_boundary_data_not_lost(influx_client):
    """Regression test: Data written at hour boundary must be included in downsample."""
    write_api = influx_client.write_api(write_options=SYNCHRONOUS)

    # Write a point at 10:59:55 (5 seconds before hour boundary)
    boundary_point = Point("cpu_usage") \
        .tag("host", "server-01") \
        .field("value", 99.0) \
        .time(datetime(2024, 1, 15, 10, 59, 55, tzinfo=timezone.utc))
    write_api.write(bucket="test-bucket", record=boundary_point)

    # Query the window that should include this point
    query_api = influx_client.query_api()
    result = query_api.query('''
        from(bucket: "test-bucket")
          |> range(start: 2024-01-15T10:00:00Z, stop: 2024-01-15T11:00:00Z)
          |> filter(fn: (r) => r._measurement == "cpu_usage")
          |> filter(fn: (r) => r.host == "server-01")
          |> aggregateWindow(every: 1h, fn: mean, createEmpty: false)
    ''')

    records = result[0].records
    assert len(records) == 1, "Boundary data point was dropped from aggregation window"
    assert records[0].get_value() == 99.0

# Fix: Changed downsample task range from
#   range(start: -1h, stop: -10s)   <-- BUG: excluded last 10 seconds
# to:
#   range(start: -1h)               <-- FIXED: includes all data up to task execution
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

## 3. Data Modeling

### Core Data Elements

```text
Measurement: Container for tags, fields, and timestamps
├── Tags: Indexed metadata (strings only)
│   └── Use for: Dimensions, categorical data, filtering
├── Fields: Non-indexed values (any type)
│   └── Use for: Measured values, metrics
└── Timestamp: Nanosecond precision (adjustable)
```

### Best Practices

#### Tags vs Fields Decision Matrix

```yaml
Use TAGS for:
  - Frequently queried metadata
  - Group by operations
  - Where clause filtering
  - Low to medium cardinality data
  - Categorical information

Use FIELDS for:
  - Measured numeric values
  - High cardinality data
  - Data not used for filtering
  - Values that change frequently
  - Unique identifiers (UUIDs, hashes)
```

#### Example: Good Data Model

```line-protocol
# Good: Tags for metadata, fields for measurements
weather,location=us-midwest,sensor=sensor-001 temperature=82.5,humidity=65.2,pressure=1013.25 1704067200000000000

# Good: Low cardinality tags
cpu_usage,host=server-01,region=us-east,env=production value=45.2,idle=54.8 1704067200000000000
```

#### Example: Bad Data Model

```line-protocol
# Bad: High cardinality in tags (user IDs)
requests,user_id=a1b2c3d4,endpoint=/api/data status=200 1704067200000000000

# Bad: Measured values in tags
temperature,value=82.5,location=us-midwest reading=1 1704067200000000000

# Bad: Timestamp in tag
metrics,timestamp=2024-01-01T00:00:00Z,host=server-01 value=100 1704067200000000000
```

### Data Structure Rules

**CRITICAL:**
```text
1. Store data in tag/field VALUES, not in keys or measurements
2. All points in a measurement should have the same tags
3. Use unique names for tags and fields (no overlap)
4. Tags and fields with same name cause column conflicts
```

## 4. Schema Design

### Schema Design Principles

#### Cardinality Considerations

```yaml
# InfluxDB 3.x: Unlimited cardinality support
InfluxDB_3:
  Tag_Cardinality: Infinite
  Performance_Impact: Minimal
  Use_Case: IIoT, high-dimensional data

# InfluxDB 2.x: Cardinality affects performance
InfluxDB_2_TSM:
  Tag_Cardinality: Keep as low as possible
  High_Cardinality_Impact: High memory usage, slow queries
  Recommendation: Use fields for high cardinality data
```

#### Series Key Design

```python
# Series key = measurement + tag set
# Example series keys:
"cpu_usage,host=server-01,region=us-east,env=prod"
"cpu_usage,host=server-02,region=us-east,env=prod"
"cpu_usage,host=server-01,region=us-west,env=prod"

# Total series cardinality = product of unique tag values
hosts = 1000
regions = 4
environments = 3
total_series = 1000 * 4 * 3 = 12,000 series
```

### Wide Schema Considerations

**Avoid wide schemas:**
```yaml
Problems:
  - Too many columns (tags + fields)
  - Complex primary keys with many tags
  - Reduced sorting performance
  - Poor query performance when selecting many columns

Solution:
  - Limit to 200-250 columns per measurement
  - Use separate measurements for different metric types
  - Normalize data where appropriate
```

### Tag Ordering Priority (InfluxDB 3.x)

```yaml
# Order tags by query frequency (most queried first)
Good_Order:
  measurement,region,host,env field1=value1,field2=value2

Reason:
  - Optimizes columnar storage
  - Improves query performance
  - Benefits WHERE clause filtering
  - Aids in join operations
```

## 5. Write Optimization

### Optimal Batch Sizes (2026)

```yaml
InfluxDB_2_x:
  Batch_Size: 5,000 lines of line protocol
  Max_Request_Size: ~5MB recommended

InfluxDB_3_x:
  Batch_Size: 10,000 lines or 10 MB (whichever first)
  Performance: Optimized for larger batches
```

### Write Best Practices

#### 1. Batch Writes

```python
# Python example: Batch writing
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS

client = InfluxDBClient(url="http://localhost:8086", token="my-token", org="my-org")
write_api = client.write_api(write_options=SYNCHRONOUS)

# Batch multiple points
points = []
for i in range(5000):
    point = Point("measurement") \
        .tag("location", "us-east") \
        .field("value", i) \
        .time(timestamp)
    points.append(point)

# Single batch write
write_api.write(bucket="my-bucket", record=points)
```

#### 2. Sort Tags Lexicographically

```python
# Good: Tags sorted alphabetically
cpu,env=prod,host=server-01,region=us-east value=45.2

# Bad: Tags not sorted
cpu,host=server-01,env=prod,region=us-east value=45.2
```

**Why sorting matters:**
- Improves compression
- Faster series key lookups
- Better TSM file organization
- Reduced memory usage

#### 3. Use Coarse Timestamp Precision

```yaml
# Choose appropriate precision
Millisecond: 1704067200000 (e.g., application metrics)
Second: 1704067200 (e.g., infrastructure monitoring)
Nanosecond: 1704067200000000000 (only if needed)

Performance_Impact:
  - Coarser precision = better compression
  - Less storage overhead
  - Faster writes
```

#### 4. Enable Gzip Compression

```bash
# HTTP request with gzip
curl -X POST "http://localhost:8086/api/v2/write?org=my-org&bucket=my-bucket" \
  -H "Authorization: Token my-token" \
  -H "Content-Encoding: gzip" \
  -H "Content-Type: text/plain; charset=utf-8" \
  --data-binary @data.txt.gz

# Performance: Up to 5x speed improvement
```

#### 5. Write Order Optimization

```yaml
Best_Practices:
  - Write batches in ascending time order (oldest first)
  - Sort tags lexicographically before writing
  - Group related measurements together
  - Avoid out-of-order writes when possible
```

### Shard Group Duration

```yaml
# Adjust based on throughput and retention
Default: 7 days (for unlimited retention)

High_Throughput:
  Duration: 1 day
  Benefit: Reduces compaction overhead

Low_Throughput:
  Duration: 7-30 days
  Benefit: Fewer shard groups to manage

Configuration:
  Longer duration = Better compression
  Longer duration = Faster queries (fewer files)
  Longer duration = Less data duplication
```

## 6. Query Optimization

### Flux vs InfluxQL (2026)

```yaml
Status_2026:
  Flux: Maintenance mode (no breaking changes)
  InfluxQL: Still supported
  SQL: Emerging in InfluxDB 3.x

Recommendation:
  - InfluxDB 2.x: Flux for complex queries, InfluxQL for simple
  - InfluxDB 3.x: SQL or InfluxQL preferred
```

### Flux Query Optimization

#### 1. Use Pushdown Functions

```flux
// Good: Pushdown to storage engine
from(bucket: "my-bucket")
  |> range(start: -1h)           // Pushdown
  |> filter(fn: (r) => r._measurement == "cpu")  // Pushdown
  |> filter(fn: (r) => r.host == "server-01")    // Pushdown

// Bad: Pulling data into memory first
from(bucket: "my-bucket")
  |> range(start: -1h)
  |> map(fn: (r) => ({ r with value: r._value * 2 }))  // Stops pushdown
  |> filter(fn: (r) => r.host == "server-01")         // In-memory
```

**Pushdown functions:**
```text
✓ range()
✓ filter() (with column references)
✓ group()
✓ aggregateWindow()
```

#### 2. Optimize Filter Operations

```flux
// Good: Filter with column references
filter(fn: (r) => r.host == "server-01")

// Bad: Inline processing (prevents pushdown)
filter(fn: (r) => r.host == getValue())
```

#### 3. Balance Time Range and Precision

```flux
// Query large time ranges from downsampled data
from(bucket: "downsampled-1h")
  |> range(start: -30d)
  |> filter(fn: (r) => r._measurement == "cpu")
  |> aggregateWindow(every: 1d, fn: mean)

// Query recent data from raw bucket
from(bucket: "raw")
  |> range(start: -1h)
  |> filter(fn: (r) => r._measurement == "cpu")
```

#### 4. Use Profiler for Debugging

```flux
import "profiler"

option profiler.enabledProfilers = ["query", "operator"]

from(bucket: "my-bucket")
  |> range(start: -1h)
  |> filter(fn: (r) => r._measurement == "cpu")
  |> mean()
  |> profiler.profile()
```

### InfluxQL Best Practices

```sql
-- Good: Specific time range and tags
SELECT mean("value")
FROM "cpu_usage"
WHERE time > now() - 1h
  AND "host" = 'server-01'
GROUP BY time(5m)

-- Bad: No time range (scans all data)
SELECT mean("value") FROM "cpu_usage"

-- Good: Use tag filters (indexed)
SELECT * FROM "cpu" WHERE "region" = 'us-east'

-- Bad: Field filter (full scan)
SELECT * FROM "cpu" WHERE "value" > 80
```

## 7. Compaction and TSM Management

**Note:** This section applies to InfluxDB 2.x and earlier. InfluxDB 3.x uses Parquet files.

### TSM Compaction Configuration

```toml
# influxdb.conf (v1.x/2.x)

[data]
  # Maximum throughput for compaction (bytes/second)
  compact-throughput = "48m"

  # Burst allowance above throughput limit
  compact-throughput-burst = "48m"

  # Concurrent full and level compactions
  # 0 = 50% of CPU cores
  max-concurrent-compactions = 0

  # Compact all TSM files in shard after no writes
  compact-full-write-cold-duration = "4h"

  # Maximum size of TSM files after compaction
  max-tsm-file-size = "2147483648"  # 2GB
```

### Compaction Levels

```yaml
Snapshot_Compaction:
  Trigger: Cache threshold or time interval
  Purpose: Free memory, persist to disk
  Result: Level 1 TSM files

Level_1_to_4:
  Process: Merge smaller files into larger ones
  Optimization: Better compression, fewer files
  Performance: Improved query speed

Index_Optimization:
  Trigger: Many level 4 files accumulated
  Purpose: Split series and indices
  Result: Manageable index sizes

Full_Compaction:
  Trigger: Shard becomes cold or deletes occur
  Purpose: Maximum optimization
  Result: Minimal TSM files, best performance
```

### Monitoring Compaction

```bash
# Check compaction status
curl -G 'http://localhost:8086/debug/vars' | jq '.database.compactions'

# Monitor TSM files
ls -lh /var/lib/influxdb/data/mydb/autogen/
```

### Compaction Best Practices

```yaml
Production_Settings:
  - Set max-concurrent-compactions based on CPU cores
  - Monitor compaction queue length
  - Adjust compact-throughput for I/O capacity
  - Use SSD storage for better compaction performance

Memory_Constrained:
  - Reduce cache-max-memory-size
  - Increase compact-full-write-cold-duration
  - Lower max-concurrent-compactions
```

## 8. Retention Policies and Downsampling

### Retention Policy Strategy

```sql
-- InfluxDB 1.x: Create retention policies
CREATE RETENTION POLICY "one_hour" ON "mydb" DURATION 1h REPLICATION 1
CREATE RETENTION POLICY "one_day" ON "mydb" DURATION 1d REPLICATION 1
CREATE RETENTION POLICY "one_week" ON "mydb" DURATION 7d REPLICATION 1
CREATE RETENTION POLICY "infinite" ON "mydb" DURATION INF REPLICATION 1 DEFAULT
```

### Downsampling Architecture

```yaml
# Typical downsampling strategy
Raw_Data:
  Retention: 7 days
  Resolution: 10 seconds
  Storage: ~500GB

Hourly_Rollup:
  Retention: 90 days
  Resolution: 1 minute
  Storage: ~50GB

Daily_Rollup:
  Retention: 2 years
  Resolution: 1 hour
  Storage: ~10GB
```

### InfluxDB 2.x: Task-Based Downsampling

```flux
// Downsampling task
option task = {
  name: "downsample-hourly",
  every: 1h,
  offset: 10m  // Wait for late data
}

from(bucket: "raw")
  |> range(start: -1h)
  |> filter(fn: (r) => r._measurement == "cpu_usage")
  |> aggregateWindow(every: 1m, fn: mean, createEmpty: false)
  |> to(bucket: "hourly", org: "my-org")
```

### InfluxDB 1.x: Continuous Query Downsampling

```sql
-- Downsample to 5-minute averages
CREATE CONTINUOUS QUERY "cq_5m" ON "mydb"
BEGIN
  SELECT mean("value") AS "mean_value"
  INTO "one_week"."downsampled_5m"
  FROM "one_hour"."raw_data"
  GROUP BY time(5m), *
END

-- Downsample with RESAMPLE for late data
CREATE CONTINUOUS QUERY "cq_1h" ON "mydb"
RESAMPLE EVERY 1h FOR 2h
BEGIN
  SELECT mean("value") AS "mean_value"
  INTO "infinite"."downsampled_1h"
  FROM "one_week"."downsampled_5m"
  GROUP BY time(1h), *
END
```

### Best Practices

```yaml
Downsampling_Guidelines:
  - Schedule tasks before retention period ends
  - Use offset for late-arriving data
  - Preserve tags with GROUP BY *
  - Test downsampling on non-production data first

Aggregation_Functions:
  Metrics: mean, median, sum
  Counters: sum, increase
  Gauges: mean, min, max, last
  Percentiles: Preserve with histogram buckets
```

## 9. Continuous Queries and Tasks

### Migration: CQs to Tasks (2026)

```yaml
Status:
  InfluxDB_1_x: Continuous Queries (InfluxQL)
  InfluxDB_2_x: Tasks (Flux)
  InfluxDB_3_x: Tasks (Flux in maintenance mode)
```

### Continuous Queries (InfluxDB 1.x)

```sql
-- Basic continuous query
CREATE CONTINUOUS QUERY "cq_mean" ON "mydb"
BEGIN
  SELECT mean("value")
  INTO "average_data"
  FROM "raw_data"
  GROUP BY time(1h), *
END

-- Advanced CQ with RESAMPLE
CREATE CONTINUOUS QUERY "cq_advanced" ON "mydb"
RESAMPLE EVERY 30m FOR 90m
BEGIN
  SELECT mean("temperature") AS "mean_temp",
         median("temperature") AS "median_temp"
  INTO "climate_data"."autogen"."aggregated_temp"
  FROM "climate_data"."one_hour"."raw_temp"
  GROUP BY time(15m), "location"
END
```

### Tasks (InfluxDB 2.x)

```flux
// Basic downsampling task
option task = {
  name: "downsample-cpu",
  every: 5m,
  offset: 30s
}

from(bucket: "raw")
  |> range(start: -5m)
  |> filter(fn: (r) => r._measurement == "cpu")
  |> aggregateWindow(every: 1m, fn: mean)
  |> to(bucket: "downsampled", org: "my-org")

// Task with error handling
option task = {
  name: "safe-downsample",
  every: 1h,
  offset: 10m
}

data = from(bucket: "raw")
  |> range(start: -1h)
  |> filter(fn: (r) => r._measurement == "metrics")

data
  |> aggregateWindow(every: 5m, fn: mean, createEmpty: false)
  |> to(bucket: "hourly")

// Alert on task errors
import "slack"

slack.message(
  url: "https://hooks.slack.com/services/YOUR/WEBHOOK/URL",
  text: "Task completed: ${task.name}"
)
```

### Converting CQs to Tasks

```flux
// InfluxQL CQ:
// CREATE CONTINUOUS QUERY "cq_example" ON "mydb"
// BEGIN
//   SELECT mean("value") INTO "aggregated" FROM "raw" GROUP BY time(1h), *
// END

// Equivalent Flux Task:
option task = {
  name: "cq_example_equivalent",
  every: 1h
}

from(bucket: "mydb/autogen")
  |> range(start: -1h)
  |> filter(fn: (r) => r._measurement == "raw")
  |> aggregateWindow(every: 1h, fn: mean)
  |> set(key: "_measurement", value: "aggregated")
  |> to(bucket: "mydb/autogen")
```

## 10. Aggregation Strategies

### Pre-Aggregation vs On-Demand

```yaml
Pre_Aggregation:
  Pros:
    - Fast query response
    - Reduced data volume
    - Lower query load
  Cons:
    - Storage overhead
    - Limited flexibility
    - Must plan aggregations ahead
  Use_When:
    - Known query patterns
    - Dashboard queries
    - Long-term data

On_Demand_Aggregation:
  Pros:
    - Maximum flexibility
    - No storage overhead
    - Access to raw data
  Cons:
    - Slower queries
    - Higher compute cost
    - Memory intensive
  Use_When:
    - Ad-hoc analysis
    - Recent data queries
    - Changing requirements
```

### Flux Aggregation Patterns

```flux
// 1. Simple window aggregation
from(bucket: "my-bucket")
  |> range(start: -1h)
  |> filter(fn: (r) => r._measurement == "cpu")
  |> aggregateWindow(every: 5m, fn: mean)

// 2. Multiple aggregations
from(bucket: "my-bucket")
  |> range(start: -1h)
  |> filter(fn: (r) => r._measurement == "temperature")
  |> aggregateWindow(
      every: 10m,
      fn: (column, tables=<-) => tables
        |> mean(column: column)
        |> set(key: "aggregation", value: "mean")
    )

// 3. Percentile aggregations
from(bucket: "my-bucket")
  |> range(start: -1h)
  |> filter(fn: (r) => r._measurement == "response_time")
  |> aggregateWindow(
      every: 5m,
      fn: (tables=<-, column) => tables
        |> quantile(column: column, q: 0.95)
    )

// 4. Custom aggregations
from(bucket: "my-bucket")
  |> range(start: -1h)
  |> filter(fn: (r) => r._measurement == "requests")
  |> aggregateWindow(
      every: 1m,
      fn: (tables=<-, column) => tables
        |> sum(column: column)
        |> map(fn: (r) => ({ r with _value: r._value / 60.0 }))
    )
```

### InfluxQL Aggregation

```sql
-- Time-based aggregation
SELECT mean("value"), max("value"), min("value")
FROM "cpu_usage"
WHERE time > now() - 1h
GROUP BY time(5m), "host"

-- Nested aggregations
SELECT max("mean_value") FROM (
  SELECT mean("value") AS "mean_value"
  FROM "cpu_usage"
  WHERE time > now() - 1h
  GROUP BY time(1m), "host"
)

-- Percentiles
SELECT percentile("response_time", 95) AS "p95",
       percentile("response_time", 99) AS "p99"
FROM "requests"
WHERE time > now() - 1h
GROUP BY time(5m)
```

### Aggregation Performance Tips

```yaml
Optimization:
  - Filter before aggregating (reduce data volume)
  - Use coarser windows for large time ranges
  - Leverage pre-aggregated data for dashboards
  - Limit GROUP BY cardinality

Memory_Management:
  - Avoid unbounded aggregations
  - Use streaming aggregations when possible
  - Set query memory limits
  - Monitor query performance
```

## 11. Cardinality Management

### Understanding Cardinality

```python
# Series cardinality calculation
measurement = "cpu_usage"
tags = {
    "host": 1000,      # 1000 unique hosts
    "region": 4,       # 4 regions
    "env": 3           # 3 environments
}

# Total series = product of tag cardinalities
total_series = 1000 * 4 * 3 = 12,000

# Memory impact (InfluxDB 2.x TSM):
# ~1KB per series in index
# 12,000 series ≈ 12 MB index memory
```

### InfluxDB Version Differences

```yaml
InfluxDB_3_x:
  Cardinality: Infinite (unlimited)
  Performance: No degradation with high cardinality
  Storage: Apache Parquet columnar format
  Use_Case: IIoT, high-dimensional data, user-level tracking

InfluxDB_2_x_TSM:
  Cardinality: Limited (performance degrades)
  Impact: High memory usage, slow queries
  Storage: TSM with in-memory index
  Recommendation: Keep cardinality < 1 million series
```

### Measuring Cardinality

```flux
// Flux: Measure series cardinality
import "influxdata/influxdb"

influxdb.cardinality(
  bucket: "my-bucket",
  start: -30d
)

// Show cardinality by measurement
influxdb.cardinality(
  bucket: "my-bucket",
  start: -30d,
  predicate: (r) => true
)
  |> group(columns: ["_measurement"])
  |> count()
```

```sql
-- InfluxQL: Show series cardinality
SHOW SERIES CARDINALITY
SHOW SERIES CARDINALITY ON mydb

-- Show measurement cardinality
SHOW MEASUREMENT CARDINALITY ON mydb

-- Show tag key cardinality
SHOW TAG KEY CARDINALITY ON mydb
SHOW TAG KEY CARDINALITY ON mydb FROM cpu_usage

-- Show tag value cardinality
SHOW TAG VALUES CARDINALITY ON mydb WITH KEY = "host"
```

### Identifying High Cardinality Problems

```flux
// Find high cardinality tags
import "influxdata/influxdb/schema"

schema.tagValues(
  bucket: "my-bucket",
  tag: "user_id",
  start: -7d
)
  |> count()
  |> filter(fn: (r) => r._value > 10000)  // High cardinality threshold
```

### Resolving High Cardinality

```yaml
Prevention:
  - Never use unique IDs in tags (UUIDs, session IDs)
  - Avoid timestamps in tags
  - Don't use unbounded strings in tags
  - Use fields for high cardinality data

Refactoring:
  - Move high cardinality tags to fields
  - Reduce tag value diversity
  - Use separate measurements
  - Implement tag value limits

Example_Bad:
  tags: user_id=abc123  # Millions of users

Example_Good:
  fields: user_id="abc123"
  tags: user_tier=premium  # Limited values
```

### Common High Cardinality Causes

```yaml
Anti_Patterns:
  - User IDs as tags
  - Session IDs as tags
  - Request IDs as tags
  - Timestamps as tags
  - Log messages as tags
  - Full URLs as tags
  - Random strings/hashes as tags

Solutions:
  - Use fields for unique identifiers
  - Hash or bucket continuous values
  - Use categorical values in tags
  - Limit tag value sets
```

## 12. Memory Optimization

### Time Series Index (TSI) Configuration

```bash
# InfluxDB 2.x: Use TSI instead of in-memory index
docker run -d \
  -p 8086:8086 \
  -e INFLUXDB_DATA_INDEX_VERSION="tsi1" \
  influxdb:2.7

# Benefits:
# - Reduced memory usage
# - Disk-backed index
# - Better scaling for high cardinality
```

### Memory Configuration

```toml
# influxdb.conf (v1.x/2.x)

[data]
  # Maximum size of in-memory cache
  cache-max-memory-size = "1g"

  # Snapshot cache when this size is reached
  cache-snapshot-memory-size = "25m"

  # Interval to check if cache should be snapshotted
  cache-snapshot-write-cold-duration = "10m"

[coordinator]
  # Maximum time a query can run
  query-timeout = "0s"  # 0 = unlimited (not recommended)

  # Maximum concurrent queries
  max-concurrent-queries = 0  # 0 = unlimited (not recommended)

  # Maximum number of points for SELECT statements
  max-select-point = 0
```

### Query Memory Limits

```flux
// Set memory limit for query
option task = {
  name: "memory-limited-query",
  every: 5m
}

option v = {
  memoryLimit: 500MB
}

from(bucket: "my-bucket")
  |> range(start: -5m)
  |> filter(fn: (r) => r._measurement == "cpu")
```

### Memory Optimization Strategies

```yaml
Reduce_Memory_Usage:
  1_Enable_TSI:
    - Use index-version = "tsi1"
    - Disk-backed index instead of in-memory

  2_Optimize_Cache:
    - Lower cache-max-memory-size
    - Increase snapshot frequency
    - Tune cache-snapshot-memory-size

  3_Query_Optimization:
    - Limit query time ranges
    - Use downsampled data for large ranges
    - Set max-select-point limits
    - Enable query timeouts

  4_Cardinality_Control:
    - Reduce high cardinality tags
    - Use fields for unique identifiers
    - Monitor series cardinality

  5_Retention_Policies:
    - Delete old data automatically
    - Use shorter retention periods
    - Implement aggressive downsampling
```

### Memory Monitoring

```bash
# Check memory usage
curl -G 'http://localhost:8086/debug/vars' | jq '.memstats'

# Monitor cache size
curl -G 'http://localhost:8086/debug/vars' | jq '.database.cache'

# Check query memory
curl -G 'http://localhost:8086/debug/vars' | jq '.queryExecutor'
```

## 13. Security

### Authentication and Authorization

```yaml
# InfluxDB 2.x: Token-based authentication
Default_Setup:
  Authentication: Required by default
  Authorization: Token-based (JWT)
  Admin_Setup: Required on first run

# InfluxDB 1.x: User-based authentication
Legacy_Setup:
  Authentication: Optional (enable in config)
  Authorization: User/password
  Recommendation: Enable for production
```

### Enabling TLS/SSL

```toml
# influxdb.conf (v1.x)
[http]
  https-enabled = true
  https-certificate = "/etc/ssl/influxdb-selfsigned.crt"
  https-private-key = "/etc/ssl/influxdb-selfsigned.key"
```

```yaml
# InfluxDB 2.x: TLS configuration
tls-cert: /etc/ssl/influxdb.crt
tls-key: /etc/ssl/influxdb.key

# TLS version requirements (2026)
Minimum_TLS_Version: 1.2
Rejected: TLS 1.1 and earlier
```

### Certificate Types

```yaml
Production_Certificate:
  Type: Single domain certificate
  Signed_By: Trusted CA (Let's Encrypt, DigiCert, etc.)
  Features:
    - Cryptographic security
    - Client identity verification
  Requirement: Unique certificate per instance

Self_Signed_Certificate:
  Type: Self-signed
  Features:
    - Cryptographic security only
    - No identity verification
  Use_Case: Development, testing
  Not_Recommended: Production
```

### Generating Self-Signed Certificate

```bash
# Generate self-signed certificate (development only)
openssl req -x509 -nodes -newkey rsa:2048 \
  -keyout /etc/ssl/influxdb-selfsigned.key \
  -out /etc/ssl/influxdb-selfsigned.crt \
  -days 365 \
  -subj "/C=US/ST=State/L=City/O=Org/CN=influxdb.local"

# Set permissions
chmod 600 /etc/ssl/influxdb-selfsigned.key
chmod 644 /etc/ssl/influxdb-selfsigned.crt
```

### Token Management (InfluxDB 2.x)

```bash
# Create token with specific permissions
influx auth create \
  --org my-org \
  --description "Application token" \
  --read-bucket abc123 \
  --write-bucket abc123

# List tokens
influx auth list

# Delete token
influx auth delete --id <token-id>
```

### Security Best Practices

```yaml
Production_Checklist:
  ✓ Enable authentication
  ✓ Use TLS/SSL (minimum TLS 1.2)
  ✓ Use signed certificates from trusted CA
  ✓ Implement least-privilege access
  ✓ Rotate tokens/passwords regularly
  ✓ Use different tokens for different apps
  ✓ Enable network encryption
  ✓ Limit network access (firewall rules)
  ✓ Monitor authentication logs
  ✓ Use secrets management (Vault, etc.)

Docker_Security:
  ✓ Use Docker secrets for tokens
  ✓ Don't expose credentials in environment variables
  ✓ Use read-only root filesystem when possible
  ✓ Run as non-root user (UID 1500)
  ✓ Limit container resources
```

### Network Security

```yaml
Firewall_Rules:
  InfluxDB_Port: 8086 (HTTP API)
  Restrict_Access: Whitelist IPs only
  Internal_Only: Bind to internal interface

Docker_Network:
  Use_Private_Networks: true
  Expose_Ports_Selectively: true
  Enable_Network_Isolation: true
```

## 14. Clustering and High Availability

### InfluxDB Version Comparison (2026)

```yaml
InfluxDB_3_Enterprise:
  Version: 3.8+ (2026)
  Features:
    - Clustering with high availability
    - Advanced security (SSO, ABAC)
    - 100x faster queries on high cardinality
    - 45x faster ingestion vs Enterprise v1
  License: Commercial

InfluxDB_3_Clustered:
  Version: Self-managed cluster
  Features:
    - On-premises deployment
    - High availability
    - Distributed architecture
  License: Commercial
  Note: Replaces older InfluxDB Enterprise

InfluxDB_OSS:
  Version: 2.x / 3.x Core
  Features:
    - Single node only
    - No native clustering
    - No high availability
  License: MIT/Apache 2.0
  Workarounds: Third-party solutions (influxdb-ha)
```

### InfluxDB 3 Enterprise Architecture

```yaml
Components:
  Ingester:
    Role: Receives and validates writes
    Scaling: Horizontal (multiple instances)

  Compactor:
    Role: Compacts Parquet files
    Scaling: Dedicated nodes with high CPU

  Querier:
    Role: Executes queries
    Scaling: Horizontal (multiple instances)

  Catalog:
    Role: Metadata management
    Storage: PostgreSQL-compatible

Storage:
  Object_Store: S3, Azure Blob, Google Cloud Storage
  Format: Apache Parquet
  Replication: Handled by object store
```

### High Availability Strategies

```yaml
# InfluxDB 3 Enterprise: Built-in HA
Enterprise_HA:
  Ingester_HA:
    - Multiple ingester nodes
    - Load balancing across ingesters
    - Automatic failover

  Querier_HA:
    - Multiple querier nodes
    - Read replicas
    - Query routing

  Storage_HA:
    - Object store replication
    - Multi-region support
    - Disaster recovery

# InfluxDB OSS: External HA solutions
OSS_Workarounds:
  Relay:
    - influxdb-relay for write HA
    - Duplicate writes to multiple nodes
    - Application-level failover

  Proxy:
    - Load balancer (HAProxy, nginx)
    - Health checks
    - Failover routing

  Backup_Restore:
    - Regular automated backups
    - Standby instance
    - Manual failover process
```

### Load Balancing Configuration

```nginx
# nginx load balancer for InfluxDB
upstream influxdb_backend {
    least_conn;
    server influxdb1:8086 max_fails=3 fail_timeout=30s;
    server influxdb2:8086 max_fails=3 fail_timeout=30s;
    server influxdb3:8086 max_fails=3 fail_timeout=30s;
}

server {
    listen 8086;

    location / {
        proxy_pass http://influxdb_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_connect_timeout 5s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }
}
```

### Kubernetes HA Deployment

```yaml
# StatefulSet for InfluxDB with multiple replicas
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: influxdb
spec:
  serviceName: influxdb
  replicas: 3
  selector:
    matchLabels:
      app: influxdb
  template:
    metadata:
      labels:
        app: influxdb
    spec:
      containers:
      - name: influxdb
        image: influxdb:3.8
        ports:
        - containerPort: 8086
        volumeMounts:
        - name: influxdb-storage
          mountPath: /var/lib/influxdb
  volumeClaimTemplates:
  - metadata:
      name: influxdb-storage
    spec:
      accessModes: [ "ReadWriteOnce" ]
      resources:
        requests:
          storage: 100Gi
```

## 15. Backup and Restore

### Backup Strategies by Version

```yaml
InfluxDB_2_x:
  Command: influx backup
  Format: Proprietary
  Compatibility: Minor versions only (2.6 <-> 2.7)
  Full_Backup: Data + metadata

InfluxDB_1_x:
  Command: influx backup / influx_inspect export
  Formats:
    - Legacy (pre-1.5)
    - Portable (1.5+, recommended)
  Compatibility: Portable format supports version migration

InfluxDB_3_x:
  Method: Object store snapshots
  Frequency: Hourly and daily
  Location: /catalog_backup_file_lists
  RPO: < 15 minutes (typical)
```

### InfluxDB 2.x Backup

```bash
# Full backup
influx backup /path/to/backup \
  --host http://localhost:8086 \
  --token my-token

# Backup specific bucket
influx backup /path/to/backup \
  --host http://localhost:8086 \
  --token my-token \
  --bucket my-bucket

# Restore backup
influx restore /path/to/backup \
  --host http://localhost:8086 \
  --token my-token \
  --new-bucket restored-bucket \
  --new-org restored-org
```

### InfluxDB 1.x Backup

```bash
# Portable format backup (recommended)
influxd backup -portable /path/to/backup

# Backup specific database
influxd backup -portable -database mydb /path/to/backup

# Backup with retention policy
influxd backup -portable \
  -database mydb \
  -retention autogen \
  /path/to/backup

# Restore portable backup
influxd restore -portable /path/to/backup

# Restore to specific database
influxd restore -portable \
  -db mydb \
  -newdb restored_db \
  /path/to/backup
```

### InfluxDB 1.x: Export/Import (Large Datasets)

```bash
# Export as line protocol
influx_inspect export \
  -database mydb \
  -retention autogen \
  -datadir /var/lib/influxdb/data \
  -waldir /var/lib/influxdb/wal \
  -out export.txt

# Compress export
gzip export.txt

# Import line protocol
influx -import -path=export.txt -database=mydb -precision=ns
```

### InfluxDB 3 Enterprise/Clustered Backup

```bash
# Backups are automatic in object storage
# Verify snapshot completeness by checking bloom filter
ls -lh /object-store/catalog_backup_file_lists/*/bloom.bin.gz

# Restore from snapshot
influxdb3-restore \
  --snapshot-path /object-store/catalog_backup_file_lists/2026-01-01 \
  --target-path /var/lib/influxdb3
```

### Backup Best Practices

```yaml
Scheduling:
  Frequency:
    - Production: Daily minimum
    - Critical: Hourly
    - Very critical: Continuous replication

  Timing:
    - Schedule during low-traffic periods
    - Avoid peak write times
    - Consider compaction cycles

  Retention:
    - Daily backups: 7-30 days
    - Weekly backups: 3-6 months
    - Monthly backups: 1-2 years

Storage:
  Location:
    - Off-server storage (S3, NFS, etc.)
    - Multi-region for DR
    - Separate from InfluxDB data directory

  Verification:
    - Test restore procedures regularly
    - Verify backup integrity (checksums)
    - Check bloom.bin.gz exists (v3)
    - Monitor backup size trends

Automation:
  Tools:
    - Cron jobs
    - Kubernetes CronJobs
    - CI/CD pipelines

  Monitoring:
    - Alert on backup failures
    - Track backup duration
    - Monitor storage usage
```

### Docker Backup

```bash
# Backup Docker volume
docker run --rm \
  -v influxdb-storage:/source \
  -v $(pwd)/backup:/backup \
  alpine tar czf /backup/influxdb-backup-$(date +%Y%m%d).tar.gz -C /source .

# Restore Docker volume
docker run --rm \
  -v influxdb-storage:/target \
  -v $(pwd)/backup:/backup \
  alpine tar xzf /backup/influxdb-backup-20260101.tar.gz -C /target
```

### Backup Script Example

```bash
#!/bin/bash
# InfluxDB backup script

BACKUP_DIR="/backups/influxdb"
DATE=$(date +%Y%m%d-%H%M%S)
RETENTION_DAYS=30

# Create backup
influx backup "${BACKUP_DIR}/${DATE}" \
  --host http://localhost:8086 \
  --token "${INFLUX_TOKEN}"

# Compress backup
tar czf "${BACKUP_DIR}/${DATE}.tar.gz" -C "${BACKUP_DIR}" "${DATE}"
rm -rf "${BACKUP_DIR}/${DATE}"

# Upload to S3
aws s3 cp "${BACKUP_DIR}/${DATE}.tar.gz" \
  s3://my-backups/influxdb/

# Clean old backups
find "${BACKUP_DIR}" -name "*.tar.gz" -mtime +${RETENTION_DAYS} -delete

# Alert on failure
if [ $? -ne 0 ]; then
  echo "Backup failed" | mail -s "InfluxDB Backup Failure" admin@example.com
fi
```

## 16. Monitoring and Operations

### InfluxDB Metrics Endpoint

```bash
# InfluxDB 2.x/3.x: Prometheus-compatible metrics
curl http://localhost:8086/metrics

# Key metrics exposed:
# - go_* (Go runtime metrics)
# - http_* (HTTP request metrics)
# - task_* (Task execution metrics)
# - storage_* (Storage metrics)
```

### Internal Monitoring Database

```yaml
# InfluxDB 1.x: _internal database
Database: _internal
Measurements:
  - cq: Continuous query stats
  - database: Database metrics
  - httpd: HTTP request metrics
  - queryExecutor: Query execution stats
  - shard: Shard metrics
  - tsm1_cache: Cache statistics
  - tsm1_engine: TSM engine metrics
  - write: Write statistics

Warning: Not recommended for production clusters (overhead)
```

### Monitoring with Telegraf

```toml
# telegraf.conf - Monitor InfluxDB
[[inputs.influxdb]]
  urls = ["http://localhost:8086/debug/vars"]
  timeout = "5s"

[[inputs.prometheus]]
  urls = ["http://localhost:8086/metrics"]
  metric_version = 2

[[outputs.influxdb_v2]]
  urls = ["http://monitoring-influxdb:8086"]
  token = "${INFLUX_TOKEN}"
  organization = "monitoring"
  bucket = "telegraf"
```

### Key Metrics to Monitor

```yaml
System_Metrics:
  CPU:
    - CPU usage per core
    - Process CPU usage
    - IOWait

  Memory:
    - Total memory usage
    - Cache memory usage
    - Heap allocations
    - GC pause time

  Disk:
    - Disk usage (data directory)
    - I/O wait time
    - Read/write throughput
    - TSM/Parquet file count

Database_Metrics:
  Writes:
    - Write throughput (points/sec)
    - Write errors
    - Batch size distribution
    - WAL size

  Queries:
    - Query rate
    - Query duration (p50, p95, p99)
    - Query errors
    - Query memory usage

  Storage:
    - Database size
    - Shard count
    - Series cardinality
    - Compaction queue length

Performance_Metrics:
  Compaction:
    - Compaction duration
    - Compaction errors
    - TSM files before/after

  Cache:
    - Cache hit ratio
    - Cache evictions
    - Cache memory usage

  Network:
    - HTTP request rate
    - HTTP error rate (4xx, 5xx)
    - Connection count
```

### Grafana Dashboard Setup

```yaml
# Grafana datasource configuration
apiVersion: 1
datasources:
  - name: InfluxDB
    type: influxdb
    access: proxy
    url: http://influxdb:8086
    jsonData:
      version: Flux
      organization: my-org
      defaultBucket: monitoring
      tlsSkipVerify: false
    secureJsonData:
      token: ${INFLUX_TOKEN}
```

### Alert Rules

```flux
// Critical: High write errors
import "influxdata/influxdb/monitor"
import "slack"

option task = {
  name: "alert-write-errors",
  every: 1m
}

from(bucket: "monitoring")
  |> range(start: -2m)
  |> filter(fn: (r) => r._measurement == "influxdb_write")
  |> filter(fn: (r) => r._field == "errors")
  |> aggregateWindow(every: 1m, fn: sum)
  |> filter(fn: (r) => r._value > 100)
  |> monitor.notify(
      endpoint: slack.endpoint(url: "https://hooks.slack.com/..."),
      message: "High write errors: ${r._value} in the last minute"
    )

// Warning: High memory usage
from(bucket: "monitoring")
  |> range(start: -5m)
  |> filter(fn: (r) => r._measurement == "mem")
  |> filter(fn: (r) => r._field == "used_percent")
  |> aggregateWindow(every: 1m, fn: mean)
  |> filter(fn: (r) => r._value > 80.0)
  |> monitor.notify(
      endpoint: slack.endpoint(url: "https://hooks.slack.com/..."),
      message: "Memory usage above 80%: ${r._value}%"
    )
```

### Health Checks

```bash
# InfluxDB 2.x health endpoint
curl http://localhost:8086/health

# Response: {"name":"influxdb","message":"ready for queries and writes","status":"pass"}

# Kubernetes liveness probe
livenessProbe:
  httpGet:
    path: /health
    port: 8086
  initialDelaySeconds: 30
  periodSeconds: 10

# Kubernetes readiness probe
readinessProbe:
  httpGet:
    path: /health
    port: 8086
  initialDelaySeconds: 5
  periodSeconds: 5
```

### Operational Runbooks

```yaml
High_Memory_Usage:
  1_Identify:
    - Check series cardinality
    - Review query patterns
    - Examine cache size
  2_Resolve:
    - Enable TSI if not enabled
    - Reduce cardinality
    - Optimize queries
    - Increase memory limits

Slow_Queries:
  1_Identify:
    - Check query logs
    - Profile queries with Flux profiler
    - Review time ranges
  2_Resolve:
    - Add time filters
    - Use downsampled data
    - Optimize schema
    - Add query limits

Write_Failures:
  1_Identify:
    - Check disk space
    - Review error logs
    - Verify authentication
  2_Resolve:
    - Free disk space
    - Fix authentication tokens
    - Check schema conflicts
    - Reduce batch size

Compaction_Issues:
  1_Identify:
    - Monitor compaction queue
    - Check TSM file count
    - Review compaction logs
  2_Resolve:
    - Increase compaction concurrency
    - Adjust compaction throughput
    - Check disk I/O capacity
    - Consider manual compaction
```

## 17. Docker Deployment

### Critical 2026 Update

```yaml
Important_Change:
  Date: April 7, 2026
  Change: "latest" tag points to InfluxDB 3 Core
  Action: Use specific version tags to avoid unexpected upgrades

Recommended_Tags:
  InfluxDB_2_x: "influxdb:2.7"
  InfluxDB_3_Core: "influxdb:3.8" or "influxdb:3-core"
  Specific_Version: "influxdb:2.7.5"
```

### Basic Docker Deployment

```bash
# InfluxDB 2.x
docker run -d \
  --name influxdb \
  -p 8086:8086 \
  -v influxdb-storage:/var/lib/influxdb2 \
  -e DOCKER_INFLUXDB_INIT_MODE=setup \
  -e DOCKER_INFLUXDB_INIT_USERNAME=admin \
  -e DOCKER_INFLUXDB_INIT_PASSWORD=securepassword \
  -e DOCKER_INFLUXDB_INIT_ORG=my-org \
  -e DOCKER_INFLUXDB_INIT_BUCKET=my-bucket \
  -e DOCKER_INFLUXDB_INIT_ADMIN_TOKEN=my-super-secret-token \
  influxdb:2.7

# InfluxDB 3 Core
docker run -d \
  --name influxdb3 \
  -p 8086:8086 \
  -v influxdb3-storage:/var/lib/influxdb3 \
  -e INFLUXDB3_OBJECT_STORE=file \
  -e INFLUXDB3_STORAGE_PATH=/var/lib/influxdb3 \
  influxdb:3-core
```

### Docker Compose Configuration

```yaml
# docker-compose.yml
version: '3.8'

services:
  influxdb:
    image: influxdb:2.7
    container_name: influxdb
    restart: unless-stopped
    ports:
      - "8086:8086"
    volumes:
      - influxdb-storage:/var/lib/influxdb2
      - ./influxdb.conf:/etc/influxdb/influxdb.conf:ro
    environment:
      - DOCKER_INFLUXDB_INIT_MODE=setup
      - DOCKER_INFLUXDB_INIT_USERNAME_FILE=/run/secrets/influx_username
      - DOCKER_INFLUXDB_INIT_PASSWORD_FILE=/run/secrets/influx_password
      - DOCKER_INFLUXDB_INIT_ADMIN_TOKEN_FILE=/run/secrets/influx_token
      - DOCKER_INFLUXDB_INIT_ORG=my-org
      - DOCKER_INFLUXDB_INIT_BUCKET=my-bucket
    secrets:
      - influx_username
      - influx_password
      - influx_token
    networks:
      - monitoring
    healthcheck:
      test: ["CMD", "influx", "ping"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

volumes:
  influxdb-storage:
    driver: local

secrets:
  influx_username:
    file: ./secrets/influx_username.txt
  influx_password:
    file: ./secrets/influx_password.txt
  influx_token:
    file: ./secrets/influx_token.txt

networks:
  monitoring:
    driver: bridge
```

### Docker Compose with TLS

```yaml
# docker-compose-tls.yml
version: '3.8'

services:
  influxdb:
    image: influxdb:2.7
    container_name: influxdb-secure
    restart: unless-stopped
    ports:
      - "8086:8086"
    volumes:
      - influxdb-storage:/var/lib/influxdb2
      - ./certs/influxdb.crt:/etc/ssl/influxdb.crt:ro
      - ./certs/influxdb.key:/etc/ssl/influxdb.key:ro
    environment:
      - DOCKER_INFLUXDB_INIT_MODE=setup
      - DOCKER_INFLUXDB_INIT_USERNAME=admin
      - DOCKER_INFLUXDB_INIT_PASSWORD=${INFLUX_PASSWORD}
      - DOCKER_INFLUXDB_INIT_ORG=my-org
      - DOCKER_INFLUXDB_INIT_BUCKET=my-bucket
      - DOCKER_INFLUXDB_INIT_ADMIN_TOKEN=${INFLUX_TOKEN}
    command: >
      influxd
      --tls-cert=/etc/ssl/influxdb.crt
      --tls-key=/etc/ssl/influxdb.key
    networks:
      - monitoring

volumes:
  influxdb-storage:

networks:
  monitoring:
```

### Production Docker Configuration

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  influxdb:
    image: influxdb:2.7
    container_name: influxdb-prod
    restart: always
    ports:
      - "127.0.0.1:8086:8086"  # Bind to localhost only
    volumes:
      - /data/influxdb:/var/lib/influxdb2
      - ./influxdb.conf:/etc/influxdb/influxdb.conf:ro
    environment:
      - DOCKER_INFLUXDB_INIT_MODE=setup
      - DOCKER_INFLUXDB_INIT_USERNAME_FILE=/run/secrets/influx_username
      - DOCKER_INFLUXDB_INIT_PASSWORD_FILE=/run/secrets/influx_password
      - DOCKER_INFLUXDB_INIT_ADMIN_TOKEN_FILE=/run/secrets/influx_token
      - DOCKER_INFLUXDB_INIT_ORG=production
      - DOCKER_INFLUXDB_INIT_BUCKET=metrics
    secrets:
      - influx_username
      - influx_password
      - influx_token
    user: "1500:1500"  # InfluxDB user
    security_opt:
      - no-new-privileges:true
    read_only: true
    tmpfs:
      - /tmp
      - /var/run
    deploy:
      resources:
        limits:
          cpus: '4'
          memory: 8G
        reservations:
          cpus: '2'
          memory: 4G
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
    healthcheck:
      test: ["CMD", "influx", "ping"]
      interval: 30s
      timeout: 10s
      retries: 3
    networks:
      - monitoring

secrets:
  influx_username:
    file: /secure/influx_username.txt
  influx_password:
    file: /secure/influx_password.txt
  influx_token:
    file: /secure/influx_token.txt

networks:
  monitoring:
    driver: bridge
    internal: true
```

### Volume Permissions

```bash
# InfluxDB runs as user influxdb (UID/GID 1500)
# Set correct permissions on host
sudo mkdir -p /data/influxdb
sudo chown -R 1500:1500 /data/influxdb
sudo chmod 755 /data/influxdb

# Or use Docker to set permissions
docker run --rm \
  -v /data/influxdb:/data \
  alpine chown -R 1500:1500 /data
```

### Docker Best Practices

```yaml
Security:
  ✓ Use Docker secrets for credentials
  ✓ Run as non-root user (1500:1500)
  ✓ Use read-only root filesystem
  ✓ Bind to localhost or private network
  ✓ Enable TLS for production
  ✓ Set resource limits
  ✓ Use specific version tags

Performance:
  ✓ Use volumes for persistent storage (not bind mounts)
  ✓ Allocate sufficient memory (min 2GB, recommended 8GB+)
  ✓ Use SSD storage for volumes
  ✓ Set appropriate CPU limits

Reliability:
  ✓ Enable health checks
  ✓ Use restart policies (unless-stopped/always)
  ✓ Implement proper logging
  ✓ Monitor container metrics
  ✓ Regular backups of volumes

Networking:
  ✓ Use Docker networks for isolation
  ✓ Limit port exposure
  ✓ Use reverse proxy (nginx, traefik)
  ✓ Enable network encryption
```

## 18. Kubernetes Deployment

### InfluxDB Operator

```yaml
# Install InfluxDB Operator
Official_Operator: github.com/influxdata/influxdata-operator
Status: Active (2026)
Platforms: GKE, EKS, AKS, OpenShift

Features:
  - Automated deployment
  - Backup management
  - Persistent volume support
  - Multi-instance management
```

### Helm Deployment

```bash
# Add InfluxData Helm repository
helm repo add influxdata https://helm.influxdata.com/
helm repo update

# Install InfluxDB 2.x
helm install influxdb influxdata/influxdb2 \
  --namespace monitoring \
  --create-namespace \
  --set persistence.enabled=true \
  --set persistence.size=100Gi \
  --set resources.requests.memory=4Gi \
  --set resources.requests.cpu=2 \
  --set resources.limits.memory=8Gi \
  --set resources.limits.cpu=4

# Install InfluxDB 3 Clustered (Enterprise)
helm install influxdb3 influxdata/influxdb3-clustered \
  --namespace influxdb3 \
  --create-namespace \
  --values values.yaml
```

### StatefulSet Deployment

```yaml
# influxdb-statefulset.yaml
apiVersion: v1
kind: Service
metadata:
  name: influxdb
  namespace: monitoring
spec:
  ports:
  - port: 8086
    name: http
  clusterIP: None
  selector:
    app: influxdb
---
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: influxdb
  namespace: monitoring
spec:
  serviceName: influxdb
  replicas: 1
  selector:
    matchLabels:
      app: influxdb
  template:
    metadata:
      labels:
        app: influxdb
    spec:
      containers:
      - name: influxdb
        image: influxdb:2.7
        ports:
        - containerPort: 8086
          name: http
        env:
        - name: DOCKER_INFLUXDB_INIT_MODE
          value: "setup"
        - name: DOCKER_INFLUXDB_INIT_USERNAME
          valueFrom:
            secretKeyRef:
              name: influxdb-auth
              key: username
        - name: DOCKER_INFLUXDB_INIT_PASSWORD
          valueFrom:
            secretKeyRef:
              name: influxdb-auth
              key: password
        - name: DOCKER_INFLUXDB_INIT_ADMIN_TOKEN
          valueFrom:
            secretKeyRef:
              name: influxdb-auth
              key: token
        - name: DOCKER_INFLUXDB_INIT_ORG
          value: "my-org"
        - name: DOCKER_INFLUXDB_INIT_BUCKET
          value: "my-bucket"
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
          limits:
            memory: "8Gi"
            cpu: "4"
        volumeMounts:
        - name: influxdb-storage
          mountPath: /var/lib/influxdb2
        livenessProbe:
          httpGet:
            path: /health
            port: 8086
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
        readinessProbe:
          httpGet:
            path: /health
            port: 8086
          initialDelaySeconds: 5
          periodSeconds: 5
          timeoutSeconds: 3
      securityContext:
        fsGroup: 1500
        runAsUser: 1500
        runAsNonRoot: true
  volumeClaimTemplates:
  - metadata:
      name: influxdb-storage
    spec:
      accessModes: [ "ReadWriteOnce" ]
      storageClassName: fast-ssd
      resources:
        requests:
          storage: 100Gi
```

### Secrets Management

```yaml
# influxdb-secrets.yaml
apiVersion: v1
kind: Secret
metadata:
  name: influxdb-auth
  namespace: monitoring
type: Opaque
stringData:
  username: admin
  password: change-me-in-production
  token: my-super-secret-token-change-me
---
# Using sealed-secrets or external-secrets (recommended)
apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata:
  name: influxdb-auth
  namespace: monitoring
spec:
  refreshInterval: 1h
  secretStoreRef:
    name: vault-backend
    kind: SecretStore
  target:
    name: influxdb-auth
    creationPolicy: Owner
  data:
  - secretKey: username
    remoteRef:
      key: secret/influxdb
      property: username
  - secretKey: password
    remoteRef:
      key: secret/influxdb
      property: password
  - secretKey: token
    remoteRef:
      key: secret/influxdb
      property: token
```

### Service and Ingress

```yaml
# influxdb-service.yaml
apiVersion: v1
kind: Service
metadata:
  name: influxdb-external
  namespace: monitoring
spec:
  type: LoadBalancer
  ports:
  - port: 8086
    targetPort: 8086
    protocol: TCP
  selector:
    app: influxdb
---
# influxdb-ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: influxdb-ingress
  namespace: monitoring
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/backend-protocol: "HTTP"
    nginx.ingress.kubernetes.io/proxy-body-size: "10m"
spec:
  ingressClassName: nginx
  tls:
  - hosts:
    - influxdb.example.com
    secretName: influxdb-tls
  rules:
  - host: influxdb.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: influxdb-external
            port:
              number: 8086
```

### Backup CronJob

```yaml
# influxdb-backup-cronjob.yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: influxdb-backup
  namespace: monitoring
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
            image: influxdb:2.7
            command:
            - /bin/sh
            - -c
            - |
              BACKUP_PATH="/backups/$(date +%Y%m%d-%H%M%S)"
              influx backup "$BACKUP_PATH" \
                --host http://influxdb:8086 \
                --token "$INFLUX_TOKEN"
              tar czf "$BACKUP_PATH.tar.gz" -C /backups "$(basename $BACKUP_PATH)"
              rm -rf "$BACKUP_PATH"
              # Upload to S3 (requires aws-cli)
              aws s3 cp "$BACKUP_PATH.tar.gz" s3://my-backups/influxdb/
            env:
            - name: INFLUX_TOKEN
              valueFrom:
                secretKeyRef:
                  name: influxdb-auth
                  key: token
            volumeMounts:
            - name: backup-storage
              mountPath: /backups
          restartPolicy: OnFailure
          volumes:
          - name: backup-storage
            persistentVolumeClaim:
              claimName: influxdb-backup-pvc
```

### Resource Quotas and Limits

```yaml
# influxdb-resource-quota.yaml
apiVersion: v1
kind: ResourceQuota
metadata:
  name: influxdb-quota
  namespace: monitoring
spec:
  hard:
    requests.cpu: "8"
    requests.memory: 16Gi
    limits.cpu: "16"
    limits.memory: 32Gi
    persistentvolumeclaims: "5"
---
# influxdb-limit-range.yaml
apiVersion: v1
kind: LimitRange
metadata:
  name: influxdb-limits
  namespace: monitoring
spec:
  limits:
  - max:
      cpu: "8"
      memory: 16Gi
    min:
      cpu: "1"
      memory: 2Gi
    type: Container
```

### Kubernetes Best Practices

```yaml
High_Availability:
  - Use StatefulSets for stable network identity
  - Deploy across multiple availability zones
  - Use pod anti-affinity rules
  - Implement proper health checks

Storage:
  - Use SSD-backed storage classes
  - Enable volume snapshots
  - Set appropriate PVC size (100Gi+ recommended)
  - Use ReadWriteOnce access mode

Security:
  - Store credentials in Secrets (or external secret manager)
  - Enable RBAC
  - Use NetworkPolicies
  - Run as non-root user (1500)
  - Enable pod security standards

Monitoring:
  - Enable Prometheus metrics endpoint
  - Use ServiceMonitor for Prometheus scraping
  - Implement alerts for critical metrics
  - Monitor PVC usage

Scaling:
  - Vertical scaling: Increase resources (CPU, memory)
  - Horizontal scaling: Use InfluxDB 3 Enterprise/Clustered
  - Consider read replicas for query workload
```

## 19. Performance Tuning

### InfluxDB 3.x Performance Configuration

```bash
# InfluxDB 3 Core/Enterprise performance settings
influxdb3 serve \
  --num-io-threads=12 \
  --datafusion-num-threads=20 \
  --exec-mem-pool-bytes=80% \
  --wal-flush-interval=100ms \
  --parquet-mem-cache-size=4GB \
  --parquet-mem-cache-prune-interval=1m \
  --parquet-mem-cache-prune-percentage=20 \
  --wal-snapshot-size=100MB \
  --force-snapshot-mem-threshold=80%
```

### Thread Allocation

```yaml
# 32-core system example
IO_Threads:
  Default: 2 (often insufficient)
  High_Ingest: 8-16+
  Purpose: HTTP requests, line protocol parsing
  Setting: --num-io-threads=12

DataFusion_Threads:
  Default: Number of CPU cores
  Recommended: 60-70% of cores for query-heavy
  Purpose: Query execution
  Setting: --datafusion-num-threads=20

Example_Configurations:
  Write_Heavy_32_Core:
    io_threads: 16
    datafusion_threads: 16

  Read_Heavy_32_Core:
    io_threads: 8
    datafusion_threads: 24

  Balanced_32_Core:
    io_threads: 12
    datafusion_threads: 20
```

### Cache Configuration

```yaml
# InfluxDB 3.x Parquet cache
Parquet_Cache:
  Size: --parquet-mem-cache-size=4GB
  Prune_Interval: --parquet-mem-cache-prune-interval=1m
  Prune_Percentage: --parquet-mem-cache-prune-percentage=20

  Sizing_Guidelines:
    Small_Deployment: 1-2GB
    Medium_Deployment: 4-8GB
    Large_Deployment: 16GB+

# InfluxDB 2.x TSM cache
TSM_Cache:
  Max_Size: cache-max-memory-size=1g
  Snapshot_Size: cache-snapshot-memory-size=25m
  Snapshot_Interval: cache-snapshot-write-cold-duration=10m
```

### WAL Configuration

```yaml
# InfluxDB 3.x WAL settings
WAL_Flush_Interval:
  Default: 1s
  Low_Latency: 100ms (local disk recommended)
  High_Throughput: 1-5s
  Setting: --wal-flush-interval=100ms

WAL_Snapshot_Size:
  Default: 100MB
  Range: 50MB-500MB
  Setting: --wal-snapshot-size=100MB

Memory_Snapshot_Threshold:
  Default: 80%
  Setting: --force-snapshot-mem-threshold=80%
```

### Compaction Tuning (InfluxDB 2.x)

```toml
# influxdb.conf - TSM compaction settings
[data]
  # Throughput limits
  compact-throughput = "96m"  # 96 MB/s
  compact-throughput-burst = "96m"

  # Concurrency
  max-concurrent-compactions = 4  # Or 0 for 50% of cores

  # Timing
  compact-full-write-cold-duration = "2h"

  # File sizes
  max-tsm-file-size = "2147483648"  # 2GB
```

### Query Optimization

```yaml
# Query memory limits
InfluxDB_3:
  Setting: --exec-mem-pool-bytes=80%
  Default: 70% of system memory

InfluxDB_2:
  Max_Memory: Not directly configurable
  Use: Query timeouts and point limits

# Query performance settings
Query_Timeout:
  Setting: query-timeout=0s (unlimited, not recommended)
  Recommended: 60s-300s for production

Max_Select_Point:
  Setting: max-select-point=0 (unlimited)
  Recommended: 100000-1000000

Max_Concurrent_Queries:
  Setting: max-concurrent-queries=0 (unlimited)
  Recommended: 10-50 based on resources
```

### Memory Pool Configuration

```yaml
# InfluxDB 3.x memory management
Execution_Memory_Pool:
  Percentage: --exec-mem-pool-bytes=80%
  Absolute: --exec-mem-pool-bytes=32GB

  Guidelines:
    - Leave 20-30% for OS and other processes
    - Monitor OOM kills
    - Adjust based on query patterns

System_Memory_Planning:
  16GB_System: exec-mem-pool-bytes=12GB (75%)
  32GB_System: exec-mem-pool-bytes=24GB (75%)
  64GB_System: exec-mem-pool-bytes=48GB (75%)
  128GB_System: exec-mem-pool-bytes=96GB (75%)
```

### Storage Performance

```yaml
Disk_Recommendations:
  Type: SSD (NVMe preferred)
  IOPS: 3000+ for production
  Throughput: 500+ MB/s

Mount_Options:
  - noatime (reduce write overhead)
  - nodiratime
  - discard (for SSD TRIM)

File_System:
  Recommended: ext4, xfs
  Avoid: NFS (high latency)
```

### Network Performance

```yaml
Network_Tuning:
  TCP_Buffer_Size:
    - Increase for high-throughput
    - sysctl net.core.rmem_max=134217728
    - sysctl net.core.wmem_max=134217728

  Connection_Pooling:
    - Use client connection pools
    - Reuse connections
    - Keep-alive connections

  Compression:
    - Enable gzip for writes (5x improvement)
    - Trade CPU for network bandwidth
```

### Performance Benchmarking

```bash
# InfluxDB benchmarking tool
# Install influx-stress or use custom scripts

# Write performance test
influx-stress insert \
  --batchsize 10000 \
  --factor 10 \
  --host http://localhost:8086 \
  --token $INFLUX_TOKEN \
  --bucket test

# Monitor performance during test
watch -n 1 'curl -s http://localhost:8086/metrics | grep influx'
```

### Production Performance Checklist

```yaml
Hardware:
  ✓ SSD storage (NVMe preferred)
  ✓ Sufficient RAM (8GB minimum, 32GB+ recommended)
  ✓ Multi-core CPU (8+ cores recommended)
  ✓ 1Gbps+ network

Configuration:
  ✓ Tune IO threads for workload
  ✓ Optimize cache sizes
  ✓ Set appropriate WAL flush interval
  ✓ Configure compaction concurrency
  ✓ Enable TSI (InfluxDB 2.x)

Schema:
  ✓ Minimize series cardinality
  ✓ Use appropriate tag/field selection
  ✓ Implement downsampling
  ✓ Set retention policies

Operations:
  ✓ Monitor performance metrics
  ✓ Regular backups
  ✓ Capacity planning
  ✓ Query optimization
```

## 20. Low Latency Configuration

### Optimal Low-Latency Setup (InfluxDB 3.x)

```bash
# 32-core system optimized for low latency
influxdb3 serve \
  --num-io-threads=16 \
  --datafusion-num-threads=16 \
  --exec-mem-pool-bytes=80% \
  --wal-flush-interval=100ms \
  --parquet-mem-cache-size=8GB \
  --parquet-mem-cache-prune-interval=30s \
  --force-snapshot-mem-threshold=75%
```

### Low Latency Configuration Matrix

```yaml
# InfluxDB 3.x Low Latency Settings
Critical_Settings:
  WAL_Flush_Interval:
    Ultra_Low_Latency: 50ms (local SSD required)
    Low_Latency: 100ms (recommended)
    Standard: 1000ms (default)

  IO_Threads:
    Low_Latency: 12-16 (high parallelism)
    Standard: 4-8

  Cache_Size:
    Low_Latency: 8GB-16GB (reduce disk hits)
    Standard: 2GB-4GB

  Cache_Prune_Interval:
    Low_Latency: 30s (aggressive)
    Standard: 1m (default)

# Memory vs Latency Trade-off
Configuration_Profiles:
  Ultra_Low_Latency:
    wal_flush: 50ms
    io_threads: 16
    cache_size: 16GB
    memory_needed: 24GB+

  Low_Latency:
    wal_flush: 100ms
    io_threads: 12
    cache_size: 8GB
    memory_needed: 16GB+

  Balanced:
    wal_flush: 500ms
    io_threads: 8
    cache_size: 4GB
    memory_needed: 8GB+
```

### Write Latency Optimization

```yaml
Client_Side:
  Batch_Size:
    Too_Small: High network overhead, slow writes
    Too_Large: High memory usage, longer processing
    Optimal_InfluxDB_3: 10,000 points or 10MB
    Optimal_InfluxDB_2: 5,000 points

  Connection_Reuse:
    - Use persistent HTTP connections
    - Enable keep-alive
    - Use connection pooling

  Compression:
    - Enable gzip compression
    - Reduce network transfer time
    - Balance CPU vs network latency

  Tag_Sorting:
    - Sort tags lexicographically before writing
    - Improves series key lookups
    - Reduces processing time

Server_Side:
  WAL_Configuration:
    - Use local SSD (not network storage)
    - Reduce flush interval (100ms for low latency)
    - Ensure sufficient I/O throughput

  Threading:
    - Increase IO threads (12-16 for write-heavy)
    - Reduce thread contention
    - Scale with write concurrency
```

### Query Latency Optimization

```yaml
Schema_Design:
  - Use tags for frequently filtered columns
  - Limit field count per measurement
  - Avoid wide schemas (250+ columns)
  - Order tags by query frequency (InfluxDB 3)

Query_Patterns:
  - Always include time range filters
  - Use tag filters (indexed)
  - Avoid field filters when possible
  - Limit result set size

Caching:
  - Increase Parquet cache size
  - Use pre-aggregated data for dashboards
  - Implement application-level caching

Data_Organization:
  - Query downsampled data for large time ranges
  - Use appropriate retention policies
  - Partition data logically
```

### Network Latency Reduction

```yaml
Client_Configuration:
  Location:
    - Deploy clients close to InfluxDB server
    - Use same data center/region
    - Minimize network hops

  Protocol:
    - Use HTTP/2 when available
    - Enable compression
    - Use connection pooling
    - Reuse connections

Server_Configuration:
  Network_Stack:
    - Increase TCP buffer sizes
    - Enable TCP fast open
    - Tune kernel network parameters

  Load_Balancing:
    - Use local load balancers
    - Implement health checks
    - Session affinity for consistency
```

### Example Low-Latency Python Client

```python
from influxdb_client import InfluxDBClient, Point, WriteOptions
from influxdb_client.client.write_api import ASYNCHRONOUS
import gzip
import time

# Configure client for low latency
client = InfluxDBClient(
    url="http://localhost:8086",
    token="my-token",
    org="my-org",
    enable_gzip=True,  # Reduce network transfer time
    timeout=5000,      # 5 second timeout
)

# Asynchronous writes with optimal batching
write_api = client.write_api(write_options=WriteOptions(
    batch_size=10_000,        # Optimal for InfluxDB 3.x
    flush_interval=1_000,     # 1 second flush interval
    jitter_interval=0,        # No jitter for predictable latency
    retry_interval=1_000,     # Quick retry
    max_retries=3,
    exponential_base=2,
))

# Pre-sort tags for better performance
def create_point(measurement, tags, fields, timestamp):
    # Sort tags lexicographically
    sorted_tags = sorted(tags.items())

    point = Point(measurement)
    for tag_key, tag_value in sorted_tags:
        point.tag(tag_key, tag_value)
    for field_key, field_value in fields.items():
        point.field(field_key, field_value)
    point.time(timestamp)

    return point

# Write with timing
start = time.time()
points = []
for i in range(10000):
    point = create_point(
        "metrics",
        {"host": "server-01", "region": "us-east", "env": "prod"},
        {"value": i, "status": 1},
        time.time_ns()
    )
    points.append(point)

write_api.write(bucket="my-bucket", record=points)
write_api.flush()
latency = (time.time() - start) * 1000
print(f"Write latency: {latency:.2f}ms for 10k points")

client.close()
```

### Low Latency Monitoring

```flux
// Monitor write latency
from(bucket: "monitoring")
  |> range(start: -5m)
  |> filter(fn: (r) => r._measurement == "influxdb_write")
  |> filter(fn: (r) => r._field == "duration_us")
  |> map(fn: (r) => ({ r with _value: float(v: r._value) / 1000.0 }))  // Convert to ms
  |> aggregateWindow(every: 10s, fn: mean)
  |> yield(name: "write_latency_ms")

// Monitor query latency
from(bucket: "monitoring")
  |> range(start: -5m)
  |> filter(fn: (r) => r._measurement == "query")
  |> filter(fn: (r) => r._field == "duration_ms")
  |> aggregateWindow(every: 10s, fn: percentile, column: "_value", percentile: 0.95)
  |> yield(name: "p95_query_latency")
```

### Latency SLO Example

```yaml
# Service Level Objectives for latency
Write_Latency:
  Target_p50: < 10ms
  Target_p95: < 50ms
  Target_p99: < 100ms

Query_Latency:
  Simple_Query_p95: < 100ms
  Complex_Query_p95: < 500ms
  Dashboard_Query_p95: < 200ms

Configuration_for_SLO:
  - WAL flush interval: 100ms
  - IO threads: 12-16
  - Parquet cache: 8GB
  - SSD storage: Required
  - Network: < 1ms RTT
```

## 21. Migration Strategies

### Version Migration Paths (2026)

```yaml
Common_Migration_Paths:
  InfluxDB_1x_to_2x:
    Complexity: Medium
    Method: Automated upgrade or manual export/import
    Breaking_Changes: Yes (CQs → Tasks, HTTP API changes)

  InfluxDB_2x_to_3x_Core:
    Complexity: Medium-High
    Method: Export/import, no direct upgrade path
    Breaking_Changes: Yes (Flux maintenance mode)

  InfluxDB_1x_to_3x:
    Complexity: High
    Method: Multi-step migration or Historian tool
    Recommendation: Migrate to 2.x first, then to 3.x

  InfluxDB_TSM_to_3x_Clustered:
    Complexity: High
    Method: Data export/import or Historian
    Consideration: Schema restrictions (250 column limit)
```

### InfluxDB 1.x → 2.x Migration

#### Automated Upgrade

```bash
# Automatic upgrade (InfluxDB 2.0-2.7)
influxd upgrade \
  --config-file /etc/influxdb/influxdb.conf \
  --engine-path /var/lib/influxdb/data \
  --continuous-query-export-path /tmp/cqs.txt

# Process:
# 1. Reads 1.x config file
# 2. Creates 2.x config file
# 3. Exports continuous queries
# 4. Checks space availability
# 5. Migrates data and metadata
```

#### Manual Migration

```bash
# Step 1: Export data from InfluxDB 1.x
influx_inspect export \
  -database mydb \
  -retention autogen \
  -datadir /var/lib/influxdb/data \
  -waldir /var/lib/influxdb/wal \
  -out export.lp \
  -start 2024-01-01T00:00:00Z \
  -end 2024-12-31T23:59:59Z

# Compress export
gzip export.lp

# Step 2: Import to InfluxDB 2.x
influx write \
  --bucket mydb/autogen \
  --org my-org \
  --token my-token \
  --file export.lp.gz \
  --format lp \
  --compression gzip

# Step 3: Create DBRP mappings (for InfluxQL compatibility)
influx v1 dbrp create \
  --bucket-id <bucket-id> \
  --db mydb \
  --rp autogen \
  --org my-org \
  --token my-token \
  --default

# Step 4: Migrate continuous queries to tasks
# Convert CQs manually to Flux tasks (see section 8)
```

### InfluxDB 2.x → 3.x Migration

```bash
# No automated upgrade path available
# Use export/import approach

# Step 1: Export from InfluxDB 2.x
influx backup /tmp/backup \
  --host http://localhost:8086 \
  --token my-token

# Or export as line protocol
influx query \
  --org my-org \
  --token my-token \
  "from(bucket: \"my-bucket\")
    |> range(start: 2024-01-01T00:00:00Z)
    |> pivot(rowKey:[\"_time\"], columnKey: [\"_field\"], valueColumn: \"_value\")" \
  --raw > export.lp

# Step 2: Import to InfluxDB 3.x
# Use influx write or API to import data
```

### Migration Checklist

```yaml
Pre_Migration:
  Planning:
    ✓ Inventory databases, measurements, retention policies
    ✓ Document continuous queries and Kapacitor tasks
    ✓ Identify schema incompatibilities
    ✓ Estimate data volume and migration time
    ✓ Plan downtime window (if needed)

  Preparation:
    ✓ Backup source database (full backup)
    ✓ Test migration in non-production environment
    ✓ Prepare target InfluxDB instance
    ✓ Verify network connectivity and bandwidth
    ✓ Update client libraries and applications

  Compatibility_Checks:
    ✓ Check for duplicate tag/field names (v3 restriction)
    ✓ Verify measurement column count < 250 (v3 restriction)
    ✓ Review Flux usage (maintenance mode in v3)
    ✓ Identify high cardinality data (v3 handles better)

During_Migration:
  Execution:
    ✓ Stop or redirect writes to source
    ✓ Export data (by time range or DBRP)
    ✓ Import data to target
    ✓ Verify data integrity (count, samples)
    ✓ Migrate metadata (dashboards, alerts)

  Validation:
    ✓ Compare record counts
    ✓ Verify sample data points
    ✓ Test queries on target
    ✓ Validate authentication and authorization

Post_Migration:
  Cutover:
    ✓ Update application connection strings
    ✓ Redirect writes to new instance
    ✓ Monitor for errors and performance issues
    ✓ Keep source instance running (backup)

  Cleanup:
    ✓ Decommission source after validation period
    ✓ Update documentation
    ✓ Remove temporary export files
    ✓ Archive backups
```

### Zero-Downtime Migration Strategy

```yaml
# Dual-write approach for zero downtime
Phase_1_Setup:
  - Deploy new InfluxDB instance
  - Configure applications for dual-write
  - Verify writes to both instances

Phase_2_Backfill:
  - Export historical data from source
  - Import to target in batches
  - Verify data consistency

Phase_3_Validation:
  - Run read queries against both
  - Compare results
  - Monitor for discrepancies

Phase_4_Cutover:
  - Switch reads to new instance
  - Stop writes to old instance
  - Monitor performance and errors

Phase_5_Cleanup:
  - Remove dual-write logic
  - Decommission old instance
  - Archive data and backups
```

### Migration with Historian Tool

```yaml
# Historian: Migration tool for InfluxDB 3.0
Purpose:
  - Migrate from 1.x/2.x to 3.0
  - Re-import Parquet data
  - Cold-tier storage solution

Usage:
  - Export data to Parquet format
  - Store in object storage
  - Import to InfluxDB 3.0 using Historian

Benefit:
  - Preserves historical data
  - Efficient Parquet-to-Parquet migration
  - Handles large datasets
```

### Migration Performance Tips

```yaml
Optimization:
  Export:
    - Export by time ranges (parallel exports)
    - Use multiple DBRP combinations
    - Compress exports (gzip)
    - Export to fast storage (SSD)

  Import:
    - Import in large batches (10k-100k points)
    - Use multiple concurrent imports
    - Enable gzip compression
    - Monitor target instance resources

  Network:
    - Use same data center for source/target
    - Increase bandwidth allocation
    - Use compression for transfers

  Resources:
    - Allocate sufficient memory on target
    - Use fast storage (SSD/NVMe)
    - Monitor CPU and I/O utilization
```

### Common Migration Issues

```yaml
Problems_and_Solutions:
  Duplicate_Names:
    Problem: Tag and field with same name (v3 error)
    Solution: Rename fields before migration

  High_Column_Count:
    Problem: > 250 columns per measurement (v3 limit)
    Solution: Split into multiple measurements

  Flux_Dependencies:
    Problem: Heavy Flux usage (maintenance mode in v3)
    Solution: Migrate to InfluxQL or SQL

  Authentication:
    Problem: Different auth models (1.x vs 2.x vs 3.x)
    Solution: Update tokens/credentials after migration

  CQ_Migration:
    Problem: Continuous queries don't auto-convert
    Solution: Manually convert to Flux tasks or InfluxQL CQs

  Cardinality:
    Problem: High cardinality issues in TSM (1.x/2.x)
    Solution: Benefit from unlimited cardinality in v3
```

---

## Additional Resources

### Official Documentation

- [InfluxDB 3 Core Documentation](https://docs.influxdata.com/influxdb3/core/)
- [InfluxDB 2.x Documentation](https://docs.influxdata.com/influxdb/v2/)
- [InfluxDB 1.x Documentation](https://docs.influxdata.com/influxdb/v1/)

### Community Resources

- [InfluxData Community Forums](https://community.influxdata.com/)
- [InfluxDB GitHub Repository](https://github.com/influxdata/influxdb)
- [InfluxData Blog](https://www.influxdata.com/blog/)

### Related Guides

- See `docker-compose.md` for Docker orchestration patterns
- See `dockerfile_style.md` for container image best practices
- See `monitoring.md` for observability strategies

---

**Last Updated:** 2026-02-06
**InfluxDB Versions:** 3.8 (Core/Enterprise), 2.7 (OSS), 1.11 (Legacy)
**Target Audience:** DevOps Engineers, SREs, Platform Engineers

---

**End of InfluxDB Development Guidelines**
