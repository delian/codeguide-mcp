# DuckDB Best Practices Guide

**Version:** 1.0
**Last Updated:** February 2026
**Target Version:** DuckDB 1.0+, 1.1+

## Table of Contents

1. [Architecture and Fundamentals](#1-architecture-and-fundamentals)
2. [Performance Optimization](#2-performance-optimization)
3. [Data Import and Export](#3-data-import-and-export)
4. [Query Optimization](#4-query-optimization)
5. [Analytical Query Patterns](#5-analytical-query-patterns)
6. [Partitioning and Parallelism](#6-partitioning-and-parallelism)
7. [Extensions and Plugins](#7-extensions-and-plugins)
8. [Python Integration](#8-python-integration)
9. [R Integration](#9-r-integration)
10. [Pandas and Arrow Integration](#10-pandas-and-arrow-integration)
11. [Remote File Access](#11-remote-file-access)
12. [Schema Design for Analytics](#12-schema-design-for-analytics)
13. [Aggregations and Window Functions](#13-aggregations-and-window-functions)
14. [Data Types and Storage](#14-data-types-and-storage)
15. [Transaction and Concurrency](#15-transaction-and-concurrency)
16. [Use Cases and Limitations](#16-use-cases-and-limitations)
17. [Comparison with Other Databases](#17-comparison-with-other-databases)
18. [Deployment Patterns](#18-deployment-patterns)
19. [Testing and Benchmarking](#19-testing-and-benchmarking)
20. [Version-Specific Features](#20-version-specific-features)

---

## 1. Architecture and Fundamentals

### What is DuckDB?

DuckDB is an **in-process SQL OLAP database** designed for **analytical workloads**. Often called "**SQLite for analytics**", it provides:

- ✅ **Embedded/In-process** (no server, runs in your application)
- ✅ **Columnar storage** (optimized for analytics)
- ✅ **Vectorized execution** (process data in batches)
- ✅ **Parallel query execution** (multi-core utilization)
- ✅ **Zero external dependencies** (single binary/library)
- ✅ **ACID transactions** (full consistency guarantees)
- ✅ **Direct file querying** (Parquet, CSV, JSON without loading)

### Architecture Overview

```
┌─────────────────────────────────────────────┐
│         SQL Parser & Binder                 │
├─────────────────────────────────────────────┤
│         Query Optimizer                     │
│  - Join Reordering                          │
│  - Predicate Pushdown                       │
│  - Filter Pushdown to Parquet               │
├─────────────────────────────────────────────┤
│    Vectorized Execution Engine              │
│  - Process 2048 rows at a time              │
│  - SIMD optimizations                       │
│  - Multi-threaded execution                 │
├─────────────────────────────────────────────┤
│         Columnar Storage                    │
│  - Compression (dictionary, RLE, etc.)      │
│  - Statistics (min/max, null count)         │
├─────────────────────────────────────────────┤
│         File Formats                        │
│  - Native DuckDB format                     │
│  - Parquet (read/write)                     │
│  - CSV, JSON, Excel, etc.                   │
└─────────────────────────────────────────────┘
```

### Key Characteristics

**Columnar Storage:**
```
Traditional Row Storage (OLTP):
Row 1: [id=1, name='Alice', age=30, salary=50000]
Row 2: [id=2, name='Bob',   age=25, salary=45000]

DuckDB Column Storage (OLAP):
id:     [1, 2, 3, ...]
name:   ['Alice', 'Bob', 'Carol', ...]
age:    [30, 25, 28, ...]
salary: [50000, 45000, 52000, ...]

Benefits:
- Better compression (similar values together)
- Faster aggregations (read only needed columns)
- Efficient for analytical queries
```

**Vectorized Execution:**
```
Traditional: Process one row at a time
Vectorized: Process 2048 rows in a batch (vector)

SELECT SUM(salary) FROM employees WHERE age > 30;

Traditional: Loop through each row, check age, add salary
Vectorized:  Load 2048 ages → filter → load salaries → sum
             (Leverages CPU cache, SIMD instructions)
```

### When to Use DuckDB

**✅ Excellent For:**

1. **Data Analysis and Exploration:**
   - Jupyter notebooks / data science workflows
   - Ad-hoc analytical queries on CSV/Parquet
   - Data profiling and quality checks

2. **ETL and Data Processing:**
   - Transform data between formats (CSV → Parquet)
   - Data pipeline processing
   - Log analysis and aggregation

3. **Analytical Dashboards:**
   - Business intelligence queries
   - Reporting and analytics applications
   - Time-series analysis

4. **Embedded Analytics:**
   - Desktop applications with analytics
   - Mobile apps (SQLite replacement for analytics)
   - Command-line tools

5. **Data Lake Queries:**
   - Query Parquet files on S3/GCS directly
   - Federated queries across file formats
   - Data lakehouse pattern

6. **Testing and Development:**
   - Fast analytical query testing
   - Development without database setup
   - CI/CD pipelines

### When NOT to Use DuckDB

**❌ Not Recommended For:**

1. **High-Concurrency OLTP:**
   - Single-writer limitation (like SQLite)
   - Not designed for 1000s of concurrent writes
   - Use PostgreSQL, MySQL for OLTP

2. **Distributed Systems:**
   - No built-in clustering/replication
   - Single-node only
   - Use ClickHouse, Snowflake for distributed analytics

3. **Real-Time Streaming:**
   - Batch-oriented, not stream-oriented
   - Use Kafka, Flink for streaming

4. **Multi-User Concurrent Writes:**
   - Best for single-writer scenarios
   - Readers don't block writers (WAL mode)
   - Use traditional RDBMS for multi-user writes

5. **Very Large Datasets (>TB on single node):**
   - Limited by single-node resources
   - Use distributed systems (ClickHouse, BigQuery)

---

## 2. Performance Optimization

### Memory Configuration

**Set Memory Limit:**
```sql
-- Set maximum memory usage (default: 80% of system RAM)
SET memory_limit = '16GB';

-- Check current setting
SELECT current_setting('memory_limit');

-- Temporary table memory
SET temp_directory = '/fast/ssd/temp';
```

**Python Configuration:**
```python
import duckdb

# Create connection with memory limit
conn = duckdb.connect(':memory:')
conn.execute("SET memory_limit = '8GB'")

# Or configure at connection time
conn = duckdb.connect(
    database=':memory:',
    config={
        'memory_limit': '8GB',
        'threads': 8,
        'max_memory': '10GB'
    }
)
```

### Thread Configuration

**Parallel Execution:**
```sql
-- Set number of threads (default: number of CPU cores)
SET threads TO 8;

-- Check setting
SELECT current_setting('threads');

-- Disable parallelism (debugging)
SET threads TO 1;
```

**Python:**
```python
conn = duckdb.connect()
conn.execute("SET threads = 16")
```

### Query Optimization Settings

**Enable Query Profiling:**
```sql
-- Enable profiling
PRAGMA enable_profiling;

-- Set output format
PRAGMA profile_output = 'profile.json';

-- Run query
SELECT * FROM large_table WHERE condition;

-- View profile
PRAGMA disable_profiling;
```

**Optimizer Settings:**
```sql
-- Enable/disable various optimizations
SET enable_optimizer = true;
SET enable_filter_pushdown = true;
SET enable_projection_pushdown = true;

-- Force parallel execution
SET force_parallelism = true;

-- Preserve insertion order (default: false for performance)
SET preserve_insertion_order = false;
```

### Storage and Caching

**Buffer Manager:**
```sql
-- Set buffer pool size
SET buffer_pool_size = '4GB';

-- Checkpoint threshold
SET checkpoint_threshold = '1GB';
```

**Temporary Storage:**
```sql
-- Set temp directory for spilling
SET temp_directory = '/nvme/duckdb-temp';

-- Allow out-of-core processing
SET enable_external_access = true;
```

### File Format Optimizations

**Parquet Reading:**
```sql
-- Read only necessary columns (column pruning)
SELECT name, salary FROM 'employees.parquet';

-- Predicate pushdown to Parquet
SELECT * FROM 'data.parquet' WHERE year = 2024;
-- DuckDB pushes filter to Parquet reader (only reads matching row groups)

-- Parallel Parquet reading
SELECT * FROM 'data/**/*.parquet';  -- Reads multiple files in parallel
```

**CSV Reading:**
```sql
-- Optimized CSV reading with options
SELECT * FROM read_csv(
    'data.csv',
    header = true,
    parallel = true,           -- Parallel reading
    auto_detect = true,        -- Auto-detect types
    sample_size = 100000,      -- Sample size for type detection
    delim = ',',
    quote = '"'
);
```

### Best Practices for Performance

**1. Use Appropriate File Formats:**
```sql
-- SLOW: CSV (no compression, no stats, no column pruning)
SELECT AVG(price) FROM 'sales.csv';

-- FAST: Parquet (compressed, column pruning, stats)
SELECT AVG(price) FROM 'sales.parquet';
-- 10-100x faster for analytical queries
```

**2. Partition Large Datasets:**
```
-- Hive-style partitioning
data/
  year=2023/
    month=01/
      data.parquet
    month=02/
      data.parquet
  year=2024/
    month=01/
      data.parquet

-- Query automatically filters by partition
SELECT * FROM 'data/**/*.parquet' WHERE year = 2024 AND month = 1;
-- Only reads year=2024/month=01/ files
```

**3. Use Persistent Database for Repeated Queries:**
```python
# SLOW: Re-scan files every time
conn = duckdb.connect(':memory:')
conn.execute("SELECT * FROM 'large.parquet' WHERE condition")

# FAST: Import once, query many times
conn = duckdb.connect('analytics.duckdb')
conn.execute("CREATE TABLE data AS SELECT * FROM 'large.parquet'")
conn.execute("SELECT * FROM data WHERE condition")  # Much faster
```

**4. Use Appropriate Data Types:**
```sql
-- Use smallest appropriate type
CREATE TABLE events (
    id UBIGINT,                    -- Unsigned big int
    event_type TINYINT,            -- 0-255
    timestamp TIMESTAMP,
    amount DECIMAL(10,2),
    metadata JSON
);
```

---

## 3. Data Import and Export

### Parquet Files

**Reading Parquet:**
```sql
-- Read single Parquet file
SELECT * FROM 'data.parquet';

-- Read with alias
SELECT * FROM 'data.parquet' AS data;

-- Read multiple files (glob pattern)
SELECT * FROM 'data/*.parquet';

-- Read Hive-partitioned data
SELECT * FROM 'data/**/*.parquet';

-- Read specific files
SELECT * FROM 'data/part-*.parquet';

-- Read with schema inference
DESCRIBE SELECT * FROM 'data.parquet';
```

**Writing Parquet:**
```sql
-- Export query result to Parquet
COPY (SELECT * FROM table WHERE condition) TO 'output.parquet';

-- Write with compression
COPY (SELECT * FROM table) TO 'output.parquet' (COMPRESSION 'zstd');

-- Compression options: uncompressed, snappy, gzip, zstd (recommended)
COPY (SELECT * FROM table) TO 'output.parquet' (COMPRESSION 'zstd', COMPRESSION_LEVEL 9);

-- Write partitioned Parquet
COPY (SELECT * FROM sales)
TO 'output' (FORMAT PARQUET, PARTITION_BY (year, month));
-- Creates: output/year=2024/month=01/data.parquet
```

**Parquet Metadata:**
```sql
-- View Parquet file metadata
SELECT * FROM parquet_metadata('data.parquet');

-- View schema
SELECT * FROM parquet_schema('data.parquet');

-- View row group stats
SELECT * FROM parquet_file_metadata('data.parquet');
```

### CSV Files

**Reading CSV:**
```sql
-- Simple CSV read (auto-detect)
SELECT * FROM 'data.csv';

-- Read with options
SELECT * FROM read_csv('data.csv', header=true, delim=',', quote='"');

-- Auto-detect with large sample
SELECT * FROM read_csv('data.csv', auto_detect=true, sample_size=100000);

-- Specify schema explicitly
SELECT * FROM read_csv('data.csv',
    columns = {
        'id': 'BIGINT',
        'name': 'VARCHAR',
        'created_at': 'TIMESTAMP',
        'price': 'DECIMAL(10,2)'
    },
    header = true
);

-- Handle malformed CSV
SELECT * FROM read_csv('data.csv',
    ignore_errors = true,
    max_line_size = 1048576  -- 1MB per line
);

-- Parallel CSV reading
SELECT * FROM read_csv('data/*.csv', parallel=true);
```

**Writing CSV:**
```sql
-- Export to CSV
COPY (SELECT * FROM table) TO 'output.csv' (HEADER, DELIMITER ',');

-- CSV with custom options
COPY (SELECT * FROM table) TO 'output.csv' (
    HEADER true,
    DELIMITER '|',
    QUOTE '"',
    ESCAPE '\',
    NULL 'NULL'
);

-- Compressed CSV
COPY (SELECT * FROM table) TO 'output.csv.gz' (COMPRESSION gzip);
```

### JSON Files

**Reading JSON:**
```sql
-- Read JSON file
SELECT * FROM 'data.json';

-- Read NDJSON (newline-delimited JSON)
SELECT * FROM read_json('data.ndjson', format='newline_delimited');

-- Read with schema
SELECT * FROM read_json('data.json',
    columns = {
        'id': 'BIGINT',
        'data': 'JSON',
        'timestamp': 'TIMESTAMP'
    }
);

-- Read JSON with auto-detect
SELECT * FROM read_json_auto('data.json');

-- Extract nested JSON fields
SELECT
    data->>'$.user.name' as user_name,
    data->>'$.event.type' as event_type
FROM 'events.json';
```

**Writing JSON:**
```sql
-- Export to JSON
COPY (SELECT * FROM table) TO 'output.json';

-- NDJSON format
COPY (SELECT * FROM table) TO 'output.ndjson' (FORMAT JSON);
```

### Excel Files

**Reading Excel:**
```sql
-- Requires httpfs extension
INSTALL spatial;  -- For Excel support
LOAD spatial;

-- Read Excel file
SELECT * FROM st_read('data.xlsx');

-- Specific sheet
SELECT * FROM st_read('data.xlsx', layer='Sheet1');
```

### Other Formats

**SQLite:**
```sql
-- Attach SQLite database
ATTACH 'database.db' AS sqlite_db (TYPE SQLITE);

-- Query SQLite tables
SELECT * FROM sqlite_db.table_name;

-- Copy from SQLite to DuckDB
CREATE TABLE local_table AS SELECT * FROM sqlite_db.table_name;
```

**PostgreSQL:**
```sql
-- Attach PostgreSQL database
ATTACH 'dbname=mydb user=postgres host=localhost' AS postgres_db (TYPE POSTGRES);

-- Query PostgreSQL
SELECT * FROM postgres_db.table_name;

-- Export to PostgreSQL
COPY (SELECT * FROM local_table) TO postgres_db.remote_table;
```

### Bulk Insert

**INSERT FROM SELECT:**
```sql
-- Import CSV into table
CREATE TABLE events AS SELECT * FROM 'events.csv';

-- Append to existing table
INSERT INTO events SELECT * FROM 'new_events.csv';

-- Insert with transformation
INSERT INTO events
SELECT
    id,
    UPPER(name) as name,
    price * 1.1 as price_with_tax
FROM 'data.csv';
```

**Batch Inserts:**
```python
import duckdb

conn = duckdb.connect('analytics.duckdb')

# Efficient batch insert
data = [
    (1, 'Alice', 30),
    (2, 'Bob', 25),
    # ... thousands of rows
]

conn.executemany("INSERT INTO users VALUES (?, ?, ?)", data)

# Even better: Use DataFrame
import pandas as pd
df = pd.DataFrame(data, columns=['id', 'name', 'age'])
conn.execute("INSERT INTO users SELECT * FROM df")
```

---

## 4. Query Optimization

### EXPLAIN and Query Analysis

**EXPLAIN Statement:**
```sql
-- View query plan
EXPLAIN SELECT * FROM sales WHERE year = 2024;

-- Detailed query plan
EXPLAIN ANALYZE SELECT * FROM sales WHERE year = 2024;

-- JSON output
EXPLAIN (FORMAT JSON) SELECT * FROM sales WHERE year = 2024;
```

**Query Profiling:**
```sql
-- Enable profiling
PRAGMA enable_profiling;

-- Run query
SELECT
    category,
    SUM(sales) as total_sales
FROM products
GROUP BY category;

-- Get profile
PRAGMA profile_output;

-- Disable profiling
PRAGMA disable_profiling;
```

**Python Profiling:**
```python
import duckdb

conn = duckdb.connect()
conn.execute("PRAGMA enable_profiling")

# Run query
result = conn.execute("""
    SELECT category, SUM(sales)
    FROM 'products.parquet'
    GROUP BY category
""").fetchall()

# Get profile as DataFrame
profile = conn.execute("PRAGMA last_profiling_output").df()
print(profile)
```

### Filter Pushdown

**Parquet Filter Pushdown:**
```sql
-- Filters pushed to Parquet reader (only reads matching row groups)
SELECT * FROM 'sales.parquet'
WHERE year = 2024 AND month = 1;

-- DuckDB reads Parquet statistics and skips row groups
-- Much faster than reading entire file and filtering
```

**Projection Pushdown:**
```sql
-- Only reads 'name' and 'salary' columns from Parquet
SELECT name, salary FROM 'employees.parquet';

-- Columnar format makes this extremely efficient
```

### Join Optimization

**Join Types:**
```sql
-- Hash join (default for most queries)
SELECT *
FROM orders o
JOIN customers c ON o.customer_id = c.id;

-- Force hash join
SELECT /*+ HASH_JOIN */ *
FROM orders o
JOIN customers c ON o.customer_id = c.id;

-- Merge join (for sorted data)
SELECT /*+ MERGE_JOIN */ *
FROM orders o
JOIN customers c ON o.customer_id = c.id;
```

**Join Order:**
```sql
-- DuckDB automatically optimizes join order
-- Join smaller tables first

-- Explicit join order (if needed)
SELECT STRAIGHT_JOIN *
FROM large_table
JOIN small_table ON ...;
```

### Aggregation Optimization

**Grouped Aggregations:**
```sql
-- Efficient aggregation (vectorized)
SELECT
    category,
    COUNT(*) as count,
    AVG(price) as avg_price,
    SUM(quantity) as total_quantity
FROM products
GROUP BY category;

-- Multiple aggregation functions in single pass
SELECT
    category,
    COUNT(*) as count,
    MIN(price) as min_price,
    MAX(price) as max_price,
    AVG(price) as avg_price,
    STDDEV(price) as stddev_price
FROM products
GROUP BY category;
```

**Approximate Aggregations:**
```sql
-- Approximate distinct count (faster for large datasets)
SELECT APPROX_COUNT_DISTINCT(user_id) FROM events;

-- Exact count (slower)
SELECT COUNT(DISTINCT user_id) FROM events;

-- Approximate quantiles
SELECT APPROX_QUANTILE(price, 0.5) as median FROM products;
SELECT APPROX_QUANTILE(price, [0.25, 0.5, 0.75]) as quartiles FROM products;
```

### Sampling

**Fast Data Sampling:**
```sql
-- Sample 10% of rows (fast, uses reservoir sampling)
SELECT * FROM large_table USING SAMPLE 10%;

-- Sample exact number of rows
SELECT * FROM large_table USING SAMPLE 1000 ROWS;

-- Bernoulli sampling (row-level)
SELECT * FROM large_table USING SAMPLE 5% (BERNOULLI);

-- System sampling (block-level, faster)
SELECT * FROM large_table USING SAMPLE 5% (SYSTEM);
```

### Common Table Expressions (CTEs)

**Using CTEs:**
```sql
-- CTEs for query organization
WITH monthly_sales AS (
    SELECT
        strftime(order_date, '%Y-%m') as month,
        SUM(total) as sales
    FROM orders
    GROUP BY month
),
sales_growth AS (
    SELECT
        month,
        sales,
        LAG(sales) OVER (ORDER BY month) as prev_month_sales,
        sales - LAG(sales) OVER (ORDER BY month) as growth
    FROM monthly_sales
)
SELECT * FROM sales_growth ORDER BY month;
```

**Recursive CTEs:**
```sql
-- Generate date series
WITH RECURSIVE dates(date) AS (
    SELECT DATE '2024-01-01'
    UNION ALL
    SELECT date + INTERVAL 1 DAY
    FROM dates
    WHERE date < DATE '2024-12-31'
)
SELECT * FROM dates;
```

### Materialized CTEs

**Optimize with Materialization:**
```sql
-- DuckDB automatically materializes CTEs when beneficial
-- No explicit syntax needed (unlike PostgreSQL)

WITH expensive_computation AS (
    SELECT
        user_id,
        COUNT(*) as event_count,
        AVG(value) as avg_value
    FROM large_events_table
    GROUP BY user_id
)
SELECT * FROM expensive_computation WHERE event_count > 100
UNION ALL
SELECT * FROM expensive_computation WHERE avg_value > 50;

-- 'expensive_computation' computed once, used twice
```

---

## 5. Analytical Query Patterns

### Time-Series Analysis

**Date/Time Functions:**
```sql
-- Extract date parts
SELECT
    EXTRACT(YEAR FROM timestamp) as year,
    EXTRACT(MONTH FROM timestamp) as month,
    EXTRACT(DAY FROM timestamp) as day,
    EXTRACT(HOUR FROM timestamp) as hour
FROM events;

-- Date formatting
SELECT
    strftime(timestamp, '%Y-%m-%d') as date,
    strftime(timestamp, '%Y-%m') as year_month,
    strftime(timestamp, '%H:%M:%S') as time
FROM events;

-- Date arithmetic
SELECT
    timestamp,
    timestamp + INTERVAL 1 DAY as tomorrow,
    timestamp - INTERVAL 1 HOUR as hour_ago
FROM events;

-- Generate time series
SELECT * FROM generate_series(
    DATE '2024-01-01',
    DATE '2024-12-31',
    INTERVAL 1 DAY
) AS t(date);
```

**Time-Based Aggregations:**
```sql
-- Daily aggregations
SELECT
    DATE_TRUNC('day', timestamp) as day,
    COUNT(*) as event_count,
    SUM(value) as total_value
FROM events
GROUP BY day
ORDER BY day;

-- Hourly aggregations
SELECT
    DATE_TRUNC('hour', timestamp) as hour,
    COUNT(*) as requests
FROM logs
GROUP BY hour
ORDER BY hour;

-- Weekly aggregations
SELECT
    DATE_TRUNC('week', order_date) as week,
    SUM(total) as weekly_sales
FROM orders
GROUP BY week
ORDER BY week;
```

**Moving Averages:**
```sql
-- 7-day moving average
SELECT
    date,
    value,
    AVG(value) OVER (
        ORDER BY date
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ) as moving_avg_7d
FROM daily_metrics
ORDER BY date;

-- 30-day moving sum
SELECT
    date,
    sales,
    SUM(sales) OVER (
        ORDER BY date
        ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
    ) as rolling_30d_sales
FROM daily_sales;
```

### Cohort Analysis

**User Cohorts:**
```sql
-- Cohort analysis by registration month
WITH user_cohorts AS (
    SELECT
        user_id,
        DATE_TRUNC('month', registration_date) as cohort_month
    FROM users
),
user_activity AS (
    SELECT
        u.user_id,
        u.cohort_month,
        DATE_TRUNC('month', e.event_date) as activity_month,
        COUNT(*) as events
    FROM user_cohorts u
    JOIN events e ON u.user_id = e.user_id
    GROUP BY u.user_id, u.cohort_month, activity_month
)
SELECT
    cohort_month,
    activity_month,
    COUNT(DISTINCT user_id) as active_users,
    SUM(events) as total_events
FROM user_activity
GROUP BY cohort_month, activity_month
ORDER BY cohort_month, activity_month;
```

**Retention Analysis:**
```sql
-- Monthly retention
WITH first_activity AS (
    SELECT user_id, MIN(DATE_TRUNC('month', event_date)) as first_month
    FROM events
    GROUP BY user_id
),
monthly_activity AS (
    SELECT DISTINCT
        user_id,
        DATE_TRUNC('month', event_date) as activity_month
    FROM events
)
SELECT
    f.first_month as cohort,
    DATE_DIFF('month', f.first_month, m.activity_month) as months_since_first,
    COUNT(DISTINCT m.user_id) as retained_users
FROM first_activity f
JOIN monthly_activity m ON f.user_id = m.user_id
GROUP BY cohort, months_since_first
ORDER BY cohort, months_since_first;
```

### Funnel Analysis

**Conversion Funnels:**
```sql
-- E-commerce funnel
WITH funnel_steps AS (
    SELECT user_id, event_type, timestamp
    FROM events
    WHERE event_type IN ('page_view', 'add_to_cart', 'checkout', 'purchase')
),
user_funnel AS (
    SELECT
        user_id,
        MAX(CASE WHEN event_type = 'page_view' THEN 1 ELSE 0 END) as viewed,
        MAX(CASE WHEN event_type = 'add_to_cart' THEN 1 ELSE 0 END) as added_to_cart,
        MAX(CASE WHEN event_type = 'checkout' THEN 1 ELSE 0 END) as checked_out,
        MAX(CASE WHEN event_type = 'purchase' THEN 1 ELSE 0 END) as purchased
    FROM funnel_steps
    GROUP BY user_id
)
SELECT
    SUM(viewed) as step1_viewers,
    SUM(added_to_cart) as step2_add_to_cart,
    SUM(checked_out) as step3_checkout,
    SUM(purchased) as step4_purchase,
    SUM(added_to_cart) * 100.0 / SUM(viewed) as conversion_to_cart_pct,
    SUM(purchased) * 100.0 / SUM(viewed) as overall_conversion_pct
FROM user_funnel;
```

### Pivot and Unpivot

**Pivot Tables:**
```sql
-- Pivot (wide format)
PIVOT (
    SELECT category, year, SUM(sales) as total_sales
    FROM sales_data
    GROUP BY category, year
)
ON year
USING SUM(total_sales);

-- Result:
-- category | 2022 | 2023 | 2024
-- ---------|------|------|------
-- Books    | 100  | 120  | 140
-- Electronics | 200 | 250 | 300
```

**Unpivot (Long Format):**
```sql
-- Unpivot
UNPIVOT (
    SELECT * FROM wide_table
)
ON column1, column2, column3
INTO
    NAME metric
    VALUE value;
```

### Top-N Queries

**Top 10 by Category:**
```sql
-- Top 10 products per category
SELECT *
FROM (
    SELECT
        *,
        ROW_NUMBER() OVER (
            PARTITION BY category
            ORDER BY sales DESC
        ) as rank
    FROM products
)
WHERE rank <= 10;
```

**Dense Rank:**
```sql
-- Ranking with ties
SELECT
    name,
    sales,
    DENSE_RANK() OVER (ORDER BY sales DESC) as rank
FROM products
ORDER BY rank;
```

---

## 6. Partitioning and Parallelism

### File-Based Partitioning

**Hive-Style Partitioning:**
```
Directory structure:
data/
  year=2023/
    month=01/
      data.parquet
    month=02/
      data.parquet
  year=2024/
    month=01/
      data.parquet
    month=02/
      data.parquet
```

**Query Partitioned Data:**
```sql
-- Automatically filters by partition
SELECT * FROM 'data/**/*.parquet'
WHERE year = 2024 AND month = 1;

-- Only reads: data/year=2024/month=01/data.parquet
-- Skips other partitions entirely (partition pruning)
```

**Write Partitioned Data:**
```sql
-- Write Hive-partitioned Parquet
COPY (SELECT * FROM sales)
TO 'output_data' (
    FORMAT PARQUET,
    PARTITION_BY (year, month),
    COMPRESSION 'zstd'
);

-- Creates:
-- output_data/year=2023/month=12/data_0.parquet
-- output_data/year=2024/month=01/data_0.parquet
-- etc.
```

### Parallel Query Execution

**Configure Parallelism:**
```sql
-- Set number of threads
SET threads = 16;

-- Check parallelism
EXPLAIN SELECT COUNT(*) FROM large_table;
-- Look for PARALLEL_SCAN in query plan
```

**Multi-File Parallelism:**
```sql
-- Reads multiple files in parallel
SELECT * FROM 'data/*.parquet';

-- Each file read by separate thread
-- Scales linearly with number of files
```

### Parallel Aggregations

**Group By Parallelism:**
```sql
-- Parallel hash aggregation
SELECT
    category,
    COUNT(*) as count,
    SUM(sales) as total_sales
FROM products
GROUP BY category;

-- DuckDB parallelizes:
-- 1. Scan (multiple threads read data)
-- 2. Local aggregation (each thread aggregates subset)
-- 3. Global aggregation (combine local results)
```

### Join Parallelism

**Parallel Hash Joins:**
```sql
-- Parallel hash join
SELECT *
FROM large_table l
JOIN small_table s ON l.id = s.id;

-- DuckDB:
-- 1. Builds hash table on small_table (parallel)
-- 2. Probes with large_table (parallel)
-- 3. Merges results
```

### Controlling Parallelism

**Force Serial Execution:**
```sql
-- Disable parallelism (debugging)
SET threads = 1;

-- Run query
SELECT * FROM data;

-- Re-enable
SET threads = 8;
```

**Force Parallel:**
```sql
-- Force parallel execution
SET force_parallelism = true;

-- Useful for small datasets in testing
```

---

## 7. Extensions and Plugins

### Core Extensions

**List Available Extensions:**
```sql
-- Show installed extensions
SELECT * FROM duckdb_extensions();

-- Show available extensions
SELECT extension_name, loaded, installed
FROM duckdb_extensions()
ORDER BY extension_name;
```

### httpfs Extension (Remote Files)

**Install and Load:**
```sql
INSTALL httpfs;
LOAD httpfs;

-- Set S3 credentials
SET s3_region='us-east-1';
SET s3_access_key_id='YOUR_ACCESS_KEY';
SET s3_secret_access_key='YOUR_SECRET_KEY';

-- Query S3 directly
SELECT * FROM 's3://my-bucket/data.parquet';

-- HTTPS files
SELECT * FROM 'https://example.com/data.csv';
```

**S3 Configuration:**
```python
import duckdb

conn = duckdb.connect()
conn.install_extension("httpfs")
conn.load_extension("httpfs")

# Configure S3
conn.execute("SET s3_region='us-west-2'")
conn.execute("SET s3_access_key_id='...'")
conn.execute("SET s3_secret_access_key='...'")

# Query S3
df = conn.execute("SELECT * FROM 's3://bucket/data.parquet'").df()
```

### Parquet Extension (Built-in)

**Parquet Functions:**
```sql
-- Read Parquet
SELECT * FROM 'data.parquet';

-- Write Parquet
COPY (SELECT * FROM table) TO 'output.parquet';

-- Parquet metadata
SELECT * FROM parquet_metadata('data.parquet');

-- Parquet schema
SELECT * FROM parquet_schema('data.parquet');
```

### JSON Extension

**JSON Functions:**
```sql
INSTALL json;
LOAD json;

-- Parse JSON
SELECT * FROM read_json('data.json');

-- JSON extraction
SELECT
    data->>'$.user.name' as name,
    data->>'$.user.email' as email
FROM json_table;

-- JSON array
SELECT json_extract(data, '$.items') FROM json_table;
```

### ICU Extension (Internationalization)

**Text Processing:**
```sql
INSTALL icu;
LOAD icu;

-- Case-insensitive comparison (locale-aware)
SELECT * FROM users
WHERE LOWER(name) = 'müller';

-- Collation
SELECT * FROM products
ORDER BY name COLLATE de_DE;
```

### Excel Extension

**Read Excel Files:**
```sql
INSTALL spatial;  -- Includes Excel support
LOAD spatial;

-- Read Excel
SELECT * FROM st_read('data.xlsx');

-- Specific sheet
SELECT * FROM st_read('data.xlsx', layer='Sheet1');
```

### PostgreSQL Extension

**PostgreSQL Scanner:**
```sql
INSTALL postgres_scanner;
LOAD postgres_scanner;

-- Attach PostgreSQL database
ATTACH 'dbname=mydb user=postgres host=localhost port=5432' AS pg (TYPE POSTGRES);

-- Query PostgreSQL tables
SELECT * FROM pg.users;

-- Join DuckDB and PostgreSQL
SELECT
    d.order_id,
    p.user_name
FROM local_orders d
JOIN pg.users p ON d.user_id = p.id;
```

### MySQL Extension

**MySQL Scanner:**
```sql
INSTALL mysql_scanner;
LOAD mysql_scanner;

-- Attach MySQL database
ATTACH 'host=localhost user=root port=3306 database=mydb' AS mysql_db (TYPE MYSQL);

-- Query MySQL
SELECT * FROM mysql_db.products;
```

### SQLite Extension

**SQLite Scanner:**
```sql
INSTALL sqlite_scanner;
LOAD sqlite_scanner;

-- Attach SQLite database
ATTACH 'data.db' AS sqlite_db (TYPE SQLITE);

-- Query SQLite
SELECT * FROM sqlite_db.table_name;

-- Migrate from SQLite to DuckDB
CREATE TABLE local_copy AS SELECT * FROM sqlite_db.table_name;
```

### FTS (Full-Text Search) Extension

**Full-Text Search:**
```sql
INSTALL fts;
LOAD fts;

-- Create FTS index
PRAGMA create_fts_index('documents', 'id', 'title', 'content');

-- Search
SELECT * FROM (
    SELECT * FROM documents
) WHERE fts_main_documents.match_bm25('title', 'search query') > 0;
```

### Custom Extensions

**Build Custom Extension:**
```cpp
// C++ extension example
// See: https://duckdb.org/docs/extensions/overview
```

---

## 8. Python Integration

### Installation

**Install DuckDB:**
```bash
pip install duckdb

# With specific version
pip install duckdb==1.0.0

# Development version
pip install duckdb --pre
```

### Basic Usage

**Connect and Query:**
```python
import duckdb

# In-memory database
conn = duckdb.connect(':memory:')

# Persistent database
conn = duckdb.connect('analytics.duckdb')

# Execute query
result = conn.execute("SELECT 42 as answer").fetchall()
print(result)  # [(42,)]

# Fetch as DataFrame
df = conn.execute("SELECT * FROM 'data.parquet'").df()

# Fetch as Arrow
arrow_table = conn.execute("SELECT * FROM 'data.parquet'").arrow()

# Fetch as NumPy
numpy_result = conn.execute("SELECT * FROM data").fetchnumpy()

# Close connection
conn.close()
```

### Query from Pandas DataFrame

**Direct DataFrame Queries:**
```python
import pandas as pd
import duckdb

# Create DataFrame
df = pd.DataFrame({
    'id': [1, 2, 3],
    'name': ['Alice', 'Bob', 'Carol'],
    'age': [30, 25, 35]
})

# Query DataFrame directly (no registration needed!)
result = duckdb.query("SELECT * FROM df WHERE age > 25").df()
print(result)

# More complex query
result = duckdb.query("""
    SELECT
        name,
        age,
        age - AVG(age) OVER () as age_diff_from_avg
    FROM df
    ORDER BY age DESC
""").df()
```

**Register DataFrame:**
```python
# Register DataFrame explicitly
conn = duckdb.connect()
conn.register('my_table', df)

# Query registered table
result = conn.execute("SELECT * FROM my_table").df()

# Unregister
conn.unregister('my_table')
```

### Integration with Polars

**Polars DataFrames:**
```python
import polars as pl
import duckdb

# Create Polars DataFrame
df_polars = pl.DataFrame({
    'id': [1, 2, 3],
    'value': [10, 20, 30]
})

# Query Polars DataFrame
result = duckdb.query("SELECT * FROM df_polars WHERE value > 15").pl()
print(result)  # Returns Polars DataFrame
```

### Replacement Scans

**Auto-Register DataFrames:**
```python
import duckdb
import pandas as pd

df1 = pd.DataFrame({'a': [1, 2, 3]})
df2 = pd.DataFrame({'b': [4, 5, 6]})

# DuckDB automatically detects DataFrame names in query
result = duckdb.query("""
    SELECT df1.a, df2.b
    FROM df1
    CROSS JOIN df2
""").df()
```

### Python UDFs (User-Defined Functions)

**Create Python Functions:**
```python
import duckdb

conn = duckdb.connect()

# Create Python UDF
def add_suffix(s):
    return s + '_suffix'

conn.create_function('add_suffix', add_suffix)

# Use in query
result = conn.execute("""
    SELECT add_suffix(name) as new_name
    FROM (VALUES ('Alice'), ('Bob')) AS t(name)
""").fetchall()
```

**Vectorized UDFs:**
```python
import numpy as np

def add_numpy(arr):
    return arr + 10

conn.create_function('add_numpy', add_numpy, return_type='DOUBLE')

# Use with arrays
result = conn.execute("""
    SELECT add_numpy([1, 2, 3, 4, 5])
""").fetchone()
```

### Context Manager

**With Statement:**
```python
import duckdb

with duckdb.connect('analytics.duckdb') as conn:
    result = conn.execute("SELECT * FROM data").df()
    # Connection automatically closed
```

### Parameterized Queries

**Prevent SQL Injection:**
```python
# Using parameters
user_input = "Alice'; DROP TABLE users; --"

# Safe parameterized query
result = conn.execute(
    "SELECT * FROM users WHERE name = ?",
    [user_input]
).fetchall()

# Multiple parameters
result = conn.execute(
    "SELECT * FROM users WHERE name = ? AND age > ?",
    ['Alice', 25]
).fetchall()
```

### Concurrent Access

**Multiple Connections:**
```python
import duckdb
from concurrent.futures import ThreadPoolExecutor

def query_data(query):
    # Each thread needs its own connection
    conn = duckdb.connect('analytics.duckdb', read_only=True)
    result = conn.execute(query).fetchall()
    conn.close()
    return result

with ThreadPoolExecutor(max_workers=4) as executor:
    queries = [
        "SELECT COUNT(*) FROM table1",
        "SELECT SUM(value) FROM table2",
        "SELECT AVG(price) FROM table3",
    ]
    results = executor.map(query_data, queries)
```

### Advanced Python Integration

**Custom Types:**
```python
import duckdb
from datetime import datetime, date

# Insert Python objects
conn = duckdb.connect()
conn.execute("""
    CREATE TABLE events (
        id INTEGER,
        event_date DATE,
        event_time TIMESTAMP
    )
""")

conn.execute(
    "INSERT INTO events VALUES (?, ?, ?)",
    [1, date(2024, 1, 1), datetime(2024, 1, 1, 12, 0, 0)]
)
```

**Lazy Execution:**
```python
# Create relation (lazy)
rel = conn.from_df(df)

# Chain operations (still lazy)
rel = rel.filter("age > 25")
rel = rel.project("name, age")
rel = rel.order("age DESC")

# Execute (eager)
result = rel.df()
```

---

## 9. R Integration

### Installation

**Install R Package:**
```r
install.packages("duckdb")

# Development version
# install.packages("remotes")
# remotes::install_github("duckdb/duckdb-r")
```

### Basic Usage

**Connect and Query:**
```r
library(duckdb)

# Create connection
con <- dbConnect(duckdb::duckdb(), dbdir = ":memory:")

# Or persistent database
con <- dbConnect(duckdb::duckdb(), dbdir = "analytics.duckdb")

# Execute query
result <- dbGetQuery(con, "SELECT 42 as answer")
print(result)

# Disconnect
dbDisconnect(con, shutdown = TRUE)
```

### Query R DataFrames

**Direct DataFrame Queries:**
```r
library(duckdb)
library(DBI)

# Create data frame
df <- data.frame(
    id = c(1, 2, 3),
    name = c("Alice", "Bob", "Carol"),
    age = c(30, 25, 35)
)

# Query data frame directly
result <- dbGetQuery(con, "SELECT * FROM df WHERE age > 25")
print(result)
```

### Integration with dplyr

**dplyr Backend:**
```r
library(duckdb)
library(dplyr)

# Connect
con <- dbConnect(duckdb::duckdb())

# Register data frame
duckdb_register(con, "my_table", df)

# Use dplyr
tbl(con, "my_table") %>%
    filter(age > 25) %>%
    select(name, age) %>%
    arrange(desc(age)) %>%
    collect()

# Lazy evaluation until collect()
```

**dplyr on Parquet:**
```r
library(duckdb)
library(dplyr)

con <- dbConnect(duckdb::duckdb())

# Query Parquet with dplyr
tbl(con, "read_parquet('data.parquet')") %>%
    filter(year == 2024) %>%
    group_by(category) %>%
    summarize(
        total_sales = sum(sales),
        avg_price = mean(price)
    ) %>%
    collect()
```

### Arrow Integration

**Arrow Tables:**
```r
library(duckdb)
library(arrow)

# Read Parquet with Arrow
arrow_table <- read_parquet("data.parquet")

# Query Arrow table with DuckDB
con <- dbConnect(duckdb::duckdb())
result <- dbGetQuery(con, "SELECT * FROM arrow_table WHERE condition")
```

### Write Results

**Export to Files:**
```r
# Write to Parquet
dbExecute(con, "
    COPY (SELECT * FROM my_table)
    TO 'output.parquet' (FORMAT PARQUET)
")

# Write to CSV
dbExecute(con, "
    COPY (SELECT * FROM my_table)
    TO 'output.csv' (HEADER, DELIMITER ',')
")
```

### R Function Integration

**Register R Functions:**
```r
library(duckdb)

con <- dbConnect(duckdb::duckdb())

# Register R function as SQL function
# (Limited support compared to Python)
```

---

## 10. Pandas and Arrow Integration

### Pandas Integration

**Read Parquet to Pandas:**
```python
import duckdb
import pandas as pd

# Method 1: DuckDB query → Pandas
df = duckdb.query("SELECT * FROM 'data.parquet'").df()

# Method 2: DuckDB connection
conn = duckdb.connect()
df = conn.execute("SELECT * FROM 'data.parquet'").df()

# Method 3: Direct file read
df = duckdb.read_parquet('data.parquet').df()
```

**Query Pandas, Return Pandas:**
```python
import pandas as pd
import duckdb

# Create Pandas DataFrame
df_input = pd.DataFrame({
    'product': ['A', 'B', 'C', 'A', 'B'],
    'sales': [100, 200, 150, 120, 180]
})

# Query and get result as Pandas
df_result = duckdb.query("""
    SELECT
        product,
        SUM(sales) as total_sales,
        AVG(sales) as avg_sales
    FROM df_input
    GROUP BY product
    ORDER BY total_sales DESC
""").df()

print(df_result)
```

**Write Pandas to Parquet:**
```python
# Query Pandas, write to Parquet
duckdb.query("""
    COPY (SELECT * FROM df_input)
    TO 'output.parquet' (FORMAT PARQUET, COMPRESSION 'zstd')
""")
```

### Arrow Integration

**Apache Arrow:**
```python
import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

# Read Parquet to Arrow
arrow_table = duckdb.query("SELECT * FROM 'data.parquet'").arrow()

# Query Arrow table
result = duckdb.query("SELECT * FROM arrow_table WHERE condition").arrow()

# Write Arrow to Parquet
pq.write_table(arrow_table, 'output.parquet')
```

**Zero-Copy Arrow:**
```python
import duckdb
import pyarrow as pa

# Create Arrow table
schema = pa.schema([
    ('id', pa.int64()),
    ('name', pa.string()),
    ('value', pa.float64())
])

data = [
    pa.array([1, 2, 3]),
    pa.array(['A', 'B', 'C']),
    pa.array([1.1, 2.2, 3.3])
]

arrow_table = pa.Table.from_arrays(data, schema=schema)

# Query Arrow (zero-copy!)
result = duckdb.query("SELECT * FROM arrow_table WHERE value > 2.0").arrow()
```

### Arrow Flight

**Arrow Flight SQL:**
```python
# DuckDB can serve data via Arrow Flight
# Useful for networked analytics
```

### Performance Comparison

**Pandas vs DuckDB:**
```python
import pandas as pd
import duckdb
import time

# Large dataset
df = pd.DataFrame({
    'category': ['A', 'B', 'C'] * 1_000_000,
    'value': range(3_000_000)
})

# Pandas aggregation
start = time.time()
result_pandas = df.groupby('category')['value'].sum()
print(f"Pandas: {time.time() - start:.2f}s")

# DuckDB aggregation
start = time.time()
result_duckdb = duckdb.query("""
    SELECT category, SUM(value)
    FROM df
    GROUP BY category
""").df()
print(f"DuckDB: {time.time() - start:.2f}s")

# DuckDB is typically 5-50x faster for analytical queries
```

### Polars Integration

**Polars DataFrames:**
```python
import polars as pl
import duckdb

# Create Polars DataFrame
df_polars = pl.DataFrame({
    'id': [1, 2, 3, 4, 5],
    'value': [10, 20, 30, 40, 50]
})

# Query with DuckDB, return as Polars
result = duckdb.query("""
    SELECT
        id,
        value,
        value * 2 as doubled
    FROM df_polars
""").pl()

print(result)
```

**Polars → DuckDB → Parquet:**
```python
import polars as pl
import duckdb

df = pl.read_csv('large_data.csv')

# Process with DuckDB, write to Parquet
duckdb.query("""
    COPY (
        SELECT * FROM df
        WHERE value > 100
    )
    TO 'filtered_data.parquet' (FORMAT PARQUET)
""")
```

---

## 11. Remote File Access

### S3 Access

**Configure S3:**
```sql
INSTALL httpfs;
LOAD httpfs;

-- Set credentials
SET s3_region='us-east-1';
SET s3_access_key_id='YOUR_ACCESS_KEY';
SET s3_secret_access_key='YOUR_SECRET_KEY';

-- Optional: Session token
SET s3_session_token='YOUR_SESSION_TOKEN';

-- Query S3
SELECT * FROM 's3://my-bucket/data.parquet';

-- Write to S3
COPY (SELECT * FROM local_table)
TO 's3://my-bucket/output.parquet';
```

**S3 Glob Patterns:**
```sql
-- Read multiple files
SELECT * FROM 's3://my-bucket/data/*.parquet';

-- Hive partitioning
SELECT * FROM 's3://my-bucket/data/year=2024/**/*.parquet'
WHERE year = 2024 AND month = 1;
```

**Python S3 Configuration:**
```python
import duckdb

conn = duckdb.connect()
conn.install_extension("httpfs")
conn.load_extension("httpfs")

# AWS credentials
conn.execute("SET s3_region='us-west-2'")
conn.execute("SET s3_access_key_id='...'")
conn.execute("SET s3_secret_access_key='...'")

# Query S3
df = conn.execute("SELECT * FROM 's3://bucket/data/*.parquet'").df()
```

### Google Cloud Storage (GCS)

**GCS Access:**
```sql
INSTALL httpfs;
LOAD httpfs;

-- Set GCS credentials
SET gcs_access_key_id='YOUR_ACCESS_KEY';
SET gcs_secret_access_key='YOUR_SECRET_KEY';

-- Query GCS
SELECT * FROM 'gcs://my-bucket/data.parquet';

-- Or use gs:// protocol
SELECT * FROM 'gs://my-bucket/data.parquet';
```

### Azure Blob Storage

**Azure Access:**
```sql
INSTALL azure;
LOAD azure;

-- Set Azure credentials
SET azure_storage_connection_string='DefaultEndpointsProtocol=https;...';

-- Query Azure Blob
SELECT * FROM 'azure://container/data.parquet';
```

### HTTP/HTTPS Files

**Direct HTTP Access:**
```sql
INSTALL httpfs;
LOAD httpfs;

-- Read CSV from URL
SELECT * FROM 'https://example.com/data.csv';

-- Read Parquet from URL
SELECT * FROM 'https://example.com/data.parquet';

-- GitHub raw files
SELECT * FROM 'https://raw.githubusercontent.com/user/repo/main/data.csv';
```

### Authentication

**AWS IAM Roles:**
```python
import duckdb

conn = duckdb.connect()
conn.execute("INSTALL httpfs")
conn.execute("LOAD httpfs")

# Use IAM role (no explicit credentials needed)
conn.execute("SET s3_region='us-east-1'")
conn.execute("SET s3_use_ssl=true")

# Query S3 with IAM role
df = conn.execute("SELECT * FROM 's3://bucket/data.parquet'").df()
```

**AWS Profile:**
```python
import duckdb
import os

# Set AWS profile
os.environ['AWS_PROFILE'] = 'my-profile'

conn = duckdb.connect()
conn.execute("INSTALL httpfs")
conn.execute("LOAD httpfs")

# Uses credentials from ~/.aws/credentials
df = conn.execute("SELECT * FROM 's3://bucket/data.parquet'").df()
```

---

## 12. Schema Design for Analytics

### Star Schema

**Fact and Dimension Tables:**
```sql
-- Dimension tables
CREATE TABLE dim_customer (
    customer_id BIGINT PRIMARY KEY,
    customer_name VARCHAR,
    country VARCHAR,
    segment VARCHAR
);

CREATE TABLE dim_product (
    product_id BIGINT PRIMARY KEY,
    product_name VARCHAR,
    category VARCHAR,
    subcategory VARCHAR
);

CREATE TABLE dim_date (
    date_id INTEGER PRIMARY KEY,
    date DATE,
    year INTEGER,
    quarter INTEGER,
    month INTEGER,
    day_of_week VARCHAR
);

-- Fact table
CREATE TABLE fact_sales (
    sale_id BIGINT PRIMARY KEY,
    customer_id BIGINT,
    product_id BIGINT,
    date_id INTEGER,
    quantity INTEGER,
    unit_price DECIMAL(10,2),
    total_amount DECIMAL(10,2),
    FOREIGN KEY (customer_id) REFERENCES dim_customer(customer_id),
    FOREIGN KEY (product_id) REFERENCES dim_product(product_id),
    FOREIGN KEY (date_id) REFERENCES dim_date(date_id)
);
```

**Star Schema Query:**
```sql
-- Typical analytical query
SELECT
    d.year,
    d.month,
    p.category,
    c.country,
    SUM(f.quantity) as total_quantity,
    SUM(f.total_amount) as total_sales
FROM fact_sales f
JOIN dim_date d ON f.date_id = d.date_id
JOIN dim_product p ON f.product_id = p.product_id
JOIN dim_customer c ON f.customer_id = c.customer_id
WHERE d.year = 2024
GROUP BY d.year, d.month, p.category, c.country
ORDER BY d.month, total_sales DESC;
```

### Columnar Storage Benefits

**Wide vs Narrow Tables:**
```sql
-- WIDE TABLE (better for columnar storage)
CREATE TABLE events_wide (
    event_id BIGINT,
    timestamp TIMESTAMP,
    user_id BIGINT,
    event_type VARCHAR,
    page_url VARCHAR,
    duration INTEGER,
    -- ... 100 more columns
);

-- Query only reads needed columns
SELECT event_type, COUNT(*) FROM events_wide GROUP BY event_type;
-- Only reads 'event_type' column (very fast!)

-- NARROW TABLE (worse for columnar)
CREATE TABLE events_narrow (
    event_id BIGINT,
    attribute_name VARCHAR,
    attribute_value VARCHAR
);
-- Must read entire table to filter by attribute
```

### Denormalization for Analytics

**Pre-Aggregate Common Queries:**
```sql
-- Original query (slow on large fact table)
SELECT
    DATE_TRUNC('day', timestamp) as day,
    category,
    SUM(amount) as daily_sales
FROM raw_transactions
GROUP BY day, category;

-- Materialized aggregation (fast)
CREATE TABLE daily_sales_by_category AS
SELECT
    DATE_TRUNC('day', timestamp) as day,
    category,
    SUM(amount) as daily_sales
FROM raw_transactions
GROUP BY day, category;

-- Query pre-aggregated table
SELECT * FROM daily_sales_by_category WHERE day >= '2024-01-01';
```

### Data Types for Analytics

**Optimal Types:**
```sql
CREATE TABLE analytics_events (
    -- Use smallest appropriate integer type
    event_id UBIGINT,              -- Unsigned big int (0 to 2^64-1)
    user_id UINTEGER,              -- Unsigned int (0 to 4.2B)
    event_type UTINYINT,           -- 0-255 (store as enum mapping)

    -- Timestamps
    event_time TIMESTAMP,          -- Microsecond precision
    event_date DATE,               -- Just date (smaller)

    -- Decimals for precision
    amount DECIMAL(10,2),          -- Currency

    -- Strings
    url VARCHAR,                   -- Variable length
    country VARCHAR(2),            -- Fixed length

    -- Enums (stored as integers internally)
    status ENUM('pending', 'completed', 'failed'),

    -- Lists
    tags VARCHAR[],                -- Array of strings

    -- JSON
    metadata JSON
);
```

### Partitioning Strategy

**Time-Based Partitioning:**
```sql
-- Partition by year and month in directory structure
-- See section 6 for details

-- Best practices:
-- 1. Partition by time (year, month, day)
-- 2. Partition by high-cardinality dimension (country, product_id)
-- 3. Keep partition size 100MB - 1GB
-- 4. Don't over-partition (too many small files)
```

---

## 13. Aggregations and Window Functions

### Aggregation Functions

**Common Aggregations:**
```sql
SELECT
    category,
    COUNT(*) as count,
    COUNT(DISTINCT user_id) as unique_users,
    SUM(amount) as total,
    AVG(amount) as average,
    MIN(amount) as minimum,
    MAX(amount) as maximum,
    STDDEV(amount) as std_deviation,
    VARIANCE(amount) as variance,
    MEDIAN(amount) as median
FROM transactions
GROUP BY category;
```

**Approximate Aggregations:**
```sql
-- Faster for large datasets
SELECT
    category,
    APPROX_COUNT_DISTINCT(user_id) as approx_unique_users,
    APPROX_QUANTILE(amount, 0.5) as approx_median,
    APPROX_QUANTILE(amount, [0.25, 0.5, 0.75]) as quartiles
FROM large_transactions
GROUP BY category;

-- Much faster than exact COUNT(DISTINCT) and MEDIAN
```

### Window Functions

**Row Number and Rank:**
```sql
SELECT
    product_name,
    sales,
    ROW_NUMBER() OVER (ORDER BY sales DESC) as row_num,
    RANK() OVER (ORDER BY sales DESC) as rank,
    DENSE_RANK() OVER (ORDER BY sales DESC) as dense_rank,
    PERCENT_RANK() OVER (ORDER BY sales DESC) as percent_rank
FROM products
ORDER BY sales DESC;
```

**Partition Window Functions:**
```sql
-- Top 3 products per category
SELECT *
FROM (
    SELECT
        category,
        product_name,
        sales,
        ROW_NUMBER() OVER (
            PARTITION BY category
            ORDER BY sales DESC
        ) as rank_in_category
    FROM products
)
WHERE rank_in_category <= 3;
```

**Cumulative Aggregations:**
```sql
-- Running total
SELECT
    date,
    sales,
    SUM(sales) OVER (
        ORDER BY date
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) as running_total
FROM daily_sales
ORDER BY date;
```

**Moving Averages:**
```sql
-- 7-day moving average
SELECT
    date,
    temperature,
    AVG(temperature) OVER (
        ORDER BY date
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ) as moving_avg_7d,
    AVG(temperature) OVER (
        ORDER BY date
        ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
    ) as moving_avg_30d
FROM weather_data
ORDER BY date;
```

**Lead and Lag:**
```sql
-- Compare with previous/next row
SELECT
    date,
    price,
    LAG(price, 1) OVER (ORDER BY date) as prev_price,
    LEAD(price, 1) OVER (ORDER BY date) as next_price,
    price - LAG(price, 1) OVER (ORDER BY date) as price_change
FROM stock_prices
ORDER BY date;
```

**First and Last Value:**
```sql
SELECT
    date,
    sales,
    FIRST_VALUE(sales) OVER (
        PARTITION BY EXTRACT(MONTH FROM date)
        ORDER BY date
    ) as first_day_sales,
    LAST_VALUE(sales) OVER (
        PARTITION BY EXTRACT(MONTH FROM date)
        ORDER BY date
        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
    ) as last_day_sales
FROM daily_sales
ORDER BY date;
```

### QUALIFY Clause

**Filter Window Results:**
```sql
-- Top 5 products per category (simplified)
SELECT
    category,
    product_name,
    sales,
    ROW_NUMBER() OVER (PARTITION BY category ORDER BY sales DESC) as rank
FROM products
QUALIFY rank <= 5
ORDER BY category, rank;

-- No need for subquery!
```

### Grouping Sets

**CUBE and ROLLUP:**
```sql
-- CUBE: All combinations of grouping
SELECT
    year,
    quarter,
    category,
    SUM(sales) as total_sales
FROM sales_data
GROUP BY CUBE (year, quarter, category);

-- ROLLUP: Hierarchical grouping
SELECT
    year,
    quarter,
    month,
    SUM(sales) as total_sales
FROM sales_data
GROUP BY ROLLUP (year, quarter, month);

-- GROUPING SETS: Specific combinations
SELECT
    category,
    region,
    SUM(sales) as total_sales
FROM sales_data
GROUP BY GROUPING SETS (
    (category),
    (region),
    (category, region),
    ()  -- Grand total
);
```

---

## 14. Data Types and Storage

### Numeric Types

**Integer Types:**
```sql
TINYINT       -- 1 byte: -128 to 127
UTINYINT      -- 1 byte: 0 to 255
SMALLINT      -- 2 bytes: -32,768 to 32,767
USMALLINT     -- 2 bytes: 0 to 65,535
INTEGER       -- 4 bytes: -2.1B to 2.1B
UINTEGER      -- 4 bytes: 0 to 4.2B
BIGINT        -- 8 bytes: -9.2 quintillion to 9.2 quintillion
UBIGINT       -- 8 bytes: 0 to 18.4 quintillion
HUGEINT       -- 16 bytes: -170 undecillion to 170 undecillion
```

**Floating Point:**
```sql
REAL / FLOAT   -- 4 bytes: single precision
DOUBLE         -- 8 bytes: double precision
DECIMAL(p,s)   -- Arbitrary precision (use for money)
```

**Example:**
```sql
CREATE TABLE numeric_examples (
    id UBIGINT,
    count UINTEGER,
    price DECIMAL(10,2),      -- $99,999,999.99
    percentage REAL,
    scientific DOUBLE
);
```

### String Types

**String Types:**
```sql
VARCHAR          -- Variable length string
VARCHAR(n)       -- Max length n
CHAR(n)          -- Fixed length (rarely used)
TEXT             -- Alias for VARCHAR
```

**String Functions:**
```sql
SELECT
    UPPER('hello') as upper_case,           -- HELLO
    LOWER('WORLD') as lower_case,           -- world
    CONCAT('Hello', ' ', 'World'),          -- Hello World
    SUBSTRING('DuckDB', 1, 4) as sub,       -- Duck
    LENGTH('DuckDB') as len,                -- 6
    TRIM('  spaces  ') as trimmed,          -- spaces
    REPLACE('DuckDB', 'Duck', 'Goose'),     -- GooseDB
    REGEXP_MATCHES('test123', '[0-9]+');    -- true
```

### Temporal Types

**Date and Time:**
```sql
DATE           -- Calendar date (year, month, day)
TIME           -- Time of day (hour, minute, second, microsecond)
TIMESTAMP      -- Date + time (microsecond precision)
INTERVAL       -- Time span
```

**Examples:**
```sql
CREATE TABLE temporal_examples (
    event_date DATE,
    event_time TIME,
    event_timestamp TIMESTAMP,
    duration INTERVAL
);

INSERT INTO temporal_examples VALUES (
    DATE '2024-01-15',
    TIME '14:30:00',
    TIMESTAMP '2024-01-15 14:30:00',
    INTERVAL '2 hours'
);

-- Date arithmetic
SELECT
    DATE '2024-01-01' + INTERVAL 7 DAY as next_week,
    TIMESTAMP '2024-01-01 12:00:00' - INTERVAL 1 HOUR as hour_ago,
    DATE_DIFF('day', DATE '2024-01-01', DATE '2024-12-31') as days_diff;
```

### Complex Types

**Arrays:**
```sql
-- Integer array
CREATE TABLE array_example (
    id INTEGER,
    tags INTEGER[]
);

INSERT INTO array_example VALUES
    (1, [10, 20, 30]),
    (2, [40, 50]);

SELECT
    id,
    tags,
    tags[1] as first_tag,        -- Access element
    array_length(tags) as count,  -- Array length
    list_contains(tags, 20) as has_20
FROM array_example;
```

**Structs (Records):**
```sql
-- Nested structures
CREATE TABLE struct_example (
    id INTEGER,
    person STRUCT(name VARCHAR, age INTEGER, address STRUCT(city VARCHAR, country VARCHAR))
);

INSERT INTO struct_example VALUES
    (1, {'name': 'Alice', 'age': 30, 'address': {'city': 'NYC', 'country': 'USA'}});

SELECT
    id,
    person.name,
    person.age,
    person.address.city
FROM struct_example;
```

**Maps:**
```sql
-- Key-value pairs
CREATE TABLE map_example (
    id INTEGER,
    attributes MAP(VARCHAR, INTEGER)
);

INSERT INTO map_example VALUES
    (1, MAP(['height', 'weight'], [180, 75]));

SELECT
    id,
    attributes['height'] as height
FROM map_example;
```

**JSON:**
```sql
-- JSON type
CREATE TABLE json_example (
    id INTEGER,
    data JSON
);

INSERT INTO json_example VALUES
    (1, '{"user": {"name": "Alice", "age": 30}, "tags": ["a", "b"]}');

SELECT
    id,
    data->>'$.user.name' as user_name,
    data->>'$.user.age' as age,
    json_extract(data, '$.tags') as tags
FROM json_example;
```

**UNION Types:**
```sql
-- Multiple possible types
CREATE TABLE union_example (
    id INTEGER,
    value UNION(num INTEGER, str VARCHAR, arr INTEGER[])
);

INSERT INTO union_example VALUES
    (1, 42),
    (2, 'hello'),
    (3, [1, 2, 3]);
```

### Storage and Compression

**Automatic Compression:**
```
DuckDB automatically compresses data using:
- Dictionary encoding (repeated values)
- Run-length encoding (RLE)
- Bit-packing
- Frame-of-reference
- FSST (string compression)

No manual configuration needed!
```

**Check Storage Size:**
```sql
-- Database size
SELECT
    database_size,
    block_size,
    total_blocks
FROM pragma_database_size();

-- Table size
SELECT
    table_name,
    estimated_size
FROM duckdb_tables();
```

---

## 15. Transaction and Concurrency

### ACID Properties

DuckDB provides full ACID guarantees:
- **Atomicity:** All or nothing
- **Consistency:** Constraints enforced
- **Isolation:** Serializable isolation
- **Durability:** WAL-based persistence

### Transactions

**Basic Transactions:**
```sql
BEGIN TRANSACTION;

INSERT INTO accounts (id, balance) VALUES (1, 1000);
UPDATE accounts SET balance = balance - 100 WHERE id = 1;
INSERT INTO transactions (account_id, amount) VALUES (1, -100);

COMMIT;
-- Or ROLLBACK
```

**Python Transactions:**
```python
import duckdb

conn = duckdb.connect('analytics.duckdb')

try:
    conn.begin()
    conn.execute("INSERT INTO users VALUES (1, 'Alice')")
    conn.execute("INSERT INTO orders VALUES (1, 1, 100.00)")
    conn.commit()
except Exception as e:
    conn.rollback()
    print(f"Transaction failed: {e}")
```

### Concurrency Model

**Single Writer, Multiple Readers:**
```
Similar to SQLite:
- One write transaction at a time
- Multiple concurrent read transactions
- Readers don't block writers (in WAL mode)
- Writers don't block readers (in WAL mode)

Use cases:
✅ Analytical workloads (read-heavy)
✅ Single-user applications
✅ ETL pipelines (sequential writes)
❌ High-concurrency OLTP (use PostgreSQL)
```

**WAL Mode:**
```python
import duckdb

# Persistent database automatically uses WAL
conn = duckdb.connect('analytics.duckdb')

# Readers can query while writer is active
```

### Checkpointing

**Manual Checkpoint:**
```sql
-- Force checkpoint (write WAL to main database)
CHECKPOINT;

-- Check WAL size
PRAGMA wal_autocheckpoint;
```

### Read-Only Mode

**Open Read-Only:**
```python
import duckdb

# Read-only connection (allows multiple processes)
conn = duckdb.connect('analytics.duckdb', read_only=True)

# Query (no writes allowed)
df = conn.execute("SELECT * FROM data").df()
```

**Multiple Read-Only Connections:**
```python
# Multiple processes can read simultaneously
from multiprocessing import Pool

def read_data(query):
    conn = duckdb.connect('analytics.duckdb', read_only=True)
    return conn.execute(query).fetchall()

with Pool(4) as p:
    results = p.map(read_data, [
        "SELECT COUNT(*) FROM table1",
        "SELECT SUM(value) FROM table2",
        "SELECT AVG(price) FROM table3",
        "SELECT MAX(timestamp) FROM table4"
    ])
```

---

## 16. Use Cases and Limitations

### Ideal Use Cases

**1. Data Science and Analytics:**
```python
import duckdb
import pandas as pd

# Load large CSV, analyze with SQL
df = duckdb.query("""
    SELECT
        category,
        AVG(price) as avg_price,
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY price) as median
    FROM 'large_data.csv'
    GROUP BY category
""").df()
```

**2. ETL Pipelines:**
```python
# Transform CSV to Parquet with filtering
duckdb.query("""
    COPY (
        SELECT
            id,
            UPPER(name) as name,
            price * 1.1 as price_with_tax
        FROM 'raw_data.csv'
        WHERE price > 0
    )
    TO 'clean_data.parquet' (FORMAT PARQUET, COMPRESSION 'zstd')
""")
```

**3. Data Lake Queries:**
```sql
-- Query Parquet files on S3 directly
SELECT
    year,
    month,
    SUM(sales) as total_sales
FROM 's3://datalake/sales/year=*/month=*/*.parquet'
WHERE year >= 2023
GROUP BY year, month
ORDER BY year, month;
```

**4. Embedded Analytics:**
```python
# Desktop app with embedded analytics
import duckdb

class AnalyticsEngine:
    def __init__(self, db_path):
        self.conn = duckdb.connect(db_path)

    def get_sales_summary(self, start_date, end_date):
        return self.conn.execute("""
            SELECT
                DATE_TRUNC('day', sale_date) as day,
                SUM(amount) as daily_sales
            FROM sales
            WHERE sale_date BETWEEN ? AND ?
            GROUP BY day
            ORDER BY day
        """, [start_date, end_date]).df()
```

**5. Log Analysis:**
```sql
-- Analyze log files
SELECT
    EXTRACT(HOUR FROM timestamp) as hour,
    status_code,
    COUNT(*) as request_count,
    AVG(response_time) as avg_response_time
FROM read_json('logs/**/*.json')
WHERE timestamp >= CURRENT_DATE - INTERVAL 7 DAY
GROUP BY hour, status_code
ORDER BY hour, status_code;
```

### Limitations

**1. Single-Node Only:**
```
❌ No clustering or distributed execution
✅ Use ClickHouse, Snowflake for distributed analytics
```

**2. Write Concurrency:**
```
❌ One writer at a time (like SQLite)
✅ Use PostgreSQL, MySQL for multi-user OLTP
```

**3. No Built-In Replication:**
```
❌ No master-slave or multi-master replication
✅ Backup files or use external tools
```

**4. In-Memory or Single-File:**
```
❌ Not designed for very large (>TB) datasets on disk
✅ Use partitioned Parquet files on S3 for larger data
```

**5. No Network Protocol:**
```
❌ No client-server protocol (embedded only)
✅ Can use Arrow Flight or build REST API wrapper
```

### Performance Characteristics

**Typical Performance:**
```
Operation               | Performance
------------------------|------------------------------------
Simple SELECT           | Millions of rows/second
Aggregation             | 100M+ rows/second (vectorized)
Parquet scan            | GB/second (parallel, compressed)
CSV parsing             | 100s MB/second
Join (hash)             | Millions of rows/second
Window functions        | Fast (optimized)
Full-text search        | Extension required
```

**Memory Requirements:**
```
- Can process larger-than-memory datasets
- Automatic spilling to disk
- Efficient memory usage (columnar compression)
- Recommended: 16GB+ RAM for large datasets
```

---

## 17. Comparison with Other Databases

### DuckDB vs SQLite

| Feature | DuckDB | SQLite |
|---------|--------|--------|
| **Purpose** | OLAP (analytics) | OLTP (transactions) |
| **Storage** | Columnar | Row-based |
| **Performance** | Fast aggregations | Fast point queries |
| **Concurrency** | Single writer | Single writer |
| **Data size** | GB to TB (with partitioning) | MB to GB |
| **File formats** | Parquet, CSV, JSON | SQLite format |
| **Best for** | Analytics, data science | CRUD, mobile apps |

**When to use each:**
```
DuckDB: SELECT AVG(price) FROM 1B rows → Milliseconds
SQLite: SELECT * FROM users WHERE id = 1 → Microseconds

Use DuckDB for: Analytics, aggregations, large scans
Use SQLite for: CRUD operations, indexes, transactions
```

### DuckDB vs PostgreSQL

| Feature | DuckDB | PostgreSQL |
|---------|--------|------------|
| **Architecture** | Embedded | Client-server |
| **OLAP** | ⚡ Excellent | Good |
| **OLTP** | Limited | ⚡ Excellent |
| **Concurrency** | Limited writes | High concurrency |
| **Setup** | Zero config | Server setup |
| **Use case** | Analytics, data science | Production apps |

### DuckDB vs Pandas

| Feature | DuckDB | Pandas |
|---------|--------|--------|
| **Performance** | 5-50x faster (analytics) | Baseline |
| **Memory** | Larger-than-memory | In-memory only |
| **Query** | SQL | Python API |
| **Parallel** | Automatic | Manual |
| **File formats** | Direct Parquet queries | Load to memory |

**Example:**
```python
# Pandas (slower, loads entire file)
df = pd.read_parquet('large.parquet')
result = df.groupby('category')['sales'].sum()

# DuckDB (faster, streams data)
result = duckdb.query("""
    SELECT category, SUM(sales)
    FROM 'large.parquet'
    GROUP BY category
""").df()
```

### DuckDB vs ClickHouse

| Feature | DuckDB | ClickHouse |
|---------|--------|------------|
| **Architecture** | Embedded | Distributed |
| **Scale** | Single-node | Cluster |
| **Setup** | Zero | Complex |
| **Data size** | GB-TB | PB+ |
| **Use case** | Local analytics | Production OLAP |

### DuckDB vs Polars

| Feature | DuckDB | Polars |
|---------|--------|--------|
| **Interface** | SQL | DataFrame API |
| **Performance** | Comparable | Comparable |
| **File queries** | Direct | Load first |
| **Ecosystem** | SQL-based | Rust/Python |

**Combined:**
```python
import polars as pl
import duckdb

# Use both!
df = pl.read_csv('data.csv')
result = duckdb.query("SELECT * FROM df WHERE value > 100").pl()
```

---

## 18. Deployment Patterns

### In-Memory Analytics

**Pure In-Memory:**
```python
import duckdb

# In-memory database
conn = duckdb.connect(':memory:')

# Load data
conn.execute("CREATE TABLE data AS SELECT * FROM 'file.parquet'")

# Query
result = conn.execute("SELECT * FROM data WHERE condition").df()

# Data lost when connection closed
```

### Persistent Database

**Single File Database:**
```python
import duckdb

# Persistent database (single file)
conn = duckdb.connect('analytics.duckdb')

# Import data (persistent)
conn.execute("CREATE TABLE sales AS SELECT * FROM 'sales.parquet'")

# Query anytime
result = conn.execute("SELECT * FROM sales").df()

# Close (data persists)
conn.close()
```

### Data Lakehouse Pattern

**Query Files Directly:**
```python
import duckdb

conn = duckdb.connect()

# Query Parquet files on S3 (no import needed)
result = conn.execute("""
    SELECT
        year,
        month,
        SUM(sales) as total_sales
    FROM 's3://datalake/sales/**/*.parquet'
    WHERE year = 2024
    GROUP BY year, month
""").df()

# Lakehouse: Query data where it lives (S3, GCS, etc.)
```

### REST API Wrapper

**Flask API:**
```python
from flask import Flask, request, jsonify
import duckdb

app = Flask(__name__)
conn = duckdb.connect('analytics.duckdb', read_only=True)

@app.route('/query', methods=['POST'])
def query():
    sql = request.json.get('sql')

    # Validate query (prevent SQL injection)
    if not sql.upper().startswith('SELECT'):
        return jsonify({'error': 'Only SELECT queries allowed'}), 400

    try:
        result = conn.execute(sql).fetchall()
        return jsonify({'data': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/sales/summary')
def sales_summary():
    result = conn.execute("""
        SELECT
            DATE_TRUNC('month', sale_date) as month,
            SUM(amount) as total_sales
        FROM sales
        WHERE sale_date >= CURRENT_DATE - INTERVAL 12 MONTH
        GROUP BY month
        ORDER BY month
    """).df()
    return jsonify(result.to_dict(orient='records'))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

### Docker Deployment

**Dockerfile:**
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install DuckDB
RUN pip install duckdb pandas

# Copy application
COPY app.py .
COPY analytics.duckdb .

# Expose port
EXPOSE 5000

CMD ["python", "app.py"]
```

**Docker Compose:**
```yaml
version: '3.8'

services:
  duckdb-api:
    build: .
    ports:
      - "5000:5000"
    volumes:
      - ./data:/app/data
      - ./analytics.duckdb:/app/analytics.duckdb
    environment:
      - DUCKDB_DATABASE=/app/analytics.duckdb
    restart: unless-stopped
```

### Jupyter/Notebook Environment

**Jupyter Integration:**
```python
# Install: pip install duckdb jupysql

%load_ext sql
%sql duckdb:///:memory:

%%sql
SELECT
    category,
    AVG(price) as avg_price
FROM 'products.parquet'
GROUP BY category
ORDER BY avg_price DESC
LIMIT 10
```

### Scheduled ETL Jobs

**Cron Job:**
```python
#!/usr/bin/env python3
# daily_etl.py

import duckdb
from datetime import datetime

def run_etl():
    conn = duckdb.connect('warehouse.duckdb')

    # Extract from source, transform, load
    conn.execute("""
        INSERT INTO daily_sales
        SELECT
            DATE_TRUNC('day', timestamp) as date,
            category,
            SUM(amount) as total_sales
        FROM 's3://raw-data/sales/*.parquet'
        WHERE DATE_TRUNC('day', timestamp) = CURRENT_DATE - INTERVAL 1 DAY
        GROUP BY date, category
    """)

    print(f"ETL completed at {datetime.now()}")
    conn.close()

if __name__ == '__main__':
    run_etl()
```

**Crontab:**
```bash
# Run daily at 2 AM
0 2 * * * /usr/bin/python3 /opt/etl/daily_etl.py >> /var/log/etl.log 2>&1
```

### Cloud Function / Lambda

**AWS Lambda:**
```python
import duckdb
import boto3
import json

def lambda_handler(event, context):
    # Query data from S3
    conn = duckdb.connect(':memory:')
    conn.execute("INSTALL httpfs")
    conn.execute("LOAD httpfs")

    result = conn.execute("""
        SELECT
            category,
            SUM(sales) as total_sales
        FROM 's3://my-bucket/data/*.parquet'
        WHERE date = CURRENT_DATE - INTERVAL 1 DAY
        GROUP BY category
    """).df()

    return {
        'statusCode': 200,
        'body': json.dumps(result.to_dict(orient='records'))
    }
```

---

## 19. Testing and Benchmarking

### Unit Testing

**pytest Example:**
```python
import pytest
import duckdb
import pandas as pd

@pytest.fixture
def db_connection():
    conn = duckdb.connect(':memory:')
    conn.execute("""
        CREATE TABLE users (
            id INTEGER,
            name VARCHAR,
            age INTEGER
        )
    """)
    conn.execute("INSERT INTO users VALUES (1, 'Alice', 30), (2, 'Bob', 25)")
    yield conn
    conn.close()

def test_select_users(db_connection):
    result = db_connection.execute("SELECT * FROM users").fetchall()
    assert len(result) == 2

def test_filter_users(db_connection):
    result = db_connection.execute("""
        SELECT * FROM users WHERE age > 26
    """).fetchall()
    assert len(result) == 1
    assert result[0][1] == 'Alice'

def test_aggregation(db_connection):
    result = db_connection.execute("""
        SELECT AVG(age) as avg_age FROM users
    """).fetchone()
    assert result[0] == 27.5
```

### Benchmarking

**Simple Benchmark:**
```python
import duckdb
import time
import pandas as pd

def benchmark_query(query, iterations=10):
    conn = duckdb.connect(':memory:')

    times = []
    for _ in range(iterations):
        start = time.time()
        conn.execute(query).fetchall()
        times.append(time.time() - start)

    print(f"Query: {query[:50]}...")
    print(f"Avg time: {sum(times)/len(times)*1000:.2f}ms")
    print(f"Min time: {min(times)*1000:.2f}ms")
    print(f"Max time: {max(times)*1000:.2f}ms")

# Benchmark
benchmark_query("SELECT * FROM 'large_file.parquet'")
```

**DuckDB vs Pandas Benchmark:**
```python
import duckdb
import pandas as pd
import time

# Create large dataset
df = pd.DataFrame({
    'category': ['A', 'B', 'C'] * 1_000_000,
    'value': range(3_000_000)
})

# Pandas benchmark
start = time.time()
result_pandas = df.groupby('category')['value'].sum()
pandas_time = time.time() - start

# DuckDB benchmark
start = time.time()
result_duckdb = duckdb.query("""
    SELECT category, SUM(value) as total
    FROM df
    GROUP BY category
""").df()
duckdb_time = time.time() - start

print(f"Pandas: {pandas_time:.3f}s")
print(f"DuckDB: {duckdb_time:.3f}s")
print(f"Speedup: {pandas_time/duckdb_time:.1f}x")
```

### Profiling

**Query Profiling:**
```python
import duckdb

conn = duckdb.connect()
conn.execute("PRAGMA enable_profiling")

# Run query
result = conn.execute("""
    SELECT
        category,
        COUNT(*) as count,
        AVG(price) as avg_price
    FROM 'products.parquet'
    GROUP BY category
""").fetchall()

# Get profile
profile = conn.execute("PRAGMA last_profiling_output").fetchall()
print(profile)

conn.execute("PRAGMA disable_profiling")
```

**Python Profiling:**
```python
import cProfile
import duckdb

def run_query():
    conn = duckdb.connect()
    return conn.execute("""
        SELECT * FROM 'large.parquet'
        WHERE condition
    """).fetchall()

# Profile
cProfile.run('run_query()')
```

---

## 20. Version-Specific Features

### DuckDB 1.1 (2025)

**New Features:**
- Enhanced JSON support
- Improved Parquet writer
- Better error messages
- Performance improvements

### DuckDB 1.0 (2024)

**Major Features:**
- Stable API guarantee
- Production-ready
- Full ACID compliance
- Improved performance

**New SQL Features:**
```sql
-- QUALIFY clause (filter window results)
SELECT
    category,
    product,
    sales,
    ROW_NUMBER() OVER (PARTITION BY category ORDER BY sales DESC) as rank
FROM products
QUALIFY rank <= 5;

-- EXCLUDE clause
SELECT * EXCLUDE (sensitive_column) FROM users;

-- REPLACE clause
SELECT * REPLACE (UPPER(name) as name) FROM users;
```

### DuckDB 0.10 (2024)

**Features:**
- Improved extension system
- Better error handling
- Enhanced optimizer

### DuckDB 0.9 (2023)

**Features:**
- Fixed-point decimals
- Improved join performance
- Better memory management

### Migration Guide

**Upgrade from 0.x to 1.0:**
```python
import duckdb

# Old database
old_conn = duckdb.connect('old_db.duckdb')

# Export to Parquet
old_conn.execute("COPY (SELECT * FROM table1) TO 'backup/table1.parquet'")
old_conn.close()

# Create new database with 1.0
new_conn = duckdb.connect('new_db.duckdb')
new_conn.execute("CREATE TABLE table1 AS SELECT * FROM 'backup/table1.parquet'")
```

---

## References and Resources

### Official Documentation
- **DuckDB Website:** https://duckdb.org/
- **Documentation:** https://duckdb.org/docs/
- **SQL Reference:** https://duckdb.org/docs/sql/introduction
- **API Documentation:** https://duckdb.org/docs/api/overview

### GitHub and Community
- **GitHub Repository:** https://github.com/duckdb/duckdb
- **Discussions:** https://github.com/duckdb/duckdb/discussions
- **Discord:** https://discord.duckdb.org/
- **Twitter:** @duckdb

### Extensions
- **Extension Repository:** https://github.com/duckdb/duckdb/tree/main/extension
- **Community Extensions:** https://community-extensions.duckdb.org/

### Tutorials and Blogs
- **DuckDB Blog:** https://duckdb.org/news/
- **Awesome DuckDB:** https://github.com/davidgasquez/awesome-duckdb
- **MotherDuck Blog:** https://motherduck.com/blog/

### Integration Guides
- **Python:** https://duckdb.org/docs/api/python/overview
- **R:** https://duckdb.org/docs/api/r
- **Node.js:** https://duckdb.org/docs/api/nodejs/overview
- **Java:** https://duckdb.org/docs/api/java

### Books and Courses
- DuckDB documentation (most comprehensive)
- Community tutorials and blog posts
- YouTube tutorials

---

**Document Maintenance:**
- Review quarterly for DuckDB updates
- Update with new extensions and features
- Add community-discovered patterns
- Benchmark new versions

**Last Updated:** February 2026
**Next Review:** May 2026
