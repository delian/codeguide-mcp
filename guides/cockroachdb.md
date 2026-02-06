# CockroachDB Development Guidelines
Mandatory coding standards and development practices for CockroachDB development. CockroachDB 23.x+, SQL (PostgreSQL-compatible), multi-region, backup/restore.

---

**Agent Profile**: The CockroachDB Expert
**Role**: Senior Distributed Database Engineer & SQL Specialist
**Objective**: Generate production-ready, resilient and scalable distributed SQL solutions.
**Tools**: CockroachDB 23.x+, SQL (PostgreSQL-compatible), multi-region, backup/restore

---

**Version:** 1.0 | **Last Updated:** February 2026 | **Target Version:** CockroachDB 23.x+

## Table of Contents

1. [Core Philosophies: DISTRIBUTED-FIRST](#1-core-philosophies-distributed-first)
2. [Architecture and Fundamentals](#2-architecture-and-fundamentals)
3. [SQL and Query Language](#3-sql-and-query-language)
4. [Schema Design](#4-schema-design)
5. [Indexes and Constraints](#5-indexes-and-constraints)
6. [Performance Optimization](#6-performance-optimization)
7. [Transactions and Concurrency](#7-transactions-and-concurrency)
8. [Multi-Region Configuration](#8-multi-region-configuration)
9. [Cluster Configuration](#9-cluster-configuration)
10. [Data Distribution and Sharding](#10-data-distribution-and-sharding)
11. [Backup and Recovery](#11-backup-and-recovery)
12. [Security Best Practices](#12-security-best-practices)
13. [Monitoring and Troubleshooting](#13-monitoring-and-troubleshooting)
14. [High Availability and Survivability](#14-high-availability-and-survivability)
15. [Application Integration](#15-application-integration)
16. [Production Deployment](#16-production-deployment)
17. [Scaling Strategies](#17-scaling-strategies)
18. [Migration Strategies](#18-migration-strategies)
19. [Time Travel and Change Data Capture](#19-time-travel-and-change-data-capture)
20. [Comparison with Other Databases](#20-comparison-with-other-databases)
21. [Production Checklist](#21-production-checklist)

---

## 1. Core Philosophies: DISTRIBUTED-FIRST

The agent must adhere to the **DISTRIBUTED-FIRST** principles for every CockroachDB implementation:

**Test-Driven Development (TDD)**: ALWAYS write tests BEFORE implementation (Red-Green-Refactor cycle mandatory).
**Regression Shield**: EVERY bug discovered MUST receive a test BEFORE fixing to prevent regression.

- **D**istributed by design: Prefer topology-aware schema and locality; use multi-region features.
- **I**ndexes and constraints: Use appropriate indexes and constraints for distributed execution.
- **S**erializable isolation: Rely on strong consistency; design for serializable semantics.
- **T**ransactions: Keep transactions short and avoid cross-region hotspots.
- **R**esilience: Plan for node/region failure; use backup, restore, and survivability settings.
- **I**dempotency: Prefer idempotent operations and application-level retries.
- **B**ackup and time travel: Use backup/restore and AS OF SYSTEM TIME for safety.
- **U**nified SQL: Use PostgreSQL-compatible SQL; avoid unsupported or deprecated features.
- **T**esting: Test with multi-node and failure scenarios where possible.

**Verified Code**: Agent-generated code MUST use parameterized SQL, run against a cluster or dev setup, and pass tests before delivery.

---

## 2. Architecture and Fundamentals

### What is CockroachDB?

**CockroachDB** is a distributed, resilient SQL database built for cloud-native applications:

- ✅ **Distributed SQL** (PostgreSQL wire protocol compatible)
- ✅ **Strong consistency** (serializable isolation)
- ✅ **Horizontal scalability** (add nodes without downtime)
- ✅ **Geo-distributed** (multi-region/multi-cloud)
- ✅ **Survivable** (automatic replication and failover)
- ✅ **Cloud-native** (Kubernetes-ready)
- ✅ **Zero downtime** (rolling upgrades, schema changes)

### Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│          CockroachDB Cluster Architecture            │
├─────────────────────────────────────────────────────┤
│                                                      │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐   │
│  │   Node 1   │  │   Node 2   │  │   Node 3   │   │
│  ├────────────┤  ├────────────┤  ├────────────┤   │
│  │  SQL Layer │  │  SQL Layer │  │  SQL Layer │   │
│  │    (PG)    │  │    (PG)    │  │    (PG)    │   │
│  ├────────────┤  ├────────────┤  ├────────────┤   │
│  │Transaction │  │Transaction │  │Transaction │   │
│  │   Layer    │  │   Layer    │  │   Layer    │   │
│  ├────────────┤  ├────────────┤  ├────────────┤   │
│  │Distribution│  │Distribution│  │Distribution│   │
│  │   Layer    │  │   Layer    │  │   Layer    │   │
│  ├────────────┤  ├────────────┤  ├────────────┤   │
│  │Replication │  │Replication │  │Replication │   │
│  │  (Raft)    │  │  (Raft)    │  │  (Raft)    │   │
│  ├────────────┤  ├────────────┤  ├────────────┤   │
│  │  Storage   │  │  Storage   │  │  Storage   │   │
│  │  (RocksDB) │  │  (RocksDB) │  │  (RocksDB) │   │
│  └────────────┘  └────────────┘  └────────────┘   │
│         │               │               │           │
│         └───────────────┴───────────────┘           │
│              Gossip Protocol                        │
└─────────────────────────────────────────────────────┘
```

### Key Concepts

**Ranges:**
```
Data is split into 64MB ranges (default)
Each range is replicated across nodes (default: 3 replicas)

Table: users (1GB)
├── Range 1: rows 1-100K     → Replicas: Node1, Node2, Node3
├── Range 2: rows 100K-200K  → Replicas: Node2, Node3, Node4
├── Range 3: rows 200K-300K  → Replicas: Node1, Node3, Node4
└── Range N: ..
```

**Leaseholders:**
```
Each range has one leaseholder (handles reads/writes)
Other replicas are followers (backup)

Range 1:
- Leaseholder: Node1 (serves requests)
- Follower: Node2 (replicates data)
- Follower: Node3 (replicates data)
```

**Raft Consensus:**
```
Uses Raft protocol for consistency
Requires majority (quorum) for writes

3-node cluster:
- Quorum: 2 nodes
- Can tolerate 1 node failure

5-node cluster:
- Quorum: 3 nodes
- Can tolerate 2 node failures
```

### Layers Explained

**SQL Layer:**
- PostgreSQL-compatible interface
- Query parsing and planning
- Cost-based optimizer

**Transaction Layer:**
- ACID guarantees
- Serializable isolation
- Multi-version concurrency control (MVCC)

**Distribution Layer:**
- Range splitting and merging
- Load balancing
- Data placement

**Replication Layer:**
- Raft consensus
- Automatic rebalancing
- Zone configurations

**Storage Layer:**
- RocksDB (embedded key-value store)
- LSM trees
- Compaction

### When to Use CockroachDB

**✅ Excellent For:**

1. **Global Applications:**
   - Multi-region deployments
   - Low-latency worldwide access
   - Regulatory data sovereignty

2. **High Availability:**
   - 99.99%+ uptime requirements
   - Zero-downtime operations
   - Automatic failover

3. **Horizontal Scaling:**
   - Unpredictable growth
   - Need to add capacity dynamically
   - Cloud-native architecture

4. **Strong Consistency:**
   - Financial transactions
   - Inventory management
   - Critical data integrity

5. **Cloud Portability:**
   - Multi-cloud strategy
   - Avoid vendor lock-in
   - Kubernetes deployments

**❌ Not Recommended For:**

1. **OLAP/Analytics:**
   - Use ClickHouse, Snowflake instead
   - Better specialized solutions exist

2. **Single-Region, Small Scale:**
   - PostgreSQL may be simpler
   - Overhead not justified

3. **Document-Heavy Workloads:**
   - Use MongoDB for document store
   - Better fit for unstructured data

4. **Graph Queries:**
   - Use Neo4j for graph databases
   - CockroachDB not optimized for this

---

## 3. SQL and Query Language

### PostgreSQL Compatibility

**Supported Features:**
```sql
-- CockroachDB supports most PostgreSQL SQL
-- Wire protocol compatible (use psql, pgAdmin, etc.)

-- Data types
INTEGER, BIGINT, DECIMAL, FLOAT
TEXT, VARCHAR, CHAR
TIMESTAMP, DATE, TIME, INTERVAL
BOOLEAN, BYTEA, UUID, JSONB, ARRAY

-- Standard SQL operations
SELECT, INSERT, UPDATE, DELETE
JOIN (INNER, LEFT, RIGHT, FULL)
Subqueries, CTEs (WITH)
Window functions
Aggregations
```

### Basic Queries

**CRUD Operations:**
```sql
-- Create table
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email STRING UNIQUE NOT NULL,
    name STRING NOT NULL,
    created_at TIMESTAMP DEFAULT now()
);

-- Insert data
INSERT INTO users (email, name)
VALUES ('alice@example.com', 'Alice');

-- Insert multiple rows
INSERT INTO users (email, name) VALUES
    ('bob@example.com', 'Bob'),
    ('carol@example.com', 'Carol');

-- Select data
SELECT * FROM users WHERE email = 'alice@example.com';

-- Update data
UPDATE users
SET name = 'Alice Smith'
WHERE email = 'alice@example.com';

-- Delete data
DELETE FROM users WHERE id = '123e4567-e89b-12d3-a456-426614174000';
```

**Joins:**
```sql
-- Inner join
SELECT u.name, o.total
FROM users u
INNER JOIN orders o ON u.id = o.user_id
WHERE o.status = 'completed';

-- Left join
SELECT u.name, COUNT(o.id) AS order_count
FROM users u
LEFT JOIN orders o ON u.id = o.user_id
GROUP BY u.id, u.name;

-- Complex join with aggregation
SELECT
    u.name,
    COUNT(DISTINCT o.id) AS total_orders,
    SUM(o.total) AS total_spent
FROM users u
LEFT JOIN orders o ON u.id = o.user_id
WHERE u.created_at > '2024-01-01'
GROUP BY u.id, u.name
HAVING COUNT(o.id) > 5
ORDER BY total_spent DESC;
```

### Advanced SQL Features

**Common Table Expressions (CTEs):**
```sql
-- Recursive CTE for hierarchies
WITH RECURSIVE employee_hierarchy AS (
    -- Base case: top-level employees
    SELECT id, name, manager_id, 1 AS level
    FROM employees
    WHERE manager_id IS NULL

    UNION ALL

    -- Recursive case: employees with managers
    SELECT e.id, e.name, e.manager_id, eh.level + 1
    FROM employees e
    INNER JOIN employee_hierarchy eh ON e.manager_id = eh.id
)
SELECT * FROM employee_hierarchy
ORDER BY level, name;

-- Multiple CTEs
WITH
    active_users AS (
        SELECT * FROM users WHERE last_login > now() - INTERVAL '30 days'
    ),
    high_value_orders AS (
        SELECT * FROM orders WHERE total > 1000
    )
SELECT
    au.name,
    COUNT(hvo.id) AS high_value_order_count
FROM active_users au
LEFT JOIN high_value_orders hvo ON au.id = hvo.user_id
GROUP BY au.id, au.name;
```

**Window Functions:**
```sql
-- Running totals
SELECT
    date,
    amount,
    SUM(amount) OVER (ORDER BY date) AS running_total
FROM transactions
ORDER BY date;

-- Ranking
SELECT
    name,
    salary,
    department,
    RANK() OVER (PARTITION BY department ORDER BY salary DESC) AS dept_rank
FROM employees;

-- Moving average
SELECT
    date,
    revenue,
    AVG(revenue) OVER (
        ORDER BY date
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ) AS moving_avg_7day
FROM daily_revenue;
```

**JSONB Operations:**
```sql
-- Create table with JSONB
CREATE TABLE events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    event_type STRING,
    data JSONB,
    created_at TIMESTAMP DEFAULT now()
);

-- Insert JSONB data
INSERT INTO events (event_type, data) VALUES
    ('user_signup', '{"email": "alice@example.com", "plan": "premium"}'),
    ('purchase', '{"product_id": "123", "amount": 99.99, "currency": "USD"}');

-- Query JSONB fields
SELECT * FROM events
WHERE data->>'event_type' = 'purchase';

-- Extract and aggregate JSONB
SELECT
    data->>'plan' AS plan,
    COUNT(*) AS signup_count
FROM events
WHERE event_type = 'user_signup'
GROUP BY data->>'plan';

-- Index on JSONB field
CREATE INDEX ON events ((data->>'plan'));
```

**Array Operations:**
```sql
-- Create table with array
CREATE TABLE users (
    id UUID PRIMARY KEY,
    name STRING,
    tags STRING[]
);

-- Insert arrays
INSERT INTO users (id, name, tags) VALUES
    (gen_random_uuid(), 'Alice', ARRAY['admin', 'developer']),
    (gen_random_uuid(), 'Bob', ARRAY['user', 'beta-tester']);

-- Query arrays
SELECT * FROM users WHERE 'admin' = ANY(tags);

-- Array functions
SELECT name, array_length(tags, 1) AS tag_count
FROM users;
```

### Query Optimization

**EXPLAIN and EXPLAIN ANALYZE:**
```sql
-- Show query plan
EXPLAIN SELECT * FROM users WHERE email = 'alice@example.com';

-- Show execution statistics
EXPLAIN ANALYZE SELECT * FROM users WHERE email = 'alice@example.com';

-- Detailed plan with costs
EXPLAIN (VERBOSE)
SELECT u.name, COUNT(o.id)
FROM users u
LEFT JOIN orders o ON u.id = o.user_id
GROUP BY u.id, u.name;

-- Look for:
-- - Index usage (scan vs seek)
-- - Distribution plan (which nodes involved)
-- - Join algorithms
-- - Row estimates vs actual
```

---

## 4. Schema Design

### Primary Keys

**UUID Primary Keys (Recommended):**
```sql
-- ✅ GOOD: UUID primary keys prevent hotspots
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email STRING UNIQUE NOT NULL,
    name STRING NOT NULL
);

-- Best for distributed systems
-- Avoids sequential key hotspot issues
```

**Sequential IDs (Avoid for High-Write Tables):**
```sql
-- ❌ BAD: Auto-incrementing can cause hotspots
CREATE TABLE users (
    id INT PRIMARY KEY DEFAULT unique_rowid(),
    email STRING
);

-- All writes go to the same range (last range)
-- Creates write bottleneck
```

**Composite Primary Keys:**
```sql
-- Good for time-series or partitioned data
CREATE TABLE metrics (
    region STRING,
    timestamp TIMESTAMP,
    metric_name STRING,
    value FLOAT,
    PRIMARY KEY (region, timestamp, metric_name)
);

-- Distributes data by region
-- Efficient range scans by timestamp
```

### Data Types

**Recommended Types:**
```sql
-- ✅ Use appropriate types
CREATE TABLE products (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Strings
    name STRING NOT NULL,              -- Variable length
    sku STRING(20) NOT NULL,           -- Fixed max length

    -- Numbers
    price DECIMAL(10, 2) NOT NULL,     -- Exact decimal
    quantity INT NOT NULL,             -- Integer
    weight FLOAT,                      -- Approximate

    -- Dates
    created_at TIMESTAMP DEFAULT now(),
    shipped_at TIMESTAMPTZ,            -- With timezone

    -- Boolean
    active BOOLEAN DEFAULT true,

    -- Binary
    image_data BYTEA,

    -- JSON
    metadata JSONB,

    -- Arrays
    tags STRING[]
);
```

### Table Design Patterns

**Denormalization for Performance:**
```sql
-- ❌ Highly normalized (many joins)
CREATE TABLE orders (
    id UUID PRIMARY KEY,
    user_id UUID NOT NULL,
    status STRING
);

CREATE TABLE users (
    id UUID PRIMARY KEY,
    name STRING,
    email STRING
);

-- ✅ Denormalized (fewer joins)
CREATE TABLE orders (
    id UUID PRIMARY KEY,
    user_id UUID NOT NULL,
    user_email STRING NOT NULL,  -- Denormalized
    user_name STRING,             -- Denormalized
    status STRING
);

-- Trade-off: Faster reads, more complex writes
-- Use for frequently accessed data
```

**Partitioning Tables:**
```sql
-- Partition by range (time-series data)
CREATE TABLE events (
    id UUID DEFAULT gen_random_uuid(),
    event_type STRING,
    data JSONB,
    created_at TIMESTAMP NOT NULL,
    PRIMARY KEY (created_at, id)
) PARTITION BY RANGE (created_at);

-- Create partitions
CREATE TABLE events_2024_01 PARTITION OF events
    FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');

CREATE TABLE events_2024_02 PARTITION OF events
    FOR VALUES FROM ('2024-02-01') TO ('2024-03-01');

-- Automatically routes data to correct partition
-- Efficient for time-based queries and data retention
```

**Interleaved Tables (Deprecated in v21.1+):**
```sql
-- Use foreign key with ON DELETE CASCADE instead
CREATE TABLE users (
    id UUID PRIMARY KEY,
    name STRING
);

CREATE TABLE orders (
    id UUID PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    total DECIMAL(10, 2)
);

-- Co-locate related data for performance
```

### Schema Evolution

**Adding Columns:**
```sql
-- Add nullable column (fast)
ALTER TABLE users ADD COLUMN phone STRING;

-- Add column with default (slower in older versions)
ALTER TABLE users ADD COLUMN country STRING DEFAULT 'US';

-- Add NOT NULL column (requires default)
ALTER TABLE users ADD COLUMN verified BOOLEAN NOT NULL DEFAULT false;
```

**Modifying Columns:**
```sql
-- Change column type (may require rewrite)
ALTER TABLE users ALTER COLUMN phone TYPE STRING(20);

-- Set default value
ALTER TABLE users ALTER COLUMN country SET DEFAULT 'US';

-- Drop default
ALTER TABLE users ALTER COLUMN country DROP DEFAULT;

-- Rename column
ALTER TABLE users RENAME COLUMN phone TO phone_number;
```

**Dropping Columns:**
```sql
-- Drop column
ALTER TABLE users DROP COLUMN phone;

-- Drop multiple columns
ALTER TABLE users
    DROP COLUMN phone,
    DROP COLUMN fax;
```

---

## 5. Indexes and Constraints

### Index Types

**Primary Index:**
```sql
-- Automatically created with PRIMARY KEY
CREATE TABLE users (
    id UUID PRIMARY KEY,  -- Creates primary index
    email STRING
);
```

**Secondary Index:**
```sql
-- Single column index
CREATE INDEX ON users (email);

-- Named index
CREATE INDEX idx_users_email ON users (email);

-- Unique index
CREATE UNIQUE INDEX ON users (email);

-- Index with included columns (covering index)
CREATE INDEX ON users (email) STORING (name, created_at);
-- Query can be satisfied entirely from index
```

**Composite Index:**
```sql
-- Multi-column index
CREATE INDEX ON orders (user_id, created_at);

-- Order matters! Supports:
-- WHERE user_id = 'x'
-- WHERE user_id = 'x' AND created_at > 'y'
-- Does NOT efficiently support:
-- WHERE created_at > 'y' alone
```

**Partial Index:**
```sql
-- Index only rows matching condition
CREATE INDEX ON orders (created_at)
WHERE status = 'pending';

-- Smaller index, faster for specific queries
SELECT * FROM orders
WHERE status = 'pending' AND created_at > now() - INTERVAL '24 hours';
```

**Expression Index:**
```sql
-- Index on computed expression
CREATE INDEX ON users (lower(email));

-- Efficient query
SELECT * FROM users WHERE lower(email) = 'alice@example.com';

-- JSON expression index
CREATE INDEX ON events ((data->>'user_id'));
```

**Inverted Index (JSONB, Arrays):**
```sql
-- Index JSONB fields
CREATE INVERTED INDEX ON events (data);

-- Efficient JSONB queries
SELECT * FROM events WHERE data @> '{"status": "active"}';

-- Index arrays
CREATE INVERTED INDEX ON users (tags);

-- Efficient array queries
SELECT * FROM users WHERE tags @> ARRAY['admin'];
```

### Constraints

**Primary Key:**
```sql
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email STRING
);

-- Composite primary key
CREATE TABLE metrics (
    timestamp TIMESTAMP,
    sensor_id STRING,
    value FLOAT,
    PRIMARY KEY (sensor_id, timestamp)
);
```

**Unique Constraint:**
```sql
-- Single column
CREATE TABLE users (
    id UUID PRIMARY KEY,
    email STRING UNIQUE NOT NULL
);

-- Multiple columns (composite unique)
CREATE TABLE products (
    id UUID PRIMARY KEY,
    sku STRING,
    region STRING,
    UNIQUE (sku, region)
);
```

**Foreign Key:**
```sql
CREATE TABLE users (
    id UUID PRIMARY KEY,
    name STRING
);

CREATE TABLE orders (
    id UUID PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES users(id),
    total DECIMAL(10, 2)
);

-- With cascade delete
CREATE TABLE orders (
    id UUID PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    total DECIMAL(10, 2)
);

-- With restrict (prevent deletion)
CREATE TABLE orders (
    id UUID PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE RESTRICT,
    total DECIMAL(10, 2)
);
```

**Check Constraint:**
```sql
CREATE TABLE products (
    id UUID PRIMARY KEY,
    name STRING NOT NULL,
    price DECIMAL(10, 2) CHECK (price > 0),
    quantity INT CHECK (quantity >= 0),
    discount FLOAT CHECK (discount >= 0 AND discount <= 1)
);

-- Named constraint
ALTER TABLE products
ADD CONSTRAINT positive_price CHECK (price > 0);
```

**NOT NULL:**
```sql
CREATE TABLE users (
    id UUID PRIMARY KEY,
    email STRING NOT NULL,
    name STRING NOT NULL,
    phone STRING  -- nullable
);
```

### Index Management

**List Indexes:**
```sql
-- Show all indexes
SHOW INDEXES FROM users;

-- Show indexes with details
SELECT * FROM [SHOW INDEXES FROM users];
```

**Drop Index:**
```sql
DROP INDEX idx_users_email;

-- Drop if exists
DROP INDEX IF EXISTS idx_users_email;
```

**Index Recommendations:**
```sql
-- CockroachDB provides index recommendations
-- Check EXPLAIN output for suggestions

EXPLAIN SELECT * FROM orders WHERE user_id = 'xxx' AND created_at > 'yyy';
-- May suggest: CREATE INDEX ON orders (user_id, created_at);
```

---

## 6. Performance Optimization

### Query Optimization

**Use Indexes Effectively:**
```sql
-- ❌ BAD: Full table scan
SELECT * FROM users WHERE LOWER(email) = 'alice@example.com';

-- ✅ GOOD: Create expression index
CREATE INDEX ON users (LOWER(email));
SELECT * FROM users WHERE LOWER(email) = 'alice@example.com';
```

**Avoid SELECT *:**
```sql
-- ❌ BAD: Fetches all columns
SELECT * FROM users WHERE id = 'xxx';

-- ✅ GOOD: Fetch only needed columns
SELECT id, name, email FROM users WHERE id = 'xxx';

-- Even better with covering index
CREATE INDEX ON users (id) STORING (name, email);
```

**Batch Operations:**
```sql
-- ❌ BAD: Multiple round trips
INSERT INTO users (id, name) VALUES (gen_random_uuid(), 'Alice');
INSERT INTO users (id, name) VALUES (gen_random_uuid(), 'Bob');
INSERT INTO users (id, name) VALUES (gen_random_uuid(), 'Carol');

-- ✅ GOOD: Single batch insert
INSERT INTO users (id, name) VALUES
    (gen_random_uuid(), 'Alice'),
    (gen_random_uuid(), 'Bob'),
    (gen_random_uuid(), 'Carol');

-- ✅ GOOD: Batch update
UPDATE users
SET updated_at = now()
WHERE id IN ('id1', 'id2', 'id3');
```

**Use UPSERT:**
```sql
-- Efficient insert or update
UPSERT INTO users (id, name, email) VALUES
    ('id1', 'Alice', 'alice@example.com'),
    ('id2', 'Bob', 'bob@example.com');

-- Equivalent to:
INSERT INTO users (id, name, email) VALUES (...)
ON CONFLICT (id) DO UPDATE SET name = excluded.name, email = excluded.email;
```

### Connection Pooling

**Recommended Settings:**
```python
# Python example
import psycopg2
from psycopg2 import pool

# Create connection pool
connection_pool = pool.SimpleConnectionPool(
    minconn=5,
    maxconn=20,
    host='cockroach-lb.example.com',
    port=26257,
    database='mydb',
    user='myuser',
    password='mypassword'
)

# Get connection from pool
conn = connection_pool.getconn()

try:
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE id = %s", (user_id,))
    # ... process results
finally:
    # Return connection to pool
    connection_pool.putconn(conn)
```

### Caching Strategies

**Application-Level Caching:**
```python
import redis
import psycopg2

class UserService:
    def __init__(self):
        self.redis = redis.Redis(host='localhost', port=6379)
        self.db = psycopg2.connect(...)

    def get_user(self, user_id):
        # Check cache first
        cached = self.redis.get(f"user:{user_id}")
        if cached:
            return json.loads(cached)

        # Query database
        cursor = self.db.cursor()
        cursor.execute(
            "SELECT id, name, email FROM users WHERE id = %s",
            (user_id,)
        )
        user = cursor.fetchone()

        if user:
            # Cache for 5 minutes
            self.redis.setex(
                f"user:{user_id}",
                300,
                json.dumps(user)
            )

        return user
```

### Statement Caching

**Prepared Statements:**
```python
# Reuse prepared statements
conn = psycopg2.connect(...)
cursor = conn.cursor()

# Prepare statement once
cursor.execute("PREPARE get_user AS SELECT * FROM users WHERE id = $1")

# Execute multiple times
cursor.execute("EXECUTE get_user(%s)", ('id1',))
cursor.execute("EXECUTE get_user(%s)", ('id2',))
cursor.execute("EXECUTE get_user(%s)", ('id3',))
```

### Monitoring Query Performance

**Slow Query Logging:**
```sql
-- Enable slow query logging
SET CLUSTER SETTING sql.log.slow_query.latency_threshold = '100ms';

-- View slow queries
SELECT * FROM crdb_internal.node_statement_statistics
WHERE service_lat > INTERVAL '100ms'
ORDER BY service_lat DESC
LIMIT 10;
```

---

## 7. Transactions and Concurrency

### Transaction Basics

**ACID Properties:**
```
CockroachDB provides full ACID guarantees:
- Atomicity: All or nothing
- Consistency: Constraints enforced
- Isolation: Serializable by default
- Durability: Replicated across nodes
```

**Basic Transaction:**
```sql
-- Begin transaction
BEGIN;

-- Execute statements
INSERT INTO accounts (id, balance) VALUES ('alice', 1000);
INSERT INTO accounts (id, balance) VALUES ('bob', 500);

-- Commit
COMMIT;

-- Or rollback
ROLLBACK;
```

### Isolation Levels

**Serializable (Default):**
```sql
-- Default isolation level
BEGIN;
SET TRANSACTION ISOLATION LEVEL SERIALIZABLE;

-- Guaranteed serializable execution
-- Strongest consistency guarantee
-- May retry on conflicts

COMMIT;
```

**Read Committed:**
```sql
-- Lower isolation (better performance)
BEGIN;
SET TRANSACTION ISOLATION LEVEL READ COMMITTED;

-- Reads only committed data
-- Less strict than serializable
-- Fewer retries

COMMIT;
```

### Handling Transaction Retries

**Python with psycopg2:**
```python
import psycopg2
from psycopg2 import errorcodes
import time

def run_transaction(conn, op):
    """Execute transaction with automatic retry logic."""
    max_retries = 3

    for attempt in range(max_retries):
        try:
            with conn.cursor() as cursor:
                # Execute transaction operations
                op(cursor)
                conn.commit()
                return  # Success

        except psycopg2.Error as e:
            # Check if retry error
            if e.pgcode == errorcodes.SERIALIZATION_FAILURE:
                conn.rollback()
                if attempt < max_retries - 1:
                    # Exponential backoff
                    time.sleep(0.1 * (2 ** attempt))
                    continue
                else:
                    raise
            else:
                # Non-retry error
                conn.rollback()
                raise

# Usage
def transfer_money(cursor):
    cursor.execute(
        "UPDATE accounts SET balance = balance - 100 WHERE id = 'alice'"
    )
    cursor.execute(
        "UPDATE accounts SET balance = balance + 100 WHERE id = 'bob'"
    )

conn = psycopg2.connect(...)
run_transaction(conn, transfer_money)
```

**Transaction Retry Loop (Application Level):**
```python
from cockroachdb.sqlalchemy import run_transaction
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

engine = create_engine('cockroachdb://user:pass@host:26257/mydb')
Session = sessionmaker(bind=engine)

def transfer_funds(session, from_id, to_id, amount):
    """Transfer money between accounts."""
    # Debit
    session.execute(
        "UPDATE accounts SET balance = balance - :amount WHERE id = :id",
        {"amount": amount, "id": from_id}
    )
    # Credit
    session.execute(
        "UPDATE accounts SET balance = balance + :amount WHERE id = :id",
        {"amount": amount, "id": to_id}
    )

# Automatic retry handling
run_transaction(
    Session,
    lambda session: transfer_funds(session, 'alice', 'bob', 100)
)
```

### Locking

**SELECT FOR UPDATE:**
```sql
-- Explicit row locking
BEGIN;

SELECT * FROM accounts
WHERE id = 'alice'
FOR UPDATE;

-- Locked row until commit
UPDATE accounts
SET balance = balance - 100
WHERE id = 'alice';

COMMIT;
```

**Optimistic Locking (Version Column):**
```sql
-- Add version column
ALTER TABLE accounts ADD COLUMN version INT NOT NULL DEFAULT 0;

-- Update with version check
BEGIN;

-- Read current state
SELECT id, balance, version
FROM accounts
WHERE id = 'alice';
-- Returns: alice, 1000, 5

-- Update with version check
UPDATE accounts
SET balance = 900, version = version + 1
WHERE id = 'alice' AND version = 5;

-- If 0 rows updated, concurrent modification occurred
-- Retry transaction

COMMIT;
```

### Savepoints

**Using Savepoints:**
```sql
BEGIN;

INSERT INTO users (id, name) VALUES ('id1', 'Alice');

-- Create savepoint
SAVEPOINT sp1;

INSERT INTO users (id, name) VALUES ('id2', 'Bob');

-- Rollback to savepoint (undoes Bob insert)
ROLLBACK TO SAVEPOINT sp1;

-- Release savepoint
RELEASE SAVEPOINT sp1;

COMMIT;
-- Only Alice is inserted
```

---

## 8. Multi-Region Configuration

### Multi-Region Overview

**Deployment Patterns:**
```
┌─────────────────────────────────────────────────────┐
│          Multi-Region Cluster                        │
├─────────────────────────────────────────────────────┤
│                                                      │
│  Region: us-east (Primary)                          │
│  ┌────────┐  ┌────────┐  ┌────────┐               │
│  │ Node 1 │  │ Node 2 │  │ Node 3 │               │
│  └────────┘  └────────┘  └────────┘               │
│                                                      │
│  Region: us-west                                     │
│  ┌────────┐  ┌────────┐  ┌────────┐               │
│  │ Node 4 │  │ Node 5 │  │ Node 6 │               │
│  └────────┘  └────────┘  └────────┘               │
│                                                      │
│  Region: eu-west                                     │
│  ┌────────┐  ┌────────┐  ┌────────┐               │
│  │ Node 7 │  │ Node 8 │  │ Node 9 │               │
│  └────────┘  └────────┘  └────────┘               │
└─────────────────────────────────────────────────────┘
```

### Setting Up Multi-Region

**Start Nodes in Different Regions:**
```bash
# Region: us-east
cockroach start \
  --locality=region=us-east,zone=us-east-1 \
  --advertise-addr=node1.us-east.example.com:26257 \
  --join=node1.us-east.example.com:26257,node4.us-west.example.com:26257 \
  --certs-dir=certs

# Region: us-west
cockroach start \
  --locality=region=us-west,zone=us-west-1 \
  --advertise-addr=node4.us-west.example.com:26257 \
  --join=node1.us-east.example.com:26257,node4.us-west.example.com:26257 \
  --certs-dir=certs

# Region: eu-west
cockroach start \
  --locality=region=eu-west,zone=eu-west-1 \
  --advertise-addr=node7.eu-west.example.com:26257 \
  --join=node1.us-east.example.com:26257,node4.us-west.example.com:26257 \
  --certs-dir=certs
```

**Initialize Multi-Region Database:**
```sql
-- Set cluster regions
ALTER DATABASE mydb SET PRIMARY REGION 'us-east';
ALTER DATABASE mydb ADD REGION 'us-west';
ALTER DATABASE mydb ADD REGION 'eu-west';

-- Check regions
SHOW REGIONS FROM DATABASE mydb;
```

### Survival Goals

**Zone Survival (Default):**
```sql
-- Survive availability zone failure
ALTER DATABASE mydb SURVIVE ZONE FAILURE;

-- Requires 3+ nodes per region
-- Replicas distributed across AZs in same region
```

**Region Survival:**
```sql
-- Survive entire region failure
ALTER DATABASE mydb SURVIVE REGION FAILURE;

-- Requires 3+ regions
-- Replicas distributed across regions
-- Higher latency for writes
```

### Table Locality

**REGIONAL BY ROW:**
```sql
-- Each row pinned to specific region
CREATE TABLE users (
    id UUID PRIMARY KEY,
    email STRING,
    region crdb_internal_region AS (
        CASE
            WHEN email LIKE '%.uk' THEN 'eu-west'
            WHEN email LIKE '%.jp' THEN 'ap-northeast'
            ELSE 'us-east'
        END
    ) STORED,
    name STRING
) LOCALITY REGIONAL BY ROW AS region;

-- Data stored close to user
-- Low-latency local reads
-- Higher latency cross-region reads
```

**REGIONAL BY TABLE:**
```sql
-- Entire table pinned to one region
ALTER TABLE products SET LOCALITY REGIONAL BY TABLE IN 'us-east';

-- All data in one region
-- Fast local reads
-- Slow reads from other regions
```

**GLOBAL:**
```sql
-- Read replicas in all regions
ALTER TABLE catalog SET LOCALITY GLOBAL;

-- Fast reads from any region
-- Slower writes (must reach consensus across regions)
-- Perfect for read-heavy reference data
```

### Geo-Partitioning

**Partition by Region:**
```sql
-- Create partitioned table
CREATE TABLE orders (
    id UUID DEFAULT gen_random_uuid(),
    user_id UUID,
    region STRING NOT NULL,
    total DECIMAL(10, 2),
    created_at TIMESTAMP DEFAULT now(),
    PRIMARY KEY (region, id)
) PARTITION BY LIST (region) (
    PARTITION us_orders VALUES IN ('us-east', 'us-west'),
    PARTITION eu_orders VALUES IN ('eu-west'),
    PARTITION asia_orders VALUES IN ('ap-northeast')
);

-- Configure partition zones
ALTER PARTITION us_orders OF TABLE orders
CONFIGURE ZONE USING constraints = '[+region=us-east]';

ALTER PARTITION eu_orders OF TABLE orders
CONFIGURE ZONE USING constraints = '[+region=eu-west]';

ALTER PARTITION asia_orders OF TABLE orders
CONFIGURE ZONE USING constraints = '[+region=ap-northeast]';
```

### Follow-the-Workload

**Automatic Leaseholder Movement:**
```sql
-- CockroachDB automatically moves leaseholders closer to queries
-- No configuration needed

-- Example: European users querying user table
-- Leaseholders automatically move to EU region
-- Reduces latency for EU users
```

---

## 9. Cluster Configuration

### Starting a Cluster

**Single-Node (Development):**
```bash
# Start single node
cockroach start-single-node \
  --insecure \
  --listen-addr=localhost:26257 \
  --http-addr=localhost:8080 \
  --store=path=/data/cockroach
```

**Multi-Node Production:**
```bash
# Node 1
cockroach start \
  --certs-dir=certs \
  --advertise-addr=node1.example.com:26257 \
  --join=node1.example.com:26257,node2.example.com:26257,node3.example.com:26257 \
  --cache=25% \
  --max-sql-memory=25% \
  --store=path=/data/cockroach

# Node 2
cockroach start \
  --certs-dir=certs \
  --advertise-addr=node2.example.com:26257 \
  --join=node1.example.com:26257,node2.example.com:26257,node3.example.com:26257 \
  --cache=25% \
  --max-sql-memory=25% \
  --store=path=/data/cockroach

# Node 3
cockroach start \
  --certs-dir=certs \
  --advertise-addr=node3.example.com:26257 \
  --join=node1.example.com:26257,node2.example.com:26257,node3.example.com:26257 \
  --cache=25% \
  --max-sql-memory=25% \
  --store=path=/data/cockroach

# Initialize cluster (run once)
cockroach init --certs-dir=certs --host=node1.example.com:26257
```

### Cluster Settings

**View Cluster Settings:**
```sql
-- Show all settings
SHOW CLUSTER SETTINGS;

-- Show specific setting
SHOW CLUSTER SETTING sql.defaults.default_int_size;
```

**Important Cluster Settings:**
```sql
-- Default replication factor
SET CLUSTER SETTING kv.range_merge.queue_enabled = true;

-- Query timeout
SET CLUSTER SETTING sql.defaults.statement_timeout = '30s';

-- Max memory per query
SET CLUSTER SETTING sql.defaults.distsql_max_running_flows = 500;

-- Enable audit logging
SET CLUSTER SETTING sql.log.all_statements.enabled = true;

-- Slow query threshold
SET CLUSTER SETTING sql.log.slow_query.latency_threshold = '100ms';
```

### Node Management

**Add Node:**
```bash
# Start new node with --join flag
cockroach start \
  --certs-dir=certs \
  --advertise-addr=node4.example.com:26257 \
  --join=node1.example.com:26257,node2.example.com:26257,node3.example.com:26257 \
  --cache=25% \
  --max-sql-memory=25% \
  --store=path=/data/cockroach

# Cluster automatically rebalances data to new node
```

**Remove Node:**
```bash
# Decommission node (graceful removal)
cockroach node decommission 4 --certs-dir=certs --host=node1.example.com:26257

# Check decommission status
cockroach node status --certs-dir=certs --host=node1.example.com:26257

# Stop node after decommissioning complete
cockroach quit --certs-dir=certs --host=node4.example.com:26257
```

**Node Status:**
```bash
# Check cluster status
cockroach node status --certs-dir=certs --host=node1.example.com:26257

# Check node health
curl http://node1.example.com:8080/health
```

### Replication Zones

**Default Zone Configuration:**
```sql
-- View default zone config
SHOW ZONE CONFIGURATION FOR RANGE default;

-- Modify default replication factor
ALTER RANGE default CONFIGURE ZONE USING num_replicas = 5;
```

**Database Zone Configuration:**
```sql
-- Set zone config for database
ALTER DATABASE mydb CONFIGURE ZONE USING
  num_replicas = 3,
  gc.ttlseconds = 90000;  -- 25 hours GC
```

**Table Zone Configuration:**
```sql
-- Set zone config for table
ALTER TABLE users CONFIGURE ZONE USING
  num_replicas = 5,
  constraints = '[+region=us-east]',
  lease_preferences = '[[+region=us-east]]';

-- High-value table with more replicas
ALTER TABLE financial_records CONFIGURE ZONE USING
  num_replicas = 7,
  gc.ttlseconds = 604800;  -- 7 days GC
```

---

## 10. Data Distribution and Sharding

### Range Splits

**Automatic Splitting:**
```sql
-- CockroachDB automatically splits ranges at 64MB (default)
-- No manual intervention needed

-- Check range splits
SHOW RANGES FROM TABLE users;

-- Check range distribution
SELECT
    range_id,
    start_pretty,
    end_pretty,
    replicas,
    lease_holder
FROM crdb_internal.ranges
WHERE table_name = 'users';
```

**Manual Splitting:**
```sql
-- Split range at specific key
ALTER TABLE users SPLIT AT VALUES ('specific-uuid');

-- Split into N ranges
ALTER TABLE users SPLIT AT
    SELECT gen_random_uuid() FROM generate_series(1, 10);

-- Useful for pre-splitting before large import
```

### Load Balancing

**Automatic Rebalancing:**
```sql
-- CockroachDB automatically rebalances ranges
-- Moves ranges from overloaded to underutilized nodes

-- Check rebalancing status
SELECT * FROM crdb_internal.cluster_queries
WHERE query LIKE '%rebalance%';

-- View range distribution
SELECT
    store_id,
    COUNT(*) AS range_count
FROM crdb_internal.ranges
GROUP BY store_id
ORDER BY range_count DESC;
```

### Data Locality

**Zone Constraints:**
```sql
-- Pin data to specific region
ALTER TABLE users CONFIGURE ZONE USING
  constraints = '[+region=us-east]';

-- Pin to specific datacenter
ALTER TABLE eu_orders CONFIGURE ZONE USING
  constraints = '[+region=eu-west, +datacenter=dc1]';

-- Require diversity (no two replicas same AZ)
ALTER TABLE critical_data CONFIGURE ZONE USING
  constraints = '{+region=us-east: 1, +region=us-west: 1, +region=eu-west: 1}';
```

**Leaseholder Preferences:**
```sql
-- Prefer leaseholder in specific region
ALTER TABLE products CONFIGURE ZONE USING
  lease_preferences = '[[+region=us-east]]';

-- Multiple preferences (fallback order)
ALTER TABLE orders CONFIGURE ZONE USING
  lease_preferences = '[[+region=us-east], [+region=us-west], [+region=eu-west]]';
```

### Hotspot Prevention

**Use UUID Primary Keys:**
```sql
-- ✅ GOOD: Random distribution
CREATE TABLE events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    event_type STRING,
    created_at TIMESTAMP
);
```

**Hash Sharding:**
```sql
-- Distribute sequential keys across ranges
CREATE TABLE orders (
    id INT PRIMARY KEY,
    user_id UUID,
    total DECIMAL(10, 2),
    created_at TIMESTAMP
) WITH (experimental_autocommit);

-- Or use hash sharded index
CREATE INDEX ON orders (created_at) USING HASH WITH BUCKET_COUNT = 8;
```

---

## 11. Backup and Recovery

### Backup Types

**Full Backup:**
```sql
-- Backup entire cluster
BACKUP INTO 's3://bucket/backup?AWS_ACCESS_KEY_ID=xxx&AWS_SECRET_ACCESS_KEY=yyy';

-- Backup specific database
BACKUP DATABASE mydb INTO 's3://bucket/backup';

-- Backup specific tables
BACKUP TABLE users, orders INTO 's3://bucket/backup';

-- Backup with revision history (for point-in-time recovery)
BACKUP INTO 's3://bucket/backup' WITH revision_history;
```

**Incremental Backup:**
```sql
-- Full backup first
BACKUP INTO 's3://bucket/backup';

-- Incremental backups (append to same location)
BACKUP INTO LATEST IN 's3://bucket/backup';
BACKUP INTO LATEST IN 's3://bucket/backup';

-- CockroachDB tracks which backup contains which data
```

### Backup Destinations

**Cloud Storage:**
```sql
-- AWS S3
BACKUP INTO 's3://bucket/path?AWS_ACCESS_KEY_ID=xxx&AWS_SECRET_ACCESS_KEY=yyy';

-- Google Cloud Storage
BACKUP INTO 'gs://bucket/path?AUTH=specified&CREDENTIALS=base64-encoded-json';

-- Azure Blob Storage
BACKUP INTO 'azure://bucket/path?AZURE_ACCOUNT_NAME=xxx&AZURE_ACCOUNT_KEY=yyy';

-- Local/NFS (for testing only)
BACKUP INTO 'nodelocal://1/backup';
```

### Scheduled Backups

**Create Backup Schedule:**
```sql
-- Daily full backup, hourly incremental
CREATE SCHEDULE daily_backup
FOR BACKUP DATABASE mydb INTO 's3://bucket/backup'
RECURRING '@daily'
FULL BACKUP '@weekly'
WITH SCHEDULE OPTIONS first_run = 'now';

-- Check schedules
SHOW SCHEDULES;

-- Pause schedule
PAUSE SCHEDULE 123456789;

-- Resume schedule
RESUME SCHEDULE 123456789;

-- Drop schedule
DROP SCHEDULE 123456789;
```

### Restore Operations

**Restore Full Backup:**
```sql
-- Restore entire cluster
RESTORE FROM LATEST IN 's3://bucket/backup';

-- Restore specific database
RESTORE DATABASE mydb FROM LATEST IN 's3://bucket/backup';

-- Restore specific tables
RESTORE TABLE users FROM LATEST IN 's3://bucket/backup';
```

**Point-in-Time Restore:**
```sql
-- Restore to specific timestamp
RESTORE DATABASE mydb
FROM LATEST IN 's3://bucket/backup'
AS OF SYSTEM TIME '2024-02-06 10:00:00';

-- Restore to 1 hour ago
RESTORE DATABASE mydb
FROM LATEST IN 's3://bucket/backup'
AS OF SYSTEM TIME '-1h';
```

**Restore Options:**
```sql
-- Restore into different database
RESTORE DATABASE mydb FROM LATEST IN 's3://bucket/backup'
WITH into_db = 'mydb_restored';

-- Skip missing foreign keys
RESTORE TABLE orders FROM LATEST IN 's3://bucket/backup'
WITH skip_missing_foreign_keys;

-- Skip missing sequences
RESTORE DATABASE mydb FROM LATEST IN 's3://bucket/backup'
WITH skip_missing_sequences;
```

### Export/Import Data

**Export to CSV:**
```sql
-- Export table to CSV
EXPORT INTO CSV 's3://bucket/export/'
FROM SELECT * FROM users WHERE created_at > '2024-01-01';

-- Export with custom delimiter
EXPORT INTO CSV 's3://bucket/export/'
WITH delimiter = '|'
FROM SELECT * FROM users;
```

**Import CSV:**
```sql
-- Import CSV data
IMPORT INTO users (id, name, email)
CSV DATA ('s3://bucket/data/users.csv')
WITH skip = '1';  -- Skip header row

-- Import with custom delimiter
IMPORT INTO users (id, name, email)
CSV DATA ('s3://bucket/data/users.csv')
WITH delimiter = '|', skip = '1';
```

**Import from PostgreSQL:**
```sql
-- Dump from PostgreSQL
-- pg_dump -h postgres.example.com -U user -d mydb --format=plain > dump.sql

-- Import to CockroachDB
-- cockroach sql --url "postgresql://user@host:26257/mydb" < dump.sql
```

---

## 12. Security Best Practices

### Authentication

**Certificate-Based Authentication:**
```bash
# Create CA certificate
cockroach cert create-ca \
  --certs-dir=certs \
  --ca-key=my-safe-directory/ca.key

# Create node certificates
cockroach cert create-node \
  node1.example.com \
  node1-internal.example.com \
  localhost \
  127.0.0.1 \
  --certs-dir=certs \
  --ca-key=my-safe-directory/ca.key

# Create client certificate
cockroach cert create-client \
  myuser \
  --certs-dir=certs \
  --ca-key=my-safe-directory/ca.key

# Connect with certificate
cockroach sql \
  --certs-dir=certs \
  --host=node1.example.com:26257 \
  --user=myuser
```

**Password Authentication:**
```sql
-- Create user with password
CREATE USER alice WITH PASSWORD 'securePassword123!';

-- Change password
ALTER USER alice WITH PASSWORD 'newPassword456!';

-- Connect with password
-- cockroach sql --url "postgresql://alice:password@host:26257/mydb?sslmode=require"
```

### Authorization

**Create Roles:**
```sql
-- Create role
CREATE ROLE readonly;
CREATE ROLE readwrite;
CREATE ROLE admin;

-- Grant privileges to role
GRANT SELECT ON DATABASE mydb TO readonly;
GRANT SELECT, INSERT, UPDATE, DELETE ON DATABASE mydb TO readwrite;
GRANT ALL ON DATABASE mydb TO admin;

-- Assign role to user
GRANT readonly TO alice;
GRANT readwrite TO bob;
GRANT admin TO charlie;
```

**Database Privileges:**
```sql
-- Grant database access
GRANT CONNECT ON DATABASE mydb TO alice;

-- Grant schema access
GRANT USAGE ON SCHEMA public TO alice;

-- Grant table access
GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE users TO alice;

-- Grant all privileges
GRANT ALL ON TABLE users TO alice;

-- Revoke privileges
REVOKE INSERT ON TABLE users FROM alice;
```

**Table-Level Permissions:**
```sql
-- Grant specific column access
GRANT SELECT (id, name, email) ON users TO alice;

-- Grant with grant option
GRANT SELECT ON users TO alice WITH GRANT OPTION;

-- View grants
SHOW GRANTS ON TABLE users;

-- View user grants
SHOW GRANTS FOR alice;
```

### Encryption

**Encryption at Rest:**
```bash
# Enable encryption at rest
cockroach start \
  --enterprise-encryption=path=/data/cockroach,key=/keys/store.key,old-key=/keys/old-store.key \
  --certs-dir=certs \
  --advertise-addr=node1.example.com:26257
```

**Encryption in Transit (TLS):**
```bash
# All CockroachDB communication encrypted with TLS
# When using certificates (not --insecure)

# Verify TLS connection
cockroach sql \
  --certs-dir=certs \
  --host=node1.example.com:26257 \
  --execute="SHOW CLUSTER SETTING server.host_based_authentication.configuration"
```

### Network Security

**Host-Based Authentication:**
```sql
-- Configure HBA rules
SET CLUSTER SETTING server.host_based_authentication.configuration =
'# TYPE  DATABASE  USER      ADDRESS       METHOD
 host   all       all       10.0.0.0/8    cert
 host   all       all       0.0.0.0/0     password
 local  all       all                     cert';

-- Allow certificate auth from internal network
-- Require password from external
```

**Firewall Configuration:**
```bash
# Allow only necessary ports
# 26257: SQL/gRPC (inter-node and client)
# 8080: Admin UI (restrict to VPN/bastion)

# UFW example
ufw allow from 10.0.0.0/8 to any port 26257
ufw allow from 10.0.0.0/8 to any port 8080
```

### Audit Logging

**Enable Audit Logging:**
```sql
-- Enable cluster-wide audit log
SET CLUSTER SETTING sql.log.all_statements.enabled = true;

-- Enable for specific database
ALTER DATABASE mydb SET sql.log.all_statements.enabled = true;

-- Enable for specific table
ALTER TABLE sensitive_data SET sql.log.all_statements.enabled = true;

-- View audit logs
-- Logs written to cockroach-sql-audit.log
```

**Log Sensitive Operations:**
```sql
-- Log only sensitive operations
SET CLUSTER SETTING sql.log.admin_audit.enabled = true;

-- Logs:
-- - User/role changes
-- - Privilege changes
-- - Schema changes
```

---

## 13. Monitoring and Troubleshooting

### Admin UI

**Access Admin UI:**
```
http://node1.example.com:8080

Features:
- Cluster overview
- Node health
- Database/table metrics
- SQL queries
- Hardware metrics
- Hot ranges
```

### Metrics Collection

**Prometheus Integration:**
```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'cockroachdb'
    static_configs:
      - targets:
        - 'node1.example.com:8080'
        - 'node2.example.com:8080'
        - 'node3.example.com:8080'
    metrics_path: '/_status/vars'
```

**Key Metrics:**
```
Performance:
- sql_query_latency_p99
- sql_txn_latency_p99
- sql_conns (active connections)

Capacity:
- capacity_used_bytes
- capacity_available_bytes
- replicas_leaders (per node)

Replication:
- replicas_leaseholders
- ranges_unavailable
- ranges_underreplicated

Health:
- liveness_heartbeats
- sys_cpu_user_percent
- sys_memory_used_bytes
```

### Query Performance

**Statement Statistics:**
```sql
-- View slow queries
SELECT
    query,
    count,
    mean_latency,
    max_latency
FROM crdb_internal.node_statement_statistics
WHERE mean_latency > INTERVAL '100ms'
ORDER BY mean_latency DESC
LIMIT 20;

-- Reset statistics
SELECT crdb_internal.reset_sql_stats();
```

**Active Queries:**
```sql
-- Show running queries
SHOW QUERIES;

-- Show sessions
SHOW SESSIONS;

-- Cancel query
CANCEL QUERY '16b38f6e5c9d6c4d0000000000000001';

-- Cancel session
CANCEL SESSION '16b38f6e5c9d6c4d';
```

### Troubleshooting Tools

**Check Cluster Health:**
```bash
# Node status
cockroach node status --certs-dir=certs --host=node1.example.com:26257

# Health check endpoint
curl http://node1.example.com:8080/health

# Detailed status
curl http://node1.example.com:8080/_status/details/local
```

**Debug Commands:**
```bash
# Generate debug zip
cockroach debug zip /tmp/debug.zip \
  --certs-dir=certs \
  --host=node1.example.com:26257

# Contains:
# - Cluster events
# - Range information
# - Node logs
# - Schema dumps
```

**Query Plans:**
```sql
-- Explain query plan
EXPLAIN SELECT * FROM users WHERE email = 'alice@example.com';

-- Explain with costs
EXPLAIN (VERBOSE) SELECT * FROM users WHERE email = 'alice@example.com';

-- Analyze execution
EXPLAIN ANALYZE SELECT * FROM users WHERE email = 'alice@example.com';
```

### Common Issues

**High Latency:**
```sql
-- Check query statistics
SELECT query, mean_latency, count
FROM crdb_internal.node_statement_statistics
ORDER BY mean_latency DESC
LIMIT 10;

-- Check for missing indexes
-- Look at EXPLAIN output for "full scan"

-- Check cross-region queries
-- Consider table locality settings
```

**Replication Lag:**
```sql
-- Check underreplicated ranges
SELECT
    range_id,
    start_key,
    end_key
FROM crdb_internal.ranges
WHERE under_replicated = true;

-- Check for unavailable ranges
SELECT count(*) FROM crdb_internal.ranges WHERE unavailable = true;
```

**Out of Memory:**
```bash
# Increase memory limits
cockroach start \
  --cache=50% \           # Increase cache
  --max-sql-memory=50% \  # Increase SQL memory
  ..

# Or reduce per-query memory
SET CLUSTER SETTING sql.distsql.temp_storage.workmem = '128MB';
```

---

## 14. High Availability and Survivability

### Replication

**Replication Factor:**
```sql
-- Default: 3 replicas
SHOW ZONE CONFIGURATION FOR RANGE default;

-- Change default replication factor
ALTER RANGE default CONFIGURE ZONE USING num_replicas = 5;

-- Per-table replication
ALTER TABLE critical_data CONFIGURE ZONE USING num_replicas = 7;
```

**Replica Placement:**
```sql
-- Require replicas in specific regions
ALTER TABLE orders CONFIGURE ZONE USING
  constraints = '{+region=us-east: 1, +region=us-west: 1, +region=eu-west: 1}';

-- Prohibit certain placements
ALTER TABLE eu_only_data CONFIGURE ZONE USING
  constraints = '{+region=eu-west, -region=us-east, -region=us-west}';
```

### Failover

**Automatic Failover:**
```
Node failure detected (heartbeat timeout: ~5 seconds)
   ↓
Raft elects new leader for affected ranges (~1-2 seconds)
   ↓
New leaseholder serves requests
   ↓
Total failover time: ~10 seconds
```

**Testing Failover:**
```bash
# Simulate node failure
cockroach quit --host=node2.example.com:26257 --certs-dir=certs

# Cluster continues operating
# Queries automatically route to remaining nodes

# Check cluster status
cockroach node status --certs-dir=certs --host=node1.example.com:26257

# Restart node
cockroach start \
  --certs-dir=certs \
  --advertise-addr=node2.example.com:26257 \
  --join=node1.example.com:26257,node2.example.com:26257,node3.example.com:26257

# Data automatically rebalances
```

### Split-Brain Prevention

**Raft Quorum:**
```
3-node cluster:
- Quorum: 2 nodes minimum
- 1 node failure: Still operational
- 2 node failures: Cluster unavailable (prevents split-brain)

5-node cluster:
- Quorum: 3 nodes minimum
- 2 node failures: Still operational
- 3 node failures: Cluster unavailable
```

### Disaster Recovery

**Multi-Region Setup:**
```sql
-- Primary region failure scenario
-- Setup: 3 regions (us-east, us-west, eu-west)
-- Each region: 3 nodes
-- Replication: 3 replicas per range

-- Region survival configuration
ALTER DATABASE mydb SURVIVE REGION FAILURE;

-- If us-east region fails:
-- - Cluster continues operating
-- - Ranges with quorum in us-west + eu-west remain available
-- - Automatic failover to remaining regions
```

**Backup Strategy:**
```sql
-- Daily full backup with incremental
CREATE SCHEDULE disaster_recovery
FOR BACKUP DATABASE mydb INTO 's3://backup-bucket/prod'
RECURRING '@daily'
FULL BACKUP '@weekly'
WITH SCHEDULE OPTIONS first_run = 'now';

-- Also backup to different region/provider
CREATE SCHEDULE disaster_recovery_secondary
FOR BACKUP DATABASE mydb INTO 'gs://backup-bucket-secondary/prod'
RECURRING '@daily'
FULL BACKUP '@weekly';
```

---

## 15. Application Integration

### Python (psycopg2)

**Installation:**
```bash
pip install psycopg2-binary cockroachdb-python
```

**Basic Usage:**
```python
import psycopg2
from cockroachdb.sqlalchemy import run_transaction
from psycopg2 import errorcodes

# Connection string
conn_string = "postgresql://user:password@host:26257/mydb?sslmode=require"

# Connect
conn = psycopg2.connect(conn_string)

# Execute query
cursor = conn.cursor()
cursor.execute("SELECT * FROM users WHERE email = %s", ('alice@example.com',))
rows = cursor.fetchall()

# Close
cursor.close()
conn.close()
```

**With Connection Pool:**
```python
from psycopg2 import pool

# Create connection pool
connection_pool = pool.SimpleConnectionPool(
    minconn=5,
    maxconn=20,
    host='cockroach-lb.example.com',
    port=26257,
    database='mydb',
    user='myuser',
    password='mypassword',
    sslmode='require'
)

def get_user(user_id):
    conn = connection_pool.getconn()
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, name, email FROM users WHERE id = %s",
            (user_id,)
        )
        return cursor.fetchone()
    finally:
        connection_pool.putconn(conn)
```

**Transaction with Retry:**
```python
def run_transaction_with_retry(conn, op):
    max_retries = 3
    for attempt in range(max_retries):
        try:
            with conn.cursor() as cursor:
                op(cursor)
                conn.commit()
                return
        except psycopg2.Error as e:
            if e.pgcode == errorcodes.SERIALIZATION_FAILURE:
                conn.rollback()
                if attempt < max_retries - 1:
                    continue
                raise
            else:
                conn.rollback()
                raise

# Usage
def transfer_funds(cursor):
    cursor.execute("UPDATE accounts SET balance = balance - 100 WHERE id = 'alice'")
    cursor.execute("UPDATE accounts SET balance = balance + 100 WHERE id = 'bob'")

conn = psycopg2.connect(conn_string)
run_transaction_with_retry(conn, transfer_funds)
```

### Python (SQLAlchemy)

**Installation:**
```bash
pip install sqlalchemy cockroachdb
```

**Usage:**
```python
from sqlalchemy import create_engine, Column, String, Integer, DECIMAL
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.dialects.postgresql import UUID
from cockroachdb.sqlalchemy import run_transaction
import uuid

# Create engine
engine = create_engine(
    'cockroachdb://user:password@host:26257/mydb?sslmode=require',
    pool_size=10,
    max_overflow=20
)

Base = declarative_base()

# Define model
class User(Base):
    __tablename__ = 'users'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String, unique=True, nullable=False)
    name = Column(String, nullable=False)

# Create tables
Base.metadata.create_all(engine)

# Create session
Session = sessionmaker(bind=engine)

# Insert data
def create_user(session, email, name):
    user = User(email=email, name=name)
    session.add(user)

# Use transaction with automatic retry
run_transaction(
    Session,
    lambda session: create_user(session, 'alice@example.com', 'Alice')
)

# Query data
session = Session()
users = session.query(User).filter_by(email='alice@example.com').all()
session.close()
```

### Node.js (pg)

**Installation:**
```bash
npm install pg
```

**Usage:**
```javascript
const { Pool } = require('pg');

// Create connection pool
const pool = new Pool({
    host: 'cockroach-lb.example.com',
    port: 26257,
    database: 'mydb',
    user: 'myuser',
    password: 'mypassword',
    ssl: { rejectUnauthorized: false },
    max: 20,
    idleTimeoutMillis: 30000
});

// Query
async function getUser(userId) {
    const client = await pool.connect();
    try {
        const result = await client.query(
            'SELECT id, name, email FROM users WHERE id = $1',
            [userId]
        );
        return result.rows[0];
    } finally {
        client.release();
    }
}

// Transaction with retry
async function runTransaction(op, maxRetries = 3) {
    const client = await pool.connect();

    for (let attempt = 0; attempt < maxRetries; attempt++) {
        try {
            await client.query('BEGIN');
            await op(client);
            await client.query('COMMIT');
            return;
        } catch (err) {
            await client.query('ROLLBACK');
            if (err.code === '40001' && attempt < maxRetries - 1) {
                // Serialization failure, retry
                continue;
            }
            throw err;
        } finally {
            if (attempt === maxRetries - 1) {
                client.release();
            }
        }
    }
}

// Usage
runTransaction(async (client) => {
    await client.query(
        'UPDATE accounts SET balance = balance - 100 WHERE id = $1',
        ['alice']
    );
    await client.query(
        'UPDATE accounts SET balance = balance + 100 WHERE id = $1',
        ['bob']
    );
});
```

### Go

**Installation:**
```bash
go get github.com/lib/pq
```

**Usage:**
```go
package main

import (
    "database/sql"
    "fmt"
    "log"

    _ "github.com/lib/pq"
)

func main() {
    // Connection string
    connStr := "postgresql://user:password@host:26257/mydb?sslmode=require"

    // Open connection
    db, err := sql.Open("postgres", connStr)
    if err != nil {
        log.Fatal(err)
    }
    defer db.Close()

    // Configure pool
    db.SetMaxOpenConns(20)
    db.SetMaxIdleConns(5)

    // Query
    var id, name, email string
    err = db.QueryRow(
        "SELECT id, name, email FROM users WHERE id = $1",
        "user-id",
    ).Scan(&id, &name, &email)

    if err != nil {
        log.Fatal(err)
    }

    fmt.Printf("User: %s <%s>\n", name, email)
}

// Transaction with retry
func runTransaction(db *sql.DB, fn func(*sql.Tx) error) error {
    maxRetries := 3

    for attempt := 0; attempt < maxRetries; attempt++ {
        tx, err := db.Begin()
        if err != nil {
            return err
        }

        err = fn(tx)
        if err == nil {
            return tx.Commit()
        }

        tx.Rollback()

        // Check for serialization failure
        if pqErr, ok := err.(*pq.Error); ok && pqErr.Code == "40001" {
            if attempt < maxRetries - 1 {
                continue
            }
        }

        return err
    }

    return nil
}
```

---

## 16. Production Deployment

### Docker Deployment

**Single Node (Development):**
```yaml
# docker-compose.yml
version: '3.8'

services:
  cockroachdb:
    image: cockroachdb/cockroach:latest
    command: start-single-node --insecure
    ports:
      - "26257:26257"
      - "8080:8080"
    volumes:
      - cockroach_data:/cockroach/cockroach-data
    environment:
      - COCKROACH_DATABASE=mydb
      - COCKROACH_USER=myuser

volumes:
  cockroach_data:
```

**Multi-Node Cluster:**
```yaml
# docker-compose.yml
version: '3.8'

services:
  roach1:
    image: cockroachdb/cockroach:latest
    command: start --insecure --join=roach1,roach2,roach3
    ports:
      - "26257:26257"
      - "8080:8080"
    volumes:
      - roach1_data:/cockroach/cockroach-data
    networks:
      - cockroach_network

  roach2:
    image: cockroachdb/cockroach:latest
    command: start --insecure --join=roach1,roach2,roach3
    ports:
      - "26258:26257"
      - "8081:8080"
    volumes:
      - roach2_data:/cockroach/cockroach-data
    networks:
      - cockroach_network

  roach3:
    image: cockroachdb/cockroach:latest
    command: start --insecure --join=roach1,roach2,roach3
    ports:
      - "26259:26257"
      - "8082:8080"
    volumes:
      - roach3_data:/cockroach/cockroach-data
    networks:
      - cockroach_network

  init:
    image: cockroachdb/cockroach:latest
    command: init --insecure --host=roach1
    depends_on:
      - roach1
    networks:
      - cockroach_network

volumes:
  roach1_data:
  roach2_data:
  roach3_data:

networks:
  cockroach_network:
```

### Kubernetes Deployment

**StatefulSet:**
```yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: cockroachdb
spec:
  serviceName: cockroachdb
  replicas: 3
  selector:
    matchLabels:
      app: cockroachdb
  template:
    metadata:
      labels:
        app: cockroachdb
    spec:
      containers:
      - name: cockroachdb
        image: cockroachdb/cockroach:latest
        command:
          - "/cockroach/cockroach"
          - "start"
          - "--logtostderr"
          - "--certs-dir=/cockroach/cockroach-certs"
          - "--advertise-host=$(POD_NAME).cockroachdb"
          - "--http-addr=0.0.0.0"
          - "--join=cockroachdb-0.cockroachdb,cockroachdb-1.cockroachdb,cockroachdb-2.cockroachdb"
          - "--cache=25%"
          - "--max-sql-memory=25%"
        env:
        - name: POD_NAME
          valueFrom:
            fieldRef:
              fieldPath: metadata.name
        ports:
        - containerPort: 26257
          name: grpc
        - containerPort: 8080
          name: http
        volumeMounts:
        - name: datadir
          mountPath: /cockroach/cockroach-data
        - name: certs
          mountPath: /cockroach/cockroach-certs
        resources:
          requests:
            memory: "8Gi"
            cpu: "2"
          limits:
            memory: "16Gi"
            cpu: "4"
      volumes:
      - name: certs
        secret:
          secretName: cockroachdb-certs
  volumeClaimTemplates:
  - metadata:
      name: datadir
    spec:
      accessModes: ["ReadWriteOnce"]
      storageClassName: fast-ssd
      resources:
        requests:
          storage: 100Gi
---
apiVersion: v1
kind: Service
metadata:
  name: cockroachdb-public
spec:
  type: LoadBalancer
  ports:
  - port: 26257
    targetPort: 26257
    name: grpc
  - port: 8080
    targetPort: 8080
    name: http
  selector:
    app: cockroachdb
---
apiVersion: v1
kind: Service
metadata:
  name: cockroachdb
spec:
  clusterIP: None
  ports:
  - port: 26257
    targetPort: 26257
    name: grpc
  - port: 8080
    targetPort: 8080
    name: http
  selector:
    app: cockroachdb
```

### Cloud Deployments

**CockroachDB Serverless (Managed):**
```bash
# Create free cluster at cockroachlabs.cloud
# Connection string provided:
postgresql://user:password@cluster-name.cockroachdb.cloud:26257/defaultdb?sslmode=require

# Benefits:
# - Fully managed
# - Auto-scaling
# - Built-in backups
# - No infrastructure management
```

**CockroachDB Dedicated (Managed):**
```bash
# Enterprise-grade managed service
# Custom hardware
# Multi-region support
# VPC peering
# 99.99% SLA
```

### Load Balancer Configuration

**HAProxy:**
```haproxy
# haproxy.cfg
global
    log stdout local0

defaults
    mode tcp
    timeout connect 10s
    timeout client 1m
    timeout server 1m
    option clitcpka

listen cockroachdb
    bind :26257
    mode tcp
    balance roundrobin
    option httpchk GET /health?ready=1
    server node1 node1.example.com:26257 check port 8080
    server node2 node2.example.com:26257 check port 8080
    server node3 node3.example.com:26257 check port 8080

listen cockroachdb-ui
    bind :8080
    mode tcp
    balance roundrobin
    server node1 node1.example.com:8080 check
    server node2 node2.example.com:8080 check
    server node3 node3.example.com:8080 check
```

---

## 17. Scaling Strategies

### Vertical Scaling

**Resource Guidelines:**
```
Development:
- 2 CPU cores
- 4GB RAM
- 50GB SSD

Small Production:
- 4 CPU cores
- 16GB RAM
- 200GB SSD

Medium Production:
- 8-16 CPU cores
- 32-64GB RAM
- 500GB-1TB NVMe SSD

Large Production:
- 16-32+ CPU cores
- 128-256GB+ RAM
- 2TB+ NVMe SSD

Memory allocation:
- Cache: 25-30% of RAM
- SQL Memory: 25-30% of RAM
- OS/Other: 40-50% of RAM
```

### Horizontal Scaling

**Adding Nodes:**
```bash
# Add node to existing cluster
cockroach start \
  --certs-dir=certs \
  --advertise-addr=node4.example.com:26257 \
  --join=node1.example.com:26257,node2.example.com:26257,node3.example.com:26257 \
  --cache=25% \
  --max-sql-memory=25%

# Cluster automatically rebalances
# No downtime required
# Increased capacity and throughput
```

**Scaling Best Practices:**
```
Recommended cluster sizes:
- Minimum: 3 nodes (production)
- Small: 3-5 nodes
- Medium: 5-9 nodes
- Large: 9+ nodes

Scaling considerations:
- Add nodes in multiples of replication factor
- For 3x replication, add 3, 6, 9 nodes
- Odd numbers help with quorum

Performance scaling:
- Linear read scaling (add nodes)
- Write scaling (up to network limits)
- Each node added: +33% capacity (3x replication)
```

### Read Scaling

**Read Replicas:**
```sql
-- Configure follower reads
SET CLUSTER SETTING kv.closed_timestamp.target_duration = '1s';

-- Application uses follower reads
SET SESSION CHARACTERISTICS AS TRANSACTION READ ONLY;
SET TRANSACTION AS OF SYSTEM TIME follower_read_timestamp();

-- Or per-query
SELECT * FROM users AS OF SYSTEM TIME follower_read_timestamp();

-- Benefits:
-- - Reads from local replica (lower latency)
-- - Reduced load on leaseholder
-- - Slight staleness (1-5 seconds)
```

### Write Scaling

**Distribute Writes:**
```sql
-- Use UUID primary keys (distributes writes)
CREATE TABLE events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    event_type STRING,
    data JSONB
);

-- Hash sharded indexes
CREATE INDEX ON orders (created_at) USING HASH WITH BUCKET_COUNT = 16;

-- Pre-split tables for large bulk inserts
ALTER TABLE events SPLIT AT
SELECT gen_random_uuid() FROM generate_series(1, 100);
```

### Caching Layer

**Application-Level Caching:**
```python
import redis
from functools import wraps

class CachedDB:
    def __init__(self, db_pool):
        self.db = db_pool
        self.redis = redis.Redis(host='localhost', port=6379)

    def cached_query(self, key, query, params, ttl=300):
        # Check cache
        cached = self.redis.get(key)
        if cached:
            return json.loads(cached)

        # Query database
        conn = self.db.getconn()
        try:
            cursor = conn.cursor()
            cursor.execute(query, params)
            result = cursor.fetchall()

            # Cache result
            self.redis.setex(key, ttl, json.dumps(result))

            return result
        finally:
            self.db.putconn(conn)
```

---

## 18. Migration Strategies

### From PostgreSQL

**Schema Migration:**
```bash
# 1. Dump PostgreSQL schema
pg_dump -h postgres.example.com -U user -d mydb \
  --schema-only --no-owner --no-acl > schema.sql

# 2. Modify schema for CockroachDB compatibility
# - Change SERIAL to UUID or INT DEFAULT unique_rowid()
# - Remove unsupported features (e.g., triggers, custom types)
# - Add STORING clauses to indexes where beneficial

# 3. Import schema to CockroachDB
cockroach sql --url "postgresql://user@host:26257/mydb" < schema.sql
```

**Data Migration:**
```bash
# Option 1: COPY (for smaller datasets)
# Export from PostgreSQL
psql -h postgres.example.com -U user -d mydb \
  -c "\COPY users TO '/tmp/users.csv' CSV HEADER"

# Import to CockroachDB
cockroach sql --url "postgresql://user@host:26257/mydb" \
  --execute="IMPORT INTO users (id, name, email) CSV DATA ('file:///tmp/users.csv') WITH skip='1';"

# Option 2: Live migration with dual-write
# See dual-write example below
```

**Compatibility Checks:**
```sql
-- Check for unsupported features
-- CockroachDB does NOT support:
-- - Triggers
-- - Stored procedures (use application logic)
-- - Custom types (use built-in types)
-- - Partial foreign keys
-- - LISTEN/NOTIFY

-- Supported with differences:
-- - SERIAL → Use UUID or unique_rowid()
-- - Sequences → Supported but less efficient than UUIDs
-- - JSON → Use JSONB (no JSON type)
```

### From MySQL

**Schema Conversion:**
```sql
-- MySQL
CREATE TABLE users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    email VARCHAR(255) UNIQUE,
    name VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Convert to CockroachDB
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),  -- Or INT DEFAULT unique_rowid()
    email STRING UNIQUE,                             -- VARCHAR → STRING
    name STRING,
    created_at TIMESTAMP DEFAULT now()              -- CURRENT_TIMESTAMP → now()
);
```

### Dual-Write Migration

**Zero-Downtime Migration:**
```python
class DualWriteDB:
    """Write to both old and new database during migration."""

    def __init__(self, postgres_pool, cockroach_pool):
        self.pg = postgres_pool
        self.crdb = cockroach_pool
        self.migration_complete = False

    def create_user(self, email, name):
        # Write to primary (PostgreSQL)
        pg_conn = self.pg.getconn()
        try:
            cursor = pg_conn.cursor()
            cursor.execute(
                "INSERT INTO users (email, name) VALUES (%s, %s) RETURNING id",
                (email, name)
            )
            user_id = cursor.fetchone()[0]
            pg_conn.commit()
        except Exception as e:
            pg_conn.rollback()
            raise
        finally:
            self.pg.putconn(pg_conn)

        # Write to secondary (CockroachDB)
        crdb_conn = self.crdb.getconn()
        try:
            cursor = crdb_conn.cursor()
            cursor.execute(
                "INSERT INTO users (id, email, name) VALUES (%s, %s, %s)",
                (user_id, email, name)
            )
            crdb_conn.commit()
        except Exception as e:
            # Log error but don't fail
            logger.error(f"CockroachDB write failed: {e}")
        finally:
            self.crdb.putconn(crdb_conn)

        return user_id

    def get_user(self, user_id):
        # Read from CockroachDB after migration
        if self.migration_complete:
            conn = self.crdb.getconn()
            db_name = "CockroachDB"
        else:
            conn = self.pg.getconn()
            db_name = "PostgreSQL"

        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, email, name FROM users WHERE id = %s",
                (user_id,)
            )
            return cursor.fetchone()
        finally:
            if self.migration_complete:
                self.crdb.putconn(conn)
            else:
                self.pg.putconn(conn)

# Migration steps:
# 1. Deploy dual-write code
# 2. Backfill historical data to CockroachDB
# 3. Verify data consistency
# 4. Switch reads to CockroachDB (set migration_complete=True)
# 5. Remove PostgreSQL writes after monitoring period
```

### Data Validation

**Compare Data:**
```python
def validate_migration():
    """Compare PostgreSQL and CockroachDB data."""
    pg_conn = psycopg2.connect(pg_connection_string)
    crdb_conn = psycopg2.connect(crdb_connection_string)

    # Compare row counts
    pg_cursor = pg_conn.cursor()
    pg_cursor.execute("SELECT COUNT(*) FROM users")
    pg_count = pg_cursor.fetchone()[0]

    crdb_cursor = crdb_conn.cursor()
    crdb_cursor.execute("SELECT COUNT(*) FROM users")
    crdb_count = crdb_cursor.fetchone()[0]

    print(f"PostgreSQL: {pg_count} rows")
    print(f"CockroachDB: {crdb_count} rows")

    # Validate sample data
    pg_cursor.execute("SELECT id, email, name FROM users ORDER BY id LIMIT 100")
    pg_data = pg_cursor.fetchall()

    for row in pg_data:
        user_id, email, name = row
        crdb_cursor.execute(
            "SELECT email, name FROM users WHERE id = %s",
            (user_id,)
        )
        crdb_row = crdb_cursor.fetchone()

        if not crdb_row or crdb_row != (email, name):
            print(f"Mismatch for user {user_id}")

    pg_conn.close()
    crdb_conn.close()
```

---

## 19. Time Travel and Change Data Capture

### Time Travel Queries

**AS OF SYSTEM TIME:**
```sql
-- Query data as it existed at specific time
SELECT * FROM users
AS OF SYSTEM TIME '2024-02-06 10:00:00';

-- Query data 1 hour ago
SELECT * FROM users
AS OF SYSTEM TIME '-1h';

-- Query data at specific transaction
SELECT * FROM users
AS OF SYSTEM TIME '1612345678.0000000000';

-- Compare current vs historical data
SELECT
    current.id,
    current.balance AS current_balance,
    historical.balance AS balance_1h_ago,
    current.balance - historical.balance AS change
FROM accounts current
JOIN accounts AS OF SYSTEM TIME '-1h' historical
  ON current.id = historical.id;
```

**Use Cases:**
```sql
-- Audit: Who changed what and when?
SELECT * FROM orders
AS OF SYSTEM TIME '2024-02-06 09:00:00'
WHERE order_id = '123';

-- Recovery: Restore accidentally deleted data
INSERT INTO users
SELECT * FROM users AS OF SYSTEM TIME '-30m'
WHERE id = 'deleted-user-id';

-- Analysis: Historical trends
SELECT
    date_trunc('hour', timestamp) AS hour,
    COUNT(*) AS order_count
FROM orders AS OF SYSTEM TIME '-24h'
GROUP BY hour;
```

### Follower Reads

**Low-Latency Reads:**
```sql
-- Read from nearest replica (slight staleness)
SELECT * FROM products
AS OF SYSTEM TIME follower_read_timestamp();

-- Configure staleness tolerance
SET CLUSTER SETTING kv.closed_timestamp.target_duration = '1s';

-- Benefits:
-- - Read from local replica (lower latency)
-- - Reduced load on leaseholder
-- - 1-5 second staleness
```

**Application Integration:**
```python
def get_product_catalog():
    """Get product catalog with follower reads."""
    cursor.execute("""
        SELECT id, name, price, description
        FROM products
        AS OF SYSTEM TIME follower_read_timestamp()
        WHERE active = true
        ORDER BY name
    """)
    return cursor.fetchall()

# Use for:
# - Product catalogs
# - Static content
# - Analytics dashboards
# - Read-heavy workloads where slight staleness is acceptable
```

### Change Data Capture (CDC)

**Changefeeds:**
```sql
-- Create changefeed (Enterprise)
CREATE CHANGEFEED FOR TABLE users
INTO 'kafka://kafka.example.com:9092?topic_prefix=cockroach';

-- Multiple tables
CREATE CHANGEFEED FOR TABLE users, orders, products
INTO 'kafka://kafka.example.com:9092';

-- Cloud storage
CREATE CHANGEFEED FOR TABLE users
INTO 's3://bucket/changefeeds?AWS_ACCESS_KEY_ID=xxx&AWS_SECRET_ACCESS_KEY=yyy'
WITH format = json, diff;

-- With filters
CREATE CHANGEFEED FOR TABLE orders
INTO 'kafka://kafka.example.com:9092'
WHERE region = 'us-east' AND total > 1000;
```

**Changefeed Options:**
```sql
-- Include primary key in payload
CREATE CHANGEFEED FOR TABLE users
INTO 'kafka://kafka.example.com:9092'
WITH key_in_value;

-- Include before/after values
CREATE CHANGEFEED FOR TABLE users
INTO 'kafka://kafka.example.com:9092'
WITH diff;

-- Custom envelope format
CREATE CHANGEFEED FOR TABLE users
INTO 'kafka://kafka.example.com:9092'
WITH envelope = wrapped;

-- Avro format
CREATE CHANGEFEED FOR TABLE users
INTO 'kafka://kafka.example.com:9092'
WITH format = avro, confluent_schema_registry = 'http://schema-registry:8081';
```

**Manage Changefeeds:**
```sql
-- List changefeeds
SHOW CHANGEFEED JOBS;

-- Pause changefeed
PAUSE JOB 123456789;

-- Resume changefeed
RESUME JOB 123456789;

-- Cancel changefeed
CANCEL JOB 123456789;
```

**Use Cases:**
```
1. Real-time Analytics
   - Stream changes to data warehouse
   - Power dashboards with fresh data

2. Event-Driven Architecture
   - Trigger microservices on data changes
   - Event sourcing

3. Search Index Sync
   - Keep Elasticsearch/Algolia in sync
   - Update search indexes in real-time

4. Cache Invalidation
   - Invalidate Redis cache on updates
   - Keep distributed caches consistent

5. Data Replication
   - Replicate to other databases
   - Multi-datacenter sync
```

### GC TTL Configuration

**Control Historical Data Retention:**
```sql
-- Default GC: 25 hours
-- Time travel available for 25 hours

-- Increase retention for longer history
ALTER TABLE audit_logs CONFIGURE ZONE USING
  gc.ttlseconds = 604800;  -- 7 days

-- Decrease for less storage
ALTER TABLE temp_data CONFIGURE ZONE USING
  gc.ttlseconds = 3600;  -- 1 hour

-- Check current setting
SHOW ZONE CONFIGURATION FOR TABLE users;
```

---

## 20. Comparison with Other Databases

### CockroachDB vs. PostgreSQL

| Feature | CockroachDB | PostgreSQL |
|---------|-------------|------------|
| **Distribution** | Distributed by design | Single-node (or manual sharding) |
| **Scalability** | Horizontal (add nodes) | Vertical (bigger machine) |
| **High Availability** | Built-in automatic failover | Requires manual setup (Patroni, etc.) |
| **Consistency** | Serializable (default) | Read Committed (default) |
| **Multi-Region** | Native support | Requires complex setup |
| **SQL Compatibility** | PostgreSQL wire protocol | Full PostgreSQL |
| **Performance** | Excellent (distributed) | Excellent (single node) |
| **Best For** | Global apps, HA required | Single-region, full PG features |

### CockroachDB vs. MySQL

| Feature | CockroachDB | MySQL |
|---------|-------------|-------|
| **Distribution** | Native distributed | Single-node (or manual) |
| **Replication** | Automatic, synchronous | Manual, async/semi-sync |
| **Transactions** | Serializable | Read Committed/Repeatable Read |
| **Sharding** | Automatic | Manual (Vitess, ProxySQL) |
| **Cloud-Native** | Yes | Requires additional tools |
| **Use Cases** | Global applications | Traditional web apps |

### CockroachDB vs. Spanner

| Feature | CockroachDB | Google Spanner |
|---------|-------------|----------------|
| **Deployment** | Self-hosted or managed | Google Cloud only |
| **SQL** | PostgreSQL-compatible | Custom SQL dialect |
| **Pricing** | Open source + Enterprise | Per-node + storage |
| **Global** | Multi-region/multi-cloud | Google Cloud regions |
| **Maturity** | Mature (since 2015) | Mature (since 2012) |
| **Lock-in** | No vendor lock-in | Google Cloud |

### CockroachDB vs. MongoDB

| Feature | CockroachDB | MongoDB |
|---------|-------------|---------|
| **Data Model** | SQL (relational) | Document (NoSQL) |
| **Consistency** | Strong (serializable) | Eventual (by default) |
| **Transactions** | Multi-row, multi-table | Multi-document (4.0+) |
| **Queries** | SQL | MongoDB Query Language |
| **Schema** | Structured (with flexibility) | Schema-less |
| **Use Cases** | Structured data, ACID | Unstructured data, flexible |

### CockroachDB vs. Cassandra

| Feature | CockroachDB | Cassandra |
|---------|-------------|-----------|
| **Consistency** | Strong (default) | Tunable (eventual default) |
| **SQL** | Full SQL support | CQL (limited) |
| **Transactions** | Full ACID | Limited |
| **Writes** | Lower throughput | Very high throughput |
| **Reads** | Fast (with indexes) | Fast (with proper model) |
| **Use Cases** | General purpose | Write-heavy, time-series |

---

## 21. Production Checklist

### Pre-Deployment

**Infrastructure:**
- [ ] Hardware sized appropriately (CPU, RAM, storage)
- [ ] Minimum 3 nodes for production
- [ ] SSD/NVMe storage (not HDD)
- [ ] Network bandwidth sufficient (10Gbps+ for multi-region)
- [ ] Time synchronization configured (NTP)
- [ ] Load balancer configured (HAProxy, etc.)
- [ ] Monitoring system setup (Prometheus, Grafana)
- [ ] Backup storage configured (S3, GCS, etc.)

**Security:**
- [ ] TLS certificates generated and configured
- [ ] Certificate-based authentication for inter-node
- [ ] User authentication configured (password/cert)
- [ ] Database users created with appropriate roles
- [ ] Network firewalls configured
- [ ] Admin UI access restricted (VPN/bastion)
- [ ] Audit logging enabled

**Database Configuration:**
- [ ] Cluster initialized
- [ ] Databases created
- [ ] Replication factor configured
- [ ] Zone configurations set for critical tables
- [ ] Multi-region configured (if applicable)
- [ ] GC TTL configured appropriately
- [ ] Cluster settings optimized
- [ ] Connection pooling configured

**Schema:**
- [ ] Tables created with appropriate primary keys (UUIDs)
- [ ] Indexes created for common queries
- [ ] Foreign keys configured
- [ ] Constraints added (UNIQUE, CHECK, NOT NULL)
- [ ] Table locality configured (for multi-region)
- [ ] Initial data loaded

**Backup:**
- [ ] Scheduled backups configured
- [ ] Backup destination accessible
- [ ] Backup retention policy defined
- [ ] Restore procedure tested
- [ ] Backup monitoring alerts configured

### Post-Deployment

**Verification:**
- [ ] All nodes healthy and connected
- [ ] Replication working (check under-replicated ranges)
- [ ] Queries executing successfully
- [ ] Application can connect
- [ ] Load balancer distributing traffic
- [ ] Backups running successfully
- [ ] Monitoring dashboards populated
- [ ] Alerts firing correctly

**Operations:**
- [ ] Runbooks documented
  - Node failure recovery
  - Backup/restore procedures
  - Adding/removing nodes
  - Schema changes
  - Troubleshooting common issues
- [ ] On-call rotation established
- [ ] Incident response procedures documented
- [ ] Change management process defined

### Performance Tuning

**Query Performance:**
- [ ] Slow queries identified (statement statistics)
- [ ] Missing indexes created
- [ ] Query plans analyzed (EXPLAIN)
- [ ] Connection pooling optimized
- [ ] Prepared statements used where beneficial
- [ ] Batch operations used for bulk writes

**Cluster Performance:**
- [ ] Hot ranges identified and resolved
- [ ] Range splits optimized for workload
- [ ] Leaseholder preferences configured
- [ ] Follower reads enabled where appropriate
- [ ] Table locality optimized for access patterns

**Resource Utilization:**
- [ ] CPU usage reasonable (<80% avg)
- [ ] Memory usage stable
- [ ] Disk I/O not saturated
- [ ] Network bandwidth sufficient
- [ ] Connection count within limits

### Monitoring Metrics

**Critical Metrics:**
```
Cluster Health:
- [ ] Node liveness (all nodes alive)
- [ ] Ranges unavailable = 0
- [ ] Ranges underreplicated = 0

Performance:
- [ ] SQL query latency P99 < 100ms
- [ ] Transaction latency P99 < 200ms
- [ ] Connection count < 80% of max

Capacity:
- [ ] Disk usage < 80%
- [ ] Memory usage < 85%
- [ ] CPU usage < 80% (average)

Replication:
- [ ] Replication queue length < 100
- [ ] Raft log size reasonable

Backup:
- [ ] Scheduled backups succeeding
- [ ] Backup size monitored
```

### Ongoing Maintenance

**Daily:**
- [ ] Check cluster health dashboard
- [ ] Review alerts and anomalies
- [ ] Monitor disk space
- [ ] Verify backups completed
- [ ] Check slow query log

**Weekly:**
- [ ] Review performance metrics
- [ ] Analyze growth trends
- [ ] Check for CockroachDB updates
- [ ] Review and optimize slow queries
- [ ] Test alerts

**Monthly:**
- [ ] Test backup restore procedure
- [ ] Review and optimize indexes
- [ ] Capacity planning review
- [ ] Security audit
- [ ] Update documentation

**Quarterly:**
- [ ] Disaster recovery drill
- [ ] Performance tuning review
- [ ] Version upgrade planning
- [ ] Schema optimization review
- [ ] Team training on new features

### Upgrade Procedure

**Rolling Upgrade:**
```bash
# 1. Check release notes for breaking changes

# 2. Take full backup
BACKUP INTO 's3://bucket/pre-upgrade-backup';

# 3. Upgrade one node at a time
# Drain node
cockroach node drain <node-id> --host=<node>

# Stop node
systemctl stop cockroachdb

# Upgrade binary
# Replace cockroach binary with new version

# Start node
systemctl start cockroachdb

# Verify node joined cluster
cockroach node status

# 4. Repeat for remaining nodes

# 5. Finalize upgrade (if required)
SET CLUSTER SETTING version = '23.1';

# 6. Verify cluster health
cockroach node status
SHOW CLUSTER SETTING version
```

---

## References and Resources

### Official Documentation
- **CockroachDB Docs:** https://www.cockroachlabs.com/docs/
- **SQL Reference:** https://www.cockroachlabs.com/docs/stable/sql-statements
- **Architecture:** https://www.cockroachlabs.com/docs/stable/architecture/overview
- **Best Practices:** https://www.cockroachlabs.com/docs/stable/performance-best-practices-overview

### Learning Resources
- **Cockroach University:** https://university.cockroachlabs.com/
- **Interactive Tutorials:** https://www.cockroachlabs.com/docs/stable/tutorials
- **Blog:** https://www.cockroachlabs.com/blog/
- **Webinars:** https://www.cockroachlabs.com/webinars/

### Community
- **Forum:** https://forum.cockroachlabs.com/
- **GitHub:** https://github.com/cockroachdb/cockroach
- **Slack:** https://cockroachdb.slack.com/
- **Stack Overflow:** `[cockroachdb]` tag

### Tools
- **CockroachDB Cloud:** Fully managed service
- **Admin UI:** Built-in monitoring (port 8080)
- **DB Console:** Web-based management
- **cockroach CLI:** Command-line interface

### Books and Papers
- **"CockroachDB: The Definitive Guide"** (O'Reilly)
- **"Spanner" Paper** (Google, inspiration for CockroachDB)
- **"Raft Consensus Algorithm"** (Diego Ongaro)

---

**Document Maintenance:**
- Review quarterly for CockroachDB updates
- Update with new SQL features
- Add production patterns and lessons learned
- Test examples with latest version

**Last Updated:** February 2026
**Next Review:** May 2026

---

**End of CockroachDB Development Guidelines**
