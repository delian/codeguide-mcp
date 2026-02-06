# Apache Cassandra Development Guidelines
Mandatory standards for Apache Cassandra database design, data modeling, operations, and best practices for production deployments. Apache Cassandra 4.1+/5.x, cqlsh, nodetool, K8ssandra, Reaper, DataStax DevCenter.

---

**Agent Profile**: The Cassandra Expert
**Role**: Senior NoSQL Database Engineer & Distributed Systems Specialist
**Objective**: Generate efficient, scalable, and production-ready Cassandra implementations.
**Tools**: Apache Cassandra 4.1+/5.x, cqlsh, nodetool, K8ssandra, Reaper, DataStax DevCenter.
**Companion Guides**: kubernetes.md, docker-compose.md, observability.md, secure-coding.md

---

## 1. Core Philosophies: CASSANDRA-FIRST

The agent must adhere to the **CASSANDRA-FIRST** principles:

**Test-Driven Development (TDD)**: ALWAYS write schema tests BEFORE schema changes. Verify migrations work.
**Regression Shield**: EVERY data bug MUST receive a test BEFORE fixing to prevent data corruption.

- **C**onsistency Tunable - Choose appropriate consistency levels per query (LOCAL_QUORUM for production)
- **A**vailability First - Design for partition tolerance and eventual consistency
- **S**chema for Queries - Model tables around query patterns, not normalized forms
- **S**ingle Partition Reads - Optimize for single-partition queries, avoid multi-partition scans
- **A**nti-Entropy Repair - Run regular repairs within gc_grace_seconds window
- **N**o Joins - Denormalize data, duplicate as needed for query patterns
- **D**istributed by Design - Embrace distributed architecture, avoid single points of failure
- **R**eplication Factor ≥3 - Always use RF=3+ in production for fault tolerance
- **A**ppropriate Compaction - Choose compaction strategy based on workload (UCS for 5.x)

**Additional Principles:**

- **Partition Size Limits** - Keep partitions under 100MB, ideally under 10MB
- **Immutable by Default** - Prefer inserts over updates, use TTLs for expiration
- **Monitoring Required** - Track metrics, compaction, repairs, and query performance
- **Security Hardened** - Enable authentication, authorization, and encryption (TLS)

---

## 2. Architecture Overview

### A. Cassandra 4.x/5.x Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Cassandra Cluster                        │
│                                                              │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐             │
│  │  Node 1  │◄──►│  Node 2  │◄──►│  Node 3  │             │
│  │          │    │          │    │          │             │
│  │ Memtable │    │ Memtable │    │ Memtable │             │
│  │ CommitLog│    │ CommitLog│    │ CommitLog│             │
│  │ SSTables │    │ SSTables │    │ SSTables │             │
│  └──────────┘    └──────────┘    └──────────┘             │
│       │                │                │                   │
│       └────────────────┴────────────────┘                   │
│              Gossip Protocol (Ring)                         │
└─────────────────────────────────────────────────────────────┘
```

**Key Components:**

- **Memtable**: In-memory structure for writes (trie-based in 5.x)
- **CommitLog**: Append-only log for durability
- **SSTables**: Sorted String Tables on disk (trie-indexed in 5.x)
- **Compaction**: Merges SSTables, removes tombstones
- **Gossip**: Peer-to-peer discovery and state propagation

### B. Hardware Recommendations (2026)

```yaml
# Minimum Production Node Specifications
CPU: 8+ cores (16+ recommended for high throughput)
RAM: 32GB minimum (64GB+ for large heaps with G1GC)
Disk: SSD required (NVMe preferred)
  - Data: 1-2 TB per node maximum (for optimal compaction)
  - Separate commitlog disk for write-heavy workloads
Network: 10 Gbps minimum for production clusters
```

**Capacity Planning:**

- Run at 50-80% disk capacity (not full capacity)
- 1-2 TB per node maximum for best performance
- Minimum 3 nodes per datacenter (RF=3)
- Number of nodes should be multiple of replication factor

---

## 3. Data Modeling (MANDATORY)

### A. Query-Driven Design

**CRITICAL: Design tables for your query patterns, not normalized forms.**

```cql
-- ❌ WRONG: Normalized relational thinking
CREATE TABLE users (
    user_id UUID PRIMARY KEY,
    username TEXT,
    email TEXT
);

CREATE TABLE posts (
    post_id UUID PRIMARY KEY,
    user_id UUID,  -- Requires application-level join
    title TEXT,
    content TEXT,
    created_at TIMESTAMP
);

-- Query requires 2 reads (inefficient)
-- 1. SELECT * FROM posts WHERE user_id = ?  -- ❌ No partition key!
-- 2. SELECT * FROM users WHERE user_id = ?

-- ✅ CORRECT: Denormalized for query pattern
CREATE TABLE posts_by_user (
    user_id UUID,           -- Partition key
    created_at TIMESTAMP,   -- Clustering key (for ordering)
    post_id UUID,           -- Clustering key (for uniqueness)
    username TEXT,          -- Denormalized user data
    email TEXT,
    title TEXT,
    content TEXT,
    PRIMARY KEY ((user_id), created_at, post_id)
) WITH CLUSTERING ORDER BY (created_at DESC);

-- Single partition read (efficient)
SELECT * FROM posts_by_user
WHERE user_id = 123e4567-e89b-12d3-a456-426614174000
LIMIT 10;
```

### B. Partition Key Design

**Goals:**

1. Distribute data evenly across nodes
2. Minimize partitions read per query
3. Keep partition size bounded

```cql
-- ❌ ANTI-PATTERN: Unbounded partition
CREATE TABLE sensor_data (
    sensor_id TEXT,      -- Partition key
    timestamp TIMESTAMP, -- Clustering key
    value DOUBLE,
    PRIMARY KEY (sensor_id, timestamp)
);
-- Problem: Partition grows indefinitely as time passes!

-- ✅ CORRECT: Time-bucketed partition
CREATE TABLE sensor_data_by_day (
    sensor_id TEXT,
    day DATE,            -- Bucket by day
    timestamp TIMESTAMP,
    value DOUBLE,
    PRIMARY KEY ((sensor_id, day), timestamp)
) WITH CLUSTERING ORDER BY (timestamp DESC);
-- Bounded: Each partition contains only one day of data
-- TTL can be used to expire old buckets

-- Query with time bucket
SELECT * FROM sensor_data_by_day
WHERE sensor_id = 'sensor_123'
  AND day = '2026-02-06'
  AND timestamp >= '2026-02-06 10:00:00'
  AND timestamp < '2026-02-06 11:00:00';
```

### C. Composite Partition Keys

```cql
-- Use composite partition key for better distribution
CREATE TABLE user_events (
    user_id UUID,
    event_type TEXT,    -- Adds to partition key for distribution
    event_time TIMESTAMP,
    event_id TIMEUUID,
    details TEXT,
    PRIMARY KEY ((user_id, event_type), event_time, event_id)
) WITH CLUSTERING ORDER BY (event_time DESC);

-- Query by user and event type
SELECT * FROM user_events
WHERE user_id = 123e4567-e89b-12d3-a456-426614174000
  AND event_type = 'login'
LIMIT 50;
```

### D. Partition Size Guidelines

**MANDATORY Limits:**

- **Max partition size**: 100 MB (hard limit)
- **Ideal partition size**: < 10 MB
- **Max rows per partition**: < 100,000

```cql
-- Check partition sizes
SELECT * FROM system.size_estimates
WHERE keyspace_name = 'my_keyspace'
  AND table_name = 'my_table';

-- Monitor with nodetool
-- nodetool tablestats keyspace_name.table_name
```

### E. Clustering Key Design

```cql
-- Clustering keys provide ordering within partition
CREATE TABLE posts_by_user (
    user_id UUID,
    created_at TIMESTAMP,
    post_id UUID,
    title TEXT,
    content TEXT,
    likes INT,
    PRIMARY KEY ((user_id), created_at, post_id)
) WITH CLUSTERING ORDER BY (created_at DESC, post_id ASC);

-- Range queries on clustering columns
SELECT * FROM posts_by_user
WHERE user_id = 123e4567-e89b-12d3-a456-426614174000
  AND created_at >= '2026-01-01'
  AND created_at < '2026-02-01';

-- Slice queries within partition
SELECT * FROM posts_by_user
WHERE user_id = 123e4567-e89b-12d3-a456-426614174000
LIMIT 20;  -- Latest 20 posts (DESC order)
```

### F. Data Types

```cql
-- Modern Cassandra data types
CREATE TABLE example_types (
    -- UUIDs
    id UUID PRIMARY KEY,
    time_id TIMEUUID,  -- Includes timestamp

    -- Numeric
    count BIGINT,
    price DECIMAL,
    rating DOUBLE,

    -- Text
    username TEXT,
    email TEXT,

    -- Temporal
    created_at TIMESTAMP,
    birth_date DATE,
    event_time TIME,

    -- Collections (use sparingly, max ~100 items)
    tags SET<TEXT>,
    properties MAP<TEXT, TEXT>,
    history LIST<TEXT>,

    -- JSON (Cassandra 5.x)
    metadata TEXT,  -- Store JSON as TEXT

    -- Frozen types (immutable)
    address FROZEN<user_defined_type>,

    -- Vector (Cassandra 5.x for AI/ML)
    embedding VECTOR<FLOAT, 1536>  -- For vector search
);

-- User-defined types
CREATE TYPE address (
    street TEXT,
    city TEXT,
    state TEXT,
    zip_code TEXT,
    country TEXT
);

CREATE TABLE users (
    user_id UUID PRIMARY KEY,
    home_address FROZEN<address>,
    work_address FROZEN<address>
);
```

---

## 4. Replication Strategies (MANDATORY)

### A. NetworkTopologyStrategy (Production Standard)

**CRITICAL: Always use NetworkTopologyStrategy in production.**

```cql
-- ✅ CORRECT: NetworkTopologyStrategy
CREATE KEYSPACE my_app WITH REPLICATION = {
    'class': 'NetworkTopologyStrategy',
    'dc1': 3,  -- 3 replicas in datacenter 1
    'dc2': 3   -- 3 replicas in datacenter 2
};

-- ❌ WRONG: SimpleStrategy (testing only)
CREATE KEYSPACE test_only WITH REPLICATION = {
    'class': 'SimpleStrategy',
    'replication_factor': 1  -- Never use in production!
};
```

**Best Practices:**

- **RF ≥ 3**: Always use replication factor of 3 or more
- **Multi-DC**: Use NetworkTopologyStrategy even for single DC (easier to expand)
- **Rack Awareness**: Cassandra distributes replicas across racks automatically
- **Alter Carefully**: Changing RF requires running `nodetool repair` afterward

```bash
# After changing replication settings
ALTER KEYSPACE my_app WITH REPLICATION = {
    'class': 'NetworkTopologyStrategy',
    'dc1': 5  # Changed from 3 to 5
};

# MUST run repair after RF change
nodetool repair --full my_app
```

### B. Common Configurations

```cql
-- Single datacenter (3 replicas for fault tolerance)
CREATE KEYSPACE app_single_dc WITH REPLICATION = {
    'class': 'NetworkTopologyStrategy',
    'datacenter1': 3
};

-- Multi-datacenter (active-active)
CREATE KEYSPACE app_multi_dc WITH REPLICATION = {
    'class': 'NetworkTopologyStrategy',
    'us_east': 3,
    'us_west': 3,
    'eu_west': 2
};

-- Analytics datacenter (read-only replicas)
CREATE KEYSPACE app_with_analytics WITH REPLICATION = {
    'class': 'NetworkTopologyStrategy',
    'production_dc': 3,
    'analytics_dc': 1  -- Lower RF for analytics
};
```

---

## 5. Consistency Levels (MANDATORY)

### A. Tunable Consistency

**Formula: R + W > RF = Strong Consistency**

- R = Read consistency level
- W = Write consistency level
- RF = Replication factor

```cql
-- Production recommended: LOCAL_QUORUM
-- Writes
INSERT INTO users (user_id, username, email)
VALUES (uuid(), 'john_doe', 'john@example.com')
USING CONSISTENCY LOCAL_QUORUM;

-- Reads
SELECT * FROM users WHERE user_id = ?
USING CONSISTENCY LOCAL_QUORUM;
```

### B. Consistency Level Reference

| Level | Description | Use Case | Availability | Latency |
|-------|-------------|----------|--------------|---------|
| **LOCAL_QUORUM** | Quorum in local DC | **Production default** | High | Medium |
| **QUORUM** | Quorum across all DCs | Strong consistency needed | Medium | High |
| **LOCAL_ONE** | 1 replica in local DC | Read-heavy, eventual consistency OK | Highest | Lowest |
| **ONE** | 1 replica anywhere | Testing, non-critical data | Highest | Lowest |
| **ALL** | All replicas | **Avoid in production** | Lowest | Highest |
| **ANY** | Hinted handoff OK | **Never use (data loss risk)** | Highest | Lowest |
| **EACH_QUORUM** | Quorum in each DC | Multi-DC strong consistency | Medium | Highest |

### C. Production Patterns

```cql
-- Pattern 1: Balanced (most common)
-- Write: LOCAL_QUORUM, Read: LOCAL_QUORUM
-- Trade-off: Medium latency, strong consistency in local DC

-- Pattern 2: Read-optimized
-- Write: QUORUM, Read: ONE
-- Trade-off: Lower read latency, higher write latency

-- Pattern 3: Write-optimized
-- Write: ONE, Read: QUORUM
-- Trade-off: Lower write latency, higher read latency

-- Pattern 4: Critical data
-- Write: LOCAL_QUORUM, Read: LOCAL_QUORUM
-- With RF=3: Tolerates 1 node failure
```

### D. Application Configuration

```python
# Python driver example
from cassandra.cluster import Cluster
from cassandra import ConsistencyLevel

cluster = Cluster(['10.0.0.1', '10.0.0.2', '10.0.0.3'])
session = cluster.connect('my_keyspace')

# Set default consistency
session.default_consistency_level = ConsistencyLevel.LOCAL_QUORUM

# Per-query consistency
query = "SELECT * FROM users WHERE user_id = ?"
prepared = session.prepare(query)
prepared.consistency_level = ConsistencyLevel.LOCAL_ONE

result = session.execute(prepared, [user_id])
```

---

## 6. Compaction Strategies (MANDATORY)

### A. Unified Compaction Strategy (UCS) - Cassandra 5.x

**CRITICAL: UCS is the recommended strategy for Cassandra 5.x+**

```cql
-- ✅ BEST (Cassandra 5.x): Unified Compaction Strategy
CREATE TABLE users (
    user_id UUID PRIMARY KEY,
    username TEXT,
    email TEXT
) WITH compaction = {
    'class': 'UnifiedCompactionStrategy',
    'scaling_parameters': 'T4',  -- Tuning preset
    'min_sstable_size_in_mb': 100
};

-- UCS Scaling Parameters
-- T2: Lower space overhead, higher read amplification
-- T4: Balanced (default)
-- L4: Lower read amplification, higher space overhead
-- N: STCS-like behavior (write-optimized)
```

**UCS Benefits (Cassandra 5.x):**

- Runtime tunable (no full compaction needed)
- Adapts to workload changes
- Better than STCS/LCS for most workloads
- Density-based grouping (vs size-based or level-based)

### B. Legacy Strategies (Cassandra 4.x)

```cql
-- Time-series data with TTL
CREATE TABLE sensor_readings (
    sensor_id TEXT,
    day DATE,
    timestamp TIMESTAMP,
    value DOUBLE,
    PRIMARY KEY ((sensor_id, day), timestamp)
) WITH compaction = {
    'class': 'TimeWindowCompactionStrategy',
    'compaction_window_size': 1,
    'compaction_window_unit': 'DAYS'
}
AND default_time_to_live = 2592000;  -- 30 days

-- Read-heavy workload (Cassandra 4.x)
CREATE TABLE products (
    product_id UUID PRIMARY KEY,
    name TEXT,
    description TEXT,
    price DECIMAL
) WITH compaction = {
    'class': 'LeveledCompactionStrategy',
    'sstable_size_in_mb': 160
};

-- Write-heavy workload (Cassandra 4.x)
CREATE TABLE logs (
    log_id TIMEUUID PRIMARY KEY,
    level TEXT,
    message TEXT,
    created_at TIMESTAMP
) WITH compaction = {
    'class': 'SizeTieredCompactionStrategy',
    'min_threshold': 4,
    'max_threshold': 32
};
```

### C. Compaction Strategy Selection (2026)

| Strategy | Use Case | Cassandra Version | Recommendation |
|----------|----------|-------------------|----------------|
| **UCS** | All workloads | 5.x+ | **Default choice for 5.x** |
| **TWCS** | Time-series with TTL | 4.x, 5.x | Best for expiring data |
| **LCS** | Read-heavy | 4.x | Migrate to UCS in 5.x |
| **STCS** | Write-heavy | 4.x | Migrate to UCS in 5.x |

### D. Monitoring Compaction

```bash
# Check compaction status
nodetool compactionstats

# Check table compaction settings
nodetool describecluster

# Monitor pending compactions
nodetool tpstats | grep CompactionExecutor
```

---

## 7. Indexing (MANDATORY)

### A. Storage-Attached Indexing (SAI) - Cassandra 5.x

**CRITICAL: SAI is the modern indexing solution in Cassandra 5.x**

```cql
-- ✅ BEST (Cassandra 5.x): Storage-Attached Index (SAI)
CREATE TABLE products (
    product_id UUID PRIMARY KEY,
    name TEXT,
    category TEXT,
    price DECIMAL,
    description TEXT,
    tags SET<TEXT>,
    created_at TIMESTAMP
);

-- SAI on single column
CREATE INDEX ON products (category) USING 'sai';

-- SAI on text column (case-insensitive)
CREATE INDEX ON products (name)
USING 'sai'
WITH OPTIONS = {'case_sensitive': 'false'};

-- SAI on collection
CREATE INDEX ON products (tags) USING 'sai';

-- Query with SAI
SELECT * FROM products WHERE category = 'Electronics';
SELECT * FROM products WHERE name = 'laptop';  -- Case insensitive
SELECT * FROM products WHERE tags CONTAINS 'featured';

-- Multiple SAI indexes (AND queries)
SELECT * FROM products
WHERE category = 'Electronics'
  AND price > 100.00
  AND price < 1000.00;
```

**SAI Benefits:**

- Multiple indexes on one table
- Efficient filtering without materialized views
- Supports text, numeric, collections
- Lower disk usage than legacy secondary indexes
- Native database operation (low latency)

### B. Legacy Secondary Indexes (Cassandra 4.x)

```cql
-- ❌ Avoid in Cassandra 5.x (use SAI instead)
CREATE INDEX users_email_idx ON users (email);

-- Limited to single-column lookups
SELECT * FROM users WHERE email = 'user@example.com';
```

**Secondary Index Limitations (4.x):**

- Local to each node (queries all nodes)
- Poor performance on high-cardinality columns
- Not recommended for low-cardinality columns
- No support for range queries

### C. Materialized Views

```cql
-- Materialized views (denormalized tables)
CREATE MATERIALIZED VIEW users_by_email AS
SELECT user_id, username, email, created_at
FROM users
WHERE email IS NOT NULL AND user_id IS NOT NULL
PRIMARY KEY (email, user_id);

-- Query by email (efficient)
SELECT * FROM users_by_email WHERE email = 'user@example.com';
```

**Materialized View Considerations:**

- Automatic maintenance (write overhead)
- Eventually consistent
- Cannot update view directly
- Use SAI if possible (less overhead)

### D. Vector Search (Cassandra 5.x)

```cql
-- Vector embeddings for AI/ML
CREATE TABLE documents (
    doc_id UUID PRIMARY KEY,
    content TEXT,
    embedding VECTOR<FLOAT, 1536>  -- OpenAI embedding size
);

-- Create SAI vector index
CREATE CUSTOM INDEX ON documents (embedding)
USING 'StorageAttachedIndex';

-- Vector similarity search (ANN)
SELECT * FROM documents
ORDER BY embedding ANN OF [0.1, 0.2, ..., 0.9]
LIMIT 10;
```

---

## 8. Lightweight Transactions (LWT)

### A. When to Use LWT

**Use ONLY when you need linearizable consistency (compare-and-set operations).**

```cql
-- ✅ CORRECT: Account registration (prevent duplicates)
INSERT INTO users (user_id, username, email)
VALUES (uuid(), 'john_doe', 'john@example.com')
IF NOT EXISTS;

-- ✅ CORRECT: Optimistic locking
UPDATE inventory
SET quantity = quantity - 1
WHERE product_id = 123
IF quantity > 0;

-- ✅ CORRECT: Compare-and-swap
UPDATE session_tokens
SET token = 'new_token_value'
WHERE user_id = 123
IF token = 'old_token_value';
```

### B. LWT Performance Cost

**WARNING: LWTs are 3-4x slower than normal writes**

- Uses Paxos consensus (multiple round trips)
- Higher latency and resource usage
- Monitor contention metrics

```python
# Python example with conditional insert
from cassandra.cluster import Cluster

cluster = Cluster(['10.0.0.1'])
session = cluster.connect('my_keyspace')

# LWT with IF NOT EXISTS
query = """
INSERT INTO users (user_id, username, email)
VALUES (?, ?, ?)
IF NOT EXISTS
"""
prepared = session.prepare(query)
result = session.execute(prepared, [user_id, username, email])

# Check if applied
if result.one().applied:
    print("User created successfully")
else:
    print("Username already exists")
```

### C. Batching LWTs

```cql
-- Single-partition batch (atomic)
BEGIN BATCH
    INSERT INTO users (user_id, username)
    VALUES (uuid(), 'john') IF NOT EXISTS;

    INSERT INTO user_emails (email, user_id)
    VALUES ('john@example.com', uuid()) IF NOT EXISTS;
APPLY BATCH;

-- ❌ WRONG: Multi-partition LWT batch (not supported efficiently)
-- All statements must target the same partition key
```

### D. Reading LWT Data

```cql
-- Use SERIAL consistency for reads
SELECT * FROM users
WHERE user_id = 123
USING CONSISTENCY SERIAL;  -- Or LOCAL_SERIAL
```

### E. Paxos V2 (Cassandra 4.1+)

**Cassandra 4.1+ uses Paxos V2 with significant improvements:**

- Better performance
- Reduced WAN traffic in multi-DC
- Requires periodic repair: `nodetool repair --paxos`

---

## 9. Batching

### A. Logged Batches (Atomic)

```cql
-- ✅ CORRECT: Single partition batch (efficient)
BEGIN BATCH
    INSERT INTO users (user_id, username, email)
    VALUES (123, 'john', 'john@example.com');

    INSERT INTO user_activity (user_id, activity_type, timestamp)
    VALUES (123, 'registered', toTimestamp(now()));
APPLY BATCH;

-- ❌ WRONG: Multi-partition batch (use UNLOGGED)
-- Logged batches across partitions are expensive
```

**Logged Batch Rules:**

- Atomic across multiple tables
- Use for maintaining consistency
- Avoid large batches (>100 statements)
- Single partition preferred

### B. Unlogged Batches

```cql
-- Multiple partitions, no atomicity needed
BEGIN UNLOGGED BATCH
    INSERT INTO events (event_id, type, data)
    VALUES (uuid(), 'click', 'button1');

    INSERT INTO events (event_id, type, data)
    VALUES (uuid(), 'view', 'page1');
APPLY BATCH;
```

**Unlogged Batch Rules:**

- Not atomic (individual statements may fail)
- Lower overhead than logged batches
- Use for bulk loading
- Still sends to coordinator as single request

### C. Anti-Patterns

```cql
-- ❌ ANTI-PATTERN: Batching for performance
-- Don't batch unrelated inserts thinking it's faster
BEGIN BATCH
    INSERT INTO table1 (...) VALUES (...);
    INSERT INTO table2 (...) VALUES (...);
    INSERT INTO table3 (...) VALUES (...);
APPLY BATCH;
-- This is SLOWER than individual async inserts!

-- ✅ CORRECT: Use async execution
-- session.execute_async() for concurrent inserts
```

---

## 10. Performance Optimization

### A. JVM Tuning (Cassandra 4.x/5.x)

```yaml
# jvm.options or jvm11-server.options

# Heap Size (8-16 GB recommended, max 64 GB with G1GC)
-Xms16G
-Xmx16G

# G1 Garbage Collector (default in modern Cassandra)
-XX:+UseG1GC
-XX:MaxGCPauseMillis=200
-XX:G1HeapRegionSize=16m
-XX:MaxTenuringThreshold=1

# GC Logging (essential for production)
-Xlog:gc*:file=/var/log/cassandra/gc.log:time,uptime:filecount=10,filesize=10m

# JDK 17 support (Cassandra 5.x)
# Use latest JDK 17 for best performance

# Heap dump on OOM
-XX:+HeapDumpOnOutOfMemoryError
-XX:HeapDumpPath=/var/log/cassandra/heap_dumps
```

**Heap Sizing Guidelines:**

- **8-16 GB**: Standard for most workloads
- **32-64 GB**: Large workloads with G1GC
- **Min = Max**: Prevent resize pauses

### B. Cassandra Configuration (cassandra.yaml)

```yaml
# Read/Write Performance
concurrent_reads: 32
concurrent_writes: 32
concurrent_counter_writes: 32

# Memtable settings (adjust based on heap)
memtable_heap_space_in_mb: 2048
memtable_offheap_space_in_mb: 2048

# Commit log
commitlog_sync: periodic
commitlog_sync_period_in_ms: 10000
commitlog_segment_size_in_mb: 32

# Compaction
compaction_throughput_mb_per_sec: 64  # 0 for unlimited

# Cache sizes (percent of heap)
key_cache_size_in_mb: 0  # Disabled by default (rarely needed)
row_cache_size_in_mb: 0  # Use OS page cache instead

# Timeouts (milliseconds)
read_request_timeout_in_ms: 5000
write_request_timeout_in_ms: 2000
range_request_timeout_in_ms: 10000

# Tombstones (anti-entropy)
gc_grace_seconds: 864000  # 10 days (default)
tombstone_warn_threshold: 1000
tombstone_failure_threshold: 100000
```

### C. Query Optimization

```cql
-- ✅ CORRECT: Single partition query
SELECT * FROM posts_by_user
WHERE user_id = 123
LIMIT 10;

-- ❌ WRONG: Full table scan
SELECT * FROM posts_by_user LIMIT 100 ALLOW FILTERING;

-- ❌ WRONG: Multi-partition query without partition key
SELECT * FROM posts_by_user WHERE created_at > '2026-01-01';

-- ✅ CORRECT: Use partition key with clustering filter
SELECT * FROM posts_by_user
WHERE user_id = 123
  AND created_at > '2026-01-01';
```

### D. Tracing Queries

```cql
-- Enable tracing for slow query analysis
TRACING ON;

SELECT * FROM users WHERE user_id = 123;

-- View trace
SHOW SESSION TRACING;

TRACING OFF;
```

---

## 11. Security (MANDATORY)

### A. Authentication & Authorization

```yaml
# cassandra.yaml
authenticator: PasswordAuthenticator  # Default: AllowAllAuthenticator
authorizer: CassandraAuthorizer        # Default: AllowAllAuthorizer

# Enable client encryption
client_encryption_options:
  enabled: true
  optional: false
  keystore: /path/to/keystore.jks
  keystore_password: changeit
  require_client_auth: true
  truststore: /path/to/truststore.jks
  truststore_password: changeit
  protocol: TLS
  algorithm: SunX509
  store_type: JKS
  cipher_suites: [TLS_RSA_WITH_AES_256_CBC_SHA]

# Enable inter-node encryption
server_encryption_options:
  internode_encryption: all  # all, dc, rack, none
  keystore: /path/to/keystore.jks
  keystore_password: changeit
  truststore: /path/to/truststore.jks
  truststore_password: changeit
  protocol: TLS
  algorithm: SunX509
  store_type: JKS
  require_client_auth: true
```

### B. User Management

```cql
-- Create admin user
CREATE ROLE admin WITH PASSWORD = 'secure_password_here'
AND SUPERUSER = true
AND LOGIN = true;

-- Create application user with limited permissions
CREATE ROLE app_user WITH PASSWORD = 'app_password'
AND LOGIN = true;

-- Grant permissions
GRANT SELECT ON KEYSPACE my_app TO app_user;
GRANT MODIFY ON TABLE my_app.users TO app_user;

-- Create read-only user
CREATE ROLE readonly_user WITH PASSWORD = 'readonly_pass'
AND LOGIN = true;

GRANT SELECT ON ALL KEYSPACES TO readonly_user;

-- Revoke default superuser (cassandra)
ALTER ROLE cassandra WITH PASSWORD = 'new_strong_password';
-- Or disable after creating admin:
-- ALTER ROLE cassandra WITH LOGIN = false;
```

### C. Network Security

```yaml
# cassandra.yaml - Bind to specific IPs
listen_address: 10.0.1.10
rpc_address: 10.0.1.10
broadcast_address: 10.0.1.10

# Enable native protocol authentication
native_transport_port: 9042
native_transport_port_ssl: 9142
```

### D. Audit Logging (DataStax Enterprise feature)

```yaml
# For Apache Cassandra, use external tools:
# - Application-level audit logging
# - Database activity monitoring (DAM) tools
# - Log aggregation (Elasticsearch, Splunk)
```

---

## 12. Monitoring & Operations

### A. Nodetool Commands

```bash
# Cluster status
nodetool status
nodetool info
nodetool describecluster

# Table statistics
nodetool tablestats keyspace_name
nodetool tablestats keyspace_name.table_name
nodetool tablehistograms keyspace_name table_name

# Performance monitoring
nodetool tpstats         # Thread pool stats
nodetool compactionstats # Compaction progress
nodetool proxyhistograms # Coordinator latency
nodetool netstats        # Network operations

# Operations
nodetool repair          # Anti-entropy repair
nodetool cleanup         # Remove unnecessary data after topology change
nodetool flush           # Flush memtables to disk
nodetool drain           # Prepare for shutdown
nodetool snapshot        # Create snapshot

# Cache
nodetool invalidatekeycache
nodetool invalidaterowcache

# Troubleshooting
nodetool gossipinfo
nodetool ring
nodetool getendpoints keyspace_name table_name partition_key
```

### B. JMX Metrics

```bash
# Connect to JMX (default port 7199)
jconsole localhost:7199

# Key MBeans to monitor:
# - org.apache.cassandra.metrics:type=ClientRequest
# - org.apache.cassandra.metrics:type=Storage
# - org.apache.cassandra.metrics:type=Compaction
# - org.apache.cassandra.metrics:type=Cache
# - org.apache.cassandra.metrics:type=ThreadPools
```

### C. Prometheus + Grafana

```yaml
# Use JMX exporter for Prometheus
# Download: https://github.com/prometheus/jmx_exporter

# Run with Cassandra
JVM_OPTS="$JVM_OPTS -javaagent:/path/to/jmx_prometheus_javaagent.jar=7070:/path/to/cassandra.yml"

# Prometheus scrape config
scrape_configs:
  - job_name: 'cassandra'
    static_configs:
      - targets: ['node1:7070', 'node2:7070', 'node3:7070']
```

### D. Key Metrics to Monitor

| Metric | Description | Alert Threshold |
|--------|-------------|-----------------|
| **Read Latency** | p99 read latency | > 100ms |
| **Write Latency** | p99 write latency | > 50ms |
| **Pending Compactions** | Compaction backlog | > 20 |
| **Pending Repairs** | Repair tasks pending | > 0 |
| **Disk Usage** | Used disk space | > 70% |
| **Heap Usage** | JVM heap usage | > 75% |
| **GC Pause Time** | Garbage collection pause | > 1 second |
| **Dropped Messages** | Dropped write/read messages | > 0 |
| **Hinted Handoffs** | Undelivered writes | > 1000 |
| **Tombstone Scans** | Tombstones scanned per query | > 1000 |

---

## 13. Repair Strategies

### A. Incremental Repair (Cassandra 4.x+)

```bash
# Incremental repair (default, recommended)
nodetool repair --full

# Incremental repair per datacenter
nodetool repair -dc dc1

# Incremental repair single keyspace
nodetool repair my_keyspace

# Full repair (occasional, not daily)
nodetool repair --full my_keyspace

# Partition range repair (for large datasets)
nodetool repair -pr  # Primary range only
```

**Repair Frequency:**

- **Incremental**: Every 1-3 days
- **Full**: Every 1-3 weeks
- **Required**: Within gc_grace_seconds (default 10 days)

### B. Cassandra Reaper (Recommended)

```yaml
# Use Reaper for automated repair orchestration
# https://cassandra-reaper.io/

# Docker Compose example
version: '3.7'
services:
  reaper:
    image: thelastpickle/cassandra-reaper:latest
    environment:
      REAPER_STORAGE_TYPE: cassandra
      REAPER_CASS_CONTACT_POINTS: ["cassandra-node1"]
      REAPER_CASS_KEYSPACE: reaper_db
    ports:
      - "8080:8080"
      - "8081:8081"
```

### C. Paxos Repair (Cassandra 4.1+ with LWT)

```bash
# Repair Paxos state (if using lightweight transactions)
nodetool repair --paxos-only my_keyspace
```

---

## 14. Backup & Restore

### A. Snapshots

```bash
# Create snapshot
nodetool snapshot -t my_backup_20260206 my_keyspace

# List snapshots
nodetool listsnapshots

# Snapshot location
# /var/lib/cassandra/data/keyspace_name/table_name/snapshots/snapshot_name/

# Clear old snapshots
nodetool clearsnapshot -t my_backup_20260206
nodetool clearsnapshot --all
```

### B. Incremental Backups

```yaml
# cassandra.yaml
incremental_backups: true

# Backup location
# /var/lib/cassandra/data/keyspace_name/table_name/backups/
```

### C. Restore Procedure

```bash
# 1. Stop Cassandra
sudo systemctl stop cassandra

# 2. Clear existing data (if full restore)
rm -rf /var/lib/cassandra/data/keyspace_name/table_name/*

# 3. Copy snapshot files
cp -r /path/to/snapshot/files/* /var/lib/cassandra/data/keyspace_name/table_name/

# 4. Change ownership
chown -R cassandra:cassandra /var/lib/cassandra/data

# 5. Restart Cassandra
sudo systemctl start cassandra

# 6. Refresh (if files copied manually)
nodetool refresh keyspace_name table_name
```

### D. sstableloader (Cross-cluster restore)

```bash
# Use sstableloader for restoring to different cluster
sstableloader -d target_cluster_ip \
  /path/to/snapshot/keyspace_name/table_name/
```

### E. Backup to Cloud Storage

```bash
# Example: Backup to AWS S3
#!/bin/bash
SNAPSHOT_NAME="backup_$(date +%Y%m%d_%H%M%S)"
KEYSPACE="my_keyspace"

# Create snapshot
nodetool snapshot -t $SNAPSHOT_NAME $KEYSPACE

# Upload to S3
aws s3 sync /var/lib/cassandra/data/$KEYSPACE/ \
  s3://my-cassandra-backups/$(hostname)/$SNAPSHOT_NAME/ \
  --include "*/snapshots/$SNAPSHOT_NAME/*"

# Clean local snapshot
nodetool clearsnapshot -t $SNAPSHOT_NAME
```

---

## 15. Upgrade Procedures

### A. Cassandra 4.x to 5.x Upgrade (3-Step Process)

**CRITICAL: Cassandra 5.x requires a phased upgrade approach**

```bash
# Step 1: Upgrade to 5.x with backward compatibility
# On each node (rolling upgrade):

# 1.1 Drain node
nodetool drain

# 1.2 Stop Cassandra
sudo systemctl stop cassandra

# 1.3 Backup cassandra.yaml
cp /etc/cassandra/cassandra.yaml /etc/cassandra/cassandra.yaml.backup

# 1.4 Install Cassandra 5.x
# (package manager or tarball)

# 1.5 Restore cassandra.yaml (merge new settings)
# Review and update configuration

# 1.6 Start Cassandra
sudo systemctl start cassandra

# 1.7 Verify node is UP
nodetool status

# 1.8 Wait for node to fully join
# Monitor logs: tail -f /var/log/cassandra/system.log

# Repeat for all nodes (one at a time)

# Step 2: Enable format migration (after all nodes on 5.x)
# This allows SSTables to be migrated to new format

# On each node:
nodetool upgradesstables

# Step 3: Full optimization (final step)
# After all nodes upgraded and SSTables migrated
# New writes will use BTI format (trie-indexed SSTables)
```

### B. Version Compatibility

**Upgrade Paths:**

- 3.x → 4.x → 5.x (cannot skip major versions)
- 4.0 → 4.1 → 5.0 (recommended path)
- 4.1 → 5.0 (direct upgrade supported)

### C. Pre-Upgrade Checklist

```bash
# Verify cluster health
nodetool status
nodetool describecluster

# Run repairs
nodetool repair --full

# Check disk space (50%+ free needed for upgrade)
df -h

# Backup schema
cqlsh -e "DESCRIBE SCHEMA" > schema_backup.cql

# Create snapshots
nodetool snapshot

# Review release notes for breaking changes
# https://cassandra.apache.org/doc/latest/cassandra/new/index.html
```

### D. Post-Upgrade Verification

```bash
# Check cluster status
nodetool status

# Verify schema agreement
nodetool describecluster

# Check for errors
tail -f /var/log/cassandra/system.log

# Run repairs
nodetool repair --full

# Monitor metrics
nodetool tablestats
nodetool tpstats
```

---

## 16. Kubernetes Deployment

### A. K8ssandra Operator (Recommended)

```yaml
# Install K8ssandra Operator
helm repo add k8ssandra https://helm.k8ssandra.io/stable
helm repo update

helm install k8ssandra-operator k8ssandra/k8ssandra-operator \
  -n k8ssandra-operator \
  --create-namespace

# Deploy Cassandra cluster
apiVersion: k8ssandra.io/v1alpha1
kind: K8ssandraCluster
metadata:
  name: demo-cluster
spec:
  cassandra:
    serverVersion: "5.0.0"
    storageConfig:
      cassandraDataVolumeClaimSpec:
        accessModes:
          - ReadWriteOnce
        resources:
          requests:
            storage: 100Gi
        storageClassName: fast-ssd
    config:
      jvmOptions:
        heapSize: 16Gi
    datacenters:
      - metadata:
          name: dc1
        size: 3
        resources:
          requests:
            cpu: 4
            memory: 32Gi
          limits:
            cpu: 4
            memory: 32Gi
```

### B. StatefulSet Example (Manual)

```yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: cassandra
  namespace: cassandra
spec:
  serviceName: cassandra
  replicas: 3
  selector:
    matchLabels:
      app: cassandra
  template:
    metadata:
      labels:
        app: cassandra
    spec:
      containers:
      - name: cassandra
        image: cassandra:5.0
        ports:
        - containerPort: 7000
          name: intra-node
        - containerPort: 7001
          name: tls-intra-node
        - containerPort: 7199
          name: jmx
        - containerPort: 9042
          name: cql
        env:
        - name: CASSANDRA_SEEDS
          value: "cassandra-0.cassandra.cassandra.svc.cluster.local"
        - name: MAX_HEAP_SIZE
          value: "16G"
        - name: HEAP_NEWSIZE
          value: "4G"
        - name: CASSANDRA_CLUSTER_NAME
          value: "K8s-Cluster"
        - name: CASSANDRA_DC
          value: "DC1"
        - name: CASSANDRA_RACK
          value: "Rack1"
        volumeMounts:
        - name: cassandra-data
          mountPath: /var/lib/cassandra
        resources:
          requests:
            cpu: 4
            memory: 32Gi
          limits:
            cpu: 4
            memory: 32Gi
        livenessProbe:
          exec:
            command:
            - /bin/bash
            - -c
            - nodetool status | grep UN
          initialDelaySeconds: 90
          periodSeconds: 30
        readinessProbe:
          exec:
            command:
            - /bin/bash
            - -c
            - nodetool status | grep UN
          initialDelaySeconds: 60
          periodSeconds: 10
  volumeClaimTemplates:
  - metadata:
      name: cassandra-data
    spec:
      accessModes: [ "ReadWriteOnce" ]
      storageClassName: fast-ssd
      resources:
        requests:
          storage: 100Gi
```

### C. Service Configuration

```yaml
apiVersion: v1
kind: Service
metadata:
  name: cassandra
  namespace: cassandra
  labels:
    app: cassandra
spec:
  clusterIP: None
  selector:
    app: cassandra
  ports:
  - port: 9042
    name: cql
  - port: 7000
    name: intra-node
  - port: 7001
    name: tls-intra-node
  - port: 7199
    name: jmx
```

### D. Cass-Operator (Alternative)

```bash
# Install cass-operator
helm repo add k8ssandra https://helm.k8ssandra.io/stable
helm install cass-operator k8ssandra/cass-operator \
  -n cass-operator \
  --create-namespace

# Deploy Cassandra datacenter
apiVersion: cassandra.datastax.com/v1beta1
kind: CassandraDatacenter
metadata:
  name: dc1
spec:
  clusterName: cluster1
  serverType: cassandra
  serverVersion: "5.0.0"
  managementApiAuth:
    insecure: {}
  size: 3
  storageConfig:
    cassandraDataVolumeClaimSpec:
      storageClassName: fast-ssd
      accessModes:
        - ReadWriteOnce
      resources:
        requests:
          storage: 100Gi
  config:
    cassandra-yaml:
      authenticator: PasswordAuthenticator
      authorizer: CassandraAuthorizer
    jvm-server-options:
      initial_heap_size: "16G"
      max_heap_size: "16G"
  resources:
    requests:
      cpu: 4
      memory: 32Gi
    limits:
      cpu: 4
      memory: 32Gi
```

---

## 17. Docker Deployment

### A. Docker Compose (Development)

```yaml
version: '3.8'

services:
  cassandra-1:
    image: cassandra:5.0
    container_name: cassandra-1
    hostname: cassandra-1
    environment:
      - CASSANDRA_CLUSTER_NAME=dev-cluster
      - CASSANDRA_DC=dc1
      - CASSANDRA_RACK=rack1
      - CASSANDRA_SEEDS=cassandra-1
      - CASSANDRA_ENDPOINT_SNITCH=GossipingPropertyFileSnitch
      - MAX_HEAP_SIZE=2G
      - HEAP_NEWSIZE=512M
    ports:
      - "9042:9042"
      - "7199:7199"
    volumes:
      - cassandra-1-data:/var/lib/cassandra
    networks:
      - cassandra-net
    healthcheck:
      test: ["CMD", "cqlsh", "-e", "describe cluster"]
      interval: 30s
      timeout: 10s
      retries: 5

  cassandra-2:
    image: cassandra:5.0
    container_name: cassandra-2
    hostname: cassandra-2
    environment:
      - CASSANDRA_CLUSTER_NAME=dev-cluster
      - CASSANDRA_DC=dc1
      - CASSANDRA_RACK=rack1
      - CASSANDRA_SEEDS=cassandra-1
      - CASSANDRA_ENDPOINT_SNITCH=GossipingPropertyFileSnitch
      - MAX_HEAP_SIZE=2G
      - HEAP_NEWSIZE=512M
    volumes:
      - cassandra-2-data:/var/lib/cassandra
    networks:
      - cassandra-net
    depends_on:
      cassandra-1:
        condition: service_healthy

  cassandra-3:
    image: cassandra:5.0
    container_name: cassandra-3
    hostname: cassandra-3
    environment:
      - CASSANDRA_CLUSTER_NAME=dev-cluster
      - CASSANDRA_DC=dc1
      - CASSANDRA_RACK=rack1
      - CASSANDRA_SEEDS=cassandra-1
      - CASSANDRA_ENDPOINT_SNITCH=GossipingPropertyFileSnitch
      - MAX_HEAP_SIZE=2G
      - HEAP_NEWSIZE=512M
    volumes:
      - cassandra-3-data:/var/lib/cassandra
    networks:
      - cassandra-net
    depends_on:
      cassandra-1:
        condition: service_healthy

volumes:
  cassandra-1-data:
  cassandra-2-data:
  cassandra-3-data:

networks:
  cassandra-net:
    driver: bridge
```

### B. Production Docker Deployment

```bash
# Use official Cassandra image with custom configuration
docker run -d \
  --name cassandra-node1 \
  --network cassandra-net \
  -p 9042:9042 \
  -p 7199:7199 \
  -v /data/cassandra:/var/lib/cassandra \
  -v /etc/cassandra/cassandra.yaml:/etc/cassandra/cassandra.yaml \
  -e CASSANDRA_CLUSTER_NAME=production-cluster \
  -e CASSANDRA_DC=dc1 \
  -e CASSANDRA_RACK=rack1 \
  -e CASSANDRA_SEEDS=cassandra-node1,cassandra-node2 \
  -e MAX_HEAP_SIZE=16G \
  -e HEAP_NEWSIZE=4G \
  cassandra:5.0
```

---

## 18. Migration Strategies

### A. Dual-Write Pattern

```python
# Migrate from existing database to Cassandra
from cassandra.cluster import Cluster
import psycopg2

# Old database connection
pg_conn = psycopg2.connect("dbname=olddb user=postgres")
pg_cursor = pg_conn.cursor()

# Cassandra connection
cluster = Cluster(['10.0.0.1', '10.0.0.2'])
session = cluster.connect('new_keyspace')

# Prepare statements
insert_stmt = session.prepare("""
    INSERT INTO users (user_id, username, email, created_at)
    VALUES (?, ?, ?, ?)
""")

def create_user(username, email):
    """Dual-write: Write to both databases"""

    # Write to PostgreSQL (old)
    pg_cursor.execute(
        "INSERT INTO users (username, email) VALUES (%s, %s) RETURNING id",
        (username, email)
    )
    user_id = pg_cursor.fetchone()[0]
    pg_conn.commit()

    # Write to Cassandra (new)
    from uuid import UUID
    import datetime

    session.execute(insert_stmt, [
        UUID(int=user_id),
        username,
        email,
        datetime.datetime.now()
    ])

    return user_id
```

### B. Bulk Migration

```python
# Bulk data migration script
from cassandra.cluster import Cluster
from cassandra.concurrent import execute_concurrent_with_args
import psycopg2

# Source database
pg_conn = psycopg2.connect("dbname=olddb")
pg_cursor = pg_conn.cursor()

# Cassandra cluster
cluster = Cluster(['10.0.0.1'])
session = cluster.connect('new_keyspace')

# Prepare insert statement
insert_stmt = session.prepare("""
    INSERT INTO users (user_id, username, email, created_at)
    VALUES (?, ?, ?, ?)
""")

# Fetch data from source
pg_cursor.execute("SELECT id, username, email, created_at FROM users")

# Batch insert to Cassandra
batch_size = 1000
batch = []

for row in pg_cursor:
    batch.append((
        UUID(int=row[0]),
        row[1],
        row[2],
        row[3]
    ))

    if len(batch) >= batch_size:
        execute_concurrent_with_args(
            session, insert_stmt, batch,
            concurrency=50
        )
        batch = []
        print(f"Migrated {batch_size} records")

# Insert remaining
if batch:
    execute_concurrent_with_args(
        session, insert_stmt, batch,
        concurrency=50
    )
```

### C. Change Data Capture (CDC)

```yaml
# Enable CDC in cassandra.yaml
cdc_enabled: true
cdc_raw_directory: /var/lib/cassandra/cdc_raw

# Create table with CDC
CREATE TABLE users (
    user_id UUID PRIMARY KEY,
    username TEXT,
    email TEXT
) WITH cdc = true;

# Use CDC for real-time replication to Cassandra
# Tools: Debezium, Maxwell, custom CDC consumer
```

---

## 19. Cassandra 5.x New Features (2024-2026)

### A. Storage-Attached Indexing (SAI)

```cql
-- Modern indexing in Cassandra 5.x
CREATE TABLE products (
    product_id UUID PRIMARY KEY,
    name TEXT,
    category TEXT,
    price DECIMAL,
    tags SET<TEXT>
);

-- Create SAI indexes
CREATE INDEX ON products (category) USING 'sai';
CREATE INDEX ON products (name) USING 'sai'
WITH OPTIONS = {'case_sensitive': 'false'};
CREATE INDEX ON products (tags) USING 'sai';

-- Complex queries with multiple filters
SELECT * FROM products
WHERE category = 'Electronics'
  AND price > 100.00
  AND price < 1000.00;
```

### B. Trie Memtables and SSTables

**Performance improvements in Cassandra 5.x:**

- Trie-based memtables (more efficient memory usage)
- Trie-indexed SSTables (faster reads, smaller indexes)
- Reduced GC pressure
- Better compression

```yaml
# No configuration change needed
# Cassandra 5.x uses tries by default (BTI format)

# Check SSTable format
nodetool tablestats keyspace.table | grep "SSTable"
```

### C. Vector Search

```cql
-- AI/ML embeddings in Cassandra 5.x
CREATE TABLE documents (
    doc_id UUID PRIMARY KEY,
    title TEXT,
    content TEXT,
    embedding VECTOR<FLOAT, 1536>  -- OpenAI ada-002 dimension
);

-- Insert with vector
INSERT INTO documents (doc_id, title, content, embedding)
VALUES (
    uuid(),
    'Cassandra Best Practices',
    'Document content...',
    [0.023, -0.041, ..., 0.019]  -- 1536 dimensions
);

-- Create vector index
CREATE CUSTOM INDEX ON documents (embedding)
USING 'StorageAttachedIndex';

-- Approximate Nearest Neighbor (ANN) search
SELECT doc_id, title, content
FROM documents
ORDER BY embedding ANN OF [0.021, -0.039, ..., 0.018]
LIMIT 10;
```

### D. Unified Compaction Strategy (UCS)

```cql
-- UCS: Adaptive compaction in Cassandra 5.x
CREATE TABLE events (
    event_id UUID PRIMARY KEY,
    event_type TEXT,
    timestamp TIMESTAMP,
    data TEXT
) WITH compaction = {
    'class': 'UnifiedCompactionStrategy',
    'scaling_parameters': 'T4'  -- Balanced preset
};

-- Runtime tuning (no rewrite needed!)
ALTER TABLE events WITH compaction = {
    'class': 'UnifiedCompactionStrategy',
    'scaling_parameters': 'L4'  -- Lower read amplification
};
```

### E. Dynamic Data Masking

```cql
-- Data masking for sensitive fields (Cassandra 5.x)
CREATE TABLE users (
    user_id UUID PRIMARY KEY,
    username TEXT,
    email TEXT MASKED WITH DEFAULT,  -- Masked by default
    ssn TEXT MASKED WITH REPLACE('XXX-XX-', 7, 4)  -- Partial masking
);

-- Unmasked read (requires UNMASK permission)
SELECT * FROM users UNMASK WHERE user_id = ?;
```

### F. JDK 17 Support

**Cassandra 5.x supports JDK 17 (LTS):**

- Better performance than JDK 11
- Modern garbage collectors
- Improved security

```bash
# Use JDK 17 for Cassandra 5.x
java -version
# openjdk version "17.0.9" 2023-10-17 LTS
```

### G. Enhanced Guardrails

```yaml
# cassandra.yaml - Production safety limits
guardrails:
  # Keyspace-level
  keyspaces_warn_threshold: 40
  keyspaces_fail_threshold: 50

  # Table-level
  tables_warn_threshold: 150
  tables_fail_threshold: 200

  # Query-level
  partition_keys_in_select_warn_threshold: 20
  partition_keys_in_select_fail_threshold: 100

  # Data size
  page_size_warn_threshold: 4096
  page_size_fail_threshold: 8192
```

---

## 20. Production Deployment Checklist

### A. Pre-Production Verification

**Before deploying to production:**

- [ ] Cluster topology planned (RF ≥ 3, multi-rack awareness)
- [ ] Hardware meets requirements (SSD, 32GB+ RAM, 8+ cores)
- [ ] NetworkTopologyStrategy configured
- [ ] Authentication enabled (PasswordAuthenticator)
- [ ] Authorization enabled (CassandraAuthorizer)
- [ ] TLS/SSL configured (client-to-node, node-to-node)
- [ ] JVM tuned (heap size, G1GC, GC logging)
- [ ] Compaction strategy selected (UCS for 5.x)
- [ ] Monitoring configured (Prometheus, Grafana, JMX)
- [ ] Backup strategy implemented (snapshots, S3/cloud storage)
- [ ] Repair automation configured (Cassandra Reaper)
- [ ] Schema designed for query patterns (not normalized)
- [ ] Consistency levels chosen (LOCAL_QUORUM recommended)
- [ ] Load testing completed (simulate production traffic)
- [ ] Disaster recovery plan documented
- [ ] Capacity planning completed (disk, memory, nodes)
- [ ] Alerting configured (metrics thresholds)

### B. Data Modeling Review

- [ ] Partition keys distribute data evenly
- [ ] Partition sizes bounded (< 100 MB, ideally < 10 MB)
- [ ] Single-partition queries preferred
- [ ] Denormalization applied for query patterns
- [ ] Time-bucketing used for time-series data
- [ ] Clustering keys provide desired ordering
- [ ] No ALLOW FILTERING in production queries
- [ ] Appropriate data types used (UUIDs, TIMEUUID, etc.)
- [ ] Collections limited in size (< 100 items)
- [ ] TTL configured for expiring data

### C. Operational Readiness

- [ ] Runbooks documented (incident response)
- [ ] Team trained on Cassandra operations
- [ ] nodetool commands tested
- [ ] Upgrade procedure tested in staging
- [ ] Rollback plan prepared
- [ ] On-call rotation established
- [ ] SLAs defined (latency, availability)
- [ ] Compliance requirements met (GDPR, HIPAA, etc.)

---

## 21. Common Anti-Patterns (AVOID)

### A. Schema Design

```cql
-- ❌ ANTI-PATTERN: Queue pattern (delete after read)
CREATE TABLE task_queue (
    queue_id TEXT,
    task_id TIMEUUID,
    status TEXT,
    PRIMARY KEY (queue_id, task_id)
);

-- Problem: Tombstones accumulate, degrading performance
DELETE FROM task_queue WHERE queue_id = 'queue1' AND task_id = ?;

-- ✅ CORRECT: Use dedicated message queue (Kafka, RabbitMQ)

-- ❌ ANTI-PATTERN: Unbounded partition
CREATE TABLE logs (
    application TEXT,
    timestamp TIMESTAMP,
    message TEXT,
    PRIMARY KEY (application, timestamp)
);
-- Problem: Partition grows forever!

-- ✅ CORRECT: Time-bucketed partition
CREATE TABLE logs_by_day (
    application TEXT,
    day DATE,
    timestamp TIMESTAMP,
    message TEXT,
    PRIMARY KEY ((application, day), timestamp)
) WITH default_time_to_live = 604800;  -- 7 days TTL
```

### B. Query Patterns

```cql
-- ❌ ANTI-PATTERN: ALLOW FILTERING in production
SELECT * FROM users WHERE age > 25 ALLOW FILTERING;
-- Problem: Full table scan!

-- ✅ CORRECT: Design table for the query
CREATE TABLE users_by_age (
    age_bucket INT,
    age INT,
    user_id UUID,
    username TEXT,
    PRIMARY KEY ((age_bucket), age, user_id)
);

-- ❌ ANTI-PATTERN: Multi-partition IN query
SELECT * FROM users WHERE user_id IN (?, ?, ?, ...);  -- 1000 IDs
-- Problem: Queries all partitions, high latency

-- ✅ CORRECT: Async individual queries
-- Use driver's async API to fetch concurrently
```

### C. Operations

```cql
-- ❌ ANTI-PATTERN: Using consistency ONE in production
-- Writes: ONE, Reads: ONE
-- Problem: Data loss risk, no fault tolerance

-- ✅ CORRECT: Use LOCAL_QUORUM
-- Writes: LOCAL_QUORUM, Reads: LOCAL_QUORUM
-- Tolerates 1 node failure with RF=3

-- ❌ ANTI-PATTERN: Never running repairs
-- Problem: Data inconsistency, tombstone buildup

-- ✅ CORRECT: Automated repairs with Reaper
-- Schedule repairs every 1-3 days
```

---

## 22. Quick Reference

### A. Common CQL Commands

```cql
-- Schema
DESCRIBE KEYSPACE my_keyspace;
DESCRIBE TABLE my_table;
DESCRIBE INDEX my_index;

-- Data
SELECT * FROM users WHERE user_id = ? LIMIT 10;
INSERT INTO users (user_id, username) VALUES (?, ?);
UPDATE users SET email = ? WHERE user_id = ?;
DELETE FROM users WHERE user_id = ?;

-- Batch
BEGIN BATCH
    INSERT INTO users (...) VALUES (...);
    INSERT INTO user_activity (...) VALUES (...);
APPLY BATCH;

-- Tracing
TRACING ON;
SELECT * FROM users WHERE user_id = ?;
SHOW SESSION TRACING;
```

### B. Nodetool Commands

```bash
# Cluster
nodetool status
nodetool info
nodetool ring
nodetool describecluster

# Operations
nodetool repair
nodetool cleanup
nodetool flush
nodetool drain
nodetool snapshot

# Performance
nodetool tablestats
nodetool tpstats
nodetool compactionstats
nodetool proxyhistograms

# Troubleshooting
nodetool gossipinfo
nodetool getendpoints keyspace table partition_key
nodetool cfstats
```

### C. Client Driver Example (Python)

```python
from cassandra.cluster import Cluster
from cassandra import ConsistencyLevel
from cassandra.query import SimpleStatement

# Connect to cluster
cluster = Cluster(
    contact_points=['10.0.0.1', '10.0.0.2', '10.0.0.3'],
    port=9042,
    protocol_version=5
)
session = cluster.connect('my_keyspace')

# Set default consistency
session.default_consistency_level = ConsistencyLevel.LOCAL_QUORUM

# Prepared statement (best practice)
prepared = session.prepare("""
    INSERT INTO users (user_id, username, email)
    VALUES (?, ?, ?)
""")

session.execute(prepared, [
    uuid.uuid4(),
    'john_doe',
    'john@example.com'
])

# Query with parameters
query = "SELECT * FROM users WHERE user_id = ?"
prepared_query = session.prepare(query)
rows = session.execute(prepared_query, [user_id])

for row in rows:
    print(row.username, row.email)

# Async execution
from cassandra.concurrent import execute_concurrent_with_args

parameters = [
    (uuid.uuid4(), 'user1', 'user1@example.com'),
    (uuid.uuid4(), 'user2', 'user2@example.com'),
]

execute_concurrent_with_args(
    session,
    prepared,
    parameters,
    concurrency=50
)

# Close connection
cluster.shutdown()
```

---

## 23. Why This Configuration Works

**Query-Driven Design**:
- Cassandra excels when tables are designed for specific access patterns, eliminating expensive joins and enabling single-partition reads for optimal performance.

**Tunable Consistency**:
- The ability to choose consistency levels per query allows balancing between latency, availability, and consistency based on business requirements.

**Distributed Architecture**:
- No single point of failure, linear scalability, and multi-datacenter replication provide enterprise-grade availability and disaster recovery.

**Cassandra 5.x Improvements**:
- SAI eliminates the need for multiple materialized views, UCS adapts to changing workloads, trie-based structures improve performance and reduce GC pressure, and vector search enables modern AI/ML applications.

**Production Hardening**:
- NetworkTopologyStrategy with RF ≥ 3, LOCAL_QUORUM consistency, authentication/authorization, TLS encryption, and automated repairs ensure data durability and security.

---

**Sources:**

- [Apache Cassandra Database: Complete 2025 Guide](https://www.knowi.com/blog/apache-cassandra-database-complete-2025-guide-architecture-use-cases/)
- [Apache Cassandra 2025: Performance, Community & What's Coming in 2026](https://axonops.com/blog/cassandra-in-2025-a-year-in-review)
- [Data modeling best practices - Amazon Keyspaces](https://docs.aws.amazon.com/keyspaces/latest/devguide/data-modeling.html)
- [Basic Rules of Apache Cassandra Data Modeling](https://www.datastax.com/blog/basic-rules-cassandra-data-modeling)
- [Replication Strategies in Cassandra - Baeldung](https://www.baeldung.com/cassandra-replication-partitioning)
- [Consistency Levels in Cassandra - Baeldung](https://www.baeldung.com/cassandra-consistency-levels)
- [Cassandra's Tunable Consistency Model](https://medium.com/@preethikcs01/cassandras-tunable-consistency-model-a-game-changer-for-distributed-systems-%EF%B8%8F-132a295749ce)
- [Tuning Java resources for Apache Cassandra](https://docs.datastax.com/en/cassandra-oss/3.0/cassandra/operations/opsTuneJVM.html)
- [Tuning JVM for Apache Cassandra](https://medium.com/@serg-digitalis/tuning-jvm-for-apache-cassandra-cea066c858df)
- [Unified Compaction Strategy (UCS)](https://cassandra.apache.org/doc/latest/cassandra/managing/operating/compaction/ucs.html)
- [Deep Dive into Cassandra Compaction Strategies](https://medium.com/@sevanthi404rt/deep-dive-into-cassandra-compaction-strategies-4eb76316ebdf)
- [Storage-attached indexing (SAI) concepts](https://cassandra.apache.org/doc/latest/cassandra/developing/cql/indexing/sai/sai-concepts.html)
- [Boost Cassandra Data Models with Storage-Attached Indexing](https://thenewstack.io/boost-cassandra-data-models-with-storage-attached-indexing/)
- [Lightweight Transactions (LWTs) in Apache Cassandra](https://axonops.com/blog/paxos-v2-and-lightweight-transactions)
- [Apache Cassandra SSL Guide](https://axonops.com/blog/apache-cassandra-ssl-deep-dive)
- [Monitoring a Cassandra cluster](https://docs.datastax.com/en/cassandra-oss/3.x/cassandra/operations/opsMonitoring.html)
- [Cassandra Metrics Reference](https://axonops.com/docs/data-platforms/cassandra/operations/jmx-reference/metrics/)
- [Repair - Apache Cassandra Documentation](https://cassandra.apache.org/doc/stable/cassandra/managing/operating/repair.html)
- [Incremental Repair Improvements in Cassandra 4](https://thelastpickle.com/blog/2018/09/10/incremental-repair-improvements-in-cassandra-4.html)
- [Backups - Apache Cassandra Documentation](https://cassandra.apache.org/doc/4.0/cassandra/operating/backups.html)
- [Cassandra Backup and Restore Methods](https://www.securekloud.com/blog/cassandra-backup-and-restore-methods/)
- [How to Upgrade Cassandra from 4.x to 5.x Without Downtime](https://digitalis.io/post/how-to-upgrade-cassandra-from-4-x-to-5-x-without-downtime-3-proven-steps)
- [K8ssandra, Apache Cassandra on Kubernetes](https://docs.k8ssandra.io)
- [Four New Apache Cassandra 5.0 Features to Be Excited About](https://medium.com/@Sarahmoradi/four-new-apache-cassandra-5-0-features-to-be-excited-about-aeddd1406ed2)
- [Apache Cassandra 5.0 Features: Vector Search](https://cassandra.apache.org/_/blog/Apache-Cassandra-5.0-Features-Vector-Search.html)
- [Trie Memtables and Trie-Indexed SSTables](https://cassandra.apache.org/_/blog/Apache-Cassandra-5.0-Features-Trie-Memtables-and-Trie-Indexed-SSTables.html)

---

**End of Apache Cassandra Guidelines**
