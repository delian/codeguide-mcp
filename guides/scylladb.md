# ScyllaDB Development Guidelines

This document provides mandatory standards for ScyllaDB database design, performance optimization, replication, and best practices.

---

**Agent Profile**: The ScyllaDB Expert
**Role**: Senior Database Engineer & Distributed Systems Specialist
**Objective**: Generate efficient, scalable, and maintainable ScyllaDB implementations.
**Tools**: ScyllaDB 5.x+, ScyllaDB Manager, CQL, ScyllaDB Monitoring Stack, cqlsh.

---

## 1. Core Philosophies: SCYLLA-FIRST

- **S**hard-per-Core: Leverage ScyllaDB's shard-per-core architecture
- **C**onsistency Levels: Choose appropriate consistency for multi-DC
- **Y**our Query Patterns: Design schema around access patterns, not data
- **L**ow Latency: Optimize for latency with LOCAL_QUORUM and shard-aware drivers
- **L**inear Scalability: Add nodes to scale horizontally
- **A**utonomous Operation: Leverage ScyllaDB's auto-tuning capabilities

---

## 2. ScyllaDB Architecture (CRITICAL UNDERSTANDING)

### A. Shard-per-Core Architecture

ScyllaDB's defining feature: each CPU core operates as an independent shard.

```
┌─────────────────────────────────────────────────────────────┐
│                         Node                                 │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │ Shard 0  │  │ Shard 1  │  │ Shard 2  │  │ Shard 3  │   │
│  │  CPU 0   │  │  CPU 1   │  │  CPU 2   │  │  CPU 3   │   │
│  │  Memory  │  │  Memory  │  │  Memory  │  │  Memory  │   │
│  │  I/O     │  │  I/O     │  │  I/O     │  │  I/O     │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
└─────────────────────────────────────────────────────────────┘
```

**Key Characteristics:**
- Each shard has dedicated CPU, memory, network, and storage
- No locks, no cross-shard communication for operations
- Linear scalability: 2x cores = 2x performance
- Zero garbage collection (C++ implementation)

### B. Token Ring Architecture

```
         Token Range: 0 - 2^63
    ┌────────────────────────────────┐
    │                                │
Node 1 ───────────────────────────── Node 2
    │                                │
    │         Data distributed       │
    │         by partition key       │
    │         hash (murmur3)         │
    │                                │
Node 4 ───────────────────────────── Node 3
    │                                │
    └────────────────────────────────┘
```

**Distribution:**
- Data distributed via consistent hashing
- Each node owns token ranges
- Replication factor (RF) determines copies
- Virtual nodes (vnodes) for balanced distribution

### C. ScyllaDB vs Cassandra

| Feature | ScyllaDB | Cassandra |
|---------|----------|-----------|
| Language | C++ | Java |
| GC Pauses | None | Yes (can be seconds) |
| Architecture | Shard-per-core | Thread-per-core |
| Performance | 2-5x higher throughput | Baseline |
| Latency | 75% lower p99 | Baseline |
| Compatibility | CQL compatible | Native |

---

## 3. Data Modeling (MANDATORY)

### A. Partition Key Design (CRITICAL)

**Design Principle:** Schema follows queries, not data structure.

```sql
-- ❌ BAD: Single partition for all users (hot partition)
CREATE TABLE users (
    org_id UUID,
    user_id UUID,
    name TEXT,
    email TEXT,
    PRIMARY KEY (org_id, user_id)
);
-- All users in same org = same partition = poor performance

-- ✅ GOOD: User ID as partition key (even distribution)
CREATE TABLE users (
    user_id UUID,
    org_id UUID,
    name TEXT,
    email TEXT,
    created_at TIMESTAMP,
    PRIMARY KEY (user_id)
);

-- ✅ BETTER: Composite partition key for bucketing
CREATE TABLE user_events (
    user_id UUID,
    date TEXT,          -- '2026-02-06' for bucketing
    event_time TIMESTAMP,
    event_type TEXT,
    data TEXT,
    PRIMARY KEY ((user_id, date), event_time)
);
-- Partition per user per day = bounded partition size
```

### B. Partition Key Selection Criteria

**High Cardinality:**
```sql
-- ❌ BAD: Low cardinality (2-3 values)
PRIMARY KEY (status)  -- 'active', 'inactive', 'suspended'

-- ✅ GOOD: High cardinality
PRIMARY KEY (user_id)  -- Unique per user

-- ✅ BEST: Composite with bucketing
PRIMARY KEY ((user_id, year_month), timestamp)
```

**Even Distribution:**
```sql
-- ❌ BAD: Skewed distribution
CREATE TABLE orders (
    country TEXT,
    order_id UUID,
    amount DECIMAL,
    PRIMARY KEY (country, order_id)
);
-- 80% of orders from US = hot partition

-- ✅ GOOD: Even distribution
CREATE TABLE orders (
    order_id UUID,
    country TEXT,
    user_id UUID,
    amount DECIMAL,
    created_at TIMESTAMP,
    PRIMARY KEY (order_id)
);
```

**Bounded Growth:**
```sql
-- ❌ BAD: Unbounded partition (will grow forever)
CREATE TABLE user_activity (
    user_id UUID,
    timestamp TIMESTAMP,
    activity TEXT,
    PRIMARY KEY (user_id, timestamp)
);
-- Active user's partition grows without limit

-- ✅ GOOD: Time-bucketed partition
CREATE TABLE user_activity (
    user_id UUID,
    bucket TEXT,  -- 'YYYY-MM' or 'YYYY-MM-DD'
    timestamp TIMESTAMP,
    activity TEXT,
    PRIMARY KEY ((user_id, bucket), timestamp)
) WITH CLUSTERING ORDER BY (timestamp DESC);
-- Maximum 1 month or 1 day of data per partition
```

### C. Clustering Keys

```sql
-- Clustering key determines sort order within partition
CREATE TABLE sensor_data (
    sensor_id UUID,
    timestamp TIMESTAMP,
    temperature DECIMAL,
    humidity DECIMAL,
    PRIMARY KEY (sensor_id, timestamp)
) WITH CLUSTERING ORDER BY (timestamp DESC);
-- Latest readings first

-- Composite clustering key
CREATE TABLE messages (
    conversation_id UUID,
    year_month TEXT,
    timestamp TIMESTAMP,
    message_id UUID,
    sender_id UUID,
    content TEXT,
    PRIMARY KEY ((conversation_id, year_month), timestamp, message_id)
) WITH CLUSTERING ORDER BY (timestamp DESC, message_id DESC);
```

### D. Data Modeling Best Practices

**1. Denormalization:**
```sql
-- Don't be afraid to duplicate data for different query patterns

-- Query 1: Get user by ID
CREATE TABLE users_by_id (
    user_id UUID PRIMARY KEY,
    email TEXT,
    name TEXT,
    org_id UUID
);

-- Query 2: Get users by organization
CREATE TABLE users_by_org (
    org_id UUID,
    user_id UUID,
    email TEXT,
    name TEXT,
    PRIMARY KEY (org_id, user_id)
);
-- Same data, different access patterns
```

**2. Partition Size Limits:**
- Keep partitions under 100 MB
- Maximum 2 billion cells per partition (row × columns)
- Monitor partition sizes with `nodetool cfstats`

**3. Collections:**
```sql
-- Use collections for small, bounded data
CREATE TABLE users (
    user_id UUID PRIMARY KEY,
    name TEXT,
    email_addresses SET<TEXT>,      -- Small set
    preferences MAP<TEXT, TEXT>,     -- Limited map
    recent_logins LIST<TIMESTAMP>    -- Bounded list
);

-- ❌ AVOID: Unbounded collections
-- tags SET<TEXT>  -- Can grow without limit
```

---

## 4. Replication Strategies (MANDATORY)

### A. NetworkTopologyStrategy (Production Standard)

```sql
-- Multi-datacenter replication
CREATE KEYSPACE production WITH REPLICATION = {
    'class': 'NetworkTopologyStrategy',
    'DC1': 3,  -- 3 replicas in datacenter 1
    'DC2': 3,  -- 3 replicas in datacenter 2
    'DC3': 2   -- 2 replicas in datacenter 3
};

-- Single datacenter
CREATE KEYSPACE app_data WITH REPLICATION = {
    'class': 'NetworkTopologyStrategy',
    'datacenter1': 3
};
```

**Replication Factor Guidelines:**
- **RF=3**: Production standard (survives 1 node failure with QUORUM)
- **RF=2**: Development only (no fault tolerance with QUORUM)
- **RF=1**: Testing only (no redundancy)

### B. SimpleStrategy (Development Only)

```sql
-- ❌ NEVER use in production
CREATE KEYSPACE test WITH REPLICATION = {
    'class': 'SimpleStrategy',
    'replication_factor': 3
};
-- Cannot specify per-DC replication
```

### C. Multi-Datacenter Replication

**Data Flow:**
```
DC1 (US-EAST)                    DC2 (EU-WEST)
┌──────────┐                     ┌──────────┐
│  Node 1  │──────async────────→ │  Node 4  │
│  Node 2  │──────async────────→ │  Node 5  │
│  Node 3  │──────async────────→ │  Node 6  │
└──────────┘                     └──────────┘
```

**Configuration:**
```sql
-- Configure snitch (topology awareness)
-- In scylla.yaml:
endpoint_snitch: GossipingPropertyFileSnitch

-- In cassandra-rackdc.properties:
dc=DC1
rack=RACK1
```

**Best Practices:**
- Asynchronous replication between DCs
- Use LOCAL_QUORUM for reads/writes
- Each DC should have >= RF nodes
- Network latency between DCs: < 300ms ideal

---

## 5. Consistency Levels (CRITICAL)

### A. Consistency Level Overview

```
Consistency Level    Replicas   Cross-DC   Use Case
─────────────────────────────────────────────────────────────
ANY                  1 (hint)   No         Never use
ONE                  1          No         Low consistency
TWO                  2          No         Rarely used
THREE                3          No         Rarely used
QUORUM               n/2 + 1    Yes        Strong consistency
ALL                  n          Yes        Highest consistency
LOCAL_ONE            1          No         Local read
LOCAL_QUORUM         n/2 + 1    No         ✅ Multi-DC standard
EACH_QUORUM          n/2 + 1    Yes        Writes to all DCs
```

### B. Recommended Consistency Levels

**Multi-Datacenter Deployment:**
```python
# ✅ RECOMMENDED: LOCAL_QUORUM for both reads and writes
from cassandra.cluster import Cluster
from cassandra import ConsistencyLevel

cluster = Cluster(['10.0.0.1', '10.0.0.2'])
session = cluster.connect('production')

# Write with LOCAL_QUORUM
query = "INSERT INTO users (user_id, name, email) VALUES (?, ?, ?)"
prepared = session.prepare(query)
prepared.consistency_level = ConsistencyLevel.LOCAL_QUORUM
session.execute(prepared, (user_id, name, email))

# Read with LOCAL_QUORUM
query = "SELECT * FROM users WHERE user_id = ?"
prepared = session.prepare(query)
prepared.consistency_level = ConsistencyLevel.LOCAL_QUORUM
result = session.execute(prepared, (user_id,))
```

**Single Datacenter:**
```python
# Use QUORUM for reads and writes
prepared.consistency_level = ConsistencyLevel.QUORUM
```

### C. Consistency Level Selection

```sql
-- Write: LOCAL_QUORUM, Read: LOCAL_QUORUM
-- Strong consistency within DC, low latency

-- Write: EACH_QUORUM, Read: LOCAL_QUORUM
-- Strong consistency across DCs, higher write latency

-- Write: LOCAL_QUORUM, Read: QUORUM
-- Strong consistency, acceptable for most cases
```

**Availability vs Consistency:**

| CL | Writes Succeed | Reads Succeed | Latency | Consistency |
|----|----------------|---------------|---------|-------------|
| ONE | If any node up | If any node up | Lowest | Weak |
| LOCAL_QUORUM | If local majority up | If local majority up | Low | Strong (local) |
| QUORUM | If global majority up | If global majority up | Medium | Strong (global) |
| EACH_QUORUM | If majority in ALL DCs | N/A | High | Strongest |
| ALL | If all replicas up | If all replicas up | Highest | Strongest |

---

## 6. Performance Optimization and Latency (MANDATORY)

### A. Shard-Aware Drivers

```python
# Python driver with shard awareness
from cassandra.cluster import Cluster, ExecutionProfile
from cassandra.policies import TokenAwarePolicy, DCAwareRoundRobinPolicy

# ✅ RECOMMENDED: Shard-aware connection
profile = ExecutionProfile(
    load_balancing_policy=TokenAwarePolicy(
        DCAwareRoundRobinPolicy(local_dc='DC1')
    )
)

cluster = Cluster(
    ['10.0.0.1', '10.0.0.2', '10.0.0.3'],
    execution_profiles={'default': profile},
    # DON'T specify shard-aware port in contact points
    # Driver discovers it automatically
)

session = cluster.connect('keyspace_name')
```

```java
// Java driver with shard awareness
import com.datastax.oss.driver.api.core.CqlSession;
import com.datastax.oss.driver.api.core.config.DriverConfigLoader;

CqlSession session = CqlSession.builder()
    .addContactPoint(new InetSocketAddress("10.0.0.1", 9042))
    .withLocalDatacenter("DC1")
    .withConfigLoader(DriverConfigLoader.fromClasspath("application.conf"))
    .build();

// In application.conf:
// datastax-java-driver {
//   basic.load-balancing-policy {
//     class = DefaultLoadBalancingPolicy
//     local-datacenter = DC1
//   }
// }
```

```javascript
// Node.js driver with token awareness
const cassandra = require('cassandra-driver');

const client = new cassandra.Client({
    contactPoints: ['10.0.0.1', '10.0.0.2', '10.0.0.3'],
    localDataCenter: 'DC1',
    policies: {
        loadBalancing: new cassandra.policies.loadBalancing.TokenAwarePolicy(
            new cassandra.policies.loadBalancing.DCAwareRoundRobinPolicy('DC1')
        )
    },
    keyspace: 'production'
});
```

**Benefits:**
- Connects directly to shard handling the data
- Eliminates inter-shard communication
- Reduces latency by 30-50%

### B. Prepared Statements (CRITICAL)

```python
# ❌ BAD: String concatenation (security risk + performance)
user_id = "123"
query = f"SELECT * FROM users WHERE user_id = {user_id}"
session.execute(query)

# ✅ GOOD: Prepared statements
query = "SELECT * FROM users WHERE user_id = ?"
prepared = session.prepare(query)
session.execute(prepared, (user_id,))
```

**Benefits:**
- Statement parsed and cached once
- Token-aware routing optimization
- Protection against CQL injection
- 10-100x faster for repeated queries

### C. Batch Operations

```sql
-- ✅ GOOD: Batched writes to same partition
BEGIN BATCH
    INSERT INTO user_events (user_id, date, event_time, event_type)
    VALUES (uuid1, '2026-02-06', timestamp1, 'login');
    INSERT INTO user_events (user_id, date, event_time, event_type)
    VALUES (uuid1, '2026-02-06', timestamp2, 'view_page');
    INSERT INTO user_events (user_id, date, event_time, event_type)
    VALUES (uuid1, '2026-02-06', timestamp3, 'logout');
APPLY BATCH;

-- ❌ BAD: Batch across multiple partitions (performance penalty)
BEGIN BATCH
    INSERT INTO users (user_id, name) VALUES (uuid1, 'Alice');
    INSERT INTO users (user_id, name) VALUES (uuid2, 'Bob');
    INSERT INTO users (user_id, name) VALUES (uuid3, 'Charlie');
APPLY BATCH;
-- Coordinator must route to multiple nodes
```

**Batch Guidelines:**
- Use batches ONLY for same partition key
- Limit batch size to 100 statements
- Use UNLOGGED batches for performance
- Use LOGGED batches only when atomicity required

### D. Query Optimization

```sql
-- ✅ ALLOW FILTERING: Last resort only
-- Better: Create appropriate secondary index or materialized view

-- ❌ BAD: ALLOW FILTERING on large table
SELECT * FROM users WHERE email = 'user@example.com' ALLOW FILTERING;
-- Scans entire table

-- ✅ GOOD: Secondary index
CREATE INDEX ON users (email);
SELECT * FROM users WHERE email = 'user@example.com';

-- ✅ BETTER: Materialized view
CREATE MATERIALIZED VIEW users_by_email AS
    SELECT * FROM users
    WHERE email IS NOT NULL AND user_id IS NOT NULL
    PRIMARY KEY (email, user_id);
SELECT * FROM users_by_email WHERE email = 'user@example.com';
```

### E. Connection Pool Sizing

```python
# Python driver connection pool
from cassandra.cluster import Cluster

cluster = Cluster(
    ['10.0.0.1', '10.0.0.2'],
    protocol_version=4,
    # Core connections per host
    max_connections_per_host=5,
    # Maximum requests per connection
    max_requests_per_connection=32768,
    # Connection timeout
    connect_timeout=10
)
```

```java
// Java driver connection pool
datastax-java-driver {
  advanced.connection {
    max-requests-per-connection = 32768
    pool {
      local.size = 5
      remote.size = 1
    }
  }
}
```

**Sizing Guidelines:**
- Local DC: 5-10 connections per host
- Remote DC: 1-2 connections per host
- Monitor connection saturation
- Scale connections with load

### F. Latency Optimization Checklist

- [ ] Use shard-aware drivers
- [ ] Use prepared statements for all queries
- [ ] Use LOCAL_QUORUM consistency level
- [ ] Deploy in same region as application
- [ ] Use SSDs (NVMe preferred)
- [ ] Enable compression (LZ4 recommended)
- [ ] Monitor p99 latency, not average
- [ ] Use token-aware load balancing
- [ ] Batch statements to same partition only
- [ ] Avoid ALLOW FILTERING

---

## 7. Compaction Strategies (MANDATORY)

### A. Compaction Strategy Selection

```sql
-- STCS (Size-Tiered Compaction Strategy) - Default
CREATE TABLE logs (
    log_id UUID PRIMARY KEY,
    timestamp TIMESTAMP,
    message TEXT,
    level TEXT
) WITH compaction = {
    'class': 'SizeTieredCompactionStrategy',
    'min_threshold': 4,
    'max_threshold': 32
};
-- Use for: Write-heavy workloads, random updates
-- Space overhead: 100% (2x storage needed during compaction)

-- LCS (Leveled Compaction Strategy)
CREATE TABLE users (
    user_id UUID PRIMARY KEY,
    name TEXT,
    email TEXT,
    created_at TIMESTAMP
) WITH compaction = {
    'class': 'LeveledCompactionStrategy',
    'sstable_size_in_mb': 160
};
-- Use for: Read-heavy workloads, small frequent writes
-- Space overhead: 10% (more predictable)
-- 90% of reads touch 1 SSTable

-- ICS (Incremental Compaction Strategy) - Enterprise Only
CREATE TABLE events (
    event_id UUID PRIMARY KEY,
    timestamp TIMESTAMP,
    data TEXT
) WITH compaction = {
    'class': 'IncrementalCompactionStrategy',
    'sstable_size_in_mb': 1024
};
-- Use for: Large datasets, better than STCS
-- Space overhead: 50% (half of STCS)
-- Fixes STCS space amplification issues

-- TWCS (Time-Window Compaction Strategy)
CREATE TABLE metrics (
    sensor_id UUID,
    bucket TEXT,
    timestamp TIMESTAMP,
    value DECIMAL,
    PRIMARY KEY ((sensor_id, bucket), timestamp)
) WITH compaction = {
    'class': 'TimeWindowCompactionStrategy',
    'compaction_window_unit': 'DAYS',
    'compaction_window_size': 1
};
-- Use for: Time-series data, IoT, logs
-- Never compacts across time windows
-- Perfect for TTL-based expiration
```

### B. Compaction Strategy Decision Tree

```
Is data time-series? (write once, expire)
  └─ YES → TWCS
  └─ NO  → Continue

Using ScyllaDB Enterprise?
  └─ YES → ICS (best all-around)
  └─ NO  → Continue

Read-heavy with uniform distribution?
  └─ YES → LCS
  └─ NO  → STCS (default)
```

### C. Compaction Monitoring

```bash
# Check compaction statistics
nodetool compactionstats

# Check per-table compaction
nodetool cfstats keyspace.table | grep Compaction

# Monitor compaction in ScyllaDB Manager
# Dashboard → Compaction → Active Tasks
```

---

## 8. Secondary Indexes and Materialized Views

### A. Secondary Index Types

**Global Secondary Indexes (GSI):**
```sql
-- Create global secondary index
CREATE INDEX user_email_idx ON users (email);

-- Query using index
SELECT * FROM users WHERE email = 'user@example.com';

-- How it works:
-- - Creates hidden table: users_user_email_idx
-- - Indexed data distributed across all nodes
-- - Query may require scatter-gather across cluster
```

**Local Secondary Indexes (LSI):**
```sql
-- Create local secondary index (same partition key)
CREATE TABLE orders (
    user_id UUID,
    order_id UUID,
    status TEXT,
    created_at TIMESTAMP,
    PRIMARY KEY (user_id, order_id)
);

CREATE INDEX orders_status_idx ON orders ((user_id), status);

-- Query with partition key + indexed column
SELECT * FROM orders WHERE user_id = ? AND status = 'pending';

-- Benefits:
-- - No cross-node communication
-- - All data on same node as partition
-- - Much faster than global index
```

### B. Materialized Views

```sql
-- Base table
CREATE TABLE users (
    user_id UUID PRIMARY KEY,
    email TEXT,
    name TEXT,
    org_id UUID,
    created_at TIMESTAMP
);

-- Materialized view for different access pattern
CREATE MATERIALIZED VIEW users_by_email AS
    SELECT user_id, email, name, org_id, created_at
    FROM users
    WHERE email IS NOT NULL AND user_id IS NOT NULL
    PRIMARY KEY (email, user_id);

-- Query by email (fast)
SELECT * FROM users_by_email WHERE email = 'user@example.com';

-- Another MV for organization queries
CREATE MATERIALIZED VIEW users_by_org AS
    SELECT user_id, email, name, org_id, created_at
    FROM users
    WHERE org_id IS NOT NULL AND user_id IS NOT NULL
    PRIMARY KEY (org_id, user_id);

-- Query by organization (fast)
SELECT * FROM users_by_org WHERE org_id = ?;
```

### C. Choosing Between Index Types

| Requirement | Solution | Notes |
|-------------|----------|-------|
| Query by non-PK column, have partition key | Local Secondary Index | Fastest, no scatter-gather |
| Query by non-PK column, no partition key | Global Secondary Index | Scatter-gather required |
| Complex query patterns | Materialized View | Full table copy |
| Need different sort order | Materialized View | Specify clustering order |
| Simple equality lookups | Secondary Index | Lower storage overhead |

**Production Status (October 2024):**
- ✅ Materialized Views: Production ready
- ✅ Global Secondary Indexes: Production ready
- ✅ Local Secondary Indexes: Production ready

---

## 9. Lightweight Transactions (LWT)

### A. Conditional Updates

```sql
-- Insert only if not exists
INSERT INTO users (user_id, email, name)
VALUES (uuid1, 'user@example.com', 'Alice')
IF NOT EXISTS;

-- Update only if condition met
UPDATE users
SET status = 'active'
WHERE user_id = uuid1
IF status = 'pending';

-- Delete only if condition met
DELETE FROM users
WHERE user_id = uuid1
IF status = 'inactive';

-- Compare-and-swap
UPDATE inventory
SET quantity = 95
WHERE product_id = uuid1
IF quantity = 100;
```

### B. Conditional Batches

```sql
-- ✅ GOOD: Conditional batch on same partition
BEGIN BATCH
    UPDATE account SET balance = 900 WHERE user_id = uuid1 IF balance >= 100;
    INSERT INTO transactions (user_id, tx_id, amount)
    VALUES (uuid1, uuid2, -100);
APPLY BATCH;

-- ❌ BAD: Conditional batch across partitions
BEGIN BATCH
    UPDATE account SET balance = 900 WHERE user_id = uuid1 IF balance >= 100;
    UPDATE account SET balance = 1100 WHERE user_id = uuid2;
APPLY BATCH;
-- Not supported
```

### C. LWT Performance Considerations

**Characteristics:**
- Uses Paxos consensus protocol
- 4 round trips (vs 1 for normal write)
- Latency: 2-4x normal writes
- ScyllaDB's LWT implementation is more efficient than Cassandra

**Best Practices:**
```python
# Check LWT result
result = session.execute(
    "UPDATE users SET email = ? WHERE user_id = ? IF email = ?",
    (new_email, user_id, old_email)
)

if result.was_applied:
    print("Update successful")
else:
    print("Condition failed, current value:", result.one())
```

**When to Use LWT:**
- User registration (email uniqueness)
- Inventory management (prevent overselling)
- Leader election
- Distributed locks

**When NOT to Use LWT:**
- High-throughput writes
- Eventually consistent data is acceptable
- Can use application-level logic

---

## 10. Security (MANDATORY)

### A. Authentication

**Password Authentication:**
```sql
-- Enable authentication (in scylla.yaml)
authenticator: PasswordAuthenticator

-- Create superuser
CREATE ROLE admin WITH PASSWORD = 'strong_password' AND SUPERUSER = true AND LOGIN = true;

-- Create application user
CREATE ROLE app_user WITH PASSWORD = 'app_password' AND LOGIN = true;
```

**Certificate-Based Authentication (TLS):**
```yaml
# scylla.yaml configuration
client_encryption_options:
    enabled: true
    certificate: /path/to/server.crt
    keyfile: /path/to/server.key
    truststore: /path/to/ca.pem
    require_client_auth: true
```

```python
# Python driver with TLS
from ssl import SSLContext, PROTOCOL_TLS, CERT_REQUIRED

ssl_context = SSLContext(PROTOCOL_TLS)
ssl_context.load_verify_locations('/path/to/ca.pem')
ssl_context.load_cert_chain(
    certfile='/path/to/client.crt',
    keyfile='/path/to/client.key'
)
ssl_context.verify_mode = CERT_REQUIRED

cluster = Cluster(
    ['10.0.0.1'],
    ssl_context=ssl_context,
    ssl_options={'ca_certs': '/path/to/ca.pem'}
)
```

### B. Authorization (RBAC)

```sql
-- Create custom role
CREATE ROLE read_only_user WITH LOGIN = true AND PASSWORD = 'password';

-- Grant permissions
GRANT SELECT ON KEYSPACE production TO read_only_user;

-- Create read-write role
CREATE ROLE app_writer WITH LOGIN = true AND PASSWORD = 'password';
GRANT SELECT, MODIFY ON KEYSPACE production TO app_writer;

-- Grant table-level permissions
GRANT SELECT ON production.users TO analytics_user;
GRANT MODIFY ON production.logs TO logger_app;

-- Revoke permissions
REVOKE MODIFY ON KEYSPACE production FROM read_only_user;

-- List roles and permissions
LIST ROLES;
LIST PERMISSIONS ON KEYSPACE production;
```

**Role Hierarchy:**
```sql
-- Create role hierarchy
CREATE ROLE developers;
GRANT SELECT ON ALL KEYSPACES TO developers;

CREATE ROLE alice WITH PASSWORD = 'pass' AND LOGIN = true;
GRANT developers TO alice;

-- Alice inherits SELECT on all keyspaces
```

### C. Encryption

**Encryption in Transit (TLS/SSL):**
```yaml
# Client-to-node encryption (scylla.yaml)
client_encryption_options:
    enabled: true
    certificate: /path/to/scylla.crt
    keyfile: /path/to/scylla.key
    truststore: /path/to/ca.pem
    require_client_auth: false  # Set true for mutual TLS

# Node-to-node encryption
server_encryption_options:
    internode_encryption: all  # all, dc, rack, or none
    certificate: /path/to/scylla.crt
    keyfile: /path/to/scylla.key
    truststore: /path/to/ca.pem
    require_client_auth: true
```

**Disable Weak TLS Versions:**
```yaml
# In scylla.yaml
client_encryption_options:
    enabled: true
    certificate: /path/to/scylla.crt
    keyfile: /path/to/scylla.key
    # Disable TLS 1.0 and 1.1
    priority_string: "SECURE128:-VERS-TLS1.0:-VERS-TLS1.1"
```

**Encryption at Rest (ScyllaDB Enterprise):**
```yaml
# scylla.yaml
data_encryption_options:
    enabled: true
    chunk_length_kb: 64
    cipher: AES/CBC/PKCS5Padding
    key_provider: KmipKeyProviderFactory
    kmip_host: kmip.example.com:5696
```

### D. Network Security

```yaml
# Bind to specific interface (scylla.yaml)
listen_address: 10.0.1.5
rpc_address: 10.0.1.5

# Firewall rules (example with iptables)
# CQL (client connections)
iptables -A INPUT -p tcp --dport 9042 -s 10.0.0.0/8 -j ACCEPT

# Inter-node communication
iptables -A INPUT -p tcp --dport 7000 -s 10.0.0.0/8 -j ACCEPT

# JMX (monitoring - restrict access)
iptables -A INPUT -p tcp --dport 7199 -s 10.0.1.0/24 -j ACCEPT

# Drop all other traffic
iptables -A INPUT -p tcp --dport 9042 -j DROP
iptables -A INPUT -p tcp --dport 7000 -j DROP
```

### E. Audit Logging (ScyllaDB Enterprise)

```yaml
# scylla.yaml
audit_logging_options:
    enabled: true
    logger: SyslogAuditWriter
    included_keyspaces: "production,sensitive_data"
    excluded_keyspaces: "system"
    included_categories: "AUTH,DCL,DDL,DML"
```

### F. Security Checklist

**Pre-Production:**
- [ ] Authentication enabled (not AllowAllAuthenticator)
- [ ] TLS encryption enabled (client-to-node)
- [ ] TLS encryption enabled (node-to-node)
- [ ] Weak TLS versions disabled (1.0, 1.1)
- [ ] Encryption at rest configured (Enterprise)
- [ ] Custom roles created (not using cassandra superuser)
- [ ] Least privilege principle applied
- [ ] Network access restricted (firewall/VPC)
- [ ] JMX access restricted
- [ ] Audit logging enabled (Enterprise)
- [ ] Default passwords changed
- [ ] Monitoring credentials secured

**Ongoing:**
- [ ] Rotate passwords periodically
- [ ] Review user permissions quarterly
- [ ] Monitor audit logs
- [ ] Update TLS certificates before expiry
- [ ] Apply security patches promptly
- [ ] Run with dedicated user (not root)

---

## 11. Monitoring and Operations

### A. ScyllaDB Monitoring Stack

**Setup:**
```bash
# Install ScyllaDB Monitoring Stack
git clone https://github.com/scylladb/scylla-monitoring.git
cd scylla-monitoring

# Configure prometheus targets
cat > prometheus/scylla_servers.yml <<EOF
- targets:
  - 10.0.0.1:9180
  - 10.0.0.2:9180
  - 10.0.0.3:9180
  labels:
    cluster: production
    dc: DC1
EOF

# Start monitoring stack
./start-all.sh -d /path/to/data/dir
```

**Access Dashboards:**
- Grafana: http://localhost:3000 (admin/admin)
- Prometheus: http://localhost:9090

### B. Key Metrics to Monitor

**Latency Metrics:**
```bash
# p99 latency (critical)
scylla_storage_proxy_coordinator_read_latency_p99
scylla_storage_proxy_coordinator_write_latency_p99

# p95 latency
scylla_storage_proxy_coordinator_read_latency_p95
scylla_storage_proxy_coordinator_write_latency_p95

# Mean latency
scylla_storage_proxy_coordinator_read_latency_mean
scylla_storage_proxy_coordinator_write_latency_mean
```

**Throughput Metrics:**
```bash
# Operations per second
scylla_storage_proxy_coordinator_reads_per_sec
scylla_storage_proxy_coordinator_writes_per_sec

# Bytes per second
scylla_transport_cql_reads_per_second
scylla_transport_cql_writes_per_second
```

**Resource Utilization:**
```bash
# CPU usage per shard
scylla_reactor_utilization

# Memory usage
scylla_memory_allocated_memory
scylla_memory_free_memory

# Disk I/O
scylla_io_queue_requests
scylla_io_queue_latency
```

**Cluster Health:**
```bash
# Nodes up
scylla_node_operation_mode

# Pending compactions
scylla_compaction_manager_pending_tasks

# Pending hints
scylla_hints_pending_drains
```

### C. nodetool Commands

```bash
# Cluster status
nodetool status

# Node statistics
nodetool info

# Table statistics
nodetool tablestats keyspace.table

# Compaction statistics
nodetool compactionstats

# Repair cluster
# DON'T run manually - use ScyllaDB Manager
scylla-manager repair --cluster prod

# Flush memtables to disk
nodetool flush

# Clear caches
nodetool clearsnapshot

# View ring token distribution
nodetool ring

# Decommission node (graceful removal)
nodetool decommission

# Drain node (before restart)
nodetool drain
```

### D. ScyllaDB Manager

**Installation:**
```bash
# Install ScyllaDB Manager server
sudo yum install scylla-manager-server
sudo systemctl start scylla-manager

# Install agent on each ScyllaDB node
sudo yum install scylla-manager-agent
sudo systemctl start scylla-manager-agent
```

**Operations:**
```bash
# Add cluster
sctool cluster add --host 10.0.0.1 --name production

# Schedule repair (weekly)
sctool repair --cluster production --interval 7d

# Schedule backup
sctool backup --cluster production \
    --location s3:my-bucket/backups \
    --interval 1d \
    --retention 30

# Check repair status
sctool task progress repair/cluster-id

# Health check
sctool status --cluster production
```

### E. Operational Best Practices

**Repairs:**
- Run repairs weekly (via ScyllaDB Manager)
- NEVER run `nodetool repair` manually in production
- Use ScyllaDB Manager's distributed repair
- Monitor repair progress and errors

**Backups:**
```bash
# Snapshot backup
nodetool snapshot -t snapshot_name keyspace_name

# Incremental backup (via ScyllaDB Manager)
sctool backup --cluster prod --location s3:bucket/path
```

**Rolling Restarts:**
```bash
# Proper rolling restart procedure
for node in node1 node2 node3; do
    echo "Restarting $node"
    ssh $node "nodetool drain"
    ssh $node "sudo systemctl restart scylla-server"
    sleep 60  # Wait for node to join
    nodetool status  # Verify node is UP
done
```

**Adding Nodes:**
```bash
# 1. Install ScyllaDB on new node
# 2. Configure scylla.yaml (seeds, cluster_name, etc.)
# 3. Start ScyllaDB
sudo systemctl start scylla-server

# 4. Verify node joined
nodetool status

# 5. Wait for streaming to complete
nodetool netstats

# 6. Run cleanup on existing nodes (removes old data)
nodetool cleanup keyspace_name
```

---

## 12. Migration Strategies

### A. Migrating from Cassandra to ScyllaDB

ScyllaDB is a drop-in replacement for Apache Cassandra with full CQL compatibility.

**Migration Approaches:**

**1. Dual-Write Migration (Zero Downtime):**
```
Phase 1: Setup
┌─────────────┐
│ Application │
│   (reads)   │──────────→ Cassandra Cluster
└─────────────┘

Phase 2: Dual-Write
┌─────────────┐
│ Application │
│             │──────────→ Cassandra Cluster
│  (writes)   │──────────→ ScyllaDB Cluster (new)
└─────────────┘

Phase 3: Backfill Data
ScyllaDB ←─────── Cassandra
         (sstableloader/Spark)

Phase 4: Switch Reads
┌─────────────┐
│ Application │──────────→ ScyllaDB Cluster
│   (reads)   │
│             │──────────→ Cassandra Cluster (writes only)
└─────────────┘

Phase 5: Complete
┌─────────────┐
│ Application │──────────→ ScyllaDB Cluster
└─────────────┘
```

**Implementation:**
```python
# Python example with dual-write
from cassandra.cluster import Cluster

# Connect to both clusters
cassandra_cluster = Cluster(['cassandra-host'])
scylla_cluster = Cluster(['scylla-host'])

cassandra_session = cassandra_cluster.connect('keyspace')
scylla_session = scylla_cluster.connect('keyspace')

def write_data(user_id, name, email):
    query = "INSERT INTO users (user_id, name, email) VALUES (?, ?, ?)"

    # Write to Cassandra (primary)
    cassandra_session.execute(query, (user_id, name, email))

    # Write to ScyllaDB (shadow)
    try:
        scylla_session.execute(query, (user_id, name, email))
    except Exception as e:
        # Log but don't fail on ScyllaDB write
        logger.error(f"ScyllaDB write failed: {e}")

def read_data(user_id):
    query = "SELECT * FROM users WHERE user_id = ?"
    # Read from Cassandra initially
    return cassandra_session.execute(query, (user_id,)).one()
```

**2. Snapshot and Restore Migration:**
```bash
# On Cassandra cluster: Take snapshot
nodetool snapshot -t migration_snapshot keyspace_name

# Copy snapshots to ScyllaDB nodes
# Data location: /var/lib/cassandra/data/keyspace/table/snapshots/

# On ScyllaDB cluster: Restore using sstableloader
for table in users orders products; do
    sstableloader -d scylla-node1,scylla-node2,scylla-node3 \
        /path/to/snapshots/keyspace/$table/
done

# Verify data count
cqlsh -e "SELECT COUNT(*) FROM keyspace.users;"
```

**3. Spark-Based Migration (Large Datasets):**
```scala
// Scala Spark job for migration
import com.datastax.spark.connector._
import org.apache.spark.sql.SparkSession

val spark = SparkSession.builder()
  .appName("Cassandra to ScyllaDB Migration")
  .config("spark.cassandra.connection.host", "cassandra-host")
  .getOrCreate()

val scyllaConfig = Map(
  "spark.cassandra.connection.host" -> "scylla-host",
  "spark.cassandra.output.consistency.level" -> "LOCAL_QUORUM"
)

// Read from Cassandra
val data = spark.read
  .format("org.apache.spark.sql.cassandra")
  .options(Map("table" -> "users", "keyspace" -> "production"))
  .load()

// Write to ScyllaDB
data.write
  .format("org.apache.spark.sql.cassandra")
  .options(scyllaConfig ++ Map("table" -> "users", "keyspace" -> "production"))
  .mode("append")
  .save()
```

### B. Schema Migrations

**Schema Versioning Pattern:**
```sql
-- Create schema version tracking table
CREATE TABLE IF NOT EXISTS schema_migrations (
    version INT PRIMARY KEY,
    description TEXT,
    applied_at TIMESTAMP,
    script TEXT
);

-- Migration script structure
-- V001__initial_schema.cql
CREATE KEYSPACE IF NOT EXISTS production WITH REPLICATION = {
    'class': 'NetworkTopologyStrategy', 'DC1': 3
};

CREATE TABLE IF NOT EXISTS users (
    user_id UUID PRIMARY KEY,
    email TEXT,
    name TEXT,
    created_at TIMESTAMP
);

INSERT INTO schema_migrations (version, description, applied_at, script)
VALUES (1, 'Initial schema', toTimestamp(now()), 'V001__initial_schema.cql');

-- V002__add_user_status.cql
ALTER TABLE users ADD status TEXT;

INSERT INTO schema_migrations (version, description, applied_at, script)
VALUES (2, 'Add user status column', toTimestamp(now()), 'V002__add_user_status.cql');
```

**Migration Script Runner (Python):**
```python
#!/usr/bin/env python3
from cassandra.cluster import Cluster
import glob
import re

def get_current_version(session):
    """Get the current schema version"""
    try:
        result = session.execute(
            "SELECT MAX(version) as version FROM schema_migrations"
        ).one()
        return result.version if result else 0
    except Exception:
        return 0

def apply_migrations(session, migration_dir):
    """Apply pending migrations"""
    current_version = get_current_version(session)
    migration_files = sorted(glob.glob(f"{migration_dir}/V*.cql"))

    for migration_file in migration_files:
        # Extract version from filename (e.g., V001__description.cql)
        match = re.match(r'V(\d+)__(.+)\.cql', migration_file.split('/')[-1])
        if not match:
            continue

        version = int(match.group(1))
        description = match.group(2).replace('_', ' ')

        if version <= current_version:
            print(f"Skipping migration {version}: {description} (already applied)")
            continue

        print(f"Applying migration {version}: {description}")

        with open(migration_file, 'r') as f:
            statements = f.read().split(';')
            for statement in statements:
                statement = statement.strip()
                if statement:
                    session.execute(statement)

        # Record migration
        session.execute(
            """
            INSERT INTO schema_migrations (version, description, applied_at, script)
            VALUES (?, ?, toTimestamp(now()), ?)
            """,
            (version, description, migration_file)
        )

        print(f"Migration {version} applied successfully")

# Usage
cluster = Cluster(['localhost'])
session = cluster.connect('production')
apply_migrations(session, './migrations')
```

### C. Data Migration Between Keyspaces

```sql
-- Create new keyspace with different replication
CREATE KEYSPACE production_v2 WITH REPLICATION = {
    'class': 'NetworkTopologyStrategy',
    'DC1': 3,
    'DC2': 3  -- Adding new datacenter
};

-- Recreate tables in new keyspace
CREATE TABLE production_v2.users (
    user_id UUID PRIMARY KEY,
    email TEXT,
    name TEXT,
    status TEXT,
    created_at TIMESTAMP
);
```

```python
# Python script for data migration
from cassandra.cluster import Cluster
from cassandra.query import SimpleStatement

cluster = Cluster(['localhost'])
session = cluster.connect()

# Read from old keyspace
source_query = "SELECT * FROM production.users"
source_stmt = SimpleStatement(source_query, fetch_size=1000)

# Prepare insert for new keyspace
target_query = """
    INSERT INTO production_v2.users (user_id, email, name, status, created_at)
    VALUES (?, ?, ?, ?, ?)
"""
target_prepared = session.prepare(target_query)

# Migrate data in batches
count = 0
for row in session.execute(source_stmt):
    session.execute(target_prepared, (
        row.user_id,
        row.email,
        row.name,
        row.status if hasattr(row, 'status') else 'active',
        row.created_at
    ))
    count += 1
    if count % 1000 == 0:
        print(f"Migrated {count} rows")

print(f"Migration complete: {count} total rows")
```

### D. Migration Checklist

- [ ] Test migration on staging environment first
- [ ] Verify schema compatibility
- [ ] Estimate migration time based on data volume
- [ ] Plan dual-write period (minimum 1-2 weeks)
- [ ] Set up monitoring for both clusters during migration
- [ ] Verify data consistency after backfill
- [ ] Test application with ScyllaDB before switching reads
- [ ] Prepare rollback plan
- [ ] Document migration steps and timeline
- [ ] Schedule migration during low-traffic period

---

## 13. Upgrade Strategies

### A. ScyllaDB Version Upgrades

**Upgrade Types:**
- **Minor Version** (e.g., 5.2.1 → 5.2.5): Bug fixes, low risk
- **Major Version** (e.g., 5.x → 6.x): New features, higher risk

**Rolling Upgrade Procedure (Zero Downtime):**

```bash
#!/bin/bash
# rolling_upgrade.sh

NODES=(node1 node2 node3 node4 node5 node6)
NEW_VERSION="5.4.0"

for node in "${NODES[@]}"; do
    echo "=== Upgrading $node to $NEW_VERSION ==="

    # 1. Drain node
    ssh $node "nodetool drain"
    echo "Node drained"

    # 2. Stop ScyllaDB
    ssh $node "sudo systemctl stop scylla-server"
    echo "ScyllaDB stopped"

    # 3. Upgrade package
    ssh $node "sudo yum update scylla -y" # or apt-get for Ubuntu
    echo "Package upgraded to $NEW_VERSION"

    # 4. Start ScyllaDB
    ssh $node "sudo systemctl start scylla-server"
    echo "ScyllaDB started"

    # 5. Wait for node to become UN (Up/Normal)
    while true; do
        status=$(nodetool status | grep $node | awk '{print $1}')
        if [ "$status" == "UN" ]; then
            echo "$node is UP and NORMAL"
            break
        fi
        echo "Waiting for $node to join... (current status: $status)"
        sleep 10
    done

    # 6. Wait for schema agreement
    echo "Waiting for schema agreement..."
    sleep 30

    # 7. Verify version
    version=$(ssh $node "scylla --version")
    echo "Verified version: $version"

    # 8. Wait before next node (safety buffer)
    echo "Waiting 2 minutes before upgrading next node..."
    sleep 120
done

echo "=== Upgrade complete ==="
nodetool status
```

### B. Upgrade Best Practices

**Pre-Upgrade Checklist:**
- [ ] Review release notes and breaking changes
- [ ] Test upgrade on staging environment
- [ ] Take full cluster snapshot
- [ ] Verify cluster health (`nodetool status`)
- [ ] Check no pending repairs or compactions
- [ ] Verify sufficient disk space (30%+ free)
- [ ] Schedule during low-traffic window
- [ ] Notify team and stakeholders
- [ ] Prepare rollback plan
- [ ] Update monitoring dashboards for new version

**During Upgrade:**
```bash
# Monitor cluster health continuously
watch -n 5 'nodetool status'

# Monitor logs on upgrading node
ssh node1 "sudo journalctl -u scylla-server -f"

# Check for errors
ssh node1 "sudo journalctl -u scylla-server | grep -i error | tail -20"

# Verify replication
nodetool describecluster
```

**Post-Upgrade Verification:**
```bash
# Verify all nodes on new version
for node in node1 node2 node3; do
    echo "=== $node ==="
    ssh $node "scylla --version"
done

# Check cluster status
nodetool status

# Verify schema agreement
nodetool describecluster | grep "Schema versions"

# Run test queries
cqlsh -e "SELECT * FROM system.local;"

# Check performance metrics
# Access Grafana dashboards and verify latency, throughput
```

### C. Upgrading SSTables

After version upgrade, upgrade SSTable format:

```bash
# Upgrade SSTables to new format (one node at a time)
nodetool upgradesstables

# Or upgrade specific keyspace
nodetool upgradesstables keyspace_name

# Monitor progress
nodetool compactionstats
```

**Best Practices:**
- Run during low-traffic period (I/O intensive)
- One node at a time
- Monitor disk I/O and latency
- Can take hours for large datasets

### D. ScyllaDB Manager Upgrade

```bash
# Stop ScyllaDB Manager
sudo systemctl stop scylla-manager

# Upgrade package
sudo yum update scylla-manager -y

# Start ScyllaDB Manager
sudo systemctl start scylla-manager

# Verify version
sctool version

# Check all clusters are healthy
sctool status
```

---

## 14. Rollback Strategies

### A. Version Rollback

**When to Rollback:**
- Critical bugs discovered in new version
- Performance degradation
- Application incompatibility
- Data corruption

**Rollback Procedure:**

```bash
#!/bin/bash
# rollback.sh

NODES=(node1 node2 node3)
PREVIOUS_VERSION="5.2.1"

echo "=== ROLLBACK to $PREVIOUS_VERSION ==="
echo "WARNING: This will rollback to previous version"
read -p "Continue? (yes/no): " confirm

if [ "$confirm" != "yes" ]; then
    echo "Rollback cancelled"
    exit 1
fi

for node in "${NODES[@]}"; do
    echo "=== Rolling back $node ==="

    # 1. Drain node
    ssh $node "nodetool drain"

    # 2. Stop ScyllaDB
    ssh $node "sudo systemctl stop scylla-server"

    # 3. Downgrade package
    ssh $node "sudo yum downgrade scylla-$PREVIOUS_VERSION -y"

    # 4. Start ScyllaDB
    ssh $node "sudo systemctl start scylla-server"

    # 5. Wait for UN status
    while true; do
        status=$(nodetool status | grep $node | awk '{print $1}')
        if [ "$status" == "UN" ]; then
            echo "$node is UP and NORMAL"
            break
        fi
        sleep 10
    done

    # 6. Verify version
    ssh $node "scylla --version"

    echo "Waiting before next node..."
    sleep 120
done

echo "=== Rollback complete ==="
```

**Important Notes:**
- Rollback only supported within same major version (5.2.5 → 5.2.1 ✅, 6.0 → 5.x ❌)
- SSTable format must be compatible
- Test rollback procedure in staging first
- Data written with new version features may be incompatible

### B. Schema Rollback

**Rollback Migration:**
```sql
-- Create rollback script for each migration
-- V002__add_user_status.cql
ALTER TABLE users ADD status TEXT;

-- V002_rollback.cql
ALTER TABLE users DROP status;

DELETE FROM schema_migrations WHERE version = 2;
```

**Automated Rollback Script:**
```python
#!/usr/bin/env python3
from cassandra.cluster import Cluster

def rollback_migration(session, target_version):
    """Rollback to specific schema version"""
    current_version = get_current_version(session)

    if target_version >= current_version:
        print(f"Cannot rollback: target version {target_version} >= current {current_version}")
        return

    # Get migrations to rollback (in reverse order)
    migrations = session.execute(
        "SELECT version, script FROM schema_migrations WHERE version > ? ALLOW FILTERING",
        (target_version,)
    )

    for migration in sorted(migrations, key=lambda m: m.version, reverse=True):
        rollback_file = migration.script.replace('.cql', '_rollback.cql')

        print(f"Rolling back migration {migration.version}")

        try:
            with open(rollback_file, 'r') as f:
                statements = f.read().split(';')
                for statement in statements:
                    statement = statement.strip()
                    if statement:
                        session.execute(statement)

            # Remove migration record
            session.execute(
                "DELETE FROM schema_migrations WHERE version = ?",
                (migration.version,)
            )

            print(f"Migration {migration.version} rolled back successfully")
        except FileNotFoundError:
            print(f"WARNING: Rollback file not found: {rollback_file}")
        except Exception as e:
            print(f"ERROR rolling back migration {migration.version}: {e}")
            raise

# Usage
cluster = Cluster(['localhost'])
session = cluster.connect('production')
rollback_migration(session, target_version=1)
```

### C. Data Restoration Rollback

If data corruption occurs, restore from backup:

```bash
# 1. Stop writes to cluster
# Update application to read-only mode

# 2. Restore from snapshot
sctool restore --cluster prod \
    --snapshot-tag backup_20260206_0300 \
    --location s3:my-bucket/backups

# 3. Verify data integrity
cqlsh -e "SELECT COUNT(*) FROM production.users;"

# 4. Resume normal operations
# Re-enable writes in application
```

### D. Rollback Decision Matrix

| Issue Type | Severity | Action | Rollback Required |
|------------|----------|--------|-------------------|
| Minor bug | Low | Apply hotfix | No |
| Performance degradation < 10% | Medium | Investigate, tune | Maybe |
| Performance degradation > 20% | High | Rollback immediately | Yes |
| Application errors | High | Fix app or rollback | Yes |
| Data corruption | Critical | Restore from backup | Yes |
| Node crashes | Critical | Rollback immediately | Yes |

---

## 15. Backup Strategies

### A. Backup Types

**1. Snapshot Backup (Point-in-Time):**
```bash
# Create snapshot on all nodes
nodetool snapshot -t backup_20260206 production

# Snapshot location
# /var/lib/scylla/data/production/users-*/snapshots/backup_20260206/

# Copy snapshots to remote storage
for node in node1 node2 node3; do
    ssh $node "
        cd /var/lib/scylla/data
        tar czf /tmp/backup_20260206_$node.tar.gz */*/snapshots/backup_20260206
        aws s3 cp /tmp/backup_20260206_$node.tar.gz s3://backups/scylla/
    "
done

# Clear snapshots (after verification)
nodetool clearsnapshot -t backup_20260206
```

**2. ScyllaDB Manager Backup (Recommended):**
```bash
# One-time backup
sctool backup --cluster prod \
    --location s3:my-bucket/backups \
    --snapshot-tag manual_backup_20260206

# Scheduled backup (daily at 2 AM)
sctool backup --cluster prod \
    --location s3:my-bucket/backups \
    --interval 24h \
    --start-date 2026-02-07T02:00:00Z \
    --retention 30  # Keep 30 days

# Incremental backup
sctool backup --cluster prod \
    --location s3:my-bucket/backups \
    --interval 6h \
    --retention 7
```

**3. Continuous Backup (Commitlog Archive):**
```yaml
# scylla.yaml configuration
commitlog_archiving:
    enabled: true
    archive_command: "/usr/local/bin/archive_commitlog.sh %path"
    restore_command: "/usr/local/bin/restore_commitlog.sh %from %to"
```

```bash
# archive_commitlog.sh
#!/bin/bash
COMMITLOG_FILE=$1
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
aws s3 cp $COMMITLOG_FILE s3://backups/commitlogs/$(basename $COMMITLOG_FILE)_$TIMESTAMP
```

### B. Backup Strategies by Use Case

**Production Environment:**
```bash
# Daily full snapshot
sctool backup --cluster prod \
    --location s3:prod-backups/daily \
    --interval 24h \
    --start-date 2026-02-07T03:00:00Z \
    --retention 30

# Hourly incremental
sctool backup --cluster prod \
    --location s3:prod-backups/hourly \
    --interval 1h \
    --retention 7
```

**Compliance/Long-term Retention:**
```bash
# Weekly archival backup
sctool backup --cluster prod \
    --location s3:archive-backups/weekly \
    --interval 168h \
    --retention 365  # 1 year

# Monthly archival
sctool backup --cluster prod \
    --location s3:archive-backups/monthly \
    --interval 720h \
    --retention 2555  # 7 years
```

### C. Multi-Datacenter Backup

```bash
# Backup from each datacenter separately
sctool backup --cluster prod \
    --location s3:backups-dc1/ \
    --dc DC1 \
    --interval 24h

sctool backup --cluster prod \
    --location s3:backups-dc2/ \
    --dc DC2 \
    --interval 24h

# Or backup entire cluster (all DCs)
sctool backup --cluster prod \
    --location s3:backups-all/ \
    --interval 24h
```

### D. Restore Procedures

**Full Cluster Restore:**
```bash
# List available backups
sctool backup list --cluster prod

# Restore from specific snapshot
sctool restore --cluster prod \
    --snapshot-tag backup_20260206_0300 \
    --location s3:my-bucket/backups

# Monitor restore progress
sctool task progress restore/task-id

# Verify data after restore
cqlsh -e "SELECT COUNT(*) FROM production.users;"
```

**Restore to New Cluster:**
```bash
# 1. Create new cluster with same schema
cqlsh new-cluster -f schema.cql

# 2. Restore data using sstableloader
for backup_file in $(aws s3 ls s3://backups/snapshot_20260206/ | awk '{print $4}'); do
    aws s3 cp s3://backups/snapshot_20260206/$backup_file /tmp/
    tar xzf /tmp/$backup_file -C /tmp/restore/
done

sstableloader -d new-cluster-node1,new-cluster-node2 \
    /tmp/restore/production/users/

# 3. Verify data
cqlsh new-cluster -e "SELECT COUNT(*) FROM production.users;"
```

**Point-in-Time Recovery:**
```bash
# 1. Restore base snapshot
sctool restore --cluster prod \
    --snapshot-tag backup_20260206_0000

# 2. Replay commitlogs up to specific timestamp
for commitlog in $(aws s3 ls s3://backups/commitlogs/ | awk '{print $4}'); do
    timestamp=$(echo $commitlog | grep -oP '\d{8}_\d{6}')
    if [ "$timestamp" -le "20260206_120000" ]; then
        aws s3 cp s3://backups/commitlogs/$commitlog /var/lib/scylla/commitlog/
    fi
done

# 3. Restart ScyllaDB (will replay commitlogs)
sudo systemctl restart scylla-server
```

### E. Table-Level Restore

```bash
# Restore specific table from snapshot
nodetool refresh production users

# Or using sstableloader
sstableloader -d localhost \
    /path/to/backup/production/users/
```

### F. Backup Verification

```bash
# Automated backup verification script
#!/bin/bash

BACKUP_TAG="backup_20260206_0300"

# 1. Create test cluster or use staging
# 2. Restore backup
sctool restore --cluster staging \
    --snapshot-tag $BACKUP_TAG \
    --location s3:backups/

# 3. Verify row counts match production
for table in users orders products; do
    prod_count=$(cqlsh prod-node -e "SELECT COUNT(*) FROM production.$table;" | grep -oP '\d+')
    staging_count=$(cqlsh staging-node -e "SELECT COUNT(*) FROM production.$table;" | grep -oP '\d+')

    if [ "$prod_count" -eq "$staging_count" ]; then
        echo "✓ $table: $prod_count rows (match)"
    else
        echo "✗ $table: prod=$prod_count staging=$staging_count (MISMATCH)"
        exit 1
    fi
done

# 4. Run sample queries
cqlsh staging-node -e "SELECT * FROM production.users LIMIT 10;"

echo "Backup verification complete"
```

### G. Backup Best Practices

**Backup Strategy:**
- [ ] Automated daily snapshots (minimum)
- [ ] Hourly incremental backups for critical data
- [ ] Off-site backup storage (different region/provider)
- [ ] Encryption at rest for backups
- [ ] Backup from each datacenter independently
- [ ] Tag backups with meaningful names (date, version, purpose)
- [ ] Document backup and restore procedures

**Retention Policy:**
- [ ] Daily backups: 30 days
- [ ] Weekly backups: 90 days
- [ ] Monthly backups: 1-7 years (compliance)
- [ ] Pre-upgrade backups: Until upgrade verified successful
- [ ] Automatic cleanup of old backups

**Verification:**
- [ ] Test restore monthly in staging environment
- [ ] Verify backup sizes are consistent
- [ ] Monitor backup success/failure alerts
- [ ] Validate backup integrity with checksums
- [ ] Document restore time (RTO)
- [ ] Measure data loss window (RPO)

**Backup Monitoring:**
```bash
# Monitor backup task status
sctool task list --cluster prod | grep backup

# Check last successful backup
sctool backup list --cluster prod --limit 1

# Alert on backup failure
if sctool task list --cluster prod | grep -q "backup.*ERROR"; then
    echo "ALERT: Backup failed" | mail -s "ScyllaDB Backup Failure" ops@example.com
fi
```

### H. Disaster Recovery

**DR Strategy:**
```
Primary DC (DC1)          Backup DC (DC2)
┌──────────────┐          ┌──────────────┐
│   Active     │          │   Standby    │
│   RF = 3     │────┬────→│   RF = 3     │
└──────────────┘    │     └──────────────┘
                    │
                    ↓
            S3 Backups (Cross-Region)
```

**Failover Procedure:**
```bash
# 1. Verify DC2 is healthy
nodetool status | grep DC2

# 2. Update application to point to DC2
# Update DNS or load balancer

# 3. Verify consistency level can be met
# If using EACH_QUORUM, change to LOCAL_QUORUM

# 4. Monitor replication lag
nodetool tpstats

# 5. If DC1 recoverable, rebuild
nodetool rebuild --datacenter DC1
```

**Recovery Time Objective (RTO):**
- Snapshot restore: 1-4 hours (depending on data size)
- Multi-DC failover: < 5 minutes
- Point-in-time recovery: 2-6 hours

**Recovery Point Objective (RPO):**
- Hourly backups: < 1 hour data loss
- Continuous commitlog archiving: < 5 minutes data loss
- Multi-DC replication: Near-zero (asynchronous lag)

---

## 16. Container Deployment (Docker & Kubernetes)

### A. Why Use Containers for ScyllaDB

**Benefits:**
- Consistent environment across dev/staging/production
- Easy horizontal scaling
- Resource isolation and limits
- Simplified deployment and upgrades
- Cloud-native integration (Kubernetes)

**Considerations:**
- Requires persistent storage configuration
- Network performance overhead (use host networking)
- Resource limits must be carefully tuned
- Not recommended for maximum performance (bare metal is better)
- Good for development, staging, and cloud deployments

### B. Docker Deployment

**Single Node (Development):**
```bash
# Pull official ScyllaDB image
docker pull scylladb/scylla:5.4

# Run single node
docker run --name scylla-node1 \
  --hostname scylla-node1 \
  -d \
  --restart unless-stopped \
  -p 9042:9042 \
  -p 9160:9160 \
  -p 7000:7000 \
  -p 7001:7001 \
  -p 10000:10000 \
  -v scylla-data:/var/lib/scylla \
  scylladb/scylla:5.4 \
  --smp 2 \
  --memory 4G \
  --overprovisioned 1

# Check node status
docker exec scylla-node1 nodetool status

# Connect with cqlsh
docker exec -it scylla-node1 cqlsh
```

**Multi-Node Cluster (Docker):**
```bash
# Create dedicated network
docker network create scylla-network

# Start seed node
docker run --name scylla-node1 \
  --hostname scylla-node1 \
  --network scylla-network \
  -d \
  -p 9042:9042 \
  -v scylla-data1:/var/lib/scylla \
  scylladb/scylla:5.4 \
  --seeds scylla-node1 \
  --smp 2 \
  --memory 4G \
  --overprovisioned 1

# Wait for node1 to start (30 seconds)
sleep 30

# Start node2
docker run --name scylla-node2 \
  --hostname scylla-node2 \
  --network scylla-network \
  -d \
  -v scylla-data2:/var/lib/scylla \
  scylladb/scylla:5.4 \
  --seeds scylla-node1 \
  --smp 2 \
  --memory 4G \
  --overprovisioned 1

# Start node3
docker run --name scylla-node3 \
  --hostname scylla-node3 \
  --network scylla-network \
  -d \
  -v scylla-data3:/var/lib/scylla \
  scylladb/scylla:5.4 \
  --seeds scylla-node1 \
  --smp 2 \
  --memory 4G \
  --overprovisioned 1

# Verify cluster
docker exec scylla-node1 nodetool status
```

### C. Docker Compose (Recommended for Development)

**docker-compose.yml:**
```yaml
version: '3.8'

services:
  scylla-node1:
    image: scylladb/scylla:5.4
    container_name: scylla-node1
    hostname: scylla-node1
    restart: unless-stopped
    ports:
      - "9042:9042"
      - "19042:19042"  # Shard-aware port
    volumes:
      - scylla-data1:/var/lib/scylla
    command: --seeds=scylla-node1 --smp 2 --memory 4G --overprovisioned 1
    networks:
      - scylla-net
    healthcheck:
      test: ["CMD", "cqlsh", "-e", "SELECT * FROM system.local"]
      interval: 30s
      timeout: 10s
      retries: 5

  scylla-node2:
    image: scylladb/scylla:5.4
    container_name: scylla-node2
    hostname: scylla-node2
    restart: unless-stopped
    volumes:
      - scylla-data2:/var/lib/scylla
    command: --seeds=scylla-node1 --smp 2 --memory 4G --overprovisioned 1
    networks:
      - scylla-net
    depends_on:
      scylla-node1:
        condition: service_healthy

  scylla-node3:
    image: scylladb/scylla:5.4
    container_name: scylla-node3
    hostname: scylla-node3
    restart: unless-stopped
    volumes:
      - scylla-data3:/var/lib/scylla
    command: --seeds=scylla-node1 --smp 2 --memory 4G --overprovisioned 1
    networks:
      - scylla-net
    depends_on:
      scylla-node1:
        condition: service_healthy

  # Optional: ScyllaDB Monitoring
  prometheus:
    image: prom/prometheus:latest
    container_name: scylla-prometheus
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus-data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
    ports:
      - "9090:9090"
    networks:
      - scylla-net

  grafana:
    image: grafana/grafana:latest
    container_name: scylla-grafana
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana-data:/var/lib/grafana
    ports:
      - "3000:3000"
    networks:
      - scylla-net

networks:
  scylla-net:
    driver: bridge

volumes:
  scylla-data1:
  scylla-data2:
  scylla-data3:
  prometheus-data:
  grafana-data:
```

**Usage:**
```bash
# Start cluster
docker-compose up -d

# Check cluster status
docker exec scylla-node1 nodetool status

# View logs
docker-compose logs -f scylla-node1

# Connect to cluster
docker exec -it scylla-node1 cqlsh

# Stop cluster
docker-compose down

# Stop and remove volumes (DESTRUCTIVE)
docker-compose down -v
```

### D. Kubernetes Deployment (Production)

**Recommended: Use ScyllaDB Operator**

The ScyllaDB Operator simplifies deployment, scaling, and management in Kubernetes.

**Install ScyllaDB Operator:**
```bash
# Add ScyllaDB Helm repository
helm repo add scylla https://scylla-operator-charts.storage.googleapis.com/stable
helm repo update

# Install cert-manager (prerequisite)
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.0/cert-manager.yaml

# Install ScyllaDB Operator
kubectl create namespace scylla-operator
helm install scylla-operator scylla/scylla-operator \
  --namespace scylla-operator \
  --set image.tag=latest

# Verify operator is running
kubectl get pods -n scylla-operator
```

**ScyllaDB Cluster Definition:**
```yaml
# scylla-cluster.yaml
apiVersion: scylla.scylladb.com/v1
kind: ScyllaCluster
metadata:
  name: scylla-cluster
  namespace: scylla
spec:
  version: 5.4.0
  agentVersion: 3.2.0
  datacenter:
    name: dc1
    racks:
      - name: rack1
        members: 3
        storage:
          capacity: 500Gi
          storageClassName: fast-ssd  # Use fast SSD storage class
        resources:
          requests:
            cpu: 8
            memory: 32Gi
          limits:
            cpu: 8
            memory: 32Gi
        placement:
          nodeAffinity:
            requiredDuringSchedulingIgnoredDuringExecution:
              nodeSelectorTerms:
                - matchExpressions:
                    - key: scylla.scylladb.com/node-type
                      operator: In
                      values:
                        - scylla
          podAntiAffinity:
            requiredDuringSchedulingIgnoredDuringExecution:
              - labelSelector:
                  matchExpressions:
                    - key: app.kubernetes.io/name
                      operator: In
                      values:
                        - scylla
                topologyKey: kubernetes.io/hostname
  # Alternator API (DynamoDB compatible) - optional
  alternator:
    port: 8000
    writeIsolation: always
  # Enable ScyllaDB Manager
  scyllaManager:
    namespace: scylla-manager
    name: scylla-manager
```

**Deploy Cluster:**
```bash
# Create namespace
kubectl create namespace scylla

# Apply cluster definition
kubectl apply -f scylla-cluster.yaml

# Watch cluster creation
kubectl get scyllaclusters -n scylla -w

# Check pods
kubectl get pods -n scylla

# Check cluster status
kubectl exec -it scylla-cluster-dc1-rack1-0 -n scylla -- nodetool status
```

**StatefulSet Configuration (Manual Deployment):**

If not using the operator, deploy with StatefulSet:

```yaml
# scylla-statefulset.yaml
apiVersion: v1
kind: Service
metadata:
  name: scylla
  namespace: scylla
  labels:
    app: scylla
spec:
  clusterIP: None
  selector:
    app: scylla
  ports:
    - port: 9042
      name: cql
    - port: 7000
      name: intra-node
    - port: 7001
      name: tls-intra-node
    - port: 9180
      name: prometheus
---
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: scylla
  namespace: scylla
spec:
  serviceName: scylla
  replicas: 3
  selector:
    matchLabels:
      app: scylla
  template:
    metadata:
      labels:
        app: scylla
    spec:
      affinity:
        podAntiAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            - labelSelector:
                matchExpressions:
                  - key: app
                    operator: In
                    values:
                      - scylla
              topologyKey: kubernetes.io/hostname
      containers:
        - name: scylla
          image: scylladb/scylla:5.4
          imagePullPolicy: IfNotPresent
          ports:
            - containerPort: 9042
              name: cql
            - containerPort: 7000
              name: intra-node
            - containerPort: 7001
              name: tls-intra-node
            - containerPort: 9180
              name: prometheus
          env:
            - name: POD_IP
              valueFrom:
                fieldRef:
                  fieldPath: status.podIP
          resources:
            requests:
              cpu: 8
              memory: 32Gi
            limits:
              cpu: 8
              memory: 32Gi
          command:
            - /docker-entrypoint.py
            - --seeds=scylla-0.scylla.scylla.svc.cluster.local
            - --smp=8
            - --memory=30G
            - --overprovisioned=0
          volumeMounts:
            - name: scylla-data
              mountPath: /var/lib/scylla
          livenessProbe:
            exec:
              command:
                - /bin/bash
                - -c
                - nodetool status | grep -E "^UN\s+${POD_IP}"
            initialDelaySeconds: 90
            periodSeconds: 10
          readinessProbe:
            exec:
              command:
                - /bin/bash
                - -c
                - nodetool status | grep -E "^UN\s+${POD_IP}"
            initialDelaySeconds: 90
            periodSeconds: 10
  volumeClaimTemplates:
    - metadata:
        name: scylla-data
      spec:
        accessModes: ["ReadWriteOnce"]
        storageClassName: fast-ssd
        resources:
          requests:
            storage: 500Gi
```

### E. Container Best Practices

**Resource Configuration:**
```bash
# For production, allocate resources properly
--smp 8              # Number of CPU cores (match container CPU limit)
--memory 30G         # Leave 2GB for overhead if container has 32GB
--overprovisioned 0  # Set to 0 for production, 1 for dev/shared environments

# For development
--smp 2
--memory 4G
--overprovisioned 1  # Allows running on shared/overcommitted systems
```

**Storage Best Practices:**
- Use local SSDs or fast persistent volumes
- StorageClass with `volumeBindingMode: WaitForFirstConsumer`
- Minimum 100 IOPS per GB
- Prefer local-path provisioner or local PVs for best performance

**Networking:**
```yaml
# Use host networking for maximum performance (if possible)
hostNetwork: true
dnsPolicy: ClusterFirstWithHostNet

# Or use dedicated network plugin
# Calico, Cilium with eBPF for low latency
```

**Resource Limits:**
```yaml
resources:
  requests:
    cpu: "8"           # Guaranteed CPU
    memory: "32Gi"     # Guaranteed memory
  limits:
    cpu: "8"           # Maximum CPU (same as request)
    memory: "32Gi"     # Maximum memory (same as request)
```

**Environment Variables:**
```yaml
env:
  - name: SCYLLA_CLUSTER_NAME
    value: "production-cluster"
  - name: SCYLLA_ENDPOINT_SNITCH
    value: "GossipingPropertyFileSnitch"
  - name: SCYLLA_DC
    value: "dc1"
  - name: SCYLLA_RACK
    value: "rack1"
```

### F. Monitoring Containers

**Prometheus Configuration:**
```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'scylla'
    static_configs:
      - targets:
          - 'scylla-node1:9180'
          - 'scylla-node2:9180'
          - 'scylla-node3:9180'
    relabel_configs:
      - source_labels: [__address__]
        regex: '([^:]+)(?::\d+)?'
        target_label: instance
```

**In Kubernetes:**
```yaml
# ServiceMonitor for Prometheus Operator
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: scylla
  namespace: scylla
spec:
  selector:
    matchLabels:
      app: scylla
  endpoints:
    - port: prometheus
      interval: 30s
```

### G. Backup in Containers

**Using ScyllaDB Manager in Kubernetes:**
```bash
# Install ScyllaDB Manager
helm install scylla-manager scylla/scylla-manager \
  --namespace scylla-manager \
  --create-namespace

# Configure backup location
kubectl apply -f - <<EOF
apiVersion: scylla.scylladb.com/v1alpha1
kind: ScyllaBackup
metadata:
  name: daily-backup
  namespace: scylla
spec:
  cluster: scylla-cluster
  dc: dc1
  location: s3:my-bucket/scylla-backups
  interval: 24h
  retention: 30
EOF
```

**Docker Volume Backup:**
```bash
# Backup Docker volume
docker run --rm \
  -v scylla-data1:/data \
  -v /backup:/backup \
  alpine tar czf /backup/scylla-data1-$(date +%Y%m%d).tar.gz /data

# Restore Docker volume
docker run --rm \
  -v scylla-data1:/data \
  -v /backup:/backup \
  alpine tar xzf /backup/scylla-data1-20260206.tar.gz -C /
```

### H. Container Deployment Checklist

**Development Environment:**
- [ ] Use Docker Compose for local development
- [ ] Use `--overprovisioned 1` flag
- [ ] Allocate 2-4 cores per node
- [ ] Allocate 4-8 GB RAM per node
- [ ] Use Docker volumes for persistence
- [ ] Enable health checks

**Production Kubernetes:**
- [ ] Use ScyllaDB Operator (recommended)
- [ ] Deploy as StatefulSet with persistent volumes
- [ ] Use fast SSD storage class (local-path or cloud SSD)
- [ ] Configure pod anti-affinity for fault tolerance
- [ ] Set CPU/memory requests equal to limits
- [ ] Use `--overprovisioned 0` for production
- [ ] Allocate 8+ cores per pod
- [ ] Allocate 32+ GB RAM per pod
- [ ] Configure liveness and readiness probes
- [ ] Set up monitoring with Prometheus
- [ ] Configure ScyllaDB Manager for backups
- [ ] Use dedicated node pools for ScyllaDB pods
- [ ] Configure resource quotas per namespace
- [ ] Test disaster recovery procedures

**Networking:**
- [ ] Use host networking if possible (best performance)
- [ ] Or use high-performance CNI (Cilium, Calico)
- [ ] Expose CQL port (9042) via Service
- [ ] Expose shard-aware port (19042) for drivers
- [ ] Configure NetworkPolicy for security

**Storage:**
- [ ] Use local SSDs or premium cloud disks
- [ ] Configure storage class with appropriate IOPS
- [ ] Size volumes with 50% headroom for compaction
- [ ] Enable volume snapshots for backups
- [ ] Test volume restoration procedures

### I. Container Migration Example

**Migrating from VMs to Kubernetes:**

```bash
# Phase 1: Backup existing cluster
nodetool snapshot -t migration_backup

# Phase 2: Deploy Kubernetes cluster
kubectl apply -f scylla-cluster.yaml

# Phase 3: Restore data using sstableloader
for table in users orders products; do
    sstableloader -d scylla-0.scylla.scylla.svc.cluster.local \
        /path/to/backup/keyspace/$table/
done

# Phase 4: Verify data
kubectl exec -it scylla-0 -n scylla -- cqlsh -e "SELECT COUNT(*) FROM keyspace.users;"

# Phase 5: Switch application to Kubernetes service
# Update connection string to: scylla.scylla.svc.cluster.local:9042
```

### J. Troubleshooting Containers

**Common Issues:**

**1. Pod Crashes on Startup:**
```bash
# Check logs
kubectl logs scylla-0 -n scylla

# Common cause: Insufficient memory
# Solution: Increase memory limits and adjust --memory flag

# Check resource availability
kubectl describe pod scylla-0 -n scylla
```

**2. Slow Performance:**
```bash
# Check if overprovisioned flag is set correctly
kubectl exec -it scylla-0 -n scylla -- scylla --help | grep overprovisioned

# For production, should be --overprovisioned 0
# Update StatefulSet to remove --overprovisioned 1

# Check storage performance
kubectl exec -it scylla-0 -n scylla -- fio --name=test --size=1G --rw=randread --bs=4k
```

**3. Nodes Not Joining Cluster:**
```bash
# Check network connectivity
kubectl exec -it scylla-0 -n scylla -- nodetool status

# Verify seeds are correct
kubectl exec -it scylla-1 -n scylla -- grep seeds /etc/scylla/scylla.yaml

# Check logs for errors
kubectl logs scylla-1 -n scylla | grep -i error
```

---

## 17. Deployment Checklist

### Pre-Production Requirements

#### Deployment Method Selection
- [ ] Deployment method chosen (bare metal/VM/container)
- [ ] For containers: Kubernetes or Docker Compose
- [ ] For Kubernetes: ScyllaDB Operator installed
- [ ] Resource requirements validated for chosen method

#### Hardware Specifications (Bare Metal / VM)
- [ ] CPU: 20-60 logical cores (SSE4.2 instruction set required)
- [ ] Memory: 64-256 GB RAM for medium-high workload
- [ ] Storage: SSDs (NVMe preferred), 1TB per 4 cores
- [ ] Network: 10 Gbps+ network interface
- [ ] Disk: 10TB maximum per node

#### Container Specifications (Docker / Kubernetes)
- [ ] Container runtime: Docker 20.10+ or containerd
- [ ] Kubernetes version: 1.24+ (if using K8s)
- [ ] ScyllaDB Operator installed (recommended for K8s)
- [ ] CPU: 8+ cores per pod/container
- [ ] Memory: 32+ GB per pod/container
- [ ] Storage: Local SSD or premium cloud disks (500GB+)
- [ ] Storage class configured with high IOPS
- [ ] Pod anti-affinity rules configured
- [ ] Resource requests = limits (guaranteed QoS)
- [ ] `--overprovisioned 0` for production
- [ ] Health checks configured (liveness/readiness)
- [ ] Persistent volumes configured
- [ ] Monitoring stack deployed

#### Recommended Cloud Instances (Non-Container)
- [ ] AWS: i3, i3en, i4i instances (local NVMe SSDs)
- [ ] GCP: n1-highmem, n2-highmem with local SSDs
- [ ] Azure: Lsv2 series (NVMe local storage)
- [ ] Avoid: Over-committed/shared CPU instances

#### Data Modeling
- [ ] Schema designed around query patterns
- [ ] Partition keys provide even distribution
- [ ] Partition keys have high cardinality
- [ ] Partitions bounded (< 100 MB)
- [ ] No unbounded collections
- [ ] Appropriate compaction strategy selected
- [ ] TTL configured for time-series data
- [ ] Denormalization strategy defined

#### Replication
- [ ] NetworkTopologyStrategy configured
- [ ] Replication factor >= 3 per datacenter
- [ ] Consistency level documented (LOCAL_QUORUM recommended)
- [ ] Multi-DC topology configured (if applicable)
- [ ] Snitch configured (GossipingPropertyFileSnitch)
- [ ] Racks defined for failure isolation

#### Performance
- [ ] Shard-aware drivers configured
- [ ] Prepared statements used for all queries
- [ ] Token-aware load balancing enabled
- [ ] Connection pooling configured
- [ ] No ALLOW FILTERING in production queries
- [ ] Secondary indexes or MVs for non-PK queries
- [ ] Batch operations limited to same partition
- [ ] Compression enabled (LZ4 recommended)

#### Security (MANDATORY)
- [ ] Authentication enabled (NOT AllowAllAuthenticator)
- [ ] Custom roles created with least privilege
- [ ] Default cassandra superuser disabled/password changed
- [ ] TLS encryption enabled (client-to-node)
- [ ] TLS encryption enabled (node-to-node)
- [ ] Weak TLS versions disabled (1.0, 1.1)
- [ ] Encryption at rest configured (Enterprise)
- [ ] Network access restricted (firewall/VPC)
- [ ] JMX access restricted to monitoring hosts only
- [ ] Audit logging enabled (Enterprise)
- [ ] Secrets management configured

#### Monitoring
- [ ] ScyllaDB Monitoring Stack deployed
- [ ] Grafana dashboards configured
- [ ] Alerts configured (p99 latency, disk usage, node down)
- [ ] ScyllaDB Manager installed and configured
- [ ] Monitoring resources: 15GB+ RAM, 2-4 vCPUs
- [ ] Log aggregation configured
- [ ] Backup monitoring configured

#### Operations
- [ ] ScyllaDB Manager configured for repairs
- [ ] Weekly repair scheduled
- [ ] Daily backup scheduled
- [ ] Backup retention policy defined (30 days minimum)
- [ ] Restore procedures tested
- [ ] Rolling restart procedure documented
- [ ] Node addition procedure documented
- [ ] Disaster recovery runbook created

#### Migration and Upgrades
- [ ] Migration strategy documented (if migrating from Cassandra)
- [ ] Schema migration process established
- [ ] Schema versioning implemented
- [ ] Upgrade procedure documented and tested
- [ ] Rollback procedure documented and tested
- [ ] Pre-upgrade backup process defined
- [ ] SSTable upgrade strategy documented

#### Backup and Recovery
- [ ] Automated daily snapshots configured
- [ ] Incremental backup schedule defined
- [ ] Off-site backup storage configured
- [ ] Backup encryption enabled
- [ ] Backup verification process automated
- [ ] Monthly restore tests scheduled
- [ ] RTO (Recovery Time Objective) defined
- [ ] RPO (Recovery Point Objective) defined
- [ ] Disaster recovery plan documented
- [ ] Multi-DC failover procedure tested

### Capacity Planning

#### Cluster Sizing
- [ ] Data volume estimated (2-3 year projection)
- [ ] Query throughput estimated (peak and average)
- [ ] Replication factor accounted for storage
- [ ] Compaction overhead accounted (50-100% extra)
- [ ] Growth headroom: 50% spare capacity

#### Scaling Triggers
- [ ] Disk usage > 70%: Add nodes
- [ ] CPU utilization > 80%: Add nodes or upgrade instances
- [ ] p99 latency degradation: Investigate and optimize
- [ ] Compaction backlog growing: Tune compaction or add capacity

### Ongoing Operations

#### Regular Maintenance
- [ ] Weekly repairs via ScyllaDB Manager
- [ ] Daily backups
- [ ] Monthly backup restore tests
- [ ] Quarterly security permission review
- [ ] Monitor and drop unused indexes/MVs
- [ ] Review and optimize slow queries
- [ ] Update ScyllaDB version (stay within 2 releases of latest)

#### Performance Tuning
- [ ] Monitor p99 latency (not average)
- [ ] Analyze slow query logs
- [ ] Review compaction statistics
- [ ] Check partition size distribution
- [ ] Monitor cache hit rates
- [ ] Review connection pool saturation
- [ ] Optimize data model based on access patterns

---

## 18. Quick Reference

### CQL Commands

```sql
-- Keyspace operations
CREATE KEYSPACE prod WITH REPLICATION = {
    'class': 'NetworkTopologyStrategy', 'DC1': 3
};
USE prod;
DROP KEYSPACE test;
ALTER KEYSPACE prod WITH REPLICATION = {'class': 'NetworkTopologyStrategy', 'DC1': 5};

-- Table operations
CREATE TABLE users (user_id UUID PRIMARY KEY, name TEXT, email TEXT);
DROP TABLE users;
ALTER TABLE users ADD phone TEXT;
TRUNCATE users;

-- Index operations
CREATE INDEX ON users (email);
CREATE INDEX IF NOT EXISTS user_email_idx ON users (email);
DROP INDEX user_email_idx;

-- Materialized view
CREATE MATERIALIZED VIEW users_by_email AS
    SELECT * FROM users WHERE email IS NOT NULL AND user_id IS NOT NULL
    PRIMARY KEY (email, user_id);
DROP MATERIALIZED VIEW users_by_email;

-- Data operations
INSERT INTO users (user_id, name, email) VALUES (uuid(), 'Alice', 'alice@example.com');
UPDATE users SET name = 'Alice Smith' WHERE user_id = ?;
DELETE FROM users WHERE user_id = ?;
SELECT * FROM users WHERE user_id = ?;

-- Batch operations
BEGIN BATCH
    INSERT INTO users (user_id, name) VALUES (uuid(), 'Bob');
    UPDATE users SET email = 'new@example.com' WHERE user_id = ?;
APPLY BATCH;

-- Lightweight transactions
INSERT INTO users (user_id, email) VALUES (?, ?) IF NOT EXISTS;
UPDATE users SET status = 'active' WHERE user_id = ? IF status = 'pending';

-- TTL
INSERT INTO sessions (session_id, data) VALUES (?, ?) USING TTL 3600;
UPDATE users USING TTL 86400 SET temp_data = ? WHERE user_id = ?;

-- Role management
CREATE ROLE app_user WITH PASSWORD = 'password' AND LOGIN = true;
GRANT SELECT ON KEYSPACE prod TO app_user;
REVOKE MODIFY ON KEYSPACE prod FROM app_user;
LIST ROLES;
LIST PERMISSIONS ON KEYSPACE prod;
```

### nodetool Commands

```bash
# Cluster information
nodetool status                    # Cluster status
nodetool info                      # Node information
nodetool describecluster           # Cluster details
nodetool ring                      # Token ring

# Statistics
nodetool tablestats keyspace.table # Table statistics
nodetool tablehistograms keyspace.table  # Latency histograms
nodetool cfstats                   # Column family stats
nodetool compactionstats           # Compaction status

# Operations
nodetool flush                     # Flush memtables
nodetool cleanup                   # Clean up after adding nodes
nodetool decommission              # Remove node from cluster
nodetool drain                     # Drain before restart
nodetool rebuild                   # Rebuild from another DC

# Maintenance
nodetool snapshot -t name keyspace # Create snapshot
nodetool clearsnapshot             # Clear snapshots
nodetool scrub keyspace.table      # Rebuild SSTables
nodetool upgradesstables           # Rewrite SSTables

# Monitoring
nodetool tpstats                   # Thread pool statistics
nodetool netstats                  # Network statistics
nodetool proxyhistograms           # Latency histograms
nodetool gcstats                   # GC statistics (N/A for ScyllaDB)
```

### ScyllaDB Manager Commands

```bash
# Cluster management
sctool cluster add --host 10.0.0.1 --name prod
sctool cluster list
sctool status --cluster prod

# Repairs
sctool repair --cluster prod --interval 7d
sctool task progress repair/task-id
sctool task list --cluster prod

# Backups
sctool backup --cluster prod --location s3:bucket/path --interval 1d
sctool restore --cluster prod --snapshot snapshot-name
sctool backup list --cluster prod

# Health checks
sctool healthcheck --cluster prod
```

### Driver Configuration Examples

**Python:**
```python
from cassandra.cluster import Cluster
from cassandra.policies import TokenAwarePolicy, DCAwareRoundRobinPolicy
from cassandra import ConsistencyLevel

cluster = Cluster(
    ['10.0.0.1', '10.0.0.2'],
    load_balancing_policy=TokenAwarePolicy(DCAwareRoundRobinPolicy('DC1')),
    protocol_version=4
)
session = cluster.connect('keyspace')
prepared = session.prepare("SELECT * FROM users WHERE user_id = ?")
prepared.consistency_level = ConsistencyLevel.LOCAL_QUORUM
result = session.execute(prepared, (user_id,))
```

**Java:**
```java
CqlSession session = CqlSession.builder()
    .addContactPoint(new InetSocketAddress("10.0.0.1", 9042))
    .withLocalDatacenter("DC1")
    .withKeyspace("keyspace")
    .build();

PreparedStatement prepared = session.prepare(
    "SELECT * FROM users WHERE user_id = ?"
);
ResultSet result = session.execute(
    prepared.bind(userId).setConsistencyLevel(DefaultConsistencyLevel.LOCAL_QUORUM)
);
```

**Node.js:**
```javascript
const cassandra = require('cassandra-driver');

const client = new cassandra.Client({
    contactPoints: ['10.0.0.1', '10.0.0.2'],
    localDataCenter: 'DC1',
    keyspace: 'keyspace',
    policies: {
        loadBalancing: new cassandra.policies.loadBalancing.TokenAwarePolicy(
            new cassandra.policies.loadBalancing.DCAwareRoundRobinPolicy('DC1')
        )
    }
});

const query = 'SELECT * FROM users WHERE user_id = ?';
const result = await client.execute(query, [userId], { prepare: true });
```

**Go:**
```go
import (
    "github.com/gocql/gocql"
)

cluster := gocql.NewCluster("10.0.0.1", "10.0.0.2")
cluster.Keyspace = "keyspace"
cluster.Consistency = gocql.LocalQuorum
cluster.PoolConfig.HostSelectionPolicy = gocql.TokenAwareHostPolicy(
    gocql.DCAwareRoundRobinPolicy("DC1"),
)
session, _ := cluster.CreateSession()
defer session.Close()

var name string
if err := session.Query(`SELECT name FROM users WHERE user_id = ?`, userID).Scan(&name); err != nil {
    log.Fatal(err)
}
```

---

## 19. ScyllaDB vs Cassandra Quick Comparison

| Feature | ScyllaDB | Cassandra |
|---------|----------|-----------|
| **Language** | C++ | Java |
| **Architecture** | Shard-per-core | Thread-per-core |
| **GC Pauses** | None | Yes (can be seconds) |
| **Throughput** | 2-5x higher | Baseline |
| **Latency (p99)** | 75% lower | Baseline |
| **Resource Efficiency** | Higher | Lower |
| **CQL Compatibility** | Yes (100%) | Native |
| **Drivers** | Cassandra drivers work | Native |
| **Operational Complexity** | Lower (auto-tuning) | Higher (manual tuning) |
| **Cost (TCO)** | 75% savings | Baseline |
| **Lightweight Transactions** | More efficient | Less efficient |
| **Monitoring** | ScyllaDB Monitoring Stack | Separate solutions |
| **Management** | ScyllaDB Manager | Separate tools |

---

**Last Updated:** 2026-02-06
**Version:** 1.2
**Maintainer:** Database Team

**Changelog:**
- v1.2 (2026-02-06): Added comprehensive container deployment section covering Docker (single node, multi-node, Docker Compose), Kubernetes (ScyllaDB Operator, StatefulSets), container best practices, resource configuration, networking, monitoring, backup strategies, migration examples, and troubleshooting.
- v1.1 (2026-02-06): Added comprehensive sections on migration strategies (Cassandra to ScyllaDB, schema migrations, data migrations), upgrade strategies (rolling upgrades, SSTable upgrades), rollback strategies (version rollback, schema rollback, data restoration), and backup strategies (snapshot backups, ScyllaDB Manager backups, continuous backups, disaster recovery, point-in-time recovery).
- v1.0 (2026-02-06): Initial release with comprehensive coverage of ScyllaDB 5.x/6.x features including shard-per-core architecture, data modeling, replication strategies, consistency levels, performance optimization, compaction strategies, secondary indexes, materialized views, lightweight transactions, security, monitoring, and deployment best practices.
