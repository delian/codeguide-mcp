# MySQL & MariaDB Development Guidelines
Mandatory coding standards and development practices for MySQL and MariaDB development. MySQL 8.4+/9.0+, MariaDB 11.0+/11.4+, InnoDB, replication, ProxySQL/HAProxy.

---

**Agent Profile**: The MySQL/MariaDB Expert
**Role**: Senior Database Engineer & OLTP Specialist
**Objective**: Generate production-ready, reliable and performant relational database solutions.
**Tools**: MySQL 8.4+/9.0+, MariaDB 11.0+/11.4+, InnoDB, replication, ProxySQL/HAProxy

---

**Version:** 1.0 | **Last Updated:** February 2026 | **Target Versions:** MySQL 8.4+, 9.0+ | MariaDB 11.0+, 11.4+

## Table of Contents

1. [Core Philosophies: OLTP-FIRST](#1-core-philosophies-oltp-first)
2. [Architecture and Fundamentals](#2-architecture-and-fundamentals)
3. [Storage Engines (InnoDB vs MyISAM)](#3-storage-engines-innodb-vs-myisam)
4. [Performance Optimization](#4-performance-optimization)
5. [Replication Strategies](#5-replication-strategies)
6. [Sharding and Partitioning](#6-sharding-and-partitioning)
7. [Indexing Strategies](#7-indexing-strategies)
8. [Query Optimization](#8-query-optimization)
9. [Transaction Management](#9-transaction-management)
10. [Connection Pooling](#10-connection-pooling)
11. [High Availability](#11-high-availability)
12. [Security Best Practices](#12-security-best-practices)
13. [Backup and Recovery](#13-backup-and-recovery)
14. [Migration Strategies](#14-migration-strategies)
15. [Schema Design](#15-schema-design)
16. [Monitoring and Troubleshooting](#16-monitoring-and-troubleshooting)
17. [Container Deployment](#17-container-deployment)
18. [MySQL vs MariaDB Differences](#18-mysql-vs-mariadb-differences)
19. [Version-Specific Features](#19-version-specific-features)
20. [Performance Tuning Checklist](#20-performance-tuning-checklist)
21. [Production Deployment Patterns](#21-production-deployment-patterns)

---

## 1. Core Philosophies: OLTP-FIRST

The agent must adhere to the **OLTP-FIRST** principles for every MySQL/MariaDB implementation:

**Test-Driven Development (TDD)**: ALWAYS write tests BEFORE implementation (Red-Green-Refactor cycle mandatory).
**Regression Shield**: EVERY bug discovered MUST receive a test BEFORE fixing to prevent regression.

- **O**ptimize for transactions: Use InnoDB; design for ACID and concurrency.
- **L**ock wisely: Prefer row-level locking; avoid long transactions and deadlock-prone patterns.
- **T**est with real data: Use representative datasets and replication in tests.
- **P**repared statements: Always use parameterized queries; never concatenate user input.
- **F**oreign keys and constraints: Enforce referential integrity; use utf8mb4.

**Verified Code**: Agent-generated code MUST use prepared statements, run migrations safely, and pass tests before delivery.

---

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

### Example TDD Workflow for MySQL/MariaDB

```python
# Step 1: RED - Write failing test for a stored procedure
import pytest
import mysql.connector

@pytest.fixture
def db():
    conn = mysql.connector.connect(
        host='localhost', database='test_db',
        user='test_user', password='test_pass'
    )
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS orders ("
                   "  id INT AUTO_INCREMENT PRIMARY KEY,"
                   "  user_id INT NOT NULL,"
                   "  total_amount DECIMAL(12,2) NOT NULL,"
                   "  status ENUM('pending','confirmed','shipped') DEFAULT 'pending',"
                   "  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
                   ") ENGINE=InnoDB")
    conn.commit()
    yield conn
    cursor.execute("DROP TABLE IF EXISTS orders")
    conn.commit()
    conn.close()

def test_get_user_order_summary(db):
    """Test stored procedure returns correct order summary per user."""
    cursor = db.cursor()
    cursor.execute("INSERT INTO orders (user_id, total_amount, status) VALUES "
                   "(1, 50.00, 'confirmed'), (1, 75.00, 'shipped'), "
                   "(2, 120.00, 'pending')")
    db.commit()

    cursor.callproc('get_user_order_summary', [1])
    for result in cursor.stored_results():
        row = result.fetchone()
    assert row[0] == 2       # order_count
    assert row[1] == 125.00  # total_spent

# Run: pytest test_orders.py::test_get_user_order_summary
# FAILS - Procedure test_db.get_user_order_summary does not exist

# Step 2: GREEN - Create the stored procedure
def apply_migration(db):
    cursor = db.cursor()
    cursor.execute("""
        CREATE PROCEDURE get_user_order_summary(IN p_user_id INT)
        BEGIN
            SELECT
                COUNT(*) AS order_count,
                COALESCE(SUM(total_amount), 0) AS total_spent
            FROM orders
            WHERE user_id = p_user_id
              AND status IN ('confirmed', 'shipped');
        END
    """)
    db.commit()

# Run: pytest test_orders.py::test_get_user_order_summary
# PASSES

# Step 3: REFACTOR - Add composite index for the query
def optimize_migration(db):
    cursor = db.cursor()
    cursor.execute("""
        CREATE INDEX ix_orders_user_status
        ON orders(user_id, status, total_amount)
    """)
    db.commit()
# Tests still pass
```

### Example TDD for Constraints and Triggers

```python
def test_order_amount_must_be_positive(db):
    """Test CHECK constraint rejects negative order amounts."""
    cursor = db.cursor()
    cursor.execute("""
        ALTER TABLE orders
        ADD CONSTRAINT chk_positive_amount CHECK (total_amount >= 0)
    """)
    db.commit()

    with pytest.raises(mysql.connector.errors.DatabaseError):
        cursor.execute("INSERT INTO orders (user_id, total_amount) VALUES (1, -10.00)")
        db.commit()
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
# Bug Report BUG-782: get_user_order_summary counts cancelled orders in
# the total, inflating user spend figures.

import pytest
import mysql.connector

def test_cancelled_orders_excluded_from_summary(db):
    """Regression test for BUG-782: cancelled orders must not count in summary."""
    cursor = db.cursor()
    cursor.execute("INSERT INTO orders (user_id, total_amount, status) VALUES "
                   "(1, 50.00, 'confirmed'), (1, 200.00, 'cancelled')")
    db.commit()

    cursor.callproc('get_user_order_summary', [1])
    for result in cursor.stored_results():
        row = result.fetchone()

    assert row[0] == 1       # order_count - only confirmed
    assert row[1] == 50.00   # total_spent - excludes cancelled

# Run: pytest test_orders.py::test_cancelled_orders_excluded_from_summary
# FAILS - cancelled orders are included (returns count=2, total=250.00)

# Fix: Update stored procedure to explicitly exclude cancelled status
def fix_procedure(db):
    cursor = db.cursor()
    cursor.execute("DROP PROCEDURE IF EXISTS get_user_order_summary")
    cursor.execute("""
        CREATE PROCEDURE get_user_order_summary(IN p_user_id INT)
        BEGIN
            SELECT
                COUNT(*) AS order_count,
                COALESCE(SUM(total_amount), 0) AS total_spent
            FROM orders
            WHERE user_id = p_user_id
              AND status IN ('confirmed', 'shipped')
              AND status != 'cancelled';
        END
    """)
    db.commit()

# Run: pytest test_orders.py::test_cancelled_orders_excluded_from_summary
# PASSES - bug fixed, regression prevented
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

## 2. Architecture and Fundamentals

### MySQL Architecture Overview

MySQL/MariaDB follows a **layered architecture** with client-server model:

```
┌─────────────────────────────────────────────┐
│         Connection Pool / Thread Handler    │
├─────────────────────────────────────────────┤
│         SQL Interface & Parser              │
├─────────────────────────────────────────────┤
│         Query Optimizer                     │
├─────────────────────────────────────────────┤
│         Query Cache (removed MySQL 8.0+)    │
├─────────────────────────────────────────────┤
│         Pluggable Storage Engines           │
│  ┌──────────┬──────────┬─────────────────┐ │
│  │  InnoDB  │ MyISAM   │  ColumnStore    │ │
│  │ (default)│          │  (MariaDB)      │ │
│  └──────────┴──────────┴─────────────────┘ │
├─────────────────────────────────────────────┤
│         File System Layer                   │
└─────────────────────────────────────────────┘
```

### Key Components

**InnoDB Storage Engine (Default):**
- ACID-compliant transactions
- Row-level locking
- Foreign key support
- Crash recovery with redo/undo logs
- MVCC (Multi-Version Concurrency Control)

**Connection Handling:**
- Thread-per-connection model (traditional)
- Thread pool plugin (MariaDB default, MySQL Enterprise)
- Max connections: 151 (default), configurable up to ~100,000

**Memory Architecture:**
```sql
-- Key memory components
innodb_buffer_pool_size    -- Most critical: 70-80% of RAM
key_buffer_size            -- For MyISAM indexes
query_cache_size           -- Removed in MySQL 8.0
tmp_table_size             -- In-memory temporary tables
max_heap_table_size        -- MEMORY tables
sort_buffer_size           -- Per-connection sorting
read_buffer_size           -- Sequential scans
join_buffer_size           -- Table joins
```

### When to Use MySQL/MariaDB

**✅ Excellent For:**
- Web applications (WordPress, Drupal, etc.)
- E-commerce platforms (Magento, WooCommerce)
- High read/write concurrency
- OLTP (Online Transaction Processing)
- Multi-master replication needs
- Horizontal scaling with sharding
- SaaS applications
- Content management systems

**❌ Consider Alternatives For:**
- Heavy analytical queries (use ClickHouse, Redshift)
- Time-series data (use InfluxDB, TimescaleDB)
- Document-oriented data (use MongoDB, PostgreSQL JSONB)
- Graph relationships (use Neo4j)
- Full-text search (use Elasticsearch, Meilisearch)

---

## 3. Storage Engines (InnoDB vs MyISAM)

### InnoDB (Recommended - Default since MySQL 5.5)

**Advantages:**
- ✅ **ACID transactions** (atomicity, consistency, isolation, durability)
- ✅ **Row-level locking** (high concurrency)
- ✅ **Foreign keys** (referential integrity)
- ✅ **Crash recovery** (automatic)
- ✅ **MVCC** (readers don't block writers)
- ✅ **Online DDL** (non-blocking schema changes in MySQL 8.0+)

**Best For:**
- Transactional applications
- High concurrency workloads
- Data integrity requirements
- 99.9% of use cases

**Configuration:**
```sql
-- Check current engine
SHOW TABLE STATUS WHERE Name = 'users';

-- Set default engine
SET default_storage_engine = InnoDB;

-- Create table with InnoDB
CREATE TABLE users (
    id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    email VARCHAR(255) NOT NULL UNIQUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB
  DEFAULT CHARSET=utf8mb4
  COLLATE=utf8mb4_unicode_ci
  ROW_FORMAT=DYNAMIC;
```

### MyISAM (Legacy)

**Advantages:**
- Fast for read-heavy workloads
- Smaller disk footprint
- Full-text search (before MySQL 5.6)

**Disadvantages:**
- ❌ **No transactions**
- ❌ **Table-level locking** (poor concurrency)
- ❌ **No foreign keys**
- ❌ **No crash recovery**
- ❌ **Not recommended for modern applications**

**Migration from MyISAM to InnoDB:**
```sql
-- Convert single table
ALTER TABLE table_name ENGINE=InnoDB;

-- Convert all tables in database
SELECT CONCAT('ALTER TABLE ', table_name, ' ENGINE=InnoDB;')
FROM information_schema.tables
WHERE table_schema = 'your_database'
  AND engine = 'MyISAM';
```

### MariaDB-Specific Engines

**ColumnStore (Analytics):**
```sql
-- Create columnar table for analytics
CREATE TABLE analytics_events (
    event_time DATETIME,
    user_id BIGINT,
    event_type VARCHAR(50),
    value DECIMAL(10,2)
) ENGINE=ColumnStore;

-- Optimized for aggregation queries
SELECT
    DATE(event_time) as day,
    event_type,
    SUM(value) as total
FROM analytics_events
GROUP BY day, event_type;
```

**Aria (Enhanced MyISAM):**
- Crash-safe MyISAM replacement
- Better caching
- Still table-level locking

---

## 4. Performance Optimization

### InnoDB Configuration (Critical)

**Buffer Pool (Most Important Setting):**
```ini
# my.cnf / my.ini

# Set to 70-80% of available RAM (dedicated database server)
innodb_buffer_pool_size = 8G

# Multiple instances for better concurrency (1GB per instance)
innodb_buffer_pool_instances = 8

# Chunk size (MySQL 8.0+)
innodb_buffer_pool_chunk_size = 128M
```

**Log Files:**
```ini
# Redo log size (larger = better write performance, slower crash recovery)
innodb_log_file_size = 1G          # MySQL 5.7
innodb_redo_log_capacity = 2G      # MySQL 8.0.30+

# Log buffer
innodb_log_buffer_size = 64M

# Flush method (O_DIRECT recommended for Linux)
innodb_flush_method = O_DIRECT

# Flush log at transaction commit
innodb_flush_log_at_trx_commit = 1  # Safest (default)
# innodb_flush_log_at_trx_commit = 2  # Faster, slight risk on OS crash
```

**I/O Configuration:**
```ini
# I/O capacity (based on storage - SSD values shown)
innodb_io_capacity = 2000          # Baseline I/O ops/sec
innodb_io_capacity_max = 4000      # Max I/O ops/sec

# Read/write threads
innodb_read_io_threads = 8
innodb_write_io_threads = 8

# File-per-table (recommended)
innodb_file_per_table = ON

# Adaptive hash index
innodb_adaptive_hash_index = ON
```

**Connection and Thread Configuration:**
```ini
# Max connections
max_connections = 500

# Thread cache (reuse threads)
thread_cache_size = 100

# Thread pool (MariaDB default, MySQL Enterprise)
thread_handling = pool-of-threads
thread_pool_size = 16              # Number of thread groups

# Connection timeout
wait_timeout = 600                 # 10 minutes
interactive_timeout = 600
```

**Query Cache (Removed in MySQL 8.0):**
```ini
# MySQL 5.7 and earlier only
query_cache_type = 0               # Disabled (recommended)
query_cache_size = 0

# Why disabled?
# - High contention on cache mutex
# - Any write invalidates cache
# - Use application-level caching (Redis, Memcached)
```

**Temporary Tables:**
```ini
# In-memory temp table size
tmp_table_size = 256M
max_heap_table_size = 256M

# Temporary directory (use SSD)
tmpdir = /var/tmp
```

**Binary Logging (for replication/backups):**
```ini
# Enable binary logs
log_bin = /var/log/mysql/mysql-bin
binlog_format = ROW                # ROW, STATEMENT, or MIXED
binlog_expire_logs_seconds = 604800  # 7 days (MySQL 8.0+)
expire_logs_days = 7               # MySQL 5.7

# Binary log cache
binlog_cache_size = 32M

# Sync to disk (1 = safest, 0 = fastest)
sync_binlog = 1
```

### Per-Connection Buffers

**Sort and Join Buffers:**
```ini
# These are allocated PER CONNECTION - be conservative!
sort_buffer_size = 2M              # For ORDER BY operations
join_buffer_size = 2M              # For table joins without indexes
read_buffer_size = 2M              # Sequential scans
read_rnd_buffer_size = 4M          # Random reads (ORDER BY)

# Calculate max memory: max_connections * (sort + join + read buffers)
# Example: 500 * (2M + 2M + 2M + 4M) = 5GB
```

### Character Set and Collation

**UTF8MB4 (Recommended):**
```sql
-- Default character set (supports emojis and all Unicode)
CREATE DATABASE myapp
CHARACTER SET utf8mb4
COLLATE utf8mb4_unicode_ci;

-- Table level
CREATE TABLE users (
    id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(255)
) CHARACTER SET utf8mb4
  COLLATE utf8mb4_unicode_ci;

-- Server default (my.cnf)
[mysqld]
character_set_server = utf8mb4
collation_server = utf8mb4_unicode_ci
```

**Character Set Gotchas:**
- ⚠️ `utf8` in MySQL = `utf8mb3` (max 3 bytes, doesn't support emojis)
- ✅ Always use `utf8mb4` (max 4 bytes, full Unicode support)
- ⚠️ VARCHAR(255) with utf8mb4 = 1020 bytes (4 bytes per char)

### Row Format

**Dynamic vs Compact vs Compressed:**
```sql
-- Dynamic (recommended for modern MySQL/MariaDB)
CREATE TABLE users (
    id BIGINT PRIMARY KEY,
    data TEXT
) ROW_FORMAT=DYNAMIC;

-- Compressed (save disk space, CPU overhead)
CREATE TABLE logs (
    id BIGINT PRIMARY KEY,
    message TEXT
) ROW_FORMAT=COMPRESSED;

-- Check row format
SELECT table_name, row_format
FROM information_schema.tables
WHERE table_schema = 'myapp';
```

---

## 5. Replication Strategies

### Master-Slave Replication (Traditional)

**Architecture:**
```
┌─────────┐        Binary Log        ┌─────────┐
│ Master  │ ────────────────────────> │ Slave 1 │
│ (Write) │                           │ (Read)  │
└─────────┘                           └─────────┘
     │
     │              Binary Log
     └──────────────────────────────> ┌─────────┐
                                      │ Slave 2 │
                                      │ (Read)  │
                                      └─────────┘
```

**Master Configuration:**
```ini
# my.cnf on master
[mysqld]
server_id = 1
log_bin = /var/log/mysql/mysql-bin
binlog_format = ROW
binlog_expire_logs_seconds = 604800

# Optional: Binary log filtering
binlog_do_db = myapp
# binlog_ignore_db = test

# GTID (Global Transaction ID) - recommended
gtid_mode = ON
enforce_gtid_consistency = ON
```

**Slave Configuration:**
```ini
# my.cnf on slave
[mysqld]
server_id = 2
relay_log = /var/log/mysql/relay-bin
read_only = ON                    # Prevent writes to slave

# GTID
gtid_mode = ON
enforce_gtid_consistency = ON

# Replication options
replicate_do_db = myapp
# replicate_ignore_db = test
```

**Setup Replication:**
```sql
-- On master: Create replication user
CREATE USER 'replicator'@'%' IDENTIFIED BY 'strong_password';
GRANT REPLICATION SLAVE ON *.* TO 'replicator'@'%';
FLUSH PRIVILEGES;

-- Get master status
SHOW MASTER STATUS;
-- Note: File and Position (or use GTID)

-- On slave: Configure replication
CHANGE MASTER TO
    MASTER_HOST='master_ip',
    MASTER_USER='replicator',
    MASTER_PASSWORD='strong_password',
    MASTER_LOG_FILE='mysql-bin.000001',  -- From SHOW MASTER STATUS
    MASTER_LOG_POS=154;                  -- From SHOW MASTER STATUS

-- With GTID (recommended)
CHANGE MASTER TO
    MASTER_HOST='master_ip',
    MASTER_USER='replicator',
    MASTER_PASSWORD='strong_password',
    MASTER_AUTO_POSITION=1;

-- Start replication
START SLAVE;

-- Check replication status
SHOW SLAVE STATUS\G

-- Key fields to monitor:
-- Slave_IO_Running: Yes
-- Slave_SQL_Running: Yes
-- Seconds_Behind_Master: 0 (or low)
-- Last_Error: (should be empty)
```

### Multi-Master Replication (MariaDB Galera)

**Galera Cluster (MariaDB):**
```
┌──────────┐     ┌──────────┐     ┌──────────┐
│  Node 1  │<───>│  Node 2  │<───>│  Node 3  │
│ (Master) │     │ (Master) │     │ (Master) │
└──────────┘     └──────────┘     └──────────┘
  Read/Write      Read/Write       Read/Write
```

**Galera Configuration:**
```ini
# my.cnf on all nodes
[mysqld]
binlog_format = ROW
default_storage_engine = InnoDB
innodb_autoinc_lock_mode = 2

# Galera settings
wsrep_on = ON
wsrep_provider = /usr/lib/libgalera_smm.so

# Cluster configuration
wsrep_cluster_name = "my_cluster"
wsrep_cluster_address = "gcomm://node1_ip,node2_ip,node3_ip"
wsrep_node_name = "node1"
wsrep_node_address = "node1_ip"

# SST method (full sync)
wsrep_sst_method = mariabackup
wsrep_sst_auth = "sst_user:sst_password"

# Replication threads
wsrep_slave_threads = 4
```

**Bootstrap Galera Cluster:**
```bash
# On first node only (bootstrap)
galera_new_cluster

# On other nodes (join cluster)
systemctl start mariadb

# Check cluster status
mysql -e "SHOW STATUS LIKE 'wsrep_cluster_size';"
mysql -e "SHOW STATUS LIKE 'wsrep_local_state_comment';"
```

### MySQL Group Replication (MySQL 8.0+)

**Single-Primary Mode:**
```sql
-- Install plugin
INSTALL PLUGIN group_replication SONAME 'group_replication.so';

-- Configure group replication
SET PERSIST group_replication_group_name = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa";
SET PERSIST group_replication_start_on_boot = OFF;
SET PERSIST group_replication_local_address = "node1:33061";
SET PERSIST group_replication_group_seeds = "node1:33061,node2:33061,node3:33061";
SET PERSIST group_replication_bootstrap_group = OFF;

-- Create replication user
CREATE USER 'repl'@'%' IDENTIFIED BY 'password';
GRANT REPLICATION SLAVE ON *.* TO 'repl'@'%';

-- Configure recovery channel
CHANGE REPLICATION SOURCE TO SOURCE_USER='repl', SOURCE_PASSWORD='password'
FOR CHANNEL 'group_replication_recovery';

-- Bootstrap group (first node only)
SET GLOBAL group_replication_bootstrap_group = ON;
START GROUP_REPLICATION;
SET GLOBAL group_replication_bootstrap_group = OFF;

-- Join group (other nodes)
START GROUP_REPLICATION;

-- Check group status
SELECT * FROM performance_schema.replication_group_members;
```

### Semi-Synchronous Replication

**Enhanced Durability:**
```sql
-- Install plugins (master and slave)
INSTALL PLUGIN rpl_semi_sync_master SONAME 'semisync_master.so';
INSTALL PLUGIN rpl_semi_sync_slave SONAME 'semisync_slave.so';

-- Enable on master
SET GLOBAL rpl_semi_sync_master_enabled = 1;
SET GLOBAL rpl_semi_sync_master_timeout = 1000;  -- 1 second

-- Enable on slave
SET GLOBAL rpl_semi_sync_slave_enabled = 1;

-- Restart replication on slave
STOP SLAVE IO_THREAD;
START SLAVE IO_THREAD;

-- Verify
SHOW STATUS LIKE 'Rpl_semi_sync%';
```

---

## 6. Sharding and Partitioning

### Table Partitioning (Single Server)

**Range Partitioning:**
```sql
-- Partition by date range
CREATE TABLE orders (
    id BIGINT UNSIGNED AUTO_INCREMENT,
    user_id BIGINT UNSIGNED,
    order_date DATE,
    total DECIMAL(10,2),
    PRIMARY KEY (id, order_date)
) PARTITION BY RANGE (YEAR(order_date)) (
    PARTITION p2022 VALUES LESS THAN (2023),
    PARTITION p2023 VALUES LESS THAN (2024),
    PARTITION p2024 VALUES LESS THAN (2025),
    PARTITION p2025 VALUES LESS THAN (2026),
    PARTITION p_future VALUES LESS THAN MAXVALUE
);

-- Add new partition
ALTER TABLE orders ADD PARTITION (
    PARTITION p2026 VALUES LESS THAN (2027)
);

-- Drop old partition (fast DELETE)
ALTER TABLE orders DROP PARTITION p2022;
```

**Hash Partitioning:**
```sql
-- Distribute data evenly across partitions
CREATE TABLE users (
    id BIGINT UNSIGNED AUTO_INCREMENT,
    email VARCHAR(255),
    PRIMARY KEY (id)
) PARTITION BY HASH(id)
PARTITIONS 8;
```

**List Partitioning:**
```sql
-- Partition by specific values
CREATE TABLE customers (
    id BIGINT UNSIGNED AUTO_INCREMENT,
    country VARCHAR(2),
    name VARCHAR(255),
    PRIMARY KEY (id, country)
) PARTITION BY LIST COLUMNS(country) (
    PARTITION p_us VALUES IN ('US'),
    PARTITION p_eu VALUES IN ('DE', 'FR', 'UK', 'IT'),
    PARTITION p_asia VALUES IN ('JP', 'CN', 'IN'),
    PARTITION p_other VALUES IN (DEFAULT)
);
```

**Key Partitioning:**
```sql
-- Partition by primary key
CREATE TABLE sessions (
    session_id VARCHAR(128),
    data TEXT,
    expires_at TIMESTAMP,
    PRIMARY KEY (session_id)
) PARTITION BY KEY(session_id)
PARTITIONS 16;
```

### Application-Level Sharding

**Horizontal Sharding Strategy:**
```
Application Layer
       │
       ├──────┬──────┬──────┬──────┐
       │      │      │      │      │
    Shard0 Shard1 Shard2 Shard3 Shard4
    (users  (users (users (users (users
     0-199) 200-399) 400-599) 600-799) 800-999)
```

**Shard Key Selection:**
```python
# Python example: Shard by user_id
def get_shard(user_id, num_shards=4):
    return user_id % num_shards

# Route query to correct shard
user_id = 12345
shard_id = get_shard(user_id)
connection = shard_connections[shard_id]
cursor = connection.cursor()
cursor.execute("SELECT * FROM users WHERE id = %s", (user_id,))
```

**Shard Configuration:**
```python
# Database connections for each shard
SHARDS = {
    0: {'host': 'shard0.db.example.com', 'database': 'myapp_shard0'},
    1: {'host': 'shard1.db.example.com', 'database': 'myapp_shard1'},
    2: {'host': 'shard2.db.example.com', 'database': 'myapp_shard2'},
    3: {'host': 'shard3.db.example.com', 'database': 'myapp_shard3'},
}

class ShardManager:
    def __init__(self, shards_config):
        self.connections = {}
        for shard_id, config in shards_config.items():
            self.connections[shard_id] = self.create_connection(config)

    def get_connection(self, user_id):
        shard_id = user_id % len(self.connections)
        return self.connections[shard_id]
```

### Vitess (MySQL Sharding Platform)

**Vitess Architecture:**
- Developed by YouTube for MySQL sharding
- Handles routing, connection pooling, query rewriting
- Transparent sharding from application perspective

**Vitess Configuration Example:**
```yaml
# vschema.json
{
  "sharded": true,
  "vindexes": {
    "hash": {
      "type": "hash"
    }
  },
  "tables": {
    "users": {
      "column_vindexes": [
        {
          "column": "user_id",
          "name": "hash"
        }
      ]
    }
  }
}
```

---

## 7. Indexing Strategies

### Index Types

**B-Tree Index (Default):**
```sql
-- Single column index
CREATE INDEX idx_users_email ON users(email);

-- Multi-column index (order matters!)
CREATE INDEX idx_users_name ON users(last_name, first_name);

-- Covering index (includes all columns in query)
CREATE INDEX idx_users_covering ON users(email, name, created_at);
```

**Unique Index:**
```sql
-- Enforce uniqueness
CREATE UNIQUE INDEX idx_users_email_unique ON users(email);

-- Unique composite index
CREATE UNIQUE INDEX idx_users_username_tenant
ON users(username, tenant_id);
```

**Full-Text Index:**
```sql
-- Full-text search (InnoDB supports since MySQL 5.6)
CREATE FULLTEXT INDEX idx_posts_content ON posts(title, content);

-- Full-text search queries
SELECT * FROM posts
WHERE MATCH(title, content) AGAINST ('mysql performance' IN NATURAL LANGUAGE MODE);

-- Boolean mode search
SELECT * FROM posts
WHERE MATCH(title, content) AGAINST ('+mysql -oracle' IN BOOLEAN MODE);

-- With relevance score
SELECT *, MATCH(title, content) AGAINST ('mysql') as relevance
FROM posts
WHERE MATCH(title, content) AGAINST ('mysql')
ORDER BY relevance DESC;
```

**Spatial Index (GIS):**
```sql
-- For geospatial data
CREATE TABLE places (
    id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(255),
    location POINT NOT NULL,
    SPATIAL INDEX idx_location (location)
) ENGINE=InnoDB;

-- Find nearby locations
SELECT name, ST_Distance(location, POINT(40.7128, -74.0060)) as distance
FROM places
WHERE ST_Distance_Sphere(location, POINT(40.7128, -74.0060)) < 1000  -- 1km
ORDER BY distance;
```

**Prefix Index:**
```sql
-- Index first N characters (save space)
CREATE INDEX idx_users_email_prefix ON users(email(20));

-- Useful for long VARCHAR/TEXT columns
CREATE INDEX idx_posts_content_prefix ON posts(content(100));
```

**Functional/Expression Index (MySQL 8.0+):**
```sql
-- Index on computed expression
CREATE INDEX idx_users_lower_email ON users((LOWER(email)));

-- Query can use the index
SELECT * FROM users WHERE LOWER(email) = 'user@example.com';

-- JSON field index
CREATE INDEX idx_users_json_city ON users((CAST(profile->>'$.city' AS CHAR(50))));
```

### Indexing Best Practices

**When to Create Indexes:**
- ✅ Columns in WHERE clauses
- ✅ Columns in JOIN conditions
- ✅ Columns in ORDER BY
- ✅ Columns in GROUP BY
- ✅ Foreign key columns
- ✅ Columns with high cardinality

**When NOT to Index:**
- ❌ Small tables (<1000 rows)
- ❌ Columns with low cardinality (gender, boolean)
- ❌ Columns updated frequently (index maintenance overhead)
- ❌ Wide columns (BLOBs, large TEXT)

**Composite Index Order:**
```sql
-- LEFT-TO-RIGHT prefix rule
CREATE INDEX idx_users ON users(status, created_at, email);

-- These queries use the index:
WHERE status = 'active'
WHERE status = 'active' AND created_at > '2024-01-01'
WHERE status = 'active' AND created_at > '2024-01-01' AND email LIKE 'user%'

-- This query does NOT use the index efficiently:
WHERE created_at > '2024-01-01'  -- Doesn't start with status
```

**Index Maintenance:**
```sql
-- Show indexes for a table
SHOW INDEX FROM users;

-- Analyze table (update statistics)
ANALYZE TABLE users;

-- Optimize table (rebuild indexes, reclaim space)
OPTIMIZE TABLE users;

-- Drop unused index
DROP INDEX idx_users_old ON users;

-- Check index usage
SELECT
    table_schema,
    table_name,
    index_name,
    rows_read,
    rows_inserted,
    rows_updated,
    rows_deleted
FROM performance_schema.table_io_waits_summary_by_index_usage
WHERE table_schema = 'myapp'
ORDER BY rows_read DESC;
```

### Invisible Indexes (MySQL 8.0+)

**Test Index Before Dropping:**
```sql
-- Make index invisible (optimizer ignores it)
ALTER TABLE users ALTER INDEX idx_users_email INVISIBLE;

-- Monitor query performance
-- If no issues, drop the index

-- Make visible again
ALTER TABLE users ALTER INDEX idx_users_email VISIBLE;

-- Or drop it
DROP INDEX idx_users_email ON users;
```

---

## 8. Query Optimization

### EXPLAIN and Query Analysis

**EXPLAIN Statement:**
```sql
EXPLAIN SELECT u.name, COUNT(o.id) as order_count
FROM users u
LEFT JOIN orders o ON u.id = o.user_id
WHERE u.status = 'active'
GROUP BY u.id;

-- Key columns in EXPLAIN output:
-- type: ALL (bad), index, range, ref, eq_ref, const (good)
-- key: Index used (NULL = no index)
-- rows: Estimated rows scanned
-- Extra: Additional information
```

**EXPLAIN ANALYZE (MySQL 8.0.18+):**
```sql
-- Shows actual execution statistics
EXPLAIN ANALYZE
SELECT * FROM orders WHERE order_date > '2024-01-01';

-- Output includes:
-- - Actual rows scanned
-- - Execution time
-- - Cost estimates
```

**Visual EXPLAIN (MySQL Workbench):**
```sql
-- Use Visual EXPLAIN in MySQL Workbench for graphical query plan
```

### Query Optimization Techniques

**Avoid SELECT \*:**
```sql
-- BAD: Retrieves all columns (waste of I/O and memory)
SELECT * FROM users WHERE id = 1;

-- GOOD: Select only needed columns
SELECT id, email, name FROM users WHERE id = 1;
```

**Use Covering Indexes:**
```sql
-- Query: SELECT email, name FROM users WHERE status = 'active';

-- Create covering index (includes all columns in query)
CREATE INDEX idx_users_status_covering ON users(status, email, name);

-- EXPLAIN will show: "Using index" (no table access needed)
```

**Avoid Functions on Indexed Columns:**
```sql
-- BAD: Function prevents index usage
SELECT * FROM users WHERE YEAR(created_at) = 2024;

-- GOOD: Range query uses index
SELECT * FROM users
WHERE created_at >= '2024-01-01'
  AND created_at < '2025-01-01';
```

**Use LIMIT:**
```sql
-- Always use LIMIT for large result sets
SELECT * FROM orders ORDER BY created_at DESC LIMIT 100;

-- Pagination with OFFSET (works but slow for large offsets)
SELECT * FROM orders ORDER BY id LIMIT 100 OFFSET 10000;

-- Better: Keyset pagination
SELECT * FROM orders
WHERE id > 10000
ORDER BY id
LIMIT 100;
```

**JOIN Optimization:**
```sql
-- Use INNER JOIN when possible (faster than LEFT JOIN)
SELECT u.name, o.total
FROM users u
INNER JOIN orders o ON u.id = o.user_id;

-- Index foreign key columns
CREATE INDEX idx_orders_user_id ON orders(user_id);

-- STRAIGHT_JOIN to force join order (use carefully)
SELECT STRAIGHT_JOIN u.name, o.total
FROM users u
JOIN orders o ON u.id = o.user_id;
```

**Subquery Optimization:**
```sql
-- BAD: Correlated subquery (executes for each row)
SELECT name,
    (SELECT COUNT(*) FROM orders WHERE user_id = u.id) as order_count
FROM users u;

-- GOOD: JOIN with GROUP BY
SELECT u.name, COUNT(o.id) as order_count
FROM users u
LEFT JOIN orders o ON u.id = o.user_id
GROUP BY u.id;

-- Or use derived table
SELECT u.name, COALESCE(o.order_count, 0) as order_count
FROM users u
LEFT JOIN (
    SELECT user_id, COUNT(*) as order_count
    FROM orders
    GROUP BY user_id
) o ON u.id = o.user_id;
```

**EXISTS vs IN:**
```sql
-- Use EXISTS for better performance (stops at first match)
SELECT * FROM users u
WHERE EXISTS (
    SELECT 1 FROM orders WHERE user_id = u.id
);

-- IN works well for small lists
SELECT * FROM users WHERE id IN (1, 2, 3, 4, 5);

-- Avoid IN with subquery on large tables
-- BAD:
SELECT * FROM users WHERE id IN (SELECT user_id FROM orders);
-- GOOD (rewrite as JOIN):
SELECT DISTINCT u.* FROM users u
INNER JOIN orders o ON u.id = o.user_id;
```

### Query Cache Alternatives (MySQL 8.0+)

**Application-Level Caching:**
```python
import redis
import json
import mysql.connector

cache = redis.Redis(host='localhost', port=6379)

def get_user(user_id):
    # Try cache first
    cache_key = f"user:{user_id}"
    cached = cache.get(cache_key)
    if cached:
        return json.loads(cached)

    # Query database
    conn = mysql.connector.connect(host='localhost', database='myapp')
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM users WHERE id = %s", (user_id,))
    user = cursor.fetchone()

    # Store in cache (expire in 1 hour)
    cache.setex(cache_key, 3600, json.dumps(user))

    return user
```

### Slow Query Log

**Enable Slow Query Logging:**
```ini
# my.cnf
[mysqld]
slow_query_log = 1
slow_query_log_file = /var/log/mysql/slow-query.log
long_query_time = 2                    # Queries taking > 2 seconds
log_queries_not_using_indexes = 1      # Log queries without indexes
```

**Analyze Slow Queries:**
```bash
# mysqldumpslow (summarize slow query log)
mysqldumpslow -s t -t 10 /var/log/mysql/slow-query.log

# Options:
# -s t: Sort by query time
# -s c: Sort by count
# -s l: Sort by lock time
# -t 10: Top 10 queries

# pt-query-digest (Percona Toolkit - recommended)
pt-query-digest /var/log/mysql/slow-query.log
```

---

## 9. Transaction Management

### ACID Properties

MySQL InnoDB provides full ACID guarantees:
- **Atomicity:** All or nothing
- **Consistency:** Constraints enforced
- **Isolation:** Transactions isolated from each other
- **Durability:** Committed data survives crashes

### Transaction Syntax

**Basic Transactions:**
```sql
START TRANSACTION;

INSERT INTO accounts (user_id, balance) VALUES (1, 1000);
UPDATE accounts SET balance = balance - 100 WHERE user_id = 1;
INSERT INTO transactions (user_id, amount) VALUES (1, -100);

COMMIT;
-- Or ROLLBACK to undo changes
```

**With Error Handling (Python):**
```python
import mysql.connector

conn = mysql.connector.connect(host='localhost', database='myapp')
try:
    conn.start_transaction()

    cursor = conn.cursor()
    cursor.execute("UPDATE accounts SET balance = balance - 100 WHERE user_id = %s", (1,))
    cursor.execute("UPDATE accounts SET balance = balance + 100 WHERE user_id = %s", (2,))

    conn.commit()
    print("Transaction committed successfully")
except Exception as e:
    conn.rollback()
    print(f"Transaction rolled back: {e}")
finally:
    conn.close()
```

### Isolation Levels

**Four Isolation Levels:**
```sql
-- 1. READ UNCOMMITTED (dirty reads possible)
SET SESSION TRANSACTION ISOLATION LEVEL READ UNCOMMITTED;

-- 2. READ COMMITTED (no dirty reads, but non-repeatable reads)
SET SESSION TRANSACTION ISOLATION LEVEL READ COMMITTED;

-- 3. REPEATABLE READ (default in MySQL, prevents non-repeatable reads)
SET SESSION TRANSACTION ISOLATION LEVEL REPEATABLE READ;

-- 4. SERIALIZABLE (strictest, full isolation)
SET SESSION TRANSACTION ISOLATION LEVEL SERIALIZABLE;

-- Check current level
SELECT @@transaction_isolation;
```

**Isolation Level Comparison:**

| Level | Dirty Read | Non-Repeatable Read | Phantom Read |
|-------|------------|---------------------|--------------|
| READ UNCOMMITTED | Yes | Yes | Yes |
| READ COMMITTED | No | Yes | Yes |
| REPEATABLE READ | No | No | Yes* |
| SERIALIZABLE | No | No | No |

*InnoDB prevents phantom reads in REPEATABLE READ using gap locks

### Locking Mechanisms

**Explicit Locking:**
```sql
-- Shared lock (read lock)
SELECT * FROM users WHERE id = 1 LOCK IN SHARE MODE;

-- Exclusive lock (write lock)
SELECT * FROM users WHERE id = 1 FOR UPDATE;

-- Skip locked rows (MySQL 8.0+)
SELECT * FROM queue
WHERE status = 'pending'
LIMIT 1
FOR UPDATE SKIP LOCKED;

-- Wait for lock with timeout (MySQL 8.0+)
SELECT * FROM users WHERE id = 1 FOR UPDATE NOWAIT;
```

**Row-Level Locking:**
```sql
-- InnoDB uses row-level locks automatically
UPDATE users SET name = 'John' WHERE id = 1;
-- Locks only row with id=1, not entire table
```

**Table-Level Locking:**
```sql
-- Explicit table lock
LOCK TABLES users WRITE, orders READ;

-- Perform operations
UPDATE users SET status = 'active' WHERE id = 1;
SELECT * FROM orders WHERE user_id = 1;

-- Release locks
UNLOCK TABLES;
```

### Savepoints

**Nested Transactions:**
```sql
START TRANSACTION;

INSERT INTO users (email) VALUES ('user1@example.com');

SAVEPOINT sp1;

INSERT INTO users (email) VALUES ('user2@example.com');

-- Error occurs, rollback to savepoint
ROLLBACK TO sp1;

-- user1 will be inserted, user2 will not
COMMIT;
```

### Deadlock Detection

**Deadlock Handling:**
```sql
-- InnoDB automatically detects deadlocks and rolls back one transaction

-- View deadlock information
SHOW ENGINE INNODB STATUS\G

-- Look for "LATEST DETECTED DEADLOCK" section
```

**Prevent Deadlocks:**
```python
# 1. Access tables in same order
# 2. Keep transactions short
# 3. Use appropriate isolation level
# 4. Retry on deadlock

def transfer_money(from_user, to_user, amount, max_retries=3):
    for attempt in range(max_retries):
        try:
            conn.start_transaction()

            # Always lock accounts in same order (by ID)
            first_id, second_id = sorted([from_user, to_user])

            cursor.execute("SELECT balance FROM accounts WHERE id = %s FOR UPDATE", (first_id,))
            cursor.execute("SELECT balance FROM accounts WHERE id = %s FOR UPDATE", (second_id,))

            cursor.execute("UPDATE accounts SET balance = balance - %s WHERE id = %s", (amount, from_user))
            cursor.execute("UPDATE accounts SET balance = balance + %s WHERE id = %s", (amount, to_user))

            conn.commit()
            return True
        except mysql.connector.errors.InternalError as e:
            if e.errno == 1213:  # Deadlock
                conn.rollback()
                if attempt < max_retries - 1:
                    time.sleep(0.1 * (2 ** attempt))  # Exponential backoff
                    continue
            raise
    return False
```

---

## 10. Connection Pooling

### Why Connection Pooling?

- ✅ **Reduces connection overhead** (creating connections is expensive)
- ✅ **Limits max connections** (prevents overwhelming database)
- ✅ **Reuses connections** (better resource utilization)
- ✅ **Faster query execution** (no connection setup time)

### Python Connection Pooling

**MySQL Connector Pool:**
```python
import mysql.connector.pooling

# Create connection pool
pool = mysql.connector.pooling.MySQLConnectionPool(
    pool_name="myapp_pool",
    pool_size=10,              # Number of connections in pool
    pool_reset_session=True,   # Reset session on connection return
    host='localhost',
    database='myapp',
    user='appuser',
    password='password'
)

# Get connection from pool
connection = pool.get_connection()
try:
    cursor = connection.cursor()
    cursor.execute("SELECT * FROM users WHERE id = %s", (1,))
    result = cursor.fetchone()
finally:
    connection.close()  # Returns connection to pool
```

**SQLAlchemy Pool (Recommended):**
```python
from sqlalchemy import create_engine, pool

# Create engine with connection pool
engine = create_engine(
    'mysql+mysqlconnector://user:password@localhost/myapp',
    pool_size=20,              # Number of connections to maintain
    max_overflow=10,           # Max connections beyond pool_size
    pool_timeout=30,           # Seconds to wait for available connection
    pool_recycle=3600,         # Recycle connections after 1 hour
    pool_pre_ping=True,        # Test connection before using
    echo_pool=True             # Log pool activity (debug only)
)

# Use connection
with engine.connect() as connection:
    result = connection.execute("SELECT * FROM users WHERE id = %s", (1,))
    user = result.fetchone()
```

### Node.js Connection Pooling

**mysql2 Pool:**
```javascript
const mysql = require('mysql2/promise');

// Create connection pool
const pool = mysql.createPool({
  host: 'localhost',
  user: 'appuser',
  password: 'password',
  database: 'myapp',
  waitForConnections: true,
  connectionLimit: 10,        // Max connections
  queueLimit: 0,              // Unlimited queue
  enableKeepAlive: true,
  keepAliveInitialDelay: 10000
});

// Execute query
async function getUser(userId) {
  const [rows] = await pool.execute(
    'SELECT * FROM users WHERE id = ?',
    [userId]
  );
  return rows[0];
}

// Close pool on shutdown
process.on('SIGTERM', async () => {
  await pool.end();
});
```

### Java (HikariCP - Fastest Pool)

**HikariCP Configuration:**
```java
import com.zaxxer.hikari.HikariConfig;
import com.zaxxer.hikari.HikariDataSource;

HikariConfig config = new HikariConfig();
config.setJdbcUrl("jdbc:mysql://localhost:3306/myapp");
config.setUsername("appuser");
config.setPassword("password");
config.setMaximumPoolSize(20);
config.setMinimumIdle(5);
config.setConnectionTimeout(30000);      // 30 seconds
config.setIdleTimeout(600000);           // 10 minutes
config.setMaxLifetime(1800000);          // 30 minutes
config.setAutoCommit(true);
config.setPoolName("MyAppPool");

// Recommended settings
config.addDataSourceProperty("cachePrepStmts", "true");
config.addDataSourceProperty("prepStmtCacheSize", "250");
config.addDataSourceProperty("prepStmtCacheSqlLimit", "2048");

HikariDataSource ds = new HikariDataSource(config);

// Use connection
try (Connection conn = ds.getConnection()) {
    PreparedStatement stmt = conn.prepareStatement("SELECT * FROM users WHERE id = ?");
    stmt.setInt(1, userId);
    ResultSet rs = stmt.executeQuery();
}
```

### Pool Sizing Guidelines

**Formula:**
```
connections = ((core_count * 2) + effective_spindle_count)

For web applications:
pool_size = (available_memory / memory_per_connection) * 0.8

Typical settings:
- Small app: 5-10 connections
- Medium app: 20-50 connections
- Large app: 100-200 connections
```

**Monitor Pool Usage:**
```sql
-- Check current connections
SHOW PROCESSLIST;

-- Count connections by state
SELECT
    command,
    COUNT(*) as connection_count
FROM information_schema.processlist
GROUP BY command;

-- Connection statistics
SHOW STATUS LIKE 'Threads%';
-- Threads_connected: Current connections
-- Threads_running: Active queries
-- Threads_created: Total threads created
```

---

## 11. High Availability

### MySQL HA Architectures

**1. Master-Slave with Failover:**
```
┌─────────┐                      ┌─────────┐
│ Master  │ ───── Replication ──>│ Slave   │
│ (Active)│                      │(Standby)│
└─────────┘                      └─────────┘
     │                                 │
     └─── Failover Tool ───────────────┘
          (MHA, Orchestrator)
```

**2. Galera Cluster (MariaDB):**
```
┌──────────┐     ┌──────────┐     ┌──────────┐
│  Node 1  │<───>│  Node 2  │<───>│  Node 3  │
│ (Active) │     │ (Active) │     │ (Active) │
└──────────┘     └──────────┘     └──────────┘
      ^                ^                ^
      └────────────────┴────────────────┘
           Load Balancer (HAProxy)
```

**3. Group Replication (MySQL 8.0+):**
```
┌──────────┐     ┌──────────┐     ┌──────────┐
│ Primary  │────>│Secondary │────>│Secondary │
│  (R/W)   │     │   (R)    │     │   (R)    │
└──────────┘     └──────────┘     └──────────┘
```

### MySQL HA Tools

**MHA (Master High Availability):**
```ini
# /etc/mha/app1.conf
[server default]
manager_workdir=/var/log/mha/app1
manager_log=/var/log/mha/app1/manager.log
remote_workdir=/var/log/mha/app1

master_binlog_dir=/var/lib/mysql
ssh_user=root
repl_user=replicator
repl_password=password

[server1]
hostname=db1.example.com
candidate_master=1

[server2]
hostname=db2.example.com
candidate_master=1

[server3]
hostname=db3.example.com
no_master=1
```

**Orchestrator (Automated Failover):**
```json
{
  "MySQLTopologyUser": "orchestrator",
  "MySQLTopologyPassword": "password",
  "MySQLOrchestratorHost": "localhost",
  "MySQLOrchestratorPort": 3306,
  "MySQLOrchestratorDatabase": "orchestrator",
  "RecoveryPeriodBlockSeconds": 3600,
  "AutoFailover": true,
  "FailoverPeriodBlockMinutes": 60
}
```

### ProxySQL (Load Balancing & Query Routing)

**ProxySQL Configuration:**
```sql
-- Add MySQL servers
INSERT INTO mysql_servers (hostgroup_id, hostname, port) VALUES
(0, 'master.db.example.com', 3306),
(1, 'slave1.db.example.com', 3306),
(1, 'slave2.db.example.com', 3306);

-- Configure users
INSERT INTO mysql_users (username, password, default_hostgroup) VALUES
('appuser', 'password', 0);

-- Query routing rules
INSERT INTO mysql_query_rules (rule_id, active, match_pattern, destination_hostgroup, apply)
VALUES
(1, 1, '^SELECT.*FOR UPDATE', 0, 1),  -- Write queries to master
(2, 1, '^SELECT', 1, 1);              -- Read queries to slaves

-- Load config
LOAD MYSQL SERVERS TO RUNTIME;
LOAD MYSQL USERS TO RUNTIME;
LOAD MYSQL QUERY RULES TO RUNTIME;

-- Save to disk
SAVE MYSQL SERVERS TO DISK;
SAVE MYSQL USERS TO DISK;
SAVE MYSQL QUERY RULES TO DISK;
```

### HAProxy (Load Balancing)

**HAProxy Configuration:**
```conf
# /etc/haproxy/haproxy.cfg

global
    maxconn 4096
    log 127.0.0.1 local0

defaults
    mode tcp
    timeout connect 10s
    timeout client 1m
    timeout server 1m

# MySQL master (write)
listen mysql-master
    bind *:3306
    mode tcp
    option mysql-check user haproxy
    server master1 db1.example.com:3306 check

# MySQL slaves (read)
listen mysql-slaves
    bind *:3307
    mode tcp
    balance roundrobin
    option mysql-check user haproxy
    server slave1 db2.example.com:3306 check
    server slave2 db3.example.com:3306 check
```

### Virtual IP Failover (Keepalived)

**Keepalived Configuration:**
```conf
# /etc/keepalived/keepalived.conf

vrrp_script check_mysql {
    script "/usr/bin/mysql -u root -p'password' -e 'SELECT 1'"
    interval 2
    weight 2
}

vrrp_instance VI_1 {
    state MASTER
    interface eth0
    virtual_router_id 51
    priority 100
    advert_int 1

    virtual_ipaddress {
        192.168.1.100/24
    }

    track_script {
        check_mysql
    }
}
```

---

## 12. Security Best Practices

### User and Privilege Management

**Create Users with Least Privilege:**
```sql
-- Create application user (local access only)
CREATE USER 'appuser'@'localhost' IDENTIFIED BY 'strong_password_here';

-- Grant specific privileges
GRANT SELECT, INSERT, UPDATE, DELETE ON myapp.* TO 'appuser'@'localhost';

-- Read-only user (reports, analytics)
CREATE USER 'readonly'@'%' IDENTIFIED BY 'strong_password';
GRANT SELECT ON myapp.* TO 'readonly'@'%';

-- Replication user
CREATE USER 'replicator'@'%' IDENTIFIED BY 'repl_password';
GRANT REPLICATION SLAVE ON *.* TO 'replicator'@'%';

-- Apply changes
FLUSH PRIVILEGES;
```

**Password Policies (MySQL 8.0+):**
```sql
-- Set password validation policy
INSTALL COMPONENT 'file://component_validate_password';

SET GLOBAL validate_password.policy = STRONG;
SET GLOBAL validate_password.length = 12;
SET GLOBAL validate_password.mixed_case_count = 1;
SET GLOBAL validate_password.number_count = 1;
SET GLOBAL validate_password.special_char_count = 1;

-- Password expiration
ALTER USER 'appuser'@'localhost' PASSWORD EXPIRE INTERVAL 90 DAY;

-- Password reuse prevention
SET GLOBAL password_history = 5;
SET GLOBAL password_reuse_interval = 365;
```

**Audit Existing Privileges:**
```sql
-- Show all users
SELECT user, host FROM mysql.user;

-- Show user privileges
SHOW GRANTS FOR 'appuser'@'localhost';

-- Find users with dangerous privileges
SELECT user, host FROM mysql.user
WHERE Super_priv = 'Y' OR File_priv = 'Y';

-- Revoke unnecessary privileges
REVOKE SUPER ON *.* FROM 'appuser'@'localhost';
```

### Network Security

**Bind to Specific Interface:**
```ini
# my.cnf
[mysqld]
bind_address = 127.0.0.1     # Local only
# bind_address = 0.0.0.0     # All interfaces (use with firewall)
```

**Require SSL/TLS:**
```sql
-- Check SSL status
SHOW VARIABLES LIKE '%ssl%';

-- Require SSL for user
ALTER USER 'appuser'@'%' REQUIRE SSL;

-- Require specific cipher
ALTER USER 'appuser'@'%' REQUIRE CIPHER 'AES256-SHA';

-- Require X509 certificate
ALTER USER 'appuser'@'%' REQUIRE X509;
```

**Configure SSL (my.cnf):**
```ini
[mysqld]
ssl_ca = /etc/mysql/ssl/ca-cert.pem
ssl_cert = /etc/mysql/ssl/server-cert.pem
ssl_key = /etc/mysql/ssl/server-key.pem

require_secure_transport = ON
```

### Data Encryption

**Encryption at Rest (InnoDB):**
```sql
-- Enable encryption for tablespace
ALTER TABLE users ENCRYPTION='Y';

-- Create encrypted table
CREATE TABLE sensitive_data (
    id BIGINT PRIMARY KEY,
    ssn VARCHAR(11),
    credit_card VARCHAR(20)
) ENCRYPTION='Y';

-- Configure keyring (my.cnf)
[mysqld]
early-plugin-load=keyring_file.so
keyring_file_data=/var/lib/mysql-keyring/keyring
```

**Encryption in Transit (SSL):**
```python
import mysql.connector

# Python client with SSL
config = {
    'host': 'db.example.com',
    'user': 'appuser',
    'password': 'password',
    'database': 'myapp',
    'ssl_ca': '/path/to/ca-cert.pem',
    'ssl_verify_cert': True
}

conn = mysql.connector.connect(**config)
```

**Column-Level Encryption:**
```sql
-- Using AES encryption functions
CREATE TABLE users (
    id BIGINT PRIMARY KEY,
    email VARCHAR(255),
    ssn VARBINARY(256)  -- Encrypted column
);

-- Insert encrypted data
INSERT INTO users (id, email, ssn) VALUES
(1, 'user@example.com', AES_ENCRYPT('123-45-6789', 'encryption_key'));

-- Query encrypted data
SELECT
    id,
    email,
    CAST(AES_DECRYPT(ssn, 'encryption_key') AS CHAR) as ssn_decrypted
FROM users
WHERE id = 1;
```

### Audit Logging

**Enable Audit Log (MySQL Enterprise / MariaDB):**
```sql
-- MariaDB Audit Plugin
INSTALL PLUGIN server_audit SONAME 'server_audit.so';

SET GLOBAL server_audit_logging = ON;
SET GLOBAL server_audit_events = 'CONNECT,QUERY,TABLE';
SET GLOBAL server_audit_file_path = '/var/log/mysql/audit.log';
```

**General Query Log (Development Only):**
```sql
-- Enable general log (logs ALL queries - performance impact!)
SET GLOBAL general_log = 1;
SET GLOBAL general_log_file = '/var/log/mysql/general.log';

-- Disable in production
SET GLOBAL general_log = 0;
```

### SQL Injection Prevention

**Use Prepared Statements:**
```python
# WRONG: SQL injection vulnerability
user_input = "'; DROP TABLE users; --"
cursor.execute(f"SELECT * FROM users WHERE email = '{user_input}'")

# RIGHT: Parameterized query
cursor.execute("SELECT * FROM users WHERE email = %s", (user_input,))
```

**Input Validation:**
```python
import re

def validate_email(email):
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

# Validate before query
email = request.form['email']
if not validate_email(email):
    return "Invalid email format"

cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
```

---

## 13. Backup and Recovery

### Backup Strategies

**1. Logical Backup (mysqldump):**
```bash
# Full database backup
mysqldump -u root -p --single-transaction --routines --triggers \
  --databases myapp > myapp_backup.sql

# All databases
mysqldump -u root -p --all-databases --single-transaction > full_backup.sql

# Compressed backup
mysqldump -u root -p --single-transaction myapp | gzip > myapp_backup.sql.gz

# Single table
mysqldump -u root -p myapp users > users_backup.sql

# Restore
mysql -u root -p myapp < myapp_backup.sql
gunzip < myapp_backup.sql.gz | mysql -u root -p myapp
```

**2. Physical Backup (Percona XtraBackup):**
```bash
# Install XtraBackup
apt-get install percona-xtrabackup-80  # MySQL 8.0

# Full backup
xtrabackup --backup --target-dir=/backup/full \
  --user=root --password=password

# Prepare backup
xtrabackup --prepare --target-dir=/backup/full

# Incremental backup
xtrabackup --backup --target-dir=/backup/inc1 \
  --incremental-basedir=/backup/full

# Restore
systemctl stop mysql
rm -rf /var/lib/mysql/*
xtrabackup --copy-back --target-dir=/backup/full
chown -R mysql:mysql /var/lib/mysql
systemctl start mysql
```

**3. Binary Log Backup (Point-in-Time Recovery):**
```bash
# Flush binary logs
mysql -u root -p -e "FLUSH BINARY LOGS"

# Backup binary logs
cp /var/log/mysql/mysql-bin.* /backup/binlogs/

# Point-in-time recovery
mysqlbinlog /backup/binlogs/mysql-bin.000001 \
  --start-datetime="2024-02-01 10:00:00" \
  --stop-datetime="2024-02-01 10:30:00" \
  | mysql -u root -p myapp
```

**4. Snapshot Backup (LVM/Cloud):**
```bash
# LVM snapshot
lvcreate --size 10G --snapshot --name mysql-snapshot /dev/vg0/mysql-data

# Mount and backup
mount /dev/vg0/mysql-snapshot /mnt/snapshot
tar czf mysql-snapshot.tar.gz /mnt/snapshot
umount /mnt/snapshot
lvremove /dev/vg0/mysql-snapshot

# AWS RDS snapshot
aws rds create-db-snapshot \
  --db-instance-identifier mydb \
  --db-snapshot-identifier mydb-snapshot-$(date +%Y%m%d)
```

### Automated Backup Script

**Bash Backup Script:**
```bash
#!/bin/bash
# mysql-backup.sh

# Configuration
MYSQL_USER="root"
MYSQL_PASSWORD="password"
BACKUP_DIR="/backup/mysql"
RETENTION_DAYS=30
DATE=$(date +%Y%m%d_%H%M%S)
LOG_FILE="/var/log/mysql-backup.log"

# Create backup directory
mkdir -p "$BACKUP_DIR"

# Backup function
backup_database() {
    local db=$1
    local backup_file="${BACKUP_DIR}/${db}_${DATE}.sql.gz"

    echo "$(date): Backing up database: $db" >> "$LOG_FILE"

    mysqldump -u "$MYSQL_USER" -p"$MYSQL_PASSWORD" \
        --single-transaction \
        --routines \
        --triggers \
        --databases "$db" | gzip > "$backup_file"

    if [ $? -eq 0 ]; then
        echo "$(date): Backup successful: $backup_file" >> "$LOG_FILE"

        # Upload to S3 (optional)
        # aws s3 cp "$backup_file" s3://my-backups/mysql/

        # Delete old backups
        find "$BACKUP_DIR" -name "${db}_*.sql.gz" -mtime +$RETENTION_DAYS -delete
    else
        echo "$(date): Backup failed: $db" >> "$LOG_FILE"
        exit 1
    fi
}

# Get list of databases
DATABASES=$(mysql -u "$MYSQL_USER" -p"$MYSQL_PASSWORD" -e "SHOW DATABASES;" | grep -Ev "(Database|information_schema|performance_schema|mysql|sys)")

# Backup each database
for db in $DATABASES; do
    backup_database "$db"
done

echo "$(date): All backups completed" >> "$LOG_FILE"
```

**Cron Job:**
```cron
# Daily backup at 2 AM
0 2 * * * /usr/local/bin/mysql-backup.sh

# Weekly full backup on Sunday at 1 AM
0 1 * * 0 /usr/local/bin/mysql-full-backup.sh
```

### Recovery Procedures

**Full Database Restore:**
```bash
# Stop application
systemctl stop myapp

# Drop database
mysql -u root -p -e "DROP DATABASE myapp"

# Create database
mysql -u root -p -e "CREATE DATABASE myapp CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"

# Restore from backup
gunzip < myapp_backup.sql.gz | mysql -u root -p myapp

# Verify restore
mysql -u root -p -e "USE myapp; SHOW TABLES;"

# Start application
systemctl start myapp
```

**Table-Level Restore:**
```bash
# Export single table from backup
mysql -u root -p myapp < users_backup.sql

# Or extract from full backup
gunzip < full_backup.sql.gz | grep -A 10000 "CREATE TABLE users" | mysql -u root -p myapp
```

**Point-in-Time Recovery:**
```bash
# 1. Restore full backup
mysql -u root -p myapp < full_backup.sql

# 2. Apply binary logs up to specific point
mysqlbinlog --start-datetime="2024-02-01 00:00:00" \
            --stop-datetime="2024-02-01 09:59:59" \
            /var/log/mysql/mysql-bin.* | mysql -u root -p myapp

# 3. Verify data
mysql -u root -p myapp -e "SELECT COUNT(*) FROM users"
```

---

## 14. Migration Strategies

### Schema Migration Tools

**1. Flyway (Java-based):**
```sql
-- V1__initial_schema.sql
CREATE TABLE users (
    id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    email VARCHAR(255) NOT NULL UNIQUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- V2__add_users_name.sql
ALTER TABLE users ADD COLUMN name VARCHAR(255) NOT NULL DEFAULT '';

-- V3__add_users_index.sql
CREATE INDEX idx_users_email ON users(email);
```

**2. Liquibase:**
```xml
<!-- changelog.xml -->
<databaseChangeLog>
    <changeSet id="1" author="developer">
        <createTable tableName="users">
            <column name="id" type="BIGINT" autoIncrement="true">
                <constraints primaryKey="true"/>
            </column>
            <column name="email" type="VARCHAR(255)">
                <constraints nullable="false" unique="true"/>
            </column>
        </createTable>
    </changeSet>
</databaseChangeLog>
```

**3. Alembic (Python/SQLAlchemy):**
```python
# alembic/versions/001_initial_schema.py
def upgrade():
    op.create_table(
        'users',
        sa.Column('id', sa.BigInteger(), nullable=False),
        sa.Column('email', sa.String(255), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_users_email', 'users', ['email'], unique=True)

def downgrade():
    op.drop_index('idx_users_email', 'users')
    op.drop_table('users')
```

**4. Node.js (Knex.js):**
```javascript
// migrations/20240201000000_create_users.js
exports.up = function(knex) {
  return knex.schema.createTable('users', function(table) {
    table.bigIncrements('id').primary();
    table.string('email', 255).notNullable().unique();
    table.timestamp('created_at').defaultTo(knex.fn.now());
  });
};

exports.down = function(knex) {
  return knex.schema.dropTable('users');
};
```

### Online Schema Changes

**pt-online-schema-change (Percona Toolkit):**
```bash
# Add column without locking table
pt-online-schema-change \
  --alter "ADD COLUMN phone VARCHAR(20)" \
  --execute \
  D=myapp,t=users

# Add index without locking
pt-online-schema-change \
  --alter "ADD INDEX idx_email (email)" \
  --execute \
  D=myapp,t=users

# Change column type
pt-online-schema-change \
  --alter "MODIFY COLUMN status VARCHAR(50)" \
  --execute \
  D=myapp,t=users
```

**gh-ost (GitHub's Online Schema Migration):**
```bash
# Install gh-ost
wget https://github.com/github/gh-ost/releases/download/v1.1.5/gh-ost-binary-linux-amd64-20230405092837.tar.gz
tar -xzf gh-ost-binary-linux-amd64-20230405092837.tar.gz

# Run migration
gh-ost \
  --user=root \
  --password=password \
  --host=localhost \
  --database=myapp \
  --table=users \
  --alter="ADD COLUMN last_login TIMESTAMP NULL" \
  --execute \
  --allow-on-master \
  --initially-drop-ghost-table \
  --initially-drop-old-table
```

**MySQL 8.0 Instant DDL:**
```sql
-- Some ALTER operations are instant in MySQL 8.0+
ALTER TABLE users ADD COLUMN phone VARCHAR(20), ALGORITHM=INSTANT;

-- Check if operation can be instant
ALTER TABLE users ADD COLUMN phone VARCHAR(20), ALGORITHM=INSTANT, LOCK=NONE;

-- Supported instant operations (MySQL 8.0.29+):
-- - ADD COLUMN (at the end)
-- - DROP COLUMN
-- - RENAME COLUMN
-- - Modify column default value
```

### Data Migration

**Bulk Data Import:**
```sql
-- Load data from CSV
LOAD DATA INFILE '/tmp/users.csv'
INTO TABLE users
FIELDS TERMINATED BY ','
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS
(email, name, created_at);

-- Load data with LOCAL (client-side file)
LOAD DATA LOCAL INFILE '/path/to/users.csv'
INTO TABLE users
FIELDS TERMINATED BY ','
LINES TERMINATED BY '\n';
```

**ETL Migration Script (Python):**
```python
import mysql.connector
import csv

source_conn = mysql.connector.connect(
    host='old-db.example.com',
    database='legacy_db',
    user='root',
    password='password'
)

target_conn = mysql.connector.connect(
    host='new-db.example.com',
    database='new_db',
    user='root',
    password='password'
)

# Extract
source_cursor = source_conn.cursor(dictionary=True)
source_cursor.execute("SELECT * FROM old_users")

# Transform & Load
target_cursor = target_conn.cursor()
batch_size = 1000
batch = []

for row in source_cursor:
    # Transform data
    transformed = {
        'email': row['user_email'].lower(),
        'name': f"{row['first_name']} {row['last_name']}",
        'created_at': row['registration_date']
    }

    batch.append((transformed['email'], transformed['name'], transformed['created_at']))

    if len(batch) >= batch_size:
        # Batch insert
        target_cursor.executemany(
            "INSERT INTO users (email, name, created_at) VALUES (%s, %s, %s)",
            batch
        )
        target_conn.commit()
        batch = []

# Insert remaining
if batch:
    target_cursor.executemany(
        "INSERT INTO users (email, name, created_at) VALUES (%s, %s, %s)",
        batch
    )
    target_conn.commit()

source_conn.close()
target_conn.close()
```

### Database Migration (MySQL to MariaDB / Vice Versa)

**MySQL → MariaDB:**
```bash
# 1. Backup MySQL database
mysqldump -u root -p --all-databases --single-transaction > mysql_backup.sql

# 2. Install MariaDB
apt-get install mariadb-server

# 3. Restore to MariaDB
mysql -u root -p < mysql_backup.sql

# 4. Upgrade system tables
mysql_upgrade -u root -p
```

**MariaDB → MySQL:**
```bash
# Similar process, but check compatibility
# MariaDB-specific features may not work in MySQL:
# - ColumnStore engine
# - Galera cluster
# - Some JSON functions
```

---

## 15. Schema Design

### Normalization

**Third Normal Form (3NF) - Recommended:**
```sql
-- Users table
CREATE TABLE users (
    id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    email VARCHAR(255) NOT NULL UNIQUE,
    name VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_email (email)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Addresses table (1:many)
CREATE TABLE addresses (
    id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    user_id BIGINT UNSIGNED NOT NULL,
    street VARCHAR(255),
    city VARCHAR(100),
    country VARCHAR(2),
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    INDEX idx_user_id (user_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Orders table
CREATE TABLE orders (
    id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    user_id BIGINT UNSIGNED NOT NULL,
    total DECIMAL(10,2) NOT NULL,
    status ENUM('pending', 'processing', 'completed', 'cancelled'),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id),
    INDEX idx_user_id (user_id),
    INDEX idx_status_created (status, created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Order items (many:many)
CREATE TABLE order_items (
    id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    order_id BIGINT UNSIGNED NOT NULL,
    product_id BIGINT UNSIGNED NOT NULL,
    quantity INT UNSIGNED NOT NULL,
    price DECIMAL(10,2) NOT NULL,
    FOREIGN KEY (order_id) REFERENCES orders(id) ON DELETE CASCADE,
    FOREIGN KEY (product_id) REFERENCES products(id),
    INDEX idx_order_id (order_id),
    INDEX idx_product_id (product_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
```

### Denormalization for Performance

**When to Denormalize:**
```sql
-- Store computed/aggregated values for fast reads
CREATE TABLE user_stats (
    user_id BIGINT UNSIGNED PRIMARY KEY,
    total_orders INT UNSIGNED DEFAULT 0,
    total_spent DECIMAL(10,2) DEFAULT 0,
    last_order_at TIMESTAMP NULL,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
) ENGINE=InnoDB;

-- Update with triggers
DELIMITER //
CREATE TRIGGER update_user_stats_after_order
AFTER INSERT ON orders
FOR EACH ROW
BEGIN
    INSERT INTO user_stats (user_id, total_orders, total_spent, last_order_at)
    VALUES (NEW.user_id, 1, NEW.total, NEW.created_at)
    ON DUPLICATE KEY UPDATE
        total_orders = total_orders + 1,
        total_spent = total_spent + NEW.total,
        last_order_at = NEW.created_at;
END//
DELIMITER ;
```

### Data Types Best Practices

**Choose Appropriate Types:**
```sql
CREATE TABLE examples (
    -- IDs: BIGINT UNSIGNED (8 bytes, up to 18 quintillion)
    id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,

    -- Small integers: TINYINT (1 byte, -128 to 127 or 0-255 unsigned)
    age TINYINT UNSIGNED,

    -- Medium integers: INT (4 bytes, ±2 billion)
    view_count INT UNSIGNED DEFAULT 0,

    -- Booleans: TINYINT(1) or BOOLEAN (alias)
    is_active BOOLEAN DEFAULT TRUE,

    -- Money: DECIMAL(10,2) - avoid FLOAT/DOUBLE for currency
    price DECIMAL(10,2) NOT NULL,

    -- Strings: VARCHAR with appropriate length
    email VARCHAR(255) NOT NULL,        -- Max email length
    name VARCHAR(100),                  -- Typical name length
    short_code VARCHAR(10),             -- Fixed-ish length

    -- Text: Use TEXT for large content
    description TEXT,                   -- Up to 64KB
    content MEDIUMTEXT,                 -- Up to 16MB

    -- Dates/Times
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    birth_date DATE,

    -- JSON (MySQL 5.7.8+)
    metadata JSON,

    -- ENUM for fixed values
    status ENUM('draft', 'published', 'archived') DEFAULT 'draft',

    -- Binary data
    profile_picture BLOB
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
```

### UUID vs Auto-Increment

**Auto-Increment (Recommended for most cases):**
```sql
CREATE TABLE users (
    id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    email VARCHAR(255)
) ENGINE=InnoDB;

-- Pros: Sequential, compact, fast inserts, better indexing
-- Cons: Predictable, exposes record count
```

**UUID (Use for distributed systems):**
```sql
CREATE TABLE users (
    id BINARY(16) PRIMARY KEY,
    email VARCHAR(255)
) ENGINE=InnoDB;

-- Insert UUID
INSERT INTO users (id, email) VALUES
(UNHEX(REPLACE(UUID(), '-', '')), 'user@example.com');

-- Query UUID
SELECT HEX(id) as uuid, email FROM users;

-- MySQL 8.0: UUID_TO_BIN / BIN_TO_UUID
INSERT INTO users (id, email) VALUES
(UUID_TO_BIN(UUID()), 'user@example.com');

SELECT BIN_TO_UUID(id) as uuid, email FROM users;

-- Pros: Globally unique, no coordination needed, secure
-- Cons: Slower inserts, larger indexes, random order (bad for clustering)
```

**Ordered UUID (Best of both):**
```sql
-- MySQL 8.0: Time-ordered UUID
INSERT INTO users (id, email) VALUES
(UUID_TO_BIN(UUID(), 1), 'user@example.com');  -- 1 = swap time fields

-- Pros: Unique + sequential benefits
```

### Soft Deletes

**Soft Delete Pattern:**
```sql
CREATE TABLE users (
    id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    email VARCHAR(255) NOT NULL,
    deleted_at TIMESTAMP NULL,
    INDEX idx_deleted_at (deleted_at)
) ENGINE=InnoDB;

-- Soft delete
UPDATE users SET deleted_at = CURRENT_TIMESTAMP WHERE id = 1;

-- Query active users
SELECT * FROM users WHERE deleted_at IS NULL;

-- Include deleted users
SELECT * FROM users;

-- Permanent delete (after retention period)
DELETE FROM users WHERE deleted_at < DATE_SUB(NOW(), INTERVAL 90 DAY);
```

---

## 16. Monitoring and Troubleshooting

### Key Metrics to Monitor

**1. Query Performance:**
```sql
-- Enable performance schema
UPDATE performance_schema.setup_instruments
SET ENABLED = 'YES', TIMED = 'YES';

-- Top queries by execution time
SELECT
    DIGEST_TEXT as query,
    COUNT_STAR as exec_count,
    AVG_TIMER_WAIT/1000000000 as avg_time_ms,
    SUM_TIMER_WAIT/1000000000 as total_time_ms
FROM performance_schema.events_statements_summary_by_digest
ORDER BY total_time_ms DESC
LIMIT 10;

-- Queries currently running
SELECT
    id,
    user,
    host,
    db,
    command,
    time,
    state,
    info
FROM information_schema.processlist
WHERE command != 'Sleep'
ORDER BY time DESC;
```

**2. Connection Statistics:**
```sql
SHOW STATUS LIKE 'Threads%';
-- Threads_connected: Current connections
-- Threads_running: Active threads
-- Threads_created: Total threads created

SHOW STATUS LIKE 'Max_used_connections';
SHOW VARIABLES LIKE 'max_connections';

-- Connection usage percentage
SELECT
    (SELECT VARIABLE_VALUE FROM performance_schema.global_status
     WHERE VARIABLE_NAME='Threads_connected') /
    (SELECT VARIABLE_VALUE FROM performance_schema.global_variables
     WHERE VARIABLE_NAME='max_connections') * 100
AS connection_usage_pct;
```

**3. InnoDB Metrics:**
```sql
SHOW ENGINE INNODB STATUS\G

-- Buffer pool hit ratio (should be >99%)
SHOW STATUS LIKE 'Innodb_buffer_pool%';

SELECT
    (1 - (Innodb_buffer_pool_reads / Innodb_buffer_pool_read_requests)) * 100
    AS buffer_pool_hit_ratio;

-- InnoDB row operations
SHOW STATUS LIKE 'Innodb_rows%';
```

**4. Replication Lag:**
```sql
-- On slave
SHOW SLAVE STATUS\G

-- Key fields:
-- Seconds_Behind_Master: Replication lag in seconds
-- Slave_IO_Running: Yes/No
-- Slave_SQL_Running: Yes/No
-- Last_Error: Error message if any
```

### Monitoring Tools

**1. Percona Monitoring and Management (PMM):**
```bash
# Install PMM Server (Docker)
docker run -d \
  -p 443:443 \
  -v pmm-data:/srv \
  --name pmm-server \
  percona/pmm-server:latest

# Install PMM Client
wget https://www.percona.com/downloads/pmm2/2.41.0/binary/debian/bullseye/x86_64/pmm2-client_2.41.0-1.bullseye_amd64.deb
dpkg -i pmm2-client_2.41.0-1.bullseye_amd64.deb

# Add MySQL to monitoring
pmm-admin add mysql \
  --username=pmm \
  --password=password \
  --query-source=perfschema
```

**2. Prometheus + mysqld_exporter:**
```yaml
# docker-compose.yml
version: '3.8'
services:
  mysqld-exporter:
    image: prom/mysqld-exporter
    environment:
      - DATA_SOURCE_NAME=exporter:password@(mysql:3306)/
    ports:
      - 9104:9104

  prometheus:
    image: prom/prometheus
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    ports:
      - 9090:9090

  grafana:
    image: grafana/grafana
    ports:
      - 3000:3000
```

**3. MySQL Enterprise Monitor:**
```bash
# Commercial solution from Oracle
# Features: Query analyzer, replication monitoring, alerts
```

**4. pt-query-digest (Percona Toolkit):**
```bash
# Analyze slow query log
pt-query-digest /var/log/mysql/slow-query.log

# Analyze from processlist
pt-query-digest --processlist h=localhost,u=root,p=password

# Analyze binary logs
mysqlbinlog mysql-bin.000001 | pt-query-digest --type binlog
```

### Common Issues and Solutions

**1. High CPU Usage:**
```sql
-- Find expensive queries
SELECT * FROM sys.statements_with_runtimes_in_95th_percentile;

-- Kill long-running query
KILL <thread_id>;

-- Optimize queries (add indexes, rewrite)
EXPLAIN SELECT ...;
```

**2. Disk Space Issues:**
```sql
-- Find largest tables
SELECT
    table_schema,
    table_name,
    ROUND(((data_length + index_length) / 1024 / 1024), 2) AS size_mb
FROM information_schema.tables
WHERE table_schema NOT IN ('information_schema', 'mysql', 'performance_schema', 'sys')
ORDER BY size_mb DESC
LIMIT 20;

-- Purge binary logs
PURGE BINARY LOGS BEFORE DATE_SUB(NOW(), INTERVAL 7 DAY);

-- Optimize tables (reclaim space)
OPTIMIZE TABLE large_table;
```

**3. Replication Issues:**
```sql
-- Check replication status
SHOW SLAVE STATUS\G

-- Skip replication error (use carefully!)
STOP SLAVE;
SET GLOBAL sql_slave_skip_counter = 1;
START SLAVE;

-- Reset replication
STOP SLAVE;
RESET SLAVE ALL;
-- Reconfigure replication from scratch
```

**4. Deadlocks:**
```sql
-- View recent deadlock
SHOW ENGINE INNODB STATUS\G

-- Find deadlock-prone queries
SELECT * FROM sys.innodb_lock_waits;

-- Solutions:
-- 1. Access tables in same order
-- 2. Keep transactions short
-- 3. Use appropriate isolation level
-- 4. Add indexes to reduce lock range
```

**5. Connection Limit Reached:**
```sql
-- Check current connections
SHOW STATUS LIKE 'Threads_connected';
SHOW VARIABLES LIKE 'max_connections';

-- Increase max connections (temporarily)
SET GLOBAL max_connections = 500;

-- Make permanent (my.cnf)
[mysqld]
max_connections = 500

-- Kill idle connections
SELECT CONCAT('KILL ', id, ';')
FROM information_schema.processlist
WHERE command = 'Sleep'
  AND time > 300;  -- Idle for 5+ minutes
```

---

## 17. Container Deployment

### Docker Deployment

**Official MySQL Docker Image:**
```bash
# Pull image
docker pull mysql:8.4

# Run MySQL container
docker run -d \
  --name mysql-server \
  -e MYSQL_ROOT_PASSWORD=root_password \
  -e MYSQL_DATABASE=myapp \
  -e MYSQL_USER=appuser \
  -e MYSQL_PASSWORD=apppassword \
  -p 3306:3306 \
  -v mysql-data:/var/lib/mysql \
  mysql:8.4 \
  --character-set-server=utf8mb4 \
  --collation-server=utf8mb4_unicode_ci

# MariaDB
docker pull mariadb:11.4
docker run -d \
  --name mariadb-server \
  -e MARIADB_ROOT_PASSWORD=root_password \
  -e MARIADB_DATABASE=myapp \
  -e MARIADB_USER=appuser \
  -e MARIADB_PASSWORD=apppassword \
  -p 3306:3306 \
  -v mariadb-data:/var/lib/mysql \
  mariadb:11.4
```

**Docker Compose (Production-Ready):**
```yaml
version: '3.8'

services:
  mysql:
    image: mysql:8.4
    container_name: mysql-server
    restart: unless-stopped

    environment:
      MYSQL_ROOT_PASSWORD: ${MYSQL_ROOT_PASSWORD}
      MYSQL_DATABASE: ${MYSQL_DATABASE}
      MYSQL_USER: ${MYSQL_USER}
      MYSQL_PASSWORD: ${MYSQL_PASSWORD}

    ports:
      - "3306:3306"

    volumes:
      - mysql-data:/var/lib/mysql
      - ./my.cnf:/etc/mysql/conf.d/my.cnf:ro
      - ./init.sql:/docker-entrypoint-initdb.d/init.sql:ro

    command: >
      --character-set-server=utf8mb4
      --collation-server=utf8mb4_unicode_ci
      --default-authentication-plugin=mysql_native_password
      --max-connections=500
      --innodb-buffer-pool-size=1G

    healthcheck:
      test: ["CMD", "mysqladmin", "ping", "-h", "localhost", "-u", "root", "-p$$MYSQL_ROOT_PASSWORD"]
      interval: 10s
      timeout: 5s
      retries: 5

    networks:
      - app-network

  phpmyadmin:
    image: phpmyadmin/phpmyadmin
    container_name: phpmyadmin
    restart: unless-stopped
    environment:
      PMA_HOST: mysql
      PMA_PORT: 3306
    ports:
      - "8080:80"
    depends_on:
      - mysql
    networks:
      - app-network

volumes:
  mysql-data:
    driver: local

networks:
  app-network:
    driver: bridge
```

**Custom my.cnf for Docker:**
```ini
# my.cnf
[mysqld]
# InnoDB settings
innodb_buffer_pool_size = 1G
innodb_log_file_size = 256M
innodb_flush_method = O_DIRECT

# Connection settings
max_connections = 500
wait_timeout = 600

# Binary logging
log_bin = /var/lib/mysql/mysql-bin
binlog_expire_logs_seconds = 604800
binlog_format = ROW

# Slow query log
slow_query_log = 1
slow_query_log_file = /var/lib/mysql/slow-query.log
long_query_time = 2
```

### Kubernetes Deployment

**MySQL StatefulSet:**
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: mysql-config
data:
  my.cnf: |
    [mysqld]
    bind-address = 0.0.0.0
    default-authentication-plugin = mysql_native_password
    max_connections = 500
    innodb_buffer_pool_size = 2G
    character-set-server = utf8mb4
    collation-server = utf8mb4_unicode_ci

---
apiVersion: v1
kind: Secret
metadata:
  name: mysql-secret
type: Opaque
stringData:
  root-password: "your_root_password"
  user: "appuser"
  password: "apppassword"
  database: "myapp"

---
apiVersion: v1
kind: Service
metadata:
  name: mysql
spec:
  ports:
  - port: 3306
    name: mysql
  clusterIP: None  # Headless service
  selector:
    app: mysql

---
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: mysql
spec:
  serviceName: mysql
  replicas: 1
  selector:
    matchLabels:
      app: mysql
  template:
    metadata:
      labels:
        app: mysql
    spec:
      containers:
      - name: mysql
        image: mysql:8.4
        ports:
        - containerPort: 3306
          name: mysql

        env:
        - name: MYSQL_ROOT_PASSWORD
          valueFrom:
            secretKeyRef:
              name: mysql-secret
              key: root-password
        - name: MYSQL_DATABASE
          valueFrom:
            secretKeyRef:
              name: mysql-secret
              key: database
        - name: MYSQL_USER
          valueFrom:
            secretKeyRef:
              name: mysql-secret
              key: user
        - name: MYSQL_PASSWORD
          valueFrom:
            secretKeyRef:
              name: mysql-secret
              key: password

        volumeMounts:
        - name: mysql-storage
          mountPath: /var/lib/mysql
        - name: mysql-config
          mountPath: /etc/mysql/conf.d

        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"

        livenessProbe:
          exec:
            command:
            - mysqladmin
            - ping
            - -h
            - localhost
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5

        readinessProbe:
          exec:
            command:
            - mysql
            - -h
            - localhost
            - -u
            - root
            - -p$(MYSQL_ROOT_PASSWORD)
            - -e
            - SELECT 1
          initialDelaySeconds: 10
          periodSeconds: 5

      volumes:
      - name: mysql-config
        configMap:
          name: mysql-config

  volumeClaimTemplates:
  - metadata:
      name: mysql-storage
    spec:
      accessModes: [ "ReadWriteOnce" ]
      storageClassName: fast-ssd
      resources:
        requests:
          storage: 100Gi
```

**MySQL Operator (Recommended for Production):**
```yaml
# Using Percona Operator for MySQL
apiVersion: pxc.percona.com/v1
kind: PerconaXtraDBCluster
metadata:
  name: my-cluster
spec:
  crVersion: 1.13.0
  secretsName: my-cluster-secrets

  pxc:
    size: 3
    image: percona/percona-xtradb-cluster:8.0.33
    resources:
      requests:
        memory: 4Gi
        cpu: 2
      limits:
        memory: 8Gi
        cpu: 4
    volumeSpec:
      persistentVolumeClaim:
        storageClassName: fast-ssd
        accessModes: [ "ReadWriteOnce" ]
        resources:
          requests:
            storage: 100Gi

  haproxy:
    enabled: true
    size: 3
    image: percona/percona-xtradb-cluster-operator:1.13.0-haproxy

  proxysql:
    enabled: true
    size: 3
    image: percona/percona-xtradb-cluster-operator:1.13.0-proxysql

  backup:
    enabled: true
    schedule:
      - name: daily-backup
        schedule: "0 2 * * *"
        keep: 7
        storageName: s3-storage
```

### Cloud Deployments

**AWS RDS MySQL:**
```bash
# Create RDS instance via AWS CLI
aws rds create-db-instance \
  --db-instance-identifier myapp-mysql \
  --db-instance-class db.r6g.xlarge \
  --engine mysql \
  --engine-version 8.0.35 \
  --master-username admin \
  --master-user-password MyPassword123 \
  --allocated-storage 100 \
  --storage-type gp3 \
  --backup-retention-period 7 \
  --preferred-backup-window "03:00-04:00" \
  --multi-az \
  --vpc-security-group-ids sg-12345678 \
  --db-subnet-group-name my-subnet-group \
  --publicly-accessible false
```

**Google Cloud SQL:**
```bash
# Create Cloud SQL instance
gcloud sql instances create myapp-mysql \
  --database-version=MYSQL_8_0 \
  --tier=db-n1-highmem-4 \
  --region=us-central1 \
  --storage-size=100GB \
  --storage-type=SSD \
  --storage-auto-increase \
  --backup-start-time=03:00 \
  --enable-bin-log \
  --availability-type=REGIONAL
```

**Azure Database for MySQL:**
```bash
# Create Azure MySQL server
az mysql flexible-server create \
  --resource-group myResourceGroup \
  --name myapp-mysql \
  --location eastus \
  --admin-user admin \
  --admin-password MyPassword123 \
  --sku-name Standard_D4ds_v4 \
  --tier GeneralPurpose \
  --storage-size 100 \
  --version 8.0.21 \
  --high-availability Enabled \
  --zone 1 \
  --standby-zone 2
```

---

## 18. MySQL vs MariaDB Differences

### Feature Comparison

| Feature | MySQL 8.4/9.0 | MariaDB 11.4 |
|---------|---------------|--------------|
| **Storage Engines** | InnoDB, MyISAM | InnoDB, Aria, ColumnStore, MyRocks |
| **Default Engine** | InnoDB | InnoDB |
| **Clustering** | Group Replication | Galera Cluster (built-in) |
| **Query Cache** | Removed (8.0+) | Removed (10.6+) |
| **Window Functions** | ✅ Yes (8.0+) | ✅ Yes (10.2+) |
| **CTEs** | ✅ Yes (8.0+) | ✅ Yes (10.2+) |
| **JSON** | ✅ Native type | ⚠️ TEXT alias (longtext) |
| **Oracle Compatibility** | Some | More features (PL/SQL, sequences) |
| **Thread Pool** | Enterprise only | ✅ Built-in |
| **Max Connections** | 151 default | 151 default |
| **License** | GPL + Commercial | GPL (no dual licensing) |

### MariaDB-Specific Features

**1. Galera Cluster (Built-in):**
```sql
-- Already covered in section 4 (Replication)
-- Multi-master synchronous replication
wsrep_on = ON
```

**2. ColumnStore (Analytics Engine):**
```sql
CREATE TABLE analytics_data (
    event_date DATE,
    user_id BIGINT,
    event_type VARCHAR(50),
    value DECIMAL(10,2)
) ENGINE=ColumnStore;

-- Optimized for OLAP queries
SELECT
    event_type,
    DATE_FORMAT(event_date, '%Y-%m') as month,
    SUM(value) as total_value
FROM analytics_data
GROUP BY event_type, month;
```

**3. Sequences (Oracle-compatible):**
```sql
-- Create sequence
CREATE SEQUENCE order_seq START WITH 1000 INCREMENT BY 1;

-- Use sequence
INSERT INTO orders (id, total) VALUES (NEXT VALUE FOR order_seq, 100.00);

-- Get current value
SELECT PREVIOUS VALUE FOR order_seq;
```

**4. Temporal Tables (System-Versioned):**
```sql
CREATE TABLE users (
    id BIGINT PRIMARY KEY,
    name VARCHAR(255),
    email VARCHAR(255)
) WITH SYSTEM VERSIONING;

-- Automatic history tracking
UPDATE users SET name = 'New Name' WHERE id = 1;

-- Query historical data
SELECT * FROM users FOR SYSTEM_TIME AS OF TIMESTAMP '2024-01-01 00:00:00';

-- See all versions
SELECT * FROM users FOR SYSTEM_TIME ALL WHERE id = 1;
```

**5. Invisible Columns:**
```sql
CREATE TABLE users (
    id BIGINT PRIMARY KEY,
    email VARCHAR(255),
    internal_notes TEXT INVISIBLE  -- Not in SELECT *
);

-- Internal column not returned
SELECT * FROM users;

-- Explicitly select invisible column
SELECT id, email, internal_notes FROM users;
```

### MySQL-Specific Features

**1. Document Store (X DevAPI):**
```python
import mysqlx

session = mysqlx.get_session('mysqlx://user:password@localhost:33060')
schema = session.get_schema('myapp')

# Create collection
users = schema.create_collection('users')

# Insert JSON documents
users.add({
    'name': 'John Doe',
    'email': 'john@example.com',
    'age': 30
}).execute()

# Query documents
docs = users.find('age > 25').execute()
```

**2. Binary Log Transaction Compression (MySQL 8.0.20+):**
```sql
SET GLOBAL binlog_transaction_compression = ON;
SET GLOBAL binlog_transaction_compression_level_zstd = 3;
```

**3. Multi-Valued Indexes (MySQL 8.0.17+):**
```sql
CREATE TABLE products (
    id BIGINT PRIMARY KEY,
    name VARCHAR(255),
    tags JSON,
    INDEX idx_tags ((CAST(tags->'$[*]' AS CHAR(50) ARRAY)))
);

-- Query using multi-valued index
SELECT * FROM products
WHERE JSON_CONTAINS(tags, '"electronics"');
```

### Migration Considerations

**MySQL → MariaDB:**
- ✅ Generally seamless
- ⚠️ Test JSON workloads (different implementation)
- ✅ Gain Galera, thread pool, ColumnStore

**MariaDB → MySQL:**
- ⚠️ Test Galera migration (use Group Replication)
- ⚠️ Remove MariaDB-specific features (sequences, temporal tables)
- ⚠️ Thread pool requires MySQL Enterprise

---

## 19. Version-Specific Features

### MySQL 9.0 (Released 2025)

**JavaScript Stored Programs:**
```sql
-- JavaScript stored function (new in 9.0)
CREATE FUNCTION calculate_discount(price DECIMAL(10,2), rate DECIMAL(3,2))
RETURNS DECIMAL(10,2)
LANGUAGE JAVASCRIPT
AS $$
  return price * (1 - rate);
$$;

SELECT calculate_discount(100.00, 0.15);  -- Returns 85.00
```

### MySQL 8.4 LTS (Released 2024)

**Vector Search:**
```sql
-- Store embeddings
CREATE TABLE documents (
    id BIGINT PRIMARY KEY,
    content TEXT,
    embedding VECTOR(1536)  -- OpenAI embeddings
);

-- Similarity search
SELECT id, content,
       COSINE_SIMILARITY(embedding, :query_embedding) as similarity
FROM documents
ORDER BY similarity DESC
LIMIT 10;
```

### MySQL 8.0 Features

**Window Functions (8.0+):**
```sql
-- Running total
SELECT
    order_date,
    amount,
    SUM(amount) OVER (ORDER BY order_date) as running_total
FROM orders;

-- Ranking
SELECT
    name,
    department,
    salary,
    RANK() OVER (PARTITION BY department ORDER BY salary DESC) as dept_rank
FROM employees;
```

**CTEs (Common Table Expressions):**
```sql
WITH RECURSIVE employee_hierarchy AS (
    -- Base case
    SELECT id, name, manager_id, 1 as level
    FROM employees
    WHERE manager_id IS NULL

    UNION ALL

    -- Recursive case
    SELECT e.id, e.name, e.manager_id, eh.level + 1
    FROM employees e
    INNER JOIN employee_hierarchy eh ON e.manager_id = eh.id
)
SELECT * FROM employee_hierarchy;
```

**Descending Indexes:**
```sql
CREATE INDEX idx_orders_date_desc ON orders (order_date DESC);
```

**Invisible Indexes:**
```sql
ALTER TABLE users ALTER INDEX idx_email INVISIBLE;
```

**Instant DDL:**
```sql
ALTER TABLE users ADD COLUMN phone VARCHAR(20), ALGORITHM=INSTANT;
```

### MariaDB 11.4 Features

**UUID Data Type:**
```sql
CREATE TABLE sessions (
    id UUID PRIMARY KEY DEFAULT UUID(),
    user_id BIGINT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

**InnoDB Page Compression:**
```sql
CREATE TABLE large_data (
    id BIGINT PRIMARY KEY,
    data TEXT
) PAGE_COMPRESSED=1 PAGE_COMPRESSION_LEVEL=9;
```

---

## 20. Performance Tuning Checklist

### Server Configuration Checklist

```ini
# Critical Settings (my.cnf)

[mysqld]
# InnoDB Buffer Pool (70-80% of RAM)
innodb_buffer_pool_size = 8G
innodb_buffer_pool_instances = 8

# InnoDB Logs
innodb_log_file_size = 1G             # MySQL 5.7
innodb_redo_log_capacity = 2G         # MySQL 8.0.30+
innodb_flush_log_at_trx_commit = 1    # 1=safest, 2=faster

# I/O Configuration
innodb_flush_method = O_DIRECT
innodb_io_capacity = 2000
innodb_io_capacity_max = 4000

# Connections
max_connections = 500
thread_cache_size = 100

# Binary Logging
log_bin = mysql-bin
binlog_format = ROW
sync_binlog = 1

# Character Set
character_set_server = utf8mb4
collation_server = utf8mb4_unicode_ci

# Query Cache (MySQL 5.7 only - disabled)
query_cache_type = 0

# Slow Query Log
slow_query_log = 1
long_query_time = 2
```

### Schema Optimization Checklist

- ✅ Use InnoDB for all tables
- ✅ Define primary key on every table
- ✅ Use BIGINT UNSIGNED for auto-increment IDs
- ✅ Use appropriate data types (avoid oversized VARCHAR)
- ✅ Use DECIMAL for currency (not FLOAT/DOUBLE)
- ✅ Use utf8mb4 character set
- ✅ Normalize to 3NF (denormalize strategically)
- ✅ Define foreign keys for referential integrity
- ✅ Use ENUM for fixed value sets

### Indexing Checklist

- ✅ Index all foreign keys
- ✅ Index columns in WHERE clauses
- ✅ Index columns in JOIN conditions
- ✅ Index columns in ORDER BY
- ✅ Create covering indexes for frequent queries
- ✅ Use prefix indexes for long VARCHAR/TEXT
- ✅ Drop unused indexes
- ✅ Run ANALYZE TABLE regularly

### Query Optimization Checklist

- ✅ Avoid SELECT *
- ✅ Use LIMIT for large result sets
- ✅ Use EXISTS instead of COUNT(*) for existence checks
- ✅ Avoid functions on indexed columns in WHERE
- ✅ Use UNION ALL instead of UNION when duplicates ok
- ✅ Use JOIN instead of subqueries when possible
- ✅ Batch INSERT/UPDATE in transactions
- ✅ Use prepared statements
- ✅ Add EXPLAIN to all slow queries

### Monitoring Checklist

- ✅ Monitor slow query log
- ✅ Track buffer pool hit ratio (>99%)
- ✅ Monitor replication lag
- ✅ Track connection usage
- ✅ Monitor disk space
- ✅ Set up alerts for errors
- ✅ Track query execution times
- ✅ Monitor InnoDB deadlocks

---

## 21. Production Deployment Patterns

### Single Server Pattern

**Use Case:** Small to medium applications, <10,000 req/min

```
┌──────────────┐
│  Application │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│    MySQL     │
│  (Standalone)│
└──────────────┘
```

**Configuration:**
- InnoDB buffer pool: 70-80% of RAM
- Regular backups (daily full, hourly incremental)
- Monitor disk space and performance

---

### Master-Slave Replication Pattern

**Use Case:** Read-heavy applications, high availability

```
┌──────────────┐
│  Application │
└───┬──────┬───┘
    │      │
Write│      │Read
    ▼      ▼
┌────────┐ ┌────────┐
│ Master │─►│ Slave1 │
│  (R/W) │ │  (R)   │
└────────┘ └────────┘
     │
     └─────►┌────────┐
            │ Slave2 │
            │  (R)   │
            └────────┘
```

**Application Code:**
```python
# Separate read/write connections
DATABASES = {
    'write': {
        'host': 'master.db.example.com',
        'database': 'myapp'
    },
    'read': {
        'hosts': [
            'slave1.db.example.com',
            'slave2.db.example.com'
        ],
        'database': 'myapp'
    }
}

def get_write_connection():
    return mysql.connector.connect(**DATABASES['write'])

def get_read_connection():
    host = random.choice(DATABASES['read']['hosts'])
    return mysql.connector.connect(host=host, database=DATABASES['read']['database'])
```

---

### Galera Cluster Pattern (MariaDB)

**Use Case:** Multi-master, high availability, write scaling

```
┌──────────────┐
│   HAProxy    │  (Load Balancer)
└──────┬───────┘
       │
  ┌────┴────┬────────┐
  ▼         ▼        ▼
┌────┐   ┌────┐   ┌────┐
│ N1 │◄─►│ N2 │◄─►│ N3 │
│(RW)│   │(RW)│   │(RW)│
└────┘   └────┘   └────┘
```

**Benefits:**
- True multi-master (write to any node)
- Automatic failover
- Zero data loss
- Read and write scaling

---

### Sharded Architecture

**Use Case:** Very large datasets, horizontal scaling

```
┌──────────────┐
│  Application │
│   (Router)   │
└──────┬───────┘
       │
  ┌────┼────┬────┐
  ▼    ▼    ▼    ▼
┌───┐┌───┐┌───┐┌───┐
│S0 ││S1 ││S2 ││S3 │
└───┘└───┘└───┘└───┘
```

**Shard by user_id:**
```python
def get_shard(user_id, num_shards=4):
    return user_id % num_shards
```

---

### Cloud Managed Service Pattern

**Use Case:** Simplified operations, auto-scaling, backups

```
┌──────────────┐
│  Application │
└──────┬───────┘
       │
       ▼
┌──────────────────────┐
│  AWS RDS / Cloud SQL │
│  (Managed MySQL)     │
│  - Auto backups      │
│  - Multi-AZ          │
│  - Read replicas     │
└──────────────────────┘
```

---

## 22. Deployment Checklist

### Build and Configuration
- [ ] MySQL/MariaDB version pinned and documented
- [ ] Character set and collation set to `utf8mb4` / `utf8mb4_0900_ai_ci`
- [ ] `innodb_buffer_pool_size` set to 70-80% of available RAM
- [ ] `innodb_log_file_size` sized for workload (1-2 GB typical)
- [ ] `max_connections` tuned for expected concurrency
- [ ] Slow query log enabled with appropriate threshold
- [ ] Binary logging enabled for replication and point-in-time recovery

### Testing
- [ ] Schema migrations tested with `pt-online-schema-change` or `gh-ost`
- [ ] All queries profiled with `EXPLAIN ANALYZE`
- [ ] Load testing completed with production-scale data
- [ ] Failover and replica promotion tested
- [ ] Backup and restore procedure verified end-to-end
- [ ] Connection pool sizing validated under peak load

### Security
- [ ] Root remote login disabled
- [ ] Application-specific users with least-privilege grants
- [ ] TLS/SSL enabled for all connections
- [ ] `validate_password` plugin enabled
- [ ] Audit logging configured (Enterprise or MariaDB Audit Plugin)
- [ ] `SUPER` privilege removed from application accounts
- [ ] Network access restricted via firewall rules

### Agent Workflow
- [ ] Schema change scripts reviewed for backward compatibility
- [ ] Migration rollback scripts prepared and tested
- [ ] Monitoring alerts configured (replication lag, slow queries, disk usage)
- [ ] Automated backups scheduled with retention policy
- [ ] Runbooks documented for common failure scenarios

---

## 23. Why This Configuration Works

**InnoDB Buffer Pool Optimization**:
- Caching frequently accessed data and indexes in memory eliminates disk I/O for the majority of read operations, providing consistent sub-millisecond query latency.

**Binary Log Replication**:
- Row-based replication with GTIDs ensures reliable data synchronization across replicas, enables point-in-time recovery, and supports zero-downtime failover.

**Online Schema Migrations**:
- Tools like `pt-online-schema-change` and `gh-ost` allow schema evolution on live tables without locking, enabling continuous delivery without maintenance windows.

**Query Optimizer and Indexing**:
- The cost-based optimizer combined with covering indexes, index condition pushdown, and hash joins (MySQL 8.0+) delivers efficient execution plans for complex analytical and transactional queries.

**Connection Pooling with ProxySQL**:
- Multiplexing application connections through ProxySQL reduces server-side resource consumption, enables query routing to replicas, and provides transparent failover handling.

---

## 24. Quick Reference

### Common Commands

```bash
# Connect to MySQL
mysql -u root -p -h localhost -P 3306

# Check server status
mysqladmin -u root -p status

# Show running queries
mysql -e "SHOW PROCESSLIST;"

# Kill a long-running query
mysql -e "KILL <thread_id>;"

# Check replication status
mysql -e "SHOW REPLICA STATUS\G"

# Analyze query performance
mysql -e "EXPLAIN ANALYZE SELECT * FROM my_table WHERE id = 1;"

# Logical backup
mysqldump --single-transaction --routines --triggers --all-databases > backup.sql

# Physical backup (Percona XtraBackup)
xtrabackup --backup --target-dir=/backup/full

# Online schema change
pt-online-schema-change --alter "ADD COLUMN new_col INT" D=mydb,t=mytable --execute

# Check table sizes
mysql -e "SELECT table_name, ROUND(data_length/1024/1024, 2) AS 'Data (MB)', ROUND(index_length/1024/1024, 2) AS 'Index (MB)' FROM information_schema.tables WHERE table_schema='mydb';"

# Check InnoDB status
mysql -e "SHOW ENGINE INNODB STATUS\G"
```

---

## References and Resources

### Official Documentation
- **MySQL:** https://dev.mysql.com/doc/
- **MariaDB:** https://mariadb.com/kb/en/
- **MySQL 8.4 Reference:** https://dev.mysql.com/doc/refman/8.4/en/
- **MariaDB 11.4 Docs:** https://mariadb.com/kb/en/mariadb-1140-release-notes/

### Tools
- **Percona Toolkit:** https://www.percona.com/software/database-tools/percona-toolkit
- **pt-online-schema-change:** Schema migrations without downtime
- **gh-ost:** GitHub's schema migration tool
- **mysqldump:** Logical backups
- **Percona XtraBackup:** Physical backups
- **ProxySQL:** Query routing and load balancing
- **Vitess:** Horizontal sharding platform

### Monitoring
- **Percona Monitoring and Management (PMM):** https://www.percona.com/software/database-tools/percona-monitoring-and-management
- **MySQL Enterprise Monitor:** https://www.mysql.com/products/enterprise/monitor.html
- **Prometheus + mysqld_exporter**
- **Grafana dashboards**

### Books
- "High Performance MySQL" by Baron Schwartz, Peter Zaitsev (O'Reilly)
- "MySQL Cookbook" by Paul DuBois (O'Reilly)
- "Effective MySQL" series by Ronald Bradford

### Community
- MySQL Forums: https://forums.mysql.com/
- MariaDB Community: https://mariadb.org/get-involved/
- Stack Overflow: `[mysql]` and `[mariadb]` tags
- Reddit: r/mysql

---

**Document Maintenance:**
- Review quarterly for version updates
- Update benchmarks and best practices
- Validate container deployment patterns
- Incorporate community feedback

**Last Updated:** February 2026
**Next Review:** May 2026

---

**End of MySQL & MariaDB Development Guidelines**
