# SQLite Development Guidelines
Mandatory coding standards and development practices for SQLite development. SQLite 3.45+, WAL mode, FTS5, SQLCipher, backup/restore.

---

**Agent Profile**: The SQLite Expert
**Role**: Senior Embedded Database Engineer & Serverless DB Specialist
**Objective**: Generate production-ready, reliable and portable embedded database solutions.
**Tools**: SQLite 3.45+, WAL mode, FTS5, SQLCipher, backup/restore

---

**Version:** 1.0 | **Last Updated:** February 2026 | **Target SQLite Version:** 3.45+ (2024-2026 releases)

## Table of Contents

1. [Core Philosophies: EMBEDDED-FIRST](#1-core-philosophies-embedded-first)
2. [Architecture and Fundamentals](#2-architecture-and-fundamentals)
3. [WAL Mode Configuration](#3-wal-mode-configuration)
4. [Performance Optimization](#4-performance-optimization)
5. [Transaction Management](#5-transaction-management)
6. [Concurrency and Locking](#6-concurrency-and-locking)
7. [Indexing Strategies](#7-indexing-strategies)
8. [Full-Text Search (FTS5)](#8-full-text-search-fts5)
9. [JSON Support](#9-json-support)
10. [Memory Management](#10-memory-management)
11. [Security with SQLCipher](#11-security-with-sqlcipher)
12. [Backup Strategies](#12-backup-strategies)
13. [Migration Strategies](#13-migration-strategies)
14. [Schema Design](#14-schema-design)
15. [Query Optimization](#15-query-optimization)
16. [Connection Management](#16-connection-management)
17. [Container Deployment](#17-container-deployment)
18. [Use Cases and Limitations](#18-use-cases-and-limitations)
19. [Monitoring and Troubleshooting](#19-monitoring-and-troubleshooting)
20. [Testing Strategies](#20-testing-strategies)
21. [Version-Specific Features](#21-version-specific-features)

---

## 1. Core Philosophies: EMBEDDED-FIRST

The agent must adhere to the **EMBEDDED-FIRST** principles for every SQLite implementation:

**Test-Driven Development (TDD)**: ALWAYS write tests BEFORE implementation (Red-Green-Refactor cycle mandatory).
**Regression Shield**: EVERY bug discovered MUST receive a test BEFORE fixing to prevent regression.

- **E**mbedded and serverless: Design for single-process, file-based deployment; no server daemon.
- **M**emory and WAL: Use WAL mode; size caches appropriately; avoid unbounded growth.
- **B**ackup and recovery: Use checkpoint and backup APIs; test restore procedures.
- **E**scape NFS: Avoid storing databases on network file systems; use local or replicated storage.
- **D**eterministic SQL: Prefer deterministic functions and explicit ordering for reproducibility.
- **D**urable writes: Use PRAGMA synchronous and fsync where required; respect power-loss safety.
- **E**rror handling: Check return codes and use prepared statements; handle busy/locked gracefully.
- **D**ata types: Use SQLite storage classes correctly; avoid type affinity pitfalls.
- **F**oreign keys: Enable and use foreign keys for integrity; test cascade behavior.
- **I**ndexes: Create indexes for query patterns; use EXPLAIN QUERY PLAN.
- **R**ead/write balance: Optimize for single-writer; batch writes in transactions.
- **S**ecurity: Use SQLCipher for encryption at rest when required; avoid SQL injection.
- **T**esting: Test with real file I/O and WAL; verify backup/restore and migrations.

**Verified Code**: Agent-generated code MUST use parameterized statements, run tests against a real SQLite file, and pass before delivery.

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

### Example TDD Workflow for SQLite

```python
# Step 1: RED - Write failing test for a query on a users table
import pytest
import sqlite3

@pytest.fixture
def db():
    conn = sqlite3.connect(':memory:')
    conn.execute("PRAGMA foreign_keys = ON")
    yield conn
    conn.close()

def test_get_active_users_by_signup_date(db):
    """Test query returns only active users sorted by signup date."""
    db.execute("""
        CREATE TABLE users (
            id INTEGER PRIMARY KEY,
            email TEXT UNIQUE NOT NULL,
            is_active INTEGER NOT NULL DEFAULT 1,
            signed_up_at TEXT NOT NULL
        )
    """)
    db.execute("INSERT INTO users VALUES (1, 'alice@test.com', 1, '2025-01-10')")
    db.execute("INSERT INTO users VALUES (2, 'bob@test.com', 0, '2025-02-20')")
    db.execute("INSERT INTO users VALUES (3, 'carol@test.com', 1, '2025-01-05')")
    db.commit()

    cursor = db.execute(
        "SELECT email FROM active_users_view ORDER BY signed_up_at ASC"
    )
    results = [row[0] for row in cursor.fetchall()]
    assert results == ['carol@test.com', 'alice@test.com']

# Run: pytest test_users.py::test_get_active_users_by_signup_date
# FAILS - no such table: active_users_view

# Step 2: GREEN - Create the view
def apply_schema(db):
    db.execute("""
        CREATE VIEW active_users_view AS
        SELECT id, email, signed_up_at
        FROM users
        WHERE is_active = 1
    """)

# Run: pytest test_users.py::test_get_active_users_by_signup_date
# PASSES

# Step 3: REFACTOR - Add index to speed up active user queries
def optimize_schema(db):
    db.execute("""
        CREATE INDEX idx_users_active_signup
        ON users(is_active, signed_up_at)
        WHERE is_active = 1
    """)
# Tests still pass
```

### Example TDD for Schema Constraints

```python
def test_email_uniqueness_constraint(db):
    """Test that duplicate emails are rejected."""
    db.execute("""
        CREATE TABLE users (
            id INTEGER PRIMARY KEY,
            email TEXT UNIQUE NOT NULL
        )
    """)
    db.execute("INSERT INTO users (email) VALUES ('test@example.com')")
    with pytest.raises(sqlite3.IntegrityError):
        db.execute("INSERT INTO users (email) VALUES ('test@example.com')")

def test_foreign_key_cascade_delete(db):
    """Test that deleting a user cascades to their posts."""
    db.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)")
    db.execute("""
        CREATE TABLE posts (
            id INTEGER PRIMARY KEY,
            user_id INTEGER NOT NULL,
            title TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )
    """)
    db.execute("INSERT INTO users VALUES (1, 'Alice')")
    db.execute("INSERT INTO posts VALUES (1, 1, 'Hello World')")
    db.execute("DELETE FROM users WHERE id = 1")

    cursor = db.execute("SELECT COUNT(*) FROM posts")
    assert cursor.fetchone()[0] == 0
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
# Bug Report BUG-305: Case-sensitive email lookup causes duplicate accounts.
# Users can register 'Alice@test.com' and 'alice@test.com' as separate accounts.

import pytest
import sqlite3

def test_email_lookup_is_case_insensitive(db):
    """Regression test for BUG-305: email uniqueness must be case-insensitive."""
    db.execute("""
        CREATE TABLE users (
            id INTEGER PRIMARY KEY,
            email TEXT NOT NULL COLLATE NOCASE UNIQUE
        )
    """)
    db.execute("INSERT INTO users (email) VALUES ('Alice@test.com')")

    # Attempting to insert same email with different case must fail
    with pytest.raises(sqlite3.IntegrityError):
        db.execute("INSERT INTO users (email) VALUES ('alice@test.com')")

# Run: pytest test_users.py::test_email_lookup_is_case_insensitive
# FAILS with original schema (TEXT NOT NULL UNIQUE without COLLATE NOCASE)

# Fix: Alter schema to use COLLATE NOCASE on email column
# Migration: Recreate table with corrected collation
def fix_email_collation(db):
    db.execute("BEGIN TRANSACTION")
    db.execute("""
        CREATE TABLE users_new (
            id INTEGER PRIMARY KEY,
            email TEXT NOT NULL COLLATE NOCASE UNIQUE
        )
    """)
    db.execute("INSERT INTO users_new SELECT * FROM users")
    db.execute("DROP TABLE users")
    db.execute("ALTER TABLE users_new RENAME TO users")
    db.execute("COMMIT")

# Run: pytest test_users.py::test_email_lookup_is_case_insensitive
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

### Embedded Serverless Architecture

SQLite is a **serverless**, **self-contained**, **zero-configuration** embedded database engine. Unlike client-server databases, SQLite reads and writes directly to ordinary disk files.

**Key Characteristics:**
- **Single file database** - entire database in one file on disk
- **Zero configuration** - no server setup or administration
- **Cross-platform** - database files portable across systems
- **ACID compliant** - atomic, consistent, isolated, durable
- **Small footprint** - ~600KB library size

**Architecture Components:**
```
┌─────────────────────────────────────┐
│      SQL Interface Layer            │
├─────────────────────────────────────┤
│      SQL Compiler (Parser/CodeGen)  │
├─────────────────────────────────────┤
│      Virtual Machine (VDBE)         │
├─────────────────────────────────────┤
│      B-Tree Engine                  │
├─────────────────────────────────────┤
│      Pager (Cache Management)       │
├─────────────────────────────────────┤
│      OS Interface (VFS)             │
└─────────────────────────────────────┘
```

### When to Use SQLite

**✅ Excellent For:**
- Embedded applications (mobile, desktop, IoT)
- Websites with <100,000 visitors/day
- Single-user applications
- Prototyping and development
- Data analysis and reporting
- Cache for enterprise data
- Internal/temporary databases

**❌ Not Recommended For:**
- High-concurrency writes (>1 writer)
- Multi-server distributed systems
- Very large datasets (>140TB theoretical, >1TB practical)
- Network file systems (NFS, SMB)
- Applications requiring granular access control

---

## 3. WAL Mode Configuration

### Write-Ahead Logging (WAL)

**WAL mode provides 2-4x write performance** and allows concurrent readers during writes. This is **critical for production use**.

**Enable WAL Mode:**
```sql
PRAGMA journal_mode = WAL;
```

**Benefits:**
- **Concurrent reads and writes** - readers don't block writers
- **Faster writes** - 2-4x performance improvement
- **Better crash recovery** - atomic commit of changes
- **Reduced fsync() calls** - fewer disk operations

**WAL Configuration:**
```sql
-- Enable WAL mode (persist across connections)
PRAGMA journal_mode = WAL;

-- Set WAL checkpoint threshold (default 1000 pages = ~4MB)
PRAGMA wal_autocheckpoint = 1000;

-- Synchronous mode for WAL (NORMAL recommended)
PRAGMA synchronous = NORMAL;  -- Safe with WAL mode

-- WAL file size limit (bytes)
PRAGMA wal_checkpoint(TRUNCATE);  -- Manually truncate WAL
```

**WAL vs Rollback Journal:**

| Feature | WAL Mode | Rollback Journal |
|---------|----------|------------------|
| Concurrent reads | ✅ Yes | ❌ No |
| Write performance | ⚡ 2-4x faster | Baseline |
| File count | 3 files | 2 files |
| Network filesystem | ⚠️ Limited | ⚠️ Limited |
| Portability | ✅ Good | ✅ Better |

**WAL Checkpointing:**
```sql
-- Passive checkpoint (don't block readers/writers)
PRAGMA wal_checkpoint(PASSIVE);

-- Full checkpoint (wait for readers to finish)
PRAGMA wal_checkpoint(FULL);

-- Restart checkpoint (reset WAL to beginning)
PRAGMA wal_checkpoint(RESTART);

-- Truncate checkpoint (shrink WAL file)
PRAGMA wal_checkpoint(TRUNCATE);
```

**Production Recommendation:**
```sql
-- Optimal WAL configuration for production
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
PRAGMA wal_autocheckpoint = 1000;
PRAGMA busy_timeout = 5000;  -- 5 seconds
```

---

## 4. Performance Optimization

### Critical PRAGMA Statements

**Essential Performance Settings:**
```sql
-- Journal mode (CRITICAL - use WAL)
PRAGMA journal_mode = WAL;

-- Synchronous mode (NORMAL safe with WAL, FULL for rollback)
PRAGMA synchronous = NORMAL;

-- Cache size (negative = KB, positive = pages)
PRAGMA cache_size = -64000;  -- 64MB cache

-- Memory-mapped I/O (bytes, 0 = disabled)
PRAGMA mmap_size = 268435456;  -- 256MB mmap

-- Temp store in memory
PRAGMA temp_store = MEMORY;

-- Locking mode (EXCLUSIVE for single-connection apps)
PRAGMA locking_mode = NORMAL;  -- or EXCLUSIVE

-- Page size (must set before creating tables)
PRAGMA page_size = 4096;  -- Match OS page size

-- Auto-vacuum mode
PRAGMA auto_vacuum = INCREMENTAL;  -- or FULL/NONE

-- Busy timeout (milliseconds)
PRAGMA busy_timeout = 5000;

-- Analysis limit (query planner)
PRAGMA analysis_limit = 1000;
```

### Performance Tuning Guidelines

**Memory Configuration:**
```sql
-- Aggressive caching for read-heavy workloads
PRAGMA cache_size = -128000;    -- 128MB cache
PRAGMA mmap_size = 1073741824;  -- 1GB mmap
PRAGMA temp_store = MEMORY;

-- Conservative for memory-constrained environments
PRAGMA cache_size = -16000;     -- 16MB cache
PRAGMA mmap_size = 67108864;    -- 64MB mmap
PRAGMA temp_store = FILE;
```

**Write Performance:**
```sql
-- Maximum write performance (use with caution)
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;     -- or OFF for maximum speed (unsafe)
PRAGMA cache_size = -64000;
PRAGMA locking_mode = EXCLUSIVE; -- Single connection only
PRAGMA temp_store = MEMORY;

-- Batch inserts in transaction
BEGIN TRANSACTION;
-- INSERT statements here (thousands)
COMMIT;
```

**Read Performance:**
```sql
-- Query optimization
PRAGMA cache_size = -128000;
PRAGMA mmap_size = 536870912;  -- 512MB
PRAGMA temp_store = MEMORY;
PRAGMA query_only = ON;         -- Read-only mode

-- Create covering indexes
CREATE INDEX idx_covering ON table(col1, col2, col3);
```

### Benchmarking Results

**Expected Performance (Modern SSD):**
- **SELECTs:** 100,000+ queries/second (simple indexed queries)
- **INSERTs:** 50,000+ rows/second (in transaction, WAL mode)
- **UPDATEs:** 30,000+ rows/second (indexed, in transaction)
- **Database size:** Handles databases up to hundreds of GB efficiently

---

## 5. Transaction Management

### ACID Compliance

SQLite provides full ACID guarantees:
- **Atomic:** All or nothing execution
- **Consistent:** Constraints enforced
- **Isolated:** Serializable isolation (default)
- **Durable:** Changes survive crashes (with synchronous=FULL)

### Transaction Types

**Deferred Transaction (Default):**
```sql
BEGIN DEFERRED TRANSACTION;
-- Lock acquired on first read/write
INSERT INTO users VALUES (1, 'Alice');
COMMIT;
```

**Immediate Transaction:**
```sql
BEGIN IMMEDIATE TRANSACTION;
-- Acquires RESERVED lock immediately
UPDATE users SET name = 'Bob' WHERE id = 1;
COMMIT;
```

**Exclusive Transaction:**
```sql
BEGIN EXCLUSIVE TRANSACTION;
-- Acquires EXCLUSIVE lock immediately
-- Blocks all other connections
DELETE FROM users WHERE id = 1;
COMMIT;
```

### Best Practices

**Batch Operations:**
```sql
-- WRONG: 1,000 transactions = slow
for i in range(1000):
    cursor.execute("INSERT INTO data VALUES (?)", (i,))

-- RIGHT: 1 transaction = 100x faster
cursor.execute("BEGIN TRANSACTION")
for i in range(1000):
    cursor.execute("INSERT INTO data VALUES (?)", (i,))
cursor.execute("COMMIT")
```

**Error Handling:**
```python
import sqlite3

conn = sqlite3.connect('app.db')
try:
    conn.execute("BEGIN IMMEDIATE TRANSACTION")
    conn.execute("INSERT INTO users VALUES (?, ?)", (1, 'Alice'))
    conn.execute("INSERT INTO orders VALUES (?, ?)", (1, 'Product'))
    conn.commit()
except sqlite3.Error as e:
    conn.rollback()
    print(f"Transaction failed: {e}")
finally:
    conn.close()
```

**Savepoints:**
```sql
BEGIN TRANSACTION;
  INSERT INTO users VALUES (1, 'Alice');
  SAVEPOINT sp1;

  INSERT INTO users VALUES (2, 'Bob');
  -- Error occurs
  ROLLBACK TO sp1;  -- Undo Bob, keep Alice

  INSERT INTO users VALUES (3, 'Carol');
COMMIT;
```

---

## 6. Concurrency and Locking

### Locking Mechanism

SQLite uses **database-level locking** with five lock states:

```
UNLOCKED → SHARED → RESERVED → PENDING → EXCLUSIVE
```

**Lock Types:**
1. **UNLOCKED:** No locks held
2. **SHARED:** Read lock (multiple readers allowed)
3. **RESERVED:** Intent to write (one per database)
4. **PENDING:** Waiting for readers to finish
5. **EXCLUSIVE:** Write lock (exclusive access)

### Concurrency Limitations

**Single Writer Limitation:**
- **Only ONE writer at a time** (database-level lock)
- Multiple readers can read concurrently
- Readers don't block readers
- Writers block all readers (in rollback mode)
- Writers don't block readers (in WAL mode)

**Handling Concurrent Writes:**
```python
import sqlite3
import time

def write_with_retry(conn, sql, params, max_retries=5):
    """Retry on SQLITE_BUSY errors"""
    for attempt in range(max_retries):
        try:
            conn.execute(sql, params)
            conn.commit()
            return True
        except sqlite3.OperationalError as e:
            if "database is locked" in str(e):
                if attempt < max_retries - 1:
                    time.sleep(0.1 * (2 ** attempt))  # Exponential backoff
                else:
                    raise
    return False
```

### Busy Timeout

```sql
-- Set busy timeout (wait up to 5 seconds for lock)
PRAGMA busy_timeout = 5000;
```

```python
# Python
conn = sqlite3.connect('app.db', timeout=5.0)

# Set busy handler
def busy_handler(attempts):
    if attempts < 10:
        time.sleep(0.1)
        return 1  # Retry
    return 0  # Abort

conn.set_busy_handler(busy_handler)
```

### WAL Mode Concurrency

**WAL Mode Advantages:**
```sql
PRAGMA journal_mode = WAL;
```

- ✅ **Readers don't block writers**
- ✅ **Writers don't block readers**
- ✅ **One writer, multiple readers simultaneously**
- ⚠️ **Still only one writer at a time**

---

## 7. Indexing Strategies

### Index Types

**B-Tree Index (Default):**
```sql
CREATE INDEX idx_user_email ON users(email);
```

**Unique Index:**
```sql
CREATE UNIQUE INDEX idx_user_email_unique ON users(email);
```

**Partial Index:**
```sql
-- Index only active users (saves space)
CREATE INDEX idx_active_users ON users(email)
WHERE active = 1;
```

**Covering Index:**
```sql
-- Index contains all columns needed by query
CREATE INDEX idx_user_covering ON users(email, name, created_at);

-- Query uses index-only scan (no table lookup)
SELECT email, name, created_at FROM users WHERE email = 'user@example.com';
```

**Expression Index:**
```sql
-- Index on computed expression
CREATE INDEX idx_user_lower_email ON users(LOWER(email));

-- Query can use the index
SELECT * FROM users WHERE LOWER(email) = 'user@example.com';
```

**Multi-Column Index:**
```sql
-- Left-to-right prefix rule applies
CREATE INDEX idx_user_search ON users(last_name, first_name, city);

-- These queries use the index:
WHERE last_name = 'Smith'
WHERE last_name = 'Smith' AND first_name = 'John'
WHERE last_name = 'Smith' AND first_name = 'John' AND city = 'NYC'

-- This query does NOT use the index efficiently:
WHERE first_name = 'John'  -- Doesn't start with last_name
```

### Index Best Practices

**When to Create Indexes:**
- ✅ Columns in WHERE clauses
- ✅ Columns in JOIN conditions
- ✅ Columns in ORDER BY
- ✅ Foreign key columns
- ✅ Columns used in GROUP BY

**When NOT to Index:**
- ❌ Very small tables (<1000 rows)
- ❌ Columns with low cardinality (few distinct values)
- ❌ Columns updated frequently (write overhead)
- ❌ Tables with high INSERT/UPDATE ratio

**Index Maintenance:**
```sql
-- Rebuild all indexes and update statistics
VACUUM;

-- Update query planner statistics
ANALYZE;

-- View index usage
EXPLAIN QUERY PLAN SELECT * FROM users WHERE email = 'test@example.com';

-- List all indexes
SELECT name, tbl_name, sql FROM sqlite_master
WHERE type = 'index' AND sql IS NOT NULL;

-- Drop unused index
DROP INDEX idx_user_email;
```

**Index Size Analysis:**
```sql
-- Check database and index sizes
SELECT
    name,
    SUM(pgsize) as size_bytes,
    SUM(pgsize)/1024/1024 as size_mb
FROM dbstat
GROUP BY name
ORDER BY size_bytes DESC;
```

---

## 8. Full-Text Search (FTS5)

### FTS5 Overview

FTS5 is SQLite's modern full-text search extension with **better performance and features** than FTS3/FTS4.

**Creating FTS5 Table:**
```sql
-- Create FTS5 virtual table
CREATE VIRTUAL TABLE documents_fts USING fts5(
    title,
    content,
    tags,
    tokenize='porter unicode61'  -- Porter stemming + Unicode
);

-- Insert data
INSERT INTO documents_fts (rowid, title, content, tags)
VALUES (1, 'SQLite Guide', 'Full-text search tutorial', 'database search');
```

### FTS5 Query Syntax

**Basic Searches:**
```sql
-- Simple phrase search
SELECT * FROM documents_fts WHERE documents_fts MATCH 'sqlite';

-- AND operator (implicit)
SELECT * FROM documents_fts WHERE documents_fts MATCH 'sqlite database';

-- OR operator
SELECT * FROM documents_fts WHERE documents_fts MATCH 'sqlite OR postgres';

-- NOT operator
SELECT * FROM documents_fts WHERE documents_fts MATCH 'sqlite NOT mysql';

-- Phrase search
SELECT * FROM documents_fts WHERE documents_fts MATCH '"full text search"';

-- Column-specific search
SELECT * FROM documents_fts WHERE documents_fts MATCH 'title:sqlite';

-- Prefix search
SELECT * FROM documents_fts WHERE documents_fts MATCH 'data*';
```

**Advanced Features:**
```sql
-- Ranking (relevance score)
SELECT *, rank FROM documents_fts
WHERE documents_fts MATCH 'sqlite'
ORDER BY rank;

-- Snippet extraction (context around match)
SELECT snippet(documents_fts, 1, '<b>', '</b>', '...', 15)
FROM documents_fts
WHERE documents_fts MATCH 'search';

-- Highlighting
SELECT highlight(documents_fts, 1, '<mark>', '</mark>')
FROM documents_fts
WHERE documents_fts MATCH 'sqlite';
```

### FTS5 with External Content

**Separate FTS index from main table:**
```sql
-- Main table
CREATE TABLE documents (
    id INTEGER PRIMARY KEY,
    title TEXT,
    content TEXT,
    created_at DATETIME
);

-- FTS5 index (references main table)
CREATE VIRTUAL TABLE documents_fts USING fts5(
    title,
    content,
    content='documents',     -- External content table
    content_rowid='id'       -- Rowid mapping
);

-- Triggers to keep FTS in sync
CREATE TRIGGER documents_ai AFTER INSERT ON documents BEGIN
    INSERT INTO documents_fts(rowid, title, content)
    VALUES (new.id, new.title, new.content);
END;

CREATE TRIGGER documents_ad AFTER DELETE ON documents BEGIN
    DELETE FROM documents_fts WHERE rowid = old.id;
END;

CREATE TRIGGER documents_au AFTER UPDATE ON documents BEGIN
    UPDATE documents_fts SET title=new.title, content=new.content
    WHERE rowid = old.id;
END;
```

### FTS5 Performance

**Optimization:**
```sql
-- Rebuild FTS5 index
INSERT INTO documents_fts(documents_fts) VALUES('rebuild');

-- Optimize index (merge b-tree segments)
INSERT INTO documents_fts(documents_fts) VALUES('optimize');

-- Check integrity
INSERT INTO documents_fts(documents_fts) VALUES('integrity-check');
```

---

## 9. JSON Support

### JSON Functions (SQLite 3.38+)

SQLite includes built-in JSON functions for **storing and querying JSON data**.

**Creating JSON Data:**
```sql
CREATE TABLE api_logs (
    id INTEGER PRIMARY KEY,
    event_data TEXT,  -- JSON stored as TEXT
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Insert JSON
INSERT INTO api_logs (event_data) VALUES
(json_object('user_id', 123, 'action', 'login', 'ip', '192.168.1.1'));
```

**JSON Extraction:**
```sql
-- Extract JSON field (-> returns JSON, ->> returns SQL value)
SELECT
    json_extract(event_data, '$.user_id') as user_id,
    event_data->>'$.action' as action,
    event_data->'$.metadata' as metadata
FROM api_logs;

-- Array access
SELECT json_extract('[1,2,3,4]', '$[2]');  -- Returns 3
```

**JSON Aggregation:**
```sql
-- Build JSON object
SELECT json_object(
    'id', id,
    'name', name,
    'email', email
) FROM users WHERE id = 1;

-- Build JSON array
SELECT json_group_array(name) FROM users;

-- Build array of objects
SELECT json_group_array(
    json_object('id', id, 'name', name)
) FROM users;
```

**JSON Table Function:**
```sql
-- Convert JSON array to table
SELECT value FROM json_each('[1,2,3,4,5]');

-- Convert JSON object to table
SELECT key, value FROM json_each('{"a":1,"b":2,"c":3}');

-- Nested JSON traversal
SELECT
    fullkey,
    value
FROM json_tree('{"users":[{"id":1,"name":"Alice"},{"id":2,"name":"Bob"}]}')
WHERE fullkey LIKE '%.name';
```

### JSON Indexing

**Index on JSON Field:**
```sql
-- Create index on extracted JSON field
CREATE INDEX idx_user_id ON api_logs(json_extract(event_data, '$.user_id'));

-- Query uses index
SELECT * FROM api_logs
WHERE json_extract(event_data, '$.user_id') = 123;
```

**Generated Column (SQLite 3.31+):**
```sql
CREATE TABLE api_logs (
    id INTEGER PRIMARY KEY,
    event_data TEXT,
    user_id INTEGER GENERATED ALWAYS AS (json_extract(event_data, '$.user_id')) STORED
);

-- Index the generated column
CREATE INDEX idx_user_id ON api_logs(user_id);

-- Query is fast
SELECT * FROM api_logs WHERE user_id = 123;
```

---

## 10. Memory Management

### Cache Configuration

**Cache Size:**
```sql
-- Default: 2000 pages (~8MB with 4KB pages)
-- Negative value = KB, positive = pages

PRAGMA cache_size = -64000;  -- 64MB cache (recommended)
PRAGMA cache_size = -128000; -- 128MB for read-heavy workloads
PRAGMA cache_size = -16000;  -- 16MB for embedded/mobile
```

**Query Cache Size:**
```sql
-- Check current cache size
PRAGMA cache_size;

-- Check page size
PRAGMA page_size;

-- Calculate cache in MB
SELECT (cache_size * page_size) / 1024 / 1024 as cache_mb;
```

### Memory-Mapped I/O

**mmap Configuration:**
```sql
-- Memory-map up to 256MB of database file
PRAGMA mmap_size = 268435456;

-- Disable mmap (use traditional I/O)
PRAGMA mmap_size = 0;

-- Aggressive mmap for large databases
PRAGMA mmap_size = 1073741824;  -- 1GB
```

**Benefits:**
- ✅ Faster reads (eliminate kernel copy)
- ✅ Reduced memory usage (OS manages pages)
- ⚠️ May increase memory pressure
- ❌ Not available on all platforms

### Temp Store

**Temporary Tables Location:**
```sql
-- Store temporary tables/indexes in memory (default: FILE)
PRAGMA temp_store = MEMORY;

-- Check current setting
PRAGMA temp_store;
```

**Options:**
- `DEFAULT` (0): Use compile-time default
- `FILE` (1): Temporary files on disk
- `MEMORY` (2): Temporary tables in RAM

### Memory Limits

**Hard Limits:**
```sql
-- Set maximum memory usage (bytes)
-- Available via C API: sqlite3_config(SQLITE_CONFIG_MEMSTATUS)
```

**Soft Heap Limit:**
```python
import sqlite3

# Soft heap limit (SQLite attempts to stay below)
conn = sqlite3.connect('app.db')
conn.execute("PRAGMA soft_heap_limit = 67108864")  # 64MB
```

### Memory-Constrained Environments

**Minimal Configuration:**
```sql
-- IoT/Embedded devices
PRAGMA page_size = 1024;        -- Smaller pages
PRAGMA cache_size = -4000;      -- 4MB cache
PRAGMA mmap_size = 0;           -- Disable mmap
PRAGMA temp_store = FILE;       -- Temp on disk
PRAGMA locking_mode = EXCLUSIVE; -- Reduce lock overhead
```

---

## 11. Security with SQLCipher

### SQLCipher Overview

**SQLCipher** provides **AES-256 encryption** for SQLite databases at the page level.

**Installation:**
```bash
# macOS
brew install sqlcipher

# Ubuntu/Debian
apt-get install libsqlcipher-dev

# Python
pip install pysqlcipher3
```

### Encryption Configuration

**Creating Encrypted Database:**
```python
from pysqlcipher3 import dbapi2 as sqlite

# Connect and set key
conn = sqlite.connect('encrypted.db')
conn.execute("PRAGMA key = 'your-strong-passphrase-here'")
conn.execute("CREATE TABLE secrets (id INTEGER PRIMARY KEY, data TEXT)")
conn.commit()
conn.close()
```

**Opening Encrypted Database:**
```python
conn = sqlite.connect('encrypted.db')
conn.execute("PRAGMA key = 'your-strong-passphrase-here'")
cursor = conn.execute("SELECT * FROM secrets")
```

**Changing Encryption Key:**
```python
conn = sqlite.connect('encrypted.db')
conn.execute("PRAGMA key = 'old-passphrase'")
conn.execute("PRAGMA rekey = 'new-passphrase'")
conn.close()
```

### SQLCipher Performance

**Encryption Settings:**
```sql
-- Set encryption cipher
PRAGMA cipher = 'aes-256-cbc';

-- KDF iterations (higher = more secure but slower)
PRAGMA kdf_iter = 256000;  -- Default: 256,000 (SQLCipher 4.x)

-- Page size
PRAGMA cipher_page_size = 4096;

-- HMAC algorithm
PRAGMA cipher_hmac_algorithm = 'HMAC_SHA512';

-- KDF algorithm
PRAGMA cipher_kdf_algorithm = 'PBKDF2_HMAC_SHA512';
```

**Performance Impact:**
- Expect **10-15% performance overhead** for encryption
- KDF iterations affect open time (not query time)
- Use hardware AES acceleration when available

### Security Best Practices

**Key Management:**
```python
import os
import sqlite3
from cryptography.fernet import Fernet

# Generate secure key
key = Fernet.generate_key()

# Store key securely (not in code!)
# Use: OS keychain, environment variables, secrets management
os.environ['DB_ENCRYPTION_KEY'] = key.decode()

# Retrieve key
db_key = os.environ.get('DB_ENCRYPTION_KEY')
conn = sqlite.connect('app.db')
conn.execute(f"PRAGMA key = '{db_key}'")
```

**Standard SQLite Security:**
```sql
-- Disable loading extensions (prevent injection)
PRAGMA trusted_schema = OFF;

-- Read-only mode
PRAGMA query_only = ON;

-- Prevent writes
PRAGMA journal_mode = DELETE;
PRAGMA locking_mode = NORMAL;
```

**Input Sanitization:**
```python
# WRONG: SQL injection vulnerability
user_input = "'; DROP TABLE users; --"
cursor.execute(f"SELECT * FROM users WHERE name = '{user_input}'")

# RIGHT: Use parameterized queries
cursor.execute("SELECT * FROM users WHERE name = ?", (user_input,))
```

---

## 12. Backup Strategies

### Online Backup API

**SQLite Backup (Hot Backup):**
```python
import sqlite3

def backup_database(source_db, backup_db):
    """Perform online backup while database is in use"""
    source = sqlite3.connect(source_db)
    backup = sqlite3.connect(backup_db)

    with backup:
        source.backup(backup, pages=100, progress=callback)

    source.close()
    backup.close()

def callback(status, remaining, total):
    print(f'Copied {total-remaining} of {total} pages...')

# Backup main database
backup_database('production.db', 'backup_20260206.db')
```

### File-Based Backup

**Simple File Copy (Cold Backup):**
```bash
# Stop application first
sqlite3 app.db "VACUUM INTO 'backup.db'"

# Or checkpoint WAL and copy files
sqlite3 app.db "PRAGMA wal_checkpoint(TRUNCATE)"
cp app.db backup.db
cp app.db-wal backup.db-wal  # If exists
cp app.db-shm backup.db-shm  # If exists
```

**SQL Dump:**
```bash
# Export to SQL script
sqlite3 app.db .dump > backup.sql

# Restore from SQL script
sqlite3 restored.db < backup.sql

# Compressed backup
sqlite3 app.db .dump | gzip > backup.sql.gz

# Restore compressed
gunzip < backup.sql.gz | sqlite3 restored.db
```

### Incremental Backup

**Using WAL Checkpointing:**
```sql
-- Checkpoint to backup file
PRAGMA wal_checkpoint(TRUNCATE);

-- Copy WAL segments incrementally
-- (Requires external tooling)
```

### Automated Backup Script

**Bash Backup Script:**
```bash
#!/bin/bash
# sqlite-backup.sh

DB_PATH="/var/lib/myapp/app.db"
BACKUP_DIR="/var/backups/sqlite"
DATE=$(date +%Y%m%d_%H%M%S)
RETENTION_DAYS=30

# Create backup directory
mkdir -p "$BACKUP_DIR"

# Checkpoint WAL
sqlite3 "$DB_PATH" "PRAGMA wal_checkpoint(TRUNCATE);"

# Perform vacuum into backup
sqlite3 "$DB_PATH" "VACUUM INTO '$BACKUP_DIR/app_${DATE}.db';"

# Verify backup
if sqlite3 "$BACKUP_DIR/app_${DATE}.db" "PRAGMA integrity_check;" | grep -q "ok"; then
    echo "Backup successful: app_${DATE}.db"

    # Compress backup
    gzip "$BACKUP_DIR/app_${DATE}.db"

    # Delete old backups
    find "$BACKUP_DIR" -name "app_*.db.gz" -mtime +$RETENTION_DAYS -delete
else
    echo "Backup verification failed!"
    exit 1
fi
```

**Cron Job:**
```cron
# Daily backup at 2 AM
0 2 * * * /usr/local/bin/sqlite-backup.sh
```

### Point-in-Time Recovery

**Transaction Logging:**
```python
import sqlite3
import time

def logged_execute(conn, sql, params=()):
    """Log all SQL statements for replay"""
    timestamp = time.time()

    # Log to audit table
    conn.execute("""
        INSERT INTO audit_log (timestamp, sql_statement, parameters)
        VALUES (?, ?, ?)
    """, (timestamp, sql, str(params)))

    # Execute original statement
    conn.execute(sql, params)
    conn.commit()
```

---

## 13. Migration Strategies

### Schema Versioning

**User Version Pragma:**
```sql
-- Set schema version
PRAGMA user_version = 1;

-- Check schema version
PRAGMA user_version;
```

**Migration Framework:**
```python
import sqlite3

class DatabaseMigrator:
    def __init__(self, db_path):
        self.conn = sqlite3.connect(db_path)
        self.migrations = {
            1: self.migrate_to_v1,
            2: self.migrate_to_v2,
            3: self.migrate_to_v3
        }

    def get_version(self):
        cursor = self.conn.execute("PRAGMA user_version")
        return cursor.fetchone()[0]

    def set_version(self, version):
        self.conn.execute(f"PRAGMA user_version = {version}")

    def migrate(self):
        current_version = self.get_version()
        target_version = max(self.migrations.keys())

        for version in range(current_version + 1, target_version + 1):
            print(f"Migrating to version {version}...")
            self.migrations[version]()
            self.set_version(version)
            print(f"Migration to version {version} complete")

    def migrate_to_v1(self):
        self.conn.execute("""
            CREATE TABLE users (
                id INTEGER PRIMARY KEY,
                email TEXT UNIQUE NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.commit()

    def migrate_to_v2(self):
        # Add column with default value
        self.conn.execute("""
            ALTER TABLE users ADD COLUMN name TEXT DEFAULT ''
        """)
        self.conn.commit()

    def migrate_to_v3(self):
        # Create new table with modified schema
        self.conn.execute("""
            CREATE TABLE users_new (
                id INTEGER PRIMARY KEY,
                email TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Copy data
        self.conn.execute("""
            INSERT INTO users_new (id, email, name, created_at)
            SELECT id, email, name, created_at FROM users
        """)

        # Drop old table
        self.conn.execute("DROP TABLE users")

        # Rename new table
        self.conn.execute("ALTER TABLE users_new RENAME TO users")

        self.conn.commit()

# Usage
migrator = DatabaseMigrator('app.db')
migrator.migrate()
```

### Column Operations

**Adding Columns:**
```sql
-- Simple add (SQLite 3.35+)
ALTER TABLE users ADD COLUMN phone TEXT;

-- With default value
ALTER TABLE users ADD COLUMN status TEXT DEFAULT 'active';

-- With NOT NULL and default
ALTER TABLE users ADD COLUMN verified INTEGER DEFAULT 0 NOT NULL;
```

**Renaming Columns (SQLite 3.25+):**
```sql
ALTER TABLE users RENAME COLUMN name TO full_name;
```

**Dropping Columns (SQLite 3.35+):**
```sql
ALTER TABLE users DROP COLUMN phone;
```

### Table Restructuring

**Complex Schema Changes:**
```sql
-- When ALTER TABLE isn't sufficient, recreate table:

BEGIN TRANSACTION;

-- Create new table with desired schema
CREATE TABLE users_new (
    id INTEGER PRIMARY KEY,
    email TEXT UNIQUE NOT NULL,
    full_name TEXT NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Copy data with transformation
INSERT INTO users_new (id, email, full_name, created_at)
SELECT id, email, first_name || ' ' || last_name, created_at
FROM users;

-- Drop old table
DROP TABLE users;

-- Rename new table
ALTER TABLE users_new RENAME TO users;

-- Recreate indexes
CREATE INDEX idx_users_email ON users(email);

-- Recreate triggers
CREATE TRIGGER update_timestamp AFTER UPDATE ON users
BEGIN
    UPDATE users SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

COMMIT;
```

### Data Migration

**Bulk Data Import:**
```python
import sqlite3
import csv

def import_csv(db_path, csv_path, table_name):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames

        # Prepare bulk insert
        conn.execute("BEGIN TRANSACTION")

        for row in reader:
            placeholders = ','.join('?' * len(columns))
            sql = f"INSERT INTO {table_name} ({','.join(columns)}) VALUES ({placeholders})"
            cursor.execute(sql, list(row.values()))

        conn.commit()

    conn.close()
```

---

## 14. Schema Design

### Data Type Best Practices

**SQLite Type Affinity:**
```sql
-- SQLite has 5 storage classes: NULL, INTEGER, REAL, TEXT, BLOB
-- Type affinity determines how values are stored

CREATE TABLE examples (
    id INTEGER PRIMARY KEY,          -- INTEGER affinity
    price REAL,                      -- REAL affinity
    name TEXT,                       -- TEXT affinity
    data BLOB,                       -- BLOB affinity
    count NUMERIC,                   -- NUMERIC affinity (prefers INTEGER/REAL)
    flag BOOLEAN,                    -- Stored as INTEGER (0/1)
    created_at DATETIME,             -- Stored as TEXT, INTEGER, or REAL
    metadata JSON                    -- Stored as TEXT
);
```

**Recommended Types:**
```sql
CREATE TABLE users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    uuid TEXT NOT NULL UNIQUE,                    -- UUIDs as TEXT
    email TEXT NOT NULL UNIQUE COLLATE NOCASE,    -- Case-insensitive
    password_hash TEXT NOT NULL,
    is_active INTEGER DEFAULT 1,                  -- Boolean (0/1)
    balance REAL,                                 -- Currency (or INTEGER for cents)
    created_at INTEGER NOT NULL,                  -- Unix timestamp
    updated_at INTEGER NOT NULL,
    metadata TEXT,                                -- JSON data
    profile_picture BLOB                          -- Binary data
);
```

### Primary Keys

**Integer Primary Key:**
```sql
-- Aliased to ROWID (no extra storage)
CREATE TABLE users (
    id INTEGER PRIMARY KEY,  -- Alias to ROWID
    email TEXT
);

-- AUTOINCREMENT (use sparingly, slight overhead)
CREATE TABLE orders (
    id INTEGER PRIMARY KEY AUTOINCREMENT,  -- Guarantees unique across deletes
    user_id INTEGER
);
```

**UUID Primary Key:**
```sql
CREATE TABLE sessions (
    id TEXT PRIMARY KEY DEFAULT (hex(randomblob(16))),  -- UUID
    user_id INTEGER NOT NULL,
    expires_at INTEGER NOT NULL
);
```

### Foreign Keys

**Enable Foreign Keys:**
```sql
-- MUST enable on each connection
PRAGMA foreign_keys = ON;
```

**Foreign Key Constraints:**
```sql
CREATE TABLE users (
    id INTEGER PRIMARY KEY,
    email TEXT NOT NULL
);

CREATE TABLE posts (
    id INTEGER PRIMARY KEY,
    user_id INTEGER NOT NULL,
    title TEXT NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(id)
        ON DELETE CASCADE
        ON UPDATE CASCADE
);
```

**Cascading Actions:**
```sql
-- Delete posts when user is deleted
ON DELETE CASCADE

-- Prevent deletion if posts exist
ON DELETE RESTRICT

-- Set user_id to NULL when user deleted
ON DELETE SET NULL

-- Set user_id to default value
ON DELETE SET DEFAULT

-- Same options for ON UPDATE
```

### Normalization

**Normalized Schema:**
```sql
-- Users table
CREATE TABLE users (
    id INTEGER PRIMARY KEY,
    email TEXT UNIQUE NOT NULL,
    created_at INTEGER NOT NULL
);

-- User profiles (1:1)
CREATE TABLE user_profiles (
    user_id INTEGER PRIMARY KEY,
    first_name TEXT,
    last_name TEXT,
    bio TEXT,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- Posts (1:many)
CREATE TABLE posts (
    id INTEGER PRIMARY KEY,
    user_id INTEGER NOT NULL,
    title TEXT NOT NULL,
    content TEXT,
    created_at INTEGER NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- Tags (many:many via junction table)
CREATE TABLE tags (
    id INTEGER PRIMARY KEY,
    name TEXT UNIQUE NOT NULL
);

CREATE TABLE post_tags (
    post_id INTEGER NOT NULL,
    tag_id INTEGER NOT NULL,
    PRIMARY KEY (post_id, tag_id),
    FOREIGN KEY (post_id) REFERENCES posts(id) ON DELETE CASCADE,
    FOREIGN KEY (tag_id) REFERENCES tags(id) ON DELETE CASCADE
);
```

### Denormalization Strategies

**When to Denormalize:**
```sql
-- Store computed values for fast reads
CREATE TABLE order_summary (
    order_id INTEGER PRIMARY KEY,
    total_items INTEGER NOT NULL,
    total_amount REAL NOT NULL,
    created_at INTEGER NOT NULL
);

-- Trigger to maintain denormalized data
CREATE TRIGGER update_order_summary AFTER INSERT ON order_items
BEGIN
    UPDATE order_summary
    SET total_items = total_items + NEW.quantity,
        total_amount = total_amount + (NEW.quantity * NEW.price)
    WHERE order_id = NEW.order_id;
END;
```

### Constraints

**Check Constraints:**
```sql
CREATE TABLE products (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    price REAL NOT NULL CHECK(price > 0),
    stock INTEGER DEFAULT 0 CHECK(stock >= 0),
    category TEXT CHECK(category IN ('electronics', 'clothing', 'food'))
);
```

**Unique Constraints:**
```sql
-- Single column
CREATE TABLE users (
    id INTEGER PRIMARY KEY,
    email TEXT UNIQUE NOT NULL
);

-- Multiple columns (composite unique)
CREATE TABLE enrollments (
    student_id INTEGER,
    course_id INTEGER,
    UNIQUE(student_id, course_id)
);
```

---

## 15. Query Optimization

### Query Planner

**EXPLAIN QUERY PLAN:**
```sql
EXPLAIN QUERY PLAN
SELECT u.email, p.title
FROM users u
JOIN posts p ON u.id = p.user_id
WHERE u.email = 'user@example.com';

-- Output:
-- QUERY PLAN
-- |--SEARCH TABLE users AS u USING INDEX idx_users_email (email=?)
-- `--SEARCH TABLE posts AS p USING INDEX idx_posts_user_id (user_id=?)
```

**EXPLAIN:**
```sql
-- Show VDBE bytecode (advanced)
EXPLAIN
SELECT * FROM users WHERE id = 1;
```

### Index Usage

**Verify Index Usage:**
```sql
-- Query should use index
EXPLAIN QUERY PLAN
SELECT * FROM users WHERE email = 'test@example.com';
-- Look for: SEARCH TABLE users USING INDEX idx_users_email

-- Query does NOT use index (table scan)
EXPLAIN QUERY PLAN
SELECT * FROM users WHERE LOWER(email) = 'test@example.com';
-- Look for: SCAN TABLE users

-- Fix: Create expression index
CREATE INDEX idx_users_lower_email ON users(LOWER(email));
```

### Query Optimization Techniques

**Use Covering Indexes:**
```sql
-- Without covering index: query reads index + table
SELECT name, email FROM users WHERE email = 'test@example.com';

-- Create covering index
CREATE INDEX idx_users_email_name ON users(email, name);

-- Now query only reads index (faster)
EXPLAIN QUERY PLAN
SELECT name, email FROM users WHERE email = 'test@example.com';
-- SEARCH TABLE users USING COVERING INDEX idx_users_email_name
```

**Avoid Functions on Indexed Columns:**
```sql
-- BAD: Function prevents index usage
SELECT * FROM users WHERE SUBSTR(email, 1, 5) = 'admin';

-- GOOD: Use LIKE with index
SELECT * FROM users WHERE email LIKE 'admin%';
```

**Use EXISTS Instead of COUNT:**
```sql
-- BAD: Counts all matching rows
SELECT EXISTS(SELECT 1 FROM users WHERE email = 'test@example.com' LIMIT 1);

-- GOOD: Stops at first match
SELECT EXISTS(SELECT 1 FROM users WHERE email = 'test@example.com');
```

**Subquery vs JOIN:**
```sql
-- Subquery (may be slower)
SELECT * FROM users
WHERE id IN (SELECT user_id FROM posts WHERE published = 1);

-- JOIN (often faster)
SELECT DISTINCT u.* FROM users u
INNER JOIN posts p ON u.id = p.user_id
WHERE p.published = 1;
```

### Statistics and Analysis

**Update Query Planner Statistics:**
```sql
-- Analyze entire database
ANALYZE;

-- Analyze specific table
ANALYZE users;

-- Check statistics
SELECT * FROM sqlite_stat1;
SELECT * FROM sqlite_stat4;  -- If compiled with SQLITE_ENABLE_STAT4
```

**Auto-Analyze:**
```sql
-- Enable automatic ANALYZE after N changes
PRAGMA optimize;  -- Run periodically (e.g., on app shutdown)
```

### Query Performance Tips

**Batch Operations:**
```python
# SLOW: 1000 individual commits
for i in range(1000):
    conn.execute("INSERT INTO data VALUES (?)", (i,))
    conn.commit()

# FAST: 1 commit for 1000 inserts (100x faster)
conn.execute("BEGIN TRANSACTION")
for i in range(1000):
    conn.execute("INSERT INTO data VALUES (?)", (i,))
conn.commit()
```

**Prepared Statements:**
```python
# Prepare once, execute many times
stmt = conn.execute("SELECT * FROM users WHERE id = ?")
for user_id in user_ids:
    cursor = stmt.execute((user_id,))
    result = cursor.fetchone()
```

---

## 16. Connection Management

### Connection Pooling

**Python Connection Pool:**
```python
import sqlite3
from queue import Queue
from threading import Lock

class SQLitePool:
    def __init__(self, database, pool_size=5):
        self.database = database
        self.pool = Queue(maxsize=pool_size)
        self.lock = Lock()

        # Initialize pool
        for _ in range(pool_size):
            conn = sqlite3.connect(database, check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            self.pool.put(conn)

    def get_connection(self):
        return self.pool.get()

    def return_connection(self, conn):
        self.pool.put(conn)

    def close_all(self):
        while not self.pool.empty():
            conn = self.pool.get()
            conn.close()

# Usage
pool = SQLitePool('app.db', pool_size=10)

def query_users():
    conn = pool.get_connection()
    try:
        cursor = conn.execute("SELECT * FROM users")
        return cursor.fetchall()
    finally:
        pool.return_connection(conn)
```

**Context Manager:**
```python
from contextlib import contextmanager

@contextmanager
def get_db_connection(pool):
    conn = pool.get_connection()
    try:
        yield conn
    finally:
        pool.return_connection(conn)

# Usage
with get_db_connection(pool) as conn:
    cursor = conn.execute("SELECT * FROM users")
    users = cursor.fetchall()
```

### Connection Configuration

**Per-Connection Settings:**
```python
import sqlite3

def create_connection(db_path):
    conn = sqlite3.connect(db_path)

    # Enable WAL mode
    conn.execute("PRAGMA journal_mode=WAL")

    # Foreign keys
    conn.execute("PRAGMA foreign_keys=ON")

    # Cache size
    conn.execute("PRAGMA cache_size=-64000")  # 64MB

    # Busy timeout
    conn.execute("PRAGMA busy_timeout=5000")  # 5 seconds

    # Temp store
    conn.execute("PRAGMA temp_store=MEMORY")

    # Synchronous mode
    conn.execute("PRAGMA synchronous=NORMAL")

    # Row factory (dict-like rows)
    conn.row_factory = sqlite3.Row

    return conn
```

### Thread Safety

**SQLite Threading Modes:**
1. **Single-thread** - No thread safety
2. **Multi-thread** - Connections cannot be shared
3. **Serialized** - Full thread safety (default)

**Python Thread Safety:**
```python
import sqlite3
from threading import Thread, Lock

# Option 1: One connection per thread
import threading
local_storage = threading.local()

def get_connection():
    if not hasattr(local_storage, 'conn'):
        local_storage.conn = sqlite3.connect('app.db')
    return local_storage.conn

# Option 2: Shared connection with lock
class ThreadSafeConnection:
    def __init__(self, db_path):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.lock = Lock()

    def execute(self, sql, params=()):
        with self.lock:
            return self.conn.execute(sql, params)
```

### Read-Only Connections

**Read-Only Mode:**
```python
# URI connection for read-only
conn = sqlite3.connect('file:app.db?mode=ro', uri=True)

# Or using PRAGMA
conn = sqlite3.connect('app.db')
conn.execute("PRAGMA query_only=ON")
```

---

## 17. Container Deployment

### Docker Deployment

**Dockerfile for SQLite Application:**
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install SQLite (usually included in Python image)
RUN apt-get update && \
    apt-get install -y sqlite3 && \
    rm -rf /var/lib/apt/lists/*

# Copy application
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Create directory for database with correct permissions
RUN mkdir -p /app/data && chmod 777 /app/data

# Volume for persistent database storage
VOLUME ["/app/data"]

# Set environment variables
ENV DATABASE_PATH=/app/data/app.db
ENV SQLITE_BUSY_TIMEOUT=5000

CMD ["python", "app.py"]
```

**Docker Compose:**
```yaml
version: '3.8'

services:
  app:
    build: .
    volumes:
      # Named volume for database persistence
      - sqlite-data:/app/data
    environment:
      - DATABASE_PATH=/app/data/app.db
      - SQLITE_WAL_MODE=1
    restart: unless-stopped

    # Health check
    healthcheck:
      test: ["CMD", "sqlite3", "/app/data/app.db", "SELECT 1"]
      interval: 30s
      timeout: 10s
      retries: 3

volumes:
  sqlite-data:
    driver: local
```

### Volume Management

**Persistent Storage:**
```bash
# Create named volume
docker volume create sqlite-data

# Run container with volume
docker run -d \
  --name myapp \
  -v sqlite-data:/app/data \
  myapp:latest

# Backup database from volume
docker run --rm \
  -v sqlite-data:/data \
  -v $(pwd):/backup \
  ubuntu tar czf /backup/db-backup.tar.gz -C /data .

# Restore database to volume
docker run --rm \
  -v sqlite-data:/data \
  -v $(pwd):/backup \
  ubuntu tar xzf /backup/db-backup.tar.gz -C /data
```

### Kubernetes Deployment

**PersistentVolumeClaim:**
```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: sqlite-pvc
spec:
  accessModes:
    - ReadWriteOnce  # Single-node only (SQLite limitation)
  resources:
    requests:
      storage: 10Gi
  storageClassName: fast-ssd
```

**Deployment:**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: myapp
spec:
  replicas: 1  # MUST be 1 for SQLite (single-node database)
  selector:
    matchLabels:
      app: myapp
  template:
    metadata:
      labels:
        app: myapp
    spec:
      containers:
      - name: app
        image: myapp:latest
        env:
        - name: DATABASE_PATH
          value: /data/app.db
        volumeMounts:
        - name: sqlite-storage
          mountPath: /data

        # Readiness probe
        readinessProbe:
          exec:
            command:
            - sqlite3
            - /data/app.db
            - SELECT 1
          initialDelaySeconds: 5
          periodSeconds: 10

        # Liveness probe
        livenessProbe:
          exec:
            command:
            - sqlite3
            - /data/app.db
            - PRAGMA integrity_check
          initialDelaySeconds: 30
          periodSeconds: 60

      volumes:
      - name: sqlite-storage
        persistentVolumeClaim:
          claimName: sqlite-pvc
```

### Container Best Practices

**⚠️ SQLite Container Limitations:**
- **Cannot scale horizontally** (replicas > 1 not supported)
- **Not suitable for network filesystems** (NFS, EFS, etc.)
- **Use local SSD volumes only**
- **No multi-pod deployments**

**Recommended Container Use Cases:**
- Single-instance applications
- Development/testing environments
- Sidecar containers with local storage
- Edge/IoT deployments (single device)

**Alternative for Scale:**
```yaml
# For horizontal scaling, use client-server database
# PostgreSQL, MySQL, or distributed SQLite alternatives
# like LiteFS, rqlite, or Dqlite
```

---

## 18. Use Cases and Limitations

### Ideal Use Cases

**✅ Excellent For:**

1. **Embedded Applications:**
   - Mobile apps (iOS, Android)
   - Desktop applications
   - Browser applications (via WASM)
   - IoT devices and edge computing

2. **Low-to-Medium Traffic Websites:**
   - <100,000 visitors per day
   - <100 requests per second
   - Primarily read-heavy workloads

3. **Development and Testing:**
   - Unit testing (fast, in-memory)
   - Prototyping applications
   - Local development environments

4. **Data Analysis:**
   - Analysis of CSV/JSON files
   - Ad-hoc queries on datasets
   - Reporting and analytics

5. **Cache and Session Storage:**
   - Application cache layer
   - Session management
   - Temporary data storage

6. **Configuration and Metadata:**
   - Application settings
   - User preferences
   - Metadata storage

### Known Limitations

**❌ Not Recommended For:**

1. **High Concurrency Writes:**
   - Only one writer at a time (database-level lock)
   - Concurrent writes will serialize or fail with SQLITE_BUSY

2. **Network File Systems:**
   - **Do not use on NFS, SMB, or network volumes**
   - File locking issues and corruption risk
   - Use local SSD/disk only

3. **Large Datasets:**
   - Practical limit ~1TB (theoretical 140TB)
   - Complex queries slow down on very large tables
   - Better alternatives: PostgreSQL, ClickHouse

4. **High Availability / Replication:**
   - No built-in replication
   - No clustering support
   - Use LiteFS, rqlite, or Dqlite for distributed SQLite

5. **Complex Queries:**
   - Limited query optimizer compared to PostgreSQL
   - No query parallelization
   - No advanced indexing (GiST, GIN, BRIN)

6. **Granular Permissions:**
   - File-level permissions only
   - No user/role management
   - No row-level security

### Performance Characteristics

**Typical Performance:**
```
┌────────────────────────────────────────────┐
│ Operation      │ Queries/Second            │
├────────────────┼───────────────────────────┤
│ Simple SELECT  │ 100,000+ (indexed)        │
│ INSERT (batch) │ 50,000+ (in transaction)  │
│ UPDATE (batch) │ 30,000+ (indexed)         │
│ Complex JOIN   │ 1,000 - 10,000            │
│ Full table     │ Limited by I/O            │
│ scan           │                           │
└────────────────────────────────────────────┘
```

**Database Size Limits:**
- **Maximum database size:** 140 terabytes
- **Maximum row size:** 1 gigabyte
- **Maximum table count:** 2,147,483,646
- **Maximum column count:** 2,000 (can be increased to 32,767)
- **Maximum SQL length:** 1 billion bytes

### Alternative Solutions

**When to Consider Alternatives:**

| Requirement | Alternative |
|-------------|-------------|
| High write concurrency | PostgreSQL, MySQL |
| Distributed/replicated | rqlite, Dqlite, LiteFS |
| Time-series data | InfluxDB, TimescaleDB |
| Analytics/OLAP | ClickHouse, DuckDB |
| Document store | MongoDB, PostgreSQL (JSONB) |
| Full-text search | Elasticsearch, Meilisearch |
| Graph queries | Neo4j, DGraph |

---

## 19. Monitoring and Troubleshooting

### Database Statistics

**Check Database Integrity:**
```sql
-- Full integrity check
PRAGMA integrity_check;

-- Quick integrity check
PRAGMA quick_check;

-- Check for corruption
SELECT * FROM sqlite_dbpage LIMIT 1;
```

**Database Information:**
```sql
-- Database file size and page statistics
SELECT
    page_count * page_size as db_size_bytes,
    page_count * page_size / 1024 / 1024 as db_size_mb,
    page_count,
    page_size,
    freelist_count
FROM pragma_page_count(), pragma_page_size(), pragma_freelist_count();

-- Table sizes
SELECT
    name,
    SUM(pgsize) as size_bytes,
    SUM(pgsize)/1024/1024 as size_mb
FROM dbstat
WHERE name NOT LIKE 'sqlite_%'
GROUP BY name
ORDER BY size_bytes DESC;

-- Index sizes
SELECT
    name,
    tbl_name,
    SUM(pgsize) as size_bytes,
    SUM(pgsize)/1024/1024 as size_mb
FROM dbstat
WHERE name LIKE 'sqlite_autoindex%' OR name IN (
    SELECT name FROM sqlite_master WHERE type='index'
)
GROUP BY name
ORDER BY size_bytes DESC;
```

**Schema Information:**
```sql
-- List all tables
SELECT name FROM sqlite_master WHERE type='table';

-- List all indexes
SELECT name, tbl_name, sql FROM sqlite_master WHERE type='index';

-- List all triggers
SELECT name, tbl_name, sql FROM sqlite_master WHERE type='trigger';

-- Table schema
PRAGMA table_info(users);

-- Index details
PRAGMA index_list(users);
PRAGMA index_info(idx_users_email);
```

### Query Performance Analysis

**SQLite Profiling:**
```python
import sqlite3
import time

class ProfilingConnection:
    def __init__(self, db_path):
        self.conn = sqlite3.connect(db_path)
        self.conn.set_trace_callback(self.trace_callback)

    def trace_callback(self, statement):
        print(f"Executing: {statement}")
        return True

# Usage
conn = ProfilingConnection('app.db')
cursor = conn.conn.execute("SELECT * FROM users WHERE id = 1")
```

**Query Timing:**
```python
def time_query(conn, sql, params=()):
    start = time.perf_counter()
    cursor = conn.execute(sql, params)
    results = cursor.fetchall()
    elapsed = time.perf_counter() - start
    print(f"Query took {elapsed*1000:.2f}ms, returned {len(results)} rows")
    return results
```

### Common Issues and Solutions

**SQLITE_BUSY Error:**
```python
# Symptom: Database is locked
# Cause: Another connection has a write lock

# Solution 1: Increase busy timeout
conn.execute("PRAGMA busy_timeout = 5000")

# Solution 2: Retry with exponential backoff
import time
from sqlite3 import OperationalError

def execute_with_retry(conn, sql, params=(), max_retries=5):
    for attempt in range(max_retries):
        try:
            return conn.execute(sql, params)
        except OperationalError as e:
            if "database is locked" in str(e) and attempt < max_retries - 1:
                time.sleep(0.1 * (2 ** attempt))
            else:
                raise

# Solution 3: Use WAL mode
conn.execute("PRAGMA journal_mode=WAL")
```

**Database Corruption:**
```bash
# Symptom: "database disk image is malformed"

# Solution 1: Recover with .dump
sqlite3 corrupted.db .dump > backup.sql
sqlite3 recovered.db < backup.sql

# Solution 2: Attempt repair with PRAGMA
sqlite3 corrupted.db "PRAGMA integrity_check"
sqlite3 corrupted.db "REINDEX"
sqlite3 corrupted.db "VACUUM"

# Solution 3: Restore from backup
cp backup.db production.db
```

**WAL File Growing:**
```sql
-- Symptom: WAL file grows unbounded

-- Solution: Manual checkpoint
PRAGMA wal_checkpoint(TRUNCATE);

-- Automatic checkpointing
PRAGMA wal_autocheckpoint = 1000;  -- Checkpoint every 1000 pages
```

**Slow Queries:**
```sql
-- Symptom: Queries are slow

-- Diagnosis:
EXPLAIN QUERY PLAN SELECT * FROM users WHERE email = 'test@example.com';

-- Solution 1: Add index
CREATE INDEX idx_users_email ON users(email);

-- Solution 2: Update statistics
ANALYZE;

-- Solution 3: Increase cache
PRAGMA cache_size = -128000;  -- 128MB

-- Solution 4: Vacuum fragmented database
VACUUM;
```

### Logging and Debugging

**Enable Query Logging:**
```python
import sqlite3

# Log all SQL statements
sqlite3.enable_callback_tracebacks(True)

conn = sqlite3.connect('app.db')
conn.set_trace_callback(print)  # Print all SQL statements

# Execute queries
conn.execute("SELECT * FROM users")
```

**Python Error Handling:**
```python
import sqlite3
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

try:
    conn = sqlite3.connect('app.db')
    conn.execute("SELECT * FROM non_existent_table")
except sqlite3.OperationalError as e:
    logger.error(f"Operational error: {e}")
except sqlite3.IntegrityError as e:
    logger.error(f"Integrity constraint violated: {e}")
except sqlite3.DatabaseError as e:
    logger.error(f"Database error: {e}")
finally:
    conn.close()
```

---

## 20. Testing Strategies

### Unit Testing with SQLite

**In-Memory Database for Tests:**
```python
import sqlite3
import unittest

class TestUserModel(unittest.TestCase):
    def setUp(self):
        # Create in-memory database for each test
        self.conn = sqlite3.connect(':memory:')
        self.conn.execute("""
            CREATE TABLE users (
                id INTEGER PRIMARY KEY,
                email TEXT UNIQUE NOT NULL,
                name TEXT
            )
        """)

    def tearDown(self):
        self.conn.close()

    def test_insert_user(self):
        self.conn.execute("INSERT INTO users (email, name) VALUES (?, ?)",
                          ('test@example.com', 'Test User'))
        cursor = self.conn.execute("SELECT * FROM users WHERE email = ?",
                                   ('test@example.com',))
        user = cursor.fetchone()
        self.assertIsNotNone(user)
        self.assertEqual(user[1], 'test@example.com')

    def test_unique_constraint(self):
        self.conn.execute("INSERT INTO users (email, name) VALUES (?, ?)",
                          ('test@example.com', 'Test User'))
        with self.assertRaises(sqlite3.IntegrityError):
            self.conn.execute("INSERT INTO users (email, name) VALUES (?, ?)",
                              ('test@example.com', 'Duplicate User'))

if __name__ == '__main__':
    unittest.main()
```

### Fixture Management

**Pytest Fixtures:**
```python
import pytest
import sqlite3

@pytest.fixture
def db_connection():
    """Create in-memory database for each test"""
    conn = sqlite3.connect(':memory:')
    conn.execute("""
        CREATE TABLE users (
            id INTEGER PRIMARY KEY,
            email TEXT UNIQUE NOT NULL
        )
    """)
    yield conn
    conn.close()

@pytest.fixture
def populated_db(db_connection):
    """Database with test data"""
    db_connection.execute("INSERT INTO users (email) VALUES ('user1@example.com')")
    db_connection.execute("INSERT INTO users (email) VALUES ('user2@example.com')")
    db_connection.commit()
    return db_connection

def test_user_count(populated_db):
    cursor = populated_db.execute("SELECT COUNT(*) FROM users")
    count = cursor.fetchone()[0]
    assert count == 2
```

### Migration Testing

**Test Schema Migrations:**
```python
import unittest
from migrations import DatabaseMigrator

class TestMigrations(unittest.TestCase):
    def test_migration_from_v1_to_v3(self):
        # Create v1 database
        conn = sqlite3.connect(':memory:')
        conn.execute("PRAGMA user_version = 0")

        # Run migrator
        migrator = DatabaseMigrator(':memory:')
        migrator.conn = conn
        migrator.migrate()

        # Verify final version
        version = migrator.get_version()
        self.assertEqual(version, 3)

        # Verify schema
        cursor = conn.execute("PRAGMA table_info(users)")
        columns = [row[1] for row in cursor.fetchall()]
        self.assertIn('updated_at', columns)
```

### Performance Testing

**Benchmark Queries:**
```python
import sqlite3
import time

def benchmark_query(conn, sql, iterations=1000):
    """Benchmark query performance"""
    start = time.perf_counter()
    for _ in range(iterations):
        conn.execute(sql).fetchall()
    elapsed = time.perf_counter() - start

    print(f"Query: {sql}")
    print(f"Iterations: {iterations}")
    print(f"Total time: {elapsed:.2f}s")
    print(f"Avg time: {elapsed/iterations*1000:.2f}ms")
    print(f"Queries/sec: {iterations/elapsed:.0f}")

# Usage
conn = sqlite3.connect('test.db')
benchmark_query(conn, "SELECT * FROM users WHERE id = 1")
```

### Mock Database for Testing

**Database Abstraction Layer:**
```python
class DatabaseInterface:
    def get_user(self, user_id):
        raise NotImplementedError

class SQLiteDatabase(DatabaseInterface):
    def __init__(self, db_path):
        self.conn = sqlite3.connect(db_path)

    def get_user(self, user_id):
        cursor = self.conn.execute("SELECT * FROM users WHERE id = ?", (user_id,))
        return cursor.fetchone()

class MockDatabase(DatabaseInterface):
    def __init__(self):
        self.users = {1: ('test@example.com', 'Test User')}

    def get_user(self, user_id):
        return self.users.get(user_id)

# Tests use MockDatabase
def test_user_service():
    db = MockDatabase()
    user = db.get_user(1)
    assert user is not None
```

---

## 21. Version-Specific Features

### SQLite 3.45 (2024)

**JSON Improvements:**
```sql
-- Enhanced JSON functions
SELECT json_patch('{"a":1}', '{"b":2}');  -- {"a":1,"b":2}

-- JSON array aggregation with ordering
SELECT json_group_array(name ORDER BY created_at DESC) FROM users;
```

### SQLite 3.38-3.44

**RETURNING Clause (3.35+):**
```sql
-- Return inserted rows
INSERT INTO users (email, name) VALUES ('test@example.com', 'Test')
RETURNING id, created_at;

-- Return updated rows
UPDATE users SET name = 'Updated' WHERE id = 1
RETURNING *;

-- Return deleted rows
DELETE FROM users WHERE id = 1
RETURNING email;
```

**DROP COLUMN (3.35+):**
```sql
ALTER TABLE users DROP COLUMN phone;
```

**Materialized CTEs (3.35+):**
```sql
-- Force CTE materialization
WITH MATERIALIZED cte AS (
    SELECT * FROM large_table WHERE expensive_condition
)
SELECT * FROM cte WHERE additional_filter;
```

**STRICT Tables (3.37+):**
```sql
-- Enforce strict type checking
CREATE TABLE users (
    id INTEGER PRIMARY KEY,
    email TEXT NOT NULL,
    age INTEGER
) STRICT;

-- This will fail:
INSERT INTO users (email, age) VALUES ('test@example.com', 'twenty');  -- Error
```

**Math Functions (3.35+):**
```sql
SELECT
    ceil(4.3),      -- 5
    floor(4.8),     -- 4
    trunc(4.8),     -- 4
    ln(2.718),      -- 1.0
    log10(100),     -- 2.0
    exp(1),         -- 2.718
    power(2, 3),    -- 8
    sqrt(16),       -- 4
    sin(0),         -- 0
    cos(0);         -- 1
```

### SQLite 3.25-3.37

**Window Functions (3.25+):**
```sql
-- Running total
SELECT
    date,
    amount,
    SUM(amount) OVER (ORDER BY date) as running_total
FROM transactions;

-- Row number
SELECT
    ROW_NUMBER() OVER (PARTITION BY category ORDER BY price DESC) as rank,
    name,
    category,
    price
FROM products;

-- Moving average
SELECT
    date,
    value,
    AVG(value) OVER (
        ORDER BY date
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ) as moving_avg_7day
FROM metrics;
```

**RENAME COLUMN (3.25+):**
```sql
ALTER TABLE users RENAME COLUMN name TO full_name;
```

**UPSERT (3.24+):**
```sql
INSERT INTO users (id, email, login_count)
VALUES (1, 'user@example.com', 1)
ON CONFLICT(id) DO UPDATE SET
    login_count = login_count + 1,
    last_login = CURRENT_TIMESTAMP;
```

### Compilation Options

**Check SQLite Version and Features:**
```sql
SELECT sqlite_version();          -- e.g., '3.45.0'
SELECT sqlite_source_id();        -- Commit hash and date
```

**Check Compile Options:**
```sql
PRAGMA compile_options;

-- Common options to check:
-- ENABLE_FTS5
-- ENABLE_RTREE
-- ENABLE_JSON1
-- ENABLE_STAT4
-- THREADSAFE
-- ENABLE_COLUMN_METADATA
```

**Python Check:**
```python
import sqlite3
print(f"SQLite version: {sqlite3.sqlite_version}")
print(f"Python sqlite3 version: {sqlite3.version}")

# Check for specific feature
conn = sqlite3.connect(':memory:')
try:
    conn.execute("CREATE VIRTUAL TABLE test USING fts5(content)")
    print("FTS5 is available")
except sqlite3.OperationalError:
    print("FTS5 is NOT available")
```

---

## 22. Deployment Checklist

### Build and Configuration
- [ ] SQLite version pinned and documented
- [ ] WAL mode enabled (`PRAGMA journal_mode=WAL`)
- [ ] Foreign keys enabled (`PRAGMA foreign_keys=ON`)
- [ ] Busy timeout set (`PRAGMA busy_timeout=5000`)
- [ ] Synchronous mode set appropriately (`NORMAL` for WAL mode)
- [ ] Page size optimized for workload (`PRAGMA page_size=4096` or `8192`)
- [ ] `mmap_size` configured for read-heavy workloads

### Testing
- [ ] All queries profiled with `EXPLAIN QUERY PLAN`
- [ ] Indexes verified for all frequent query patterns
- [ ] Concurrent write behavior tested under expected load
- [ ] Database integrity verified with `PRAGMA integrity_check`
- [ ] Backup procedure tested with `.backup` command or SQLite Online Backup API
- [ ] Maximum database size validated for deployment target

### Security
- [ ] Database file permissions restricted (0600 or equivalent)
- [ ] WAL and SHM file permissions match database file
- [ ] No database files served via public web paths
- [ ] SQLite Encryption Extension (SEE) or SQLCipher configured if encryption required
- [ ] Parameterized queries used exclusively (no string interpolation)
- [ ] `SQLITE_DBCONFIG_DEFENSIVE` enabled to prevent corruption via SQL

### Agent Workflow
- [ ] Schema migration scripts version-controlled and sequential
- [ ] Application handles `SQLITE_BUSY` gracefully with retry logic
- [ ] Monitoring configured for database file size and WAL checkpoint frequency
- [ ] Automated backups scheduled with Litestream or application-level backup
- [ ] Runbooks for database recovery, WAL reset, and corruption handling

---

## 23. Why This Configuration Works

**WAL Mode for Concurrency**:
- Write-ahead logging allows concurrent readers during writes, eliminating reader-writer contention and providing significantly higher throughput for mixed read-write workloads.

**Zero-Configuration Deployment**:
- As a serverless embedded database, SQLite requires no separate process management, network configuration, or user authentication setup, reducing operational complexity to near zero.

**ACID Compliance with Full Durability**:
- Atomic commits, rollback journals or WAL, and configurable synchronous modes ensure data integrity even during power failures or crashes without sacrificing performance.

**Single-File Portability**:
- The entire database lives in a single cross-platform file, enabling simple backups (file copy), easy testing (in-memory databases), and straightforward deployment across any environment.

**Extensive SQL Feature Coverage**:
- Support for window functions, CTEs, JSON functions, FTS5 full-text search, and R-tree spatial indexes provides relational database capabilities without server overhead.

---

## 24. Quick Reference

### Common Commands

```bash
# Open database
sqlite3 mydb.db

# Enable WAL mode
sqlite3 mydb.db "PRAGMA journal_mode=WAL;"

# Check database integrity
sqlite3 mydb.db "PRAGMA integrity_check;"

# Analyze tables for query optimizer
sqlite3 mydb.db "ANALYZE;"

# Backup database
sqlite3 mydb.db ".backup /path/to/backup.db"

# Dump database to SQL
sqlite3 mydb.db ".dump" > backup.sql

# Restore from SQL dump
sqlite3 newdb.db < backup.sql

# Show tables and schema
sqlite3 mydb.db ".tables"
sqlite3 mydb.db ".schema tablename"

# Explain query plan
sqlite3 mydb.db "EXPLAIN QUERY PLAN SELECT * FROM users WHERE email = 'test@example.com';"

# Check database size
sqlite3 mydb.db "SELECT page_count * page_size AS size FROM pragma_page_count(), pragma_page_size();"

# Vacuum database to reclaim space
sqlite3 mydb.db "VACUUM;"

# Checkpoint WAL
sqlite3 mydb.db "PRAGMA wal_checkpoint(TRUNCATE);"
```

---

## References and Resources

### Official Documentation
- **SQLite Official Website:** https://www.sqlite.org/
- **SQLite Documentation:** https://www.sqlite.org/docs.html
- **SQL Syntax:** https://www.sqlite.org/lang.html
- **PRAGMA Statements:** https://www.sqlite.org/pragma.html

### Performance and Optimization
- **Query Planning:** https://www.sqlite.org/queryplanner.html
- **Optimization FAQ:** https://www.sqlite.org/faq.html#q19
- **Benchmarking:** https://www.sqlite.org/speed.html

### Books and Guides
- "The Definitive Guide to SQLite" by Grant Allen and Mike Owens
- "Using SQLite" by Jay A. Kreibich
- SQLite official documentation (most comprehensive)

### Tools
- **SQLite Browser:** https://sqlitebrowser.org/
- **Litestream:** Real-time SQLite replication
- **LiteFS:** Distributed SQLite
- **rqlite:** Distributed SQLite with Raft consensus

### Community
- SQLite Forum: https://sqlite.org/forum/
- Stack Overflow: `[sqlite]` tag
- SQLite Discord/Slack communities

---

**Document Maintenance:**
- Review quarterly for SQLite updates
- Update benchmarks with new hardware/versions
- Add community-discovered best practices
- Verify container deployment patterns

**Last Updated:** February 2026
**Next Review:** May 2026

---

**End of SQLite Development Guidelines**
