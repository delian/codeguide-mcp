# Couchbase Development Guidelines
Mandatory coding standards and development practices for Couchbase development. Couchbase Server 7.x+, N1QL, SDKs (Java/Python/Node.js), XDCR, Sync Gateway.

---

**Agent Profile**: The Couchbase Expert
**Role**: Senior NoSQL/Document DB Engineer & N1QL Specialist
**Objective**: Generate production-ready, performant and scalable document and cache solutions.
**Tools**: Couchbase Server 7.x+, N1QL, SDKs (Java/Python/Node.js), XDCR, Sync Gateway

---

**Version:** 1.0 | **Last Updated:** February 2026 | **Target Version:** Couchbase Server 7.x+

## Table of Contents

1. [Core Philosophies: DOCUMENT-FIRST](#1-core-philosophies-document-first)
2. [Architecture and Fundamentals](#2-architecture-and-fundamentals)
3. [Document Model and Data Modeling](#3-document-model-and-data-modeling)
4. [N1QL Query Language](#4-n1ql-query-language)
5. [Key-Value Operations](#5-key-value-operations)
6. [Indexes and Query Performance](#6-indexes-and-query-performance)
7. [Full-Text Search](#7-full-text-search)
8. [Caching and Memory Management](#8-caching-and-memory-management)
9. [Clustering and Scaling](#9-clustering-and-scaling)
10. [Cross-Datacenter Replication (XDCR)](#10-cross-datacenter-replication-xdcr)
11. [Security and Access Control](#11-security-and-access-control)
12. [Backup and Recovery](#12-backup-and-recovery)
13. [Monitoring and Troubleshooting](#13-monitoring-and-troubleshooting)
14. [Mobile and Edge Sync](#14-mobile-and-edge-sync)
15. [Analytics Service](#15-analytics-service)
16. [Eventing Service](#16-eventing-service)
17. [Application Integration](#17-application-integration)
18. [Production Deployment](#18-production-deployment)
19. [Performance Optimization](#19-performance-optimization)
20. [Migration Strategies](#20-migration-strategies)
21. [Production Checklist](#21-production-checklist)

---

## 1. Core Philosophies: DOCUMENT-FIRST

The agent must adhere to the **DOCUMENT-FIRST** principles for every Couchbase implementation:

**Test-Driven Development (TDD)**: ALWAYS write tests BEFORE implementation (Red-Green-Refactor cycle mandatory).
**Regression Shield**: EVERY bug discovered MUST receive a test BEFORE fixing to prevent regression.

- **D**ocument model: Design around JSON documents and key-value access; use N1QL for querying.
- **O**perations: Prefer key-value when possible; use N1QL and indexes for complex queries.
- **C**luster awareness: Design for multi-node and XDCR; avoid single-node assumptions.
- **U**se indexes: Create and maintain indexes required by N1QL; monitor index usage.
- **M**emory and cache: Respect memory-first architecture; size buckets and caches appropriately.
- **E**rror handling: Use SDK retry and backoff; handle transient and replication failures.
- **N**1QL best practices: Parameterize queries; use EXPLAIN; avoid full bucket scans.
- **T**esting: Test with representative documents and cluster topology.

**Verified Code**: Agent-generated code MUST use parameterized N1QL, run against a cluster or mock, and pass tests before delivery.

---

## 2. Architecture and Fundamentals

### What is Couchbase?

**Couchbase Server** is a distributed NoSQL document database with integrated caching:

- ✅ **JSON documents** (flexible schema)
- ✅ **Memory-first architecture** (sub-millisecond operations)
- ✅ **N1QL SQL queries** (SQL for JSON)
- ✅ **Built-in cache** (Memcached compatible)
- ✅ **Multi-dimensional scaling** (independent service scaling)
- ✅ **XDCR** (cross-datacenter replication)
- ✅ **Mobile sync** (Couchbase Lite + Sync Gateway)
- ✅ **Full-text search** (integrated search engine)
- ✅ **Analytics** (separate analytics service)

### Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│          Couchbase Server Architecture               │
├─────────────────────────────────────────────────────┤
│                                                      │
│  Client Applications                                 │
│  ┌────────────────────────────────────────────┐    │
│  │  SDK (Java, Python, Node.js, etc.)         │    │
│  └────────────────────────────────────────────┘    │
│                      │                              │
│                      ▼                              │
│  ┌────────────────────────────────────────────┐    │
│  │         Cluster Manager                    │    │
│  │  - Orchestration                           │    │
│  │  - Service management                      │    │
│  └────────────────────────────────────────────┘    │
│                      │                              │
│                      ▼                              │
│  ┌────────────────────────────────────────────┐    │
│  │         Services (MDS)                     │    │
│  │  ┌─────────────────────────────────────┐  │    │
│  │  │ Data Service                        │  │    │
│  │  │ - vBuckets (1024 per bucket)       │  │    │
│  │  │ - Managed cache                     │  │    │
│  │  │ - DCP (replication protocol)       │  │    │
│  │  └─────────────────────────────────────┘  │    │
│  │                                             │    │
│  │  ┌─────────────────────────────────────┐  │    │
│  │  │ Query Service (N1QL)                │  │    │
│  │  │ - SQL++ queries                     │  │    │
│  │  │ - Query optimizer                   │  │    │
│  │  └─────────────────────────────────────┘  │    │
│  │                                             │    │
│  │  ┌─────────────────────────────────────┐  │    │
│  │  │ Index Service                       │  │    │
│  │  │ - Global secondary indexes (GSI)   │  │    │
│  │  │ - Memory-optimized indexes          │  │    │
│  │  └─────────────────────────────────────┘  │    │
│  │                                             │    │
│  │  ┌─────────────────────────────────────┐  │    │
│  │  │ Search Service (FTS)                │  │    │
│  │  │ - Full-text search                  │  │    │
│  │  │ - Bleve search engine               │  │    │
│  │  └─────────────────────────────────────┘  │    │
│  │                                             │    │
│  │  ┌─────────────────────────────────────┐  │    │
│  │  │ Analytics Service                   │  │    │
│  │  │ - Analytical queries (OLAP)         │  │    │
│  │  │ - Separate from operational data    │  │    │
│  │  └─────────────────────────────────────┘  │    │
│  │                                             │    │
│  │  ┌─────────────────────────────────────┐  │    │
│  │  │ Eventing Service                    │  │    │
│  │  │ - Functions-as-a-service            │  │    │
│  │  │ - Data change triggers              │  │    │
│  │  └─────────────────────────────────────┘  │    │
│  └────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────┘
```

### Key Concepts

**Buckets:**
```
Bucket = Database
- Container for documents
- Independent namespaces
- Configurable memory quota
- Replication settings

Types:
1. Couchbase (default) - Full features
2. Ephemeral - Memory-only, no persistence
3. Memcached - Cache-only, no indexing
```

**vBuckets:**
```
Each bucket divided into 1024 vBuckets
Documents distributed across vBuckets by hash

Document ID → Hash → vBucket (0-1023) → Node

Benefits:
- Even distribution
- Easy rebalancing
- No coordination overhead
```

**Multi-Dimensional Scaling (MDS):**
```
Services can run on different nodes:

Node 1: Data + Index
Node 2: Data + Query
Node 3: Query + Search
Node 4: Analytics + Eventing

Benefits:
- Independent scaling
- Resource isolation
- Performance optimization
```

**Memory-First Architecture:**
```
Working Set:
┌─────────────────────────────┐
│   Managed Cache (RAM)        │ ← Active data
│   - Fast reads/writes        │
│   - LRU eviction             │
└─────────────────────────────┘
            ↕
┌─────────────────────────────┐
│   Persistent Storage (SSD)   │ ← All data
│   - Durability               │
│   - Recovery                 │
└─────────────────────────────┘
```

### When to Use Couchbase

**✅ Excellent For:**

1. **High-Performance Applications:**
   - Sub-millisecond latency required
   - High throughput (100K+ ops/sec)
   - Real-time applications
   - Gaming, AdTech, IoT

2. **Caching + Persistence:**
   - Need both cache and database
   - Replace Memcached + Database
   - Unified platform

3. **Distributed Applications:**
   - Multi-datacenter deployment
   - Global distribution
   - Active-active replication (XDCR)

4. **Mobile/Edge Sync:**
   - Offline-first mobile apps
   - Edge computing
   - Couchbase Lite + Sync Gateway

5. **Flexible Schema:**
   - Evolving data models
   - JSON documents
   - Mixed workloads (KV + SQL)

**❌ Not Recommended For:**

1. **Complex Transactions:**
   - Multi-document ACID
   - Use PostgreSQL, CockroachDB

2. **Heavy Analytics:**
   - OLAP workloads
   - Use ClickHouse, Snowflake
   - (Though Analytics Service helps)

3. **Graph Queries:**
   - Complex relationships
   - Use Neo4j

4. **Document Relationships:**
   - Heavy JOIN operations
   - Use relational database

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

### Example TDD Workflow for Couchbase (Python with pytest and Couchbase SDK)

```python
# Step 1: RED - Write failing test first
import pytest
from couchbase.cluster import Cluster
from couchbase.options import ClusterOptions
from couchbase.auth import PasswordAuthenticator

@pytest.fixture
def cb_collection():
    cluster = Cluster(
        "couchbase://localhost",
        ClusterOptions(PasswordAuthenticator("admin", "password"))
    )
    bucket = cluster.bucket("test_bucket")
    collection = bucket.default_collection()
    yield collection
    cluster.close()

def test_upsert_and_get_document(cb_collection):
    """Test upserting and retrieving a JSON document."""
    repo = UserRepository(cb_collection)
    repo.upsert_user("user::1001", {"name": "Alice", "email": "alice@example.com"})
    user = repo.get_user("user::1001")
    assert user["name"] == "Alice"
    assert user["email"] == "alice@example.com"

# Run: pytest test_couchbase.py -v
# FAILS - NameError: name 'UserRepository' is not defined

# Step 2: GREEN - Write minimal implementation
class UserRepository:
    def __init__(self, collection):
        self.collection = collection

    def upsert_user(self, doc_id, user_data):
        self.collection.upsert(doc_id, user_data)

    def get_user(self, doc_id):
        result = self.collection.get(doc_id)
        return result.content_as[dict]

# Run: pytest test_couchbase.py -v
# PASSES

# Step 3: REFACTOR - Add N1QL query support and type field
from couchbase.cluster import QueryOptions

class UserRepository:
    def __init__(self, collection, cluster=None):
        self.collection = collection
        self.cluster = cluster

    def upsert_user(self, doc_id, user_data):
        user_data["type"] = "user"
        self.collection.upsert(doc_id, user_data)

    def get_user(self, doc_id):
        result = self.collection.get(doc_id)
        return result.content_as[dict]

    def find_users_by_email(self, email):
        query = "SELECT META().id, * FROM test_bucket WHERE type = 'user' AND email = $email"
        result = self.cluster.query(query, QueryOptions(named_parameters={"email": email}))
        return [row for row in result]

# Tests still pass
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
6. Document the bug in test comments
```

### Example Bug Fix

```python
# Bug: get_user() raises DocumentNotFoundException for missing documents
# instead of returning None, causing unhandled crashes in the API layer

import pytest
from couchbase.exceptions import DocumentNotFoundException

# Step 1: Write test that reproduces the bug
def test_get_missing_user_returns_none(cb_collection):
    """Regression: get_user() should return None for missing documents,
    not raise DocumentNotFoundException."""
    repo = UserRepository(cb_collection)
    result = repo.get_user("user::nonexistent")
    assert result is None

# FAILS - DocumentNotFoundException is raised

# Step 2: Fix the bug
class UserRepository:
    # ... existing code ...

    def get_user(self, doc_id):
        try:
            result = self.collection.get(doc_id)
            return result.content_as[dict]
        except DocumentNotFoundException:
            return None

# PASSES - bug fixed, regression prevented
```

---

## 3. Document Model and Data Modeling

### Document Structure

**Basic Document:**
```json
{
  "type": "user",
  "id": "user::alice",
  "name": "Alice Smith",
  "email": "alice@example.com",
  "age": 30,
  "address": {
    "street": "123 Main St",
    "city": "Springfield",
    "zip": "12345"
  },
  "orders": ["order::123", "order::456"],
  "created_at": "2024-02-06T10:00:00Z",
  "updated_at": "2024-02-06T10:00:00Z"
}
```

**Document Key Design:**
```
Strategies:

1. Semantic Keys (Recommended)
   user::alice
   order::12345
   product::widget

2. UUID
   550e8400-e29b-41d4-a716-446655440000

3. Composite Keys
   user::alice::profile
   order::2024::12345

4. Type Prefix (for queries)
   user_alice
   order_12345

Best Practice:
- Use :: as delimiter
- Include type for filtering
- Human-readable preferred
```

### Data Modeling Patterns

**Embedding vs. Referencing:**
```json
// Embedding (denormalized)
{
  "type": "user",
  "id": "user::alice",
  "name": "Alice",
  "orders": [
    {
      "id": "order::123",
      "date": "2024-01-01",
      "total": 100.00
    }
  ]
}

// ✅ Good for: Frequently accessed together
// ❌ Bad for: Large arrays, independent updates

// Referencing (normalized)
{
  "type": "user",
  "id": "user::alice",
  "name": "Alice",
  "order_ids": ["order::123", "order::456"]
}

{
  "type": "order",
  "id": "order::123",
  "user_id": "user::alice",
  "total": 100.00
}

// ✅ Good for: Independent updates, large data
// ❌ Bad for: Requires multiple lookups
```

**Type Field Pattern:**
```json
{
  "type": "user",  // Essential for queries
  "id": "user::alice",
  "name": "Alice"
}

// Enables type-specific queries
SELECT * FROM bucket WHERE type = "user"
```

**Subdocuments:**
```json
{
  "type": "user",
  "id": "user::alice",
  "profile": {
    "bio": "Software engineer",
    "avatar": "https://..."
  },
  "settings": {
    "theme": "dark",
    "notifications": true
  },
  "metadata": {
    "created_at": "2024-01-01",
    "updated_at": "2024-02-06"
  }
}

// Access subdocuments efficiently
bucket.lookupIn("user::alice").get("profile.bio").execute()
```

### Document Size and Limits

**Size Limits:**
```
Maximum document size: 20MB
Recommended size: < 1MB
Subdocument operations: More efficient for large docs

If document > 1MB:
- Split into multiple documents
- Use references
- Store binary data in S3/external storage
```

**Array Limits:**
```json
// ❌ BAD: Unbounded array growth
{
  "type": "user",
  "id": "user::alice",
  "activity_log": [
    // 100,000+ entries
  ]
}

// ✅ GOOD: Separate documents
{
  "type": "user",
  "id": "user::alice"
}

{
  "type": "activity",
  "id": "activity::alice::2024-02-06",
  "user_id": "user::alice",
  "date": "2024-02-06",
  "events": [
    // Today's events only
  ]
}
```

### Document Metadata

**System Metadata:**
```json
// Couchbase automatically maintains:
{
  "_id": "user::alice",        // Document key
  "_cas": 1707217200000000,    // Compare-and-swap (versioning)
  "_expiration": 1707303600,   // TTL (0 = no expiration)
  "_flags": 0,                 // SDK flags
  "_type": "json"              // Document type
}

// Access via SDK
result = bucket.get("user::alice")
cas = result.cas()  // For optimistic locking
```

---

## 4. N1QL Query Language

### Basic Queries

**SELECT:**
```sql
-- Simple SELECT
SELECT * FROM bucket WHERE type = "user";

-- Specific fields
SELECT name, email, age
FROM bucket
WHERE type = "user" AND age > 25;

-- With aliases
SELECT u.name AS userName, u.email
FROM bucket u
WHERE u.type = "user";

-- LIMIT and OFFSET
SELECT * FROM bucket
WHERE type = "user"
ORDER BY created_at DESC
LIMIT 10 OFFSET 20;
```

**WHERE Clauses:**
```sql
-- Comparison operators
SELECT * FROM bucket
WHERE age >= 18 AND age < 65;

-- IN operator
SELECT * FROM bucket
WHERE status IN ["active", "pending"];

-- LIKE operator
SELECT * FROM bucket
WHERE email LIKE "%@example.com";

-- IS NULL / IS NOT NULL
SELECT * FROM bucket
WHERE phone IS NOT NULL;

-- BETWEEN
SELECT * FROM bucket
WHERE created_at BETWEEN "2024-01-01" AND "2024-12-31";
```

**Nested Properties:**
```sql
-- Access nested fields
SELECT name, address.city, address.zip
FROM bucket
WHERE type = "user" AND address.city = "Springfield";

-- Array contains
SELECT * FROM bucket
WHERE type = "user"
  AND ANY tag IN tags SATISFIES tag = "premium" END;
```

### JOIN Operations

**Types of JOINs:**
```sql
-- INNER JOIN
SELECT u.name, o.total
FROM bucket u
INNER JOIN bucket o ON KEYS u.order_ids
WHERE u.type = "user" AND o.type = "order";

-- LEFT JOIN
SELECT u.name, o.total
FROM bucket u
LEFT JOIN bucket o ON KEYS u.order_ids
WHERE u.type = "user";

-- INDEX JOIN (most efficient)
SELECT u.name, o.total
FROM bucket u
JOIN bucket o ON KEY o.user_id FOR u
WHERE u.type = "user" AND o.type = "order";

-- UNNEST (flatten arrays)
SELECT u.name, tag
FROM bucket u
UNNEST u.tags AS tag
WHERE u.type = "user";
```

### Aggregations

**Aggregate Functions:**
```sql
-- COUNT
SELECT COUNT(*) AS total_users
FROM bucket
WHERE type = "user";

-- SUM, AVG, MIN, MAX
SELECT
  AVG(age) AS average_age,
  MIN(age) AS youngest,
  MAX(age) AS oldest,
  SUM(age) AS total_age
FROM bucket
WHERE type = "user";

-- GROUP BY
SELECT status, COUNT(*) AS count
FROM bucket
WHERE type = "order"
GROUP BY status
ORDER BY count DESC;

-- HAVING
SELECT city, COUNT(*) AS user_count
FROM bucket
WHERE type = "user"
GROUP BY address.city
HAVING COUNT(*) > 10;
```

**Array Aggregation:**
```sql
-- ARRAY_AGG (collect into array)
SELECT user_id, ARRAY_AGG(order_id) AS orders
FROM bucket
WHERE type = "order"
GROUP BY user_id;

-- Aggregate array elements
SELECT
  AVG(price) AS avg_price,
  SUM(quantity) AS total_quantity
FROM bucket
UNNEST items AS item
WHERE type = "order";
```

### Advanced Queries

**Subqueries:**
```sql
-- Subquery in WHERE
SELECT * FROM bucket
WHERE type = "user"
  AND id IN (
    SELECT DISTINCT user_id FROM bucket WHERE type = "order"
  );

-- Correlated subquery
SELECT u.name,
  (SELECT COUNT(*) FROM bucket o
   WHERE o.type = "order" AND o.user_id = u.id) AS order_count
FROM bucket u
WHERE u.type = "user";
```

**Common Table Expressions (CTE):**
```sql
-- WITH clause
WITH active_users AS (
  SELECT * FROM bucket
  WHERE type = "user" AND status = "active"
)
SELECT au.name, COUNT(o.id) AS order_count
FROM active_users au
LEFT JOIN bucket o ON KEY o.user_id FOR au
WHERE o.type = "order"
GROUP BY au.name;
```

**Window Functions:**
```sql
-- ROW_NUMBER
SELECT name, salary,
  ROW_NUMBER() OVER (PARTITION BY department ORDER BY salary DESC) AS rank
FROM bucket
WHERE type = "employee";

-- RANK, DENSE_RANK
SELECT name, score,
  RANK() OVER (ORDER BY score DESC) AS rank,
  DENSE_RANK() OVER (ORDER BY score DESC) AS dense_rank
FROM bucket
WHERE type = "player";
```

### DML Operations

**INSERT:**
```sql
-- Insert single document
INSERT INTO bucket (KEY, VALUE)
VALUES ("user::bob", {
  "type": "user",
  "name": "Bob",
  "email": "bob@example.com"
});

-- Insert multiple
INSERT INTO bucket (KEY, VALUE)
VALUES
  ("user::carol", {"type": "user", "name": "Carol"}),
  ("user::dave", {"type": "user", "name": "Dave"});
```

**UPDATE:**
```sql
-- Update documents
UPDATE bucket
SET age = 31, updated_at = NOW_STR()
WHERE type = "user" AND name = "Alice";

-- Conditional update
UPDATE bucket
SET status = "inactive"
WHERE type = "user" AND last_login < "2023-01-01";

-- Update nested field
UPDATE bucket
SET address.city = "New York"
WHERE type = "user" AND id = "user::alice";
```

**DELETE:**
```sql
-- Delete documents
DELETE FROM bucket
WHERE type = "user" AND status = "deleted";

-- Delete with limit
DELETE FROM bucket
WHERE type = "temp_data" AND created_at < "2024-01-01"
LIMIT 1000;
```

**UPSERT:**
```sql
-- Insert or update
UPSERT INTO bucket (KEY, VALUE)
VALUES ("user::alice", {
  "type": "user",
  "name": "Alice Smith",
  "email": "alice@example.com"
});
```

---

## 5. Key-Value Operations

### Basic KV Operations

**GET:**
```python
from couchbase.cluster import Cluster
from couchbase.auth import PasswordAuthenticator
from couchbase.options import ClusterOptions

# Connect
cluster = Cluster('couchbase://localhost',
  ClusterOptions(PasswordAuthenticator('Administrator', 'password')))
bucket = cluster.bucket('mybucket')
collection = bucket.default_collection()

# Get document
result = collection.get('user::alice')
doc = result.content_as[dict]
print(doc['name'])  # Alice

# Get with options
result = collection.get('user::alice', timeout=timedelta(seconds=5))
```

**INSERT:**
```python
# Insert document (fails if exists)
doc = {
  'type': 'user',
  'name': 'Bob',
  'email': 'bob@example.com'
}
collection.insert('user::bob', doc)
```

**UPSERT:**
```python
# Upsert (insert or update)
doc = {
  'type': 'user',
  'name': 'Carol',
  'email': 'carol@example.com'
}
collection.upsert('user::carol', doc)
```

**REPLACE:**
```python
# Replace existing document
result = collection.get('user::alice')
doc = result.content_as[dict]
doc['age'] = 31

# Replace with CAS (optimistic locking)
collection.replace('user::alice', doc, cas=result.cas)
```

**REMOVE:**
```python
# Delete document
collection.remove('user::alice')

# Delete with CAS
result = collection.get('user::alice')
collection.remove('user::alice', cas=result.cas)
```

### Subdocument Operations

**lookupIn (Partial Get):**
```python
from couchbase.subdocument import get

# Get specific fields
result = collection.lookup_in('user::alice', [
  get('name'),
  get('email'),
  get('address.city')
])

name = result.content_as[str](0)
email = result.content_as[str](1)
city = result.content_as[str](2)
```

**mutateIn (Partial Update):**
```python
from couchbase.subdocument import upsert, insert, array_append

# Update specific fields
collection.mutate_in('user::alice', [
  upsert('age', 31),
  upsert('updated_at', '2024-02-06T10:00:00Z')
])

# Array operations
collection.mutate_in('user::alice', [
  array_append('tags', 'premium', create_parents=True)
])

# Insert if not exists
collection.mutate_in('user::alice', [
  insert('phone', '+1-555-1234', create_parents=True)
])
```

### Atomic Counters

**Counter Operations:**
```python
from couchbase.subdocument import increment, decrement

# Increment counter
result = collection.mutate_in('counter::page_views', [
  increment('count', 1)
])
new_count = result.content_as[int](0)

# Decrement counter
collection.mutate_in('inventory::widget', [
  decrement('stock', 1)
])

# Initialize counter if not exists
collection.upsert('counter::visits', {'count': 0})
```

### Bulk Operations

**Batch Get:**
```python
from couchbase.options import GetMultiOptions

# Get multiple documents
keys = ['user::alice', 'user::bob', 'user::carol']
results = collection.get_multi(keys)

for key, result in results.items():
  if result.success:
    doc = result.content_as[dict]
    print(f"{key}: {doc['name']}")
```

**Batch Upsert:**
```python
# Upsert multiple documents
docs = {
  'user::alice': {'type': 'user', 'name': 'Alice'},
  'user::bob': {'type': 'user', 'name': 'Bob'}
}

results = collection.upsert_multi(docs)
for key, result in results.items():
  if result.success:
    print(f"{key} upserted")
```

### TTL (Time To Live)

**Document Expiration:**
```python
from datetime import timedelta

# Set TTL (expires in 1 hour)
collection.upsert('session::abc123',
  {'user_id': 'user::alice', 'data': '...'},
  expiry=timedelta(hours=1)
)

# Set TTL in seconds
collection.upsert('cache::key', {'value': '...'},
  expiry=3600  # 1 hour in seconds
)

# Get with touch (refresh TTL)
result = collection.get_and_touch('session::abc123',
  expiry=timedelta(hours=1)
)
```

---

## 6. Indexes and Query Performance

### Index Types

**Primary Index:**
```sql
-- Create primary index (scans all documents)
CREATE PRIMARY INDEX ON bucket;

-- Drop primary index
DROP PRIMARY INDEX ON bucket;

-- Named primary index
CREATE PRIMARY INDEX idx_primary ON bucket;
```

**Secondary Index (GSI):**
```sql
-- Single field index
CREATE INDEX idx_type ON bucket(type);

-- Composite index
CREATE INDEX idx_user_status ON bucket(type, status)
WHERE type = "user";

-- Covering index (includes all queried fields)
CREATE INDEX idx_user_email ON bucket(type, email, name)
WHERE type = "user";

-- Index with ORDER BY
CREATE INDEX idx_user_created ON bucket(created_at DESC)
WHERE type = "user";
```

**Partial Index:**
```sql
-- Index only active users
CREATE INDEX idx_active_users ON bucket(email, name)
WHERE type = "user" AND status = "active";

-- Index only recent orders
CREATE INDEX idx_recent_orders ON bucket(created_at, total)
WHERE type = "order" AND created_at > "2024-01-01";
```

**Array Index:**
```sql
-- Index array elements
CREATE INDEX idx_tags ON bucket(DISTINCT ARRAY tag FOR tag IN tags END)
WHERE type = "user";

-- Query with array index
SELECT * FROM bucket
WHERE type = "user"
  AND ANY tag IN tags SATISFIES tag = "premium" END;
```

**Functional Index:**
```sql
-- Index on expression
CREATE INDEX idx_email_lower ON bucket(LOWER(email))
WHERE type = "user";

-- Query using expression
SELECT * FROM bucket
WHERE type = "user" AND LOWER(email) = "alice@example.com";
```

### Index Management

**List Indexes:**
```sql
-- View all indexes
SELECT * FROM system:indexes WHERE keyspace_id = "bucket";

-- Check index status
SELECT name, state, index_key
FROM system:indexes
WHERE keyspace_id = "bucket";
```

**Build Deferred Indexes:**
```sql
-- Create indexes with defer_build
CREATE INDEX idx_user_1 ON bucket(field1) WITH {"defer_build": true};
CREATE INDEX idx_user_2 ON bucket(field2) WITH {"defer_build": true};

-- Build all deferred indexes at once
BUILD INDEX ON bucket(idx_user_1, idx_user_2);
```

**Drop Index:**
```sql
DROP INDEX bucket.idx_user_status;
```

### Query Optimization

**EXPLAIN:**
```sql
-- View query plan
EXPLAIN SELECT * FROM bucket
WHERE type = "user" AND email = "alice@example.com";

-- Look for:
-- - IntersectScan (good - uses indexes)
-- - PrimaryScan (bad - full table scan)
-- - Index usage
```

**USE INDEX:**
```sql
-- Force specific index
SELECT * FROM bucket USE INDEX (idx_user_email)
WHERE type = "user" AND email = "alice@example.com";

-- Use multiple indexes
SELECT * FROM bucket USE INDEX (idx_type, idx_status USING GSI)
WHERE type = "user" AND status = "active";
```

**Index Advisor:**
```sql
-- Enable index advisor
ADVISE SELECT * FROM bucket
WHERE type = "user" AND status = "active" AND age > 25;

-- Returns index recommendations
```

### Index Best Practices

**Index Design:**
```sql
-- ✅ GOOD: Selective leading field
CREATE INDEX idx_user_email ON bucket(type, email)
WHERE type = "user";

-- ❌ BAD: Low selectivity leading field
CREATE INDEX idx_active ON bucket(is_active, user_id);
-- is_active has only 2 values (true/false)

-- ✅ GOOD: Covering index
CREATE INDEX idx_user_profile ON bucket(type, email, name, age)
WHERE type = "user";

-- Query can use covering index (no document fetch)
SELECT email, name, age FROM bucket USE INDEX (idx_user_profile)
WHERE type = "user" AND email = "alice@example.com";
```

---

## 7. Full-Text Search

### FTS Index Creation

**Basic Search Index:**
```json
// Create via REST API or UI
{
  "name": "fts_users",
  "type": "fulltext-index",
  "sourceType": "couchbase",
  "sourceName": "bucket",
  "planParams": {
    "maxPartitionsPerPIndex": 64
  },
  "params": {
    "doc_config": {
      "mode": "type_field",
      "type_field": "type"
    },
    "mapping": {
      "default_mapping": {
        "enabled": false
      },
      "types": {
        "user": {
          "enabled": true,
          "properties": {
            "name": {
              "enabled": true,
              "fields": [
                {
                  "name": "name",
                  "type": "text",
                  "analyzer": "standard"
                }
              ]
            },
            "email": {
              "enabled": true,
              "fields": [
                {
                  "name": "email",
                  "type": "text"
                }
              ]
            }
          }
        }
      }
    }
  }
}
```

### Search Queries

**Simple Search:**
```python
from couchbase.search import SearchQuery, TermQuery

# Search for term
query = TermQuery("Alice")
result = cluster.search_query(
  'fts_users',
  query,
  SearchOptions(limit=10)
)

for hit in result.rows():
  print(hit.id, hit.score)
```

**Match Query:**
```python
from couchbase.search import MatchQuery

# Match query (fuzzy matching)
query = MatchQuery("Alice Smith").field("name")
result = cluster.search_query('fts_users', query)
```

**Boolean Query:**
```python
from couchbase.search import ConjunctionQuery, DisjunctionQuery

# AND query
query = ConjunctionQuery(
  MatchQuery("engineer").field("bio"),
  MatchQuery("Python").field("skills")
)

# OR query
query = DisjunctionQuery(
  MatchQuery("Python").field("skills"),
  MatchQuery("Java").field("skills")
)
```

**Phrase Search:**
```python
from couchbase.search import MatchPhraseQuery

# Exact phrase match
query = MatchPhraseQuery("software engineer").field("bio")
result = cluster.search_query('fts_users', query)
```

**Wildcard and Regex:**
```python
from couchbase.search import WildcardQuery, RegexpQuery

# Wildcard query
query = WildcardQuery("alice*").field("email")

# Regex query
query = RegexpQuery(".*@example\\.com").field("email")
```

### Facets

**Faceted Search:**
```python
from couchbase.search import TermFacet, NumericRangeFacet

# Facet by category
facets = {
  'category': TermFacet('category', 10),
  'price_ranges': NumericRangeFacet('price', [
    {'name': 'cheap', 'max': 50},
    {'name': 'medium', 'min': 50, 'max': 200},
    {'name': 'expensive', 'min': 200}
  ])
}

result = cluster.search_query(
  'fts_products',
  MatchQuery("laptop"),
  SearchOptions(facets=facets)
)

# Access facets
for name, facet in result.facets().items():
  print(f"Facet: {name}")
  for item in facet:
    print(f"  {item.name}: {item.count}")
```

### Geospatial Search

**Geo Queries:**
```python
from couchbase.search import GeoDistanceQuery, GeoBoundingBoxQuery

# Search within radius
query = GeoDistanceQuery(
  40.7128,  # latitude
  -74.0060,  # longitude
  "10mi"  # distance
).field("location")

# Search within bounding box
query = GeoBoundingBoxQuery(
  40.7128, -74.0060,  # top-left
  40.7000, -73.9900   # bottom-right
).field("location")
```

---

## 8. Caching and Memory Management

### Managed Cache

**Cache Architecture:**
```
Couchbase Managed Cache:
- Automatic cache management
- LRU eviction
- Working set in memory
- Cache miss → disk fetch

Memory Quota:
- Per-bucket memory allocation
- Configured during bucket creation
- Monitor memory usage
```

**Bucket Memory Configuration:**
```bash
# Set bucket memory quota (256MB)
couchbase-cli bucket-create \
  -c localhost:8091 \
  -u Administrator -p password \
  --bucket mybucket \
  --bucket-type couchbase \
  --bucket-ramsize 256 \
  --bucket-replica 1
```

### Ejection Policies

**Value Ejection:**
```
Value Ejection (default):
- Only values ejected from memory
- Keys and metadata remain
- Fast key lookup
- Recommended for most workloads
```

**Full Ejection:**
```
Full Ejection:
- Both keys and values ejected
- More memory efficient
- Slower key lookups
- Use for large datasets (100M+ docs)
```

**Configure Ejection:**
```bash
couchbase-cli bucket-edit \
  -c localhost:8091 \
  -u Administrator -p password \
  --bucket mybucket \
  --bucket-eviction-policy valueOnly  # or fullEviction
```

### Metadata Overhead

**Memory Calculation:**
```
Per-document metadata overhead:
- Key: ~40-50 bytes
- Metadata: ~56 bytes
- Total: ~96-106 bytes per document

Example:
1M documents
1M × 100 bytes metadata = 100MB metadata

Total memory = Data + Metadata
256MB bucket = ~156MB data + 100MB metadata
```

### Ephemeral Buckets

**Memory-Only Buckets:**
```bash
# Create ephemeral bucket (no persistence)
couchbase-cli bucket-create \
  -c localhost:8091 \
  -u Administrator -p password \
  --bucket cache_bucket \
  --bucket-type ephemeral \
  --bucket-ramsize 512 \
  --bucket-replica 1 \
  --bucket-eviction-policy nruEviction  # or noEviction
```

**Eviction Policies:**
```
nruEviction (Not Recently Used):
- Evicts least recently used items
- Like LRU but more efficient

noEviction:
- No automatic eviction
- Rejects writes when full
- Use for cache-only scenarios
```

### Cache Warming

**Automatic Warming:**
```
On node restart:
- Couchbase automatically loads working set
- Background process loads frequently accessed keys
- Gradual performance improvement

Monitor warming:
- Check ep_warmup_state
- Wait for "done" before production traffic
```

**Manual Warming:**
```python
# Pre-warm cache with important keys
important_keys = ['user::alice', 'config::app', 'session::active']

for key in important_keys:
  try:
    collection.get(key)  # Loads into cache
  except DocumentNotFoundException:
    pass
```

---

## 9. Clustering and Scaling

### Cluster Setup

**Initialize Cluster:**
```bash
# Initialize first node
couchbase-cli cluster-init \
  -c localhost:8091 \
  -u Administrator -p password \
  --cluster-ramsize 2048 \
  --cluster-index-ramsize 512 \
  --cluster-fts-ramsize 512 \
  --cluster-eventing-ramsize 256 \
  --cluster-analytics-ramsize 1024 \
  --services data,index,query
```

**Add Nodes:**
```bash
# Add node to cluster
couchbase-cli server-add \
  -c node1:8091 \
  -u Administrator -p password \
  --server-add node2:8091 \
  --server-add-username Administrator \
  --server-add-password password \
  --services data,query

# Rebalance cluster
couchbase-cli rebalance \
  -c node1:8091 \
  -u Administrator -p password
```

**Remove Node:**
```bash
# Remove node (graceful)
couchbase-cli rebalance \
  -c node1:8091 \
  -u Administrator -p password \
  --server-remove node3:8091
```

### Multi-Dimensional Scaling (MDS)

**Service Distribution:**
```bash
# Node 1: Data + Index
# Node 2: Data + Query
# Node 3: Query + Search
# Node 4: Analytics

# Add node with specific services
couchbase-cli server-add \
  -c node1:8091 \
  -u Administrator -p password \
  --server-add node4:8091 \
  --server-add-username Administrator \
  --server-add-password password \
  --services analytics,eventing
```

**Service Isolation Benefits:**
```
Data Service:
- I/O intensive
- Requires fast storage (SSD)

Index Service:
- CPU + Memory intensive
- Benefits from fast CPU

Query Service:
- CPU intensive
- Can scale independently

Search Service:
- CPU + Memory
- Independent scaling

Analytics Service:
- Separate from operational queries
- No impact on production
```

### Rebalance

**Automatic Rebalancing:**
```bash
# Rebalance moves vBuckets across nodes
# - Distributes data evenly
# - No downtime
# - Gradual migration

# Monitor rebalance
couchbase-cli rebalance-status \
  -c localhost:8091 \
  -u Administrator -p password
```

**Rebalance Progress:**
```bash
# Check rebalance progress
curl -u Administrator:password \
  http://localhost:8091/pools/default/tasks

# Response shows:
# - Progress percentage
# - vBuckets moved
# - Estimated completion time
```

### Auto-Failover

**Configure Auto-Failover:**
```bash
# Enable auto-failover
couchbase-cli setting-autofailover \
  -c localhost:8091 \
  -u Administrator -p password \
  --enable-auto-failover 1 \
  --auto-failover-timeout 120  # 120 seconds

# Failover triggers:
# - Node unresponsive > timeout
# - Automatic promotion of replica
# - No data loss (if replicas exist)
```

### Cross-Cluster Replication (XDCR)

See [Section 9](#9-cross-datacenter-replication-xdcr)

---

## 10. Cross-Datacenter Replication (XDCR)

### XDCR Overview

**Replication Types:**
```
Unidirectional:
Cluster A → Cluster B

Bidirectional (Active-Active):
Cluster A ←→ Cluster B

Multi-Master:
Cluster A ←→ Cluster B ←→ Cluster C
```

### Setup XDCR

**Create Remote Cluster Reference:**
```bash
# Add remote cluster
couchbase-cli xdcr-setup \
  -c localhost:8091 \
  -u Administrator -p password \
  --create \
  --xdcr-cluster-name cluster_b \
  --xdcr-hostname remote-cluster:8091 \
  --xdcr-username Administrator \
  --xdcr-password password
```

**Create Replication:**
```bash
# Create replication stream
couchbase-cli xdcr-replicate \
  -c localhost:8091 \
  -u Administrator -p password \
  --create \
  --xdcr-cluster-name cluster_b \
  --xdcr-from-bucket mybucket \
  --xdcr-to-bucket mybucket \
  --xdcr-replication-mode continuous
```

### Replication Modes

**Continuous Replication:**
```
Real-time replication:
- Changes replicated immediately
- Asynchronous
- Low latency

Use for:
- Active-active scenarios
- Disaster recovery
- Geographic distribution
```

**Filtered Replication:**
```javascript
// Filter by document type
function(doc, meta) {
  if (doc.type === "user") {
    return true;  // Replicate
  }
  return false;  // Don't replicate
}

// Filter by field value
function(doc, meta) {
  return doc.region === "US";
}

// Filter by key pattern
function(doc, meta) {
  return meta.id.indexOf("user::") === 0;
}
```

### Conflict Resolution

**Automatic Conflict Resolution:**
```
Conflict detection:
- Same document modified on both clusters
- XDCR detects conflict via CAS

Resolution strategies:
1. Revision ID (default) - highest revision wins
2. Timestamp - most recent write wins
3. Custom - application handles conflicts

Configure:
couchbase-cli bucket-edit \
  --conflict-resolution-type lww  # Last Write Wins
```

**Application-Level Resolution:**
```python
# Handle conflicts in application
def resolve_conflict(doc_a, doc_b):
  # Custom merge logic
  if doc_a['updated_at'] > doc_b['updated_at']:
    return doc_a
  return doc_b
```

### Monitoring XDCR

**Check Replication Status:**
```bash
# View XDCR statistics
curl -u Administrator:password \
  http://localhost:8091/pools/default/buckets/mybucket/stats

# Key metrics:
# - xdc_ops: Operations replicated
# - xdc_items_remaining: Queue size
# - xdc_ops_failed: Failed replications
```

**Pause/Resume Replication:**
```bash
# Pause replication
couchbase-cli xdcr-replicate \
  -c localhost:8091 \
  -u Administrator -p password \
  --pause \
  --xdcr-replicator=<replication-id>

# Resume replication
couchbase-cli xdcr-replicate \
  -c localhost:8091 \
  -u Administrator -p password \
  --resume \
  --xdcr-replicator=<replication-id>
```

---

## 11. Security and Access Control

### Authentication

**Local Users:**
```bash
# Create user
couchbase-cli user-manage \
  -c localhost:8091 \
  -u Administrator -p password \
  --set \
  --rbac-username alice \
  --rbac-password alicepassword \
  --rbac-name "Alice Smith" \
  --roles bucket_admin[mybucket]
```

**LDAP/Active Directory:**
```bash
# Configure LDAP
couchbase-cli setting-ldap \
  -c localhost:8091 \
  -u Administrator -p password \
  --ldap-enabled 1 \
  --ldap-hosts ldap://ldap.example.com \
  --ldap-bind-dn "cn=admin,dc=example,dc=com" \
  --ldap-bind-password password \
  --ldap-user-dn-query "(&(objectClass=person)(uid=%u))"
```

### Role-Based Access Control (RBAC)

**Built-in Roles:**
```
Cluster Roles:
- admin: Full cluster access
- cluster_admin: Cluster configuration
- ro_admin: Read-only admin

Bucket Roles:
- bucket_admin[bucket]: Full bucket access
- bucket_full_access[bucket]: Read/write data
- views_admin[bucket]: View management
- data_reader[bucket]: Read-only data access
- data_writer[bucket]: Write data only
- data_dcp_reader[bucket]: DCP stream access

Query Roles:
- query_select[bucket]: SELECT queries
- query_update[bucket]: UPDATE/DELETE queries
- query_insert[bucket]: INSERT queries
```

**Assign Roles:**
```bash
# Assign multiple roles
couchbase-cli user-manage \
  -c localhost:8091 \
  -u Administrator -p password \
  --set \
  --rbac-username bob \
  --rbac-password bobpassword \
  --roles bucket_full_access[mybucket],query_select[mybucket]
```

**Custom Roles (Enterprise):**
```bash
# Create custom role
couchbase-cli user-manage \
  -c localhost:8091 \
  -u Administrator -p password \
  --set-group \
  --group-name developers \
  --roles bucket_full_access[mybucket],query_select[mybucket]

# Assign user to group
couchbase-cli user-manage \
  -c localhost:8091 \
  -u Administrator -p password \
  --set \
  --rbac-username charlie \
  --rbac-password charliepassword \
  --groups developers
```

### Encryption

**Encryption at Rest:**
```bash
# Enable encryption at rest (Enterprise only)
couchbase-cli node-to-node-encryption \
  -c localhost:8091 \
  -u Administrator -p password \
  --enable
```

**Encryption in Transit (TLS):**
```bash
# Configure TLS
couchbase-cli ssl-manage \
  -c localhost:8091 \
  -u Administrator -p password \
  --upload-cluster-ca=/path/to/ca.pem

# Enable TLS for client connections
couchbase-cli setting-security \
  -c localhost:8091 \
  -u Administrator -p password \
  --set \
  --tls-min-version tlsv1.2

# Client connection with TLS
cluster = Cluster('couchbases://localhost?ssl=no_verify',
  ClusterOptions(PasswordAuthenticator('admin', 'password')))
```

### Audit Logging

**Enable Auditing:**
```bash
# Enable audit logging
couchbase-cli setting-audit \
  -c localhost:8091 \
  -u Administrator -p password \
  --set \
  --audit-enabled 1 \
  --audit-log-path /opt/couchbase/var/lib/couchbase/logs \
  --audit-log-rotate-interval 86400
```

**Audit Events:**
```
Logged events:
- Authentication attempts
- User creation/deletion
- Bucket operations
- Query execution
- Configuration changes
- Data access (with appropriate role)
```

---

## 12. Backup and Recovery

### cbbackupmgr

**Create Backup Repository:**
```bash
# Initialize backup repository
cbbackupmgr config \
  --archive /backup/archive \
  --repo mybucket_backup

# Configure repository
cbbackupmgr config \
  --archive /backup/archive \
  --repo mybucket_backup \
  --config \
  --exclude-buckets temp,cache
```

**Full Backup:**
```bash
# Create full backup
cbbackupmgr backup \
  --archive /backup/archive \
  --repo mybucket_backup \
  --cluster couchbase://localhost \
  --username Administrator \
  --password password \
  --full-backup
```

**Incremental Backup:**
```bash
# Incremental backup
cbbackupmgr backup \
  --archive /backup/archive \
  --repo mybucket_backup \
  --cluster couchbase://localhost \
  --username Administrator \
  --password password
```

**Scheduled Backups:**
```bash
#!/bin/bash
# Daily backup script

DATE=$(date +%Y%m%d)

# Full backup on Sunday
if [ $(date +%u) -eq 7 ]; then
  cbbackupmgr backup \
    --archive /backup/archive \
    --repo mybucket_backup \
    --cluster couchbase://localhost \
    --username Administrator \
    --password password \
    --full-backup
else
  # Incremental on weekdays
  cbbackupmgr backup \
    --archive /backup/archive \
    --repo mybucket_backup \
    --cluster couchbase://localhost \
    --username Administrator \
    --password password
fi

# Retention: Keep last 30 days
find /backup/archive -type d -mtime +30 -exec rm -rf {} \;
```

### Restore Operations

**List Backups:**
```bash
# View available backups
cbbackupmgr list \
  --archive /backup/archive \
  --repo mybucket_backup
```

**Restore Full:**
```bash
# Restore entire backup
cbbackupmgr restore \
  --archive /backup/archive \
  --repo mybucket_backup \
  --cluster couchbase://localhost \
  --username Administrator \
  --password password \
  --force-updates
```

**Restore Specific Bucket:**
```bash
# Restore single bucket
cbbackupmgr restore \
  --archive /backup/archive \
  --repo mybucket_backup \
  --cluster couchbase://localhost \
  --username Administrator \
  --password password \
  --include-buckets mybucket \
  --force-updates
```

**Point-in-Time Restore:**
```bash
# Restore to specific date/time
cbbackupmgr restore \
  --archive /backup/archive \
  --repo mybucket_backup \
  --cluster couchbase://localhost \
  --username Administrator \
  --password password \
  --end 2024-02-06T10:00:00 \
  --force-updates
```

### Export/Import

**cbexport:**
```bash
# Export to JSON
cbexport json \
  --cluster couchbase://localhost \
  --username Administrator \
  --password password \
  --bucket mybucket \
  --format lines \
  --output /export/mybucket.json

# Export with filter
cbexport json \
  --cluster couchbase://localhost \
  --username Administrator \
  --password password \
  --bucket mybucket \
  --format lines \
  --output /export/users.json \
  --include-key "user::"
```

**cbimport:**
```bash
# Import from JSON
cbimport json \
  --cluster couchbase://localhost \
  --username Administrator \
  --password password \
  --bucket mybucket \
  --format lines \
  --generate-key %type%::%id% \
  --dataset /import/data.json
```

---

## 13. Monitoring and Troubleshooting

### Web Console

**Admin Console:**
```
Access: http://localhost:8091
- Dashboard overview
- Cluster statistics
- Bucket details
- Query workbench
- XDCR monitoring
- Alerts and warnings
```

### Metrics and Statistics

**REST API Monitoring:**
```bash
# Cluster statistics
curl -u Administrator:password \
  http://localhost:8091/pools/default

# Bucket statistics
curl -u Administrator:password \
  http://localhost:8091/pools/default/buckets/mybucket/stats

# Node statistics
curl -u Administrator:password \
  http://localhost:8091/pools/nodes
```

**Key Metrics:**
```
Per-Bucket Metrics:
- ops: Operations per second
- cmd_get: GET operations
- cmd_set: SET operations
- get_hits: Cache hits
- ep_cache_miss_rate: Cache miss rate
- curr_items: Current item count
- mem_used: Memory usage
- disk_write_queue: Write queue size

Query Metrics:
- query_requests: Query count
- query_avg_req_time: Average query time
- query_errors: Query errors

Index Metrics:
- index_num_docs_indexed: Documents indexed
- index_data_size: Index size
- index_memory_used: Index memory
```

### Prometheus Integration

**Configure Prometheus:**
```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'couchbase'
    static_configs:
      - targets:
        - 'node1:8091'
        - 'node2:8091'
        - 'node3:8091'
    metrics_path: '/metrics'
    basic_auth:
      username: 'Administrator'
      password: 'password'
```

### Logging

**Log Locations:**
```bash
# Log directory
/opt/couchbase/var/lib/couchbase/logs/

# Key log files:
# - couchdb.log: Data service
# - query.log: Query service
# - indexer.log: Index service
# - fts.log: Full-text search
# - xdcr.log: XDCR replication
# - memcached.log: Cache layer
```

**Collect Logs:**
```bash
# Collect diagnostic information
cbcollect_info /tmp/cbcollect.zip

# Upload for support
# Contains:
# - All log files
# - Configuration
# - Statistics
# - System information
```

### Slow Query Logging

**Enable Query Monitoring:**
```sql
-- Set query threshold
UPDATE system:completed_requests
SET threshold = 5000;  -- 5 seconds

-- View slow queries
SELECT * FROM system:completed_requests
WHERE elapsedTime > 5000
ORDER BY elapsedTime DESC;
```

### Common Issues

**High Memory Usage:**
```bash
# Check bucket memory
cbstats localhost:11210 -u Administrator -p password -b mybucket all | grep mem

# Solutions:
# 1. Increase bucket RAM quota
# 2. Enable value ejection
# 3. Add nodes
# 4. Reduce working set
```

**Rebalance Failed:**
```bash
# Check rebalance status
couchbase-cli rebalance-status \
  -c localhost:8091 \
  -u Administrator -p password

# Solutions:
# 1. Check node connectivity
# 2. Verify disk space
# 3. Check error logs
# 4. Retry rebalance
```

**Query Timeout:**
```python
# Increase query timeout
cluster = Cluster('couchbase://localhost',
  ClusterOptions(
    PasswordAuthenticator('admin', 'password'),
    timeout_options=ClusterTimeoutOptions(
      query_timeout=timedelta(seconds=300)
    )
  ))
```

---

## 14. Mobile and Edge Sync

### Couchbase Lite

**Mobile Database:**
```swift
// iOS/Swift example
import CouchbaseLiteSwift

// Create database
let database = try Database(name: "mydb")

// Create document
let doc = MutableDocument(id: "user::alice")
doc.setString("Alice Smith", forKey: "name")
doc.setString("alice@example.com", forKey: "email")
try database.saveDocument(doc)

// Query
let query = QueryBuilder
  .select(SelectResult.all())
  .from(DataSource.database(database))
  .where(Expression.property("type").equalTo(Expression.string("user")))

for result in try query.execute() {
  let dict = result.toDictionary()
  print(dict)
}
```

### Sync Gateway

**Configure Sync Gateway:**
```json
{
  "logging": {
    "log_file_path": "/var/log/sync_gateway",
    "console": {
      "log_level": "info"
    }
  },
  "databases": {
    "mydb": {
      "server": "couchbase://localhost",
      "username": "sync_gateway",
      "password": "password",
      "bucket": "mybucket",
      "users": {
        "alice": {
          "password": "alicepassword",
          "admin_channels": ["user::alice"]
        }
      },
      "sync": `
        function(doc, oldDoc) {
          // Routing function
          if (doc.type === 'user') {
            channel('user::' + doc.id);
          }
          if (doc.type === 'public') {
            channel('!');  // Public channel
          }
        }
      `
    }
  }
}
```

**Start Sync Gateway:**
```bash
sync_gateway /path/to/sync-gateway-config.json
```

### Replication

**Push Replication:**
```swift
// Push local changes to server
let url = URL(string: "ws://localhost:4984/mydb")!
let targetEndpoint = URLEndpoint(url: url)

let replConfig = ReplicatorConfiguration(
  database: database,
  target: targetEndpoint
)
replConfig.replicatorType = .push
replConfig.authenticator = BasicAuthenticator(
  username: "alice",
  password: "alicepassword"
)

let replicator = Replicator(config: replConfig)
replicator.start()
```

**Pull Replication:**
```swift
// Pull server changes to local
replConfig.replicatorType = .pull
let replicator = Replicator(config: replConfig)
replicator.start()
```

**Bidirectional Sync:**
```swift
// Two-way sync
replConfig.replicatorType = .pushAndPull
replConfig.continuous = true  // Continuous sync

let replicator = Replicator(config: replConfig)

// Listen for changes
replicator.addChangeListener { change in
  if let error = change.status.error {
    print("Replication error: \(error)")
  } else {
    print("Replication status: \(change.status.activity)")
  }
}

replicator.start()
```

### Conflict Resolution

**Automatic Resolution:**
```swift
// Default: Last write wins

// Custom conflict resolver
replConfig.conflictResolver = LocalWinConflictResolver()

class LocalWinConflictResolver: ConflictResolverProtocol {
  func resolve(conflict: Conflict) -> Document? {
    // Always prefer local version
    return conflict.localDocument
  }
}
```

**Manual Merge:**
```swift
class MergeConflictResolver: ConflictResolverProtocol {
  func resolve(conflict: Conflict) -> Document? {
    guard let local = conflict.localDocument,
          let remote = conflict.remoteDocument else {
      return nil
    }

    // Merge logic
    let merged = local.toMutable()

    // Take newest timestamp
    if let localTime = local.date(forKey: "updated_at"),
       let remoteTime = remote.date(forKey: "updated_at"),
       remoteTime > localTime {
      merged.setDate(remoteTime, forKey: "updated_at")
      // Copy other fields from remote..
    }

    return merged
  }
}
```

---

## 15. Analytics Service

### Analytics Setup

**Create Analytics Dataset:**
```sql
-- Connect bucket to analytics
CREATE BUCKET mybucket WITH {"name":"mybucket"};

-- Create dataset (shadow of bucket data)
CREATE DATASET users ON mybucket WHERE type = "user";

-- Create indexes for analytics
CREATE INDEX idx_email ON users(email);
```

### Analytics Queries

**Run Analytics Query:**
```sql
-- Analytics query (separate from operational queries)
SELECT u.name, COUNT(o.id) AS order_count
FROM users u
LEFT JOIN mybucket o ON o.user_id = u.id
WHERE o.type = "order"
GROUP BY u.name
ORDER BY order_count DESC
LIMIT 10;
```

**Complex Analytics:**
```sql
-- Window functions
SELECT
  name,
  revenue,
  RANK() OVER (ORDER BY revenue DESC) AS rank,
  SUM(revenue) OVER (ORDER BY date
    ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) AS moving_sum_7d
FROM sales_dataset
WHERE date >= "2024-01-01";
```

### Benefits of Analytics Service

**Workload Isolation:**
```
Operational (N1QL):
- Real-time queries
- Latency-sensitive
- OLTP workloads

Analytics:
- Batch queries
- Complex aggregations
- OLAP workloads
- No impact on production
```

**Columnar Storage:**
```
Analytics uses columnar format:
- Faster aggregations
- Better compression
- Optimized for scanning
```

---

## 16. Eventing Service

### Eventing Functions

**Create Function:**
```javascript
// OnUpdate function (triggered on document changes)
function OnUpdate(doc, meta) {
  // Only process orders
  if (doc.type !== 'order') {
    return;
  }

  // Calculate order total
  var total = 0;
  if (doc.items && doc.items.length > 0) {
    doc.items.forEach(function(item) {
      total += item.price * item.quantity;
    });
  }

  // Update order total
  doc.total = total;
  doc.updated_at = new Date().toISOString();

  // Save back to bucket
  mybucket[meta.id] = doc;
}
```

**Timer Function:**
```javascript
// OnTimer function (scheduled execution)
function OnTimer() {
  // Find expired sessions
  var query = SELECT meta().id
              FROM mybucket
              WHERE type = "session"
                AND expires_at < NOW_STR();

  for (var row of query) {
    delete mybucket[row.id];
  }
}

// Schedule: Daily at 2 AM
// Timer configured in UI
```

### Use Cases

**Data Enrichment:**
```javascript
function OnUpdate(doc, meta) {
  if (doc.type === 'user' && !doc.enriched) {
    // Enrich with external API
    var response = curl('GET', 'https://api.example.com/user/' + doc.id);
    var data = JSON.parse(response.body);

    doc.location = data.location;
    doc.timezone = data.timezone;
    doc.enriched = true;

    mybucket[meta.id] = doc;
  }
}
```

**Data Aggregation:**
```javascript
function OnUpdate(doc, meta) {
  if (doc.type === 'order' && doc.status === 'completed') {
    // Update user's total spent
    var user = mybucket['user::' + doc.user_id];
    if (user) {
      user.total_spent = (user.total_spent || 0) + doc.total;
      user.order_count = (user.order_count || 0) + 1;
      mybucket['user::' + doc.user_id] = user;
    }
  }
}
```

**Cascade Deletes:**
```javascript
function OnDelete(meta) {
  // Delete related documents
  if (meta.id.startsWith('user::')) {
    var userId = meta.id;

    // Delete user's orders
    var query = SELECT meta().id
                FROM mybucket
                WHERE type = "order" AND user_id = $userId;

    for (var row of query) {
      delete mybucket[row.id];
    }
  }
}
```

---

## 17. Application Integration

### Python SDK

**Installation:**
```bash
pip install couchbase
```

**Basic Usage:**
```python
from couchbase.cluster import Cluster
from couchbase.auth import PasswordAuthenticator
from couchbase.options import ClusterOptions
from datetime import timedelta

# Connect
cluster = Cluster(
  'couchbase://localhost',
  ClusterOptions(PasswordAuthenticator('Administrator', 'password'))
)

# Get bucket and collection
bucket = cluster.bucket('mybucket')
collection = bucket.default_collection()

# Insert
doc = {
  'type': 'user',
  'name': 'Alice',
  'email': 'alice@example.com'
}
result = collection.insert('user::alice', doc)
print(f"CAS: {result.cas}")

# Get
result = collection.get('user::alice')
user = result.content_as[dict]
print(user['name'])

# Query
from couchbase.n1ql import QueryOptions

result = cluster.query(
  "SELECT name, email FROM mybucket WHERE type = 'user'",
  QueryOptions(timeout=timedelta(seconds=10))
)

for row in result.rows():
  print(row)
```

### Node.js SDK

**Installation:**
```bash
npm install couchbase
```

**Usage:**
```javascript
const couchbase = require('couchbase');

async function main() {
  // Connect
  const cluster = await couchbase.connect('couchbase://localhost', {
    username: 'Administrator',
    password: 'password'
  });

  const bucket = cluster.bucket('mybucket');
  const collection = bucket.defaultCollection();

  // Insert
  await collection.insert('user::bob', {
    type: 'user',
    name: 'Bob',
    email: 'bob@example.com'
  });

  // Get
  const result = await collection.get('user::bob');
  console.log(result.content);

  // Query
  const query = `
    SELECT name, email
    FROM mybucket
    WHERE type = 'user' AND name LIKE 'B%'
  `;

  const queryResult = await cluster.query(query);
  queryResult.rows.forEach(row => {
    console.log(row);
  });
}

main().catch(console.error);
```

### Java SDK

**Maven Dependency:**
```xml
<dependency>
  <groupId>com.couchbase.client</groupId>
  <artifactId>java-client</artifactId>
  <version>3.4.11</version>
</dependency>
```

**Usage:**
```java
import com.couchbase.client.java.*;
import com.couchbase.client.java.json.*;

public class CouchbaseExample {
  public static void main(String[] args) {
    // Connect
    Cluster cluster = Cluster.connect(
      "localhost",
      "Administrator",
      "password"
    );

    Bucket bucket = cluster.bucket("mybucket");
    Collection collection = bucket.defaultCollection();

    // Insert
    JsonObject user = JsonObject.create()
      .put("type", "user")
      .put("name", "Carol")
      .put("email", "carol@example.com");

    collection.insert("user::carol", user);

    // Get
    JsonObject doc = collection.get("user::carol")
      .contentAsObject();
    System.out.println(doc.getString("name"));

    // Query
    QueryResult result = cluster.query(
      "SELECT name, email FROM mybucket WHERE type = 'user'"
    );

    for (JsonObject row : result.rowsAsObject()) {
      System.out.println(row);
    }

    cluster.disconnect();
  }
}
```

---

## 18. Production Deployment

### System Requirements

**Hardware Sizing:**
```
Small (Development):
- 4 CPU cores
- 8GB RAM
- 50GB SSD

Medium (Production):
- 8-16 CPU cores
- 32-64GB RAM
- 500GB-1TB NVMe SSD

Large (Enterprise):
- 32+ CPU cores
- 128-256GB+ RAM
- 2TB+ NVMe SSD

Memory allocation:
- Data Service: 60-70% of RAM
- Index Service: 20-30% of RAM
- Query Service: 10-20% of RAM
```

### Docker Deployment

**Single Node:**
```yaml
# docker-compose.yml
version: '3.8'

services:
  couchbase:
    image: couchbase:enterprise-7.2.0
    ports:
      - "8091-8096:8091-8096"
      - "11210:11210"
    environment:
      - CLUSTER_RAM_QUOTA=2048
      - INDEX_RAM_QUOTA=512
    volumes:
      - couchbase_data:/opt/couchbase/var
    restart: unless-stopped

volumes:
  couchbase_data:
```

**Cluster with Docker:**
```yaml
version: '3.8'

services:
  couchbase1:
    image: couchbase:enterprise-7.2.0
    ports:
      - "8091-8096:8091-8096"
      - "11210:11210"
    volumes:
      - cb1_data:/opt/couchbase/var
    networks:
      - couchbase_network

  couchbase2:
    image: couchbase:enterprise-7.2.0
    volumes:
      - cb2_data:/opt/couchbase/var
    networks:
      - couchbase_network

  couchbase3:
    image: couchbase:enterprise-7.2.0
    volumes:
      - cb3_data:/opt/couchbase/var
    networks:
      - couchbase_network

volumes:
  cb1_data:
  cb2_data:
  cb3_data:

networks:
  couchbase_network:
```

### Kubernetes Deployment

**Couchbase Autonomous Operator:**
```yaml
apiVersion: couchbase.com/v2
kind: CouchbaseCluster
metadata:
  name: cb-cluster
spec:
  image: couchbase/server:enterprise-7.2.0
  security:
    adminSecret: cb-admin-secret
  servers:
  - size: 3
    name: data
    services:
    - data
    - query
    - index
    pod:
      resources:
        limits:
          cpu: 4
          memory: 16Gi
        requests:
          cpu: 2
          memory: 8Gi
      volumeClaimTemplates:
      - metadata:
          name: couchbase-data
        spec:
          accessModes:
          - ReadWriteOnce
          storageClassName: fast-ssd
          resources:
            requests:
              storage: 500Gi
```

### Load Balancing

**HAProxy Configuration:**
```haproxy
# haproxy.cfg
global
    log stdout local0

defaults
    mode tcp
    timeout connect 5s
    timeout client 50s
    timeout server 50s

frontend couchbase_frontend
    bind *:8091
    default_backend couchbase_backend

backend couchbase_backend
    balance roundrobin
    option httpchk GET /pools/default
    http-check expect status 200
    server cb1 cb1:8091 check
    server cb2 cb2:8091 check
    server cb3 cb3:8091 check

frontend couchbase_kv
    bind *:11210
    default_backend couchbase_kv_backend

backend couchbase_kv_backend
    balance leastconn
    server cb1 cb1:11210 check
    server cb2 cb2:11210 check
    server cb3 cb3:11210 check
```

---

## 19. Performance Optimization

### Query Optimization

**Use Indexes:**
```sql
-- Create appropriate indexes
CREATE INDEX idx_user_status ON mybucket(type, status, created_at)
WHERE type = "user";

-- Use covering indexes
CREATE INDEX idx_user_profile ON mybucket(type, email, name, age)
WHERE type = "user";

-- Query uses covering index (no document fetch)
EXPLAIN SELECT email, name, age
FROM mybucket USE INDEX (idx_user_profile)
WHERE type = "user" AND email = "alice@example.com";
```

**Avoid Full Scans:**
```sql
-- ❌ BAD: No index, full bucket scan
SELECT * FROM mybucket WHERE name = "Alice";

-- ✅ GOOD: Use type field with index
SELECT * FROM mybucket
WHERE type = "user" AND name = "Alice";

-- ✅ Create index
CREATE INDEX idx_user_name ON mybucket(type, name)
WHERE type = "user";
```

### Connection Pooling

**Optimal Pool Size:**
```python
from couchbase.cluster import Cluster
from couchbase.options import ClusterOptions, ClusterTimeoutOptions
from datetime import timedelta

# Configure connection pooling
options = ClusterOptions(
  PasswordAuthenticator('admin', 'password'),
  timeout_options=ClusterTimeoutOptions(
    kv_timeout=timedelta(seconds=10),
    query_timeout=timedelta(seconds=75)
  )
)

# Max connections handled by SDK
cluster = Cluster('couchbase://localhost', options)
```

### Caching Strategies

**Application-Level Cache:**
```python
from functools import lru_cache

class UserService:
  def __init__(self, collection):
    self.collection = collection

  @lru_cache(maxsize=1000)
  def get_user(self, user_id):
    result = self.collection.get(user_id)
    return result.content_as[dict]

  def update_user(self, user_id, updates):
    # Update document
    result = self.collection.get(user_id)
    doc = result.content_as[dict]
    doc.update(updates)
    self.collection.replace(user_id, doc)

    # Invalidate cache
    self.get_user.cache_clear()
```

### Bulk Operations

**Batch Writes:**
```python
# Batch upsert for better performance
docs = {
  f'user::{i}': {'type': 'user', 'name': f'User{i}'}
  for i in range(1000)
}

results = collection.upsert_multi(docs)

# Check results
for key, result in results.items():
  if not result.success:
    print(f"Failed: {key}, {result.exception}")
```

---

## 20. Migration Strategies

### From MongoDB

**Schema Mapping:**
```javascript
// MongoDB document
{
  "_id": ObjectId("507f1f77bcf86cd799439011"),
  "name": "Alice",
  "email": "alice@example.com",
  "orders": [
    {"order_id": "order123", "total": 100}
  ]
}

// Couchbase document
{
  "type": "user",
  "id": "user::507f1f77bcf86cd799439011",
  "name": "Alice",
  "email": "alice@example.com",
  "order_ids": ["order::123"]
}

{
  "type": "order",
  "id": "order::123",
  "user_id": "user::507f1f77bcf86cd799439011",
  "total": 100
}
```

**Migration Script:**
```python
from pymongo import MongoClient
from couchbase.cluster import Cluster

# Connect to MongoDB
mongo_client = MongoClient('mongodb://localhost:27017')
mongo_db = mongo_client['myapp']

# Connect to Couchbase
cb_cluster = Cluster('couchbase://localhost',
  ClusterOptions(PasswordAuthenticator('admin', 'password')))
cb_collection = cb_cluster.bucket('mybucket').default_collection()

# Migrate users
for user in mongo_db.users.find():
  doc = {
    'type': 'user',
    'name': user['name'],
    'email': user['email']
  }
  key = f"user::{user['_id']}"
  cb_collection.upsert(key, doc)
```

### From Redis

**Cache to Database:**
```python
import redis
from couchbase.cluster import Cluster

# Connect
r = redis.Redis(host='localhost', port=6379)
cluster = Cluster('couchbase://localhost',
  ClusterOptions(PasswordAuthenticator('admin', 'password')))
collection = cluster.bucket('mybucket').default_collection()

# Migrate keys
for key in r.scan_iter(match="user:*"):
  value = r.get(key)
  doc = json.loads(value)
  doc['type'] = 'user'

  # Set TTL if exists
  ttl = r.ttl(key)
  if ttl > 0:
    collection.upsert(key.decode(), doc, expiry=ttl)
  else:
    collection.upsert(key.decode(), doc)
```

### Dual-Write Pattern

**During Migration:**
```python
class DualWriteService:
  def __init__(self, mongo_db, cb_collection):
    self.mongo = mongo_db
    self.cb = cb_collection
    self.migration_complete = False

  def create_user(self, user_data):
    # Write to primary (MongoDB)
    result = self.mongo.users.insert_one(user_data)
    user_id = str(result.inserted_id)

    # Write to secondary (Couchbase)
    try:
      doc = user_data.copy()
      doc['type'] = 'user'
      self.cb.upsert(f'user::{user_id}', doc)
    except Exception as e:
      print(f"Couchbase write failed: {e}")

    return user_id

  def get_user(self, user_id):
    # Read from Couchbase after migration
    if self.migration_complete:
      result = self.cb.get(f'user::{user_id}')
      return result.content_as[dict]

    # Otherwise read from MongoDB
    return self.mongo.users.find_one({'_id': ObjectId(user_id)})
```

---

## 21. Production Checklist

### Pre-Deployment

**Infrastructure:**
- [ ] Cluster sized appropriately (3+ nodes recommended)
- [ ] Services distributed across nodes (MDS)
- [ ] Network connectivity verified
- [ ] Load balancer configured
- [ ] Monitoring system setup (Prometheus/Grafana)
- [ ] Backup strategy defined
- [ ] SSL/TLS certificates configured

**Security:**
- [ ] Admin password changed from default
- [ ] Users created with appropriate roles (RBAC)
- [ ] LDAP/AD integration configured (if needed)
- [ ] Encryption at rest enabled (Enterprise)
- [ ] Encryption in transit (TLS) enabled
- [ ] Audit logging enabled
- [ ] Firewall rules configured

**Database Configuration:**
- [ ] Buckets created with appropriate memory quotas
- [ ] Replication configured (minimum 1 replica)
- [ ] Auto-failover enabled
- [ ] Indexes created for common queries
- [ ] Full-text search indexes configured (if needed)
- [ ] XDCR configured (if multi-datacenter)

**Application:**
- [ ] SDK version compatible with server version
- [ ] Connection pooling configured
- [ ] Retry logic implemented
- [ ] Error handling for timeouts
- [ ] Monitoring/logging integrated

### Post-Deployment

**Verification:**
- [ ] All nodes healthy and in cluster
- [ ] Buckets accessible and responding
- [ ] Replication working (check replica count)
- [ ] Queries executing successfully
- [ ] Backups running successfully
- [ ] Monitoring dashboards populated
- [ ] Alerts configured and firing correctly

**Operations:**
- [ ] Backup schedule: Daily full + hourly incremental
- [ ] Log rotation configured
- [ ] Maintenance window scheduled
- [ ] On-call rotation established
- [ ] Runbooks documented
- [ ] Capacity planning monitored

### Performance Tuning

**Query Performance:**
- [ ] Slow queries identified (>100ms)
- [ ] Indexes optimized
- [ ] Covering indexes used where possible
- [ ] Query patterns analyzed
- [ ] Connection pooling verified

**Configuration:**
- [ ] Bucket memory quotas optimized
- [ ] Ejection policy appropriate for workload
- [ ] Auto-compaction configured
- [ ] DCP timeout settings tuned

**Monitoring Metrics:**
```
Critical Metrics:
- ops/sec: Operations throughput
- get_hits/get_misses: Cache hit ratio (>90% good)
- ep_cache_miss_rate: Cache miss rate (<10% good)
- mem_used: Memory usage (<80% of quota)
- disk_write_queue: Write queue size (<100K)
- query_avg_req_time: Query latency (<100ms p95)
```

### Ongoing Maintenance

**Daily:**
- [ ] Check cluster health dashboard
- [ ] Monitor memory usage
- [ ] Verify backups completed
- [ ] Review error logs

**Weekly:**
- [ ] Review performance metrics
- [ ] Analyze slow queries
- [ ] Check disk space
- [ ] Review capacity trends

**Monthly:**
- [ ] Test backup restore
- [ ] Review and optimize indexes
- [ ] Capacity planning review
- [ ] Update documentation

**Quarterly:**
- [ ] Disaster recovery drill
- [ ] Security audit
- [ ] Version upgrade planning
- [ ] Performance tuning review

---

## 22. Deployment Checklist

### Agent-Generated Code Verification (MANDATORY)

#### Build & Compilation
- [ ] Code compiles/runs without errors
- [ ] All imports/dependencies resolved (Couchbase SDK)
- [ ] Code formatted per project standards

#### Testing
- [ ] All tests pass
- [ ] Coverage meets minimum threshold (>80%)
- [ ] Integration tests pass against Couchbase test cluster

#### Security
- [ ] Dependency scan: 0 HIGH/CRITICAL vulnerabilities
- [ ] No hardcoded credentials or secrets
- [ ] Connection strings use environment variables

#### Agent Workflow Completed
- [ ] Agent verified code builds successfully
- [ ] Agent ran all tests and verified they pass
- [ ] Agent verified documentation

---

## 23. Why This Configuration Works

**Multi-Model Architecture in a Single Platform**: Couchbase combines key-value, document, full-text search, analytics, and eventing services, eliminating the need for separate database systems and reducing operational complexity.

**Memory-First Design with Managed Caching**: The integrated caching layer serves reads from RAM with sub-millisecond latency, removing the need for a separate caching tier like Redis or Memcached.

**N1QL Provides SQL Familiarity for JSON Documents**: Developers can query JSON documents using familiar SQL syntax with JOINs, aggregations, and subqueries, lowering the learning curve compared to other NoSQL query languages.

**Built-In Cross-Datacenter Replication (XDCR)**: Active-active replication across data centers provides disaster recovery and geo-locality without third-party tooling or complex configuration.

---

## 24. Quick Reference

### Common Commands

```bash
# Initialize a new Couchbase cluster
couchbase-cli cluster-init -c localhost:8091 \
  --cluster-username admin --cluster-password password \
  --cluster-ramsize 1024 --services data,index,query

# Create a new bucket
couchbase-cli bucket-create -c localhost:8091 \
  -u admin -p password --bucket mybucket --bucket-ramsize 512

# Run a N1QL query from CLI
cbq -e http://localhost:8093 -u admin -p password \
  --script="SELECT * FROM mybucket LIMIT 10"

# Backup a cluster
cbbackupmgr backup -a /backup/archive -r myrepo \
  -c couchbase://localhost -u admin -p password

# Restore from backup
cbbackupmgr restore -a /backup/archive -r myrepo \
  -c couchbase://localhost -u admin -p password

# Check cluster server list
couchbase-cli server-list -c localhost:8091 -u admin -p password
```

---

## References and Resources

### Official Documentation
- **Couchbase Docs:** https://docs.couchbase.com/
- **N1QL Reference:** https://docs.couchbase.com/server/current/n1ql/n1ql-language-reference/
- **SDKs:** https://docs.couchbase.com/home/sdk.html
- **Best Practices:** https://docs.couchbase.com/server/current/learn/

### Learning Resources
- **Couchbase Academy:** https://learn.couchbase.com/
- **YouTube Channel:** Couchbase Official
- **Blog:** https://blog.couchbase.com/
- **GitHub:** https://github.com/couchbase

### Community
- **Forums:** https://forums.couchbase.com/
- **Stack Overflow:** `[couchbase]` tag
- **Slack:** https://couchbase.com/slack
- **Discord:** Couchbase Community

### Tools
- **Web Console:** Built-in (port 8091)
- **cbbackupmgr:** Backup/restore utility
- **cbc:** Command-line client
- **Query Workbench:** Web-based query IDE

---

**Document Maintenance:**
- Review quarterly for Couchbase updates
- Update with new N1QL features
- Add community best practices
- Test examples with latest version

**Last Updated:** February 2026
**Next Review:** May 2026

---

**End of Couchbase Development Guidelines**
