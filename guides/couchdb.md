# CouchDB Best Practices Guide

**Version:** 1.0
**Last Updated:** February 2026
**Target Version:** CouchDB 3.x+

## Table of Contents

1. [Architecture and Fundamentals](#1-architecture-and-fundamentals)
2. [Document Model and Design](#2-document-model-and-design)
3. [HTTP API and REST Interface](#3-http-api-and-rest-interface)
4. [Mango Query Language](#4-mango-query-language)
5. [MapReduce Views](#5-mapreduce-views)
6. [Indexes and Performance](#6-indexes-and-performance)
7. [Replication and Sync](#7-replication-and-sync)
8. [Clustering and Sharding](#8-clustering-and-sharding)
9. [Conflict Resolution](#9-conflict-resolution)
10. [Security and Authentication](#10-security-and-authentication)
11. [Backup and Recovery](#11-backup-and-recovery)
12. [Monitoring and Troubleshooting](#12-monitoring-and-troubleshooting)
13. [PouchDB and Offline-First](#13-pouchdb-and-offline-first)
14. [Change Feeds and Live Updates](#14-change-feeds-and-live-updates)
15. [Application Integration](#15-application-integration)
16. [Production Deployment](#16-production-deployment)
17. [Performance Optimization](#17-performance-optimization)
18. [Migration Strategies](#18-migration-strategies)
19. [Comparison with Other Databases](#19-comparison-with-other-databases)
20. [Production Checklist](#20-production-checklist)

---

## 1. Architecture and Fundamentals

### What is CouchDB?

**CouchDB** is a NoSQL document database designed for ease of use and web-native features:

- ✅ **Document-oriented** (JSON documents)
- ✅ **HTTP/REST API** (database as a web service)
- ✅ **Multi-master replication** (bidirectional sync)
- ✅ **Eventual consistency** (AP in CAP theorem)
- ✅ **Offline-first** (PouchDB compatibility)
- ✅ **MVCC** (Multi-Version Concurrency Control)
- ✅ **MapReduce views** (incremental indexing)
- ✅ **Web-friendly** (CORS, attachments)

### Architecture Overview

```
┌─────────────────────────────────────────────────┐
│          CouchDB Architecture                    │
├─────────────────────────────────────────────────┤
│                                                  │
│  ┌────────────────────────────────────────┐    │
│  │         HTTP API Layer                 │    │
│  │  - REST endpoints                      │    │
│  │  - JSON protocol                       │    │
│  │  - Authentication                      │    │
│  └────────────────────────────────────────┘    │
│                      │                          │
│                      ▼                          │
│  ┌────────────────────────────────────────┐    │
│  │         Query Engine                   │    │
│  │  - Mango Query                         │    │
│  │  - MapReduce Views                     │    │
│  │  - Indexing                            │    │
│  └────────────────────────────────────────┘    │
│                      │                          │
│                      ▼                          │
│  ┌────────────────────────────────────────┐    │
│  │         Replication Engine             │    │
│  │  - Multi-master sync                   │    │
│  │  - Conflict detection                  │    │
│  │  - Change feeds                        │    │
│  └────────────────────────────────────────┘    │
│                      │                          │
│                      ▼                          │
│  ┌────────────────────────────────────────┐    │
│  │         Storage Engine                 │    │
│  │  - Append-only B+ tree                 │    │
│  │  - MVCC (Copy-on-Write)                │    │
│  │  - Compaction                          │    │
│  └────────────────────────────────────────┘    │
└─────────────────────────────────────────────────┘
```

### Key Concepts

**Documents:**
```json
{
  "_id": "user:alice",
  "_rev": "1-967a00dff5e02add41819138abb3284d",
  "type": "user",
  "name": "Alice",
  "email": "alice@example.com",
  "created_at": "2024-02-06T10:00:00Z"
}

Key fields:
- _id: Unique document identifier
- _rev: Revision identifier (for MVCC)
- Other fields: Application data
```

**Revisions (MVCC):**
```
Document lifecycle:
1. Create: _rev = "1-abc123"
2. Update: _rev = "2-def456"  (new revision)
3. Update: _rev = "3-ghi789"  (another revision)

Each update creates new revision
Old revisions kept temporarily
Conflicts detected by revision tree
```

**Append-Only Storage:**
```
CouchDB uses append-only B+ tree:

Benefits:
✓ Never overwrites data (crash-safe)
✓ MVCC without locks
✓ Efficient sequential writes

Trade-off:
✗ Requires compaction to reclaim space
✗ Database file grows until compaction
```

**Databases:**
```
CouchDB organizes documents into databases:

Cluster:
├── _users (system database)
├── _replicator (replication jobs)
├── myapp (application database)
├── logs (another database)
└── ...

Each database:
- Independent namespace
- Separate access control
- Own compaction schedule
```

### When to Use CouchDB

**✅ Excellent For:**

1. **Offline-First Applications:**
   - Mobile apps with sync
   - Progressive Web Apps (PWA)
   - Occasionally connected devices
   - Local-first software

2. **Multi-Master Replication:**
   - Distributed systems
   - Edge computing
   - Geographic distribution
   - Peer-to-peer sync

3. **Document Storage:**
   - JSON-native applications
   - Schema-less data
   - Binary attachments
   - Content management

4. **Web Applications:**
   - RESTful architecture
   - Direct browser access
   - CORS-enabled APIs
   - HTTP-based integration

5. **Audit and Versioning:**
   - Document history tracking
   - Change feeds
   - Event sourcing
   - Compliance requirements

**❌ Not Recommended For:**

1. **Complex Joins:**
   - Relational data
   - Use PostgreSQL instead

2. **High-Write Throughput:**
   - Time-series data
   - Use InfluxDB, TimescaleDB

3. **ACID Transactions Across Documents:**
   - Banking transactions
   - Use PostgreSQL, CockroachDB

4. **Real-Time Analytics:**
   - OLAP workloads
   - Use ClickHouse, Snowflake

5. **Strong Consistency Requirements:**
   - Stock trading
   - Use RDBMS with ACID

---

## 2. Document Model and Design

### Document Structure

**Basic Document:**
```json
{
  "_id": "order:12345",
  "_rev": "1-abc123",
  "type": "order",
  "user_id": "user:alice",
  "status": "pending",
  "items": [
    {
      "product_id": "product:widget",
      "quantity": 2,
      "price": 29.99
    }
  ],
  "total": 59.98,
  "created_at": "2024-02-06T10:00:00Z",
  "updated_at": "2024-02-06T10:00:00Z"
}
```

**Document ID Strategies:**
```json
// Strategy 1: Semantic IDs
{
  "_id": "user:alice",
  "_id": "order:12345",
  "_id": "product:widget"
}

// Strategy 2: UUIDs
{
  "_id": "550e8400-e29b-41d4-a716-446655440000"
}

// Strategy 3: Timestamps (for time-series)
{
  "_id": "log:2024-02-06T10:00:00.000Z"
}

// Strategy 4: Compound keys
{
  "_id": "user:alice:order:12345"
}

// Best practice: Use semantic IDs for better readability
// and range queries (startkey/endkey)
```

### Schema Design Patterns

**Embedded Documents:**
```json
{
  "_id": "user:alice",
  "name": "Alice",
  "email": "alice@example.com",
  "address": {
    "street": "123 Main St",
    "city": "Springfield",
    "zip": "12345"
  },
  "orders": [
    {
      "id": "order:1",
      "date": "2024-01-01",
      "total": 100.00
    }
  ]
}

// ✅ Good for: Frequently accessed together
// ❌ Bad for: Large arrays that grow unbounded
```

**Referenced Documents:**
```json
// User document
{
  "_id": "user:alice",
  "name": "Alice",
  "email": "alice@example.com"
}

// Order document (references user)
{
  "_id": "order:12345",
  "user_id": "user:alice",  // Reference
  "total": 100.00
}

// ✅ Good for: Large, independently updated data
// ❌ Bad for: Requires multiple requests (no joins)
```

**Denormalization:**
```json
// Denormalized order (includes user info)
{
  "_id": "order:12345",
  "user_id": "user:alice",
  "user_name": "Alice",        // Denormalized
  "user_email": "alice@example.com",  // Denormalized
  "total": 100.00
}

// ✅ Trade-off:
// - Faster reads (no lookup needed)
// - Slower writes (must update duplicated data)
// - Eventual consistency (updates may lag)
```

### Document Limits

**Size Limits:**
```
Maximum document size: 4GB (practical limit ~1-2MB)
Maximum attachment size: Unlimited (but chunked)

Recommendations:
- Keep documents < 1MB
- Use attachments for large binary data
- Split large documents into smaller ones
```

**Best Practices:**
```json
// ❌ BAD: Large embedded array
{
  "_id": "user:alice",
  "orders": [
    // 10,000 orders embedded
  ]
}

// ✅ GOOD: Separate documents
{
  "_id": "user:alice",
  "total_orders": 10000
}

{
  "_id": "order:12345",
  "user_id": "user:alice"
}
```

### Attachments

**Storing Binary Data:**
```bash
# Add attachment via HTTP
curl -X PUT \
  http://localhost:5984/mydb/user:alice/photo.jpg?rev=1-abc123 \
  -H "Content-Type: image/jpeg" \
  --data-binary @photo.jpg
```

**Document with Attachment:**
```json
{
  "_id": "user:alice",
  "_rev": "2-def456",
  "name": "Alice",
  "_attachments": {
    "photo.jpg": {
      "content_type": "image/jpeg",
      "length": 45123,
      "revpos": 2,
      "digest": "md5-abc123...",
      "stub": true
    }
  }
}
```

**Inline Attachments:**
```json
{
  "_id": "document123",
  "_attachments": {
    "file.txt": {
      "content_type": "text/plain",
      "data": "SGVsbG8gV29ybGQh"  // Base64 encoded
    }
  }
}
```

---

## 3. HTTP API and REST Interface

### Database Operations

**Create Database:**
```bash
# Create database
curl -X PUT http://localhost:5984/mydb

# Response
{"ok": true}
```

**List Databases:**
```bash
# List all databases
curl http://localhost:5984/_all_dbs

# Response
["_replicator", "_users", "mydb"]
```

**Database Info:**
```bash
# Get database information
curl http://localhost:5984/mydb

# Response
{
  "db_name": "mydb",
  "doc_count": 1000,
  "doc_del_count": 50,
  "update_seq": "1050-g1AAAAG...",
  "purge_seq": 0,
  "compact_running": false,
  "disk_size": 16842752,
  "data_size": 8421376,
  "instance_start_time": "0",
  "disk_format_version": 8
}
```

**Delete Database:**
```bash
# Delete database (CAREFUL!)
curl -X DELETE http://localhost:5984/mydb
```

### Document CRUD Operations

**Create Document:**
```bash
# Create with generated ID
curl -X POST http://localhost:5984/mydb \
  -H "Content-Type: application/json" \
  -d '{"name": "Alice", "email": "alice@example.com"}'

# Response
{
  "ok": true,
  "id": "0c0a2d9f5c8e1a7b3e4d2f1a6b5c8e9d",
  "rev": "1-967a00dff5e02add41819138abb3284d"
}

# Create with specified ID
curl -X PUT http://localhost:5984/mydb/user:alice \
  -H "Content-Type: application/json" \
  -d '{"name": "Alice", "email": "alice@example.com"}'
```

**Read Document:**
```bash
# Get document by ID
curl http://localhost:5984/mydb/user:alice

# Response
{
  "_id": "user:alice",
  "_rev": "1-967a00dff5e02add41819138abb3284d",
  "name": "Alice",
  "email": "alice@example.com"
}

# Get specific revision
curl http://localhost:5984/mydb/user:alice?rev=1-967a00dff5e02add41819138abb3284d
```

**Update Document:**
```bash
# Update requires _rev (optimistic locking)
curl -X PUT http://localhost:5984/mydb/user:alice \
  -H "Content-Type: application/json" \
  -d '{
    "_id": "user:alice",
    "_rev": "1-967a00dff5e02add41819138abb3284d",
    "name": "Alice Smith",
    "email": "alice@example.com"
  }'

# Response
{
  "ok": true,
  "id": "user:alice",
  "rev": "2-7051cbe5c8faecd085a3fa619e6e6337"
}
```

**Delete Document:**
```bash
# Delete (requires _rev)
curl -X DELETE http://localhost:5984/mydb/user:alice?rev=2-7051cbe5...

# Creates tombstone document (soft delete)
# Document still exists with _deleted: true
```

### Bulk Operations

**Bulk Get:**
```bash
# Get multiple documents
curl -X POST http://localhost:5984/mydb/_all_docs?include_docs=true \
  -H "Content-Type: application/json" \
  -d '{
    "keys": ["user:alice", "user:bob"]
  }'
```

**Bulk Insert/Update:**
```bash
# Bulk operations
curl -X POST http://localhost:5984/mydb/_bulk_docs \
  -H "Content-Type: application/json" \
  -d '{
    "docs": [
      {
        "_id": "user:alice",
        "name": "Alice"
      },
      {
        "_id": "user:bob",
        "name": "Bob"
      }
    ]
  }'

# Response
[
  {
    "ok": true,
    "id": "user:alice",
    "rev": "1-abc123"
  },
  {
    "ok": true,
    "id": "user:bob",
    "rev": "1-def456"
  }
]
```

### Query Documents

**All Documents:**
```bash
# Get all document IDs
curl http://localhost:5984/mydb/_all_docs

# Include documents
curl http://localhost:5984/mydb/_all_docs?include_docs=true

# Pagination
curl "http://localhost:5984/mydb/_all_docs?limit=10&skip=20"

# Range query (by key)
curl "http://localhost:5984/mydb/_all_docs?startkey=\"user:a\"&endkey=\"user:z\""
```

---

## 4. Mango Query Language

### Basic Queries

**Find Documents:**
```bash
# Simple query
curl -X POST http://localhost:5984/mydb/_find \
  -H "Content-Type: application/json" \
  -d '{
    "selector": {
      "type": "user",
      "age": {"$gt": 25}
    }
  }'
```

**Selector Operators:**
```json
{
  "selector": {
    // Equality
    "name": "Alice",

    // Comparison
    "age": {"$gt": 25},
    "age": {"$gte": 25},
    "age": {"$lt": 65},
    "age": {"$lte": 65},
    "age": {"$ne": 30},

    // Logical
    "$and": [
      {"age": {"$gt": 25}},
      {"age": {"$lt": 65}}
    ],
    "$or": [
      {"status": "active"},
      {"status": "pending"}
    ],
    "$not": {
      "status": "deleted"
    },

    // Existence
    "email": {"$exists": true},

    // Type checking
    "age": {"$type": "number"},

    // Array
    "tags": {"$in": ["admin", "moderator"]},
    "tags": {"$nin": ["banned"]},
    "tags": {"$all": ["verified", "active"]},
    "tags": {"$size": 3},

    // Regex
    "email": {"$regex": ".*@example\\.com$"},

    // Element match
    "addresses": {
      "$elemMatch": {
        "city": "Springfield",
        "state": "IL"
      }
    }
  }
}
```

### Sorting and Pagination

**Sort Results:**
```json
{
  "selector": {
    "type": "user"
  },
  "sort": [
    {"name": "asc"},
    {"created_at": "desc"}
  ],
  "limit": 10,
  "skip": 0
}
```

**Field Projection:**
```json
{
  "selector": {
    "type": "user"
  },
  "fields": ["_id", "name", "email"],
  "limit": 100
}
```

### Indexes for Mango

**Create Index:**
```bash
# Create index for query optimization
curl -X POST http://localhost:5984/mydb/_index \
  -H "Content-Type: application/json" \
  -d '{
    "index": {
      "fields": ["type", "age"]
    },
    "name": "type-age-index",
    "type": "json"
  }'
```

**List Indexes:**
```bash
# View all indexes
curl http://localhost:5984/mydb/_index

# Response
{
  "indexes": [
    {
      "ddoc": "_design/type-age-index",
      "name": "type-age-index",
      "type": "json",
      "def": {
        "fields": [
          {"type": "asc"},
          {"age": "asc"}
        ]
      }
    }
  ]
}
```

**Use Indexes:**
```json
{
  "selector": {
    "type": "user",
    "age": {"$gt": 25}
  },
  "use_index": "type-age-index"
}
```

### Complex Queries

**Nested Queries:**
```json
{
  "selector": {
    "$and": [
      {
        "$or": [
          {"status": "active"},
          {"status": "pending"}
        ]
      },
      {
        "age": {"$gte": 18}
      },
      {
        "email": {"$exists": true}
      }
    ]
  },
  "sort": [{"created_at": "desc"}],
  "limit": 50
}
```

**Query Nested Objects:**
```json
{
  "selector": {
    "address.city": "Springfield",
    "address.state": "IL"
  }
}
```

---

## 5. MapReduce Views

### View Basics

**Create Design Document with View:**
```bash
curl -X PUT http://localhost:5984/mydb/_design/users \
  -H "Content-Type: application/json" \
  -d '{
    "_id": "_design/users",
    "views": {
      "by_email": {
        "map": "function(doc) { if (doc.type === \"user\") { emit(doc.email, doc.name); } }"
      }
    }
  }'
```

**Map Function:**
```javascript
function(doc) {
  // Emit key-value pairs
  if (doc.type === 'user') {
    emit(doc.email, {
      name: doc.name,
      created_at: doc.created_at
    });
  }
}

// emit(key, value)
// - key: Used for indexing and querying
// - value: Returned in query results
```

**Query View:**
```bash
# Query by key
curl http://localhost:5984/mydb/_design/users/_view/by_email?key="alice@example.com"

# Range query
curl "http://localhost:5984/mydb/_design/users/_view/by_email?startkey=\"a\"&endkey=\"m\""

# Descending order
curl "http://localhost:5984/mydb/_design/users/_view/by_email?descending=true"

# Limit results
curl "http://localhost:5984/mydb/_design/users/_view/by_email?limit=10"
```

### Reduce Functions

**Built-in Reduce Functions:**
```javascript
{
  "views": {
    "count_by_type": {
      "map": "function(doc) { emit(doc.type, 1); }",
      "reduce": "_count"  // Built-in: counts rows
    },
    "sum_by_category": {
      "map": "function(doc) { if (doc.type === 'order') { emit(doc.category, doc.total); } }",
      "reduce": "_sum"  // Built-in: sums values
    },
    "stats_by_product": {
      "map": "function(doc) { if (doc.type === 'sale') { emit(doc.product_id, doc.price); } }",
      "reduce": "_stats"  // Built-in: min, max, sum, count, sumsqr
    }
  }
}
```

**Custom Reduce:**
```javascript
{
  "views": {
    "average_by_category": {
      "map": "function(doc) { if (doc.type === 'product') { emit(doc.category, doc.price); } }",
      "reduce": "function(keys, values, rereduce) { return sum(values) / values.length; }"
    }
  }
}
```

### Advanced Views

**Complex Keys (Composite):**
```javascript
function(doc) {
  if (doc.type === 'order') {
    // Emit array as key for hierarchical queries
    emit([doc.user_id, doc.created_at], doc.total);
  }
}

// Query by user
// ?startkey=["user:alice"]&endkey=["user:alice", {}]

// Query by user and date range
// ?startkey=["user:alice", "2024-01-01"]&endkey=["user:alice", "2024-12-31"]
```

**Linked Documents:**
```javascript
function(doc) {
  if (doc.type === 'order') {
    // Emit user_id to enable linked document fetch
    emit(doc._id, {_id: doc.user_id});
  }
}

// Query with include_docs to fetch linked documents
// ?include_docs=true
```

**Collation Views:**
```javascript
function(doc) {
  if (doc.type === 'user') {
    // Multiple emits for different views
    emit(['by_name', doc.name], null);
    emit(['by_email', doc.email], null);
    emit(['by_created', doc.created_at], null);
  }
}

// Query specific collation
// ?startkey=["by_name"]&endkey=["by_name", {}]
```

---

## 6. Indexes and Performance

### Index Types

**Mango JSON Indexes:**
```bash
# Create index
curl -X POST http://localhost:5984/mydb/_index \
  -H "Content-Type: application/json" \
  -d '{
    "index": {
      "fields": ["type", "status", "created_at"]
    },
    "name": "type-status-created-idx",
    "type": "json"
  }'
```

**Text Indexes (Lucene):**
```bash
# Create text index for full-text search
curl -X POST http://localhost:5984/mydb/_index \
  -H "Content-Type: application/json" \
  -d '{
    "index": {
      "fields": [
        {
          "name": "name",
          "type": "string"
        },
        {
          "name": "description",
          "type": "string"
        }
      ]
    },
    "name": "text-search-idx",
    "type": "text"
  }'

# Query text index
curl -X POST http://localhost:5984/mydb/_find \
  -H "Content-Type: application/json" \
  -d '{
    "selector": {
      "$text": "search terms"
    }
  }'
```

### Index Management

**View All Indexes:**
```bash
curl http://localhost:5984/mydb/_index

# Response
{
  "total_rows": 3,
  "indexes": [
    {
      "ddoc": null,
      "name": "_all_docs",
      "type": "special",
      "def": {"fields": [{"_id": "asc"}]}
    },
    {
      "ddoc": "_design/idx-type-status",
      "name": "type-status-idx",
      "type": "json",
      "def": {"fields": [{"type": "asc"}, {"status": "asc"}]}
    }
  ]
}
```

**Delete Index:**
```bash
# Delete Mango index
curl -X DELETE "http://localhost:5984/mydb/_index/_design/idx-type-status/json/type-status-idx"

# Delete design document (removes all views)
curl -X DELETE "http://localhost:5984/mydb/_design/users?rev=1-abc123"
```

### Query Performance

**Explain Query Plan:**
```bash
# Analyze query execution
curl -X POST http://localhost:5984/mydb/_explain \
  -H "Content-Type: application/json" \
  -d '{
    "selector": {
      "type": "user",
      "age": {"$gt": 25}
    }
  }'

# Response shows:
# - Index used
# - Selector used
# - Query complexity
# - Estimated cost
```

**View Index Building:**
```bash
# Trigger view index update
curl http://localhost:5984/mydb/_design/users/_view/by_email?limit=0

# Check view index status
curl http://localhost:5984/mydb/_design/users/_info

# Response
{
  "name": "users",
  "view_index": {
    "compact_running": false,
    "updater_running": false,
    "waiting_commit": false,
    "waiting_clients": 0,
    "update_seq": 1000
  }
}
```

### Indexing Best Practices

**Index Design:**
```javascript
// ✅ GOOD: Index frequently queried fields
{
  "index": {
    "fields": ["type", "status"]  // Commonly queried together
  }
}

// ✅ GOOD: Most selective field first
{
  "index": {
    "fields": ["user_id", "created_at"]  // user_id is more selective
  }
}

// ❌ BAD: Too many fields
{
  "index": {
    "fields": ["a", "b", "c", "d", "e", "f"]  // Rarely all used
  }
}

// ❌ BAD: Non-selective fields
{
  "index": {
    "fields": ["is_active"]  // Boolean: only 2 values
  }
}
```

---

## 7. Replication and Sync

### Replication Basics

**One-Time Replication:**
```bash
# Push replication (local -> remote)
curl -X POST http://localhost:5984/_replicate \
  -H "Content-Type: application/json" \
  -d '{
    "source": "http://localhost:5984/mydb",
    "target": "http://remote:5984/mydb"
  }'

# Pull replication (remote -> local)
curl -X POST http://localhost:5984/_replicate \
  -H "Content-Type: application/json" \
  -d '{
    "source": "http://remote:5984/mydb",
    "target": "http://localhost:5984/mydb"
  }'
```

**Continuous Replication:**
```bash
# Continuous sync
curl -X POST http://localhost:5984/_replicate \
  -H "Content-Type: application/json" \
  -d '{
    "source": "http://localhost:5984/mydb",
    "target": "http://remote:5984/mydb",
    "continuous": true,
    "create_target": true
  }'
```

### Replication with _replicator Database

**Create Replication Document:**
```bash
# Persistent replication via _replicator database
curl -X PUT http://localhost:5984/_replicator/mydb-sync \
  -H "Content-Type: application/json" \
  -d '{
    "_id": "mydb-sync",
    "source": "http://localhost:5984/mydb",
    "target": "http://remote:5984/mydb",
    "continuous": true,
    "create_target": true,
    "owner": "admin"
  }'
```

**Monitor Replication:**
```bash
# Check replication status
curl http://localhost:5984/_replicator/mydb-sync

# Response
{
  "_id": "mydb-sync",
  "_rev": "2-def456",
  "source": "http://localhost:5984/mydb",
  "target": "http://remote:5984/mydb",
  "continuous": true,
  "_replication_state": "triggered",
  "_replication_state_time": "2024-02-06T10:00:00Z",
  "_replication_stats": {
    "revisions_checked": 1000,
    "missing_revisions_found": 0,
    "docs_read": 1000,
    "docs_written": 1000,
    "doc_write_failures": 0
  }
}
```

**Cancel Replication:**
```bash
# Delete replication document
curl -X DELETE "http://localhost:5984/_replicator/mydb-sync?rev=2-def456"
```

### Selective Replication

**Filter by Document Type:**
```bash
# Replicate only specific documents
curl -X POST http://localhost:5984/_replicate \
  -H "Content-Type: application/json" \
  -d '{
    "source": "http://localhost:5984/mydb",
    "target": "http://remote:5984/mydb",
    "selector": {
      "type": "user",
      "status": "active"
    }
  }'
```

**Filter Function:**
```javascript
// Create design document with filter
{
  "_id": "_design/filters",
  "filters": {
    "users_only": "function(doc, req) { return doc.type === 'user'; }"
  }
}

// Use filter in replication
{
  "source": "http://localhost:5984/mydb",
  "target": "http://remote:5984/mydb",
  "filter": "filters/users_only",
  "continuous": true
}
```

### Bidirectional Sync

**Two-Way Replication:**
```bash
# Create bidirectional sync
# Replication 1: A -> B
curl -X PUT http://localhost:5984/_replicator/a-to-b \
  -d '{
    "source": "http://nodeA:5984/mydb",
    "target": "http://nodeB:5984/mydb",
    "continuous": true
  }'

# Replication 2: B -> A
curl -X PUT http://localhost:5984/_replicator/b-to-a \
  -d '{
    "source": "http://nodeB:5984/mydb",
    "target": "http://nodeA:5984/mydb",
    "continuous": true
  }'

# Conflicts automatically detected and flagged
```

---

## 8. Clustering and Sharding

### Cluster Setup

**Single Node vs. Cluster:**
```
Single Node:
- Standalone CouchDB instance
- No high availability
- Simpler setup

Cluster (3+ nodes recommended):
- Distributed across nodes
- High availability
- Automatic failover
- Horizontal scaling
```

**Start Cluster Nodes:**
```bash
# Node 1
docker run -d --name couch1 \
  -e COUCHDB_USER=admin \
  -e COUCHDB_PASSWORD=password \
  -e COUCHDB_SECRET=secret-cookie \
  -p 5984:5984 \
  couchdb:3

# Node 2
docker run -d --name couch2 \
  -e COUCHDB_USER=admin \
  -e COUCHDB_PASSWORD=password \
  -e COUCHDB_SECRET=secret-cookie \
  -p 5985:5984 \
  couchdb:3

# Node 3
docker run -d --name couch3 \
  -e COUCHDB_USER=admin \
  -e COUCHDB_PASSWORD=password \
  -e COUCHDB_SECRET=secret-cookie \
  -p 5986:5984 \
  couchdb:3
```

**Enable Cluster:**
```bash
# Enable cluster mode on first node
curl -X POST \
  http://admin:password@localhost:5984/_cluster_setup \
  -H "Content-Type: application/json" \
  -d '{
    "action": "enable_cluster",
    "bind_address": "0.0.0.0",
    "username": "admin",
    "password": "password",
    "node_count": 3
  }'

# Add other nodes
curl -X POST \
  http://admin:password@localhost:5984/_cluster_setup \
  -H "Content-Type: application/json" \
  -d '{
    "action": "add_node",
    "host": "couch2",
    "port": 5984,
    "username": "admin",
    "password": "password"
  }'

# Finish cluster setup
curl -X POST \
  http://admin:password@localhost:5984/_cluster_setup \
  -H "Content-Type: application/json" \
  -d '{"action": "finish_cluster"}'
```

### Sharding Configuration

**Default Sharding:**
```
CouchDB automatically shards databases:

Default settings:
- q=8  (8 shards per database)
- n=3  (3 replicas per shard)

Example: 3-node cluster, q=8, n=3
- 8 shards created
- Each shard replicated 3 times
- 24 total shard copies distributed across nodes
```

**Custom Sharding:**
```bash
# Create database with custom sharding
curl -X PUT \
  "http://admin:password@localhost:5984/mydb?q=16&n=3"

# q=16: 16 shards (more shards = better distribution)
# n=3:  3 replicas (quorum = n/2 + 1 = 2)
```

**View Shard Map:**
```bash
# View how shards are distributed
curl http://admin:password@localhost:5984/mydb/_shards

# Response
{
  "shards": {
    "00000000-1fffffff": [
      "couchdb@couch1",
      "couchdb@couch2",
      "couchdb@couch3"
    ],
    "20000000-3fffffff": [
      "couchdb@couch1",
      "couchdb@couch2",
      "couchdb@couch3"
    ]
    // ... more shards
  }
}
```

### Cluster Management

**Node Status:**
```bash
# Check cluster membership
curl http://admin:password@localhost:5984/_membership

# Response
{
  "all_nodes": [
    "couchdb@couch1",
    "couchdb@couch2",
    "couchdb@couch3"
  ],
  "cluster_nodes": [
    "couchdb@couch1",
    "couchdb@couch2",
    "couchdb@couch3"
  ]
}
```

**Add/Remove Nodes:**
```bash
# Add node to cluster
curl -X PUT \
  http://admin:password@localhost:5984/_node/_local/_config/couchdb/uuid \
  -d '"unique-uuid-here"'

# Remove node (drain first)
# 1. Stop writes to node
# 2. Wait for replication to complete
# 3. Remove from cluster
curl -X DELETE \
  http://admin:password@localhost:5984/_node/couchdb@couch3
```

---

## 9. Conflict Resolution

### Understanding Conflicts

**How Conflicts Occur:**
```
Multi-master replication can create conflicts:

Timeline:
1. Node A and Node B both have doc rev 1-abc
2. Node A updates to rev 2-def (offline)
3. Node B updates to rev 2-ghi (offline)
4. Nodes sync
5. Conflict detected: two rev 2-xxx versions exist
```

**Conflict Detection:**
```bash
# Query for conflicts
curl http://localhost:5984/mydb/user:alice?conflicts=true

# Response (with conflict)
{
  "_id": "user:alice",
  "_rev": "2-def456",
  "name": "Alice Smith",
  "_conflicts": ["2-ghi789"]  // Conflicting revision
}
```

### Automatic Conflict Resolution

**Deterministic Winner:**
```
CouchDB automatically picks "winning" revision:
- Lexicographically highest revision ID
- Both revisions kept as branches
- Application can resolve conflict

Example:
rev 2-def456 (winner)
rev 2-ghi789 (loser, but kept)
```

### Manual Conflict Resolution

**Read All Revisions:**
```bash
# Get all conflicting revisions
curl "http://localhost:5984/mydb/user:alice?conflicts=true"

# Get specific revision
curl "http://localhost:5984/mydb/user:alice?rev=2-ghi789"

# Get all revisions with details
curl "http://localhost:5984/mydb/user:alice?open_revs=all"
```

**Resolve Conflict:**
```bash
# Strategy 1: Keep winner, delete losers
curl -X DELETE \
  "http://localhost:5984/mydb/user:alice?rev=2-ghi789"

# Strategy 2: Merge and create new revision
curl -X PUT http://localhost:5984/mydb/user:alice \
  -H "Content-Type: application/json" \
  -d '{
    "_id": "user:alice",
    "_rev": "2-def456",
    "name": "Alice Johnson",
    "email": "alice@example.com",
    "_deleted_conflicts": ["2-ghi789"]
  }'

# Strategy 3: Application-specific merge
# Read both revisions
# Apply business logic
# Write merged document
# Delete conflict revisions
```

### Conflict-Free Design

**Use Timestamps:**
```json
{
  "_id": "counter",
  "counts": {
    "2024-02-06T10:00:00Z": 1,
    "2024-02-06T10:01:00Z": 2,
    "2024-02-06T10:02:00Z": 3
  }
}

// Conflicts merge naturally (different keys)
// Application sums all values
```

**Append-Only Patterns:**
```json
{
  "_id": "log:2024-02-06",
  "entries": [
    {"time": "10:00:00", "event": "login"},
    {"time": "10:01:00", "event": "logout"}
  ]
}

// Add entries, never modify
// Conflicts less likely
```

**Use Arrays with Unique IDs:**
```json
{
  "_id": "shopping-cart",
  "items": [
    {"item_id": "uuid-1", "quantity": 2},
    {"item_id": "uuid-2", "quantity": 1}
  ]
}

// Use item_id to merge conflicts
// Deduplicate by item_id
```

---

## 10. Security and Authentication

### Admin Party Mode

**Default Setup (Insecure):**
```bash
# Fresh CouchDB has no admin (admin party)
# Anyone can do anything

# FIRST THING: Create admin user
curl -X PUT http://localhost:5984/_node/_local/_config/admins/admin \
  -d '"password"'

# Or via HTTP API
curl -X PUT http://localhost:5984/_users/org.couchdb.user:admin \
  -H "Content-Type: application/json" \
  -d '{
    "_id": "org.couchdb.user:admin",
    "name": "admin",
    "password": "password",
    "roles": ["_admin"],
    "type": "user"
  }'
```

### User Management

**Create User:**
```bash
# Create regular user
curl -X PUT http://admin:password@localhost:5984/_users/org.couchdb.user:alice \
  -H "Content-Type: application/json" \
  -d '{
    "_id": "org.couchdb.user:alice",
    "name": "alice",
    "password": "alicepassword",
    "roles": ["user"],
    "type": "user"
  }'
```

**User Document Structure:**
```json
{
  "_id": "org.couchdb.user:alice",
  "_rev": "1-abc123",
  "name": "alice",
  "roles": ["user", "editor"],
  "type": "user",
  "password_scheme": "pbkdf2",
  "iterations": 10,
  "derived_key": "...",
  "salt": "..."
}
```

### Database Security

**Set Database Security:**
```bash
# Configure database permissions
curl -X PUT http://admin:password@localhost:5984/mydb/_security \
  -H "Content-Type: application/json" \
  -d '{
    "admins": {
      "names": ["admin"],
      "roles": ["_admin"]
    },
    "members": {
      "names": ["alice", "bob"],
      "roles": ["user"]
    }
  }'

# admins: Can create/delete database, modify security
# members: Can read/write documents
```

**Per-Document Access Control:**
```javascript
// Validate document update function
function(newDoc, oldDoc, userCtx) {
  // Only admins can delete
  if (newDoc._deleted && userCtx.roles.indexOf('_admin') === -1) {
    throw({forbidden: 'Only admins can delete documents'});
  }

  // Users can only edit their own documents
  if (newDoc.owner !== userCtx.name && userCtx.roles.indexOf('_admin') === -1) {
    throw({forbidden: 'You can only edit your own documents'});
  }
}
```

### Authentication Methods

**Basic Auth:**
```bash
# HTTP Basic Authentication
curl -u admin:password http://localhost:5984/_all_dbs
```

**Cookie Auth:**
```bash
# Get session cookie
curl -X POST http://localhost:5984/_session \
  -H "Content-Type: application/json" \
  -d '{"name": "admin", "password": "password"}'

# Response
{
  "ok": true,
  "name": "admin",
  "roles": ["_admin"]
}

# Use cookie for subsequent requests
curl -b cookie.txt http://localhost:5984/_all_dbs
```

**JWT Auth:**
```bash
# Configure JWT (in local.ini)
[jwt_auth]
required_claims = exp

[jwt_keys]
hmac:_default = aGVsbG8=

# Use JWT token
curl -H "Authorization: Bearer <jwt-token>" \
  http://localhost:5984/_all_dbs
```

### CORS Configuration

**Enable CORS:**
```bash
# Enable CORS for web applications
curl -X PUT http://admin:password@localhost:5984/_node/_local/_config/httpd/enable_cors \
  -d '"true"'

curl -X PUT http://admin:password@localhost:5984/_node/_local/_config/cors/origins \
  -d '"https://myapp.example.com"'

curl -X PUT http://admin:password@localhost:5984/_node/_local/_config/cors/credentials \
  -d '"true"'

curl -X PUT http://admin:password@localhost:5984/_node/_local/_config/cors/methods \
  -d '"GET, POST, PUT, DELETE, OPTIONS"'
```

### SSL/TLS Configuration

**Enable HTTPS:**
```ini
# local.ini
[ssl]
enable = true
cert_file = /path/to/cert.pem
key_file = /path/to/key.pem
cacert_file = /path/to/ca.pem

[daemons]
httpsd = {couch_httpd, start_link, [https]}
```

---

## 11. Backup and Recovery

### Backup Strategies

**Replication-Based Backup:**
```bash
# Replicate to backup database
curl -X POST http://admin:password@localhost:5984/_replicate \
  -H "Content-Type: application/json" \
  -d '{
    "source": "http://localhost:5984/mydb",
    "target": "http://backup-server:5984/mydb-backup-2024-02-06",
    "create_target": true
  }'

# Benefits:
# - No downtime
# - Incremental (only changed docs)
# - Can backup to remote server
```

**File-Based Backup:**
```bash
# Stop CouchDB
systemctl stop couchdb

# Backup data directory
tar -czf couchdb-backup-$(date +%Y%m%d).tar.gz \
  /opt/couchdb/data

# Restart CouchDB
systemctl start couchdb

# Benefits:
# - Complete backup including views
# - Faster for initial backup
#
# Drawbacks:
# - Requires downtime
# - Larger backup size
```

**Per-Database Backup:**
```bash
# Replicate specific database
curl -X POST http://admin:password@localhost:5984/_replicate \
  -d '{
    "source": "mydb",
    "target": "file:///backup/mydb-$(date +%Y%m%d).couch",
    "create_target": true
  }'
```

### Scheduled Backups

**Continuous Backup Replication:**
```bash
# Create continuous backup replication
curl -X PUT http://admin:password@localhost:5984/_replicator/mydb-backup \
  -H "Content-Type: application/json" \
  -d '{
    "_id": "mydb-backup",
    "source": "http://localhost:5984/mydb",
    "target": "http://backup-server:5984/mydb-backup",
    "continuous": true,
    "create_target": true
  }'
```

**Cron-Based Backup:**
```bash
#!/bin/bash
# backup-couchdb.sh

DATE=$(date +%Y%m%d-%H%M%S)
BACKUP_DB="mydb-backup-$DATE"

curl -X POST http://admin:password@localhost:5984/_replicate \
  -H "Content-Type: application/json" \
  -d "{
    \"source\": \"http://localhost:5984/mydb\",
    \"target\": \"http://backup:5984/$BACKUP_DB\",
    \"create_target\": true
  }"

# Retention: Delete backups older than 30 days
# (Requires separate cleanup script)
```

```cron
# Daily backup at 2 AM
0 2 * * * /scripts/backup-couchdb.sh
```

### Restore Operations

**Restore from Replication:**
```bash
# Restore by replicating backup back
curl -X POST http://admin:password@localhost:5984/_replicate \
  -H "Content-Type: application/json" \
  -d '{
    "source": "http://backup-server:5984/mydb-backup-2024-02-06",
    "target": "http://localhost:5984/mydb-restored",
    "create_target": true
  }'
```

**Restore from File:**
```bash
# Stop CouchDB
systemctl stop couchdb

# Clear data directory
rm -rf /opt/couchdb/data/*

# Extract backup
tar -xzf couchdb-backup-20240206.tar.gz -C /opt/couchdb/data

# Fix permissions
chown -R couchdb:couchdb /opt/couchdb/data

# Restart CouchDB
systemctl start couchdb
```

### Point-in-Time Recovery

**Using Update Sequences:**
```bash
# Get current sequence
curl http://admin:password@localhost:5984/mydb

# Response
{
  "update_seq": "1050-g1AAAAG..."
}

# Replicate up to specific sequence
curl -X POST http://admin:password@localhost:5984/_replicate \
  -d '{
    "source": "mydb",
    "target": "mydb-restored",
    "since_seq": 0,
    "create_target": true
  }'
```

---

## 12. Monitoring and Troubleshooting

### Built-in Monitoring

**Server Stats:**
```bash
# Get server statistics
curl http://admin:password@localhost:5984/_node/_local/_stats

# Response (excerpt)
{
  "couchdb": {
    "auth_cache_hits": {"value": 1000},
    "auth_cache_misses": {"value": 50},
    "database_reads": {"value": 10000},
    "database_writes": {"value": 5000},
    "open_databases": {"value": 10},
    "open_os_files": {"value": 100},
    "request_time": {
      "min": 1,
      "max": 500,
      "mean": 25,
      "median": 20
    }
  },
  "httpd": {
    "requests": {"value": 15000},
    "bulk_requests": {"value": 100},
    "view_reads": {"value": 2000}
  }
}
```

**Active Tasks:**
```bash
# View running tasks
curl http://admin:password@localhost:5984/_active_tasks

# Response
[
  {
    "type": "indexer",
    "database": "mydb",
    "design_document": "_design/users",
    "started_on": 1707217200,
    "updated_on": 1707217250,
    "progress": 75,
    "changes_done": 7500,
    "total_changes": 10000
  },
  {
    "type": "replication",
    "replication_id": "abc123+continuous",
    "source": "http://localhost:5984/mydb",
    "target": "http://remote:5984/mydb",
    "continuous": true,
    "docs_written": 100,
    "docs_read": 100,
    "doc_write_failures": 0
  }
]
```

### Prometheus Metrics

**Configure Prometheus Exporter:**
```ini
# local.ini
[prometheus]
port = 17986

# Additional metrics configuration
```

**Scrape Configuration:**
```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'couchdb'
    static_configs:
      - targets:
        - 'localhost:17986'
```

**Key Metrics:**
```
# Request metrics
couchdb_httpd_requests_total
couchdb_httpd_status_codes_total
couchdb_httpd_request_time_seconds

# Database metrics
couchdb_database_reads_total
couchdb_database_writes_total
couchdb_database_open_databases
couchdb_database_open_files

# View metrics
couchdb_httpd_view_reads_total

# Replication metrics
couchdb_couch_replicator_jobs_total
couchdb_couch_replicator_docs_total
```

### Logging

**Log Configuration:**
```ini
# local.ini
[log]
level = info
writer = file
file = /var/log/couchdb/couch.log

# Log rotation
max_file_size = 10000000  # 10MB
rotation = 5
```

**Log Levels:**
```
debug - Detailed debug information
info  - General information messages
notice - Normal but significant events
warning - Warning messages
error - Error messages
critical - Critical conditions
alert - Action must be taken immediately
emergency - System is unusable
```

**View Logs via API:**
```bash
# Get recent log entries
curl http://admin:password@localhost:5984/_node/_local/_log

# Response
"[info] 2024-02-06T10:00:00.000000Z couchdb@localhost <0.1234.0> -------- Application couch started on node couchdb@localhost\n"
```

### Common Issues

**View Build Timeout:**
```bash
# Increase timeout
curl -X PUT http://admin:password@localhost:5984/_node/_local/_config/couchdb/os_process_timeout \
  -d '"10000"'  # 10 seconds
```

**Database Compaction:**
```bash
# Trigger manual compaction
curl -X POST \
  http://admin:password@localhost:5984/mydb/_compact

# Monitor compaction progress
curl http://admin:password@localhost:5984/mydb

# compact_running: true indicates in progress
```

**View Index Rebuild:**
```bash
# Force view rebuild
curl -X POST \
  http://admin:password@localhost:5984/mydb/_view_cleanup

# Delete design document and recreate
curl -X DELETE \
  "http://admin:password@localhost:5984/mydb/_design/users?rev=1-abc"

curl -X PUT \
  http://admin:password@localhost:5984/mydb/_design/users \
  -d '{"views": {...}}'
```

---

## 13. PouchDB and Offline-First

### PouchDB Basics

**Installation:**
```bash
# Node.js
npm install pouchdb

# Browser
<script src="https://cdn.jsdelivr.net/npm/pouchdb@8.0.0/dist/pouchdb.min.js"></script>
```

**Create Database:**
```javascript
// In-browser database
const db = new PouchDB('mydb');

// Remote CouchDB
const remoteDB = new PouchDB('http://localhost:5984/mydb', {
  auth: {
    username: 'admin',
    password: 'password'
  }
});
```

### CRUD Operations

**Create/Update:**
```javascript
// Create document
await db.put({
  _id: 'user:alice',
  name: 'Alice',
  email: 'alice@example.com'
});

// Update document
const doc = await db.get('user:alice');
doc.name = 'Alice Smith';
await db.put(doc);
```

**Read:**
```javascript
// Get document
const doc = await db.get('user:alice');

// Get all documents
const result = await db.allDocs({
  include_docs: true,
  startkey: 'user:',
  endkey: 'user:\ufff0'
});
```

**Delete:**
```javascript
const doc = await db.get('user:alice');
await db.remove(doc);
```

### Sync with CouchDB

**One-Time Sync:**
```javascript
// Push local changes to remote
await db.replicate.to(remoteDB);

// Pull remote changes to local
await db.replicate.from(remoteDB);

// Bidirectional sync
await db.sync(remoteDB);
```

**Continuous Sync:**
```javascript
// Live bidirectional sync
const sync = db.sync(remoteDB, {
  live: true,
  retry: true
}).on('change', function (change) {
  console.log('Change detected:', change);
}).on('error', function (err) {
  console.error('Sync error:', err);
}).on('paused', function () {
  console.log('Sync paused');
}).on('active', function () {
  console.log('Sync resumed');
}).on('complete', function (info) {
  console.log('Sync complete:', info);
});

// Cancel sync
sync.cancel();
```

### Offline-First Patterns

**Offline Detection:**
```javascript
let isOnline = navigator.onLine;

window.addEventListener('online', () => {
  isOnline = true;
  // Resume sync
  startSync();
});

window.addEventListener('offline', () => {
  isOnline = false;
  // Pause sync
  stopSync();
});
```

**Queue Operations:**
```javascript
class OfflineQueue {
  constructor(db) {
    this.db = db;
    this.queue = [];
  }

  async add(operation) {
    if (navigator.onLine) {
      // Execute immediately
      await operation();
    } else {
      // Queue for later
      this.queue.push(operation);
      await this.db.put({
        _id: `queue:${Date.now()}`,
        type: 'queued_operation',
        operation: operation.toString()
      });
    }
  }

  async processQueue() {
    while (this.queue.length > 0) {
      const operation = this.queue.shift();
      await operation();
    }
  }
}
```

### Conflict Handling

**Client-Side Conflict Resolution:**
```javascript
// Handle conflicts
db.get('user:alice').then(doc => {
  return db.get('user:alice', { conflicts: true });
}).then(doc => {
  if (doc._conflicts) {
    console.log('Conflicts detected:', doc._conflicts);

    // Fetch conflicting revisions
    return Promise.all(
      doc._conflicts.map(rev =>
        db.get('user:alice', { rev: rev })
      )
    ).then(conflicts => {
      // Resolve conflict (application logic)
      const resolved = resolveConflicts(doc, conflicts);

      // Save resolved version
      return db.put(resolved);
    }).then(() => {
      // Delete conflicting revisions
      return Promise.all(
        doc._conflicts.map(rev =>
          db.remove('user:alice', rev)
        )
      );
    });
  }
});
```

---

## 14. Change Feeds and Live Updates

### Change Feed Types

**Normal Feed:**
```bash
# Get all changes since sequence 0
curl "http://localhost:5984/mydb/_changes"

# Response
{
  "results": [
    {
      "seq": "1-g1AAAAG...",
      "id": "user:alice",
      "changes": [{"rev": "1-abc123"}]
    },
    {
      "seq": "2-g1AAAAG...",
      "id": "user:bob",
      "changes": [{"rev": "1-def456"}]
    }
  ],
  "last_seq": "2-g1AAAAG...",
  "pending": 0
}
```

**Long-Polling:**
```bash
# Wait for new changes
curl "http://localhost:5984/mydb/_changes?feed=longpoll&since=now"

# Blocks until new change occurs
```

**Continuous Feed:**
```bash
# Stream changes continuously
curl "http://localhost:5984/mydb/_changes?feed=continuous&heartbeat=5000"

# Streams JSON objects for each change
# Heartbeat keeps connection alive
```

### Change Feed Options

**Filter Changes:**
```bash
# Include document content
curl "http://localhost:5984/mydb/_changes?include_docs=true"

# Filter by document IDs
curl -X POST "http://localhost:5984/mydb/_changes?filter=_doc_ids" \
  -H "Content-Type: application/json" \
  -d '{"doc_ids": ["user:alice", "user:bob"]}'

# Custom filter function
curl "http://localhost:5984/mydb/_changes?filter=users/active"
```

**Create Filter Function:**
```javascript
// Design document with filter
{
  "_id": "_design/users",
  "filters": {
    "active": "function(doc, req) { return doc.type === 'user' && doc.status === 'active'; }"
  }
}
```

### Application Integration

**Node.js Change Listener:**
```javascript
const nano = require('nano')('http://admin:password@localhost:5984');
const db = nano.db.use('mydb');

// Create change feed
const feed = db.follow({
  since: 'now',
  include_docs: true
});

feed.on('change', function(change) {
  console.log('Document changed:', change.id);
  console.log('New revision:', change.doc._rev);

  // Handle change
  processChange(change.doc);
});

feed.on('error', function(err) {
  console.error('Error:', err);
});

feed.follow();

// Stop feed
// feed.stop();
```

**PouchDB Change Listener:**
```javascript
// Listen for changes
const changes = db.changes({
  since: 'now',
  live: true,
  include_docs: true
}).on('change', function(change) {
  console.log('Change detected:', change);

  // Update UI
  updateUI(change.doc);
}).on('error', function(err) {
  console.error('Error:', err);
});

// Cancel listener
changes.cancel();
```

### Real-Time Updates

**WebSocket Proxy:**
```javascript
// Client-side WebSocket listener
const ws = new WebSocket('ws://localhost:8080/changes');

ws.onmessage = function(event) {
  const change = JSON.parse(event.data);
  console.log('Change received:', change);

  // Update UI in real-time
  updateDocument(change.doc);
};

// Server-side (Node.js)
const WebSocket = require('ws');
const wss = new WebSocket.Server({ port: 8080 });

wss.on('connection', function(ws) {
  const feed = db.follow({ since: 'now', include_docs: true });

  feed.on('change', function(change) {
    ws.send(JSON.stringify(change));
  });

  ws.on('close', function() {
    feed.stop();
  });

  feed.follow();
});
```

---

## 15. Application Integration

### Python Client

**Installation:**
```bash
pip install couchdb
```

**Basic Usage:**
```python
import couchdb

# Connect to server
couch = couchdb.Server('http://admin:password@localhost:5984/')

# Create/access database
if 'mydb' not in couch:
    db = couch.create('mydb')
else:
    db = couch['mydb']

# Create document
doc_id, doc_rev = db.save({
    'type': 'user',
    'name': 'Alice',
    'email': 'alice@example.com'
})

# Read document
doc = db[doc_id]
print(doc['name'])  # Alice

# Update document
doc['name'] = 'Alice Smith'
db.save(doc)

# Delete document
db.delete(doc)

# Query view
for row in db.view('users/by_email'):
    print(row.key, row.value)
```

**Bulk Operations:**
```python
# Bulk insert
docs = [
    {'type': 'user', 'name': 'Alice'},
    {'type': 'user', 'name': 'Bob'},
    {'type': 'user', 'name': 'Carol'}
]

results = db.update(docs)

for result in results:
    if result[0]:  # Success
        print(f"Created: {result[1]}")
    else:
        print(f"Error: {result[1]}")
```

### Node.js Client (nano)

**Installation:**
```bash
npm install nano
```

**Usage:**
```javascript
const nano = require('nano')('http://admin:password@localhost:5984');

// Create database
async function setupDatabase() {
  try {
    await nano.db.create('mydb');
  } catch (err) {
    if (err.statusCode !== 412) {  // Already exists
      throw err;
    }
  }

  const db = nano.db.use('mydb');

  // Insert document
  const response = await db.insert({
    type: 'user',
    name: 'Alice',
    email: 'alice@example.com'
  }, 'user:alice');

  console.log('Created:', response.id, response.rev);

  // Get document
  const doc = await db.get('user:alice');
  console.log('Retrieved:', doc);

  // Update document
  doc.name = 'Alice Smith';
  await db.insert(doc, doc._id);

  // Delete document
  await db.destroy(doc._id, doc._rev);
}

setupDatabase().catch(console.error);
```

**Views and Queries:**
```javascript
// Query view
async function queryUsers() {
  const db = nano.db.use('mydb');

  const result = await db.view('users', 'by_email', {
    include_docs: true,
    startkey: 'a',
    endkey: 'm'
  });

  result.rows.forEach(row => {
    console.log(row.key, row.doc.name);
  });
}

// Mango query
async function findUsers() {
  const db = nano.db.use('mydb');

  const result = await db.find({
    selector: {
      type: 'user',
      age: { $gt: 25 }
    },
    limit: 10
  });

  result.docs.forEach(doc => {
    console.log(doc.name, doc.age);
  });
}
```

### Java Client (Ektorp)

**Maven Dependency:**
```xml
<dependency>
  <groupId>org.ektorp</groupId>
  <artifactId>org.ektorp</artifactId>
  <version>1.5.0</version>
</dependency>
```

**Usage:**
```java
import org.ektorp.CouchDbConnector;
import org.ektorp.CouchDbInstance;
import org.ektorp.http.HttpClient;
import org.ektorp.http.StdHttpClient;
import org.ektorp.impl.StdCouchDbConnector;
import org.ektorp.impl.StdCouchDbInstance;

public class CouchDBExample {
    public static void main(String[] args) throws Exception {
        // Connect to CouchDB
        HttpClient httpClient = new StdHttpClient.Builder()
            .url("http://localhost:5984")
            .username("admin")
            .password("password")
            .build();

        CouchDbInstance dbInstance = new StdCouchDbInstance(httpClient);
        CouchDbConnector db = new StdCouchDbConnector("mydb", dbInstance);
        db.createDatabaseIfNotExists();

        // Create document
        User user = new User();
        user.setId("user:alice");
        user.setName("Alice");
        user.setEmail("alice@example.com");

        db.create(user);

        // Read document
        User retrieved = db.get(User.class, "user:alice");
        System.out.println("Name: " + retrieved.getName());

        // Update document
        retrieved.setName("Alice Smith");
        db.update(retrieved);

        // Delete document
        db.delete(retrieved);
    }
}

// User POJO
class User {
    private String id;
    private String rev;
    private String name;
    private String email;

    // Getters and setters...
}
```

### REST API Wrappers

**cURL Examples:**
```bash
# Authentication
AUTH="admin:password"
HOST="http://localhost:5984"

# Create document
curl -X POST "$HOST/mydb" \
  -u "$AUTH" \
  -H "Content-Type: application/json" \
  -d '{"name": "Alice", "email": "alice@example.com"}'

# Get document
curl -X GET "$HOST/mydb/user:alice" \
  -u "$AUTH"

# Update document
curl -X PUT "$HOST/mydb/user:alice" \
  -u "$AUTH" \
  -H "Content-Type: application/json" \
  -d '{
    "_id": "user:alice",
    "_rev": "1-abc123",
    "name": "Alice Smith",
    "email": "alice@example.com"
  }'

# Delete document
curl -X DELETE "$HOST/mydb/user:alice?rev=2-def456" \
  -u "$AUTH"
```

---

## 16. Production Deployment

### Docker Deployment

**Single Node:**
```yaml
# docker-compose.yml
version: '3.8'

services:
  couchdb:
    image: couchdb:3
    container_name: couchdb
    ports:
      - "5984:5984"
    environment:
      - COUCHDB_USER=admin
      - COUCHDB_PASSWORD=password
    volumes:
      - couchdb_data:/opt/couchdb/data
      - couchdb_config:/opt/couchdb/etc/local.d
    restart: unless-stopped

volumes:
  couchdb_data:
  couchdb_config:
```

**Cluster Deployment:**
```yaml
version: '3.8'

services:
  couchdb1:
    image: couchdb:3
    environment:
      - COUCHDB_USER=admin
      - COUCHDB_PASSWORD=password
      - COUCHDB_SECRET=secret-cookie
      - NODENAME=couchdb1.local
    ports:
      - "5984:5984"
    volumes:
      - couch1_data:/opt/couchdb/data
    networks:
      - couchdb_network

  couchdb2:
    image: couchdb:3
    environment:
      - COUCHDB_USER=admin
      - COUCHDB_PASSWORD=password
      - COUCHDB_SECRET=secret-cookie
      - NODENAME=couchdb2.local
    ports:
      - "5985:5984"
    volumes:
      - couch2_data:/opt/couchdb/data
    networks:
      - couchdb_network

  couchdb3:
    image: couchdb:3
    environment:
      - COUCHDB_USER=admin
      - COUCHDB_PASSWORD=password
      - COUCHDB_SECRET=secret-cookie
      - NODENAME=couchdb3.local
    ports:
      - "5986:5984"
    volumes:
      - couch3_data:/opt/couchdb/data
    networks:
      - couchdb_network

volumes:
  couch1_data:
  couch2_data:
  couch3_data:

networks:
  couchdb_network:
```

### Kubernetes Deployment

**StatefulSet:**
```yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: couchdb
spec:
  serviceName: couchdb
  replicas: 3
  selector:
    matchLabels:
      app: couchdb
  template:
    metadata:
      labels:
        app: couchdb
    spec:
      containers:
      - name: couchdb
        image: couchdb:3
        ports:
        - containerPort: 5984
          name: couchdb
        - containerPort: 4369
          name: epmd
        - containerPort: 9100
          name: erlang
        env:
        - name: COUCHDB_USER
          valueFrom:
            secretKeyRef:
              name: couchdb-secret
              key: user
        - name: COUCHDB_PASSWORD
          valueFrom:
            secretKeyRef:
              name: couchdb-secret
              key: password
        - name: COUCHDB_SECRET
          valueFrom:
            secretKeyRef:
              name: couchdb-secret
              key: cookie
        - name: ERL_FLAGS
          value: "-name couchdb@$(POD_NAME).couchdb.$(POD_NAMESPACE).svc.cluster.local -setcookie $(COUCHDB_SECRET)"
        volumeMounts:
        - name: data
          mountPath: /opt/couchdb/data
        resources:
          requests:
            memory: "2Gi"
            cpu: "500m"
          limits:
            memory: "4Gi"
            cpu: "2"
  volumeClaimTemplates:
  - metadata:
      name: data
    spec:
      accessModes: ["ReadWriteOnce"]
      storageClassName: "fast-ssd"
      resources:
        requests:
          storage: 50Gi
---
apiVersion: v1
kind: Service
metadata:
  name: couchdb
spec:
  clusterIP: None
  ports:
  - port: 5984
    name: couchdb
  selector:
    app: couchdb
---
apiVersion: v1
kind: Service
metadata:
  name: couchdb-lb
spec:
  type: LoadBalancer
  ports:
  - port: 5984
    targetPort: 5984
  selector:
    app: couchdb
```

### Configuration Best Practices

**Production Configuration:**
```ini
# local.ini

[couchdb]
max_dbs_open = 500
max_document_size = 8000000  # 8MB
os_process_timeout = 10000   # 10 seconds

[httpd]
bind_address = 0.0.0.0
port = 5984
max_connections = 2048

[chttpd]
bind_address = 0.0.0.0
port = 5984
max_connections = 2048
require_valid_user = true

[log]
level = info
writer = file
file = /var/log/couchdb/couch.log

[compaction_daemon]
check_interval = 300
min_file_size = 131072

[query_server_config]
reduce_limit = true

[cluster]
q = 8
n = 3
```

### Load Balancing

**HAProxy Configuration:**
```haproxy
# haproxy.cfg
global
    log stdout local0

defaults
    mode http
    timeout connect 5s
    timeout client 50s
    timeout server 50s

frontend couchdb_front
    bind *:5984
    default_backend couchdb_back

backend couchdb_back
    balance roundrobin
    option httpchk GET /_up
    http-check expect status 200
    server couch1 couch1:5984 check
    server couch2 couch2:5984 check
    server couch3 couch3:5984 check
```

---

## 17. Performance Optimization

### Database Compaction

**Manual Compaction:**
```bash
# Compact database
curl -X POST \
  http://admin:password@localhost:5984/mydb/_compact

# Compact view indexes
curl -X POST \
  http://admin:password@localhost:5984/mydb/_compact/_design/users
```

**Automatic Compaction:**
```ini
# local.ini
[compaction_daemon]
check_interval = 300
min_file_size = 131072  # 128KB

[mydb]
_data_size = 500000000  # 500MB

[compactions]
mydb = [{db_fragmentation, "70%"}, {view_fragmentation, "60%"}]
```

### View Performance

**Stale Views:**
```bash
# Query without updating view (stale=ok)
curl "http://localhost:5984/mydb/_design/users/_view/by_email?stale=ok"

# Use cached view, update after (stale=update_after)
curl "http://localhost:5984/mydb/_design/users/_view/by_email?stale=update_after"

# Benefits:
# - Faster queries (no wait for view update)
# - Trade-off: Slightly stale data
```

**View Collation:**
```javascript
// Use built-in reduce functions (_count, _sum, _stats)
// Faster than custom JavaScript reduces

{
  "views": {
    "count_by_type": {
      "map": "function(doc) { emit(doc.type, 1); }",
      "reduce": "_count"  // Built-in, faster
    }
  }
}
```

### Caching

**Application-Level Caching:**
```javascript
const NodeCache = require('node-cache');
const cache = new NodeCache({ stdTTL: 300 }); // 5 min TTL

async function getUser(userId) {
  // Check cache first
  const cached = cache.get(userId);
  if (cached) {
    return cached;
  }

  // Query CouchDB
  const doc = await db.get(userId);

  // Cache result
  cache.set(userId, doc);

  return doc;
}

// Invalidate cache on updates
async function updateUser(userId, updates) {
  const doc = await db.get(userId);
  Object.assign(doc, updates);
  await db.insert(doc);

  // Invalidate cache
  cache.del(userId);
}
```

### Query Optimization

**Index Selection:**
```bash
# Create indexes for common queries
curl -X POST http://localhost:5984/mydb/_index \
  -H "Content-Type: application/json" \
  -d '{
    "index": {
      "fields": ["type", "status", "created_at"]
    }
  }'

# Query uses index automatically
curl -X POST http://localhost:5984/mydb/_find \
  -H "Content-Type: application/json" \
  -d '{
    "selector": {
      "type": "order",
      "status": "pending"
    },
    "sort": [{"created_at": "desc"}]
  }'
```

**Avoid Full Scans:**
```javascript
// ❌ BAD: Full database scan
{
  "selector": {
    "name": {"$regex": ".*Alice.*"}
  }
}

// ✅ GOOD: Use index
{
  "selector": {
    "type": "user",
    "name": {"$eq": "Alice"}
  }
}
```

### Connection Pooling

**Node.js Connection Pool:**
```javascript
const nano = require('nano')({
  url: 'http://admin:password@localhost:5984',
  requestDefaults: {
    pool: {
      maxSockets: 50
    }
  }
});
```

---

## 18. Migration Strategies

### From MongoDB

**Schema Mapping:**
```javascript
// MongoDB document
{
  "_id": ObjectId("507f1f77bcf86cd799439011"),
  "name": "Alice",
  "email": "alice@example.com",
  "orders": [
    {
      "order_id": "order123",
      "total": 100.00
    }
  ]
}

// CouchDB document
{
  "_id": "user:507f1f77bcf86cd799439011",
  "_rev": "1-abc123",
  "type": "user",
  "name": "Alice",
  "email": "alice@example.com",
  "orders": [
    {
      "order_id": "order:123",
      "total": 100.00
    }
  ]
}
```

**Migration Script:**
```javascript
const MongoClient = require('mongodb').MongoClient;
const nano = require('nano')('http://admin:password@localhost:5984');

async function migrate() {
  // Connect to MongoDB
  const mongoClient = await MongoClient.connect('mongodb://localhost:27017');
  const mongoDB = mongoClient.db('myapp');
  const collection = mongoDB.collection('users');

  // Connect to CouchDB
  const couchDB = nano.db.use('myapp');

  // Migrate documents
  const cursor = collection.find();
  const batch = [];

  await cursor.forEach(doc => {
    // Transform document
    const couchDoc = {
      _id: `user:${doc._id.toString()}`,
      type: 'user',
      name: doc.name,
      email: doc.email,
      orders: doc.orders || []
    };

    batch.push(couchDoc);

    // Bulk insert every 100 docs
    if (batch.length >= 100) {
      await couchDB.bulk({ docs: batch });
      batch.length = 0;
    }
  });

  // Insert remaining
  if (batch.length > 0) {
    await couchDB.bulk({ docs: batch });
  }

  await mongoClient.close();
}

migrate().catch(console.error);
```

### From SQL Database

**Relational to Document Mapping:**
```sql
-- SQL Schema
CREATE TABLE users (
  id INT PRIMARY KEY,
  name VARCHAR(100),
  email VARCHAR(100)
);

CREATE TABLE orders (
  id INT PRIMARY KEY,
  user_id INT,
  total DECIMAL(10,2),
  FOREIGN KEY (user_id) REFERENCES users(id)
);
```

**Convert to CouchDB:**
```javascript
// Strategy 1: Embed related data
{
  "_id": "user:123",
  "type": "user",
  "name": "Alice",
  "email": "alice@example.com",
  "orders": [
    {
      "order_id": "order:456",
      "total": 100.00
    }
  ]
}

// Strategy 2: Reference documents
{
  "_id": "user:123",
  "type": "user",
  "name": "Alice",
  "email": "alice@example.com"
}

{
  "_id": "order:456",
  "type": "order",
  "user_id": "user:123",
  "total": 100.00
}
```

**Migration from PostgreSQL:**
```javascript
const { Client } = require('pg');
const nano = require('nano')('http://admin:password@localhost:5984');

async function migrateFromPostgres() {
  const pgClient = new Client({
    host: 'localhost',
    database: 'myapp',
    user: 'postgres',
    password: 'password'
  });

  await pgClient.connect();
  const couchDB = nano.db.use('myapp');

  // Migrate users
  const result = await pgClient.query('SELECT * FROM users');

  const docs = result.rows.map(row => ({
    _id: `user:${row.id}`,
    type: 'user',
    name: row.name,
    email: row.email
  }));

  await couchDB.bulk({ docs });

  await pgClient.end();
}
```

### Dual-Write Pattern

**During Migration:**
```javascript
class DualWriteService {
  constructor(pgClient, couchDB) {
    this.pg = pgClient;
    this.couch = couchDB;
    this.migrationComplete = false;
  }

  async createUser(userData) {
    // Write to primary (PostgreSQL)
    const result = await this.pg.query(
      'INSERT INTO users (name, email) VALUES ($1, $2) RETURNING id',
      [userData.name, userData.email]
    );
    const userId = result.rows[0].id;

    // Write to secondary (CouchDB)
    try {
      await this.couch.insert({
        _id: `user:${userId}`,
        type: 'user',
        ...userData
      });
    } catch (err) {
      console.error('CouchDB write failed:', err);
      // Log but don't fail
    }

    return userId;
  }

  async getUser(userId) {
    // Read from CouchDB after migration
    if (this.migrationComplete) {
      return await this.couch.get(`user:${userId}`);
    }

    // Otherwise read from PostgreSQL
    const result = await this.pg.query(
      'SELECT * FROM users WHERE id = $1',
      [userId]
    );
    return result.rows[0];
  }
}
```

---

## 19. Comparison with Other Databases

### CouchDB vs. MongoDB

| Feature | CouchDB | MongoDB |
|---------|---------|---------|
| **Data Model** | JSON documents | BSON documents |
| **API** | HTTP/REST | Binary protocol |
| **Replication** | Multi-master | Master-slave |
| **Consistency** | Eventual | Tunable |
| **Offline Support** | Excellent (PouchDB) | Limited |
| **Querying** | MapReduce, Mango | Aggregation pipeline |
| **Transactions** | Single document | Multi-document (4.0+) |
| **Best For** | Offline-first, sync | General purpose |

### CouchDB vs. PostgreSQL

| Feature | CouchDB | PostgreSQL |
|---------|---------|------------|
| **Type** | NoSQL document | Relational SQL |
| **Schema** | Schema-less | Rigid schema |
| **Joins** | Application level | Native SQL joins |
| **ACID** | Document level | Full ACID |
| **Replication** | Multi-master | Master-slave |
| **Use Case** | Distributed, offline | Complex queries, transactions |

### CouchDB vs. Firebase

| Feature | CouchDB | Firebase Realtime DB |
|---------|---------|----------------------|
| **Hosting** | Self-hosted | Google Cloud only |
| **Data Model** | JSON documents | JSON tree |
| **Offline** | PouchDB sync | Built-in offline |
| **Querying** | Flexible queries | Limited queries |
| **Pricing** | Free (self-hosted) | Pay-as-you-go |
| **Lock-in** | No vendor lock-in | Google Cloud |

---

## 20. Production Checklist

### Pre-Deployment

**Infrastructure:**
- [ ] Hardware/VM sized appropriately (CPU, RAM, disk)
- [ ] Clustering configured (3+ nodes recommended)
- [ ] Load balancer setup (HAProxy, nginx)
- [ ] Backup strategy defined
- [ ] Monitoring system configured (Prometheus/Grafana)
- [ ] SSL/TLS certificates obtained

**Security:**
- [ ] Admin user created (no admin party!)
- [ ] Regular users created with appropriate roles
- [ ] Database security configured (_security documents)
- [ ] CORS configured for web apps
- [ ] Network firewall rules applied
- [ ] Audit logging enabled

**Database Configuration:**
- [ ] Compaction settings configured
- [ ] View timeout settings adjusted
- [ ] Query limits set
- [ ] Log levels configured
- [ ] Replication jobs created (if needed)

**Schema Design:**
- [ ] Document structure defined
- [ ] ID strategy chosen
- [ ] Indexes created for common queries
- [ ] Views created and tested
- [ ] Validation functions implemented (if needed)

### Post-Deployment

**Verification:**
- [ ] All nodes healthy and responding
- [ ] Cluster membership correct
- [ ] Replication working (if configured)
- [ ] Backups running successfully
- [ ] Monitoring dashboards populated
- [ ] Alerts configured and firing correctly

**Operations:**
- [ ] Backup schedule: daily incrementals
- [ ] Compaction schedule: automated
- [ ] View index updates: monitored
- [ ] Log rotation: configured
- [ ] On-call rotation: established
- [ ] Runbooks documented

### Performance Tuning

**Query Performance:**
- [ ] Slow queries identified
- [ ] Indexes optimized
- [ ] View updates monitored
- [ ] Stale view queries used where appropriate

**Configuration:**
- [ ] max_dbs_open tuned
- [ ] max_connections set appropriately
- [ ] Compaction thresholds adjusted
- [ ] View timeout increased if needed

**Monitoring Metrics:**
```
Critical Metrics:
- HTTP request rate
- HTTP request latency (p95, p99)
- Database read/write rate
- View update lag
- Disk usage
- Compaction frequency
- Replication lag (if applicable)
```

### Ongoing Maintenance

**Daily:**
- [ ] Check cluster health
- [ ] Monitor disk space
- [ ] Verify backups completed
- [ ] Review error logs

**Weekly:**
- [ ] Review performance metrics
- [ ] Check for CouchDB updates
- [ ] Analyze slow queries
- [ ] Review replication status

**Monthly:**
- [ ] Test backup restore
- [ ] Review and optimize views
- [ ] Capacity planning review
- [ ] Update documentation

**Quarterly:**
- [ ] Disaster recovery drill
- [ ] Security audit
- [ ] Version upgrade planning
- [ ] Performance tuning review

---

## References and Resources

### Official Documentation
- **CouchDB Docs:** https://docs.couchdb.org/
- **CouchDB Guide:** https://guide.couchdb.org/
- **API Reference:** https://docs.couchdb.org/en/stable/api/
- **Best Practices:** https://docs.couchdb.org/en/stable/best-practices/

### Learning Resources
- **PouchDB Docs:** https://pouchdb.com/guides/
- **CouchDB Tutorial:** https://guide.couchdb.org/
- **CouchDB Blog:** https://blog.couchdb.org/
- **Books:** "CouchDB: The Definitive Guide" (O'Reilly)

### Community
- **Slack:** https://couchdb.apache.org/slack
- **Mailing List:** https://couchdb.apache.org/#mailing-list
- **GitHub:** https://github.com/apache/couchdb
- **Stack Overflow:** `[couchdb]` tag

### Tools
- **Fauxton:** Web UI (built-in at /_utils)
- **PouchDB:** JavaScript sync library
- **PouchDB Inspector:** Browser DevTools extension
- **CouchDB Nano:** Node.js client

---

**Document Maintenance:**
- Review quarterly for CouchDB updates
- Update with new features and best practices
- Test examples with latest version
- Add community patterns and lessons learned

**Last Updated:** February 2026
**Next Review:** May 2026
