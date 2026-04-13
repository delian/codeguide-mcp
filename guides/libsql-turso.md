# libSQL & Turso Development Guidelines
Mandatory coding standards and development practices for libSQL and Turso development.

---

**Agent Profile**: The libSQL/Turso Expert
**Role**: Senior Edge Database Engineer & SQLite/libSQL Specialist
**Objective**: Generate production-ready, low-latency and reliable edge and local-first database solutions.
**Tools**: libSQL, Turso Platform, client SDKs (JS/TS, Rust, Go, Swift), HTTP/WebSocket API, embedded replicas

---

**Database Type**: Distributed Edge SQLite | **Engine**: libSQL (SQLite fork) | **Platform**: Turso (managed)  
**Companion Guides**: sql.md, sqlite.md, postgresql.md, testing.md, docker-compose.md

---

## 1. Core Philosophies: EDGE-FIRST

The agent must adhere to the **EDGE-FIRST** principles for every libSQL/Turso implementation:

**Test-Driven Development (TDD)**: ALWAYS write tests BEFORE implementation (Red-Green-Refactor cycle mandatory).
**Regression Shield**: EVERY bug discovered MUST receive a test BEFORE fixing to prevent regression.

- **E**mbedded and edge: Prefer local replicas for reads; design for sync and offline-first when applicable.
- **D**istributed awareness: Single primary writer; use replicas for read scale; respect Turso topology.
- **G**lobal and low-latency: Route reads to nearest replica; use Turso client for replica URLs.
- **E**rror handling: Check responses and Status; handle replication lag and connection failures.
- **F**ull SQLite compatibility: Use standard SQL and migrations; avoid unsupported extensions on Turso.
- **I**dempotent migrations: Schema changes safe to retry; use versioned migrations.
- **R**eplicas and sync: Understand embedded replica lifecycle; test sync and conflict behavior.
- **S**ecurity: Use auth tokens and TLS; no secrets in client code; follow Turso IAM.
- **T**esting: Test against local libSQL and Turso; verify replica and primary behavior.
- **Verified Code**: Agent-generated code MUST use parameterized SQL, handle errors, and pass tests before delivery.

---

## 2. Core Concepts

### What is libSQL?

**libSQL** is an open-source fork of SQLite with several enhancements:

```
SQLite 3.x  →  libSQL
    ├── Core SQLite compatibility ✓
    ├── Additional features:
    │   ├── Encryption at rest
    │   ├── Randomized ROWID
    │   ├── WebAssembly user-defined functions
    │   ├── Remote replication protocol
    │   └── Enhanced security features
    └── 100% SQLite SQL syntax compatible
```

### What is Turso?

**Turso** is a distributed database platform built on libSQL:

```
┌─────────────────────────────────────────────────────────┐
│                    Turso Platform                        │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Primary Database                                        │
│  ┌────────────────────────────┐                         │
│  │  libSQL Primary Instance   │                         │
│  │  (Single-writer)           │                         │
│  └──────────┬─────────────────┘                         │
│             │                                            │
│             │ Replication                                │
│             ├──────────────────────────────┐             │
│             │                              │             │
│  ┌──────────▼──────────┐       ┌──────────▼──────────┐  │
│  │  Edge Replica       │       │  Edge Replica       │  │
│  │  (Read-only)        │       │  (Read-only)        │  │
│  │  US-East            │       │  EU-West            │  │
│  └─────────────────────┘       └─────────────────────┘  │
│                                                          │
│  Embedded Replicas (in application)                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐              │
│  │ Local DB │  │ Local DB │  │ Local DB │              │
│  │ (Sync)   │  │ (Sync)   │  │ (Sync)   │              │
│  └──────────┘  └──────────┘  └──────────┘              │
└─────────────────────────────────────────────────────────┘

Key Features:
✓ Multi-region replication
✓ Embedded replicas for local-first apps
✓ Sub-millisecond read latency
✓ Automatic failover
✓ Point-in-time recovery
✓ Schema migrations
```

### Use Cases

**Perfect For:**
- Edge computing applications
- Serverless functions (Cloudflare Workers, Vercel, AWS Lambda)
- Mobile/desktop apps with offline-first requirements
- Global applications needing low-latency reads
- Multi-tenant SaaS applications
- JAMstack and static site generators
- Real-time collaborative applications

**Not Ideal For:**
- Heavy write workloads (single-writer limitation)
- Complex analytical queries (OLAP workloads)
- Extremely large datasets (>100GB per database)
- Applications requiring distributed transactions

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

### Example TDD Workflow for libSQL/Turso (Python with pytest and libsql-experimental)

```python
# Step 1: RED - Write failing test first
import pytest
import libsql_experimental as libsql

@pytest.fixture
def db_conn():
    conn = libsql.connect(":memory:")
    conn.execute("CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, name TEXT, email TEXT)")
    conn.commit()
    yield conn
    conn.close()

def test_insert_and_select_user(db_conn):
    """Test inserting and selecting a user via SQL."""
    repo = UserRepository(db_conn)
    repo.add_user("Alice", "alice@example.com")
    user = repo.get_user_by_name("Alice")
    assert user["name"] == "Alice"
    assert user["email"] == "alice@example.com"

# Run: pytest test_libsql.py -v
# FAILS - NameError: name 'UserRepository' is not defined

# Step 2: GREEN - Write minimal implementation
class UserRepository:
    def __init__(self, conn):
        self.conn = conn

    def add_user(self, name, email):
        self.conn.execute(
            "INSERT INTO users (name, email) VALUES (?, ?)",
            (name, email)
        )
        self.conn.commit()

    def get_user_by_name(self, name):
        cursor = self.conn.execute(
            "SELECT id, name, email FROM users WHERE name = ?",
            (name,)
        )
        row = cursor.fetchone()
        return {"id": row[0], "name": row[1], "email": row[2]}

# Run: pytest test_libsql.py -v
# PASSES

# Step 3: REFACTOR - Add parameterized queries and batch insert
class UserRepository:
    def __init__(self, conn):
        self.conn = conn

    def add_user(self, name, email):
        self.conn.execute(
            "INSERT INTO users (name, email) VALUES (?, ?)",
            (name, email)
        )
        self.conn.commit()

    def add_users_batch(self, users):
        self.conn.executemany(
            "INSERT INTO users (name, email) VALUES (?, ?)",
            [(u["name"], u["email"]) for u in users]
        )
        self.conn.commit()

    def get_user_by_name(self, name):
        cursor = self.conn.execute(
            "SELECT id, name, email FROM users WHERE name = ?",
            (name,)
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return {"id": row[0], "name": row[1], "email": row[2]}

    def search_users(self, query):
        cursor = self.conn.execute(
            "SELECT id, name, email FROM users WHERE name LIKE ?",
            (f"%{query}%",)
        )
        return [{"id": r[0], "name": r[1], "email": r[2]} for r in cursor.fetchall()]

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
# Bug: add_user() allows duplicate emails, violating business rule
# that emails must be unique across users

import pytest

# Step 1: Write test that reproduces the bug
def test_add_user_rejects_duplicate_email(db_conn):
    """Regression: add_user() should raise ValueError when inserting
    a user with an email that already exists."""
    repo = UserRepository(db_conn)
    repo.add_user("Alice", "alice@example.com")
    with pytest.raises(ValueError, match="Email already exists"):
        repo.add_user("Bob", "alice@example.com")

# FAILS - duplicate row inserted without error

# Step 2: Fix the bug
class UserRepository:
    # ... existing code ...

    def add_user(self, name, email):
        cursor = self.conn.execute(
            "SELECT COUNT(*) FROM users WHERE email = ?",
            (email,)
        )
        if cursor.fetchone()[0] > 0:
            raise ValueError(f"Email already exists: {email}")
        self.conn.execute(
            "INSERT INTO users (name, email) VALUES (?, ?)",
            (name, email)
        )
        self.conn.commit()

# PASSES - bug fixed, regression prevented
```

---

## 3. Architecture and Design

### Database Hierarchy

```
Organization
  └── Databases (multiple)
       ├── Primary Location (single writer)
       ├── Replica Locations (multiple readers)
       └── Groups (logical database groups)
```

### Replication Model

**Primary-Replica Architecture:**

```cypher
Write Path:
Application → Primary Database → WAL → Replicas

Read Path (Embedded Replica):
Application → Local Embedded Replica (instant)
             ↓ (periodic sync)
          Primary Database

Read Path (Edge Replica):
Application → Nearest Edge Replica → (if needed) Primary
```

**Consistency Model:**
- **Writes**: Strong consistency (single writer)
- **Reads**: Eventual consistency on replicas
- **Embedded Replicas**: Configurable sync frequency

### Data Model

**SQLite-Compatible Schema:**

```sql
-- Standard SQLite tables
CREATE TABLE users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    email TEXT NOT NULL UNIQUE,
    name TEXT NOT NULL,
    created_at INTEGER DEFAULT (unixepoch()),
    updated_at INTEGER DEFAULT (unixepoch())
);

-- Indexes
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_created ON users(created_at);

-- Triggers for updated_at
CREATE TRIGGER update_users_updated_at
    AFTER UPDATE ON users
    FOR EACH ROW
BEGIN
    UPDATE users SET updated_at = unixepoch() WHERE id = NEW.id;
END;

-- Foreign keys (must be enabled)
PRAGMA foreign_keys = ON;

CREATE TABLE posts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    title TEXT NOT NULL,
    content TEXT,
    published_at INTEGER,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);
```

### Multi-Tenancy Patterns

**Database-per-Tenant (Recommended for Turso):**

```typescript
// Create database per tenant
const tenantDb = turso.getDatabase(`tenant-${tenantId}`);

// Advantages:
// ✓ Strong isolation
// ✓ Independent scaling
// ✓ Easy backup/restore per tenant
// ✓ Compliance-friendly
// ✓ No query complexity
```

**Schema-based Multi-Tenancy:**

```sql
-- All tenants in one database (for small scale)
CREATE TABLE tenants (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    created_at INTEGER DEFAULT (unixepoch())
);

CREATE TABLE tenant_users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tenant_id TEXT NOT NULL,
    email TEXT NOT NULL,
    name TEXT NOT NULL,
    FOREIGN KEY (tenant_id) REFERENCES tenants(id),
    UNIQUE(tenant_id, email)
);

-- Always filter by tenant_id
CREATE INDEX idx_tenant_users_tenant ON tenant_users(tenant_id);

-- Row-level security via queries
SELECT * FROM tenant_users WHERE tenant_id = ?;
```

---

## 4. Setup and Installation

### Turso CLI Installation

```bash
# Install Turso CLI
curl -sSfL https://get.tur.so/install.sh | bash

# Or via Homebrew (macOS)
brew install tursodatabase/tap/turso

# Or via npm
npm install -g @turso/cli

# Verify installation
turso --version

# Login/signup
turso auth signup
turso auth login
```

### Local libSQL Installation

```bash
# Install libSQL server (for local development)
curl -sSfL https://github.com/tursodatabase/libsql/releases/latest/download/libsql-server-linux-amd64 -o libsql-server
chmod +x libsql-server

# Run local server
./libsql-server --http-addr 127.0.0.1:8080

# Or use Docker
docker run -p 8080:8080 ghcr.io/tursodatabase/libsql-server:latest
```

### Creating Your First Database

```bash
# Create a database
turso db create my-app-db

# List databases
turso db list

# Show database details
turso db show my-app-db

# Get connection URL
turso db show my-app-db --url

# Create auth token
turso db tokens create my-app-db

# Create replica in another region
turso db replicate my-app-db --region fra

# List replicas
turso db show my-app-db
```

### Database Groups

```bash
# Create database group
turso group create production --location iad

# Add database to group
turso db create prod-db --group production

# List groups
turso group list

# Update group locations
turso group locations add production fra
turso group locations list production
```

---

## 5. Client SDKs and Connections

### JavaScript/TypeScript SDK

**Installation:**

```bash
npm install @libsql/client
# or
pnpm add @libsql/client
```

**Connection Configuration:**

```typescript
import { createClient } from '@libsql/client';

// Remote connection (Turso)
const client = createClient({
  url: process.env.TURSO_DATABASE_URL!,
  authToken: process.env.TURSO_AUTH_TOKEN!,
});

// Local file database
const localClient = createClient({
  url: 'file:./local.db',
});

// In-memory database
const memClient = createClient({
  url: ':memory:',
});

// Local development server
const devClient = createClient({
  url: 'http://127.0.0.1:8080',
});
```

**Basic Operations:**

```typescript
// Execute query
const result = await client.execute('SELECT * FROM users');
console.log(result.rows);

// Parameterized query (prevent SQL injection)
const user = await client.execute({
  sql: 'SELECT * FROM users WHERE email = ?',
  args: ['alice@example.com'],
});

// Named parameters
const user2 = await client.execute({
  sql: 'SELECT * FROM users WHERE email = :email',
  args: { email: 'bob@example.com' },
});

// Insert and get last insert ID
const insert = await client.execute({
  sql: 'INSERT INTO users (email, name) VALUES (?, ?)',
  args: ['charlie@example.com', 'Charlie'],
});
console.log('Inserted ID:', insert.lastInsertRowid);

// Batch operations
const batch = await client.batch([
  { sql: 'INSERT INTO users (email, name) VALUES (?, ?)', args: ['dave@example.com', 'Dave'] },
  { sql: 'INSERT INTO users (email, name) VALUES (?, ?)', args: ['eve@example.com', 'Eve'] },
]);
```

**Transactions:**

```typescript
// Execute transaction
const transaction = await client.transaction('write');
try {
  await transaction.execute({
    sql: 'INSERT INTO users (email, name) VALUES (?, ?)',
    args: ['frank@example.com', 'Frank'],
  });

  await transaction.execute({
    sql: 'INSERT INTO posts (user_id, title) VALUES (?, ?)',
    args: [1, 'My First Post'],
  });

  await transaction.commit();
} catch (error) {
  await transaction.rollback();
  throw error;
}
```

### Embedded Replicas (Local-First)

**Sync Configuration:**

```typescript
import { createClient } from '@libsql/client';

// Embedded replica with automatic sync
const client = createClient({
  url: 'file:./local-replica.db',
  syncUrl: process.env.TURSO_DATABASE_URL!,
  authToken: process.env.TURSO_AUTH_TOKEN!,
  syncInterval: 60, // Sync every 60 seconds
});

// Manual sync
await client.sync();

// Read from local replica (instant)
const users = await client.execute('SELECT * FROM users');

// Write operations go to remote (automatically)
await client.execute({
  sql: 'INSERT INTO users (email, name) VALUES (?, ?)',
  args: ['local@example.com', 'Local User'],
});
```

**Sync Strategies:**

```typescript
// Periodic sync
const client = createClient({
  url: 'file:./local.db',
  syncUrl: remoteUrl,
  authToken: token,
  syncInterval: 5000, // Every 5 seconds
});

// Manual sync on-demand
async function syncOnDemand() {
  const syncResult = await client.sync();
  console.log('Synced frames:', syncResult.framesApplied);
}

// Sync before critical reads
async function getUserProfile(userId: number) {
  await client.sync(); // Ensure latest data
  return await client.execute({
    sql: 'SELECT * FROM users WHERE id = ?',
    args: [userId],
  });
}
```

### Python SDK

**Installation:**

```bash
pip install libsql-client
```

**Usage:**

```python
import libsql_client

# Connect to Turso
client = libsql_client.create_client(
    url=os.environ["TURSO_DATABASE_URL"],
    auth_token=os.environ["TURSO_AUTH_TOKEN"]
)

# Execute query
result = client.execute("SELECT * FROM users")
for row in result.rows:
    print(row)

# Parameterized query
result = client.execute(
    "SELECT * FROM users WHERE email = ?",
    ["alice@example.com"]
)

# Insert
result = client.execute(
    "INSERT INTO users (email, name) VALUES (?, ?)",
    ["test@example.com", "Test User"]
)
print(f"Inserted ID: {result.last_insert_rowid}")

# Batch operations
client.batch([
    ("INSERT INTO users (email, name) VALUES (?, ?)", ["user1@test.com", "User 1"]),
    ("INSERT INTO users (email, name) VALUES (?, ?)", ["user2@test.com", "User 2"]),
])

# Close connection
client.close()
```

### Rust SDK

**Cargo.toml:**

```toml
[dependencies]
libsql = "0.3"
tokio = { version = "1", features = ["full"] }
```

**Usage:**

```rust
use libsql::Builder;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Connect to Turso
    let db = Builder::new_remote(
        std::env::var("TURSO_DATABASE_URL")?,
        std::env::var("TURSO_AUTH_TOKEN")?
    )
    .build()
    .await?;

    let conn = db.connect()?;

    // Execute query
    let mut rows = conn.query("SELECT * FROM users", ()).await?;

    while let Some(row) = rows.next().await? {
        let id: i64 = row.get(0)?;
        let email: String = row.get(1)?;
        println!("User: {} - {}", id, email);
    }

    // Insert
    conn.execute(
        "INSERT INTO users (email, name) VALUES (?1, ?2)",
        [("alice@example.com", "Alice")]
    ).await?;

    Ok(())
}
```

### Go SDK

**Installation:**

```bash
go get github.com/tursodatabase/libsql-client-go/libsql
```

**Usage:**

```go
package main

import (
    "database/sql"
    "fmt"
    "os"

    _ "github.com/tursodatabase/libsql-client-go/libsql"
)

func main() {
    url := os.Getenv("TURSO_DATABASE_URL")
    token := os.Getenv("TURSO_AUTH_TOKEN")

    connector, err := libsql.NewConnector(url, libsql.WithAuthToken(token))
    if err != nil {
        panic(err)
    }

    db := sql.OpenDB(connector)
    defer db.Close()

    // Query
    rows, err := db.Query("SELECT id, email, name FROM users")
    if err != nil {
        panic(err)
    }
    defer rows.Close()

    for rows.Next() {
        var id int
        var email, name string
        rows.Scan(&id, &email, &name)
        fmt.Printf("User: %d - %s (%s)\n", id, email, name)
    }

    // Insert
    result, err := db.Exec(
        "INSERT INTO users (email, name) VALUES (?, ?)",
        "test@example.com", "Test User",
    )
    if err != nil {
        panic(err)
    }

    lastID, _ := result.LastInsertId()
    fmt.Printf("Inserted ID: %d\n", lastID)
}
```

---

## 6. Schema Management

### Schema Design Best Practices

```sql
-- Use INTEGER PRIMARY KEY for rowid optimization
CREATE TABLE users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    email TEXT NOT NULL UNIQUE,
    name TEXT NOT NULL,
    status TEXT DEFAULT 'active' CHECK(status IN ('active', 'suspended', 'deleted')),
    created_at INTEGER DEFAULT (unixepoch()),
    updated_at INTEGER DEFAULT (unixepoch())
);

-- Create indexes for common queries
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_status ON users(status) WHERE status = 'active';

-- Partial indexes for filtered queries
CREATE INDEX idx_active_users_created
    ON users(created_at)
    WHERE status = 'active';

-- Covering indexes (include columns in index)
CREATE INDEX idx_users_email_name ON users(email, name);

-- Full-text search (FTS5)
CREATE VIRTUAL TABLE users_fts USING fts5(
    name,
    email,
    content=users,
    content_rowid=id
);

-- Trigger to keep FTS in sync
CREATE TRIGGER users_fts_insert AFTER INSERT ON users BEGIN
    INSERT INTO users_fts(rowid, name, email) VALUES (new.id, new.name, new.email);
END;

CREATE TRIGGER users_fts_update AFTER UPDATE ON users BEGIN
    UPDATE users_fts SET name = new.name, email = new.email WHERE rowid = new.id;
END;

CREATE TRIGGER users_fts_delete AFTER DELETE ON users BEGIN
    DELETE FROM users_fts WHERE rowid = old.id;
END;
```

### Migrations with Turso CLI

```bash
# Create migration file
mkdir -p migrations
cat > migrations/001_create_users.sql << 'EOF'
-- Migration: Create users table
-- Date: 2026-02-06

CREATE TABLE users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    email TEXT NOT NULL UNIQUE,
    name TEXT NOT NULL,
    created_at INTEGER DEFAULT (unixepoch())
);

CREATE INDEX idx_users_email ON users(email);
EOF

# Apply migration
turso db shell my-app-db < migrations/001_create_users.sql

# Or use interactive shell
turso db shell my-app-db
```

### Schema Migrations with Code

**TypeScript Migration Runner:**

```typescript
import { createClient } from '@libsql/client';
import fs from 'fs';
import path from 'path';

interface Migration {
  version: number;
  name: string;
  sql: string;
}

class MigrationRunner {
  constructor(private client: ReturnType<typeof createClient>) {}

  async initialize() {
    // Create migrations table
    await this.client.execute(`
      CREATE TABLE IF NOT EXISTS schema_migrations (
        version INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        applied_at INTEGER DEFAULT (unixepoch())
      )
    `);
  }

  async getCurrentVersion(): Promise<number> {
    const result = await this.client.execute(
      'SELECT MAX(version) as version FROM schema_migrations'
    );
    return (result.rows[0]?.version as number) || 0;
  }

  async loadMigrations(dir: string): Promise<Migration[]> {
    const files = fs.readdirSync(dir).sort();
    return files.map(file => {
      const match = file.match(/^(\d+)_(.+)\.sql$/);
      if (!match) throw new Error(`Invalid migration file: ${file}`);

      const version = parseInt(match[1]);
      const name = match[2];
      const sql = fs.readFileSync(path.join(dir, file), 'utf-8');

      return { version, name, sql };
    });
  }

  async migrate(migrationsDir: string) {
    await this.initialize();

    const currentVersion = await this.getCurrentVersion();
    const migrations = await this.loadMigrations(migrationsDir);

    const pending = migrations.filter(m => m.version > currentVersion);

    if (pending.length === 0) {
      console.log('No pending migrations');
      return;
    }

    for (const migration of pending) {
      console.log(`Applying migration ${migration.version}: ${migration.name}`);

      const tx = await this.client.transaction('write');
      try {
        // Execute migration SQL
        await tx.execute(migration.sql);

        // Record migration
        await tx.execute({
          sql: 'INSERT INTO schema_migrations (version, name) VALUES (?, ?)',
          args: [migration.version, migration.name],
        });

        await tx.commit();
        console.log(`✓ Applied migration ${migration.version}`);
      } catch (error) {
        await tx.rollback();
        throw new Error(`Migration ${migration.version} failed: ${error}`);
      }
    }
  }
}

// Usage
const client = createClient({
  url: process.env.TURSO_DATABASE_URL!,
  authToken: process.env.TURSO_AUTH_TOKEN!,
});

const runner = new MigrationRunner(client);
await runner.migrate('./migrations');
```

### Schema Versioning

```sql
-- Track schema version
CREATE TABLE IF NOT EXISTS schema_info (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at INTEGER DEFAULT (unixepoch())
);

INSERT OR REPLACE INTO schema_info (key, value) VALUES ('version', '1.0.0');

-- Query schema version
SELECT value FROM schema_info WHERE key = 'version';
```

---

## 7. Query Patterns and Optimization

### Efficient Query Patterns

```typescript
// ✅ GOOD: Use parameterized queries
const users = await client.execute({
  sql: 'SELECT * FROM users WHERE email = ?',
  args: [email],
});

// ❌ BAD: String concatenation (SQL injection risk!)
// const users = await client.execute(`SELECT * FROM users WHERE email = '${email}'`);

// ✅ GOOD: Use indexes
// CREATE INDEX idx_users_email ON users(email);
const user = await client.execute({
  sql: 'SELECT * FROM users WHERE email = ? AND status = ?',
  args: ['alice@example.com', 'active'],
});

// ✅ GOOD: Limit results
const recentUsers = await client.execute({
  sql: 'SELECT * FROM users ORDER BY created_at DESC LIMIT 10',
});

// ✅ GOOD: Pagination with OFFSET
const page2 = await client.execute({
  sql: 'SELECT * FROM users ORDER BY id LIMIT ? OFFSET ?',
  args: [20, 20], // Page 2, 20 items per page
});

// ✅ BETTER: Keyset pagination (more efficient)
const nextPage = await client.execute({
  sql: 'SELECT * FROM users WHERE id > ? ORDER BY id LIMIT ?',
  args: [lastId, 20],
});
```

### Aggregations and Joins

```sql
-- Count with filters
SELECT COUNT(*) as total
FROM users
WHERE status = 'active'
  AND created_at > unixepoch() - 86400;

-- Group by with aggregates
SELECT
    status,
    COUNT(*) as count,
    AVG(created_at) as avg_created
FROM users
GROUP BY status;

-- Join tables
SELECT
    u.name,
    u.email,
    p.title,
    p.created_at
FROM users u
INNER JOIN posts p ON p.user_id = u.id
WHERE u.status = 'active'
ORDER BY p.created_at DESC
LIMIT 20;

-- Subqueries
SELECT *
FROM users
WHERE id IN (
    SELECT user_id
    FROM posts
    WHERE published_at IS NOT NULL
    GROUP BY user_id
    HAVING COUNT(*) > 5
);

-- Common Table Expressions (CTE)
WITH active_users AS (
    SELECT id, name, email
    FROM users
    WHERE status = 'active'
),
user_post_counts AS (
    SELECT user_id, COUNT(*) as post_count
    FROM posts
    GROUP BY user_id
)
SELECT
    au.name,
    au.email,
    COALESCE(upc.post_count, 0) as posts
FROM active_users au
LEFT JOIN user_post_counts upc ON upc.user_id = au.id
ORDER BY posts DESC;
```

### Full-Text Search

```sql
-- Create FTS5 virtual table
CREATE VIRTUAL TABLE posts_fts USING fts5(
    title,
    content,
    tokenize='porter'
);

-- Insert data
INSERT INTO posts_fts (rowid, title, content)
SELECT id, title, content FROM posts;

-- Search
SELECT
    rowid,
    title,
    snippet(posts_fts, 0, '<b>', '</b>', '...', 32) as snippet,
    rank
FROM posts_fts
WHERE posts_fts MATCH 'sqlite database'
ORDER BY rank
LIMIT 10;

-- Boolean search
SELECT * FROM posts_fts WHERE posts_fts MATCH 'sqlite AND (database OR tutorial)';

-- Phrase search
SELECT * FROM posts_fts WHERE posts_fts MATCH '"distributed database"';
```

### JSON Support

```sql
-- Store JSON data
CREATE TABLE documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    data TEXT CHECK(json_valid(data)),
    created_at INTEGER DEFAULT (unixepoch())
);

-- Insert JSON
INSERT INTO documents (data) VALUES (
    json_object('name', 'Alice', 'age', 30, 'tags', json_array('admin', 'user'))
);

-- Query JSON fields
SELECT
    id,
    json_extract(data, '$.name') as name,
    json_extract(data, '$.age') as age
FROM documents
WHERE json_extract(data, '$.age') > 25;

-- JSON array operations
SELECT id, value as tag
FROM documents, json_each(json_extract(data, '$.tags'))
WHERE value = 'admin';

-- Update JSON field
UPDATE documents
SET data = json_set(data, '$.age', 31)
WHERE id = 1;
```

### Query Performance Analysis

```typescript
// Use EXPLAIN QUERY PLAN to analyze queries
const plan = await client.execute({
  sql: 'EXPLAIN QUERY PLAN SELECT * FROM users WHERE email = ?',
  args: ['test@example.com'],
});

console.log(plan.rows);
// Look for:
// - "SEARCH" (uses index - good)
// - "SCAN" (table scan - bad for large tables)
```

---

## 8. Edge Deployment Patterns

### Cloudflare Workers

**Installation:**

```bash
npm create cloudflare@latest my-worker
cd my-worker
npm install @libsql/client
```

**Worker Code:**

```typescript
import { createClient } from '@libsql/client';

export interface Env {
  TURSO_DATABASE_URL: string;
  TURSO_AUTH_TOKEN: string;
}

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    // Create client (connection pooling handled automatically)
    const client = createClient({
      url: env.TURSO_DATABASE_URL,
      authToken: env.TURSO_AUTH_TOKEN,
    });

    try {
      const url = new URL(request.url);

      if (url.pathname === '/users' && request.method === 'GET') {
        const result = await client.execute('SELECT id, name, email FROM users LIMIT 20');
        return Response.json(result.rows);
      }

      if (url.pathname === '/users' && request.method === 'POST') {
        const body = await request.json() as { name: string; email: string };

        const result = await client.execute({
          sql: 'INSERT INTO users (name, email) VALUES (?, ?) RETURNING id',
          args: [body.name, body.email],
        });

        return Response.json(result.rows[0], { status: 201 });
      }

      return new Response('Not Found', { status: 404 });
    } catch (error: any) {
      return Response.json({ error: error.message }, { status: 500 });
    }
  },
};
```

**Deploy:**

```bash
# Add secrets
echo "TURSO_DATABASE_URL" | wrangler secret put TURSO_DATABASE_URL
echo "TURSO_AUTH_TOKEN" | wrangler secret put TURSO_AUTH_TOKEN

# Deploy
wrangler deploy
```

### Vercel Edge Functions

**api/users.ts:**

```typescript
import { createClient } from '@libsql/client';
import type { NextRequest } from 'next/server';

export const config = {
  runtime: 'edge',
};

const client = createClient({
  url: process.env.TURSO_DATABASE_URL!,
  authToken: process.env.TURSO_AUTH_TOKEN!,
});

export default async function handler(req: NextRequest) {
  try {
    if (req.method === 'GET') {
      const result = await client.execute('SELECT * FROM users LIMIT 20');
      return new Response(JSON.stringify(result.rows), {
        headers: { 'Content-Type': 'application/json' },
      });
    }

    if (req.method === 'POST') {
      const body = await req.json();
      const result = await client.execute({
        sql: 'INSERT INTO users (name, email) VALUES (?, ?) RETURNING id',
        args: [body.name, body.email],
      });

      return new Response(JSON.stringify(result.rows[0]), {
        status: 201,
        headers: { 'Content-Type': 'application/json' },
      });
    }

    return new Response('Method Not Allowed', { status: 405 });
  } catch (error: any) {
    return new Response(JSON.stringify({ error: error.message }), {
      status: 500,
      headers: { 'Content-Type': 'application/json' },
    });
  }
}
```

### Next.js App Router

**app/api/users/route.ts:**

```typescript
import { createClient } from '@libsql/client';
import { NextResponse } from 'next/server';

const client = createClient({
  url: process.env.TURSO_DATABASE_URL!,
  authToken: process.env.TURSO_AUTH_TOKEN!,
});

export async function GET() {
  try {
    const result = await client.execute('SELECT * FROM users ORDER BY created_at DESC LIMIT 20');
    return NextResponse.json(result.rows);
  } catch (error: any) {
    return NextResponse.json({ error: error.message }, { status: 500 });
  }
}

export async function POST(request: Request) {
  try {
    const body = await request.json();

    const result = await client.execute({
      sql: 'INSERT INTO users (name, email) VALUES (?, ?) RETURNING *',
      args: [body.name, body.email],
    });

    return NextResponse.json(result.rows[0], { status: 201 });
  } catch (error: any) {
    return NextResponse.json({ error: error.message }, { status: 500 });
  }
}
```

**Server Component:**

```typescript
// app/users/page.tsx
import { createClient } from '@libsql/client';

const client = createClient({
  url: process.env.TURSO_DATABASE_URL!,
  authToken: process.env.TURSO_AUTH_TOKEN!,
});

export default async function UsersPage() {
  const result = await client.execute('SELECT * FROM users ORDER BY created_at DESC');
  const users = result.rows;

  return (
    <div>
      <h1>Users</h1>
      <ul>
        {users.map((user: any) => (
          <li key={user.id}>
            {user.name} ({user.email})
          </li>
        ))}
      </ul>
    </div>
  );
}
```

### Deno Deploy

```typescript
import { createClient } from 'npm:@libsql/client';
import { serve } from 'https://deno.land/std@0.208.0/http/server.ts';

const client = createClient({
  url: Deno.env.get('TURSO_DATABASE_URL')!,
  authToken: Deno.env.get('TURSO_AUTH_TOKEN')!,
});

serve(async (req: Request) => {
  const url = new URL(req.url);

  if (url.pathname === '/users' && req.method === 'GET') {
    const result = await client.execute('SELECT * FROM users LIMIT 20');
    return new Response(JSON.stringify(result.rows), {
      headers: { 'Content-Type': 'application/json' },
    });
  }

  return new Response('Not Found', { status: 404 });
});
```

---

## 9. Local-First Applications

### Embedded Replica Pattern

**React Application:**

```typescript
// db/client.ts
import { createClient, type Client } from '@libsql/client';

let client: Client | null = null;

export function getClient(): Client {
  if (client) return client;

  client = createClient({
    url: 'file:./local.db',
    syncUrl: import.meta.env.VITE_TURSO_DATABASE_URL,
    authToken: import.meta.env.VITE_TURSO_AUTH_TOKEN,
    syncInterval: 5000, // Sync every 5 seconds
  });

  return client;
}

// Sync on demand
export async function syncDatabase(): Promise<void> {
  const client = getClient();
  await client.sync();
}
```

**React Hook:**

```typescript
// hooks/useDatabase.ts
import { useEffect, useState } from 'react';
import { getClient, syncDatabase } from '../db/client';

export function useQuery<T>(sql: string, args: any[] = []) {
  const [data, setData] = useState<T[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        const client = getClient();
        const result = await client.execute({ sql, args });
        setData(result.rows as T[]);
      } catch (err) {
        setError(err as Error);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, [sql, JSON.stringify(args)]);

  return { data, loading, error };
}

export function useMutation() {
  const execute = async (sql: string, args: any[] = []) => {
    const client = getClient();
    const result = await client.execute({ sql, args });
    await syncDatabase(); // Sync after mutation
    return result;
  };

  return { execute };
}
```

**Component Usage:**

```typescript
// components/UserList.tsx
import { useQuery, useMutation } from '../hooks/useDatabase';

interface User {
  id: number;
  name: string;
  email: string;
}

export function UserList() {
  const { data: users, loading, error } = useQuery<User>(
    'SELECT * FROM users ORDER BY created_at DESC'
  );
  const { execute } = useMutation();

  const handleAddUser = async (name: string, email: string) => {
    await execute(
      'INSERT INTO users (name, email) VALUES (?, ?)',
      [name, email]
    );
    // Data will be refetched automatically
  };

  if (loading) return <div>Loading...</div>;
  if (error) return <div>Error: {error.message}</div>;

  return (
    <div>
      <h1>Users</h1>
      <ul>
        {users.map(user => (
          <li key={user.id}>{user.name} - {user.email}</li>
        ))}
      </ul>
      <button onClick={() => handleAddUser('New User', 'new@example.com')}>
        Add User
      </button>
    </div>
  );
}
```

### Electron Application

```typescript
// main.ts (Main Process)
import { app } from 'electron';
import { createClient } from '@libsql/client';
import path from 'path';

const client = createClient({
  url: `file:${path.join(app.getPath('userData'), 'app.db')}`,
  syncUrl: process.env.TURSO_DATABASE_URL,
  authToken: process.env.TURSO_AUTH_TOKEN,
  syncInterval: 10000,
});

// Expose to renderer via IPC
ipcMain.handle('db:query', async (event, sql: string, args: any[]) => {
  const result = await client.execute({ sql, args });
  return result.rows;
});

ipcMain.handle('db:sync', async () => {
  await client.sync();
});
```

### Mobile Apps (React Native)

```typescript
import { createClient } from '@libsql/client';
import * as FileSystem from 'expo-file-system';

const dbPath = `${FileSystem.documentDirectory}SQLite/app.db`;

const client = createClient({
  url: `file:${dbPath}`,
  syncUrl: process.env.EXPO_PUBLIC_TURSO_DATABASE_URL,
  authToken: process.env.EXPO_PUBLIC_TURSO_AUTH_TOKEN,
  syncInterval: 15000, // Sync every 15 seconds when online
});

// Sync when app comes to foreground
import { AppState } from 'react-native';

AppState.addEventListener('change', (nextAppState) => {
  if (nextAppState === 'active') {
    client.sync().catch(console.error);
  }
});
```

---

## 10. Security Best Practices

### Authentication Tokens

```bash
# Create read-only token
turso db tokens create my-app-db --expiration 7d --read-only

# Create full-access token (default)
turso db tokens create my-app-db --expiration 30d

# Create token with no expiration (not recommended for production)
turso db tokens create my-app-db --expiration none

# Revoke all tokens (forces rotation)
turso db tokens invalidate my-app-db

# List tokens (shows active tokens)
turso db tokens list my-app-db
```

### Token Rotation

```typescript
// Environment-based token management
const config = {
  url: process.env.TURSO_DATABASE_URL!,
  authToken: process.env.TURSO_AUTH_TOKEN!,
};

// Rotate tokens periodically
// 1. Create new token via CLI
// 2. Update environment variables
// 3. Invalidate old tokens after grace period
```

### SQL Injection Prevention

```typescript
// ✅ CORRECT: Parameterized queries
const email = userInput;
const result = await client.execute({
  sql: 'SELECT * FROM users WHERE email = ?',
  args: [email],
});

// ✅ CORRECT: Named parameters
const result = await client.execute({
  sql: 'SELECT * FROM users WHERE email = :email AND status = :status',
  args: { email: userInput, status: 'active' },
});

// ❌ WRONG: String interpolation (SQL INJECTION RISK!)
// const sql = `SELECT * FROM users WHERE email = '${userInput}'`;
// await client.execute(sql);
```

### Row-Level Security (Application-Level)

```typescript
// Middleware for multi-tenant security
interface Context {
  userId: number;
  tenantId: string;
}

class SecureRepository {
  constructor(
    private client: Client,
    private context: Context
  ) {}

  async getUsers() {
    // Automatically filter by tenant
    return await this.client.execute({
      sql: 'SELECT * FROM users WHERE tenant_id = ?',
      args: [this.context.tenantId],
    });
  }

  async getUser(userId: number) {
    // Ensure user belongs to tenant
    const result = await this.client.execute({
      sql: 'SELECT * FROM users WHERE id = ? AND tenant_id = ?',
      args: [userId, this.context.tenantId],
    });

    if (result.rows.length === 0) {
      throw new Error('User not found or access denied');
    }

    return result.rows[0];
  }

  async createUser(data: { name: string; email: string }) {
    // Automatically set tenant_id
    return await this.client.execute({
      sql: 'INSERT INTO users (tenant_id, name, email) VALUES (?, ?, ?) RETURNING *',
      args: [this.context.tenantId, data.name, data.email],
    });
  }
}
```

### Rate Limiting

```typescript
import { Ratelimit } from '@upstash/ratelimit';
import { Redis } from '@upstash/redis';

const ratelimit = new Ratelimit({
  redis: Redis.fromEnv(),
  limiter: Ratelimit.slidingWindow(10, '10 s'),
});

export async function handleRequest(request: Request, env: Env) {
  const ip = request.headers.get('CF-Connecting-IP') || 'unknown';
  const { success } = await ratelimit.limit(ip);

  if (!success) {
    return new Response('Rate limit exceeded', { status: 429 });
  }

  // Process request
  const client = createClient({
    url: env.TURSO_DATABASE_URL,
    authToken: env.TURSO_AUTH_TOKEN,
  });

  // ... handle database operations
}
```

### Environment Variables Security

```bash
# .env.local (never commit!)
TURSO_DATABASE_URL=libsql://my-app-db-user.turso.io
TURSO_AUTH_TOKEN=eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9..

# Use separate tokens for different environments
TURSO_DATABASE_URL_DEV=libsql://dev-db.turso.io
TURSO_AUTH_TOKEN_DEV=..

TURSO_DATABASE_URL_PROD=libsql://prod-db.turso.io
TURSO_AUTH_TOKEN_PROD=..
```

**.env.example (can be committed):**

```bash
# Turso Configuration
TURSO_DATABASE_URL=your_database_url_here
TURSO_AUTH_TOKEN=your_auth_token_here
```

---

## 11. Performance Optimization

### Connection Pooling

```typescript
// Singleton pattern for connection reuse
class DatabaseManager {
  private static instance: Client;

  static getClient(): Client {
    if (!DatabaseManager.instance) {
      DatabaseManager.instance = createClient({
        url: process.env.TURSO_DATABASE_URL!,
        authToken: process.env.TURSO_AUTH_TOKEN!,
      });
    }
    return DatabaseManager.instance;
  }
}

// Reuse connection across requests
const client = DatabaseManager.getClient();
```

### Batch Operations

```typescript
// ✅ EFFICIENT: Batch insert
const users = [
  ['alice@example.com', 'Alice'],
  ['bob@example.com', 'Bob'],
  ['charlie@example.com', 'Charlie'],
];

await client.batch(
  users.map(([email, name]) => ({
    sql: 'INSERT INTO users (email, name) VALUES (?, ?)',
    args: [email, name],
  }))
);

// ❌ INEFFICIENT: Individual inserts
// for (const [email, name] of users) {
//   await client.execute({
//     sql: 'INSERT INTO users (email, name) VALUES (?, ?)',
//     args: [email, name],
//   });
// }
```

### Query Optimization

```sql
-- Analyze query performance
EXPLAIN QUERY PLAN
SELECT u.*, COUNT(p.id) as post_count
FROM users u
LEFT JOIN posts p ON p.user_id = u.id
WHERE u.status = 'active'
GROUP BY u.id;

-- Create appropriate indexes
CREATE INDEX idx_users_status ON users(status);
CREATE INDEX idx_posts_user_id ON posts(user_id);

-- Use covering indexes
CREATE INDEX idx_users_status_name_email
ON users(status, name, email);

-- Query now only uses index (no table access needed)
SELECT name, email FROM users WHERE status = 'active';
```

### Caching Strategies

```typescript
// In-memory cache with TTL
class CachedRepository {
  private cache = new Map<string, { data: any; expires: number }>();

  constructor(private client: Client) {}

  async getUser(id: number, ttl: number = 60000): Promise<any> {
    const cacheKey = `user:${id}`;
    const cached = this.cache.get(cacheKey);

    if (cached && cached.expires > Date.now()) {
      return cached.data;
    }

    const result = await this.client.execute({
      sql: 'SELECT * FROM users WHERE id = ?',
      args: [id],
    });

    if (result.rows.length > 0) {
      this.cache.set(cacheKey, {
        data: result.rows[0],
        expires: Date.now() + ttl,
      });
      return result.rows[0];
    }

    return null;
  }

  invalidateCache(id: number) {
    this.cache.delete(`user:${id}`);
  }
}
```

### Embedded Replica Performance

```typescript
// Optimize sync frequency based on use case
const client = createClient({
  url: 'file:./local.db',
  syncUrl: remoteUrl,
  authToken: token,

  // High-frequency updates (real-time apps)
  syncInterval: 1000, // 1 second

  // Medium-frequency (most apps)
  // syncInterval: 5000, // 5 seconds

  // Low-frequency (occasional sync)
  // syncInterval: 60000, // 1 minute
});

// Manual sync for critical operations
async function updateUserProfile(data: any) {
  await client.execute({
    sql: 'UPDATE users SET name = ?, email = ? WHERE id = ?',
    args: [data.name, data.email, data.id],
  });

  // Ensure immediate sync for critical data
  await client.sync();
}
```

### Write Optimization

```sql
-- Use transactions for multiple writes
BEGIN TRANSACTION;
INSERT INTO users (name, email) VALUES ('Alice', 'alice@example.com');
INSERT INTO posts (user_id, title) VALUES (last_insert_rowid(), 'My Post');
COMMIT;

-- Disable synchronous writes for bulk operations (use with caution!)
PRAGMA synchronous = OFF; -- Faster but less durable
-- ... bulk operations ..
PRAGMA synchronous = FULL; -- Restore safety
```

---

## 12. Monitoring and Observability

### Database Metrics

```bash
# View database statistics
turso db show my-app-db

# Monitor usage
turso db usage my-app-db

# Check replica status
turso db replicas my-app-db
```

### Application-Level Monitoring

```typescript
// Query timing middleware
class MonitoredClient {
  constructor(private client: Client) {}

  async execute(stmt: { sql: string; args?: any[] }) {
    const start = Date.now();

    try {
      const result = await this.client.execute(stmt);
      const duration = Date.now() - start;

      // Log slow queries
      if (duration > 1000) {
        console.warn(`Slow query (${duration}ms):`, stmt.sql);
      }

      // Send metrics to monitoring service
      this.recordMetric('query.duration', duration, { sql: stmt.sql });
      this.recordMetric('query.success', 1);

      return result;
    } catch (error) {
      const duration = Date.now() - start;

      // Log error
      console.error(`Query failed (${duration}ms):`, stmt.sql, error);

      // Send error metrics
      this.recordMetric('query.error', 1, { sql: stmt.sql });

      throw error;
    }
  }

  private recordMetric(name: string, value: number, tags?: Record<string, string>) {
    // Send to your monitoring service (DataDog, New Relic, etc.)
  }
}
```

### Health Checks

```typescript
// Endpoint for health monitoring
export async function healthCheck(env: Env): Promise<Response> {
  try {
    const client = createClient({
      url: env.TURSO_DATABASE_URL,
      authToken: env.TURSO_AUTH_TOKEN,
    });

    const start = Date.now();
    await client.execute('SELECT 1');
    const latency = Date.now() - start;

    return Response.json({
      status: 'healthy',
      database: 'connected',
      latency_ms: latency,
      timestamp: new Date().toISOString(),
    });
  } catch (error: any) {
    return Response.json({
      status: 'unhealthy',
      database: 'disconnected',
      error: error.message,
      timestamp: new Date().toISOString(),
    }, { status: 503 });
  }
}
```

### Error Tracking

```typescript
import * as Sentry from '@sentry/node';

async function executeQuery(client: Client, sql: string, args: any[]) {
  try {
    return await client.execute({ sql, args });
  } catch (error) {
    // Capture error context
    Sentry.captureException(error, {
      tags: {
        service: 'database',
        query_type: sql.split(' ')[0],
      },
      extra: {
        sql,
        args,
      },
    });

    throw error;
  }
}
```

---

## 13. Backup and Recovery

### Export Database

```bash
# Dump database to SQL file
turso db shell my-app-db ".dump" > backup.sql

# Dump specific table
turso db shell my-app-db ".dump users" > users_backup.sql

# Export to CSV
turso db shell my-app-db <<EOF
.mode csv
.output users.csv
SELECT * FROM users;
EOF
```

### Import Database

```bash
# Import from SQL dump
turso db shell my-app-db < backup.sql

# Import CSV
turso db shell my-app-db <<EOF
.mode csv
.import users.csv users
EOF
```

### Point-in-Time Recovery

```bash
# Create database from timestamp
turso db create restored-db --from my-app-db --timestamp 2026-02-05T10:00:00Z

# List available recovery points
turso db show my-app-db --recovery-points
```

### Automated Backups

```bash
#!/bin/bash
# backup.sh

DB_NAME="my-app-db"
BACKUP_DIR="./backups"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
BACKUP_FILE="$BACKUP_DIR/$DB_NAME-$TIMESTAMP.sql"

mkdir -p $BACKUP_DIR

# Create backup
turso db shell $DB_NAME ".dump" > $BACKUP_FILE

# Compress
gzip $BACKUP_FILE

# Upload to S3
aws s3 cp "${BACKUP_FILE}.gz" "s3://my-backups/$DB_NAME/"

# Cleanup old backups (keep last 30 days)
find $BACKUP_DIR -name "*.sql.gz" -mtime +30 -delete

echo "Backup completed: ${BACKUP_FILE}.gz"
```

**Cron Schedule:**

```cron
# Run daily at 2 AM
0 2 * * * /path/to/backup.sh
```

---

## 14. Testing Strategies

### Unit Tests with In-Memory Database

```typescript
// test/db.test.ts
import { createClient } from '@libsql/client';
import { describe, test, expect, beforeEach } from 'vitest';

describe('User Repository', () => {
  let client: Client;

  beforeEach(async () => {
    // Create in-memory database for each test
    client = createClient({ url: ':memory:' });

    // Setup schema
    await client.execute(`
      CREATE TABLE users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        email TEXT NOT NULL UNIQUE,
        name TEXT NOT NULL,
        created_at INTEGER DEFAULT (unixepoch())
      )
    `);
  });

  test('should create user', async () => {
    const result = await client.execute({
      sql: 'INSERT INTO users (email, name) VALUES (?, ?) RETURNING *',
      args: ['test@example.com', 'Test User'],
    });

    expect(result.rows).toHaveLength(1);
    expect(result.rows[0].email).toBe('test@example.com');
    expect(result.rows[0].name).toBe('Test User');
  });

  test('should enforce unique email', async () => {
    await client.execute({
      sql: 'INSERT INTO users (email, name) VALUES (?, ?)',
      args: ['duplicate@example.com', 'User 1'],
    });

    await expect(
      client.execute({
        sql: 'INSERT INTO users (email, name) VALUES (?, ?)',
        args: ['duplicate@example.com', 'User 2'],
      })
    ).rejects.toThrow();
  });

  test('should list users with pagination', async () => {
    // Insert test data
    for (let i = 0; i < 25; i++) {
      await client.execute({
        sql: 'INSERT INTO users (email, name) VALUES (?, ?)',
        args: [`user${i}@example.com`, `User ${i}`],
      });
    }

    // Test pagination
    const page1 = await client.execute({
      sql: 'SELECT * FROM users ORDER BY id LIMIT ? OFFSET ?',
      args: [10, 0],
    });
    expect(page1.rows).toHaveLength(10);

    const page2 = await client.execute({
      sql: 'SELECT * FROM users ORDER BY id LIMIT ? OFFSET ?',
      args: [10, 10],
    });
    expect(page2.rows).toHaveLength(10);

    const page3 = await client.execute({
      sql: 'SELECT * FROM users ORDER BY id LIMIT ? OFFSET ?',
      args: [10, 20],
    });
    expect(page3.rows).toHaveLength(5);
  });
});
```

### Integration Tests

```typescript
// test/integration.test.ts
import { createClient } from '@libsql/client';
import { describe, test, expect, beforeAll, afterAll } from 'vitest';

describe('Integration Tests', () => {
  let client: Client;
  let testDbName: string;

  beforeAll(async () => {
    // Create temporary test database
    testDbName = `test-${Date.now()}`;
    await exec(`turso db create ${testDbName}`);

    // Get connection details
    const url = await exec(`turso db show ${testDbName} --url`);
    const token = await exec(`turso db tokens create ${testDbName}`);

    client = createClient({
      url: url.trim(),
      authToken: token.trim(),
    });

    // Setup schema
    await client.execute(`
      CREATE TABLE users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        email TEXT NOT NULL UNIQUE,
        name TEXT NOT NULL
      )
    `);
  });

  afterAll(async () => {
    // Cleanup
    await exec(`turso db destroy ${testDbName} --yes`);
  });

  test('should work with remote database', async () => {
    const result = await client.execute({
      sql: 'INSERT INTO users (email, name) VALUES (?, ?) RETURNING *',
      args: ['remote@example.com', 'Remote User'],
    });

    expect(result.rows[0].email).toBe('remote@example.com');
  });
});
```

### End-to-End Tests

```typescript
// e2e/api.test.ts
import { test, expect } from '@playwright/test';

test.describe('API Endpoints', () => {
  test('should create and retrieve user', async ({ request }) => {
    // Create user
    const createResponse = await request.post('/api/users', {
      data: {
        email: 'e2e@example.com',
        name: 'E2E User',
      },
    });
    expect(createResponse.ok()).toBeTruthy();

    const created = await createResponse.json();
    expect(created.id).toBeDefined();

    // Retrieve user
    const getResponse = await request.get(`/api/users/${created.id}`);
    expect(getResponse.ok()).toBeTruthy();

    const retrieved = await getResponse.json();
    expect(retrieved.email).toBe('e2e@example.com');
  });
});
```

---

## 15. Migration from SQLite

### Export SQLite Database

```bash
# Export existing SQLite database
sqlite3 existing.db .dump > dump.sql

# Or export specific tables
sqlite3 existing.db <<EOF
.output users.sql
.dump users
.output posts.sql
.dump posts
EOF
```

### Import to Turso

```bash
# Create new Turso database
turso db create migrated-db

# Import dump
turso db shell migrated-db < dump.sql

# Verify import
turso db shell migrated-db "SELECT COUNT(*) FROM users;"
```

### Application Migration

**Before (SQLite):**

```typescript
import Database from 'better-sqlite3';

const db = new Database('./local.db');

const users = db.prepare('SELECT * FROM users').all();
```

**After (Turso/libSQL):**

```typescript
import { createClient } from '@libsql/client';

const client = createClient({
  url: process.env.TURSO_DATABASE_URL!,
  authToken: process.env.TURSO_AUTH_TOKEN!,
});

const result = await client.execute('SELECT * FROM users');
const users = result.rows;
```

### Migration Checklist

- [ ] Export schema from SQLite
- [ ] Create Turso database
- [ ] Import schema to Turso
- [ ] Test schema compatibility
- [ ] Export data from SQLite
- [ ] Import data to Turso
- [ ] Verify data integrity
- [ ] Update application connection code
- [ ] Update queries from synchronous to async
- [ ] Test all database operations
- [ ] Deploy with new connection configuration
- [ ] Monitor for errors
- [ ] Decommission old SQLite database

---

## 16. Common Patterns and Recipes

### Repository Pattern

```typescript
// repositories/UserRepository.ts
import { Client } from '@libsql/client';

export interface User {
  id: number;
  email: string;
  name: string;
  created_at: number;
}

export class UserRepository {
  constructor(private client: Client) {}

  async findAll(limit: number = 20, offset: number = 0): Promise<User[]> {
    const result = await this.client.execute({
      sql: 'SELECT * FROM users ORDER BY created_at DESC LIMIT ? OFFSET ?',
      args: [limit, offset],
    });
    return result.rows as User[];
  }

  async findById(id: number): Promise<User | null> {
    const result = await this.client.execute({
      sql: 'SELECT * FROM users WHERE id = ?',
      args: [id],
    });
    return result.rows[0] as User || null;
  }

  async findByEmail(email: string): Promise<User | null> {
    const result = await this.client.execute({
      sql: 'SELECT * FROM users WHERE email = ?',
      args: [email],
    });
    return result.rows[0] as User || null;
  }

  async create(data: { email: string; name: string }): Promise<User> {
    const result = await this.client.execute({
      sql: 'INSERT INTO users (email, name) VALUES (?, ?) RETURNING *',
      args: [data.email, data.name],
    });
    return result.rows[0] as User;
  }

  async update(id: number, data: Partial<{ email: string; name: string }>): Promise<User> {
    const updates: string[] = [];
    const args: any[] = [];

    if (data.email !== undefined) {
      updates.push('email = ?');
      args.push(data.email);
    }
    if (data.name !== undefined) {
      updates.push('name = ?');
      args.push(data.name);
    }

    args.push(id);

    const result = await this.client.execute({
      sql: `UPDATE users SET ${updates.join(', ')} WHERE id = ? RETURNING *`,
      args,
    });

    if (result.rows.length === 0) {
      throw new Error('User not found');
    }

    return result.rows[0] as User;
  }

  async delete(id: number): Promise<void> {
    await this.client.execute({
      sql: 'DELETE FROM users WHERE id = ?',
      args: [id],
    });
  }

  async count(): Promise<number> {
    const result = await this.client.execute('SELECT COUNT(*) as count FROM users');
    return result.rows[0].count as number;
  }
}
```

### Pagination Helper

```typescript
interface PaginationOptions {
  page: number;
  perPage: number;
}

interface PaginatedResult<T> {
  data: T[];
  total: number;
  page: number;
  perPage: number;
  totalPages: number;
}

async function paginate<T>(
  client: Client,
  tableName: string,
  options: PaginationOptions,
  where?: { sql: string; args: any[] }
): Promise<PaginatedResult<T>> {
  const { page, perPage } = options;
  const offset = (page - 1) * perPage;

  // Build WHERE clause
  const whereClause = where ? `WHERE ${where.sql}` : '';
  const whereArgs = where?.args || [];

  // Get total count
  const countResult = await client.execute({
    sql: `SELECT COUNT(*) as count FROM ${tableName} ${whereClause}`,
    args: whereArgs,
  });
  const total = countResult.rows[0].count as number;

  // Get data
  const dataResult = await client.execute({
    sql: `SELECT * FROM ${tableName} ${whereClause} LIMIT ? OFFSET ?`,
    args: [...whereArgs, perPage, offset],
  });

  return {
    data: dataResult.rows as T[],
    total,
    page,
    perPage,
    totalPages: Math.ceil(total / perPage),
  };
}

// Usage
const result = await paginate<User>(
  client,
  'users',
  { page: 2, perPage: 20 },
  { sql: 'status = ?', args: ['active'] }
);
```

### Soft Delete Pattern

```sql
-- Add deleted_at column
ALTER TABLE users ADD COLUMN deleted_at INTEGER;

-- Create index for active users
CREATE INDEX idx_users_active ON users(id) WHERE deleted_at IS NULL;
```

```typescript
class SoftDeleteRepository {
  async softDelete(id: number): Promise<void> {
    await this.client.execute({
      sql: 'UPDATE users SET deleted_at = unixepoch() WHERE id = ?',
      args: [id],
    });
  }

  async restore(id: number): Promise<void> {
    await this.client.execute({
      sql: 'UPDATE users SET deleted_at = NULL WHERE id = ?',
      args: [id],
    });
  }

  async findAllActive(): Promise<User[]> {
    const result = await this.client.execute(
      'SELECT * FROM users WHERE deleted_at IS NULL ORDER BY created_at DESC'
    );
    return result.rows as User[];
  }

  async findAllWithDeleted(): Promise<User[]> {
    const result = await this.client.execute(
      'SELECT * FROM users ORDER BY created_at DESC'
    );
    return result.rows as User[];
  }
}
```

### Audit Log Pattern

```sql
CREATE TABLE audit_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    table_name TEXT NOT NULL,
    record_id INTEGER NOT NULL,
    action TEXT NOT NULL CHECK(action IN ('INSERT', 'UPDATE', 'DELETE')),
    old_values TEXT,
    new_values TEXT,
    user_id INTEGER,
    created_at INTEGER DEFAULT (unixepoch())
);

CREATE INDEX idx_audit_logs_table_record ON audit_logs(table_name, record_id);
CREATE INDEX idx_audit_logs_user ON audit_logs(user_id);
```

```typescript
async function auditedUpdate(
  client: Client,
  table: string,
  id: number,
  newData: any,
  userId: number
): Promise<void> {
  const tx = await client.transaction('write');

  try {
    // Get old values
    const oldResult = await tx.execute({
      sql: `SELECT * FROM ${table} WHERE id = ?`,
      args: [id],
    });
    const oldValues = oldResult.rows[0];

    // Perform update
    const updates = Object.keys(newData).map(k => `${k} = ?`).join(', ');
    const values = Object.values(newData);

    await tx.execute({
      sql: `UPDATE ${table} SET ${updates} WHERE id = ?`,
      args: [...values, id],
    });

    // Log audit
    await tx.execute({
      sql: `
        INSERT INTO audit_logs (table_name, record_id, action, old_values, new_values, user_id)
        VALUES (?, ?, 'UPDATE', ?, ?, ?)
      `,
      args: [
        table,
        id,
        JSON.stringify(oldValues),
        JSON.stringify(newData),
        userId,
      ],
    });

    await tx.commit();
  } catch (error) {
    await tx.rollback();
    throw error;
  }
}
```

---

## 17. Pricing and Scaling

### Turso Pricing Tiers (as of 2026)

**Starter (Free):**
- 9 GB total storage
- 1 billion row reads/month
- Unlimited databases
- Unlimited locations
- Community support

**Scaler ($29/month):**
- 50 GB storage
- 1 trillion row reads/month
- Everything in Starter
- Email support
- Point-in-time recovery

**Enterprise (Custom):**
- Custom storage
- Unlimited reads
- SLA
- Dedicated support
- Custom contracts

### Scaling Strategies

**Vertical Scaling:**
- Increase storage limits
- Optimize queries and indexes
- Use embedded replicas to reduce remote reads

**Horizontal Scaling:**
- Database per tenant (multi-tenancy)
- Functional sharding (separate databases per domain)
- Geographic distribution with replicas

**Read Scaling:**
- Add replicas in more regions
- Use embedded replicas for zero-latency reads
- Cache frequently accessed data

---

## 18. Troubleshooting

### Common Issues

**Connection Errors:**

```typescript
try {
  const result = await client.execute('SELECT 1');
} catch (error: any) {
  if (error.message.includes('ECONNREFUSED')) {
    console.error('Cannot connect to database. Check URL and network.');
  } else if (error.message.includes('Unauthorized')) {
    console.error('Invalid auth token. Check TURSO_AUTH_TOKEN.');
  } else {
    console.error('Database error:', error);
  }
}
```

**Sync Issues (Embedded Replicas):**

```typescript
try {
  await client.sync();
} catch (error: any) {
  if (error.message.includes('network')) {
    console.warn('Sync failed due to network. Will retry automatically.');
    // App continues with local data
  } else {
    console.error('Sync error:', error);
  }
}
```

**Query Debugging:**

```typescript
// Enable detailed logging
const client = createClient({
  url: process.env.TURSO_DATABASE_URL!,
  authToken: process.env.TURSO_AUTH_TOKEN!,
});

// Log all queries
const originalExecute = client.execute.bind(client);
client.execute = async (stmt: any) => {
  console.log('[SQL]', stmt.sql, stmt.args);
  const start = Date.now();
  try {
    const result = await originalExecute(stmt);
    console.log('[SQL] Done in', Date.now() - start, 'ms');
    return result;
  } catch (error) {
    console.error('[SQL] Error:', error);
    throw error;
  }
};
```

### Performance Issues

```sql
-- Identify missing indexes
EXPLAIN QUERY PLAN SELECT * FROM users WHERE email = ?;
-- Look for SCAN (bad) vs SEARCH (good)

-- Check table statistics
SELECT
    name,
    (SELECT COUNT(*) FROM pragma_table_info(name)) as columns,
    (SELECT COUNT(*) FROM sqlite_master WHERE type='index' AND tbl_name=name) as indexes
FROM sqlite_master
WHERE type='table';

-- Analyze database
ANALYZE;
```

---

## 19. Best Practices Summary

### Development

- ✅ Use parameterized queries always
- ✅ Create indexes for frequently queried columns
- ✅ Use transactions for multiple related operations
- ✅ Implement proper error handling
- ✅ Use embedded replicas for local-first apps
- ✅ Test with in-memory databases
- ✅ Version control your schema migrations

### Production

- ✅ Use separate databases per environment (dev, staging, prod)
- ✅ Rotate auth tokens regularly
- ✅ Monitor query performance
- ✅ Set up automated backups
- ✅ Use read replicas in multiple regions
- ✅ Implement connection pooling/reuse
- ✅ Add health check endpoints
- ✅ Log slow queries
- ✅ Use environment variables for credentials
- ✅ Test disaster recovery procedures

### Security

- ✅ Never commit auth tokens
- ✅ Use read-only tokens where appropriate
- ✅ Validate user input at application level
- ✅ Implement rate limiting
- ✅ Use row-level security in application code
- ✅ Audit sensitive operations
- ✅ Enable FOREIGN_KEYS pragma
- ✅ Sanitize error messages to users

---

## 20. Resources and References

### Official Documentation

- **libSQL GitHub**: https://github.com/tursodatabase/libsql
- **Turso Docs**: https://docs.turso.tech/
- **Turso CLI**: https://docs.turso.tech/reference/turso-cli
- **Client SDKs**: https://docs.turso.tech/sdk

### Community

- **Discord**: https://discord.gg/turso
- **Twitter**: @tursodatabase
- **Blog**: https://blog.turso.tech/

### Guides

- SQLite Documentation: https://www.sqlite.org/docs.html
- SQL Best Practices: See [sql.md](sql.md)
- Testing Guidelines: See [testing.md](testing.md)

---

## 21. Quick Reference

### CLI Commands

```bash
# Database management
turso db create <name>              # Create database
turso db list                       # List databases
turso db show <name>                # Show details
turso db destroy <name>             # Delete database
turso db shell <name>               # Interactive shell

# Replicas
turso db replicate <name> --region <region>
turso db replicas <name>

# Tokens
turso db tokens create <name>
turso db tokens list <name>
turso db tokens invalidate <name>

# Groups
turso group create <name> --location <region>
turso group list
turso group locations add <name> <region>
```

### Connection Patterns

```typescript
// Remote only
const client = createClient({
  url: process.env.TURSO_DATABASE_URL!,
  authToken: process.env.TURSO_AUTH_TOKEN!,
});

// Embedded replica
const client = createClient({
  url: 'file:./local.db',
  syncUrl: process.env.TURSO_DATABASE_URL!,
  authToken: process.env.TURSO_AUTH_TOKEN!,
  syncInterval: 5000,
});

// Local development
const client = createClient({
  url: 'file:./dev.db',
});
```

### Common SQL

```sql
-- CRUD operations
SELECT * FROM users WHERE email = ?;
INSERT INTO users (email, name) VALUES (?, ?) RETURNING *;
UPDATE users SET name = ? WHERE id = ?;
DELETE FROM users WHERE id = ?;

-- Indexes
CREATE INDEX idx_users_email ON users(email);
CREATE UNIQUE INDEX idx_users_email ON users(email);

-- Transactions
BEGIN TRANSACTION;
-- ... queries ..
COMMIT;
-- or ROLLBACK;
```

---

## 22. Deployment Checklist

### Build and Configuration
- [ ] Turso CLI version and libSQL client SDK version pinned
- [ ] Database group created in appropriate primary region
- [ ] Replica locations added for target user regions
- [ ] Auth tokens generated with appropriate scopes (read-only vs read-write)
- [ ] Embedded replica sync interval configured for latency requirements
- [ ] Schema migrations tracked and applied via Turso CLI or migration tool

### Testing
- [ ] All queries profiled with `EXPLAIN QUERY PLAN`
- [ ] Embedded replica sync behavior tested (conflict resolution, latency)
- [ ] Connection handling tested for token expiration and renewal
- [ ] Batch transaction performance validated
- [ ] Backup and restore tested with `turso db shell` export/import
- [ ] Edge function cold start tested with embedded replica initialization

### Security
- [ ] Auth tokens stored in environment variables or secrets manager
- [ ] No tokens committed to source control
- [ ] Token rotation procedure documented and tested
- [ ] Database-level access scoped per service/application
- [ ] TLS enforced for all remote connections (default with Turso)
- [ ] Group-level access control configured

### Agent Workflow
- [ ] Schema migration scripts version-controlled
- [ ] CI pipeline validates migrations against test database
- [ ] Monitoring configured for sync lag, query latency, and storage usage
- [ ] Alerting on auth token expiration
- [ ] Runbooks for region failover and replica re-initialization

---

## 23. Why This Configuration Works

**Edge-Local Reads with Embedded Replicas**:
- Embedded replicas store a local SQLite copy that syncs with the remote primary, delivering sub-millisecond read latency at the edge while maintaining global consistency through periodic synchronization.

**SQLite Compatibility**:
- Full SQLite wire and SQL compatibility means existing SQLite knowledge, tooling, and libraries work without modification, reducing migration effort and enabling local-first development.

**Global Distribution with Turso Platform**:
- Database groups with multi-region replicas place data close to users worldwide, providing low-latency access without application-level sharding or routing logic.

**Serverless Cost Model**:
- Pay-per-query pricing with automatic scaling eliminates capacity planning, making it cost-effective for applications with variable or unpredictable traffic patterns.

**Seamless Local Development**:
- The same client SDK connects to a local SQLite file for development, an embedded replica for staging, or a remote Turso database for production, with only a URL change.

---

**Document Version**: 1.0
**Last Updated**: February 2026
**Compatible with**: libSQL 0.3+, Turso Platform 2026

For updates and contributions, see the [companion guides](README.md).

---

**End of libSQL & Turso Development Guidelines**
