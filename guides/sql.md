# SQL Development Guidelines
Comprehensive standards for SQL query writing, database schema design, normalization, indexing, performance optimization, security, and migration strategies across all SQL databases and programming languages. PostgreSQL, MySQL, SQLite, SQL Server, MariaDB, CockroachDB, Query analyzers, EXPLAIN plans, Migration tools, Type-safe query generators.

---

**Agent Profile**: The Database Architecture Expert
**Role**: Senior Database Engineer, Schema Architect & SQL Performance Specialist
**Objective**: Generate efficient, secure, maintainable, and portable SQL code following modern best practices.
**Tools**: PostgreSQL, MySQL, SQLite, SQL Server, MariaDB, CockroachDB, Query analyzers, EXPLAIN plans, Migration tools, Type-safe query generators.
**Companion Guides**: sqlc.md, postgresql.md, mongodb.md, testing.md, secure-coding.md

---

## 1. Core Philosophies: DATA-FIRST

The agent must adhere to the **DATA-FIRST** principles:

**Test-Driven Development (TDD)**: ALWAYS write database tests BEFORE schema changes. Verify migrations up AND down.
**Regression Shield**: EVERY data bug MUST receive a test BEFORE fixing to prevent data corruption.

- **D**esign for Integrity - Constraints at database level, not application level
- **A**void Premature Optimization - Normalize first, denormalize only with evidence
- **T**ype Safety First - Use type-safe query builders; avoid raw string queries
- **A**udit Everything - Schema changes, data changes, access patterns

- **F**ail Fast - Validate constraints in database, not application
- **I**ndex Strategically - Every query plan reviewed before production
- **R**eproducible Migrations - Idempotent, reversible, tested migrations
- **S**ecurity by Default - Least privilege, parameterized queries, encryption
- **T**ransaction Boundaries - Explicit transaction control, proper isolation

**Additional Principles:**

- **Portability**: Write ANSI SQL when possible; isolate vendor-specific features
- **Versioned Schema**: Schema is code; track in version control
- **Observable**: Log slow queries, track query patterns, monitor locks
- **Documentation**: Every table, column, and constraint documented

---

## 2. Agent Code Generation Requirements (MANDATORY)

### A. Pre-Query Verification Protocol

**CRITICAL: Agents MUST verify database context before generating SQL.**

#### Pre-Task Checklist

**Before writing ANY SQL, the agent MUST:**

1. **Identify Target Database**:
   ```sql
   -- Determine database engine and version
   -- PostgreSQL
   SELECT version();

   -- MySQL
   SELECT VERSION();

   -- SQLite
   SELECT sqlite_version();

   -- SQL Server
   SELECT @@VERSION;
   ```

2. **Understand Schema Context**:
   ```sql
   -- List tables (PostgreSQL/MySQL)
   SELECT table_name FROM information_schema.tables
   WHERE table_schema = 'public';

   -- Check column types before writing queries
   SELECT column_name, data_type, is_nullable
   FROM information_schema.columns
   WHERE table_name = 'target_table';
   ```

3. **Verify Query Plan**:
   ```sql
   -- ALWAYS explain complex queries before production
   EXPLAIN ANALYZE <your_query>;
   ```

### B. Query Writing Verification Checklist

- [ ] Parameterized queries used (NO string concatenation)
- [ ] Explicit column list (NO `SELECT *` in production)
- [ ] Appropriate indexes exist for WHERE/JOIN/ORDER BY
- [ ] Query plan reviewed with EXPLAIN
- [ ] Transaction boundaries defined for multi-statement operations
- [ ] Null handling explicit (COALESCE, IS NULL checks)
- [ ] Pagination implemented for large result sets
- [ ] Timeout/resource limits considered

### C. Schema Change Verification Checklist

- [ ] Migration file created with up AND down scripts
- [ ] Migration tested in development environment
- [ ] Backward compatibility verified (can old code run?)
- [ ] Forward compatibility verified (can new code run with old schema?)
- [ ] Data migration script tested with production-like data
- [ ] Rollback procedure documented and tested
- [ ] Index creation uses CONCURRENTLY where supported
- [ ] Lock duration minimized for ALTER TABLE operations

### D. Prohibited Practices

**NEVER do the following:**

- [ ] Use string concatenation for query building (SQL injection risk)
- [ ] Use `SELECT *` in production code
- [ ] Create tables without primary keys
- [ ] Skip foreign key constraints for "performance"
- [ ] Store money as FLOAT/DOUBLE (use DECIMAL/NUMERIC)
- [ ] Store dates as strings
- [ ] Ignore query plans for complex queries
- [ ] Deploy schema changes without tested rollback
- [ ] Use database superuser for application connections
- [ ] Store plaintext passwords or sensitive data unencrypted

---

## 3. Schema Design Standards (MANDATORY)

### A. Table Design Template

```sql
-- ============================================================
-- Table: users
-- Description: Core user accounts and authentication data
-- Created: 2026-01-15
-- Modified: 2026-01-22
-- ============================================================

CREATE TABLE users (
    -- Primary Key: Use appropriate type for scale
    -- BIGINT for high-volume, UUID for distributed systems
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    -- Alternative: UUID primary key (distributed-friendly)
    -- id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Natural key with unique constraint
    email VARCHAR(255) NOT NULL,

    -- Normalized name fields
    first_name VARCHAR(100) NOT NULL,
    last_name VARCHAR(100) NOT NULL,

    -- Status with constrained values (prefer ENUM or CHECK)
    status VARCHAR(20) NOT NULL DEFAULT 'pending'
        CONSTRAINT ck_users_status
        CHECK (status IN ('pending', 'active', 'suspended', 'deleted')),

    -- Timestamps with timezone (ALWAYS use timezone-aware)
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    deleted_at TIMESTAMPTZ,  -- Soft delete support

    -- Unique constraints
    CONSTRAINT uq_users_email UNIQUE (email)
);

-- Table and column comments (documentation)
COMMENT ON TABLE users IS 'Core user accounts for authentication and profile';
COMMENT ON COLUMN users.id IS 'Unique identifier, auto-generated';
COMMENT ON COLUMN users.email IS 'User email address, used for login';
COMMENT ON COLUMN users.status IS 'Account status: pending, active, suspended, deleted';
COMMENT ON COLUMN users.deleted_at IS 'Soft delete timestamp, NULL if active';
```

### B. Naming Conventions (MANDATORY)

```sql
-- ============================================================
-- NAMING CONVENTIONS
-- ============================================================

-- Tables: plural, snake_case
CREATE TABLE users (...);
CREATE TABLE order_items (...);
CREATE TABLE user_preferences (...);
CREATE TABLE audit_logs (...);

-- Columns: singular, snake_case, descriptive
CREATE TABLE users (
    id BIGINT,                      -- Primary key
    email VARCHAR(255),             -- Simple name
    first_name VARCHAR(100),        -- Compound name
    created_at TIMESTAMPTZ,         -- Timestamp suffix
    is_verified BOOLEAN,            -- Boolean prefix
    login_count INTEGER,            -- Counter suffix
    password_hash VARCHAR(255),     -- Type suffix for derived
    profile_image_url TEXT          -- Purpose + type
);

-- Indexes: idx_{table}_{columns}
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_status_created ON users(status, created_at);
CREATE INDEX idx_orders_user_created ON orders(user_id, created_at DESC);

-- Unique indexes: uq_{table}_{columns}
CREATE UNIQUE INDEX uq_users_email ON users(email);
CREATE UNIQUE INDEX uq_accounts_provider_external
    ON accounts(provider, external_id);

-- Foreign keys: fk_{table}_{referenced_table}
ALTER TABLE orders
ADD CONSTRAINT fk_orders_users
FOREIGN KEY (user_id) REFERENCES users(id);

-- Check constraints: ck_{table}_{description}
ALTER TABLE orders
ADD CONSTRAINT ck_orders_positive_amount
CHECK (total_amount >= 0);

ALTER TABLE products
ADD CONSTRAINT ck_products_price_range
CHECK (price BETWEEN 0 AND 1000000);

-- Default constraints: df_{table}_{column}
ALTER TABLE users
ALTER COLUMN status SET DEFAULT 'pending';

-- Enums/Types: {entity}_{attribute} or descriptive
CREATE TYPE user_status AS ENUM ('pending', 'active', 'suspended', 'deleted');
CREATE TYPE order_state AS ENUM ('draft', 'pending', 'confirmed', 'shipped', 'delivered', 'cancelled');

-- Triggers: tr_{table}_{timing}_{event}
CREATE TRIGGER tr_users_before_update
    BEFORE UPDATE ON users
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at();

-- Functions: fn_{action}_{description} or {action}_{entity}
CREATE FUNCTION fn_update_updated_at() RETURNS TRIGGER ...;
CREATE FUNCTION calculate_order_total(order_id BIGINT) RETURNS NUMERIC ...;

-- Views: vw_{description}
CREATE VIEW vw_active_users AS ...;
CREATE VIEW vw_order_summary AS ...;

-- Materialized views: mv_{description}
CREATE MATERIALIZED VIEW mv_daily_sales AS ...;
```

### C. Data Types Guide (MANDATORY)

```sql
-- ============================================================
-- DATA TYPE SELECTION GUIDE
-- Choose the smallest type that safely holds your data
-- ============================================================

-- IDENTIFIERS
-- -----------
-- BIGINT: Large tables, high insert volume (Twitter, logs)
id BIGINT GENERATED ALWAYS AS IDENTITY

-- INTEGER: Medium tables (< 2 billion rows)
id INTEGER GENERATED ALWAYS AS IDENTITY

-- UUID: Distributed systems, external exposure, no sequential info leak
id UUID DEFAULT gen_random_uuid()

-- STRINGS
-- -------
-- VARCHAR(n): Known max length, enforced limit
email VARCHAR(255)              -- Email (RFC 5321: 254 chars max)
phone VARCHAR(20)               -- Phone numbers
country_code CHAR(2)            -- Fixed length (ISO codes)
postal_code VARCHAR(20)         -- Varies by country

-- TEXT: Unknown/unlimited length (PostgreSQL: same performance as VARCHAR)
description TEXT
content TEXT
notes TEXT

-- CITEXT: Case-insensitive text (PostgreSQL extension)
username CITEXT                 -- Prevents 'John' vs 'john' duplicates

-- NUMBERS
-- -------
-- NUMERIC/DECIMAL: Exact precision (money, measurements)
price NUMERIC(12, 2)            -- Up to 9999999999.99
tax_rate NUMERIC(5, 4)          -- Up to 9.9999 (99.99%)
latitude NUMERIC(9, 6)          -- -180.000000 to 180.000000

-- INTEGER types: Counting, IDs, quantities
quantity INTEGER                -- -2B to 2B
small_count SMALLINT           -- -32K to 32K (age, rating)
tiny_flag SMALLINT             -- When no TINYINT available

-- FLOAT/DOUBLE: Scientific data (NEVER for money)
sensor_reading DOUBLE PRECISION
temperature REAL

-- BOOLEAN
-- -------
is_active BOOLEAN DEFAULT false
is_verified BOOLEAN NOT NULL DEFAULT false
has_accepted_terms BOOLEAN NOT NULL

-- DATES AND TIMES
-- ---------------
-- TIMESTAMPTZ: Moments in time (ALWAYS prefer over TIMESTAMP)
created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
expires_at TIMESTAMPTZ
scheduled_for TIMESTAMPTZ

-- TIMESTAMP (without timezone): Avoid unless intentional
-- Use only for abstract times not tied to a moment

-- DATE: Calendar dates without time
birth_date DATE
effective_date DATE

-- TIME: Time of day without date
opening_time TIME
closing_time TIME

-- INTERVAL: Durations
retention_period INTERVAL DEFAULT '30 days'
cooldown_duration INTERVAL

-- BINARY
-- ------
-- BYTEA (PostgreSQL) / BLOB (MySQL/SQLite): Binary data
file_content BYTEA
thumbnail BYTEA

-- JSON
-- ----
-- JSONB (PostgreSQL): Structured data, queryable
metadata JSONB DEFAULT '{}'
preferences JSONB
api_response JSONB

-- JSON: When you don't need querying (slightly smaller)
raw_payload JSON

-- ARRAYS (PostgreSQL)
-- ------
tags TEXT[]
category_ids INTEGER[]
```

### D. Constraint Patterns

```sql
-- ============================================================
-- CONSTRAINT PATTERNS
-- Enforce data integrity at database level
-- ============================================================

-- PRIMARY KEY
-- -----------
-- Single column
CREATE TABLE users (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY
);

-- Composite key (junction tables)
CREATE TABLE user_roles (
    user_id BIGINT NOT NULL,
    role_id BIGINT NOT NULL,
    granted_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (user_id, role_id)
);

-- FOREIGN KEYS
-- ------------
-- Standard reference with cascading
CREATE TABLE orders (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    user_id BIGINT NOT NULL,

    CONSTRAINT fk_orders_users
    FOREIGN KEY (user_id) REFERENCES users(id)
    ON DELETE RESTRICT      -- Prevent deletion of referenced user
    ON UPDATE CASCADE       -- Update if user.id changes (rare)
);

-- Different cascade behaviors
ON DELETE CASCADE           -- Delete child rows (use carefully!)
ON DELETE RESTRICT          -- Prevent parent deletion (safe default)
ON DELETE SET NULL          -- Set FK to NULL (requires nullable FK)
ON DELETE SET DEFAULT       -- Set to default value
ON DELETE NO ACTION         -- Similar to RESTRICT

-- Self-referencing FK (hierarchies)
CREATE TABLE employees (
    id BIGINT PRIMARY KEY,
    manager_id BIGINT,
    CONSTRAINT fk_employees_manager
    FOREIGN KEY (manager_id) REFERENCES employees(id)
    ON DELETE SET NULL
);

-- UNIQUE CONSTRAINTS
-- ------------------
-- Single column
ALTER TABLE users ADD CONSTRAINT uq_users_email UNIQUE (email);

-- Composite unique
ALTER TABLE subscriptions ADD CONSTRAINT uq_subscriptions_user_plan
UNIQUE (user_id, plan_id);

-- Partial unique (PostgreSQL) - unique only for active records
CREATE UNIQUE INDEX uq_users_active_email
ON users(email)
WHERE deleted_at IS NULL;

-- CHECK CONSTRAINTS
-- -----------------
-- Value range
ALTER TABLE products ADD CONSTRAINT ck_products_positive_price
CHECK (price >= 0);

-- Value list
ALTER TABLE orders ADD CONSTRAINT ck_orders_valid_status
CHECK (status IN ('pending', 'confirmed', 'shipped', 'delivered', 'cancelled'));

-- Complex validation
ALTER TABLE events ADD CONSTRAINT ck_events_valid_dates
CHECK (end_date >= start_date);

ALTER TABLE discounts ADD CONSTRAINT ck_discounts_valid_percentage
CHECK (percentage BETWEEN 0 AND 100);

-- NOT NULL (simplest constraint)
ALTER TABLE users ALTER COLUMN email SET NOT NULL;

-- DEFAULT VALUES
-- --------------
ALTER TABLE users ALTER COLUMN created_at SET DEFAULT NOW();
ALTER TABLE users ALTER COLUMN status SET DEFAULT 'pending';
ALTER TABLE counters ALTER COLUMN count SET DEFAULT 0;
```

---

## 4. Normalization Guidelines (MANDATORY)

### A. Normal Forms Reference

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         NORMALIZATION LEVELS                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  1NF (First Normal Form)                                                      │
│  └── Atomic values only, no repeating groups                                 │
│      ✗ products: "red,blue,green"                                            │
│      ✓ product_colors: product_id, color                                     │
│                                                                               │
│  2NF (Second Normal Form)                                                     │
│  └── 1NF + No partial dependencies on composite key                          │
│      ✗ order_items: order_id, product_id, product_name                       │
│      ✓ order_items: order_id, product_id (product_name in products)          │
│                                                                               │
│  3NF (Third Normal Form)                                                      │
│  └── 2NF + No transitive dependencies                                        │
│      ✗ orders: order_id, user_id, user_email                                 │
│      ✓ orders: order_id, user_id (user_email in users)                       │
│                                                                               │
│  BCNF (Boyce-Codd Normal Form)                                                │
│  └── 3NF + Every determinant is a candidate key                              │
│                                                                               │
│  4NF (Fourth Normal Form)                                                     │
│  └── BCNF + No multi-valued dependencies                                     │
│                                                                               │
│  5NF (Fifth Normal Form)                                                      │
│  └── 4NF + No join dependencies                                              │
│                                                                               │
│  Practical Target: 3NF/BCNF for OLTP, denormalize for OLAP with evidence    │
│                                                                               │
└─────────────────────────────────────────────────────────────────────────────┘
```

### B. Normalization Examples

```sql
-- ============================================================
-- 1NF VIOLATION AND FIX
-- ============================================================

-- ❌ WRONG: Repeating groups / non-atomic values
CREATE TABLE orders_bad (
    id INTEGER PRIMARY KEY,
    customer_name VARCHAR(100),
    products VARCHAR(500)  -- "Widget,Gadget,Gizmo" - BAD!
);

-- ✅ CORRECT: Atomic values, separate table
CREATE TABLE orders (
    id INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL REFERENCES customers(id)
);

CREATE TABLE order_items (
    order_id INTEGER NOT NULL REFERENCES orders(id),
    product_id INTEGER NOT NULL REFERENCES products(id),
    quantity INTEGER NOT NULL DEFAULT 1,
    PRIMARY KEY (order_id, product_id)
);

-- ============================================================
-- 2NF VIOLATION AND FIX
-- ============================================================

-- ❌ WRONG: Partial dependency (product_name depends only on product_id)
CREATE TABLE order_items_bad (
    order_id INTEGER,
    product_id INTEGER,
    product_name VARCHAR(100),  -- Depends only on product_id!
    quantity INTEGER,
    PRIMARY KEY (order_id, product_id)
);

-- ✅ CORRECT: No partial dependencies
CREATE TABLE products (
    id INTEGER PRIMARY KEY,
    name VARCHAR(100) NOT NULL
);

CREATE TABLE order_items (
    order_id INTEGER REFERENCES orders(id),
    product_id INTEGER REFERENCES products(id),
    quantity INTEGER NOT NULL,
    PRIMARY KEY (order_id, product_id)
);

-- ============================================================
-- 3NF VIOLATION AND FIX
-- ============================================================

-- ❌ WRONG: Transitive dependency (city -> zip_code -> state)
CREATE TABLE customers_bad (
    id INTEGER PRIMARY KEY,
    name VARCHAR(100),
    zip_code VARCHAR(10),
    city VARCHAR(100),      -- Depends on zip_code, not customer!
    state VARCHAR(50)       -- Depends on zip_code, not customer!
);

-- ✅ CORRECT: No transitive dependencies
CREATE TABLE zip_codes (
    code VARCHAR(10) PRIMARY KEY,
    city VARCHAR(100) NOT NULL,
    state VARCHAR(50) NOT NULL
);

CREATE TABLE customers (
    id INTEGER PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    zip_code VARCHAR(10) REFERENCES zip_codes(code)
);
```

### C. When to Denormalize

```sql
-- ============================================================
-- STRATEGIC DENORMALIZATION
-- Only with measured evidence, not speculation
-- ============================================================

-- ACCEPTABLE: Caching frequently-joined data with triggers
CREATE TABLE orders (
    id BIGINT PRIMARY KEY,
    user_id BIGINT NOT NULL REFERENCES users(id),
    -- Denormalized for display (sync via trigger)
    user_email VARCHAR(255) NOT NULL,
    user_name VARCHAR(200) NOT NULL,

    total_amount NUMERIC(12,2) NOT NULL,
    item_count INTEGER NOT NULL DEFAULT 0,  -- Cached count
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Trigger to maintain denormalized data
CREATE FUNCTION fn_sync_order_user_data() RETURNS TRIGGER AS $$
BEGIN
    SELECT email, first_name || ' ' || last_name
    INTO NEW.user_email, NEW.user_name
    FROM users WHERE id = NEW.user_id;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER tr_orders_before_insert
    BEFORE INSERT ON orders
    FOR EACH ROW
    EXECUTE FUNCTION fn_sync_order_user_data();

-- ACCEPTABLE: Materialized views for complex aggregations
CREATE MATERIALIZED VIEW mv_product_stats AS
SELECT
    p.id AS product_id,
    p.name AS product_name,
    COUNT(oi.id) AS times_ordered,
    SUM(oi.quantity) AS total_quantity_sold,
    AVG(oi.unit_price) AS average_price
FROM products p
LEFT JOIN order_items oi ON oi.product_id = p.id
LEFT JOIN orders o ON o.id = oi.order_id
WHERE o.status = 'delivered'
GROUP BY p.id, p.name;

-- Refresh strategy
REFRESH MATERIALIZED VIEW CONCURRENTLY mv_product_stats;

-- WHEN TO DENORMALIZE:
-- ✓ Read-heavy with complex joins (verified by query analysis)
-- ✓ Reporting/analytics (OLAP) workloads
-- ✓ Caching computed values (with sync mechanism)
-- ✓ Historical snapshots (order contained these values at time)

-- WHEN NOT TO DENORMALIZE:
-- ✗ "Just in case" or "for performance" without evidence
-- ✗ Frequently updated source data (sync overhead)
-- ✗ When application can cache effectively
-- ✗ Before measuring actual query performance
```

---

## 5. Indexing Strategy (MANDATORY)

### A. Index Types Reference

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            INDEX TYPES                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  B-Tree (Default)                                                             │
│  └── Best for: =, <, >, <=, >=, BETWEEN, IN, IS NULL                         │
│  └── Most common index type, ordered data                                    │
│  └── Use for: Primary keys, foreign keys, most lookups                       │
│                                                                               │
│  Hash                                                                         │
│  └── Best for: = comparisons only                                            │
│  └── Faster equality checks, no ordering                                     │
│  └── Use for: Exact match lookups (rare, B-tree usually better)              │
│                                                                               │
│  GiST (Generalized Search Tree)                                               │
│  └── Best for: Geometric, full-text, range types                             │
│  └── Use for: PostGIS, ltree, range queries                                  │
│                                                                               │
│  GIN (Generalized Inverted Index)                                             │
│  └── Best for: Multiple values per row                                       │
│  └── Use for: Arrays, JSONB, full-text search                                │
│                                                                               │
│  BRIN (Block Range Index)                                                     │
│  └── Best for: Physically sorted data, very large tables                     │
│  └── Use for: Time-series, append-only tables, logs                          │
│                                                                               │
│  Covering Index (INCLUDE)                                                     │
│  └── Best for: Index-only scans                                              │
│  └── Use for: Avoiding table lookups for SELECT columns                      │
│                                                                               │
│  Partial Index (WHERE clause)                                                 │
│  └── Best for: Subset of rows commonly queried                               │
│  └── Use for: Active records, recent data, specific statuses                 │
│                                                                               │
└─────────────────────────────────────────────────────────────────────────────┘
```

### B. Index Design Patterns

```sql
-- ============================================================
-- ESSENTIAL INDEXES
-- ============================================================

-- PRIMARY KEY (automatic index)
CREATE TABLE users (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY
);

-- FOREIGN KEY INDEXES (ALWAYS create these manually!)
-- Most databases don't auto-create FK indexes
CREATE INDEX idx_orders_user_id ON orders(user_id);
CREATE INDEX idx_order_items_order_id ON order_items(order_id);
CREATE INDEX idx_order_items_product_id ON order_items(product_id);

-- UNIQUE CONSTRAINTS (automatic index)
ALTER TABLE users ADD CONSTRAINT uq_users_email UNIQUE (email);

-- ============================================================
-- COMPOSITE INDEXES
-- ============================================================

-- Column order matters! Most selective first, or match query order
-- For: WHERE user_id = ? AND status = ? ORDER BY created_at DESC
CREATE INDEX idx_orders_user_status_created
ON orders(user_id, status, created_at DESC);

-- The above index supports these queries:
-- ✓ WHERE user_id = ?
-- ✓ WHERE user_id = ? AND status = ?
-- ✓ WHERE user_id = ? AND status = ? ORDER BY created_at DESC
-- ✗ WHERE status = ? (user_id not specified)

-- ============================================================
-- COVERING INDEXES (Index-only scans)
-- ============================================================

-- INCLUDE columns retrieved without table lookup
CREATE INDEX idx_users_email_covering
ON users(email)
INCLUDE (first_name, last_name, status);

-- Now this query uses index-only scan:
-- SELECT first_name, last_name, status FROM users WHERE email = ?

-- ============================================================
-- PARTIAL INDEXES (Filtered)
-- ============================================================

-- Index only active records (smaller, faster)
CREATE INDEX idx_users_active_email
ON users(email)
WHERE deleted_at IS NULL;

-- Index only pending orders
CREATE INDEX idx_orders_pending
ON orders(created_at DESC)
WHERE status = 'pending';

-- Unique constraint only for active (allows soft delete reuse)
CREATE UNIQUE INDEX uq_users_active_email
ON users(email)
WHERE deleted_at IS NULL;

-- ============================================================
-- EXPRESSION INDEXES
-- ============================================================

-- Index on lowercase email for case-insensitive search
CREATE INDEX idx_users_email_lower
ON users(LOWER(email));

-- Index on extracted JSON field
CREATE INDEX idx_orders_metadata_priority
ON orders((metadata->>'priority'));

-- Index on computed date part
CREATE INDEX idx_orders_year_month
ON orders(DATE_TRUNC('month', created_at));

-- ============================================================
-- SPECIALIZED INDEXES
-- ============================================================

-- GIN for JSONB queries
CREATE INDEX idx_products_attributes ON products USING GIN (attributes);
-- Supports: WHERE attributes @> '{"color": "red"}'

-- GIN for array contains
CREATE INDEX idx_posts_tags ON posts USING GIN (tags);
-- Supports: WHERE tags @> ARRAY['sql', 'database']

-- GIN for full-text search
CREATE INDEX idx_articles_search ON articles
USING GIN (to_tsvector('english', title || ' ' || content));

-- BRIN for time-series data (very efficient for sorted data)
CREATE INDEX idx_logs_created_brin ON logs USING BRIN (created_at);

-- ============================================================
-- CONCURRENT INDEX CREATION (No table lock)
-- ============================================================

-- For production tables - doesn't block reads/writes
CREATE INDEX CONCURRENTLY idx_users_status ON users(status);

-- Note: CONCURRENTLY cannot be in transaction
-- Note: Takes longer, uses more resources
```

### C. Index Maintenance

```sql
-- ============================================================
-- INDEX ANALYSIS AND MAINTENANCE
-- ============================================================

-- Find unused indexes (PostgreSQL)
SELECT
    schemaname || '.' || relname AS table,
    indexrelname AS index,
    pg_size_pretty(pg_relation_size(indexrelid)) AS size,
    idx_scan AS scans
FROM pg_stat_user_indexes
WHERE idx_scan = 0
AND indexrelname NOT LIKE '%_pkey'
ORDER BY pg_relation_size(indexrelid) DESC;

-- Find missing indexes (tables with seq scans)
SELECT
    relname AS table,
    seq_scan,
    seq_tup_read,
    idx_scan,
    n_live_tup AS rows
FROM pg_stat_user_tables
WHERE seq_scan > idx_scan
AND n_live_tup > 10000
ORDER BY seq_tup_read DESC;

-- Index size report
SELECT
    tablename,
    indexname,
    pg_size_pretty(pg_relation_size(indexname::regclass)) AS size
FROM pg_indexes
WHERE schemaname = 'public'
ORDER BY pg_relation_size(indexname::regclass) DESC;

-- Reindex bloated indexes (maintenance window)
REINDEX INDEX CONCURRENTLY idx_users_email;

-- Analyze table statistics (query planner)
ANALYZE users;
ANALYZE orders;
```

---

## 6. Query Optimization (MANDATORY)

### A. Query Analysis Protocol

```sql
-- ============================================================
-- ALWAYS EXPLAIN BEFORE PRODUCTION
-- ============================================================

-- Basic explain
EXPLAIN SELECT * FROM users WHERE email = 'test@example.com';

-- Detailed with actual execution
EXPLAIN ANALYZE SELECT * FROM users WHERE email = 'test@example.com';

-- Full analysis (PostgreSQL)
EXPLAIN (ANALYZE, BUFFERS, FORMAT TEXT)
SELECT u.*, COUNT(o.id) AS order_count
FROM users u
LEFT JOIN orders o ON o.user_id = u.id
WHERE u.status = 'active'
GROUP BY u.id;

-- ============================================================
-- EXPLAIN OUTPUT INTERPRETATION
-- ============================================================

/*
Key things to look for:

1. Seq Scan on large tables = MISSING INDEX
   "Seq Scan on users" with 1M+ rows = problem

2. Nested Loop with large outer = WRONG JOIN TYPE
   Consider adding index or restructuring query

3. Sort operation = POTENTIAL INDEX OPPORTUNITY
   If sorting frequently, index that column

4. High "actual rows" vs "planned rows" = STALE STATISTICS
   Run ANALYZE on the table

5. "actual time" in explain = ACTUAL PERFORMANCE
   First number is startup time, second is total time

GOOD OUTPUT:
Index Scan using idx_users_email on users
  (cost=0.42..8.44 rows=1 width=...)
  (actual time=0.025..0.027 rows=1 loops=1)

BAD OUTPUT:
Seq Scan on users
  (cost=0.00..25000.00 rows=1000000 width=...)
  (actual time=1500.000..1500.025 rows=1 loops=1)
  Filter: (email = 'test@example.com')
  Rows Removed by Filter: 999999
*/
```

### B. Common Optimization Patterns

```sql
-- ============================================================
-- PATTERN: SELECT ONLY NEEDED COLUMNS
-- ============================================================

-- ❌ WRONG: Select all columns
SELECT * FROM users WHERE id = 1;

-- ✅ CORRECT: Explicit columns
SELECT id, email, first_name, last_name, status
FROM users WHERE id = 1;

-- ============================================================
-- PATTERN: AVOID FUNCTIONS ON INDEXED COLUMNS
-- ============================================================

-- ❌ SLOW: Function on indexed column prevents index use
SELECT * FROM users WHERE LOWER(email) = 'test@example.com';
SELECT * FROM orders WHERE YEAR(created_at) = 2024;

-- ✅ FASTER: Store normalized or use expression index
SELECT * FROM users WHERE email = 'test@example.com'; -- Store lowercase
SELECT * FROM orders
WHERE created_at >= '2024-01-01' AND created_at < '2025-01-01';

-- ============================================================
-- PATTERN: REPLACE OR WITH UNION (when using different indexes)
-- ============================================================

-- ❌ SLOW: OR often prevents index use
SELECT * FROM orders
WHERE user_id = 1 OR status = 'pending';

-- ✅ FASTER: UNION uses both indexes
SELECT * FROM orders WHERE user_id = 1
UNION
SELECT * FROM orders WHERE status = 'pending';

-- ============================================================
-- PATTERN: USE EXISTS INSTEAD OF IN FOR LARGE SUBQUERIES
-- ============================================================

-- ❌ SLOW: IN with large subquery
SELECT * FROM users
WHERE id IN (SELECT user_id FROM orders WHERE total_amount > 1000);

-- ✅ FASTER: EXISTS (short-circuits)
SELECT * FROM users u
WHERE EXISTS (
    SELECT 1 FROM orders o
    WHERE o.user_id = u.id AND o.total_amount > 1000
);

-- ============================================================
-- PATTERN: REPLACE NOT IN WITH LEFT JOIN
-- ============================================================

-- ❌ SLOW: NOT IN with subquery
SELECT * FROM users
WHERE id NOT IN (SELECT user_id FROM orders);

-- ✅ FASTER: LEFT JOIN with NULL check
SELECT u.*
FROM users u
LEFT JOIN orders o ON o.user_id = u.id
WHERE o.id IS NULL;

-- ============================================================
-- PATTERN: AVOID CORRELATED SUBQUERIES
-- ============================================================

-- ❌ SLOW: Correlated subquery (runs for each row)
SELECT *,
    (SELECT COUNT(*) FROM orders WHERE user_id = users.id) AS order_count
FROM users;

-- ✅ FASTER: JOIN with aggregation
SELECT u.*, COALESCE(o.order_count, 0) AS order_count
FROM users u
LEFT JOIN (
    SELECT user_id, COUNT(*) AS order_count
    FROM orders
    GROUP BY user_id
) o ON o.user_id = u.id;

-- ============================================================
-- PATTERN: EFFICIENT PAGINATION
-- ============================================================

-- ❌ SLOW: OFFSET with large values (scans skipped rows)
SELECT * FROM users
ORDER BY created_at DESC
LIMIT 20 OFFSET 100000;  -- Must scan 100,000 rows!

-- ✅ FASTER: Keyset/cursor pagination
SELECT * FROM users
WHERE created_at < '2024-01-15T10:00:00Z'  -- Last item from prev page
ORDER BY created_at DESC
LIMIT 20;

-- For deterministic ordering (same created_at):
SELECT * FROM users
WHERE (created_at, id) < ('2024-01-15T10:00:00Z', 12345)
ORDER BY created_at DESC, id DESC
LIMIT 20;

-- ============================================================
-- PATTERN: LIMIT EARLY IN SUBQUERIES
-- ============================================================

-- ❌ SLOW: Processes all matches before limiting
SELECT u.*, o.created_at AS last_order
FROM users u
LEFT JOIN orders o ON o.user_id = u.id
ORDER BY u.created_at DESC
LIMIT 10;

-- ✅ FASTER: Limit users first, then join
SELECT u.*, o.created_at AS last_order
FROM (
    SELECT * FROM users
    ORDER BY created_at DESC
    LIMIT 10
) u
LEFT JOIN LATERAL (
    SELECT created_at FROM orders
    WHERE user_id = u.id
    ORDER BY created_at DESC
    LIMIT 1
) o ON true;
```

### C. Join Optimization

```sql
-- ============================================================
-- JOIN BEST PRACTICES
-- ============================================================

-- 1. Always use explicit JOIN syntax
-- ❌ AVOID: Implicit joins (comma syntax)
SELECT * FROM users, orders WHERE users.id = orders.user_id;

-- ✅ CORRECT: Explicit JOIN
SELECT * FROM users u
INNER JOIN orders o ON o.user_id = u.id;

-- 2. Join order matters (smaller table first as driver)
-- Query planner usually optimizes, but for complex queries:
SELECT /*+ LEADING(small_table) */ ..

-- 3. Use appropriate join type
INNER JOIN  -- Only matching rows (most common)
LEFT JOIN   -- All left + matching right (preserve left side)
RIGHT JOIN  -- Matching left + all right (rare, rewrite as LEFT)
FULL JOIN   -- All from both sides (rare)
CROSS JOIN  -- Cartesian product (very rare, be careful!)

-- 4. Avoid joining on expressions when possible
-- ❌ SLOW: Join on function result
SELECT * FROM users u
JOIN user_history h ON LOWER(h.email) = LOWER(u.email);

-- ✅ FASTER: Normalize data, join on indexed column
SELECT * FROM users u
JOIN user_history h ON h.user_id = u.id;

-- 5. Consider denormalizing for heavy join workloads
-- Only after measuring actual performance problems
```

---

## 7. Security Best Practices (MANDATORY)

### A. SQL Injection Prevention

```sql
-- ============================================================
-- PARAMETERIZED QUERIES (MANDATORY)
-- Never concatenate user input into SQL
-- ============================================================

-- ❌ NEVER: String concatenation (SQL injection vulnerability!)
-- query = "SELECT * FROM users WHERE email = '" + userInput + "'"
-- Attacker input: ' OR '1'='1' --
-- Results in: SELECT * FROM users WHERE email = '' OR '1'='1' --'

-- ✅ ALWAYS: Parameterized queries

-- PostgreSQL (native)
PREPARE user_query (text) AS
SELECT id, email, name FROM users WHERE email = $1;
EXECUTE user_query('test@example.com');

-- Using application code (examples):

-- Python (psycopg2/asyncpg)
-- cursor.execute("SELECT * FROM users WHERE email = %s", (email,))

-- Go (database/sql)
-- db.Query("SELECT * FROM users WHERE email = $1", email)

-- JavaScript (pg)
-- client.query('SELECT * FROM users WHERE email = $1', [email])

-- Java (JDBC)
-- PreparedStatement ps = conn.prepareStatement(
--     "SELECT * FROM users WHERE email = ?");
-- ps.setString(1, email);

-- TypeScript with sqlc
-- Generated type-safe functions prevent injection by design

-- ============================================================
-- DYNAMIC QUERIES (when unavoidable)
-- ============================================================

-- If dynamic column/table names are needed, whitelist them:
-- ❌ NEVER: Dynamic input directly
-- query = f"SELECT * FROM {table_name}"

-- ✅ ALWAYS: Whitelist and quote identifiers
-- allowed_tables = {'users', 'orders', 'products'}
-- if table_name not in allowed_tables:
--     raise ValueError("Invalid table")
-- query = f"SELECT * FROM {quote_ident(table_name)}"
```

### B. Access Control

```sql
-- ============================================================
-- PRINCIPLE OF LEAST PRIVILEGE
-- ============================================================

-- Create application-specific roles (not superuser!)
CREATE ROLE app_readonly;
CREATE ROLE app_readwrite;
CREATE ROLE app_admin;

-- Grant minimal permissions per role
GRANT SELECT ON ALL TABLES IN SCHEMA public TO app_readonly;

GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA public TO app_readwrite;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA public TO app_readwrite;
-- Note: No DELETE for app_readwrite (use soft delete)

GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO app_admin;

-- Create application users with specific roles
CREATE USER app_api WITH PASSWORD 'secure_random_password';
GRANT app_readwrite TO app_api;

CREATE USER app_reports WITH PASSWORD 'secure_random_password';
GRANT app_readonly TO app_reports;

-- Revoke public access
REVOKE ALL ON ALL TABLES IN SCHEMA public FROM PUBLIC;
REVOKE ALL ON DATABASE myapp FROM PUBLIC;

-- ============================================================
-- ROW-LEVEL SECURITY (RLS)
-- ============================================================

-- Enable RLS on table
ALTER TABLE orders ENABLE ROW LEVEL SECURITY;

-- Policy: Users can only see their own orders
CREATE POLICY orders_user_isolation ON orders
    FOR ALL
    USING (user_id = current_setting('app.current_user_id')::bigint);

-- Policy: Admins can see all orders
CREATE POLICY orders_admin_all ON orders
    FOR ALL
    TO app_admin
    USING (true);

-- Set current user in session
SET app.current_user_id = '12345';

-- Now queries automatically filtered
SELECT * FROM orders; -- Only sees user 12345's orders
```

### C. Data Protection

```sql
-- ============================================================
-- SENSITIVE DATA HANDLING
-- ============================================================

-- Never store plaintext passwords
-- Use bcrypt/argon2 in application code, store only hash
CREATE TABLE users (
    id BIGINT PRIMARY KEY,
    email VARCHAR(255) NOT NULL,
    password_hash VARCHAR(255) NOT NULL,  -- bcrypt/argon2 hash
    -- NEVER: password VARCHAR(255) -- plaintext password
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Encrypt sensitive data at rest (PostgreSQL pgcrypto)
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- Symmetric encryption for PII
INSERT INTO customers (ssn_encrypted)
VALUES (pgp_sym_encrypt('123-45-6789', current_setting('app.encryption_key')));

-- Retrieve decrypted
SELECT pgp_sym_decrypt(ssn_encrypted, current_setting('app.encryption_key')) AS ssn
FROM customers WHERE id = 1;

-- ============================================================
-- AUDIT LOGGING
-- ============================================================

-- Audit log table
CREATE TABLE audit_log (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    table_name VARCHAR(100) NOT NULL,
    record_id BIGINT NOT NULL,
    action VARCHAR(10) NOT NULL CHECK (action IN ('INSERT', 'UPDATE', 'DELETE')),
    old_values JSONB,
    new_values JSONB,
    changed_by VARCHAR(100),
    changed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    ip_address INET,
    user_agent TEXT
);

-- Generic audit trigger
CREATE OR REPLACE FUNCTION fn_audit_trigger() RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'DELETE' THEN
        INSERT INTO audit_log (table_name, record_id, action, old_values, changed_by)
        VALUES (TG_TABLE_NAME, OLD.id, 'DELETE', to_jsonb(OLD), current_setting('app.current_user', true));
        RETURN OLD;
    ELSIF TG_OP = 'UPDATE' THEN
        INSERT INTO audit_log (table_name, record_id, action, old_values, new_values, changed_by)
        VALUES (TG_TABLE_NAME, NEW.id, 'UPDATE', to_jsonb(OLD), to_jsonb(NEW), current_setting('app.current_user', true));
        RETURN NEW;
    ELSIF TG_OP = 'INSERT' THEN
        INSERT INTO audit_log (table_name, record_id, action, new_values, changed_by)
        VALUES (TG_TABLE_NAME, NEW.id, 'INSERT', to_jsonb(NEW), current_setting('app.current_user', true));
        RETURN NEW;
    END IF;
END;
$$ LANGUAGE plpgsql;

-- Apply audit trigger to sensitive tables
CREATE TRIGGER tr_users_audit
    AFTER INSERT OR UPDATE OR DELETE ON users
    FOR EACH ROW EXECUTE FUNCTION fn_audit_trigger();
```

---

## 8. Transaction Management (MANDATORY)

### A. Transaction Patterns

```sql
-- ============================================================
-- EXPLICIT TRANSACTION CONTROL
-- ============================================================

-- Basic transaction
BEGIN;
    UPDATE accounts SET balance = balance - 100 WHERE id = 1;
    UPDATE accounts SET balance = balance + 100 WHERE id = 2;
COMMIT;
-- On error: ROLLBACK;

-- Savepoints for partial rollback
BEGIN;
    INSERT INTO orders (user_id, total) VALUES (1, 100.00);
    SAVEPOINT order_created;

    INSERT INTO order_items (order_id, product_id) VALUES (1, 999);
    -- If this fails:
    ROLLBACK TO SAVEPOINT order_created;
    -- Order still exists, can try different items

COMMIT;

-- ============================================================
-- ISOLATION LEVELS
-- ============================================================

-- Read Committed (PostgreSQL default)
-- Sees only committed data, may see changes between queries
BEGIN ISOLATION LEVEL READ COMMITTED;
-- Good for: Most OLTP operations

-- Repeatable Read
-- Sees consistent snapshot, no phantom reads
BEGIN ISOLATION LEVEL REPEATABLE READ;
-- Good for: Reports, complex multi-query operations

-- Serializable
-- Full isolation, as if transactions ran sequentially
BEGIN ISOLATION LEVEL SERIALIZABLE;
-- Good for: Financial transactions, critical consistency
-- Note: May fail with serialization error, must retry

-- ============================================================
-- ROW LOCKING
-- ============================================================

-- Lock rows for update (prevent concurrent modification)
BEGIN;
SELECT * FROM accounts WHERE id = 1 FOR UPDATE;
-- Other transactions block on this row until commit
UPDATE accounts SET balance = balance - 100 WHERE id = 1;
COMMIT;

-- Lock options:
FOR UPDATE           -- Exclusive lock, blocks all
FOR NO KEY UPDATE    -- Allows FK checks by other transactions
FOR SHARE            -- Shared lock, blocks UPDATE/DELETE
FOR KEY SHARE        -- Allows non-key updates

-- Skip locked rows (queue processing)
SELECT * FROM jobs
WHERE status = 'pending'
FOR UPDATE SKIP LOCKED
LIMIT 1;

-- ============================================================
-- DEADLOCK PREVENTION
-- ============================================================

-- Always acquire locks in consistent order
-- ❌ WRONG: Transaction A locks 1 then 2, Transaction B locks 2 then 1
-- ✅ CORRECT: Both transactions lock in same order (e.g., by ID ascending)

BEGIN;
SELECT * FROM accounts WHERE id IN (1, 2) ORDER BY id FOR UPDATE;
-- Process both accounts
COMMIT;

-- Set lock timeout
SET lock_timeout = '5s';  -- Fail rather than wait forever
```

### B. Transaction Best Practices

```sql
-- ============================================================
-- TRANSACTION GUIDELINES
-- ============================================================

-- 1. Keep transactions short
-- ❌ WRONG: Long transaction with user interaction
BEGIN;
SELECT * FROM products;  -- Display to user
-- ... user thinks for 5 minutes ..
UPDATE products SET stock = stock - 1;  -- Lock held entire time!
COMMIT;

-- ✅ CORRECT: Verify and act in single short transaction
BEGIN;
SELECT stock FROM products WHERE id = 1 FOR UPDATE;
-- Check stock >= 1 in application
UPDATE products SET stock = stock - 1 WHERE id = 1;
INSERT INTO orders (...);
COMMIT;

-- 2. Don't mix DDL and DML in same transaction
-- DDL may have implicit commits (database-dependent)

-- 3. Handle errors properly (application code)
-- try:
--     begin()
--     execute(...)
--     commit()
-- except:
--     rollback()
--     raise

-- 4. Use appropriate isolation level
-- Don't use SERIALIZABLE everywhere "just to be safe"
-- Higher isolation = more contention = less throughput
```

---

## 9. Migrations Best Practices (MANDATORY)

### A. Migration File Structure

```
db/
├── migrations/
│   ├── 000001_create_users.up.sql
│   ├── 000001_create_users.down.sql
│   ├── 000002_add_user_status.up.sql
│   ├── 000002_add_user_status.down.sql
│   ├── 000003_create_orders.up.sql
│   ├── 000003_create_orders.down.sql
│   ├── 000004_add_order_indexes.up.sql
│   └── 000004_add_order_indexes.down.sql
├── seeds/
│   ├── development/
│   │   └── 001_sample_users.sql
│   └── test/
│       └── 001_test_fixtures.sql
└── schema.sql  -- Generated: pg_dump --schema-only
```

### B. Migration Patterns

```sql
-- ============================================================
-- SAFE MIGRATION PATTERNS
-- ============================================================

-- Pattern: ADD NULLABLE COLUMN (always safe)
-- 000002_add_user_phone.up.sql
ALTER TABLE users ADD COLUMN phone VARCHAR(20);

-- 000002_add_user_phone.down.sql
ALTER TABLE users DROP COLUMN phone;


-- Pattern: ADD NOT NULL WITH DEFAULT (safe in PostgreSQL 11+)
-- 000003_add_user_verified.up.sql
ALTER TABLE users ADD COLUMN is_verified BOOLEAN NOT NULL DEFAULT false;

-- 000003_add_user_verified.down.sql
ALTER TABLE users DROP COLUMN is_verified;


-- Pattern: ADD INDEX CONCURRENTLY (no lock)
-- 000004_add_user_phone_index.up.sql
CREATE INDEX CONCURRENTLY idx_users_phone ON users(phone);

-- 000004_add_user_phone_index.down.sql
DROP INDEX CONCURRENTLY idx_users_phone;

-- Note: CONCURRENTLY cannot be in transaction block


-- Pattern: ADD CHECK CONSTRAINT (multi-step)
-- Step 1: Add as NOT VALID (no scan)
-- 000005_add_order_amount_check_step1.up.sql
ALTER TABLE orders
ADD CONSTRAINT ck_orders_positive_amount
CHECK (total_amount >= 0) NOT VALID;

-- Step 2: Validate existing data (can be slow)
-- 000006_add_order_amount_check_step2.up.sql
ALTER TABLE orders
VALIDATE CONSTRAINT ck_orders_positive_amount;


-- Pattern: RENAME COLUMN (with backward compatibility)
-- 000007_rename_user_name.up.sql
-- Step 1: Add new column
ALTER TABLE users ADD COLUMN display_name VARCHAR(200);

-- Step 2: Copy data
UPDATE users SET display_name = name WHERE display_name IS NULL;

-- Step 3: Create view for old code (temporary)
CREATE VIEW users_v1 AS
SELECT *, display_name AS name FROM users;

-- 000007_rename_user_name.down.sql
DROP VIEW IF EXISTS users_v1;
ALTER TABLE users DROP COLUMN display_name;

-- Step 4 (separate migration after code deployment):
-- ALTER TABLE users DROP COLUMN name;


-- ============================================================
-- DATA MIGRATIONS
-- ============================================================

-- Backfill in batches (prevent locks, allow progress)
-- 000010_backfill_user_slugs.up.sql
DO $$
DECLARE
    batch_size INT := 1000;
    total_updated INT := 0;
    batch_updated INT;
BEGIN
    LOOP
        UPDATE users
        SET slug = LOWER(REPLACE(name, ' ', '-'))
        WHERE id IN (
            SELECT id FROM users
            WHERE slug IS NULL
            LIMIT batch_size
            FOR UPDATE SKIP LOCKED
        );

        GET DIAGNOSTICS batch_updated = ROW_COUNT;
        total_updated := total_updated + batch_updated;

        RAISE NOTICE 'Updated % rows (total: %)', batch_updated, total_updated;

        EXIT WHEN batch_updated = 0;

        -- Optional: Sleep between batches to reduce load
        PERFORM pg_sleep(0.1);
    END LOOP;
END $$;
```

### C. Dangerous Operations

```sql
-- ============================================================
-- DANGEROUS OPERATIONS - REQUIRE EXTRA CARE
-- ============================================================

-- ⚠️ DROPPING COLUMNS
-- - Backup data first
-- - Verify no code references
ALTER TABLE users DROP COLUMN IF EXISTS deprecated_field;

-- ⚠️ DROPPING TABLES
-- - Backup table first
-- - Ensure no foreign key references
DROP TABLE IF EXISTS deprecated_table;

-- ⚠️ CHANGING COLUMN TYPE
-- - May lock table for rewrite
-- - May fail if data doesn't fit
ALTER TABLE products ALTER COLUMN price TYPE NUMERIC(12,2);

-- ⚠️ ADDING NOT NULL TO EXISTING COLUMN
-- - Verify no nulls exist first
-- - Add default first, backfill, then add constraint
UPDATE users SET phone = '' WHERE phone IS NULL;
ALTER TABLE users ALTER COLUMN phone SET NOT NULL;

-- ⚠️ CHANGING PRIMARY KEY
-- - Cascades to all foreign keys
-- - Major operation, plan carefully
-- - Consider using logical replication for zero-downtime

-- ⚠️ RENAMING TABLES
-- - All code must be updated simultaneously
-- - Consider view-based migration
ALTER TABLE users RENAME TO user_accounts;
CREATE VIEW users AS SELECT * FROM user_accounts; -- Temporary compatibility
```

### D. Migration Testing

```bash
#!/bin/bash
# test_migrations.sh

set -e

echo "Testing migrations..."

# Create test database
createdb migration_test

# Run all up migrations
migrate -path ./migrations -database "postgres://localhost/migration_test" up

# Verify schema
pg_dump --schema-only migration_test > /tmp/schema_up.sql

# Run all down migrations
migrate -path ./migrations -database "postgres://localhost/migration_test" down -all

# Verify clean state
TABLE_COUNT=$(psql -t -c "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema='public'" migration_test)
if [ "$TABLE_COUNT" -gt 0 ]; then
    echo "ERROR: Tables remain after down migrations!"
    exit 1
fi

# Run up again (verify reproducibility)
migrate -path ./migrations -database "postgres://localhost/migration_test" up

# Compare schemas
pg_dump --schema-only migration_test > /tmp/schema_up2.sql
diff /tmp/schema_up.sql /tmp/schema_up2.sql || echo "WARNING: Schema differs on second run"

# Cleanup
dropdb migration_test

echo "Migration tests passed!"
```

---

## 10. ORM and Query Builder Integration

### A. When to Use What

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    DATABASE ACCESS STRATEGY DECISION                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  Type-Safe Query Generators (RECOMMENDED)                                     │
│  ├── Examples: sqlc (Go/Python/TS/Kotlin), jOOQ (Java), Prisma (TS)         │
│  ├── Best for: Known queries, type safety, performance                       │
│  ├── Write SQL → Generate code → Compile-time verification                   │
│  └── SQL remains source of truth                                             │
│                                                                               │
│  Query Builders                                                               │
│  ├── Examples: Knex (JS), SQLAlchemy Core (Python), Diesel (Rust)           │
│  ├── Best for: Dynamic queries, multi-database support                       │
│  └── Programmatic SQL construction with type safety                          │
│                                                                               │
│  ORMs (Object-Relational Mappers)                                             │
│  ├── Examples: SQLAlchemy ORM, Hibernate, Entity Framework, ActiveRecord    │
│  ├── Best for: Rapid development, simple CRUD                                │
│  ├── Risk: N+1 queries, abstraction leaks, hidden complexity                │
│  └── Watch: Generated SQL, lazy loading, transaction boundaries              │
│                                                                               │
│  Raw SQL                                                                      │
│  ├── Use for: Complex queries, performance-critical paths, database-specific│
│  ├── Always: Parameterized queries                                           │
│  └── Risk: Manual type handling, no compile-time checks                      │
│                                                                               │
│  RECOMMENDATION PRIORITY:                                                     │
│  1. Type-safe generators (sqlc) for static queries                           │
│  2. Query builders for dynamic queries                                        │
│  3. ORM for simple CRUD + careful monitoring                                  │
│  4. Raw SQL for database-specific features                                    │
│                                                                               │
└─────────────────────────────────────────────────────────────────────────────┘
```

### B. sqlc Integration (Recommended)

```yaml
# sqlc.yaml - Type-safe SQL for multiple languages
version: "2"

sql:
  - engine: "postgresql"
    queries: "db/queries/"
    schema: "db/migrations/"
    gen:
      go:
        package: "db"
        out: "internal/db"
        sql_package: "pgx/v5"
        emit_json_tags: true
        emit_interface: true
```

```sql
-- db/queries/users.sql
-- sqlc generates type-safe functions from these queries

-- name: GetUser :one
SELECT id, email, first_name, last_name, status, created_at
FROM users
WHERE id = $1 AND deleted_at IS NULL;

-- name: ListActiveUsers :many
SELECT id, email, first_name, last_name, created_at
FROM users
WHERE status = 'active' AND deleted_at IS NULL
ORDER BY created_at DESC
LIMIT $1 OFFSET $2;

-- name: CreateUser :one
INSERT INTO users (email, first_name, last_name, password_hash)
VALUES ($1, $2, $3, $4)
RETURNING id, email, first_name, last_name, status, created_at;
```

See [sqlc.md](./sqlc.md) for comprehensive sqlc guidelines.

### C. ORM Best Practices

```python
# SQLAlchemy example - avoiding common pitfalls

from sqlalchemy import select
from sqlalchemy.orm import joinedload, selectinload

# ❌ WRONG: N+1 query problem
users = session.query(User).all()
for user in users:
    print(user.orders)  # Each access = new query!

# ✅ CORRECT: Eager loading
users = session.query(User).options(
    selectinload(User.orders)  # One additional query for all orders
).all()

# ❌ WRONG: Loading too much data
users = session.query(User).all()  # Loads all columns

# ✅ CORRECT: Load only needed columns
users = session.execute(
    select(User.id, User.email, User.name)
    .where(User.status == 'active')
).all()

# ❌ WRONG: Implicit transactions
user = session.query(User).get(1)
user.status = 'active'
# When does commit happen? Depends on session config..

# ✅ CORRECT: Explicit transaction boundaries
with session.begin():
    user = session.query(User).with_for_update().get(1)
    user.status = 'active'
# Auto-commit at end of block, auto-rollback on exception

# ALWAYS: Check generated SQL in development
import logging
logging.getLogger('sqlalchemy.engine').setLevel(logging.INFO)
```

---

## 11. Database Portability

### A. ANSI SQL Compatibility

```sql
-- ============================================================
-- PORTABLE SQL PATTERNS
-- Write for ANSI SQL, isolate vendor-specific features
-- ============================================================

-- ✅ PORTABLE: Standard aggregate functions
SELECT COUNT(*), SUM(amount), AVG(amount), MIN(amount), MAX(amount)
FROM orders;

-- ✅ PORTABLE: Standard joins
SELECT u.name, o.total
FROM users u
INNER JOIN orders o ON o.user_id = u.id;

-- ✅ PORTABLE: Standard subqueries
SELECT * FROM users
WHERE id IN (SELECT user_id FROM orders WHERE total > 100);

-- ✅ PORTABLE: Standard CASE expressions
SELECT
    name,
    CASE status
        WHEN 'active' THEN 'Active'
        WHEN 'pending' THEN 'Pending'
        ELSE 'Unknown'
    END AS status_label
FROM users;

-- ============================================================
-- VENDOR-SPECIFIC FEATURES (Isolate in database layer)
-- ============================================================

-- PostgreSQL-specific
-- JSONB, Arrays, CITEXT, LATERAL, UPSERT syntax
-- Use views or functions to abstract if portability needed

-- MySQL-specific
-- AUTO_INCREMENT, ENUM without CREATE TYPE
-- REPLACE INTO, INSERT IGNORE

-- SQLite-specific
-- ROWID, typeof(), json_* functions
-- Limited ALTER TABLE support

-- SQL Server-specific
-- TOP instead of LIMIT, OUTPUT clause
-- Different date functions
```

### B. Abstraction Layer Pattern

```sql
-- ============================================================
-- ABSTRACT VENDOR-SPECIFIC FEATURES IN VIEWS/FUNCTIONS
-- ============================================================

-- Example: UUID generation abstraction
-- PostgreSQL: gen_random_uuid()
-- MySQL: UUID()
-- SQLite: lower(hex(randomblob(16)))

-- Create compatibility function (PostgreSQL)
CREATE OR REPLACE FUNCTION generate_id() RETURNS UUID AS $$
BEGIN
    RETURN gen_random_uuid();
END;
$$ LANGUAGE plpgsql;

-- Use in application code
INSERT INTO users (id, email) VALUES (generate_id(), 'test@example.com');

-- Example: Date truncation abstraction
-- PostgreSQL: DATE_TRUNC('month', date)
-- MySQL: DATE_FORMAT(date, '%Y-%m-01')
-- SQL Server: DATEADD(MONTH, DATEDIFF(MONTH, 0, date), 0)

CREATE OR REPLACE FUNCTION trunc_month(d TIMESTAMPTZ) RETURNS DATE AS $$
BEGIN
    RETURN DATE_TRUNC('month', d)::DATE;
END;
$$ LANGUAGE plpgsql;
```

---

## 12. Testing Database Code (MANDATORY)

### A. Test Database Setup

```yaml
# docker-compose.test.yml
version: '3.8'
services:
  test-db:
    image: postgres:16
    environment:
      POSTGRES_DB: test
      POSTGRES_USER: test
      POSTGRES_PASSWORD: test
    ports:
      - "5433:5432"
    tmpfs:
      - /var/lib/postgresql/data  # RAM disk for speed
    command:
      - postgres
      - -c
      - fsync=off  # Faster for tests (not for production!)
      - -c
      - synchronous_commit=off
```

### B. Test Patterns

```go
// Go example with testcontainers
package db_test

import (
    "context"
    "testing"

    "github.com/testcontainers/testcontainers-go/modules/postgres"
)

var testDB *sql.DB

func TestMain(m *testing.M) {
    ctx := context.Background()

    // Start container
    container, _ := postgres.Run(ctx, "postgres:16",
        postgres.WithDatabase("test"),
        postgres.WithUsername("test"),
        postgres.WithPassword("test"),
    )
    defer container.Terminate(ctx)

    // Get connection
    connStr, _ := container.ConnectionString(ctx)
    testDB, _ = sql.Open("postgres", connStr)

    // Run migrations
    runMigrations(testDB)

    os.Exit(m.Run())
}

func TestCreateUser(t *testing.T) {
    // Clean state
    cleanupUsers(t)

    // Test
    user, err := queries.CreateUser(ctx, CreateUserParams{
        Email: "test@example.com",
        Name: "Test User",
    })

    assert.NoError(t, err)
    assert.NotEmpty(t, user.ID)
    assert.Equal(t, "test@example.com", user.Email)
}

func TestUserUniqueEmail(t *testing.T) {
    cleanupUsers(t)

    // Create first user
    _, err := queries.CreateUser(ctx, CreateUserParams{
        Email: "duplicate@example.com",
        Name: "First",
    })
    require.NoError(t, err)

    // Try duplicate - should fail
    _, err = queries.CreateUser(ctx, CreateUserParams{
        Email: "duplicate@example.com",
        Name: "Second",
    })

    // Verify unique constraint violation
    var pgErr *pgconn.PgError
    require.ErrorAs(t, err, &pgErr)
    assert.Equal(t, "23505", pgErr.Code) // unique_violation
}
```

### C. Migration Testing

```bash
#!/bin/bash
# test_migrations.sh

set -euo pipefail

# Test up migrations
echo "Testing UP migrations..."
migrate -path ./migrations -database "$TEST_DATABASE_URL" up
echo "UP migrations: OK"

# Test down migrations
echo "Testing DOWN migrations..."
migrate -path ./migrations -database "$TEST_DATABASE_URL" down -all
echo "DOWN migrations: OK"

# Test idempotency (up again)
echo "Testing idempotency..."
migrate -path ./migrations -database "$TEST_DATABASE_URL" up
echo "Idempotency: OK"

# Verify schema matches expected
echo "Verifying schema..."
pg_dump --schema-only "$TEST_DATABASE_URL" > /tmp/actual_schema.sql
diff expected_schema.sql /tmp/actual_schema.sql
echo "Schema verification: OK"

echo "All migration tests passed!"
```

---

## 13. Deployment Checklist

### Pre-Deployment

**Schema Changes:**
- [ ] Migration files created with up AND down scripts
- [ ] Migrations tested in development environment
- [ ] Migrations tested in staging with production-like data
- [ ] Rollback procedure documented and tested
- [ ] Backward compatibility verified (old code with new schema)
- [ ] Forward compatibility verified (new code with old schema)
- [ ] Lock duration analyzed for ALTER TABLE operations
- [ ] Index creation uses CONCURRENTLY where supported

**Query Performance:**
- [ ] All new queries have EXPLAIN ANALYZE output reviewed
- [ ] Indexes created for new access patterns
- [ ] No sequential scans on large tables
- [ ] Pagination implemented for list endpoints
- [ ] Query timeout configured

**Security:**
- [ ] Parameterized queries used exclusively
- [ ] No SELECT * in production queries
- [ ] Application uses least-privilege database role
- [ ] Sensitive data encrypted at rest
- [ ] Audit logging enabled for sensitive operations
- [ ] Connection strings secured (not in code)

**Testing:**
- [ ] Unit tests for query logic
- [ ] Integration tests with test database
- [ ] Migration tests (up and down)
- [ ] Performance tests with production-scale data
- [ ] Rollback tested in staging

### Post-Deployment

- [ ] Monitor slow query logs
- [ ] Check for lock contention
- [ ] Verify connection pool health
- [ ] Confirm migration success
- [ ] Validate data integrity

---

## 14. Quick Reference

### Common Patterns

```sql
-- Soft delete
UPDATE users SET deleted_at = NOW() WHERE id = 1;
SELECT * FROM users WHERE deleted_at IS NULL;

-- Optimistic locking
UPDATE products
SET stock = stock - 1, version = version + 1
WHERE id = 1 AND version = 5;  -- Fails if version changed

-- Keyset pagination (efficient for large offsets)
SELECT * FROM users
WHERE (created_at, id) < ('2024-01-15', 12345)
ORDER BY created_at DESC, id DESC
LIMIT 20;

-- Upsert (INSERT or UPDATE)
INSERT INTO settings (user_id, key, value)
VALUES (1, 'theme', 'dark')
ON CONFLICT (user_id, key)
DO UPDATE SET value = EXCLUDED.value, updated_at = NOW();

-- Conditional aggregation
SELECT
    COUNT(*) FILTER (WHERE status = 'active') AS active_count,
    COUNT(*) FILTER (WHERE status = 'pending') AS pending_count
FROM users;

-- Window functions for ranking
SELECT
    id, name, score,
    RANK() OVER (ORDER BY score DESC) AS rank
FROM players;

-- CTE for readability
WITH active_users AS (
    SELECT id, email FROM users WHERE status = 'active'
)
SELECT au.email, COUNT(o.id) AS order_count
FROM active_users au
LEFT JOIN orders o ON o.user_id = au.id
GROUP BY au.email;
```

### SQL Type Mapping

| Concept | PostgreSQL | MySQL | SQLite | SQL Server |
|---------|------------|-------|--------|------------|
| Auto ID | `BIGINT GENERATED ALWAYS AS IDENTITY` | `BIGINT AUTO_INCREMENT` | `INTEGER PRIMARY KEY` | `BIGINT IDENTITY` |
| UUID | `UUID DEFAULT gen_random_uuid()` | `CHAR(36) / BINARY(16)` | `TEXT` | `UNIQUEIDENTIFIER` |
| Boolean | `BOOLEAN` | `TINYINT(1)` | `INTEGER` | `BIT` |
| JSON | `JSONB` | `JSON` | `TEXT` | `NVARCHAR(MAX)` |
| Timestamp+TZ | `TIMESTAMPTZ` | `DATETIME` | `TEXT` | `DATETIMEOFFSET` |
| Money | `NUMERIC(12,2)` | `DECIMAL(12,2)` | `REAL` | `DECIMAL(12,2)` |
| Text | `TEXT` | `TEXT/LONGTEXT` | `TEXT` | `NVARCHAR(MAX)` |

---

## 15. Migration Tools Comparison (MANDATORY)

### A. Tool Selection Matrix

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                           MIGRATION TOOL COMPARISON                                       │
├──────────────────┬──────────────┬─────────────┬─────────────┬─────────────┬─────────────┤
│ Feature          │ golang-migrate│ Atlas       │ Flyway      │ Alembic     │ Liquibase   │
├──────────────────┼──────────────┼─────────────┼─────────────┼─────────────┼─────────────┤
│ Language         │ Go (CLI)     │ Go (CLI)    │ Java        │ Python      │ Java        │
│ Declarative      │ ✗            │ ✓           │ ✗           │ ✗           │ ✓           │
│ Imperative       │ ✓            │ ✓           │ ✓           │ ✓           │ ✓           │
│ Schema Diff      │ ✗            │ ✓           │ ✗           │ ✓ (autogen) │ ✓           │
│ Dry Run          │ ✗            │ ✓           │ ✗           │ ✗           │ ✓           │
│ Rollback         │ ✓ (manual)   │ ✓ (manual)  │ ✓ (undo)    │ ✓           │ ✓           │
│ Version Control  │ Filesystem   │ Filesystem  │ DB table    │ DB table    │ DB table    │
│ Multi-DB         │ ✓ (15+)      │ ✓ (10+)     │ ✓ (20+)     │ ✓ (dialects)│ ✓ (20+)     │
│ CI/CD Native     │ ✓            │ ✓           │ ✓           │ ✓           │ ✓           │
│ Cloud Service    │ ✗            │ Atlas Cloud │ Flyway Teams│ ✗           │ Liquibase Hub│
│ License          │ MIT          │ Apache 2.0  │ Apache 2.0  │ MIT         │ Apache 2.0  │
├──────────────────┼──────────────┼─────────────┼─────────────┼─────────────┼─────────────┤
│ Best For         │ Simple,      │ Modern,     │ Enterprise  │ Python/     │ Enterprise  │
│                  │ lightweight  │ declarative │ Java        │ SQLAlchemy  │ complex     │
└──────────────────┴──────────────┴─────────────┴─────────────┴─────────────┴─────────────┘
```

### B. Tool-Specific Configurations

#### golang-migrate (Recommended for Go/General)

```bash
# Installation
go install -tags 'postgres mysql sqlite3' \
    github.com/golang-migrate/migrate/v4/cmd/migrate@latest

# Or download binary
curl -L https://github.com/golang-migrate/migrate/releases/download/v4.17.0/migrate.linux-amd64.tar.gz | tar xz
```

```bash
# Create migration
migrate create -ext sql -dir db/migrations -seq create_users

# Run migrations
migrate -path db/migrations -database "$DATABASE_URL" up
migrate -path db/migrations -database "$DATABASE_URL" up 1  # One step

# Rollback
migrate -path db/migrations -database "$DATABASE_URL" down 1
migrate -path db/migrations -database "$DATABASE_URL" down -all

# Force version (fix dirty state)
migrate -path db/migrations -database "$DATABASE_URL" force 20240115

# Check version
migrate -path db/migrations -database "$DATABASE_URL" version
```

#### Atlas (Recommended for Declarative)

```hcl
# atlas.hcl - Configuration file
env "local" {
  src = "file://schema.sql"
  url = "postgres://localhost:5432/myapp?sslmode=disable"
  dev = "docker://postgres/16/dev"

  migration {
    dir = "file://migrations"
  }
}

env "prod" {
  url = getenv("DATABASE_URL")
  migration {
    dir = "file://migrations"
  }
}
```

```sql
-- schema.sql - Declarative schema (source of truth)
CREATE TABLE users (
    id BIGINT PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
    email VARCHAR(255) NOT NULL UNIQUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE orders (
    id BIGINT PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
    user_id BIGINT NOT NULL REFERENCES users(id),
    total NUMERIC(12,2) NOT NULL
);

CREATE INDEX idx_orders_user_id ON orders(user_id);
```

```bash
# Generate migration from schema diff
atlas migrate diff create_orders \
  --env local \
  --to file://schema.sql

# Apply migrations
atlas migrate apply --env prod

# Schema inspection
atlas schema inspect --url "$DATABASE_URL"

# Lint migrations for issues
atlas migrate lint --env local --latest 1

# Dry run (show SQL without executing)
atlas migrate apply --env prod --dry-run
```

#### Flyway (Recommended for Java/Enterprise)

```properties
# flyway.conf
flyway.url=jdbc:postgresql://localhost:5432/myapp
flyway.user=app
flyway.password=${FLYWAY_PASSWORD}
flyway.locations=filesystem:./migrations
flyway.validateOnMigrate=true
flyway.outOfOrder=false
flyway.baselineOnMigrate=true
```

```bash
# Run migrations
flyway migrate

# Show status
flyway info

# Validate migrations
flyway validate

# Repair checksum issues
flyway repair

# Undo last migration (Teams edition)
flyway undo
```

#### Alembic (Recommended for Python/SQLAlchemy)

```ini
# alembic.ini
[alembic]
script_location = alembic
sqlalchemy.url = driver://user:pass@localhost/dbname

[post_write_hooks]
hooks = black
black.type = console_scripts
black.entrypoint = black
```

```python
# alembic/env.py
from myapp.models import Base
target_metadata = Base.metadata

def run_migrations_online():
    connectable = engine_from_config(config.get_section(config.config_ini_section))
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,  # Detect type changes
            compare_server_default=True,
        )
        with context.begin_transaction():
            context.run_migrations()
```

```bash
# Auto-generate migration from model changes
alembic revision --autogenerate -m "Add users table"

# Run migrations
alembic upgrade head
alembic upgrade +1  # One step

# Rollback
alembic downgrade -1
alembic downgrade base

# Show history
alembic history --verbose
```

### C. Migration Versioning Strategies

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    VERSIONING STRATEGIES                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  Sequential (RECOMMENDED for most cases)                                      │
│  ├── Format: NNNNNN_description.sql                                          │
│  ├── Example: 000001_create_users.sql, 000002_add_orders.sql                 │
│  ├── Pros: Clear order, easy to understand                                   │
│  └── Cons: Merge conflicts in team environments                              │
│                                                                               │
│  Timestamp-Based                                                              │
│  ├── Format: YYYYMMDDHHMMSS_description.sql                                  │
│  ├── Example: 20240115143022_create_users.sql                                │
│  ├── Pros: No conflicts, globally unique                                     │
│  └── Cons: Harder to see order at glance                                     │
│                                                                               │
│  Branch-Based (for feature branches)                                          │
│  ├── Format: YYYYMMDD_branch_description.sql                                 │
│  ├── Example: 20240115_feature_auth_create_users.sql                         │
│  └── Merge to sequential on main branch                                      │
│                                                                               │
│  Hybrid (RECOMMENDED for teams)                                               │
│  ├── Development: Timestamp-based                                            │
│  ├── Release: Squash to sequential                                           │
│  └── Production: Sequential versions only                                    │
│                                                                               │
└─────────────────────────────────────────────────────────────────────────────┘
```

### D. Zero-Downtime Migration Patterns

```sql
-- ============================================================
-- EXPAND-CONTRACT PATTERN
-- Safe migrations without downtime
-- ============================================================

-- Phase 1: EXPAND (add new, keep old)
-- Deploy migration, then deploy new code

-- Migration 1: Add new column
ALTER TABLE users ADD COLUMN display_name VARCHAR(200);

-- Migration 2: Backfill data
UPDATE users SET display_name = first_name || ' ' || last_name
WHERE display_name IS NULL;

-- Migration 3: Add trigger to sync during transition
CREATE FUNCTION sync_display_name() RETURNS TRIGGER AS $$
BEGIN
    NEW.display_name := NEW.first_name || ' ' || NEW.last_name;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER tr_users_sync_display_name
    BEFORE INSERT OR UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION sync_display_name();

-- Phase 2: MIGRATE (update all code to use new column)
-- Deploy code that reads/writes display_name

-- Phase 3: CONTRACT (remove old)
-- After all code deployed and verified

-- Migration 4: Drop trigger
DROP TRIGGER tr_users_sync_display_name ON users;
DROP FUNCTION sync_display_name();

-- Migration 5: Drop old columns (optional, do later)
ALTER TABLE users DROP COLUMN first_name;
ALTER TABLE users DROP COLUMN last_name;
```

---

## 16. OLTP vs OLAP Design Patterns (MANDATORY)

### A. Workload Characteristics

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    OLTP vs OLAP COMPARISON                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  OLTP (Online Transaction Processing)                                         │
│  ├── Purpose: Day-to-day operations, transactions                            │
│  ├── Queries: Simple, short, frequent                                        │
│  ├── Data: Current state, normalized (3NF+)                                  │
│  ├── Users: Many concurrent, application-driven                              │
│  ├── Optimization: Write performance, low latency                            │
│  ├── Examples: E-commerce orders, banking transactions                       │
│  └── Databases: PostgreSQL, MySQL, SQL Server                                │
│                                                                               │
│  OLAP (Online Analytical Processing)                                          │
│  ├── Purpose: Analysis, reporting, business intelligence                     │
│  ├── Queries: Complex, long-running, aggregations                            │
│  ├── Data: Historical, denormalized (star/snowflake schema)                  │
│  ├── Users: Few analysts, query-driven                                       │
│  ├── Optimization: Read performance, throughput                              │
│  ├── Examples: Sales analytics, customer behavior analysis                   │
│  └── Databases: ClickHouse, Snowflake, BigQuery, DuckDB, Redshift           │
│                                                                               │
│  HTAP (Hybrid Transactional/Analytical Processing)                            │
│  ├── Purpose: Both OLTP and OLAP in single system                            │
│  ├── Trade-offs: Complexity, cost, neither fully optimized                   │
│  └── Databases: TiDB, CockroachDB, SingleStore, AlloyDB                      │
│                                                                               │
└─────────────────────────────────────────────────────────────────────────────┘
```

### B. OLTP Best Practices

```sql
-- ============================================================
-- OLTP SCHEMA DESIGN
-- Optimized for transactions, normalized, write-heavy
-- ============================================================

-- Normalized schema (3NF)
CREATE TABLE customers (
    id BIGINT PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
    email VARCHAR(255) NOT NULL UNIQUE,
    name VARCHAR(200) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE products (
    id BIGINT PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
    sku VARCHAR(50) NOT NULL UNIQUE,
    name VARCHAR(200) NOT NULL,
    price NUMERIC(12,2) NOT NULL,
    stock_quantity INTEGER NOT NULL DEFAULT 0,
    CHECK (price >= 0),
    CHECK (stock_quantity >= 0)
);

CREATE TABLE orders (
    id BIGINT PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
    customer_id BIGINT NOT NULL REFERENCES customers(id),
    status VARCHAR(20) NOT NULL DEFAULT 'pending',
    total_amount NUMERIC(12,2) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE order_items (
    id BIGINT PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
    order_id BIGINT NOT NULL REFERENCES orders(id) ON DELETE CASCADE,
    product_id BIGINT NOT NULL REFERENCES products(id),
    quantity INTEGER NOT NULL,
    unit_price NUMERIC(12,2) NOT NULL,
    CHECK (quantity > 0)
);

-- OLTP Indexes: Focused on lookup and foreign keys
CREATE INDEX idx_orders_customer_id ON orders(customer_id);
CREATE INDEX idx_orders_status ON orders(status) WHERE status != 'delivered';
CREATE INDEX idx_order_items_order_id ON order_items(order_id);
CREATE INDEX idx_order_items_product_id ON order_items(product_id);

-- OLTP Query Patterns
-- Short, indexed lookups
SELECT * FROM orders WHERE id = $1;
SELECT * FROM orders WHERE customer_id = $1 ORDER BY created_at DESC LIMIT 10;

-- Transaction-safe inventory update
BEGIN;
SELECT stock_quantity FROM products WHERE id = $1 FOR UPDATE;
UPDATE products SET stock_quantity = stock_quantity - $2 WHERE id = $1;
INSERT INTO order_items (...) VALUES (...);
COMMIT;
```

### C. OLAP Best Practices

```sql
-- ============================================================
-- OLAP SCHEMA DESIGN (Star Schema)
-- Optimized for analytics, denormalized, read-heavy
-- ============================================================

-- Dimension Tables (denormalized, slowly changing)
CREATE TABLE dim_customer (
    customer_key BIGINT PRIMARY KEY,  -- Surrogate key
    customer_id BIGINT NOT NULL,      -- Natural key from OLTP
    email VARCHAR(255) NOT NULL,
    name VARCHAR(200) NOT NULL,
    segment VARCHAR(50),              -- Derived attribute
    lifetime_value NUMERIC(12,2),     -- Pre-computed
    first_order_date DATE,
    -- SCD Type 2 fields
    valid_from DATE NOT NULL,
    valid_to DATE,
    is_current BOOLEAN NOT NULL DEFAULT true
);

CREATE TABLE dim_product (
    product_key BIGINT PRIMARY KEY,
    product_id BIGINT NOT NULL,
    sku VARCHAR(50) NOT NULL,
    name VARCHAR(200) NOT NULL,
    category VARCHAR(100),
    subcategory VARCHAR(100),
    brand VARCHAR(100),
    valid_from DATE NOT NULL,
    valid_to DATE,
    is_current BOOLEAN NOT NULL DEFAULT true
);

CREATE TABLE dim_date (
    date_key INTEGER PRIMARY KEY,  -- YYYYMMDD format
    full_date DATE NOT NULL UNIQUE,
    year INTEGER NOT NULL,
    quarter INTEGER NOT NULL,
    month INTEGER NOT NULL,
    month_name VARCHAR(20) NOT NULL,
    week INTEGER NOT NULL,
    day_of_week INTEGER NOT NULL,
    day_name VARCHAR(20) NOT NULL,
    is_weekend BOOLEAN NOT NULL,
    is_holiday BOOLEAN NOT NULL DEFAULT false,
    fiscal_year INTEGER,
    fiscal_quarter INTEGER
);

CREATE TABLE dim_time (
    time_key INTEGER PRIMARY KEY,  -- HHMMSS format
    full_time TIME NOT NULL,
    hour INTEGER NOT NULL,
    minute INTEGER NOT NULL,
    second INTEGER NOT NULL,
    am_pm VARCHAR(2) NOT NULL,
    hour_12 INTEGER NOT NULL
);

-- Fact Table (transactions, measures)
CREATE TABLE fact_sales (
    sale_key BIGINT PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
    -- Dimension keys
    date_key INTEGER NOT NULL REFERENCES dim_date(date_key),
    time_key INTEGER NOT NULL REFERENCES dim_time(time_key),
    customer_key BIGINT NOT NULL REFERENCES dim_customer(customer_key),
    product_key BIGINT NOT NULL REFERENCES dim_product(product_key),
    -- Degenerate dimensions (no separate table needed)
    order_id BIGINT NOT NULL,
    order_item_id BIGINT NOT NULL,
    -- Measures
    quantity INTEGER NOT NULL,
    unit_price NUMERIC(12,2) NOT NULL,
    discount_amount NUMERIC(12,2) NOT NULL DEFAULT 0,
    tax_amount NUMERIC(12,2) NOT NULL DEFAULT 0,
    total_amount NUMERIC(12,2) NOT NULL,
    cost_amount NUMERIC(12,2),  -- For margin calculations
    -- Audit
    etl_loaded_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- OLAP Indexes: Focused on aggregation and filtering
-- Bitmap indexes (if supported) for low-cardinality dimensions
CREATE INDEX idx_fact_sales_date ON fact_sales(date_key);
CREATE INDEX idx_fact_sales_customer ON fact_sales(customer_key);
CREATE INDEX idx_fact_sales_product ON fact_sales(product_key);

-- Composite for common query patterns
CREATE INDEX idx_fact_sales_date_product ON fact_sales(date_key, product_key);

-- OLAP Query Patterns
-- Aggregations across dimensions
SELECT
    d.year,
    d.quarter,
    p.category,
    SUM(f.total_amount) AS revenue,
    SUM(f.quantity) AS units_sold,
    COUNT(DISTINCT f.customer_key) AS unique_customers,
    SUM(f.total_amount) / COUNT(DISTINCT f.order_id) AS avg_order_value
FROM fact_sales f
JOIN dim_date d ON d.date_key = f.date_key
JOIN dim_product p ON p.product_key = f.product_key
WHERE d.year = 2024
GROUP BY d.year, d.quarter, p.category
ORDER BY d.year, d.quarter, revenue DESC;
```

### D. ETL Pipeline Pattern

```sql
-- ============================================================
-- ETL: OLTP to OLAP Data Pipeline
-- ============================================================

-- Step 1: Extract from OLTP (incremental)
CREATE TABLE etl_checkpoint (
    table_name VARCHAR(100) PRIMARY KEY,
    last_extracted_at TIMESTAMPTZ NOT NULL,
    last_extracted_id BIGINT
);

-- Extract new/changed records
SELECT *
FROM oltp.orders
WHERE updated_at > (SELECT last_extracted_at FROM etl_checkpoint WHERE table_name = 'orders')
ORDER BY updated_at
LIMIT 10000;

-- Step 2: Transform and load to staging
INSERT INTO staging.orders
SELECT
    o.id,
    o.customer_id,
    TO_CHAR(o.created_at, 'YYYYMMDD')::INTEGER AS date_key,
    TO_CHAR(o.created_at, 'HH24MISS')::INTEGER AS time_key,
    o.total_amount,
    o.created_at AS source_created_at,
    NOW() AS etl_loaded_at
FROM extracted_orders o;

-- Step 3: Load to fact table with dimension lookups
INSERT INTO fact_sales (date_key, time_key, customer_key, product_key, ...)
SELECT
    s.date_key,
    s.time_key,
    dc.customer_key,
    dp.product_key,
    ..
FROM staging.orders s
JOIN dim_customer dc ON dc.customer_id = s.customer_id AND dc.is_current = true
JOIN dim_product dp ON dp.product_id = s.product_id AND dp.is_current = true;

-- Step 4: Update checkpoint
UPDATE etl_checkpoint
SET last_extracted_at = NOW(), last_extracted_id = (SELECT MAX(id) FROM extracted_orders)
WHERE table_name = 'orders';
```

---

## 17. Scalability Guidelines (MANDATORY)

### A. Scalability Patterns Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    SCALABILITY PATTERNS                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  Vertical Scaling (Scale Up)                                                  │
│  ├── Add more CPU, RAM, faster storage                                       │
│  ├── Simplest approach, limited ceiling                                      │
│  └── Best for: Initial growth, simpler operations                            │
│                                                                               │
│  Read Replicas (Scale Reads)                                                  │
│  ├── Primary handles writes, replicas handle reads                           │
│  ├── Async replication (eventual consistency)                                │
│  └── Best for: Read-heavy workloads (10:1+ read:write)                       │
│                                                                               │
│  Connection Pooling                                                           │
│  ├── Reduce connection overhead                                              │
│  ├── Tools: PgBouncer, ProxySQL, built-in pools                              │
│  └── Best for: Many short-lived connections                                  │
│                                                                               │
│  Caching Layer                                                                │
│  ├── Redis, Memcached for hot data                                           │
│  ├── Reduce database load significantly                                      │
│  └── Best for: Frequently accessed, slowly changing data                     │
│                                                                               │
│  Partitioning (Scale Storage)                                                 │
│  ├── Split large tables into smaller pieces                                  │
│  ├── Same database instance                                                  │
│  └── Best for: Large tables with natural partition key                       │
│                                                                               │
│  Sharding (Scale Everything)                                                  │
│  ├── Distribute data across multiple database instances                      │
│  ├── Complex, requires application changes                                   │
│  └── Best for: Extreme scale, multi-region                                   │
│                                                                               │
└─────────────────────────────────────────────────────────────────────────────┘
```

### B. Scalability Rules

```sql
-- ============================================================
-- RULE 1: Design for Horizontal Scalability from Day 1
-- ============================================================

-- Use UUIDs or distributed ID generators for primary keys
-- Allows data to be moved between shards without conflicts
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    -- Or use a distributed ID generator like Snowflake
    -- id BIGINT PRIMARY KEY DEFAULT next_snowflake_id(),
    ..
);

-- Always include tenant/shard key in queries
-- This enables future sharding without query changes
CREATE TABLE orders (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,  -- Shard key
    user_id UUID NOT NULL,
    ..
);

-- Index includes shard key
CREATE INDEX idx_orders_tenant_user ON orders(tenant_id, user_id);

-- Queries always filter by shard key
SELECT * FROM orders WHERE tenant_id = $1 AND user_id = $2;

-- ============================================================
-- RULE 2: Avoid Cross-Shard Operations
-- ============================================================

-- ❌ WRONG: Joins that might span shards
SELECT o.*, u.email
FROM orders o
JOIN users u ON u.id = o.user_id;

-- ✅ CORRECT: Denormalize or query separately
-- Option A: Denormalize user info into orders
CREATE TABLE orders (
    ..
    user_id UUID NOT NULL,
    user_email VARCHAR(255) NOT NULL,  -- Denormalized
    ..
);

-- Option B: Query in application, join in code
-- 1. SELECT * FROM orders WHERE tenant_id = ? AND id = ?
-- 2. SELECT email FROM users WHERE tenant_id = ? AND id = ?
-- 3. Combine in application

-- ============================================================
-- RULE 3: Use Monotonically Increasing Keys Carefully
-- ============================================================

-- ❌ PROBLEM: Auto-increment creates hotspots
-- All inserts go to the same "last" partition/shard
CREATE TABLE events (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    ..
);

-- ✅ BETTER: UUID or time-based distributed ID
CREATE TABLE events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    ..
);

-- ✅ BETTER: Composite key with shard distribution
CREATE TABLE events (
    shard_id INTEGER NOT NULL,
    event_id BIGINT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (shard_id, event_id)
);

-- ============================================================
-- RULE 4: Stateless Application Tier
-- ============================================================

-- Don't rely on database for session state
-- ❌ WRONG: Session table with sticky routing
CREATE TABLE sessions (
    id UUID PRIMARY KEY,
    user_id UUID NOT NULL,
    data JSONB NOT NULL
);

-- ✅ CORRECT: Use Redis/Memcached for sessions
-- Or use JWT tokens (stateless)
-- Database only stores persistent data

-- ============================================================
-- RULE 5: Batch Operations for Efficiency
-- ============================================================

-- ❌ SLOW: Individual inserts
FOR i IN 1..1000 LOOP
    INSERT INTO events (type, data) VALUES ('click', '{}');
END LOOP;

-- ✅ FAST: Batch insert
INSERT INTO events (type, data)
VALUES
    ('click', '{}'),
    ('click', '{}'),
    ...  -- Up to 1000 rows
;

-- ✅ FAST: COPY for bulk loads (PostgreSQL)
COPY events (type, data) FROM STDIN;
click	{}
click	{}
\.
```

### C. Connection Pooling Configuration

```ini
# PgBouncer configuration (pgbouncer.ini)
[databases]
myapp = host=127.0.0.1 port=5432 dbname=myapp

[pgbouncer]
listen_addr = 0.0.0.0
listen_port = 6432
auth_type = scram-sha-256
auth_file = /etc/pgbouncer/userlist.txt

# Pool mode:
# - session: Connection assigned for entire session (default)
# - transaction: Connection returned after transaction (RECOMMENDED)
# - statement: Connection returned after each statement (limited use)
pool_mode = transaction

# Pool sizing
default_pool_size = 20          # Connections per user/database pair
min_pool_size = 5               # Minimum connections to keep
reserve_pool_size = 5           # Extra connections for burst
max_client_conn = 1000          # Max client connections

# Timeouts
server_idle_timeout = 600       # Close idle server connections
client_idle_timeout = 0         # 0 = no timeout
query_timeout = 30              # Kill queries longer than this

# Safety
server_reset_query = DISCARD ALL
server_check_query = SELECT 1
server_check_delay = 30
```

---

## 18. Partitioning Strategies (MANDATORY)

### A. Partitioning Types

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PARTITIONING STRATEGIES                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  Range Partitioning                                                           │
│  ├── Partition by value ranges (dates, IDs)                                  │
│  ├── Best for: Time-series data, logs, events                                │
│  └── Example: One partition per month/year                                   │
│                                                                               │
│  List Partitioning                                                            │
│  ├── Partition by specific values                                            │
│  ├── Best for: Geographic regions, categories                                │
│  └── Example: One partition per country                                      │
│                                                                               │
│  Hash Partitioning                                                            │
│  ├── Partition by hash of column value                                       │
│  ├── Best for: Even distribution, no natural partition key                   │
│  └── Example: Partition by hash(user_id) % 16                                │
│                                                                               │
│  Composite Partitioning                                                       │
│  ├── Combine multiple strategies                                             │
│  ├── Best for: Complex requirements                                          │
│  └── Example: Range by date, then hash by user_id                            │
│                                                                               │
└─────────────────────────────────────────────────────────────────────────────┘
```

### B. PostgreSQL Partitioning

```sql
-- ============================================================
-- RANGE PARTITIONING (Time-Series Data)
-- ============================================================

-- Parent table (no data stored here)
CREATE TABLE events (
    id UUID DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    event_type VARCHAR(50) NOT NULL,
    payload JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
) PARTITION BY RANGE (created_at);

-- Create partitions for each month
CREATE TABLE events_2024_01 PARTITION OF events
    FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');

CREATE TABLE events_2024_02 PARTITION OF events
    FOR VALUES FROM ('2024-02-01') TO ('2024-03-01');

-- Add index to parent (automatically applied to partitions)
CREATE INDEX idx_events_tenant_created ON events (tenant_id, created_at);

-- Automatic partition creation (pg_partman extension or cron job)
-- CREATE EXTENSION pg_partman;
-- SELECT create_parent('public.events', 'created_at', 'native', 'monthly');

-- Drop old partitions (data retention)
DROP TABLE events_2023_01;

-- Detach before dropping (safer)
ALTER TABLE events DETACH PARTITION events_2023_01;
DROP TABLE events_2023_01;

-- ============================================================
-- LIST PARTITIONING (Geographic/Category)
-- ============================================================

CREATE TABLE orders (
    id UUID DEFAULT gen_random_uuid(),
    region VARCHAR(10) NOT NULL,
    customer_id UUID NOT NULL,
    total NUMERIC(12,2) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
) PARTITION BY LIST (region);

CREATE TABLE orders_us PARTITION OF orders FOR VALUES IN ('us-east', 'us-west');
CREATE TABLE orders_eu PARTITION OF orders FOR VALUES IN ('eu-west', 'eu-central');
CREATE TABLE orders_ap PARTITION OF orders FOR VALUES IN ('ap-south', 'ap-northeast');
CREATE TABLE orders_default PARTITION OF orders DEFAULT;  -- Catch-all

-- ============================================================
-- HASH PARTITIONING (Even Distribution)
-- ============================================================

CREATE TABLE sessions (
    id UUID DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL,
    data JSONB,
    expires_at TIMESTAMPTZ NOT NULL
) PARTITION BY HASH (user_id);

-- Create 16 partitions
CREATE TABLE sessions_p0 PARTITION OF sessions FOR VALUES WITH (MODULUS 16, REMAINDER 0);
CREATE TABLE sessions_p1 PARTITION OF sessions FOR VALUES WITH (MODULUS 16, REMAINDER 1);
-- ... repeat for p2 through p15

-- ============================================================
-- COMPOSITE PARTITIONING
-- ============================================================

-- First level: Range by date
CREATE TABLE logs (
    id UUID DEFAULT gen_random_uuid(),
    service VARCHAR(50) NOT NULL,
    level VARCHAR(10) NOT NULL,
    message TEXT,
    created_at TIMESTAMPTZ NOT NULL
) PARTITION BY RANGE (created_at);

-- Second level: List by service within each date range
CREATE TABLE logs_2024_01 PARTITION OF logs
    FOR VALUES FROM ('2024-01-01') TO ('2024-02-01')
    PARTITION BY LIST (service);

CREATE TABLE logs_2024_01_api PARTITION OF logs_2024_01
    FOR VALUES IN ('api', 'gateway');

CREATE TABLE logs_2024_01_worker PARTITION OF logs_2024_01
    FOR VALUES IN ('worker', 'scheduler');

CREATE TABLE logs_2024_01_default PARTITION OF logs_2024_01 DEFAULT;
```

### C. MySQL Partitioning

```sql
-- Range partitioning in MySQL
CREATE TABLE events (
    id BIGINT AUTO_INCREMENT,
    tenant_id BINARY(16) NOT NULL,
    event_type VARCHAR(50) NOT NULL,
    payload JSON,
    created_at DATETIME NOT NULL,
    PRIMARY KEY (id, created_at)  -- Partition key must be in PK
) PARTITION BY RANGE (YEAR(created_at) * 100 + MONTH(created_at)) (
    PARTITION p2024_01 VALUES LESS THAN (202402),
    PARTITION p2024_02 VALUES LESS THAN (202403),
    PARTITION p2024_03 VALUES LESS THAN (202404),
    PARTITION pmax VALUES LESS THAN MAXVALUE
);

-- Add partition
ALTER TABLE events ADD PARTITION (
    PARTITION p2024_04 VALUES LESS THAN (202405)
);

-- Drop partition
ALTER TABLE events DROP PARTITION p2023_01;

-- Reorganize partitions
ALTER TABLE events REORGANIZE PARTITION pmax INTO (
    PARTITION p2024_05 VALUES LESS THAN (202406),
    PARTITION pmax VALUES LESS THAN MAXVALUE
);
```

---

## 19. Sharding Strategies (MANDATORY)

### A. Sharding Patterns

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    SHARDING PATTERNS                                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  Application-Level Sharding                                                   │
│  ├── Application determines which shard to query                             │
│  ├── Most flexible, most complex                                             │
│  └── Best for: Custom logic, gradual migration                               │
│                                                                               │
│  Proxy-Based Sharding                                                         │
│  ├── Proxy layer routes queries to correct shard                             │
│  ├── Transparent to application (mostly)                                     │
│  ├── Tools: Vitess, ProxySQL, Citus                                          │
│  └── Best for: Existing applications, MySQL                                  │
│                                                                               │
│  Database-Native Sharding                                                     │
│  ├── Database handles distribution internally                                │
│  ├── Easiest to use, less control                                            │
│  ├── Examples: CockroachDB, YugabyteDB, TiDB, Citus                          │
│  └── Best for: New applications, NewSQL adoption                             │
│                                                                               │
│  SHARD KEY SELECTION (Critical Decision)                                      │
│  ├── High cardinality (many unique values)                                   │
│  ├── Even distribution (no hotspots)                                         │
│  ├── Query isolation (queries hit single shard)                              │
│  ├── Common choices: tenant_id, user_id, region                              │
│  └── Avoid: dates (hot partition), low cardinality                           │
│                                                                               │
└─────────────────────────────────────────────────────────────────────────────┘
```

### B. Application-Level Sharding

```python
# Python sharding router example

from hashlib import md5
from typing import List, Optional
import databases

class ShardRouter:
    def __init__(self, shard_urls: List[str]):
        self.shards = [databases.Database(url) for url in shard_urls]
        self.shard_count = len(self.shards)

    def get_shard(self, shard_key: str) -> databases.Database:
        """Consistent hashing to determine shard."""
        hash_value = int(md5(shard_key.encode()).hexdigest(), 16)
        shard_index = hash_value % self.shard_count
        return self.shards[shard_index]

    async def execute_on_shard(self, shard_key: str, query: str, values: dict):
        """Execute query on specific shard."""
        shard = self.get_shard(shard_key)
        return await shard.execute(query, values)

    async def execute_on_all_shards(self, query: str, values: dict):
        """Execute query on all shards (scatter-gather)."""
        results = []
        for shard in self.shards:
            result = await shard.fetch_all(query, values)
            results.extend(result)
        return results

# Usage
router = ShardRouter([
    "postgresql://shard1.example.com/db",
    "postgresql://shard2.example.com/db",
    "postgresql://shard3.example.com/db",
])

# Query single shard
await router.execute_on_shard(
    shard_key=tenant_id,
    query="SELECT * FROM orders WHERE tenant_id = :tenant_id",
    values={"tenant_id": tenant_id}
)

# Query all shards (expensive, avoid if possible)
await router.execute_on_all_shards(
    query="SELECT COUNT(*) FROM orders WHERE created_at > :date",
    values={"date": "2024-01-01"}
)
```

### C. Schema Design for Sharding

```sql
-- ============================================================
-- SHARDING-READY SCHEMA DESIGN
-- ============================================================

-- Global tables (replicated to all shards)
-- Small, rarely updated reference data
CREATE TABLE countries (
    code CHAR(2) PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    region VARCHAR(50) NOT NULL
);

CREATE TABLE product_categories (
    id INTEGER PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    parent_id INTEGER REFERENCES product_categories(id)
);

-- Sharded tables (distributed by tenant_id)
CREATE TABLE tenants (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(200) NOT NULL,
    plan VARCHAR(50) NOT NULL DEFAULT 'free',
    -- Shard assignment (for explicit routing if needed)
    shard_id INTEGER NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,  -- SHARD KEY
    email VARCHAR(255) NOT NULL,
    name VARCHAR(200) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    -- Unique within tenant, not globally
    UNIQUE (tenant_id, email)
);

CREATE TABLE orders (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,  -- SHARD KEY
    user_id UUID NOT NULL,
    -- Don't use FK to users if cross-shard possible
    -- REFERENCES users(id) -- REMOVED
    status VARCHAR(20) NOT NULL DEFAULT 'pending',
    total NUMERIC(12,2) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- All indexes include shard key first
CREATE INDEX idx_users_tenant_email ON users(tenant_id, email);
CREATE INDEX idx_orders_tenant_user ON orders(tenant_id, user_id);
CREATE INDEX idx_orders_tenant_status ON orders(tenant_id, status);

-- ============================================================
-- CROSS-SHARD REFERENCE PATTERN
-- ============================================================

-- Instead of foreign keys across shards, use composite references
CREATE TABLE order_items (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,  -- SHARD KEY
    order_id UUID NOT NULL,
    -- Product might be on different shard or global catalog
    product_id UUID NOT NULL,
    product_sku VARCHAR(50) NOT NULL,   -- Denormalized
    product_name VARCHAR(200) NOT NULL,  -- Denormalized
    quantity INTEGER NOT NULL,
    unit_price NUMERIC(12,2) NOT NULL
);

-- Application enforces consistency, not database FK
```

---

## 20. Replication Patterns (MANDATORY)

### A. Replication Topologies

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    REPLICATION TOPOLOGIES                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  Primary-Replica (Master-Slave)                                               │
│  ├── One writable primary, multiple read replicas                            │
│  ├── Async or sync replication                                               │
│  ├── Simple, well-understood                                                 │
│  └── Best for: Read scaling, high availability                               │
│                                                                               │
│       [Primary] ──writes──▶                                                   │
│           │                                                                   │
│           ├───async replication───▶ [Replica 1] ──reads──▶                   │
│           ├───async replication───▶ [Replica 2] ──reads──▶                   │
│           └───sync replication────▶ [Replica 3] (failover)                   │
│                                                                               │
│  Multi-Primary (Master-Master)                                                │
│  ├── Multiple writable nodes                                                 │
│  ├── Conflict resolution required                                            │
│  ├── Complex, use with caution                                               │
│  └── Best for: Geo-distributed writes, high availability                     │
│                                                                               │
│       [Primary A] ◀──bidirectional──▶ [Primary B]                            │
│           │                               │                                   │
│           ▼                               ▼                                   │
│       [Replica A1]                    [Replica B1]                           │
│                                                                               │
│  Chain Replication                                                            │
│  ├── Replicas cascade from each other                                        │
│  ├── Reduces primary load                                                    │
│  └── Best for: Many replicas, cross-region                                   │
│                                                                               │
│       [Primary] ──▶ [Replica 1] ──▶ [Replica 2] ──▶ [Replica 3]             │
│                                                                               │
└─────────────────────────────────────────────────────────────────────────────┘
```

### B. Multi-Primary Replication Design

```sql
-- ============================================================
-- MULTI-PRIMARY SAFE SCHEMA DESIGN
-- Avoid conflicts by design
-- ============================================================

-- RULE 1: Use UUIDs for primary keys (no auto-increment conflicts)
CREATE TABLE orders (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    ..
);

-- RULE 2: Include origin/region in composite keys
CREATE TABLE orders (
    id UUID DEFAULT gen_random_uuid(),
    origin_region VARCHAR(20) NOT NULL,
    PRIMARY KEY (origin_region, id),
    ..
);

-- RULE 3: Use append-only patterns where possible
-- Instead of UPDATE, INSERT new versions
CREATE TABLE inventory_changes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    product_id UUID NOT NULL,
    change_type VARCHAR(20) NOT NULL,  -- 'add', 'remove', 'adjust'
    quantity INTEGER NOT NULL,
    origin_region VARCHAR(20) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Current inventory = SUM of all changes
CREATE VIEW current_inventory AS
SELECT
    product_id,
    SUM(CASE change_type WHEN 'add' THEN quantity ELSE -quantity END) AS quantity
FROM inventory_changes
GROUP BY product_id;

-- RULE 4: Use CRDTs (Conflict-free Replicated Data Types)
-- Last-Write-Wins with vector clock
CREATE TABLE user_settings (
    user_id UUID NOT NULL,
    setting_key VARCHAR(100) NOT NULL,
    setting_value TEXT,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    origin_region VARCHAR(20) NOT NULL,
    vector_clock JSONB NOT NULL DEFAULT '{}',
    PRIMARY KEY (user_id, setting_key)
);

-- Conflict resolution: latest timestamp wins, origin_region as tiebreaker
-- Application must merge vector clocks

-- RULE 5: Avoid foreign keys across regions
-- ❌ WRONG: FK might reference non-replicated data
FOREIGN KEY (user_id) REFERENCES users(id)

-- ✅ CORRECT: Application-level consistency
-- Store enough denormalized data to work independently
CREATE TABLE orders (
    id UUID PRIMARY KEY,
    user_id UUID NOT NULL,
    user_email VARCHAR(255) NOT NULL,  -- Denormalized
    user_name VARCHAR(200) NOT NULL,   -- Denormalized
    ..
);

-- ============================================================
-- CONFLICT DETECTION AND RESOLUTION
-- ============================================================

-- Conflict log table
CREATE TABLE replication_conflicts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    table_name VARCHAR(100) NOT NULL,
    record_id TEXT NOT NULL,
    local_data JSONB NOT NULL,
    remote_data JSONB NOT NULL,
    resolution VARCHAR(50),  -- 'local_wins', 'remote_wins', 'merged', 'manual'
    resolved_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Conflict resolution strategies:
-- 1. Last-Write-Wins (LWW): Based on timestamp
-- 2. First-Write-Wins: Based on timestamp (opposite)
-- 3. Origin-Wins: Specific region takes precedence
-- 4. Merge: Combine non-conflicting fields
-- 5. Manual: Flag for human review
```

### C. Geo-Replication Configuration

```sql
-- ============================================================
-- GEO-REPLICATION PATTERNS
-- ============================================================

-- Region-aware schema
CREATE TABLE data_regions (
    code VARCHAR(20) PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    primary_endpoint VARCHAR(255) NOT NULL,
    replica_endpoints TEXT[] NOT NULL
);

INSERT INTO data_regions VALUES
    ('us-east', 'US East', 'us-east.db.example.com', ARRAY['us-west.db.example.com']),
    ('eu-west', 'EU West', 'eu-west.db.example.com', ARRAY['eu-central.db.example.com']),
    ('ap-south', 'Asia Pacific', 'ap-south.db.example.com', ARRAY['ap-northeast.db.example.com']);

-- User region assignment
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) NOT NULL,
    data_region VARCHAR(20) NOT NULL REFERENCES data_regions(code),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Data residency enforcement
-- Data stays in assigned region, replicated within region only
CREATE TABLE user_data (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id),
    data_region VARCHAR(20) NOT NULL,  -- Denormalized for partition routing
    content BYTEA NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
) PARTITION BY LIST (data_region);

CREATE TABLE user_data_us PARTITION OF user_data FOR VALUES IN ('us-east', 'us-west');
CREATE TABLE user_data_eu PARTITION OF user_data FOR VALUES IN ('eu-west', 'eu-central');
CREATE TABLE user_data_ap PARTITION OF user_data FOR VALUES IN ('ap-south', 'ap-northeast');
```

### D. PostgreSQL Streaming Replication

```bash
# Primary configuration (postgresql.conf)
wal_level = replica
max_wal_senders = 10
wal_keep_size = 1GB
synchronous_commit = on  # For sync replica
synchronous_standby_names = 'replica1'  # Sync replica name

# Primary pg_hba.conf
host replication replicator 10.0.0.0/8 scram-sha-256

# Replica setup
pg_basebackup -h primary.example.com -D /var/lib/postgresql/data -U replicator -P -R

# Replica postgresql.conf
primary_conninfo = 'host=primary.example.com port=5432 user=replicator password=xxx'
hot_standby = on
```

```sql
-- Monitor replication status (on primary)
SELECT
    client_addr,
    state,
    sent_lsn,
    write_lsn,
    flush_lsn,
    replay_lsn,
    sync_state,
    pg_wal_lsn_diff(sent_lsn, replay_lsn) AS replication_lag_bytes
FROM pg_stat_replication;

-- Monitor on replica
SELECT
    pg_is_in_recovery() AS is_replica,
    pg_last_wal_receive_lsn() AS received_lsn,
    pg_last_wal_replay_lsn() AS replayed_lsn,
    pg_last_xact_replay_timestamp() AS last_replay_time,
    NOW() - pg_last_xact_replay_timestamp() AS replication_lag;
```

---

## 21. Why This Configuration Works

**Data Integrity First:**
- Constraints at database level catch bugs that slip through application code
- Foreign keys maintain referential integrity automatically
- Check constraints enforce business rules at lowest level

**Type Safety:**
- Type-safe query generators (sqlc) eliminate runtime SQL errors
- Compile-time verification catches typos and type mismatches
- SQL remains the source of truth, not ORM mappings

**Performance by Design:**
- Indexing strategy based on actual query patterns
- Query plan analysis prevents production surprises
- Normalization provides flexibility; denormalization is evidence-based

**Security by Default:**
- Parameterized queries prevent SQL injection by design
- Least privilege access limits blast radius
- Audit logging provides accountability

**Reliable Migrations:**
- Version-controlled schema changes
- Tested up and down migrations
- Backward-compatible deployment patterns

**Scalability Ready:**
- UUID primary keys enable future sharding
- Tenant-aware schema design from the start
- Partitioning strategies for large tables
- Replication-safe patterns avoid conflicts

---

## 22. References

### Companion Guides
- [sqlc.md](./sqlc.md) - **RECOMMENDED**: Type-safe SQL with sqlc
- [postgresql.md](./postgresql.md) - PostgreSQL-specific guidelines
- [secure-coding.md](./secure-coding.md) - Security best practices
- [testing.md](./testing.md) - Testing strategies
- [ci-cd.md](./ci-cd.md) - CI/CD pipeline integration

### Migration Tools
| Tool | Language | Type | Best For |
|------|----------|------|----------|
| [golang-migrate](https://github.com/golang-migrate/migrate) | Go | Imperative | Simple, multi-DB |
| [Atlas](https://atlasgo.io/) | Go | Declarative | Modern, schema-diff |
| [Flyway](https://flywaydb.org/) | Java | Imperative | Enterprise Java |
| [Alembic](https://alembic.sqlalchemy.org/) | Python | Imperative | SQLAlchemy projects |
| [Liquibase](https://www.liquibase.org/) | Java | Both | Enterprise, multi-DB |
| [Sqitch](https://sqitch.org/) | Perl | Imperative | Database-native SQL |
| [dbmate](https://github.com/amacneil/dbmate) | Go | Imperative | Lightweight, multi-DB |

### Query Generators
- [sqlc](https://sqlc.dev/) - Go, Python, TypeScript, Kotlin
- [jOOQ](https://www.jooq.org/) - Java
- [Prisma](https://www.prisma.io/) - TypeScript/JavaScript
- [Diesel](https://diesel.rs/) - Rust

### Distributed Databases
| Database | Type | Sharding | Replication | Best For |
|----------|------|----------|-------------|----------|
| [CockroachDB](https://www.cockroachlabs.com/) | NewSQL | Native | Multi-active | Global distribution |
| [YugabyteDB](https://www.yugabyte.com/) | NewSQL | Native | Multi-active | PostgreSQL compatible |
| [TiDB](https://pingcap.com/tidb/) | NewSQL | Native | Multi-active | MySQL compatible |
| [Citus](https://www.citusdata.com/) | PostgreSQL ext | Native | Primary-replica | PostgreSQL sharding |
| [Vitess](https://vitess.io/) | Proxy | Application | Primary-replica | MySQL sharding |

### OLAP Databases
| Database | Type | Best For |
|----------|------|----------|
| [ClickHouse](https://clickhouse.com/) | Column-store | Real-time analytics |
| [DuckDB](https://duckdb.org/) | Embedded | Local analytics, OLAP |
| [Apache Druid](https://druid.apache.org/) | Column-store | Real-time + batch |
| [Snowflake](https://www.snowflake.com/) | Cloud DW | Enterprise analytics |
| [BigQuery](https://cloud.google.com/bigquery) | Cloud DW | GCP analytics |
| [Redshift](https://aws.amazon.com/redshift/) | Cloud DW | AWS analytics |

### Documentation
- [Use The Index, Luke](https://use-the-index-luke.com/) - Index optimization guide
- [PostgreSQL Documentation](https://www.postgresql.org/docs/)
- [MySQL Documentation](https://dev.mysql.com/doc/)
- [SQLite Documentation](https://www.sqlite.org/docs.html)
- [Designing Data-Intensive Applications](https://dataintensive.net/) - Essential reading

---

**Last Updated:** 2026-02-06
**Version:** 2.1
**Maintainer:** Database Architecture Team


**End of SQL Development Guidelines**
