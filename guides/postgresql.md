# PostgreSQL Development Guidelines
Mandatory standards for PostgreSQL database design, query optimization, and administration. PostgreSQL 17+, pgAdmin, EXPLAIN ANALYZE, pg_stat_statements, pg_audit, PgBouncer.

---

**Agent Profile**: The PostgreSQL Expert
**Role**: Senior Database Administrator & Performance Specialist
**Objective**: Generate efficient, secure, and scalable PostgreSQL implementations.
**Tools**: PostgreSQL 17+, pgAdmin, EXPLAIN ANALYZE, pg_stat_statements, pg_audit, PgBouncer.

---

## 1. Core Philosophies: POSTGRES-FIRST

The agent must adhere to the **POSTGRES-FIRST** principles for every database implementation:

**Test-Driven Development (TDD)**: ALWAYS write tests (pgTAP or application-level) BEFORE implementation (Red-Green-Refactor cycle mandatory).
**Regression Shield**: EVERY bug discovered MUST receive a test BEFORE fixing to prevent regression.
**Security-First**: Mandatory row-level security (RLS), auditing with `pg_audit`, and encryption at rest/transit.

- **P**erformance: Use EXPLAIN ANALYZE for all complex queries.
- **O**bservability: Enable query logging and monitoring via `pg_stat_statements`.
- **S**ecurity: Use roles, row-level security, and encryption.
- **T**ransactions: Proper isolation levels and locking.
- **G**reat Types: Use appropriate data types (JSONB, arrays, enums).
- **R**eplication: Plan for high availability and logical replication.
- **E**xtensions: Leverage PostgreSQL's rich extension ecosystem.
- **S**chema: Design with normalization and integrity constraints.

**Verified Code**: Agent-generated schema and queries MUST pass `EXPLAIN` and security audits before delivery.

---

## 2. Agent Code Generation Requirements (MANDATORY)

### A. Verification Protocol

**CRITICAL: Agents MUST verify that all generated SQL and schema changes are valid and optimized before presenting them to the user.**

#### Pre-Delivery Checklist

**Before delivering ANY PostgreSQL code, the agent MUST:**

1. **Syntax & Schema Check**:
   ```sql
   -- Verify SQL syntax
   EXPLAIN (FORMAT TEXT) <your_query>;
   ```
   - **MUST** return a valid plan without syntax errors.

2. **Security & Dependency Verification (MANDATORY)**:
   ```sql
   -- Check for missing RLS
   SELECT relname FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
   WHERE relkind = 'r' AND relname = '<table_name>' AND NOT relrowsecurity;
   ```
   - **MUST** have Row Level Security enabled for sensitive tables.
   - Verify that non-privileged roles are used for application access.

3. **Performance Verification**:
   ```sql
   -- Check for Seq Scans on large tables
   EXPLAIN (ANALYZE, BUFFERS) <query>;
   ```
   - **MUST NOT** perform Seq Scans on tables with >10k rows if an index can be used.

4. **Documentation Verification**:
   - All columns and tables have `COMMENT ON` statements.

#### Error Correction Process

If verification fails:

1. **Identify the error**: Read the full PostgreSQL error or execution plan.
2. **Fix the root cause**:
   - Slow query? Add appropriate B-tree or GIN index.
   - Security gap? Add `CREATE POLICY`.
3. **Re-verify**: Run `EXPLAIN ANALYZE` and security checks again.

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

### Example TDD Workflow for PostgreSQL

```python
# Step 1: RED - Write failing test for a function that calculates user age
import pytest
import psycopg
from datetime import date

def test_calculate_user_age(db_conn):
    """Test the calculate_age SQL function returns correct age from birth_date."""
    db_conn.execute("""
        INSERT INTO users (id, name, birth_date)
        VALUES (1, 'Alice', '1990-06-15')
    """)
    result = db_conn.execute(
        "SELECT calculate_age(birth_date) FROM users WHERE id = 1"
    ).fetchone()
    expected_age = (date.today() - date(1990, 6, 15)).days // 365
    assert result[0] == expected_age

# Run: pytest test_users.py::test_calculate_user_age
# FAILS - function "calculate_age" does not exist

# Step 2: GREEN - Implement the SQL function
def apply_migration(db_conn):
    db_conn.execute("""
        CREATE OR REPLACE FUNCTION calculate_age(birth_date DATE)
        RETURNS INTEGER AS $$
        BEGIN
            RETURN EXTRACT(YEAR FROM age(CURRENT_DATE, birth_date));
        END;
        $$ LANGUAGE plpgsql IMMUTABLE;
    """)

# Run: pytest test_users.py::test_calculate_user_age
# PASSES

# Step 3: REFACTOR - Add index for queries filtering by age
def optimize_migration(db_conn):
    db_conn.execute("""
        CREATE INDEX ix_users_birth_date ON users(birth_date);
    """)
# Tests still pass
```

### Example TDD with pgTAP (SQL-native testing)

```sql
-- Step 1: RED - Write pgTAP test for a CHECK constraint
BEGIN;
SELECT plan(2);

SELECT has_check('orders', 'positive_amount',
    'orders should have a positive_amount check constraint');

SELECT throws_ok(
    $$INSERT INTO orders (user_id, total_amount) VALUES (1, -10.00)$$,
    23514,  -- check_violation error code
    NULL,
    'Negative amounts should be rejected by check constraint'
);

SELECT * FROM finish();
ROLLBACK;

-- Step 2: GREEN - Add the constraint
ALTER TABLE orders ADD CONSTRAINT positive_amount CHECK (total_amount >= 0);

-- Step 3: REFACTOR - Add partial index for pending orders
CREATE INDEX ix_orders_pending ON orders(created_at) WHERE status = 'pending';
-- pgTAP tests still pass
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
# Bug Report BUG-1042: Users with NULL birth_date cause calculate_age() to crash
# instead of returning NULL gracefully.

import pytest
import psycopg

def test_calculate_age_with_null_birth_date(db_conn):
    """Regression test for BUG-1042: calculate_age must handle NULL birth_date."""
    db_conn.execute("""
        INSERT INTO users (id, name, birth_date)
        VALUES (99, 'NullUser', NULL)
    """)
    result = db_conn.execute(
        "SELECT calculate_age(birth_date) FROM users WHERE id = 99"
    ).fetchone()
    # Should return None, not crash
    assert result[0] is None

# Run: pytest test_users.py::test_calculate_age_with_null_birth_date
# FAILS - function crashes on NULL input

# Fix: Update the function to handle NULL
def fix_calculate_age(db_conn):
    db_conn.execute("""
        CREATE OR REPLACE FUNCTION calculate_age(birth_date DATE)
        RETURNS INTEGER AS $$
        BEGIN
            IF birth_date IS NULL THEN
                RETURN NULL;
            END IF;
            RETURN EXTRACT(YEAR FROM age(CURRENT_DATE, birth_date));
        END;
        $$ LANGUAGE plpgsql IMMUTABLE STRICT;
    """)

# Run: pytest test_users.py::test_calculate_age_with_null_birth_date
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

## 2. Data Types (MANDATORY)

### A. Choosing the Right Type

```sql
-- Primary Keys: Use BIGINT with IDENTITY
CREATE TABLE users (
    id BIGINT PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
    -- ..
);

-- UUIDs: When distributed generation needed
CREATE TABLE events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    -- ..
);

-- Text: Use TEXT, not VARCHAR without limit
CREATE TABLE posts (
    title TEXT NOT NULL,              -- ✅ Preferred
    -- title VARCHAR(255) NOT NULL,   -- Only if limit is business requirement
);

-- Money: Use NUMERIC, never FLOAT
CREATE TABLE orders (
    amount NUMERIC(12, 2) NOT NULL,   -- ✅ Exact precision
    -- amount FLOAT                   -- ❌ Never for money!
);

-- Timestamps: Always use TIMESTAMPTZ
CREATE TABLE events (
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    -- created_at TIMESTAMP           -- ❌ Loses timezone info
);

-- Boolean: Use BOOLEAN, not integers
CREATE TABLE users (
    is_active BOOLEAN NOT NULL DEFAULT true,
    -- is_active INTEGER             -- ❌ Don't use 0/1
);

-- Enums: For fixed sets of values
CREATE TYPE order_status AS ENUM ('pending', 'processing', 'shipped', 'delivered');
CREATE TABLE orders (
    status order_status NOT NULL DEFAULT 'pending'
);

-- JSON: Use JSONB, not JSON
CREATE TABLE settings (
    preferences JSONB NOT NULL DEFAULT '{}',
    -- preferences JSON              -- ❌ JSON is slower, no indexing
);

-- Arrays: For small, simple lists
CREATE TABLE posts (
    tags TEXT[] NOT NULL DEFAULT '{}'
);
```

### B. Identity Columns (GENERATED ALWAYS AS IDENTITY)

Identity columns are the modern replacement for SERIAL/BIGSERIAL. They comply with the SQL standard and provide stronger guarantees against accidental manual value insertion.

```sql
-- ✅ PREFERRED: GENERATED ALWAYS AS IDENTITY (prevents manual ID insertion)
CREATE TABLE users (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    name TEXT NOT NULL
);

-- Attempting to insert an explicit ID will fail:
-- INSERT INTO users (id, name) VALUES (999, 'Hacker');  -- ERROR

-- If you must override (e.g., data migration), use OVERRIDING SYSTEM VALUE:
INSERT INTO users (id, name) OVERRIDING SYSTEM VALUE VALUES (999, 'Migrated');

-- GENERATED BY DEFAULT AS IDENTITY (allows manual ID insertion)
-- Use only when you need to allow explicit ID values
CREATE TABLE imported_data (
    id BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
    source TEXT NOT NULL
);

-- Custom sequence options
CREATE TABLE audit_log (
    id BIGINT GENERATED ALWAYS AS IDENTITY (START WITH 1000 INCREMENT BY 1) PRIMARY KEY,
    action TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ❌ DEPRECATED: Avoid SERIAL/BIGSERIAL in new code
-- CREATE TABLE old_style (id SERIAL PRIMARY KEY);        -- ❌ Legacy
-- CREATE TABLE old_style (id BIGSERIAL PRIMARY KEY);     -- ❌ Legacy
```

### C. Common Type Mistakes

```sql
-- ❌ WRONG: Integer for large IDs
CREATE TABLE events (id SERIAL PRIMARY KEY);  -- Max ~2 billion

-- ✅ CORRECT: BIGINT for scalability
CREATE TABLE events (id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY);

-- ❌ WRONG: VARCHAR(255) habit from MySQL
CREATE TABLE users (bio VARCHAR(255));

-- ✅ CORRECT: TEXT with check constraint if needed
CREATE TABLE users (
    bio TEXT,
    CONSTRAINT bio_length CHECK (length(bio) <= 10000)
);

-- ❌ WRONG: String for IP addresses
CREATE TABLE logs (ip_address VARCHAR(45));

-- ✅ CORRECT: INET type
CREATE TABLE logs (ip_address INET NOT NULL);

-- ❌ WRONG: SERIAL for identity
CREATE TABLE accounts (id SERIAL PRIMARY KEY);

-- ✅ CORRECT: Identity column
CREATE TABLE accounts (id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY);
```

---

## 3. Indexing Strategy (MANDATORY)

### A. Index Types and When to Use Each

```sql
-- B-tree (default): Equality, range queries, ORDER BY, BETWEEN
-- Best for: Most queries. Supports <, <=, =, >=, >, BETWEEN, IN, IS NULL
-- Size: Moderate. Works well up to hundreds of millions of rows.
CREATE INDEX ix_users_email ON users(email);
CREATE INDEX ix_orders_created_at ON orders(created_at DESC);

-- Hash: Only strict equality (=)
-- Best for: Large values where you only do = comparisons
-- Cannot support range queries, ORDER BY, or IS NULL
CREATE INDEX ix_users_token_hash ON users USING hash(reset_token);

-- GIN (Generalized Inverted Index): Multi-valued columns
-- Best for: Full-text search, JSONB containment (@>, ?), arrays, trigram similarity
-- Slower to build/update but very fast for lookups
CREATE INDEX ix_posts_content_gin ON posts USING gin(to_tsvector('english', content));
CREATE INDEX ix_users_preferences_gin ON users USING gin(preferences);
CREATE INDEX ix_posts_tags_gin ON posts USING gin(tags);

-- GIN with jsonb_path_ops: Smaller and faster for @> (containment) only
-- Does NOT support ?, ?|, ?& operators - only @>
CREATE INDEX ix_users_profile_pathops ON users USING gin(profile jsonb_path_ops);

-- GiST (Generalized Search Tree): Geometric, range types, nearest-neighbor
-- Best for: PostGIS geometry, range types, exclusion constraints
CREATE INDEX ix_locations_coords ON locations USING gist(coordinates);
CREATE INDEX ix_reservations_daterange ON reservations USING gist(date_range);

-- BRIN (Block Range Index): Very large tables with natural physical ordering
-- Best for: Append-only tables (logs, events) where column correlates with physical order
-- Extremely small index size (orders of magnitude smaller than B-tree)
-- Only effective when physical row order matches column value order
CREATE INDEX ix_logs_created_at_brin ON logs USING brin(created_at)
    WITH (pages_per_range = 32);
```

### Index Type Selection Guide

```
Decision tree for index selection:

1. Is the column JSONB, array, or tsvector?
   → YES: Use GIN
   → For JSONB with only @> queries: Use GIN with jsonb_path_ops

2. Is the column geometric or a range type?
   → YES: Use GiST

3. Is the table append-only and very large (100M+ rows)?
   → YES and column correlates with insert order: Use BRIN
   → Otherwise: Use B-tree

4. Do you only need exact equality checks?
   → YES and values are large: Consider Hash
   → Otherwise: Use B-tree (default, most versatile)

5. Default choice: B-tree
```

### B. Composite Indexes

```sql
-- Order matters! Match your WHERE clause order
-- For: WHERE user_id = ? AND status = ? ORDER BY created_at DESC
CREATE INDEX ix_orders_user_status_created
ON orders(user_id, status, created_at DESC);

-- Covering index (include columns to avoid table lookup)
CREATE INDEX ix_orders_user_covering
ON orders(user_id)
INCLUDE (status, total_amount, created_at);
```

### C. Partial Indexes

```sql
-- Index only relevant rows
CREATE INDEX ix_orders_pending
ON orders(created_at)
WHERE status = 'pending';

-- For soft deletes
CREATE INDEX ix_users_active_email
ON users(email)
WHERE deleted_at IS NULL;

-- Unique constraint with condition
CREATE UNIQUE INDEX uq_users_email_active
ON users(email)
WHERE deleted_at IS NULL;
```

### D. Expression Indexes

```sql
-- Index on function result
CREATE INDEX ix_users_email_lower ON users(LOWER(email));

-- Index on JSONB field
CREATE INDEX ix_users_settings_theme
ON users((preferences->>'theme'));

-- Index on computed date
CREATE INDEX ix_orders_date
ON orders(DATE(created_at));
```

---

## 4. Query Optimization (MANDATORY)

### A. EXPLAIN ANALYZE Reading Guide

```sql
-- Always analyze complex queries with BUFFERS for I/O detail
EXPLAIN (ANALYZE, BUFFERS, FORMAT TEXT)
SELECT u.id, u.email, COUNT(o.id) as order_count
FROM users u
LEFT JOIN orders o ON o.user_id = u.id
WHERE u.created_at >= '2024-01-01'
GROUP BY u.id, u.email
ORDER BY order_count DESC
LIMIT 100;
```

Reading EXPLAIN output systematically:

```
Key fields in each node:
- "cost=X..Y"     → X = startup cost, Y = total cost (in arbitrary units)
- "rows=N"        → Estimated rows (from planner statistics)
- "actual time"   → Real execution time in milliseconds
- "actual rows"   → Real row count (compare to estimated!)
- "Buffers"       → shared hit = cached, shared read = disk I/O
- "loops=N"       → Node executed N times (multiply actual time * loops for true cost)

Red flags to look for:
1. Seq Scan on large tables          → Add an index
2. rows=1 but actual rows=100000    → Run ANALYZE, statistics are stale
3. Nested Loop with loops=50000     → Consider Hash Join or add index
4. Sort Method: external merge      → Work_mem too low or add index
5. Buffers: shared read >> shared hit → Data not cached, possible I/O bottleneck
6. "actual time" very different from "cost" → Planner misestimate
```

```sql
-- Use FORMAT JSON for programmatic analysis
EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON)
SELECT * FROM orders WHERE user_id = 42;

-- Use VERBOSE to see output column lists
EXPLAIN (ANALYZE, VERBOSE, BUFFERS)
SELECT * FROM orders WHERE status = 'pending';

-- Use SETTINGS to show non-default planner settings affecting the query
EXPLAIN (ANALYZE, BUFFERS, SETTINGS)
SELECT * FROM large_table WHERE category = 'A';

-- WAL option (PG 13+): Show WAL usage for write queries
EXPLAIN (ANALYZE, BUFFERS, WAL)
UPDATE orders SET status = 'shipped' WHERE id = 100;
```

### B. Common Optimizations

```sql
-- ❌ SLOW: OR prevents index use
SELECT * FROM orders WHERE user_id = 1 OR status = 'pending';

-- ✅ FASTER: UNION uses separate indexes
SELECT * FROM orders WHERE user_id = 1
UNION
SELECT * FROM orders WHERE status = 'pending';

-- ❌ SLOW: Function on indexed column
SELECT * FROM users WHERE LOWER(email) = 'test@example.com';

-- ✅ FASTER: Expression index or store lowercase
SELECT * FROM users WHERE email_lower = 'test@example.com';

-- ❌ SLOW: NOT IN with subquery
SELECT * FROM users WHERE id NOT IN (SELECT user_id FROM orders);

-- ✅ FASTER: LEFT JOIN + NULL check
SELECT u.* FROM users u
LEFT JOIN orders o ON o.user_id = u.id
WHERE o.id IS NULL;

-- ✅ EVEN FASTER: NOT EXISTS
SELECT * FROM users u
WHERE NOT EXISTS (SELECT 1 FROM orders o WHERE o.user_id = u.id);
```

### C. Pagination

```sql
-- ❌ SLOW: OFFSET for deep pagination
SELECT * FROM posts ORDER BY created_at DESC LIMIT 20 OFFSET 10000;

-- ✅ FASTER: Keyset pagination
SELECT * FROM posts
WHERE created_at < '2024-01-15T10:00:00Z'
ORDER BY created_at DESC
LIMIT 20;

-- For complex sorts, use composite cursor
SELECT * FROM posts
WHERE (created_at, id) < ('2024-01-15T10:00:00Z', 12345)
ORDER BY created_at DESC, id DESC
LIMIT 20;
```

### D. CTEs vs Subqueries

```sql
-- CTEs (WITH clauses) are optimization fences in PG 11 and earlier.
-- In PG 12+, the planner can inline non-recursive CTEs (unless MATERIALIZED is used).

-- ✅ GOOD: CTE for readability (PG 12+ will inline this)
WITH active_users AS (
    SELECT id, email FROM users WHERE is_active = true
)
SELECT au.email, COUNT(o.id) AS order_count
FROM active_users au
JOIN orders o ON o.user_id = au.id
GROUP BY au.email;

-- Force materialization when the CTE result is used multiple times
-- (prevents re-executing an expensive subquery)
WITH MATERIALIZED expensive_calc AS (
    SELECT user_id, SUM(total_amount) AS lifetime_value
    FROM orders
    GROUP BY user_id
)
SELECT * FROM expensive_calc WHERE lifetime_value > 1000
UNION ALL
SELECT * FROM expensive_calc WHERE lifetime_value BETWEEN 500 AND 1000;

-- Force inlining when you know the planner should push filters down
WITH NOT MATERIALIZED user_orders AS (
    SELECT * FROM orders
)
SELECT * FROM user_orders WHERE user_id = 42;  -- Filter pushed into CTE

-- ❌ AVOID: Recursive CTE without a LIMIT or cycle detection
-- Always include a termination condition
WITH RECURSIVE org_tree AS (
    SELECT id, name, manager_id, 1 AS depth
    FROM employees
    WHERE manager_id IS NULL
    UNION ALL
    SELECT e.id, e.name, e.manager_id, ot.depth + 1
    FROM employees e
    JOIN org_tree ot ON ot.id = e.manager_id
    WHERE ot.depth < 10  -- Always limit recursion depth
)
SELECT * FROM org_tree;
```

### E. MERGE Statement (PostgreSQL 15+)

```sql
-- MERGE combines INSERT, UPDATE, and DELETE in a single statement.
-- Replaces complex INSERT ... ON CONFLICT patterns for multi-action upserts.

-- Basic upsert with MERGE
MERGE INTO inventory AS target
USING incoming_shipment AS source
ON target.product_id = source.product_id
WHEN MATCHED THEN
    UPDATE SET
        quantity = target.quantity + source.quantity,
        last_restocked = NOW()
WHEN NOT MATCHED THEN
    INSERT (product_id, quantity, last_restocked)
    VALUES (source.product_id, source.quantity, NOW());

-- MERGE with DELETE action
MERGE INTO user_sessions AS target
USING (
    SELECT user_id, session_token, expires_at
    FROM new_session_data
) AS source
ON target.user_id = source.user_id
WHEN MATCHED AND source.expires_at < NOW() THEN
    DELETE
WHEN MATCHED THEN
    UPDATE SET
        session_token = source.session_token,
        expires_at = source.expires_at
WHEN NOT MATCHED THEN
    INSERT (user_id, session_token, expires_at)
    VALUES (source.user_id, source.session_token, source.expires_at);

-- MERGE with conditional logic
MERGE INTO product_prices AS target
USING price_updates AS source
ON target.sku = source.sku
WHEN MATCHED AND source.new_price > target.price * 1.5 THEN
    UPDATE SET price = target.price * 1.5,  -- Cap at 50% increase
               updated_at = NOW()
WHEN MATCHED THEN
    UPDATE SET price = source.new_price,
               updated_at = NOW()
WHEN NOT MATCHED THEN
    INSERT (sku, price, updated_at)
    VALUES (source.sku, source.new_price, NOW());
```

### F. Materialized Views

```sql
-- Materialized views store query results physically for fast reads.
-- Use for expensive aggregations, reports, or denormalized data.

CREATE MATERIALIZED VIEW mv_user_order_summary AS
SELECT
    u.id AS user_id,
    u.email,
    COUNT(o.id) AS total_orders,
    COALESCE(SUM(o.total_amount), 0) AS lifetime_value,
    MAX(o.created_at) AS last_order_at
FROM users u
LEFT JOIN orders o ON o.user_id = u.id
GROUP BY u.id, u.email
WITH DATA;  -- Populate immediately (use WITH NO DATA to defer)

-- Add indexes on the materialized view
CREATE UNIQUE INDEX ix_mv_user_summary_uid ON mv_user_order_summary(user_id);
CREATE INDEX ix_mv_user_summary_ltv ON mv_user_order_summary(lifetime_value DESC);

-- Refresh the materialized view
REFRESH MATERIALIZED VIEW mv_user_order_summary;

-- Concurrent refresh (does not lock reads, requires UNIQUE index)
REFRESH MATERIALIZED VIEW CONCURRENTLY mv_user_order_summary;

-- Automate refresh with pg_cron (see Extensions section)
-- Or trigger refresh from application after batch updates
```

### G. Parallel Query Optimization

```sql
-- PostgreSQL can parallelize Seq Scans, Hash Joins, Aggregates, and more.
-- Ensure parallel query is enabled (defaults are usually good):

-- postgresql.conf settings
-- max_parallel_workers_per_gather = 2   -- Workers per query node (default: 2)
-- max_parallel_workers = 8              -- Total parallel workers (default: 8)
-- parallel_tuple_cost = 0.1             -- Lower = more likely to parallelize
-- min_parallel_table_scan_size = 8MB    -- Minimum table size for parallel scan

-- Check if a query uses parallel workers
EXPLAIN (ANALYZE)
SELECT COUNT(*) FROM large_table WHERE category = 'electronics';
-- Look for: "Workers Planned: 2" and "Workers Launched: 2"

-- Force parallel for testing (do not use in production config)
SET parallel_tuple_cost = 0;
SET parallel_setup_cost = 0;
SET min_parallel_table_scan_size = 0;

-- ❌ Parallel query is disabled for:
-- - Queries inside functions declared without PARALLEL SAFE
-- - Queries with serializable isolation level
-- - Queries using cursors
-- - Queries modifying data (INSERT/UPDATE/DELETE) on the target table

-- ✅ Mark functions as PARALLEL SAFE when they are:
CREATE OR REPLACE FUNCTION get_discount_rate(category TEXT)
RETURNS NUMERIC AS $$
BEGIN
    RETURN CASE category
        WHEN 'electronics' THEN 0.05
        WHEN 'books' THEN 0.10
        ELSE 0.02
    END;
END;
$$ LANGUAGE plpgsql IMMUTABLE PARALLEL SAFE;
```

---

## 5. JSONB Operations

### A. When to Use JSONB vs Normalized Columns

```
Use JSONB when:
- Schema varies per row (user preferences, feature flags, metadata)
- Storing third-party API responses where schema is not controlled
- Rapid prototyping where schema is evolving
- The JSON data is read as a blob and rarely queried field-by-field
- Storing nested/hierarchical data that would require many join tables

Use normalized columns when:
- Data has a well-known, stable schema
- Fields are frequently used in WHERE, JOIN, or ORDER BY clauses
- Referential integrity is required (foreign keys)
- Aggregations (SUM, AVG, COUNT) are performed on the field
- The field is part of a unique constraint
- You need column-level permissions

Anti-pattern: Do NOT store data as JSONB just to avoid schema migrations.
If you query a JSONB field in most of your queries, extract it to a column.
```

### B. Querying JSONB

```sql
-- Create table with JSONB
CREATE TABLE users (
    id BIGINT PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
    profile JSONB NOT NULL DEFAULT '{}'
);

-- Insert JSON data
INSERT INTO users (profile) VALUES
('{"name": "John", "age": 30, "settings": {"theme": "dark", "notifications": true}, "tags": ["admin", "beta"]}');

-- Access fields
SELECT
    profile->>'name' AS name,                    -- Text extraction
    profile->'settings'->>'theme' AS theme,      -- Nested text extraction
    (profile->>'age')::integer AS age,           -- Cast to integer
    (profile->'settings'->>'notifications')::boolean AS notifications
FROM users;

-- Operator reference:
-- ->   returns JSONB (preserves type)
-- ->>  returns TEXT (extracts as string)
-- #>   path extraction returning JSONB:   profile #> '{settings,theme}'
-- #>>  path extraction returning TEXT:    profile #>> '{settings,theme}'
-- @>   containment: does left contain right?
-- <@   contained by: is left contained by right?
-- ?    key exists
-- ?|   any key exists
-- ?&   all keys exist

-- Filter by JSON field
SELECT * FROM users WHERE profile->>'name' = 'John';
SELECT * FROM users WHERE profile @> '{"settings": {"theme": "dark"}}';

-- Check key exists
SELECT * FROM users WHERE profile ? 'name';
SELECT * FROM users WHERE profile->'settings' ? 'theme';

-- Check if any of these keys exist
SELECT * FROM users WHERE profile ?| ARRAY['name', 'email'];

-- Check if all keys exist
SELECT * FROM users WHERE profile ?& ARRAY['name', 'age'];

-- Array operations in JSONB
SELECT * FROM users WHERE profile->'tags' @> '"admin"';
SELECT * FROM users WHERE profile->'tags' @> '["admin", "beta"]';
```

### C. JSONPath Expressions (PostgreSQL 12+)

```sql
-- JSONPath provides SQL/JSON standard path language for querying JSONB.
-- It is more powerful than the -> and ->> operators for complex queries.

-- Basic JSONPath queries
SELECT jsonb_path_query(profile, '$.name') FROM users;
SELECT jsonb_path_query(profile, '$.settings.theme') FROM users;

-- Filter with JSONPath predicates
SELECT * FROM users
WHERE jsonb_path_exists(profile, '$.age ? (@ > 25)');

-- Extract values matching a condition
SELECT jsonb_path_query(profile, '$.tags[*] ? (@ == "admin")') FROM users;

-- JSONPath with variables
SELECT * FROM users
WHERE jsonb_path_exists(
    profile,
    '$.age ? (@ >= $min && @ <= $max)',
    '{"min": 18, "max": 65}'
);

-- Return first matching value
SELECT jsonb_path_query_first(profile, '$.tags[0]') FROM users;

-- Return all matches as an array
SELECT jsonb_path_query_array(profile, '$.tags[*]') FROM users;

-- Check if path exists (returns boolean, good for WHERE clauses)
SELECT * FROM users
WHERE jsonb_path_exists(profile, '$.settings.theme ? (@ == "dark")');

-- Arithmetic in JSONPath
SELECT jsonb_path_query(
    '{"price": 100, "discount": 0.15}'::jsonb,
    '$.price * (1 - $.discount)'
);

-- SQL/JSON standard functions (PostgreSQL 16+)
-- JSON_EXISTS - test for existence
SELECT * FROM users
WHERE JSON_EXISTS(profile, '$.settings.theme');

-- JSON_VALUE - extract a scalar value
SELECT JSON_VALUE(profile, '$.name' RETURNING TEXT) AS name
FROM users;

-- JSON_QUERY - extract a JSON object or array
SELECT JSON_QUERY(profile, '$.settings') AS settings
FROM users;
```

### D. JSONB Indexing Strategies

```sql
-- Full GIN index: Supports all JSONB operators (?, ?|, ?&, @>, etc.)
CREATE INDEX ix_users_profile ON users USING gin(profile);

-- GIN with jsonb_path_ops: Smaller, faster, but only supports @> containment
-- Use when you primarily query with @> operator
CREATE INDEX ix_users_profile_pathops ON users USING gin(profile jsonb_path_ops);

-- Expression index on specific field: Best for equality on a known field
CREATE INDEX ix_users_profile_name ON users((profile->>'name'));

-- Expression index with type cast for range queries
CREATE INDEX ix_users_profile_age ON users(((profile->>'age')::integer));

-- Partial index on JSONB field: Index only rows matching a condition
CREATE INDEX ix_users_premium_settings
ON users USING gin(profile)
WHERE (profile->>'plan') = 'premium';

-- Partial index for rows where a key exists
CREATE INDEX ix_users_with_address
ON users USING gin(profile)
WHERE profile ? 'address';

-- Combined: Expression index with partial condition
CREATE INDEX ix_users_active_name
ON users((profile->>'name'))
WHERE (profile->>'is_active')::boolean = true;

-- Unique constraint on JSONB field
CREATE UNIQUE INDEX uq_users_profile_email
ON users((profile->>'email'))
WHERE profile ? 'email';
```

### E. Modifying JSONB

```sql
-- Update nested value
UPDATE users
SET profile = jsonb_set(profile, '{settings,theme}', '"light"')
WHERE id = 1;

-- Add new field
UPDATE users
SET profile = profile || '{"verified": true}'
WHERE id = 1;

-- Remove field
UPDATE users
SET profile = profile - 'temporary_field'
WHERE id = 1;

-- Remove nested field
UPDATE users
SET profile = profile #- '{settings,deprecated_flag}'
WHERE id = 1;

-- Deep merge
UPDATE users
SET profile = profile || '{"settings": {"language": "en"}}'::jsonb
WHERE id = 1;

-- Append to a JSONB array
UPDATE users
SET profile = jsonb_set(
    profile,
    '{tags}',
    (profile->'tags') || '"new_tag"'::jsonb
)
WHERE id = 1;

-- Remove element from JSONB array by value
UPDATE users
SET profile = jsonb_set(
    profile,
    '{tags}',
    (SELECT jsonb_agg(elem)
     FROM jsonb_array_elements(profile->'tags') AS elem
     WHERE elem #>> '{}' != 'beta')
)
WHERE id = 1;

-- Bulk update JSONB across many rows
UPDATE users
SET profile = profile || jsonb_build_object('migrated', true, 'version', 2)
WHERE (profile->>'version')::int = 1 OR NOT (profile ? 'version');
```

### F. JSONB Aggregation and Construction

```sql
-- Build JSONB from columns
SELECT jsonb_build_object(
    'user_id', u.id,
    'email', u.email,
    'orders', (
        SELECT jsonb_agg(jsonb_build_object(
            'id', o.id,
            'amount', o.total_amount,
            'status', o.status
        ))
        FROM orders o WHERE o.user_id = u.id
    )
) AS user_json
FROM users u
WHERE u.id = 42;

-- Aggregate rows into JSONB array
SELECT jsonb_agg(
    jsonb_build_object('id', id, 'name', name)
    ORDER BY name
) AS users_json
FROM users
WHERE is_active = true;

-- Expand JSONB to rows (useful for joining or filtering)
SELECT u.id, kv.key, kv.value
FROM users u,
     jsonb_each(u.profile) AS kv(key, value)
WHERE u.id = 1;

-- Expand JSONB array to rows
SELECT u.id, tag.value AS tag
FROM users u,
     jsonb_array_elements_text(u.profile->'tags') AS tag(value);
```

---

## 6. Table Partitioning (MANDATORY for Large Tables)

### A. When to Partition

```
Partition tables when:
- Table exceeds 100GB or 100M+ rows
- Queries consistently filter on a partition key (date, region, tenant)
- You need to efficiently purge old data (DROP PARTITION vs DELETE)
- Bulk loads benefit from partition-level operations
- Index sizes become unmanageable on the full table

Do NOT partition when:
- Table is small (< 10M rows) - overhead outweighs benefits
- Queries do not filter on the partition key
- You need cross-partition unique constraints (not supported natively)
```

### B. Range Partitioning (Time-Based)

```sql
-- Create partitioned table
CREATE TABLE events (
    id BIGINT GENERATED ALWAYS AS IDENTITY,
    event_type TEXT NOT NULL,
    payload JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
) PARTITION BY RANGE (created_at);

-- Create partitions for each month
CREATE TABLE events_2024_01 PARTITION OF events
    FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');
CREATE TABLE events_2024_02 PARTITION OF events
    FOR VALUES FROM ('2024-02-01') TO ('2024-03-01');
CREATE TABLE events_2024_03 PARTITION OF events
    FOR VALUES FROM ('2024-03-01') TO ('2024-04-01');

-- Create a default partition for data that does not match any range
CREATE TABLE events_default PARTITION OF events DEFAULT;

-- Add indexes (created on each partition automatically)
CREATE INDEX ix_events_created_at ON events(created_at);
CREATE INDEX ix_events_type ON events(event_type);

-- Partition pruning happens automatically when queries filter on created_at:
EXPLAIN SELECT * FROM events WHERE created_at >= '2024-02-01' AND created_at < '2024-03-01';
-- Only events_2024_02 is scanned
```

### C. List Partitioning

```sql
-- Partition by discrete values (region, tenant, status)
CREATE TABLE orders (
    id BIGINT GENERATED ALWAYS AS IDENTITY,
    region TEXT NOT NULL,
    user_id BIGINT NOT NULL,
    total_amount NUMERIC(12, 2) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
) PARTITION BY LIST (region);

CREATE TABLE orders_us PARTITION OF orders FOR VALUES IN ('us-east', 'us-west');
CREATE TABLE orders_eu PARTITION OF orders FOR VALUES IN ('eu-west', 'eu-central');
CREATE TABLE orders_apac PARTITION OF orders FOR VALUES IN ('apac-east', 'apac-south');
CREATE TABLE orders_default PARTITION OF orders DEFAULT;
```

### D. Hash Partitioning

```sql
-- Distribute data evenly when there is no natural range or list
CREATE TABLE session_data (
    session_id UUID NOT NULL DEFAULT gen_random_uuid(),
    user_id BIGINT NOT NULL,
    data JSONB,
    expires_at TIMESTAMPTZ NOT NULL
) PARTITION BY HASH (session_id);

-- Create 4 hash partitions
CREATE TABLE session_data_0 PARTITION OF session_data FOR VALUES WITH (MODULUS 4, REMAINDER 0);
CREATE TABLE session_data_1 PARTITION OF session_data FOR VALUES WITH (MODULUS 4, REMAINDER 1);
CREATE TABLE session_data_2 PARTITION OF session_data FOR VALUES WITH (MODULUS 4, REMAINDER 2);
CREATE TABLE session_data_3 PARTITION OF session_data FOR VALUES WITH (MODULUS 4, REMAINDER 3);
```

### E. Partition Maintenance

```sql
-- Detach old partition (non-blocking in PG 14+)
ALTER TABLE events DETACH PARTITION events_2023_01 CONCURRENTLY;

-- Archive or drop detached partition
-- Option 1: Move to archive schema
ALTER TABLE events_2023_01 SET SCHEMA archive;
-- Option 2: Drop entirely
DROP TABLE events_2023_01;

-- Attach an existing table as a new partition
-- The table must match the partition schema and constraints
ALTER TABLE events ATTACH PARTITION events_2024_04
    FOR VALUES FROM ('2024-04-01') TO ('2024-05-01');

-- Automatic partition creation with pg_partman extension
CREATE EXTENSION pg_partman;
SELECT partman.create_parent(
    p_parent_table := 'public.events',
    p_control := 'created_at',
    p_type := 'native',
    p_interval := 'monthly',
    p_premake := 3  -- Create 3 future partitions
);

-- Schedule partition maintenance (run daily via pg_cron)
SELECT cron.schedule('partition-maintenance', '0 3 * * *',
    $$SELECT partman.run_maintenance('public.events')$$
);
```

---

## 7. Full-Text Search

### A. Basic Setup

```sql
-- Add tsvector column
ALTER TABLE posts ADD COLUMN search_vector tsvector;

-- Create GIN index
CREATE INDEX ix_posts_search ON posts USING gin(search_vector);

-- Update trigger
CREATE OR REPLACE FUNCTION posts_search_update() RETURNS trigger AS $$
BEGIN
    NEW.search_vector := to_tsvector('english',
        COALESCE(NEW.title, '') || ' ' ||
        COALESCE(NEW.content, '')
    );
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER posts_search_update_trigger
    BEFORE INSERT OR UPDATE ON posts
    FOR EACH ROW EXECUTE FUNCTION posts_search_update();

-- Search
SELECT * FROM posts
WHERE search_vector @@ plainto_tsquery('english', 'postgresql tutorial');

-- With ranking
SELECT *, ts_rank(search_vector, query) AS rank
FROM posts, plainto_tsquery('english', 'postgresql') query
WHERE search_vector @@ query
ORDER BY rank DESC;
```

### B. Fuzzy Text Search with pg_trgm

```sql
-- pg_trgm provides trigram-based similarity for fuzzy matching.
-- Essential for "did you mean?" and typo-tolerant search.
CREATE EXTENSION pg_trgm;

-- GIN trigram index for LIKE/ILIKE and similarity
CREATE INDEX ix_users_name_trgm ON users USING gin(name gin_trgm_ops);

-- GiST trigram index (smaller, supports ORDER BY similarity)
CREATE INDEX ix_products_name_trgm ON products USING gist(name gist_trgm_ops);

-- Fuzzy matching with similarity threshold
SELECT name, similarity(name, 'Jonh') AS sim
FROM users
WHERE similarity(name, 'Jonh') > 0.3
ORDER BY sim DESC;

-- Fast LIKE and ILIKE (uses trigram index automatically)
SELECT * FROM users WHERE name ILIKE '%john%';

-- Nearest-neighbor search (requires GiST index)
SELECT name, name <-> 'postgresql' AS distance
FROM products
ORDER BY name <-> 'postgresql'
LIMIT 10;

-- Set similarity threshold
SET pg_trgm.similarity_threshold = 0.3;
SELECT * FROM users WHERE name % 'Jonh';  -- Uses % operator with threshold

-- Combine full-text search with fuzzy matching for best results
SELECT p.*, ts_rank(p.search_vector, q) AS rank
FROM posts p, plainto_tsquery('english', 'postgresql') q
WHERE p.search_vector @@ q
   OR similarity(p.title, 'postgresql') > 0.3
ORDER BY rank DESC;
```

---

## 8. Constraints and Data Integrity

### A. Constraints

```sql
CREATE TABLE orders (
    id BIGINT PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
    user_id BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    status order_status NOT NULL DEFAULT 'pending',
    total_amount NUMERIC(12, 2) NOT NULL,
    discount_amount NUMERIC(12, 2) NOT NULL DEFAULT 0,

    -- Check constraints
    CONSTRAINT positive_amount CHECK (total_amount >= 0),
    CONSTRAINT valid_discount CHECK (discount_amount >= 0 AND discount_amount <= total_amount),

    -- Unique constraints
    CONSTRAINT uq_orders_reference UNIQUE (reference_number),

    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Deferred constraints (checked at commit)
ALTER TABLE order_items
ADD CONSTRAINT fk_order_items_orders
FOREIGN KEY (order_id) REFERENCES orders(id)
DEFERRABLE INITIALLY DEFERRED;
```

### B. Row-Level Security (RLS)

```sql
-- Enable RLS on the table
ALTER TABLE documents ENABLE ROW LEVEL SECURITY;

-- IMPORTANT: Table owners bypass RLS by default.
-- Force RLS for table owners too:
ALTER TABLE documents FORCE ROW LEVEL SECURITY;

-- Policy: Users can only see their own documents
CREATE POLICY documents_select_own ON documents
    FOR SELECT
    USING (user_id = current_setting('app.current_user_id')::bigint);

-- Policy: Users can only insert documents for themselves
CREATE POLICY documents_insert_own ON documents
    FOR INSERT
    WITH CHECK (user_id = current_setting('app.current_user_id')::bigint);

-- Policy: Users can update only their own documents
CREATE POLICY documents_update_own ON documents
    FOR UPDATE
    USING (user_id = current_setting('app.current_user_id')::bigint)
    WITH CHECK (user_id = current_setting('app.current_user_id')::bigint);

-- Policy: Users can delete only their own documents
CREATE POLICY documents_delete_own ON documents
    FOR DELETE
    USING (user_id = current_setting('app.current_user_id')::bigint);

-- Policy: Admins can see and modify all documents
CREATE POLICY documents_admin ON documents
    FOR ALL
    USING (current_setting('app.user_role') = 'admin');

-- Tenant isolation for multi-tenant applications
CREATE POLICY tenant_isolation ON orders
    FOR ALL
    USING (tenant_id = current_setting('app.tenant_id')::bigint)
    WITH CHECK (tenant_id = current_setting('app.tenant_id')::bigint);

-- Set context in application (per-transaction)
BEGIN;
SET LOCAL app.current_user_id = '123';
SET LOCAL app.user_role = 'user';
SET LOCAL app.tenant_id = '456';
-- All queries in this transaction now respect RLS policies
SELECT * FROM documents;  -- Only returns user 123's documents
COMMIT;

-- ❌ NEVER use SET (session-level) for RLS context in connection pools
-- SET app.current_user_id = '123';  -- Persists across transactions!
-- ✅ ALWAYS use SET LOCAL (transaction-level) with pooled connections
```

### C. Column-Level Privileges

```sql
-- Grant access to specific columns only
GRANT SELECT (id, name, email) ON users TO app_readonly;
GRANT SELECT (id, name) ON users TO public_api_role;

-- Hide sensitive columns from application roles
REVOKE ALL ON users FROM app_role;
GRANT SELECT (id, name, email, created_at) ON users TO app_role;
GRANT UPDATE (name, email) ON users TO app_role;
-- Columns like password_hash, ssn, internal_notes are not accessible

-- Use views for column-level filtering (alternative approach)
CREATE VIEW users_public AS
SELECT id, name, email, created_at FROM users;
GRANT SELECT ON users_public TO app_readonly;
```

### D. Role Management Best Practices

```sql
-- Create a hierarchy of roles (never use superuser for applications)

-- 1. Read-only role
CREATE ROLE app_readonly NOLOGIN;
GRANT CONNECT ON DATABASE myapp TO app_readonly;
GRANT USAGE ON SCHEMA public TO app_readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO app_readonly;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO app_readonly;

-- 2. Read-write role (inherits from readonly)
CREATE ROLE app_readwrite NOLOGIN;
GRANT app_readonly TO app_readwrite;
GRANT INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO app_readwrite;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA public TO app_readwrite;
ALTER DEFAULT PRIVILEGES IN SCHEMA public
    GRANT INSERT, UPDATE, DELETE ON TABLES TO app_readwrite;
ALTER DEFAULT PRIVILEGES IN SCHEMA public
    GRANT USAGE ON SEQUENCES TO app_readwrite;

-- 3. Admin role (schema changes, but not superuser)
CREATE ROLE app_admin NOLOGIN;
GRANT app_readwrite TO app_admin;
GRANT CREATE ON SCHEMA public TO app_admin;

-- 4. Login roles (actual users or services)
CREATE ROLE web_app LOGIN PASSWORD 'strong_password_here';
GRANT app_readwrite TO web_app;

CREATE ROLE analytics_user LOGIN PASSWORD 'strong_password_here';
GRANT app_readonly TO analytics_user;

CREATE ROLE migration_user LOGIN PASSWORD 'strong_password_here';
GRANT app_admin TO migration_user;

-- Password expiration
ALTER ROLE web_app VALID UNTIL '2025-12-31';

-- Connection limits per role
ALTER ROLE web_app CONNECTION LIMIT 50;
ALTER ROLE analytics_user CONNECTION LIMIT 5;
```

### E. SSL/TLS Configuration

```sql
-- postgresql.conf: Require SSL
-- ssl = on
-- ssl_cert_file = '/path/to/server.crt'
-- ssl_key_file = '/path/to/server.key'
-- ssl_ca_file = '/path/to/ca.crt'

-- pg_hba.conf: Require SSL for remote connections
-- hostssl  myapp  all  0.0.0.0/0  scram-sha-256
-- hostnossl myapp all  0.0.0.0/0  reject

-- Verify SSL is active
SELECT ssl, version, cipher FROM pg_stat_ssl WHERE pid = pg_backend_pid();

-- Force SSL in connection strings
-- postgresql://user:pass@host:5432/myapp?sslmode=verify-full&sslrootcert=/path/to/ca.crt
```

```python
# Python: Require SSL in connection
import psycopg

conn = psycopg.connect(
    "host=db.example.com dbname=myapp user=web_app password=secret sslmode=verify-full sslrootcert=/path/to/ca.crt"
)
```

---

## 9. Transactions and Locking

### A. Isolation Levels

```sql
-- Read Committed (default) - Good for most cases
BEGIN;
-- ..
COMMIT;

-- Repeatable Read - Consistent snapshot
BEGIN ISOLATION LEVEL REPEATABLE READ;
-- ..
COMMIT;

-- Serializable - Full isolation (careful: may need retries)
BEGIN ISOLATION LEVEL SERIALIZABLE;
-- ..
COMMIT;
```

### B. Advisory Locks

```sql
-- Application-level locking
-- Lock for processing order
SELECT pg_advisory_lock(hashtext('order:' || order_id::text));
-- Process order..
SELECT pg_advisory_unlock(hashtext('order:' || order_id::text));

-- Try lock (non-blocking)
SELECT pg_try_advisory_lock(hashtext('job:daily-report'));

-- Session vs Transaction locks
SELECT pg_advisory_xact_lock(123);  -- Released at transaction end
```

### C. Row Locking

```sql
-- Lock rows for update
SELECT * FROM accounts WHERE id = 1 FOR UPDATE;

-- Skip locked rows (for job queues)
SELECT * FROM jobs
WHERE status = 'pending'
ORDER BY created_at
LIMIT 1
FOR UPDATE SKIP LOCKED;

-- No wait (fail immediately if locked)
SELECT * FROM accounts WHERE id = 1 FOR UPDATE NOWAIT;
```

---

## 10. Performance Monitoring

### A. pg_stat_statements (MANDATORY)

```sql
-- Enable extension (add to shared_preload_libraries in postgresql.conf)
-- shared_preload_libraries = 'pg_stat_statements'
-- pg_stat_statements.track = all
-- pg_stat_statements.max = 10000

CREATE EXTENSION IF NOT EXISTS pg_stat_statements;

-- Find queries consuming the most total time
SELECT
    calls,
    round(total_exec_time::numeric, 2) AS total_time_ms,
    round(mean_exec_time::numeric, 2) AS mean_time_ms,
    round(stddev_exec_time::numeric, 2) AS stddev_ms,
    round((100 * total_exec_time / sum(total_exec_time) OVER ())::numeric, 2) AS percent,
    rows,
    query
FROM pg_stat_statements
ORDER BY total_exec_time DESC
LIMIT 20;

-- Find queries with highest average time (potential optimization targets)
SELECT
    calls,
    round(mean_exec_time::numeric, 2) AS mean_time_ms,
    round(min_exec_time::numeric, 2) AS min_ms,
    round(max_exec_time::numeric, 2) AS max_ms,
    query
FROM pg_stat_statements
WHERE calls > 100  -- Only queries called frequently
ORDER BY mean_exec_time DESC
LIMIT 20;

-- Find queries doing the most I/O (shared blocks read = disk, hit = cache)
SELECT
    calls,
    shared_blks_hit,
    shared_blks_read,
    round(100.0 * shared_blks_hit / NULLIF(shared_blks_hit + shared_blks_read, 0), 2) AS cache_hit_pct,
    query
FROM pg_stat_statements
WHERE shared_blks_read > 0
ORDER BY shared_blks_read DESC
LIMIT 20;

-- Find queries with worst planning time (PG 13+)
SELECT
    calls,
    round(total_plan_time::numeric, 2) AS total_plan_ms,
    round(mean_plan_time::numeric, 2) AS mean_plan_ms,
    query
FROM pg_stat_statements
WHERE total_plan_time > 0
ORDER BY total_plan_time DESC
LIMIT 10;

-- Reset statistics (do periodically to track recent performance)
SELECT pg_stat_statements_reset();
```

### B. Table Statistics

```sql
-- Table sizes
SELECT
    relname AS table_name,
    pg_size_pretty(pg_total_relation_size(relid)) AS total_size,
    pg_size_pretty(pg_relation_size(relid)) AS table_size,
    pg_size_pretty(pg_indexes_size(relid)) AS index_size
FROM pg_stat_user_tables
ORDER BY pg_total_relation_size(relid) DESC;

-- Index usage
SELECT
    indexrelname AS index_name,
    idx_scan AS times_used,
    pg_size_pretty(pg_relation_size(indexrelid)) AS size
FROM pg_stat_user_indexes
ORDER BY idx_scan ASC;

-- Unused indexes (candidates for removal)
SELECT
    schemaname || '.' || relname AS table,
    indexrelname AS index,
    pg_size_pretty(pg_relation_size(indexrelid)) AS size
FROM pg_stat_user_indexes
WHERE idx_scan = 0
AND indexrelid NOT IN (
    SELECT conindid FROM pg_constraint  -- Exclude constraint-backing indexes
)
ORDER BY pg_relation_size(indexrelid) DESC;

-- Table bloat estimation
SELECT
    relname AS table,
    n_live_tup,
    n_dead_tup,
    round(100.0 * n_dead_tup / NULLIF(n_live_tup + n_dead_tup, 0), 2) AS dead_pct,
    last_vacuum,
    last_autovacuum,
    last_analyze,
    last_autoanalyze
FROM pg_stat_user_tables
ORDER BY n_dead_tup DESC;
```

### C. Vacuum and Autovacuum Tuning

```sql
-- Autovacuum settings (postgresql.conf)
-- autovacuum = on                          -- Must be on (default)
-- autovacuum_vacuum_threshold = 50         -- Minimum dead tuples before vacuum
-- autovacuum_vacuum_scale_factor = 0.2     -- Fraction of table that triggers vacuum
-- autovacuum_analyze_threshold = 50        -- Minimum changes before analyze
-- autovacuum_analyze_scale_factor = 0.1    -- Fraction triggering analyze
-- autovacuum_vacuum_cost_delay = 2ms       -- Throttling delay (lower = faster vacuum)
-- autovacuum_vacuum_cost_limit = 200       -- I/O budget per round

-- Formula: Vacuum triggers when dead_tuples > threshold + scale_factor * n_live_tup
-- For a table with 10M rows at default settings:
-- Vacuum triggers after 50 + 0.2 * 10,000,000 = 2,000,050 dead tuples
-- This is often too late for large tables!

-- Per-table autovacuum settings for high-churn tables
ALTER TABLE orders SET (
    autovacuum_vacuum_scale_factor = 0.01,    -- Vacuum after 1% dead tuples
    autovacuum_vacuum_threshold = 1000,
    autovacuum_analyze_scale_factor = 0.005,
    autovacuum_analyze_threshold = 500
);

-- Per-table settings for append-only tables (rarely need vacuum)
ALTER TABLE audit_log SET (
    autovacuum_vacuum_scale_factor = 0.5,     -- Less aggressive
    autovacuum_enabled = true                  -- Still keep it on for freezing
);

-- Manual maintenance commands
VACUUM ANALYZE orders;                  -- Standard vacuum + update statistics
VACUUM (VERBOSE) orders;               -- Verbose output for debugging
VACUUM FULL orders;                    -- Reclaims space but locks table (use rarely!)
ANALYZE orders;                        -- Update planner statistics only

-- Monitor vacuum progress (PG 12+)
SELECT * FROM pg_stat_progress_vacuum;

-- Check for tables at risk of transaction ID wraparound
SELECT
    relname,
    age(relfrozenxid) AS xid_age,
    pg_size_pretty(pg_total_relation_size(oid)) AS size
FROM pg_class
WHERE relkind = 'r'
AND age(relfrozenxid) > 100000000  -- Approaching wraparound
ORDER BY age(relfrozenxid) DESC;
```

---

## 11. Connection Pooling

### A. PgBouncer Configuration

```ini
; pgbouncer.ini
[databases]
myapp = host=localhost port=5432 dbname=myapp

[pgbouncer]
listen_addr = 0.0.0.0
listen_port = 6432
auth_type = scram-sha-256       ; Use scram-sha-256 over md5 for security
auth_file = /etc/pgbouncer/userlist.txt
pool_mode = transaction          ; See pool mode guide below
max_client_conn = 1000           ; Max connections from clients
default_pool_size = 20           ; Server connections per user/db pair
min_pool_size = 5                ; Minimum server connections to keep open
reserve_pool_size = 5            ; Extra connections for burst traffic
reserve_pool_timeout = 3         ; Seconds before using reserve pool
server_idle_timeout = 300        ; Close idle server connections after 5 min
server_lifetime = 3600           ; Close server connections after 1 hour
client_idle_timeout = 0          ; 0 = no timeout for idle clients
query_timeout = 0                ; 0 = no query timeout (set in application)
log_connections = 1
log_disconnections = 1
stats_period = 60                ; Stats logging interval in seconds
```

### B. Pool Mode Selection

```
Transaction Pooling (pool_mode = transaction) - RECOMMENDED
  Connections are returned to the pool after each transaction completes.
  Most efficient for web applications with short-lived transactions.

  Limitations (cannot use with transaction pooling):
  - SET/RESET (session-level settings)
  - LISTEN/NOTIFY
  - Prepared statements (PREPARE/EXECUTE)
  - Session-level advisory locks (pg_advisory_lock)
  - Temporary tables that span transactions
  - Cursors across transaction boundaries

Session Pooling (pool_mode = session)
  One server connection per client connection for the entire session.
  Full PostgreSQL feature support but less connection multiplexing.

  Use when: Application needs LISTEN/NOTIFY, session-level SET,
  prepared statements across transactions, or temp tables.

Statement Pooling (pool_mode = statement)
  Connection returned after each individual statement.
  No multi-statement transactions allowed. Rarely appropriate.
```

### C. Connection Pool Sizing

```
Formulas for sizing your connection pool:

PostgreSQL max_connections:
  Rule of thumb: max_connections = (CPU cores * 2) + effective_spindle_count
  For SSD: max_connections = CPU cores * 4
  Example: 8-core server with SSD -> max_connections = 32

  More connections is NOT better. Each idle connection uses ~10MB RAM.
  300 connections = ~3GB just for connection overhead.

PgBouncer default_pool_size (per user/database pair):
  default_pool_size = max_connections / number_of_user_db_pairs * 0.8
  Example: max_connections=32, 2 databases -> default_pool_size = 12

Application pool_size (per application instance):
  Total across all instances must not exceed PgBouncer max_client_conn.
  With PgBouncer in transaction mode, application pool_size can be larger
  since PgBouncer multiplexes them onto fewer server connections.
```

### D. Application Connection Settings

```python
# SQLAlchemy with connection pooling (point to PgBouncer, not PostgreSQL directly)
from sqlalchemy import create_engine

engine = create_engine(
    "postgresql://user:pass@pgbouncer-host:6432/myapp",
    pool_size=10,           # Connections to keep in pool
    max_overflow=20,        # Additional connections allowed under load
    pool_timeout=30,        # Seconds to wait for connection before error
    pool_recycle=1800,      # Recycle connections after 30 min
    pool_pre_ping=True,     # Verify connection is alive before using
)
```

```python
# psycopg3 with built-in connection pool
import psycopg_pool

pool = psycopg_pool.ConnectionPool(
    conninfo="host=pgbouncer-host port=6432 dbname=myapp user=web_app",
    min_size=4,             # Minimum connections to keep
    max_size=10,            # Maximum connections
    max_idle=300,           # Close idle connections after 5 min
    max_lifetime=3600,      # Recycle connections after 1 hour
)

# Use the pool
with pool.connection() as conn:
    conn.execute("SELECT * FROM users WHERE id = %s", (42,))
```

### E. Health Checks and Monitoring

```sql
-- PgBouncer admin console (connect to pgbouncer virtual database)
-- psql -p 6432 -U pgbouncer pgbouncer

-- Show active pools and their sizes
SHOW POOLS;

-- Show connected clients
SHOW CLIENTS;

-- Show server connections (actual PostgreSQL connections)
SHOW SERVERS;

-- Key stats
SHOW STATS;

-- Key metrics to monitor:
-- cl_active:  Clients actively running a query
-- cl_waiting: Clients waiting for a server connection (should be near 0)
-- sv_active:  Server connections executing a query
-- sv_idle:    Server connections idle in pool (available)
-- maxwait:    Maximum time a client has been waiting (should be < 1s)
```

```python
# Application-level database health check
import psycopg

def check_database_health(conninfo: str) -> dict:
    """Verify database connectivity and connection pool status."""
    try:
        with psycopg.connect(conninfo, connect_timeout=5) as conn:
            conn.execute("SELECT 1")
            result = conn.execute("""
                SELECT count(*) AS total,
                       count(*) FILTER (WHERE state = 'active') AS active,
                       count(*) FILTER (WHERE state = 'idle') AS idle,
                       count(*) FILTER (WHERE wait_event_type = 'Lock') AS waiting
                FROM pg_stat_activity
                WHERE datname = current_database()
            """).fetchone()
            return {
                "healthy": True,
                "total_connections": result[0],
                "active": result[1],
                "idle": result[2],
                "waiting_on_locks": result[3],
            }
    except Exception as e:
        return {"healthy": False, "error": str(e)}
```

---

## 12. Backup and Recovery

### A. pg_dump

```bash
# Logical backup (custom format, compressed)
pg_dump -Fc myapp > myapp.dump

# With maximum compression
pg_dump -Fc -Z9 myapp > myapp.dump

# Schema only
pg_dump -s myapp > schema.sql

# Specific tables
pg_dump -t users -t orders myapp > subset.sql

# Parallel dump for large databases (PG 12+)
pg_dump -Fc -j 4 myapp > myapp.dump

# Restore
pg_restore -d myapp myapp.dump

# Parallel restore
pg_restore -d myapp -j 4 myapp.dump
```

### B. Continuous Archiving (WAL)

```sql
-- postgresql.conf
-- archive_mode = on
-- archive_command = 'cp %p /backup/wal/%f'
-- wal_level = replica
```

### C. Logical Replication (PostgreSQL 16+ Improvements)

```sql
-- Publisher (source database)
-- postgresql.conf: wal_level = logical

-- Create a publication
CREATE PUBLICATION my_publication FOR TABLE users, orders;

-- Publish all tables
CREATE PUBLICATION all_tables_pub FOR ALL TABLES;

-- Publish with row filter (PG 15+)
CREATE PUBLICATION filtered_pub FOR TABLE orders WHERE (region = 'us-east');

-- Publish specific columns (PG 15+)
CREATE PUBLICATION partial_pub FOR TABLE users (id, name, email);

-- Subscriber (target database)
CREATE SUBSCRIPTION my_subscription
    CONNECTION 'host=source-db port=5432 dbname=myapp user=replicator'
    PUBLICATION my_publication;

-- PG 16+: Parallel apply for logical replication
-- Set on subscriber for faster replication:
ALTER SUBSCRIPTION my_subscription SET (streaming = 'parallel');

-- Monitor replication lag
SELECT
    slot_name,
    confirmed_flush_lsn,
    pg_current_wal_lsn(),
    pg_size_pretty(
        pg_wal_lsn_diff(pg_current_wal_lsn(), confirmed_flush_lsn)
    ) AS replication_lag
FROM pg_replication_slots;
```

---

## 13. Modern Extensions (RECOMMENDED)

### A. pg_stat_statements (Query Performance)

See section 10A for detailed usage. This extension is considered mandatory for all production deployments.

### B. pg_trgm (Fuzzy Text Search)

See section 7B for detailed usage with trigram-based similarity search.

### C. pgvector (Vector Embeddings and Similarity Search)

```sql
-- Install pgvector extension
CREATE EXTENSION vector;

-- Create table with vector column
CREATE TABLE documents (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    content TEXT NOT NULL,
    embedding vector(1536)  -- Dimension matches your embedding model (e.g., OpenAI ada-002)
);

-- Insert embeddings (typically from application code)
INSERT INTO documents (content, embedding)
VALUES ('PostgreSQL is a powerful database', '[0.1, 0.2, ...]'::vector);

-- Exact nearest neighbor search (slow for large datasets)
SELECT id, content, embedding <-> '[0.1, 0.2, ...]'::vector AS distance
FROM documents
ORDER BY distance
LIMIT 10;

-- Create index for approximate nearest neighbor (ANN) search
-- IVFFlat: Good for medium datasets, requires training
CREATE INDEX ix_documents_embedding ON documents
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);  -- lists = sqrt(row_count) is a good starting point

-- HNSW: Better recall, higher memory usage, no training needed
CREATE INDEX ix_documents_embedding_hnsw ON documents
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 200);

-- Distance operators:
-- <->  L2 (Euclidean) distance
-- <#>  Negative inner product
-- <=>  Cosine distance

-- Cosine similarity search
SELECT id, content, 1 - (embedding <=> '[0.1, 0.2, ...]'::vector) AS similarity
FROM documents
ORDER BY embedding <=> '[0.1, 0.2, ...]'::vector
LIMIT 10;

-- Set probes for IVFFlat (higher = better recall, slower)
SET ivfflat.probes = 10;

-- Set ef_search for HNSW (higher = better recall, slower)
SET hnsw.ef_search = 100;
```

### D. PostGIS (Geospatial Data)

```sql
-- Install PostGIS
CREATE EXTENSION postgis;

-- Create table with geometry column
CREATE TABLE locations (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    name TEXT NOT NULL,
    coords GEOGRAPHY(POINT, 4326) NOT NULL  -- WGS84 coordinates
);

-- Insert location data
INSERT INTO locations (name, coords) VALUES
('Office', ST_MakePoint(-73.9857, 40.7484)::geography),  -- lon, lat
('Warehouse', ST_MakePoint(-74.0060, 40.7128)::geography);

-- Spatial index
CREATE INDEX ix_locations_coords ON locations USING gist(coords);

-- Find locations within 5km of a point
SELECT name, ST_Distance(coords, ST_MakePoint(-73.99, 40.75)::geography) AS distance_m
FROM locations
WHERE ST_DWithin(coords, ST_MakePoint(-73.99, 40.75)::geography, 5000)
ORDER BY distance_m;

-- K-nearest neighbor search
SELECT name, coords <-> ST_MakePoint(-73.99, 40.75)::geography AS distance_m
FROM locations
ORDER BY coords <-> ST_MakePoint(-73.99, 40.75)::geography
LIMIT 5;
```

### E. pg_cron (Scheduled Jobs)

```sql
-- Install pg_cron (requires shared_preload_libraries = 'pg_cron')
CREATE EXTENSION pg_cron;

-- Schedule a job: clean up expired sessions every hour
SELECT cron.schedule('cleanup-sessions', '0 * * * *',
    $$DELETE FROM sessions WHERE expires_at < NOW()$$
);

-- Schedule a job: refresh materialized view daily at 3 AM
SELECT cron.schedule('refresh-mv', '0 3 * * *',
    $$REFRESH MATERIALIZED VIEW CONCURRENTLY mv_user_order_summary$$
);

-- Schedule a job: vacuum analyze high-churn tables nightly
SELECT cron.schedule('vacuum-orders', '30 2 * * *',
    $$VACUUM ANALYZE orders$$
);

-- Schedule partition maintenance (with pg_partman)
SELECT cron.schedule('partition-maint', '0 4 * * *',
    $$SELECT partman.run_maintenance()$$
);

-- List scheduled jobs
SELECT * FROM cron.job;

-- View job execution history
SELECT * FROM cron.job_run_details ORDER BY start_time DESC LIMIT 20;

-- Unschedule a job
SELECT cron.unschedule('cleanup-sessions');
```

### F. TimescaleDB Integration Notes

```sql
-- TimescaleDB extends PostgreSQL for time-series workloads.
-- Install as an extension (requires separate package installation).
CREATE EXTENSION timescaledb;

-- Convert a regular table to a hypertable
CREATE TABLE sensor_data (
    time TIMESTAMPTZ NOT NULL,
    sensor_id INTEGER NOT NULL,
    temperature DOUBLE PRECISION,
    humidity DOUBLE PRECISION
);

SELECT create_hypertable('sensor_data', 'time');

-- TimescaleDB automatically partitions by time.
-- Standard PostgreSQL queries work transparently:
SELECT time_bucket('1 hour', time) AS hour,
       sensor_id,
       AVG(temperature) AS avg_temp,
       MAX(humidity) AS max_humidity
FROM sensor_data
WHERE time > NOW() - INTERVAL '7 days'
GROUP BY hour, sensor_id
ORDER BY hour DESC;

-- Continuous aggregates (auto-updating materialized views)
CREATE MATERIALIZED VIEW sensor_hourly
WITH (timescaledb.continuous) AS
SELECT time_bucket('1 hour', time) AS hour,
       sensor_id,
       AVG(temperature) AS avg_temp,
       MIN(temperature) AS min_temp,
       MAX(temperature) AS max_temp
FROM sensor_data
GROUP BY hour, sensor_id;

-- Retention policy: automatically drop data older than 90 days
SELECT add_retention_policy('sensor_data', INTERVAL '90 days');

-- Compression policy: compress chunks older than 7 days
ALTER TABLE sensor_data SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'sensor_id',
    timescaledb.compress_orderby = 'time DESC'
);
SELECT add_compression_policy('sensor_data', INTERVAL '7 days');
```

---

## 14. PostgreSQL 16/17 Modern Features

### A. SQL/JSON Standard Functions (PostgreSQL 16+)

```sql
-- JSON_EXISTS: Test if a path exists
SELECT * FROM users
WHERE JSON_EXISTS(profile, '$.settings.theme');

-- JSON_VALUE: Extract a scalar value with type casting
SELECT JSON_VALUE(profile, '$.name' RETURNING TEXT) AS name,
       JSON_VALUE(profile, '$.age' RETURNING INTEGER) AS age
FROM users;

-- JSON_VALUE with default and error handling
SELECT JSON_VALUE(
    profile,
    '$.settings.language'
    RETURNING TEXT
    DEFAULT 'en' ON EMPTY
    DEFAULT 'en' ON ERROR
) AS language
FROM users;

-- JSON_QUERY: Extract JSON objects or arrays
SELECT JSON_QUERY(profile, '$.settings') AS settings_obj,
       JSON_QUERY(profile, '$.tags') AS tags_array
FROM users;

-- JSON_TABLE: Convert JSON to relational rows (PG 17)
SELECT jt.*
FROM users u,
     JSON_TABLE(
         u.profile,
         '$'
         COLUMNS (
             name TEXT PATH '$.name',
             age INTEGER PATH '$.age',
             theme TEXT PATH '$.settings.theme'
         )
     ) AS jt;

-- IS JSON predicates
SELECT * FROM raw_data
WHERE payload IS JSON;

SELECT * FROM raw_data
WHERE payload IS JSON OBJECT;

SELECT * FROM raw_data
WHERE payload IS JSON ARRAY;
```

### B. MERGE Statement Patterns (PostgreSQL 15+)

See section 4E for detailed MERGE examples.

### C. Incremental Sort (PostgreSQL 13+, improved in 16/17)

```sql
-- Incremental sort optimizes ORDER BY when the data is already partially sorted.
-- PostgreSQL can sort remaining columns incrementally instead of a full re-sort.

-- Example: Composite index on (user_id) but need ORDER BY (user_id, created_at)
-- PostgreSQL uses the index for user_id ordering, then incrementally sorts created_at
EXPLAIN (ANALYZE)
SELECT * FROM orders
WHERE status = 'shipped'
ORDER BY user_id, created_at DESC;
-- Look for "Incremental Sort" in the plan

-- Ensure incremental sort is enabled (on by default)
-- SET enable_incremental_sort = on;
```

### D. Identity Columns

See section 2B for detailed GENERATED ALWAYS AS IDENTITY patterns.

---

## 15. Deployment Checklist

### Schema Design
- [ ] Appropriate data types used (TEXT, NUMERIC, TIMESTAMPTZ, etc.)
- [ ] Primary keys are BIGINT GENERATED ALWAYS AS IDENTITY or UUID
- [ ] No SERIAL/BIGSERIAL in new tables (use Identity columns)
- [ ] Foreign keys defined with appropriate ON DELETE action
- [ ] Check constraints for business rule validation
- [ ] All timestamps use TIMESTAMPTZ, never TIMESTAMP
- [ ] JSONB used only for truly dynamic/semi-structured data
- [ ] Tables over 100M rows evaluated for partitioning

### Indexing
- [ ] Indexes on all foreign key columns
- [ ] Indexes for common WHERE, JOIN, and ORDER BY patterns
- [ ] Partial indexes for frequently filtered subsets
- [ ] GIN indexes for JSONB, arrays, and full-text search columns
- [ ] BRIN indexes considered for large append-only tables
- [ ] No unused indexes (checked via pg_stat_user_indexes)
- [ ] Covering indexes (INCLUDE) for high-frequency index-only scans

### Performance
- [ ] EXPLAIN ANALYZE run on all complex queries
- [ ] pg_stat_statements enabled and monitored
- [ ] Connection pooling configured (PgBouncer in transaction mode)
- [ ] Autovacuum tuned for high-churn tables
- [ ] Materialized views with CONCURRENTLY refresh for expensive aggregations
- [ ] Keyset pagination used instead of OFFSET for deep pagination
- [ ] Functions marked PARALLEL SAFE where applicable

### Security
- [ ] Application connects with least-privilege role (never superuser)
- [ ] Role hierarchy established (readonly, readwrite, admin)
- [ ] Row-level security enabled for multi-tenant tables
- [ ] Column-level privileges restrict sensitive columns
- [ ] SSL/TLS required for all remote connections
- [ ] Credentials stored in secrets manager, never hardcoded
- [ ] RLS context set with SET LOCAL (not SET) in pooled connections
- [ ] Password authentication uses scram-sha-256

### Extensions
- [ ] pg_stat_statements loaded in shared_preload_libraries
- [ ] pg_trgm installed for fuzzy search requirements
- [ ] pgvector installed for embedding similarity search
- [ ] pg_cron configured for scheduled maintenance jobs

### Backup and Recovery
- [ ] WAL archiving enabled (wal_level = replica)
- [ ] Regular pg_dump backups scheduled and tested
- [ ] Point-in-time recovery (PITR) tested
- [ ] Backup restoration procedure documented and tested
- [ ] Replication lag monitored for logical replication

---

## 16. Quick Reference

```sql
-- Common psql commands
\l                      -- List databases
\dt                     -- List tables
\di                     -- List indexes
\d+ table_name         -- Describe table with storage info
\df                     -- List functions
\dv                     -- List views
\dm                     -- List materialized views
\dp table_name         -- Show table privileges
\timing on              -- Show query timing
\x auto                 -- Toggle expanded display

-- Maintenance
VACUUM ANALYZE table_name;          -- Reclaim space + update statistics
VACUUM (VERBOSE) table_name;        -- Verbose vacuum output
REINDEX INDEX CONCURRENTLY ix_name; -- Rebuild index without locking
ANALYZE table_name;                 -- Update planner statistics only
CLUSTER table_name USING ix_name;   -- Physically reorder table by index

-- Kill long-running queries
SELECT pid, now() - query_start AS duration, state, query
FROM pg_stat_activity
WHERE state = 'active'
AND query_start < NOW() - INTERVAL '5 minutes'
ORDER BY duration DESC;

-- Cancel a query (graceful)
SELECT pg_cancel_backend(pid);

-- Terminate a connection (forceful)
SELECT pg_terminate_backend(pid);

-- Check locks and blocking queries
SELECT
    blocked.pid AS blocked_pid,
    blocked.query AS blocked_query,
    blocking.pid AS blocking_pid,
    blocking.query AS blocking_query
FROM pg_stat_activity blocked
JOIN pg_locks bl ON bl.pid = blocked.pid
JOIN pg_locks kl ON kl.locktype = bl.locktype
    AND kl.database IS NOT DISTINCT FROM bl.database
    AND kl.relation IS NOT DISTINCT FROM bl.relation
    AND kl.page IS NOT DISTINCT FROM bl.page
    AND kl.tuple IS NOT DISTINCT FROM bl.tuple
    AND kl.transactionid IS NOT DISTINCT FROM bl.transactionid
    AND kl.pid != bl.pid
    AND kl.granted
JOIN pg_stat_activity blocking ON blocking.pid = kl.pid
WHERE NOT bl.granted;

-- Database-level statistics
SELECT
    datname,
    numbackends AS connections,
    xact_commit AS commits,
    xact_rollback AS rollbacks,
    blks_read AS disk_reads,
    blks_hit AS cache_hits,
    round(100.0 * blks_hit / NULLIF(blks_hit + blks_read, 0), 2) AS cache_hit_pct
FROM pg_stat_database
WHERE datname = current_database();

-- Current database size
SELECT pg_size_pretty(pg_database_size(current_database()));

-- List all active connections
SELECT pid, usename, application_name, client_addr, state, query
FROM pg_stat_activity
WHERE datname = current_database()
ORDER BY state, query_start;
```

---

**Last Updated:** 2026-02-27
**Version:** 2.0
**Maintainer:** Database Team


**End of PostgreSQL Development Guidelines**
