# PostgreSQL Development Guidelines
Mandatory standards for PostgreSQL database design, query optimization, and administration. PostgreSQL 15+, pgAdmin, EXPLAIN ANALYZE, pg_stat_statements, pgBadger.

---

**Agent Profile**: The PostgreSQL Expert
**Role**: Senior Database Administrator & Performance Specialist
**Objective**: Generate efficient, secure, and scalable PostgreSQL implementations.
**Tools**: PostgreSQL 15+, pgAdmin, EXPLAIN ANALYZE, pg_stat_statements, pgBadger.

---

## 1. Core Philosophies: POSTGRES-FIRST

- **P**erformance: Use EXPLAIN ANALYZE for all complex queries
- **O**bservability: Enable query logging and monitoring
- **S**ecurity: Use roles, row-level security, and encryption
- **T**ransactions: Proper isolation levels and locking
- **G**reat Types: Use appropriate data types (JSONB, arrays, enums)
- **R**eplication: Plan for high availability from the start
- **E**xtensions: Leverage PostgreSQL's rich extension ecosystem
- **S**chema: Design with normalization and integrity constraints

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

### B. Common Type Mistakes

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
```

---

## 3. Indexing Strategy (MANDATORY)

### A. Index Types

```sql
-- B-tree (default): Equality and range queries
CREATE INDEX ix_users_email ON users(email);
CREATE INDEX ix_orders_created_at ON orders(created_at DESC);

-- Hash: Only equality (rarely needed, B-tree is usually better)
CREATE INDEX ix_users_token_hash ON users USING hash(reset_token);

-- GIN: Full-text search, JSONB, arrays
CREATE INDEX ix_posts_content_gin ON posts USING gin(to_tsvector('english', content));
CREATE INDEX ix_users_preferences_gin ON users USING gin(preferences);
CREATE INDEX ix_posts_tags_gin ON posts USING gin(tags);

-- GiST: Geometric data, range types, full-text
CREATE INDEX ix_locations_coords ON locations USING gist(coordinates);

-- BRIN: Very large tables with natural ordering
CREATE INDEX ix_logs_created_at_brin ON logs USING brin(created_at);
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

### A. EXPLAIN ANALYZE

```sql
-- Always analyze complex queries
EXPLAIN (ANALYZE, BUFFERS, FORMAT TEXT)
SELECT u.id, u.email, COUNT(o.id) as order_count
FROM users u
LEFT JOIN orders o ON o.user_id = u.id
WHERE u.created_at >= '2024-01-01'
GROUP BY u.id, u.email
ORDER BY order_count DESC
LIMIT 100;

-- Key metrics to watch:
-- - Seq Scan on large tables (need index?)
-- - High "rows" vs "actual rows" (statistics outdated?)
-- - Nested Loop with many iterations (wrong join type?)
-- - Sort with high memory (add index?)
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

---

## 5. JSONB Operations

### A. Querying JSONB

```sql
-- Create table with JSONB
CREATE TABLE users (
    id BIGINT PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
    profile JSONB NOT NULL DEFAULT '{}'
);

-- Insert JSON data
INSERT INTO users (profile) VALUES
('{"name": "John", "settings": {"theme": "dark", "notifications": true}}');

-- Access fields
SELECT
    profile->>'name' AS name,                    -- Text
    profile->'settings'->>'theme' AS theme,      -- Nested text
    (profile->'settings'->>'notifications')::boolean AS notifications
FROM users;

-- Filter by JSON field
SELECT * FROM users WHERE profile->>'name' = 'John';
SELECT * FROM users WHERE profile @> '{"settings": {"theme": "dark"}}';

-- Check key exists
SELECT * FROM users WHERE profile ? 'name';
SELECT * FROM users WHERE profile->'settings' ? 'theme';

-- Index for fast lookups
CREATE INDEX ix_users_profile ON users USING gin(profile);
CREATE INDEX ix_users_profile_name ON users((profile->>'name'));
```

### B. Modifying JSONB

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

-- Deep merge
UPDATE users
SET profile = profile || '{"settings": {"language": "en"}}'::jsonb
WHERE id = 1;
```

---

## 6. Full-Text Search

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

---

## 7. Constraints and Data Integrity

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

### B. Row-Level Security

```sql
-- Enable RLS
ALTER TABLE documents ENABLE ROW LEVEL SECURITY;

-- Policy: Users can only see their own documents
CREATE POLICY documents_isolation ON documents
    FOR ALL
    USING (user_id = current_setting('app.current_user_id')::bigint);

-- Policy: Admins can see all
CREATE POLICY documents_admin ON documents
    FOR ALL
    USING (current_setting('app.user_role') = 'admin');

-- Set context in application
SET app.current_user_id = '123';
SET app.user_role = 'user';
```

---

## 8. Transactions and Locking

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

## 9. Performance Monitoring

### A. pg_stat_statements

```sql
-- Enable extension
CREATE EXTENSION pg_stat_statements;

-- Find slow queries
SELECT
    calls,
    round(total_exec_time::numeric, 2) AS total_time_ms,
    round(mean_exec_time::numeric, 2) AS mean_time_ms,
    round((100 * total_exec_time / sum(total_exec_time) OVER ())::numeric, 2) AS percent,
    query
FROM pg_stat_statements
ORDER BY total_exec_time DESC
LIMIT 20;

-- Reset statistics
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

-- Unused indexes
SELECT indexrelname FROM pg_stat_user_indexes WHERE idx_scan = 0;
```

---

## 10. Connection Pooling

### A. PgBouncer Configuration

```ini
; pgbouncer.ini
[databases]
myapp = host=localhost port=5432 dbname=myapp

[pgbouncer]
listen_addr = 0.0.0.0
listen_port = 6432
auth_type = md5
auth_file = /etc/pgbouncer/userlist.txt
pool_mode = transaction  ; or session
max_client_conn = 1000
default_pool_size = 20
min_pool_size = 5
reserve_pool_size = 5
```

### B. Application Connection Settings

```python
# SQLAlchemy with connection pooling
from sqlalchemy import create_engine

engine = create_engine(
    "postgresql://user:pass@localhost:5432/myapp",
    pool_size=10,           # Number of connections to keep
    max_overflow=20,        # Additional connections allowed
    pool_timeout=30,        # Seconds to wait for connection
    pool_recycle=1800,      # Recycle connections after 30 min
    pool_pre_ping=True,     # Check connection health
)
```

---

## 11. Backup and Recovery

### A. pg_dump

```bash
# Logical backup
pg_dump -Fc myapp > myapp.dump

# With compression
pg_dump -Fc -Z9 myapp > myapp.dump

# Schema only
pg_dump -s myapp > schema.sql

# Specific tables
pg_dump -t users -t orders myapp > subset.sql

# Restore
pg_restore -d myapp myapp.dump
```

### B. Continuous Archiving (WAL)

```sql
-- postgresql.conf
archive_mode = on
archive_command = 'cp %p /backup/wal/%f'
wal_level = replica
```

---

## 12. Deployment Checklist

### Schema Design
- [ ] Appropriate data types used
- [ ] Primary keys are BIGINT or UUID
- [ ] Foreign keys defined with appropriate ON DELETE
- [ ] Check constraints for data validation
- [ ] Timestamps use TIMESTAMPTZ

### Indexing
- [ ] Indexes on foreign keys
- [ ] Indexes for common query patterns
- [ ] Partial indexes where appropriate
- [ ] No unused indexes

### Performance
- [ ] EXPLAIN ANALYZE run on complex queries
- [ ] pg_stat_statements enabled
- [ ] Connection pooling configured
- [ ] Autovacuum tuned

### Security
- [ ] Application uses non-superuser role
- [ ] Row-level security where needed
- [ ] SSL connections required
- [ ] Credentials not hardcoded

---

## 13. Quick Reference

```sql
-- Common commands
\l                      -- List databases
\dt                     -- List tables
\di                     -- List indexes
\d+ table_name         -- Describe table
\timing on              -- Show query timing

-- Maintenance
VACUUM ANALYZE table_name;
REINDEX INDEX index_name;
ANALYZE table_name;

-- Kill long-running queries
SELECT pg_terminate_backend(pid)
FROM pg_stat_activity
WHERE state = 'active'
AND query_start < NOW() - INTERVAL '5 minutes';
```

---

**Last Updated:** 2026-01-31
**Version:** 1.0
**Maintainer:** Database Team


**End of PostgreSQL Development Guidelines**
