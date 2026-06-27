# SQL Development Guidelines
Canonical, vendor-neutral standards for relational SQL: querying (joins, subqueries, CTEs, window functions, set operations), data modeling and normalization, indexing, query optimization, transactions and isolation, NULL semantics, constraints, views, and schema migration. ANSI SQL, sqlfluff, sqlc.

---
name: sql
title: SQL Development Guidelines
version: 2.0
last_reviewed: 2026-06-05
kind: cross-cutting
tools: [ansi-sql, sqlfluff, sqlc]
requires: []
recommends:
  - secure-coding
  - performance
  - postgresql
  - mysql-mariadb
  - error-handling
provides:
  - sql-queries
  - joins-ctes-windows
  - normalization
  - sql-indexing
  - transactions-isolation
  - query-optimization
---

> 🧭 Authored per [`CONVENTIONS.md`](guides://CONVENTIONS.md): shared concerns are referenced, not restated. This guide owns the **engine-agnostic, ANSI-leaning** SQL body of knowledge that the relational datastore guides (postgresql, mysql-mariadb, sqlite, …) build on. Vendor syntax and tuning live in those guides, not here.

---

## 0. Prerequisites & References

This guide is the canonical owner of general SQL knowledge. It defers cross-cutting concerns to their owners and engine specifics to the vendor guides.

> 📎 **RECOMMENDED — fetch when the task touches them:**
> - [`secure-coding.md`](guides://secure-coding.md) — injection defense, secrets, least privilege. *(SQL binding: parameterized/bound queries only; app role is never a superuser.)*
> - [`performance.md`](guides://performance.md) — measure-first methodology, budgets, profiling. *(SQL binding: profile with the engine's query planner before optimizing.)*
> - [`error-handling.md`](guides://error-handling.md) — failure strategy. *(SQL binding: map constraint violations — unique, FK, check, not-null — to typed domain errors.)*
> - [`postgresql.md`](guides://postgresql.md) · [`mysql-mariadb.md`](guides://mysql-mariadb.md) — engine-specific types, planner, replication, partitioning, tuning.

> 📎 **SEE ALSO:** [`sqlite.md`](guides://sqlite.md) · [`sqlc.md`](guides://sqlc.md) (type-safe codegen) · [`sqlalchemy-alembic.md`](guides://sqlalchemy-alembic.md) (Python ORM + migrations) · [`tdd.md`](guides://tdd.md) (test-first; assert exact result sets inside a transaction and `ROLLBACK`).

---

## 1. Core Philosophies

SQL-specific principles only. Security, performance methodology, and error strategy come from §0; engine tuning comes from the vendor guides.

- **Declare intent, not procedure.** SQL is set-based and declarative — express *what* result you want and let the planner choose the access path. Prefer one set-based statement over row-by-row loops.
- **Integrity at the database, not the application.** Primary keys, foreign keys, `UNIQUE`, `CHECK`, and `NOT NULL` are the last line of defense and survive every buggy client. Constraints are documentation the engine enforces.
- **Normalize first, denormalize on evidence.** Model to 3NF/BCNF by default; denormalize only with a measured read pattern that justifies the write/sync cost.
- **The plan is the truth.** Read the query plan before trusting a query in production. Indexes, statistics, and sargable predicates exist to serve the plan.
- **Portability by default, vendor features by exception.** Write ANSI SQL where practical; isolate engine-specific features so they are easy to find and swap. Restate vendor behavior in the vendor guide, not here.
- **Schema is versioned code.** Every change is a reviewed, reversible, tested migration — never an ad-hoc `ALTER` against production.

---

## 2. Requirements (MANDATORY, auditable)

RFC-2119 keywords. IDs `SQL-<TOPIC>-<NN>`. Rows that bind a cross-cutting rule cite the owner.

| ID | Requirement | Verify | Gate |
|----|-------------|--------|------|
| SQL-QRY-01 | Production queries MUST name columns explicitly; no `SELECT *` | `sqlfluff lint` / grep | 0 `SELECT *` |
| SQL-QRY-02 | Joins MUST use explicit `JOIN … ON`; never comma/implicit joins | review / lint | 0 implicit joins |
| SQL-QRY-03 | NULL tested only with `IS [NOT] NULL` / `IS [NOT] DISTINCT FROM`; three-valued logic handled | review | exit 0 |
| SQL-SEC-01 | All user input MUST be passed as bound parameters, never concatenated (see `secure-coding.md`) | review / lint | 0 string-built SQL |
| SQL-SEC-02 | App connects with a least-privilege role, not a superuser (see `secure-coding.md`) | role audit | non-superuser |
| SQL-MOD-01 | Every table MUST have a primary key | catalog query | 0 PK-less tables |
| SQL-MOD-02 | OLTP schemas MUST reach 3NF/BCNF unless denormalization is evidence-backed & documented | schema review | documented |
| SQL-MOD-03 | Money MUST be `DECIMAL`/`NUMERIC`; timestamps timezone-aware; no dates-as-strings | schema grep | 0 violations |
| SQL-MOD-04 | Referential integrity MUST be enforced by FK constraints (or documented sharding exception) | catalog query | FKs present |
| SQL-IDX-01 | Every foreign-key column MUST be indexed | catalog query | all FKs indexed |
| SQL-IDX-02 | New access patterns MUST be index-covered; no seq scan on large tables in plan | `EXPLAIN` | no large seq scan |
| SQL-PERF-01 | Non-trivial queries MUST have an `EXPLAIN`/`EXPLAIN ANALYZE` reviewed before ship (see `performance.md`) | plan review | plan attached |
| SQL-PERF-02 | Predicates MUST be sargable — no functions/implicit casts on indexed columns | `EXPLAIN` | index used |
| SQL-PERF-03 | Paginated lists MUST use keyset/cursor pagination, not large `OFFSET` | review | 0 large OFFSET |
| SQL-PERF-04 | No N+1 access patterns; use set-based joins/aggregation (see `performance.md`) | review | set-based |
| SQL-TXN-01 | Multi-statement writes MUST run in one transaction with explicit COMMIT/ROLLBACK | review | atomic |
| SQL-TXN-02 | Isolation level chosen explicitly per workload; serialization failures retried | review | documented |
| SQL-TXN-03 | Concurrent lock acquisition ordered consistently; `lock_timeout` set | review | deadlock-safe |
| SQL-ERR-01 | Constraint violations mapped to typed errors, not leaked raw (see `error-handling.md`) | review | mapped |
| SQL-MIG-01 | Every migration ships tested `up` AND `down`; reversibility verified in CI | migration test | up+down pass |
| SQL-MIG-02 | Schema changes MUST be backward-compatible (expand/contract); index builds non-blocking | review | no breaking change |

> **Forbidden:** string-concatenated SQL, `SELECT *` in production, tables without a primary key, money in `FLOAT`/`DOUBLE`, dates stored as strings, dropping the FK index "for write speed", deploying schema changes without a tested rollback, or running the app as DB superuser.

---

## 3. Querying

### A. SELECT, projection, filtering

```sql
-- Explicit columns (SQL-QRY-01); predicates first, then projection
SELECT id, email, status, created_at
FROM users
WHERE status = 'active'
  AND created_at >= DATE '2025-01-01'
ORDER BY created_at DESC
LIMIT 50;
```

`DISTINCT` deduplicates the whole row — reach for it deliberately, not to paper over a join that multiplies rows (fix the join instead). `CASE` expresses conditional projection portably:

```sql
SELECT name,
       CASE WHEN balance < 0 THEN 'overdrawn'
            WHEN balance = 0 THEN 'empty'
            ELSE 'funded' END AS state
FROM accounts;
```

### B. Joins

Always use explicit `JOIN … ON` (SQL-QRY-02). Join semantics, language-agnostic:

| Join | Result |
|------|--------|
| `INNER JOIN` | only matching rows (most common) |
| `LEFT JOIN` | all left rows + matches; unmatched right side is NULL |
| `RIGHT JOIN` | mirror of LEFT — rewrite as LEFT for readability |
| `FULL JOIN` | all rows from both sides |
| `CROSS JOIN` | Cartesian product — intentional only |

A `LEFT JOIN … WHERE right.col IS NULL` is the canonical "rows with no match" (anti-join) and usually beats `NOT IN`. Beware the classic bug: a one-to-many join multiplies rows and inflates `SUM`/`COUNT` — aggregate in a subquery before joining, or join on the deduplicated grain.

### C. Subqueries

- **Scalar** subquery returns one row/column for projection or comparison.
- **`IN` / `EXISTS`** test membership; `EXISTS` short-circuits and is usually preferred for correlated existence checks.
- **Derived tables** (subquery in `FROM`) and **lateral joins** (a subquery that references the outer row — `LATERAL` / `CROSS APPLY`) let you compute per-row top-N.

### D. Set operations

`UNION` (distinct), `UNION ALL` (keep duplicates — cheaper, prefer when you know rows are disjoint), `INTERSECT`, `EXCEPT`. Branches must be union-compatible (same column count and compatible types).

### E. Aggregation: GROUP BY / HAVING

```sql
SELECT customer_id,
       COUNT(*)                              AS order_count,
       SUM(amount)                           AS revenue,
       COUNT(*) FILTER (WHERE status='paid') AS paid_orders  -- ANSI filtered aggregate
FROM orders
GROUP BY customer_id
HAVING SUM(amount) > 1000;          -- HAVING filters groups; WHERE filters rows
```

`WHERE` filters rows *before* grouping; `HAVING` filters *after*. Every non-aggregated `SELECT` column must appear in `GROUP BY` (or be functionally dependent on the grouped key). Aggregates skip NULLs except `COUNT(*)`.

---

## 4. CTEs and Window Functions

### A. Common Table Expressions

Name a subquery with `WITH` for readability and reuse. Treat a CTE as an optimization fence only if your engine materializes it — check the vendor guide (modern PostgreSQL and others inline by default).

```sql
WITH active AS (
    SELECT id, email FROM users WHERE status = 'active'
)
SELECT a.email, COUNT(o.id) AS orders
FROM active a
LEFT JOIN orders o ON o.user_id = a.id
GROUP BY a.email;
```

**Recursive CTEs** traverse hierarchies and graphs (org charts, bills-of-material, threaded comments):

```sql
WITH RECURSIVE subordinates AS (
    SELECT id, manager_id, name FROM employees WHERE id = :root   -- anchor
    UNION ALL
    SELECT e.id, e.manager_id, e.name
    FROM employees e
    JOIN subordinates s ON e.manager_id = s.id                    -- recursive step
)
SELECT * FROM subordinates;
```

Always guarantee termination (a strictly-shrinking frontier or a depth guard) to avoid infinite recursion on cyclic data.

### B. Window functions

Windows compute across a row set **without collapsing rows** — the modern replacement for self-joins and correlated subqueries for ranking, running totals, and gap/lag analysis.

```sql
SELECT
    id, customer_id, amount, created_at,
    ROW_NUMBER() OVER w                      AS seq,         -- unique ordinal
    RANK()       OVER w                      AS rnk,         -- ties share rank, gaps after
    DENSE_RANK() OVER w                      AS dense_rnk,   -- ties share rank, no gaps
    SUM(amount)  OVER (PARTITION BY customer_id ORDER BY created_at
                       ROWS UNBOUNDED PRECEDING) AS running_total,
    LAG(amount)  OVER w                      AS prev_amount  -- value from prior row
FROM orders
WINDOW w AS (PARTITION BY customer_id ORDER BY created_at DESC);
```

Per-group top-N idiom: window with `ROW_NUMBER() OVER (PARTITION BY g ORDER BY …)` in a subquery, then filter `WHERE seq <= N` in the outer query (windows can't sit in `WHERE`). Frame clauses (`ROWS`/`RANGE … PRECEDING/FOLLOWING`) define the running window; default frame is `RANGE UNBOUNDED PRECEDING TO CURRENT ROW`.

---

## 5. NULL Semantics & Three-Valued Logic

SQL logic is **three-valued**: TRUE, FALSE, UNKNOWN. Any comparison with NULL yields UNKNOWN, and a `WHERE` keeps a row only when its predicate is TRUE.

- `x = NULL` is never TRUE — use `x IS NULL` / `x IS NOT NULL`. `IS [NOT] DISTINCT FROM` compares treating NULL as a value (no UNKNOWN).
- `NOT IN (subquery)` returns no rows if the subquery yields a single NULL — prefer `NOT EXISTS` or an anti-join (this is the most common NULL bug).
- Aggregates ignore NULL; `COUNT(col)` ≠ `COUNT(*)` when `col` has NULLs.
- `NULL` propagates through arithmetic and concatenation — guard with `COALESCE(x, default)` or `NULLIF`.
- `UNIQUE` permits multiple NULLs (they are not "equal"); use a partial/filtered unique index or `NOT NULL` if you need single-occurrence semantics.

Decide deliberately whether a column is nullable; default to `NOT NULL` and model "unknown/absent" explicitly only when it carries meaning.

---

## 6. Data Modeling & Normalization

### A. Normal forms

| Form | Rule | Fix |
|------|------|-----|
| **1NF** | Atomic values, no repeating groups | `products = 'red,blue'` → child table `product_colors(product_id, color)` |
| **2NF** | 1NF + no partial dependency on part of a composite key | move `product_name` out of `order_items(order_id, product_id, …)` into `products` |
| **3NF** | 2NF + no transitive dependency (non-key → non-key) | `orders(…, user_email)` → keep only `user_id`, look up email in `users` |
| **BCNF** | 3NF + every determinant is a candidate key | split tables where a non-candidate-key column determines another |
| **4NF/5NF** | Remove multi-valued / join dependencies | rarely needed; split independent multi-valued facts |

**Practical target: 3NF/BCNF for OLTP.** Normalization eliminates update anomalies and keeps a single source of truth.

### B. When to denormalize

Denormalize only with measured evidence (a verified read pattern), and only with a sync mechanism (trigger, application write path, or materialized view) and an owner for the redundancy.

- ✅ Read-heavy paths with expensive joins proven slow; reporting/OLAP star schemas; cached/computed aggregates; **immutable historical snapshots** (e.g. an order stores the price *as charged*).
- ❌ "Just in case", frequently-mutated source data (sync cost dominates), or before measuring.

Star/snowflake schemas, columnar stores, and HTAP trade-offs are OLAP territory — see the analytics datastore guides rather than restating them here.

### C. Type selection (engine-agnostic rules)

- **Identifiers:** sequential `BIGINT IDENTITY` for locality; `UUID` for distributed/externally-exposed IDs (no sequential leak, shard-friendly).
- **Money/exact decimals:** `DECIMAL`/`NUMERIC(p,s)` — never `FLOAT`/`DOUBLE` (SQL-MOD-03).
- **Time:** timezone-aware timestamps for moments in time; `DATE`/`TIME` for civil dates; store UTC.
- **Text:** prefer variable-length text; constrain length only when a real domain limit exists.
- **Booleans, enums:** native boolean; constrain enumerations with `CHECK (col IN (…))` or a reference table.

Exact type names and extensions (`TIMESTAMPTZ`, `JSONB`, `CITEXT`, `ENUM`, `BINARY(16)`, …) are vendor-specific — see [`postgresql.md`](guides://postgresql.md) / [`mysql-mariadb.md`](guides://mysql-mariadb.md).

---

## 7. Constraints

Enforce business invariants at the lowest level (SQL-MOD-04):

```sql
-- Primary key (single or composite for junction tables)
PRIMARY KEY (user_id, role_id)

-- Foreign key with explicit referential action
CONSTRAINT fk_orders_users FOREIGN KEY (user_id) REFERENCES users(id)
    ON DELETE RESTRICT   -- safe default; CASCADE / SET NULL / SET DEFAULT / NO ACTION as needed
    ON UPDATE CASCADE

-- Unique (composite) and CHECK
CONSTRAINT uq_subscriptions_user_plan UNIQUE (user_id, plan_id)
CONSTRAINT ck_events_dates CHECK (end_date >= start_date)
```

Referential actions: `RESTRICT`/`NO ACTION` (default-safe), `CASCADE` (use carefully — silent deletes), `SET NULL`/`SET DEFAULT` (require a compatible nullable/default column). Name constraints explicitly so violations are diagnosable and so `error-handling.md` mappings are stable. Self-referencing FKs model hierarchies. For soft-delete "unique among live rows", use a **partial/filtered unique index** (`… WHERE deleted_at IS NULL`).

---

## 8. Indexing

### A. Index types (conceptual)

| Type | Best for |
|------|----------|
| **B-tree** (default) | `=`, `<`, `>`, `BETWEEN`, `IN`, `ORDER BY`, prefix of a composite key |
| **Hash** | equality only; rarely beats B-tree |
| **Inverted / GIN-style** | multi-value columns: arrays, JSON, full-text |
| **Spatial / GiST-style** | geometric, range, nearest-neighbor |
| **Block-range (BRIN-style)** | huge, physically-sorted append-only tables (time-series) |

Exact index-type names and operator classes are engine-specific — see the vendor guides.

### B. Design rules

- **Index every foreign key** (SQL-IDX-01) — engines do not do this automatically, and unindexed FKs cause slow joins and locking on parent deletes.
- **Composite index column order matters.** Put equality/most-selective columns first; an index on `(a, b, c)` serves predicates on `a`, `a,b`, `a,b,c`, and `ORDER BY` matching that prefix — but **not** `b` alone. Match the index to the query's leading predicates.
- **Covering indexes** (include the projected columns) enable index-only scans, skipping the table fetch.
- **Partial/filtered indexes** index only the hot subset (`WHERE status='pending'`) — smaller and faster.
- **Expression indexes** support predicates on a computed value (`LOWER(email)`); they only help if the query uses the *same* expression.

### C. Selectivity & when indexes hurt

Indexes help when a predicate is **selective** (returns a small fraction of rows); a low-cardinality column (e.g. boolean) rarely benefits unless paired in a composite or partial index. Every index has a cost: it slows `INSERT`/`UPDATE`/`DELETE`, consumes storage, and can be chosen wrongly by a planner working from stale statistics. Drop unused and redundant indexes; keep table statistics fresh so the planner estimates correctly. Build/drop indexes with the engine's non-blocking option in production (SQL-MIG-02) — see the vendor guide.

---

## 9. Query Optimization

> Optimize only what you have measured — methodology and budgets are owned by [`performance.md`](guides://performance.md). Below is the SQL-specific layer.

### A. Read the plan

`EXPLAIN` shows the planned access path; `EXPLAIN ANALYZE` runs it and reports actual rows/time. Red flags: sequential scan on a large table (missing/ignored index), a large gap between estimated and actual rows (stale statistics — re-`ANALYZE`), unexpected sorts (indexable `ORDER BY`), and nested loops over large inputs.

### B. Sargability

A predicate is **sargable** when it can use an index. Wrapping an indexed column in a function or an implicit cast defeats the index:

```sql
-- ❌ non-sargable: function on the indexed column
WHERE LOWER(email) = 'a@b.com'
WHERE date_trunc('year', created_at) = '2025-01-01'
-- ✅ sargable: rewrite to range / store normalized / use an expression index
WHERE email = 'a@b.com'
WHERE created_at >= DATE '2025-01-01' AND created_at < DATE '2026-01-01'
```

### C. Common rewrites

- Replace correlated per-row subqueries with a single `JOIN` + aggregation.
- Replace `NOT IN (subquery)` with `NOT EXISTS` / anti-join (also fixes the NULL trap, §5).
- Use `EXISTS` instead of `IN` for large existence checks (short-circuits).
- Push `LIMIT`/filters down before joining; use `LATERAL` for per-row top-N.

### D. Avoid N+1 (SQL-PERF-04)

The N+1 pattern — one query per parent row — is the most common ORM-induced performance bug. Fetch sets in one round trip (a join or a single `WHERE id IN (…)`), or use the ORM's eager-loading. Watch the generated SQL in development. ORM/codegen specifics live in [`sqlc.md`](guides://sqlc.md) / [`sqlalchemy-alembic.md`](guides://sqlalchemy-alembic.md).

### E. Keyset pagination over OFFSET (SQL-PERF-03)

`LIMIT … OFFSET n` must scan and discard `n` rows — cost grows with page depth. Use **keyset (cursor) pagination** on the ordered key:

```sql
-- ❌ deep OFFSET scans 100k rows to return 20
SELECT id, created_at FROM users ORDER BY created_at DESC LIMIT 20 OFFSET 100000;

-- ✅ keyset: pass the last row's sort key from the previous page
SELECT id, created_at FROM users
WHERE (created_at, id) < (:last_created_at, :last_id)   -- row-value comparison for tie-break
ORDER BY created_at DESC, id DESC
LIMIT 20;
```

The sort key must be unique and stable (append the primary key to break ties on non-unique columns).

---

## 10. Transactions, ACID, Isolation & Locking

### A. ACID

A transaction is **Atomic** (all-or-nothing), **Consistent** (constraints hold at commit), **Isolated** (concurrent transactions don't corrupt each other), and **Durable** (committed data survives crash). Wrap every multi-statement write in one transaction with explicit `COMMIT`/`ROLLBACK` (SQL-TXN-01); keep transactions **short** — never hold one open across user think-time, network calls, or unbounded work. Use `SAVEPOINT` for partial rollback within a transaction.

### B. Isolation levels and the anomalies they prevent

| Level | Dirty read | Non-repeatable read | Phantom read |
|-------|:---:|:---:|:---:|
| READ UNCOMMITTED | possible | possible | possible |
| **READ COMMITTED** | no | possible | possible |
| REPEATABLE READ | no | no | possible* |
| **SERIALIZABLE** | no | no | no |

- **Dirty read:** seeing another transaction's uncommitted data.
- **Non-repeatable read:** re-reading a row returns a different value (another txn updated/committed).
- **Phantom read:** re-running a range query returns new rows.

*Snapshot-based engines often eliminate phantoms at REPEATABLE READ; exact guarantees and defaults differ — **READ COMMITTED** is the common default and fits most OLTP, **SERIALIZABLE** is for strict invariants (financial/inventory). Choose the level **explicitly** per workload (SQL-TXN-02); don't blanket-apply SERIALIZABLE (it raises contention). SERIALIZABLE (and snapshot) transactions can abort with a serialization failure — **retry with backoff** is mandatory, not optional.

### C. Locking & MVCC concepts

Most modern engines use **MVCC**: writers create new row versions instead of blocking readers, so reads don't block writes and vice versa. Explicit row locks coordinate concurrent writers:

```sql
BEGIN;
SELECT * FROM accounts WHERE id = :id FOR UPDATE;   -- exclusive row lock until commit
UPDATE accounts SET balance = balance - :amt WHERE id = :id;
COMMIT;
```

Lock modes (ANSI/common): `FOR UPDATE` (exclusive), `FOR SHARE` (shared, blocks writers). `SKIP LOCKED` enables work-queue patterns (each worker grabs unlocked rows). **Optimistic locking** avoids held locks via a version column: `UPDATE … SET version=version+1 WHERE id=:id AND version=:v` and check the affected-row count.

**Deadlock avoidance:** acquire locks in a consistent global order (e.g. ascending key), keep transactions short, and set a `lock_timeout` so a stuck transaction fails fast instead of hanging (SQL-TXN-03). Engine-specific lock granularities and MVCC vacuum/undo behavior are in the vendor guides.

### D. Constraint-violation handling

Catch and translate violations (unique, FK, check, not-null) into typed domain errors per [`error-handling.md`](guides://error-handling.md) — e.g. a unique-violation on `email` becomes a domain "email already registered", not a leaked driver error. Use the engine's `INSERT … ON CONFLICT` / upsert to make idempotent writes race-safe rather than catching-and-retrying.

---

## 11. Views

- **Views** name a stored query for abstraction, security (expose a column/row subset), and stable contracts over evolving tables. They carry no storage and are always current; an updatable view must map cleanly to one base table.
- **Materialized views** persist the result for expensive aggregations and must be refreshed (on a schedule or trigger) — they trade staleness for read speed and are a denormalization tool (§6.B). Refresh strategy and concurrency are engine-specific.

Use views to enforce row/column-level access and to give applications a forward-compatible surface during schema migration (§12).

---

## 12. Schema Migration Principles

Schema is versioned code (SQL-MIG-01/02). Tooling (Alembic, golang-migrate, Atlas, Flyway, Liquibase, Sqitch, dbmate, …) differs, but the principles are universal:

- **Every change is a migration** in version control, with a forward (`up`) and a tested reverse (`down`); reversibility verified in CI on a throwaway database.
- **Sequential or timestamped ordering**, one logical change per migration, never edited once merged/applied.
- **Backward compatibility via expand/contract** for zero-downtime: (1) *expand* — add the new nullable column/table and dual-write; (2) *migrate* — backfill in bounded batches and switch reads; (3) *contract* — drop the old object in a later migration after all code is deployed. Never rename/drop a column in the same release that stops using it.
- **Non-blocking DDL in production:** build indexes and validate new constraints without long table locks (add `CHECK`/FK as `NOT VALID` then `VALIDATE` in a separate step); the exact concurrent syntax is engine-specific.
- **Backfill large data in batches** with a key cursor and short transactions to avoid long locks and bloat.
- **High-risk operations** (drop column/table, change type, add `NOT NULL` to a populated column, change primary key, rename) require a backup, a staged plan, and a tested rollback.

```sql
-- expand/contract: add a CHECK without scanning the whole table up front
ALTER TABLE orders ADD CONSTRAINT ck_orders_amount CHECK (total >= 0) NOT VALID;  -- fast
ALTER TABLE orders VALIDATE CONSTRAINT ck_orders_amount;                          -- separate, online
```

Test migrations end to end: run all `up`, then `down -all`, then `up` again, and diff the schema for reproducibility.

---

## 13. Portability

Write ANSI SQL by default; isolate vendor features so a port is a contained change. Portable: standard joins, subqueries, set operations, `CASE`, ANSI aggregates, window functions, CTEs, standard constraints. Engine-specific (keep behind a thin abstraction or a vendor data-access layer): identity/auto-increment syntax, upsert (`ON CONFLICT` vs `ON DUPLICATE KEY`), JSON/array operators, full-text search, `LIMIT`/`OFFSET` vs `FETCH`/`TOP`, date functions, and procedural language. The vendor guides own these mappings — don't restate them here.

---

## 14. Deployment Checklist

Generated 1:1 from §2. All gates must pass before merge/deploy.

**Querying & security**
- [ ] No `SELECT *` in production; explicit columns (SQL-QRY-01)
- [ ] Explicit `JOIN … ON` only; no implicit joins (SQL-QRY-02)
- [ ] NULL handled with `IS [NOT] NULL` / 3VL correct (SQL-QRY-03)
- [ ] All user input bound/parameterized (SQL-SEC-01)
- [ ] App uses a least-privilege role (SQL-SEC-02)

**Modeling**
- [ ] Every table has a primary key (SQL-MOD-01)
- [ ] 3NF/BCNF, or denormalization documented with evidence (SQL-MOD-02)
- [ ] Money is `DECIMAL`/`NUMERIC`; timestamps timezone-aware (SQL-MOD-03)
- [ ] Referential integrity enforced by FKs (SQL-MOD-04)

**Indexing & performance**
- [ ] Every foreign key indexed (SQL-IDX-01)
- [ ] New access patterns index-covered; no large seq scans (SQL-IDX-02)
- [ ] `EXPLAIN`/`EXPLAIN ANALYZE` reviewed for non-trivial queries (SQL-PERF-01)
- [ ] Predicates sargable (SQL-PERF-02)
- [ ] Keyset pagination, not deep OFFSET (SQL-PERF-03)
- [ ] No N+1 patterns (SQL-PERF-04)

**Transactions & migrations**
- [ ] Multi-statement writes are transactional (SQL-TXN-01)
- [ ] Isolation level explicit; serialization failures retried (SQL-TXN-02)
- [ ] Consistent lock order; `lock_timeout` set (SQL-TXN-03)
- [ ] Constraint violations mapped to typed errors (SQL-ERR-01)
- [ ] Migrations have tested up + down (SQL-MIG-01)
- [ ] Schema changes backward-compatible; non-blocking DDL (SQL-MIG-02)

---

**End of SQL Development Guidelines**
