# SQLite Development Guidelines
Mandatory standards for embedded SQLite: WAL mode, STRICT tables, enforced foreign keys, parameterized SQL, tested backups. SQLite 3.46+, FTS5, JSON1.

---
name: sqlite
title: SQLite Development Guidelines
version: 2.0
last_reviewed: 2026-06-05
kind: datastore
tools: [sqlite@3.46, fts5, json1, sqlcipher]
requires:
  - sql
  - secure-coding
recommends:
  - error-handling
  - performance
  - libsql-turso
  - env-config
provides:
  - sqlite-embedded
  - wal-mode
  - strict-tables
  - type-affinity
  - fts5
---

> 🧭 Authored per [`CONVENTIONS.md`](guides://CONVENTIONS.md): shared concerns are referenced, not restated. This guide covers only what is unique to SQLite — the embedded engine, its dynamic-typing footgun, WAL concurrency, and the PRAGMAs/features that make a production deployment correct.

---

## 0. Prerequisites & References

Fetch and apply these **before** writing SQLite code. Their rules are assumed below and not repeated.

> 📎 **REQUIRED — fetch & apply first:**
> - [`sql.md`](guides://sql.md) — general relational modeling, query style, joins, normalization, transactions/ACID, migrations. *(This guide does not restate generic SQL.)*
> - [`secure-coding.md`](guides://secure-coding.md) — injection, secrets, supply chain. *(SQLite binding: **always** parameterize; never string-interpolate SQL; chmod the DB file `0600`; keep encryption keys out of source.)*

> 📎 **RECOMMENDED — fetch when the task touches them:**
> - [`error-handling.md`](guides://error-handling.md) — error strategy *(binding: classify and retry `SQLITE_BUSY`/`SQLITE_LOCKED` with backoff; never swallow `SQLITE_CORRUPT`)*
> - [`performance.md`](guides://performance.md) — profiling/tuning policy *(binding: WAL + PRAGMA cache/mmap, covering indexes, batched writes)*
> - [`libsql-turso.md`](guides://libsql-turso.md) — the networked SQLite fork (libSQL/Turso) for replication, edge, and HTTP access — reach for it when SQLite's single-node/single-writer model is the blocker.
> - [`env-config.md`](guides://env-config.md) — DB path, busy-timeout, and key material come from config, never hardcoded literals.

> 📎 **SEE ALSO:** [`duckdb.md`](guides://duckdb.md) *(analytics/OLAP alternative)* · [`postgresql.md`](guides://postgresql.md) *(when write concurrency outgrows SQLite)* · [`sqlalchemy-alembic.md`](guides://sqlalchemy-alembic.md) / [`sqlc.md`](guides://sqlc.md) *(schema migrations & typed queries)*

---

## 1. Core Philosophies: EMBEDDED-FIRST

SQLite-specific principles only. TDD, generic SQL, security, and error policy come from §0.

- **E**mbedded & serverless: in-process, single-file, zero-config. Design for one machine and local disk — **never** a network filesystem (NFS/SMB corrupts under SQLite's `fcntl` locking).
- **W**AL by default: enable Write-Ahead Logging so readers never block the single writer.
- **S**trict typing: declare `STRICT` tables — SQLite's default dynamic typing is a footgun, not a feature.
- **F**oreign keys on: `PRAGMA foreign_keys = ON` is **off by default** and per-connection; set it on every connection.
- **S**ingle writer: exactly one writer at a time. Batch writes in transactions; handle `SQLITE_BUSY`. If you need concurrent writers, you need a different store (see §11).
- **T**ested durability: WAL + `synchronous = NORMAL`; back up with `VACUUM INTO`/Online Backup API and **test the restore**.

**Verified Code**: agent-generated SQLite MUST use parameterized statements, run against a real on-disk WAL database (not just `:memory:`), and pass every §2 gate before delivery.

---

## 2. Requirements (MANDATORY, auditable)

RFC-2119 keywords. IDs `SQLITE-<TOPIC>-<NN>`. Each row has a binary gate; rows binding a shared rule cite its owner.

| ID | Requirement | Verify | Gate |
|----|-------------|--------|------|
| SQLITE-WAL-01 | Production DBs MUST run in WAL mode | `PRAGMA journal_mode;` | returns `wal` |
| SQLITE-CFG-01 | `synchronous` MUST be `NORMAL` (WAL) or stricter; never `OFF` in prod | `PRAGMA synchronous;` | `1` (NORMAL) or `2` |
| SQLITE-CFG-02 | `busy_timeout` MUST be set (≥ 5000 ms) on every connection | `PRAGMA busy_timeout;` | > 0 |
| SQLITE-FK-01 | `foreign_keys` MUST be ON for every connection that writes | `PRAGMA foreign_keys;` | `1` |
| SQLITE-TYP-01 | New tables MUST be declared `STRICT` (or justify in ADR) | `grep -i 'STRICT' schema.sql` / `sqlite_master` | all CREATE TABLE strict |
| SQLITE-SEC-01 | All SQL MUST be parameterized — no string interpolation (see `secure-coding.md`) | grep/code review | no f-string/`%`/`+` SQL |
| SQLITE-SEC-02 | DB file permissions MUST be `0600`; key material out of source (see `secure-coding.md`) | `stat -c %a app.db` | `600` |
| SQLITE-TXN-01 | Bulk writes MUST run inside a single transaction | code review | one BEGIN/COMMIT per batch |
| SQLITE-ERR-01 | `SQLITE_BUSY`/`SQLITE_LOCKED` MUST be retried with backoff (see `error-handling.md`) | code review / test | retry present |
| SQLITE-IDX-01 | Hot-path queries MUST be index-backed (no unintended full scans) | `EXPLAIN QUERY PLAN` | uses index, not `SCAN` |
| SQLITE-INT-01 | CI MUST run an integrity check on a representative DB | `PRAGMA integrity_check;` | returns `ok` |
| SQLITE-BAK-01 | A backup MUST be produced by `VACUUM INTO`/Online Backup API and a restore tested | restore + `integrity_check` | restored DB `ok` |
| SQLITE-FS-01 | DB MUST NOT live on a network filesystem | deployment review | local disk only |

> **Forbidden**: shipping with dynamic typing where STRICT applies, leaving `foreign_keys` off, copying a live WAL database with `cp` (use `VACUUM INTO`/backup API), `synchronous=OFF` in production, or putting the database on NFS/SMB.

---

## 3. Verification Protocol

Run, in order, against a real on-disk database before presenting code. Fix → re-run until green.

```bash
sqlite3 app.db "PRAGMA journal_mode;"      # SQLITE-WAL-01 → wal
sqlite3 app.db "PRAGMA foreign_keys;"      # SQLITE-FK-01  → 1 (per connection)
sqlite3 app.db "PRAGMA integrity_check;"   # SQLITE-INT-01 → ok
sqlite3 app.db "EXPLAIN QUERY PLAN <hot query>;"  # SQLITE-IDX-01 → USING INDEX
sqlite3 app.db "VACUUM INTO 'restore-test.db';" && \
  sqlite3 restore-test.db "PRAGMA integrity_check;"  # SQLITE-BAK-01
stat -c %a app.db                          # SQLITE-SEC-02 → 600
```

The *why* behind each gate lives in its §0 owner; do not re-derive it here.

---

## 4. Connection Bootstrap (the canonical PRAGMA preamble)

SQLite has **no server config file** — correctness lives in the per-connection PRAGMA preamble. Most settings (`foreign_keys`, `busy_timeout`, `cache_size`) are **connection-scoped and reset every open**; only `journal_mode=WAL`, `page_size`, and `auto_vacuum` persist in the file. Apply this on **every** connection:

```sql
PRAGMA journal_mode = WAL;        -- persists in file; concurrent reads during writes
PRAGMA synchronous = NORMAL;      -- safe with WAL; fewer fsyncs than FULL
PRAGMA foreign_keys = ON;         -- OFF by default, per-connection — easy to forget
PRAGMA busy_timeout = 5000;       -- wait up to 5s for a lock instead of erroring
PRAGMA cache_size = -64000;       -- negative = KiB → 64 MB page cache
PRAGMA temp_store = MEMORY;       -- temp tables/indices in RAM
PRAGMA mmap_size = 268435456;     -- 256 MB memory-mapped I/O (read-heavy wins)
```

| PRAGMA | Scope | Why it matters |
|--------|-------|----------------|
| `journal_mode=WAL` | file (sticky) | Readers never block the writer; 2–4× faster writes |
| `synchronous=NORMAL` | connection | Durability vs. speed; safe with WAL. `OFF` risks corruption on power loss |
| `foreign_keys=ON` | connection | Integrity enforcement is opt-in — the #1 SQLite surprise |
| `busy_timeout` | connection | Turns instant `SQLITE_BUSY` into a bounded wait |
| `cache_size` / `mmap_size` | connection | Tuning (policy: `performance.md`) |

Tuning trade-offs (cache sizing, mmap, write batching) follow [`performance.md`](guides://performance.md); the bindings above are the SQLite levers.

---

## 5. Architecture & When to Choose SQLite

SQLite is a **serverless, in-process, single-file** engine (~700 KB library) that reads/writes an ordinary disk file. There is no daemon, no port, no network — it is a *library*, not a service. The full stack is SQL compiler → VDBE bytecode → B-tree → pager/cache → VFS.

**SQLite is the RIGHT choice for:**
- Application local storage (desktop, mobile, IoT, CLI state, browser-like apps).
- Edge / serverless functions where a single file ships with the app (often via [`libsql-turso.md`](guides://libsql-turso.md) when replication is needed).
- Read-heavy websites and embedded caches.
- Test databases and reproducible fixtures (fast, disposable, no server to stand up).
- On-disk analytical scratch / reporting (small–medium); for heavy OLAP prefer [`duckdb.md`](guides://duckdb.md).

**SQLite is the WRONG choice for** (use [`postgresql.md`](guides://postgresql.md) or [`libsql-turso.md`](guides://libsql-turso.md)):
- **High write concurrency** — exactly one writer at a time; concurrent writers serialize or hit `SQLITE_BUSY`.
- Multi-server / horizontally-scaled writers; built-in replication is absent (libSQL, LiteFS, rqlite, or Dqlite fill this gap).
- Network filesystems (NFS/SMB) — locking is unreliable → corruption.
- Fine-grained access control / row-level security (file permissions only).
- Very large or write-hot datasets (practical sweet spot well under ~1 TB).

---

## 6. Typing: Affinity Footgun → STRICT Tables

SQLite is **dynamically typed**: by default a column's declared type is only an *affinity* (a hint), and any value of any type can be stored in (almost) any column. `age INTEGER` will happily store `'twenty'`. This is the single biggest SQLite correctness trap.

Storage classes: `NULL`, `INTEGER`, `REAL`, `TEXT`, `BLOB`. Affinities: `INTEGER`, `REAL`, `TEXT`, `BLOB`, `NUMERIC`. Unknown types (`BOOLEAN`, `DATETIME`, `JSON`) get `NUMERIC`/`TEXT` affinity and are stored as INTEGER/TEXT — there are **no** native boolean, date, or decimal types.

**Fix: declare `STRICT` tables (SQLite 3.37+)** — they reject mismatched types and forbid bogus column types:

```sql
CREATE TABLE users (
    id    INTEGER PRIMARY KEY,         -- alias of ROWID, no extra storage
    email TEXT NOT NULL UNIQUE COLLATE NOCASE,   -- case-insensitive unique
    age   INTEGER,
    is_active INTEGER NOT NULL DEFAULT 1 CHECK (is_active IN (0,1)),  -- boolean idiom
    created_at INTEGER NOT NULL,        -- Unix epoch seconds (or TEXT ISO-8601)
    metadata TEXT                        -- JSON as TEXT (see §9)
) STRICT;

INSERT INTO users(email, age, created_at) VALUES ('a@x.com','twenty',0); -- ERROR in STRICT
```

SQLite-specific typing idioms:
- **Booleans** → `INTEGER` `0`/`1` with a `CHECK` constraint. **Money** → `INTEGER` cents (or `TEXT`), never `REAL`. **Timestamps** → Unix epoch `INTEGER` or ISO-8601 `TEXT`; there is no `DATETIME` type. **UUIDs** → `TEXT` (`hex(randomblob(16))`) or `BLOB`.
- `INTEGER PRIMARY KEY` aliases the rowid (fast, no extra B-tree). Add `AUTOINCREMENT` only when you must never reuse a deleted id — it costs a `sqlite_sequence` lookup.
- `STRICT` columns must use one of: `INT`, `INTEGER`, `REAL`, `TEXT`, `BLOB`, `ANY`. Use `ANY` deliberately when you really want dynamic storage.
- Generic relational modeling (normalization, keys, constraints) is owned by [`sql.md`](guides://sql.md).

---

## 7. Transactions & the Single-Writer Model

ACID semantics are owned by [`sql.md`](guides://sql.md); SQLite's specialization is its locking model and the single-writer constraint.

**Transaction modes** (choose the lock acquisition timing):
- `BEGIN DEFERRED` (default) — lock acquired lazily on first read/write. Risk: a read-then-write can deadlock against another writer.
- `BEGIN IMMEDIATE` — acquires the write (RESERVED) lock up front. **Prefer this for any transaction that will write** to fail fast/queue cleanly instead of mid-transaction `SQLITE_BUSY`.
- `BEGIN EXCLUSIVE` — blocks all other connections; rarely needed.

**Batch writes in one transaction** — the dominant SQLite performance lever. Per-statement autocommit does one fsync each; wrapping thousands of inserts in a single transaction is often 100× faster (SQLITE-TXN-01). Use `SAVEPOINT`/`ROLLBACK TO` for nested partial rollback.

**Concurrency reality**: locking is **database-level**, not row-level. In WAL mode readers and the single writer run concurrently, but there is still only **one writer at a time**. Contention surfaces as `SQLITE_BUSY` — retry it with exponential backoff (the idiom belongs to [`error-handling.md`](guides://error-handling.md); `busy_timeout` handles the simple case automatically):

```python
# SQLITE-ERR-01 binding: bounded retry around SQLITE_BUSY
for attempt in range(5):
    try:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute("UPDATE accounts SET bal = bal - ? WHERE id = ?", (amt, acc))
        conn.commit(); break
    except sqlite3.OperationalError as e:
        conn.rollback()
        if "locked" in str(e) and attempt < 4:
            time.sleep(0.05 * 2**attempt)   # backoff
        else:
            raise
```

---

## 8. Indexing & Query Plans

Generic indexing theory lives in [`sql.md`](guides://sql.md); SQLite specifics:

- **B-tree** is the only ordinary index. Supports **partial** (`... WHERE active = 1` — smaller, hot-subset), **expression** (`CREATE INDEX i ON t(LOWER(email))`), **covering** (include all selected columns for an index-only scan), and **multi-column** (left-prefix rule: an index on `(a,b,c)` serves `a`, `a,b`, `a,b,c` — not `b` alone).
- **Verify with `EXPLAIN QUERY PLAN`** (SQLITE-IDX-01): `SEARCH ... USING INDEX` is good; a bare `SCAN <table>` on a hot path is a missing index.
- Run `ANALYZE` to populate `sqlite_stat1` for the planner; `PRAGMA optimize;` periodically (e.g., on connection close) keeps stats fresh. `VACUUM` rebuilds/defragments the file and indexes.
- Don't over-index write-hot tables (every index is write amplification) or tiny/low-cardinality columns.

---

## 9. Full-Text Search (FTS5) & JSON

These are SQLite's flagship built-in extensions — canonical to this guide.

**FTS5** — virtual table with an inverted index:

```sql
CREATE VIRTUAL TABLE docs_fts USING fts5(
    title, content,
    content='docs', content_rowid='id',     -- external-content: index only, no duplicate storage
    tokenize='porter unicode61'              -- stemming + Unicode folding
);
-- MATCH queries: 'sqlite database' (AND), 'a OR b', 'a NOT b', '"exact phrase"', 'title:term', 'pre*'
SELECT highlight(docs_fts, 1, '<b>', '</b>'),
       snippet(docs_fts, 1, '[', ']', '…', 10)
FROM docs_fts WHERE docs_fts MATCH 'search' ORDER BY rank;   -- rank = relevance
```

External-content tables need INSERT/UPDATE/DELETE **triggers** on the base table to keep the index in sync; maintain with `INSERT INTO docs_fts(docs_fts) VALUES('optimize'|'rebuild'|'integrity-check')`. For multilingual/edge search at scale, evaluate a dedicated engine, but FTS5 covers most app-local needs.

**JSON** (JSON1, built in since 3.38) — JSON stored as `TEXT`:

```sql
SELECT event->>'$.user_id' AS uid,        -- ->> returns SQL scalar; -> returns JSON
       json_extract(event, '$.action') AS action
FROM logs;
SELECT value FROM json_each('[1,2,3]');                 -- table-valued; json_tree for nested
SELECT json_group_array(json_object('id', id, 'name', name)) FROM users;  -- aggregate to JSON
```

Index JSON paths via a **stored generated column** (don't index `json_extract` of mutable JSON blindly):

```sql
ALTER TABLE logs ADD COLUMN user_id INTEGER
  GENERATED ALWAYS AS (json_extract(event, '$.user_id')) STORED;
CREATE INDEX idx_logs_uid ON logs(user_id);
```

Treat JSON as a flexible sidecar; promote queried fields to real (generated) columns.

---

## 10. Backup, Integrity & Migrations

- **Never `cp` a live WAL database** — you'll miss the `-wal`/`-shm` and get a torn copy. Use one of:
  - `VACUUM INTO 'backup.db'` — consistent, defragmented single-file snapshot (preferred for hot backups).
  - **Online Backup API** (`source.backup(dest)` in most bindings) — page-by-page, safe while in use.
  - `.dump` to SQL text for portable/version-independent archives: `sqlite3 app.db .dump | gzip > app.sql.gz`.
- Always validate a backup by restoring and running `PRAGMA integrity_check;` (SQLITE-BAK-01/INT-01). `PRAGMA quick_check;` is a faster CI variant.
- **Checkpointing**: `PRAGMA wal_checkpoint(TRUNCATE)` flushes WAL into the main file and shrinks it; tune `wal_autocheckpoint` (default ~1000 pages) for write bursts.
- **Schema migrations** are owned by [`sql.md`](guides://sql.md) / [`sqlalchemy-alembic.md`](guides://sqlalchemy-alembic.md). SQLite-specific friction: limited `ALTER TABLE` (`ADD`/`RENAME`/`DROP COLUMN` exist since 3.25/3.35; no general column alter) — complex changes use the 12-step *create-new → copy → drop → rename* table rebuild. `PRAGMA schema_version` / `PRAGMA user_version` track migration state.

---

## 11. Limits, Security & When to Graduate

**Hard limits** (rarely the real constraint): 281 TB max DB size, 1 GB max row/SQL-text, 2000 columns (raisable to 32 767). The *practical* ceiling is **write concurrency and dataset size**, not these numbers.

**Encryption at rest**: core SQLite is unencrypted. Use **SQLCipher** (or the SQLite Encryption Extension) when the file may be exfiltrated; supply the key via `PRAGMA key` from a secrets manager/OS keychain — **never** hardcode it (policy: [`secure-coding.md`](guides://secure-coding.md)). File permissions `0600` (SQLITE-SEC-02) are the baseline.

**Graduation paths** when SQLite stops fitting:

| Need | Reach for |
|------|-----------|
| Concurrent writers / multi-server | [`postgresql.md`](guides://postgresql.md), [`mysql-mariadb.md`](guides://mysql-mariadb.md) |
| Replication / edge / HTTP access (still SQLite) | [`libsql-turso.md`](guides://libsql-turso.md) (libSQL/Turso), LiteFS, rqlite |
| Analytics / OLAP | [`duckdb.md`](guides://duckdb.md) |
| Dedicated full-text at scale | `elasticsearch-opensearch.md` |

libSQL/Turso is the lowest-friction graduation: SQLite-compatible, adds embedded replicas, server mode, and edge replication — see [`libsql-turso.md`](guides://libsql-turso.md).

---

## 12. Quick Reference

```bash
sqlite3 app.db                         # open shell
sqlite3 app.db "PRAGMA journal_mode=WAL;"   # enable WAL (persists)
sqlite3 app.db "PRAGMA integrity_check;"    # verify health
sqlite3 app.db "ANALYZE;"                   # refresh planner stats
sqlite3 app.db "VACUUM INTO 'backup.db';"   # hot backup snapshot
sqlite3 app.db ".dump" > backup.sql         # portable SQL dump
sqlite3 app.db "EXPLAIN QUERY PLAN SELECT …;"  # check index usage
sqlite3 app.db "PRAGMA wal_checkpoint(TRUNCATE);"  # flush + shrink WAL
```

Apply the §4 PRAGMA preamble on every connection — it is the difference between a correct deployment and a corrupt one.

---

## 13. Deployment Checklist

Generated from §2 — one box per requirement ID.

- [ ] SQLITE-WAL-01 — `journal_mode = wal`
- [ ] SQLITE-CFG-01/02 — `synchronous` NORMAL+, `busy_timeout` ≥ 5000 set per connection
- [ ] SQLITE-FK-01 — `foreign_keys = ON` on every writing connection
- [ ] SQLITE-TYP-01 — new tables declared `STRICT`
- [ ] SQLITE-SEC-01/02 — all SQL parameterized; DB file `0600`, keys out of source
- [ ] SQLITE-TXN-01 — bulk writes batched in one transaction
- [ ] SQLITE-ERR-01 — `SQLITE_BUSY`/`LOCKED` retried with backoff
- [ ] SQLITE-IDX-01 — hot queries index-backed (`EXPLAIN QUERY PLAN`)
- [ ] SQLITE-INT-01 — CI runs `integrity_check` → ok
- [ ] SQLITE-BAK-01 — backup produced via `VACUUM INTO`/backup API, restore tested
- [ ] SQLITE-FS-01 — database on local disk, never NFS/SMB
- [ ] Agent ran every §3 command and documented any fixes

---
**End of SQLite Guidelines**
