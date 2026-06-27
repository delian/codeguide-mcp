# Berkeley DB Development Guidelines
Mandatory standards for Berkeley DB: the classic embedded ACID key-value library — access methods, DB_ENV subsystems, transactions/WAL, CDS vs TDS, replication, recovery. Berkeley DB 18.1, C/C++.

---
name: berkeleydb
title: Berkeley DB Development Guidelines
version: 2.0
last_reviewed: 2026-06-05
kind: datastore
tools: [berkeleydb@18.1, c, cpp]
requires:
  - secure-coding
  - error-handling
recommends:
  - performance
  - c
  - cpp
provides:
  - bdb-access-methods
  - bdb-environment
  - bdb-transactions
  - cds-vs-tds
---

> 🧭 Authored per [`CONVENTIONS.md`](guides://CONVENTIONS.md): shared concerns are referenced, not restated. This guide covers only what is unique to Berkeley DB.

---

## 0. Prerequisites & References

Fetch and apply these **before** generating Berkeley DB code. Their rules are assumed below and not repeated.

> 📎 **REQUIRED — fetch & apply first:**
> - [`secure-coding.md`](guides://secure-coding.md) — secrets, supply chain, CVEs. *(BDB binding: `DB_ENV->set_encrypt(... DB_ENCRYPT_AES)` for at-rest AES; never hardcode the passphrase; `chmod 600/700` on the env dir.)*
> - [`error-handling.md`](guides://error-handling.md) — error strategy & propagation. *(BDB binding: every call returns an `int`; `0` is success, `DB_NOTFOUND`/`DB_KEYEXIST` are expected non-errors, `DB_LOCK_DEADLOCK` MUST trigger abort-and-retry — see §6.)*

> 📎 **RECOMMENDED — fetch when the task touches them:**
> - [`performance.md`](guides://performance.md) — cache/page-size tuning policy *(binding: `set_cachesize`, `set_pagesize`, `set_lg_bsize`)*
> - [`c.md`](guides://c.md) · [`cpp.md`](guides://cpp.md) — the API languages (memory ownership, RAII wrappers for handles).

> 📎 **SEE ALSO — modern alternatives (evaluate before adopting BDB on a new project; see §1 licensing):**
> - [`lmdb`] memory-mapped B+tree — *not yet a guide*; closest spiritual successor for read-heavy embedded KV.
> - [`rocksdb.md`](guides://rocksdb.md) · [`leveldb.md`](guides://leveldb.md) — LSM write-optimized embedded KV.
> - [`sqlite.md`](guides://sqlite.md) · [`libsql-turso.md`](guides://libsql-turso.md) — embedded SQL with ACID (BDB also ships a SQLite-compatible API).

---

## 1. Core Philosophies & the Licensing Caveat

Berkeley DB (originally Sleepycat, now Oracle) is an **embedded** library — it links into your process; there is no server, no network hop, no separate daemon to administer. You opt into exactly the subsystems you need via the environment. BDB-specific principles only; TDD, security, error handling, and performance policy come from §0.

- **Library, not server:** persistence is direct function calls (`db->put`, `db->get`). No client/server, no SQL parser (unless you use the SQLite-compatible API).
- **Pick the access method per workload** (§3): BTree (sorted/ranges), Hash (point lookups), Queue (fixed-len FIFO), Recno (record-number array).
- **The environment owns the shared state** (§4): cache (memory pool), lock table, log, and transaction region live in `DB_ENV`, shared across all `DB` handles and (optionally) processes.
- **Opt into subsystems explicitly:** `DB_INIT_MPOOL` (cache), `DB_INIT_LOCK`, `DB_INIT_LOG`, `DB_INIT_TXN`. ACID requires all four; a bare data store needs only MPOOL.
- **Concurrency tier is a deliberate choice** (§7): DS (single-writer), **CDS** (multi-reader/single-writer, no logging), or **TDS** (full transactions). Do not pay for TDS if CDS suffices, and never run multi-writer without at least CDS.

> ⚠️ **Licensing — flag this on every new-project decision.** Up to and including 6.0.19, BDB shipped under the **Sleepycat license** (BSD-style with a copyleft "distribute source of the linking application" clause). In **2013 Oracle relicensed all newer releases (6.0.20+, including 18.1) to AGPLv3**. AGPL's network-copyleft means **any** application that links BDB and is conveyed to users (or, on a strict reading, made available over a network) must release its complete source under AGPL — or you must buy a commercial license from Oracle. This is why Debian, Bitcoin Core, and others froze on or migrated off BDB. **For a new project, default to a modern alternative (§9) unless you have a legacy/embedded ACID constraint that specifically needs BDB and an AGPL-or-commercial license is acceptable.**

**Verified Code**: Agent-generated Berkeley DB code MUST pass every gate in §2 before delivery.

---

## 2. Requirements (MANDATORY, auditable)

RFC-2119 keywords. IDs `BDB-<TOPIC>-<NN>`. Each row has a binary gate; rows binding a shared rule cite its owner.

| ID | Requirement | Verify | Gate |
|----|-------------|--------|------|
| BDB-TST-01 | Every feature MUST be test-first (see `tdd.md`) | `make test` / `ctest` | exit 0, 0 skips |
| BDB-TST-02 | Each bug MUST get a regression test before the fix (see `tdd.md`) | re-run test | failing→passing |
| BDB-ERR-01 | Every BDB call's return code MUST be checked; `DB_NOTFOUND`/`DB_KEYEXIST` handled as values, not errors (see `error-handling.md`) | review / grep for unchecked calls | no unchecked `ret` |
| BDB-ERR-02 | Every transactional operation MUST retry on `DB_LOCK_DEADLOCK` (abort + bounded retry) (see `error-handling.md`) | review / §6 pattern present | retry loop present |
| BDB-TXN-01 | Writes needing durability MUST run inside a transaction in a `DB_INIT_TXN` env; never multi-writer in DS mode | review | all writes txn-scoped or CDS |
| BDB-TXN-02 | On any in-txn error the code MUST `txn->abort()`, never leak an open txn | review / leak test | no leaked txn handles |
| BDB-REC-01 | Env MUST be opened with `DB_RECOVER` by exactly one process at startup; `db_recover` documented for crash restart | review / runbook | recovery path defined |
| BDB-SEC-01 | At-rest sensitive data MUST use `DB_ENCRYPT_AES`; passphrase from secret store, not source (see `secure-coding.md`) | grep for literal passphrase | 0 hardcoded |
| BDB-SEC-02 | Env dir + db files MUST be `chmod 600/700`; 0 known CVEs in libdb (see `secure-coding.md`) | `ls -l` / CVE scan | not world-readable, 0 high/critical |
| BDB-RES-01 | Every `DB`/`DBC`/`DB_ENV` handle MUST be closed (cursors before db, db before env); no handle leaks | valgrind / review | 0 leaks |
| BDB-PERF-01 | `set_cachesize` and `set_pagesize` MUST be set for the workload, not defaulted (see `performance.md`) | review / `db_stat -m` | explicitly configured |
| BDB-LIC-01 | AGPLv3-vs-commercial license obligation MUST be acknowledged in an ADR (see `adr.md`) | ADR exists | recorded |

> **Forbidden**: ignoring a return code; running concurrent writers in DS mode (corruption); holding a cursor open across a long operation while other writers wait; opening a TDS env without `DB_RECOVER` planning; committing a passphrase or world-readable db file.

---

## 3. Access Methods — Pick Per Workload

The access method is chosen at `db->open` time and is immutable for the database's life. There is no "default best" — match it to the access pattern.

| Method | Constant | Key | Order | Choose when |
|--------|----------|-----|-------|-------------|
| **BTree** | `DB_BTREE` | arbitrary bytes | sorted | ranges, prefix scans, variable keys, general purpose (start here) |
| **Hash** | `DB_HASH` | arbitrary bytes | unordered | huge datasets, pure point lookups, working set exceeds cache (extended linear hashing degrades more gracefully than BTree) |
| **Queue** | `DB_QUEUE` | logical record # | FIFO | **fixed-length** records, fast head/tail, concurrent producer/consumer with record-level locking |
| **Recno** | `DB_RECNO` | logical record # | by number | variable-length records addressed like an array; backing-flat-file logs |

- BTree supports sorted duplicates (`DB_DUPSORT`) — required for secondary indexes (§8).
- Queue is the only method with true record-level (not page-level) locking — best for high-concurrency work queues.
- Recno can be backed by a text file (`set_re_source`) so a flat file and the database stay in sync.

---

## 4. The Environment (`DB_ENV`) & Subsystems

The environment is a directory holding shared regions (`__db.NNN`), logs (`log.*`), and database files. It is the unit of caching, locking, logging, and transactions — multiple `DB` handles (and processes) share one `DB_ENV`. You compose behavior by OR-ing subsystem flags into `env->open`.

```c
#include <db.h>

DB_ENV *env;
int ret = db_env_create(&env, 0);                 /* check every ret (BDB-ERR-01) */
env->set_errpfx(env, "myapp");
env->set_errfile(env, stderr);
env->set_cachesize(env, 0, 256 * 1024 * 1024, 1); /* BDB-PERF-01: 256MB, 1 region */
env->set_lk_detect(env, DB_LOCK_DEFAULT);          /* auto deadlock detection */

ret = env->open(env, "./data",
    DB_CREATE   |
    DB_INIT_MPOOL |   /* memory pool / cache  — needed by all */
    DB_INIT_LOCK  |   /* locking subsystem    — CDS/TDS */
    DB_INIT_LOG   |   /* write-ahead log      — TDS */
    DB_INIT_TXN   |   /* transaction manager  — TDS */
    DB_RECOVER    |   /* run recovery on open — exactly ONE process (BDB-REC-01) */
    DB_THREAD,        /* handle is free-threaded */
    0);
```

| Subsystem | Flag | Provides |
|-----------|------|----------|
| Memory pool | `DB_INIT_MPOOL` | shared page cache (the performance lever — §performance.md) |
| Locking | `DB_INIT_LOCK` | two-phase locking, deadlock detection |
| Logging | `DB_INIT_LOG` | write-ahead log (durability + recovery substrate) |
| Transactions | `DB_INIT_TXN` | ACID begin/commit/abort, nested txns |

- **`DB_THREAD`** is mandatory if a handle is shared across threads; DBTs that BDB allocates must then use `DB_DBT_MALLOC`/`DB_DBT_USERMEM`.
- Open the env once per process; pass it to every `db_create(&db, env, 0)`.
- RAII the handles in C++ (close cursor → db → env) to satisfy BDB-RES-01. See [`cpp.md`](guides://cpp.md) for the wrapper idiom.

### DBT — the data container

Keys and values are `DBT` structs (`{void *data; u_int32_t size; ...}`). `memset(&dbt, 0, sizeof(dbt))` before use. Read into caller memory with `DB_DBT_USERMEM` + `ulen`, or let BDB allocate with `DB_DBT_MALLOC` (you `free`). This raw-pointer surface is the classic footgun — apply [`c.md`](guides://c.md) bounds discipline.

---

## 5. Transactions & Write-Ahead Logging (TDS)

ACID in BDB is built on **write-ahead logging**: log records hit stable storage before the data pages, so `db_recover` can roll the database forward/back to a consistent state after a crash. Transactions require an env opened with all four subsystems.

```c
DB_TXN *txn;
int ret = env->txn_begin(env, NULL /* parent */, &txn, 0);

ret = db->put(db, txn, &key, &data, 0);
if (ret != 0) { txn->abort(txn); return ret; }   /* BDB-TXN-02: never leak */

ret = txn->commit(txn, 0);                         /* 0 = sync; DB_TXN_NOSYNC = fast, less durable */
```

- **Nested transactions:** pass a parent to `txn_begin`; a child commit is provisional until the parent commits, a child abort undoes only the child.
- **Auto-commit:** opening with `DB_AUTO_COMMIT` makes each standalone `put`/`del` its own transaction.
- **Durability vs throughput knob:** `commit(txn, 0)` flushes the log (durable). `DB_TXN_NOSYNC` / `DB_TXN_WRITE_NOSYNC` trade the D in ACID for speed — acceptable only where a few seconds of post-crash loss is tolerable. Treat as a [`performance.md`](guides://performance.md) decision, recorded in an ADR.
- **Isolation:** default is repeatable-read (page-level locks); `DB_READ_COMMITTED` (degree 2) and `DB_READ_UNCOMMITTED` relax it; snapshot isolation (MVCC) via `DB_MULTIVERSION` + `DB_TXN_SNAPSHOT`.
- **`DB_RMW`** on a `get` takes a write lock immediately, avoiding a read→upgrade deadlock in read-modify-write flows.

---

## 6. Concurrency: DS vs CDS vs TDS

BDB offers three concurrency tiers. Choosing the lowest tier that meets your needs is the central BDB performance/correctness tradeoff.

| Tier | Env flags | Concurrency | Logging/recovery | Use when |
|------|-----------|-------------|------------------|----------|
| **DS** (Data Store) | `DB_INIT_MPOOL` only | **single process/thread**, no concurrent writers | none | single-threaded embedded store, read-mostly with one owner |
| **CDS** (Concurrent Data Store) | `+ DB_INIT_CDB` | many readers + **one writer at a time**, automatic locking | none (no txns) | multiple readers, occasional writes, no need for atomic multi-op or crash recovery |
| **TDS** (Transactional Data Store) | `+ DB_INIT_LOCK/LOG/TXN` | full multi-reader/multi-writer | WAL + `db_recover` | atomicity across operations, durability, crash recovery, replication |

```c
/* CDS — no transactions, BDB serializes writers for you */
env->open(env, "./data", DB_CREATE | DB_INIT_CDB | DB_INIT_MPOOL, 0);
/* Use db->cursor(db, NULL, &dbc, DB_WRITECURSOR) for the writing cursor. */
```

- **Never run concurrent writers in DS mode** — it silently corrupts (BDB-TXN-01).
- CDS gives you safe concurrency cheaply, but no atomic multi-key updates and no recovery — a crash mid-write can leave a partially written page.
- **Deadlocks are normal in TDS** and surface as `DB_LOCK_DEADLOCK`. The application MUST detect, abort, and retry (BDB-ERR-02). Policy lives in [`error-handling.md`](guides://error-handling.md); the BDB binding:

```c
int op_with_retry(DB_ENV *env, DB *db, DBT *key, DBT *data) {
    DB_TXN *txn; int ret;
    for (int attempt = 0; attempt < MAX_RETRIES; attempt++) {
        if ((ret = env->txn_begin(env, NULL, &txn, 0)) != 0) return ret;
        ret = db->put(db, txn, key, data, 0);
        if (ret == 0) ret = txn->commit(txn, 0);
        if (ret == 0) return 0;
        txn->abort(txn);
        if (ret != DB_LOCK_DEADLOCK) return ret;   /* real error: propagate */
        /* else: contention — loop and retry */
    }
    return DB_LOCK_DEADLOCK;
}
```

Run deadlock detection automatically (`env->set_lk_detect`) or via the `db_deadlock` utility/`DB_AUTO_COMMIT` background thread.

---

## 7. Secondary Indexes

A secondary database maps a derived key → primary key, kept in sync automatically. Provide a callback that extracts the secondary key from the primary record, then `associate`. Query with `pget` (returns the primary data directly).

```c
/* callback: derive secondary key from primary data */
int by_email(DB *sec, const DBT *pkey, const DBT *pdata, DBT *skey) {
    memset(skey, 0, sizeof(DBT));
    skey->data = extract_email(pdata->data);   /* validate/bounds-check input */
    skey->size = email_len;
    return 0;                                   /* or DB_DONOTINDEX to skip */
}

sdb->set_flags(sdb, DB_DUPSORT);                /* many primaries per email */
sdb->open(sdb, NULL, "by_email.db", NULL, DB_BTREE, DB_CREATE | DB_AUTO_COMMIT, 0);
primary->associate(primary, NULL, sdb, by_email, 0);
/* query: cursor->pget(c, &skey, &pkey, &pdata, DB_SET) → primary data */
```

Updates/deletes on the primary cascade to the secondary; never write the secondary directly.

---

## 8. Replication & High Availability (BDB-HA)

BDB-HA is **single-master, multi-replica** with synchronous-or-async log shipping and automatic leader election (Paxos-like). Use the **Replication Manager** (`repmgr_*`) for the built-in TCP transport; the lower-level `rep_*` "Base API" is for custom transports.

```c
env->repmgr_set_local_site(env, host, port, 0);
env->repmgr_set_ack_policy(env, DB_REPMGR_ACKS_QUORUM);   /* durability vs latency */
env->rep_set_priority(env, 100);                          /* 0 = never become master */
env->set_event_notify(env, on_rep_event);                 /* MASTER/CLIENT/NEWMASTER/STARTUPDONE/PERM_FAILED */
env->repmgr_start(env, 3 /* threads */, DB_REP_ELECTION);
```

- **Only the master accepts writes**; replicas are read-only and may lag. Plan for `DB_EVENT_REP_PERM_FAILED` (a committed txn could not be durably replicated).
- Ack policy (`ACKS_ALL`/`ACKS_QUORUM`/`ACKS_NONE`) is the consistency-vs-throughput dial.
- This is HA within a small cluster, **not** a horizontally-sharded distributed database. If you need multi-master or partitioning, BDB is the wrong tool (§9).

---

## 9. Recovery, Backup & Operations

WAL makes BDB crash-tolerant, but recovery must be driven explicitly.

```bash
db_recover  -h ./data          # normal recovery — run on startup after a crash
db_recover  -h ./data -c        # catastrophic recovery — after restoring from backup
db_hotbackup -h ./data -b ./bak # online backup (DB + needed logs)
db_archive  -h ./data -l        # list logs safe to archive
db_archive  -h ./data -d        # delete archived logs (or DB_LOG_AUTO_REMOVE)
db_verify   ./data/mydb.db      # integrity check
db_stat     -h ./data -m        # memory-pool / cache hit-rate stats (BDB-PERF-01)
db_dump -r / db_load            # salvage / reload (corruption recovery)
```

- **Exactly one process** runs recovery (`DB_RECOVER`) at env startup (BDB-REC-01); others attach after.
- Backup = a consistent copy of the database files **plus** the log files needed to recover them; `db_hotbackup` handles ordering. Test the restore, not just the backup.
- Checkpoint (`txn_checkpoint`) periodically to bound recovery time and let logs be archived.

---

## 10. When Berkeley DB Fits — and Modern Alternatives

**BDB fits when:** you maintain a legacy system already on it; you need a battle-tested, in-process, ACID, transactional KV store with built-in HA and no server to operate; the AGPLv3 (or a commercial license) is acceptable.

**Prefer a modern alternative for new work:**

| Need | Reach for | Why over BDB |
|------|-----------|--------------|
| Read-heavy embedded KV, simplest ops | **LMDB** | tiny, mmap'd, MVCC, lock-free reads, OpenLDAP-driven, OpenLDAP/BSD license |
| Write-heavy / large datasets | [`rocksdb.md`](guides://rocksdb.md), [`leveldb.md`](guides://leveldb.md) | LSM trees, compression, far better write amplification; permissive license |
| Need SQL / relational | [`sqlite.md`](guides://sqlite.md), [`libsql-turso.md`](guides://libsql-turso.md) | full SQL, public-domain, ubiquitous (BDB's own SQL API is a SQLite shim) |

The decision (and the AGPL obligation) MUST be recorded in an ADR — see [`adr.md`](guides://adr.md) (BDB-LIC-01).

---

## 11. Verification Protocol

Run before presenting code. Fix → re-run until every gate is green. The *why* lives in §0 owners.

```bash
ctest / make test                  # BDB-TST-01/02
grep -n "db->\|env->\|txn->" src/   # BDB-ERR-01: confirm every ret is checked
valgrind --leak-check=full ./app    # BDB-RES-01: no handle/DBT leaks
db_verify ./data/*.db               # integrity after tests
db_stat -h ./data -m                # BDB-PERF-01: cache hit-rate sane
ls -l ./data                        # BDB-SEC-02: not world-readable
grep -RnE "set_encrypt\(.*\"" src/  # BDB-SEC-01: no literal passphrase
```

---

## 12. Deployment Checklist

Generated from §2 — one box per requirement ID.

- [ ] BDB-TST-01/02 — tests pass test-first; bugs have regression tests
- [ ] BDB-ERR-01 — every return code checked; `DB_NOTFOUND`/`DB_KEYEXIST` handled as values
- [ ] BDB-ERR-02 — `DB_LOCK_DEADLOCK` abort-and-retry loop present
- [ ] BDB-TXN-01/02 — durable writes are txn-scoped (or CDS); no leaked txns
- [ ] BDB-REC-01 — single-process `DB_RECOVER`; `db_recover` runbook documented
- [ ] BDB-SEC-01/02 — AES at-rest, passphrase external; files `chmod 600/700`, 0 CVEs
- [ ] BDB-RES-01 — all cursor/db/env handles closed in order; valgrind clean
- [ ] BDB-PERF-01 — cache & page size explicitly tuned (`db_stat -m`)
- [ ] BDB-LIC-01 — AGPLv3-vs-commercial decision recorded in an ADR
- [ ] Agent ran every §11 command and documented any fixes

---
**End of Berkeley DB Guidelines**
