# LMDB Development Guidelines
Mandatory standards for LMDB: the memory-mapped, copy-on-write B+tree key-value store. Zero-copy reads, single-writer MVCC, and the map-size discipline. LMDB 0.9.x, C API.

---
name: lmdb
title: LMDB Development Guidelines
version: 2.0
last_reviewed: 2026-06-05
kind: datastore
tools: [lmdb@0.9, mdb_stat, mdb_copy, mdb_dump, mdb_load, valgrind]
requires:
  - secure-coding
  - error-handling
recommends:
  - performance
  - c
  - parallelism
provides:
  - memory-mapped-btree
  - lmdb-mvcc
  - zero-copy-reads
  - single-writer-many-readers
---

> 🧭 Authored per [`CONVENTIONS.md`](guides://CONVENTIONS.md): shared concerns are referenced, not restated. This guide covers only what is unique to LMDB.

---

## 0. Prerequisites & References

Fetch and apply these **before** generating LMDB code. Their rules are assumed below and not repeated.

> 📎 **REQUIRED — fetch & apply first:**
> - [`secure-coding.md`](guides://secure-coding.md) — supply chain, secrets, input validation. *(LMDB binding: no built-in encryption — encryption-at-rest is the filesystem's or the app's job; §8.)*
> - [`error-handling.md`](guides://error-handling.md) — error strategy & propagation. *(LMDB binding: every `mdb_*` call returns an `int` rc; check it, map `MDB_*` codes; §6.)*

> 📎 **RECOMMENDED — fetch when the task touches them:**
> - [`performance.md`](guides://performance.md) — mmap, page cache, zero-copy cost model. *(LMDB binding: reads are pointers into the mmap; the OS page cache is your read cache.)*
> - [`c.md`](guides://c.md) — the C API is the canonical binding; pointer lifetime, memory safety, `valgrind`.
> - [`parallelism.md`](guides://parallelism.md) — the reader/writer concurrency model. *(LMDB binding: one writer at a time, many lock-free readers; §5.)*

> 📎 **SEE ALSO:** [`tdd.md`](guides://tdd.md) · [`leveldb.md`](guides://leveldb.md) · [`rocksdb.md`](guides://rocksdb.md) · [`berkeleydb.md`](guides://berkeleydb.md) *(LSM / BDB alternatives — see §9 trade-offs)*

---

## 1. Core Philosophies: LMDB-FIRST

LMDB-specific principles only. Testing, security, and error strategy come from §0.

- **L**ightning reads: reads are zero-copy — `mdb_get` returns a pointer **into the mmap**, valid only for the life of its transaction. Never copy unless the data must outlive the txn.
- **M**ap-size first: you MUST size `mdb_env_set_mapsize` up front; the map is a hard ceiling. Outgrowing it is `MDB_MAP_FULL`, not auto-growth. This is the #1 footgun.
- **D**urable by design: no WAL, no recovery pass — the DB is always consistent because pages are never overwritten in place. The trade-off is durability flags (`MDB_NOSYNC`/`NOMETASYNC`), not corruption risk.
- **B**+tree, single writer: exactly one write txn at a time; keep it short. Readers never block writers and writers never block readers (MVCC snapshots).
- **Verified Code**: all access goes through a transaction; every `mdb_*` rc is checked; read pointers never escape their txn; the guide passes §2 before delivery.

---

## 2. Requirements (MANDATORY, auditable)

RFC-2119 keywords. IDs `LMDB-<TOPIC>-<NN>`. Each row has a binary gate; rows binding a shared rule cite its owner.

| ID | Requirement | Verify | Gate |
|----|-------------|--------|------|
| LMDB-MAP-01 | `mdb_env_set_mapsize` MUST be called before `mdb_env_open` with a ceiling above peak expected size | grep / review | mapsize set, not default |
| LMDB-MAP-02 | `MDB_MAP_FULL` MUST be handled (grow map or fail cleanly), never ignored | review / test with tiny map | handled, no data loss |
| LMDB-TXN-01 | All reads/writes MUST occur inside a txn; write txns MUST be short and serialized | review | no bare access |
| LMDB-TXN-02 | Every `mdb_*` rc MUST be checked; abort/cleanup on error (see `error-handling.md`) | review / `-Wall` | no unchecked rc |
| LMDB-READ-01 | A pointer from `mdb_get`/cursor MUST NOT be used after its txn ends; copy if it must outlive | review / `valgrind` | no use-after-txn |
| LMDB-READ-02 | Read txns MUST be short-lived or reset/renewed; no long-held reader pinning old pages | review / `mdb_reader_check` | no long readers |
| LMDB-RES-01 | Cursors, txns, and the env MUST be closed/freed on every path (incl. errors) | `valgrind --leak-check=full` | 0 leaks |
| LMDB-SEC-01 | DB file + `-lock` MUST have least-privilege perms; secrets not in plaintext (see `secure-coding.md`) | `stat -c '%a'` | 0600/0640, no plaintext secrets |
| LMDB-SEC-02 | Key/value sizes MUST be validated; keys ≤ 511 B (default) (see `secure-coding.md`) | review / test | bounded inputs |
| LMDB-TST-01 | Behavior MUST be test-first against a temp env; bugs get a regression test first (see `tdd.md`) | project test runner | exit 0, 0 skips |
| LMDB-DUR-01 | Durability flags (`MDB_NOSYNC`/`NOMETASYNC`/`MAPASYNC`) MUST be a documented, justified choice | review | rationale recorded |

> **Forbidden**: returning a read pointer past its txn; ignoring an `mdb_*` rc; opening with default 10 MiB map in production; holding a read txn open across long work; using `MDB_FIXEDMAP` without a hard reason (experimental, portability-fragile).

---

## 3. Architecture (what makes LMDB unique)

LMDB (Lightning Memory-Mapped Database, OpenLDAP/Symas) is an embedded KV store: a **single memory-mapped file** holding a **copy-on-write B+tree**, with **MVCC** snapshot isolation. ~10k lines of C, no dependencies, no background threads, no compaction.

```
mmap(file) → process address space
  B+tree: root → branch pages → leaf pages (keys sorted, byte-ordered by default)
  Read:  traverse O(log n) pages already in memory → return pointer into mmap (zero copy)
  Write: copy-on-write touched pages → update parents → atomic flip of one of two meta pages
         old pages stay readable by in-flight read snapshots (MVCC)
```

**Killer properties**
- **Zero-copy reads** — values are pointers directly into the mmap; no deserialization, no read buffer, no per-read lock. Read speed ≈ memory bandwidth.
- **Lock-free readers** — readers take a snapshot (a meta page + reader-table slot); they never acquire the write lock. Readers never block writers; writers never block readers.
- **Single-level store, crash-proof** — no WAL and no recovery process. A commit atomically flips the meta page; a crash leaves the previous committed state intact. Page checksums are not used; integrity rests on never overwriting live pages.
- **Full ACID** — atomic commit, snapshot isolation, durable on sync.

**Cost model**
- Writes are copy-on-write: touching a leaf rewrites the path to the root (low write amplification, 1–2×, but every write dirties ≥1 page per tree level).
- Free pages from old versions are reclaimed only once no reader still references them — hence §LMDB-READ-02.
- On-disk size only ever **grows to the high-water mark** and stays there; space is reused in place but the file does not shrink. To reclaim, compact via copy (`mdb_env_copy2(..., MDB_CP_COMPACT)` / `mdb_copy -c`).

---

## 4. C API — the canonical binding

Pointer-lifetime and memory-safety rules are owned by [`c.md`](guides://c.md). LMDB-specific shape:

```c
#include <lmdb.h>
#define E(expr) do { int rc_ = (expr); if (rc_) { \
    fprintf(stderr, "%s: %s\n", #expr, mdb_strerror(rc_)); goto fail; } } while (0)

MDB_env *env; MDB_dbi dbi; MDB_txn *txn;

E(mdb_env_create(&env));
E(mdb_env_set_mapsize(env, 10ULL<<30));   /* 10 GiB ceiling — set BEFORE open (LMDB-MAP-01) */
E(mdb_env_set_maxdbs(env, 8));            /* only if using named sub-DBs */
E(mdb_env_set_maxreaders(env, 126));      /* default 126; raise for many reader threads */
E(mdb_env_open(env, "data.mdb", MDB_NOSUBDIR, 0600));  /* file, not dir; tight perms */

/* write */
E(mdb_txn_begin(env, NULL, 0, &txn));      /* flags=0 → write txn (serialized) */
E(mdb_dbi_open(txn, NULL, 0, &dbi));       /* NULL = unnamed/main DB */
MDB_val k = { 5, "user1" }, v = { 5, "Alice" };  /* {mv_size, mv_data} */
E(mdb_put(txn, dbi, &k, &v, 0));
E(mdb_txn_commit(txn));                     /* commit frees the writer; abort on any error above */

/* read (zero-copy) */
E(mdb_txn_begin(env, NULL, MDB_RDONLY, &txn));
if (mdb_get(txn, dbi, &k, &v) == MDB_SUCCESS) {
    /* v.mv_data points INTO the mmap — valid only until txn ends (LMDB-READ-01) */
    fwrite(v.mv_data, 1, v.mv_size, stdout);
}
mdb_txn_abort(txn);   /* read txns are aborted, never committed */
```

Key facts the API hides:
- `MDB_val` is a plain `{size_t mv_size; void *mv_data;}` — no ownership. On `mdb_get` and cursors, `mv_data` is borrowed from the mmap.
- A write txn holds the single writer lock from `begin` to `commit`/`abort` — do no slow work (network, sleeps) inside it.
- `mdb_dbi_open` opens a sub-DB; open DBIs once and reuse the handle (don't open per-txn).

---

## 5. Transactions, MVCC & the reader discipline

Concurrency model is owned by [`parallelism.md`](guides://parallelism.md); the LMDB binding:

- **One writer, many readers.** A write txn is globally serialized (a process-wide mutex). Read txns are unlimited and lock-free, each pinned to the snapshot live at `begin`.
- **Reader table.** Each read txn occupies a slot (cap = `maxreaders`); `MDB_READERS_FULL` means slots exhausted or leaked. `mdb_reader_check` clears slots from dead processes.
- **The long-lived-read-transaction footgun (LMDB-READ-02).** A read txn pins every page version live at its snapshot. While it stays open, writers cannot reclaim those superseded pages, so the map keeps growing toward `MDB_MAP_FULL` even under steady-state churn. Symptoms: file balloons, free list never shrinks.
  - Fix: keep read txns short. For a long-running reader thread, use `mdb_txn_reset` + `mdb_txn_renew` to drop and re-acquire a fresh snapshot cheaply between units of work.
- **Nested (child) write txns** are supported (`mdb_txn_begin(env, parent, ...)`) for sub-transaction rollback; they are not compatible with `MDB_WRITEMAP`.

---

## 6. Errors & return codes (`error-handling.md` binding)

Strategy is owned by [`error-handling.md`](guides://error-handling.md). Every `mdb_*` returns `int`; `0`/`MDB_SUCCESS` is success, `>0` are system `errno`, `<0` are `MDB_*`. Decode with `mdb_strerror`. The ones to handle explicitly:

| Code | Meaning | Action |
|------|---------|--------|
| `MDB_NOTFOUND` | key absent | normal "miss" — not an error |
| `MDB_MAP_FULL` | map ceiling hit | abort txn; grow via `mdb_env_set_mapsize` (no readers/writers active) then retry; or fail (LMDB-MAP-02) |
| `MDB_MAP_RESIZED` | another process grew the map | call `mdb_env_set_mapsize(env, 0)` to adopt the new size, restart txn |
| `MDB_READERS_FULL` | reader slots exhausted | `mdb_reader_check`; raise `maxreaders`; fix leaked read txns |
| `MDB_TXN_FULL` | too many dirty pages in one txn | commit and split into smaller txns |
| `MDB_KEYEXIST` | duplicate under `MDB_NOOVERWRITE`/`MDB_APPEND` | expected signal, not a crash |
| `MDB_BAD_TXN` / `MDB_BAD_RSLOT` | txn used after abort / reader-slot misuse | bug — fix lifetime, don't retry |

Always `mdb_txn_abort` on any error inside a txn before returning.

---

## 7. Named databases, cursors & range scans

**Named sub-DBs.** One env can hold many named B+trees (set `maxdbs`, then `mdb_dbi_open(txn, "name", MDB_CREATE, &dbi)`). Use them for separate keyspaces or secondary indexes — all updated atomically within one write txn.

```c
mdb_dbi_open(txn, "users", MDB_CREATE, &users);
mdb_dbi_open(txn, "by_email", MDB_CREATE, &idx);   /* secondary index in same txn */
mdb_put(txn, users, &id, &record, 0);
mdb_put(txn, idx, &email, &id, 0);                 /* index → primary key; atomic together */
```

**Cursors & ordered scans.** Keys are stored in byte order (or a custom comparator), so range/prefix scans are the core read idiom:

```c
MDB_cursor *cur; MDB_val k = { plen, prefix }, v;
mdb_cursor_open(txn, dbi, &cur);
int rc = mdb_cursor_get(cur, &k, &v, MDB_SET_RANGE);   /* first key >= prefix */
while (rc == 0 && k.mv_size >= plen && memcmp(k.mv_data, prefix, plen) == 0) {
    /* use k/v (zero-copy) */
    rc = mdb_cursor_get(cur, &k, &v, MDB_NEXT);
}
mdb_cursor_close(cur);
```

Cursor ops: `MDB_FIRST/LAST/NEXT/PREV`, `MDB_SET` (exact), `MDB_SET_RANGE` (>=), `MDB_GET_CURRENT`; with `MDB_DUPSORT` add `MDB_NEXT_DUP/PREV_DUP` for multi-values per key. A cursor is bound to its txn — close before the txn ends.

**Append mode.** For bulk-loading keys already in sorted order, `mdb_put(..., MDB_APPEND)` skips the B+tree search and appends to the rightmost leaf — dramatically faster, near-zero rebalancing. Keys MUST be strictly increasing or you get `MDB_KEYEXIST`. Ideal for migrations and initial loads (batch ~10k puts per write txn to bound `MDB_TXN_FULL` and dirty-page memory).

---

## 8. Security (`secure-coding.md` binding)

Policy is owned by [`secure-coding.md`](guides://secure-coding.md). LMDB specifics:

- **No built-in encryption.** Encrypt at rest with the filesystem (LUKS/dm-crypt) or encrypt values in the application before `mdb_put`. Note: app-level encryption defeats zero-copy (you must decrypt into a buffer) and ordered scans over encrypted keys.
- **File permissions.** Open with `0600`/`0640`; lock down both `data.mdb` and `data.mdb-lock` (LMDB-SEC-01). With `MDB_NOSUBDIR` the lock file sits beside the data file.
- **Input bounds.** Default max key size is 511 bytes; validate key/value sizes before write (LMDB-SEC-02). Oversized keys → `MDB_BAD_VALSIZE`.
- **No network surface.** LMDB is in-process; there is no auth/wire layer to harden — the trust boundary is the OS file permissions and the calling process.

---

## 9. When LMDB shines vs. its limits

**Shines:** read-heavy and read-mostly workloads; embedded/edge and predictable single-process or single-host services; data that fits memory-ish (mmap + OS cache); workloads needing lock-free concurrent readers and crash-proof ACID with a tiny dependency footprint (OpenLDAP, Bitcoin Core chainstate, Knot DNS, Postfix, Samba).

**Limits / when to pick something else:**
- **Single writer** — write-concurrency-bound workloads suit RocksDB (LSM, sharded writes) — see [`rocksdb.md`](guides://rocksdb.md).
- **You must size the map up front** and handle `MDB_MAP_FULL`; unbounded growth needs active map management.
- **On-disk size = the largest the DB ever was** until you compact via copy — bursty churn leaves a big file.
- **Write amplification per-op** (path-to-root COW) makes very write-heavy, large-value workloads cheaper on an LSM store. For a simpler single-file ACID store with SQL, consider [`sqlite.md`](guides://sqlite.md); for the closest B+tree relative, [`leveldb.md`](guides://leveldb.md)/[`berkeleydb.md`](guides://berkeleydb.md).

---

## 10. Operations & tooling

```bash
mdb_stat -a data.mdb          # entries, depth, page counts, reader table
mdb_copy -c src.mdb dst.mdb   # hot copy + compact (reclaims high-water-mark slack)
mdb_dump data.mdb > dump.txt  # export to portable text
mdb_load data.mdb < dump.txt  # import
```

- **Hot backup:** `mdb_env_copy2(env, path, MDB_CP_COMPACT)` produces a consistent snapshot of a live env (readers/writers continue). A plain `cp` is only safe if no writer is active.
- **Map management:** grow the live map with `mdb_env_set_mapsize(env, bigger)` when no txn is open in the process; other processes adopt it on `MDB_MAP_RESIZED`.
- **Durability flags** (LMDB-DUR-01): default is fully synced. `MDB_NOMETASYNC` (skip meta fsync), `MDB_NOSYNC` (skip data fsync), `MDB_WRITEMAP`+`MDB_MAPASYNC` (writable mmap, async flush) trade durability/crash-window for throughput — document the choice.
- **Memory safety:** run integration tests under `valgrind --leak-check=full` (LMDB-RES-01); the zero-copy API makes use-after-txn bugs easy to write and hard to spot.
- **System tuning** (read [`performance.md`](guides://performance.md) for the model): on Linux raise `vm.max_map_count`, lower `vm.swappiness` for DB hosts; mmap size may exceed RAM (the file is sparse — pages fault in on demand).

---

## 11. Quick Reference

```bash
gcc -O2 app.c -llmdb -o app        # build (C)
mdb_stat -a data.mdb               # inspect
mdb_copy -c data.mdb compact.mdb   # compact
valgrind --leak-check=full ./app   # LMDB-RES-01
```

Lifecycle: `env_create → set_mapsize → set_maxdbs/maxreaders → env_open → (txn_begin → dbi_open/get/put/cursor → commit|abort) → env_close`.

---

## 12. Deployment Checklist

Generated from §2 — one box per requirement ID.

- [ ] LMDB-MAP-01 — `mdb_env_set_mapsize` set before open, ceiling above peak
- [ ] LMDB-MAP-02 — `MDB_MAP_FULL` handled, no data loss
- [ ] LMDB-TXN-01 — all access in txns; write txns short & serialized
- [ ] LMDB-TXN-02 — every `mdb_*` rc checked, clean abort on error
- [ ] LMDB-READ-01 — no read pointer escapes its txn (valgrind clean)
- [ ] LMDB-READ-02 — no long-lived readers; reset/renew for long threads
- [ ] LMDB-RES-01 — cursors/txns/env freed on all paths; 0 leaks
- [ ] LMDB-SEC-01 — DB + lock file least-privilege perms, no plaintext secrets
- [ ] LMDB-SEC-02 — key/value sizes validated (keys ≤ 511 B)
- [ ] LMDB-TST-01 — tests pass, bugs have regression tests
- [ ] LMDB-DUR-01 — durability flags chosen & documented
- [ ] Agent ran §11 build + valgrind and documented any fixes

---
**End of LMDB Guidelines**
