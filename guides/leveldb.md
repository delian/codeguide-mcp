# LevelDB Development Guidelines
Mandatory standards for LevelDB: the original embedded LSM key-value library from Google. Single-process ordered KV store, Put/Get/Delete/WriteBatch, iterators, snapshots, custom comparators. LevelDB 1.23, C++17.

---
name: leveldb
title: LevelDB Development Guidelines
version: 2.0
last_reviewed: 2026-06-05
kind: datastore
tools: [leveldb@1.23, c++17, snappy]
requires:
  - secure-coding
  - error-handling
recommends:
  - performance
  - cpp
  - rocksdb
provides:
  - leveldb-lsm
  - embedded-kv
  - leveldb-iterators
  - leveldb-limitations
---

> 🧭 Authored per [`CONVENTIONS.md`](guides://CONVENTIONS.md): shared concerns are referenced, not restated. This guide covers only what is unique to LevelDB.

---

## 0. Prerequisites & References

> 📎 **REQUIRED — fetch & apply first:**
> - [`secure-coding.md`](guides://secure-coding.md) — secrets, supply chain, input validation, CVE policy. *(LevelDB binding: it has **no built-in encryption or auth** — encrypt values app-side and restrict DB-directory file permissions; see §7.)*
> - [`error-handling.md`](guides://error-handling.md) — error strategy & propagation. *(LevelDB binding: every operation returns a `leveldb::Status`; check `.ok()` / `.IsNotFound()` on **every** call — see §6.)*

> 📎 **RECOMMENDED — fetch when the task touches them:**
> - [`cpp.md`](guides://cpp.md) — the API language: RAII, smart pointers, `-std=c++17`, no leaked `Iterator*`/`Snapshot*`.
> - [`performance.md`](guides://performance.md) — measurement-first tuning methodology *(binding: §8 options & `db_bench`)*.
> - [`rocksdb.md`](guides://rocksdb.md) — the high-performance fork. **If you need column families, transactions, parallel/tunable compaction, bloom-tuned reads, merge operators, or backup engines, use RocksDB, not LevelDB** (see §9). RocksDB owns advanced LSM tuning.

> 📎 **SEE ALSO:** [`tdd.md`](guides://tdd.md) — test-first & regression-test-before-fix (use a temp-dir DB per test). Other embedded KV options: [`sqlite.md`](guides://sqlite.md) (SQL/transactions), [`berkeleydb.md`](guides://berkeleydb.md).

---

## 1. Core Philosophies

LevelDB-specific principles only. Security, error strategy, and architecture come from §0.

- **Minimal by design.** LevelDB is a small, stable, dependency-light ordered KV library (~Snappy optional). It does *not* evolve much — that is a feature. Choose it when you want a tiny, predictable embedded store; choose RocksDB when you need features or raw throughput (§9).
- **Embedded, single-process.** No network layer, no server, no client. **Exactly one process may open a given DB directory at a time** (an exclusive `LOCK` file enforces this). Concurrency *within* that process is the application's job.
- **Single writer, many readers.** Writes serialize through an internal mutex; reads scale across threads. Use snapshots for consistent point-in-time reads.
- **Ordered keys are the data model.** Keys are byte strings sorted by a comparator; range scans and prefix iteration are first-class. Design keys for the scans you need (§5).
- **Atomic batches, not transactions.** `WriteBatch` is the only atomic unit (all-or-nothing). There is **no** multi-step ACID transaction, no isolation between a read and a later write.
- **Durability is a per-write choice.** `WriteOptions.sync` trades throughput for crash durability (§4).

**Verified Code**: agent-generated LevelDB code MUST build, check every `Status`, free every `Iterator`/`Snapshot`, and pass tests before delivery.

---

## 2. Requirements (MANDATORY, auditable)

RFC-2119 keywords. IDs `LDB-<TOPIC>-<NN>`. Each row has a binary gate; rows binding a shared rule cite its owner.

| ID | Requirement | Verify | Gate |
|----|-------------|--------|------|
| LDB-ERR-01 | Every `DB`/`WriteBatch`/iterator op MUST check `Status` (`.ok()`/`.IsNotFound()`) before using results (see `error-handling.md`) | grep for unchecked `db_->Get/Put/Write/Open`; review | no ignored Status |
| LDB-ERR-02 | Iterators MUST be checked with `it->status()` after the loop | review / lint | every loop checks status |
| LDB-RES-01 | Every `NewIterator()` and `GetSnapshot()` MUST be released (`delete it` / `ReleaseSnapshot`) — prefer RAII (see `cpp.md`) | ASan/LSan run, review | no leaks |
| LDB-CONC-01 | At most ONE process MUST open a DB dir; intra-process writes MUST be serialized or routed to one writer | design review; open returns IO error if locked | single opener |
| LDB-DUR-01 | Writes whose loss is unacceptable MUST set `WriteOptions.sync = true` (or batch + periodic sync) | review of `WriteOptions` | critical writes synced |
| LDB-SEC-01 | DB directory MUST be `0700`, files `0600`; sensitive values MUST be encrypted app-side (no built-in encryption) (see `secure-coding.md`) | `stat` perms; review | perms set, no plaintext secrets |
| LDB-SEC-02 | Key/value sizes from untrusted input MUST be bounded; scans MUST cap result count | review | bounds enforced |
| LDB-CMP-01 | A custom `Comparator::Name()` MUST be stable for the life of the DB (changing it corrupts ordering) | review; open fails on mismatch | name unchanged |
| LDB-TST-01 | Each test MUST use an isolated temp-dir DB and clean up; bugs get a regression test first (see `tdd.md`) | run test suite | exit 0, isolated |
| LDB-BLD-01 | Code MUST build clean at `-std=c++17 -O2 -Wall` and link `-lleveldb` | compile | exit 0 |
| LDB-FIT-01 | If the workload needs column families / transactions / heavy write tuning, RocksDB MUST be chosen over LevelDB (see `rocksdb.md`) | design review | justified choice |

> **Forbidden**: ignoring a returned `Status`; leaking iterators/snapshots; opening one DB dir from two processes; relying on `WriteBatch` for cross-operation isolation; changing a live comparator's `Name()`; storing secrets unencrypted.

---

## 3. Architecture: the LSM tree (what LevelDB owns)

LevelDB is an ordered map from byte-string keys to byte-string values, built on a **Log-Structured Merge tree**. Understanding the write/read path is the core value of this guide.

**Write path:** `Put`/`Delete` → append to **WAL** (write-ahead log, crash recovery) → insert into the **memtable** (in-RAM sorted skip list). When the memtable fills (`write_buffer_size`, default 4 MB) it becomes an **immutable memtable** and a fresh one takes writes; a background thread flushes the immutable memtable to a **Level-0 SST** (Sorted String Table — an immutable on-disk file). A delete is a *tombstone* record, not an in-place erase.

**Compaction:** a **single** background thread merges SSTs downward. L0 files may have overlapping key ranges; **L1–L6 are non-overlapping and sorted**, each level ~10× the previous. Compaction reclaims tombstones/overwrites and keeps read cost bounded. The cost is **write amplification** (data is rewritten several times) — the LSM trade for fast sequential writes.

**Read path:** check memtable → immutable memtable → each L0 SST (may be several, since they overlap) → binary-search one file per level L1–L6. A per-SST **bloom filter** (`options.filter_policy = NewBloomFilterPolicy(10)`) skips files that cannot hold the key, cutting disk reads for misses. A **block cache** (LRU over decompressed SST blocks) serves hot reads from RAM.

```
write → WAL → memtable → (full) → immutable memtable → flush → L0 SST
                                                          ↓ compaction (1 thread)
                                              L1 … L6  (sorted, non-overlapping, 10×/level)
read: memtable → imm → L0(scan all) → L1..L6(bsearch 1/level), gated by bloom filter + block cache
```

Snappy compression (per block) is on by default and is usually a net win (less I/O). Disable only when CPU-bound and space is irrelevant.

---

## 4. Core API (C++)

`#include <leveldb/db.h>`. All ops return `leveldb::Status`; obey §0 `error-handling`. Manage handles with RAII (§0 `cpp`).

```cpp
leveldb::DB* raw = nullptr;
leveldb::Options options;
options.create_if_missing = true;
// options.error_if_exists = true;          // refuse to open an existing DB
options.block_cache  = leveldb::NewLRUCache(64 * 1024 * 1024);   // 64MB hot-read cache
options.filter_policy = leveldb::NewBloomFilterPolicy(10);       // ~10 bits/key → fewer disk reads
leveldb::Status s = leveldb::DB::Open(options, "/var/lib/app/db", &raw);
if (!s.ok()) { /* handle: corruption, lock held by another process, IO */ }
std::unique_ptr<leveldb::DB> db(raw);       // RAII; destructor closes the DB
// NOTE: filter_policy and block_cache must outlive the DB — own them alongside it.
```

**Point operations** — `WriteOptions`/`ReadOptions` carry the durability/consistency knobs:
```cpp
leveldb::WriteOptions w;            // w.sync = true → fsync WAL before returning (durable, slower)
db->Put(w, key, value);
db->Delete(w, key);                 // writes a tombstone

std::string value;
leveldb::Status g = db->Get(leveldb::ReadOptions(), key, &value);
if (g.IsNotFound()) { /* absent — NOT an error */ }
else if (!g.ok())   { /* real failure */ }
```

**Atomic batch** — the only atomicity primitive; apply many mutations all-or-nothing, and far faster than N separate writes:
```cpp
leveldb::WriteBatch batch;
batch.Put("user:1:name", "Alice");
batch.Put("email:alice@x.com", "1");   // secondary index, kept consistent with the row
batch.Delete("user:1:tmp");
leveldb::Status s = db->Write(w, &batch);   // single WAL append + memtable insert
```
Bulk loading: accumulate ~1k–10k ops per `WriteBatch`, `sync=false`, then one final synced write.

**`sync` semantics:** `sync=false` (default) returns once the write is in the OS buffer — a process crash is safe but an OS/power crash can lose the last writes. `sync=true` fsyncs the WAL first — durable, ~100× slower per call. Batch many ops behind one synced write to get both.

---

## 5. Iterators, range scans & snapshots

Sorted keys make iteration the headline feature. Iterators are owned by the caller — always release them (LDB-RES-01) and check `status()` (LDB-ERR-02).

```cpp
std::unique_ptr<leveldb::Iterator> it(db->NewIterator(leveldb::ReadOptions()));
for (it->Seek(prefix); it->Valid(); it->Next()) {
    if (!it->key().starts_with(prefix)) break;        // prefix scan
    process(it->key(), it->value());
}
if (!it->status().ok()) { /* IO/corruption surfaced here, not mid-loop */ }
```
- `Seek(k)` positions at the first key ≥ `k`; `SeekToFirst()`/`SeekToLast()`; `Next()`/`Prev()` walk in comparator order (reverse iteration is slower).
- Range `[start,end)`: `Seek(start)`, stop when `it->key() >= end`.
- For pure counting/bulk scans set `ReadOptions.fill_cache = false` so the scan doesn't evict hot blocks.
- `key()`/`value()` return `Slice`s into the iterator's buffer — copy out before `Next()` if you keep them.

**Snapshots** give a consistent point-in-time view without blocking writers:
```cpp
const leveldb::Snapshot* snap = db->GetSnapshot();   // RAII-wrap this
leveldb::ReadOptions ro; ro.snapshot = snap;
db->Get(ro, k1, &v1); db->Get(ro, k2, &v2);          // both see the same instant
db->ReleaseSnapshot(snap);                            // MUST release — held snapshots pin SSTs and bloat disk
```
An iterator created without a snapshot is implicitly consistent for its own lifetime. Long-held snapshots/iterators delay compaction cleanup — release promptly.

**Custom comparators** change key ordering (e.g. big-endian numeric instead of lexicographic). Subclass `leveldb::Comparator`, set `options.comparator`. **`Name()` is baked into the DB at creation** — reopening with a different name fails by design (LDB-CMP-01), so version the name if the ordering ever changes. Tip: encode integers/timestamps big-endian so default bytewise order already sorts them.

---

## 6. Operations, monitoring & recovery

- **Properties:** `db->GetProperty("leveldb.stats", &out)`, `"leveldb.sstables"`, `"leveldb.num-files-at-level<N>"`, `"leveldb.approximate-memory-usage"`. Many L0 files (>4–8) signals compaction falling behind → write stalls.
- **Sizes:** `GetApproximateSizes(ranges, n, sizes)` for capacity planning.
- **Manual compaction:** `db->CompactRange(nullptr, nullptr)` forces a full compaction (drops tombstones, flattens levels) — useful after bulk deletes; expensive, run off-peak.
- **Corruption:** `options.paranoid_checks = true` and `ReadOptions.verify_checksums = true` for critical data. Recover a damaged DB with `leveldb::RepairDB(path, Options())` (salvages what it can — may drop unrecoverable data; back up first).
- **Backup:** the DB must be **closed** (or a filesystem-snapshot taken) before copying its directory — there is no online hot-backup engine (RocksDB has one; §9). For a logical export, iterate a snapshot and stream key/value pairs.

---

## 7. Security binding

Policy is owned by [`secure-coding.md`](guides://secure-coding.md). LevelDB specifics:

- **No encryption at rest, no access control.** Anyone who can read the directory reads the data. Use filesystem encryption (LUKS/dm-crypt) for whole-disk, and/or encrypt sensitive **values** app-side (e.g. AES-256-GCM) before `Put`. Keys are not encrypted and leak structure — don't put secrets in key names.
- **File permissions:** create the DB dir `0700` and files `0600` (LDB-SEC-01); run the process as a non-root, least-privilege user.
- **Untrusted input:** bound key and value sizes and cap scan result counts to prevent memory-exhaustion DoS (LDB-SEC-02). Treat stored bytes as untrusted on read-back.

---

## 8. Performance binding

Methodology (measure first, one change at a time, `db_bench`) is owned by [`performance.md`](guides://performance.md). The few knobs LevelDB exposes:

| Option | Effect | Write-heavy | Read-heavy |
|---|---|---|---|
| `write_buffer_size` | bigger memtable → fewer L0 files, less compaction | 32–64 MB | default 4 MB |
| `block_cache` (LRU) | caches decompressed blocks; biggest read lever | modest | 25–50% of RAM |
| `block_size` | larger = better scans, smaller = better point reads | 32–64 KB | 4–16 KB |
| `max_open_files` | SST file handles kept open | raise vs `ulimit -n` | high (e.g. 5000) |
| `filter_policy` | bloom filter → skip SSTs on misses | yes | **yes (essential)** |
| `compression` | Snappy (default) trades CPU for I/O | keep | keep unless CPU-bound |

LevelDB's compaction is **single-threaded and not tunable** — sustained heavy write workloads will stall here. That ceiling is the main reason to move to RocksDB (§9). Avoid storing large blobs as values (bloats memtable/cache and slows compaction) — store blobs externally and keep only metadata in LevelDB.

---

## 9. LevelDB vs RocksDB (and when to leave)

RocksDB is Meta's fork of LevelDB with the same simple core API. **Reach for [`rocksdb.md`](guides://rocksdb.md) instead of LevelDB when you need any of:**

| Need | LevelDB | RocksDB |
|---|---|---|
| Column families (logical partitions) | ❌ | ✅ |
| Transactions (optimistic/pessimistic) | ❌ (WriteBatch only) | ✅ |
| Parallel / leveled+universal / tunable compaction | ❌ single thread | ✅ |
| Merge operators, prefix bloom, multi-compression (zstd/lz4) | ❌ | ✅ |
| Online backup engine, checkpoints | ❌ (close-and-copy) | ✅ |
| Active upstream development | minimal | active |

**Stay on LevelDB when** you want a tiny, stable, low-dependency embedded ordered KV store with modest write rates and no need for the above — e.g. local app state, caches, indexes, single-node metadata. It still ships inside Chromium (IndexedDB), Bitcoin Core, and go-ethereum.

Migration is mechanical: both share `Put/Get/Delete/WriteBatch/Iterator/Snapshot`; iterate the old DB and `Write` batched into the new one. **Other directions:** [`sqlite.md`](guides://sqlite.md) if you need SQL or real transactions; LMDB if you need read-latency-optimized B+tree with low write amplification.

---

## 10. Install & build

```bash
# Debian/Ubuntu
sudo apt-get install -y libleveldb-dev libsnappy-dev
# macOS
brew install leveldb
```
```cmake
find_package(leveldb REQUIRED)
target_link_libraries(myapp leveldb::leveldb)   # link Snappy too if built with it
```
```bash
g++ -std=c++17 -O2 -Wall app.cpp -lleveldb -lpthread -o app   # LDB-BLD-01
```
Bindings: Python `plyvel`, Go `syndtr/goleveldb`, Node `level`. They wrap the same model — the Status/iterator/snapshot/`sync` semantics above all carry over.

---

## 11. Deployment Checklist

Generated from §2 — one box per requirement ID.

- [ ] LDB-ERR-01/02 — every Status checked; iterator `status()` checked after each loop
- [ ] LDB-RES-01 — every iterator/snapshot released (RAII); ASan/LSan clean
- [ ] LDB-CONC-01 — exactly one process opens the DB dir; intra-process writes serialized
- [ ] LDB-DUR-01 — critical writes use `sync=true` (or batched + synced)
- [ ] LDB-SEC-01/02 — dir `0700`/files `0600`, secrets encrypted app-side, input/scan bounds enforced
- [ ] LDB-CMP-01 — custom comparator `Name()` stable for the DB's life
- [ ] LDB-TST-01 — tests use isolated temp DBs; bugs got a regression test first
- [ ] LDB-BLD-01 — builds clean `-std=c++17 -O2 -Wall`, links `-lleveldb`
- [ ] LDB-FIT-01 — RocksDB chosen if column families/transactions/heavy-write tuning needed

---

**End of LevelDB Guidelines**
