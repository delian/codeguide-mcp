# RocksDB Development Guidelines
Mandatory standards for the embedded RocksDB LSM key-value engine: tuning the amplification tradeoff, column families, compaction, transactions, backups. RocksDB 9.x, C++17, CMake, gtest.

---
name: rocksdb
title: RocksDB Development Guidelines
version: 2.0
last_reviewed: 2026-06-05
kind: datastore
tools: [rocksdb@9, cmake@3.16, gtest, db_bench, ldb]
requires:
  - secure-coding
  - error-handling
recommends:
  - performance
  - cpp
  - leveldb
  - parallelism
provides:
  - lsm-tree
  - rocksdb-tuning
  - column-families
  - amplification-tradeoffs
  - compaction
---

> 🧭 Authored per [`CONVENTIONS.md`](guides://CONVENTIONS.md): shared concerns are referenced, not restated. This guide covers only what is unique to RocksDB.

---

## 0. Prerequisites & References

Fetch and apply these **before** generating RocksDB code. Their rules are assumed below and not repeated.

> 📎 **REQUIRED — fetch & apply first:**
> - [`secure-coding.md`](guides://secure-coding.md) — input validation, secrets, supply chain. *(RocksDB binding: it is an **in-process library with no auth and no built-in encryption** — access control, encryption-at-rest, and key-size/value-size limits are the application's job, see §11.)*
> - [`error-handling.md`](guides://error-handling.md) — error strategy & propagation. *(RocksDB binding: **every** API call returns a `rocksdb::Status`; check `.ok()` / `.IsNotFound()` on every Put/Get/Write/Iterator, see §2.)*

> 📎 **RECOMMENDED — fetch when the task touches them:**
> - [`performance.md`](guides://performance.md) — the read/write/space amplification tradeoff and tuning method (§9 binds it to RocksDB knobs).
> - [`cpp.md`](guides://cpp.md) — RAII, smart pointers, the primary API language. *(Wrap `DB*`, `Iterator*`, `Snapshot*`, `Transaction*`, and CF handles in RAII; never leak them.)*
> - [`parallelism.md`](guides://parallelism.md) — RocksDB is multi-writer with background compaction/flush threads; concurrency policy lives here.
> - [`leveldb.md`](guides://leveldb.md) — RocksDB's ancestor; the API and LSM concepts are shared, see §12.

> 📎 **SEE ALSO:** [`tdd.md`](guides://tdd.md) · [`observability.md`](guides://observability.md) · [`cmake.md`](guides://cmake.md)

---

## 1. Core Philosophies: LSM-FIRST

RocksDB-specific principles only. TDD, security, error handling, performance, and concurrency policy come from §0.

- **L**SM-shaped data: design keys for sorted range scans and prefix seeks; embrace memtable→WAL→SST→compaction. Never fight the LSM with random-update-heavy schemas better served by a B-tree (see §12).
- **S**tatus-checked: treat `rocksdb::Status` like an error union — `.ok()` checked on every operation (policy: `error-handling.md`).
- **M**easure amplification: every tuning decision moves a point on the **read/write/space amplification triangle** (§9). Optimize for *your* workload's bottleneck, never blindly.
- **F**amilies over prefixes for heterogeneous data: use column families for independent workloads (TTL vs archive vs hot) sharing one WAL (§6).
- **I**dempotent batches & transactions: multi-key atomicity via `WriteBatch`; cross-key invariants via optimistic/pessimistic transactions (§8).
- **R**eproducible builds: pin the RocksDB version and compression libraries; a DB written with one option set must reopen with a compatible one (comparator/merge-operator/prefix-extractor names are persisted).
- **S**napshot for consistency: use snapshots/checkpoints for consistent reads and hot backups, never an open-DB file copy.
- **T**uned, not defaulted: defaults target a generic SSD; production sets block cache, bloom filters, write buffer, and compaction style deliberately.

**Verified Code**: Agent-generated RocksDB code MUST build, check every `Status`, and pass the §2 gates before delivery.

---

## 2. Requirements (MANDATORY, auditable)

RFC-2119 keywords. IDs `ROCKS-<TOPIC>-<NN>`. Each row has a binary gate; rows binding a shared rule cite its owner.

| ID | Requirement | Verify | Gate |
|----|-------------|--------|------|
| ROCKS-TST-01 | Every feature MUST be test-first against a temp DB (see `tdd.md`) | `ctest` / `pytest` | exit 0, 0 skips |
| ROCKS-TST-02 | Each bug MUST get a regression test before the fix (see `tdd.md`) | test suite | failing→passing |
| ROCKS-ERR-01 | Every Put/Get/Write/Merge/Iterator/Commit MUST check `Status` (see `error-handling.md`) | review / lint | no unchecked Status |
| ROCKS-ERR-02 | `Get` MUST distinguish `IsNotFound()` from real errors | review | both handled |
| ROCKS-RES-01 | `DB`, `Iterator`, `Snapshot`, `Transaction`, CF handles MUST be RAII-owned/released (see `cpp.md`) | review / ASan | no leaks |
| ROCKS-DUR-01 | Durability-critical writes MUST set `WriteOptions::sync` or use the WAL deliberately | review | documented per write path |
| ROCKS-TUNE-01 | Block cache, bloom filter, write buffer & compaction style MUST be set explicitly (not defaults) | review of `Options` | all set |
| ROCKS-AMP-01 | Chosen tuning MUST name which amplification it optimizes & accepts (see `performance.md`) | ADR / comment | tradeoff stated |
| ROCKS-CF-01 | Column families MUST be opened with the SAME descriptors used to create them | open succeeds | no `InvalidArgument` |
| ROCKS-BAK-01 | Backups MUST use BackupEngine/Checkpoint, never a copy of an open DB dir | review | no live-dir copy |
| ROCKS-SEC-01 | Key/value sizes validated; no untrusted unbounded scans (see `secure-coding.md`) | review | limits enforced |
| ROCKS-SEC-02 | Encryption-at-rest provided by app/FS layer; DB dir perms `0700` (see `secure-coding.md`) | `stat` / review | enforced |
| ROCKS-VER-01 | RocksDB version + compression libs pinned in build | build config | pinned |

> **Forbidden**: ignoring a returned `Status`; copying a live DB directory as a "backup"; reopening with a different comparator/merge-operator/prefix-extractor name; shipping default `Options` into production; storing large blobs (>~1 MB) as values (§11).

---

## 3. Verification Protocol

Run before presenting code. Fix → re-run until green.

```bash
cmake --build build -j                       # ROCKS-VER-01: builds against pinned RocksDB
ctest --test-dir build --output-on-failure   # ROCKS-TST-01/02
build/app --asan  # or: run tests under -fsanitize=address,leak   # ROCKS-RES-01
ldb --db=/path/to/db check_consistency       # post-write integrity
```

The *why* behind each gate lives in its §0 owner; do not re-derive it here.

---

## 4. The Engine: LSM Architecture & Embedded Model

RocksDB is an **embedded, in-process** ordered key→value store forked from LevelDB and tuned for SSD/flash and high write throughput (parallel compaction, column families, transactions, rich tuning). No network, no query language, no auth — it is a library, not a server.

```
WRITE PATH                          READ PATH (newest → oldest)
 Put/Write                           Get
   │                                  │
   ├─► WAL (durability, fsync opt) ── ├─► active MemTable      (fastest)
   │                                  ├─► immutable MemTables
   └─► active MemTable (skiplist)     ├─► Block cache
        │ when full → immutable       ├─► L0 SSTs (overlapping ranges)
        ▼ flush                       └─► L1..Ln SSTs (non-overlapping,
     L0 SST files                          ~10× per level; bloom filter
        │ background compaction            short-circuits absent keys)
        ▼
     L1..Ln (leveled / universal / FIFO)
```

- **MemTable**: in-memory sorted skiplist; absorbs writes at O(log n). When full it becomes immutable and is flushed to an L0 SST.
- **WAL**: write-ahead log gives crash recovery. `WriteOptions::sync=true` fsyncs per write (durable, slow); `false` relies on OS/periodic flush (fast, loses last writes on power loss); `disableWAL=true` only for re-derivable data.
- **SST files**: immutable, sorted, block-structured (data blocks + index + bloom/ribbon filter). L0 files may overlap; L1+ are partitioned and non-overlapping.
- **Compaction**: background threads merge SSTs, drop deletes/overwrites, and push data down levels — the source of write amplification and the lever for read/space amplification (§9).
- **Embedded tradeoffs**: microsecond latency, no serialization cost, simple deploy, process-level isolation — but single-machine, no replication, app owns concurrency and durability choices.

```cpp
#include <rocksdb/db.h>
auto opts = rocksdb::Options{};
opts.create_if_missing = true;
rocksdb::DB* raw = nullptr;
rocksdb::Status s = rocksdb::DB::Open(opts, "/var/lib/app/db", &raw);
if (!s.ok()) { /* handle: corruption, no space, busy lock — see error-handling.md */ }
std::unique_ptr<rocksdb::DB> db(raw);   // RAII (ROCKS-RES-01)
```

---

## 5. The Options Surface & Read/Write Path

The options surface is vast; production must set these deliberately (ROCKS-TUNE-01). The high-leverage knobs:

```cpp
rocksdb::Options o;
rocksdb::BlockBasedTableOptions t;

// --- Read path ---
t.block_cache = rocksdb::NewLRUCache(512ULL << 20);   // hot blocks; 30–50% RAM
t.cache_index_and_filter_blocks = true;               // bound metadata memory
t.pin_l0_filter_and_index_blocks_in_cache = true;
t.filter_policy.reset(rocksdb::NewBloomFilterPolicy(10));   // 10 bits/key ≈ ~1% FP
// Ribbon filter: NewRibbonFilterPolicy(10) — more space-efficient, slightly more CPU.
t.block_size = 16 * 1024;                             // 4K point-lookup, 16–64K scans
o.table_factory.reset(rocksdb::NewBlockBasedTableFactory(t));

// --- Write path ---
o.write_buffer_size = 64ULL << 20;                    // memtable size; bigger = fewer L0 files
o.max_write_buffer_number = 3;
o.compression_per_level = {                           // cheap/none up top, strong at the bottom
    rocksdb::kNoCompression, rocksdb::kNoCompression,
    rocksdb::kLZ4Compression, rocksdb::kLZ4Compression, rocksdb::kZSTD};
o.bottommost_compression = rocksdb::kZSTD;            // most data lives here

// --- Background work (see parallelism.md) ---
o.max_background_jobs = 6;                             // flush + compaction threads
o.max_subcompactions = 2;                             // parallelize one compaction
o.rate_limiter.reset(rocksdb::NewGenericRateLimiter(50ULL << 20));  // cap background I/O
o.statistics = rocksdb::CreateDBStatistics();         // observability (see observability.md)
o.create_if_missing = true;
```

**The read/write/space amplification triangle** (the central tuning concept — policy in `performance.md`):
- **Write amplification (WA)** = bytes written to disk ÷ bytes from the app (typically 10–30× under leveled compaction). Driven by how often compaction rewrites data.
- **Read amplification (RA)** = SST/blocks touched per Get. Bloom filters and the block cache crush RA for point lookups; range scans pay per overlapping level.
- **Space amplification (SA)** = disk bytes ÷ live bytes. Stale versions awaiting compaction inflate it.
- You can't minimize all three — **pick the one your workload is bottlenecked on and trade away another** (ROCKS-AMP-01). See §9.

Key read knobs: `ReadOptions::verify_checksums` (off = faster, less safe), `fill_cache`, `async_io` for MultiGet, `prefix_same_as_start` + `total_order_seek=false` to use the prefix bloom. Prefer `MultiGet` over N `Get`s — it coalesces I/O.

---

## 6. Column Families

Column families (CFs) are independent LSM trees in one DB — separate memtables, SSTs, and compaction, **sharing a single WAL** so writes across CFs are atomic in one `WriteBatch`. Use them to give heterogeneous data different tuning (e.g. read-optimized "hot", space-optimized "archive", FIFO-TTL "events") instead of overloading key prefixes.

```cpp
rocksdb::ColumnFamilyOptions hot;      // big cache, low L0 trigger
rocksdb::ColumnFamilyOptions archive;  // kZSTD bottommost, level 9
std::vector<rocksdb::ColumnFamilyDescriptor> cfs = {
    {rocksdb::kDefaultColumnFamilyName, {}}, {"hot", hot}, {"archive", archive}};
std::vector<rocksdb::ColumnFamilyHandle*> handles;
rocksdb::DB* raw;
auto s = rocksdb::DB::Open(db_opts, path, cfs, &handles, &raw);  // create_missing_column_families=true
// ROCKS-CF-01: you MUST list ALL existing CFs (with compatible options) on every open, or Open fails.
db->Put({}, handles[1], key, value);    // write into "hot"
// Release handles via db->DestroyColumnFamilyHandle(h) before closing the DB (ROCKS-RES-01).
```

---

## 7. Iterators, Snapshots, Prefix Extractors & Merge Operators

**Iterators** give sorted forward/backward traversal and range/prefix scans. Always check `it->status()` after the loop and RAII-own the iterator.

```cpp
std::unique_ptr<rocksdb::Iterator> it(db->NewIterator(ropts));
for (it->Seek(prefix); it->Valid() && it->key().starts_with(prefix); it->Next()) { /* … */ }
if (!it->status().ok()) { /* I/O or corruption */ }
// Reverse: it->SeekForPrev(end); … it->Prev();
```

**Snapshots** pin a consistent point-in-time view across reads; set `ReadOptions::snapshot` and `ReleaseSnapshot` when done (RAII wrapper). Cheap (a sequence number), but a long-held snapshot pins obsolete versions and *raises space amplification* — release promptly.

**Prefix extractors** (`options.prefix_extractor = NewFixedPrefixTransform(n)` or `NewCappedPrefixTransform`) enable a **prefix bloom filter** so prefix seeks skip SSTs that can't contain the prefix. Persisted by name; once set, scans with `total_order_seek=false` use it. Use `total_order_seek=true` to bypass for a full-order scan.

**Merge operators** turn read-modify-write into a single write, deferring the merge to read/compaction time — ideal for counters, append-lists, and high-contention aggregates (vs the hot-key anti-pattern, §11). The operator is registered by name in `Options` and persisted; reopening with a different name breaks the DB.

```cpp
class CounterMerge : public rocksdb::AssociativeMergeOperator { /* sum existing + delta */ };
o.merge_operator = std::make_shared<CounterMerge>();
db->Merge({}, "counter", "1");   // no read needed; resolved on Get/compaction
```

---

## 8. Transactions & Atomicity

Three escalating levels of atomicity:

1. **`WriteBatch`** — atomic, all-or-nothing group of Put/Delete/Merge (possibly across CFs via the shared WAL). No conflict detection. The default tool for multi-key consistency.
2. **Optimistic transactions** (`OptimisticTransactionDB`) — no locks; conflicts detected at `Commit()`, which fails if a touched key changed. Best under **low contention** — cheap reads, retry on conflict.
3. **Pessimistic transactions** (`TransactionDB`) — row-level locks via `GetForUpdate`; `transaction_lock_timeout` and `deadlock_detect` configurable. Best under **high contention** — blocks instead of retrying, with deadlock detection.

```cpp
// Optimistic: retry loop on commit conflict
auto* txn = otxn_db->BeginTransaction({});
txn->Get(ropts, k, &v); /* … */ txn->Put(k, v2);
rocksdb::Status s = txn->Commit();   // Status::Busy → retry; else handle
delete txn;

// Pessimistic: lock-on-read
txn->GetForUpdate(ropts, k, &v);     // acquires lock; other writers wait/timeout
```

Choose optimistic for read-mostly / rare-conflict workloads, pessimistic when many writers contend on the same keys. Both honor snapshot isolation for reads.

---

## 9. Compaction Strategies & Tuning the Tradeoff

Compaction style is the biggest single lever on the amplification triangle (§5). Pick by workload (ROCKS-AMP-01):

| Style | Optimizes | Pays | Use when |
|-------|-----------|------|----------|
| **Leveled** (`kCompactionStyleLevel`, default) | low space amp, low read amp | high **write** amp (10–30×) | read-heavy / balanced, space-conscious |
| **Universal** (`kCompactionStyleUniversal`) | low **write** amp | high **space** amp (≤2×+) | write-heavy / ingest, transient data |
| **FIFO** (`kCompactionStyleFIFO`) | near-zero compaction | no historical reads | TTL/time-series, cache, logs |

**Leveled knobs** (default): `level0_file_num_compaction_trigger` (L0→L1, default 4), `level0_slowdown_writes_trigger`/`level0_stop_writes_trigger` (backpressure), `max_bytes_for_level_base` + `max_bytes_for_level_multiplier` (≈10× level growth), `target_file_size_base` (SST size), `compaction_pri = kMinOverlappingRatio` (pick least-overlapping files — modern default that lowers WA).

**Reduce write amplification**: bigger `write_buffer_size` and `max_write_buffer_number` (fewer, fatter L0 flushes); higher `level0_file_num_compaction_trigger`; larger `target_file_size_base`/`max_bytes_for_level_base`; or switch to **universal** and accept `max_size_amplification_percent` (e.g. 200 = 2× space). `use_direct_io_for_flush_and_compaction` avoids double-caching.

**Reduce read latency**: large `block_cache`; bloom/ribbon filter with adequate `bits_per_key`; partitioned index+filters (`kTwoLevelIndexSearch`, `partition_filters=true`) for big DBs; prefix bloom for range queries; `data_block_hash_table_util_ratio` for fast in-block lookup; `MultiGet` with `async_io`.

```cpp
o.compaction_style = rocksdb::kCompactionStyleUniversal;     // write-heavy example
rocksdb::CompactionOptionsUniversal u;
u.max_size_amplification_percent = 200;                      // bound space amp at 2×
u.min_merge_width = 4;
o.compaction_options_universal = u;
```

Manual `db->CompactRange(nullptr, nullptr)` forces full compaction (e.g. after bulk load) but is expensive — not a routine operation.

---

## 10. Backup, Checkpoint & Recovery

Never copy a live DB directory (ROCKS-BAK-01) — files mutate under compaction. Two safe mechanisms:

- **`BackupEngine`** — incremental, SST-deduplicating backups to a separate dir (`share_table_files=true`, `sync=true`). Supports `RestoreDBFromLatestBackup`, retention via `PurgeOldBackups(n)`, and backup info listing.
- **`Checkpoint`** — `Checkpoint::Create(db, …)->CreateCheckpoint(dir)` makes an near-instant point-in-time copy via **hardlinks** to immutable SSTs (on the same filesystem); the live DB keeps serving. Ideal for hot backups and cheap clones, then copy the checkpoint off-box in the background.

```cpp
rocksdb::Checkpoint* cp; rocksdb::Checkpoint::Create(db, &cp);
auto s = cp->CreateCheckpoint("/snap/db-" + ts);   // hardlinks, zero downtime
delete cp;
```

Inspect/repair with `ldb` and `sst_dump`; `ldb check_consistency` after suspect writes.

---

## 11. Security & Common Mistakes

RocksDB has **no authentication, no authorization, and no built-in encryption** — it trusts its caller. Security is the application's responsibility (policy: `secure-coding.md`):
- **Encryption at rest**: use filesystem/volume encryption (LUKS/dm-crypt) or encrypt values app-side; RocksDB's `EncryptedEnv` is a building block, not a managed solution. Never log or commit keys.
- **Access control & DoS**: validate key/value sizes; cap untrusted scans (`max_results`); set DB dir perms to `0700` (ROCKS-SEC-01/02).
- **Integrity**: keep block/WAL checksums on for untrusted or critical data; only disable `verify_checksums` for trusted hot paths.

**Tuning & usage mistakes to avoid:**
- Shipping default `Options` — no explicit block cache/bloom/write-buffer/compaction (ROCKS-TUNE-01).
- **Large values** (>~1 MB blobs): bloat the memtable, inflate write amp, thrash the block cache. Store blobs externally (object store/FS) and keep only metadata in RocksDB — or use **BlobDB / `enable_blob_files`** (key-value separation, WiscKey-style) for large values.
- **Hot keys**: a single contended counter caps throughput. Shard the key and aggregate, or use a **merge operator** (§7).
- Holding snapshots/iterators open for a long time — pins obsolete versions, raises space amp.
- Reopening with a changed comparator / merge-operator / prefix-extractor *name* — corrupts/refuses the DB.
- Ignoring write stalls: many L0 files or high `estimate-pending-compaction-bytes` → raise `max_background_jobs`, write buffer, or L0 triggers (diagnose via `rocksdb.stats` and `perf_context`).

---

## 12. RocksDB vs LevelDB & Other Engines

RocksDB is a fork of **LevelDB** (see [`leveldb.md`](guides://leveldb.md)) — same LSM model and core API, so most LevelDB code ports directly (`leveldb::` → `rocksdb::`; SSTs ingestible via `IngestExternalFile`). **What RocksDB adds over LevelDB:** column families, optimistic & pessimistic transactions, merge operators, multiple compaction styles (universal/FIFO), pluggable/multiple compression (LZ4/ZSTD), parallel multi-threaded compaction, rate limiting, rich statistics/`perf_context`, BackupEngine & Checkpoints, prefix bloom & partitioned filters, BlobDB. Choose LevelDB only for minimal embedded/prototype needs; choose RocksDB for production write-heavy or tunable workloads.

| vs | RocksDB | Other |
|----|---------|-------|
| **LMDB** | LSM-tree, high WA, low RA via bloom, multi-writer | B+ tree, WA ~1–2×, single-writer, read-heavy |
| **SQLite** | KV scans/prefix seeks, manual indexing, very high write throughput | relational SQL, auto B-tree indexes, structured data |

RocksDB is the right embedded engine when you need **high write throughput on SSD/flash, sorted range scans, per-workload tuning, and an in-process library** — and you don't need SQL, a B-tree's low write amp, or built-in replication.

---

## 13. Deployment Checklist

Generated from §2 — one box per requirement ID.

- [ ] ROCKS-TST-01/02 — tests pass against temp DBs, bugs have regression tests
- [ ] ROCKS-ERR-01/02 — every `Status` checked; `IsNotFound` vs error distinguished
- [ ] ROCKS-RES-01 — DB/iterator/snapshot/txn/CF handles RAII-owned, no leaks (ASan clean)
- [ ] ROCKS-DUR-01 — durability per write path documented (`sync`/WAL deliberate)
- [ ] ROCKS-TUNE-01 — block cache, bloom, write buffer, compaction style set explicitly
- [ ] ROCKS-AMP-01 — chosen tuning states which amplification it optimizes/accepts
- [ ] ROCKS-CF-01 — all column families opened with compatible descriptors
- [ ] ROCKS-BAK-01 — backups via BackupEngine/Checkpoint, never live-dir copy
- [ ] ROCKS-SEC-01/02 — key/value limits, scan caps, `0700` perms, encryption-at-rest provided
- [ ] ROCKS-VER-01 — RocksDB version + compression libs pinned
- [ ] Agent ran every §3 command and documented any fixes

---
**End of RocksDB Guidelines**
