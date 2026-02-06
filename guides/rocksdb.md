# RocksDB Development Guidelines

Mandatory coding standards and development practices for RocksDB development. RocksDB C++ API, Python bindings, column families, compaction tuning, backup/restore.

---

**Agent Profile**: The RocksDB Expert
**Role**: Senior Embedded Storage Engineer & LSM/Key-Value Specialist
**Objective**: Generate production-ready, high-performance and reliable embedded storage solutions using RocksDB.
**Tools**: RocksDB C++ API, Python bindings, column families, compaction tuning, backup/restore

---

## 1. Core Philosophies: LSM-FIRST

The agent must adhere to the **LSM-FIRST** principles for every RocksDB implementation:

**Test-Driven Development (TDD)**: ALWAYS write tests BEFORE implementation (Red-Green-Refactor cycle mandatory).
**Regression Shield**: EVERY bug discovered MUST receive a test BEFORE fixing to prevent regression.

- **L**SM-aware: Design for write amplification, compaction, and level structure; tune for workload (write-heavy vs read-heavy).
- **S**tatus checks: Always check Status/Result on every operation; never ignore errors or assume success.
- **M**emory and cache: Tune block cache, memtable size, and write buffer for your access pattern and hardware.
- **F**amilies: Use column families for logical partitioning when needed; share WAL, separate LSM trees.
- **I**dempotent recovery: Handle crash recovery and WAL replay; use backups for point-in-time recovery.
- **R**esource limits: Set max open files, compaction threads, and memory limits for production stability.
- **S**tability: Test under failure (disk full, I/O errors); avoid undefined behavior from invalid options.
- **T**esting: Unit test with in-memory or temp DB; integration test with real storage; benchmark before tuning.
**Verified Code**: Agent-generated code MUST check every RocksDB return status, use safe options, and pass tests before delivery.

---

## 2. Core Concepts and Architecture

RocksDB is a high-performance embedded key-value store optimized for fast storage (SSD, NVMe). Originally forked from LevelDB by Facebook (Meta), it's built on Log-Structured Merge (LSM) tree architecture and designed for write-heavy workloads with low-latency read requirements.

### LSM-Tree Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        MemTable (RAM)                        │
│                   Active writes go here                      │
└─────────────────────────────────────────────────────────────┘
                            ↓ (when full)
┌─────────────────────────────────────────────────────────────┐
│                   Immutable MemTable (RAM)                   │
│                    Waiting to be flushed                     │
└─────────────────────────────────────────────────────────────┘
                            ↓ (background flush)
┌─────────────────────────────────────────────────────────────┐
│  Level 0 (L0) - SST Files (4-8 files, may overlap)         │
└─────────────────────────────────────────────────────────────┘
                            ↓ (compaction)
┌─────────────────────────────────────────────────────────────┐
│  Level 1 (L1) - SST Files (~10x L0, no overlap)            │
└─────────────────────────────────────────────────────────────┘
                            ↓ (compaction)
┌─────────────────────────────────────────────────────────────┐
│  Level 2-6 (L2-L6) - Each level 10x previous               │
│                    Sorted, non-overlapping                   │
└─────────────────────────────────────────────────────────────┘
```

**Key Characteristics:**
- **Write Amplification**: Data written multiple times during compaction
- **Read Amplification**: May need to check multiple levels
- **Space Amplification**: Temporary extra space during compaction
- **Column Families**: Logical partitions sharing WAL but independent LSM trees
- **Block Cache**: Shared LRU cache for hot data blocks
- **Bloom Filters**: Probabilistic data structure to skip SST files on reads

### Embedded Library Model

```cpp
// RocksDB runs in-process (no client-server)
#include <rocksdb/db.h>

int main() {
    rocksdb::DB* db;
    rocksdb::Options options;

    // Direct function calls - no network overhead
    rocksdb::Status status = rocksdb::DB::Open(options, "/path/to/db", &db);

    // Operations execute in nanoseconds to microseconds
    db->Put(rocksdb::WriteOptions(), "key", "value");

    delete db;
    return 0;
}
```

**Performance Benefits:**
- Zero network latency
- Direct memory access
- No serialization overhead
- Sub-microsecond operations possible
- Single-process data integrity

## 3. Installation and Setup

### C++ Installation

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y \
    librocksdb-dev \
    libsnappy-dev \
    libgflags-dev \
    liblz4-dev \
    libzstd-dev

# Build from source for latest version and optimizations
git clone https://github.com/facebook/rocksdb.git
cd rocksdb
git checkout v9.0.0  # Latest stable

# Production build with all optimizations
make clean
DEBUG_LEVEL=0 \
  USE_RTTI=1 \
  PORTABLE=0 \
  FORCE_SSE42=1 \
  make -j$(nproc) shared_lib

sudo make install
```

### Modern CMake Integration

```cmake
# CMakeLists.txt
cmake_minimum_required(VERSION 3.16)
project(MyRocksDBApp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable compiler optimizations
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native -flto")

# Find RocksDB
find_package(RocksDB REQUIRED)

add_executable(myapp main.cpp)
target_link_libraries(myapp RocksDB::rocksdb)

# Link compression libraries
target_link_libraries(myapp snappy lz4 zstd)
```

### Language Bindings

```bash
# Python
pip install rocksdb

# Go
go get github.com/linxGnu/grocksdb

# Rust (high-performance binding)
cargo add rocksdb

# Java (official JNI binding)
# Add to pom.xml
<dependency>
    <groupId>org.rocksdb</groupId>
    <artifactId>rocksdbjni</artifactId>
    <version>9.0.0</version>
</dependency>
```

## 4. Modern C++ API - Basic Operations

### Production-Ready Database Setup

```cpp
#include <rocksdb/db.h>
#include <rocksdb/options.h>
#include <rocksdb/table.h>
#include <rocksdb/filter_policy.h>
#include <rocksdb/rate_limiter.h>
#include <rocksdb/statistics.h>
#include <memory>
#include <iostream>

class RocksDBWrapper {
private:
    std::unique_ptr<rocksdb::DB> db_;
    rocksdb::Options options_;

public:
    rocksdb::Status Open(const std::string& db_path) {
        // Block-based table options for optimal read performance
        rocksdb::BlockBasedTableOptions table_options;
        table_options.block_cache = rocksdb::NewLRUCache(512 * 1024 * 1024); // 512MB
        table_options.block_size = 16 * 1024; // 16KB blocks
        table_options.cache_index_and_filter_blocks = true;
        table_options.pin_l0_filter_and_index_blocks_in_cache = true;

        // Bloom filter for faster negative lookups (10 bits per key)
        table_options.filter_policy.reset(
            rocksdb::NewBloomFilterPolicy(10, false));

        options_.table_factory.reset(
            rocksdb::NewBlockBasedTableFactory(table_options));

        // Write buffer settings
        options_.write_buffer_size = 64 * 1024 * 1024; // 64MB memtable
        options_.max_write_buffer_number = 3;
        options_.min_write_buffer_number_to_merge = 2;

        // Level settings for balanced performance
        options_.level0_file_num_compaction_trigger = 4;
        options_.level0_slowdown_writes_trigger = 20;
        options_.level0_stop_writes_trigger = 36;

        // Compaction settings
        options_.max_background_jobs = 6; // Parallelism
        options_.max_subcompactions = 2;
        options_.compaction_style = rocksdb::kCompactionStyleLevel;

        // Compression per level (none for L0-L1, then increasingly aggressive)
        options_.compression_per_level = {
            rocksdb::kNoCompression,      // L0
            rocksdb::kNoCompression,      // L1
            rocksdb::kLZ4Compression,     // L2
            rocksdb::kLZ4Compression,     // L3
            rocksdb::kZSTD,               // L4+
        };

        // Rate limiting for background operations (50 MB/s)
        options_.rate_limiter.reset(
            rocksdb::NewGenericRateLimiter(50 * 1024 * 1024));

        // Statistics for monitoring
        options_.statistics = rocksdb::CreateDBStatistics();

        // Create DB if missing
        options_.create_if_missing = true;
        options_.create_missing_column_families = true;

        // WAL settings
        options_.WAL_ttl_seconds = 0;
        options_.WAL_size_limit_MB = 0;

        rocksdb::DB* db_ptr;
        rocksdb::Status status = rocksdb::DB::Open(options_, db_path, &db_ptr);
        db_.reset(db_ptr);

        return status;
    }

    rocksdb::DB* Get() { return db_.get(); }
    const rocksdb::Options& GetOptions() const { return options_; }
};
```

### High-Performance Write Patterns

```cpp
// Single write
rocksdb::Status Put(rocksdb::DB* db, const std::string& key,
                    const std::string& value) {
    rocksdb::WriteOptions write_options;
    write_options.sync = false;  // Don't fsync every write (batching is better)
    write_options.disableWAL = false;  // Keep WAL for durability

    return db->Put(write_options, key, value);
}

// Batch writes for maximum throughput
rocksdb::Status BatchWrite(rocksdb::DB* db,
                          const std::vector<std::pair<std::string, std::string>>& kvs) {
    rocksdb::WriteBatch batch;

    for (const auto& [key, value] : kvs) {
        batch.Put(key, value);
    }

    rocksdb::WriteOptions write_options;
    write_options.sync = false;

    return db->Write(write_options, &batch);
}

// Atomic read-modify-write with merge operator
class CounterMergeOperator : public rocksdb::AssociativeMergeOperator {
public:
    bool Merge(const rocksdb::Slice& key,
               const rocksdb::Slice* existing_value,
               const rocksdb::Slice& value,
               std::string* new_value,
               rocksdb::Logger* logger) const override {
        int64_t existing = 0;
        if (existing_value) {
            existing = std::stoll(existing_value->ToString());
        }
        int64_t delta = std::stoll(value.ToString());
        *new_value = std::to_string(existing + delta);
        return true;
    }

    const char* Name() const override {
        return "CounterMergeOperator";
    }
};

// Usage
void IncrementCounter(rocksdb::DB* db, const std::string& key, int64_t delta) {
    rocksdb::WriteOptions options;
    db->Merge(options, key, std::to_string(delta));
}
```

### Low-Latency Read Patterns

```cpp
// Simple read
rocksdb::Status Get(rocksdb::DB* db, const std::string& key, std::string* value) {
    rocksdb::ReadOptions read_options;
    read_options.verify_checksums = false;  // Skip checksum for lower latency
    read_options.fill_cache = true;          // Cache hot data

    return db->Get(read_options, key, value);
}

// Multi-get for batch reads (single disk seek for nearby keys)
std::vector<rocksdb::Status> MultiGet(
    rocksdb::DB* db,
    const std::vector<std::string>& keys,
    std::vector<std::string>* values) {

    rocksdb::ReadOptions read_options;
    read_options.verify_checksums = false;

    std::vector<rocksdb::Slice> key_slices;
    key_slices.reserve(keys.size());
    for (const auto& key : keys) {
        key_slices.emplace_back(key);
    }

    values->resize(keys.size());
    std::vector<rocksdb::Status> statuses =
        db->MultiGet(read_options, key_slices, values);

    return statuses;
}

// Range scan with prefix
void PrefixScan(rocksdb::DB* db, const std::string& prefix) {
    rocksdb::ReadOptions read_options;
    read_options.prefix_same_as_start = true;  // Optimize prefix scan
    read_options.total_order_seek = false;      // Use prefix bloom filter

    std::unique_ptr<rocksdb::Iterator> it(db->NewIterator(read_options));

    for (it->Seek(prefix); it->Valid() && it->key().starts_with(prefix); it->Next()) {
        std::cout << it->key().ToString() << ": "
                  << it->value().ToString() << std::endl;
    }

    if (!it->status().ok()) {
        std::cerr << "Iterator error: " << it->status().ToString() << std::endl;
    }
}

// Reverse iteration
void ReverseScan(rocksdb::DB* db, const std::string& start_key, int limit) {
    rocksdb::ReadOptions read_options;
    std::unique_ptr<rocksdb::Iterator> it(db->NewIterator(read_options));

    int count = 0;
    for (it->SeekForPrev(start_key); it->Valid() && count < limit; it->Prev()) {
        std::cout << it->key().ToString() << ": "
                  << it->value().ToString() << std::endl;
        count++;
    }
}
```

## 5. Column Families - Advanced Data Organization

Column families allow multiple independent LSM trees in a single database, sharing WAL but having separate memtables, SSTables, and compaction.

```cpp
#include <rocksdb/db.h>
#include <rocksdb/options.h>

class MultiColumnDB {
private:
    rocksdb::DB* db_;
    std::map<std::string, rocksdb::ColumnFamilyHandle*> cf_handles_;

public:
    rocksdb::Status Open(const std::string& db_path) {
        rocksdb::DBOptions db_options;
        db_options.create_if_missing = true;
        db_options.create_missing_column_families = true;
        db_options.max_background_jobs = 8;

        // Different options for different workloads
        rocksdb::ColumnFamilyOptions cf_default;
        cf_default.write_buffer_size = 64 * 1024 * 1024;

        // Hot data: optimized for reads
        rocksdb::ColumnFamilyOptions cf_hot;
        cf_hot.write_buffer_size = 128 * 1024 * 1024;
        cf_hot.level0_file_num_compaction_trigger = 2;
        rocksdb::BlockBasedTableOptions hot_table;
        hot_table.block_cache = rocksdb::NewLRUCache(1024 * 1024 * 1024); // 1GB
        hot_table.filter_policy.reset(rocksdb::NewBloomFilterPolicy(10));
        cf_hot.table_factory.reset(rocksdb::NewBlockBasedTableFactory(hot_table));

        // Archive data: optimized for space
        rocksdb::ColumnFamilyOptions cf_archive;
        cf_archive.write_buffer_size = 256 * 1024 * 1024;
        cf_archive.compression = rocksdb::kZSTD;
        cf_archive.bottommost_compression = rocksdb::kZSTD;
        cf_archive.bottommost_compression_opts.level = 9; // Max compression

        std::vector<rocksdb::ColumnFamilyDescriptor> column_families = {
            {rocksdb::kDefaultColumnFamilyName, cf_default},
            {"hot_data", cf_hot},
            {"archive", cf_archive},
            {"indexes", cf_default},
        };

        std::vector<rocksdb::ColumnFamilyHandle*> handles;
        rocksdb::Status status = rocksdb::DB::Open(
            db_options, db_path, column_families, &handles, &db_);

        if (status.ok()) {
            cf_handles_["default"] = handles[0];
            cf_handles_["hot_data"] = handles[1];
            cf_handles_["archive"] = handles[2];
            cf_handles_["indexes"] = handles[3];
        }

        return status;
    }

    rocksdb::Status PutInFamily(const std::string& cf_name,
                                const std::string& key,
                                const std::string& value) {
        auto it = cf_handles_.find(cf_name);
        if (it == cf_handles_.end()) {
            return rocksdb::Status::InvalidArgument("Column family not found");
        }

        rocksdb::WriteOptions options;
        return db_->Put(options, it->second, key, value);
    }

    rocksdb::Status GetFromFamily(const std::string& cf_name,
                                  const std::string& key,
                                  std::string* value) {
        auto it = cf_handles_.find(cf_name);
        if (it == cf_handles_.end()) {
            return rocksdb::Status::InvalidArgument("Column family not found");
        }

        rocksdb::ReadOptions options;
        return db_->Get(options, it->second, key, value);
    }

    ~MultiColumnDB() {
        for (auto& [name, handle] : cf_handles_) {
            db_->DestroyColumnFamilyHandle(handle);
        }
        delete db_;
    }
};
```

**Use Cases for Column Families:**
- **Separate workloads**: User data vs. indexes vs. analytics
- **Different retention**: Hot data vs. cold archive
- **Isolation**: One CF's compaction doesn't block another
- **TTL per family**: Different expiration policies

## 6. Python API - Modern Async Patterns

```python
import rocksdb
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, List, Dict
import struct

class AsyncRocksDB:
    """Async wrapper for RocksDB operations."""

    def __init__(self, path: str, thread_pool_size: int = 4):
        opts = rocksdb.Options()
        opts.create_if_missing = True
        opts.max_open_files = 10000

        # Write optimizations
        opts.write_buffer_size = 67108864  # 64MB
        opts.max_write_buffer_number = 3
        opts.target_file_size_base = 67108864

        # Compaction
        opts.level0_file_num_compaction_trigger = 8
        opts.level0_slowdown_writes_trigger = 17
        opts.level0_stop_writes_trigger = 24
        opts.max_background_compactions = 4
        opts.max_background_flushes = 2

        # Compression
        opts.compression = rocksdb.CompressionType.lz4_compression

        # Block cache
        opts.table_factory = rocksdb.BlockBasedTableFactory(
            filter_policy=rocksdb.BloomFilterPolicy(10),
            block_cache=rocksdb.LRUCache(512 * 1024 * 1024),  # 512MB
            block_cache_compressed=rocksdb.LRUCache(128 * 1024 * 1024),
        )

        # Statistics
        opts.statistics = True

        self.db = rocksdb.DB(path, opts)
        self.executor = ThreadPoolExecutor(max_workers=thread_pool_size)

    async def get(self, key: bytes) -> Optional[bytes]:
        """Async get operation."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self.db.get, key)

    async def put(self, key: bytes, value: bytes) -> None:
        """Async put operation."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(self.executor, self.db.put, key, value)

    async def multi_get(self, keys: List[bytes]) -> List[Optional[bytes]]:
        """Async multi-get operation."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self.db.multi_get,
            keys
        )

    async def batch_write(self, operations: List[tuple]) -> None:
        """Async batch write."""
        def _write():
            batch = rocksdb.WriteBatch()
            for op, key, *args in operations:
                if op == 'put':
                    batch.put(key, args[0])
                elif op == 'delete':
                    batch.delete(key)
            self.db.write(batch)

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(self.executor, _write)

    async def iterator_range(self, start_key: bytes, end_key: bytes) -> List[tuple]:
        """Async range scan."""
        def _scan():
            results = []
            it = self.db.iteritems()
            it.seek(start_key)
            for key, value in it:
                if key >= end_key:
                    break
                results.append((key, value))
            return results

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, _scan)

    def get_stats(self) -> Dict[str, int]:
        """Get database statistics."""
        stats_str = self.db.get_property(b'rocksdb.stats').decode('utf-8')
        return {'stats': stats_str}

    def close(self):
        """Close database and thread pool."""
        del self.db
        self.executor.shutdown(wait=True)

# Usage example
async def main():
    db = AsyncRocksDB('/tmp/mydb')

    # Concurrent writes
    await asyncio.gather(
        db.put(b'key1', b'value1'),
        db.put(b'key2', b'value2'),
        db.put(b'key3', b'value3'),
    )

    # Batch operations for better performance
    await db.batch_write([
        ('put', b'user:1', b'{"name":"Alice","age":30}'),
        ('put', b'user:2', b'{"name":"Bob","age":25}'),
        ('put', b'user:3', b'{"name":"Charlie","age":35}'),
    ])

    # Multi-get
    values = await db.multi_get([b'user:1', b'user:2', b'user:3'])
    print(values)

    # Range scan
    results = await db.iterator_range(b'user:1', b'user:4')
    print(results)

    db.close()

if __name__ == '__main__':
    asyncio.run(main())
```

## 7. Transactions and Consistency

RocksDB supports optimistic and pessimistic transactions for ACID guarantees.

### Optimistic Transactions

```cpp
#include <rocksdb/utilities/optimistic_transaction_db.h>
#include <rocksdb/utilities/transaction.h>

class OptimisticTxnManager {
private:
    rocksdb::OptimisticTransactionDB* txn_db_;

public:
    rocksdb::Status Open(const std::string& path) {
        rocksdb::Options options;
        options.create_if_missing = true;
        options.max_background_jobs = 4;

        rocksdb::OptimisticTransactionDB* db;
        rocksdb::Status status = rocksdb::OptimisticTransactionDB::Open(
            options, path, &db);

        txn_db_ = db;
        return status;
    }

    // Transfer with conflict detection
    rocksdb::Status Transfer(const std::string& from_key,
                            const std::string& to_key,
                            int64_t amount) {
        rocksdb::WriteOptions write_options;
        rocksdb::ReadOptions read_options;
        rocksdb::OptimisticTransactionOptions txn_options;

        rocksdb::Transaction* txn = txn_db_->BeginTransaction(
            write_options, txn_options);

        std::string from_balance_str, to_balance_str;
        rocksdb::Status s;

        // Read with snapshot isolation
        s = txn->Get(read_options, from_key, &from_balance_str);
        if (!s.ok()) {
            delete txn;
            return s;
        }

        s = txn->Get(read_options, to_key, &to_balance_str);
        if (!s.ok()) {
            delete txn;
            return s;
        }

        int64_t from_balance = std::stoll(from_balance_str);
        int64_t to_balance = std::stoll(to_balance_str);

        if (from_balance < amount) {
            delete txn;
            return rocksdb::Status::InvalidArgument("Insufficient funds");
        }

        from_balance -= amount;
        to_balance += amount;

        txn->Put(from_key, std::to_string(from_balance));
        txn->Put(to_key, std::to_string(to_balance));

        // Commit - will fail if keys were modified by another transaction
        s = txn->Commit();
        delete txn;

        return s;
    }

    ~OptimisticTxnManager() {
        delete txn_db_;
    }
};
```

### Pessimistic Transactions with Locking

```cpp
#include <rocksdb/utilities/transaction_db.h>

class PessimisticTxnManager {
private:
    rocksdb::TransactionDB* txn_db_;

public:
    rocksdb::Status Open(const std::string& path) {
        rocksdb::Options options;
        options.create_if_missing = true;
        options.max_background_jobs = 4;

        rocksdb::TransactionDBOptions txn_db_options;
        txn_db_options.transaction_lock_timeout = 1000;  // 1 second
        txn_db_options.default_lock_timeout = 1000;

        rocksdb::TransactionDB* db;
        rocksdb::Status status = rocksdb::TransactionDB::Open(
            options, txn_db_options, path, &db);

        txn_db_ = db;
        return status;
    }

    // Transfer with row-level locking
    rocksdb::Status TransferWithLock(const std::string& from_key,
                                     const std::string& to_key,
                                     int64_t amount) {
        rocksdb::WriteOptions write_options;
        rocksdb::ReadOptions read_options;
        rocksdb::TransactionOptions txn_options;

        // Deadlock detection
        txn_options.deadlock_detect = true;
        txn_options.lock_timeout = 500;  // 500ms

        rocksdb::Transaction* txn = txn_db_->BeginTransaction(
            write_options, txn_options);

        std::string from_balance_str, to_balance_str;
        rocksdb::Status s;

        // GetForUpdate acquires read lock, prevents other writes
        s = txn->GetForUpdate(read_options, from_key, &from_balance_str);
        if (!s.ok()) {
            txn->Rollback();
            delete txn;
            return s;
        }

        s = txn->GetForUpdate(read_options, to_key, &to_balance_str);
        if (!s.ok()) {
            txn->Rollback();
            delete txn;
            return s;
        }

        int64_t from_balance = std::stoll(from_balance_str);
        int64_t to_balance = std::stoll(to_balance_str);

        if (from_balance < amount) {
            txn->Rollback();
            delete txn;
            return rocksdb::Status::InvalidArgument("Insufficient funds");
        }

        from_balance -= amount;
        to_balance += amount;

        txn->Put(from_key, std::to_string(from_balance));
        txn->Put(to_key, std::to_string(to_balance));

        // Commit and release locks
        s = txn->Commit();
        delete txn;

        return s;
    }

    rocksdb::DB* GetBaseDB() {
        return txn_db_->GetBaseDB();
    }

    ~PessimisticTxnManager() {
        delete txn_db_;
    }
};
```

## 8. Performance Optimization - Write Amplification

Write amplification is the ratio of bytes written to storage vs. bytes written by the application. RocksDB rewrites data during compaction.

### Measuring Write Amplification

```cpp
void PrintWriteAmplification(rocksdb::DB* db) {
    std::string stats;
    db->GetProperty("rocksdb.stats", &stats);

    // Parse cumulative writes
    uint64_t bytes_written = 0;
    uint64_t wal_bytes = 0;
    uint64_t compact_bytes = 0;

    db->GetIntProperty("rocksdb.total-sst-files-size", &bytes_written);

    std::cout << "Total SST files size: " << bytes_written / (1024*1024) << " MB" << std::endl;
    std::cout << stats << std::endl;

    // Write amplification = (WAL + SST writes) / User writes
    // Typical values: 10-30x depending on configuration
}
```

### Reducing Write Amplification

```cpp
rocksdb::Options GetLowWriteAmpOptions() {
    rocksdb::Options options;

    // Larger memtable = fewer L0 files = less compaction
    options.write_buffer_size = 128 * 1024 * 1024;  // 128MB
    options.max_write_buffer_number = 4;

    // Increase L0 trigger = fewer compactions
    options.level0_file_num_compaction_trigger = 8;  // Default: 4

    // Larger SST files = fewer files = less compaction overhead
    options.target_file_size_base = 128 * 1024 * 1024;  // 128MB
    options.target_file_size_multiplier = 2;

    // Larger level size ratio = less compaction frequency
    options.max_bytes_for_level_base = 512 * 1024 * 1024;  // 512MB
    options.max_bytes_for_level_multiplier = 10;

    // Use universal compaction for write-heavy workloads
    options.compaction_style = rocksdb::kCompactionStyleUniversal;

    rocksdb::CompactionOptionsUniversal universal_options;
    universal_options.size_ratio = 1;
    universal_options.min_merge_width = 4;
    universal_options.max_size_amplification_percent = 200;  // 2x space amp
    options.compaction_options_universal = universal_options;

    // Enable direct I/O to bypass OS cache (reduce double caching)
    options.use_direct_reads = true;
    options.use_direct_io_for_flush_and_compaction = true;

    return options;
}
```

## 9. Performance Optimization - Read Latency

### Bloom Filters and Partitioned Indexes

```cpp
rocksdb::Options GetLowLatencyReadOptions() {
    rocksdb::Options options;

    rocksdb::BlockBasedTableOptions table_options;

    // Large block cache for hot data
    table_options.block_cache = rocksdb::NewLRUCache(
        2ULL * 1024 * 1024 * 1024);  // 2GB

    // Partitioned index/filters for large databases
    table_options.index_type = rocksdb::BlockBasedTableOptions::kTwoLevelIndexSearch;
    table_options.partition_filters = true;
    table_options.metadata_block_size = 4096;
    table_options.cache_index_and_filter_blocks = true;
    table_options.cache_index_and_filter_blocks_with_high_priority = true;
    table_options.pin_l0_filter_and_index_blocks_in_cache = true;
    table_options.pin_top_level_index_and_filter = true;

    // Ribbon filter (more space-efficient than Bloom)
    table_options.filter_policy.reset(
        rocksdb::NewRibbonFilterPolicy(10.0));  // 10 bits per key

    // Smaller blocks for point lookups
    table_options.block_size = 4 * 1024;  // 4KB

    // Enable hash index for fast lookups within block
    table_options.data_block_index_type =
        rocksdb::BlockBasedTableOptions::kDataBlockBinaryAndHash;
    table_options.data_block_hash_table_util_ratio = 0.75;

    options.table_factory.reset(
        rocksdb::NewBlockBasedTableFactory(table_options));

    // Prefix bloom filter for range queries
    options.prefix_extractor.reset(
        rocksdb::NewFixedPrefixTransform(8));  // 8-byte prefix

    // Optimize for more L0 files (less write amp, but need good bloom)
    options.level0_file_num_compaction_trigger = 10;

    // Parallel reads
    options.max_file_opening_threads = 16;

    return options;
}

// Usage with prefix bloom
void FastPrefixLookup(rocksdb::DB* db, const std::string& prefix) {
    rocksdb::ReadOptions read_options;
    read_options.prefix_same_as_start = true;
    read_options.total_order_seek = false;  // Use prefix bloom
    read_options.adaptive_readahead = true;  // Prefetch for sequential reads

    auto it = db->NewIterator(read_options);
    for (it->Seek(prefix); it->Valid() && it->key().starts_with(prefix); it->Next()) {
        // Process
    }
    delete it;
}
```

### MultiGet Optimization

```cpp
// MultiGet with batching for minimum latency
std::vector<rocksdb::Status> FastMultiGet(
    rocksdb::DB* db,
    const std::vector<std::string>& keys) {

    rocksdb::ReadOptions read_options;
    read_options.verify_checksums = false;  // Skip checksum for speed
    read_options.fill_cache = true;
    read_options.async_io = true;  // Parallel I/O

    std::vector<rocksdb::Slice> key_slices;
    std::vector<std::string> values(keys.size());
    std::vector<rocksdb::Status> statuses(keys.size());

    for (const auto& key : keys) {
        key_slices.emplace_back(key);
    }

    // Single call with I/O coalescing
    db->MultiGet(read_options, db->DefaultColumnFamily(),
                 key_slices.size(), key_slices.data(),
                 values.data(), statuses.data());

    return statuses;
}
```

## 10. Compaction Strategies

### Level Compaction (Default)

```cpp
rocksdb::Options GetLevelCompactionOptions() {
    rocksdb::Options options;

    options.compaction_style = rocksdb::kCompactionStyleLevel;

    // L0 -> L1 compaction
    options.level0_file_num_compaction_trigger = 4;
    options.level0_slowdown_writes_trigger = 20;
    options.level0_stop_writes_trigger = 36;

    // Level sizes
    options.max_bytes_for_level_base = 256 * 1024 * 1024;  // 256MB for L1
    options.max_bytes_for_level_multiplier = 10;  // L2=2.5GB, L3=25GB, etc.

    // File sizes
    options.target_file_size_base = 64 * 1024 * 1024;  // 64MB
    options.target_file_size_multiplier = 2;  // Double each level

    // Parallelism
    options.max_background_compactions = 4;
    options.max_subcompactions = 2;  // Parallel compaction of single file

    // Compaction priority
    options.compaction_pri = rocksdb::kMinOverlappingRatio;

    return options;
}
```

### Universal Compaction (Write-Heavy)

```cpp
rocksdb::Options GetUniversalCompactionOptions() {
    rocksdb::Options options;

    options.compaction_style = rocksdb::kCompactionStyleUniversal;

    rocksdb::CompactionOptionsUniversal universal;
    universal.size_ratio = 1;  // Percent size difference to trigger compaction
    universal.min_merge_width = 2;  // Minimum files to compact
    universal.max_merge_width = UINT_MAX;  // Maximum files
    universal.max_size_amplification_percent = 200;  // 2x space overhead
    universal.compression_size_percent = 80;  // Compress bottom 80%
    universal.stop_style = rocksdb::kCompactionStopStyleTotalSize;

    options.compaction_options_universal = universal;

    // Larger buffers for universal
    options.write_buffer_size = 256 * 1024 * 1024;
    options.max_write_buffer_number = 4;

    return options;
}
```

### FIFO Compaction (TTL Data)

```cpp
rocksdb::Options GetFIFOCompactionOptions(uint64_t ttl_seconds) {
    rocksdb::Options options;

    options.compaction_style = rocksdb::kCompactionStyleFIFO;

    rocksdb::CompactionOptionsFIFO fifo;
    fifo.max_table_files_size = 10ULL * 1024 * 1024 * 1024;  // 10GB total
    fifo.ttl = ttl_seconds;  // Delete files older than TTL
    fifo.allow_compaction = true;  // Allow trivial moves

    options.compaction_options_fifo = fifo;

    // Disable levels
    options.num_levels = 1;

    return options;
}
```

## 11. Backup and Recovery

### Incremental Backup

```cpp
#include <rocksdb/utilities/backup_engine.h>

class BackupManager {
private:
    rocksdb::BackupEngine* backup_engine_;

public:
    rocksdb::Status Open(const std::string& backup_dir) {
        rocksdb::BackupEngineOptions options(backup_dir);
        options.share_table_files = true;  // Dedup SST files
        options.sync = true;  // Fsync for durability
        options.max_background_operations = 4;

        rocksdb::BackupEngine* engine;
        rocksdb::Status s = rocksdb::BackupEngine::Open(
            rocksdb::Env::Default(), options, &engine);

        backup_engine_ = engine;
        return s;
    }

    rocksdb::Status CreateBackup(rocksdb::DB* db, bool flush_before_backup = true) {
        return backup_engine_->CreateNewBackup(db, flush_before_backup);
    }

    rocksdb::Status RestoreLatestBackup(const std::string& db_dir,
                                       const std::string& wal_dir = "") {
        std::string wal = wal_dir.empty() ? db_dir : wal_dir;
        return backup_engine_->RestoreDBFromLatestBackup(db_dir, wal);
    }

    rocksdb::Status RestoreBackup(uint32_t backup_id,
                                  const std::string& db_dir,
                                  const std::string& wal_dir = "") {
        std::string wal = wal_dir.empty() ? db_dir : wal_dir;
        return backup_engine_->RestoreDBFromBackup(backup_id, db_dir, wal);
    }

    void GetBackupInfo(std::vector<rocksdb::BackupInfo>* backup_info) {
        backup_engine_->GetBackupInfo(backup_info);
    }

    rocksdb::Status DeleteBackup(uint32_t backup_id) {
        return backup_engine_->DeleteBackup(backup_id);
    }

    rocksdb::Status PurgeOldBackups(uint32_t num_backups_to_keep) {
        return backup_engine_->PurgeOldBackups(num_backups_to_keep);
    }

    ~BackupManager() {
        delete backup_engine_;
    }
};

// Usage
void PerformBackup(rocksdb::DB* db) {
    BackupManager backup_mgr;
    backup_mgr.Open("/backups/mydb");

    auto status = backup_mgr.CreateBackup(db);
    if (!status.ok()) {
        std::cerr << "Backup failed: " << status.ToString() << std::endl;
        return;
    }

    std::vector<rocksdb::BackupInfo> info;
    backup_mgr.GetBackupInfo(&info);

    std::cout << "Backups:" << std::endl;
    for (const auto& backup : info) {
        std::cout << "  ID: " << backup.backup_id
                  << ", Size: " << backup.size / (1024*1024) << " MB"
                  << ", Files: " << backup.number_files << std::endl;
    }

    // Keep only last 5 backups
    backup_mgr.PurgeOldBackups(5);
}
```

### Checkpoints for Point-in-Time Snapshots

```cpp
#include <rocksdb/utilities/checkpoint.h>

rocksdb::Status CreateCheckpoint(rocksdb::DB* db, const std::string& checkpoint_dir) {
    rocksdb::Checkpoint* checkpoint;
    rocksdb::Status s = rocksdb::Checkpoint::Create(db, &checkpoint);
    if (!s.ok()) {
        return s;
    }

    // Creates hardlinks to SST files (instant, no copying)
    s = checkpoint->CreateCheckpoint(checkpoint_dir);

    delete checkpoint;
    return s;
}

// Usage for zero-downtime backup
void HotBackup(rocksdb::DB* db) {
    std::string checkpoint_dir = "/tmp/checkpoint_" +
        std::to_string(std::time(nullptr));

    auto status = CreateCheckpoint(db, checkpoint_dir);
    if (status.ok()) {
        // Copy checkpoint to backup location in background
        // Original DB continues serving requests
        std::cout << "Checkpoint created at " << checkpoint_dir << std::endl;
    }
}
```

## 12. Monitoring and Statistics

### Real-Time Statistics

```cpp
#include <rocksdb/statistics.h>
#include <rocksdb/iostats_context.h>
#include <rocksdb/perf_context.h>

class RocksDBMonitor {
public:
    static void PrintStatistics(rocksdb::DB* db) {
        std::string stats;

        // Compaction stats
        db->GetProperty("rocksdb.stats", &stats);
        std::cout << stats << std::endl;

        // Detailed statistics
        uint64_t value;

        // Reads
        db->GetIntProperty("rocksdb.estimate-num-keys", &value);
        std::cout << "Estimated keys: " << value << std::endl;

        db->GetIntProperty("rocksdb.num-snapshots", &value);
        std::cout << "Active snapshots: " << value << std::endl;

        // Memory usage
        db->GetIntProperty("rocksdb.cur-size-all-mem-tables", &value);
        std::cout << "MemTable size: " << value / (1024*1024) << " MB" << std::endl;

        db->GetIntProperty("rocksdb.block-cache-usage", &value);
        std::cout << "Block cache usage: " << value / (1024*1024) << " MB" << std::endl;

        // Disk usage
        db->GetIntProperty("rocksdb.total-sst-files-size", &value);
        std::cout << "Total SST size: " << value / (1024*1024) << " MB" << std::endl;

        db->GetIntProperty("rocksdb.live-sst-files-size", &value);
        std::cout << "Live SST size: " << value / (1024*1024) << " MB" << std::endl;

        // Compaction pending
        db->GetIntProperty("rocksdb.estimate-pending-compaction-bytes", &value);
        std::cout << "Pending compaction: " << value / (1024*1024) << " MB" << std::endl;

        db->GetIntProperty("rocksdb.num-running-compactions", &value);
        std::cout << "Running compactions: " << value << std::endl;

        db->GetIntProperty("rocksdb.num-running-flushes", &value);
        std::cout << "Running flushes: " << value << std::endl;
    }

    static void EnablePerContextStats() {
        rocksdb::SetPerfLevel(rocksdb::kEnableTimeExceptForMutex);
        rocksdb::get_perf_context()->Reset();
        rocksdb::get_iostats_context()->Reset();
    }

    static void PrintPerContextStats() {
        auto perf = rocksdb::get_perf_context();
        auto io = rocksdb::get_iostats_context();

        std::cout << "Operation stats:" << std::endl;
        std::cout << "  User key comparison count: " << perf->user_key_comparison_count << std::endl;
        std::cout << "  Block cache hit count: " << perf->block_cache_hit_count << std::endl;
        std::cout << "  Block cache miss count: " << perf->block_cache_miss_count << std::endl;
        std::cout << "  Bloom filter useful: " << perf->bloom_filter_useful << std::endl;
        std::cout << "  Get snapshot time: " << perf->get_snapshot_time << " ns" << std::endl;
        std::cout << "  Get from memtable time: " << perf->get_from_memtable_time << " ns" << std::endl;
        std::cout << "  Seek on memtable time: " << perf->seek_on_memtable_time << " ns" << std::endl;

        std::cout << "\nI/O stats:" << std::endl;
        std::cout << "  Bytes read: " << io->bytes_read << std::endl;
        std::cout << "  Bytes written: " << io->bytes_written << std::endl;
    }
};

// Usage in critical path
void MeasuredOperation(rocksdb::DB* db) {
    RocksDBMonitor::EnablePerContextStats();

    // Perform operations
    std::string value;
    db->Get(rocksdb::ReadOptions(), "key", &value);

    RocksDBMonitor::PrintPerContextStats();
}
```

### Histogram Statistics

```cpp
rocksdb::Options GetOptionsWithHistograms() {
    rocksdb::Options options;

    // Enable statistics collection
    options.statistics = rocksdb::CreateDBStatistics();
    options.stats_dump_period_sec = 600;  // Dump every 10 minutes

    return options;
}

void PrintHistograms(const std::shared_ptr<rocksdb::Statistics>& stats) {
    // Latency histograms
    std::cout << "DB Get latency (micros):" << std::endl;
    std::cout << stats->getHistogramString(rocksdb::DB_GET) << std::endl;

    std::cout << "DB Write latency (micros):" << std::endl;
    std::cout << stats->getHistogramString(rocksdb::DB_WRITE) << std::endl;

    std::cout << "Compaction time (micros):" << std::endl;
    std::cout << stats->getHistogramString(rocksdb::COMPACTION_TIME) << std::endl;

    // Counters
    std::cout << "Number of keys written: "
              << stats->getTickerCount(rocksdb::NUMBER_KEYS_WRITTEN) << std::endl;
    std::cout << "Number of keys read: "
              << stats->getTickerCount(rocksdb::NUMBER_KEYS_READ) << std::endl;
    std::cout << "Bloom filter prefix checked: "
              << stats->getTickerCount(rocksdb::BLOOM_FILTER_PREFIX_CHECKED) << std::endl;
    std::cout << "Bloom filter prefix useful: "
              << stats->getTickerCount(rocksdb::BLOOM_FILTER_PREFIX_USEFUL) << std::endl;
}
```

## 13. Security Best Practices

### Encryption at Rest

```cpp
#include <rocksdb/env_encryption.h>
#include <openssl/rand.h>
#include <openssl/evp.h>

class SecureRocksDB {
public:
    static rocksdb::Status OpenEncrypted(const std::string& db_path,
                                         const std::string& key,
                                         rocksdb::DB** db) {
        rocksdb::Options options;
        options.create_if_missing = true;

        // Create encryption provider (AES-256-CTR)
        std::shared_ptr<rocksdb::EncryptionProvider> provider;
        rocksdb::Status s = rocksdb::NewEncryptionProvider(
            rocksdb::EncryptionProvider::kAES256CTR,
            key,
            &provider);

        if (!s.ok()) {
            return s;
        }

        // Create encrypted environment
        std::unique_ptr<rocksdb::Env> encrypted_env;
        s = rocksdb::NewEncryptedEnv(
            rocksdb::Env::Default(),
            provider,
            &encrypted_env);

        if (!s.ok()) {
            return s;
        }

        options.env = encrypted_env.get();

        return rocksdb::DB::Open(options, db_path, db);
    }

    static std::string GenerateKey() {
        unsigned char key[32];  // 256 bits
        RAND_bytes(key, sizeof(key));
        return std::string(reinterpret_cast<char*>(key), sizeof(key));
    }
};

// Usage with key management
void SecureDatabase() {
    // In production, load key from secure key management system (KMS)
    std::string encryption_key = SecureRocksDB::GenerateKey();

    // Store key securely (e.g., AWS KMS, HashiCorp Vault, etc.)
    // NEVER hardcode or commit keys to version control

    rocksdb::DB* db;
    auto status = SecureRocksDB::OpenEncrypted("/secure/db", encryption_key, &db);

    if (status.ok()) {
        // All data written to disk is encrypted
        db->Put(rocksdb::WriteOptions(), "secret", "encrypted value");
    }

    delete db;
}
```

### Access Control and Input Validation

```cpp
#include <regex>
#include <limits>

class SecureKeyValueStore {
private:
    rocksdb::DB* db_;
    static constexpr size_t MAX_KEY_SIZE = 1024;
    static constexpr size_t MAX_VALUE_SIZE = 10 * 1024 * 1024;  // 10MB

    bool ValidateKey(const std::string& key) {
        // Prevent excessively large keys
        if (key.empty() || key.size() > MAX_KEY_SIZE) {
            return false;
        }

        // Whitelist allowed characters (alphanumeric, -, _, :)
        static const std::regex key_pattern("^[a-zA-Z0-9_:-]+$");
        return std::regex_match(key, key_pattern);
    }

    bool ValidateValue(const std::string& value) {
        // Prevent excessively large values
        return value.size() <= MAX_VALUE_SIZE;
    }

public:
    SecureKeyValueStore(rocksdb::DB* db) : db_(db) {}

    rocksdb::Status SecurePut(const std::string& key, const std::string& value) {
        if (!ValidateKey(key)) {
            return rocksdb::Status::InvalidArgument("Invalid key format");
        }

        if (!ValidateValue(value)) {
            return rocksdb::Status::InvalidArgument("Value too large");
        }

        rocksdb::WriteOptions options;
        options.sync = false;
        options.disableWAL = false;  // Always use WAL for durability

        return db_->Put(options, key, value);
    }

    rocksdb::Status SecureGet(const std::string& key, std::string* value) {
        if (!ValidateKey(key)) {
            return rocksdb::Status::InvalidArgument("Invalid key format");
        }

        rocksdb::ReadOptions options;
        options.verify_checksums = true;  // Detect corruption

        return db_->Get(options, key, value);
    }

    // Prevent iteration over entire database (DOS risk)
    rocksdb::Status SecureScan(const std::string& prefix,
                              std::vector<std::pair<std::string, std::string>>* results,
                              size_t max_results = 1000) {
        if (!ValidateKey(prefix)) {
            return rocksdb::Status::InvalidArgument("Invalid prefix");
        }

        rocksdb::ReadOptions options;
        options.prefix_same_as_start = true;

        auto it = db_->NewIterator(options);
        size_t count = 0;

        for (it->Seek(prefix); it->Valid() && it->key().starts_with(prefix) && count < max_results; it->Next()) {
            results->emplace_back(it->key().ToString(), it->value().ToString());
            count++;
        }

        auto status = it->status();
        delete it;

        return status;
    }
};
```

### Checksums and Data Integrity

```cpp
rocksdb::Options GetSecureOptions() {
    rocksdb::Options options;

    // Enable paranoid checks
    options.paranoid_checks = true;

    // Enable checksum verification
    rocksdb::BlockBasedTableOptions table_options;
    table_options.checksum = rocksdb::kCRC32c;  // Hardware-accelerated CRC
    options.table_factory.reset(
        rocksdb::NewBlockBasedTableFactory(table_options));

    // Verify checksums on reads (production should enable)
    // Can disable for latency-critical reads if data corruption is rare

    // Enable background verification
    options.stats_dump_period_sec = 600;

    return options;
}

void VerifyChecksums(rocksdb::DB* db) {
    rocksdb::ReadOptions options;
    options.verify_checksums = true;

    auto it = db->NewIterator(options);
    for (it->SeekToFirst(); it->Valid(); it->Next()) {
        // Iteration verifies checksums of all blocks
    }

    if (!it->status().ok()) {
        std::cerr << "Checksum verification failed: "
                  << it->status().ToString() << std::endl;
    }

    delete it;
}
```

## 14. Testing Strategies

### Unit Testing with Mock Environment

```cpp
#include <gtest/gtest.h>
#include <rocksdb/db.h>
#include <rocksdb/utilities/transaction_db.h>

class RocksDBTest : public ::testing::Test {
protected:
    std::string db_path_;
    rocksdb::DB* db_;
    rocksdb::Options options_;

    void SetUp() override {
        db_path_ = "/tmp/test_db_" + std::to_string(getpid()) + "_" +
                   std::to_string(std::time(nullptr));

        options_.create_if_missing = true;
        options_.error_if_exists = true;

        rocksdb::Status status = rocksdb::DB::Open(options_, db_path_, &db_);
        ASSERT_TRUE(status.ok()) << status.ToString();
    }

    void TearDown() override {
        delete db_;
        db_ = nullptr;

        // Clean up test database
        rocksdb::DestroyDB(db_path_, options_);
    }
};

TEST_F(RocksDBTest, BasicPutGet) {
    rocksdb::WriteOptions write_opts;
    rocksdb::ReadOptions read_opts;

    // Write
    auto status = db_->Put(write_opts, "key1", "value1");
    ASSERT_TRUE(status.ok());

    // Read
    std::string value;
    status = db_->Get(read_opts, "key1", &value);
    ASSERT_TRUE(status.ok());
    EXPECT_EQ(value, "value1");

    // Not found
    status = db_->Get(read_opts, "key2", &value);
    EXPECT_TRUE(status.IsNotFound());
}

TEST_F(RocksDBTest, BatchWrite) {
    rocksdb::WriteBatch batch;
    batch.Put("key1", "value1");
    batch.Put("key2", "value2");
    batch.Delete("key3");

    auto status = db_->Write(rocksdb::WriteOptions(), &batch);
    ASSERT_TRUE(status.ok());

    std::string value;
    ASSERT_TRUE(db_->Get(rocksdb::ReadOptions(), "key1", &value).ok());
    EXPECT_EQ(value, "value1");
}

TEST_F(RocksDBTest, Iterator) {
    // Insert test data
    db_->Put(rocksdb::WriteOptions(), "a", "1");
    db_->Put(rocksdb::WriteOptions(), "b", "2");
    db_->Put(rocksdb::WriteOptions(), "c", "3");

    // Forward iteration
    std::vector<std::string> keys;
    auto it = db_->NewIterator(rocksdb::ReadOptions());
    for (it->SeekToFirst(); it->Valid(); it->Next()) {
        keys.push_back(it->key().ToString());
    }
    ASSERT_TRUE(it->status().ok());
    delete it;

    EXPECT_EQ(keys, std::vector<std::string>({"a", "b", "c"}));
}

TEST_F(RocksDBTest, TransactionIsolation) {
    rocksdb::TransactionDB* txn_db;
    rocksdb::TransactionDBOptions txn_options;

    delete db_;  // Close regular DB

    auto status = rocksdb::TransactionDB::Open(
        options_, txn_options, db_path_, &txn_db);
    ASSERT_TRUE(status.ok());

    // Transaction 1: Write
    auto txn1 = txn_db->BeginTransaction(rocksdb::WriteOptions());
    txn1->Put("key", "value1");

    // Transaction 2: Should not see uncommitted write
    auto txn2 = txn_db->BeginTransaction(rocksdb::WriteOptions());
    std::string value;
    status = txn2->Get(rocksdb::ReadOptions(), "key", &value);
    EXPECT_TRUE(status.IsNotFound());

    // Commit and verify
    txn1->Commit();
    status = txn2->Get(rocksdb::ReadOptions(), "key", &value);
    EXPECT_TRUE(status.IsNotFound());  // Snapshot isolation

    delete txn1;
    delete txn2;
    delete txn_db;
}
```

### Stress Testing and Benchmarking

```cpp
#include <benchmark/benchmark.h>
#include <random>

class RocksDBBenchmark {
private:
    rocksdb::DB* db_;
    std::mt19937 rng_;

public:
    RocksDBBenchmark(rocksdb::DB* db) : db_(db), rng_(std::random_device{}()) {}

    std::string RandomKey(size_t length = 16) {
        static const char alphanum[] =
            "0123456789"
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "abcdefghijklmnopqrstuvwxyz";

        std::string key;
        key.reserve(length);

        for (size_t i = 0; i < length; ++i) {
            key += alphanum[rng_() % (sizeof(alphanum) - 1)];
        }

        return key;
    }

    std::string RandomValue(size_t length = 100) {
        return RandomKey(length);
    }
};

static void BM_Put(benchmark::State& state) {
    rocksdb::DB* db;
    rocksdb::Options options;
    options.create_if_missing = true;
    rocksdb::DB::Open(options, "/tmp/bench_db", &db);

    RocksDBBenchmark bench(db);
    rocksdb::WriteOptions write_opts;

    for (auto _ : state) {
        auto key = bench.RandomKey();
        auto value = bench.RandomValue();
        db->Put(write_opts, key, value);
    }

    state.SetItemsProcessed(state.iterations());
    delete db;
    rocksdb::DestroyDB("/tmp/bench_db", options);
}

static void BM_Get(benchmark::State& state) {
    rocksdb::DB* db;
    rocksdb::Options options;
    options.create_if_missing = true;
    rocksdb::DB::Open(options, "/tmp/bench_db", &db);

    RocksDBBenchmark bench(db);

    // Pre-populate
    for (int i = 0; i < 10000; ++i) {
        db->Put(rocksdb::WriteOptions(), bench.RandomKey(), bench.RandomValue());
    }

    rocksdb::ReadOptions read_opts;
    std::string value;

    for (auto _ : state) {
        auto key = bench.RandomKey();
        db->Get(read_opts, key, &value);
    }

    state.SetItemsProcessed(state.iterations());
    delete db;
    rocksdb::DestroyDB("/tmp/bench_db", options);
}

BENCHMARK(BM_Put)->Threads(1)->Threads(4)->Threads(8);
BENCHMARK(BM_Get)->Threads(1)->Threads(4)->Threads(8);

BENCHMARK_MAIN();
```

## 15. Production Deployment Patterns

### Container Deployment (Docker)

```dockerfile
# Dockerfile for RocksDB application
FROM ubuntu:22.04 AS builder

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    libsnappy-dev \
    libgflags-dev \
    liblz4-dev \
    libzstd-dev \
    && rm -rf /var/lib/apt/lists/*

# Build RocksDB with optimizations
WORKDIR /tmp
RUN git clone https://github.com/facebook/rocksdb.git && \
    cd rocksdb && \
    git checkout v9.0.0 && \
    DEBUG_LEVEL=0 PORTABLE=0 make -j$(nproc) shared_lib && \
    make install

# Build application
COPY . /app
WORKDIR /app
RUN mkdir build && cd build && \
    cmake -DCMAKE_BUILD_TYPE=Release .. && \
    make -j$(nproc)

# Runtime image
FROM ubuntu:22.04

RUN apt-get update && apt-get install -y \
    libsnappy1v5 \
    libgflags2.2 \
    liblz4-1 \
    libzstd1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/lib/librocksdb.so* /usr/local/lib/
COPY --from=builder /app/build/myapp /usr/local/bin/

# Data volume
VOLUME ["/data"]

# Run as non-root
RUN useradd -m -u 1000 rocksdb
USER rocksdb

CMD ["/usr/local/bin/myapp", "--db_path=/data"]
```

### Kubernetes Deployment

```yaml
# rocksdb-statefulset.yaml
apiVersion: v1
kind: Service
metadata:
  name: rocksdb-app
spec:
  clusterIP: None
  selector:
    app: rocksdb-app
  ports:
  - port: 8080
    name: http
---
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: rocksdb-app
spec:
  serviceName: "rocksdb-app"
  replicas: 3
  selector:
    matchLabels:
      app: rocksdb-app
  template:
    metadata:
      labels:
        app: rocksdb-app
    spec:
      containers:
      - name: app
        image: myapp:latest
        ports:
        - containerPort: 8080
          name: http
        volumeMounts:
        - name: data
          mountPath: /data
        env:
        - name: ROCKSDB_DB_PATH
          value: "/data/db"
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
  volumeClaimTemplates:
  - metadata:
      name: data
    spec:
      accessModes: [ "ReadWriteOnce" ]
      storageClassName: "fast-ssd"
      resources:
        requests:
          storage: 100Gi
```

### Production Configuration

```cpp
rocksdb::Options GetProductionOptions() {
    rocksdb::Options options;

    // Database creation
    options.create_if_missing = true;
    options.max_open_files = -1;  // Unlimited (rely on OS)

    // Write buffer
    options.write_buffer_size = 128 * 1024 * 1024;  // 128MB
    options.max_write_buffer_number = 4;
    options.min_write_buffer_number_to_merge = 2;

    // SST files
    options.target_file_size_base = 128 * 1024 * 1024;  // 128MB
    options.target_file_size_multiplier = 2;
    options.max_bytes_for_level_base = 512 * 1024 * 1024;  // 512MB
    options.max_bytes_for_level_multiplier = 10;

    // Compaction
    options.level0_file_num_compaction_trigger = 4;
    options.level0_slowdown_writes_trigger = 20;
    options.level0_stop_writes_trigger = 36;
    options.max_background_jobs = 6;
    options.max_subcompactions = 2;

    // Compression
    options.compression_per_level = {
        rocksdb::kNoCompression,
        rocksdb::kNoCompression,
        rocksdb::kLZ4Compression,
        rocksdb::kLZ4Compression,
        rocksdb::kZSTD,
        rocksdb::kZSTD,
    };

    // Block cache (adjust based on available RAM)
    rocksdb::BlockBasedTableOptions table_options;
    table_options.block_cache = rocksdb::NewLRUCache(2ULL * 1024 * 1024 * 1024);
    table_options.block_size = 16 * 1024;
    table_options.cache_index_and_filter_blocks = true;
    table_options.pin_l0_filter_and_index_blocks_in_cache = true;
    table_options.filter_policy.reset(rocksdb::NewBloomFilterPolicy(10));
    options.table_factory.reset(rocksdb::NewBlockBasedTableFactory(table_options));

    // Rate limiting (prevent I/O spikes)
    options.rate_limiter.reset(rocksdb::NewGenericRateLimiter(
        100 * 1024 * 1024));  // 100 MB/s for compaction

    // WAL
    options.WAL_ttl_seconds = 0;
    options.WAL_size_limit_MB = 0;
    options.max_total_wal_size = 1024 * 1024 * 1024;  // 1GB

    // Statistics
    options.statistics = rocksdb::CreateDBStatistics();
    options.stats_dump_period_sec = 600;  // 10 minutes

    // Logging
    options.info_log_level = rocksdb::INFO_LEVEL;
    options.keep_log_file_num = 10;
    options.max_log_file_size = 100 * 1024 * 1024;  // 100MB

    // Safety
    options.paranoid_checks = true;

    return options;
}
```

## 16. Common Patterns and Anti-Patterns

### Pattern: Time-Series Data

```cpp
class TimeSeriesStore {
private:
    rocksdb::DB* db_;

    std::string MakeKey(uint64_t timestamp, const std::string& metric_id) {
        // Big-endian timestamp for proper sorting
        char ts_bytes[8];
        ts_bytes[0] = (timestamp >> 56) & 0xFF;
        ts_bytes[1] = (timestamp >> 48) & 0xFF;
        ts_bytes[2] = (timestamp >> 40) & 0xFF;
        ts_bytes[3] = (timestamp >> 32) & 0xFF;
        ts_bytes[4] = (timestamp >> 24) & 0xFF;
        ts_bytes[5] = (timestamp >> 16) & 0xFF;
        ts_bytes[6] = (timestamp >> 8) & 0xFF;
        ts_bytes[7] = timestamp & 0xFF;

        return std::string(ts_bytes, 8) + ":" + metric_id;
    }

public:
    TimeSeriesStore(rocksdb::DB* db) : db_(db) {}

    void RecordMetric(const std::string& metric_id, uint64_t timestamp, double value) {
        std::string key = MakeKey(timestamp, metric_id);
        std::string value_str = std::to_string(value);
        db_->Put(rocksdb::WriteOptions(), key, value_str);
    }

    std::vector<double> QueryRange(const std::string& metric_id,
                                   uint64_t start_ts,
                                   uint64_t end_ts) {
        std::vector<double> results;

        std::string start_key = MakeKey(start_ts, metric_id);
        std::string end_key = MakeKey(end_ts, metric_id);

        rocksdb::ReadOptions opts;
        auto it = db_->NewIterator(opts);

        for (it->Seek(start_key); it->Valid() && it->key().ToString() <= end_key; it->Next()) {
            results.push_back(std::stod(it->value().ToString()));
        }

        delete it;
        return results;
    }
};
```

### Pattern: Secondary Indexing

```cpp
class SecondaryIndex {
private:
    rocksdb::DB* db_;

public:
    // Primary: user:<id> -> JSON
    // Index: email:<email> -> <id>
    // Index: age:<age>:<id> -> ""

    rocksdb::Status AddUser(const std::string& id,
                           const std::string& email,
                           int age,
                           const std::string& user_json) {
        rocksdb::WriteBatch batch;

        // Primary record
        batch.Put("user:" + id, user_json);

        // Email index (unique)
        batch.Put("email:" + email, id);

        // Age index (non-unique, include ID in key)
        batch.Put("age:" + std::to_string(age) + ":" + id, "");

        return db_->Write(rocksdb::WriteOptions(), &batch);
    }

    rocksdb::Status GetUserByEmail(const std::string& email,
                                  std::string* user_json) {
        std::string id;
        auto s = db_->Get(rocksdb::ReadOptions(), "email:" + email, &id);
        if (!s.ok()) return s;

        return db_->Get(rocksdb::ReadOptions(), "user:" + id, user_json);
    }

    std::vector<std::string> GetUsersByAge(int age) {
        std::vector<std::string> results;
        std::string prefix = "age:" + std::to_string(age) + ":";

        rocksdb::ReadOptions opts;
        opts.prefix_same_as_start = true;

        auto it = db_->NewIterator(opts);
        for (it->Seek(prefix); it->Valid() && it->key().starts_with(prefix); it->Next()) {
            // Extract ID from key
            std::string key = it->key().ToString();
            std::string id = key.substr(prefix.length());

            std::string user_json;
            if (db_->Get(rocksdb::ReadOptions(), "user:" + id, &user_json).ok()) {
                results.push_back(user_json);
            }
        }

        delete it;
        return results;
    }
};
```

### Anti-Pattern: Large Values

```cpp
// ❌ BAD: Storing large blobs directly
void BadPattern(rocksdb::DB* db) {
    std::string large_video_data(100 * 1024 * 1024, 'x');  // 100MB
    db->Put(rocksdb::WriteOptions(), "video:1", large_video_data);
    // Problems:
    // - Bloats memtable
    // - Increases write amplification
    // - Slows down compaction
    // - Wastes block cache
}

// ✅ GOOD: Store large blobs externally, keep metadata in RocksDB
void GoodPattern(rocksdb::DB* db) {
    // Store video in S3/filesystem
    std::string s3_url = UploadToS3("video_data");

    // Store only metadata in RocksDB
    std::string metadata = R"({
        "url": ")" + s3_url + R"(",
        "size": 104857600,
        "duration": 120,
        "format": "mp4"
    })";

    db->Put(rocksdb::WriteOptions(), "video:1:meta", metadata);
}
```

### Anti-Pattern: Hot Key

```cpp
// ❌ BAD: Single counter with high contention
void BadCounterPattern(rocksdb::DB* db) {
    for (int i = 0; i < 1000; ++i) {
        std::string value;
        db->Get(rocksdb::ReadOptions(), "global_counter", &value);
        int count = value.empty() ? 0 : std::stoi(value);
        count++;
        db->Put(rocksdb::WriteOptions(), "global_counter", std::to_string(count));
    }
    // Problem: Single hot key limits write throughput
}

// ✅ GOOD: Shard counters
void GoodCounterPattern(rocksdb::DB* db) {
    int num_shards = 10;

    for (int i = 0; i < 1000; ++i) {
        int shard = rand() % num_shards;
        std::string key = "counter:shard:" + std::to_string(shard);

        // Use merge operator for atomic increment
        db->Merge(rocksdb::WriteOptions(), key, "1");
    }

    // Read total by summing shards
    int total = 0;
    for (int shard = 0; shard < num_shards; ++shard) {
        std::string key = "counter:shard:" + std::to_string(shard);
        std::string value;
        if (db->Get(rocksdb::ReadOptions(), key, &value).ok()) {
            total += std::stoi(value);
        }
    }
}
```

## 17. Migration Strategies

### Migrating from LevelDB

```cpp
// LevelDB and RocksDB have similar APIs
// Key differences:
// 1. Column families (RocksDB only)
// 2. Transaction support (RocksDB only)
// 3. Better compression options (RocksDB)
// 4. More tuning knobs (RocksDB)

// LevelDB code
leveldb::DB* leveldb;
leveldb::Options leveldb_options;
leveldb::DB::Open(leveldb_options, "/path/to/db", &leveldb);

// Equivalent RocksDB code (mostly compatible)
rocksdb::DB* rocksdb;
rocksdb::Options rocksdb_options;
rocksdb::DB::Open(rocksdb_options, "/path/to/db", &rocksdb);

// Data migration: LevelDB SST files can be imported
rocksdb::IngestExternalFileOptions ingest_options;
rocksdb->IngestExternalFile({"/path/to/leveldb/file.sst"}, ingest_options);
```

### Migrating from SQLite

```cpp
#include <sqlite3.h>
#include <rocksdb/db.h>

class SQLiteToRocksDB {
public:
    static void Migrate(const std::string& sqlite_path,
                       const std::string& rocksdb_path) {
        // Open SQLite
        sqlite3* sqlite_db;
        sqlite3_open(sqlite_path.c_str(), &sqlite_db);

        // Open RocksDB
        rocksdb::DB* rocks_db;
        rocksdb::Options options;
        options.create_if_missing = true;
        rocksdb::DB::Open(options, rocksdb_path, &rocks_db);

        // Migrate data
        const char* sql = "SELECT key, value FROM kv_table";
        sqlite3_stmt* stmt;
        sqlite3_prepare_v2(sqlite_db, sql, -1, &stmt, nullptr);

        rocksdb::WriteBatch batch;
        int count = 0;

        while (sqlite3_step(stmt) == SQLITE_ROW) {
            const unsigned char* key = sqlite3_column_text(stmt, 0);
            const unsigned char* value = sqlite3_column_text(stmt, 1);

            batch.Put(
                reinterpret_cast<const char*>(key),
                reinterpret_cast<const char*>(value)
            );

            count++;
            if (count % 10000 == 0) {
                rocks_db->Write(rocksdb::WriteOptions(), &batch);
                batch.Clear();
                std::cout << "Migrated " << count << " records" << std::endl;
            }
        }

        // Final batch
        if (batch.Count() > 0) {
            rocks_db->Write(rocksdb::WriteOptions(), &batch);
        }

        sqlite3_finalize(stmt);
        sqlite3_close(sqlite_db);
        delete rocks_db;

        std::cout << "Migration complete: " << count << " records" << std::endl;
    }
};
```

## 18. Troubleshooting Guide

### Write Stalls

```cpp
// Symptom: Writes become very slow or blocked
// Cause: Too many L0 files

void DiagnoseWriteStall(rocksdb::DB* db) {
    uint64_t num_l0_files;
    db->GetIntProperty("rocksdb.num-files-at-level0", &num_l0_files);

    uint64_t pending_compaction;
    db->GetIntProperty("rocksdb.estimate-pending-compaction-bytes",
                      &pending_compaction);

    std::cout << "L0 files: " << num_l0_files << std::endl;
    std::cout << "Pending compaction: " << pending_compaction / (1024*1024)
              << " MB" << std::endl;

    // Solution 1: Increase L0 trigger
    // options.level0_slowdown_writes_trigger = 20;
    // options.level0_stop_writes_trigger = 36;

    // Solution 2: More background compaction threads
    // options.max_background_compactions = 8;

    // Solution 3: Larger write buffer
    // options.write_buffer_size = 256 * 1024 * 1024;
}
```

### High Read Latency

```cpp
void DiagnoseReadLatency(rocksdb::DB* db) {
    // Enable per-operation stats
    rocksdb::SetPerfLevel(rocksdb::kEnableTimeExceptForMutex);
    rocksdb::get_perf_context()->Reset();

    // Perform read
    std::string value;
    db->Get(rocksdb::ReadOptions(), "test_key", &value);

    auto perf = rocksdb::get_perf_context();

    std::cout << "Block cache hit: " << perf->block_cache_hit_count << std::endl;
    std::cout << "Block cache miss: " << perf->block_cache_miss_count << std::endl;
    std::cout << "Bloom filter useful: " << perf->bloom_filter_useful << std::endl;
    std::cout << "Get from memtable time: " << perf->get_from_memtable_time << " ns" << std::endl;

    // Solutions:
    // - Low cache hit rate: Increase block_cache size
    // - Bloom filter not helping: Increase bits_per_key
    // - High get_from_memtable_time: Data might be in memtable (good)
    // - Many block reads: Data is in SST files, check if hot data should be cached
}
```

### Memory Usage

```cpp
void DiagnoseMemoryUsage(rocksdb::DB* db) {
    uint64_t memtable_size, block_cache_size, table_readers_size;

    db->GetIntProperty("rocksdb.cur-size-all-mem-tables", &memtable_size);
    db->GetIntProperty("rocksdb.block-cache-usage", &block_cache_size);
    db->GetIntProperty("rocksdb.estimate-table-readers-mem", &table_readers_size);

    std::cout << "Memory usage:" << std::endl;
    std::cout << "  MemTables: " << memtable_size / (1024*1024) << " MB" << std::endl;
    std::cout << "  Block cache: " << block_cache_size / (1024*1024) << " MB" << std::endl;
    std::cout << "  Index/filters: " << table_readers_size / (1024*1024) << " MB" << std::endl;
    std::cout << "  Total: " << (memtable_size + block_cache_size + table_readers_size) / (1024*1024)
              << " MB" << std::endl;

    // Solutions for high memory usage:
    // - Reduce write_buffer_size
    // - Reduce block_cache size
    // - Enable cache_index_and_filter_blocks with smaller cache
}
```

### Compaction Issues

```cpp
void DiagnoseCompaction(rocksdb::DB* db) {
    std::string stats;
    db->GetProperty("rocksdb.stats", &stats);
    std::cout << stats << std::endl;

    uint64_t running_compactions, running_flushes;
    db->GetIntProperty("rocksdb.num-running-compactions", &running_compactions);
    db->GetIntProperty("rocksdb.num-running-flushes", &running_flushes);

    std::cout << "Running compactions: " << running_compactions << std::endl;
    std::cout << "Running flushes: " << running_flushes << std::endl;

    // Check per-level stats
    for (int level = 0; level < 7; ++level) {
        uint64_t num_files, size;
        std::string prop = "rocksdb.num-files-at-level" + std::to_string(level);
        db->GetIntProperty(prop, &num_files);

        prop = "rocksdb.total-sst-files-size-at-level" + std::to_string(level);
        db->GetIntProperty(prop, &size);

        std::cout << "Level " << level << ": " << num_files
                  << " files, " << size / (1024*1024) << " MB" << std::endl;
    }
}
```

## 19. Performance Tuning Checklist

### Hardware Optimization

```markdown
1. **Storage**
   - Use NVMe SSDs for best performance
   - Enable TRIM/discard for SSDs
   - Use XFS or ext4 with noatime mount option
   - RAID 0 for maximum throughput (if reliability handled elsewhere)

2. **Memory**
   - Block cache should be 30-50% of available RAM
   - Leave room for OS page cache
   - Consider huge pages for large block cache

3. **CPU**
   - More cores help compaction parallelism
   - Modern CPU with SSE4.2 for CRC32c checksums
   - Compile with -march=native for CPU-specific optimizations

4. **Network** (if applicable)
   - 10GbE or faster for replication
   - Low-latency network for distributed systems
```

### Software Optimization

```cpp
rocksdb::Options GetOptimizedOptions() {
    rocksdb::Options options;

    // === Write Performance ===
    options.write_buffer_size = 128 * 1024 * 1024;
    options.max_write_buffer_number = 4;
    options.level0_file_num_compaction_trigger = 8;
    options.target_file_size_base = 128 * 1024 * 1024;

    // === Read Performance ===
    rocksdb::BlockBasedTableOptions table_options;
    table_options.block_cache = rocksdb::NewLRUCache(2ULL * 1024 * 1024 * 1024);
    table_options.filter_policy.reset(rocksdb::NewBloomFilterPolicy(10));
    table_options.cache_index_and_filter_blocks = true;
    table_options.pin_l0_filter_and_index_blocks_in_cache = true;
    options.table_factory.reset(rocksdb::NewBlockBasedTableFactory(table_options));

    // === Compaction ===
    options.max_background_jobs = 8;
    options.max_subcompactions = 2;

    // === Compression ===
    options.compression = rocksdb::kLZ4Compression;
    options.bottommost_compression = rocksdb::kZSTD;

    // === Direct I/O (bypass OS cache) ===
    options.use_direct_reads = true;
    options.use_direct_io_for_flush_and_compaction = true;

    // === Statistics ===
    options.statistics = rocksdb::CreateDBStatistics();

    return options;
}
```

### Monitoring Metrics

```cpp
void MonitorCriticalMetrics(rocksdb::DB* db) {
    auto stats = db->GetOptions().statistics;

    // Write throughput
    uint64_t bytes_written = stats->getTickerCount(rocksdb::BYTES_WRITTEN);

    // Read throughput
    uint64_t bytes_read = stats->getTickerCount(rocksdb::BYTES_READ);

    // Cache hit rate
    uint64_t cache_hit = stats->getTickerCount(rocksdb::BLOCK_CACHE_HIT);
    uint64_t cache_miss = stats->getTickerCount(rocksdb::BLOCK_CACHE_MISS);
    double hit_rate = (double)cache_hit / (cache_hit + cache_miss) * 100.0;

    // Bloom filter effectiveness
    uint64_t bloom_useful = stats->getTickerCount(rocksdb::BLOOM_FILTER_USEFUL);
    uint64_t bloom_checked = stats->getTickerCount(rocksdb::BLOOM_FILTER_PREFIX_CHECKED);
    double bloom_rate = (double)bloom_useful / bloom_checked * 100.0;

    // Compaction stats
    uint64_t compaction_bytes = stats->getTickerCount(rocksdb::COMPACT_WRITE_BYTES);
    uint64_t compaction_time = stats->getTickerCount(rocksdb::COMPACTION_CPU_TIME);

    std::cout << "Cache hit rate: " << hit_rate << "%" << std::endl;
    std::cout << "Bloom filter useful rate: " << bloom_rate << "%" << std::endl;
    std::cout << "Bytes written: " << bytes_written / (1024*1024) << " MB" << std::endl;
    std::cout << "Compaction time: " << compaction_time / 1000000 << " ms" << std::endl;
}
```

## 20. RocksDB vs Alternatives

### RocksDB vs LevelDB

| Feature | RocksDB | LevelDB |
|---------|---------|---------|
| Write throughput | Higher (parallel compaction) | Lower |
| Column families | ✅ Yes | ❌ No |
| Transactions | ✅ Yes | ❌ No |
| Compression | Multiple algorithms | Snappy only |
| Tuning options | Extensive | Minimal |
| Maintenance | Active (Meta) | Minimal (Google) |
| Use case | Production databases | Embedded, prototypes |

### RocksDB vs LMDB

| Feature | RocksDB | LMDB |
|---------|---------|------|
| Architecture | LSM-tree | B+ tree |
| Write amplification | High (10-30x) | Low (1-2x) |
| Read amplification | Lower (bloom filters) | Higher (no bloom) |
| Space amplification | Higher (compaction) | Lower |
| Concurrent reads | Excellent | Excellent |
| Concurrent writes | Excellent | Single writer |
| Use case | Write-heavy | Read-heavy, single writer |

### RocksDB vs SQLite

| Feature | RocksDB | SQLite |
|---------|---------|--------|
| Data model | Key-value | Relational (SQL) |
| Query language | Scans, prefix seeks | SQL |
| Indexing | Manual (secondary KV) | Automatic (B-tree) |
| Transactions | Optimistic/Pessimistic | ACID with WAL |
| Write throughput | Much higher | Lower |
| Use case | High-performance KV | Structured data, SQL |

## 21. Resources and References

### Official Documentation
- **RocksDB Wiki**: https://github.com/facebook/rocksdb/wiki
- **Tuning Guide**: https://github.com/facebook/rocksdb/wiki/RocksDB-Tuning-Guide
- **API Documentation**: https://rocksdb.org/docs/
- **Blog**: https://rocksdb.org/blog/

### Books and Papers
- *RocksDB: Evolution of Development Priorities in a Key-Value Store Serving Large-Scale Applications* (USENIX 2021)
- *WiscKey: Separating Keys from Values in SSD-conscious Storage* (FAST 2016)

### Performance and Benchmarking
- Official benchmarks: https://github.com/facebook/rocksdb/wiki/Performance-Benchmarks
- db_bench tool for custom benchmarks
- YCSB (Yahoo! Cloud Serving Benchmark) RocksDB binding

### Community and Support
- **GitHub**: https://github.com/facebook/rocksdb
- **Discussions**: https://github.com/facebook/rocksdb/discussions
- **Slack**: RocksDB community (link in GitHub README)
- **Stack Overflow**: Tag `rocksdb`

### Production Users
- **Meta (Facebook)**: MyRocks MySQL storage engine, ZippyDB
- **Netflix**: EVCache
- **LinkedIn**: LinkedIn Feeds, Ambry
- **Uber**: Cherami message queue
- **Airbnb**: Database caching layer
- **Yahoo**: Cloud Serving (YCS)

### Related Technologies
- **MyRocks**: MySQL with RocksDB storage engine
- **MongoRocks**: MongoDB with RocksDB storage
- **CockroachDB**: Uses RocksDB as storage engine
- **TiKV**: Distributed KV store built on RocksDB

### Tools
- **ldb**: RocksDB command-line tool for inspection
- **sst_dump**: Inspect SST file contents
- **db_bench**: Official benchmarking tool
- **trace_analyzer**: Analyze RocksDB traces for optimization

---

## Quick Start Example

```cpp
#include <rocksdb/db.h>
#include <iostream>

int main() {
    rocksdb::DB* db;
    rocksdb::Options options;
    options.create_if_missing = true;

    // Open database
    rocksdb::Status status = rocksdb::DB::Open(options, "/tmp/testdb", &db);
    if (!status.ok()) {
        std::cerr << "Unable to open database: " << status.ToString() << std::endl;
        return 1;
    }

    // Write
    status = db->Put(rocksdb::WriteOptions(), "key1", "value1");
    if (!status.ok()) {
        std::cerr << "Write failed: " << status.ToString() << std::endl;
    }

    // Read
    std::string value;
    status = db->Get(rocksdb::ReadOptions(), "key1", &value);
    if (status.ok()) {
        std::cout << "key1: " << value << std::endl;
    } else {
        std::cerr << "Read failed: " << status.ToString() << std::endl;
    }

    // Cleanup
    delete db;
    return 0;
}
```

Compile with:
```bash
g++ -std=c++17 -O3 example.cpp -lrocksdb -lpthread -ldl -lz -lsnappy -llz4 -lzstd -o example
./example
```

This guide covers the essential aspects of RocksDB with emphasis on modern C++ practices, security, and performance optimization for production deployments.

---

**End of RocksDB Development Guidelines**
