# LevelDB Development Guidelines
Mandatory coding standards and development practices for LevelDB development. LevelDB C++ API, WriteBatch, Snapshots, custom comparators, language bindings.

---

**Agent Profile**: The LevelDB Expert
**Role**: Senior Embedded Storage Engineer & Key-Value Store Specialist
**Objective**: Generate production-ready, performant and reliable embedded key-value storage solutions.
**Tools**: LevelDB C++ API, WriteBatch, Snapshots, custom comparators, language bindings

---

## 1. Core Philosophies: LEVELDB-FIRST

The agent must adhere to the **LEVELDB-FIRST** principles for every LevelDB implementation:

**Test-Driven Development (TDD)**: ALWAYS write tests BEFORE implementation (Red-Green-Refactor cycle mandatory).
**Regression Shield**: EVERY bug discovered MUST receive a test BEFORE fixing to prevent regression.

- **L**SM-aware: Design for single-writer, batch writes, and compaction; avoid write amplification pitfalls.
- **E**mbedded model: No network layer; in-process only; application manages concurrency and deployment.
- **V**alue semantics: Use WriteBatch for atomic multi-key updates; use snapshots for consistent reads.
- **E**rror handling: Check Status on every operation; handle corruption and I/O errors explicitly.
- **L**ock ordering: Single writer; multiple readers; document any application-level locking.
- **D**ata layout: Design keys for range scans and prefix iteration; use custom comparators when needed.
- **B**ackup and recovery: Use consistent backup procedures; test restore; avoid open DB during backup.
- **D**eterministic: Avoid undefined behavior; use fixed key/value formats; test on target platforms.

**Verified Code**: Agent-generated code MUST build, pass tests, and handle LevelDB Status/errors before delivery.

---

## 2. Core Concepts and Architecture

LevelDB is a fast key-value storage library developed by Google that provides an ordered mapping from string keys to string values. It's an embedded database library (not a client-server system) built on Log-Structured Merge (LSM) tree architecture.

### LSM-Tree Architecture

```
Write Path:
┌─────────────────────────────────────────────────────────────┐
│ 1. Write to WAL (Write-Ahead Log)                          │
│    → Crash recovery guarantee                               │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. Write to MemTable (in-memory sorted structure)          │
│    → Skip list implementation                               │
│    → Fast writes: O(log n)                                  │
└─────────────────────────────────────────────────────────────┘
                            ↓ (when full)
┌─────────────────────────────────────────────────────────────┐
│ 3. Flush to Level 0 SSTable (Sorted String Table)          │
│    → Immutable files on disk                                │
│    → May have overlapping key ranges                        │
└─────────────────────────────────────────────────────────────┘
                            ↓ (background compaction)
┌─────────────────────────────────────────────────────────────┐
│ 4. Compact to Level 1-6                                     │
│    → Non-overlapping key ranges per level                   │
│    → Each level ~10x size of previous                       │
│    → Sorted merge during compaction                         │
└─────────────────────────────────────────────────────────────┘

Read Path:
1. Check MemTable (fastest)
2. Check Immutable MemTable
3. Check Level 0 SSTables (may check multiple)
4. Check Level 1-6 SSTables (binary search per level)
```

### Key Characteristics

**Single Writer, Multiple Readers:**
- Only ONE write thread at a time (internal mutex)
- Multiple concurrent read threads supported
- Snapshots provide consistent point-in-time views

**Write Amplification:**
- Data rewritten multiple times during compaction
- Typical amplification: 10-20x
- Trade-off for good read performance

**Ordered Keys:**
- Keys stored in sorted order
- Enables efficient range scans
- Custom comparators supported

**No Transactions:**
- Atomic WriteBatch operations only
- No multi-operation ACID transactions
- No isolation between reads and writes

### Embedded Library Model

```cpp
// LevelDB runs in-process (no network layer)
#include <leveldb/db.h>

int main() {
    leveldb::DB* db;
    leveldb::Options options;
    options.create_if_missing = true;

    // Direct function calls - microsecond latency
    leveldb::Status status = leveldb::DB::Open(options, "/tmp/testdb", &db);

    // Put, Get, Delete are simple method calls
    db->Put(leveldb::WriteOptions(), "key", "value");

    delete db;
    return 0;
}
```

**Benefits:**
- No network overhead (nanosecond to microsecond latency)
- No serialization/deserialization cost
- Simple deployment (just a library)
- Process-level data isolation

**Limitations:**
- Single-machine only
- No built-in replication
- No query language
- Application manages concurrency

## 3. Installation and Setup

### Ubuntu/Debian Installation

```bash
# Install from package manager
sudo apt-get update
sudo apt-get install -y libleveldb-dev libleveldb1d

# Install Snappy compression library (recommended)
sudo apt-get install -y libsnappy-dev
```

### Building from Source

```bash
# Clone repository
git clone --recurse-submodules https://github.com/google/leveldb.git
cd leveldb

# Build with CMake
mkdir -p build
cd build

# Release build with optimizations
cmake -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=/usr/local \
      -DLEVELDB_BUILD_TESTS=OFF \
      -DLEVELDB_BUILD_BENCHMARKS=OFF \
      ..

# Compile
make -j$(nproc)

# Install
sudo make install
sudo ldconfig
```

### macOS Installation

```bash
# Using Homebrew
brew install leveldb

# Or build from source
git clone --recurse-submodules https://github.com/google/leveldb.git
cd leveldb
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(sysctl -n hw.ncpu)
sudo make install
```

### CMake Integration

```cmake
# CMakeLists.txt
cmake_minimum_required(VERSION 3.10)
project(MyLevelDBApp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compiler optimizations
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native")

# Find LevelDB
find_package(leveldb REQUIRED)

add_executable(myapp main.cpp)
target_link_libraries(myapp leveldb::leveldb)

# Link Snappy for compression
find_package(Snappy)
if(Snappy_FOUND)
    target_link_libraries(myapp ${Snappy_LIBRARIES})
endif()
```

### Verification

```cpp
// test_leveldb.cpp
#include <leveldb/db.h>
#include <iostream>

int main() {
    leveldb::DB* db;
    leveldb::Options options;
    options.create_if_missing = true;

    leveldb::Status status = leveldb::DB::Open(options, "/tmp/testdb", &db);

    if (!status.ok()) {
        std::cerr << "Failed to open database: " << status.ToString() << std::endl;
        return 1;
    }

    std::cout << "LevelDB version: " << leveldb::kMajorVersion << "."
              << leveldb::kMinorVersion << std::endl;

    delete db;
    return 0;
}
```

Compile and run:
```bash
g++ -std=c++17 -O3 test_leveldb.cpp -lleveldb -o test_leveldb
./test_leveldb
```

## 4. C++ API - Basic Operations

### Opening and Closing Database

```cpp
#include <leveldb/db.h>
#include <leveldb/options.h>
#include <leveldb/write_batch.h>
#include <iostream>
#include <memory>

class LevelDBWrapper {
private:
    std::unique_ptr<leveldb::DB> db_;
    leveldb::Options options_;

public:
    leveldb::Status Open(const std::string& db_path) {
        options_.create_if_missing = true;

        // Write buffer size (default 4MB, increase for write-heavy workloads)
        options_.write_buffer_size = 16 * 1024 * 1024;  // 16MB

        // Number of open files (default 1000)
        options_.max_open_files = 5000;

        // Block size (default 4KB)
        options_.block_size = 16 * 1024;  // 16KB

        // Compression (Snappy by default)
        options_.compression = leveldb::kSnappyCompression;

        // Block cache (default 8MB)
        options_.block_cache = leveldb::NewLRUCache(256 * 1024 * 1024);  // 256MB

        // Error if database exists (for new databases)
        // options_.error_if_exists = true;

        leveldb::DB* db_ptr;
        leveldb::Status status = leveldb::DB::Open(options_, db_path, &db_ptr);

        if (status.ok()) {
            db_.reset(db_ptr);
        }

        return status;
    }

    leveldb::DB* Get() { return db_.get(); }

    ~LevelDBWrapper() {
        // Destructor automatically closes database
        db_.reset();
    }
};

// Usage
int main() {
    LevelDBWrapper wrapper;
    leveldb::Status status = wrapper.Open("/tmp/mydb");

    if (!status.ok()) {
        std::cerr << "Failed to open: " << status.ToString() << std::endl;
        return 1;
    }

    // Use database..

    return 0;
}
```

### Put, Get, Delete Operations

```cpp
#include <leveldb/db.h>
#include <string>

class KeyValueStore {
private:
    leveldb::DB* db_;

public:
    KeyValueStore(leveldb::DB* db) : db_(db) {}

    // Write a key-value pair
    leveldb::Status Put(const std::string& key, const std::string& value) {
        leveldb::WriteOptions write_options;
        write_options.sync = false;  // Async write (faster, less durable)
        // write_options.sync = true;  // Sync write (slower, more durable)

        return db_->Put(write_options, key, value);
    }

    // Read a value by key
    leveldb::Status Get(const std::string& key, std::string* value) {
        leveldb::ReadOptions read_options;
        read_options.verify_checksums = false;  // Skip checksum (faster)
        read_options.fill_cache = true;          // Add to cache (default)

        return db_->Get(read_options, key, value);
    }

    // Delete a key
    leveldb::Status Delete(const std::string& key) {
        leveldb::WriteOptions write_options;
        write_options.sync = false;

        return db_->Delete(write_options, key);
    }

    // Check if key exists (without reading value)
    bool Exists(const std::string& key) {
        std::string value;
        leveldb::Status status = db_->Get(leveldb::ReadOptions(), key, &value);
        return status.ok();
    }
};

// Usage examples
void BasicOperations(leveldb::DB* db) {
    KeyValueStore store(db);

    // Write
    leveldb::Status s = store.Put("user:1:name", "Alice");
    if (!s.ok()) {
        std::cerr << "Put failed: " << s.ToString() << std::endl;
    }

    // Read
    std::string value;
    s = store.Get("user:1:name", &value);
    if (s.ok()) {
        std::cout << "Value: " << value << std::endl;
    } else if (s.IsNotFound()) {
        std::cout << "Key not found" << std::endl;
    } else {
        std::cerr << "Get failed: " << s.ToString() << std::endl;
    }

    // Check existence
    if (store.Exists("user:1:name")) {
        std::cout << "Key exists" << std::endl;
    }

    // Delete
    s = store.Delete("user:1:name");
    if (!s.ok()) {
        std::cerr << "Delete failed: " << s.ToString() << std::endl;
    }
}
```

### Atomic Batch Writes

```cpp
#include <leveldb/write_batch.h>

class BatchOperations {
public:
    // Atomic batch write (all or nothing)
    static leveldb::Status BatchWrite(
        leveldb::DB* db,
        const std::vector<std::pair<std::string, std::string>>& puts,
        const std::vector<std::string>& deletes) {

        leveldb::WriteBatch batch;

        // Add all put operations
        for (const auto& [key, value] : puts) {
            batch.Put(key, value);
        }

        // Add all delete operations
        for (const auto& key : deletes) {
            batch.Delete(key);
        }

        // Write batch atomically
        leveldb::WriteOptions options;
        options.sync = false;

        return db->Write(options, &batch);
    }

    // Transfer operation example (debit one account, credit another)
    static leveldb::Status Transfer(leveldb::DB* db,
                                    const std::string& from_account,
                                    const std::string& to_account,
                                    int amount) {
        // Read current balances
        std::string from_balance_str, to_balance_str;

        leveldb::Status s = db->Get(leveldb::ReadOptions(), from_account, &from_balance_str);
        if (!s.ok()) return s;

        s = db->Get(leveldb::ReadOptions(), to_account, &to_balance_str);
        if (!s.ok()) return s;

        int from_balance = std::stoi(from_balance_str);
        int to_balance = std::stoi(to_balance_str);

        // Validate
        if (from_balance < amount) {
            return leveldb::Status::InvalidArgument("Insufficient funds");
        }

        // Update balances
        from_balance -= amount;
        to_balance += amount;

        // Write atomically
        leveldb::WriteBatch batch;
        batch.Put(from_account, std::to_string(from_balance));
        batch.Put(to_account, std::to_string(to_balance));

        return db->Write(leveldb::WriteOptions(), &batch);
    }
};

// Usage
void ExampleBatchWrite(leveldb::DB* db) {
    // Multiple writes in single atomic operation
    std::vector<std::pair<std::string, std::string>> puts = {
        {"user:1:name", "Alice"},
        {"user:1:email", "alice@example.com"},
        {"user:1:age", "30"}
    };

    std::vector<std::string> deletes = {
        "user:1:temp_data"
    };

    leveldb::Status s = BatchOperations::BatchWrite(db, puts, deletes);
    if (!s.ok()) {
        std::cerr << "Batch write failed: " << s.ToString() << std::endl;
    }
}
```

### Iteration and Range Scans

```cpp
#include <leveldb/iterator.h>

class RangeScan {
public:
    // Forward iteration (ascending order)
    static void ForwardScan(leveldb::DB* db, const std::string& start_key) {
        leveldb::ReadOptions options;
        std::unique_ptr<leveldb::Iterator> it(db->NewIterator(options));

        for (it->Seek(start_key); it->Valid(); it->Next()) {
            std::cout << it->key().ToString() << ": "
                      << it->value().ToString() << std::endl;
        }

        // Check for errors
        if (!it->status().ok()) {
            std::cerr << "Iterator error: " << it->status().ToString() << std::endl;
        }
    }

    // Backward iteration (descending order)
    static void BackwardScan(leveldb::DB* db) {
        leveldb::ReadOptions options;
        std::unique_ptr<leveldb::Iterator> it(db->NewIterator(options));

        for (it->SeekToLast(); it->Valid(); it->Prev()) {
            std::cout << it->key().ToString() << ": "
                      << it->value().ToString() << std::endl;
        }

        if (!it->status().ok()) {
            std::cerr << "Iterator error: " << it->status().ToString() << std::endl;
        }
    }

    // Range scan with prefix
    static std::vector<std::pair<std::string, std::string>>
    PrefixScan(leveldb::DB* db, const std::string& prefix) {
        std::vector<std::pair<std::string, std::string>> results;

        leveldb::ReadOptions options;
        std::unique_ptr<leveldb::Iterator> it(db->NewIterator(options));

        for (it->Seek(prefix); it->Valid(); it->Next()) {
            std::string key = it->key().ToString();

            // Stop if key doesn't start with prefix
            if (key.compare(0, prefix.length(), prefix) != 0) {
                break;
            }

            results.emplace_back(key, it->value().ToString());
        }

        return results;
    }

    // Range scan between two keys
    static std::vector<std::pair<std::string, std::string>>
    RangeBetween(leveldb::DB* db,
                 const std::string& start_key,
                 const std::string& end_key) {
        std::vector<std::pair<std::string, std::string>> results;

        leveldb::ReadOptions options;
        std::unique_ptr<leveldb::Iterator> it(db->NewIterator(options));

        for (it->Seek(start_key); it->Valid(); it->Next()) {
            std::string key = it->key().ToString();

            // Stop if beyond end key
            if (key > end_key) {
                break;
            }

            results.emplace_back(key, it->value().ToString());
        }

        return results;
    }

    // Count keys with prefix
    static size_t CountPrefix(leveldb::DB* db, const std::string& prefix) {
        size_t count = 0;

        leveldb::ReadOptions options;
        options.fill_cache = false;  // Don't pollute cache during count

        std::unique_ptr<leveldb::Iterator> it(db->NewIterator(options));

        for (it->Seek(prefix); it->Valid(); it->Next()) {
            if (it->key().ToString().compare(0, prefix.length(), prefix) != 0) {
                break;
            }
            count++;
        }

        return count;
    }
};

// Usage
void IteratorExamples(leveldb::DB* db) {
    // Populate test data
    db->Put(leveldb::WriteOptions(), "user:1:name", "Alice");
    db->Put(leveldb::WriteOptions(), "user:2:name", "Bob");
    db->Put(leveldb::WriteOptions(), "user:3:name", "Charlie");
    db->Put(leveldb::WriteOptions(), "product:1:name", "Widget");

    // Scan all users
    std::cout << "All users:" << std::endl;
    auto users = RangeScan::PrefixScan(db, "user:");
    for (const auto& [key, value] : users) {
        std::cout << "  " << key << " = " << value << std::endl;
    }

    // Count users
    size_t user_count = RangeScan::CountPrefix(db, "user:");
    std::cout << "Total users: " << user_count << std::endl;

    // Range scan
    auto range = RangeScan::RangeBetween(db, "user:1", "user:2");
    std::cout << "Range user:1 to user:2:" << std::endl;
    for (const auto& [key, value] : range) {
        std::cout << "  " << key << " = " << value << std::endl;
    }
}
```

## 5. Snapshots - Consistent Point-in-Time Views

Snapshots provide a consistent read view of the database at a specific point in time.

```cpp
#include <leveldb/snapshot.h>

class SnapshotExample {
public:
    // Consistent read across multiple operations
    static void ConsistentRead(leveldb::DB* db) {
        // Take snapshot
        const leveldb::Snapshot* snapshot = db->GetSnapshot();

        leveldb::ReadOptions options;
        options.snapshot = snapshot;

        // All reads see database state at snapshot time
        std::string value1, value2;
        db->Get(options, "key1", &value1);
        db->Get(options, "key2", &value2);

        // Even if other threads modify data, these reads are consistent

        // Release snapshot when done
        db->ReleaseSnapshot(snapshot);
    }

    // RAII wrapper for snapshot
    class ScopedSnapshot {
    private:
        leveldb::DB* db_;
        const leveldb::Snapshot* snapshot_;

    public:
        explicit ScopedSnapshot(leveldb::DB* db) : db_(db) {
            snapshot_ = db_->GetSnapshot();
        }

        ~ScopedSnapshot() {
            if (snapshot_) {
                db_->ReleaseSnapshot(snapshot_);
            }
        }

        const leveldb::Snapshot* Get() const { return snapshot_; }

        // Delete copy constructor and assignment
        ScopedSnapshot(const ScopedSnapshot&) = delete;
        ScopedSnapshot& operator=(const ScopedSnapshot&) = delete;
    };

    // Iterate with snapshot consistency
    static std::vector<std::pair<std::string, std::string>>
    SnapshotScan(leveldb::DB* db, const std::string& prefix) {
        std::vector<std::pair<std::string, std::string>> results;

        ScopedSnapshot snapshot(db);

        leveldb::ReadOptions options;
        options.snapshot = snapshot.Get();

        std::unique_ptr<leveldb::Iterator> it(db->NewIterator(options));

        for (it->Seek(prefix); it->Valid(); it->Next()) {
            std::string key = it->key().ToString();
            if (key.compare(0, prefix.length(), prefix) != 0) {
                break;
            }
            results.emplace_back(key, it->value().ToString());
        }

        return results;
    }

    // Compare before and after state
    static void CompareStates(leveldb::DB* db) {
        // Take snapshot of current state
        const leveldb::Snapshot* before = db->GetSnapshot();

        // Make modifications
        db->Put(leveldb::WriteOptions(), "counter", "1");
        db->Put(leveldb::WriteOptions(), "timestamp", "2024-01-01");

        // Take snapshot of new state
        const leveldb::Snapshot* after = db->GetSnapshot();

        // Compare values
        leveldb::ReadOptions before_opts, after_opts;
        before_opts.snapshot = before;
        after_opts.snapshot = after;

        std::string before_value, after_value;
        db->Get(before_opts, "counter", &before_value);
        db->Get(after_opts, "counter", &after_value);

        std::cout << "Before: " << (before_value.empty() ? "empty" : before_value) << std::endl;
        std::cout << "After: " << after_value << std::endl;

        // Cleanup
        db->ReleaseSnapshot(before);
        db->ReleaseSnapshot(after);
    }
};

// Usage
void SnapshotDemo(leveldb::DB* db) {
    // Use RAII snapshot
    {
        SnapshotExample::ScopedSnapshot snapshot(db);

        leveldb::ReadOptions options;
        options.snapshot = snapshot.Get();

        // Consistent reads
        std::string value;
        db->Get(options, "mykey", &value);

    }  // Snapshot automatically released

    // Scan with snapshot consistency
    auto results = SnapshotExample::SnapshotScan(db, "user:");
}
```

## 6. Language Bindings

### Python (plyvel)

```bash
# Installation
pip install plyvel
```

```python
import plyvel
from typing import Optional, List, Tuple
import json

class LevelDBStore:
    def __init__(self, db_path: str):
        self.db = plyvel.DB(
            db_path,
            create_if_missing=True,
            write_buffer_size=16 * 1024 * 1024,  # 16MB
            max_open_files=5000,
            block_cache_size=256 * 1024 * 1024,  # 256MB
            block_size=16 * 1024,  # 16KB
            compression='snappy'
        )

    def put(self, key: bytes, value: bytes, sync: bool = False) -> None:
        """Write a key-value pair."""
        self.db.put(key, value, sync=sync)

    def get(self, key: bytes) -> Optional[bytes]:
        """Read a value by key."""
        return self.db.get(key)

    def delete(self, key: bytes, sync: bool = False) -> None:
        """Delete a key."""
        self.db.delete(key, sync=sync)

    def batch_write(self, operations: List[Tuple[str, bytes, Optional[bytes]]]) -> None:
        """Atomic batch write.

        Args:
            operations: List of (op, key, value) tuples
                       op can be 'put' or 'delete'
        """
        with self.db.write_batch(sync=False) as batch:
            for op, key, *args in operations:
                if op == 'put':
                    batch.put(key, args[0])
                elif op == 'delete':
                    batch.delete(key)

    def scan_prefix(self, prefix: bytes) -> List[Tuple[bytes, bytes]]:
        """Scan all keys with given prefix."""
        results = []
        for key, value in self.db.iterator(start=prefix, include_value=True):
            if not key.startswith(prefix):
                break
            results.append((key, value))
        return results

    def count_prefix(self, prefix: bytes) -> int:
        """Count keys with given prefix."""
        count = 0
        for key in self.db.iterator(start=prefix, include_value=False):
            if not key.startswith(prefix):
                break
            count += 1
        return count

    def snapshot(self):
        """Get a snapshot for consistent reads."""
        return self.db.snapshot()

    def close(self):
        """Close database."""
        self.db.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# Usage examples
def python_examples():
    with LevelDBStore('/tmp/pyleveldb') as store:
        # Basic operations
        store.put(b'user:1:name', b'Alice')
        store.put(b'user:1:email', b'alice@example.com')
        store.put(b'user:2:name', b'Bob')

        # Read
        name = store.get(b'user:1:name')
        print(f"Name: {name.decode()}")

        # JSON storage
        user_data = {'name': 'Charlie', 'age': 30, 'city': 'NYC'}
        store.put(b'user:3:data', json.dumps(user_data).encode())

        # Batch write
        store.batch_write([
            ('put', b'user:4:name', b'David'),
            ('put', b'user:4:email', b'david@example.com'),
            ('delete', b'user:1:temp'),
        ])

        # Scan users
        users = store.scan_prefix(b'user:')
        for key, value in users:
            print(f"{key.decode()}: {value.decode()}")

        # Count users
        user_count = store.count_prefix(b'user:')
        print(f"Total users: {user_count}")

        # Snapshot
        with store.snapshot() as snapshot:
            # Consistent reads
            value1 = snapshot.get(b'user:1:name')
            value2 = snapshot.get(b'user:2:name')


# Context manager for snapshots
class SnapshotContext:
    def __init__(self, db: plyvel.DB):
        self.db = db
        self.snapshot = None

    def __enter__(self):
        self.snapshot = self.db.snapshot()
        return self.snapshot

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.snapshot.close()


# Advanced iteration
def iterate_range(db: plyvel.DB, start_key: bytes, end_key: bytes):
    """Iterate between two keys."""
    for key, value in db.iterator(start=start_key, include_value=True):
        if key > end_key:
            break
        print(f"{key.decode()}: {value.decode()}")


# Reverse iteration
def reverse_scan(db: plyvel.DB, start_key: bytes, limit: int = 100):
    """Scan in reverse order."""
    count = 0
    for key, value in db.iterator(start=start_key, reverse=True, include_value=True):
        print(f"{key.decode()}: {value.decode()}")
        count += 1
        if count >= limit:
            break
```

### Go (goleveldb)

```bash
# Installation
go get github.com/syndtr/goleveldb/leveldb
```

```go
package main

import (
    "fmt"
    "log"
    "github.com/syndtr/goleveldb/leveldb"
    "github.com/syndtr/goleveldb/leveldb/opt"
    "github.com/syndtr/goleveldb/leveldb/util"
)

type LevelDBStore struct {
    db *leveldb.DB
}

func NewLevelDBStore(path string) (*LevelDBStore, error) {
    opts := &opt.Options{
        WriteBuffer:        16 * 1024 * 1024, // 16MB
        BlockCacheCapacity: 256 * 1024 * 1024, // 256MB
        BlockSize:          16 * 1024, // 16KB
        Compression:        opt.SnappyCompression,
        OpenFilesCacheCapacity: 5000,
    }

    db, err := leveldb.OpenFile(path, opts)
    if err != nil {
        return nil, err
    }

    return &LevelDBStore{db: db}, nil
}

func (s *LevelDBStore) Put(key, value []byte) error {
    return s.db.Put(key, value, nil)
}

func (s *LevelDBStore) Get(key []byte) ([]byte, error) {
    return s.db.Get(key, nil)
}

func (s *LevelDBStore) Delete(key []byte) error {
    return s.db.Delete(key, nil)
}

func (s *LevelDBStore) Exists(key []byte) (bool, error) {
    return s.db.Has(key, nil)
}

func (s *LevelDBStore) BatchWrite(operations []struct {
    Op    string // "put" or "delete"
    Key   []byte
    Value []byte
}) error {
    batch := new(leveldb.Batch)

    for _, op := range operations {
        switch op.Op {
        case "put":
            batch.Put(op.Key, op.Value)
        case "delete":
            batch.Delete(op.Key)
        }
    }

    return s.db.Write(batch, nil)
}

func (s *LevelDBStore) ScanPrefix(prefix []byte) ([][2][]byte, error) {
    var results [][2][]byte

    iter := s.db.NewIterator(util.BytesPrefix(prefix), nil)
    defer iter.Release()

    for iter.Next() {
        key := make([]byte, len(iter.Key()))
        value := make([]byte, len(iter.Value()))
        copy(key, iter.Key())
        copy(value, iter.Value())
        results = append(results, [2][]byte{key, value})
    }

    return results, iter.Error()
}

func (s *LevelDBStore) ScanRange(start, limit []byte) ([][2][]byte, error) {
    var results [][2][]byte

    iter := s.db.NewIterator(&util.Range{Start: start, Limit: limit}, nil)
    defer iter.Release()

    for iter.Next() {
        key := make([]byte, len(iter.Key()))
        value := make([]byte, len(iter.Value()))
        copy(key, iter.Key())
        copy(value, iter.Value())
        results = append(results, [2][]byte{key, value})
    }

    return results, iter.Error()
}

func (s *LevelDBStore) CountPrefix(prefix []byte) (int, error) {
    count := 0

    iter := s.db.NewIterator(util.BytesPrefix(prefix), nil)
    defer iter.Release()

    for iter.Next() {
        count++
    }

    return count, iter.Error()
}

func (s *LevelDBStore) Close() error {
    return s.db.Close()
}

// Usage
func main() {
    store, err := NewLevelDBStore("/tmp/goleveldb")
    if err != nil {
        log.Fatal(err)
    }
    defer store.Close()

    // Write
    err = store.Put([]byte("user:1:name"), []byte("Alice"))
    if err != nil {
        log.Fatal(err)
    }

    // Read
    value, err := store.Get([]byte("user:1:name"))
    if err != nil {
        log.Fatal(err)
    }
    fmt.Printf("Name: %s\n", value)

    // Batch write
    ops := []struct {
        Op    string
        Key   []byte
        Value []byte
    }{
        {"put", []byte("user:2:name"), []byte("Bob")},
        {"put", []byte("user:2:email"), []byte("bob@example.com")},
        {"delete", []byte("user:temp"), nil},
    }
    err = store.BatchWrite(ops)
    if err != nil {
        log.Fatal(err)
    }

    // Scan
    results, err := store.ScanPrefix([]byte("user:"))
    if err != nil {
        log.Fatal(err)
    }

    for _, kv := range results {
        fmt.Printf("%s: %s\n", kv[0], kv[1])
    }

    // Count
    count, err := store.CountPrefix([]byte("user:"))
    if err != nil {
        log.Fatal(err)
    }
    fmt.Printf("Total users: %d\n", count)
}
```

## 7. Custom Comparators

LevelDB allows custom key ordering through comparators.

```cpp
#include <leveldb/comparator.h>

// Integer key comparator (instead of lexicographic)
class IntegerComparator : public leveldb::Comparator {
public:
    const char* Name() const override {
        return "IntegerComparator";
    }

    int Compare(const leveldb::Slice& a, const leveldb::Slice& b) const override {
        // Parse integers from slices
        int num_a = std::stoi(a.ToString());
        int num_b = std::stoi(b.ToString());

        if (num_a < num_b) return -1;
        if (num_a > num_b) return 1;
        return 0;
    }

    void FindShortestSeparator(std::string* start,
                               const leveldb::Slice& limit) const override {
        // Optional: optimize key range
    }

    void FindShortSuccessor(std::string* key) const override {
        // Optional: optimize key range
    }
};

// Reverse comparator (descending order)
class ReverseComparator : public leveldb::Comparator {
public:
    const char* Name() const override {
        return "ReverseComparator";
    }

    int Compare(const leveldb::Slice& a, const leveldb::Slice& b) const override {
        // Reverse of default bytewise comparison
        return -a.compare(b);
    }

    void FindShortestSeparator(std::string* start,
                               const leveldb::Slice& limit) const override {}

    void FindShortSuccessor(std::string* key) const override {}
};

// Usage
void CustomComparatorExample() {
    IntegerComparator cmp;

    leveldb::Options options;
    options.create_if_missing = true;
    options.comparator = &cmp;

    leveldb::DB* db;
    leveldb::Status status = leveldb::DB::Open(options, "/tmp/intdb", &db);

    if (status.ok()) {
        // Keys stored in integer order
        db->Put(leveldb::WriteOptions(), "10", "ten");
        db->Put(leveldb::WriteOptions(), "2", "two");
        db->Put(leveldb::WriteOptions(), "1", "one");

        // Iteration returns: 1, 2, 10 (not 1, 10, 2)
        leveldb::Iterator* it = db->NewIterator(leveldb::ReadOptions());
        for (it->SeekToFirst(); it->Valid(); it->Next()) {
            std::cout << it->key().ToString() << std::endl;
        }
        delete it;

        delete db;
    }
}
```

## 8. Performance Optimization

### Write Performance

```cpp
leveldb::Options GetWriteOptimizedOptions() {
    leveldb::Options options;
    options.create_if_missing = true;

    // Larger write buffer = fewer L0 files = less compaction
    options.write_buffer_size = 64 * 1024 * 1024;  // 64MB (default 4MB)

    // Larger block size for sequential writes
    options.block_size = 64 * 1024;  // 64KB (default 4KB)

    // Disable compression for faster writes (if space not a concern)
    options.compression = leveldb::kNoCompression;

    // Larger file size = fewer files
    options.max_file_size = 64 * 1024 * 1024;  // 64MB (default 2MB)

    return options;
}

// Batch writes for maximum throughput
void BulkWrite(leveldb::DB* db, size_t num_records) {
    const size_t batch_size = 10000;
    leveldb::WriteBatch batch;
    leveldb::WriteOptions options;
    options.sync = false;  // Async writes

    for (size_t i = 0; i < num_records; ++i) {
        std::string key = "key_" + std::to_string(i);
        std::string value = "value_" + std::to_string(i);

        batch.Put(key, value);

        // Write batch every 10000 records
        if ((i + 1) % batch_size == 0) {
            db->Write(options, &batch);
            batch.Clear();
        }
    }

    // Write remaining
    if (batch.Count() > 0) {
        db->Write(options, &batch);
    }
}
```

### Read Performance

```cpp
leveldb::Options GetReadOptimizedOptions() {
    leveldb::Options options;
    options.create_if_missing = true;

    // Large block cache for hot data
    options.block_cache = leveldb::NewLRUCache(1024 * 1024 * 1024);  // 1GB

    // Smaller blocks for point lookups
    options.block_size = 4 * 1024;  // 4KB

    // Keep more files open
    options.max_open_files = 10000;

    // Enable compression to reduce I/O
    options.compression = leveldb::kSnappyCompression;

    return options;
}

// Efficient range scan
void OptimizedScan(leveldb::DB* db, const std::string& prefix) {
    leveldb::ReadOptions options;
    options.verify_checksums = false;  // Skip checksum verification
    options.fill_cache = true;          // Cache blocks

    leveldb::Iterator* it = db->NewIterator(options);

    for (it->Seek(prefix); it->Valid(); it->Next()) {
        if (!it->key().starts_with(prefix)) {
            break;
        }

        // Process key-value
        const std::string& key = it->key().ToString();
        const std::string& value = it->value().ToString();
    }

    delete it;
}
```

### Compaction Control

```cpp
// Manual compaction to optimize read performance
void ManualCompaction(leveldb::DB* db) {
    // Compact entire key range
    db->CompactRange(nullptr, nullptr);

    // Compact specific range
    leveldb::Slice start("user:0");
    leveldb::Slice end("user:z");
    db->CompactRange(&start, &end);
}

// Check approximate sizes
void CheckSizes(leveldb::DB* db) {
    leveldb::Range ranges[2];
    ranges[0] = leveldb::Range("user:0", "user:5");
    ranges[1] = leveldb::Range("user:5", "user:z");

    uint64_t sizes[2];
    db->GetApproximateSizes(ranges, 2, sizes);

    std::cout << "Range 1 size: " << sizes[0] << " bytes" << std::endl;
    std::cout << "Range 2 size: " << sizes[1] << " bytes" << std::endl;
}
```

## 9. Backup and Recovery

### Manual Backup

```cpp
#include <filesystem>
#include <fstream>

class BackupManager {
public:
    // Create backup by copying database files
    static bool CreateBackup(const std::string& db_path,
                            const std::string& backup_path) {
        namespace fs = std::filesystem;

        try {
            // Create backup directory
            fs::create_directories(backup_path);

            // Copy all files
            for (const auto& entry : fs::directory_iterator(db_path)) {
                fs::path dest = fs::path(backup_path) / entry.path().filename();
                fs::copy_file(entry.path(), dest,
                            fs::copy_options::overwrite_existing);
            }

            return true;
        } catch (const std::exception& e) {
            std::cerr << "Backup failed: " << e.what() << std::endl;
            return false;
        }
    }

    // Restore from backup
    static bool RestoreBackup(const std::string& backup_path,
                             const std::string& db_path) {
        namespace fs = std::filesystem;

        try {
            // Remove existing database
            if (fs::exists(db_path)) {
                fs::remove_all(db_path);
            }

            // Create target directory
            fs::create_directories(db_path);

            // Copy backup files
            for (const auto& entry : fs::directory_iterator(backup_path)) {
                fs::path dest = fs::path(db_path) / entry.path().filename();
                fs::copy_file(entry.path(), dest);
            }

            return true;
        } catch (const std::exception& e) {
            std::cerr << "Restore failed: " << e.what() << std::endl;
            return false;
        }
    }
};

// Usage
void BackupExample() {
    // Close database before backup
    leveldb::DB* db = nullptr;
    // ... use database ..
    delete db;  // Close

    // Create backup
    BackupManager::CreateBackup("/tmp/mydb", "/backups/mydb_20240101");

    // Reopen database
    leveldb::DB::Open(leveldb::Options(), "/tmp/mydb", &db);
}
```

### Snapshot Export

```cpp
class SnapshotExport {
public:
    // Export snapshot to file
    static bool ExportToFile(leveldb::DB* db,
                            const std::string& filename) {
        std::ofstream out(filename, std::ios::binary);
        if (!out) {
            return false;
        }

        const leveldb::Snapshot* snapshot = db->GetSnapshot();
        leveldb::ReadOptions options;
        options.snapshot = snapshot;

        leveldb::Iterator* it = db->NewIterator(options);

        for (it->SeekToFirst(); it->Valid(); it->Next()) {
            // Write key length, key, value length, value
            uint32_t key_len = it->key().size();
            uint32_t val_len = it->value().size();

            out.write(reinterpret_cast<const char*>(&key_len), sizeof(key_len));
            out.write(it->key().data(), key_len);
            out.write(reinterpret_cast<const char*>(&val_len), sizeof(val_len));
            out.write(it->value().data(), val_len);
        }

        delete it;
        db->ReleaseSnapshot(snapshot);
        out.close();

        return true;
    }

    // Import from file
    static bool ImportFromFile(leveldb::DB* db,
                              const std::string& filename) {
        std::ifstream in(filename, std::ios::binary);
        if (!in) {
            return false;
        }

        leveldb::WriteBatch batch;
        const size_t batch_size = 10000;
        size_t count = 0;

        while (in.peek() != EOF) {
            uint32_t key_len, val_len;

            in.read(reinterpret_cast<char*>(&key_len), sizeof(key_len));
            std::string key(key_len, '\0');
            in.read(&key[0], key_len);

            in.read(reinterpret_cast<char*>(&val_len), sizeof(val_len));
            std::string value(val_len, '\0');
            in.read(&value[0], val_len);

            batch.Put(key, value);
            count++;

            if (count % batch_size == 0) {
                db->Write(leveldb::WriteOptions(), &batch);
                batch.Clear();
            }
        }

        if (batch.Count() > 0) {
            db->Write(leveldb::WriteOptions(), &batch);
        }

        in.close();
        return true;
    }
};
```

## 10. Monitoring and Statistics

### Property Queries

```cpp
#include <leveldb/db.h>

class LevelDBMonitor {
public:
    static void PrintStatistics(leveldb::DB* db) {
        std::string stats;

        // Get database statistics
        if (db->GetProperty("leveldb.stats", &stats)) {
            std::cout << "LevelDB Statistics:\n" << stats << std::endl;
        }

        // Number of files at each level
        for (int level = 0; level < 7; ++level) {
            std::string num_files_prop = "leveldb.num-files-at-level" + std::to_string(level);
            std::string num_files;

            if (db->GetProperty(num_files_prop, &num_files)) {
                std::cout << "Level " << level << ": " << num_files << " files" << std::endl;
            }
        }

        // Approximate memory usage
        std::string mem_usage;
        if (db->GetProperty("leveldb.approximate-memory-usage", &mem_usage)) {
            std::cout << "Memory usage: " << mem_usage << " bytes" << std::endl;
        }

        // SSTable info
        std::string sstables;
        if (db->GetProperty("leveldb.sstables", &sstables)) {
            std::cout << "SSTable info:\n" << sstables << std::endl;
        }
    }

    static void MonitorCompaction(leveldb::DB* db) {
        // Check if compaction is needed
        std::string num_l0_files;
        db->GetProperty("leveldb.num-files-at-level0", &num_l0_files);

        int l0_count = std::stoi(num_l0_files);

        if (l0_count > 4) {
            std::cout << "Warning: " << l0_count << " L0 files, compaction may be needed" << std::endl;
        }
    }

    static size_t EstimateKeyCount(leveldb::DB* db) {
        size_t count = 0;

        leveldb::ReadOptions options;
        options.fill_cache = false;

        leveldb::Iterator* it = db->NewIterator(options);
        for (it->SeekToFirst(); it->Valid(); it->Next()) {
            count++;
        }

        delete it;
        return count;
    }

    static void PrintDatabaseSize(leveldb::DB* db) {
        leveldb::Range range("", "~~~");  // All keys
        uint64_t size;

        db->GetApproximateSizes(&range, 1, &size);

        std::cout << "Approximate database size: "
                  << size / (1024 * 1024) << " MB" << std::endl;
    }
};

// Usage
void MonitoringExample(leveldb::DB* db) {
    LevelDBMonitor::PrintStatistics(db);
    LevelDBMonitor::MonitorCompaction(db);

    size_t keys = LevelDBMonitor::EstimateKeyCount(db);
    std::cout << "Estimated key count: " << keys << std::endl;

    LevelDBMonitor::PrintDatabaseSize(db);
}
```

## 11. Security Best Practices

### Input Validation

```cpp
class SecureKeyValueStore {
private:
    leveldb::DB* db_;
    static constexpr size_t MAX_KEY_SIZE = 1024;
    static constexpr size_t MAX_VALUE_SIZE = 10 * 1024 * 1024;  // 10MB

    bool ValidateKey(const std::string& key) const {
        if (key.empty() || key.size() > MAX_KEY_SIZE) {
            return false;
        }

        // Check for null bytes (can cause issues)
        if (key.find('\0') != std::string::npos) {
            return false;
        }

        return true;
    }

    bool ValidateValue(const std::string& value) const {
        return value.size() <= MAX_VALUE_SIZE;
    }

public:
    SecureKeyValueStore(leveldb::DB* db) : db_(db) {}

    leveldb::Status SecurePut(const std::string& key,
                             const std::string& value) {
        if (!ValidateKey(key)) {
            return leveldb::Status::InvalidArgument("Invalid key");
        }

        if (!ValidateValue(value)) {
            return leveldb::Status::InvalidArgument("Value too large");
        }

        leveldb::WriteOptions options;
        options.sync = false;

        return db_->Put(options, key, value);
    }

    leveldb::Status SecureGet(const std::string& key,
                             std::string* value) {
        if (!ValidateKey(key)) {
            return leveldb::Status::InvalidArgument("Invalid key");
        }

        leveldb::ReadOptions options;
        options.verify_checksums = true;  // Verify data integrity

        return db_->Get(options, key, value);
    }

    leveldb::Status SecureDelete(const std::string& key) {
        if (!ValidateKey(key)) {
            return leveldb::Status::InvalidArgument("Invalid key");
        }

        return db_->Delete(leveldb::WriteOptions(), key);
    }

    // Limit iteration to prevent DoS
    std::vector<std::pair<std::string, std::string>>
    SecureScan(const std::string& prefix, size_t max_results = 1000) {
        std::vector<std::pair<std::string, std::string>> results;

        if (!ValidateKey(prefix)) {
            return results;
        }

        leveldb::ReadOptions options;
        leveldb::Iterator* it = db_->NewIterator(options);

        size_t count = 0;
        for (it->Seek(prefix); it->Valid() && count < max_results; it->Next()) {
            if (!it->key().starts_with(prefix)) {
                break;
            }

            results.emplace_back(it->key().ToString(), it->value().ToString());
            count++;
        }

        delete it;
        return results;
    }
};
```

### File Permissions

```cpp
#include <sys/stat.h>

void SetSecurePermissions(const std::string& db_path) {
    // Set directory permissions to 0700 (rwx------)
    chmod(db_path.c_str(), S_IRWXU);

    // Set file permissions for all database files
    namespace fs = std::filesystem;
    for (const auto& entry : fs::directory_iterator(db_path)) {
        chmod(entry.path().c_str(), S_IRUSR | S_IWUSR);
    }
}
```

### Encryption at Rest

LevelDB doesn't have built-in encryption. For encryption at rest, use:

1. **Filesystem-level encryption**: LUKS, dm-crypt, or encrypted volumes
2. **Application-level encryption**: Encrypt values before storing

```cpp
#include <openssl/evp.h>
#include <openssl/rand.h>

class EncryptedStore {
private:
    leveldb::DB* db_;
    unsigned char key_[32];  // 256-bit key

    std::string Encrypt(const std::string& plaintext) {
        // Implementation using OpenSSL AES-256-GCM
        // ... encryption code ..
        return "encrypted_data";
    }

    std::string Decrypt(const std::string& ciphertext) {
        // Implementation using OpenSSL AES-256-GCM
        // ... decryption code ..
        return "decrypted_data";
    }

public:
    EncryptedStore(leveldb::DB* db, const unsigned char* key) : db_(db) {
        memcpy(key_, key, 32);
    }

    leveldb::Status PutEncrypted(const std::string& key,
                                const std::string& value) {
        std::string encrypted = Encrypt(value);
        return db_->Put(leveldb::WriteOptions(), key, encrypted);
    }

    leveldb::Status GetDecrypted(const std::string& key,
                                std::string* value) {
        std::string encrypted;
        leveldb::Status s = db_->Get(leveldb::ReadOptions(), key, &encrypted);

        if (s.ok()) {
            *value = Decrypt(encrypted);
        }

        return s;
    }
};
```

## 12. Testing Strategies

### Unit Testing

```cpp
#include <gtest/gtest.h>
#include <leveldb/db.h>
#include <filesystem>

class LevelDBTest : public ::testing::Test {
protected:
    std::string db_path_;
    leveldb::DB* db_;
    leveldb::Options options_;

    void SetUp() override {
        db_path_ = "/tmp/test_leveldb_" + std::to_string(getpid());

        options_.create_if_missing = true;
        options_.error_if_exists = true;

        leveldb::Status status = leveldb::DB::Open(options_, db_path_, &db_);
        ASSERT_TRUE(status.ok()) << status.ToString();
    }

    void TearDown() override {
        delete db_;
        db_ = nullptr;

        // Clean up test database
        std::filesystem::remove_all(db_path_);
    }
};

TEST_F(LevelDBTest, BasicPutGet) {
    leveldb::WriteOptions write_opts;
    leveldb::ReadOptions read_opts;

    // Write
    leveldb::Status s = db_->Put(write_opts, "key1", "value1");
    ASSERT_TRUE(s.ok());

    // Read
    std::string value;
    s = db_->Get(read_opts, "key1", &value);
    ASSERT_TRUE(s.ok());
    EXPECT_EQ(value, "value1");

    // Not found
    s = db_->Get(read_opts, "key2", &value);
    EXPECT_TRUE(s.IsNotFound());
}

TEST_F(LevelDBTest, Delete) {
    db_->Put(leveldb::WriteOptions(), "key1", "value1");

    // Delete
    leveldb::Status s = db_->Delete(leveldb::WriteOptions(), "key1");
    ASSERT_TRUE(s.ok());

    // Verify deleted
    std::string value;
    s = db_->Get(leveldb::ReadOptions(), "key1", &value);
    EXPECT_TRUE(s.IsNotFound());
}

TEST_F(LevelDBTest, BatchWrite) {
    leveldb::WriteBatch batch;
    batch.Put("key1", "value1");
    batch.Put("key2", "value2");
    batch.Delete("key3");

    leveldb::Status s = db_->Write(leveldb::WriteOptions(), &batch);
    ASSERT_TRUE(s.ok());

    std::string value;
    ASSERT_TRUE(db_->Get(leveldb::ReadOptions(), "key1", &value).ok());
    EXPECT_EQ(value, "value1");
}

TEST_F(LevelDBTest, Iterator) {
    // Insert test data
    db_->Put(leveldb::WriteOptions(), "a", "1");
    db_->Put(leveldb::WriteOptions(), "b", "2");
    db_->Put(leveldb::WriteOptions(), "c", "3");

    // Forward iteration
    std::vector<std::string> keys;
    leveldb::Iterator* it = db_->NewIterator(leveldb::ReadOptions());

    for (it->SeekToFirst(); it->Valid(); it->Next()) {
        keys.push_back(it->key().ToString());
    }

    ASSERT_TRUE(it->status().ok());
    delete it;

    EXPECT_EQ(keys, std::vector<std::string>({"a", "b", "c"}));
}

TEST_F(LevelDBTest, Snapshot) {
    // Initial value
    db_->Put(leveldb::WriteOptions(), "key", "value1");

    // Take snapshot
    const leveldb::Snapshot* snapshot = db_->GetSnapshot();

    // Modify after snapshot
    db_->Put(leveldb::WriteOptions(), "key", "value2");

    // Read with snapshot (should see old value)
    leveldb::ReadOptions options;
    options.snapshot = snapshot;

    std::string value;
    db_->Get(options, "key", &value);
    EXPECT_EQ(value, "value1");

    // Read without snapshot (should see new value)
    db_->Get(leveldb::ReadOptions(), "key", &value);
    EXPECT_EQ(value, "value2");

    db_->ReleaseSnapshot(snapshot);
}
```

### Benchmarking

```cpp
#include <benchmark/benchmark.h>

static void BM_Put(benchmark::State& state) {
    leveldb::DB* db;
    leveldb::Options options;
    options.create_if_missing = true;
    leveldb::DB::Open(options, "/tmp/bench_db", &db);

    leveldb::WriteOptions write_opts;
    write_opts.sync = false;

    for (auto _ : state) {
        std::string key = "key_" + std::to_string(state.iterations());
        std::string value = "value_" + std::to_string(state.iterations());
        db->Put(write_opts, key, value);
    }

    state.SetItemsProcessed(state.iterations());

    delete db;
    std::filesystem::remove_all("/tmp/bench_db");
}

static void BM_Get(benchmark::State& state) {
    leveldb::DB* db;
    leveldb::Options options;
    options.create_if_missing = true;
    leveldb::DB::Open(options, "/tmp/bench_db", &db);

    // Prepopulate
    for (int i = 0; i < 10000; ++i) {
        db->Put(leveldb::WriteOptions(),
               "key_" + std::to_string(i),
               "value_" + std::to_string(i));
    }

    leveldb::ReadOptions read_opts;
    std::string value;
    int key_num = 0;

    for (auto _ : state) {
        std::string key = "key_" + std::to_string(key_num++ % 10000);
        db->Get(read_opts, key, &value);
    }

    state.SetItemsProcessed(state.iterations());

    delete db;
    std::filesystem::remove_all("/tmp/bench_db");
}

static void BM_Scan(benchmark::State& state) {
    leveldb::DB* db;
    leveldb::Options options;
    options.create_if_missing = true;
    leveldb::DB::Open(options, "/tmp/bench_db", &db);

    // Prepopulate
    for (int i = 0; i < 10000; ++i) {
        db->Put(leveldb::WriteOptions(),
               "key_" + std::to_string(i),
               "value_" + std::to_string(i));
    }

    for (auto _ : state) {
        leveldb::Iterator* it = db->NewIterator(leveldb::ReadOptions());
        int count = 0;

        for (it->SeekToFirst(); it->Valid() && count < 1000; it->Next()) {
            count++;
        }

        delete it;
    }

    state.SetItemsProcessed(state.iterations() * 1000);

    delete db;
    std::filesystem::remove_all("/tmp/bench_db");
}

BENCHMARK(BM_Put);
BENCHMARK(BM_Get);
BENCHMARK(BM_Scan);

BENCHMARK_MAIN();
```

## 13. Production Deployment

### Docker Deployment

```dockerfile
# Dockerfile
FROM ubuntu:22.04 AS builder

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    libsnappy-dev \
    && rm -rf /var/lib/apt/lists/*

# Build LevelDB
WORKDIR /tmp
RUN git clone --recurse-submodules https://github.com/google/leveldb.git && \
    cd leveldb && \
    mkdir build && cd build && \
    cmake -DCMAKE_BUILD_TYPE=Release \
          -DCMAKE_INSTALL_PREFIX=/usr/local .. && \
    make -j$(nproc) && \
    make install

# Build application
COPY . /app
WORKDIR /app
RUN mkdir build && cd build && \
    cmake -DCMAKE_BUILD_TYPE=Release ... && \
    make -j$(nproc)

# Runtime image
FROM ubuntu:22.04

RUN apt-get update && apt-get install -y \
    libsnappy1v5 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/lib/libleveldb.* /usr/local/lib/
COPY --from=builder /app/build/myapp /usr/local/bin/

RUN ldconfig

# Data volume
VOLUME ["/data"]

# Run as non-root
RUN useradd -m -u 1000 leveldb
USER leveldb

CMD ["/usr/local/bin/myapp", "--db_path=/data/leveldb"]
```

### Production Configuration

```cpp
leveldb::Options GetProductionOptions() {
    leveldb::Options options;
    options.create_if_missing = true;

    // Write buffer (adjust based on write workload)
    options.write_buffer_size = 32 * 1024 * 1024;  // 32MB

    // Block cache (adjust based on available RAM and working set)
    options.block_cache = leveldb::NewLRUCache(512 * 1024 * 1024);  // 512MB

    // Block size (4KB for random reads, 16-64KB for sequential)
    options.block_size = 16 * 1024;  // 16KB

    // File descriptors
    options.max_open_files = 5000;

    // File size
    options.max_file_size = 32 * 1024 * 1024;  // 32MB

    // Compression (Snappy is good balance of speed/ratio)
    options.compression = leveldb::kSnappyCompression;

    // Paranoid checks (enable for critical data)
    options.paranoid_checks = true;

    return options;
}
```

### System Limits

```bash
# Increase file descriptor limit
# /etc/security/limits.conf
* soft nofile 65536
* hard nofile 65536

# Kernel parameters
# /etc/sysctl.conf
fs.file-max = 2097152
vm.swappiness = 10
vm.dirty_ratio = 60
vm.dirty_background_ratio = 5
```

## 14. Common Patterns and Anti-Patterns

### Pattern: Secondary Indexing

```cpp
class IndexedStore {
private:
    leveldb::DB* db_;

public:
    // Store user with email index
    // Primary: user:<id> -> JSON
    // Index: email:<email> -> <id>
    leveldb::Status AddUser(const std::string& id,
                           const std::string& email,
                           const std::string& data) {
        leveldb::WriteBatch batch;

        // Primary record
        batch.Put("user:" + id, data);

        // Email index
        batch.Put("email:" + email, id);

        return db_->Write(leveldb::WriteOptions(), &batch);
    }

    leveldb::Status GetUserByEmail(const std::string& email,
                                   std::string* data) {
        // Lookup ID from index
        std::string id;
        leveldb::Status s = db_->Get(leveldb::ReadOptions(),
                                     "email:" + email, &id);
        if (!s.ok()) {
            return s;
        }

        // Get user data
        return db_->Get(leveldb::ReadOptions(), "user:" + id, data);
    }

    leveldb::Status DeleteUser(const std::string& id) {
        // Read user to get email
        std::string data;
        leveldb::Status s = db_->Get(leveldb::ReadOptions(),
                                     "user:" + id, &data);
        if (!s.ok()) {
            return s;
        }

        // Parse email from data (simplified)
        std::string email = "extracted_email";

        // Delete both primary and index
        leveldb::WriteBatch batch;
        batch.Delete("user:" + id);
        batch.Delete("email:" + email);

        return db_->Write(leveldb::WriteOptions(), &batch);
    }
};
```

### Pattern: Time-Series Data

```cpp
class TimeSeriesStore {
private:
    leveldb::DB* db_;

    std::string MakeKey(uint64_t timestamp, const std::string& metric_id) {
        // Big-endian timestamp for proper sorting
        char ts_bytes[8];
        for (int i = 0; i < 8; ++i) {
            ts_bytes[7 - i] = (timestamp >> (i * 8)) & 0xFF;
        }

        return std::string(ts_bytes, 8) + ":" + metric_id;
    }

public:
    TimeSeriesStore(leveldb::DB* db) : db_(db) {}

    void Record(const std::string& metric_id,
               uint64_t timestamp,
               double value) {
        std::string key = MakeKey(timestamp, metric_id);
        std::string value_str = std::to_string(value);

        db_->Put(leveldb::WriteOptions(), key, value_str);
    }

    std::vector<double> Query(const std::string& metric_id,
                             uint64_t start_ts,
                             uint64_t end_ts) {
        std::vector<double> results;

        std::string start_key = MakeKey(start_ts, metric_id);
        std::string end_key = MakeKey(end_ts, metric_id);

        leveldb::Iterator* it = db_->NewIterator(leveldb::ReadOptions());

        for (it->Seek(start_key); it->Valid(); it->Next()) {
            if (it->key().ToString() > end_key) {
                break;
            }

            results.push_back(std::stod(it->value().ToString()));
        }

        delete it;
        return results;
    }
};
```

### Anti-Pattern: Large Values

```cpp
// ❌ BAD: Storing large blobs directly
void BadPattern(leveldb::DB* db) {
    std::string large_image(10 * 1024 * 1024, 'x');  // 10MB
    db->Put(leveldb::WriteOptions(), "image:1", large_image);

    // Problems:
    // - Bloats write buffer
    // - Slows down compaction
    // - Wastes block cache space
}

// ✅ GOOD: Store large blobs externally
void GoodPattern(leveldb::DB* db) {
    // Store image in filesystem or object storage
    std::string file_path = "/storage/images/image_1.jpg";
    // ... write to file ..

    // Store only metadata in LevelDB
    std::string metadata = R"({"path":")" + file_path + R"(","size":10485760})";
    db->Put(leveldb::WriteOptions(), "image:1:meta", metadata);
}
```

### Anti-Pattern: Hot Key

```cpp
// ❌ BAD: Single counter (write bottleneck)
void BadCounterPattern(leveldb::DB* db) {
    std::string value;
    db->Get(leveldb::ReadOptions(), "counter", &value);

    int count = value.empty() ? 0 : std::stoi(value);
    count++;

    db->Put(leveldb::WriteOptions(), "counter", std::to_string(count));

    // Problem: Only one write at a time due to LevelDB's single-writer model
}

// ✅ GOOD: Sharded counters
void GoodCounterPattern(leveldb::DB* db, int thread_id, int num_threads) {
    std::string key = "counter:shard:" + std::to_string(thread_id % num_threads);

    std::string value;
    db->Get(leveldb::ReadOptions(), key, &value);

    int count = value.empty() ? 0 : std::stoi(value);
    count++;

    db->Put(leveldb::WriteOptions(), key, std::to_string(count));

    // To get total: sum all shards
}
```

## 15. Migration from LevelDB to RocksDB

If you need features like column families, transactions, or better performance, migrate to RocksDB.

```cpp
#include <rocksdb/db.h>
#include <leveldb/db.h>

class Migration {
public:
    // Copy data from LevelDB to RocksDB
    static bool MigrateLevelDBToRocksDB(const std::string& leveldb_path,
                                       const std::string& rocksdb_path) {
        // Open LevelDB
        leveldb::DB* level_db;
        leveldb::Options level_opts;
        level_opts.create_if_missing = false;

        leveldb::Status level_status = leveldb::DB::Open(level_opts,
                                                         leveldb_path,
                                                         &level_db);
        if (!level_status.ok()) {
            return false;
        }

        // Open RocksDB
        rocksdb::DB* rocks_db;
        rocksdb::Options rocks_opts;
        rocks_opts.create_if_missing = true;

        rocksdb::Status rocks_status = rocksdb::DB::Open(rocks_opts,
                                                         rocksdb_path,
                                                         &rocks_db);
        if (!rocks_status.ok()) {
            delete level_db;
            return false;
        }

        // Migrate data
        rocksdb::WriteBatch batch;
        int count = 0;
        const int batch_size = 10000;

        leveldb::Iterator* it = level_db->NewIterator(leveldb::ReadOptions());

        for (it->SeekToFirst(); it->Valid(); it->Next()) {
            batch.Put(it->key(), it->value());
            count++;

            if (count % batch_size == 0) {
                rocks_db->Write(rocksdb::WriteOptions(), &batch);
                batch.Clear();
                std::cout << "Migrated " << count << " keys..." << std::endl;
            }
        }

        // Write remaining
        if (batch.Count() > 0) {
            rocks_db->Write(rocksdb::WriteOptions(), &batch);
        }

        delete it;
        delete level_db;
        delete rocks_db;

        std::cout << "Migration complete: " << count << " keys" << std::endl;
        return true;
    }
};
```

## 16. Troubleshooting Guide

### Corruption Recovery

```cpp
#include <leveldb/db.h>

void AttemptRepair(const std::string& db_path) {
    leveldb::Status status = leveldb::RepairDB(db_path, leveldb::Options());

    if (status.ok()) {
        std::cout << "Database repaired successfully" << std::endl;
    } else {
        std::cerr << "Repair failed: " << status.ToString() << std::endl;
    }
}
```

### Common Issues

```cpp
class Troubleshooting {
public:
    // Issue: Too many open files
    static void FixTooManyFiles(leveldb::Options& options) {
        options.max_open_files = 1000;  // Reduce from default

        // Or increase system limit:
        // ulimit -n 10000
    }

    // Issue: Write stalls
    static void DiagnoseWriteStalls(leveldb::DB* db) {
        std::string num_l0_files;
        db->GetProperty("leveldb.num-files-at-level0", &num_l0_files);

        std::cout << "L0 files: " << num_l0_files << std::endl;

        // If too many L0 files, compaction is slow
        // Solution: Increase write_buffer_size or trigger manual compaction
    }

    // Issue: High memory usage
    static void ReduceMemoryUsage(leveldb::Options& options) {
        // Reduce write buffer
        options.write_buffer_size = 4 * 1024 * 1024;  // 4MB

        // Reduce block cache
        options.block_cache = leveldb::NewLRUCache(128 * 1024 * 1024);  // 128MB

        // Reduce max open files
        options.max_open_files = 500;
    }

    // Issue: Slow reads
    static void DiagnoseSlowReads(leveldb::DB* db) {
        // Check number of files
        for (int level = 0; level < 7; ++level) {
            std::string prop = "leveldb.num-files-at-level" + std::to_string(level);
            std::string value;
            db->GetProperty(prop, &value);
            std::cout << "Level " << level << ": " << value << " files" << std::endl;
        }

        // Many L0 files = slower reads
        // Solution: Trigger manual compaction
        db->CompactRange(nullptr, nullptr);
    }
};
```

## 17. Performance Tuning Checklist

### Hardware Optimization

```markdown
**Storage:**
- Use SSD for best performance (100x faster than HDD)
- NVMe better than SATA SSD
- Avoid network storage for low latency

**Memory:**
- Block cache should be 25-50% of available RAM
- Leave room for OS page cache
- More RAM = better read performance

**CPU:**
- Modern CPU with fast single-core performance
- Compression benefits from multiple cores
- Snappy compression is CPU-efficient
```

### Configuration Tuning

```cpp
// Workload-specific configurations

// Write-heavy workload
leveldb::Options WriteHeavy() {
    leveldb::Options opts;
    opts.write_buffer_size = 64 * 1024 * 1024;  // Large buffer
    opts.max_file_size = 64 * 1024 * 1024;      // Larger files
    opts.compression = leveldb::kSnappyCompression;
    opts.block_size = 64 * 1024;                // Larger blocks
    return opts;
}

// Read-heavy workload
leveldb::Options ReadHeavy() {
    leveldb::Options opts;
    opts.block_cache = leveldb::NewLRUCache(1024 * 1024 * 1024);  // Large cache
    opts.block_size = 4 * 1024;                 // Smaller blocks
    opts.max_open_files = 10000;                // Keep files open
    opts.compression = leveldb::kSnappyCompression;
    return opts;
}

// Balanced workload
leveldb::Options Balanced() {
    leveldb::Options opts;
    opts.write_buffer_size = 16 * 1024 * 1024;  // 16MB
    opts.block_cache = leveldb::NewLRUCache(256 * 1024 * 1024);  // 256MB
    opts.block_size = 16 * 1024;                // 16KB
    opts.max_open_files = 5000;
    opts.compression = leveldb::kSnappyCompression;
    return opts;
}
```

## 18. Comparison with Alternatives

### LevelDB vs RocksDB

| Feature | LevelDB | RocksDB |
|---------|---------|---------|
| **Development** | Google (minimal updates) | Meta (active) |
| **Write Performance** | Good | Better (parallel compaction) |
| **Read Performance** | Good | Better (bloom filters, prefixing) |
| **Column Families** | ❌ No | ✅ Yes |
| **Transactions** | ❌ No | ✅ Yes |
| **Tuning Options** | Limited | Extensive |
| **Compression** | Snappy only | Multiple algorithms |
| **Production Use** | Decreasing | Increasing |

### LevelDB vs LMDB

| Feature | LevelDB | LMDB |
|---------|---------|------|
| **Architecture** | LSM-tree | B+ tree |
| **Write Amplification** | High (10-20x) | Low (1-2x) |
| **Write Throughput** | High | Lower |
| **Read Latency** | Good | Excellent |
| **Concurrent Writes** | Single writer | Single writer |
| **Memory Usage** | Configurable cache | Memory-mapped |
| **Data Safety** | WAL | Memory-mapped, sync |

### LevelDB vs Berkeley DB

| Feature | LevelDB | Berkeley DB |
|---------|---------|-------------|
| **Data Model** | Key-value only | Key-value + multiple access methods |
| **Write Performance** | Better | Good |
| **API Complexity** | Simple | Complex |
| **Transactions** | Batch only | Full ACID |
| **Maintenance** | None required | Configuration needed |
| **Replication** | ❌ No | ✅ Yes |

## 19. Production Checklist

```markdown
**Pre-Deployment:**
- [ ] Choose appropriate configuration (write-heavy/read-heavy/balanced)
- [ ] Set proper file descriptor limits
- [ ] Configure block cache size based on available RAM
- [ ] Enable compression (Snappy recommended)
- [ ] Test backup/restore procedures
- [ ] Implement monitoring for database size and L0 files
- [ ] Set up file permissions (0700 for directory, 0600 for files)

**Post-Deployment:**
- [ ] Monitor write latency (should be <1ms)
- [ ] Monitor read latency (should be <10ms)
- [ ] Check L0 file count (should be <10)
- [ ] Monitor disk usage and growth rate
- [ ] Verify backups are successful
- [ ] Test restore procedure in non-prod
- [ ] Monitor file descriptor usage
- [ ] Check for log errors

**Maintenance:**
- [ ] Daily: Check disk space, verify backups
- [ ] Weekly: Review performance metrics, check L0 file count
- [ ] Monthly: Test restore procedure, review capacity planning
- [ ] Quarterly: Consider compaction if needed
```

## 20. Resources and References

### Official Documentation
- **LevelDB GitHub**: https://github.com/google/leveldb
- **Documentation**: https://github.com/google/leveldb/blob/main/doc/index.md
- **Implementation Notes**: https://github.com/google/leveldb/blob/main/doc/impl.md

### Papers
- **Bigtable: A Distributed Storage System** (Google, 2006) - LSM-tree foundation
- **The Log-Structured Merge-Tree (LSM-Tree)** (O'Neil et al., 1996) - Original LSM paper

### Language Bindings
- **Python**: plyvel - https://plyvel.readthedocs.io/
- **Go**: goleveldb - https://github.com/syndtr/goleveldb
- **Node.js**: level - https://github.com/Level/level
- **Rust**: leveldb - https://github.com/skade/leveldb
- **Java**: leveldbjni - https://github.com/fusesource/leveldbjni

### Tools
- **leveldb-cli**: Command-line interface for LevelDB
- **ldb**: Inspection and debugging tool
- **Benchmarking**: db_bench (included in LevelDB source)

### Production Users
- **Chrome/Chromium**: IndexedDB storage
- **Bitcoin Core**: UTXO database (migrated to LevelDB)
- **Ethereum**: Geth client uses LevelDB
- **Minecraft Bedrock**: World storage
- **Riak**: Alternative backend

### Migration Paths
- **To RocksDB**: For better performance and more features
- **To LMDB**: For read-heavy workloads
- **To SQLite**: For SQL queries and complex transactions

---

## Quick Start Example

```cpp
#include <leveldb/db.h>
#include <iostream>

int main() {
    // Open database
    leveldb::DB* db;
    leveldb::Options options;
    options.create_if_missing = true;

    leveldb::Status status = leveldb::DB::Open(options, "/tmp/testdb", &db);
    if (!status.ok()) {
        std::cerr << "Unable to open database: " << status.ToString() << std::endl;
        return 1;
    }

    // Write
    status = db->Put(leveldb::WriteOptions(), "key1", "value1");
    if (!status.ok()) {
        std::cerr << "Write failed: " << status.ToString() << std::endl;
    }

    // Read
    std::string value;
    status = db->Get(leveldb::ReadOptions(), "key1", &value);
    if (status.ok()) {
        std::cout << "key1: " << value << std::endl;
    } else {
        std::cerr << "Read failed: " << status.ToString() << std::endl;
    }

    // Iterate
    leveldb::Iterator* it = db->NewIterator(leveldb::ReadOptions());
    for (it->SeekToFirst(); it->Valid(); it->Next()) {
        std::cout << it->key().ToString() << ": "
                  << it->value().ToString() << std::endl;
    }
    delete it;

    // Cleanup
    delete db;
    return 0;
}
```

Compile with:
```bash
g++ -std=c++17 -O3 example.cpp -lleveldb -lpthread -o example
./example
```

This guide provides comprehensive coverage of LevelDB for production use, from basic operations to advanced patterns and troubleshooting.

---

**End of LevelDB Development Guidelines**
