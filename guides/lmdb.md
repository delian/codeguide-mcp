# LMDB Development Guidelines
Mandatory coding standards and development practices for LMDB development. LMDB C API, language bindings (Python, Node, Rust, Go), mdb_stat/mdb_copy, valgrind.

---

**Agent Profile**: The LMDB Expert
**Role**: Senior Embedded Database Engineer & Key-Value Store Specialist
**Objective**: Generate production-ready, high-performance and reliable LMDB-backed storage solutions.
**Tools**: LMDB C API, language bindings (Python, Node, Rust, Go), mdb_stat/mdb_copy, valgrind

---

## 1. Core Philosophies: LMDB-FIRST

The agent must adhere to the **LMDB-FIRST** principles for every LMDB implementation:

**Test-Driven Development (TDD)**: ALWAYS write tests BEFORE implementation (Red-Green-Refactor cycle mandatory).
**Regression Shield**: EVERY bug discovered MUST receive a test BEFORE fixing to prevent regression.

- **L**ightning-fast reads: Design for read-heavy workloads; prefer zero-copy access; avoid unnecessary copies.
- **M**emory-mapped semantics: Never hold pointers from read transactions after txn ends; respect mmap and process layout.
- **D**urable commits: Use sync (or explicit MDB_NOSYNC only when acceptable); open/close env safely; handle disk full.
- **B**-tree and single-writer: Use one writer at a time; keep write transactions short; use cursors for range scans.
- **Verified Code**: Agent-generated code MUST use transactions for all access, handle MDB_* errors, and pass tests before delivery.

---

## 2. Core Concepts and Architecture

LMDB (Lightning Memory-Mapped Database) is an ultra-fast, ultra-compact embedded key-value database developed by Symas Corporation. It uses memory-mapped files and a B+ tree architecture for exceptional read performance with ACID transactions.

### Memory-Mapped B+ Tree Architecture

```
LMDB Architecture:

┌─────────────────────────────────────────────────────────────┐
│                   Memory-Mapped File                         │
│                                                              │
│  ┌────────────────────────────────────────────────────┐    │
│  │              B+ Tree Structure                      │    │
│  │                                                     │    │
│  │         Root Node (in memory)                      │    │
│  │              ↓                                      │    │
│  │    ┌─────────┴─────────┐                          │    │
│  │    │                    │                          │    │
│  │  Branch              Branch                        │    │
│  │    ↓                    ↓                          │    │
│  │  Leaf → Leaf → Leaf → Leaf                        │    │
│  │  (Key-Value pairs in sorted order)                │    │
│  └────────────────────────────────────────────────────┘    │
│                                                              │
│  Direct memory access via mmap()                            │
│  Zero-copy reads                                            │
│  Page size: 4KB default                                     │
└─────────────────────────────────────────────────────────────┘

Read Operation:
1. Memory mapping → Direct page access (no I/O)
2. B+ tree traversal → O(log n) page reads
3. Zero-copy → Return pointer to memory
4. Result: Sub-microsecond latency

Write Operation (MVCC Copy-on-Write):
1. Create transaction
2. Copy-on-write modified pages
3. Update parent pointers
4. Atomic commit (update meta page)
5. Old pages available for concurrent readers
```

### Key Characteristics

**Memory-Mapped I/O:**
- Database file mapped directly into process address space
- OS manages paging and caching
- Zero-copy reads (pointer to mapped memory)
- Writes use copy-on-write (COW)

**MVCC (Multi-Version Concurrency Control):**
- Single writer, multiple readers
- Readers never block writers
- Writers never block readers
- Snapshot isolation for readers
- No read locks

**ACID Transactions:**
- Full ACID compliance
- Atomic commits
- Durable writes
- Isolation via snapshots
- Consistency guarantees

**Performance Profile:**
```
Reads:  Extremely fast (sub-microsecond)
        - Memory-mapped access
        - Zero-copy operations
        - No serialization

Writes: Good (but slower than reads)
        - Copy-on-write overhead
        - Single writer limitation
        - Fsync for durability

Space:  Efficient
        - Low write amplification (1-2x)
        - Compact B+ tree structure
        - No compaction needed
```

### Comparison: LMDB vs LSM-tree Databases

| Aspect | LMDB (B+ Tree) | LevelDB/RocksDB (LSM) |
|--------|----------------|----------------------|
| **Read Speed** | Faster (direct memory) | Slower (block cache) |
| **Write Speed** | Good | Better (sequential) |
| **Write Amplification** | Low (1-2x) | High (10-30x) |
| **Space Amplification** | Low | Higher (compaction) |
| **Compaction** | Not needed | Required |
| **Concurrent Readers** | Unlimited | Unlimited |
| **Concurrent Writers** | 1 | 1 (LevelDB), Many (RocksDB) |

## 3. Installation and Setup

### Ubuntu/Debian Installation

```bash
# Install from package manager
sudo apt-get update
sudo apt-get install -y liblmdb-dev lmdb-utils

# Verify installation
mdb_stat -V
```

### Building from Source

```bash
# Clone repository
git clone https://git.openldap.org/openldap/openldap.git
cd openldap/libraries/liblmdb

# Build
make

# Install
sudo make install
sudo ldconfig

# Verify
ls -la /usr/local/lib/liblmdb.*
```

### macOS Installation

```bash
# Using Homebrew
brew install lmdb

# Verify
mdb_stat -V
```

### CMake Integration

```cmake
# CMakeLists.txt
cmake_minimum_required(VERSION 3.10)
project(MyLMDBApp)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

# Compiler optimizations
set(CMAKE_C_FLAGS_RELEASE "-O3 -DNDEBUG -march=native")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native")

# Find LMDB
find_library(LMDB_LIBRARY lmdb REQUIRED)
find_path(LMDB_INCLUDE_DIR lmdb.h REQUIRED)

include_directories(${LMDB_INCLUDE_DIR})

add_executable(myapp main.c)
target_link_libraries(myapp ${LMDB_LIBRARY} pthread)
```

### Basic Verification

```c
// test_lmdb.c
#include <lmdb.h>
#include <stdio.h>
#include <string.h>

int main() {
    MDB_env *env;
    MDB_dbi dbi;
    MDB_txn *txn;
    int rc;

    // Create environment
    rc = mdb_env_create(&env);
    if (rc != 0) {
        fprintf(stderr, "mdb_env_create: %s\n", mdb_strerror(rc));
        return 1;
    }

    // Open environment
    rc = mdb_env_open(env, "./testdb", MDB_FIXEDMAP, 0664);
    if (rc != 0) {
        fprintf(stderr, "mdb_env_open: %s\n", mdb_strerror(rc));
        mdb_env_close(env);
        return 1;
    }

    printf("LMDB version: %s\n", MDB_VERSION_STRING);
    printf("LMDB initialized successfully\n");

    mdb_env_close(env);
    return 0;
}
```

Compile and run:
```bash
gcc -O3 test_lmdb.c -llmdb -o test_lmdb
./test_lmdb
```

## 4. C API - Basic Operations

### Environment and Database Setup

```c
#include <lmdb.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    MDB_env *env;
    MDB_dbi dbi;
} LMDBStore;

int lmdb_init(LMDBStore *store, const char *path, size_t mapsize) {
    int rc;

    // Create environment
    rc = mdb_env_create(&store->env);
    if (rc != 0) {
        fprintf(stderr, "mdb_env_create: %s\n", mdb_strerror(rc));
        return rc;
    }

    // Set map size (maximum database size)
    // Must be set before opening environment
    rc = mdb_env_set_mapsize(store->env, mapsize);
    if (rc != 0) {
        fprintf(stderr, "mdb_env_set_mapsize: %s\n", mdb_strerror(rc));
        mdb_env_close(store->env);
        return rc;
    }

    // Set max databases (if using multiple databases)
    rc = mdb_env_set_maxdbs(store->env, 10);
    if (rc != 0) {
        fprintf(stderr, "mdb_env_set_maxdbs: %s\n", mdb_strerror(rc));
        mdb_env_close(store->env);
        return rc;
    }

    // Open environment
    // MDB_FIXEDMAP: Use fixed address for mmap (better performance)
    // MDB_NOSYNC: Don't fsync after commit (faster, less durable)
    // MDB_WRITEMAP: Use writable mmap (better write performance)
    // MDB_MAPASYNC: Async msync (with MDB_WRITEMAP)
    rc = mdb_env_open(store->env, path,
                      MDB_FIXEDMAP | MDB_NOSUBDIR,
                      0664);
    if (rc != 0) {
        fprintf(stderr, "mdb_env_open: %s\n", mdb_strerror(rc));
        mdb_env_close(store->env);
        return rc;
    }

    // Open database in environment
    MDB_txn *txn;
    rc = mdb_txn_begin(store->env, NULL, 0, &txn);
    if (rc != 0) {
        fprintf(stderr, "mdb_txn_begin: %s\n", mdb_strerror(rc));
        mdb_env_close(store->env);
        return rc;
    }

    rc = mdb_dbi_open(txn, NULL, MDB_CREATE, &store->dbi);
    if (rc != 0) {
        fprintf(stderr, "mdb_dbi_open: %s\n", mdb_strerror(rc));
        mdb_txn_abort(txn);
        mdb_env_close(store->env);
        return rc;
    }

    rc = mdb_txn_commit(txn);
    if (rc != 0) {
        fprintf(stderr, "mdb_txn_commit: %s\n", mdb_strerror(rc));
        mdb_env_close(store->env);
        return rc;
    }

    return MDB_SUCCESS;
}

void lmdb_close(LMDBStore *store) {
    mdb_dbi_close(store->env, store->dbi);
    mdb_env_close(store->env);
}
```

### Put, Get, Delete Operations

```c
#include <lmdb.h>
#include <string.h>

// Write a key-value pair
int lmdb_put(LMDBStore *store, const char *key, const char *value) {
    MDB_txn *txn;
    MDB_val mdb_key, mdb_value;
    int rc;

    // Begin write transaction
    rc = mdb_txn_begin(store->env, NULL, 0, &txn);
    if (rc != 0) {
        return rc;
    }

    // Setup key and value
    mdb_key.mv_data = (void *)key;
    mdb_key.mv_size = strlen(key);
    mdb_value.mv_data = (void *)value;
    mdb_value.mv_size = strlen(value);

    // Put key-value
    rc = mdb_put(txn, store->dbi, &mdb_key, &mdb_value, 0);
    if (rc != 0) {
        mdb_txn_abort(txn);
        return rc;
    }

    // Commit transaction
    rc = mdb_txn_commit(txn);
    return rc;
}

// Read a value by key
int lmdb_get(LMDBStore *store, const char *key, char **value, size_t *value_len) {
    MDB_txn *txn;
    MDB_val mdb_key, mdb_value;
    int rc;

    // Begin read-only transaction
    rc = mdb_txn_begin(store->env, NULL, MDB_RDONLY, &txn);
    if (rc != 0) {
        return rc;
    }

    // Setup key
    mdb_key.mv_data = (void *)key;
    mdb_key.mv_size = strlen(key);

    // Get value
    rc = mdb_get(txn, store->dbi, &mdb_key, &mdb_value);
    if (rc == 0) {
        // Copy value (mdb_value.mv_data is only valid during transaction)
        *value = malloc(mdb_value.mv_size + 1);
        memcpy(*value, mdb_value.mv_data, mdb_value.mv_size);
        (*value)[mdb_value.mv_size] = '\0';
        *value_len = mdb_value.mv_size;
    }

    // Abort read transaction (no commit needed for read-only)
    mdb_txn_abort(txn);
    return rc;
}

// Delete a key
int lmdb_delete(LMDBStore *store, const char *key) {
    MDB_txn *txn;
    MDB_val mdb_key;
    int rc;

    // Begin write transaction
    rc = mdb_txn_begin(store->env, NULL, 0, &txn);
    if (rc != 0) {
        return rc;
    }

    // Setup key
    mdb_key.mv_data = (void *)key;
    mdb_key.mv_size = strlen(key);

    // Delete key
    rc = mdb_del(txn, store->dbi, &mdb_key, NULL);
    if (rc != 0) {
        mdb_txn_abort(txn);
        return rc;
    }

    // Commit transaction
    rc = mdb_txn_commit(txn);
    return rc;
}

// Check if key exists
int lmdb_exists(LMDBStore *store, const char *key) {
    MDB_txn *txn;
    MDB_val mdb_key, mdb_value;
    int rc;

    rc = mdb_txn_begin(store->env, NULL, MDB_RDONLY, &txn);
    if (rc != 0) {
        return 0;
    }

    mdb_key.mv_data = (void *)key;
    mdb_key.mv_size = strlen(key);

    rc = mdb_get(txn, store->dbi, &mdb_key, &mdb_value);
    mdb_txn_abort(txn);

    return (rc == 0);
}
```

### Usage Example

```c
int main() {
    LMDBStore store;
    int rc;

    // Initialize LMDB (1GB map size)
    rc = lmdb_init(&store, "./mydb.mdb", 1ULL * 1024 * 1024 * 1024);
    if (rc != 0) {
        fprintf(stderr, "Failed to initialize LMDB\n");
        return 1;
    }

    // Write
    rc = lmdb_put(&store, "user:1:name", "Alice");
    if (rc != 0) {
        fprintf(stderr, "Put failed: %s\n", mdb_strerror(rc));
    }

    // Read
    char *value;
    size_t value_len;
    rc = lmdb_get(&store, "user:1:name", &value, &value_len);
    if (rc == 0) {
        printf("Value: %s\n", value);
        free(value);
    } else if (rc == MDB_NOTFOUND) {
        printf("Key not found\n");
    } else {
        fprintf(stderr, "Get failed: %s\n", mdb_strerror(rc));
    }

    // Check existence
    if (lmdb_exists(&store, "user:1:name")) {
        printf("Key exists\n");
    }

    // Delete
    rc = lmdb_delete(&store, "user:1:name");
    if (rc != 0) {
        fprintf(stderr, "Delete failed: %s\n", mdb_strerror(rc));
    }

    // Cleanup
    lmdb_close(&store);
    return 0;
}
```

## 5. Transactions and MVCC

LMDB provides full ACID transactions with snapshot isolation.

### Read-Only Transactions

```c
// Read-only transaction (never blocks writers)
int read_transaction_example(LMDBStore *store) {
    MDB_txn *txn;
    MDB_val key, value;
    int rc;

    // Begin read-only transaction
    rc = mdb_txn_begin(store->env, NULL, MDB_RDONLY, &txn);
    if (rc != 0) {
        return rc;
    }

    // Read multiple keys with consistent snapshot
    key.mv_data = "key1";
    key.mv_size = strlen("key1");
    rc = mdb_get(txn, store->dbi, &key, &value);
    if (rc == 0) {
        printf("key1: %.*s\n", (int)value.mv_size, (char *)value.mv_data);
    }

    key.mv_data = "key2";
    key.mv_size = strlen("key2");
    rc = mdb_get(txn, store->dbi, &key, &value);
    if (rc == 0) {
        printf("key2: %.*s\n", (int)value.mv_size, (char *)value.mv_data);
    }

    // Abort transaction (no commit needed for read-only)
    mdb_txn_abort(txn);
    return MDB_SUCCESS;
}
```

### Write Transactions

```c
// Write transaction with multiple operations
int write_transaction_example(LMDBStore *store) {
    MDB_txn *txn;
    MDB_val key, value;
    int rc;

    // Begin write transaction
    rc = mdb_txn_begin(store->env, NULL, 0, &txn);
    if (rc != 0) {
        return rc;
    }

    // Multiple write operations in single transaction
    key.mv_data = "user:1:name";
    key.mv_size = strlen("user:1:name");
    value.mv_data = "Alice";
    value.mv_size = strlen("Alice");
    rc = mdb_put(txn, store->dbi, &key, &value, 0);
    if (rc != 0) {
        mdb_txn_abort(txn);
        return rc;
    }

    key.mv_data = "user:1:email";
    key.mv_size = strlen("user:1:email");
    value.mv_data = "alice@example.com";
    value.mv_size = strlen("alice@example.com");
    rc = mdb_put(txn, store->dbi, &key, &value, 0);
    if (rc != 0) {
        mdb_txn_abort(txn);
        return rc;
    }

    // Commit transaction (atomic)
    rc = mdb_txn_commit(txn);
    return rc;
}
```

### Transfer Example (Atomic Update)

```c
// Atomic account transfer
int transfer_funds(LMDBStore *store,
                  const char *from_account,
                  const char *to_account,
                  int amount) {
    MDB_txn *txn;
    MDB_val key, value;
    int rc;

    // Begin write transaction
    rc = mdb_txn_begin(store->env, NULL, 0, &txn);
    if (rc != 0) {
        return rc;
    }

    // Read from_account balance
    key.mv_data = (void *)from_account;
    key.mv_size = strlen(from_account);
    rc = mdb_get(txn, store->dbi, &key, &value);
    if (rc != 0) {
        mdb_txn_abort(txn);
        return rc;
    }

    int from_balance = atoi((char *)value.mv_data);

    // Read to_account balance
    key.mv_data = (void *)to_account;
    key.mv_size = strlen(to_account);
    rc = mdb_get(txn, store->dbi, &key, &value);
    if (rc != 0) {
        mdb_txn_abort(txn);
        return rc;
    }

    int to_balance = atoi((char *)value.mv_data);

    // Validate
    if (from_balance < amount) {
        mdb_txn_abort(txn);
        return -1;  // Insufficient funds
    }

    // Update balances
    from_balance -= amount;
    to_balance += amount;

    char buffer[32];

    // Write from_account
    key.mv_data = (void *)from_account;
    key.mv_size = strlen(from_account);
    snprintf(buffer, sizeof(buffer), "%d", from_balance);
    value.mv_data = buffer;
    value.mv_size = strlen(buffer);
    rc = mdb_put(txn, store->dbi, &key, &value, 0);
    if (rc != 0) {
        mdb_txn_abort(txn);
        return rc;
    }

    // Write to_account
    key.mv_data = (void *)to_account;
    key.mv_size = strlen(to_account);
    snprintf(buffer, sizeof(buffer), "%d", to_balance);
    value.mv_data = buffer;
    value.mv_size = strlen(buffer);
    rc = mdb_put(txn, store->dbi, &key, &value, 0);
    if (rc != 0) {
        mdb_txn_abort(txn);
        return rc;
    }

    // Commit transaction (atomic)
    rc = mdb_txn_commit(txn);
    return rc;
}
```

### Nested Transactions (Child Transactions)

```c
// Nested transaction support
int nested_transaction_example(LMDBStore *store) {
    MDB_txn *parent_txn, *child_txn;
    MDB_val key, value;
    int rc;

    // Begin parent transaction
    rc = mdb_txn_begin(store->env, NULL, 0, &parent_txn);
    if (rc != 0) {
        return rc;
    }

    // Write in parent
    key.mv_data = "key1";
    key.mv_size = strlen("key1");
    value.mv_data = "value1";
    value.mv_size = strlen("value1");
    mdb_put(parent_txn, store->dbi, &key, &value, 0);

    // Begin child transaction
    rc = mdb_txn_begin(store->env, parent_txn, 0, &child_txn);
    if (rc != 0) {
        mdb_txn_abort(parent_txn);
        return rc;
    }

    // Write in child
    key.mv_data = "key2";
    key.mv_size = strlen("key2");
    value.mv_data = "value2";
    value.mv_size = strlen("value2");
    rc = mdb_put(child_txn, store->dbi, &key, &value, 0);
    if (rc != 0) {
        mdb_txn_abort(child_txn);
        mdb_txn_abort(parent_txn);
        return rc;
    }

    // Commit child
    rc = mdb_txn_commit(child_txn);
    if (rc != 0) {
        mdb_txn_abort(parent_txn);
        return rc;
    }

    // Commit parent (also commits child)
    rc = mdb_txn_commit(parent_txn);
    return rc;
}
```

## 6. Cursors and Range Scans

Cursors provide efficient iteration over key-value pairs.

### Basic Cursor Operations

```c
#include <lmdb.h>

// Forward iteration
int cursor_forward_scan(LMDBStore *store, const char *prefix) {
    MDB_txn *txn;
    MDB_cursor *cursor;
    MDB_val key, value;
    int rc;

    // Begin read-only transaction
    rc = mdb_txn_begin(store->env, NULL, MDB_RDONLY, &txn);
    if (rc != 0) {
        return rc;
    }

    // Open cursor
    rc = mdb_cursor_open(txn, store->dbi, &cursor);
    if (rc != 0) {
        mdb_txn_abort(txn);
        return rc;
    }

    // Position cursor at prefix
    key.mv_data = (void *)prefix;
    key.mv_size = strlen(prefix);
    rc = mdb_cursor_get(cursor, &key, &value, MDB_SET_RANGE);

    // Iterate forward
    while (rc == 0) {
        // Check if key still matches prefix
        if (key.mv_size < strlen(prefix) ||
            memcmp(key.mv_data, prefix, strlen(prefix)) != 0) {
            break;
        }

        printf("%.*s: %.*s\n",
               (int)key.mv_size, (char *)key.mv_data,
               (int)value.mv_size, (char *)value.mv_data);

        // Move to next key
        rc = mdb_cursor_get(cursor, &key, &value, MDB_NEXT);
    }

    // Cleanup
    mdb_cursor_close(cursor);
    mdb_txn_abort(txn);
    return MDB_SUCCESS;
}

// Reverse iteration
int cursor_reverse_scan(LMDBStore *store) {
    MDB_txn *txn;
    MDB_cursor *cursor;
    MDB_val key, value;
    int rc;

    rc = mdb_txn_begin(store->env, NULL, MDB_RDONLY, &txn);
    if (rc != 0) {
        return rc;
    }

    rc = mdb_cursor_open(txn, store->dbi, &cursor);
    if (rc != 0) {
        mdb_txn_abort(txn);
        return rc;
    }

    // Position at last key
    rc = mdb_cursor_get(cursor, &key, &value, MDB_LAST);

    // Iterate backward
    while (rc == 0) {
        printf("%.*s: %.*s\n",
               (int)key.mv_size, (char *)key.mv_data,
               (int)value.mv_size, (char *)value.mv_data);

        // Move to previous key
        rc = mdb_cursor_get(cursor, &key, &value, MDB_PREV);
    }

    mdb_cursor_close(cursor);
    mdb_txn_abort(txn);
    return MDB_SUCCESS;
}

// Range scan between two keys
int cursor_range_scan(LMDBStore *store,
                     const char *start_key,
                     const char *end_key) {
    MDB_txn *txn;
    MDB_cursor *cursor;
    MDB_val key, value;
    int rc;

    rc = mdb_txn_begin(store->env, NULL, MDB_RDONLY, &txn);
    if (rc != 0) {
        return rc;
    }

    rc = mdb_cursor_open(txn, store->dbi, &cursor);
    if (rc != 0) {
        mdb_txn_abort(txn);
        return rc;
    }

    // Position at start_key (or first key >= start_key)
    key.mv_data = (void *)start_key;
    key.mv_size = strlen(start_key);
    rc = mdb_cursor_get(cursor, &key, &value, MDB_SET_RANGE);

    // Iterate until end_key
    while (rc == 0) {
        // Check if beyond end_key
        int cmp = memcmp(key.mv_data, end_key,
                        (key.mv_size < strlen(end_key)) ? key.mv_size : strlen(end_key));
        if (cmp > 0) {
            break;
        }

        printf("%.*s: %.*s\n",
               (int)key.mv_size, (char *)key.mv_data,
               (int)value.mv_size, (char *)value.mv_data);

        rc = mdb_cursor_get(cursor, &key, &value, MDB_NEXT);
    }

    mdb_cursor_close(cursor);
    mdb_txn_abort(txn);
    return MDB_SUCCESS;
}

// Count keys with prefix
int cursor_count_prefix(LMDBStore *store, const char *prefix, size_t *count) {
    MDB_txn *txn;
    MDB_cursor *cursor;
    MDB_val key, value;
    int rc;

    *count = 0;

    rc = mdb_txn_begin(store->env, NULL, MDB_RDONLY, &txn);
    if (rc != 0) {
        return rc;
    }

    rc = mdb_cursor_open(txn, store->dbi, &cursor);
    if (rc != 0) {
        mdb_txn_abort(txn);
        return rc;
    }

    key.mv_data = (void *)prefix;
    key.mv_size = strlen(prefix);
    rc = mdb_cursor_get(cursor, &key, &value, MDB_SET_RANGE);

    while (rc == 0) {
        if (key.mv_size < strlen(prefix) ||
            memcmp(key.mv_data, prefix, strlen(prefix)) != 0) {
            break;
        }

        (*count)++;
        rc = mdb_cursor_get(cursor, &key, &value, MDB_NEXT);
    }

    mdb_cursor_close(cursor);
    mdb_txn_abort(txn);
    return MDB_SUCCESS;
}
```

### Cursor Positioning Operations

```c
// MDB_FIRST      - First key in database
// MDB_LAST       - Last key in database
// MDB_NEXT       - Next key
// MDB_PREV       - Previous key
// MDB_SET        - Position at key (exact match)
// MDB_SET_RANGE  - Position at key >= specified key
// MDB_GET_CURRENT - Get current key/value
// MDB_NEXT_DUP   - Next duplicate key (MDB_DUPSORT)
// MDB_PREV_DUP   - Previous duplicate key (MDB_DUPSORT)

// Example: Find first key >= "user:100"
void position_cursor_example(LMDBStore *store) {
    MDB_txn *txn;
    MDB_cursor *cursor;
    MDB_val key, value;

    mdb_txn_begin(store->env, NULL, MDB_RDONLY, &txn);
    mdb_cursor_open(txn, store->dbi, &cursor);

    key.mv_data = "user:100";
    key.mv_size = strlen("user:100");

    // Find first key >= "user:100"
    int rc = mdb_cursor_get(cursor, &key, &value, MDB_SET_RANGE);
    if (rc == 0) {
        printf("Found: %.*s\n", (int)key.mv_size, (char *)key.mv_data);
    }

    mdb_cursor_close(cursor);
    mdb_txn_abort(txn);
}
```

## 7. Language Bindings

### Python (lmdb)

```bash
# Installation
pip install lmdb
```

```python
import lmdb
from typing import Optional, List, Tuple
import json

class LMDBStore:
    def __init__(self, path: str, map_size: int = 10 * 1024 * 1024 * 1024):
        """Initialize LMDB store.

        Args:
            path: Database file path
            map_size: Maximum database size in bytes (default 10GB)
        """
        self.env = lmdb.open(
            path,
            map_size=map_size,
            subdir=False,
            max_dbs=10,
            readonly=False,
            metasync=True,
            sync=True,
            map_async=False,
            writemap=False,
            lock=True
        )

    def put(self, key: bytes, value: bytes) -> None:
        """Write a key-value pair."""
        with self.env.begin(write=True) as txn:
            txn.put(key, value)

    def get(self, key: bytes) -> Optional[bytes]:
        """Read a value by key."""
        with self.env.begin() as txn:
            return txn.get(key)

    def delete(self, key: bytes) -> bool:
        """Delete a key."""
        with self.env.begin(write=True) as txn:
            return txn.delete(key)

    def exists(self, key: bytes) -> bool:
        """Check if key exists."""
        with self.env.begin() as txn:
            return txn.get(key) is not None

    def scan_prefix(self, prefix: bytes) -> List[Tuple[bytes, bytes]]:
        """Scan all keys with given prefix."""
        results = []
        with self.env.begin() as txn:
            cursor = txn.cursor()
            if cursor.set_range(prefix):
                for key, value in cursor:
                    if not key.startswith(prefix):
                        break
                    results.append((key, value))
        return results

    def count_prefix(self, prefix: bytes) -> int:
        """Count keys with given prefix."""
        count = 0
        with self.env.begin() as txn:
            cursor = txn.cursor()
            if cursor.set_range(prefix):
                for key, _ in cursor:
                    if not key.startswith(prefix):
                        break
                    count += 1
        return count

    def batch_write(self, operations: List[Tuple[str, bytes, Optional[bytes]]]) -> None:
        """Atomic batch write.

        Args:
            operations: List of (op, key, value) tuples
                       op can be 'put' or 'delete'
        """
        with self.env.begin(write=True) as txn:
            for op, key, *args in operations:
                if op == 'put':
                    txn.put(key, args[0])
                elif op == 'delete':
                    txn.delete(key)

    def iterator(self, start: Optional[bytes] = None,
                reverse: bool = False) -> List[Tuple[bytes, bytes]]:
        """Iterate over keys."""
        results = []
        with self.env.begin() as txn:
            cursor = txn.cursor()

            if reverse:
                if start:
                    cursor.set_range(start)
                else:
                    cursor.last()

                for key, value in cursor.iterprev():
                    results.append((key, value))
            else:
                if start:
                    cursor.set_range(start)
                else:
                    cursor.first()

                for key, value in cursor:
                    results.append((key, value))

        return results

    def transaction(self, write: bool = False):
        """Get transaction context manager."""
        return self.env.begin(write=write)

    def stat(self) -> dict:
        """Get database statistics."""
        with self.env.begin() as txn:
            return txn.stat()

    def close(self):
        """Close database."""
        self.env.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# Usage examples
def python_examples():
    with LMDBStore('/tmp/pylmdb.mdb') as store:
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
            ('delete', b'user:temp'),
        ])

        # Scan users
        users = store.scan_prefix(b'user:')
        for key, value in users:
            print(f"{key.decode()}: {value.decode()}")

        # Count users
        user_count = store.count_prefix(b'user:')
        print(f"Total users: {user_count}")

        # Statistics
        stats = store.stat()
        print(f"Stats: {stats}")


# Advanced: Transaction context
def transaction_example():
    store = LMDBStore('/tmp/pylmdb.mdb')

    # Manual transaction control
    with store.transaction(write=True) as txn:
        txn.put(b'key1', b'value1')
        txn.put(b'key2', b'value2')
        # Automatic commit on exit

    # Read-only transaction
    with store.transaction(write=False) as txn:
        value = txn.get(b'key1')
        print(f"Value: {value}")

    store.close()
```

### Go (lmdb-go)

```bash
# Installation
go get github.com/bmatsuo/lmdb-go/lmdb
```

```go
package main

import (
    "fmt"
    "log"
    "github.com/bmatsuo/lmdb-go/lmdb"
)

type LMDBStore struct {
    env *lmdb.Env
    dbi lmdb.DBI
}

func NewLMDBStore(path string, mapSize int64) (*LMDBStore, error) {
    env, err := lmdb.NewEnv()
    if err != nil {
        return nil, err
    }

    err = env.SetMapSize(mapSize)
    if err != nil {
        env.Close()
        return nil, err
    }

    err = env.SetMaxDBs(10)
    if err != nil {
        env.Close()
        return nil, err
    }

    err = env.Open(path, lmdb.NoSubdir, 0664)
    if err != nil {
        env.Close()
        return nil, err
    }

    var dbi lmdb.DBI
    err = env.Update(func(txn *lmdb.Txn) error {
        var txnErr error
        dbi, txnErr = txn.OpenDBI("", lmdb.Create)
        return txnErr
    })
    if err != nil {
        env.Close()
        return nil, err
    }

    return &LMDBStore{env: env, dbi: dbi}, nil
}

func (s *LMDBStore) Put(key, value []byte) error {
    return s.env.Update(func(txn *lmdb.Txn) error {
        return txn.Put(s.dbi, key, value, 0)
    })
}

func (s *LMDBStore) Get(key []byte) ([]byte, error) {
    var value []byte
    err := s.env.View(func(txn *lmdb.Txn) error {
        val, txnErr := txn.Get(s.dbi, key)
        if txnErr == nil {
            value = make([]byte, len(val))
            copy(value, val)
        }
        return txnErr
    })
    return value, err
}

func (s *LMDBStore) Delete(key []byte) error {
    return s.env.Update(func(txn *lmdb.Txn) error {
        return txn.Del(s.dbi, key, nil)
    })
}

func (s *LMDBStore) Exists(key []byte) (bool, error) {
    err := s.env.View(func(txn *lmdb.Txn) error {
        _, txnErr := txn.Get(s.dbi, key)
        return txnErr
    })
    if err == lmdb.NotFound {
        return false, nil
    }
    return err == nil, err
}

func (s *LMDBStore) ScanPrefix(prefix []byte) ([][2][]byte, error) {
    var results [][2][]byte

    err := s.env.View(func(txn *lmdb.Txn) error {
        cursor, txnErr := txn.OpenCursor(s.dbi)
        if txnErr != nil {
            return txnErr
        }
        defer cursor.Close()

        for {
            key, val, txnErr := cursor.Get(nil, nil, lmdb.Next)
            if lmdb.IsNotFound(txnErr) {
                break
            }
            if txnErr != nil {
                return txnErr
            }

            if !bytes.HasPrefix(key, prefix) {
                if len(results) > 0 {
                    break  // Past prefix range
                }
                continue
            }

            keyCopy := make([]byte, len(key))
            valCopy := make([]byte, len(val))
            copy(keyCopy, key)
            copy(valCopy, val)

            results = append(results, [2][]byte{keyCopy, valCopy})
        }

        return nil
    })

    return results, err
}

func (s *LMDBStore) Close() error {
    s.env.CloseDBI(s.dbi)
    s.env.Close()
    return nil
}

// Usage
func main() {
    store, err := NewLMDBStore("./mydb.mdb", 1024*1024*1024)
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

    // Scan
    results, err := store.ScanPrefix([]byte("user:"))
    if err != nil {
        log.Fatal(err)
    }

    for _, kv := range results {
        fmt.Printf("%s: %s\n", kv[0], kv[1])
    }
}
```

## 8. Performance Optimization

### Configuration Tuning

```c
// Production-optimized configuration
int lmdb_open_optimized(MDB_env **env, const char *path) {
    int rc;

    rc = mdb_env_create(env);
    if (rc != 0) {
        return rc;
    }

    // Map size: Set to max expected database size
    // Can be larger than physical storage (sparse file)
    rc = mdb_env_set_mapsize(*env, 100ULL * 1024 * 1024 * 1024);  // 100GB
    if (rc != 0) {
        mdb_env_close(*env);
        return rc;
    }

    // Max readers: Number of concurrent read transactions
    rc = mdb_env_set_maxreaders(*env, 126);  // Default: 126
    if (rc != 0) {
        mdb_env_close(*env);
        return rc;
    }

    // Max DBs: Number of named databases
    rc = mdb_env_set_maxdbs(*env, 10);
    if (rc != 0) {
        mdb_env_close(*env);
        return rc;
    }

    // Open flags for performance
    unsigned int flags = 0;

    // MDB_FIXEDMAP: Use fixed address (better performance on some systems)
    flags |= MDB_FIXEDMAP;

    // MDB_NOSYNC: Don't fsync after commit (MUCH faster, but less durable)
    // USE WITH CAUTION in production
    // flags |= MDB_NOSYNC;

    // MDB_NOMETASYNC: Don't sync metadata (slightly less durable)
    // flags |= MDB_NOMETASYNC;

    // MDB_WRITEMAP: Use writable mmap (better write performance)
    // Cannot use with nested transactions
    flags |= MDB_WRITEMAP;

    // MDB_MAPASYNC: Async msync (with MDB_WRITEMAP)
    // flags |= MDB_MAPASYNC;

    // MDB_NOSUBDIR: DB is a file, not directory
    flags |= MDB_NOSUBDIR;

    rc = mdb_env_open(*env, path, flags, 0664);
    if (rc != 0) {
        mdb_env_close(*env);
        return rc;
    }

    return MDB_SUCCESS;
}
```

### Write Performance

```c
// Batch writes for maximum throughput
int bulk_write_optimized(LMDBStore *store, size_t num_records) {
    MDB_txn *txn;
    MDB_val key, value;
    int rc;

    // Begin write transaction
    rc = mdb_txn_begin(store->env, NULL, 0, &txn);
    if (rc != 0) {
        return rc;
    }

    char key_buf[64], value_buf[128];

    for (size_t i = 0; i < num_records; i++) {
        snprintf(key_buf, sizeof(key_buf), "key_%zu", i);
        snprintf(value_buf, sizeof(value_buf), "value_%zu", i);

        key.mv_data = key_buf;
        key.mv_size = strlen(key_buf);
        value.mv_data = value_buf;
        value.mv_size = strlen(value_buf);

        rc = mdb_put(txn, store->dbi, &key, &value, MDB_APPEND);  // MDB_APPEND for sorted keys
        if (rc != 0) {
            mdb_txn_abort(txn);
            return rc;
        }

        // Commit every 10000 records to avoid transaction too large
        if ((i + 1) % 10000 == 0) {
            rc = mdb_txn_commit(txn);
            if (rc != 0) {
                return rc;
            }

            // Begin new transaction
            rc = mdb_txn_begin(store->env, NULL, 0, &txn);
            if (rc != 0) {
                return rc;
            }
        }
    }

    // Commit remaining
    rc = mdb_txn_commit(txn);
    return rc;
}
```

### Read Performance

```c
// Read optimization: reuse transactions
void optimized_read_pattern(LMDBStore *store) {
    MDB_txn *txn;
    MDB_val key, value;

    // Single long-lived read transaction for multiple reads
    mdb_txn_begin(store->env, NULL, MDB_RDONLY, &txn);

    // Multiple reads in same transaction
    for (int i = 0; i < 1000; i++) {
        char key_buf[64];
        snprintf(key_buf, sizeof(key_buf), "key_%d", i);

        key.mv_data = key_buf;
        key.mv_size = strlen(key_buf);

        int rc = mdb_get(txn, store->dbi, &key, &value);
        if (rc == 0) {
            // Process value (it's valid until transaction ends)
            // NO COPY NEEDED - zero-copy read
        }
    }

    // Cleanup
    mdb_txn_abort(txn);

    // Note: For long-running read transactions, be aware of
    // preventing writers from reclaiming old pages
}
```

## 9. Backup and Recovery

### Online Backup (Hot Backup)

```c
#include <lmdb.h>
#include <sys/stat.h>
#include <fcntl.h>

// Copy database while it's running
int lmdb_hot_backup(MDB_env *env, const char *backup_path) {
    int rc;

    // Use mdb_env_copy2 for efficient backup
    // MDB_CP_COMPACT: Compact while copying (slower but smaller)
    rc = mdb_env_copy2(env, backup_path, MDB_CP_COMPACT);

    if (rc != 0) {
        fprintf(stderr, "Backup failed: %s\n", mdb_strerror(rc));
        return rc;
    }

    printf("Backup completed: %s\n", backup_path);
    return MDB_SUCCESS;
}

// Backup without compaction (faster)
int lmdb_fast_backup(MDB_env *env, const char *backup_path) {
    return mdb_env_copy(env, backup_path);
}
```

### Filesystem-Level Backup

```bash
#!/bin/bash
# Backup script using filesystem snapshots

DB_PATH="/data/mydb.mdb"
BACKUP_DIR="/backups"
DATE=$(date +%Y%m%d_%H%M%S)

# Option 1: Simple copy (database must be consistent)
cp "$DB_PATH" "$BACKUP_DIR/mydb_$DATE.mdb"

# Option 2: Using rsync
rsync -av "$DB_PATH" "$BACKUP_DIR/mydb_$DATE.mdb"

# Option 3: LVM snapshot (if on LVM)
# lvcreate -L 10G -s -n snap_mydb /dev/vg0/lv_data
# mount /dev/vg0/snap_mydb /mnt/snapshot
# cp /mnt/snapshot/mydb.mdb "$BACKUP_DIR/mydb_$DATE.mdb"
# umount /mnt/snapshot
# lvremove -f /dev/vg0/snap_mydb

# Retention: Keep last 7 days
find "$BACKUP_DIR" -name "mydb_*.mdb" -mtime +7 -delete
```

### Incremental Backup (via snapshots)

```c
// Export specific key range to file
int export_range(MDB_env *env, MDB_dbi dbi,
                const char *start_key, const char *end_key,
                const char *output_file) {
    MDB_txn *txn;
    MDB_cursor *cursor;
    MDB_val key, value;
    FILE *fp;
    int rc;

    fp = fopen(output_file, "wb");
    if (!fp) {
        return -1;
    }

    rc = mdb_txn_begin(env, NULL, MDB_RDONLY, &txn);
    if (rc != 0) {
        fclose(fp);
        return rc;
    }

    rc = mdb_cursor_open(txn, dbi, &cursor);
    if (rc != 0) {
        mdb_txn_abort(txn);
        fclose(fp);
        return rc;
    }

    // Position at start_key
    key.mv_data = (void *)start_key;
    key.mv_size = strlen(start_key);
    rc = mdb_cursor_get(cursor, &key, &value, MDB_SET_RANGE);

    while (rc == 0) {
        // Check if beyond end_key
        if (strcmp((char *)key.mv_data, end_key) > 0) {
            break;
        }

        // Write key length, key, value length, value
        uint32_t key_len = key.mv_size;
        uint32_t val_len = value.mv_size;

        fwrite(&key_len, sizeof(key_len), 1, fp);
        fwrite(key.mv_data, 1, key_len, fp);
        fwrite(&val_len, sizeof(val_len), 1, fp);
        fwrite(value.mv_data, 1, val_len, fp);

        rc = mdb_cursor_get(cursor, &key, &value, MDB_NEXT);
    }

    mdb_cursor_close(cursor);
    mdb_txn_abort(txn);
    fclose(fp);

    return MDB_SUCCESS;
}
```

### Recovery

```c
// Check and repair corrupted database
int lmdb_check_integrity(const char *db_path) {
    MDB_env *env;
    MDB_txn *txn;
    MDB_cursor *cursor;
    MDB_dbi dbi;
    MDB_val key, value;
    int rc;
    size_t count = 0;

    rc = mdb_env_create(&env);
    if (rc != 0) {
        return rc;
    }

    // Open read-only
    rc = mdb_env_open(env, db_path, MDB_RDONLY | MDB_NOSUBDIR, 0);
    if (rc != 0) {
        fprintf(stderr, "Cannot open database: %s\n", mdb_strerror(rc));
        mdb_env_close(env);
        return rc;
    }

    rc = mdb_txn_begin(env, NULL, MDB_RDONLY, &txn);
    if (rc != 0) {
        mdb_env_close(env);
        return rc;
    }

    rc = mdb_dbi_open(txn, NULL, 0, &dbi);
    if (rc != 0) {
        mdb_txn_abort(txn);
        mdb_env_close(env);
        return rc;
    }

    rc = mdb_cursor_open(txn, dbi, &cursor);
    if (rc != 0) {
        mdb_txn_abort(txn);
        mdb_env_close(env);
        return rc;
    }

    // Iterate all keys to verify integrity
    rc = mdb_cursor_get(cursor, &key, &value, MDB_FIRST);
    while (rc == 0) {
        count++;
        rc = mdb_cursor_get(cursor, &key, &value, MDB_NEXT);
    }

    if (rc == MDB_NOTFOUND) {
        printf("Database OK: %zu keys\n", count);
        rc = MDB_SUCCESS;
    } else {
        fprintf(stderr, "Database corrupted at key %zu: %s\n",
                count, mdb_strerror(rc));
    }

    mdb_cursor_close(cursor);
    mdb_txn_abort(txn);
    mdb_env_close(env);

    return rc;
}
```

## 10. Monitoring and Statistics

### Database Statistics

```c
#include <lmdb.h>

void print_lmdb_stats(MDB_env *env, MDB_dbi dbi) {
    MDB_txn *txn;
    MDB_stat stat;
    MDB_envinfo info;
    int rc;

    // Get environment info
    rc = mdb_env_info(env, &info);
    if (rc == 0) {
        printf("Environment Info:\n");
        printf("  Map size: %zu bytes (%.2f GB)\n",
               info.me_mapsize, info.me_mapsize / (1024.0 * 1024.0 * 1024.0));
        printf("  Last page number: %zu\n", info.me_last_pgno);
        printf("  Last transaction ID: %zu\n", info.me_last_txnid);
        printf("  Max readers: %u\n", info.me_maxreaders);
        printf("  Num readers: %u\n", info.me_numreaders);
    }

    // Get database statistics
    rc = mdb_txn_begin(env, NULL, MDB_RDONLY, &txn);
    if (rc != 0) {
        return;
    }

    rc = mdb_stat(txn, dbi, &stat);
    if (rc == 0) {
        printf("\nDatabase Statistics:\n");
        printf("  Page size: %u bytes\n", stat.ms_psize);
        printf("  B-tree depth: %u\n", stat.ms_depth);
        printf("  Branch pages: %zu\n", stat.ms_branch_pages);
        printf("  Leaf pages: %zu\n", stat.ms_leaf_pages);
        printf("  Overflow pages: %zu\n", stat.ms_overflow_pages);
        printf("  Entries: %zu\n", stat.ms_entries);

        size_t total_pages = stat.ms_branch_pages + stat.ms_leaf_pages + stat.ms_overflow_pages;
        size_t total_size = total_pages * stat.ms_psize;
        printf("  Total size: %zu bytes (%.2f MB)\n",
               total_size, total_size / (1024.0 * 1024.0));
    }

    mdb_txn_abort(txn);
}

// Monitor reader slots
void print_reader_info(MDB_env *env) {
    printf("\nReader Status:\n");

    // Check reader table
    int rc = mdb_reader_check(env, NULL);
    if (rc > 0) {
        printf("  Cleared %d stale readers\n", rc);
    }

    // List active readers
    mdb_reader_list(env, [](const char *msg, void *ctx) -> int {
        printf("  %s\n", msg);
        return 0;
    }, NULL);
}
```

### Performance Monitoring

```c
#include <time.h>

// Measure operation latency
void benchmark_operations(LMDBStore *store) {
    struct timespec start, end;
    double elapsed;

    // Write benchmark
    clock_gettime(CLOCK_MONOTONIC, &start);

    for (int i = 0; i < 10000; i++) {
        char key[32], value[128];
        snprintf(key, sizeof(key), "bench_key_%d", i);
        snprintf(value, sizeof(value), "bench_value_%d", i);
        lmdb_put(store, key, value);
    }

    clock_gettime(CLOCK_MONOTONIC, &end);
    elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    printf("Write: 10000 ops in %.3f sec (%.0f ops/sec)\n",
           elapsed, 10000.0 / elapsed);

    // Read benchmark
    clock_gettime(CLOCK_MONOTONIC, &start);

    for (int i = 0; i < 10000; i++) {
        char key[32];
        char *value;
        size_t value_len;
        snprintf(key, sizeof(key), "bench_key_%d", i);
        lmdb_get(store, key, &value, &value_len);
        free(value);
    }

    clock_gettime(CLOCK_MONOTONIC, &end);
    elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    printf("Read: 10000 ops in %.3f sec (%.0f ops/sec)\n",
           elapsed, 10000.0 / elapsed);
}
```

## 11. Security Best Practices

### File Permissions

```c
#include <sys/stat.h>

// Open database with secure permissions
int lmdb_open_secure(MDB_env **env, const char *path) {
    int rc;

    rc = mdb_env_create(env);
    if (rc != 0) {
        return rc;
    }

    rc = mdb_env_set_mapsize(*env, 10ULL * 1024 * 1024 * 1024);
    if (rc != 0) {
        mdb_env_close(*env);
        return rc;
    }

    // Open with restrictive permissions (0600 = rw-------)
    rc = mdb_env_open(*env, path, MDB_FIXEDMAP | MDB_NOSUBDIR, 0600);
    if (rc != 0) {
        mdb_env_close(*env);
        return rc;
    }

    // Set file permissions after creation
    chmod(path, 0600);
    chmod((std::string(path) + "-lock").c_str(), 0600);

    return MDB_SUCCESS;
}
```

### Input Validation

```c
#include <limits.h>

#define MAX_KEY_SIZE 511    // LMDB limit
#define MAX_VALUE_SIZE (1024 * 1024 * 100)  // 100MB limit

int validate_key(const char *key, size_t key_len) {
    if (key == NULL || key_len == 0 || key_len > MAX_KEY_SIZE) {
        return -1;
    }
    return 0;
}

int validate_value(const char *value, size_t value_len) {
    if (value == NULL || value_len > MAX_VALUE_SIZE) {
        return -1;
    }
    return 0;
}

// Secure put with validation
int lmdb_put_secure(LMDBStore *store,
                   const char *key, size_t key_len,
                   const char *value, size_t value_len) {
    if (validate_key(key, key_len) != 0) {
        return -1;
    }

    if (validate_value(value, value_len) != 0) {
        return -1;
    }

    MDB_txn *txn;
    MDB_val mdb_key, mdb_value;
    int rc;

    rc = mdb_txn_begin(store->env, NULL, 0, &txn);
    if (rc != 0) {
        return rc;
    }

    mdb_key.mv_data = (void *)key;
    mdb_key.mv_size = key_len;
    mdb_value.mv_data = (void *)value;
    mdb_value.mv_size = value_len;

    rc = mdb_put(txn, store->dbi, &mdb_key, &mdb_value, 0);
    if (rc != 0) {
        mdb_txn_abort(txn);
        return rc;
    }

    rc = mdb_txn_commit(txn);
    return rc;
}
```

### Encryption at Rest

LMDB doesn't have built-in encryption. Use:

1. **Filesystem-level encryption**: LUKS, dm-crypt, eCryptfs
2. **Application-level encryption**: Encrypt values before storing

```c
// Example: Application-level encryption (pseudo-code)
#include <openssl/evp.h>

// Encrypt value before storing
int put_encrypted(LMDBStore *store, const char *key, const char *plaintext) {
    // Encrypt plaintext
    unsigned char *ciphertext = encrypt_aes256(plaintext, encryption_key);

    // Store encrypted data
    int rc = lmdb_put(store, key, (char *)ciphertext);

    free(ciphertext);
    return rc;
}

// Decrypt value after reading
int get_decrypted(LMDBStore *store, const char *key, char **plaintext) {
    char *ciphertext;
    size_t len;

    // Read encrypted data
    int rc = lmdb_get(store, key, &ciphertext, &len);
    if (rc != 0) {
        return rc;
    }

    // Decrypt
    *plaintext = decrypt_aes256((unsigned char *)ciphertext, encryption_key);

    free(ciphertext);
    return 0;
}
```

## 12. Common Patterns and Anti-Patterns

### Pattern: Secondary Indexing

```c
// Multi-database secondary index
typedef struct {
    MDB_env *env;
    MDB_dbi primary_dbi;
    MDB_dbi email_index_dbi;
} IndexedStore;

int indexed_store_init(IndexedStore *store, const char *path) {
    int rc;

    rc = mdb_env_create(&store->env);
    if (rc != 0) return rc;

    rc = mdb_env_set_mapsize(store->env, 1ULL * 1024 * 1024 * 1024);
    if (rc != 0) {
        mdb_env_close(store->env);
        return rc;
    }

    rc = mdb_env_set_maxdbs(store->env, 10);
    if (rc != 0) {
        mdb_env_close(store->env);
        return rc;
    }

    rc = mdb_env_open(store->env, path, MDB_FIXEDMAP | MDB_NOSUBDIR, 0664);
    if (rc != 0) {
        mdb_env_close(store->env);
        return rc;
    }

    // Open primary database
    MDB_txn *txn;
    rc = mdb_txn_begin(store->env, NULL, 0, &txn);
    if (rc != 0) {
        mdb_env_close(store->env);
        return rc;
    }

    rc = mdb_dbi_open(txn, "users", MDB_CREATE, &store->primary_dbi);
    if (rc != 0) {
        mdb_txn_abort(txn);
        mdb_env_close(store->env);
        return rc;
    }

    // Open email index
    rc = mdb_dbi_open(txn, "email_index", MDB_CREATE, &store->email_index_dbi);
    if (rc != 0) {
        mdb_txn_abort(txn);
        mdb_env_close(store->env);
        return rc;
    }

    rc = mdb_txn_commit(txn);
    return rc;
}

// Add user with email index
int add_user_indexed(IndexedStore *store,
                    const char *user_id,
                    const char *email,
                    const char *data) {
    MDB_txn *txn;
    MDB_val key, value;
    int rc;

    rc = mdb_txn_begin(store->env, NULL, 0, &txn);
    if (rc != 0) return rc;

    // Store in primary database
    key.mv_data = (void *)user_id;
    key.mv_size = strlen(user_id);
    value.mv_data = (void *)data;
    value.mv_size = strlen(data);

    rc = mdb_put(txn, store->primary_dbi, &key, &value, 0);
    if (rc != 0) {
        mdb_txn_abort(txn);
        return rc;
    }

    // Store in email index
    key.mv_data = (void *)email;
    key.mv_size = strlen(email);
    value.mv_data = (void *)user_id;
    value.mv_size = strlen(user_id);

    rc = mdb_put(txn, store->email_index_dbi, &key, &value, 0);
    if (rc != 0) {
        mdb_txn_abort(txn);
        return rc;
    }

    rc = mdb_txn_commit(txn);
    return rc;
}

// Lookup by email
int get_user_by_email(IndexedStore *store,
                     const char *email,
                     char **data) {
    MDB_txn *txn;
    MDB_val key, value;
    int rc;

    rc = mdb_txn_begin(store->env, NULL, MDB_RDONLY, &txn);
    if (rc != 0) return rc;

    // Lookup user_id from email index
    key.mv_data = (void *)email;
    key.mv_size = strlen(email);

    rc = mdb_get(txn, store->email_index_dbi, &key, &value);
    if (rc != 0) {
        mdb_txn_abort(txn);
        return rc;
    }

    // Get user data from primary
    MDB_val user_id_key;
    user_id_key.mv_data = value.mv_data;
    user_id_key.mv_size = value.mv_size;

    rc = mdb_get(txn, store->primary_dbi, &user_id_key, &value);
    if (rc == 0) {
        *data = malloc(value.mv_size + 1);
        memcpy(*data, value.mv_data, value.mv_size);
        (*data)[value.mv_size] = '\0';
    }

    mdb_txn_abort(txn);
    return rc;
}
```

### Anti-Pattern: Long-Lived Read Transactions

```c
// ❌ BAD: Long-lived read transaction prevents page reclamation
void bad_pattern(MDB_env *env, MDB_dbi dbi) {
    MDB_txn *txn;
    mdb_txn_begin(env, NULL, MDB_RDONLY, &txn);

    // Do lots of work for hours..
    sleep(3600);  // Blocks writers from reclaiming old pages!

    mdb_txn_abort(txn);
}

// ✅ GOOD: Short-lived transactions
void good_pattern(MDB_env *env, MDB_dbi dbi) {
    // Start transaction
    MDB_txn *txn;
    mdb_txn_begin(env, NULL, MDB_RDONLY, &txn);

    // Quick read
    MDB_val key, value;
    key.mv_data = "mykey";
    key.mv_size = strlen("mykey");
    mdb_get(txn, dbi, &key, &value);

    // Copy data if needed for later use
    char *data_copy = malloc(value.mv_size);
    memcpy(data_copy, value.mv_data, value.mv_size);

    // End transaction quickly
    mdb_txn_abort(txn);

    // Process data outside transaction
    process_data(data_copy);
    free(data_copy);
}
```

### Pattern: Append-Only Workload Optimization

```c
// Use MDB_APPEND for sorted inserts (much faster)
int append_sorted_keys(MDB_env *env, MDB_dbi dbi) {
    MDB_txn *txn;
    MDB_val key, value;
    int rc;

    rc = mdb_txn_begin(env, NULL, 0, &txn);
    if (rc != 0) return rc;

    // Keys must be in sorted order for MDB_APPEND
    for (int i = 0; i < 100000; i++) {
        char key_buf[32], value_buf[128];
        snprintf(key_buf, sizeof(key_buf), "key_%08d", i);  // Zero-padded
        snprintf(value_buf, sizeof(value_buf), "value_%d", i);

        key.mv_data = key_buf;
        key.mv_size = strlen(key_buf);
        value.mv_data = value_buf;
        value.mv_size = strlen(value_buf);

        // MDB_APPEND: Much faster for sorted keys (no B-tree rebalancing)
        rc = mdb_put(txn, dbi, &key, &value, MDB_APPEND);
        if (rc != 0) {
            mdb_txn_abort(txn);
            return rc;
        }
    }

    rc = mdb_txn_commit(txn);
    return rc;
}
```

## 13. Production Deployment

### Docker Deployment

```dockerfile
# Dockerfile
FROM ubuntu:22.04

RUN apt-get update && apt-get install -y \
    liblmdb-dev \
    lmdb-utils \
    && rm -rf /var/lib/apt/lists/*

# Copy application
COPY myapp /usr/local/bin/

# Data volume
VOLUME ["/data"]

# Run as non-root
RUN useradd -m -u 1000 lmdb
USER lmdb

CMD ["/usr/local/bin/myapp", "--db=/data/mydb.mdb"]
```

### System Configuration

```bash
# /etc/sysctl.conf
# Increase memory-mapped limits
vm.max_map_count = 1048576

# Optimize for database workload
vm.swappiness = 10
vm.dirty_ratio = 40
vm.dirty_background_ratio = 10

# File handles
fs.file-max = 2097152

# Apply settings
sudo sysctl -p
```

### Production Configuration

```c
// Production-ready configuration
MDB_env *open_production_lmdb(const char *path) {
    MDB_env *env;
    int rc;

    rc = mdb_env_create(&env);
    if (rc != 0) {
        return NULL;
    }

    // Large map size for growth
    rc = mdb_env_set_mapsize(env, 100ULL * 1024 * 1024 * 1024);  // 100GB
    if (rc != 0) {
        mdb_env_close(env);
        return NULL;
    }

    // Sufficient readers
    rc = mdb_env_set_maxreaders(env, 126);
    if (rc != 0) {
        mdb_env_close(env);
        return NULL;
    }

    // Multiple databases
    rc = mdb_env_set_maxdbs(env, 20);
    if (rc != 0) {
        mdb_env_close(env);
        return NULL;
    }

    // Production flags:
    // - MDB_FIXEDMAP: Better performance
    // - MDB_NOSUBDIR: Database is a file
    // - MDB_WRITEMAP: Better write performance
    // - NOT using MDB_NOSYNC: Need durability
    unsigned int flags = MDB_FIXEDMAP | MDB_NOSUBDIR | MDB_WRITEMAP;

    rc = mdb_env_open(env, path, flags, 0664);
    if (rc != 0) {
        mdb_env_close(env);
        return NULL;
    }

    return env;
}
```

## 14. Troubleshooting Guide

### Common Issues

```c
// Issue: MDB_MAP_FULL - Database file is full
// Solution: Increase map size

int handle_map_full(MDB_env *env) {
    // Close environment
    mdb_env_close(env);

    // Reopen with larger map size
    mdb_env_create(&env);
    mdb_env_set_mapsize(env, 200ULL * 1024 * 1024 * 1024);  // Double size
    mdb_env_open(env, "mydb.mdb", MDB_FIXEDMAP | MDB_NOSUBDIR, 0664);

    return 0;
}

// Issue: MDB_READERS_FULL - Too many concurrent readers
// Solution: Increase max readers or clean stale readers

int handle_readers_full(MDB_env *env) {
    // Check for stale readers
    int stale;
    mdb_reader_check(env, &stale);
    printf("Cleared %d stale readers\n", stale);

    // Or increase max readers
    mdb_env_close(env);
    mdb_env_create(&env);
    mdb_env_set_maxreaders(env, 256);  // Increase limit
    mdb_env_open(env, "mydb.mdb", MDB_FIXEDMAP | MDB_NOSUBDIR, 0664);

    return 0;
}

// Issue: MDB_TXN_FULL - Transaction too large
// Solution: Break into smaller transactions

int handle_txn_full(MDB_env *env, MDB_dbi dbi) {
    // Commit current transaction
    // Begin new transaction
    // Continue operations

    return 0;
}

// Issue: Stale lock file
// Solution: Remove lock file if no processes are using database

void cleanup_stale_lock(const char *db_path) {
    char lock_path[512];
    snprintf(lock_path, sizeof(lock_path), "%s-lock", db_path);

    // Check if any process is using the database
    // If not, remove lock file
    unlink(lock_path);

    printf("Removed stale lock file: %s\n", lock_path);
}
```

### Debugging Tools

```bash
# mdb_stat: View database statistics
mdb_stat -a mydb.mdb

# mdb_dump: Export database to text
mdb_dump mydb.mdb > backup.txt

# mdb_load: Import database from text
mdb_load mydb.mdb < backup.txt

# mdb_copy: Copy/compact database
mdb_copy -c mydb.mdb mydb_compacted.mdb
```

## 15. Performance Tuning Checklist

```markdown
**Hardware:**
- [ ] Use SSD or NVMe storage
- [ ] Sufficient RAM (map size can exceed RAM)
- [ ] Fast CPU for B-tree operations

**Configuration:**
- [ ] Set map size larger than expected database size
- [ ] Use MDB_WRITEMAP for better write performance
- [ ] Use MDB_MAPASYNC with MDB_WRITEMAP (with caution)
- [ ] Increase max readers if needed
- [ ] Use MDB_APPEND for sorted inserts

**Application:**
- [ ] Keep read transactions short
- [ ] Batch writes in single transaction
- [ ] Reuse read transactions when possible
- [ ] Use cursors for iteration
- [ ] Avoid frequent environment reopening

**System:**
- [ ] Increase vm.max_map_count
- [ ] Optimize vm.dirty_ratio
- [ ] Sufficient file descriptors
- [ ] Disable swapping for performance

**Monitoring:**
- [ ] Monitor database size vs map size
- [ ] Check for stale readers
- [ ] Monitor transaction latency
- [ ] Track page utilization
```

## 16. Comparison with Alternatives

### LMDB vs LevelDB/RocksDB

| Feature | LMDB | LevelDB/RocksDB |
|---------|------|-----------------|
| **Architecture** | B+ tree | LSM-tree |
| **Read Performance** | Faster | Good |
| **Write Performance** | Good | Faster |
| **Write Amplification** | Low (1-2x) | High (10-30x) |
| **Space Amplification** | Low | Higher |
| **Compaction** | Not needed | Required |
| **Memory Usage** | Depends on map size | Configurable cache |
| **Durability** | ACID, sync options | WAL with sync options |
| **Best For** | Read-heavy, memory-mapped | Write-heavy, SSD |

### LMDB vs Berkeley DB

| Feature | LMDB | Berkeley DB |
|---------|------|-------------|
| **API Complexity** | Simple | Complex |
| **Data Structures** | Key-value only | Multiple (B-tree, Hash, Queue) |
| **Performance** | Faster reads | Good overall |
| **Memory Model** | Memory-mapped | Buffer cache |
| **Maintenance** | None | Minimal |
| **Replication** | No | Yes |

## 17. Migration Strategies

### From LevelDB to LMDB

```c
// Export LevelDB data and import into LMDB
#include <leveldb/db.h>
#include <lmdb.h>

int migrate_leveldb_to_lmdb(const char *leveldb_path,
                            const char *lmdb_path) {
    // Open LevelDB
    leveldb::DB *leveldb;
    leveldb::Options level_opts;
    level_opts.create_if_missing = false;

    leveldb::Status status = leveldb::DB::Open(level_opts, leveldb_path, &leveldb);
    if (!status.ok()) {
        return -1;
    }

    // Open LMDB
    MDB_env *env;
    MDB_dbi dbi;
    MDB_txn *txn;

    mdb_env_create(&env);
    mdb_env_set_mapsize(env, 10ULL * 1024 * 1024 * 1024);
    mdb_env_open(env, lmdb_path, MDB_FIXEDMAP | MDB_NOSUBDIR, 0664);

    mdb_txn_begin(env, NULL, 0, &txn);
    mdb_dbi_open(txn, NULL, MDB_CREATE, &dbi);

    // Migrate data
    leveldb::Iterator *it = leveldb->NewIterator(leveldb::ReadOptions());
    int count = 0;

    for (it->SeekToFirst(); it->Valid(); it->Next()) {
        MDB_val key, value;

        key.mv_data = (void *)it->key().data();
        key.mv_size = it->key().size();
        value.mv_data = (void *)it->value().data();
        value.mv_size = it->value().size();

        mdb_put(txn, dbi, &key, &value, MDB_APPEND);
        count++;

        if (count % 10000 == 0) {
            mdb_txn_commit(txn);
            mdb_txn_begin(env, NULL, 0, &txn);
            printf("Migrated %d records...\n", count);
        }
    }

    mdb_txn_commit(txn);
    delete it;
    delete leveldb;
    mdb_env_close(env);

    printf("Migration complete: %d records\n", count);
    return 0;
}
```

## 18. Resources and References

### Official Documentation
- **LMDB Homepage**: http://www.lmdb.tech/doc/
- **Source Code**: https://git.openldap.org/openldap/openldap/-/tree/mdb.master/libraries/liblmdb
- **Technical Paper**: "LMDB: The Fastest Database You've Never Heard Of"

### Language Bindings
- **Python**: lmdb - https://lmdb.readthedocs.io/
- **Go**: lmdb-go - https://github.com/bmatsuo/lmdb-go
- **Rust**: lmdb-rs - https://github.com/danburkert/lmdb-rs
- **Node.js**: node-lmdb - https://github.com/Venemo/node-lmdb
- **Java**: lmdbjava - https://github.com/lmdbjava/lmdbjava

### Tools
- **mdb_stat**: Database statistics
- **mdb_dump**: Export to text format
- **mdb_load**: Import from text format
- **mdb_copy**: Copy/compact database

### Production Users
- **OpenLDAP**: Main database backend
- **Bitcoin Core**: Used for chainstate (migrated from LevelDB)
- **Knot DNS**: DNS server storage
- **Postfix**: Mail server caching
- **Samba**: Active Directory storage

### Benchmarks and Comparisons
- **Symas Benchmarks**: http://www.lmdb.tech/bench/microbench/
- **Comparison with other databases**: Shows LMDB's superior read performance

---

## Quick Start Example

```c
#include <lmdb.h>
#include <stdio.h>
#include <string.h>

int main() {
    MDB_env *env;
    MDB_dbi dbi;
    MDB_txn *txn;
    MDB_val key, value;
    int rc;

    // Create environment
    rc = mdb_env_create(&env);
    if (rc != 0) {
        fprintf(stderr, "mdb_env_create: %s\n", mdb_strerror(rc));
        return 1;
    }

    // Set map size
    rc = mdb_env_set_mapsize(env, 1ULL * 1024 * 1024 * 1024);  // 1GB
    if (rc != 0) {
        fprintf(stderr, "mdb_env_set_mapsize: %s\n", mdb_strerror(rc));
        return 1;
    }

    // Open environment
    rc = mdb_env_open(env, "./testdb.mdb", MDB_FIXEDMAP | MDB_NOSUBDIR, 0664);
    if (rc != 0) {
        fprintf(stderr, "mdb_env_open: %s\n", mdb_strerror(rc));
        return 1;
    }

    // Open database
    rc = mdb_txn_begin(env, NULL, 0, &txn);
    rc = mdb_dbi_open(txn, NULL, MDB_CREATE, &dbi);
    rc = mdb_txn_commit(txn);

    // Write
    rc = mdb_txn_begin(env, NULL, 0, &txn);
    key.mv_data = "key1";
    key.mv_size = strlen("key1");
    value.mv_data = "value1";
    value.mv_size = strlen("value1");
    rc = mdb_put(txn, dbi, &key, &value, 0);
    rc = mdb_txn_commit(txn);
    printf("Wrote key1=value1\n");

    // Read
    rc = mdb_txn_begin(env, NULL, MDB_RDONLY, &txn);
    key.mv_data = "key1";
    key.mv_size = strlen("key1");
    rc = mdb_get(txn, dbi, &key, &value);
    if (rc == 0) {
        printf("Read key1=%.*s\n", (int)value.mv_size, (char *)value.mv_data);
    }
    mdb_txn_abort(txn);

    // Cleanup
    mdb_dbi_close(env, dbi);
    mdb_env_close(env);
    return 0;
}
```

Compile with:
```bash
gcc -O3 example.c -llmdb -o example
./example
```

This guide provides comprehensive coverage of LMDB for production use, emphasizing its unique memory-mapped architecture and exceptional read performance.

---

**End of LMDB Development Guidelines**
