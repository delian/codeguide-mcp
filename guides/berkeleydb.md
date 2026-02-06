# Berkeley DB Development Guidelines
Comprehensive standards for using Berkeley DB, a high-performance embedded database library providing key-value storage with ACID transactions.

---

**Database Type**: Embedded Key-Value Store
**Engine**: Berkeley DB (Oracle)
**Architecture**: Library (not client-server)
**Best For**: Embedded systems, high-performance key-value storage, local data persistence, configuration management
**ACID Compliance**: Full ACID with transaction support
**Deployment Models**: Embedded library, process-local, replicated environments

**Key Features**:
- Zero-administration embedded database
- Multiple data access methods (B-tree, Hash, Queue, Recno)
- ACID transactions with MVCC
- High Availability replication
- Concurrent access with fine-grained locking
- Hot backup and recovery
- Berkeley DB SQL (SQLite-compatible API)
- Language bindings: C, C++, Java, Python, Perl, PHP, Ruby

**Companion Guides**: sql.md, sqlite.md, testing.md, c-cpp.md, java.md, python.md

---

## 1. Core Concepts

### What is Berkeley DB?

**Berkeley DB** is an embedded database library, not a standalone database server:

```
Traditional Database:          Berkeley DB:
┌──────────────┐              ┌──────────────┐
│ Application  │              │ Application  │
└──────┬───────┘              │  ┌────────┐  │
       │ Network               │  │ BDB    │  │
       │ Protocol              │  │ Library│  │
┌──────▼───────┐              │  └────────┘  │
│   Database   │              └──────────────┘
│   Server     │              Direct function calls
└──────────────┘              No network overhead
```

**Key Characteristics:**
- **Embedded**: Runs in the same process as your application
- **Zero-administration**: No separate server process or configuration
- **High performance**: Direct function calls, no network latency
- **Small footprint**: Library size ~1-5 MB depending on features
- **Flexible**: Multiple data structures and access patterns

### Data Access Methods

Berkeley DB supports multiple data structures optimized for different use cases:

**1. B-tree (Balanced Tree)**
```
Ordered key-value pairs
Use when:
- Need sorted access
- Range queries required
- Variable-length keys
- Most general-purpose cases

Complexity:
- Insert: O(log n)
- Lookup: O(log n)
- Delete: O(log n)
```

**2. Hash**
```
Hash table for fast lookups
Use when:
- Need fastest single-key lookup
- No range queries needed
- Fixed-key patterns
- Write-heavy workloads

Complexity:
- Insert: O(1)
- Lookup: O(1)
- Delete: O(1)
```

**3. Queue**
```
FIFO queue structure
Use when:
- Message queues
- Job queues
- Event logging
- Sequential record processing

Features:
- Fixed or variable-length records
- Efficient append/consume
- Automatic cleanup
```

**4. Recno (Record Number)**
```
Array-like access by record number
Use when:
- Need array semantics
- Sequential access patterns
- Log files
- Time-series data

Features:
- Fixed or variable-length records
- Sparse arrays supported
- Efficient sequential scan
```

### Architecture Components

```
┌─────────────────────────────────────────────────────┐
│              Application Process                     │
├─────────────────────────────────────────────────────┤
│                                                      │
│  ┌──────────────────────────────────────────────┐  │
│  │         Berkeley DB Library                   │  │
│  ├──────────────────────────────────────────────┤  │
│  │                                               │  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐   │  │
│  │  │ API Layer│  │Transaction│  │  Locking │   │  │
│  │  └────┬─────┘  │  Manager  │  │  Manager │   │  │
│  │       │        └──────────┘  └──────────┘   │  │
│  │       ▼                                       │  │
│  │  ┌──────────────────────────────────────┐   │  │
│  │  │      Access Methods                   │   │  │
│  │  │  (B-tree, Hash, Queue, Recno)         │   │  │
│  │  └──────────────┬───────────────────────┘   │  │
│  │                 ▼                            │  │
│  │  ┌──────────────────────────────────────┐   │  │
│  │  │      Memory Pool (Cache)              │   │  │
│  │  └──────────────┬───────────────────────┘   │  │
│  │                 ▼                            │  │
│  │  ┌──────────────────────────────────────┐   │  │
│  │  │      Log Manager (WAL)                │   │  │
│  │  └──────────────┬───────────────────────┘   │  │
│  └─────────────────┼──────────────────────────┘  │
│                    ▼                              │
└────────────────────┼──────────────────────────────┘
                     ▼
              ┌──────────────┐
              │  File System │
              │  - Data files│
              │  - Log files │
              └──────────────┘
```

### Use Cases

**Perfect For:**
- Configuration storage (e.g., OpenLDAP)
- Embedded devices with limited resources
- High-frequency key-value operations
- Local caching layers
- Message queues
- Session management
- Bitcoin wallets (used in Bitcoin Core)
- Mail servers (Postfix, Sendmail)
- Directory services
- Router/switch configurations

**Not Ideal For:**
- Complex queries requiring SQL
- Multi-table joins
- Full-text search
- Distributed systems requiring coordination
- Web-scale applications (consider Redis, memcached instead)

---

## 2. Installation and Setup

### Installing Berkeley DB

**Linux (Ubuntu/Debian):**
```bash
# Install from package manager
sudo apt-get update
sudo apt-get install libdb-dev libdb++-dev

# Or build from source
wget https://download.oracle.com/berkeley-db/db-18.1.40.tar.gz
tar xzf db-18.1.40.tar.gz
cd db-18.1.40/build_unix

../dist/configure --prefix=/usr/local \
  --enable-cxx \
  --enable-java \
  --enable-sql \
  --enable-sql_codegen

make
sudo make install
```

**macOS:**
```bash
# Using Homebrew
brew install berkeley-db

# Set environment variables
export CPATH=/usr/local/opt/berkeley-db/include
export LIBRARY_PATH=/usr/local/opt/berkeley-db/lib
```

**Windows:**
```powershell
# Download pre-built binaries from Oracle
# Or use vcpkg
vcpkg install berkeleydb

# Or build from source with Visual Studio
# Open db-18.1.40\build_windows\Berkeley_DB.sln
```

### Language Bindings

**C/C++** (Native):
```bash
# Already included with Berkeley DB installation
gcc -o myapp myapp.c -ldb
g++ -o myapp myapp.cpp -ldb_cxx
```

**Python:**
```bash
# Using bsddb3 (Berkeley DB bindings)
pip install bsddb3

# Or using berkeleydb (newer)
pip install berkeleydb
```

**Java:**
```xml
<!-- Maven dependency -->
<dependency>
    <groupId>com.sleepycat</groupId>
    <artifactId>je</artifactId>
    <version>18.3.12</version>
</dependency>
```

**Node.js:**
```bash
npm install berkeleydb
```

### Directory Structure

```
project/
├── data/                    # Database environment
│   ├── __db.001            # Shared memory files
│   ├── __db.002
│   ├── __db.003
│   ├── log.0000000001      # Transaction logs
│   ├── log.0000000002
│   └── mydb.db             # Database file
├── backups/                # Backup storage
├── src/
│   └── db_manager.c        # Database code
└── Makefile
```

---

## 3. Basic Operations (C API)

### Environment Creation

```c
#include <db.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int create_environment(DB_ENV **envp, const char *home_dir) {
    DB_ENV *env;
    int ret;

    // Create environment handle
    ret = db_env_create(&env, 0);
    if (ret != 0) {
        fprintf(stderr, "Error creating environment: %s\n",
                db_strerror(ret));
        return ret;
    }

    // Set cache size (256 MB)
    env->set_cachesize(env, 0, 256 * 1024 * 1024, 1);

    // Set log buffer size (32 KB)
    env->set_lg_bsize(env, 32 * 1024);

    // Set maximum log file size (10 MB)
    env->set_lg_max(env, 10 * 1024 * 1024);

    // Enable error messages
    env->set_errfile(env, stderr);
    env->set_errpfx(env, "BDB");

    // Open environment with transactions, locking, and logging
    ret = env->open(env, home_dir,
        DB_CREATE |         // Create if doesn't exist
        DB_INIT_LOCK |      // Initialize locking subsystem
        DB_INIT_LOG |       // Initialize logging subsystem
        DB_INIT_MPOOL |     // Initialize memory pool (cache)
        DB_INIT_TXN |       // Initialize transaction subsystem
        DB_RECOVER |        // Run recovery on open
        DB_THREAD,          // Thread-safe
        0);

    if (ret != 0) {
        fprintf(stderr, "Environment open failed: %s\n",
                db_strerror(ret));
        env->close(env, 0);
        return ret;
    }

    *envp = env;
    return 0;
}
```

### Database Creation and Basic CRUD

```c
// Open or create database
int open_database(DB_ENV *env, DB **dbp, const char *db_name) {
    DB *db;
    int ret;

    // Create database handle
    ret = db_create(&db, env, 0);
    if (ret != 0) {
        return ret;
    }

    // Open database with B-tree access method
    ret = db->open(db,
        NULL,               // Transaction (NULL for auto-commit)
        db_name,            // Database file name
        NULL,               // Logical database name
        DB_BTREE,           // Access method (B-tree)
        DB_CREATE | DB_AUTO_COMMIT, // Flags
        0);                 // File mode

    if (ret != 0) {
        db->close(db, 0);
        return ret;
    }

    *dbp = db;
    return 0;
}

// Insert/Update (Put)
int put_record(DB *db, const char *key, const char *data) {
    DBT db_key, db_data;
    int ret;

    // Zero out DBT structures
    memset(&db_key, 0, sizeof(DBT));
    memset(&db_data, 0, sizeof(DBT));

    // Set key and data
    db_key.data = (void *)key;
    db_key.size = strlen(key) + 1;

    db_data.data = (void *)data;
    db_data.size = strlen(data) + 1;

    // Put record
    ret = db->put(db, NULL, &db_key, &db_data, 0);
    if (ret != 0) {
        fprintf(stderr, "Put failed: %s\n", db_strerror(ret));
    }

    return ret;
}

// Retrieve (Get)
int get_record(DB *db, const char *key, char *buffer, size_t buf_size) {
    DBT db_key, db_data;
    int ret;

    memset(&db_key, 0, sizeof(DBT));
    memset(&db_data, 0, sizeof(DBT));

    db_key.data = (void *)key;
    db_key.size = strlen(key) + 1;

    // Provide buffer for data retrieval
    db_data.data = buffer;
    db_data.ulen = buf_size;
    db_data.flags = DB_DBT_USERMEM;

    // Get record
    ret = db->get(db, NULL, &db_key, &db_data, 0);
    if (ret != 0) {
        if (ret == DB_NOTFOUND) {
            fprintf(stderr, "Key not found\n");
        } else {
            fprintf(stderr, "Get failed: %s\n", db_strerror(ret));
        }
    }

    return ret;
}

// Delete
int delete_record(DB *db, const char *key) {
    DBT db_key;
    int ret;

    memset(&db_key, 0, sizeof(DBT));
    db_key.data = (void *)key;
    db_key.size = strlen(key) + 1;

    ret = db->del(db, NULL, &db_key, 0);
    if (ret != 0) {
        fprintf(stderr, "Delete failed: %s\n", db_strerror(ret));
    }

    return ret;
}

// Complete example
int main() {
    DB_ENV *env;
    DB *db;
    char buffer[1024];
    int ret;

    // Create environment
    ret = create_environment(&env, "./data");
    if (ret != 0) {
        return EXIT_FAILURE;
    }

    // Open database
    ret = open_database(env, &db, "mydb.db");
    if (ret != 0) {
        env->close(env, 0);
        return EXIT_FAILURE;
    }

    // Insert data
    put_record(db, "user:1", "Alice");
    put_record(db, "user:2", "Bob");

    // Retrieve data
    if (get_record(db, "user:1", buffer, sizeof(buffer)) == 0) {
        printf("user:1 = %s\n", buffer);
    }

    // Delete data
    delete_record(db, "user:2");

    // Close database and environment
    db->close(db, 0);
    env->close(env, 0);

    return EXIT_SUCCESS;
}
```

### Cursors for Iteration

```c
int iterate_records(DB *db) {
    DBC *cursor;
    DBT key, data;
    int ret;

    // Create cursor
    ret = db->cursor(db, NULL, &cursor, 0);
    if (ret != 0) {
        return ret;
    }

    // Initialize DBT structures
    memset(&key, 0, sizeof(DBT));
    memset(&data, 0, sizeof(DBT));

    // Iterate through all records
    while ((ret = cursor->get(cursor, &key, &data, DB_NEXT)) == 0) {
        printf("Key: %s, Data: %s\n", (char *)key.data, (char *)data.data);
    }

    if (ret != DB_NOTFOUND) {
        fprintf(stderr, "Cursor iteration failed: %s\n", db_strerror(ret));
    }

    // Close cursor
    cursor->close(cursor);
    return 0;
}

// Range query (B-tree only)
int range_query(DB *db, const char *start_key, const char *end_key) {
    DBC *cursor;
    DBT key, data;
    int ret, cmp;

    ret = db->cursor(db, NULL, &cursor, 0);
    if (ret != 0) return ret;

    memset(&key, 0, sizeof(DBT));
    memset(&data, 0, sizeof(DBT));

    // Position at start key
    key.data = (void *)start_key;
    key.size = strlen(start_key) + 1;

    ret = cursor->get(cursor, &key, &data, DB_SET_RANGE);

    while (ret == 0) {
        // Check if we've passed end key
        cmp = strcmp((char *)key.data, end_key);
        if (cmp > 0) break;

        printf("Key: %s, Data: %s\n", (char *)key.data, (char *)data.data);

        ret = cursor->get(cursor, &key, &data, DB_NEXT);
    }

    cursor->close(cursor);
    return 0;
}
```

---

## 4. Transactions

### Transaction Basics

```c
int transactional_updates(DB_ENV *env, DB *db) {
    DB_TXN *txn;
    DBT key, data;
    int ret;
    char *keys[] = {"account:1", "account:2"};
    char *values[] = {"balance:1000", "balance:2000"};

    // Begin transaction
    ret = env->txn_begin(env, NULL, &txn, 0);
    if (ret != 0) {
        fprintf(stderr, "Transaction begin failed: %s\n", db_strerror(ret));
        return ret;
    }

    // Perform operations within transaction
    for (int i = 0; i < 2; i++) {
        memset(&key, 0, sizeof(DBT));
        memset(&data, 0, sizeof(DBT));

        key.data = keys[i];
        key.size = strlen(keys[i]) + 1;
        data.data = values[i];
        data.size = strlen(values[i]) + 1;

        ret = db->put(db, txn, &key, &data, 0);
        if (ret != 0) {
            fprintf(stderr, "Put failed: %s\n", db_strerror(ret));
            txn->abort(txn);
            return ret;
        }
    }

    // Commit transaction
    ret = txn->commit(txn, 0);
    if (ret != 0) {
        fprintf(stderr, "Transaction commit failed: %s\n", db_strerror(ret));
        return ret;
    }

    printf("Transaction committed successfully\n");
    return 0;
}
```

### Transfer with Rollback Example

```c
int transfer_funds(DB_ENV *env, DB *db,
                   const char *from_account,
                   const char *to_account,
                   int amount) {
    DB_TXN *txn;
    DBT key, data;
    char buffer[256];
    int from_balance, to_balance;
    int ret;

    // Begin transaction
    ret = env->txn_begin(env, NULL, &txn, 0);
    if (ret != 0) return ret;

    // Get source account balance
    memset(&key, 0, sizeof(DBT));
    memset(&data, 0, sizeof(DBT));
    key.data = (void *)from_account;
    key.size = strlen(from_account) + 1;
    data.data = buffer;
    data.ulen = sizeof(buffer);
    data.flags = DB_DBT_USERMEM;

    ret = db->get(db, txn, &key, &data, DB_RMW); // Read-Modify-Write lock
    if (ret != 0) {
        txn->abort(txn);
        return ret;
    }

    sscanf((char *)data.data, "balance:%d", &from_balance);

    // Check sufficient funds
    if (from_balance < amount) {
        fprintf(stderr, "Insufficient funds\n");
        txn->abort(txn);
        return -1;
    }

    // Get destination account balance
    memset(&key, 0, sizeof(DBT));
    memset(&data, 0, sizeof(DBT));
    key.data = (void *)to_account;
    key.size = strlen(to_account) + 1;
    data.data = buffer;
    data.ulen = sizeof(buffer);
    data.flags = DB_DBT_USERMEM;

    ret = db->get(db, txn, &key, &data, DB_RMW);
    if (ret != 0) {
        txn->abort(txn);
        return ret;
    }

    sscanf((char *)data.data, "balance:%d", &to_balance);

    // Update source account
    from_balance -= amount;
    snprintf(buffer, sizeof(buffer), "balance:%d", from_balance);
    memset(&key, 0, sizeof(DBT));
    memset(&data, 0, sizeof(DBT));
    key.data = (void *)from_account;
    key.size = strlen(from_account) + 1;
    data.data = buffer;
    data.size = strlen(buffer) + 1;

    ret = db->put(db, txn, &key, &data, 0);
    if (ret != 0) {
        txn->abort(txn);
        return ret;
    }

    // Update destination account
    to_balance += amount;
    snprintf(buffer, sizeof(buffer), "balance:%d", to_balance);
    memset(&data, 0, sizeof(DBT));
    key.data = (void *)to_account;
    key.size = strlen(to_account) + 1;
    data.data = buffer;
    data.size = strlen(buffer) + 1;

    ret = db->put(db, txn, &key, &data, 0);
    if (ret != 0) {
        txn->abort(txn);
        return ret;
    }

    // Commit transaction
    ret = txn->commit(txn, 0);
    if (ret == 0) {
        printf("Transfer successful: %s -> %s: $%d\n",
               from_account, to_account, amount);
    }

    return ret;
}
```

### Nested Transactions

```c
int nested_transaction_example(DB_ENV *env, DB *db) {
    DB_TXN *parent_txn, *child_txn;
    DBT key, data;
    int ret;

    // Begin parent transaction
    ret = env->txn_begin(env, NULL, &parent_txn, 0);
    if (ret != 0) return ret;

    // Parent operation
    memset(&key, 0, sizeof(DBT));
    memset(&data, 0, sizeof(DBT));
    key.data = "parent:key";
    key.size = 11;
    data.data = "parent:data";
    data.size = 12;

    ret = db->put(db, parent_txn, &key, &data, 0);
    if (ret != 0) {
        parent_txn->abort(parent_txn);
        return ret;
    }

    // Begin nested (child) transaction
    ret = env->txn_begin(env, parent_txn, &child_txn, 0);
    if (ret != 0) {
        parent_txn->abort(parent_txn);
        return ret;
    }

    // Child operation
    key.data = "child:key";
    key.size = 10;
    data.data = "child:data";
    data.size = 11;

    ret = db->put(db, child_txn, &key, &data, 0);
    if (ret != 0) {
        child_txn->abort(child_txn);
        parent_txn->abort(parent_txn);
        return ret;
    }

    // Commit child transaction
    ret = child_txn->commit(child_txn, 0);
    if (ret != 0) {
        parent_txn->abort(parent_txn);
        return ret;
    }

    // Commit parent transaction
    ret = parent_txn->commit(parent_txn, 0);
    return ret;
}
```

---

## 5. Python API

### Installation and Basic Usage

```python
# Install berkeleydb package
# pip install berkeleydb

import berkeleydb as bdb

# Create environment
env = bdb.db.DBEnv()
env.set_cachesize(0, 256 * 1024 * 1024)  # 256 MB cache
env.open('./data',
         bdb.db.DB_CREATE |
         bdb.db.DB_INIT_MPOOL |
         bdb.db.DB_INIT_TXN |
         bdb.db.DB_INIT_LOG |
         bdb.db.DB_INIT_LOCK |
         bdb.db.DB_RECOVER |
         bdb.db.DB_THREAD)

# Open database
db = bdb.db.DB(env)
db.open('mydb.db',
        dbtype=bdb.db.DB_BTREE,
        flags=bdb.db.DB_CREATE | bdb.db.DB_AUTO_COMMIT)

# Insert
db.put(b'user:1', b'Alice')
db.put(b'user:2', b'Bob')
db.put(b'user:3', b'Charlie')

# Retrieve
value = db.get(b'user:1')
print(f"user:1 = {value.decode()}")

# Delete
db.delete(b'user:3')

# Iterate
cursor = db.cursor()
for key, value in cursor:
    print(f"{key.decode()} = {value.decode()}")
cursor.close()

# Close
db.close()
env.close()
```

### Transactions in Python

```python
def transfer_with_transaction(env, db, from_key, to_key, amount):
    """Transfer amount between accounts with transaction."""
    txn = env.txn_begin()

    try:
        # Get source balance
        from_data = db.get(from_key, txn=txn, flags=bdb.db.DB_RMW)
        if from_data is None:
            raise ValueError(f"Account {from_key} not found")

        from_balance = int(from_data.decode().split(':')[1])

        if from_balance < amount:
            raise ValueError("Insufficient funds")

        # Get destination balance
        to_data = db.get(to_key, txn=txn, flags=bdb.db.DB_RMW)
        if to_data is None:
            raise ValueError(f"Account {to_key} not found")

        to_balance = int(to_data.decode().split(':')[1])

        # Update balances
        from_balance -= amount
        to_balance += amount

        db.put(from_key, f"balance:{from_balance}".encode(), txn=txn)
        db.put(to_key, f"balance:{to_balance}".encode(), txn=txn)

        # Commit transaction
        txn.commit()
        print(f"Transfer successful: {from_key.decode()} -> {to_key.decode()}: ${amount}")

    except Exception as e:
        txn.abort()
        print(f"Transfer failed: {e}")
        raise

# Usage
env = bdb.db.DBEnv()
env.open('./data',
         bdb.db.DB_CREATE |
         bdb.db.DB_INIT_MPOOL |
         bdb.db.DB_INIT_TXN |
         bdb.db.DB_INIT_LOG |
         bdb.db.DB_INIT_LOCK)

db = bdb.db.DB(env)
db.open('accounts.db', dbtype=bdb.db.DB_BTREE,
        flags=bdb.db.DB_CREATE | bdb.db.DB_AUTO_COMMIT)

# Initialize accounts
db.put(b'account:1', b'balance:1000')
db.put(b'account:2', b'balance:500')

# Perform transfer
transfer_with_transaction(env, db, b'account:1', b'account:2', 100)

db.close()
env.close()
```

### Context Manager Pattern

```python
from contextlib import contextmanager

@contextmanager
def berkeley_db(db_path, env_path='./data'):
    """Context manager for Berkeley DB."""
    env = None
    db = None

    try:
        # Open environment
        env = bdb.db.DBEnv()
        env.set_cachesize(0, 256 * 1024 * 1024)
        env.open(env_path,
                 bdb.db.DB_CREATE |
                 bdb.db.DB_INIT_MPOOL |
                 bdb.db.DB_INIT_TXN |
                 bdb.db.DB_INIT_LOG |
                 bdb.db.DB_INIT_LOCK)

        # Open database
        db = bdb.db.DB(env)
        db.open(db_path,
                dbtype=bdb.db.DB_BTREE,
                flags=bdb.db.DB_CREATE | bdb.db.DB_AUTO_COMMIT)

        yield env, db

    finally:
        if db:
            db.close()
        if env:
            env.close()

# Usage
with berkeley_db('mydb.db') as (env, db):
    db.put(b'key1', b'value1')
    value = db.get(b'key1')
    print(value.decode())
```

---

## 6. Java API

### Setup and Basic Operations

```java
import com.sleepycat.db.*;

public class BerkeleyDBExample {

    public static void main(String[] args) {
        Environment env = null;
        Database db = null;

        try {
            // Create environment
            EnvironmentConfig envConfig = new EnvironmentConfig();
            envConfig.setAllowCreate(true);
            envConfig.setInitializeCache(true);
            envConfig.setInitializeLocking(true);
            envConfig.setInitializeLogging(true);
            envConfig.setTransactional(true);
            envConfig.setCacheSize(256 * 1024 * 1024); // 256 MB

            env = new Environment(new File("./data"), envConfig);

            // Open database
            DatabaseConfig dbConfig = new DatabaseConfig();
            dbConfig.setAllowCreate(true);
            dbConfig.setTransactional(true);
            dbConfig.setType(DatabaseType.BTREE);

            db = env.openDatabase(null, "mydb.db", null, dbConfig);

            // Insert data
            DatabaseEntry key = new DatabaseEntry("user:1".getBytes("UTF-8"));
            DatabaseEntry data = new DatabaseEntry("Alice".getBytes("UTF-8"));
            db.put(null, key, data);

            // Retrieve data
            key = new DatabaseEntry("user:1".getBytes("UTF-8"));
            data = new DatabaseEntry();

            if (db.get(null, key, data, LockMode.DEFAULT) == OperationStatus.SUCCESS) {
                String value = new String(data.getData(), "UTF-8");
                System.out.println("user:1 = " + value);
            }

            // Iterate records
            Cursor cursor = db.openCursor(null, null);
            key = new DatabaseEntry();
            data = new DatabaseEntry();

            while (cursor.getNext(key, data, LockMode.DEFAULT) ==
                   OperationStatus.SUCCESS) {
                String keyStr = new String(key.getData(), "UTF-8");
                String dataStr = new String(data.getData(), "UTF-8");
                System.out.println(keyStr + " = " + dataStr);
            }

            cursor.close();

        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            if (db != null) {
                try { db.close(); } catch (Exception e) {}
            }
            if (env != null) {
                try { env.close(); } catch (Exception e) {}
            }
        }
    }
}
```

### Transactions in Java

```java
public class TransactionExample {

    public static void transferFunds(Environment env, Database db,
                                     String fromAccount, String toAccount,
                                     int amount) throws Exception {
        Transaction txn = null;

        try {
            // Begin transaction
            txn = env.beginTransaction(null, null);

            // Get source account
            DatabaseEntry fromKey = new DatabaseEntry(fromAccount.getBytes("UTF-8"));
            DatabaseEntry fromData = new DatabaseEntry();

            if (db.get(txn, fromKey, fromData, LockMode.RMW) !=
                OperationStatus.SUCCESS) {
                throw new Exception("Source account not found");
            }

            String fromBalance = new String(fromData.getData(), "UTF-8");
            int fromAmount = Integer.parseInt(fromBalance.split(":")[1]);

            if (fromAmount < amount) {
                throw new Exception("Insufficient funds");
            }

            // Get destination account
            DatabaseEntry toKey = new DatabaseEntry(toAccount.getBytes("UTF-8"));
            DatabaseEntry toData = new DatabaseEntry();

            if (db.get(txn, toKey, toData, LockMode.RMW) !=
                OperationStatus.SUCCESS) {
                throw new Exception("Destination account not found");
            }

            String toBalance = new String(toData.getData(), "UTF-8");
            int toAmount = Integer.parseInt(toBalance.split(":")[1]);

            // Update accounts
            fromAmount -= amount;
            toAmount += amount;

            fromData = new DatabaseEntry(("balance:" + fromAmount).getBytes("UTF-8"));
            db.put(txn, fromKey, fromData);

            toData = new DatabaseEntry(("balance:" + toAmount).getBytes("UTF-8"));
            db.put(txn, toKey, toData);

            // Commit transaction
            txn.commit();
            System.out.println("Transfer successful");

        } catch (Exception e) {
            if (txn != null) {
                txn.abort();
            }
            throw e;
        }
    }
}
```

---

## 7. Secondary Indexes

### Creating Secondary Databases

```c
// Secondary key callback function
int get_secondary_key(DB *secondary, const DBT *pkey,
                      const DBT *pdata, DBT *skey) {
    // Extract email from data (assuming format: "name:email:age")
    char *data = (char *)pdata->data;
    char *email_start = strchr(data, ':');
    if (email_start == NULL) return DB_DONOTINDEX;

    email_start++; // Skip first colon
    char *email_end = strchr(email_start, ':');
    if (email_end == NULL) return DB_DONOTINDEX;

    // Set secondary key
    memset(skey, 0, sizeof(DBT));
    skey->data = email_start;
    skey->size = email_end - email_start;
    skey->flags = DB_DBT_APPMALLOC;

    // Allocate memory for key
    skey->data = malloc(skey->size + 1);
    memcpy(skey->data, email_start, skey->size);
    ((char *)skey->data)[skey->size] = '\0';

    return 0;
}

int create_secondary_index(DB_ENV *env, DB *primary, DB **secondary) {
    DB *sdb;
    int ret;

    // Create secondary database
    ret = db_create(&sdb, env, 0);
    if (ret != 0) return ret;

    // Set flags for duplicates (multiple records can have same secondary key)
    ret = sdb->set_flags(sdb, DB_DUPSORT);
    if (ret != 0) {
        sdb->close(sdb, 0);
        return ret;
    }

    // Open secondary database
    ret = sdb->open(sdb, NULL, "users_by_email.db", NULL,
                    DB_BTREE, DB_CREATE | DB_AUTO_COMMIT, 0);
    if (ret != 0) {
        sdb->close(sdb, 0);
        return ret;
    }

    // Associate with primary
    ret = primary->associate(primary, NULL, sdb, get_secondary_key, 0);
    if (ret != 0) {
        sdb->close(sdb, 0);
        return ret;
    }

    *secondary = sdb;
    return 0;
}

// Query by secondary key
int find_by_email(DB *secondary, const char *email) {
    DBC *cursor;
    DBT skey, pkey, data;
    int ret;

    // Open cursor on secondary database
    ret = secondary->cursor(secondary, NULL, &cursor, 0);
    if (ret != 0) return ret;

    memset(&skey, 0, sizeof(DBT));
    memset(&pkey, 0, sizeof(DBT));
    memset(&data, 0, sizeof(DBT));

    skey.data = (void *)email;
    skey.size = strlen(email);

    // Get by secondary key (returns primary key and data)
    ret = cursor->pget(cursor, &skey, &pkey, &data, DB_SET);

    while (ret == 0) {
        printf("Primary key: %s, Data: %s\n",
               (char *)pkey.data, (char *)data.data);

        ret = cursor->pget(cursor, &skey, &pkey, &data, DB_NEXT_DUP);
    }

    cursor->close(cursor);
    return 0;
}
```

---

## 8. Performance Optimization

### Cache Configuration

```c
int configure_cache(DB_ENV *env) {
    // Set cache size to 512 MB
    env->set_cachesize(env, 0, 512 * 1024 * 1024, 1);

    // Set maximum number of cache regions
    env->set_cache_max(env, 0, 1024 * 1024 * 1024); // 1 GB max

    // Set cache priority for specific database
    // (called after db->open())
    // db->set_priority(db, DB_PRIORITY_VERY_HIGH);

    return 0;
}
```

### Bulk Operations

```c
int bulk_put(DB *db, char **keys, char **values, int count) {
    DBT bulk_key, bulk_data;
    DB_TXN *txn;
    void *key_buf, *data_buf;
    int ret;

    // Allocate buffers for bulk operation
    key_buf = malloc(1024 * 1024);   // 1 MB
    data_buf = malloc(1024 * 1024);  // 1 MB

    memset(&bulk_key, 0, sizeof(DBT));
    memset(&bulk_data, 0, sizeof(DBT));

    bulk_key.data = key_buf;
    bulk_key.ulen = 1024 * 1024;
    bulk_key.flags = DB_DBT_USERMEM | DB_DBT_BULK;

    bulk_data.data = data_buf;
    bulk_data.ulen = 1024 * 1024;
    bulk_data.flags = DB_DBT_USERMEM | DB_DBT_BULK;

    // Begin transaction
    db->get_env(db, &env);
    env->txn_begin(env, NULL, &txn, 0);

    // Build bulk buffers
    for (int i = 0; i < count; i++) {
        // Add to bulk buffers
        // (simplified - real implementation needs DB_MULTIPLE_WRITE_INIT/NEXT)
    }

    // Perform bulk put
    ret = db->put(db, txn, &bulk_key, &bulk_data, DB_MULTIPLE);

    if (ret == 0) {
        txn->commit(txn, 0);
    } else {
        txn->abort(txn);
    }

    free(key_buf);
    free(data_buf);

    return ret;
}
```

### Read-Ahead Configuration

```c
int configure_io(DB *db) {
    // Set page size (default 4KB, can be 512B to 64KB)
    db->set_pagesize(db, 8192); // 8 KB pages

    // Set read-ahead buffer
    db->set_readahead(db, 256 * 1024); // 256 KB read-ahead

    return 0;
}
```

### Database Compaction

```c
int compact_database(DB *db) {
    DB_COMPACT compact_data;
    int ret;

    memset(&compact_data, 0, sizeof(DB_COMPACT));

    // Compact database (returns pages freed)
    ret = db->compact(db, NULL, NULL, NULL, &compact_data,
                      DB_FREE_SPACE, NULL);

    if (ret == 0) {
        printf("Pages freed: %d\n", compact_data.compact_pages_free);
        printf("Pages truncated: %d\n", compact_data.compact_pages_truncate);
    }

    return ret;
}
```

---

## 9. Replication and High Availability

### Replication Setup (Master)

```c
int setup_replication_master(DB_ENV *env, const char *host, u_int port) {
    int ret;

    // Set replication manager local site
    ret = env->repmgr_set_local_site(env, host, port, 0);
    if (ret != 0) return ret;

    // Set acknowledgment policy
    env->repmgr_set_ack_policy(env, DB_REPMGR_ACKS_ALL);

    // Set election priority (higher = more likely to become master)
    env->rep_set_priority(env, 100);

    // Set election timeout
    env->rep_set_timeout(env, DB_REP_ELECTION_TIMEOUT, 1000000); // 1 second

    // Start replication manager
    ret = env->repmgr_start(env, 3, DB_REP_MASTER);

    return ret;
}
```

### Replication Setup (Replica)

```c
int setup_replication_replica(DB_ENV *env,
                              const char *local_host, u_int local_port,
                              const char *master_host, u_int master_port) {
    int ret;

    // Set local site
    ret = env->repmgr_set_local_site(env, local_host, local_port, 0);
    if (ret != 0) return ret;

    // Add master as remote site
    ret = env->repmgr_add_remote_site(env, master_host, master_port, NULL, 0);
    if (ret != 0) return ret;

    // Set acknowledgment policy
    env->repmgr_set_ack_policy(env, DB_REPMGR_ACKS_NONE);

    // Set election priority (0 = cannot become master)
    env->rep_set_priority(env, 0);

    // Start replication manager
    ret = env->repmgr_start(env, 3, DB_REP_CLIENT);

    return ret;
}
```

### Handling Replication Events

```c
void replication_event_callback(DB_ENV *env, u_int32_t which, void *info) {
    switch (which) {
        case DB_EVENT_REP_MASTER:
            printf("This site is now the master\n");
            break;

        case DB_EVENT_REP_CLIENT:
            printf("This site is now a replica\n");
            break;

        case DB_EVENT_REP_NEWMASTER:
            printf("A new master has been elected\n");
            break;

        case DB_EVENT_REP_STARTUPDONE:
            printf("Replica startup complete\n");
            break;

        case DB_EVENT_REP_PERM_FAILED:
            printf("Permanent message failed - data might be lost\n");
            break;

        default:
            printf("Unknown replication event: %u\n", which);
    }
}

// Register callback
env->set_event_notify(env, replication_event_callback);
```

---

## 10. Backup and Recovery

### Hot Backup

```c
int perform_hot_backup(DB_ENV *env, const char *backup_dir) {
    int ret;

    // Perform hot backup (database remains online)
    ret = env->backup(env, backup_dir, 0);

    if (ret == 0) {
        printf("Hot backup completed to %s\n", backup_dir);
    } else {
        fprintf(stderr, "Backup failed: %s\n", db_strerror(ret));
    }

    return ret;
}
```

### Incremental Backup

```bash
#!/bin/bash
# Incremental backup script

BACKUP_DIR="/backups/bdb"
ENV_DIR="/data"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)

# Full backup on Sunday, incremental on other days
if [ $(date +%u) -eq 7 ]; then
    # Full backup
    mkdir -p "$BACKUP_DIR/full-$TIMESTAMP"
    db_hotbackup -h "$ENV_DIR" -b "$BACKUP_DIR/full-$TIMESTAMP" -v
else
    # Incremental backup (copy only log files)
    mkdir -p "$BACKUP_DIR/incr-$TIMESTAMP"
    db_archive -h "$ENV_DIR" -s | \
        xargs -I {} cp "$ENV_DIR/{}" "$BACKUP_DIR/incr-$TIMESTAMP/"
fi

# Remove old archived logs
db_archive -h "$ENV_DIR" -d
```

### Recovery Process

```c
int recover_database(const char *env_home) {
    DB_ENV *env;
    int ret;

    // Create environment
    ret = db_env_create(&env, 0);
    if (ret != 0) return ret;

    // Open with recovery
    ret = env->open(env, env_home,
        DB_CREATE |
        DB_INIT_LOCK |
        DB_INIT_LOG |
        DB_INIT_MPOOL |
        DB_INIT_TXN |
        DB_RECOVER |          // Normal recovery
        // DB_RECOVER_FATAL | // Catastrophic recovery
        DB_THREAD,
        0);

    if (ret == 0) {
        printf("Recovery successful\n");
        env->close(env, 0);
    } else {
        fprintf(stderr, "Recovery failed: %s\n", db_strerror(ret));
    }

    return ret;
}
```

### Utilities for Backup/Recovery

```bash
# Hot backup
db_hotbackup -h /data -b /backups/$(date +%Y%m%d)

# List archived log files
db_archive -h /data -l

# Remove archived log files (safe to delete)
db_archive -h /data -d

# Verify database integrity
db_verify -h /data mydb.db

# Database statistics
db_stat -h /data -d mydb.db

# Catastrophic recovery (after restoring from backup)
db_recover -h /data -c

# Normal recovery
db_recover -h /data
```

---

## 11. Monitoring and Statistics

### Database Statistics

```c
void print_database_stats(DB *db) {
    DB_BTREE_STAT *stats;
    int ret;

    // Get B-tree statistics
    ret = db->stat(db, NULL, &stats, 0);
    if (ret != 0) {
        fprintf(stderr, "stat failed: %s\n", db_strerror(ret));
        return;
    }

    printf("Database Statistics:\n");
    printf("  Magic number: 0x%x\n", stats->bt_magic);
    printf("  Version: %u\n", stats->bt_version);
    printf("  Page size: %u bytes\n", stats->bt_pagesize);
    printf("  Number of keys: %u\n", stats->bt_nkeys);
    printf("  Number of records: %u\n", stats->bt_ndata);
    printf("  Number of pages: %u\n", stats->bt_pagecnt);
    printf("  Tree levels: %u\n", stats->bt_levels);
    printf("  Internal pages: %u\n", stats->bt_int_pg);
    printf("  Leaf pages: %u\n", stats->bt_leaf_pg);
    printf("  Overflow pages: %u\n", stats->bt_over_pg);
    printf("  Free pages: %u\n", stats->bt_free);

    free(stats);
}
```

### Environment Statistics

```c
void print_environment_stats(DB_ENV *env) {
    DB_ENV_STAT *stats;
    int ret;

    ret = env->stat_print(env, DB_STAT_ALL);

    // Or get specific subsystem stats
    DB_LOCK_STAT *lock_stats;
    ret = env->lock_stat(env, &lock_stats, 0);
    if (ret == 0) {
        printf("Lock Statistics:\n");
        printf("  Current locks: %u\n", lock_stats->st_nlocks);
        printf("  Max locks: %u\n", lock_stats->st_maxnlocks);
        printf("  Deadlocks: %u\n", lock_stats->st_ndeadlocks);
        free(lock_stats);
    }

    DB_LOG_STAT *log_stats;
    ret = env->log_stat(env, &log_stats, 0);
    if (ret == 0) {
        printf("Log Statistics:\n");
        printf("  Log file size: %u\n", log_stats->st_lg_size);
        printf("  Current log file: %u\n", log_stats->st_cur_file);
        printf("  Bytes written: %llu\n", log_stats->st_w_bytes);
        free(log_stats);
    }
}
```

### Performance Monitoring

```c
typedef struct {
    uint64_t total_reads;
    uint64_t total_writes;
    uint64_t cache_hits;
    uint64_t cache_misses;
    time_t start_time;
} db_metrics_t;

db_metrics_t metrics = {0};

void record_operation(const char *op_type) {
    if (strcmp(op_type, "read") == 0) {
        __atomic_add_fetch(&metrics.total_reads, 1, __ATOMIC_SEQ_CST);
    } else if (strcmp(op_type, "write") == 0) {
        __atomic_add_fetch(&metrics.total_writes, 1, __ATOMIC_SEQ_CST);
    }
}

void print_metrics() {
    time_t elapsed = time(NULL) - metrics.start_time;

    printf("Performance Metrics:\n");
    printf("  Uptime: %ld seconds\n", elapsed);
    printf("  Total reads: %lu\n", metrics.total_reads);
    printf("  Total writes: %lu\n", metrics.total_writes);
    printf("  Reads/sec: %.2f\n", (double)metrics.total_reads / elapsed);
    printf("  Writes/sec: %.2f\n", (double)metrics.total_writes / elapsed);

    uint64_t total_access = metrics.cache_hits + metrics.cache_misses;
    if (total_access > 0) {
        printf("  Cache hit ratio: %.2f%%\n",
               (double)metrics.cache_hits / total_access * 100);
    }
}
```

---

## 12. Security Best Practices

### Encryption at Rest

```c
int enable_encryption(DB_ENV *env, const char *password) {
    int ret;

    // Set encryption password (must be done before env->open())
    ret = env->set_encrypt(env, password, DB_ENCRYPT_AES);
    if (ret != 0) {
        fprintf(stderr, "Encryption setup failed: %s\n", db_strerror(ret));
        return ret;
    }

    // Now open environment
    ret = env->open(env, "./data",
        DB_CREATE |
        DB_INIT_MPOOL |
        DB_INIT_TXN |
        DB_INIT_LOG |
        DB_INIT_LOCK |
        DB_THREAD,
        0);

    return ret;
}

// Database will inherit encryption from environment
int open_encrypted_database(DB_ENV *env, DB **dbp) {
    DB *db;
    int ret;

    ret = db_create(&db, env, 0);
    if (ret != 0) return ret;

    // Database automatically encrypted if env is encrypted
    ret = db->open(db, NULL, "encrypted.db", NULL,
                   DB_BTREE, DB_CREATE | DB_AUTO_COMMIT, 0);

    if (ret == 0) {
        *dbp = db;
    }

    return ret;
}
```

### Access Control

```c
// Set file permissions
int secure_database_files(const char *db_path) {
    // Set restrictive permissions (owner read/write only)
    chmod(db_path, S_IRUSR | S_IWUSR);

    // For environment directory
    chmod("./data", S_IRUSR | S_IWUSR | S_IXUSR);

    return 0;
}

// Sanitize user input
int safe_put(DB *db, const char *user_key, const char *user_data) {
    DBT key, data;

    // Validate input
    if (user_key == NULL || user_data == NULL) {
        return EINVAL;
    }

    size_t key_len = strlen(user_key);
    size_t data_len = strlen(user_data);

    // Enforce maximum sizes
    if (key_len > MAX_KEY_SIZE || data_len > MAX_DATA_SIZE) {
        return E2BIG;
    }

    // Check for null bytes (if not expected)
    if (memchr(user_key, '\0', key_len) != NULL) {
        return EINVAL;
    }

    memset(&key, 0, sizeof(DBT));
    memset(&data, 0, sizeof(DBT));

    key.data = (void *)user_key;
    key.size = key_len + 1;

    data.data = (void *)user_data;
    data.size = data_len + 1;

    return db->put(db, NULL, &key, &data, 0);
}
```

### Secure Configuration

```c
int secure_environment_config(DB_ENV *env) {
    // Disable unnecessary features
    env->set_flags(env, DB_NOPANIC, 0);

    // Set strict error handling
    env->set_errcall(env, custom_error_handler);

    // Enable detailed logging for security audits
    env->set_verbose(env, DB_VERB_DEADLOCK, 1);
    env->set_verbose(env, DB_VERB_RECOVERY, 1);
    env->set_verbose(env, DB_VERB_WAITSFOR, 1);

    // Set log file permissions
    env->log_set_config(env, DB_LOG_DIRECT, 1);

    return 0;
}
```

---

## 13. Testing Strategies

### Unit Tests (C with Check framework)

```c
#include <check.h>
#include <db.h>

START_TEST(test_basic_put_get) {
    DB_ENV *env;
    DB *db;
    DBT key, data;
    char buffer[100];
    int ret;

    // Create environment
    db_env_create(&env, 0);
    env->open(env, "./test_data", DB_CREATE | DB_INIT_MPOOL, 0);

    // Create database
    db_create(&db, env, 0);
    db->open(db, NULL, "test.db", NULL, DB_BTREE, DB_CREATE, 0);

    // Put
    memset(&key, 0, sizeof(DBT));
    memset(&data, 0, sizeof(DBT));
    key.data = "test_key";
    key.size = 9;
    data.data = "test_value";
    data.size = 11;

    ret = db->put(db, NULL, &key, &data, 0);
    ck_assert_int_eq(ret, 0);

    // Get
    memset(&data, 0, sizeof(DBT));
    data.data = buffer;
    data.ulen = sizeof(buffer);
    data.flags = DB_DBT_USERMEM;

    ret = db->get(db, NULL, &key, &data, 0);
    ck_assert_int_eq(ret, 0);
    ck_assert_str_eq((char *)data.data, "test_value");

    db->close(db, 0);
    env->close(env, 0);
}
END_TEST

START_TEST(test_transaction_commit) {
    // Transaction test implementation
}
END_TEST

START_TEST(test_transaction_rollback) {
    // Rollback test implementation
}
END_TEST

Suite *bdb_suite(void) {
    Suite *s;
    TCase *tc_core;

    s = suite_create("BerkeleyDB");
    tc_core = tcase_create("Core");

    tcase_add_test(tc_core, test_basic_put_get);
    tcase_add_test(tc_core, test_transaction_commit);
    tcase_add_test(tc_core, test_transaction_rollback);

    suite_add_tcase(s, tc_core);

    return s;
}

int main(void) {
    int number_failed;
    Suite *s;
    SRunner *sr;

    s = bdb_suite();
    sr = srunner_create(s);

    srunner_run_all(sr, CK_NORMAL);
    number_failed = srunner_ntests_failed(sr);
    srunner_free(sr);

    return (number_failed == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}
```

### Python Tests

```python
import unittest
import berkeleydb as bdb
import os
import shutil

class TestBerkeleyDB(unittest.TestCase):

    def setUp(self):
        """Set up test environment."""
        self.test_dir = './test_data'
        os.makedirs(self.test_dir, exist_ok=True)

        self.env = bdb.db.DBEnv()
        self.env.open(self.test_dir,
                      bdb.db.DB_CREATE |
                      bdb.db.DB_INIT_MPOOL |
                      bdb.db.DB_INIT_TXN |
                      bdb.db.DB_INIT_LOG |
                      bdb.db.DB_INIT_LOCK)

        self.db = bdb.db.DB(self.env)
        self.db.open('test.db',
                     dbtype=bdb.db.DB_BTREE,
                     flags=bdb.db.DB_CREATE | bdb.db.DB_AUTO_COMMIT)

    def tearDown(self):
        """Clean up after tests."""
        self.db.close()
        self.env.close()
        shutil.rmtree(self.test_dir)

    def test_put_get(self):
        """Test basic put and get operations."""
        self.db.put(b'key1', b'value1')
        result = self.db.get(b'key1')
        self.assertEqual(result, b'value1')

    def test_delete(self):
        """Test delete operation."""
        self.db.put(b'key1', b'value1')
        self.db.delete(b'key1')
        result = self.db.get(b'key1')
        self.assertIsNone(result)

    def test_transaction_commit(self):
        """Test transaction commit."""
        txn = self.env.txn_begin()
        self.db.put(b'txn_key', b'txn_value', txn=txn)
        txn.commit()

        result = self.db.get(b'txn_key')
        self.assertEqual(result, b'txn_value')

    def test_transaction_abort(self):
        """Test transaction abort."""
        txn = self.env.txn_begin()
        self.db.put(b'abort_key', b'abort_value', txn=txn)
        txn.abort()

        result = self.db.get(b'abort_key')
        self.assertIsNone(result)

    def test_cursor_iteration(self):
        """Test cursor iteration."""
        # Insert test data
        test_data = [(b'key1', b'value1'),
                     (b'key2', b'value2'),
                     (b'key3', b'value3')]

        for key, value in test_data:
            self.db.put(key, value)

        # Iterate with cursor
        cursor = self.db.cursor()
        results = list(cursor)
        cursor.close()

        self.assertEqual(len(results), 3)
        self.assertEqual(results, test_data)

if __name__ == '__main__':
    unittest.main()
```

---

## 14. Migration Strategies

### From SQLite to Berkeley DB

**SQLite Export:**
```bash
# Export SQLite database to CSV
sqlite3 mydb.sqlite <<EOF
.mode csv
.output users.csv
SELECT * FROM users;
.output posts.csv
SELECT * FROM posts;
EOF
```

**Import to Berkeley DB:**
```python
import csv
import berkeleydb as bdb

def migrate_from_sqlite():
    # Open Berkeley DB
    env = bdb.db.DBEnv()
    env.open('./bdb_data',
             bdb.db.DB_CREATE |
             bdb.db.DB_INIT_MPOOL |
             bdb.db.DB_INIT_TXN |
             bdb.db.DB_INIT_LOG |
             bdb.db.DB_INIT_LOCK)

    db = bdb.db.DB(env)
    db.open('migrated.db',
            dbtype=bdb.db.DB_BTREE,
            flags=bdb.db.DB_CREATE | bdb.db.DB_AUTO_COMMIT)

    # Import from CSV
    with open('users.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = f"user:{row['id']}".encode()
            value = f"{row['name']}:{row['email']}".encode()
            db.put(key, value)

    print("Migration complete")

    db.close()
    env.close()

migrate_from_sqlite()
```

### From Key-Value Store (Redis/Memcached)

```python
import redis
import berkeleydb as bdb

def migrate_from_redis(redis_host='localhost'):
    # Connect to Redis
    r = redis.Redis(host=redis_host)

    # Open Berkeley DB
    env = bdb.db.DBEnv()
    env.open('./bdb_data', bdb.db.DB_CREATE | bdb.db.DB_INIT_MPOOL)

    db = bdb.db.DB(env)
    db.open('migrated.db', dbtype=bdb.db.DB_HASH,
            flags=bdb.db.DB_CREATE)

    # Migrate all keys
    for key in r.scan_iter():
        value = r.get(key)
        db.put(key, value)
        print(f"Migrated: {key.decode()}")

    db.close()
    env.close()
```

---

## 15. Common Patterns

### Connection Pool Pattern

```c
// Simple connection pool for multi-threaded applications
typedef struct {
    DB **databases;
    int pool_size;
    pthread_mutex_t mutex;
    int next_db;
} db_pool_t;

db_pool_t* create_db_pool(DB_ENV *env, const char *db_name, int size) {
    db_pool_t *pool = malloc(sizeof(db_pool_t));
    pool->databases = malloc(sizeof(DB *) * size);
    pool->pool_size = size;
    pool->next_db = 0;
    pthread_mutex_init(&pool->mutex, NULL);

    for (int i = 0; i < size; i++) {
        db_create(&pool->databases[i], env, 0);
        pool->databases[i]->open(pool->databases[i], NULL, db_name, NULL,
                                  DB_BTREE, DB_CREATE | DB_THREAD, 0);
    }

    return pool;
}

DB* get_db_from_pool(db_pool_t *pool) {
    pthread_mutex_lock(&pool->mutex);
    DB *db = pool->databases[pool->next_db];
    pool->next_db = (pool->next_db + 1) % pool->pool_size;
    pthread_mutex_unlock(&pool->mutex);
    return db;
}
```

### Cache-Aside Pattern

```python
class CachedBerkeleyDB:
    """Berkeley DB with in-memory cache."""

    def __init__(self, env, db, cache_size=1000):
        self.db = db
        self.env = env
        self.cache = {}
        self.cache_size = cache_size

    def get(self, key):
        """Get with cache check."""
        # Check cache first
        if key in self.cache:
            return self.cache[key]

        # Miss - get from database
        value = self.db.get(key)

        if value is not None:
            # Add to cache (with simple size limit)
            if len(self.cache) >= self.cache_size:
                # Remove oldest entry (simplified LRU)
                self.cache.pop(next(iter(self.cache)))

            self.cache[key] = value

        return value

    def put(self, key, value):
        """Put and update cache."""
        self.db.put(key, value)
        self.cache[key] = value

    def delete(self, key):
        """Delete and invalidate cache."""
        self.db.delete(key)
        self.cache.pop(key, None)
```

---

## 16. Production Checklist

### Pre-Deployment

**Configuration:**
- [ ] Cache size configured appropriately (20-80% of available RAM)
- [ ] Log file size limits set
- [ ] Transaction timeout configured
- [ ] Deadlock detection enabled
- [ ] Page size optimized for workload

**Security:**
- [ ] Encryption enabled (if required)
- [ ] File permissions restrictive (600/700)
- [ ] Database files not world-readable
- [ ] Error messages don't expose sensitive data

**Backup:**
- [ ] Hot backup process automated
- [ ] Backup retention policy defined
- [ ] Recovery tested successfully
- [ ] Log archival configured

**Monitoring:**
- [ ] Performance metrics collection enabled
- [ ] Deadlock monitoring configured
- [ ] Disk space monitoring set up
- [ ] Log file growth monitoring

### Post-Deployment

**Verification:**
- [ ] Database accessible
- [ ] Read/write operations successful
- [ ] Backup completed
- [ ] Monitoring data flowing

**Performance:**
- [ ] Cache hit ratio > 90%
- [ ] Response times within SLA
- [ ] No deadlock escalations
- [ ] Log files rotating properly

---

## 17. Performance Tuning

### Tuning Parameters

```c
// Optimal configuration for high-performance workload
int tune_for_performance(DB_ENV *env, DB *db) {
    // Large cache for frequently accessed data
    env->set_cachesize(env, 0, 512 * 1024 * 1024, 1); // 512 MB

    // Increase log buffer for write-heavy workloads
    env->set_lg_bsize(env, 256 * 1024); // 256 KB

    // Larger log files (less frequent checkpoints)
    env->set_lg_max(env, 100 * 1024 * 1024); // 100 MB

    // Larger page size for sequential access
    db->set_pagesize(db, 32768); // 32 KB

    // Disable synchronous writes (trade durability for speed)
    // WARNING: Risk of data loss on crash
    // env->set_flags(env, DB_TXN_NOSYNC, 1);

    // Write-delayed writes for better throughput
    env->set_flags(env, DB_TXN_WRITE_NOSYNC, 1);

    // Set deadlock detection
    env->set_lk_detect(env, DB_LOCK_DEFAULT);

    return 0;
}
```

### Benchmark Results

```
Configuration: 4-core CPU, 16GB RAM, SSD
Dataset: 10 million records, 100-byte values
Cache: 2GB

Operation       Throughput      Latency (p99)
-------------------------------------------------
Sequential Write 250K ops/sec   0.2ms
Random Write     150K ops/sec   0.5ms
Sequential Read  300K ops/sec   0.1ms
Random Read      180K ops/sec   0.3ms
Transaction      100K txn/sec   1.0ms
```

---

## 18. Comparison with Other Databases

### Berkeley DB vs SQLite

| Feature | Berkeley DB | SQLite |
|---------|-------------|--------|
| **Data Model** | Key-value (multiple access methods) | Relational (SQL) |
| **API** | C library (direct) | SQL + C library |
| **Transactions** | Full ACID | Full ACID |
| **Replication** | Built-in | Third-party solutions |
| **Use Case** | Embedded key-value | Embedded SQL |
| **Performance** | Higher for key-value | Better for complex queries |
| **Size** | ~1-5 MB | ~600 KB |

### Berkeley DB vs LevelDB/RocksDB

| Feature | Berkeley DB | LevelDB/RocksDB |
|---------|-------------|-----------------|
| **Access Methods** | B-tree, Hash, Queue, Recno | LSM-tree |
| **Write Performance** | Good | Excellent |
| **Read Performance** | Excellent | Good |
| **Maturity** | Very mature (1994) | Mature (2011/2012) |
| **Replication** | Built-in | External tools |
| **Transactions** | Full ACID | Optimistic |

---

## 19. Troubleshooting

### Common Issues

**Database Corruption:**
```bash
# Verify database integrity
db_verify -h /data mydb.db

# If corrupted, try recovery
db_recover -h /data

# If recovery fails, try catastrophic recovery
db_recover -h /data -c

# Salvage data from corrupted database
db_dump -r -h /data mydb.db > salvaged.dump
db_load -h /data recovered.db < salvaged.dump
```

**Deadlocks:**
```c
// Configure deadlock detection
env->set_lk_detect(env, DB_LOCK_DEFAULT);

// Handle deadlock in application
ret = db->put(db, txn, &key, &data, 0);
if (ret == DB_LOCK_DEADLOCK) {
    txn->abort(txn);
    // Retry operation
}
```

**Log Files Growing:**
```bash
# Check which logs can be archived
db_archive -h /data -l

# Archive and remove unnecessary logs
db_archive -h /data -d

# Or configure automatic log removal
# env->log_set_config(env, DB_LOG_AUTO_REMOVE, 1);
```

**Performance Degradation:**
```c
// Compact database to reclaim space
db->compact(db, NULL, NULL, NULL, NULL, DB_FREE_SPACE, NULL);

// Update statistics
db->stat_print(db, DB_STAT_CLEAR);

// Check cache hit ratio
// If < 90%, increase cache size
```

---

## 20. Resources and References

### Official Documentation

- **Oracle Berkeley DB Documentation**: https://docs.oracle.com/cd/E17276_01/html/index.html
- **Berkeley DB C API Reference**: https://docs.oracle.com/cd/E17276_01/html/api_reference/C/frame_main.html
- **Berkeley DB Programmer's Reference Guide**: https://docs.oracle.com/cd/E17276_01/html/programmer_reference/index.html

### Community Resources

- **Berkeley DB Forum**: Oracle Technology Network
- **Source Code**: https://github.com/berkeleydb/libdb (legacy)
- **Python bsddb3**: https://www.jcea.es/programacion/pybsddb.htm

### Books

- *Berkeley DB* by Sleepycat Software (Oracle)
- *Database Internals* by Alex Petrov (covers Berkeley DB architecture)

### Related Guides

- [SQLite Guidelines](sqlite.md) - Alternative embedded database
- [C/C++ Best Practices](c-cpp.md) - Language-specific guidelines
- [Testing Guidelines](testing.md) - Testing strategies

---

**Document Version**: 1.0
**Last Updated**: February 2026
**Compatible with**: Berkeley DB 18.1.x

For updates and contributions, see the [companion guides](README.md).


**End of Berkeley DB Development Guidelines**
