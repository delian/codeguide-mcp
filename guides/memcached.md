# Memcached Development Guidelines
Mandatory coding standards and development practices for Memcached development. Memcached server, client libraries (Python, Node, PHP, Java, Go), text/binary protocol, consistent-hashing clients.

---

**Agent Profile**: The Memcached Expert
**Role**: Senior Cache & Distributed Systems Engineer
**Objective**: Generate production-ready, fast and reliable caching layers using Memcached.
**Tools**: Memcached server, client libraries (Python, Node, PHP, Java, Go), text/binary protocol, consistent-hashing clients

---

## 1. Core Philosophies: CACHE-FIRST

The agent must adhere to the **CACHE-FIRST** principles for every Memcached implementation:

**Test-Driven Development (TDD)**: ALWAYS write tests BEFORE implementation (Red-Green-Refactor cycle mandatory).
**Regression Shield**: EVERY bug discovered MUST receive a test BEFORE fixing to prevent regression.

- **C**ache semantics: Treat Memcached as cache only; never as source of truth; always handle cache miss and rehydration.
- **A**vailability: Design for server failure and network issues; use connection pooling, timeouts, and fallback to origin.
- **C**onsistent hashing: Use client-side consistent hashing; understand key distribution and rebalance on topology change.
- **H**it/miss and TTL: Design keys and expiry for hit rate; avoid thundering herd on cold cache or stampede.
- **E**viction awareness: Expect LRU eviction; set sensible memory limits and TTLs; avoid unbounded key growth.
- **Verified Code**: Agent-generated code MUST handle get/set errors, respect TTL and key size limits, and pass tests before delivery.

---

## 2. Core Concepts and Architecture

Memcached is a high-performance, distributed memory object caching system designed to speed up dynamic web applications by alleviating database load. It's an in-memory key-value store for small chunks of arbitrary data (strings, objects) from results of database calls, API calls, or page rendering.

### Architecture Overview

```
Application Layer:
┌──────────────────────────────────────────────────────────┐
│  Web Application / API Service                           │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐        │
│  │  App 1     │  │  App 2     │  │  App 3     │        │
│  └─────┬──────┘  └─────┬──────┘  └─────┬──────┘        │
│        │                │                │               │
└────────┼────────────────┼────────────────┼───────────────┘
         │                │                │
         └────────────────┼────────────────┘
                          ↓
┌──────────────────────────────────────────────────────────┐
│  Memcached Cluster (Distributed Hash Table)              │
├──────────────────────────────────────────────────────────┤
│                                                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │ Memcached 1 │  │ Memcached 2 │  │ Memcached 3 │     │
│  │  Port: 11211│  │  Port: 11211│  │  Port: 11211│     │
│  │  Memory: 2GB│  │  Memory: 2GB│  │  Memory: 2GB│     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
│                                                           │
│  Client-side consistent hashing determines which         │
│  server stores each key                                  │
└──────────────────────────────────────────────────────────┘
         │                │                │
         └────────────────┼────────────────┘
                          ↓
┌──────────────────────────────────────────────────────────┐
│  Database / Persistent Storage                           │
│  ┌────────────────────────────────────────────────┐     │
│  │  PostgreSQL / MySQL / MongoDB                  │     │
│  │  Source of truth for data                      │     │
│  └────────────────────────────────────────────────┘     │
└──────────────────────────────────────────────────────────┘
```

### Key Characteristics

**In-Memory Storage:**
- All data stored in RAM (no disk persistence)
- Extremely fast: sub-millisecond latency
- Volatile: data lost on restart

**Distributed Architecture:**
- No communication between servers (shared-nothing)
- Client determines which server to use (consistent hashing)
- Horizontal scalability by adding servers

**LRU Eviction:**
- Least Recently Used items evicted when memory full
- Configurable memory limits per server
- Automatic memory management

**Simple Protocol:**
- Text protocol: Human-readable, easy to debug
- Binary protocol: More efficient, less overhead
- Commands: GET, SET, DELETE, ADD, REPLACE, INCR, DECR, etc.

**Multi-threaded:**
- Efficient use of multi-core CPUs
- Concurrent request handling
- Lock-free operations where possible

### Data Flow Example

```
Cache Hit (Fast Path):
1. App requests user:123
2. Client hashes key → Server 2
3. Memcached returns cached data
4. App uses data
⏱ Latency: ~1ms

Cache Miss (Slow Path):
1. App requests user:456
2. Client hashes key → Server 1
3. Memcached returns NULL (miss)
4. App queries database
5. App stores result in Memcached
6. App uses data
⏱ Latency: ~50-100ms (database query)

Subsequent requests for user:456 are cache hits
```

### Memory Model

```
Memcached Memory Structure:

┌─────────────────────────────────────────┐
│  Memory Pool (e.g., 2GB)                │
├─────────────────────────────────────────┤
│                                          │
│  Slab Class 1 (88 bytes/item)           │
│  ┌────┐┌────┐┌────┐┌────┐              │
│  │Item││Item││Item││Item│ ...          │
│  └────┘└────┘└────┘└────┘              │
│                                          │
│  Slab Class 2 (112 bytes/item)          │
│  ┌────┐┌────┐┌────┐                    │
│  │Item││Item││Item│ ...                │
│  └────┘└────┘└────┘                    │
│                                          │
│  Slab Class 3 (144 bytes/item)          │
│  ...                                     │
│                                          │
│  Slab Class N (1MB/item)                │
│  ┌─────────┐┌─────────┐                │
│  │  Item   ││  Item   │ ...            │
│  └─────────┘└─────────┘                │
│                                          │
│  LRU chains per slab class               │
│  Automatic eviction when full            │
└─────────────────────────────────────────┘
```

---

## 2A. Test-Driven Development (TDD) Protocol (MANDATORY)

**CRITICAL: Follow the Red-Green-Refactor cycle for ALL new code.**

### TDD Cycle

```
1. RED: Write a failing test first
   ↓
2. GREEN: Write minimal code to make it pass
   ↓
3. REFACTOR: Improve code while keeping tests green
   ↓
   Repeat
```

### Example TDD Workflow for Memcached (Python with pytest and pymemcache)

```python
# Step 1: RED - Write failing test first
import pytest
from pymemcache.client.base import Client

@pytest.fixture
def mc_client():
    client = Client("localhost:11211")
    client.flush_all()
    yield client
    client.close()

def test_set_and_get_cache_value(mc_client):
    """Test setting and retrieving a cached value."""
    cache = CacheService(mc_client)
    cache.set("user:1001", "Alice", ttl=300)
    result = cache.get("user:1001")
    assert result == "Alice"

# Run: pytest test_memcached.py -v
# FAILS - NameError: name 'CacheService' is not defined

# Step 2: GREEN - Write minimal implementation
class CacheService:
    def __init__(self, client):
        self.client = client

    def set(self, key, value, ttl=0):
        self.client.set(key, value, expire=ttl)

    def get(self, key):
        result = self.client.get(key)
        if result is not None:
            return result.decode("utf-8") if isinstance(result, bytes) else result
        return None

# Run: pytest test_memcached.py -v
# PASSES

# Step 3: REFACTOR - Add delete, multi-get, and cache-aside pattern
import json

class CacheService:
    def __init__(self, client):
        self.client = client

    def set(self, key, value, ttl=0):
        self.client.set(key, value, expire=ttl)

    def get(self, key):
        result = self.client.get(key)
        if result is not None:
            return result.decode("utf-8") if isinstance(result, bytes) else result
        return None

    def delete(self, key):
        self.client.delete(key, noreply=False)

    def get_multi(self, keys):
        results = self.client.get_many(keys)
        return {k: v.decode("utf-8") if isinstance(v, bytes) else v
                for k, v in results.items()}

    def cache_aside(self, key, fetch_func, ttl=300):
        """Cache-aside pattern: check cache first, fetch and store on miss."""
        value = self.get(key)
        if value is None:
            value = fetch_func()
            if value is not None:
                self.set(key, value, ttl=ttl)
        return value

# Tests still pass
```

---

## 2B. Bug Fix Protocol (MANDATORY)

**CRITICAL: Every bug MUST receive a regression test BEFORE fixing.**

### Bug Fix Workflow

```
1. Bug Reported/Discovered
   ↓
2. Write a test that REPRODUCES the bug (test will FAIL)
   ↓
3. Verify the test fails for the right reason
   ↓
4. Fix the bug (make the test pass)
   ↓
5. Verify the test now PASSES
   ↓
6. Document the bug in test comments
```

### Example Bug Fix

```python
# Bug: delete() returns success even when the key does not exist,
# causing the caller to believe a stale entry was removed

import pytest

# Step 1: Write test that reproduces the bug
def test_delete_nonexistent_key_returns_false(mc_client):
    """Regression: delete() should return False when the key does not
    exist in the cache, not silently succeed."""
    cache = CacheService(mc_client)
    result = cache.delete("nonexistent_key")
    assert result is False

# FAILS - AssertionError: True != False (delete reported success)

# Step 2: Fix the bug
class CacheService:
    # ... existing code ...

    def delete(self, key):
        return self.client.delete(key, noreply=False)

# PASSES - bug fixed, regression prevented
# pymemcache's delete() with noreply=False returns True if deleted,
# False if key didn't exist
```

---

## 3. Installation and Setup

### Ubuntu/Debian Installation

```bash
# Install Memcached
sudo apt-get update
sudo apt-get install -y memcached libmemcached-tools

# Check version
memcached -h

# Start service
sudo systemctl start memcached
sudo systemctl enable memcached

# Check status
sudo systemctl status memcached

# Test connection
echo "stats" | nc localhost 11211
```

### macOS Installation

```bash
# Using Homebrew
brew install memcached

# Start Memcached
memcached -m 64 -p 11211 -u $(whoami) -l 127.0.0.1

# Or as a service
brew services start memcached
```

### Docker Installation

```bash
# Run Memcached in Docker
docker run -d \
  --name memcached \
  -p 11211:11211 \
  memcached:latest \
  memcached -m 256 -c 1024

# With custom configuration
docker run -d \
  --name memcached \
  -p 11211:11211 \
  memcached:alpine \
  memcached \
    -m 512 \
    -c 2048 \
    -t 4 \
    -v
```

### Configuration

```bash
# /etc/memcached.conf

# Memory allocation (in megabytes)
-m 512

# Maximum simultaneous connections
-c 1024

# Number of threads
-t 4

# Listen on localhost only (more secure)
-l 127.0.0.1

# Port
-p 11211

# User to run as
-u memcache

# Maximize core file limit
-r

# Verbose mode (for debugging)
# -v
# -vv (very verbose)

# Enable large memory pages (for better performance)
-L

# Maximum item size (default 1MB)
-I 5m

# Disable CAS (Compare-And-Swap) for better performance
# -C
```

### Starting Memcached

```bash
# Basic start
memcached -m 64 -p 11211 -u nobody -l 127.0.0.1

# Production start with optimal settings
memcached \
  -m 2048 \           # 2GB memory
  -p 11211 \          # Port
  -c 2048 \           # Max connections
  -t 4 \              # 4 threads
  -u memcache \       # Run as memcache user
  -l 0.0.0.0 \        # Listen on all interfaces
  -L \                # Large memory pages
  -I 5m \             # Max item size 5MB
  -o modern \         # Modern mode (faster hash, etc.)
  -v                  # Verbose logging

# As daemon
memcached -d -m 2048 -p 11211 -u memcache -l 127.0.0.1
```

### Client Libraries

```bash
# Python
pip install pymemcache python-memcached

# Node.js
npm install memcached memjs

# PHP
sudo apt-get install php-memcached

# Java
# Add to pom.xml:
# <dependency>
#   <groupId>net.spy</groupId>
#   <artifactId>spymemcached</artifactId>
#   <version>2.12.3</version>
# </dependency>

# Go
go get github.com/bradfitz/gomemcache/memcache

# Ruby
gem install dalli
```

## 4. Basic Operations

### Python Client (pymemcache)

```python
from pymemcache.client.base import Client
from pymemcache.client.hash import HashClient
from typing import Optional, Any
import json
import pickle

class MemcachedClient:
    def __init__(self, servers=['127.0.0.1:11211'], serializer=None, deserializer=None):
        """Initialize Memcached client.

        Args:
            servers: List of server addresses
            serializer: Function to serialize values (default: pickle)
            deserializer: Function to deserialize values (default: pickle)
        """
        if len(servers) == 1:
            # Single server
            host, port = servers[0].split(':')
            self.client = Client(
                (host, int(port)),
                serializer=serializer or self._serialize,
                deserializer=deserializer or self._deserialize,
                connect_timeout=2,
                timeout=2
            )
        else:
            # Multiple servers with consistent hashing
            server_tuples = [(s.split(':')[0], int(s.split(':')[1])) for s in servers]
            self.client = HashClient(
                server_tuples,
                serializer=serializer or self._serialize,
                deserializer=deserializer or self._deserialize
            )

    @staticmethod
    def _serialize(key, value):
        """Serialize value using pickle."""
        if isinstance(value, (str, bytes)):
            return value.encode('utf-8') if isinstance(value, str) else value, 1
        return pickle.dumps(value), 2

    @staticmethod
    def _deserialize(key, value, flags):
        """Deserialize value."""
        if flags == 1:
            return value.decode('utf-8')
        elif flags == 2:
            return pickle.loads(value)
        return value

    def set(self, key: str, value: Any, expire: int = 0) -> bool:
        """Set a key-value pair.

        Args:
            key: Cache key
            value: Value to store
            expire: Expiration time in seconds (0 = no expiration)

        Returns:
            True if successful
        """
        return self.client.set(key, value, expire=expire)

    def get(self, key: str) -> Optional[Any]:
        """Get value by key.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found
        """
        return self.client.get(key)

    def get_many(self, keys: list) -> dict:
        """Get multiple values.

        Args:
            keys: List of cache keys

        Returns:
            Dictionary of key-value pairs
        """
        return self.client.get_many(keys)

    def set_many(self, mapping: dict, expire: int = 0) -> list:
        """Set multiple key-value pairs.

        Args:
            mapping: Dictionary of key-value pairs
            expire: Expiration time in seconds

        Returns:
            List of keys that failed to set
        """
        return self.client.set_many(mapping, expire=expire)

    def delete(self, key: str) -> bool:
        """Delete a key.

        Args:
            key: Cache key

        Returns:
            True if key was deleted
        """
        return self.client.delete(key)

    def delete_many(self, keys: list) -> bool:
        """Delete multiple keys.

        Args:
            keys: List of cache keys

        Returns:
            True if all keys were deleted
        """
        return self.client.delete_many(keys)

    def add(self, key: str, value: Any, expire: int = 0) -> bool:
        """Add key only if it doesn't exist.

        Args:
            key: Cache key
            value: Value to store
            expire: Expiration time in seconds

        Returns:
            True if key was added, False if already exists
        """
        return self.client.add(key, value, expire=expire)

    def replace(self, key: str, value: Any, expire: int = 0) -> bool:
        """Replace value only if key exists.

        Args:
            key: Cache key
            value: New value
            expire: Expiration time in seconds

        Returns:
            True if key was replaced, False if doesn't exist
        """
        return self.client.replace(key, value, expire=expire)

    def incr(self, key: str, delta: int = 1) -> Optional[int]:
        """Increment numeric value.

        Args:
            key: Cache key
            delta: Amount to increment

        Returns:
            New value or None if key doesn't exist
        """
        return self.client.incr(key, delta)

    def decr(self, key: str, delta: int = 1) -> Optional[int]:
        """Decrement numeric value.

        Args:
            key: Cache key
            delta: Amount to decrement

        Returns:
            New value or None if key doesn't exist
        """
        return self.client.decr(key, delta)

    def flush_all(self) -> bool:
        """Clear all cached data.

        Returns:
            True if successful
        """
        return self.client.flush_all()

    def stats(self) -> dict:
        """Get server statistics.

        Returns:
            Dictionary of statistics
        """
        return self.client.stats()

    def close(self):
        """Close connection."""
        self.client.close()


# Usage examples
def examples():
    # Single server
    cache = MemcachedClient(['127.0.0.1:11211'])

    # Set/Get basic values
    cache.set('user:123', 'Alice', expire=3600)  # 1 hour
    user = cache.get('user:123')
    print(f"User: {user}")

    # Store complex objects
    cache.set('user:123:profile', {
        'name': 'Alice',
        'email': 'alice@example.com',
        'age': 30
    }, expire=1800)  # 30 minutes

    profile = cache.get('user:123:profile')
    print(f"Profile: {profile}")

    # Batch operations
    cache.set_many({
        'counter:views': 1000,
        'counter:clicks': 500,
        'counter:users': 50
    }, expire=3600)

    counters = cache.get_many(['counter:views', 'counter:clicks', 'counter:users'])
    print(f"Counters: {counters}")

    # Atomic operations
    cache.set('page:views', '0')
    cache.incr('page:views', 1)
    views = cache.get('page:views')
    print(f"Page views: {views}")

    # Add (only if not exists)
    added = cache.add('user:456', 'Bob', expire=3600)
    print(f"Added: {added}")

    # Statistics
    stats = cache.stats()
    print(f"Stats: {stats}")

    # Cleanup
    cache.close()
```

### Node.js Client

```javascript
const Memcached = require('memcached');

class MemcachedClient {
    constructor(servers = '127.0.0.1:11211', options = {}) {
        this.client = new Memcached(servers, {
            retries: 3,
            retry: 10000,
            remove: true,
            failOverServers: ['192.168.0.103:11211'],
            ...options
        });
    }

    set(key, value, lifetime = 0) {
        return new Promise((resolve, reject) => {
            this.client.set(key, value, lifetime, (err) => {
                if (err) reject(err);
                else resolve(true);
            });
        });
    }

    get(key) {
        return new Promise((resolve, reject) => {
            this.client.get(key, (err, data) => {
                if (err) reject(err);
                else resolve(data);
            });
        });
    }

    getMulti(keys) {
        return new Promise((resolve, reject) => {
            this.client.getMulti(keys, (err, data) => {
                if (err) reject(err);
                else resolve(data);
            });
        });
    }

    delete(key) {
        return new Promise((resolve, reject) => {
            this.client.del(key, (err) => {
                if (err) reject(err);
                else resolve(true);
            });
        });
    }

    add(key, value, lifetime = 0) {
        return new Promise((resolve, reject) => {
            this.client.add(key, value, lifetime, (err) => {
                if (err) reject(err);
                else resolve(true);
            });
        });
    }

    replace(key, value, lifetime = 0) {
        return new Promise((resolve, reject) => {
            this.client.replace(key, value, lifetime, (err) => {
                if (err) reject(err);
                else resolve(true);
            });
        });
    }

    incr(key, amount = 1) {
        return new Promise((resolve, reject) => {
            this.client.incr(key, amount, (err, result) => {
                if (err) reject(err);
                else resolve(result);
            });
        });
    }

    decr(key, amount = 1) {
        return new Promise((resolve, reject) => {
            this.client.decr(key, amount, (err, result) => {
                if (err) reject(err);
                else resolve(result);
            });
        });
    }

    flush() {
        return new Promise((resolve, reject) => {
            this.client.flush((err) => {
                if (err) reject(err);
                else resolve(true);
            });
        });
    }

    stats() {
        return new Promise((resolve, reject) => {
            this.client.stats((err, stats) => {
                if (err) reject(err);
                else resolve(stats);
            });
        });
    }

    end() {
        this.client.end();
    }
}

// Usage
async function main() {
    const cache = new MemcachedClient(['127.0.0.1:11211', '127.0.0.1:11212']);

    // Set value
    await cache.set('user:123', { name: 'Alice', age: 30 }, 3600);

    // Get value
    const user = await cache.get('user:123');
    console.log('User:', user);

    // Increment counter
    await cache.set('views', 0, 3600);
    await cache.incr('views', 1);
    const views = await cache.get('views');
    console.log('Views:', views);

    // Batch get
    const data = await cache.getMulti(['user:123', 'views']);
    console.log('Data:', data);

    // Stats
    const stats = await cache.stats();
    console.log('Stats:', stats);

    // Cleanup
    cache.end();
}

main().catch(console.error);
```

## 5. Cache Patterns and Strategies

### Cache-Aside (Lazy Loading)

```python
import time
from typing import Optional

class CacheAside:
    """Cache-aside pattern implementation."""

    def __init__(self, cache, database):
        self.cache = cache
        self.db = database

    def get_user(self, user_id: int) -> Optional[dict]:
        """Get user with cache-aside pattern.

        1. Check cache first
        2. If miss, query database
        3. Store in cache
        4. Return data
        """
        cache_key = f"user:{user_id}"

        # Try cache first
        user = self.cache.get(cache_key)
        if user is not None:
            print(f"Cache HIT: {cache_key}")
            return user

        print(f"Cache MISS: {cache_key}")

        # Query database
        user = self.db.query_user(user_id)
        if user is None:
            return None

        # Store in cache
        self.cache.set(cache_key, user, expire=3600)  # 1 hour TTL

        return user

    def update_user(self, user_id: int, data: dict) -> bool:
        """Update user and invalidate cache."""
        cache_key = f"user:{user_id}"

        # Update database
        success = self.db.update_user(user_id, data)

        if success:
            # Invalidate cache
            self.cache.delete(cache_key)

        return success


# Advanced: Cache-aside with automatic retry
class CacheAsideWithRetry:
    def __init__(self, cache, database, ttl=3600, retry_on_miss=True):
        self.cache = cache
        self.db = database
        self.ttl = ttl
        self.retry_on_miss = retry_on_miss

    def get(self, key: str, fetch_func, *args, **kwargs):
        """Generic cache-aside with automatic fetch.

        Args:
            key: Cache key
            fetch_func: Function to fetch data on miss
            *args, **kwargs: Arguments for fetch_func

        Returns:
            Cached or fetched data
        """
        # Try cache
        value = self.cache.get(key)
        if value is not None:
            return value

        # Fetch from source
        value = fetch_func(*args, **kwargs)

        if value is not None:
            # Store in cache
            self.cache.set(key, value, expire=self.ttl)

        return value


# Usage
def fetch_user_from_db(user_id: int) -> dict:
    """Fetch user from database."""
    # Simulate database query
    time.sleep(0.1)
    return {'id': user_id, 'name': 'Alice', 'email': 'alice@example.com'}

cache_aside = CacheAsideWithRetry(cache, None, ttl=3600)
user = cache_aside.get('user:123', fetch_user_from_db, 123)
```

### Write-Through Cache

```python
class WriteThrough:
    """Write-through cache pattern."""

    def __init__(self, cache, database):
        self.cache = cache
        self.db = database

    def create_user(self, user_data: dict) -> int:
        """Create user with write-through caching.

        1. Write to database first
        2. Write to cache
        3. Return result
        """
        # Write to database
        user_id = self.db.insert_user(user_data)

        # Write to cache
        cache_key = f"user:{user_id}"
        self.cache.set(cache_key, user_data, expire=3600)

        return user_id

    def update_user(self, user_id: int, updates: dict) -> bool:
        """Update user with write-through caching."""
        cache_key = f"user:{user_id}"

        # Update database
        success = self.db.update_user(user_id, updates)

        if success:
            # Update cache
            user = self.db.query_user(user_id)
            self.cache.set(cache_key, user, expire=3600)

        return success
```

### Write-Behind (Write-Back) Cache

```python
import queue
import threading
import time

class WriteBehind:
    """Write-behind cache pattern with async writes."""

    def __init__(self, cache, database, flush_interval=5):
        self.cache = cache
        self.db = database
        self.write_queue = queue.Queue()
        self.flush_interval = flush_interval
        self.running = True

        # Start background writer
        self.writer_thread = threading.Thread(target=self._flush_worker)
        self.writer_thread.daemon = True
        self.writer_thread.start()

    def update_user(self, user_id: int, updates: dict):
        """Update user with write-behind caching.

        1. Update cache immediately
        2. Queue database write
        3. Background worker persists to database
        """
        cache_key = f"user:{user_id}"

        # Update cache immediately (fast)
        user = self.cache.get(cache_key)
        if user:
            user.update(updates)
            self.cache.set(cache_key, user, expire=3600)

        # Queue database write (async)
        self.write_queue.put(('update', user_id, updates))

    def _flush_worker(self):
        """Background worker to flush writes to database."""
        while self.running:
            try:
                # Batch writes
                batch = []
                deadline = time.time() + self.flush_interval

                while time.time() < deadline and len(batch) < 100:
                    try:
                        item = self.write_queue.get(timeout=0.1)
                        batch.append(item)
                    except queue.Empty:
                        break

                # Flush batch to database
                if batch:
                    self._flush_batch(batch)

            except Exception as e:
                print(f"Flush error: {e}")

    def _flush_batch(self, batch):
        """Flush batch of writes to database."""
        for operation, user_id, updates in batch:
            try:
                if operation == 'update':
                    self.db.update_user(user_id, updates)
            except Exception as e:
                print(f"Database write error: {e}")

    def shutdown(self):
        """Shutdown and flush remaining writes."""
        self.running = False
        self.writer_thread.join()

        # Flush remaining items
        remaining = []
        while not self.write_queue.empty():
            remaining.append(self.write_queue.get())
        if remaining:
            self._flush_batch(remaining)
```

### Cache Warming

```python
class CacheWarming:
    """Proactive cache warming strategies."""

    def __init__(self, cache, database):
        self.cache = cache
        self.db = database

    def warm_popular_items(self, limit: int = 100):
        """Pre-load popular items into cache."""
        # Get most accessed items
        popular_items = self.db.query_popular_items(limit)

        # Batch load into cache
        cache_data = {}
        for item in popular_items:
            cache_key = f"item:{item['id']}"
            cache_data[cache_key] = item

        self.cache.set_many(cache_data, expire=3600)

        print(f"Warmed {len(cache_data)} items")

    def warm_user_data(self, user_id: int):
        """Pre-load user's frequently accessed data."""
        cache_keys = {}

        # User profile
        user = self.db.query_user(user_id)
        cache_keys[f"user:{user_id}"] = user

        # User's recent posts
        posts = self.db.query_user_posts(user_id, limit=10)
        cache_keys[f"user:{user_id}:posts"] = posts

        # User's friends
        friends = self.db.query_user_friends(user_id)
        cache_keys[f"user:{user_id}:friends"] = friends

        self.cache.set_many(cache_keys, expire=1800)

    def scheduled_warming(self, interval: int = 300):
        """Periodically warm cache."""
        import schedule

        schedule.every(interval).seconds.do(self.warm_popular_items)

        while True:
            schedule.run_pending()
            time.sleep(1)
```

### Multi-Level Caching

```python
class MultiLevelCache:
    """Multi-level caching with L1 (local) and L2 (Memcached)."""

    def __init__(self, memcached_client, l1_size: int = 1000):
        self.l2_cache = memcached_client  # Memcached (shared)
        self.l1_cache = {}  # Local memory (per-process)
        self.l1_size = l1_size
        self.l1_access = {}  # Track access for LRU

    def get(self, key: str):
        """Get with L1 -> L2 -> database fallback."""
        # Check L1 cache (fastest)
        if key in self.l1_cache:
            self.l1_access[key] = time.time()
            return self.l1_cache[key]

        # Check L2 cache (Memcached)
        value = self.l2_cache.get(key)
        if value is not None:
            # Promote to L1
            self._set_l1(key, value)
            return value

        return None

    def set(self, key: str, value, expire: int = 0):
        """Set in both L1 and L2 caches."""
        # Set in L2 (Memcached)
        self.l2_cache.set(key, value, expire=expire)

        # Set in L1 (local memory)
        self._set_l1(key, value)

    def _set_l1(self, key: str, value):
        """Set in L1 cache with LRU eviction."""
        # Evict if L1 is full
        if len(self.l1_cache) >= self.l1_size:
            # Remove least recently used
            lru_key = min(self.l1_access, key=self.l1_access.get)
            del self.l1_cache[lru_key]
            del self.l1_access[lru_key]

        self.l1_cache[key] = value
        self.l1_access[key] = time.time()

    def delete(self, key: str):
        """Delete from both caches."""
        self.l2_cache.delete(key)
        self.l1_cache.pop(key, None)
        self.l1_access.pop(key, None)

    def clear_l1(self):
        """Clear L1 cache only."""
        self.l1_cache.clear()
        self.l1_access.clear()
```

## 6. Consistent Hashing and Distribution

### Consistent Hashing Implementation

```python
import hashlib
import bisect
from typing import List

class ConsistentHash:
    """Consistent hashing for distributed caching."""

    def __init__(self, servers: List[str], virtual_nodes: int = 150):
        """Initialize consistent hash ring.

        Args:
            servers: List of server addresses
            virtual_nodes: Number of virtual nodes per server
        """
        self.servers = servers
        self.virtual_nodes = virtual_nodes
        self.ring = {}
        self.sorted_keys = []

        self._build_ring()

    def _hash(self, key: str) -> int:
        """Hash function using MD5."""
        return int(hashlib.md5(key.encode('utf-8')).hexdigest(), 16)

    def _build_ring(self):
        """Build hash ring with virtual nodes."""
        for server in self.servers:
            for i in range(self.virtual_nodes):
                virtual_key = f"{server}:{i}"
                hash_val = self._hash(virtual_key)
                self.ring[hash_val] = server
                bisect.insort(self.sorted_keys, hash_val)

        print(f"Built hash ring with {len(self.ring)} virtual nodes")

    def get_server(self, key: str) -> str:
        """Get server for a given key.

        Args:
            key: Cache key

        Returns:
            Server address
        """
        if not self.ring:
            return None

        hash_val = self._hash(key)

        # Find first server >= hash_val
        idx = bisect.bisect(self.sorted_keys, hash_val)

        # Wrap around to first server
        if idx == len(self.sorted_keys):
            idx = 0

        return self.ring[self.sorted_keys[idx]]

    def add_server(self, server: str):
        """Add new server to ring."""
        self.servers.append(server)
        for i in range(self.virtual_nodes):
            virtual_key = f"{server}:{i}"
            hash_val = self._hash(virtual_key)
            self.ring[hash_val] = server
            bisect.insort(self.sorted_keys, hash_val)

        print(f"Added server {server}")

    def remove_server(self, server: str):
        """Remove server from ring."""
        self.servers.remove(server)

        # Remove all virtual nodes for this server
        keys_to_remove = [k for k, v in self.ring.items() if v == server]
        for key in keys_to_remove:
            del self.ring[key]
            self.sorted_keys.remove(key)

        print(f"Removed server {server}")

    def get_distribution(self, num_keys: int = 10000) -> dict:
        """Analyze key distribution across servers.

        Args:
            num_keys: Number of test keys

        Returns:
            Distribution statistics
        """
        distribution = {server: 0 for server in self.servers}

        for i in range(num_keys):
            key = f"test_key_{i}"
            server = self.get_server(key)
            distribution[server] += 1

        # Calculate percentages
        stats = {}
        for server, count in distribution.items():
            percentage = (count / num_keys) * 100
            stats[server] = {
                'count': count,
                'percentage': f"{percentage:.2f}%"
            }

        return stats


# Usage
servers = [
    '192.168.1.1:11211',
    '192.168.1.2:11211',
    '192.168.1.3:11211'
]

ch = ConsistentHash(servers, virtual_nodes=150)

# Get server for key
server = ch.get_server('user:123')
print(f"Key 'user:123' maps to {server}")

# Analyze distribution
stats = ch.get_distribution(10000)
print(f"Distribution: {stats}")

# Add new server (minimal redistribution)
ch.add_server('192.168.1.4:11211')
```

### Distributed Memcached Client

```python
from pymemcache.client.base import Client

class DistributedMemcachedClient:
    """Distributed Memcached client with consistent hashing."""

    def __init__(self, servers: List[str], virtual_nodes: int = 150):
        """Initialize distributed client.

        Args:
            servers: List of Memcached server addresses
            virtual_nodes: Virtual nodes per server for consistent hashing
        """
        self.consistent_hash = ConsistentHash(servers, virtual_nodes)
        self.clients = {}

        # Create connection pool for each server
        for server in servers:
            host, port = server.split(':')
            self.clients[server] = Client((host, int(port)))

    def _get_client(self, key: str) -> Client:
        """Get Memcached client for key."""
        server = self.consistent_hash.get_server(key)
        return self.clients[server]

    def set(self, key: str, value, expire: int = 0) -> bool:
        """Set key-value pair."""
        client = self._get_client(key)
        return client.set(key, value, expire=expire)

    def get(self, key: str):
        """Get value by key."""
        client = self._get_client(key)
        return client.get(key)

    def get_many(self, keys: List[str]) -> dict:
        """Get multiple values (grouped by server)."""
        # Group keys by server
        server_keys = {}
        for key in keys:
            server = self.consistent_hash.get_server(key)
            if server not in server_keys:
                server_keys[server] = []
            server_keys[server].append(key)

        # Fetch from each server
        results = {}
        for server, key_list in server_keys.items():
            client = self.clients[server]
            server_results = client.get_many(key_list)
            results.update(server_results)

        return results

    def delete(self, key: str) -> bool:
        """Delete key."""
        client = self._get_client(key)
        return client.delete(key)

    def close_all(self):
        """Close all connections."""
        for client in self.clients.values():
            client.close()


# Usage
servers = ['192.168.1.1:11211', '192.168.1.2:11211', '192.168.1.3:11211']
cache = DistributedMemcachedClient(servers)

# Set/Get distributed across servers
cache.set('user:123', {'name': 'Alice'}, expire=3600)
cache.set('user:456', {'name': 'Bob'}, expire=3600)
cache.set('user:789', {'name': 'Charlie'}, expire=3600)

# Each key automatically routed to appropriate server
user = cache.get('user:123')
print(f"User: {user}")

# Batch get across multiple servers
users = cache.get_many(['user:123', 'user:456', 'user:789'])
print(f"Users: {users}")

cache.close_all()
```

## 7. Performance Optimization

### Connection Pooling

```python
import queue
import threading
from contextlib import contextmanager

class MemcachedPool:
    """Connection pool for Memcached clients."""

    def __init__(self, host: str, port: int, pool_size: int = 10):
        """Initialize connection pool.

        Args:
            host: Memcached host
            port: Memcached port
            pool_size: Maximum number of connections
        """
        self.host = host
        self.port = port
        self.pool_size = pool_size
        self.pool = queue.Queue(maxsize=pool_size)
        self.lock = threading.Lock()

        # Pre-create connections
        for _ in range(pool_size):
            client = Client((host, port))
            self.pool.put(client)

    @contextmanager
    def connection(self):
        """Get connection from pool."""
        client = self.pool.get(timeout=5)
        try:
            yield client
        finally:
            self.pool.put(client)

    def execute(self, func, *args, **kwargs):
        """Execute function with pooled connection."""
        with self.connection() as client:
            return func(client, *args, **kwargs)

    def close_all(self):
        """Close all connections."""
        while not self.pool.empty():
            client = self.pool.get()
            client.close()


# Usage
pool = MemcachedPool('localhost', 11211, pool_size=20)

# Execute operations with pooled connections
def set_value(client, key, value):
    return client.set(key, value)

def get_value(client, key):
    return client.get(key)

pool.execute(set_value, 'user:123', {'name': 'Alice'})
user = pool.execute(get_value, 'user:123')
```

### Batch Operations

```python
class BatchMemcached:
    """Optimized batch operations for Memcached."""

    def __init__(self, client):
        self.client = client

    def batch_set(self, items: dict, expire: int = 0, chunk_size: int = 100):
        """Batch set operations.

        Args:
            items: Dictionary of key-value pairs
            expire: Expiration time
            chunk_size: Number of items per batch
        """
        keys = list(items.keys())
        for i in range(0, len(keys), chunk_size):
            chunk_keys = keys[i:i + chunk_size]
            chunk_data = {k: items[k] for k in chunk_keys}
            self.client.set_many(chunk_data, expire=expire)

    def batch_get(self, keys: List[str], chunk_size: int = 100) -> dict:
        """Batch get operations.

        Args:
            keys: List of keys to fetch
            chunk_size: Number of keys per batch

        Returns:
            Dictionary of results
        """
        results = {}

        for i in range(0, len(keys), chunk_size):
            chunk_keys = keys[i:i + chunk_size]
            chunk_results = self.client.get_many(chunk_keys)
            results.update(chunk_results)

        return results

    def batch_delete(self, keys: List[str], chunk_size: int = 100):
        """Batch delete operations.

        Args:
            keys: List of keys to delete
            chunk_size: Number of keys per batch
        """
        for i in range(0, len(keys), chunk_size):
            chunk_keys = keys[i:i + chunk_size]
            self.client.delete_many(chunk_keys)
```

### Key Compression

```python
import zlib
import pickle

class CompressedMemcached:
    """Memcached client with automatic compression."""

    def __init__(self, client, compression_threshold: int = 1024):
        """Initialize with compression.

        Args:
            client: Memcached client
            compression_threshold: Compress values larger than this (bytes)
        """
        self.client = client
        self.threshold = compression_threshold

    def set(self, key: str, value, expire: int = 0) -> bool:
        """Set with automatic compression."""
        # Serialize
        serialized = pickle.dumps(value)

        # Compress if larger than threshold
        if len(serialized) > self.threshold:
            compressed = zlib.compress(serialized)
            # Store with compression flag
            data = (1, compressed)  # Flag=1 means compressed
        else:
            data = (0, serialized)  # Flag=0 means uncompressed

        return self.client.set(key, data, expire=expire)

    def get(self, key: str):
        """Get with automatic decompression."""
        data = self.client.get(key)
        if data is None:
            return None

        flag, content = data

        # Decompress if needed
        if flag == 1:
            content = zlib.decompress(content)

        # Deserialize
        return pickle.loads(content)
```

## 8. Monitoring and Statistics

### Statistics Collection

```python
import time
from typing import Dict

class MemcachedMonitor:
    """Monitor Memcached performance and statistics."""

    def __init__(self, client):
        self.client = client

    def get_stats(self) -> Dict:
        """Get comprehensive statistics."""
        stats = self.client.stats()

        # Parse important metrics
        metrics = {
            'uptime': int(stats.get(b'uptime', 0)),
            'current_connections': int(stats.get(b'curr_connections', 0)),
            'total_connections': int(stats.get(b'total_connections', 0)),
            'current_items': int(stats.get(b'curr_items', 0)),
            'total_items': int(stats.get(b'total_items', 0)),
            'bytes_used': int(stats.get(b'bytes', 0)),
            'bytes_limit': int(stats.get(b'limit_maxbytes', 0)),
            'get_hits': int(stats.get(b'get_hits', 0)),
            'get_misses': int(stats.get(b'get_misses', 0)),
            'evictions': int(stats.get(b'evictions', 0)),
            'bytes_read': int(stats.get(b'bytes_read', 0)),
            'bytes_written': int(stats.get(b'bytes_written', 0)),
        }

        # Calculate derived metrics
        total_gets = metrics['get_hits'] + metrics['get_misses']
        if total_gets > 0:
            metrics['hit_rate'] = (metrics['get_hits'] / total_gets) * 100
        else:
            metrics['hit_rate'] = 0

        if metrics['bytes_limit'] > 0:
            metrics['memory_usage_pct'] = (metrics['bytes_used'] / metrics['bytes_limit']) * 100
        else:
            metrics['memory_usage_pct'] = 0

        return metrics

    def print_stats(self):
        """Print formatted statistics."""
        stats = self.get_stats()

        print("=" * 60)
        print("Memcached Statistics")
        print("=" * 60)
        print(f"Uptime: {stats['uptime']} seconds ({stats['uptime'] / 3600:.1f} hours)")
        print(f"Current connections: {stats['current_connections']}")
        print(f"Total connections: {stats['total_connections']}")
        print(f"\nMemory:")
        print(f"  Used: {stats['bytes_used'] / (1024**2):.2f} MB")
        print(f"  Limit: {stats['bytes_limit'] / (1024**2):.2f} MB")
        print(f"  Usage: {stats['memory_usage_pct']:.2f}%")
        print(f"\nItems:")
        print(f"  Current: {stats['current_items']}")
        print(f"  Total: {stats['total_items']}")
        print(f"  Evictions: {stats['evictions']}")
        print(f"\nCache Performance:")
        print(f"  Hits: {stats['get_hits']}")
        print(f"  Misses: {stats['get_misses']}")
        print(f"  Hit Rate: {stats['hit_rate']:.2f}%")
        print(f"\nNetwork:")
        print(f"  Bytes Read: {stats['bytes_read'] / (1024**2):.2f} MB")
        print(f"  Bytes Written: {stats['bytes_written'] / (1024**2):.2f} MB")
        print("=" * 60)

    def monitor_continuous(self, interval: int = 60):
        """Continuously monitor statistics.

        Args:
            interval: Monitoring interval in seconds
        """
        import schedule

        def log_stats():
            stats = self.get_stats()
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S')

            log_line = (
                f"{timestamp} | "
                f"Hit Rate: {stats['hit_rate']:.2f}% | "
                f"Memory: {stats['memory_usage_pct']:.2f}% | "
                f"Items: {stats['current_items']} | "
                f"Evictions: {stats['evictions']} | "
                f"Connections: {stats['current_connections']}"
            )
            print(log_line)

            # Alert on high eviction rate
            if stats['evictions'] > 1000:
                print(f"WARNING: High eviction rate ({stats['evictions']})")

            # Alert on low hit rate
            if stats['hit_rate'] < 80:
                print(f"WARNING: Low hit rate ({stats['hit_rate']:.2f}%)")

        schedule.every(interval).seconds.do(log_stats)

        print(f"Starting continuous monitoring (interval: {interval}s)")
        while True:
            schedule.run_pending()
            time.sleep(1)


# Usage
from pymemcache.client.base import Client

client = Client(('localhost', 11211))
monitor = MemcachedMonitor(client)

# One-time stats
monitor.print_stats()

# Continuous monitoring
# monitor.monitor_continuous(interval=60)
```

### Performance Metrics

```python
import time
from dataclasses import dataclass
from typing import Optional

@dataclass
class PerformanceMetrics:
    """Cache performance metrics."""
    requests: int = 0
    hits: int = 0
    misses: int = 0
    total_latency: float = 0.0
    min_latency: Optional[float] = None
    max_latency: Optional[float] = None

    @property
    def hit_rate(self) -> float:
        """Calculate hit rate percentage."""
        if self.requests == 0:
            return 0.0
        return (self.hits / self.requests) * 100

    @property
    def miss_rate(self) -> float:
        """Calculate miss rate percentage."""
        return 100 - self.hit_rate

    @property
    def avg_latency(self) -> float:
        """Calculate average latency in milliseconds."""
        if self.requests == 0:
            return 0.0
        return (self.total_latency / self.requests) * 1000


class InstrumentedCache:
    """Cache client with performance instrumentation."""

    def __init__(self, client):
        self.client = client
        self.metrics = PerformanceMetrics()

    def get(self, key: str):
        """Get with latency tracking."""
        start = time.time()
        value = self.client.get(key)
        latency = time.time() - start

        self.metrics.requests += 1
        self.metrics.total_latency += latency

        if value is not None:
            self.metrics.hits += 1
        else:
            self.metrics.misses += 1

        # Track min/max latency
        if self.metrics.min_latency is None or latency < self.metrics.min_latency:
            self.metrics.min_latency = latency
        if self.metrics.max_latency is None or latency > self.metrics.max_latency:
            self.metrics.max_latency = latency

        return value

    def set(self, key: str, value, expire: int = 0):
        """Set with latency tracking."""
        start = time.time()
        result = self.client.set(key, value, expire=expire)
        latency = time.time() - start

        self.metrics.total_latency += latency

        return result

    def print_metrics(self):
        """Print performance metrics."""
        m = self.metrics

        print("\nPerformance Metrics:")
        print(f"  Requests: {m.requests}")
        print(f"  Hits: {m.hits}")
        print(f"  Misses: {m.misses}")
        print(f"  Hit Rate: {m.hit_rate:.2f}%")
        print(f"  Miss Rate: {m.miss_rate:.2f}%")
        print(f"  Avg Latency: {m.avg_latency:.3f} ms")
        print(f"  Min Latency: {m.min_latency * 1000:.3f} ms")
        print(f"  Max Latency: {m.max_latency * 1000:.3f} ms")

    def reset_metrics(self):
        """Reset metrics."""
        self.metrics = PerformanceMetrics()
```

## 9. Security Best Practices

### Network Security

```bash
# Firewall rules (iptables)
# Only allow connections from application servers

# Allow from specific IP
sudo iptables -A INPUT -p tcp -s 192.168.1.100 --dport 11211 -j ACCEPT

# Allow from subnet
sudo iptables -A INPUT -p tcp -s 192.168.1.0/24 --dport 11211 -j ACCEPT

# Drop all other connections
sudo iptables -A INPUT -p tcp --dport 11211 -j DROP

# Save rules
sudo iptables-save > /etc/iptables/rules.v4
```

```bash
# UFW (Uncomplicated Firewall)
# Allow from specific IP
sudo ufw allow from 192.168.1.100 to any port 11211

# Deny all others
sudo ufw deny 11211
```

### Bind to Specific Interface

```bash
# Start Memcached bound to private interface only
memcached -l 10.0.1.5 -p 11211 -m 2048

# Or in /etc/memcached.conf
-l 10.0.1.5
```

### SASL Authentication

```bash
# Install SASL support
sudo apt-get install sasl2-bin libsasl2-modules

# Create SASL user
sudo saslpasswd2 -a memcached -c username

# Start Memcached with SASL
memcached -S -v -m 2048
```

```python
# Python client with SASL
from pymemcache.client.base import Client

client = Client(
    ('localhost', 11211),
    username='myuser',
    password='mypassword'
)
```

### Input Validation

```python
import re

class SecureMemcached:
    """Memcached client with security validations."""

    MAX_KEY_LENGTH = 250  # Memcached limit
    MAX_VALUE_SIZE = 1024 * 1024  # 1MB limit

    def __init__(self, client):
        self.client = client

    def _validate_key(self, key: str) -> bool:
        """Validate cache key.

        Rules:
        - No whitespace
        - No control characters
        - Max 250 characters
        """
        if not key or len(key) > self.MAX_KEY_LENGTH:
            raise ValueError(f"Invalid key length: {len(key)}")

        # Check for invalid characters
        if not re.match(r'^[^\s\x00-\x1f\x7f]+$', key):
            raise ValueError(f"Invalid characters in key: {key}")

        return True

    def _validate_value_size(self, value) -> bool:
        """Validate value size."""
        import sys

        value_size = sys.getsizeof(value)
        if value_size > self.MAX_VALUE_SIZE:
            raise ValueError(f"Value too large: {value_size} bytes")

        return True

    def set(self, key: str, value, expire: int = 0) -> bool:
        """Set with validation."""
        self._validate_key(key)
        self._validate_value_size(value)

        return self.client.set(key, value, expire=expire)

    def get(self, key: str):
        """Get with validation."""
        self._validate_key(key)

        return self.client.get(key)
```

### Rate Limiting

```python
import time
from collections import defaultdict

class RateLimitedCache:
    """Cache with rate limiting."""

    def __init__(self, client, max_requests_per_second: int = 100):
        self.client = client
        self.max_requests = max_requests_per_second
        self.request_times = defaultdict(list)

    def _check_rate_limit(self, ip: str) -> bool:
        """Check if IP is within rate limit."""
        now = time.time()
        minute_ago = now - 60

        # Clean old requests
        self.request_times[ip] = [t for t in self.request_times[ip] if t > minute_ago]

        # Check limit
        if len(self.request_times[ip]) >= self.max_requests:
            return False

        # Add current request
        self.request_times[ip].append(now)
        return True

    def get(self, key: str, client_ip: str):
        """Get with rate limiting."""
        if not self._check_rate_limit(client_ip):
            raise Exception(f"Rate limit exceeded for {client_ip}")

        return self.client.get(key)
```

## 10. High Availability Patterns

### Active-Active with Client Failover

```python
from typing import List
import random

class FailoverMemcached:
    """Memcached client with automatic failover."""

    def __init__(self, servers: List[tuple], retry_timeout: int = 2):
        """Initialize with multiple servers.

        Args:
            servers: List of (host, port) tuples
            retry_timeout: Seconds before retrying failed server
        """
        self.servers = servers
        self.retry_timeout = retry_timeout
        self.clients = {}
        self.failed_servers = {}

        for host, port in servers:
            self.clients[(host, port)] = Client((host, port))

    def _get_available_servers(self) -> List[tuple]:
        """Get list of available servers."""
        now = time.time()
        available = []

        for server in self.servers:
            # Check if server recently failed
            if server in self.failed_servers:
                if now - self.failed_servers[server] < self.retry_timeout:
                    continue
                else:
                    # Retry timeout expired
                    del self.failed_servers[server]

            available.append(server)

        return available

    def _mark_failed(self, server: tuple):
        """Mark server as failed."""
        self.failed_servers[server] = time.time()
        print(f"Marked server {server} as failed")

    def get(self, key: str):
        """Get with automatic failover."""
        available = self._get_available_servers()

        if not available:
            raise Exception("No available servers")

        # Try servers in order
        for server in available:
            try:
                client = self.clients[server]
                return client.get(key)
            except Exception as e:
                print(f"Server {server} failed: {e}")
                self._mark_failed(server)
                continue

        return None

    def set(self, key: str, value, expire: int = 0) -> bool:
        """Set with automatic failover."""
        available = self._get_available_servers()

        if not available:
            raise Exception("No available servers")

        # Try first available server
        for server in available:
            try:
                client = self.clients[server]
                return client.set(key, value, expire=expire)
            except Exception as e:
                print(f"Server {server} failed: {e}")
                self._mark_failed(server)
                continue

        return False


# Usage
servers = [
    ('192.168.1.1', 11211),
    ('192.168.1.2', 11211),
    ('192.168.1.3', 11211)
]

cache = FailoverMemcached(servers, retry_timeout=5)

# Automatic failover on server failure
cache.set('user:123', {'name': 'Alice'})
user = cache.get('user:123')
```

### Replication with mcrouter

```yaml
# mcrouter configuration
{
  "pools": {
    "main_pool": {
      "servers": [
        "192.168.1.1:11211",
        "192.168.1.2:11211",
        "192.168.1.3:11211"
      ]
    }
  },
  "route": {
    "type": "OperationSelectorRoute",
    "operation_policies": {
      "add": "AllSyncRoute|Pool|main_pool",
      "delete": "AllSyncRoute|Pool|main_pool",
      "get": "LatestRoute|Pool|main_pool",
      "set": "AllSyncRoute|Pool|main_pool"
    }
  }
}
```

```bash
# Start mcrouter
mcrouter -p 11211 -f config.json
```

## 11. Production Deployment

### Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  memcached-1:
    image: memcached:alpine
    container_name: memcached-1
    ports:
      - "11211:11211"
    command: memcached -m 512 -c 1024 -t 4 -v
    restart: unless-stopped
    networks:
      - cache-network

  memcached-2:
    image: memcached:alpine
    container_name: memcached-2
    ports:
      - "11212:11211"
    command: memcached -m 512 -c 1024 -t 4 -v
    restart: unless-stopped
    networks:
      - cache-network

  memcached-3:
    image: memcached:alpine
    container_name: memcached-3
    ports:
      - "11213:11211"
    command: memcached -m 512 -c 1024 -t 4 -v
    restart: unless-stopped
    networks:
      - cache-network

  # Optional: mcrouter for routing
  mcrouter:
    image: denji/mcrouter:latest
    container_name: mcrouter
    ports:
      - "5000:5000"
    volumes:
      - ./mcrouter-config.json:/etc/mcrouter/config.json
    command: mcrouter -p 5000 -f /etc/mcrouter/config.json
    depends_on:
      - memcached-1
      - memcached-2
      - memcached-3
    networks:
      - cache-network

networks:
  cache-network:
    driver: bridge
```

### Kubernetes Deployment

```yaml
# memcached-statefulset.yaml
apiVersion: v1
kind: Service
metadata:
  name: memcached
spec:
  clusterIP: None
  selector:
    app: memcached
  ports:
  - port: 11211
    name: memcached
---
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: memcached
spec:
  serviceName: "memcached"
  replicas: 3
  selector:
    matchLabels:
      app: memcached
  template:
    metadata:
      labels:
        app: memcached
    spec:
      containers:
      - name: memcached
        image: memcached:alpine
        ports:
        - containerPort: 11211
          name: memcached
        command:
        - memcached
        - -m
        - "512"
        - -c
        - "1024"
        - -t
        - "4"
        - -v
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "1Gi"
            cpu: "1"
        livenessProbe:
          tcpSocket:
            port: 11211
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          tcpSocket:
            port: 11211
          initialDelaySeconds: 5
          periodSeconds: 5
```

### Systemd Service

```ini
# /etc/systemd/system/memcached.service
[Unit]
Description=Memcached
After=network.target

[Service]
Type=simple
User=memcache
Group=memcache
ExecStart=/usr/bin/memcached \
    -m 2048 \
    -p 11211 \
    -c 2048 \
    -t 4 \
    -u memcache \
    -l 10.0.1.5 \
    -L \
    -I 5m \
    -o modern \
    -v

Restart=always
RestartSec=5

# Security
PrivateTmp=yes
ProtectSystem=full
NoNewPrivileges=true

[Install]
WantedBy=multi-user.target
```

```bash
# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable memcached
sudo systemctl start memcached
sudo systemctl status memcached
```

## 12. Common Anti-Patterns

### Anti-Pattern: Storing Sessions in Memcached

```python
# ❌ BAD: Using Memcached for critical session data
def store_session_bad(cache, session_id, session_data):
    """DON'T store sessions in Memcached - data can be evicted!"""
    cache.set(f"session:{session_id}", session_data, expire=3600)
    # Problem: Data can be evicted due to memory pressure
    # User gets logged out unexpectedly

# ✅ GOOD: Use Redis or database for sessions
def store_session_good(redis, session_id, session_data):
    """Store sessions in Redis (persistent) or database."""
    redis.setex(f"session:{session_id}", 3600, session_data)
    # Redis won't evict by default, provides persistence
```

### Anti-Pattern: Caching Large Objects

```python
# ❌ BAD: Caching very large objects
def cache_large_object_bad(cache):
    large_report = generate_huge_report()  # 5MB object
    cache.set('report:daily', large_report)
    # Problems:
    # - Wastes memory
    # - Network transfer overhead
    # - Memcached has 1MB default limit

# ✅ GOOD: Cache summary or pointer
def cache_large_object_good(cache, file_storage):
    large_report = generate_huge_report()

    # Store full report in S3/filesystem
    file_path = file_storage.save('reports/daily.json', large_report)

    # Cache only metadata/summary
    cache.set('report:daily', {
        'path': file_path,
        'generated_at': time.time(),
        'summary': generate_summary(large_report)
    }, expire=3600)
```

### Anti-Pattern: Cache Stampede

```python
# ❌ BAD: Cache stampede (thundering herd)
def get_popular_item_bad(cache, db, item_id):
    """Multiple requests hit database simultaneously on cache miss."""
    cache_key = f"item:{item_id}"

    item = cache.get(cache_key)
    if item is None:
        # 1000 concurrent requests all miss cache
        # All 1000 hit database simultaneously
        item = db.query_item(item_id)
        cache.set(cache_key, item, expire=3600)

    return item

# ✅ GOOD: Use locking to prevent stampede
import threading

cache_locks = {}

def get_popular_item_good(cache, db, item_id):
    """Use lock to ensure only one request fetches from database."""
    cache_key = f"item:{item_id}"

    item = cache.get(cache_key)
    if item is not None:
        return item

    # Acquire lock for this key
    if cache_key not in cache_locks:
        cache_locks[cache_key] = threading.Lock()

    with cache_locks[cache_key]:
        # Double-check after acquiring lock
        item = cache.get(cache_key)
        if item is not None:
            return item

        # Only one thread reaches here
        item = db.query_item(item_id)
        cache.set(cache_key, item, expire=3600)

    return item

# ✅ BETTER: Probabilistic early expiration
import random

def get_with_early_expiration(cache, db, item_id, ttl=3600, beta=1.0):
    """Refresh cache probabilistically before expiration."""
    cache_key = f"item:{item_id}"
    cache_data = cache.get(cache_key)  # Assume stored as (value, timestamp)

    if cache_data is not None:
        value, stored_at = cache_data
        elapsed = time.time() - stored_at

        # XFetch: Probabilistic early refresh
        # https://en.wikipedia.org/wiki/Cache_stampede#Probabilistic_early_expiration
        if elapsed >= ttl * beta * -1 * math.log(random.random()):
            # Refresh cache
            value = db.query_item(item_id)
            cache.set(cache_key, (value, time.time()), expire=ttl)

        return value

    # Cache miss
    value = db.query_item(item_id)
    cache.set(cache_key, (value, time.time()), expire=ttl)
    return value
```

## 13. Troubleshooting Guide

### Common Issues

```python
def diagnose_memcached(client):
    """Diagnose common Memcached issues."""

    stats = client.stats()

    # Issue 1: High eviction rate
    evictions = int(stats.get(b'evictions', 0))
    if evictions > 1000:
        print(f"⚠ HIGH EVICTIONS: {evictions}")
        print("  Solutions:")
        print("  - Increase memory (-m parameter)")
        print("  - Reduce TTL values")
        print("  - Add more Memcached servers")
        print("  - Review what's being cached")

    # Issue 2: Low hit rate
    hits = int(stats.get(b'get_hits', 0))
    misses = int(stats.get(b'get_misses', 0))
    total = hits + misses

    if total > 0:
        hit_rate = (hits / total) * 100
        if hit_rate < 70:
            print(f"⚠ LOW HIT RATE: {hit_rate:.2f}%")
            print("  Solutions:")
            print("  - Increase TTL values")
            print("  - Warm cache on startup")
            print("  - Review cache key strategy")

    # Issue 3: Connection issues
    curr_conn = int(stats.get(b'curr_connections', 0))
    max_conn = 1024  # From -c parameter

    if curr_conn > max_conn * 0.8:
        print(f"⚠ HIGH CONNECTION COUNT: {curr_conn}/{max_conn}")
        print("  Solutions:")
        print("  - Increase max connections (-c parameter)")
        print("  - Use connection pooling")
        print("  - Check for connection leaks")

    # Issue 4: Memory usage
    bytes_used = int(stats.get(b'bytes', 0))
    bytes_limit = int(stats.get(b'limit_maxbytes', 0))
    mem_pct = (bytes_used / bytes_limit) * 100

    if mem_pct > 90:
        print(f"⚠ HIGH MEMORY USAGE: {mem_pct:.2f}%")
        print("  Solutions:")
        print("  - Increase memory allocation")
        print("  - Reduce TTL values")
        print("  - Clear unused keys")

    print("\n✓ Diagnostics complete")
```

### Testing Connection

```bash
# Test Memcached connection
echo -e "stats\nquit" | nc localhost 11211

# Test set/get
printf "set test 0 3600 5\r\nhello\r\n" | nc localhost 11211
printf "get test\r\n" | nc localhost 11211

# Telnet (interactive)
telnet localhost 11211
stats
get mykey
set mykey 0 3600 5
hello
quit
```

### Debug Mode

```bash
# Start Memcached in verbose mode
memcached -vv -m 64 -p 11211

# Watch logs
tail -f /var/log/memcached.log
```

## 14. Performance Tuning Checklist

```markdown
**System Configuration:**
- [ ] Increase file descriptor limit (ulimit -n 65536)
- [ ] Use large memory pages (-L flag)
- [ ] Bind to specific CPUs (taskset)
- [ ] Disable swap (sysctl vm.swappiness=0)
- [ ] Use SSD for temporary storage

**Memcached Configuration:**
- [ ] Set appropriate memory limit (-m)
- [ ] Configure thread count (-t) = CPU cores
- [ ] Increase max connections (-c) as needed
- [ ] Set max item size (-I) appropriately
- [ ] Use binary protocol for efficiency
- [ ] Enable modern mode (-o modern)

**Application:**
- [ ] Use connection pooling
- [ ] Implement batch operations
- [ ] Set appropriate TTL values
- [ ] Use consistent hashing for distribution
- [ ] Compress large values
- [ ] Implement cache warming
- [ ] Monitor hit rates and adjust strategy

**Network:**
- [ ] Use local network (low latency)
- [ ] Monitor network saturation
- [ ] Consider dedicated cache network
- [ ] Use jumbo frames if possible

**Monitoring:**
- [ ] Track hit rate (>80% target)
- [ ] Monitor eviction rate
- [ ] Watch memory usage
- [ ] Alert on high connection count
- [ ] Monitor latency (p50, p95, p99)
```

## 15. Comparison with Alternatives

### Memcached vs Redis

| Feature | Memcached | Redis |
|---------|-----------|-------|
| **Data Structures** | Key-value only | Strings, lists, sets, hashes, sorted sets, etc. |
| **Persistence** | None (volatile) | Yes (RDB, AOF) |
| **Replication** | No (client-side sharding) | Yes (master-replica) |
| **Threading** | Multi-threaded | Single-threaded (with I/O threads) |
| **Use Case** | Simple cache | Cache + data structures + pub/sub |
| **Memory Efficiency** | Better for simple K-V | More features, slightly more overhead |
| **Speed** | Slightly faster for simple ops | Fast, more versatile |
| **Clustering** | Client-side | Built-in (Redis Cluster) |
| **Complex Queries** | No | Yes (Lua scripts, modules) |

**Choose Memcached when:**
- Simple key-value caching
- Maximum throughput
- Multi-core scaling
- Temporary cache only (no persistence needed)

**Choose Redis when:**
- Need persistence
- Complex data structures
- Built-in replication
- Pub/sub messaging
- Session storage

## 16. Deployment Checklist

### Agent-Generated Code Verification (MANDATORY)

#### Build & Compilation
- [ ] Code compiles/runs without errors
- [ ] All imports/dependencies resolved (Memcached client libraries)
- [ ] Code formatted per project standards

#### Testing
- [ ] All tests pass
- [ ] Coverage meets minimum threshold (>80%)
- [ ] Integration tests pass against Memcached test instance

#### Security
- [ ] Dependency scan: 0 HIGH/CRITICAL vulnerabilities
- [ ] No hardcoded credentials or secrets
- [ ] Connection strings use environment variables

#### Agent Workflow Completed
- [ ] Agent verified code builds successfully
- [ ] Agent ran all tests and verified they pass
- [ ] Agent verified documentation

---

## 17. Why This Configuration Works

**In-Memory Hash Table for Sub-Millisecond Latency**: Memcached stores all data in RAM using an efficient slab allocator and hash table, delivering consistent sub-millisecond response times regardless of dataset size.

**Consistent Hashing for Horizontal Scaling**: Client-side consistent hashing distributes keys across servers with minimal disruption when nodes are added or removed, enabling linear scaling of cache capacity.

**Protocol Simplicity Ensures Reliability**: The text and binary protocols are intentionally minimal (get/set/delete), making the server extremely stable and predictable under load with virtually no edge cases.

**LRU Eviction Provides Self-Managing Cache**: Automatic least-recently-used eviction means the cache manages its own memory without manual intervention, gracefully handling capacity limits without crashing or blocking.

---

## 18. Quick Reference

### Common Commands

```bash
# Start Memcached with 1GB memory on default port
memcached -m 1024 -p 11211 -d

# Start with verbose logging for debugging
memcached -m 256 -p 11211 -vv

# Connect with telnet for manual testing
telnet localhost 11211

# Set a value (via telnet)
# set mykey 0 3600 5\r\nhello\r\n

# Get a value (via telnet)
# get mykey\r\n

# View server stats (via telnet)
# stats\r\n

# Flush all cached data
# flush_all\r\n

# Check stats via command line
echo "stats" | nc localhost 11211
```

---

## 19. Resources and References

### Official Documentation
- **Memcached Wiki**: https://github.com/memcached/memcached/wiki
- **Protocol Specification**: https://github.com/memcached/memcached/blob/master/doc/protocol.txt
- **Best Practices**: https://github.com/memcached/memcached/wiki/ConfiguringServer

### Client Libraries
- **Python**: pymemcache, python-memcached
- **Node.js**: memcached, memjs
- **PHP**: memcached extension
- **Java**: spymemcached, xmemcached
- **Go**: gomemcache
- **Ruby**: dalli

### Tools
- **memcached-tool**: Stats and management
- **mcrouter**: Facebook's routing proxy
- **twemproxy**: Twitter's proxy
- **telnet/nc**: Manual testing

### Production Users
- **Facebook**: Massive scale (thousands of servers)
- **Twitter**: Cache layer
- **YouTube**: Video metadata caching
- **Wikipedia**: Page caching
- **Reddit**: Session and data caching

---

## Quick Start Example

```python
from pymemcache.client.base import Client

# Connect to Memcached
client = Client(('localhost', 11211))

# Set value
client.set('user:123', b'Alice', expire=3600)  # 1 hour

# Get value
user = client.get('user:123')
print(f"User: {user.decode()}")

# Increment counter
client.set('views', b'0')
client.incr('views', 1)
views = client.get('views')
print(f"Views: {views.decode()}")

# Batch operations
client.set_many({
    'key1': b'value1',
    'key2': b'value2',
    'key3': b'value3'
}, expire=3600)

values = client.get_many([b'key1', b'key2', b'key3'])
print(f"Values: {values}")

# Statistics
stats = client.stats()
print(f"Memcached stats: {stats}")

# Cleanup
client.close()
```

This guide provides comprehensive coverage of Memcached for modern production deployments, from basic operations to advanced patterns and optimization strategies.

---

**End of Memcached Development Guidelines**
