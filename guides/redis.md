# Redis Best Practices and Style Guide
Comprehensive guidelines for working with Redis effectively, covering data modeling, performance optimization, caching patterns, and operational best practices. This guide is language-agnostic. Redis 7.x+, Redis Stack, Redis Cluster, Redis Sentinel, Redis Streams, Lua scripting.

---

**Agent Profile**: The Redis Expert
**Role**: Senior Data Engineer & Cache Architect
**Objective**: Design and implement high-performance Redis solutions with proper data modeling, efficient memory usage, and production-ready reliability patterns.
**Tools**: Redis 7.x+, Redis Stack, Redis Cluster, Redis Sentinel, Redis Streams, Lua scripting.

---

## 1. Core Philosophies: REDIS-FIRST

The agent must adhere to the **REDIS-FIRST** principles for every Redis implementation:

- **R**ight Data Structure: Choose the optimal data structure for each use case
- **E**fficient Keys: Design keys for readability, scannability, and memory efficiency
- **D**ata Expiration: Always set TTLs; Redis is not a primary database
- **I**dempotent Operations: Design for retry safety and crash recovery
- **S**ingle-Purpose: Each Redis instance should serve a focused purpose

**Additional Principles:**

- **Memory is Precious**: Optimize for memory efficiency; monitor usage constantly
- **Network Matters**: Minimize round trips with pipelining and Lua scripts
- **Fail Gracefully**: Application must work (degraded) if Redis is unavailable
- **No Blocking**: Avoid commands that block the single-threaded event loop

---

## 2. Data Structures Selection

### A. Data Structure Overview

```
REDIS DATA STRUCTURES:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  STRUCTURE      │ USE CASE                      │ TIME COMPLEXITY       │
│  ───────────────┼───────────────────────────────┼─────────────────────  │
│                 │                               │                       │
│  STRING         │ Caching, counters, flags     │ O(1) get/set          │
│                 │ Session data, rate limiting   │                       │
│                 │                               │                       │
│  HASH           │ Objects with fields           │ O(1) per field        │
│                 │ User profiles, product data   │ O(N) for all fields   │
│                 │                               │                       │
│  LIST           │ Queues, recent items          │ O(1) push/pop ends    │
│                 │ Activity feeds, logs          │ O(N) index access     │
│                 │                               │                       │
│  SET            │ Unique collections            │ O(1) add/remove/check │
│                 │ Tags, unique visitors         │ O(N) for all members  │
│                 │                               │                       │
│  SORTED SET     │ Rankings, leaderboards        │ O(log N) add/remove   │
│  (ZSET)         │ Time-series, priority queues  │ O(log N + M) range    │
│                 │                               │                       │
│  STREAM         │ Event logs, message queues    │ O(1) append           │
│                 │ Activity streams, CDC         │ O(N) range read       │
│                 │                               │                       │
│  HYPERLOGLOG    │ Cardinality estimation        │ O(1) add/count        │
│                 │ Unique counts (approx)        │ 12KB fixed size       │
│                 │                               │                       │
│  BITMAP         │ Flags, presence tracking      │ O(1) bit operations   │
│                 │ Feature flags, daily active   │ Memory efficient      │
│                 │                               │                       │
│  GEOSPATIAL     │ Location-based queries        │ O(log N) add          │
│                 │ Nearby search, distance       │ O(N+log M) radius     │
│                 │                               │                       │
└─────────────────────────────────────────────────────────────────────────┘
```

### B. Data Structure Selection Guide

```
CHOOSING THE RIGHT DATA STRUCTURE:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  QUESTION                              │ DATA STRUCTURE                 │
│  ──────────────────────────────────────┼──────────────────────────────  │
│                                        │                                │
│  Simple key-value caching?             │ STRING                         │
│  Need atomic increment/decrement?      │ STRING (INCR/DECR)             │
│  Store JSON object?                    │ HASH (fields) or STRING (JSON) │
│  Object with many fields, partial read?│ HASH                           │
│  Need FIFO queue?                      │ LIST (LPUSH + RPOP)            │
│  Need LIFO stack?                      │ LIST (LPUSH + LPOP)            │
│  Recent N items?                       │ LIST (with LTRIM)              │
│  Unique members, no order?             │ SET                            │
│  Membership check needed?              │ SET                            │
│  Set operations (union, intersect)?    │ SET                            │
│  Ranked items / leaderboard?           │ SORTED SET                     │
│  Range queries by score?               │ SORTED SET                     │
│  Time-series data?                     │ SORTED SET or STREAM           │
│  Message queue with consumers?         │ STREAM                         │
│  Event sourcing / audit log?           │ STREAM                         │
│  Count unique items (approximate)?     │ HYPERLOGLOG                    │
│  Binary flags for millions of items?   │ BITMAP                         │
│  Location-based queries?               │ GEOSPATIAL                     │
│                                        │                                │
└─────────────────────────────────────────────────────────────────────────┘
```

### C. Data Structure Examples

```
DATA STRUCTURE USAGE EXAMPLES:

// ═══════════════════════════════════════════════════════════════════════
// STRING - Simple caching, counters, flags
// ═══════════════════════════════════════════════════════════════════════

// Cache a value with expiration
SET user:123:profile "{...json...}" EX 3600

// Atomic counter
INCR api:rate:user:123:minute:202401211530
EXPIRE api:rate:user:123:minute:202401211530 60

// Distributed lock (with NX = only if not exists)
SET lock:order:456 "owner-id" NX EX 30

// ═══════════════════════════════════════════════════════════════════════
// HASH - Objects with fields
// ═══════════════════════════════════════════════════════════════════════

// Store user profile
HSET user:123 name "John" email "john@example.com" plan "premium"

// Get single field
HGET user:123 email

// Get multiple fields
HMGET user:123 name email

// Increment numeric field
HINCRBY user:123 login_count 1

// ═══════════════════════════════════════════════════════════════════════
// LIST - Queues, recent items
// ═══════════════════════════════════════════════════════════════════════

// Queue: producer adds, consumer removes
LPUSH queue:emails "{email-job-1}"    // Add to queue
BRPOP queue:emails 30                  // Blocking pop (consumer)

// Recent items (keep last 100)
LPUSH user:123:recent_views "product:789"
LTRIM user:123:recent_views 0 99       // Keep only last 100

// ═══════════════════════════════════════════════════════════════════════
// SET - Unique collections, membership
// ═══════════════════════════════════════════════════════════════════════

// Track unique visitors
SADD daily:visitors:2024-01-21 "user:123" "user:456"

// Check membership
SISMEMBER daily:visitors:2024-01-21 "user:123"

// Set operations
SINTER daily:visitors:2024-01-21 daily:visitors:2024-01-20  // Both days
SUNION daily:visitors:2024-01-21 daily:visitors:2024-01-20  // Either day

// ═══════════════════════════════════════════════════════════════════════
// SORTED SET - Rankings, time-series
// ═══════════════════════════════════════════════════════════════════════

// Leaderboard (score = points)
ZADD leaderboard 1500 "player:alice" 1200 "player:bob" 1800 "player:carol"
ZREVRANGE leaderboard 0 9 WITHSCORES   // Top 10

// Time-series (score = timestamp)
ZADD user:123:events 1705849200 "{event1}" 1705849260 "{event2}"
ZRANGEBYSCORE user:123:events 1705849000 1705850000  // Events in time range

// Rate limiting sliding window
ZADD rate:user:123 1705849200.123 "req1" 1705849200.456 "req2"
ZREMRANGEBYSCORE rate:user:123 0 (NOW - 60)  // Remove old
ZCARD rate:user:123                           // Count in window

// ═══════════════════════════════════════════════════════════════════════
// STREAM - Event logs, message queues
// ═══════════════════════════════════════════════════════════════════════

// Add event
XADD orders:events * action "created" order_id "456" amount "99.99"

// Read new events (consumer group)
XREADGROUP GROUP processors consumer1 COUNT 10 STREAMS orders:events >

// Acknowledge processed
XACK orders:events processors 1705849200000-0

// ═══════════════════════════════════════════════════════════════════════
// HYPERLOGLOG - Approximate unique counts (12KB max)
// ═══════════════════════════════════════════════════════════════════════

// Count unique visitors (approximate, ~0.81% error)
PFADD unique:visitors:2024-01 "user:123" "user:456" "user:789"
PFCOUNT unique:visitors:2024-01

// Merge multiple periods
PFMERGE unique:visitors:2024-q1 unique:visitors:2024-01 \
        unique:visitors:2024-02 unique:visitors:2024-03
```

---

## 2A. Test-Driven Development (TDD) Protocol (MANDATORY)

**CRITICAL: Follow the Red-Green-Refactor cycle for ALL new Redis implementation code.**

### TDD Cycle

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│                      RED-GREEN-REFACTOR CYCLE                           │
│                                                                         │
│         ┌─────────┐                                                     │
│         │   RED   │  Write a failing test first                         │
│         │  (FAIL) │  Test should verify Redis behavior                  │
│         └────┬────┘                                                     │
│              │                                                          │
│              ▼                                                          │
│         ┌─────────┐                                                     │
│         │  GREEN  │  Write minimal code to make test pass               │
│         │ (PASS)  │  Implement Redis operation correctly                │
│         └────┬────┘                                                     │
│              │                                                          │
│              ▼                                                          │
│         ┌─────────┐                                                     │
│         │REFACTOR │  Improve code quality                               │
│         │ (PASS)  │  Optimize Redis usage, keep tests green             │
│         └────┬────┘                                                     │
│              │                                                          │
│              └──────────► Repeat for next feature                       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Example TDD Workflow for Redis Operations

**Scenario: Implementing a Rate Limiter with Sliding Window**

```
TDD WORKFLOW - RATE LIMITER IMPLEMENTATION:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  STEP 1: RED - Write Failing Test                                       │
│  ─────────────────────────────────                                      │
│                                                                         │
│  // Test: Rate limiter should allow requests under limit                │
│  function test_rate_limiter_allows_under_limit() {                      │
│      limiter = new RateLimiter(redis, limit=5, window=60)              │
│      userId = "user:123"                                                │
│                                                                         │
│      // First 5 requests should succeed                                 │
│      for i in 1..5:                                                     │
│          assert limiter.checkLimit(userId) == true                      │
│  }                                                                      │
│                                                                         │
│  // Run test:                                                           │
│  $ test_runner --run test_rate_limiter                                 │
│  ❌ FAILS - RateLimiter class does not exist                           │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  STEP 2: GREEN - Minimal Implementation                                 │
│  ──────────────────────────────────────                                 │
│                                                                         │
│  class RateLimiter {                                                    │
│      constructor(redis, limit, windowSeconds) {                         │
│          this.redis = redis                                             │
│          this.limit = limit                                             │
│          this.window = windowSeconds                                    │
│      }                                                                  │
│                                                                         │
│      function checkLimit(userId) {                                      │
│          key = "rate:" + userId                                         │
│          current = this.redis.INCR(key)                                │
│                                                                         │
│          if (current == 1) {                                            │
│              this.redis.EXPIRE(key, this.window)                       │
│          }                                                              │
│                                                                         │
│          return current <= this.limit                                   │
│      }                                                                  │
│  }                                                                      │
│                                                                         │
│  // Run test:                                                           │
│  $ test_runner --run test_rate_limiter                                 │
│  ✅ PASSES - Basic implementation works                                │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  STEP 3: RED - Add More Tests (Edge Cases)                              │
│  ─────────────────────────────────────────                              │
│                                                                         │
│  function test_rate_limiter_blocks_over_limit() {                       │
│      limiter = new RateLimiter(redis, limit=5, window=60)              │
│      userId = "user:456"                                                │
│                                                                         │
│      // Exhaust limit                                                   │
│      for i in 1..5:                                                     │
│          limiter.checkLimit(userId)                                     │
│                                                                         │
│      // 6th request should fail                                         │
│      assert limiter.checkLimit(userId) == false                         │
│  }                                                                      │
│                                                                         │
│  function test_rate_limiter_resets_after_window() {                     │
│      limiter = new RateLimiter(redis, limit=5, window=1)  // 1 sec     │
│      userId = "user:789"                                                │
│                                                                         │
│      // Exhaust limit                                                   │
│      for i in 1..5:                                                     │
│          limiter.checkLimit(userId)                                     │
│                                                                         │
│      // Wait for window to expire                                       │
│      sleep(1.1)                                                         │
│                                                                         │
│      // Should allow again                                              │
│      assert limiter.checkLimit(userId) == true                          │
│  }                                                                      │
│                                                                         │
│  // Run tests:                                                          │
│  $ test_runner --run test_rate_limiter                                 │
│  ✅ PASSES - All edge cases covered                                    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  STEP 4: REFACTOR - Improve Implementation                              │
│  ─────────────────────────────────────────                              │
│                                                                         │
│  class RateLimiter {                                                    │
│      constructor(redis, limit, windowSeconds) {                         │
│          this.redis = redis                                             │
│          this.limit = limit                                             │
│          this.window = windowSeconds                                    │
│          this.script = this._loadLuaScript()  // Atomic operation      │
│      }                                                                  │
│                                                                         │
│      function checkLimit(userId) {                                      │
│          key = "rate:" + userId + ":" + this._currentWindow()          │
│                                                                         │
│          // Use Lua script for atomicity                                │
│          result = this.redis.EVALSHA(                                  │
│              this.script,                                               │
│              1, key,                                                    │
│              this.limit, this.window                                    │
│          )                                                              │
│                                                                         │
│          return result == 1                                             │
│      }                                                                  │
│                                                                         │
│      function _currentWindow() {                                        │
│          return floor(now() / this.window)                             │
│      }                                                                  │
│                                                                         │
│      function _loadLuaScript() {                                        │
│          // Atomic increment with limit check                           │
│          return redis.SCRIPT_LOAD("""                                  │
│              local current = redis.call('INCR', KEYS[1])               │
│              if current == 1 then                                       │
│                  redis.call('EXPIRE', KEYS[1], ARGV[2])                │
│              end                                                        │
│              if current > tonumber(ARGV[1]) then                        │
│                  return 0                                               │
│              end                                                        │
│              return 1                                                   │
│          """)                                                           │
│      }                                                                  │
│  }                                                                      │
│                                                                         │
│  // Run all tests:                                                      │
│  $ test_runner --run test_rate_limiter                                 │
│  ✅ PASSES - Refactored code still works                               │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Visual Step-by-Step Example

```
TDD VISUAL FLOW - CACHE IMPLEMENTATION:

    TEST FIRST                    IMPLEMENT                     REFACTOR
    ──────────                    ─────────                     ────────

    ┌─────────────┐              ┌─────────────┐              ┌─────────────┐
    │ Write test: │              │ Implement:  │              │ Improve:    │
    │             │              │             │              │             │
    │ getCache()  │     →        │ GET key     │     →        │ Add TTL     │
    │ returns     │              │ return val  │              │ Add jitter  │
    │ cached val  │              │             │              │ Add metrics │
    └─────────────┘              └─────────────┘              └─────────────┘
          │                            │                            │
          ▼                            ▼                            ▼
       ❌ FAIL                      ✅ PASS                      ✅ PASS
    (no impl yet)              (minimal impl)               (optimized impl)


COMPLETE TDD CYCLE FOR CACHE-ASIDE PATTERN:

┌────────────────────────────────────────────────────────────────────────────┐
│                                                                            │
│  1. RED: Test cache miss behavior                                          │
│     ──────────────────────────────                                         │
│     test_cache_miss_fetches_from_database()                                │
│     → FAILS (no implementation)                                            │
│                                                                            │
│  2. GREEN: Implement cache-aside read                                      │
│     ────────────────────────────────                                       │
│     GET key → miss → query DB → SET key → return                          │
│     → PASSES                                                               │
│                                                                            │
│  3. RED: Test cache hit behavior                                           │
│     ─────────────────────────────                                          │
│     test_cache_hit_returns_cached_value()                                  │
│     → PASSES (already works)                                               │
│                                                                            │
│  4. RED: Test TTL behavior                                                 │
│     ──────────────────────                                                 │
│     test_cache_expires_after_ttl()                                         │
│     → FAILS (no TTL set)                                                   │
│                                                                            │
│  5. GREEN: Add TTL to SET                                                  │
│     ─────────────────────                                                  │
│     SETEX key ttl value                                                    │
│     → PASSES                                                               │
│                                                                            │
│  6. REFACTOR: Add TTL jitter to prevent stampede                           │
│     ──────────────────────────────────────────                             │
│     ttl = base_ttl + random(0, jitter)                                     │
│     → PASSES (all tests still green)                                       │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## 2B. Bug Fix Protocol (MANDATORY)

**CRITICAL: Every Redis bug MUST receive a regression test BEFORE fixing.**

### Bug Fix Workflow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│                        BUG FIX WORKFLOW                                 │
│                                                                         │
│  ┌──────────────────┐                                                   │
│  │  Bug Reported    │  "Cache returns stale data after update"         │
│  │  or Discovered   │                                                   │
│  └────────┬─────────┘                                                   │
│           │                                                             │
│           ▼                                                             │
│  ┌──────────────────┐                                                   │
│  │  Write Test That │  test_cache_invalidated_on_update()              │
│  │  REPRODUCES Bug  │  → Should FAIL (proving bug exists)              │
│  └────────┬─────────┘                                                   │
│           │                                                             │
│           ▼                                                             │
│  ┌──────────────────┐                                                   │
│  │  Verify Test     │  Confirm test fails for the RIGHT reason         │
│  │  Fails Correctly │  (stale data returned, not other error)          │
│  └────────┬─────────┘                                                   │
│           │                                                             │
│           ▼                                                             │
│  ┌──────────────────┐                                                   │
│  │  Fix the Bug     │  Add cache invalidation on update                │
│  │                  │  DEL cache_key after database update             │
│  └────────┬─────────┘                                                   │
│           │                                                             │
│           ▼                                                             │
│  ┌──────────────────┐                                                   │
│  │  Verify Test     │  test_cache_invalidated_on_update()              │
│  │  Now PASSES      │  → Should PASS (bug fixed)                       │
│  └────────┬─────────┘                                                   │
│           │                                                             │
│           ▼                                                             │
│  ┌──────────────────┐                                                   │
│  │  Run ALL Tests   │  Ensure no regressions introduced                │
│  │                  │  → All tests PASS                                │
│  └────────┬─────────┘                                                   │
│           │                                                             │
│           ▼                                                             │
│  ┌──────────────────┐                                                   │
│  │  Document Bug    │  Add bug ID and description to test              │
│  │  in Test         │  // Regression test for BUG-1234                 │
│  └──────────────────┘                                                   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Example Bug Fix with Regression Test

```
BUG FIX EXAMPLE - CACHE INVALIDATION BUG:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  BUG REPORT #1234:                                                      │
│  "User profile shows old data after update. Cache not invalidated."     │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

STEP 1-2: Write Regression Test (MUST FAIL)
───────────────────────────────────────────

// Regression test for BUG-1234: Cache not invalidated on profile update
function test_bug_1234_cache_invalidated_on_profile_update() {
    userId = "user:test:1234"
    cacheKey = "cache:user:" + userId + ":profile"

    // Setup: Create user and cache their profile
    originalProfile = { name: "John", email: "john@old.com" }
    database.insert(userId, originalProfile)
    redis.SET(cacheKey, serialize(originalProfile), "EX", 3600)

    // Action: Update profile in database
    newProfile = { name: "John", email: "john@new.com" }
    userService.updateProfile(userId, newProfile)

    // Verify: Cache should return NEW data (not stale)
    cachedProfile = deserialize(redis.GET(cacheKey))

    // This assertion will FAIL if bug exists
    assert cachedProfile.email == "john@new.com"
    // Bug: Returns "john@old.com" (stale cached data)
}

// Run test:
$ test_runner --run test_bug_1234
❌ FAILS - AssertionError: "john@old.com" != "john@new.com"
   Bug confirmed! Stale data returned from cache.


STEP 3: Fix the Bug
───────────────────

// BEFORE (buggy code):
function updateProfile(userId, newProfile) {
    // Only updates database, forgets to invalidate cache
    database.update("users", userId, newProfile)
    return newProfile
}

// AFTER (fixed code):
function updateProfile(userId, newProfile) {
    // Update database
    database.update("users", userId, newProfile)

    // Invalidate cache - FIX FOR BUG-1234
    cacheKey = "cache:user:" + userId + ":profile"
    redis.DEL(cacheKey)

    return newProfile
}


STEP 4: Verify Fix
──────────────────

// Run regression test:
$ test_runner --run test_bug_1234
✅ PASSES - Cache correctly invalidated

// Run all tests to check for regressions:
$ test_runner --run all
✅ ALL TESTS PASS - No regressions introduced


STEP 5: Document in Test
────────────────────────

// Regression test for BUG-1234: Cache not invalidated on profile update
// Issue: User profile showed stale data after update
// Root cause: updateProfile() did not invalidate cache after DB write
// Fix: Added redis.DEL(cacheKey) after database update
// Date: 2024-01-21
// Author: [developer]
function test_bug_1234_cache_invalidated_on_profile_update() {
    // ... test implementation ..
}
```

### Common Redis Bug Patterns and Regression Tests

```
COMMON BUG PATTERNS:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  BUG TYPE                 │ REGRESSION TEST PATTERN                     │
│  ─────────────────────────┼───────────────────────────────────────────  │
│                           │                                             │
│  Cache stampede           │ test_concurrent_cache_miss_single_db_call() │
│                           │ Verify: Only 1 DB call under concurrent     │
│                           │ requests for same key                       │
│                           │                                             │
│  Stale cache after update │ test_cache_invalidated_on_update()          │
│                           │ Verify: GET returns new data after SET      │
│                           │                                             │
│  TTL not set              │ test_cache_key_has_ttl()                    │
│                           │ Verify: TTL key returns > 0                 │
│                           │                                             │
│  Race condition in lock   │ test_lock_prevents_concurrent_execution()   │
│                           │ Verify: Only 1 process enters critical      │
│                           │ section at a time                           │
│                           │                                             │
│  Lock not released        │ test_lock_released_after_work()             │
│                           │ Verify: Lock key deleted after function     │
│                           │ completes (success or failure)              │
│                           │                                             │
│  Memory leak (no TTL)     │ test_all_cache_keys_have_expiration()       │
│                           │ Verify: SCAN all cache:* keys, check TTL    │
│                           │                                             │
│  Connection pool exhaust  │ test_connections_returned_to_pool()         │
│                           │ Verify: Pool available connections after    │
│                           │ many operations                             │
│                           │                                             │
│  Wrong data structure     │ test_data_structure_operations_correct()    │
│                           │ Verify: Operations work as expected         │
│                           │ (e.g., LPUSH vs RPUSH order)               │
│                           │                                             │
└─────────────────────────────────────────────────────────────────────────┘

EXAMPLE - CONNECTION LEAK BUG FIX:

// Regression test for BUG-5678: Connection pool exhaustion
function test_bug_5678_connections_returned_after_error() {
    pool = createRedisPool(maxConnections=5)

    // Simulate 100 operations that throw errors
    for i in 1..100:
        try {
            conn = pool.acquire()
            // Simulate error during operation
            throw new Error("Simulated failure")
        } catch (e) {
            // Connection should be returned even on error
        }

    // Pool should still have available connections
    // Bug: Connections were not returned on error, pool exhausted
    availableConnections = pool.getAvailableCount()
    assert availableConnections > 0
}

// Fix: Use try-finally to always return connection
function executeRedisOperation(operation) {
    conn = pool.acquire()
    try {
        return operation(conn)
    } finally {
        pool.release(conn)  // Always return connection
    }
}
```

---

## 3. Key Design Patterns

### A. Key Naming Conventions

```
KEY NAMING BEST PRACTICES:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  FORMAT: object-type:id:field[:sub-field]                               │
│  SEPARATOR: Use colons (:) as standard separator                        │
│                                                                         │
│  ✅ GOOD KEY NAMES:                                                     │
│                                                                         │
│  user:123                          # User object                        │
│  user:123:profile                  # User's profile                     │
│  user:123:sessions                 # User's sessions set                │
│  order:456:items                   # Order's items                      │
│  cache:api:users:list:page:1       # Cached API response                │
│  rate:api:user:123:minute:1530     # Rate limit counter                 │
│  lock:order:456                    # Distributed lock                   │
│  queue:emails:high                 # High priority email queue          │
│  stream:orders:events              # Order events stream                │
│                                                                         │
│  ❌ BAD KEY NAMES:                                                      │
│                                                                         │
│  123                               # No context                         │
│  user_123_profile                  # Inconsistent separator             │
│  getUserProfile:123                # Function name style                │
│  user:123:a:b:c:d:e:f:g            # Too deep nesting                   │
│  user:*                            # Wildcards in key names             │
│  user:123:verylongfieldname...     # Unnecessarily long                 │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

KEY NAMING PATTERNS:

1. ENTITY KEYS
   {entity}:{id}
   user:123
   product:456
   order:789

2. ENTITY FIELD KEYS
   {entity}:{id}:{field}
   user:123:profile
   user:123:settings
   user:123:permissions

3. RELATIONSHIP KEYS
   {entity}:{id}:{relationship}
   user:123:followers        # Set of follower IDs
   user:123:orders           # List/Set of order IDs
   product:456:reviews       # Set of review IDs

4. CACHE KEYS
   cache:{service}:{resource}:{identifier}:{params-hash}
   cache:api:products:list:a1b2c3d4
   cache:api:users:123:profile

5. RATE LIMIT KEYS
   rate:{resource}:{identifier}:{window}
   rate:api:user:123:minute:202401211530
   rate:login:ip:192.168.1.1:hour:2024012115

6. LOCK KEYS
   lock:{resource}:{identifier}
   lock:order:456
   lock:inventory:product:789

7. SESSION KEYS
   session:{session-id}
   session:abc123def456

8. QUEUE KEYS
   queue:{name}:{priority}
   queue:emails:high
   queue:notifications:normal
```

### B. Key Expiration Strategy

```
TTL (TIME TO LIVE) BEST PRACTICES:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  RULE: Almost every key should have a TTL                               │
│  Redis is a CACHE, not a primary database                               │
│                                                                         │
│  USE CASE                    │ RECOMMENDED TTL                          │
│  ────────────────────────────┼────────────────────────────────────────  │
│  Session data                │ 24 hours - 7 days                        │
│  API response cache          │ 1 minute - 1 hour                        │
│  Rate limiting               │ Window size (1 min, 1 hour)              │
│  Distributed locks           │ 10-30 seconds (with renewal)             │
│  User profile cache          │ 15 minutes - 1 hour                      │
│  Search results              │ 5-15 minutes                             │
│  Computed aggregations       │ 1 hour - 24 hours                        │
│  Feature flags               │ 5 minutes (for quick updates)            │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

SETTING TTL:

// Set TTL at creation (preferred)
SET key value EX 3600              // Expires in 3600 seconds
SET key value PX 3600000           // Expires in 3600000 milliseconds
SETEX key 3600 value               // Equivalent to SET + EX

// Set TTL on existing key
EXPIRE key 3600                    // Set TTL in seconds
PEXPIRE key 3600000                // Set TTL in milliseconds
EXPIREAT key 1705932000            // Set absolute Unix timestamp

// For HASH, LIST, SET, ZSET - set TTL on the key, not individual elements
HSET user:123 name "John" email "john@example.com"
EXPIRE user:123 3600

// Check remaining TTL
TTL key                            // Returns seconds, -1 if no TTL, -2 if not exists
PTTL key                           // Returns milliseconds

AVOIDING TTL STAMPEDE:

// Problem: All keys expire at same time → thundering herd to database
// Solution: Add random jitter to TTL

base_ttl = 3600
jitter = random(0, 300)            // 0-5 minutes random
final_ttl = base_ttl + jitter

SET key value EX final_ttl
```

---

## 4. Caching Patterns

### A. Cache-Aside (Lazy Loading)

```
CACHE-ASIDE PATTERN:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  Most common caching pattern - Application manages cache                │
│                                                                         │
│  READ FLOW:                                                             │
│                                                                         │
│  ┌─────────────┐    1. Check cache    ┌─────────────┐                  │
│  │ Application │ ──────────────────► │    Redis    │                  │
│  └──────┬──────┘                      └──────┬──────┘                  │
│         │                                    │                          │
│         │ 2. Cache miss                      │ Cache hit: return data   │
│         │                                    │                          │
│         ▼                                    │                          │
│  ┌─────────────┐                             │                          │
│  │  Database   │                             │                          │
│  └──────┬──────┘                             │                          │
│         │                                    │                          │
│         │ 3. Get data                        │                          │
│         │                                    │                          │
│         ▼                                    │                          │
│  ┌─────────────┐    4. Store in cache ┌─────────────┐                  │
│  │ Application │ ──────────────────► │    Redis    │                  │
│  └─────────────┘                      └─────────────┘                  │
│         │                                                               │
│         │ 5. Return data to caller                                      │
│         ▼                                                               │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

IMPLEMENTATION:

function getUser(userId) {
    cacheKey = "user:" + userId

    // 1. Try cache first
    cached = redis.GET(cacheKey)
    if (cached != null) {
        return deserialize(cached)
    }

    // 2. Cache miss - fetch from database
    user = database.query("SELECT * FROM users WHERE id = ?", userId)

    if (user != null) {
        // 3. Store in cache with TTL
        redis.SETEX(cacheKey, 3600, serialize(user))
    }

    return user
}

PROS:
  ✅ Simple to implement
  ✅ Cache only contains requested data
  ✅ Resilient - works if cache fails (with DB fallback)

CONS:
  ❌ Cache miss penalty (extra DB round trip)
  ❌ Data can be stale until TTL expires
  ❌ Cache stampede on popular keys
```

### B. Write-Through

```
WRITE-THROUGH PATTERN:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  Write to cache and database together (synchronously)                   │
│                                                                         │
│  WRITE FLOW:                                                            │
│                                                                         │
│  ┌─────────────┐                                                       │
│  │ Application │                                                       │
│  └──────┬──────┘                                                       │
│         │                                                               │
│         │ 1. Write request                                              │
│         ▼                                                               │
│  ┌─────────────────────────────────────────────────┐                   │
│  │               Cache Service                      │                   │
│  └──────┬───────────────────────────────┬──────────┘                   │
│         │                               │                               │
│         │ 2. Write to DB                │ 3. Write to cache            │
│         ▼                               ▼                               │
│  ┌─────────────┐                 ┌─────────────┐                       │
│  │  Database   │                 │    Redis    │                       │
│  └─────────────┘                 └─────────────┘                       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

IMPLEMENTATION:

function updateUser(userId, userData) {
    // 1. Write to database first (source of truth)
    database.execute(
        "UPDATE users SET name = ?, email = ? WHERE id = ?",
        userData.name, userData.email, userId
    )

    // 2. Write to cache
    cacheKey = "user:" + userId
    redis.SETEX(cacheKey, 3600, serialize(userData))

    return userData
}

PROS:
  ✅ Cache always has latest data
  ✅ No stale data issues
  ✅ Simple consistency model

CONS:
  ❌ Higher write latency (two writes)
  ❌ Cache may contain data never read
  ❌ Must handle write failures to either store
```

### C. Write-Behind (Write-Back)

```
WRITE-BEHIND PATTERN:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  Write to cache immediately, persist to database asynchronously         │
│                                                                         │
│  WRITE FLOW:                                                            │
│                                                                         │
│  ┌─────────────┐    1. Write    ┌─────────────┐                        │
│  │ Application │ ─────────────► │    Redis    │                        │
│  └─────────────┘    (fast)      └──────┬──────┘                        │
│                                        │                                │
│         ◄──────────────────────────────┘                                │
│         │ 2. Return immediately                                         │
│         │                                                               │
│                                        │                                │
│                                        │ 3. Async persist              │
│                                        │    (background worker)         │
│                                        ▼                                │
│                                 ┌─────────────┐                        │
│                                 │  Database   │                        │
│                                 └─────────────┘                        │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

IMPLEMENTATION:

function updateUser(userId, userData) {
    cacheKey = "user:" + userId

    // 1. Write to cache immediately
    redis.SETEX(cacheKey, 3600, serialize(userData))

    // 2. Queue async database write
    redis.LPUSH("queue:db-writes", serialize({
        type: "user_update",
        id: userId,
        data: userData,
        timestamp: now()
    }))

    return userData  // Return immediately
}

// Background worker
function processDbWrites() {
    while (true) {
        job = redis.BRPOP("queue:db-writes", 30)
        if (job != null) {
            writeToDatabase(job)
        }
    }
}

PROS:
  ✅ Very fast writes (cache only)
  ✅ Reduced database load (batching possible)
  ✅ Better for write-heavy workloads

CONS:
  ❌ Risk of data loss if Redis fails before persist
  ❌ Complex failure handling
  ❌ Eventually consistent
  ❌ Not suitable for critical data
```

### D. Cache Invalidation Patterns

```
CACHE INVALIDATION STRATEGIES:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  "There are only two hard things in Computer Science:                   │
│   cache invalidation and naming things." - Phil Karlton                 │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

1. TIME-BASED EXPIRATION (TTL)
   ─────────────────────────────
   // Simple, predictable, but data can be stale until TTL
   SET user:123 "{...}" EX 3600

2. EXPLICIT INVALIDATION
   ─────────────────────────────
   // Delete cache when data changes
   function updateUser(userId, data) {
       database.update(userId, data)
       redis.DEL("user:" + userId)           // Delete single key
       redis.DEL("cache:users:list:*")       // Invalidate related caches
   }

3. VERSION-BASED KEYS
   ─────────────────────────────
   // Include version in key, increment version to invalidate
   version = redis.GET("user:123:version") or "1"
   cacheKey = "user:123:v" + version

   // To invalidate: increment version
   redis.INCR("user:123:version")
   // Old cache entries naturally expire via TTL

4. TAG-BASED INVALIDATION
   ─────────────────────────────
   // Track related cache keys with tags
   SET cache:products:123 "{...}" EX 3600
   SADD tag:category:electronics "cache:products:123"
   SADD tag:category:electronics "cache:products:456"

   // Invalidate all products in category
   function invalidateCategory(category) {
       keys = redis.SMEMBERS("tag:category:" + category)
       if (keys.length > 0) {
           redis.DEL(...keys)
           redis.DEL("tag:category:" + category)
       }
   }

5. PUB/SUB INVALIDATION
   ─────────────────────────────
   // Publish invalidation events to all app instances
   function updateUser(userId, data) {
       database.update(userId, data)
       redis.PUBLISH("cache:invalidate", "user:" + userId)
   }

   // Each app instance subscribes
   redis.SUBSCRIBE("cache:invalidate", (key) => {
       localCache.delete(key)
       redis.DEL(key)
   })

INVALIDATION BEST PRACTICES:

  ✅ Prefer short TTLs over complex invalidation logic
  ✅ Use explicit DELETE for critical consistency
  ✅ Batch invalidations when possible
  ✅ Consider eventual consistency trade-offs
  ❌ Don't use KEYS command in production (blocks Redis)
  ❌ Don't rely solely on invalidation for consistency
```

---

## 5. Connection Management

### A. Connection Pooling

```
CONNECTION POOLING:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  RULE: Always use connection pooling in production                      │
│  Creating connections is expensive                                      │
│                                                                         │
│  WITHOUT POOLING:                   WITH POOLING:                       │
│                                                                         │
│  Request 1 → New Connection         Request 1 ──┐                       │
│  Request 2 → New Connection         Request 2 ──┼──► Pool ──► Redis    │
│  Request 3 → New Connection         Request 3 ──┘    (reuse)           │
│                                                                         │
│  ❌ Connection overhead per request  ✅ Connections reused              │
│  ❌ Risk of connection exhaustion    ✅ Bounded resource usage          │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

POOL CONFIGURATION:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  SETTING              │ RECOMMENDATION           │ NOTES                │
│  ─────────────────────┼──────────────────────────┼────────────────────  │
│  Min pool size        │ 5-10                     │ Pre-warmed connections│
│  Max pool size        │ 20-50                    │ Based on load        │
│  Connection timeout   │ 1-5 seconds              │ Fail fast            │
│  Idle timeout         │ 30-60 seconds            │ Release unused       │
│  Max lifetime         │ 30-60 minutes            │ Refresh connections  │
│  Validation on borrow │ Optional PING            │ Ensure connection OK │
│                                                                         │
│  SIZING FORMULA:                                                        │
│  max_pool = (concurrent_requests * avg_redis_calls_per_request)         │
│             / expected_throughput_per_connection                        │
│                                                                         │
│  Example: 100 concurrent requests × 3 Redis calls = 300                │
│           If each connection handles 20 ops/sec → 15 connections       │
│           Add 50% buffer → 20-25 max pool size                         │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

PSEUDO-CODE CONFIGURATION:

pool = RedisPool({
    host: "redis.example.com",
    port: 6379,
    password: getSecret("redis-password"),

    // Pool settings
    minConnections: 10,
    maxConnections: 50,
    connectionTimeout: 3000,      // 3 seconds
    idleTimeout: 30000,           // 30 seconds
    maxLifetime: 1800000,         // 30 minutes

    // Retry settings
    retryAttempts: 3,
    retryDelay: 100,              // 100ms between retries

    // TLS (if required)
    tls: {
        enabled: true,
        rejectUnauthorized: true
    }
})
```

### B. Pipelining

```
PIPELINING:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  Batch multiple commands in single network round trip                   │
│                                                                         │
│  WITHOUT PIPELINING:                WITH PIPELINING:                    │
│                                                                         │
│  Client    Redis                    Client    Redis                     │
│    │         │                        │         │                       │
│    │──CMD1──►│                        │──CMD1──►│                       │
│    │◄──RES1──│                        │──CMD2──►│                       │
│    │──CMD2──►│                        │──CMD3──►│                       │
│    │◄──RES2──│                        │         │                       │
│    │──CMD3──►│                        │◄──RES1──│                       │
│    │◄──RES3──│                        │◄──RES2──│                       │
│    │         │                        │◄──RES3──│                       │
│                                                                         │
│  6 network round trips              2 network round trips               │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

WHEN TO USE PIPELINING:

  ✅ Multiple independent commands
  ✅ Bulk data loading
  ✅ Batch reads or writes
  ✅ Commands don't depend on previous results

  ❌ Commands depend on previous results (use Lua instead)
  ❌ Need atomic execution (use MULTI/EXEC or Lua)

EXAMPLE:

// Without pipelining - 3 round trips
redis.SET("user:1:name", "Alice")
redis.SET("user:2:name", "Bob")
redis.SET("user:3:name", "Carol")

// With pipelining - 1 round trip
pipeline = redis.pipeline()
pipeline.SET("user:1:name", "Alice")
pipeline.SET("user:2:name", "Bob")
pipeline.SET("user:3:name", "Carol")
results = pipeline.execute()

// Reading multiple keys
pipeline = redis.pipeline()
pipeline.GET("user:1:name")
pipeline.GET("user:2:name")
pipeline.GET("user:3:name")
names = pipeline.execute()  // ["Alice", "Bob", "Carol"]

BATCH SIZE RECOMMENDATION:
  • 100-1000 commands per pipeline is typical
  • Don't exceed 10,000 commands (memory overhead)
  • Monitor Redis memory during large pipelines
```

---

## 6. Transactions and Atomicity

### A. MULTI/EXEC Transactions

```
REDIS TRANSACTIONS (MULTI/EXEC):

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  Execute multiple commands atomically (all or nothing queued)           │
│  NOTE: Not the same as database transactions - no rollback              │
│                                                                         │
│  Commands are QUEUED, then executed SEQUENTIALLY when EXEC is called   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

BASIC TRANSACTION:

MULTI                              // Start transaction
SET user:123:name "Alice"          // Queued
INCR user:123:login_count          // Queued
EXPIRE user:123:name 3600          // Queued
EXEC                               // Execute all queued commands

// All commands execute atomically - no other client can interleave

OPTIMISTIC LOCKING WITH WATCH:

// Transfer balance between accounts (optimistic concurrency)

WATCH account:A account:B          // Watch for changes

balanceA = GET account:A
balanceB = GET account:B

if (balanceA >= transferAmount) {
    MULTI
    DECRBY account:A transferAmount
    INCRBY account:B transferAmount
    EXEC                           // Returns null if watched keys changed
} else {
    UNWATCH
    throw InsufficientFunds
}

// If another client modified account:A or account:B after WATCH,
// EXEC returns null and transaction is aborted - retry needed

TRANSACTION LIMITATIONS:

  ❌ No rollback - if one command fails, others still execute
  ❌ No conditional logic inside transaction
  ❌ All commands must be known upfront (no read-then-write)
  ❌ WATCH can cause retries under contention

  ✅ For complex logic, use Lua scripts instead
```

### B. Lua Scripting (Recommended)

```
LUA SCRIPTS:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  PREFERRED for complex atomic operations                                │
│  Script executes atomically on Redis server                             │
│  Can include conditional logic, loops, and reads before writes          │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

EXAMPLE 1: Atomic counter with limit

// Rate limiter: increment if under limit, return new count or -1 if exceeded
local key = KEYS[1]
local limit = tonumber(ARGV[1])
local ttl = tonumber(ARGV[2])

local current = tonumber(redis.call('GET', key) or '0')

if current >= limit then
    return -1  -- Rate limit exceeded
end

current = redis.call('INCR', key)

if current == 1 then
    redis.call('EXPIRE', key, ttl)
end

return current

// Call from application
EVALSHA <script-sha> 1 rate:user:123 100 60

EXAMPLE 2: Conditional update (compare-and-set)

// Update only if version matches
local key = KEYS[1]
local expected_version = ARGV[1]
local new_value = ARGV[2]
local new_version = ARGV[3]

local current_version = redis.call('HGET', key, 'version')

if current_version ~= expected_version then
    return 0  -- Version mismatch, update rejected
end

redis.call('HSET', key, 'data', new_value, 'version', new_version)
return 1  -- Success

EXAMPLE 3: Distributed lock with proper release

// Release lock only if we own it
local key = KEYS[1]
local owner = ARGV[1]

if redis.call('GET', key) == owner then
    return redis.call('DEL', key)
else
    return 0  -- Not our lock
end

SCRIPT BEST PRACTICES:

  ✅ Use EVALSHA with script caching (not EVAL each time)
  ✅ Keep scripts short and focused
  ✅ Pass keys as KEYS[], values as ARGV[]
  ✅ Return meaningful results

  ❌ Don't make scripts too complex (hard to debug)
  ❌ Don't use non-deterministic commands (TIME, RANDOMKEY)
  ❌ Don't perform heavy computation (blocks Redis)
  ❌ Don't access keys not passed in KEYS[]

SCRIPT LOADING:

// Load script once, use SHA for calls
sha = redis.SCRIPT_LOAD(script)

// Call by SHA (efficient)
result = redis.EVALSHA(sha, num_keys, keys..., args...)
```

---

## 7. Redis Streams

### A. Stream Basics

```
REDIS STREAMS:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  Append-only log data structure (like Kafka)                            │
│  Perfect for event sourcing, message queues, activity feeds             │
│                                                                         │
│  Stream: orders:events                                                  │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │ 1705849200000-0 │ 1705849200001-0 │ 1705849200002-0 │ ...      │    │
│  │ action:created  │ action:paid     │ action:shipped  │          │    │
│  │ order_id:123    │ order_id:123    │ order_id:123    │          │    │
│  │ amount:99.99    │ payment_id:p456 │ tracking:TRK789 │          │    │
│  └────────────────────────────────────────────────────────────────┘    │
│                                                                         │
│  Entry ID format: <timestamp>-<sequence>                                │
│  Auto-generated: * (Redis generates)                                    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

BASIC OPERATIONS:

// Add entry to stream
XADD orders:events * action created order_id 123 amount 99.99
// Returns: "1705849200000-0"

// Add with max length (trim old entries)
XADD orders:events MAXLEN ~ 10000 * action shipped order_id 123

// Read entries
XRANGE orders:events - +                    // All entries
XRANGE orders:events - + COUNT 10           // First 10
XRANGE orders:events 1705849200000-0 +      // From specific ID

// Read new entries (blocking)
XREAD BLOCK 5000 STREAMS orders:events $    // $ = only new entries

// Stream info
XLEN orders:events                          // Entry count
XINFO STREAM orders:events                  // Stream metadata
```

### B. Consumer Groups

```
CONSUMER GROUPS:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  Distribute stream processing across multiple consumers                 │
│  Each message delivered to ONE consumer in the group                    │
│  Supports acknowledgment and pending message tracking                   │
│                                                                         │
│  Stream: orders:events                                                  │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │ entry-1 │ entry-2 │ entry-3 │ entry-4 │ entry-5 │ entry-6 │    │    │
│  └────────────────────────────────────────────────────────────────┘    │
│       │         │         │         │         │         │              │
│       ▼         ▼         ▼         ▼         ▼         ▼              │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │              Consumer Group: processors                          │   │
│  │  ┌───────────┐  ┌───────────┐  ┌───────────┐                    │   │
│  │  │ consumer1 │  │ consumer2 │  │ consumer3 │                    │   │
│  │  │ entry-1   │  │ entry-2   │  │ entry-3   │                    │   │
│  │  │ entry-4   │  │ entry-5   │  │ entry-6   │                    │   │
│  │  └───────────┘  └───────────┘  └───────────┘                    │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

SETUP:

// Create consumer group
XGROUP CREATE orders:events processors $ MKSTREAM
// $ = start from new messages
// 0 = start from beginning

// Consumer reads from group
XREADGROUP GROUP processors consumer1 COUNT 10 STREAMS orders:events >
// > = only new messages not delivered to other consumers

// Acknowledge processed message
XACK orders:events processors 1705849200000-0

// Check pending (unacknowledged) messages
XPENDING orders:events processors

// Claim stuck message (consumer died)
XCLAIM orders:events processors consumer2 60000 1705849200000-0
// 60000 = message must be pending for 60 seconds

CONSUMER PATTERN:

function processOrders(consumerId) {
    while (true) {
        // Read new messages (blocking)
        entries = redis.XREADGROUP(
            GROUP, "processors", consumerId,
            COUNT, 10,
            BLOCK, 5000,
            STREAMS, "orders:events", ">"
        )

        for (entry in entries) {
            try {
                processOrder(entry)
                redis.XACK("orders:events", "processors", entry.id)
            } catch (error) {
                // Don't ACK - message will be reprocessed
                log.error("Failed to process", entry.id, error)
            }
        }

        // Also check for stuck pending messages
        claimStuckMessages(consumerId)
    }
}

function claimStuckMessages(consumerId) {
    // Claim messages pending > 5 minutes
    pending = redis.XPENDING("orders:events", "processors", "-", "+", 10)

    for (msg in pending) {
        if (msg.idle_time > 300000) {  // 5 minutes
            redis.XCLAIM(
                "orders:events", "processors", consumerId,
                300000, msg.id
            )
        }
    }
}
```

---

## 8. High Availability

### A. Redis Sentinel

```
REDIS SENTINEL:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  Automatic failover for Redis master-replica setup                      │
│                                                                         │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                 │
│  │ Sentinel 1  │    │ Sentinel 2  │    │ Sentinel 3  │                 │
│  └──────┬──────┘    └──────┬──────┘    └──────┬──────┘                 │
│         │                  │                  │                         │
│         └──────────────────┼──────────────────┘                         │
│                            │ Monitor & Coordinate                       │
│                            ▼                                            │
│         ┌─────────────────────────────────────────┐                    │
│         │                                         │                    │
│    ┌────┴────┐         ┌──────────┐         ┌────┴────┐               │
│    │ Master  │────────►│ Replica  │         │ Replica │               │
│    │ (write) │  sync   │  (read)  │         │  (read) │               │
│    └─────────┘         └──────────┘         └─────────┘               │
│                                                                         │
│  On master failure:                                                     │
│    1. Sentinels detect failure (quorum)                                │
│    2. Elect new master from replicas                                   │
│    3. Reconfigure other replicas                                       │
│    4. Notify clients of new master                                     │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

CLIENT CONFIGURATION:

// Connect via Sentinel (not directly to Redis)
client = RedisSentinelClient({
    sentinels: [
        { host: "sentinel1.example.com", port: 26379 },
        { host: "sentinel2.example.com", port: 26379 },
        { host: "sentinel3.example.com", port: 26379 }
    ],
    masterName: "mymaster",
    password: getSecret("redis-password")
})

// Client automatically:
// - Discovers current master
// - Reconnects on failover
// - Routes writes to master, reads to replicas (if configured)
```

### B. Redis Cluster

```
REDIS CLUSTER:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  Horizontal scaling with automatic sharding                             │
│  Data distributed across multiple masters                               │
│                                                                         │
│  16384 hash slots distributed across masters                            │
│                                                                         │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │                      Hash Slots 0-16383                         │    │
│  │  ┌──────────────┬──────────────┬──────────────┬─────────────┐  │    │
│  │  │  0-5460      │  5461-10922  │ 10923-16383  │             │  │    │
│  │  └──────┬───────┴──────┬───────┴──────┬───────┘             │  │    │
│  │         │              │              │                      │  │    │
│  │         ▼              ▼              ▼                      │  │    │
│  │  ┌────────────┐ ┌────────────┐ ┌────────────┐               │  │    │
│  │  │  Master A  │ │  Master B  │ │  Master C  │               │  │    │
│  │  │  Replica   │ │  Replica   │ │  Replica   │               │  │    │
│  │  └────────────┘ └────────────┘ └────────────┘               │  │    │
│  │                                                              │  │    │
│  └────────────────────────────────────────────────────────────────┘    │
│                                                                         │
│  Key assignment: slot = CRC16(key) mod 16384                           │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

HASH TAGS - Force keys to same slot:

// Keys with same {tag} go to same slot
SET user:{123}:profile "..."       // slot = CRC16("123")
SET user:{123}:settings "..."      // slot = CRC16("123") - same slot!
SET user:{123}:sessions "..."      // slot = CRC16("123") - same slot!

// Enables multi-key operations on related data
MGET user:{123}:profile user:{123}:settings

// Without hash tags, these might be on different nodes:
SET user:123:profile "..."         // slot = CRC16("user:123:profile")
SET user:123:settings "..."        // slot = CRC16("user:123:settings")

CLUSTER LIMITATIONS:

  ❌ Multi-key operations only work if keys on same slot
  ❌ Lua scripts must use keys on same slot
  ❌ WATCH/MULTI transactions limited to single slot
  ❌ Database selection (SELECT) not supported
  ❌ More complex operations than Sentinel

CLIENT CONFIGURATION:

client = RedisClusterClient({
    nodes: [
        { host: "redis1.example.com", port: 6379 },
        { host: "redis2.example.com", port: 6379 },
        { host: "redis3.example.com", port: 6379 }
    ],
    password: getSecret("redis-password"),

    // Handle MOVED/ASK redirections automatically
    followRedirects: true,

    // Read from replicas for read scaling
    readFromReplicas: true
})
```

---

## 9. Performance Optimization

### A. Memory Optimization

```
MEMORY OPTIMIZATION:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  RULE: Monitor memory usage constantly                                  │
│  Redis data must fit in RAM                                             │
│                                                                         │
│  INFO memory                                                            │
│  MEMORY USAGE key                                                       │
│  MEMORY DOCTOR                                                          │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

MEMORY-EFFICIENT DATA STRUCTURES:

1. USE HASHES FOR SMALL OBJECTS
   // Instead of multiple strings:
   SET user:123:name "Alice"        // ~56 bytes overhead per key
   SET user:123:email "a@b.com"
   SET user:123:age "30"

   // Use a hash (much more efficient for small hashes):
   HSET user:123 name "Alice" email "a@b.com" age "30"
   // Redis uses ziplist encoding for small hashes (~60-70% memory savings)

2. SHORT KEY NAMES (for high-volume keys)
   // If you have millions of keys, shorter names save memory
   u:123:p instead of user:123:profile
   // But balance readability vs memory savings

3. USE INTEGERS WHEN POSSIBLE
   // Redis optimizes integer storage
   SET counter 12345              // Stored as integer, not string

4. COMPRESS VALUES
   // For large values, compress before storing
   SET key (compress(largeValue))
   value = decompress(GET key)

5. USE APPROPRIATE DATA STRUCTURES
   // HYPERLOGLOG for cardinality (~12KB vs potentially GBs)
   PFADD visitors user1 user2 user3
   PFCOUNT visitors

   // BITMAP for binary flags (~1 bit per flag)
   SETBIT user:123:features 0 1    // Feature 0 enabled
   SETBIT user:123:features 1 0    // Feature 1 disabled

EVICTION POLICIES:

maxmemory 4gb
maxmemory-policy allkeys-lru       // Evict least recently used

POLICIES:
  noeviction        - Return errors when memory limit reached
  allkeys-lru       - Evict LRU keys (recommended for cache)
  allkeys-lfu       - Evict least frequently used
  volatile-lru      - Evict LRU keys with TTL set
  volatile-lfu      - Evict LFU keys with TTL set
  volatile-ttl      - Evict keys with shortest TTL
  allkeys-random    - Evict random keys
  volatile-random   - Evict random keys with TTL set
```

### B. Command Optimization

```
COMMAND OPTIMIZATION:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  ❌ AVOID THESE COMMANDS IN PRODUCTION:                                 │
│                                                                         │
│  KEYS *              - Blocks Redis, O(N) scan of all keys              │
│  FLUSHALL/FLUSHDB    - Deletes everything, blocks                       │
│  DEBUG               - Debug commands can crash Redis                   │
│  SAVE                - Synchronous disk write, blocks                   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

USE SCAN INSTEAD OF KEYS:

// ❌ NEVER do this in production
KEYS user:*                        // Blocks until complete

// ✅ Use SCAN for iteration
cursor = 0
do {
    (cursor, keys) = SCAN cursor MATCH "user:*" COUNT 100
    // Process keys
} while (cursor != 0)

// SCAN is:
// - Non-blocking (returns in small batches)
// - Cursor-based (can resume)
// - Safe for production

EFFICIENT PATTERNS:

1. BATCH READS
   // ❌ Multiple round trips
   GET key1
   GET key2
   GET key3

   // ✅ Single command
   MGET key1 key2 key3

2. USE PIPELINES
   // ❌ 100 round trips
   for i in 1..100:
       SET key:i value:i

   // ✅ 1 round trip
   pipeline = redis.pipeline()
   for i in 1..100:
       pipeline.SET(key:i, value:i)
   pipeline.execute()

3. AVOID LARGE VALUES
   // ❌ Single 100MB value
   SET bigkey (100MB data)

   // ✅ Split into chunks
   for i, chunk in enumerate(chunks(data, 1MB)):
       SET bigkey:chunk:i chunk

4. USE APPROPRIATE COMMANDS
   // ❌ Get all then count in application
   members = SMEMBERS myset
   count = len(members)

   // ✅ Let Redis count
   count = SCARD myset

5. AVOID HOT KEYS
   // ❌ Single counter hit by all traffic
   INCR global:counter

   // ✅ Shard counters
   shard = hash(request_id) % 10
   INCR global:counter:shard:{shard}
   // Aggregate periodically or on read
```

---

## 10. Security Best Practices

### A. Authentication and Authorization

```
SECURITY CONFIGURATION:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  Redis should NEVER be exposed to the internet without protection       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

1. REQUIRE PASSWORD (Basic - Redis < 6)
   requirepass YourStrongPasswordHere

2. ACL (Access Control Lists - Redis 6+)
   // Create user with limited permissions
   ACL SETUSER app_user on >password ~cache:* +get +set +del -@dangerous

   // User can only:
   // - Access keys matching cache:*
   // - Run GET, SET, DEL commands
   // - Cannot run dangerous commands

   // Read-only user
   ACL SETUSER readonly_user on >password ~* +@read -@write

3. TLS ENCRYPTION
   tls-port 6379
   port 0                          // Disable non-TLS port
   tls-cert-file /path/to/redis.crt
   tls-key-file /path/to/redis.key
   tls-ca-cert-file /path/to/ca.crt
   tls-auth-clients yes            // Require client certificates

4. NETWORK SECURITY
   bind 127.0.0.1 10.0.0.1         // Only listen on specific interfaces
   protected-mode yes               // Reject external connections without password

   // Use firewall rules
   // Only allow application servers to connect

5. RENAME/DISABLE DANGEROUS COMMANDS
   rename-command FLUSHALL ""       // Disable completely
   rename-command FLUSHDB ""
   rename-command DEBUG ""
   rename-command CONFIG "CONFIG_b840fc02"  // Rename to obscure name
```

### B. Data Protection

```
DATA PROTECTION:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  1. ENCRYPT SENSITIVE DATA BEFORE STORING                               │
│                                                                         │
│     // Application-level encryption                                     │
│     encryptedData = encrypt(sensitiveData, key)                        │
│     redis.SET("user:123:ssn", encryptedData)                           │
│                                                                         │
│  2. DON'T STORE SECRETS IN KEY NAMES                                    │
│                                                                         │
│     ❌ SET session:abc123secret456 data                                │
│     ✅ SET session:hashed_id data                                      │
│                                                                         │
│  3. SET TTL ON SENSITIVE DATA                                           │
│                                                                         │
│     SET session:123 data EX 3600                                       │
│                                                                         │
│  4. AUDIT LOGGING                                                       │
│                                                                         │
│     // Enable slow log                                                  │
│     slowlog-log-slower-than 10000  // Log commands > 10ms              │
│     slowlog-max-len 128                                                │
│                                                                         │
│  5. REGULAR BACKUPS                                                     │
│                                                                         │
│     // RDB snapshots                                                    │
│     save 900 1                     // Save if 1 key changed in 900s    │
│     save 300 10                    // Save if 10 keys changed in 300s  │
│                                                                         │
│     // AOF persistence                                                  │
│     appendonly yes                                                      │
│     appendfsync everysec                                               │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 11. Common Patterns

### A. Distributed Locking

```
DISTRIBUTED LOCKING (REDLOCK):

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  Use for mutual exclusion across distributed systems                    │
│  NOT for performance - use for correctness                              │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

SIMPLE LOCK (Single Redis):

// Acquire lock
function acquireLock(lockKey, ownerId, ttlMs) {
    // NX = only if not exists, PX = milliseconds TTL
    result = redis.SET(lockKey, ownerId, "NX", "PX", ttlMs)
    return result == "OK"
}

// Release lock (only if we own it)
// Use Lua script for atomicity
RELEASE_SCRIPT = """
if redis.call('GET', KEYS[1]) == ARGV[1] then
    return redis.call('DEL', KEYS[1])
else
    return 0
end
"""

function releaseLock(lockKey, ownerId) {
    return redis.EVAL(RELEASE_SCRIPT, 1, lockKey, ownerId)
}

// Usage
lockKey = "lock:order:123"
ownerId = generateUniqueId()  // UUID or similar
ttl = 30000  // 30 seconds

if (acquireLock(lockKey, ownerId, ttl)) {
    try {
        // Do protected work
        processOrder(123)
    } finally {
        releaseLock(lockKey, ownerId)
    }
} else {
    // Lock held by another process
    throw LockNotAcquiredError
}

LOCK WITH RENEWAL:

function withLock(lockKey, ttlMs, work) {
    ownerId = generateUniqueId()

    if (!acquireLock(lockKey, ownerId, ttlMs)) {
        throw LockNotAcquiredError
    }

    // Start renewal thread
    renewalTask = scheduleRepeatedly(ttlMs / 3, () => {
        // Extend lock if we still own it
        redis.EVAL("""
            if redis.call('GET', KEYS[1]) == ARGV[1] then
                return redis.call('PEXPIRE', KEYS[1], ARGV[2])
            end
            return 0
        """, 1, lockKey, ownerId, ttlMs)
    })

    try {
        return work()
    } finally {
        renewalTask.cancel()
        releaseLock(lockKey, ownerId)
    }
}

REDLOCK (Multiple Redis instances - for critical sections):

// Acquire lock on N/2+1 of N independent Redis instances
// More complex but tolerates Redis failures
// See: https://redis.io/docs/manual/patterns/distributed-locks/
```

### B. Rate Limiting

```
RATE LIMITING PATTERNS:

1. FIXED WINDOW (Simple)
─────────────────────────

function checkRateLimit(userId, limit, windowSeconds) {
    key = "rate:" + userId + ":" + (now() / windowSeconds)

    current = redis.INCR(key)
    if (current == 1) {
        redis.EXPIRE(key, windowSeconds)
    }

    return current <= limit
}

// Problem: Boundary issue - 2x burst possible at window edges

2. SLIDING WINDOW LOG (Precise)
───────────────────────────────

function checkRateLimit(userId, limit, windowMs) {
    key = "rate:" + userId
    now = currentTimeMs()
    windowStart = now - windowMs

    // Remove old entries, add new, count
    pipeline = redis.pipeline()
    pipeline.ZREMRANGEBYSCORE(key, 0, windowStart)
    pipeline.ZADD(key, now, now + ":" + randomId())
    pipeline.ZCARD(key)
    pipeline.EXPIRE(key, windowMs / 1000 + 1)
    results = pipeline.execute()

    count = results[2]
    return count <= limit
}

// Precise but memory-intensive (stores each request)

3. SLIDING WINDOW COUNTER (Balanced)
────────────────────────────────────

// Lua script for atomic sliding window
SLIDING_WINDOW_SCRIPT = """
local key = KEYS[1]
local window = tonumber(ARGV[1])
local limit = tonumber(ARGV[2])
local now = tonumber(ARGV[3])

local current_window = math.floor(now / window)
local previous_window = current_window - 1
local current_key = key .. ':' .. current_window
local previous_key = key .. ':' .. previous_window

local current_count = tonumber(redis.call('GET', current_key) or '0')
local previous_count = tonumber(redis.call('GET', previous_key) or '0')

-- Weight previous window by how much of it is still in the sliding window
local elapsed_in_current = now % window
local weight = (window - elapsed_in_current) / window
local weighted_count = math.floor(previous_count * weight) + current_count

if weighted_count >= limit then
    return 0  -- Rate limited
end

redis.call('INCR', current_key)
redis.call('EXPIRE', current_key, window * 2)
return 1  -- Allowed
"""

function checkRateLimit(userId, limit, windowSeconds) {
    return redis.EVAL(
        SLIDING_WINDOW_SCRIPT, 1,
        "rate:" + userId,
        windowSeconds, limit, now()
    ) == 1
}

4. TOKEN BUCKET (Smooth)
────────────────────────

// Allows bursts up to bucket size, refills at constant rate
BUCKET_SCRIPT = """
local key = KEYS[1]
local capacity = tonumber(ARGV[1])
local refill_rate = tonumber(ARGV[2])  -- tokens per second
local now = tonumber(ARGV[3])

local bucket = redis.call('HMGET', key, 'tokens', 'last_refill')
local tokens = tonumber(bucket[1]) or capacity
local last_refill = tonumber(bucket[2]) or now

-- Refill tokens
local elapsed = now - last_refill
local refill = elapsed * refill_rate
tokens = math.min(capacity, tokens + refill)

if tokens < 1 then
    return 0  -- No tokens available
end

-- Consume one token
tokens = tokens - 1
redis.call('HMSET', key, 'tokens', tokens, 'last_refill', now)
redis.call('EXPIRE', key, capacity / refill_rate * 2)
return 1
"""
```

### C. Session Management

```
SESSION MANAGEMENT:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  Redis is excellent for session storage                                 │
│  Fast, supports TTL, can be shared across app instances                 │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

SESSION STORAGE PATTERN:

// Create session
function createSession(userId, data, ttlSeconds = 86400) {
    sessionId = generateSecureRandomId()
    key = "session:" + sessionId

    sessionData = {
        userId: userId,
        createdAt: now(),
        ...data
    }

    redis.SETEX(key, ttlSeconds, serialize(sessionData))

    // Index for user lookup (optional)
    redis.SADD("user:" + userId + ":sessions", sessionId)
    redis.EXPIRE("user:" + userId + ":sessions", ttlSeconds)

    return sessionId
}

// Get session
function getSession(sessionId) {
    key = "session:" + sessionId
    data = redis.GET(key)

    if (data == null) {
        return null
    }

    // Refresh TTL on access (sliding expiration)
    redis.EXPIRE(key, 86400)

    return deserialize(data)
}

// Update session
function updateSession(sessionId, updates) {
    key = "session:" + sessionId

    // Use WATCH for optimistic locking
    redis.WATCH(key)
    data = redis.GET(key)

    if (data == null) {
        redis.UNWATCH()
        return false
    }

    session = deserialize(data)
    session = { ...session, ...updates }

    pipeline = redis.MULTI()
    pipeline.SETEX(key, 86400, serialize(session))
    result = pipeline.EXEC()

    return result != null
}

// Delete session
function deleteSession(sessionId) {
    key = "session:" + sessionId

    // Get userId to clean up index
    data = redis.GET(key)
    if (data != null) {
        session = deserialize(data)
        redis.SREM("user:" + session.userId + ":sessions", sessionId)
    }

    redis.DEL(key)
}

// Delete all sessions for user
function deleteAllUserSessions(userId) {
    key = "user:" + userId + ":sessions"
    sessionIds = redis.SMEMBERS(key)

    if (sessionIds.length > 0) {
        keys = sessionIds.map(id => "session:" + id)
        redis.DEL(...keys, key)
    }
}
```

---

## 12. Anti-Patterns to Avoid

```
REDIS ANTI-PATTERNS:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  ❌ USING REDIS AS PRIMARY DATABASE                                     │
│  ────────────────────────────────────                                   │
│  Redis is a cache/data structure store, not a primary database          │
│  Data loss possible on restart without persistence                      │
│  → Use Redis as cache, keep source of truth elsewhere                   │
│                                                                         │
│  ❌ STORING LARGE VALUES                                                │
│  ────────────────────────────                                           │
│  Values > 1MB cause performance issues                                  │
│  → Split large values, use object storage for big data                  │
│                                                                         │
│  ❌ USING KEYS COMMAND                                                  │
│  ────────────────────────                                               │
│  KEYS * blocks Redis while scanning all keys                            │
│  → Use SCAN for iteration                                               │
│                                                                         │
│  ❌ NO TTL ON CACHE KEYS                                                │
│  ────────────────────────                                               │
│  Memory fills up, eviction becomes unpredictable                        │
│  → Always set TTL, even if long                                         │
│                                                                         │
│  ❌ HOT KEYS                                                            │
│  ─────────────                                                          │
│  Single key receiving all traffic                                       │
│  → Shard hot keys, use read replicas                                    │
│                                                                         │
│  ❌ NOT HANDLING CONNECTION FAILURES                                    │
│  ───────────────────────────────────                                    │
│  App crashes when Redis unavailable                                     │
│  → Implement circuit breaker, graceful degradation                      │
│                                                                         │
│  ❌ SYNCHRONOUS PERSISTENCE CALLS                                       │
│  ───────────────────────────────────                                    │
│  Using SAVE instead of BGSAVE                                           │
│  → Use BGSAVE for background persistence                                │
│                                                                         │
│  ❌ IGNORING MEMORY LIMITS                                              │
│  ───────────────────────────                                            │
│  Not setting maxmemory, letting Redis use all RAM                       │
│  → Set maxmemory and eviction policy                                    │
│                                                                         │
│  ❌ UNBOUNDED COLLECTIONS                                               │
│  ────────────────────────                                               │
│  Lists/Sets that grow forever                                           │
│  → Use LTRIM for lists, TTL for cleanup, MAXLEN for streams            │
│                                                                         │
│  ❌ NOT USING PIPELINING                                                │
│  ───────────────────────                                                │
│  Many individual commands when batching is possible                     │
│  → Use pipelines for bulk operations                                    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 13. Verification Checklist

```
REDIS IMPLEMENTATION CHECKLIST:

□ Data Structure Selection
  □ Chosen appropriate data structure for each use case
  □ Using hashes for objects (memory efficient)
  □ Using sorted sets for ranked/time-series data
  □ Using streams for event logs/queues

□ Key Design
  □ Consistent naming convention (object:id:field)
  □ Using colons as separators
  □ Keys are not too long (< 1KB)
  □ Using hash tags for cluster slot affinity

□ TTL Strategy
  □ Every key has appropriate TTL
  □ TTL jitter to prevent stampede
  □ Sliding expiration where needed

□ Connection Management
  □ Using connection pooling
  □ Pool size appropriate for load
  □ Connection timeouts configured
  □ TLS enabled (if required)

□ Performance
  □ Using pipelining for batch operations
  □ No KEYS command in production
  □ No large values (< 1MB)
  □ Hot keys identified and sharded

□ Reliability
  □ Circuit breaker for Redis failures
  □ Graceful degradation when cache unavailable
  □ Application works (degraded) without Redis

□ Security
  □ Password/ACL configured
  □ Network access restricted
  □ Dangerous commands disabled/renamed
  □ TLS for remote connections

□ Monitoring
  □ Memory usage monitored
  □ Connection count monitored
  □ Slow log enabled
  □ Key metrics dashboarded

□ High Availability
  □ Replication configured
  □ Sentinel or Cluster for automatic failover
  □ Client configured for failover handling
```

---

## 14. Summary

### Key Recommendations

| Use Case | Recommendation |
|----------|----------------|
| Simple caching | STRING with TTL |
| Object caching | HASH (field access) or STRING (full object) |
| Counters | STRING with INCR |
| Session storage | STRING or HASH with TTL |
| Rate limiting | Sorted Set or Lua script |
| Leaderboards | Sorted Set |
| Queues | LIST (simple) or STREAM (with acks) |
| Pub/Sub | Native Pub/Sub or STREAM |
| Unique counts | HYPERLOGLOG (approximate) or SET (exact) |
| Distributed locks | SET with NX + Lua for release |

### Remember

> "Redis is a cache, not a database. Always have a source of truth elsewhere."

> "Memory is precious. Set TTLs, monitor usage, configure eviction."

> "Network round trips matter. Use pipelining and Lua scripts."

> "Fail gracefully. Your application should work (degraded) without Redis."

---

## 15. Quick Reference

### Common redis-cli Commands

```
REDIS-CLI QUICK REFERENCE:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  CONNECTION & SERVER                                                    │
│  ───────────────────                                                    │
│  redis-cli                          # Connect to localhost:6379         │
│  redis-cli -h host -p port          # Connect to specific host/port     │
│  redis-cli -a password              # Connect with password             │
│  redis-cli --tls                    # Connect with TLS                  │
│  redis-cli -n 2                     # Select database 2                 │
│                                                                         │
│  PING                               # Test connection (returns PONG)    │
│  INFO                               # Server info and statistics        │
│  INFO memory                        # Memory usage details              │
│  INFO replication                   # Replication status                │
│  CONFIG GET maxmemory               # Get config value                  │
│  CONFIG SET maxmemory 4gb           # Set config value                  │
│  CLIENT LIST                        # List connected clients            │
│  DBSIZE                             # Number of keys in database        │
│                                                                         │
│  STRING OPERATIONS                                                      │
│  ─────────────────                                                      │
│  SET key value                      # Set string value                  │
│  SET key value EX 3600              # Set with 1 hour TTL               │
│  SET key value NX                   # Set only if not exists            │
│  SET key value XX                   # Set only if exists                │
│  SETEX key 3600 value               # Set with TTL (seconds)            │
│  SETNX key value                    # Set if not exists                 │
│  GET key                            # Get string value                  │
│  MGET key1 key2 key3                # Get multiple values               │
│  MSET k1 v1 k2 v2                   # Set multiple values               │
│  INCR key                           # Increment by 1                    │
│  INCRBY key 10                      # Increment by 10                   │
│  DECR key                           # Decrement by 1                    │
│  APPEND key value                   # Append to string                  │
│  STRLEN key                         # String length                     │
│                                                                         │
│  HASH OPERATIONS                                                        │
│  ───────────────                                                        │
│  HSET key field value               # Set hash field                    │
│  HGET key field                     # Get hash field                    │
│  HMSET key f1 v1 f2 v2              # Set multiple fields               │
│  HMGET key f1 f2                    # Get multiple fields               │
│  HGETALL key                        # Get all fields and values         │
│  HDEL key field                     # Delete field                      │
│  HEXISTS key field                  # Check field exists                │
│  HINCRBY key field 1                # Increment field                   │
│  HKEYS key                          # Get all field names               │
│  HVALS key                          # Get all values                    │
│  HLEN key                           # Number of fields                  │
│                                                                         │
│  LIST OPERATIONS                                                        │
│  ───────────────                                                        │
│  LPUSH key value                    # Push to head                      │
│  RPUSH key value                    # Push to tail                      │
│  LPOP key                           # Pop from head                     │
│  RPOP key                           # Pop from tail                     │
│  LRANGE key 0 -1                    # Get all elements                  │
│  LRANGE key 0 9                     # Get first 10 elements             │
│  LLEN key                           # List length                       │
│  LINDEX key 0                       # Get element at index              │
│  LTRIM key 0 99                     # Keep only first 100               │
│  BRPOP key 30                       # Blocking pop (30s timeout)        │
│  BLPOP key 30                       # Blocking pop from head            │
│                                                                         │
│  SET OPERATIONS                                                         │
│  ──────────────                                                         │
│  SADD key member                    # Add member                        │
│  SREM key member                    # Remove member                     │
│  SMEMBERS key                       # Get all members                   │
│  SISMEMBER key member               # Check membership                  │
│  SCARD key                          # Set cardinality (size)            │
│  SINTER key1 key2                   # Intersection                      │
│  SUNION key1 key2                   # Union                             │
│  SDIFF key1 key2                    # Difference                        │
│                                                                         │
│  SORTED SET OPERATIONS                                                  │
│  ─────────────────────                                                  │
│  ZADD key score member              # Add with score                    │
│  ZREM key member                    # Remove member                     │
│  ZSCORE key member                  # Get score                         │
│  ZRANK key member                   # Get rank (0-based)                │
│  ZREVRANK key member                # Get reverse rank                  │
│  ZRANGE key 0 9                     # Get by rank range                 │
│  ZREVRANGE key 0 9 WITHSCORES       # Top 10 with scores                │
│  ZRANGEBYSCORE key min max          # Get by score range                │
│  ZCARD key                          # Set size                          │
│  ZINCRBY key 1 member               # Increment score                   │
│                                                                         │
│  KEY OPERATIONS                                                         │
│  ──────────────                                                         │
│  DEL key                            # Delete key                        │
│  EXISTS key                         # Check exists (returns 0/1)        │
│  TYPE key                           # Get key type                      │
│  EXPIRE key 3600                    # Set TTL (seconds)                 │
│  PEXPIRE key 3600000                # Set TTL (milliseconds)            │
│  TTL key                            # Get TTL (seconds)                 │
│  PTTL key                           # Get TTL (milliseconds)            │
│  PERSIST key                        # Remove TTL                        │
│  RENAME key newkey                  # Rename key                        │
│  SCAN 0 MATCH pattern COUNT 100     # Iterate keys (safe)               │
│                                                                         │
│  ⚠️  AVOID IN PRODUCTION:                                               │
│  KEYS *                             # Blocks Redis (use SCAN instead)   │
│  FLUSHALL                           # Deletes everything                │
│  FLUSHDB                            # Deletes current database          │
│                                                                         │
│  TRANSACTIONS & SCRIPTING                                               │
│  ────────────────────────                                               │
│  MULTI                              # Start transaction                 │
│  EXEC                               # Execute transaction               │
│  DISCARD                            # Abort transaction                 │
│  WATCH key                          # Watch key for changes             │
│  EVAL "script" numkeys keys args    # Run Lua script                    │
│  EVALSHA sha numkeys keys args      # Run cached script                 │
│  SCRIPT LOAD "script"               # Load and cache script             │
│                                                                         │
│  STREAM OPERATIONS                                                      │
│  ─────────────────                                                      │
│  XADD stream * field value          # Add entry (auto ID)               │
│  XLEN stream                        # Stream length                     │
│  XRANGE stream - +                  # Read all entries                  │
│  XREAD STREAMS stream 0             # Read from beginning               │
│  XREAD BLOCK 5000 STREAMS stream $  # Blocking read (new only)         │
│  XGROUP CREATE stream group $ MKSTREAM  # Create consumer group        │
│  XREADGROUP GROUP g c STREAMS s >   # Read as consumer                  │
│  XACK stream group id               # Acknowledge message               │
│  XPENDING stream group              # Pending messages                  │
│                                                                         │
│  PUB/SUB                                                                │
│  ───────                                                                │
│  PUBLISH channel message            # Publish message                   │
│  SUBSCRIBE channel                  # Subscribe to channel              │
│  PSUBSCRIBE pattern                 # Subscribe to pattern              │
│                                                                         │
│  DEBUGGING & MONITORING                                                 │
│  ──────────────────────                                                 │
│  MONITOR                            # Watch all commands (careful!)     │
│  SLOWLOG GET 10                     # Get slow queries                  │
│  MEMORY USAGE key                   # Memory used by key                │
│  MEMORY DOCTOR                      # Memory issues report              │
│  DEBUG OBJECT key                   # Debug info about key              │
│  OBJECT ENCODING key                # Internal encoding                 │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Redis Patterns Cheat Sheet

```
REDIS PATTERNS CHEAT SHEET:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  PATTERN                    │ IMPLEMENTATION                            │
│  ──────────────────────────────────────────────────────────────────────│
│                                                                         │
│  CACHE-ASIDE               │ GET key                                    │
│  (Lazy Loading)            │ if miss: fetch DB → SET key val EX ttl    │
│                            │                                            │
│  WRITE-THROUGH             │ UPDATE DB → SET key val EX ttl            │
│                            │ (sync write to both)                       │
│                            │                                            │
│  WRITE-BEHIND              │ SET key val → queue DB write              │
│                            │ (async persist)                            │
│                            │                                            │
│  CACHE INVALIDATION        │ UPDATE DB → DEL cache_key                 │
│                            │ (delete on change)                         │
│                            │                                            │
│  DISTRIBUTED LOCK          │ SET lock:x owner NX EX 30                 │
│                            │ Release: Lua script (check owner first)   │
│                            │                                            │
│  RATE LIMIT (Fixed)        │ INCR rate:user:123:min:1530               │
│                            │ EXPIRE 60 (if count == 1)                 │
│                            │                                            │
│  RATE LIMIT (Sliding)      │ ZADD rate:user score=now member=reqid    │
│                            │ ZREMRANGEBYSCORE 0 (now-window)           │
│                            │ ZCARD rate:user                            │
│                            │                                            │
│  LEADERBOARD               │ ZADD leaderboard score player             │
│                            │ ZREVRANGE leaderboard 0 9 WITHSCORES      │
│                            │                                            │
│  SESSION STORAGE           │ SETEX session:id ttl json_data            │
│                            │ GET session:id                             │
│                            │                                            │
│  QUEUE (Simple)            │ Producer: LPUSH queue job                 │
│                            │ Consumer: BRPOP queue timeout             │
│                            │                                            │
│  QUEUE (Reliable)          │ XADD stream * field value                 │
│                            │ XREADGROUP GROUP g c STREAMS s >          │
│                            │ XACK stream group id                       │
│                            │                                            │
│  RECENT ITEMS              │ LPUSH recent item                          │
│                            │ LTRIM recent 0 99 (keep last 100)         │
│                            │                                            │
│  UNIQUE VISITORS           │ PFADD visitors:today user_id              │
│                            │ PFCOUNT visitors:today                     │
│                            │                                            │
│  FEATURE FLAGS             │ HSET features flag_name 1/0               │
│                            │ HGET features flag_name                    │
│                            │                                            │
│  COUNTING (Exact)          │ SADD users:active:today user_id           │
│                            │ SCARD users:active:today                   │
│                            │                                            │
│  COUNTING (Approx)         │ PFADD counter user_id                     │
│                            │ PFCOUNT counter (~0.81% error)            │
│                            │                                            │
│  PUB/SUB EVENTS            │ PUBLISH events:user:created json          │
│                            │ SUBSCRIBE events:user:*                    │
│                            │                                            │
│  DEDUPLICATION             │ SETNX processed:event:id 1 EX 86400       │
│                            │ if result == 1: process event              │
│                            │                                            │
│  SEMAPHORE                 │ LPUSH sem:resource tokens (N tokens)      │
│                            │ BRPOP sem:resource timeout (acquire)      │
│                            │ LPUSH sem:resource token (release)        │
│                            │                                            │
└─────────────────────────────────────────────────────────────────────────┘
```

### Configuration Structure

```
REDIS CONFIGURATION REFERENCE:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  MEMORY CONFIGURATION                                                   │
│  ────────────────────                                                   │
│  maxmemory 4gb                      # Maximum memory limit              │
│  maxmemory-policy allkeys-lru       # Eviction policy                   │
│                                                                         │
│  EVICTION POLICIES:                                                     │
│  • noeviction      - Return errors when limit reached                   │
│  • allkeys-lru     - Evict LRU from all keys (recommended for cache)   │
│  • allkeys-lfu     - Evict least frequently used                        │
│  • volatile-lru    - Evict LRU from keys with TTL                       │
│  • volatile-lfu    - Evict LFU from keys with TTL                       │
│  • volatile-ttl    - Evict keys with shortest TTL                       │
│  • allkeys-random  - Evict random keys                                  │
│  • volatile-random - Evict random keys with TTL                         │
│                                                                         │
│  PERSISTENCE CONFIGURATION                                              │
│  ─────────────────────────                                              │
│  # RDB Snapshots                                                        │
│  save 900 1                         # Save if 1 key changed in 15min   │
│  save 300 10                        # Save if 10 keys changed in 5min  │
│  save 60 10000                      # Save if 10000 keys in 1min       │
│  dbfilename dump.rdb                # RDB filename                      │
│  dir /var/lib/redis                 # Data directory                    │
│                                                                         │
│  # AOF (Append Only File)                                               │
│  appendonly yes                     # Enable AOF                        │
│  appendfilename "appendonly.aof"    # AOF filename                      │
│  appendfsync everysec               # Sync every second (recommended)   │
│  # appendfsync always               # Sync on every write (slow)        │
│  # appendfsync no                   # Let OS handle (fast, risky)       │
│                                                                         │
│  NETWORK CONFIGURATION                                                  │
│  ─────────────────────                                                  │
│  bind 127.0.0.1 10.0.0.1            # Listen interfaces                 │
│  port 6379                          # Listen port                       │
│  protected-mode yes                 # Require auth for external         │
│  timeout 0                          # Client timeout (0 = disabled)     │
│  tcp-keepalive 300                  # TCP keepalive interval            │
│                                                                         │
│  SECURITY CONFIGURATION                                                 │
│  ──────────────────────                                                 │
│  requirepass YourStrongPassword     # Password authentication           │
│                                                                         │
│  # ACL (Redis 6+)                                                       │
│  user default off                   # Disable default user              │
│  user app on >password ~* +@all     # Full access user                  │
│  user readonly on >pass ~* +@read   # Read-only user                    │
│                                                                         │
│  # Disable dangerous commands                                           │
│  rename-command FLUSHALL ""         # Disable                           │
│  rename-command FLUSHDB ""          # Disable                           │
│  rename-command DEBUG ""            # Disable                           │
│  rename-command CONFIG "CFG_x7k2"   # Rename to obscure name           │
│                                                                         │
│  # TLS Configuration                                                    │
│  tls-port 6379                      # TLS port                          │
│  port 0                             # Disable non-TLS                   │
│  tls-cert-file /path/redis.crt      # Server certificate                │
│  tls-key-file /path/redis.key       # Server private key                │
│  tls-ca-cert-file /path/ca.crt      # CA certificate                    │
│  tls-auth-clients yes               # Require client certs              │
│                                                                         │
│  REPLICATION CONFIGURATION                                              │
│  ─────────────────────────                                              │
│  # On replica:                                                          │
│  replicaof master_host 6379         # Set master                        │
│  masterauth master_password         # Master password                   │
│  replica-read-only yes              # Read-only replica                 │
│                                                                         │
│  CLUSTER CONFIGURATION                                                  │
│  ─────────────────────                                                  │
│  cluster-enabled yes                # Enable cluster mode               │
│  cluster-config-file nodes.conf     # Cluster state file                │
│  cluster-node-timeout 5000          # Node timeout (ms)                 │
│  cluster-replica-validity-factor 10 # Replica validity                  │
│                                                                         │
│  PERFORMANCE TUNING                                                     │
│  ──────────────────                                                     │
│  # Slow log                                                             │
│  slowlog-log-slower-than 10000      # Log commands > 10ms              │
│  slowlog-max-len 128                # Keep last 128 slow commands       │
│                                                                         │
│  # Client output buffer limits                                          │
│  client-output-buffer-limit normal 0 0 0                                │
│  client-output-buffer-limit replica 256mb 64mb 60                       │
│  client-output-buffer-limit pubsub 32mb 8mb 60                          │
│                                                                         │
│  # Lazy freeing (background deletion)                                   │
│  lazyfree-lazy-eviction yes         # Async eviction                    │
│  lazyfree-lazy-expire yes           # Async expiration                  │
│  lazyfree-lazy-server-del yes       # Async DEL                         │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

CONNECTION POOL CONFIGURATION (Application Side):

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  SETTING                  │ RECOMMENDED         │ NOTES                 │
│  ─────────────────────────┼─────────────────────┼─────────────────────  │
│  minConnections           │ 5-10                │ Pre-warmed pool       │
│  maxConnections           │ 20-50               │ Based on load         │
│  connectionTimeout        │ 1000-5000ms         │ Fail fast             │
│  socketTimeout            │ 1000-5000ms         │ Read/write timeout    │
│  idleTimeout              │ 30000-60000ms       │ Release unused        │
│  maxLifetime              │ 1800000-3600000ms   │ Refresh connections   │
│  retryAttempts            │ 3                   │ Retry on failure      │
│  retryDelay               │ 100-1000ms          │ Backoff between tries │
│                                                                         │
│  Example Pool Configuration:                                            │
│                                                                         │
│  pool = RedisPool({                                                     │
│      host: "redis.example.com",                                         │
│      port: 6379,                                                        │
│      password: getSecret("redis-password"),                            │
│      minConnections: 10,                                                │
│      maxConnections: 50,                                                │
│      connectionTimeout: 3000,                                           │
│      socketTimeout: 3000,                                               │
│      idleTimeout: 30000,                                                │
│      maxLifetime: 1800000,                                              │
│      retryAttempts: 3,                                                  │
│      retryDelay: 100,                                                   │
│      tls: { enabled: true }                                             │
│  })                                                                     │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 16. Why This Configuration Works

**In-Memory Data Store with Persistence**:
- All data resides in memory for sub-millisecond access, while AOF and RDB persistence options provide configurable durability guarantees without sacrificing read/write performance.

**Versatile Data Structures**:
- Native support for strings, hashes, lists, sets, sorted sets, streams, and HyperLogLog enables solving caching, queueing, leaderboards, rate limiting, and pub/sub with a single system instead of multiple specialized tools.

**Redis Cluster for Horizontal Scaling**:
- Automatic hash-slot-based sharding across nodes with built-in failover provides linear throughput scaling and high availability without external coordination services.

**Atomic Operations and Lua Scripting**:
- Single-threaded execution guarantees atomicity for individual commands, while server-side Lua scripts enable complex multi-key operations without race conditions or distributed locking overhead.

**Pub/Sub and Streams for Real-Time Messaging**:
- Redis Streams provide persistent, consumer-group-based message processing with acknowledgment and replay, bridging the gap between simple pub/sub and full message brokers for event-driven architectures.

---

## Related Guides

- **[kafka.md](kafka.md)**: Apache Kafka for event streaming (when Redis Streams isn't enough)
- **[microservices.md](microservices.md)**: Microservices patterns using Redis for caching and coordination
- **[kubernetes.md](kubernetes.md)**: Running Redis on Kubernetes
- **[designpatterns.md](designpatterns.md)**: Caching patterns and strategies


**End of Redis Best Practices and Style Guide**
