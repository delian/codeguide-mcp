# Redis Development Guidelines
Mandatory standards for Redis/Valkey: right data structure, TTL everywhere, atomic ops, graceful degradation. Redis 7.4+ / Valkey 8+, redis-cli, ACLs, Streams, Functions/Lua.

---
name: redis
title: Redis Development Guidelines
version: 2.0
last_reviewed: 2026-06-05
kind: datastore
tools: [redis@7.4, valkey@8, redis-cli]
requires:
  - secure-coding
  - error-handling
recommends:
  - observability
  - performance
  - docker-compose
  - env-config
  - microservices
provides:
  - redis-data-structures
  - caching-patterns
  - redis-streams
  - distributed-locks
  - eviction-persistence
---

> 🧭 Authored per [`CONVENTIONS.md`](guides://CONVENTIONS.md): shared concerns are referenced, not restated. This guide covers only what is unique to Redis.

> **Redis vs Valkey:** Redis relicensed under RSALv2/SSPLv1 (7.4+); the Linux Foundation **Valkey** fork (BSD) is the drop-in OSS successor and is wire/command compatible. Everything below applies to both. Pick per your license constraints; for self-hosted OSS deployments prefer Valkey.

---

## 0. Prerequisites & References

Fetch and apply these **before** generating Redis code or configuration. Their rules are assumed below and not repeated.

> 📎 **REQUIRED — fetch & apply first:**
> - [`secure-coding.md`](guides://secure-coding.md) — secrets, network exposure, supply chain. *(Redis binding: ACLs, TLS, `bind`/`protected-mode`, command renaming — see §8.)*
> - [`error-handling.md`](guides://error-handling.md) — error strategy, retries, circuit breakers. *(Redis binding: connection/failover handling, graceful degradation — see §6.)*

> 📎 **RECOMMENDED — fetch when the task touches them:**
> - [`observability.md`](guides://observability.md) — metrics/tracing policy *(binding: `INFO`, `SLOWLOG`, `LATENCY`, keyspace metrics — §9)*
> - [`performance.md`](guides://performance.md) — perf methodology *(binding: pipelining, memory, hot keys — §7)*
> - [`env-config.md`](guides://env-config.md) — config & connection-string policy *(binding: never hardcode host/password)*
> - [`docker-compose.md`](guides://docker-compose.md) — local Redis/Sentinel/Cluster topology
> - [`microservices.md`](guides://microservices.md) — Redis for cross-service caching, rate limiting, locks

> 📎 **SEE ALSO:** [`designpatterns.md`](guides://designpatterns.md) · [`kafka.md`](guides://kafka.md) *(when Streams isn't enough)* · [`kubernetes.md`](guides://kubernetes.md) · [`memcached.md`](guides://memcached.md) *(pure cache alternative)*

---

## 1. Core Philosophies: REDIS-FIRST

Redis-specific principles only. Security, error handling, observability, and performance methodology come from §0.

- **R**ight data structure: pick the optimal structure per access pattern (§3) — it is the single most consequential Redis decision.
- **E**fficient keys: namespaced, scannable, bounded-length keys (§4); shard hot keys.
- **D**ata expires: almost every key MUST carry a TTL — Redis is a cache/derived store, not the source of truth (§4.B, §1 cache-vs-primary).
- **I**dempotent & atomic: design for retry safety; use single commands, `MULTI/WATCH`, or Lua/Functions for atomicity (§5).
- **S**ingle-purpose, single-threaded: one instance = one focused workload; never block the event loop (no `KEYS`, no big `SAVE`, no O(N) on huge keys).

**Cache vs primary store.** Default posture: Redis is a *cache or derived index*, the durable system of record lives elsewhere; the application MUST work (degraded) when Redis is down (§6). Using Redis as a primary store is a deliberate exception requiring AOF `appendfsync everysec`/`always`, replication, and accepting its durability window (§4.D) — not the default.

**Verified Code**: Agent-generated Redis code/config MUST pass every gate in §2 before delivery.

---

## 2. Requirements (MANDATORY, auditable)

RFC-2119 keywords. IDs `REDIS-<TOPIC>-<NN>`. Each row has a binary gate; rows binding a shared rule cite its owner.

| ID | Requirement | Verify | Gate |
|----|-------------|--------|------|
| REDIS-DS-01 | Data structure MUST fit the access pattern (no JSON-blob-in-STRING when fields are read/written partially → HASH) | review vs §3 table | justified per key |
| REDIS-KEY-01 | Keys MUST follow `type:id[:field]` colon namespacing; no wildcards/spaces in key names | review / `--scan` sample | consistent |
| REDIS-TTL-01 | Every cache/session/lock/rate key MUST set a TTL (with jitter for bulk) | `redis-cli --scan` + `TTL` sample | no `-1` on cache keys |
| REDIS-MEM-01 | `maxmemory` AND `maxmemory-policy` MUST be set explicitly | `redis-cli CONFIG GET maxmemory maxmemory-policy` | non-zero + non-`noeviction`* |
| REDIS-ATOM-01 | Multi-step read-modify-write MUST be atomic (single cmd, `WATCH`/`MULTI`, or Lua/Function) | review | no check-then-act race |
| REDIS-LOCK-01 | Distributed locks MUST be owner-fenced & released via Lua compare-del; never bare `DEL` | review | owner-checked release |
| REDIS-BLOCK-01 | No O(N) blocking cmds on prod hot path (`KEYS`, `FLUSHALL`, `SAVE`, full `SMEMBERS`/`HGETALL` on huge keys) | `grep`/`SLOWLOG GET` | none in code; slowlog clean |
| REDIS-PIPE-01 | Bulk/independent ops MUST pipeline or use multi-key cmds (see `performance.md`) | review | no N round-trip loops |
| REDIS-SEC-01 | AUTH/ACL on, no internet exposure, dangerous cmds renamed/disabled (see `secure-coding.md`) | `redis-cli ACL WHOAMI` / `CONFIG GET bind protected-mode` | auth required, not 0.0.0.0 |
| REDIS-SEC-02 | Credentials & TLS from config/secrets, never hardcoded (see `env-config.md`,`secure-coding.md`) | grep source/config | no literals |
| REDIS-ERR-01 | App degrades gracefully on Redis outage; pooled conns w/ timeouts & retry (see `error-handling.md`) | failure test (kill Redis) | app serves degraded |
| REDIS-PERSIST-01 | Persistence (RDB/AOF) MUST match the data's durability requirement, documented | `CONFIG GET save appendonly` | matches stated need |
| REDIS-OBS-01 | `SLOWLOG` enabled; memory/latency/keyspace monitored (see `observability.md`) | `CONFIG GET slowlog-log-slower-than` | enabled, dashboarded |

> *`noeviction` is permitted only when Redis is the documented primary store (§1). **Forbidden**: `KEYS`/`FLUSHALL`/synchronous `SAVE` on a production instance; non-atomic check-then-act; releasing a lock you may not own; storing the only copy of critical data in a cache-configured instance; hardcoded credentials; Redis bound to a public interface without ACL+firewall.

---

## 3. Data Structures — the core decision

Pick by access pattern. All commands below are `redis-cli` syntax (language clients mirror them).

| Structure | Use for | Key complexity | Notes |
|---|---|---|---|
| **String** | cache blob, counter, flag, lock, rate counter | O(1) get/set, `INCR` | integers stored compactly; max 512MB (keep ≪1MB) |
| **Hash** | object with fields (partial read/write) | O(1)/field, O(N) all | memory-efficient (listpack) for small hashes — prefer over many strings |
| **List** | FIFO/LIFO queue, recent-N feed | O(1) ends, O(N) index | `LPUSH`+`BRPOP` queue; `LTRIM` to bound |
| **Set** | unique membership, tags, set algebra | O(1) add/check | `SINTER`/`SUNION`/`SDIFF` |
| **Sorted Set (ZSET)** | leaderboard, priority queue, time-series, sliding-window rate limit | O(log N) add, O(log N+M) range | score = points/timestamp; `ZRANGEBYSCORE`, `ZADD GT/LT` |
| **Stream** | event log, durable queue w/ consumer groups, CDC | O(1) append | replayable + acked delivery — see §5C |
| **HyperLogLog** | approx cardinality (uniques) | O(1), 12KB fixed | ~0.81% error; `PFADD`/`PFCOUNT`/`PFMERGE` |
| **Bitmap** | per-id boolean (DAU, feature flags) | O(1) bit | `SETBIT`/`BITCOUNT`; ~1 bit/id |
| **Geo** | nearby/radius search | O(log N) add | `GEOADD`/`GEOSEARCH` (ZSET-backed) |
| **Bitfield / probabilistic** | packed ints; Bloom/Cuckoo/Count-Min/Top-K via RedisBloom/Valkey modules | varies | use modules instead of rolling your own |

```bash
# STRING: cache w/ TTL, atomic counter, lock
SET user:123:profile '{...}' EX 3600
INCR rate:user:123:minute:202606271530
SET lock:order:456 <owner-uuid> NX EX 30
# HASH: partial object access (cheaper than N strings)
HSET user:123 name John email j@x.com plan premium
HINCRBY user:123 login_count 1
HMGET user:123 name email
# LIST: queue + bounded recent feed
LPUSH queue:emails '{job}';  BRPOP queue:emails 30
LPUSH user:123:recent product:789;  LTRIM user:123:recent 0 99
# SET: uniques + algebra
SADD daily:visitors:2026-06-27 user:123;  SISMEMBER daily:visitors:2026-06-27 user:123
SINTER daily:visitors:2026-06-27 daily:visitors:2026-06-26
# ZSET: leaderboard, time-range, sliding-window count
ZADD leaderboard 1500 alice 1200 bob;  ZREVRANGE leaderboard 0 9 WITHSCORES
ZRANGEBYSCORE user:123:events 1705849000 1705850000
# HLL / bitmap
PFADD uniq:2026-06 user:123;  PFCOUNT uniq:2026-06
SETBIT dau:2026-06-27 123 1;   BITCOUNT dau:2026-06-27
```

---

## 4. Key design, TTL, eviction, persistence

### A. Key naming
`object-type:id:field[:sub]`, colon-separated. Good: `user:123:profile`, `cache:api:products:list:a1b2c3`, `rate:api:user:123:minute:1530`, `lock:order:456`, `session:<id>`, `stream:orders:events`. Bad: bare `123`, mixed separators (`user_123_profile`), function-style (`getUser:123`), `>5` levels deep, wildcards/spaces in names, secrets in key names. Keep keys short for very high-cardinality keyspaces (memory adds up), but not at the cost of readability.

**Cluster:** slot = `CRC16(key) mod 16384`. Force related keys to one slot with a **hash tag** `{...}` so multi-key ops/Lua work: `user:{123}:profile`, `user:{123}:settings` → same slot → `MGET user:{123}:profile user:{123}:settings`.

### B. TTL strategy
Almost every key gets a TTL (REDIS-TTL-01). Set at creation: `SET k v EX 3600` / `SETEX`; on existing: `EXPIRE`/`PEXPIRE`/`EXPIREAT`. Check with `TTL` (`-1` = no expiry, `-2` = missing). Typical: sessions 1–7d, API cache 1m–1h, rate windows = window size, locks 10–30s (with renewal), aggregations 1–24h.

**Anti-stampede:** add random jitter so bulk keys don't expire together: `ttl = base + rand(0, base*0.1)`. For hot keys also use early/probabilistic recompute, not synchronous mass reload.

### C. Eviction (`maxmemory-policy`) — set it explicitly (REDIS-MEM-01)
```
maxmemory 4gb
maxmemory-policy allkeys-lru     # pure cache: evict any LRU key
```
| Policy | When |
|---|---|
| `allkeys-lru` / `allkeys-lfu` | pure cache; LFU when access skews to a hot subset |
| `volatile-lru`/`-lfu`/`-ttl` | mixed: only evict keys that have a TTL (protects persistent keys) |
| `allkeys-random`/`volatile-random` | rarely; cheap eviction |
| `noeviction` | primary-store only — writes error at limit instead of evicting |

### D. Persistence — RDB vs AOF (durability tradeoff)
| | RDB snapshot | AOF (append-only log) |
|---|---|---|
| What | periodic point-in-time fork dump | logs every write; replayed on restart |
| Durability | lose everything since last snapshot | `everysec` → ≤1s loss; `always` → ~0 (slow); `no` → OS-flushed |
| Restart/restore | fast load, compact | slower replay, larger file (auto-rewrite compacts) |
| Use | backups, replica seeding, tolerable loss | low data-loss requirement |
```
save 900 1                 # RDB: snapshot if ≥1 key changed in 15m
save 300 10
appendonly yes             # AOF (combine with RDB for best of both)
appendfsync everysec       # recommended default; 'always' = durable+slow
```
Pure cache → RDB (or none) is fine. Redis-as-primary → AOF `everysec`+ replication, and accept the durability window. Always `BGSAVE`, never blocking `SAVE` (REDIS-BLOCK-01).

---

## 5. Atomicity: transactions, Lua, Functions

Redis is single-threaded, so individual commands are atomic. For multi-step logic:

### A. MULTI/EXEC + WATCH (optimistic locking)
Commands are **queued** then run sequentially at `EXEC` — **no rollback**, no conditional logic, all commands fixed upfront. `WATCH` aborts `EXEC` (returns nil) if a watched key changed → retry.
```bash
WATCH account:A
# read, decide in client...
MULTI
DECRBY account:A 100
INCRBY account:B 100
EXEC            # nil if account:A changed since WATCH → retry loop
```
Use `WATCH/MULTI` for simple compare-and-set; prefer Lua/Functions when you need reads-before-writes or branching.

### B. Lua scripts & Functions (preferred for atomic read-modify-write)
A script runs atomically server-side with conditionals/loops. Pass all accessed keys via `KEYS[]` (cluster-correctness), data via `ARGV[]`; stay deterministic (no `TIME`/`RANDOMKEY`); keep it short (it blocks the server). Load once, call by SHA:
```bash
SCRIPT LOAD "<lua>"          # → sha
EVALSHA <sha> 1 rate:user:123 100 60
```
**Redis 7+ Functions** (`FUNCTION LOAD`, libraries registered by name) are the modern successor to ad-hoc `EVAL`: versioned, named, persisted with the dataset, and replicated — prefer them over inline `EVAL` for anything reused across the codebase. Example atomic rate-limit body (increment-under-limit): `GET` → compare to limit → `INCR` + set `EXPIRE` on first hit → return count or `-1`.

### C. Streams + consumer groups (durable queues / event sourcing)
Append-only log; entry id `<ms>-<seq>` (`*` = auto). Unlike pub/sub it is **persistent, replayable, and acked**.
```bash
XADD orders:events MAXLEN '~' 100000 * action created order_id 123   # cap length
XGROUP CREATE orders:events processors $ MKSTREAM        # $=new only, 0=from start
XREADGROUP GROUP processors c1 COUNT 10 BLOCK 5000 STREAMS orders:events >
XACK orders:events processors <id>                       # ack after success
XAUTOCLAIM orders:events processors c1 60000 0           # reclaim stuck pending (7.0+)
XPENDING orders:events processors                        # inspect un-acked
```
Consumer pattern: read `>` (blocking) → process → `XACK` on success / leave un-acked to retry; periodically `XAUTOCLAIM` messages idle past a threshold (dead consumer). Bound length with `MAXLEN ~`. When you need partitioning, long retention, or cross-datacenter, reach for [`kafka.md`](guides://kafka.md) instead.

### D. Pub/Sub
Fire-and-forget fan-out (`PUBLISH`/`SUBSCRIBE`), **no persistence or delivery guarantee** — subscribers offline at publish time miss the message. Use for cache-invalidation broadcasts and live notifications; use Streams when you need durability/replay.

---

## 6. Connection & failure handling (binding for `error-handling.md`)

Policy (retries, backoff, circuit breakers, graceful degradation) is owned by [`error-handling.md`](guides://error-handling.md). Redis binding:

- **Pool** connections (creating them is expensive); set bounded max, connect/socket timeouts (fail fast, 1–5s), idle/max-lifetime, retry-with-backoff, and optional `PING`-on-borrow. Source host/password/TLS from config/secrets (REDIS-SEC-02).
- **Degrade gracefully** (REDIS-ERR-01): cache-aside reads fall back to the source of truth on a Redis miss *or* error; wrap calls in a circuit breaker so an outage doesn't cascade. The app MUST serve (slower) without Redis.
- **HA topologies** — client must be failover-aware:
  - **Sentinel** (master + replicas + ≥3 sentinels): quorum detects master loss, elects a replica, reconfigures, notifies clients. Connect via a **Sentinel-aware client** (give it sentinel addresses + master name), not directly to a host.
  - **Cluster** (16384 slots sharded across masters, each with replica): use a **cluster client** that follows `MOVED`/`ASK` redirects; optionally read from replicas. Multi-key ops/Lua/transactions are confined to a single slot — use hash tags (§4.A). No cross-slot `SELECT`/db indexes.

---

## 7. Performance binding (for `performance.md`)

Methodology owned by [`performance.md`](guides://performance.md). Redis specifics:

- **Pipeline** independent commands into one round-trip (100–1000/batch, not >10k); use multi-key commands (`MGET`/`MSET`, `SCARD` instead of `len(SMEMBERS)`) (REDIS-PIPE-01).
- **Never `KEYS *`** in prod — use cursor-based `SCAN ... MATCH ... COUNT` (non-blocking) (REDIS-BLOCK-01).
- **Memory:** HASH (listpack) beats many strings for small objects; integers stored compactly; HLL/bitmap for counting; compress large values; split values ≫1MB. Inspect with `MEMORY USAGE <key>`, `MEMORY DOCTOR`, `INFO memory`.
- **Hot keys:** shard a hammered counter (`global:counter:shard:{n}`, aggregate on read); use read replicas. Enable `lazyfree-lazy-*` so big deletions/evictions free memory in the background.
- Avoid O(N) commands on huge collections (`HGETALL`/`SMEMBERS`/`LRANGE 0 -1`) on the hot path.

---

## 8. Security binding (for `secure-coding.md`)

Policy owned by [`secure-coding.md`](guides://secure-coding.md). Redis hardening (REDIS-SEC-01/02):

- **Never expose Redis to the internet.** `bind` to private interfaces, `protected-mode yes`, firewall to app hosts only.
- **AuthN/Z via ACLs** (Redis 6+), not just `requirepass`. Least privilege per user/key-pattern/command-category:
  ```
  ACL SETUSER app on >$(secret) ~cache:* +@read +@write -@dangerous
  ACL SETUSER readonly on >$(secret) ~* +@read -@write
  user default off                 # disable the default user
  ```
- **TLS** for any non-loopback traffic (`tls-port`, `port 0`, cert/key/CA, `tls-auth-clients yes`).
- **Rename/disable dangerous commands:** `rename-command FLUSHALL ""`, `FLUSHDB ""`, `DEBUG ""`, `CONFIG "<obscure>"`.
- App-encrypt sensitive values before storing; never put secrets in key names; TTL sensitive data; enable `slowlog` for audit/latency.
- Credentials come from secrets/config (see [`env-config.md`](guides://env-config.md)) — never literals in source or committed `redis.conf`.

---

## 9. Observability binding (for `observability.md`)

Policy owned by [`observability.md`](guides://observability.md). Redis signals to scrape/alert on:
- `INFO` sections: `memory` (`used_memory`, `maxmemory`, `mem_fragmentation_ratio`, `evicted_keys`), `stats` (`keyspace_hits/misses` → hit ratio, `instantaneous_ops_per_sec`), `clients` (`connected_clients`, `blocked_clients`), `replication` (`master_link_status`, lag), `persistence` (`rdb_last_bgsave_status`, `aof_last_write_status`).
- `SLOWLOG GET` / `slowlog-log-slower-than 10000` (REDIS-OBS-01); `LATENCY DOCTOR`/`LATENCY HISTORY` for spikes.
- Keyspace notifications (`notify-keyspace-events`) for expiry/eviction-driven workflows.
- Export via the Prometheus redis_exporter; alert on hit-ratio drop, eviction surge, replica-link down, memory near `maxmemory`.

---

## 10. Common patterns (Redis-owned)

- **Cache-aside (lazy):** read cache → on miss load source → `SETEX` → return. Resilient (works degraded), but has miss penalty + possible stampede (mitigate with TTL jitter/locks). The default caching pattern.
- **Write-through:** write source then cache synchronously — fresh cache, higher write latency.
- **Write-behind:** write cache + enqueue async DB persist (`LPUSH` job → worker `BRPOP`) — fast, but risks loss; not for critical data.
- **Invalidation:** prefer short TTL; explicit `DEL` on change for strong consistency; version-in-key (`INCR` version) or tag-sets (`SADD tag:... key`; `DEL` members) for grouped invalidation; pub/sub broadcast to evict app-local caches. Never `KEYS` to find keys to invalidate.
- **Rate limiting:** fixed-window (`INCR`+`EXPIRE`, cheap, edge-burst); sliding-window log (ZSET of timestamps, precise, memory-heavy); sliding-window counter (Lua, balanced); token bucket (Lua HASH, smooth bursts). Do the read-modify-write in one Lua/Function call (REDIS-ATOM-01).
- **Distributed locks (REDIS-LOCK-01):** `SET lock:x <uuid> NX PX <ttl>` to acquire; release via Lua compare-and-`DEL` (only if value == our uuid) — **never bare `DEL`** (you may delete someone else's renewed lock). Renew via Lua `PEXPIRE`-if-owner from a watchdog at ~ttl/3. **Redlock** (N independent instances, quorum N/2+1) tolerates instance failure but is debated (Kleppmann vs antirez) — it provides *liveness*, not guaranteed mutual exclusion under GC pauses/clock skew; for true correctness use a fencing token validated by the protected resource. Locks are for correctness, not performance.
- **Sessions:** `SETEX session:<secure-random-id> ttl <data>`; sliding expiration by refreshing TTL on access; optional `user:<id>:sessions` SET index for "log out everywhere". TTL everything.

---

## 11. Anti-patterns

- Redis as the **only** copy of critical data (cache-configured instance loses data on eviction/restart).
- Values ≫1MB; unbounded lists/sets/streams (use `LTRIM`/TTL/`MAXLEN`).
- `KEYS`/`FLUSHALL`/blocking `SAVE` in production; O(N) reads on huge keys on the hot path.
- No TTL on cache keys; no `maxmemory`/eviction policy set.
- Hot keys with no sharding/replicas.
- Bare-`DEL` lock release; non-atomic check-then-act.
- App crashing (not degrading) on Redis outage.
- N individual round-trips where a pipeline/multi-key command applies.

---

## 12. Quick Reference

```bash
# inspect
redis-cli INFO memory ; redis-cli INFO stats
redis-cli --scan --pattern 'user:*' | head      # never KEYS in prod
redis-cli MEMORY USAGE user:123
redis-cli SLOWLOG GET 10 ; redis-cli LATENCY DOCTOR
redis-cli ACL WHOAMI ; redis-cli CONFIG GET maxmemory maxmemory-policy
redis-cli CONFIG GET save appendonly
redis-cli --bigkeys ; redis-cli --hotkeys      # find offenders
# streams
redis-cli XINFO STREAM orders:events ; redis-cli XPENDING orders:events processors
```

---

## 13. Deployment Checklist

Generated from §2 — one box per requirement ID.

- [ ] REDIS-DS-01 — data structures fit access patterns
- [ ] REDIS-KEY-01 — consistent `type:id:field` colon namespacing
- [ ] REDIS-TTL-01 — every cache/session/lock/rate key has a TTL (jittered for bulk)
- [ ] REDIS-MEM-01 — `maxmemory` + `maxmemory-policy` set explicitly
- [ ] REDIS-ATOM-01 — multi-step writes atomic (single cmd / WATCH / Lua / Function)
- [ ] REDIS-LOCK-01 — locks owner-fenced, released via Lua compare-del
- [ ] REDIS-BLOCK-01 — no `KEYS`/`FLUSHALL`/`SAVE`/O(N)-on-huge-keys in prod; slowlog clean
- [ ] REDIS-PIPE-01 — bulk ops pipelined / multi-key
- [ ] REDIS-SEC-01 — ACL/AUTH on, not internet-exposed, dangerous cmds renamed (see `secure-coding.md`)
- [ ] REDIS-SEC-02 — credentials & TLS from secrets, no literals (see `env-config.md`)
- [ ] REDIS-ERR-01 — pooled conns w/ timeouts+retry; app degrades gracefully (see `error-handling.md`)
- [ ] REDIS-PERSIST-01 — RDB/AOF matches durability requirement, documented
- [ ] REDIS-OBS-01 — slowlog on; memory/latency/keyspace monitored (see `observability.md`)
- [ ] Agent ran the §12 inspection commands and documented any fixes

---
**End of Redis Guidelines**
