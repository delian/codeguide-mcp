# Memcached Development Guidelines
Mandatory standards for Memcached: a deliberately simple, volatile, multithreaded in-memory key-value cache. Memcached 1.6.x, meta commands, SASL, consistent-hashing clients (ketama), pymemcache/libmemcached, mcrouter.

---
name: memcached
title: Memcached Development Guidelines
version: 2.0
last_reviewed: 2026-06-05
kind: datastore
tools: [memcached@1.6, pymemcache, libmemcached, mcrouter]
requires:
  - secure-coding
recommends:
  - redis
  - performance
  - observability
  - error-handling
  - env-config
provides:
  - memcached-slab-model
  - simple-kv-cache
  - consistent-hashing
  - cache-stampede
---

> 🧭 Authored per [`CONVENTIONS.md`](guides://CONVENTIONS.md): shared concerns are referenced, not restated. This guide covers only what is unique to Memcached.

---

## 0. Prerequisites & References

Fetch and apply these **before** generating Memcached code. Their rules are assumed below and not repeated.

> 📎 **REQUIRED — fetch & apply first:**
> - [`secure-coding.md`](guides://secure-coding.md) — network exposure, secrets, supply chain. *(Memcached binding: never bind to a public interface; disable UDP; use SASL — see §7.)*

> 📎 **RECOMMENDED — fetch when the task touches them:**
> - [`redis.md`](guides://redis.md) — **the primary alternative.** Redis owns data-structure, persistence, pub/sub, server-side-cluster, and Lua-scripting patterns. Reach for Redis the moment you need any of those; this guide does **not** restate them. §9 is the honest pick-one decision.
> - [`error-handling.md`](guides://error-handling.md) — a cache is unreliable by design; the app MUST degrade on miss/down. *(binding: §6.)*
> - [`performance.md`](guides://performance.md) — slab/throughput tuning *(binding: §4, §8.)*
> - [`observability.md`](guides://observability.md) — metrics/alerts *(binding: `stats`, §8.)*
> - [`env-config.md`](guides://env-config.md) — server list, credentials, TTLs as config, never hardcoded.

> 📎 **SEE ALSO:** [`microservices.md`](guides://microservices.md) · [`kubernetes.md`](guides://kubernetes.md) · [`docker-compose.md`](guides://docker-compose.md)

---

## 1. Core Philosophies: CACHE-FIRST

Memcached-specific principles only. Security and error-handling come from §0.

- **C**ache, not store: Memcached is volatile RAM with **no persistence and no replication**. The source of truth lives elsewhere; a flushed or restarted node loses everything by design — and that is acceptable.
- **A**lways TTL: every item gets an explicit expiration. The cache is allowed to forget; correctness MUST NOT depend on an item still being present.
- **C**onstrained values: opaque blobs only — no server-side data structures, no queries, no scripting. Keys ≤ 250 bytes, values ≤ 1 MB (default). Need structure/persistence → that is `redis.md`, not Memcached.
- **H**ash on the client: there is **no server-side clustering**. Servers are shared-nothing; a consistent-hashing (ketama) client decides which node owns each key.
- **E**mbrace the miss: a miss and an unreachable node are normal control flow, not errors (see `error-handling.md`).

**Verified Code**: Agent-generated Memcached integrations MUST pass every gate in §2 before delivery.

---

## 2. Requirements (MANDATORY, auditable)

RFC-2119 keywords. IDs `MC-<TOPIC>-<NN>`. Each row has a binary gate; rows binding a shared rule cite its owner.

| ID | Requirement | Verify | Gate |
|----|-------------|--------|------|
| MC-NET-01 | Server MUST NOT listen on a public interface (see `secure-coding.md`) | `memcached -h` config / `ss -tlnp \| grep 11211` | bound to loopback/private only |
| MC-NET-02 | UDP MUST be disabled unless explicitly required (`-U 0`) — amplification history (see `secure-coding.md`) | `ps - ef \| grep memcached` / config | `-U 0` present |
| MC-SEC-01 | SASL auth MUST be enabled on any shared/multi-tenant network (`-S`) (see `secure-coding.md`) | server flags + client creds | auth required |
| MC-CFG-01 | Server list, credentials, and TTLs MUST come from config, never hardcoded (see `env-config.md`) | review / grep | no literals |
| MC-TTL-01 | Every stored item MUST set an explicit, non-zero TTL | grep `set(`/`ms` calls | no `expire=0` for cache data |
| MC-SIZE-01 | Keys MUST be ≤ 250 B and values ≤ item-size limit (1 MB default); MUST NOT split-store large blobs blindly | client-side validation test | oversize rejected before send |
| MC-MISS-01 | App MUST treat miss **and** node-down as normal and fall back to source (see `error-handling.md`) | test with cache stopped | serves correct data degraded |
| MC-STAMP-01 | Hot/expensive keys MUST have stampede mitigation (lock or probabilistic early refresh) | concurrent-miss test | ≤ 1 origin fetch under load |
| MC-HASH-01 | Multi-server clients MUST use consistent hashing (ketama) | inspect client config | ketama/consistent enabled |
| MC-POOL-01 | Connections MUST be pooled and reused; pool MUST not leak on error | pool-after-error test | available > 0 after failures |
| MC-OBS-01 | Hit ratio + evictions MUST be monitored and alerted (see `observability.md`) | dashboard/alert exists | metrics scraped |
| MC-TST-01 | Cache integration paths MUST have tests against a real/ephemeral instance | test suite | exit 0 |

> **Forbidden**: exposing Memcached to the internet; enabling UDP without need; storing data whose loss breaks correctness (sessions, the only copy of anything); relying on an item still being cached; using modulo (`hash % N`) sharding that reshuffles every key when a node is added/removed.

---

## 3. The Memcached Model (why it is simple on purpose)

Memcached deliberately does **one thing**: a volatile, multithreaded hash table of opaque blobs.

- **No persistence** — RAM only; restart = empty. (Want durability/snapshots → `redis.md`.)
- **No data structures** — values are bytes. No lists/sets/sorted-sets/hashes/queries/pub-sub/scripting. (Want those → `redis.md`.)
- **No server-side clustering or replication** — nodes never talk to each other (shared-nothing). Distribution is a *client* concern (§5).
- **Multithreaded** — scales vertically across cores via `-t <n>` (default 4). This is its headline advantage over Redis's single command thread: a big multi-core box saturates CPU for raw GET/SET throughput. (`mcrouter` adds routing/replication *in front of* nodes; nodes stay dumb.)
- **Tiny op surface**: `get/gets`, `set/add/replace/append/prepend/cas`, `delete`, `incr/decr`, `touch`, `flush_all`, `stats`. `add` = set-if-absent; `cas` = set-if-version-matches (optimistic concurrency via the token from `gets`).

> Its simplicity *is* the feature: fewer moving parts, lower per-key memory overhead, trivial to operate. Choose it when you want exactly a fast LRU cache and nothing more (§9).

---

## 4. Memory Model: Slabs, Chunks, Eviction

This is Memcached's defining internal mechanic — understand it or you will see items vanish before their TTL.

### Slab allocation
- Memory (`-m`, MB) is carved into 1 MB **pages**. Each page belongs to one **slab class**; a class stores fixed-size **chunks**.
- Chunk sizes grow geometrically by the **growth factor** (`-f`, default **1.25**): e.g. 96 B → 120 B → 152 B → … up to the max item size. Smallest chunk base is `-n` (default item overhead ~48 B).
- An item is placed in the **smallest chunk that fits**. A 130 B item in a 152 B chunk **wastes 22 B** — internal fragmentation. Tune `-f` down (e.g. 1.08–1.12) to pack a narrow size distribution tighter; tune up for fewer classes.
- Inspect reality, do not guess: `stats slabs`, `stats items`, `stats sizes`.

### Why items get evicted *before* their TTL
- LRU eviction is **per slab class**, not global. When class N is full, Memcached evicts the LRU item *in class N* — even if other classes have free pages. A flood of one size can evict live items of that size while RAM sits free elsewhere. This is **slab calcification**.
- Modern mitigation (default-on in 1.5+/1.6.x): the **slab automover** + `lru_crawler` reclaim and reassign pages between classes; the **segmented LRU** (HOT/WARM/COLD, plus TEMP for short TTLs) driven by `lru_maintainer` keeps hot items resident. Keep these enabled; watch `evictions`, `evicted_unfetched`, and `slab_reassign` counters.
- Rule: **rising evictions with non-zero free memory ⇒ a slab/size-distribution problem, not a capacity problem.** Add RAM only when evictions rise *and* memory is genuinely full.

```bash
# growth factor + larger max item size + explicit threads
memcached -m 2048 -f 1.12 -I 2m -t 8 -c 4096 -l 10.0.1.5 -U 0
```

---

## 5. Distribution: Client-Side Consistent Hashing

There is no cluster. With multiple nodes, the **client** maps key → node. Use **consistent hashing (ketama)** so adding/removing a node remaps only ~`1/N` of keys — naive `hash % N` remaps *almost all* keys and stampedes the origin.

- Use the library's built-in ketama hasher (`pymemcache`'s `HashClient`, `libmemcached`'s `MEMCACHED_DISTRIBUTION_CONSISTENT`) rather than rolling your own.
- ~100–200 virtual nodes per server keeps the ring balanced (verify with a key-distribution sample).
- A down node should fail over to the next point on the ring (its keys become misses → refetched). Keep node identity (name/port) **stable** so the ring is reproducible across clients.

```python
from pymemcache.client.hash import HashClient
client = HashClient(
    [("10.0.1.5", 11211), ("10.0.1.6", 11211), ("10.0.1.7", 11211)],
    use_pooling=True,            # MC-POOL-01
)  # HashClient uses consistent hashing across nodes — MC-HASH-01
```

> For routing, in-front replication, connection fan-out, or shadow pools, run **mcrouter** between clients and the (still dumb) nodes.

---

## 6. The Protocol: prefer meta commands

Three wire protocols exist; pick deliberately.

| Protocol | Status (1.6.x) | Use |
|---|---|---|
| **Text** (classic `get`/`set`) | supported, human-readable | debugging via `telnet`/`nc`; legacy clients |
| **Binary** | **deprecated**, slated for removal | do not build new systems on it |
| **Meta** (`mg/ms/md/ma/mn/me`) | **current, recommended** | new code — flag-driven, supports CAS/TTL/flags/base64 keys/probabilistic refresh hints in one round trip |

```
# Meta get: return value(v), TTL remaining(t), CAS(c), and bump nothing(q quiet)
mg user:123 v t c
# Meta set: 5-byte value, TTL 3600, with a CAS token requirement
ms user:123 5 T3600 C42
hello
# Meta arithmetic (incr by 1, autovivify to 0 with 60s TTL); meta delete
ma counter:hits N60 J0
md user:123
```

- Meta flags carry behaviors that previously needed separate commands (`gat`/`gets`/`touch`), cutting round trips.
- Keep using **text** for ad-hoc debugging: `echo -e 'stats\r' | nc 127.0.0.1 11211`.
- Optimistic concurrency: read a CAS token (`mg ... c` / `gets`), write only if it still matches (`ms ... C<token>` / `cas`); on mismatch, retry.

---

## 7. Security (bindings — policy is `secure-coding.md`)

Memcached has a notorious exposure history; the policy lives in [`secure-coding.md`](guides://secure-coding.md). Bindings:

- **Never reachable from the internet (MC-NET-01).** Bind to loopback or a private interface (`-l 10.0.1.5`); firewall 11211 to app subnets only. An open Memcached is an unauthenticated RAM dump *and* a write primitive.
- **Disable UDP (MC-NET-02): `-U 0`.** UDP on 11211 powered record-setting reflection/amplification DDoS attacks (2018, ~1.3 Tbps). Modern builds default UDP off — keep it off unless you have a measured reason.
- **SASL (MC-SEC-01): `-S`** for any shared/multi-tenant or untrusted-network deployment; pass credentials from secrets (`pymemcache` `username=/password=`). SASL requires the meta/binary protocol.
- **Validate before send (MC-SIZE-01):** enforce key ≤ 250 B (no spaces/control chars/newlines) and value ≤ item-size limit client-side; reject oversize rather than letting the server error.

---

## 8. Operations: Monitoring & Tuning

`stats` is the single source of operational truth; export it to your `observability.md` stack (MC-OBS-01).

- **Hit ratio** = `get_hits / (get_hits + get_misses)`. A persistently low ratio means wrong TTLs, wrong key set, or undersized memory.
- **`evictions`** rising → see §4 (capacity *or* slab calcification). Cross-check `bytes` vs `limit_maxbytes` and `evicted_unfetched`.
- **`curr_connections` / `listen_disabled_num`** → connection pressure; raise `-c` and pool on the client (MC-POOL-01).
- **`cmd_get/cmd_set`, `bytes_read/written`** → throughput/bandwidth.
- Useful breakdowns: `stats slabs`, `stats items`, `stats sizes`.

```bash
echo -e 'stats\r'       | nc 127.0.0.1 11211   # global counters
echo -e 'stats slabs\r' | nc 127.0.0.1 11211   # per-slab chunk/eviction detail
```

Connection pooling (`use_pooling=True`, bounded size) is mandatory; pools MUST return connections on error paths (try/finally) so failures don't exhaust them (MC-POOL-01).

---

## 9. Cache Patterns & the Stampede Problem

Standard caching patterns (cache-aside, write-through, write-behind, invalidation strategies) are owned by [`redis.md`](guides://redis.md) §4 and apply unchanged to Memcached — **do not duplicate them**; the only difference is the verb set (`get`/`set`/`add`/`delete`/`cas` instead of Redis commands). Memcached-specific notes:

- **Cache-aside** is the default: `get` → miss → load from source → `set` with TTL → return. Use `add` (set-if-absent) to avoid clobbering a concurrent writer.
- **No `KEYS`/`SCAN`** — you cannot enumerate keys. Invalidate by exact key, by versioned key (`user:123:v7`; bump the version to orphan old entries to TTL), or by `flush_all` (blunt). There is no tag/pattern delete.

### Cache stampede (thundering herd) — owned here
On a popular key's expiry, N concurrent misses all hit the origin at once. Mitigate:

1. **TTL jitter** — `base_ttl + random(0, spread)` so keys don't expire in lockstep.
2. **Single-flight lock** — first miss takes a short-lived lock (`add lock:<key>` with small TTL); others briefly wait/serve stale; only the lock holder refetches.
3. **Probabilistic early expiration (XFetch)** — refresh *before* expiry with probability rising as TTL nears, so one request refreshes ahead of the herd:

```python
import time, math, random
def get_xfetch(client, db, key, ttl=3600, beta=1.0):
    packed = client.get(key)                      # stored as (value, delta, expires_at)
    if packed:
        value, delta, expires_at = packed
        if time.time() - delta * beta * math.log(random.random()) < expires_at:
            return value                          # still fresh enough
    start = time.time()
    value = db.load(key)
    delta = time.time() - start                   # cost of recompute
    client.set(key, (value, delta, time.time() + ttl), expire=ttl)
    return value
```

---

## 10. Memcached vs Redis — pick one honestly

Both are in-memory caches; they diverge sharply beyond that. Redis owns the data-structure/persistence story (see [`redis.md`](guides://redis.md)) — bind the decision, don't restate Redis here.

| Dimension | Memcached | Redis |
|---|---|---|
| Model | Opaque blobs only | Strings, hashes, lists, sets, sorted sets, streams, … |
| Persistence | None (volatile) | RDB/AOF optional |
| Replication / cluster | None — client-side sharding | Built-in replication + Redis Cluster |
| Threading | **Multithreaded** (scales on cores) | Single command thread (+ I/O threads) |
| Per-key memory overhead | Lower | Higher (richer metadata) |
| Scripting / pub-sub / queues | No | Yes (Lua, pub/sub, streams) |
| Eviction | Per-slab LRU/segmented LRU | Multiple global policies (LRU/LFU/TTL) |

**Choose Memcached when:** you want *only* a fast volatile LRU cache; values are simple blobs; you want to scale GET/SET throughput across many cores on one box with minimal operational surface and the lowest per-key overhead.

**Choose Redis when:** you need persistence, replication/failover, server-side data structures, pub/sub, queues/streams, atomic multi-step ops (Lua), or anything whose loss is not acceptable (e.g. sessions). When in doubt and the use case might grow, default to Redis.

> **Anti-patterns:** sessions or any sole-copy data in Memcached (a restart loses them — use Redis/DB); caching > 1 MB objects (cache a summary/pointer instead, or raise `-I` deliberately); modulo sharding; treating a miss as an error.

---

## 11. Deployment Checklist

Generated from §2 — one box per requirement ID.

- [ ] MC-NET-01 — bound to loopback/private interface, firewalled to app subnets
- [ ] MC-NET-02 — UDP disabled (`-U 0`)
- [ ] MC-SEC-01 — SASL enabled on shared/untrusted networks
- [ ] MC-CFG-01 — servers, credentials, TTLs from config (no literals)
- [ ] MC-TTL-01 — every cached item has an explicit non-zero TTL
- [ ] MC-SIZE-01 — keys ≤ 250 B, values ≤ item limit, validated client-side
- [ ] MC-MISS-01 — app degrades correctly on miss and node-down
- [ ] MC-STAMP-01 — hot keys have stampede mitigation
- [ ] MC-HASH-01 — multi-server client uses consistent hashing (ketama)
- [ ] MC-POOL-01 — connections pooled, returned on error
- [ ] MC-OBS-01 — hit ratio + evictions monitored and alerted
- [ ] MC-TST-01 — cache paths tested against a real/ephemeral instance
- [ ] Agent ran the §8 `stats` checks and documented any tuning

---
**End of Memcached Guidelines**
