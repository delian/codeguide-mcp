# RethinkDB Development Guidelines
Mandatory standards for RethinkDB: realtime-first design with changefeeds, composable ReQL, index-driven access, and sharded/replicated clusters. RethinkDB 2.4, ReQL, official drivers (Python, Node).

---
name: rethinkdb
title: RethinkDB Development Guidelines
version: 2.0
last_reviewed: 2026-06-05
kind: datastore
tools: [rethinkdb@2.4, reql, rethinkdb-python@2.4, rethinkdbdash, rethinkdb-js]
requires:
  - secure-coding
  - error-handling
recommends:
  - observability
  - performance
  - websocket
  - env-config
provides:
  - reql
  - changefeeds
  - rethink-realtime
  - rethink-indexing
---

> 🧭 Authored per [`CONVENTIONS.md`](guides://CONVENTIONS.md): shared concerns are referenced, not restated. This guide covers only what is unique to RethinkDB.

> ⚠️ **Maintenance status — read before adopting.** RethinkDB's original company shut down in 2016; the code was open-sourced and stewardship passed (via The Linux Foundation / CNCF) to a volunteer community. Releases since are infrequent and security/driver maintenance is best-effort, **not** comparable to actively-developed databases (PostgreSQL, MongoDB, etc.). Choose RethinkDB only when its realtime changefeed model is a genuine differentiator for the workload, and budget for self-supporting the stack. For new greenfield realtime needs also evaluate alternatives: PostgreSQL `LISTEN/NOTIFY` + logical replication, MongoDB change streams ([`mongodb.md`](guides://mongodb.md)), or a CDC/streaming layer ([`kafka.md`](guides://kafka.md)).

---

## 0. Prerequisites & References

Fetch and apply these **before** generating RethinkDB code. Their rules are assumed below and not repeated.

> 📎 **REQUIRED — fetch & apply first:**
> - [`secure-coding.md`](guides://secure-coding.md) — secrets, supply chain, CVE policy. *(Binding: bind admin user + password, enable TLS, never expose ports 28015/29015/8080 publicly.)*
> - [`error-handling.md`](guides://error-handling.md) — error strategy & propagation. *(Binding: handle `ReqlDriverError` / `ReqlOpFailedError` and changefeed disconnects with reconnect + backfill.)*

> 📎 **RECOMMENDED — fetch when the task touches them:**
> - [`websocket.md`](guides://websocket.md) — changefeeds almost always fan out to browser clients over a realtime transport; bind a changefeed to a WebSocket/SSE channel, never expose the DB connection to the client.
> - [`performance.md`](guides://performance.md) — N+1 avoidance, batching, hot-path budgets *(binding: server-side ReQL, `eq_join`, index everything filtered/sorted).*
> - [`observability.md`](guides://observability.md) — metrics/tracing *(binding: scrape `r.db('rethinkdb').table('stats')`, alert on changefeed lag and replica health).*
> - [`env-config.md`](guides://env-config.md) — host/port/db/credentials come from config, never hardcoded.

> 📎 **SEE ALSO:** [`tdd.md`](guides://tdd.md) · [`code-review.md`](guides://code-review.md) · [`mongodb.md`](guides://mongodb.md) · [`kafka.md`](guides://kafka.md)

---

## 1. Core Philosophies: REALTIME-FIRST

RethinkDB-specific principles only. Security, error strategy, config, and observability come from §0.

- **R**ealtime push: design for `.changes()` from the start — let the server push deltas instead of polling. The changefeed is the reason to pick RethinkDB.
- **E**mbedded ReQL: queries are method chains in the host language (composable, type-checkable, injection-resistant), executed server-side via `.run(conn)` — not interpolated strings.
- **A**ccess by index: every `filter`/`order_by`/join key on a hot path MUST be backed by a secondary index; an un-indexed predicate is a full table scan.
- **L**ifecycle discipline: connections and changefeed cursors are resources — open via context manager/pool, always `.close()`, reconnect with backoff and re-subscribe on drop.
- **T**ransactional honesty: RethinkDB gives single-document atomicity only. No multi-document transactions — model so each invariant lives in one document; use atomic `update`/`r.row` expressions.
- **I**dempotent writes: changefeeds are at-least-once; design consumers to dedupe. Use deterministic primary keys and `conflict='update'` upserts.
- **M**odeled for the cluster: pick shard keys for even distribution, set replicas for HA, and choose `durability` per write criticality.
- **E**valuate first: confirm the maintenance trade-off (see banner) is acceptable for this project before building on it.

**Verified Code**: Agent-generated RethinkDB code MUST use composable ReQL (no string-built queries), handle connection and feed errors, index every hot-path predicate, and pass §2 gates before delivery.

---

## 2. Requirements (MANDATORY, auditable)

RFC-2119 keywords. IDs `RETHINK-<TOPIC>-<NN>`. Each row has a binary gate; rows binding a shared rule cite its owner.

| ID | Requirement | Verify | Gate |
|----|-------------|--------|------|
| RETHINK-SEC-01 | Cluster MUST require auth (admin password set) and bind/driver ports MUST NOT be publicly reachable (see `secure-coding.md`) | `r.db('rethinkdb').table('users')` review; `ss -tlnp` | auth on, no 0.0.0.0 exposure |
| RETHINK-SEC-02 | Driver↔server traffic MUST use TLS in production (see `secure-coding.md`) | connect with `ssl={'ca_certs': ...}`; config review | TLS enforced |
| RETHINK-SEC-03 | Queries MUST be composable ReQL; no `r.js`/string interpolation of untrusted input | grep for `r.js(` / string-concat queries | none with user input |
| RETHINK-CFG-01 | Connection params MUST come from config, not literals (see `env-config.md`) | grep for hardcoded host/port/password | none |
| RETHINK-ERR-01 | Connection & changefeed errors MUST be handled with reconnect/backfill (see `error-handling.md`) | review; kill-node integration test | feed resumes, no data loss |
| RETHINK-IDX-01 | Every hot-path `filter`/`order_by`/join key MUST use a secondary index (no table scans) | `query.run(conn, profile=True)` → no `SEQ_SCAN` on hot paths | indexed |
| RETHINK-RT-01 | Changefeeds MUST handle `include_types`/`initial` states and cursor close; consumers MUST be idempotent | review; reconnect test | dedupe + lifecycle correct |
| RETHINK-WRITE-01 | Write results MUST be checked (`errors`/`inserted`/`replaced`); critical writes MUST set `durability='hard'` | review write-result handling | checked + durable |
| RETHINK-MODEL-01 | Invariants MUST fit one document (no multi-doc transaction assumed); updates atomic | schema/review | single-doc atomic |
| RETHINK-HA-01 | Production tables MUST have ≥3 replicas and a deliberate shard count | `r.table(t).config()` review | replicas≥3 |
| RETHINK-OBS-01 | Cluster + changefeed lag MUST be monitored (see `observability.md`) | `rethinkdb.table('stats')` scraped | alerts wired |

> **Forbidden**: exposing 8080/28015/29015 to the internet; building queries by string concatenation; assuming multi-document transactions; un-indexed filters on hot paths; ignoring write-result `errors`; leaking the DB connection to browser clients (proxy changefeeds through a server — see `websocket.md`).

---

## 3. Verification Protocol

Run before presenting code. Fix → re-run until green.

```bash
# Static review
grep -rn "r.js(" src/                 # RETHINK-SEC-03: no server-side JS on user input
grep -rn "host='" src/                # RETHINK-CFG-01: no hardcoded connection params
ss -tlnp | grep -E '28015|29015|8080' # RETHINK-SEC-01: ports not on 0.0.0.0 in prod
```
```python
# Query profiling — confirm index usage (RETHINK-IDX-01)
res = query.run(conn, profile=True)   # inspect plan; reject SEQ_SCAN on hot paths
# Replica/shard check (RETHINK-HA-01)
r.table('users').config().run(conn)   # shards/replicas as intended
# Cluster health (RETHINK-OBS-01)
r.db('rethinkdb').table('current_issues').run(conn)   # must be empty
```

The *why* behind each gate lives in its §0 owner; do not re-derive it here.

---

## 4. Connection & Project Structure

Use the **instantiated** driver (`RethinkDB()`), not the legacy global import. Pool connections; never share one connection across threads/coroutines.

```python
from rethinkdb import RethinkDB
r = RethinkDB()                       # modern 2.4 driver; r.set_loop_type('asyncio') for async

# Config from env (see env-config.md), never literals
conn = r.connect(
    host=settings.RETHINK_HOST, port=28015,
    db=settings.RETHINK_DB,
    user=settings.RETHINK_USER, password=settings.RETHINK_PASSWORD,
    ssl={"ca_certs": settings.RETHINK_CA},   # TLS in prod (RETHINK-SEC-02)
)
```

```
project/
├── src/
│   ├── domain/        # entities + invariants (one invariant per document)
│   ├── repositories/  # ReQL queries, one module per table
│   ├── feeds/         # changefeed subscribers → app event bus / websocket
│   └── adapters/      # connection pool, config
├── migrations/        # table_create + index_create, idempotent
└── tests/             # against a disposable test DB (see tdd.md)
```

- One repository module per table; keep ReQL out of HTTP handlers.
- Changefeed subscribers live behind a supervised task that reconnects on drop.

---

## 5. ReQL — the Query Language (owned)

ReQL is a **chainable, embedded** DSL: each method returns a query object; nothing executes until `.run(conn)`. Because queries are built from native expressions (not strings), they are inherently parameterized and injection-resistant.

```python
# CRUD — every write returns a result dict; CHECK it (RETHINK-WRITE-01)
res = r.table("users").insert(doc, conflict="update", durability="hard").run(conn)
if res["errors"]:
    raise WriteError(res["first_error"])     # see error-handling.md
new_id = res["generated_keys"][0]            # UUID if no primary key supplied

r.table("users").get(uid).update({"age": r.row["age"] + 1}).run(conn)  # ATOMIC, server-side
r.table("users").get(uid).replace(full_doc).run(conn)
r.table("users").get(uid).delete().run(conn)

# Composable server-side query — runs in the cluster, returns a cursor
cursor = (r.table("orders")
          .get_all("paid", index="status")          # index, not a scan
          .filter(lambda o: o["total"] > 100)
          .order_by(index=r.desc("created_at"))
          .pluck("id", "user_id", "total")           # fetch only needed fields
          .limit(50)
          .run(conn))
for doc in cursor:                                   # stream; don't list() huge results
    ...
```

- **Atomic updates**: `update({"n": r.row["n"] + 1})` is atomic per document. A function that can't be proven deterministic (e.g. `r.js`, `r.now`) requires `non_atomic=True` — avoid it on user input.
- **`conflict`**: `"error"` (default), `"replace"`, `"update"` — use `"update"` for idempotent upserts.
- **`durability`**: `"hard"` (fsync, default) vs `"soft"` (ack before disk) — `"soft"` only for loss-tolerant high-throughput writes.
- **Cursors stream**: iterate; only `list()` bounded results.

---

## 6. Changefeeds — the Differentiator (owned)

`.changes()` turns any query into a live feed: the server pushes `{old_val, new_val}` deltas as matching documents change. This is RethinkDB's core value; bind feeds to a realtime transport for clients (see [`websocket.md`](guides://websocket.md)) — never hand the DB connection to a browser.

```python
feed = (r.table("messages")
        .filter({"room": room_id})
        .changes(include_initial=True, include_types=True, squash=True)
        .run(conn))
try:
    for c in feed:
        match c["type"]:                  # include_types=True
            case "initial" | "add":  push_to_websocket(c["new_val"])
            case "change":           push_to_websocket(c["new_val"])
            case "remove":           push_removal(c["old_val"])
            # "state" / "uninitial" also possible
except r.ReqlError as e:
    log.warning("feed dropped", error=e)  # reconnect + re-subscribe (RETHINK-RT-01)
finally:
    feed.close()                          # cursors are resources
```

Feed options that matter:
- `include_initial=True` — emit current matches before live deltas (snapshot + tail in one feed; pair with `include_types` to tell `initial` from `add`).
- `include_types=True` — tag each change `add|remove|change|initial|uninitial|state`.
- `squash=True` / `squash=N` — coalesce rapid changes to one doc (reduces client churn).
- Feeds work on point queries (`.get(id).changes()`), filtered queries, `get_all`/`between` ranges, and some joins (`eq_join(...).zip().changes()`); `order_by(index=...).limit()` gives ordered feeds.

**At-least-once**: a reconnect may replay; consumers MUST be idempotent (key by `new_val.id` + version/timestamp). On drop, reconnect with backoff and re-issue the feed (use `include_initial` to backfill missed state).

---

## 7. Data Modeling (owned)

Document-oriented JSON. Because there are **no multi-document transactions**, model so every invariant lives in a single document (RETHINK-MODEL-01).

- **Embed** one-to-few, read-together data (post + its tags) → one atomic read/write.
- **Reference** one-to-many / many-to-many (users ↔ posts) via foreign keys + an index on the join field; resolve with `eq_join` server-side.
- **Hybrid**: embed a denormalized snapshot of hot fields (author name on a comment) and accept eventual reconciliation via changefeed.
- **Primary key** is `id` (auto-UUID) unless a natural deterministic key enables idempotent upserts — then set it explicitly.
- **Time-series / append-heavy**: shard on a high-cardinality key, index `timestamp`, query ranges with `between(..., index="timestamp")`.

---

## 8. Indexing (owned)

A `filter` without an index is a table scan. Create the index, **wait for it**, then query through it.

```python
r.table("users").index_create("email").run(conn)                       # simple
r.table("users").index_create("city_age", [r.row["city"], r.row["age"]]).run(conn)  # compound
r.table("users").index_create("interests", multi=True).run(conn)       # multi (array fields)
r.table("users").index_create("full_name",
    lambda d: d["first"] + " " + d["last"]).run(conn)                   # functional
r.table("places").index_create("loc", geo=True).run(conn)              # geospatial
r.table("users").index_wait().run(conn)                                # MUST wait before use
```

Query through indexes:
```python
r.table("users").get_all(email, index="email").run(conn)               # equality
r.table("users").get_all([city, age], index="city_age").run(conn)      # compound equality
r.table("users").between(18, 30, index="age").run(conn)                # range
r.table("users").get_all(tag, index="interests").run(conn)             # multi-index
r.table("places").get_intersecting(r.circle(r.point(lng, lat), 500, unit="m"),
                                    index="loc").run(conn)              # geo (GeoJSON is [lng,lat])
```

- Compound index field order = allowed prefix queries and `order_by(index=...)`.
- Indexes cost write throughput and disk — index what you actually filter/sort/join.
- Confirm usage with `run(conn, profile=True)`.

---

## 9. Joins & Aggregation (owned)

Joins run server-side; always join on an indexed field to avoid scans.

```python
# Inner join + flatten
r.table("posts").eq_join("user_id", r.table("users")).zip().run(conn)
# Join via secondary index
r.table("posts").eq_join("user_id", r.table("users"), index="id").zip().run(conn)
# Left-outer-ish via map/branch
(r.table("posts").eq_join("user_id", r.table("users"), ordered=True)
   .map(lambda j: j["left"].merge(j.has_fields("right").branch(j["right"], {})))
   .run(conn))
# Aggregation
r.table("orders").group("status").count().run(conn)
r.table("orders").group("user_id").sum("total").run(conn)
r.table("events").map(lambda e: e["amount"]).reduce(lambda a, b: a + b).run(conn)
```

Prefer `eq_join` over fetching IDs then querying per row (the N+1 trap — see [`performance.md`](guides://performance.md)). `concat_map`/`group`/`map`/`reduce` keep aggregation in the cluster.

---

## 10. Cluster: Sharding, Replication & Durability (owned)

RethinkDB auto-shards and replicates; you set the topology per table and the consistency knobs per query.

```python
# Topology (RETHINK-HA-01)
r.table("users").reconfigure(shards=3, replicas=3).run(conn)
r.table("users").wait(wait_for="ready_for_writes").run(conn)
r.table("users").config().run(conn)      # inspect shard/replica placement
r.table("users").status().run(conn)      # ready_for_reads/writes, all_replicas_ready
```

- **Shards** spread data for write/throughput scaling — choose a shard key with even, high-cardinality distribution. **Replicas** provide HA + read scaling; ≥3 enables majority failover.
- **Consistency knobs per query**: `read_mode` = `single` (default, from primary) | `majority` (consistent) | `outdated` (fast, possibly stale); `durability` = `hard`|`soft`.
- **Failover** is automatic when a replica majority survives. Loss of majority needs manual `reconfigure(emergency_repair=...)` and accepts data loss — avoid by keeping odd replica counts across failure domains.
- Cluster nodes join via the intracluster port (29015); the web admin UI (8080) and driver port (28015) MUST stay private (RETHINK-SEC-01).

---

## 11. When RethinkDB Fits

- **Good fit**: collaborative apps, live dashboards, multiplayer/chat, streaming analytics, notification fan-out — anything where pushing query-result deltas beats polling.
- **Poor fit**: workloads needing multi-document ACID transactions, heavy relational joins, or a guarantee of long-term vendor/security maintenance — see the maintenance banner and prefer a mainstream store.

---

## 12. Deployment Checklist

Generated from §2 — one box per requirement ID.

- [ ] RETHINK-SEC-01 — auth enabled, ports 8080/28015/29015 private
- [ ] RETHINK-SEC-02 — driver↔server TLS in production
- [ ] RETHINK-SEC-03 — composable ReQL only, no `r.js`/string queries on user input
- [ ] RETHINK-CFG-01 — connection params from config, no literals
- [ ] RETHINK-ERR-01 — connection/changefeed errors reconnect + backfill
- [ ] RETHINK-IDX-01 — hot-path predicates indexed, no SEQ_SCAN
- [ ] RETHINK-RT-01 — changefeed states/lifecycle handled, consumers idempotent
- [ ] RETHINK-WRITE-01 — write results checked, critical writes `durability='hard'`
- [ ] RETHINK-MODEL-01 — invariants fit one document, updates atomic
- [ ] RETHINK-HA-01 — ≥3 replicas, deliberate shard count
- [ ] RETHINK-OBS-01 — cluster + changefeed lag monitored
- [ ] Maintenance trade-off reviewed and accepted (§ banner)
- [ ] Agent ran every §3 check and documented any fixes

---
**End of RethinkDB Guidelines**
