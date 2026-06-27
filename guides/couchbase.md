# Couchbase Development Guidelines
Mandatory standards for Couchbase: memory-first KV + JSON documents, SQL++ querying, GSI indexing, durable replication, offline mobile sync. Couchbase Server 7.6, SQL++, cbq, SDKs (Java/Python/Node.js), Couchbase Lite 3.x, Sync Gateway 3.x.

---
name: couchbase
title: Couchbase Development Guidelines
version: 2.0
last_reviewed: 2026-06-05
kind: datastore
tools: [couchbase-server@7.6, sql++, cbq, couchbase-cli, cbbackupmgr, couchbase-lite@3, sync-gateway@3]
requires:
  - secure-coding
  - error-handling
recommends:
  - observability
  - performance
  - sql
  - env-config
provides:
  - couchbase-kv-document
  - sql++-n1ql
  - gsi-indexing
  - xdcr
  - couchbase-mobile-sync
---

> 🧭 Authored per [`CONVENTIONS.md`](guides://CONVENTIONS.md): shared concerns are referenced, not restated. This guide covers only what is unique to Couchbase.

---

## 0. Prerequisites & References

Fetch and apply these **before** generating Couchbase code. Their rules are assumed below and not repeated.

> 📎 **REQUIRED — fetch & apply first:**
> - [`secure-coding.md`](guides://secure-coding.md) — secrets, TLS, supply chain, least privilege. *(Couchbase binding: RBAC roles/groups scoped to bucket·scope·collection, TLS client connections, encryption at rest.)*
> - [`error-handling.md`](guides://error-handling.md) — error strategy, retries, idempotency. *(Couchbase binding: SDK best-effort retry + backoff, CAS optimistic-locking conflicts, durability ambiguity on timeout.)*

> 📎 **RECOMMENDED — fetch when the task touches them:**
> - [`sql.md`](guides://sql.md) — SQL++ (N1QL) **is SQL for JSON**; general SELECT/JOIN/GROUP BY/window-function semantics live there. This guide covers only the JSON/Couchbase extensions.
> - [`performance.md`](guides://performance.md) — memory/working-set sizing, index covering, batch sizing.
> - [`observability.md`](guides://observability.md) — metrics/tracing *(binding: Prometheus `/metrics`, slow-query log, `system:` keyspaces)*.
> - [`env-config.md`](guides://env-config.md) — connection strings, credentials, env separation.

> 📎 **SEE ALSO:** [`tdd.md`](guides://tdd.md) *(test-first cycle the §2 CB-TST rows bind)* · [`memcached.md`](guides://memcached.md) · [`redis.md`](guides://redis.md) · [`mongodb.md`](guides://mongodb.md) · [`couchdb.md`](guides://couchdb.md)

---

## 1. Core Philosophies: DOCUMENT-FIRST

Couchbase-specific principles only. Security, error/retry strategy, and SQL fundamentals come from §0.

- **D**ocument model: design around self-describing JSON documents with a `type` discriminator; key-value access is the fast path, SQL++ is for ad-hoc/relational access.
- **O**perations: prefer KV (sub-millisecond, by-key) and **sub-document** ops over fetch-mutate-replace; reach for SQL++ only when you query by value, not by key.
- **C**luster awareness: design for vBucket auto-sharding, replicas, and XDCR; never assume a single node or a single datacenter.
- **U**se indexes: every SQL++ predicate that runs in production MUST be served by a GSI — a `PrimaryScan` in `EXPLAIN` is a defect, not a fallback.
- **M**emory-first: respect the managed-cache working-set model; size bucket RAM quota to the working set, not the dataset.
- **E**xplicit durability: choose a durability level per write by data criticality; never rely on the async default for money/state.
- **N**aming with scopes/collections: use `bucket`.`scope`.`collection` keyspaces (7.x) — `_default` only for trivial apps; collections replace the legacy `type`-field-only modeling.
- **T**est against a real topology: integration-test KV + SQL++ + index paths against a containerized cluster, not just a mock.

**Verified Code**: Agent-generated Couchbase code MUST use parameterized SQL++, index-backed queries (no primary scans), explicit durability for critical writes, and pass every §2 gate before delivery.

---

## 2. Requirements (MANDATORY, auditable)

RFC-2119 keywords. IDs `CB-<TOPIC>-<NN>`. Each row has a binary gate; rows binding a shared rule cite its owner.

| ID | Requirement | Verify | Gate |
|----|-------------|--------|------|
| CB-TST-01 | Every feature MUST be test-first against a real cluster or testcontainer (see `tdd.md`) | run integration suite | exit 0, 0 skips |
| CB-TST-02 | Each bug MUST get a regression test before the fix (see `tdd.md`) | rerun suite | failing→passing |
| CB-QRY-01 | All SQL++ MUST be parameterized — no string-built predicates (injection, see `secure-coding.md`) | grep/review for `$`/positional params | no interpolation |
| CB-IDX-01 | No production query MUST hit a `PrimaryScan`/full scan | `EXPLAIN <query>` | IndexScan only |
| CB-IDX-02 | Indexes MUST be defined as code (DDL committed), not ad-hoc | review migration files | DDL in VCS |
| CB-KV-01 | Concurrent mutations MUST use CAS or sub-document ops (no blind read-modify-write) | review | CAS/mutateIn used |
| CB-DUR-01 | Critical writes MUST set an explicit durability level (see `error-handling.md`) | review write options | level set |
| CB-SEC-01 | Connections MUST use TLS and a least-privilege RBAC user — never `Administrator` (see `secure-coding.md`) | inspect conn string + user roles | `couchbases://` + scoped role |
| CB-SEC-02 | Credentials MUST come from config/secret store, not source (see `env-config.md`) | grep for literals | no secrets in code |
| CB-KEY-01 | Document keys MUST follow a deterministic, type-prefixed scheme | review key builders | `type::id` scheme |
| CB-OBS-01 | Slow-query log + cluster metrics MUST be scraped (see `observability.md`) | check Prometheus targets / slow-query threshold | metrics present |
| CB-BAK-01 | Backups MUST run on a schedule and restore MUST be tested | `cbbackupmgr info` + restore drill | recent backup + verified restore |

> **Forbidden**: building SQL++ via string concatenation; shipping a query whose `EXPLAIN` shows a primary scan; blind read-modify-write without CAS; connecting as `Administrator` from an app; async-default durability for money/state; unbounded array growth inside a single document.

---

## 3. Architecture & Fundamentals

Couchbase is a distributed document database with an integrated memory-first cache, SQL++ query engine, and independently-scalable services.

**Memory-first (managed cache).** Active data (the *working set*) lives in RAM for sub-millisecond KV; all data persists to disk asynchronously (or synchronously, per durability). Size the bucket RAM quota to the working set, not the whole dataset. Bucket types: **Couchbase** (cache + persistence + indexing), **Ephemeral** (RAM-only, no disk — sessions/cache), **Memcached** (legacy pure cache; prefer Ephemeral).

**Namespacing (7.x).** `bucket` → `scope` → `collection` → document. Collections replace the old single-bucket + `type`-field pattern: model each entity as its own collection (`app`.`sales`.`orders`) for isolation, per-collection RBAC, and cleaner SQL++. `_default`.`_default` exists for migration/trivial cases only.

**vBuckets & auto-sharding.** Each bucket is split into 1024 vBuckets. `hash(documentID) → vBucket → node`. This gives even distribution, online rebalance, and no client-side shard logic. Replicas (0–3) are copies of each vBucket on other nodes; failover promotes a replica.

**Multi-Dimensional Scaling (MDS).** Services — **Data (KV)**, **Query (SQL++)**, **Index (GSI)**, **Search (FTS)**, **Eventing**, **Analytics** — run and scale on independent node sets so a heavy analytical or index workload cannot starve KV latency. DCP is the internal change-stream protocol feeding indexes, XDCR, Eventing, and Sync Gateway.

**When Couchbase fits:** sub-ms KV at high throughput; cache + system-of-record in one engine; multi-datacenter active-active (XDCR); offline-first mobile/edge (Lite + Sync Gateway); flexible/evolving JSON with mixed KV+SQL access. **Poor fit:** heavy multi-document ACID across entities (use PostgreSQL/CockroachDB), graph traversal (Neo4j), or large-scale columnar OLAP (use the Analytics service, ClickHouse, or a warehouse).

---

## 4. Data Modeling & Key Design

**Document = JSON + discriminator.** Keep a `type` field even with collections (cheap to filter, survives merges). Store timestamps as ISO-8601 strings or epoch ints consistently.

**Embed vs. reference:**
- **Embed** data read/written together and bounded in size (a user's settings, an order's line items).
- **Reference** when sub-entities are large, updated independently, or grow unboundedly. Then resolve via KV `getMulti` (fast) or SQL++ `ON KEYS` join.
- **Never** let an array inside one document grow unbounded — split into per-period/child documents (`activity::alice::2026-06-27`). Hard cap: 20 MB/doc; target < 1 MB.

**Key design (CB-KEY-01).** Keys are the primary access path — make them deterministic and human-derivable: `user::alice`, `order::2026::12345`, `session::<uuid>`. Use `::` as delimiter, prefix with entity type. A good key scheme lets you `getMulti` related docs without touching an index.

**System metadata** is per-document and not part of the body: `CAS` (compare-and-swap version for optimistic locking), `expiry`/TTL, flags. Read CAS from a `get` result and pass it back on `replace`/`mutateIn` for safe concurrent updates.

> General relational vs. document normalization trade-offs are owned by [`sql.md`](guides://sql.md) and [`mongodb.md`](guides://mongodb.md) — this section covers only the Couchbase key/collection specifics.

---

## 5. Key-Value & Sub-Document Operations

KV is the fast path: by-key `get`/`insert`/`upsert`/`replace`/`remove`, all single-key-atomic.

```python
# 7.x keyspace: cluster → bucket → scope → collection
cluster = Cluster("couchbases://db.example.com",
                  ClusterOptions(PasswordAuthenticator(user, pw)))  # TLS scheme (CB-SEC-01)
coll = cluster.bucket("app").scope("sales").collection("orders")

coll.insert("order::123", doc)          # fails if key exists
coll.upsert("order::123", doc)          # insert-or-replace
res = coll.get("order::123"); cas = res.cas
coll.replace("order::123", doc, ReplaceOptions(cas=cas))  # CAS guard (CB-KV-01)
```

**Sub-document ops (prefer over full fetch/replace).** Read or mutate paths inside a document server-side — less network, atomic on the path, no lost-update race:
```python
coll.lookup_in("user::alice", [SD.get("name"), SD.get("address.city")])
coll.mutate_in("user::alice", [SD.upsert("age", 31),
                               SD.array_append("tags", "premium"),
                               SD.increment("login_count", 1)])  # atomic counter
```

**TTL/expiry** for sessions and caches: `upsert(..., expiry=timedelta(hours=1))`; refresh on read with `get_and_touch`. **Bulk** ops (`get_multi`/`upsert_multi`) amortize round-trips — size batches per [`performance.md`](guides://performance.md).

**Durability levels (CB-DUR-01).** Choose per write by criticality; the default is async (acked when in active node's memory):
| Level | Acked when | Use for |
|---|---|---|
| `none` (default) | in active node RAM | caches, derived data |
| `majority` | replicated to RAM of majority of replicas | most app writes |
| `majority_and_persist_to_active` | majority RAM + active disk | important writes |
| `persist_to_majority` | persisted to disk on majority | money/ledger/state |

```python
coll.upsert("ledger::tx::1", tx,
            UpsertOptions(durability=ServerDurability(DurabilityLevel.PERSIST_TO_MAJORITY)))
```
A durability **timeout is ambiguous** (write may have committed) — make writes idempotent and reconcile per [`error-handling.md`](guides://error-handling.md). Multi-document **ACID transactions** exist (`cluster.transactions`) for the rare cross-document case — keep them small; they are not a substitute for good single-document modeling.

---

## 6. SQL++ (N1QL) — SQL for JSON

SQL++ is ANSI-SQL extended for nested JSON. General SELECT/WHERE/GROUP BY/HAVING/JOIN/window-function/CTE semantics are owned by [`sql.md`](guides://sql.md) — below are only the Couchbase/JSON extensions. `cbq` is the shell; SDKs run queries via `cluster.query(...)`.

**Keyspaces & parameters (CB-QRY-01).** Query `bucket`.`scope`.`collection`; **always** parameterize:
```sql
SELECT o.* FROM `app`.`sales`.`orders` o
WHERE o.status = $status AND o.total > $min;     -- named params, never string-built
```

**Nested data & arrays** — the core differentiator:
```sql
SELECT name, address.city FROM orders o            -- dotted path into nested object
WHERE ANY tag IN o.tags SATISFIES tag = "vip" END; -- array predicate
SELECT o.id, line FROM orders o UNNEST o.items AS line;   -- flatten array → rows
SELECT u.*, o FROM users u NEST orders o ON KEYS u.order_ids;  -- collect children into array
```

**Key-based access in SQL++** (bridges KV and query):
```sql
SELECT * FROM orders USE KEYS ["order::123", "order::124"];   -- direct key fetch, no index
-- Index/lookup joins follow document references:
SELECT u.name, o.total FROM users u JOIN orders o ON KEY o.user_id FOR u;  -- index join
SELECT u.name, o.total FROM users u JOIN orders o ON KEYS u.order_ids;     -- lookup join
```

**DML:** `INSERT INTO ... (KEY, VALUE) VALUES (...)`, `UPSERT`, `UPDATE ... SET path = ...` (can set nested paths), `DELETE`. `RETURNING` echoes affected docs. `meta().id`, `meta().cas`, `meta().expiration` expose document metadata in queries.

---

## 7. Indexing (GSI)

The Index service holds **Global Secondary Indexes**. Every production predicate must be index-backed (CB-IDX-01); verify with `EXPLAIN`.

```sql
CREATE PRIMARY INDEX ON `app`.`sales`.`orders`;          -- dev only; remove in prod
CREATE INDEX idx_status ON orders(status, total);        -- composite (order matters: equality→range)
CREATE INDEX idx_active ON orders(customer_id) WHERE status = "active";  -- partial (smaller/faster)
CREATE INDEX idx_email_ci ON users(LOWER(email));        -- functional/expression index
CREATE INDEX idx_tags ON users(DISTINCT ARRAY t FOR t IN tags END);     -- array index
CREATE INDEX idx_items ON orders(ALL ARRAY i.sku FOR i IN items END);   -- array of objects
```

- **Covering index** — include every projected + filtered field so the query never fetches the document (`EXPLAIN` shows no `Fetch`). This is the single biggest SQL++ speedup.
- **Composite order**: equality predicates first, then range, then sort key; a leading low-selectivity field (e.g. a boolean) wastes the index.
- **Deferred build**: create many indexes with `WITH {"defer_build": true}`, then `BUILD INDEX ON orders(idx_a, idx_b)` once — far faster than one-at-a-time.
- **Adaptive index** (`CREATE INDEX ... ON orders(DISTINCT PAIRS(SELF))`) covers many unpredictable equality predicates with one index — useful for flexible filtering, at higher maintenance cost.
- Tools: `EXPLAIN <q>` (look for `IndexScan3`, no `PrimaryScan`), `ADVISE <q>` for recommendations, `system:indexes` for state/usage. Replicate indexes (`num_replica`) for HA + read scale.

---

## 8. Search, Eventing & Analytics

These are independently-scaled services on the same data; pick the right tool instead of overloading SQL++.

**Search (FTS, Bleve engine).** Linguistic/relevance search, fuzzy/wildcard/phrase, **facets**, and geospatial (radius/bounding-box) — things SQL++ `LIKE` cannot do well. Define an index over collection fields (analyzers per language) and query via the Search SDK or `SEARCH()` inside SQL++. Use for autocomplete, relevance ranking, multi-field text.

**Eventing.** Server-side JavaScript triggered by DCP mutations: `OnUpdate(doc, meta)`, `OnDelete(meta)`, and `OnTimer` (scheduled). Use for denormalization/enrichment, cascade deletes, TTL-driven cleanup, and emitting to other systems — keep handlers small and idempotent (they re-run on rebalance).

**Analytics.** A separate, columnar, shadow copy of operational data for ad-hoc OLAP joins/aggregations **without impacting KV/Query latency** (workload isolation). Define `CREATE DATASET` (or Analytics Collections), query with the same SQL++ dialect. Use it instead of running heavy `GROUP BY`/scans on the operational Query service.

---

## 9. Durability, Replication & XDCR

**Within a cluster:** configure replicas (1–3) per bucket; auto-failover promotes a replica when a node is unresponsive past the timeout (no data loss if replicas + durability are sufficient). Per-write durability levels are in §5.

**XDCR (Cross-Datacenter Replication)** asynchronously streams mutations between clusters for DR and geo-distribution. Topologies: unidirectional (A→B DR), bidirectional active-active (A↔B), and multi-master. Configure per bucket/collection, optionally **filtered** by a predicate (replicate only `region = "US"` or a key prefix).

**Conflict resolution** (active-active): choose the bucket's conflict-resolution mode at creation — **sequence-number/revision** (default, deterministic) or **LWW timestamp** (requires synced clocks). Conflicts are detected via CAS. For semantic merges, model documents to be merge-friendly (CRDT-style counters, per-field timestamps) and resolve in the application, since XDCR itself only picks a winner.

> Retry/idempotency and the timeout-ambiguity contract for replicated writes are owned by [`error-handling.md`](guides://error-handling.md).

---

## 10. Mobile & Edge Sync (offline-first)

Couchbase's edge stack delivers **offline-first** apps: an embedded database on-device that syncs when connectivity returns.

**Couchbase Lite 3.x** — embedded NoSQL DB for iOS/Android/.NET/JS with local KV, SQL++ queries, full-text search, and a `Replicator`. Apps read/write locally with zero latency and no network dependency.

**Sync Gateway 3.x** — the secure sync tier between Lite and Couchbase Server. It enforces **channels** (data-routing/authorization via a `sync` JS function) and access control so each user/device replicates only its authorized documents. Replication runs over WebSocket; types are `push`, `pull`, or `pushAndPull`, one-shot or `continuous`.

**Conflict resolution** on sync defaults to deterministic automatic (revision wins); supply a `ConflictResolver` for local-wins, remote-wins, or custom field-merge logic (e.g. newest `updated_at` per field). Design documents so independent edits merge cleanly. Channel design (who-sees-what) is the security boundary — bind it to [`secure-coding.md`](guides://secure-coding.md) least-privilege.

---

## 11. SDKs, Connection & Security Binding

Official SDKs (Java, Python, Node.js, .NET, Go, C) share the same model: `Cluster → Bucket → Scope → Collection`, KV + sub-doc + Query + Search + Transactions APIs, automatic topology awareness and best-effort retry.

```python
cluster = Cluster("couchbases://node1,node2",           # couchbases:// = TLS (CB-SEC-01)
    ClusterOptions(PasswordAuthenticator(SETTINGS.cb_user, SETTINGS.cb_pass)))  # from config (CB-SEC-02)
cluster.wait_until_ready(timedelta(seconds=10))
coll = cluster.bucket("app").scope("sales").collection("orders")
```

- **Reuse one `Cluster`** per process (it pools connections); never open per request.
- **RBAC least privilege (CB-SEC-01):** create a user with only the roles/buckets·scopes·collections it needs (e.g. `data_reader`/`data_writer`/`query_select` on `app.sales`), never the app running as `Administrator`. Roles can be assigned via groups; integrate LDAP/SAML where required.
- **Encryption:** TLS for all client + XDCR traffic; encryption-at-rest (Enterprise). Audit logging captures admin/data actions.

> Vulnerability scanning, secret management, and the full RBAC/least-privilege rationale are owned by [`secure-coding.md`](guides://secure-coding.md); config layering/env separation by [`env-config.md`](guides://env-config.md). Metrics, tracing, and the slow-query/`system:` keyspace surface bind to [`observability.md`](guides://observability.md).

---

## 12. Operations Quick Reference

```bash
# Cluster / bucket (couchbase-cli)
couchbase-cli cluster-init -c HOST --cluster-username U --cluster-password P \
  --services data,index,query,fts --cluster-ramsize 4096
couchbase-cli bucket-create -c HOST -u U -p P --bucket app \
  --bucket-type couchbase --bucket-ramsize 1024 --bucket-replica 1
couchbase-cli collection-manage -c HOST -u U -p P --bucket app \
  --create-scope sales        # then --create-collection sales.orders

# Query shell
cbq -e couchbases://HOST -u U -p P            # interactive SQL++

# Backup / restore (CB-BAK-01)
cbbackupmgr config  --archive /backups --repo daily
cbbackupmgr backup  --archive /backups --repo daily --cluster couchbases://HOST -u U -p P
cbbackupmgr restore --archive /backups --repo daily --cluster couchbases://HOST -u U -p P
cbbackupmgr info    --archive /backups --repo daily     # verify backups exist

# Observability
curl http://HOST:8091/pools/default/buckets/app/stats   # metrics; Prometheus scrapes /metrics (9102)
```

---

## 13. Deployment Checklist

Generated from §2 — one box per requirement ID.

- [ ] CB-TST-01/02 — features test-first against a real/containerized cluster; bugs have regression tests
- [ ] CB-QRY-01 — all SQL++ parameterized, no string interpolation
- [ ] CB-IDX-01 — `EXPLAIN` shows index scans only, no primary/full scans
- [ ] CB-IDX-02 — index DDL committed as code/migrations
- [ ] CB-KV-01 — concurrent mutations use CAS or sub-document ops
- [ ] CB-DUR-01 — critical writes set an explicit durability level
- [ ] CB-SEC-01 — TLS (`couchbases://`) + least-privilege RBAC user, never `Administrator`
- [ ] CB-SEC-02 — credentials from config/secret store, none in source
- [ ] CB-KEY-01 — deterministic, type-prefixed key scheme
- [ ] CB-OBS-01 — slow-query log + cluster metrics scraped
- [ ] CB-BAK-01 — scheduled backups + tested restore
- [ ] Agent ran every verification command and documented any fixes

---
**End of Couchbase Guidelines**
