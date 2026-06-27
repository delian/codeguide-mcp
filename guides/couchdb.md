# CouchDB Development Guidelines
Mandatory standards for CouchDB: document modeling, MVCC/revisions, replication, conflict resolution, views, and Mango. CouchDB 3.4, Fauxton, Mango, MapReduce, PouchDB.

---
name: couchdb
title: CouchDB Development Guidelines
version: 2.0
last_reviewed: 2026-06-05
kind: datastore
tools: [couchdb@3.4, fauxton, mango, pouchdb@9, nano]
requires:
  - secure-coding
  - error-handling
recommends:
  - observability
  - performance
  - rest
  - env-config
provides:
  - couchdb-mvcc
  - document-revisions
  - couch-replication
  - conflict-resolution
  - map-reduce-views
---

> 🧭 Authored per [`CONVENTIONS.md`](guides://CONVENTIONS.md): shared concerns are referenced, not restated. This guide covers only what is unique to CouchDB.

---

## 0. Prerequisites & References

Fetch and apply these **before** generating CouchDB code. Their rules are assumed below and not repeated.

> 📎 **REQUIRED — fetch & apply first:**
> - [`secure-coding.md`](guides://secure-coding.md) — auth, least privilege, secrets, TLS. *(Couch binding: kill admin-party, `_security` object, `validate_doc_update`, HTTPS — §9.)*
> - [`error-handling.md`](guides://error-handling.md) — error strategy & propagation. *(Couch binding: **HTTP 409 conflict** read-rev-retry, 412/417 on bulk — §3, §6.)*

> 📎 **RECOMMENDED — fetch when the task touches them:**
> - [`rest.md`](guides://rest.md) — CouchDB **is** an HTTP/REST API; method/status semantics, ETags, conditional requests are owned there. This guide binds Couch's resource layout on top (§4).
> - [`performance.md`](guides://performance.md) — perf methodology *(binding: view/index build cost, `_explain`, compaction — §5, §8)*
> - [`observability.md`](guides://observability.md) — metrics/monitoring *(binding: `/_node/.../_stats`, Prometheus endpoint, `_active_tasks` — §8)*
> - [`env-config.md`](guides://env-config.md) — connection/config policy *(binding: URL credentials out of code; `.ini`/env config — §9)*

> 📎 **SEE ALSO:** [`tdd.md`](guides://tdd.md) *(test replication & conflicts)* · [`websocket.md`](guides://websocket.md) *(proxying `_changes`)* · [`couchbase.md`](guides://couchbase.md) · [`mongodb.md`](guides://mongodb.md) *(comparison only)*

---

## 1. Core Philosophies: SYNC-FIRST

CouchDB-specific principles only. Security and error strategy come from §0.

- **S**ync as the default: design `_id`s, document granularity, and filters for multi-master replication and offline clients — never assume a single writer.
- **Y**ield to eventual consistency: reads can be stale, replication is asynchronous, cluster reads are quorum-based — never assume read-your-write across nodes.
- **N**arrow documents: one document = one independently-syncable unit; semantic `_id`s; **no unbounded arrays** (they conflict and bloat). Split growth into separate docs.
- **C**onflicts are normal: multi-master sync **will** create conflicts. CouchDB picks a deterministic winner but keeps losers — the application MUST detect (`?conflicts=true`) and resolve them. This is the central design obligation.
- **F**ilters & changes: drive replication, feeds, and indexes off `_changes`; use selectors/filter functions to replicate subsets.
- **I**ndex before you query: every non-trivial Mango query is index-backed (verified by `_explain`); views are the B-tree index for aggregation.
- **R**EST correctly: respect HTTP verbs, status codes, ETags, and `_rev` optimistic concurrency (see `rest.md`).
- **T**est sync paths: replication, conflict resolution, and offline scenarios are part of the test suite, not an afterthought.

**Verified Code**: Agent-generated CouchDB code MUST pass every gate in §2 before delivery.

---

## 2. Requirements (MANDATORY, auditable)

RFC-2119 keywords. IDs `COUCH-<TOPIC>-<NN>`. Each row has a binary gate; rows binding a shared rule cite its owner.

| ID | Requirement | Verify | Gate |
|----|-------------|--------|------|
| COUCH-TST-01 | Features (incl. replication/conflict paths) MUST be test-first (see `tdd.md`) | run test suite | exit 0, 0 skips |
| COUCH-TST-02 | Each bug MUST get a regression test before the fix (see `tdd.md`) | run test suite | failing→passing |
| COUCH-REV-01 | Every write MUST send the current `_rev`; a **409** MUST be re-read-and-retried, never blindly forced (see `error-handling.md`) | code review | retry-on-409 present |
| COUCH-CFL-01 | Code MUST detect conflicts (`?conflicts=true` / `_conflicts` view) and resolve them deterministically | conflict integration test | conflicts reach 0 |
| COUCH-IDX-01 | Mango queries MUST be index-backed; no full-scan in production | `POST /_explain` | `index ≠ _all_docs` |
| COUCH-DOC-01 | Documents MUST carry a `type` field, semantic `_id`, and no unbounded arrays | schema review / lint | no growth fields |
| COUCH-SEC-01 | Admin-party MUST be disabled; every DB MUST have a `_security` object (see `secure-coding.md`) | `GET /_users`, `GET /db/_security` | admins set, no party |
| COUCH-SEC-02 | Credentials/URLs MUST come from config, never hardcoded; prod MUST use TLS (see `secure-coding.md`, `env-config.md`) | grep / `GET /_node/.../_config/ssl` | no literals, TLS on |
| COUCH-SEC-03 | Untrusted writes MUST be gated by a `validate_doc_update` function (see `secure-coding.md`) | review design docs | VDU present |
| COUCH-VIEW-01 | Map functions MUST be deterministic & side-effect-free; design docs versioned in VCS | review | pure, in repo |
| COUCH-REP-01 | Persistent replication MUST use the `_replicator` DB (not one-shot `/_replicate`) with monitored state | `GET /_scheduler/docs` | state=running |
| COUCH-OBS-01 | Health/metrics scraped; compaction & view lag monitored (see `observability.md`) | `GET /_node/.../_stats`, `/_active_tasks` | metrics flowing |

> **Forbidden**: forcing a write past a 409 without merging, unbounded embedded arrays, running production in admin-party, ad-hoc Mango full-scans, non-deterministic map/reduce, one-shot `/_replicate` for durable sync, or credentials embedded in URLs in source.

---

## 3. Verification Protocol

Run before presenting code. Fix → re-run until green.

```bash
# COUCH-IDX-01 — confirm the query uses a real index, not _all_docs
curl -sX POST $DB/_explain -H 'Content-Type: application/json' \
  -d '{"selector":{"type":"user","age":{"$gt":25}}}' | jq '.index.name'

# COUCH-CFL-01 — surface unresolved conflicts (define a _conflicts view, or:)
curl -s "$DB/doc_id?conflicts=true" | jq '._conflicts'

# COUCH-SEC-01 — admin-party off + per-DB security
curl -s $HOST/_node/_local/_config/admins | jq 'keys'        # non-empty
curl -s $DB/_security | jq '.admins, .members'               # populated

# COUCH-REP-01 — replication state
curl -s $HOST/_scheduler/docs | jq '.docs[] | {id,state}'    # state=="running"

# COUCH-OBS-01 — running tasks / lag
curl -s $HOST/_active_tasks | jq '.[] | {type,progress}'
```

The *why* behind each gate lives in its §0 owner; do not re-derive it here.

---

## 4. Document Model & the HTTP API (`provides: document-revisions`)

CouchDB stores schema-less **JSON documents** in **databases**, exposed entirely over HTTP/REST. The REST semantics (verbs, status codes, ETags, conditional requests) are owned by [`rest.md`](guides://rest.md); below is only Couch's resource layout and document rules.

### A. Document anatomy
```json
{ "_id": "order:2026:12345",      // semantic, sortable, app-controlled
  "_rev": "3-ghi789",             // MVCC revision (see §5)
  "type": "order",                // ALWAYS present — drives views/Mango/filters
  "user_id": "user:alice",        // reference (no joins; resolve client-side or via view)
  "total": 59.98,
  "updated_at": "2026-06-27T10:00:00Z" }
```
- **`_id` strategy**: prefer semantic, lexicographically meaningful IDs (`type:key`) — they enable range scans over `_all_docs` (`startkey`/`endkey`) and readable references. Use server UUIDs only when no natural key exists. Avoid sequential IDs that collide across offline clients.
- **Modeling**: embed data accessed together and bounded in size; **reference** large or independently-updated entities; denormalize read-hot fields knowing writes must fan-out (eventual consistency). **Never** embed an unbounded, growing array — it maximizes conflict surface and document rewrite cost (whole-doc rewrite on every update). Keep docs well under ~1 MB.

### B. CRUD over HTTP
```
PUT  /db/{id}            create/update with explicit _id (body must carry current _rev to update)
POST /db                 create with server-generated _id
GET  /db/{id}            read   (?rev= , ?conflicts=true , ?revs_info=true)
DELETE /db/{id}?rev=     soft-delete → tombstone {_deleted:true}; replicates (purge separately to reclaim)
POST /db/_bulk_docs      batch insert/update/delete; per-doc {ok|error} results, 409s isolated to their doc
POST /db/_all_docs       primary _id index; keys[], include_docs, startkey/endkey, limit/skip
```
Updates are **optimistic-concurrency**: omit/mismatch `_rev` → **409 Conflict** (see §5, `error-handling.md`). `DELETE` leaves a tombstone so the deletion replicates; physical space returns only on compaction/purge.

### C. Attachments
Store binary blobs as attachments, not base64 inside JSON. Reference by stub:
```bash
curl -X PUT "$DB/user:alice/photo.jpg?rev=2-def" -H 'Content-Type: image/jpeg' --data-binary @photo.jpg
```
Attachments replicate with the doc and bust the doc's cache on every change — keep large/volatile blobs in object storage and reference by URL instead.

---

## 5. MVCC, Revisions & Conflict Resolution (`provides: couchdb-mvcc, conflict-resolution`)

**The defining concept.** Every write produces a new `_rev` (`N-hash`); CouchDB stores a revision **tree** per document, and the database file is append-only (crash-safe, lock-free MVCC — reclaimed by compaction).

### A. Optimistic concurrency (single node)
`_rev` is a CAS token. To update: read current doc, mutate, write back with its `_rev`. A stale `_rev` → **HTTP 409**. The correct handler is **re-read → re-apply → retry** (bounded), never force:
```
loop (max N): GET doc → merge changes onto latest → PUT with latest _rev → on 409 retry, on 2xx done
```
This is the canonical Couch binding of [`error-handling.md`](guides://error-handling.md)'s retryable-conflict pattern.

### B. Conflicts from replication (multi-master)
Two nodes editing the same `_rev` while disconnected produce **two sibling revisions** after sync. CouchDB does **not** lose data and does **not** merge:
1. It picks a **deterministic winning revision** (highest generation; ties broken by lexicographically highest rev hash) — identical on every replica, so reads are consistent everywhere.
2. The losing revision is **kept** as a conflict branch until you resolve it.

A 409 here is silent — you only discover conflicts by asking:
```bash
GET /db/doc?conflicts=true          # → "_conflicts": ["2-ghi789", ...] on the winner
GET /db/doc?rev=2-ghi789            # fetch a losing branch to merge
```
Build a **conflicts view** to find them in bulk:
```javascript
function (doc) { if (doc._conflicts) emit(doc._id, doc._conflicts); }
```

### C. Resolving (you MUST — COUCH-CFL-01)
Read winner + every conflict branch, apply business merge logic, then in one `_bulk_docs`: PUT the merged doc on the winner's `_rev` **and** DELETE each losing `_rev` (so the deletion replicates). Strategies: last-write-wins by `updated_at`, field-level merge, or domain rules. Unresolved conflicts accumulate forever and skew views — treat conflict count as a monitored metric.

### D. Conflict-avoidant design (cheaper than resolving)
- **Append, don't mutate**: model events/log entries as **separate documents**, not edits to one doc.
- **Immutable/map shapes**: keyed maps (`{ "<uuid>": value }`) merge cleanly because branches touch different keys; counters as summed event docs (CRDT-style) instead of one mutable integer.
- Smaller, single-purpose documents → smaller conflict surface.

---

## 6. Querying: Mango & MapReduce Views (`provides: map-reduce-views`)

Two complementary query systems. **Mango** is the declarative default; **views** are the powerful, incremental B-tree index for aggregation and complex keys. Both MUST be index-backed (COUCH-IDX-01). Performance methodology is owned by [`performance.md`](guides://performance.md).

### A. Mango (`_find`) — start here
JSON selector query, MongoDB-like, no JavaScript required:
```json
{ "selector": { "type": "user", "age": {"$gt": 25}, "status": {"$in": ["active","pending"]} },
  "fields": ["_id","name","email"], "sort": [{"created_at":"desc"}], "limit": 50, "use_index": "type-age-idx" }
```
Operators: `$gt/$gte/$lt/$lte/$ne`, `$and/$or/$not/$nor`, `$exists`, `$type`, `$in/$nin/$all/$size`, `$elemMatch`, `$regex` (unindexed — avoid in hot paths). Nested fields use dotted keys (`"address.city"`).

Create a JSON index for the query's fields **before** running it, then verify:
```bash
curl -XPOST $DB/_index -d '{"index":{"fields":["type","age"]},"name":"type-age-idx","type":"json"}'
curl -XPOST $DB/_explain -d '{"selector":{"type":"user","age":{"$gt":25}}}' | jq .index.name  # must NOT be _all_docs
```
Index field **order** matters (most selective / equality-then-range first); avoid huge composite indexes that few queries use. A selector with no matching index falls back to a full collection scan — that is a COUCH-IDX-01 failure.

### B. MapReduce views — for aggregation & rich keys
A **design document** (`_design/<name>`) holds `views`. Each view is a **deterministic, pure** JavaScript `map` (+ optional `reduce`) that incrementally builds a persisted B-tree, updated lazily on first query after a change.
```javascript
// map: emit(key, value) — key drives ordering & range queries
function (doc) { if (doc.type === 'order') emit([doc.user_id, doc.created_at], doc.total); }
```
- **Compound/array keys** enable hierarchical range queries: `?startkey=["user:alice"]&endkey=["user:alice",{}]`, or a date range as the second key element.
- **Built-in reduces** are preferred (C-fast, rereduce-safe): `_count`, `_sum`, `_stats`, `_approx_count_distinct`. Write a custom JS reduce only when necessary and it **MUST** be rereduce-safe and bounded in output size (no growing arrays/objects — the classic reduce footgun). Use `group=true`/`group_level=N` to aggregate by key prefix.
- Querying: `GET /db/_design/d/_view/v?key=…|startkey=…&endkey=…|descending=true|include_docs=true|reduce=false`.
- **Rules**: map output depends only on the document (no `Date.now()`, no external state); version design docs in VCS; deploy design-doc changes carefully — saving a new view rebuilds the whole index (use `_design` staging / `mango`-style `use_index` for blue-green). For full-text, use the **Nouveau**/Lucene-based search service (CouchDB 3.4) or `type:"text"` Mango indexes rather than hand-rolled scans.

### C. Mango vs. views
Use **Mango** for ad-hoc filters, secondary lookups, and developer ergonomics. Use **views** for aggregation/grouping, reduce, compound-key range scans, and when you need the index materialized exactly. Mango JSON indexes are themselves implemented as views under the hood.

---

## 7. Replication & Offline-First (`provides: couch-replication`)

CouchDB's standout feature: **incremental, multi-master, bidirectional** replication over HTTP — the same protocol whether between servers, across data centers, or to an in-browser **PouchDB** client.

### A. Durable replication via `_replicator`
Persist jobs as documents in the `_replicator` DB (survives restarts; managed by the scheduler) — **never** rely on one-shot `POST /_replicate` for ongoing sync (COUCH-REP-01):
```bash
curl -X PUT $HOST/_replicator/mydb-sync -H 'Content-Type: application/json' -d '{
  "source":"https://a.example/mydb","target":"https://b.example/mydb",
  "continuous":true, "create_target":true }'
curl -s $HOST/_scheduler/docs | jq '.docs[] | {id,state}'   # monitor: running/crashing/failed
```
Bidirectional sync = two continuous jobs (A→B and B→A); multi-master conflicts are then handled per §5.

### B. Selective / filtered replication
Replicate a subset to push less data to edge/offline clients:
- **Mango selector** (preferred, JS-free): `"selector": {"type":"user","status":"active"}` in the replication doc.
- **Filter function** in a design doc: `"filter":"filters/users_only"` (`function(doc,req){ return doc.type==='user'; }`).
- Or `_doc_ids` / `_design`-only filters. Filtered replication is slower (runs per change) — prefer selectors and per-user databases where possible.

### C. Offline-first with PouchDB
PouchDB speaks the replication protocol in the browser/Node, giving local-first reads/writes and background sync:
```javascript
const local = new PouchDB('mydb');
const remote = new PouchDB('https://host/mydb');           // creds via auth, not in URL string in source
const sync = local.sync(remote, { live: true, retry: true })
  .on('change', i => render(i)).on('error', e => report(e));
// resolve conflicts client-side exactly as §5: get({conflicts:true}) → merge → put + remove losers
```
Patterns: **one database per user** to isolate access and shrink sync sets; queue offline writes locally and let `retry:true` flush on reconnect; treat `navigator.onLine` as a hint, not a guarantee. Conflict resolution is the **client's** job too.

### D. Change feed (`_changes`) — the event backbone
Replication, indexes, and live UIs all ride the ordered `_changes` feed:
```bash
GET /db/_changes?since=now&feed=longpoll&include_docs=true     # one batch when something changes
GET /db/_changes?feed=continuous&heartbeat=10000&filter=users/active   # streamed, filtered
```
`feed=normal|longpoll|continuous|eventsource`; persist `last_seq` to resume. For browser push, proxy `_changes` over a WebSocket (see [`websocket.md`](guides://websocket.md)) or use PouchDB's `db.changes({live:true})` — don't expose raw admin credentials to clients.

---

## 8. Clustering, Performance & Operations

### A. Cluster, sharding & quorum
A production cluster is **3+ nodes**. Databases are sharded (`q`, default 8) and replicated (`n`, default 3); reads/writes use quorum (`r`/`w`, default `n/2+1`). Set `q`/`n` **at creation** (`PUT /db?q=16&n=3`) — `q` cannot change later without re-creating the DB; size `q` to expected data/node count (rule of thumb ≤ ~10 GB/shard). Set up clusters with the `_cluster_setup` endpoint (or Fauxton); inspect with `GET /_membership` and `GET /db/_shards`. Cross-node reads are eventually consistent.

### B. Performance (see `performance.md`)
- **Compaction**: the append-only file grows on every write/update; schedule database **and** view compaction (auto-compaction config or `POST /db/_compact`, `/db/_compact/ddoc`) to reclaim space. Monitor `disk_size` vs `data_size`.
- **View build cost**: a new/changed view rebuilds its entire index on first query — warm it (`?limit=0`) off the hot path; never edit a hot design doc in place without staging.
- **Bulk over chatty**: use `_bulk_docs` / `_bulk_get` instead of per-doc round trips; reuse HTTP connections (keep-alive / client pool).
- **Read tuning**: `?include_docs=true` adds a fetch per row — emit needed fields into the view value instead when possible. Avoid deep `skip` pagination; page by `startkey`/`startkey_docid`.

### C. Monitoring (see `observability.md`)
Scrape `GET /_node/{node}/_stats` (and the Prometheus endpoint on port 17986 when enabled), `GET /_active_tasks` (indexer/replication/compaction progress and lag), `GET /_scheduler/jobs`, and per-DB `GET /db` (`doc_count`, `disk_size`). Alert on replication `crashing`/`failed`, growing conflict counts, and view-index lag.

---

## 9. Security (`provides:` n/a — owned by `secure-coding.md`)

Policy is owned by [`secure-coding.md`](guides://secure-coding.md); credentials/config layering by [`env-config.md`](guides://env-config.md). CouchDB bindings:

- **Kill admin-party first** (COUCH-SEC-01): a fresh node lets anyone do anything. Create an admin (`PUT /_node/_local/_config/admins/<name>`), then no anonymous access remains.
- **Per-database `_security`**: set `admins`/`members` (`names` + `roles`) on every DB; without it, any authenticated user can read it.
```bash
curl -X PUT $DB/_security -d '{"admins":{"roles":["_admin"]},"members":{"roles":["user"]}}'
```
- **`validate_doc_update` (VDU)** (COUCH-SEC-03): per-DB write gate in a design doc — enforce ownership, schema, and immutability server-side; `throw({forbidden|unauthorized: "..."})` to reject. This is the only place to enforce per-document rules that survive replication.
- **Users & roles** live in the `_users` DB (`org.couchdb.user:<name>`, hashed via pbkdf2). Auth methods: Basic (TLS only), cookie/session (`POST /_session`), or **JWT** (`[jwt_keys]`) for stateless/SSO — JWT is the modern choice for app backends.
- **No credentials in source/URLs** (COUCH-SEC-02): inject via env/secret store (see `env-config.md`); enable TLS (`[ssl]`) or terminate at a reverse proxy; lock down CORS `origins` to known web apps rather than `*`.

---

## 10. Quick Reference

```bash
curl -XPUT  $DB/{id} -d '{...,"_rev":"N-hash"}'         # update (CAS on _rev; 409 ⇒ re-read+retry)
curl -XPOST $DB/_bulk_docs -d '{"docs":[...]}'          # batch write
curl -s "$DB/{id}?conflicts=true" | jq '._conflicts'    # find conflict branches
curl -XPOST $DB/_find -d '{"selector":{...}}'           # Mango query
curl -XPOST $DB/_explain -d '{"selector":{...}}'        # confirm index (≠ _all_docs)
curl -XPOST $DB/_index -d '{"index":{"fields":[...]}}'  # create Mango index
curl "$DB/_design/d/_view/v?group=true"                 # query/aggregate a view
curl -XPUT  $HOST/_replicator/job -d '{...,"continuous":true}'   # durable replication
curl -s "$DB/_changes?feed=continuous&since=now"        # change feed
curl -XPOST $DB/_compact                                # reclaim space
```

---

## 11. Deployment Checklist

Generated from §2 — one box per requirement ID.

- [ ] COUCH-TST-01/02 — features (incl. replication/conflict) test-first; bugs have regression tests
- [ ] COUCH-REV-01 — writes send current `_rev`; 409 handled by re-read+retry
- [ ] COUCH-CFL-01 — conflicts detected and resolved; conflict count → 0
- [ ] COUCH-IDX-01 — Mango queries index-backed (`_explain` ≠ `_all_docs`)
- [ ] COUCH-DOC-01 — `type` + semantic `_id`, no unbounded arrays
- [ ] COUCH-SEC-01 — admin-party off; every DB has `_security`
- [ ] COUCH-SEC-02 — no hardcoded creds/URLs; TLS enforced
- [ ] COUCH-SEC-03 — `validate_doc_update` gates untrusted writes
- [ ] COUCH-VIEW-01 — map/reduce pure & deterministic; design docs in VCS
- [ ] COUCH-REP-01 — durable replication via `_replicator`, state monitored
- [ ] COUCH-OBS-01 — stats/tasks scraped; compaction & view lag watched
- [ ] Agent ran every §3 command and documented any fixes

---
**End of CouchDB Guidelines**
