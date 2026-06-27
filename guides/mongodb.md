# MongoDB Development Guidelines
Mandatory standards for MongoDB document modeling, aggregation, indexing, and operations. MongoDB 8.0, mongosh, Compass, drivers.

---
name: mongodb
title: MongoDB Development Guidelines
version: 2.0
last_reviewed: 2026-06-05
kind: datastore
tools: [mongodb@8.0, mongosh, compass]
requires:
  - secure-coding
  - error-handling
recommends:
  - observability
  - performance
  - docker-compose
  - env-config
  - sql
provides:
  - document-modeling
  - aggregation-pipeline
  - mongo-indexing-esr
  - sharding
  - change-streams
---

> 🧭 Authored per [`CONVENTIONS.md`](guides://CONVENTIONS.md): shared concerns are referenced, not restated. This guide covers only what is unique to MongoDB.

---

## 0. Prerequisites & References

Fetch and apply these **before** designing schemas or writing queries. Their rules are assumed below and not repeated.

> 📎 **REQUIRED — fetch & apply first:**
> - [`secure-coding.md`](guides://secure-coding.md) — auth, RBAC, secrets, injection, CVE policy. *(Mongo binding: SCRAM/X.509, least-privilege roles, operator-injection defense, Queryable Encryption — see §10.)*
> - [`error-handling.md`](guides://error-handling.md) — error strategy & propagation. *(Mongo binding: write concern, retryable writes, transient-transaction retry — see §7–8.)*

> 📎 **RECOMMENDED — fetch when the task touches them:**
> - [`observability.md`](guides://observability.md) — metrics/tracing policy *(binding: database profiler, `serverStatus`, Atlas monitoring — §11)*
> - [`performance.md`](guides://performance.md) — perf policy *(binding: index/working-set sizing, covered queries — §4, §11)*
> - [`env-config.md`](guides://env-config.md) — config policy *(binding: connection string in env, never hardcoded — §11)*
> - [`docker-compose.md`](guides://docker-compose.md) — local replica-set/sharded stacks for tests
> - [`sql.md`](guides://sql.md) — relational comparison; read when deciding document vs relational (§3)

> 📎 **SEE ALSO:** [`mongoose.md`](guides://mongoose.md) *(the Node.js ODM, builds on this guide)* · [`tdd.md`](guides://tdd.md) (test schemas/pipelines against `mongodb-memory-server` or a Compose replica set) · [`timescaledb.md`](guides://timescaledb.md) · [`postgresql.md`](guides://postgresql.md)

---

## 1. Core Philosophies: MONGO-FIRST

MongoDB-specific principles only. Security, error handling, observability come from §0.

- **M**odel for access patterns: design the schema around the queries the app runs, not around normalized entities. The shape of reads dictates the document shape.
- **O**ne decision dominates: **embed vs reference** (§3). Get cardinality and access-locality right first; everything else follows.
- **N**o unbounded growth: never let an embedded array grow without bound — respect the 16 MB document limit; bucket or reference instead.
- **G**ate every query on an index: the ESR rule (§4) governs compound indexes; `explain()` must show `IXSCAN`, never `COLLSCAN`, on hot paths.
- **O**perate for durability: `w:"majority"` writes, replica-set reads tuned per workload, and a validated shard key before you scale out.

**Verified Code**: Agent-generated schemas, indexes, and queries MUST pass every gate in §2 before delivery.

---

## 2. Requirements (MANDATORY, auditable)

RFC-2119 keywords. IDs `MONGO-<TOPIC>-<NN>`. Each row has a binary gate; rows binding a shared rule cite its owner.

| ID | Requirement | Verify | Gate |
|----|-------------|--------|------|
| MONGO-MODEL-01 | Embed-vs-reference MUST be justified by cardinality & access locality; no unbounded embedded arrays | design review vs §3 | documented, bounded |
| MONGO-MODEL-02 | Collections holding business data MUST have `$jsonSchema` validation (`validationAction:"error"`) | `db.getCollectionInfos({name})[0].options.validator` | validator present |
| MONGO-MODEL-03 | No document MUST approach the 16 MB BSON limit | `db.col.aggregate([{$project:{s:{$bsonSize:"$$ROOT"}}},{$sort:{s:-1}},{$limit:1}])` | max ≪ 16 MB |
| MONGO-IDX-01 | Every hot query/sort MUST be index-backed (no COLLSCAN) | `db.col.find(q).explain("executionStats")` | stage `IXSCAN`, `totalDocsExamined`≈`nReturned` |
| MONGO-IDX-02 | Compound indexes MUST follow ESR (Equality, Sort, Range) | review index vs query shape | ESR order |
| MONGO-IDX-03 | Ephemeral data MUST use TTL indexes; partial indexes used where the query is selective | `db.col.getIndexes()` | TTL/partial present |
| MONGO-WC-01 | Writes MUST use `w:"majority"` with `wtimeoutMS` (see `error-handling.md`) | default RW concern / driver opts | majority + timeout |
| MONGO-WC-02 | Retryable writes & transient-transaction retry MUST be enabled (see `error-handling.md`) | connection string `retryWrites=true`; txn retry loop | enabled |
| MONGO-RS-01 | Production MUST run P-S-S (no P-S-A with majority writes) | `rs.conf().members` | ≥3 data-bearing |
| MONGO-SHARD-01 | Shard keys MUST be high-cardinality & non-monotonic, validated before sharding | `db.adminCommand({analyzeShardKey,...})` | good cardinality/frequency |
| MONGO-SEC-01 | Auth enabled, TLS required, least-privilege roles, no `root` app user (see `secure-coding.md`) | `db.getUsers()` / launch flags | scoped roles, TLS on |
| MONGO-SEC-02 | No operator injection; PII fields encrypted (Queryable Encryption) (see `secure-coding.md`) | code review; `encryptedFieldsMap` | no `$where`/raw `$`, encrypted PII |
| MONGO-CFG-01 | Connection string MUST come from env/secret store, never hardcoded (see `env-config.md`) | grep source | no literal URIs |
| MONGO-OBS-01 | Slow-query profiler / monitoring MUST be enabled (see `observability.md`) | `db.getProfilingStatus()` | level 1 with `slowms` |

> **Forbidden**: unbounded embedded arrays; `COLLSCAN` on a hot path; `w:0` / `w:1` for durable writes in production; P-S-A with majority writes; monotonic shard keys (`_id:1`, raw timestamp); building queries by string-concatenating user input into `$where`/`$expr`; hardcoded connection strings or credentials.

---

## 3. Document Modeling (the central decision)

MongoDB is schemaless on the server but your application is not — model deliberately. The governing trade-off is **embed vs reference**. (Coming from relational? See [`sql.md`](guides://sql.md): normalize for write-integrity, embed for read-locality.)

### A. Embed vs reference

**Embed** when the related data is owned by the parent, bounded, and read together:
- one-to-one and one-to-few (≲ dozens, never unbounded)
- always fetched with the parent; no independent lifecycle
- gives single-document atomic reads/writes — no `$lookup`

**Reference** when:
- one-to-many (large/growing) or many-to-many
- the child has independent access, lifecycle, or high update churn
- embedding would breach the **16 MB** document limit or bloat the working set

```javascript
// Embedded: addresses live and die with the user, small & bounded
{ _id, email, addresses: [{ type:"home", city:"NYC", isDefault:true }] }

// Referenced: orders are many, grow forever, queried on their own
// users:  { _id, email }
// orders: { _id, userId /* ref */, total, status, createdAt }
```

Rules of thumb: **arrays that grow without bound → reference** (MONGO-MODEL-01). **Snapshot, don't reference, immutable point-in-time data** (e.g. the shipping address / price *as charged*) so history can't be mutated by later edits.

### B. Schema design patterns

- **Extended Reference** — copy the few hot fields of a referenced doc onto the parent (e.g. author `name`+`avatar` on a post) to avoid a `$lookup`; keep the `_id` for the source of truth. Trade read speed for fan-out updates on change.
- **Subset** — embed only the *recent/top* N children (latest comments) plus a `count`; page the rest from their own collection.
- **Computed** — store pre-aggregated values (`orderCount`, rolling totals) updated on write, instead of recomputing on every read.
- **Bucketing** — group many small time/event records into one document per (entity, time-window) to cut document count and index size. (For pure time-series, prefer a time-series collection — §9.) 
- **Outlier** — handle the rare huge document (the user with 10M followers) with an overflow flag + spill collection rather than sizing every doc for the worst case.
- **Schema Versioning** — stamp a `schemaVersion` field; migrate lazily on read or eagerly via an `updateMany` aggregation pipeline. Run dual indexes during transitions, drop the old one when complete.

### C. Schema validation (`$jsonSchema`)

Enforce structure at the server (MONGO-MODEL-02). Validate types, required fields, enums, ranges:

```javascript
db.createCollection("orders", {
  validator: { $jsonSchema: {
    bsonType: "object",
    required: ["userId", "total", "status", "createdAt"],
    properties: {
      userId:    { bsonType: "objectId" },
      total:     { bsonType: "decimal", minimum: 0 },   // money → decimal128, not double
      status:    { enum: ["pending","confirmed","shipped","delivered","cancelled"] },
      createdAt: { bsonType: "date" }
    },
    additionalProperties: true
  }},
  validationLevel: "strict",
  validationAction: "error"   // reject invalid writes; "warn" only during rollout
});
```

Use `decimal128` for money, `date` for time, real `objectId` refs — not stringly-typed everything.

---

## 4. Indexing (ESR)

Indexes are the single biggest performance lever. Validate with `explain("executionStats")`: want `IXSCAN`, `totalDocsExamined` ≈ `nReturned`, no in-memory `SORT`.

### A. The ESR rule (compound index field order)

Order compound-index keys **Equality → Sort → Range**:

```javascript
// Query: { status:"active", createdAt:{$gte:d} } sorted by { priority:-1 }
db.tasks.createIndex({ status: 1, priority: -1, createdAt: 1 });
//                     Equality   Sort         Range
```

Equality fields first (they narrow the scan to a contiguous range), then the sort field (so the index supplies sort order — no blocking sort), then range fields last. A range before the sort field forces an in-memory sort.

### B. Index types

```javascript
db.users.createIndex({ email: 1 }, { unique: true });                 // unique
db.orders.createIndex({ userId: 1, status: 1, createdAt: -1 });        // compound (ESR)
db.users.createIndex({ "addresses.city": 1 });                        // multikey (array)
db.sessions.createIndex({ expiresAt: 1 }, { expireAfterSeconds: 0 }); // TTL auto-expiry
db.orders.createIndex({ createdAt: -1 },                              // partial: index a subset
  { partialFilterExpression: { status: "pending" } });
db.posts.createIndex({ title: "text", body: "text" });               // text search
db.places.createIndex({ loc: "2dsphere" });                          // geospatial
db.products.createIndex({ "attrs.$**": 1 });                         // wildcard (dynamic keys)
```

### C. Covered queries & analysis

A **covered query** is served entirely from the index (`totalDocsExamined: 0`) — every projected field is in the index and `_id` is excluded:

```javascript
db.users.createIndex({ email: 1, status: 1 });
db.users.find({ email:"a@b.com" }, { status:1, _id:0 });  // covered
```

```javascript
db.col.getIndexes();                              // list
db.col.aggregate([{ $indexStats: {} }]);          // usage — drop indexes with ops:0
db.col.find(q).explain("executionStats");         // IXSCAN vs COLLSCAN, docs examined
```
Build indexes that cover queries; drop unused ones (they cost on every write). Keep the index working set in RAM (see [`performance.md`](guides://performance.md)).

---

## 5. Aggregation Pipeline

The aggregation framework is MongoDB's analytical engine. Compose stages; the optimizer reorders some, but **you must front-load selectivity**.

```javascript
db.orders.aggregate([
  { $match: { status:"delivered", createdAt:{ $gte: ISODate("2026-01-01") } } }, // 1. filter FIRST (uses index)
  { $group: { _id: { $dateTrunc: { date:"$createdAt", unit:"month" } },
              revenue: { $sum:"$total" }, n: { $sum:1 }, avg: { $avg:"$total" } } },
  { $sort:  { revenue: -1 } },
  { $project:{ _id:0, month:"$_id", revenue:{ $round:["$revenue",2] }, n:1 } }
]);
```

Key stages: `$match` (filter — put first so it uses an index), `$group` (aggregate), `$lookup` (left-join another collection), `$unwind` (flatten arrays), `$project`/`$set`/`$unset` (reshape), `$facet` (multiple sub-pipelines in one pass — results + counts + buckets), `$bucket`/`$bucketAuto`, window functions (`$setWindowFields`), `$merge`/`$out` (materialize).

**Pipeline optimization:**
- `$match` and `$project` as early as possible — shrink the document set and field width before expensive stages.
- `$match` before `$lookup`/`$group`/`$unwind`; only an index-eligible `$match` at the *front* uses an index.
- A `$sort` immediately before `$group` can let the group short-circuit; a `$sort`+`$limit` together is optimized to a top-k.
- `$lookup` is a nested-loop join — index the `foreignField`, and filter both sides first. Prefer the extended-reference pattern (§3.B) over `$lookup` on hot paths.
- Run `explain()` on pipelines too; watch for `$lookup`/`$group` spilling to disk (`allowDiskUse` is a smell that the pipeline is doing too much).

---

## 6. Query & Write API

```javascript
// Read — always project to the fields you need
db.users.find({ status:"active" }, { email:1, profile:1, _id:0 });
db.orders.find({ total:{ $gte:100 }, status:{ $in:["pending","processing"] },
                 cancelledAt:{ $exists:false } });
db.posts.find({ tags: "mongodb" });                 // array contains
db.posts.find({ tags: { $all:["mongodb","atlas"] } });

// Write — operators, upsert, array updates
db.users.updateOne({ _id }, { $set:{ "profile.name":"Jane" }, $currentDate:{ updatedAt:true } });
db.settings.updateOne({ userId }, { $set:{ theme:"dark" } }, { upsert:true });
db.users.updateOne({ _id, "addresses.type":"home" }, { $set:{ "addresses.$.city":"LA" } });

// Bulk — unordered for throughput (continues past per-doc errors)
db.col.bulkWrite(ops, { ordered: false });
```

**Cursor (keyset) pagination — never `skip()` on large offsets** (it scans+discards):
```javascript
// SLOW:  db.posts.find().sort({createdAt:-1}).skip(10000).limit(20)
// FAST:  page by the last seen sort key (+ _id tiebreaker for uniqueness)
db.posts.find({ $or:[ { createdAt:{ $lt:lastTs } },
                      { createdAt:lastTs, _id:{ $lt:lastId } } ] })
        .sort({ createdAt:-1, _id:-1 }).limit(20);
```

Prefer soft deletes (`deletedAt`) where audit/history matters. Avoid `$where` and unanchored `$regex` (can't use an index, and `$where` is an injection vector — MONGO-SEC-02).

---

## 7. Transactions

Single-document writes are **already atomic** — the document model lets you avoid most multi-document transactions by embedding what changes together. Reach for a multi-document transaction only when invariants span collections/documents (e.g. money transfer).

```javascript
const session = client.startSession();
try {
  await session.withTransaction(async () => {            // auto-retries TransientTransactionError
    await accounts.updateOne({ _id:from }, { $inc:{ balance:-amount } }, { session });
    await accounts.updateOne({ _id:to   }, { $inc:{ balance: amount } }, { session });
    await ledger.insertOne({ from, to, amount, at:new Date() }, { session });
  }, { writeConcern:{ w:"majority" }, readConcern:{ level:"snapshot" } });
} finally { await session.endSession(); }
```

`withTransaction` handles the retry loop for transient/commit-unknown errors (see [`error-handling.md`](guides://error-handling.md)). Keep transactions short (<60 s default), touch few documents, and design to *avoid* them where embedding suffices. Distributed (cross-shard) transactions must read from `primary`.

---

## 8. Replica Sets, Concerns & Read Preference

A replica set is the unit of durability and HA. Run **P-S-S** minimum (3 data-bearing members); never P-S-A with majority writes (an arbiter can't acknowledge data, so majority stalls on one node loss) — MONGO-RS-01.

**Write concern** (durability — see `error-handling.md`):
- `w:"majority", wtimeoutMS:5000` — default for durable writes; survives failover without rollback. Add `j:true` when you need journal durability.
- `w:1` (primary-only) and `w:0` (fire-and-forget) risk data loss on failover — not for production business data.
- Set a cluster default: `db.adminCommand({ setDefaultRWConcern:1, defaultWriteConcern:{ w:"majority", wtimeoutMS:5000 } })`.

**Read concern**: `"majority"` (no rollback risk) for critical reads; `"local"` for lowest latency; `"snapshot"` inside transactions; `"linearizable"` only when you must read your own latest write (high cost, always set `maxTimeMS`).

**Read preference**: `primary` (default, most current) for user-facing reads; `secondaryPreferred` for background/analytics jobs; `nearest` for lowest latency in geo-distributed deployments; tag sets to pin analytics to a hidden member. Transactions must use `primary`.

Keep the **oplog window** ≥ 24 h (72 h+ for production) so secondaries, change streams (§10), and resharding can catch up: `rs.printReplicationInfo()`.

---

## 9. Sharding

Shard to scale **beyond a single replica set** (storage or throughput) — not before vertical scaling and indexing are exhausted. The **shard key choice is the single most consequential and hard-to-reverse decision**.

```javascript
db.adminCommand({ analyzeShardKey:"mydb.events", key:{ userId:1, ts:1 } }); // 7.0+: validate FIRST
```

Shard-key criteria (MONGO-SHARD-01):
- **High cardinality** — many distinct values, so chunks can split. Low-cardinality keys (`country`) create giant unsplittable chunks.
- **Low frequency** — no single value dominates (no hot key).
- **Non-monotonic** — `_id:1` or a raw timestamp routes every new write to one shard (hotspot). Use a **hashed** key for even write distribution, or a **compound** key with a high-cardinality, non-monotonic prefix.

```javascript
sh.shardCollection("mydb.events", { userId:"hashed" });          // even writes, no range queries
sh.shardCollection("mydb.events", { userId:1, ts:1 });           // ranged: targeted user+time queries
```

Queries that include the shard key are **targeted** (one shard); those without are **scatter-gather** (all shards) — design keys so hot queries are targeted. **Zone sharding** pins data ranges to regions for locality/GDPR (`sh.addShardToZone` + `sh.updateZoneKeyRange`). **Resharding** (`reshardCollection`) is available and far faster in 8.0, but still expensive — get the key right up front.

---

## 10. Change Streams

Change streams expose the oplog as a resumable, filterable real-time event feed (requires a replica set) — the foundation for CDC, cache invalidation, and event-driven sync without polling.

```javascript
const stream = db.collection("orders").watch(
  [ { $match:{ operationType:{ $in:["insert","update"] }, "fullDocument.status":"pending" } } ],
  { fullDocument:"updateLookup", resumeAfter: await loadToken() }   // resume after restart
);
for await (const change of stream) {
  await handle(change);
  await saveToken(change._id);     // persist resume token for fault tolerance
}
```

Production notes:
- **Persist the resume token** (`change._id` / `postBatchResumeToken`) so a restart resumes exactly-once-ish from where it stopped.
- Handle oplog roll-over: on `ChangeStreamHistoryLost`/`CappedPositionLost` (code 286/136) the token is gone — restart from a checkpoint (`startAtOperationTime`) and reconcile. Size the oplog for max consumer downtime (§8).
- For documents near 16 MB, enable `$changeStreamSplitLargeEvent`.
- Filter server-side in the pipeline; many wide-open streams add cluster load.

---

## 11. Operations: time-series, monitoring, performance, config

### A. Time-series collections
For append-heavy, time-stamped metrics, use a native time-series collection (columnar storage, automatic bucketing, far less index/cache than a bucketed regular collection):
```javascript
db.createCollection("sensor_data", {
  timeseries: { timeField:"ts", metaField:"deviceId", granularity:"seconds" },
  expireAfterSeconds: 2592000          // 30-day TTL
});
```
Use a stable `metaField` (the series identity), batch inserts per metaField, and query/shard on `metaField` sub-fields. MongoDB 8.0 brings block processing and large throughput/cache gains over 7.0.

### B. Monitoring & profiling (see `observability.md`)
```javascript
db.setProfilingLevel(1, { slowms: 100 });               // log queries > 100 ms (MONGO-OBS-01)
db.system.profile.find().sort({ ts:-1 }).limit(10);
db.serverStatus().connections; db.serverStatus().opcounters;
```
Track: replication lag, oplog window, working-set vs RAM, cache hit ratio, connection-pool checkouts/waits, p95/p99 latency. **Atlas Search / vector search**: full-text and `$vectorSearch` ANN indexes are managed separately from regular indexes — define them via Atlas Search index config, not `createIndex`.

### C. Performance (see `performance.md`)
The dominant lever is keeping the **working set (hot data + indexes) in RAM** and serving hot queries from indexes (§4). Project to needed fields, paginate by keyset (§6), batch with `bulkWrite({ordered:false})`, and prefer `decimal128`/short field names for large high-cardinality collections. Connection pooling lives in the driver — size `maxPoolSize` to real concurrency; total connections = app instances × poolSize × members.

### D. Configuration (see `env-config.md`)
The connection string carries credentials and topology — load it from the environment/secret store, never hardcode it (MONGO-CFG-01). For local dev and integration tests, stand up a replica set / sharded cluster with [`docker-compose.md`](guides://docker-compose.md) (replica set required for transactions & change streams).
```
MONGODB_URI=mongodb+srv://app:${MONGO_PW}@cluster0.example.mongodb.net/mydb?retryWrites=true&w=majority
```

---

## 12. Security binding

Policy is owned by [`secure-coding.md`](guides://secure-coding.md). MongoDB bindings (MONGO-SEC-01/02):

- **AuthN**: enable auth (`--auth`); SCRAM by default, X.509 for service-to-service. No anonymous access.
- **AuthZ (RBAC)**: least-privilege custom roles scoped to specific DBs/collections/actions; never give an application the `root`/`dbOwner` role.
- **Injection**: MongoDB has no SQL, but operator injection is real — never let user input become a query operator. Pass user values as data, cast types, reject keys starting with `$`/`.`, and never use `$where`/`$expr` with untrusted input.
- **TLS in transit**: `--tlsMode requireTLS` (always on in Atlas).
- **Encryption at rest**: storage-engine encryption / KMIP (Enterprise/Atlas).
- **Queryable Encryption**: encrypt PII (SSN, salary, email) client-side while still querying it — equality and range (and prefix/suffix in 8.2 preview). Server never sees plaintext. Prefer it over legacy CSFLE for new apps; CSFLE only when you need per-tenant keys on the same field.
```javascript
const encryptedFieldsMap = { "mydb.users": { fields: [
  { path:"ssn",    bsonType:"string", queries:{ queryType:"equality" } },
  { path:"salary", bsonType:"int",    queries:{ queryType:"range", min:0, max:1_000_000 } }
] } };
```
- **Network**: bind to private interfaces, IP allow-list / VPC peering / PrivateLink (Atlas).

---

## 13. Quick Reference

```javascript
// shell / model
use mydb; show collections; db.col.stats()
db.createCollection("c", { validator:{ $jsonSchema:{...} }, validationAction:"error" })

// index & explain
db.col.createIndex({ a:1, b:-1, c:1 })            // ESR: Equality, Sort, Range
db.col.find(q).explain("executionStats")          // want IXSCAN, docsExamined≈nReturned
db.col.aggregate([{ $indexStats:{} }])            // drop unused

// aggregation
db.col.aggregate([{ $match:{...} }, { $group:{...} }, { $sort:{...} }])  // $match first

// replica set / sharding
rs.status(); rs.printReplicationInfo()            // oplog window
sh.shardCollection("mydb.col", { k:"hashed" })
db.adminCommand({ analyzeShardKey:"mydb.col", key:{ k:1 } })

// ops
db.setProfilingLevel(1, { slowms:100 })
db.adminCommand({ setDefaultRWConcern:1, defaultWriteConcern:{ w:"majority", wtimeoutMS:5000 } })
mongodump --uri="$MONGODB_URI" --out=/backup    # backup / mongorestore to restore
```

---

## 14. Deployment Checklist

Generated from §2 — one box per requirement ID.

- [ ] MONGO-MODEL-01 — embed/reference justified; no unbounded arrays
- [ ] MONGO-MODEL-02 — `$jsonSchema` validation on business collections
- [ ] MONGO-MODEL-03 — max document size well under 16 MB
- [ ] MONGO-IDX-01 — hot queries index-backed, no COLLSCAN
- [ ] MONGO-IDX-02 — compound indexes follow ESR
- [ ] MONGO-IDX-03 — TTL/partial indexes where applicable; unused indexes dropped
- [ ] MONGO-WC-01 — `w:"majority"` + `wtimeoutMS` (see `error-handling.md`)
- [ ] MONGO-WC-02 — retryable writes & transient-txn retry enabled
- [ ] MONGO-RS-01 — P-S-S topology (not P-S-A)
- [ ] MONGO-SHARD-01 — shard key validated (cardinality, non-monotonic) via `analyzeShardKey`
- [ ] MONGO-SEC-01 — auth + TLS + least-privilege roles, no root app user (see `secure-coding.md`)
- [ ] MONGO-SEC-02 — no operator injection; PII under Queryable Encryption
- [ ] MONGO-CFG-01 — connection string from env/secret store (see `env-config.md`)
- [ ] MONGO-OBS-01 — slow-query profiler / monitoring enabled (see `observability.md`)
- [ ] Agent ran the §2 verify commands and documented any fixes

---
**End of MongoDB Guidelines**
