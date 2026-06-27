# libSQL & Turso Development Guidelines
Mandatory standards for libSQL/Turso: embedded replicas for local-first reads, single-writer remote primary, scoped auth tokens, tested sync, native vector search. libSQL, Turso CLI, @libsql/client.

---
name: libsql-turso
title: libSQL & Turso Development Guidelines
version: 2.0
last_reviewed: 2026-06-05
kind: datastore
tools: [libsql, turso-cli, "@libsql/client@0.14", libsql-server]
requires:
  - sql
  - secure-coding
recommends:
  - sqlite
  - error-handling
  - performance
  - env-config
provides:
  - libsql-embedded-replicas
  - turso-edge
  - libsql-sync
  - database-per-tenant
---

> 🧭 Authored per [`CONVENTIONS.md`](guides://CONVENTIONS.md): shared concerns are referenced, not restated. This guide covers only the **libSQL/Turso delta over plain SQLite** — the network engine, embedded replicas, sync, edge/multi-tenant deployment, scoped tokens, and native vector search.

---

## 0. Prerequisites & References

Fetch and apply these **before** writing libSQL/Turso code. Their rules are assumed below and not repeated.

> 📎 **REQUIRED — fetch & apply first:**
> - [`sql.md`](guides://sql.md) — relational modeling, query style, joins, transactions/ACID, migrations. *(This guide does not restate generic SQL.)*
> - [`secure-coding.md`](guides://secure-coding.md) — injection, secrets, supply chain. *(libSQL binding: **always** parameterize; mint **least-privilege, expiring** auth tokens; keep token + DB URL out of source.)*

> 📎 **RECOMMENDED — fetch when the task touches them:**
> - [`sqlite.md`](guides://sqlite.md) — **the base engine.** WAL, `STRICT` tables, `PRAGMA foreign_keys`, type affinity, FTS5, `VACUUM INTO` all apply unchanged to local/embedded-replica files. This guide binds only the **libSQL deltas** and never restates those PRAGMas.
> - [`error-handling.md`](guides://error-handling.md) — *(binding: classify network/`SYNC` errors vs SQL errors; degrade to local replica on sync failure; retry `SQLITE_BUSY` with backoff)*
> - [`performance.md`](guides://performance.md) — *(binding: serve reads from the local embedded replica or nearest edge replica; never round-trip the primary on a read path)*
> - [`env-config.md`](guides://env-config.md) — *(binding: `TURSO_DATABASE_URL` + `TURSO_AUTH_TOKEN` come from env/secret manager, never literals; per-environment tokens)*

> 📎 **SEE ALSO:** [`postgresql.md`](guides://postgresql.md) *(when you outgrow single-writer entirely)* · [`chroma-vectordb.md`](guides://chroma-vectordb.md) *(dedicated vector store at scale)* · [`nextjs.md`](guides://nextjs.md) / [`deno.md`](guides://deno.md) *(edge runtimes)*

---

## 1. Core Philosophies: EDGE-FIRST

libSQL/Turso-specific principles only. TDD, generic SQL, the SQLite engine, security, and error policy come from §0.

- **E**mbedded replicas are the headline: a full local SQLite file kept in sync from the remote primary. Reads hit local disk (µs); writes are forwarded to the primary and applied locally → **read-your-writes** with zero read latency.
- **D**istributed single-writer: exactly one logical writer (the primary). Replicas (edge or embedded) are read paths with **eventual** consistency. Design around this, not against it.
- **G**lobal reads, central writes: route reads to the nearest replica; accept that writes pay a round-trip to the primary's region.
- **E**xplicit sync: choose `syncInterval` (periodic) and/or call `sync()` on demand before consistency-critical reads. Treat sync as fallible I/O.
- **F**ull SQLite compatibility: the on-disk format and SQL dialect are SQLite's — but extension loading is restricted on hosted Turso, and network transactions are latency-sensitive (prefer `batch()`).
- **I**solation by database: prefer **database-per-tenant** at the edge over a shared schema; scope a token to each tenant DB.
- **R**est on the base guide: PRAGMAs/WAL/STRICT/backups are owned by [`sqlite.md`](guides://sqlite.md). This guide adds only what the network/replica layer changes.
- **T**ested sync: integration-test embedded-replica sync, read-your-writes, and offline degradation — not just `:memory:`.

**Verified Code**: agent-generated libSQL/Turso code MUST parameterize SQL, use a scoped non-expiring-forbidden token, handle sync/network failure, and pass every §2 gate before delivery.

---

## 2. Requirements (MANDATORY, auditable)

RFC-2119 keywords. IDs `TURSO-<TOPIC>-<NN>`. Each row has a binary gate; rows binding a shared rule cite its owner. SQLite-engine gates (WAL, STRICT, foreign_keys, integrity_check, parameterization) are inherited from [`sqlite.md`](guides://sqlite.md) and not duplicated here.

| ID | Requirement | Verify | Gate |
|----|-------------|--------|------|
| TURSO-SEC-01 | All SQL MUST be parameterized — no string interpolation (see `secure-coding.md`, `sqlite.md`) | grep / code review | no f-string/`%`/`+`/template SQL |
| TURSO-SEC-02 | Auth tokens MUST be least-privilege: read paths use `--read-only` tokens; tokens MUST carry an expiration (see `secure-coding.md`) | `turso db tokens create … --expiration` in IaC | no non-expiring full-access token in prod |
| TURSO-CFG-01 | DB URL + token MUST come from env/secret manager, never source (see `env-config.md`) | grep for `libsql://`/`eyJ` literals | none in repo |
| TURSO-URL-01 | Remote connections MUST use TLS (`libsql://`/`https://`/`wss://`); never plaintext `http://`/`ws://` in prod | review connection string | TLS scheme |
| TURSO-REPL-01 | Embedded-replica writes MUST be read-your-writes verified (write then local read) | integration test | local read reflects write |
| TURSO-SYNC-01 | `sync()` failures MUST be handled and degrade to local data, not crash (see `error-handling.md`) | test sync under network failure | app serves stale-local |
| TURSO-SYNC-02 | Consistency-critical reads MUST `sync()` (or use a fresh remote read) before reading | code review | sync/remote before critical read |
| TURSO-TXN-01 | Multi-statement writes MUST use `batch()`/transaction, not N network round-trips | code review | one batch/txn per unit |
| TURSO-PERF-01 | Read paths MUST hit a replica (embedded/edge), never round-trip the primary (see `performance.md`) | review / latency check | reads local/edge |
| TURSO-TENANT-01 | Multi-tenant apps MUST isolate per tenant (database-per-tenant) with a per-DB token | review | no cross-tenant token/DB reuse |
| TURSO-VEC-01 | Vector columns MUST use `F32_BLOB(dim)` + a vector index for ANN queries | `EXPLAIN QUERY PLAN` on vector query | uses `libsql_vector_idx` |
| TURSO-MIG-01 | Schema migrations MUST be versioned and idempotent; applied to primary | migration runner | re-run is a no-op |
| TURSO-BAK-01 | Point-in-time restore MUST be tested (restore to a scratch DB + integrity check) | `turso db create --from-db … --timestamp` + `PRAGMA integrity_check` | restored DB `ok` |
| TURSO-TEST-01 | Tests MUST run against a real `libsql-server`/replica file, not only `:memory:` (see `sqlite.md`) | test config | uses on-disk/server target |

> **Forbidden**: non-expiring full-access tokens in production, tokens or URLs committed to source, reads that round-trip the primary, a shared-schema tenant table where database-per-tenant was required, interactive multi-round-trip network transactions, or assuming a replica read reflects the latest write without a prior `sync()`.

---

## 3. Verification Protocol

Run before presenting code. Fix → re-run until green. SQLite-engine checks (WAL/FK/integrity) run per [`sqlite.md`](guides://sqlite.md) on the local/replica file.

```bash
turso db show <db> --url                       # TURSO-URL-01 → libsql:// (TLS)
turso db tokens create <db> --read-only --expiration 7d   # TURSO-SEC-02 (read path)
grep -rEn "libsql://|eyJ[A-Za-z0-9_-]{10,}" src/          # TURSO-CFG-01 → no hits
# Embedded-replica read-your-writes + sync-failure tests (TURSO-REPL-01/SYNC-01) run in the test suite
turso db shell <db> "EXPLAIN QUERY PLAN SELECT * FROM docs ORDER BY vector_distance_cos(embedding, vector32('[...]')) LIMIT 5"  # TURSO-VEC-01
turso db create restore-test --from-db <db> --timestamp <ts>   # TURSO-BAK-01
```

The *why* behind each gate lives in its §0 owner; do not re-derive it here.

---

## 4. Connection Modes (the core libSQL decision)

libSQL exposes the same SQLite engine through several **client modes**. Picking the mode is the central design choice; the SQL on top is identical.

| Mode | Builder / URL | Where data lives | Reads | Writes | Use when |
|------|---------------|------------------|-------|--------|----------|
| **Local file** | `file:./app.db` | local disk only | local | local | dev, tests, pure embedded app (same as `sqlite.md`) |
| **In-memory** | `:memory:` | RAM | local | local | unit tests |
| **Remote** | `libsql://<db>.turso.io` + token | remote primary/edge | network round-trip | primary | serverless/edge functions that can't keep a file |
| **Embedded replica** ⭐ | `file:./local.db` + `syncUrl` + token | local file, synced from remote | **local (µs)** | forwarded to primary, applied locally | long-lived apps, local-first, desktop/mobile, read-heavy services |

URL schemes: `libsql://` (Hrana over WebSocket/HTTP, TLS), `https://`/`http://` (Hrana over HTTP — good for edge runtimes without WS), `wss://`/`ws://`, `file:`. Use TLS schemes for anything remote (TURSO-URL-01). The wire protocol is **Hrana**; the HTTP variant is what makes Turso work inside Cloudflare Workers / Vercel Edge where raw TCP/WS is unavailable.

```typescript
import { createClient } from "@libsql/client";        // edge: "@libsql/client/web"

// Remote (serverless/edge)
const remote = createClient({
  url: process.env.TURSO_DATABASE_URL!,                // libsql://...
  authToken: process.env.TURSO_AUTH_TOKEN!,
});

// Embedded replica — the headline feature
const replica = createClient({
  url: "file:./local.db",                              // local SQLite file
  syncUrl: process.env.TURSO_DATABASE_URL!,            // remote primary
  authToken: process.env.TURSO_AUTH_TOKEN!,
  syncInterval: 60,                                    // seconds; periodic background sync
});
```

SDKs: **TS/JS** `@libsql/client` (`/web` build for edge), **Rust** `libsql` crate (`Builder::new_remote_replica`/`new_synced_database`), **Python** `libsql` (the maintained successor to `libsql-client`), **Go** `go-libsql` (embedded replicas) / `libsql-client-go` (remote). API shape is consistent: `execute(sql, args)`, `batch([...])`, `transaction()`, `sync()`.

---

## 5. Embedded Replicas & Sync (the libSQL delta)

A full local copy of the database synced from the remote primary. **Reads** are served from local disk at SQLite speed; **writes** are sent to the primary *and* applied to the local file so the same connection reads its own writes (TURSO-REPL-01).

```typescript
// Periodic background sync (set at construction) + on-demand sync
await replica.sync();                                  // pull latest frames now
const r = await replica.sync();  // r.frames_synced / frame_no for observability

// Sync before a consistency-critical read (TURSO-SYNC-02)
async function readReceipt(id: number) {
  await replica.sync();                                // ensure freshest data
  return replica.execute({ sql: "SELECT * FROM receipts WHERE id = ?", args: [id] });
}
```

**Sync is fallible network I/O — handle it (TURSO-SYNC-01):**

```typescript
try {
  await replica.sync();
} catch (err) {
  // Offline/degraded: keep serving the local replica; surface staleness, do not crash.
  logger.warn({ err }, "sync failed; serving local replica");
}
```

- **Choosing `syncInterval`**: real-time UI ~1–5 s; typical app ~30–60 s; occasional ~minutes. Shorter = fresher reads + more bandwidth/primary load. Combine periodic interval with manual `sync()` after important writes and on app foreground (mobile/desktop).
- **Storage**: the replica is a normal WAL SQLite file — `sqlite.md` PRAGMas/backups apply. Don't put it on a network FS (`sqlite.md` SQLITE-FS-01).
- **Read-your-writes** holds on the writing connection; **other** replicas converge only after their next sync (eventual consistency).
- Encryption at rest for the local replica file is available (`encryptionKey`) — keep the key out of source (see `secure-coding.md`).

---

## 6. Turso Platform Model

```
Organization → Databases → Group (region set: primary + edge replicas)
                         → per-DB auth tokens (full / read-only, expiring)
```

```bash
turso db create my-app --group default        # create DB in a group
turso db show my-app --url                     # libsql:// connection URL
turso db tokens create my-app --read-only --expiration 30d   # scoped token
turso group locations add default fra lhr      # add edge regions (replicas)
turso db create my-app-restore --from-db my-app --timestamp 2026-06-20T10:00:00Z  # PITR/branch
```

- **Groups** hold the primary + edge replica regions; a database belongs to a group and is replicated to every region in it. Adding a location adds an **edge replica** (read-only, low-latency for nearby clients).
- **Server modes (self-host)**: `libsql-server` (formerly `sqld`) runs the same engine for local dev (`http://127.0.0.1:8080`), on-prem, or CI. Use it for `TURSO-TEST-01` integration tests instead of mocking.
- **Branching / PITR**: `turso db create --from-db <db> [--timestamp <ts>]` forks a database (a dev branch or a point-in-time restore). Test the restore (TURSO-BAK-01).

### Multi-tenancy — database-per-tenant (TURSO-TENANT-01)

The Turso-idiomatic pattern: one **database per tenant**, created on demand, each with its own scoped token. Cheap because databases are lightweight; gives hard isolation, per-tenant backup/restore, independent scaling, and trivial GDPR delete (drop the DB). Reserve shared-schema (`WHERE tenant_id = ?`) for very small/uniform tenants only.

```typescript
// Provision + connect per tenant
await platform.createDatabase({ name: `tenant-${id}`, group: "default" });
const token = await platform.createToken(`tenant-${id}`, { authorization: "read-only" });
const db = createClient({ url: `libsql://tenant-${id}-${org}.turso.io`, authToken: token });
```

---

## 7. How libSQL Differs from Plain SQLite

Bind only the deltas; everything else is [`sqlite.md`](guides://sqlite.md).

- **Concurrency**: a Turso/`libsql-server` primary accepts many concurrent client connections and serializes writes server-side — you get connection-level concurrency without managing local file locks. Still a **single logical writer**; write-heavy fan-out is the wrong fit (→ `postgresql.md`).
- **Network transactions**: interactive `transaction()` holds a connection across round-trips and is latency-sensitive; prefer `batch()` (one round-trip, atomic) for multi-statement writes (TURSO-TXN-01).
- **Extensions**: hosted Turso restricts loading arbitrary native extensions (`.so`/`.dylib`). FTS5/JSON1 are built in; don't depend on `load_extension` against Turso.
- **Native vector search** (built into libSQL — no extension): store embeddings in a `F32_BLOB(n)` column, insert with `vector32('[...]')`, query with `vector_distance_cos`/`vector_distance_l2`, and build an ANN index with `libsql_vector_idx` (TURSO-VEC-01):

```sql
CREATE TABLE docs (id INTEGER PRIMARY KEY, body TEXT, embedding F32_BLOB(768));
CREATE INDEX docs_vec ON docs (libsql_vector_idx(embedding));

INSERT INTO docs (body, embedding) VALUES (?, vector32(?));   -- ? = '[0.1, 0.2, ...]'

SELECT id, body
FROM vector_top_k('docs_vec', vector32(:query), 5) AS v   -- ANN, index-backed
JOIN docs ON docs.rowid = v.id;
```

- **ALTER/schema**: standard SQLite `ALTER TABLE` plus Turso "schema databases" for fan-out migrations across a database group. Keep migrations versioned + idempotent (TURSO-MIG-01).
- **Forwarded writes**: from an embedded replica, an `INSERT`/`UPDATE` transparently goes to the primary — there is no separate "write client".

---

## 8. Edge & Serverless Binding

Edge runtimes (Cloudflare Workers, Vercel/Next.js Edge, Deno Deploy) can't hold a long-lived file or raw socket → use **remote** mode over HTTP-Hrana with the `/web` client; create the client **per request** (no module-level singleton that outlives the isolate).

```typescript
// Cloudflare Worker / Vercel Edge
import { createClient } from "@libsql/client/web";
export default {
  async fetch(req: Request, env: Env) {
    const db = createClient({ url: env.TURSO_DATABASE_URL, authToken: env.TURSO_AUTH_TOKEN });
    const { rows } = await db.execute("SELECT 1");
    return Response.json(rows);
  },
};
```

Long-lived servers (Node, containers, desktop, mobile) → prefer an **embedded replica** for read latency. Secrets via the platform's env/secret store (`env-config.md`), never bundled.

---

## 9. Choosing Turso vs SQLite vs a Server DB

- **Plain SQLite** ([`sqlite.md`](guides://sqlite.md)): single machine, no remote access, no replication needed. Simplest — start here.
- **libSQL/Turso**: you need remote access from serverless/edge, multi-region low-latency reads, **local-first with sync**, or cheap **database-per-tenant** — while keeping SQLite's model and SQL. Still single-writer.
- **PostgreSQL / a clustered server DB** ([`postgresql.md`](guides://postgresql.md)): high write concurrency/throughput, multi-writer, large datasets, or rich server-side features beyond SQLite. Move here when single-writer is the bottleneck.

---

## 10. Deployment Checklist

Generated from §2 — one box per requirement ID. No new requirements.

- [ ] TURSO-SEC-01 — all SQL parameterized
- [ ] TURSO-SEC-02 — tokens least-privilege + expiring; no non-expiring full-access in prod
- [ ] TURSO-CFG-01 — URL/token from env/secret manager, none in source
- [ ] TURSO-URL-01 — remote connections use TLS scheme
- [ ] TURSO-REPL-01 — embedded-replica read-your-writes verified
- [ ] TURSO-SYNC-01 — sync failures degrade to local, no crash
- [ ] TURSO-SYNC-02 — critical reads sync first
- [ ] TURSO-TXN-01 — multi-statement writes use batch/transaction
- [ ] TURSO-PERF-01 — read paths hit a replica, not the primary
- [ ] TURSO-TENANT-01 — per-tenant DB + per-DB token
- [ ] TURSO-VEC-01 — vector columns use F32_BLOB + vector index
- [ ] TURSO-MIG-01 — migrations versioned & idempotent
- [ ] TURSO-BAK-01 — point-in-time restore tested
- [ ] TURSO-TEST-01 — tests run against a real server/replica file
- [ ] SQLite-engine gates (WAL, STRICT, foreign_keys, integrity_check) green per `sqlite.md`
- [ ] Agent ran every §3 command and documented any fixes

---
**End of libSQL & Turso Guidelines**
