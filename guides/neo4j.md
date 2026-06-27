# Neo4j Development Guidelines
Mandatory standards for Neo4j graph modeling, Cypher, indexing, traversal, and operations. Neo4j 5.x, Cypher, APOC, GDS 2.x, official drivers.

---
name: neo4j
title: Neo4j Development Guidelines
version: 2.0
last_reviewed: 2026-06-05
kind: datastore
tools: [neo4j@5, cypher, apoc, gds@2, neo4j-driver]
requires:
  - secure-coding
  - error-handling
recommends:
  - observability
  - performance
  - sql
  - env-config
provides:
  - property-graph-model
  - cypher
  - graph-modeling
  - neo4j-indexing
  - graph-traversal
---

> 🧭 Authored per [`CONVENTIONS.md`](guides://CONVENTIONS.md): shared concerns are referenced, not restated. This guide covers only what is unique to Neo4j.

---

## 0. Prerequisites & References

Fetch and apply these **before** modeling a graph or writing Cypher. Their rules are assumed below and not repeated.

> 📎 **REQUIRED — fetch & apply first:**
> - [`secure-coding.md`](guides://secure-coding.md) — auth, RBAC, secrets, injection, CVE policy. *(Neo4j binding: native/LDAP/SSO auth, fine-grained RBAC, **Cypher injection defended by `$` parameters**, TLS — see §9.)*
> - [`error-handling.md`](guides://error-handling.md) — error strategy & propagation. *(Neo4j binding: retry `TransientError` — deadlocks, leader switches, `Neo.TransientError.*` — via **managed transaction functions**; see §6.)*

> 📎 **RECOMMENDED — fetch when the task touches them:**
> - [`performance.md`](guides://performance.md) — perf policy *(binding: `EXPLAIN`/`PROFILE`, index-backed starts, page cache — §5, §7)*
> - [`sql.md`](guides://sql.md) — relational comparison; read when deciding **graph vs relational** (§1, §3)
> - [`observability.md`](guides://observability.md) — metrics/tracing policy *(binding: query log, Prometheus/JMX metrics — §8)*
> - [`env-config.md`](guides://env-config.md) — config policy *(binding: Bolt URI + credentials from env/secret store — §9)*

> 📎 **SEE ALSO:** [`tdd.md`](guides://tdd.md) (test Cypher against an ephemeral Neo4j via Testcontainers / `neo4j` Docker) · [`postgresql.md`](guides://postgresql.md) · [`mongodb.md`](guides://mongodb.md) · [`docker-compose.md`](guides://docker-compose.md)

---

## 1. Core Philosophies: GRAPH-FIRST

Neo4j-specific principles only. Security, error handling, observability come from §0.

- **G**raph-native or don't: Neo4j wins when **relationships are the query** — deep/variable-length traversals, pathfinding, pattern detection (fraud rings, recommendations, knowledge graphs, dependency/impact analysis). For tabular CRUD, aggregate reporting, or data with few connections, a relational store fits better ([`sql.md`](guides://sql.md)). Don't force a graph where rows-and-joins is the natural shape.
- **R**elationships are first-class: model connections as typed, directed relationships (index-free adjacency → O(1) hop), never as foreign-key-style properties or string lists.
- **A**nchor every query: start from an indexed node (constraint/index), not an `AllNodesScan` or `NodeByLabelScan`; bound variable-length paths (`*1..3`, never bare `*`).
- **P**arameterize always: queries use `$params` — for the query-plan cache **and** to make Cypher injection impossible (see `secure-coding.md`).
- **H**ermetic & idempotent: `MERGE` against a uniqueness constraint for idempotent upserts; batch writes with `UNWIND`; test against a throwaway Neo4j instance.

**Verified Code**: Agent-generated models, constraints, and Cypher MUST pass every gate in §2 before delivery.

---

## 2. Requirements (MANDATORY, auditable)

RFC-2119 keywords. IDs `NEO-<TOPIC>-<NN>`. Each row has a binary gate; rows binding a shared rule cite its owner.

| ID | Requirement | Verify | Gate |
|----|-------------|--------|------|
| NEO-MODEL-01 | Entities MUST be nodes and connections MUST be relationships; no entity-as-property or FK/id-list emulation of edges | design review vs §3 | edges modeled as relationships |
| NEO-MODEL-02 | Identity writes MUST be idempotent via `MERGE` on a key backed by a uniqueness/node-key constraint | review; `SHOW CONSTRAINTS` | constraint present, MERGE used |
| NEO-MODEL-03 | Variable-length/traversal queries MUST be upper-bounded and supernodes mitigated | grep for bare `*`; review hot labels | bounded paths, no unbounded fan-out |
| NEO-CYPHER-01 | All Cypher MUST pass user values as `$parameters`, never string-concatenated (see `secure-coding.md`) | code review / grep | no string-built Cypher |
| NEO-IDX-01 | Hot query start points MUST be index-backed (no `AllNodesScan`/`NodeByLabelScan` on large labels) | `EXPLAIN <query>` | plan shows `NodeIndexSeek`/`NodeUniqueIndexSeek` |
| NEO-IDX-02 | Natural keys MUST have a uniqueness or node-key constraint (auto-creates the index) | `SHOW CONSTRAINTS` | constraint per natural key |
| NEO-PERF-01 | Non-trivial queries MUST be profiled; no cartesian products; no `Eager` in batched writes (see `performance.md`) | `PROFILE <query>` | no `CartesianProduct`/unexpected `Eager`; db hits sane |
| NEO-TXN-01 | Transient errors MUST be retried via managed transaction functions (see `error-handling.md`) | code review | `execute_read`/`execute_write` used, not raw `tx` |
| NEO-SEC-01 | Auth enabled, default `neo4j` password rotated, least-privilege RBAC roles, TLS (`neo4j+s`) (see `secure-coding.md`) | `SHOW USERS`/`SHOW ROLES`; connection scheme | scoped roles, TLS on, no default creds |
| NEO-CFG-01 | Bolt URI & credentials MUST come from env/secret store, never hardcoded (see `env-config.md`) | grep source | no literal URIs/passwords |
| NEO-OBS-01 | Query logging / metrics MUST be enabled (see `observability.md`) | `neo4j.conf`; metrics endpoint | query log + metrics on |
| NEO-TST-01 | Schema & queries MUST be tested against an ephemeral Neo4j (see `tdd.md`) | CI test run | tests green |

> **Forbidden**: building Cypher by string-concatenating user input; unbounded variable-length paths (`-[*]-`); cartesian products from disconnected `MATCH` patterns; `DETACH DELETE n` over `MATCH (n)` in production; running an app as `neo4j`/admin or with the default password; hardcoded connection strings or credentials; ignoring `TransientError` instead of retrying.

---

## 3. The Property Graph Model & Modeling (the central decision)

A **property graph** = **nodes** (entities; one or more **labels** + properties) connected by **relationships** (a single **type**, a **direction**, optional properties). Properties are key-value pairs on either. Coming from relational? See [`sql.md`](guides://sql.md): a table row → a node; a foreign key → a relationship; a join table → a relationship (often with properties).

### A. Nodes vs relationships vs properties

```cypher
// Entities are nodes; connections are typed relationships
CREATE (a:Person {name:'Alice'})-[:WORKS_AT {since: date('2021-01-01')}]->(c:Company {name:'Acme'})
```

Decide:
- **Node** — anything with independent identity, its own properties, or that other things connect to (a `Person`, `Company`, `Product`).
- **Relationship** — a connection between two nodes. Put data that describes the *connection* (`since`, `weight`, `role`) on the relationship.
- **Property** — a scalar/attribute of a node or relationship. **Don't** encode a connected entity as a property (`{company:'Acme'}`) or a list of foreign ids (`{friends:['Bob','Carol']}`) — that throws away index-free adjacency and forces scans.
- **Promote a relationship to a node** ("reified" / intermediate node) when the connection itself has identity or connects to more things — e.g. a `Role`/`Booking`/`Order` between actor and movie, so you can attach awards, multiple participants, or time validity.

### B. Patterns

- **Many-to-many** — just a relationship with properties: `(s:Student)-[:ENROLLED_IN {grade:'A'}]->(c:Course)`. No join table.
- **Hierarchies / trees** — `(:Employee)-[:REPORTS_TO]->(:Manager)`; query the whole subtree with a bounded variable-length path `<-[:REPORTS_TO*1..5]-`.
- **Time-validity** — keep `from`/`to` on the relationship; "current" = `WHERE r.to IS NULL`. For full history, model each version as a node.
- **Conventions** — labels `PascalCase` singular (`:Person`), relationship types `UPPER_SNAKE` verbs (`:WORKS_AT`), properties `camelCase`. Direction is stored once; you can traverse either way — pick the direction that reads naturally and **omit direction in MATCH** only when you truly mean both ways.

### C. Supernodes — the key footgun (NEO-MODEL-03)

A **supernode** has disproportionately many relationships (millions of `:LIKES` into one celebrity/country/"USA" node). Traversals *through* it explode. Mitigate:
- Split the hot relationship type by a discriminator (e.g. `:LIKED_IN_2026`), or add intermediate "category" nodes to fan out.
- Always anchor on the *other*, selective end and bound the path; never traverse a supernode unbounded.
- Move filterable data onto the relationship so you can prune early (relationship property indexes help).

### D. Denormalize deliberately

Graphs avoid most relational denormalization (the relationship *is* the join). Cache a computed value (`friendCount`) or a hot copied field only when a measured read path needs it — and own the write-time update. Don't pre-duplicate by default.

---

## 4. Cypher

Cypher is the declarative pattern-matching query language: you draw the pattern with ASCII-art `()-[]->()` and the engine finds matches.

### A. Read — MATCH / WHERE / OPTIONAL MATCH

```cypher
MATCH (p:Person {name:$name})-[:KNOWS]->(friend:Person)   // anchor on indexed property
WHERE friend.age > $minAge
RETURN friend.name, friend.age ORDER BY friend.age DESC LIMIT 25;

MATCH (p:Person {name:$name})-[:KNOWS]->()-[:KNOWS]->(fof)  // friends-of-friends
RETURN DISTINCT fof.name;

MATCH (p:Person {name:$name})
OPTIONAL MATCH (p)-[:WORKS_AT]->(c:Company)                 // LEFT-JOIN equivalent
RETURN p.name, c.name;
```

- **Parameters** (`$name`) are mandatory (NEO-CYPHER-01) — they enable plan caching and make injection impossible.
- **`WHERE`** filters; supports the Neo4j 5 **type predicate** `WHERE p.age IS :: INTEGER` and `IS NOT NULL`.
- `WITH` pipes/aggregates between stages (the Cypher analogue of a subquery): `MATCH ... WITH c, count(p) AS n WHERE n > 10 RETURN ...`.

### B. Variable-length & path functions

```cypher
MATCH p = (a:Person {name:$a})-[:KNOWS*1..4]->(b:Person {name:$b})  // 1–4 hops, BOUNDED
RETURN p, length(p) AS hops ORDER BY hops LIMIT 1;

MATCH (a:Person {name:$a}), (b:Person {name:$b})
MATCH p = shortestPath((a)-[:KNOWS*..6]-(b))                        // built-in shortest path
RETURN p;
```

Always cap the upper bound. `shortestPath`/`allShortestPaths` are built in for unweighted hops; weighted/large-scale pathfinding belongs in GDS (§7).

### C. Write — CREATE / MERGE / SET / DELETE

```cypher
// CREATE always inserts new — use when you KNOW it doesn't exist
CREATE (p:Person {id: randomUUID(), name:$name});

// MERGE = match-or-create on a key. The key MUST be backed by a uniqueness constraint
// or concurrent runs create duplicates. ON CREATE/ON MATCH set the rest.
MERGE (p:Person {email:$email})
  ON CREATE SET p.createdAt = datetime(), p.name = $name
  ON MATCH  SET p.lastSeen  = datetime();

// MERGE a relationship between already-matched nodes (don't MERGE a full disconnected pattern)
MATCH (a:Person {email:$a}), (b:Person {email:$b})
MERGE (a)-[r:KNOWS]->(b) ON CREATE SET r.since = date();

MATCH (p:Person {email:$email}) SET p += $props;        // bulk property update (map merge)
MATCH (p:Person {email:$email}) DETACH DELETE p;        // delete node + its relationships
```

**MERGE semantics matter (NEO-MODEL-02):** `MERGE` matches the *entire* pattern or creates *all* of it. Merging on a non-unique key, or merging a multi-node pattern where only part exists, silently duplicates. Rule: MERGE each node on its constrained key first, then MERGE the relationship.

### D. Batch writes with UNWIND

```cypher
UNWIND $rows AS row                                     // $rows = list of maps from the driver
MERGE (p:Person {id: row.id}) SET p.name = row.name, p.email = row.email;
```

Send 1k–10k rows per transaction. For very large jobs use `CALL { ... } IN TRANSACTIONS OF 1000 ROWS` (Neo4j 5; replaces the deprecated `USING PERIODIC COMMIT`) or `apoc.periodic.iterate` (§7).

---

## 5. Indexing & Constraints

Indexes turn a `NodeByLabelScan` into a `NodeIndexSeek`; constraints guarantee integrity *and* create a backing index. Verify with `EXPLAIN` (plan only) / `PROFILE` (real db hits).

### A. Index types (Neo4j 5 syntax)

```cypher
CREATE RANGE INDEX person_age      FOR (p:Person)        ON (p.age);              // default: equality, range, ordering
CREATE RANGE INDEX person_name_idx FOR (p:Person)        ON (p.lastName, p.firstName);  // COMPOSITE (prefix-usable)
CREATE TEXT INDEX person_desc      FOR (p:Person)        ON (p.description);      // CONTAINS / ENDS WITH / STARTS WITH
CREATE POINT INDEX loc_idx         FOR (l:Location)      ON (l.coordinates);      // spatial: point.distance / bbox
CREATE FULLTEXT INDEX person_search FOR (p:Person)       ON EACH [p.name, p.bio]; // Lucene-backed, scored
CREATE RANGE INDEX rel_since       FOR ()-[r:KNOWS]-()   ON (r.since);            // relationship property index
SHOW INDEXES;   DROP INDEX person_age;
```

Pick by query: equality/range/ordering → **RANGE**; substring → **TEXT**; geo → **POINT**; natural-language/scored search → **FULLTEXT** (query via `CALL db.index.fulltext.queryNodes('person_search', $q) YIELD node, score`). A composite index serves queries on a prefix of its keys.

### B. Constraints (these create the backing index — prefer them on keys)

```cypher
CREATE CONSTRAINT person_email_unique FOR (p:Person) REQUIRE p.email IS UNIQUE;            // uniqueness (+ index)
CREATE CONSTRAINT person_key          FOR (p:Person) REQUIRE (p.firstName, p.lastName, p.dob) IS NODE KEY; // unique + exists
CREATE CONSTRAINT person_name_exists  FOR (p:Person) REQUIRE p.name IS NOT NULL;           // existence (Enterprise)
CREATE CONSTRAINT person_age_type     FOR (p:Person) REQUIRE p.age IS :: INTEGER;          // property type (Neo4j 5, Enterprise)
SHOW CONSTRAINTS;
```

A uniqueness constraint is what makes `MERGE` safe under concurrency (NEO-MODEL-02). Define constraints/indexes as versioned migration scripts, run them at deploy — not ad hoc.

---

## 6. Transactions

Neo4j is fully **ACID** (serializable isolation, WAL durability). Locks (shared read / exclusive write) are taken automatically; deadlocks are auto-detected and surface as a retryable `TransientError`.

**Use managed transaction functions** — the driver runs them, and **auto-retries transient errors** (deadlocks, leader switches in a cluster) with backoff (NEO-TXN-01, see [`error-handling.md`](guides://error-handling.md)). Don't hand-roll `begin/commit` for app code.

```python
from neo4j import GraphDatabase

with GraphDatabase.driver(uri, auth=(user, pwd)) as driver:        # one driver per app (pooled)
    def add_friend(tx, a, b):
        return tx.run(
            "MATCH (a:Person {email:$a}), (b:Person {email:$b}) "
            "MERGE (a)-[r:KNOWS]->(b) RETURN r", a=a, b=b
        ).single()

    with driver.session(database="neo4j") as session:
        session.execute_write(add_friend, "a@x.com", "b@x.com")    # retried on TransientError
        people = session.execute_read(
            lambda tx: tx.run("MATCH (p:Person) WHERE p.age > $n RETURN p.name", n=21).data()
        )
```

Keep transactions short and touching few nodes to limit lock contention. The same `execute_read`/`execute_write` pattern exists in the JavaScript (`executeRead`/`executeWrite`) and Java drivers. Routing (`neo4j://`) sends writes to the leader and reads to followers/secondaries automatically (§9).

---

## 7. Traversal, Path Algorithms & GDS

Built-in Cypher covers pattern traversal and `shortestPath` (§4.B). For graph *analytics* at scale, use the **Graph Data Science (GDS) library** — it projects an in-memory graph and runs optimized parallel algorithms.

```cypher
// 1. Project (Neo4j 5 / GDS 2.x: native projection)
CALL gds.graph.project('social', 'Person', {KNOWS: {properties: 'weight'}});

// 2a. Centrality — who is influential
CALL gds.pageRank.stream('social', {maxIterations: 20, dampingFactor: 0.85})
YIELD nodeId, score
RETURN gds.util.asNode(nodeId).name AS name, score ORDER BY score DESC LIMIT 10;

// 2b. Community detection
CALL gds.louvain.write('social', {writeProperty: 'community'}) YIELD communityCount, modularity;

// 2c. Weighted shortest path
MATCH (src:Person {name:$a})
CALL gds.shortestPath.dijkstra.stream('social', {sourceNode: src, relationshipWeightProperty: 'weight'})
YIELD targetNode, totalCost
RETURN gds.util.asNode(targetNode).name AS target, totalCost ORDER BY totalCost;

CALL gds.graph.drop('social');   // free the in-memory projection when done
```

Algorithm families: **centrality** (PageRank, betweenness, degree), **community** (Louvain, label propagation, triangle/WCC), **pathfinding** (Dijkstra, A*, Yen's k-shortest), **similarity** (node similarity, kNN), **embeddings & link prediction** (ML pipelines). Each runs in `.stream` (return rows), `.write` (persist a property), `.mutate` (write to the projection), or `.stats` mode. Use **elementId**, not the deprecated `id()`, when projecting via Cypher.

**APOC** complements GDS for utilities/ETL: `apoc.periodic.iterate` (batched writes), `apoc.load.json`/`apoc.load.jdbc` (import from APIs/RDBMS), `apoc.export.*`, `apoc.path.expandConfig` (configurable expansions), date/text/conversion helpers. Both APOC and GDS are plugins — allowlist them: `dbms.security.procedures.allowlist=apoc.*,gds.*`.

---

## 8. Performance & Observability

The query plan is the source of truth. `EXPLAIN` shows the plan without running; `PROFILE` runs it and reports rows + **db hits** per operator (see [`performance.md`](guides://performance.md)).

- **Index-backed start (NEO-IDX-01):** the leaf operator should be `NodeIndexSeek`/`NodeUniqueIndexSeek`/`NodeByLabelScan` on a *small* label — never `AllNodesScan`. Anchor on a constrained property.
- **Avoid cartesian products:** two `MATCH` patterns with no shared variable produce `CartesianProduct` (n×m). Connect them with a relationship or `WITH`.
- **The Eager pitfall:** an `Eager` operator forces the whole upstream result to materialize before the next clause (Cypher inserts it to protect read-before-write correctness). Harmless in reads, but in large `LOAD CSV`/batch writes it blows up memory and disables periodic commit — restructure (split MATCH and CREATE, use `CALL { } IN TRANSACTIONS`) to remove it.
- **Limit early; bound paths;** prune in `WHERE` before expanding; use `DISTINCT` only when variable-length paths can duplicate.
- **Config:** **page cache** (`server.memory.pagecache.size`) is the single biggest lever — size it to hold the working set (graph + indexes); set `server.memory.heap.max_size` separately. Cap runaway queries with `db.transaction.timeout`.
- **Observability (NEO-OBS-01, see `observability.md`):** enable the query log (`db.logs.query.enabled=true`, `db.logs.query.threshold=1s`, `parameter_logging_enabled=true`) and metrics (`server.metrics.enabled=true`, Prometheus/JMX) for latency, page-cache hit ratio, transaction/lock stats, and replication lag.

---

## 9. Security, Config & Clustering

### A. Security binding (NEO-SEC-01, owned by [`secure-coding.md`](guides://secure-coding.md))

- **AuthN:** enable auth; rotate the default `neo4j` password on first start. Native users, or LDAP/Active Directory / SSO (OIDC) in Enterprise.
- **AuthZ (RBAC):** least-privilege custom roles. Fine-grained graph privileges — grant/deny per label, relationship type, or **property**:
```cypher
CREATE ROLE analyst;
GRANT TRAVERSE ON GRAPH neo4j NODES Person, Company TO analyst;
GRANT READ {name, email} ON GRAPH neo4j NODES Person TO analyst;
DENY  READ {ssn, salary} ON GRAPH neo4j NODES Person TO analyst;   // hide sensitive props
GRANT ROLE analyst TO alice;
```
- **Injection:** Neo4j has no SQL, but concatenating user input into Cypher is still an injection vector. **Always pass `$parameters`** (NEO-CYPHER-01); never build label/relationship-type names from raw user input (those can't be parameterized — validate against an allowlist).
- **In transit:** TLS via the `neo4j+s://` / `bolt+s://` schemes (`server.bolt.tls_level=REQUIRED`).
- **At rest / network:** disk encryption at the OS/volume layer; bind to private interfaces and firewall Bolt (7687), HTTP(S) (7474/7473), and cluster ports.

### B. Configuration (NEO-CFG-01, see [`env-config.md`](guides://env-config.md))

The Bolt URI carries host, routing scheme, and credentials — load it from the environment/secret store, never hardcode (one pooled `Driver` per process):
```
NEO4J_URI=neo4j+s://my-instance.databases.neo4j.io:7687
NEO4J_USER=app
NEO4J_PASSWORD=${SECRET}
```
For local dev and integration tests, run the `neo4j:5-enterprise` Docker image / a [`docker-compose.md`](guides://docker-compose.md) stack, or Testcontainers.

### C. Clustering & causal consistency

Neo4j 5 uses **autonomous clustering**: a set of **servers** host databases as **primaries** (Raft-replicated, accept writes) and **secondaries** (async read scale-out). Run an odd number of primaries (≥3) so a majority survives a node loss.

- **Routing:** connect with the `neo4j://` (or `neo4j+s://`) scheme — the driver fetches the routing table and sends `execute_write` to a primary, `execute_read` to secondaries automatically. Use `bolt://` only for a single instance.
- **Causal consistency:** within a session, bookmarks make a read observe your own prior writes even across cluster members — keep related reads/writes in the **same session** to get read-your-writes.
- Inspect with `SHOW SERVERS` and `SHOW DATABASES`. Back up with `neo4j-admin database backup` (Enterprise, online) or `neo4j-admin database dump`/`load` (offline, all editions).

---

## 10. Quick Reference

```cypher
// model & write (parameterized, idempotent)
CREATE CONSTRAINT person_email FOR (p:Person) REQUIRE p.email IS UNIQUE;
MERGE (p:Person {email:$email}) ON CREATE SET p.name=$name;        // upsert on constrained key
UNWIND $rows AS r MERGE (p:Person {id:r.id}) SET p += r;            // batch

// read & traverse
MATCH (p:Person {email:$email})-[:KNOWS*1..3]->(fof) RETURN DISTINCT fof.name;  // bounded path
MATCH p = shortestPath((a:Person {email:$a})-[:KNOWS*..6]-(b:Person {email:$b})) RETURN p;

// index & plan
CREATE RANGE INDEX FOR (p:Person) ON (p.lastName, p.firstName);    // composite
EXPLAIN MATCH (p:Person {email:$e}) RETURN p;                      // want NodeUniqueIndexSeek
PROFILE <query>;                                                   // real db hits, no CartesianProduct/Eager

// analytics (GDS) & ops
CALL gds.graph.project('g','Person','KNOWS'); CALL gds.pageRank.write('g',{writeProperty:'rank'});
SHOW CONSTRAINTS; SHOW INDEXES; SHOW SERVERS; SHOW DATABASES;
```

---

## 11. Deployment Checklist

Generated from §2 — one box per requirement ID.

- [ ] NEO-MODEL-01 — entities are nodes, connections are relationships (no FK/list emulation)
- [ ] NEO-MODEL-02 — identity writes use `MERGE` on a constraint-backed key
- [ ] NEO-MODEL-03 — variable-length paths bounded; supernodes mitigated
- [ ] NEO-CYPHER-01 — all Cypher parameterized, never string-built (see `secure-coding.md`)
- [ ] NEO-IDX-01 — hot queries index-backed (`EXPLAIN` shows index seek)
- [ ] NEO-IDX-02 — natural keys have uniqueness / node-key constraints
- [ ] NEO-PERF-01 — queries profiled; no cartesian products / stray `Eager` (see `performance.md`)
- [ ] NEO-TXN-01 — transient errors retried via managed transaction functions (see `error-handling.md`)
- [ ] NEO-SEC-01 — auth on, default password rotated, least-privilege RBAC, TLS (see `secure-coding.md`)
- [ ] NEO-CFG-01 — URI & credentials from env/secret store (see `env-config.md`)
- [ ] NEO-OBS-01 — query log & metrics enabled (see `observability.md`)
- [ ] NEO-TST-01 — schema & queries tested against ephemeral Neo4j (see `tdd.md`)
- [ ] Agent ran the §2 verify commands and documented any fixes

---
**End of Neo4j Guidelines**
