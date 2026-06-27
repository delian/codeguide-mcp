# Elasticsearch & OpenSearch Development Guidelines
Mandatory standards for Elasticsearch/OpenSearch: text-vs-keyword mapping discipline, filter-context queries, right-sized shards, data streams + ILM, search_after pagination, kNN/semantic search. Elasticsearch 8.x, OpenSearch 2.x, Query DSL, ILM/ISM, dense_vector/kNN.

---
name: elasticsearch-opensearch
title: Elasticsearch & OpenSearch Development Guidelines
version: 2.0
last_reviewed: 2026-06-05
kind: datastore
tools: [elasticsearch@8.x, opensearch@2.x, query-dsl, ilm, ism, kibana, opensearch-dashboards]
requires:
  - secure-coding
  - error-handling
recommends:
  - observability
  - performance
  - rest
  - env-config
provides:
  - inverted-index
  - text-vs-keyword-mapping
  - query-dsl
  - aggregations
  - shard-design
  - knn-search
---

> 🧭 Authored per [`CONVENTIONS.md`](guides://CONVENTIONS.md): shared concerns are referenced, not restated. This guide covers only what is unique to Elasticsearch/OpenSearch.

> **Elasticsearch vs OpenSearch (fork history — be honest about it).** Both are distributed search/analytics engines on Apache Lucene. In 2021 Elastic relicensed Elasticsearch from Apache-2.0 to the dual SSPL/Elastic License; AWS forked the last Apache-2.0 release (7.10) into **OpenSearch** (Apache-2.0, Linux Foundation). They share heritage and most of the Query DSL but **have diverged** — they are *not* drop-in compatible at the version level. Differences to expect: security (Elastic X-Pack `xpack.security.*` vs OpenSearch Security plugin `plugins.security.*`), lifecycle (Elastic **ILM** vs OpenSearch **ISM**), vectors (Elastic `dense_vector`+`knn` query vs OpenSearch `knn_vector`+`knn` plugin), ESQL/transforms/runtime-fields (Elastic-only), and client libraries (`elasticsearch`/`@elastic/elasticsearch` deliberately reject OpenSearch servers — use `opensearch-py`/`@opensearch-project/opensearch` against OpenSearch). Elastic re-added an AGPL option in 2024, but version lines remain separate. **Pick one per cluster, pin the matching client, and test against the real target** — do not assume a query/setting valid on one works unchanged on the other.

---

## 0. Prerequisites & References

Fetch and apply these **before** generating Elasticsearch/OpenSearch code, mappings, or queries. Their rules are assumed below and not repeated.

> 📎 **REQUIRED — fetch & apply first:**
> - [`secure-coding.md`](guides://secure-coding.md) — secrets, network exposure, supply chain. *(ES/OS binding: security plugin on, TLS, RBAC, never-open-cluster — §11.)*
> - [`error-handling.md`](guides://error-handling.md) — error strategy, retries, backoff. *(ES/OS binding: bulk partial failures, 429 rejections, client retry — §9, §10.)*

> 📎 **RECOMMENDED — fetch when the task touches them:**
> - [`rest.md`](guides://rest.md) — it is a REST/JSON-over-HTTP API; resource/verb/status-code discipline applies to every call.
> - [`observability.md`](guides://observability.md) — metrics/tracing policy *(binding: ES/OS is itself the storage/search tier for logs & APM; monitor cluster health, slow logs — §13)*.
> - [`performance.md`](guides://performance.md) — perf methodology *(binding: sharding, refresh, filter cache, query profiling — §12)*.
> - [`env-config.md`](guides://env-config.md) — config & connection policy *(binding: cluster URL/credentials/TLS from secrets, never hardcoded)*.

> 📎 **SEE ALSO:** [`postgresql.md`](guides://postgresql.md) *(system-of-record behind a search index)* · [`kafka.md`](guides://kafka.md) *(ingest pipeline)* · [`docker-compose.md`](guides://docker-compose.md) *(local single-node cluster for tests)* · [`chroma-vectordb.md`](guides://chroma-vectordb.md) *(dedicated vector store alternative)*.

---

## 1. Core Philosophies: SEARCH-FIRST

Elasticsearch/OpenSearch-specific principles only. Security, error handling, observability, and performance methodology come from §0.

- **S**chema is a decision, not an accident: every production index gets an **explicit mapping** with `dynamic: strict`; getting `text` vs `keyword` wrong is the single most expensive mistake (§4) and a reindex to fix.
- **E**xploit the inverted index: model fields for how they are queried — analyzed `text` for relevance, `keyword` for exact/aggregate/sort, `dense_vector` for semantic (§3, §4, §10).
- **A**nalytics via aggregations: push bucketing/metrics into the engine (§7); avoid `script`/painless on the hot path.
- **R**elevance is engineered: understand BM25, query-vs-filter context, and boosting (§6); put non-scoring predicates in **filter context** so they cache and skip scoring.
- **C**luster economics: right-size shards (§8) — oversharding is the classic footgun; use **data streams + ILM/ISM** for time-series/logs, not hand-rolled daily indices.
- **H**TTP/JSON discipline: it's a REST API (`rest.md`); batch writes with the **bulk API**, paginate deep results with **search_after**, never `from`/`size` into the thousands.

**Secondary index, not source of truth.** Default posture: ES/OS is a *derived search/analytics layer*; the durable system of record (e.g. PostgreSQL) lives elsewhere and the index is rebuildable by reindex. It has **no multi-document ACID transactions** and is eventually consistent (refresh-bounded). Using it as a primary store is a deliberate, documented exception.

**Verified Code**: Agent-generated mappings/queries/config MUST pass every gate in §2 before delivery.

---

## 2. Requirements (MANDATORY, auditable)

RFC-2119 keywords. IDs `ES-<TOPIC>-<NN>`. Each row has a binary gate; rows binding a shared rule cite its owner.

| ID | Requirement | Verify | Gate |
|----|-------------|--------|------|
| ES-MAP-01 | Production indices MUST have an explicit mapping with `dynamic: strict` (no mapping explosion) | `GET /<index>/_mapping` | explicit + strict |
| ES-MAP-02 | String fields MUST be modelled deliberately: `text` for full-text, `keyword` for exact/aggregate/sort, multi-field when both are needed | review mapping vs query usage | every string justified |
| ES-IDX-01 | Shards MUST be right-sized (target ~10–50 GB/shard); no oversharding | `GET /_cat/shards?v` | within target, no tiny shards |
| ES-IDX-02 | Time-series/log indices MUST use data streams + ILM (ES) / ISM (OS), not unmanaged indices | `GET /_data_stream`, lifecycle policy exists | managed lifecycle |
| ES-QRY-01 | Non-scoring predicates MUST run in **filter context** (cacheable, no scoring) | review queries / `_search?profile=true` | filters not in `must` |
| ES-QRY-02 | No leading-wildcard, unbounded `regexp`, or `script` queries on the hot path | review / slow-query log | none on hot path |
| ES-PAGE-01 | Deep pagination MUST use `search_after`+PIT (not `from`/`size` beyond a few pages) | review / reject `from`>1000 | search_after used |
| ES-BULK-01 | Batch writes MUST use the bulk API AND inspect per-item `errors` (see `error-handling.md`) | code review | `errors` checked per item |
| ES-KNN-01 | Vector search MUST use a vector field (`dense_vector`/`knn_vector`) with declared dims + similarity, not brute-force scripts | `GET /<index>/_mapping` | vector field + ANN index |
| ES-SEC-01 | Security plugin ON, TLS on transport+HTTP, RBAC least-privilege, cluster not internet-exposed (see `secure-coding.md`) | `GET /_cluster/health` over TLS w/ auth; `GET /_security/role` | auth required, TLS on |
| ES-SEC-02 | Cluster URL/credentials/CA from config/secrets, never hardcoded (see `env-config.md`,`secure-coding.md`) | grep source/config | no literals |
| ES-ERR-01 | Client MUST retry with backoff and handle 429/bulk rejections; bounded timeouts (see `error-handling.md`) | review / fault test | retries + backoff present |
| ES-OBS-01 | Cluster health, slow logs, and key node stats MUST be monitored (see `observability.md`) | `GET /_cluster/health`; slowlog settings | green/yellow tracked, slowlog on |

> **Forbidden**: indexing into an index without an explicit mapping in production; using `text` where you need exact-match/aggregation/sort (or `keyword` where you need full-text); deep `from`/`size` paging; firing N single-document index calls instead of bulk; ignoring the `errors` flag on a bulk response; brute-force vector scoring via painless instead of an ANN field; a cluster reachable from the public internet or with security disabled.

---

## 3. The engine model — inverted index & Lucene

ES/OS are distributed coordinators over per-shard **Apache Lucene** indices. The core structure is the **inverted index**: for analyzed text, each *term* maps to a posting list of the documents containing it, enabling sub-second full-text lookups over billions of docs with relevance scoring. Aggregations and sorts instead read **doc values** (a columnar, on-disk per-field store) — which is *why* exact/aggregate/sort fields want `keyword`, not analyzed `text`.

- **Cluster → node → index → shard → segment.** An index is split into **primary shards** (fixed at creation — you cannot change the count without reindex) each with N **replicas** (changeable live; provide HA + read throughput). A shard is a self-contained Lucene index of immutable **segments**.
- **Documents** are JSON, addressed by `_id`, grouped in an index governed by a **mapping** (the schema). Routing (`hash(_routing) % num_primary_shards`) places a doc on a shard.
- **REST/JSON over HTTP** (`rest.md`): `PUT /<index>`, `POST /<index>/_doc`, `GET /<index>/_search`. Honour status codes (404 missing, 409 version conflict, 429 rejected/backpressure).

---

## 4. Mappings & field types — text vs keyword is THE decision

The mapping is the schema. A wrong type is not editable in place — it costs a **reindex**. Define mappings explicitly; set `dynamic: strict` so unexpected fields are rejected rather than silently (mis)typed and exploding the field count (ES-MAP-01).

### A. `text` vs `keyword` — never confuse them (ES-MAP-02)
| | `text` | `keyword` |
|---|---|---|
| Analyzed? | yes — tokenized, lowercased, stemmed | no — stored verbatim, one term |
| Built for | full-text relevance (`match`) | exact match, aggregations, sorting, `term` |
| Query | `match`, `match_phrase`, `multi_match` | `term`, `terms`, prefix, sort, aggregate |
| Aggregatable/sortable | no (needs costly `fielddata`) | yes (doc values) |

Need both on one field (search the prose *and* aggregate the exact value)? Use a **multi-field**:
```json
PUT /products
{ "mappings": { "dynamic": "strict", "properties": {
  "name":     { "type": "text", "analyzer": "english",
                "fields": { "raw": { "type": "keyword", "ignore_above": 256 } } },
  "category": { "type": "keyword" },
  "price":    { "type": "scaled_float", "scaling_factor": 100 },
  "created":  { "type": "date" },
  "location": { "type": "geo_point" },
  "tags":     { "type": "keyword" },
  "embedding":{ "type": "dense_vector", "dims": 768, "index": true, "similarity": "cosine" }
}}}
```
`name` → relevance via `match name`; `name.raw` → `terms`/sort. Other key types: numeric (`integer`/`long`/`float`/`double`/`scaled_float` — prefer `scaled_float` for money), `boolean`, `date`, `ip`, `geo_point`/`geo_shape`, `object` (flattened) vs **`nested`** (each array element queried independently — required when array-element fields must match together).

### B. Dynamic mapping — discipline
`dynamic: true` (default) auto-detects types: convenient in dev, dangerous in prod (a stray string field maps as `text`+`keyword`, a date-looking string maps wrong, field count explodes). For production: `dynamic: "strict"` (reject unknowns) or `dynamic: "false"` (store, don't index). Use **dynamic templates** only for known-shape dynamic data (e.g. map all unknown strings to `keyword`).

### C. Index templates & aliases
Use an **index template** (`PUT /_index_template/...` matching `index_patterns`) so every new backing index inherits settings/mappings — essential for data streams. Read/write through an **alias** (or data-stream name), never a raw index, so you can **zero-downtime reindex**: build `v2`, `_reindex` from `v1`, then atomically swap the alias in one `_aliases` action.

---

## 5. Analysis pipeline — analyzers & tokenizers

Indexing a `text` field runs the **analysis chain**: *char filters → tokenizer → token filters* → terms stored in the inverted index. The **same analyzer must apply at index and query time** or matches silently fail.

- **Tokenizer** splits text into tokens (`standard`, `whitespace`, `keyword`, `pattern`, `ngram`/`edge_ngram` for autocomplete).
- **Token filters** transform tokens (`lowercase`, `stop`, `stemmer`/`snowball`, `synonym`, `asciifolding`).
- **Built-in analyzers**: `standard` (default), language analyzers (`english` — stemming + stop words), `keyword` (no-op). Define a **custom analyzer** in index `settings.analysis` and reference it from the field.
```json
"settings": { "analysis": { "analyzer": { "en_search": {
  "type": "custom", "tokenizer": "standard",
  "filter": ["lowercase", "asciifolding", "english_stop", "english_stemmer"] }}}}
```
Test before committing with `POST /<index>/_analyze` `{ "analyzer": "en_search", "text": "Running Shoes" }`. Autocomplete: prefer `search_as_you_type` or `edge_ngram` index-side + a non-ngram **search analyzer** (set `search_analyzer` separately to avoid expanding the query). Changing an analyzer requires reindexing existing docs.

---

## 6. Query DSL — match vs term, context, relevance

Two query families and **two contexts** — conflating them is the most common correctness/perf bug.

- **Full-text (`match`, `multi_match`, `match_phrase`)** runs the field's analyzer on the query → matches **terms**. Use on `text`.
- **Term-level (`term`, `terms`, `range`, `prefix`, `exists`)** is **not** analyzed → matches the exact stored term. Use on `keyword`/numeric/date. (`term` on a `text` field is a classic zero-hits bug — the stored term is lowercased/stemmed but your `term` value isn't.)

### Query context vs filter context (ES-QRY-01)
Inside `bool`:
- **`must` / `should`** = *query context* → compute a relevance `_score`.
- **`filter` / `must_not`** = *filter context* → yes/no only, **no scoring, and results are cached** in the node filter cache. Put every non-scoring predicate (exact category, range, boolean, geo) in `filter` — it is faster and cacheable.
```json
{ "query": { "bool": {
  "must":   [ { "match": { "name": "wireless headphones" } } ],     // scored
  "filter": [ { "term":  { "category": "audio" } },                  // cached, unscored
              { "range": { "price": { "lte": 200 } } },
              { "term":  { "in_stock": true } } ],
  "should": [ { "match": { "brand": "Sony" } } ],                    // boosts score
  "minimum_should_match": 0 }}}
```

### Relevance & boosting
Default scoring is **BM25** (term frequency saturated, length-normalized, rarer terms weighted higher via IDF). Tune relevance with field boosts (`"fields": ["title^3", "body"]`), `boost` on clauses, `function_score`/`rank_feature` for business signals, and `match_phrase`+`slop` for proximity. Profile a slow/ wrong-scoring query with `_search?profile=true` or `_explain`. Keep heavy `script`/`script_score` and leading-wildcard/unbounded `regexp` off the hot path (ES-QRY-02).

---

## 7. Aggregations — the analytics powerhouse

Aggregations run server-side over the matched set (apply `"size": 0` to skip hits when you only want analytics). Two families compose into trees:
- **Bucket** aggs group docs: `terms` (top values), `date_histogram` (time series), `histogram`, `range`, `filters`, `nested`, `composite` (paginated, exhaustive bucketing for exports).
- **Metric** aggs compute over a bucket: `avg`/`sum`/`min`/`max`/`stats`, `cardinality` (approx distinct, HyperLogLog), `percentiles`, `top_hits`.
```json
{ "size": 0, "query": { "bool": { "filter": [ { "range": { "created": { "gte": "now-7d" } } } ] } },
  "aggs": { "per_day": { "date_histogram": { "field": "created", "calendar_interval": "day" },
    "aggs": { "revenue": { "sum": { "field": "price" } },
              "buyers":  { "cardinality": { "field": "user_id" } } } } } }
```
Aggregate on `keyword`/numeric/date (doc values), never analyzed `text`. `terms` aggs on very high-cardinality fields are memory-heavy — bound with `size`, or use `composite` to paginate. Prefer aggregations over pulling raw docs to the client to count.

---

## 8. Index design — shards, replicas, ILM, data streams

Shard sizing is the defining operational decision.

- **Primary count is immutable** (changing it = reindex); replicas are live-adjustable. Target **~10–50 GB per shard**; too-small shards waste heap on overhead (the **oversharding** footgun: thousands of tiny shards exhaust master/heap), too-large shards slow recovery/rebalance (ES-IDX-01). For a fixed corpus, fewer larger shards; for growth, plan with rollover.
- **Replicas** (default 1) provide HA and add read throughput; 0 replicas risks data loss on node failure — set ≥1 in production.
- **Data streams (ES) / rollover aliases (OS)** are the right model for append-only time-series (logs, metrics, events): one write alias over auto-rolled backing indices named by generation. Combine with **ILM (Elastic) / ISM (OpenSearch)** to move indices through hot → warm → cold → frozen tiers and delete on age/size — never hand-manage `logs-2026.06.27` indices.
```json
PUT /_index_template/logs            // ES: data stream + ILM
{ "index_patterns": ["logs-*"], "data_stream": {},
  "template": { "settings": { "number_of_shards": 1, "number_of_replicas": 1,
                              "index.lifecycle.name": "logs-policy" } } }
```
ILM policy: rollover at e.g. 50 GB / 1 day, warm after 7d (fewer replicas, force-merge), delete after retention. OpenSearch ISM expresses the same as a state-machine policy. Use `_cat/shards`, `_cat/indices` to audit.

---

## 9. Read/write path, refresh & near-real-time

- **Write path:** doc → primary shard → translog (durability) + in-memory buffer → replicated to replicas. A **refresh** (default every `1s`, `index.refresh_interval`) makes new docs *searchable* by opening a new segment — this is the **near-real-time** delay (a just-indexed doc isn't instantly visible). A **flush** fsyncs segments and trims the translog.
- **Tuning:** for high-throughput bulk loads, raise/disable `refresh_interval` (e.g. `-1`) and restore it after — refreshing too often kills indexing throughput. For read-your-write needs, index with `?refresh=wait_for` (not `refresh=true`, which forces an expensive refresh). The common "search returns 0 hits right after indexing" bug is a missing refresh, not a lost write.
- **Updates** are read-delete-reindex of the whole doc (segments are immutable) — ES/OS is append-optimized; very high update rates are an anti-pattern. Use optimistic concurrency (`if_seq_no`/`if_primary_term`, or `version`) to avoid lost updates (409 on conflict).
- **Bulk API (ES-BULK-01):** batch index/update/delete in one request (NDJSON action+source lines). Bulk responses are **HTTP 200 even when individual items fail** — you MUST inspect `response.errors` and each item's status; route permanent failures to a DLQ, retry only the rejected (429) items. Tune batch size to a few MB / a few thousand docs, not "all of them". Error/retry policy is owned by [`error-handling.md`](guides://error-handling.md).

---

## 10. Vector / kNN / semantic search

Model embeddings as a vector field and use the engine's **approximate nearest-neighbor (ANN, HNSW)** index — never brute-force cosine in a painless `script_score` over all docs (ES-KNN-01).

- **Elasticsearch 8.x:** `dense_vector` with `index: true`, `dims`, `similarity` (`cosine`/`dot_product`/`l2_norm`); query with the top-level `knn` clause (or `knn` inside `_search`), combine with filters and lexical queries; **hybrid search** via reciprocal-rank-fusion (`rank: { rrf: {} }`) or `sub_searches` blends BM25 + vector. Elastic also offers ELSER (sparse `text_expansion`) and `semantic_text` for managed embeddings.
- **OpenSearch 2.x:** the **k-NN plugin** with `knn_vector` (engines: Lucene HNSW / `nmslib` / `faiss`), `space_type`, queried via the `knn` query; neural-search plugin + ml-commons for managed embeddings/hybrid pipelines.
- **Common discipline:** declare dims/similarity up front (immutable); pre-filter to shrink the candidate set; vectors are RAM-hungry (HNSW graphs live in memory) — size nodes accordingly; consider quantization (`int8`/`bbq`) for large corpora. For a pure/very-large vector workload a dedicated store may fit better — see [`chroma-vectordb.md`](guides://chroma-vectordb.md).

---

## 11. Pagination (ES-PAGE-01)

- **`from`/`size`** is fine only for the first few shallow pages. It is **O(from+size) per shard** (each shard returns `from+size` to the coordinator) and is hard-capped by `index.max_result_window` (10 000) — deep `from` is a cluster-killer.
- **`search_after`** is the correct deep/infinite pagination: sort by a unique tiebreaker (e.g. `[ {"created":"desc"}, {"_shard_doc":"asc"} ]`), pass the last hit's `sort` values as `search_after` on the next request. Pair with a **Point-in-Time** (`_pit`, ES) / PIT (OS) for a consistent snapshot across pages.
- **`_search/scroll`** is legacy for one-shot exports — prefer `search_after`+PIT; reserve scroll for backward compatibility. For exhaustive aggregation export use a **`composite`** aggregation.

---

## 12. Performance binding (for `performance.md`)

Methodology owned by [`performance.md`](guides://performance.md). ES/OS specifics:
- **Filter, don't score:** non-scoring predicates in filter context cache and skip BM25 (ES-QRY-01); reuse stable filters so the node query cache hits.
- **Right-size shards** (§8, ES-IDX-01); force-merge read-only (warm/cold) indices to one segment; avoid heap pressure from huge `terms` aggs and `fielddata` on `text`.
- **Bulk + tuned refresh** for ingest (§9): batch writes, raise `refresh_interval` during loads, increase replicas only after the load.
- **`_source` filtering / `docvalue_fields`:** return only needed fields; disable `_source` only if you never need the original doc.
- **Profile** slow queries with `_search?profile=true`, `_search/_explain`, and the **slow log** (`index.search.slowlog.*`, `index.indexing.slowlog.*`). Watch leading wildcards, regexp, scripts, and deep pagination (ES-QRY-02, ES-PAGE-01).

---

## 13. Security binding (for `secure-coding.md`)

Policy owned by [`secure-coding.md`](guides://secure-coding.md). ES/OS hardening (ES-SEC-01/02) — the historical, repeated cause of mass data leaks is **an unsecured cluster bound to a public interface**:
- **Never expose the cluster to the internet.** `network.host` to private interfaces, firewall to app hosts; put a gateway/VPC boundary in front. Default-deny.
- **Security plugin ON** (it is the default and MUST stay on): Elastic `xpack.security.enabled: true`; OpenSearch Security plugin enabled. Disabling it in prod is forbidden.
- **TLS on both layers** — transport (node-to-node) *and* HTTP (client-to-node); verify certs.
- **RBAC, least privilege:** scoped roles per service (index patterns + actions); **field-level** (FLS) and **document-level** (DLS) security for sensitive data; prefer time-bounded **API keys** over shared user passwords.
- **Audit logging** to a separate index/SIEM; alert on auth failures.
- Cluster URL, credentials, and CA come from secrets/config (see [`env-config.md`](guides://env-config.md)) — never literals in source or committed YAML.

---

## 14. Error-handling binding (for `error-handling.md`)

Policy owned by [`error-handling.md`](guides://error-handling.md). ES/OS bindings (ES-ERR-01, ES-BULK-01):
- **Bulk partial failure:** a bulk request returns 200 with `errors: true` and per-item status — inspect every item, retry only rejected (429) ones with backoff, DLQ permanent mapping/parse errors. Never assume bulk success from the HTTP code.
- **429 / backpressure:** the cluster rejects writes/searches under load — retry with exponential backoff + jitter, cap in-flight bulk size; do not hammer.
- **Version conflicts (409):** optimistic concurrency mismatch — reread and retry or surface a conflict, don't blind-overwrite.
- **Timeouts & failover:** bounded `request_timeout`, sniff/round-robin across nodes, retry on connection errors. Official clients (`elasticsearch`/`opensearch-py` and JS equivalents) have built-in retry/backoff — enable it; pin the client to the matching server.

---

## 15. Observability binding (for `observability.md`)

Policy owned by [`observability.md`](guides://observability.md). Note ES/OS is frequently *itself* the storage/search tier for logs and APM — keep that cluster's own monitoring separate. Signals to scrape/alert (ES-OBS-01):
- **Cluster health** `GET /_cluster/health` (status green/yellow/red, unassigned shards, pending tasks); **`_cat/`** APIs (`_cat/shards`, `_cat/nodes`, `_cat/thread_pool`).
- **Node stats** `GET /_nodes/stats`: JVM heap pressure & GC, indexing/search rate & latency, **rejected** thread-pool counts (write/search), disk watermarks, segment/merge load, field-data/query-cache.
- **Slow logs** for search and indexing (ES-OBS-01) to catch heavy queries/mappings.
- Export via Prometheus exporter / Elastic Stack monitoring / OpenSearch monitoring; alert on red status, unassigned shards, heap >75%, write rejections, low disk watermark.

---

## 16. Anti-patterns

- No explicit mapping / `dynamic: true` in production → mapping explosion and mistyped fields.
- `text` where you need exact-match/aggregate/sort (or `term` on a `text` field → zero hits); forgetting the multi-field.
- Non-scoring predicates in `must` instead of `filter` (no cache, wasted scoring).
- Deep `from`/`size` pagination; scroll for live pagination instead of `search_after`+PIT.
- N single-document index calls instead of bulk; ignoring the bulk `errors` flag.
- Hand-managed daily log indices instead of data streams + ILM/ISM; oversharding (thousands of tiny shards) or one giant shard.
- 0 replicas in production; treating ES/OS as the only copy of critical data.
- Searching immediately after indexing without refresh and calling it a lost write.
- Brute-force vector scoring in painless instead of an ANN `dense_vector`/`knn_vector` field.
- Cluster on a public interface / security disabled; credentials hardcoded.
- Assuming an Elasticsearch query/setting works unchanged on OpenSearch (or wrong client against the server).

---

## 17. Quick Reference

```bash
# cluster & shards
curl -s localhost:9200/_cluster/health?pretty
curl -s 'localhost:9200/_cat/indices?v'  ;  curl -s 'localhost:9200/_cat/shards?v'
curl -s 'localhost:9200/_cat/nodes?v&h=name,heap.percent,ram.percent,cpu'
# mapping & analysis
curl -s localhost:9200/products/_mapping?pretty
curl -s -XPOST localhost:9200/products/_analyze -H 'Content-Type: application/json' \
  -d '{"analyzer":"english","text":"Running Shoes"}'
# query profiling
curl -s 'localhost:9200/products/_search?profile=true' -H 'Content-Type: application/json' \
  -d '{"query":{"bool":{"filter":[{"term":{"category":"audio"}}]}}}'
# lifecycle (ES ILM / OS ISM)
curl -s localhost:9200/_data_stream?pretty            # ES data streams
curl -s localhost:9200/_ilm/policy?pretty             # ES
```

---

## 18. Deployment Checklist

Generated from §2 — one box per requirement ID.

- [ ] ES-MAP-01 — production indices have explicit mapping, `dynamic: strict`
- [ ] ES-MAP-02 — `text` vs `keyword` chosen deliberately; multi-fields where both needed
- [ ] ES-IDX-01 — shards right-sized (~10–50 GB), no oversharding
- [ ] ES-IDX-02 — time-series/logs on data streams + ILM/ISM
- [ ] ES-QRY-01 — non-scoring predicates in filter context (cached)
- [ ] ES-QRY-02 — no leading-wildcard/unbounded-regexp/script on hot path
- [ ] ES-PAGE-01 — deep pagination via `search_after`+PIT, not `from`/`size`
- [ ] ES-BULK-01 — bulk API used; per-item `errors` inspected (see `error-handling.md`)
- [ ] ES-KNN-01 — vector search via ANN `dense_vector`/`knn_vector`, not scripts
- [ ] ES-SEC-01 — security on, TLS both layers, RBAC least-privilege, not internet-exposed (see `secure-coding.md`)
- [ ] ES-SEC-02 — cluster URL/credentials/CA from secrets (see `env-config.md`)
- [ ] ES-ERR-01 — client retries with backoff; 429/bulk rejections handled (see `error-handling.md`)
- [ ] ES-OBS-01 — cluster health, slow logs, node stats monitored (see `observability.md`)
- [ ] Agent ran the §17 inspection commands and documented any fixes

---
**End of Elasticsearch & OpenSearch Guidelines**
