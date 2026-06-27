# Chroma Vector Database Guidelines
Mandatory standards for building embedding/RAG systems on Chroma: consistent embeddings, tuned ANN indexes, metadata-aware retrieval. Chroma 0.5.x/1.0, Python/JS clients, HNSW; pgvector/Qdrant/Weaviate/Milvus for scale.

---
name: chroma-vectordb
title: Chroma Vector Database Guidelines
version: 2.0
last_reviewed: 2026-06-05
kind: datastore
tools: [chromadb@1.0, sentence-transformers, hnswlib, openai-embeddings@v3, pip-audit]
requires:
  - secure-coding
  - error-handling
recommends:
  - python
  - mlops
  - performance
  - env-config
provides:
  - vector-search
  - embeddings
  - ann-hnsw
  - rag-patterns
  - metadata-filtering
---

> 🧭 Authored per [`CONVENTIONS.md`](guides://CONVENTIONS.md): shared concerns are referenced, not restated. This guide covers only what is unique to vector databases and Chroma.

---

## 0. Prerequisites & References

Fetch and apply these **before** generating Chroma code. Their rules are assumed below and not repeated.

> 📎 **REQUIRED — fetch & apply first:**
> - [`secure-coding.md`](guides://secure-coding.md) — secrets, supply-chain, CVE policy. *(Binding: embedding-API keys via env/secrets manager; `pip-audit`/`npm audit` on `chromadb`; PII-in-vectors handling; auth on server mode.)*
> - [`error-handling.md`](guides://error-handling.md) — error strategy & propagation. *(Binding: embedding-API timeouts/retries, dimension-mismatch and empty-result handling.)*

> 📎 **RECOMMENDED — fetch when the task touches them:**
> - [`python.md`](guides://python.md) — the primary client language: `uv`, typing, `pytest`, packaging.
> - [`mlops.md`](guides://mlops.md) — embedding-model lifecycle, versioning, retrieval **evaluation** (recall@k / nDCG), reproducibility.
> - [`performance.md`](guides://performance.md) — latency/throughput method behind HNSW recall tuning.
> - [`env-config.md`](guides://env-config.md) — config layering for paths, hosts, model names.

> 📎 **SEE ALSO:** [`postgresql.md`](guides://postgresql.md) *(pgvector alternative)* · [`rest.md`](guides://rest.md) *(server-mode API)*

---

## 1. Core Philosophies: VECTOR-FIRST

Vector-DB-specific principles only. TDD, security, error handling, and performance method come from §0.

- **V**ector-native data model: embeddings are high-dimensional vectors; retrieval is **approximate nearest-neighbor (ANN)** by distance, not keyword match. Design for similarity, not `WHERE text LIKE`.
- **E**mbedding consistency: the **same model + version + preprocessing** indexes and queries a collection. A mismatch silently returns garbage — there is no error.
- **C**hunk before you embed: retrieval quality is dominated by the upstream chunking decision; the index can only rank what you stored.
- **T**une the accuracy/speed tradeoff explicitly: HNSW `M` / `ef_construction` / `ef_search` and the distance metric are deliberate choices measured against a recall target.
- **O**bservable retrieval: validate with a labeled eval set (recall@k / nDCG, see `mlops.md`), not vibes.
- **R**ight tool for the scale: Chroma for local/prototyping/small-to-mid; graduate to pgvector/Qdrant/Weaviate/Milvus past that (see §9).

**Verified Code**: Agent-generated code MUST pass every gate in §2 before delivery.

---

## 2. Requirements (MANDATORY, auditable)

RFC-2119 keywords. IDs `CHROMA-<TOPIC>-<NN>`. Each row has a binary gate; rows binding a shared rule cite its owner.

| ID | Requirement | Verify | Gate |
|----|-------------|--------|------|
| CHROMA-TST-01 | Retrieval logic MUST be test-first with ephemeral `chromadb.Client()` (see `tdd.md` via `python.md`) | `uv run pytest` | exit 0, 0 skips |
| CHROMA-TST-02 | Each retrieval bug MUST get a regression test before the fix (see `tdd.md`) | `uv run pytest` | failing→passing |
| CHROMA-EMB-01 | Embedding model name+version MUST be pinned and identical for index & query | grep config / review | one pinned model per collection |
| CHROMA-EMB-02 | Query embedding dimensionality MUST match the collection's | integration test | no dimension-mismatch error |
| CHROMA-DIST-01 | Distance metric MUST match the embedding model's training objective (cosine for most text) | review `hnsw:space` | metric justified |
| CHROMA-IDX-01 | HNSW params MUST be set deliberately and measured against a recall target (see `performance.md`) | recall@k eval | recall ≥ target |
| CHROMA-EVAL-01 | Retrieval quality MUST be measured on a labeled set, not assumed (see `mlops.md`) | eval script | recall@k/nDCG recorded |
| CHROMA-SEC-01 | Embedding-API keys & server tokens MUST come from env/secrets, never source (see `secure-coding.md`) | `grep -ri "sk-\|api_key=" src/` | 0 hits |
| CHROMA-SEC-02 | Server mode MUST be authenticated and network-restricted (see `secure-coding.md`) | config/network review | not public + unauth |
| CHROMA-SEC-03 | 0 known CVEs in `chromadb` & deps (see `secure-coding.md`) | `uv run pip-audit` | 0 vulnerabilities |
| CHROMA-ERR-01 | Embedding-API calls MUST handle timeout/rate-limit/retry; empty results handled (see `error-handling.md`) | review / test | no unguarded calls |
| CHROMA-PERSIST-01 | Production data MUST use a persisted client/path with a tested backup+restore | restore drill | data recovered |

> **Forbidden**: querying a collection with a different embedding model than it was built with; hardcoding API keys; exposing an unauthenticated Chroma server to the internet; shipping retrieval changes without a recall eval; using ephemeral `Client()` for production data.

---

## 3. Verification Protocol

Run before presenting code. Fix → re-run until green. (Python toolchain owned by [`python.md`](guides://python.md).)

```bash
uv run pytest                 # CHROMA-TST-01/02  (ephemeral client unit tests)
uv run python eval/recall.py  # CHROMA-IDX-01/EVAL-01  (recall@k vs target)
grep -ri "api_key=\|sk-" src/ # CHROMA-SEC-01  (expect no hits)
uv run pip-audit              # CHROMA-SEC-03
```

The *why* behind each gate lives in its §0 owner; do not re-derive it here.

---

## 4. The Vector Database Model

A vector DB stores **embeddings** — dense float vectors (typically 384–3072 dims) produced by a model that maps semantically similar inputs to nearby points. Retrieval finds the **k nearest neighbors** of a query vector by a distance metric. This powers **semantic search**, **RAG** (feeding retrieved context to an LLM), and **recommendations** (item-to-item similarity).

Chroma's core objects:

- **Collection** — a named set of vectors sharing one embedding function, distance metric, and HNSW index. One collection per *(embedding model + domain + version)*; naming e.g. `docs_minilm_v2`.
- **Document** — text (or any input) stored with a unique **id**, optional **metadata**, and its **embedding** (auto-computed on `add`, or supplied pre-computed).
- **Embedding function** — converts inputs → vectors. Built-in: default `all-MiniLM-L6-v2` (Sentence Transformers, 384d), OpenAI `text-embedding-3-{small,large}`, Cohere, HuggingFace; or a custom `EmbeddingFunction.__call__`. **It is part of the collection's identity** — changing it invalidates the index.

```python
import chromadb
client = chromadb.PersistentClient(path="./chroma_data")   # persistent (prod)
col = client.get_or_create_collection(
    name="docs_minilm_v2",
    metadata={"hnsw:space": "cosine"},   # metric is fixed at creation
)
col.add(ids=["d1"], documents=["The cat sat on the mat"],
        metadatas=[{"source": "wiki"}])
res = col.query(query_texts=["where did the cat sit?"], n_results=5)
# res["ids"], res["documents"], res["distances"] (lower = closer), res["metadatas"]
```

Client modes: `chromadb.Client()` (ephemeral, in-memory — **tests only**); `PersistentClient(path=...)` (embedded + on-disk, single process); `HttpClient(host, port, headers=...)` (client/server, shared/scaled). JS/TS: `new ChromaClient({ path })` then `getOrCreateCollection` — same model, async API.

> Idempotent setup uses `get_or_create_collection`. Mutations: `add` (insert), `update` (existing only), `upsert` (insert-or-update), `delete(ids=... | where=...)`. Using `add` where `update` was meant creates duplicates — a classic regression-test target.

---

## 5. ANN Indexing & the Accuracy/Speed Tradeoff

Chroma indexes vectors with **HNSW** (Hierarchical Navigable Small World) — a graph giving sub-linear *approximate* search. Approximate means **recall < 100%**: you trade exactness for speed. Brute-force (exact) is only viable for tiny collections. Tune three parameters and **measure recall** (see [`performance.md`](guides://performance.md) for the method):

| Param (`metadata` key) | Phase | Higher value → | Default |
|---|---|---|---|
| `hnsw:M` | build | better recall, more memory, slower build | 16 |
| `hnsw:construction_ef` | build | better graph quality, slower indexing | 100 |
| `hnsw:search_ef` | query | better recall, slower queries | 10 |

```python
col = client.create_collection(
    name="tuned",
    metadata={
        "hnsw:space": "cosine",
        "hnsw:M": 32,                 # 8–16 speed/memory · 32–64 high recall
        "hnsw:construction_ef": 200,  # raise for index quality
        "hnsw:search_ef": 100,        # raise until recall@k hits target
    },
)
```

Tuning loop: fix `M` and `construction_ef` at build; sweep `search_ef` at query time against a labeled eval set until recall@k meets target at acceptable p95 latency. `M`/`construction_ef` are **build-time** — changing them requires a rebuild.

### Distance metrics — must match the embedding model

The metric is set at creation (`hnsw:space`) and **must align with what the model was trained for** — otherwise scores are meaningless.

| `hnsw:space` | Use when | Notes |
|---|---|---|
| `cosine` | most text embeddings (direction matters, magnitude doesn't) | default; safe choice for sentence-transformers/OpenAI |
| `l2` | Euclidean / spatial / image embeddings where magnitude matters | sensitive to vector length |
| `ip` (inner product) | vectors already L2-normalized; max-similarity | fastest; wrong if inputs aren't normalized |

Check the model card: OpenAI `text-embedding-3-*` and most sentence-transformers are tuned for cosine.

### Embedding model choice & dimensions

| Model | Dims | Tradeoff |
|---|---|---|
| `all-MiniLM-L6-v2` (default) | 384 | fast, local, good baseline |
| `all-mpnet-base-v2` | 768 | higher quality, slower |
| OpenAI `text-embedding-3-small` | 1536 | strong quality, API latency/cost |
| OpenAI `text-embedding-3-large` | 3072 | best quality, highest cost/memory |
| `paraphrase-multilingual-MiniLM-L12-v2` | 384 | multilingual |

More dimensions ≠ always better: they raise memory, index size, and query cost. Pick the smallest model that hits your recall target on your eval set. Model selection, versioning, and evaluation are owned by [`mlops.md`](guides://mlops.md).

---

## 6. Metadata Filtering

Combine semantic search with structured predicates. **Pre-filtering** (filter the candidate set, then ANN-rank) is what `where=` does in Chroma and is preferred — it narrows the search space and keeps results relevant. **Post-filtering** (retrieve k, then drop non-matching in app code) wastes the ANN budget and can return fewer than k usable hits; avoid it.

```python
res = col.query(
    query_texts=["best practices"],
    n_results=10,
    where={                       # metadata predicate (pre-filter)
        "$and": [
            {"lang": "en"},
            {"year": {"$gte": 2023}},
            {"$or": [{"category": "engineering"}, {"category": "devops"}]},
        ]
    },
    where_document={"$contains": "production"},   # raw-text predicate
    include=["documents", "distances", "metadatas"],   # fetch only what you need
)
```

Operators: `$eq $ne $gt $gte $lt $lte` (scalars), `$in $nin` (lists), `$and $or` (logical); `where_document`: `$contains` / `$not_contains`. Design the metadata schema around **query patterns**, not source structure — indexable fields you filter on most. After mutating metadata, old filters must not match the updated doc (use `update`/`upsert`, not `add`).

---

## 7. Chunking — the upstream decision that dominates retrieval

The index can only return what you stored, so chunking is the highest-leverage knob. Too small loses context; too large dilutes relevance and blows the LLM context budget.

- **Target** ~200–500 tokens/chunk; **overlap** 10–20% to preserve context across boundaries.
- Prefer **structure-aware** splitting (headings, paragraphs, sentences) over fixed-length cuts — e.g. LangChain `RecursiveCharacterTextSplitter(separators=["\n\n","\n",". "," ",""])`.
- Carry provenance in metadata (`source_doc`, `chunk_index`, `total_chunks`) so retrieved chunks are traceable and re-assemblable.
- Validate chunking on representative data before bulk ingest — it is expensive to redo after embedding millions of vectors.

Batch ingestion: `upsert` in batches of ~500–1000 (avoid >5000/call); embedding is the bottleneck, so batch the embedding API calls too.

---

## 8. RAG Patterns

Pipeline: **chunk → embed → store → retrieve top-k → (rerank) → prompt LLM with context**. Keep retrieval and generation decoupled and individually testable.

**Two-stage retrieval + reranking** — fast bi-encoder ANN recall, then a precise cross-encoder reorders. The standard quality win when raw top-k is noisy:

```python
from sentence_transformers import CrossEncoder
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")

hits = col.query(query_texts=[q], n_results=20)          # stage 1: recall (cheap)
docs = hits["documents"][0]
scores = reranker.predict([[q, d] for d in docs])        # stage 2: precision
top = [docs[i] for i in scores.argsort()[::-1][:5]]      # rerank → final k
```

**Hybrid search** — combine **dense** (embedding/ANN) with **sparse/keyword** (BM25/full-text) retrieval and fuse the rankings (e.g. Reciprocal Rank Fusion). Dense captures meaning; sparse nails exact terms, codes, and rare tokens. Chroma is dense-only; for production hybrid, run a keyword index alongside (or use a store with built-in hybrid like Qdrant/Weaviate/Elasticsearch).

**Grounding the LLM** — pass retrieved chunks as context with explicit instructions to answer only from context and cite sources; this is what reduces hallucination. The LLM/provider-specific prompting, token budgeting, and model choice are out of scope here.

Frameworks: LangChain (`Chroma.from_documents`, `similarity_search`) and LlamaIndex (`ChromaVectorStore`) wrap these steps — use them for glue, but the chunking/metric/eval rules above still govern quality.

---

## 9. When Chroma Fits — and When to Graduate

**Chroma is the right call for:** local development and prototyping; embedded single-process apps; collections roughly up to low-millions of vectors; fast iteration with minimal ops. Its simple API and persistent/embedded mode make it ideal for getting a RAG system working.

**Graduate to another store when** you need horizontal scale (tens of millions+ vectors), high write throughput/streaming, built-in hybrid search, multi-tenancy, or managed HA:

| Alternative | Pick it for |
|---|---|
| **pgvector** (Postgres) | you already run Postgres and want vectors next to relational data (see `postgresql.md`) |
| **Qdrant** | high-performance ANN, payload filtering, built-in hybrid, self-host or cloud |
| **Weaviate** | hybrid search + modules, GraphQL, schema-rich workloads |
| **Milvus** | very large scale (billions), GPU indexing, distributed |

This is an honest fit decision, not a deficiency of Chroma — start simple, migrate when a concrete scaling/feature limit is hit. Keep the embedding/chunking/eval layer portable so the store is swappable.

---

## 10. Persistence, Operations & Security Binding

- **Persistence**: `PersistentClient(path=...)` for embedded prod; server mode (`chroma run --path ...` / Docker `chromadb/chroma`) for shared access. Set `IS_PERSISTENT=TRUE`. Never use ephemeral `Client()` for data you must keep.
- **Backup/restore**: snapshot the data directory (stop or quiesce writes first), or export via `collection.get(include=[...])` → re-`add`. **Test restore**, not just backup (CHROMA-PERSIST-01).
- **Security** (policy owned by [`secure-coding.md`](guides://secure-coding.md)): API keys (OpenAI/Cohere/HF) and Chroma tokens from env/secrets, never source; in server mode require auth headers and restrict network exposure (VPC/firewall — never public+unauth); scope collections per app/tenant; validate untrusted input before embedding (prompt-injection/PII); pin model versions so vectors don't silently shift; `pip-audit`/`npm audit` in CI.
- **Errors** (policy owned by [`error-handling.md`](guides://error-handling.md)): embedding-API calls need timeout + retry/backoff on rate limits; handle empty/insufficient query results explicitly; surface dimension-mismatch early.

---

## 11. Quick Reference

```text
CLIENT MODES   Client() ephemeral(tests) · PersistentClient(path) · HttpClient(host,port)
CORE OPS       create_collection · get_or_create_collection · add · query · update · upsert · delete
METRIC         hnsw:space = cosine(text,default) | l2(spatial) | ip(normalized)  — match the model
HNSW           M(16) build·recall/mem · construction_ef(100) build · search_ef(10) query·recall
FILTER         where{$eq $ne $gt $gte $lt $lte $in $nin $and $or} · where_document{$contains}
CHUNKING       200–500 tokens · 10–20% overlap · structure-aware · carry provenance metadata
RAG            chunk→embed→store→retrieve k→rerank(cross-encoder)→prompt; hybrid = dense+sparse(RRF)
SCALE          Chroma: proto/local/≤low-millions · pgvector/Qdrant/Weaviate/Milvus: scale/hybrid/HA
```

---

## 12. Deployment Checklist

Generated from §2 — one box per requirement ID. No new requirements here.

- [ ] CHROMA-TST-01/02 — retrieval tests pass (ephemeral client), bugs have regression tests
- [ ] CHROMA-EMB-01/02 — model pinned, identical for index & query, dimensions match
- [ ] CHROMA-DIST-01 — `hnsw:space` matches the embedding model
- [ ] CHROMA-IDX-01 — HNSW params set deliberately, recall@k meets target
- [ ] CHROMA-EVAL-01 — retrieval measured on a labeled set (recall@k/nDCG recorded)
- [ ] CHROMA-SEC-01/02/03 — no keys in source, server authed+restricted, `pip-audit` clean
- [ ] CHROMA-ERR-01 — embedding-API timeouts/retries and empty results handled
- [ ] CHROMA-PERSIST-01 — persisted storage with a tested backup+restore
- [ ] Agent ran every §3 command and documented any fixes

---
**End of Chroma Vector Database Guidelines**
