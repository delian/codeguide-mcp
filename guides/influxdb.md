# InfluxDB Development Guidelines
Mandatory standards for time-series data in InfluxDB: the measurement/tag/field/timestamp model, line protocol, cardinality control, retention/downsampling, and the 1.x/2.x/3.x version split. InfluxDB 3.x (Core/Enterprise), Telegraf, line protocol.

---
name: influxdb
title: InfluxDB Development Guidelines
version: 2.0
last_reviewed: 2026-06-05
kind: datastore
tools: [influxdb@3-core, influxdb@3-enterprise, influxdb@2.7, telegraf, line-protocol, flux, influxql]
requires:
  - secure-coding
  - error-handling
recommends:
  - observability
  - performance
  - timescaledb
  - env-config
provides:
  - timeseries-model
  - tag-field-design
  - cardinality-management
  - line-protocol
  - influx-versions
---

> 🧭 Authored per [`CONVENTIONS.md`](guides://CONVENTIONS.md): shared concerns are referenced, not restated. This guide covers only what is unique to InfluxDB — the time-series data model, line protocol, cardinality, retention, and the version landscape.

---

## 0. Prerequisites & References

Fetch and apply these **before** generating InfluxDB code. Their rules are assumed below and not repeated.

> 📎 **REQUIRED — fetch & apply first:**
> - [`secure-coding.md`](guides://secure-coding.md) — secrets, supply chain, CVE policy. *(InfluxDB binding: API tokens are bearer credentials — never hardcode; scope each token to the minimum buckets/orgs with read/write separation; pin the image to an explicit version tag, never `latest`.)*
> - [`error-handling.md`](guides://error-handling.md) — error strategy & propagation. *(Binding: classify write/batch failures — `4xx` schema/partial-write errors are non-retryable, `429`/`503` are retryable with exponential backoff; handle backpressure rather than dropping data.)*

> 📎 **RECOMMENDED — fetch when the task touches them:**
> - [`observability.md`](guides://observability.md) — InfluxDB is most often the **storage backend** for metrics/events. observability.md owns the *instrumentation* (what to emit, RED/USE, trace/metric semantics); InfluxDB owns where it lands and how it is queried. Also monitor InfluxDB itself via its `/metrics` endpoint.
> - [`performance.md`](guides://performance.md) — methodology behind the two levers that matter here: **cardinality** and **batch writes**.
> - [`timescaledb.md`](guides://timescaledb.md) — the main relational alternative; see §10 for when to pick which.
> - [`env-config.md`](guides://env-config.md) — config policy *(binding: URL, org, token, bucket come from env/secrets, never literals)*.

> 📎 **SEE ALSO:** [`tdd.md`](guides://tdd.md) *(test-first owner; run queries/writes against an ephemeral test bucket)* · [`logging.md`](guides://logging.md) · [`docker-compose.md`](guides://docker-compose.md) · [`kubernetes.md`](guides://kubernetes.md)

---

## 1. Core Philosophies: TSDB-FIRST

InfluxDB-specific principles only. Security, error handling, and perf *policy* come from §0.

- **T**ags are indexed dimensions, fields are values: filter/group by **tags**, store measurements in **fields**. This single decision drives both query speed and the #1 failure mode (§6).
- **S**chema follows query patterns: model measurements, tags, and retention around *how you read*, partitioned by time — never around how the source emits.
- **D**esign against cardinality first: series cardinality = product of unique tag-value counts. On TSM engines (1.x/2.x) it is the dominant cost and the main way people break InfluxDB. Treat every tag as a cardinality decision.
- **B**atch and order writes: write in batches (thousands of lines), tags sorted lexicographically, coarsest acceptable timestamp precision; never one point per request.
- **E**xpire and downsample by design: raw data has a short retention; tasks/continuous queries roll it into longer-lived downsampled buckets.

**Verified Code**: Agent-generated schema, writes, and queries MUST pass every gate in §2 before delivery.

---

## 2. Requirements (MANDATORY, auditable)

RFC-2119 keywords. IDs `INFLUX-<TOPIC>-<NN>`. Each row has a binary gate; rows binding a shared rule cite its owner.

| ID | Requirement | Verify | Gate |
|----|-------------|--------|------|
| INFLUX-MODEL-01 | Measured values MUST be **fields**; filter/group dimensions MUST be **tags** (never the reverse) | schema review of line protocol | no values-as-tags |
| INFLUX-MODEL-02 | A tag and field MUST NOT share a name within a measurement (3.x rejects it) | review / write test | no name collision |
| INFLUX-CARD-01 | No unbounded/unique value (UUID, session/request ID, raw timestamp, full URL) MUST be a tag | review of tag keys | none present |
| INFLUX-CARD-02 | Series cardinality MUST be measured and budgeted (esp. TSM 1.x/2.x) | `SHOW SERIES CARDINALITY` / `influxdb.cardinality()` | within budget |
| INFLUX-WRITE-01 | Writes MUST be batched, not point-per-request | client config review | batched |
| INFLUX-WRITE-02 | Retryable write failures (`429`/`503`) MUST use backoff; non-retryable surfaced (see `error-handling.md`) | retry-path test | backoff present |
| INFLUX-QRY-01 | Every query MUST carry a bounded time range | query review | no unbounded `range`/`WHERE` |
| INFLUX-RET-01 | Each bucket/database MUST have an explicit retention; long-range reads hit downsampled data | bucket/RP review | retention set |
| INFLUX-SEC-01 | Tokens/credentials MUST come from secrets/env, least-privilege scoped (see `secure-coding.md`) | grep / token audit | no literals |
| INFLUX-SEC-02 | Image/client versions MUST be pinned; TLS ≥1.2 in production (see `secure-coding.md`) | manifest review | pinned, TLS on |
| INFLUX-VER-01 | Code MUST target one declared engine version; query language matches it (§4) | review | version declared |
| INFLUX-TST-01 | Writes/queries MUST be tested against an ephemeral bucket (see `tdd.md`) | test runner | exit 0, 0 skips |

> **Forbidden**: putting unique IDs or timestamps in tags; querying without a time range; one-point-per-request writes; hardcoded tokens; running a query language that the target engine version does not support (e.g. assuming Flux on 3.x); using `latest` image tags in production.

---

## 3. Verification Protocol

Run before presenting schema/writes/queries. Fix → re-run until green.

```bash
# INFLUX-CARD-02 — measure series cardinality (TSM engines)
influx query 'import "influxdata/influxdb" influxdb.cardinality(bucket:"my-bucket", start:-30d)'
# 1.x: SHOW SERIES CARDINALITY ON mydb   /   SHOW TAG KEY CARDINALITY ON mydb

# INFLUX-QRY-01 — confirm every query has a bounded range (grep for range(start: / WHERE time)
# INFLUX-RET-01 — buckets have retention
influx bucket list --org "$INFLUX_ORG"

# INFLUX-SEC-01/02 — no hardcoded tokens, pinned versions
grep -RinE 'token\s*[:=]\s*["'\'']?[A-Za-z0-9_-]{20,}' src/ || echo "clean"

# INFLUX-WRITE-* — dry-run a batch against a test bucket and assert it parses
curl -sS -XPOST "$INFLUX_URL/api/v2/write?org=$INFLUX_ORG&bucket=test&precision=s" \
  -H "Authorization: Token $INFLUX_TOKEN" --data-binary @sample.lp
```

The *why* behind security/error/perf gates lives in the §0 owners; do not re-derive it here.

---

## 4. The Version Landscape (read this first)

InfluxDB is **three different products** sharing a name. Targeting the wrong one is the most common source of broken examples. Declare your engine version (INFLUX-VER-01) before writing anything.

| | **1.x** (legacy) | **2.x** | **3.x** (current direction) |
|---|---|---|---|
| Storage engine | TSM (Time-Structured Merge tree) | TSM | **Columnar**: Apache Parquet + Arrow/DataFusion (FDAP stack, Rust) |
| Organizing unit | database + retention policy (RP) | **bucket** (in an **org**) | bucket/database |
| Query language | **InfluxQL** | **Flux** (primary) + InfluxQL | **SQL** (FlightSQL) + **InfluxQL**; Flux dropped |
| Auth | user/password | **API tokens** | API tokens |
| Downsampling | **Continuous Queries** (InfluxQL) | **Tasks** (Flux) | Tasks / external scheduling |
| Cardinality | tightly limited by TSM index | limited by TSM index | effectively **unbounded** (columnar) |
| Editions | OSS / Enterprise | OSS / Cloud | **Core** (OSS) / **Enterprise** / Clustered / Cloud Dedicated |

Key facts:
- **3.x is the current direction.** TSM is gone; data is stored as columnar Parquet files in object storage (S3/Azure/GCS/local), queried with Apache DataFusion. This removes the historical cardinality ceiling and adds native SQL. Flux is **not** carried forward — use **SQL or InfluxQL** on 3.x.
- **Flux** (2.x) is a functional pipeline language, now in maintenance mode. New work targeting 3.x should not invest in Flux.
- The Docker `latest` tag points at **InfluxDB 3 Core**. Always pin an explicit tag (INFLUX-SEC-02) to avoid a surprise major upgrade.
- There is **no in-place 2.x→3.x upgrade**: migration is export line protocol → import. Watch for 3.x restrictions: no duplicate tag/field names, and a per-table column limit.

---

## 5. The Data Model & Line Protocol

The model is fixed and identical in spirit across versions:

```text
measurement,tagKey=tagValue,... fieldKey=fieldValue,... timestamp
└ measurement   container (like a table)
└ tags          INDEXED string dimensions  → WHERE / GROUP BY
└ fields        the actual values (int/float/bool/string), NOT indexed
└ timestamp     nanosecond precision (overridable per write)
```

A **series** = measurement + a unique tag set. A **point** = a series at one timestamp.

### Line protocol (the write format)

```line-protocol
# measurement,<sorted tags> <fields> <timestamp>
weather,location=us-midwest,sensor=s001 temperature=82.5,humidity=65.2 1704067200000000000
cpu,host=server-01,region=us-east,env=prod usage=45.2,idle=54.8 1704067200
```

Rules: tags **sorted lexicographically** (better compression + faster series lookup); strings quoted, no quotes on tags; escape commas/spaces/`=` in tag/field keys and values; choose the **coarsest** precision you need (`s` over `ns`) — coarser timestamps compress far better. Store data in tag/field **values**, never encode it into measurement or key names.

### Tags vs fields — the decision that defines the schema

| Use a **TAG** when… | Use a **FIELD** when… |
|---|---|
| you filter or `GROUP BY` it | it is a measured value |
| it is low/medium cardinality, categorical | it is high cardinality or unique (IDs, hashes) |
| `host`, `region`, `env`, `sensor_type` | `usage`, `temperature`, `latency_ms`, `user_id` |

Same measurement should carry the same tag keys across points. **Anything you would otherwise filter on but whose value set is unbounded belongs in a field, not a tag** — that is the bridge to §6.

---

## 6. Cardinality — the #1 design concern (and #1 footgun)

**Series cardinality is the product of unique tag-value counts**, summed across measurements. It, not data volume, is what breaks InfluxDB on TSM engines.

```text
cpu_usage with tags: host(1000) × region(4) × env(3)  → 12,000 series
add user_id(1,000,000) as a tag                       → 12,000,000,000 series  ← cardinality explosion
```

On **TSM (1.x/2.x)** every series sits in an in-memory/TSI index (~roughly 1 KB each). High cardinality → ballooning RAM, OOM kills, slow queries, slow startup. On **3.x columnar** the ceiling is effectively removed — but high cardinality still costs compaction and query work, so it remains a design concern, just not a hard wall.

**The cause is almost always a unique value placed in a tag.** Anti-patterns (all forbidden by INFLUX-CARD-01): user IDs, session/request IDs, full URLs, log messages, raw timestamps, random hashes as **tags**.

Fixes:
- Move the high-cardinality value to a **field** (queryable, but not indexed → no cardinality cost).
- Bucket/quantize continuous values (e.g. `status_class=2xx` instead of exact code, latency buckets).
- Use a separate measurement instead of an exploding tag.
- Measure before and after: `SHOW SERIES CARDINALITY` / `SHOW TAG KEY CARDINALITY` (InfluxQL) or `influxdb.cardinality()` (Flux). Budget cardinality the way you budget memory.

The methodology for reasoning about this cost lives in [`performance.md`](guides://performance.md).

---

## 7. Writing & Ingestion

### Batching (INFLUX-WRITE-01)

Never write one point per request. Accumulate and flush in batches; enable gzip on the HTTP body.

| Engine | Recommended batch |
|---|---|
| 2.x | ~5,000 lines / ~5 MB per request |
| 3.x | ~10,000 lines / ~10 MB per request |

Client libraries (Python/Go/JS) provide batching write APIs with `batch_size`, `flush_interval`, and retry options — use them rather than hand-rolling. Write batches in ascending time order, tags pre-sorted.

### Failure handling (INFLUX-WRITE-02)

Bind to [`error-handling.md`](guides://error-handling.md): a write is **partial-success-aware** — a `400` means some lines were rejected (schema/type conflict, bad line protocol) and MUST NOT be blindly retried; `429`/`503` are backpressure and MUST be retried with exponential backoff + jitter. Never silently drop points on failure; surface or buffer them.

### Telegraf (the standard ingestion agent)

Telegraf is InfluxData's plugin-driven collector (300+ input plugins: system, Docker, Prometheus scrape, Kafka, MQTT, SNMP, cloud). Prefer it over bespoke collectors for metrics ingestion.

```toml
[[inputs.cpu]]
  percpu = true
[[inputs.prometheus]]
  urls = ["http://app:9090/metrics"]
[[outputs.influxdb_v2]]
  urls   = ["http://influxdb:8086"]
  token  = "${INFLUX_TOKEN}"        # from env/secret, never inline (INFLUX-SEC-01)
  organization = "my-org"
  bucket = "telegraf"
```

---

## 8. Querying

Pick the language by engine version (§4). **Every query carries a time range** (INFLUX-QRY-01) — an unbounded query scans all data.

### SQL (3.x)

```sql
SELECT date_bin(INTERVAL '5 minutes', time) AS t, mean(usage)
FROM cpu
WHERE time >= now() - INTERVAL '1 hour' AND host = 'server-01'
GROUP BY t ORDER BY t;
```

### InfluxQL (1.x, 2.x, 3.x)

```sql
SELECT mean("usage") FROM "cpu"
WHERE time > now() - 1h AND "host" = 'server-01'
GROUP BY time(5m);
-- Filter on TAGS (indexed); filtering on a field forces a full scan.
```

### Flux (2.x only — maintenance mode)

```flux
from(bucket: "my-bucket")
  |> range(start: -1h)                                  // pushdown
  |> filter(fn: (r) => r._measurement == "cpu")        // pushdown
  |> filter(fn: (r) => r.host == "server-01")           // pushdown (tag, indexed)
  |> aggregateWindow(every: 5m, fn: mean, createEmpty: false)
```

Optimization rules (all engines): filter on tags before aggregating; keep `range`/`filter`/`group`/`aggregateWindow` as **pushdown** operations (a `map` early in a Flux pipeline stops pushdown and pulls data into memory); query downsampled buckets for long ranges, raw buckets only for recent windows.

---

## 9. Retention, Buckets & Downsampling

Every bucket/database gets an explicit retention (INFLUX-RET-01); raw high-resolution data is short-lived and **downsampled** into longer-lived, coarser buckets. Typical tiering:

```text
raw      10s resolution   7 days     →  hourly   1m   90 days   →  daily   1h   2 years
```

The downsampling mechanism follows the version:

- **3.x / 2.x — Tasks (Flux or scheduled SQL).** Schedule before retention expires; use an `offset` to capture late data; preserve tags when grouping.

  ```flux
  option task = {name: "downsample-hourly", every: 1h, offset: 10m}
  from(bucket: "raw") |> range(start: -1h)
    |> filter(fn: (r) => r._measurement == "cpu")
    |> aggregateWindow(every: 1m, fn: mean, createEmpty: false)
    |> to(bucket: "hourly")
  ```

- **1.x — Continuous Queries (InfluxQL) + Retention Policies.**

  ```sql
  CREATE RETENTION POLICY "one_week" ON "mydb" DURATION 7d REPLICATION 1;
  CREATE CONTINUOUS QUERY "cq_5m" ON "mydb" BEGIN
    SELECT mean("value") AS "value" INTO "one_week"."downsampled_5m"
    FROM "raw" GROUP BY time(5m), * END;
  ```

Pick aggregations by metric type: gauges → mean/min/max/last; counters → sum/increase; latency → quantiles (preserve percentiles, don't average them).

---

## 10. When InfluxDB Fits — and When TimescaleDB Doesn't

**Reach for InfluxDB when** the workload is high-ingest, time-ordered, append-mostly, low-relational: infrastructure/app **metrics**, **IoT/IIoT** sensor telemetry, real-time **events**, and you want line protocol + Telegraf + built-in retention/downsampling out of the box. 3.x's columnar engine additionally handles high-cardinality and ad-hoc SQL analytics well.

**Prefer [`timescaledb.md`](guides://timescaledb.md) (PostgreSQL) when** you need real **relational** features around the time series — joins to dimension tables, foreign keys, transactions, `JSONB`, mature SQL tooling, and a single store for both operational and time-series data. TimescaleDB keeps full PostgreSQL semantics; InfluxDB trades them for a purpose-built ingestion/retention pipeline and (on 3.x) cheap object-store columnar storage.

Rule of thumb: **metrics/IoT firehose with simple per-series queries → InfluxDB; time series that must live next to relational data → TimescaleDB.** Other stores (Prometheus for pull-based monitoring, ClickHouse/DuckDB for analytical columns) compete at the edges; the binding above is the primary comparison.

---

## 11. Deployment & Operations (brief)

InfluxDB ships as a single container/binary (3.x Core/Enterprise, 2.7 OSS). Container, compose, and orchestration *patterns* are owned by [`docker-compose.md`](guides://docker-compose.md) and [`kubernetes.md`](guides://kubernetes.md) — apply them; InfluxDB-specific notes only:

- Pin the image tag (INFLUX-SEC-02); `latest` = InfluxDB 3 Core.
- Inject `token`/`org`/`bucket`/credentials via secrets/env (INFLUX-SEC-01, see `env-config.md`); run as non-root (UID 1500), bind to a private interface, TLS ≥1.2 in production.
- Persist storage on SSD/NVMe; 3.x points at an object store (S3/Azure/GCS or local file). OSS editions are single-node — HA/clustering is an Enterprise/Clustered feature.
- Health: `GET /health`; Prometheus-format self-metrics: `GET /metrics` (scrape with Telegraf/Prometheus — instrumentation policy in `observability.md`).
- Backup: 2.x `influx backup/restore`; 3.x relies on object-store snapshots.

---

## 12. Deployment Checklist

Generated from §2 — one box per requirement ID. No new requirements here.

- [ ] INFLUX-MODEL-01 — values are fields, dimensions are tags
- [ ] INFLUX-MODEL-02 — no tag/field name collisions
- [ ] INFLUX-CARD-01 — no unique/unbounded values used as tags
- [ ] INFLUX-CARD-02 — series cardinality measured & within budget
- [ ] INFLUX-WRITE-01 — writes are batched
- [ ] INFLUX-WRITE-02 — retryable failures backed off, others surfaced
- [ ] INFLUX-QRY-01 — every query has a bounded time range
- [ ] INFLUX-RET-01 — explicit retention; long ranges read downsampled data
- [ ] INFLUX-SEC-01 — tokens from secrets, least-privilege scoped
- [ ] INFLUX-SEC-02 — versions pinned, TLS ≥1.2 in production
- [ ] INFLUX-VER-01 — one declared engine version; matching query language
- [ ] INFLUX-TST-01 — writes/queries tested against an ephemeral bucket
- [ ] Agent ran every §3 command and documented any fixes

---
**End of InfluxDB Guidelines**
