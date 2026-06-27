# Apache Kafka Guidelines
Mandatory standards for Apache Kafka as a durable, replayable event-streaming log: partitions, consumer groups, idempotent/transactional producers, exactly-once semantics, and schema evolution. Kafka 3.8 (KRaft), Schema Registry, Kafka Streams.

---
name: kafka
title: Apache Kafka Guidelines
version: 2.0
last_reviewed: 2026-06-05
kind: infra
tools: [kafka@3.8, kraft, schema-registry, kafka-streams, kafka-connect]
requires:
  - secure-coding
  - error-handling
  - observability
recommends:
  - microservices
  - performance
  - kubernetes
  - env-config
provides:
  - event-streaming
  - topics-partitions
  - consumer-groups
  - exactly-once
  - schema-registry
---

> 🧭 Authored per [`CONVENTIONS.md`](guides://CONVENTIONS.md): shared concerns are referenced, not restated. This guide covers only what is unique to Kafka. There is no dedicated Avro guide, so schema/serialization is covered inline here (§9).

---

## 0. Prerequisites & References

Fetch and apply these **before** designing Kafka topics, producers, or consumers. Their rules are assumed below and not repeated.

> 📎 **REQUIRED — fetch & apply first:**
> - [`secure-coding.md`](guides://secure-coding.md) — supply chain, secrets, transport security. *(Kafka binding: SASL + mTLS + ACLs, §14.)*
> - [`error-handling.md`](guides://error-handling.md) — retry/backoff strategy, error classification. *(Kafka binding: retryable→backoff, non-retryable→DLQ, §11.)*
> - [`observability.md`](guides://observability.md) — metrics/tracing policy. *(Kafka binding: consumer lag is the key SLI, JMX metrics, §13.)*

> 📎 **RECOMMENDED — fetch when the task touches them:**
> - [`microservices.md`](guides://microservices.md) — event-driven architecture, sagas, outbox. *(Kafka is the event backbone.)*
> - [`performance.md`](guides://performance.md) — throughput/latency policy *(binding: batching, compression, §10)*
> - [`kubernetes.md`](guides://kubernetes.md) — running Kafka/clients on k8s *(binding: Strimzi operator, StatefulSets, §15)*
> - [`env-config.md`](guides://env-config.md) — externalize bootstrap servers, credentials, topic names.

> 📎 **SEE ALSO:** [`tdd.md`](guides://tdd.md) *(test-first; integration tests via Testcontainers)* · [`hexagonal.md`](guides://hexagonal.md) *(producers/consumers as adapters)* · [`grpc.md`](guides://grpc.md) · [`openapi.md`](guides://openapi.md)

---

## 1. Core Philosophies: LOG-FIRST

Kafka-specific principles only. Security, error strategy, observability, and architecture come from §0.

- **L**og as source of truth: a topic is an immutable, append-only, replayable commit log — not a queue you drain and forget. Design for replay, not fire-and-forget.
- **O**rdering by key: ordering is guaranteed only *within a partition*; the partition is the unit of both parallelism and ordering. Related events MUST share a key.
- **G**uaranteed delivery: durability comes from `acks=all` + `min.insync.replicas≥2` + idempotent producers; pick the weakest delivery semantic the data can tolerate, and document it.
- **F**ault-tolerant by design: assume crashes; recover state by replaying from the last committed offset. Offset = checkpoint.
- **I**dempotent consumers: at-least-once is the default; consumers MUST tolerate duplicates unless full exactly-once (transactions) is in force.
- **R**egistered schemas: every event has a registered, compatibility-checked schema. Events outlive code — evolve schemas, never break readers.
- **S**ized for the job: reach for Kafka for high-throughput streaming, replay, and fan-out; reach for a simple queue (SQS/RabbitMQ) for task dispatch (§12.D).
- **T**ransactional EOS only where it pays: exactly-once via transactions for financial/critical read-process-write; otherwise at-least-once + idempotency.

**Verified Design**: Agent-generated Kafka config and code MUST satisfy every gate in §2 before delivery.

---

## 2. Requirements (MANDATORY, auditable)

RFC-2119 keywords. IDs `KAFKA-<TOPIC>-<NN>`. Each row has a binary gate; rows binding a shared rule cite its owner.

| ID | Requirement | Verify | Gate |
|----|-------------|--------|------|
| KAFKA-DUR-01 | Production topics MUST have `replication.factor≥3`, `min.insync.replicas≥2`, `unclean.leader.election.enable=false` | `kafka-topics.sh --describe` / `kafka-configs.sh` | values met |
| KAFKA-PROD-01 | Producers MUST set `acks=all` and `enable.idempotence=true` | config review / producer JMX | both set |
| KAFKA-EOS-01 | Read-process-write pipelines MUST be transactional (`transactional.id`, `sendOffsetsToTransaction`, consumer `isolation.level=read_committed`) OR consumers MUST be idempotent | code review / integration test | no dup output |
| KAFKA-OFF-01 | `enable.auto.commit=false`; offsets committed only AFTER successful processing | config review / test | no commit-before-process |
| KAFKA-PART-01 | Events requiring ordering MUST carry a non-null partition key with ≥10× more keys than partitions | producer review | key set, no hot partition |
| KAFKA-SCHEMA-01 | Every topic value MUST use a registered schema with compatibility `BACKWARD`/`FULL` (never `NONE`) (see §9) | Schema Registry API / `mvn schema-registry:test-compatibility` | compatible |
| KAFKA-DLQ-01 | Non-retryable records MUST be routed to a DLQ with error metadata (see `error-handling.md`) | code review / integration test | poison→DLQ |
| KAFKA-MSG-01 | Records MUST be <1 MB; large payloads passed by reference (claim-check) | producer review / `max.message.bytes` | size ok |
| KAFKA-SEC-01 | Transport MUST use TLS + SASL auth; topic access restricted by ACL (see `secure-coding.md`) | `kafka-acls.sh --list`, TLS handshake | enforced |
| KAFKA-OBS-01 | Consumer lag MUST be exported and alerted; under-replicated/offline partitions alerted (see `observability.md`) | `kafka-consumer-groups.sh --describe`, alert rules | alerts exist |
| KAFKA-TST-01 | Producers/consumers MUST have integration tests against a real broker (Testcontainers) (see `tdd.md`) | test runner | exit 0 |
| KAFKA-TST-02 | Each bug MUST get a regression test before the fix (see `tdd.md`) | test runner | failing→passing |

> **Forbidden**: `acks=0`, `replication.factor=1`, `enable.auto.commit=true` on transactional/critical consumers, committing offsets before processing, `transactional.id` shared across instances, schema compatibility `NONE` on production topics, treating Kafka as a queryable database, or a single partition used solely to force global ordering.

---

## 3. Verification Protocol

Run before presenting Kafka config or code. Fix → re-run until green.

```bash
kafka-topics.sh --bootstrap-server $BROKER --describe --topic $T   # KAFKA-DUR-01 (RF, ISR)
kafka-configs.sh --bootstrap-server $BROKER --entity-type topics --entity-name $T --describe
kafka-consumer-groups.sh --bootstrap-server $BROKER --describe --group $G   # KAFKA-OBS-01 (lag)
kafka-acls.sh --bootstrap-server $BROKER --command-config admin.properties --list  # KAFKA-SEC-01
# Schema compatibility (Confluent): KAFKA-SCHEMA-01
curl -s $SR/compatibility/subjects/$T-value/versions/latest -d @new-schema.json -H 'Content-Type: application/json'
# Integration tests against Testcontainers broker: KAFKA-TST-01/02
<test_runner>
```

The *why* behind each gate lives in its §0 owner; do not re-derive it here.

---

## 4. The Log Model (owned)

A **topic** is split into **partitions**; each partition is an ordered, immutable, append-only sequence of records addressed by a monotonically increasing **offset**.

```
Topic "orders" (3 partitions)         producer key → hash % partitions → partition
  P0: [0][1][2][3][4]──▶ append       Consumer A @ offset 3   (lag = LEO - committed)
  P1: [0][1][2]──▶                    Consumer B @ offset 1
  P2: [0][1][2][3][4][5]──▶
```

- **Ordering** is per-partition only; there is no global order across partitions.
- **Parallelism** is bounded by partition count: at most one consumer *per group* reads a given partition, so max useful consumers = partition count.
- **Retention** is independent of consumption — records expire by `retention.ms`/`retention.bytes`, or never (`-1`). Consumers can re-`seek` to any retained offset; replay is a first-class operation.
- **Log compaction** (`cleanup.policy=compact`) keeps only the latest record per key (plus tombstones for deletes), turning a topic into a recoverable key→latest-value table. Use for state/changelog topics (user profiles, balances, config). Use `delete` for event logs, `compact,delete` for bounded changelogs.

```bash
# Event log: time-bounded, ordered history
--config cleanup.policy=delete  --config retention.ms=604800000   # 7d
# State store: latest-per-key forever
--config cleanup.policy=compact --config retention.ms=-1 --config min.cleanable.dirty.ratio=0.1
```

Cannot *decrease* partition count, and increasing it re-hashes keys → breaks ordering for in-flight keys. Size partitions up front (§8).

---

## 5. Producers (owned)

Partition assignment is driven by the record **key**: `partition = hash(key) % numPartitions` (default murmur2). Same key → same partition → ordered. `null` key → round-robin (sticky batching), no ordering.

**Durable, idempotent producer (the default you ship):**
```properties
acks=all                                   # all in-sync replicas must ack (KAFKA-PROD-01)
enable.idempotence=true                    # broker dedups producer retries via (PID, seq)
max.in.flight.requests.per.connection=5    # safe & ordered WITH idempotence
retries=2147483647                         # bounded by delivery.timeout.ms, not count
delivery.timeout.ms=120000
compression.type=zstd                      # zstd/lz4; cuts network + disk (see performance.md)
linger.ms=10  batch.size=65536             # batch for throughput; linger.ms=0 for low latency
```

- **Idempotent producer** (default since 3.0) eliminates duplicates from *retries* within a producer session via a producer-id + per-partition sequence number. It does NOT span sessions or topics — that needs transactions (§8).
- **Delivery semantics**: `acks=0` at-most-once (lossy; metrics only); `acks=all`+idempotence at-least-once (the norm); transactions exactly-once (§8). Choose the weakest the data tolerates and document it.
- Always handle the send callback/future: a failed send is a real failure — log, count a metric, and DLQ or retry per [`error-handling.md`](guides://error-handling.md). Never fire-and-forget on durable data.

---

## 6. Consumers & Consumer Groups (owned)

A **consumer group** shares the work of a topic: each partition is assigned to exactly one member; members beyond the partition count sit idle. Adding/removing members or partitions triggers a **rebalance**.

```properties
group.id=order-processors
enable.auto.commit=false                   # commit AFTER processing (KAFKA-OFF-01)
auto.offset.reset=earliest
partition.assignment.strategy=org.apache.kafka.clients.consumer.CooperativeStickyAssignor
isolation.level=read_committed             # with transactional producers (§8)
max.poll.records=500
max.poll.interval.ms=300000                # MUST exceed worst-case batch processing time
session.timeout.ms=45000  heartbeat.interval.ms=15000   # heartbeat ≈ 1/3 session
```

- **Poll-loop discipline**: `poll()` both fetches records *and* sends heartbeats. Do all processing within `max.poll.interval.ms`; offload long work or shrink `max.poll.records`. Blowing the interval → the member is evicted → rebalance storm.
- **Cooperative rebalancing** (`CooperativeStickyAssignor`) revokes only the moving partitions instead of stop-the-world; prefer it. Newer brokers support the **KIP-848 consumer-group protocol** (broker-coordinated, lighter rebalances) — adopt where available.
- **Offset management**: commit *after* processing for at-least-once. In a `ConsumerRebalanceListener.onPartitionsRevoked`, commit (or, in EOS, finish the transaction) before partitions are taken away, or you reprocess.
- **Consumer lag** = log-end-offset − committed-offset. It is the primary health SLI (§13); growing lag means production outpaces consumption — scale consumers (up to partition count), speed processing, or add partitions.

---

## 7. Brokers, Replication & KRaft (owned)

- Each partition has one **leader** and `replication.factor−1` followers. Producers/consumers talk to the leader; followers replicate. The **in-sync replica (ISR)** set is the replicas caught up to the leader.
- `acks=all` means "acked by all ISR members"; `min.insync.replicas=2` means a write is rejected if fewer than 2 replicas are in sync — this is what makes `acks=all` durable. RF=3 + ISR=2 tolerates one broker loss for writes.
- `unclean.leader.election.enable=false` forbids electing an out-of-sync replica as leader (which would silently drop committed data). Keep it off.
- **KRaft (no ZooKeeper)**: modern Kafka (3.3+ GA; default in 3.5+; ZooKeeper removed in 4.0) stores metadata in an internal Raft-replicated `__cluster_metadata` topic managed by **controller** nodes. Run a dedicated controller quorum (3 or 5 nodes) or combined mode for small clusters. Do not stand up ZooKeeper for new deployments; migrate existing ones before upgrading to 4.x.

```properties
process.roles=broker,controller          # or split: dedicated controller nodes
controller.quorum.voters=1@c1:9093,2@c2:9093,3@c3:9093
default.replication.factor=3
min.insync.replicas=2
unclean.leader.election.enable=false
```

---

## 8. Exactly-Once Semantics & Transactions (owned)

EOS within Kafka = **idempotent producer** (no retry dups) + **transactions** (atomic writes across partitions/topics *and* the consumer offset) + **`read_committed`** consumers (never see aborted/uncommitted data).

The read-process-write loop — commit output and input offset atomically:
```text
producer.initTransactions()                 // once, claims/fences the transactional.id
while (running) {
  records = consumer.poll(timeout)
  if (records.isEmpty()) continue
  producer.beginTransaction()
  try {
    for (r in records) producer.send(outputTopic, transform(r))
    producer.sendOffsetsToTransaction(consumer.offsets(), consumer.groupMetadata())  // offset IN txn
    producer.commitTransaction()            // atomic: outputs + offsets, or nothing
  } catch (ProducerFencedException e) { shutdown() }   // a newer instance took the txn.id
    catch (KafkaException e) { producer.abortTransaction() }  // nothing committed; reprocess
}
```

```properties
transactional.id=order-processor-${INSTANCE}   # MUST be stable per instance & unique across them
transaction.timeout.ms=60000                   # < broker transaction.max.timeout.ms
# consumer side: isolation.level=read_committed, enable.auto.commit=false
```

- **`transactional.id` = zombie fencing**: on `initTransactions()` the broker bumps the producer **epoch**; an older instance reusing the same id is fenced (`ProducerFencedException`) and must die. A shared or random-per-restart id breaks this — keep it stable per logical instance and unique across instances.
- EOS costs latency/throughput (commit markers, `read_committed` buffering). Use it for read-process-write of financial/critical data; for everything else use at-least-once + idempotent consumers (e.g. dedup on an event id, or upsert by key).
- **Kafka Streams** gives EOS for free via `processing.guarantee=exactly_once_v2` — prefer it over hand-rolled transactions for stream topologies.

---

## 9. Schema Management & Serialization (owned — no dedicated Avro guide)

Events are a long-lived contract; serialization and schema evolution are first-class. Use a **Schema Registry** (Confluent or Apicurio) with **Avro** (compact, rich evolution) or **Protobuf** (cross-language, gRPC-aligned). Avoid schemaless JSON for inter-service events.

**Wire format**: each message is `[magic byte][4-byte schema id][serialized payload]`. The producer registers/looks up the schema and writes its id; the consumer reads the id and fetches (and caches) the schema to deserialize. Schemas live in the registry, not in every message.

**Compatibility modes (set per subject):**

| Mode | Guarantee | Upgrade order | Allowed change |
|------|-----------|---------------|----------------|
| `BACKWARD` (default) | new schema reads old data | consumers first | add optional/defaulted field, remove field |
| `FORWARD` | old schema reads new data | producers first | add field, remove optional field |
| `FULL` | both | either | add/remove **optional** (defaulted) fields only |
| `NONE` | none | — | ❌ forbidden in production (KAFKA-SCHEMA-01) |

**Safe evolution (FULL/BACKWARD):** ✅ add a field *with a default*, remove a defaulted field, append enum symbols. ❌ add a required field without default, change a field's type, rename a field (use aliases), reuse a Protobuf field number.

```json
// Avro: every field that may be added later has a default → BACKWARD/FULL safe
{ "type": "record", "name": "Order", "namespace": "com.acme.orders",
  "fields": [
    { "name": "orderId",    "type": "string" },
    { "name": "customerId", "type": "string" },
    { "name": "amount",     "type": "double" },
    { "name": "currency",   "type": "string", "default": "USD" },
    { "name": "metadata",   "type": ["null", {"type":"map","values":"string"}], "default": null }
  ]
}
```

Register/validate compatibility in CI before producers ship; the build MUST fail on an incompatible change. Subject naming strategy (`TopicNameStrategy` vs `RecordNameStrategy`) decides whether a topic carries one schema or many — pick deliberately.

---

## 10. Partitioning & Key Design (owned)

- **Key choice = ordering + distribution.** Good keys: `order_id`, `customer_id`, `account_id`, `device_id`, `tenant_id:entity_id` — anything whose events must stay ordered together. Bad keys: timestamp (hot partitions), random UUID (no ordering benefit), `null` (no ordering), low-cardinality enums (skew).
- Keep key cardinality ≥ ~10× partition count to spread load; watch for hot keys (one whale customer) and consider a composite key.
- **Partition count** = `max(target_throughput / per-partition_throughput, max_consumers_needed)`. Start 6–12 for most topics; 30–100 for high throughput. More partitions = more parallelism but more open files, memory, and longer leader-election/rebalance; keep well under a few thousand per broker. Plan ahead — you can grow but not shrink, and growth re-hashes keys.

```bash
kafka-topics.sh --bootstrap-server $BROKER --create --topic orders \
  --partitions 12 --replication-factor 3 \
  --config min.insync.replicas=2 --config retention.ms=604800000
```

---

## 11. Error Handling & Dead Letter Queues (binds `error-handling.md`)

Retry/backoff/classification policy is owned by [`error-handling.md`](guides://error-handling.md). Kafka binding:

- **Classify**: *retryable* (network/timeout/downstream-unavailable) → bounded exponential backoff with jitter; *non-retryable* (validation/poison/schema failure) → DLQ immediately. Never block the whole partition retrying a poison record in-line — it stalls every key behind it.
- **DLQ** = a separate topic (`orders.dlq`) holding the original key/value/headers plus error metadata (exception type/message, source topic-partition-offset, attempt count, timestamp, consumer group). Alert on any DLQ traffic (§13); triage, fix, and replay.
- For tiered retry without head-of-line blocking, use **retry topics** (`orders.retry.5s`, `orders.retry.1m`) or a framework that implements them (Spring Kafka `@RetryableTopic`, Kafka Connect `errors.tolerance`+`errors.deadletterqueue.topic.name`).
- On producer send failure inside a transaction, abort and reprocess; outside a transaction, retry then DLQ on exhaustion.

---

## 12. Patterns & Fit (owned)

- **Event sourcing**: the topic *is* the state. Services rebuild state by replaying events (optionally from a snapshot + offset to bound replay time). Compacted topics serve as the materialized latest-state.
- **CDC**: stream database changes into Kafka via **Kafka Connect + Debezium** (or the transactional **outbox** pattern, see [`microservices.md`](guides://microservices.md)) instead of dual-writes.
- **Stream processing**: prefer **Kafka Streams** (JVM library, EOS, stateful joins/aggregations, RocksDB state stores backed by compacted changelog topics) or **ksqlDB** for SQL-style transforms; **Apache Flink** for advanced windowing/large state. Don't hand-roll stateful stream logic on raw consumers when Streams fits.
- **Event-driven microservices / sagas / CQRS**: Kafka is the backbone — architecture patterns owned by [`microservices.md`](guides://microservices.md).

**D. When Kafka fits vs a simple queue (be honest):**

| Use Kafka when | Use a queue (SQS/RabbitMQ) when |
|---|---|
| High throughput, many consumers fan-out | Simple task dispatch to a worker pool |
| You need **replay** / event history / audit | Once consumed, the message is gone — fine |
| Multiple independent consumer groups per stream | One logical consumer; competing-consumers |
| Stream processing / event sourcing / CDC | Per-message ack/visibility-timeout, priority, delay |
| Ordered per-key at scale | Low ops budget; no partition/operations overhead |

Kafka has real operational weight (brokers, partitions, schema registry, rebalancing). For RPC use [`grpc.md`](guides://grpc.md)/[`rest.md`](guides://rest.md); for a job queue, a queue is simpler and cheaper. Don't reach for Kafka by default.

---

## 13. Observability (binds `observability.md`)

Metrics/tracing policy is owned by [`observability.md`](guides://observability.md). Kafka surfaces metrics via JMX (scrape with the Prometheus JMX exporter). Key SLIs:

- **Consumer group lag** (per group/topic/partition) — *the* primary SLI; alert when sustained or growing. Causes: slow processing, too few consumers, downstream slowness, traffic spike. Fixes: scale consumers (≤ partition count), optimize processing, add partitions.
- **Broker health**: `UnderReplicatedPartitions>0`, `OfflinePartitionsCount>0`, `ActiveControllerCount!=1`, `UncleanLeaderElectionsPerSec>0` — all page-worthy.
- **Producer**: `record-error-rate>0`, `request-latency`, low `batch-size-avg` (inefficient batching), `buffer-available-bytes` low (backpressure).
- **Consumer**: `rebalance-rate` (group instability), `commit-latency`, `records-lag-max`.

```yaml
# Prometheus binding (policy in observability.md)
- alert: KafkaConsumerLagHigh
  expr: kafka_consumergroup_lag > 10000
  for: 5m
  labels: { severity: warning }
- alert: KafkaUnderReplicatedPartitions
  expr: kafka_server_replicamanager_underreplicatedpartitions > 0
  for: 5m
  labels: { severity: critical }
```

---

## 14. Security (binds `secure-coding.md`)

Transport/secret/supply-chain policy is owned by [`secure-coding.md`](guides://secure-coding.md). Kafka binding:

- **Encrypt + authenticate every connection**: `SASL_SSL` listeners; SASL mechanism `SCRAM-SHA-512` or mTLS (`ssl.client.auth=required`). No `PLAINTEXT` listeners in production.
- **Authorize with ACLs**: deny-by-default; grant least-privilege Write/Read per principal per topic/group. Place controllers on an internal-only network; separate internal vs external listeners.
- Inject keystore/credential secrets from the environment/secret manager (see [`env-config.md`](guides://env-config.md)); never commit them. Scan client libraries (`pip-audit`, `mvn dependency-check`, `npm audit`, `govulncheck`) and broker images (`trivy`) per `secure-coding.md`.

```properties
listeners=SASL_SSL://0.0.0.0:9093
security.inter.broker.protocol=SASL_SSL
sasl.enabled.mechanisms=SCRAM-SHA-512
ssl.keystore.location=/etc/kafka/ssl/server.keystore.jks
ssl.keystore.password=${KEYSTORE_PASSWORD}
ssl.client.auth=required
authorizer.class.name=org.apache.kafka.metadata.authorizer.StandardAuthorizer   # KRaft authorizer
```

---

## 15. Running Kafka (binds `kubernetes.md`)

- On Kubernetes, use the **Strimzi** operator (or Confluent for Kubernetes): it manages KRaft `KafkaNodePool`s as StatefulSets, persistent volumes, rolling upgrades, listeners/TLS, ACLs, topics (`KafkaTopic` CRDs), and users (`KafkaUser` CRDs). Don't hand-roll StatefulSets. Cluster/operator deployment policy → [`kubernetes.md`](guides://kubernetes.md).
- Dedicate fast disks to log dirs (no shared/network storage for hot data); JVM heap 6–8 GB and never >50% RAM (Kafka relies on the OS page cache).
- Externalize bootstrap servers, topic names, group ids, and credentials as config/secrets (see [`env-config.md`](guides://env-config.md)).

---

## 16. Quick Reference

```bash
# Topics
kafka-topics.sh --bootstrap-server $B --list
kafka-topics.sh --bootstrap-server $B --create --topic orders --partitions 12 \
  --replication-factor 3 --config min.insync.replicas=2 --config retention.ms=604800000
kafka-topics.sh --bootstrap-server $B --describe --topic orders
kafka-topics.sh --bootstrap-server $B --alter --topic orders --partitions 24   # increase only

# Consumer groups (lag!)
kafka-consumer-groups.sh --bootstrap-server $B --describe --group order-processors
kafka-consumer-groups.sh --bootstrap-server $B --group g --topic orders \
  --reset-offsets --to-earliest --execute        # or --to-offset N / --to-datetime ...

# Config
kafka-configs.sh --bootstrap-server $B --entity-type topics --entity-name orders \
  --alter --add-config cleanup.policy=compact

# ACLs
kafka-acls.sh --bootstrap-server $B --command-config admin.properties \
  --add --allow-principal User:producer-app --operation Write --topic orders

# Debug produce/consume
kafka-console-producer.sh --bootstrap-server $B --topic orders --property parse.key=true --property key.separator=:
kafka-console-consumer.sh --bootstrap-server $B --topic orders --from-beginning --property print.key=true
```

| Semantic | Config |
|---|---|
| At-most-once | `acks=0` (lossy; metrics/logs only) |
| At-least-once | `acks=all`, `enable.idempotence=true`, manual commit + idempotent consumer |
| Exactly-once | `transactional.id`, `isolation.level=read_committed`, offsets in txn (or Streams `exactly_once_v2`) |

---

## 17. Deployment Checklist

Generated from §2 — one box per requirement ID. No new requirements.

- [ ] KAFKA-DUR-01 — RF≥3, ISR≥2, unclean leader election off
- [ ] KAFKA-PROD-01 — `acks=all` + `enable.idempotence=true`
- [ ] KAFKA-EOS-01 — read-process-write transactional, or consumers idempotent
- [ ] KAFKA-OFF-01 — auto-commit off; commit after processing
- [ ] KAFKA-PART-01 — ordering keys set; cardinality ≥10× partitions; no hot partitions
- [ ] KAFKA-SCHEMA-01 — schemas registered, compatibility BACKWARD/FULL, CI-checked
- [ ] KAFKA-DLQ-01 — non-retryable records → DLQ with metadata (see `error-handling.md`)
- [ ] KAFKA-MSG-01 — records <1 MB; large payloads by reference
- [ ] KAFKA-SEC-01 — TLS + SASL + ACLs (see `secure-coding.md`)
- [ ] KAFKA-OBS-01 — consumer lag + broker health alerting (see `observability.md`)
- [ ] KAFKA-TST-01/02 — integration tests against a real broker; bugs get regression tests first (see `tdd.md`)
- [ ] Agent ran every §3 verification command and documented any fixes

---
**End of Apache Kafka Guidelines**
