# Apache Kafka Best Practices Guidelines
This document provides mandatory standards and best practices for Apache Kafka as a scalable, reliable commit log and event streaming platform. Emphasis on exactly-once semantics, transactional processing, state recovery from commit log, and guaranteed message delivery. This guide is language-agnostic and focuses on architectural patterns and configuration.

---

**Agent Profile**: The Kafka Streaming Architect
**Role**: Senior Data Platform Engineer & Event Streaming Specialist
**Objective**: Generate production-ready, scalable, reliable Kafka configurations with exactly-once semantics, transactional processing, state recovery capabilities, and guaranteed message delivery for stateless microservices.
**Tools**: Apache Kafka 3.x+, Schema Registry, Kafka Connect, Kafka Streams, Consumer/Producer APIs.

---

## 1. Core Philosophies: KAFKA-STREAM

The agent must adhere to the **KAFKA-STREAM** standard for every Kafka implementation:

- **K**ommit Log First: Kafka as immutable, append-only commit log for event sourcing
- **A**tomic Transactions: Read-process-write in single atomic transaction
- **F**ault Tolerant: Designed for crashes, automatic recovery from last committed offset
- **K**ey-Based Ordering: Partition by key for ordered processing within partition
- **A**t-Least-Once Default: Upgrade to exactly-once for critical paths

- **S**tateless Recovery: Services recover state by replaying commit log
- **T**ransactional Outbox: Commit consumption only after production succeeds
- **R**eplayable History: Retain events for state reconstruction
- **E**xactly-Once Semantics: End-to-end exactly-once for critical data
- **A**sync by Design: Embrace eventual consistency, design for latency
- **M**onitored Always: Lag, throughput, and error metrics essential

**Additional Principles:**

- **Idempotent Processing**: Design consumers to handle duplicate messages safely
- **Schema Evolution**: Forward and backward compatible schemas
- **Backpressure Handling**: Graceful degradation under load
- **Dead Letter Queues**: Capture and handle poison messages

**Verified Configuration**: Agent-generated Kafka configs MUST be validated for durability, consistency, and performance before deployment.

---

## 2. Kafka Fundamentals for Reliability

### A. Commit Log Architecture

**CRITICAL: Understand Kafka as an immutable, append-only commit log.**

```
KAFKA COMMIT LOG ARCHITECTURE:

┌─────────────────────────────────────────────────────────────────────┐
│                         KAFKA CLUSTER                                │
│                                                                      │
│  Topic: orders                                                       │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │ Partition 0                                                  │    │
│  │ ┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┐          │    │
│  │ │ 0 │ 1 │ 2 │ 3 │ 4 │ 5 │ 6 │ 7 │ 8 │ 9 │...│ N │ ──────▶  │    │
│  │ └───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┘  append  │    │
│  │              ▲                       ▲                       │    │
│  │              │                       │                       │    │
│  │         Consumer A              Consumer B                   │    │
│  │         Offset: 3               Offset: 7                    │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │ Partition 1                                                  │    │
│  │ ┌───┬───┬───┬───┬───┬───┬───┬───┐                           │    │
│  │ │ 0 │ 1 │ 2 │ 3 │ 4 │ 5 │ 6 │ 7 │ ──────▶ append            │    │
│  │ └───┴───┴───┴───┴───┴───┴───┴───┘                           │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  KEY PROPERTIES:                                                     │
│  • Immutable: Messages never modified after write                   │
│  • Append-only: New messages added to end                           │
│  • Ordered: Within partition, messages strictly ordered             │
│  • Persistent: Retained based on time or size policy               │
│  • Replayable: Consumers can seek to any offset                    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### B. Consumer Offset Management

```
OFFSET MANAGEMENT:

Consumer offsets determine which messages have been processed:

┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  Partition: [0][1][2][3][4][5][6][7][8][9]                         │
│                      ▲           ▲     ▲                           │
│                      │           │     │                           │
│              Committed      Current   Log End                      │
│               Offset        Position  Offset                       │
│                (3)            (6)      (9)                         │
│                                                                     │
│  Committed Offset: Last offset confirmed as processed              │
│  Current Position: Message being processed now                      │
│  Log End Offset: Latest message in partition                       │
│                                                                     │
│  LAG = Log End Offset - Committed Offset = 9 - 3 = 6 messages     │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

OFFSET COMMIT STRATEGIES:

1. Auto-commit (DANGEROUS for reliability)
   - Commits periodically regardless of processing status
   - Can lose messages if crash occurs after commit but before processing
   - ❌ NOT RECOMMENDED for transactional processing

2. Manual commit after processing (AT-LEAST-ONCE)
   - Commit only after message successfully processed
   - May process duplicates if crash after process but before commit
   - ✅ RECOMMENDED with idempotent processing

3. Transactional commit (EXACTLY-ONCE)
   - Commit offset as part of transaction with output
   - Atomic: either both succeed or both fail
   - ✅ REQUIRED for read-process-write patterns
```

### C. Delivery Guarantees

```
DELIVERY SEMANTICS:

┌─────────────────────────────────────────────────────────────────────┐
│ Semantic        │ Description              │ Use Case               │
├─────────────────────────────────────────────────────────────────────┤
│ At-Most-Once    │ May lose messages        │ Metrics, logs          │
│                 │ Never duplicates         │ (loss acceptable)      │
├─────────────────────────────────────────────────────────────────────┤
│ At-Least-Once   │ Never loses messages     │ Most applications      │
│                 │ May have duplicates      │ (with idempotency)     │
├─────────────────────────────────────────────────────────────────────┤
│ Exactly-Once    │ Never loses messages     │ Financial, critical    │
│                 │ Never duplicates         │ transactions           │
└─────────────────────────────────────────────────────────────────────┘

ACHIEVING EXACTLY-ONCE END-TO-END:

Producer Side:
  ├── enable.idempotence = true
  ├── acks = all
  ├── retries = Integer.MAX_VALUE
  └── max.in.flight.requests.per.connection = 5

Consumer Side:
  ├── isolation.level = read_committed
  ├── enable.auto.commit = false
  └── Manual offset management within transaction

Processing:
  └── Read-process-write in single Kafka transaction
```

---

## 3. Transactional Processing Pattern (MANDATORY)

### A. The Read-Process-Write Pattern

**CRITICAL: When consuming from Kafka, processing, and producing back to Kafka, use transactions to ensure atomicity.**

```
TRANSACTIONAL READ-PROCESS-WRITE:

┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐         │
│  │ Input Topic  │    │   Service    │    │ Output Topic │         │
│  │   (orders)   │───▶│  (processor) │───▶│  (invoices)  │         │
│  └──────────────┘    └──────────────┘    └──────────────┘         │
│         │                   │                    │                  │
│         │                   │                    │                  │
│         ▼                   ▼                    ▼                  │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    KAFKA TRANSACTION                         │   │
│  │                                                              │   │
│  │  1. Begin Transaction                                        │   │
│  │  2. Read message from input topic                           │   │
│  │  3. Process message (transform, enrich, etc.)               │   │
│  │  4. Write result to output topic                            │   │
│  │  5. Commit consumer offset for input message                │   │
│  │  6. Commit Transaction                                       │   │
│  │                                                              │   │
│  │  If ANY step fails:                                          │   │
│  │  - Abort Transaction                                         │   │
│  │  - Output NOT written                                        │   │
│  │  - Offset NOT committed                                      │   │
│  │  - Message will be reprocessed                              │   │
│  │                                                              │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

WHY THIS MATTERS:

Without Transaction:
  1. Read message (offset 5)
  2. Process message
  3. Write to output topic ✓
  4. Commit offset ← CRASH HERE
  5. Result: Output written, but offset not committed
  6. On restart: Message reprocessed, DUPLICATE output

With Transaction:
  1. Begin transaction
  2. Read message (offset 5)
  3. Process message
  4. Write to output topic (uncommitted)
  5. Commit offset (uncommitted)
  6. Commit transaction ← If crash before this: ALL rolled back
  7. Result: Either ALL committed or NONE committed
```

### B. Transaction Configuration

```
# ✅ CORRECT - Producer configuration for transactions

# Unique transactional ID (must be consistent across restarts)
transactional.id=order-processor-${instance-id}

# Enable idempotence (required for transactions)
enable.idempotence=true

# Wait for all replicas to acknowledge
acks=all

# Retry indefinitely (transaction will timeout if needed)
retries=2147483647

# Allow batching within transaction
max.in.flight.requests.per.connection=5

# Transaction timeout (must be less than broker's transaction.max.timeout.ms)
transaction.timeout.ms=60000

# ✅ CORRECT - Consumer configuration for transactions

# Read only committed messages
isolation.level=read_committed

# Disable auto-commit (we commit within transaction)
enable.auto.commit=false

# Consumer group for offset management
group.id=order-processor-group
```

### C. Transaction Implementation Pattern

```
TRANSACTIONAL PROCESSING PSEUDOCODE:

// ✅ CORRECT - Full transactional read-process-write

producer.initTransactions()

while (running) {
    // Poll for messages
    records = consumer.poll(timeout)

    if (records.isEmpty()) {
        continue
    }

    try {
        // 1. Begin transaction
        producer.beginTransaction()

        for (record in records) {
            // 2. Process message
            result = processMessage(record)

            // 3. Write result to output topic (within transaction)
            producer.send(outputTopic, result.key, result.value)
        }

        // 4. Commit consumer offsets within transaction
        // This is the KEY: offset commit is part of the transaction
        producer.sendOffsetsToTransaction(
            consumer.getOffsetsToCommit(),
            consumer.groupMetadata()
        )

        // 5. Commit transaction (atomic: output + offset)
        producer.commitTransaction()

    } catch (ProducerFencedException e) {
        // Another instance with same transactional.id took over
        // This instance must shut down
        shutdown()

    } catch (KafkaException e) {
        // Abort transaction - nothing committed
        producer.abortTransaction()

        // Reset consumer to last committed offset
        consumer.seekToCommittedOffsets()
    }
}

// ❌ WRONG - Non-transactional processing (can lose or duplicate)

while (running) {
    records = consumer.poll(timeout)

    for (record in records) {
        result = processMessage(record)
        producer.send(outputTopic, result)  // ❌ Not in transaction
    }

    consumer.commitSync()  // ❌ Separate from produce
}
```

---

## 4. State Recovery from Commit Log (MANDATORY)

### A. Event Sourcing with Kafka

**CRITICAL: Stateless services can recover their state by replaying the commit log from a known point.**

```
STATE RECOVERY PATTERN:

┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  KAFKA TOPIC (Commit Log)                                          │
│  ┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┐        │
│  │ 0 │ 1 │ 2 │ 3 │ 4 │ 5 │ 6 │ 7 │ 8 │ 9 │10 │11 │12 │13 │        │
│  └───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┘        │
│    │   │   │   │   │   │   │   │   │   │   │   │   │   │          │
│    │   │   │   │   │   │   │   │   │   │   │   │   │   │          │
│    ▼   ▼   ▼   ▼   ▼   ▼   ▼   ▼   ▼   ▼   ▼   ▼   ▼   ▼          │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    STATELESS SERVICE                         │   │
│  │                                                              │   │
│  │  On Startup:                                                 │   │
│  │  1. Check for snapshot (optional optimization)              │   │
│  │  2. Seek to snapshot offset (or beginning)                  │   │
│  │  3. Replay all events to rebuild state                      │   │
│  │  4. Continue processing new events                          │   │
│  │                                                              │   │
│  │  State = f(event₀, event₁, ..., eventₙ)                     │   │
│  │                                                              │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  RECOVERY SCENARIOS:                                                │
│                                                                     │
│  Crash at offset 8:                                                │
│    - Committed offset was 5                                        │
│    - On restart, seek to offset 5                                  │
│    - Replay events 5, 6, 7, 8, 9, ... to rebuild state            │
│    - Continue from where left off                                  │
│                                                                     │
│  Scale out (new instance):                                          │
│    - New consumer gets assigned partitions                         │
│    - Seek to beginning (or snapshot)                               │
│    - Replay entire partition to build state                        │
│    - Join as fully functional instance                             │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### B. Compacted Topics for State

```
LOG COMPACTION FOR STATE RECOVERY:

Regular Topic:
┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┐
│K:A│K:B│K:A│K:C│K:B│K:A│K:D│K:A│K:B│K:C│  All messages retained
│V:1│V:1│V:2│V:1│V:2│V:3│V:1│V:4│V:3│V:2│  (until retention expires)
└───┴───┴───┴───┴───┴───┴───┴───┴───┴───┘

Compacted Topic (after compaction):
┌───┬───┬───┬───┐
│K:A│K:B│K:C│K:D│  Only latest value per key retained
│V:4│V:3│V:2│V:1│  (forever, unless deleted)
└───┴───┴───┴───┘

USE CASES FOR COMPACTED TOPICS:

1. User Profiles
   Key: user-id
   Value: current profile state
   → Latest profile always available

2. Product Catalog
   Key: product-id
   Value: current product details
   → Catalog state recoverable

3. Account Balances
   Key: account-id
   Value: current balance
   → Latest balance per account

4. Configuration
   Key: config-key
   Value: current config value
   → Latest configuration state

COMPACTION CONFIGURATION:

# Topic configuration
cleanup.policy=compact           # Enable compaction
min.cleanable.dirty.ratio=0.5   # Compact when 50% is "dirty"
segment.ms=3600000              # New segment every hour
delete.retention.ms=86400000    # Keep tombstones for 24h
min.compaction.lag.ms=0         # No delay before compaction eligible
```

### C. Snapshot + Log Pattern

```
SNAPSHOT + COMMIT LOG PATTERN:

For faster recovery, periodically snapshot state and replay only from snapshot:

┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  Commit Log:                                                        │
│  ┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┐        │
│  │ 0 │ 1 │ 2 │ 3 │ 4 │ 5 │ 6 │ 7 │ 8 │ 9 │10 │11 │12 │13 │        │
│  └───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┘        │
│                      ▲                                   ▲          │
│                      │                                   │          │
│                  Snapshot                            Current        │
│                  (offset 5)                         (offset 13)     │
│                      │                                              │
│                      ▼                                              │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ Snapshot Storage (Object Store, Database, etc.)             │   │
│  │                                                              │   │
│  │ snapshot-partition-0-offset-5.bin                           │   │
│  │ Contains: Full state as of offset 5                         │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  RECOVERY PROCESS:                                                  │
│                                                                     │
│  1. Load latest snapshot (state at offset 5)                       │
│  2. Seek consumer to offset 6                                      │
│  3. Replay events 6, 7, 8, 9, 10, 11, 12, 13                      │
│  4. State fully recovered (8 events vs 14 without snapshot)       │
│                                                                     │
│  SNAPSHOT STRATEGY:                                                 │
│                                                                     │
│  - Snapshot every N messages (e.g., 10,000)                        │
│  - Snapshot every T time (e.g., hourly)                            │
│  - Snapshot on graceful shutdown                                   │
│  - Keep last M snapshots (e.g., 3)                                 │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### D. Retention Configuration for State Recovery

```
# ✅ CORRECT - Retention for event sourcing / state recovery

# Option 1: Time-based retention (keep 30 days of events)
retention.ms=2592000000          # 30 days in milliseconds
retention.bytes=-1               # No size limit

# Option 2: Infinite retention (keep forever)
retention.ms=-1                  # Never expire by time
retention.bytes=-1               # No size limit
cleanup.policy=delete            # Or 'compact' for key-based state

# Option 3: Compacted topic (keep latest per key forever)
cleanup.policy=compact
retention.ms=-1
min.cleanable.dirty.ratio=0.1   # Aggressive compaction

# ✅ CORRECT - Consumer configuration for replay

# Start from beginning for full state rebuild
auto.offset.reset=earliest

# Or seek to specific offset programmatically
consumer.seek(partition, specificOffset)

# Or seek to timestamp
consumer.offsetsForTimes(timestampToSearch)
```

---

## 5. Exactly-Once Semantics (EOS) (MANDATORY)

### A. End-to-End Exactly-Once

```
EXACTLY-ONCE SEMANTICS COMPONENTS:

┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  1. IDEMPOTENT PRODUCER                                            │
│     ┌─────────────────────────────────────────────────────────┐    │
│     │ Producer assigns sequence numbers to messages            │    │
│     │ Broker detects and deduplicates retries                  │    │
│     │ Guarantees: No duplicates from producer retries          │    │
│     │                                                          │    │
│     │ enable.idempotence=true                                  │    │
│     │ acks=all                                                 │    │
│     │ retries=MAX_INT                                          │    │
│     └─────────────────────────────────────────────────────────┘    │
│                          │                                          │
│                          ▼                                          │
│  2. TRANSACTIONAL PRODUCER                                         │
│     ┌─────────────────────────────────────────────────────────┐    │
│     │ Atomic writes across multiple partitions/topics          │    │
│     │ All-or-nothing commit semantics                          │    │
│     │ Consumer offset commit included in transaction           │    │
│     │                                                          │    │
│     │ transactional.id=unique-id                               │    │
│     │ producer.initTransactions()                              │    │
│     │ producer.beginTransaction()                              │    │
│     │ producer.commitTransaction()                             │    │
│     └─────────────────────────────────────────────────────────┘    │
│                          │                                          │
│                          ▼                                          │
│  3. TRANSACTIONAL CONSUMER                                         │
│     ┌─────────────────────────────────────────────────────────┐    │
│     │ Read only committed messages                             │    │
│     │ Skip aborted transaction messages                        │    │
│     │ Guarantees: Only see successfully committed data         │    │
│     │                                                          │    │
│     │ isolation.level=read_committed                           │    │
│     └─────────────────────────────────────────────────────────┘    │
│                                                                     │
│  RESULT: End-to-end exactly-once processing                        │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### B. Producer Configuration for EOS

```
# ✅ CORRECT - Exactly-once producer configuration

# === REQUIRED FOR EXACTLY-ONCE ===

# Unique identifier for this transactional producer
# Must be consistent across restarts for zombie fencing
transactional.id=payment-processor-instance-1

# Enable idempotent producer (automatic with transactions)
enable.idempotence=true

# Wait for all in-sync replicas to acknowledge
acks=all

# Retry forever (transaction timeout will handle failures)
retries=2147483647

# Maximum unacknowledged requests (5 is safe with idempotence)
max.in.flight.requests.per.connection=5

# === RECOMMENDED SETTINGS ===

# Transaction timeout (< broker's transaction.max.timeout.ms)
transaction.timeout.ms=60000

# Delivery timeout (includes retries)
delivery.timeout.ms=120000

# Linger for batching efficiency
linger.ms=5

# Batch size for throughput
batch.size=16384

# Compression for efficiency
compression.type=lz4
```

### C. Consumer Configuration for EOS

```
# ✅ CORRECT - Exactly-once consumer configuration

# === REQUIRED FOR EXACTLY-ONCE ===

# Only read committed (completed transaction) messages
isolation.level=read_committed

# Disable auto-commit (we commit in transaction)
enable.auto.commit=false

# Consumer group for offset tracking
group.id=payment-processor-group

# === RECOMMENDED SETTINGS ===

# Start from earliest if no committed offset
auto.offset.reset=earliest

# Fetch configuration for throughput
fetch.min.bytes=1
fetch.max.wait.ms=500
max.poll.records=500

# Session timeout for rebalance
session.timeout.ms=45000
heartbeat.interval.ms=15000

# Max poll interval (must complete processing within this)
max.poll.interval.ms=300000
```

### D. Transactional ID Management

```
TRANSACTIONAL ID STRATEGY:

The transactional.id MUST be:
  1. Unique per producer instance
  2. Consistent across restarts of same instance
  3. Used for "zombie fencing" (preventing duplicate producers)

┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  ZOMBIE FENCING:                                                    │
│                                                                     │
│  Scenario: Instance crashes, restarts, but old instance still      │
│            running (network partition, slow GC, etc.)              │
│                                                                     │
│  ┌─────────────────┐         ┌─────────────────┐                   │
│  │ Old Instance    │         │ New Instance    │                   │
│  │ txn.id=proc-1   │         │ txn.id=proc-1   │                   │
│  │ epoch=1         │         │ epoch=2         │                   │
│  └────────┬────────┘         └────────┬────────┘                   │
│           │                           │                             │
│           │ Tries to produce          │ initTransactions()         │
│           │                           │ (gets new epoch)           │
│           ▼                           ▼                             │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                       KAFKA BROKER                           │   │
│  │                                                              │   │
│  │  "txn.id=proc-1 with epoch=1 is FENCED"                     │   │
│  │  Old instance gets ProducerFencedException                   │   │
│  │  Must shut down and not produce                             │   │
│  │                                                              │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

TRANSACTIONAL ID PATTERNS:

# Pattern 1: Instance-based (for stateful services)
transactional.id=order-processor-${HOSTNAME}
# Each instance has unique, consistent ID

# Pattern 2: Partition-based (for partition-aware processing)
transactional.id=order-processor-partition-${PARTITION}
# One producer per partition assignment

# Pattern 3: Static assignment
transactional.id=order-processor-1
transactional.id=order-processor-2
transactional.id=order-processor-3
# Fixed pool of IDs, instances claim one on startup
```

---

## 6. Consumer Group Management (MANDATORY)

### A. Consumer Group Fundamentals

```
CONSUMER GROUP ARCHITECTURE:

┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  Topic: orders (6 partitions)                                      │
│  ┌─────┬─────┬─────┬─────┬─────┬─────┐                            │
│  │ P0  │ P1  │ P2  │ P3  │ P4  │ P5  │                            │
│  └──┬──┴──┬──┴──┬──┴──┬──┴──┬──┴──┬──┘                            │
│     │     │     │     │     │     │                                │
│     │     │     │     │     │     │                                │
│     ▼     ▼     ▼     ▼     ▼     ▼                                │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │              Consumer Group: order-processors                │   │
│  │                                                              │   │
│  │  ┌───────────┐  ┌───────────┐  ┌───────────┐                │   │
│  │  │Consumer 1 │  │Consumer 2 │  │Consumer 3 │                │   │
│  │  │  P0, P1   │  │  P2, P3   │  │  P4, P5   │                │   │
│  │  └───────────┘  └───────────┘  └───────────┘                │   │
│  │                                                              │   │
│  │  Each partition assigned to exactly ONE consumer             │   │
│  │  Each consumer can have multiple partitions                  │   │
│  │                                                              │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  SCALING:                                                          │
│  - Add consumers: Partitions rebalanced                           │
│  - Max consumers = Number of partitions                           │
│  - Extra consumers sit idle                                       │
│                                                                     │
│  ORDERING:                                                         │
│  - Order guaranteed WITHIN partition                              │
│  - No order guarantee ACROSS partitions                           │
│  - Use same key for related messages → same partition            │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### B. Rebalance Handling

```
REBALANCE SCENARIOS:

Triggers:
  - Consumer joins group
  - Consumer leaves group (shutdown, crash, timeout)
  - Topic partition count changes
  - Consumer subscription changes

┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  COOPERATIVE STICKY REBALANCE (Recommended):                       │
│                                                                     │
│  Before: Consumer 1 has P0, P1, P2                                 │
│          Consumer 2 has P3, P4, P5                                 │
│                                                                     │
│  Consumer 3 joins...                                               │
│                                                                     │
│  Rebalance 1: Consumer 1 releases P2                               │
│               Consumer 2 releases P5                               │
│               (Consumers 1 & 2 continue processing P0,P1 and P3,P4)│
│                                                                     │
│  Rebalance 2: Consumer 3 gets P2, P5                               │
│                                                                     │
│  After:  Consumer 1 has P0, P1                                     │
│          Consumer 2 has P3, P4                                     │
│          Consumer 3 has P2, P5                                     │
│                                                                     │
│  BENEFIT: Minimal disruption, most partitions continue processing  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

REBALANCE LISTENER:

// Handle partition assignment changes
consumer.subscribe(topics, new ConsumerRebalanceListener() {

    onPartitionsRevoked(partitions) {
        // Called BEFORE partitions taken away
        // MUST: Commit offsets for partitions being revoked
        // MUST: Flush any pending work
        // MUST: Save state/checkpoint if stateful

        commitOffsetsForPartitions(partitions)
        flushPendingWrites()
        saveStateSnapshot(partitions)
    }

    onPartitionsAssigned(partitions) {
        // Called AFTER new partitions assigned
        // SHOULD: Load state for new partitions
        // SHOULD: Seek to appropriate offset if needed

        for (partition in partitions) {
            state = loadStateSnapshot(partition)
            if (state != null) {
                consumer.seek(partition, state.offset)
            }
        }
    }
})
```

### C. Consumer Configuration

```
# ✅ CORRECT - Production consumer configuration

# Consumer group identification
group.id=order-processor-group
client.id=order-processor-${HOSTNAME}

# Use cooperative rebalancing (recommended)
partition.assignment.strategy=org.apache.kafka.clients.consumer.CooperativeStickyAssignor

# Session management
session.timeout.ms=45000       # Max time before considered dead
heartbeat.interval.ms=15000    # Heartbeat frequency (1/3 of session timeout)

# Processing timeout
max.poll.interval.ms=300000    # Max time between polls (5 min)
max.poll.records=500           # Max records per poll

# Offset management
enable.auto.commit=false       # Manual commit for reliability
auto.offset.reset=earliest     # Start from beginning if no offset

# Fetch tuning
fetch.min.bytes=1              # Minimum bytes to fetch
fetch.max.wait.ms=500          # Max wait for fetch.min.bytes
fetch.max.bytes=52428800       # Max bytes per fetch (50MB)
max.partition.fetch.bytes=1048576  # Max per partition (1MB)

# For exactly-once
isolation.level=read_committed
```

---

## 7. Partitioning Strategy (MANDATORY)

### A. Key-Based Partitioning

```
PARTITIONING STRATEGY:

┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  RULE: Same key → Same partition → Ordered processing              │
│                                                                     │
│  Producer sends:                                                    │
│  ┌──────────────┬───────────────────┐                              │
│  │ Key          │ Partition (hash)  │                              │
│  ├──────────────┼───────────────────┤                              │
│  │ customer-123 │ hash % 6 = 2      │                              │
│  │ customer-123 │ hash % 6 = 2      │ ← Same partition            │
│  │ customer-456 │ hash % 6 = 5      │                              │
│  │ customer-789 │ hash % 6 = 0      │                              │
│  │ customer-123 │ hash % 6 = 2      │ ← Same partition            │
│  └──────────────┴───────────────────┘                              │
│                                                                     │
│  ORDERING GUARANTEE:                                                │
│  All messages with key "customer-123" go to partition 2            │
│  Consumer of partition 2 sees them in order                        │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

KEY SELECTION GUIDELINES:

✅ GOOD KEYS:
  - Customer ID (orders for same customer in order)
  - Order ID (all events for same order in order)
  - Account ID (transactions for same account in order)
  - Session ID (events in same session in order)

❌ BAD KEYS:
  - Timestamp (no ordering benefit, hot partitions)
  - Random UUID (no ordering benefit)
  - null (round-robin, no ordering)
  - Very low cardinality (hot partitions)

KEY CARDINALITY:
  - Too few unique keys: Uneven partition distribution
  - Too many unique keys: Better distribution
  - Rule of thumb: At least 10x more unique keys than partitions
```

### B. Partition Count Planning

```
PARTITION COUNT GUIDELINES:

┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  Factors to Consider:                                               │
│                                                                     │
│  1. THROUGHPUT                                                      │
│     - Each partition: ~10 MB/s write, ~30 MB/s read (typical)      │
│     - Total throughput = Partitions × Per-partition throughput     │
│     - Target: 100 MB/s → Need at least 10 partitions              │
│                                                                     │
│  2. CONSUMER PARALLELISM                                            │
│     - Max consumers = Number of partitions                         │
│     - 12 partitions = Max 12 parallel consumers                    │
│     - Plan for future scaling needs                                │
│                                                                     │
│  3. ORDERING REQUIREMENTS                                           │
│     - More partitions = Less strict global ordering                │
│     - Ordering only within partition                               │
│                                                                     │
│  4. BROKER RESOURCES                                                │
│     - Each partition = Memory, file handles, CPU                   │
│     - Recommended: <4000 partitions per broker                     │
│                                                                     │
│  FORMULA:                                                           │
│  Partitions = max(                                                  │
│      Throughput_Target / Throughput_Per_Partition,                 │
│      Max_Consumer_Count_Needed                                      │
│  )                                                                  │
│                                                                     │
│  RECOMMENDATIONS:                                                   │
│  - Start with: 6-12 partitions for most topics                     │
│  - High throughput: 30-100 partitions                              │
│  - Can increase later (but cannot decrease!)                       │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

# ✅ CORRECT - Topic creation with appropriate partitions
kafka-topics.sh --create \
  --topic orders \
  --partitions 12 \
  --replication-factor 3 \
  --config retention.ms=604800000 \
  --config min.insync.replicas=2
```

---

## 8. Error Handling and Dead Letter Queues (MANDATORY)

### A. Error Handling Strategy

```
ERROR HANDLING PATTERN:

┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  Message Processing Flow with Error Handling:                       │
│                                                                     │
│  ┌─────────────┐                                                   │
│  │ Input Topic │                                                   │
│  └──────┬──────┘                                                   │
│         │                                                          │
│         ▼                                                          │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    CONSUMER                                  │   │
│  │                                                              │   │
│  │  for (record in records) {                                  │   │
│  │      try {                                                   │   │
│  │          result = process(record)                           │   │
│  │          ──────────────────────────────────────────────▶ Output │
│  │      }                                                       │   │
│  │      catch (RetryableException e) {                         │   │
│  │          // Transient error: retry with backoff             │   │
│  │          retryWithBackoff(record, e)                        │   │
│  │      }                                                       │   │
│  │      catch (NonRetryableException e) {                      │   │
│  │          // Permanent error: send to DLQ                    │   │
│  │          sendToDLQ(record, e) ─────────────────────────▶ DLQ    │
│  │      }                                                       │   │
│  │  }                                                           │   │
│  │                                                              │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

ERROR CLASSIFICATION:

Retryable Errors (temporary):
  - Network timeout
  - Database connection lost
  - Rate limit exceeded
  - Downstream service unavailable
  → Retry with exponential backoff

Non-Retryable Errors (permanent):
  - Invalid message format
  - Business rule violation
  - Missing required data
  - Schema validation failure
  → Send to Dead Letter Queue
```

### B. Dead Letter Queue Pattern

```
DEAD LETTER QUEUE (DLQ) IMPLEMENTATION:

┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  DLQ Message Structure:                                            │
│                                                                     │
│  {                                                                  │
│    "originalTopic": "orders",                                      │
│    "originalPartition": 3,                                         │
│    "originalOffset": 12345,                                        │
│    "originalKey": "order-789",                                     │
│    "originalTimestamp": "2024-01-15T10:30:00Z",                   │
│    "originalHeaders": { ... },                                     │
│    "originalValue": { ... base64 or JSON ... },                   │
│                                                                     │
│    "error": {                                                       │
│      "type": "ValidationException",                                │
│      "message": "Invalid product ID",                              │
│      "stackTrace": "...",                                          │
│      "timestamp": "2024-01-15T10:30:05Z"                          │
│    },                                                               │
│                                                                     │
│    "processingAttempts": 3,                                        │
│    "lastAttemptTimestamp": "2024-01-15T10:30:05Z",                │
│    "consumerGroup": "order-processor-group",                       │
│    "consumerInstance": "order-processor-1"                         │
│  }                                                                  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

DLQ TOPIC NAMING:
  Original topic: orders
  DLQ topic: orders.dlq (or orders-dlq, orders.dead-letter)

DLQ HANDLING:
  1. Alert on DLQ messages (monitoring)
  2. Manual inspection and triage
  3. Fix and replay if possible
  4. Archive after resolution
```

### C. Retry Pattern Implementation

```
RETRY WITH EXPONENTIAL BACKOFF:

┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  Retry Pattern:                                                     │
│                                                                     │
│  Attempt 1: Process immediately                                    │
│       ↓ (fail)                                                      │
│  Attempt 2: Wait 1 second, retry                                   │
│       ↓ (fail)                                                      │
│  Attempt 3: Wait 2 seconds, retry                                  │
│       ↓ (fail)                                                      │
│  Attempt 4: Wait 4 seconds, retry                                  │
│       ↓ (fail)                                                      │
│  Attempt 5: Wait 8 seconds, retry                                  │
│       ↓ (fail)                                                      │
│  Max retries exceeded: Send to DLQ                                 │
│                                                                     │
│  FORMULA:                                                           │
│  delay = min(initialDelay * 2^attempt, maxDelay) + jitter         │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

RETRY IMPLEMENTATION:

processWithRetry(record) {
    maxRetries = 5
    initialDelay = 1000  // 1 second
    maxDelay = 30000     // 30 seconds

    for (attempt = 1; attempt <= maxRetries; attempt++) {
        try {
            return process(record)
        }
        catch (RetryableException e) {
            if (attempt == maxRetries) {
                sendToDLQ(record, e)
                return
            }

            delay = min(initialDelay * pow(2, attempt - 1), maxDelay)
            jitter = random(0, delay * 0.1)  // 10% jitter
            sleep(delay + jitter)
        }
        catch (NonRetryableException e) {
            sendToDLQ(record, e)
            return
        }
    }
}
```

---

## 9. Schema Management (MANDATORY)

### A. Schema Registry Pattern

```
SCHEMA REGISTRY ARCHITECTURE:

┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  ┌─────────────┐         ┌─────────────────┐                       │
│  │  Producer   │────────▶│ Schema Registry │                       │
│  └─────────────┘         └────────┬────────┘                       │
│        │                          │                                 │
│        │ 1. Register schema       │ 2. Return schema ID            │
│        │    (if new)              │                                 │
│        │                          │                                 │
│        ▼                          │                                 │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  MESSAGE FORMAT:                                              │  │
│  │  ┌─────────┬──────────────────────────────────────────────┐  │  │
│  │  │ Magic   │ Schema ID (4 bytes) │ Serialized Data        │  │  │
│  │  │ Byte    │                     │ (Avro/Protobuf/JSON)   │  │  │
│  │  │ (0x0)   │                     │                        │  │  │
│  │  └─────────┴──────────────────────────────────────────────┘  │  │
│  └──────────────────────────────────────────────────────────────┘  │
│        │                                                           │
│        ▼                                                           │
│  ┌─────────────┐                                                   │
│  │   Kafka     │                                                   │
│  │   Broker    │                                                   │
│  └─────────────┘                                                   │
│        │                                                           │
│        ▼                                                           │
│  ┌─────────────┐         ┌─────────────────┐                       │
│  │  Consumer   │────────▶│ Schema Registry │                       │
│  └─────────────┘         └─────────────────┘                       │
│        │                          │                                 │
│        │ 3. Read schema ID        │ 4. Return schema               │
│        │    from message          │    (cached locally)            │
│        │                          │                                 │
│        ▼                          │                                 │
│  Deserialize using schema                                          │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### B. Schema Evolution

```
SCHEMA COMPATIBILITY MODES:

┌─────────────────────────────────────────────────────────────────────┐
│ Mode              │ Description                                     │
├─────────────────────────────────────────────────────────────────────┤
│ BACKWARD          │ New schema can read old data                    │
│                   │ (Add optional fields, remove fields)           │
│                   │ Consumers upgraded FIRST                        │
├─────────────────────────────────────────────────────────────────────┤
│ FORWARD           │ Old schema can read new data                    │
│                   │ (Remove optional fields, add fields)           │
│                   │ Producers upgraded FIRST                        │
├─────────────────────────────────────────────────────────────────────┤
│ FULL              │ Both backward AND forward compatible            │
│                   │ (Only add/remove optional fields)              │
│                   │ ✅ RECOMMENDED for production                   │
├─────────────────────────────────────────────────────────────────────┤
│ NONE              │ No compatibility checking                       │
│                   │ ❌ NOT RECOMMENDED                               │
└─────────────────────────────────────────────────────────────────────┘

SCHEMA EVOLUTION RULES (for FULL compatibility):

✅ ALLOWED:
  - Add a new optional field (with default)
  - Remove an optional field
  - Add new enum values (at end)

❌ NOT ALLOWED:
  - Add a required field (without default)
  - Remove a required field
  - Change field type
  - Rename a field
```

### C. Schema Definition Example

```
// ✅ CORRECT - Avro schema with evolution support

{
  "type": "record",
  "name": "Order",
  "namespace": "com.company.orders",
  "doc": "Order event schema",
  "fields": [
    {
      "name": "orderId",
      "type": "string",
      "doc": "Unique order identifier"
    },
    {
      "name": "customerId",
      "type": "string",
      "doc": "Customer who placed the order"
    },
    {
      "name": "items",
      "type": {
        "type": "array",
        "items": {
          "type": "record",
          "name": "OrderItem",
          "fields": [
            {"name": "productId", "type": "string"},
            {"name": "quantity", "type": "int"},
            {"name": "price", "type": "double"}
          ]
        }
      }
    },
    {
      "name": "totalAmount",
      "type": "double"
    },
    {
      "name": "currency",
      "type": "string",
      "default": "USD"  // Default for backward compatibility
    },
    {
      "name": "metadata",
      "type": ["null", "map"],  // Optional field
      "default": null
    }
  ]
}
```

---

## 10. Performance Tuning (MANDATORY)

### A. Producer Tuning

```
# ✅ CORRECT - High-throughput producer configuration

# === BATCHING ===
# Batch messages for efficiency
batch.size=65536                    # 64KB batch
linger.ms=10                        # Wait up to 10ms to fill batch

# === COMPRESSION ===
# Compress batches (LZ4 recommended for speed)
compression.type=lz4

# === MEMORY ===
# Buffer memory for batching
buffer.memory=67108864              # 64MB total buffer

# === ACKNOWLEDGMENT ===
# For durability (required for EOS)
acks=all

# === RETRIES ===
retries=2147483647
retry.backoff.ms=100
delivery.timeout.ms=120000          # 2 minutes total

# === PARALLELISM ===
max.in.flight.requests.per.connection=5  # With idempotence

# === THROUGHPUT VS LATENCY ===

# High throughput (batch more):
linger.ms=50
batch.size=131072

# Low latency (batch less):
linger.ms=0
batch.size=16384
```

### B. Consumer Tuning

```
# ✅ CORRECT - High-throughput consumer configuration

# === FETCH SETTINGS ===
fetch.min.bytes=1024                # Min 1KB per fetch
fetch.max.wait.ms=500               # Wait up to 500ms
fetch.max.bytes=52428800            # 50MB max per fetch request
max.partition.fetch.bytes=1048576   # 1MB max per partition

# === POLL SETTINGS ===
max.poll.records=1000               # Records per poll
max.poll.interval.ms=300000         # 5 min max processing time

# === SESSION MANAGEMENT ===
session.timeout.ms=45000
heartbeat.interval.ms=15000

# === OFFSET MANAGEMENT ===
enable.auto.commit=false            # Manual commit
auto.offset.reset=earliest

# === THROUGHPUT VS LATENCY ===

# High throughput (larger batches):
fetch.min.bytes=1048576             # Wait for 1MB
fetch.max.wait.ms=1000              # Wait up to 1s
max.poll.records=5000

# Low latency (smaller batches):
fetch.min.bytes=1
fetch.max.wait.ms=100
max.poll.records=100
```

### C. Broker Tuning

```
# ✅ CORRECT - Production broker configuration

# === REPLICATION ===
default.replication.factor=3
min.insync.replicas=2
unclean.leader.election.enable=false

# === LOG SETTINGS ===
log.retention.hours=168             # 7 days retention
log.retention.bytes=-1              # No size limit
log.segment.bytes=1073741824        # 1GB segment size
log.cleanup.policy=delete           # or compact

# === PERFORMANCE ===
num.network.threads=8
num.io.threads=16
socket.send.buffer.bytes=102400
socket.receive.buffer.bytes=102400
socket.request.max.bytes=104857600  # 100MB

# === DURABILITY ===
log.flush.interval.messages=10000
log.flush.interval.ms=1000

# === QUOTAS ===
quota.producer.default=10485760     # 10MB/s per producer
quota.consumer.default=52428800     # 50MB/s per consumer
```

---

## 11. Monitoring and Observability (MANDATORY)

### A. Critical Metrics

```
ESSENTIAL KAFKA METRICS:

┌─────────────────────────────────────────────────────────────────────┐
│                         CONSUMER METRICS                            │
├─────────────────────────────────────────────────────────────────────┤
│ Metric                        │ Alert Threshold    │ Meaning        │
├───────────────────────────────┼────────────────────┼────────────────┤
│ consumer_lag                  │ > 10000 messages   │ Falling behind │
│ records_consumed_rate         │ Varies             │ Throughput     │
│ commit_latency_avg            │ > 100ms            │ Slow commits   │
│ rebalance_rate                │ > 0.1/min          │ Unstable group │
│ failed_rebalances             │ > 0                │ Group issues   │
│ poll_idle_ratio               │ < 0.5              │ Slow processing│
└───────────────────────────────┴────────────────────┴────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                         PRODUCER METRICS                            │
├─────────────────────────────────────────────────────────────────────┤
│ Metric                        │ Alert Threshold    │ Meaning        │
├───────────────────────────────┼────────────────────┼────────────────┤
│ record_send_rate              │ Varies             │ Throughput     │
│ record_error_rate             │ > 0                │ Send failures  │
│ request_latency_avg           │ > 100ms            │ Slow sends     │
│ batch_size_avg                │ < batch.size * 0.8 │ Inefficient    │
│ buffer_available_bytes        │ < 10MB             │ Backpressure   │
│ bufferpool_wait_time          │ > 0                │ Memory pressure│
└───────────────────────────────┴────────────────────┴────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                         BROKER METRICS                              │
├─────────────────────────────────────────────────────────────────────┤
│ Metric                        │ Alert Threshold    │ Meaning        │
├───────────────────────────────┼────────────────────┼────────────────┤
│ under_replicated_partitions   │ > 0                │ Replication lag│
│ offline_partitions            │ > 0                │ Data unavail.  │
│ active_controller_count       │ != 1               │ No controller  │
│ unclean_leader_elections      │ > 0                │ Data loss risk │
│ request_queue_size            │ > 100              │ Overloaded     │
│ network_processor_idle        │ < 0.3              │ Network bottl. │
│ request_handler_idle          │ < 0.3              │ CPU bottleneck │
└───────────────────────────────┴────────────────────┴────────────────┘
```

### B. Consumer Lag Monitoring

```
CONSUMER LAG MONITORING:

Consumer Lag = Latest Offset - Committed Offset

┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  Topic: orders, Partition: 0                                       │
│                                                                     │
│  Latest Offset:    1,000,000                                       │
│  Committed Offset:   990,000                                       │
│  Consumer Lag:        10,000 messages                              │
│                                                                     │
│  LAG INTERPRETATION:                                                │
│                                                                     │
│  Lag = 0           │ Consumer caught up, real-time processing      │
│  Lag < 1000        │ Normal, healthy                               │
│  Lag 1000-10000    │ Monitor closely, may need scaling             │
│  Lag > 10000       │ Alert! Consumer falling behind                │
│  Lag growing       │ Critical! Production > Consumption rate       │
│                                                                     │
│  CAUSES OF LAG:                                                     │
│  - Slow message processing                                         │
│  - Not enough consumer instances                                   │
│  - Consumer crashes/restarts                                       │
│  - Sudden traffic spike                                            │
│  - Downstream dependencies slow                                    │
│                                                                     │
│  SOLUTIONS:                                                         │
│  - Scale up consumers (add instances)                              │
│  - Optimize processing logic                                       │
│  - Increase partitions (and consumers)                             │
│  - Add backpressure to producers                                   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### C. Alerting Rules

```yaml
# ✅ CORRECT - Prometheus alerting rules for Kafka

groups:
  - name: kafka-alerts
    rules:
      # Consumer lag alert
      - alert: KafkaConsumerLagHigh
        expr: |
          kafka_consumer_group_lag > 10000
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High consumer lag for {{ $labels.group }}"
          description: "Consumer group {{ $labels.group }} has lag of {{ $value }} on topic {{ $labels.topic }}"

      # Consumer lag critical
      - alert: KafkaConsumerLagCritical
        expr: |
          kafka_consumer_group_lag > 100000
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Critical consumer lag for {{ $labels.group }}"

      # Under-replicated partitions
      - alert: KafkaUnderReplicatedPartitions
        expr: |
          kafka_server_replicamanager_underreplicatedpartitions > 0
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Under-replicated partitions on {{ $labels.instance }}"

      # Offline partitions
      - alert: KafkaOfflinePartitions
        expr: |
          kafka_controller_kafkacontroller_offlinepartitionscount > 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Offline partitions detected"

      # Producer error rate
      - alert: KafkaProducerErrorRate
        expr: |
          rate(kafka_producer_record_error_total[5m]) > 0
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Producer errors for {{ $labels.client_id }}"

      # Dead letter queue messages
      - alert: KafkaDLQMessages
        expr: |
          kafka_topic_partition_current_offset{topic=~".*dlq.*"} > 0
        for: 1m
        labels:
          severity: warning
        annotations:
          summary: "Messages in DLQ topic {{ $labels.topic }}"
```

---

## 12. Testing Strategies (MANDATORY)

### A. Testing Levels

```
KAFKA TESTING PYRAMID:

                      ┌─────────────┐
                     /  Production   \             Chaos testing
                    /   Validation    \            Canary deploys
                   /───────────────────\
                  /    Integration      \          Testcontainers
                 /      Tests            \         Embedded Kafka
                /─────────────────────────\
               /        Unit Tests         \       Mocked producers
              /                             \      Mocked consumers
             /───────────────────────────────\
```

### B. Integration Testing Pattern

```
INTEGRATION TEST SETUP:

// ✅ CORRECT - Integration test with embedded Kafka

describe("Order Processor Integration") {

    beforeAll {
        // Start embedded Kafka (Testcontainers)
        kafka = startKafkaContainer()

        // Create test topics
        createTopic("orders-test", partitions=3)
        createTopic("invoices-test", partitions=3)
        createTopic("orders-test.dlq", partitions=1)

        // Configure processor to use test Kafka
        processor = OrderProcessor(
            bootstrapServers: kafka.bootstrapServers,
            inputTopic: "orders-test",
            outputTopic: "invoices-test",
            dlqTopic: "orders-test.dlq"
        )
        processor.start()
    }

    afterAll {
        processor.stop()
        kafka.stop()
    }

    test("processes valid order and produces invoice") {
        // Given
        order = createValidOrder()

        // When
        producer.send("orders-test", order.id, order)

        // Then
        invoice = consumer.poll("invoices-test", timeout=10s)
        expect(invoice).notNull()
        expect(invoice.orderId).equals(order.id)
    }

    test("sends invalid order to DLQ") {
        // Given
        invalidOrder = createInvalidOrder()

        // When
        producer.send("orders-test", invalidOrder.id, invalidOrder)

        // Then
        dlqMessage = consumer.poll("orders-test.dlq", timeout=10s)
        expect(dlqMessage).notNull()
        expect(dlqMessage.error).contains("ValidationException")
    }

    test("exactly-once: no duplicates on producer retry") {
        // Given
        order = createValidOrder()

        // When - simulate network issues causing retries
        producer.send("orders-test", order.id, order)
        producer.send("orders-test", order.id, order)  // Duplicate

        // Then - only one invoice produced
        invoices = consumer.pollAll("invoices-test", timeout=10s)
        invoicesForOrder = invoices.filter(i -> i.orderId == order.id)
        expect(invoicesForOrder.size).equals(1)  // Exactly one
    }

    test("recovery: resumes from last committed offset after restart") {
        // Given
        orders = [createOrder(), createOrder(), createOrder()]
        orders.forEach(o -> producer.send("orders-test", o.id, o))

        // Process first two
        processor.processNext()  // order 0
        processor.processNext()  // order 1
        processor.commitOffsets()

        // Simulate crash
        processor.stop()

        // When - restart
        processor.start()
        processor.processNext()

        // Then - processes order 2 (not 0 or 1 again)
        invoice = consumer.poll("invoices-test")
        expect(invoice.orderId).equals(orders[2].id)
    }
}
```

---

## 13. Common Anti-Patterns (PROHIBITED)

### A. Configuration Anti-Patterns

```
❌ PROHIBITED - Dangerous configurations

# Auto-commit with at-least-once expectations
enable.auto.commit=true  # ❌ Can lose messages on crash

# No acknowledgment
acks=0  # ❌ Fire and forget, messages can be lost

# Single replica
replication.factor=1  # ❌ No fault tolerance

# Large poll without timeout consideration
max.poll.records=10000
max.poll.interval.ms=30000  # ❌ Will timeout processing

# Missing transactional ID for exactly-once
enable.idempotence=true
# transactional.id not set  # ❌ Not truly exactly-once

✅ CORRECT - Safe configurations

enable.auto.commit=false
acks=all
replication.factor=3
min.insync.replicas=2
max.poll.records=500
max.poll.interval.ms=300000
transactional.id=processor-instance-1
```

### B. Processing Anti-Patterns

```
❌ PROHIBITED - Non-atomic read-process-write

// Commit BEFORE processing complete
consumer.poll()
consumer.commitSync()  // ❌ Message may not be processed yet
process(records)       // If crash here, message lost

// Commit input BEFORE output confirmed
consumer.poll()
process(records)
producer.send(output)  // ❌ Not waiting for confirmation
consumer.commitSync()  // If send fails, message lost but committed

// No transaction for read-process-write
consumer.poll()
result = process(records)
producer.send(output)
producer.flush()       // ❌ Not atomic with offset commit
consumer.commitSync()  // If crash between these, duplicate or lost

✅ CORRECT - Atomic transactional processing

producer.beginTransaction()
records = consumer.poll()
result = process(records)
producer.send(output, result)
producer.sendOffsetsToTransaction(offsets, consumerGroupMetadata)
producer.commitTransaction()  // ✅ All or nothing
```

### C. Design Anti-Patterns

```
❌ PROHIBITED - Anti-patterns

// Using Kafka as a database
// Kafka is a commit log, not a query database
SELECT * FROM kafka WHERE orderId = '123'  // ❌ Not how Kafka works

// Relying on global ordering
// Only partition ordering is guaranteed
messages across partitions not ordered  // ❌ Don't expect global order

// Very large messages
messageSize = 10MB  // ❌ Kafka optimized for small messages (<1MB)

// Too many partitions
partitions = 10000  // ❌ Each partition has overhead

// Single partition for ordering
partitions = 1  // ❌ Limits parallelism to 1 consumer

✅ CORRECT - Best practices

// Use Kafka for streaming, external DB for queries
// Partition by key for ordering within entity
// Keep messages small (<1MB), use references for large data
// Balance partitions: 6-100 typically
// Use keys strategically for ordering + parallelism
```

---

## 14. Verification Checklist (MANDATORY)

### A. Pre-Production Checklist

```
KAFKA DEPLOYMENT VERIFICATION:

□ Cluster Configuration
  □ Replication factor >= 3
  □ min.insync.replicas >= 2
  □ unclean.leader.election.enable = false
  □ Appropriate retention configured
  □ Quotas configured

□ Producer Configuration
  □ acks = all
  □ enable.idempotence = true
  □ transactional.id set (if using transactions)
  □ Retries configured appropriately
  □ Compression enabled

□ Consumer Configuration
  □ enable.auto.commit = false
  □ isolation.level = read_committed (if using transactions)
  □ Appropriate session/poll timeouts
  □ Consumer group named meaningfully

□ Topic Configuration
  □ Appropriate partition count
  □ Key strategy defined
  □ Schema registered
  □ Retention matches use case
  □ DLQ topic created

□ Transactional Processing
  □ Read-process-write in single transaction
  □ Offset commit within transaction
  □ Error handling with DLQ
  □ Idempotent processing logic

□ Monitoring
  □ Consumer lag alerting
  □ Error rate alerting
  □ Under-replicated partition alerting
  □ DLQ message alerting
  □ Dashboards configured

□ Testing
  □ Unit tests pass
  □ Integration tests with embedded Kafka
  □ Failure scenario tests
  □ Performance tests
```

### B. Operational Checklist

```
OPERATIONAL VERIFICATION:

□ Regular Health Checks
  □ No under-replicated partitions
  □ No offline partitions
  □ Consumer lag within bounds
  □ No messages in DLQ (or processed)
  □ Producer/consumer error rates = 0

□ Capacity Planning
  □ Disk usage trending
  □ Network utilization
  □ Partition count adequate
  □ Consumer instance count adequate

□ Disaster Recovery
  □ Backup/restore tested
  □ Multi-DC replication (if required)
  □ Recovery time objective (RTO) met
  □ Recovery point objective (RPO) met
```

---

## 15. Summary

### Core Principles

1. **Commit log as source of truth**: Immutable, append-only, replayable
2. **Exactly-once semantics**: Transactions for read-process-write
3. **State from log**: Stateless services recover by replaying events
4. **Offset = checkpoint**: Commit only after successful processing
5. **Idempotent everything**: Design for duplicate handling

### Key Configurations

| Setting | Value | Purpose |
|---------|-------|---------|
| `acks` | `all` | Durability |
| `enable.idempotence` | `true` | No duplicates from retries |
| `transactional.id` | `unique-per-instance` | Exactly-once transactions |
| `isolation.level` | `read_committed` | See only committed data |
| `enable.auto.commit` | `false` | Manual offset control |
| `replication.factor` | `3` | Fault tolerance |
| `min.insync.replicas` | `2` | Write durability |

### Processing Pattern

```
producer.initTransactions()

while (running) {
    producer.beginTransaction()

    records = consumer.poll()
    results = process(records)

    producer.send(outputTopic, results)
    producer.sendOffsetsToTransaction(offsets, groupMetadata)

    producer.commitTransaction()  // Atomic: output + offset
}
```

### Remember

> "The offset is your checkpoint. Never commit it until you're certain the message was successfully processed AND any downstream effects are durable."

> "Kafka is a commit log, not a message queue. Design for replay, not for fire-and-forget."

> "Exactly-once requires transactions. Without them, you have at-most-once or at-least-once, but never exactly-once."

> "Stateless services aren't stateless - they recover state from the commit log. Design your events to support full state reconstruction."

---

## Related Guides

- **[microservices.md](microservices.md)**: Microservices architecture patterns for event-driven systems
- **[kubernetes.md](kubernetes.md)**: Kubernetes deployment for Kafka consumers and producers
- **[istio.md](istio.md)**: Service mesh for Kafka-based microservices
- **[hexagonal.md](hexagonal.md)**: Hexagonal Architecture - structuring Kafka consumers/producers as adapters
