# WebSocket Development Guidelines
Mandatory standards for real-time bidirectional comms over WebSocket: connection lifecycle, heartbeats, reconnection/backoff, framing, backpressure, and pub/sub fan-out scaling. RFC 6455, ws (Node), gorilla/websocket (Go), Django Channels, Socket.IO.

---
name: websocket
title: WebSocket Development Guidelines
version: 2.0
last_reviewed: 2026-06-05
kind: cross-cutting
tools: [rfc6455, ws@8, gorilla-websocket, django-channels@4, socket.io@4]
requires: []
recommends:
  - secure-coding
  - oauth
  - error-handling
  - observability
  - rest
provides:
  - ws-lifecycle
  - heartbeats
  - reconnection
  - backpressure
  - realtime-scaling
---

> 🧭 Authored per [`CONVENTIONS.md`](guides://CONVENTIONS.md): shared concerns are referenced, not restated. This guide owns WebSocket / real-time bidirectional communication and spends its tokens on the protocol's unique surface.

---

## 0. Prerequisites & References

This guide has no hard prerequisites, but real-world WebSocket work almost always crosses these owners. Fetch them when the task touches their concern; do not restate their rules here.

> 📎 **RECOMMENDED — fetch when the task touches them:**
> - [`secure-coding.md`](guides://secure-coding.md) — input validation, secrets, TLS, CVE policy. *(WS binding: `wss://` only, `Origin` allow-list, validate every frame as untrusted input.)*
> - [`oauth.md`](guides://oauth.md) — token issuance/verification, scopes, refresh. *(WS binding: authenticate on the HTTP **upgrade**, never trust a query-string token without TLS.)*
> - [`error-handling.md`](guides://error-handling.md) — error strategy, timeouts, retries/backoff. *(WS binding: reconnect backoff + jitter, request timeouts, close-code semantics.)*
> - [`observability.md`](guides://observability.md) — metrics, tracing, SLOs. *(WS binding: connection gauge, message throughput, close-code histogram, RTT from ping/pong.)*
> - [`rest.md`](guides://rest.md) — request/response API style; consult to decide REST vs WS (see §1).

> 📎 **SEE ALSO:** [`redis.md`](guides://redis.md) · [`kafka.md`](guides://kafka.md) (durable fan-out) · [`graphql.md`](guides://graphql.md) (subscriptions) · [`grpc.md`](guides://grpc.md) (server streaming) · [`logging.md`](guides://logging.md) · [`zod.md`](guides://zod.md) (frame schemas)

---

## 1. Core Philosophies: REALTIME-FIRST

WebSocket-specific principles only. Auth, validation, error policy, and metrics come from §0.

- **R**econnect-resilient: the client owns recovery — backoff + jitter, resumable subscriptions, an outbound queue. Treat every disconnect as expected, not exceptional.
- **E**nvelope-everything: one typed message envelope multiplexes logical channels; never send naked strings.
- **A**uth-on-upgrade: identity is established during the HTTP handshake, before a frame flows (see `oauth.md`).
- **L**iveness-by-heartbeat: ping/pong (not TCP) is the source of truth for "is this peer alive".
- **T**hrottle & backpressure: bound per-connection send buffers and inbound rate; a slow consumer MUST NOT exhaust server memory.
- **I**dempotent & ordered: messages carry IDs/sequence so replays after reconnect are safe.
- **M**onitored: connection count, message rate, and close codes are first-class metrics (see `observability.md`).
- **E**dge-aware: prefer `wss://` end to end; tune proxy/idle timeouts to outlive the heartbeat interval.

**When NOT to use WebSocket:** if traffic is request/response or client-initiated polling, use REST (see [`rest.md`](guides://rest.md)). If it is server→client one-way streaming only, prefer SSE. Reach for WebSocket only when you need **low-latency bidirectional** frames over one long-lived connection.

**Verified Code**: Agent-generated WebSocket code MUST pass every gate in §2 before delivery.

---

## 2. Requirements (MANDATORY, auditable)

RFC-2119 keywords. IDs `WS-<TOPIC>-<NN>`. Each row has a binary gate; rows binding a shared rule cite its owner.

| ID | Requirement | Verify | Gate |
|----|-------------|--------|------|
| WS-SEC-01 | Production endpoints MUST be `wss://` (TLS); plaintext `ws://` only on loopback in tests (see `secure-coding.md`) | grep configs / scheme audit | no `ws://` to a public host |
| WS-SEC-02 | Server MUST validate the `Origin` header against an allow-list on upgrade | unit test: forged Origin → 403 | non-allowed Origin rejected |
| WS-AUTH-01 | Connections MUST be authenticated during the HTTP upgrade, before any app frame (see `oauth.md`) | unit test: no/invalid token → handshake rejected | 401/403 before `connection` |
| WS-AUTH-02 | Token expiry MUST close the socket (close `4001`); short-lived tokens MUST be refreshable without dropping (see `oauth.md`) | test: expire token → socket closes 4001 | closed, no zombie session |
| WS-VAL-01 | Every inbound frame MUST be schema-validated before handling (see `secure-coding.md`) | test: malformed frame → typed `error`, no throw | rejected, connection survives |
| WS-LIFE-01 | Each connection MUST run ping/pong heartbeat; unanswered → terminate | test: silent peer terminated within 2 intervals | dead peers reaped |
| WS-LIFE-02 | `close`/`error` handlers MUST release all per-connection resources (rooms, subs, timers) | test: close → maps/intervals empty | no leak |
| WS-LIFE-03 | Server MUST drain/close sockets gracefully on shutdown (close `1001`) | test: SIGTERM → clients get 1001 | clean drain |
| WS-RECON-01 | Client MUST reconnect with exponential backoff + jitter, capped (see `error-handling.md`) | test: server bounce → reconnect, no thundering herd | bounded reconnect |
| WS-BP-01 | Per-connection send buffer MUST be bounded; over-limit slow consumers MUST be shed (close `1013`/`1011`) | load test: slow reader does not grow server RSS unbounded | memory bounded |
| WS-RATE-01 | Inbound messages MUST be rate-limited per connection (see `secure-coding.md`) | test: flood → `RATE_LIMITED`, throttled | limit enforced |
| WS-SCALE-01 | Multi-node deployments MUST fan out via an external broker (pub/sub), not in-process lists | integration: msg from node A reaches node B client | cross-node delivery |
| WS-OBS-01 | Connection count, message rate, and close-code distribution MUST be exported (see `observability.md`) | scrape `/metrics` | gauges/counters present |
| WS-TST-01 | Lifecycle (open/message/close/error), reconnect, and routing MUST be tested (see `tdd.md`) | test suite | exit 0, 0 skips |

> **Forbidden**: trusting a client-supplied identity after upgrade without verifying a token; unbounded server-side send buffers; reconnect loops without backoff; broadcasting from in-process state across a horizontally-scaled fleet; sending secrets in the URL query string over `ws://`.

---

## 3. Connection Lifecycle (owned)

A WebSocket is an HTTP `Upgrade` (RFC 6455) that yields a long-lived full-duplex frame stream. The lifecycle is: **handshake → authenticated upgrade → open → frames (+ heartbeats) → close**. Auth policy lives in `oauth.md`; this section owns the *protocol mechanics*.

### A. Authenticated upgrade (the only safe place to authenticate)
Authenticate in the `upgrade`/handshake handler so unauthorized clients never get an open socket. Bind identity to the connection object and proceed only on success.

```javascript
// Node 'ws' — handle the raw upgrade, gate on auth + Origin
const wss = new WebSocketServer({ noServer: true });

server.on('upgrade', async (req, socket, head) => {
  if (!ORIGIN_ALLOWLIST.has(req.headers.origin)) {        // WS-SEC-02
    return abort(socket, 403, 'Forbidden Origin');
  }
  const user = await verifyToken(extractToken(req));       // oauth.md owns verifyToken
  if (!user) return abort(socket, 401, 'Unauthorized');    // WS-AUTH-01

  wss.handleUpgrade(req, socket, head, (ws) => {
    ws.user = user;                                         // identity bound to the socket
    wss.emit('connection', ws, req);
  });
});

function abort(socket, code, msg) {                         // reject before opening the socket
  socket.write(`HTTP/1.1 ${code} ${msg}\r\n\r\n`);
  socket.destroy();
}
```

**Token transport, ranked:** (1) `Sec-WebSocket-Protocol` subprotocol header or an `Authorization` header on the upgrade (browsers can't set arbitrary upgrade headers — use the subprotocol trick); (2) a short-lived, single-use ticket fetched over HTTPS then passed as a query param. A query-string token is only acceptable over `wss://` and SHOULD be short-lived, because URLs leak into logs/proxies. Verification, scopes, and refresh are owned by [`oauth.md`](guides://oauth.md).

### B. Close codes — use the standard semantics
| Code | Meaning | Reconnect? |
|---|---|---|
| 1000 | Normal closure | no (intentional) |
| 1001 | Going away (server shutdown — WS-LIFE-03) | yes, after delay |
| 1006 | Abnormal (no close frame — network drop) | yes, with backoff |
| 1008 | Policy violation | no (fix request) |
| 1011 | Server error | yes, with backoff |
| 1013 | Try again later (overload/backpressure shed) | yes, honor delay |
| 4000–4999 | Application-defined (e.g. `4001` = auth expired) | per app policy |

Clients decide reconnect behavior from the close code: never auto-reconnect on `1000`/`1008`; always back off on `1006`/`1011`/`1013`.

### C. Resource cleanup (WS-LIFE-02)
Every per-connection resource (room memberships, broker subscriptions, refresh timers, rate-limiter entries) MUST be torn down in the `close` *and* `error` paths — both fire for a dead connection, so make cleanup idempotent. Leaks here are the #1 cause of WebSocket memory growth.

---

## 4. Heartbeats & Liveness (owned)

TCP will not promptly tell you a peer vanished (half-open connections survive for minutes). Application-level ping/pong is the only reliable liveness signal.

```javascript
// Server: reap peers that miss a pong (WS-LIFE-01)
const HEARTBEAT_MS = 30_000;
const beat = setInterval(() => {
  for (const ws of wss.clients) {
    if (ws.isAlive === false) { ws.terminate(); continue; } // missed last pong → dead
    ws.isAlive = false;
    ws.ping();                                              // protocol-level PING frame
  }
}, HEARTBEAT_MS);
wss.on('connection', (ws) => { ws.isAlive = true; ws.on('pong', () => { ws.isAlive = true; }); });
wss.on('close', () => clearInterval(beat));
```

Rules:
- Use **protocol** ping/pong frames (opcodes 0x9/0xA), not an application `{"type":"ping"}` message, for liveness — the runtime answers pongs even while app code is busy. Reserve app-level ping only where the platform hides protocol frames (some browsers/load balancers).
- Heartbeat interval MUST be shorter than every idle timeout in the path (proxy `proxy_read_timeout`, LB idle timeout, cloud gateway). A 30 s ping behind a 60 s idle proxy is safe; behind a 25 s proxy it is not.
- Derive connection **RTT** from ping→pong and export it (see `observability.md`).

---

## 5. Reconnection & Message Delivery (owned)

The client is responsible for recovery. Retry *strategy* (backoff, jitter, caps) is owned by [`error-handling.md`](guides://error-handling.md); below is the WebSocket binding.

```javascript
// Backoff with full jitter, capped — avoids thundering herd on server bounce (WS-RECON-01)
function nextDelay(attempt, base = 1000, cap = 30_000) {
  const ceil = Math.min(cap, base * 2 ** attempt);
  return Math.random() * ceil;                  // full jitter
}
```

Delivery guarantees the protocol does **not** give you (you must build these):
- **At-least-once on reconnect:** buffer unacked outbound frames; replay on reopen. Each frame carries a unique `id`; consumers dedupe (idempotency) — naked WebSocket has no redelivery.
- **Resume subscriptions:** on reopen, re-send `subscribe`/`join` and (if supported) a `last-seen sequence` so the server replays the gap. Without a sequence cursor, a reconnect silently drops everything sent during the outage.
- **Outbound queue:** while `readyState !== OPEN`, enqueue; flush in order on `open`. Bound the queue and drop-oldest or fail-fast when full (this is client-side backpressure).

```javascript
send(frame) {
  if (this.ws?.readyState === WebSocket.OPEN) this.ws.send(JSON.stringify(frame));
  else if (this.queue.length < this.maxQueue) this.queue.push(frame); // bounded
  else this.onDrop(frame);                                            // shed, don't grow unbounded
}
```

---

## 6. Message Framing & Protocol (owned)

Multiplex logical streams over the single connection with a typed envelope. Validate every inbound frame as untrusted input (see [`secure-coding.md`](guides://secure-coding.md)); schema enforcement via Zod/Ajv/JSON-Schema (see [`zod.md`](guides://zod.md)).

```ts
// One envelope for every direction. type → routing; id → tracking; correlationId → req/resp; seq → ordering.
interface Envelope<T = unknown> {
  type: string;            // 'subscribe' | 'publish' | 'message' | 'error' | 'ack' | ...
  id: string;              // unique per frame (dedupe + ack target)
  payload: T;
  ts: string;              // ISO 8601
  correlationId?: string;  // ties a response to its request
  seq?: number;            // per-stream ordering / resume cursor
}
```

Conventions:
- **Request/response over a stream:** sender sets `id`; responder echoes it as `correlationId`. Track pending requests with a timeout (timeout policy → `error-handling.md`); reject on timeout, resolve on matching `correlationId`.
- **Acknowledgement:** for at-least-once, the receiver emits an `ack` referencing the frame `id`; the sender clears it from the replay buffer.
- **Text vs binary:** JSON text is fine for control and low-rate data. For high-throughput or large payloads use a binary codec (MessagePack/CBOR/Protobuf) over binary frames — measure first (see [`performance.md`](guides://performance.md)).
- **Versioning:** negotiate the protocol version via `Sec-WebSocket-Protocol` (subprotocol) at handshake; bump it under SemVer (see [`semver.md`](guides://semver.md)) and reject unknown versions on upgrade.

---

## 7. Backpressure (owned)

A WebSocket can deliver faster than a peer drains. Unbounded buffering is a memory-exhaustion DoS — this is the failure mode unique to long-lived streaming connections.

**Server → client (slow consumer):** watch `ws.bufferedAmount` (browser) / the socket write buffer (Node `ws` exposes `_socket.writableLength`; Go: a bounded send channel per client). Policy when the buffer exceeds a high-water mark (WS-BP-01): for **droppable** data (telemetry, presence) drop-oldest or coalesce; for **must-deliver** data, stop reading from the producer (pause the source) or shed the slow client with close `1013`. Never let one slow reader grow server RSS without bound.

```go
// Go (gorilla/websocket): one bounded send channel per client; full channel ⇒ shed
select {
case client.send <- frame:                 // ok
default:                                    // buffer full — slow consumer
    close(client.send); hub.unregister <- client   // shed (peer gets 1011/abnormal)
}
```

**Client → server (flood):** rate-limit inbound per connection (WS-RATE-01); on breach reply `{type:"error", payload:{code:"RATE_LIMITED", retryAfter}}` and optionally close `1008` on repeat abuse. The rate-limit *policy* is part of [`secure-coding.md`](guides://secure-coding.md); the per-socket enforcement point is owned here.

The reader loop must also **pause** when downstream is full — apply `ws.pause()`/stop the read pump rather than spawning unbounded goroutines/promises per frame.

---

## 8. Scaling: Pub/Sub Fan-Out (owned)

A single process holds connections in memory, so an in-process broadcast list only reaches clients on *that* node. Across a fleet, fan out through an external broker (WS-SCALE-01).

```
client ── LB (sticky) ──► node A ─┐
                                  ├─► broker (Redis pub/sub / Kafka topic) ─► every node
client ── LB (sticky) ──► node B ─┘            re-broadcasts to its LOCAL sockets
```

- **Pattern:** each node subscribes to broker channels for the rooms/users it hosts; on publish it (a) delivers to local sockets and (b) publishes to the broker, tagging its own `nodeId` so it ignores the echo. Other nodes re-broadcast to their local sockets.
- **Broker choice:** Redis pub/sub for fire-and-forget fan-out (see [`redis.md`](guides://redis.md)); Kafka/streams when you need durability, replay, or a resume cursor (see [`kafka.md`](guides://kafka.md)). Socket.IO ships a Redis adapter that implements this pattern.
- **Connection directory:** to target a user who may be on another node, keep a `userId → nodeId` map in the broker/store and route via the broker rather than broadcasting blindly.
- **Sticky sessions:** a connection lives on one node for its lifetime, so the LB MUST pin it (`ip_hash`, cookie, or consistent-hash) — but state that must survive a node loss MUST live in the broker/store, not in process. Externalized state (WS-LIFE-02) is what lets any node accept the reconnect.

```nginx
# nginx: WebSocket upgrade + sticky + idle timeout > heartbeat (see §4)
upstream ws_pool { ip_hash; server ws1:8080; server ws2:8080; }
location /ws {
  proxy_pass http://ws_pool;
  proxy_http_version 1.1;
  proxy_set_header Upgrade $http_upgrade;
  proxy_set_header Connection "upgrade";
  proxy_read_timeout 75s;          # MUST exceed the 30s heartbeat
}
```

---

## 9. Observability (binding)

Metrics/tracing policy is owned by [`observability.md`](guides://observability.md). Export at minimum (WS-OBS-01): active connections (gauge), connect/disconnect rate, messages in/out (counter, by type), frame bytes, close codes (histogram/counter — a spike in `1006`/`1011` is your alarm), heartbeat RTT, and send-buffer high-water occurrences. Propagate a trace/correlation context in the envelope so a frame can be linked to upstream HTTP/event spans. Structured connection logs (with `userId`, `nodeId`, close code) per [`logging.md`](guides://logging.md).

---

## 10. Testing (binding)

Test-first, Red-Green-Refactor, and regression-test-before-fix are owned by [`tdd.md`](guides://tdd.md). The WebSocket-specific surface to cover (WS-TST-01):

- **Lifecycle:** open with valid/invalid token (handshake accept/reject), message echo/route, clean close (`1000`), abnormal close (`1006`), and `error`. Always close clients and `wss`/server in teardown to avoid leaked handles.
- **Heartbeat:** a silent peer is terminated within ≤ 2 intervals; pong resets liveness.
- **Reconnect:** simulate a server bounce; assert backoff timing and that the outbound queue flushes in order on reopen.
- **Backpressure:** a deliberately slow reader does not grow server memory unbounded; over-limit producers are shed.
- **Routing:** broadcast, room-scoped, and direct-to-user delivery; and (multi-node) a message published on one node reaches a client on another (use two server instances + the broker).
- **Validation:** malformed/oversized/forbidden-Origin frames are rejected without crashing the connection.

```javascript
test('rejects connection with invalid token', (done) => {
  const c = new WebSocket(`ws://localhost:${port}/ws?token=bad`);
  c.on('error', () => done());          // handshake fails before open (WS-AUTH-01)
  c.on('open', () => done(new Error('should not open')));
});
```

---

## 11. Quick Reference

```javascript
// readyState
WebSocket.CONNECTING /*0*/  OPEN /*1*/  CLOSING /*2*/  CLOSED /*3*/

ws.ping();                 // protocol PING (liveness)
ws.send(JSON.stringify(envelope));
ws.close(1000, 'done');    // graceful, no reconnect
ws.terminate();            // force-close a dead peer (server)
ws.bufferedAmount;         // backpressure signal (browser)
// Origin allow-list + token verify happen on the UPGRADE, not after.
```

---

## 12. Deployment Checklist

Generated from §2 — one box per requirement ID.

- [ ] WS-SEC-01 — `wss://`/TLS in production; no public `ws://`
- [ ] WS-SEC-02 — `Origin` allow-list enforced on upgrade
- [ ] WS-AUTH-01 — authenticated on upgrade before any frame (see `oauth.md`)
- [ ] WS-AUTH-02 — token expiry closes socket (4001); refresh without drop
- [ ] WS-VAL-01 — every inbound frame schema-validated (see `secure-coding.md`)
- [ ] WS-LIFE-01 — ping/pong heartbeat reaps dead peers
- [ ] WS-LIFE-02 — close/error release all per-connection resources
- [ ] WS-LIFE-03 — graceful drain on shutdown (1001)
- [ ] WS-RECON-01 — client backoff + jitter, capped (see `error-handling.md`)
- [ ] WS-BP-01 — bounded send buffers; slow consumers shed
- [ ] WS-RATE-01 — inbound rate limiting per connection
- [ ] WS-SCALE-01 — external broker fan-out for multi-node
- [ ] WS-OBS-01 — connection/throughput/close-code metrics exported (see `observability.md`)
- [ ] WS-TST-01 — lifecycle, reconnect, routing, backpressure tested (see `tdd.md`)
- [ ] Agent ran every §2 verification and documented any fixes

---
**End of WebSocket Guidelines**
