# gRPC Development Guidelines
Mandatory standards for designing and operating gRPC/Protocol Buffers services: schema-first contracts, streaming, deadlines, status-code error model, backward-compatible proto evolution. Protocol Buffers 3, gRPC, Buf, protovalidate, grpcurl, grpc-gateway.

---
name: grpc
title: gRPC Development Guidelines
version: 2.0
last_reviewed: 2026-06-05
kind: cross-cutting
tools: [protobuf@3, buf, protovalidate, grpcurl, evans, grpc-gateway]
requires: []
recommends:
  - rest
  - secure-coding
  - error-handling
  - observability
  - microservices
provides:
  - protobuf-design
  - grpc-streaming
  - deadlines
  - proto-evolution
---

> 🧭 Authored per [`CONVENTIONS.md`](guides://CONVENTIONS.md): shared concerns are referenced, not restated. This guide owns the gRPC/protobuf contract surface — service & message design, streaming, deadlines/cancellation, interceptors, the status-code error model, and backward-compatible proto evolution. It does not restate the language guide that generates the stubs.

---

## 0. Prerequisites & References

gRPC is language-agnostic: the generated stubs are written in Go/Python/Java/etc., so the owning **language guide** (test runner, formatter, type checker, lockfile, CVE scan) governs the implementation. This guide assumes those rules and adds only the gRPC-specific contract layer.

> 📎 **RECOMMENDED — fetch when the task touches them:**
> - [`rest.md`](guides://rest.md) — choosing REST vs gRPC, and REST transcoding via grpc-gateway. *(gRPC binding: expose REST only at the edge; service-to-service stays gRPC.)*
> - [`secure-coding.md`](guides://secure-coding.md) — transport security, secrets, supply chain, CVEs. *(gRPC binding: TLS 1.3 / mTLS via transport credentials; scan generated-stub deps.)*
> - [`error-handling.md`](guides://error-handling.md) — error strategy & propagation. *(gRPC binding: map domain errors to `google.rpc.Status` codes + `error_details`.)*
> - [`observability.md`](guides://observability.md) — tracing, metrics, context propagation. *(gRPC binding: OpenTelemetry stats handler + W3C `traceparent` in metadata.)*
> - [`microservices.md`](guides://microservices.md) — service boundaries, retries, timeouts, idempotency.

> 📎 **SEE ALSO:** the implementation **language guide** (e.g. [`go.md`](guides://go.md), [`python.md`](guides://python.md), [`java.md`](guides://java.md)) for the build, test, lint and CVE gates that wrap the generated code · [`graphql.md`](guides://graphql.md) · [`openapi.md`](guides://openapi.md) · [`websocket.md`](guides://websocket.md) (alternative API styles) · [`logging.md`](guides://logging.md) · [`semver.md`](guides://semver.md) (versioning the proto package).

---

## 1. Core Philosophies: GRPC-FIRST

gRPC-specific principles only. Test-first, security, error strategy, tracing and the language toolchain come from §0.

- **G**enerated, never hand-written: stubs and serialization come from `.proto` via `buf generate`; no hand-rolled wire code.
- **R**eliable by deadline: every call carries a deadline and is cancellation-aware; deadlines propagate down the call chain.
- **P**roto is the contract: design the schema first; the schema is the API. Field numbers and the status-code model are part of the contract.
- **C**ompatible forever: proto evolution is wire-compatible by construction — add, never repurpose; reserve, never reuse (see §5).

**Verified Code**: Agent-generated protos and services MUST pass every gate in §2 before delivery.

---

## 2. Requirements (MANDATORY, auditable)

RFC-2119 keywords. IDs `GRPC-<TOPIC>-<NN>`. Each row has a binary gate; rows binding a shared rule cite its owner. `<lang>` denotes the language guide's commands.

| ID | Requirement | Verify | Gate |
|----|-------------|--------|------|
| GRPC-TST-01 | Service & streaming RPCs MUST be test-first via the language guide's runner (see language guide, `tdd.md`) | `<lang> test` | exit 0, 0 skips |
| GRPC-LINT-01 | Protos MUST pass the Buf linter (style, enum-zero, naming) | `buf lint` | exit 0 |
| GRPC-EVO-01 | No backward-incompatible proto change vs the released contract | `buf breaking --against '.git#branch=main'` | exit 0 |
| GRPC-FMT-01 | Protos MUST be formatted | `buf format -d --exit-code` | no diff |
| GRPC-GEN-01 | Generated code MUST be reproducible & in sync with protos | `buf generate && git diff --exit-code` | no diff |
| GRPC-ENUM-01 | Every enum MUST have a `_UNSPECIFIED = 0` zero value | `buf lint` (ENUM_ZERO_VALUE) | exit 0 |
| GRPC-VAL-01 | Externally-facing message fields MUST declare protovalidate constraints | review / grep `(buf.validate` | constraints present |
| GRPC-DDL-01 | Every server RPC MUST honor `ctx`/stream deadline & cancellation | review / test with expired ctx | returns `DeadlineExceeded`/`Cancelled` |
| GRPC-ERR-01 | Errors MUST be returned as gRPC status codes, never bare strings (see `error-handling.md`) | review / test asserts `status.Code` | typed codes only |
| GRPC-SEC-01 | Production servers MUST use TLS 1.3 (mTLS for service-to-service) (see `secure-coding.md`) | config review / `openssl s_client -alpn h2` | TLS enabled |
| GRPC-OBS-01 | Servers & clients MUST install tracing/metrics interceptors (see `observability.md`) | review / trace present in backend | spans emitted |
| GRPC-HLT-01 | Servers MUST register the standard `grpc.health.v1.Health` service & reflection | `grpcurl plaintext :PORT grpc.health.v1.Health/Check` | `SERVING` |
| GRPC-SEC-02 | 0 known CVEs in gRPC/protobuf runtime deps (see `secure-coding.md`) | `<lang>` CVE scanner | 0 high/critical |

> **Forbidden**: reusing or renumbering a released field tag; an enum without an `_UNSPECIFIED = 0`; returning raw error strings instead of `status` codes; a server RPC that ignores `ctx.Done()`; shipping a breaking proto change without a new package version; `grpc.WithInsecure`/plaintext in production.

---

## 3. Verification Protocol

Run, in order, before presenting code. Fix → re-run until every gate is green. Language-level gates (format/lint/type/test/CVE of the *generated* and *handler* code) come from the language guide.

```bash
buf format -d --exit-code                       # GRPC-FMT-01
buf lint                                         # GRPC-LINT-01, GRPC-ENUM-01
buf breaking --against '.git#branch=main'        # GRPC-EVO-01
buf generate && git diff --exit-code             # GRPC-GEN-01
<lang> test                                       # GRPC-TST-01, GRPC-DDL-01, GRPC-ERR-01
grpcurl -plaintext localhost:PORT grpc.health.v1.Health/Check   # GRPC-HLT-01
```

The *why* behind each shared gate (test-first, CVE policy, tracing) lives in its §0 owner.

---

## 4. Protobuf & Service Design (owned)

The contract is the schema. Design it deliberately; everything downstream is generated.

### A. File & package layout

```protobuf
// proto/myapp/user/v1/user.proto
syntax = "proto3";
package myapp.user.v1;                 // <org>.<domain>.<version>; version in the package

option go_package = "github.com/myorg/myapp/gen/myapp/user/v1;userv1";

import "google/protobuf/timestamp.proto";
import "buf/validate/validate.proto";  // protovalidate (PGV is deprecated)

// User is a registered account.
message User {
  string id = 1;
  string email = 2 [(buf.validate.field).string.email = true];
  string display_name = 3 [(buf.validate.field).string = {min_len: 1, max_len: 100}];
  UserRole role = 4;
  google.protobuf.Timestamp created_at = 5;
}

enum UserRole {
  USER_ROLE_UNSPECIFIED = 0;   // GRPC-ENUM-01: zero value is always _UNSPECIFIED
  USER_ROLE_USER = 1;
  USER_ROLE_ADMIN = 2;
}
```

Rules that are part of the contract:
- One major API version per package path (`.../v1`, `.../v2`); never mix breaking versions in one package.
- Enum values are prefixed with the enum name and start at `_UNSPECIFIED = 0` (the wire default must be "unset", not a real value).
- Field/message/RPC naming is `snake_case`/`PascalCase` per the Buf style guide — `buf lint` enforces it; do not hand-police it.

### B. Service & method shape

```protobuf
service UserService {
  rpc GetUser(GetUserRequest) returns (GetUserResponse);
  rpc ListUsers(ListUsersRequest) returns (ListUsersResponse);   // paginated
  rpc CreateUser(CreateUserRequest) returns (CreateUserResponse);
  rpc WatchUsers(WatchUsersRequest) returns (stream UserEvent);  // server stream
}
```

- **Wrap every request and response in its own message** (`GetUserResponse`, not bare `User`) — you can add fields later without a breaking change. This is the single most important evolution rule.
- Use AIP-style list pagination: `page_size` + `page_token` in the request, `next_page_token` in the response.
- Mutations that may be retried carry an idempotency/request key; the server dedupes on it (see `microservices.md` for the retry/idempotency contract).
- Partial updates use `google.protobuf.FieldMask update_mask`; never overload "empty field means no change".

### C. Field validation — protovalidate binding

Security/input-validation *policy* is owned by [`secure-coding.md`](guides://secure-coding.md). The gRPC binding is **protovalidate** (CEL-based; the modern successor to protoc-gen-validate), declared in the schema and enforced by a server interceptor so handlers never see malformed input:

```protobuf
message CreateOrderRequest {
  string customer_id = 1 [(buf.validate.field).string.uuid = true];
  repeated OrderItem items = 2 [(buf.validate.field).repeated = {min_items: 1, max_items: 100}];
  string currency = 3 [(buf.validate.field).string = {in: ["USD", "EUR", "GBP"]}];
  int64 amount_cents = 4 [(buf.validate.field).int64 = {gte: 1}];
}
```

---

## 5. Backward-Compatible Proto Evolution (owned)

The proto file is a permanent contract; old binaries must keep working against new ones. `buf breaking` (GRPC-EVO-01) gates this, but the agent MUST design for it:

**Safe (non-breaking):** add a new field with a fresh tag; add a new RPC; add a new enum value; add a new message. Old readers ignore unknown fields and treat unknown enum values as the zero value.

**Breaking — never do these on a released package:**
- Change or reuse a field number → corrupts the wire for old clients.
- Change a field's type, or its `repeated`/scalar/`oneof` membership.
- Rename a field in proto3 JSON, or rename an enum value's number mapping.
- Delete a field/enum value without `reserved` → the tag can later be reused by accident.

**When you remove a field or value, reserve it forever:**
```protobuf
message User {
  reserved 4, 7 to 9;                 // retired field numbers — never reused
  reserved "legacy_role", "phone";    // retired field names
}
```

**When a change is genuinely breaking**, do not mutate `v1` — publish `myapp.user.v2`, run both in parallel, and migrate clients. Version the proto *package*, not just a URL. Versioning policy for the package itself is owned by [`semver.md`](guides://semver.md).

---

## 6. Deadlines, Cancellation & Resource Discipline (owned)

This is gRPC's signature reliability feature and is mandatory (GRPC-DDL-01).

- **Clients set an absolute deadline on every call** (`context.WithTimeout` / `deadline=` in Python). A call without a deadline can hang forever and leak a server goroutine.
- **Servers honor the context.** Long operations and streams `select` on `ctx.Done()` and abort with `Canceled`/`DeadlineExceeded`; pass `ctx` into every downstream call so the deadline propagates and budget shrinks down the chain.
- **Deadline propagation** prevents cascading timeouts: if the caller has 200 ms left, the downstream call inherits ~200 ms, not its own fresh 5 s.

```go
ctx, cancel := context.WithTimeout(ctx, 200*time.Millisecond)
defer cancel()                       // always cancel to release resources
resp, err := client.GetUser(ctx, req)
```

```go
// Streaming server: stop work the instant the client goes away.
for {
    select {
    case <-stream.Context().Done():
        return stream.Context().Err()      // Canceled / DeadlineExceeded
    case ev := <-events:
        if err := stream.Send(ev); err != nil {
            return err                       // client gone / transport error
        }
    }
}
```

Retry/backoff policy belongs with [`microservices.md`](guides://microservices.md); only retry idempotent calls on retryable codes (`Unavailable`, `ResourceExhausted`, `Aborted`) — never on `Internal`/`InvalidArgument`. Prefer gRPC's built-in service-config retry policy over hand-rolled client loops.

---

## 7. Streaming Patterns (owned)

Four call types: unary, server-stream, client-stream, bidi. Choose streaming for large payloads, live feeds, and long-running work; use unary otherwise. Logic shown in Go; the *shape* is identical across languages — the language guide owns its idioms.

**Server streaming** — server emits N messages for one request (live feeds, large result sets):
```go
func (s *Svc) WatchUsers(req *pb.WatchUsersRequest, stream pb.UserService_WatchUsersServer) error {
    for ev := range s.subscribe(stream.Context(), req.GetFilter()) {
        if err := stream.Send(ev); err != nil { return err }   // honors ctx via subscribe
    }
    return nil
}
```

**Client streaming** — client uploads N messages, server replies once (chunked uploads, batch ingest). Frame with a `oneof { Metadata; bytes chunk }` so the first message carries metadata:
```go
func (s *Svc) Upload(stream pb.UploadService_UploadServer) error {
    for {
        req, err := stream.Recv()
        if errors.Is(err, io.EOF) { break }     // client finished
        if err != nil { return err }
        // accumulate req.GetChunk()
    }
    return stream.SendAndClose(&pb.UploadResponse{/* ... */})
}
```

**Bidirectional streaming** — independent read/write loops (chat, interactive sessions). Run the receive loop and send loop concurrently and key both off `stream.Context()`; on the server, returning from the handler closes the stream.

**Stream footguns:**
- Unbounded streams without a deadline or `ctx` check → leaked goroutines/connections.
- Per-message error: `Send`/`Recv` returning non-EOF error means the stream is dead — return, don't loop.
- Backpressure: a slow consumer blocks `Send`; bound your buffers and let the deadline cut it off.
- Flow control & message size: cap with `MaxRecvMsgSize`/`MaxSendMsgSize`; very large single messages should be a client stream of chunks instead.

---

## 8. Error Model — gRPC status codes (owned binding)

Error *strategy* (when to fail, wrap, retry) is owned by [`error-handling.md`](guides://error-handling.md). gRPC's binding is the **status code + `google.rpc.Status` details** model (GRPC-ERR-01): return a typed `status`, never a bare string or a `200`-with-error-body.

Canonical domain → code mapping:

| Code | When |
|------|------|
| `INVALID_ARGUMENT` (3) | client sent malformed/invalid data (validation failure) |
| `FAILED_PRECONDITION` (9) | system state forbids the op (e.g. non-empty dir) |
| `OUT_OF_RANGE` (11) | value past valid range (distinct from INVALID_ARGUMENT, retryable after state change) |
| `NOT_FOUND` (5) / `ALREADY_EXISTS` (6) | resource missing / duplicate |
| `UNAUTHENTICATED` (16) / `PERMISSION_DENIED` (7) | no/invalid creds vs valid creds without rights |
| `RESOURCE_EXHAUSTED` (8) | quota/rate limit (retryable with backoff) |
| `DEADLINE_EXCEEDED` (4) / `CANCELED` (1) | deadline hit / caller canceled |
| `UNAVAILABLE` (14) | transient; safe to retry idempotent calls |
| `ABORTED` (10) | concurrency conflict (retry the transaction) |
| `INTERNAL` (13) / `UNKNOWN` (2) | server bug; never leak internals to the client message |

Attach machine-readable details with `error_details` (`BadRequest` field violations, `RetryInfo`, `QuotaFailure`, `ErrorInfo`) rather than stuffing context into the free-text message:

```go
st := status.New(codes.InvalidArgument, "invalid request")
st, _ = st.WithDetails(&errdetails.BadRequest{FieldViolations: violations})
return nil, st.Err()
```

Clients branch on `status.Code(err)` and read typed details — never parse the message string.

---

## 9. Interceptors / Middleware (owned)

Cross-cutting concerns are interceptors (unary + stream), not handler code. Standard server chain order: **recovery → tracing/metrics → logging → auth → validation → rate-limit → handler**.

- **Recovery** turns a panic into `Internal` so one bad RPC can't crash the server.
- **Tracing/metrics**: install the **OpenTelemetry gRPC stats handler** (`otelgrpc`) rather than hand-writing trace plumbing; it extracts/injects W3C `traceparent` from/to gRPC metadata automatically. Tracing/metrics policy and IDs are owned by [`observability.md`](guides://observability.md) — bind, don't restate.
- **Auth** reads credentials from `metadata` (e.g. `authorization`) and rejects with `Unauthenticated`/`PermissionDenied`. Auth *policy* (token formats, OAuth) lives in [`secure-coding.md`](guides://secure-coding.md)/`oauth.md`.
- **Validation** runs protovalidate so handlers receive only valid messages.

```go
srv := grpc.NewServer(
    grpc.StatsHandler(otelgrpc.NewServerHandler()),    // GRPC-OBS-01 (see observability.md)
    grpc.ChainUnaryInterceptor(recovery, logging, auth, validate),
    grpc.Creds(credentials.NewTLS(tlsConfig)),          // GRPC-SEC-01 (see secure-coding.md)
)
```

Clients compose interceptors the same way (deadline-injection, retry via service config, trace propagation).

---

## 10. Transport, Health & Tooling (owned binding)

- **Transport security (GRPC-SEC-01):** production uses TLS 1.3; service-to-service uses **mTLS** via transport credentials. Plaintext/`WithInsecure` is dev-only. Certificate issuance, rotation and secret storage are owned by [`secure-coding.md`](guides://secure-coding.md) (cert-manager / Vault PKI). Verify with `openssl s_client -connect host:443 -alpn h2`.
- **Health & reflection (GRPC-HLT-01):** register `grpc.health.v1.Health` and tie its status to dependency probes; enable server reflection so `grpcurl`/Evans can introspect. Probe from k8s with `grpc_health_probe`.
- **REST edge:** if HTTP/JSON clients exist, transcode with **grpc-gateway** from the same protos and keep service-to-service on gRPC. The REST-vs-gRPC decision is owned by [`rest.md`](guides://rest.md).
- **Toolchain:** `buf` is the standard — `buf lint`, `buf format`, `buf breaking`, `buf generate` (config in `buf.yaml`/`buf.gen.yaml`); use a BSR module or vendored deps instead of copying `google/` protos. Explore running services with `grpcurl`/`evans`.

```bash
buf generate                     # GRPC-GEN-01: regenerate stubs from protos
grpcurl -plaintext localhost:50051 list                       # introspect via reflection
grpcurl -plaintext -d '{"id":"u1"}' localhost:50051 myapp.user.v1.UserService/GetUser
```

---

## 11. Quick Reference

```protobuf
// Scalars: string bytes bool int32 int64 uint32 uint64 sint* fixed* float double
repeated Type field = N;          // list
map<string, Type> field = N;      // map (keys: scalar, not float/bytes)
oneof choice { TypeA a = 1; TypeB b = 2; }
reserved 4, 7 to 9; reserved "old_name";        // retire tags/names — never reuse
import "google/protobuf/{timestamp,duration,empty,field_mask,wrappers}.proto";
```

```bash
buf lint                                  # GRPC-LINT-01 / ENUM-01
buf format -d --exit-code                 # GRPC-FMT-01
buf breaking --against '.git#branch=main' # GRPC-EVO-01
buf generate && git diff --exit-code      # GRPC-GEN-01
grpcurl -plaintext :PORT grpc.health.v1.Health/Check   # GRPC-HLT-01
```

---

## 12. Deployment Checklist

Generated from §2 — one box per requirement ID. No new requirements here.

- [ ] GRPC-LINT-01 — `buf lint` clean
- [ ] GRPC-ENUM-01 — every enum has `_UNSPECIFIED = 0`
- [ ] GRPC-FMT-01 — `buf format` clean
- [ ] GRPC-EVO-01 — `buf breaking` clean (no incompatible change vs released contract)
- [ ] GRPC-GEN-01 — generated stubs regenerated & committed (no diff)
- [ ] GRPC-VAL-01 — external message fields carry protovalidate constraints
- [ ] GRPC-DDL-01 — RPCs honor deadlines & cancellation
- [ ] GRPC-ERR-01 — errors returned as gRPC status codes + details
- [ ] GRPC-SEC-01 — TLS 1.3 / mTLS enabled in production
- [ ] GRPC-SEC-02 — 0 high/critical CVEs in gRPC/protobuf deps
- [ ] GRPC-OBS-01 — tracing/metrics interceptors installed
- [ ] GRPC-HLT-01 — health service + reflection registered, returns SERVING
- [ ] GRPC-TST-01 — service & streaming RPCs test-first, all green
- [ ] Agent ran every §3 command and documented any fixes

---
**End of gRPC Guidelines**
