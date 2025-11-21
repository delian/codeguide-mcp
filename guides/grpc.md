# gRPC Development Guidelines

This document provides mandatory standards for designing and implementing gRPC services.

---

**Agent Profile**: The gRPC Expert
**Role**: Senior API Engineer & Distributed Systems Architect
**Objective**: Generate efficient, type-safe, and scalable gRPC services following best practices.
**Tools**: Protocol Buffers 3, gRPC, grpcurl, Evans, Buf, grpc-gateway.

---

## 1. Core Philosophies: GRPC-FIRST

- **G**enerated: Type-safe code generation from proto files
- **R**eliable: Built-in retries, deadlines, and cancellation
- **P**erformant: Binary serialization with HTTP/2
- **C**ontracted: Schema-first API design

---

## 2. Protocol Buffer Design (MANDATORY)

### A. File Organization

```protobuf
// api/v1/user.proto
syntax = "proto3";

package myapp.api.v1;

option go_package = "github.com/myorg/myapp/gen/go/api/v1;apiv1";
option java_package = "com.myorg.myapp.api.v1";
option java_multiple_files = true;

import "google/protobuf/timestamp.proto";
import "google/protobuf/empty.proto";
import "google/protobuf/field_mask.proto";
import "google/api/annotations.proto";
import "validate/validate.proto";

// User represents a registered user in the system.
message User {
  // Unique identifier for the user.
  string id = 1;

  // User's email address.
  string email = 2 [(validate.rules).string.email = true];

  // User's display name.
  string display_name = 3 [(validate.rules).string = {
    min_len: 1,
    max_len: 100
  }];

  // User's role in the system.
  UserRole role = 4;

  // When the user was created.
  google.protobuf.Timestamp created_at = 5;

  // When the user was last updated.
  google.protobuf.Timestamp updated_at = 6;
}

enum UserRole {
  USER_ROLE_UNSPECIFIED = 0;
  USER_ROLE_USER = 1;
  USER_ROLE_ADMIN = 2;
  USER_ROLE_MODERATOR = 3;
}
```

### B. Service Definition

```protobuf
// api/v1/user_service.proto
syntax = "proto3";

package myapp.api.v1;

import "api/v1/user.proto";
import "google/protobuf/empty.proto";
import "google/protobuf/field_mask.proto";

// UserService manages user accounts.
service UserService {
  // GetUser retrieves a single user by ID.
  rpc GetUser(GetUserRequest) returns (User) {
    option (google.api.http) = {
      get: "/v1/users/{id}"
    };
  }

  // ListUsers retrieves a paginated list of users.
  rpc ListUsers(ListUsersRequest) returns (ListUsersResponse) {
    option (google.api.http) = {
      get: "/v1/users"
    };
  }

  // CreateUser creates a new user.
  rpc CreateUser(CreateUserRequest) returns (User) {
    option (google.api.http) = {
      post: "/v1/users"
      body: "user"
    };
  }

  // UpdateUser updates an existing user.
  rpc UpdateUser(UpdateUserRequest) returns (User) {
    option (google.api.http) = {
      patch: "/v1/users/{user.id}"
      body: "user"
    };
  }

  // DeleteUser removes a user.
  rpc DeleteUser(DeleteUserRequest) returns (google.protobuf.Empty) {
    option (google.api.http) = {
      delete: "/v1/users/{id}"
    };
  }
}
```

### C. Request/Response Messages

```protobuf
// Standard request patterns
message GetUserRequest {
  string id = 1 [(validate.rules).string.uuid = true];
}

message ListUsersRequest {
  // Maximum number of users to return.
  int32 page_size = 1 [(validate.rules).int32 = {gte: 1, lte: 100}];

  // Token for pagination.
  string page_token = 2;

  // Filter expression (e.g., "role=ADMIN").
  string filter = 3;

  // Order by expression (e.g., "created_at desc").
  string order_by = 4;
}

message ListUsersResponse {
  repeated User users = 1;
  string next_page_token = 2;
  int32 total_count = 3;
}

message CreateUserRequest {
  User user = 1 [(validate.rules).message.required = true];

  // Idempotency key to prevent duplicate creation.
  string request_id = 2;
}

message UpdateUserRequest {
  User user = 1 [(validate.rules).message.required = true];

  // Fields to update. If empty, all fields are updated.
  google.protobuf.FieldMask update_mask = 2;
}

message DeleteUserRequest {
  string id = 1 [(validate.rules).string.uuid = true];
}
```

---

## 3. Error Handling (MANDATORY)

### Protocol-Specific Design Note

**Why gRPC error format differs from REST/GraphQL:**

| Aspect | gRPC | REST | GraphQL |
|--------|------|------|---------|
| **Error format** | Status codes with `errdetails` | JSON with `error`, `message` | `errors[]` with `extensions` |
| **Pagination** | Request message (`page_size`, `page_token`) | URL params | Relay connections |
| **Naming** | snake_case (proto fields) | snake_case | camelCase |
| **Rate limiting** | Interceptor-based + `ResourceExhausted` | HTTP headers | Query complexity |

These differences are **intentional and appropriate** for gRPC:
- gRPC uses Protocol Buffers' built-in status code system
- Binary serialization makes proto-native error details efficient
- Interceptors provide cross-cutting concerns like rate limiting
- snake_case follows Protocol Buffers style guide

**Cross-API services** should use grpc-gateway for REST transcoding.

---

### A. Status Codes

```go
// Use appropriate gRPC status codes
import (
    "google.golang.org/grpc/codes"
    "google.golang.org/grpc/status"
)

func (s *UserService) GetUser(ctx context.Context, req *pb.GetUserRequest) (*pb.User, error) {
    user, err := s.repo.FindByID(ctx, req.Id)
    if err != nil {
        if errors.Is(err, ErrNotFound) {
            return nil, status.Errorf(codes.NotFound, "user %s not found", req.Id)
        }
        return nil, status.Errorf(codes.Internal, "failed to fetch user: %v", err)
    }
    return user, nil
}

// Status code mapping:
// - NotFound (5): Resource doesn't exist
// - InvalidArgument (3): Client sent invalid data
// - FailedPrecondition (9): System not in required state
// - Unauthenticated (16): Missing or invalid credentials
// - PermissionDenied (7): Valid credentials but no permission
// - AlreadyExists (6): Resource already exists
// - ResourceExhausted (8): Rate limit or quota exceeded
// - Internal (13): Server error
// - Unavailable (14): Service temporarily unavailable
// - DeadlineExceeded (4): Operation timed out
```

### B. Error Details

```protobuf
// Define rich error details
import "google/rpc/error_details.proto";

// In server code
func (s *UserService) CreateUser(ctx context.Context, req *pb.CreateUserRequest) (*pb.User, error) {
    if err := s.validator.Validate(req); err != nil {
        st := status.New(codes.InvalidArgument, "invalid request")

        // Add field violations
        violations := &errdetails.BadRequest{}
        for _, e := range err.Errors {
            violations.FieldViolations = append(violations.FieldViolations,
                &errdetails.BadRequest_FieldViolation{
                    Field:       e.Field,
                    Description: e.Message,
                })
        }

        st, _ = st.WithDetails(violations)
        return nil, st.Err()
    }

    // ... create user
}
```

```go
// Client-side error handling
user, err := client.GetUser(ctx, &pb.GetUserRequest{Id: id})
if err != nil {
    st, ok := status.FromError(err)
    if !ok {
        return fmt.Errorf("unknown error: %v", err)
    }

    switch st.Code() {
    case codes.NotFound:
        return ErrUserNotFound
    case codes.InvalidArgument:
        // Extract details
        for _, detail := range st.Details() {
            if br, ok := detail.(*errdetails.BadRequest); ok {
                for _, violation := range br.FieldViolations {
                    log.Printf("Field %s: %s", violation.Field, violation.Description)
                }
            }
        }
        return ErrInvalidRequest
    case codes.Unavailable:
        // Retry logic
        return retry(func() error { return s.GetUser(ctx, id) })
    default:
        return fmt.Errorf("rpc error: %s", st.Message())
    }
}
```

---

## 4. Streaming Patterns (MANDATORY)

### A. Server Streaming

```protobuf
service OrderService {
  // StreamOrders returns a stream of orders for a user.
  rpc StreamOrders(StreamOrdersRequest) returns (stream Order);
}

message StreamOrdersRequest {
  string user_id = 1;
  bool include_history = 2;
}
```

```go
// Server implementation
func (s *OrderService) StreamOrders(req *pb.StreamOrdersRequest, stream pb.OrderService_StreamOrdersServer) error {
    ctx := stream.Context()

    // Stream existing orders
    orders, err := s.repo.GetUserOrders(ctx, req.UserId)
    if err != nil {
        return status.Errorf(codes.Internal, "failed to fetch orders: %v", err)
    }

    for _, order := range orders {
        if err := stream.Send(order); err != nil {
            return err
        }
    }

    // Subscribe to new orders
    sub := s.pubsub.Subscribe(ctx, "orders:"+req.UserId)
    defer sub.Close()

    for {
        select {
        case <-ctx.Done():
            return ctx.Err()
        case order := <-sub.Channel():
            if err := stream.Send(order); err != nil {
                return err
            }
        }
    }
}
```

### B. Client Streaming

```protobuf
service UploadService {
  // UploadFile receives chunks of a file and returns metadata.
  rpc UploadFile(stream UploadFileRequest) returns (UploadFileResponse);
}

message UploadFileRequest {
  oneof data {
    FileMetadata metadata = 1;
    bytes chunk = 2;
  }
}

message FileMetadata {
  string filename = 1;
  string content_type = 2;
}

message UploadFileResponse {
  string file_id = 1;
  int64 size = 2;
  string checksum = 3;
}
```

```go
// Server implementation
func (s *UploadService) UploadFile(stream pb.UploadService_UploadFileServer) error {
    var metadata *pb.FileMetadata
    var buffer bytes.Buffer

    for {
        req, err := stream.Recv()
        if err == io.EOF {
            break
        }
        if err != nil {
            return err
        }

        switch data := req.Data.(type) {
        case *pb.UploadFileRequest_Metadata:
            metadata = data.Metadata
        case *pb.UploadFileRequest_Chunk:
            buffer.Write(data.Chunk)
        }
    }

    fileID, err := s.storage.Save(metadata.Filename, buffer.Bytes())
    if err != nil {
        return status.Errorf(codes.Internal, "failed to save file: %v", err)
    }

    return stream.SendAndClose(&pb.UploadFileResponse{
        FileId:   fileID,
        Size:     int64(buffer.Len()),
        Checksum: calculateChecksum(buffer.Bytes()),
    })
}
```

### C. Bidirectional Streaming

```protobuf
service ChatService {
  // Chat enables real-time bidirectional messaging.
  rpc Chat(stream ChatMessage) returns (stream ChatMessage);
}

message ChatMessage {
  string room_id = 1;
  string user_id = 2;
  string content = 3;
  google.protobuf.Timestamp timestamp = 4;
}
```

```go
func (s *ChatService) Chat(stream pb.ChatService_ChatServer) error {
    ctx := stream.Context()

    // First message should contain room info
    firstMsg, err := stream.Recv()
    if err != nil {
        return err
    }

    roomID := firstMsg.RoomId
    userID := firstMsg.UserId

    // Join room
    room := s.rooms.Join(roomID, userID)
    defer room.Leave(userID)

    // Handle incoming messages
    go func() {
        for {
            msg, err := stream.Recv()
            if err != nil {
                return
            }
            room.Broadcast(msg)
        }
    }()

    // Send messages to client
    for {
        select {
        case <-ctx.Done():
            return ctx.Err()
        case msg := <-room.Messages(userID):
            if err := stream.Send(msg); err != nil {
                return err
            }
        }
    }
}
```

---

## 5. Interceptors (MANDATORY)

### A. Server Interceptors

```go
import (
    "google.golang.org/grpc"
    "google.golang.org/grpc/metadata"
)

// Logging interceptor
func LoggingUnaryInterceptor(
    ctx context.Context,
    req interface{},
    info *grpc.UnaryServerInfo,
    handler grpc.UnaryHandler,
) (interface{}, error) {
    start := time.Now()

    resp, err := handler(ctx, req)

    duration := time.Since(start)
    code := status.Code(err)

    log.Printf("method=%s duration=%s code=%s",
        info.FullMethod, duration, code)

    return resp, err
}

// Authentication interceptor
func AuthUnaryInterceptor(
    ctx context.Context,
    req interface{},
    info *grpc.UnaryServerInfo,
    handler grpc.UnaryHandler,
) (interface{}, error) {
    // Skip auth for certain methods
    if info.FullMethod == "/myapp.api.v1.HealthService/Check" {
        return handler(ctx, req)
    }

    md, ok := metadata.FromIncomingContext(ctx)
    if !ok {
        return nil, status.Error(codes.Unauthenticated, "missing metadata")
    }

    tokens := md.Get("authorization")
    if len(tokens) == 0 {
        return nil, status.Error(codes.Unauthenticated, "missing token")
    }

    userID, err := validateToken(tokens[0])
    if err != nil {
        return nil, status.Error(codes.Unauthenticated, "invalid token")
    }

    // Add user to context
    ctx = context.WithValue(ctx, userIDKey, userID)
    return handler(ctx, req)
}

// Recovery interceptor
func RecoveryUnaryInterceptor(
    ctx context.Context,
    req interface{},
    info *grpc.UnaryServerInfo,
    handler grpc.UnaryHandler,
) (resp interface{}, err error) {
    defer func() {
        if r := recover(); r != nil {
            log.Printf("panic recovered: %v\n%s", r, debug.Stack())
            err = status.Errorf(codes.Internal, "internal error")
        }
    }()
    return handler(ctx, req)
}

// Apply interceptors
server := grpc.NewServer(
    grpc.ChainUnaryInterceptor(
        RecoveryUnaryInterceptor,
        LoggingUnaryInterceptor,
        AuthUnaryInterceptor,
    ),
)
```

### B. Client Interceptors

```go
// Retry interceptor
func RetryUnaryInterceptor(
    ctx context.Context,
    method string,
    req, reply interface{},
    cc *grpc.ClientConn,
    invoker grpc.UnaryInvoker,
    opts ...grpc.CallOption,
) error {
    maxRetries := 3
    backoff := 100 * time.Millisecond

    var lastErr error
    for i := 0; i < maxRetries; i++ {
        err := invoker(ctx, method, req, reply, cc, opts...)
        if err == nil {
            return nil
        }

        st, ok := status.FromError(err)
        if !ok || !isRetryable(st.Code()) {
            return err
        }

        lastErr = err
        time.Sleep(backoff * time.Duration(1<<i))
    }
    return lastErr
}

func isRetryable(code codes.Code) bool {
    switch code {
    case codes.Unavailable, codes.ResourceExhausted, codes.Aborted:
        return true
    }
    return false
}

// Timeout interceptor
func TimeoutUnaryInterceptor(timeout time.Duration) grpc.UnaryClientInterceptor {
    return func(
        ctx context.Context,
        method string,
        req, reply interface{},
        cc *grpc.ClientConn,
        invoker grpc.UnaryInvoker,
        opts ...grpc.CallOption,
    ) error {
        ctx, cancel := context.WithTimeout(ctx, timeout)
        defer cancel()
        return invoker(ctx, method, req, reply, cc, opts...)
    }
}
```

---

## 6. Distributed Tracing (MANDATORY)

**CRITICAL: gRPC services MUST propagate trace IDs via metadata for observability.**

### A. Trace ID Propagation

```go
import (
    "google.golang.org/grpc/metadata"
)

// Trace ID metadata keys (support multiple formats)
const (
    TraceIDKey     = "x-trace-id"
    TraceParentKey = "traceparent"  // W3C Trace Context
)

// Extract trace ID from incoming metadata
func extractTraceID(ctx context.Context) string {
    md, ok := metadata.FromIncomingContext(ctx)
    if !ok {
        return generateTraceID()
    }

    // Try x-trace-id first
    if ids := md.Get(TraceIDKey); len(ids) > 0 {
        return ids[0]
    }

    // Try W3C traceparent
    if parents := md.Get(TraceParentKey); len(parents) > 0 {
        // Extract trace ID from traceparent: 00-{trace-id}-{span-id}-{flags}
        parts := strings.Split(parents[0], "-")
        if len(parts) >= 2 {
            return parts[1]
        }
    }

    return generateTraceID()
}

// Tracing interceptor
func TracingUnaryInterceptor(
    ctx context.Context,
    req interface{},
    info *grpc.UnaryServerInfo,
    handler grpc.UnaryHandler,
) (interface{}, error) {
    traceID := extractTraceID(ctx)

    // Add trace ID to context for logging
    ctx = context.WithValue(ctx, traceIDKey, traceID)

    // Add trace ID to outgoing metadata for downstream calls
    ctx = metadata.AppendToOutgoingContext(ctx, TraceIDKey, traceID)

    // Log with trace ID
    log.Printf("method=%s trace_id=%s", info.FullMethod, traceID)

    return handler(ctx, req)
}
```

### B. Client-Side Trace Propagation

```go
// Client interceptor to propagate trace ID
func TracingClientInterceptor(
    ctx context.Context,
    method string,
    req, reply interface{},
    cc *grpc.ClientConn,
    invoker grpc.UnaryInvoker,
    opts ...grpc.CallOption,
) error {
    // Get trace ID from context
    traceID, _ := ctx.Value(traceIDKey).(string)
    if traceID == "" {
        traceID = generateTraceID()
    }

    // Add to outgoing metadata
    ctx = metadata.AppendToOutgoingContext(ctx, TraceIDKey, traceID)

    return invoker(ctx, method, req, reply, cc, opts...)
}
```

**Cross-reference:** See logging.md Section 5 for trace ID implementation patterns.

---

## 7. Health Checks (MANDATORY)

```protobuf
// Use the standard health check proto
syntax = "proto3";

package grpc.health.v1;

service Health {
  rpc Check(HealthCheckRequest) returns (HealthCheckResponse);
  rpc Watch(HealthCheckRequest) returns (stream HealthCheckResponse);
}

message HealthCheckRequest {
  string service = 1;
}

message HealthCheckResponse {
  enum ServingStatus {
    UNKNOWN = 0;
    SERVING = 1;
    NOT_SERVING = 2;
    SERVICE_UNKNOWN = 3;
  }
  ServingStatus status = 1;
}
```

```go
import "google.golang.org/grpc/health"
import healthpb "google.golang.org/grpc/health/grpc_health_v1"

// Register health service
healthServer := health.NewServer()
healthpb.RegisterHealthServer(server, healthServer)

// Update service health
healthServer.SetServingStatus("myapp.api.v1.UserService", healthpb.HealthCheckResponse_SERVING)

// Check dependencies and update status
go func() {
    for {
        if s.db.Ping() == nil && s.cache.Ping() == nil {
            healthServer.SetServingStatus("", healthpb.HealthCheckResponse_SERVING)
        } else {
            healthServer.SetServingStatus("", healthpb.HealthCheckResponse_NOT_SERVING)
        }
        time.Sleep(10 * time.Second)
    }
}()
```

---

## 7. Testing (MANDATORY)

### A. Unit Tests

```go
func TestUserService_GetUser(t *testing.T) {
    // Setup
    mockRepo := &MockUserRepository{
        users: map[string]*pb.User{
            "user-1": {Id: "user-1", Email: "test@example.com"},
        },
    }
    service := NewUserService(mockRepo)

    tests := []struct {
        name    string
        req     *pb.GetUserRequest
        want    *pb.User
        wantErr codes.Code
    }{
        {
            name: "existing user",
            req:  &pb.GetUserRequest{Id: "user-1"},
            want: &pb.User{Id: "user-1", Email: "test@example.com"},
        },
        {
            name:    "non-existing user",
            req:     &pb.GetUserRequest{Id: "user-999"},
            wantErr: codes.NotFound,
        },
        {
            name:    "empty id",
            req:     &pb.GetUserRequest{Id: ""},
            wantErr: codes.InvalidArgument,
        },
    }

    for _, tt := range tests {
        t.Run(tt.name, func(t *testing.T) {
            got, err := service.GetUser(context.Background(), tt.req)

            if tt.wantErr != codes.OK {
                require.Error(t, err)
                st, _ := status.FromError(err)
                assert.Equal(t, tt.wantErr, st.Code())
                return
            }

            require.NoError(t, err)
            assert.Equal(t, tt.want.Id, got.Id)
            assert.Equal(t, tt.want.Email, got.Email)
        })
    }
}
```

### B. Integration Tests

```go
func TestUserService_Integration(t *testing.T) {
    if testing.Short() {
        t.Skip("skipping integration test")
    }

    // Start test server
    lis, err := net.Listen("tcp", "localhost:0")
    require.NoError(t, err)

    server := grpc.NewServer()
    pb.RegisterUserServiceServer(server, NewUserService(testRepo))

    go server.Serve(lis)
    defer server.Stop()

    // Create client
    conn, err := grpc.Dial(lis.Addr().String(), grpc.WithInsecure())
    require.NoError(t, err)
    defer conn.Close()

    client := pb.NewUserServiceClient(conn)

    t.Run("create and get user", func(t *testing.T) {
        ctx := context.Background()

        // Create user
        created, err := client.CreateUser(ctx, &pb.CreateUserRequest{
            User: &pb.User{
                Email:       "integration@test.com",
                DisplayName: "Test User",
            },
        })
        require.NoError(t, err)
        assert.NotEmpty(t, created.Id)

        // Get user
        fetched, err := client.GetUser(ctx, &pb.GetUserRequest{Id: created.Id})
        require.NoError(t, err)
        assert.Equal(t, created.Id, fetched.Id)
        assert.Equal(t, "integration@test.com", fetched.Email)
    })
}
```

---

## 8. Performance (MANDATORY)

### A. Connection Management

```go
// Client connection pool
type ClientPool struct {
    conns []*grpc.ClientConn
    index uint64
}

func NewClientPool(target string, size int) (*ClientPool, error) {
    pool := &ClientPool{
        conns: make([]*grpc.ClientConn, size),
    }

    for i := 0; i < size; i++ {
        conn, err := grpc.Dial(target,
            grpc.WithTransportCredentials(insecure.NewCredentials()),
            grpc.WithKeepaliveParams(keepalive.ClientParameters{
                Time:                10 * time.Second,
                Timeout:             3 * time.Second,
                PermitWithoutStream: true,
            }),
        )
        if err != nil {
            return nil, err
        }
        pool.conns[i] = conn
    }

    return pool, nil
}

func (p *ClientPool) Get() *grpc.ClientConn {
    idx := atomic.AddUint64(&p.index, 1)
    return p.conns[idx%uint64(len(p.conns))]
}
```

### B. Server Configuration

```go
server := grpc.NewServer(
    grpc.MaxRecvMsgSize(10 * 1024 * 1024), // 10MB
    grpc.MaxSendMsgSize(10 * 1024 * 1024),
    grpc.KeepaliveParams(keepalive.ServerParameters{
        MaxConnectionIdle:     15 * time.Minute,
        MaxConnectionAge:      30 * time.Minute,
        MaxConnectionAgeGrace: 5 * time.Minute,
        Time:                  5 * time.Minute,
        Timeout:               1 * time.Minute,
    }),
    grpc.KeepaliveEnforcementPolicy(keepalive.EnforcementPolicy{
        MinTime:             1 * time.Minute,
        PermitWithoutStream: true,
    }),
)
```

---

## 9. Deployment Checklist

### Proto Design
- [ ] Use proto3 syntax
- [ ] Package names follow conventions
- [ ] Messages use appropriate types
- [ ] Enums have UNSPECIFIED as first value
- [ ] Field numbers never reused

### Service Implementation
- [ ] Proper error codes used
- [ ] Deadlines/timeouts configured
- [ ] Interceptors for logging, auth, recovery
- [ ] Health checks implemented

### Performance
- [ ] Connection pooling configured
- [ ] Keepalive settings tuned
- [ ] Message sizes appropriate
- [ ] Streaming for large data

### Security
- [ ] TLS enabled in production
- [ ] Authentication interceptor
- [ ] Input validation
- [ ] Rate limiting

---

## 10. Quick Reference

```protobuf
// Message types
string, bytes, bool
int32, int64, uint32, uint64
float, double
enum MyEnum { UNSPECIFIED = 0; VALUE = 1; }
repeated Type field = N;
map<KeyType, ValueType> field = N;
oneof name { Type1 a = 1; Type2 b = 2; }

// Common imports
import "google/protobuf/timestamp.proto";
import "google/protobuf/duration.proto";
import "google/protobuf/empty.proto";
import "google/protobuf/wrappers.proto";
import "google/protobuf/field_mask.proto";
```

```go
// Status codes
codes.OK              // 0 - Success
codes.Cancelled       // 1 - Cancelled by client
codes.Unknown         // 2 - Unknown error
codes.InvalidArgument // 3 - Invalid argument
codes.DeadlineExceeded // 4 - Timeout
codes.NotFound        // 5 - Not found
codes.AlreadyExists   // 6 - Already exists
codes.PermissionDenied // 7 - No permission
codes.ResourceExhausted // 8 - Rate limit
codes.Internal        // 13 - Server error
codes.Unavailable     // 14 - Service unavailable
codes.Unauthenticated // 16 - Not authenticated
```

---

**Last Updated:** 2026-01-31
**Version:** 1.0
**Maintainer:** Platform Team
