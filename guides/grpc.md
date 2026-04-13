# gRPC Development Guidelines
Mandatory standards for designing and implementing gRPC services. Protocol Buffers 3, gRPC, grpcurl, Evans, Buf, grpc-gateway.

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

## 2A. Test-Driven Development (TDD) Protocol (MANDATORY)

**CRITICAL: Follow the Red-Green-Refactor cycle for ALL new code.**

### TDD Cycle

```
1. RED: Write a failing test first
   ↓
2. GREEN: Write minimal code to make it pass
   ↓
3. REFACTOR: Improve code while keeping tests green
   ↓
   Repeat
```

### Example TDD Workflow for gRPC

```go
// Step 1: RED - Write failing test
package service_test

import (
    "context"
    "testing"

    pb "github.com/myorg/myapp/gen/go/api/v1"
    "github.com/stretchr/testify/assert"
    "github.com/stretchr/testify/require"
    "google.golang.org/grpc/codes"
    "google.golang.org/grpc/status"
)

func TestUserService_CreateUser(t *testing.T) {
    svc := NewUserService(NewMockUserRepository())

    req := &pb.CreateUserRequest{
        User: &pb.User{
            Email:       "alice@example.com",
            DisplayName: "Alice",
            Role:        pb.UserRole_USER_ROLE_USER,
        },
        RequestId: "req-001",
    }

    resp, err := svc.CreateUser(context.Background(), req)

    require.NoError(t, err)
    assert.NotEmpty(t, resp.Id)
    assert.Equal(t, "alice@example.com", resp.Email)
    assert.Equal(t, "Alice", resp.DisplayName)
}
// FAILS - CreateUser not implemented yet

// Step 2: GREEN - Implement the RPC method
func (s *UserService) CreateUser(ctx context.Context, req *pb.CreateUserRequest) (*pb.User, error) {
    if req.User == nil {
        return nil, status.Error(codes.InvalidArgument, "user is required")
    }

    user, err := s.repo.Create(ctx, req.User)
    if err != nil {
        return nil, status.Errorf(codes.Internal, "failed to create user: %v", err)
    }

    return user, nil
}
// PASSES

// Step 3: REFACTOR - Add idempotency check, input validation, logging
func (s *UserService) CreateUser(ctx context.Context, req *pb.CreateUserRequest) (*pb.User, error) {
    if err := s.validator.Validate(req); err != nil {
        return nil, status.Errorf(codes.InvalidArgument, "validation failed: %v", err)
    }

    if req.RequestId != "" {
        if existing, err := s.repo.FindByRequestID(ctx, req.RequestId); err == nil {
            return existing, nil // Idempotent: return previously created user
        }
    }

    user, err := s.repo.Create(ctx, req.User)
    if err != nil {
        return nil, status.Errorf(codes.Internal, "failed to create user: %v", err)
    }

    return user, nil
}
// All tests still PASS
```

### gRPC-Specific TDD Practices

- Use Go's `testing` package (or the equivalent in your language: pytest, JUnit) for unit tests.
- Test service implementations with mock repositories, not live servers.
- Validate proto message constraints and gRPC status codes in every test case.
- Use table-driven tests for comprehensive coverage of success and error paths.
- Test streaming RPCs by verifying send/receive sequences.

---

## 2B. Bug Fix Protocol (MANDATORY)

**CRITICAL: Every bug MUST receive a regression test BEFORE fixing.**

### Bug Fix Workflow

```
1. Bug Reported/Discovered
   ↓
2. Write a test that REPRODUCES the bug (test will FAIL)
   ↓
3. Verify the test fails for the right reason
   ↓
4. Fix the bug (make the test pass)
   ↓
5. Verify the test now PASSES
   ↓
6. Document the bug in test comments (include bug ID)
   ↓
7. Deploy with confidence (regression prevented)
```

### Example Bug Fix

```go
// Bug Report: BUG-2890 - GetUser returns Internal error instead of
// NotFound when user does not exist.

func TestGetUser_NotFound_Bug2890(t *testing.T) {
    mockRepo := &MockUserRepository{
        users: map[string]*pb.User{}, // empty repository
    }
    svc := NewUserService(mockRepo)

    resp, err := svc.GetUser(context.Background(), &pb.GetUserRequest{
        Id: "non-existent-user-id",
    })

    // BUG-2890: Previously returned codes.Internal instead of codes.NotFound
    require.Error(t, err)
    assert.Nil(t, resp)

    st, ok := status.FromError(err)
    require.True(t, ok)
    assert.Equal(t, codes.NotFound, st.Code(), "BUG-2890: must return NotFound for missing user")
}

// Fix: Updated GetUser to check for ErrNotFound from the repository
// and return status.Errorf(codes.NotFound, ...) instead of codes.Internal.
//
// func (s *UserService) GetUser(ctx context.Context, req *pb.GetUserRequest) (*pb.User, error) {
//     user, err := s.repo.FindByID(ctx, req.Id)
//     if err != nil {
//         if errors.Is(err, ErrNotFound) {
//             return nil, status.Errorf(codes.NotFound, "user %s not found", req.Id)
//         }
//         return nil, status.Errorf(codes.Internal, "failed to fetch user: %v", err)
//     }
//     return user, nil
// }
```

### Prohibited Practices for Bug Fixes

**NEVER:**
- Fix a bug without adding a regression test first
- Write implementation before writing tests (violates TDD)
- Skip the Red-Green-Refactor cycle
- Commit code with failing tests
- Remove tests to make code pass
- Skip validation of proto message constraints and gRPC status codes

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

## 9. Security & Dependency Management (MANDATORY)

### A. Infrastructure Security Scanning

```bash
# Scan protobuf/gRPC library dependencies based on your language stack

# Go projects
govulncheck ./...
go list -m all | nancy sleuth

# Python projects
pip-audit
pip-audit --requirement requirements.txt --fix --dry-run
safety check --full-report

# Node.js projects (grpc-js, protobufjs)
npm audit
npm audit fix --dry-run

# Java/Kotlin projects (grpc-java, protobuf-java)
mvn dependency-check:check
gradle dependencyCheckAnalyze

# Buf - lint and breaking change detection for proto files
buf lint
buf breaking --against '.git#branch=main'
```

### B. Vulnerability Scanning

```bash
# TLS/mTLS configuration verification
# Verify server TLS certificate
openssl s_client -connect grpc-server:443 -alpn h2

# Test gRPC health with TLS
grpcurl -cacert ca.pem -cert client.pem -key client-key.pem \
  grpc-server:443 grpc.health.v1.Health/Check

# Scan container images running gRPC services
trivy image myregistry.io/grpc-service:latest
trivy image --severity CRITICAL,HIGH --exit-code 1 myregistry.io/grpc-service:latest

# Scan IaC for gRPC service deployments
trivy config ./k8s/
checkov -d ./k8s/
```

### C. Policy & Compliance

```go
// TLS/mTLS - MANDATORY for production gRPC services
// Server-side mTLS configuration
creds, err := credentials.NewServerTLSFromFile("server.pem", "server-key.pem")
if err != nil {
    log.Fatalf("failed to load TLS credentials: %v", err)
}

// mTLS - require client certificates
cert, _ := tls.LoadX509KeyPair("server.pem", "server-key.pem")
caCert, _ := os.ReadFile("ca.pem")
caPool := x509.NewCertPool()
caPool.AppendCertsFromPEM(caCert)

tlsConfig := &tls.Config{
    Certificates: []tls.Certificate{cert},
    ClientCAs:    caPool,
    ClientAuth:   tls.RequireAndVerifyClientCert,
    MinVersion:   tls.VersionTLS13,
}

server := grpc.NewServer(
    grpc.Creds(credentials.NewTLS(tlsConfig)),
)
```

```protobuf
// Input validation for protobuf messages - use protovalidate
// Enforce field constraints to prevent malformed input
message CreateOrderRequest {
  string customer_id = 1 [(validate.rules).string.uuid = true];
  repeated OrderItem items = 2 [(validate.rules).repeated = {min_items: 1, max_items: 100}];
  string currency = 3 [(validate.rules).string = {in: ["USD", "EUR", "GBP"]}];
  int64 amount_cents = 4 [(validate.rules).int64 = {gte: 1, lte: 999999999}];
}
```

```bash
# Certificate management best practices
# Use short-lived certificates with automatic rotation
# Integrate with cert-manager (Kubernetes) or Vault PKI
# Monitor certificate expiry
openssl x509 -in server.pem -noout -enddate
```

---

## 10. Deployment Checklist

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

## 11. Quick Reference

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

## 12. Why This Configuration Works

1. **Proto-First Schema Design**: Defining services and messages in `.proto` files before writing any implementation code ensures a language-neutral contract. Code generation produces type-safe clients and servers in any supported language, eliminating hand-written serialization bugs.

2. **HTTP/2 with Binary Serialization**: gRPC's use of HTTP/2 multiplexing and Protocol Buffer binary encoding delivers significantly lower latency and bandwidth usage compared to JSON-over-HTTP/1.1, making it ideal for high-throughput microservice communication.

3. **Built-in Deadlines and Cancellation**: Propagating deadlines through the entire call chain ensures that slow downstream services cannot cause cascading timeouts. Automatic cancellation frees resources immediately when a client disconnects or a deadline expires.

4. **Interceptor Middleware Pattern**: Layering authentication, logging, metrics, and recovery logic as interceptors keeps service implementations focused on business logic. Interceptors compose cleanly and can be shared across all services in a deployment.

5. **Streaming for Large Data and Real-Time Updates**: Server streaming, client streaming, and bidirectional streaming provide first-class support for use cases that REST APIs handle awkwardly (file uploads, live feeds, long-running operations), all with the same type-safe contract.

---

**Last Updated:** 2026-01-31
**Version:** 1.0
**Maintainer:** Platform Team


**End of gRPC Development Guidelines**
