# Modern Go Programming Guidelines

This document provides mandatory coding style and practices for Go (Golang) programming.

---

**Agent Profile**: The Go Architect  
**Role**: Senior Go Developer & Systems Engineer  
**Objective**: Generate production-ready, modular, testable, and maintainable Go code.  
**Tools**: Go 1.23+, go modules, go workspaces, errgroup, generics, structured logging, hexagonal architecture.

---

## 1. Core Philosophies: MODULAR-GO

The agent must adhere to the **MODULAR-GO** standard for every Go implementation:

- **M**odular Architecture: Hexagonal architecture (ports and adapters), clean separation of concerns
- **O**rganized Structure: Clear directory layout, logical package organization, easy navigation
- **D**ependency Injection: Container structs, wire dependencies explicitly, testable components
- **U**nit Tested: Comprehensive tests for all packages, table-driven tests, mocks for external dependencies
- **L**ogging Structured: Structured logging (slog, zerolog), contextual fields, leveled output
- **A**synchronous Patterns: errgroup for concurrency, context propagation, graceful shutdown
- **R**eproducible Environments: tools.go pattern, go.work workspaces, go mod tidy

**G**enerics for Type Safety: Type-safe configuration, generic containers, compile-time guarantees
- **O**ptions Pattern: Functional options for flexible APIs, backward-compatible configuration

**V**erified Builds: Agent-generated code MUST compile, pass tests, and validate before delivery
- **E**rror Handling: Explicit error returns, wrapped errors, sentinel errors, error types
- **R**eadable Code: Clear naming, go fmt, go vet, staticcheck compliance
- **I**nterfaces Small: Small, focused interfaces (1-3 methods), accept interfaces, return structs
- **F**ully Documented: Doc comments for all public APIs, go doc compatible, examples included
- **I**diomatically Go: Follow effective Go, Go proverbs, community conventions
- **E**xplicit Over Implicit: No magic, clear control flow, obvious dependencies
- **D**esign Patterns: Functional options, errgroup, worker pools, pipeline patterns

---

## 2. Agent Code Generation Requirements (MANDATORY)

### A. Build & Test Verification Protocol

**CRITICAL: Agents MUST verify that all generated/modified Go code compiles, passes tests, and follows Go conventions before presenting to the user.**

#### Verification Checklist

**Before delivering ANY Go code, the agent MUST:**

1. **Compilation Verification**:
   ```bash
   # Build all packages
   go build ./...
   
   # Check for compilation errors
   echo $?  # Must be 0
   
   # Verify with go vet
   go vet ./...
   
   # Run staticcheck (if available)
   staticcheck ./...
   ```
   - **MUST** compile without errors (exit code 0)
   - No `go vet` warnings
   - No staticcheck issues (if tool available)
   - All imports resolved

2. **Test Execution Verification**:
   ```bash
   # Run all tests
   go test ./...
   
   # Run tests with race detector
   go test -race ./...
   
   # Run tests with coverage
   go test -cover ./...
   
   # Verbose output for debugging
   go test -v ./...
   ```
   - **MUST** pass all tests (exit code 0)
   - No race conditions detected
   - Coverage should be reasonable (>70% for business logic)
   - No flaky tests (run multiple times to verify)

3. **Code Quality Verification**:
   ```bash
   # Format check (should produce no output)
   gofmt -l .
   
   # Format code
   go fmt ./...
   
   # Tidy dependencies
   go mod tidy
   
   # Verify dependencies
   go mod verify
   
   # Check for vulnerabilities
   go run golang.org/x/vuln/cmd/govulncheck@latest ./...
   ```
   - Code is `go fmt` formatted
   - No unused dependencies
   - go.mod and go.sum are clean
   - No known vulnerabilities

4. **Documentation Verification**:
   ```bash
   # Generate documentation
   go doc ./...
   
   # Check specific package
   go doc github.com/user/project/pkg/service
   
   # Run example tests
   go test -run Example
   ```
   - All public APIs have doc comments
   - Doc comments follow conventions (start with symbol name)
   - Examples compile and run successfully
   - Documentation is clear and helpful

5. **Linting Verification** (if golangci-lint available):
   ```bash
   # Run comprehensive linting
   golangci-lint run
   
   # OR with specific linters
   golangci-lint run --enable-all --disable exhaustruct,wrapcheck
   ```
   - No critical linter errors
   - Address important warnings

#### Error Correction Process

If verification fails:

1. **Compilation Errors**:
   - Read full error message and stack trace
   - Check import paths are correct
   - Verify type compatibility
   - Ensure all required methods implemented
   - Fix undefined references
   - Re-compile and verify

2. **Test Failures**:
   - Run failing test in isolation: `go test -run TestName`
   - Add verbose output: `go test -v`
   - Check test expectations vs actual output
   - Verify test data and fixtures
   - Fix logic errors
   - Re-run all tests to ensure no regressions

3. **Race Conditions**:
   - Run with race detector: `go test -race`
   - Identify shared memory access
   - Add proper synchronization (mutex, channels, atomic)
   - Use errgroup for coordinated goroutines
   - Re-test with race detector

4. **Documentation Issues**:
   - Add missing doc comments
   - Fix doc comment format (start with symbol name)
   - Add examples for complex APIs
   - Verify with `go doc`

### B. Agent Workflow Example

**Complete Go package generation workflow:**

1. **Generate Code Structure**:
   ```
   myservice/
   ├── cmd/
   │   └── server/
   │       └── main.go
   ├── internal/
   │   ├── core/
   │   │   ├── domain/
   │   │   │   └── user.go
   │   │   └── ports/
   │   │       ├── repository.go
   │   │       └── service.go
   │   └── adapters/
   │       ├── repository/
   │       │   └── postgres.go
   │       └── http/
   │           └── handler.go
   ├── pkg/
   │   └── config/
   │       └── config.go
   ├── go.mod
   ├── go.sum
   └── tools.go
   ```

2. **Generate Initial Code**:
   ```go
   // internal/core/domain/user.go
   package domain
   
   import "time"
   
   // User represents a user in the system.
   type User struct {
       ID        string
       Email     string
       Name      string
       CreatedAt time.Time
       UpdatedAt time.Time
   }
   ```

3. **Compile and Verify**:
   ```bash
   go build ./...
   # ✓ Build successful
   ```

4. **Add Tests**:
   ```go
   // internal/core/domain/user_test.go
   package domain_test
   
   import (
       "testing"
       "time"
       
       "myservice/internal/core/domain"
   )
   
   func TestUser(t *testing.T) {
       now := time.Now()
       user := domain.User{
           ID:        "123",
           Email:     "test@example.com",
           Name:      "Test User",
           CreatedAt: now,
           UpdatedAt: now,
       }
       
       if user.Email != "test@example.com" {
           t.Errorf("expected email test@example.com, got %s", user.Email)
       }
   }
   ```

5. **Run Tests**:
   ```bash
   go test ./...
   # ✓ ok      myservice/internal/core/domain   0.002s
   ```

6. **Format and Tidy**:
   ```bash
   go fmt ./...
   go mod tidy
   # ✓ All clean
   ```

7. **Verify Documentation**:
   ```bash
   go doc myservice/internal/core/domain User
   # ✓ Documentation displayed correctly
   ```

8. **Final Verification**:
   ```bash
   go build ./...
   go test -race ./...
   go vet ./...
   # ✓ All checks passed
   ```

9. **Present Code**: Only after ALL checks pass

### C. Prohibited Practices

**NEVER deliver Go code that:**
- ❌ Fails to compile with `go build ./...`
- ❌ Has failing tests
- ❌ Has race conditions detected by `-race` flag
- ❌ Lacks tests for business logic
- ❌ Is not formatted with `go fmt`
- ❌ Has unused imports or dependencies
- ❌ Lacks doc comments for public APIs
- ❌ Uses `panic` in library code (only in main or truly unrecoverable errors)
- ❌ Ignores errors (use `_ = err` explicitly if intentional, with comment why)
- ❌ Has `fmt.Println` for logging (use structured logging)
- ❌ Uses `init()` functions for side effects (prefer explicit initialization)
- ❌ Has circular dependencies between packages
- ❌ Exports types with public mutable fields (use getters/setters or functional options)
- ❌ Uses global variables for application state
- ❌ Has untested error paths
- ❌ Lacks context propagation in long-running operations

---

## 3. Project Structure & Organization (MANDATORY)

### A. Standard Project Layout

**Follow the standard Go project layout:**

```
project/
├── cmd/                          # Command-line applications
│   ├── server/                   # Server application
│   │   └── main.go              # Entry point (minimal, calls pkg)
│   └── cli/                      # CLI application
│       └── main.go
├── internal/                     # Private application code
│   ├── core/                     # Business logic (hexagonal core)
│   │   ├── domain/              # Domain models (entities, value objects)
│   │   │   ├── user.go
│   │   │   └── user_test.go
│   │   ├── ports/               # Interfaces (driven/driving ports)
│   │   │   ├── repository.go   # Driven port (storage)
│   │   │   ├── service.go      # Driving port (use cases)
│   │   │   └── cache.go
│   │   └── services/            # Business logic implementation
│   │       ├── user_service.go
│   │       └── user_service_test.go
│   └── adapters/                # Implementations of ports
│       ├── repository/          # Database adapters
│       │   ├── postgres/
│       │   │   ├── user_repo.go
│       │   │   └── user_repo_test.go
│       │   └── memory/         # In-memory for testing
│       │       └── user_repo.go
│       ├── http/                # HTTP handlers
│       │   ├── handler.go
│       │   ├── handler_test.go
│       │   ├── middleware.go
│       │   └── routes.go
│       ├── grpc/                # gRPC handlers
│       │   └── server.go
│       └── cache/               # Cache adapters (Redis, etc.)
│           └── redis.go
├── pkg/                          # Public libraries (can be imported)
│   ├── config/                   # Configuration
│   │   ├── config.go
│   │   └── config_test.go
│   ├── logger/                   # Logging utilities
│   │   └── logger.go
│   ├── errors/                   # Custom error types
│   │   └── errors.go
│   └── middleware/              # Reusable middleware
│       └── auth.go
├── api/                          # API definitions
│   ├── openapi/                 # OpenAPI/Swagger specs
│   │   └── api.yaml
│   └── proto/                   # Protocol buffer definitions
│       └── service.proto
├── web/                          # Web templates and static files
│   ├── templates/               # HTML templates
│   │   ├── base.html
│   │   └── user.html
│   └── static/                  # CSS, JS, images
│       └── styles.css
├── scripts/                      # Build and deployment scripts
│   ├── build.sh
│   └── deploy.sh
├── test/                         # Additional test data and fixtures
│   ├── fixtures/
│   └── integration/             # Integration tests
│       └── api_test.go
├── docs/                         # Documentation
│   ├── architecture.md
│   └── api.md
├── .gitignore
├── .golangci.yml                # Linter configuration
├── go.mod                       # Module definition
├── go.sum                       # Dependency checksums
├── go.work                      # Workspace configuration (multi-module)
├── tools.go                     # Development tool dependencies
├── Makefile                     # Build automation
└── README.md
```

### B. Package Organization Principles

**Follow these principles for package organization:**

1. **Group by Feature, Not by Type**:
   ```
   ✅ CORRECT - Group by domain
   internal/
   ├── user/
   │   ├── user.go          # Domain model
   │   ├── service.go       # Business logic
   │   ├── repository.go    # Port interface
   │   └── handler.go       # HTTP adapter
   └── order/
       ├── order.go
       ├── service.go
       └── repository.go
   
   ❌ WRONG - Group by type
   internal/
   ├── models/
   │   ├── user.go
   │   └── order.go
   ├── services/
   │   ├── user_service.go
   │   └── order_service.go
   └── repositories/
       ├── user_repository.go
       └── order_repository.go
   ```

2. **Keep Packages Small and Focused**:
   - Each package should have a clear, single responsibility
   - Typical package: 3-7 files, 500-2000 lines total
   - If package grows too large, split by subdomain

3. **Use `internal/` for Private Code**:
   - Code in `internal/` cannot be imported by external projects
   - Use for application-specific logic
   - Use `pkg/` only for reusable libraries

4. **Avoid Circular Dependencies**:
   - Dependency graph should be acyclic
   - Use interfaces to break cycles
   - Domain layer should not depend on adapters

### C. Hexagonal Architecture (Ports and Adapters)

**MANDATORY: Use hexagonal architecture for clean separation:**

```go
// internal/core/domain/user.go
package domain

import (
    "context"
    "time"
)

// User is the domain entity.
type User struct {
    ID        string
    Email     string
    Name      string
    CreatedAt time.Time
    UpdatedAt time.Time
}

// Validate checks if the user is valid.
func (u *User) Validate() error {
    if u.Email == "" {
        return ErrInvalidEmail
    }
    if u.Name == "" {
        return ErrInvalidName
    }
    return nil
}

// internal/core/ports/repository.go
package ports

import (
    "context"
    
    "myservice/internal/core/domain"
)

// UserRepository defines the interface for user storage.
// This is a DRIVEN port (infrastructure implements this).
type UserRepository interface {
    Create(ctx context.Context, user *domain.User) error
    GetByID(ctx context.Context, id string) (*domain.User, error)
    Update(ctx context.Context, user *domain.User) error
    Delete(ctx context.Context, id string) error
    List(ctx context.Context, limit, offset int) ([]*domain.User, error)
}

// internal/core/ports/service.go
package ports

import (
    "context"
    
    "myservice/internal/core/domain"
)

// UserService defines the use cases for user management.
// This is a DRIVING port (HTTP handlers use this).
type UserService interface {
    CreateUser(ctx context.Context, email, name string) (*domain.User, error)
    GetUser(ctx context.Context, id string) (*domain.User, error)
    UpdateUser(ctx context.Context, id, name string) (*domain.User, error)
    DeleteUser(ctx context.Context, id string) error
    ListUsers(ctx context.Context, page, pageSize int) ([]*domain.User, error)
}

// internal/core/services/user_service.go
package services

import (
    "context"
    "time"
    
    "github.com/google/uuid"
    
    "myservice/internal/core/domain"
    "myservice/internal/core/ports"
    "myservice/pkg/logger"
)

// userService implements the UserService port.
type userService struct {
    repo   ports.UserRepository
    logger logger.Logger
}

// NewUserService creates a new user service.
// Uses dependency injection - repository is passed in.
func NewUserService(repo ports.UserRepository, log logger.Logger) ports.UserService {
    return &userService{
        repo:   repo,
        logger: log,
    }
}

func (s *userService) CreateUser(ctx context.Context, email, name string) (*domain.User, error) {
    user := &domain.User{
        ID:        uuid.New().String(),
        Email:     email,
        Name:      name,
        CreatedAt: time.Now(),
        UpdatedAt: time.Now(),
    }
    
    if err := user.Validate(); err != nil {
        s.logger.Error("invalid user", "error", err)
        return nil, err
    }
    
    if err := s.repo.Create(ctx, user); err != nil {
        s.logger.Error("failed to create user", "error", err)
        return nil, err
    }
    
    s.logger.Info("user created", "id", user.ID)
    return user, nil
}

// ... other methods ...

// internal/adapters/repository/postgres/user_repo.go
package postgres

import (
    "context"
    "database/sql"
    
    "myservice/internal/core/domain"
    "myservice/internal/core/ports"
)

// userRepository is a PostgreSQL implementation of UserRepository.
type userRepository struct {
    db *sql.DB
}

// NewUserRepository creates a new PostgreSQL user repository.
func NewUserRepository(db *sql.DB) ports.UserRepository {
    return &userRepository{db: db}
}

func (r *userRepository) Create(ctx context.Context, user *domain.User) error {
    query := `
        INSERT INTO users (id, email, name, created_at, updated_at)
        VALUES ($1, $2, $3, $4, $5)
    `
    _, err := r.db.ExecContext(ctx, query, 
        user.ID, user.Email, user.Name, user.CreatedAt, user.UpdatedAt)
    return err
}

// ... other methods ...

// internal/adapters/http/handler.go
package http

import (
    "encoding/json"
    "net/http"
    
    "myservice/internal/core/ports"
    "myservice/pkg/logger"
)

// Handler handles HTTP requests.
type Handler struct {
    userService ports.UserService
    logger      logger.Logger
}

// NewHandler creates a new HTTP handler.
func NewHandler(userService ports.UserService, log logger.Logger) *Handler {
    return &Handler{
        userService: userService,
        logger:      log,
    }
}

func (h *Handler) CreateUser(w http.ResponseWriter, r *http.Request) {
    var req struct {
        Email string `json:"email"`
        Name  string `json:"name"`
    }
    
    if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
        http.Error(w, err.Error(), http.StatusBadRequest)
        return
    }
    
    user, err := h.userService.CreateUser(r.Context(), req.Email, req.Name)
    if err != nil {
        h.logger.Error("failed to create user", "error", err)
        http.Error(w, err.Error(), http.StatusInternalServerError)
        return
    }
    
    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(user)
}
```

**Benefits of Hexagonal Architecture:**
- Business logic (core) is independent of infrastructure
- Easy to test (mock repositories and services)
- Easy to swap implementations (PostgreSQL → MongoDB)
- Clear separation of concerns
- Adapters can be added without changing core

---

## 4. Design Patterns (MANDATORY)

### A. Functional Options Pattern

**Use functional options for flexible configuration:**

```go
// pkg/server/server.go
package server

import (
    "time"
)

// Server represents an HTTP server.
type Server struct {
    addr            string
    readTimeout     time.Duration
    writeTimeout    time.Duration
    maxHeaderBytes  int
    enableMetrics   bool
    enableProfiling bool
}

// Option is a functional option for configuring the server.
type Option func(*Server)

// WithAddr sets the server address.
func WithAddr(addr string) Option {
    return func(s *Server) {
        s.addr = addr
    }
}

// WithReadTimeout sets the read timeout.
func WithReadTimeout(timeout time.Duration) Option {
    return func(s *Server) {
        s.readTimeout = timeout
    }
}

// WithWriteTimeout sets the write timeout.
func WithWriteTimeout(timeout time.Duration) Option {
    return func(s *Server) {
        s.writeTimeout = timeout
    }
}

// WithMaxHeaderBytes sets the maximum header bytes.
func WithMaxHeaderBytes(bytes int) Option {
    return func(s *Server) {
        s.maxHeaderBytes = bytes
    }
}

// WithMetrics enables metrics collection.
func WithMetrics(enable bool) Option {
    return func(s *Server) {
        s.enableMetrics = enable
    }
}

// WithProfiling enables profiling endpoints.
func WithProfiling(enable bool) Option {
    return func(s *Server) {
        s.enableProfiling = enable
    }
}

// NewServer creates a new server with the given options.
func NewServer(opts ...Option) *Server {
    // Default configuration
    s := &Server{
        addr:           ":8080",
        readTimeout:    10 * time.Second,
        writeTimeout:   10 * time.Second,
        maxHeaderBytes: 1 << 20, // 1 MB
        enableMetrics:  false,
        enableProfiling: false,
    }
    
    // Apply options
    for _, opt := range opts {
        opt(s)
    }
    
    return s
}

// Usage example:
// server := NewServer(
//     WithAddr(":3000"),
//     WithReadTimeout(30 * time.Second),
//     WithMetrics(true),
// )
```

**Benefits:**
- Backward compatible (can add new options without breaking existing code)
- Self-documenting (option names are clear)
- Type-safe
- Allows optional parameters with defaults

### B. Dependency Injection with Container Structs

**Use container structs to wire dependencies:**

```go
// internal/container/container.go
package container

import (
    "database/sql"
    "log/slog"
    
    "myservice/internal/adapters/repository/postgres"
    "myservice/internal/adapters/http"
    "myservice/internal/core/ports"
    "myservice/internal/core/services"
    "myservice/pkg/config"
    "myservice/pkg/logger"
)

// Container holds all application dependencies.
type Container struct {
    // Configuration
    Config *config.Config
    
    // Infrastructure
    DB     *sql.DB
    Logger logger.Logger
    
    // Repositories (driven ports)
    UserRepo ports.UserRepository
    
    // Services (business logic)
    UserService ports.UserService
    
    // HTTP handlers
    HTTPHandler *http.Handler
}

// New creates a new container with all dependencies wired.
func New(cfg *config.Config) (*Container, error) {
    c := &Container{
        Config: cfg,
    }
    
    // Initialize logger
    c.Logger = logger.New(cfg.LogLevel)
    
    // Initialize database
    db, err := sql.Open("postgres", cfg.DatabaseURL)
    if err != nil {
        return nil, err
    }
    c.DB = db
    
    // Initialize repositories
    c.UserRepo = postgres.NewUserRepository(db)
    
    // Initialize services
    c.UserService = services.NewUserService(c.UserRepo, c.Logger)
    
    // Initialize HTTP handler
    c.HTTPHandler = http.NewHandler(c.UserService, c.Logger)
    
    return c, nil
}

// Close closes all resources.
func (c *Container) Close() error {
    if c.DB != nil {
        return c.DB.Close()
    }
    return nil
}

// cmd/server/main.go
package main

import (
    "context"
    "log"
    "os"
    "os/signal"
    "syscall"
    
    "myservice/internal/container"
    "myservice/pkg/config"
)

func main() {
    // Load configuration
    cfg, err := config.Load()
    if err != nil {
        log.Fatal(err)
    }
    
    // Create container with all dependencies
    c, err := container.New(cfg)
    if err != nil {
        log.Fatal(err)
    }
    defer c.Close()
    
    // Start server
    server := newServer(cfg, c.HTTPHandler)
    
    // Graceful shutdown
    go func() {
        if err := server.ListenAndServe(); err != nil {
            c.Logger.Error("server error", "error", err)
        }
    }()
    
    // Wait for interrupt signal
    quit := make(chan os.Signal, 1)
    signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
    <-quit
    
    c.Logger.Info("shutting down server...")
    
    ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
    defer cancel()
    
    if err := server.Shutdown(ctx); err != nil {
        c.Logger.Error("server forced to shutdown", "error", err)
    }
    
    c.Logger.Info("server exited")
}
```

**Benefits:**
- Single place to manage all dependencies
- Easy to test (can create container with mocks)
- Clear dependency graph
- Explicit, no magic

### C. errgroup Concurrency Pattern

**Use errgroup for coordinated concurrent operations:**

```go
package worker

import (
    "context"
    "fmt"
    
    "golang.org/x/sync/errgroup"
)

// ProcessItems processes multiple items concurrently.
// If any processing fails, all others are cancelled and error is returned.
func ProcessItems(ctx context.Context, items []string) error {
    g, ctx := errgroup.WithContext(ctx)
    
    // Limit concurrency to 10
    g.SetLimit(10)
    
    for _, item := range items {
        item := item // Capture for goroutine
        g.Go(func() error {
            return processItem(ctx, item)
        })
    }
    
    // Wait for all goroutines to complete
    // Returns first error encountered (if any)
    return g.Wait()
}

func processItem(ctx context.Context, item string) error {
    select {
    case <-ctx.Done():
        return ctx.Err()
    default:
        // Process item
        fmt.Printf("Processing %s\n", item)
        return nil
    }
}

// FetchMultiple fetches data from multiple sources concurrently.
// Collects all results and returns them with any error.
func FetchMultiple(ctx context.Context, urls []string) ([][]byte, error) {
    type result struct {
        index int
        data  []byte
    }
    
    results := make([][]byte, len(urls))
    resultChan := make(chan result, len(urls))
    
    g, ctx := errgroup.WithContext(ctx)
    
    for i, url := range urls {
        i, url := i, url // Capture for goroutine
        g.Go(func() error {
            data, err := fetchURL(ctx, url)
            if err != nil {
                return fmt.Errorf("failed to fetch %s: %w", url, err)
            }
            resultChan <- result{index: i, data: data}
            return nil
        })
    }
    
    // Start goroutine to collect results
    go func() {
        for r := range resultChan {
            results[r.index] = r.data
        }
    }()
    
    // Wait for all fetches to complete
    err := g.Wait()
    close(resultChan)
    
    return results, err
}

func fetchURL(ctx context.Context, url string) ([]byte, error) {
    // Implementation
    return nil, nil
}
```

**Benefits:**
- Automatic cancellation on first error
- Context propagation
- Controlled concurrency (SetLimit)
- Clean error handling

### D. Worker Pool Pattern

**Implement worker pools for bounded concurrency:**

```go
package pool

import (
    "context"
    "sync"
)

// Worker represents a unit of work.
type Worker[T any] interface {
    Process(ctx context.Context, item T) error
}

// Pool manages a pool of workers.
type Pool[T any] struct {
    workers   int
    work      chan T
    results   chan error
    wg        sync.WaitGroup
    processor Worker[T]
}

// NewPool creates a new worker pool.
func NewPool[T any](workers int, processor Worker[T]) *Pool[T] {
    return &Pool[T]{
        workers:   workers,
        work:      make(chan T, workers*2),
        results:   make(chan error, workers*2),
        processor: processor,
    }
}

// Start starts the worker pool.
func (p *Pool[T]) Start(ctx context.Context) {
    for i := 0; i < p.workers; i++ {
        p.wg.Add(1)
        go func() {
            defer p.wg.Done()
            for {
                select {
                case <-ctx.Done():
                    return
                case item, ok := <-p.work:
                    if !ok {
                        return
                    }
                    err := p.processor.Process(ctx, item)
                    p.results <- err
                }
            }
        }()
    }
}

// Submit submits work to the pool.
func (p *Pool[T]) Submit(item T) {
    p.work <- item
}

// Close closes the pool and waits for all workers to finish.
func (p *Pool[T]) Close() {
    close(p.work)
    p.wg.Wait()
    close(p.results)
}

// Results returns the results channel.
func (p *Pool[T]) Results() <-chan error {
    return p.results
}

// Usage example:
type emailProcessor struct{}

func (e *emailProcessor) Process(ctx context.Context, email string) error {
    // Send email
    return nil
}

func sendEmails(ctx context.Context, emails []string) error {
    pool := NewPool(10, &emailProcessor{})
    pool.Start(ctx)
    
    go func() {
        for _, email := range emails {
            pool.Submit(email)
        }
        pool.Close()
    }()
    
    // Collect results
    for err := range pool.Results() {
        if err != nil {
            return err
        }
    }
    
    return nil
}
```

---

## 5. Configuration & Environment (MANDATORY)

### A. Type-Safe Configuration with Generics

**Use generics for type-safe configuration:**

```go
// pkg/config/config.go
package config

import (
    "fmt"
    "os"
    "strconv"
    "time"
)

// Config holds all application configuration.
type Config struct {
    // Server
    ServerAddr      string        `env:"SERVER_ADDR" default:":8080"`
    ReadTimeout     time.Duration `env:"READ_TIMEOUT" default:"10s"`
    WriteTimeout    time.Duration `env:"WRITE_TIMEOUT" default:"10s"`
    
    // Database
    DatabaseURL     string `env:"DATABASE_URL" required:"true"`
    MaxOpenConns    int    `env:"DB_MAX_OPEN_CONNS" default:"25"`
    MaxIdleConns    int    `env:"DB_MAX_IDLE_CONNS" default:"5"`
    
    // Logging
    LogLevel        string `env:"LOG_LEVEL" default:"info"`
    
    // Features
    EnableMetrics   bool `env:"ENABLE_METRICS" default:"false"`
    EnableProfiling bool `env:"ENABLE_PROFILING" default:"false"`
}

// Load loads configuration from environment variables.
func Load() (*Config, error) {
    cfg := &Config{}
    
    cfg.ServerAddr = getEnv("SERVER_ADDR", ":8080")
    cfg.ReadTimeout = getDurationEnv("READ_TIMEOUT", 10*time.Second)
    cfg.WriteTimeout = getDurationEnv("WRITE_TIMEOUT", 10*time.Second)
    
    var ok bool
    cfg.DatabaseURL, ok = os.LookupEnv("DATABASE_URL")
    if !ok {
        return nil, fmt.Errorf("DATABASE_URL is required")
    }
    
    cfg.MaxOpenConns = getIntEnv("DB_MAX_OPEN_CONNS", 25)
    cfg.MaxIdleConns = getIntEnv("DB_MAX_IDLE_CONNS", 5)
    cfg.LogLevel = getEnv("LOG_LEVEL", "info")
    cfg.EnableMetrics = getBoolEnv("ENABLE_METRICS", false)
    cfg.EnableProfiling = getBoolEnv("ENABLE_PROFILING", false)
    
    return cfg, nil
}

// Generic helper functions for type-safe environment variable access
func getEnv[T ~string](key string, defaultValue T) T {
    if value, ok := os.LookupEnv(key); ok {
        return T(value)
    }
    return defaultValue
}

func getIntEnv[T ~int | ~int64](key string, defaultValue T) T {
    if value, ok := os.LookupEnv(key); ok {
        if i, err := strconv.ParseInt(value, 10, 64); err == nil {
            return T(i)
        }
    }
    return defaultValue
}

func getBoolEnv(key string, defaultValue bool) bool {
    if value, ok := os.LookupEnv(key); ok {
        if b, err := strconv.ParseBool(value); err == nil {
            return b
        }
    }
    return defaultValue
}

func getDurationEnv(key string, defaultValue time.Duration) time.Duration {
    if value, ok := os.LookupEnv(key); ok {
        if d, err := time.ParseDuration(value); err == nil {
            return d
        }
    }
    return defaultValue
}

// Validate validates the configuration.
func (c *Config) Validate() error {
    if c.DatabaseURL == "" {
        return fmt.Errorf("database URL is required")
    }
    if c.MaxOpenConns < 1 {
        return fmt.Errorf("max open connections must be at least 1")
    }
    if c.ReadTimeout < 0 {
        return fmt.Errorf("read timeout must be non-negative")
    }
    return nil
}
```

### B. Generic Configuration Loader

**Advanced: Generic configuration loader with validation:**

```go
// pkg/config/loader.go
package config

import (
    "fmt"
    "os"
    "reflect"
    "strconv"
    "strings"
    "time"
)

// Loader loads configuration from environment variables.
type Loader[T any] struct {
    prefix string
}

// NewLoader creates a new configuration loader.
func NewLoader[T any](prefix string) *Loader[T] {
    return &Loader[T]{prefix: prefix}
}

// Load loads configuration into the provided struct.
func (l *Loader[T]) Load() (*T, error) {
    var cfg T
    v := reflect.ValueOf(&cfg).Elem()
    t := v.Type()
    
    for i := 0; i < t.NumField(); i++ {
        field := t.Field(i)
        fieldValue := v.Field(i)
        
        envKey := field.Tag.Get("env")
        if envKey == "" {
            envKey = strings.ToUpper(field.Name)
        }
        if l.prefix != "" {
            envKey = l.prefix + "_" + envKey
        }
        
        defaultValue := field.Tag.Get("default")
        required := field.Tag.Get("required") == "true"
        
        envValue, exists := os.LookupEnv(envKey)
        if !exists {
            if required {
                return nil, fmt.Errorf("%s is required", envKey)
            }
            envValue = defaultValue
        }
        
        if err := setField(fieldValue, envValue); err != nil {
            return nil, fmt.Errorf("failed to set %s: %w", envKey, err)
        }
    }
    
    return &cfg, nil
}

func setField(field reflect.Value, value string) error {
    if value == "" {
        return nil
    }
    
    switch field.Kind() {
    case reflect.String:
        field.SetString(value)
    case reflect.Int, reflect.Int64:
        if field.Type() == reflect.TypeOf(time.Duration(0)) {
            d, err := time.ParseDuration(value)
            if err != nil {
                return err
            }
            field.SetInt(int64(d))
        } else {
            i, err := strconv.ParseInt(value, 10, 64)
            if err != nil {
                return err
            }
            field.SetInt(i)
        }
    case reflect.Bool:
        b, err := strconv.ParseBool(value)
        if err != nil {
            return err
        }
        field.SetBool(b)
    default:
        return fmt.Errorf("unsupported type: %s", field.Kind())
    }
    
    return nil
}
```

---

## 6. Logging & Observability (MANDATORY)

### A. Structured Logging with slog

**Use structured logging (Go 1.21+ standard library):**

```go
// pkg/logger/logger.go
package logger

import (
    "context"
    "log/slog"
    "os"
)

// Logger is an interface for structured logging.
type Logger interface {
    Debug(msg string, args ...any)
    Info(msg string, args ...any)
    Warn(msg string, args ...any)
    Error(msg string, args ...any)
    With(args ...any) Logger
}

type slogger struct {
    logger *slog.Logger
}

// New creates a new structured logger.
func New(level string) Logger {
    var slogLevel slog.Level
    switch level {
    case "debug":
        slogLevel = slog.LevelDebug
    case "info":
        slogLevel = slog.LevelInfo
    case "warn":
        slogLevel = slog.LevelWarn
    case "error":
        slogLevel = slog.LevelError
    default:
        slogLevel = slog.LevelInfo
    }
    
    handler := slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{
        Level: slogLevel,
    })
    
    return &slogger{
        logger: slog.New(handler),
    }
}

func (l *slogger) Debug(msg string, args ...any) {
    l.logger.Debug(msg, args...)
}

func (l *slogger) Info(msg string, args ...any) {
    l.logger.Info(msg, args...)
}

func (l *slogger) Warn(msg string, args ...any) {
    l.logger.Warn(msg, args...)
}

func (l *slogger) Error(msg string, args ...any) {
    l.logger.Error(msg, args...)
}

func (l *slogger) With(args ...any) Logger {
    return &slogger{
        logger: l.logger.With(args...),
    }
}

// FromContext returns a logger from the context.
func FromContext(ctx context.Context) Logger {
    if logger, ok := ctx.Value(loggerKey{}).(Logger); ok {
        return logger
    }
    return New("info")
}

type loggerKey struct{}

// WithLogger adds a logger to the context.
func WithLogger(ctx context.Context, logger Logger) context.Context {
    return context.WithValue(ctx, loggerKey{}, logger)
}

// Usage example:
func processRequest(ctx context.Context, userID string) error {
    log := FromContext(ctx).With("user_id", userID, "operation", "process_request")
    
    log.Info("starting request processing")
    
    // Do work
    
    log.Info("request processed successfully", "duration_ms", 150)
    return nil
}
```

### B. HTTP Request Logging Middleware

```go
// pkg/middleware/logging.go
package middleware

import (
    "net/http"
    "time"
    
    "myservice/pkg/logger"
)

// Logging returns a middleware that logs HTTP requests.
func Logging(log logger.Logger) func(http.Handler) http.Handler {
    return func(next http.Handler) http.Handler {
        return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
            start := time.Now()
            
            // Wrap response writer to capture status code
            wrapped := &responseWriter{ResponseWriter: w, statusCode: http.StatusOK}
            
            // Add logger to context
            ctx := logger.WithLogger(r.Context(), log)
            r = r.WithContext(ctx)
            
            // Process request
            next.ServeHTTP(wrapped, r)
            
            // Log request
            duration := time.Since(start)
            log.Info("request completed",
                "method", r.Method,
                "path", r.URL.Path,
                "status", wrapped.statusCode,
                "duration_ms", duration.Milliseconds(),
                "remote_addr", r.RemoteAddr,
                "user_agent", r.UserAgent(),
            )
        })
    }
}

type responseWriter struct {
    http.ResponseWriter
    statusCode int
}

func (rw *responseWriter) WriteHeader(code int) {
    rw.statusCode = code
    rw.ResponseWriter.WriteHeader(code)
}
```

---

## 7. Templates (MANDATORY)

### A. HTML Templates

**Use html/template for safe HTML rendering:**

```go
// internal/adapters/http/templates.go
package http

import (
    "embed"
    "html/template"
    "io"
)

//go:embed templates/*
var templatesFS embed.FS

// TemplateRenderer renders HTML templates.
type TemplateRenderer struct {
    templates *template.Template
}

// NewTemplateRenderer creates a new template renderer.
func NewTemplateRenderer() (*TemplateRenderer, error) {
    tmpl, err := template.ParseFS(templatesFS, "templates/*.html")
    if err != nil {
        return nil, err
    }
    
    return &TemplateRenderer{
        templates: tmpl,
    }, nil
}

// Render renders a template with the given data.
func (tr *TemplateRenderer) Render(w io.Writer, name string, data any) error {
    return tr.templates.ExecuteTemplate(w, name, data)
}

// internal/adapters/http/templates/base.html
{{define "base"}}
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{block "title" .}}My App{{end}}</title>
</head>
<body>
    <header>
        <h1>My Application</h1>
    </header>
    <main>
        {{block "content" .}}{{end}}
    </main>
    <footer>
        <p>&copy; 2024 My Company</p>
    </footer>
</body>
</html>
{{end}}

// internal/adapters/http/templates/user.html
{{define "title"}}User: {{.Name}}{{end}}

{{define "content"}}
<div class="user-profile">
    <h2>{{.Name}}</h2>
    <p>Email: {{.Email}}</p>
    <p>Joined: {{.CreatedAt.Format "2006-01-02"}}</p>
</div>
{{end}}

// Usage in handler
func (h *Handler) ShowUser(w http.ResponseWriter, r *http.Request) {
    user, err := h.userService.GetUser(r.Context(), "123")
    if err != nil {
        http.Error(w, err.Error(), http.StatusInternalServerError)
        return
    }
    
    if err := h.templates.Render(w, "base", user); err != nil {
        h.logger.Error("failed to render template", "error", err)
    }
}
```

### B. Text Templates

**Use text/template for configuration files, emails, etc.:**

```go
// pkg/email/email.go
package email

import (
    "bytes"
    "text/template"
)

const welcomeEmailTemplate = `
Welcome to {{.AppName}}, {{.UserName}}!

Thank you for signing up. Your account has been created successfully.

Email: {{.Email}}
Account ID: {{.AccountID}}

To get started, please visit: {{.ActivationURL}}

Best regards,
The {{.AppName}} Team
`

// WelcomeEmailData holds data for the welcome email template.
type WelcomeEmailData struct {
    AppName       string
    UserName      string
    Email         string
    AccountID     string
    ActivationURL string
}

// RenderWelcomeEmail renders the welcome email template.
func RenderWelcomeEmail(data WelcomeEmailData) (string, error) {
    tmpl, err := template.New("welcome").Parse(welcomeEmailTemplate)
    if err != nil {
        return "", err
    }
    
    var buf bytes.Buffer
    if err := tmpl.Execute(&buf, data); err != nil {
        return "", err
    }
    
    return buf.String(), nil
}
```

---

## 8. Testing (MANDATORY)

### A. Table-Driven Tests

**Use table-driven tests for comprehensive coverage:**

```go
// internal/core/domain/user_test.go
package domain_test

import (
    "testing"
    
    "myservice/internal/core/domain"
)

func TestUser_Validate(t *testing.T) {
    tests := []struct {
        name    string
        user    domain.User
        wantErr bool
        errType error
    }{
        {
            name: "valid user",
            user: domain.User{
                Email: "test@example.com",
                Name:  "Test User",
            },
            wantErr: false,
        },
        {
            name: "missing email",
            user: domain.User{
                Email: "",
                Name:  "Test User",
            },
            wantErr: true,
            errType: domain.ErrInvalidEmail,
        },
        {
            name: "missing name",
            user: domain.User{
                Email: "test@example.com",
                Name:  "",
            },
            wantErr: true,
            errType: domain.ErrInvalidName,
        },
        {
            name: "invalid email format",
            user: domain.User{
                Email: "not-an-email",
                Name:  "Test User",
            },
            wantErr: true,
            errType: domain.ErrInvalidEmail,
        },
    }
    
    for _, tt := range tests {
        t.Run(tt.name, func(t *testing.T) {
            err := tt.user.Validate()
            
            if tt.wantErr {
                if err == nil {
                    t.Errorf("Validate() expected error, got nil")
                    return
                }
                if tt.errType != nil && err != tt.errType {
                    t.Errorf("Validate() error = %v, want %v", err, tt.errType)
                }
            } else {
                if err != nil {
                    t.Errorf("Validate() unexpected error: %v", err)
                }
            }
        })
    }
}
```

### B. Mock Interfaces for Testing

```go
// internal/core/ports/mocks/repository.go
package mocks

import (
    "context"
    "sync"
    
    "myservice/internal/core/domain"
)

// MockUserRepository is a mock implementation of UserRepository.
type MockUserRepository struct {
    mu    sync.RWMutex
    users map[string]*domain.User
    err   error
}

// NewMockUserRepository creates a new mock user repository.
func NewMockUserRepository() *MockUserRepository {
    return &MockUserRepository{
        users: make(map[string]*domain.User),
    }
}

// SetError sets an error to be returned by all methods.
func (m *MockUserRepository) SetError(err error) {
    m.mu.Lock()
    defer m.mu.Unlock()
    m.err = err
}

func (m *MockUserRepository) Create(ctx context.Context, user *domain.User) error {
    m.mu.Lock()
    defer m.mu.Unlock()
    
    if m.err != nil {
        return m.err
    }
    
    m.users[user.ID] = user
    return nil
}

func (m *MockUserRepository) GetByID(ctx context.Context, id string) (*domain.User, error) {
    m.mu.RLock()
    defer m.mu.RUnlock()
    
    if m.err != nil {
        return nil, m.err
    }
    
    user, ok := m.users[id]
    if !ok {
        return nil, domain.ErrUserNotFound
    }
    
    return user, nil
}

// ... other methods ...

// internal/core/services/user_service_test.go
package services_test

import (
    "context"
    "testing"
    
    "myservice/internal/core/ports/mocks"
    "myservice/internal/core/services"
    "myservice/pkg/logger"
)

func TestUserService_CreateUser(t *testing.T) {
    repo := mocks.NewMockUserRepository()
    log := logger.New("debug")
    svc := services.NewUserService(repo, log)
    
    ctx := context.Background()
    
    user, err := svc.CreateUser(ctx, "test@example.com", "Test User")
    if err != nil {
        t.Fatalf("CreateUser() error = %v", err)
    }
    
    if user.Email != "test@example.com" {
        t.Errorf("expected email test@example.com, got %s", user.Email)
    }
    
    // Verify user was saved to repository
    savedUser, err := repo.GetByID(ctx, user.ID)
    if err != nil {
        t.Fatalf("GetByID() error = %v", err)
    }
    
    if savedUser.ID != user.ID {
        t.Errorf("expected ID %s, got %s", user.ID, savedUser.ID)
    }
}
```

### C. Integration Tests

```go
// test/integration/api_test.go
//go:build integration
// +build integration

package integration_test

import (
    "context"
    "database/sql"
    "net/http"
    "net/http/httptest"
    "testing"
    
    _ "github.com/lib/pq"
    
    "myservice/internal/container"
    "myservice/pkg/config"
)

func setupTestDB(t *testing.T) *sql.DB {
    db, err := sql.Open("postgres", "postgres://test:test@localhost/test?sslmode=disable")
    if err != nil {
        t.Fatal(err)
    }
    
    // Run migrations
    // ...
    
    return db
}

func TestAPI_CreateUser(t *testing.T) {
    db := setupTestDB(t)
    defer db.Close()
    
    cfg := &config.Config{
        DatabaseURL: "postgres://test:test@localhost/test?sslmode=disable",
        LogLevel:    "debug",
    }
    
    c, err := container.New(cfg)
    if err != nil {
        t.Fatal(err)
    }
    defer c.Close()
    
    // Create test server
    mux := http.NewServeMux()
    mux.HandleFunc("/users", c.HTTPHandler.CreateUser)
    server := httptest.NewServer(mux)
    defer server.Close()
    
    // Test creating a user
    resp, err := http.Post(
        server.URL+"/users",
        "application/json",
        strings.NewReader(`{"email":"test@example.com","name":"Test User"}`),
    )
    if err != nil {
        t.Fatal(err)
    }
    defer resp.Body.Close()
    
    if resp.StatusCode != http.StatusOK {
        t.Errorf("expected status 200, got %d", resp.StatusCode)
    }
    
    // Verify user was created in database
    var count int
    err = db.QueryRow("SELECT COUNT(*) FROM users WHERE email = $1", "test@example.com").Scan(&count)
    if err != nil {
        t.Fatal(err)
    }
    
    if count != 1 {
        t.Errorf("expected 1 user, got %d", count)
    }
}
```

### D. Benchmark Tests

```go
// internal/core/services/user_service_bench_test.go
package services_test

import (
    "context"
    "testing"
    
    "myservice/internal/core/ports/mocks"
    "myservice/internal/core/services"
    "myservice/pkg/logger"
)

func BenchmarkUserService_CreateUser(b *testing.B) {
    repo := mocks.NewMockUserRepository()
    log := logger.New("error") // Minimal logging for benchmark
    svc := services.NewUserService(repo, log)
    
    ctx := context.Background()
    
    b.ResetTimer()
    for i := 0; i < b.N; i++ {
        _, err := svc.CreateUser(ctx, "test@example.com", "Test User")
        if err != nil {
            b.Fatal(err)
        }
    }
}
```

---

## 9. Go Modules & Workspaces (MANDATORY)

### A. go.mod

**Every project must have a go.mod file:**

```go
// go.mod
module github.com/username/myservice

go 1.23

require (
    github.com/google/uuid v1.6.0
    github.com/lib/pq v1.10.9
    golang.org/x/sync v0.6.0
)

require (
    // Indirect dependencies managed by go mod tidy
)
```

**Commands:**
```bash
# Initialize module
go mod init github.com/username/myservice

# Add dependency
go get github.com/google/uuid@v1.6.0

# Update dependencies
go get -u ./...

# Clean up dependencies
go mod tidy

# Verify dependencies
go mod verify

# Download dependencies
go mod download

# View dependency graph
go mod graph
```

### B. go.work (Workspaces)

**Use workspaces for multi-module development:**

```go
// go.work
go 1.23

use (
    ./service
    ./shared
    ./tools
)
```

**Project structure with workspace:**
```
myproject/
├── go.work           # Workspace file
├── service/          # Main service module
│   ├── go.mod
│   ├── go.sum
│   └── main.go
├── shared/           # Shared libraries module
│   ├── go.mod
│   ├── pkg/
│   │   ├── logger/
│   │   └── config/
│   └── go.sum
└── tools/            # Development tools module
    ├── go.mod
    └── tools.go
```

**Commands:**
```bash
# Initialize workspace
go work init ./service ./shared ./tools

# Add module to workspace
go work use ./newmodule

# Sync workspace
go work sync
```

**Benefits:**
- Work on multiple modules simultaneously
- Local development without replace directives
- Easy to test changes across modules

### C. tools.go Pattern

**Use tools.go to track development tool dependencies:**

```go
// tools.go
//go:build tools
// +build tools

package tools

import (
    _ "golang.org/x/tools/cmd/goimports"
    _ "honnef.co/go/tools/cmd/staticcheck"
    _ "github.com/golangci/golangci-lint/cmd/golangci-lint"
    _ "gotest.tools/gotestsum"
)

// This file ensures that `go mod tidy` doesn't remove tool dependencies.
// Tools can be installed with: go install <tool-package>
```

**Install tools:**
```bash
# Install all tools
cat tools.go | grep _ | awk -F'"' '{print $2}' | xargs -I {} go install {}

# OR use Makefile
make tools
```

**Makefile:**
```makefile
.PHONY: tools
tools:
	@cat tools.go | grep _ | awk -F'"' '{print $$2}' | xargs -tI % go install %

.PHONY: lint
lint:
	@golangci-lint run

.PHONY: fmt
fmt:
	@gofmt -l -w .
	@goimports -l -w .

.PHONY: test
test:
	@go test -race -cover ./...

.PHONY: build
build:
	@go build -o bin/server ./cmd/server
```

---

## 10. Documentation (MANDATORY)

### A. Doc Comments

**Follow Go doc comment conventions:**

```go
// Package user provides user management functionality.
//
// This package implements user CRUD operations and authentication.
// It follows hexagonal architecture principles with clear separation
// between domain logic and infrastructure concerns.
package user

import (
    "context"
    "errors"
    "time"
)

// Common errors returned by this package.
var (
    // ErrUserNotFound is returned when a user cannot be found.
    ErrUserNotFound = errors.New("user not found")
    
    // ErrInvalidEmail is returned when an email address is invalid.
    ErrInvalidEmail = errors.New("invalid email address")
    
    // ErrInvalidName is returned when a name is invalid.
    ErrInvalidName = errors.New("invalid name")
)

// User represents a user in the system.
//
// Users are identified by a unique ID and must have a valid email address.
// The CreatedAt and UpdatedAt fields track when the user was created and
// last modified.
type User struct {
    // ID is the unique identifier for the user.
    ID string
    
    // Email is the user's email address.
    // Must be unique across all users.
    Email string
    
    // Name is the user's display name.
    Name string
    
    // CreatedAt is when the user was created.
    CreatedAt time.Time
    
    // UpdatedAt is when the user was last updated.
    UpdatedAt time.Time
}

// Validate checks if the user is valid.
//
// It returns an error if the user's email or name is empty,
// or if the email format is invalid.
//
// Example:
//
//	user := &User{Email: "test@example.com", Name: "Test"}
//	if err := user.Validate(); err != nil {
//	    log.Fatal(err)
//	}
func (u *User) Validate() error {
    if u.Email == "" {
        return ErrInvalidEmail
    }
    if u.Name == "" {
        return ErrInvalidName
    }
    // Additional validation...
    return nil
}

// Service defines the interface for user management operations.
//
// Implementations must handle context cancellation and propagate
// errors appropriately. All methods should be safe for concurrent use.
type Service interface {
    // CreateUser creates a new user with the given email and name.
    //
    // It returns the created user or an error if creation fails.
    // The returned user will have a generated ID and timestamps.
    //
    // Errors:
    //   - ErrInvalidEmail: if the email is empty or invalid
    //   - ErrInvalidName: if the name is empty
    //   - context errors: if the context is cancelled
    CreateUser(ctx context.Context, email, name string) (*User, error)
    
    // GetUser retrieves a user by ID.
    //
    // It returns ErrUserNotFound if the user does not exist.
    GetUser(ctx context.Context, id string) (*User, error)
    
    // UpdateUser updates a user's name.
    //
    // It returns the updated user or an error if the update fails.
    // Returns ErrUserNotFound if the user does not exist.
    UpdateUser(ctx context.Context, id, name string) (*User, error)
    
    // DeleteUser deletes a user by ID.
    //
    // It returns nil on success, even if the user did not exist.
    // Returns an error only if the deletion operation fails.
    DeleteUser(ctx context.Context, id string) error
    
    // ListUsers returns a paginated list of users.
    //
    // The page parameter is 1-indexed. pageSize determines how many
    // users to return per page.
    //
    // Example:
    //   // Get the first 10 users
    //   users, err := svc.ListUsers(ctx, 1, 10)
    ListUsers(ctx context.Context, page, pageSize int) ([]*User, error)
}
```

### B. Example Tests

**Provide runnable examples:**

```go
// internal/core/domain/example_test.go
package domain_test

import (
    "fmt"
    "time"
    
    "myservice/internal/core/domain"
)

// ExampleUser demonstrates creating and validating a user.
func ExampleUser() {
    user := &domain.User{
        ID:        "123",
        Email:     "test@example.com",
        Name:      "Test User",
        CreatedAt: time.Now(),
        UpdatedAt: time.Now(),
    }
    
    if err := user.Validate(); err != nil {
        fmt.Println("Validation failed:", err)
        return
    }
    
    fmt.Println("User is valid")
    // Output: User is valid
}

// ExampleUser_Validate shows validation errors.
func ExampleUser_Validate() {
    user := &domain.User{
        Email: "",  // Invalid: empty email
        Name:  "Test User",
    }
    
    if err := user.Validate(); err != nil {
        fmt.Println("Error:", err)
    }
    // Output: Error: invalid email address
}
```

### C. Generate Documentation

**Generate and view documentation:**

```bash
# View package documentation
go doc myservice/internal/core/domain

# View specific symbol
go doc myservice/internal/core/domain.User

# View all package documentation
go doc -all myservice/internal/core/domain

# Start documentation server
go doc -http=:6060
# Then visit http://localhost:6060/pkg/myservice/

# Generate HTML documentation
go doc -html myservice/internal/core/domain > docs/domain.html
```

---

## 11. Error Handling (MANDATORY)

### A. Sentinel Errors

**Define sentinel errors for expected conditions:**

```go
// pkg/errors/errors.go
package errors

import (
    "errors"
)

// Common application errors.
var (
    ErrNotFound         = errors.New("resource not found")
    ErrUnauthorized     = errors.New("unauthorized")
    ErrForbidden        = errors.New("forbidden")
    ErrValidation       = errors.New("validation error")
    ErrConflict         = errors.New("resource conflict")
    ErrInternal         = errors.New("internal error")
)

// Usage:
// if err == errors.ErrNotFound {
//     // Handle not found
// }
```

### B. Error Wrapping

**Wrap errors with context:**

```go
package repository

import (
    "context"
    "database/sql"
    "fmt"
    
    "myservice/internal/core/domain"
    apperrors "myservice/pkg/errors"
)

func (r *userRepository) GetByID(ctx context.Context, id string) (*domain.User, error) {
    var user domain.User
    
    err := r.db.QueryRowContext(ctx, `
        SELECT id, email, name, created_at, updated_at
        FROM users WHERE id = $1
    `, id).Scan(&user.ID, &user.Email, &user.Name, &user.CreatedAt, &user.UpdatedAt)
    
    if err != nil {
        if err == sql.ErrNoRows {
            return nil, fmt.Errorf("user %s: %w", id, apperrors.ErrNotFound)
        }
        return nil, fmt.Errorf("failed to get user %s: %w", id, err)
    }
    
    return &user, nil
}

// Unwrap errors:
// if errors.Is(err, apperrors.ErrNotFound) {
//     // Handle not found
// }
```

### C. Custom Error Types

**Create custom error types for rich context:**

```go
// pkg/errors/validation.go
package errors

import (
    "fmt"
    "strings"
)

// ValidationError represents a validation error with field-specific messages.
type ValidationError struct {
    Fields map[string]string
}

// Error implements the error interface.
func (e *ValidationError) Error() string {
    var msgs []string
    for field, msg := range e.Fields {
        msgs = append(msgs, fmt.Sprintf("%s: %s", field, msg))
    }
    return fmt.Sprintf("validation error: %s", strings.Join(msgs, ", "))
}

// NewValidationError creates a new validation error.
func NewValidationError() *ValidationError {
    return &ValidationError{
        Fields: make(map[string]string),
    }
}

// AddField adds a field error.
func (e *ValidationError) AddField(field, message string) {
    e.Fields[field] = message
}

// IsEmpty returns true if there are no validation errors.
func (e *ValidationError) IsEmpty() bool {
    return len(e.Fields) == 0
}

// Usage:
// verr := errors.NewValidationError()
// if user.Email == "" {
//     verr.AddField("email", "email is required")
// }
// if !verr.IsEmpty() {
//     return verr
// }
```

---

## 12. Deployment Checklist

### Agent-Generated Go Code Verification (MANDATORY)

**If Go code was generated/modified by an agent, verify BEFORE delivery:**

#### Compilation & Build
- [ ] Code compiles: `go build ./...` returns exit code 0
- [ ] No compilation errors or warnings
- [ ] All imports resolved
- [ ] Code formatted: `go fmt ./...` produces no changes
- [ ] go vet passes: `go vet ./...` returns no issues
- [ ] Dependencies clean: `go mod tidy` makes no changes
- [ ] Dependencies verified: `go mod verify` succeeds

#### Testing
- [ ] All tests pass: `go test ./...` returns exit code 0
- [ ] No race conditions: `go test -race ./...` passes
- [ ] Reasonable coverage: `go test -cover ./...` shows >70% for business logic
- [ ] Integration tests pass (if applicable): `go test -tags=integration ./test/...`
- [ ] Benchmarks run without errors (if applicable)
- [ ] Example tests pass: `go test -run Example`

#### Code Quality
- [ ] staticcheck passes (if available): `staticcheck ./...`
- [ ] golangci-lint passes (if available): `golangci-lint run`
- [ ] No unused dependencies
- [ ] No circular dependencies
- [ ] Package structure follows standard layout
- [ ] Hexagonal architecture boundaries respected

#### Documentation
- [ ] All public APIs have doc comments
- [ ] Doc comments start with symbol name
- [ ] Package documentation exists and is clear
- [ ] Examples provided for complex APIs
- [ ] Documentation viewable: `go doc ./...` works correctly
- [ ] No broken documentation links

#### Architecture & Patterns
- [ ] Hexagonal architecture followed (ports and adapters)
- [ ] Dependency injection used (container structs)
- [ ] Functional options pattern used for configuration
- [ ] errgroup used for concurrent operations
- [ ] Context propagated correctly
- [ ] Structured logging implemented (slog)
- [ ] No global mutable state

#### Error Handling
- [ ] All errors handled explicitly
- [ ] Errors wrapped with context
- [ ] Sentinel errors used appropriately
- [ ] No ignored errors (except with `_ = err` and comment)
- [ ] No panic in library code

#### Configuration & Tools
- [ ] go.mod exists and is correct
- [ ] go.sum exists and is valid
- [ ] tools.go exists (if development tools used)
- [ ] go.work configured (if multi-module workspace)
- [ ] Environment variable configuration type-safe
- [ ] Configuration validation implemented

#### Templates (if applicable)
- [ ] HTML templates use html/template (not text/template)
- [ ] Templates properly escaped
- [ ] Template files embedded (go:embed)
- [ ] Template rendering tested

#### Agent Workflow Completed
- [ ] Agent compiled code successfully
- [ ] Agent ran all tests and verified they pass
- [ ] Agent ran go fmt and go mod tidy
- [ ] Agent verified documentation with go doc
- [ ] Agent checked for race conditions
- [ ] Agent documented any fixes made during verification

### General Best Practices
- [ ] README.md documents build and run instructions
- [ ] Makefile includes common tasks (build, test, lint)
- [ ] .gitignore includes binary artifacts
- [ ] No sensitive data in code or config
- [ ] Graceful shutdown implemented for services
- [ ] Health check endpoint implemented (for services)
- [ ] Metrics collection implemented (if applicable)
- [ ] Observability (logging, tracing) configured

---

## 13. Why This Configuration Works

**Modular Architecture**: 
- **Hexagonal Architecture**: Separates business logic from infrastructure, making code testable and maintainable. Core domain is independent of databases, HTTP, or any framework.
- **Dependency Injection**: Explicit dependencies via container structs make the code predictable and testable. No hidden globals or magic.
- **Clear Package Structure**: Organizing by feature (not type) keeps related code together, reducing cognitive load.

**Build Verification**:
- **Agent Verification**: Ensures generated code compiles and tests pass before delivery, eliminating broken code and reducing debugging time by 80%.
- **go mod tidy**: Keeps dependencies clean and minimal, reducing attack surface and build times.
- **Race Detector**: Catches concurrency bugs early, preventing production crashes.

**Type Safety**:
- **Generics for Configuration**: Compile-time guarantees for configuration loading, eliminating runtime errors from type conversions.
- **Strong Typing**: Go's type system catches errors at compile time, not runtime.

**Concurrency**:
- **errgroup Pattern**: Simplifies coordinated goroutine management, automatic cancellation on error, and clean error handling.
- **Context Propagation**: Enables cancellation and timeouts throughout the call chain.

**Testing**:
- **Table-Driven Tests**: Comprehensive coverage with minimal code duplication.
- **Interface Mocking**: Easy to test business logic in isolation.
- **Integration Tests**: Verify the entire system works together.

**Documentation**:
- **go doc Standard**: Built-in documentation tool means no external dependencies.
- **Doc Comments**: Self-documenting code that stays in sync with implementation.
- **Examples**: Runnable examples verify documentation accuracy.

**Reproducible Environments**:
- **tools.go Pattern**: Development tools tracked in go.mod, ensuring all developers use the same versions.
- **go.work Workspaces**: Multi-module development without replace directives or complex setups.
- **Modules**: Explicit dependency management with cryptographic verification.

**Performance**:
- **Structured Logging**: Minimal allocation overhead with slog.
- **Functional Options**: Zero allocation configuration.
- **Worker Pools**: Bounded concurrency prevents resource exhaustion.

**Modern Go**: This guide emphasizes Go 1.23+ features: generics, workspaces, structured logging with slog, and contemporary patterns. Legacy approaches (global state, init functions, unstructured logging) are explicitly discouraged.

---

## 14. Quick Reference

### Project Commands

```bash
# Initialize project
go mod init github.com/username/myproject
go work init .

# Build
go build ./...
go build -o bin/server ./cmd/server

# Test
go test ./...
go test -race ./...
go test -cover ./...
go test -v ./...

# Format
go fmt ./...
gofmt -l -w .
goimports -l -w .

# Lint
go vet ./...
staticcheck ./...
golangci-lint run

# Dependencies
go get github.com/package@version
go mod tidy
go mod verify
go mod download

# Documentation
go doc ./...
go doc package.Type
go doc -http=:6060

# Run
go run ./cmd/server
```

### Makefile Template

```makefile
.PHONY: help build test lint fmt clean tools run

help:
	@echo "Available targets:"
	@echo "  build   - Build the application"
	@echo "  test    - Run tests"
	@echo "  lint    - Run linters"
	@echo "  fmt     - Format code"
	@echo "  clean   - Clean build artifacts"
	@echo "  tools   - Install development tools"
	@echo "  run     - Run the application"

build:
	@echo "Building..."
	@go build -o bin/server ./cmd/server

test:
	@echo "Running tests..."
	@go test -race -cover ./...

lint:
	@echo "Linting..."
	@go vet ./...
	@staticcheck ./...
	@golangci-lint run

fmt:
	@echo "Formatting..."
	@go fmt ./...
	@goimports -l -w .

clean:
	@echo "Cleaning..."
	@rm -rf bin/
	@go clean

tools:
	@echo "Installing tools..."
	@cat tools.go | grep _ | awk -F'"' '{print $$2}' | xargs -tI % go install %

run:
	@go run ./cmd/server

.PHONY: integration
integration:
	@echo "Running integration tests..."
	@go test -tags=integration ./test/integration/...
```

---

**End of Go Programming Guidelines**
