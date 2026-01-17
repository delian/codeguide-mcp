# Zig Development Guidelines
This document provides mandatory coding standards and development practices for modern Zig applications

---
Agent Profile: The Zig Expert
Role: Senior Zig Engineer & Systems Programming Specialist
Objective: Generate production-ready, safe, performant, well-documented, and maintainable Zig code.
Tools: Zig 0.12+, zig build, zig test, zig fmt, comptime, std library.

## 1. Core Philosophies

The agent must adhere to the "ZIG-FIRST" principles for every Zig project:

**Zero Hidden Control Flow**: No hidden allocations, explicit everything, visible side effects.
**Intentional Memory Management**: Allocators passed explicitly, defer for cleanup, arena allocators.
**Generics via Comptime**: Use comptime for generic programming, type introspection, compile-time execution.

**Fast & Efficient**: Data-oriented design, cache-friendly layouts, comptime optimizations.
**Immutable Preferred**: Prefer const, immutable data structures, functional patterns.
**Result Types**: Error unions (!Type), explicit error handling, no exceptions.
**Safe & Explicit**: No undefined behavior, bounds checking, explicit casts.
**Testable Code**: Built-in testing, test blocks in source, modular design.

**Hexagonal Architecture**: Domain core, ports, adapters, clear boundaries.
**Explicit Allocators**: Pass allocators, no global state, defer cleanup immediately.
**CQRS Pattern**: Separate commands and queries, clear data flow.
**Uniform Design**: Consistent init/deinit patterns, method syntax, struct composition.
**Reusable Modules**: Clean module boundaries, public interfaces, encapsulation.

**Comptime Everything**: Leverage comptime for polymorphism, generics, code generation.
**Defer for Safety**: Pair allocations with defer, resource cleanup guaranteed.
**Data-Oriented**: Struct of arrays, cache-friendly, ECS when appropriate.

**Verified Always**: All code must compile with `zig build`, pass tests with `zig test`.
**Async-Aware**: Use async/await when applicable, event loop patterns.
**Documented Code**: Doc comments for all public APIs, generated documentation.

## 2. Agent Code Generation Requirements (MANDATORY)

### A. Verification Protocol

**CRITICAL: Agents MUST verify that all generated Zig code compiles and passes tests before presenting it to the user.**

#### Pre-Delivery Checklist

**Before delivering ANY Zig code, the agent MUST:**

1. **Compilation Check**:
   ```bash
   # Verify code compiles without errors
   zig build
   # Exit code MUST be 0
   
   # Check specific module
   zig build-lib src/module.zig
   
   # Compile with optimizations
   zig build -Doptimize=ReleaseFast
   ```

2. **Formatting Check**:
   ```bash
   # Verify code is formatted
   zig fmt --check src/
   # Exit code MUST be 0
   
   # Auto-format code
   zig fmt src/
   ```

3. **Test Execution**:
   ```bash
   # Run all tests
   zig test src/main.zig
   # Exit code MUST be 0, all tests pass
   
   # Run tests with coverage
   zig build test
   
   # Run specific test
   zig test src/domain/user.zig
   ```

4. **Build System Check**:
   ```bash
   # Verify build.zig works
   zig build
   
   # Run with different modes
   zig build -Doptimize=Debug
   zig build -Doptimize=ReleaseSafe
   zig build -Doptimize=ReleaseFast
   ```

5. **Documentation Check**:
   ```bash
   # Generate documentation
   zig build docs
   
   # Or manually
   zig test -femit-docs src/main.zig
   ```

6. **Memory Leak Check** (using allocator debugging):
   ```zig
   // In tests, use testing allocator
   test "no memory leaks" {
       const allocator = std.testing.allocator;
       // ... code that allocates
       // If leaks exist, test will fail
   }
   ```

### B. Error Correction Process

If verification fails:

1. **Read the compiler error** (Zig errors are detailed and helpful)
2. **Identify the root cause** (type error, memory error, comptime error, etc.)
3. **Fix the issue** following Zig idioms
4. **Re-run verification** until all checks pass
5. **Document allocator usage** with comments

### C. Agent Workflow Example

**Complete workflow for generating a function:**

1. **Generate code with documentation**:
   ```zig
   const std = @import("std");
   
   /// Parses a user ID from a string.
   ///
   /// Returns an error if the input is empty or contains invalid characters.
   ///
   /// # Arguments
   /// * `allocator` - Memory allocator for string allocation
   /// * `input` - The input string to parse
   ///
   /// # Returns
   /// Returns `UserId` on success, or `ParseError` on failure.
   ///
   /// # Example
   /// ```zig
   /// const allocator = std.heap.page_allocator;
   /// const user_id = try parseUserId(allocator, "user-123");
   /// defer user_id.deinit();
   /// ```
   pub fn parseUserId(allocator: std.mem.Allocator, input: []const u8) !UserId {
       if (input.len == 0) {
           return error.EmptyInput;
       }
       
       const id_copy = try allocator.dupe(u8, input);
       errdefer allocator.free(id_copy);
       
       return UserId{ .id = id_copy, .allocator = allocator };
   }
   
   test "parseUserId valid input" {
       const allocator = std.testing.allocator;
       
       const user_id = try parseUserId(allocator, "user-123");
       defer user_id.deinit();
       
       try std.testing.expectEqualStrings("user-123", user_id.id);
   }
   
   test "parseUserId empty input" {
       const allocator = std.testing.allocator;
       
       const result = parseUserId(allocator, "");
       try std.testing.expectError(error.EmptyInput, result);
   }
   ```

2. **Verify compilation**:
   ```bash
   zig build-lib user.zig
   # ✓ Compiled successfully
   ```

3. **Run tests**:
   ```bash
   zig test user.zig
   # ✓ All 2 tests passed
   ```

4. **Format code**:
   ```bash
   zig fmt user.zig
   # ✓ Formatted
   ```

5. **Present code** to user - only after ALL checks pass

### D. Prohibited Practices

**NEVER deliver Zig code that:**
- ❌ Has compilation errors
- ❌ Has memory leaks (test with `std.testing.allocator`)
- ❌ Uses hidden allocations (global allocators)
- ❌ Lacks defer for allocated resources
- ❌ Fails tests
- ❌ Lacks tests for new functionality
- ❌ Lacks doc comments for public APIs
- ❌ Uses undefined behavior
- ❌ Ignores error return values
- ❌ Has poor naming (non-idiomatic Zig)
- ❌ Doesn't use comptime when appropriate
- ❌ Uses `unreachable` without justification

---

## 3. Project Structure (Hexagonal Architecture)

### A. Directory Layout

```
my-app/
├── build.zig              # Build configuration
├── build.zig.zon          # Dependency management
├── README.md
├── .gitignore
├── src/
│   ├── main.zig           # Binary entry point
│   ├── lib.zig            # Library entry point
│   │
│   ├── domain/            # Domain layer (core business logic)
│   │   ├── entities/
│   │   │   ├── user.zig
│   │   │   └── order.zig
│   │   ├── value_objects/
│   │   │   ├── user_id.zig
│   │   │   └── email.zig
│   │   ├── services/
│   │   │   └── order_service.zig
│   │   └── errors.zig
│   │
│   ├── application/       # Application layer (use cases)
│   │   ├── commands/      # Commands (write - CQRS)
│   │   │   ├── create_user.zig
│   │   │   └── place_order.zig
│   │   ├── queries/       # Queries (read - CQRS)
│   │   │   ├── get_user.zig
│   │   │   └── list_orders.zig
│   │   └── ports/         # Ports (interfaces)
│   │       ├── user_repository.zig
│   │       └── email_service.zig
│   │
│   ├── infrastructure/    # Infrastructure layer (adapters)
│   │   ├── persistence/
│   │   │   ├── memory_user_repo.zig
│   │   │   └── file_user_repo.zig
│   │   ├── http/
│   │   │   ├── routes.zig
│   │   │   └── handlers.zig
│   │   └── email/
│   │       └── console_email.zig
│   │
│   └── config/
│       └── settings.zig
│
└── tests/                 # Integration tests
    └── user_workflow_test.zig
```

### B. build.zig Structure

```zig
const std = @import("std");

pub fn build(b: *std.Build) void {
    // Standard target and optimization options
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Library
    const lib = b.addStaticLibrary(.{
        .name = "myapp",
        .root_source_file = .{ .path = "src/lib.zig" },
        .target = target,
        .optimize = optimize,
    });
    b.installArtifact(lib);

    // Executable
    const exe = b.addExecutable(.{
        .name = "myapp",
        .root_source_file = .{ .path = "src/main.zig" },
        .target = target,
        .optimize = optimize,
    });
    b.installArtifact(exe);

    // Run command
    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }
    
    const run_step = b.step("run", "Run the app");
    run_step.dependOn(&run_cmd.step);

    // Tests
    const lib_tests = b.addTest(.{
        .root_source_file = .{ .path = "src/lib.zig" },
        .target = target,
        .optimize = optimize,
    });
    
    const run_lib_tests = b.addRunArtifact(lib_tests);
    
    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_lib_tests.step);

    // Documentation
    const docs = lib_tests;
    docs.emit_docs = .emit;
    
    const docs_step = b.step("docs", "Generate documentation");
    docs_step.dependOn(&docs.step);
}
```

### C. build.zig.zon (Dependencies)

```zig
.{
    .name = "myapp",
    .version = "0.1.0",
    .dependencies = .{
        // Pure Zig dependencies only
        .@"zig-string" = .{
            .url = "https://github.com/JakubSzark/zig-string/archive/main.tar.gz",
            .hash = "...",
        },
    },
    .paths = .{
        "build.zig",
        "build.zig.zon",
        "src",
    },
}
```

---

## 4. Explicit Memory Management (MANDATORY)

### A. Allocator Passing

```zig
const std = @import("std");

/// User entity with explicit allocator
pub const User = struct {
    id: []const u8,
    name: []const u8,
    email: []const u8,
    allocator: std.mem.Allocator,

    /// Creates a new user
    ///
    /// Caller owns the returned User and must call deinit().
    pub fn init(
        allocator: std.mem.Allocator,
        id: []const u8,
        name: []const u8,
        email: []const u8,
    ) !User {
        // Allocate copies of strings
        const id_copy = try allocator.dupe(u8, id);
        errdefer allocator.free(id_copy);
        
        const name_copy = try allocator.dupe(u8, name);
        errdefer allocator.free(name_copy);
        
        const email_copy = try allocator.dupe(u8, email);
        errdefer allocator.free(email_copy);

        return User{
            .id = id_copy,
            .name = name_copy,
            .email = email_copy,
            .allocator = allocator,
        };
    }

    /// Frees all allocated memory
    pub fn deinit(self: *User) void {
        self.allocator.free(self.id);
        self.allocator.free(self.name);
        self.allocator.free(self.email);
    }
};

test "User init and deinit" {
    const allocator = std.testing.allocator;
    
    var user = try User.init(
        allocator,
        "user-123",
        "John Doe",
        "john@example.com",
    );
    defer user.deinit();
    
    try std.testing.expectEqualStrings("user-123", user.id);
    try std.testing.expectEqualStrings("John Doe", user.name);
}
```

### B. Defer Pattern

```zig
/// Loads configuration from file
pub fn loadConfig(allocator: std.mem.Allocator, path: []const u8) !Config {
    // Open file
    const file = try std.fs.cwd().openFile(path, .{});
    defer file.close(); // Always closes, even on error
    
    // Read contents
    const contents = try file.readToEndAlloc(allocator, 1024 * 1024);
    defer allocator.free(contents); // Always frees
    
    // Parse
    return try parseConfig(allocator, contents);
}

/// Processes multiple users with proper cleanup
pub fn processUsers(
    allocator: std.mem.Allocator,
    user_ids: []const []const u8,
) !void {
    var users = std.ArrayList(User).init(allocator);
    defer users.deinit();
    defer {
        // Clean up all users
        for (users.items) |*user| {
            user.deinit();
        }
    }
    
    for (user_ids) |id| {
        const user = try fetchUser(allocator, id);
        try users.append(user);
    }
    
    // Process users...
}
```

### C. Arena Allocator Pattern

```zig
/// Request handler using arena allocator
pub fn handleRequest(
    parent_allocator: std.mem.Allocator,
    request: Request,
) !Response {
    // Create arena for request lifetime
    var arena = std.heap.ArenaAllocator.init(parent_allocator);
    defer arena.deinit(); // Frees all allocations at once
    
    const allocator = arena.allocator();
    
    // All allocations use arena
    const user = try fetchUser(allocator, request.user_id);
    const orders = try fetchOrders(allocator, user.id);
    const processed = try processOrders(allocator, orders);
    
    // No need for individual defers
    // All memory freed when arena.deinit() is called
    
    return Response{ .data = processed };
}

/// Game frame using arena allocator
pub fn gameFrame(parent_allocator: std.mem.Allocator) !void {
    var arena = std.heap.ArenaAllocator.init(parent_allocator);
    defer arena.deinit(); // Clear all per-frame allocations
    
    const allocator = arena.allocator();
    
    // Frame-specific allocations
    const entities = try loadEntities(allocator);
    const events = try processInput(allocator);
    try updateGameState(allocator, entities, events);
    try render(allocator, entities);
    
    // All memory automatically freed at end of frame
}
```

---

## 5. Comptime Programming (MANDATORY)

### A. Generic Functions

```zig
const std = @import("std");

/// Generic find function using comptime
pub fn find(comptime T: type, items: []const T, target: T) ?usize {
    for (items, 0..) |item, i| {
        if (std.meta.eql(item, target)) {
            return i;
        }
    }
    return null;
}

test "find generic" {
    const numbers = [_]i32{ 1, 2, 3, 4, 5 };
    const index = find(i32, &numbers, 3);
    try std.testing.expectEqual(@as(?usize, 2), index);
    
    const strings = [_][]const u8{ "a", "b", "c" };
    const str_index = find([]const u8, &strings, "b");
    try std.testing.expectEqual(@as(?usize, 1), str_index);
}

/// Generic repository interface using comptime
pub fn Repository(comptime Entity: type, comptime Id: type) type {
    return struct {
        const Self = @This();
        
        allocator: std.mem.Allocator,
        vtable: *const VTable,
        
        pub const VTable = struct {
            findById: *const fn (ctx: *anyopaque, id: Id) anyerror!?Entity,
            save: *const fn (ctx: *anyopaque, entity: Entity) anyerror!void,
            delete: *const fn (ctx: *anyopaque, id: Id) anyerror!void,
        };
        
        pub fn findById(self: Self, id: Id) !?Entity {
            return self.vtable.findById(self.ptr, id);
        }
        
        pub fn save(self: Self, entity: Entity) !void {
            return self.vtable.save(self.ptr, entity);
        }
        
        pub fn delete(self: Self, id: Id) !void {
            return self.vtable.delete(self.ptr, id);
        }
        
        ptr: *anyopaque,
    };
}

// Usage
const UserRepository = Repository(User, UserId);
const OrderRepository = Repository(Order, OrderId);
```

### B. Compile-Time Type Introspection

```zig
/// Prints all fields of a struct at compile time
pub fn printStructFields(comptime T: type) void {
    comptime {
        const info = @typeInfo(T);
        if (info != .Struct) {
            @compileError("Expected struct type");
        }
        
        inline for (info.Struct.fields) |field| {
            @compileLog("Field: ", field.name, " Type: ", field.type);
        }
    }
}

/// Validates struct has required fields at compile time
pub fn validateStruct(comptime T: type, comptime required_fields: []const []const u8) void {
    comptime {
        const info = @typeInfo(T);
        if (info != .Struct) {
            @compileError("Expected struct type");
        }
        
        for (required_fields) |required| {
            var found = false;
            for (info.Struct.fields) |field| {
                if (std.mem.eql(u8, field.name, required)) {
                    found = true;
                    break;
                }
            }
            if (!found) {
                @compileError("Missing required field: " ++ required);
            }
        }
    }
}

/// Generic serializer using comptime
pub fn serialize(
    comptime T: type,
    value: T,
    writer: anytype,
) !void {
    const info = @typeInfo(T);
    switch (info) {
        .Int => try writer.print("{}", .{value}),
        .Float => try writer.print("{d:.2}", .{value}),
        .Bool => try writer.writeAll(if (value) "true" else "false"),
        .Struct => |s| {
            try writer.writeByte('{');
            inline for (s.fields, 0..) |field, i| {
                if (i > 0) try writer.writeByte(',');
                try writer.print("\"{s}\":", .{field.name});
                try serialize(field.type, @field(value, field.name), writer);
            }
            try writer.writeByte('}');
        },
        else => @compileError("Unsupported type for serialization"),
    }
}
```

### C. Comptime Code Generation

```zig
/// Generates enum from string array at compile time
pub fn makeEnum(comptime names: []const []const u8) type {
    comptime {
        var fields: [names.len]std.builtin.Type.EnumField = undefined;
        for (names, 0..) |name, i| {
            fields[i] = .{
                .name = name,
                .value = i,
            };
        }
        
        return @Type(.{
            .Enum = .{
                .tag_type = usize,
                .fields = &fields,
                .decls = &.{},
                .is_exhaustive = true,
            },
        });
    }
}

const Status = makeEnum(&.{ "pending", "confirmed", "shipped", "delivered" });

test "generated enum" {
    const status = Status.confirmed;
    try std.testing.expectEqual(Status.confirmed, status);
}
```

---

## 6. Error Handling (MANDATORY)

### A. Error Union Types

```zig
const std = @import("std");

/// Domain errors
pub const UserError = error{
    NotFound,
    AlreadyExists,
    InvalidEmail,
    EmptyName,
};

/// Parse errors
pub const ParseError = error{
    EmptyInput,
    InvalidFormat,
} || std.mem.Allocator.Error;

/// Validates email format
pub fn validateEmail(email: []const u8) UserError!void {
    if (email.len == 0) {
        return error.InvalidEmail;
    }
    
    var has_at = false;
    var has_dot = false;
    
    for (email) |c| {
        if (c == '@') has_at = true;
        if (c == '.' and has_at) has_dot = true;
    }
    
    if (!has_at or !has_dot) {
        return error.InvalidEmail;
    }
}

/// Creates user with validation
pub fn createUser(
    allocator: std.mem.Allocator,
    name: []const u8,
    email: []const u8,
) !User {
    if (name.len == 0) {
        return error.EmptyName;
    }
    
    try validateEmail(email);
    
    return User.init(allocator, generateId(), name, email);
}

test "validateEmail success" {
    try validateEmail("user@example.com");
}

test "validateEmail failure" {
    const result = validateEmail("invalid");
    try std.testing.expectError(error.InvalidEmail, result);
}
```

### B. Error Propagation

```zig
/// Fetches and processes user
pub fn fetchAndProcessUser(
    allocator: std.mem.Allocator,
    user_id: []const u8,
) !ProcessedUser {
    // Errors automatically propagate with try
    const user = try fetchUser(allocator, user_id);
    defer user.deinit();
    
    const validated = try validateUser(user);
    const enriched = try enrichUserData(allocator, validated);
    
    return enriched;
}

/// Catch and handle specific errors
pub fn getUserOrDefault(
    allocator: std.mem.Allocator,
    user_id: []const u8,
) !User {
    return fetchUser(allocator, user_id) catch |err| switch (err) {
        error.NotFound => return createDefaultUser(allocator),
        error.InvalidId => return error.InvalidId,
        else => return err,
    };
}

/// errdefer for cleanup on error
pub fn loadUsers(
    allocator: std.mem.Allocator,
    count: usize,
) ![]User {
    const users = try allocator.alloc(User, count);
    errdefer allocator.free(users); // Only freed if error occurs
    
    for (users, 0..) |*user, i| {
        user.* = try loadUser(allocator, i);
        errdefer user.deinit(); // Clean up this user if subsequent operations fail
    }
    
    return users;
}
```

### C. Result Patterns

```zig
/// Result type pattern
pub fn Result(comptime T: type) type {
    return union(enum) {
        ok: T,
        err: anyerror,
        
        pub fn isOk(self: @This()) bool {
            return self == .ok;
        }
        
        pub fn isErr(self: @This()) bool {
            return self == .err;
        }
        
        pub fn unwrap(self: @This()) T {
            return switch (self) {
                .ok => |value| value,
                .err => unreachable,
            };
        }
        
        pub fn unwrapOr(self: @This(), default: T) T {
            return switch (self) {
                .ok => |value| value,
                .err => default,
            };
        }
    };
}

// Usage
pub fn safeDiv(a: i32, b: i32) Result(i32) {
    if (b == 0) {
        return .{ .err = error.DivisionByZero };
    }
    return .{ .ok = @divTrunc(a, b) };
}
```

---

## 7. Data-Oriented Design (MANDATORY)

### A. Struct of Arrays (SoA)

```zig
const std = @import("std");

/// Bad: Array of Structs (AoS) - Poor cache performance
pub const EntitiesAoS = struct {
    entities: []Entity,
    
    pub const Entity = struct {
        id: u32,
        x: f32,
        y: f32,
        z: f32,
        health: f32,
        // ... more fields
    };
    
    pub fn updatePositions(self: *EntitiesAoS, dt: f32) void {
        for (self.entities) |*entity| {
            entity.x += dt; // Poor cache locality
            entity.y += dt;
            entity.z += dt;
        }
    }
};

/// Good: Struct of Arrays (SoA) - Excellent cache performance
pub const EntitiesSoA = struct {
    ids: []u32,
    x_positions: []f32,
    y_positions: []f32,
    z_positions: []f32,
    healths: []f32,
    count: usize,
    allocator: std.mem.Allocator,
    
    pub fn init(allocator: std.mem.Allocator, capacity: usize) !EntitiesSoA {
        return EntitiesSoA{
            .ids = try allocator.alloc(u32, capacity),
            .x_positions = try allocator.alloc(f32, capacity),
            .y_positions = try allocator.alloc(f32, capacity),
            .z_positions = try allocator.alloc(f32, capacity),
            .healths = try allocator.alloc(f32, capacity),
            .count = 0,
            .allocator = allocator,
        };
    }
    
    pub fn deinit(self: *EntitiesSoA) void {
        self.allocator.free(self.ids);
        self.allocator.free(self.x_positions);
        self.allocator.free(self.y_positions);
        self.allocator.free(self.z_positions);
        self.allocator.free(self.healths);
    }
    
    /// Update positions with excellent cache locality
    pub fn updatePositions(self: *EntitiesSoA, dt: f32) void {
        for (0..self.count) |i| {
            self.x_positions[i] += dt; // Sequential access, cache-friendly
            self.y_positions[i] += dt;
            self.z_positions[i] += dt;
        }
    }
    
    /// SIMD-friendly processing
    pub fn updatePositionsSIMD(self: *EntitiesSoA, dt: f32) void {
        // Zig can auto-vectorize this loop
        for (0..self.count) |i| {
            self.x_positions[i] += dt;
        }
        for (0..self.count) |i| {
            self.y_positions[i] += dt;
        }
        for (0..self.count) |i| {
            self.z_positions[i] += dt;
        }
    }
};
```

### B. Entity Component System (ECS)

```zig
/// Simple ECS implementation
pub const ECS = struct {
    positions: std.ArrayList(Position),
    velocities: std.ArrayList(Velocity),
    healths: std.ArrayList(Health),
    entity_count: usize,
    allocator: std.mem.Allocator,
    
    pub const Position = struct { x: f32, y: f32, z: f32 };
    pub const Velocity = struct { dx: f32, dy: f32, dz: f32 };
    pub const Health = struct { current: f32, max: f32 };
    
    pub fn init(allocator: std.mem.Allocator) ECS {
        return ECS{
            .positions = std.ArrayList(Position).init(allocator),
            .velocities = std.ArrayList(Velocity).init(allocator),
            .healths = std.ArrayList(Health).init(allocator),
            .entity_count = 0,
            .allocator = allocator,
        };
    }
    
    pub fn deinit(self: *ECS) void {
        self.positions.deinit();
        self.velocities.deinit();
        self.healths.deinit();
    }
    
    /// System: Update positions based on velocities
    pub fn updateMovement(self: *ECS, dt: f32) void {
        for (0..self.entity_count) |i| {
            if (i >= self.positions.items.len) break;
            if (i >= self.velocities.items.len) break;
            
            self.positions.items[i].x += self.velocities.items[i].dx * dt;
            self.positions.items[i].y += self.velocities.items[i].dy * dt;
            self.positions.items[i].z += self.velocities.items[i].dz * dt;
        }
    }
    
    /// System: Regenerate health
    pub fn regenerateHealth(self: *ECS, rate: f32) void {
        for (self.healths.items) |*health| {
            health.current = @min(health.current + rate, health.max);
        }
    }
};
```

---

## 8. CQRS Pattern (MANDATORY)

### A. Command Side (Write Operations)

```zig
const std = @import("std");

/// Command: Create user
pub const CreateUserCommand = struct {
    name: []const u8,
    email: []const u8,
};

/// Command result
pub const CreateUserResult = struct {
    user_id: []const u8,
    allocator: std.mem.Allocator,
    
    pub fn deinit(self: *CreateUserResult) void {
        self.allocator.free(self.user_id);
    }
};

/// Command handler
pub const CreateUserHandler = struct {
    repository: *UserRepository,
    allocator: std.mem.Allocator,
    
    pub fn init(allocator: std.mem.Allocator, repository: *UserRepository) CreateUserHandler {
        return .{
            .repository = repository,
            .allocator = allocator,
        };
    }
    
    /// Handles create user command
    pub fn handle(
        self: *CreateUserHandler,
        command: CreateUserCommand,
    ) !CreateUserResult {
        // Validate
        try validateEmail(command.email);
        if (command.name.len == 0) {
            return error.EmptyName;
        }
        
        // Check if user exists
        const existing = try self.repository.findByEmail(command.email);
        if (existing != null) {
            return error.AlreadyExists;
        }
        
        // Create user
        const user_id = try generateUserId(self.allocator);
        errdefer self.allocator.free(user_id);
        
        var user = try User.init(
            self.allocator,
            user_id,
            command.name,
            command.email,
        );
        defer user.deinit();
        
        // Save
        try self.repository.save(user);
        
        return CreateUserResult{
            .user_id = try self.allocator.dupe(u8, user_id),
            .allocator = self.allocator,
        };
    }
};

test "CreateUserHandler success" {
    const allocator = std.testing.allocator;
    
    var repo = InMemoryUserRepository.init(allocator);
    defer repo.deinit();
    
    var handler = CreateUserHandler.init(allocator, &repo);
    
    const command = CreateUserCommand{
        .name = "John Doe",
        .email = "john@example.com",
    };
    
    var result = try handler.handle(command);
    defer result.deinit();
    
    try std.testing.expect(result.user_id.len > 0);
}
```

### B. Query Side (Read Operations)

```zig
/// Query: Get user by ID
pub const GetUserQuery = struct {
    user_id: []const u8,
};

/// Query result
pub const GetUserResult = struct {
    user: User,
    
    pub fn deinit(self: *GetUserResult) void {
        self.user.deinit();
    }
};

/// Query handler
pub const GetUserHandler = struct {
    repository: *UserRepository,
    allocator: std.mem.Allocator,
    
    pub fn init(allocator: std.mem.Allocator, repository: *UserRepository) GetUserHandler {
        return .{
            .repository = repository,
            .allocator = allocator,
        };
    }
    
    pub fn handle(
        self: *GetUserHandler,
        query: GetUserQuery,
    ) !GetUserResult {
        const user = try self.repository.findById(query.user_id) orelse {
            return error.NotFound;
        };
        
        return GetUserResult{ .user = user };
    }
};

/// Query: List users with pagination
pub const ListUsersQuery = struct {
    limit: usize,
    offset: usize,
};

pub const ListUsersResult = struct {
    users: []User,
    allocator: std.mem.Allocator,
    
    pub fn deinit(self: *ListUsersResult) void {
        for (self.users) |*user| {
            user.deinit();
        }
        self.allocator.free(self.users);
    }
};

pub const ListUsersHandler = struct {
    repository: *UserRepository,
    allocator: std.mem.Allocator,
    
    pub fn init(allocator: std.mem.Allocator, repository: *UserRepository) ListUsersHandler {
        return .{
            .repository = repository,
            .allocator = allocator,
        };
    }
    
    pub fn handle(
        self: *ListUsersHandler,
        query: ListUsersQuery,
    ) !ListUsersResult {
        const users = try self.repository.list(query.limit, query.offset);
        
        return ListUsersResult{
            .users = users,
            .allocator = self.allocator,
        };
    }
};
```

---

## 9. Functional Programming Patterns (MANDATORY)

### A. Immutability and Iterators

```zig
const std = @import("std");

/// Filters users by activity status (functional style)
pub fn filterActive(
    allocator: std.mem.Allocator,
    users: []const User,
) ![]User {
    var result = std.ArrayList(User).init(allocator);
    errdefer result.deinit();
    
    for (users) |user| {
        if (user.is_active) {
            try result.append(try user.clone(allocator));
        }
    }
    
    return result.toOwnedSlice();
}

/// Maps users to their names
pub fn extractNames(
    allocator: std.mem.Allocator,
    users: []const User,
) ![][]const u8 {
    var names = try allocator.alloc([]const u8, users.len);
    errdefer allocator.free(names);
    
    for (users, 0..) |user, i| {
        names[i] = try allocator.dupe(u8, user.name);
    }
    
    return names;
}

/// Reduces array to single value (fold)
pub fn sumAges(users: []const User) u32 {
    var total: u32 = 0;
    for (users) |user| {
        total += user.age;
    }
    return total;
}

/// Generic filter function
pub fn filter(
    comptime T: type,
    allocator: std.mem.Allocator,
    items: []const T,
    predicate: fn (T) bool,
) ![]T {
    var result = std.ArrayList(T).init(allocator);
    errdefer result.deinit();
    
    for (items) |item| {
        if (predicate(item)) {
            try result.append(item);
        }
    }
    
    return result.toOwnedSlice();
}

/// Generic map function
pub fn map(
    comptime T: type,
    comptime U: type,
    allocator: std.mem.Allocator,
    items: []const T,
    mapper: fn (std.mem.Allocator, T) anyerror!U,
) ![]U {
    var result = try allocator.alloc(U, items.len);
    errdefer allocator.free(result);
    
    for (items, 0..) |item, i| {
        result[i] = try mapper(allocator, item);
    }
    
    return result;
}

/// Generic reduce function
pub fn reduce(
    comptime T: type,
    comptime U: type,
    items: []const T,
    initial: U,
    reducer: fn (U, T) U,
) U {
    var accumulator = initial;
    for (items) |item| {
        accumulator = reducer(accumulator, item);
    }
    return accumulator;
}

test "functional patterns" {
    const allocator = std.testing.allocator;
    
    const numbers = [_]i32{ 1, 2, 3, 4, 5 };
    
    // Filter even numbers
    const isEven = struct {
        fn call(n: i32) bool {
            return @mod(n, 2) == 0;
        }
    }.call;
    
    const evens = try filter(i32, allocator, &numbers, isEven);
    defer allocator.free(evens);
    
    try std.testing.expectEqual(@as(usize, 2), evens.len);
    
    // Reduce (sum)
    const sum = reduce(i32, i32, &numbers, 0, struct {
        fn call(acc: i32, n: i32) i32 {
            return acc + n;
        }
    }.call);
    
    try std.testing.expectEqual(@as(i32, 15), sum);
}
```

### B. Higher-Order Functions

```zig
/// Function composition
pub fn compose(
    comptime A: type,
    comptime B: type,
    comptime C: type,
    f: fn (B) C,
    g: fn (A) B,
) fn (A) C {
    return struct {
        fn composed(x: A) C {
            return f(g(x));
        }
    }.composed;
}

/// Currying pattern
pub fn curry(
    comptime A: type,
    comptime B: type,
    comptime C: type,
) type {
    return struct {
        pub fn apply(f: fn (A, B) C) fn (A) fn (B) C {
            return struct {
                fn curried(a: A) fn (B) C {
                    return struct {
                        fn inner(b: B) C {
                            return f(a, b);
                        }
                    }.inner;
                }
            }.curried;
        }
    };
}

// Usage
fn add(a: i32, b: i32) i32 {
    return a + b;
}

test "currying" {
    const CurriedAdd = curry(i32, i32, i32).apply(add);
    const add5 = CurriedAdd(5);
    
    try std.testing.expectEqual(@as(i32, 15), add5(10));
}
```

---

## 10. Async/Await Patterns

```zig
const std = @import("std");

/// Async function example
pub fn fetchUserAsync(user_id: []const u8) !User {
    // Simulate async operation
    suspend {
        // Yield control
        resume @frame();
    }
    
    // Fetch user
    return try fetchUserFromDb(user_id);
}

/// Event loop pattern
pub const EventLoop = struct {
    frames: std.ArrayList(@Frame(handleRequest)),
    allocator: std.mem.Allocator,
    
    pub fn init(allocator: std.mem.Allocator) EventLoop {
        return .{
            .frames = std.ArrayList(@Frame(handleRequest)).init(allocator),
            .allocator = allocator,
        };
    }
    
    pub fn deinit(self: *EventLoop) void {
        self.frames.deinit();
    }
    
    pub fn schedule(self: *EventLoop, request: Request) !void {
        const frame = try self.allocator.create(@Frame(handleRequest));
        frame.* = async handleRequest(request);
        try self.frames.append(frame.*);
    }
    
    pub fn run(self: *EventLoop) !void {
        while (self.frames.items.len > 0) {
            for (self.frames.items) |*frame| {
                if (await frame) {
                    // Frame completed
                }
            }
        }
    }
};

fn handleRequest(request: Request) !Response {
    suspend {
        // Wait for I/O
        resume @frame();
    }
    
    return Response{ .status = 200 };
}
```

---

## 11. Testing (MANDATORY)

### A. Unit Tests

```zig
const std = @import("std");

test "User creation" {
    const allocator = std.testing.allocator;
    
    var user = try User.init(
        allocator,
        "user-123",
        "John Doe",
        "john@example.com",
    );
    defer user.deinit();
    
    try std.testing.expectEqualStrings("user-123", user.id);
    try std.testing.expectEqualStrings("John Doe", user.name);
}

test "Email validation success" {
    try validateEmail("valid@example.com");
}

test "Email validation failure" {
    try std.testing.expectError(error.InvalidEmail, validateEmail("invalid"));
}

test "Memory leak detection" {
    const allocator = std.testing.allocator;
    
    // This test will fail if there are memory leaks
    var user = try User.init(allocator, "1", "Test", "test@test.com");
    defer user.deinit(); // If forgotten, test fails
    
    _ = user;
}

test "Array bounds checking" {
    const numbers = [_]i32{ 1, 2, 3 };
    
    // This would be caught at runtime in Debug mode
    // try std.testing.expectEqual(@as(i32, 0), numbers[10]);
    
    try std.testing.expectEqual(@as(i32, 2), numbers[1]);
}
```

### B. Integration Tests

```zig
// tests/user_workflow_test.zig

const std = @import("std");
const myapp = @import("myapp");

test "Full user workflow" {
    const allocator = std.testing.allocator;
    
    // Setup
    var repo = myapp.InMemoryUserRepository.init(allocator);
    defer repo.deinit();
    
    var create_handler = myapp.CreateUserHandler.init(allocator, &repo);
    var get_handler = myapp.GetUserHandler.init(allocator, &repo);
    
    // Create user
    const create_cmd = myapp.CreateUserCommand{
        .name = "Integration Test",
        .email = "integration@test.com",
    };
    
    var create_result = try create_handler.handle(create_cmd);
    defer create_result.deinit();
    
    // Retrieve user
    const get_query = myapp.GetUserQuery{
        .user_id = create_result.user_id,
    };
    
    var get_result = try get_handler.handle(get_query);
    defer get_result.deinit();
    
    // Verify
    try std.testing.expectEqualStrings("Integration Test", get_result.user.name);
}
```

### C. Benchmark Tests

```zig
test "Benchmark: SoA vs AoS" {
    const allocator = std.testing.allocator;
    const count = 100000;
    
    // Benchmark SoA
    var soa = try EntitiesSoA.init(allocator, count);
    defer soa.deinit();
    
    const start_soa = std.time.nanoTimestamp();
    for (0..1000) |_| {
        soa.updatePositions(0.016);
    }
    const end_soa = std.time.nanoTimestamp();
    
    std.debug.print("SoA time: {}ns\n", .{end_soa - start_soa});
}
```

---

## 12. Documentation (MANDATORY)

### A. Doc Comments

```zig
/// A user in the system.
///
/// This struct represents a registered user with validated email
/// and basic profile information.
///
/// # Memory Management
/// The caller owns the User and must call `deinit()` to free memory.
///
/// # Example
/// ```zig
/// const allocator = std.heap.page_allocator;
/// var user = try User.init(
///     allocator,
///     "user-123",
///     "John Doe",
///     "john@example.com",
/// );
/// defer user.deinit();
/// ```
pub const User = struct {
    /// Unique identifier for the user
    id: []const u8,
    
    /// User's display name (non-empty)
    name: []const u8,
    
    /// Validated email address
    email: []const u8,
    
    /// Allocator used for memory management
    allocator: std.mem.Allocator,
    
    /// Creates a new user with validated fields.
    ///
    /// # Arguments
    /// * `allocator` - Memory allocator for string allocations
    /// * `id` - Unique user identifier
    /// * `name` - User's display name (must not be empty)
    /// * `email` - Validated email address
    ///
    /// # Returns
    /// Returns `User` on success, or error on failure.
    ///
    /// # Errors
    /// Returns `error.EmptyName` if name is empty.
    /// Returns `error.OutOfMemory` if allocation fails.
    ///
    /// # Memory
    /// Allocates memory for id, name, and email copies.
    /// Caller must call `deinit()` to free memory.
    pub fn init(
        allocator: std.mem.Allocator,
        id: []const u8,
        name: []const u8,
        email: []const u8,
    ) !User {
        // Implementation
    }
    
    /// Frees all allocated memory.
    ///
    /// After calling this, the User instance is no longer valid.
    pub fn deinit(self: *User) void {
        // Implementation
    }
};
```

### B. Module Documentation

```zig
//! User domain module.
//!
//! This module contains the core user domain logic including:
//! - User entity and value objects
//! - User business rules and validations
//! - User-related errors
//!
//! # Architecture
//!
//! The user module follows hexagonal architecture principles:
//! - Domain entities are pure business logic
//! - Repository ports define interfaces
//! - Adapters implement persistence
//!
//! # Example
//!
//! ```zig
//! const std = @import("std");
//! const user = @import("domain/user.zig");
//!
//! pub fn main() !void {
//!     var gpa = std.heap.GeneralPurposeAllocator(.{}){};
//!     defer _ = gpa.deinit();
//!     const allocator = gpa.allocator();
//!
//!     var u = try user.User.init(allocator, "1", "John", "john@test.com");
//!     defer u.deinit();
//! }
//! ```

const std = @import("std");

pub const User = @import("entities/user.zig").User;
pub const UserId = @import("value_objects/user_id.zig").UserId;
pub const Email = @import("value_objects/email.zig").Email;
```

### C. Generating Documentation

```bash
# Generate documentation
zig test src/main.zig -femit-docs

# Or via build system
zig build docs

# Documentation generated in zig-out/docs/
```

---

## 13. Complete Example

```zig
//! Order management module demonstrating Zig patterns.
//!
//! This module shows hexagonal architecture, CQRS, comptime,
//! and explicit memory management in Zig.

const std = @import("std");

// ============================================================================
// Domain - Value Objects
// ============================================================================

/// Order ID newtype
pub const OrderId = struct {
    value: []const u8,
    allocator: std.mem.Allocator,
    
    pub fn init(allocator: std.mem.Allocator, value: []const u8) !OrderId {
        const value_copy = try allocator.dupe(u8, value);
        return OrderId{
            .value = value_copy,
            .allocator = allocator,
        };
    }
    
    pub fn deinit(self: *OrderId) void {
        self.allocator.free(self.value);
    }
    
    pub fn eql(self: OrderId, other: OrderId) bool {
        return std.mem.eql(u8, self.value, other.value);
    }
};

/// Money value object
pub const Money = struct {
    amount: f64,
    currency: Currency,
    
    pub const Currency = enum {
        USD,
        EUR,
        GBP,
    };
    
    pub fn init(amount: f64, currency: Currency) !Money {
        if (amount < 0) {
            return error.NegativeAmount;
        }
        return Money{ .amount = amount, .currency = currency };
    }
    
    pub fn add(self: Money, other: Money) !Money {
        if (self.currency != other.currency) {
            return error.CurrencyMismatch;
        }
        return Money{
            .amount = self.amount + other.amount,
            .currency = self.currency,
        };
    }
};

// ============================================================================
// Domain - Entities
// ============================================================================

/// Order aggregate root
pub const Order = struct {
    id: OrderId,
    customer_id: []const u8,
    items: []OrderItem,
    state: OrderState,
    created_at: i64,
    allocator: std.mem.Allocator,
    
    pub const OrderItem = struct {
        product_id: []const u8,
        quantity: u32,
        unit_price: Money,
    };
    
    /// Creates a new pending order
    pub fn create(
        allocator: std.mem.Allocator,
        customer_id: []const u8,
        items: []const OrderItem,
    ) !Order {
        if (items.len == 0) {
            return error.EmptyOrder;
        }
        
        const id = try OrderId.init(allocator, try generateOrderId(allocator));
        errdefer id.deinit();
        
        const customer_copy = try allocator.dupe(u8, customer_id);
        errdefer allocator.free(customer_copy);
        
        const items_copy = try allocator.dupe(OrderItem, items);
        errdefer allocator.free(items_copy);
        
        return Order{
            .id = id,
            .customer_id = customer_copy,
            .items = items_copy,
            .state = .pending,
            .created_at = std.time.timestamp(),
            .allocator = allocator,
        };
    }
    
    pub fn deinit(self: *Order) void {
        self.id.deinit();
        self.allocator.free(self.customer_id);
        self.allocator.free(self.items);
    }
    
    /// Calculates total order value
    pub fn total(self: Order) !Money {
        var result = try Money.init(0, .USD);
        
        for (self.items) |item| {
            const subtotal = Money{
                .amount = item.unit_price.amount * @as(f64, @floatFromInt(item.quantity)),
                .currency = item.unit_price.currency,
            };
            result = try result.add(subtotal);
        }
        
        return result;
    }
    
    /// Confirms the order
    pub fn confirm(self: *Order) !void {
        self.state = try self.state.confirm();
    }
};

// ============================================================================
// Domain - State Machine
// ============================================================================

/// Order state machine
pub const OrderState = enum {
    pending,
    confirmed,
    shipped,
    delivered,
    cancelled,
    
    pub fn confirm(self: OrderState) !OrderState {
        return switch (self) {
            .pending => .confirmed,
            else => error.InvalidStateTransition,
        };
    }
    
    pub fn ship(self: OrderState) !OrderState {
        return switch (self) {
            .confirmed => .shipped,
            else => error.InvalidStateTransition,
        };
    }
};

// ============================================================================
// Application - Ports
// ============================================================================

/// Repository port
pub const OrderRepository = struct {
    ptr: *anyopaque,
    vtable: *const VTable,
    
    pub const VTable = struct {
        findById: *const fn (ctx: *anyopaque, id: OrderId) anyerror!?Order,
        save: *const fn (ctx: *anyopaque, order: Order) anyerror!void,
    };
    
    pub fn findById(self: OrderRepository, id: OrderId) !?Order {
        return self.vtable.findById(self.ptr, id);
    }
    
    pub fn save(self: OrderRepository, order: Order) !void {
        return self.vtable.save(self.ptr, order);
    }
};

// ============================================================================
// Application - Commands (CQRS)
// ============================================================================

/// Command: Create order
pub const CreateOrderCommand = struct {
    customer_id: []const u8,
    items: []const Order.OrderItem,
};

/// Command handler
pub const CreateOrderHandler = struct {
    repository: OrderRepository,
    allocator: std.mem.Allocator,
    
    pub fn init(allocator: std.mem.Allocator, repository: OrderRepository) CreateOrderHandler {
        return .{
            .repository = repository,
            .allocator = allocator,
        };
    }
    
    pub fn handle(self: CreateOrderHandler, cmd: CreateOrderCommand) !OrderId {
        var order = try Order.create(
            self.allocator,
            cmd.customer_id,
            cmd.items,
        );
        defer order.deinit();
        
        try self.repository.save(order);
        
        return try OrderId.init(self.allocator, order.id.value);
    }
};

// ============================================================================
// Infrastructure - In-Memory Adapter
// ============================================================================

pub const InMemoryOrderRepository = struct {
    orders: std.StringHashMap(Order),
    allocator: std.mem.Allocator,
    
    pub fn init(allocator: std.mem.Allocator) InMemoryOrderRepository {
        return .{
            .orders = std.StringHashMap(Order).init(allocator),
            .allocator = allocator,
        };
    }
    
    pub fn deinit(self: *InMemoryOrderRepository) void {
        var it = self.orders.iterator();
        while (it.next()) |entry| {
            var order = entry.value_ptr.*;
            order.deinit();
        }
        self.orders.deinit();
    }
    
    fn findByIdImpl(ctx: *anyopaque, id: OrderId) !?Order {
        const self: *InMemoryOrderRepository = @ptrCast(@alignCast(ctx));
        return self.orders.get(id.value);
    }
    
    fn saveImpl(ctx: *anyopaque, order: Order) !void {
        const self: *InMemoryOrderRepository = @ptrCast(@alignCast(ctx));
        try self.orders.put(order.id.value, order);
    }
    
    pub fn repository(self: *InMemoryOrderRepository) OrderRepository {
        return .{
            .ptr = self,
            .vtable = &.{
                .findById = findByIdImpl,
                .save = saveImpl,
            },
        };
    }
};

// ============================================================================
// Helpers
// ============================================================================

fn generateOrderId(allocator: std.mem.Allocator) ![]const u8 {
    return try std.fmt.allocPrint(allocator, "order-{}", .{std.time.timestamp()});
}

// ============================================================================
// Tests
// ============================================================================

test "Order creation" {
    const allocator = std.testing.allocator;
    
    const items = [_]Order.OrderItem{
        .{
            .product_id = "prod-1",
            .quantity = 2,
            .unit_price = try Money.init(10.0, .USD),
        },
    };
    
    var order = try Order.create(allocator, "customer-1", &items);
    defer order.deinit();
    
    try std.testing.expectEqual(OrderState.pending, order.state);
    try std.testing.expectEqual(@as(usize, 1), order.items.len);
}

test "Order total calculation" {
    const allocator = std.testing.allocator;
    
    const items = [_]Order.OrderItem{
        .{
            .product_id = "prod-1",
            .quantity = 2,
            .unit_price = try Money.init(10.0, .USD),
        },
        .{
            .product_id = "prod-2",
            .quantity = 1,
            .unit_price = try Money.init(15.0, .USD),
        },
    };
    
    var order = try Order.create(allocator, "customer-1", &items);
    defer order.deinit();
    
    const total = try order.total();
    try std.testing.expectEqual(@as(f64, 35.0), total.amount);
}

test "State transitions" {
    const allocator = std.testing.allocator;
    
    const items = [_]Order.OrderItem{
        .{
            .product_id = "prod-1",
            .quantity = 1,
            .unit_price = try Money.init(10.0, .USD),
        },
    };
    
    var order = try Order.create(allocator, "customer-1", &items);
    defer order.deinit();
    
    try order.confirm();
    try std.testing.expectEqual(OrderState.confirmed, order.state);
}

test "Memory leak detection" {
    const allocator = std.testing.allocator;
    
    const items = [_]Order.OrderItem{
        .{
            .product_id = "prod-1",
            .quantity = 1,
            .unit_price = try Money.init(10.0, .USD),
        },
    };
    
    var order = try Order.create(allocator, "customer-1", &items);
    defer order.deinit(); // If forgotten, test fails
    
    _ = order;
}
```

---

## 14. Deployment Checklist

### Pre-Production Validation

#### Compilation (MANDATORY)
- [ ] **Builds successfully**: `zig build` passes
- [ ] **Release builds**: `zig build -Doptimize=ReleaseFast` passes
- [ ] **Safe release builds**: `zig build -Doptimize=ReleaseSafe` passes
- [ ] **No warnings**: Compilation produces no warnings
- [ ] **Cross-compilation tested**: Tested on target platforms

#### Formatting (MANDATORY)
- [ ] **Code formatted**: `zig fmt --check src/` passes
- [ ] **Consistent style**: All files follow Zig style

#### Testing (MANDATORY)
- [ ] **All tests pass**: `zig test src/main.zig` returns exit code 0
- [ ] **No memory leaks**: Tests use `std.testing.allocator`
- [ ] **Integration tests pass**: `zig build test` succeeds
- [ ] **Benchmarks run**: Performance tests complete

#### Documentation (MANDATORY)
- [ ] **All public APIs documented**: Doc comments on all `pub` items
- [ ] **Documentation builds**: `zig build docs` succeeds
- [ ] **Examples in docs**: Code examples in doc comments
- [ ] **Module docs present**: Top-level module documentation

#### Memory Management
- [ ] **Allocators explicit**: All allocations pass allocators
- [ ] **Defer cleanup**: All allocations paired with defer
- [ ] **Arena allocators used**: Where appropriate
- [ ] **No memory leaks**: Verified with testing allocator

#### Architecture
- [ ] **Hexagonal architecture**: Clear layer separation
- [ ] **CQRS implemented**: Commands and queries separated
- [ ] **Ports as interfaces**: Repository pattern used
- [ ] **Pure Zig**: No C/C++ dependencies unless necessary

#### Performance
- [ ] **Data-oriented**: SoA used where appropriate
- [ ] **Comptime leveraged**: Generic code uses comptime
- [ ] **Cache-friendly**: Sequential data access patterns

---

## 15. Why This Configuration Works

1. **Explicit Memory Management**: No hidden allocations, predictable performance, easier debugging.

2. **Comptime Programming**: Zero-cost abstractions, type-safe generics, compile-time validation.

3. **Data-Oriented Design**: Cache-friendly layouts, SIMD-friendly code, 10-100x performance improvements.

4. **Hexagonal Architecture**: Testable in isolation, clear boundaries, flexible adapters.

5. **CQRS**: Optimized read/write paths, scalable architecture, clear data flow.

6. **Error Union Types**: Explicit error handling, compile-time error checking, no exceptions.

7. **Defer Pattern**: Guaranteed cleanup, no resource leaks, simple error handling.

8. **Pure Zig**: Portable across platforms, no FFI complexity, easier deployment.

9. **Built-in Testing**: Tests alongside code, no external framework, fast execution.

10. **Agent Verification**: Ensures all code compiles and tests pass, eliminates broken code.

---

## References

- [Zig Language Reference](https://ziglang.org/documentation/master/)
- [Zig Standard Library](https://ziglang.org/documentation/master/std/)
- [Zig Build System](https://ziglang.org/learn/build-system/)
- [Zig By Example](https://zigbyexample.github.io/)
- [Data-Oriented Design](https://www.dataorienteddesign.com/dodbook/)

---

**Last Updated:** 2026-01-17
**Version:** 1.0
**Maintainer:** Development Team
