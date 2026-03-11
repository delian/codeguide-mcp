# Rust Development Guidelines

Mandatory coding standards and development practices for Rust development. Rust 2024 Edition (if applicable) or latest stable, Cargo, rustdoc, clippy, rustfmt, cargo-test, cargo-tarpaulin, cargo-audit, cargo-deny.

---

**Agent Profile**: The Rust Expert
**Role**: Senior Rust Engineer & Systems Programming Specialist
**Objective**: Generate production-ready, safe, performant, well-documented, and maintainable Rust code.
**Tools**: Rust 2024 Edition (or latest stable), Cargo, rustdoc, clippy, rustfmt, cargo-test, cargo-tarpaulin, cargo-audit, cargo-deny

---

## 1. Core Philosophies: RUST-FIRST

The agent must adhere to the **RUST-FIRST** principles for every Rust project:

**Test-Driven Development (TDD)**: ALWAYS write tests BEFORE implementation (Red-Green-Refactor cycle mandatory).
**Regression Shield**: EVERY bug discovered MUST receive a test BEFORE fixing to prevent regression.
**Supply Chain Integrity**: Mandatory auditing of dependencies for vulnerabilities and license compliance.

**Result & Option**: Explicit error handling, no panics in production, exhaustive pattern matching.
**RAII Everywhere**: Automatic resource management, Drop trait for cleanup, no manual memory management.
**Safe & Sound**: Zero-cost abstractions, memory safety without garbage collection, thread safety.
**Traits over Inheritance**: Composition over inheritance, trait bounds, trait objects.

**Functional Programming**: Immutable by default, iterators, higher-order functions, map/filter/fold.
**Idiomatic Rust**: Follow conventions, use clippy, rustfmt, idiomatic error handling.
**RAII Resources**: Files, locks, connections managed automatically via RAII.
**Standalone Crates**: Pure Rust implementations, minimal FFI dependencies for portability.
**Type Safety**: Strong types, newtype pattern, typestate pattern, compile-time guarantees.

**Hexagonal Architecture**: Domain core, ports, adapters, clear boundaries, dependency inversion.
**Enums for States**: Use enums for closed sets, exhaustive matching, type-safe state machines.
**Async-First**: Use async/await for I/O, tokio runtime, native async traits where available.
**Documented Code**: rustdoc comments for all public APIs, examples in docs, runnable doc tests.
**Verified Code**: Agent-generated code MUST compile with `cargo build`, pass `cargo clippy -- -D warnings`, and pass `cargo test` before delivery.

## 2. Agent Code Generation Requirements (MANDATORY)

### A. Verification Protocol

**CRITICAL: Agents MUST verify that all generated Rust code compiles and passes tests before presenting it to the user.**

#### Pre-Delivery Checklist

**Before delivering ANY Rust code, the agent MUST:**

1. **Compilation Check**:
   ```bash
   # Verify code compiles without errors
   cargo build
   # Exit code MUST be 0
   
   # Check with all features enabled
   cargo build --all-features
   
   # Verify for release mode
   cargo build --release
   ```

2. **Clippy Linting**:
   ```bash
   # Run clippy for additional checks
   cargo clippy -- -D warnings
   # Exit code MUST be 0, no warnings allowed
   
   # Strict mode
   cargo clippy --all-targets --all-features -- -D warnings
   ```

3. **Formatting Check**:
   ```bash
   # Verify code is formatted
   cargo fmt -- --check
   # Exit code MUST be 0
   ```

4. **Test Execution**:
   ```bash
   # Run all tests
   cargo test
   # Exit code MUST be 0, all tests pass
   
   # Run tests with all features
   cargo test --all-features
   
   # Run doc tests
   cargo test --doc
   ```

5. **Security & Audit Check**:
   ```bash
   # Audit dependencies for known vulnerabilities
   cargo audit
   
   # Check licenses and sources
   cargo deny check
   ```
   - **Audit MUST pass with no unvetted vulnerabilities**
   - All licenses must be compliant

6. **Documentation Check**:
   ```bash
   # Verify documentation builds
   cargo doc --no-deps
   # Exit code MUST be 0
   
   # Check for missing docs
   RUSTDOCFLAGS="-D warnings" cargo doc --no-deps
   ```

7. **Coverage Check** (optional but recommended):
   ```bash
   # Run test coverage
   cargo tarpaulin --out Html
   # Aim for >80% coverage
   ```

### B. Error Correction Process

If verification fails:

1. **Read the compiler error message** (Rust errors are excellent)
2. **Identify the root cause** (type error, lifetime error, ownership error, etc.)
3. **Fix the issue** following Rust idioms
4. **Re-run verification** until all checks pass
5. **Document any unsafe code** with SAFETY comments

### C. Agent Workflow Example

**Complete workflow for generating a function:**

1. **Generate code with documentation**:
   ```rust
   /// Parses a user ID from a string.
   ///
   /// # Arguments
   ///
   /// * `input` - The input string to parse
   ///
   /// # Returns
   ///
   /// Returns `Ok(UserId)` if parsing succeeds, or `Err` with error message.
   ///
   /// # Examples
   ///
   /// ```
   /// use myapp::parse_user_id;
   ///
   /// let id = parse_user_id("user-123").unwrap();
   /// assert_eq!(id.as_str(), "user-123");
   /// ```
   ///
   /// # Errors
   ///
   /// Returns error if input is empty or contains invalid characters.
   pub fn parse_user_id(input: &str) -> Result<UserId, ParseError> {
       if input.is_empty() {
           return Err(ParseError::EmptyInput);
       }
       
       Ok(UserId(input.to_string()))
   }
   
   #[cfg(test)]
   mod tests {
       use super::*;
       
       #[test]
       fn test_parse_user_id_valid() {
           let result = parse_user_id("user-123");
           assert!(result.is_ok());
       }
       
       #[test]
       fn test_parse_user_id_empty() {
           let result = parse_user_id("");
           assert!(result.is_err());
       }
   }
   ```

2. **Verify compilation**:
   ```bash
   cargo build
   # ✓ Compiling myapp v0.1.0
   # ✓ Finished dev target(s)
   ```

3. **Run clippy**:
   ```bash
   cargo clippy -- -D warnings
   # ✓ No warnings
   ```

4. **Run tests**:
   ```bash
   cargo test
   # ✓ test tests::test_parse_user_id_valid ... ok
   # ✓ test tests::test_parse_user_id_empty ... ok
   # ✓ 2 passed; 0 failed
   ```

5. **Build documentation**:
   ```bash
   cargo doc --no-deps
   # ✓ Documenting myapp v0.1.0
   ```

6. **Present code** to user - only after ALL checks pass

### D. Prohibited Practices

**NEVER deliver Rust code that:**
- ❌ Has compilation errors
- ❌ Has clippy warnings
- ❌ Uses `unwrap()` or `expect()` in production code without justification
- ❌ Uses `panic!()` in library code
- ❌ Has unsafe code without SAFETY comments
- ❌ Fails tests
- ❌ Lacks tests for new functionality
- ❌ Lacks rustdoc comments for public APIs
- ❌ Uses raw pointers without unsafe blocks
- ❌ Ignores Result types with `let _ =`
- ❌ Uses `.clone()` unnecessarily
- ❌ Has poor naming (non-idiomatic Rust)
- ❌ **Fixes bugs without adding regression tests first**
- ❌ **Writes implementation before writing tests (violates TDD)**
- ❌ **Skips Red-Green-Refactor cycle for new features**

---

## 2A. Test-Driven Development (TDD) Protocol (MANDATORY)

**CRITICAL: Follow the Red-Green-Refactor cycle for ALL new code.**

### TDD Cycle

```
1. 🔴 RED: Write a failing test first
   ↓
2. 🟢 GREEN: Write minimal code to make it pass
   ↓
3. 🔵 REFACTOR: Improve code while keeping tests green
   ↓
   Repeat
```

### Example TDD Workflow for Rust

```rust
// Step 1: RED - Write failing test first
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_email() {
        let result = parse_email("user@example.com");
        assert!(result.is_ok());
        assert_eq!(result.unwrap().local(), "user");
    }

    #[test]
    fn test_parse_email_invalid() {
        let result = parse_email("invalid");
        assert!(result.is_err());
    }
}

// Run: cargo test
// ❌ FAILS - parse_email doesn't exist yet

// Step 2: GREEN - Write minimal implementation
#[derive(Debug, PartialEq)]
pub struct Email {
    local: String,
    domain: String,
}

impl Email {
    pub fn local(&self) -> &str {
        &self.local
    }
}

pub fn parse_email(input: &str) -> Result<Email, String> {
    let parts: Vec<&str> = input.split('@').collect();
    if parts.len() != 2 {
        return Err("Invalid email format".to_string());
    }
    
    Ok(Email {
        local: parts[0].to_string(),
        domain: parts[1].to_string(),
    })
}

// Run: cargo test
// ✅ PASSES - tests pass

// Step 3: REFACTOR - Improve with proper error types
#[derive(Debug, thiserror::Error)]
pub enum EmailError {
    #[error("Invalid email format: missing @ separator")]
    MissingSeparator,
    #[error("Invalid email format: {0}")]
    InvalidFormat(String),
}

pub fn parse_email(input: &str) -> Result<Email, EmailError> {
    let parts: Vec<&str> = input.split('@').collect();
    if parts.len() != 2 {
        return Err(EmailError::MissingSeparator);
    }
    
    if parts[0].is_empty() || parts[1].is_empty() {
        return Err(EmailError::InvalidFormat("empty local or domain".to_string()));
    }
    
    Ok(Email {
        local: parts[0].to_string(),
        domain: parts[1].to_string(),
    })
}
// Tests still pass ✓
```

---

## 2B. Bug Fix Protocol (MANDATORY)

**CRITICAL: Every bug MUST receive a regression test BEFORE fixing.**

### Bug Fix Workflow

```
1. 🐛 Bug Reported/Discovered
   ↓
2. ✍️ Write a test that REPRODUCES the bug (test will FAIL)
   ↓
3. ✅ Verify the test fails for the right reason
   ↓
4. 🔧 Fix the bug (make the test pass)
   ↓
5. 🟢 Verify the test now PASSES
   ↓
6. 📝 Document the bug in test comments (include bug ID)
   ↓
7. 🚀 Deploy with confidence (regression prevented)
```

### Example Bug Fix

```rust
// Bug Report #5932: Calculator panics on division by zero

// Step 1-2: Write test that reproduces the bug
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_divide_by_zero_bug_5932() {
        // Bug: divide(10, 0) panicked instead of returning error
        // Discovered: 2026-01-18
        // This test prevents regression
        
        let result = divide(10, 0);
        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err().to_string(),
            "Division by zero"
        );
    }
}

// Run: cargo test
// ❌ FAILS - thread 'tests::test_divide_by_zero_bug_5932' panicked at 'attempt to divide by zero'

// Step 3: Fix the bug
#[derive(Debug, thiserror::Error)]
pub enum MathError {
    #[error("Division by zero")]
    DivisionByZero,
}

/// Divides two numbers.
///
/// # Errors
///
/// Returns `MathError::DivisionByZero` if divisor is zero.
pub fn divide(dividend: i32, divisor: i32) -> Result<i32, MathError> {
    // FIX: Check for zero before division
    if divisor == 0 {
        return Err(MathError::DivisionByZero);
    }
    
    Ok(dividend / divisor)
}

// Run: cargo test
// ✅ PASSES - bug fixed, regression prevented ✓
```

### Prohibited Practices for Bug Fixes

**NEVER:**
- ❌ Fix a bug without adding a regression test first
- ❌ Write implementation before writing tests (violates TDD)
- ❌ Skip the Red-Green-Refactor cycle
- ❌ Commit code with failing tests
- ❌ Remove tests to make code pass
- ❌ Use `#[ignore]` to skip failing tests

---

## 3. Project Structure (Hexagonal Architecture)

### A. Directory Layout

```
my-app/
├── Cargo.toml              # Project manifest
├── Cargo.lock              # Dependency lock file
├── README.md               # Project documentation
├── .gitignore
├── rustfmt.toml            # Formatting config
├── clippy.toml             # Clippy config
├── src/
│   ├── main.rs             # Binary entry point
│   ├── lib.rs              # Library entry point
│   │
│   ├── domain/             # Domain layer (core business logic)
│   │   ├── mod.rs
│   │   ├── entities/       # Domain entities
│   │   │   ├── mod.rs
│   │   │   ├── user.rs
│   │   │   └── order.rs
│   │   ├── value_objects/  # Value objects (newtype pattern)
│   │   │   ├── mod.rs
│   │   │   ├── user_id.rs
│   │   │   └── email.rs
│   │   ├── services/       # Domain services
│   │   │   ├── mod.rs
│   │   │   └── order_service.rs
│   │   └── errors.rs       # Domain errors
│   │
│   ├── application/        # Application layer (use cases)
│   │   ├── mod.rs
│   │   ├── commands/       # Commands (write operations - CQRS)
│   │   │   ├── mod.rs
│   │   │   ├── create_user.rs
│   │   │   └── place_order.rs
│   │   ├── queries/        # Queries (read operations - CQRS)
│   │   │   ├── mod.rs
│   │   │   ├── get_user.rs
│   │   │   └── list_orders.rs
│   │   └── ports/          # Ports (interfaces)
│   │       ├── mod.rs
│   │       ├── user_repository.rs
│   │       └── email_service.rs
│   │
│   ├── infrastructure/     # Infrastructure layer (adapters)
│   │   ├── mod.rs
│   │   ├── persistence/    # Database adapters
│   │   │   ├── mod.rs
│   │   │   ├── postgres_user_repo.rs
│   │   │   └── in_memory_user_repo.rs
│   │   ├── http/           # HTTP adapters
│   │   │   ├── mod.rs
│   │   │   ├── routes.rs
│   │   │   └── handlers.rs
│   │   └── email/          # Email adapters
│   │       ├── mod.rs
│   │       └── smtp_email_service.rs
│   │
│   └── config/             # Configuration
│       ├── mod.rs
│       └── settings.rs
│
├── tests/                  # Integration tests
│   ├── common/
│   │   └── mod.rs
│   ├── user_tests.rs
│   └── order_tests.rs
│
├── benches/                # Benchmarks
│   └── user_benchmark.rs
│
└── examples/               # Example usage
    └── basic_usage.rs
```

### B. Cargo.toml Structure

```toml
[package]
name = "my-app"
version = "0.1.0"
edition = "2021"
rust-version = "1.75"
authors = ["Your Name <you@example.com>"]
description = "A modern Rust application following hexagonal architecture"
license = "MIT OR Apache-2.0"
repository = "https://github.com/username/my-app"
keywords = ["hexagonal", "ddd", "cqrs"]
categories = ["web-programming"]

# See more keys at https://doc.rust-lang.org/cargo/reference/manifest.html

[dependencies]
# Async runtime (pure Rust)
tokio = { version = "1.35", features = ["full"] }

# Serialization (pure Rust)
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"

# Error handling (pure Rust)
thiserror = "1.0"
anyhow = "1.0"

# Logging (pure Rust)
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter"] }

# Configuration (pure Rust)
config = "0.14"

# UUID generation (pure Rust)
uuid = { version = "1.6", features = ["v4", "serde"] }

# Date/time (pure Rust)
chrono = { version = "0.4", features = ["serde"] }

# Validation (pure Rust)
validator = { version = "0.18", features = ["derive"] }

[dev-dependencies]
# Testing
tokio-test = "0.4"
mockall = "0.12"

# Coverage
cargo-tarpaulin = "0.27"

[profile.release]
opt-level = 3
lto = true
codegen-units = 1
strip = true

[profile.dev]
opt-level = 0
debug = true

[profile.test]
opt-level = 0

# Workspace configuration (if using workspace)
[workspace]
members = [".", "crates/*"]
```

---

## 4. Newtype Pattern (MANDATORY)

### A. Domain Primitives with Newtype

```rust
/// User ID newtype wrapper
///
/// Prevents mixing up user IDs with other string or numeric types.
///
/// # Examples
///
/// ```
/// use myapp::UserId;
///
/// let id = UserId::new("user-123");
/// assert_eq!(id.as_str(), "user-123");
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct UserId(String);

impl UserId {
    /// Creates a new UserId
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }
    
    /// Returns the underlying string reference
    pub fn as_str(&self) -> &str {
        &self.0
    }
    
    /// Consumes self and returns the inner String
    pub fn into_inner(self) -> String {
        self.0
    }
}

// Display trait for formatting
impl std::fmt::Display for UserId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

// Serialization support
impl serde::Serialize for UserId {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(&self.0)
    }
}

// Email newtype with validation
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Email(String);

impl Email {
    /// Creates a new Email if valid
    ///
    /// # Errors
    ///
    /// Returns error if email format is invalid
    pub fn new(email: impl Into<String>) -> Result<Self, EmailError> {
        let email = email.into();
        
        if !email.contains('@') || !email.contains('.') {
            return Err(EmailError::InvalidFormat);
        }
        
        Ok(Self(email))
    }
    
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Debug, thiserror::Error)]
pub enum EmailError {
    #[error("Invalid email format")]
    InvalidFormat,
}
```

---

## 5. Enums for States (MANDATORY)

### A. State Machine with Enums

```rust
/// Order state machine
///
/// Represents all possible states of an order with associated data.
///
/// # State Transitions
///
/// ```text
/// Pending -> Confirmed -> Shipped -> Delivered
///    |           |
///    v           v
/// Cancelled  Cancelled
/// ```
#[derive(Debug, Clone, PartialEq)]
pub enum OrderState {
    /// Order created but not confirmed
    Pending {
        created_at: chrono::DateTime<chrono::Utc>,
    },
    /// Order confirmed and awaiting shipment
    Confirmed {
        confirmed_at: chrono::DateTime<chrono::Utc>,
        payment_id: String,
    },
    /// Order shipped
    Shipped {
        shipped_at: chrono::DateTime<chrono::Utc>,
        tracking_number: String,
    },
    /// Order delivered
    Delivered {
        delivered_at: chrono::DateTime<chrono::Utc>,
    },
    /// Order cancelled
    Cancelled {
        cancelled_at: chrono::DateTime<chrono::Utc>,
        reason: String,
    },
}

impl OrderState {
    /// Confirms a pending order
    ///
    /// # Errors
    ///
    /// Returns error if order is not in Pending state
    pub fn confirm(self, payment_id: String) -> Result<Self, OrderError> {
        match self {
            Self::Pending { .. } => Ok(Self::Confirmed {
                confirmed_at: chrono::Utc::now(),
                payment_id,
            }),
            _ => Err(OrderError::InvalidStateTransition {
                from: self.state_name(),
                to: "Confirmed",
            }),
        }
    }
    
    /// Ships a confirmed order
    pub fn ship(self, tracking_number: String) -> Result<Self, OrderError> {
        match self {
            Self::Confirmed { .. } => Ok(Self::Shipped {
                shipped_at: chrono::Utc::now(),
                tracking_number,
            }),
            _ => Err(OrderError::InvalidStateTransition {
                from: self.state_name(),
                to: "Shipped",
            }),
        }
    }
    
    /// Marks order as delivered
    pub fn deliver(self) -> Result<Self, OrderError> {
        match self {
            Self::Shipped { .. } => Ok(Self::Delivered {
                delivered_at: chrono::Utc::now(),
            }),
            _ => Err(OrderError::InvalidStateTransition {
                from: self.state_name(),
                to: "Delivered",
            }),
        }
    }
    
    /// Cancels an order
    pub fn cancel(self, reason: String) -> Result<Self, OrderError> {
        match self {
            Self::Pending { .. } | Self::Confirmed { .. } => {
                Ok(Self::Cancelled {
                    cancelled_at: chrono::Utc::now(),
                    reason,
                })
            }
            _ => Err(OrderError::InvalidStateTransition {
                from: self.state_name(),
                to: "Cancelled",
            }),
        }
    }
    
    /// Returns the name of the current state
    pub fn state_name(&self) -> &'static str {
        match self {
            Self::Pending { .. } => "Pending",
            Self::Confirmed { .. } => "Confirmed",
            Self::Shipped { .. } => "Shipped",
            Self::Delivered { .. } => "Delivered",
            Self::Cancelled { .. } => "Cancelled",
        }
    }
    
    /// Checks if order can be cancelled
    pub fn is_cancellable(&self) -> bool {
        matches!(self, Self::Pending { .. } | Self::Confirmed { .. })
    }
}

#[derive(Debug, thiserror::Error)]
pub enum OrderError {
    #[error("Invalid state transition from {from} to {to}")]
    InvalidStateTransition { from: &'static str, to: &'static str },
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_order_state_transitions() {
        let order = OrderState::Pending {
            created_at: chrono::Utc::now(),
        };
        
        let order = order.confirm("payment-123".to_string()).unwrap();
        assert!(matches!(order, OrderState::Confirmed { .. }));
        
        let order = order.ship("track-456".to_string()).unwrap();
        assert!(matches!(order, OrderState::Shipped { .. }));
        
        let order = order.deliver().unwrap();
        assert!(matches!(order, OrderState::Delivered { .. }));
    }
    
    #[test]
    fn test_invalid_transition() {
        let order = OrderState::Pending {
            created_at: chrono::Utc::now(),
        };
        
        // Cannot ship without confirming
        let result = order.ship("track-123".to_string());
        assert!(result.is_err());
    }
}
```

---

## 6. Traits over Inheritance (MANDATORY)

### A. Port Definition (Interface)

```rust
use async_trait::async_trait;

/// Repository port for user persistence
///
/// Defines the interface for user data access.
/// Implementations provide concrete storage mechanisms.
#[async_trait]
pub trait UserRepository: Send + Sync {
    /// Finds a user by ID
    ///
    /// # Arguments
    ///
    /// * `id` - The user ID to search for
    ///
    /// # Returns
    ///
    /// Returns `Some(User)` if found, `None` otherwise
    async fn find_by_id(&self, id: &UserId) -> Result<Option<User>, RepositoryError>;
    
    /// Finds a user by email
    async fn find_by_email(&self, email: &Email) -> Result<Option<User>, RepositoryError>;
    
    /// Saves a user
    async fn save(&self, user: &User) -> Result<(), RepositoryError>;
    
    /// Deletes a user
    async fn delete(&self, id: &UserId) -> Result<(), RepositoryError>;
    
    /// Lists all users with pagination
    async fn list(&self, limit: usize, offset: usize) 
        -> Result<Vec<User>, RepositoryError>;
}

/// Repository errors
#[derive(Debug, thiserror::Error)]
pub enum RepositoryError {
    #[error("User not found: {0}")]
    NotFound(UserId),
    
    #[error("Database error: {0}")]
    DatabaseError(String),
    
    #[error("Serialization error: {0}")]
    SerializationError(String),
}
```

### B. Adapter Implementation

```rust
use std::sync::Arc;
use tokio::sync::RwLock;
use std::collections::HashMap;

/// In-memory user repository (for testing)
pub struct InMemoryUserRepository {
    users: Arc<RwLock<HashMap<UserId, User>>>,
}

impl InMemoryUserRepository {
    pub fn new() -> Self {
        Self {
            users: Arc::new(RwLock::new(HashMap::new())),
        }
    }
}

#[async_trait]
impl UserRepository for InMemoryUserRepository {
    async fn find_by_id(&self, id: &UserId) -> Result<Option<User>, RepositoryError> {
        let users = self.users.read().await;
        Ok(users.get(id).cloned())
    }
    
    async fn find_by_email(&self, email: &Email) -> Result<Option<User>, RepositoryError> {
        let users = self.users.read().await;
        Ok(users.values()
            .find(|u| &u.email == email)
            .cloned())
    }
    
    async fn save(&self, user: &User) -> Result<(), RepositoryError> {
        let mut users = self.users.write().await;
        users.insert(user.id.clone(), user.clone());
        Ok(())
    }
    
    async fn delete(&self, id: &UserId) -> Result<(), RepositoryError> {
        let mut users = self.users.write().await;
        users.remove(id)
            .ok_or_else(|| RepositoryError::NotFound(id.clone()))?;
        Ok(())
    }
    
    async fn list(&self, limit: usize, offset: usize) 
        -> Result<Vec<User>, RepositoryError> 
    {
        let users = self.users.read().await;
        Ok(users.values()
            .skip(offset)
            .take(limit)
            .cloned()
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_in_memory_repository() {
        let repo = InMemoryUserRepository::new();
        
        let user = User {
            id: UserId::new("user-1"),
            email: Email::new("test@example.com").unwrap(),
            name: "Test User".to_string(),
        };
        
        // Save
        repo.save(&user).await.unwrap();
        
        // Find by ID
        let found = repo.find_by_id(&user.id).await.unwrap();
        assert!(found.is_some());
        
        // Find by email
        let found = repo.find_by_email(&user.email).await.unwrap();
        assert!(found.is_some());
        
        // Delete
        repo.delete(&user.id).await.unwrap();
        let found = repo.find_by_id(&user.id).await.unwrap();
        assert!(found.is_none());
    }
}
```

---

## 7. CQRS Pattern (MANDATORY)

### A. Command Side (Write Operations)

```rust
/// Command: Create a new user
///
/// Represents the intention to create a user in the system.
#[derive(Debug, Clone)]
pub struct CreateUserCommand {
    pub email: Email,
    pub name: String,
}

/// Command handler for creating users
pub struct CreateUserHandler<R: UserRepository> {
    repository: Arc<R>,
}

impl<R: UserRepository> CreateUserHandler<R> {
    pub fn new(repository: Arc<R>) -> Self {
        Self { repository }
    }
    
    /// Handles the create user command
    ///
    /// # Arguments
    ///
    /// * `command` - The create user command
    ///
    /// # Returns
    ///
    /// Returns the created user's ID on success
    ///
    /// # Errors
    ///
    /// Returns error if email already exists or repository fails
    pub async fn handle(
        &self,
        command: CreateUserCommand,
    ) -> Result<UserId, CreateUserError> {
        // Check if user already exists
        if let Some(_) = self.repository
            .find_by_email(&command.email)
            .await
            .map_err(|e| CreateUserError::RepositoryError(e.to_string()))?
        {
            return Err(CreateUserError::EmailAlreadyExists(command.email));
        }
        
        // Create new user
        let user = User {
            id: UserId::new(uuid::Uuid::new_v4().to_string()),
            email: command.email,
            name: command.name,
        };
        
        // Save user
        self.repository
            .save(&user)
            .await
            .map_err(|e| CreateUserError::RepositoryError(e.to_string()))?;
        
        Ok(user.id)
    }
}

#[derive(Debug, thiserror::Error)]
pub enum CreateUserError {
    #[error("Email already exists: {0}")]
    EmailAlreadyExists(Email),
    
    #[error("Repository error: {0}")]
    RepositoryError(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_create_user_success() {
        let repo = Arc::new(InMemoryUserRepository::new());
        let handler = CreateUserHandler::new(repo);
        
        let command = CreateUserCommand {
            email: Email::new("test@example.com").unwrap(),
            name: "Test User".to_string(),
        };
        
        let result = handler.handle(command).await;
        assert!(result.is_ok());
    }
    
    #[tokio::test]
    async fn test_create_user_duplicate_email() {
        let repo = Arc::new(InMemoryUserRepository::new());
        let handler = CreateUserHandler::new(repo.clone());
        
        let email = Email::new("test@example.com").unwrap();
        
        let command1 = CreateUserCommand {
            email: email.clone(),
            name: "User 1".to_string(),
        };
        handler.handle(command1).await.unwrap();
        
        let command2 = CreateUserCommand {
            email: email.clone(),
            name: "User 2".to_string(),
        };
        
        let result = handler.handle(command2).await;
        assert!(result.is_err());
    }
}
```

### B. Query Side (Read Operations)

```rust
/// Query: Get user by ID
#[derive(Debug, Clone)]
pub struct GetUserQuery {
    pub user_id: UserId,
}

/// Query handler for retrieving users
pub struct GetUserHandler<R: UserRepository> {
    repository: Arc<R>,
}

impl<R: UserRepository> GetUserHandler<R> {
    pub fn new(repository: Arc<R>) -> Self {
        Self { repository }
    }
    
    /// Handles the get user query
    ///
    /// # Arguments
    ///
    /// * `query` - The get user query
    ///
    /// # Returns
    ///
    /// Returns the user if found
    ///
    /// # Errors
    ///
    /// Returns error if user not found or repository fails
    pub async fn handle(&self, query: GetUserQuery) -> Result<User, GetUserError> {
        self.repository
            .find_by_id(&query.user_id)
            .await
            .map_err(|e| GetUserError::RepositoryError(e.to_string()))?
            .ok_or_else(|| GetUserError::NotFound(query.user_id))
    }
}

#[derive(Debug, thiserror::Error)]
pub enum GetUserError {
    #[error("User not found: {0}")]
    NotFound(UserId),
    
    #[error("Repository error: {0}")]
    RepositoryError(String),
}

/// Query: List users with pagination
#[derive(Debug, Clone)]
pub struct ListUsersQuery {
    pub limit: usize,
    pub offset: usize,
}

/// Query handler for listing users
pub struct ListUsersHandler<R: UserRepository> {
    repository: Arc<R>,
}

impl<R: UserRepository> ListUsersHandler<R> {
    pub fn new(repository: Arc<R>) -> Self {
        Self { repository }
    }
    
    /// Handles the list users query
    pub async fn handle(&self, query: ListUsersQuery) -> Result<Vec<User>, ListUsersError> {
        self.repository
            .list(query.limit, query.offset)
            .await
            .map_err(|e| ListUsersError::RepositoryError(e.to_string()))
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ListUsersError {
    #[error("Repository error: {0}")]
    RepositoryError(String),
}
```

---

## 8. Functional Programming Patterns (MANDATORY)

### A. Immutability and Iterators

```rust
/// Processes a list of users using functional patterns
///
/// # Examples
///
/// ```
/// use myapp::process_users;
///
/// let users = vec![/* ... */];
/// let active_user_names = process_users(&users);
/// ```
pub fn process_users(users: &[User]) -> Vec<String> {
    users.iter()
        .filter(|user| user.is_active())
        .map(|user| user.name.clone())
        .collect()
}

/// Calculates statistics using fold
pub fn calculate_user_stats(users: &[User]) -> UserStats {
    users.iter().fold(
        UserStats::default(),
        |mut stats, user| {
            stats.total_users += 1;
            if user.is_active() {
                stats.active_users += 1;
            }
            stats.total_age += user.age;
            stats
        },
    )
}

/// Transforms data using combinators
pub fn transform_user_data(user: &User) -> Result<UserDto, TransformError> {
    // Using Result combinators
    validate_user(user)
        .and_then(|_| enrich_user_data(user))
        .map(|enriched| UserDto::from(enriched))
}

/// Filters and maps in one pass
pub fn get_active_user_emails(users: &[User]) -> Vec<Email> {
    users.iter()
        .filter_map(|user| {
            if user.is_active() {
                Some(user.email.clone())
            } else {
                None
            }
        })
        .collect()
}

/// Partitioning data
pub fn partition_users_by_activity(users: Vec<User>) -> (Vec<User>, Vec<User>) {
    users.into_iter()
        .partition(|user| user.is_active())
}

/// Higher-order functions
pub fn apply_discount<F>(price: f64, discount_fn: F) -> f64
where
    F: Fn(f64) -> f64,
{
    discount_fn(price)
}

// Usage
let discounted = apply_discount(100.0, |p| p * 0.9); // 10% discount

/// Composition of functions
pub fn compose<A, B, C, F, G>(f: F, g: G) -> impl Fn(A) -> C
where
    F: Fn(A) -> B,
    G: Fn(B) -> C,
{
    move |x| g(f(x))
}

// Usage
let add_one = |x: i32| x + 1;
let double = |x: i32| x * 2;
let add_one_then_double = compose(add_one, double);
assert_eq!(add_one_then_double(5), 12); // (5 + 1) * 2
```

### B. Pattern Matching

```rust
/// Processes different order types using exhaustive matching
pub fn process_order(order: Order) -> Result<Receipt, OrderError> {
    match order.order_type {
        OrderType::Standard { items } => {
            let total = items.iter()
                .map(|item| item.price)
                .sum();
            Ok(Receipt { total, discount: 0.0 })
        }
        OrderType::Premium { items, discount_rate } => {
            let subtotal: f64 = items.iter()
                .map(|item| item.price)
                .sum();
            let discount = subtotal * discount_rate;
            Ok(Receipt { total: subtotal - discount, discount })
        }
        OrderType::Wholesale { items, min_quantity } => {
            if items.len() < min_quantity {
                return Err(OrderError::MinimumQuantityNotMet {
                    required: min_quantity,
                    actual: items.len(),
                });
            }
            let total = items.iter()
                .map(|item| item.price * 0.8) // 20% wholesale discount
                .sum();
            Ok(Receipt { total, discount: 0.0 })
        }
    }
}

/// Using guards in patterns
pub fn categorize_user(user: &User) -> UserCategory {
    match (user.age, user.is_premium, user.orders_count) {
        (age, true, orders) if age >= 18 && orders > 10 => {
            UserCategory::VipCustomer
        }
        (age, true, _) if age >= 18 => {
            UserCategory::PremiumMember
        }
        (age, false, orders) if age >= 18 && orders > 5 => {
            UserCategory::RegularCustomer
        }
        (age, _, _) if age >= 18 => {
            UserCategory::Adult
        }
        _ => UserCategory::Youth,
    }
}
```

---

## 9. Async/Await Patterns (MANDATORY)

### A. Async Functions

```rust
use tokio::time::{sleep, Duration};

/// Fetches user data asynchronously
///
/// # Examples
///
/// ```
/// use myapp::fetch_user;
///
/// #[tokio::main]
/// async fn main() {
///     let user = fetch_user("user-123").await.unwrap();
///     println!("User: {}", user.name);
/// }
/// ```
pub async fn fetch_user(id: &str) -> Result<User, FetchError> {
    // Simulate async I/O
    sleep(Duration::from_millis(100)).await;
    
    // Fetch from repository
    let user = USER_REPOSITORY
        .find_by_id(&UserId::new(id))
        .await?
        .ok_or_else(|| FetchError::NotFound(id.to_string()))?;
    
    Ok(user)
}

/// Concurrent operations with join
pub async fn fetch_user_with_orders(
    user_id: &UserId,
) -> Result<(User, Vec<Order>), FetchError> {
    // Execute both operations concurrently
    let (user_result, orders_result) = tokio::join!(
        fetch_user_from_db(user_id),
        fetch_orders_for_user(user_id)
    );
    
    Ok((user_result?, orders_result?))
}

/// Parallel processing with try_join
pub async fn fetch_multiple_users(
    ids: Vec<UserId>,
) -> Result<Vec<User>, FetchError> {
    let futures: Vec<_> = ids.into_iter()
        .map(|id| fetch_user_from_db(&id))
        .collect();
    
    // All must succeed or return first error
    tokio::try_join_all(futures).await
}

/// Sequential async operations
pub async fn create_user_workflow(
    email: Email,
    name: String,
) -> Result<UserId, WorkflowError> {
    // Step 1: Validate email
    validate_email(&email).await?;
    
    // Step 2: Create user
    let user_id = create_user_in_db(email.clone(), name.clone()).await?;
    
    // Step 3: Send welcome email
    send_welcome_email(&email, &name).await?;
    
    // Step 4: Create initial profile
    create_user_profile(&user_id).await?;
    
    Ok(user_id)
}

/// Timeout handling
pub async fn fetch_with_timeout(
    id: &UserId,
    timeout: Duration,
) -> Result<User, FetchError> {
    tokio::time::timeout(timeout, fetch_user_from_db(id))
        .await
        .map_err(|_| FetchError::Timeout)?
}

/// Retry logic
pub async fn fetch_with_retry(
    id: &UserId,
    max_retries: u32,
) -> Result<User, FetchError> {
    let mut retries = 0;
    
    loop {
        match fetch_user_from_db(id).await {
            Ok(user) => return Ok(user),
            Err(e) if retries < max_retries => {
                retries += 1;
                sleep(Duration::from_millis(100 * retries as u64)).await;
            }
            Err(e) => return Err(e),
        }
    }
}
```

### B. Async Traits

```rust
use async_trait::async_trait;

/// Service trait for user operations
#[async_trait]
pub trait UserService: Send + Sync {
    /// Registers a new user
    async fn register_user(&self, email: Email, name: String) 
        -> Result<UserId, ServiceError>;
    
    /// Authenticates a user
    async fn authenticate(&self, email: &Email, password: &str) 
        -> Result<AuthToken, ServiceError>;
    
    /// Updates user profile
    async fn update_profile(&self, id: &UserId, updates: ProfileUpdates) 
        -> Result<(), ServiceError>;
}

/// Implementation
pub struct DefaultUserService<R: UserRepository> {
    repository: Arc<R>,
}

#[async_trait]
impl<R: UserRepository> UserService for DefaultUserService<R> {
    async fn register_user(&self, email: Email, name: String) 
        -> Result<UserId, ServiceError> 
    {
        // Implementation
        todo!()
    }
    
    async fn authenticate(&self, email: &Email, password: &str) 
        -> Result<AuthToken, ServiceError> 
    {
        // Implementation
        todo!()
    }
    
    async fn update_profile(&self, id: &UserId, updates: ProfileUpdates) 
        -> Result<(), ServiceError> 
    {
        // Implementation
        todo!()
    }
}
```

---

## 10. Error Handling (MANDATORY)

### A. Result and Option

```rust
use thiserror::Error;

/// Application errors using thiserror
#[derive(Debug, Error)]
pub enum AppError {
    #[error("User not found: {0}")]
    UserNotFound(UserId),
    
    #[error("Invalid email: {0}")]
    InvalidEmail(String),
    
    #[error("Database error: {0}")]
    DatabaseError(#[from] sqlx::Error),
    
    #[error("Validation error: {0}")]
    ValidationError(String),
    
    #[error("Authentication failed")]
    AuthenticationFailed,
}

/// Using Result type
pub fn parse_user_id(input: &str) -> Result<UserId, AppError> {
    if input.is_empty() {
        return Err(AppError::ValidationError("Empty input".to_string()));
    }
    
    Ok(UserId::new(input))
}

/// Using Option type
pub fn find_user_by_email(email: &Email) -> Option<User> {
    // Returns Some(user) if found, None otherwise
    USERS.get(email).cloned()
}

/// Converting Option to Result
pub fn get_user_or_error(email: &Email) -> Result<User, AppError> {
    find_user_by_email(email)
        .ok_or_else(|| AppError::UserNotFound(UserId::new("unknown")))
}

/// Using ? operator for error propagation
pub fn complex_operation(id: &UserId) -> Result<ProcessedData, AppError> {
    let user = fetch_user(id)?;
    let validated = validate_user(&user)?;
    let processed = process_data(&validated)?;
    Ok(processed)
}

/// Combining Results
pub fn validate_and_save(user: User) -> Result<(), AppError> {
    validate_email(&user.email)
        .and_then(|_| validate_name(&user.name))
        .and_then(|_| save_user(&user))
}

/// Using anyhow for application-level errors
use anyhow::{Context, Result as AnyhowResult};

pub fn read_config() -> AnyhowResult<Config> {
    let contents = std::fs::read_to_string("config.toml")
        .context("Failed to read config file")?;
    
    let config: Config = toml::from_str(&contents)
        .context("Failed to parse config")?;
    
    Ok(config)
}
```

### B. Custom Error Types

```rust
/// Domain-specific error type
#[derive(Debug, Error)]
pub enum DomainError {
    #[error("Business rule violation: {0}")]
    BusinessRuleViolation(String),
    
    #[error("Invalid state transition from {from} to {to}")]
    InvalidStateTransition { from: String, to: String },
    
    #[error("Insufficient permissions for user {user_id}")]
    InsufficientPermissions { user_id: UserId },
}

/// Propagating different error types
pub fn handle_request(req: Request) -> Result<Response, Box<dyn std::error::Error>> {
    let user = fetch_user(&req.user_id)?;
    let validated = validate_request(&req)?;
    let result = process_request(validated)?;
    Ok(Response::from(result))
}
```

---

## 11. Testing (MANDATORY)

### A. Unit Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_user_creation() {
        let email = Email::new("test@example.com").unwrap();
        let user = User::new(email, "Test User".to_string());
        
        assert_eq!(user.name, "Test User");
        assert_eq!(user.email.as_str(), "test@example.com");
    }
    
    #[test]
    fn test_email_validation() {
        assert!(Email::new("valid@example.com").is_ok());
        assert!(Email::new("invalid").is_err());
        assert!(Email::new("").is_err());
    }
    
    #[test]
    #[should_panic(expected = "Empty name")]
    fn test_empty_name_panics() {
        User::new(
            Email::new("test@example.com").unwrap(),
            "".to_string(),
        );
    }
    
    /// Property-based testing
    #[test]
    fn test_user_id_roundtrip() {
        for i in 0..100 {
            let id = UserId::new(format!("user-{}", i));
            let serialized = serde_json::to_string(&id).unwrap();
            let deserialized: UserId = serde_json::from_str(&serialized).unwrap();
            assert_eq!(id, deserialized);
        }
    }
}
```

### B. Async Tests

```rust
#[cfg(test)]
mod async_tests {
    use super::*;
    
    #[tokio::test]
    async fn test_async_user_creation() {
        let repo = Arc::new(InMemoryUserRepository::new());
        let handler = CreateUserHandler::new(repo);
        
        let command = CreateUserCommand {
            email: Email::new("test@example.com").unwrap(),
            name: "Test User".to_string(),
        };
        
        let result = handler.handle(command).await;
        assert!(result.is_ok());
    }
    
    #[tokio::test]
    async fn test_concurrent_operations() {
        let repo = Arc::new(InMemoryUserRepository::new());
        
        // Create 100 users concurrently
        let handles: Vec<_> = (0..100)
            .map(|i| {
                let repo = repo.clone();
                tokio::spawn(async move {
                    let user = User {
                        id: UserId::new(format!("user-{}", i)),
                        email: Email::new(format!("user{}@example.com", i)).unwrap(),
                        name: format!("User {}", i),
                    };
                    repo.save(&user).await
                })
            })
            .collect();
        
        // Wait for all tasks
        for handle in handles {
            handle.await.unwrap().unwrap();
        }
        
        // Verify
        let users = repo.list(1000, 0).await.unwrap();
        assert_eq!(users.len(), 100);
    }
}
```

### C. Integration Tests

```rust
// tests/integration_test.rs

use myapp::*;

#[tokio::test]
async fn test_full_user_workflow() {
    // Setup
    let repo = Arc::new(InMemoryUserRepository::new());
    let create_handler = CreateUserHandler::new(repo.clone());
    let get_handler = GetUserHandler::new(repo.clone());
    
    // Create user
    let create_cmd = CreateUserCommand {
        email: Email::new("integration@test.com").unwrap(),
        name: "Integration Test".to_string(),
    };
    
    let user_id = create_handler.handle(create_cmd).await.unwrap();
    
    // Retrieve user
    let get_query = GetUserQuery { user_id: user_id.clone() };
    let user = get_handler.handle(get_query).await.unwrap();
    
    // Verify
    assert_eq!(user.id, user_id);
    assert_eq!(user.name, "Integration Test");
}
```

### D. Mock Testing

```rust
use mockall::predicate::*;
use mockall::*;

#[automock]
#[async_trait]
pub trait EmailService: Send + Sync {
    async fn send_email(&self, to: &Email, subject: &str, body: &str) 
        -> Result<(), EmailError>;
}

#[cfg(test)]
mod mock_tests {
    use super::*;
    
    #[tokio::test]
    async fn test_with_mock_email_service() {
        let mut mock_email = MockEmailService::new();
        
        mock_email
            .expect_send_email()
            .with(
                eq(Email::new("test@example.com").unwrap()),
                eq("Welcome"),
                always(),
            )
            .times(1)
            .returning(|_, _, _| Ok(()));
        
        // Use mock in test
        let result = mock_email
            .send_email(
                &Email::new("test@example.com").unwrap(),
                "Welcome",
                "Welcome message",
            )
            .await;
        
        assert!(result.is_ok());
    }
}
```

---

## 12. Documentation (MANDATORY)

### A. rustdoc Comments

```rust
/// A user in the system.
///
/// This struct represents a registered user with validated email
/// and basic profile information.
///
/// # Examples
///
/// ```
/// use myapp::{User, UserId, Email};
///
/// let user = User {
///     id: UserId::new("user-123"),
///     email: Email::new("user@example.com").unwrap(),
///     name: "John Doe".to_string(),
/// };
///
/// assert_eq!(user.name, "John Doe");
/// ```
///
/// # Safety
///
/// All fields are validated at construction time. Email addresses
/// are guaranteed to be in valid format.
#[derive(Debug, Clone, PartialEq)]
pub struct User {
    /// Unique identifier for the user
    pub id: UserId,
    
    /// Validated email address
    pub email: Email,
    
    /// User's display name (non-empty)
    pub name: String,
}

impl User {
    /// Creates a new user with validated fields.
    ///
    /// # Arguments
    ///
    /// * `email` - A validated email address
    /// * `name` - User's display name (must not be empty)
    ///
    /// # Returns
    ///
    /// Returns `Ok(User)` if validation passes.
    ///
    /// # Errors
    ///
    /// Returns `UserError::EmptyName` if name is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use myapp::{User, Email};
    ///
    /// let email = Email::new("test@example.com").unwrap();
    /// let user = User::create(email, "John Doe").unwrap();
    /// ```
    ///
    /// # Panics
    ///
    /// This function does not panic.
    pub fn create(email: Email, name: String) -> Result<Self, UserError> {
        if name.is_empty() {
            return Err(UserError::EmptyName);
        }
        
        Ok(Self {
            id: UserId::new(uuid::Uuid::new_v4().to_string()),
            email,
            name,
        })
    }
}
```

### B. Module Documentation

```rust
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
//! - Domain entities are in `entities/`
//! - Value objects are in `value_objects/`
//! - Repository ports are in `../application/ports/`
//!
//! # Examples
//!
//! ```
//! use myapp::domain::user::*;
//!
//! let email = Email::new("user@example.com")?;
//! let user = User::create(email, "John Doe")?;
//! ```

pub mod entities;
pub mod value_objects;
pub mod services;
pub mod errors;
```

### C. Generating Documentation

```bash
# Generate documentation
cargo doc --no-deps --open

# Generate with private items
cargo doc --no-deps --document-private-items

# Check documentation coverage
cargo doc --no-deps 2>&1 | grep warning

# Test documentation examples
cargo test --doc
```

---

## 13. Deployment Checklist

### Pre-Production Validation

#### Compilation (MANDATORY)
- [ ] **Builds successfully**: `cargo build --release` passes
- [ ] **No warnings**: `cargo build --release --quiet` produces no output
- [ ] **All features compile**: `cargo build --all-features` passes
- [ ] **Cross-compilation tested**: Tested on target platforms

#### Linting (MANDATORY)
- [ ] **Clippy passes**: `cargo clippy -- -D warnings` returns exit code 0
- [ ] **No clippy warnings**: All suggestions addressed or explicitly allowed
- [ ] **Formatting correct**: `cargo fmt -- --check` passes

#### Testing (MANDATORY)
- [ ] **All tests pass**: `cargo test` returns exit code 0
- [ ] **Integration tests pass**: `cargo test --test '*'` passes
- [ ] **Doc tests pass**: `cargo test --doc` passes
- [ ] **Coverage ≥ 80%**: `cargo tarpaulin` shows adequate coverage
- [ ] **Benchmarks run**: `cargo bench` completes successfully

#### Documentation (MANDATORY)
- [ ] **All public APIs documented**: rustdoc comments on all `pub` items
- [ ] **Documentation builds**: `cargo doc --no-deps` succeeds
- [ ] **Examples in docs**: Code examples in rustdoc comments
- [ ] **Doc tests pass**: Examples in docs execute correctly
- [ ] **README up to date**: Installation and usage instructions current

#### Code Quality
- [ ] **No unwrap/expect in lib**: Production code uses proper error handling
- [ ] **No panic in lib**: Library code doesn't panic
- [ ] **Unsafe justified**: All `unsafe` blocks have SAFETY comments
- [ ] **Dependencies audited**: `cargo audit` passes
- [ ] **Minimal dependencies**: Only necessary crates included
- [ ] **Pure Rust crates**: No FFI dependencies unless necessary

#### Architecture
- [ ] **Hexagonal architecture**: Clear separation of layers
- [ ] **CQRS implemented**: Commands and queries separated
- [ ] **Traits for abstractions**: Ports defined as traits
- [ ] **Newtype pattern used**: Domain primitives wrapped
- [ ] **Enums for states**: State machines use enums

#### Performance
- [ ] **Release optimizations**: Profile.release configured
- [ ] **No unnecessary clones**: `clone()` used judiciously
- [ ] **Async where appropriate**: I/O operations use async/await
- [ ] **Benchmarks added**: Performance-critical code benchmarked

---

## 14. Complete Example

```rust
//! Order management module demonstrating hexagonal architecture.
//!
//! This module shows a complete implementation of CQRS, DDD, and
//! functional programming patterns in Rust.

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use thiserror::Error;
use uuid::Uuid;

// ============================================================================
// Domain - Value Objects (Newtype Pattern)
// ============================================================================

/// Order ID newtype
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct OrderId(Uuid);

impl OrderId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
    
    pub fn from_uuid(id: Uuid) -> Self {
        Self(id)
    }
}

impl std::fmt::Display for OrderId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Product ID newtype
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ProductId(String);

impl ProductId {
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }
}

/// Money value object
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct Money {
    amount: f64,
    currency: Currency,
}

impl Money {
    pub fn new(amount: f64, currency: Currency) -> Result<Self, MoneyError> {
        if amount < 0.0 {
            return Err(MoneyError::NegativeAmount);
        }
        Ok(Self { amount, currency })
    }
    
    pub fn amount(&self) -> f64 {
        self.amount
    }
    
    pub fn add(&self, other: &Money) -> Result<Money, MoneyError> {
        if self.currency != other.currency {
            return Err(MoneyError::CurrencyMismatch);
        }
        Ok(Money {
            amount: self.amount + other.amount,
            currency: self.currency,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Currency {
    USD,
    EUR,
    GBP,
}

#[derive(Debug, Error)]
pub enum MoneyError {
    #[error("Amount cannot be negative")]
    NegativeAmount,
    
    #[error("Currency mismatch")]
    CurrencyMismatch,
}

// ============================================================================
// Domain - Entities
// ============================================================================

/// Order aggregate root
#[derive(Debug, Clone, PartialEq)]
pub struct Order {
    pub id: OrderId,
    pub customer_id: String,
    pub items: Vec<OrderItem>,
    pub state: OrderState,
    pub created_at: DateTime<Utc>,
}

impl Order {
    /// Creates a new pending order
    pub fn create(customer_id: String, items: Vec<OrderItem>) -> Result<Self, OrderError> {
        if items.is_empty() {
            return Err(OrderError::EmptyOrder);
        }
        
        Ok(Self {
            id: OrderId::new(),
            customer_id,
            items,
            state: OrderState::Pending,
            created_at: Utc::now(),
        })
    }
    
    /// Calculates total order value
    pub fn total(&self) -> Result<Money, OrderError> {
        self.items
            .iter()
            .try_fold(
                Money::new(0.0, Currency::USD).unwrap(),
                |acc, item| {
                    acc.add(&item.subtotal())
                        .map_err(|_| OrderError::CalculationError)
                },
            )
    }
    
    /// Confirms the order
    pub fn confirm(mut self) -> Result<Self, OrderError> {
        self.state = self.state.confirm()?;
        Ok(self)
    }
    
    /// Ships the order
    pub fn ship(mut self, tracking: String) -> Result<Self, OrderError> {
        self.state = self.state.ship(tracking)?;
        Ok(self)
    }
}

/// Order item
#[derive(Debug, Clone, PartialEq)]
pub struct OrderItem {
    pub product_id: ProductId,
    pub quantity: u32,
    pub unit_price: Money,
}

impl OrderItem {
    pub fn subtotal(&self) -> Money {
        Money::new(
            self.unit_price.amount() * self.quantity as f64,
            Currency::USD,
        )
        .unwrap()
    }
}

// ============================================================================
// Domain - State Machine (Enums)
// ============================================================================

/// Order state machine
#[derive(Debug, Clone, PartialEq)]
pub enum OrderState {
    Pending,
    Confirmed { confirmed_at: DateTime<Utc> },
    Shipped { tracking_number: String, shipped_at: DateTime<Utc> },
    Delivered { delivered_at: DateTime<Utc> },
    Cancelled { reason: String },
}

impl OrderState {
    pub fn confirm(self) -> Result<Self, OrderError> {
        match self {
            Self::Pending => Ok(Self::Confirmed {
                confirmed_at: Utc::now(),
            }),
            _ => Err(OrderError::InvalidStateTransition {
                from: self.name(),
                to: "Confirmed",
            }),
        }
    }
    
    pub fn ship(self, tracking: String) -> Result<Self, OrderError> {
        match self {
            Self::Confirmed { .. } => Ok(Self::Shipped {
                tracking_number: tracking,
                shipped_at: Utc::now(),
            }),
            _ => Err(OrderError::InvalidStateTransition {
                from: self.name(),
                to: "Shipped",
            }),
        }
    }
    
    pub fn name(&self) -> &'static str {
        match self {
            Self::Pending => "Pending",
            Self::Confirmed { .. } => "Confirmed",
            Self::Shipped { .. } => "Shipped",
            Self::Delivered { .. } => "Delivered",
            Self::Cancelled { .. } => "Cancelled",
        }
    }
}

// ============================================================================
// Domain - Errors
// ============================================================================

#[derive(Debug, Error)]
pub enum OrderError {
    #[error("Order cannot be empty")]
    EmptyOrder,
    
    #[error("Invalid state transition from {from} to {to}")]
    InvalidStateTransition { from: &'static str, to: &'static str },
    
    #[error("Calculation error")]
    CalculationError,
    
    #[error("Order not found: {0}")]
    NotFound(OrderId),
}

// ============================================================================
// Application - Ports (Traits)
// ============================================================================

/// Repository port for order persistence
#[async_trait]
pub trait OrderRepository: Send + Sync {
    async fn find_by_id(&self, id: &OrderId) -> Result<Option<Order>, RepositoryError>;
    async fn save(&self, order: &Order) -> Result<(), RepositoryError>;
    async fn list_by_customer(&self, customer_id: &str) -> Result<Vec<Order>, RepositoryError>;
}

#[derive(Debug, Error)]
pub enum RepositoryError {
    #[error("Database error: {0}")]
    DatabaseError(String),
}

// ============================================================================
// Application - Commands (CQRS Write Side)
// ============================================================================

/// Command: Create order
#[derive(Debug, Clone)]
pub struct CreateOrderCommand {
    pub customer_id: String,
    pub items: Vec<CreateOrderItem>,
}

#[derive(Debug, Clone)]
pub struct CreateOrderItem {
    pub product_id: ProductId,
    pub quantity: u32,
    pub unit_price: Money,
}

/// Command handler
pub struct CreateOrderHandler<R: OrderRepository> {
    repository: Arc<R>,
}

impl<R: OrderRepository> CreateOrderHandler<R> {
    pub fn new(repository: Arc<R>) -> Self {
        Self { repository }
    }
    
    /// Handles order creation
    pub async fn handle(&self, cmd: CreateOrderCommand) -> Result<OrderId, OrderError> {
        // Convert command to domain items
        let items: Vec<OrderItem> = cmd
            .items
            .into_iter()
            .map(|item| OrderItem {
                product_id: item.product_id,
                quantity: item.quantity,
                unit_price: item.unit_price,
            })
            .collect();
        
        // Create order using domain logic
        let order = Order::create(cmd.customer_id, items)?;
        
        // Persist
        self.repository
            .save(&order)
            .await
            .map_err(|_| OrderError::CalculationError)?;
        
        Ok(order.id)
    }
}

// ============================================================================
// Application - Queries (CQRS Read Side)
// ============================================================================

/// Query: Get order by ID
#[derive(Debug, Clone)]
pub struct GetOrderQuery {
    pub order_id: OrderId,
}

/// Query handler
pub struct GetOrderHandler<R: OrderRepository> {
    repository: Arc<R>,
}

impl<R: OrderRepository> GetOrderHandler<R> {
    pub fn new(repository: Arc<R>) -> Self {
        Self { repository }
    }
    
    pub async fn handle(&self, query: GetOrderQuery) -> Result<Order, OrderError> {
        self.repository
            .find_by_id(&query.order_id)
            .await
            .map_err(|_| OrderError::CalculationError)?
            .ok_or_else(|| OrderError::NotFound(query.order_id))
    }
}

// ============================================================================
// Infrastructure - In-Memory Adapter
// ============================================================================

use std::collections::HashMap;
use tokio::sync::RwLock;

pub struct InMemoryOrderRepository {
    orders: Arc<RwLock<HashMap<OrderId, Order>>>,
}

impl InMemoryOrderRepository {
    pub fn new() -> Self {
        Self {
            orders: Arc::new(RwLock::new(HashMap::new())),
        }
    }
}

#[async_trait]
impl OrderRepository for InMemoryOrderRepository {
    async fn find_by_id(&self, id: &OrderId) -> Result<Option<Order>, RepositoryError> {
        let orders = self.orders.read().await;
        Ok(orders.get(id).cloned())
    }
    
    async fn save(&self, order: &Order) -> Result<(), RepositoryError> {
        let mut orders = self.orders.write().await;
        orders.insert(order.id.clone(), order.clone());
        Ok(())
    }
    
    async fn list_by_customer(&self, customer_id: &str) -> Result<Vec<Order>, RepositoryError> {
        let orders = self.orders.read().await;
        Ok(orders
            .values()
            .filter(|o| o.customer_id == customer_id)
            .cloned()
            .collect())
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_order_creation() {
        let items = vec![OrderItem {
            product_id: ProductId::new("prod-1"),
            quantity: 2,
            unit_price: Money::new(10.0, Currency::USD).unwrap(),
        }];
        
        let order = Order::create("customer-1".to_string(), items).unwrap();
        
        assert_eq!(order.state, OrderState::Pending);
        assert_eq!(order.items.len(), 1);
    }
    
    #[test]
    fn test_order_total() {
        let items = vec![
            OrderItem {
                product_id: ProductId::new("prod-1"),
                quantity: 2,
                unit_price: Money::new(10.0, Currency::USD).unwrap(),
            },
            OrderItem {
                product_id: ProductId::new("prod-2"),
                quantity: 1,
                unit_price: Money::new(15.0, Currency::USD).unwrap(),
            },
        ];
        
        let order = Order::create("customer-1".to_string(), items).unwrap();
        let total = order.total().unwrap();
        
        assert_eq!(total.amount(), 35.0); // (2 * 10) + (1 * 15)
    }
    
    #[test]
    fn test_state_transitions() {
        let items = vec![OrderItem {
            product_id: ProductId::new("prod-1"),
            quantity: 1,
            unit_price: Money::new(10.0, Currency::USD).unwrap(),
        }];
        
        let order = Order::create("customer-1".to_string(), items).unwrap();
        
        // Confirm
        let order = order.confirm().unwrap();
        assert!(matches!(order.state, OrderState::Confirmed { .. }));
        
        // Ship
        let order = order.ship("TRACK123".to_string()).unwrap();
        assert!(matches!(order.state, OrderState::Shipped { .. }));
    }
    
    #[tokio::test]
    async fn test_create_order_handler() {
        let repo = Arc::new(InMemoryOrderRepository::new());
        let handler = CreateOrderHandler::new(repo);
        
        let cmd = CreateOrderCommand {
            customer_id: "customer-1".to_string(),
            items: vec![CreateOrderItem {
                product_id: ProductId::new("prod-1"),
                quantity: 2,
                unit_price: Money::new(10.0, Currency::USD).unwrap(),
            }],
        };
        
        let result = handler.handle(cmd).await;
        assert!(result.is_ok());
    }
}
```

---

## 15. Why This Configuration Works

1. **Hexagonal Architecture**: Clear separation of concerns, testable in isolation, easy to swap adapters.

2. **CQRS**: Commands and queries separated, optimized independently, scalable architecture.

3. **Newtype Pattern**: Prevents primitive obsession, catches errors at compile time, self-documenting code.

4. **Enums for States**: Type-safe state machines, exhaustive matching, impossible states prevented.

5. **Traits over Inheritance**: Flexible composition, dependency inversion, easy mocking for tests.

6. **Functional Programming**: Immutable by default, predictable behavior, easier reasoning, less bugs.

7. **Result & Option**: Explicit error handling, no null pointer exceptions, compiler-enforced handling.

8. **Async/Await**: Efficient I/O, scalable concurrency, readable asynchronous code.

9. **Pure Rust Crates**: Portable across platforms, no FFI complexity, easier deployment.

10. **rustdoc**: Documentation stays in sync with code, runnable examples, auto-generated docs.

11. **Agent Verification**: Ensures all code compiles and tests pass, eliminates broken code.

---

## 16. Quick Reference

### Common Commands

```bash
# Build
cargo build
cargo build --release

# Test
cargo test
cargo test --no-fail-fast
cargo test -- --nocapture

# Lint & Format
cargo fmt --check
cargo fmt
cargo clippy
cargo clippy -- -D warnings

# Run
cargo run
cargo run --release

# Documentation
cargo doc --open
cargo doc --no-deps

# Dependencies
cargo add <crate>
cargo update
cargo tree

# Check (fast compilation check)
cargo check

# Coverage
cargo tarpaulin --out Html
```

### Common Patterns Cheat Sheet

```rust
// Result handling
let value = result?;                    // Early return on error
let value = result.unwrap_or(default);  // Default on error
let value = result.ok();                // Convert to Option

// Option handling
let value = option?;                    // Early return on None
let value = option.unwrap_or(default);  // Default on None
let value = option.ok_or(Error::new())?; // Convert to Result

// Iterators
items.iter().map(|x| x * 2).collect::<Vec<_>>();
items.iter().filter(|x| x > &0).sum::<i32>();
items.iter().find(|x| x.id == target_id);

// Error creation (thiserror)
#[derive(Error, Debug)]
enum MyError {
    #[error("not found: {0}")]
    NotFound(String),
    #[error("validation failed")]
    Validation,
}
```

### Project Structure

```
my_project/
├── Cargo.toml
├── src/
│   ├── lib.rs           # Library root
│   ├── main.rs          # Binary entry
│   ├── domain/          # Domain models
│   ├── ports/           # Traits/interfaces
│   └── adapters/        # Implementations
└── tests/
    └── integration_test.rs
```

### Cargo.toml Essentials

```toml
[package]
name = "my_project"
version = "0.1.0"
edition = "2021"

[dependencies]
tokio = { version = "1", features = ["full"] }
serde = { version = "1", features = ["derive"] }
thiserror = "1"

[dev-dependencies]
mockall = "0.11"

[lints.rust]
unsafe_code = "forbid"

[lints.clippy]
all = "warn"
```

---

## References

- [The Rust Programming Language](https://doc.rust-lang.org/book/)
- [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/)
- [The Cargo Book](https://doc.rust-lang.org/cargo/)
- [rustdoc Book](https://doc.rust-lang.org/rustdoc/)
- [Rust Design Patterns](https://rust-unofficial.github.io/patterns/)
- [async-trait](https://docs.rs/async-trait/)
- [tokio](https://tokio.rs/)
- [thiserror](https://docs.rs/thiserror/)

---

**Last Updated:** 2026-01-17
**Version:** 1.0
**Maintainer:** Development Team

---

**End of Rust Development Guidelines**
