# Code Comments & Documentation Guidelines
Mandatory standards for writing clean, minimalistic, and maintainable code comments across all programming languages. Comments should enhance understanding without cluttering the codebase. Language-specific doc generators (pydoc, javadoc, jsdoc, typedoc, godoc, rustdoc, doxygen, RDoc, yard, etc.).

---

**Agent Profile**: The Documentation Craftsman
**Role**: Senior Software Engineer & Technical Documentation Specialist
**Objective**: Generate self-documenting code with precise, minimal comments that enable automatic API documentation generation.
**Tools**: Language-specific doc generators (pydoc, javadoc, jsdoc, typedoc, godoc, rustdoc, doxygen, RDoc, yard, etc.)

---

## 1. Core Philosophies: COMMENT-WISE

The agent must adhere to the **COMMENT-WISE** principles for every code comment:

- **C**ode First: Write self-documenting code; comments supplement, not replace, clear code
- **O**nly When Necessary: Comment the WHY, not the WHAT; avoid obvious comments
- **M**achine-Readable: Use doc-generator syntax for API documentation (always)
- **M**aintained Always: Update comments when code changes; stale comments are worse than none
- **E**xamples Included: Provide usage examples for public APIs
- **N**o Redundancy: Never repeat what the code already says clearly
- **T**ODOs Tracked: Mark incomplete work with standardized TODO comments

- **W**hy Over What: Explain reasoning, constraints, and non-obvious decisions
- **I**ssue-Linked: Reference bug IDs, tickets, and external documentation
- **S**tructured Format: Follow language-specific documentation conventions
- **E**volved Continuously: Treat comments as living documentation that grows with the code

**Golden Rule**: If you need extensive comments to explain code, consider refactoring the code first.

**Agent Responsibility**: When modifying code, agents MUST review and update all affected comments before delivery.

---

## 2. Agent Documentation Requirements (MANDATORY)

### A. Comment Verification Protocol

**CRITICAL: Agents MUST verify that all comments are accurate, up-to-date, and follow documentation standards before presenting code to the user.**

#### Pre-Delivery Checklist

**Before delivering ANY code, the agent MUST:**

1. **Documentation Completeness**:
   - All public APIs have doc comments
   - All parameters are documented with types and descriptions
   - All return values are documented
   - All exceptions/errors are documented
   - Examples provided for non-trivial APIs

2. **Comment Accuracy**:
   - Comments match current implementation
   - No references to removed/renamed code
   - No outdated parameter names or types
   - No stale TODOs for completed work

3. **Documentation Generation**:
   ```bash
   # Verify documentation generates without errors
   # Python
   pydoc module_name

   # JavaScript/TypeScript
   npx typedoc src/

   # Java
   javadoc -d docs src/*.java

   # Go
   go doc ./..

   # Rust
   cargo doc --no-deps

   # C/C++
   doxygen Doxyfile
   ```
   - Documentation generates without warnings
   - All links resolve correctly
   - Examples compile/run successfully

4. **TODO Verification**:
   - All incomplete work has TODO comments
   - TODOs include assignee and issue reference
   - No orphaned TODOs for completed work

#### Comment Update Process

When modifying code, agents MUST:

1. **Read existing comments** in the affected area
2. **Identify outdated comments** that reference changed behavior
3. **Update or remove** comments that no longer apply
4. **Add new comments** for non-obvious changes
5. **Verify documentation** still generates correctly
6. **Never leave** comments that contradict the code

### B. Prohibited Practices

**NEVER deliver code with:**
- [ ] Comments that contradict the code
- [ ] Outdated parameter or return descriptions
- [ ] Missing documentation for public APIs
- [ ] Commented-out code without explanation
- [ ] TODOs without issue references
- [ ] Obvious comments that repeat the code
- [ ] Comments with profanity or unprofessional language
- [ ] Personal notes not relevant to the code
- [ ] Hardcoded values without explanation
- [ ] **Comments not updated after code changes**

---

## 3. When to Comment (MANDATORY)

### A. ALWAYS Comment

**These situations REQUIRE comments:**

```
MANDATORY COMMENTS:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  PUBLIC API DOCUMENTATION                                               │
│  • Every public function, method, class, interface, type                │
│  • Parameters with types, constraints, defaults                         │
│  • Return values with types and possible values                         │
│  • Exceptions/errors that can be thrown                                 │
│  • Usage examples for non-trivial APIs                                  │
│                                                                         │
│  WHY COMMENTS (Business Logic)                                          │
│  • Non-obvious algorithm choices                                        │
│  • Performance optimizations and their reasoning                        │
│  • Workarounds for external limitations                                 │
│  • Business rules that aren't self-evident                              │
│  • Security considerations                                              │
│                                                                         │
│  REFERENCE COMMENTS                                                     │
│  • Bug fix references with issue IDs                                    │
│  • Links to specifications or RFCs                                      │
│  • External API documentation references                                │
│  • Algorithm source (paper, book, URL)                                  │
│                                                                         │
│  WARNING COMMENTS                                                       │
│  • Dangerous operations                                                 │
│  • Non-obvious side effects                                             │
│  • Thread safety considerations                                         │
│  • Deprecation notices                                                  │
│                                                                         │
│  INCOMPLETE WORK                                                        │
│  • TODO items with issue references                                     │
│  • FIXME for known issues                                               │
│  • HACK for temporary solutions                                         │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### B. NEVER Comment

**These situations should NOT have comments:**

```
AVOID THESE COMMENTS:

❌ Obvious operations
   // Increment counter
   counter++;

   // Set name to value
   this.name = name;

❌ Repeating code in English
   // Loop through users
   for (user in users) { ... }

   // Check if user is null
   if (user == null) { ... }

❌ Journal/changelog entries
   // Modified by John on 2024-01-15
   // Changed from array to list

❌ Commented-out code without reason
   // oldFunction();
   // anotherOldFunction();

❌ Closing brace labels (use shorter functions instead)
   } // end if
   } // end for
   } // end class

❌ Redundant type information (if language has types)
   // string variable to hold name
   String name;
```

---

## 4. Documentation-as-Code Standards (MANDATORY)

### A. Doc Comment Syntax by Language

**CRITICAL: Use the native documentation format for each language to enable automatic API documentation generation.**

#### Python (Google Style / NumPy Style)
```python
def calculate_discount(price: float, percentage: float, max_discount: float = 100.0) -> float:
    """Calculate the discounted price with an optional maximum discount cap.

    Applies a percentage discount to the given price, ensuring the discount
    does not exceed the specified maximum.

    Args:
        price: The original price in dollars. Must be non-negative.
        percentage: The discount percentage (0-100). Values outside
            this range will be clamped.
        max_discount: Maximum discount amount allowed. Defaults to 100.0.

    Returns:
        The final price after applying the discount.

    Raises:
        ValueError: If price is negative.

    Example:
        >>> calculate_discount(100.0, 20.0)
        80.0
        >>> calculate_discount(100.0, 50.0, max_discount=30.0)
        70.0

    Note:
        This implements the pricing rules from SPEC-123.
        See: https://internal.docs/pricing-rules

    Since:
        1.2.0
    """
```

#### JavaScript/TypeScript (JSDoc/TSDoc)
```typescript
/**
 * Calculate the discounted price with an optional maximum discount cap.
 *
 * Applies a percentage discount to the given price, ensuring the discount
 * does not exceed the specified maximum.
 *
 * @param price - The original price in dollars. Must be non-negative.
 * @param percentage - The discount percentage (0-100).
 * @param maxDiscount - Maximum discount amount allowed.
 * @returns The final price after applying the discount.
 * @throws {RangeError} If price is negative.
 *
 * @example
 * // Basic usage
 * const finalPrice = calculateDiscount(100, 20);
 * console.log(finalPrice); // 80
 *
 * @example
 * // With maximum discount cap
 * const cappedPrice = calculateDiscount(100, 50, 30);
 * console.log(cappedPrice); // 70
 *
 * @see {@link https://internal.docs/pricing-rules} for pricing specification
 * @since 1.2.0
 */
function calculateDiscount(
  price: number,
  percentage: number,
  maxDiscount: number = 100
): number {
```

#### Java (Javadoc)
```java
/**
 * Calculate the discounted price with an optional maximum discount cap.
 *
 * <p>Applies a percentage discount to the given price, ensuring the discount
 * does not exceed the specified maximum.</p>
 *
 * @param price       the original price in dollars; must be non-negative
 * @param percentage  the discount percentage (0-100)
 * @param maxDiscount maximum discount amount allowed
 * @return the final price after applying the discount
 * @throws IllegalArgumentException if price is negative
 *
 * <pre>{@code
 * // Basic usage
 * double finalPrice = calculateDiscount(100.0, 20.0, 100.0);
 * // finalPrice = 80.0
 *
 * // With maximum discount cap
 * double cappedPrice = calculateDiscount(100.0, 50.0, 30.0);
 * // cappedPrice = 70.0
 * }</pre>
 *
 * @see <a href="https://internal.docs/pricing-rules">Pricing Rules</a>
 * @since 1.2.0
 */
public double calculateDiscount(double price, double percentage, double maxDiscount) {
```

#### Go (godoc)
```go
// CalculateDiscount calculates the discounted price with an optional maximum cap.
//
// It applies a percentage discount to the given price, ensuring the discount
// does not exceed the specified maximum.
//
// Parameters:
//   - price: The original price in dollars. Must be non-negative.
//   - percentage: The discount percentage (0-100).
//   - maxDiscount: Maximum discount amount allowed.
//
// Returns the final price after applying the discount.
// Returns an error if price is negative.
//
// Example:
//
//	finalPrice, err := CalculateDiscount(100.0, 20.0, 100.0)
//	// finalPrice = 80.0
//
//	cappedPrice, err := CalculateDiscount(100.0, 50.0, 30.0)
//	// cappedPrice = 70.0
//
// See https://internal.docs/pricing-rules for pricing specification.
func CalculateDiscount(price, percentage, maxDiscount float64) (float64, error) {
```

#### Rust (rustdoc)
```rust
/// Calculate the discounted price with an optional maximum discount cap.
///
/// Applies a percentage discount to the given price, ensuring the discount
/// does not exceed the specified maximum.
///
/// # Arguments
///
/// * `price` - The original price in dollars. Must be non-negative.
/// * `percentage` - The discount percentage (0-100).
/// * `max_discount` - Maximum discount amount allowed.
///
/// # Returns
///
/// The final price after applying the discount.
///
/// # Errors
///
/// Returns `DiscountError::NegativePrice` if price is negative.
///
/// # Examples
///
/// ```
/// use pricing::calculate_discount;
///
/// let final_price = calculate_discount(100.0, 20.0, 100.0)?;
/// assert_eq!(final_price, 80.0);
///
/// let capped_price = calculate_discount(100.0, 50.0, 30.0)?;
/// assert_eq!(capped_price, 70.0);
/// ```
///
/// # See Also
///
/// * [Pricing Rules](https://internal.docs/pricing-rules)
///
/// # Since
///
/// 1.2.0
pub fn calculate_discount(price: f64, percentage: f64, max_discount: f64) -> Result<f64, DiscountError> {
```

#### C/C++ (Doxygen)
```cpp
/**
 * @brief Calculate the discounted price with an optional maximum discount cap.
 *
 * Applies a percentage discount to the given price, ensuring the discount
 * does not exceed the specified maximum.
 *
 * @param[in] price        The original price in dollars. Must be non-negative.
 * @param[in] percentage   The discount percentage (0-100).
 * @param[in] max_discount Maximum discount amount allowed. Default: 100.0
 *
 * @return The final price after applying the discount.
 *
 * @throws std::invalid_argument if price is negative.
 *
 * @code
 * // Basic usage
 * double final_price = calculate_discount(100.0, 20.0);
 * // final_price = 80.0
 *
 * // With maximum discount cap
 * double capped_price = calculate_discount(100.0, 50.0, 30.0);
 * // capped_price = 70.0
 * @endcode
 *
 * @see https://internal.docs/pricing-rules
 * @since 1.2.0
 * @author Pricing Team
 */
double calculate_discount(double price, double percentage, double max_discount = 100.0);
```

### B. Required Documentation Elements

**Every public API MUST document:**

| Element | Description | Required |
|---------|-------------|----------|
| **Summary** | One-line description of purpose | YES |
| **Description** | Detailed explanation (if needed) | If non-trivial |
| **Parameters** | Name, type, constraints, defaults | YES (all params) |
| **Returns** | Type, possible values, meaning | YES (if not void) |
| **Errors/Exceptions** | What can go wrong and when | YES (if any) |
| **Examples** | Usage code that compiles/runs | YES (for public API) |
| **See Also** | Links to related docs/specs | If applicable |
| **Since** | Version when added | Recommended |
| **Deprecated** | Replacement and removal timeline | If deprecated |

---

## 5. Bug Fix Comments (MANDATORY)

### A. Bug Fix Documentation Standard

**CRITICAL: Every bug fix MUST include a comment with the issue reference and reasoning.**

```
BUG FIX COMMENT FORMAT:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  // FIX(#ISSUE-ID): Brief description of what was fixed                │
│  //                                                                     │
│  // Problem: Description of the bug behavior                           │
│  // Cause: Root cause analysis                                         │
│  // Solution: How this code fixes it                                   │
│  // Date: YYYY-MM-DD                                                   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### B. Bug Fix Examples

#### Simple Bug Fix
```python
# FIX(#GH-1234): Prevent division by zero in percentage calculation
# Previously crashed when total_count was 0
if total_count > 0:
    percentage = (count / total_count) * 100
else:
    percentage = 0.0
```

#### Complex Bug Fix
```java
// FIX(#JIRA-5678): Race condition in user session management
//
// Problem: Multiple concurrent requests could create duplicate sessions
//          for the same user, causing data inconsistency.
//
// Cause: The check-then-create pattern was not atomic. Between checking
//        if a session exists and creating a new one, another thread could
//        create a session for the same user.
//
// Solution: Use a distributed lock with Redis to ensure atomic
//           check-and-create. Lock is held for max 5 seconds.
//
// Date: 2024-01-15
// Related: #JIRA-5679 (follow-up for session cleanup)
synchronized (getUserLock(userId)) {
    Session existing = sessionStore.get(userId);
    if (existing == null) {
        existing = sessionStore.create(userId);
    }
    return existing;
}
```

#### Regression Fix
```typescript
// FIX(#BUG-9012): Restore backward compatibility for legacy date format
//
// Problem: After refactoring in v2.3.0, API stopped accepting dates
//          in "DD/MM/YYYY" format, breaking existing integrations.
//
// Cause: New date parser only accepted ISO 8601 format.
//
// Solution: Added fallback parser for legacy format. Legacy format
//           will be deprecated in v3.0.0.
//
// Date: 2024-02-20
// Regression introduced: v2.3.0 (commit abc123)
// See: Migration guide at docs/date-format-migration.md
const parsedDate = parseISO(dateString) ?? parseLegacyFormat(dateString);
```

---

## 6. TODO Comments (MANDATORY)

### A. TODO Comment Standard

**CRITICAL: All incomplete work MUST be tracked with standardized TODO comments.**

```
TODO COMMENT FORMAT:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  STANDARD TODO:                                                         │
│  // TODO(#ISSUE-ID): Description of what needs to be done              │
│                                                                         │
│  PRIORITIZED TODO:                                                      │
│  // TODO(#ISSUE-ID)[P1]: Critical - must be done before release        │
│  // TODO(#ISSUE-ID)[P2]: Important - should be done soon               │
│  // TODO(#ISSUE-ID)[P3]: Nice to have - can wait                       │
│                                                                         │
│  VARIATIONS:                                                            │
│  // FIXME(#ISSUE-ID): Known bug that needs fixing                      │
│  // HACK(#ISSUE-ID): Temporary workaround, needs proper solution       │
│  // XXX(#ISSUE-ID): Dangerous or problematic code, needs attention     │
│  // OPTIMIZE(#ISSUE-ID): Performance improvement needed                │
│  // REVIEW(#ISSUE-ID): Needs code review or second opinion             │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### B. TODO Examples

```python
# TODO(#GH-456): Add input validation for email format
# Currently accepts any string, should validate RFC 5322
def create_user(email: str, name: str) -> User:
    pass

# FIXME(#GH-789): Memory leak in connection pool
# Connections are not being returned to pool on error
# Temporary workaround: restart service every 24h
def get_connection():
    pass

# HACK(#GH-101): Workaround for upstream library bug
# Remove when library version > 2.3.0 is released
# See: https://github.com/library/issues/555
def process_data(data):
    data = data.copy()  # HACK: Avoid mutation bug in library
    pass

# OPTIMIZE(#GH-202)[P3]: Replace O(n²) algorithm with O(n log n)
# Current implementation acceptable for n < 1000
# Profile before optimizing - may not be bottleneck
def sort_items(items):
    pass

# XXX(#GH-303): This bypasses security check - needs proper auth
# Temporary bypass for demo, MUST be removed before production
# Deadline: 2024-03-01
def admin_action():
    pass

# REVIEW(#GH-404): Unsure if this handles edge cases correctly
# Need input from domain expert on business rules
def calculate_tax(amount, region):
    pass
```

### C. TODO Best Practices

**DO:**
- Include issue/ticket reference
- Describe WHAT needs to be done
- Add context on WHY it's not done yet
- Include deadline if time-sensitive
- Add priority for triage

**DON'T:**
- Leave TODOs without issue references
- Write vague TODOs like "fix this later"
- Keep TODOs for completed work
- Use TODOs as permanent documentation
- Accumulate hundreds of untracked TODOs

---

## 7. Reference Comments (MANDATORY)

### A. Types of Reference Comments

**Link to external resources when they clarify the code:**

```
REFERENCE COMMENT TYPES:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  SPECIFICATION REFERENCES                                               │
│  // Implements RFC 7231, Section 6.5.1 (400 Bad Request)               │
│  // See: https://tools.ietf.org/html/rfc7231#section-6.5.1             │
│                                                                         │
│  ALGORITHM REFERENCES                                                   │
│  // Uses Dijkstra's algorithm for shortest path                        │
│  // Reference: Introduction to Algorithms, Cormen et al., Ch. 24       │
│  // Complexity: O((V + E) log V) with binary heap                      │
│                                                                         │
│  API DOCUMENTATION                                                      │
│  // Stripe API webhook signature verification                          │
│  // See: https://stripe.com/docs/webhooks/signatures                   │
│                                                                         │
│  INTERNAL DOCUMENTATION                                                 │
│  // Business rules defined in SPEC-123                                 │
│  // See: https://confluence.company.com/display/SPEC/Pricing+Rules     │
│                                                                         │
│  STACK OVERFLOW / COMMUNITY                                            │
│  // Workaround for browser quirk in Safari                             │
│  // See: https://stackoverflow.com/a/12345678                          │
│                                                                         │
│  DESIGN DECISIONS                                                       │
│  // Architecture Decision Record: ADR-042                              │
│  // See: docs/adr/042-event-sourcing.md                               │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### B. Reference Examples

```go
// ParseJWT validates and parses a JWT token according to RFC 7519.
//
// Security considerations:
// - Validates signature using RS256 (RSA + SHA-256)
// - Rejects tokens with "none" algorithm (CVE-2015-2951)
// - Validates exp, nbf, and iat claims
//
// References:
// - JWT Spec: https://tools.ietf.org/html/rfc7519
// - JWS Spec: https://tools.ietf.org/html/rfc7515
// - Security Best Practices: https://auth0.com/blog/critical-vulnerabilities-in-json-web-token-libraries/
func ParseJWT(token string, publicKey *rsa.PublicKey) (*Claims, error) {
```

```python
def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Calculate the Levenshtein (edit) distance between two strings.

    Uses Wagner-Fischer algorithm with O(min(m,n)) space optimization.

    Algorithm Reference:
        Wagner, R. A., & Fischer, M. J. (1974).
        "The String-to-String Correction Problem"
        Journal of the ACM, 21(1), 168-173.
        https://doi.org/10.1145/321796.321811

    Complexity:
        Time: O(m * n) where m, n are string lengths
        Space: O(min(m, n)) using rolling array optimization

    See Also:
        - Damerau-Levenshtein for transposition support
        - Jaro-Winkler for similarity scoring
    """
```

---

## 8. Inline Comments (MANDATORY)

### A. When to Use Inline Comments

**Use inline comments sparingly for non-obvious code:**

```python
# GOOD: Explains WHY, not WHAT
timeout = 30  # Matches upstream service SLA (see SLA-DOC-123)

# GOOD: Explains business rule
if age >= 65:  # Senior discount eligibility per policy POL-456
    apply_discount(0.15)

# GOOD: Explains non-obvious technical choice
buffer_size = 8192  # Optimal for SSD page size, benchmarked in PERF-789

# GOOD: Warns about subtle behavior
result = data.copy()  # Must copy - original is mutated by process()

# BAD: States the obvious
counter = 0  # Initialize counter to zero
i += 1  # Increment i

# BAD: Repeats the code
if user.is_active:  # Check if user is active
    return True  # Return true
```

### B. Complex Logic Comments

**For complex algorithms, add a block comment explaining the approach:**

```java
/*
 * Binary search with fuzzy matching for autocomplete suggestions.
 *
 * Algorithm:
 * 1. Binary search to find first prefix match
 * 2. Expand in both directions to collect all matches
 * 3. Score matches by:
 *    - Exact match: 100 points
 *    - Prefix match: 80 points
 *    - Contains match: 50 points
 * 4. Return top N by score, then alphabetically
 *
 * Why not linear search?
 * With 100K+ terms, linear search takes ~50ms.
 * This approach: ~0.5ms (measured in PERF-234)
 *
 * Edge cases handled:
 * - Empty query: returns popular terms
 * - No matches: returns empty list
 * - Unicode: normalized to NFC before comparison
 */
public List<Suggestion> findSuggestions(String query, int limit) {
```

---

## 9. Class/Module Comments (MANDATORY)

### A. Class Documentation Standard

**Every public class/module MUST have a documentation header:**

```typescript
/**
 * Manages user authentication and session lifecycle.
 *
 * This service handles:
 * - User login/logout with multiple providers (OAuth, SAML, local)
 * - Session creation, validation, and refresh
 * - Token management (access tokens, refresh tokens)
 * - Rate limiting for authentication attempts
 *
 * Thread Safety:
 *   All public methods are thread-safe. Internal state is protected
 *   by read-write locks optimized for read-heavy workloads.
 *
 * Configuration:
 *   Required environment variables:
 *   - AUTH_JWT_SECRET: Secret for JWT signing
 *   - AUTH_SESSION_TTL: Session timeout in seconds (default: 3600)
 *   - AUTH_MAX_ATTEMPTS: Max login attempts before lockout (default: 5)
 *
 * Dependencies:
 *   - UserRepository: For user credential lookup
 *   - TokenService: For JWT generation/validation
 *   - CacheService: For session storage (Redis recommended)
 *
 * Example:
 *   ```typescript
 *   const auth = new AuthenticationService(userRepo, tokenService, cache);
 *
 *   // Login
 *   const session = await auth.login(email, password);
 *
 *   // Validate
 *   const user = await auth.validateSession(session.token);
 *
 *   // Logout
 *   await auth.logout(session.token);
 *   ```
 *
 * @see {@link UserRepository} for user data access
 * @see {@link TokenService} for token operations
 * @see docs/architecture/authentication.md for design decisions
 *
 * @since 1.0.0
 * @author Security Team
 */
export class AuthenticationService {
```

### B. Module/Package Documentation

```python
"""
User management module for the application.

This module provides comprehensive user lifecycle management including
registration, authentication, profile management, and access control.

Modules:
    authentication: Login, logout, password reset, MFA
    registration: User signup, email verification
    profile: User profile CRUD operations
    permissions: Role-based access control (RBAC)

Quick Start:
    >>> from user import UserService
    >>> service = UserService(db_connection)
    >>> user = service.create_user("john@example.com", "John Doe")
    >>> session = service.authenticate("john@example.com", "password")

Configuration:
    The module reads configuration from environment variables:
    - USER_DB_URL: Database connection string
    - USER_HASH_ROUNDS: bcrypt hash rounds (default: 12)
    - USER_SESSION_TTL: Session timeout in seconds

Architecture:
    Follows hexagonal architecture with clear separation:
    - Domain: Core business logic (user.domain)
    - Ports: Interfaces (user.ports)
    - Adapters: Implementations (user.adapters)

See Also:
    - Architecture docs: docs/architecture/user-module.md
    - API reference: docs/api/user.md
    - Security guidelines: docs/security/authentication.md

Note:
    This module handles sensitive data. Ensure all deployments
    follow the security checklist in docs/security/checklist.md
"""
```

---

## 10. Comment Maintenance (MANDATORY)

### A. Comment Update Rules

**CRITICAL: Comments MUST be updated whenever code changes.**

```
COMMENT MAINTENANCE PROTOCOL:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  WHEN CODE CHANGES, CHECK AND UPDATE:                                   │
│                                                                         │
│  1. Function/Method Signature Changes                                   │
│     → Update parameter descriptions                                     │
│     → Update return value documentation                                 │
│     → Update exception documentation                                    │
│     → Update examples                                                   │
│                                                                         │
│  2. Behavior Changes                                                    │
│     → Update description of what the code does                         │
│     → Update any edge case documentation                               │
│     → Update performance characteristics if changed                    │
│                                                                         │
│  3. Bug Fixes                                                          │
│     → Add bug fix comment with issue reference                         │
│     → Update any comments that described buggy behavior                │
│                                                                         │
│  4. Refactoring                                                        │
│     → Review all comments in refactored code                           │
│     → Remove comments for deleted code                                 │
│     → Update file/module-level documentation                           │
│                                                                         │
│  5. Dependency Changes                                                 │
│     → Update version references                                        │
│     → Update workaround comments if issue is fixed                     │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### B. Detecting Stale Comments

**Signs of stale comments:**

```python
# STALE: References non-existent parameter
def process(data):  # config parameter removed in v2.0
    """
    Process the data with the given configuration.  # ❌ No config param!

    Args:
        data: The input data
        config: Configuration options  # ❌ This parameter doesn't exist!
    """
    pass

# STALE: Describes old behavior
def get_users():
    """Returns all users from the database."""  # ❌ Now returns paginated!
    return paginate(User.query.all(), page=1, per_page=50)

# STALE: References old code structure
# See UserManager.validate() for validation logic  # ❌ Class was renamed!
user_validator.check(user)

# STALE: Workaround for fixed issue
# HACK: Workaround for bug in library v1.2  # ❌ Using v2.0 now!
result = library.fixed_function()  # Bug was fixed in v1.5
```

### C. Agent Comment Update Checklist

**Before delivering modified code, agents MUST:**

- [ ] Read all comments in files being modified
- [ ] Identify comments affected by the changes
- [ ] Update parameter documentation if signatures changed
- [ ] Update behavior descriptions if logic changed
- [ ] Remove comments for deleted code
- [ ] Add bug fix comments for bug fixes
- [ ] Verify examples still work
- [ ] Run documentation generator to check for warnings
- [ ] Search for references to renamed/deleted items

---

## 11. Anti-Patterns (PROHIBITED)

### A. Comment Anti-Patterns

**NEVER use these patterns:**

```
PROHIBITED COMMENT PATTERNS:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  ❌ NOISE COMMENTS                                                      │
│     // Constructor                                                      │
│     public UserService() { }                                            │
│                                                                         │
│     // Getter                                                          │
│     public String getName() { return name; }                           │
│                                                                         │
│  ❌ REDUNDANT COMMENTS                                                  │
│     i++; // Increment i by 1                                           │
│     return null; // Return null                                        │
│                                                                         │
│  ❌ JOURNAL COMMENTS                                                    │
│     // 2024-01-15: Added validation (John)                             │
│     // 2024-01-20: Fixed bug (Jane)                                    │
│     // Use git history instead!                                        │
│                                                                         │
│  ❌ COMMENTED-OUT CODE                                                  │
│     // oldImplementation();                                            │
│     // if (legacy) { doOldThing(); }                                   │
│     // Delete it! Git has history.                                     │
│                                                                         │
│  ❌ CLOSING BRACE COMMENTS                                             │
│     } // end if                                                        │
│     } // end for                                                       │
│     } // end try                                                       │
│     // Refactor to smaller functions instead!                          │
│                                                                         │
│  ❌ POSITION MARKERS                                                   │
│     ///////////////// SECTION 1 /////////////////                      │
│     // ============ HELPERS ============ //                            │
│     // Refactor into separate files/classes!                           │
│                                                                         │
│  ❌ SCARY WARNINGS WITHOUT CONTEXT                                     │
│     // DON'T TOUCH THIS!                                               │
│     // HERE BE DRAGONS                                                 │
│     // Explain WHY it's dangerous instead!                             │
│                                                                         │
│  ❌ ATTRIBUTION IN CODE                                                │
│     // Written by John Smith                                           │
│     // Copyright 2024 Company Inc                                      │
│     // Use file headers or LICENSE files instead!                      │
│                                                                         │
│  ❌ MANDATED COMMENTS                                                  │
│     // Every function must have a comment (even if trivial)            │
│     /** Gets the ID. @return the ID */                                │
│     public long getId() { return id; }                                 │
│     // Only comment when it adds value!                                │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### B. How to Fix Anti-Patterns

| Anti-Pattern | Solution |
|--------------|----------|
| Noise comments | Delete them; code should be self-explanatory |
| Journal comments | Use git commit messages and blame |
| Commented-out code | Delete it; git preserves history |
| Closing braces | Extract to smaller functions |
| Position markers | Split into multiple files/classes |
| Scary warnings | Add context: WHY and WHAT happens if touched |
| Redundant docs | Only document non-obvious behavior |

---

## 12. Documentation Generation (MANDATORY)

### A. Documentation Tools by Language

| Language | Tool | Command | Output |
|----------|------|---------|--------|
| Python | pydoc / Sphinx | `sphinx-build -b html docs/ _build/` | HTML |
| JavaScript | JSDoc | `jsdoc -c jsdoc.json` | HTML |
| TypeScript | TypeDoc | `typedoc --out docs src/` | HTML |
| Java | Javadoc | `javadoc -d docs -sourcepath src` | HTML |
| Go | godoc | `go doc -all ./...` | Text/HTML |
| Rust | rustdoc | `cargo doc --no-deps` | HTML |
| C/C++ | Doxygen | `doxygen Doxyfile` | HTML/PDF |
| Ruby | YARD | `yard doc` | HTML |
| PHP | phpDocumentor | `phpdoc -d src -t docs` | HTML |
| Kotlin | Dokka | `./gradlew dokkaHtml` | HTML |
| Swift | DocC | `swift package generate-documentation` | HTML |

### B. Documentation Generation Requirements

**MANDATORY: Documentation MUST be generated and verified.**

```bash
# Example CI/CD step for documentation verification
documentation:
  stage: verify
  script:
    # Generate documentation
    - npm run docs:generate

    # Check for warnings (fail on warnings)
    - npm run docs:generate 2>&1 | tee docs.log
    - "! grep -i 'warning' docs.log"

    # Verify all public APIs are documented
    - npm run docs:coverage -- --threshold 100

    # Test code examples in documentation
    - npm run docs:test-examples
  artifacts:
    paths:
      - docs/_build/
```

### C. Documentation Quality Checks

**Verify documentation quality:**

```bash
# Check documentation coverage
# Python
interrogate -vv --fail-under 100 src/

# TypeScript
typedoc --validation.notExported

# Java
javadoc -Xdoclint:all

# Go (check for missing comments)
golint ./..

# Rust
cargo doc --document-private-items 2>&1 | grep -i "warning"
```

---

## 13. Deployment Checklist

### Documentation Verification (MANDATORY)

**Before delivering ANY code, verify:**

#### Comment Quality
- [ ] All public APIs have doc comments
- [ ] All parameters documented with types and constraints
- [ ] All return values documented
- [ ] All exceptions/errors documented
- [ ] Examples provided for non-trivial APIs
- [ ] No comments contradict the code
- [ ] No stale comments referencing old code

#### Comment Maintenance
- [ ] Comments updated for all code changes
- [ ] Bug fix comments include issue references
- [ ] TODOs include issue references
- [ ] Removed comments for deleted code
- [ ] No commented-out code without explanation

#### Documentation Generation
- [ ] Documentation generates without errors
- [ ] Documentation generates without warnings
- [ ] All links in documentation resolve
- [ ] Code examples in docs compile/run
- [ ] Documentation coverage meets threshold

#### Agent Workflow Completed
- [ ] Agent reviewed all comments in modified files
- [ ] Agent updated affected comments
- [ ] Agent verified documentation generates correctly
- [ ] Agent tested examples in documentation
- [ ] Agent removed stale comments

---

## 14. Why These Guidelines Work

**Code as Primary Documentation**:
- Self-documenting code reduces maintenance burden
- Comments explain intent, code explains implementation
- Reduces risk of comments diverging from code

**Machine-Readable Documentation**:
- Automatic API doc generation ensures consistency
- Docs stay in sync with code (same repo, same PR)
- IDE integration provides inline documentation

**Minimal Comments**:
- Less to maintain = less to go stale
- Forces better naming and structure
- Comments that exist are valuable, not noise

**Bug Fix Traceability**:
- Issue references enable bisecting and understanding
- Future developers can find related discussions
- Prevents re-introduction of fixed bugs

**TODO Tracking**:
- Makes technical debt visible
- Enables prioritization and planning
- Prevents forgotten incomplete work

---

## 15. Quick Reference

### Comment Decision Tree

```
Should I write a comment?

                    ┌─────────────────────┐
                    │ Is it a public API? │
                    └─────────┬───────────┘
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
             YES                              NO
              │                               │
              ▼                               ▼
    ┌─────────────────┐           ┌─────────────────────┐
    │ Write doc       │           │ Is the code unclear │
    │ comment with    │           │ after refactoring?  │
    │ full API docs   │           └─────────┬───────────┘
    └─────────────────┘                     │
                              ┌─────────────┴─────────────┐
                              ▼                           ▼
                             YES                          NO
                              │                           │
                              ▼                           ▼
                    ┌─────────────────┐       ┌─────────────────┐
                    │ Comment WHY,    │       │ No comment      │
                    │ not WHAT        │       │ needed          │
                    └─────────────────┘       └─────────────────┘
```

### Comment Templates

```
// === PUBLIC API ===
/**
 * Brief description.
 *
 * Detailed description if needed.
 *
 * @param name - Description
 * @returns Description
 * @throws Error - When X happens
 * @example
 * code example here
 */

// === BUG FIX ===
// FIX(#ISSUE-ID): Brief description
// Problem: What was happening
// Solution: How this fixes it

// === TODO ===
// TODO(#ISSUE-ID): What needs to be done
// Context: Why it's not done yet

// === REFERENCE ===
// Implements SPEC-123 / RFC-7231 / Algorithm Name
// See: https://link.to/documentation

// === WARNING ===
// WARNING: Description of danger
// Why: Explanation of consequences
// Safe usage: How to use safely
```

---

**End of Code Comments & Documentation Guidelines**
